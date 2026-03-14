"""ViT-only lung segmentation model: DeiT-Tiny encoder + progressive decoder.

Ported from thesis ablation study (Ablation_study_2_ViT_only.ipynb).
Layer names and structure are identical to the original for weight compatibility.
"""

import logging

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ViTDecoder(nn.Module):
    """Progressive upsampling decoder for ViT patch token features.

    Reshapes linear patch tokens to spatial feature maps, then
    upsamples through transposed convolutions to full resolution.
    No skip connections (pure ViT path has no multi-scale features).
    """

    def __init__(self, embed_dim: int = 256, img_size: int = 512, patch_size: int = 16) -> None:
        """Initialize ViT decoder.

        Args:
            embed_dim: Projected embedding dimension from ViT.
            img_size: Input image size.
            patch_size: ViT patch size (determines spatial dimensions).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.spatial_size = img_size // patch_size  # 32 for 512/16

        self.up1 = self._make_up_block(embed_dim, 128)  # 32 -> 64
        self.up2 = self._make_up_block(128, 64)  # 64 -> 128
        self.up3 = self._make_up_block(64, 32)  # 128 -> 256
        self.up4 = self._make_up_block(32, 16)  # 256 -> 512

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _make_up_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Create upsampling block with transpose conv + conv refinement."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode patch features to segmentation mask.

        Args:
            x: Patch features (B, num_patches, embed_dim).

        Returns:
            Segmentation logits (B, 1, img_size, img_size).
        """
        b = x.shape[0]

        # Reshape from sequence to spatial: [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(b, self.embed_dim, self.spatial_size, self.spatial_size)

        x = self.up1(x)  # [B, 128, 64, 64]
        x = self.up2(x)  # [B, 64, 128, 128]
        x = self.up3(x)  # [B, 32, 256, 256]
        x = self.up4(x)  # [B, 16, 512, 512]

        output = self.final_conv(x)  # [B, 1, 512, 512]
        return output


class ViTLungSegmentation(nn.Module):
    """Vision Transformer lung segmentation model.

    Architecture:
        - Encoder: DeiT-Tiny (pretrained, adapted to 512x512 input)
        - Feature projection: Linear 192 -> 256
        - Decoder: Progressive upsampling from patch tokens
        - Input: 512x512 chest X-ray (3-channel RGB)
        - Output: binary segmentation logits (1-channel)
        - Parameters: ~8M

    Attribute names (vit, feature_proj, decoder) match the thesis
    checkpoint for weight loading compatibility.
    """

    def __init__(self, img_size: int = 512, pretrained: bool = True) -> None:
        """Initialize ViT lung segmentation model.

        Args:
            img_size: Input image size.
            pretrained: Use pretrained DeiT-Tiny weights.
        """
        super().__init__()

        self.vit = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
        )

        self.embed_dim = 192  # DeiT-Tiny embedding dimension
        self.feature_proj = nn.Linear(self.embed_dim, 256)
        self.decoder = ViTDecoder(embed_dim=256, img_size=img_size)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "ViTLungSegmentation initialized: %s params (~%.1fM)",
            f"{total_params:,}",
            total_params / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ViT encoding, projection, decoding.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Segmentation logits (B, 1, H, W).
        """
        # ViT encoding: [B, 1+N, 192] where N = (img_size/patch_size)^2
        vit_features = self.vit.forward_features(x)

        # Remove CLS token, keep only patch tokens
        patch_features = vit_features[:, 1:]  # [B, N, 192]

        # Project to decoder dimension
        projected_features = self.feature_proj(patch_features)  # [B, N, 256]

        # Decode to segmentation mask
        output = self.decoder(projected_features)
        return output

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Run inference and return binary mask.

        Args:
            x: Input tensor (B, 3, H, W).
            threshold: Binarization threshold for sigmoid output.

        Returns:
            Binary mask (B, 1, H, W).
        """
        with torch.no_grad():
            logits = self.forward(x)
            return (torch.sigmoid(logits) > threshold).float()

    def get_param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_info(self) -> dict[str, int]:
        """Get detailed parameter breakdown by component."""
        return {
            "vit_encoder": sum(p.numel() for p in self.vit.parameters()),
            "feature_proj": sum(p.numel() for p in self.feature_proj.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
