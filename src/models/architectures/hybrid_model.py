"""Hybrid CNN-ViT lung segmentation model with cross-attention fusion.

Ported from thesis code (lung_seg/models/hybrid_model.py).
Layer names and structure are identical to the original for weight compatibility.

Architecture:
    1. CNN encoder (ResNet-18) for local features
    2. ViT encoder (DeiT-Tiny) for global features
    3. Cross-attention fusion between CNN and ViT
    4. U-Net style decoder with skip connections from CNN
"""

import logging

import timm
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torchvision import models

logger = logging.getLogger(__name__)


class LightCNNEncoder(nn.Module):
    """Lightweight CNN encoder using ResNet-18.

    Extracts multi-scale features at 4 resolution levels.
    The first 3 levels provide skip connections to the decoder,
    while the deepest level feeds into cross-attention fusion.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize encoder with ResNet-18 backbone.

        Args:
            pretrained: Use ImageNet pretrained weights.
        """
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        self.conv1 = resnet.conv1  # 64 channels
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 ch,  H/4
        self.layer2 = resnet.layer2  # 128 ch, H/8
        self.layer3 = resnet.layer3  # 256 ch, H/16
        self.layer4 = resnet.layer4  # 512 ch, H/32

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Tuple of feature maps (c1, c2, c3, c4) at decreasing resolutions.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)  # [B, 64,  H/4,  W/4]
        c2 = self.layer2(c1)  # [B, 128, H/8,  W/8]
        c3 = self.layer3(c2)  # [B, 256, H/16, W/16]
        c4 = self.layer4(c3)  # [B, 512, H/32, W/32]

        return c1, c2, c3, c4


class LightViTEncoder(nn.Module):
    """Lightweight ViT encoder using DeiT-Tiny.

    Extracts global context features and projects them to match
    the CNN encoder's deepest spatial resolution for cross-attention.
    """

    def __init__(self, img_size: int = 512, pretrained: bool = True) -> None:
        """Initialize ViT encoder.

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
        self.embed_dim = 192
        self.feature_proj = nn.Conv2d(192, 512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project ViT features to spatial format.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Projected features (B, 512, 16, 16).
        """
        features = self.vit.forward_features(x)  # [B, 1+N, 192]
        patch_features = features[:, 1:]  # remove CLS token

        b = patch_features.shape[0]
        vit_spatial = patch_features.transpose(1, 2).reshape(b, 192, 32, 32)
        vit_resized = nnf.interpolate(
            vit_spatial, size=(16, 16), mode="bilinear", align_corners=False
        )
        vit_projected = self.feature_proj(vit_resized)  # [B, 512, 16, 16]

        return vit_projected


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between CNN and ViT feature maps.

    CNN features attend to ViT features (gaining global context)
    and ViT features attend to CNN features (gaining local detail).
    Results are concatenated and fused through a convolutional layer.
    """

    def __init__(self, channels: int = 512, num_heads: int = 8) -> None:
        """Initialize cross-attention fusion.

        Args:
            channels: Feature channel dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.cnn_to_vit = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.vit_to_cnn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """Fuse CNN and ViT features via bidirectional cross-attention.

        Args:
            cnn_feat: CNN features (B, C, H, W).
            vit_feat: ViT features (B, C, H, W).

        Returns:
            Fused features (B, C, H, W).
        """
        b, c, h, w = cnn_feat.shape

        cnn_flat = cnn_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        vit_flat = vit_feat.flatten(2).transpose(1, 2)

        cnn_enhanced, _ = self.cnn_to_vit(cnn_flat, vit_flat, vit_flat)
        cnn_enhanced = self.norm1(cnn_enhanced + cnn_flat)

        vit_enhanced, _ = self.vit_to_cnn(vit_flat, cnn_flat, cnn_flat)
        vit_enhanced = self.norm2(vit_enhanced + vit_flat)

        cnn_enhanced = cnn_enhanced.transpose(1, 2).reshape(b, c, h, w)
        vit_enhanced = vit_enhanced.transpose(1, 2).reshape(b, c, h, w)

        combined = torch.cat([cnn_enhanced, vit_enhanced], dim=1)
        fused = self.fusion_conv(combined)
        return fused


class SimpleDecoder(nn.Module):
    """U-Net style decoder with skip connections from the CNN encoder.

    Takes fused features from cross-attention and skip connections
    from the first 3 CNN encoder levels (c1, c2, c3).
    """

    def __init__(self, encoder_channels: list[int] | None = None) -> None:
        """Initialize decoder blocks.

        Args:
            encoder_channels: Channel counts from encoder levels.
                Defaults to [64, 128, 256, 512].
        """
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]

        self.up1 = self._make_decoder_block(512, 256)  # 16 -> 32
        self.up2 = self._make_decoder_block(256 + 256, 128)  # + skip c3
        self.up3 = self._make_decoder_block(128 + 128, 64)  # + skip c2
        self.up4 = self._make_decoder_block(64 + 64, 32)  # + skip c1

        self.final_up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.final_conv = nn.Conv2d(16, 1, 1)

    @staticmethod
    def _make_decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a decoder block with transposed conv + conv refinement."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        skip_features: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Decode fused features with CNN skip connections.

        Args:
            fused_features: Cross-attention output (B, 512, H/32, W/32).
            skip_features: Tuple (c1, c2, c3) from CNN encoder.

        Returns:
            Segmentation logits (B, 1, H, W).
        """
        c1, c2, c3 = skip_features

        x = self.up1(fused_features)
        x = torch.cat([x, c3], dim=1)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)

        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)

        x = self.up4(x)

        x = self.final_up(x)
        output = self.final_conv(x)
        return output


class HybridLungSegmentation(nn.Module):
    """Parallel CNN-ViT hybrid with cross-attention for lung segmentation.

    Architecture:
        - CNN encoder (ResNet-18): local features + skip connections
        - ViT encoder (DeiT-Tiny): global features
        - Cross-attention fusion at the deepest level
        - Decoder with skip connections from CNN
        - Input: 512x512 chest X-ray (3-channel RGB)
        - Output: binary segmentation logits (1-channel)
        - Parameters: ~4.2M
        - Best Dice: 96.65% on Montgomery dataset

    Attribute names (cnn_encoder, vit_encoder, cross_attention, decoder)
    match the thesis checkpoint for weight loading compatibility.
    """

    def __init__(self, img_size: int = 512, pretrained: bool = True) -> None:
        """Initialize hybrid lung segmentation model.

        Args:
            img_size: Input image size.
            pretrained: Use pretrained weights for encoders.
        """
        super().__init__()
        self.cnn_encoder = LightCNNEncoder(pretrained=pretrained)
        self.vit_encoder = LightViTEncoder(img_size=img_size, pretrained=pretrained)
        self.cross_attention = CrossAttentionFusion(channels=512, num_heads=8)
        self.decoder = SimpleDecoder()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "HybridLungSegmentation initialized: %s params (~%.1fM)",
            f"{total_params:,}",
            total_params / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: dual encoding, cross-attention fusion, decoding.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Segmentation logits (B, 1, H, W).
        """
        cnn_features = self.cnn_encoder(x)  # (c1, c2, c3, c4)
        vit_features = self.vit_encoder(x)  # [B, 512, 16, 16]

        fused = self.cross_attention(cnn_features[-1], vit_features)
        output = self.decoder(fused, cnn_features[:-1])  # skips = c1, c2, c3
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
            "cnn_encoder": sum(p.numel() for p in self.cnn_encoder.parameters()),
            "vit_encoder": sum(p.numel() for p in self.vit_encoder.parameters()),
            "cross_attention": sum(p.numel() for p in self.cross_attention.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
