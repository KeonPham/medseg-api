"""CNN-only lung segmentation model: ResNet-18 encoder + U-Net decoder.

Ported from thesis ablation study (Ablation_study_1_cnn_only.ipynb).
Layer names and structure are identical to the original for weight compatibility.
"""

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class LightCNNEncoder(nn.Module):
    """Lightweight CNN encoder using ResNet-18.

    Extracts multi-scale features at 4 resolution levels using
    a pretrained ResNet-18 backbone. Skip connections from each
    level feed into the decoder.

    Output channels per level: [64, 128, 256, 512]
    Output spatial sizes (for 512 input): [H/4, H/8, H/16, H/32]
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


class SimpleDecoder(nn.Module):
    """U-Net style decoder with skip connections from the CNN encoder.

    Progressively upsamples from the deepest encoder features while
    concatenating skip connections from earlier encoder levels.
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

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Decode with skip connections.

        Args:
            features: Tuple (c1, c2, c3, c4) from the CNN encoder.

        Returns:
            Segmentation logits (B, 1, H, W).
        """
        c1, c2, c3, c4 = features

        x = self.up1(c4)  # [B, 256, 32, 32]
        x = torch.cat([x, c3], dim=1)

        x = self.up2(x)  # [B, 128, 64, 64]
        x = torch.cat([x, c2], dim=1)

        x = self.up3(x)  # [B, 64, 128, 128]
        x = torch.cat([x, c1], dim=1)

        x = self.up4(x)  # [B, 32, 256, 256]

        x = self.final_up(x)  # [B, 16, 512, 512]
        output = self.final_conv(x)  # [B, 1, 512, 512]
        return output


class CNNLungSegmentation(nn.Module):
    """CNN-only lung segmentation model.

    Architecture:
        - Encoder: ResNet-18 with multi-scale feature extraction
        - Decoder: U-Net style with skip connections
        - Input: 512x512 chest X-ray (3-channel RGB)
        - Output: binary segmentation logits (1-channel)
        - Parameters: ~15M

    Attribute names (cnn_encoder, decoder) match the thesis checkpoint
    for weight loading compatibility.
    """

    def __init__(self, img_size: int = 512, pretrained: bool = True) -> None:
        """Initialize CNN lung segmentation model.

        Args:
            img_size: Input image size (unused, kept for API consistency).
            pretrained: Use ImageNet pretrained weights for encoder.
        """
        super().__init__()
        self.cnn_encoder = LightCNNEncoder(pretrained=pretrained)
        self.decoder = SimpleDecoder()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CNNLungSegmentation initialized: %s params (~%.1fM)",
            f"{total_params:,}",
            total_params / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: CNN encoding then decoding with skip connections.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Segmentation logits (B, 1, H, W).
        """
        cnn_features = self.cnn_encoder(x)
        output = self.decoder(cnn_features)
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
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
