"""ViT-only lung segmentation model: DeiT-Tiny encoder + progressive decoder."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ViTLungSeg(nn.Module):
    """Vision Transformer lung segmentation model.

    Architecture:
        - Encoder: DeiT-Tiny (pretrained)
        - Decoder: Progressive upsampling with learned positional embeddings
        - Input: 512x512 chest X-ray (3-channel RGB)
        - Output: binary segmentation mask (1-channel)
        - Parameters: ~8-10M
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        """Initialize the ViT lung segmentation model.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output mask channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        logger.info("ViTLungSeg initialized (stub — architecture not yet ported)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Segmentation logits of shape (B, 1, H, W).
        """
        raise NotImplementedError("Architecture not yet ported from thesis code")

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Run inference and return binary mask.

        Args:
            x: Input tensor of shape (B, C, H, W).
            threshold: Binarization threshold for sigmoid output.

        Returns:
            Binary mask of shape (B, 1, H, W).
        """
        logits = self.forward(x)
        return (torch.sigmoid(logits) > threshold).float()

    def get_param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
