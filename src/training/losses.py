"""Loss functions for lung segmentation training."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        """Initialize Dice loss.

        Args:
            smooth: Smoothing factor to prevent division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred: Predicted probabilities (after sigmoid).
            target: Ground truth binary mask.

        Returns:
            Scalar Dice loss value.
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """Initialize Focal loss.

        Args:
            alpha: Balancing factor.
            gamma: Focusing parameter.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal loss.

        Args:
            pred: Predicted logits (before sigmoid).
            target: Ground truth binary mask.

        Returns:
            Scalar Focal loss value.
        """
        raise NotImplementedError("Focal loss computation not yet implemented")


class CombinedLoss(nn.Module):
    """Combined Focal + Dice + Boundary loss as used in thesis training."""

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_weight: float = 0.3,
    ) -> None:
        """Initialize combined loss.

        Args:
            focal_alpha: Alpha parameter for focal loss.
            focal_gamma: Gamma parameter for focal loss.
            boundary_weight: Weight for the boundary loss component.
        """
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.boundary_weight = boundary_weight
        logger.info(
            "CombinedLoss initialized: focal(a=%.2f, g=%.1f) + dice + boundary(w=%.1f)",
            focal_alpha,
            focal_gamma,
            boundary_weight,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Predicted logits.
            target: Ground truth binary mask.

        Returns:
            Scalar combined loss value.
        """
        raise NotImplementedError("Combined loss computation not yet implemented")
