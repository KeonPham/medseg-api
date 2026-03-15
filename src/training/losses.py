"""Loss functions for lung segmentation training."""

import logging

import torch
import torch.nn as nn
from torch.nn import functional as nnf

logger = logging.getLogger(__name__)


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
        bce = nnf.binary_cross_entropy_with_logits(pred, target, reduction="none")
        probs = torch.sigmoid(pred)
        p_t = probs * target + (1 - probs) * (1 - target)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


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


class BoundaryLoss(nn.Module):
    """Boundary-aware loss that penalizes errors at mask edges."""

    def __init__(self, weight: float = 0.3) -> None:
        """Initialize Boundary loss.

        Args:
            weight: Scaling weight for the boundary loss component.
        """
        super().__init__()
        self.weight = weight
        # Laplacian kernel for edge detection
        kernel = (
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.register_buffer("kernel", kernel)

    def _extract_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary pixels from a binary mask using Laplacian filter."""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        edges = nnf.conv2d(mask, self.kernel, padding=1)
        return (edges.abs() > 0).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Boundary loss.

        Args:
            pred: Predicted probabilities (after sigmoid).
            target: Ground truth binary mask.

        Returns:
            Weighted boundary loss value.
        """
        boundary = self._extract_boundary(target)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        diff = (pred - target.unsqueeze(1) if target.dim() == 3 else pred - target) ** 2
        boundary_loss = (diff * boundary).sum() / (boundary.sum() + 1e-6)
        return self.weight * boundary_loss


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
        self.boundary = BoundaryLoss(weight=boundary_weight)
        logger.info(
            "CombinedLoss: focal(a=%.2f, g=%.1f) + dice + boundary(w=%.1f)",
            focal_alpha,
            focal_gamma,
            boundary_weight,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Predicted logits (before sigmoid).
            target: Ground truth binary mask.

        Returns:
            Scalar combined loss value.
        """
        focal_loss = self.focal(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = self.dice(pred_sigmoid, target)
        boundary_loss = self.boundary(pred_sigmoid, target)
        return focal_loss + dice_loss + boundary_loss
