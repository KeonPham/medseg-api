"""Evaluation metrics for segmentation model performance."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice similarity coefficient.

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.
        smooth: Smoothing factor.

    Returns:
        Dice coefficient value in [0, 1].
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (Jaccard index).

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.
        smooth: Smoothing factor.

    Returns:
        IoU score in [0, 1].
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute pixel-wise accuracy.

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.

    Returns:
        Pixel accuracy in [0, 1].
    """
    return np.mean(pred.flatten() == target.flatten())


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute all segmentation metrics.

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.

    Returns:
        Dictionary with dice, iou, and pixel_accuracy.
    """
    return {
        "dice": dice_coefficient(pred, target),
        "iou": iou_score(pred, target),
        "pixel_accuracy": pixel_accuracy(pred, target),
    }
