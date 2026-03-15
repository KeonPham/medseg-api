"""Evaluation metrics for segmentation model performance."""

import logging
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt

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
    pred_flat = pred.flatten().astype(float)
    target_flat = target.flatten().astype(float)
    intersection = np.sum(pred_flat * target_flat)
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (Jaccard index).

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.
        smooth: Smoothing factor.

    Returns:
        IoU score in [0, 1].
    """
    pred_flat = pred.flatten().astype(float)
    target_flat = target.flatten().astype(float)
    intersection = np.sum(pred_flat * target_flat)
    union = pred_flat.sum() + target_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance between two binary masks.

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.

    Returns:
        HD95 value. Returns 0.0 if both masks are empty or identical.
    """
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    if not pred_bool.any() and not target_bool.any():
        return 0.0
    if not pred_bool.any() or not target_bool.any():
        return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))

    # Distance from pred boundary to nearest target boundary
    pred_dist = distance_transform_edt(~target_bool)
    target_dist = distance_transform_edt(~pred_bool)

    # Get boundary pixels
    pred_boundary = pred_bool ^ _erode(pred_bool)
    target_boundary = target_bool ^ _erode(target_bool)

    if not pred_boundary.any() or not target_boundary.any():
        return 0.0

    d_pred_to_target = pred_dist[pred_boundary]
    d_target_to_pred = target_dist[target_boundary]

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_distances, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Erode a binary mask by 1 pixel using distance transform."""
    dist = distance_transform_edt(mask)
    return dist > 1


def sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute sensitivity (true positive rate / recall).

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.

    Returns:
        Sensitivity in [0, 1]. Returns 0.0 if no positive pixels in target.
    """
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    tp = np.sum(pred_bool & target_bool)
    fn = np.sum(~pred_bool & target_bool)
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def specificity(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute specificity (true negative rate).

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.

    Returns:
        Specificity in [0, 1]. Returns 0.0 if no negative pixels in target.
    """
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    tn = np.sum(~pred_bool & ~target_bool)
    fp = np.sum(pred_bool & ~target_bool)
    if tn + fp == 0:
        return 0.0
    return float(tn / (tn + fp))


class MetricsTracker:
    """Accumulates and averages metrics over an epoch."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        """Add a batch of metric values.

        Args:
            metrics: Dictionary of metric name -> value.
        """
        for k, v in metrics.items():
            self._values[k].append(v)

    def compute(self) -> dict[str, float]:
        """Compute the mean of all accumulated metrics.

        Returns:
            Dictionary of metric name -> mean value.
        """
        return {k: float(np.mean(v)) for k, v in self._values.items()}

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._values.clear()
