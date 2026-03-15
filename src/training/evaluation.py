"""Model evaluation and comparison utilities."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.metrics import (
    MetricsTracker,
    dice_coefficient,
    hausdorff_distance_95,
    iou_score,
    sensitivity,
    specificity,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate a model on a test set and return all metrics.

    Args:
        model: Trained segmentation model in eval mode.
        test_loader: DataLoader for the test set.
        device: Device to run inference on.

    Returns:
        Dict with dice, iou, hd95, sensitivity, specificity.
    """
    model.eval()
    model.to(device)
    tracker = MetricsTracker()

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"]

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            targets = masks.numpy()

            for i in range(preds.shape[0]):
                p = preds[i].squeeze().astype(np.uint8)
                t = targets[i].squeeze().astype(np.uint8)
                tracker.update(
                    {
                        "dice": dice_coefficient(p, t),
                        "iou": iou_score(p, t),
                        "hd95": hausdorff_distance_95(p, t),
                        "sensitivity": sensitivity(p, t),
                        "specificity": specificity(p, t),
                    }
                )

    metrics = tracker.compute()
    logger.info(
        "Evaluation: dice=%.4f iou=%.4f hd95=%.2f sens=%.4f spec=%.4f",
        metrics["dice"],
        metrics["iou"],
        metrics["hd95"],
        metrics["sensitivity"],
        metrics["specificity"],
    )
    return metrics


def compare_models(
    old_metrics: dict[str, float],
    new_metrics: dict[str, float],
    dice_threshold: float = 0.005,
) -> dict:
    """Compare old and new model metrics and recommend promotion.

    Args:
        old_metrics: Metrics from the current production model.
        new_metrics: Metrics from the newly trained model.
        dice_threshold: Minimum Dice improvement to recommend promotion.

    Returns:
        Dict with 'deltas', 'improved' (bool), and 'recommendation' (str).
    """
    deltas = {}
    for key in new_metrics:
        old_val = old_metrics.get(key, 0.0)
        new_val = new_metrics[key]
        # For hd95 lower is better, for everything else higher is better
        if key == "hd95":
            deltas[key] = old_val - new_val
        else:
            deltas[key] = new_val - old_val

    dice_delta = new_metrics.get("dice", 0.0) - old_metrics.get("dice", 0.0)
    improved = dice_delta >= dice_threshold

    if improved:
        recommendation = (
            f"PROMOTE: new model Dice {new_metrics['dice']:.4f} >= "
            f"old {old_metrics.get('dice', 0.0):.4f} + {dice_threshold} threshold"
        )
    else:
        recommendation = (
            f"KEEP: new model Dice {new_metrics.get('dice', 0.0):.4f} "
            f"did not improve by >= {dice_threshold} over "
            f"old {old_metrics.get('dice', 0.0):.4f}"
        )

    logger.info(recommendation)
    return {
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "deltas": deltas,
        "improved": improved,
        "recommendation": recommendation,
    }
