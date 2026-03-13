"""Data and model drift detection for production monitoring."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DriftDetector:
    """Monitors input data and prediction distributions for drift.

    Compares recent prediction statistics against a baseline
    to detect potential data or concept drift.
    """

    def __init__(self, window_size: int = 100) -> None:
        """Initialize the drift detector.

        Args:
            window_size: Number of recent predictions to keep in the sliding window.
        """
        self.window_size = window_size
        self.baseline_stats: dict[str, float] | None = None
        self.recent_scores: list[float] = []
        logger.info("DriftDetector initialized with window_size=%d", window_size)

    def set_baseline(self, scores: list[float]) -> None:
        """Set baseline statistics from a reference dataset.

        Args:
            scores: List of Dice scores from the reference evaluation.
        """
        arr = np.array(scores)
        self.baseline_stats = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        logger.info("Drift baseline set: mean=%.4f, std=%.4f", arr.mean(), arr.std())

    def add_score(self, score: float) -> None:
        """Add a prediction score to the sliding window.

        Args:
            score: Dice score from a recent prediction.
        """
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores = self.recent_scores[-self.window_size :]

    def check_drift(self, threshold: float = 2.0) -> dict:
        """Check if recent predictions show drift from baseline.

        Args:
            threshold: Number of standard deviations for drift detection.

        Returns:
            Dictionary with drift detection results.
        """
        if not self.baseline_stats or len(self.recent_scores) < 10:
            return {"drift_detected": False, "reason": "insufficient data"}

        recent_mean = np.mean(self.recent_scores)
        z_score = abs(recent_mean - self.baseline_stats["mean"]) / max(
            self.baseline_stats["std"], 1e-6
        )

        drift_detected = z_score > threshold
        if drift_detected:
            logger.warning("Drift detected: z_score=%.2f, threshold=%.2f", z_score, threshold)

        return {
            "drift_detected": drift_detected,
            "z_score": float(z_score),
            "recent_mean": float(recent_mean),
            "baseline_mean": self.baseline_stats["mean"],
        }
