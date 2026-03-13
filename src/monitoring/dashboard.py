"""Dashboard data preparation for the Streamlit monitoring app."""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DashboardData:
    """Prepares and aggregates data for the monitoring dashboard.

    Collects prediction logs, model performance metrics,
    and system health data for visualization.
    """

    def __init__(self) -> None:
        """Initialize dashboard data collector."""
        self.prediction_log: list[dict] = []
        logger.info("DashboardData collector initialized")

    def log_prediction(
        self,
        model_id: str,
        filename: str,
        dice_score: float | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Log a prediction event.

        Args:
            model_id: Model variant used.
            filename: Input image filename.
            dice_score: Dice score if ground truth was available.
            latency_ms: Prediction latency in milliseconds.
        """
        self.prediction_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "model_id": model_id,
                "filename": filename,
                "dice_score": dice_score,
                "latency_ms": latency_ms,
            }
        )

    def get_summary(self) -> dict:
        """Get summary statistics for the dashboard.

        Returns:
            Dictionary with prediction counts, average latency, etc.
        """
        total = len(self.prediction_log)
        if total == 0:
            return {"total_predictions": 0}

        latencies = [p["latency_ms"] for p in self.prediction_log]
        return {
            "total_predictions": total,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "models_used": list({p["model_id"] for p in self.prediction_log}),
        }
