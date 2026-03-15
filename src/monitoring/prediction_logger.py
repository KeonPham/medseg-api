"""Prediction logging to a relational database."""

import logging
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT,
    model_version TEXT,
    inference_time_ms REAL,
    image_hash TEXT,
    confidence_score REAL,
    lung_coverage_pct REAL,
    symmetry_ratio REAL
)
"""

_INSERT_SQL = """
INSERT INTO predictions
    (request_id, timestamp, model_name, model_version,
     inference_time_ms, image_hash, confidence_score,
     lung_coverage_pct, symmetry_ratio)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_RECENT_SQL = """
SELECT id, request_id, timestamp, model_name, model_version,
       inference_time_ms, image_hash, confidence_score,
       lung_coverage_pct, symmetry_ratio
FROM predictions
WHERE timestamp >= ?
ORDER BY timestamp DESC
"""

_SUMMARY_SQL = """
SELECT
    COUNT(*) as total_predictions,
    AVG(inference_time_ms) as avg_inference_ms,
    MIN(inference_time_ms) as min_inference_ms,
    MAX(inference_time_ms) as max_inference_ms,
    AVG(confidence_score) as avg_confidence,
    AVG(lung_coverage_pct) as avg_lung_coverage
FROM predictions
WHERE timestamp >= ?
"""

_MODEL_DISTRIBUTION_SQL = """
SELECT model_name, COUNT(*) as count
FROM predictions
WHERE timestamp >= ?
GROUP BY model_name
ORDER BY count DESC
"""

_COLUMN_NAMES = [
    "id",
    "request_id",
    "timestamp",
    "model_name",
    "model_version",
    "inference_time_ms",
    "image_hash",
    "confidence_score",
    "lung_coverage_pct",
    "symmetry_ratio",
]


class PredictionLogger:
    """Logs prediction metadata to SQLite for monitoring.

    Thread-safe: each method opens its own connection.
    """

    def __init__(self, db_url: str = "sqlite:///./predictions.db") -> None:
        """Initialize the logger and create the table if needed.

        Args:
            db_url: Database URL. Supports sqlite:/// prefix.
        """
        # Strip the sqlite:/// prefix to get the file path
        if db_url.startswith("sqlite:///"):
            self._db_path = db_url[len("sqlite:///") :]
        else:
            self._db_path = db_url

        self._ensure_table()
        logger.info("PredictionLogger initialized: %s", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    def log_prediction(
        self,
        request_id: str,
        model_name: str,
        model_version: str,
        inference_time_ms: float,
        image_hash: str,
        confidence_score: float,
        lung_coverage_pct: float,
        symmetry_ratio: float,
    ) -> None:
        """Insert a prediction record.

        Args:
            request_id: Unique request identifier.
            model_name: Model variant used.
            model_version: Model version string.
            inference_time_ms: Inference latency in milliseconds.
            image_hash: SHA256 hash of the input image.
            confidence_score: Mean confidence on positive pixels.
            lung_coverage_pct: Percentage of image covered by lung mask.
            symmetry_ratio: Left/right lung symmetry ratio.
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    request_id,
                    now,
                    model_name,
                    model_version,
                    inference_time_ms,
                    image_hash,
                    confidence_score,
                    lung_coverage_pct,
                    symmetry_ratio,
                ),
            )

    def get_recent_predictions(self, hours: int = 24) -> list[dict]:
        """Fetch predictions from the last N hours.

        Args:
            hours: Number of hours to look back.

        Returns:
            List of prediction dicts.
        """
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_SQL, (since,)).fetchall()
        return [dict(zip(_COLUMN_NAMES, row)) for row in rows]

    def get_metrics_summary(self, hours: int = 24) -> dict:
        """Get aggregated metrics for the last N hours.

        Args:
            hours: Number of hours to look back.

        Returns:
            Dict with total_predictions, avg_inference_ms, model_distribution, etc.
        """
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            summary_row = conn.execute(_SUMMARY_SQL, (since,)).fetchone()
            model_rows = conn.execute(_MODEL_DISTRIBUTION_SQL, (since,)).fetchall()

        summary_keys = [
            "total_predictions",
            "avg_inference_ms",
            "min_inference_ms",
            "max_inference_ms",
            "avg_confidence",
            "avg_lung_coverage",
        ]
        summary = dict(zip(summary_keys, summary_row)) if summary_row else {}

        # Replace None with 0 for empty periods
        for k in summary_keys:
            if summary.get(k) is None:
                summary[k] = 0

        summary["model_distribution"] = {row[0]: row[1] for row in model_rows}
        return summary
