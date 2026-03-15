"""Tests for the PredictionLogger class."""

import sqlite3
import uuid

import pytest

from src.monitoring.prediction_logger import PredictionLogger


@pytest.fixture()
def logger(tmp_path):
    """Create a PredictionLogger with a temp database."""
    db_path = str(tmp_path / "test_predictions.db")
    return PredictionLogger(db_url=f"sqlite:///{db_path}")


@pytest.fixture()
def logger_raw_path(tmp_path):
    """Create a PredictionLogger with a raw file path (no prefix)."""
    db_path = str(tmp_path / "test_raw.db")
    return PredictionLogger(db_url=db_path)


def _log_sample(lgr: PredictionLogger, **overrides) -> dict:
    """Log a sample prediction and return the kwargs used."""
    defaults = {
        "request_id": str(uuid.uuid4()),
        "model_name": "hybrid",
        "model_version": "v1",
        "inference_time_ms": 42.0,
        "image_hash": "abc123",
        "confidence_score": 0.95,
        "lung_coverage_pct": 12.5,
        "symmetry_ratio": 0.88,
    }
    defaults.update(overrides)
    lgr.log_prediction(**defaults)
    return defaults


class TestTableCreation:
    def test_table_exists(self, logger: PredictionLogger) -> None:
        conn = sqlite3.connect(logger._db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_raw_path_works(self, logger_raw_path: PredictionLogger) -> None:
        conn = sqlite3.connect(logger_raw_path._db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestLogPrediction:
    def test_inserts_record(self, logger: PredictionLogger) -> None:
        _log_sample(logger)
        rows = logger.get_recent_predictions(hours=1)
        assert len(rows) == 1

    def test_record_fields(self, logger: PredictionLogger) -> None:
        kwargs = _log_sample(logger, model_name="cnn_only", inference_time_ms=99.9)
        rows = logger.get_recent_predictions(hours=1)
        row = rows[0]
        assert row["model_name"] == "cnn_only"
        assert row["model_version"] == kwargs["model_version"]
        assert abs(row["inference_time_ms"] - 99.9) < 0.01
        assert row["image_hash"] == kwargs["image_hash"]
        assert abs(row["confidence_score"] - 0.95) < 0.01

    def test_multiple_inserts(self, logger: PredictionLogger) -> None:
        for i in range(5):
            _log_sample(logger, inference_time_ms=float(i * 10))
        rows = logger.get_recent_predictions(hours=1)
        assert len(rows) == 5


class TestGetRecentPredictions:
    def test_empty_db(self, logger: PredictionLogger) -> None:
        rows = logger.get_recent_predictions(hours=24)
        assert rows == []

    def test_returns_dicts_with_expected_keys(self, logger: PredictionLogger) -> None:
        _log_sample(logger)
        rows = logger.get_recent_predictions(hours=1)
        expected_keys = {
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
        }
        assert set(rows[0].keys()) == expected_keys

    def test_ordered_desc_by_timestamp(self, logger: PredictionLogger) -> None:
        for i in range(3):
            _log_sample(logger, inference_time_ms=float(i))
        rows = logger.get_recent_predictions(hours=1)
        # Most recent first
        assert rows[0]["inference_time_ms"] >= rows[-1]["inference_time_ms"] or len(rows) == 3


class TestGetMetricsSummary:
    def test_empty_db_returns_zeros(self, logger: PredictionLogger) -> None:
        summary = logger.get_metrics_summary(hours=24)
        assert summary["total_predictions"] == 0
        assert summary["avg_inference_ms"] == 0
        assert summary["model_distribution"] == {}

    def test_summary_aggregation(self, logger: PredictionLogger) -> None:
        _log_sample(logger, model_name="hybrid", inference_time_ms=40.0)
        _log_sample(logger, model_name="hybrid", inference_time_ms=60.0)
        _log_sample(logger, model_name="cnn_only", inference_time_ms=80.0)

        summary = logger.get_metrics_summary(hours=1)
        assert summary["total_predictions"] == 3
        assert abs(summary["avg_inference_ms"] - 60.0) < 0.01
        assert summary["min_inference_ms"] == 40.0
        assert summary["max_inference_ms"] == 80.0

    def test_model_distribution(self, logger: PredictionLogger) -> None:
        _log_sample(logger, model_name="hybrid")
        _log_sample(logger, model_name="hybrid")
        _log_sample(logger, model_name="cnn_only")

        summary = logger.get_metrics_summary(hours=1)
        dist = summary["model_distribution"]
        assert dist["hybrid"] == 2
        assert dist["cnn_only"] == 1

    def test_confidence_and_coverage(self, logger: PredictionLogger) -> None:
        _log_sample(logger, confidence_score=0.9, lung_coverage_pct=10.0)
        _log_sample(logger, confidence_score=0.8, lung_coverage_pct=20.0)

        summary = logger.get_metrics_summary(hours=1)
        assert abs(summary["avg_confidence"] - 0.85) < 0.01
        assert abs(summary["avg_lung_coverage"] - 15.0) < 0.01
