"""Tests verifying prediction logging is called during predict endpoints."""

import io

import numpy as np
import pytest
from httpx import AsyncClient
from PIL import Image

from src.monitoring.prediction_logger import PredictionLogger


def _make_png(width: int = 64, height: int = 64) -> bytes:
    img = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_predict_single_logs_prediction(
    client_with_logger: AsyncClient, tmp_logger: PredictionLogger
) -> None:
    """Single prediction should create a log entry."""
    png = _make_png()
    resp = await client_with_logger.post(
        "/api/v1/predict",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200

    rows = tmp_logger.get_recent_predictions(hours=1)
    assert len(rows) == 1
    assert rows[0]["model_name"] == "hybrid"
    assert rows[0]["inference_time_ms"] == 42.0


@pytest.mark.asyncio
async def test_predict_batch_logs_all(
    client_with_logger: AsyncClient, tmp_logger: PredictionLogger
) -> None:
    """Batch prediction should create a log entry per image."""
    png = _make_png()
    resp = await client_with_logger.post(
        "/api/v1/batch",
        files=[
            ("files", ("img1.png", png, "image/png")),
            ("files", ("img2.png", png, "image/png")),
        ],
    )
    assert resp.status_code == 200

    # The mock returns 1 result in the batch, so we expect 1 log entry
    rows = tmp_logger.get_recent_predictions(hours=1)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_predict_without_logger_still_works(client: AsyncClient) -> None:
    """When no logger is attached, prediction should still succeed."""
    png = _make_png()
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200
