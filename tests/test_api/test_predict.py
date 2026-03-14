"""Tests for prediction endpoints."""

import io
from unittest.mock import AsyncMock

import numpy as np
import pytest
from httpx import AsyncClient
from PIL import Image


def _make_png(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal PNG image."""
    img = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- POST /predict ---


@pytest.mark.asyncio
async def test_predict_single_success(client: AsyncClient) -> None:
    png = _make_png()
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "hybrid"
    assert "mask_base64" in data
    assert "metrics" in data
    assert "disclaimer" in data


@pytest.mark.asyncio
async def test_predict_single_with_model_param(client: AsyncClient) -> None:
    png = _make_png()
    resp = await client.post(
        "/api/v1/predict?model_name=cnn_only",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_predict_single_unknown_model(client: AsyncClient) -> None:
    png = _make_png()
    resp = await client.post(
        "/api/v1/predict?model_name=nonexistent",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_predict_single_invalid_format(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_predict_single_with_overlay(
    client: AsyncClient, mock_pipeline: AsyncMock
) -> None:
    png = _make_png()
    resp = await client.post(
        "/api/v1/predict?return_overlay=true",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200
    mock_pipeline.predict_single.assert_called_once()
    call_kwargs = mock_pipeline.predict_single.call_args
    assert call_kwargs.kwargs.get("return_overlay") is True


# --- POST /batch ---


@pytest.mark.asyncio
async def test_batch_success(client: AsyncClient) -> None:
    png = _make_png()
    resp = await client.post(
        "/api/v1/batch",
        files=[
            ("files", ("img1.png", png, "image/png")),
            ("files", ("img2.png", png, "image/png")),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "total_time_ms" in data
    assert data["count"] == 1  # from mock


@pytest.mark.asyncio
async def test_batch_too_many_files(client: AsyncClient) -> None:
    png = _make_png()
    files = [("files", (f"img{i}.png", png, "image/png")) for i in range(11)]
    resp = await client.post("/api/v1/batch", files=files)
    assert resp.status_code == 413
    assert "too many" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_batch_invalid_file_format(client: AsyncClient) -> None:
    png = _make_png()
    resp = await client.post(
        "/api/v1/batch",
        files=[
            ("files", ("good.png", png, "image/png")),
            ("files", ("bad.bmp", b"data", "image/bmp")),
        ],
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_batch_unknown_model(client: AsyncClient) -> None:
    png = _make_png()
    resp = await client.post(
        "/api/v1/batch?model_name=nonexistent",
        files=[("files", ("img.png", png, "image/png"))],
    )
    assert resp.status_code == 404
