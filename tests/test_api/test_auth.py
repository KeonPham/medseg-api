"""Tests for API key authentication."""

import io

import numpy as np
import pytest
from httpx import AsyncClient
from PIL import Image


def _make_png() -> bytes:
    img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Public endpoints (no auth needed) ──────────────────────


@pytest.mark.asyncio
async def test_health_no_auth(client_no_key: AsyncClient) -> None:
    resp = await client_no_key.get("/api/v1/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_ready_no_auth(client_no_key: AsyncClient) -> None:
    resp = await client_no_key.get("/api/v1/ready")
    assert resp.status_code == 200


# ── Protected endpoints reject missing key ─────────────────


@pytest.mark.asyncio
async def test_predict_requires_key(client_no_key: AsyncClient) -> None:
    png = _make_png()
    resp = await client_no_key.post(
        "/api/v1/predict",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 422  # missing required header


@pytest.mark.asyncio
async def test_batch_requires_key(client_no_key: AsyncClient) -> None:
    png = _make_png()
    resp = await client_no_key.post(
        "/api/v1/batch",
        files=[("files", ("img.png", png, "image/png"))],
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_models_requires_key(client_no_key: AsyncClient) -> None:
    resp = await client_no_key.get("/api/v1/models")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_metrics_requires_key(client_no_key: AsyncClient) -> None:
    resp = await client_no_key.get("/api/v1/metrics")
    assert resp.status_code == 422


# ── Invalid key returns 401 ────────────────────────────────


@pytest.mark.asyncio
async def test_predict_invalid_key(client_no_key: AsyncClient) -> None:
    png = _make_png()
    resp = await client_no_key.post(
        "/api/v1/predict",
        files={"file": ("test.png", png, "image/png")},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_models_invalid_key(client_no_key: AsyncClient) -> None:
    resp = await client_no_key.get(
        "/api/v1/models",
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# ── Valid key succeeds ─────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_valid_key(client: AsyncClient) -> None:
    png = _make_png()
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("test.png", png, "image/png")},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_models_valid_key(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
