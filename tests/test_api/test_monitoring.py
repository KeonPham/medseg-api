"""Tests for monitoring API endpoints."""

import pytest
from httpx import AsyncClient

from src.monitoring.prediction_logger import PredictionLogger


@pytest.mark.asyncio
async def test_metrics_returns_summary(
    client_with_logger: AsyncClient, tmp_logger: PredictionLogger
) -> None:
    tmp_logger.log_prediction(
        request_id="r1",
        model_name="hybrid",
        model_version="v1",
        inference_time_ms=50.0,
        image_hash="aaa",
        confidence_score=0.9,
        lung_coverage_pct=15.0,
        symmetry_ratio=0.85,
    )
    resp = await client_with_logger.get("/api/v1/metrics?hours=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_predictions"] == 1
    assert data["avg_inference_ms"] == 50.0
    assert "hybrid" in data["model_distribution"]


@pytest.mark.asyncio
async def test_metrics_empty(client_with_logger: AsyncClient) -> None:
    resp = await client_with_logger.get("/api/v1/metrics?hours=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_predictions"] == 0


@pytest.mark.asyncio
async def test_predictions_returns_list(
    client_with_logger: AsyncClient, tmp_logger: PredictionLogger
) -> None:
    tmp_logger.log_prediction(
        request_id="r1",
        model_name="hybrid",
        model_version="v1",
        inference_time_ms=50.0,
        image_hash="aaa",
        confidence_score=0.9,
        lung_coverage_pct=15.0,
        symmetry_ratio=0.85,
    )
    resp = await client_with_logger.get("/api/v1/predictions?hours=1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["model_name"] == "hybrid"


@pytest.mark.asyncio
async def test_predictions_empty(client_with_logger: AsyncClient) -> None:
    resp = await client_with_logger.get("/api/v1/predictions?hours=1")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_metrics_without_logger(client: AsyncClient) -> None:
    """When prediction_logger is not set, /metrics returns an error dict."""
    resp = await client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_predictions_without_logger(client: AsyncClient) -> None:
    """When prediction_logger is not set, /predictions returns empty list."""
    resp = await client.get("/api/v1/predictions")
    assert resp.status_code == 200
    assert resp.json() == []
