"""Tests for health check endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_healthy(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_health_has_disclaimer_header(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/health")
    assert "X-Medical-Disclaimer" in resp.headers


@pytest.mark.asyncio
async def test_ready_returns_true_when_models_registered(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


@pytest.mark.asyncio
async def test_ready_returns_false_when_no_models(
    client: AsyncClient, fake_registry: object
) -> None:
    fake_registry._models = {}
    resp = await client.get("/api/v1/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is False
