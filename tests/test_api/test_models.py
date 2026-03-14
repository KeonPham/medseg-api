"""Tests for model management endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_models_returns_all(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_model"] == "hybrid"
    assert len(data["models"]) == 2


@pytest.mark.asyncio
async def test_list_models_contains_expected_fields(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/models")
    model = resp.json()["models"][0]
    assert "name" in model
    assert "version" in model
    assert "architecture" in model


@pytest.mark.asyncio
async def test_get_model_info_found(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/models/hybrid")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "hybrid"
    assert data["version"] == "v1"


@pytest.mark.asyncio
async def test_get_model_info_not_found(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/models/nonexistent")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
