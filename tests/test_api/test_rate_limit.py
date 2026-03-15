"""Tests for rate limiting middleware."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_rate_limit_allows_normal_traffic(client: AsyncClient) -> None:
    """A few requests should succeed without hitting any limit."""
    for _ in range(5):
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_exempt_from_rate_limit(client_no_key: AsyncClient) -> None:
    """Health endpoint should never be rate-limited."""
    for _ in range(15):
        resp = await client_no_key.get("/api/v1/health")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_per_minute_returns_429(client: AsyncClient) -> None:
    """Exceeding per-minute limit (10) should return 429 with Retry-After."""
    # Send 10 requests to exhaust the per-minute limit
    for i in range(10):
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200, f"Request {i + 1} failed unexpectedly"

    # 11th request should be rate-limited
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
    retry_after = int(resp.headers["Retry-After"])
    assert 0 < retry_after <= 61
