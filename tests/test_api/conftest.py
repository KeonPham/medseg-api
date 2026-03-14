"""Shared fixtures for API tests."""

import io
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.api.main import create_app
from src.api.schemas.response import BatchResult, SegmentationResult


@dataclass
class FakeModelInfo:
    """Minimal model info for testing."""

    name: str = "hybrid"
    version: str = "v1"
    architecture: str = "hybrid"
    metrics: dict = field(default_factory=dict)
    file_path: str = ""
    loaded_at: datetime | None = None
    is_active: bool = False
    param_count: int = 0


class FakeRegistry:
    """Fake model registry for API tests."""

    def __init__(self) -> None:
        self.default_model = "hybrid"
        self._models = {
            "hybrid": FakeModelInfo(name="hybrid", version="v1", architecture="hybrid"),
            "cnn_only": FakeModelInfo(name="cnn_only", version="v1", architecture="cnn"),
        }

    def list_models(self) -> list[FakeModelInfo]:
        return list(self._models.values())

    def get_model_info(self, name: str, version: str = "latest") -> FakeModelInfo | None:
        return self._models.get(name)


def _make_segmentation_result(model_name: str = "hybrid") -> SegmentationResult:
    """Create a fake SegmentationResult for mocking."""
    return SegmentationResult(
        model_name=model_name,
        model_version="v1",
        inference_time_ms=42.0,
        mask_base64="dGVzdA==",
        metrics={"lung_coverage_pct": 12.5, "confidence_score": 0.95, "symmetry_ratio": 0.88},
        image_size={"width": 64, "height": 64},
        timestamp=datetime(2026, 1, 1),
    )


@pytest.fixture()
def white_png_bytes() -> bytes:
    """Generate a 64x64 white PNG image as bytes."""
    img = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def fake_registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture()
def mock_pipeline() -> AsyncMock:
    pipeline = AsyncMock()
    pipeline.predict_single = AsyncMock(return_value=_make_segmentation_result())
    pipeline.predict_batch = AsyncMock(
        return_value=BatchResult(
            results=[_make_segmentation_result()],
            total_time_ms=50.0,
            count=1,
        )
    )
    return pipeline


@pytest_asyncio.fixture()
async def client(
    fake_registry: FakeRegistry, mock_pipeline: AsyncMock
) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with mocked app state."""
    app = create_app()
    app.state.settings = None
    app.state.registry = fake_registry
    app.state.pipeline = mock_pipeline
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
