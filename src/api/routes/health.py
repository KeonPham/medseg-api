"""Health check routes for liveness and readiness probes."""

import logging

from fastapi import APIRouter, Request

from src.api.schemas.response import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness check — confirms the API process is running."""
    return HealthResponse(status="healthy")


@router.get("/ready")
async def readiness_check(request: Request) -> dict:
    """Readiness check — confirms models are registered and ready for inference."""
    registry = request.app.state.registry
    models = registry.list_models()
    ready = len(models) > 0
    return {"ready": ready}
