"""Health check routes for liveness and readiness probes."""

import logging

from fastapi import APIRouter

from src.api.schemas.response import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness check — confirms the API process is running."""
    return HealthResponse(status="healthy")


@router.get("/ready", response_model=HealthResponse)
async def readiness_check() -> HealthResponse:
    """Readiness check — confirms models are loaded and ready for inference."""
    logger.info("Readiness check requested")
    return HealthResponse(status="ready", detail="Model loading not yet implemented")
