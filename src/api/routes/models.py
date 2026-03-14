"""Model management routes."""

import logging

from fastapi import APIRouter

from src.api.schemas.response import ModelListResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List all available segmentation models with metadata."""
    logger.info("Listing available models")
    raise NotImplementedError("Model listing not yet implemented")


@router.get("/models/{model_id}/info")
async def get_model_info(model_id: str) -> dict:
    """Get detailed information about a specific model."""
    logger.info("Getting info for model: %s", model_id)
    raise NotImplementedError("Model info endpoint not yet implemented")
