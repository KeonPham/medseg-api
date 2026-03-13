"""Model management routes."""

import logging

from fastapi import APIRouter

from src.api.schemas.response import ModelInfoResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models", response_model=list[ModelInfoResponse])
async def list_models() -> list[ModelInfoResponse]:
    """List all available segmentation models with metadata."""
    logger.info("Listing available models")
    raise NotImplementedError("Model listing not yet implemented")


@router.get("/models/{model_id}/info", response_model=ModelInfoResponse)
async def get_model_info(model_id: str) -> ModelInfoResponse:
    """Get detailed information about a specific model."""
    logger.info("Getting info for model: %s", model_id)
    raise NotImplementedError("Model info endpoint not yet implemented")
