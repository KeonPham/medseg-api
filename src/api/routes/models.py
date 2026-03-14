"""Model management routes."""

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.response import ModelListResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """List all available segmentation models with metadata."""
    registry = request.app.state.registry
    models = registry.list_models()
    model_dicts = []
    for m in models:
        d = asdict(m)
        # Convert datetime to string for JSON serialization
        if d.get("loaded_at") is not None:
            d["loaded_at"] = d["loaded_at"].isoformat()
        model_dicts.append(d)
    return ModelListResponse(
        models=model_dicts,
        default_model=registry.default_model,
    )


@router.get("/models/{model_name}")
async def get_model_info(request: Request, model_name: str) -> dict:
    """Get detailed information about a specific model."""
    registry = request.app.state.registry
    info = registry.get_model_info(model_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    d = asdict(info)
    if d.get("loaded_at") is not None:
        d["loaded_at"] = d["loaded_at"].isoformat()
    return d
