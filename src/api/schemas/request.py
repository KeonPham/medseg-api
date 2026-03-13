"""Request schema models for the segmentation API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Parameters for a segmentation prediction request."""

    model_id: str = Field(default="hybrid", description="Model variant to use for prediction")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Binarization threshold")
    return_overlay: bool = Field(default=False, description="Whether to return mask overlay image")


class BatchPredictionRequest(BaseModel):
    """Parameters for a batch segmentation request."""

    model_id: str = Field(default="hybrid", description="Model variant to use for prediction")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Binarization threshold")
    max_images: int = Field(default=10, ge=1, le=10, description="Maximum images in batch")
