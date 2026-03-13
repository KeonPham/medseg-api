"""Response schema models for the segmentation API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    detail: str | None = Field(default=None, description="Additional status information")


class PredictionResponse(BaseModel):
    """Response for a single segmentation prediction."""

    filename: str = Field(description="Original filename")
    model_id: str = Field(description="Model variant used")
    mask_base64: str = Field(description="Base64-encoded segmentation mask PNG")
    dice_score: float | None = Field(default=None, description="Dice score if ground truth given")
    disclaimer: str = Field(
        default=(
            "Research and educational use only. Not intended for clinical diagnosis or "
            "medical decision-making. Always consult qualified healthcare professionals."
        ),
        description="Medical disclaimer",
    )


class ModelInfoResponse(BaseModel):
    """Response with model metadata."""

    model_id: str = Field(description="Model identifier")
    architecture: str = Field(description="Architecture type (cnn, vit, hybrid)")
    parameters: int = Field(description="Total trainable parameters")
    input_size: int = Field(default=512, description="Expected input image size")
    dice_score: float | None = Field(default=None, description="Best validation Dice score")
    status: str = Field(default="available", description="Model availability status")
