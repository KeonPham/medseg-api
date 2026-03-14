"""Response schema models for the segmentation API."""

from datetime import datetime

from pydantic import BaseModel, Field

MEDICAL_DISCLAIMER = (
    "Research and educational use only. Not intended for clinical diagnosis or "
    "medical decision-making. Always consult qualified healthcare professionals."
)


class SegmentationResult(BaseModel):
    """Result of a single segmentation prediction."""

    model_name: str = Field(description="Model variant used")
    model_version: str = Field(default="latest", description="Model version")
    inference_time_ms: float = Field(description="Inference latency in milliseconds")
    mask_base64: str = Field(description="Base64-encoded segmentation mask PNG")
    overlay_base64: str | None = Field(default=None, description="Base64 overlay image")
    metrics: dict = Field(
        default_factory=dict,
        description="Computed metrics: lung_coverage_pct, confidence_score, symmetry_ratio",
    )
    image_size: dict = Field(
        default_factory=dict, description="Original image dimensions: width, height"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    disclaimer: str = Field(default=MEDICAL_DISCLAIMER)


class BatchResult(BaseModel):
    """Result of a batch segmentation request."""

    results: list[SegmentationResult]
    total_time_ms: float = Field(description="Total processing time in milliseconds")
    count: int = Field(description="Number of images processed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0", description="API version")


class ModelListResponse(BaseModel):
    """Response listing all available models."""

    models: list[dict] = Field(description="Available model metadata")
    default_model: str = Field(description="Default model name")
