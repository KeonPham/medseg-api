"""Prediction routes for single and batch image segmentation."""

import hashlib
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile

from src.api.middleware.auth import verify_api_key
from src.api.schemas.response import BatchResult, SegmentationResult

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "application/dicom",
}

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}

MAX_BATCH_SIZE = 10


def _validate_image_file(file: UploadFile) -> None:
    """Validate that an uploaded file is an acceptable image format.

    Args:
        file: The uploaded file to validate.

    Raises:
        HTTPException: If the file format is not supported.
    """
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    dotted_ext = f".{ext}" if ext else ""

    if dotted_ext and dotted_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format: .{ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )


@router.post("/predict", response_model=SegmentationResult)
async def predict_single(
    request: Request,
    file: UploadFile,
    model_name: str = Query(default="hybrid", description="Model variant to use"),
    return_overlay: bool = Query(default=False, description="Return overlay image"),
    return_explainability: bool = Query(
        default=False, description="Return explainability data (heatmap, GradCAM, findings)"
    ),
    _client: str = Depends(verify_api_key),
) -> SegmentationResult:
    """Run segmentation on a single chest X-ray image."""
    _validate_image_file(file)

    registry = request.app.state.registry
    if registry.get_model_info(model_name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found",
        )

    image_bytes = await file.read()
    pipeline = request.app.state.pipeline
    result = await pipeline.predict_single(
        image_bytes,
        model_name=model_name,
        return_overlay=return_overlay,
        return_explainability=return_explainability,
    )

    # Log prediction
    pred_logger = getattr(request.app.state, "prediction_logger", None)
    if pred_logger is not None:
        pred_logger.log_prediction(
            request_id=str(uuid.uuid4()),
            model_name=result.model_name,
            model_version=result.model_version,
            inference_time_ms=result.inference_time_ms,
            image_hash=hashlib.sha256(image_bytes).hexdigest(),
            confidence_score=result.metrics.get("confidence_score", 0.0),
            lung_coverage_pct=result.metrics.get("lung_coverage_pct", 0.0),
            symmetry_ratio=result.metrics.get("symmetry_ratio", 0.0),
        )

    return result


@router.post("/batch", response_model=BatchResult)
async def predict_batch(
    request: Request,
    files: list[UploadFile],
    model_name: str = Query(default="hybrid", description="Model variant to use"),
    return_overlay: bool = Query(default=False, description="Return overlay image"),
    _client: str = Depends(verify_api_key),
) -> BatchResult:
    """Run segmentation on a batch of chest X-ray images (max 10)."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files. Maximum batch size is {MAX_BATCH_SIZE}.",
        )

    for f in files:
        _validate_image_file(f)

    registry = request.app.state.registry
    if registry.get_model_info(model_name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found",
        )

    images = [await f.read() for f in files]
    pipeline = request.app.state.pipeline
    batch_result = await pipeline.predict_batch(
        images, model_name=model_name, return_overlay=return_overlay
    )

    # Log each prediction in the batch
    pred_logger = getattr(request.app.state, "prediction_logger", None)
    if pred_logger is not None:
        for i, result in enumerate(batch_result.results):
            pred_logger.log_prediction(
                request_id=str(uuid.uuid4()),
                model_name=result.model_name,
                model_version=result.model_version,
                inference_time_ms=result.inference_time_ms,
                image_hash=hashlib.sha256(images[i]).hexdigest(),
                confidence_score=result.metrics.get("confidence_score", 0.0),
                lung_coverage_pct=result.metrics.get("lung_coverage_pct", 0.0),
                symmetry_ratio=result.metrics.get("symmetry_ratio", 0.0),
            )

    return batch_result
