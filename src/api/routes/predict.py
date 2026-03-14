"""Prediction routes for single and batch image segmentation."""

import logging

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile

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
                f"Unsupported file format: .{ext}. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )


@router.post("/predict", response_model=SegmentationResult)
async def predict_single(
    request: Request,
    file: UploadFile,
    model_name: str = Query(default="hybrid", description="Model variant to use"),
    return_overlay: bool = Query(default=False, description="Return overlay image"),
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
    return await pipeline.predict_single(
        image_bytes, model_name=model_name, return_overlay=return_overlay
    )


@router.post("/batch", response_model=BatchResult)
async def predict_batch(
    request: Request,
    files: list[UploadFile],
    model_name: str = Query(default="hybrid", description="Model variant to use"),
    return_overlay: bool = Query(default=False, description="Return overlay image"),
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
    return await pipeline.predict_batch(
        images, model_name=model_name, return_overlay=return_overlay
    )
