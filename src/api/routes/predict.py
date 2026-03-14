"""Prediction routes for single and batch image segmentation."""

import logging

from fastapi import APIRouter, UploadFile

from src.api.schemas.response import BatchResult, SegmentationResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=SegmentationResult)
async def predict_single(file: UploadFile) -> SegmentationResult:
    """Run segmentation on a single chest X-ray image."""
    logger.info("Received prediction request: %s", file.filename)
    raise NotImplementedError("Prediction endpoint not yet implemented")


@router.post("/batch", response_model=BatchResult)
async def predict_batch(files: list[UploadFile]) -> BatchResult:
    """Run segmentation on a batch of chest X-ray images (max 10)."""
    logger.info("Received batch prediction request: %d files", len(files))
    raise NotImplementedError("Batch prediction endpoint not yet implemented")
