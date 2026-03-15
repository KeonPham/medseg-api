"""Monitoring routes for metrics and prediction history."""

import logging

from fastapi import APIRouter, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
async def get_metrics(
    request: Request,
    hours: int = Query(default=24, description="Look-back window in hours"),
) -> dict:
    """Return aggregated prediction metrics."""
    pred_logger = getattr(request.app.state, "prediction_logger", None)
    if pred_logger is None:
        return {"error": "Prediction logger not configured"}
    return pred_logger.get_metrics_summary(hours=hours)


@router.get("/predictions")
async def get_recent_predictions(
    request: Request,
    hours: int = Query(default=24, description="Look-back window in hours"),
) -> list[dict]:
    """Return recent prediction records."""
    pred_logger = getattr(request.app.state, "prediction_logger", None)
    if pred_logger is None:
        return []
    return pred_logger.get_recent_predictions(hours=hours)
