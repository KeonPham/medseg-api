"""Monitoring routes for metrics and A/B test results."""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
async def get_metrics() -> dict:
    """Return Prometheus-compatible metrics."""
    logger.info("Metrics requested")
    return {"message": "Metrics endpoint not yet implemented"}


@router.get("/ab-results")
async def get_ab_results() -> dict:
    """Return A/B test comparison results between model variants."""
    logger.info("A/B results requested")
    return {"message": "A/B results endpoint not yet implemented"}
