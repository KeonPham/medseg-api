"""FastAPI application entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, models, monitoring, predict

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = (
    "Research and educational use only. Not intended for clinical diagnosis or "
    "medical decision-making. Always consult qualified healthcare professionals."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    logger.info("Starting MedSeg API")
    yield
    logger.info("Shutting down MedSeg API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MedSeg API",
        description=f"Medical image segmentation API. {MEDICAL_DISCLAIMER}",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])

    return app


app = create_app()
