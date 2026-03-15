"""FastAPI application entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.middleware.auth import APIKeyValidator, set_validator
from src.api.middleware.rate_limit import RateLimitConfig, RateLimitMiddleware
from src.api.routes import health, models, monitoring, predict
from src.api.schemas.response import MEDICAL_DISCLAIMER
from src.models.inference import InferencePipeline
from src.models.registry import ModelRegistry
from src.monitoring.prediction_logger import PredictionLogger
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

API_KEYS_PATH = Path("configs/api_keys.json")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    logger.info("Starting MedSegAPI")

    settings = get_settings()
    registry = ModelRegistry()
    pipeline = InferencePipeline(registry=registry, config=settings.model)

    pred_logger = PredictionLogger(db_url=settings.database.url)

    # Configure API key auth — reload keys from disk at startup
    set_validator(
        APIKeyValidator(
            keys_path=API_KEYS_PATH,
            enabled=API_KEYS_PATH.exists(),
        )
    )

    app.state.settings = settings
    app.state.registry = registry
    app.state.pipeline = pipeline
    app.state.prediction_logger = pred_logger

    logger.info("MedSegAPI ready (default_model=%s)", registry.default_model)
    yield
    logger.info("Shutting down MedSegAPI")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MedSegAPI",
        description="Production medical image segmentation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        RateLimitMiddleware,
        config=RateLimitConfig(per_minute=10, per_hour=100),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_disclaimer_header(request: Request, call_next: object) -> Response:
        """Add medical disclaimer header to every response."""
        response = await call_next(request)
        response.headers["X-Medical-Disclaimer"] = MEDICAL_DISCLAIMER
        return response

    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        """Serve the single-page frontend."""
        return FileResponse("frontend/index.html", media_type="text/html")

    return app


app = create_app()
