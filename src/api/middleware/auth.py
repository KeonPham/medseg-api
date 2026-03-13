"""API key authentication middleware."""

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate API key from request headers."""

    def __init__(self, app: object, api_key: str, enabled: bool = False) -> None:
        """Initialize with the expected API key."""
        super().__init__(app)
        self.api_key = api_key
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check API key if authentication is enabled."""
        if not self.enabled:
            return await call_next(request)

        if request.url.path in ("/api/v1/health", "/docs", "/openapi.json"):
            return await call_next(request)

        key = request.headers.get("X-API-Key")
        if key != self.api_key:
            logger.warning("Invalid API key from %s", request.client.host if request.client else "unknown")
            return Response(content="Invalid API key", status_code=401)

        return await call_next(request)
