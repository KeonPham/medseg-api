"""In-memory rate limiting per API key using sliding window counters."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

# Paths exempt from rate limiting
PUBLIC_PATHS = frozenset(
    {
        "/api/v1/health",
        "/api/v1/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


@dataclass
class RateLimitConfig:
    """Rate limit thresholds.

    Args:
        per_minute: Maximum requests per 60-second window.
        per_hour: Maximum requests per 3600-second window.
    """

    per_minute: int = 10
    per_hour: int = 100


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by API key (falls back to client IP).

    Enforces both per-minute and per-hour limits. Returns 429 with a
    Retry-After header when either limit is exceeded.
    """

    def __init__(
        self,
        app: object,
        config: RateLimitConfig | None = None,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            app: The ASGI application.
            config: Rate limit thresholds. Uses defaults if not provided.
        """
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_key(self, request: Request) -> str:
        """Extract the rate-limit key from the request.

        Prefers the X-API-Key header so limits are per-client, not per-IP.

        Args:
            request: The incoming request.

        Returns:
            A string identifying the client.
        """
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    def _prune(self, client_key: str, now: float) -> None:
        """Remove timestamps older than the hour window."""
        cutoff = now - 3600
        self._requests[client_key] = [t for t in self._requests[client_key] if t > cutoff]

    def _check_limit(self, timestamps: list[float], now: float, window: int, limit: int) -> bool:
        """Check whether the limit is exceeded for a given window.

        Args:
            timestamps: Sorted list of request timestamps.
            now: Current time.
            window: Window size in seconds.
            limit: Maximum allowed requests in the window.

        Returns:
            True if the limit is exceeded.
        """
        cutoff = now - window
        count = sum(1 for t in timestamps if t > cutoff)
        return count >= limit

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Enforce per-minute and per-hour rate limits."""
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        client_key = self._get_client_key(request)
        now = time.time()
        self._prune(client_key, now)

        timestamps = self._requests[client_key]

        # Check per-minute limit first (shorter retry)
        if self._check_limit(timestamps, now, 60, self.config.per_minute):
            oldest_in_minute = min((t for t in timestamps if t > now - 60), default=now)
            retry_after = int(60 - (now - oldest_in_minute)) + 1
            logger.warning("Per-minute rate limit exceeded for %s", client_key)
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        # Check per-hour limit
        if self._check_limit(timestamps, now, 3600, self.config.per_hour):
            oldest_in_hour = min((t for t in timestamps if t > now - 3600), default=now)
            retry_after = int(3600 - (now - oldest_in_hour)) + 1
            logger.warning("Per-hour rate limit exceeded for %s", client_key)
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        return await call_next(request)
