"""Simple in-memory rate limiting middleware.

Uses a per-IP sliding window counter. Not suitable for multi-process
deployments without shared state (Redis, etc).
"""
import time
import threading
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.config.settings import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate-limits requests per client IP using a fixed window."""

    def __init__(self, app):
        super().__init__(app)
        self._lock = threading.Lock()
        self._windows: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        if not settings.rate_limit.enabled:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = settings.rate_limit.window_seconds
        max_requests = settings.rate_limit.max_requests

        with self._lock:
            # Prune expired entries
            self._windows[client_ip] = [
                t for t in self._windows[client_ip] if now - t < window
            ]

            if len(self._windows[client_ip]) >= max_requests:
                oldest = self._windows[client_ip][0]
                retry_after = int(window - (now - oldest)) + 1
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                    headers={"Retry-After": str(retry_after)},
                )

            self._windows[client_ip].append(now)

        return await call_next(request)
