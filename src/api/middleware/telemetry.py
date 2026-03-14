"""Middleware that counts all HTTP requests and errors."""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.utils.metrics import get_metrics_collector
from src.config.settings import get_settings


class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        if not settings.telemetry.enabled:
            return await call_next(request)

        collector = get_metrics_collector()
        collector.record_request()

        response = await call_next(request)

        if response.status_code >= 500:
            collector.record_error()

        return response
