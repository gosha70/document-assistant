import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

ADMIN_PATH_PREFIX = "/admin"


class AuthMiddleware(BaseHTTPMiddleware):
    """Stub auth middleware. Enforces admin token on /admin routes when auth is enabled."""

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        if not settings.auth.enabled:
            return await call_next(request)

        if request.url.path.startswith(ADMIN_PATH_PREFIX):
            token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if not token or token != settings.auth.admin_token:
                return JSONResponse(status_code=403, content={"detail": "Admin access denied"})

        return await call_next(request)
