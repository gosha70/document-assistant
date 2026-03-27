import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Public routes that never require auth
PUBLIC_PATHS = {"/health", "/status", "/docs", "/openapi.json", "/redoc"}

# Routes that require at least user-level auth
USER_PATH_PREFIXES = ("/chat", "/ingest")

# Routes that require admin-level auth
ADMIN_PATH_PREFIX = "/admin"


class AuthMiddleware(BaseHTTPMiddleware):
    """Role-based auth middleware.

    When auth is enabled:
      - Public paths (/health, /status) are always open
      - /chat, /ingest require a valid user or admin token
      - /admin/* requires a valid admin token
      - All other paths require at least a user token

    When auth is disabled (default for local dev), all routes are open.
    """

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        if not settings.auth.enabled:
            return await call_next(request)

        # Allow CORS preflight requests through so the CORS middleware can handle them
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path

        # Public routes are always open
        if path in PUBLIC_PATHS or path.rstrip("/") in PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            token = None
        else:
            token = auth_header.removeprefix("Bearer ").strip()

        # Admin routes require admin token
        if path.startswith(ADMIN_PATH_PREFIX):
            if not token or token != settings.auth.admin_token:
                return JSONResponse(status_code=403, content={"detail": "Admin access denied"})
            return await call_next(request)

        # All other protected routes accept either user or admin token
        valid_tokens = set()
        if settings.auth.admin_token:
            valid_tokens.add(settings.auth.admin_token)
        if settings.auth.user_token:
            valid_tokens.add(settings.auth.user_token)

        if not token or token not in valid_tokens:
            return JSONResponse(status_code=403, content={"detail": "Authentication required"})

        return await call_next(request)
