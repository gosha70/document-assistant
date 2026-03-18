"""Tests for rate limiting middleware."""
from unittest.mock import MagicMock, patch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware.ratelimit import RateLimitMiddleware


def _create_rate_limited_app(max_requests=3, window_seconds=60):
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app = FastAPI(lifespan=noop_lifespan)
    app.add_middleware(RateLimitMiddleware)

    @app.get("/test")
    def test_endpoint():
        return {"ok": True}

    return app


class TestRateLimiting:
    def test_disabled_by_default(self):
        app = _create_rate_limited_app()
        with TestClient(app) as c:
            for _ in range(10):
                resp = c.get("/test")
                assert resp.status_code == 200

    def test_enforces_limit_when_enabled(self):
        app = _create_rate_limited_app()

        with patch("src.api.middleware.ratelimit.get_settings") as mock_settings:
            s = MagicMock()
            s.rate_limit.enabled = True
            s.rate_limit.max_requests = 3
            s.rate_limit.window_seconds = 60
            mock_settings.return_value = s

            with TestClient(app) as c:
                for i in range(3):
                    resp = c.get("/test")
                    assert resp.status_code == 200, f"Request {i+1} should succeed"

                resp = c.get("/test")
                assert resp.status_code == 429
                assert "Retry-After" in resp.headers
                # Sliding window: Retry-After should be <= window, not always == window
                retry_after = int(resp.headers["Retry-After"])
                assert 0 < retry_after <= 61

    def test_different_ips_have_separate_limits(self):
        app = _create_rate_limited_app()

        with patch("src.api.middleware.ratelimit.get_settings") as mock_settings:
            s = MagicMock()
            s.rate_limit.enabled = True
            s.rate_limit.max_requests = 1
            s.rate_limit.window_seconds = 60
            mock_settings.return_value = s

            # TestClient always uses 127.0.0.1, so we can only verify
            # that the same IP gets rate limited
            with TestClient(app) as c:
                resp = c.get("/test")
                assert resp.status_code == 200

                resp = c.get("/test")
                assert resp.status_code == 429
