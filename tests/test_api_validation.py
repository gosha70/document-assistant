"""Automated tests for manual test plan gaps.

Covers:
  - Ingestion validation (sections 3.5-3.9)
  - Auth comprehensive (sections 10.5-10.10)
  - CORS headers (sections 19.1-19.3)
  - OpenAPI docs (section 21.2)
  - Edge cases (sections 20.7-20.8)
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import deps
from src.rag.vectorstore import VectorStoreBackend


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _create_test_app():
    """Create a FastAPI app with a no-op lifespan (skips real model loading)."""

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    from fastapi.middleware.cors import CORSMiddleware
    from src.api.middleware.auth import AuthMiddleware
    from src.api.middleware.ratelimit import RateLimitMiddleware
    from src.api.middleware.telemetry import TelemetryMiddleware
    from src.api.routes import chat, ingest, status, admin
    from src.config.settings import get_settings

    settings = get_settings()
    app = FastAPI(title=settings.app.name, lifespan=noop_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TelemetryMiddleware)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(status.router)
    app.include_router(admin.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def _make_mock_backend():
    backend = MagicMock(spec=VectorStoreBackend)
    backend.count.return_value = 0
    backend.list_collections.return_value = []
    backend.get_collection_info.return_value = {}
    backend.search.return_value = []
    backend.hybrid_search.return_value = []
    backend.get_embedding_provenance.return_value = None
    backend.find_by_source.return_value = []
    return backend


def _override_deps(backend):
    deps._backend = backend
    deps._llm = None
    deps._reranker = None
    deps._embedding = None


def _cleanup_deps():
    deps._backend = None
    deps._llm = None
    deps._reranker = None
    deps._embedding = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from src.utils.jobs import reset_jobs

    app = _create_test_app()
    backend = _make_mock_backend()
    _override_deps(backend)
    reset_jobs()
    with TestClient(app) as c:
        yield c
    _cleanup_deps()


@pytest.fixture
def mock_backend():
    return _make_mock_backend()


# ---------------------------------------------------------------------------
# Section 3.5-3.9: Ingestion validation
# ---------------------------------------------------------------------------


class TestIngestValidation:
    def test_unsupported_file_extension_rejected(self, client):
        """A .png upload must be rejected with HTTP 400 referencing file type."""
        data = {"files": ("image.png", io.BytesIO(b"\x89PNG\r\n"), "image/png")}
        resp = client.post("/ingest", files=data)
        assert resp.status_code == 400
        body = resp.json()
        assert "detail" in body
        detail = body["detail"].lower()
        assert "png" in detail or "not allowed" in detail or "file type" in detail

    def test_empty_file_handled_gracefully(self, client):
        """A 0-byte .txt upload must not crash the server (200 with 0 docs or a handled error)."""
        data = {"files": ("empty.txt", io.BytesIO(b""), "text/plain")}
        resp = client.post("/ingest", files=data)
        # Acceptable outcomes: 200 with document_count 0, or a graceful 4xx — never a 500
        assert resp.status_code != 500
        if resp.status_code == 200:
            assert resp.json()["document_count"] == 0

    def test_path_traversal_filename_rejected(self):
        """Filename with path traversal components must be rejected with HTTP 400."""
        app = _create_test_app()
        backend = _make_mock_backend()
        _override_deps(backend)
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                # multipart: set the filename field to a traversal path
                data = {"files": ("../../etc/passwd", io.BytesIO(b"content"), "text/plain")}
                resp = c.post("/ingest", files=data)
                assert resp.status_code == 400
                body = resp.json()
                assert "detail" in body
        finally:
            _cleanup_deps()


# ---------------------------------------------------------------------------
# Section 10.5-10.10: Auth comprehensive
# (test_api.py already covers: admin blocked without token, chat requires token,
#  public routes open, OPTIONS passes — this class adds the remaining gaps)
# ---------------------------------------------------------------------------


class TestAuthComprehensive:
    def _make_auth_settings(self, admin_token="admin-secret", user_token="user-secret"):
        s = MagicMock()
        s.auth.enabled = True
        s.auth.admin_token = admin_token
        s.auth.user_token = user_token
        return s

    def _make_client_with_auth(self, backend=None):
        app = _create_test_app()
        be = backend or _make_mock_backend()
        _override_deps(be)
        return app, be

    def test_admin_route_with_user_token_returns_403(self):
        """A valid user token must not grant access to admin routes."""
        app, _ = self._make_client_with_auth()
        try:
            with patch("src.api.middleware.auth.get_settings") as mock_settings:
                mock_settings.return_value = self._make_auth_settings()
                with TestClient(app, raise_server_exceptions=False) as c:
                    resp = c.get(
                        "/admin/collections",
                        headers={"Authorization": "Bearer user-secret"},
                    )
                    assert resp.status_code == 403
        finally:
            _cleanup_deps()

    def test_admin_route_with_admin_token_succeeds(self):
        """A valid admin token must grant access to admin routes."""
        backend = _make_mock_backend()
        backend.list_collections.return_value = []
        app, _ = self._make_client_with_auth(backend)
        try:
            with patch("src.api.middleware.auth.get_settings") as mock_settings:
                mock_settings.return_value = self._make_auth_settings()
                with TestClient(app, raise_server_exceptions=False) as c:
                    resp = c.get(
                        "/admin/collections",
                        headers={"Authorization": "Bearer admin-secret"},
                    )
                    assert resp.status_code == 200
        finally:
            _cleanup_deps()

    def test_invalid_token_returns_403(self):
        """An unrecognised token must be rejected with 403."""
        app, _ = self._make_client_with_auth()
        try:
            with patch("src.api.middleware.auth.get_settings") as mock_settings:
                mock_settings.return_value = self._make_auth_settings()
                with TestClient(app, raise_server_exceptions=False) as c:
                    resp = c.get(
                        "/admin/collections",
                        headers={"Authorization": "Bearer not-a-real-token"},
                    )
                    assert resp.status_code == 403
        finally:
            _cleanup_deps()

    def test_malformed_auth_header_returns_403(self):
        """Authorization header without 'Bearer ' prefix must be rejected even if the
        bare value matches a valid token (RFC 6750 requires the Bearer scheme)."""
        app, _ = self._make_client_with_auth()
        try:
            with patch("src.api.middleware.auth.get_settings") as mock_settings:
                mock_settings.return_value = self._make_auth_settings()
                with TestClient(app, raise_server_exceptions=False) as c:
                    # Send the exact configured admin token WITHOUT the "Bearer " prefix
                    resp = c.get(
                        "/admin/collections",
                        headers={"Authorization": "admin-secret"},
                    )
                    assert resp.status_code == 403
        finally:
            _cleanup_deps()

    def test_admin_token_grants_user_access(self):
        """An admin token must also be accepted on user-level routes such as /chat."""
        backend = _make_mock_backend()
        backend.hybrid_search.return_value = []
        app, _ = self._make_client_with_auth(backend)
        deps._llm = MagicMock()
        deps._embedding = MagicMock()
        try:
            with patch("src.api.middleware.auth.get_settings") as mock_settings:
                mock_settings.return_value = self._make_auth_settings()
                with TestClient(app, raise_server_exceptions=False) as c:
                    resp = c.post(
                        "/chat",
                        json={"question": "test?"},
                        headers={"Authorization": "Bearer admin-secret"},
                    )
                    assert resp.status_code == 200
        finally:
            _cleanup_deps()


# ---------------------------------------------------------------------------
# Section 19.1-19.3: CORS headers
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    def test_simple_cors_request_includes_allow_origin(self, client):
        """A request carrying an Origin header must receive Access-Control-Allow-Origin."""
        resp = client.get("/health", headers={"Origin": "http://example.com"})
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_preflight_returns_cors_headers(self, client):
        """An OPTIONS preflight must return Allow-Methods and Allow-Headers."""
        resp = client.options(
            "/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        # CORS preflight is handled before auth; status must not be 403
        assert resp.status_code != 403
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert "access-control-allow-methods" in headers_lower
        assert "access-control-allow-headers" in headers_lower

    def test_preflight_with_auth_enabled_not_blocked(self):
        """OPTIONS bypass auth and still carry CORS headers when auth is enabled."""
        app = _create_test_app()
        backend = _make_mock_backend()
        _override_deps(backend)
        try:
            with patch("src.api.middleware.auth.get_settings") as mock_settings:
                s = MagicMock()
                s.auth.enabled = True
                s.auth.admin_token = "admin-secret"
                s.auth.user_token = "user-secret"
                mock_settings.return_value = s

                with TestClient(app, raise_server_exceptions=False) as c:
                    resp = c.options(
                        "/chat",
                        headers={
                            "Origin": "http://example.com",
                            "Access-Control-Request-Method": "POST",
                            "Access-Control-Request-Headers": "Authorization",
                        },
                    )
                    assert resp.status_code != 403
                    assert "access-control-allow-origin" in resp.headers
        finally:
            _cleanup_deps()


# ---------------------------------------------------------------------------
# Section 21.2: OpenAPI docs
# ---------------------------------------------------------------------------


class TestOpenAPIDocs:
    def test_openapi_json_returns_valid_schema(self, client):
        """GET /openapi.json must return HTTP 200 with a valid OpenAPI object."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "info" in schema


# ---------------------------------------------------------------------------
# Section 20.7-20.8: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_extremely_long_question_does_not_crash(self, client):
        """A 10,000-character question must not cause a 500 error."""
        long_question = "a" * 10_000
        resp = client.post("/chat", json={"question": long_question})
        assert resp.status_code != 500

    def test_special_characters_in_question_treated_as_text(self, client):
        """Questions containing XSS and SQL injection patterns must not cause a 500."""
        malicious_question = "<script>alert('xss')</script> OR '1'='1'; DROP TABLE users;--"
        resp = client.post("/chat", json={"question": malicious_question})
        assert resp.status_code != 500
