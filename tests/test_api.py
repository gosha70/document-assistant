"""Tests for the FastAPI application routes."""
import pytest
from unittest.mock import MagicMock, patch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import deps
from src.api.schemas import CollectionInfo
from src.rag.vectorstore import VectorStoreBackend


def _create_test_app():
    """Create a FastAPI app with a no-op lifespan (skips real model loading)."""

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    from src.api.main import create_app as _real_create
    from fastapi.middleware.cors import CORSMiddleware
    from src.api.middleware.auth import AuthMiddleware
    from src.api.middleware.telemetry import TelemetryMiddleware
    from src.api.routes import chat, ingest, status, admin
    from src.config.settings import get_settings

    settings = get_settings()
    app = FastAPI(title=settings.app.name, lifespan=noop_lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(AuthMiddleware)
    app.add_middleware(TelemetryMiddleware)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(status.router)
    app.include_router(admin.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


@pytest.fixture
def mock_backend():
    backend = MagicMock(spec=VectorStoreBackend)
    backend.count.return_value = 42
    backend.list_collections.return_value = [
        {"name": "test_col", "backend": "chroma", "document_count": 42, "persist_directory": "/tmp", "embedding_model": "test"}
    ]
    backend.get_collection_info.return_value = {
        "name": "test_col", "backend": "chroma", "document_count": 42, "persist_directory": "/tmp", "embedding_model": "test"
    }
    backend.search.return_value = []
    backend.hybrid_search.return_value = []
    return backend


@pytest.fixture
def client(mock_backend):
    app = _create_test_app()
    deps._backend = mock_backend
    deps._llm = None
    with TestClient(app) as c:
        yield c
    deps._backend = None
    deps._llm = None


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestStatusEndpoint:
    def test_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] == 42
        assert data["app_name"] == "Document Assistant (D.O.T.)"
        assert data["vectorstore_backend"] == "chroma"


class TestChatEndpoint:
    def test_chat_no_llm_returns_503(self, client):
        resp = client.post("/chat", json={"question": "What is Python?"})
        assert resp.status_code == 503

    def test_chat_no_documents_found(self, client, mock_backend):
        mock_backend.hybrid_search.return_value = []
        mock_llm = MagicMock()
        deps._llm = mock_llm
        resp = client.post("/chat", json={"question": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "No relevant documents" in data["answer"]
        assert data["sources"] == []

    def test_chat_returns_citations(self, client, mock_backend):
        from langchain_core.documents import Document

        mock_backend.hybrid_search.return_value = [
            Document(page_content="Python is a language.", metadata={"source": "intro.txt", "page": 1}),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Python is a programming language."
        deps._llm = mock_llm

        resp = client.post("/chat", json={"question": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Python is a programming language."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file"] == "intro.txt"
        assert data["sources"][0]["page"] == 1

    def test_chat_empty_question_rejected(self, client):
        resp = client.post("/chat", json={"question": ""})
        assert resp.status_code == 422


class TestAdminEndpoints:
    def test_list_collections(self, client):
        resp = client.get("/admin/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "test_col"

    def test_get_collection(self, client):
        resp = client.get("/admin/collections/test_col")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test_col"

    def test_reindex_stub(self, client):
        resp = client.post("/admin/collections/test_col/reindex")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_export_stub(self, client):
        resp = client.post("/admin/collections/test_col/export")
        assert resp.status_code == 200
        assert resp.json()["type"] == "export"

    def test_list_jobs_empty(self, client):
        resp = client.get("/admin/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_job_not_found(self, client):
        resp = client.get("/admin/jobs/nonexistent")
        assert resp.status_code == 404

    def test_metrics_runtime(self, client):
        resp = client.get("/admin/metrics/runtime")
        assert resp.status_code == 200
        data = resp.json()
        assert "application" in data
        assert "process" in data
        assert "uptime_seconds" in data["application"]
        assert "request_count" in data["application"]
        assert "llm_call_count" in data["application"]
        assert "memory_rss_mb" in data["process"]
        # request_count should be > 0 since the middleware counts this request
        assert data["application"]["request_count"] > 0

    def test_reports_summary(self, client):
        resp = client.get("/admin/reports/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "total_errors" in data
        assert "uptime_seconds" in data
        assert data["total_requests"] > 0


    def test_get_nonexistent_collection_returns_404(self, client, mock_backend):
        mock_backend.get_collection_info.side_effect = ValueError("Collection 'nope' not found")
        resp = client.get("/admin/collections/nope")
        assert resp.status_code == 404


class TestBackendFactory:
    def test_unknown_backend_raises(self):
        from src.api.main import _create_backend
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.vectorstore.backend = "unknown_db"
        with pytest.raises(ValueError, match="Unknown vectorstore backend"):
            _create_backend(settings)


class TestAuthMiddleware:
    def test_admin_blocked_when_auth_enabled(self):
        app = _create_test_app()
        mock_backend = MagicMock(spec=VectorStoreBackend)
        mock_backend.list_collections.return_value = []
        deps._backend = mock_backend

        with patch("src.api.middleware.auth.get_settings") as mock_settings:
            s = MagicMock()
            s.auth.enabled = True
            s.auth.admin_token = "secret-token"
            mock_settings.return_value = s

            with TestClient(app, raise_server_exceptions=False) as c:
                # No token
                resp = c.get("/admin/collections")
                assert resp.status_code == 403

                # Wrong token
                resp = c.get("/admin/collections", headers={"Authorization": "Bearer wrong"})
                assert resp.status_code == 403

                # Correct token
                resp = c.get("/admin/collections", headers={"Authorization": "Bearer secret-token"})
                assert resp.status_code == 200

        deps._backend = None
