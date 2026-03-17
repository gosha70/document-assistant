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
    from src.api.middleware.ratelimit import RateLimitMiddleware
    from src.api.middleware.telemetry import TelemetryMiddleware
    from src.api.routes import chat, ingest, status, admin
    from src.config.settings import get_settings

    settings = get_settings()
    app = FastAPI(title=settings.app.name, lifespan=noop_lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
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


@pytest.fixture
def mock_backend():
    backend = MagicMock(spec=VectorStoreBackend)
    backend.count.return_value = 42
    backend.list_collections.return_value = [
        {
            "name": "test_col",
            "backend": "chroma",
            "document_count": 42,
            "persist_directory": "/tmp",
            "embedding_model": "test",
        }
    ]
    backend.get_collection_info.return_value = {
        "name": "test_col",
        "backend": "chroma",
        "document_count": 42,
        "persist_directory": "/tmp",
        "embedding_model": "test",
    }
    backend.search.return_value = []
    backend.hybrid_search.return_value = []
    backend.get_embedding_provenance.return_value = None
    return backend


@pytest.fixture
def client(mock_backend):
    from src.utils.jobs import reset_jobs

    app = _create_test_app()
    deps._backend = mock_backend
    deps._llm = None
    deps._reranker = None
    deps._embedding = None
    reset_jobs()
    with TestClient(app) as c:
        yield c
    deps._backend = None
    deps._llm = None
    deps._reranker = None
    deps._embedding = None


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
        deps._embedding = MagicMock()
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
        deps._embedding = MagicMock()

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


class TestChatStreamEndpoint:
    def test_stream_no_llm_returns_503(self, client):
        resp = client.post("/chat/stream", json={"question": "What is Python?"})
        assert resp.status_code == 503

    def test_stream_no_documents_returns_empty(self, client, mock_backend):
        mock_backend.hybrid_search.return_value = []
        mock_llm = MagicMock()
        deps._llm = mock_llm
        deps._embedding = MagicMock()
        resp = client.post("/chat/stream", json={"question": "What is Python?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "No relevant documents" in body
        assert "[DONE]" in body

    def test_stream_returns_tokens_and_sources(self, client, mock_backend):
        from langchain_core.documents import Document

        mock_backend.hybrid_search.return_value = [
            Document(page_content="Python is great.", metadata={"source": "intro.txt", "page": 1}),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Python is a programming language."
        # No .stream attribute, so generator falls back to invoke
        if hasattr(mock_llm, "stream"):
            del mock_llm.stream
        deps._llm = mock_llm
        deps._embedding = MagicMock()

        resp = client.post("/chat/stream", json={"question": "What is Python?"})
        assert resp.status_code == 200
        body = resp.text
        assert "token" in body
        assert "sources" in body
        assert "intro.txt" in body
        assert "[DONE]" in body


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

    def test_reindex_creates_job(self, client):
        resp = client.post("/admin/collections/test_col/reindex")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "reindex"
        assert data["status"] in ("queued", "running", "completed")
        assert data["collection_name"] == "test_col"
        assert data["id"]
        assert data["created_at"]

    def test_export_creates_job(self, client):
        resp = client.post("/admin/collections/test_col/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "export"
        assert data["collection_name"] == "test_col"

    def test_list_jobs_empty(self, client):
        resp = client.get("/admin/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_jobs_after_submit(self, client):
        client.post("/admin/collections/test_col/reindex")
        resp = client.get("/admin/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) == 1
        assert jobs[0]["type"] == "reindex"

    def test_get_job_by_id(self, client):
        create_resp = client.post("/admin/collections/test_col/reindex")
        job_id = create_resp.json()["id"]
        resp = client.get(f"/admin/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id

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
        settings.embedding.type = "instructor"
        with patch("src.rag.embeddings.InstructorEmbeddingAdapter"):
            with pytest.raises(ValueError, match="Unknown vectorstore backend"):
                _create_backend(settings)


class TestLLMFactory:
    def test_openai_backend(self):
        import sys

        mock_module = MagicMock()
        mock_chat_cls = MagicMock()
        mock_module.ChatOpenAI = mock_chat_cls
        with patch.dict(sys.modules, {"langchain_openai": mock_module}):
            from importlib import reload
            import src.api.main as main_mod

            reload(main_mod)

            settings = MagicMock()
            settings.llm.backend = "openai"
            settings.llm.model = "gpt-4o"
            settings.llm.base_url = None
            settings.llm.api_key = "test-key"
            settings.llm.temperature = 0.2
            settings.llm.max_tokens = 512

            main_mod._create_llm(settings)
            mock_chat_cls.assert_called_once_with(
                model="gpt-4o",
                temperature=0.2,
                max_tokens=512,
                api_key="test-key",
            )

    def test_ollama_backend_defaults(self):
        import sys

        mock_module = MagicMock()
        mock_chat_cls = MagicMock()
        mock_module.ChatOpenAI = mock_chat_cls
        with patch.dict(sys.modules, {"langchain_openai": mock_module}):
            from importlib import reload
            import src.api.main as main_mod

            reload(main_mod)

            settings = MagicMock()
            settings.llm.backend = "ollama"
            settings.llm.model = None
            settings.llm.base_url = None
            settings.llm.api_key = None
            settings.llm.temperature = 0.2
            settings.llm.max_tokens = 512

            main_mod._create_llm(settings)
            mock_chat_cls.assert_called_once_with(
                model="llama3.2",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                temperature=0.2,
                max_tokens=512,
            )

    def test_unknown_llm_backend_raises(self):
        from src.api.main import _create_llm

        settings = MagicMock()
        settings.llm.backend = "unknown"
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            _create_llm(settings)


class TestAuthMiddleware:
    def _make_auth_settings(self, admin_token="admin-secret", user_token="user-secret"):
        s = MagicMock()
        s.auth.enabled = True
        s.auth.admin_token = admin_token
        s.auth.user_token = user_token
        return s

    def test_admin_blocked_when_auth_enabled(self):
        app = _create_test_app()
        mock_be = MagicMock(spec=VectorStoreBackend)
        mock_be.list_collections.return_value = []
        deps._backend = mock_be

        with patch("src.api.middleware.auth.get_settings") as mock_settings:
            mock_settings.return_value = self._make_auth_settings()

            with TestClient(app, raise_server_exceptions=False) as c:
                # No token
                resp = c.get("/admin/collections")
                assert resp.status_code == 403

                # Wrong token
                resp = c.get("/admin/collections", headers={"Authorization": "Bearer wrong"})
                assert resp.status_code == 403

                # User token cannot access admin
                resp = c.get("/admin/collections", headers={"Authorization": "Bearer user-secret"})
                assert resp.status_code == 403

                # Admin token works
                resp = c.get("/admin/collections", headers={"Authorization": "Bearer admin-secret"})
                assert resp.status_code == 200

        deps._backend = None
        deps._reranker = None

    def test_chat_requires_user_token(self):
        app = _create_test_app()
        mock_be = MagicMock(spec=VectorStoreBackend)
        mock_be.hybrid_search.return_value = []
        deps._backend = mock_be
        deps._llm = MagicMock()
        deps._embedding = MagicMock()

        with patch("src.api.middleware.auth.get_settings") as mock_settings:
            mock_settings.return_value = self._make_auth_settings()

            with TestClient(app, raise_server_exceptions=False) as c:
                # No token
                resp = c.post("/chat", json={"question": "test?"})
                assert resp.status_code == 403

                # User token works
                resp = c.post("/chat", json={"question": "test?"}, headers={"Authorization": "Bearer user-secret"})
                assert resp.status_code == 200

                # Admin token also works for user routes
                resp = c.post("/chat", json={"question": "test?"}, headers={"Authorization": "Bearer admin-secret"})
                assert resp.status_code == 200

        deps._backend = None
        deps._llm = None
        deps._reranker = None
        deps._embedding = None

    def test_public_routes_always_open(self):
        app = _create_test_app()
        mock_be = MagicMock(spec=VectorStoreBackend)
        mock_be.count.return_value = 0
        mock_be.get_embedding_provenance.return_value = None
        deps._backend = mock_be

        with patch("src.api.middleware.auth.get_settings") as mock_settings:
            mock_settings.return_value = self._make_auth_settings()

            with TestClient(app, raise_server_exceptions=False) as c:
                # Health is public
                resp = c.get("/health")
                assert resp.status_code == 200

                # Status is public
                with patch("src.api.routes.status.get_settings") as mock_status_settings:
                    mock_status_settings.return_value = self._make_auth_settings()
                    mock_status_settings.return_value.vectorstore.collection_name = "test"
                    mock_status_settings.return_value.app.name = "Test"
                    mock_status_settings.return_value.vectorstore.backend = "chroma"
                    resp = c.get("/status")
                    assert resp.status_code == 200

        deps._backend = None
        deps._reranker = None

    def test_options_preflight_passes_without_token(self):
        app = _create_test_app()
        mock_be = MagicMock(spec=VectorStoreBackend)
        deps._backend = mock_be

        with patch("src.api.middleware.auth.get_settings") as mock_settings:
            mock_settings.return_value = self._make_auth_settings()

            with TestClient(app, raise_server_exceptions=False) as c:
                # OPTIONS on protected routes should not return 403
                resp = c.options("/chat")
                assert resp.status_code != 403

                resp = c.options("/admin/collections")
                assert resp.status_code != 403

        deps._backend = None
        deps._reranker = None
