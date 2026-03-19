"""Tests for /admin/collections/{name}/chunks and /admin/collections/{name}/sources."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import deps
from src.rag.vectorstore import VectorStoreBackend


# ---------------------------------------------------------------------------
# Test app factory (mirrors test_api.py)
# ---------------------------------------------------------------------------


def _create_test_app():
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
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TelemetryMiddleware)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(status.router)
    app.include_router(admin.router)
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend():
    backend = MagicMock(spec=VectorStoreBackend)
    backend.count.return_value = 5
    backend.list_collections.return_value = []
    backend.get_collection_info.return_value = {
        "name": "test_col",
        "backend": "chroma",
        "document_count": 5,
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


# ---------------------------------------------------------------------------
# /admin/collections/{name}/chunks
# ---------------------------------------------------------------------------


class TestSampleChunksEndpoint:
    def _make_chunks_result(self, chunks):
        return {
            "total_count": len(chunks),
            "chunks": chunks,
        }

    def test_returns_chunks_and_total_count(self, client, mock_backend):
        mock_backend.sample_chunks.return_value = self._make_chunks_result(
            [
                {"id": "c1", "text": "Hello world", "metadata": {"source": "doc.pdf", "page": 1}},
                {"id": "c2", "text": "Machine learning basics", "metadata": {"source": "ml.txt"}},
            ]
        )
        resp = client.get("/admin/collections/test_col/chunks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["collection_name"] == "test_col"
        assert data["total_count"] == 2
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["id"] == "c1"
        assert data["chunks"][0]["text"] == "Hello world"
        assert data["chunks"][0]["metadata"]["source"] == "doc.pdf"

    def test_default_limit_is_10(self, client, mock_backend):
        mock_backend.sample_chunks.return_value = {"total_count": 0, "chunks": []}
        client.get("/admin/collections/test_col/chunks")
        mock_backend.sample_chunks.assert_called_once_with("test_col", limit=10)

    def test_limit_param_forwarded(self, client, mock_backend):
        mock_backend.sample_chunks.return_value = {"total_count": 0, "chunks": []}
        client.get("/admin/collections/test_col/chunks?limit=5")
        mock_backend.sample_chunks.assert_called_once_with("test_col", limit=5)

    def test_offset_param_ignored(self, client, mock_backend):
        # offset is not a supported query param; verify it is silently ignored
        mock_backend.sample_chunks.return_value = {"total_count": 0, "chunks": []}
        resp = client.get("/admin/collections/test_col/chunks?offset=20")
        assert resp.status_code == 200
        mock_backend.sample_chunks.assert_called_once_with("test_col", limit=10)

    def test_limit_capped_at_50(self, client, mock_backend):
        mock_backend.sample_chunks.return_value = {"total_count": 0, "chunks": []}
        client.get("/admin/collections/test_col/chunks?limit=999")
        mock_backend.sample_chunks.assert_called_once_with("test_col", limit=50)

    def test_404_for_nonexistent_collection(self, client, mock_backend):
        mock_backend.sample_chunks.side_effect = ValueError("Collection 'nope' not found")
        resp = client.get("/admin/collections/nope/chunks")
        assert resp.status_code == 404
        assert "nope" in resp.json()["detail"]

    def test_empty_collection_returns_empty_list(self, client, mock_backend):
        mock_backend.sample_chunks.return_value = {"total_count": 0, "chunks": []}
        resp = client.get("/admin/collections/test_col/chunks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 0
        assert data["chunks"] == []

    def test_text_already_truncated_by_backend(self, client, mock_backend):
        # The route passes through whatever text the backend returns;
        # truncation to 500 chars is the backend's responsibility.
        long_text = "x" * 500
        mock_backend.sample_chunks.return_value = self._make_chunks_result(
            [{"id": "c1", "text": long_text, "metadata": {}}]
        )
        resp = client.get("/admin/collections/test_col/chunks")
        assert resp.status_code == 200
        assert resp.json()["chunks"][0]["text"] == long_text


# ---------------------------------------------------------------------------
# /admin/collections/{name}/sources
# ---------------------------------------------------------------------------


class TestListSourcesEndpoint:
    def _make_sources_result(self, sources, truncated=False, scanned=None):
        return {
            "sources": sources,
            "truncated": truncated,
            "scanned_chunks": scanned if scanned is not None else sum(s["chunk_count"] for s in sources),
        }

    def test_returns_sources_and_counts(self, client, mock_backend):
        mock_backend.list_sources.return_value = self._make_sources_result(
            [
                {"filename": "doc.pdf", "chunk_count": 10},
                {"filename": "notes.txt", "chunk_count": 3},
            ]
        )
        resp = client.get("/admin/collections/test_col/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["collection_name"] == "test_col"
        assert len(data["sources"]) == 2
        filenames = [s["filename"] for s in data["sources"]]
        assert "doc.pdf" in filenames
        assert "notes.txt" in filenames

    def test_chunk_counts_correct(self, client, mock_backend):
        mock_backend.list_sources.return_value = self._make_sources_result([{"filename": "a.pdf", "chunk_count": 7}])
        resp = client.get("/admin/collections/test_col/sources")
        assert resp.json()["sources"][0]["chunk_count"] == 7

    def test_404_for_nonexistent_collection(self, client, mock_backend):
        mock_backend.list_sources.side_effect = ValueError("Collection 'ghost' not found")
        resp = client.get("/admin/collections/ghost/sources")
        assert resp.status_code == 404
        assert "ghost" in resp.json()["detail"]

    def test_empty_collection_returns_empty_sources(self, client, mock_backend):
        mock_backend.list_sources.return_value = self._make_sources_result([], scanned=0)
        resp = client.get("/admin/collections/test_col/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == []
        assert data["scanned_chunks"] == 0

    def test_truncated_flag_forwarded(self, client, mock_backend):
        mock_backend.list_sources.return_value = self._make_sources_result(
            [{"filename": "big.pdf", "chunk_count": 10000}],
            truncated=True,
            scanned=10000,
        )
        resp = client.get("/admin/collections/test_col/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["truncated"] is True
        assert data["scanned_chunks"] == 10000

    def test_scanned_chunks_reported(self, client, mock_backend):
        mock_backend.list_sources.return_value = self._make_sources_result(
            [{"filename": "f.pdf", "chunk_count": 42}], scanned=42
        )
        resp = client.get("/admin/collections/test_col/sources")
        assert resp.json()["scanned_chunks"] == 42


# ---------------------------------------------------------------------------
# Regression: existing admin endpoints unaffected
# ---------------------------------------------------------------------------


class TestExistingAdminEndpointsUnchanged:
    def test_list_collections(self, client, mock_backend):
        mock_backend.list_collections.return_value = [
            {
                "name": "col1",
                "backend": "chroma",
                "document_count": 2,
                "persist_directory": None,
                "embedding_model": "m",
            }
        ]
        resp = client.get("/admin/collections")
        assert resp.status_code == 200
        assert resp.json()[0]["name"] == "col1"

    def test_get_collection(self, client, mock_backend):
        resp = client.get("/admin/collections/test_col")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test_col"

    def test_list_jobs(self, client):
        resp = client.get("/admin/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_runtime_metrics(self, client):
        resp = client.get("/admin/metrics/runtime")
        assert resp.status_code == 200

    def test_reports_summary(self, client):
        resp = client.get("/admin/reports/summary")
        assert resp.status_code == 200
