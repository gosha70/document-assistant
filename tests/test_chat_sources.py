"""Tests for richer source citations and orchestrator metadata streaming."""

import json
import pytest
from unittest.mock import MagicMock
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api import deps
from src.api.schemas import SourceCitation
from src.rag.generation import Generator
from src.rag.vectorstore import VectorStoreBackend


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


class TestExtractSources:
    def test_includes_chunk_id_and_search_type(self):
        doc = Document(
            page_content="A" * 300,
            metadata={"source": "file.txt", "page": 2, "id": "abc123def456", "search_type": "hybrid"},
        )
        sources = Generator.extract_sources([doc])
        assert len(sources) == 1
        s = sources[0]
        assert s["file"] == "file.txt"
        assert s["page"] == 2
        assert s["chunk_id"] == "abc123def456"
        assert s["search_type"] == "hybrid"

    def test_excerpt_truncated_to_200(self):
        doc = Document(page_content="X" * 600, metadata={"source": "f.txt"})
        sources = Generator.extract_sources([doc])
        assert len(sources[0]["excerpt"]) == 200

    def test_full_excerpt_truncated_to_500(self):
        doc = Document(page_content="Y" * 600, metadata={"source": "f.txt"})
        sources = Generator.extract_sources([doc])
        assert len(sources[0]["full_excerpt"]) == 500

    def test_missing_metadata_fields_are_none(self):
        doc = Document(page_content="Hello", metadata={"source": "f.txt"})
        sources = Generator.extract_sources([doc])
        assert sources[0]["chunk_id"] is None
        assert sources[0]["search_type"] is None

    def test_empty_documents_list(self):
        assert Generator.extract_sources([]) == []


class TestSourceCitationSchema:
    def test_accepts_new_optional_fields(self):
        citation = SourceCitation(
            file="doc.pdf",
            page=1,
            excerpt="short excerpt",
            chunk_id="abc12345",
            search_type="dense",
            full_excerpt="longer excerpt here",
        )
        assert citation.chunk_id == "abc12345"
        assert citation.search_type == "dense"
        assert citation.full_excerpt == "longer excerpt here"

    def test_new_fields_default_to_none(self):
        citation = SourceCitation(file="doc.pdf", excerpt="text")
        assert citation.chunk_id is None
        assert citation.search_type is None
        assert citation.full_excerpt is None

    def test_backward_compatible_without_new_fields(self):
        citation = SourceCitation(file="doc.pdf", excerpt="text", page=3)
        assert citation.file == "doc.pdf"
        assert citation.page == 3


class TestStreamingMetadataEvent:
    def _parse_sse_events(self, body: str) -> list[dict]:
        events = []
        for line in body.splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events

    def test_metadata_event_emitted_when_orchestrator_features_active(self, client, mock_backend):
        mock_backend.hybrid_search.return_value = [
            Document(page_content="Python is great.", metadata={"source": "intro.txt"}),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Python is a language."
        if hasattr(mock_llm, "stream"):
            del mock_llm.stream
        deps._llm = mock_llm
        deps._embedding = MagicMock()

        # Patch get_orchestrator as imported in the chat route module so the
        # wrapper is in effect when the route handler calls it.
        from src.api import deps as _deps

        original_get = _deps.get_orchestrator

        def patched_get_orchestrator(collection_name, template_type, use_history):
            orch = original_get(collection_name, template_type, use_history)
            original_run_stream = orch.run_stream

            def mock_run_stream(question):
                docs, token_iter, get_sources, metadata = original_run_stream(question)
                metadata["decomposed_queries"] = ["sub-q1", "sub-q2"]
                metadata["hyde_used"] = True
                metadata["retrieval_confidence"] = 0.45
                metadata["corrective_triggered"] = True
                return docs, token_iter, get_sources, metadata

            orch.run_stream = mock_run_stream
            return orch

        import src.api.routes.chat as chat_module

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(chat_module, "get_orchestrator", patched_get_orchestrator)
            resp = client.post("/chat/stream", json={"question": "What is Python?"})

        assert resp.status_code == 200
        events = self._parse_sse_events(resp.text)
        types = [e["type"] for e in events]
        assert "metadata" in types

        meta_event = next(e for e in events if e["type"] == "metadata")
        data = meta_event["data"]
        assert data.get("decomposed_queries") == ["sub-q1", "sub-q2"]
        assert data.get("hyde_used") is True
        assert data.get("retrieval_confidence") == pytest.approx(0.45)
        assert data.get("corrective_triggered") is True

    def test_metadata_event_omitted_when_no_pipeline_features(self, client, mock_backend):
        mock_backend.hybrid_search.return_value = [
            Document(page_content="Python is great.", metadata={"source": "intro.txt"}),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Python is a language."
        if hasattr(mock_llm, "stream"):
            del mock_llm.stream
        deps._llm = mock_llm
        deps._embedding = MagicMock()

        resp = client.post("/chat/stream", json={"question": "What is Python?"})
        assert resp.status_code == 200
        events = self._parse_sse_events(resp.text)
        types = [e["type"] for e in events]
        assert "metadata" not in types
