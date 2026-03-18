"""Tests for the /study endpoints and StudyOutputGenerator."""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.api import deps
from src.rag.study_outputs import StudyOutputGenerator
from src.rag.vectorstore import VectorStoreBackend


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _create_test_app():
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    from fastapi.middleware.cors import CORSMiddleware
    from src.api.middleware.auth import AuthMiddleware
    from src.api.middleware.ratelimit import RateLimitMiddleware
    from src.api.middleware.telemetry import TelemetryMiddleware
    from src.api.routes import chat, ingest, status, admin, study
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
    app.include_router(study.router)
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


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"source": "intro.txt", "page": 1},
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml.txt", "page": 3},
        ),
    ]


# ---------------------------------------------------------------------------
# /study/summarize
# ---------------------------------------------------------------------------


class TestSummarizeEndpoint:
    def test_503_when_llm_not_initialised(self, client):
        resp = client.post("/study/summarize", json={"query": "What is Python?"})
        assert resp.status_code == 503

    def test_404_when_no_documents_found(self, client, mock_backend):
        mock_backend.hybrid_search.return_value = []
        mock_llm = MagicMock()
        deps._llm = mock_llm
        resp = client.post("/study/summarize", json={"query": "What is Python?"})
        assert resp.status_code == 404
        assert "No relevant documents" in resp.json()["detail"]

    def test_returns_summary_and_sources(self, client, mock_backend, sample_docs):
        mock_backend.hybrid_search.return_value = sample_docs
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Python is a high-level language [Source: intro.txt, page 1]."
        deps._llm = mock_llm

        resp = client.post("/study/summarize", json={"query": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "Python" in data["summary"]
        assert "sources" in data
        assert len(data["sources"]) == 2
        assert data["sources"][0]["file"] == "intro.txt"
        assert data["sources"][0]["page"] == 1

    def test_empty_query_rejected(self, client):
        resp = client.post("/study/summarize", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_rejected(self, client):
        resp = client.post("/study/summarize", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /study/glossary
# ---------------------------------------------------------------------------


class TestGlossaryEndpoint:
    def test_503_when_llm_not_initialised(self, client):
        resp = client.post("/study/glossary", json={"query": "What is Python?"})
        assert resp.status_code == 503

    def test_404_when_no_documents_found(self, client, mock_backend):
        mock_backend.hybrid_search.return_value = []
        deps._llm = MagicMock()
        resp = client.post("/study/glossary", json={"query": "Python"})
        assert resp.status_code == 404

    def test_returns_glossary_terms_and_sources(self, client, mock_backend, sample_docs):
        mock_backend.hybrid_search.return_value = sample_docs
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '[{"term": "Python", "definition": "A high-level language.", "source": "[Source: intro.txt, page 1]"}]'
        )
        deps._llm = mock_llm

        resp = client.post("/study/glossary", json={"query": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        assert "terms" in data
        assert "sources" in data
        assert len(data["terms"]) == 1
        term = data["terms"][0]
        assert term["term"] == "Python"
        assert "high-level" in term["definition"]
        assert "[Source:" in term["source"]

    def test_skips_items_missing_required_fields(self, client, mock_backend, sample_docs):
        mock_backend.hybrid_search.return_value = sample_docs
        mock_llm = MagicMock()
        # One item missing 'definition', one valid
        mock_llm.invoke.return_value = (
            '[{"term": "Python"}, '
            '{"term": "ML", "definition": "Machine learning.", "source": "[Source: ml.txt, page 3]"}]'
        )
        deps._llm = mock_llm

        resp = client.post("/study/glossary", json={"query": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        # Only the valid item should appear
        assert len(data["terms"]) == 1
        assert data["terms"][0]["term"] == "ML"

    def test_handles_invalid_llm_json(self, client, mock_backend, sample_docs):
        mock_backend.hybrid_search.return_value = sample_docs
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Sorry, I cannot extract terms."
        deps._llm = mock_llm

        resp = client.post("/study/glossary", json={"query": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["terms"] == []
        assert len(data["sources"]) == 2


# ---------------------------------------------------------------------------
# /study/flashcards
# ---------------------------------------------------------------------------


class TestFlashcardsEndpoint:
    def test_503_when_llm_not_initialised(self, client):
        resp = client.post("/study/flashcards", json={"query": "What is Python?"})
        assert resp.status_code == 503

    def test_404_when_no_documents_found(self, client, mock_backend):
        mock_backend.hybrid_search.return_value = []
        deps._llm = MagicMock()
        resp = client.post("/study/flashcards", json={"query": "Python"})
        assert resp.status_code == 404

    def test_returns_flashcards_and_sources(self, client, mock_backend, sample_docs):
        mock_backend.hybrid_search.return_value = sample_docs
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '[{"front": "What is Python?", "back": "A high-level language.", "source": "[Source: intro.txt, page 1]"}]'
        )
        deps._llm = mock_llm

        resp = client.post("/study/flashcards", json={"query": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        assert "cards" in data
        assert "sources" in data
        assert len(data["cards"]) == 1
        card = data["cards"][0]
        assert card["front"] == "What is Python?"
        assert "high-level" in card["back"]
        assert "[Source:" in card["source"]

    def test_max_cards_field_accepted(self, client, mock_backend, sample_docs):
        mock_backend.hybrid_search.return_value = sample_docs
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "[]"
        deps._llm = mock_llm

        resp = client.post("/study/flashcards", json={"query": "Python", "max_cards": 5})
        assert resp.status_code == 200

    def test_k_is_forwarded_to_retriever(self, client, mock_backend, sample_docs):
        mock_backend.hybrid_search.return_value = sample_docs
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "[]"
        deps._llm = mock_llm

        with patch("src.api.routes.study.get_retriever") as mock_get_retriever:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = sample_docs
            mock_get_retriever.return_value = mock_retriever

            client.post("/study/flashcards", json={"query": "Python", "k": 7})
            mock_get_retriever.assert_called_once_with(None, k=7)

    def test_max_cards_out_of_range_rejected(self, client):
        deps._llm = MagicMock()
        resp = client.post("/study/flashcards", json={"query": "Python", "max_cards": 0})
        assert resp.status_code == 422

        resp = client.post("/study/flashcards", json={"query": "Python", "max_cards": 51})
        assert resp.status_code == 422

    def test_skips_cards_missing_required_fields(self, client, mock_backend, sample_docs):
        mock_backend.hybrid_search.return_value = sample_docs
        mock_llm = MagicMock()
        # One card missing 'back', one valid
        mock_llm.invoke.return_value = (
            '[{"front": "Q1"}, ' '{"front": "Q2", "back": "A2", "source": "[Source: ml.txt, page 3]"}]'
        )
        deps._llm = mock_llm

        resp = client.post("/study/flashcards", json={"query": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["cards"]) == 1
        assert data["cards"][0]["front"] == "Q2"


# ---------------------------------------------------------------------------
# StudyOutputGenerator unit tests
# ---------------------------------------------------------------------------


class TestParseJsonList:
    def test_valid_json_array(self):
        text = '[{"term": "A", "definition": "B"}]'
        result = StudyOutputGenerator._parse_json_list(text)
        assert result == [{"term": "A", "definition": "B"}]

    def test_json_array_embedded_in_text(self):
        text = 'Here is the result:\n[{"front": "Q", "back": "A"}]\nEnd.'
        result = StudyOutputGenerator._parse_json_list(text)
        assert result == [{"front": "Q", "back": "A"}]

    def test_empty_array(self):
        assert StudyOutputGenerator._parse_json_list("[]") == []

    def test_invalid_text_returns_empty_list(self):
        assert StudyOutputGenerator._parse_json_list("Not JSON at all.") == []

    def test_json_object_not_list_returns_empty(self):
        assert StudyOutputGenerator._parse_json_list('{"term": "A"}') == []

    def test_whitespace_stripped(self):
        text = '  [{"a": 1}]  '
        result = StudyOutputGenerator._parse_json_list(text)
        assert result == [{"a": 1}]

    def test_multiline_json_embedded(self):
        text = 'prefix\n[\n  {"term": "T", "definition": "D"}\n]\nsuffix'
        result = StudyOutputGenerator._parse_json_list(text)
        assert len(result) == 1
        assert result[0]["term"] == "T"


class TestStudyOutputGeneratorInvoke:
    def test_summarize_calls_llm_and_returns_text(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Summary text."
        gen = StudyOutputGenerator(mock_llm)
        docs = [Document(page_content="Hello world", metadata={"source": "f.txt"})]

        result = gen.summarize(docs)

        mock_llm.invoke.assert_called_once()
        assert result == "Summary text."

    def test_summarize_llm_returns_content_attribute(self):
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "Content attribute value."
        mock_llm.invoke.return_value = response
        gen = StudyOutputGenerator(mock_llm)
        docs = [Document(page_content="Text", metadata={"source": "f.txt"})]

        result = gen.summarize(docs)
        assert result == "Content attribute value."

    def test_extract_glossary_parses_json(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '[{"term": "T", "definition": "D", "source": "S"}]'
        gen = StudyOutputGenerator(mock_llm)
        docs = [Document(page_content="Text", metadata={"source": "f.txt"})]

        result = gen.extract_glossary(docs)
        assert result == [{"term": "T", "definition": "D", "source": "S"}]

    def test_generate_flashcards_passes_max_cards(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "[]"
        gen = StudyOutputGenerator(mock_llm)
        docs = [Document(page_content="Text", metadata={"source": "f.txt"})]

        gen.generate_flashcards(docs, max_cards=7)

        call_args = mock_llm.invoke.call_args[0][0]
        assert "7" in call_args

    def test_build_context_includes_source_labels(self):
        docs = [
            Document(page_content="First.", metadata={"source": "doc.pdf", "page": 1}),
            Document(page_content="Second.", metadata={"source": "doc.pdf", "page": 2}),
        ]
        context = StudyOutputGenerator._build_context(docs)
        assert "[Source: doc.pdf, page 1]\nFirst." in context
        assert "[Source: doc.pdf, page 2]\nSecond." in context

    def test_build_context_source_without_page(self):
        docs = [Document(page_content="Text.", metadata={"source": "notes.txt"})]
        context = StudyOutputGenerator._build_context(docs)
        assert context == "[Source: notes.txt]\nText."

    def test_build_context_unknown_source_when_no_metadata(self):
        docs = [Document(page_content="Content.", metadata={})]
        context = StudyOutputGenerator._build_context(docs)
        assert "[Source: unknown]" in context
        assert "Content." in context

    def test_build_context_empty_list(self):
        assert StudyOutputGenerator._build_context([]) == ""


# ---------------------------------------------------------------------------
# get_retriever initial_k scaling
# ---------------------------------------------------------------------------


class TestGetRetrieverKScaling:
    """get_retriever must ensure initial_k >= final_k so the backend is queried
    for enough candidates to satisfy the requested k."""

    def _make_retriever(self, mock_backend, k):
        from src.api.deps import get_retriever

        deps._backend = mock_backend
        deps._reranker = None
        return get_retriever(k=k)

    def test_large_k_scales_initial_k(self, mock_backend):
        r = self._make_retriever(mock_backend, k=30)
        assert r._final_k == 30
        assert r._initial_k >= 30

    def test_k_at_boundary_scales_initial_k(self, mock_backend):
        r = self._make_retriever(mock_backend, k=50)
        assert r._final_k == 50
        assert r._initial_k >= 50

    def test_small_k_keeps_default_initial_k(self, mock_backend):
        r = self._make_retriever(mock_backend, k=5)
        assert r._final_k == 5
        assert r._initial_k == 20  # default, unchanged

    def test_no_k_uses_settings_defaults(self, mock_backend):
        from src.config.settings import get_settings

        r = self._make_retriever(mock_backend, k=None)
        settings = get_settings()
        assert r._final_k == settings.reranker.top_k
        assert r._initial_k >= r._final_k
