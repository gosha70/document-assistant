"""Tests for QueryOrchestrator."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.orchestrator import QueryOrchestrator


def _make_settings(**overrides):
    """Create a mock Settings with query_pipeline defaults."""
    qp = MagicMock()
    qp.decomposition_enabled = False
    qp.hyde_enabled = False
    qp.corrective_retrieval_enabled = False
    qp.corrective_retrieval_threshold = 0.4
    qp.verification_enabled = False
    qp.hyde_cache_size = 100
    qp.max_sub_queries = 5
    for k, v in overrides.items():
        setattr(qp, k, v)

    settings = MagicMock()
    settings.query_pipeline = qp
    settings.telemetry.enabled = False
    return settings


def _make_orchestrator(docs=None, gen_result=None, **settings_overrides):
    """Build an orchestrator with mocked retriever/generator."""
    retriever = MagicMock()
    retriever.retrieve.return_value = docs if docs is not None else []
    retriever.retrieve_with_vector.return_value = docs if docs is not None else []
    retriever._reranker = MagicMock()
    retriever._final_k = 5

    generator = MagicMock()
    gen_result = gen_result or {
        "answer": "test answer",
        "sources": [{"file": "a.txt", "page": None, "excerpt": "exc"}],
    }
    generator.generate.return_value = gen_result
    generator.generate_stream.return_value = iter(["test ", "answer"])
    generator.extract_sources.return_value = gen_result["sources"]

    llm = MagicMock()
    embedding = MagicMock()
    embedding.embed_documents.return_value = [[0.1] * 384]
    settings = _make_settings(**settings_overrides)

    orch = QueryOrchestrator(
        retriever=retriever,
        generator=generator,
        llm=llm,
        embedding=embedding,
        settings=settings,
    )
    return orch, retriever, generator


class TestOrchestratorRun:
    def test_run_no_docs_returns_empty(self):
        orch, retriever, generator = _make_orchestrator(docs=[])
        result = orch.run("test query")

        assert result["answer"] == "No relevant documents found for your question."
        assert result["sources"] == []
        assert "metadata" in result
        generator.generate.assert_not_called()

    def test_run_with_docs_delegates(self):
        docs = [Document(page_content="hello", metadata={"source": "a.txt"})]
        orch, retriever, generator = _make_orchestrator(docs=docs)

        result = orch.run("test query")

        retriever.retrieve.assert_called_once_with("test query")
        generator.generate.assert_called_once_with(
            query="test query",
            documents=docs,
            history="",
        )
        assert result["answer"] == "test answer"
        assert len(result["sources"]) == 1
        assert isinstance(result["metadata"], dict)

    def test_run_passes_history(self):
        docs = [Document(page_content="hello", metadata={})]
        orch, retriever, generator = _make_orchestrator(docs=docs)

        orch.run("q", history="prev context")
        generator.generate.assert_called_once_with(
            query="q",
            documents=docs,
            history="prev context",
        )


class TestOrchestratorRunStream:
    def test_stream_no_docs(self):
        orch, _, _ = _make_orchestrator(docs=[])
        documents, token_iter, get_sources, metadata = orch.run_stream("q")

        assert documents is None
        assert list(token_iter) == []
        assert get_sources() == []
        assert isinstance(metadata, dict)

    def test_stream_with_docs(self):
        docs = [Document(page_content="hello", metadata={"source": "a.txt"})]
        orch, retriever, generator = _make_orchestrator(docs=docs)

        documents, token_iter, get_sources, metadata = orch.run_stream("q")

        assert documents == docs
        assert list(token_iter) == ["test ", "answer"]
        sources = get_sources()
        assert len(sources) == 1
        assert isinstance(metadata, dict)


class TestOrchestratorMetrics:
    def test_metrics_recorded_when_telemetry_enabled(self):
        docs = [Document(page_content="hello", metadata={})]
        orch, _, _ = _make_orchestrator(docs=docs)
        orch._settings.telemetry.enabled = True

        with patch("src.rag.orchestrator.get_metrics_collector") as mock_metrics:
            collector = MagicMock()
            mock_metrics.return_value = collector
            orch.run("q")
            collector.record_retrieval.assert_called_once()

    def test_no_source_alert_when_empty(self):
        orch, _, _ = _make_orchestrator(docs=[])
        orch._settings.telemetry.enabled = True
        orch._settings.alerting.window_seconds = 3600

        with patch("src.rag.orchestrator.get_metrics_collector") as mock_metrics:
            collector = MagicMock()
            mock_metrics.return_value = collector
            orch.run("q")
            collector.record_retrieval_no_source.assert_called_once_with(3600)


class TestDecomposition:
    @patch("src.rag.query_transform.get_settings")
    def test_decomposition_calls_sub_queries(self, mock_qt_settings):
        mock_qt_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content="hello", metadata={"id": "1"})]
        orch, retriever, generator = _make_orchestrator(
            docs=docs,
            decomposition_enabled=True,
        )
        # Mock LLM to return decomposed queries
        llm_result = MagicMock()
        llm_result.content = '["What is X?", "What is Y?"]'
        orch._llm.invoke.return_value = llm_result

        result = orch.run("What is X and Y?")

        assert "decomposed_queries" in result["metadata"]
        assert len(result["metadata"]["decomposed_queries"]) == 2
        # Should have called retrieve twice (once per sub-query)
        assert retriever.retrieve.call_count == 2

    @patch("src.rag.query_transform.get_settings")
    def test_single_query_no_decomposition(self, mock_qt_settings):
        mock_qt_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content="hello", metadata={})]
        orch, retriever, _ = _make_orchestrator(
            docs=docs,
            decomposition_enabled=True,
        )
        llm_result = MagicMock()
        llm_result.content = '["What is X?"]'
        orch._llm.invoke.return_value = llm_result

        orch.run("What is X?")
        retriever.retrieve.assert_called_once()

    @patch("src.rag.query_transform.get_settings")
    def test_decomposition_trims_to_final_k_without_reranker(self, mock_qt_settings):
        mock_qt_settings.return_value.telemetry.enabled = False
        # Create more docs than final_k across sub-queries
        docs = [Document(page_content=f"doc{i}", metadata={"id": str(i)}) for i in range(10)]
        orch, retriever, _ = _make_orchestrator(
            docs=docs,
            decomposition_enabled=True,
        )
        # Disable reranker
        orch._retriever._reranker = None
        orch._retriever._final_k = 5

        llm_result = MagicMock()
        llm_result.content = '["What is X?", "What is Y?"]'
        orch._llm.invoke.return_value = llm_result

        orch.run("What is X and Y?")
        # After retrieval from orchestrator, generator receives documents
        # that were passed to generate(). Verify the merged set was trimmed.
        call_args = orch._generator.generate.call_args
        passed_docs = call_args.kwargs.get("documents") or call_args[1].get("documents")
        assert len(passed_docs) <= 5


class TestHyDE:
    @patch("src.rag.query_transform.get_settings")
    def test_hyde_uses_vector_retrieval(self, mock_qt_settings):
        mock_qt_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content="hello", metadata={})]
        orch, retriever, _ = _make_orchestrator(
            docs=docs,
            hyde_enabled=True,
        )
        llm_result = MagicMock()
        llm_result.content = "A hypothetical document about the topic."
        orch._llm.invoke.return_value = llm_result

        result = orch.run("What is X?")

        assert result["metadata"].get("hyde_used") is True
        orch._embedding.embed_documents.assert_called_once()
        retriever.retrieve_with_vector.assert_called_once()
        retriever.retrieve.assert_not_called()


class TestCorrectiveRetrieval:
    @patch("src.rag.query_transform.get_settings")
    def test_no_corrective_when_confidence_high(self, mock_qt_settings):
        mock_qt_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content="hello", metadata={"reranker_score": 8.0})]
        orch, retriever, _ = _make_orchestrator(
            docs=docs,
            corrective_retrieval_enabled=True,
        )

        result = orch.run("What is X?")
        assert "corrective_triggered" not in result["metadata"]

    @patch("src.rag.query_transform.get_settings")
    def test_corrective_triggered_on_low_confidence(self, mock_qt_settings):
        mock_qt_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content="hello", metadata={"reranker_score": -8.0})]
        orch, retriever, _ = _make_orchestrator(
            docs=docs,
            corrective_retrieval_enabled=True,
            corrective_retrieval_threshold=0.4,
        )
        llm_result = MagicMock()
        llm_result.content = "hypothetical doc"
        orch._llm.invoke.return_value = llm_result

        result = orch.run("What is X?")
        assert result["metadata"].get("corrective_triggered") is True
        assert result["metadata"]["retrieval_confidence"] < 0.4

    @patch("src.rag.query_transform.get_settings")
    def test_corrective_trims_to_final_k_without_reranker(self, mock_qt_settings):
        mock_qt_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content=f"doc{i}", metadata={"id": str(i), "reranker_score": -8.0}) for i in range(10)]
        orch, retriever, _ = _make_orchestrator(
            docs=docs,
            corrective_retrieval_enabled=True,
            corrective_retrieval_threshold=0.4,
        )
        orch._retriever._reranker = None
        orch._retriever._final_k = 5

        llm_result = MagicMock()
        llm_result.content = "hypothetical doc"
        orch._llm.invoke.return_value = llm_result

        orch.run("What is X?")
        call_args = orch._generator.generate.call_args
        passed_docs = call_args.kwargs.get("documents") or call_args[1].get("documents")
        assert len(passed_docs) <= 5


class TestVerification:
    @patch("src.rag.verification.get_settings")
    def test_verification_passes_through(self, mock_v_settings):
        mock_v_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content="hello", metadata={})]
        orch, _, _ = _make_orchestrator(
            docs=docs,
            verification_enabled=True,
        )
        llm_result = MagicMock()
        llm_result.content = '{"verified": true, "unsupported_claims": []}'
        orch._llm.invoke.return_value = llm_result

        result = orch.run("What is X?")
        assert result["metadata"]["verification"]["verified"] is True
        assert result["answer"] == "test answer"

    @patch("src.rag.verification.get_settings")
    def test_verification_revises_answer(self, mock_v_settings):
        mock_v_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content="hello", metadata={})]
        orch, _, _ = _make_orchestrator(
            docs=docs,
            verification_enabled=True,
        )
        llm_result = MagicMock()
        llm_result.content = '{"verified": false, "unsupported_claims": ["claim X"]}'
        orch._llm.invoke.return_value = llm_result

        result = orch.run("What is X?")
        assert result["metadata"]["verification"]["verified"] is False
        assert "could not be verified" in result["answer"]

    @patch("src.rag.verification.get_settings")
    def test_stream_verification(self, mock_v_settings):
        mock_v_settings.return_value.telemetry.enabled = False
        docs = [Document(page_content="hello", metadata={})]
        orch, _, generator = _make_orchestrator(
            docs=docs,
            verification_enabled=True,
        )
        generator.generate_stream.return_value = iter(["tok1", "tok2"])
        llm_result = MagicMock()
        llm_result.content = '{"verified": true, "unsupported_claims": []}'
        orch._llm.invoke.return_value = llm_result

        documents, token_iter, get_sources, metadata = orch.run_stream("q")
        tokens = list(token_iter)

        assert tokens == ["tok1", "tok2"]
        assert metadata["verification"]["verified"] is True
