"""Tests for reranking implementations."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.reranking import Reranker, NoOpReranker, CrossEncoderReranker


class TestNoOpReranker:
    def test_truncates_to_top_k(self):
        docs = [Document(page_content=f"doc{i}", metadata={}) for i in range(10)]
        reranker = NoOpReranker()
        result = reranker.rerank("query", docs, top_k=3)
        assert len(result) == 3
        assert result[0].page_content == "doc0"

    def test_fewer_than_top_k(self):
        docs = [Document(page_content="only one", metadata={})]
        result = NoOpReranker().rerank("query", docs, top_k=5)
        assert len(result) == 1

    def test_empty_docs(self):
        result = NoOpReranker().rerank("query", [], top_k=5)
        assert result == []


class TestCrossEncoderReranker:
    def test_reranks_by_score(self):
        mock_model = MagicMock()
        # Scores: doc0=0.1, doc1=0.9, doc2=0.5
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker(model_name="test-model")

        docs = [
            Document(page_content="low relevance", metadata={"id": 0}),
            Document(page_content="high relevance", metadata={"id": 1}),
            Document(page_content="medium relevance", metadata={"id": 2}),
        ]
        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].metadata["id"] == 1  # highest score
        assert result[1].metadata["id"] == 2  # second highest

    def test_stores_reranker_scores_in_metadata(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker(model_name="test-model")

        docs = [
            Document(page_content="a", metadata={}),
            Document(page_content="b", metadata={}),
            Document(page_content="c", metadata={}),
        ]
        reranker.rerank("query", docs, top_k=3)

        # All docs should have reranker_score in metadata
        assert docs[0].metadata["reranker_score"] == 0.1
        assert docs[1].metadata["reranker_score"] == 0.9
        assert docs[2].metadata["reranker_score"] == 0.5

    def test_rerank_empty_docs(self):
        mock_model = MagicMock()
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker(model_name="test-model")

        result = reranker.rerank("query", [], top_k=5)
        assert result == []
        mock_model.predict.assert_not_called()

    def test_rerank_fewer_than_top_k(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker(model_name="test-model")

        docs = [Document(page_content="only doc", metadata={})]
        result = reranker.rerank("query", docs, top_k=5)
        assert len(result) == 1

    def test_model_name_property(self):
        mock_model = MagicMock()
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_implements_reranker(self):
        mock_model = MagicMock()
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker(model_name="test")
        assert isinstance(reranker, Reranker)


class TestRerankerFactory:
    def test_disabled_returns_none(self):
        from src.api.main import _create_reranker

        settings = MagicMock()
        settings.reranker.enabled = False
        assert _create_reranker(settings) is None

    def test_enabled_creates_cross_encoder(self):
        from src.api.main import _create_reranker

        settings = MagicMock()
        settings.reranker.enabled = True
        settings.reranker.model_name = "test-model"

        with patch("sentence_transformers.CrossEncoder"):
            reranker = _create_reranker(settings)
        assert isinstance(reranker, CrossEncoderReranker)


class TestEmbeddingFactory:
    def test_instructor_type(self):
        from src.api.main import _create_embedding

        settings = MagicMock()
        settings.embedding.type = "instructor"
        settings.embedding.model_name = "test"
        settings.embedding.device = "cpu"
        settings.embedding.normalize_embeddings = True

        with patch("src.rag.embeddings.InstructorEmbeddingAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = _create_embedding(settings)
        mock_cls.assert_called_once()

    def test_huggingface_type(self):
        from src.api.main import _create_embedding

        settings = MagicMock()
        settings.embedding.type = "huggingface"
        settings.embedding.model_name = "test"
        settings.embedding.device = "cpu"
        settings.embedding.normalize_embeddings = True

        with patch("src.rag.embeddings.HuggingFaceEmbeddingAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = _create_embedding(settings)
        mock_cls.assert_called_once()

    def test_unknown_type_raises(self):
        from src.api.main import _create_embedding

        settings = MagicMock()
        settings.embedding.type = "unknown"

        with pytest.raises(ValueError, match="Unknown embedding type"):
            _create_embedding(settings)
