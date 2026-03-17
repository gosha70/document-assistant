"""Tests for query transformation utilities."""

import math
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.query_transform import (
    decompose_query,
    merge_sub_results,
    generate_hypothetical_document,
    compute_retrieval_confidence,
)


class TestDecomposeQuery:
    def _mock_llm(self, response_text):
        llm = MagicMock()
        result = MagicMock()
        result.content = response_text
        llm.invoke.return_value = result
        return llm

    @patch("src.rag.query_transform.get_settings")
    def test_returns_sub_queries(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = self._mock_llm('["What is X?", "What is Y?"]')

        result = decompose_query("What is X and what is Y?", llm)
        assert result == ["What is X?", "What is Y?"]

    @patch("src.rag.query_transform.get_settings")
    def test_simple_query_returns_single(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = self._mock_llm('["What is Python?"]')

        result = decompose_query("What is Python?", llm)
        assert result == ["What is Python?"]

    @patch("src.rag.query_transform.get_settings")
    def test_malformed_json_returns_original(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = self._mock_llm("not valid json")

        result = decompose_query("my query", llm)
        assert result == ["my query"]

    @patch("src.rag.query_transform.get_settings")
    def test_respects_max_sub_queries(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = self._mock_llm('["a", "b", "c", "d", "e", "f"]')

        result = decompose_query("complex", llm, max_sub_queries=3)
        assert len(result) == 3

    @patch("src.rag.query_transform.get_settings")
    def test_empty_list_returns_original(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = self._mock_llm("[]")

        result = decompose_query("my query", llm)
        assert result == ["my query"]


class TestMergeSubResults:
    def test_flattens_and_deduplicates(self):
        doc_a = Document(page_content="alpha", metadata={"id": "1"})
        doc_b = Document(page_content="beta", metadata={"id": "2"})
        doc_a_dup = Document(page_content="alpha", metadata={"id": "1"})

        result = merge_sub_results([[doc_a, doc_b], [doc_a_dup]])
        assert len(result) == 2

    def test_tags_sub_query_index(self):
        doc_a = Document(page_content="alpha", metadata={"id": "1"})
        doc_b = Document(page_content="beta", metadata={"id": "2"})

        result = merge_sub_results([[doc_a], [doc_b]])
        assert result[0].metadata["sub_query_index"] == 0
        assert result[1].metadata["sub_query_index"] == 1


class TestGenerateHypotheticalDocument:
    @patch("src.rag.query_transform.get_settings")
    def test_returns_stripped_text(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = MagicMock()
        result = MagicMock()
        result.content = "  A hypothetical answer passage.  "
        llm.invoke.return_value = result

        text = generate_hypothetical_document("What is X?", llm)
        assert text == "A hypothetical answer passage."


class TestComputeRetrievalConfidence:
    def test_no_scores_returns_one(self):
        docs = [Document(page_content="a", metadata={})]
        assert compute_retrieval_confidence(docs) == 1.0

    def test_high_scores_high_confidence(self):
        docs = [
            Document(page_content="a", metadata={"reranker_score": 8.0}),
            Document(page_content="b", metadata={"reranker_score": 7.0}),
        ]
        conf = compute_retrieval_confidence(docs)
        assert conf > 0.9

    def test_low_scores_low_confidence(self):
        docs = [
            Document(page_content="a", metadata={"reranker_score": -8.0}),
            Document(page_content="b", metadata={"reranker_score": -7.0}),
        ]
        conf = compute_retrieval_confidence(docs)
        assert conf < 0.1

    def test_zero_score_is_half(self):
        docs = [Document(page_content="a", metadata={"reranker_score": 0.0})]
        conf = compute_retrieval_confidence(docs)
        assert abs(conf - 0.5) < 0.01

    def test_empty_docs_returns_one(self):
        assert compute_retrieval_confidence([]) == 1.0
