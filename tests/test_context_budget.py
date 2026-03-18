"""Tests for the context-window token budget guard in generation.py."""

from langchain_core.documents import Document

from src.rag.generation import _truncate_context


def _doc(content: str) -> Document:
    return Document(page_content=content, metadata={})


class TestTruncateContext:
    def test_all_docs_fit_within_budget(self):
        docs = [_doc("abc"), _doc("def"), _doc("ghi")]
        result = _truncate_context(docs, max_chars=100)
        assert result == docs

    def test_truncates_when_total_exceeds_budget(self):
        docs = [_doc("a" * 50), _doc("b" * 50), _doc("c" * 50)]
        result = _truncate_context(docs, max_chars=80)
        assert len(result) == 1
        assert result[0].page_content == "a" * 50

    def test_includes_exactly_fitting_docs(self):
        docs = [_doc("a" * 40), _doc("b" * 40), _doc("c" * 40)]
        result = _truncate_context(docs, max_chars=80)
        assert len(result) == 2

    def test_single_oversized_doc_is_still_returned(self):
        docs = [_doc("x" * 200)]
        result = _truncate_context(docs, max_chars=50)
        assert len(result) == 1
        assert result[0] is docs[0]

    def test_empty_list_returns_empty(self):
        result = _truncate_context([], max_chars=100)
        assert result == []

    def test_exact_budget_boundary_includes_doc(self):
        docs = [_doc("a" * 100), _doc("b" * 1)]
        result = _truncate_context(docs, max_chars=100)
        assert len(result) == 1

    def test_preserves_document_order(self):
        docs = [_doc("first"), _doc("second"), _doc("third")]
        result = _truncate_context(docs, max_chars=1000)
        assert [d.page_content for d in result] == ["first", "second", "third"]
