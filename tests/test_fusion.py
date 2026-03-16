"""Tests for RRF fusion utility."""

from langchain_core.documents import Document

from src.rag.fusion import rrf_fuse


class TestRRFFuse:
    def test_fuse_disjoint_lists(self):
        dense = [Document(page_content="a", metadata={"id": "1"})]
        sparse = [Document(page_content="b", metadata={"id": "2"})]
        result = rrf_fuse(dense, sparse, k=60)
        assert len(result) == 2

    def test_fuse_overlapping_by_id(self):
        dense = [
            Document(page_content="doc a", metadata={"id": "1"}),
            Document(page_content="doc b", metadata={"id": "2"}),
        ]
        sparse = [
            Document(page_content="doc b", metadata={"id": "2"}),
            Document(page_content="doc c", metadata={"id": "3"}),
        ]
        result = rrf_fuse(dense, sparse, k=60)
        assert len(result) == 3
        # doc with id "2" appears in both lists so should rank highest
        assert result[0].metadata["id"] == "2"

    def test_fuse_respects_top_n(self):
        dense = [Document(page_content=f"d{i}", metadata={"id": f"{i}"}) for i in range(5)]
        sparse = [Document(page_content=f"s{i}", metadata={"id": f"{i+5}"}) for i in range(5)]
        result = rrf_fuse(dense, sparse, k=60, top_n=3)
        assert len(result) == 3

    def test_fuse_identical_content_different_ids_preserved(self):
        dense = [Document(page_content="same text", metadata={"id": "1", "source": "a.txt"})]
        sparse = [Document(page_content="same text", metadata={"id": "2", "source": "b.txt"})]
        result = rrf_fuse(dense, sparse, k=60)
        assert len(result) == 2
        ids = {doc.metadata["id"] for doc in result}
        assert ids == {"1", "2"}

    def test_fuse_falls_back_to_content_without_id(self):
        dense = [Document(page_content="no id doc", metadata={"source": "a.txt"})]
        sparse = [Document(page_content="no id doc", metadata={"source": "b.txt"})]
        result = rrf_fuse(dense, sparse, k=60)
        # Same content, no IDs → merged into one
        assert len(result) == 1

    def test_fuse_empty_lists(self):
        result = rrf_fuse([], [], k=60)
        assert result == []

    def test_fuse_one_empty_list(self):
        dense = [Document(page_content="a", metadata={"id": "1"})]
        result = rrf_fuse(dense, [], k=60)
        assert len(result) == 1
