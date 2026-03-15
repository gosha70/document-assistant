"""Tests for BM25Index — add, search, delete, save/load, rebuild."""

import os
from src.rag.bm25_index import BM25Index


class TestBM25Index:
    def test_add_and_search(self):
        idx = BM25Index()
        idx.add(["d1", "d2", "d3"], ["the quick brown fox", "lazy dog sleeps", "quick fox jumps"])
        results = idx.search("quick fox", k=2)
        assert len(results) == 2
        ids = [doc_id for doc_id, _ in results]
        # "quick fox" should rank d1 or d3 highly
        assert "d1" in ids or "d3" in ids

    def test_search_empty_index(self):
        idx = BM25Index()
        results = idx.search("hello", k=5)
        assert results == []

    def test_delete_removes_documents(self):
        idx = BM25Index()
        idx.add(["d1", "d2", "d3"], ["alpha beta", "gamma delta", "alpha gamma"])
        idx.delete(["d1"])
        assert idx.size == 2
        results = idx.search("alpha", k=3)
        ids = [doc_id for doc_id, _ in results]
        assert "d1" not in ids

    def test_delete_all_leaves_empty(self):
        idx = BM25Index()
        idx.add(["d1", "d2"], ["hello world", "foo bar"])
        idx.delete(["d1", "d2"])
        assert idx.size == 0
        assert idx.search("hello", k=5) == []

    def test_save_and_load(self, tmp_dir):
        pkl_path = os.path.join(tmp_dir, "test_bm25.pkl")
        idx = BM25Index(persist_path=pkl_path)
        idx.add(["d1", "d2"], ["machine learning models", "deep neural networks"])
        idx.save()

        loaded = BM25Index.load(pkl_path)
        assert loaded.size == 2
        results = loaded.search("neural networks", k=2)
        assert len(results) > 0

    def test_save_noop_when_in_memory(self):
        idx = BM25Index(persist_path=None)
        idx.add(["d1"], ["test document"])
        idx.save()  # should not raise
        assert idx.size == 1

    def test_rebuild_replaces_contents(self):
        idx = BM25Index()
        idx.add(["d1"], ["old content"])
        idx.rebuild(["d2", "d3"], ["new content alpha", "new content beta"])
        assert idx.size == 2
        results = idx.search("alpha", k=2)
        ids = [doc_id for doc_id, _ in results]
        assert "d2" in ids
        assert "d1" not in ids

    def test_load_missing_file_returns_empty(self, tmp_dir):
        pkl_path = os.path.join(tmp_dir, "nonexistent.pkl")
        idx = BM25Index.load(pkl_path)
        assert idx.size == 0
