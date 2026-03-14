"""Tests for the RAG interface layer: vectorstore, retrieval, reranking, chunking, generation."""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.rag.vectorstore import VectorStoreBackend
from src.rag.retrieval import Retriever
from src.rag.reranking import NoOpReranker, Reranker
from src.rag.chunking import get_text_splitter, LANGUAGE_MAP


class TestRetriever:
    def _make_docs(self, n):
        return [Document(page_content=f"doc {i}", metadata={"source": f"f{i}.txt"}) for i in range(n)]

    def test_retrieve_uses_hybrid_by_default(self):
        backend = MagicMock(spec=VectorStoreBackend)
        backend.hybrid_search.return_value = self._make_docs(10)
        r = Retriever(backend=backend, collection_name="col")
        docs = r.retrieve("query")
        backend.hybrid_search.assert_called_once()
        assert len(docs) == 5  # default final_k

    def test_retrieve_dense_only(self):
        backend = MagicMock(spec=VectorStoreBackend)
        backend.search.return_value = self._make_docs(3)
        r = Retriever(backend=backend, collection_name="col", use_hybrid=False, final_k=5)
        docs = r.retrieve("query")
        backend.search.assert_called_once()
        assert len(docs) == 3

    def test_reranker_called_when_provided(self):
        backend = MagicMock(spec=VectorStoreBackend)
        backend.hybrid_search.return_value = self._make_docs(20)
        reranker = MagicMock(spec=Reranker)
        reranker.rerank.return_value = self._make_docs(3)

        r = Retriever(backend=backend, collection_name="col", reranker=reranker, initial_k=20, final_k=5)
        docs = r.retrieve("query")
        reranker.rerank.assert_called_once()
        assert len(docs) == 3


class TestNoOpReranker:
    def test_truncates_to_top_k(self):
        docs = [Document(page_content=f"d{i}") for i in range(10)]
        result = NoOpReranker().rerank("q", docs, top_k=3)
        assert len(result) == 3
        assert result[0].page_content == "d0"


class TestChunking:
    def test_default_splitter(self):
        splitter = get_text_splitter()
        assert splitter._chunk_size == 512
        assert splitter._chunk_overlap == 50

    def test_python_splitter_has_language_separators(self):
        splitter = get_text_splitter("py")
        # Python separators include class/def patterns
        assert splitter._separators is not None
        assert len(splitter._separators) > 0

    def test_unknown_extension_uses_default_separators(self):
        splitter = get_text_splitter("xyz")
        # Unknown extensions get default RecursiveCharacterTextSplitter separators
        # (not language-specific ones)
        python_splitter = get_text_splitter("py")
        assert splitter._separators != python_splitter._separators

    def test_all_language_map_entries_produce_splitters(self):
        for ext in LANGUAGE_MAP:
            splitter = get_text_splitter(ext)
            assert splitter._separators is not None, f"No separators for extension '{ext}'"


class TestChromaBackendReadPaths:
    """Verify that read-only operations don't mutate state."""

    def test_get_collection_info_raises_for_nonexistent(self, tmp_dir):
        from src.rag.chroma_backend import ChromaBackend

        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir)

        with pytest.raises(ValueError, match="not found"):
            backend.get_collection_info("nonexistent_collection")

    def test_count_returns_zero_for_nonexistent(self, tmp_dir):
        from src.rag.chroma_backend import ChromaBackend

        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir)
        assert backend.count("nonexistent") == 0

    def test_list_collections_finds_persisted(self, tmp_dir):
        """Collections created outside this process should appear in list_collections."""
        import chromadb
        from src.rag.chroma_backend import CHROMA_SETTINGS

        # Create a collection directly via chromadb client (same settings as ChromaBackend)
        client = chromadb.PersistentClient(path=tmp_dir, settings=CHROMA_SETTINGS)
        col = client.create_collection("preexisting_col")
        col.add(ids=["1"], documents=["test doc"])
        del client

        from src.rag.chroma_backend import ChromaBackend

        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir)
        collections = backend.list_collections()

        names = [c["name"] for c in collections]
        assert "preexisting_col" in names
        # Verify doc count is accurate
        matching = [c for c in collections if c["name"] == "preexisting_col"]
        assert matching[0]["document_count"] == 1
