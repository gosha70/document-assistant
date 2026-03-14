"""Tests for Chroma vectorstore CRUD operations using a temp directory."""
import os
import logging
import shutil
import pytest
from unittest.mock import patch, MagicMock

import chromadb
from langchain_core.documents import Document

from src.rag.chroma_backend import ChromaBackend, CHROMA_SETTINGS

LEGACY_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "legacy_chroma_0_5")


class TestChromaCRUD:
    """Test Chroma CRUD with mocked embeddings to avoid downloading models."""

    @pytest.fixture
    def mock_embedding(self):
        emb = MagicMock()
        emb.embed_documents.return_value = [[0.1] * 384]
        emb.embed_query.return_value = [0.1] * 384
        return emb

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "test.txt", "page": 0}),
            Document(page_content="Python is a programming language.", metadata={"source": "test2.txt", "page": 0}),
        ]

    def test_create_and_query_chroma(self, tmp_dir, mock_embedding, sample_docs):
        from langchain_chroma import Chroma

        mock_embedding.embed_documents.return_value = [[0.1 * i] * 384 for i in range(len(sample_docs))]

        db = Chroma.from_documents(
            documents=sample_docs,
            embedding=mock_embedding,
            persist_directory=tmp_dir,
            collection_name="test_collection",
        )

        ids = db.get()["ids"]
        assert len(ids) == 2

    def test_add_documents_to_existing(self, tmp_dir, mock_embedding, sample_docs):
        from langchain_chroma import Chroma

        mock_embedding.embed_documents.return_value = [[0.1] * 384 for _ in sample_docs]

        db = Chroma.from_documents(
            documents=sample_docs,
            embedding=mock_embedding,
            persist_directory=tmp_dir,
            collection_name="test_add",
        )

        new_doc = Document(page_content="New document content.", metadata={"source": "new.txt"})
        mock_embedding.embed_documents.return_value = [[0.2] * 384]
        db.add_documents([new_doc])

        ids = db.get()["ids"]
        assert len(ids) == 3

    def test_similarity_search(self, tmp_dir, mock_embedding, sample_docs):
        from langchain_chroma import Chroma

        mock_embedding.embed_documents.return_value = [[float(i) * 0.1] * 384 for i in range(len(sample_docs))]

        db = Chroma.from_documents(
            documents=sample_docs,
            embedding=mock_embedding,
            persist_directory=tmp_dir,
            collection_name="test_search",
        )

        results = db.similarity_search("fox", k=1)
        assert len(results) >= 1
        assert hasattr(results[0], "page_content")

    def test_load_persisted_store(self, tmp_dir, mock_embedding, sample_docs):
        from langchain_chroma import Chroma

        mock_embedding.embed_documents.return_value = [[0.1] * 384 for _ in sample_docs]

        Chroma.from_documents(
            documents=sample_docs,
            embedding=mock_embedding,
            persist_directory=tmp_dir,
            collection_name="test_persist",
        )

        # Reload from disk
        db2 = Chroma(
            persist_directory=tmp_dir,
            collection_name="test_persist",
            embedding_function=mock_embedding,
        )

        ids = db2.get()["ids"]
        assert len(ids) == 2

    def test_delete_documents(self, tmp_dir, mock_embedding, sample_docs):
        from langchain_chroma import Chroma

        mock_embedding.embed_documents.return_value = [[0.1] * 384 for _ in sample_docs]

        db = Chroma.from_documents(
            documents=sample_docs,
            embedding=mock_embedding,
            persist_directory=tmp_dir,
            collection_name="test_delete",
        )

        ids = db.get()["ids"]
        assert len(ids) == 2

        db.delete(ids=[ids[0]])
        remaining = db.get()["ids"]
        assert len(remaining) == 1


class TestCrossVersionPersistence:
    """Verify that stores created outside the current stack can be loaded."""

    def test_raw_chromadb_store_queryable_via_backend(self, tmp_dir):
        """Simulate a pre-upgrade store: create with raw chromadb, query via ChromaBackend."""
        # Phase 1: create a collection directly via chromadb (as the old stack would)
        client = chromadb.PersistentClient(path=tmp_dir, settings=CHROMA_SETTINGS)
        col = client.create_collection("legacy_col")
        col.add(
            ids=["doc1", "doc2"],
            documents=["The fox jumped over the fence.", "Python is great."],
            embeddings=[[0.1] * 384, [0.2] * 384],
            metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
        )
        del client

        # Phase 2: open the same directory via ChromaBackend (post-upgrade path)
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir)

        # Verify read-only operations work
        assert backend.count("legacy_col") == 2
        info = backend.get_collection_info("legacy_col")
        assert info["document_count"] == 2
        assert info["name"] == "legacy_col"

        collections = backend.list_collections()
        names = [c["name"] for c in collections]
        assert "legacy_col" in names

    def test_raw_chromadb_store_search_returns_documents(self, tmp_dir):
        """Verify that search against a pre-existing store returns Document objects."""
        client = chromadb.PersistentClient(path=tmp_dir, settings=CHROMA_SETTINGS)
        col = client.create_collection("search_legacy")
        col.add(
            ids=["d1"],
            documents=["LangChain is a framework for LLM applications."],
            embeddings=[[0.5] * 384],
            metadatas=[{"source": "intro.txt", "page": 1}],
        )
        del client

        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        lc_emb = MagicMock()
        lc_emb.embed_query.return_value = [0.5] * 384
        mock_emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir)
        results = backend.search("LLM framework", collection_name="search_legacy", k=1)

        assert len(results) == 1
        assert "LangChain" in results[0].page_content


class TestLegacyChroma05Migration:
    """Test loading a real Chroma 0.5.23 fixture with the upgraded stack.

    The fixture at tests/fixtures/legacy_chroma_0_5/ was generated by chromadb 0.5.23
    before the Phase 3 upgrade. It contains a 'legacy_test_col' collection with 3
    documents and 384-dim embeddings.
    """

    @pytest.fixture
    def legacy_dir(self, tmp_dir):
        """Copy the checked-in fixture to a temp dir so tests don't mutate it."""
        dest = os.path.join(tmp_dir, "legacy_chroma_0_5")
        shutil.copytree(LEGACY_FIXTURE_DIR, dest)
        return dest

    def test_legacy_store_count(self, legacy_dir):
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(embedding=mock_emb, persist_directory=legacy_dir)
        assert backend.count("legacy_test_col") == 3

    def test_legacy_store_collection_info(self, legacy_dir):
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(embedding=mock_emb, persist_directory=legacy_dir)
        info = backend.get_collection_info("legacy_test_col")
        assert info["name"] == "legacy_test_col"
        assert info["document_count"] == 3

    def test_legacy_store_list_collections(self, legacy_dir):
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(embedding=mock_emb, persist_directory=legacy_dir)
        collections = backend.list_collections()
        names = [c["name"] for c in collections]
        assert "legacy_test_col" in names

    def test_legacy_store_search(self, legacy_dir):
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        lc_emb = MagicMock()
        lc_emb.embed_query.return_value = [0.1] * 384
        mock_emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=mock_emb, persist_directory=legacy_dir)
        results = backend.search("programming", collection_name="legacy_test_col", k=1)
        assert len(results) == 1
        assert isinstance(results[0], Document)
