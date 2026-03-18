"""Tests for Chroma vectorstore CRUD operations using a temp directory."""

import os
import shutil
import pytest
from unittest.mock import MagicMock

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
            Document(
                page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "test.txt", "page": 0}
            ),
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


class TestChromaIDContract:
    """Verify that hybrid search results include document IDs on both sides."""

    def test_all_hybrid_results_have_ids(self, tmp_dir):
        """Every document from hybrid_search must have metadata['id']."""
        client = chromadb.PersistentClient(path=tmp_dir, settings=CHROMA_SETTINGS)
        col = client.create_collection("id_test")
        col.add(
            ids=["doc-1", "doc-2"],
            documents=["first document about cats", "second document about dogs"],
            embeddings=[[0.1] * 384, [0.9] * 384],
            metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
        )
        del client

        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        lc_emb = MagicMock()
        lc_emb.embed_query.return_value = [0.1] * 384
        mock_emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir, allow_legacy_collections=True)
        backend._hybrid_enabled = True
        results = backend.hybrid_search("cats", collection_name="id_test", k=2)

        assert len(results) >= 1
        for doc in results:
            assert "id" in doc.metadata, f"Document missing metadata['id']: {doc.metadata}"

    def test_overlapping_dense_and_sparse_hits_fuse(self, tmp_dir):
        """A document appearing in both dense and BM25 results must not duplicate."""
        client = chromadb.PersistentClient(path=tmp_dir, settings=CHROMA_SETTINGS)
        col = client.create_collection("fuse_test")
        # doc-1 has embedding very close to query and contains the query keyword
        # so it should appear in both dense and BM25 results
        col.add(
            ids=["doc-1", "doc-2", "doc-3"],
            documents=[
                "cats are wonderful pets",
                "dogs are loyal companions",
                "birds can fly high",
            ],
            embeddings=[[0.1] * 384, [0.5] * 384, [0.9] * 384],
            metadatas=[{"source": "a.txt"}, {"source": "b.txt"}, {"source": "c.txt"}],
        )
        del client

        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        lc_emb = MagicMock()
        lc_emb.embed_query.return_value = [0.1] * 384  # closest to doc-1
        mock_emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir, allow_legacy_collections=True)
        backend._hybrid_enabled = True
        results = backend.hybrid_search("cats", collection_name="fuse_test", k=3)

        # doc-1 should appear exactly once, not duplicated across dense/sparse
        ids = [doc.metadata["id"] for doc in results]
        assert ids == list(dict.fromkeys(ids)), f"Duplicate IDs in fused results: {ids}"


class TestEmbeddingTextContract:
    """Verify that metadata['embedding_text'] triggers the raw chromadb store path."""

    def test_augmented_docs_use_raw_chromadb_path(self, tmp_dir):
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        lc_emb = MagicMock()
        lc_emb.embed_documents.return_value = [[0.1] * 384]
        mock_emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir)

        docs = [
            Document(
                page_content="original text",
                metadata={"source": "a.txt", "embedding_text": "context\n\noriginal text"},
            )
        ]
        ids = backend.store(docs, collection_name="aug_test")
        assert len(ids) == 1

        # embed_documents should have been called with augmented text
        lc_emb.embed_documents.assert_called_once_with(["context\n\noriginal text"])

        # Stored document text should be the original page_content
        collection = backend._client.get_collection("aug_test")
        results = collection.get(ids=ids, include=["documents"])
        assert results["documents"][0] == "original text"

    def test_non_augmented_docs_use_langchain_path(self, tmp_dir):
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        lc_emb = MagicMock()
        lc_emb.embed_documents.return_value = [[0.1] * 384]
        mock_emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir)

        docs = [Document(page_content="plain text", metadata={"source": "a.txt"})]
        ids = backend.store(docs, collection_name="plain_test")
        assert len(ids) == 1

        # Verify docs were stored successfully via LangChain path
        collection = backend._client.get_collection("plain_test")
        assert collection.count() == 1

    def test_bm25_rebuild_prefers_embedding_text(self, tmp_dir):
        """BM25 rebuild from Chroma should use embedding_text when available."""
        # Store augmented docs via raw chromadb to set up metadata
        client = chromadb.PersistentClient(path=tmp_dir, settings=CHROMA_SETTINGS)
        col = client.create_collection("bm25_rebuild_test")
        col.add(
            ids=["doc-1"],
            documents=["original text"],
            embeddings=[[0.1] * 384],
            metadatas=[{"source": "a.txt", "embedding_text": "context\n\noriginal text"}],
        )
        del client

        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(
            embedding=mock_emb,
            persist_directory=tmp_dir,
            allow_legacy_collections=True,
            hybrid_settings={"enabled": True, "rrf_k": 60},
        )
        bm25 = backend._get_bm25_index("bm25_rebuild_test")

        # BM25 index should have been rebuilt with the augmented text
        assert bm25.size == 1
        assert bm25._doc_texts[0] == "context\n\noriginal text"


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

        backend = ChromaBackend(embedding=mock_emb, persist_directory=tmp_dir, allow_legacy_collections=True)
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

        backend = ChromaBackend(embedding=mock_emb, persist_directory=legacy_dir, allow_legacy_collections=True)
        results = backend.search("programming", collection_name="legacy_test_col", k=1)
        assert len(results) == 1
        assert isinstance(results[0], Document)

    def test_legacy_store_search_blocked_without_legacy_flag(self, legacy_dir):
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        lc_emb = MagicMock()
        lc_emb.embed_query.return_value = [0.1] * 384
        mock_emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=mock_emb, persist_directory=legacy_dir)
        with pytest.raises(ValueError, match="no embedding provenance"):
            backend.search("programming", collection_name="legacy_test_col", k=1)

    def test_legacy_store_blocked_on_write_without_legacy_flag(self, legacy_dir):
        mock_emb = MagicMock()
        mock_emb.model_name = "test"
        mock_emb.get_langchain_embeddings.return_value = MagicMock()

        backend = ChromaBackend(embedding=mock_emb, persist_directory=legacy_dir)
        docs = [Document(page_content="new doc", metadata={"source": "x.txt"})]
        with pytest.raises(ValueError, match="no embedding provenance"):
            backend.store(docs, collection_name="legacy_test_col")
