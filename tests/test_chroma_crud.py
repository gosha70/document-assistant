"""Tests for Chroma vectorstore CRUD operations using a temp directory."""
import os
import logging
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document


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
