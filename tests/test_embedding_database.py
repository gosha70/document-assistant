"""Tests for embedding_database.py — verifies bug fixes and core functionality."""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call

from embeddings.embedding_database import (
    create_manifest,
    create_vector_store,
    adjust_batch_size,
    create_embedding_database,
    create_embedding_database_from_zip,
)
from embeddings.embeddings_constants import DEFAULT_COLLECTION_NAME, BATCH_SIZE


class TestCreateManifest:
    def test_creates_manifest_file(self, tmp_dir):
        result = create_manifest("test_collection", "test_model", tmp_dir)
        assert result is True
        manifest_path = os.path.join(tmp_dir, "META-INF", "MANIFEST.MF")
        assert os.path.exists(manifest_path)

    def test_manifest_content_has_correct_collection_name(self, tmp_dir):
        """Bug fix: 'Collectio-Name' typo → 'Collection-Name'."""
        create_manifest("my_collection", "my_model", tmp_dir)
        manifest_path = os.path.join(tmp_dir, "META-INF", "MANIFEST.MF")
        with open(manifest_path) as f:
            content = f.read()
        assert "Collection-Name: my_collection" in content
        assert "Collectio-Name" not in content

    def test_manifest_includes_model_name(self, tmp_dir):
        create_manifest("col", "hkunlp/instructor-large", tmp_dir)
        manifest_path = os.path.join(tmp_dir, "META-INF", "MANIFEST.MF")
        with open(manifest_path) as f:
            content = f.read()
        assert "hkunlp/instructor-large" in content

    def test_returns_false_for_none_directory(self):
        result = create_manifest("col", "model", None)
        assert result is False

    def test_defaults_collection_name(self, tmp_dir):
        create_manifest(None, "model", tmp_dir)
        manifest_path = os.path.join(tmp_dir, "META-INF", "MANIFEST.MF")
        with open(manifest_path) as f:
            content = f.read()
        assert f"Collection-Name: {DEFAULT_COLLECTION_NAME}" in content


class TestCreateVectorStore:
    @patch("embeddings.embedding_database.create_embedding_database")
    def test_documents_kwarg_reaches_create_embedding_database(self, mock_create_db):
        """Directly verify that create_vector_store passes documents= correctly."""
        mock_docs = [MagicMock()]
        mock_create_db.return_value = MagicMock()

        # asyncio.run will call the coroutine; mock create_embedding_database
        # to capture what was passed
        import asyncio

        # create_embedding_database is async, mock needs to return a coroutine
        async def fake_create(**kwargs):
            return MagicMock()

        mock_create_db.side_effect = fake_create

        create_vector_store(
            documents=mock_docs,
            model_name="test_model",
            collection_name="test_col",
            persist_directory="/tmp/test",
        )

        mock_create_db.assert_called_once()
        call_kwargs = mock_create_db.call_args[1]
        assert call_kwargs["documents"] is mock_docs
        assert call_kwargs["model_name"] == "test_model"
        assert call_kwargs["collection_name"] == "test_col"
        assert call_kwargs["persist_directory"] == "/tmp/test"
        assert call_kwargs["chunk_size"] == BATCH_SIZE

    def test_returns_none_for_empty_documents(self):
        result = create_vector_store(
            documents=[],
            model_name="test",
            collection_name="test",
            persist_directory="/tmp/test",
        )
        assert result is None

    def test_returns_none_for_none_documents(self):
        result = create_vector_store(
            documents=None,
            model_name="test",
            collection_name="test",
            persist_directory="/tmp/test",
        )
        assert result is None


class TestAdjustBatchSize:
    def test_normal_batch(self):
        result = adjust_batch_size(300, 1000)
        assert result == 300

    def test_limits_concurrent_threads(self):
        # When items/batch > 11, should increase batch_size
        result = adjust_batch_size(10, 200)
        assert result == int(200 / 11)

    def test_small_items(self):
        result = adjust_batch_size(300, 100)
        assert result == 300


class TestCreateEmbeddingDatabaseFromZip:
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_zip(self):
        """Verify early return when zip contains nothing."""
        with patch("embeddings.embedding_database.load_zip_with_splits", return_value=None):
            result = await create_embedding_database_from_zip(
                zip_file="test.zip",
                model_name="test",
                chunk_size=100,
                collection_name="test",
                persist_directory="/tmp/test",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_non_empty_zip_calls_process_files_with_correct_kwargs(self, tmp_dir):
        """Bug fix: zip path was passing splits_directory= and model_name= kwargs
        instead of embedding= and file_paths= to process_files_in_chunks.

        Verify the positive path passes the correct keyword arguments.
        """
        # Create a fake unzipped folder with two files
        unzip_dir = os.path.join(tmp_dir, "unzipped")
        os.makedirs(unzip_dir)
        for name in ["doc1.json", "doc2.json"]:
            with open(os.path.join(unzip_dir, name), "w") as f:
                f.write("{}")

        mock_embedding = MagicMock()

        async def fake_process(**kwargs):
            return MagicMock()

        with (
            patch("embeddings.embedding_database.load_zip_with_splits", return_value=unzip_dir),
            patch("embeddings.embedding_database.ModelInfo.create_embedding", return_value=mock_embedding),
            patch("embeddings.embedding_database.process_files_in_chunks", side_effect=fake_process) as mock_process,
        ):
            await create_embedding_database_from_zip(
                zip_file="test.zip",
                model_name="test_model",
                chunk_size=50,
                collection_name="test_col",
                persist_directory="/tmp/test",
            )

            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            # Must pass embedding=, not model_name=
            assert call_kwargs["embedding"] is mock_embedding
            # Must pass file_paths=, not splits_directory=
            assert "file_paths" in call_kwargs
            assert len(call_kwargs["file_paths"]) == 2
            assert all(fp.endswith(".json") for fp in call_kwargs["file_paths"])
            assert call_kwargs["chunk_size"] == 50
            assert call_kwargs["collection_name"] == "test_col"
            assert call_kwargs["persist_directory"] == "/tmp/test"
            # Must NOT contain the old broken kwargs
            assert "splits_directory" not in call_kwargs
            assert "model_name" not in call_kwargs
