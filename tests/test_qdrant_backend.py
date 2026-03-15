"""Tests for QdrantBackend with mocked qdrant_client."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.vectorstore import VectorStoreBackend


def _make_embedding_mock(dim=4):
    """Create a mock EmbeddingAdapter that returns fixed-size vectors."""
    emb = MagicMock()
    emb.model_name = "test-model"
    emb.embed_query.return_value = [0.1] * dim
    emb.embed_documents.return_value = [[0.1] * dim]
    return emb


def _make_collection_info(vector_size=4, points_count=0):
    """Create a mock collection info with nested config.params.vectors.size."""
    info = MagicMock()
    info.points_count = points_count
    info.config.params.vectors.size = vector_size
    return info


def _make_backend(mock_client, embedding=None, allow_legacy_collections=True):
    """Create a QdrantBackend with a pre-injected mock client."""
    emb = embedding or _make_embedding_mock()
    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        from src.rag.qdrant_backend import QdrantBackend

        backend = QdrantBackend(
            embedding=emb,
            url="http://localhost:6333",
            allow_legacy_collections=allow_legacy_collections,
        )
    return backend


class TestQdrantBackendInterface:
    def test_implements_vectorstore_backend(self):
        mock_client = MagicMock()
        backend = _make_backend(mock_client)
        assert isinstance(backend, VectorStoreBackend)


class TestStore:
    def test_store_creates_collection_if_missing(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        backend = _make_backend(mock_client)

        docs = [Document(page_content="test", metadata={"source": "a.txt"})]
        ids = backend.store(docs, "test_col")

        mock_client.create_collection.assert_called_once()
        # upsert called twice: once for provenance sentinel, once for documents
        assert mock_client.upsert.call_count == 2
        assert len(ids) == 1

    def test_store_validates_dimension_if_exists(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []  # no provenance sentinel
        backend = _make_backend(mock_client)

        docs = [Document(page_content="test", metadata={})]
        backend.store(docs, "existing_col")

        mock_client.create_collection.assert_not_called()
        # upsert called twice: once for provenance sentinel, once for documents
        assert mock_client.upsert.call_count == 2

    def test_store_rejects_dimension_mismatch(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=768)
        mock_client.retrieve.return_value = []
        backend = _make_backend(mock_client, embedding=_make_embedding_mock(dim=4))

        docs = [Document(page_content="test", metadata={})]
        with pytest.raises(ValueError, match="vector size 768"):
            backend.store(docs, "wrong_dim_col")


class TestSearch:
    def test_search_returns_documents(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []

        mock_point = MagicMock()
        mock_point.payload = {"page_content": "found it", "metadata": {"source": "b.txt"}}
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_client.query_points.return_value = mock_result

        backend = _make_backend(mock_client)
        results = backend.search("query", "test_col", k=3)

        assert len(results) == 1
        assert results[0].page_content == "found it"
        assert results[0].metadata["source"] == "b.txt"
        mock_client.query_points.assert_called_once()

    def test_search_nonexistent_collection_returns_empty(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        backend = _make_backend(mock_client)

        results = backend.search("query", "nope", k=3)

        assert results == []
        mock_client.create_collection.assert_not_called()
        mock_client.query_points.assert_not_called()

    def test_search_rejects_dimension_mismatch(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=768)
        mock_client.retrieve.return_value = []
        backend = _make_backend(mock_client, embedding=_make_embedding_mock(dim=4))

        with pytest.raises(ValueError, match="vector size 768"):
            backend.search("query", "mismatched_col")

    def test_search_validates_dimension_only_once(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result

        backend = _make_backend(mock_client)
        backend.search("q1", "test_col")
        backend.search("q2", "test_col")

        # get_collection called once for validation, not on second search
        assert mock_client.get_collection.call_count == 1

    def test_hybrid_search_falls_back_to_dense(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []

        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result

        backend = _make_backend(mock_client)
        results = backend.hybrid_search("query", "test_col")

        assert results == []
        mock_client.query_points.assert_called_once()

    def test_hybrid_search_nonexistent_returns_empty(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        backend = _make_backend(mock_client)

        results = backend.hybrid_search("query", "nope")

        assert results == []
        mock_client.create_collection.assert_not_called()


class TestDelete:
    def test_delete_calls_client(self):
        mock_client = MagicMock()
        backend = _make_backend(mock_client)

        backend.delete(["id1", "id2"], "test_col")
        mock_client.delete.assert_called_once()


class TestCollectionInfo:
    def test_get_collection_info(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(points_count=42)
        mock_client.retrieve.return_value = []

        backend = _make_backend(mock_client)
        info = backend.get_collection_info("test_col")

        assert info["name"] == "test_col"
        assert info["backend"] == "qdrant"
        assert info["document_count"] == 42
        assert info["embedding_model"] == "unknown"

    def test_get_collection_info_with_provenance(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(points_count=11)
        prov_point = MagicMock()
        prov_point.payload = {"embedding_model_name": "my-model", "embedding_type": "instructor"}

        # Provenance sentinel exists, sparse vocab sentinel does not
        def _retrieve(collection_name, ids, **kwargs):
            if ids == ["00000000-0000-0000-0000-000000000000"]:
                return [prov_point]
            return []

        mock_client.retrieve.side_effect = _retrieve

        backend = _make_backend(mock_client)
        info = backend.get_collection_info("test_col")

        assert info["embedding_model"] == "my-model"
        assert info["document_count"] == 10  # provenance sentinel subtracted

    def test_get_collection_info_not_found(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        backend = _make_backend(mock_client)
        with pytest.raises(ValueError, match="not found"):
            backend.get_collection_info("nope")


class TestListCollections:
    def test_list_collections(self):
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.name = "col_a"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_col]
        mock_client.get_collections.return_value = mock_collections
        mock_client.get_collection.return_value = _make_collection_info(points_count=10)
        mock_client.retrieve.return_value = []

        backend = _make_backend(mock_client)
        cols = backend.list_collections()

        assert len(cols) == 1
        assert cols[0]["name"] == "col_a"
        assert cols[0]["backend"] == "qdrant"
        assert cols[0]["document_count"] == 10


class TestCount:
    def test_count_existing(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(points_count=7)
        mock_client.retrieve.return_value = []  # no provenance sentinel

        backend = _make_backend(mock_client)
        assert backend.count("test_col") == 7

    def test_count_nonexistent(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        backend = _make_backend(mock_client)
        assert backend.count("nope") == 0


class TestProvenance:
    def test_search_filters_out_provenance_sentinel(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []

        sentinel = MagicMock()
        sentinel.payload = {"_provenance": True, "embedding_model_name": "test-model"}
        real_point = MagicMock()
        real_point.payload = {"page_content": "real doc", "metadata": {"source": "c.txt"}}
        mock_result = MagicMock()
        mock_result.points = [sentinel, real_point]
        mock_client.query_points.return_value = mock_result

        backend = _make_backend(mock_client)
        results = backend.search("query", "test_col")

        assert len(results) == 1
        assert results[0].page_content == "real doc"

    def test_validate_rejects_model_mismatch(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        # Provenance says collection was built with a different model
        prov_point = MagicMock()
        prov_point.payload = {
            "embedding_model_name": "other-model",
            "embedding_type": "instructor",
        }
        mock_client.retrieve.return_value = [prov_point]

        backend = _make_backend(mock_client)
        with pytest.raises(ValueError, match="other-model"):
            backend.search("query", "test_col")

    def test_validate_rejects_type_mismatch(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        # Same model name, different type
        prov_point = MagicMock()
        prov_point.payload = {
            "embedding_model_name": "test-model",
            "embedding_type": "instructor",
        }
        mock_client.retrieve.return_value = [prov_point]

        emb = _make_embedding_mock()
        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            from src.rag.qdrant_backend import QdrantBackend

            backend = QdrantBackend(
                embedding=emb,
                url="http://localhost:6333",
                embedding_type="huggingface",
            )
        with pytest.raises(ValueError, match="embedding type 'instructor'"):
            backend.search("query", "test_col")

    def test_get_embedding_provenance_returns_stored(self):
        mock_client = MagicMock()
        prov_point = MagicMock()
        prov_point.payload = {
            "embedding_model_name": "my-model",
            "embedding_type": "huggingface",
        }
        mock_client.retrieve.return_value = [prov_point]

        backend = _make_backend(mock_client)
        prov = backend.get_embedding_provenance("test_col")

        assert prov == {"model_name": "my-model", "type": "huggingface"}

    def test_get_embedding_provenance_returns_none_when_empty(self):
        mock_client = MagicMock()
        mock_client.retrieve.return_value = []

        backend = _make_backend(mock_client)
        assert backend.get_embedding_provenance("test_col") is None

    def test_validate_rejects_missing_provenance_by_default(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []  # no provenance

        backend = _make_backend(mock_client, allow_legacy_collections=False)
        with pytest.raises(ValueError, match="no embedding provenance"):
            backend.search("query", "legacy_col")

    def test_validate_allows_missing_provenance_when_legacy_enabled(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result

        backend = _make_backend(mock_client, allow_legacy_collections=True)
        results = backend.search("query", "legacy_col")
        assert results == []

    def test_document_count_subtracts_sentinel(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(points_count=11)
        prov_point = MagicMock()
        prov_point.payload = {
            "embedding_model_name": "test-model",
            "embedding_type": "instructor",
        }

        # Only provenance sentinel exists, not sparse vocab
        def _retrieve(collection_name, ids, **kwargs):
            if ids == ["00000000-0000-0000-0000-000000000000"]:
                return [prov_point]
            return []

        mock_client.retrieve.side_effect = _retrieve

        backend = _make_backend(mock_client)
        assert backend.count("test_col") == 10

    def test_document_count_no_sentinel_no_subtraction(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(points_count=5)
        mock_client.retrieve.return_value = []

        backend = _make_backend(mock_client, allow_legacy_collections=True)
        assert backend.count("test_col") == 5


class TestBackendFactory:
    def test_qdrant_backend_wired_in_factory(self):
        from src.api.main import _create_backend

        settings = MagicMock()
        settings.vectorstore.backend = "qdrant"
        settings.vectorstore.qdrant_url = "http://localhost:6333"
        settings.vectorstore.qdrant_api_key = None
        settings.vectorstore.qdrant_prefer_grpc = False
        settings.embedding.type = "instructor"
        settings.embedding.model_name = "test-model"
        settings.embedding.device = "cpu"
        settings.embedding.normalize_embeddings = True

        with patch("qdrant_client.QdrantClient"):
            with patch("src.rag.embeddings.InstructorEmbeddingAdapter") as mock_emb_cls:
                mock_emb = MagicMock()
                mock_emb.embed_query.return_value = [0.1] * 4
                mock_emb_cls.return_value = mock_emb
                backend = _create_backend(settings)

        from src.rag.qdrant_backend import QdrantBackend

        assert isinstance(backend, QdrantBackend)

    def test_unknown_backend_error_message_includes_qdrant(self):
        from src.api.main import _create_backend

        settings = MagicMock()
        settings.vectorstore.backend = "unknown_db"
        settings.embedding.type = "instructor"

        with patch("src.rag.embeddings.InstructorEmbeddingAdapter"):
            with pytest.raises(ValueError, match="qdrant"):
                _create_backend(settings)
