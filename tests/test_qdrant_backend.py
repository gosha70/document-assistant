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

    def test_store_rejects_lc_ingest_into_legacy_collection(self):
        """LC ingest into an existing no-provenance collection must raise before any writes."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.retrieve.return_value = []  # no provenance sentinel → legacy
        backend = _make_backend(mock_client)

        docs = [
            Document(
                page_content="chunk",
                metadata={
                    "embedding_strategy": "late_chunking",
                    "late_chunking_model": "test-lc-model",
                    "_precomputed_vector": [0.1, 0.2, 0.3, 0.4],
                },
            )
        ]

        with pytest.raises(ValueError, match="no embedding provenance"):
            backend.store(docs, "legacy_col")

        # Validation must fire before any collection mutation
        mock_client.create_collection.assert_not_called()
        mock_client.upsert.assert_not_called()

    def test_store_rejects_lc_docs_missing_model_name(self):
        """LC docs with vectors but no late_chunking_model must raise before any writes."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        backend = _make_backend(mock_client)

        docs = [
            Document(
                page_content="chunk",
                metadata={
                    "embedding_strategy": "late_chunking",
                    # deliberately omit late_chunking_model
                    "_precomputed_vector": [0.1, 0.2, 0.3, 0.4],
                },
            )
        ]

        with pytest.raises(ValueError, match="late_chunking_model"):
            backend.store(docs, "test_col")

        mock_client.create_collection.assert_not_called()
        mock_client.upsert.assert_not_called()

    def test_store_rejects_lc_docs_with_no_precomputed_vectors(self):
        """LC-marked docs where none have _precomputed_vector must raise before any writes."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False  # fresh collection
        backend = _make_backend(mock_client)

        docs = [
            Document(
                page_content="chunk",
                metadata={
                    "embedding_strategy": "late_chunking",
                    "late_chunking_model": "test-lc-model",
                },
            )
        ]

        with pytest.raises(ValueError, match="none have '_precomputed_vector'"):
            backend.store(docs, "test_col")

        mock_client.create_collection.assert_not_called()
        mock_client.upsert.assert_not_called()


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

        assert prov == {"model_name": "my-model", "type": "huggingface", "embedding_strategy": "standard"}

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


class TestIDContract:
    """Verify that search results include metadata["id"] for fusion/dedup."""

    def test_search_returns_id_in_metadata(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []

        mock_point = MagicMock()
        mock_point.id = "abc-123"
        mock_point.payload = {"page_content": "test doc", "metadata": {"source": "a.txt"}}
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_client.query_points.return_value = mock_result

        backend = _make_backend(mock_client)
        results = backend.search("query", "test_col", k=5)

        assert len(results) == 1
        assert results[0].metadata["id"] == "abc-123"

    def test_hybrid_search_returns_id_in_metadata(self):
        """Hybrid falls back to dense for legacy collections; verify ID still present."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _make_collection_info(vector_size=4)
        mock_client.retrieve.return_value = []

        mock_point = MagicMock()
        mock_point.id = "def-456"
        mock_point.payload = {"page_content": "hybrid doc", "metadata": {"source": "b.txt"}}
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_client.query_points.return_value = mock_result

        backend = _make_backend(mock_client)
        results = backend.hybrid_search("query", "test_col", k=5)

        assert len(results) == 1
        assert results[0].metadata["id"] == "def-456"


class TestSparseVocabRoundTrip:
    """Verify the shared-vocabulary design: store docs, read vocab back."""

    def test_store_and_load_vocab(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.retrieve.return_value = []

        emb = _make_embedding_mock()
        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            from src.rag.qdrant_backend import QdrantBackend

            backend = QdrantBackend(
                embedding=emb,
                url="http://localhost:6333",
                hybrid_settings={"enabled": True, "rrf_k": 60},
            )

        docs = [
            Document(page_content="machine learning models", metadata={"source": "a.txt"}),
            Document(page_content="deep neural networks", metadata={"source": "b.txt"}),
        ]
        emb.embed_documents.return_value = [[0.1] * 4, [0.2] * 4]
        backend.store(docs, "hybrid_col")

        # Find the upsert call that wrote the sparse vocab
        from src.rag.qdrant_backend import _SPARSE_VOCAB_POINT_ID

        vocab_calls = [
            call
            for call in mock_client.upsert.call_args_list
            if any(
                getattr(p, "id", None) == _SPARSE_VOCAB_POINT_ID
                for p in call.kwargs.get("points", call.args[0] if call.args else [])
                if hasattr(p, "id")
            )
        ]
        assert len(vocab_calls) >= 1, "Sparse vocab point should be upserted"

        # Extract the vocab from the last upsert
        last_call = vocab_calls[-1]
        points = last_call.kwargs.get("points", last_call.args[0] if last_call.args else [])
        vocab_point = [p for p in points if getattr(p, "id", None) == _SPARSE_VOCAB_POINT_ID][0]
        stored_vocab = vocab_point.payload["vocab"]

        # Verify token mappings are present
        assert "machine" in stored_vocab
        assert "learning" in stored_vocab
        assert "deep" in stored_vocab
        assert "neural" in stored_vocab
        assert "networks" in stored_vocab
        assert "models" in stored_vocab

        # Verify a second instance can reload the vocab and produce the same encodings
        from src.rag.sparse import TokenEncoder

        enc1 = backend._get_token_encoder("hybrid_col")
        enc2 = TokenEncoder(vocab=stored_vocab)

        sv1 = enc1.encode_query_tf("machine learning")
        sv2 = enc2.encode_query_tf("machine learning")
        assert sv1.indices == sv2.indices
        assert sv1.values == sv2.values


class TestEmbeddingTextContract:
    """Verify that metadata['embedding_text'] is used for embedding when present."""

    def test_store_embeds_augmented_text(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        emb = _make_embedding_mock()
        backend = _make_backend(mock_client, embedding=emb)

        docs = [
            Document(
                page_content="original text",
                metadata={"source": "a.txt", "embedding_text": "context\n\noriginal text"},
            )
        ]
        backend.store(docs, "test_col")

        # embed_documents should be called with the augmented text
        emb.embed_documents.assert_called_once_with(["context\n\noriginal text"])

    def test_store_preserves_page_content_in_payload(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        emb = _make_embedding_mock()
        backend = _make_backend(mock_client, embedding=emb)

        docs = [
            Document(
                page_content="original text",
                metadata={"source": "a.txt", "embedding_text": "context\n\noriginal text"},
            )
        ]
        backend.store(docs, "test_col")

        # Find the document upsert call (not the provenance one)
        upsert_calls = mock_client.upsert.call_args_list
        doc_call = [
            c
            for c in upsert_calls
            if any(getattr(p, "payload", {}).get("page_content") == "original text" for p in c.kwargs.get("points", []))
        ]
        assert len(doc_call) == 1

    def test_store_without_embedding_text_uses_page_content(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        emb = _make_embedding_mock()
        backend = _make_backend(mock_client, embedding=emb)

        docs = [Document(page_content="plain text", metadata={"source": "a.txt"})]
        backend.store(docs, "test_col")

        emb.embed_documents.assert_called_once_with(["plain text"])


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
