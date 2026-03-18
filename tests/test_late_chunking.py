"""Tests for the late chunking embedding module."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import numpy as np


class TestLateChunkingEmbedder:
    """Unit tests for LateChunkingEmbedder with mocked model."""

    def _make_embedder(self):
        """Create a LateChunkingEmbedder with mocked sentence-transformers model."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 8
            mock_model.tokenizer = MagicMock()
            mock_st.return_value = mock_model

            from src.rag.late_chunking import LateChunkingEmbedder

            embedder = LateChunkingEmbedder(
                model_name="test-model",
                max_context_tokens=512,
                pooling_strategy="mean",
                device="cpu",
            )
            return embedder

    def test_compute_chunk_token_ranges_basic(self):
        embedder = self._make_embedder()

        full_text = "Hello world. This is a test document."
        chunks = [
            Document(page_content="Hello world.", metadata={}),
            Document(page_content="This is a test document.", metadata={}),
        ]
        # offset_mapping: list of (char_start, char_end) per token
        # Simulate: [CLS]=>(0,0), Hello=>(0,5), _world=>(5,11), .=>(11,12),
        #           _This=>(13,17), _is=>(17,20), _a=>(20,22), _test=>(22,27),
        #           _document=>(27,36), .=>(36,36 — but we'll use 35,36), [SEP]=>(0,0)
        offset_mapping = [
            (0, 0),  # CLS
            (0, 5),  # Hello
            (5, 11),  # _world
            (11, 12),  # .
            (13, 17),  # _This
            (17, 20),  # _is
            (20, 22),  # _a
            (22, 27),  # _test
            (27, 35),  # _document
            (35, 36),  # .
            (0, 0),  # SEP
        ]

        ranges = embedder.compute_chunk_token_ranges(full_text, chunks, offset_mapping)

        assert len(ranges) == 2
        assert ranges[0] is not None
        assert ranges[1] is not None
        # First chunk "Hello world." spans tokens 1-3
        assert ranges[0] == (1, 4)
        # Second chunk "This is a test document." spans tokens 4-10
        assert ranges[1] == (4, 10)

    def test_compute_chunk_token_ranges_missing_chunk(self):
        embedder = self._make_embedder()

        full_text = "Some text here."
        chunks = [
            Document(page_content="not in the text at all", metadata={}),
        ]
        offset_mapping = [(0, 0), (0, 4), (4, 9), (9, 14), (14, 15), (0, 0)]

        ranges = embedder.compute_chunk_token_ranges(full_text, chunks, offset_mapping)

        assert len(ranges) == 1
        assert ranges[0] is None

    def test_pool_tokens_mean(self):
        embedder = self._make_embedder()

        token_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pooled = embedder._pool_tokens(token_embeddings, 0, 2)
        expected = np.array([0.5, 0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(pooled, expected)

    def test_pool_tokens_cls(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 4
            mock_model.tokenizer = MagicMock()
            mock_st.return_value = mock_model

            from src.rag.late_chunking import LateChunkingEmbedder

            embedder = LateChunkingEmbedder(
                model_name="test-model",
                max_context_tokens=512,
                pooling_strategy="cls",
                device="cpu",
            )

        token_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        pooled = embedder._pool_tokens(token_embeddings, 0, 2)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(pooled, expected)

    def test_pool_tokens_empty_range(self):
        embedder = self._make_embedder()

        token_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        pooled = embedder._pool_tokens(token_embeddings, 0, 0)
        assert np.allclose(pooled, np.zeros(4))


class TestLateChunkingMetadata:
    """Test that embed_document_chunks sets the right metadata."""

    def test_metadata_set_on_chunks(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 4
            mock_model.tokenizer = MagicMock()
            mock_st.return_value = mock_model

            from src.rag.late_chunking import LateChunkingEmbedder

            embedder = LateChunkingEmbedder(
                model_name="test-model",
                max_context_tokens=512,
                pooling_strategy="mean",
                device="cpu",
            )

        chunks = [
            Document(page_content="chunk one", metadata={"source": "a.txt"}),
            Document(page_content="chunk two", metadata={"source": "a.txt"}),
        ]

        # Mock embed_chunks to return vectors
        embedder.embed_chunks = MagicMock(
            return_value=[  # type: ignore[method-assign]
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ]
        )

        # Mock count_tokens to fit within context window
        with patch("src.rag.chunking.count_tokens", return_value=100):
            result = embedder.embed_document_chunks("chunk one chunk two", chunks)

        assert len(result) == 2
        for chunk in result:
            assert chunk.metadata["embedding_strategy"] == "late_chunking"
            assert chunk.metadata["late_chunking_model"] == "test-model"
            assert "_precomputed_vector" in chunk.metadata

        assert result[0].metadata["_precomputed_vector"] == [0.1, 0.2, 0.3, 0.4]
        assert result[1].metadata["_precomputed_vector"] == [0.5, 0.6, 0.7, 0.8]

    def test_metadata_without_vector_when_chunk_not_found(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 4
            mock_model.tokenizer = MagicMock()
            mock_st.return_value = mock_model

            from src.rag.late_chunking import LateChunkingEmbedder

            embedder = LateChunkingEmbedder(
                model_name="test-model",
                max_context_tokens=512,
                pooling_strategy="mean",
                device="cpu",
            )

        chunks = [
            Document(page_content="chunk one", metadata={"source": "a.txt"}),
        ]

        # Return None for unfound chunk
        embedder.embed_chunks = MagicMock(return_value=[None])  # type: ignore[method-assign]

        with patch("src.rag.chunking.count_tokens", return_value=100):
            result = embedder.embed_document_chunks("chunk one", chunks)

        assert result[0].metadata["embedding_strategy"] == "late_chunking"
        assert "_precomputed_vector" not in result[0].metadata


class TestLateChunkingMultiPass:
    """Verify multi-pass embed_document_chunks groups chunks by segment."""

    def test_embed_chunks_called_once_per_segment(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 4
            mock_model.tokenizer = MagicMock()
            mock_st.return_value = mock_model

            from src.rag.late_chunking import LateChunkingEmbedder

            embedder = LateChunkingEmbedder(
                model_name="test-model",
                max_context_tokens=512,
                pooling_strategy="mean",
                device="cpu",
            )

        segment1 = "alpha text for segment one bravo"
        segment2 = "gamma text for segment two delta"

        chunks = [
            Document(page_content="alpha text for segment one", metadata={}),
            Document(page_content="bravo", metadata={}),  # also in segment1
            Document(page_content="gamma text for segment two", metadata={}),
        ]

        call_log: list[tuple[str, int]] = []

        def mock_embed_chunks(text: str, cks):
            call_log.append((text, len(cks)))
            return [[0.1, 0.2, 0.3, 0.4]] * len(cks)

        embedder.embed_chunks = mock_embed_chunks  # type: ignore[method-assign]

        with patch("src.rag.chunking.count_tokens", return_value=10_000):
            with patch.object(embedder, "_split_into_segments", return_value=[segment1, segment2]):
                result = embedder.embed_document_chunks("full text", chunks, batch_by="page")

        # Two segments → two calls (not three calls — one per chunk)
        assert len(call_log) == 2
        # First call covers both chunks that belong to segment1
        assert call_log[0][1] == 2
        # Second call covers the single chunk in segment2
        assert call_log[1][1] == 1
        # All chunks receive a precomputed vector
        assert all(c.metadata.get("_precomputed_vector") is not None for c in result)


class TestLateChunkingBackendIntegration:
    """Test that backends handle _precomputed_vector correctly."""

    def test_chroma_store_with_precomputed_vectors(self, tmp_dir):
        from src.rag.chroma_backend import ChromaBackend

        emb = MagicMock()
        emb.model_name = "test-model"
        lc_emb = MagicMock()
        emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=emb, persist_directory=tmp_dir)

        docs = [
            Document(
                page_content="chunk one",
                metadata={
                    "source": "a.txt",
                    "embedding_strategy": "late_chunking",
                    "late_chunking_model": "test-lc-model",
                    "_precomputed_vector": [0.1, 0.2, 0.3, 0.4],
                },
            ),
            Document(
                page_content="chunk two",
                metadata={
                    "source": "a.txt",
                    "embedding_strategy": "late_chunking",
                    "late_chunking_model": "test-lc-model",
                    "_precomputed_vector": [0.5, 0.6, 0.7, 0.8],
                },
            ),
        ]

        ids = backend.store(docs, "test_late_chunking")

        assert len(ids) == 2
        # Embedding adapter should NOT have been called for embedding
        lc_emb.embed_documents.assert_not_called()

        # Verify _precomputed_vector was stripped from stored metadata
        collection = backend._client.get_collection("test_late_chunking")
        results = collection.get(ids=ids, include=["metadatas"])
        for meta in results["metadatas"]:
            assert "_precomputed_vector" not in meta
            assert meta.get("embedding_strategy") == "late_chunking"

    def test_chroma_store_mixed_precomputed_and_standard(self, tmp_dir):
        """LC chunks without a precomputed vector must fall back to the LC model, not the standard adapter."""
        from src.rag.chroma_backend import ChromaBackend

        emb = MagicMock()
        emb.model_name = "test-model"
        lc_emb = MagicMock()
        emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=emb, persist_directory=tmp_dir)

        docs = [
            Document(
                page_content="chunk one",
                metadata={
                    "source": "a.txt",
                    "embedding_strategy": "late_chunking",
                    "late_chunking_model": "test-lc-model",
                    "_precomputed_vector": [0.1, 0.2, 0.3, 0.4],
                },
            ),
            Document(
                page_content="chunk two - no vector",
                metadata={
                    "source": "a.txt",
                    "embedding_strategy": "late_chunking",
                    "late_chunking_model": "test-lc-model",
                },
            ),
        ]

        with patch.object(backend, "_get_lc_query_embedding", return_value=[0.9, 0.8, 0.7, 0.6]) as mock_lc:
            ids = backend.store(docs, "test_mixed")

        assert len(ids) == 2
        # Standard adapter must NOT be used for the fallback
        lc_emb.embed_query.assert_not_called()
        # LC encoder must be used for the chunk that lacked a precomputed vector
        mock_lc.assert_called_once_with("test-lc-model", "chunk two - no vector")

    def test_chroma_provenance_includes_strategy(self, tmp_dir):
        from src.rag.chroma_backend import ChromaBackend

        emb = MagicMock()
        emb.model_name = "test-model"
        lc_emb = MagicMock()
        emb.get_langchain_embeddings.return_value = lc_emb

        backend = ChromaBackend(embedding=emb, persist_directory=tmp_dir)

        docs = [
            Document(
                page_content="chunk",
                metadata={
                    "embedding_strategy": "late_chunking",
                    "late_chunking_model": "test-lc-model",
                    "_precomputed_vector": [0.1, 0.2, 0.3, 0.4],
                },
            ),
        ]

        backend.store(docs, "test_provenance")

        provenance = backend.get_embedding_provenance("test_provenance")
        assert provenance is not None
        assert provenance["embedding_strategy"] == "late_chunking"


class TestLateChunkingStrategyGuards:
    """Strategy enforcement: guards on legacy collections and missing vectors."""

    def test_chroma_rejects_lc_ingest_into_legacy_collection(self, tmp_dir):
        """LC ingest into an existing no-provenance (legacy) collection must be rejected."""
        from src.rag.chroma_backend import ChromaBackend

        emb = MagicMock()
        emb.model_name = "test-model"
        emb.get_langchain_embeddings.return_value = MagicMock()
        backend = ChromaBackend(embedding=emb, persist_directory=tmp_dir)
        # Pre-create a bare collection with no provenance metadata (legacy state)
        backend._client.create_collection("legacy_col")

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

        import pytest

        with pytest.raises(ValueError, match="no embedding provenance"):
            backend.store(docs, "legacy_col")

    def test_chroma_rejects_lc_docs_missing_model_name(self, tmp_dir):
        """LC docs with vectors but no late_chunking_model must be rejected before any write."""
        from src.rag.chroma_backend import ChromaBackend

        emb = MagicMock()
        emb.model_name = "test-model"
        emb.get_langchain_embeddings.return_value = MagicMock()
        backend = ChromaBackend(embedding=emb, persist_directory=tmp_dir)

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

        import pytest

        with pytest.raises(ValueError, match="late_chunking_model"):
            backend.store(docs, "test_no_model")

    def test_chroma_rejects_lc_docs_with_no_precomputed_vectors(self, tmp_dir):
        """LC-marked docs where none have _precomputed_vector must be rejected."""
        from src.rag.chroma_backend import ChromaBackend

        emb = MagicMock()
        emb.model_name = "test-model"
        emb.get_langchain_embeddings.return_value = MagicMock()
        backend = ChromaBackend(embedding=emb, persist_directory=tmp_dir)

        docs = [
            Document(
                page_content="chunk one",
                metadata={"embedding_strategy": "late_chunking", "late_chunking_model": "test-lc-model"},
            ),
            Document(
                page_content="chunk two",
                metadata={"embedding_strategy": "late_chunking", "late_chunking_model": "test-lc-model"},
            ),
        ]

        import pytest

        with pytest.raises(ValueError, match="none have '_precomputed_vector'"):
            backend.store(docs, "test_no_vectors")


class TestLateChunkingConfig:
    """Test config parsing for late chunking settings."""

    def test_defaults_loaded(self):
        from src.config.settings import get_settings

        settings = get_settings()
        lc = settings.chunking.late_chunking
        assert lc.enabled is False
        assert lc.model_name == "jinaai/jina-embeddings-v2-base-en"
        assert lc.max_context_tokens == 8192
        assert lc.pooling_strategy == "mean"
        assert lc.batch_by == "page"
        assert "pdf" in lc.file_types

    def test_late_chunking_and_contextual_independent(self):
        from src.config.settings import get_settings

        settings = get_settings()
        # Both can be configured, but ingestion prefers late_chunking when both enabled
        assert settings.chunking.contextual.enabled is False
        assert settings.chunking.late_chunking.enabled is False
