"""Tests for the Instructor embedding adapter.

Two layers:
  * mocked tests (always run) — pin the instruction/prompt contract
  * real-model test — skipped when the model is absent from the local HF cache,
    unless REQUIRE_REAL_EMBEDDING_MODEL=1, which turns that skip into a failure.
    CI sets the flag so a regression like the sentence-transformers 5.4.0 break
    (removal of SentenceTransformer._text_length) cannot pass unnoticed.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config.settings import get_settings
from src.rag.embeddings import InstructorEmbeddingAdapter

MODEL_NAME = "hkunlp/instructor-large"
EXPECTED_DIM = 768
REQUIRE_REAL_MODEL_ENV = "REQUIRE_REAL_EMBEDDING_MODEL"


def _make_adapter(**overrides):
    kwargs = {
        "model_name": MODEL_NAME,
        "embed_instruction": "DOC: ",
        "query_instruction": "QUERY: ",
        "device": "cpu",
        "normalize_embeddings": True,
    }
    kwargs.update(overrides)
    return InstructorEmbeddingAdapter(**kwargs)


class TestInstructorAdapterContract:
    """Mocked: the adapter must pass the right instruction as `prompt` per call type."""

    def test_documents_use_embed_instruction(self):
        model = MagicMock()
        model.encode.return_value = np.zeros((2, EXPECTED_DIM), dtype=np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            adapter = _make_adapter()
            adapter.embed_documents(["a", "b"])
        _, kwargs = model.encode.call_args
        assert kwargs["prompt"] == "DOC: "
        assert kwargs["normalize_embeddings"] is True

    def test_query_uses_query_instruction(self):
        model = MagicMock()
        model.encode.return_value = np.zeros((1, EXPECTED_DIM), dtype=np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            adapter = _make_adapter()
            adapter.embed_query("q")
        _, kwargs = model.encode.call_args
        assert kwargs["prompt"] == "QUERY: "

    def test_normalize_flag_is_forwarded(self):
        model = MagicMock()
        model.encode.return_value = np.zeros((1, EXPECTED_DIM), dtype=np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            adapter = _make_adapter(normalize_embeddings=False)
            adapter.embed_documents(["a"])
        _, kwargs = model.encode.call_args
        assert kwargs["normalize_embeddings"] is False

    def test_returns_plain_lists_not_arrays(self):
        model = MagicMock()
        model.encode.return_value = np.zeros((1, EXPECTED_DIM), dtype=np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            adapter = _make_adapter()
            vectors = adapter.embed_documents(["a"])
            query_vector = adapter.embed_query("q")
        assert isinstance(vectors, list) and isinstance(vectors[0], list)
        assert isinstance(query_vector, list) and isinstance(query_vector[0], float)

    def test_langchain_view_is_embeddings_compatible(self):
        model = MagicMock()
        model.encode.return_value = np.zeros((1, EXPECTED_DIM), dtype=np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            adapter = _make_adapter()
            lc = adapter.get_langchain_embeddings()
        assert callable(lc.embed_documents) and callable(lc.embed_query)


class TestConfiguredInstructions:
    """The shipped instruction strings are part of the embedding contract."""

    def test_defaults_match_stored_vector_contract(self):
        embedding = get_settings().embedding
        assert embedding.embed_instruction == "Represent the document for retrieval: "
        assert embedding.query_instruction == "Represent the question for retrieving supporting documents: "


def _model_in_local_cache() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return try_to_load_from_cache(MODEL_NAME, "modules.json") is not None


def _require_real_model() -> bool:
    return os.environ.get(REQUIRE_REAL_MODEL_ENV, "").strip() == "1"


class TestRealInstructorModel:
    """Loads the real model through the adapter's public surface."""

    @pytest.fixture(scope="class")
    def adapter(self):
        if not _model_in_local_cache():
            if _require_real_model():
                pytest.fail(
                    f"{REQUIRE_REAL_MODEL_ENV}=1 requires the real model, but {MODEL_NAME} "
                    "is not in the Hugging Face cache — provision it before running this job."
                )
            pytest.skip(
                f"{MODEL_NAME} is not in the local Hugging Face cache. "
                f"Set {REQUIRE_REAL_MODEL_ENV}=1 to turn this skip into a failure."
            )
        embedding = get_settings().embedding
        return InstructorEmbeddingAdapter(
            model_name=embedding.model_name,
            embed_instruction=embedding.embed_instruction,
            query_instruction=embedding.query_instruction,
            device=embedding.device,
            normalize_embeddings=embedding.normalize_embeddings,
        )

    def test_embed_documents(self, adapter):
        vectors = np.asarray(adapter.embed_documents(["Chroma is a vector database.", "BM25 ranks by keywords."]))
        assert vectors.shape == (2, EXPECTED_DIM)
        assert np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        # distinct inputs must not collapse to the same vector
        assert float(np.dot(vectors[0], vectors[1])) < 0.999

    def test_embed_query(self, adapter):
        vector = np.asarray(adapter.embed_query("What is a vector database?"))
        assert vector.shape == (EXPECTED_DIM,)
        assert np.isfinite(vector).all()
        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5

    def test_document_and_query_paths_differ(self, adapter):
        text = "Chroma is a vector database."
        as_document = np.asarray(adapter.embed_documents([text])[0])
        as_query = np.asarray(adapter.embed_query(text))
        # different instructions must produce different vectors for identical text
        assert float(np.dot(as_document, as_query)) < 0.9999

    def test_langchain_view_matches_adapter(self, adapter):
        text = "Reciprocal Rank Fusion merges rankings."
        via_adapter = np.asarray(adapter.embed_documents([text])[0])
        via_langchain = np.asarray(adapter.get_langchain_embeddings().embed_documents([text])[0])
        assert np.allclose(via_adapter, via_langchain, atol=1e-6)

    def test_long_input_is_truncated_not_rejected(self, adapter):
        long_text = "Retrieval augmented generation combines a retriever with a generator. " * 120
        vector = np.asarray(adapter.embed_documents([long_text])[0])
        assert vector.shape == (EXPECTED_DIM,)
        assert np.isfinite(vector).all()
