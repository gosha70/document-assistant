"""Tests for ChunkContextAugmenter."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.rag.contextual import ChunkContextAugmenter


def _make_settings(max_context_tokens=128, document_summary_tokens=256):
    settings = MagicMock()
    settings.chunking.contextual.max_context_tokens = max_context_tokens
    settings.chunking.contextual.document_summary_tokens = document_summary_tokens
    settings.chunking.tokenizer = "tiktoken:cl100k_base"
    settings.telemetry.enabled = False
    settings.telemetry.log_llm_calls = False
    return settings


def _make_llm(responses=None):
    llm = MagicMock()
    if responses is None:
        responses = ["This chunk discusses topic X."]
    results = []
    for r in responses:
        msg = MagicMock()
        msg.content = r
        results.append(msg)
    llm.invoke.side_effect = results
    return llm


class TestAugment:
    def test_augments_all_chunks(self):
        llm = _make_llm(["Context for chunk 1.", "Context for chunk 2."])
        settings = _make_settings()
        augmenter = ChunkContextAugmenter(llm=llm, settings=settings)

        chunks = [
            Document(page_content="Chunk one text.", metadata={"source": "a.txt"}),
            Document(page_content="Chunk two text.", metadata={"source": "a.txt"}),
        ]

        result = augmenter.augment("Full document text here.", chunks)

        assert len(result) == 2
        for chunk in result:
            assert "chunk_context" in chunk.metadata
            assert "embedding_text" in chunk.metadata

    def test_embedding_text_format(self):
        llm = _make_llm(["Generated context."])
        settings = _make_settings()
        augmenter = ChunkContextAugmenter(llm=llm, settings=settings)

        chunks = [Document(page_content="Original text.", metadata={})]
        result = augmenter.augment("Full doc.", chunks)

        assert result[0].metadata["embedding_text"] == "Generated context.\n\nOriginal text."

    def test_page_content_preserved(self):
        llm = _make_llm(["Some context."])
        settings = _make_settings()
        augmenter = ChunkContextAugmenter(llm=llm, settings=settings)

        chunks = [Document(page_content="Original text.", metadata={})]
        result = augmenter.augment("Full doc.", chunks)

        assert result[0].page_content == "Original text."

    def test_llm_failure_skips_chunk(self):
        llm = MagicMock()
        good_msg = MagicMock()
        good_msg.content = "Good context."
        llm.invoke.side_effect = [Exception("LLM error"), good_msg]

        settings = _make_settings()
        augmenter = ChunkContextAugmenter(llm=llm, settings=settings)

        chunks = [
            Document(page_content="Chunk 1.", metadata={}),
            Document(page_content="Chunk 2.", metadata={}),
        ]
        result = augmenter.augment("Full doc.", chunks)

        # First chunk should have no augmentation
        assert "chunk_context" not in result[0].metadata
        assert "embedding_text" not in result[0].metadata
        # Second chunk should be augmented
        assert result[1].metadata["chunk_context"] == "Good context."
        assert "embedding_text" in result[1].metadata

    def test_document_summary_truncation(self):
        llm = _make_llm(["Context."])
        # Very small summary token limit
        settings = _make_settings(document_summary_tokens=5)
        augmenter = ChunkContextAugmenter(llm=llm, settings=settings)

        long_text = "word " * 500
        chunks = [Document(page_content="Chunk.", metadata={})]
        augmenter.augment(long_text, chunks)

        # The prompt should have been called with a truncated summary
        call_args = llm.invoke.call_args[0][0]
        # The document_summary portion should be much shorter than the full text
        assert len(call_args) < len(long_text)

    def test_max_context_tokens_enforced(self):
        # LLM returns a very long context
        long_context = "word " * 500
        llm = _make_llm([long_context])
        settings = _make_settings(max_context_tokens=10)
        augmenter = ChunkContextAugmenter(llm=llm, settings=settings)

        chunks = [Document(page_content="Chunk.", metadata={})]
        result = augmenter.augment("Full doc.", chunks)

        # The stored context should be shorter than the raw LLM output
        stored_context = result[0].metadata["chunk_context"]
        assert len(stored_context) < len(long_context)

    def test_telemetry_recorded_when_enabled(self):
        llm = _make_llm(["Context."])
        settings = _make_settings()
        settings.telemetry.enabled = True
        settings.telemetry.log_llm_calls = True

        with patch("src.rag.contextual.get_metrics_collector") as mock_metrics:
            collector = MagicMock()
            mock_metrics.return_value = collector
            augmenter = ChunkContextAugmenter(llm=llm, settings=settings)

            chunks = [Document(page_content="Chunk.", metadata={})]
            augmenter.augment("Full doc.", chunks)

            collector.record_llm_call.assert_called_once()
            args = collector.record_llm_call.call_args[0]
            assert len(args) == 3  # latency, prompt_len, response_len
