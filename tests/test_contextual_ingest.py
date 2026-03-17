"""Tests for two-pass contextual ingestion wiring in _load_and_split."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def _make_settings(contextual_enabled=False):
    settings = MagicMock()
    settings.chunking.contextual.enabled = contextual_enabled
    settings.chunking.contextual.max_context_tokens = 128
    settings.chunking.contextual.document_summary_tokens = 256
    settings.chunking.tokenizer = "tiktoken:cl100k_base"
    settings.chunking.chunk_size = 512
    settings.chunking.chunk_overlap = 50
    settings.telemetry.enabled = False
    settings.telemetry.log_llm_calls = False
    return settings


class TestLoadAndSplitContextual:
    @patch("src.api.routes.ingest.get_settings")
    @patch("src.api.routes.ingest.get_text_splitter")
    @patch("src.api.routes.ingest.enrich_chunk_metadata")
    def test_disabled_mode_uses_single_pass(self, mock_enrich, mock_splitter, mock_get_settings):
        """When contextual.enabled=False, the original single-pass path is used."""
        settings = _make_settings(contextual_enabled=False)
        mock_get_settings.return_value = settings
        mock_splitter.return_value = MagicMock()

        mock_converter = MagicMock()
        docs = [Document(page_content="chunk", metadata={})]
        mock_converter.load_and_split_file.return_value = docs
        mock_enrich.return_value = docs

        with patch("embeddings.unstructured.file_type.FileType") as mock_ft:
            mock_ft.get_file_type_by_extension.return_value = "txt"
            with patch("embeddings.unstructured.document_splitter.DocumentSplitter") as mock_ds:
                mock_ds.return_value.get_converter.return_value = mock_converter

                from src.api.routes.ingest import _load_and_split

                result = _load_and_split("/tmp/test.txt")

        # Converter called exactly once (single pass)
        assert mock_converter.load_and_split_file.call_count == 1
        assert result == docs

    @patch("src.api.routes.ingest.get_settings")
    @patch("src.api.routes.ingest.get_text_splitter")
    @patch("src.api.routes.ingest.enrich_chunk_metadata")
    @patch("src.api.deps._llm", new=None)
    def test_contextual_no_llm_fallback(self, mock_enrich, mock_splitter, mock_get_settings):
        """When contextual.enabled=True but no LLM, chunks returned without augmentation."""
        settings = _make_settings(contextual_enabled=True)
        mock_get_settings.return_value = settings
        mock_splitter.return_value = MagicMock()

        full_doc = Document(page_content="Full document text.", metadata={})
        chunk = Document(page_content="chunk", metadata={})
        mock_converter = MagicMock()
        mock_converter.load_and_split_file.side_effect = [[full_doc], [chunk]]
        mock_enrich.return_value = [chunk]

        with patch("embeddings.unstructured.file_type.FileType") as mock_ft:
            mock_ft.get_file_type_by_extension.return_value = "txt"
            with patch("embeddings.unstructured.document_splitter.DocumentSplitter") as mock_ds:
                mock_ds.return_value.get_converter.return_value = mock_converter

                from src.api.routes.ingest import _load_and_split

                result = _load_and_split("/tmp/test.txt")

        # Converter called twice (two-pass) but no augmentation
        assert mock_converter.load_and_split_file.call_count == 2
        assert "embedding_text" not in result[0].metadata

    @patch("src.api.routes.ingest.get_settings")
    @patch("src.api.routes.ingest.get_text_splitter")
    @patch("src.api.routes.ingest.enrich_chunk_metadata")
    def test_contextual_with_llm_augments(self, mock_enrich, mock_splitter, mock_get_settings):
        """When contextual.enabled=True and LLM available, chunks get augmented."""
        settings = _make_settings(contextual_enabled=True)
        mock_get_settings.return_value = settings
        mock_splitter.return_value = MagicMock()

        full_doc = Document(page_content="Full document text.", metadata={})
        chunk = Document(page_content="chunk text", metadata={})
        mock_converter = MagicMock()
        mock_converter.load_and_split_file.side_effect = [[full_doc], [chunk]]
        mock_enrich.return_value = [chunk]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated context."
        mock_llm.invoke.return_value = mock_response

        with patch("embeddings.unstructured.file_type.FileType") as mock_ft:
            mock_ft.get_file_type_by_extension.return_value = "txt"
            with patch("embeddings.unstructured.document_splitter.DocumentSplitter") as mock_ds:
                mock_ds.return_value.get_converter.return_value = mock_converter
                with patch("src.api.deps._llm", new=mock_llm):
                    from src.api.routes.ingest import _load_and_split

                    result = _load_and_split("/tmp/test.txt")

        assert result[0].metadata["chunk_context"] == "Generated context."
        assert result[0].metadata["embedding_text"] == "Generated context.\n\nchunk text"
        assert result[0].page_content == "chunk text"

    @patch("src.api.routes.ingest.get_settings")
    @patch("src.api.routes.ingest.get_text_splitter")
    @patch("src.api.routes.ingest.enrich_chunk_metadata")
    def test_two_pass_uses_converter_not_textloader(self, mock_enrich, mock_splitter, mock_get_settings):
        """Two-pass path uses the converter (not TextLoader) for full-text extraction."""
        settings = _make_settings(contextual_enabled=True)
        mock_get_settings.return_value = settings
        mock_splitter.return_value = MagicMock()

        full_doc = Document(page_content="PDF extracted text.", metadata={})
        chunk = Document(page_content="chunk", metadata={})
        mock_converter = MagicMock()
        mock_converter.load_and_split_file.side_effect = [[full_doc], [chunk]]
        mock_enrich.return_value = [chunk]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Context."
        mock_llm.invoke.return_value = mock_response

        with patch("embeddings.unstructured.file_type.FileType") as mock_ft:
            mock_ft.get_file_type_by_extension.return_value = "pdf"
            with patch("embeddings.unstructured.document_splitter.DocumentSplitter") as mock_ds:
                mock_ds.return_value.get_converter.return_value = mock_converter
                with patch("src.api.deps._llm", new=mock_llm):
                    from src.api.routes.ingest import _load_and_split

                    result = _load_and_split("/tmp/test.pdf")

        # Both passes went through the converter
        assert mock_converter.load_and_split_file.call_count == 2
        # First call used the no-split splitter (chunk_size=1M)
        first_call_splitter = mock_converter.load_and_split_file.call_args_list[0]
        no_split_arg = first_call_splitter.kwargs.get("text_splitter") or first_call_splitter[1].get("text_splitter")
        if no_split_arg is None and first_call_splitter.args:
            no_split_arg = first_call_splitter.args[0]
        assert no_split_arg._chunk_size == 1_000_000
