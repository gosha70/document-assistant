"""Tests for chunk metadata enrichment during ingestion."""
from langchain_core.documents import Document

from src.rag.chunking import count_tokens, enrich_chunk_metadata, CHUNKING_VERSION


class TestCountTokens:
    def test_tiktoken_counting(self):
        count = count_tokens("Hello, world!", "tiktoken:cl100k_base")
        assert isinstance(count, int)
        assert count > 0

    def test_tiktoken_empty_string(self):
        count = count_tokens("", "tiktoken:cl100k_base")
        assert count == 0

    def test_tokenizer_is_cached(self):
        """Calling count_tokens twice with the same spec should reuse the cached tokenizer."""
        from src.rag.chunking import _tokenizer_cache
        count_tokens("test", "tiktoken:cl100k_base")
        assert "tiktoken:cl100k_base" in _tokenizer_cache
        cached = _tokenizer_cache["tiktoken:cl100k_base"]
        count_tokens("test again", "tiktoken:cl100k_base")
        assert _tokenizer_cache["tiktoken:cl100k_base"] is cached


class TestEnrichChunkMetadata:
    def test_metadata_fields_added(self):
        docs = [
            Document(page_content="Hello world", metadata={"source": "test.txt"}),
            Document(page_content="Another chunk", metadata={"source": "test.txt", "page": 2}),
        ]
        enriched = enrich_chunk_metadata(docs, "tiktoken:cl100k_base")
        for doc in enriched:
            assert "token_count" in doc.metadata
            assert doc.metadata["tokenizer"] == "tiktoken:cl100k_base"
            assert doc.metadata["chunking_version"] == CHUNKING_VERSION
            assert isinstance(doc.metadata["token_count"], int)
            assert doc.metadata["token_count"] > 0

    def test_existing_metadata_preserved(self):
        docs = [Document(page_content="test", metadata={"source": "file.pdf", "page": 3})]
        enriched = enrich_chunk_metadata(docs, "tiktoken:cl100k_base")
        assert enriched[0].metadata["source"] == "file.pdf"
        assert enriched[0].metadata["page"] == 3


class TestPdfConverterTextSplitter:
    def test_pdf_converter_uses_text_splitter(self):
        from embeddings.unstructured.pdf_converter import PdfConverter
        import inspect
        sig = inspect.signature(PdfConverter.load_and_split_file)
        assert "text_splitter" in sig.parameters


class TestLegacyPathUsesUnifiedChunking:
    def test_base_file_converter_delegates_to_src_rag_chunking(self):
        """BaseFileConverter.get_text_splitter() should delegate to src.rag.chunking."""
        from embeddings.unstructured.base_file_converter import BaseFileConverter
        from embeddings.unstructured.file_type import FileType
        splitter = BaseFileConverter.get_text_splitter(FileType.PYTHON)
        # Should use token-aware splitting (512 token chunks from config)
        assert splitter._chunk_size == 512
        assert splitter._chunk_overlap == 50
