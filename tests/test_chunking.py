"""Tests for tokenizer-aware chunking."""
import pytest
from unittest.mock import patch, MagicMock

from src.rag.chunking import get_text_splitter, _parse_tokenizer, CHUNKING_VERSION


class TestParseTokenizer:
    def test_tiktoken_spec(self):
        backend, model = _parse_tokenizer("tiktoken:cl100k_base")
        assert backend == "tiktoken"
        assert model == "cl100k_base"

    def test_huggingface_spec(self):
        backend, model = _parse_tokenizer("huggingface:bert-base-uncased")
        assert backend == "huggingface"
        assert model == "bert-base-uncased"

    def test_invalid_spec_no_colon(self):
        with pytest.raises(ValueError, match="Invalid tokenizer spec"):
            _parse_tokenizer("tiktoken")

    def test_spec_with_multiple_colons(self):
        backend, model = _parse_tokenizer("huggingface:org/model:v2")
        assert backend == "huggingface"
        assert model == "org/model:v2"


class TestGetTextSplitter:
    def test_tiktoken_splitter_created(self):
        splitter = get_text_splitter()
        # Should create a splitter that measures by tokens, not characters
        assert splitter is not None
        assert splitter._chunk_size == 512
        assert splitter._chunk_overlap == 50

    def test_language_separators_applied(self):
        splitter_py = get_text_splitter("py")
        splitter_txt = get_text_splitter("txt")
        # Python splitter should have different separators than plain text
        assert splitter_py._separators != splitter_txt._separators

    def test_unknown_extension_uses_default_separators(self):
        splitter = get_text_splitter("xyz")
        assert splitter is not None

    def test_extension_with_dot_prefix(self):
        splitter = get_text_splitter(".py")
        # Should strip the dot and still find Python separators
        assert splitter is not None

    def test_unknown_backend_raises(self):
        with patch("src.rag.chunking.get_settings") as mock:
            s = MagicMock()
            s.chunking.tokenizer = "unknown_backend:model"
            s.chunking.chunk_size = 512
            s.chunking.chunk_overlap = 50
            mock.return_value = s
            with pytest.raises(ValueError, match="Unknown tokenizer backend"):
                get_text_splitter()


class TestTokenAwareSplitting:
    def test_chunks_respect_token_limit(self):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        splitter = get_text_splitter()
        # Create text that is clearly larger than 512 tokens
        long_text = "word " * 2000
        chunks = splitter.split_text(long_text)
        assert len(chunks) > 1
        for chunk in chunks:
            token_count = len(enc.encode(chunk))
            assert token_count <= 512, f"Chunk has {token_count} tokens, exceeds limit of 512"


class TestChunkingVersion:
    def test_version_is_set(self):
        assert CHUNKING_VERSION == "2"
