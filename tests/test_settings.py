"""Tests for config/settings loading from YAML."""

import os
import tempfile
import pytest

from src.config.settings import get_settings, _deep_merge


class TestSettingsFromYaml:
    def test_loads_defaults(self):
        s = get_settings()
        assert s.app.name == "Document Assistant (D.O.T.)"
        assert s.embedding.model_name == "hkunlp/instructor-large"
        assert s.chunking.chunk_size == 512
        assert s.chunking.chunk_overlap == 50
        assert s.chunking.tokenizer == "tiktoken:cl100k_base"
        assert s.vectorstore.backend == "chroma"
        assert s.vectorstore.collection_name == "EGOGE_DOCUMENTS_DB"
        assert s.model.n_ctx == 6144
        assert s.upload.max_file_size_mb == 50
        assert "pdf" in s.upload.allowed_extensions
        assert s.auth.enabled is False

    def test_override_file_merges(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("app:\n  port: 9999\nchunking:\n  chunk_size: 256\n")
            f.flush()
            try:
                s = get_settings(override_path=f.name)
                assert s.app.port == 9999
                assert s.chunking.chunk_size == 256
                # Other values still from defaults
                assert s.embedding.model_name == "hkunlp/instructor-large"
            finally:
                os.unlink(f.name)

    def test_system_prompt_loaded(self):
        s = get_settings()
        assert "document assistant" in s.system_prompt


class TestDeepMerge:
    def test_flat_merge(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        over = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, over) == {"x": {"a": 1, "b": 3, "c": 4}}
