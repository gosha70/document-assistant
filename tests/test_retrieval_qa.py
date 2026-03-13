"""Tests for retrieval_qa.py with mocked LLM."""
import pytest
from unittest.mock import patch, MagicMock

transformers_available = True
try:
    import transformers
except ImportError:
    transformers_available = False

pytestmark = pytest.mark.skipif(
    not transformers_available,
    reason="transformers not installed"
)


class TestCreateModel:
    @patch("retrieval_qa.gguf")
    def test_gguf_model_selected_for_gguf_extension(self, mock_gguf):
        from retrieval_qa import create_model

        model_info = MagicMock()
        model_info.model_basename = "model.gguf"
        model_info.model_id = "test/model"
        model_info.device_type = "cpu"

        mock_gguf.return_value = MagicMock()
        result = create_model(model_info)

        mock_gguf.assert_called_once()

    @patch("retrieval_qa.gguf")
    def test_ggml_model_selected_for_ggml_extension(self, mock_gguf):
        from retrieval_qa import create_model

        model_info = MagicMock()
        model_info.model_basename = "model.ggml"
        model_info.model_id = "test/model"
        model_info.device_type = "cpu"

        mock_gguf.return_value = MagicMock()
        result = create_model(model_info)

        mock_gguf.assert_called_once()


class TestCreateRetrievalQA:
    def test_rejects_non_chroma_vectorstore(self):
        from retrieval_qa import create_retrieval_qa

        model_info = MagicMock()
        prompt_info = MagicMock()
        not_chroma = MagicMock(spec=[])  # Empty spec, not a Chroma

        with pytest.raises(TypeError, match="vectorstore must be of type Chroma"):
            create_retrieval_qa(model_info, prompt_info, not_chroma)
