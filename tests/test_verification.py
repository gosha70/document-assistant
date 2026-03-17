"""Tests for answer verification."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.verification import verify_answer


def _mock_llm(response_text):
    llm = MagicMock()
    result = MagicMock()
    result.content = response_text
    llm.invoke.return_value = result
    return llm


class TestVerifyAnswer:
    @patch("src.rag.verification.get_settings")
    def test_verified_answer(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = _mock_llm('{"verified": true, "unsupported_claims": []}')

        sources = [Document(page_content="Python is a language.", metadata={})]
        result = verify_answer("Python is a language.", sources, "What is Python?", llm)

        assert result["verified"] is True
        assert result["unsupported_claims"] == []
        assert result["revised_answer"] is None

    @patch("src.rag.verification.get_settings")
    def test_unverified_answer(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = _mock_llm('{"verified": false, "unsupported_claims": ["Python was created in 1995"]}')

        sources = [Document(page_content="Python is a language.", metadata={})]
        result = verify_answer("Python was created in 1995.", sources, "When was Python created?", llm)

        assert result["verified"] is False
        assert len(result["unsupported_claims"]) == 1
        assert result["revised_answer"] is not None
        assert "could not be verified" in result["revised_answer"]

    @patch("src.rag.verification.get_settings")
    def test_malformed_json_passes_through(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = _mock_llm("not valid json at all")

        sources = [Document(page_content="text", metadata={})]
        result = verify_answer("answer", sources, "question", llm)

        assert result["verified"] is True
        assert result["unsupported_claims"] == []
        assert result["revised_answer"] is None

    @patch("src.rag.verification.get_settings")
    def test_unverified_no_claims_no_revision(self, mock_settings):
        mock_settings.return_value.telemetry.enabled = False
        llm = _mock_llm('{"verified": false, "unsupported_claims": []}')

        sources = [Document(page_content="text", metadata={})]
        result = verify_answer("answer", sources, "question", llm)

        assert result["verified"] is False
        assert result["revised_answer"] is None
