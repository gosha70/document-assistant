import json
import logging
import re
import time
import yaml
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.rag.context_firewall import sanitize_document_text
from src.rag.generation import _truncate_context
from src.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class StudyOutputGenerator:
    """Generates study-oriented outputs (summary, glossary, flashcards) from retrieved context."""

    def __init__(self, llm: Any):
        self._llm = llm

    def _invoke(self, prompt_name: str, template_vars: dict) -> str:
        """Load a prompt from YAML, format it, invoke the LLM, and return the raw text."""
        path = _PROMPTS_DIR / f"{prompt_name}.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        template_str = data["templates"]["generic"]
        formatted = template_str.format(**template_vars)

        logger.info("StudyOutputGenerator [%s] prompt length: %d chars", prompt_name, len(formatted))
        settings = get_settings()
        start = time.monotonic()
        raw = self._llm.invoke(formatted)
        latency = time.monotonic() - start

        answer = raw.content if hasattr(raw, "content") else str(raw)
        if settings.telemetry.enabled and settings.telemetry.log_llm_calls:
            get_metrics_collector().record_llm_call(latency, len(formatted), len(answer))
        logger.info("StudyOutputGenerator [%s] completed in %.2fs", prompt_name, latency)
        return answer

    @staticmethod
    def _build_context(documents: list[Document]) -> str:
        parts = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            label = f"[Source: {source}, page {page}]" if page is not None else f"[Source: {source}]"
            parts.append(f"{label}\n{sanitize_document_text(doc.page_content)}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json_list(text: str) -> list[dict]:
        """Extract and parse a JSON array from LLM output. Returns [] on failure."""
        text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse JSON list from LLM output: %.200r", text)
        return []

    def summarize(self, documents: list[Document]) -> str:
        documents = _truncate_context(documents, get_settings().model.max_context_chars)
        context = self._build_context(documents)
        return self._invoke("summarize", {"context": context})

    def extract_glossary(self, documents: list[Document]) -> list[dict]:
        documents = _truncate_context(documents, get_settings().model.max_context_chars)
        context = self._build_context(documents)
        raw = self._invoke("glossary", {"context": context})
        return self._parse_json_list(raw)

    def generate_flashcards(self, documents: list[Document], max_cards: int = 10) -> list[dict]:
        documents = _truncate_context(documents, get_settings().model.max_context_chars)
        context = self._build_context(documents)
        raw = self._invoke("flashcards", {"context": context, "max_cards": str(max_cards)})
        return self._parse_json_list(raw)
