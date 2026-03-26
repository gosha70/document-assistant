import logging
import time
from typing import Any, Iterator

import yaml
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.config.settings import get_settings
from src.rag.context_firewall import sanitize_document_text
from src.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _truncate_context(documents: list[Document], max_chars: int) -> list[Document]:
    """Return a prefix of *documents* whose combined page_content fits within *max_chars*.

    Always returns at least one document so the caller always has some context.
    Logs a warning when truncation actually occurs.
    """
    total = 0
    result = []
    for doc in documents:
        if total + len(doc.page_content) > max_chars:
            break
        result.append(doc)
        total += len(doc.page_content)
    if not result and documents:
        logger.warning(
            "Context budget guard: first document (%d chars) exceeds max_context_chars=%d; "
            "including it anyway to avoid empty context.",
            len(documents[0].page_content),
            max_chars,
        )
        result = [documents[0]]
    if len(result) < len(documents):
        logger.warning(
            "Context budget guard: truncated context from %d to %d documents (%d/%d chars used).",
            len(documents),
            len(result),
            sum(len(d.page_content) for d in result),
            max_chars,
        )
    return result


def load_prompt(name: str, template_type: str = "generic", use_history: bool = False) -> PromptTemplate:
    """Load a prompt template from a YAML file in prompts/."""
    path = _PROMPTS_DIR / f"{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    templates = data.get("templates", {})
    variant = f"{template_type}_history" if use_history else template_type

    # Fall back: specific variant -> template_type -> generic -> generic without history
    template_str = (
        templates.get(variant)
        or templates.get(template_type)
        or templates.get("generic_history" if use_history else "generic")
        or templates.get("generic")
    )

    if template_str is None:
        raise ValueError(f"No template found for variant '{variant}' in {path}")

    input_variables = data.get("input_variables", ["context", "question"])
    if use_history and "history" not in input_variables:
        input_variables = ["history"] + input_variables
    if "system_prompt" in template_str and "system_prompt" not in input_variables:
        input_variables = ["system_prompt"] + input_variables

    return PromptTemplate(input_variables=input_variables, template=template_str)


class Generator:
    """Generates answers from retrieved context using an LLM."""

    def __init__(
        self,
        llm: Any,
        system_prompt: str = "",
        prompt_name: str = "qa",
        template_type: str = "generic",
        use_history: bool = False,
    ):
        self._llm = llm
        self._prompt = load_prompt(prompt_name, template_type, use_history)
        self._use_history = use_history
        self._system_prompt = system_prompt

    def generate(
        self,
        query: str,
        documents: list[Document],
        history: str = "",
    ) -> dict:
        """Generate an answer with source citations.

        Returns:
            dict with keys: answer (str), sources (list[dict])
        """
        settings = get_settings()
        documents = _truncate_context(documents, settings.model.max_context_chars)
        context = "\n\n".join(sanitize_document_text(doc.page_content) for doc in documents)

        prompt_kwargs = {"context": context, "question": query, "system_prompt": self._system_prompt}
        if self._use_history:
            prompt_kwargs["history"] = history

        formatted = self._prompt.format(**prompt_kwargs)
        logger.info(f"Generating answer for query (prompt length: {len(formatted)} chars)")

        start = time.monotonic()
        raw_answer = self._llm.invoke(formatted)
        latency = time.monotonic() - start

        answer = raw_answer.content if hasattr(raw_answer, "content") else str(raw_answer)
        if settings.telemetry.enabled and settings.telemetry.log_llm_calls:
            get_metrics_collector().record_llm_call(latency, len(formatted), len(answer))
            logger.info(f"LLM response in {latency:.2f}s ({len(answer)} chars)")

        sources = self.extract_sources(documents)
        return {"answer": answer, "sources": sources}

    def generate_stream(
        self,
        query: str,
        documents: list[Document],
        history: str = "",
    ) -> Iterator[str]:
        """Stream answer tokens. Yields individual text chunks.

        Requires an LLM that supports .stream() (e.g. ChatOpenAI).
        Falls back to non-streaming invoke if .stream() is not available.
        """
        settings = get_settings()
        documents = _truncate_context(documents, settings.model.max_context_chars)
        context = "\n\n".join(sanitize_document_text(doc.page_content) for doc in documents)

        prompt_kwargs = {"context": context, "question": query, "system_prompt": self._system_prompt}
        if self._use_history:
            prompt_kwargs["history"] = history

        formatted = self._prompt.format(**prompt_kwargs)
        logger.info(f"Streaming answer for query (prompt length: {len(formatted)} chars)")

        settings = get_settings()
        start = time.monotonic()
        output_length = 0

        if hasattr(self._llm, "stream"):
            for chunk in self._llm.stream(formatted):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                output_length += len(token)
                yield token
        else:
            raw_answer = self._llm.invoke(formatted)
            token = raw_answer.content if hasattr(raw_answer, "content") else str(raw_answer)
            output_length = len(token)
            yield token

        latency = time.monotonic() - start
        if settings.telemetry.enabled and settings.telemetry.log_llm_calls:
            get_metrics_collector().record_llm_call(latency, len(formatted), output_length)
            logger.info(f"LLM stream completed in {latency:.2f}s ({output_length} chars)")

    @staticmethod
    def extract_sources(documents: list[Document]) -> list[dict]:
        """Extract source citations from documents."""
        return [
            {
                "file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "excerpt": doc.page_content[:200],
                "chunk_id": doc.metadata.get("id"),
                "search_type": doc.metadata.get("search_type"),
                "full_excerpt": doc.page_content[:500],
            }
            for doc in documents
        ]
