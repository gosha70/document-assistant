"""Answer verification — checks that claims are grounded in cited sources."""

import json
import logging
import time
from typing import Any

from langchain_core.documents import Document

from src.rag.generation import load_prompt
from src.config.settings import get_settings
from src.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


def verify_answer(
    answer: str,
    sources: list[Document],
    query: str,
    llm: Any,
) -> dict:
    """Verify that an answer is grounded in the provided source documents.

    Returns {"verified": bool, "unsupported_claims": list[str], "revised_answer": str | None}.
    On parse failure, returns verified=True (pass-through).
    """
    source_text = "\n\n".join(f"[Source {i+1}]: {doc.page_content[:500]}" for i, doc in enumerate(sources))

    prompt = load_prompt("verification")
    formatted = prompt.format(answer=answer, sources=source_text, question=query)

    settings = get_settings()
    start = time.monotonic()
    raw = llm.invoke(formatted)
    latency = time.monotonic() - start

    text = raw.content if hasattr(raw, "content") else str(raw)
    if settings.telemetry.enabled and settings.telemetry.log_llm_calls:
        get_metrics_collector().record_llm_call(latency, len(formatted), len(text))

    try:
        parsed = json.loads(text.strip())
        verified = parsed.get("verified", True)
        unsupported = parsed.get("unsupported_claims", [])

        result = {
            "verified": verified,
            "unsupported_claims": unsupported,
            "revised_answer": None,
        }

        if not verified and unsupported:
            result["revised_answer"] = (
                answer + "\n\nNote: Some claims in this answer could not be verified " "against the provided documents."
            )

        return result
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning("Failed to parse verification response, passing through")
        return {
            "verified": True,
            "unsupported_claims": [],
            "revised_answer": None,
        }
