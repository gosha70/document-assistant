"""Query transformation utilities — decomposition, HyDE, confidence scoring."""

import json
import logging
import math
import time
from typing import Any

from langchain_core.documents import Document

from src.rag.generation import load_prompt
from src.config.settings import get_settings
from src.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


def decompose_query(query: str, llm: Any, max_sub_queries: int = 5) -> list[str]:
    """Decompose a complex query into independent sub-queries via LLM.

    Returns [query] unchanged if not decomposable or on parse failure.
    """
    prompt = load_prompt("query_decomposition")
    formatted = prompt.format(question=query)

    settings = get_settings()
    start = time.monotonic()
    raw = llm.invoke(formatted)
    latency = time.monotonic() - start

    text = raw.content if hasattr(raw, "content") else str(raw)
    if settings.telemetry.enabled and settings.telemetry.log_llm_calls:
        get_metrics_collector().record_llm_call(latency, len(formatted), len(text))

    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            result = parsed[:max_sub_queries]
            if len(result) > 0:
                return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse decomposition response, using original query")

    return [query]


def merge_sub_results(sub_results: list[list[Document]]) -> list[Document]:
    """Flatten and deduplicate documents from multiple sub-query retrievals.

    Reuses the same dedup logic as Retriever. Preserves sub_query_index in metadata.
    """
    from src.rag.retrieval import deduplicate_docs

    all_docs: list[Document] = []
    for idx, docs in enumerate(sub_results):
        for doc in docs:
            doc.metadata["sub_query_index"] = idx
            all_docs.append(doc)

    return deduplicate_docs(all_docs)


def generate_hypothetical_document(query: str, llm: Any) -> str:
    """Generate a hypothetical answer passage for HyDE."""
    prompt = load_prompt("hyde")
    formatted = prompt.format(question=query)

    settings = get_settings()
    start = time.monotonic()
    raw = llm.invoke(formatted)
    latency = time.monotonic() - start

    text = raw.content if hasattr(raw, "content") else str(raw)
    if settings.telemetry.enabled and settings.telemetry.log_llm_calls:
        get_metrics_collector().record_llm_call(latency, len(formatted), len(text))

    return text.strip()


def compute_retrieval_confidence(docs: list[Document]) -> float:
    """Compute 0.0-1.0 confidence from reranker_score metadata.

    Uses sigmoid normalization on the mean reranker score.
    ms-marco cross-encoder outputs are typically in [-10, +10].
    Returns 1.0 if no scores present (no opinion — don't trigger corrective).
    """
    scores = [doc.metadata["reranker_score"] for doc in docs if "reranker_score" in doc.metadata]
    if not scores:
        return 1.0

    mean_score = sum(scores) / len(scores)
    # Sigmoid normalization: maps [-10, +10] → [0, 1]
    return 1.0 / (1.0 + math.exp(-mean_score))
