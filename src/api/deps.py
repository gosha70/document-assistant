"""Dependency wiring for the FastAPI application.

Provides singleton-style accessors for the vector store backend, retriever,
generator, and orchestrator.  These are initialised lazily on first access.
"""

import logging
from typing import Any, Optional

from src.config.settings import get_settings
from src.rag.vectorstore import VectorStoreBackend
from src.rag.retrieval import Retriever
from src.rag.reranking import Reranker, NoOpReranker
from src.rag.generation import Generator
from src.rag.orchestrator import QueryOrchestrator
from src.rag.embeddings import EmbeddingAdapter

logger = logging.getLogger(__name__)

_backend: Optional[VectorStoreBackend] = None
_llm: Any = None
_reranker: Optional[Reranker] = None
_embedding: Optional[EmbeddingAdapter] = None


def set_vectorstore_backend(backend: VectorStoreBackend) -> None:
    """Set the vector store backend (called during app startup)."""
    global _backend
    _backend = backend


def get_vectorstore_backend() -> VectorStoreBackend:
    if _backend is None:
        raise RuntimeError("Vector store backend not initialised. Call set_vectorstore_backend() at startup.")
    return _backend


def set_llm(llm: Any) -> None:
    """Set the LLM instance (called during app startup)."""
    global _llm
    _llm = llm


def set_reranker(reranker: Optional[Reranker]) -> None:
    """Set the reranker instance (called during app startup). None means disabled."""
    global _reranker
    _reranker = reranker


def set_embedding(embedding: EmbeddingAdapter) -> None:
    """Set the embedding adapter (called during app startup)."""
    global _embedding
    _embedding = embedding


def get_retriever(collection_name: Optional[str] = None, k: Optional[int] = None) -> Retriever:
    settings = get_settings()
    collection = collection_name or settings.vectorstore.collection_name
    reranker = _reranker or NoOpReranker()
    final_k = k if k is not None else settings.reranker.top_k
    return Retriever(
        backend=get_vectorstore_backend(),
        collection_name=collection,
        reranker=reranker,
        initial_k=max(20, final_k),
        final_k=final_k,
        use_hybrid=settings.vectorstore.hybrid.enabled,
    )


def get_generator(template_type: str = "generic", use_history: bool = False) -> Generator:
    if _llm is None:
        raise RuntimeError("LLM not initialised. Call set_llm() at startup.")
    settings = get_settings()
    return Generator(
        llm=_llm,
        system_prompt=settings.system_prompt,
        prompt_name="qa",
        template_type=template_type,
        use_history=use_history,
    )


def get_orchestrator(
    collection_name: Optional[str] = None,
    template_type: str = "generic",
    use_history: bool = False,
) -> QueryOrchestrator:
    if _llm is None:
        raise RuntimeError("LLM not initialised. Call set_llm() at startup.")
    if _embedding is None:
        raise RuntimeError("Embedding not initialised. Call set_embedding() at startup.")
    settings = get_settings()
    retriever = get_retriever(collection_name)
    generator = get_generator(template_type, use_history)
    return QueryOrchestrator(
        retriever=retriever,
        generator=generator,
        llm=_llm,
        embedding=_embedding,
        settings=settings,
    )
