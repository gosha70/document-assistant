"""Dependency wiring for the FastAPI application.

Provides singleton-style accessors for the vector store backend, retriever,
and generator. These are initialised lazily on first access.
"""
import logging
from typing import Any, Optional

from src.config.settings import get_settings
from src.rag.vectorstore import VectorStoreBackend
from src.rag.retrieval import Retriever
from src.rag.reranking import NoOpReranker
from src.rag.generation import Generator

logger = logging.getLogger(__name__)

_backend: Optional[VectorStoreBackend] = None
_llm: Any = None


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


def get_retriever(collection_name: Optional[str] = None) -> Retriever:
    settings = get_settings()
    collection = collection_name or settings.vectorstore.collection_name
    return Retriever(
        backend=get_vectorstore_backend(),
        collection_name=collection,
        reranker=NoOpReranker(),
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
