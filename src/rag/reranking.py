from abc import ABC, abstractmethod
from langchain_core.documents import Document


class Reranker(ABC):
    """Interface for reranking retrieved documents. Concrete implementations in Phase 7."""

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Rerank documents by relevance to query and return top_k."""


class NoOpReranker(Reranker):
    """Pass-through reranker that simply truncates to top_k. Used until a real reranker is configured."""

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        return documents[:top_k]
