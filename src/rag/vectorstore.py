from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.documents import Document


class VectorStoreBackend(ABC):
    """Interface for vector store backends (Chroma, Qdrant, etc.)."""

    @abstractmethod
    def store(self, documents: list[Document], collection_name: str) -> list[str]:
        """Add documents to the store. Returns list of document IDs."""

    @abstractmethod
    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> list[Document]:
        """Dense vector similarity search."""

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> list[Document]:
        """Combined dense + sparse search. Falls back to dense-only if not supported."""

    @abstractmethod
    def delete(self, ids: list[str], collection_name: str) -> None:
        """Delete documents by ID."""

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> dict:
        """Return metadata about a collection: doc count, embedding model, etc."""

    @abstractmethod
    def list_collections(self) -> list[dict]:
        """List all collections with basic metadata."""

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return number of documents in a collection."""
