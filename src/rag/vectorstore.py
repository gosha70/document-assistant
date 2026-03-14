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

    @abstractmethod
    def find_by_source(self, source_name: str, collection_name: str) -> list[str]:
        """Return document IDs whose 'source' metadata matches the given filename.

        Used for duplicate detection and replacement at ingest time.
        """

    def get_embedding_provenance(self, collection_name: str) -> Optional[dict]:
        """Return the embedding model/type that was used to build a collection.

        Returns a dict with 'model_name' and 'type' keys, or None if provenance
        is unknown (e.g. collections created before provenance tracking).
        """
        return None

    @abstractmethod
    def sample_chunks(self, collection_name: str, limit: int = 10, offset: int = 0) -> dict:
        """Return a sample of chunks from a collection.

        Returns a dict with 'total_count' (int) and 'chunks' (list of dicts with
        'id', 'text', and 'metadata' keys). Text is truncated to 500 characters.
        """

    @abstractmethod
    def list_sources(self, collection_name: str) -> dict:
        """Return unique source filenames and their chunk counts.

        Returns a dict with:
          - 'sources': list of dicts with 'filename' and 'chunk_count' keys
          - 'scanned_chunks': number of chunks actually scanned
          - 'truncated': True if the scan hit the 10k limit before reading all chunks
        """
