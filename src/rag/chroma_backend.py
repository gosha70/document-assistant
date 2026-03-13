import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.vectorstore import VectorStoreBackend
from src.rag.embeddings import EmbeddingAdapter

logger = logging.getLogger(__name__)

CHROMA_SETTINGS = ChromaSettings(
    anonymized_telemetry=False,
    is_persistent=True,
)


class ChromaBackend(VectorStoreBackend):
    """Chroma-backed implementation of VectorStoreBackend."""

    def __init__(self, embedding: EmbeddingAdapter, persist_directory: Optional[str] = None):
        self._embedding = embedding
        self._persist_directory = persist_directory
        self._instances: dict[str, Chroma] = {}

        # Direct chromadb client for metadata queries (list/inspect without mutation)
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory, settings=CHROMA_SETTINGS)
        else:
            self._client = chromadb.Client(settings=CHROMA_SETTINGS)

    def _get_or_create(self, collection_name: str) -> Chroma:
        """Get or create a LangChain Chroma wrapper. Used only for write/search operations."""
        if collection_name not in self._instances:
            self._instances[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self._embedding.get_langchain_embeddings(),
                persist_directory=self._persist_directory,
                client_settings=CHROMA_SETTINGS,
            )
        return self._instances[collection_name]

    def store(self, documents: list[Document], collection_name: str) -> list[str]:
        db = self._get_or_create(collection_name)
        ids = db.add_documents(documents=documents)
        logger.info(f"Stored {len(ids)} documents in collection '{collection_name}'")
        return ids

    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> list[Document]:
        db = self._get_or_create(collection_name)
        return db.similarity_search(query, k=k)

    def hybrid_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> list[Document]:
        # Chroma does not support native hybrid search; fall back to dense
        logger.debug("Chroma does not support hybrid search; falling back to dense search")
        return self.search(query, collection_name, k)

    def delete(self, ids: list[str], collection_name: str) -> None:
        db = self._get_or_create(collection_name)
        db.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from collection '{collection_name}'")

    def get_collection_info(self, collection_name: str) -> dict:
        """Read-only collection inspection. Does not create collections that don't exist."""
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' not found")

        return {
            "name": collection_name,
            "backend": "chroma",
            "document_count": collection.count(),
            "persist_directory": self._persist_directory,
            "embedding_model": self._embedding.model_name,
        }

    def list_collections(self) -> list[dict]:
        """List all persisted collections, not just those touched in this process."""
        results = []
        for collection in self._client.list_collections():
            name = collection if isinstance(collection, str) else collection.name
            results.append({
                "name": name,
                "backend": "chroma",
                "document_count": self._client.get_collection(name).count(),
                "persist_directory": self._persist_directory,
                "embedding_model": self._embedding.model_name,
            })
        return results

    def count(self, collection_name: str) -> int:
        """Read-only count. Does not create collections that don't exist."""
        try:
            collection = self._client.get_collection(collection_name)
            return collection.count()
        except Exception:
            return 0

    def as_retriever(self, collection_name: str, **kwargs):
        """Return a LangChain retriever for backward compatibility."""
        db = self._get_or_create(collection_name)
        return db.as_retriever(**kwargs)
