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

_PROVENANCE_KEY_MODEL = "embedding_model_name"
_PROVENANCE_KEY_TYPE = "embedding_type"


class ChromaBackend(VectorStoreBackend):
    """Chroma-backed implementation of VectorStoreBackend."""

    def __init__(
        self,
        embedding: EmbeddingAdapter,
        persist_directory: Optional[str] = None,
        embedding_type: str = "instructor",
        allow_legacy_collections: bool = False,
    ):
        self._embedding = embedding
        self._embedding_type = embedding_type
        self._persist_directory = persist_directory
        self._allow_legacy_collections = allow_legacy_collections
        self._instances: dict[str, Chroma] = {}
        self._validated_collections: set[str] = set()

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

    def _validate_embedding(self, collection_name: str) -> None:
        """Validate that configured embedding matches stored provenance.

        Cached per collection. Logs a warning for legacy collections without provenance.
        """
        if collection_name in self._validated_collections:
            return

        provenance = self.get_embedding_provenance(collection_name)
        if provenance is None:
            if self._allow_legacy_collections:
                logger.warning(
                    f"Collection '{collection_name}' has no embedding provenance metadata. "
                    f"Cannot verify compatibility with configured embedding '{self._embedding.model_name}'. "
                    f"Allowing access because allow_legacy_collections is enabled. "
                    f"Consider reindexing to add provenance tracking."
                )
            else:
                raise ValueError(
                    f"Collection '{collection_name}' has no embedding provenance metadata. "
                    f"Cannot verify compatibility with configured embedding '{self._embedding.model_name}'. "
                    f"Reindex the collection to add provenance, or set "
                    f"vectorstore.allow_legacy_collections=true to bypass this check."
                )
        else:
            stored_model = provenance.get("model_name")
            stored_type = provenance.get("type")
            if stored_model != self._embedding.model_name:
                raise ValueError(
                    f"Collection '{collection_name}' was built with embedding model '{stored_model}' "
                    f"(type: {stored_type}), but the configured model is '{self._embedding.model_name}' "
                    f"(type: {self._embedding_type}). Reindex the collection or restore the original config."
                )
            if stored_type and stored_type != self._embedding_type:
                raise ValueError(
                    f"Collection '{collection_name}' was built with embedding type '{stored_type}', "
                    f"but the configured type is '{self._embedding_type}'. "
                    f"Reindex the collection or restore the original config."
                )

        self._validated_collections.add(collection_name)

    def _set_provenance(self, collection_name: str) -> None:
        """Store embedding provenance in Chroma collection metadata."""
        try:
            collection = self._client.get_or_create_collection(
                collection_name,
                metadata={
                    _PROVENANCE_KEY_MODEL: self._embedding.model_name,
                    _PROVENANCE_KEY_TYPE: self._embedding_type,
                },
            )
            collection.modify(
                metadata={
                    _PROVENANCE_KEY_MODEL: self._embedding.model_name,
                    _PROVENANCE_KEY_TYPE: self._embedding_type,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to set embedding provenance on '{collection_name}': {e}")

    def _collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection already exists in the Chroma store."""
        try:
            self._client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def store(self, documents: list[Document], collection_name: str) -> list[str]:
        exists = self._collection_exists(collection_name)
        db = self._get_or_create(collection_name)
        if exists:
            self._validate_embedding(collection_name)
        self._set_provenance(collection_name)
        ids = db.add_documents(documents=documents)
        logger.info(f"Stored {len(ids)} documents in collection '{collection_name}'")
        return ids

    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> list[Document]:
        try:
            self._client.get_collection(collection_name)
        except Exception:
            return []
        self._validate_embedding(collection_name)
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

        provenance = self.get_embedding_provenance(collection_name)
        embedding_model = provenance["model_name"] if provenance else "unknown"

        return {
            "name": collection_name,
            "backend": "chroma",
            "document_count": collection.count(),
            "persist_directory": self._persist_directory,
            "embedding_model": embedding_model,
        }

    def list_collections(self) -> list[dict]:
        """List all persisted collections, not just those touched in this process."""
        results = []
        for collection in self._client.list_collections():
            name = collection if isinstance(collection, str) else collection.name
            provenance = self.get_embedding_provenance(name)
            embedding_model = provenance["model_name"] if provenance else "unknown"
            results.append(
                {
                    "name": name,
                    "backend": "chroma",
                    "document_count": self._client.get_collection(name).count(),
                    "persist_directory": self._persist_directory,
                    "embedding_model": embedding_model,
                }
            )
        return results

    def count(self, collection_name: str) -> int:
        """Read-only count. Does not create collections that don't exist."""
        try:
            collection = self._client.get_collection(collection_name)
            return collection.count()
        except Exception:
            return 0

    def get_embedding_provenance(self, collection_name: str) -> Optional[dict]:
        """Return stored embedding provenance from Chroma collection metadata."""
        try:
            collection = self._client.get_collection(collection_name)
            meta = collection.metadata or {}
            model_name = meta.get(_PROVENANCE_KEY_MODEL)
            if model_name:
                return {
                    "model_name": model_name,
                    "type": meta.get(_PROVENANCE_KEY_TYPE),
                }
        except Exception:
            pass
        return None

    def find_by_source(self, source_name: str, collection_name: str) -> list[str]:
        """Return document IDs whose 'source' metadata matches the given filename."""
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            return []
        results = collection.get(where={"source": source_name}, include=[])
        return results.get("ids", [])

    def as_retriever(self, collection_name: str, **kwargs):
        """Return a LangChain retriever for backward compatibility."""
        db = self._get_or_create(collection_name)
        return db.as_retriever(**kwargs)
