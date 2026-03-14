import logging
import uuid
from typing import Optional

from langchain_core.documents import Document

from src.rag.vectorstore import VectorStoreBackend
from src.rag.embeddings import EmbeddingAdapter

logger = logging.getLogger(__name__)


class QdrantBackend(VectorStoreBackend):
    """Qdrant-backed implementation of VectorStoreBackend (dense retrieval only)."""

    def __init__(
        self,
        embedding: EmbeddingAdapter,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        from qdrant_client import QdrantClient

        self._embedding = embedding
        self._url = url
        self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        self._vector_size: Optional[int] = None
        self._validated_collections: set[str] = set()

    def _get_vector_size(self) -> int:
        if self._vector_size is None:
            sample = self._embedding.embed_query("dimension probe")
            self._vector_size = len(sample)
        return self._vector_size

    def _validate_collection(self, collection_name: str) -> None:
        """Validate that an existing collection's vector size matches the embedding model.

        Results are cached per collection so the check runs at most once per collection.
        """
        if collection_name in self._validated_collections:
            return
        info = self._client.get_collection(collection_name)
        existing_size = info.config.params.vectors.size
        expected_size = self._get_vector_size()
        if existing_size != expected_size:
            raise ValueError(
                f"Collection '{collection_name}' has vector size {existing_size}, "
                f"but the configured embedding model produces {expected_size}-dimensional vectors. "
                f"Reindex the collection or change the embedding model."
            )
        self._validated_collections.add(collection_name)

    def _ensure_collection_for_write(self, collection_name: str) -> None:
        """Create collection if missing; validate dimension if it exists."""
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(collection_name):
            self._validate_collection(collection_name)
        else:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._get_vector_size(),
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{collection_name}'")

    def store(self, documents: list[Document], collection_name: str) -> list[str]:
        from qdrant_client.models import PointStruct

        self._ensure_collection_for_write(collection_name)
        texts = [doc.page_content for doc in documents]
        vectors = self._embedding.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in documents]

        points = [
            PointStruct(
                id=doc_id,
                vector=vector,
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                },
            )
            for doc_id, vector, doc in zip(ids, vectors, documents)
        ]

        self._client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Stored {len(ids)} documents in Qdrant collection '{collection_name}'")
        return ids

    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> list[Document]:
        if not self._client.collection_exists(collection_name):
            return []

        self._validate_collection(collection_name)
        query_vector = self._embedding.embed_query(query)

        results = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )

        documents = []
        for point in results.points:
            payload = point.payload or {}
            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                )
            )
        return documents

    def hybrid_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> list[Document]:
        logger.debug("Qdrant hybrid search not yet enabled; falling back to dense search")
        return self.search(query, collection_name, k)

    def delete(self, ids: list[str], collection_name: str) -> None:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=ids),
        )
        logger.info(f"Deleted {len(ids)} documents from Qdrant collection '{collection_name}'")

    def get_collection_info(self, collection_name: str) -> dict:
        if not self._client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' not found")

        info = self._client.get_collection(collection_name)
        return {
            "name": collection_name,
            "backend": "qdrant",
            "document_count": info.points_count,
            "persist_directory": None,
            "embedding_model": self._embedding.model_name,
        }

    def list_collections(self) -> list[dict]:
        results = []
        for collection in self._client.get_collections().collections:
            info = self._client.get_collection(collection.name)
            results.append({
                "name": collection.name,
                "backend": "qdrant",
                "document_count": info.points_count,
                "persist_directory": None,
                "embedding_model": self._embedding.model_name,
            })
        return results

    def count(self, collection_name: str) -> int:
        if not self._client.collection_exists(collection_name):
            return 0
        info = self._client.get_collection(collection_name)
        return info.points_count
