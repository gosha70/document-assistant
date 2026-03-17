import logging
from langchain_core.documents import Document

from src.rag.vectorstore import VectorStoreBackend
from src.rag.reranking import Reranker

logger = logging.getLogger(__name__)


def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Remove duplicate documents based on stable backend ID.

    Uses metadata["id"] (set by backends at store time) as the dedup key.
    Falls back to page_content when no ID is present.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.metadata.get("id") or doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    if len(unique) < len(docs):
        logger.info(f"Deduplicated {len(docs)} → {len(unique)} candidates")
    return unique


class Retriever:
    """Retrieves documents from a vector store, optionally reranking results."""

    def __init__(
        self,
        backend: VectorStoreBackend,
        collection_name: str,
        reranker: Reranker | None = None,
        initial_k: int = 20,
        final_k: int = 5,
        use_hybrid: bool = True,
    ):
        self._backend = backend
        self._collection_name = collection_name
        self._reranker = reranker
        self._initial_k = initial_k
        self._final_k = final_k
        self._use_hybrid = use_hybrid

    def retrieve(self, query: str) -> list[Document]:
        if self._use_hybrid:
            docs = self._backend.hybrid_search(
                query=query,
                collection_name=self._collection_name,
                k=self._initial_k,
            )
        else:
            docs = self._backend.search(
                query=query,
                collection_name=self._collection_name,
                k=self._initial_k,
            )

        logger.info(f"Retrieved {len(docs)} candidates for query")

        docs = deduplicate_docs(docs)

        # Tag each document with search_type for eval tracking
        search_type = "hybrid" if self._use_hybrid else "dense"
        for doc in docs:
            doc.metadata["search_type"] = search_type

        if self._reranker and len(docs) > self._final_k:
            docs = self._reranker.rerank(query=query, documents=docs, top_k=self._final_k)
            logger.info(f"Reranked to {len(docs)} documents")
        else:
            docs = docs[: self._final_k]

        return docs

    def retrieve_with_vector(self, dense_embedding: list[float], query_text: str) -> list[Document]:
        """Retrieve using a pre-computed dense vector (for HyDE).

        Delegates to backend.hybrid_search_by_vector (or search_by_vector for
        dense-only). Dedup + rerank against query_text.
        """
        if self._use_hybrid:
            docs = self._backend.hybrid_search_by_vector(
                dense_embedding=dense_embedding,
                query_text=query_text,
                collection_name=self._collection_name,
                k=self._initial_k,
            )
        else:
            docs = self._backend.search_by_vector(
                embedding=dense_embedding,
                collection_name=self._collection_name,
                k=self._initial_k,
            )

        logger.info(f"Retrieved {len(docs)} candidates via pre-computed vector")

        docs = deduplicate_docs(docs)

        search_type = "hyde_hybrid" if self._use_hybrid else "hyde_dense"
        for doc in docs:
            doc.metadata["search_type"] = search_type

        if self._reranker and len(docs) > self._final_k:
            docs = self._reranker.rerank(query=query_text, documents=docs, top_k=self._final_k)
            logger.info(f"Reranked to {len(docs)} documents")
        else:
            docs = docs[: self._final_k]

        return docs

    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        return deduplicate_docs(docs)
