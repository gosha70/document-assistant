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
        hybrid_settings: Optional[dict] = None,
    ):
        self._embedding = embedding
        self._embedding_type = embedding_type
        self._persist_directory = persist_directory
        self._allow_legacy_collections = allow_legacy_collections
        self._instances: dict[str, Chroma] = {}
        self._validated_collections: set[str] = set()

        # Hybrid search config
        self._hybrid_enabled = bool(hybrid_settings and hybrid_settings.get("enabled", False))
        self._hybrid_rrf_k = int(hybrid_settings.get("rrf_k", 60)) if hybrid_settings else 60
        self._bm25_indexes: dict = {}  # collection_name -> BM25Index, lazy-loaded

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

    def _get_bm25_index(self, collection_name: str):
        """Lazy-load or create BM25Index for a collection."""
        if collection_name in self._bm25_indexes:
            return self._bm25_indexes[collection_name]

        from src.rag.bm25_index import BM25Index

        if self._persist_directory:
            pkl_path = f"{self._persist_directory}/.bm25_{collection_name}.pkl"
            idx = BM25Index.load(pkl_path)
        else:
            idx = BM25Index(persist_path=None)

        # Rebuild from Chroma if index is empty but collection has data
        if idx.size == 0:
            try:
                collection = self._client.get_collection(collection_name)
                count = collection.count()
                if count > 0:
                    results = collection.get(limit=count, include=["documents"])
                    doc_ids = results.get("ids") or []
                    docs = results.get("documents") or []
                    idx.rebuild(doc_ids, docs)
                    idx.save()
            except Exception:
                pass

        self._bm25_indexes[collection_name] = idx
        return idx

    def store(self, documents: list[Document], collection_name: str) -> list[str]:
        exists = self._collection_exists(collection_name)
        db = self._get_or_create(collection_name)
        if exists:
            self._validate_embedding(collection_name)
        self._set_provenance(collection_name)
        ids = db.add_documents(documents=documents)
        logger.info(f"Stored {len(ids)} documents in collection '{collection_name}'")

        # Update BM25 side-index
        if self._hybrid_enabled:
            bm25 = self._get_bm25_index(collection_name)
            bm25.add(ids, [doc.page_content for doc in documents])
            bm25.save()

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
        if not self._hybrid_enabled:
            return self.search(query, collection_name, k)

        try:
            self._client.get_collection(collection_name)
        except Exception:
            return []
        self._validate_embedding(collection_name)

        # Dense search via raw chromadb client to get IDs back
        collection = self._client.get_collection(collection_name)
        lc_emb = self._embedding.get_langchain_embeddings()
        query_embedding = lc_emb.embed_query(query)
        dense_raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,
            include=["documents", "metadatas"],
        )

        dense_results = []
        for doc_id, text, meta in zip(
            dense_raw.get("ids", [[]])[0],
            dense_raw.get("documents", [[]])[0],
            dense_raw.get("metadatas", [[]])[0],
        ):
            m = dict(meta or {})
            m["id"] = doc_id
            dense_results.append(Document(page_content=text or "", metadata=m))

        # BM25 search: k*2 candidates
        bm25 = self._get_bm25_index(collection_name)
        bm25_hits = bm25.search(query, k=k * 2)

        if not bm25_hits:
            return dense_results[:k]

        # Look up full documents from Chroma for BM25 results
        bm25_ids = [doc_id for doc_id, _ in bm25_hits]
        bm25_raw = collection.get(ids=bm25_ids, include=["documents", "metadatas"])

        sparse_results = []
        raw_ids = bm25_raw.get("ids") or []
        raw_docs = bm25_raw.get("documents") or []
        raw_metas = bm25_raw.get("metadatas") or []
        for doc_id, text, meta in zip(raw_ids, raw_docs, raw_metas):
            m = dict(meta or {})
            m["id"] = doc_id
            sparse_results.append(Document(page_content=text or "", metadata=m))

        from src.rag.fusion import rrf_fuse

        fused = rrf_fuse(dense_results, sparse_results, k=self._hybrid_rrf_k, top_n=k)
        return fused

    def delete(self, ids: list[str], collection_name: str) -> None:
        db = self._get_or_create(collection_name)
        db.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from collection '{collection_name}'")

        if self._hybrid_enabled and collection_name in self._bm25_indexes:
            self._bm25_indexes[collection_name].delete(ids)
            self._bm25_indexes[collection_name].save()

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

    def sample_chunks(self, collection_name: str, limit: int = 10, offset: int = 0) -> dict:
        """Return a sample of chunks with text truncated to 500 characters."""
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' not found")

        total_count = collection.count()
        results = collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )

        chunks = []
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            chunks.append(
                {
                    "id": chunk_id,
                    "text": (text or "")[:500],
                    "metadata": metadata or {},
                }
            )

        return {"total_count": total_count, "chunks": chunks}

    def list_sources(self, collection_name: str) -> dict:
        """Return unique source filenames and chunk counts. Scans at most 10k chunks."""
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' not found")

        total = collection.count()
        scan_limit = 10000
        results = collection.get(limit=scan_limit, include=["metadatas"])
        metadatas = results.get("metadatas") or []
        scanned = len(metadatas)

        source_counts: dict[str, int] = {}
        for metadata in metadatas:
            source = (metadata or {}).get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "sources": [{"filename": source, "chunk_count": count} for source, count in sorted(source_counts.items())],
            "scanned_chunks": scanned,
            "truncated": total > scan_limit,
        }

    def as_retriever(self, collection_name: str, **kwargs):
        """Return a LangChain retriever for backward compatibility."""
        db = self._get_or_create(collection_name)
        return db.as_retriever(**kwargs)
