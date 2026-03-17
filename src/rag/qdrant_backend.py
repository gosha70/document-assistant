import logging
import uuid
from typing import Optional

from langchain_core.documents import Document

from src.rag.vectorstore import VectorStoreBackend
from src.rag.embeddings import EmbeddingAdapter

logger = logging.getLogger(__name__)

_PROVENANCE_KEY_MODEL = "embedding_model_name"
_PROVENANCE_KEY_TYPE = "embedding_type"


_SPARSE_VOCAB_POINT_ID = "00000000-0000-0000-0000-000000000001"


class QdrantBackend(VectorStoreBackend):
    """Qdrant-backed implementation of VectorStoreBackend with optional hybrid search."""

    def __init__(
        self,
        embedding: EmbeddingAdapter,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        embedding_type: str = "instructor",
        allow_legacy_collections: bool = False,
        hybrid_settings: Optional[dict] = None,
    ):
        from qdrant_client import QdrantClient

        self._embedding = embedding
        self._embedding_type = embedding_type
        self._allow_legacy_collections = allow_legacy_collections
        self._url = url
        self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        self._vector_size: Optional[int] = None
        self._validated_collections: set[str] = set()

        # Hybrid search config
        self._hybrid_enabled = bool(hybrid_settings and hybrid_settings.get("enabled", False))
        self._hybrid_rrf_k = int(hybrid_settings.get("rrf_k", 60)) if hybrid_settings else 60

        # Per-collection state: vector format and token encoder
        self._collection_vector_format: dict[str, str] = {}  # "single" or "named"
        self._token_encoders: dict = {}  # collection_name -> TokenEncoder, lazy-loaded

    def _get_vector_size(self) -> int:
        if self._vector_size is None:
            sample = self._embedding.embed_query("dimension probe")
            self._vector_size = len(sample)
        return self._vector_size

    def _detect_vector_format(self, collection_name: str, info=None) -> str:
        """Detect whether collection uses single (legacy) or named vectors.

        Returns "single" or "named". Result is cached. Accepts optional pre-fetched
        collection info to avoid an extra get_collection call.
        """
        if collection_name in self._collection_vector_format:
            return self._collection_vector_format[collection_name]

        if info is None:
            info = self._client.get_collection(collection_name)
        vectors_config = info.config.params.vectors

        # Named vectors: vectors_config is a dict-like mapping
        if isinstance(vectors_config, dict):
            fmt = "named"
        else:
            fmt = "single"

        self._collection_vector_format[collection_name] = fmt
        return fmt

    def _validate_collection(self, collection_name: str) -> None:
        """Validate dimension and embedding provenance against existing collection.

        Results are cached per collection so checks run at most once.
        """
        if collection_name in self._validated_collections:
            return

        info = self._client.get_collection(collection_name)
        fmt = self._detect_vector_format(collection_name, info=info)

        # Dimension check — adapt for single vs named vectors
        if fmt == "named":
            dense_config = info.config.params.vectors.get("dense")
            existing_size = dense_config.size if dense_config else None
        else:
            existing_size = info.config.params.vectors.size

        expected_size = self._get_vector_size()
        if existing_size and existing_size != expected_size:
            raise ValueError(
                f"Collection '{collection_name}' has vector size {existing_size}, "
                f"but the configured embedding model produces {expected_size}-dimensional vectors. "
                f"Reindex the collection or change the embedding model."
            )

        # Log warning if hybrid enabled but collection is legacy single-vector
        if fmt == "single" and self._hybrid_enabled:
            logger.warning(
                f"Collection '{collection_name}' uses legacy single-vector format. "
                f"Hybrid search disabled for this collection; falling back to dense-only."
            )

        # Provenance check
        provenance = self.get_embedding_provenance(collection_name)
        if provenance is None:
            if self._allow_legacy_collections:
                logger.warning(
                    f"Collection '{collection_name}' has no embedding provenance metadata. "
                    f"Cannot verify compatibility with configured embedding '{self._embedding.model_name}'. "
                    f"Allowing access because allow_legacy_collections is enabled."
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
            if stored_model and stored_model != self._embedding.model_name:
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

    def _ensure_collection_for_write(self, collection_name: str) -> None:
        """Create collection if missing; validate if it exists."""
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(collection_name):
            self._validate_collection(collection_name)
        else:
            dim = self._get_vector_size()
            if self._hybrid_enabled:
                from qdrant_client.models import SparseIndexParams, SparseVectorParams, Modifier

                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(size=dim, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                            modifier=Modifier.IDF,
                        ),
                    },
                )
                self._collection_vector_format[collection_name] = "named"
                logger.info(f"Created Qdrant collection '{collection_name}' (named vectors + sparse)")
            else:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                self._collection_vector_format[collection_name] = "single"
                logger.info(f"Created Qdrant collection '{collection_name}'")

    def _set_provenance(self, collection_name: str) -> None:
        """Store embedding provenance as a sentinel point in the collection."""
        try:
            from qdrant_client.models import PointStruct

            provenance_id = "00000000-0000-0000-0000-000000000000"
            fmt = self._collection_vector_format.get(collection_name, "single")
            dim = self._get_vector_size()

            if fmt == "named":
                vector = {"dense": [0.0] * dim}
            else:
                vector = [0.0] * dim

            self._client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=provenance_id,
                        vector=vector,
                        payload={
                            "_provenance": True,
                            _PROVENANCE_KEY_MODEL: self._embedding.model_name,
                            _PROVENANCE_KEY_TYPE: self._embedding_type,
                        },
                    )
                ],
            )
        except Exception as e:
            logger.warning(f"Failed to set embedding provenance on '{collection_name}': {e}")

    def _get_token_encoder(self, collection_name: str):
        """Load or create a TokenEncoder for the collection.

        Vocabulary is stored in the Qdrant collection itself as a metadata point.
        """
        if collection_name in self._token_encoders:
            return self._token_encoders[collection_name]

        from src.rag.sparse import TokenEncoder

        vocab = self._load_sparse_vocab(collection_name)
        encoder = TokenEncoder(vocab=vocab)
        self._token_encoders[collection_name] = encoder
        return encoder

    def _load_sparse_vocab(self, collection_name: str) -> dict[str, int]:
        """Load sparse vocabulary from the collection's metadata point."""
        try:
            points = self._client.retrieve(
                collection_name=collection_name,
                ids=[_SPARSE_VOCAB_POINT_ID],
                with_payload=True,
            )
            if points:
                payload = points[0].payload or {}
                return payload.get("vocab", {})
        except Exception:
            pass
        return {}

    def _save_sparse_vocab(self, collection_name: str, vocab: dict[str, int]) -> None:
        """Save sparse vocabulary as a metadata point in the collection."""
        from qdrant_client.models import PointStruct

        fmt = self._collection_vector_format.get(collection_name, "single")
        dim = self._get_vector_size()

        if fmt == "named":
            vector = {"dense": [0.0] * dim}
        else:
            vector = [0.0] * dim

        self._client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=_SPARSE_VOCAB_POINT_ID,
                    vector=vector,
                    payload={
                        "_sparse_vocab": True,
                        "vocab": vocab,
                    },
                )
            ],
        )

    def store(self, documents: list[Document], collection_name: str) -> list[str]:
        from qdrant_client.models import PointStruct

        self._ensure_collection_for_write(collection_name)
        self._set_provenance(collection_name)
        texts = [doc.metadata.get("embedding_text") or doc.page_content for doc in documents]
        vectors = self._embedding.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in documents]

        fmt = self._collection_vector_format.get(collection_name, "single")
        use_sparse = self._hybrid_enabled and fmt == "named"

        if use_sparse:
            encoder = self._get_token_encoder(collection_name)
            sparse_vectors = [encoder.encode_tf(text) for text in texts]

        points = []
        for i, (doc_id, dense_vec, doc) in enumerate(zip(ids, vectors, documents)):
            if use_sparse:
                from qdrant_client.models import SparseVector as QdrantSparseVector

                sv = sparse_vectors[i]
                vector = {
                    "dense": dense_vec,
                    "sparse": QdrantSparseVector(indices=sv.indices, values=sv.values),
                }
            else:
                vector = dense_vec

            points.append(
                PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
            )

        self._client.upsert(collection_name=collection_name, points=points)

        # Persist updated vocabulary to collection
        if use_sparse:
            self._save_sparse_vocab(collection_name, encoder.vocab)

        logger.info(f"Stored {len(ids)} documents in Qdrant collection '{collection_name}'")
        return ids

    def _point_to_document(self, point) -> Optional[Document]:
        """Convert a Qdrant point to a Document, skipping sentinel points."""
        payload = point.payload or {}
        if payload.get("_provenance") or payload.get("_sparse_vocab"):
            return None
        metadata = dict(payload.get("metadata", {}))
        metadata["id"] = str(point.id)
        return Document(
            page_content=payload.get("page_content", ""),
            metadata=metadata,
        )

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

        fmt = self._collection_vector_format.get(collection_name, "single")
        query_arg = query_vector
        if fmt == "named":
            from qdrant_client.models import NamedVector

            query_arg = NamedVector(name="dense", vector=query_vector)

        results = self._client.query_points(
            collection_name=collection_name,
            query=query_arg,
            limit=k + 2,  # extra to account for sentinel filtering
            with_payload=True,
        )

        documents = []
        for point in results.points:
            doc = self._point_to_document(point)
            if doc is not None:
                documents.append(doc)
            if len(documents) >= k:
                break
        return documents

    def hybrid_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
    ) -> list[Document]:
        if not self._client.collection_exists(collection_name):
            return []

        self._validate_collection(collection_name)
        fmt = self._collection_vector_format.get(collection_name, "single")

        if not self._hybrid_enabled or fmt != "named":
            return self.search(query, collection_name, k)

        from qdrant_client.models import Prefetch, Fusion, FusionQuery, NamedVector, NamedSparseVector
        from qdrant_client.models import SparseVector as QdrantSparseVector

        query_vector = self._embedding.embed_query(query)
        encoder = self._get_token_encoder(collection_name)
        sparse_vec = encoder.encode_query_tf(query)

        try:
            results = self._client.query_points(
                collection_name=collection_name,
                prefetch=[
                    Prefetch(
                        query=NamedVector(name="dense", vector=query_vector),
                        limit=k * 2,
                    ),
                    Prefetch(
                        query=NamedSparseVector(
                            name="sparse",
                            vector=QdrantSparseVector(
                                indices=sparse_vec.indices,
                                values=sparse_vec.values,
                            ),
                        ),
                        limit=k * 2,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=k + 2,
                with_payload=True,
            )
        except Exception as e:
            logger.warning(f"Qdrant hybrid query failed, falling back to dense: {e}")
            return self.search(query, collection_name, k)

        documents = []
        for point in results.points:
            doc = self._point_to_document(point)
            if doc is not None:
                documents.append(doc)
            if len(documents) >= k:
                break
        return documents

    def delete(self, ids: list[str], collection_name: str) -> None:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=ids),
        )
        logger.info(f"Deleted {len(ids)} documents from Qdrant collection '{collection_name}'")

    def _document_count(self, collection_name: str, raw_points_count: int) -> int:
        """Return document count, subtracting sentinel points (provenance + sparse vocab)."""
        sentinels = 0
        provenance = self.get_embedding_provenance(collection_name)
        if provenance is not None:
            sentinels += 1
        # Check for sparse vocab sentinel
        try:
            points = self._client.retrieve(
                collection_name=collection_name,
                ids=[_SPARSE_VOCAB_POINT_ID],
                with_payload=False,
            )
            if points:
                sentinels += 1
        except Exception:
            pass
        return max(0, raw_points_count - sentinels)

    def find_by_source(self, source_name: str, collection_name: str) -> list[str]:
        """Return document IDs whose 'source' metadata matches the given filename."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if not self._client.collection_exists(collection_name):
            return []

        ids = []
        offset = None
        while True:
            results = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="metadata.source", match=MatchValue(value=source_name)),
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
            )
            points, next_offset = results
            for point in points:
                ids.append(str(point.id))
            if next_offset is None:
                break
            offset = next_offset
        return ids

    def get_collection_info(self, collection_name: str) -> dict:
        if not self._client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' not found")

        info = self._client.get_collection(collection_name)
        provenance = self.get_embedding_provenance(collection_name)
        embedding_model = provenance["model_name"] if provenance else "unknown"

        return {
            "name": collection_name,
            "backend": "qdrant",
            "document_count": self._document_count(collection_name, info.points_count),
            "persist_directory": None,
            "embedding_model": embedding_model,
        }

    def list_collections(self) -> list[dict]:
        results = []
        for collection in self._client.get_collections().collections:
            info = self._client.get_collection(collection.name)
            provenance = self.get_embedding_provenance(collection.name)
            embedding_model = provenance["model_name"] if provenance else "unknown"
            results.append(
                {
                    "name": collection.name,
                    "backend": "qdrant",
                    "document_count": self._document_count(collection.name, info.points_count),
                    "persist_directory": None,
                    "embedding_model": embedding_model,
                }
            )
        return results

    def count(self, collection_name: str) -> int:
        if not self._client.collection_exists(collection_name):
            return 0
        info = self._client.get_collection(collection_name)
        return self._document_count(collection_name, info.points_count)

    def sample_chunks(self, collection_name: str, limit: int = 10, offset: int = 0) -> dict:
        """Return a sample of chunks with text truncated to 500 characters.

        Filters out the provenance sentinel point and adjusts total_count accordingly.
        """
        if not self._client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' not found")

        info = self._client.get_collection(collection_name)
        raw_count = info.points_count or 0
        total_count = self._document_count(collection_name, raw_count)

        # Request extra to account for sentinels possibly consuming slots
        results, _next_offset = self._client.scroll(
            collection_name=collection_name,
            limit=limit + 2,
            offset=offset,
            with_payload=True,
        )

        chunks = []
        for point in results:
            payload = point.payload or {}
            if payload.get("_provenance") or payload.get("_sparse_vocab"):
                continue
            if len(chunks) >= limit:
                break
            text = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            chunks.append(
                {
                    "id": str(point.id),
                    "text": text[:500],
                    "metadata": metadata,
                }
            )

        return {"total_count": total_count, "chunks": chunks}

    def list_sources(self, collection_name: str) -> dict:
        """Return unique source filenames and chunk counts. Scans at most 10k chunks.

        Filters out the provenance sentinel point.
        """
        if not self._client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' not found")

        total = self.count(collection_name)
        scan_limit = 10000
        source_counts: dict[str, int] = {}
        real_scanned = 0
        offset = None

        while real_scanned < scan_limit:
            batch_limit = min(100, scan_limit - real_scanned + 1)
            results, next_offset = self._client.scroll(
                collection_name=collection_name,
                limit=batch_limit,
                offset=offset,
                with_payload=True,
            )
            if not results:
                break
            for point in results:
                payload = point.payload or {}
                if payload.get("_provenance") or payload.get("_sparse_vocab"):
                    continue
                real_scanned += 1
                if real_scanned > scan_limit:
                    break
                metadata = payload.get("metadata", {})
                source = metadata.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
            if next_offset is None:
                break
            offset = next_offset

        return {
            "sources": [{"filename": source, "chunk_count": count} for source, count in sorted(source_counts.items())],
            "scanned_chunks": real_scanned,
            "truncated": total > real_scanned,
        }

    def get_embedding_provenance(self, collection_name: str) -> Optional[dict]:
        """Return stored embedding provenance from sentinel point."""
        try:
            provenance_id = "00000000-0000-0000-0000-000000000000"
            points = self._client.retrieve(
                collection_name=collection_name,
                ids=[provenance_id],
                with_payload=True,
            )
            if points:
                payload = points[0].payload or {}
                model_name = payload.get(_PROVENANCE_KEY_MODEL)
                if model_name:
                    return {
                        "model_name": model_name,
                        "type": payload.get(_PROVENANCE_KEY_TYPE),
                    }
        except Exception:
            pass
        return None
