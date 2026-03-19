# ADR-003: Pluggable Vector Store Backend (Chroma + Qdrant)

**Status:** Accepted
**Date:** 2024
**Deciders:** @gosha70

---

## Context

The original application used ChromaDB exclusively. While Chroma is excellent for local development (no external service required, in-process or persistent file-based), it has limitations at production scale:

- No native horizontal scaling or replication
- No built-in access control or multi-tenancy
- Performance degrades with very large collections (millions of vectors)
- No built-in support for sparse + dense hybrid vectors

Production deployments often require a vector store with:

- Client-server architecture (separate service, not embedded)
- gRPC support for high-throughput ingestion
- Native sparse + dense vector storage for hybrid search
- Managed cloud option or self-hosted cluster support

At the same time, local development workflows benefit from zero-configuration startup — no Docker required, no separate process.

## Decision

Define a **`VectorStoreBackend` abstract base class** and provide two implementations:

| Backend | Best for |
|---------|----------|
| `ChromaBackend` | Local development, single-node deployment, zero-config startup |
| `QdrantBackend` | Production, high-volume collections, gRPC, native hybrid vectors |

The active backend is selected via `config/local.yaml`:

```yaml
vectorstore:
  backend: "chroma"   # or "qdrant"
```

All application code (routes, orchestrator, admin) targets only the `VectorStoreBackend` interface — it has no knowledge of which concrete backend is in use.

### VectorStoreBackend Interface

```python
class VectorStoreBackend(ABC):
    def store(documents, collection_name) -> list[str]
    def search(query, collection_name, k) -> list[Document]
    def hybrid_search(query, collection_name, k) -> list[Document]
    def delete(ids, collection_name) -> None
    def get_collection_info(collection_name) -> dict
    def list_collections() -> list[dict]
    def count(collection_name) -> int
    def find_by_source(source_name, collection_name) -> list[str]
    def get_embedding_provenance(collection_name) -> dict | None
    def sample_chunks(collection_name, limit) -> dict
    def list_sources(collection_name) -> dict
```

### Hybrid Search Divergence

Each backend implements hybrid search differently:

- **Chroma**: BM25 index in a pickle file + dense search → RRF fusion in application layer (`src/rag/fusion.py`)
- **Qdrant**: Dense and sparse vectors stored as named Qdrant vector fields; Qdrant performs score fusion natively on its side

### Embedding Provenance

Both backends stamp embedding provenance metadata (model name, type, embedding strategy) on every collection. This enables:

1. Detection of model mismatches when loading an existing collection with a different embedding configuration
2. Safe migration of legacy collections via the `/admin/collections/{name}/migrate` endpoint

## Rationale

An abstract backend was preferred over:

- **Hard-coding Chroma everywhere** — would require large-scale refactoring when adding Qdrant
- **Runtime branching** — scatters `if backend == "chroma" else ...` conditionals across the codebase, making each addition more fragile
- **Plugin architecture** — overcomplicated for two backends; simple ABC is sufficient

## Consequences

**Positive:**
- Zero-change upgrade path: switch from Chroma to Qdrant by changing one config line
- Backend-specific optimisations (Qdrant gRPC, Qdrant native sparse vectors) are encapsulated in each implementation
- New backends (Pinecone, Weaviate, pgvector) can be added without touching the rest of the application

**Negative:**
- Each backend must implement the full interface — missing methods raise `NotImplementedError` at runtime
- Tests primarily cover Chroma; Qdrant integration tests require a running Qdrant server
- Hybrid search quality may differ slightly between backends (application-layer RRF vs. Qdrant-native fusion)

## Configuration

```yaml
# Chroma (dev/local)
vectorstore:
  backend: "chroma"
  persist_directory: "./vectorstore"

# Qdrant (production)
vectorstore:
  backend: "qdrant"
  qdrant_url: "http://localhost:6333"
  qdrant_api_key: null          # set for Qdrant Cloud
  qdrant_prefer_grpc: false     # set true for high-throughput ingestion
```
