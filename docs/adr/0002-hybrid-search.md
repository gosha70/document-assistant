# ADR-002: Hybrid Search — Dense + BM25 + Reciprocal Rank Fusion

**Status:** Accepted
**Date:** 2024
**Deciders:** @gosha70

---

## Context

The original retrieval strategy used dense vector similarity search only: embed the query, find nearest neighbours in the vector store. This works well for semantic matching but has known failure modes:

- **Lexical precision** — if a user asks about a specific term, product name, identifier, or code symbol, dense search may retrieve semantically related but lexically different documents
- **Out-of-distribution queries** — dense models trained on general corpora can underperform on specialised technical or domain-specific terminology
- **Keyword anchoring** — users often include specific keywords they expect to appear verbatim in the result

BM25 (Best Match 25) is a classic sparse retrieval algorithm based on term frequency and inverse document frequency. It excels exactly where dense search is weak: exact keyword matching and rare-term retrieval.

## Decision

Implement **hybrid search** combining dense and sparse retrieval results via **Reciprocal Rank Fusion (RRF)**:

1. **Dense search** — top-20 results by embedding similarity
2. **BM25 sparse search** — top-20 results by term overlap
3. **RRF fusion** — merge both ranked lists into a single ranking

RRF score formula:

```
score(d) = Σ  1 / (k + rank_i(d))
           i
```

Where `k` is a smoothing constant (default 60) and `rank_i(d)` is the document's position in retrieval list `i`. Documents appearing in both lists receive additive scores from both components.

After fusion, the top-20 merged candidates are passed to the CrossEncoder reranker, which selects the final top-5.

### Implementation

- **Chroma backend** — BM25 index persisted as a pickle file alongside the vector store; lazily loaded on first query; filename derived from `sha256(collection_name)[:32]` to prevent path injection
- **Qdrant backend** — sparse vectors stored as named vectors alongside dense vectors; query uses both vector types
- **Enabled by default** — `vectorstore.hybrid.enabled: true`
- **Configurable k** — `vectorstore.hybrid.rrf_k` (default 60)

## Rationale

| Criterion | Dense Only | BM25 Only | Hybrid RRF |
|-----------|-----------|----------|-----------|
| Semantic matching | Excellent | Poor | Excellent |
| Exact keyword matching | Weak | Excellent | Excellent |
| Rare-term retrieval | Weak | Excellent | Excellent |
| Implementation complexity | Low | Medium | Medium-High |
| Latency overhead | Baseline | +BM25 search | +BM25 + fusion |

RRF was chosen over learned fusion (e.g., linear combination of scores) because:

- Requires no training data or calibration
- Score scales differ between dense cosine similarity and BM25 TF-IDF — RRF uses only rank, making it scale-invariant
- Robust default behaviour; the `k` parameter is easy to tune

## Consequences

**Positive:**
- Significantly improves recall for keyword-heavy queries and technical terms
- Handles both "find documents about concept X" (dense wins) and "find documents mentioning function Y" (BM25 wins) well
- No additional embedding model calls at query time

**Negative:**
- BM25 index must be rebuilt or updated incrementally when documents are ingested — currently rebuilt lazily on first use after ingestion
- BM25 pickle file adds filesystem state alongside the vector store
- Sparse search adds latency (~5-20ms for typical collection sizes)
- BM25 does not benefit from late chunking embeddings — the two strategies are applied independently

## Configuration

```yaml
vectorstore:
  hybrid:
    enabled: true          # set false to use dense-only
    sparse_encoder: "bm25"
    rrf_k: 60              # increase for more smoothing; 20-80 typical range
```
