# ADR-004: Composable Query Pipeline (HyDE · Decomposition · Corrective Retrieval · Verification)

**Status:** Accepted
**Date:** 2024
**Deciders:** @gosha70

---

## Context

Basic RAG — embed query, retrieve top-k, generate — works well for simple factual questions but degrades on:

- **Complex multi-part questions** — a single embedding may not represent all sub-topics equally
- **Abstract or unfamiliar queries** — the query embedding may not align well with the embedding space of the documents
- **Low-quality retrieval** — the top-k results may be marginally relevant, causing hallucinations
- **Unverified answers** — the LLM may synthesise claims not supported by the retrieved context

Advanced RAG techniques address each failure mode but add latency and LLM API cost. For a local-first tool where these costs vary widely by deployment (GPU speed, Ollama model size), each technique must be independently opt-in.

## Decision

Implement a **`QueryOrchestrator`** that executes a configurable pipeline of stages. Each stage is:

- **Independently enabled** via config (`query_pipeline.*_enabled`)
- **Additive** — disabling a stage has zero overhead and falls back to baseline behaviour
- **Metadata-annotated** — when a stage fires, its effect is recorded in the response metadata

### Pipeline Stages

```
1. Query Decomposition  (query_pipeline.decomposition_enabled)
2. HyDE                 (query_pipeline.hyde_enabled)
3. Hybrid Retrieval + CrossEncoder Rerank  (always active)
4. Corrective Retrieval  (query_pipeline.corrective_retrieval_enabled)
5. Context Firewall      (always active)
6. LLM Generation        (always active)
7. Answer Verification   (query_pipeline.verification_enabled)
```

### Stage Details

**Query Decomposition**

Complex query → LLM prompt (`prompts/query_decomposition.yaml`) → up to `max_sub_queries` (5) sub-queries → retrieve independently for each → merge and deduplicate results → rerank merged set against original query.

Addresses: multi-part questions, queries spanning multiple document topics.

**Hypothetical Document Embeddings (HyDE)**

Query → LLM prompt (`prompts/hyde.yaml`) → hypothetical answer passage → embed the passage → retrieve by passage embedding (instead of query embedding). The intuition: the embedding of a plausible answer is closer to the embedding of relevant document passages than the embedding of the original question.

Addresses: semantic gap between short query embeddings and longer document chunk embeddings. LRU cache on the hypothetical passage avoids duplicate LLM calls for repeated queries.

**Corrective Retrieval**

After reranking, compute retrieval confidence:

```
confidence = σ(mean reranker_score) = 1 / (1 + exp(-mean_score))
```

If `confidence < corrective_retrieval_threshold` (default 0.4), the retrieval is considered low-quality and a corrective re-retrieval is triggered (with adjusted strategy or query variant).

Addresses: cases where the initial retrieval returns marginally relevant or irrelevant results.

**Answer Verification**

After generation, send the answer and retrieved context to the LLM (`prompts/verification.yaml`) for self-evaluation. The LLM identifies any claims in the answer that are not supported by the context and returns:

```json
{
  "verified": false,
  "unsupported_claims": ["The document was published in 2019."],
  "revised_answer": "..."
}
```

Addresses: hallucination detection and self-correction. Adds one additional LLM call per query when enabled.

### Streaming Path

The `/chat/stream` route runs a parallel streaming variant (`run_stream()`): tokens are yielded as they are produced by the LLM, and source extraction / metadata enrichment is performed after the stream completes. Verification **is** applied in the streaming path — it runs post-stream, after all tokens have been collected. The SSE event sequence for `/chat/stream` is:

```
{"type": "token",        "content": "<token>"}   # one per token
{"type": "sources",      "sources": [...]}
{"type": "verification", "result": {...}}         # only if verification enabled
{"type": "metadata",     "data": {...}}           # pipeline flags (decomposed_queries, hyde_used, etc.)
[DONE]
```

The `verification` result arrives as its own dedicated event type, before the `metadata` event which carries only pipeline flags (decomposed queries, HyDE, corrective retrieval).

## Rationale

A composable pipeline was preferred over:

- **Always-on features** — HyDE and verification each require an extra LLM call; on slow local hardware this could triple response latency
- **A/B routing** — splitting the pipeline by feature set would produce code duplication and complicate maintenance
- **Plugin/chain composition frameworks** (LangChain Expression Language) — adds a dependency and abstraction layer for what is essentially `if enabled: do_thing()`

Each feature is implemented as a function called conditionally by the orchestrator. This keeps each stage testable in isolation.

## Consequences

**Positive:**
- Users can turn on individual features to measure their quality-latency tradeoff on their specific data
- Each stage's activation is visible in the response `metadata` field — easy to observe what the pipeline did
- Baseline performance (all features off) is identical to simple RAG — no overhead from disabled stages
- New pipeline stages can be added by implementing a function and adding an `if config.xxx_enabled:` branch

**Negative:**
- HyDE and verification each require an extra LLM call — on slow hardware with a large model, this can significantly increase response time
- Decomposition with 5 sub-queries multiplies retrieval calls by 5x
- Feature interactions (e.g., HyDE + decomposition) are not explicitly tested and may produce unexpected combinations
- Corrective retrieval threshold (0.4) is a heuristic — requires tuning per dataset

## Configuration

```yaml
query_pipeline:
  decomposition_enabled: false
  hyde_enabled: false
  corrective_retrieval_enabled: false
  corrective_retrieval_threshold: 0.4
  verification_enabled: false
  hyde_cache_size: 100       # LRU cache for hypothetical passages
  max_sub_queries: 5
```

## Response Metadata

When enabled, each stage annotates the response:

```json
{
  "metadata": {
    "decomposed_queries": ["What is X?", "When was Y introduced?"],
    "hyde_used": true,
    "retrieval_confidence": 0.62,
    "corrective_triggered": false,
    "verification": {
      "verified": true,
      "unsupported_claims": [],
      "revised_answer": null
    }
  }
}
```
