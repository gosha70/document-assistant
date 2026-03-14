---
spec_mode: full
feature_id: admin-console
risk: medium
justification: >
  Multi-section UI with new backend endpoints. Touches admin routes,
  schemas, and a new static HTML file. Classified as full due to
  new endpoints + auth-gated UI spanning multiple concerns.
---

# Plan: Admin/Monitoring Console

## Requirements (confirmed from modernization plan)

1. Collections dashboard: list, inspect metadata and embedding model, trigger reindex/export/import, sample chunks, source file listing
2. Ingestion Jobs dashboard: active/completed/failed, progress, retry, cancel
3. Runtime Metrics panel: request rates, LLM latency/tokens, retrieval latency, process stats, error rates
4. Summary Reports panel: uptime, totals, error rate

## Files to Create

- `static/admin.html` -- single-page admin console (HTML+CSS+JS, dark theme)

## Files to Modify

- `src/api/routes/admin.py` -- add 2 new endpoints (sample chunks, source list)
- `src/api/schemas.py` -- add response models for new endpoints
- `src/rag/vectorstore.py` -- add abstract methods `sample_chunks` and `list_sources`
- `src/rag/chroma_backend.py` -- implement `sample_chunks` and `list_sources`
- `src/rag/qdrant_backend.py` -- implement `sample_chunks` and `list_sources` (filter provenance sentinel)

## Delegation Plan

- **Agent A (Frontend):** `static/admin.html` -- build the entire admin console UI
- **Agent B (Backend):** `src/api/routes/admin.py`, `src/api/schemas.py`, vectorstore files -- add new endpoints
- **Lead:** review, integration, test strategy

## Implementation Order

1. Backend: add new endpoints and schemas
2. Frontend: build admin.html consuming all endpoints
3. Integration test

See `spec.md` for full design details.
