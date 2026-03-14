# SDD: Admin/Monitoring Console

## 1. Overview

Build a single-page admin console at `/static/admin.html` for the D.O.T. Document Assistant. The console provides operational visibility into collections, ingestion jobs, runtime metrics, and summary reports. It follows the same pattern as the chat UI: a single HTML file with inline CSS and JS, dark theme, no build tooling, served by FastAPI's static files mount.

### Goals

- Give the operator a single dashboard to monitor and manage the D.O.T. instance
- Reuse existing backend APIs wherever possible; add only the minimum new endpoints
- Match the chat UI's dark theme aesthetic (background `#121212`, cards `#1F1F1F`, accent `#6366F1`, danger `#EF4444`)
- Support admin auth via Bearer token stored in localStorage (same mechanism as chat UI)

### Non-Goals

- Real-time WebSocket streaming (polling is sufficient for a local-first tool)
- Historical time-series data or charts (no persistent metrics store exists yet)
- Enterprise features: RBAC, audit logs, multi-tenant dashboards

---

## 2. UI Layout and Navigation

The admin console uses a **tab-based layout** with four top-level sections. A fixed header contains the D.O.T. logo, title "D.O.T. Admin", and a settings gear icon (for token configuration). Below the header, a horizontal tab bar switches between sections. The active tab's content fills the remaining viewport.

```
+------------------------------------------------------------+
| [logo] D.O.T. Admin                              [gear]    |
+------------------------------------------------------------+
| [Collections] [Jobs] [Metrics] [Reports]                   |
+------------------------------------------------------------+
|                                                            |
|              (active tab content area)                     |
|                                                            |
+------------------------------------------------------------+
| footer                                                     |
+------------------------------------------------------------+
```

### Color Palette (matching chat UI)

| Token            | Value     | Usage                        |
|------------------|-----------|------------------------------|
| `--bg-base`      | `#121212` | Page background              |
| `--bg-card`      | `#1F1F1F` | Cards, panels                |
| `--bg-input`     | `#2D2D2D` | Input fields, table rows alt |
| `--border`       | `#3F3F46` | Borders, dividers            |
| `--text-primary` | `#E4E4E7` | Primary text                 |
| `--text-muted`   | `#999`    | Secondary text, labels       |
| `--accent`       | `#6366F1` | Active tab, buttons, badges  |
| `--danger`       | `#EF4444` | Error states, cancel buttons |
| `--success`      | `#064e3b` | Success badges (bg)          |
| `--success-text` | `#6ee7b7` | Success badge text           |
| `--warning`      | `#ca8a04` | In-progress states           |

---

## 3. Section Details

### 3.1 Collections Tab

**Purpose:** List all vector store collections, inspect their metadata and embedding model, trigger operations, and inspect sample chunks.

#### Display

A card-grid or table showing all collections. Each collection card/row displays:

| Field             | Source                               |
|-------------------|--------------------------------------|
| Name              | `CollectionInfo.name`                |
| Backend           | `CollectionInfo.backend`             |
| Document count    | `CollectionInfo.document_count`      |
| Embedding model   | `CollectionInfo.embedding_model`     |
| Persist directory | `CollectionInfo.persist_directory`   |

Active collection (from `/status`) is highlighted with accent badge.

#### Actions per collection

| Action          | API Call                                        | Button Style |
|-----------------|-------------------------------------------------|--------------|
| View Details    | `GET /admin/collections/{name}`                 | Link/icon    |
| Sample Chunks   | `GET /admin/collections/{name}/chunks` **(NEW)**| Link/icon    |
| Source Files    | `GET /admin/collections/{name}/sources` **(NEW)**| Link/icon    |
| Reindex         | `POST /admin/collections/{name}/reindex`        | Primary      |
| Export          | `POST /admin/collections/{name}/export`         | Secondary    |

Note: The `POST /admin/collections/{name}/migrate` endpoint exists but is deliberately excluded from the UI. It stamps provenance metadata without reindexing or verifying vector compatibility, which could silently mislabel a legacy collection and bypass the embedding-compatibility gate. It remains available as a CLI/API-only escape hatch for operators who understand the risk.

#### Details Expandable Panel

When "View Details" is clicked, an expandable panel below the row shows the full `CollectionInfo` response plus a "Sample Chunks" sub-panel.

#### Sample Chunks Sub-Panel

Shows up to 10 sample chunks from the collection with their metadata (source file, page number, chunk text preview truncated to 200 chars). Loaded on demand when the user clicks "Sample Chunks".

#### Source Files Sub-Panel

Shows unique source filenames and their chunk counts. Loaded on demand.

#### API Endpoints Used

| Endpoint                                          | Status   |
|---------------------------------------------------|----------|
| `GET /admin/collections`                          | EXISTS   |
| `GET /admin/collections/{name}`                   | EXISTS   |
| `POST /admin/collections/{name}/reindex`          | EXISTS   |
| `POST /admin/collections/{name}/export`           | EXISTS   |
| `POST /admin/collections/import`                  | EXISTS   |
| `GET /admin/collections/{name}/chunks?limit=10`   | **NEW**  |
| `GET /admin/collections/{name}/sources`           | **NEW**  |

---

### 3.2 Jobs Tab

**Purpose:** Monitor active, completed, and failed background jobs. Retry failed jobs, cancel queued/running jobs.

#### Display

A table with columns:

| Column          | Source              |
|-----------------|---------------------|
| ID (truncated)  | `JobInfo.id`        |
| Type            | `JobInfo.type`      |
| Collection      | `JobInfo.collection_name` |
| Status          | `JobInfo.status`    |
| Progress        | `JobInfo.progress`  |
| Error           | `JobInfo.error`     |
| Created         | `JobInfo.created_at`|
| Updated         | `JobInfo.updated_at`|

Status badges use color coding:
- `queued`: muted gray
- `running`: warning yellow with animated pulse
- `completed`: success green
- `failed`: danger red
- `cancelled`: muted gray, italic

#### Actions

| Action  | Condition              | API Call                         |
|---------|------------------------|----------------------------------|
| Retry   | status == "failed"     | `POST /admin/jobs/{id}/retry`    |
| Cancel  | status in queued/running| `DELETE /admin/jobs/{id}`        |

#### Auto-Refresh

The jobs table auto-refreshes every 5 seconds while there are any jobs with status `queued` or `running`. A manual "Refresh" button is always available.

#### API Endpoints Used

| Endpoint                       | Status |
|--------------------------------|--------|
| `GET /admin/jobs`              | EXISTS |
| `GET /admin/jobs/{id}`         | EXISTS |
| `POST /admin/jobs/{id}/retry`  | EXISTS |
| `DELETE /admin/jobs/{id}`      | EXISTS |

No new endpoints needed for this section.

---

### 3.3 Metrics Tab

**Purpose:** Show live runtime metrics including application-level stats and process-level stats.

#### Display

Two cards side by side (or stacked on narrow viewports):

**Application Metrics Card:**

| Metric                     | Source field                        |
|----------------------------|------------------------------------|
| Uptime                     | `uptime_seconds` (formatted h:m:s) |
| Total Requests             | `request_count`                    |
| Total Errors               | `error_count`                      |
| Error Rate                 | computed: `error_count / request_count * 100` |
| LLM Calls                  | `llm_call_count`                   |
| LLM Avg Latency            | `llm_avg_latency_seconds`          |
| LLM Prompt Chars (total)   | `llm_total_prompt_chars`           |
| LLM Response Chars (total) | `llm_total_response_chars`         |
| Retrieval Count             | `retrieval_count`                  |
| Retrieval Avg Latency       | `retrieval_avg_latency_seconds`    |
| Documents Ingested          | `ingest_total_docs`                |

**Process Metrics Card:**

| Metric       | Source field      |
|--------------|-------------------|
| PID          | `pid`             |
| CPU %        | `cpu_percent`     |
| Memory (RSS) | `memory_rss_mb`   |
| Memory (VMS) | `memory_vms_mb`   |
| Threads      | `threads`         |

If telemetry is disabled, the tab shows a notice: "Telemetry is disabled in configuration."

#### Auto-Refresh

A "Refresh" button plus an optional auto-refresh toggle (every 10 seconds).

#### API Endpoints Used

| Endpoint                    | Status |
|-----------------------------|--------|
| `GET /admin/metrics/runtime`| EXISTS |

No new endpoints needed.

---

### 3.4 Reports Tab

**Purpose:** Show a summary report of system activity.

#### Display

A single card with key summary stats:

| Metric                | Source field            |
|-----------------------|------------------------|
| Total Requests        | `total_requests`       |
| Total Errors          | `total_errors`         |
| Total LLM Calls       | `total_llm_calls`      |
| Documents Ingested    | `total_docs_ingested`  |
| Uptime                | `uptime_seconds`       |

Plus system status from `/status`:

| Field              | Source                          |
|--------------------|---------------------------------|
| App Name           | `StatusResponse.app_name`       |
| Active Collection  | `StatusResponse.collection_name`|
| Document Count     | `StatusResponse.document_count` |
| Embedding Model    | `StatusResponse.embedding_model`|
| Backend            | `StatusResponse.vectorstore_backend` |

#### API Endpoints Used

| Endpoint                  | Status |
|---------------------------|--------|
| `GET /admin/reports/summary` | EXISTS |
| `GET /status`             | EXISTS |

No new endpoints needed.

---

## 4. New Backend Endpoints

### 4.1 `GET /admin/collections/{name}/chunks`

**Purpose:** Return a sample of chunks from a collection for inspection.

**Query Parameters:**

| Param  | Type | Default | Description                     |
|--------|------|---------|---------------------------------|
| limit  | int  | 10      | Number of sample chunks to return (max 50) |
| offset | int  | 0       | Offset for pagination           |

**Response Schema:**

```python
class ChunkSample(BaseModel):
    id: str
    text: str  # truncated to 500 chars
    metadata: dict  # source, page, etc.

class ChunkSampleResponse(BaseModel):
    collection_name: str
    total_count: int
    chunks: list[ChunkSample]
```

**Implementation Notes:**

- Requires a new `sample_chunks(collection_name, limit, offset)` method on `VectorStoreBackend`
- Chroma implementation: use `collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])`
- Qdrant implementation: use `client.scroll(collection_name, limit=limit, offset=offset, with_payload=True)`. Must filter out the provenance sentinel point (`_provenance: True` in payload) and subtract 1 from `total_count` when provenance exists, to avoid showing a fake chunk and overcounting.
- Truncate chunk text to 500 characters server-side to keep response size reasonable

### 4.2 `GET /admin/collections/{name}/sources`

**Purpose:** Return unique source filenames and their chunk counts for a collection.

**Response Schema:**

```python
class SourceInfo(BaseModel):
    filename: str
    chunk_count: int

class SourceListResponse(BaseModel):
    collection_name: str
    sources: list[SourceInfo]
```

**Implementation Notes:**

- Requires a new `list_sources(collection_name)` method on `VectorStoreBackend`
- Chroma implementation: `collection.get(include=["metadatas"])`, then group by `metadata["source"]` and count
- Qdrant implementation: scroll all points with payload, group by source metadata. Must filter out the provenance sentinel point (`_provenance: True`).
- For large collections this could be slow; add a note in the response if the collection is > 10k chunks suggesting sampling instead
- Consider caching or limiting to first N sources if performance is a concern

---

## 5. Files to Create / Modify

### New Files

| File                | Description |
|---------------------|-------------|
| `static/admin.html` | Single-page admin console. ~800-1200 lines HTML+CSS+JS. Tab-based layout with four sections. Uses `fetch()` to call backend APIs with Bearer auth from localStorage. |

### Modified Files

| File | Changes |
|------|---------|
| `src/api/routes/admin.py` | Add `GET /admin/collections/{name}/chunks` and `GET /admin/collections/{name}/sources` route handlers |
| `src/api/schemas.py` | Add `ChunkSample`, `ChunkSampleResponse`, `SourceInfo`, `SourceListResponse` Pydantic models |
| `src/rag/vectorstore.py` | Add abstract methods `sample_chunks(collection_name, limit, offset) -> dict` and `list_sources(collection_name) -> list[dict]` |
| `src/rag/chroma_backend.py` | Implement `sample_chunks` and `list_sources` using Chroma's `collection.get()` |
| `src/rag/qdrant_backend.py` | Implement `sample_chunks` and `list_sources` using Qdrant's scroll API |

---

## 6. Authentication

The admin console reuses the same auth pattern as the chat UI:

1. Token stored in `localStorage` under key `dot_admin_token` (separate from chat UI's `dot_token` to allow different token scoping)
2. Settings modal accessible via gear icon in header
3. All API calls include `Authorization: Bearer <token>` header
4. If a 403 response is received, the UI shows an inline error prompting the user to set their admin token
5. Static HTML/CSS/JS files are served without auth (they contain no sensitive data; the APIs enforce auth)

Note: The auth middleware checks `/admin` prefix paths against `settings.auth.admin_token`. When `auth.enabled` is `false` (default for local dev), all routes are open.

---

## 7. Implementation Order

### Phase 1: Backend (Agent B)

1. Add `sample_chunks` and `list_sources` abstract methods to `VectorStoreBackend`
2. Implement in `ChromaBackend` and `QdrantBackend`
3. Add Pydantic response models to `schemas.py`
4. Add route handlers to `admin.py`
5. Write unit tests for new endpoints

### Phase 2: Frontend (Agent A)

1. Create `static/admin.html` with the base layout (header, tab bar, footer)
2. Implement the Settings modal (token management)
3. Implement the Collections tab
4. Implement the Jobs tab
5. Implement the Metrics tab
6. Implement the Reports tab
7. Add auto-refresh logic for Jobs and Metrics tabs

### Phase 3: Integration (Lead)

1. Manual end-to-end testing of all tabs against running backend
2. Test with auth enabled and disabled
3. Test with telemetry enabled and disabled
4. Verify all error states (403, 404, 500, network errors)

---

## 8. Test Strategy

### Backend Unit Tests

| Test | Description |
|------|-------------|
| `test_sample_chunks_returns_limited_results` | Verify `sample_chunks` respects limit parameter |
| `test_sample_chunks_truncates_text` | Verify chunk text is truncated to 500 chars |
| `test_sample_chunks_nonexistent_collection` | Verify 404 for unknown collection |
| `test_list_sources_groups_correctly` | Verify source grouping and counts |
| `test_list_sources_empty_collection` | Verify empty list for collection with no docs |
| `test_existing_admin_endpoints_unchanged` | Regression: existing endpoints still work |

### Frontend Manual Tests

| Test | Steps |
|------|-------|
| Tab navigation | Click each tab, verify content loads |
| Auth flow | Set token in settings, verify API calls include it |
| Auth error | Use wrong token with auth enabled, verify error message |
| Collections list | Verify collections load and display correctly |
| Sample chunks | Click sample chunks, verify chunks appear |
| Source files | Click sources, verify source list appears |
| Reindex trigger | Click reindex, verify job created and appears in Jobs tab |
| Jobs refresh | Trigger a job, verify auto-refresh shows progress |
| Job retry | Wait for a failed job, click retry, verify status changes |
| Metrics display | Verify metrics load and format correctly |
| Metrics disabled | Disable telemetry, verify disabled message shows |
| Reports display | Verify summary report loads |
| Empty state | Verify graceful handling when no collections/jobs exist |

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `list_sources` is slow on large collections (full scan) | High latency on collections with >10k chunks | Add server-side limit (scan at most 10k chunks); document the limitation |
| Chroma `collection.get()` with offset may not be supported in all versions | Endpoint returns error | Check Chroma version compatibility; fall back to `peek()` if needed |
| Auth token stored in localStorage is accessible to XSS | Low risk for local-first tool | Document that this is acceptable for local use; production deployments should use HttpOnly cookies |
| Admin console HTML grows very large | Maintainability | Keep CSS and JS well-organized with clear section comments; consider extracting to separate files in a future iteration |
| Polling for auto-refresh creates unnecessary load | Minor | Only poll when tab is visible (use `document.visibilityState`); stop polling on inactive tabs |
