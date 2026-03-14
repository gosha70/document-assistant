import logging
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    CollectionInfo,
    JobInfo,
    ChunkSampleResponse,
    ChunkSample,
    SourceListResponse,
    SourceInfo,
)
from src.api.deps import get_vectorstore_backend
from src.config.settings import get_settings
from src.utils.metrics import get_metrics_collector
from src.utils.jobs import get_job_store, get_job_executor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# --- Collections ---


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections():
    """List all known collections with metadata."""
    backend = get_vectorstore_backend()
    collections = backend.list_collections()
    return [CollectionInfo(**c) for c in collections]


@router.get("/collections/{collection_name}", response_model=CollectionInfo)
def get_collection(collection_name: str):
    """Get details for a specific collection."""
    backend = get_vectorstore_backend()
    try:
        info = backend.get_collection_info(collection_name)
        return CollectionInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/collections/{collection_name}/migrate")
def migrate_collection(collection_name: str):
    """Stamp embedding provenance on a legacy collection (no re-embedding).

    This allows legacy collections created before provenance tracking to pass
    the compatibility gate without setting allow_legacy_collections=true.
    """
    backend = get_vectorstore_backend()
    try:
        backend.get_collection_info(collection_name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    provenance = backend.get_embedding_provenance(collection_name)
    if provenance is not None:
        return {
            "status": "already_migrated",
            "collection_name": collection_name,
            "embedding_model": provenance["model_name"],
            "embedding_type": provenance.get("type"),
        }

    backend._set_provenance(collection_name)

    return {
        "status": "migrated",
        "collection_name": collection_name,
        "embedding_model": backend._embedding.model_name,
        "embedding_type": backend._embedding_type,
    }


@router.get("/collections/{collection_name}/chunks", response_model=ChunkSampleResponse)
def sample_chunks(collection_name: str, limit: int = 10, offset: int = 0):
    """Return a paginated sample of chunks from a collection."""
    limit = min(limit, 50)
    backend = get_vectorstore_backend()
    try:
        result = backend.sample_chunks(collection_name, limit=limit, offset=offset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ChunkSampleResponse(
        collection_name=collection_name,
        total_count=result["total_count"],
        chunks=[ChunkSample(**c) for c in result["chunks"]],
    )


@router.get("/collections/{collection_name}/sources", response_model=SourceListResponse)
def list_sources(collection_name: str):
    """Return unique source filenames and their chunk counts for a collection."""
    backend = get_vectorstore_backend()
    try:
        result = backend.list_sources(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SourceListResponse(
        collection_name=collection_name,
        sources=[SourceInfo(**s) for s in result["sources"]],
        truncated=result["truncated"],
        scanned_chunks=result["scanned_chunks"],
    )


@router.post("/collections/{collection_name}/reindex", response_model=JobInfo)
def trigger_reindex(collection_name: str):
    """Trigger reindex from canonical sources."""
    store = get_job_store()
    executor = get_job_executor()

    def reindex_task(job, store, **kwargs):
        store.update(job.id, progress=0.5)
        logger.info(f"Reindex task for collection '{kwargs.get('collection_name')}' — placeholder")

    job = store.create(
        job_type="reindex",
        collection_name=collection_name,
        func=reindex_task,
        kwargs={"collection_name": collection_name},
    )
    executor.submit(job)
    return JobInfo(**job.to_dict())


@router.post("/collections/{collection_name}/export", response_model=JobInfo)
def trigger_export(collection_name: str):
    """Export collection data."""
    store = get_job_store()
    executor = get_job_executor()

    def export_task(job, store, **kwargs):
        store.update(job.id, progress=0.5)
        logger.info(f"Export task for collection '{kwargs.get('collection_name')}' — placeholder")

    job = store.create(
        job_type="export",
        collection_name=collection_name,
        func=export_task,
        kwargs={"collection_name": collection_name},
    )
    executor.submit(job)
    return JobInfo(**job.to_dict())


@router.post("/collections/import", response_model=JobInfo)
def trigger_import():
    """Import collection from export artifact."""
    store = get_job_store()
    executor = get_job_executor()

    def import_task(job, store, **kwargs):
        store.update(job.id, progress=0.5)
        logger.info("Import task — placeholder")

    job = store.create(
        job_type="import",
        func=import_task,
    )
    executor.submit(job)
    return JobInfo(**job.to_dict())


# --- Jobs ---


@router.get("/jobs", response_model=list[JobInfo])
def list_jobs():
    """List all jobs."""
    store = get_job_store()
    return [JobInfo(**j.to_dict()) for j in store.list_all()]


@router.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    """Get job detail with progress."""
    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobInfo(**job.to_dict())


@router.post("/jobs/{job_id}/retry", response_model=JobInfo)
def retry_job(job_id: str):
    """Retry a failed job."""
    store = get_job_store()
    executor = get_job_executor()

    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.status != "failed":
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' is not in failed state (status: {job.status})")

    store.update(job_id, status="queued", error=None, progress=0.0)
    executor.submit(job)
    return JobInfo(**job.to_dict())


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    """Cancel a running or queued job."""
    store = get_job_store()
    executor = get_job_executor()

    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' is already {job.status}")

    cancelled = executor.cancel(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is already running and cannot be cancelled",
        )
    return {"status": "cancelled", "job_id": job_id}


# --- Metrics & Reports ---


@router.get("/metrics/runtime")
def get_runtime_metrics():
    """Live runtime metrics: application stats + process stats."""
    settings = get_settings()
    if not settings.telemetry.enabled:
        return {"status": "disabled", "message": "Telemetry is disabled in configuration"}

    collector = get_metrics_collector()
    return {
        "application": collector.get_stats(),
        "process": collector.get_process_stats(),
    }


@router.get("/reports/summary")
def get_summary_report():
    """Summary report."""
    settings = get_settings()
    if not settings.telemetry.enabled:
        return {"status": "disabled", "message": "Telemetry is disabled in configuration"}

    collector = get_metrics_collector()
    stats = collector.get_stats()
    return {
        "total_requests": stats["request_count"],
        "total_errors": stats["error_count"],
        "total_llm_calls": stats["llm_call_count"],
        "total_docs_ingested": stats["ingest_total_docs"],
        "uptime_seconds": stats["uptime_seconds"],
    }
