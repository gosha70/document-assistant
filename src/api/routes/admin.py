import logging
from fastapi import APIRouter, HTTPException

from src.api.schemas import CollectionInfo, JobInfo
from src.api.deps import get_vectorstore_backend
from src.config.settings import get_settings

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


@router.post("/collections/{collection_name}/reindex", response_model=JobInfo)
def trigger_reindex(collection_name: str):
    """Trigger reindex from canonical sources. Wired to job queue in Phase 9."""
    # Stub — returns a placeholder job until Phase 9 job orchestration
    return JobInfo(
        id="stub-reindex-001",
        type="reindex",
        status="queued",
        progress=0.0,
    )


@router.post("/collections/{collection_name}/export", response_model=JobInfo)
def trigger_export(collection_name: str):
    """Export collection data. Wired to job queue in Phase 9."""
    return JobInfo(
        id="stub-export-001",
        type="export",
        status="queued",
        progress=0.0,
    )


@router.post("/collections/import", response_model=JobInfo)
def trigger_import():
    """Import collection from export artifact. Wired to job queue in Phase 9."""
    return JobInfo(
        id="stub-import-001",
        type="import",
        status="queued",
        progress=0.0,
    )


# --- Jobs ---

@router.get("/jobs", response_model=list[JobInfo])
def list_jobs():
    """List ingestion jobs. Backed by job store in Phase 9."""
    return []


@router.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    """Get job detail with progress. Backed by job store in Phase 9."""
    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")


@router.post("/jobs/{job_id}/retry", response_model=JobInfo)
def retry_job(job_id: str):
    """Retry a failed job. Wired to job queue in Phase 9."""
    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    """Cancel a running job. Wired to job queue in Phase 9."""
    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")


# --- Metrics & Reports ---

@router.get("/metrics/runtime")
def get_runtime_metrics():
    """Live runtime metrics. Populated in Phase 4."""
    return {"status": "not_yet_implemented", "phase": 4}


@router.get("/reports/summary")
def get_summary_report():
    """Summary report. Populated in Phase 4/10."""
    return {"status": "not_yet_implemented", "phase": 10}
