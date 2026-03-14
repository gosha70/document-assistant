"""Background job store and executor.

Provides a thread-safe in-memory job store and a thread-pool-based executor
for long-running operations (reindex, export, import).
"""
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    type: str
    status: str  # queued, running, completed, failed, cancelled
    collection_name: Optional[str] = None
    progress: Optional[float] = 0.0
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    _func: Optional[Callable] = field(default=None, repr=False)
    _kwargs: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "collection_name": self.collection_name,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job_type: str, collection_name: Optional[str] = None,
               func: Optional[Callable] = None, kwargs: Optional[dict] = None) -> Job:
        now = datetime.now(timezone.utc).isoformat()
        job = Job(
            id=str(uuid.uuid4()),
            type=job_type,
            status="queued",
            collection_name=collection_name,
            progress=0.0,
            created_at=now,
            updated_at=now,
            _func=func,
            _kwargs=kwargs or {},
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def update(self, job_id: str, **kwargs: Any) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat()
            return job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None


class JobExecutor:
    """Runs jobs in background threads via a thread pool."""

    def __init__(self, store: JobStore, max_workers: int = 2):
        self._store = store
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Any] = {}

    def submit(self, job: Job) -> None:
        future = self._pool.submit(self._run, job)
        self._futures[job.id] = future

    def cancel(self, job_id: str) -> bool:
        future = self._futures.get(job_id)
        if future and not future.done():
            cancelled = future.cancel()
            if cancelled:
                self._store.update(job_id, status="cancelled")
            return cancelled
        return False

    def _run(self, job: Job) -> None:
        self._store.update(job.id, status="running", progress=0.0)
        logger.info(f"Job {job.id} ({job.type}) started")

        try:
            if job._func is not None:
                job._func(job=job, store=self._store, **job._kwargs)
            self._store.update(job.id, status="completed", progress=1.0)
            logger.info(f"Job {job.id} ({job.type}) completed")
        except Exception as e:
            self._store.update(job.id, status="failed", error=str(e))
            logger.error(f"Job {job.id} ({job.type}) failed: {e}", exc_info=True)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)
        self._shutdown = True

    @property
    def is_shutdown(self) -> bool:
        return getattr(self, "_shutdown", False)


# Module-level singletons
_job_store: Optional[JobStore] = None
_job_executor: Optional[JobExecutor] = None


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store


def get_job_executor() -> JobExecutor:
    global _job_executor
    if _job_executor is None or _job_executor.is_shutdown:
        _job_executor = JobExecutor(get_job_store())
    return _job_executor


def reset_jobs() -> None:
    """Reset singletons. Used during shutdown and testing."""
    global _job_store, _job_executor
    if _job_executor is not None and not _job_executor.is_shutdown:
        _job_executor.shutdown(wait=False)
    _job_store = None
    _job_executor = None
