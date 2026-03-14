"""Tests for background job store and executor."""
import time
import pytest
from unittest.mock import MagicMock

from src.utils.jobs import JobStore, JobExecutor


class TestJobStore:
    def test_create_returns_job(self):
        store = JobStore()
        job = store.create(job_type="reindex", collection_name="test_col")
        assert job.id
        assert job.type == "reindex"
        assert job.status == "queued"
        assert job.collection_name == "test_col"
        assert job.created_at
        assert job.progress == 0.0

    def test_get_existing(self):
        store = JobStore()
        job = store.create(job_type="export")
        retrieved = store.get(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

    def test_get_nonexistent(self):
        store = JobStore()
        assert store.get("no-such-id") is None

    def test_list_all(self):
        store = JobStore()
        store.create(job_type="reindex")
        store.create(job_type="export")
        jobs = store.list_all()
        assert len(jobs) == 2

    def test_update(self):
        store = JobStore()
        job = store.create(job_type="reindex")
        updated = store.update(job.id, status="running", progress=0.5)
        assert updated.status == "running"
        assert updated.progress == 0.5
        assert updated.updated_at >= job.created_at

    def test_update_nonexistent(self):
        store = JobStore()
        assert store.update("no-such-id", status="running") is None

    def test_delete(self):
        store = JobStore()
        job = store.create(job_type="reindex")
        assert store.delete(job.id) is True
        assert store.get(job.id) is None

    def test_delete_nonexistent(self):
        store = JobStore()
        assert store.delete("no-such-id") is False

    def test_to_dict(self):
        store = JobStore()
        job = store.create(job_type="reindex", collection_name="col_a")
        d = job.to_dict()
        assert d["type"] == "reindex"
        assert d["collection_name"] == "col_a"
        assert "id" in d
        assert "status" in d
        assert "created_at" in d


class TestJobExecutor:
    def test_runs_job_to_completion(self):
        store = JobStore()
        executed = {"called": False}

        def task(job, store, **kwargs):
            executed["called"] = True

        job = store.create(job_type="reindex", func=task)
        executor = JobExecutor(store, max_workers=1)
        try:
            executor.submit(job)
            time.sleep(0.5)
            result = store.get(job.id)
            assert executed["called"]
            assert result.status == "completed"
            assert result.progress == 1.0
        finally:
            executor.shutdown(wait=True)

    def test_failed_job_records_error(self):
        store = JobStore()

        def failing_task(job, store, **kwargs):
            raise RuntimeError("something broke")

        job = store.create(job_type="reindex", func=failing_task)
        executor = JobExecutor(store, max_workers=1)
        try:
            executor.submit(job)
            time.sleep(0.5)
            result = store.get(job.id)
            assert result.status == "failed"
            assert "something broke" in result.error
        finally:
            executor.shutdown(wait=True)

    def test_task_can_update_progress(self):
        store = JobStore()

        def progressing_task(job, store, **kwargs):
            store.update(job.id, progress=0.5)
            time.sleep(0.1)

        job = store.create(job_type="export", func=progressing_task)
        executor = JobExecutor(store, max_workers=1)
        try:
            executor.submit(job)
            time.sleep(0.5)
            result = store.get(job.id)
            assert result.status == "completed"
        finally:
            executor.shutdown(wait=True)

    def test_retry_failed_job(self):
        store = JobStore()
        call_count = {"n": 0}

        def flaky_task(job, store, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first attempt fails")

        job = store.create(job_type="reindex", func=flaky_task)
        executor = JobExecutor(store, max_workers=1)
        try:
            executor.submit(job)
            time.sleep(0.5)
            assert store.get(job.id).status == "failed"

            store.update(job.id, status="queued", error=None, progress=0.0)
            executor.submit(job)
            time.sleep(0.5)
            assert store.get(job.id).status == "completed"
            assert call_count["n"] == 2
        finally:
            executor.shutdown(wait=True)
