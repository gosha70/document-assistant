"""Unit tests for ReportStore."""

import pytest

from src.utils.report_store import ReportStore


def _snapshot(n: int) -> dict:
    return {
        "timestamp": f"2026-03-14T{n:02d}:00:00+00:00",
        "uptime_seconds": float(n * 3600),
        "request_count": n * 10,
        "error_count": n,
        "llm_call_count": n * 5,
        "ingest_total_docs": n * 20,
        "ingest_error_count": 0,
        "retrieval_no_source_count": 0,
    }


class TestReportStoreEmpty:
    def test_latest_returns_none(self):
        store = ReportStore()
        assert store.latest() is None

    def test_list_all_returns_empty(self):
        store = ReportStore()
        assert store.list_all() == []


class TestReportStoreBasic:
    def test_add_and_latest(self):
        store = ReportStore()
        store.add(_snapshot(1))
        assert store.latest() == _snapshot(1)

    def test_add_and_list_all(self):
        store = ReportStore()
        store.add(_snapshot(1))
        store.add(_snapshot(2))
        assert store.list_all() == [_snapshot(1), _snapshot(2)]

    def test_latest_returns_most_recent(self):
        store = ReportStore()
        store.add(_snapshot(1))
        store.add(_snapshot(2))
        store.add(_snapshot(3))
        assert store.latest() == _snapshot(3)


class TestReportStoreBounded:
    def test_evicts_oldest(self):
        store = ReportStore(max_snapshots=3)
        for i in range(5):
            store.add(_snapshot(i))
        all_snaps = store.list_all()
        assert len(all_snaps) == 3
        assert all_snaps[0] == _snapshot(2)
        assert all_snaps[2] == _snapshot(4)

    def test_max_one(self):
        store = ReportStore(max_snapshots=1)
        store.add(_snapshot(1))
        store.add(_snapshot(2))
        assert store.list_all() == [_snapshot(2)]
        assert store.latest() == _snapshot(2)
