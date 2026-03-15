"""Thread-safe bounded in-memory store for periodic metric snapshots."""

import threading
from collections import deque


class ReportStore:
    def __init__(self, max_snapshots: int = 24):
        self._snapshots: deque[dict] = deque(maxlen=max_snapshots)
        self._lock = threading.Lock()

    def add(self, snapshot: dict) -> None:
        with self._lock:
            self._snapshots.append(snapshot)

    def latest(self) -> dict | None:
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def list_all(self) -> list[dict]:
        with self._lock:
            return list(self._snapshots)


_store: ReportStore | None = None


def get_report_store() -> ReportStore:
    global _store
    if _store is None:
        _store = ReportStore()
    return _store


def reset_report_store() -> None:
    global _store
    _store = None
