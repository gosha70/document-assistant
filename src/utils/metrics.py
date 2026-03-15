"""Application-level metrics collection.

Tracks request counts, LLM call latency/tokens, retrieval timing,
and exposes process-level stats via psutil.

Uses running counters/sums instead of per-call records to keep
memory usage constant regardless of traffic volume.

Windowed event counters (ingest errors, retrieval-no-source) use
timestamp deques pruned to 2*window_seconds with a hard cap of 10k.
"""

import time
import threading
from collections import deque

_MAX_EVENT_ENTRIES = 10_000


class MetricsCollector:
    """Thread-safe in-memory metrics collector using fixed-size counters."""

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._request_count = 0
        self._error_count = 0
        # LLM aggregates
        self._llm_call_count = 0
        self._llm_total_latency = 0.0
        self._llm_total_prompt_chars = 0
        self._llm_total_response_chars = 0
        # Retrieval aggregates
        self._retrieval_count = 0
        self._retrieval_total_latency = 0.0
        # Ingest
        self._ingest_doc_count = 0
        # Windowed event counters (timestamp deques)
        self._ingest_error_timestamps: deque[float] = deque(maxlen=_MAX_EVENT_ENTRIES)
        self._retrieval_no_source_timestamps: deque[float] = deque(maxlen=_MAX_EVENT_ENTRIES)

    def record_request(self) -> None:
        with self._lock:
            self._request_count += 1

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def record_llm_call(self, latency: float, prompt_chars: int, response_chars: int) -> None:
        with self._lock:
            self._llm_call_count += 1
            self._llm_total_latency += latency
            self._llm_total_prompt_chars += prompt_chars
            self._llm_total_response_chars += response_chars

    def record_retrieval(self, latency: float) -> None:
        with self._lock:
            self._retrieval_count += 1
            self._retrieval_total_latency += latency

    def record_ingest(self, doc_count: int) -> None:
        with self._lock:
            self._ingest_doc_count += doc_count

    def _prune_deque(self, dq: deque, max_age: float) -> None:
        """Remove entries older than max_age seconds (caller holds lock)."""
        now = time.monotonic()
        cutoff = now - max_age
        while dq and dq[0] < cutoff:
            dq.popleft()

    def record_ingest_error(self, window_seconds: int) -> None:
        with self._lock:
            self._prune_deque(self._ingest_error_timestamps, 2 * window_seconds)
            self._ingest_error_timestamps.append(time.monotonic())

    def record_retrieval_no_source(self, window_seconds: int) -> None:
        with self._lock:
            self._prune_deque(self._retrieval_no_source_timestamps, 2 * window_seconds)
            self._retrieval_no_source_timestamps.append(time.monotonic())

    def get_windowed_counts(self, window_seconds: int) -> dict:
        """Return event counts within the last window_seconds."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            self._prune_deque(self._ingest_error_timestamps, 2 * window_seconds)
            self._prune_deque(self._retrieval_no_source_timestamps, 2 * window_seconds)
            ingest_err = sum(1 for t in self._ingest_error_timestamps if t >= cutoff)
            no_source = sum(1 for t in self._retrieval_no_source_timestamps if t >= cutoff)
        return {
            "ingest_error_count": ingest_err,
            "retrieval_no_source_count": no_source,
        }

    def get_stats(self, window_seconds: int = 7200) -> dict:
        with self._lock:
            self._prune_deque(self._ingest_error_timestamps, window_seconds)
            self._prune_deque(self._retrieval_no_source_timestamps, window_seconds)
            uptime = time.monotonic() - self._start_time
            return {
                "uptime_seconds": round(uptime, 1),
                "request_count": self._request_count,
                "error_count": self._error_count,
                "llm_call_count": self._llm_call_count,
                "llm_avg_latency_seconds": round(self._llm_total_latency / self._llm_call_count, 3)
                if self._llm_call_count
                else 0,
                "llm_total_prompt_chars": self._llm_total_prompt_chars,
                "llm_total_response_chars": self._llm_total_response_chars,
                "retrieval_count": self._retrieval_count,
                "retrieval_avg_latency_seconds": round(self._retrieval_total_latency / self._retrieval_count, 3)
                if self._retrieval_count
                else 0,
                "ingest_total_docs": self._ingest_doc_count,
                "ingest_error_count": len(self._ingest_error_timestamps),
                "retrieval_no_source_count": len(self._retrieval_no_source_timestamps),
            }

    def get_process_stats(self) -> dict:
        """Return process-level metrics via psutil."""
        try:
            import psutil

            proc = psutil.Process()
            mem = proc.memory_info()
            return {
                "pid": proc.pid,
                "cpu_percent": proc.cpu_percent(interval=0.1),
                "memory_rss_mb": round(mem.rss / (1024 * 1024), 1),
                "memory_vms_mb": round(mem.vms / (1024 * 1024), 1),
                "threads": proc.num_threads(),
            }
        except ImportError:
            return {"error": "psutil not installed"}


# Module-level singleton
_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
