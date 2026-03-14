"""Tests for the metrics collection module."""
from src.utils.metrics import MetricsCollector


class TestMetricsCollector:
    def test_initial_stats(self):
        mc = MetricsCollector()
        stats = mc.get_stats()
        assert stats["request_count"] == 0
        assert stats["error_count"] == 0
        assert stats["llm_call_count"] == 0
        assert stats["retrieval_count"] == 0
        assert stats["ingest_total_docs"] == 0
        assert stats["uptime_seconds"] >= 0

    def test_record_request(self):
        mc = MetricsCollector()
        mc.record_request()
        mc.record_request()
        assert mc.get_stats()["request_count"] == 2

    def test_record_error(self):
        mc = MetricsCollector()
        mc.record_error()
        assert mc.get_stats()["error_count"] == 1

    def test_record_llm_call(self):
        mc = MetricsCollector()
        mc.record_llm_call(latency=1.5, prompt_chars=100, response_chars=50)
        mc.record_llm_call(latency=2.5, prompt_chars=200, response_chars=75)

        stats = mc.get_stats()
        assert stats["llm_call_count"] == 2
        assert stats["llm_avg_latency_seconds"] == 2.0
        assert stats["llm_total_prompt_chars"] == 300
        assert stats["llm_total_response_chars"] == 125

    def test_record_retrieval(self):
        mc = MetricsCollector()
        mc.record_retrieval(0.1)
        mc.record_retrieval(0.3)

        stats = mc.get_stats()
        assert stats["retrieval_count"] == 2
        assert stats["retrieval_avg_latency_seconds"] == 0.2

    def test_record_ingest(self):
        mc = MetricsCollector()
        mc.record_ingest(10)
        mc.record_ingest(5)
        assert mc.get_stats()["ingest_total_docs"] == 15

    def test_process_stats_returns_dict(self):
        mc = MetricsCollector()
        ps = mc.get_process_stats()
        assert "pid" in ps
        assert "memory_rss_mb" in ps
        assert "cpu_percent" in ps

    def test_zero_division_safe(self):
        mc = MetricsCollector()
        stats = mc.get_stats()
        assert stats["llm_avg_latency_seconds"] == 0
        assert stats["retrieval_avg_latency_seconds"] == 0
