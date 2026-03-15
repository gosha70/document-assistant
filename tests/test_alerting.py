"""Unit tests for AlertChecker."""

import pytest

from src.config.settings import AlertingSettings
from src.utils.alerting import AlertChecker


def _settings(**overrides) -> AlertingSettings:
    defaults = {
        "enabled": True,
        "error_rate_threshold": 0.10,
        "memory_rss_mb_threshold": 1024,
        "ingest_error_threshold": 5,
        "retrieval_no_source_threshold": 10,
        "window_seconds": 3600,
    }
    defaults.update(overrides)
    return AlertingSettings(**defaults)


def _stats(**overrides) -> dict:
    base = {
        "uptime_seconds": 100.0,
        "request_count": 100,
        "error_count": 0,
        "llm_call_count": 50,
        "ingest_total_docs": 200,
        "ingest_error_count": 0,
        "retrieval_no_source_count": 0,
    }
    base.update(overrides)
    return base


def _process_stats(**overrides) -> dict:
    base = {
        "pid": 1234,
        "cpu_percent": 5.0,
        "memory_rss_mb": 256.0,
        "memory_vms_mb": 512.0,
        "threads": 4,
    }
    base.update(overrides)
    return base


def _windowed(**overrides) -> dict:
    base = {"ingest_error_count": 0, "retrieval_no_source_count": 0}
    base.update(overrides)
    return base


class TestAlertCheckerNoAlerts:
    def test_all_below_thresholds(self):
        alerts = AlertChecker.check(_stats(), _process_stats(), _windowed(), _settings())
        assert alerts == []

    def test_zero_requests(self):
        alerts = AlertChecker.check(
            _stats(request_count=0, error_count=0),
            _process_stats(),
            _windowed(),
            _settings(),
        )
        assert alerts == []


class TestAlertCheckerDisabled:
    def test_disabled_returns_empty(self):
        alerts = AlertChecker.check(
            _stats(error_count=99),
            _process_stats(memory_rss_mb=9999.0),
            _windowed(ingest_error_count=100),
            _settings(enabled=False),
        )
        assert alerts == []


class TestAlertCheckerErrorRate:
    def test_above_threshold(self):
        alerts = AlertChecker.check(
            _stats(request_count=100, error_count=15),
            _process_stats(),
            _windowed(),
            _settings(),
        )
        assert len(alerts) == 1
        assert alerts[0].metric == "error_rate"
        assert alerts[0].level == "warning"
        assert alerts[0].current_value == 0.15

    def test_at_threshold(self):
        alerts = AlertChecker.check(
            _stats(request_count=100, error_count=10),
            _process_stats(),
            _windowed(),
            _settings(),
        )
        assert len(alerts) == 1
        assert alerts[0].metric == "error_rate"

    def test_below_threshold(self):
        alerts = AlertChecker.check(
            _stats(request_count=100, error_count=9),
            _process_stats(),
            _windowed(),
            _settings(),
        )
        assert alerts == []


class TestAlertCheckerMemory:
    def test_above_threshold(self):
        alerts = AlertChecker.check(
            _stats(),
            _process_stats(memory_rss_mb=1200.0),
            _windowed(),
            _settings(),
        )
        assert len(alerts) == 1
        assert alerts[0].metric == "memory_rss_mb"
        assert alerts[0].current_value == 1200.0

    def test_psutil_unavailable(self):
        alerts = AlertChecker.check(
            _stats(),
            {"error": "psutil not installed"},
            _windowed(),
            _settings(),
        )
        assert alerts == []


class TestAlertCheckerIngestErrors:
    def test_above_threshold(self):
        alerts = AlertChecker.check(
            _stats(),
            _process_stats(),
            _windowed(ingest_error_count=6),
            _settings(),
        )
        assert len(alerts) == 1
        assert alerts[0].metric == "ingest_error_count"

    def test_below_threshold(self):
        alerts = AlertChecker.check(
            _stats(),
            _process_stats(),
            _windowed(ingest_error_count=4),
            _settings(),
        )
        assert alerts == []


class TestAlertCheckerNoSources:
    def test_above_threshold(self):
        alerts = AlertChecker.check(
            _stats(),
            _process_stats(),
            _windowed(retrieval_no_source_count=12),
            _settings(),
        )
        assert len(alerts) == 1
        assert alerts[0].metric == "retrieval_no_source_count"

    def test_below_threshold(self):
        alerts = AlertChecker.check(
            _stats(),
            _process_stats(),
            _windowed(retrieval_no_source_count=9),
            _settings(),
        )
        assert alerts == []


class TestAlertCheckerMultiple:
    def test_multiple_alerts(self):
        alerts = AlertChecker.check(
            _stats(request_count=100, error_count=20),
            _process_stats(memory_rss_mb=2048.0),
            _windowed(ingest_error_count=10, retrieval_no_source_count=15),
            _settings(),
        )
        assert len(alerts) == 4
        metrics = {a.metric for a in alerts}
        assert metrics == {"error_rate", "memory_rss_mb", "ingest_error_count", "retrieval_no_source_count"}
