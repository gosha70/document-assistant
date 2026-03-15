"""Stateless alert evaluator.

Given current metrics and settings, returns a list of active alerts.
All evaluation is pure arithmetic — no I/O, no side effects.
"""

from dataclasses import dataclass, asdict

from src.config.settings import AlertingSettings


@dataclass
class AlertInfo:
    level: str
    metric: str
    current_value: float
    threshold: float
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class AlertChecker:
    @staticmethod
    def check(
        stats: dict,
        process_stats: dict,
        windowed_counts: dict,
        settings: AlertingSettings,
    ) -> list[AlertInfo]:
        """Evaluate metrics against thresholds. Returns empty list if disabled."""
        if not settings.enabled:
            return []

        alerts: list[AlertInfo] = []

        # Error rate
        request_count = stats.get("request_count", 0)
        error_count = stats.get("error_count", 0)
        if request_count > 0:
            error_rate = error_count / request_count
            if error_rate >= settings.error_rate_threshold:
                alerts.append(
                    AlertInfo(
                        level="warning",
                        metric="error_rate",
                        current_value=round(error_rate, 4),
                        threshold=settings.error_rate_threshold,
                        message=f"Error rate is {error_rate * 100:.1f}%, exceeds threshold of {settings.error_rate_threshold * 100:.1f}%",
                    )
                )

        # Memory RSS
        if "error" not in process_stats:
            memory_rss_mb = process_stats.get("memory_rss_mb")
            if memory_rss_mb is not None and memory_rss_mb >= settings.memory_rss_mb_threshold:
                alerts.append(
                    AlertInfo(
                        level="warning",
                        metric="memory_rss_mb",
                        current_value=memory_rss_mb,
                        threshold=float(settings.memory_rss_mb_threshold),
                        message=f"Memory RSS is {memory_rss_mb:.0f} MB, exceeds threshold of {settings.memory_rss_mb_threshold} MB",
                    )
                )

        # Ingest errors (windowed)
        ingest_err = windowed_counts.get("ingest_error_count", 0)
        if ingest_err >= settings.ingest_error_threshold:
            alerts.append(
                AlertInfo(
                    level="warning",
                    metric="ingest_error_count",
                    current_value=float(ingest_err),
                    threshold=float(settings.ingest_error_threshold),
                    message=f"Ingest errors: {ingest_err} in last {settings.window_seconds}s, exceeds threshold of {settings.ingest_error_threshold}",
                )
            )

        # Retrieval no-source (windowed)
        no_source = windowed_counts.get("retrieval_no_source_count", 0)
        if no_source >= settings.retrieval_no_source_threshold:
            alerts.append(
                AlertInfo(
                    level="warning",
                    metric="retrieval_no_source_count",
                    current_value=float(no_source),
                    threshold=float(settings.retrieval_no_source_threshold),
                    message=f"Retrieval with no sources: {no_source} in last {settings.window_seconds}s, exceeds threshold of {settings.retrieval_no_source_threshold}",
                )
            )

        return alerts
