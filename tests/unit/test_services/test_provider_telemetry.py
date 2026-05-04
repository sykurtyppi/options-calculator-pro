from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.provider_telemetry import (
    ProviderTelemetryEvent,
    ProviderTelemetryStore,
    build_provider_telemetry_diagnostics,
    classify_error,
    record_provider_telemetry,
)


def test_provider_telemetry_records_events_and_summarizes_health(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIONS_CALCULATOR_TELEMETRY_MAX_AGE_DAYS", "365")
    monkeypatch.setenv("OPTIONS_CALCULATOR_TELEMETRY_MAX_ROWS", "100")
    store = ProviderTelemetryStore(tmp_path / "telemetry.sqlite")
    store.record(
        ProviderTelemetryEvent(
            provider_name="marketdata_app",
            endpoint_type="options_chain",
            symbol="AAPL",
            success=True,
            latency_ms=120.0,
        )
    )
    store.record(
        ProviderTelemetryEvent(
            provider_name="marketdata_app",
            endpoint_type="earnings",
            symbol="MSFT",
            success=False,
            error_category="rate_limited",
            latency_ms=240.0,
            fallback_used=True,
        )
    )
    record_provider_telemetry(
        provider_name="yfinance",
        endpoint_type="forward_loop_option_quote",
        symbol="TSLA",
        success=True,
        stale_used=True,
        response_quality_note="paper/research quote",
        store=store,
    )

    payload = build_provider_telemetry_diagnostics(store=store)

    assert payload["totals"]["events"] == 3
    assert payload["totals"]["failures"] == 1
    assert payload["totals"]["fallback_count"] == 1
    assert payload["totals"]["stale_count"] == 1
    assert payload["summary_by_provider"]["marketdata_app"]["total"] == 2
    assert payload["summary_by_provider"]["marketdata_app"]["failures"] == 1
    assert payload["summary_by_provider"]["marketdata_app"]["failure_rate"] == 0.5
    assert payload["summary_by_provider"]["marketdata_app"]["avg_latency_ms"] == 180.0
    assert payload["recent_failures"][0]["provider_name"] == "marketdata_app"
    assert payload["operational_health"]["row_count"] == 3


def test_provider_telemetry_error_classification_is_sanitized():
    assert classify_error("HTTP 429 too many requests") == "rate_limited"
    assert classify_error("HTTP 402 Payment Required") == "entitlement"
    assert classify_error("request timeout") == "timeout"
    assert classify_error("not found") == "not_found"


def test_provider_telemetry_filters_and_paginates(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIONS_CALCULATOR_TELEMETRY_MAX_AGE_DAYS", "365")
    monkeypatch.setenv("OPTIONS_CALCULATOR_TELEMETRY_MAX_ROWS", "100")
    store = ProviderTelemetryStore(tmp_path / "telemetry.sqlite")
    base = datetime.now(timezone.utc)
    events = [
        ProviderTelemetryEvent("marketdata_app", "options_chain", "AAPL", True, latency_ms=100, request_timestamp=(base - timedelta(minutes=3)).isoformat()),
        ProviderTelemetryEvent("marketdata_app", "earnings", "MSFT", False, error_category="timeout", latency_ms=250, request_timestamp=(base - timedelta(minutes=2)).isoformat()),
        ProviderTelemetryEvent("yfinance", "price_history", "AAPL", True, latency_ms=50, request_timestamp=(base - timedelta(minutes=1)).isoformat()),
    ]
    for event in events:
        store.record(event)

    payload = build_provider_telemetry_diagnostics(
        store=store,
        provider="marketdata_app",
        success=False,
        limit=1,
        offset=0,
    )

    assert payload["totals"]["events"] == 1
    assert payload["totals"]["failures"] == 1
    assert payload["recent_events"][0]["symbol"] == "MSFT"
    assert payload["filters"]["provider"] == "marketdata_app"
    assert payload["filters"]["success"] is False
    assert payload["has_more"] is False

    aapl_payload = build_provider_telemetry_diagnostics(store=store, symbol="aapl", limit=1)
    assert aapl_payload["totals"]["events"] == 2
    assert aapl_payload["has_more"] is True


def test_provider_telemetry_safe_cleanup_applies_age_and_row_caps(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIONS_CALCULATOR_TELEMETRY_MAX_AGE_DAYS", "365")
    monkeypatch.setenv("OPTIONS_CALCULATOR_TELEMETRY_MAX_ROWS", "100")
    store = ProviderTelemetryStore(tmp_path / "telemetry.sqlite")
    now = datetime.now(timezone.utc)
    for idx in range(5):
        store.record(
            ProviderTelemetryEvent(
                provider_name="yfinance",
                endpoint_type="price_history",
                symbol=f"T{idx}",
                success=True,
                request_timestamp=(now - timedelta(days=idx)).isoformat(),
            )
        )

    age_result = store.safe_cleanup(max_age_days=2, max_rows=10)
    assert age_result["deleted_by_age"] >= 2
    assert age_result["remaining_rows"] <= 3

    for idx in range(5, 10):
        store.record(
            ProviderTelemetryEvent(
                provider_name="marketdata_app",
                endpoint_type="options_chain",
                symbol=f"T{idx}",
                success=True,
                request_timestamp=(now + timedelta(minutes=idx)).isoformat(),
            )
        )
    cap_result = store.safe_cleanup(max_age_days=0, max_rows=3)
    assert cap_result["deleted_overflow"] >= 1
    assert cap_result["remaining_rows"] == 3
