"""Provider telemetry that measures provider calls (F2).

The ~57% MarketData.app failure rate in the evidence report was inflated:
1. one call logged a request success AND a separate "*_quality" failure, so
   a single empty response counted as one success plus one failure;
2. the forward loop's own quote outcomes ("no usable option quote") were
   counted as provider failures;
3. the forward loop labelled chains from the default yfinance client as
   "marketdata_app" (in telemetry and in the entry quote's provenance);
4. the rate covered the whole retained table, not a recent window;
5. rows could not be tied to a run or a candidate.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

import scripts.run_forward_loop as forward_loop
from services import provider_telemetry as pt
from tests.unit.test_scripts.test_run_forward_loop import _FakeMarketDataClient, _marketdata_option_frame


@pytest.fixture
def store(tmp_path, monkeypatch):
    telemetry = pt.ProviderTelemetryStore(tmp_path / "telemetry.sqlite")
    monkeypatch.setattr(pt, "_store", telemetry)
    yield telemetry
    telemetry.close()


def _record(store, **kwargs):
    pt.record_provider_telemetry(store=store, **kwargs)


# ── 1/2. only provider calls count toward the failure rate ──────────────────


def test_a_quality_verdict_does_not_double_count_its_request(store):
    _record(store, provider_name="marketdata_app", endpoint_type="options_chain", success=True)
    _record(store, provider_name="marketdata_app", endpoint_type="options_chain_quality",
            request_kind="quality", success=False, error_category="empty_response")
    _record(store, provider_name="marketdata_app", endpoint_type="forward_loop_option_quote",
            request_kind="pipeline", success=False, error_category="no_spot_price")
    summary = store.summary_by_provider()
    assert summary["marketdata_app"]["total"] == 1 and summary["marketdata_app"]["failure_rate"] == 0.0
    assert store.summary_by_provider(request_kind=None)["marketdata_app"]["total"] == 3

    diagnostics = pt.build_provider_telemetry_diagnostics(store=store)
    assert diagnostics["totals"]["events"] == 1 and diagnostics["totals"]["rows"] == 3
    assert diagnostics["totals"]["failure_rate"] == 0.0
    kinds = diagnostics["failure_window"]["by_kind"]
    assert kinds == {"request": {"total": 1, "failures": 0}, "quality": {"total": 1, "failures": 1},
                     "pipeline": {"total": 1, "failures": 1}}


def test_marketdata_empty_chain_is_one_successful_request_plus_a_quality_verdict(store, monkeypatch):
    from unittest.mock import MagicMock

    from services import market_data_client as mdc

    response = MagicMock(status_code=200)
    response.json.return_value = {"s": "no_data"}
    monkeypatch.setattr(mdc.requests, "get", lambda *args, **kwargs: response)
    assert mdc.MarketDataClient(token="test").get_option_chain("XYZ").empty

    kinds = sorted((row["endpoint_type"], row["request_kind"], row["success"]) for row in store.recent(limit=10))
    assert kinds == [("options_chain", "request", True), ("options_chain_quality", "quality", False)]
    assert store.summary_by_provider()["marketdata_app"]["failure_rate"] == 0.0


def test_legacy_rows_are_classified_by_endpoint(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE provider_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT, request_timestamp TEXT NOT NULL, provider_name TEXT NOT NULL,
            endpoint_type TEXT NOT NULL, symbol TEXT, success INTEGER NOT NULL, error_category TEXT,
            latency_ms REAL, stale_used INTEGER NOT NULL DEFAULT 0, fallback_used INTEGER NOT NULL DEFAULT 0,
            response_quality_note TEXT)""")
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT INTO provider_telemetry (request_timestamp, provider_name, endpoint_type, success) VALUES (?,?,?,?)",
            [(now, "marketdata_app", "options_chain", 1),
             (now, "marketdata_app", "options_chain_quality", 0),      # quality verdict
             (now, "marketdata_app", "forward_loop_option_quote", 0),  # pipeline outcome
             (now, "yfinance", "options_chain_quality", 0)],           # yfinance's only row per fetch
        )
    store = pt.ProviderTelemetryStore(path)
    try:
        summary = store.summary_by_provider()
        assert summary["marketdata_app"] == {**summary["marketdata_app"], "total": 1, "failures": 0}
        assert summary["yfinance"]["total"] == 1 and summary["yfinance"]["failures"] == 1
        columns = {row[1] for row in store._conn.execute("PRAGMA table_info(provider_telemetry)")}
        assert {"request_kind", "cycle_run_id", "candidate_key"} <= columns
        assert store.count() == 4
    finally:
        store.close()


# ── 4. a recent window, broken down ─────────────────────────────────────────


def test_failure_window_ignores_old_failures_and_breaks_down_by_endpoint_and_category(store):
    old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    store.record(pt.ProviderTelemetryEvent(provider_name="marketdata_app", endpoint_type="quotes",
                                           success=False, error_category="timeout", request_timestamp=old))
    for success in (True, True, True, False):
        _record(store, provider_name="marketdata_app", endpoint_type="options_chain", success=success,
                error_category=None if success else "rate limited 429")
    _record(store, provider_name="yfinance", endpoint_type="price_history", success=True)

    diagnostics = pt.build_provider_telemetry_diagnostics(store=store)
    window = diagnostics["failure_window"]
    assert window["window_days"] == 7
    mda = window["providers"]["marketdata_app"]
    assert (mda["requests"], mda["failures"], mda["failure_rate"]) == (4, 1, 0.25)
    assert set(mda["endpoints"]) == {"options_chain"}
    assert mda["error_categories"] == {"rate_limited": 1}
    assert window["request_failure_rate"] == 0.2
    # The all-time totals still include the old failure; the health warning
    # uses the window.
    assert diagnostics["totals"]["failures"] == 2
    assert any("last 7 days is elevated" in flag for flag in diagnostics["operational_health"]["warning_flags"])


def test_an_old_failure_spike_does_not_raise_todays_warning(store):
    old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    for _ in range(5):
        store.record(pt.ProviderTelemetryEvent(provider_name="marketdata_app", endpoint_type="quotes",
                                               success=False, error_category="timeout", request_timestamp=old))
    _record(store, provider_name="marketdata_app", endpoint_type="quotes", success=True)
    flags = pt.build_provider_telemetry_diagnostics(store=store)["operational_health"]["warning_flags"]
    assert not any("elevated" in flag and "failure" in flag for flag in flags)


# ── 5. rows carry the run and the candidate ─────────────────────────────────


def test_context_tags_rows_and_is_restored(store):
    with pt.telemetry_context(cycle_run_id="run-1"):
        pt.bind_candidate("AAPL|2026-04-30")
        _record(store, provider_name="yfinance", endpoint_type="price_history")
        with pt.telemetry_context(candidate_key="MSFT|2026-05-01"):
            _record(store, provider_name="yfinance", endpoint_type="price_history")
        _record(store, provider_name="yfinance", endpoint_type="price_history")
    _record(store, provider_name="yfinance", endpoint_type="price_history")
    tags = [(row["cycle_run_id"], row["candidate_key"]) for row in store.recent(limit=10)][::-1]
    assert tags == [("run-1", "AAPL|2026-04-30"), ("run-1", "MSFT|2026-05-01"),
                    ("run-1", "AAPL|2026-04-30"), (None, None)]


def test_forward_loop_telemetry_carries_the_cycle_and_candidate(tmp_path, store):
    from services.baseline_evidence_store import BaselineEvidenceStore
    from services.outcome_recorder import OutcomeStore

    def analyzer(symbol, mda_client=None):  # noqa: ARG001
        pt.record_provider_telemetry(provider_name="yfinance", endpoint_type="price_history", symbol=symbol)
        raise ValueError("No market data")

    forward_loop.run_forward_screener(
        today=date(2026, 4, 21), store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": [{"symbol": "AAPL", "dte": 6, "earnings_date": "2026-04-27"}]},
        analyzer=analyzer, cycle_run_id="cycle-7",
    )
    (row,) = store.recent(limit=5)
    assert (row["cycle_run_id"], row["candidate_key"]) == ("cycle-7", "AAPL|2026-04-27")
    with pt.telemetry_context():
        pass
    pt.record_provider_telemetry(provider_name="yfinance", endpoint_type="price_history")
    assert store.recent(limit=1)[0]["cycle_run_id"] is None   # the run's context did not leak


# ── 3. quotes are attributed to the provider that served them ───────────────


def _strangle_chain():
    return _marketdata_option_frame([
        {"optionSymbol": "AAPL_C_103", "side": "call", "strike": 103.0, "bid": 1.0, "ask": 1.2, "mid": 1.1,
         "lastPrice": 1.1, "underlyingPrice": 100.0},
        {"optionSymbol": "AAPL_P_97", "side": "put", "strike": 97.0, "bid": 0.8, "ask": 1.0, "mid": 0.9,
         "lastPrice": 0.9, "underlyingPrice": 100.0},
    ])


@pytest.mark.parametrize("provider, expected_source, expected_quality", [
    ("yfinance", "yfinance", "paper_research_mid_not_execution_grade"),
    ("marketdata_app", "marketdata_app", "marketdata_app_paper_research_mid_not_execution_grade"),
])
def test_quote_source_is_the_clients_provider(monkeypatch, store, provider, expected_source, expected_quality):
    monkeypatch.setattr(forward_loop.yf, "Ticker",
                        lambda _symbol: (_ for _ in ()).throw(AssertionError("raw yfinance not expected")))
    client = _FakeMarketDataClient(_strangle_chain())
    client.provider_name = provider
    quote = forward_loop.fetch_structure_quote(symbol="AAPL", structure="otm_strangle",
                                               earnings_date=date(2026, 4, 30), as_of_date=date(2026, 4, 24),
                                               mda_client=client)
    assert quote["mid"] is not None
    assert (quote["quote_source"], quote["quote_quality"]) == (expected_source, expected_quality)
    (row,) = [r for r in store.recent(limit=10) if r["endpoint_type"] == "forward_loop_option_quote"]
    assert (row["provider_name"], row["request_kind"], row["fallback_used"]) == (expected_source, "pipeline", False)


def test_an_unusable_client_chain_is_a_pipeline_row_for_that_provider(monkeypatch, store):
    client = _FakeMarketDataClient(pd.DataFrame())
    client.provider_name = "yfinance"
    monkeypatch.setattr(forward_loop.yf, "Ticker",
                        lambda _symbol: (_ for _ in ()).throw(RuntimeError("raw yfinance down")))
    with pytest.raises(RuntimeError):
        forward_loop.fetch_structure_quote(symbol="AAPL", structure="otm_strangle",
                                           earnings_date=date(2026, 4, 30), as_of_date=date(2026, 4, 24),
                                           mda_client=client)
    rows = [r for r in store.recent(limit=10) if r["endpoint_type"] == "forward_loop_option_quote"]
    assert rows and all(r["request_kind"] == "pipeline" for r in rows)
    assert all(r["provider_name"] == "yfinance" for r in rows)
    assert store.summary_by_provider() == {}   # no provider call failed
