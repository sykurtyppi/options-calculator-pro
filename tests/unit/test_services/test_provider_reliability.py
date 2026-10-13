"""Provider reliability (F3): credit budget, circuit breaker, retries, disk cache.

Before this, MarketData.app calls only retried on 429 and timeouts (a 503 or a
dropped connection failed the call outright), nothing stopped a discovery
sweep from spending the credits the exit pass needed, a provider outage was
hammered by every later candidate, the cache did not outlive the process,
and a call that never ran left no trace in the evidence funnel.
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

import scripts.run_forward_loop as forward_loop
from scripts.watch_daily_evidence_cycle import _build_combined_watchdog_status
from services import evidence_funnel as ef
from services import market_data_client as mdc
from services import provider_reliability as pr
from services import provider_telemetry as pt
from services.baseline_evidence_store import BaselineEvidenceStore
from services.outcome_recorder import OutcomeStore, make_trade_id
from services.recommendation_ledger import RecommendationLedger
from services.yfinance_market_data_client import YFinanceMarketDataClient


@pytest.fixture
def telemetry(tmp_path, monkeypatch):
    store = pt.ProviderTelemetryStore(tmp_path / "telemetry.sqlite")
    monkeypatch.setattr(pt, "_store", store)
    yield store
    store.close()


@pytest.fixture
def budget(tmp_path, monkeypatch):
    store = pr.CreditBudget(tmp_path / "budget.sqlite")
    monkeypatch.setattr(pr, "_budget", store)
    monkeypatch.delenv(pr.DAILY_CREDITS_ENV, raising=False)
    yield store
    store.close()


def _response(status=200, body=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = dict(headers or {})
    resp.json.return_value = body if body is not None else {"s": "ok", "expirations": ["2026-11-20"]}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} Client Error")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _client(monkeypatch, responses, **kwargs):
    """A MarketDataClient whose requests.get replays ``responses`` (exceptions are raised)."""
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):  # noqa: ARG001
        calls.append((url, dict(params or {})))
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(mdc.requests, "get", fake_get)
    sleeps = []
    client = mdc.MarketDataClient(token="test", sleep=sleeps.append, rng=random.Random(7),
                                  response_cache=kwargs.pop("response_cache", False), **kwargs)
    return client, calls, sleeps


@pytest.fixture
def skip_log():
    with pr.skip_log_scope():
        yield pr.start_skip_log()


def _rows(telemetry):
    return [(r["endpoint_type"], r["request_kind"], r["success"], r["error_category"])
            for r in reversed(telemetry.recent(limit=50))]


# ── retries ─────────────────────────────────────────────────────────────────


def test_retry_delay_honours_retry_after_and_jitters_exponentially():
    assert pr.retry_delay(0, retry_after="4") == 4.0
    assert pr.retry_delay(0, retry_after="900") == pr.RETRY_CAP_SECONDS
    rng = random.Random(1)
    for attempt, ceiling in ((0, 1.5), (1, 3.0), (2, 6.0), (8, pr.RETRY_CAP_SECONDS)):
        delays = [pr.retry_delay(attempt, rng=rng) for _ in range(50)]
        assert all(ceiling / 2 <= d <= ceiling for d in delays)
        assert len(set(delays)) > 1          # jittered, not fixed


@pytest.mark.parametrize("transient", [
    _response(503), _response(502), requests.exceptions.ConnectionError("reset"),
    requests.exceptions.Timeout("slow"),
])
def test_a_transient_failure_is_retried_and_logged_once(monkeypatch, telemetry, budget, transient):
    client, calls, sleeps = _client(monkeypatch, [transient, _response(200)])
    assert client.get_expirations("AAPL") == ["2026-11-20"]
    assert len(calls) == 2 and len(sleeps) == 1
    assert _rows(telemetry) == [("options_expirations", "request", True, None)]
    assert telemetry.recent(limit=1)[0]["response_quality_note"] == "retried:1"


def test_a_rate_limit_waits_for_retry_after(monkeypatch, telemetry, budget):
    client, calls, sleeps = _client(monkeypatch, [_response(429, headers={"Retry-After": "2"}), _response(200)])
    client.get_expirations("AAPL")
    assert sleeps == [2.0]


def test_client_errors_are_not_retried(monkeypatch, telemetry, budget):
    client, calls, sleeps = _client(monkeypatch, [_response(404)])
    assert client.get_expirations("AAPL") == []
    assert len(calls) == 1 and sleeps == []
    assert _rows(telemetry) == [("options_expirations", "request", False, "not_found")]
    assert pr.get_breaker("marketdata_app").consecutive_failures == 0   # not a provider-health signal


def test_exhausted_retries_log_one_failure_with_the_cause(monkeypatch, telemetry, budget):
    client, calls, sleeps = _client(monkeypatch, [_response(503)] * 3)
    assert client.get_expirations("AAPL") == []
    assert len(calls) == 3 and len(sleeps) == 2
    assert _rows(telemetry) == [("options_expirations", "request", False, "server_error")]


# ── circuit breaker ─────────────────────────────────────────────────────────


def test_the_breaker_opens_after_consecutive_failures_and_skips_calls(monkeypatch, telemetry, budget, skip_log):
    client, calls, _ = _client(monkeypatch, [_response(503)] * (3 * pr.BREAKER_FAILURE_THRESHOLD))
    for symbol in ("A", "B", "C", "D", "E"):
        client.get_expirations(symbol)
    assert len(calls) == 3 * pr.BREAKER_FAILURE_THRESHOLD
    assert pr.get_breaker("marketdata_app").state() == "open"

    log = skip_log
    assert client.get_expirations("F") == []
    assert len(calls) == 3 * pr.BREAKER_FAILURE_THRESHOLD        # not sent
    assert _rows(telemetry)[-1] == ("options_expirations", "pipeline", False, pr.SKIP_CIRCUIT_OPEN)
    assert [s["reason"] for s in log] == [pr.SKIP_CIRCUIT_OPEN]
    # Skipped calls are not provider requests: five failures, no more.
    assert telemetry.summary_by_provider()["marketdata_app"]["total"] == 5


def test_the_breaker_half_opens_after_the_cooldown():
    now = [0.0]
    breaker = pr.CircuitBreaker(threshold=2, cooldown_seconds=60, clock=lambda: now[0])
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert not breaker.allow() and breaker.state() == "open"
    now[0] = 61.0
    assert breaker.allow() and breaker.state() == "half_open"
    breaker.record_failure()                      # the trial failed: open again
    assert not breaker.allow() and breaker.open_count == 1
    now[0] = 122.0
    breaker.record_success()
    assert breaker.state() == "closed" and breaker.consecutive_failures == 0


def test_yfinance_chain_breaker_counts_outages_not_bad_symbols(telemetry):
    class _Ticker:
        def __init__(self, error):
            self.error = error

        @property
        def options(self):
            raise self.error

    errors = iter([KeyError("BADSYM")] * 8 + [Exception("Too Many Requests. Rate limited. Try after a while.")] * 5)
    client = YFinanceMarketDataClient(ticker_factory=lambda symbol: _Ticker(next(errors)))
    for _ in range(8):
        client.get_option_chain("BADSYM")
    assert pr.get_breaker("yfinance").state() == "closed"
    for _ in range(5):
        client.get_option_chain("AAPL")
    assert pr.get_breaker("yfinance").state() == "open"
    assert client.get_option_chain("MSFT").empty
    last = telemetry.recent(limit=1)[0]
    assert (last["request_kind"], last["error_category"]) == ("pipeline", pr.SKIP_CIRCUIT_OPEN)


def test_a_yfinance_success_resets_the_failure_count(telemetry):
    class _Ticker:
        def __init__(self, error):
            self.error = error

        @property
        def options(self):
            if self.error:
                raise self.error
            return []

    outage = Exception("Too Many Requests. Rate limited.")
    errors = iter([outage] * 4 + [None] + [outage] * 4)
    client = YFinanceMarketDataClient(ticker_factory=lambda symbol: _Ticker(next(errors)))
    for _ in range(9):
        client.get_option_chain("AAPL")
    assert pr.get_breaker("yfinance").state() == "closed"


# ── credit budget ───────────────────────────────────────────────────────────


def test_tiers_stop_in_order_and_exits_get_the_last_credits(budget):
    budget.charge("marketdata_app", pr.PRIORITY_DISCOVERY, 59_000)
    assert budget.allows("marketdata_app", pr.PRIORITY_DISCOVERY, 1_000)
    assert not budget.allows("marketdata_app", pr.PRIORITY_DISCOVERY, 1_001)
    budget.charge("marketdata_app", pr.PRIORITY_ENTRY, 26_000)
    assert not budget.allows("marketdata_app", pr.PRIORITY_ENTRY, 1)
    assert budget.allows("marketdata_app", pr.PRIORITY_EXIT, 15_000)
    assert not budget.allows("marketdata_app", pr.PRIORITY_EXIT, 15_001)
    assert budget.allows("yfinance", pr.PRIORITY_DISCOVERY, 10**9)       # no credits to meter


def test_the_providers_reported_quota_overrides_the_estimate(monkeypatch, telemetry, budget):
    headers = {"X-Api-Ratelimit-Limit": "10000", "X-Api-Ratelimit-Remaining": "1500",
               "X-Api-Ratelimit-Consumed": "3"}
    client, _, _ = _client(monkeypatch, [_response(200, headers=headers)])
    client.get_expirations("AAPL")
    status = budget.status()
    assert status["daily_limit"] == 10_000 and status["limit_source"] == "provider_reported"
    assert status["spent"] == 8_500                           # spend from other clients counts
    assert status["by_priority"][pr.PRIORITY_ENTRY] == {"credits": 3.0, "requests": 1, "skipped": 0}
    assert status["alert_level"] == "warning"


def test_the_configured_limit_and_live_chain_cost_are_used_without_headers(monkeypatch, telemetry, budget):
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "1000")
    chain = {"s": "ok", "optionSymbol": [f"X{i}" for i in range(40)], "strike": list(range(40)),
             "side": ["call"] * 40, "expiration": [1_800_000_000] * 40}
    client, _, _ = _client(monkeypatch, [_response(200, body=chain)])
    client.get_option_chain("AAPL", expiration="2026-11-20", strike_limit=20)
    status = budget.status()
    assert (status["daily_limit"], status["limit_source"], status["spent"]) == (1000, "env", 40)


def test_a_discovery_call_over_budget_is_skipped_not_sent(monkeypatch, telemetry, budget, skip_log):
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "100")
    budget.charge("marketdata_app", pr.PRIORITY_ENTRY, 60)
    client, calls, _ = _client(monkeypatch, [_response(200), _response(200)])
    log = skip_log
    with pr.request_priority(pr.PRIORITY_DISCOVERY):
        assert client.get_expirations("AAPL") == []
    assert calls == []
    assert [(s["reason"], s["priority"]) for s in log] == [(pr.SKIP_BUDGET, pr.PRIORITY_DISCOVERY)]
    assert _rows(telemetry) == [("options_expirations", "pipeline", False, pr.SKIP_BUDGET)]
    with pr.request_priority(pr.PRIORITY_EXIT):
        assert client.get_expirations("AAPL") == ["2026-11-20"]
    assert budget.status()["by_priority"][pr.PRIORITY_DISCOVERY]["skipped"] == 1


def test_the_budget_resets_on_the_new_york_day(tmp_path):
    clock = [datetime(2026, 10, 13, 3, 59, tzinfo=timezone.utc)]     # 23:59 New York, Oct 12
    store = pr.CreditBudget(tmp_path / "b.sqlite", now=lambda: clock[0])
    try:
        store.charge("marketdata_app", pr.PRIORITY_ENTRY, 500)
        assert store.status()["budget_day"] == "2026-10-12" and store.status()["spent"] == 500
        clock[0] += timedelta(minutes=2)
        assert store.status()["budget_day"] == "2026-10-13" and store.status()["spent"] == 0
    finally:
        store.close()


def test_quota_alerts_reach_the_watchdog_and_diagnostics(monkeypatch, telemetry, budget):
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "1000")
    budget.charge("marketdata_app", pr.PRIORITY_ENTRY, 850)
    health = pr.build_provider_budget_health(budget=budget)
    assert [(i["alertable"], i["message"].split(":")[0]) for i in health["issues"]] == [
        (False, "MarketData credit budget warning")]
    base = {"ok": True, "errors": [], "warnings": []}
    combined = _build_combined_watchdog_status(watchdog_status=base, resolver_health={}, budget_health=health)
    assert combined["ok"] and combined["warnings"] and combined["provider_budget"]["alert_level"] == "warning"

    budget.charge("marketdata_app", pr.PRIORITY_EXIT, 150)
    health = pr.build_provider_budget_health(budget=budget)
    combined = _build_combined_watchdog_status(watchdog_status=base, resolver_health={}, budget_health=health)
    assert not combined["ok"] and "exhausted" in combined["errors"][0]

    flags = pt.build_provider_telemetry_diagnostics(store=telemetry)["operational_health"]["warning_flags"]
    assert "MarketData credit budget exhausted." in flags


# ── persistent cache ────────────────────────────────────────────────────────


def test_slow_endpoints_are_shared_across_processes_and_live_chains_are_not(monkeypatch, tmp_path, telemetry, budget):
    cache_path = tmp_path / "cache.sqlite"
    chain = {"s": "ok", "optionSymbol": ["X1"], "strike": [100.0], "side": ["call"],
             "expiration": [1_800_000_000]}
    first, calls, _ = _client(monkeypatch, [_response(200), _response(200, body=chain), _response(200, body=chain)],
                              response_cache=pr.ResponseCache(cache_path))
    first.get_expirations("AAPL")
    first.get_option_chain("AAPL", date="2026-01-02")
    first.get_option_chain("AAPL", expiration="2026-11-20")
    assert len(calls) == 3

    second, calls2, _ = _client(monkeypatch, [_response(200, body=chain)], response_cache=pr.ResponseCache(cache_path))
    second.get_expirations("AAPL")
    second.get_option_chain("AAPL", date="2026-01-02")
    assert calls2 == []                                  # served from disk
    second.get_option_chain("AAPL", expiration="2026-11-20")
    assert len(calls2) == 1                              # a live chain is re-fetched
    assert b"test" not in cache_path.read_bytes().replace(b"AAPL", b"")   # no token stored


def test_disk_cache_entries_expire():
    now = [1_000.0]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        cache = pr.ResponseCache(Path(tmp) / "c.sqlite", clock=lambda: now[0])
        cache.put("k", {"s": "ok"}, ttl_seconds=60)
        assert cache.get("k") == {"s": "ok"}
        now[0] += 61
        assert cache.get("k") is None
        cache.close()


# ── the forward loop: priorities and funnel dispositions ────────────────────

TODAY = date(2026, 4, 21)


def _analysis(symbol):
    earnings = TODAY + timedelta(days=6)
    return SimpleNamespace(
        symbol=symbol, recommendation="Candidate", confidence_pct=70.0, setup_score=0.6, metrics={},
        rationale=["ok"],
        selector_output={"recommendation": "Candidate", "best_structure": "atm_straddle",
                         "earnings_date": earnings.isoformat(), "binding_gate": None},
        structure_scorecards=[{"structure": "atm_straddle", "execution_penalty": 0.03}],
        vol_snapshot={"as_of_date": TODAY.isoformat(), "earnings_date": earnings.isoformat(),
                      "days_to_earnings": 6, "data_quality_score": 0.9},
    )


@pytest.fixture
def funnel(tmp_path):
    store = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    yield store
    store.close()


def test_entry_pass_priorities_and_a_skip_is_the_quote_disposition(tmp_path, funnel):
    seen = {}

    def screener(**_):
        seen["discovery"] = pr.current_priority()
        return {"rows": [{"symbol": "SKIP", "dte": 6}, {"symbol": "FALLBACK", "dte": 6}, {"symbol": "GOOD", "dte": 6}]}

    def analyzer(symbol, mda_client=None):  # noqa: ARG001
        seen[symbol] = pr.current_priority()
        return _analysis(symbol)

    def price_fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
        if symbol in {"SKIP", "FALLBACK"}:
            pr.note_skip(provider="marketdata_app", endpoint_type="options_chain", reason=pr.SKIP_BUDGET,
                         symbol=symbol)
        if symbol == "SKIP":
            return {"mid": None, "reason": "no_spot_price"}
        return {"mid": 5.0, "context": {}}

    forward_loop.run_forward_screener(
        today=TODAY, store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        ledger=RecommendationLedger(ledger_path=tmp_path / "r.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), log_path=tmp_path / "log.jsonl",
        screener_builder=screener, analyzer=analyzer, price_fetcher=price_fetcher,
        funnel=funnel, cycle_run_id="cycle-1", symbols=["SKIP", "FALLBACK", "GOOD"],
    )
    assert seen == {"discovery": "discovery", "SKIP": "entry", "FALLBACK": "entry", "GOOD": "entry"}
    events = {(e["symbol"], e["stage"]): e for e in funnel.events("cycle-1")}
    skipped = events[("SKIP", "recorded")]
    assert (skipped["outcome"], skipped["reason"]) == ("deferred", f"missing_entry_quote:{pr.SKIP_BUDGET}")
    assert json.loads(skipped["detail_json"])["quote_reason"] == "no_spot_price"
    # A skip the fallback covered is recorded on the entry, and does not leak
    # into the next candidate.
    fallback = events[("FALLBACK", "recorded")]
    assert fallback["outcome"] == "advanced"
    assert json.loads(fallback["detail_json"])["provider_skips"][0]["reason"] == pr.SKIP_BUDGET
    assert "provider_skips" not in json.loads(events[("GOOD", "recorded")]["detail_json"])
    assert pr.current_priority() == pr.DEFAULT_PRIORITY and pr.current_skips() == []


def test_exit_pass_runs_at_exit_priority_and_records_the_skip(tmp_path, funnel):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    earnings = TODAY + timedelta(days=1)
    trade_id = make_trade_id("SHOP", TODAY - timedelta(days=4), "atm_straddle", earnings_date=earnings)
    store.insert_entry(
        trade_id=trade_id, symbol="SHOP", structure="atm_straddle", entry_date=TODAY - timedelta(days=4),
        setup_score=0.6, source_type="paper", earnings_date=earnings, entry_mid=5.0,
        execution_penalty_at_entry=0.03, notes=json.dumps({"pricing_context": {"strike": 100.0}}),
    )
    seen = []

    def price_fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
        seen.append(pr.current_priority())
        pr.note_skip(provider="marketdata_app", endpoint_type="options_chain", reason=pr.SKIP_CIRCUIT_OPEN)
        return {"mid": None, "reason": "missing_exit_mid"}

    forward_loop.run_exit_detection(
        today=TODAY, store=store, log_path=tmp_path / "log.jsonl", price_fetcher=price_fetcher,
        finalizer=lambda **kw: {"status": "finalized"}, baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
        funnel=funnel, cycle_run_id="cycle-1",
    )
    assert seen == [pr.PRIORITY_EXIT]
    resolved = [e for e in funnel.events("cycle-1") if e["stage"] == "resolved"]
    assert [(e["outcome"], e["reason"]) for e in resolved] == [("deferred", f"exit_failed:{pr.SKIP_CIRCUIT_OPEN}")]


def test_an_already_classified_category_is_stored_as_is(telemetry):
    # record_provider_telemetry classifies again; that used to turn
    # "not_found", "auth" and "entitlement" into "unknown_error".
    for category in ("not_found", "auth", "entitlement", "server_error", pr.SKIP_BUDGET):
        pt.record_provider_telemetry(provider_name="marketdata_app", endpoint_type="quote", success=False,
                                     error_category=category)
    stored = [r["error_category"] for r in reversed(telemetry.recent(limit=10))]
    assert stored == ["not_found", "auth", "entitlement", "server_error", pr.SKIP_BUDGET]
