"""PR I: provider protection that holds under the failures found in review of #168.

Reproduced before this change:
- seven consecutive yfinance chain timeouts left the breaker closed with one
  failure (success was recorded once expirations loaded, before the chains);
- a 429 reporting zero remaining credits was retried three times and the
  budget still read "ok" (quota headers were only read from successes);
- two independently admitted requests overshot the discovery ceiling (60 ->
  61: admission and charge were separate steps);
- all 16 concurrent callers passed a half-open breaker;
- the budget window was the calendar day regardless of the provider's reset.
"""
from __future__ import annotations

import multiprocessing
import threading
from datetime import datetime, timedelta, timezone

import pytest

from services import provider_reliability as pr
from services import provider_telemetry as pt
from services.yfinance_market_data_client import YFinanceMarketDataClient
from tests.unit.test_services.test_provider_reliability import _client, _response, budget, telemetry  # noqa: F401


# ── the yfinance breaker counts chain failures ──────────────────────────────


class _Ticker:
    def __init__(self, chain_error):
        self.options = ["2026-11-20"]
        self.fast_info = {"last_price": 100.0}
        self._error = chain_error

    def option_chain(self, _expiry):
        raise self._error


def test_repeated_chain_timeouts_open_the_yfinance_breaker(telemetry):  # noqa: F811
    client = YFinanceMarketDataClient(ticker_factory=lambda _s: _Ticker(TimeoutError("read timed out")))
    for _ in range(7):
        client.get_option_chain("AAPL")
    breaker = pr.get_breaker("yfinance")
    assert breaker.state() == "open" and breaker.open_count == 1
    skipped = [r for r in telemetry.recent(limit=20) if r["error_category"] == pr.SKIP_CIRCUIT_OPEN]
    assert len(skipped) == 2                              # calls 6 and 7 were not sent


def test_a_bad_symbol_neither_counts_nor_holds_the_probe():
    breaker = pr.get_breaker("yfinance")
    client = YFinanceMarketDataClient(ticker_factory=lambda _s: _Ticker(KeyError("NOPE")))
    for _ in range(7):
        client.get_option_chain("NOPE")
    assert breaker.state() == "closed" and breaker.consecutive_failures == 0


# ── one half-open probe ─────────────────────────────────────────────────────


def _opened(clock):
    breaker = pr.CircuitBreaker(threshold=1, cooldown_seconds=60, clock=lambda: clock[0])
    breaker.record_failure()
    clock[0] += 61
    return breaker


def test_half_open_lets_exactly_one_probe_through():
    clock = [0.0]
    breaker = _opened(clock)
    allowed = []
    barrier = threading.Barrier(16)

    def caller():
        barrier.wait()
        allowed.append(breaker.allow())

    threads = [threading.Thread(target=caller) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert allowed.count(True) == 1


def test_a_probe_that_reports_nothing_is_released_or_expires():
    clock = [0.0]
    breaker = _opened(clock)
    assert breaker.allow() and not breaker.allow()
    breaker.release_probe()                                # e.g. a 404: no health signal
    assert breaker.allow() and not breaker.allow()
    clock[0] += 61                                         # the probe was lost entirely
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state() == "closed" and breaker.allow() and breaker.allow()


def test_a_failed_probe_reopens_for_a_full_cooldown():
    clock = [0.0]
    breaker = _opened(clock)
    assert breaker.allow()
    breaker.record_failure()
    clock[0] += 30
    assert not breaker.allow()


# ── 429s and quota headers ──────────────────────────────────────────────────


def test_a_quota_exhausted_429_is_not_retried_and_reaches_the_budget(monkeypatch, telemetry, budget):  # noqa: F811
    exhausted = _response(429, headers={"X-Api-Ratelimit-Limit": "100", "X-Api-Ratelimit-Remaining": "0"})
    client, calls, sleeps = _client(monkeypatch, [exhausted, _response(200), _response(200)])
    assert client.get_expirations("AAPL") == []
    assert len(calls) == 1 and sleeps == []
    status = budget.status()
    assert (status["daily_limit"], status["alert_level"], status["remaining"]) == (100, "exhausted", 0)
    assert telemetry.recent(limit=1)[0]["error_category"] == "quota_exhausted"
    # Every tier now skips instead of making futile calls.
    with pr.request_priority(pr.PRIORITY_EXIT):
        assert client.get_expirations("MSFT") == []
    assert len(calls) == 1
    assert pr.get_breaker("marketdata_app").consecutive_failures == 0   # quota is the budget's job


def test_a_quota_message_without_headers_is_exhaustion_too(monkeypatch, telemetry, budget):  # noqa: F811
    resp = _response(429, body={"s": "error", "errmsg": "You have used your daily credit limit."})
    client, calls, _ = _client(monkeypatch, [resp])
    client.get_expirations("AAPL")
    assert len(calls) == 1 and budget.status()["alert_level"] == "exhausted"


def test_concurrency_throttling_is_retried(monkeypatch, telemetry, budget):  # noqa: F811
    throttled = _response(429, body={"s": "error", "errmsg": "Too many concurrent requests."},
                          headers={"Retry-After": "1"})
    client, calls, sleeps = _client(monkeypatch, [throttled, _response(200)])
    assert client.get_expirations("AAPL") == ["2026-11-20"]
    assert len(calls) == 2 and sleeps == [1.0]
    client2, calls2, _ = _client(monkeypatch, [throttled] * 3)
    client2.get_expirations("MSFT")
    assert telemetry.recent(limit=1)[0]["error_category"] == "concurrency_throttled"


def test_quota_headers_count_on_error_responses(monkeypatch, telemetry, budget):  # noqa: F811
    headers = {"X-Api-Ratelimit-Limit": "5000", "X-Api-Ratelimit-Remaining": "4000"}
    client, _, _ = _client(monkeypatch, [_response(503, headers=headers)] * 3)
    client.get_expirations("AAPL")
    status = budget.status()
    assert (status["daily_limit"], status["spent"]) == (5000, 1000)


# ── atomic credit reservations ──────────────────────────────────────────────


def test_reservations_hold_credits_until_settled_or_released(monkeypatch, budget):  # noqa: F811
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "100")
    first = budget.reserve("marketdata_app", pr.PRIORITY_DISCOVERY, 31)
    assert first and budget.status()["reserved"] == 31
    assert budget.reserve("marketdata_app", pr.PRIORITY_DISCOVERY, 30) is None     # 31 + 30 > 60
    budget.release(first)
    second = budget.reserve("marketdata_app", pr.PRIORITY_DISCOVERY, 30)
    budget.settle(second, "marketdata_app", pr.PRIORITY_DISCOVERY, 45)              # actual cost
    status = budget.status()
    assert (status["spent"], status["reserved"]) == (45, 0)
    assert budget.reserve("yfinance", pr.PRIORITY_DISCOVERY, 10**9) == pr.UNMETERED


def test_a_reservation_left_by_a_dead_process_stays_counted_as_unreconciled(tmp_path, monkeypatch):
    # PR L: the process may have been charged before it died, so its hold
    # never just stops counting; past the TTL it becomes unreconciled spend.
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "100")
    clock = [datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)]
    store = pr.CreditBudget(tmp_path / "b.sqlite", now=lambda: clock[0])
    try:
        assert store.reserve("marketdata_app", pr.PRIORITY_DISCOVERY, 60)
        assert store.reserve("marketdata_app", pr.PRIORITY_DISCOVERY, 1) is None
        clock[0] += timedelta(seconds=pr.RESERVATION_TTL_SECONDS + 1)
        assert store.status()["reserved"] == 60                # still held, even before a purge
        assert store.reserve("marketdata_app", pr.PRIORITY_DISCOVERY, 1) is None
        status = store.status()
        assert (status["reserved"], status["unreconciled"], status["spent"]) == (0, 60, 60)
        assert store.reserve("marketdata_app", pr.PRIORITY_ENTRY, 1)    # higher tiers still fit
    finally:
        store.close()


def _reserve_after(args):
    path, barrier = args
    import os

    os.environ[pr.DAILY_CREDITS_ENV] = "100"
    store = pr.CreditBudget(path)
    barrier.wait()
    granted = store.reserve("marketdata_app", pr.PRIORITY_DISCOVERY, 31) is not None
    store.close()
    return granted


def test_two_processes_cannot_both_pass_the_ceiling(tmp_path):
    path = tmp_path / "budget.sqlite"
    pr.CreditBudget(path).close()
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    procs = []
    results = ctx.Queue()

    def run():
        results.put(_reserve_after((path, barrier)))

    for _ in range(2):
        proc = ctx.Process(target=run)
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join(timeout=30)
    assert sorted(results.get(timeout=5) for _ in procs) == [False, True]


def test_a_served_call_settles_at_its_actual_cost(monkeypatch, telemetry, budget):  # noqa: F811
    chain = {"s": "ok", "optionSymbol": [f"X{i}" for i in range(120)], "strike": list(range(120)),
             "side": ["call"] * 120, "expiration": [1_800_000_000] * 120}
    client, _, _ = _client(monkeypatch, [_response(200, body=chain, headers={"X-Api-Ratelimit-Consumed": "300"})])
    client.get_option_chain("AAPL", expiration="2026-11-20", strike_limit=20)   # estimated 40
    status = budget.status()
    assert (status["spent"], status["reserved"]) == (300, 0)


def test_an_unserved_call_releases_its_reservation(monkeypatch, telemetry, budget):  # noqa: F811
    client, _, _ = _client(monkeypatch, [_response(503)] * 3)
    client.get_expirations("AAPL")
    status = budget.status()
    assert (status["spent"], status["reserved"]) == (0, 0)


def test_a_budget_skip_does_not_hold_the_half_open_probe(monkeypatch, telemetry, budget):  # noqa: F811
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "100")
    budget.charge("marketdata_app", pr.PRIORITY_ENTRY, 100)
    client, calls, _ = _client(monkeypatch, [])
    breaker = pr.get_breaker("marketdata_app")
    clock = [1000.0]
    breaker._clock = lambda: clock[0]
    breaker.consecutive_failures, breaker.opened_at = 5, 0.0          # open, cooldown passed
    client.get_expirations("AAPL")                                   # skipped by the budget
    assert calls == [] and breaker.probe_started_at is None


# ── the budget window ───────────────────────────────────────────────────────


def test_the_window_starts_at_the_configured_reset(monkeypatch):
    monkeypatch.setenv(pr.RESET_TIME_ENV, "09:30")
    before = datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc)      # 09:00 New York
    after = datetime(2026, 10, 14, 13, 31, tzinfo=timezone.utc)      # 09:31 New York
    assert (pr.budget_day(before), pr.budget_day(after)) == ("2026-10-13", "2026-10-14")
    monkeypatch.setenv(pr.RESET_TIME_ENV, "not-a-time")
    assert pr.budget_day(before) == "2026-10-14"


def test_a_reported_quota_is_dropped_once_its_reset_passes(tmp_path):
    clock = [datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)]
    store = pr.CreditBudget(tmp_path / "b.sqlite", now=lambda: clock[0])
    try:
        store.record_reported_quota("marketdata_app", reported_limit=100, reported_remaining=0,
                                    reported_reset_at=clock[0] + timedelta(hours=1))
        assert store.status()["alert_level"] == "exhausted"
        clock[0] += timedelta(hours=1, minutes=1)
        assert store.status()["alert_level"] == "ok"
    finally:
        store.close()


def test_new_categories_survive_recording(telemetry):  # noqa: F811
    for category in ("quota_exhausted", "concurrency_throttled"):
        pt.record_provider_telemetry(provider_name="marketdata_app", endpoint_type="quote", success=False,
                                     error_category=category)
    assert [r["error_category"] for r in telemetry.recent(limit=2)] == ["concurrency_throttled", "quota_exhausted"]


class _GoodTicker(_Ticker):
    def __init__(self):
        super().__init__(None)

    def option_chain(self, _expiry):
        import pandas as pd
        from types import SimpleNamespace

        legs = pd.DataFrame([{"contractSymbol": "AAPL261120C00100000", "strike": 100.0, "bid": 1.0, "ask": 1.2,
                              "lastPrice": 1.1, "impliedVolatility": 0.3, "volume": 10, "openInterest": 10}])
        return SimpleNamespace(calls=legs, puts=legs.assign(contractSymbol="AAPL261120P00100000"))


def test_a_complete_yfinance_fetch_resets_the_failure_count():
    current = {}
    client = YFinanceMarketDataClient(ticker_factory=lambda _s: current["ticker"])
    for ticker in [_Ticker(TimeoutError("t"))] * 4 + [_GoodTicker()] + [_Ticker(TimeoutError("t"))] * 4:
        current["ticker"] = ticker                     # get_quote builds its own ticker too
        client.get_option_chain("AAPL")
    breaker = pr.get_breaker("yfinance")
    assert breaker.state() == "closed" and breaker.consecutive_failures == 4


def test_a_bad_symbol_probe_lets_the_next_probe_through():
    breaker = pr.get_breaker("yfinance")
    clock = [1000.0]
    breaker._clock = lambda: clock[0]
    breaker.consecutive_failures, breaker.opened_at = 5, 0.0      # open, cooldown passed
    current = {"ticker": _Ticker(KeyError("NOPE"))}
    client = YFinanceMarketDataClient(ticker_factory=lambda _s: current["ticker"])
    client.get_option_chain("NOPE")
    current["ticker"] = _GoodTicker()
    assert not client.get_option_chain("AAPL").empty           # the next probe was allowed and closed it
    assert breaker.state() == "closed"


def test_a_client_error_probe_lets_the_next_probe_through(monkeypatch, telemetry, budget):  # noqa: F811
    breaker = pr.get_breaker("marketdata_app")
    clock = [1000.0]
    breaker._clock = lambda: clock[0]
    breaker.consecutive_failures, breaker.opened_at = 5, 0.0
    client, calls, _ = _client(monkeypatch, [_response(404), _response(200)])
    assert client.get_expirations("NOPE") == []
    assert client.get_expirations("AAPL") == ["2026-11-20"]
    assert len(calls) == 2 and breaker.state() == "closed"
