"""PR L: the credit budget keeps its promise when the ledger is busy, late or broken.

Regression tests for the audit of main at 469c6ff:
- settlement released the hold and charged the actual cost in two
  transactions, so a concurrent reservation fit into the gap (60 spent + 60
  reserved against a 60-credit discovery ceiling), and a failure between the
  two erased the cost of a served call;
- an unreadable ledger turned every call into an unmetered one;
- a provider-reported reset later than local midnight was forgotten when the
  local budget day changed (exhausted -> ok before the provider reset);
- a late response with an older, higher remaining balance overwrote a newer
  lower one (10 -> 90).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services import market_data_client as mdc
from services import provider_reliability as pr
from tests.unit.test_services.test_provider_reliability import _client, _response, budget, telemetry  # noqa: F401

MD = pr.MARKETDATA_PROVIDER
T0 = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)


def _store(tmp_path, clock, name="b.sqlite"):
    return pr.CreditBudget(tmp_path / name, now=lambda: clock[0])


# ── settlement is one transaction ───────────────────────────────────────────


def test_no_reservation_fits_between_releasing_a_hold_and_charging_it(tmp_path, monkeypatch):
    # The audit's repro: 60-credit discovery ceiling (limit 100), A holds 60
    # and settles 60 while B tries to reserve 60 mid-settlement.
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "100")
    clock = [T0]
    a, b = _store(tmp_path, clock), _store(tmp_path, clock)
    b._conn.execute("PRAGMA busy_timeout=0")
    try:
        held = a.reserve(MD, pr.PRIORITY_DISCOVERY, 60)
        attempts = []
        real_bump = a._bump_locked

        def bump_then_compete(*args, **kwargs):
            try:
                attempts.append(b.reserve(MD, pr.PRIORITY_DISCOVERY, 60))
            except sqlite3.OperationalError as exc:
                attempts.append(str(exc))
            return real_bump(*args, **kwargs)

        monkeypatch.setattr(a, "_bump_locked", bump_then_compete)
        a.settle(held, MD, pr.PRIORITY_DISCOVERY, 60)
        assert attempts and "locked" in str(attempts[0])        # B waited: no gap to fit into
        assert b.reserve(MD, pr.PRIORITY_DISCOVERY, 60) is None
        status = a.status()
        assert (status["spent"], status["reserved"]) == (60, 0)
    finally:
        a.close()
        b.close()


def test_a_failed_settlement_keeps_the_hold_and_it_ends_as_unreconciled_spend(tmp_path, monkeypatch):
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "100")
    clock = [T0]
    store = _store(tmp_path, clock)
    try:
        held = store.reserve(MD, pr.PRIORITY_DISCOVERY, 40)

        def broken(*_args, **_kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(store, "_bump_locked", broken)
        with pytest.raises(sqlite3.OperationalError):
            store.settle(held, MD, pr.PRIORITY_DISCOVERY, 40)
        assert store.status()["reserved"] == 40                 # rolled back: the hold is still there
        monkeypatch.undo()
        monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "100")
        clock[0] += timedelta(seconds=pr.RESERVATION_TTL_SECONDS + 1)
        store.reserve(MD, pr.PRIORITY_EXIT, 1)                   # any write moves stale holds over
        status = store.status()
        assert (status["unreconciled"], status["spent"]) == (40, 40)
        health = pr.build_provider_budget_health(budget=store, unmetered_log=tmp_path / "none.jsonl")
        assert health["summary"]["unreconciled"] == 40
        assert any("never settled" in issue["message"] for issue in health["issues"])
    finally:
        store.close()


def test_a_served_call_whose_settlement_fails_keeps_its_hold(monkeypatch, telemetry, budget):  # noqa: F811
    client, calls, _ = _client(monkeypatch, [_response(200)])

    def broken(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(budget, "settle", broken)
    assert client.get_expirations("AAPL") == ["2026-11-20"]   # the response is not lost
    assert len(calls) == 1
    assert budget.status()["reserved"] > 0                     # and its cost is never released


# ── an unusable ledger fails closed, except an audited exit ─────────────────


class _BrokenBudget:
    def reserve(self, *_args, **_kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def record_skip(self, *_args, **_kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def release(self, *_args, **_kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def settle(self, *_args, **_kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def record_reported_quota(self, *_args, **_kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def status(self, *_args, **_kwargs):
        raise sqlite3.DatabaseError("file is not a database")


@pytest.mark.parametrize("priority", [pr.PRIORITY_DISCOVERY, pr.PRIORITY_ENTRY])
def test_discovery_and_entry_make_no_paid_call_without_a_ledger(monkeypatch, telemetry, priority):  # noqa: F811
    monkeypatch.setattr(pr, "_budget", _BrokenBudget())
    client, calls, _ = _client(monkeypatch, [_response(200)])
    with pr.request_priority(priority):
        assert client.get_expirations("AAPL") == []
    assert calls == []                                         # nothing went to the provider
    skipped = [r for r in telemetry.recent(limit=10) if r["error_category"] == pr.SKIP_BUDGET_UNAVAILABLE]
    assert len(skipped) == 1 and skipped[0]["request_kind"] == "pipeline"
    assert not Path(pr._DEFAULT_UNMETERED_LOG).exists()


def test_an_exit_goes_out_unmetered_only_with_a_durable_record(monkeypatch, telemetry):  # noqa: F811
    monkeypatch.setattr(pr, "_budget", _BrokenBudget())
    client, calls, _ = _client(monkeypatch, [_response(200)])
    with pr.request_priority(pr.PRIORITY_EXIT):
        assert client.get_expirations("AAPL") == ["2026-11-20"]
    assert len(calls) == 1
    (record,) = [json.loads(line) for line in Path(pr._DEFAULT_UNMETERED_LOG).read_text().splitlines()]
    assert (record["priority"], record["endpoint_type"], record["symbol"]) == ("exit", "options_expirations", "AAPL")
    assert "not a database" in record["ledger_error"]
    # The bypass alerts, as does the unreadable ledger itself.
    health = pr.build_provider_budget_health(budget=_BrokenBudget())
    assert health["summary"]["unmetered_exit_calls_7d"] == 1
    messages = [issue["message"] for issue in health["issues"] if issue["alertable"]]
    assert any("UNMETERED" in message for message in messages)
    assert any("discovery and entry calls are refused" in message for message in messages)


def test_an_exit_bypass_that_cannot_be_recorded_is_not_taken(monkeypatch, tmp_path, telemetry):  # noqa: F811
    monkeypatch.setattr(pr, "_budget", _BrokenBudget())
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setattr(pr, "_DEFAULT_UNMETERED_LOG", blocker / "log.jsonl")
    client, calls, _ = _client(monkeypatch, [_response(200)])
    with pr.request_priority(pr.PRIORITY_EXIT):
        assert client.get_expirations("AAPL") == []
    assert calls == []


def test_the_advisory_check_fails_closed_too(monkeypatch):
    monkeypatch.setattr(pr, "get_credit_budget", lambda: _BrokenBudget().status())
    assert pr.blocked_reason(MD, pr.PRIORITY_DISCOVERY) == pr.SKIP_BUDGET_UNAVAILABLE
    assert pr.blocked_reason(MD, pr.PRIORITY_EXIT) is None


def test_old_bypass_records_age_out_of_the_alert(tmp_path):
    log = tmp_path / "log.jsonl"
    old = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
    log.write_text(json.dumps({"at": old}) + "\n" + "garbage\n")
    since = datetime.now(timezone.utc) - timedelta(days=7)
    assert pr.unmetered_exit_calls(since=since, log_path=log) == [{"at": None, "unreadable": True}]


# ── the provider's reset, not local midnight ────────────────────────────────


def test_a_reported_reset_after_local_midnight_still_holds(tmp_path, monkeypatch):
    # 23:00 New York; the provider reports exhaustion and a reset at 09:30.
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "100")
    clock = [datetime(2026, 10, 15, 3, 0, tzinfo=timezone.utc)]
    reset = datetime(2026, 10, 15, 13, 30, tzinfo=timezone.utc)
    store = _store(tmp_path, clock)
    try:
        store.record_reported_quota(MD, reported_limit=100, reported_remaining=0, reported_reset_at=reset)
        assert store.status()["alert_level"] == "exhausted"
        clock[0] = datetime(2026, 10, 15, 5, 0, tzinfo=timezone.utc)    # 01:00: a new local day
        assert store.status()["budget_day"] == "2026-10-15"
        assert store.status()["alert_level"] == "exhausted"             # still before the provider's reset
        assert store.reserve(MD, pr.PRIORITY_EXIT, 1) is None
        clock[0] = reset + timedelta(seconds=1)
        assert store.status()["alert_level"] == "ok"
    finally:
        store.close()


# ── quota updates arrive out of order ───────────────────────────────────────


def test_a_late_response_cannot_raise_the_remaining_balance(tmp_path):
    clock = [T0]
    reset = T0 + timedelta(hours=6)
    store = _store(tmp_path, clock)
    try:
        store.record_reported_quota(MD, reported_limit=100, reported_remaining=10, reported_reset_at=reset)
        store.record_reported_quota(MD, reported_limit=100, reported_remaining=90, reported_reset_at=reset)
        assert store.status()["spent"] == 90
        # Without reset headers, the same holds within the budget day.
        store.record_reported_quota(MD, reported_remaining=95)
        store.record_reported_quota(MD, reported_remaining=50)
        assert store.status()["spent"] == 90                   # the reset window is the current one
    finally:
        store.close()


def test_a_later_window_replaces_and_a_stale_one_is_ignored(tmp_path):
    clock = [T0]
    store = _store(tmp_path, clock)
    try:
        first = T0 + timedelta(hours=1)
        store.record_reported_quota(MD, reported_limit=100, reported_remaining=5, reported_reset_at=first)
        clock[0] = first + timedelta(minutes=5)
        second = first + timedelta(days=1)
        store.record_reported_quota(MD, reported_limit=100, reported_remaining=99, reported_reset_at=second)
        assert store.status()["spent"] == 1                    # a fresh window
        store.record_reported_quota(MD, reported_limit=100, reported_remaining=2, reported_reset_at=first)
        assert store.status()["spent"] == 1                    # a response from the window that reset
    finally:
        store.close()


def test_a_charge_with_headers_folds_quota_in_the_same_transaction(tmp_path):
    clock = [T0]
    store = _store(tmp_path, clock)
    try:
        store.charge(MD, pr.PRIORITY_ENTRY, 3, reported_limit=100, reported_remaining=40,
                     reported_reset_at=T0 + timedelta(hours=2))
        store.charge(MD, pr.PRIORITY_ENTRY, 3, reported_limit=100, reported_remaining=70,
                     reported_reset_at=T0 + timedelta(hours=2))
        status = store.status()
        assert status["spent"] == 60 and status["by_priority"]["entry"]["requests"] == 2
    finally:
        store.close()


def test_the_ledger_does_not_bypass_for_other_providers():
    assert pr.unmetered_exit_allowed(provider="x", priority=pr.PRIORITY_ENTRY, endpoint_type="e", symbol=None,
                                     error=RuntimeError("x")) is False


def test_skip_reason_is_a_known_category():
    from services.provider_telemetry import classify_error

    assert classify_error(pr.SKIP_BUDGET_UNAVAILABLE) == pr.SKIP_BUDGET_UNAVAILABLE
    assert mdc.MarketDataError


# ── independent audit of main at bfc48f5 ─────────────────────────────────────


def test_a_failed_single_write_never_leaves_the_ledger_stuck(tmp_path):
    # A lock timeout on release() used to leave an implicit transaction open,
    # so every later reserve() raised and the healthy ledger looked unusable.
    from services.sqlite_helpers import open_db_conn

    store = pr.CreditBudget(tmp_path / "b.sqlite")
    try:
        store._conn.execute("PRAGMA busy_timeout=100")
        held = store.reserve(MD, pr.PRIORITY_ENTRY, 10)
        other = open_db_conn(tmp_path / "b.sqlite", busy_timeout_ms=0)
        other.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError):
            store.release(held)
        other.rollback()
        other.close()
        assert store._conn.in_transaction is False
        assert store.reserve(MD, pr.PRIORITY_ENTRY, 1)
    finally:
        store.close()


def test_a_failed_skip_count_never_leaves_the_ledger_stuck(tmp_path):
    # record_skip is one write without its own transaction: in autocommit it
    # cannot leave one open when the lock times out.
    from services.sqlite_helpers import open_db_conn

    store = pr.CreditBudget(tmp_path / "b.sqlite")
    try:
        store._conn.execute("PRAGMA busy_timeout=100")
        other = open_db_conn(tmp_path / "b.sqlite", busy_timeout_ms=0)
        other.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError):
            store.record_skip(MD, pr.PRIORITY_DISCOVERY)
        other.rollback()
        other.close()
        assert store._conn.in_transaction is False
        assert store.reserve(MD, pr.PRIORITY_ENTRY, 1)
    finally:
        store.close()


def test_a_cache_write_that_fails_is_logged_not_raised(tmp_path):
    from services.sqlite_helpers import open_db_conn

    cache = pr.ResponseCache(tmp_path / "cache.sqlite")
    cache._conn.execute("PRAGMA busy_timeout=50")
    blocker = open_db_conn(tmp_path / "cache.sqlite", busy_timeout_ms=0)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        cache.put("k", {"a": 1}, 60)                 # must not raise
    finally:
        blocker.rollback()
        blocker.close()
    try:
        assert cache._conn.in_transaction is False
        cache.put("k", {"a": 1}, 60)
        assert cache.get("k") == {"a": 1}
    finally:
        cache.close()


def test_an_exhaustion_without_a_reset_lands_in_the_current_window(tmp_path):
    clock = [T0]
    store = _store(tmp_path, clock)
    try:
        reset = T0 + timedelta(hours=9)
        store.record_reported_quota(MD, reported_limit=100_000, reported_remaining=50_000, reported_reset_at=reset)
        store.record_reported_quota(MD, reported_remaining=0.0)        # a 429 "quota used up" with no headers
        assert store.status()["alert_level"] == "exhausted"
        assert store.reserve(MD, pr.PRIORITY_DISCOVERY, 1) is None
        clock[0] = reset + timedelta(minutes=1)                       # the provider reset as it said
        assert store.status()["alert_level"] == "ok"
    finally:
        store.close()


def test_a_day_row_from_before_a_passed_reset_is_not_read(tmp_path):
    clock = [T0]
    store = _store(tmp_path, clock)
    try:
        store.record_reported_quota(MD, reported_remaining=0.0)        # no reset known yet: the day row
        reset = T0 + timedelta(hours=1)
        clock[0] = T0 + timedelta(minutes=5)
        store.record_reported_quota(MD, reported_limit=100, reported_remaining=0, reported_reset_at=reset)
        clock[0] = reset + timedelta(minutes=1)
        assert store.status()["alert_level"] == "ok"                   # not blocked until local midnight
        assert store.reserve(MD, pr.PRIORITY_EXIT, 1)
    finally:
        store.close()


def test_a_hold_taken_before_midnight_counts_after_it(tmp_path, monkeypatch):
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "1000")
    clock = [datetime(2026, 10, 20, 3, 59, 59, tzinfo=timezone.utc)]   # 23:59:59 New York
    store = _store(tmp_path, clock)
    try:
        assert store.reserve(MD, pr.PRIORITY_DISCOVERY, 480)
        clock[0] += timedelta(seconds=2)
        assert store.status()["reserved"] == 480
        assert store.reserve(MD, pr.PRIORITY_DISCOVERY, 480) is None  # 480 + 480 > the 600 ceiling
    finally:
        store.close()


def test_a_late_settlement_is_not_counted_twice(tmp_path, monkeypatch):
    monkeypatch.setenv(pr.DAILY_CREDITS_ENV, "10000")
    clock = [T0]
    store = _store(tmp_path, clock)
    try:
        slow = store.reserve(MD, pr.PRIORITY_ENTRY, 400)
        late = store.reserve(MD, pr.PRIORITY_ENTRY, 300)
        clock[0] += timedelta(seconds=pr.RESERVATION_TTL_SECONDS + 1)
        store.reserve(MD, pr.PRIORITY_EXIT, 1)                         # moves both to unreconciled
        store.settle(slow, MD, pr.PRIORITY_ENTRY, 400)
        store.release(late)
        status = store.status()
        assert (status["spent"], status["unreconciled"]) == (400, 0)
    finally:
        store.close()


def test_a_zero_limit_is_exhausted_not_ok(tmp_path):
    store = _store(tmp_path, [T0])
    try:
        store.record_reported_quota(MD, reported_limit=0, reported_remaining=0)
        health = pr.build_provider_budget_health(budget=store, unmetered_log=tmp_path / "none.jsonl")
        assert health["summary"]["alert_level"] == "exhausted" and health["issues"]
    finally:
        store.close()


def test_a_timezone_less_bypass_line_is_unreadable_not_a_crash(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"at": "2026-10-19T10:00:00"}) + "\n" + json.dumps(["not", "a", "dict"]) + "\n")
    calls = pr.unmetered_exit_calls(since=datetime.now(timezone.utc) - timedelta(days=7), log_path=log)
    assert calls == [{"at": None, "unreadable": True}] * 2


def _chain_response(body_text=None, headers=None, status=200):
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.status_code = status
    resp.headers = dict(headers or {})
    if body_text is None:
        resp.json.return_value = {"s": "ok"}
    else:
        resp.json.side_effect = ValueError("Unterminated string")
    resp.raise_for_status.return_value = None
    return resp


def test_a_served_but_unreadable_response_is_charged(monkeypatch, telemetry, budget):  # noqa: F811
    client, calls, _ = _client(monkeypatch, [_chain_response(body_text="{", headers={"X-Api-Ratelimit-Consumed": "480"}),
                                             _chain_response(body_text="{")])
    with pr.request_priority(pr.PRIORITY_DISCOVERY):
        client.get_option_chain("AAPL", expiration="2026-11-20", strike_limit=20)
        client.get_option_chain("MSFT", expiration="2026-11-20", strike_limit=20)
    status = budget.status()
    # 480 as reported, then the 40-credit estimate when nothing was reported.
    assert (status["spent"], status["reserved"]) == (520, 0)


def test_a_client_error_that_reports_a_charge_is_charged(monkeypatch, telemetry, budget):  # noqa: F811
    client, _, _ = _client(monkeypatch, [_response(404, headers={"X-Api-Ratelimit-Consumed": "1"}), _response(404)])
    client.get_expirations("AAPL")
    client.get_expirations("MSFT")
    assert (budget.status()["spent"], budget.status()["reserved"]) == (1, 0)


def test_an_all_expirations_chain_is_sized_from_the_known_list():
    assert mdc._estimated_credits("options_chain", {"strikeLimit": 10, "expiration": "all"},
                                  expirations_known=30) == 600
    assert mdc._estimated_credits("options_chain", {"strikeLimit": 10, "expiration": "all"}) == (
        10 * 2 * mdc.DEFAULT_ALL_EXPIRATIONS)
    assert mdc._estimated_credits("options_chain", {"strikeLimit": 10, "expiration": "2026-11-20"}) == 20


def test_the_client_remembers_each_symbols_expiration_count(monkeypatch, telemetry, budget):  # noqa: F811
    exps = {"s": "ok", "expirations": [f"2026-11-{d:02d}" for d in range(1, 31)]}
    client, _, _ = _client(monkeypatch, [_response(200, body=exps)])
    client.get_expirations("spy")
    assert client._expiration_counts["SPY"] == 30


def test_a_failed_cache_write_keeps_the_paid_response(tmp_path, monkeypatch, telemetry, budget):  # noqa: F811
    from services.sqlite_helpers import open_db_conn

    cache = pr.ResponseCache(tmp_path / "cache.sqlite")
    cache._conn.execute("PRAGMA busy_timeout=50")
    blocker = open_db_conn(tmp_path / "cache.sqlite", busy_timeout_ms=0)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        body = {"s": "ok", "expirations": ["2026-11-20"]}
        client, _, _ = _client(monkeypatch, [_response(200, body=body)], response_cache=cache)
        assert client.get_expirations("AAPL") == ["2026-11-20"]
    finally:
        blocker.rollback()
        blocker.close()
        cache.close()


def test_a_cache_that_raises_does_not_lose_the_paid_response(monkeypatch, telemetry, budget):  # noqa: F811
    class _Broken:
        def get_with_age(self, key):
            return None

        def put(self, key, payload, ttl):
            raise RuntimeError("disk full")

    body = {"s": "ok", "expirations": ["2026-11-20"]}
    client, _, _ = _client(monkeypatch, [_response(200, body=body)], response_cache=_Broken())
    assert client.get_expirations("AAPL") == ["2026-11-20"]


def test_a_disk_entry_keeps_its_age_in_memory(tmp_path, monkeypatch, telemetry, budget):  # noqa: F811
    wall, mono = [1_000_000.0], [5_000.0]
    cache = pr.ResponseCache(tmp_path / "cache.sqlite", clock=lambda: wall[0])
    url = f"{mdc._MARKETDATA_BASE_URL}/options/expirations/AAPL/"
    cache.put(url + "|[]", {"s": "ok", "expirations": ["2026-10-23"]}, 300)
    wall[0] += 299
    mono[0] += 299
    client, calls, _ = _client(monkeypatch, [_response(200, body={"s": "ok", "expirations": ["2026-10-30"]})],
                               response_cache=cache)
    monkeypatch.setattr(mdc.time, "monotonic", lambda: mono[0])
    assert client.get_expirations("AAPL") == ["2026-10-23"] and calls == []
    wall[0] += 290
    mono[0] += 290                                       # 589 s after it was stored: stale
    assert client.get_expirations("AAPL") == ["2026-10-30"] and len(calls) == 1
    cache.close()
