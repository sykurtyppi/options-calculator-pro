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
