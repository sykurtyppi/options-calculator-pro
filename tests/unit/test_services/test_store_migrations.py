"""PR N: every store migrates its schema under the database write lock.

The recommendation-ledger migration race was fixed in PR G, but the audit of
main at 469c6ff reproduced the same race in the other stores: 12 processes
opening an older OutcomeStore or BaselineEvidenceStore at once crashed with
"duplicate column name: claim_contract_version", and two funnel initializers
raced to add ``superseded_by`` (one failed, so that process lost evidence
collection). Each test builds a store, removes a migrated column to make it
"older", then opens it from 12 processes released together.

Also: a funnel that stays unreadable past a grace period now fails the
evidence-health check (it was a non-alertable warning however long it lasted).
"""
from __future__ import annotations

import json
import multiprocessing
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.watch_daily_evidence_cycle import _build_combined_watchdog_status
from services import evidence_funnel as ef

PROCESSES = 12


def _open_outcome(path):
    from services.outcome_recorder import OutcomeStore

    return OutcomeStore(store_path=path)


def _open_baseline(path):
    from services.baseline_evidence_store import BaselineEvidenceStore

    return BaselineEvidenceStore(path)


def _open_funnel(path):
    from services.evidence_funnel import EvidenceFunnelStore

    return EvidenceFunnelStore(path)


def _open_budget(path):
    from services.provider_reliability import CreditBudget

    return CreditBudget(path)


def _open_telemetry(path):
    from services.provider_telemetry import ProviderTelemetryStore

    return ProviderTelemetryStore(path)


def _open_research_db(path):
    from services.institutional_ml_db import InstitutionalMLDatabase

    return InstitutionalMLDatabase(db_path=str(path))


def _open_iv_labels(path):
    import logging

    from scripts.build_earnings_iv_labels import _ensure_schema

    _ensure_schema(path, logging.getLogger("test"))


STORES = {
    # opener, table, a migrated column, rows the migration must keep
    "outcome": (_open_outcome, "outcome_trades", "claim_contract_version"),
    "baseline": (_open_baseline, "baseline_trades", "claim_contract_version"),
    "funnel": (_open_funnel, "candidate_rechecks", "superseded_by"),
    "budget": (_open_budget, "provider_reported_quota", "reported_reset_at"),
    "telemetry": (_open_telemetry, "provider_telemetry", "request_kind"),
    "research_db": (_open_research_db, "backtest_trades", "crush_confidence"),
    "iv_labels": (_open_iv_labels, "earnings_iv_decay_labels", "front_expiry"),
}


def _make_old(path: Path, name: str) -> None:
    opener, table, column = STORES[name]
    store = opener(path)
    close = getattr(store, "close", None)
    if close:
        close()
    conn = sqlite3.connect(str(path))
    conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    conn.commit()
    conn.close()


_barrier = None


def _init(barrier):
    global _barrier
    _barrier = barrier


def _open_after_barrier(args):
    name, path = args
    _barrier.wait()
    try:
        STORES[name][0](Path(path))
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return "ok"


@pytest.mark.parametrize("name", sorted(STORES))
def test_many_processes_opening_an_older_store_all_succeed(tmp_path, name):
    path = tmp_path / f"{name}.sqlite"
    _make_old(path, name)
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(PROCESSES)
    with ctx.Pool(PROCESSES, initializer=_init, initargs=(barrier,)) as pool:
        results = pool.map(_open_after_barrier, [(name, str(path))] * PROCESSES)
    assert results == ["ok"] * PROCESSES
    _, table, column = STORES[name]
    conn = sqlite3.connect(str(path))
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    conn.close()
    assert columns.count(column) == 1


# ── sustained funnel blindness alerts ───────────────────────────────────────

T0 = datetime(2026, 10, 14, 21, 0, tzinfo=timezone.utc)


class _UnreadableFunnel:
    def __init__(self, path):
        self.path = path

    def cycles_without_analysis(self):
        raise sqlite3.DatabaseError("file is not a database")


def _combined(health):
    return _build_combined_watchdog_status(
        watchdog_status={"ok": True, "errors": [], "warnings": []},
        resolver_health={"issues": [], "summary": {}},
        job_freshness={"issues": [], "summary": {}},
        funnel_health=health,
    )


def test_a_briefly_unreadable_funnel_warns_and_a_lasting_one_fails_the_check(tmp_path):
    broken = _UnreadableFunnel(tmp_path / "funnel.sqlite")
    first = ef.build_evidence_funnel_health(store=broken, now=T0)
    (issue,) = first["issues"]
    assert issue["alertable"] is False and "0.0 h" in issue["message"]
    assert _combined(first)["ok"] is True                      # one cycle: a warning only

    later = T0 + timedelta(seconds=ef.FUNNEL_UNAVAILABLE_GRACE_SECONDS)
    lasting = ef.build_evidence_funnel_health(store=broken, now=later)
    (issue,) = lasting["issues"]
    assert issue["alertable"] is True and lasting["summary"]["unavailable_since"] == T0.isoformat()
    combined = _combined(lasting)
    assert combined["ok"] is False and any("not being measured" in error for error in combined["errors"])


def test_a_readable_funnel_ends_the_outage(tmp_path):
    path = tmp_path / "funnel.sqlite"
    ef.build_evidence_funnel_health(store=_UnreadableFunnel(path), now=T0)
    store = ef.EvidenceFunnelStore(path)
    try:
        assert ef.build_evidence_funnel_health(store=store, now=T0 + timedelta(hours=2))["issues"] == []
    finally:
        store.close()
    # A new failure starts a new grace period.
    again = ef.build_evidence_funnel_health(store=_UnreadableFunnel(path), now=T0 + timedelta(hours=3))
    assert again["issues"][0]["alertable"] is False


def test_an_outage_that_cannot_be_tracked_alerts_at_once(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    health = ef.build_evidence_funnel_health(store=_UnreadableFunnel(blocker / "funnel.sqlite"), now=T0)
    (issue,) = health["issues"]
    assert issue["alertable"] is True and "unknown time" in issue["message"]


def test_the_forward_loops_failed_open_counts_toward_the_outage(monkeypatch):
    def locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ef, "get_evidence_funnel", locked)
    assert ef.open_evidence_funnel() is None
    marker = json.loads(ef._unavailable_marker().read_text())
    assert marker["observations"] == 1 and "locked" in marker["last_error"]
    assert ef.open_evidence_funnel() is None
    assert json.loads(ef._unavailable_marker().read_text())["since"] == marker["since"]
    monkeypatch.setattr(ef, "get_evidence_funnel", lambda: object())
    assert ef.open_evidence_funnel() is not None
    assert not ef._unavailable_marker().exists()


def test_the_labels_schema_step_closes_its_connection(tmp_path, monkeypatch):
    # A leaked connection's file handle drops the process's SQLite locks when
    # it is finally closed (a forked worker, say), undoing the write lock.
    import logging
    import types

    import scripts.build_earnings_iv_labels as labels

    opened = []

    class _Tracked(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    def connect(path, *args, **kwargs):
        conn = sqlite3.connect(path, *args, factory=_Tracked, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(labels, "sqlite3", types.SimpleNamespace(connect=connect))
    labels._ensure_schema(tmp_path / "labels.sqlite", logging.getLogger("test"))
    assert opened and all(conn.closed for conn in opened)
