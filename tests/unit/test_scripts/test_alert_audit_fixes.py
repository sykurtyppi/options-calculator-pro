"""Alert-delivery findings from the independent audit of main at bfc48f5.

P0  a run that checked "no quarantine marker" just before another run
    quarantined the outbox then created a fresh outbox and resent;
P0  an empty outbox file, or one without its table, was treated as fresh;
P2  concurrent quarantines left the marker naming a file that was not there;
P2  a failed quarantine move was logged as done, and its -wal was left behind;
P2  a fork failure was logged as "after osascript started" (delivery unknown),
    and nothing let an operator resolve a delivery_unknown alert;
P2  --outbox-reconciled did nothing on a day when nothing qualified;
P1  the watchdog's iMessage retried after a timeout by default, and an
    unreadable watchdog state was read as "nothing sent".
"""
from __future__ import annotations

import errno
import gc
import json
import multiprocessing
import os
import pathlib
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.premarket_screener_alert as alert
from services import alert_outbox as ob
from services import automation_watchdog as wd
from services.external_io_gate import Category
from tests.unit.test_scripts.alert_fixtures import alert_row

OUTBOX = "premarket_alert_outbox.sqlite"


def _table(rows=None):
    rows = [alert_row("TSLA", 0.90)] if rows is None else rows
    return {"rows": rows, "universe_size": 187, "in_entry_window": len(rows)}


def _run(tmp_path, *, send=None, extra=(), table=None):
    calls = []

    def fake_send(message, *, config, retries=0, **_):
        calls.append(retries)
        if send is not None:
            return send(message)
        return {"to": "***1"}

    with patch.object(alert, "build_ranked_screener", return_value=table or _table()), \
            patch.object(alert.IMessageConfig, "from_env", return_value=object()), \
            patch.object(alert, "send_imessage", side_effect=fake_send):
        code = alert.main(["--state-path", str(tmp_path / "state.json"),
                           "--audit-path", str(tmp_path / "audit.jsonl"), *extra])
    return code, calls


# ── P0: the marker check and the outbox open are one critical section ──────

def test_a_run_waits_for_a_quarantine_in_progress_and_then_sends_nothing(tmp_path):
    assert _run(tmp_path)[0] == 0
    (tmp_path / "state.json").unlink()                     # only the outbox remembers the send
    (tmp_path / OUTBOX).write_bytes(b"this is not a database" * 100)
    for side in ("-wal", "-shm"):
        (tmp_path / (OUTBOX + side)).unlink(missing_ok=True)
    started = threading.Event()
    results = {}

    def quarantining_run():
        with ob.outbox_lock(tmp_path / OUTBOX):
            started.set()
            time.sleep(0.3)                                 # another run checks the marker meanwhile
            results["moved"] = ob.quarantine_outbox(tmp_path / OUTBOX, reason="test")

    worker = threading.Thread(target=quarantining_run)
    worker.start()
    started.wait(5)
    code, calls = _run(tmp_path)                           # waits for the lock, then sees the marker
    worker.join()
    assert results["moved"] is not None
    assert code == 1 and calls == []
    assert not (tmp_path / OUTBOX).exists()                # no fresh outbox was created


def _concurrent_run(args):
    tmp, = args
    try:
        return _run(Path(tmp))[0]
    except BaseException as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def test_many_runs_against_a_damaged_outbox_quarantine_once_and_send_nothing(tmp_path):
    assert _run(tmp_path)[0] == 0
    (tmp_path / "state.json").unlink()
    (tmp_path / OUTBOX).write_bytes(b"this is not a database" * 100)
    for side in ("-wal", "-shm"):
        (tmp_path / (OUTBOX + side)).unlink(missing_ok=True)
    sends = tmp_path / "sends"

    def recorder(message):
        with open(sends, "a") as fh:
            fh.write("x\n")
        return {"to": "***1"}

    ctx = multiprocessing.get_context("fork")
    with patch.object(alert, "send_imessage", side_effect=lambda *a, **k: recorder(a[0])):
        with ctx.Pool(8) as pool:
            codes = pool.map(_concurrent_run, [(str(tmp_path),)] * 8)
    assert codes == [1] * 8
    assert not sends.exists()
    marker = ob.read_tombstone(tmp_path / OUTBOX)
    quarantined = sorted(p for p in tmp_path.glob(f"{OUTBOX}.corrupt-*") if not p.name.endswith(("-wal", "-shm")))
    assert len(quarantined) == 1 and marker["moved_to"] == str(quarantined[0])


# ── P0: an empty outbox, or one without its table, is damage ───────────────

def test_an_empty_outbox_file_is_quarantined_not_reused(tmp_path):
    assert _run(tmp_path)[0] == 0
    gc.collect()                                           # the run's connection, so nothing writes back
    (tmp_path / "state.json").unlink()
    (tmp_path / OUTBOX).write_bytes(b"")
    for side in ("-wal", "-shm"):
        (tmp_path / (OUTBOX + side)).unlink(missing_ok=True)
    code, calls = _run(tmp_path)
    assert code == 1 and calls == []
    assert ob.read_tombstone(tmp_path / OUTBOX)["reason"] == "outbox file is empty"


def test_an_outbox_without_its_table_is_damage(tmp_path):
    outbox = ob.AlertOutbox(tmp_path / OUTBOX)
    outbox.reserve("k", event_id="e")
    outbox.close()
    with sqlite3.connect(tmp_path / OUTBOX) as conn:
        conn.execute("DROP TABLE alert_outbox")
    with pytest.raises(ob.OutboxCorrupt, match="no alert_outbox table"):
        ob.AlertOutbox(tmp_path / OUTBOX)


def test_a_new_outbox_is_created_whole(tmp_path):
    path = tmp_path / "nested" / OUTBOX
    outbox = ob.AlertOutbox(path)
    try:
        assert outbox.reserve("k", event_id="e").acquired
    finally:
        outbox.close()
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".new")]
    ob._create_outbox_file(path)                            # an existing file is never replaced
    assert ob.AlertOutbox(path).get("k")["status"] == ob.PREPARED


# ── P2: quarantine is exclusive and honest about failures ──────────────────

def _damaged(tmp_path):
    path = tmp_path / OUTBOX
    path.write_bytes(b"this is not a database" * 100)
    return path


def test_a_second_quarantine_does_not_overwrite_the_first_marker(tmp_path):
    path = _damaged(tmp_path)
    moved = ob.quarantine_outbox(path, reason="first")
    path.write_bytes(b"also damaged")                       # a racer's view of the same path
    assert ob.quarantine_outbox(path, reason="second") is None
    marker = ob.read_tombstone(path)
    assert marker["reason"] == "first" and marker["moved_to"] == str(moved) and moved.exists()


def test_a_file_replaced_since_it_was_found_damaged_is_not_moved(tmp_path):
    path = _damaged(tmp_path)
    inode = path.stat().st_ino
    replacement = tmp_path / "other"
    replacement.write_bytes(b"x")
    os.replace(replacement, path)
    assert ob.quarantine_outbox(path, reason="r", expected_inode=inode) is None
    assert path.exists() and ob.read_tombstone(path)["moved_to"] is None


def test_side_files_move_first_and_a_failed_move_is_reported(tmp_path):
    path = _damaged(tmp_path)
    (tmp_path / (OUTBOX + "-wal")).write_bytes(b"wal")
    real_replace = pathlib.Path.replace

    def busy_main(self, target):
        if self.name == OUTBOX:
            raise OSError(errno.EBUSY, "Resource busy")
        return real_replace(self, target)

    with patch.object(pathlib.Path, "replace", busy_main):
        with pytest.raises(ob.QuarantineIncomplete, match=f"moving {OUTBOX} failed"):
            ob.quarantine_outbox(path, reason="r")
    assert not (tmp_path / (OUTBOX + "-wal")).exists()      # never left to be replayed into a new file
    marker = ob.read_tombstone(path)
    assert "Resource busy" in marker["move_failed"]


def test_the_run_logs_a_failed_quarantine_move_as_failed(tmp_path, caplog):
    _damaged(tmp_path)
    real_replace = pathlib.Path.replace

    def busy_main(self, target):
        if self.name == OUTBOX:
            raise OSError(errno.EBUSY, "Resource busy")
        return real_replace(self, target)

    with patch.object(pathlib.Path, "replace", busy_main):
        code, calls = _run(tmp_path)
    assert code == 1 and calls == []
    assert "moving it aside failed" in caplog.text and "moved to" not in caplog.text
    assert _run(tmp_path) == (1, [])                       # the marker keeps blocking


# ── P2: a hand-off that never started is not "delivery unknown" ────────────

def test_a_fork_failure_is_a_failed_send_not_an_unknown_one(tmp_path):
    def fork_fails(message):
        raise wd.HandoffNotStarted(errno.EAGAIN, "osascript not started: Resource temporarily unavailable")

    code, calls = _run(tmp_path, send=fork_fails)
    assert code == 1 and len(calls) == 1
    outbox = ob.AlertOutbox(tmp_path / OUTBOX)
    (row,) = outbox._conn.execute("SELECT status FROM alert_outbox").fetchall()
    assert row[0] == ob.FAILED
    outbox.close()
    code, calls = _run(tmp_path)                           # nothing went out: the next run may send
    assert code == 0 and len(calls) == 1


def test_send_imessage_reports_a_process_that_never_started(monkeypatch):
    def no_fork(*_a, **_k):
        raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(wd.subprocess, "Popen", no_fork)
    wd.external_io_gate.enable(Category.IMESSAGE)
    try:
        with pytest.raises(wd.HandoffNotStarted):
            wd.send_imessage("hi", config=wd.IMessageConfig(to_address="+15555550100"))
    finally:
        wd.external_io_gate.disable(Category.IMESSAGE)
    assert alert._never_launched(wd.HandoffNotStarted(errno.EAGAIN, "x"))
    assert not alert._never_launched(subprocess.TimeoutExpired(cmd="osascript", timeout=30))


def test_send_imessage_kills_a_hand_off_that_times_out(monkeypatch):
    class _Slow:
        killed = False
        returncode = None

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="osascript", timeout=timeout)
            return "", ""

        def kill(self):
            _Slow.killed = True

    monkeypatch.setattr(wd.subprocess, "Popen", lambda *a, **k: _Slow())
    with pytest.raises(subprocess.TimeoutExpired):
        wd._run_osascript(["osascript"], timeout_seconds=0.01)
    assert _Slow.killed


def test_an_operator_resolves_a_delivery_unknown_alert(tmp_path, capsys):
    def timeout(message):
        raise subprocess.TimeoutExpired(cmd="osascript", timeout=30)

    assert _run(tmp_path, send=timeout)[0] == 1
    outbox = ob.AlertOutbox(tmp_path / OUTBOX)
    ((key, status),) = [(r["alert_key"], r["status"]) for r in outbox.unresolved()]
    outbox.close()
    assert status == ob.DELIVERY_UNKNOWN
    assert _run(tmp_path, extra=("--list-unresolved",))[0] == 0
    assert key in capsys.readouterr().out
    assert _run(tmp_path, extra=("--resolve-delivery", key, "not_delivered")) == (0, [])
    code, calls = _run(tmp_path)                           # resolved as not delivered: sent again
    assert code == 0 and len(calls) == 1
    assert _run(tmp_path, extra=("--resolve-delivery", key, "delivered"))[0] == 1   # accepted now
    with pytest.raises(SystemExit):
        alert.parse_args(["--resolve-delivery", key, "maybe"])


def test_only_unknown_or_stale_submissions_can_be_resolved(tmp_path):
    clock = [datetime(2026, 10, 19, 12, tzinfo=timezone.utc)]
    outbox = ob.AlertOutbox(tmp_path / OUTBOX, clock=lambda: clock[0])
    outbox.reserve("k", event_id="e")
    with pytest.raises(ob.InvalidTransition):
        outbox.resolve_unknown("k", delivered=True)        # still only prepared
    outbox.transition("k", event_id="e", to=ob.PROVIDER_SUBMITTED)
    with pytest.raises(ob.InvalidTransition):
        outbox.resolve_unknown("k", delivered=True)        # a live run may still be sending
    clock[0] += timedelta(seconds=ob.RESERVATION_LEASE_SECONDS + 1)
    assert [row["alert_key"] for row in outbox.unresolved()] == ["k"]
    assert outbox.resolve_unknown("k", delivered=True) == ob.ACCEPTED
    with pytest.raises(ob.InvalidTransition):
        outbox.resolve_unknown("missing", delivered=True)


# ── P2: --outbox-reconciled works on any day ────────────────────────────────

def test_reconciling_clears_the_marker_even_when_nothing_qualifies(tmp_path):
    ob.quarantine_outbox(_damaged(tmp_path), reason="r")
    code, calls = _run(tmp_path, extra=("--outbox-reconciled",), table=_table(rows=[]))
    assert (code, calls) == (0, [])
    assert ob.read_tombstone(tmp_path / OUTBOX) is None


# ── P1: the watchdog never double-sends ─────────────────────────────────────

def _failing_status():
    return {"ok": False, "expected_date": "2026-10-19", "errors": ["boom"], "report_path": "r"}


def test_an_unreadable_watchdog_state_sends_nothing_this_tick(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text("{not json")
    sent = []
    monkeypatch.setattr(wd, "send_imessage", lambda *a, **k: sent.append(1) or {"sent": True})
    result = wd.maybe_send_watchdog_alert(_failing_status(), imessage_config=object(), state_path=state)
    assert result["attempted"] is False and result["reason"] == "alert_state_unreadable"
    assert sent == [] and not state.exists() and Path(result["moved_to"]).exists()
    # The next tick starts clean and alerts once.
    assert wd.maybe_send_watchdog_alert(_failing_status(), imessage_config=object(),
                                        state_path=state)["attempted"] is True
    assert wd.maybe_send_watchdog_alert(_failing_status(), imessage_config=object(),
                                        state_path=state)["reason"] == "duplicate_alert_suppressed"
    assert sent == [1]


def test_a_watchdog_hand_off_that_timed_out_is_not_sent_again(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    sent = []

    def timeout(*_a, **_k):
        sent.append(1)
        raise subprocess.TimeoutExpired(cmd="osascript", timeout=30)

    monkeypatch.setattr(wd, "send_imessage", timeout)
    first = wd.maybe_send_watchdog_alert(_failing_status(), imessage_config=object(), state_path=state)
    assert first["delivery_unknown"] is True and first["state_persisted"] is True
    second = wd.maybe_send_watchdog_alert(_failing_status(), imessage_config=object(), state_path=state)
    assert second["reason"] == "duplicate_alert_suppressed" and sent == [1]


def test_a_watchdog_hand_off_that_never_started_is_retried_next_tick(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    monkeypatch.setattr(wd, "send_imessage", lambda *a, **k: (_ for _ in ()).throw(
        wd.HandoffNotStarted(errno.ENOENT, "osascript not started")))
    result = wd.maybe_send_watchdog_alert(_failing_status(), imessage_config=object(), state_path=state)
    assert result["sent"] is False and not state.exists()


def test_send_imessage_does_not_retry_by_default():
    import inspect

    assert inspect.signature(wd.send_imessage).parameters["retries"].default == 0


def test_creating_the_outbox_never_replaces_a_file_that_appeared_meanwhile(tmp_path):
    path = tmp_path / OUTBOX
    path.write_bytes(b"another run's outbox")
    real_exists = pathlib.Path.exists
    with patch.object(pathlib.Path, "exists", lambda self: False if self == path else real_exists(self)):
        ob._create_outbox_file(path)                        # it looked absent a moment ago
    assert path.read_bytes() == b"another run's outbox"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".new")]
