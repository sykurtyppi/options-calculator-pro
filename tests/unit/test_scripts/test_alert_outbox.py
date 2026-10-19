"""PR H: the premarket alert is sent at most once, and its audit says what is known.

Reproduced failures this guards against:
- a successful hand-off followed by a failed state write sent the alert again
  on the next run;
- an unreadable state file was read as empty, so the alert was resent;
- two runs at once both sent (and shared one temp file);
- a timed-out hand-off was retried automatically (the first may already
  have gone out) and was audited as "send_failed" rather than unknown.
"""
from __future__ import annotations

import json
import multiprocessing
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.premarket_screener_alert as alert
from services import alert_outbox as ob
from tests.unit.test_scripts.alert_fixtures import alert_row


def _table():
    return {"rows": [alert_row("TSLA", 0.90)], "universe_size": 187, "in_entry_window": 1}


def _run(tmp_path, *, send=None, extra=()):
    calls = []

    def fake_send(message, *, config, retries=1, **_):
        calls.append(retries)
        if send is not None:
            return send(message)
        return {"to": "***1"}

    with patch.object(alert, "build_ranked_screener", return_value=_table()), \
            patch.object(alert.IMessageConfig, "from_env", return_value=object()), \
            patch.object(alert, "send_imessage", side_effect=fake_send):
        code = alert.main(["--state-path", str(tmp_path / "state.json"),
                           "--audit-path", str(tmp_path / "audit.jsonl"), *extra])
    return code, calls


def _statuses(tmp_path):
    path = tmp_path / "audit.jsonl"
    return [json.loads(line)["status"] for line in path.read_text().splitlines()] if path.exists() else []


def _outbox_row(tmp_path):
    outbox = ob.AlertOutbox(tmp_path / "premarket_alert_outbox.sqlite")
    try:
        (row,) = outbox._conn.execute("SELECT * FROM alert_outbox").fetchall()
        return dict(row)
    finally:
        outbox.close()


def test_a_sent_alert_moves_through_the_outbox_and_is_not_resent(tmp_path):
    code, calls = _run(tmp_path)
    assert code == 0 and calls == [0]                 # handed off once, never auto-retried
    assert _statuses(tmp_path) == ["prepared", "accepted"]
    assert _outbox_row(tmp_path)["status"] == ob.ACCEPTED
    code, calls = _run(tmp_path)
    assert code == 0 and calls == []


def test_main_closes_outbox_on_success(tmp_path, monkeypatch):
    real = ob.AlertOutbox(tmp_path / "premarket_alert_outbox.sqlite")
    closed = []
    original_close = real.close

    def close():
        closed.append(True)
        original_close()

    real.close = close
    monkeypatch.setattr(alert, "AlertOutbox", lambda _path: real)
    code, _ = _run(tmp_path)
    assert code == 0
    assert closed == [True]


def test_main_closes_outbox_when_reservation_raises(tmp_path, monkeypatch):
    real = ob.AlertOutbox(tmp_path / "premarket_alert_outbox.sqlite")
    closed = []
    original_close = real.close
    real.reserve = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))

    def close():
        closed.append(True)
        original_close()

    real.close = close
    monkeypatch.setattr(alert, "AlertOutbox", lambda _path: real)
    with pytest.raises(RuntimeError, match="boom"):
        _run(tmp_path)
    assert closed == [True]


def test_a_state_write_failure_after_the_send_does_not_resend(tmp_path):
    with patch.object(alert, "_save_state", side_effect=OSError("disk full")):
        code, calls = _run(tmp_path)
    assert code == 1 and len(calls) == 1               # reported, but it went out
    assert _outbox_row(tmp_path)["status"] == ob.ACCEPTED
    code, calls = _run(tmp_path)                       # state file never written
    assert code == 0 and calls == []


def test_an_unreadable_state_file_is_quarantined_and_nothing_is_sent(tmp_path):
    (tmp_path / "state.json").write_text("{not json")
    code, calls = _run(tmp_path)
    assert code == 1 and calls == []
    assert not (tmp_path / "state.json").exists()
    assert list(tmp_path.glob("state.json.corrupt-*"))


def test_an_unreadable_outbox_is_quarantined_and_nothing_is_sent(tmp_path):
    (tmp_path / "premarket_alert_outbox.sqlite").write_bytes(b"this is not a database" * 100)
    code, calls = _run(tmp_path)
    assert code == 1 and calls == []
    assert list(tmp_path.glob("premarket_alert_outbox.sqlite.corrupt-*"))


def test_after_a_quarantine_the_other_guard_still_stops_a_resend(tmp_path):
    assert _run(tmp_path)[0] == 0
    (tmp_path / "state.json").write_text("garbage")
    assert _run(tmp_path) == (1, [])                   # quarantined, not sent
    assert _run(tmp_path) == (0, [])                   # the outbox still knows it went out


def test_a_timed_out_hand_off_is_unknown_and_never_retried(tmp_path):
    def timeout(_message):
        raise subprocess.TimeoutExpired(cmd="osascript", timeout=30)

    code, calls = _run(tmp_path, send=timeout)
    assert code == 1 and calls == [0]
    assert _statuses(tmp_path) == ["prepared", "delivery_unknown"]
    assert _outbox_row(tmp_path)["status"] == ob.DELIVERY_UNKNOWN
    code, calls = _run(tmp_path)                       # later runs flag it, never resend
    assert code == 1 and calls == []


def test_a_send_that_never_started_went_nowhere_and_may_be_retried(tmp_path):
    # PR M: only an error proving osascript never ran is a retryable failure
    # (a non-zero exit after it ran is delivery_unknown, see test_alert_delivery).
    def refused(_message):
        raise FileNotFoundError(2, "No such file or directory", "osascript")

    code, calls = _run(tmp_path, send=refused)
    assert code == 1 and len(calls) == 1
    assert _statuses(tmp_path) == ["prepared", "failed"]
    code, calls = _run(tmp_path)
    assert code == 0 and len(calls) == 1
    row = _outbox_row(tmp_path)
    assert (row["status"], row["attempts"]) == (ob.ACCEPTED, 2)


def test_an_unwritable_audit_releases_the_reservation(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    with patch.object(alert, "build_ranked_screener", return_value=_table()), \
            patch.object(alert.IMessageConfig, "from_env", return_value=object()), \
            patch.object(alert, "send_imessage") as send:
        code = alert.main(["--state-path", str(tmp_path / "state.json"),
                           "--audit-path", str(blocker / "audit.jsonl")])
    assert code == 1 and send.call_count == 0
    assert _outbox_row(tmp_path)["status"] == ob.FAILED


def test_force_sends_again_under_its_own_key(tmp_path):
    assert _run(tmp_path)[0] == 0
    code, calls = _run(tmp_path, extra=["--force"])
    assert code == 0 and len(calls) == 1


# ── the outbox itself ───────────────────────────────────────────────────────


def test_a_run_that_died_before_submitting_is_reclaimed_after_the_lease(tmp_path):
    clock = [datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)]
    outbox = ob.AlertOutbox(tmp_path / "o.sqlite", clock=lambda: clock[0])
    assert outbox.reserve("k", event_id="a").acquired
    assert outbox.reserve("k", event_id="b") == ob.Reservation(False, ob.PREPARED, 1)   # in flight
    clock[0] += timedelta(seconds=ob.RESERVATION_LEASE_SECONDS + 1)
    assert outbox.reserve("k", event_id="b") == ob.Reservation(True, ob.PREPARED, 2)
    with pytest.raises(ob.InvalidTransition):
        outbox.transition("k", event_id="a", to=ob.PROVIDER_SUBMITTED)   # the dead run lost it
    outbox.close()


def test_a_run_that_died_mid_send_is_flagged_not_resent(tmp_path):
    clock = [datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)]
    outbox = ob.AlertOutbox(tmp_path / "o.sqlite", clock=lambda: clock[0])
    outbox.reserve("k", event_id="a")
    outbox.transition("k", event_id="a", to=ob.PROVIDER_SUBMITTED)
    clock[0] += timedelta(hours=2)
    assert outbox.reserve("k", event_id="b") == ob.Reservation(False, ob.PROVIDER_SUBMITTED, 1)
    assert outbox.is_stale("k")
    outbox.close()


@pytest.mark.parametrize("path", [
    (ob.PREPARED, ob.ACCEPTED),                       # never accepted without a hand-off
    (ob.PROVIDER_SUBMITTED, ob.ACCEPTED, ob.FAILED),  # terminal states stay terminal
    (ob.FAILED, ob.PROVIDER_SUBMITTED),
])
def test_transitions_only_move_forward(tmp_path, path):
    outbox = ob.AlertOutbox(tmp_path / "o.sqlite")
    outbox.reserve("k", event_id="a")
    *ok, bad = path
    for step in ok:
        if step != ob.PREPARED:
            outbox.transition("k", event_id="a", to=step)
    with pytest.raises(ob.InvalidTransition):
        outbox.transition("k", event_id="a", to=bad)
    outbox.close()


# ── two runs at once ────────────────────────────────────────────────────────


def _concurrent_run(args):
    tmp, sends = args

    def slow_send(message, *, config, retries=1, **_):
        with open(sends, "a") as handle:
            handle.write("sent\n")
        time.sleep(0.3)
        return {"to": "***1"}

    with patch.object(alert, "build_ranked_screener", return_value=_table()), \
            patch.object(alert.IMessageConfig, "from_env", return_value=object()), \
            patch.object(alert, "send_imessage", side_effect=slow_send):
        return alert.main(["--state-path", str(Path(tmp) / "state.json"),
                           "--audit-path", str(Path(tmp) / "audit.jsonl")])


def test_concurrent_runs_send_once(tmp_path):
    sends = tmp_path / "sends.txt"
    with multiprocessing.get_context("fork").Pool(6) as pool:
        codes = pool.map(_concurrent_run, [(str(tmp_path), str(sends))] * 6)
    assert sends.read_text().count("sent") == 1
    assert codes == [0] * 6
    assert json.loads((tmp_path / "state.json").read_text())["symbols"] == ["TSLA"]
    assert not list(tmp_path.glob("state.json.*.tmp"))


def test_a_damaged_but_openable_outbox_is_refused(tmp_path):
    import sqlite3

    path = tmp_path / "o.sqlite"
    outbox = ob.AlertOutbox(path)
    outbox._conn.executemany(
        "INSERT INTO alert_outbox (alert_key, event_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        [(f"key-{i}" * 20, "e", ob.ACCEPTED, "t", "t") for i in range(3000)],
    )
    outbox._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    outbox.close()
    data = bytearray(path.read_bytes())
    data[4096 * 5: 4096 * 5 + 600] = b"A" * 600                 # one page overwritten
    path.write_bytes(bytes(data))
    with sqlite3.connect(path) as probe:                        # it still opens...
        assert probe.execute("PRAGMA quick_check").fetchone()[0] != "ok"
    with pytest.raises(ob.OutboxCorrupt):                        # ...but is not trusted
        ob.AlertOutbox(path)


def test_a_send_left_mid_hand_off_is_flagged_by_later_runs(tmp_path):
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    outbox = ob.AlertOutbox(tmp_path / "premarket_alert_outbox.sqlite", clock=lambda: old)
    digest = alert._digest(datetime.now(timezone.utc).astimezone().date(), [alert_row("TSLA", 0.90)])
    outbox.reserve(digest, event_id="dead-run")
    outbox.transition(digest, event_id="dead-run", to=ob.PROVIDER_SUBMITTED)   # then the run died
    outbox.close()
    code, calls = _run(tmp_path)
    assert code == 1 and calls == []


def test_concurrent_state_writes_each_use_their_own_temp_file(tmp_path):
    import threading

    path = tmp_path / "state.json"
    errors = []

    def writer(n):
        for i in range(40):
            try:
                alert._save_state(path, {"writer": n, "i": i})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert set(json.loads(path.read_text())) == {"writer", "i"}
    assert not list(tmp_path.glob("*.tmp"))



def test_a_locked_outbox_is_waited_for_not_quarantined(tmp_path, monkeypatch):
    import sqlite3

    real_open = ob.AlertOutbox._open
    failures = iter([sqlite3.OperationalError("database is locked")] * 3)

    def flaky_open(self):
        error = next(failures, None)
        if error is not None:
            raise error
        return real_open(self)

    monkeypatch.setattr(ob.AlertOutbox, "_open", flaky_open)
    monkeypatch.setattr(ob.time, "sleep", lambda _s: None)
    code, calls = _run(tmp_path)
    assert code == 0 and len(calls) == 1
    assert not list(tmp_path.glob("*.corrupt-*"))


@pytest.mark.parametrize("error", ["disk I/O error", "database is locked"])
def test_an_unavailable_outbox_sends_nothing_and_is_left_in_place(tmp_path, monkeypatch, error):
    import sqlite3

    def failing_open(self):
        raise sqlite3.OperationalError(error)

    monkeypatch.setattr(ob.AlertOutbox, "_open", failing_open)
    monkeypatch.setattr(ob.time, "sleep", lambda _s: None)
    (tmp_path / "premarket_alert_outbox.sqlite").write_bytes(b"")
    code, calls = _run(tmp_path)
    assert code == 1 and calls == []
    assert (tmp_path / "premarket_alert_outbox.sqlite").exists()
    assert not list(tmp_path.glob("*.corrupt-*"))
