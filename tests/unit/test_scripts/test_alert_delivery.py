"""PR M: an alert that may have gone out is never sent again automatically.

Regression tests for the audit of main at 469c6ff:
- a hand-off that reached Messages and then failed decoding osascript's output
  (UnicodeDecodeError) was recorded as ``failed``, so the next run sent it
  again; the log said "nothing was sent";
- after a timeout (``delivery_unknown``) the outbox was damaged and
  quarantined, and the next run created a fresh outbox and resubmitted the
  same digest (exit codes [1, 1, 0], provider calls: timeout, then success).
"""
from __future__ import annotations

import json
import logging
import subprocess
from unittest.mock import patch

import pytest

import scripts.premarket_screener_alert as alert
from services import alert_outbox as ob
from services.external_io_gate import ExternalIOBlocked
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


OUTBOX = "premarket_alert_outbox.sqlite"


# ── errors after osascript started are unknown, not failed ──────────────────


def _handed_off_then(error):
    def send(_message):
        raise error
    return send


@pytest.mark.parametrize("error", [
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),    # the audit's repro
    subprocess.CalledProcessError(1, "osascript", stderr="execution error"),
    subprocess.TimeoutExpired(cmd="osascript", timeout=30),
    RuntimeError("anything else"),
    PermissionError(13, "Permission denied", "/var/folders/x/output.txt"),  # not osascript itself
])
def test_an_error_after_launch_is_delivery_unknown_and_never_resent(tmp_path, caplog, error):
    caplog.set_level(logging.INFO)
    code, calls = _run(tmp_path, send=_handed_off_then(error))
    assert (code, calls) == (1, [0])
    assert _statuses(tmp_path) == ["prepared", "delivery_unknown"]
    assert "nothing was sent" not in caplog.text
    code, calls = _run(tmp_path)                       # a later run flags it, never resends
    assert (code, calls) == (1, [])
    assert _statuses(tmp_path) == ["prepared", "delivery_unknown"]


@pytest.mark.parametrize("error", [
    ExternalIOBlocked("imessage external IO blocked"),
    FileNotFoundError(2, "No such file or directory", "osascript"),
    PermissionError(13, "Permission denied", "/usr/bin/osascript"),
    OSError(8, "Exec format error", "osascript"),
])
def test_an_error_proving_osascript_never_ran_is_a_retryable_failure(tmp_path, caplog, error):
    caplog.set_level(logging.INFO)
    code, calls = _run(tmp_path, send=_handed_off_then(error))
    assert (code, calls) == (1, [0])
    assert _statuses(tmp_path) == ["prepared", "failed"]
    assert "nothing was sent" in caplog.text
    code, calls = _run(tmp_path)                       # nothing went out: the next run sends
    assert (code, calls) == (0, [0])


# ── a quarantined outbox leaves a marker that blocks sending ────────────────


def _damage(tmp_path):
    (tmp_path / OUTBOX).write_bytes(b"this is not a database" * 100)
    for side in ("-wal", "-shm"):
        (tmp_path / (OUTBOX + side)).unlink(missing_ok=True)


def test_a_quarantined_outbox_blocks_the_resend_it_used_to_allow(tmp_path):
    # The audit's repro: timeout -> damaged outbox -> quarantine -> fresh outbox resent.
    codes, provider = [], []
    code, calls = _run(tmp_path, send=_handed_off_then(subprocess.TimeoutExpired("osascript", 30)))
    codes.append(code)
    provider += calls
    _damage(tmp_path)
    for _ in range(3):
        code, calls = _run(tmp_path)
        codes.append(code)
        provider += calls
    assert codes == [1, 1, 1, 1]
    assert provider == [0]                             # only the original, timed-out hand-off
    marker = ob.read_tombstone(tmp_path / OUTBOX)
    assert marker and marker["moved_to"].startswith(str(tmp_path / OUTBOX) + ".corrupt-")
    assert list(tmp_path.glob(OUTBOX + ".corrupt-*"))
    assert not (tmp_path / OUTBOX).exists()           # no fresh outbox was created


def test_an_operator_reconciles_and_sending_resumes(tmp_path):
    _damage(tmp_path)
    assert _run(tmp_path) == (1, [])
    assert _run(tmp_path) == (1, [])
    code, calls = _run(tmp_path, extra=("--outbox-reconciled",))
    assert (code, calls) == (0, [0])
    assert ob.read_tombstone(tmp_path / OUTBOX) is None
    assert _run(tmp_path) == (0, [])                  # and the new outbox guards it from here


def test_a_marker_that_cannot_be_written_leaves_the_damaged_outbox_in_place(tmp_path):
    _damage(tmp_path)
    with patch.object(ob.os, "fsync", side_effect=OSError("disk full")):
        assert _run(tmp_path) == (1, [])
    assert (tmp_path / OUTBOX).exists() and not list(tmp_path.glob(OUTBOX + ".corrupt-*"))
    assert ob.read_tombstone(tmp_path / OUTBOX) is None
    assert _run(tmp_path) == (1, [])                  # still damaged: still nothing sent
    assert ob.read_tombstone(tmp_path / OUTBOX) is not None


@pytest.mark.parametrize("content", ["[]", "null", '"quarantined"'])
def test_a_marker_that_is_not_an_object_still_blocks(tmp_path, content):
    ob.tombstone_path(tmp_path / OUTBOX).write_text(content)
    assert "unreadable" in ob.read_tombstone(tmp_path / OUTBOX)
    assert _run(tmp_path) == (1, [])


def test_an_unreadable_marker_still_blocks(tmp_path):
    ob.tombstone_path(tmp_path / OUTBOX).write_text("{garbage")
    assert "unreadable" in ob.read_tombstone(tmp_path / OUTBOX)
    assert _run(tmp_path) == (1, [])


def test_a_dry_run_is_unaffected_by_the_marker(tmp_path, capsys):
    _damage(tmp_path)
    _run(tmp_path)
    with patch.object(alert, "build_ranked_screener", return_value=_table()):
        assert alert.main(["--state-path", str(tmp_path / "state.json"), "--dry-run"]) == 0
    assert "TSLA" in capsys.readouterr().out


def test_clearing_without_a_marker_is_a_no_op(tmp_path):
    assert ob.clear_tombstone(tmp_path / OUTBOX) is False
    assert _run(tmp_path, extra=("--outbox-reconciled",)) == (0, [0])
