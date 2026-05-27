from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from services import automation_watchdog
from services.automation_watchdog import (
    EvidenceWatchdogConfig,
    IMessageConfig,
    build_evidence_watchdog_status,
    maybe_send_watchdog_alert,
)


def _write_report(path, *, as_of_date="2026-04-27", generated_at=None):
    payload = {
        "as_of_date": as_of_date,
        "generated_at": generated_at or datetime(2026, 4, 27, 22, 0, tzinfo=timezone.utc).isoformat(),
        "forward_loop": {"entries": {"entries": 1}},
        "evidence_report": {"evidence_label": "paper_research_not_execution_grade"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_watchdog_passes_with_current_report_and_completion_log(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    log_path = tmp_path / "watchdog.log"
    report_path = report_dir / "evidence_report_2026-04-27.json"
    _write_report(report_path)
    (report_dir / "latest.json").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    log_path.write_text("===== 2026-04-27T22:01:00Z daily evidence cycle complete =====\n", encoding="utf-8")

    status = build_evidence_watchdog_status(
        config=EvidenceWatchdogConfig(
            expected_date=date(2026, 4, 27),
            report_dir=report_dir,
            log_path=log_path,
        ),
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["ok"] is True
    assert status["completion_logged"] is True
    assert status["errors"] == []


def test_watchdog_fails_when_report_missing(tmp_path):
    status = build_evidence_watchdog_status(
        config=EvidenceWatchdogConfig(
            expected_date=date(2026, 4, 27),
            report_dir=tmp_path / "missing",
            log_path=tmp_path / "missing.log",
        ),
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["ok"] is False
    assert any("Missing evidence report" in err for err in status["errors"])
    assert any("completion marker" in err for err in status["errors"])


def test_watchdog_flags_stale_or_malformed_report(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    log_path = tmp_path / "watchdog.log"
    _write_report(
        report_dir / "evidence_report_2026-04-27.json",
        generated_at=(datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)).isoformat(),
    )
    log_path.write_text("===== 2026-04-27T22:01:00Z daily evidence cycle complete =====\n", encoding="utf-8")

    status = build_evidence_watchdog_status(
        config=EvidenceWatchdogConfig(
            expected_date=date(2026, 4, 27),
            report_dir=report_dir,
            log_path=log_path,
            max_report_age_hours=24,
        ),
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["ok"] is False
    assert any("stale" in err for err in status["errors"])


def test_watchdog_imessage_dedupes_failure_signature(tmp_path, monkeypatch):
    sent = []

    def fake_send(message, *, config, timeout_seconds=10.0):
        sent.append((message, config.to_address, timeout_seconds))
        return {"sent": True, "provider": "imessage", "to": "***1212"}

    monkeypatch.setattr("services.automation_watchdog.send_imessage", fake_send)
    status = {
        "ok": False,
        "expected_date": "2026-04-27",
        "errors": ["Missing evidence report for 2026-04-27."],
        "report_path": "/tmp/evidence_report_2026-04-27.json",
    }
    config = IMessageConfig("+15551211212")
    state_path = tmp_path / "state.json"

    first = maybe_send_watchdog_alert(status, imessage_config=config, state_path=state_path)
    second = maybe_send_watchdog_alert(status, imessage_config=config, state_path=state_path)

    assert first["attempted"] is True
    assert first["sent"] is True
    assert first["state_persisted"] is True
    assert second == {"attempted": False, "reason": "duplicate_alert_suppressed"}
    assert len(sent) == 1


def test_watchdog_imessage_reports_missing_configuration(tmp_path):
    status = {
        "ok": False,
        "expected_date": "2026-04-27",
        "errors": ["Missing evidence report for 2026-04-27."],
        "report_path": "/tmp/evidence_report_2026-04-27.json",
    }

    result = maybe_send_watchdog_alert(status, imessage_config=None, state_path=tmp_path / "state.json")

    assert result == {"attempted": False, "reason": "imessage_not_configured"}


def test_watchdog_alert_does_not_crash_when_state_write_fails(tmp_path, monkeypatch):
    def fake_send(message, *, config, timeout_seconds=10.0):
        return {"sent": True, "provider": "imessage", "to": "***1212"}

    def fail_write(path, payload):
        raise PermissionError("blocked")

    monkeypatch.setattr("services.automation_watchdog.send_imessage", fake_send)
    monkeypatch.setattr("services.automation_watchdog._write_alert_state", fail_write)
    status = {
        "ok": False,
        "expected_date": "2026-04-27",
        "errors": ["Missing evidence report for 2026-04-27."],
        "report_path": "/tmp/evidence_report_2026-04-27.json",
    }

    result = maybe_send_watchdog_alert(
        status,
        imessage_config=IMessageConfig("+15551211212"),
        state_path=tmp_path / "state.json",
    )

    assert result["attempted"] is True
    assert result["sent"] is True
    assert result["state_persisted"] is False
    assert "PermissionError" in result["state_error"]


# ── Hardening P2-6 / P2-7 regression tests ───────────────────────────────


def test_write_alert_state_replaces_atomically_and_leaves_no_tmp(tmp_path) -> None:
    """Hardening P2-6: _write_alert_state must use tmp + os.replace so
    a crash mid-write can't corrupt the dedup state file."""
    state_path = tmp_path / "state" / "alerts.json"
    automation_watchdog._write_alert_state(state_path, {"signature": "abc", "n": 1})

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload == {"signature": "abc", "n": 1}
    # No leftover .alerts.json.<pid>.tmp.
    assert list(state_path.parent.glob(".alerts.json.*.tmp")) == []


def test_write_alert_state_preserves_existing_on_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the encode/write step fails, the previous state file must
    survive — losing a dedup record causes either pager noise (re-alert)
    or a missed alert."""
    state_path = tmp_path / "alerts.json"
    state_path.write_text(json.dumps({"signature": "prior", "n": 0}), encoding="utf-8")

    real_dump = automation_watchdog.json.dump

    def boom(*args, **kwargs):  # noqa: ANN001
        raise OSError("simulated full disk")

    monkeypatch.setattr(automation_watchdog.json, "dump", boom)
    with pytest.raises(OSError, match="simulated full disk"):
        automation_watchdog._write_alert_state(state_path, {"signature": "new", "n": 1})
    monkeypatch.setattr(automation_watchdog.json, "dump", real_dump)

    assert json.loads(state_path.read_text(encoding="utf-8")) == {"signature": "prior", "n": 0}
    assert list(state_path.parent.glob(".alerts.json.*.tmp")) == []


def test_read_json_surfaces_os_error_on_stderr(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hardening P2-7: a perms regression must be visible on stderr so
    the operator can see the failure mode (chmod, not JSON corruption)."""
    target = tmp_path / "perms.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o000)
    try:
        result = automation_watchdog._read_json(target)
    finally:
        target.chmod(0o600)

    assert result is None
    captured = capsys.readouterr()
    assert "_read_json" in captured.err
    assert "cannot read" in captured.err


def test_read_json_silent_on_malformed_json(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Malformed JSON stays silent — the caller already produces its own
    'not valid JSON' warning."""
    target = tmp_path / "bad.json"
    target.write_text("{not json", encoding="utf-8")

    assert automation_watchdog._read_json(target) is None
    assert capsys.readouterr().err == ""
