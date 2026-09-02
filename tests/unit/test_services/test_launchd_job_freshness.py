"""Codex audit F5: the state-backup and screener-alert launchd jobs must be
monitored for freshness and exit status, and "no qualifying setups" must stay
a healthy outcome."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from services.evidence_health import EvidenceHealthConfig, build_launchd_job_freshness

NOW = datetime(2026, 9, 3, 22, 15, tzinfo=timezone.utc)


def _cfg(tmp_path: Path) -> EvidenceHealthConfig:
    return EvidenceHealthConfig(
        expected_date=date(2026, 9, 3),
        state_backup_launchd_log_path=tmp_path / "state_backup_launchd.log",
        screener_alert_launchd_log_path=tmp_path / "premarket_screener_alert_launchd.log",
    )


def _marker(ts: str, text: str) -> str:
    return f"===== {ts} {text} =====\n"


def _issues_for(status, job):
    return [i for i in status["issues"] if i["check"] == job]


def test_fresh_completions_are_healthy(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_backup_launchd_log_path.write_text(
        _marker("2026-09-03T04:00:01Z", "state backup start") + _marker("2026-09-03T04:00:09Z", "state backup complete"))
    cfg.screener_alert_launchd_log_path.write_text(
        _marker("2026-09-03T15:00:01Z", "premarket alert start")
        + "2026-09-03 15:00:14,000 | INFO | No qualifying setups; nothing to send.\n"
        + _marker("2026-09-03T15:00:14Z", "premarket alert complete"))
    status = build_launchd_job_freshness(config=cfg, now=NOW)
    assert status["ok"] and status["status"] == "OK", status
    assert status["summary"]["screener_alert"]["verdict"] == "OK"
    # "Nothing to send" is a completed run, not a failure — silence stays healthy.
    assert not _issues_for(status, "screener_alert")


def test_failed_latest_run_is_alertable_fail(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_backup_launchd_log_path.write_text(
        _marker("2026-09-02T04:00:09Z", "state backup complete")
        + _marker("2026-09-03T04:00:01Z", "state backup start")
        + _marker("2026-09-03T04:00:02Z", "state backup failed exit_code=2"))
    cfg.screener_alert_launchd_log_path.write_text(_marker("2026-09-03T15:00:14Z", "premarket alert complete"))
    status = build_launchd_job_freshness(config=cfg, now=NOW)
    assert status["status"] == "FAIL"
    [issue] = _issues_for(status, "state_backup")
    assert issue["severity"] == "FAIL" and issue["alertable"] is True
    assert "FAILED" in issue["message"]


def test_stale_completion_is_alertable_warn(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_backup_launchd_log_path.write_text(_marker("2026-08-30T04:00:09Z", "state backup complete"))  # 4.75 days
    cfg.screener_alert_launchd_log_path.write_text(_marker("2026-09-03T15:00:14Z", "premarket alert complete"))
    status = build_launchd_job_freshness(config=cfg, now=NOW)
    assert status["status"] == "WARN"
    [issue] = _issues_for(status, "state_backup")
    assert issue["alertable"] is True and "STALE" == status["summary"]["state_backup"]["verdict"]


def test_weekend_gap_does_not_false_alarm_the_weekday_alert(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.state_backup_launchd_log_path.write_text(_marker("2026-09-03T04:00:09Z", "state backup complete"))
    # Friday run, checked Monday night: 3 days, inside the 4-day window.
    cfg.screener_alert_launchd_log_path.write_text(_marker("2026-08-28T15:00:14Z", "premarket alert complete"))
    monday = datetime(2026, 8, 31, 22, 15, tzinfo=timezone.utc)
    status = build_launchd_job_freshness(config=cfg, now=monday)
    assert status["ok"], status


def test_never_ran_is_non_alertable(tmp_path):
    cfg = _cfg(tmp_path)  # neither log exists
    status = build_launchd_job_freshness(config=cfg, now=NOW)
    assert status["ok"], "a job that has never fired must not page the operator"
    assert all(i["alertable"] is False for i in status["issues"])
    assert status["summary"]["state_backup"]["verdict"] == "NEVER_RAN"


def test_watchdog_combiner_escalates_job_failures():
    from scripts.watch_daily_evidence_cycle import _build_combined_watchdog_status

    combined = _build_combined_watchdog_status(
        watchdog_status={"ok": True, "errors": [], "warnings": []},
        resolver_health={"issues": [], "summary": {}},
        job_freshness={
            "issues": [
                {"severity": "FAIL", "check": "state_backup", "message": "state_backup last run FAILED", "fix": "x", "alertable": True},
                {"severity": "WARN", "check": "screener_alert", "message": "never ran", "fix": "x", "alertable": False},
            ],
            "summary": {"state_backup": {"verdict": "FAILED"}},
        },
    )
    assert combined["ok"] is False
    assert "state_backup last run FAILED" in combined["errors"]
    assert "never ran" in combined["warnings"] and "never ran" not in combined["errors"]
    assert combined["launchd_jobs"]["state_backup"]["verdict"] == "FAILED"


def test_watchdog_combiner_is_backward_compatible_without_job_freshness():
    from scripts.watch_daily_evidence_cycle import _build_combined_watchdog_status

    combined = _build_combined_watchdog_status(
        watchdog_status={"ok": True, "errors": [], "warnings": []},
        resolver_health={"issues": [], "summary": {}},
    )
    assert combined["ok"] is True and combined["launchd_jobs"] == {}
