"""A new install must not page for a cycle it never had a chance to run."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pytest

from scripts import watch_daily_evidence_cycle as watchdog


@pytest.mark.parametrize(
    "instant,prior_log,prior_report,explicit_date,exit_code,expected_date,expected_status",
    [
        ("2026-11-10T14:35:00+00:00", False, False, False, 0, "2026-11-10", "not_due_yet"),
        ("2026-11-07T22:15:00+00:00", False, False, False, 0, "2026-11-07", "market_closed"),
        ("2026-11-26T22:15:00+00:00", False, False, False, 0, "2026-11-26", "market_closed"),
        ("2026-11-10T14:35:00+00:00", True, False, False, 2, "2026-11-09", None),
        ("2026-11-10T14:35:00+00:00", False, True, False, 2, "2026-11-09", None),
        ("2026-11-10T22:16:00+00:00", False, False, False, 2, "2026-11-10", None),
        ("2026-11-10T14:35:00+00:00", False, False, True, 2, "2026-11-09", None),
    ],
)
def test_first_install_deferred_only_before_first_due_cycle(
    monkeypatch, tmp_path, capsys, instant, prior_log, prior_report, explicit_date,
    exit_code, expected_date, expected_status,
):
    fixed = datetime.fromisoformat(instant)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz is None else fixed.astimezone(tz)

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    log_path = tmp_path / "cycle.log"
    if prior_log:
        log_path.write_text(
            "===== 2026-11-09T16:45:05Z daily evidence cycle failed exit_code=3 =====\n",
            encoding="utf-8",
        )
    if prior_report:
        (report_dir / "evidence_report_2026-11-06.json").write_text("{}", encoding="utf-8")
    args = ["watchdog", "--report-dir", str(report_dir), "--log-path", str(log_path),
            "--state-path", str(tmp_path / "alert-state.json"), "--no-fetch", "--dry-run-alert"]
    if explicit_date:
        args.extend(["--date", "2026-11-09"])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(watchdog, "datetime", Clock)
    monkeypatch.setattr(watchdog, "build_candidate_exit_resolver_health", lambda **kw: {"issues": [], "summary": {}})
    monkeypatch.setattr(watchdog, "build_launchd_job_freshness", lambda **kw: {"issues": [], "summary": {}})
    monkeypatch.setattr(watchdog, "build_evidence_funnel_health", lambda: {"issues": [], "summary": {}})
    monkeypatch.setattr(watchdog, "build_provider_budget_health", lambda: {"issues": [], "summary": {}})
    monkeypatch.setattr(watchdog, "deployment_drift", lambda **kw: {})
    monkeypatch.setattr(watchdog, "build_code_identity_health", lambda *a, **kw: {"issues": [], "summary": {}})
    sent = []
    monkeypatch.setattr(watchdog, "maybe_send_watchdog_alert", lambda status, **kw: (sent.append(status), {"attempted": False, "reason": "test"})[1])

    assert watchdog.main() == exit_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["watchdog"]["expected_date"] == expected_date
    if expected_status:
        assert payload["watchdog"]["status"] == expected_status
        assert payload["watchdog"]["ok"] is True
        assert sent == []
    else:
        assert payload["watchdog"]["ok"] is False
        assert any("Missing evidence report" in e for e in payload["watchdog"]["errors"])
        assert len(sent) == 1
