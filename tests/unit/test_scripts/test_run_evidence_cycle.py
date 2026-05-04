from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import scripts.run_evidence_cycle as evidence_cycle


def test_run_evidence_cycle_writes_weekly_export_and_optional_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        evidence_cycle,
        "run_daily_cycle",
        lambda **_kwargs: {"entries": {"entries": 1}, "exits": {"exits": 0}},
    )
    monkeypatch.setattr(
        evidence_cycle,
        "build_evidence_report",
        lambda: {
            "maturity": {"maturity_label": "Insufficient evidence"},
            "commercialization_gate": {"active_evidence_days": 3, "resolved_selector_outcomes": 0},
            "warning_flags": ["No resolved paper outcomes are available yet."],
        },
    )
    monkeypatch.setattr(
        evidence_cycle,
        "build_weekly_evidence_report",
        lambda: {
            "report_type": "weekly_evidence_report",
            "maturity": {"maturity_label": "Insufficient evidence"},
            "notes": ["System/evidence summary only, not trading advice."],
        },
    )
    monkeypatch.setattr(
        evidence_cycle,
        "build_provider_telemetry_diagnostics",
        lambda limit=25: {"totals": {"failures": 0, "fallback_count": 0, "stale_count": 0}},
    )
    monkeypatch.setattr(
        evidence_cycle,
        "_send_local_summary",
        lambda report, weekly_report=None, dry_run=False: {
            "attempted": True,
            "sent": False,
            "dry_run": dry_run,
            "maturity": report["maturity"]["maturity_label"],
            "weekly": weekly_report is not None,
        },
    )

    payload = evidence_cycle.run_evidence_cycle(
        as_of=date(2026, 4, 27),
        dry_run=False,
        report_dir=tmp_path / "daily",
        weekly_report_dir=tmp_path / "weekly",
        write_weekly=True,
        notify_summary=True,
    )

    assert payload["weekly_report_due"] is True
    assert payload["notification_summary"]["weekly"] is True
    daily_path = Path(payload["report_path"])
    weekly_path = Path(payload["weekly_report_path"])
    assert daily_path.exists()
    assert weekly_path.exists()
    weekly_payload = json.loads(weekly_path.read_text(encoding="utf-8"))
    assert weekly_payload["report_type"] == "weekly_evidence_report"
    assert (tmp_path / "weekly" / "weekly_latest.json").exists()


def test_run_evidence_cycle_dry_run_does_not_write_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evidence_cycle, "run_daily_cycle", lambda **_kwargs: {})
    monkeypatch.setattr(evidence_cycle, "build_evidence_report", lambda: {"warning_flags": []})
    monkeypatch.setattr(evidence_cycle, "build_weekly_evidence_report", lambda: {"report_type": "weekly_evidence_report"})
    monkeypatch.setattr(evidence_cycle, "build_provider_telemetry_diagnostics", lambda limit=25: {"totals": {}})

    payload = evidence_cycle.run_evidence_cycle(
        as_of=date(2026, 4, 27),
        dry_run=True,
        report_dir=tmp_path / "daily",
        weekly_report_dir=tmp_path / "weekly",
        write_weekly=True,
    )

    assert payload["weekly_report_due"] is True
    assert "report_path" not in payload
    assert not (tmp_path / "daily").exists()
    assert not (tmp_path / "weekly").exists()
