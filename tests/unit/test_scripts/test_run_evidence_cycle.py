from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

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


# ── Hardening P0-1 regression tests ──────────────────────────────────────


def test_atomic_write_json_replaces_destination_and_leaves_no_tmp(tmp_path: Path) -> None:
    """The atomic-write helper must replace the destination atomically
    and remove its tmp file even on the happy path."""
    dest = tmp_path / "report.json"
    evidence_cycle._atomic_write_json(dest, {"hello": "world"})

    assert dest.exists()
    assert json.loads(dest.read_text(encoding="utf-8")) == {"hello": "world"}
    # No leftover ``.report.json.<pid>.tmp`` siblings.
    tmps = list(tmp_path.glob(".report.json.*.tmp"))
    assert tmps == []


def test_atomic_write_json_preserves_existing_file_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the body write raises, the pre-existing destination must
    survive untouched (this is the whole point of the tmp+replace
    pattern — a launchd kill mid-write previously corrupted the
    daily report)."""
    dest = tmp_path / "report.json"
    original = {"as_of_date": "2026-04-27", "trusted": True}
    dest.write_text(json.dumps(original), encoding="utf-8")

    real_dump = evidence_cycle.json.dump

    def boom(*args, **kwargs):  # noqa: ANN001
        raise OSError("simulated disk full")

    monkeypatch.setattr(evidence_cycle.json, "dump", boom)
    with pytest.raises(OSError, match="simulated disk full"):
        evidence_cycle._atomic_write_json(dest, {"would_corrupt": True})
    monkeypatch.setattr(evidence_cycle.json, "dump", real_dump)

    # Destination untouched, tmp cleaned up.
    assert json.loads(dest.read_text(encoding="utf-8")) == original
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_append_structured_run_log_emits_stderr_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hardening P0-2: a disk-full / perms failure on the structured
    run log must surface via stderr (which launchd captures) so the
    operator can see the cycle's audit trail is broken.

    Updated for PR #73: the helper now routes through
    ``services.jsonl_helpers.append_jsonl_locked``, which coerces
    its path argument through ``Path(...)``. The old test used a
    custom ``FailingPath`` mock that doesn't survive that
    coercion; this version uses a real path whose parent is a
    pre-existing FILE (not a directory), so ``Path.parent.mkdir(...)``
    raises ``NotADirectoryError`` (an ``OSError`` subclass)
    inside the helper. That triggers the same code path the test
    is meant to verify: the WARNING goes to stderr instead of
    silently swallowing the failure.
    """
    # Pre-create a regular FILE where the parent directory would
    # need to live. The helper's
    # ``path.parent.mkdir(parents=True, exist_ok=True)`` then fails
    # because the path collides with a non-directory entry.
    obstructing_file = tmp_path / "logs"
    obstructing_file.write_text("not a directory", encoding="utf-8")
    target = obstructing_file / "evidence_cycle_runs.jsonl"

    evidence_cycle._append_structured_run_log(
        target,
        {"event_type": "evidence_cycle_run", "status": "success", "dry_run": False},
    )
    captured = capsys.readouterr()
    assert "WARNING" in captured.err, (
        f"expected WARNING on stderr; got: {captured.err!r}"
    )
    # Either NotADirectoryError or FileExistsError is acceptable —
    # both are OSError subclasses and both signal the same class of
    # disk/perms regression the operator needs to see.
    assert (
        "NotADirectoryError" in captured.err
        or "FileExistsError" in captured.err
    ), f"expected an OSError subclass name on stderr; got: {captured.err!r}"
    assert "run_evidence_cycle" in captured.err
