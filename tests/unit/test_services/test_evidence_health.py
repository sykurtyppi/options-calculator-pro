from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from services.evidence_health import EvidenceHealthConfig, build_evidence_health_status


def _init_db(path: Path, ddl: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(ddl)
        conn.commit()


def _write_daily_report(path: Path, *, as_of: str = "2026-04-27", generated_at: str = "2026-04-27T22:00:00+00:00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of_date": as_of,
        "generated_at": generated_at,
        "forward_loop": {"entries": {"ledger_records": 1}, "exits": {"exits": 0}},
        "evidence_report": {"maturity": {"maturity_label": "Insufficient evidence"}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    (path.parent / "latest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_weekly_report(path: Path, *, generated_at: str = "2026-04-27T22:30:00+00:00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": generated_at, "report_type": "weekly_evidence_report"}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_log(path: Path, *, status: str = "success", end_time: str = "2026-04-27T22:01:00+00:00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event_type": "evidence_cycle_run",
        "status": status,
        "start_time": "2026-04-27T21:30:00+00:00",
        "end_time": end_time,
        "error": "boom" if status == "fail" else None,
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")


def _write_completion_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("===== 2026-04-27T22:01:00Z daily evidence cycle complete =====\n", encoding="utf-8")


def _write_telemetry(path: Path, *, ts: str = "2026-04-27T22:00:00+00:00") -> None:
    _init_db(
        path,
        """
        CREATE TABLE IF NOT EXISTS provider_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_timestamp TEXT NOT NULL
        )
        """,
    )
    with sqlite3.connect(str(path)) as conn:
        conn.execute("DELETE FROM provider_telemetry")
        conn.execute("INSERT INTO provider_telemetry (request_timestamp) VALUES (?)", (ts,))
        conn.commit()


def _sqlite_store(path: Path) -> None:
    _init_db(path, "CREATE TABLE IF NOT EXISTS sample (id INTEGER PRIMARY KEY)")


def _config(tmp_path: Path) -> EvidenceHealthConfig:
    return EvidenceHealthConfig(
        expected_date=date(2026, 4, 27),
        report_dir=tmp_path / "reports" / "daily",
        weekly_report_dir=tmp_path / "reports" / "weekly",
        structured_run_log=tmp_path / "logs" / "evidence_cycle_runs.jsonl",
        launchd_log_path=tmp_path / "logs" / "daily_evidence_cycle_launchd.log",
        ledger_path=tmp_path / "db" / "ledger.sqlite",
        outcome_store_path=tmp_path / "db" / "outcomes.sqlite",
        baseline_store_path=tmp_path / "db" / "baselines.sqlite",
        telemetry_store_path=tmp_path / "db" / "telemetry.sqlite",
    )


def _seed_ok_files(cfg: EvidenceHealthConfig) -> None:
    _write_daily_report(cfg.report_dir / "evidence_report_2026-04-27.json")
    _write_weekly_report(cfg.weekly_report_dir / "weekly_latest.json")
    _write_run_log(cfg.structured_run_log)
    _write_completion_log(cfg.launchd_log_path)
    _write_telemetry(cfg.telemetry_store_path)
    _sqlite_store(cfg.ledger_path)
    _sqlite_store(cfg.outcome_store_path)
    _sqlite_store(cfg.baseline_store_path)


def test_evidence_health_ok_when_artifacts_are_fresh_and_readable(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "OK"
    assert status["ok"] is True
    assert status["issues"] == []


def test_evidence_health_fails_for_missing_daily_report(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    (cfg.report_dir / "evidence_report_2026-04-27.json").unlink()

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "FAIL"
    assert any(issue["check"] == "daily_evidence_report" for issue in status["issues"])
    assert all("fix" in issue for issue in status["issues"])


def test_evidence_health_warns_not_fail_when_daily_report_not_due_yet(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    (cfg.report_dir / "evidence_report_2026-04-27.json").unlink()

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "WARN"
    assert status["daily_report"]["not_due_yet"] is True
    assert any("not due yet" in issue["message"] for issue in status["issues"])
    assert not any(issue["severity"] == "FAIL" for issue in status["issues"])


def test_evidence_health_warns_for_stale_weekly_report_and_malformed_log(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_weekly_report(
        cfg.weekly_report_dir / "weekly_latest.json",
        generated_at="2026-04-10T22:30:00+00:00",
    )
    cfg.structured_run_log.write_text("not-json\n" + cfg.structured_run_log.read_text(encoding="utf-8"), encoding="utf-8")

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "WARN"
    assert any("Weekly evidence report is stale" in issue["message"] for issue in status["issues"])
    assert any("malformed JSONL" in issue["message"] for issue in status["issues"])


def test_evidence_health_detects_recent_structured_failure(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_run_log(cfg.structured_run_log, status="fail")

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "FAIL"
    assert any(issue["check"] == "structured_run_log" and "Recent evidence-cycle failure" in issue["message"] for issue in status["issues"])


def test_evidence_health_warns_for_stale_provider_telemetry(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_telemetry(cfg.telemetry_store_path, ts=(datetime(2026, 4, 20, tzinfo=timezone.utc)).isoformat())

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "WARN"
    assert any(issue["check"] == "provider_telemetry" and "stale" in issue["message"] for issue in status["issues"])


def test_evidence_health_fails_on_sqlite_corruption_like_unreadable_store(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    cfg.outcome_store_path.write_text("not a sqlite database", encoding="utf-8")

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "FAIL"
    assert any(issue["check"] == "outcome_store" for issue in status["issues"])
