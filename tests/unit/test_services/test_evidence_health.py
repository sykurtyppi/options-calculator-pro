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


def _write_candidate_resolver_log(path: Path, *, count_balance: str = "true", marker: str = "complete") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "===== 2026-04-27T12:30:00Z candidate exit resolver start =====\n"
        f"  count_balance_holds:    {count_balance}\n"
        f"===== 2026-04-27T12:31:00Z candidate exit resolver {marker} =====\n",
        encoding="utf-8",
    )


def _write_candidate_resolver_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


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
        candidate_resolver_jsonl=tmp_path / "logs" / "candidate_exit_resolutions.jsonl",
        candidate_resolver_launchd_log_path=tmp_path / "logs" / "candidate_exit_resolver_launchd.log",
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
    _write_candidate_resolver_log(cfg.candidate_resolver_launchd_log_path)
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


def test_evidence_health_fails_when_candidate_resolver_count_balance_breaks(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_candidate_resolver_log(cfg.candidate_resolver_launchd_log_path, count_balance="false")

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "FAIL"
    assert any(
        issue["check"] == "candidate_exit_resolver" and "count_balance_holds" in issue["message"]
        for issue in status["issues"]
    )


def test_evidence_health_fails_on_candidate_resolver_simulator_error(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_candidate_resolver_jsonl(
        cfg.candidate_resolver_jsonl,
        [
            {
                "ts": "2026-04-27T12:31:00+00:00",
                "recommendation_id": "rec_sim_err_1",
                "resolver_status": "permanently_failed:simulator_error",
                "failure_reason": None,
                "days_in_awaiting_state": None,
            }
        ],
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "FAIL"
    assert status["candidate_exit_resolver"]["simulator_error_count"] == 1
    assert any("simulator_error" in issue["message"] for issue in status["issues"])


def test_evidence_health_warns_on_candidate_resolver_stuck_awaiting(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_candidate_resolver_jsonl(
        cfg.candidate_resolver_jsonl,
        [
            {
                "ts": "2026-04-27T12:31:00+00:00",
                "recommendation_id": "rec_stuck_1",
                "resolver_status": "awaiting_chain_data",
                "failure_reason": None,
                "days_in_awaiting_state": 11,
                # Real JSONL rows do NOT include mid_realized_return_pct
                # on awaiting events, but a defensive test includes it
                # to enforce the "no PnL in alerts" rule.
                "mid_realized_return_pct": -99.0,
            }
        ],
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "WARN"
    assert status["candidate_exit_resolver"]["max_days_in_awaiting_state"] == 11
    assert status["candidate_exit_resolver"]["still_awaiting_row_count"] == 1
    assert any("awaiting chain data" in issue["message"] for issue in status["issues"])
    assert not any("-99" in issue["message"] for issue in status["issues"])


# ──────────────────────────────────────────────────────────────────────────
# Ops-AE C1b regression tests for the alert-stickiness fixes
# ──────────────────────────────────────────────────────────────────────────


def test_ops_ae_c1b_simulator_error_outside_age_window_does_not_alert(
    tmp_path: Path,
) -> None:
    """SEV-HIGH-1 regression: a simulator_error row with a `ts` older
    than max_candidate_resolver_run_age_hours must NOT trigger a FAIL.

    The original watchdog scanned the JSONL tail unconditionally — once
    a single sim_error landed, the FAIL persisted for hundreds of
    daily ticks until the row scrolled off. The fix bounds the count
    to events within the same age window the launchd-log check uses.
    """
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    # Sim_error from 5 days ago — well outside the 36h default window.
    _write_candidate_resolver_jsonl(
        cfg.candidate_resolver_jsonl,
        [
            {
                "ts": "2026-04-22T12:31:00+00:00",  # 5 days before now
                "recommendation_id": "rec_old_sim_err",
                "resolver_status": "permanently_failed:simulator_error",
                "failure_reason": None,
                "days_in_awaiting_state": None,
            }
        ],
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    # Sim_error row exists in the JSONL but is too old to alert on.
    # The aggregated status should be OK (no FAIL from resolver).
    assert status["status"] == "OK"
    assert status["candidate_exit_resolver"]["simulator_error_count"] == 0


def test_ops_ae_c1b_simulator_error_within_age_window_still_alerts(
    tmp_path: Path,
) -> None:
    """Positive companion to the age-window fix: a recent sim_error
    still triggers FAIL. Without this, the age filter could be too
    aggressive and silence real operational failures."""
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    # Sim_error from 6 hours ago — well within the 36h default window.
    _write_candidate_resolver_jsonl(
        cfg.candidate_resolver_jsonl,
        [
            {
                "ts": "2026-04-27T17:00:00+00:00",  # 6h before now
                "recommendation_id": "rec_recent_sim_err",
                "resolver_status": "permanently_failed:simulator_error",
                "failure_reason": None,
                "days_in_awaiting_state": None,
            }
        ],
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "FAIL"
    assert status["candidate_exit_resolver"]["simulator_error_count"] == 1


def test_ops_ae_c1b_simulator_error_without_ts_is_excluded(tmp_path: Path) -> None:
    """Defensive: events with missing/unparseable `ts` are NOT counted
    toward the simulator_error alert. They cannot be bounded by age
    so treating them as recent would re-introduce stickiness from
    corrupted historical data. The malformed-JSONL warning covers
    parse failures via a separate code path."""
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_candidate_resolver_jsonl(
        cfg.candidate_resolver_jsonl,
        [
            {
                # ts intentionally absent
                "recommendation_id": "rec_no_ts",
                "resolver_status": "permanently_failed:simulator_error",
                "failure_reason": None,
                "days_in_awaiting_state": None,
            }
        ],
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["candidate_exit_resolver"]["simulator_error_count"] == 0


def test_ops_ae_c1b_escalated_row_does_not_trigger_stuck_awaiting_warn(
    tmp_path: Path,
) -> None:
    """SEV-HIGH-2 regression: a row that spent N days awaiting and
    THEN escalated to retrying / permanently_failed must NOT keep
    triggering the stuck-awaiting WARN forever.

    The fix groups events by recommendation_id, keeps only the
    chronologically-latest event per id, and checks
    days_in_awaiting_state only across rows whose latest status is
    STILL awaiting_chain_data. The day-12 awaiting row from the
    same rec_id no longer fires once the row escalates."""
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_candidate_resolver_jsonl(
        cfg.candidate_resolver_jsonl,
        [
            # Older event: this row was awaiting for 12 days.
            {
                "ts": "2026-04-20T12:31:00+00:00",
                "recommendation_id": "rec_escalated",
                "resolver_status": "awaiting_chain_data",
                "failure_reason": None,
                "days_in_awaiting_state": 12,
            },
            # Latest event: same row has now escalated to terminal.
            # No days_in_awaiting_state on terminal rows.
            {
                "ts": "2026-04-27T12:31:00+00:00",
                "recommendation_id": "rec_escalated",
                "resolver_status": "permanently_failed:no_post_event_chain",
                "failure_reason": "no_post_event_chain",
                "days_in_awaiting_state": None,
            },
        ],
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    # The row escalated, so no stuck-awaiting WARN.
    # (The permanently_failed:no_post_event_chain row is recorded but
    # does not trigger a separate alert — the resolver's escalation
    # to terminal is the intended pipeline behavior, not an
    # operational failure.)
    assert status["candidate_exit_resolver"]["still_awaiting_row_count"] == 0
    assert status["candidate_exit_resolver"]["max_days_in_awaiting_state"] is None
    assert not any(
        issue.get("check") == "candidate_exit_resolver"
        and "awaiting chain data" in issue.get("message", "")
        for issue in status["issues"]
    )


def test_ops_ae_c1b_distinct_rows_still_counted_separately(tmp_path: Path) -> None:
    """Positive: the grouping logic must still detect a genuinely-stuck
    row across multiple recommendation_ids. Row A escalated; row B is
    still awaiting at 15 days. The WARN must still fire for row B."""
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_candidate_resolver_jsonl(
        cfg.candidate_resolver_jsonl,
        [
            # Row A — escalated, should not contribute.
            {
                "ts": "2026-04-20T12:31:00+00:00",
                "recommendation_id": "rec_a_escalated",
                "resolver_status": "awaiting_chain_data",
                "days_in_awaiting_state": 8,
            },
            {
                "ts": "2026-04-27T12:31:00+00:00",
                "recommendation_id": "rec_a_escalated",
                "resolver_status": "permanently_failed:no_post_event_chain",
                "days_in_awaiting_state": None,
            },
            # Row B — still awaiting at 15 days, must trigger WARN.
            {
                "ts": "2026-04-27T12:31:00+00:00",
                "recommendation_id": "rec_b_still_stuck",
                "resolver_status": "awaiting_chain_data",
                "days_in_awaiting_state": 15,
            },
        ],
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "WARN"
    assert status["candidate_exit_resolver"]["still_awaiting_row_count"] == 1
    assert status["candidate_exit_resolver"]["max_days_in_awaiting_state"] == 15
    assert any("15 day" in issue["message"] for issue in status["issues"])


def test_ops_ae_c1c_threshold_catches_missed_run_same_day_watchdog(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex Ops-AE C1c P2): the resolver fires at 12:30
    local, the daily watchdog fires at 22:15 local (~9h 45m later).
    A missed today-12:30 run leaves yesterday's 12:30 as the latest
    completion — about 33h 45m old at the 22:15 watchdog scan.

    With the prior 36h threshold, that miss was UNDER threshold so
    no WARN fired — the alert finally surfaced 24h later at the
    NEXT day's watchdog (yesterday's 12:30 = 57.75h old). One
    full day of silent miss.

    With the C1c-tightened 26h threshold, the same-day watchdog
    sees 33.75h > 26h → WARN fires same evening.

    This test plants exactly that scenario: a launchd log whose
    latest completion is ~33.75h old, and asserts the resolver
    health reports a stale-completion WARN.
    """
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)

    # Yesterday 12:30 UTC completion only — today's 12:30 was missed.
    # Now we're at today 22:15 UTC. Latest completion age = 33h 45m.
    cfg.candidate_resolver_launchd_log_path.write_text(
        "===== 2026-04-26T12:30:00Z candidate exit resolver start =====\n"
        "  count_balance_holds:    true\n"
        "===== 2026-04-26T12:30:00Z candidate exit resolver complete =====\n",
        encoding="utf-8",
    )

    now = datetime(2026, 4, 27, 22, 15, tzinfo=timezone.utc)
    status = build_evidence_health_status(config=cfg, now=now)

    # Stale-completion WARN must be present in the issues list.
    assert any(
        issue["check"] == "candidate_exit_resolver"
        and "stale" in issue["message"].lower()
        for issue in status["issues"]
    ), (
        f"With default 26h threshold and a ~33.75h-old completion, "
        f"stale-completion WARN must fire same-day at the 22:15 "
        f"watchdog. Got issues: {status['issues']!r}"
    )


def test_ops_ae_c1c_threshold_allows_fresh_completion(tmp_path: Path) -> None:
    """Positive companion: today's 12:30 completion (~9h 45m old at
    the 22:15 watchdog scan) must NOT trigger the stale WARN. This
    is the case the threshold tightening must preserve — false-
    positiving on a fresh same-day run would be worse than the
    sticky-miss bug it fixes.
    """
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)

    cfg.candidate_resolver_launchd_log_path.write_text(
        "===== 2026-04-27T12:30:00Z candidate exit resolver start =====\n"
        "  count_balance_holds:    true\n"
        "===== 2026-04-27T12:30:00Z candidate exit resolver complete =====\n",
        encoding="utf-8",
    )

    now = datetime(2026, 4, 27, 22, 15, tzinfo=timezone.utc)
    status = build_evidence_health_status(config=cfg, now=now)

    assert not any(
        issue["check"] == "candidate_exit_resolver"
        and "stale" in issue["message"].lower()
        for issue in status["issues"]
    ), (
        f"A fresh ~9.75h-old completion must NOT trigger stale WARN. "
        f"Got issues: {status['issues']!r}"
    )


def test_ops_ae_c1b_count_balance_holds_regex_rejects_false_positives(
    tmp_path: Path,
) -> None:
    """SEV-MEDIUM-2 regression: the count_balance_holds parser uses a
    bounded regex now. Substring matching could false-positive on
    text like "see count_balance_holds: docs; this is not false alarm"
    (contains both label and "false" but not as the actual value).

    A line containing the label but with a non-true/false token after
    the colon must NOT be classified as either true or false — it's
    treated as "no verdict in this line" and the parser keeps looking.
    """
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    # Custom launchd log: noise line + real verdict line.
    cfg.candidate_resolver_launchd_log_path.write_text(
        "===== 2026-04-27T12:30:00Z candidate exit resolver start =====\n"
        "  Note: count_balance_holds: tracking enabled (not false alarm)\n"
        "  count_balance_holds:    true\n"
        "===== 2026-04-27T12:31:00Z candidate exit resolver complete =====\n",
        encoding="utf-8",
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    # Despite the noise line containing both "count_balance_holds:"
    # and "false" (in "false alarm"), the actual verdict is true.
    assert status["candidate_exit_resolver"]["count_balance_holds"] is True
    # And NO count_balance_holds:false FAIL should be emitted.
    assert not any(
        "count_balance_holds: false" in issue.get("message", "").lower()
        for issue in status["issues"]
    )


def test_evidence_health_warns_not_fails_before_candidate_resolver_first_run(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    cfg.candidate_resolver_launchd_log_path.unlink()

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    assert status["status"] == "WARN"
    assert any(
        issue["check"] == "candidate_exit_resolver" and "does not exist yet" in issue["message"]
        for issue in status["issues"]
    )
    # Ops-AE C1d: at 23:00 UTC (well past today's 12:30 UTC + 60min
    # grace = 13:30 UTC), the missing-log WARN MUST be alertable.
    # A fresh install whose first scheduled fire never happened is
    # operationally noteworthy and should page the operator.
    missing_log_issues = [
        issue for issue in status["issues"]
        if issue["check"] == "candidate_exit_resolver"
        and "does not exist yet" in issue["message"]
    ]
    assert all(issue.get("alertable", True) for issue in missing_log_issues)


def test_ops_ae_c1d_missing_log_before_due_window_is_non_alertable(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex Ops-AE C1c P2 audit): a fresh install
    whose launchd log doesn't exist YET because today's 12:30 fire
    hasn't happened must not false-alert the operator at an earlier
    health check.

    Scenario: operator runs `scripts/check_evidence_health.py`
    manually at 10:00 UTC. Today's resolver is due at 12:30 UTC
    (default). The log doesn't exist. C1c's "all resolver WARN
    escalates" rule would have caused this to alert.

    C1d fix: when the log is missing AND now is before the resolver
    due window, the WARN is tagged ``alertable=False``. It appears
    in the JSON output (for forensics) but the watchdog dispatcher
    must filter it out of the alert decision.
    """
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    cfg.candidate_resolver_launchd_log_path.unlink()

    # 10:00 UTC — well before the default 12:30 + 60min = 13:30 UTC
    # due window.
    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 10, 0, tzinfo=timezone.utc),
    )

    # The issue must appear in the payload for visibility:
    missing_log_issues = [
        issue for issue in status["issues"]
        if issue["check"] == "candidate_exit_resolver"
        and "does not exist yet" in issue["message"]
    ]
    assert len(missing_log_issues) == 1, (
        f"Expected exactly one missing-log WARN; got {missing_log_issues!r}"
    )
    # But it must be tagged non-alertable so the dispatcher doesn't page.
    assert missing_log_issues[0].get("alertable") is False, (
        f"Missing-log issue before due window must be alertable=False; "
        f"got {missing_log_issues[0]!r}"
    )
    # And the message must say "not due yet" to give operator context
    # if they read the JSON output.
    assert "not due yet" in missing_log_issues[0]["message"]


def test_ops_ae_c1d_missing_log_after_due_window_is_alertable(
    tmp_path: Path,
) -> None:
    """Positive companion to the C1d fix: AFTER the due window
    elapses (e.g. 22:15 UTC, well past today's 13:30 UTC cutoff),
    a missing log is a real operational concern and MUST remain
    alertable. The not-due-yet exception is narrow."""
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    cfg.candidate_resolver_launchd_log_path.unlink()

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 22, 15, tzinfo=timezone.utc),
    )

    missing_log_issues = [
        issue for issue in status["issues"]
        if issue["check"] == "candidate_exit_resolver"
        and "does not exist yet" in issue["message"]
    ]
    assert len(missing_log_issues) == 1
    # Past due → alertable (the C1c escalation rule applies).
    assert missing_log_issues[0].get("alertable", True) is True
    # And the message does NOT carry the "not due yet" qualifier.
    assert "not due yet" not in missing_log_issues[0]["message"]


def test_ops_ae_c1e_completed_run_without_count_balance_verdict_alerts(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex Ops-AE C1d P2 audit): the resolver CLI ALWAYS
    prints a ``count_balance_holds: true|false`` line in its summary
    block. If the launchd log shows a completed run but no
    parseable verdict, something is wrong: stdout was truncated,
    the CLI was modified, or the wrapper redirected output
    incorrectly.

    Absence of the integrity invariant verdict is itself an
    operational anomaly that MUST alert. Previously this passed
    silently (count_balance_holds stayed None and the downstream
    check only fired on explicit False).
    """
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)

    # Log shows start + complete markers but NO count_balance_holds
    # line — the summary block was truncated or never written.
    cfg.candidate_resolver_launchd_log_path.write_text(
        "===== 2026-04-27T12:30:00Z candidate exit resolver start =====\n"
        "PR-AE resolver run @ 2026-04-27T12:30:00+00:00 (now=2026-04-27)\n"
        "  scanned:                   0 candidate rows from ledger\n"
        # NOTE: no count_balance_holds line — the bug scenario.
        "===== 2026-04-27T12:31:00Z candidate exit resolver complete =====\n",
        encoding="utf-8",
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 22, 15, tzinfo=timezone.utc),
    )

    verdict_issues = [
        issue for issue in status["issues"]
        if issue["check"] == "candidate_exit_resolver"
        and "count_balance_holds verdict" in issue["message"]
    ]
    assert len(verdict_issues) == 1, (
        f"A completed resolver run without a count_balance_holds "
        f"verdict must emit a WARN. Got: {[i['message'] for i in status['issues']]}"
    )
    # And it must be alertable (the integrity invariant is missing —
    # operator should investigate, not just observe).
    assert verdict_issues[0].get("alertable", True) is True
    assert status["candidate_exit_resolver"]["count_balance_holds"] is None


def test_ops_ae_c1e_completed_run_with_true_verdict_does_not_alert(
    tmp_path: Path,
) -> None:
    """Positive companion: a normal completed run WITH a true
    verdict must NOT trigger the new C1e missing-verdict alert."""
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    # Default _seed_ok_files writes a log with count_balance_holds: true.

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 22, 15, tzinfo=timezone.utc),
    )

    assert not any(
        "count_balance_holds verdict" in issue["message"]
        for issue in status["issues"]
    )
    assert status["candidate_exit_resolver"]["count_balance_holds"] is True


def test_ops_ae_c1e_no_completion_marker_does_not_emit_verdict_warn(
    tmp_path: Path,
) -> None:
    """When latest_complete is None (no completion marker at all),
    the C1e missing-verdict WARN must NOT also fire — that case is
    already covered by the existing "no completion marker" WARN
    and emitting both would be redundant noise.

    Plant a log with only a `start` marker (no `complete` marker
    and no count_balance_holds line) and assert the C1e verdict
    WARN does not fire.
    """
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    cfg.candidate_resolver_launchd_log_path.write_text(
        # No complete marker, no count_balance_holds — incomplete run.
        "===== 2026-04-27T12:30:00Z candidate exit resolver start =====\n"
        "PR-AE resolver run @ 2026-04-27T12:30:00+00:00 (now=2026-04-27)\n",
        encoding="utf-8",
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 22, 15, tzinfo=timezone.utc),
    )

    # Existing "no completion marker" or similar WARN may fire, but
    # the NEW C1e verdict-absence WARN must NOT (it's gated on
    # latest_complete being non-None).
    assert not any(
        "count_balance_holds verdict" in issue["message"]
        for issue in status["issues"]
    )


def test_ops_ae_c1e_resolver_due_window_respects_config_override(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex Ops-AE C1d P3 audit): the launchd plist's
    StartCalendarInterval is LOCAL time but
    EvidenceHealthConfig.resolver_due_hour_utc is UTC. Defaults match
    only on UTC machines. Users in other tz must override.

    This test exercises the override path: a config set for US/Pacific
    (12:30 PT == 20:30 UTC) correctly treats 19:00 UTC as "still
    before due window" (it is, in PT terms: 11:00 local).
    """
    cfg_pst = EvidenceHealthConfig(
        expected_date=date(2026, 4, 27),
        report_dir=tmp_path / "reports" / "daily",
        weekly_report_dir=tmp_path / "reports" / "weekly",
        structured_run_log=tmp_path / "logs" / "evidence_cycle_runs.jsonl",
        launchd_log_path=tmp_path / "logs" / "daily_evidence_cycle_launchd.log",
        candidate_resolver_jsonl=tmp_path / "logs" / "candidate_exit_resolutions.jsonl",
        candidate_resolver_launchd_log_path=tmp_path / "logs" / "candidate_exit_resolver_launchd.log",
        ledger_path=tmp_path / "db" / "ledger.sqlite",
        outcome_store_path=tmp_path / "db" / "outcomes.sqlite",
        baseline_store_path=tmp_path / "db" / "baselines.sqlite",
        telemetry_store_path=tmp_path / "db" / "telemetry.sqlite",
        # US/Pacific PST: local 12:30 = UTC 20:30.
        resolver_due_hour_utc=20,
        resolver_due_minute_utc=30,
    )
    _seed_ok_files(cfg_pst)
    cfg_pst.candidate_resolver_launchd_log_path.unlink()

    # 19:00 UTC = 11:00 PST = before today's 12:30 PT fire.
    # The override (20:30 UTC + 60min grace = 21:30 UTC) means now is
    # BEFORE the due window, so missing log → alertable=False.
    status = build_evidence_health_status(
        config=cfg_pst,
        now=datetime(2026, 4, 27, 19, 0, tzinfo=timezone.utc),
    )

    missing_log_issues = [
        issue for issue in status["issues"]
        if issue["check"] == "candidate_exit_resolver"
        and "does not exist yet" in issue["message"]
    ]
    assert len(missing_log_issues) == 1
    assert missing_log_issues[0].get("alertable") is False, (
        "With PST-correct override (due_hour_utc=20), the resolver "
        "should NOT be considered due yet at 19:00 UTC (= 11:00 PST). "
        "Missing log → alertable=False. Without the override (default "
        "due_hour_utc=12), the resolver would be considered due since "
        "9:30 UTC, and missing log at 19:00 UTC would erroneously "
        "be alertable."
    )


def test_ops_ae_c1d_existing_issue_dicts_default_alertable_true(tmp_path: Path) -> None:
    """The new `alertable` field on _issue() defaults to True so all
    existing issue-emission sites remain alertable without explicit
    opt-in. Verified by checking that any pre-existing _issue() call
    site (e.g. stuck-awaiting WARN) still pages."""
    cfg = _config(tmp_path)
    _seed_ok_files(cfg)
    _write_candidate_resolver_jsonl(
        cfg.candidate_resolver_jsonl,
        [
            {
                "ts": "2026-04-27T12:31:00+00:00",
                "recommendation_id": "rec_stuck_default",
                "resolver_status": "awaiting_chain_data",
                "days_in_awaiting_state": 14,
            }
        ],
    )

    status = build_evidence_health_status(
        config=cfg,
        now=datetime(2026, 4, 27, 23, 0, tzinfo=timezone.utc),
    )

    stuck_issues = [
        issue for issue in status["issues"]
        if issue["check"] == "candidate_exit_resolver"
        and "awaiting chain data" in issue["message"]
    ]
    assert len(stuck_issues) == 1
    # Defaults to alertable=True; the explicit-False opt-in is only
    # on the not-due-yet path.
    assert stuck_issues[0].get("alertable", True) is True
