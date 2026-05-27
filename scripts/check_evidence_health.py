#!/usr/bin/env python3
"""Check operational health of the autonomous evidence cycle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from scripts.run_forward_loop import _parse_date
from services.evidence_health import EvidenceHealthConfig, build_evidence_health_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Check evidence-cycle scheduling, persistence, and telemetry health.")
    parser.add_argument("--date", type=str, default=None, help="Expected evidence date (YYYY-MM-DD).")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--weekly-report-dir", type=Path, default=None)
    parser.add_argument("--structured-run-log", type=Path, default=None)
    parser.add_argument("--candidate-resolver-jsonl", type=Path, default=None)
    parser.add_argument("--candidate-resolver-log", type=Path, default=None)
    # Ops-AE C1e (Codex P3 audit): the launchd plist's
    # StartCalendarInterval is LOCAL time, but EvidenceHealthConfig's
    # resolver_due_hour_utc / resolver_due_minute_utc are UTC. The
    # defaults (12, 30) match the plist (Hour=12, Minute=30) only
    # when the system local timezone IS UTC (e.g. Iceland year-round).
    # Users in other tz must override here. See
    # scripts/automation/README.md for the mapping table.
    parser.add_argument(
        "--resolver-due-hour-utc",
        type=int,
        default=None,
        help=(
            "UTC hour at which the candidate exit resolver launchd "
            "job fires today. Defaults to 12. Override if your "
            "system local timezone is not UTC; e.g. for US/Pacific "
            "(PT = UTC-8) with launchd local 12:30, pass 20."
        ),
    )
    parser.add_argument(
        "--resolver-due-minute-utc",
        type=int,
        default=None,
        help="UTC minute portion of the resolver due time. Defaults to 30.",
    )
    parser.add_argument("--no-completion-log-required", action="store_true")
    args = parser.parse_args()

    expected = _parse_date(args.date) if args.date else date.today()
    if expected is None:
        raise SystemExit(f"Invalid --date value: {args.date}")

    base = EvidenceHealthConfig(expected_date=expected)
    config = EvidenceHealthConfig(
        expected_date=expected,
        report_dir=args.report_dir or base.report_dir,
        weekly_report_dir=args.weekly_report_dir or base.weekly_report_dir,
        structured_run_log=args.structured_run_log or base.structured_run_log,
        launchd_log_path=base.launchd_log_path,
        candidate_resolver_jsonl=args.candidate_resolver_jsonl or base.candidate_resolver_jsonl,
        candidate_resolver_launchd_log_path=args.candidate_resolver_log or base.candidate_resolver_launchd_log_path,
        ledger_path=base.ledger_path,
        outcome_store_path=base.outcome_store_path,
        baseline_store_path=base.baseline_store_path,
        telemetry_store_path=base.telemetry_store_path,
        max_daily_report_age_hours=base.max_daily_report_age_hours,
        max_weekly_report_age_days=base.max_weekly_report_age_days,
        max_telemetry_age_hours=base.max_telemetry_age_hours,
        max_run_log_age_hours=base.max_run_log_age_hours,
        max_candidate_resolver_run_age_hours=base.max_candidate_resolver_run_age_hours,
        max_candidate_awaiting_days=base.max_candidate_awaiting_days,
        require_completion_log=not args.no_completion_log_required,
        resolver_due_hour_utc=(
            args.resolver_due_hour_utc
            if args.resolver_due_hour_utc is not None
            else base.resolver_due_hour_utc
        ),
        resolver_due_minute_utc=(
            args.resolver_due_minute_utc
            if args.resolver_due_minute_utc is not None
            else base.resolver_due_minute_utc
        ),
    )
    payload = build_evidence_health_status(config=config, now=datetime.now().astimezone())
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if payload["status"] == "FAIL":
        return 2
    if payload["status"] == "WARN":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
