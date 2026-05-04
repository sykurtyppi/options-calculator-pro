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
        ledger_path=base.ledger_path,
        outcome_store_path=base.outcome_store_path,
        baseline_store_path=base.baseline_store_path,
        telemetry_store_path=base.telemetry_store_path,
        max_daily_report_age_hours=base.max_daily_report_age_hours,
        max_weekly_report_age_days=base.max_weekly_report_age_days,
        max_telemetry_age_hours=base.max_telemetry_age_hours,
        max_run_log_age_hours=base.max_run_log_age_hours,
        require_completion_log=not args.no_completion_log_required,
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
