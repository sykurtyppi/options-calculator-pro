#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from services.automation_watchdog import (
    DEFAULT_LOG_PATH,
    DEFAULT_MAX_REPORT_AGE_HOURS,
    DEFAULT_REPORT_DIR,
    DEFAULT_STATE_PATH,
    EvidenceWatchdogConfig,
    build_evidence_watchdog_status,
    maybe_send_watchdog_alert,
)
from scripts.run_forward_loop import _parse_date


def _default_expected_date() -> date:
    return date.today()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check daily evidence-cycle health and optionally send iMessage on failure."
    )
    parser.add_argument("--date", type=str, default=None, help="Expected evidence date (YYYY-MM-DD).")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--max-report-age-hours", type=float, default=DEFAULT_MAX_REPORT_AGE_HOURS)
    parser.add_argument("--no-completion-log-required", action="store_true")
    parser.add_argument("--not-due-before-hour", type=int, default=22)
    parser.add_argument("--not-due-before-minute", type=int, default=15)
    parser.add_argument("--force-alert", action="store_true", help="Send even if this failure was already alerted.")
    parser.add_argument(
        "--dry-run-alert",
        "--dry-run-sms",
        action="store_true",
        help="Validate alert flow without sending iMessage.",
    )
    args = parser.parse_args()

    expected = _parse_date(args.date) if args.date else _default_expected_date()
    if expected is None:
        raise SystemExit(f"Invalid --date value: {args.date}")
    now = datetime.now().astimezone()
    if args.date is None and expected == now.date():
        due_at = datetime.combine(
            expected,
            time(hour=args.not_due_before_hour, minute=args.not_due_before_minute),
            tzinfo=now.tzinfo,
        )
        if now < due_at:
            payload = {
                "generated_at": now.isoformat(),
                "watchdog": {
                    "ok": True,
                    "status": "not_due_yet",
                    "expected_date": expected.isoformat(),
                    "due_at": due_at.isoformat(),
                    "errors": [],
                    "warnings": ["Daily evidence watchdog is not due yet."],
                },
                "alert": {"attempted": False, "reason": "not_due_yet"},
            }
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            return 0

    config = EvidenceWatchdogConfig(
        expected_date=expected,
        report_dir=args.report_dir,
        log_path=args.log_path,
        max_report_age_hours=args.max_report_age_hours,
        require_completion_log=not args.no_completion_log_required,
    )
    status = build_evidence_watchdog_status(config=config)
    alert = maybe_send_watchdog_alert(
        status,
        state_path=args.state_path,
        force=args.force_alert,
        dry_run=args.dry_run_alert,
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "watchdog": status,
        "alert": alert,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if status.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
