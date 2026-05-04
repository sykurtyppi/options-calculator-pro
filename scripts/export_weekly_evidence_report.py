#!/usr/bin/env python3
"""Export the weekly evidence report without re-running the forward loop."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from services.evidence_report import build_weekly_evidence_report

DEFAULT_WEEKLY_REPORT_DIR = Path.home() / ".options_calculator_pro" / "reports" / "evidence" / "weekly"
DEFAULT_WEEKLY_LOG = Path.home() / ".options_calculator_pro" / "logs" / "weekly_evidence_exports.jsonl"


def export_weekly_evidence_report(
    *,
    as_of: date | None = None,
    report_dir: Path = DEFAULT_WEEKLY_REPORT_DIR,
    log_path: Path = DEFAULT_WEEKLY_LOG,
    dry_run: bool = False,
) -> dict:
    started_at = datetime.now(timezone.utc)
    cycle_date = as_of or date.today()
    report = build_weekly_evidence_report()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": cycle_date.isoformat(),
        "dry_run": dry_run,
        "weekly_evidence_report": report,
    }
    if not dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"weekly_evidence_report_{cycle_date.isoformat()}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        (report_dir / "weekly_latest.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        payload["weekly_report_path"] = str(path)
        _append_weekly_log(log_path, _weekly_log_payload(started_at=started_at, status="success", payload=payload))
    return payload


def _weekly_log_payload(*, started_at: datetime, status: str, payload: dict, error: str | None = None) -> dict:
    ended_at = datetime.now(timezone.utc)
    report = payload.get("weekly_evidence_report") or {}
    maturity = report.get("maturity") or {}
    forward = report.get("forward_recommendations") or {}
    return {
        "event_type": "weekly_evidence_export",
        "status": status,
        "as_of_date": payload.get("as_of_date"),
        "start_time": started_at.isoformat(),
        "end_time": ended_at.isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "resolved_outcome_count": int(forward.get("resolved_outcome_count") or 0),
        "evidence_maturity_status": maturity.get("maturity_label"),
        "weekly_report_path": payload.get("weekly_report_path"),
        "warnings": report.get("sample_size_warnings") or [],
        "error": error,
    }


def _append_weekly_log(path: Path, event: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Export weekly evidence report without running the forward loop.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date", type=str, default=None, help="Report date (YYYY-MM-DD).")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_WEEKLY_REPORT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_WEEKLY_LOG)
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    try:
        as_of = date.fromisoformat(args.date) if args.date else date.today()
        payload = export_weekly_evidence_report(
            as_of=as_of,
            report_dir=args.report_dir,
            log_path=args.log_path,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        _append_weekly_log(
            args.log_path,
            _weekly_log_payload(
                started_at=started_at,
                status="fail",
                payload={"as_of_date": args.date or date.today().isoformat()},
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
