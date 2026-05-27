#!/usr/bin/env python3
"""Run the daily forward loop and write an evidence-report snapshot.

This is the cron/launchd-friendly command for the automated 60-90 day evidence
collection window. It does not introduce new modeling logic; it orchestrates
the existing forward loop and read-only evidence report.
"""

from __future__ import annotations

import argparse
import json
import os
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

from scripts.run_forward_loop import _parse_date, run_daily_cycle
from services.automation_watchdog import IMessageConfig, send_imessage
from services.evidence_report import build_evidence_report, build_weekly_evidence_report
from services.provider_telemetry import build_provider_telemetry_diagnostics

DEFAULT_REPORT_DIR = Path.home() / ".options_calculator_pro" / "reports" / "evidence"
DEFAULT_WEEKLY_REPORT_DIR = DEFAULT_REPORT_DIR / "weekly"
DEFAULT_STRUCTURED_RUN_LOG = Path.home() / ".options_calculator_pro" / "logs" / "evidence_cycle_runs.jsonl"


def run_evidence_cycle(
    *,
    as_of: date | None = None,
    dry_run: bool = False,
    symbols: list[str] | None = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
    weekly_report_dir: Path = DEFAULT_WEEKLY_REPORT_DIR,
    structured_run_log: Path = DEFAULT_STRUCTURED_RUN_LOG,
    write_weekly: bool | None = None,
    notify_summary: bool = False,
) -> dict:
    started_at = datetime.now(timezone.utc)
    cycle_date = as_of or date.today()
    forward = run_daily_cycle(today=cycle_date, dry_run=dry_run, symbols=symbols)
    report = build_evidence_report()
    provider_telemetry = build_provider_telemetry_diagnostics(limit=25)
    weekly_due = cycle_date.weekday() == 0 if write_weekly is None else bool(write_weekly)
    weekly_report = build_weekly_evidence_report() if weekly_due else None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": cycle_date.isoformat(),
        "dry_run": dry_run,
        "forward_loop": forward,
        "evidence_report": report,
        "provider_telemetry": provider_telemetry,
        "weekly_report_due": weekly_due,
    }
    if not dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"evidence_report_{cycle_date.isoformat()}.json"
        # Hardening P0-1: atomic write via tmp + os.replace. A launchd kill
        # mid-write previously left the daily report or latest.json torn,
        # which made the watchdog emit a false-positive page (see
        # services/automation_watchdog._read_json — it returns None on
        # malformed JSON and the watchdog records that as an error).
        # Mirrors the pattern in services/structure_prior_store._save_locked.
        _atomic_write_json(path, payload)
        _atomic_write_json(report_dir / "latest.json", payload)
        payload["report_path"] = str(path)
        if weekly_report is not None:
            weekly_report_dir.mkdir(parents=True, exist_ok=True)
            weekly_path = weekly_report_dir / f"weekly_evidence_report_{cycle_date.isoformat()}.json"
            _atomic_write_json(weekly_path, weekly_report)
            _atomic_write_json(weekly_report_dir / "weekly_latest.json", weekly_report)
            payload["weekly_report_path"] = str(weekly_path)
        if notify_summary:
            payload["notification_summary"] = _send_local_summary(report, weekly_report=weekly_report, dry_run=False)
    elif notify_summary:
        payload["notification_summary"] = _send_local_summary(report, weekly_report=weekly_report, dry_run=True)
    _append_structured_run_log(
        structured_run_log,
        _run_log_payload(
            started_at=started_at,
            status="success",
            cycle_date=cycle_date,
            dry_run=dry_run,
            payload=payload,
        ),
    )
    return payload


def _send_local_summary(report: dict, *, weekly_report: dict | None = None, dry_run: bool = False) -> dict:
    config = IMessageConfig.from_env()
    if config is None:
        return {"attempted": False, "reason": "imessage_not_configured"}
    maturity = report.get("maturity") or {}
    gate = report.get("commercialization_gate") or {}
    warnings = report.get("warning_flags") or []
    weekly_note = " weekly_export=ready." if weekly_report else ""
    message = (
        "[Options Calculator Evidence] "
        f"maturity={maturity.get('maturity_label', 'unknown')}; "
        f"days={gate.get('active_evidence_days', 0)}; "
        f"resolved={gate.get('resolved_selector_outcomes', 0)}; "
        f"warnings={len(warnings)}.{weekly_note} "
        "System/evidence summary only, not trading advice."
    )
    if dry_run:
        return {"attempted": True, "sent": False, "dry_run": True, "message": message}
    try:
        sent = send_imessage(message, config=config)
    except Exception as exc:
        return {"attempted": True, "sent": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"attempted": True, **sent}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the automated daily evidence cycle.")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing trades, logs, baselines, or report files.")
    parser.add_argument("--date", type=str, default=None, help="Override as-of date (YYYY-MM-DD).")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated override universe.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for daily evidence report snapshots.")
    parser.add_argument("--weekly-report-dir", type=Path, default=DEFAULT_WEEKLY_REPORT_DIR, help="Directory for weekly evidence report exports.")
    parser.add_argument("--structured-run-log", type=Path, default=DEFAULT_STRUCTURED_RUN_LOG, help="JSONL operational run log.")
    parser.add_argument("--write-weekly", action="store_true", help="Force writing a weekly evidence export even if today is not Monday.")
    parser.add_argument("--no-weekly", action="store_true", help="Disable weekly evidence export for this run.")
    parser.add_argument("--notify-summary", action="store_true", help="Send a short iMessage system/evidence summary if configured.")
    args = parser.parse_args()

    symbols = [token.strip().upper() for token in args.symbols.split(",") if token.strip()] if args.symbols else None
    weekly_override = True if args.write_weekly else False if args.no_weekly else None
    started_at = datetime.now(timezone.utc)
    try:
        payload = run_evidence_cycle(
            as_of=_parse_date(args.date) if args.date else date.today(),
            dry_run=args.dry_run,
            symbols=symbols,
            report_dir=args.report_dir,
            weekly_report_dir=args.weekly_report_dir,
            structured_run_log=args.structured_run_log,
            write_weekly=weekly_override,
            notify_summary=args.notify_summary,
        )
    except Exception as exc:
        _append_structured_run_log(
            args.structured_run_log,
            _run_log_payload(
                started_at=started_at,
                status="fail",
                cycle_date=_parse_date(args.date) if args.date else date.today(),
                dry_run=args.dry_run,
                payload={},
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _run_log_payload(
    *,
    started_at: datetime,
    status: str,
    cycle_date: date,
    dry_run: bool,
    payload: dict,
    error: str | None = None,
) -> dict:
    ended_at = datetime.now(timezone.utc)
    forward = payload.get("forward_loop") or {}
    report = payload.get("evidence_report") or {}
    provider = (payload.get("provider_telemetry") or {}).get("totals") or {}
    maturity = report.get("maturity") or {}
    entries = forward.get("entries") or {}
    exits = forward.get("exits") or {}
    return {
        "event_type": "evidence_cycle_run",
        "status": status,
        "as_of_date": cycle_date.isoformat(),
        "dry_run": dry_run,
        "start_time": started_at.isoformat(),
        "end_time": ended_at.isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "recommendations_recorded": int(entries.get("ledger_records") or 0),
        "paper_entries_recorded": int(entries.get("entries") or 0),
        "outcomes_resolved": int(exits.get("exits") or 0),
        "provider_failures": int(provider.get("failures") or 0),
        "provider_fallback_count": int(provider.get("fallback_count") or 0),
        "provider_stale_count": int(provider.get("stale_count") or 0),
        "evidence_maturity_status": maturity.get("maturity_label"),
        "weekly_report_path": payload.get("weekly_report_path"),
        "warnings": report.get("warning_flags") or [],
        "error": error,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to *path* atomically.

    Hardening P0-1: writes to ``.<name>.<pid>.tmp`` in the destination
    directory, fsyncs the body, then ``os.replace``\\ s onto *path*.
    A crash or launchd kill between the open and the replace leaves the
    pre-existing file (if any) intact — the watchdog never observes a
    torn JSON document. Same pattern used by
    :func:`services.structure_prior_store.StructurePriorStore._save_locked`
    and :func:`services.calibration_service.CalibrationStore._save`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                # Pre-existing tmp from a prior crash; next successful
                # write will replace it. Don't propagate — the
                # destination is already correct.
                pass


def _append_structured_run_log(path: Path, event: dict) -> None:
    if bool(event.get("dry_run")):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except (OSError, ValueError) as exc:
        # Hardening P0-2: narrow the exception and surface the failure
        # via stderr (launchd captures it into the wrapper's log file).
        # Previously a bare ``except: pass`` hid disk-full and perms
        # regressions, which silently disabled the structured run log
        # — the very stream services/evidence_health.py uses to confirm
        # that the cycle ran. We still don't re-raise: the cycle's real
        # work (report files, ledger writes) has already completed by
        # the time this is called, and discarding that would be worse
        # than losing one JSONL line.
        sys.stderr.write(
            "WARNING: run_evidence_cycle failed to append structured run "
            f"log: {type(exc).__name__}: {exc} (path={path})\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
