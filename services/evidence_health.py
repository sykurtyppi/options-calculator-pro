"""Operational health checks for the autonomous evidence loop.

These checks are intentionally operational. They verify that evidence is being
collected and persisted; they do not run selectors, change models, or provide
trading advice.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from services.automation_watchdog import (
    DEFAULT_LOG_PATH as DEFAULT_LAUNCHD_LOG_PATH,
    DEFAULT_REPORT_DIR,
    EvidenceWatchdogConfig,
    build_evidence_watchdog_status,
)
from services.baseline_evidence_store import _DEFAULT_STORE as DEFAULT_BASELINE_STORE
from services.outcome_recorder import _DEFAULT_STORE as DEFAULT_OUTCOME_STORE
from services.provider_telemetry import _DEFAULT_STORE as DEFAULT_TELEMETRY_STORE
from services.recommendation_ledger import _DEFAULT_LEDGER as DEFAULT_LEDGER_STORE

DEFAULT_HOME = Path.home() / ".options_calculator_pro"
DEFAULT_STRUCTURED_RUN_LOG = DEFAULT_HOME / "logs" / "evidence_cycle_runs.jsonl"
DEFAULT_CANDIDATE_RESOLVER_JSONL = DEFAULT_HOME / "logs" / "candidate_exit_resolutions.jsonl"
DEFAULT_CANDIDATE_RESOLVER_LAUNCHD_LOG = DEFAULT_HOME / "logs" / "candidate_exit_resolver_launchd.log"
DEFAULT_WEEKLY_REPORT_DIR = DEFAULT_REPORT_DIR / "weekly"
DEFAULT_MAX_DAILY_REPORT_AGE_HOURS = 36.0
DEFAULT_MAX_WEEKLY_REPORT_AGE_DAYS = 8
DEFAULT_MAX_TELEMETRY_AGE_HOURS = 48.0
DEFAULT_MAX_RUN_LOG_AGE_HOURS = 36.0
DEFAULT_MAX_CANDIDATE_RESOLVER_RUN_AGE_HOURS = 36.0
DEFAULT_MAX_CANDIDATE_AWAITING_DAYS = 10
DEFAULT_DAILY_DUE_HOUR_UTC = 21
DEFAULT_DAILY_DUE_MINUTE_UTC = 30
DEFAULT_DAILY_DUE_GRACE_MINUTES = 60


@dataclass(frozen=True)
class EvidenceHealthConfig:
    expected_date: date
    report_dir: Path = DEFAULT_REPORT_DIR
    weekly_report_dir: Path = DEFAULT_WEEKLY_REPORT_DIR
    structured_run_log: Path = DEFAULT_STRUCTURED_RUN_LOG
    launchd_log_path: Path = DEFAULT_LAUNCHD_LOG_PATH
    candidate_resolver_jsonl: Path = DEFAULT_CANDIDATE_RESOLVER_JSONL
    candidate_resolver_launchd_log_path: Path = DEFAULT_CANDIDATE_RESOLVER_LAUNCHD_LOG
    ledger_path: Path = DEFAULT_LEDGER_STORE
    outcome_store_path: Path = DEFAULT_OUTCOME_STORE
    baseline_store_path: Path = DEFAULT_BASELINE_STORE
    telemetry_store_path: Path = DEFAULT_TELEMETRY_STORE
    max_daily_report_age_hours: float = DEFAULT_MAX_DAILY_REPORT_AGE_HOURS
    max_weekly_report_age_days: int = DEFAULT_MAX_WEEKLY_REPORT_AGE_DAYS
    max_telemetry_age_hours: float = DEFAULT_MAX_TELEMETRY_AGE_HOURS
    max_run_log_age_hours: float = DEFAULT_MAX_RUN_LOG_AGE_HOURS
    max_candidate_resolver_run_age_hours: float = DEFAULT_MAX_CANDIDATE_RESOLVER_RUN_AGE_HOURS
    max_candidate_awaiting_days: int = DEFAULT_MAX_CANDIDATE_AWAITING_DAYS
    require_completion_log: bool = True
    daily_due_hour_utc: int = DEFAULT_DAILY_DUE_HOUR_UTC
    daily_due_minute_utc: int = DEFAULT_DAILY_DUE_MINUTE_UTC
    daily_due_grace_minutes: int = DEFAULT_DAILY_DUE_GRACE_MINUTES


def build_evidence_health_status(
    *,
    config: Optional[EvidenceHealthConfig] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    now_utc = _ensure_utc(now or datetime.now(timezone.utc))
    cfg = config or EvidenceHealthConfig(expected_date=now_utc.date())
    issues: list[dict[str, str]] = []

    daily_report = _check_daily_report(cfg, now_utc)
    issues.extend(daily_report["issues"])

    weekly_report = _check_weekly_report(cfg, now_utc)
    issues.extend(weekly_report["issues"])

    run_log = _check_structured_run_log(cfg, now_utc)
    issues.extend(run_log["issues"])

    provider = _check_provider_telemetry(cfg, now_utc)
    issues.extend(provider["issues"])

    candidate_resolver = build_candidate_exit_resolver_health(config=cfg, now=now_utc)
    issues.extend(candidate_resolver["issues"])

    databases = {
        "recommendation_ledger": _check_sqlite_store("recommendation_ledger", cfg.ledger_path),
        "outcome_store": _check_sqlite_store("outcome_store", cfg.outcome_store_path),
        "baseline_store": _check_sqlite_store("baseline_store", cfg.baseline_store_path),
        "provider_telemetry": _check_sqlite_store("provider_telemetry", cfg.telemetry_store_path),
    }
    for result in databases.values():
        issues.extend(result["issues"])

    watchdog = build_evidence_watchdog_status(
        config=EvidenceWatchdogConfig(
            expected_date=cfg.expected_date,
            report_dir=cfg.report_dir,
            log_path=cfg.launchd_log_path,
            max_report_age_hours=cfg.max_daily_report_age_hours,
            require_completion_log=cfg.require_completion_log,
        ),
        now=now_utc,
    )
    before_due = _is_before_daily_due_window(cfg, now_utc)
    if not watchdog.get("ok"):
        for error in watchdog.get("errors", []):
            if before_due and _is_not_due_error(error, cfg.expected_date):
                issues.append(
                    _issue(
                        "WARN",
                        "scheduled_run_watchdog",
                        f"Daily evidence cycle for {cfg.expected_date.isoformat()} is not due yet.",
                        "Rerun the health check after the scheduled evidence-cycle window closes.",
                    )
                )
                continue
            issues.append(
                _issue(
                    "FAIL",
                    "scheduled_run_watchdog",
                    str(error),
                    "Check launchd status, run scripts/run_evidence_cycle.py manually, then inspect the launchd log.",
                )
            )
    for warning in watchdog.get("warnings", []):
        issues.append(
            _issue(
                "WARN",
                "scheduled_run_watchdog",
                str(warning),
                "Inspect latest.json and rerun scripts/check_evidence_health.py after the next scheduled cycle.",
            )
        )

    status = _aggregate_status(issues)
    return {
        "status": status,
        "ok": status == "OK",
        "checked_at": now_utc.isoformat(),
        "expected_date": cfg.expected_date.isoformat(),
        "daily_report": daily_report["summary"],
        "weekly_report": weekly_report["summary"],
        "structured_run_log": run_log["summary"],
        "provider_telemetry": provider["summary"],
        "candidate_exit_resolver": candidate_resolver["summary"],
        "databases": {key: value["summary"] for key, value in databases.items()},
        "watchdog": watchdog,
        "issues": issues,
        "summary": _summary_text(status, issues),
    }



def build_candidate_exit_resolver_health(
    *,
    config: EvidenceHealthConfig,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Return operational health for the PR-AE candidate exit resolver.

    This check is deliberately about pipeline mechanics only: scheduled-run
    freshness, count-balance status, stuck awaiting rows, and simulator
    exceptions. It must not alert on PnL, win rate, or candidate-vs-legacy
    performance.
    """
    now_utc = _ensure_utc(now or datetime.now(timezone.utc))
    issues: list[dict[str, str]] = []

    log_summary = _candidate_resolver_log_summary(config, now_utc)
    issues.extend(log_summary["issues"])

    jsonl = _read_jsonl_tail(config.candidate_resolver_jsonl, check_name="candidate_exit_resolver")
    issues.extend(jsonl["issues"])
    events = jsonl["events"]
    latest_row_ts = _latest_event_timestamp(events)
    max_awaiting_days = _max_int_field(events, "days_in_awaiting_state")
    simulator_error_count = sum(
        1
        for event in events
        if event.get("resolver_status") == "permanently_failed:simulator_error"
        or event.get("failure_reason") == "simulator_error"
    )
    latest_status = str(events[-1].get("resolver_status")) if events else None

    if max_awaiting_days is not None and max_awaiting_days > config.max_candidate_awaiting_days:
        issues.append(
            _issue(
                "WARN",
                "candidate_exit_resolver",
                f"Candidate exit resolver has row(s) awaiting chain data for {max_awaiting_days} day(s).",
                "Inspect candidate_exit_resolutions.jsonl and provider feature-store freshness; do not infer anything from PnL.",
            )
        )
    if simulator_error_count > 0:
        issues.append(
            _issue(
                "FAIL",
                "candidate_exit_resolver",
                f"Candidate exit resolver recorded {simulator_error_count} simulator_error row(s).",
                "Inspect the resolver JSONL error_message fields and rerun after fixing the simulator/runtime issue.",
            )
        )

    return {
        "summary": {
            "launchd_log_path": str(config.candidate_resolver_launchd_log_path),
            "launchd_log_exists": config.candidate_resolver_launchd_log_path.exists(),
            "latest_completion_at": log_summary["summary"].get("latest_completion_at"),
            "latest_failure_at": log_summary["summary"].get("latest_failure_at"),
            "jsonl_path": str(config.candidate_resolver_jsonl),
            "jsonl_exists": config.candidate_resolver_jsonl.exists(),
            "events_scanned": len(events),
            "latest_row_ts": latest_row_ts.isoformat() if latest_row_ts else None,
            "latest_status": latest_status,
            "max_days_in_awaiting_state": max_awaiting_days,
            "simulator_error_count": simulator_error_count,
            "count_balance_holds": log_summary["summary"].get("count_balance_holds"),
        },
        "issues": issues,
    }

def _check_daily_report(cfg: EvidenceHealthConfig, now: datetime) -> dict[str, Any]:
    path = cfg.report_dir / f"evidence_report_{cfg.expected_date.isoformat()}.json"
    latest = cfg.report_dir / "latest.json"
    issues: list[dict[str, str]] = []
    payload = _read_json(path)
    generated_at = _parse_datetime((payload or {}).get("generated_at"))
    if payload is None:
        if _is_before_daily_due_window(cfg, now):
            issues.append(
                _issue(
                    "WARN",
                    "daily_evidence_report",
                    f"Daily evidence report for {cfg.expected_date.isoformat()} is not due yet.",
                    "Rerun the health check after the scheduled evidence-cycle window closes.",
                )
            )
            return {
                "summary": {
                    "path": str(path),
                    "exists": path.exists(),
                    "latest_path": str(latest),
                    "generated_at": None,
                    "not_due_yet": True,
                },
                "issues": issues,
            }
        issues.append(
            _issue(
                "FAIL",
                "daily_evidence_report",
                f"Missing or malformed daily evidence report: {path}.",
                "Run scripts/run_evidence_cycle.py and inspect ~/.options_calculator_pro/logs/daily_evidence_cycle_launchd.log.",
            )
        )
    else:
        if str(payload.get("as_of_date") or "") != cfg.expected_date.isoformat():
            issues.append(
                _issue(
                    "FAIL",
                    "daily_evidence_report",
                    f"Daily report date mismatch for {path}.",
                    "Delete only the bad report if it is corrupt, then rerun scripts/run_evidence_cycle.py for the expected date.",
                )
            )
        if generated_at is None:
            issues.append(
                _issue(
                    "FAIL",
                    "daily_evidence_report",
                    "Daily evidence report is missing a valid generated_at timestamp.",
                    "Rerun scripts/run_evidence_cycle.py; if repeated, inspect JSON serialization errors.",
                )
            )
        elif _age_hours(generated_at, now) > cfg.max_daily_report_age_hours:
            issues.append(
                _issue(
                    "FAIL",
                    "daily_evidence_report",
                    f"Daily evidence report is stale ({_age_hours(generated_at, now):.1f}h old).",
                    "Check whether the Mac was awake and launchd job com.optionscalculator.evidence-cycle is loaded.",
                )
            )
    if not latest.exists():
        issues.append(
            _issue(
                "WARN",
                "daily_evidence_report",
                "latest.json is missing.",
                "Run scripts/run_evidence_cycle.py once; it should rewrite latest.json.",
            )
        )
    return {
        "summary": {
            "path": str(path),
            "exists": path.exists(),
            "latest_path": str(latest),
            "generated_at": generated_at.isoformat() if generated_at else None,
            "not_due_yet": False,
        },
        "issues": issues,
    }


def _check_weekly_report(cfg: EvidenceHealthConfig, now: datetime) -> dict[str, Any]:
    latest = cfg.weekly_report_dir / "weekly_latest.json"
    issues: list[dict[str, str]] = []
    payload = _read_json(latest)
    generated_at = _parse_datetime((payload or {}).get("generated_at"))
    if payload is None:
        issues.append(
            _issue(
                "WARN",
                "weekly_evidence_report",
                f"Weekly evidence report is missing or malformed: {latest}.",
                "Run scripts/export_weekly_evidence_report.py or wait for the Monday evidence cycle.",
            )
        )
    elif generated_at is None:
        issues.append(
            _issue(
                "WARN",
                "weekly_evidence_report",
                "Weekly evidence report is missing generated_at.",
                "Regenerate with scripts/export_weekly_evidence_report.py.",
            )
        )
    elif (now - generated_at).days > cfg.max_weekly_report_age_days:
        issues.append(
            _issue(
                "WARN",
                "weekly_evidence_report",
                f"Weekly evidence report is stale ({(now - generated_at).days} days old).",
                "Load com.optionscalculator.weekly-evidence-report or run scripts/export_weekly_evidence_report.py.",
            )
        )
    return {
        "summary": {
            "latest_path": str(latest),
            "exists": latest.exists(),
            "generated_at": generated_at.isoformat() if generated_at else None,
        },
        "issues": issues,
    }


def _check_structured_run_log(cfg: EvidenceHealthConfig, now: datetime) -> dict[str, Any]:
    parsed = _read_jsonl_tail(cfg.structured_run_log)
    issues = list(parsed["issues"])
    events = parsed["events"]
    successes = [event for event in events if event.get("status") == "success"]
    failures = [event for event in events if event.get("status") == "fail"]
    latest_success = successes[-1] if successes else None
    latest_failure = failures[-1] if failures else None
    latest_success_end = _parse_datetime((latest_success or {}).get("end_time"))
    if latest_success is None:
        issues.append(
            _issue(
                "WARN",
                "structured_run_log",
                f"No successful structured evidence-cycle run found in {cfg.structured_run_log}.",
                "Run scripts/run_evidence_cycle.py once after this deployment; it now writes structured JSONL run summaries.",
            )
        )
    elif latest_success_end and _age_hours(latest_success_end, now) > cfg.max_run_log_age_hours:
        issues.append(
            _issue(
                "WARN",
                "structured_run_log",
                f"Last successful structured run is stale ({_age_hours(latest_success_end, now):.1f}h old).",
                "Check launchd job com.optionscalculator.evidence-cycle and the plain launchd log.",
            )
        )
    if latest_failure:
        failure_time = _parse_datetime(latest_failure.get("end_time") or latest_failure.get("start_time"))
        if failure_time and _age_hours(failure_time, now) <= cfg.max_run_log_age_hours:
            issues.append(
                _issue(
                    "FAIL",
                    "structured_run_log",
                    f"Recent evidence-cycle failure: {latest_failure.get('error') or 'unknown error'}.",
                    "Open the structured run log and launchd log, fix the provider/runtime error, then rerun the cycle.",
                )
            )
    return {
        "summary": {
            "path": str(cfg.structured_run_log),
            "events_scanned": len(events),
            "latest_success_end": latest_success_end.isoformat() if latest_success_end else None,
            "latest_status": (events[-1].get("status") if events else None),
        },
        "issues": issues,
    }


def _check_provider_telemetry(cfg: EvidenceHealthConfig, now: datetime) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    latest_ts: datetime | None = None
    row_count = 0
    if not cfg.telemetry_store_path.exists():
        issues.append(
            _issue(
                "WARN",
                "provider_telemetry",
                f"Provider telemetry DB does not exist: {cfg.telemetry_store_path}.",
                "Run the evidence cycle; provider calls should create telemetry automatically.",
            )
        )
    else:
        try:
            with sqlite3.connect(str(cfg.telemetry_store_path), timeout=2.0) as conn:
                row_count = int(conn.execute("SELECT COUNT(*) FROM provider_telemetry").fetchone()[0])
                row = conn.execute("SELECT request_timestamp FROM provider_telemetry ORDER BY request_timestamp DESC LIMIT 1").fetchone()
                latest_ts = _parse_datetime(row[0]) if row else None
        except Exception as exc:
            issues.append(
                _issue(
                    "FAIL",
                    "provider_telemetry",
                    f"Provider telemetry read failed: {type(exc).__name__}: {exc}.",
                    "Run sqlite3 integrity checks or move the telemetry DB aside after backing it up.",
                )
            )
    if row_count == 0:
        issues.append(
            _issue(
                "WARN",
                "provider_telemetry",
                "Provider telemetry has no rows.",
                "Run scripts/run_evidence_cycle.py and confirm provider calls are being instrumented.",
            )
        )
    elif latest_ts and _age_hours(latest_ts, now) > cfg.max_telemetry_age_hours:
        issues.append(
            _issue(
                "WARN",
                "provider_telemetry",
                f"Provider telemetry is stale ({_age_hours(latest_ts, now):.1f}h since last event).",
                "Check provider credentials/network and confirm the scheduled evidence cycle is running.",
            )
        )
    return {
        "summary": {
            "path": str(cfg.telemetry_store_path),
            "row_count": row_count,
            "latest_timestamp": latest_ts.isoformat() if latest_ts else None,
        },
        "issues": issues,
    }


def _check_sqlite_store(name: str, path: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    quick_check: str | None = None
    writable = False
    if not path.exists():
        issues.append(
            _issue(
                "WARN",
                name,
                f"SQLite store does not exist yet: {path}.",
                "Run the evidence cycle once; if still missing, check filesystem permissions.",
            )
        )
    else:
        try:
            with sqlite3.connect(str(path), timeout=2.0) as conn:
                quick_check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("CREATE TEMP TABLE IF NOT EXISTS ops_health_probe (value INTEGER)")
                conn.execute("INSERT INTO ops_health_probe VALUES (1)")
                conn.rollback()
                writable = True
        except sqlite3.OperationalError as exc:
            severity = "FAIL" if "locked" in str(exc).lower() or "corrupt" in str(exc).lower() else "WARN"
            issues.append(
                _issue(
                    severity,
                    name,
                    f"SQLite store check failed: {type(exc).__name__}: {exc}.",
                    "Close other writers, rerun the health check, and run sqlite3 PRAGMA quick_check if it persists.",
                )
            )
        except Exception as exc:
            issues.append(
                _issue(
                    "FAIL",
                    name,
                    f"SQLite store check failed: {type(exc).__name__}: {exc}.",
                    "Back up the DB, inspect with sqlite3, and regenerate only if the audit trail is unrecoverable.",
                )
            )
    if quick_check and quick_check.lower() != "ok":
        issues.append(
            _issue(
                "FAIL",
                name,
                f"SQLite quick_check returned {quick_check!r}.",
                "Back up the DB immediately and inspect/repair before relying on new evidence.",
            )
        )
    return {
        "summary": {
            "path": str(path),
            "exists": path.exists(),
            "quick_check": quick_check,
            "writable": writable,
        },
        "issues": issues,
    }



def _candidate_resolver_log_summary(cfg: EvidenceHealthConfig, now: datetime) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    path = cfg.candidate_resolver_launchd_log_path
    lines = _read_text_tail(path)
    latest_complete = _latest_log_marker(lines, "candidate exit resolver complete")
    latest_failed = _latest_log_marker(lines, "candidate exit resolver failed")
    latest_start = _latest_log_marker(lines, "candidate exit resolver start")
    latest_start_index = _latest_line_index(lines, "candidate exit resolver start")
    latest_run_lines = lines[latest_start_index:] if latest_start_index is not None else lines
    count_balance_false = any(
        "count_balance_holds:" in line and "false" in line.lower()
        for line in latest_run_lines
    )
    count_balance_true = any(
        "count_balance_holds:" in line and "true" in line.lower()
        for line in latest_run_lines
    )
    count_balance_holds: bool | None
    if count_balance_false:
        count_balance_holds = False
    elif count_balance_true:
        count_balance_holds = True
    else:
        count_balance_holds = None

    if not path.exists():
        issues.append(
            _issue(
                "WARN",
                "candidate_exit_resolver",
                f"Candidate exit resolver launchd log does not exist yet: {path}.",
                "Install/load com.optionscalculator.candidate-exit-resolver and verify the first scheduled run.",
            )
        )
    elif latest_complete is None:
        issues.append(
            _issue(
                "WARN",
                "candidate_exit_resolver",
                "Candidate exit resolver has no completion marker in the launchd log.",
                "Check launchctl status and run scripts/resolve_candidate_exits.py manually with --no-jsonl for a smoke test.",
            )
        )
    elif _age_hours(latest_complete, now) > cfg.max_candidate_resolver_run_age_hours:
        issues.append(
            _issue(
                "WARN",
                "candidate_exit_resolver",
                f"Candidate exit resolver completion is stale ({_age_hours(latest_complete, now):.1f}h old).",
                "Check whether com.optionscalculator.candidate-exit-resolver is loaded and whether the Mac was awake at 12:30.",
            )
        )

    if latest_failed and (latest_complete is None or latest_failed >= latest_complete):
        issues.append(
            _issue(
                "FAIL",
                "candidate_exit_resolver",
                "Latest candidate exit resolver launchd run failed.",
                "Open candidate_exit_resolver_launchd.log, fix the resolver/runtime error, then rerun scripts/resolve_candidate_exits.py.",
            )
        )
    if count_balance_holds is False:
        issues.append(
            _issue(
                "FAIL",
                "candidate_exit_resolver",
                "Candidate exit resolver reported count_balance_holds: false.",
                "Inspect candidate_exit_resolver_launchd.log and candidate_exit_resolutions.jsonl for an unbucketed resolver outcome.",
            )
        )

    return {
        "summary": {
            "latest_start_at": latest_start.isoformat() if latest_start else None,
            "latest_completion_at": latest_complete.isoformat() if latest_complete else None,
            "latest_failure_at": latest_failed.isoformat() if latest_failed else None,
            "count_balance_holds": count_balance_holds,
        },
        "issues": issues,
    }


def _read_text_tail(path: Path, *, max_lines: int = 500) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except Exception:
        return []


def _latest_line_index(lines: list[str], marker: str) -> int | None:
    for idx in range(len(lines) - 1, -1, -1):
        if marker in lines[idx]:
            return idx
    return None


def _latest_log_marker(lines: list[str], marker: str) -> datetime | None:
    idx = _latest_line_index(lines, marker)
    if idx is None:
        return None
    return _parse_log_marker_datetime(lines[idx])


def _parse_log_marker_datetime(line: str) -> datetime | None:
    prefix = "===== "
    if not line.startswith(prefix):
        return None
    rest = line[len(prefix):]
    raw = rest.split(" ", 1)[0]
    return _parse_datetime(raw)


def _latest_event_timestamp(events: list[dict[str, Any]]) -> datetime | None:
    timestamps = [_parse_datetime(event.get("ts")) for event in events]
    parsed = [ts for ts in timestamps if ts is not None]
    return max(parsed) if parsed else None


def _max_int_field(events: list[dict[str, Any]], key: str) -> int | None:
    values: list[int] = []
    for event in events:
        value = event.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            values.append(value)
    return max(values) if values else None

def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl_tail(path: Path, *, max_lines: int = 500, check_name: str = "structured_run_log") -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    events: list[dict[str, Any]] = []
    if not path.exists():
        return {"events": [], "issues": issues}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except Exception as exc:
        return {
            "events": [],
            "issues": [
                _issue(
                    "WARN",
                    check_name,
                    f"Could not read JSONL log: {type(exc).__name__}: {exc}.",
                    "Check file permissions under ~/.options_calculator_pro/logs.",
                )
            ],
        }
    malformed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    if malformed:
        issues.append(
            _issue(
                "WARN",
                check_name,
                f"JSONL log contains {malformed} malformed JSONL line(s).",
                "Rotate or inspect the log; new runs should write one JSON object per line.",
            )
        )
    return {"events": events, "issues": issues}


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return _ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _age_hours(ts: datetime, now: datetime) -> float:
    return max(0.0, (now - _ensure_utc(ts)).total_seconds() / 3600.0)


def _daily_due_deadline(cfg: EvidenceHealthConfig) -> datetime:
    due = datetime(
        cfg.expected_date.year,
        cfg.expected_date.month,
        cfg.expected_date.day,
        cfg.daily_due_hour_utc,
        cfg.daily_due_minute_utc,
        tzinfo=timezone.utc,
    )
    return due + timedelta(minutes=cfg.daily_due_grace_minutes)


def _is_before_daily_due_window(cfg: EvidenceHealthConfig, now: datetime) -> bool:
    return _ensure_utc(now) < _daily_due_deadline(cfg)


def _is_not_due_error(error: Any, expected_date: date) -> bool:
    text = str(error)
    expected = expected_date.isoformat()
    if expected in text and "Missing evidence report" in text:
        return True
    return "Daily evidence cycle completion marker is missing" in text


def _issue(severity: str, check: str, message: str, fix: str) -> dict[str, str]:
    return {"severity": severity, "check": check, "message": message, "fix": fix}


def _aggregate_status(issues: Iterable[dict[str, str]]) -> str:
    severities = {issue.get("severity") for issue in issues}
    if "FAIL" in severities:
        return "FAIL"
    if "WARN" in severities:
        return "WARN"
    return "OK"


def _summary_text(status: str, issues: list[dict[str, str]]) -> str:
    if status == "OK":
        return "OK: evidence automation artifacts are fresh and readable."
    return f"{status}: {len(issues)} issue(s) need attention."
