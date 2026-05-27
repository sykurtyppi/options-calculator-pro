from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from services import external_io_gate


DEFAULT_HOME = Path.home() / ".options_calculator_pro"
DEFAULT_REPORT_DIR = DEFAULT_HOME / "reports" / "evidence"
DEFAULT_LOG_PATH = DEFAULT_HOME / "logs" / "daily_evidence_cycle_launchd.log"
DEFAULT_STATE_PATH = DEFAULT_HOME / "state" / "daily_evidence_watchdog_alerts.json"
DEFAULT_MAX_REPORT_AGE_HOURS = 30.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _read_json(path: Path) -> dict[str, Any] | None:
    # Hardening P2-7: distinguish "can't read the file" (perms regression,
    # mount unavailable) from "file is malformed JSON". Both still return
    # None — the callers' control flow stays unchanged — but an OSError
    # is now visible in the launchd stderr capture so an operator can see
    # whether to fix chmod vs investigate JSON corruption. JSONDecodeError
    # remains silent because the caller (build_evidence_watchdog_status)
    # already emits its own "not valid JSON" warning.
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        sys.stderr.write(
            f"automation_watchdog._read_json: cannot read {path} "
            f"({type(exc).__name__}: {exc})\n"
        )
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _completion_marker_for(day: date) -> str:
    return f"{day.isoformat()}T"


def _log_has_completion_marker(path: Path, expected_date: date) -> bool:
    if not path.exists():
        return False
    marker = "daily evidence cycle complete"
    expected_prefix = _completion_marker_for(expected_date)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    return any(expected_prefix in line and marker in line for line in lines[-500:])


@dataclass(frozen=True)
class EvidenceWatchdogConfig:
    expected_date: date
    report_dir: Path = DEFAULT_REPORT_DIR
    log_path: Path = DEFAULT_LOG_PATH
    max_report_age_hours: float = DEFAULT_MAX_REPORT_AGE_HOURS
    require_completion_log: bool = True


def build_evidence_watchdog_status(
    *,
    config: EvidenceWatchdogConfig | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a small operational health payload for the daily evidence cycle.

    This intentionally checks persistence artifacts instead of re-running the
    model. The watchdog should verify that the autonomous loop produced an
    auditable report; it must not become a second decision engine.
    """
    now_utc = now or _utc_now()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    cfg = config or EvidenceWatchdogConfig(expected_date=now_utc.date())

    report_path = cfg.report_dir / f"evidence_report_{cfg.expected_date.isoformat()}.json"
    latest_path = cfg.report_dir / "latest.json"
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] | None = None
    generated_at: datetime | None = None

    if not report_path.exists():
        errors.append(f"Missing evidence report for {cfg.expected_date.isoformat()}.")
    else:
        report = _read_json(report_path)
        if report is None:
            errors.append(f"Evidence report is not valid JSON: {report_path}.")
        else:
            report_date = str(report.get("as_of_date") or "")
            if report_date != cfg.expected_date.isoformat():
                errors.append(
                    f"Evidence report date mismatch: expected {cfg.expected_date.isoformat()}, found {report_date or 'missing'}."
                )
            generated_at = _parse_datetime(report.get("generated_at"))
            if generated_at is None:
                errors.append("Evidence report is missing a valid generated_at timestamp.")
            else:
                age_hours = (now_utc - generated_at).total_seconds() / 3600.0
                if age_hours > cfg.max_report_age_hours:
                    errors.append(
                        f"Evidence report is stale: {age_hours:.1f}h old, max {cfg.max_report_age_hours:.1f}h."
                    )
            forward_loop = report.get("forward_loop")
            evidence_report = report.get("evidence_report")
            if not isinstance(forward_loop, dict):
                errors.append("Evidence report is missing forward_loop output.")
            if not isinstance(evidence_report, dict):
                errors.append("Evidence report is missing evidence_report output.")

    latest = _read_json(latest_path) if latest_path.exists() else None
    if latest_path.exists() and latest is None:
        warnings.append("latest.json exists but is not valid JSON.")
    elif latest and str(latest.get("as_of_date") or "") != cfg.expected_date.isoformat():
        warnings.append("latest.json does not point at today's evidence report.")

    completion_logged = _log_has_completion_marker(cfg.log_path, cfg.expected_date)
    if cfg.require_completion_log and not completion_logged:
        errors.append("Daily evidence cycle completion marker is missing from launchd log.")

    return {
        "ok": not errors,
        "checked_at": now_utc.isoformat(),
        "expected_date": cfg.expected_date.isoformat(),
        "report_path": str(report_path),
        "latest_path": str(latest_path),
        "log_path": str(cfg.log_path),
        "generated_at": generated_at.isoformat() if generated_at else None,
        "completion_logged": completion_logged,
        "errors": errors,
        "warnings": warnings,
    }


@dataclass(frozen=True)
class IMessageConfig:
    to_address: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "IMessageConfig | None":
        env = environ or os.environ
        to_address = str(
            env.get("WATCHDOG_IMESSAGE_TO")
            or env.get("WATCHDOG_SMS_TO")
            or ""
        ).strip()
        if not to_address:
            return None
        return cls(to_address=to_address)


def _redacted_address(value: str) -> str:
    text = str(value or "").strip()
    if "@" in text:
        name, domain = text.split("@", 1)
        return f"{name[:2]}***@{domain}"
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return f"***{digits[-4:]}" if len(digits) >= 4 else "***"


def send_imessage(
    message: str,
    *,
    config: IMessageConfig,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    external_io_gate.assert_allowed(external_io_gate.Category.IMESSAGE)
    script = """
on run argv
  set targetAddress to item 1 of argv
  set watchdogMessage to item 2 of argv
  tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy targetAddress of targetService
    send watchdogMessage to targetBuddy
  end tell
end run
"""
    subprocess.run(
        ["osascript", "-e", script, config.to_address, message[:1500]],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "sent": True,
        "provider": "imessage",
        "to": _redacted_address(config.to_address),
    }


def _alert_signature(status: Mapping[str, Any]) -> str:
    errors = "|".join(str(item) for item in status.get("errors", []))
    return f"{status.get('expected_date')}::{errors}"


def _load_alert_state(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _write_alert_state(path: Path, payload: Mapping[str, Any]) -> None:
    # Hardening P2-6: atomic write via tmp + os.replace. A crash between
    # the open and the rename previously could corrupt the dedup state
    # file, which on the next watchdog tick would either be unreadable
    # (re-firing the same alert — pager noise) or partial (missing a
    # legitimate alert). Mirrors the pattern in
    # scripts/run_evidence_cycle._atomic_write_json.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(dict(payload), fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def maybe_send_watchdog_alert(
    status: Mapping[str, Any],
    *,
    imessage_config: IMessageConfig | None = None,
    state_path: Path = DEFAULT_STATE_PATH,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if bool(status.get("ok")):
        return {"attempted": False, "reason": "watchdog_ok"}

    signature = _alert_signature(status)
    state = _load_alert_state(state_path)
    if not force and state.get("last_alert_signature") == signature:
        return {"attempted": False, "reason": "duplicate_alert_suppressed"}

    cfg = imessage_config or IMessageConfig.from_env()
    if cfg is None:
        return {"attempted": False, "reason": "imessage_not_configured"}

    errors = "; ".join(str(item) for item in status.get("errors", []))
    message = (
        "[Options Calculator Watchdog] Daily evidence cycle failed. "
        f"Date={status.get('expected_date')}. Errors={errors or 'unknown'}. "
        f"Report={status.get('report_path')}"
    )
    if dry_run:
        result = {
            "sent": False,
            "provider": "imessage",
            "dry_run": True,
            "to": _redacted_address(cfg.to_address),
        }
    else:
        result = send_imessage(message, config=cfg)

    try:
        _write_alert_state(
            state_path,
            {
                "last_alert_signature": signature,
                "last_alerted_at": _utc_now().isoformat(),
                "last_status": dict(status),
                "last_alert_result": result,
            },
        )
    except Exception as exc:
        result = {
            **result,
            "state_persisted": False,
            "state_error": f"{type(exc).__name__}: {exc}",
        }
    else:
        result = {**result, "state_persisted": True}
    return {"attempted": True, **result}


# Backward-compatible alias for the CLI and older imports created during the
# initial watchdog pass. The transport is now iMessage, not provider SMS.
maybe_send_watchdog_sms = maybe_send_watchdog_alert
