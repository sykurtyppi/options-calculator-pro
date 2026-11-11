#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from services.code_identity import build_code_identity_health, deployment_drift
from services.evidence_funnel import build_evidence_funnel_health
from services.market_calendar import is_trading_day
from services.provider_reliability import build_provider_budget_health
from services.evidence_health import (
    EvidenceHealthConfig,
    build_candidate_exit_resolver_health,
    build_launchd_job_freshness,
)
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


DEFAULT_ACTIVATION_PATH = Path.home() / ".options_calculator_pro" / "state" / "watchdog_activated_at"
DEFAULT_CYCLE_HOUR = 16
DEFAULT_CYCLE_MINUTE = 45


def _default_expected_date() -> date:
    return date.today()


def _latest_due_trading_date(
    *,
    now: datetime,
    due_hour: int,
    due_minute: int,
) -> date:
    """Return the latest NYSE session whose watchdog deadline has passed."""
    candidate = now.date()
    due_at = datetime.combine(
        candidate,
        time(hour=due_hour, minute=due_minute),
        tzinfo=now.tzinfo,
    )
    if now < due_at or not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def _watchdog_freshness_reference(
    *,
    expected_date: date,
    now: datetime,
    due_hour: int,
    due_minute: int,
    explicit_date: bool,
) -> datetime | None:
    """Anchor automatic report freshness to the selected session deadline.

    This keeps a valid Friday report healthy through a weekend or holiday.
    Once the next trading session reaches its deadline, expected_date advances
    and the missing newer report is detected by its date instead. Explicit
    historical checks retain the existing wall-clock age behavior.
    """
    if explicit_date:
        return None
    return datetime.combine(
        expected_date,
        time(hour=due_hour, minute=due_minute),
        tzinfo=now.tzinfo,
    )


def _read_activation_time(path: Path, *, now: datetime) -> datetime | None:
    """Read a trustworthy activation marker, failing closed on bad state."""
    try:
        raw = path.read_text(encoding="utf-8").strip()
        activated_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (OSError, ValueError):
        return None
    if activated_at.tzinfo is None or activated_at > now:
        return None
    return activated_at


def _scheduler_timezone(configured: str | None = None) -> ZoneInfo | None:
    """Resolve launchd's local timezone without reducing it to a fixed offset."""
    if configured:
        try:
            return ZoneInfo(configured)
        except (ZoneInfoNotFoundError, ValueError):
            return None

    candidates: list[str] = []
    if os.environ.get("TZ"):
        candidates.append(os.environ["TZ"].removeprefix(":"))

    try:
        localtime = Path("/etc/localtime").resolve()
        parts = localtime.parts
        zoneinfo_index = parts.index("zoneinfo")
        candidates.append("/".join(parts[zoneinfo_index + 1 :]))
    except (OSError, ValueError):
        pass

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError):
            continue
    return None


def _activation_defers_session(
    *,
    activated_at: datetime | None,
    expected_date: date,
    now: datetime,
    scheduler_tz: ZoneInfo | None,
    cycle_hour: int,
    cycle_minute: int,
) -> bool:
    """Whether this install was activated after the selected cycle fired."""
    if activated_at is None or scheduler_tz is None:
        return False
    cycle_at = datetime.combine(
        expected_date,
        time(hour=cycle_hour, minute=cycle_minute),
        tzinfo=scheduler_tz,
    )
    # Fresh-install activation is stamped immediately before launchctl load.
    # Equality therefore means the job was not active when that cycle fired.
    return activated_at >= cycle_at


def _build_combined_watchdog_status(
    *,
    watchdog_status: Mapping[str, Any],
    resolver_health: Mapping[str, Any],
    job_freshness: Mapping[str, Any] | None = None,
    funnel_health: Mapping[str, Any] | None = None,
    budget_health: Mapping[str, Any] | None = None,
    code_health: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge the daily-cycle watchdog status with the candidate exit
    resolver health into a single payload suitable for
    ``maybe_send_watchdog_alert``.

    Ops-AE C1c (Codex P1 audit fix): resolver WARN-severity issues
    MUST participate in the alert-dispatch decision. The prior code
    only escalated resolver FAILs into ``combined_status['ok']``, so
    documented alertable conditions like ``days_in_awaiting_state >
    10`` and ``stale completion`` (both emitted as WARN by
    evidence_health) silently failed to alert.

    Contract:
      - Resolver FAIL          → contributes to ``ok = False`` AND
        appears in ``errors`` (for the alert message body).
      - Resolver WARN          → contributes to ``ok = False`` AND
        appears in ``errors`` (escalated per Ops-AE taxonomy).
      - Resolver ``alertable=False`` issue (Ops-AE C1d, e.g. "log
        missing but resolver not due yet") → kept in the JSON
        payload via the resolver block but NEVER contributes to
        ``ok`` / ``errors`` regardless of severity. Forensics-only.
      - Daily-cycle WARN (from build_evidence_watchdog_status) →
        stays in ``warnings`` only, observation-only (existing
        behavior preserved).

    Extracted from main() so it's directly unit-testable without
    spawning the full CLI.
    """
    resolver_issues = resolver_health.get("issues", [])
    # Ops-AE C1d: honor the explicit `alertable` flag on issues so
    # "not due yet" type WARNs (logged for forensics) cannot page
    # the operator. Default to True for backward compatibility
    # with any issue dicts produced before C1d.
    resolver_alertable = [
        issue
        for issue in resolver_issues
        if issue.get("alertable", True)
        and issue.get("severity") in {"FAIL", "WARN"}
    ]
    # Resolver issues that were explicitly tagged non-alertable still
    # belong in the JSON output's warnings list so the operator can
    # inspect them via `scripts/check_evidence_health.py` output —
    # they just don't trigger iMessage alerts.
    resolver_non_alertable = [
        issue
        for issue in resolver_issues
        if not issue.get("alertable", True)
    ]
    # Codex audit F5: state-backup + screener-alert freshness/exit status use
    # the same alertable/non-alertable contract as the resolver issues.
    # The evidence funnel's issues (e.g. a stalled entry pipeline) follow the
    # same contract.
    # So do the provider credit budget's (exhausted or nearly so), and code
    # drift (the jobs running a stale or different checkout).
    job_issues = (
        list((job_freshness or {}).get("issues", []))
        + list((funnel_health or {}).get("issues", []))
        + list((budget_health or {}).get("issues", []))
        + list((code_health or {}).get("issues", []))
    )
    job_alertable = [i for i in job_issues if i.get("alertable", True) and i.get("severity") in {"FAIL", "WARN"}]
    job_non_alertable = [i for i in job_issues if not i.get("alertable", True)]

    return {
        **watchdog_status,
        "ok": bool(watchdog_status.get("ok")) and not resolver_alertable and not job_alertable,
        "errors": (
            list(watchdog_status.get("errors", []))
            + [str(i.get("message")) for i in resolver_alertable]
            + [str(i.get("message")) for i in job_alertable]
        ),
        # Daily-cycle warnings + resolver non-alertable warnings.
        # Resolver alertable WARNs have been escalated to `errors`
        # above and intentionally do NOT appear here too — avoid
        # double-reporting in the alert payload.
        "warnings": (
            list(watchdog_status.get("warnings", []))
            + [str(i.get("message")) for i in resolver_non_alertable]
            + [str(i.get("message")) for i in job_non_alertable]
        ),
        "candidate_exit_resolver": resolver_health.get("summary", {}),
        "launchd_jobs": (job_freshness or {}).get("summary", {}),
        "evidence_funnel": (funnel_health or {}).get("summary", {}),
        "provider_budget": (budget_health or {}).get("summary", {}),
        "code_identity": (code_health or {}).get("summary", {}),
    }


def _report_code_identity(report_path: Path) -> Mapping[str, Any] | None:
    """The code identity today's report recorded, or None when it has none."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    identity = report.get("code_identity") if isinstance(report, dict) else None
    return identity if isinstance(identity, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check daily evidence-cycle health and optionally send iMessage on failure."
    )
    parser.add_argument("--date", type=str, default=None, help="Expected evidence date (YYYY-MM-DD).")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--activation-path", type=Path, default=DEFAULT_ACTIVATION_PATH)
    parser.add_argument("--max-report-age-hours", type=float, default=DEFAULT_MAX_REPORT_AGE_HOURS)
    parser.add_argument("--no-completion-log-required", action="store_true")
    parser.add_argument("--candidate-resolver-jsonl", type=Path, default=None)
    parser.add_argument("--candidate-resolver-log", type=Path, default=None)
    # Ops-AE C1e (Codex P3 audit): see scripts/check_evidence_health.py
    # for the rationale. launchd fires at LOCAL 12:30; the resolver
    # due-window check uses UTC. Defaults work for UTC machines only.
    parser.add_argument(
        "--resolver-due-hour-utc",
        type=int,
        default=None,
        help=(
            "UTC hour at which the candidate exit resolver launchd "
            "job fires today. Defaults to 12. Override if your "
            "system local timezone is not UTC."
        ),
    )
    parser.add_argument(
        "--resolver-due-minute-utc",
        type=int,
        default=None,
        help="UTC minute portion of the resolver due time. Defaults to 30.",
    )
    parser.add_argument("--not-due-before-hour", type=int, default=22)
    parser.add_argument("--not-due-before-minute", type=int, default=15)
    parser.add_argument("--cycle-hour", type=int, default=DEFAULT_CYCLE_HOUR)
    parser.add_argument("--cycle-minute", type=int, default=DEFAULT_CYCLE_MINUTE)
    parser.add_argument(
        "--scheduler-timezone",
        help="IANA timezone used by launchd; defaults to the host's /etc/localtime zone.",
    )
    parser.add_argument("--no-fetch", action="store_true",
                        help="Compare the checkout with the last fetched origin/main instead of fetching.")
    parser.add_argument("--force-alert", action="store_true", help="Send even if this failure was already alerted.")
    parser.add_argument(
        "--dry-run-alert",
        "--dry-run-sms",
        action="store_true",
        help="Validate alert flow without sending iMessage.",
    )
    args = parser.parse_args()

    now = datetime.now().astimezone()
    expected = (
        _parse_date(args.date)
        if args.date
        else _latest_due_trading_date(
            now=now,
            due_hour=args.not_due_before_hour,
            due_minute=args.not_due_before_minute,
        )
    )
    if expected is None:
        raise SystemExit(f"Invalid --date value: {args.date}")

    # A durable activation marker establishes which scheduled cycles this
    # installation actually owned. Missing/corrupt/future markers fail closed:
    # they cannot suppress an alert. Explicit --date always performs the check.
    activated_at = _read_activation_time(args.activation_path, now=now)
    scheduler_tz = _scheduler_timezone(args.scheduler_timezone)
    if args.scheduler_timezone and scheduler_tz is None:
        raise SystemExit(f"Invalid --scheduler-timezone value: {args.scheduler_timezone}")
    if args.date is None and _activation_defers_session(
        activated_at=activated_at,
        expected_date=expected,
        now=now,
        scheduler_tz=scheduler_tz,
        cycle_hour=args.cycle_hour,
        cycle_minute=args.cycle_minute,
    ):
        assert activated_at is not None
        assert scheduler_tz is not None
        cycle_at = datetime.combine(
            expected, time(hour=args.cycle_hour, minute=args.cycle_minute),
            tzinfo=scheduler_tz,
        )
        status = {
            "ok": True,
            "status": "activation_deferred",
            "expected_date": expected.isoformat(),
            "report_path": str(args.report_dir / f"evidence_report_{expected.isoformat()}.json"),
            "latest_path": str(args.report_dir / "latest.json"),
            "log_path": str(args.log_path),
            "cycle_at": cycle_at.isoformat(),
            "activation_at": activated_at.isoformat(),
            "errors": [],
            "warnings": [
                "Evidence cycle predates watchdog activation; no alert is due for this session."
            ],
        }
    else:
        config = EvidenceWatchdogConfig(
            expected_date=expected,
            report_dir=args.report_dir,
            log_path=args.log_path,
            max_report_age_hours=args.max_report_age_hours,
            require_completion_log=not args.no_completion_log_required,
            freshness_reference_time=_watchdog_freshness_reference(
                expected_date=expected,
                now=now,
                due_hour=args.not_due_before_hour,
                due_minute=args.not_due_before_minute,
                explicit_date=args.date is not None,
            ),
        )
        status = build_evidence_watchdog_status(config=config)
    resolver_base = EvidenceHealthConfig(expected_date=expected)
    resolver_health = build_candidate_exit_resolver_health(
        config=EvidenceHealthConfig(
            expected_date=expected,
            candidate_resolver_jsonl=args.candidate_resolver_jsonl or resolver_base.candidate_resolver_jsonl,
            candidate_resolver_launchd_log_path=args.candidate_resolver_log or resolver_base.candidate_resolver_launchd_log_path,
            resolver_due_hour_utc=(
                args.resolver_due_hour_utc
                if args.resolver_due_hour_utc is not None
                else resolver_base.resolver_due_hour_utc
            ),
            resolver_due_minute_utc=(
                args.resolver_due_minute_utc
                if args.resolver_due_minute_utc is not None
                else resolver_base.resolver_due_minute_utc
            ),
            max_candidate_resolver_run_age_hours=resolver_base.max_candidate_resolver_run_age_hours,
            max_candidate_awaiting_days=resolver_base.max_candidate_awaiting_days,
        ),
        now=now,
    )
    job_freshness = build_launchd_job_freshness(
        config=EvidenceHealthConfig(expected_date=expected), now=now,
    )
    combined_status = _build_combined_watchdog_status(
        watchdog_status=status,
        resolver_health=resolver_health,
        job_freshness=job_freshness,
        funnel_health=build_evidence_funnel_health(),
        budget_health=build_provider_budget_health(),
        code_health=build_code_identity_health(
            deployment_drift(fetch=not args.no_fetch),
            _report_code_identity(Path(status["report_path"])),
            report_expected=status.get("status") not in {"market_closed", "activation_deferred"},
        ),
    )
    alert = maybe_send_watchdog_alert(
        combined_status,
        state_path=args.state_path,
        force=args.force_alert,
        dry_run=args.dry_run_alert,
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "watchdog": combined_status,
        "candidate_exit_resolver": resolver_health,
        "launchd_jobs": job_freshness,
        "alert": alert,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if combined_status.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
