#!/usr/bin/env python3
"""
PR-AE C5 — scheduled one-shot CLI for the candidate exit resolver.

Drives :py:func:`services.candidate_exit_resolver.resolve_pending_candidate_exits`
once, writes per-row JSONL telemetry, prints a structured stdout
summary with sub-reason breakdowns, and asserts the count-balance
invariant before exiting. Intended for launchd / cron invocation
on a daily cadence; this script does NOT install schedulers itself.

Shape mirrors ``scripts/watch_daily_evidence_cycle.py``: argparse,
``main() -> int`` returning a process exit code, stdout output
suitable for log capture, JSONL appended to a fixed path under
``~/.options_calculator_pro/logs/``.

The script is a THIN wrapper around the resolver service module:
all eligibility, chain-selection, simulator-invocation, and ledger-
write logic lives in ``services/candidate_exit_resolver.py``. The
script's only responsibilities are:

  1. Parse CLI arguments (--ledger-path, --feature-store-root,
     --now, --max-attempts, --min-days-after-event,
     --max-lookahead-trade-days, --jsonl-path, --quiet).
  2. Construct the ledger + feature store.
  3. Call the resolver.
  4. Enrich each outcome with ``days_in_awaiting_state`` for
     awaiting rows (computed by walking the row's revision history
     to find the first awaiting revision; the design's freshness
     visibility field).
  5. Write JSONL telemetry rows.
  6. Print a structured stdout summary.
  7. Assert the count-balance invariant; exit 1 on drift.

Exit codes:
  0 — clean run; summary's count_balance_holds is True.
  1 — count-balance invariant violated (operator must investigate;
       a row was scanned but no bucket claimed it, or
       retrying_by_reason sub-totals diverged from the retrying
       total).
  2 — fatal error before the resolver ran (e.g. malformed args,
      ledger path unreadable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.candidate_exit_resolver import (  # noqa: E402
    MAX_ATTEMPTS,
    MAX_LOOKAHEAD_TRADE_DAYS,
    MIN_DAYS_AFTER_EVENT,
    RESOLVER_STATUS_AWAITING_CHAIN_DATA,
    ResolverOutcome,
    ResolverRunSummary,
    resolve_pending_candidate_exits,
)
from services.options_feature_store import OptionsFeatureStore  # noqa: E402
from services.recommendation_ledger import (  # noqa: E402
    RecommendationLedger,
    _DEFAULT_LEDGER,
)


DEFAULT_JSONL_PATH = (
    Path.home()
    / ".options_calculator_pro"
    / "logs"
    / "candidate_exit_resolutions.jsonl"
)


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve pending candidate-shadow-outcome rows in the "
            "recommendation ledger (PR-AE)."
        )
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=_DEFAULT_LEDGER,
        help="Path to the recommendation_ledger.sqlite file.",
    )
    parser.add_argument(
        "--feature-store-root",
        type=Path,
        default=None,
        help=(
            "Optional override for the OptionsFeatureStore data root. "
            "Defaults to MARKET_DATA_ROOT env var or /Volumes/T9/market_data."
        ),
    )
    parser.add_argument(
        "--now",
        type=_parse_iso_date,
        default=None,
        help=(
            "ISO date (YYYY-MM-DD) used as 'today' for eligibility "
            "math. Defaults to today (UTC)."
        ),
    )
    parser.add_argument(
        "--min-days-after-event",
        type=int,
        default=MIN_DAYS_AFTER_EVENT,
        help=(
            f"Calendar-day cutoff for eligibility (default "
            f"{MIN_DAYS_AFTER_EVENT}; 3 covers Friday-earnings → "
            f"Monday)."
        ),
    )
    parser.add_argument(
        "--max-lookahead-trade-days",
        type=int,
        default=MAX_LOOKAHEAD_TRADE_DAYS,
        help=(
            f"Business-day lookahead horizon for post-event chain "
            f"selection (default {MAX_LOOKAHEAD_TRADE_DAYS})."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_ATTEMPTS,
        help=f"Resolver retry budget per row (default {MAX_ATTEMPTS}).",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=DEFAULT_JSONL_PATH,
        help=(
            "Append per-row resolution outcomes here as JSONL. "
            f"Default: {DEFAULT_JSONL_PATH}."
        ),
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Skip JSONL writes (useful for smoke tests).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the stdout summary (JSONL still written).",
    )
    return parser


def _compute_days_in_awaiting_state(
    ledger: RecommendationLedger,
    recommendation_id: str,
    today: date,
) -> Optional[int]:
    """Walk the row's revision history; if the EARLIEST revision
    carrying status awaiting_chain_data was N calendar days ago,
    return N. Returns None if no awaiting revision is found.

    Codex C1b watch item / freshness visibility field. The daily
    watchdog can raise an alert when any row's
    ``days_in_awaiting_state`` exceeds a threshold (e.g. 10 days)
    — indicates the post-event chain genuinely never landed and
    the row needs manual review.
    """
    revisions = ledger.get_revisions(recommendation_id)
    for rev in revisions:
        payload = rev.get("record")
        if not isinstance(payload, dict):
            continue
        outcome = payload.get("candidate_shadow_outcome")
        if not isinstance(outcome, dict):
            continue
        if outcome.get("status") != "awaiting_chain_data":
            continue
        revised_at_str = str(rev.get("revised_at") or "")
        try:
            # Stored as ISO 8601 UTC; parse defensively.
            revised_at = datetime.fromisoformat(
                revised_at_str.replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            continue
        delta = (today - revised_at.date()).days
        return max(0, int(delta))
    return None


def _outcome_to_jsonl_row(
    outcome: ResolverOutcome,
    *,
    ledger: RecommendationLedger,
    today: date,
    duration_ms: float,
) -> Dict[str, Any]:
    """Render a single ResolverOutcome as the C5 JSONL telemetry
    row. Schema is fixed (see module docstring); ordering reflects
    the design doc's example payload."""
    days_in_awaiting: Optional[int] = None
    if outcome.resolver_status == RESOLVER_STATUS_AWAITING_CHAIN_DATA:
        days_in_awaiting = _compute_days_in_awaiting_state(
            ledger, outcome.recommendation_id, today,
        )
    row: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recommendation_id": outcome.recommendation_id,
        "symbol": outcome.symbol,
        "earnings_date": outcome.earnings_date,
        "exit_trade_date": outcome.exit_trade_date,
        "attempt_number": outcome.attempt_number,
        "days_in_awaiting_state": days_in_awaiting,
        "resolver_status": outcome.resolver_status,
        "simulator_status": outcome.simulator_status,
        "mid_realized_return_pct": outcome.mid_realized_return_pct,
        "failure_reason": outcome.failure_reason,
        "duration_ms": round(duration_ms, 2),
    }
    if outcome.error_message:
        row["error_message"] = outcome.error_message
    return row


def _append_jsonl(jsonl_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append one line per row to *jsonl_path*. Creates the parent
    directory if missing. Newline-terminated for cleanly tail-able
    streams."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str))
            fh.write("\n")


def _format_summary(
    summary: ResolverRunSummary,
    *,
    now: date,
    run_started_at: datetime,
    duration_seconds: float,
) -> str:
    """Build the stdout summary block. Layout matches the C1b design
    doc Telemetry section's mock — fixed-width counts so daily
    log-tailing reads cleanly.

    PR-AE C5 watch item: ``retrying`` and ``permanently_failed`` are
    each broken down by sub-reason. The roll-up counts stay at the
    top so an operator scanning logs at a glance still sees the
    headline numbers.
    """
    lines: List[str] = []
    lines.append(
        f"PR-AE resolver run @ {run_started_at.isoformat(timespec='seconds')} "
        f"(now={now.isoformat()})"
    )
    lines.append(f"  scanned:                {summary.scanned:>4d} candidate rows from ledger")
    # Codex C5 audit (P2): `skipped_ineligible` is filtered at the SQL
    # layer by list_pending_candidate_exit_resolutions, so the resolver
    # never sees those rows. Reporting a hardcoded 0 was misleading —
    # surface the semantic explicitly instead.
    lines.append(
        f"  skipped_ineligible:    not_measured  (filtered at SQL layer by "
        f"list_pending_candidate_exit_resolutions)"
    )
    lines.append(f"  skipped_malformed:      {summary.skipped_malformed:>4d}  (picker_provenance missing/malformed)")
    lines.append(f"  resolved_ok:            {summary.resolved_ok:>4d}")
    lines.append(f"  awaiting_chain_data:    {summary.awaiting_chain_data:>4d}")
    lines.append(f"  retrying:               {summary.retrying:>4d}")
    if summary.retrying_by_reason:
        for reason, count in sorted(summary.retrying_by_reason.items()):
            lines.append(f"    {reason}:{count:>{max(1, 30 - len(reason))}d}")
    lines.append(f"  permanently_failed:     {summary.permanently_failed_total:>4d}")
    if summary.permanently_failed_by_reason:
        for reason, count in sorted(summary.permanently_failed_by_reason.items()):
            lines.append(f"    {reason}:{count:>{max(1, 30 - len(reason))}d}")
    lines.append(f"  duration:              {duration_seconds:>5.2f}s")
    lines.append(
        f"  count_balance_holds:    {str(summary.count_balance_holds).lower()}"
    )
    return "\n".join(lines)


def _run(
    args: argparse.Namespace,
    *,
    stdout=sys.stdout,
    stderr=sys.stderr,
) -> int:
    """Invoke the resolver and emit telemetry. Returns the process
    exit code. Split out from ``main()`` so the smoke test can
    invoke the full pipeline without spawning a subprocess."""
    today = args.now or datetime.now(timezone.utc).date()
    run_started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    ledger = RecommendationLedger(ledger_path=args.ledger_path)
    if args.feature_store_root is not None:
        store: Any = OptionsFeatureStore(data_root=args.feature_store_root)
    else:
        store = OptionsFeatureStore()

    outcomes, summary = resolve_pending_candidate_exits(
        ledger,
        store,
        now=today,
        min_days_after_event=args.min_days_after_event,
        max_lookahead_trade_days=args.max_lookahead_trade_days,
        max_attempts=args.max_attempts,
    )

    duration_seconds = time.perf_counter() - start
    # Approximate per-row duration as the wall-clock divided by row
    # count. Per-row timing inside the resolver would require
    # wrapping each _resolve_one_row in a perf_counter; deferred to
    # a future PR if it ever matters for ops.
    per_row_duration_ms = (
        (duration_seconds * 1000.0 / max(1, len(outcomes)))
        if outcomes else 0.0
    )

    if not args.no_jsonl and outcomes:
        rows = [
            _outcome_to_jsonl_row(
                o, ledger=ledger, today=today,
                duration_ms=per_row_duration_ms,
            )
            for o in outcomes
        ]
        _append_jsonl(args.jsonl_path, rows)

    if not args.quiet:
        stdout.write(
            _format_summary(
                summary, now=today,
                run_started_at=run_started_at,
                duration_seconds=duration_seconds,
            )
        )
        stdout.write("\n")

    if not summary.count_balance_holds:
        stderr.write(
            "ERROR: ResolverRunSummary count_balance_holds is False. "
            "A scanned row was not bucketed into resolved_ok / "
            "awaiting_chain_data / retrying / skipped_malformed / "
            "permanently_failed_by_reason, OR retrying_by_reason "
            "sub-totals diverged from the retrying count. Investigate "
            "the JSONL log.\n"
        )
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"FATAL: PR-AE resolver CLI aborted before completion: {exc!r}\n"
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
