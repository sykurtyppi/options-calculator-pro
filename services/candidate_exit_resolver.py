"""
Live candidate exit resolver (PR-AE C4).

Walks ledger rows whose merged-view sample_provenance is the
forward-post-freeze constant and whose candidate_shadow_outcome is
still pending, then fills in the candidate's resolved PnL by
querying the post-event chain and re-running the shared shadow
simulator. This is the half of the PR-AE problem statement that
turns "candidate picker selection was recorded" into "candidate
picker outcome was measured" — the prerequisite for any future
promotion decision to be made on real forward evidence rather than
the in-sample +14d data.

Boundary discipline (PR-AE C2 + C3):

  - Imports stay inside ``services/`` and stdlib + pandas. NO edge
    into ``web/`` — the simulator was extracted to
    ``services/candidate_shadow_outcome.py`` in C2 precisely so the
    resolver could call it without forming a services → web
    dependency.

  - This module is a READER + REVISOR. It never originates a
    sample_provenance value — the rows it processes are already
    tagged at the live API path. The source-grep regression
    ``TestForwardProvenanceAssignmentBoundary`` enforces this at
    the codebase level: neither the authorized-assigner helper name
    nor the forward provenance string literal may appear in any
    file under ``services/`` outside the canonical definition
    module.

  - The original ``recommendations`` row is never mutated except via
    the explicit counter UPDATE inside
    ``RecommendationLedger.record_resolution_and_attempt``. All
    payload updates go through the PR-K revisions table.

Resolver status taxonomy (orthogonal to simulator status; see design
doc table for the full no-op-run rules):

  ``ok``                                        — full PnL resolved;
                                                  counter UNCHANGED.
  ``awaiting_chain_data``                       — post-event window
                                                  not yet elapsed;
                                                  counter UNCHANGED.
  ``retrying``                                  — window elapsed,
                                                  chain still missing,
                                                  attempts < MAX;
                                                  counter +1.
  ``permanently_failed:no_post_event_chain``    — window elapsed,
                                                  attempts hit MAX;
                                                  counter +1, then
                                                  row is terminal.
  ``permanently_failed:no_entry_chain_replay``  — as_of_date chain
                                                  unrecoverable;
                                                  terminal.
  ``permanently_failed:no_picker_provenance``   — picker_provenance
                                                  missing/malformed;
                                                  terminal.
  ``permanently_failed:simulator_<sim_status>`` — simulator returned
                                                  a terminal skipped:*;
                                                  terminal.
  ``permanently_failed:simulator_error``        — simulator raised
                                                  an exception
                                                  (caught here so
                                                  one bad row never
                                                  aborts the run);
                                                  terminal.

The ``ok`` resolver status is the only one that maps to a true
candidate shadow outcome dict from the simulator. Every other
status is wrapped over the simulator's empty-shape factory output so
``is_promotion_eligible`` (which requires both ``status == "ok"``
and a finite ``mid_realized_return_pct``) fails closed.

This module is the algorithmic core. The CLI / scheduler that
drives it lands in PR-AE C5 (``scripts/resolve_candidate_exits.py``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
from services.candidate_shadow_provenance import _empty_candidate_shadow_outcome
from services.recommendation_ledger import RecommendationLedger

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Constants — exported for tests to override
# ──────────────────────────────────────────────────────────────────────────

MIN_DAYS_AFTER_EVENT: int = 3
#   Calendar-day cutoff for ledger eligibility. Documented in
#   list_pending_candidate_exit_resolutions's docstring; the resolver
#   uses the ledger default unless a test overrides.

MAX_LOOKAHEAD_TRADE_DAYS: int = 5
#   Business-day window the resolver searches forward from
#   earnings_date for an available post-event chain. Past this
#   horizon, mixing 9-BDay-late exits with 3-BDay-late exits would
#   silently distort IV-crush interpretation in the aggregator.
#   See "Post-event chain date selection" in the design doc.

MAX_ATTEMPTS: int = 6
#   ≈ one week of daily resolver ticks before a row escalates from
#   retrying to permanently_failed:no_post_event_chain.


# Resolver status enum values. Kept as module-level constants so the
# CLI (C5) and tests can reference them without string drift.
RESOLVER_STATUS_OK = "ok"
RESOLVER_STATUS_AWAITING_CHAIN_DATA = "awaiting_chain_data"
RESOLVER_STATUS_RETRYING = "retrying"
RESOLVER_STATUS_PERMFAIL_NO_POST_EVENT_CHAIN = "permanently_failed:no_post_event_chain"
RESOLVER_STATUS_PERMFAIL_NO_ENTRY_CHAIN_REPLAY = "permanently_failed:no_entry_chain_replay"
RESOLVER_STATUS_PERMFAIL_NO_PICKER_PROVENANCE = "permanently_failed:no_picker_provenance"
RESOLVER_STATUS_PERMFAIL_SIMULATOR_ERROR = "permanently_failed:simulator_error"
RESOLVER_STATUS_PERMFAIL_SIMULATOR_PREFIX = "permanently_failed:simulator_"


# ──────────────────────────────────────────────────────────────────────────
# Result shapes
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ResolverOutcome:
    """Per-row resolution outcome. Returned by the resolver alongside
    the aggregated summary so the C5 CLI can write one JSONL line per
    row processed."""
    recommendation_id: str
    symbol: str
    earnings_date: Optional[str]
    exit_trade_date: Optional[str]
    attempt_number: int   # counter value AFTER this run's increment (if any)
    resolver_status: str
    simulator_status: Optional[str]
    mid_realized_return_pct: Optional[float]
    error_message: Optional[str] = None
    # PR-AE C5 watch item (Codex audit): `retrying` is the same
    # resolver_status string for two distinct underlying causes —
    # missing exit chain past the lookahead window vs. missing entry
    # chain. The persisted ledger payload's status remains the bare
    # `retrying` (no semantic change to what gets stored) but the
    # ResolverOutcome carries the sub-reason so the C5 CLI summary
    # can break down retrying by cause. Permanent failures already
    # encode the reason in resolver_status itself
    # (permanently_failed:<reason>); this field mirrors that for
    # the retrying path.
    #
    # Populated as one of:
    #   "no_post_event_chain"      — exit chain missing past window
    #   "no_entry_chain_replay"    — entry chain unrecoverable
    # None for any other resolver_status (ok, awaiting_chain_data,
    # skipped, simulator_error).
    failure_reason: Optional[str] = None


@dataclass
class ResolverRunSummary:
    """Aggregated counts over a single resolver run. The CLI (C5)
    asserts the balance invariant before printing the summary so
    drift between scanned and accounted-for counts is caught
    immediately.

    PR-AE C5 (Codex audit watch item): both ``retrying`` and
    ``permanently_failed`` carry sub-reason breakdowns so the
    operational summary distinguishes "exit chain missing past
    window" from "entry chain unrecoverable" without forcing the
    operator to drill into individual JSONL rows.
    """
    scanned: int = 0
    resolved_ok: int = 0
    awaiting_chain_data: int = 0
    retrying: int = 0
    skipped_malformed: int = 0
    retrying_by_reason: Dict[str, int] = field(default_factory=dict)
    permanently_failed_by_reason: Dict[str, int] = field(default_factory=dict)

    @property
    def permanently_failed_total(self) -> int:
        return sum(self.permanently_failed_by_reason.values())

    @property
    def count_balance_holds(self) -> bool:
        """Sum invariant: every scanned row belongs to exactly one
        outcome bucket. Returns False if the run has lost track of
        a row — the C5 CLI uses this to fail non-zero on drift.

        Note: ``retrying_by_reason`` is a *breakdown* of ``retrying``
        (sub-totals must equal the top-level retrying count); it is
        NOT a separate bucket in the count balance.
        """
        retrying_breakdown_balances = (
            sum(self.retrying_by_reason.values()) == self.retrying
        )
        bucket_sum_matches_scanned = self.scanned == (
            self.resolved_ok
            + self.awaiting_chain_data
            + self.retrying
            + self.skipped_malformed
            + self.permanently_failed_total
        )
        return bucket_sum_matches_scanned and retrying_breakdown_balances


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def resolve_pending_candidate_exits(
    ledger: RecommendationLedger,
    store: Any,
    *,
    now: Optional[date] = None,
    min_days_after_event: int = MIN_DAYS_AFTER_EVENT,
    max_lookahead_trade_days: int = MAX_LOOKAHEAD_TRADE_DAYS,
    max_attempts: int = MAX_ATTEMPTS,
) -> Tuple[List[ResolverOutcome], ResolverRunSummary]:
    """Resolve every currently-eligible candidate exit. Returns the
    per-row outcomes and an aggregated summary.

    Parameters
    ----------
    ledger:
        The :class:`RecommendationLedger` instance to read from and
        write back into. The resolver uses
        ``list_pending_candidate_exit_resolutions`` for eligibility
        and ``record_resolution_and_attempt`` for atomic write +
        counter UPDATE.
    store:
        An object implementing ``query_chain(symbol, *, trade_date,
        ...)`` compatible with ``OptionsFeatureStore``. Tests pass a
        stub.
    now:
        The "current date" for eligibility cutoff math. Defaults to
        ``datetime.now(timezone.utc).date()``. Inject an explicit date
        in tests.
    min_days_after_event, max_lookahead_trade_days, max_attempts:
        Override-able defaults; see module constants.

    Returns
    -------
    (List[ResolverOutcome], ResolverRunSummary)
        Per-row outcomes in the order they were resolved (oldest
        earnings_date first) plus an aggregated summary the CLI uses
        for stdout reporting and JSONL telemetry.

    Failure handling
    ----------------
    Every per-row resolution is wrapped in a try/except inside
    ``_resolve_one_row``. Unexpected exceptions are caught and
    converted into ``permanently_failed:simulator_error`` outcomes
    rather than aborting the run. The exception is logged at
    WARNING with the recommendation_id.
    """
    today = now or datetime.now(timezone.utc).date()
    pending = ledger.list_pending_candidate_exit_resolutions(
        now=today,
        min_days_after_event=min_days_after_event,
        max_attempts=max_attempts,
    )

    outcomes: List[ResolverOutcome] = []
    summary = ResolverRunSummary(scanned=len(pending))

    for row in pending:
        outcome = _resolve_one_row(
            row=row,
            ledger=ledger,
            store=store,
            today=today,
            max_lookahead_trade_days=max_lookahead_trade_days,
            max_attempts=max_attempts,
        )
        outcomes.append(outcome)
        _bucket_outcome(outcome, summary)

    return outcomes, summary


# ──────────────────────────────────────────────────────────────────────────
# Per-row resolution
# ──────────────────────────────────────────────────────────────────────────


def _resolve_one_row(
    *,
    row: Dict[str, Any],
    ledger: RecommendationLedger,
    store: Any,
    today: date,
    max_lookahead_trade_days: int,
    max_attempts: int,
) -> ResolverOutcome:
    """Resolve a single eligible ledger row. Returns the outcome. All
    exceptions are caught and converted into
    ``permanently_failed:simulator_error`` so one bad row never aborts
    the run."""
    rec_id = str(row["recommendation_id"])
    symbol = str(row.get("symbol") or "")
    earnings_date_str = row.get("earnings_date")
    prior_attempts = int(row.get("candidate_exit_resolver_attempts") or 0)

    try:
        # 1) picker_provenance must carry a usable candidate selection.
        picker_prov = row.get("picker_provenance_json") or {}
        candidate_sel = _extract_candidate_selection(picker_prov)
        if candidate_sel is None:
            return _write_terminal_no_simulator(
                ledger=ledger,
                rec_id=rec_id,
                symbol=symbol,
                earnings_date=earnings_date_str,
                exit_trade_date=None,
                prior_attempts=prior_attempts,
                resolver_status=RESOLVER_STATUS_PERMFAIL_NO_PICKER_PROVENANCE,
                # Distinguish from "simulator returned terminal"
                bucket_as_malformed=True,
            )

        # 2) Pick post-event trade_date via BDay lookahead.
        if not earnings_date_str:
            # Defensive — list_pending filters earnings_date IS NOT NULL,
            # but a future change could regress that.
            return _write_terminal_no_simulator(
                ledger=ledger,
                rec_id=rec_id,
                symbol=symbol,
                earnings_date=earnings_date_str,
                exit_trade_date=None,
                prior_attempts=prior_attempts,
                resolver_status=RESOLVER_STATUS_PERMFAIL_NO_POST_EVENT_CHAIN,
                bucket_as_malformed=True,
            )
        earnings_date = pd.Timestamp(earnings_date_str).date()
        candidate_dates = _post_event_bday_candidates(
            earnings_date, max_lookahead_trade_days,
        )
        exit_chain, exit_trade_date = _find_first_exit_chain(
            store, symbol, candidate_dates,
        )

        # 3) Decide between ok / awaiting / retrying / permfail based on
        #    chain availability + window-elapsed test.
        if exit_chain is None or exit_chain.empty:
            return _handle_missing_exit_chain(
                ledger=ledger,
                rec_id=rec_id,
                symbol=symbol,
                earnings_date=earnings_date,
                earnings_date_str=earnings_date_str,
                prior_attempts=prior_attempts,
                today=today,
                max_lookahead_trade_days=max_lookahead_trade_days,
                max_attempts=max_attempts,
            )

        # 4) Re-query entry chain at as_of_date. Codex C4 audit caution:
        # the entry chain SHOULD be deterministically present (it was
        # used at recommendation time), but if the underlying store has
        # transient I/O failures or in-flight re-processing windows,
        # the resolver should give it the same retry budget as the
        # exit chain rather than declaring the row permanently
        # unrecoverable on the first miss. Treats entry-chain-miss
        # exactly like exit-chain-miss past the lookahead window:
        # retrying until counter reaches max_attempts, then escalating
        # to permanently_failed:no_entry_chain_replay.
        as_of_date = row.get("as_of_date")
        entry_chain: Optional[pd.DataFrame] = None
        if as_of_date:
            try:
                entry_chain = store.query_chain(
                    symbol, trade_date=str(as_of_date),
                )
            except Exception as exc:
                logger.warning(
                    "PR-AE resolver: entry chain query failed for %s on %s: %s",
                    rec_id, as_of_date, exc,
                )
                entry_chain = None
        if entry_chain is None or entry_chain.empty:
            return _handle_missing_entry_chain(
                ledger=ledger,
                rec_id=rec_id,
                symbol=symbol,
                earnings_date_str=earnings_date_str,
                exit_trade_date=exit_trade_date,
                prior_attempts=prior_attempts,
                max_attempts=max_attempts,
            )

        # 5) Synthesize the minimal dual_picker the simulator needs.
        #    Only candidate_selection is consumed downstream; legacy and
        #    pickers_diverged are unused on this path.
        dual_picker = {
            "shadow_status": "ok",
            "candidate_selection": candidate_sel,
        }
        sim_result = simulate_candidate_shadow_outcome(
            entry_chain=entry_chain,
            exit_chain=exit_chain,
            dual_picker=dual_picker,
        )
        sim_status = str(sim_result.get("status") or "")

        # 6) Wrap the simulator result in the resolver status taxonomy.
        if sim_status == "ok":
            # Success — write through the non-incrementing path.
            ledger.record_resolution_and_attempt(
                recommendation_id=rec_id,
                candidate_shadow_outcome=sim_result,
                increment_attempts=False,
            )
            return ResolverOutcome(
                recommendation_id=rec_id,
                symbol=symbol,
                earnings_date=earnings_date_str,
                exit_trade_date=exit_trade_date,
                attempt_number=prior_attempts,
                resolver_status=RESOLVER_STATUS_OK,
                simulator_status=sim_status,
                mid_realized_return_pct=_as_float_or_none(
                    sim_result.get("mid_realized_return_pct")
                ),
            )

        # Simulator returned a terminal skipped:* — wrap as a
        # permanently_failed resolver status so is_promotion_eligible
        # fails closed, and increment the attempt counter.
        terminal_status = (
            f"{RESOLVER_STATUS_PERMFAIL_SIMULATOR_PREFIX}{sim_status}"
        )
        terminal_outcome = dict(sim_result)
        terminal_outcome["status"] = terminal_status
        ledger.record_resolution_and_attempt(
            recommendation_id=rec_id,
            candidate_shadow_outcome=terminal_outcome,
            increment_attempts=True,
        )
        return ResolverOutcome(
            recommendation_id=rec_id,
            symbol=symbol,
            earnings_date=earnings_date_str,
            exit_trade_date=exit_trade_date,
            attempt_number=prior_attempts + 1,
            resolver_status=terminal_status,
            simulator_status=sim_status,
            mid_realized_return_pct=None,
        )

    except Exception as exc:
        # Catch-all: never let a single bad row abort the resolver
        # run. Log at WARNING, attempt a best-effort terminal write,
        # and return an outcome the CLI summary will bucket as a
        # permanently_failed:simulator_error.
        logger.warning(
            "PR-AE resolver: unexpected exception resolving %s: %s",
            rec_id, exc,
        )
        terminal_outcome = _empty_candidate_shadow_outcome(
            RESOLVER_STATUS_PERMFAIL_SIMULATOR_ERROR
        )
        try:
            ledger.record_resolution_and_attempt(
                recommendation_id=rec_id,
                candidate_shadow_outcome=terminal_outcome,
                increment_attempts=True,
            )
            new_attempts = prior_attempts + 1
        except Exception:
            logger.exception(
                "PR-AE resolver: failed to record terminal error for %s",
                rec_id,
            )
            # If we cannot even write the terminal status, the counter
            # also did not bump (atomicity guarantee from C1b).
            new_attempts = prior_attempts
        return ResolverOutcome(
            recommendation_id=rec_id,
            symbol=symbol,
            earnings_date=earnings_date_str,
            exit_trade_date=None,
            attempt_number=new_attempts,
            resolver_status=RESOLVER_STATUS_PERMFAIL_SIMULATOR_ERROR,
            simulator_status=None,
            mid_realized_return_pct=None,
            error_message=str(exc),
        )


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────


def _extract_candidate_selection(picker_provenance: Any) -> Optional[Dict[str, Any]]:
    """Pull the four candidate-selection fields out of a
    picker_provenance_json payload. Returns None when the payload is
    missing, malformed, or carries a non-ok picker status (e.g.
    ``put_side_not_yet_supported``)."""
    if not isinstance(picker_provenance, dict):
        return None
    # Honor the PR-AC picker status — a row whose picker abstained
    # (put_calendar today) cannot have a candidate to resolve.
    pp_status = picker_provenance.get("status")
    if pp_status is not None and pp_status != "ok":
        return None
    candidate_contracts = picker_provenance.get("candidate_contracts")
    if not isinstance(candidate_contracts, dict):
        return None
    candidate_selection = candidate_contracts.get("candidate_selection")
    if not isinstance(candidate_selection, dict):
        return None
    for required_key in ("side", "strike", "front_expiry", "back_expiry"):
        if candidate_selection.get(required_key) in (None, ""):
            return None
    return {
        "side": candidate_selection["side"],
        "strike": candidate_selection["strike"],
        "front_expiry": candidate_selection["front_expiry"],
        "back_expiry": candidate_selection["back_expiry"],
    }


def _post_event_bday_candidates(
    earnings_date: date, max_lookahead_trade_days: int,
) -> List[str]:
    """Return ISO date strings for the post-event business-day window.

    Starts at ``earnings_date + 1 BDay`` and includes
    ``max_lookahead_trade_days`` business days. Skips weekends but
    not US market holidays — the design doc notes the 5-BDay default
    absorbs typical holidays without needing an exchange calendar
    dependency.
    """
    ts = pd.Timestamp(earnings_date)
    return [
        (ts + pd.tseries.offsets.BDay(i)).date().isoformat()
        for i in range(1, max_lookahead_trade_days + 1)
    ]


def _find_first_exit_chain(
    store: Any, symbol: str, candidate_dates: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return (first non-empty chain, its trade_date) or (None, None).

    Iterates the candidate dates in order. The first one that
    ``store.query_chain`` returns a non-empty frame for wins; failures
    are logged but never raised so a single bad date does not abort
    the resolver run.
    """
    for trade_date in candidate_dates:
        try:
            frame = store.query_chain(symbol, trade_date=trade_date)
        except Exception as exc:
            logger.debug(
                "PR-AE resolver: exit chain query failed for %s on %s: %s",
                symbol, trade_date, exc,
            )
            continue
        if frame is not None and not frame.empty:
            return frame, trade_date
    return None, None


def _handle_missing_exit_chain(
    *,
    ledger: RecommendationLedger,
    rec_id: str,
    symbol: str,
    earnings_date: date,
    earnings_date_str: str,
    prior_attempts: int,
    today: date,
    max_lookahead_trade_days: int,
    max_attempts: int,
) -> ResolverOutcome:
    """Window-elapsed decision tree when no exit chain was found.

    The design's three-way split:
      - today still inside the lookahead window → awaiting_chain_data
        (no counter increment; row eligible on next tick).
      - window elapsed, attempts will reach MAX → permanently_failed
        :no_post_event_chain (counter +1 → terminal).
      - window elapsed, attempts < MAX after increment → retrying
        (counter +1; row eligible on next tick).
    """
    # "Window elapsed" = today is >= earnings_date + max_lookahead BDays.
    # Computed in BDays since the lookahead itself is BDay-based.
    window_end = (
        pd.Timestamp(earnings_date)
        + pd.tseries.offsets.BDay(max_lookahead_trade_days)
    ).date()
    window_elapsed = today >= window_end

    if not window_elapsed:
        # Still waiting — chain may land in a future tick.
        # DO NOT increment the counter.
        terminal_outcome = _empty_candidate_shadow_outcome(
            RESOLVER_STATUS_AWAITING_CHAIN_DATA
        )
        ledger.record_resolution_and_attempt(
            recommendation_id=rec_id,
            candidate_shadow_outcome=terminal_outcome,
            increment_attempts=False,
        )
        return ResolverOutcome(
            recommendation_id=rec_id,
            symbol=symbol,
            earnings_date=earnings_date_str,
            exit_trade_date=None,
            attempt_number=prior_attempts,
            resolver_status=RESOLVER_STATUS_AWAITING_CHAIN_DATA,
            simulator_status=None,
            mid_realized_return_pct=None,
        )

    # Window elapsed, no chain. Increment and pick the right status
    # based on the post-increment attempt count.
    new_attempts = prior_attempts + 1
    if new_attempts >= max_attempts:
        resolver_status = RESOLVER_STATUS_PERMFAIL_NO_POST_EVENT_CHAIN
    else:
        resolver_status = RESOLVER_STATUS_RETRYING
    terminal_outcome = _empty_candidate_shadow_outcome(resolver_status)
    ledger.record_resolution_and_attempt(
        recommendation_id=rec_id,
        candidate_shadow_outcome=terminal_outcome,
        increment_attempts=True,
    )
    return ResolverOutcome(
        recommendation_id=rec_id,
        symbol=symbol,
        earnings_date=earnings_date_str,
        exit_trade_date=None,
        attempt_number=new_attempts,
        resolver_status=resolver_status,
        simulator_status=None,
        mid_realized_return_pct=None,
        failure_reason="no_post_event_chain",
    )


def _handle_missing_entry_chain(
    *,
    ledger: RecommendationLedger,
    rec_id: str,
    symbol: str,
    earnings_date_str: Optional[str],
    exit_trade_date: Optional[str],
    prior_attempts: int,
    max_attempts: int,
) -> ResolverOutcome:
    """Entry chain unavailable — retry-then-permfail at MAX.

    Codex C4 audit P2 fix. The entry chain SHOULD be deterministically
    present (it was used at recommendation time), but treating the
    first miss as permanent overreacts to any transient I/O failure
    or in-flight store reprocessing. This handler mirrors the
    exit-chain retry semantics: write a ``retrying`` revision while
    the counter is below ``max_attempts``, then escalate to
    ``permanently_failed:no_entry_chain_replay`` once the budget is
    exhausted. Same counter cadence as the exit-chain path so
    operators see a uniform escalation timeline regardless of which
    chain was missing.

    No window-elapsed check (unlike ``_handle_missing_exit_chain``)
    because the entry chain has no post-event window — it lives at
    the fixed ``as_of_date`` in the past. Every miss is "past the
    window" by definition.
    """
    new_attempts = prior_attempts + 1
    if new_attempts >= max_attempts:
        resolver_status = RESOLVER_STATUS_PERMFAIL_NO_ENTRY_CHAIN_REPLAY
    else:
        resolver_status = RESOLVER_STATUS_RETRYING
    terminal_outcome = _empty_candidate_shadow_outcome(resolver_status)
    ledger.record_resolution_and_attempt(
        recommendation_id=rec_id,
        candidate_shadow_outcome=terminal_outcome,
        increment_attempts=True,
    )
    return ResolverOutcome(
        recommendation_id=rec_id,
        symbol=symbol,
        earnings_date=earnings_date_str,
        exit_trade_date=exit_trade_date,
        attempt_number=new_attempts,
        resolver_status=resolver_status,
        simulator_status=None,
        mid_realized_return_pct=None,
        failure_reason="no_entry_chain_replay",
    )


def _write_terminal_no_simulator(
    *,
    ledger: RecommendationLedger,
    rec_id: str,
    symbol: str,
    earnings_date: Optional[str],
    exit_trade_date: Optional[str],
    prior_attempts: int,
    resolver_status: str,
    bucket_as_malformed: bool = False,
) -> ResolverOutcome:
    """Terminal write when no simulator output exists (picker
    provenance malformed, etc.). Uses the empty-block factory so
    the persisted shape is consistent with simulator-derived
    terminal statuses."""
    terminal_outcome = _empty_candidate_shadow_outcome(resolver_status)
    ledger.record_resolution_and_attempt(
        recommendation_id=rec_id,
        candidate_shadow_outcome=terminal_outcome,
        increment_attempts=True,
    )
    return ResolverOutcome(
        recommendation_id=rec_id,
        symbol=symbol,
        earnings_date=earnings_date,
        exit_trade_date=exit_trade_date,
        attempt_number=prior_attempts + 1,
        resolver_status=resolver_status,
        simulator_status=None,
        mid_realized_return_pct=None,
    )


def _bucket_outcome(outcome: ResolverOutcome, summary: ResolverRunSummary) -> None:
    """Increment the right summary count for *outcome*. Exhaustive
    over the resolver status taxonomy — every outcome must end up in
    exactly one bucket so the CLI's count-balance invariant holds."""
    status = outcome.resolver_status
    if status == RESOLVER_STATUS_OK:
        summary.resolved_ok += 1
    elif status == RESOLVER_STATUS_AWAITING_CHAIN_DATA:
        summary.awaiting_chain_data += 1
    elif status == RESOLVER_STATUS_RETRYING:
        summary.retrying += 1
        # PR-AE C5 watch item: also bucket under the specific failure
        # reason so the CLI summary distinguishes which chain was
        # missing. Falls back to "unknown" if the resolver returned
        # retrying without populating failure_reason — that shouldn't
        # happen, but the fallback keeps the breakdown's sub-totals
        # balanced against the top-level retrying count.
        reason = outcome.failure_reason or "unknown"
        summary.retrying_by_reason[reason] = (
            summary.retrying_by_reason.get(reason, 0) + 1
        )
    elif status == RESOLVER_STATUS_PERMFAIL_NO_PICKER_PROVENANCE:
        # Malformed picker provenance is conceptually a "skipped"
        # category, not a resolver-failure category — the resolver
        # never actually tried to compute anything. Bucket it under
        # skipped_malformed so the summary distinguishes "the
        # resolver tried and failed" from "we lacked the inputs to
        # try."
        summary.skipped_malformed += 1
    elif status.startswith("permanently_failed:"):
        # Strip the prefix for the reason key so the JSONL summary's
        # by-reason breakdown reads cleanly.
        reason = status[len("permanently_failed:"):]
        summary.permanently_failed_by_reason[reason] = (
            summary.permanently_failed_by_reason.get(reason, 0) + 1
        )
    else:
        # Unknown status — bucket under a sentinel reason to keep the
        # count balance holding. Future statuses should add an
        # explicit case above; this fallback prevents silent drops.
        summary.permanently_failed_by_reason["unknown_status:" + status] = (
            summary.permanently_failed_by_reason.get("unknown_status:" + status, 0) + 1
        )


def _as_float_or_none(value: Any) -> Optional[float]:
    """Coerce *value* to a finite float, returning None on any
    failure. Matches the type guard ``is_promotion_eligible`` uses on
    ``mid_realized_return_pct``."""
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            # bool is a subclass of int — exclude defensively, same as
            # is_promotion_eligible's bool-subclass guard.
            return None
        parsed = float(value)
        import math
        if not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None
