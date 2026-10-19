"""Evidence filed under a superseded earnings date (PR O).

The recommendation ledger gives every earnings event a durable identity and
files the keys of an event whose date moved as superseded by its current key
(``RecommendationLedger.superseded_event_keys``). Paper outcomes and shadow
baselines are separate stores keyed by symbol and earnings date, so on their
own they kept counting evidence booked for a date the event no longer has.

A paper trade or baseline entered for a superseded date was timed around a
date the event did not happen on: it is not evidence about the event, so it
is invalidated, durably and with the reason (``event_date_superseded:<current
key>``), rather than reassociated; if the date moves back, it is restored. The row stays for audit. Invalidating
before the exit pass keeps it out of exits and the learning stores; a row
already finalized into calibration/priors is reported, because those stores
cannot drop one observation.

``reconcile_moved_events`` makes that durable (run each daily cycle, before
exits); ``drop_superseded`` applies the same rule at read time, so reports
are right even before the next reconciliation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional

from services.outcome_recorder import EVENT_DATE_SUPERSEDED_REASON
from services.recommendation_ledger import make_event_key

logger = logging.getLogger(__name__)

SUPERSEDED_REASON = EVENT_DATE_SUPERSEDED_REASON


def drop_superseded(rows: Iterable[Mapping[str, Any]], superseded: Mapping[str, str]) -> List[Dict[str, Any]]:
    """Rows whose event key is current (not superseded by a moved date)."""
    out = []
    for row in rows:
        key = make_event_key(row.get("symbol"), row.get("earnings_date"))
        if key is None or key not in superseded:
            out.append(dict(row))
    return out


def one_trade_per_event(rows: Iterable[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    """Each event's first paper trade (by entry date, then creation), in the
    input order, and how many later trades on the same event were left out.

    An event is one observation: a second structure booked for it on a later
    day is not independent evidence and would inflate n (the forward loop now
    books one trade per event; this covers rows recorded before that).
    """
    rows = [dict(row) for row in rows]
    first: Dict[str, tuple] = {}
    for index, row in enumerate(rows):
        key = make_event_key(row.get("symbol"), row.get("earnings_date"))
        if key is None:
            continue
        rank = (str(row.get("entry_date") or ""), str(row.get("created_at") or ""), str(row.get("trade_id") or ""))
        if key not in first or rank < first[key][0]:
            first[key] = (rank, index)
    keep = {index for _, index in first.values()}
    kept = [row for index, row in enumerate(rows)
            if index in keep or make_event_key(row.get("symbol"), row.get("earnings_date")) is None]
    return kept, len(rows) - len(kept)


def superseded_keys(ledger: Any) -> Dict[str, str]:
    """The ledger's superseded keys, or {} for a ledger that does not track them."""
    getter = getattr(ledger, "superseded_event_keys", None)
    return dict(getter()) if callable(getter) else {}


def _is_valid(row: Mapping[str, Any]) -> bool:
    value = row.get("evidence_valid")
    return value is None or str(value).strip().lower() not in {"0", "false"}


def _was_superseded(row: Mapping[str, Any]) -> bool:
    return not _is_valid(row) and str(row.get("invalidation_reason") or "").startswith(SUPERSEDED_REASON)


def reconcile_moved_events(*, ledger: Any, outcome_store: Any, baseline_store: Any,
                           max_rows: int = 50_000) -> Dict[str, Any]:
    """Invalidate forward evidence filed under superseded dates, restore
    evidence whose date is current again (a date that moved and moved back),
    and attach every row's durable event id. Idempotent.

    A row that cannot be written (a locked store, say) is reported in
    ``errors`` and retried next cycle; it never stops the others. Exits do not
    rely on this having succeeded: they skip superseded keys themselves.
    """
    superseded = superseded_keys(ledger)
    event_id_for = getattr(ledger, "event_id_for", None)
    summary: Dict[str, Any] = {"superseded_keys": len(superseded), "outcomes_invalidated": 0,
                               "baselines_invalidated": 0, "outcomes_revalidated": 0, "baselines_revalidated": 0,
                               "skipped_finalizing": [], "learning_already_applied": [], "event_ids_recorded": 0,
                               "errors": []}

    def _attempt(label: str, fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one row never stops the rest
            summary["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
            logger.warning("event reconciliation: %s failed: %s", label, exc)
            return None

    outcome_ids: Dict[str, str] = {}
    outcome_rows = list(outcome_store.list_for_diagnostics(limit=max_rows, evidence="valid", source="forward"))
    outcome_rows += [row for row in outcome_store.list_for_diagnostics(limit=max_rows, evidence="invalid",
                                                                       source="forward") if _was_superseded(row)]
    for row in outcome_rows:
        key = make_event_key(row.get("symbol"), row.get("earnings_date"))
        if key is None:
            continue
        trade_id = str(row["trade_id"])
        if callable(event_id_for):
            outcome_ids[trade_id] = str(event_id_for(key))
        if _was_superseded(row):
            if key not in superseded and _attempt(f"revalidate {trade_id}", lambda: outcome_store.revalidate(
                    trade_id, reason_prefix=SUPERSEDED_REASON)):
                summary["outcomes_revalidated"] += 1
            continue
        if key not in superseded:
            continue
        try:
            result = outcome_store.invalidate(trade_id, reason=f"{SUPERSEDED_REASON}:{superseded[key]}")
        except ValueError as exc:
            # Mid-finalization: retried on the next cycle.
            summary["skipped_finalizing"].append(trade_id)
            logger.warning("event reconciliation: %s", exc)
            continue
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"invalidate {trade_id}: {type(exc).__name__}: {exc}")
            logger.warning("event reconciliation: invalidate %s failed: %s", trade_id, exc)
            continue
        summary["outcomes_invalidated"] += 1
        if result.get("learning_already_applied"):
            summary["learning_already_applied"].append(trade_id)
    if outcome_ids:
        summary["event_ids_recorded"] += int(_attempt("outcome event ids",
                                                      lambda: outcome_store.record_event_ids(outcome_ids)) or 0)

    baseline_ids: Dict[str, str] = {}
    for row in baseline_store.list_for_diagnostics(limit=max_rows, include_invalid=True):
        key = make_event_key(row.get("symbol"), row.get("earnings_date"))
        restorable = _was_superseded(row)
        if key is None or not (_is_valid(row) or restorable):
            continue
        baseline_id = str(row["baseline_id"])
        if callable(event_id_for):
            baseline_ids[baseline_id] = str(event_id_for(key))
        if restorable:
            if key not in superseded and _attempt(f"revalidate {baseline_id}", lambda: baseline_store.revalidate(
                    baseline_id, reason_prefix=SUPERSEDED_REASON)):
                summary["baselines_revalidated"] += 1
        elif key in superseded and _attempt(f"invalidate {baseline_id}", lambda: baseline_store.invalidate(
                baseline_id, reason=f"{SUPERSEDED_REASON}:{superseded[key]}")):
            summary["baselines_invalidated"] += 1
    if baseline_ids:
        summary["event_ids_recorded"] += int(_attempt("baseline event ids",
                                                      lambda: baseline_store.record_event_ids(baseline_ids)) or 0)

    if summary["learning_already_applied"]:
        logger.warning("event reconciliation: %d superseded trades had already reached the learning stores: %s",
                       len(summary["learning_already_applied"]), summary["learning_already_applied"][:10])
    return summary


def safe_reconcile_moved_events(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """Reconcile without ever stopping the cycle; an error is reported."""
    try:
        return reconcile_moved_events(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event reconciliation failed: %s", exc, exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


__all__ = ["SUPERSEDED_REASON", "drop_superseded", "one_trade_per_event", "reconcile_moved_events", "safe_reconcile_moved_events",
           "superseded_keys"]
