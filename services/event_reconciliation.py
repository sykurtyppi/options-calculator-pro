"""Evidence filed under a superseded earnings date (PR O).

The recommendation ledger gives every earnings event a durable identity and
files the keys of an event whose date moved as superseded by its current key
(``RecommendationLedger.superseded_event_keys``). Paper outcomes and shadow
baselines are separate stores keyed by symbol and earnings date, so on their
own they kept counting evidence booked for a date the event no longer has.

A paper trade or baseline entered for a superseded date was timed around a
date the event did not happen on: it is not evidence about the event, so it
is invalidated, durably and with the reason (``event_date_superseded:<current
key>``), rather than reassociated. The row stays for audit. Invalidating
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

from services.recommendation_ledger import make_event_key

logger = logging.getLogger(__name__)

SUPERSEDED_REASON = "event_date_superseded"


def drop_superseded(rows: Iterable[Mapping[str, Any]], superseded: Mapping[str, str]) -> List[Dict[str, Any]]:
    """Rows whose event key is current (not superseded by a moved date)."""
    out = []
    for row in rows:
        key = make_event_key(row.get("symbol"), row.get("earnings_date"))
        if key is None or key not in superseded:
            out.append(dict(row))
    return out


def superseded_keys(ledger: Any) -> Dict[str, str]:
    """The ledger's superseded keys, or {} for a ledger that does not track them."""
    getter = getattr(ledger, "superseded_event_keys", None)
    return dict(getter()) if callable(getter) else {}


def reconcile_moved_events(*, ledger: Any, outcome_store: Any, baseline_store: Any,
                           max_rows: int = 50_000) -> Dict[str, Any]:
    """Invalidate forward evidence filed under superseded dates, and attach
    every row's durable event id. Idempotent."""
    superseded = superseded_keys(ledger)
    event_id_for = getattr(ledger, "event_id_for", None)
    summary: Dict[str, Any] = {"superseded_keys": len(superseded), "outcomes_invalidated": 0,
                               "baselines_invalidated": 0, "skipped_finalizing": [],
                               "learning_already_applied": [], "event_ids_recorded": 0}

    outcome_ids: Dict[str, str] = {}
    for row in outcome_store.list_for_diagnostics(limit=max_rows, evidence="valid", source="forward"):
        key = make_event_key(row.get("symbol"), row.get("earnings_date"))
        if key is None:
            continue
        if callable(event_id_for):
            outcome_ids[str(row["trade_id"])] = str(event_id_for(key))
        if key not in superseded:
            continue
        try:
            result = outcome_store.invalidate(str(row["trade_id"]),
                                              reason=f"{SUPERSEDED_REASON}:{superseded[key]}")
        except ValueError as exc:
            # Mid-finalization: retried on the next cycle.
            summary["skipped_finalizing"].append(str(row["trade_id"]))
            logger.warning("event reconciliation: %s", exc)
            continue
        summary["outcomes_invalidated"] += 1
        if result.get("learning_already_applied"):
            summary["learning_already_applied"].append(str(row["trade_id"]))
    if outcome_ids:
        summary["event_ids_recorded"] += int(outcome_store.record_event_ids(outcome_ids) or 0)

    baseline_ids: Dict[str, str] = {}
    for row in baseline_store.list_for_diagnostics(limit=max_rows):
        key = make_event_key(row.get("symbol"), row.get("earnings_date"))
        if key is None:
            continue
        if callable(event_id_for):
            baseline_ids[str(row["baseline_id"])] = str(event_id_for(key))
        if key in superseded and baseline_store.invalidate(
                str(row["baseline_id"]), reason=f"{SUPERSEDED_REASON}:{superseded[key]}"):
            summary["baselines_invalidated"] += 1
    if baseline_ids:
        summary["event_ids_recorded"] += int(baseline_store.record_event_ids(baseline_ids) or 0)

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


__all__ = ["SUPERSEDED_REASON", "drop_superseded", "reconcile_moved_events", "safe_reconcile_moved_events",
           "superseded_keys"]
