"""Unified exits (F6): one exit-due -> resolved stage per earnings event.

Three tracks resolve the same events on their own schedules and vocabularies:

- ``paper``: the forward loop's paper trades (outcome_trades), exited at T-1;
- ``baseline``: the paired and universe baseline trades (baseline_trades);
- ``resolver``: the candidate-exit resolver's shadow outcome on the event's
  latest ledger decision (recommendations), resolved after the event.

Nothing tied them together per event, so "how many events resolved" had three
different answers. This module maps every row onto one exit vocabulary and
groups the rows by the ledger's event key (SYMBOL|earnings_date):

    open      its exit day has not come yet
    exit_due  its exit day has come but no outcome is recorded yet
    resolved  an outcome is recorded
    missing   its exit window closed without an outcome (terminal)

An event's state is the least-finished of its tracks: exit_due while any
track still waits on its exit, resolved only when every track resolved, and
missing once all tracks are terminal with at least one missing.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

from services.recommendation_ledger import make_event_key

TRACK_PAPER = "paper"
TRACK_BASELINE = "baseline"
TRACK_RESOLVER = "resolver"
TRACKS = (TRACK_PAPER, TRACK_BASELINE, TRACK_RESOLVER)

STATE_OPEN = "open"
STATE_EXIT_DUE = "exit_due"
STATE_RESOLVED = "resolved"
STATE_MISSING = "missing"
STATES = (STATE_OPEN, STATE_EXIT_DUE, STATE_RESOLVED, STATE_MISSING)
TERMINAL_STATES = frozenset({STATE_RESOLVED, STATE_MISSING})

# Paper and baseline trades exit on T-1; the resolver reads post-event chains
# from MIN_DAYS_AFTER_EVENT calendar days after the event
# (services/candidate_exit_resolver.py) and gives up with permanently_failed.
RESOLVER_MIN_DAYS_AFTER_EVENT = 3


def _parse(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _due_state(earnings: Optional[date], as_of: date, *, days_after_event: int) -> str:
    """open / exit_due for a row that has no outcome yet."""
    if earnings is None:
        return STATE_OPEN
    exit_day = earnings + timedelta(days=days_after_event)
    return STATE_EXIT_DUE if as_of >= exit_day else STATE_OPEN


def paper_state(row: Dict[str, Any], as_of: date) -> str:
    status = str(row.get("status") or "")
    if status == "exit_missing":
        return STATE_MISSING
    if status in {"finalized", "exited"} and row.get("realized_return_pct") is not None:
        return STATE_RESOLVED
    return _due_state(_parse(row.get("earnings_date")), as_of, days_after_event=-1)


def baseline_state(row: Dict[str, Any], as_of: date) -> Optional[str]:
    """None for a baseline that never entered (nothing to resolve)."""
    status = str(row.get("status") or "")
    if status == "entry_skipped":
        return None
    if status == "resolved":
        return STATE_RESOLVED
    if status == "exit_missing":
        return STATE_MISSING
    earnings = _parse(row.get("earnings_date"))
    if status == "exit_skipped" and earnings is not None and as_of >= earnings:
        # A failed exit is only retried on its T-1 day.
        return STATE_MISSING
    return _due_state(earnings, as_of, days_after_event=-1)


def resolver_state(row: Dict[str, Any], as_of: date) -> str:
    outcome = row.get("candidate_shadow_outcome_json")
    status = str(outcome.get("status") or "") if isinstance(outcome, dict) else ""
    if status == "ok":
        return STATE_RESOLVED
    if status.startswith("permanently_failed:"):
        return STATE_MISSING
    return _due_state(_parse(row.get("earnings_date")), as_of, days_after_event=RESOLVER_MIN_DAYS_AFTER_EVENT)


def combine_states(states: Iterable[str]) -> str:
    states = list(states)
    if not states:
        return STATE_OPEN
    if STATE_EXIT_DUE in states:
        return STATE_EXIT_DUE
    if STATE_OPEN in states:
        return STATE_OPEN
    return STATE_MISSING if STATE_MISSING in states else STATE_RESOLVED


def build_event_resolutions(
    *,
    as_of: date,
    paper_rows: Iterable[Dict[str, Any]],
    baseline_rows: Iterable[Dict[str, Any]],
    resolver_rows: Iterable[Dict[str, Any]],
    event_keys: Optional[set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per event key: each track's rows with their state, and the event's state."""
    events: Dict[str, Dict[str, Any]] = {}

    def _add(track: str, row: Dict[str, Any], state: Optional[str], row_id: Any) -> None:
        if state is None:
            return
        key = make_event_key(row.get("symbol"), row.get("earnings_date"))
        if key is None or (event_keys is not None and key not in event_keys):
            return
        event = events.setdefault(key, {
            "event_key": key, "symbol": key.split("|")[0], "earnings_date": key.split("|")[1],
            "tracks": {},
        })
        event["tracks"].setdefault(track, []).append({"id": str(row_id), "state": state})

    for row in paper_rows:
        _add(TRACK_PAPER, row, paper_state(row, as_of), row.get("trade_id"))
    for row in baseline_rows:
        _add(TRACK_BASELINE, row, baseline_state(row, as_of), row.get("baseline_id"))
    for row in resolver_rows:
        _add(TRACK_RESOLVER, row, resolver_state(row, as_of), row.get("recommendation_id"))

    for event in events.values():
        track_states = {
            track: combine_states(item["state"] for item in items) for track, items in event["tracks"].items()
        }
        event["track_states"] = track_states
        event["state"] = combine_states(track_states.values())
    return dict(sorted(events.items(), key=lambda item: (item[1]["earnings_date"], item[0])))


def summarize_event_resolutions(events: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Event counts by state, per track, and the share of finished events that resolved."""
    by_state = {state: 0 for state in STATES}
    by_track: Dict[str, Dict[str, int]] = {track: {state: 0 for state in STATES} for track in TRACKS}
    for event in events.values():
        by_state[event["state"]] += 1
        for track, state in event["track_states"].items():
            by_track[track][state] += 1
    finished = by_state[STATE_RESOLVED] + by_state[STATE_MISSING]
    return {
        "events": len(events),
        "by_state": by_state,
        "by_track": by_track,
        # Of the events whose every track is finished, how many resolved.
        "resolution_rate": None if finished == 0 else round(by_state[STATE_RESOLVED] / finished, 4),
        "missing_events": [key for key, event in events.items() if event["state"] == STATE_MISSING][:50],
    }


def load_event_resolutions(
    *,
    as_of: date,
    ledger: Any = None,
    outcome_store: Any = None,
    baseline_store: Any = None,
    event_keys: Optional[set[str]] = None,
    max_rows: int = 50_000,
) -> Dict[str, Dict[str, Any]]:
    """Read the three tracks from their stores (defaults: the shared stores)."""
    from services.baseline_evidence_store import get_baseline_evidence_store
    from services.outcome_recorder import get_outcome_store
    from services.recommendation_ledger import get_recommendation_ledger

    from services.event_reconciliation import drop_superseded, superseded_keys

    ledger_obj = ledger or get_recommendation_ledger()
    superseded = superseded_keys(ledger_obj)
    paper = drop_superseded((outcome_store or get_outcome_store()).list_for_diagnostics(
        limit=max_rows, evidence="valid", source="forward"), superseded)
    baselines = drop_superseded(
        (baseline_store or get_baseline_evidence_store()).list_for_diagnostics(limit=max_rows), superseded)
    resolver = ledger_obj.latest_event_rows()
    return build_event_resolutions(as_of=as_of, paper_rows=paper, baseline_rows=baselines,
                                   resolver_rows=resolver, event_keys=event_keys)


__all__ = [
    "STATES", "TRACKS", "build_event_resolutions", "combine_states", "load_event_resolutions",
    "summarize_event_resolutions",
]
