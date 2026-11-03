"""Where forward evidence stops, per earnings event, and how fast it arrives.

The daily funnel report counts candidate passes over at most 30 days: an
event analysed on seven cycles counts seven times, and it never says where an
event's journey ended. This answers the throughput question instead:

* every earnings event (symbol, earnings date) in the window is assigned the
  furthest stage it reached and how it ended there (claimable, still in
  flight, stopped with a reason, or deferred and never picked up again);
* the stage that stops the most events, and its leading reasons, is the
  binding constraint;
* claimable outcomes per week, and at that rate how long until the paid-beta
  gate's sample.

Read-only: the store is opened with ``mode=ro`` and nothing is written. An
event whose earnings date moved appears once per date (the funnel keys events
by the date it saw at the time).
"""
from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from services.evidence_funnel import (
    OUTCOME_ADVANCED,
    OUTCOME_DEFERRED,
    STAGE_ANALYZED,
    STAGE_CLAIMABLE,
    STAGES,
)

# The paid-beta gate's resolved sample (services.evidence_report.MIN_RESOLVED_SAMPLE);
# kept here so the diagnosis does not import the whole report stack.
GATE_SAMPLE = 30
DEFAULT_TRAILING_WEEKS = 8

# Final states, in report order.
FINAL_CLAIMABLE = "claimable"
FINAL_IN_FLIGHT = "in_flight"            # advanced at its furthest stage, event still ahead
FINAL_NO_FURTHER = "no_further_record"   # advanced, event passed, nothing recorded after
FINAL_STOPPED = "stopped"                # dropped or errored at its furthest stage
FINAL_DEFERRED = "deferred"              # last word was a deferral, event still ahead
FINAL_DEFERRED_LOST = "deferred_then_lost"  # deferred, and the event passed with no decision

_RANK = {OUTCOME_ADVANCED: 3, "dropped": 2, "error": 2, OUTCOME_DEFERRED: 1}


def read_funnel_rows(store_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Every funnel event and cycle row, read without touching the store."""
    conn = sqlite3.connect(f"{Path(store_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        events = [dict(row) for row in conn.execute(
            "SELECT as_of_date, stage, outcome, subject, symbol, earnings_date, reason, trade_id"
            " FROM funnel_events")]
        cycles = [dict(row) for row in conn.execute(
            "SELECT cycle_run_id, as_of_date, universe_size, discovered FROM funnel_cycles")]
    finally:
        conn.close()
    return {"events": events, "cycles": cycles}


def _week(day: str) -> str:
    d = date.fromisoformat(day[:10])
    return (d - timedelta(days=d.weekday())).isoformat()


def _event_final(rows: List[Mapping[str, Any]], earnings_date: Optional[str], as_of: date) -> Dict[str, Any]:
    deepest = max(STAGES.index(r["stage"]) for r in rows)
    stage = STAGES[deepest]
    at_stage = [r for r in rows if r["stage"] == stage]
    best = max(at_stage, key=lambda r: (_RANK.get(r["outcome"], 0), str(r["as_of_date"])))
    passed = earnings_date is not None and earnings_date[:10] < as_of.isoformat()
    if best["outcome"] == OUTCOME_ADVANCED:
        if stage == STAGE_CLAIMABLE:
            final, reason = FINAL_CLAIMABLE, None
        else:
            final, reason = (FINAL_NO_FURTHER if passed else FINAL_IN_FLIGHT), None
    elif best["outcome"] == OUTCOME_DEFERRED:
        latest = max(at_stage, key=lambda r: str(r["as_of_date"]))
        final, reason = (FINAL_DEFERRED_LOST if passed else FINAL_DEFERRED), latest["reason"]
    else:
        final, reason = FINAL_STOPPED, best["reason"]
    # Cycles on which the analyser ran for this event (it re-runs while the
    # event stays in the entry window without a trade).
    passes = len({r["as_of_date"] for r in rows if r["stage"] == STAGE_ANALYZED})
    return {"stage": stage, "final": final, "reason": reason, "passes": passes}


def _top(counter: Counter, n: int = 5) -> List[Dict[str, Any]]:
    return [{"reason": reason, "events": count}
            for reason, count in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]]


def diagnose_funnel(
    events: Iterable[Mapping[str, Any]],
    cycles: Iterable[Mapping[str, Any]],
    *,
    as_of: date,
    since: Optional[date] = None,
    trailing_weeks: int = DEFAULT_TRAILING_WEEKS,
    gate_sample: int = GATE_SAMPLE,
) -> Dict[str, Any]:
    """Per-event final stops, the binding stage, and claimable throughput.

    An event is in the window when it was first seen on or after ``since``;
    rows after ``as_of`` are ignored. Claimable throughput counts each trade
    once, in the week it first became claimable.
    """
    since_iso = since.isoformat() if since else None
    until_iso = as_of.isoformat()
    rows = [r for r in events if str(r["as_of_date"]) <= until_iso and r["stage"] in STAGES]

    by_event: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_event[(str(r["symbol"]).upper(), (str(r["earnings_date"])[:10] if r["earnings_date"] else None))].append(r)
    if since_iso:
        by_event = {k: v for k, v in by_event.items() if min(str(r["as_of_date"]) for r in v) >= since_iso}

    finals = {key: _event_final(v, key[1], as_of) for key, v in by_event.items()}
    by_final: Counter = Counter(f["final"] for f in finals.values())
    stopped_by_stage: Dict[str, Counter] = defaultdict(Counter)
    lost_by_reason: Counter = Counter()
    in_flight_by_stage: Counter = Counter()
    for f in finals.values():
        if f["final"] == FINAL_STOPPED:
            stopped_by_stage[f["stage"]][f["reason"] or "unknown"] += 1
        elif f["final"] == FINAL_DEFERRED_LOST:
            lost_by_reason[f["reason"] or "unknown"] += 1
        elif f["final"] in (FINAL_IN_FLIGHT, FINAL_NO_FURTHER):
            in_flight_by_stage[f"{f['final']}:{f['stage']}"] += 1

    # The stage that ends the most events; deferrals that were never picked
    # up again count against the stage they were deferred at.
    losses: Counter = Counter({stage: sum(c.values()) for stage, c in stopped_by_stage.items()})
    lost_stage: Counter = Counter(f["stage"] for f in finals.values() if f["final"] == FINAL_DEFERRED_LOST)
    losses.update(lost_stage)
    binding = None
    if losses:
        stage, n = max(losses.items(), key=lambda kv: (kv[1], -STAGES.index(kv[0])))
        reasons = Counter(stopped_by_stage.get(stage, Counter()))
        for f in finals.values():
            if f["final"] == FINAL_DEFERRED_LOST and f["stage"] == stage:
                reasons[f"deferred_then_lost:{f['reason']}"] += 1
        binding = {"stage": stage, "events_lost": n, "share_of_events": round(n / len(finals), 4),
                   "top_reasons": _top(reasons)}

    # Throughput: trades first claimable per week, and cycles/events per week.
    first_claimable: Dict[str, str] = {}
    for r in rows:
        if r["stage"] == STAGE_CLAIMABLE and r["outcome"] == OUTCOME_ADVANCED:
            key = str(r["trade_id"] or r["subject"])
            day = str(r["as_of_date"])
            if key not in first_claimable or day < first_claimable[key]:
                first_claimable[key] = day
    weekly: Dict[str, Dict[str, int]] = defaultdict(lambda: {"cycles": 0, "events_first_seen": 0, "claimable": 0})
    for c in cycles:
        day = str(c["as_of_date"])
        if day <= until_iso and (not since_iso or day >= since_iso):
            weekly[_week(day)]["cycles"] += 1
    for key, v in by_event.items():
        weekly[_week(min(str(r["as_of_date"]) for r in v))]["events_first_seen"] += 1
    for day in first_claimable.values():
        if not since_iso or day >= since_iso:
            weekly[_week(day)]["claimable"] += 1

    claimable_total = len(first_claimable)
    window_start = _week((as_of - timedelta(weeks=trailing_weeks - 1)).isoformat())
    recent = sum(1 for day in first_claimable.values() if _week(day) >= window_start)
    per_week = recent / trailing_weeks
    remaining = max(0, gate_sample - claimable_total)
    weeks_to_gate = 0 if remaining == 0 else (None if per_week == 0 else math.ceil(remaining / per_week))

    return {
        "as_of": until_iso,
        "since": since_iso,
        "events": {
            "total": len(finals),
            "by_final": {k: by_final.get(k, 0) for k in (FINAL_CLAIMABLE, FINAL_IN_FLIGHT, FINAL_NO_FURTHER,
                                                       FINAL_STOPPED, FINAL_DEFERRED, FINAL_DEFERRED_LOST)},
            "stopped_by_stage": {stage: {"events": sum(c.values()), "top_reasons": _top(c)}
                                 for stage, c in sorted(stopped_by_stage.items(), key=lambda kv: STAGES.index(kv[0]))},
            "deferred_then_lost_by_reason": _top(lost_by_reason),
            "in_flight": dict(sorted(in_flight_by_stage.items())),
            "mean_analysis_passes": (round(sum(f["passes"] for f in finals.values()) / len(finals), 2)
                                     if finals else None),
        },
        "binding_constraint": binding,
        "weekly": [{"week": week, **counts} for week, counts in sorted(weekly.items())],
        "throughput": {
            "claimable_trades_total": claimable_total,
            "trailing_weeks": trailing_weeks,
            "claimable_per_week": round(per_week, 3),
            "gate_sample": gate_sample,
            "remaining_to_gate": remaining,
            # None: nothing became claimable in the trailing window, so at the
            # current rate the gate is never reached.
            "weeks_to_gate_at_current_rate": weeks_to_gate,
        },
    }


def render_text(d: Mapping[str, Any]) -> str:
    ev, tp, b = d["events"], d["throughput"], d["binding_constraint"]
    lines = [f"Evidence funnel diagnosis as of {d['as_of']}" + (f" (events first seen since {d['since']})" if d["since"] else ""),
             "", f"Earnings events seen: {ev['total']}"]
    for name, n in ev["by_final"].items():
        lines.append(f"  {name:<20} {n}")
    lines.append(f"  mean analysis passes per event: {ev['mean_analysis_passes']}")
    lines.append("")
    if b:
        lines.append(f"Binding constraint: {b['stage']} ends {b['events_lost']} events ({b['share_of_events']:.0%})")
        for item in b["top_reasons"]:
            lines.append(f"  {item['events']:>5}  {item['reason']}")
    else:
        lines.append("Binding constraint: none (no event has stopped yet)")
    lines.append("")
    lines.append("Stopped events by stage:")
    for stage, info in ev["stopped_by_stage"].items():
        top = ", ".join(f"{i['reason']} ({i['events']})" for i in info["top_reasons"][:3])
        lines.append(f"  {stage:<15} {info['events']:>5}  {top}")
    if ev["deferred_then_lost_by_reason"]:
        lines.append("Deferred and never picked up again: "
                     + ", ".join(f"{i['reason']} ({i['events']})" for i in ev["deferred_then_lost_by_reason"]))
    lines.append("")
    eta = tp["weeks_to_gate_at_current_rate"]
    lines.append(f"Claimable trades so far: {tp['claimable_trades_total']} of {tp['gate_sample']} for the gate; "
                 f"{tp['claimable_per_week']}/week over the last {tp['trailing_weeks']} weeks -> "
                 + ("gate sample reached" if eta == 0 else
                    "never at this rate" if eta is None else f"about {eta} more weeks"))
    lines.append("")
    lines.append("Week        cycles  new events  claimable")
    for w in d["weekly"]:
        lines.append(f"{w['week']}  {w['cycles']:>6}  {w['events_first_seen']:>10}  {w['claimable']:>9}")
    return "\n".join(lines)
