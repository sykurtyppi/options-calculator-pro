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
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

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
            "SELECT as_of_date, stage, outcome, subject, symbol, earnings_date, reason, trade_id, recommendation_id"
            " FROM funnel_events")]
        cycles = [dict(row) for row in conn.execute(
            "SELECT cycle_run_id, as_of_date, universe_size, discovered FROM funnel_cycles")]
    finally:
        conn.close()
    return {"events": events, "cycles": cycles}


def read_recommendation_rows(
    ledger_path: Path,
    *,
    status: Optional[MutableMapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Recommendation details used to explain selector stops, read-only.

    The recommendation ledger is optional because the primary funnel report
    remains useful on installations that have not created one yet.
    """
    path = Path(ledger_path)
    if not path.exists():
        if status is not None:
            status.update({"status": "missing", "rows": 0, "error": None})
        return []
    conn = None
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "recommendations" not in tables:
            raise sqlite3.OperationalError("recommendations table is unavailable")
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(recommendations)")}
        required = {"symbol", "earnings_date", "as_of_date", "created_at", "binding_gate",
                    "structure_scorecards_json"}
        missing = sorted(required - columns)
        if missing:
            raise sqlite3.OperationalError(f"recommendations table is missing columns: {', '.join(missing)}")
        optional = [name for name in (
            "recommendation_id", "selected_structure", "selector_output_json"
        ) if name in columns]
        selected = ["symbol", "earnings_date", "as_of_date", "created_at", "binding_gate",
                    "structure_scorecards_json", *optional]
        rows = [dict(row) for row in conn.execute(
            f"SELECT {', '.join(selected)} FROM recommendations")]

        # The base recommendation row is immutable. Later scorecard/selector
        # versions live in recommendation_revisions, so overlay the newest
        # complete record when that optional table is available.
        if "recommendation_revisions" in tables and "recommendation_id" in columns:
            revision_columns = {str(row[1]) for row in conn.execute(
                "PRAGMA table_info(recommendation_revisions)")}
            if {"recommendation_id", "revised_at", "record_json"} <= revision_columns:
                latest_revisions: Dict[str, tuple[str, Mapping[str, Any]]] = {}
                for revision in conn.execute(
                    "SELECT recommendation_id, revised_at, record_json FROM recommendation_revisions"
                ):
                    try:
                        payload = json.loads(revision["record_json"])
                    except (TypeError, ValueError):
                        continue
                    recommendation_id = str(revision["recommendation_id"] or "")
                    marker = str(revision["revised_at"] or "")
                    if isinstance(payload, Mapping) and (
                        recommendation_id not in latest_revisions
                        or marker > latest_revisions[recommendation_id][0]
                    ):
                        latest_revisions[recommendation_id] = (marker, payload)
                for row in rows:
                    revision = latest_revisions.get(str(row.get("recommendation_id") or ""))
                    if revision is None:
                        continue
                    payload = dict(revision[1])
                    row.update(payload)
                    if "structure_scorecards" in payload:
                        row["structure_scorecards_json"] = payload["structure_scorecards"]
                    if "selector_output" in payload:
                        row["selector_output_json"] = payload["selector_output"]
        rows = [row for row in rows if row.get("binding_gate") in {
            "no_eligible_structure", "non_positive_edge",
        }]
        if status is not None:
            status.update({"status": "available", "rows": len(rows), "error": None})
        return rows
    except (OSError, sqlite3.Error) as exc:
        if status is not None:
            status.update({"status": "unavailable", "rows": 0, "error": str(exc)})
        return []
    finally:
        if conn is not None:
            conn.close()


def _week(day: str) -> str:
    d = date.fromisoformat(day[:10])
    return (d - timedelta(days=d.weekday())).isoformat()


def _event_final(rows: List[Mapping[str, Any]], earnings_date: Optional[str], as_of: date) -> Dict[str, Any]:
    deepest = max(STAGES.index(r["stage"]) for r in rows)
    stage = STAGES[deepest]
    at_stage = [r for r in rows if r["stage"] == stage]
    best = max(at_stage, key=lambda r: (_RANK.get(r["outcome"], 0), str(r["as_of_date"])))
    passed = earnings_date is not None and earnings_date[:10] < as_of.isoformat()
    final_row = best
    if best["outcome"] == OUTCOME_ADVANCED:
        if stage == STAGE_CLAIMABLE:
            final, reason = FINAL_CLAIMABLE, None
        else:
            final, reason = (FINAL_NO_FURTHER if passed else FINAL_IN_FLIGHT), None
    elif best["outcome"] == OUTCOME_DEFERRED:
        final_row = max(at_stage, key=lambda r: str(r["as_of_date"]))
        final, reason = (FINAL_DEFERRED_LOST if passed else FINAL_DEFERRED), final_row["reason"]
    else:
        final, reason = FINAL_STOPPED, best["reason"]
    # Cycles on which the analyser ran for this event (it re-runs while the
    # event stays in the entry window without a trade).
    passes = len({r["as_of_date"] for r in rows if r["stage"] == STAGE_ANALYZED})
    last_recorded = max(str(row["as_of_date"]) for row in rows)
    return {"stage": stage, "final": final, "reason": reason, "passes": passes,
            "as_of_date": str(final_row["as_of_date"]), "last_recorded_as_of": last_recorded,
            "recommendation_id": final_row.get("recommendation_id")}


def _top(counter: Counter, n: int = 5) -> List[Dict[str, Any]]:
    return [{"reason": reason, "events": count}
            for reason, count in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]]


def _scorecards(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    value = row.get("structure_scorecards_json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    return value if isinstance(value, list) else []


_ELIGIBILITY_FLAG_CATEGORIES = {
    "missing_earnings_date": {"event_metadata"},
    "missing_option_chain": {"option_chain"},
    "cannot_form_atm_straddle": {"missing_legs"},
    "cannot_form_iron_condor": {"missing_legs"},
    "cannot_form_otm_strangle": {"missing_legs"},
    "cannot_form_calendar": {"missing_legs"},
    "invalid_near_term_dte": {"dte"},
    "spread_exceeds_absolute_threshold": {"spread_or_liquidity"},
    "structure_spread_exceeds_absolute_threshold": {"spread_or_liquidity"},
    "surface_quality_record_only": {"surface_quality"},
    "missing_structure_leg_spread": {"missing_legs"},
    # Not emitted by services/structure_scorecard.py today; kept so older or
    # synthetic scorecards classify instead of landing in "unclassified".
    "insufficient_open_interest": {"spread_or_liquidity"},
    "insufficient_volume": {"spread_or_liquidity"},
    "spread_too_wide": {"spread_or_liquidity"},
}


def _eligibility_categories(flag: str) -> set[str]:
    return _ELIGIBILITY_FLAG_CATEGORIES.get(flag.lower(), {"unclassified"})


def _recommendation_index(recommendations: Iterable[Mapping[str, Any]]) -> Dict[tuple, Mapping[str, Any]]:
    latest: Dict[tuple, Mapping[str, Any]] = {}
    for row in recommendations:
        key = (str(row.get("symbol") or "").upper(), str(row.get("earnings_date") or "")[:10],
               str(row.get("as_of_date") or "")[:10], str(row.get("binding_gate") or ""))
        if not all(key):
            continue
        if key not in latest or str(row.get("created_at") or "") > str(latest[key].get("created_at") or ""):
            latest[key] = row
    return latest


def _binding_gate(reason: Any) -> str:
    value = str(reason or "")
    return value.removeprefix("no_trade:")


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _selected_eligible_card(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    cards = [card for card in _scorecards(row) if card.get("eligible") is True]
    selected = str(row.get("selected_structure") or "")
    selector_output = row.get("selector_output_json")
    if isinstance(selector_output, str):
        try:
            selector_output = json.loads(selector_output)
        except (TypeError, ValueError):
            selector_output = None
    if not selected and isinstance(selector_output, Mapping):
        selected = str(selector_output.get("best_structure") or "")
    if selected:
        for card in cards:
            if str(card.get("structure") or "") == selected:
                return card

    # Compatibility fallback for older rows that did not persist the selected
    # structure: reproduce structure_selector._ranking_key exactly.
    ranked = []
    for card in cards:
        composite = card.get("composite_structure_score")
        if not isinstance(composite, (int, float)) or not math.isfinite(composite):
            continue
        edge_value = _finite_float(card.get("expected_edge_pct"))
        ranked.append((
            -float(composite),
            0 if edge_value is not None else 1,
            -edge_value if edge_value is not None else 0.0,
            str(card.get("structure") or ""),
            card,
        ))
    return min(ranked)[-1] if ranked else None


MATCH_BY_ID = "recommendation_id"
MATCH_BY_EVENT_KEY = "event_key"


def _matcher(recommendations: Iterable[Mapping[str, Any]]):
    """Find the ledger row behind a funnel stop.

    The stopping funnel row carries the recommendation_id the forward loop
    wrote to the ledger, so that is matched exactly. The UI's analysis writes
    ledger rows for the same symbol and day too, and a key match would take
    the newest of them. Only a funnel row without an id (written before ids
    were recorded) falls back to the newest row for the event key and gate.
    An id with no ledger row stays unmatched rather than borrowing another.
    """
    rows = list(recommendations)
    by_id = {str(row["recommendation_id"]): row for row in rows if row.get("recommendation_id")}
    by_key = _recommendation_index(rows)

    def match(symbol: str, earnings_date: Optional[str], final: Mapping[str, Any], gate: str):
        recommendation_id = final.get("recommendation_id")
        if recommendation_id:
            row = by_id.get(str(recommendation_id))
            if row is None or str(row.get("binding_gate") or "") != gate:
                return None, MATCH_BY_ID
            return row, MATCH_BY_ID
        return by_key.get((symbol, earnings_date or "", final["as_of_date"], gate)), MATCH_BY_EVENT_KEY

    return match


def _stop_reason_details(
    finals: Mapping[tuple, Mapping[str, Any]],
    recommendations: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    match = _matcher(recommendations)
    stopped = [(key, final) for key, final in finals.items() if final["final"] == FINAL_STOPPED]
    no_eligible = [(key, final) for key, final in stopped
                   if _binding_gate(final["reason"]) == "no_eligible_structure"]
    flag_counts: Counter = Counter()
    category_counts: Counter = Counter()
    matched = parsed = usable = 0
    no_eligible_events = []
    for (symbol, earnings_date), final in no_eligible:
        row, method = match(symbol, earnings_date, final, "no_eligible_structure")
        if row is None:
            no_eligible_events.append({
                "symbol": symbol, "earnings_date": earnings_date, "as_of_date": final["as_of_date"],
                "match_status": "unmatched", "match_method": method,
                "eligibility_flags": [], "eligibility_categories": [],
            })
            continue
        matched += 1
        scorecards = _scorecards(row)
        if scorecards:
            parsed += 1
        flags = {str(flag) for card in scorecards
                 for flag in (card.get("eligibility_flags") or [])}
        categories = {category for flag in flags for category in _eligibility_categories(flag)}
        if flags:
            usable += 1
        flag_counts.update(flags)
        category_counts.update(categories)
        no_eligible_events.append({
            "symbol": symbol, "earnings_date": earnings_date, "as_of_date": final["as_of_date"],
            "match_status": "usable" if flags else ("no_flags" if scorecards else "no_scorecards"),
            "match_method": method,
            "eligibility_flags": sorted(flags), "eligibility_categories": sorted(categories),
        })

    non_positive = [(key, final) for key, final in stopped
                    if _binding_gate(final["reason"]) == "non_positive_edge"]
    edge_events = []
    for (symbol, earnings_date), final in non_positive:
        row, method = match(symbol, earnings_date, final, "non_positive_edge")
        if row is None:
            edge_events.append({
                "symbol": symbol, "earnings_date": earnings_date, "as_of_date": final["as_of_date"],
                "match_status": "unmatched", "match_method": method,
                "best_structure": None, "best_eligible_edge_pct": None,
            })
            continue
        card = _selected_eligible_card(row)
        edge_value = _finite_float(card.get("expected_edge_pct")) if card is not None else None
        edge_events.append({
            "symbol": symbol,
            "earnings_date": earnings_date,
            "as_of_date": final["as_of_date"],
            "match_status": "usable" if edge_value is not None else "no_selected_edge",
            "match_method": method,
            "best_structure": str(card.get("structure") or "unknown") if card is not None else None,
            "best_eligible_edge_pct": edge_value,
        })
    edges = [item["best_eligible_edge_pct"] for item in edge_events
             if item["best_eligible_edge_pct"] is not None]

    return {
        "no_eligible_structure": {
            "events": len(no_eligible),
            "matched_recommendations": matched,
            "scorecards_parsed": parsed,
            "events_with_eligibility_causes": usable,
            "eligibility_categories": [
                {"category": name, "events": count}
                for name, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "eligibility_flags": [
                {"flag": name, "events": count}
                for name, count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "event_details": sorted(no_eligible_events,
                                    key=lambda item: (item["earnings_date"] or "", item["symbol"])),
        },
        "non_positive_edge": {
            "events": len(non_positive),
            "matched_recommendations": sum(item["match_status"] != "unmatched" for item in edge_events),
            "events_with_selected_edge": len(edges),
            "best_eligible_edge_pct": ({
                "events": len(edges),
                "min": min(edges),
                "median": statistics.median(edges),
                "max": max(edges),
            } if edges else {"events": 0, "min": None, "median": None, "max": None}),
            "event_details": sorted(edge_events, key=lambda item: (item["earnings_date"] or "", item["symbol"])),
        },
    }


def diagnose_funnel(
    events: Iterable[Mapping[str, Any]],
    cycles: Iterable[Mapping[str, Any]],
    *,
    as_of: date,
    since: Optional[date] = None,
    trailing_weeks: int = DEFAULT_TRAILING_WEEKS,
    gate_sample: int = GATE_SAMPLE,
    recommendations: Iterable[Mapping[str, Any]] = (),
    recommendation_status: Optional[Mapping[str, Any]] = None,
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

    def event_details(final_state: str) -> List[Dict[str, Any]]:
        details = [{
            "symbol": key[0],
            "earnings_date": key[1],
            "stage": final["stage"],
            "reason": final["reason"],
            "last_recorded_as_of": final["last_recorded_as_of"],
            "analysis_passes": final["passes"],
        } for key, final in finals.items() if final["final"] == final_state]
        return sorted(details, key=lambda item: (item["earnings_date"] or "", item["symbol"]))

    return {
        "as_of": until_iso,
        "since": since_iso,
        "recommendation_ledger": dict(recommendation_status or {
            "status": "not_supplied", "rows": 0, "error": None,
        }),
        "events": {
            "total": len(finals),
            "by_final": {k: by_final.get(k, 0) for k in (FINAL_CLAIMABLE, FINAL_IN_FLIGHT, FINAL_NO_FURTHER,
                                                       FINAL_STOPPED, FINAL_DEFERRED, FINAL_DEFERRED_LOST)},
            "stopped_by_stage": {stage: {"events": sum(c.values()), "top_reasons": _top(c)}
                                 for stage, c in sorted(stopped_by_stage.items(), key=lambda kv: STAGES.index(kv[0]))},
            "deferred_then_lost_by_reason": _top(lost_by_reason),
            "deferred_then_lost_events": event_details(FINAL_DEFERRED_LOST),
            "no_further_record_events": event_details(FINAL_NO_FURTHER),
            "in_flight": dict(sorted(in_flight_by_stage.items())),
            "stop_reason_details": _stop_reason_details(finals, recommendations),
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
    details = ev["stop_reason_details"]
    no_eligible = details["no_eligible_structure"]
    if no_eligible["events"]:
        categories = ", ".join(
            f"{item['category']} ({item['events']})" for item in no_eligible["eligibility_categories"])
        lines.append(f"Eligibility causes ({no_eligible['events_with_eligibility_causes']}/"
                     f"{no_eligible['events']} usable; {no_eligible['matched_recommendations']} ledger matches): "
                     + (categories or "no scorecard details available"))
    edge = details["non_positive_edge"]
    edge_summary = edge["best_eligible_edge_pct"]
    if edge["events"]:
        if edge_summary["events"]:
            lines.append(f"Selected eligible edge at non-positive stops "
                         f"({edge['events_with_selected_edge']}/{edge['events']} usable): "
                         f"median {edge_summary['median']:.2f}%, range {edge_summary['min']:.2f}% "
                         f"to {edge_summary['max']:.2f}%")
        else:
            lines.append(f"Selected eligible edge at non-positive stops (0/{edge['events']} usable): "
                         "no usable scorecard details")
    ledger = d.get("recommendation_ledger", {})
    if ledger.get("status") == "missing":
        lines.append("Recommendation enrichment missing: optional ledger was not found")
    elif ledger.get("status") == "unavailable":
        lines.append(f"Recommendation enrichment unavailable: {ledger.get('error') or 'unknown read error'}")
    if ev["deferred_then_lost_by_reason"]:
        lines.append("Deferred and never picked up again: "
                     + ", ".join(f"{i['reason']} ({i['events']})" for i in ev["deferred_then_lost_by_reason"]))
    for item in ev["deferred_then_lost_events"]:
        lines.append(f"Deferred/lost event: {item['symbol']} {item['earnings_date']} at {item['stage']} "
                     f"({item['reason']}; last {item['last_recorded_as_of']})")
    for item in ev["no_further_record_events"]:
        lines.append(f"No-further-record event: {item['symbol']} {item['earnings_date']} after {item['stage']} "
                     f"(last {item['last_recorded_as_of']})")
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
