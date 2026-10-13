"""Evidence funnel (F1): where each forward candidate stopped, and why.

Before this, the forward loop's drop points left no per-candidate trace: the
3-14 DTE filter dropped rows silently (so "23 discovered, 0 analyzed" had no
explanation), one analyzer exception aborted the whole entry pass, and the
claimable stage could not say why nothing was claimable.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_forward_loop as forward_loop
from scripts.run_evidence_cycle import _run_log_payload
from scripts.watch_daily_evidence_cycle import _build_combined_watchdog_status
from services import evidence_funnel as ef
from services.baseline_evidence_store import BaselineEvidenceStore
from services.outcome_recorder import OutcomeStore, make_trade_id
from services.recommendation_ledger import RecommendationLedger

TODAY = date(2026, 4, 21)


@pytest.fixture
def funnel(tmp_path):
    store = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    yield store
    store.close()


# ── the store ────────────────────────────────────────────────────────────────


def test_events_need_a_known_stage_outcome_and_a_reason_to_stop(funnel):
    base = dict(cycle_run_id="c1", as_of=TODAY, symbol="AAPL")
    with pytest.raises(ValueError):
        funnel.record(stage="bogus", outcome=ef.OUTCOME_ADVANCED, **base)
    with pytest.raises(ValueError):
        funnel.record(stage=ef.STAGE_ANALYZED, outcome="bogus", **base)
    with pytest.raises(ValueError):
        funnel.record(stage=ef.STAGE_ANALYZED, outcome=ef.OUTCOME_DROPPED, **base)
    ef.safe_record(funnel, stage="bogus", outcome=ef.OUTCOME_ADVANCED, **base)  # never raises
    assert funnel.events() == []


def test_summary_counts_stages_reasons_and_conversion(funnel):
    funnel.start_cycle(cycle_run_id="c1", as_of=TODAY, universe_size=50, discovered=3, discovery_source="x")
    for symbol in ("A", "B", "C"):
        funnel.record(cycle_run_id="c1", as_of=TODAY, stage=ef.STAGE_DISCOVERED,
                      outcome=ef.OUTCOME_ADVANCED, symbol=symbol)
    funnel.record(cycle_run_id="c1", as_of=TODAY, stage=ef.STAGE_DATA_QUALIFIED,
                  outcome=ef.OUTCOME_ADVANCED, symbol="A")
    for symbol in ("B", "C"):
        funnel.record(cycle_run_id="c1", as_of=TODAY, stage=ef.STAGE_DATA_QUALIFIED,
                      outcome=ef.OUTCOME_DEFERRED, symbol=symbol, reason=ef.REASON_TOO_EARLY)
    summary = funnel.summarize(since=TODAY, until=TODAY)
    assert summary["cycles"] == 1 and summary["symbols_scanned"] == 50 and summary["discovered"] == 3
    assert summary["stages"]["data_qualified"]["deferred"] == 2
    assert summary["drop_reasons"]["data_qualified"] == {ef.REASON_TOO_EARLY: 2}
    assert summary["conversion"]["discovered->data_qualified"] == 0.3333
    assert summary["conversion"]["analyzed->eligible"] is None
    assert funnel.summarize(since=TODAY + timedelta(days=1))["cycles"] == 0


def test_stalled_cycles_are_counted_from_the_latest(funnel):
    for day in range(4):
        cycle = f"c{day}"
        funnel.start_cycle(cycle_run_id=cycle, as_of=TODAY + timedelta(days=day), universe_size=10,
                           discovered=2, discovery_source="x")
        if day == 0:
            funnel.record(cycle_run_id=cycle, as_of=TODAY, stage=ef.STAGE_ANALYZED,
                          outcome=ef.OUTCOME_ADVANCED, symbol="A")
    assert funnel.cycles_without_analysis() == 3
    report = ef.build_evidence_funnel_report(as_of=TODAY + timedelta(days=3), store=funnel)
    assert report["cycles_without_analysis"] == 3 and report["warnings"] == ["entry_pipeline_stalled"]
    health = ef.build_evidence_funnel_health(store=funnel)
    assert health["issues"] and health["issues"][0]["alertable"]

    # An analyzing cycle ends the streak: only the cycles after it count.
    funnel.record(cycle_run_id="c2", as_of=TODAY + timedelta(days=2), stage=ef.STAGE_ANALYZED,
                  outcome=ef.OUTCOME_ADVANCED, symbol="A")
    assert funnel.cycles_without_analysis() == 1
    assert ef.build_evidence_funnel_health(store=funnel)["issues"] == []


def test_watchdog_alerts_on_a_stalled_pipeline():
    funnel_health = {"summary": {"cycles_without_analysis": 3},
                     "issues": [{"severity": "WARN", "alertable": True, "message": "Entry pipeline stalled"}]}
    combined = _build_combined_watchdog_status(
        watchdog_status={"ok": True, "errors": [], "warnings": []},
        resolver_health={"issues": [], "summary": {}},
        job_freshness={"issues": [], "summary": {}},
        funnel_health=funnel_health,
    )
    assert combined["ok"] is False and "Entry pipeline stalled" in combined["errors"]
    assert combined["evidence_funnel"] == {"cycles_without_analysis": 3}


# ── the forward loop's entry stages ─────────────────────────────────────────


def _analysis(symbol, recommendation="Candidate", structure="atm_straddle", gate=None):
    earnings = TODAY + timedelta(days=6)
    return SimpleNamespace(
        symbol=symbol, recommendation=recommendation, confidence_pct=70.0, setup_score=0.6,
        metrics={}, rationale=["ok"],
        selector_output={"recommendation": recommendation, "best_structure": structure,
                         "earnings_date": earnings.isoformat(), "binding_gate": gate},
        structure_scorecards=[{"structure": structure, "execution_penalty": 0.03}],
        vol_snapshot={"as_of_date": TODAY.isoformat(), "earnings_date": earnings.isoformat(),
                      "days_to_earnings": 6, "data_quality_score": 0.9},
    )


def _run_entries(tmp_path, funnel, rows, analyses, quotes):
    def screener(**_):
        return {"rows": rows}

    def analyzer(symbol, mda_client=None):  # noqa: ARG001
        result = analyses[symbol]
        if isinstance(result, Exception):
            raise result
        return result

    def price_fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
        mid = quotes.get(symbol)
        return {"mid": mid, "reason": "missing_entry_mid"} if mid is None else {"mid": mid, "context": {}}

    return forward_loop.run_forward_screener(
        today=TODAY, store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        ledger=RecommendationLedger(ledger_path=tmp_path / "r.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), log_path=tmp_path / "log.jsonl",
        screener_builder=screener, analyzer=analyzer, price_fetcher=price_fetcher,
        funnel=funnel, cycle_run_id="cycle-1", symbols=["A", "B", "C", "D", "E", "F", "G"],
    )


def _by_symbol(funnel):
    out = {}
    for event in funnel.events("cycle-1"):
        out.setdefault(event["symbol"], {})[event["stage"]] = (event["outcome"], event["reason"])
    return out


def test_every_entry_drop_point_is_recorded_with_a_reason(tmp_path, funnel):
    rows = [
        {"symbol": "EARLY", "status": "ranked", "dte": 30},
        {"symbol": "LATE", "status": "ranked", "dte": 1},
        {"symbol": "BROKEN", "status": "ranked", "dte": 6},
        {"symbol": "NOTRADE", "status": "ranked", "dte": 6},
        {"symbol": "NOQUOTE", "status": "ranked", "dte": 6},
        {"symbol": "GOOD", "status": "ranked", "dte": 6},
    ]
    analyses = {
        "BROKEN": ValueError("No market data for BROKEN"),
        "NOTRADE": _analysis("NOTRADE", recommendation="No Trade", gate="no_trade_low_edge"),
        "NOQUOTE": _analysis("NOQUOTE"),
        "GOOD": _analysis("GOOD"),
    }
    summary = _run_entries(tmp_path, funnel, rows, analyses, {"GOOD": 5.0})

    # One failed analysis no longer aborts the pass: later symbols still run.
    assert summary["analyzer_errors"] == 1 and summary["analyzed"] == 3 and summary["entries"] == 1
    assert summary["outside_dte_window"] == 2 and summary["cycle_run_id"] == "cycle-1"
    events = _by_symbol(funnel)
    assert events["EARLY"]["data_qualified"] == ("deferred", ef.REASON_TOO_EARLY)
    assert events["LATE"]["data_qualified"] == ("dropped", ef.REASON_TOO_LATE)
    assert events["BROKEN"]["analyzed"] == ("error", "analyzer_error:ValueError")
    assert events["NOTRADE"]["eligible"] == ("dropped", "no_trade:no_trade_low_edge")
    # Still inside its entry window, so retried on the next cycle (F4).
    assert events["NOQUOTE"]["recorded"] == ("deferred", "missing_entry_quote:missing_entry_mid")
    assert events["GOOD"]["recorded"] == ("advanced", None)
    assert all(stage["discovered"] == ("advanced", None) for stage in events.values())

    cycle = funnel.summarize()
    assert cycle["symbols_scanned"] == 7 and cycle["discovered"] == 6


def test_a_dry_run_records_nothing(tmp_path, funnel):
    forward_loop.run_forward_screener(
        today=TODAY, dry_run=True, store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        log_path=tmp_path / "log.jsonl", screener_builder=lambda **_: {"rows": [{"symbol": "A", "dte": 30}]},
        analyzer=lambda *a, **k: None, funnel=funnel, cycle_run_id="cycle-1",
    )
    assert funnel.events() == []


# ── exit stages and claimability ────────────────────────────────────────────


def _insert_trade(store, symbol, earnings, *, claim_allowed=False, marker="m"):
    trade_id = make_trade_id(symbol, TODAY - timedelta(days=4), "atm_straddle", earnings_date=earnings)
    store.insert_entry(
        trade_id=trade_id, symbol=symbol, structure="atm_straddle", entry_date=TODAY - timedelta(days=4),
        setup_score=0.6, source_type="paper", earnings_date=earnings, entry_mid=5.0,
        execution_penalty_at_entry=0.03, notes=json.dumps({"pricing_context": {"marker": marker}}),
        claim_allowed=claim_allowed, evidence_quality_reasons=["quote_not_execution_grade"],
    )
    return trade_id


def test_exit_stages_and_the_reason_nothing_is_claimable(tmp_path, funnel):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    resolved_id = _insert_trade(store, "SHOP", TODAY + timedelta(days=1), marker="priced")
    unpriced_id = _insert_trade(store, "CRM", TODAY + timedelta(days=1), marker="unpriced")
    missing_id = _insert_trade(store, "OLD", TODAY - timedelta(days=1))

    def price_fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
        if (context or {}).get("marker") == "priced":
            return {"mid": 5.5, "context": context}
        return {"mid": None, "reason": "missing_exit_mid"}

    def finalizer(**kwargs):  # noqa: ARG001
        return {"status": "finalized"}

    forward_loop.run_exit_detection(
        today=TODAY, store=store, log_path=tmp_path / "log.jsonl", price_fetcher=price_fetcher,
        finalizer=finalizer, baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
        funnel=funnel, cycle_run_id="cycle-1",
    )
    events = {(e["trade_id"], e["stage"]): (e["outcome"], e["reason"]) for e in funnel.events("cycle-1")}
    assert events[(resolved_id, "exit_due")] == ("advanced", None)
    assert events[(resolved_id, "resolved")] == ("advanced", None)
    assert events[(resolved_id, "claimable")] == ("dropped", "not_claimable:quote_not_execution_grade")
    assert events[(unpriced_id, "resolved")] == ("deferred", "exit_failed:missing_exit_mid")
    assert events[(missing_id, "resolved")][0] == "dropped"
    assert events[(missing_id, "resolved")][1].startswith("exit_missing:")


def test_a_claim_allowed_resolution_is_claimable(tmp_path, funnel):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    trade_id = _insert_trade(store, "SHOP", TODAY + timedelta(days=1), claim_allowed=True, marker="priced")
    forward_loop.run_exit_detection(
        today=TODAY, store=store, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **kw: {"mid": 5.5, "context": kw.get("context")},
        finalizer=lambda **kw: {"status": "finalized"},
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), funnel=funnel, cycle_run_id="cycle-1",
    )
    claim = [e for e in funnel.events("cycle-1") if e["stage"] == "claimable"]
    assert [(e["trade_id"], e["outcome"]) for e in claim] == [(trade_id, "advanced")]


# ── the daily run log carries the funnel counts ─────────────────────────────


def test_run_log_reports_discovery_and_analysis_counts():
    from datetime import datetime, timezone

    payload = {"forward_loop": {"cycle_run_id": "c1", "entries": {
        "discovered": 23, "outside_dte_window": 23, "analyzed": 0, "analyzer_errors": 0}}}
    log = _run_log_payload(started_at=datetime.now(timezone.utc), status="success", cycle_date=TODAY,
                           dry_run=False, payload=payload)
    assert (log["candidates_discovered"], log["candidates_outside_dte_window"], log["candidates_analyzed"]) == (23, 23, 0)
    assert log["cycle_run_id"] == "c1"
