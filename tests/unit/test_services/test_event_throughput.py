"""Throughput (F4): one ledger identity per event, and a re-check schedule.

Before this:
- the ledger salted every analysis into its own id, so an event analyzed on
  ten days was ten unrelated rows, and the candidate-exit resolver resolved
  each as a separate sample of the same event;
- a too-early candidate was only looked at again if discovery happened to
  surface it on a later day;
- a failed entry quote dropped the candidate even with days left in its entry
  window, and a later success carried no record of the failed attempts.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import scripts.run_forward_loop as forward_loop
from services import evidence_funnel as ef
from services.baseline_evidence_store import BaselineEvidenceStore
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger, build_record_from_analysis, make_event_key

TODAY = date(2026, 4, 21)


# ── ledger: one identity per event ──────────────────────────────────────────


def _record(symbol, earnings, as_of, *, recommendation="Candidate", structure="call_calendar",
            provenance="forward_post_freeze"):
    analysis = SimpleNamespace(
        symbol=symbol, recommendation=recommendation, metrics={},
        selector_output={"recommendation": recommendation, "best_structure": structure,
                         "earnings_date": earnings},
        structure_scorecards=[], rationale=[], sample_provenance=provenance,
        vol_snapshot={"symbol": symbol, "as_of_date": as_of, "earnings_date": earnings},
    )
    return build_record_from_analysis(analysis)


@pytest.fixture
def ledger(tmp_path):
    store = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


def _events(ledger):
    rows = ledger._conn.execute(
        "SELECT symbol, as_of_date, event_key, event_sequence FROM recommendations ORDER BY created_at"
    ).fetchall()
    return [tuple(row) for row in rows]


def test_analyses_of_one_event_share_its_identity_in_order(ledger):
    ledger.record(_record("aapl", "2026-05-01", "2026-04-20"))
    ledger.record(_record("AAPL", "2026-05-01", "2026-04-21"))
    ledger.record(_record("AAPL", "2026-07-30", "2026-04-21"))      # next quarter: another event
    ledger.record(_record("MSFT", None, "2026-04-21"))              # no date: no event identity
    assert _events(ledger) == [
        ("AAPL", "2026-04-20", "AAPL|2026-05-01", 1),
        ("AAPL", "2026-04-21", "AAPL|2026-05-01", 2),
        ("AAPL", "2026-04-21", "AAPL|2026-07-30", 1),
        ("MSFT", "2026-04-21", None, None),
    ]
    assert make_event_key(" aapl ", date(2026, 5, 1)) == "AAPL|2026-05-01"


def test_rows_written_before_event_identity_are_backfilled_in_creation_order(tmp_path):
    path = tmp_path / "ledger.sqlite"
    first = RecommendationLedger(ledger_path=path)
    for as_of in ("2026-04-20", "2026-04-21", "2026-04-22"):
        first.record(_record("AAPL", "2026-05-01", as_of))
    first.record(_record("MSFT", "2026-05-02", "2026-04-21"))
    first._conn.execute("UPDATE recommendations SET event_key = NULL, event_sequence = NULL")
    first._conn.commit()
    first.close()

    reopened = RecommendationLedger(ledger_path=path)
    try:
        assert [row[2:] for row in _events(reopened)] == [
            ("AAPL|2026-05-01", 1), ("AAPL|2026-05-01", 2), ("AAPL|2026-05-01", 3), ("MSFT|2026-05-02", 1)]
        reopened.record(_record("AAPL", "2026-05-01", "2026-04-23"))
        assert _events(reopened)[-1][2:] == ("AAPL|2026-05-01", 4)
    finally:
        reopened.close()


def test_the_resolver_takes_one_decision_per_event(ledger):
    ledger.record(_record("AAPL", "2026-04-01", "2026-03-20"))
    ledger.record(_record("AAPL", "2026-04-01", "2026-03-25"))
    ledger.record(_record("MSFT", "2026-04-01", "2026-03-25"))
    # A later replay row for the same event never displaces the forward decision.
    ledger.record(_record("MSFT", "2026-04-01", "2026-03-31", provenance="historical_replay_in_sample_or_research"))
    pending = ledger.list_pending_candidate_exit_resolutions(now=date(2026, 5, 1))
    assert sorted((row["symbol"], row["as_of_date"]) for row in pending) == [
        ("AAPL", "2026-03-25"), ("MSFT", "2026-03-25")]


def test_summaries_count_events_by_their_latest_decision(tmp_path, ledger):
    ledger.record(_record("AAPL", "2026-05-01", "2026-04-17", recommendation="No Trade"))
    ledger.record(_record("AAPL", "2026-05-01", "2026-04-20"))
    ledger.record(_record("MSFT", "2026-05-02", "2026-04-20", recommendation="No Trade"))
    summary = ledger.summarize()
    assert summary["total"] == 3
    assert summary["events"] == {"total": 2, "by_recommendation": {"Candidate": 1, "No Trade": 1},
                                 "analyses_without_event": 0}
    diagnostics = build_forward_performance_diagnostics(
        ledger=ledger, outcome_store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )
    assert (diagnostics["total_recommendations"], diagnostics["no_trade_count"]) == (3, 2)
    assert (diagnostics["total_events"], diagnostics["no_trade_event_count"]) == (2, 1)


# ── the re-check schedule ───────────────────────────────────────────────────


@pytest.fixture
def funnel(tmp_path):
    store = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    yield store
    store.close()


def _analysis(symbol, earnings):
    return SimpleNamespace(
        symbol=symbol, recommendation="Candidate", confidence_pct=70.0, setup_score=0.6, metrics={},
        rationale=["ok"],
        selector_output={"recommendation": "Candidate", "best_structure": "atm_straddle",
                         "earnings_date": earnings.isoformat(), "binding_gate": None},
        structure_scorecards=[{"structure": "atm_straddle", "execution_penalty": 0.03}],
        vol_snapshot={"as_of_date": TODAY.isoformat(), "earnings_date": earnings.isoformat(),
                      "data_quality_score": 0.9},
    )


def _cycle(tmp_path, funnel, today, rows, *, earnings, quotes=None, broken=(), dry_run=False, cycle="c"):
    analyzed = []

    def analyzer(symbol, mda_client=None):  # noqa: ARG001
        analyzed.append(symbol)
        if symbol in broken:
            raise ConnectionError("provider down")
        return _analysis(symbol, earnings[symbol])

    def price_fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
        mid = (quotes or {}).get(symbol)
        return {"mid": None, "reason": "no_spot_price"} if mid is None else {"mid": mid, "context": {}}

    summary = forward_loop.run_forward_screener(
        today=today, dry_run=dry_run, store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        ledger=RecommendationLedger(ledger_path=tmp_path / "r.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": rows}, analyzer=analyzer, price_fetcher=price_fetcher,
        funnel=funnel, cycle_run_id=cycle, symbols=["A"],
    )
    return summary, analyzed


def _event(funnel, cycle, symbol, stage):
    (event,) = [e for e in funnel.events(cycle) if e["symbol"] == symbol and e["stage"] == stage]
    return event["outcome"], event["reason"], json.loads(event["detail_json"] or "{}")


def test_a_too_early_event_is_analyzed_on_its_recheck_day_even_if_discovery_misses_it(tmp_path, funnel):
    earnings = TODAY + timedelta(days=20)
    row = {"symbol": "EARLY", "dte": 20, "earnings_date": earnings.isoformat()}
    _cycle(tmp_path, funnel, TODAY, [row], earnings={"EARLY": earnings}, cycle="c1")
    _cycle(tmp_path, funnel, TODAY + timedelta(days=1), [{**row, "dte": 19}], earnings={"EARLY": earnings}, cycle="c2")
    (pending,) = funnel.pending_rechecks()
    assert pending["recheck_on"] == (earnings - timedelta(days=14)).isoformat()
    assert len(pending["attempts"]) == 1                     # seen again: not a new attempt

    # Not yet due: nothing is added.
    _, analyzed = _cycle(tmp_path, funnel, TODAY + timedelta(days=3), [], earnings={"EARLY": earnings}, cycle="c3")
    assert analyzed == []
    summary, analyzed = _cycle(tmp_path, funnel, earnings - timedelta(days=14), [],
                               earnings={"EARLY": earnings}, quotes={"EARLY": 4.0}, cycle="c4")
    assert analyzed == ["EARLY"] and summary["rechecks_added"] == 1 and summary["entries"] == 1
    outcome, _, detail = _event(funnel, "c4", "EARLY", "analyzed")
    assert outcome == "advanced" and detail["prior_attempts"][0]["reason"] == ef.REASON_TOO_EARLY
    assert funnel.pending_rechecks() == [] and funnel.recheck_counts()["done"] == 1


def test_a_failed_entry_quote_is_retried_while_the_window_lasts(tmp_path, funnel):
    earnings = TODAY + timedelta(days=6)
    row = {"symbol": "RETRY", "dte": 6, "earnings_date": earnings.isoformat()}
    summary, _ = _cycle(tmp_path, funnel, TODAY, [row], earnings={"RETRY": earnings}, cycle="c1")
    outcome, reason, detail = _event(funnel, "c1", "RETRY", "recorded")
    assert (outcome, reason) == ("deferred", "missing_entry_quote:no_spot_price")
    assert detail["recheck_on"] == (TODAY + timedelta(days=1)).isoformat()
    assert summary["entries"] == 0 and summary["rechecks_scheduled"] == 1
    # A second run the same day does not retry early.
    _, analyzed = _cycle(tmp_path, funnel, TODAY, [], earnings={"RETRY": earnings}, quotes={"RETRY": 4.0}, cycle="c1b")
    assert analyzed == []

    # Next day discovery does not list it; the re-check brings it back and it enters.
    summary, analyzed = _cycle(tmp_path, funnel, TODAY + timedelta(days=1), [], earnings={"RETRY": earnings},
                               quotes={"RETRY": 4.0}, cycle="c2")
    assert analyzed == ["RETRY"] and summary["entries"] == 1
    outcome, _, detail = _event(funnel, "c2", "RETRY", "recorded")
    assert outcome == "advanced"
    assert [a["reason"] for a in detail["prior_attempts"]] == ["missing_entry_quote:no_spot_price"]


def test_the_last_in_window_day_drops_instead_of_deferring(tmp_path, funnel):
    earnings = TODAY + timedelta(days=3)
    row = {"symbol": "LAST", "dte": 3, "earnings_date": earnings.isoformat()}
    _cycle(tmp_path, funnel, TODAY, [row], earnings={"LAST": earnings}, cycle="c1")
    assert _event(funnel, "c1", "LAST", "recorded")[:2] == ("dropped", "missing_entry_quote:no_spot_price")
    assert funnel.pending_rechecks() == []


def test_a_recheck_whose_window_closed_expires_with_a_reason(tmp_path, funnel):
    earnings = TODAY + timedelta(days=6)
    funnel.schedule_recheck(symbol="GONE", earnings_date=earnings, recheck_on=TODAY - timedelta(days=5),
                            reason="missing_entry_quote:no_spot_price", as_of=TODAY - timedelta(days=6))
    summary, analyzed = _cycle(tmp_path, funnel, TODAY + timedelta(days=4), [], earnings={}, cycle="c1")
    assert analyzed == [] and summary["rechecks_expired"] == 1
    outcome, reason, detail = _event(funnel, "c1", "GONE", "recorded")
    assert (outcome, reason) == ("dropped", "recheck_expired:missing_entry_quote:no_spot_price")
    assert len(detail["attempts"]) == 1
    assert funnel.recheck_counts() == {"pending": 0, "done": 0, "expired": 1}


def test_a_failed_analysis_is_retried_next_cycle(tmp_path, funnel):
    earnings = TODAY + timedelta(days=6)
    row = {"symbol": "FLAKY", "dte": 6, "earnings_date": earnings.isoformat()}
    _cycle(tmp_path, funnel, TODAY, [row], earnings={"FLAKY": earnings}, broken={"FLAKY"}, cycle="c1")
    assert _event(funnel, "c1", "FLAKY", "analyzed")[:2] == ("error", "analyzer_error:ConnectionError")
    (pending,) = funnel.pending_rechecks()
    assert pending["recheck_on"] == (TODAY + timedelta(days=1)).isoformat()


def test_a_too_late_event_closes_its_recheck(tmp_path, funnel):
    earnings = TODAY + timedelta(days=2)
    funnel.schedule_recheck(symbol="LATE", earnings_date=earnings, recheck_on=TODAY, reason="x", as_of=TODAY)
    _cycle(tmp_path, funnel, TODAY, [{"symbol": "LATE", "dte": 2, "earnings_date": earnings.isoformat()}],
           earnings={}, cycle="c1")
    assert funnel.recheck_counts()["expired"] == 1
    late = [e for e in funnel.events("c1") if e["symbol"] == "LATE" and e["outcome"] == "dropped"]
    assert [e["reason"] for e in late] == [ef.REASON_TOO_LATE]          # dropped once, not twice


def test_a_dry_run_schedules_nothing(tmp_path, funnel):
    earnings = TODAY + timedelta(days=20)
    _cycle(tmp_path, funnel, TODAY, [{"symbol": "EARLY", "dte": 20, "earnings_date": earnings.isoformat()}],
           earnings={}, dry_run=True)
    assert funnel.pending_rechecks() == []


def test_the_funnel_report_counts_rechecks(funnel):
    funnel.schedule_recheck(symbol="A", earnings_date="2026-05-01", recheck_on=TODAY, reason="x", as_of=TODAY)
    report = ef.build_evidence_funnel_report(as_of=TODAY, store=funnel)
    assert report["rechecks"] == {"pending": 1, "done": 0, "expired": 0}
