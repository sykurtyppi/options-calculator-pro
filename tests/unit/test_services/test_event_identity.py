"""PR O: one corporate event keeps one identity, and evidence follows it.

Regression tests for the audit of main at 469c6ff:
- a chain of 21-day moves (04-28 -> 05-19 -> 06-09 -> 04-28) left two
  current events and two pending re-checks (supersession compared pairs of
  dates only);
- after a date moved, the outcome and baseline booked for the old date still
  counted in performance and benchmark statistics while event counts said
  there was one event.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import scripts.run_forward_loop as forward_loop
from services import evidence_funnel as ef
from services.baseline_evidence_store import EXIT_REPRICING_BOOKED, BaselineEvidenceStore
from services.event_reconciliation import SUPERSEDED_REASON, drop_superseded, reconcile_moved_events
from services.event_resolution import load_event_resolutions
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger, build_record_from_analysis

CHAIN = ["2026-04-28", "2026-05-19", "2026-06-09", "2026-04-28"]


def _record(symbol, earnings, as_of):
    analysis = SimpleNamespace(
        symbol=symbol, recommendation="Candidate", metrics={},
        selector_output={"recommendation": "Candidate", "best_structure": "call_calendar",
                         "earnings_date": earnings},
        structure_scorecards=[], rationale=[], sample_provenance="forward_post_freeze",
        vol_snapshot={"symbol": symbol, "as_of_date": as_of, "earnings_date": earnings},
    )
    return build_record_from_analysis(analysis)


@pytest.fixture
def ledger(tmp_path):
    store = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


# ── a chain of moves is one event ───────────────────────────────────────────


def test_a_chain_of_moves_and_a_move_back_stay_one_event(ledger):
    # The audit's repro: each hop is 21 days (inside the move window), the
    # first and third are 42 apart (outside it).
    for day, earnings in enumerate(CHAIN, start=1):
        ledger.record(_record("HOP", earnings, f"2026-04-0{day}"))
    assert ledger.summarize()["events"]["total"] == 1
    assert [row["event_key"] for row in ledger.latest_event_rows()] == ["HOP|2026-04-28"]
    assert ledger.superseded_event_keys() == {"HOP|2026-05-19": "HOP|2026-04-28",
                                              "HOP|2026-06-09": "HOP|2026-04-28"}
    ids = {ledger.event_id_for(f"HOP|{d}") for d in CHAIN}
    assert ids == {"HOP|2026-04-28"}                       # the key it was first seen under


def test_a_move_back_to_a_middle_date_rejoins_its_event(ledger):
    # 04-28 -> 05-19 -> 06-09 -> 06-30 -> 05-19: the last hop is 42 days from
    # the current date, so only the key's own identity keeps it one event.
    dates = ["2026-04-28", "2026-05-19", "2026-06-09", "2026-06-30", "2026-05-19"]
    for day, earnings in enumerate(dates, start=1):
        ledger.record(_record("MID", earnings, f"2026-04-0{day}"))
    assert ledger.summarize()["events"]["total"] == 1
    assert [row["event_key"] for row in ledger.latest_event_rows()] == ["MID|2026-05-19"]


def test_a_chain_without_the_move_back_follows_the_latest_date(ledger):
    for day, earnings in enumerate(CHAIN[:3], start=1):
        ledger.record(_record("HOP", earnings, f"2026-04-0{day}"))
    assert [row["event_key"] for row in ledger.latest_event_rows()] == ["HOP|2026-06-09"]
    assert ledger.summarize()["events"]["total"] == 1


def test_distinct_quarters_stay_distinct_events(ledger):
    ledger.record(_record("QTR", "2026-04-28", "2026-04-01"))
    ledger.record(_record("QTR", "2026-07-28", "2026-04-02"))
    assert ledger.event_id_for("QTR|2026-04-28") != ledger.event_id_for("QTR|2026-07-28")
    assert ledger.summarize()["events"]["total"] == 2


def test_supersessions_written_before_event_identities_are_grouped_on_open(tmp_path):
    path = tmp_path / "ledger.sqlite"
    first = RecommendationLedger(ledger_path=path)
    first.record(_record("OLD", "2026-04-28", "2026-04-01"))
    first.record(_record("OLD", "2026-05-12", "2026-04-02"))
    first._conn.execute("DELETE FROM event_identity")       # as PR J left it
    first._conn.commit()
    first.close()
    reopened = RecommendationLedger(ledger_path=path)
    try:
        assert reopened.event_id_for("OLD|2026-05-12") == reopened.event_id_for("OLD|2026-04-28") == "OLD|2026-04-28"
        # And the chain continues from there.
        reopened.record(_record("OLD", "2026-04-28", "2026-04-03"))
        assert reopened.superseded_event_keys() == {"OLD|2026-05-12": "OLD|2026-04-28"}
    finally:
        reopened.close()


def test_a_chain_of_moves_leaves_one_pending_recheck(tmp_path):
    funnel = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    try:
        as_of = date(2026, 4, 1)
        for step, earnings in enumerate(CHAIN):
            day = as_of + timedelta(days=step)
            funnel.supersede_rechecks(symbol="HOP", earnings_date=earnings, as_of=day)
            funnel.schedule_recheck(symbol="HOP", earnings_date=earnings, recheck_on=day + timedelta(days=5),
                                    reason=ef.REASON_TOO_EARLY, as_of=day)
        pending = funnel.pending_rechecks()
        assert [row["earnings_date"] for row in pending] == ["2026-04-28"]
    finally:
        funnel.close()


def test_a_chain_date_that_already_passed_is_its_own_event(tmp_path):
    funnel = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    try:
        early = date(2026, 4, 1)
        funnel.schedule_recheck(symbol="P", earnings_date="2026-04-10", recheck_on=early, reason="x", as_of=early)
        funnel.supersede_rechecks(symbol="P", earnings_date="2026-05-01", as_of=early)     # 04-10 -> 05-01
        funnel.schedule_recheck(symbol="P", earnings_date="2026-04-10", recheck_on=early, reason="x", as_of=early)
        # Re-filed, it keeps its place on the chain ...
        assert funnel.supersede_rechecks(symbol="P", earnings_date="2026-05-01", as_of=date(2026, 4, 5))
        funnel.schedule_recheck(symbol="P", earnings_date="2026-04-10", recheck_on=early, reason="x", as_of=early)
        # ... but seen on 04-20, after 04-10 had passed, it was its own event.
        assert funnel.supersede_rechecks(symbol="P", earnings_date="2026-05-01", as_of=date(2026, 4, 20)) == []
    finally:
        funnel.close()


def test_the_chain_survives_a_refiled_recheck(tmp_path):
    # A -> B, A re-filed (A moved back), then a date 42 days from A but 21
    # from B: still the same event, through B.
    funnel = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    try:
        day = date(2026, 4, 1)
        funnel.schedule_recheck(symbol="R", earnings_date="2026-04-28", recheck_on=day, reason="x", as_of=day)
        funnel.supersede_rechecks(symbol="R", earnings_date="2026-05-19", as_of=day)
        funnel.supersede_rechecks(symbol="R", earnings_date="2026-04-28", as_of=day)
        funnel.schedule_recheck(symbol="R", earnings_date="2026-04-28", recheck_on=day, reason="y", as_of=day)
        closed = funnel.supersede_rechecks(symbol="R", earnings_date="2026-06-09", as_of=day)
        assert [row["earnings_date"] for row in closed] == ["2026-04-28"]
    finally:
        funnel.close()


def test_a_recheck_for_another_quarter_is_not_on_the_chain(tmp_path):
    funnel = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    try:
        day = date(2026, 4, 1)
        funnel.schedule_recheck(symbol="Q", earnings_date="2026-07-28", recheck_on=day, reason="x", as_of=day)
        funnel.schedule_recheck(symbol="Q", earnings_date="2026-04-28", recheck_on=day, reason="x", as_of=day)
        funnel.supersede_rechecks(symbol="Q", earnings_date="2026-05-12", as_of=day)
        assert [r["earnings_date"] for r in funnel.pending_rechecks()] == ["2026-07-28"]
    finally:
        funnel.close()


# ── evidence follows the event's current date ───────────────────────────────


def _resolved_outcome(store, symbol, earnings, trade_id, ret=12.0):
    store.insert_entry(trade_id=trade_id, symbol=symbol, structure="atm_straddle",
                       entry_date=date(2026, 4, 1), setup_score=0.6, source_type="paper",
                       earnings_date=earnings, entry_mid=4.0)
    store.update_exit(trade_id=trade_id, exit_date=date(2026, 4, 27), exit_mid=4.4, realized_return_pct=ret,
                      realized_expansion_pct=5.0)
    store.mark_finalized(trade_id)


def _baseline(store, symbol, earnings, rec):
    store.insert_entry(
        recommendation_id=rec, symbol=symbol, baseline_name="always_atm_straddle", structure="atm_straddle",
        entry_date=date(2026, 4, 1), earnings_date=earnings, selector_structure=None, entry_mid=4.0,
        modeled_cost_pct=None, execution_penalty_at_entry=None, data_quality_score_at_entry=None,
        iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None, quote_source_at_entry="marketdata_app",
        quote_quality_at_entry=None,
    )
    store.update_exit(baseline_id=f"{rec}|baseline|always_atm_straddle", exit_date=date(2026, 4, 27),
                      exit_mid=4.4, realized_return_pct=10.0, realized_expansion_pct=None,
                      quote_source_at_exit="marketdata_app", quote_quality_at_exit=None,
                      exit_repricing=EXIT_REPRICING_BOOKED)


@pytest.fixture
def moved(tmp_path, ledger):
    # MOVE was traded for 04-28, then the date moved to 05-05; KEEP did not move.
    ledger.record(_record("MOVE", "2026-04-28", "2026-04-20"))
    ledger.record(_record("MOVE", "2026-05-05", "2026-04-22"))
    outcomes = OutcomeStore(store_path=tmp_path / "o.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    _resolved_outcome(outcomes, "MOVE", date(2026, 4, 28), "T_OLD")
    _resolved_outcome(outcomes, "KEEP", date(2026, 4, 28), "T_KEEP")
    _resolved_outcome(outcomes, "MOVE", date(2026, 5, 5), "T_NEW")        # booked under the current date
    _baseline(baselines, "MOVE", date(2026, 4, 28), "rec_old")
    _baseline(baselines, "KEEP", date(2026, 4, 28), "rec_keep")
    _baseline(baselines, "MOVE", date(2026, 5, 5), "rec_new")
    return ledger, outcomes, baselines


def _diagnostics(ledger, outcomes, baselines):
    return build_forward_performance_diagnostics(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)


def test_reports_drop_old_date_evidence_even_before_reconciliation(moved):
    ledger, outcomes, baselines = moved
    report = _diagnostics(ledger, outcomes, baselines)
    assert report["resolved_outcome_count"] == 2                         # T_KEEP and T_NEW (was 3)
    assert report["benchmark_comparison"]["always_atm_straddle"]["n"] == 2   # the old-date baseline is out
    events = load_event_resolutions(as_of=date(2026, 5, 1), ledger=ledger, outcome_store=outcomes,
                                    baseline_store=baselines)
    assert "MOVE|2026-04-28" not in events and "KEEP|2026-04-28" in events


def test_reconciliation_invalidates_old_date_evidence_durably(moved, tmp_path):
    ledger, outcomes, baselines = moved
    summary = reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert (summary["outcomes_invalidated"], summary["baselines_invalidated"]) == (1, 1)
    old = outcomes.get_trade("T_OLD")
    assert old["evidence_valid"] == 0
    assert old["invalidation_reason"] == f"{SUPERSEDED_REASON}:MOVE|2026-05-05"
    assert outcomes.get_trade("T_KEEP")["evidence_valid"] is None
    visible = {row["baseline_id"] for row in baselines.list_for_diagnostics()}
    assert visible == {"rec_keep|baseline|always_atm_straddle", "rec_new|baseline|always_atm_straddle"}
    (audit,) = [r for r in baselines.list_for_diagnostics(include_invalid=True) if r["recommendation_id"] == "rec_old"]
    assert audit["invalidation_reason"] == f"{SUPERSEDED_REASON}:MOVE|2026-05-05"
    # Durable: the stores alone now say so, whatever the ledger.
    report = build_forward_performance_diagnostics(ledger=RecommendationLedger(
        ledger_path=tmp_path / "fresh.sqlite"), outcome_store=outcomes, baseline_store=baselines)
    assert report["resolved_outcome_count"] == 2
    # Idempotent.
    again = reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert (again["outcomes_invalidated"], again["baselines_invalidated"]) == (0, 0)


def test_rows_carry_their_events_durable_identity(moved):
    ledger, outcomes, baselines = moved
    reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert outcomes.get_trade("T_KEEP")["event_id"] == "KEEP|2026-04-28"
    # One event, one id, whichever date the row was booked under.
    assert outcomes.get_trade("T_OLD")["event_id"] == outcomes.get_trade("T_NEW")["event_id"] == "MOVE|2026-04-28"
    ids = {r["recommendation_id"]: r["event_id"] for r in baselines.list_for_diagnostics(include_invalid=True)}
    assert ids == {"rec_old": "MOVE|2026-04-28", "rec_new": "MOVE|2026-04-28", "rec_keep": "KEEP|2026-04-28"}


def test_learning_that_already_happened_is_reported(moved):
    ledger, outcomes, baselines = moved
    outcomes._conn.execute("UPDATE outcome_trades SET learning_update_status = 'complete' WHERE trade_id = 'T_OLD'")
    outcomes._conn.commit()
    summary = reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert summary["learning_already_applied"] == ["T_OLD"]


def test_an_invalidated_open_baseline_is_never_exited_or_marked_missing(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    store.insert_entry(
        recommendation_id="rec", symbol="MOVE", baseline_name="always_straddle", structure="atm_straddle",
        entry_date=date(2026, 4, 20), earnings_date=date(2026, 4, 28), selector_structure=None, entry_mid=4.0,
        modeled_cost_pct=None, execution_penalty_at_entry=None, data_quality_score_at_entry=None,
        iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None, quote_source_at_entry="marketdata_app",
        quote_quality_at_entry=None,
    )
    assert store.invalidate("rec|baseline|always_straddle", reason="x") is True
    assert store.invalidate("rec|baseline|always_straddle", reason="x") is False
    assert store.baselines_due_for_exit(date(2026, 4, 27)) == []
    assert store.mark_missing_exits(date(2026, 5, 1)) == []
    assert store.update_exit(baseline_id="rec|baseline|always_straddle", exit_date=date(2026, 4, 27), exit_mid=4.4,
                             realized_return_pct=10.0, realized_expansion_pct=None,
                             quote_source_at_exit="marketdata_app", quote_quality_at_exit=None) is False
    with pytest.raises(ValueError):
        store.invalidate("rec|baseline|always_straddle", reason=" ")


def test_the_daily_cycle_reconciles_before_exits(tmp_path, ledger):
    ledger.record(_record("MOVE", "2026-04-28", "2026-04-20"))
    ledger.record(_record("MOVE", "2026-05-05", "2026-04-22"))
    outcomes = OutcomeStore(store_path=tmp_path / "o.sqlite")
    outcomes.insert_entry(trade_id="T_OPEN", symbol="MOVE", structure="atm_straddle", entry_date=date(2026, 4, 20),
                          setup_score=0.6, source_type="paper", earnings_date=date(2026, 4, 28), entry_mid=4.0)
    exits_seen = []

    def price_fetcher(*, symbol, **_):
        exits_seen.append(symbol)
        return {"mid": 5.0, "context": {}}

    result = forward_loop.run_daily_cycle(
        today=date(2026, 4, 27), store=outcomes, ledger=ledger,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": []}, analyzer=lambda *a, **k: None, price_fetcher=price_fetcher,
        finalizer=lambda **_: {"status": "finalized"}, funnel=ef.EvidenceFunnelStore(tmp_path / "f.sqlite"),
        symbols=["MOVE"],
    )
    assert result["event_reconciliation"]["outcomes_invalidated"] == 1
    assert outcomes.get_trade("T_OPEN")["evidence_valid"] == 0
    assert "MOVE" not in exits_seen                         # never exited under the stale date


def test_drop_superseded_keeps_rows_without_a_key():
    rows = [{"symbol": "A", "earnings_date": None}, {"symbol": "B", "earnings_date": "2026-04-28"}]
    assert drop_superseded(rows, {"B|2026-04-28": "B|2026-05-05"}) == [rows[0]]


def test_same_as_of_date_correction_supersedes_the_old_event_key(ledger):
    ledger.record(_record("AAPL", "2026-10-25", "2026-10-19"))
    ledger.record(_record("AAPL", "2026-10-27", "2026-10-19"))
    assert ledger.superseded_event_keys()["AAPL|2026-10-25"] == "AAPL|2026-10-27"
    assert ledger.event_id_for("AAPL|2026-10-27") == "AAPL|2026-10-25"


def test_the_evidence_report_counts_only_current_date_evidence(moved):
    from services.evidence_report import build_evidence_report

    ledger, outcomes, baselines = moved
    report = build_evidence_report(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert report["selector_summary"]["n"] == 2                          # T_KEEP and T_NEW, not T_OLD
    assert report["maturity"]["inputs"]["resolved_selector_outcomes"] == 2
    assert report["exit_attrition"]["baselines"]["resolved"] == 2           # rec_keep and rec_new
