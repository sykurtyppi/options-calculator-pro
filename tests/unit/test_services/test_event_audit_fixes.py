"""Event-identity and funnel findings from the independent audit of main at bfc48f5.

P0  a date that moved and moved back left the trade invalidated for good;
P0  event grouping depended on the order rows were written (same-day
    conflicts, backfills);
P0  a reconciliation that could not write still let the exit pass exit and
    learn superseded trades;
P1  a backfilled older analysis became the event's "latest" decision;
P1  one event booked as two structures counted twice;
P1  re-check schedule/close write failures were silent;
P1  funnel health missed a discovery outage, raised false stalls in quiet
    stretches, and lost the count of a cycle whose writes all failed;
P1  a historical replay row superseded forward evidence;
P2  a trade that missed T-1 stayed "open" in the funnel cohort forever;
P2  processes opening a new ledger at once raced on the WAL switch;
P2  an expired analyzer-error re-check was dropped at the wrong stage.
"""
from __future__ import annotations

import multiprocessing
import sqlite3
import uuid
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_forward_loop as forward_loop
from services import evidence_funnel as ef
from services.automation_watchdog import _check_forward_loop
from services.baseline_evidence_store import BaselineEvidenceStore
from services.event_reconciliation import (
    SUPERSEDED_REASON,
    one_trade_per_event,
    reconcile_moved_events,
    safe_reconcile_moved_events,
)
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger, RecommendationRecord, derive_event_groups

T = date(2026, 4, 21)


def _rec(symbol, as_of, earnings, *, structure="call_calendar", provenance="forward_post_freeze",
         rid=None, created=None):
    return RecommendationRecord(
        recommendation_id=rid or f"rec_{uuid.uuid4().hex[:8]}", created_at=created or f"{as_of}T12:00:00+00:00",
        symbol=symbol, as_of_date=as_of, earnings_date=earnings, earnings_source="x",
        earnings_source_confidence=1.0, earnings_source_stale=False, recommendation="Trade",
        selected_structure=structure, no_trade_reason=None, data_quality_score=1.0,
        option_source="x", underlying_source="x", sample_provenance=provenance)


@pytest.fixture
def stores(tmp_path):
    ledger = RecommendationLedger(tmp_path / "l.sqlite")
    outcomes = OutcomeStore(store_path=tmp_path / "o.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    yield ledger, outcomes, baselines
    ledger.close()


def _trade(outcomes, trade_id, symbol, earnings, entry=date(2026, 2, 1), structure="call_calendar"):
    outcomes.insert_entry(trade_id=trade_id, symbol=symbol, structure=structure, entry_date=entry, setup_score=1,
                          source_type="paper", earnings_date=earnings, entry_mid=1.0)


# ── P0: a date that moves back restores the evidence ───────────────────────

def test_a_date_that_moves_back_restores_the_trade(stores):
    ledger, outcomes, baselines = stores
    ledger.record(_rec("AAA", "2026-02-01", "2026-02-10"))
    _trade(outcomes, "t1", "AAA", date(2026, 2, 10))
    reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    ledger.record(_rec("AAA", "2026-02-02", "2026-02-12"))         # briefly reported elsewhere
    assert reconcile_moved_events(ledger=ledger, outcome_store=outcomes,
                                  baseline_store=baselines)["outcomes_invalidated"] == 1
    assert outcomes.trades_due_for_exit(date(2026, 2, 9)) == []
    ledger.record(_rec("AAA", "2026-02-03", "2026-02-10"))         # and back
    summary = reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert summary["outcomes_revalidated"] == 1
    trade = outcomes.get_trade("t1")
    assert (trade["evidence_valid"], trade["invalidation_reason"]) == (1, None)
    assert [row["trade_id"] for row in outcomes.trades_due_for_exit(date(2026, 2, 9))] == ["t1"]
    # Idempotent.
    again = reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert (again["outcomes_revalidated"], again["outcomes_invalidated"]) == (0, 0)


def test_a_baseline_is_restored_too_and_other_invalidations_are_not(stores, tmp_path):
    ledger, outcomes, baselines = stores
    ledger.record(_rec("AAA", "2026-02-01", "2026-02-10"))
    for rid in ("r1", "r2"):
        baselines.insert_entry(recommendation_id=rid, symbol="AAA", baseline_name="always_atm_straddle",
                               structure="atm_straddle", entry_date=date(2026, 2, 1),
                               earnings_date=date(2026, 2, 10), selector_structure=None, entry_mid=2.0,
                               modeled_cost_pct=None, execution_penalty_at_entry=None,
                               data_quality_score_at_entry=None, iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None,
                               quote_source_at_entry=None, quote_quality_at_entry=None)
    _trade(outcomes, "t_manual", "AAA", date(2026, 2, 10))
    outcomes.invalidate("t_manual", reason="bad_quote")
    baselines.invalidate("r2|baseline|always_atm_straddle", reason="bad_quote")
    ledger.record(_rec("AAA", "2026-02-02", "2026-02-12"))
    assert reconcile_moved_events(ledger=ledger, outcome_store=outcomes,
                                  baseline_store=baselines)["baselines_invalidated"] == 1
    ledger.record(_rec("AAA", "2026-02-03", "2026-02-10"))
    summary = reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert (summary["baselines_revalidated"], summary["outcomes_revalidated"]) == (1, 0)
    assert outcomes.get_trade("t_manual")["invalidation_reason"] == "bad_quote"
    valid = {row["baseline_id"] for row in baselines.list_for_diagnostics()}
    assert valid == {"r1|baseline|always_atm_straddle"}
    with pytest.raises(ValueError):
        outcomes.revalidate("t_manual", reason_prefix="")


def test_a_superseded_trade_does_not_block_the_event_but_a_restorable_one_does(stores):
    _, outcomes, _ = stores
    _trade(outcomes, "t_moved", "AAA", date(2026, 2, 10))
    outcomes.invalidate("t_moved", reason=f"{SUPERSEDED_REASON}:AAA|2026-02-12")
    assert outcomes.find_active_trade_for_event(symbol="AAA", earnings_date=date(2026, 2, 10))["trade_id"] == "t_moved"
    _trade(outcomes, "t_bad", "BBB", date(2026, 2, 10))
    outcomes.invalidate("t_bad", reason="bad_quote")
    assert outcomes.find_active_trade_for_event(symbol="BBB", earnings_date=date(2026, 2, 10)) is None


# ── P0: grouping depends only on what was observed when ────────────────────

@pytest.mark.parametrize("order", [
    [("2026-02-02", "2026-02-12", "13"), ("2026-02-02", "2026-02-10", "14")],
    [("2026-02-02", "2026-02-10", "13"), ("2026-02-02", "2026-02-12", "14")],
])
def test_a_same_day_conflict_groups_the_same_whichever_is_written_first(tmp_path, order):
    ledger = RecommendationLedger(tmp_path / "l.sqlite")
    ledger.record(_rec("AAA", "2026-02-01", "2026-02-10", created="2026-02-01T12:00:00+00:00"))
    for as_of, earnings, hour in order:
        ledger.record(_rec("AAA", as_of, earnings, created=f"{as_of}T{hour}:00:00+00:00"))
    ledger.record(_rec("AAA", "2026-02-03", "2026-02-12"))
    assert ledger.superseded_event_keys() == {"AAA|2026-02-10": "AAA|2026-02-12"}
    assert ledger.summarize()["events"]["total"] == 1


def test_a_backfilled_first_analysis_joins_the_event(tmp_path):
    facts = [("2026-02-01", "2026-02-10")] + [(f"2026-02-{d:02d}", "2026-02-12") for d in (2, 6, 11)]
    results = []
    for name, order in (("chronological", facts), ("backfilled", facts[1:] + facts[:1])):
        ledger = RecommendationLedger(tmp_path / f"{name}.sqlite")
        for as_of, earnings in order:
            ledger.record(_rec("AAA", as_of, earnings))
        results.append((ledger.superseded_event_keys(), ledger.event_id_for("AAA|2026-02-12"),
                        [row["event_key"] for row in ledger.latest_event_rows()]))
    assert results[0] == results[1] == ({"AAA|2026-02-10": "AAA|2026-02-12"}, "AAA|2026-02-10",
                                        ["AAA|2026-02-12"])


def test_derived_groups_ignore_input_order():
    observations = [("2026-02-01", "a", "X|2026-02-10"), ("2026-02-02", "b", "X|2026-02-12"),
                    ("2026-02-03", "c", "X|2026-02-10"), ("2026-02-03", "d", "X|2026-05-10")]
    forward = derive_event_groups(observations)
    assert derive_event_groups(list(reversed(observations))) == forward
    assert forward["X|2026-02-12"] == {"event_id": "X|2026-02-10", "current": "X|2026-02-10",
                                       "first_as_of": "2026-02-02"}
    assert forward["X|2026-05-10"]["event_id"] == "X|2026-05-10"       # another quarter


def test_an_older_ledger_is_regrouped_once_on_open(tmp_path):
    path = tmp_path / "l.sqlite"
    ledger = RecommendationLedger(path)
    ledger.record(_rec("AAA", "2026-02-02", "2026-02-12"))
    ledger.record(_rec("AAA", "2026-02-01", "2026-02-10"))
    # As the old write-order rule left it: two events, nothing superseded.
    ledger._conn.execute("DELETE FROM event_supersessions")
    ledger._conn.execute("UPDATE event_identity SET event_id = event_key")
    ledger._conn.execute("DELETE FROM event_grouping_meta")
    ledger._conn.commit()
    ledger.close()
    reopened = RecommendationLedger(path)
    assert reopened.superseded_event_keys() == {"AAA|2026-02-10": "AAA|2026-02-12"}
    assert reopened.event_id_for("AAA|2026-02-12") == "AAA|2026-02-10"


# ── P1: latest analysis is the latest in information time ──────────────────

def test_a_backfilled_older_analysis_is_not_the_events_decision(tmp_path):
    ledger = RecommendationLedger(tmp_path / "l.sqlite")
    ledger.record(_rec("AAA", "2026-04-20", "2026-05-01", rid="r_0420"))
    ledger.record(replace(_rec("AAA", "2026-04-10", "2026-05-01", rid="r_0410"), recommendation="No Trade",
                          selected_structure=None))
    assert ledger.summarize()["events"]["by_recommendation"] == {"Trade": 1}
    assert [row["recommendation_id"] for row in ledger.latest_event_rows(selected_structure=None)] == ["r_0420"]
    pending = ledger.list_pending_candidate_exit_resolutions(now=date(2026, 5, 9))
    assert [row["recommendation_id"] for row in pending] == ["r_0420"]


def test_diagnostics_count_the_latest_analysis_in_information_time(stores):
    ledger, outcomes, baselines = stores
    ledger.record(_rec("AAA", "2026-04-20", "2026-05-01"))
    ledger.record(replace(_rec("AAA", "2026-04-10", "2026-05-01"), recommendation="No Trade"))
    report = build_forward_performance_diagnostics(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert (report["total_events"], report["no_trade_event_count"]) == (1, 0)


# ── P1: a replay never moves a forward event ───────────────────────────────

def test_a_replay_row_does_not_supersede_forward_evidence(tmp_path):
    ledger = RecommendationLedger(tmp_path / "l.sqlite")
    ledger.record(_rec("AAA", "2026-02-05", "2026-02-10"))
    ledger.record(_rec("AAA", "2026-02-08", "2026-02-11", provenance="historical_replay_in_sample_or_research",
                       created="2026-03-01T00:00:00+00:00"))
    assert ledger.superseded_event_keys() == {}
    assert [row["event_key"] for row in ledger.latest_event_rows()] == ["AAA|2026-02-10"]
    assert [row["event_key"] for row in ledger.list_pending_candidate_exit_resolutions(now=date(2026, 3, 2))] == [
        "AAA|2026-02-10"]
    assert ledger.event_id_for("AAA|2026-02-11") == "AAA|2026-02-11"


# ── P0: exits refuse superseded keys whatever reconciliation managed ───────

def _locked_reconcile(ledger, outcomes, baselines, path):
    other = sqlite3.connect(str(path))
    other.execute("BEGIN IMMEDIATE")
    try:
        outcomes._conn.execute("PRAGMA busy_timeout=50")
        return safe_reconcile_moved_events(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    finally:
        other.rollback()
        other.close()


def test_a_reconciliation_that_cannot_write_does_not_let_exits_through(stores, tmp_path):
    ledger, outcomes, baselines = stores
    ledger.record(_rec("AAA", "2026-02-01", "2026-02-10"))
    _trade(outcomes, "tA", "AAA", date(2026, 2, 10), structure="atm_straddle")
    _trade(outcomes, "tB", "BBB", date(2026, 2, 10), structure="atm_straddle")
    ledger.record(_rec("AAA", "2026-02-05", "2026-02-20"))
    result = _locked_reconcile(ledger, outcomes, baselines, tmp_path / "o.sqlite")
    assert result["outcomes_invalidated"] == 0 and result["errors"]
    assert "error" not in result
    calls = []

    def finalizer(**kw):
        calls.append(kw["trade_id"])
        return {"status": "finalized"}

    summary = forward_loop.run_exit_detection(
        today=date(2026, 2, 9), store=outcomes, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {"mid": 2.0, "context": {}}, finalizer=finalizer, baseline_store=baselines,
        funnel=ef.EvidenceFunnelStore(tmp_path / "f.sqlite"), ledger=ledger)
    assert calls == ["tB"] and summary["superseded_skipped"] == 1


def test_unreadable_superseded_keys_stop_every_exit(stores, tmp_path):
    _, outcomes, baselines = stores
    _trade(outcomes, "tB", "BBB", date(2026, 2, 10), structure="atm_straddle")

    class _Broken:
        def superseded_event_keys(self):
            raise sqlite3.OperationalError("database is locked")

    calls = []
    summary = forward_loop.run_exit_detection(
        today=date(2026, 2, 9), store=outcomes, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {"mid": 2.0, "context": {}},
        finalizer=lambda **kw: calls.append(kw) or {"status": "finalized"}, baseline_store=baselines,
        funnel=ef.EvidenceFunnelStore(tmp_path / "f.sqlite"), ledger=_Broken())
    assert calls == [] and "locked" in summary["event_identity_error"]
    errors, warnings = [], []
    _check_forward_loop({"exits": summary, "event_reconciliation": {"errors": ["invalidate tA: locked"]}},
                        errors, warnings)
    assert any("No exits ran" in e for e in errors) and any("reconciliation" in w for w in warnings)


def test_the_run_log_counts_reconciliation_errors():
    from scripts.run_evidence_cycle import _reconciliation_error_count

    assert _reconciliation_error_count({"errors": ["a", "b"]}) == 2
    assert _reconciliation_error_count({"error": "boom"}) == 1
    assert _reconciliation_error_count(None) == 0


# ── P1: one event, one observation ─────────────────────────────────────────

def _analysis(day, structure, earnings):
    def analyze(sym, mda_client=None):
        return SimpleNamespace(
            symbol=sym, recommendation="Candidate", setup_score=0.6, metrics={}, rationale=[],
            selector_output={"recommendation": "Candidate", "best_structure": structure,
                             "earnings_date": earnings.isoformat()},
            structure_scorecards=[{"structure": structure, "execution_penalty": 0.0}],
            vol_snapshot={"as_of_date": day.isoformat(), "earnings_date": earnings.isoformat(),
                          "data_quality_score": 0.9})
    return analyze


def test_a_second_structure_for_the_same_event_is_not_booked(stores, tmp_path):
    ledger, outcomes, baselines = stores
    earnings = T + timedelta(days=8)
    entries = []
    for offset, structure in ((0, "atm_straddle"), (1, "otm_strangle")):
        day = T + timedelta(days=offset)
        summary = forward_loop.run_forward_screener(
            today=day, store=outcomes, ledger=ledger, baseline_store=baselines, log_path=tmp_path / "log.jsonl",
            screener_builder=lambda **_: {"rows": [{"symbol": "X", "dte": (earnings - day).days,
                                                    "earnings_date": earnings.isoformat()}]},
            analyzer=_analysis(day, structure, earnings), price_fetcher=lambda **_: {"mid": 4.0, "context": {}},
            funnel=ef.EvidenceFunnelStore(tmp_path / "f.sqlite"), symbols=["X"])
        entries.append((summary["entries"], summary["duplicates"]))
    assert entries == [(1, 0), (0, 1)]


def test_reports_keep_each_events_first_trade_only():
    rows = [{"trade_id": "b", "symbol": "X", "earnings_date": "2026-04-29", "entry_date": "2026-04-22"},
            {"trade_id": "a", "symbol": "X", "earnings_date": "2026-04-29", "entry_date": "2026-04-21"},
            {"trade_id": "c", "symbol": "Y", "earnings_date": "2026-04-29", "entry_date": "2026-04-22"},
            {"trade_id": "d", "symbol": "Z", "earnings_date": None}]
    kept, extra = one_trade_per_event(rows)
    assert [row["trade_id"] for row in kept] == ["a", "c", "d"] and extra == 1


def test_diagnostics_count_one_event_once(stores):
    ledger, outcomes, baselines = stores
    for trade_id, structure, entry in (("t1", "atm_straddle", date(2026, 4, 21)),
                                       ("t2", "otm_strangle", date(2026, 4, 22))):
        _trade(outcomes, trade_id, "X", date(2026, 4, 29), entry=entry, structure=structure)
        outcomes.update_exit(trade_id=trade_id, exit_date=date(2026, 4, 28), exit_mid=5.0, realized_return_pct=25.0,
                             realized_pnl=1.0, realized_expansion_pct=10.0)
        outcomes.mark_finalized(trade_id)
    report = build_forward_performance_diagnostics(ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert (report["resolved_outcome_count"], report["performance_summary"]["n"],
            report["same_event_extra_trades"]) == (1, 1, 1)


# ── P1: re-check writes that fail are not silent ───────────────────────────

class _LockedOnRecheck(ef.EvidenceFunnelStore):
    def schedule_recheck(self, **kw):
        raise sqlite3.OperationalError("database is locked")


def test_a_lost_recheck_schedule_degrades_the_cycle(stores, tmp_path):
    ledger, outcomes, baselines = stores
    funnel = _LockedOnRecheck(tmp_path / "f.sqlite")
    earnings = T + timedelta(days=20)
    summary = forward_loop.run_forward_screener(
        today=T, store=outcomes, ledger=ledger, baseline_store=baselines, log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": [{"symbol": "EARLY", "dte": 20, "earnings_date": earnings.isoformat()}]},
        analyzer=lambda s, mda_client=None: None, price_fetcher=lambda **_: {"mid": None},
        funnel=funnel, cycle_run_id="c1", symbols=["EARLY"])
    assert summary["rechecks_scheduled"] == 0
    assert summary["evidence_funnel"]["status"] == "degraded"
    (event,) = [e for e in funnel.events("c1") if e["stage"] == "data_qualified"]
    assert '"recheck_on": null' in event["detail_json"]


class _FlakyClose(ef.EvidenceFunnelStore):
    fail = False

    def close_recheck(self, **kw):
        if self.fail and kw.get("status", ef.RECHECK_DONE) == ef.RECHECK_DONE:
            raise sqlite3.OperationalError("database is locked")
        return super().close_recheck(**kw)


def test_a_lost_recheck_close_is_reported_and_later_closed_as_done(stores, tmp_path):
    ledger, outcomes, baselines = stores
    funnel = _FlakyClose(tmp_path / "f.sqlite")
    earnings = T + timedelta(days=6)

    def run(day, rows, quote, cycle):
        return forward_loop.run_forward_screener(
            today=day, store=outcomes, ledger=ledger, baseline_store=baselines, log_path=tmp_path / "log.jsonl",
            screener_builder=lambda **_: {"rows": rows}, analyzer=_analysis(T, "atm_straddle", earnings),
            price_fetcher=lambda **_: {"mid": quote, "context": {}} if quote else {"mid": None, "reason": "no_spot"},
            funnel=funnel, cycle_run_id=cycle, symbols=["X"])

    run(T, [{"symbol": "X", "dte": 6, "earnings_date": earnings.isoformat()}], None, "c1")
    funnel.fail = True
    second = run(T + timedelta(days=1), [], 4.0, "c2")
    funnel.fail = False
    assert second["entries"] == 1 and second["evidence_funnel"]["status"] == "degraded"
    third = run(earnings - timedelta(days=2), [], 4.0, "c3")
    assert third["rechecks_expired"] == 0
    assert funnel.recheck_counts()["done"] == 1


@pytest.mark.parametrize("reason, stage", [
    (ef.REASON_TOO_EARLY, ef.STAGE_DATA_QUALIFIED),
    (f"{ef.REASON_ANALYZER_ERROR}:TimeoutError", ef.STAGE_ANALYZED),
    (f"{ef.REASON_MISSING_ENTRY_QUOTE}:no_spot", ef.STAGE_RECORDED),
])
def test_an_expired_recheck_drops_at_the_stage_it_was_deferred(reason, stage):
    assert forward_loop._recheck_stage(reason) == stage


# ── P1: funnel health ──────────────────────────────────────────────────────

def _cycles(tmp_path, rows_for_day, days=4):
    funnel = ef.EvidenceFunnelStore(tmp_path / "f.sqlite")
    for k in range(days):
        day = T + timedelta(days=k)
        forward_loop.run_forward_screener(
            today=day, store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
            ledger=RecommendationLedger(tmp_path / "r.sqlite"), baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
            log_path=tmp_path / "log.jsonl", screener_builder=lambda **_: {"rows": rows_for_day(day)},
            analyzer=lambda s, mda_client=None: (_ for _ in ()).throw(RuntimeError("provider down")),
            price_fetcher=lambda **_: {"mid": None}, funnel=funnel, symbols=["A"])
    return ef.build_evidence_funnel_health(store=funnel, as_of=T + timedelta(days=days - 1))


def test_a_discovery_outage_alerts(tmp_path):
    health = _cycles(tmp_path, lambda day: [])
    assert health["summary"]["cycles_without_discovery"] == 4
    assert any("Discovery returned no candidates" in i["message"] and i["alertable"] for i in health["issues"])


def test_a_quiet_stretch_is_not_a_stall(tmp_path):
    health = _cycles(tmp_path, lambda day: [{"symbol": "A", "earnings_date": (day + timedelta(days=25)).isoformat()}])
    assert health["issues"] == [] and health["summary"]["cycles_without_analysis"] == 0


def test_in_window_candidates_that_are_never_analyzed_are_a_stall(tmp_path):
    health = _cycles(tmp_path, lambda day: [{"symbol": "A", "earnings_date": (day + timedelta(days=8)).isoformat()}])
    assert health["summary"]["cycles_without_analysis"] == 4
    assert any("Entry pipeline stalled" in i["message"] for i in health["issues"])


def test_a_cycle_whose_writes_all_fail_is_still_seen_by_the_watchdog(stores, tmp_path):
    ledger, outcomes, baselines = stores
    funnel = ef.EvidenceFunnelStore(tmp_path / "f.sqlite")
    funnel._conn.execute("PRAGMA busy_timeout=50")
    other = sqlite3.connect(str(tmp_path / "f.sqlite"))
    other.execute("BEGIN IMMEDIATE")
    try:
        summary = forward_loop.run_forward_screener(
            today=T, store=outcomes, ledger=ledger, baseline_store=baselines, log_path=tmp_path / "log.jsonl",
            screener_builder=lambda **_: {"rows": [{"symbol": "A", "earnings_date": (T + timedelta(days=20)).isoformat()}]},
            analyzer=lambda s, mda_client=None: None, price_fetcher=lambda **_: {"mid": None}, funnel=funnel,
            cycle_run_id="c1", symbols=["A"])
    finally:
        other.rollback()
        other.close()
    assert summary["evidence_funnel"]["status"] == "degraded"
    assert summary["evidence_funnel"]["dropped_writes_persisted"] is True
    health = ef.build_evidence_funnel_health(store=funnel, as_of=T)
    assert health["summary"]["dropped_writes_7d"] == summary["evidence_funnel"]["dropped_writes"] > 0
    assert not ef._lost_writes_marker(funnel.path).exists()


# ── P2: a trade that skipped exit_due is not open forever ──────────────────

def test_a_trade_that_missed_t_minus_1_settles_its_cohort(tmp_path):
    funnel = ef.EvidenceFunnelStore(tmp_path / "f.sqlite")
    funnel.record(cycle_run_id="c1", as_of=date(2026, 4, 1), stage="recorded", outcome="advanced", symbol="X",
                  trade_id="t1")
    funnel.record(cycle_run_id="c9", as_of=date(2026, 4, 20), stage="resolved", outcome="dropped", symbol="X",
                  trade_id="t1", reason="exit_missing:no_exit_attempt_recorded")
    funnel.record(cycle_run_id="c1", as_of=date(2026, 4, 1), stage="recorded", outcome="advanced", symbol="Y",
                  trade_id="t2")
    funnel.record(cycle_run_id="c9", as_of=date(2026, 4, 20), stage="resolved", outcome="advanced", symbol="Y",
                  trade_id="t2")
    transitions = funnel.summarize(since=date(2026, 3, 1), until=date(2026, 6, 1))["transitions"]
    step = transitions["recorded->exit_due"]
    assert (step["cohort"], step["advanced"], step["stopped"], step["open"]) == (2, 1, 1, 0)


# ── P2: concurrent first opens survive the WAL switch ──────────────────────

_barrier = None


def _init(barrier):
    global _barrier
    _barrier = barrier


def _open_and_record(args):
    path, index = args
    _barrier.wait()
    try:
        RecommendationLedger(Path(path)).record(_rec("AAA", "2026-02-01", "2026-02-10", rid=f"new{index}"))
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return "ok"


def test_many_processes_opening_a_new_ledger_all_succeed(tmp_path):
    ctx = multiprocessing.get_context("fork")
    for trial in range(5):
        path = tmp_path / f"l{trial}.sqlite"
        sqlite3.connect(str(path)).close()             # exists, still in rollback-journal mode
        barrier = ctx.Barrier(8)
        with ctx.Pool(8, initializer=_init, initargs=(barrier,)) as pool:
            results = pool.map(_open_and_record, [(str(path), i) for i in range(8)])
        assert results == ["ok"] * 8


def test_the_wal_switch_retries_a_locked_file(tmp_path, monkeypatch):
    from services import sqlite_helpers

    calls = []

    class _Conn:
        def execute(self, sql):
            calls.append(sql)
            if sql == "PRAGMA journal_mode=WAL" and len(calls) < 4:
                raise sqlite3.OperationalError("database is locked")
            return SimpleNamespace(fetchone=lambda: ("delete",))

    monkeypatch.setattr(sqlite_helpers.time, "sleep", lambda _: None)
    sqlite_helpers.enable_wal_mode(_Conn(), timeout_ms=1000)
    assert calls.count("PRAGMA journal_mode=WAL") == 2
    class _Broken:
        def execute(self, sql):
            raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError):                 # not a lock: never retried
        sqlite_helpers.enable_wal_mode(_Broken())


def test_a_new_date_joins_the_most_recently_confirmed_of_two_candidate_events():
    observations = [("2026-02-01", "a", "X|2026-02-10"), ("2026-02-01", "b", "X|2026-02-14"),  # ambiguous
                    ("2026-02-02", "c", "X|2026-02-10"),                                       # 02-10 confirmed
                    ("2026-02-03", "d", "X|2026-02-12")]
    groups = derive_event_groups(observations)
    assert groups["X|2026-02-12"]["event_id"] == "X|2026-02-10"
    assert groups["X|2026-02-14"]["current"] == "X|2026-02-14"
