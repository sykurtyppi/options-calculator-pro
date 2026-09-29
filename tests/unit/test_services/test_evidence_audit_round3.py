"""Regressions for the independent audit of #152.

1. the paid-beta gate opened with zero claimable RESOLVED outcomes (open and
   shadow rows marked claimable at entry satisfied it);
2. legacy +/-inf rows still fed maturity, attrition and universe counts, and
   forward diagnostics averaged them;
3. one infinite replay value aborted the seed run and orphaned an open row;
plus: the weekly report was not strict JSON, a legacy inf exit was retried by
the pending-finalization sweep forever, and a non-finite selector exit had no
specific reason.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

import scripts.run_forward_loop as forward_loop
from scripts.seed_outcomes_from_replay import seed_from_trades
from services.baseline_evidence_store import BaselineEvidenceStore
from services.calibration_service import IVExpansionCalibration
from services.evidence_report import build_evidence_report, build_weekly_evidence_report
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger
from services.structure_prior_store import StructurePriorStore
from tests.unit.test_services.test_evidence_report_uncertainty import EXIT, _baseline

T_MINUS_1 = date(2026, 4, 27)
EARNINGS = T_MINUS_1 + timedelta(days=1)


def _entry(store, i, *, claim):
    store.insert_entry(
        trade_id=f"T{i}", recommendation_id=f"rec-{i}", symbol=f"S{i}", structure="otm_strangle",
        entry_date=EXIT - timedelta(days=5), earnings_date=EXIT + timedelta(days=1),
        setup_score=0.6, source_type="paper", entry_mid=2.0, claim_allowed=claim,
    )


def _resolved(store, i, ret, *, claim):
    _entry(store, i, claim=claim)
    store.update_exit(trade_id=f"T{i}", exit_date=EXIT, exit_mid=2.0, realized_return_pct=ret,
                      realized_expansion_pct=ret)
    store.mark_finalized(f"T{i}")


def _force_inf(store, trade_id):
    # A row written before the stores refused NaN/inf.
    with store._conn:
        store._conn.execute(
            "UPDATE outcome_trades SET realized_return_pct = ? WHERE trade_id = ?", (float("inf"), trade_id)
        )


# ── 1. claimable evidence means claimable RESOLVED selector outcomes ──────────


def test_gate_stays_closed_when_only_open_rows_are_claimable(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    for i in range(30):
        _resolved(outcomes, i, 3.0, claim=False)
        _baseline(baselines, rec=f"rec-{i}", ret=1.0)
    for i in range(100, 130):
        _entry(outcomes, i, claim=True)  # open: no outcome yet

    report = build_evidence_report(baseline_store=baselines, outcome_store=outcomes)

    assert report["maturity"]["inputs"]["claimable_evidence_count"] == 0
    assert report["maturity"]["edge_quality_label_allowed"] is False
    gate = report["commercialization_gate"]
    assert gate["ready_for_paid_beta"] is False
    assert "enough claimable resolved selector outcomes" in gate["blocking_reasons"]


def test_claimable_count_uses_resolved_claimable_selector_outcomes(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    for i in range(30):
        _resolved(outcomes, i, 3.0, claim=True)
    for i in range(30, 40):
        _resolved(outcomes, i, 3.0, claim=False)

    report = build_evidence_report(baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
                                   outcome_store=outcomes)

    assert report["maturity"]["inputs"]["claimable_evidence_count"] == 30
    assert report["maturity"]["edge_quality_label_allowed"] is True


# ── 2. legacy non-finite rows feed no count ───────────────────────────────────


def test_legacy_inf_rows_do_not_feed_maturity_or_counts(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    for i in range(10):
        _resolved(outcomes, i, 2.0, claim=False)
    for i in range(10, 45):
        _resolved(outcomes, i, 1.0, claim=True)
        _force_inf(outcomes, f"T{i}")
    for i in range(5):
        _baseline(baselines, rec=f"u{i}", ret=1.0, cohort="universe", selector_rec="Candidate")
    _baseline(baselines, rec="p0", ret=1.0)
    with baselines._conn:
        baselines._conn.execute(
            "UPDATE baseline_trades SET realized_return_pct = ? WHERE baseline_id LIKE 'universe|u0|%'",
            (float("inf"),),
        )

    report = build_evidence_report(baseline_store=baselines, outcome_store=outcomes)

    inputs = report["maturity"]["inputs"]
    assert inputs["resolved_selector_outcomes"] == 10
    assert inputs["claimable_evidence_count"] == 0
    assert inputs["max_bucket_sample_size"] <= 10
    assert report["maturity"]["bucket_interpretation_allowed"] is False
    assert report["non_finite_outcomes"]["selector_n"] == 35
    assert report["non_finite_outcomes"]["baseline_n"] == 1
    assert report["exit_attrition"]["baselines"]["resolved"] == 5  # 4 universe + 1 paired
    assert report["universe_shadow"]["resolved"] == 4
    assert report["universe_shadow"]["by_baseline"]["always_atm_straddle"]["all_events"]["n"] == 4


def test_forward_diagnostics_ignore_inf_rows(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    for i in range(3):
        _resolved(outcomes, i, 2.0, claim=False)
    _force_inf(outcomes, "T0")

    forward = build_forward_performance_diagnostics(
        ledger=RecommendationLedger(ledger_path=tmp_path / "l.sqlite"),
        outcome_store=outcomes, baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )

    assert forward["resolved_outcome_count"] == 2
    assert forward["performance_summary"]["avg_realized_return_pct"] == 2.0
    json.dumps(forward, allow_nan=False)


def test_weekly_report_is_strict_json(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    for i in range(3):
        _resolved(outcomes, i, 2.0, claim=True)
    _force_inf(outcomes, "T0")
    # Diagnostics injected from outside the report are sanitized too.
    weekly = build_weekly_evidence_report(
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), outcome_store=outcomes,
        data_quality_diagnostics={"warning_flags": [float("inf")]},
        provider_telemetry_diagnostics={"operational_health": {"warning_flags": []}},
    )
    json.dumps(weekly, allow_nan=False)
    assert weekly["forward_recommendations"]["resolved_outcome_count"] == 2
    assert weekly["provider_data_quality_warnings"]["data_quality"] == [None]


# ── 3. seed never aborts on non-finite replay values ─────────────────────────


def _replay(symbol, day, net, pnl=10.0):
    return {"trade_date": f"2025-01-{day:02d}", "event_date": f"2025-01-{day + 1:02d}", "symbol": symbol,
            "setup_score": 0.6, "gross_return_pct": 0.05, "net_return_pct": net,
            "pnl_per_contract": pnl, "execution_profile": "backtest", "structure": "atm_straddle",
            "pricing_source": "snapshot_replay"}


def test_seed_skips_non_finite_rows_and_finishes(tmp_path):
    trades = [_replay("AAA", 2, 0.05), _replay("BBB", 3, float("inf")), _replay("CCC", 4, 0.02),
              _replay("DDD", 5, 0.01, pnl=float("inf")), _replay("EEE", 6, 1e308)]
    kwargs = dict(structure="atm_straddle", dry_run=False, outcome_store_path=tmp_path / "o.sqlite",
                  calibration_store_path=tmp_path / "cal.json", prior_store_path=tmp_path / "pri.json")

    first = seed_from_trades(trades, **kwargs)
    second = seed_from_trades(trades, **kwargs)

    assert (first["inserted"], first["skipped_bad_data"]) == (2, 3)  # 1e308 * 100 overflows to inf
    assert (second["inserted"], second["skipped_duplicate"], second["skipped_bad_data"]) == (0, 2, 3)
    rows = {row["symbol"]: row["status"] for row in OutcomeStore(tmp_path / "o.sqlite").list_for_diagnostics()}
    assert rows == {"AAA": "finalized", "CCC": "finalized"}


def test_seed_dry_run_counts_non_finite_rows_as_bad_data(tmp_path):
    result = seed_from_trades([_replay("AAA", 2, float("-inf"))], structure="atm_straddle", dry_run=True,
                              outcome_store_path=tmp_path / "o.sqlite", calibration_store_path=tmp_path / "c.json",
                              prior_store_path=tmp_path / "p.json")
    assert (result["inserted"], result["skipped_bad_data"]) == (0, 1)


# ── legacy inf exits become terminal; selector exits get a specific reason ───


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    import services.calibration_service as calibration_service
    import services.structure_prior_store as structure_prior_store

    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    priors = StructurePriorStore(tmp_path / "priors.json")
    monkeypatch.setattr(calibration_service, "get_calibration", lambda: cal)
    monkeypatch.setattr(structure_prior_store, "get_structure_prior_store", lambda: priors)
    return cal, priors


def _open(store, trade_id):
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure="otm_strangle",
        entry_date=EARNINGS - timedelta(days=6), earnings_date=EARNINGS,
        setup_score=0.6, source_type="paper", entry_mid=2.0, execution_penalty_at_entry=0.0,
        notes=json.dumps({"pricing_context": {}}),
    )


def _run(tmp_path, store, day, mid=3.0):
    return forward_loop.run_exit_detection(
        today=day, store=store, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {"mid": mid, "context": {}},
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )


def test_legacy_inf_exit_becomes_terminal_attrition(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=1.0, realized_expansion_pct=1.0)
    _force_inf(store, "AAA")

    summary = _run(tmp_path, store, EARNINGS)

    assert summary["exit_missing"] == 1 and summary["refinalized"] == 0
    row = store.get_trade("AAA")
    assert row["status"] == "exit_missing"
    assert row["exit_missing_reason"] == forward_loop.NON_FINITE_EXIT_REASON
    assert store.trades_pending_finalization(EARNINGS + timedelta(days=5)) == []


def test_live_claim_is_not_marked_unusable(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    assert store.claim_for_finalization("AAA", owner="w")
    assert store.mark_recorded_exit_unusable("AAA", reason="x") is False
    assert store.get_trade("AAA")["status"] == "finalizing"


def test_non_finite_selector_exit_records_a_specific_reason(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")

    summary = _run(tmp_path, store, T_MINUS_1, mid=float("inf"))

    assert summary["skipped"] == 1 and summary["exits"] == 0
    row = store.get_trade("AAA")
    assert row["status"] == "open"
    assert row["last_exit_attempt_reason"] == forward_loop.NON_FINITE_EXIT_REASON
    _run(tmp_path, store, EARNINGS)
    row = store.get_trade("AAA")
    assert row["status"] == "exit_missing"
    assert row["exit_missing_reason"] == forward_loop.NON_FINITE_EXIT_REASON
