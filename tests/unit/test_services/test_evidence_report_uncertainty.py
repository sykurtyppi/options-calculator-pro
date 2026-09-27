"""The evidence report's uncertainty block, end to end through the stores."""
from __future__ import annotations

from datetime import date, timedelta

from services.baseline_evidence_store import (
    EXIT_REPRICING_BOOKED,
    EXIT_REPRICING_LEGACY,
    BaselineEvidenceStore,
    make_baseline_id,
)
from services.evidence_report import build_evidence_report
from services.outcome_recorder import OutcomeStore

EXIT = date(2026, 5, 4)


def _selector(store, i, ret):
    trade_id = f"T{i}"
    store.insert_entry(
        trade_id=trade_id, recommendation_id=f"rec-{i}", symbol=f"S{i}", structure="otm_strangle",
        entry_date=EXIT - timedelta(days=5), earnings_date=EXIT + timedelta(days=1),
        setup_score=0.6, source_type="paper", entry_mid=2.0,
    )
    store.update_exit(trade_id=trade_id, exit_date=EXIT, exit_mid=2.0,
                      realized_return_pct=ret, realized_expansion_pct=ret)
    store.mark_finalized(trade_id)


def _baseline(store, *, rec, ret, cohort="paired", repricing=EXIT_REPRICING_BOOKED,
              selector_rec=None, name="always_atm_straddle"):
    baseline_id = make_baseline_id(rec, name) if cohort == "paired" else f"universe|{rec}|{name}"
    store.insert_entry(
        recommendation_id=rec, baseline_id=baseline_id, symbol=rec, baseline_name=name,
        structure="atm_straddle", entry_date=EXIT - timedelta(days=5), earnings_date=EXIT + timedelta(days=1),
        selector_structure=None, entry_mid=5.0, modeled_cost_pct=0.0, execution_penalty_at_entry=0.0,
        data_quality_score_at_entry=0.9, iv_rv_har_at_entry=1.0, iv_rv_yz_at_entry=1.0,
        quote_source_at_entry="yfinance", quote_quality_at_entry="paper",
        cohort=cohort, selector_recommendation=selector_rec,
    )
    store.update_exit(baseline_id=baseline_id, exit_date=EXIT, exit_mid=5.0, realized_return_pct=ret,
                      realized_expansion_pct=ret, quote_source_at_exit="yfinance",
                      quote_quality_at_exit="paper", exit_repricing=repricing)


def test_report_pairs_selector_and_baseline_on_the_same_events(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    for i in range(12):
        _selector(outcomes, i, 4.0 + i)
        _baseline(baselines, rec=f"rec-{i}", ret=1.0 + i)
    # Unverified exits and invalid outcomes never enter the pairing.
    _baseline(baselines, rec="rec-0", ret=500.0, repricing=EXIT_REPRICING_LEGACY, name="always_otm_strangle")
    _selector(outcomes, 99, -900.0)
    outcomes.invalidate("T99", reason="x")

    report = build_evidence_report(baseline_store=baselines, outcome_store=outcomes)

    block = report["uncertainty"]
    assert block["selector"]["n"] == 12
    assert block["selector"]["verdict"] == "above_zero"
    straddle = block["selector_minus_baseline"]["always_atm_straddle"]
    assert straddle["pairs"] == 12
    assert straddle["mean"] == 3.0
    assert straddle["verdict"] == "above_zero"
    assert "always_otm_strangle" not in block["selector_minus_baseline"]
    assert "bootstrap" in block["method"]


def test_report_measures_picked_minus_skipped_in_the_universe(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    for i in range(12):
        _baseline(baselines, rec=f"p{i}", ret=6.0 + (i % 3), cohort="universe", selector_rec="Candidate")
        _baseline(baselines, rec=f"s{i}", ret=-4.0 - (i % 3), cohort="universe", selector_rec="No Trade")

    report = build_evidence_report(baseline_store=baselines, outcome_store=OutcomeStore(tmp_path / "o.sqlite"))

    item = report["uncertainty"]["universe_picked_minus_skipped"]["always_atm_straddle"]
    assert (item["n_first"], item["n_second"]) == (12, 12)
    assert item["mean"] == 12.0  # picked mean 7, skipped mean -5
    assert item["verdict"] == "above_zero"


def test_fragile_selector_result_is_flagged(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    for i, ret in enumerate([40.0] * 5 + [-2.0] * 20):
        _selector(outcomes, i, ret)

    report = build_evidence_report(baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), outcome_store=outcomes)

    assert report["uncertainty"]["selector"]["outlier_dependence"]["fragile"] is True
    assert any("5 best results" in warning for warning in report["warning_flags"])


def test_small_live_sample_reports_no_interval(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    for i, ret in enumerate([8.96, -23.81, -10.0]):
        _selector(outcomes, i, ret)

    report = build_evidence_report(baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), outcome_store=outcomes)

    selector = report["uncertainty"]["selector"]
    assert selector["verdict"] == "insufficient_sample"
    assert selector["ci_low"] is None
    assert not any("5 best results" in warning for warning in report["warning_flags"])
