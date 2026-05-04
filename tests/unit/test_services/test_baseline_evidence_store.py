from __future__ import annotations

from datetime import date

from services.baseline_evidence_store import BaselineEvidenceStore, make_baseline_id


def test_baseline_store_records_and_resolves_shadow_trade(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "baseline.sqlite")

    inserted = store.insert_entry(
        recommendation_id="rec-1",
        symbol="AAPL",
        baseline_name="always_atm_straddle",
        structure="atm_straddle",
        entry_date=date(2026, 4, 24),
        earnings_date=date(2026, 4, 25),
        selector_structure="otm_strangle",
        entry_mid=4.0,
        modeled_cost_pct=1.0,
        execution_penalty_at_entry=0.04,
        data_quality_score_at_entry=0.82,
        iv_rv_har_at_entry=0.94,
        iv_rv_yz_at_entry=0.97,
        quote_source_at_entry="marketdata_app",
        quote_quality_at_entry="marketdata_app_paper_research_mid_not_execution_grade",
        entry_bid_ask_mid={"legs": {"call": {"mid": 2.0}}},
        evidence_quality_status="degraded_evidence",
        evidence_quality_reasons=["quote_not_execution_grade"],
        claim_allowed=False,
        execution_grade=False,
        entry_execution_scenarios={"scenario_values": {"mid": 4.0, "cross_50": 4.5}},
    )

    assert inserted is True
    assert store.count() == 1
    assert store.insert_entry(
        recommendation_id="rec-1",
        symbol="AAPL",
        baseline_name="always_atm_straddle",
        structure="atm_straddle",
        entry_date=date(2026, 4, 24),
        earnings_date=date(2026, 4, 25),
        selector_structure="otm_strangle",
        entry_mid=4.0,
        modeled_cost_pct=1.0,
        execution_penalty_at_entry=0.04,
        data_quality_score_at_entry=0.82,
        iv_rv_har_at_entry=0.94,
        iv_rv_yz_at_entry=0.97,
        quote_source_at_entry="marketdata_app",
        quote_quality_at_entry="marketdata_app_paper_research_mid_not_execution_grade",
    ) is False

    due = store.baselines_due_for_exit(date(2026, 4, 24))
    assert len(due) == 1

    assert store.update_exit(
        baseline_id=make_baseline_id("rec-1", "always_atm_straddle"),
        exit_date=date(2026, 4, 24),
        exit_mid=5.0,
        realized_return_pct=24.0,
        realized_expansion_pct=25.0,
        quote_source_at_exit="marketdata_app",
        quote_quality_at_exit="marketdata_app_paper_research_mid_not_execution_grade",
        exit_bid_ask_mid={"legs": {"call": {"mid": 2.5}}},
        exit_execution_scenarios={
            "scenario_values": {"mid": 5.0, "cross_50": 4.8},
            "scenario_outcomes": {"realized_return_pct": {"mid": 25.0}},
        },
    ) is True

    rows = store.list_for_diagnostics()
    assert rows[0]["status"] == "resolved"
    assert rows[0]["realized_return_pct"] == 24.0
    assert rows[0]["entry_bid_ask_mid_json"]["legs"]["call"]["mid"] == 2.0
    assert rows[0]["evidence_quality_status"] == "degraded_evidence"
    assert rows[0]["claim_allowed"] == 0
    assert rows[0]["entry_execution_scenarios_json"]["scenario_values"]["cross_50"] == 4.5
    assert rows[0]["exit_execution_scenarios_json"]["scenario_outcomes"]["realized_return_pct"]["mid"] == 25.0
