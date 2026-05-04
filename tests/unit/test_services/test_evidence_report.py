from __future__ import annotations

from datetime import date

from services.baseline_evidence_store import BaselineEvidenceStore, make_baseline_id
from services.evidence_report import build_evidence_report, build_weekly_evidence_report
from services.outcome_recorder import OutcomeStore


def test_evidence_report_compares_selector_to_baselines(tmp_path):
    outcomes = OutcomeStore(tmp_path / "outcomes.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    outcomes.insert_entry(
        trade_id="AAPL|2026-04-24|otm_strangle",
        recommendation_id="rec-1",
        symbol="AAPL",
        structure="otm_strangle",
        entry_date=date(2026, 4, 24),
        earnings_date=date(2026, 4, 25),
        setup_score=0.72,
        source_type="paper",
        entry_mid=2.0,
        iv_rv_har=0.92,
        evidence_quality_status="degraded_evidence",
        evidence_quality_reasons=["quote_not_execution_grade"],
        claim_allowed=False,
        execution_grade=False,
        entry_execution_scenarios={
            "scenario_values": {"mid": 2.0, "cross_50": 2.2},
            "spread_as_pct_of_premium": 20.0,
        },
        surface_quality={
            "status": "degraded_surface",
            "warning_flags": ["sparse_strikes_around_atm"],
            "sparse_atm_expiration_count": 1,
        },
    )
    outcomes.update_exit(
        trade_id="AAPL|2026-04-24|otm_strangle",
        exit_date=date(2026, 4, 24),
        exit_mid=2.4,
        realized_return_pct=18.0,
        realized_expansion_pct=20.0,
    )
    outcomes.mark_finalized("AAPL|2026-04-24|otm_strangle")

    baselines.insert_entry(
        recommendation_id="rec-1",
        symbol="AAPL",
        baseline_name="always_atm_straddle",
        structure="atm_straddle",
        entry_date=date(2026, 4, 24),
        earnings_date=date(2026, 4, 25),
        selector_structure="otm_strangle",
        entry_mid=5.0,
        modeled_cost_pct=1.0,
        execution_penalty_at_entry=0.04,
        data_quality_score_at_entry=0.8,
        iv_rv_har_at_entry=0.92,
        iv_rv_yz_at_entry=0.94,
        quote_source_at_entry="marketdata_app",
        quote_quality_at_entry="marketdata_app_paper_research_mid_not_execution_grade",
        evidence_quality_status="degraded_evidence",
        evidence_quality_reasons=["quote_not_execution_grade"],
        claim_allowed=False,
        execution_grade=False,
        entry_execution_scenarios={
            "scenario_values": {"mid": 5.0, "cross_50": 5.3},
            "spread_as_pct_of_premium": 12.0,
        },
        surface_quality={
            "status": "degraded_surface",
            "warning_flags": ["extreme_bid_ask_spreads"],
            "extreme_spread_count": 2,
        },
    )
    baselines.update_exit(
        baseline_id=make_baseline_id("rec-1", "always_atm_straddle"),
        exit_date=date(2026, 4, 24),
        exit_mid=5.25,
        realized_return_pct=4.0,
        realized_expansion_pct=5.0,
        quote_source_at_exit="marketdata_app",
        quote_quality_at_exit="marketdata_app_paper_research_mid_not_execution_grade",
    )

    report = build_evidence_report(baseline_store=baselines, outcome_store=outcomes)

    assert report["selector_summary"]["n"] == 1
    assert report["selector_summary"]["avg_realized_return_pct"] == 18.0
    assert report["baseline_comparison"]["always_atm_straddle"]["avg_realized_return_pct"] == 4.0
    assert report["baseline_comparison"]["no_trade"]["avg_realized_return_pct"] == 0.0
    assert report["simple_iv_rv_filter"]["n"] == 1
    assert report["commercialization_gate"]["ready_for_paid_beta"] is False
    assert report["maturity"]["maturity_label"] in {"Insufficient evidence", "Early observational"}
    assert report["maturity"]["edge_quality_label_allowed"] is False
    assert report["maturity"]["benchmark_comparison_meaningful"] is False
    assert any("withheld" in warning.lower() for warning in report["warning_flags"])
    assert report["evidence_quality"]["degraded_count"] == 2
    assert report["evidence_quality"]["claim_allowed_count"] == 0
    assert report["execution_realism"]["selector_entry_scenario_rows"] == 1
    assert report["execution_realism"]["avg_entry_spread_as_pct_of_premium"] == 16.0
    assert report["surface_quality"]["sparse_atm_count"] == 1
    assert report["surface_quality"]["extreme_spread_count"] == 2
    assert "No execution-grade claimable evidence yet; keep results labeled paper/research." in report["warning_flags"]
    assert "Some option surfaces contain extreme bid/ask spreads." in report["warning_flags"]

    weekly = build_weekly_evidence_report(
        baseline_store=baselines,
        outcome_store=outcomes,
        data_quality_diagnostics={"warning_flags": ["Low data-quality recommendation rate is elevated."]},
        provider_telemetry_diagnostics={"operational_health": {"warning_flags": ["Provider failure rate is elevated."]}},
    )

    assert weekly["report_type"] == "weekly_evidence_report"
    assert weekly["forward_recommendations"]["resolved_outcome_count"] == 1
    assert weekly["resolved_outcomes"]["selector_summary"]["n"] == 1
    assert weekly["claimable_vs_non_claimable"]["claimable_evidence"]["claimable_count"] == 0
    assert weekly["evidence_quality_breakdown"]["degraded_count"] == 2
    assert weekly["surface_quality_breakdown"]["extreme_spread_count"] == 2
    assert "benchmark_comparison" in weekly
    assert "Provider failure rate is elevated." in weekly["provider_data_quality_warnings"]["provider_telemetry"]
    assert "Low data-quality recommendation rate is elevated." in weekly["provider_data_quality_warnings"]["data_quality"]
    assert any("withheld" in warning.lower() for warning in weekly["sample_size_warnings"])
    assert any("not trading advice" in note for note in weekly["notes"])
