from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

from services.baseline_evidence_store import BaselineEvidenceStore
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger, record_recommendation


def _analysis(
    *,
    symbol: str,
    recommendation: str = "Candidate",
    structure: str | None = "atm_straddle",
    confidence_pct: float = 72.0,
    expected_return_pct: float = 2.5,
    data_quality_score: float = 0.82,
    stale: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        recommendation=recommendation,
        setup_score=confidence_pct / 100.0,
        metrics={},
        rationale=["diagnostic fixture"],
        selector_output={
            "recommendation": recommendation,
            "best_structure": structure,
            "confidence_pct": confidence_pct,
            "expected_return_pct": expected_return_pct,
            "earnings_date": "2026-05-01",
            "primary_thesis": "Fixture thesis.",
            "primary_risks": ["Fixture risk."],
            "why_this_structure": ["Fixture evidence."],
            "why_not_others": {},
        },
        structure_scorecards=[
            {"structure": structure or "none", "eligible": structure is not None, "composite_structure_score": 0.78}
        ],
        vol_snapshot={
            "symbol": symbol,
            "as_of_date": "2026-04-24",
            "earnings_date": "2026-05-01",
            "earnings_source_primary": "alpha_vantage",
            "earnings_source_confidence": 0.84 if not stale else 0.45,
            "earnings_source_stale": stale,
            "option_source": "marketdata_app",
            "underlying_source": "yfinance",
            "data_quality_score": data_quality_score,
        },
    )


def _record_and_resolve(
    *,
    ledger: RecommendationLedger,
    outcomes: OutcomeStore,
    rec_id: str,
    symbol: str,
    structure: str,
    realized_return_pct: float,
    realized_expansion_pct: float,
    expected_return_pct: float = 2.0,
    confidence_pct: float = 72.0,
    data_quality_score: float = 0.82,
    stale: bool = False,
    claim_allowed: bool = False,
    iv_rv_har: float = 0.98,
) -> None:
    record_recommendation(
        _analysis(
            symbol=symbol,
            structure=structure,
            confidence_pct=confidence_pct,
            expected_return_pct=expected_return_pct,
            data_quality_score=data_quality_score,
            stale=stale,
        ),
        ledger=ledger,
        recommendation_id=rec_id,
        quote_payload={"quote_quality": "paper_research_mid_not_execution_grade"},
    )
    trade_id = f"{symbol}|2026-04-24|{structure}"
    outcomes.insert_entry(
        trade_id=trade_id,
        recommendation_id=rec_id,
        symbol=symbol,
        structure=structure,
        entry_date=date(2026, 4, 24),
        setup_score=confidence_pct / 100.0,
        source_type="paper",
        selector_recommendation="Candidate",
        selector_confidence_pct=confidence_pct,
        expected_return_pct=expected_return_pct,
        data_quality_score_at_entry=data_quality_score,
        iv_rv_har=iv_rv_har,
        earnings_date=date(2026, 5, 1),
        entry_mid=4.0,
        evidence_quality_status="degraded_evidence",
        evidence_quality_reasons=["quote_not_execution_grade"],
        claim_allowed=claim_allowed,
        execution_grade=False,
        surface_quality={
            "status": "degraded_surface" if data_quality_score < 0.5 else "clean_surface",
            "warning_flags": ["extreme_bid_ask_spreads"] if data_quality_score < 0.5 else [],
            "extreme_spread_count": 1 if data_quality_score < 0.5 else 0,
        },
        entry_execution_scenarios={
            "scenario_values": {"mid": 4.0, "cross_50": 4.4},
            "spread_as_pct_of_premium": 10.0 if data_quality_score >= 0.5 else 32.0,
        },
    )
    outcomes.update_exit(
        trade_id=trade_id,
        exit_date=date(2026, 4, 30),
        exit_mid=4.4,
        realized_return_pct=realized_return_pct,
        realized_expansion_pct=realized_expansion_pct,
        exit_execution_scenarios={
            "scenario_values": {"mid": 4.4, "cross_50": 4.0},
            "scenario_outcomes": {
                "realized_return_pct": {
                    "mid": realized_return_pct,
                    "cross_50": realized_return_pct - 2.0,
                }
            },
        },
    )


def _record_baseline(
    *,
    baselines: BaselineEvidenceStore,
    recommendation_id: str,
    symbol: str,
    baseline_name: str,
    structure: str,
    realized_return_pct: float,
) -> None:
    baselines.insert_entry(
        recommendation_id=recommendation_id,
        symbol=symbol,
        baseline_name=baseline_name,
        structure=structure,
        entry_date=date(2026, 4, 24),
        earnings_date=date(2026, 5, 1),
        selector_structure="otm_strangle",
        entry_mid=4.0,
        modeled_cost_pct=0.0,
        execution_penalty_at_entry=0.05,
        data_quality_score_at_entry=0.82,
        iv_rv_har_at_entry=0.98,
        iv_rv_yz_at_entry=0.95,
        quote_source_at_entry="marketdata_app",
        quote_quality_at_entry="marketdata_app_paper_research_mid_not_execution_grade",
        entry_execution_scenarios={
            "scenario_values": {"mid": 4.0, "cross_50": 4.4},
            "spread_as_pct_of_premium": 10.0,
        },
        surface_quality={"status": "clean_surface", "warning_flags": []},
        evidence_quality_status="degraded_evidence",
        claim_allowed=False,
        execution_grade=False,
    )
    baseline_id = f"{recommendation_id}|baseline|{baseline_name}"
    baselines.update_exit(
        baseline_id=baseline_id,
        exit_date=date(2026, 4, 30),
        exit_mid=4.2,
        realized_return_pct=realized_return_pct,
        realized_expansion_pct=5.0,
        quote_source_at_exit="marketdata_app",
        quote_quality_at_exit="marketdata_app_paper_research_mid_not_execution_grade",
        exit_execution_scenarios={
            "scenario_outcomes": {
                "realized_return_pct": {
                    "mid": realized_return_pct,
                    "cross_50": realized_return_pct - 1.0,
                }
            }
        },
    )


def test_forward_performance_aggregates_outcomes_by_structure_quality_and_bucket(tmp_path: Path) -> None:
    ledger = RecommendationLedger(tmp_path / "ledger.sqlite")
    outcomes = OutcomeStore(tmp_path / "outcomes.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    _record_and_resolve(
        ledger=ledger,
        outcomes=outcomes,
        rec_id="rec_win",
        symbol="AAPL",
        structure="atm_straddle",
        realized_return_pct=4.0,
        realized_expansion_pct=8.5,
        expected_return_pct=2.0,
        confidence_pct=82.0,
        data_quality_score=0.91,
        claim_allowed=True,
    )
    _record_and_resolve(
        ledger=ledger,
        outcomes=outcomes,
        rec_id="rec_loss",
        symbol="MSFT",
        structure="atm_straddle",
        realized_return_pct=-1.5,
        realized_expansion_pct=-2.0,
        expected_return_pct=1.0,
        confidence_pct=55.0,
        data_quality_score=0.42,
        stale=True,
        iv_rv_har=1.18,
    )
    _record_and_resolve(
        ledger=ledger,
        outcomes=outcomes,
        rec_id="rec_strangle",
        symbol="NVDA",
        structure="otm_strangle",
        realized_return_pct=3.0,
        realized_expansion_pct=6.0,
        expected_return_pct=2.5,
        confidence_pct=68.0,
        data_quality_score=0.76,
    )
    record_recommendation(
        _analysis(symbol="TSLA", recommendation="No Trade", structure=None, data_quality_score=0.35),
        ledger=ledger,
        recommendation_id="rec_no_trade",
    )
    _record_baseline(
        baselines=baselines,
        recommendation_id="rec_win",
        symbol="AAPL",
        baseline_name="always_atm_straddle",
        structure="atm_straddle",
        realized_return_pct=1.5,
    )
    _record_baseline(
        baselines=baselines,
        recommendation_id="rec_win",
        symbol="AAPL",
        baseline_name="always_otm_strangle",
        structure="otm_strangle",
        realized_return_pct=2.2,
    )

    payload = build_forward_performance_diagnostics(
        ledger=ledger,
        outcome_store=outcomes,
        baseline_store=baselines,
    )

    assert payload["evidence_label"] == "paper_research_not_execution_grade"
    assert payload["total_recommendations"] == 4
    assert payload["no_trade_count"] == 1
    assert payload["resolved_outcome_count"] == 3
    assert payload["performance_summary"]["wins"] == 2
    assert payload["performance_summary"]["losses"] == 1
    assert round(payload["performance_summary"]["avg_realized_return_pct"], 4) == 1.8333
    assert payload["by_structure"]["atm_straddle"]["n"] == 2
    assert payload["by_structure"]["otm_strangle"]["wins"] == 1
    assert payload["stale_source_comparison"]["stale_source"]["losses"] == 1
    assert payload["data_quality_comparison"]["low_quality"]["n"] == 1
    assert payload["data_quality_comparison"]["high_quality"]["n"] == 2
    assert payload["evidence_quality_comparison"]["degraded_evidence"]["n"] == 3
    assert payload["surface_quality_comparison"]["degraded_surface"]["n"] == 1
    assert payload["spread_cost_buckets"]["30pct_plus_spread_cost"]["n"] == 1
    assert payload["execution_scenario_comparison"]["mid"]["n"] == 3
    assert payload["claimable_evidence"]["claimable_count"] == 1
    assert payload["claimable_performance"]["claimable"]["n"] == 1
    assert payload["claimable_performance"]["non_claimable"]["n"] == 2
    assert payload["outcome_count_by_symbol"]["AAPL"] == 1
    assert any(row["bucket"] == "75-90" and row["n"] == 1 for row in payload["calibration_buckets"])
    assert payload["calibration_report"]["selector_score_bucket"]["0.75-0.90"]["n"] == 1
    assert payload["calibration_report"]["selected_structure"]["atm_straddle"]["n"] == 2
    assert payload["calibration_report"]["evidence_quality_status"]["degraded_evidence"]["n"] == 3
    assert payload["calibration_report"]["surface_quality_status"]["degraded_surface"]["n"] == 1
    assert payload["calibration_report"]["spread_cost_bucket"]["30pct_plus_spread_cost"]["n"] == 1
    assert payload["calibration_report"]["earnings_source_confidence_bucket"]["0.00-0.50"]["n"] == 1
    assert payload["performance_summary"]["median_realized_return_pct"] == 3.0
    assert payload["performance_summary"]["small_sample_warning"] is True
    assert payload["benchmark_comparison"]["selector"]["n"] == 3
    assert payload["benchmark_comparison"]["always_atm_straddle"]["n"] == 1
    assert payload["benchmark_comparison"]["always_otm_strangle"]["avg_realized_return_pct"] == 2.2
    assert payload["benchmark_comparison"]["no_trade"]["avg_realized_return_pct"] == 0.0
    assert payload["benchmark_comparison"]["simple_iv_rv_filter"]["skipped_by_filter"] == 1
    assert payload["benchmark_comparison"]["liquidity_only_filter"]["n"] == 2
    assert payload["benchmark_comparison"]["clean_surface_only_filter"]["n"] == 2
    assert any("paper/research" in warning for warning in payload["warning_flags"])
    assert payload["recent_resolved_outcomes"][0]["quote_quality"] == "paper_research_mid_not_execution_grade"
    assert "paper/research" in payload["benchmark_comparison"]["paper_research_label"]


def test_forward_performance_empty_system_is_safe(tmp_path: Path) -> None:
    ledger = RecommendationLedger(tmp_path / "ledger.sqlite")
    outcomes = OutcomeStore(tmp_path / "outcomes.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    payload = build_forward_performance_diagnostics(
        ledger=ledger,
        outcome_store=outcomes,
        baseline_store=baselines,
    )

    assert payload["resolved_outcome_count"] == 0
    assert payload["performance_summary"]["win_rate"] is None
    assert payload["calibration_buckets"][0]["n"] == 0
    assert payload["calibration_report"]["selector_score_bucket"] == {}
    assert payload["benchmark_comparison"]["no_trade"]["n"] == 0
    assert "No resolved paper outcomes are available yet." in payload["warning_flags"]
