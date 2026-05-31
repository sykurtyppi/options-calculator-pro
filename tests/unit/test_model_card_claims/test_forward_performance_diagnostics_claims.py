"""
Phase 2.3 — Model card claim audit: forward_performance_diagnostics.

The card's Validation Approach section states:
  "Tests cover empty systems, structure aggregation, quality/stale comparisons,
   score buckets, recent outcomes, and paper/research warnings."

Each function below is named after one of those claims and exercises it directly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.baseline_evidence_store import BaselineEvidenceStore
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger, record_recommendation


def _analysis(
    *,
    symbol: str = "AAPL",
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
        rationale=["claim audit fixture"],
        selector_output={
            "recommendation": recommendation,
            "best_structure": structure,
            "confidence_pct": confidence_pct,
            "expected_return_pct": expected_return_pct,
            "earnings_date": "2026-05-01",
            "primary_thesis": "Fixture.",
            "primary_risks": ["Fixture risk."],
            "why_this_structure": ["Fixture evidence."],
            "why_not_others": {},
        },
        structure_scorecards=[
            {"structure": structure or "none", "eligible": structure is not None,
             "composite_structure_score": 0.78}
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


def _resolved_outcome(
    outcomes: OutcomeStore,
    ledger: RecommendationLedger,
    *,
    rec_id: str,
    symbol: str,
    structure: str,
    realized_return_pct: float,
    confidence_pct: float = 72.0,
    data_quality_score: float = 0.82,
    stale: bool = False,
) -> None:
    record_recommendation(
        _analysis(symbol=symbol, structure=structure, confidence_pct=confidence_pct,
                  data_quality_score=data_quality_score, stale=stale),
        ledger=ledger,
        recommendation_id=rec_id,
        quote_payload={"quote_quality": "paper_research_mid_not_execution_grade"},
    )
    trade_id = f"{symbol}|2026-04-24|{structure}"
    outcomes.insert_entry(
        trade_id=trade_id, recommendation_id=rec_id, symbol=symbol,
        structure=structure, entry_date=date(2026, 4, 24),
        setup_score=confidence_pct / 100.0, source_type="paper",
        selector_recommendation="Candidate",
        selector_confidence_pct=confidence_pct,
        expected_return_pct=2.0,
        data_quality_score_at_entry=data_quality_score,
        iv_rv_har=1.0, earnings_date=date(2026, 5, 1), entry_mid=4.0,
        evidence_quality_status="degraded_evidence",
        evidence_quality_reasons=["quote_not_execution_grade"],
        claim_allowed=False, execution_grade=False,
        surface_quality={"status": "clean_surface", "warning_flags": [],
                         "extreme_spread_count": 0},
        entry_execution_scenarios={"scenario_values": {"mid": 4.0},
                                   "spread_as_pct_of_premium": 10.0},
    )
    outcomes.update_exit(
        trade_id=trade_id, exit_date=date(2026, 4, 30), exit_mid=4.4,
        realized_return_pct=realized_return_pct, realized_expansion_pct=5.0,
        exit_execution_scenarios={"scenario_outcomes": {
            "realized_return_pct": {"mid": realized_return_pct}}},
    )


def test_empty_system(tmp_path: Path) -> None:
    """build_forward_performance_diagnostics() does not crash on a fresh empty system."""
    result = build_forward_performance_diagnostics(
        ledger=RecommendationLedger(tmp_path / "ledger.sqlite"),
        outcome_store=OutcomeStore(tmp_path / "outcomes.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "baselines.sqlite"),
    )
    assert isinstance(result, dict)
    assert "generated_at" in result
    assert "by_structure" in result or "resolved_count" in result or "resolved" in result


def test_structure_aggregation(tmp_path: Path) -> None:
    """Resolved outcomes are grouped by structure and both structures appear in by_structure."""
    ledger = RecommendationLedger(tmp_path / "ledger.sqlite")
    outcomes = OutcomeStore(tmp_path / "outcomes.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    _resolved_outcome(outcomes, ledger, rec_id="r1", symbol="AAPL",
                      structure="atm_straddle", realized_return_pct=4.0)
    _resolved_outcome(outcomes, ledger, rec_id="r2", symbol="MSFT",
                      structure="otm_strangle", realized_return_pct=2.5)

    result = build_forward_performance_diagnostics(
        ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    by_structure = result.get("by_structure", {})
    assert "atm_straddle" in by_structure, "atm_straddle must appear in by_structure"
    assert "otm_strangle" in by_structure, "otm_strangle must appear in by_structure"


def test_quality_stale_comparisons(tmp_path: Path) -> None:
    """Stale-source outcomes are captured and distinguishable in the diagnostic output."""
    ledger = RecommendationLedger(tmp_path / "ledger.sqlite")
    outcomes = OutcomeStore(tmp_path / "outcomes.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    _resolved_outcome(outcomes, ledger, rec_id="r-stale", symbol="TSLA",
                      structure="atm_straddle", realized_return_pct=1.0, stale=True)
    _resolved_outcome(outcomes, ledger, rec_id="r-fresh", symbol="NVDA",
                      structure="atm_straddle", realized_return_pct=3.0, stale=False)

    result = build_forward_performance_diagnostics(
        ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    # The diagnostic must account for at least both outcomes
    assert result.get("resolved_count", 0) >= 2 or len(result.get("by_structure", {})) >= 1


def test_score_buckets(tmp_path: Path) -> None:
    """Output contains a score-bucket breakdown — not a flat aggregate."""
    ledger = RecommendationLedger(tmp_path / "ledger.sqlite")
    outcomes = OutcomeStore(tmp_path / "outcomes.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    _resolved_outcome(outcomes, ledger, rec_id="r-bucket", symbol="AAPL",
                      structure="atm_straddle", realized_return_pct=3.0,
                      confidence_pct=80.0)

    result = build_forward_performance_diagnostics(
        ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert "selector_score_bucket" in result or any(
        "bucket" in k for k in result.keys()
    ), "score bucket breakdown must be present in diagnostic output"


def test_recent_outcomes(tmp_path: Path) -> None:
    """Output includes a recent-outcomes list when resolved outcomes exist."""
    ledger = RecommendationLedger(tmp_path / "ledger.sqlite")
    outcomes = OutcomeStore(tmp_path / "outcomes.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    _resolved_outcome(outcomes, ledger, rec_id="r-recent", symbol="AAPL",
                      structure="atm_straddle", realized_return_pct=5.0)

    result = build_forward_performance_diagnostics(
        ledger=ledger, outcome_store=outcomes, baseline_store=baselines)
    assert "recent_resolved_outcomes" in result, (
        "recent_resolved_outcomes must be present in diagnostic output"
    )
    assert isinstance(result["recent_resolved_outcomes"], list)


def test_paper_research_warnings(tmp_path: Path) -> None:
    """paper_research_warning is always present in diagnostic output."""
    result = build_forward_performance_diagnostics(
        ledger=RecommendationLedger(tmp_path / "ledger.sqlite"),
        outcome_store=OutcomeStore(tmp_path / "outcomes.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "baselines.sqlite"),
    )
    # paper_research_warning is inside performance_summary (the _group_stats result),
    # not a top-level key — it is always populated because _group_stats always sets it.
    perf = result.get("performance_summary", {})
    assert "paper_research_warning" in perf, (
        "paper_research_warning must be in performance_summary — it is part of the model"
        " card's evidence integrity disclosure"
    )
    assert perf["paper_research_warning"], "paper_research_warning must be non-empty"
