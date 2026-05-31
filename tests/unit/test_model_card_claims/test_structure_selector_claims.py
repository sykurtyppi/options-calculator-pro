"""
Phase 2.3 — Model card claim audit: structure_selector.

The card's Validation Approach section states:
  "Tests cover clear winners, close scores, negative edge, execution failure,
   conflicting signals, and deterministic output."

Each function below is named after one of those claims and exercises it directly.
"""

from __future__ import annotations

from datetime import date

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import StructureScorecard
from services.structure_selector import (
    DOMINANT_SCORE_GAP,
    HIGH_EXECUTION_PENALTY_THRESHOLD,
    MARGINAL_EDGE_THRESHOLD_PCT,
    MIN_WALK_FORWARD_HISTORY,
    MODERATE_SCORE_GAP,
    RECOMMENDATION_BEST,
    RECOMMENDATION_CANDIDATE,
    RECOMMENDATION_NO_TRADE,
    RECOMMENDATION_WATCH,
    STRONG_EDGE_THRESHOLD_PCT,
    select_best_structure,
)


def _snapshot(**overrides) -> VolSnapshot:
    base = dict(
        symbol="AAPL", as_of_date=date(2026, 1, 15), earnings_date=date(2026, 1, 22),
        release_timing="after market close", days_to_earnings=7,
        underlying_price=100.0, option_source="provided", underlying_source="provided",
        price_staleness_minutes=0, chain_staleness_minutes=0,
        data_quality="high", data_quality_score=0.90,
        rv30_yang_zhang=0.21, rv30_estimator="yang_zhang", rv_har_forecast=0.20,
        rv_percentile_rank=50.0, vol_regime_label="Normal",
        iv30=0.25, iv45=0.28,
        near_term_dte=4, near_term_atm_iv=0.24,
        back_term_dte=25, back_term_atm_iv=0.28,
        near_back_iv_ratio=0.88, term_structure_slope=0.0024,
        near_term_implied_move_pct=5.9,
        near_term_implied_sigma_pct=5.9 * 1.2533141373155001,
        non_event_move_pct_har=1.2, event_implied_move_pct=5.78,
        event_move_share_of_total=0.88, historical_event_count=8,
        historical_median_move_pct=7.4, historical_avg_last4_move_pct=7.8,
        historical_p90_move_pct=10.2, historical_move_std_pct=2.0,
        historical_move_anchor_pct=7.6, historical_move_uncertainty_pct=0.8,
        historical_vs_implied_move_ratio=1.30, tail_vs_implied_move_ratio=1.72,
        smile_curvature=0.18, smile_concavity_flag=False, smile_points=6,
        near_term_spread_pct=2.4, near_term_liquidity_proxy=4200.0,
        atm_call_spread_pct=2.3, atm_put_spread_pct=2.5,
        atm_total_open_interest=2800.0, atm_total_volume=1400.0,
        liquidity_tier="high", iv_rv_yz=1.12, iv_rv_har=1.18,
        cheapness_score=0.64, event_risk_score=0.74,
        execution_score=0.86, timing_score=0.78,
        historical_move_source="earnings_history", null_reasons={},
    )
    base.update(overrides)
    return VolSnapshot(**base)


def _card(structure: str, **overrides) -> StructureScorecard:
    base = dict(
        eligible=True, eligibility_flags=[],
        expected_edge_pct=3.0, expected_return_pct=6.0,
        expected_iv_contribution_pct=2.0, expected_move_fit_score=0.70,
        theta_drag_penalty=0.02, execution_penalty=0.03,
        crowding_penalty=0.02, concavity_penalty=0.01,
        sample_uncertainty_penalty=0.02, sample_confidence=0.72,
        walk_forward_history_count=24, walk_forward_win_rate=0.58,
        walk_forward_avg_return_pct=6.0, walk_forward_rank_score=0.70,
        composite_structure_score=0.68, rationale_bullets=[],
    )
    base.update(overrides)
    return StructureScorecard(structure=structure, **base)


def test_clear_winners() -> None:
    """A dominant score gap with strong edge and clean execution → BEST or CANDIDATE."""
    snap = _snapshot(data_quality_score=0.90)
    cards = [
        _card("atm_straddle", composite_structure_score=0.85,
              expected_edge_pct=STRONG_EDGE_THRESHOLD_PCT + 1.0,
              execution_penalty=0.02, sample_confidence=0.75,
              walk_forward_history_count=25),
        _card("otm_strangle", composite_structure_score=0.85 - DOMINANT_SCORE_GAP - 0.01,
              expected_edge_pct=2.0),
        _card("call_calendar", composite_structure_score=0.50, expected_edge_pct=1.5),
        _card("put_calendar", composite_structure_score=0.45, expected_edge_pct=1.0),
    ]
    result = select_best_structure(snap, cards)
    assert result.recommendation in (RECOMMENDATION_BEST, RECOMMENDATION_CANDIDATE), (
        f"clear dominant score with strong edge should be BEST or CANDIDATE, got {result.recommendation}"
    )


def test_close_scores() -> None:
    """Gap below MODERATE_SCORE_GAP triggers WATCH regardless of individual score."""
    snap = _snapshot(data_quality_score=0.90)
    gap = MODERATE_SCORE_GAP / 2
    cards = [
        _card("atm_straddle", composite_structure_score=0.70, expected_edge_pct=3.5),
        _card("otm_strangle", composite_structure_score=0.70 - gap, expected_edge_pct=3.0),
        _card("call_calendar", composite_structure_score=0.50, expected_edge_pct=1.5),
        _card("put_calendar", composite_structure_score=0.40, expected_edge_pct=1.0),
    ]
    result = select_best_structure(snap, cards)
    assert result.recommendation == RECOMMENDATION_WATCH, (
        f"scores within MODERATE_SCORE_GAP should yield WATCH, got {result.recommendation}"
    )


def test_negative_edge() -> None:
    """expected_edge_pct <= 0 forces NO_TRADE regardless of other fields."""
    snap = _snapshot()
    cards = [
        _card("atm_straddle", expected_edge_pct=-0.5, composite_structure_score=0.75),
        _card("otm_strangle", expected_edge_pct=-1.0, composite_structure_score=0.60),
        _card("call_calendar", expected_edge_pct=-0.1, composite_structure_score=0.50),
        _card("put_calendar", expected_edge_pct=-2.0, composite_structure_score=0.40),
    ]
    result = select_best_structure(snap, cards)
    assert result.recommendation == RECOMMENDATION_NO_TRADE


def test_execution_failure() -> None:
    """execution_penalty >= HIGH_EXECUTION_PENALTY_THRESHOLD forces NO_TRADE."""
    snap = _snapshot()
    cards = [
        _card("atm_straddle", execution_penalty=HIGH_EXECUTION_PENALTY_THRESHOLD,
              expected_edge_pct=5.0, composite_structure_score=0.80),
        _card("otm_strangle", execution_penalty=HIGH_EXECUTION_PENALTY_THRESHOLD + 0.01,
              expected_edge_pct=4.0),
        _card("call_calendar", execution_penalty=HIGH_EXECUTION_PENALTY_THRESHOLD + 0.02,
              expected_edge_pct=3.0),
        _card("put_calendar", execution_penalty=HIGH_EXECUTION_PENALTY_THRESHOLD + 0.03,
              expected_edge_pct=2.0),
    ]
    result = select_best_structure(snap, cards)
    assert result.recommendation == RECOMMENDATION_NO_TRADE


def test_conflicting_signals() -> None:
    """Thin walk-forward evidence (few obs + low confidence) triggers WATCH even with a dominant gap.

    The thin_evidence arm of _has_conflicting_signals fires when
    walk_forward_history_count < MIN_WALK_FORWARD_HISTORY and sample_confidence < 0.55.
    All other WATCH conditions are deliberately suppressed:
      - gap > DOMINANT_SCORE_GAP  → no gap-band WATCH, no weak_history veto
      - all penalties < WATCH_EXECUTION_PENALTY_THRESHOLD → no top_penalty WATCH
    So WATCH is caused exclusively by conflicting signals, not a secondary condition.
    """
    snap = _snapshot(data_quality_score=0.90)
    cards = [
        _card("atm_straddle", composite_structure_score=0.80, expected_edge_pct=4.0,
              execution_penalty=0.03, concavity_penalty=0.02, crowding_penalty=0.02,
              walk_forward_history_count=MIN_WALK_FORWARD_HISTORY - 2,
              sample_confidence=0.50),
        _card("otm_strangle",
              composite_structure_score=0.80 - DOMINANT_SCORE_GAP - 0.05,
              expected_edge_pct=2.5),
        _card("call_calendar", composite_structure_score=0.50, expected_edge_pct=1.5),
        _card("put_calendar", composite_structure_score=0.40, expected_edge_pct=1.0),
    ]
    result = select_best_structure(snap, cards)
    assert result.recommendation == RECOMMENDATION_WATCH, (
        f"thin walk-forward evidence (history < {MIN_WALK_FORWARD_HISTORY}, confidence < 0.55) "
        f"should suppress CANDIDATE to WATCH; got {result.recommendation}"
    )


def test_deterministic_output() -> None:
    """Repeated calls with the same inputs produce byte-identical recommendation."""
    snap = _snapshot()
    cards = [
        _card("atm_straddle", composite_structure_score=0.72, expected_edge_pct=4.0),
        _card("otm_strangle", composite_structure_score=0.58, expected_edge_pct=2.5),
        _card("call_calendar", composite_structure_score=0.50, expected_edge_pct=1.8),
        _card("put_calendar", composite_structure_score=0.43, expected_edge_pct=1.2),
    ]
    r1 = select_best_structure(snap, cards)
    r2 = select_best_structure(snap, cards)
    assert r1.recommendation == r2.recommendation
    assert r1.best_structure == r2.best_structure
    assert abs(r1.confidence_pct - r2.confidence_pct) < 1e-9
