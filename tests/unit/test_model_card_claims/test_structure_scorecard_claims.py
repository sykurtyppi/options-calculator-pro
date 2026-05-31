"""
Phase 2.3 — Model card claim audit: structure_scorecard.

The card's Validation Approach section states:
  "Tests cover structure differentiation, monotonic penalties, no-trade eligibility,
   determinism, and low-sample behavior."

Each function below is named after one of those claims and exercises it directly.
If the claim becomes false (the property breaks), CI fails here before any reader
has to audit the narrative card.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import (
    WalkForwardPrior,
    build_structure_scorecards,
    score_atm_straddle,
    score_otm_strangle,
)
from services.structure_prior_store import LAPLACE_SMOOTHING_THRESHOLD


def _neutral_priors() -> dict:
    """Return a no-information WalkForwardPrior for every supported structure."""
    return {
        s: WalkForwardPrior(structure=s, history_count=0, win_rate=0.50,
                            avg_return_pct=0.0, rank_score=0.50,
                            source="neutral_test_fixture")
        for s in ("atm_straddle", "otm_strangle", "call_calendar", "put_calendar")
    }


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


def test_structure_differentiation() -> None:
    """Cheap options (low IV/RV → low crowding penalty) score better for straddle; scores differ across structures."""
    # High iv_rv_har drives crowding_intensity up → bigger crowding_penalty → lower composite score.
    snap_expensive = _snapshot(iv_rv_yz=1.40, iv_rv_har=1.45, iv30=0.30,
                               historical_vs_implied_move_ratio=1.55)
    snap_cheap = _snapshot(iv_rv_yz=0.85, iv_rv_har=0.80, iv30=0.18,
                           historical_vs_implied_move_ratio=1.10)
    with patch("services.structure_scorecard._load_walk_forward_priors", return_value=_neutral_priors()):
        straddle_expensive = score_atm_straddle(snap_expensive).composite_structure_score
        straddle_cheap = score_atm_straddle(snap_cheap).composite_structure_score
        strangle_expensive = score_otm_strangle(snap_expensive).composite_structure_score
    # Cheaper options (lower IV/RV) reduce crowding penalty → higher straddle score
    assert straddle_cheap > straddle_expensive, (
        "straddle should score better when IV/RV is low (less crowding penalty)"
    )
    # Scores differ across structures for the same snapshot
    assert straddle_expensive != strangle_expensive, "structures must differentiate for the same snapshot"


def test_monotonic_penalties() -> None:
    """Increasing bid-ask spread monotonically increases execution_penalty."""
    with patch("services.structure_scorecard._load_walk_forward_priors", return_value=_neutral_priors()):
        penalties = [
            score_atm_straddle(_snapshot(near_term_spread_pct=s)).execution_penalty
            for s in [1.0, 3.0, 6.0, 10.0]
        ]
    assert penalties == sorted(penalties), (
        f"execution_penalty must be non-decreasing as spread grows; got {penalties}"
    )


def test_no_trade_eligibility() -> None:
    """Extreme bid-ask spread makes all structures ineligible (eligible=False on all cards)."""
    snap = _snapshot(near_term_spread_pct=35.0, atm_call_spread_pct=34.0,
                     atm_put_spread_pct=35.0)
    with patch("services.structure_scorecard._load_walk_forward_priors", return_value=_neutral_priors()):
        cards = build_structure_scorecards(snap)
    assert all(not card.eligible for card in cards), (
        "all structures should be ineligible with a 35% spread"
    )


def test_determinism() -> None:
    """build_structure_scorecards() returns identical composite scores on repeated calls."""
    snap = _snapshot()
    with patch("services.structure_scorecard._load_walk_forward_priors", return_value=_neutral_priors()):
        scores_a = [c.composite_structure_score for c in build_structure_scorecards(snap)]
        scores_b = [c.composite_structure_score for c in build_structure_scorecards(snap)]
    assert scores_a == scores_b, "scorecard output must be deterministic for the same snapshot"


def test_low_sample_behavior() -> None:
    """Below LAPLACE_SMOOTHING_THRESHOLD observations, win_rate is Laplace-smoothed, not raw."""
    from services.structure_prior_store import StructurePriorStore
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as td:
        prior_path = pathlib.Path(td) / "priors.json"
        ps = StructurePriorStore(store_path=prior_path)
        # Add 5 wins (>= MIN_OBS_FOR_OVERRIDE=5, still below LAPLACE threshold=10)
        # → Laplace-smoothed: (5+1)/(5+2) ≈ 0.857, not raw 1.0
        for i in range(5):
            ps.update(
                structure="atm_straddle",
                realized_return_pct=5.0,
                realized_expansion_pct=2.0,
                source_type="replay",
                observation_date=date(2024, 1, i + 1),
                observation_id=f"low-sample-{i}",
            )
        d = ps.get_prior_dict("atm_straddle")
    assert d is not None, "5 obs >= MIN_OBS_FOR_OVERRIDE should return a dict"
    assert d["win_rate"] < 1.0, (
        f"Laplace smoothing should pull win_rate below 1.0 for 5 wins below threshold, got {d['win_rate']}"
    )
    expected = (5 + 1) / (5 + 2)
    assert abs(d["win_rate"] - expected) < 1e-9
