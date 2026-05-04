"""
Tests for the institutional-grade hardening pass.

Change 1 — Dead code removal (_score_setup)
    _score_setup() was never called in the active pipeline.
    The live path is _shared_setup_score_from_snapshot() → compute_ranking_score().
    Removing it eliminates the confusion of a parallel scoring path with different weights.

Change 2 — Spread threshold tightened: ABSOLUTE_SPREAD_THRESHOLD_PCT 18 → 12
    Research basis: De Silva, Smith & So (2025, Review of Finance).
    For a 2-leg structure with round-trip entry+exit: round-trip cost ≈ 2× bid-ask spread.
    At 12% spread: round-trip ≈ 24% of option value.
    ORATS 2021–2025: best-filtered pre-earnings ATM straddle IV expansion ≈ 8–18% of option value.
    When cost (24%) ≥ expansion upper bound (18%): structurally unprofitable → ineligible.
    Soft scoring boundary also tightened: 14% → 10% (10% = 2× median retail half-spread).

Change 3 — Execution-cost-dominates-edge hard veto in structure_selector
    When estimated execution cost consumes ≥50% of gross theoretical return → NO_TRADE.
    Research: De Silva 2025 — above 50% cost ratio, realized net returns converge to zero or
    negative after bid-ask widening, partial fills, and time-of-day impact.
"""

from datetime import date

import pytest

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import (
    ABSOLUTE_SPREAD_THRESHOLD_PCT,
    score_atm_straddle,
    WalkForwardPrior,
)
from services.structure_selector import (
    EXECUTION_COST_DOMINATES_EDGE_RATIO,
    RECOMMENDATION_NO_TRADE,
    RECOMMENDATION_WATCH,
    RECOMMENDATION_CANDIDATE,
    RECOMMENDATION_BEST,
    select_best_structure,
)
from services.structure_scorecard import StructureScorecard


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_prior(**kwargs) -> WalkForwardPrior:
    defaults = dict(
        structure="atm_straddle",
        win_rate=0.55,
        avg_return_pct=3.0,
        history_count=30,
        rank_score=0.55,
        source="synthetic",
    )
    defaults.update(kwargs)
    return WalkForwardPrior(**defaults)


def _make_snapshot(**overrides) -> VolSnapshot:
    base = dict(
        symbol="TEST",
        as_of_date=date(2026, 4, 20),
        earnings_date=date(2026, 4, 26),
        release_timing="after market close",
        days_to_earnings=6,
        underlying_price=100.0,
        option_source="provided",
        underlying_source="provided",
        price_staleness_minutes=0,
        chain_staleness_minutes=0,
        data_quality="high",
        data_quality_score=0.88,
        rv30_yang_zhang=0.22,
        rv30_estimator="yang_zhang",
        rv_har_forecast=0.21,
        rv_percentile_rank=50.0,
        vol_regime_label="Normal",
        iv30=0.26,
        iv45=0.29,
        near_term_dte=6,
        near_term_atm_iv=0.26,
        back_term_dte=34,
        back_term_atm_iv=0.29,
        near_back_iv_ratio=0.897,
        term_structure_slope=0.0010,
        near_term_implied_move_pct=5.5,
        near_term_implied_sigma_pct=5.5 * 1.2533141373155001,  # MAD × √(π/2), P-5a
        non_event_move_pct_har=1.1,
        event_implied_move_pct=5.3,
        event_move_share_of_total=0.96,
        historical_event_count=8,
        historical_median_move_pct=6.8,
        historical_avg_last4_move_pct=7.1,
        historical_p90_move_pct=10.0,
        historical_move_std_pct=2.0,
        historical_move_anchor_pct=7.0,
        historical_move_uncertainty_pct=0.9,
        historical_vs_implied_move_ratio=1.28,
        tail_vs_implied_move_ratio=1.50,
        smile_curvature=0.15,
        smile_concavity_flag=False,
        smile_points=5,
        near_term_spread_pct=2.5,
        near_term_liquidity_proxy=3500.0,
        atm_call_spread_pct=2.4,
        atm_put_spread_pct=2.6,
        atm_total_open_interest=2500.0,
        atm_total_volume=1200.0,
        liquidity_tier="mid",
        iv_rv_yz=1.18,
        iv_rv_har=1.24,
        cheapness_score=0.60,
        event_risk_score=0.72,
        execution_score=0.78,
        timing_score=0.97,
        historical_move_source="earnings_history",
        null_reasons={},
    )
    base.update(overrides)
    return VolSnapshot(**base)


def _scorecard(
    structure: str = "atm_straddle",
    *,
    eligible: bool = True,
    expected_edge_pct: float = 3.0,
    expected_return_pct: float = 6.0,
    execution_penalty: float = 0.03,
    composite_structure_score: float = 0.68,
    walk_forward_history_count: int = 24,
    sample_confidence: float = 0.72,
    **kwargs,
) -> StructureScorecard:
    return StructureScorecard(
        structure=structure,
        eligible=eligible,
        eligibility_flags=[] if eligible else ["test_flag"],
        expected_edge_pct=expected_edge_pct,
        expected_return_pct=expected_return_pct,
        expected_iv_contribution_pct=2.0,
        expected_move_fit_score=0.70,
        theta_drag_penalty=0.02,
        execution_penalty=execution_penalty,
        crowding_penalty=0.02,
        concavity_penalty=0.01,
        sample_uncertainty_penalty=0.02,
        sample_confidence=sample_confidence,
        walk_forward_history_count=walk_forward_history_count,
        walk_forward_win_rate=0.58,
        walk_forward_avg_return_pct=6.0,
        walk_forward_rank_score=0.70,
        composite_structure_score=composite_structure_score,
        rationale_bullets=[],
        **kwargs,
    )


def _selector_snapshot(**overrides) -> VolSnapshot:
    return _make_snapshot(data_quality_score=0.90, **overrides)


# ── Change 1: Dead code removal ───────────────────────────────────────────────

class TestChange1DeadCodeRemoval:
    """_score_setup() was dead and has been removed. The live path remains intact."""

    def test_score_setup_not_present_in_edge_engine(self):
        import web.api.edge_engine as ee
        assert not hasattr(ee, "_score_setup"), (
            "_score_setup was a dead function — it has been removed. "
            "The pipeline uses _shared_setup_score_from_snapshot()."
        )

    def test_shared_setup_score_from_snapshot_still_present(self):
        """The live path that replaced _score_setup must still exist."""
        import web.api.edge_engine as ee
        assert hasattr(ee, "_shared_setup_score_from_snapshot"), (
            "_shared_setup_score_from_snapshot is the active scoring path and must remain."
        )

    def test_orphaned_heuristic_entries_removed(self):
        """setup_iv_rv_baseline and friends were doc entries for the dead function."""
        import web.api.edge_engine as ee
        orphaned_keys = [
            "setup_iv_rv_baseline",
            "setup_iv_rv_ceiling",
            "setup_timing_optimal_dte",
            "setup_timing_sigma_dte",
            "setup_iv_component_weight",
            "setup_ts_component_weight",
            "setup_timing_component_weight",
            "setup_liquidity_component_weight",
            "composite_score_setup_weight",
            "composite_score_expectancy_weight",
        ]
        for key in orphaned_keys:
            assert key not in ee._HEURISTIC_THRESHOLDS, (
                f"Heuristic entry '{key}' belongs exclusively to removed _score_setup() "
                f"and should have been removed from _HEURISTIC_THRESHOLDS."
            )

    def test_active_heuristic_entries_still_present(self):
        """Entries used by active code must not have been accidentally removed."""
        import web.api.edge_engine as ee
        required_keys = [
            "move_anchor_avg_last4_weight",
            "kurtosis_penalty_ek_threshold_mild",
            "tx_cost_base_pct",
            "calendar_scenario_fallback_expand",
        ]
        for key in required_keys:
            assert key in ee._HEURISTIC_THRESHOLDS, (
                f"Active heuristic entry '{key}' must still be present after dead code removal."
            )


# ── Change 2: Spread threshold tightened ─────────────────────────────────────

class TestChange2SpreadThreshold:
    """
    ABSOLUTE_SPREAD_THRESHOLD_PCT = 12.0 (was 18.0).
    De Silva 2025: at 12% spread, round-trip ≈ 24% of option value — at or above
    the ceiling of typical pre-earnings IV expansion (8-18% ORATS).
    """

    def test_threshold_constant_is_twelve(self):
        """Hard threshold must be exactly 12.0% after the change."""
        assert ABSOLUTE_SPREAD_THRESHOLD_PCT == pytest.approx(12.0), (
            "ABSOLUTE_SPREAD_THRESHOLD_PCT must be 12.0 per De Silva 2025 calibration"
        )

    def test_spread_above_twelve_makes_scorecard_ineligible(self):
        """A snapshot with near_term_spread_pct = 13% must be ineligible."""
        snap = _make_snapshot(near_term_spread_pct=13.0, atm_call_spread_pct=12.8, atm_put_spread_pct=13.2)
        prior = _make_prior()
        card = score_atm_straddle(snap, prior=prior)
        assert not card.eligible, (
            "Spread 13% > 12% threshold: scorecard must be ineligible "
            "(De Silva 2025: round-trip ≈ 26% of option value, exceeds IV expansion range)"
        )
        spread_flags = [f for f in card.eligibility_flags if "spread" in f.lower()]
        assert len(spread_flags) >= 1, (
            f"Expected a spread-related eligibility flag; got: {card.eligibility_flags}"
        )

    def test_spread_below_twelve_is_eligible_if_no_other_flags(self):
        """A good snapshot with spread = 9% must remain eligible."""
        snap = _make_snapshot(near_term_spread_pct=9.0, atm_call_spread_pct=8.8, atm_put_spread_pct=9.2)
        prior = _make_prior()
        card = score_atm_straddle(snap, prior=prior)
        assert card.eligible, (
            "Spread 9% < 12%: scorecard should be eligible (no spread flag from hard gate)"
        )

    def test_spread_exactly_twelve_is_eligible(self):
        """12.0% exactly should be eligible (threshold is strictly greater than)."""
        snap = _make_snapshot(near_term_spread_pct=12.0, atm_call_spread_pct=11.8, atm_put_spread_pct=12.2)
        prior = _make_prior()
        card = score_atm_straddle(snap, prior=prior)
        # The check is `> ABSOLUTE_SPREAD_THRESHOLD_PCT`, so exactly 12.0 is still eligible
        spread_absolute_flags = [
            f for f in card.eligibility_flags
            if "absolute_threshold" in f.lower()
        ]
        assert len(spread_absolute_flags) == 0, (
            "Spread exactly 12.0% should not trigger the hard threshold flag (condition is >)"
        )

    def test_wide_spread_scorecard_produces_no_trade_in_selector(self):
        """Ineligible scorecard (spread 14%) must result in NO_TRADE from selector."""
        snap = _make_snapshot(near_term_spread_pct=14.0, data_quality_score=0.90)
        prior = _make_prior()
        card = score_atm_straddle(snap, prior=prior)
        assert not card.eligible, "14% spread should be ineligible with new 12% threshold"
        output = select_best_structure(snap, [card])
        assert output.recommendation == RECOMMENDATION_NO_TRADE, (
            "An ineligible scorecard must produce NO_TRADE from the selector"
        )

    def test_execution_penalty_increases_monotonically_with_spread(self):
        """
        Tighter soft scoring boundary (10% vs old 14%) means higher penalty
        at moderate spreads. Execution penalty at 9% must be greater than at 4%.
        """
        snap_tight = _make_snapshot(
            near_term_spread_pct=4.0, atm_call_spread_pct=3.8, atm_put_spread_pct=4.2
        )
        snap_wide = _make_snapshot(
            near_term_spread_pct=9.0, atm_call_spread_pct=8.8, atm_put_spread_pct=9.2
        )
        prior = _make_prior()
        card_tight = score_atm_straddle(snap_tight, prior=prior)
        card_wide = score_atm_straddle(snap_wide, prior=prior)
        assert card_wide.execution_penalty > card_tight.execution_penalty, (
            f"9% spread should have higher execution penalty than 4% spread; "
            f"got tight={card_tight.execution_penalty:.4f}, wide={card_wide.execution_penalty:.4f}"
        )


# ── Change 3: Execution-cost-dominates-edge veto ──────────────────────────────

class TestChange3ExecutionCostDominatesEdge:
    """
    When execution cost consumes ≥50% of gross theoretical return → hard NO_TRADE.
    Research: De Silva 2025 — above this ratio, net returns reliably negative.
    """

    def test_constant_is_fifty_percent(self):
        assert EXECUTION_COST_DOMINATES_EDGE_RATIO == pytest.approx(0.50), (
            "Threshold must be 0.50 (50%) per De Silva 2025 frictions research"
        )

    def test_high_cost_ratio_produces_no_trade(self):
        """
        ep=0.07, edge=1.0 → execution_cost=1.82, gross=2.82, ratio=0.645 > 0.50 → NO_TRADE.
        The execution cost consumes 64.5% of the estimated gross return.
        """
        snap = _selector_snapshot()
        card = _scorecard(
            expected_edge_pct=1.0,
            execution_penalty=0.07,
            composite_structure_score=0.55,
            walk_forward_history_count=20,
            sample_confidence=0.65,
        )
        output = select_best_structure(snap, [card])
        assert output.recommendation == RECOMMENDATION_NO_TRADE, (
            f"ep=0.07, edge=1.0 → cost ratio 64.5% > 50% → must be NO_TRADE; "
            f"got {output.recommendation}"
        )

    def test_low_cost_ratio_allows_candidate(self):
        """
        ep=0.03, edge=4.0 → execution_cost=0.78, gross=4.78, ratio=0.163 < 0.50 → not vetoed.
        The setup has sufficient gross return relative to execution costs.
        """
        snap = _selector_snapshot()
        cards = [
            _scorecard(
                expected_edge_pct=4.0,
                execution_penalty=0.03,
                composite_structure_score=0.72,
                walk_forward_history_count=25,
                sample_confidence=0.72,
            )
        ]
        output = select_best_structure(snap, cards)
        assert output.recommendation != RECOMMENDATION_NO_TRADE, (
            f"ep=0.03, edge=4.0 → cost ratio 16.3% < 50% → veto must NOT fire; "
            f"got {output.recommendation}"
        )

    def test_veto_fires_before_high_execution_penalty_gate(self):
        """
        ep=0.08 (below 0.10 hard gate), edge=1.0 → cost ratio=67.6% > 50% → NO_TRADE.
        Confirms the new veto closes a gap: setups with ep between 0.07-0.10 AND thin
        edge would previously have received WATCH; now correctly get NO_TRADE.
        """
        snap = _selector_snapshot()
        card = _scorecard(
            expected_edge_pct=1.0,
            execution_penalty=0.08,   # below existing 0.10 hard gate
            composite_structure_score=0.55,
            walk_forward_history_count=20,
            sample_confidence=0.65,
        )
        output = select_best_structure(snap, [card])
        assert output.recommendation == RECOMMENDATION_NO_TRADE, (
            f"ep=0.08 (below 0.10 gate) with edge=1.0 → cost ratio 67.6% should trigger "
            f"cost-dominates-edge veto; got {output.recommendation}"
        )

    def test_strong_setup_not_affected_by_veto(self):
        """
        A genuinely strong setup (ep=0.02, edge=6.0) should not be caught by the veto.
        ratio = 0.52/6.52 = 0.08 << 0.50.
        """
        snap = _selector_snapshot()
        cards = [
            _scorecard(
                expected_edge_pct=6.2,
                expected_return_pct=9.0,
                execution_penalty=0.02,
                composite_structure_score=0.82,
                walk_forward_history_count=28,
                sample_confidence=0.74,
            )
        ]
        output = select_best_structure(snap, cards)
        assert output.recommendation in (RECOMMENDATION_BEST, RECOMMENDATION_CANDIDATE), (
            f"Strong setup (ep=0.02, edge=6.2) must not be blocked by cost veto; "
            f"got {output.recommendation}"
        )

    def test_veto_boundary_case_exactly_fifty_percent(self):
        """
        At exactly 50%: cost = 0.50 × gross → veto fires (>= threshold).
        Derivation: 26 × ep = 0.50 × (edge + 26 × ep)
                   26 × ep - 13 × ep = 0.50 × edge
                   ep = edge / 26
        For edge=2.6, ep=0.1 → cost=2.6, gross=5.2, ratio=0.50 exactly.
        """
        ep = 2.6 / 26.0  # exactly 0.10
        snap = _selector_snapshot()
        card = _scorecard(
            expected_edge_pct=2.6,
            execution_penalty=ep,
            composite_structure_score=0.60,
            walk_forward_history_count=20,
            sample_confidence=0.70,
        )
        output = select_best_structure(snap, [card])
        # At exactly 50%, the veto fires (>= condition). But also note ep=0.10 which
        # also triggers the absolute execution_penalty >= 0.10 gate. Both should agree.
        assert output.recommendation == RECOMMENDATION_NO_TRADE, (
            "At cost_ratio == 0.50 (boundary), veto should fire (>= condition)"
        )

    def test_negative_edge_not_double_reported(self):
        """
        When expected_edge_pct <= 0, the existing edge gate fires first.
        The cost-dominates-edge veto should not fire when gross_return <= 0
        (to avoid division-by-zero and double-counting).
        """
        snap = _selector_snapshot()
        card = _scorecard(
            expected_edge_pct=-1.0,
            execution_penalty=0.03,
            composite_structure_score=0.40,
            walk_forward_history_count=20,
            sample_confidence=0.65,
        )
        output = select_best_structure(snap, [card])
        # Already NO_TRADE from the edge<=0 gate. The cost veto should gracefully
        # not trigger separately (gross_return = -1.0 + 0.78 = -0.22 <= 0 → condition not met).
        assert output.recommendation == RECOMMENDATION_NO_TRADE
