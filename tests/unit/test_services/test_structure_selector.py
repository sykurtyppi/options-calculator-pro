import unittest
from datetime import date

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import StructureScorecard
from services.structure_selector import (
    DOMINANT_SCORE_GAP,
    EXECUTION_COST_DOMINATES_EDGE_RATIO,
    HIGH_EXECUTION_PENALTY_THRESHOLD,
    MIN_DATA_QUALITY_FOR_ANY_TRADE,
    MIN_WALK_FORWARD_HISTORY,
    RECOMMENDATION_BEST,
    RECOMMENDATION_NO_TRADE,
    RECOMMENDATION_WATCH,
    select_best_structure,
)


def _snapshot(**overrides) -> VolSnapshot:
    payload = dict(
        symbol="AAPL",
        as_of_date=date(2026, 4, 20),
        earnings_date=date(2026, 4, 28),
        release_timing="after market close",
        days_to_earnings=8,
        underlying_price=100.0,
        option_source="provided",
        underlying_source="provided",
        price_staleness_minutes=0,
        chain_staleness_minutes=0,
        data_quality="high",
        data_quality_score=0.90,
        rv30_yang_zhang=0.21,
        rv30_estimator="yang_zhang",
        rv_har_forecast=0.20,
        rv_percentile_rank=50.0,
        vol_regime_label="Normal",
        iv30=0.25,
        iv45=0.28,
        near_term_dte=4,
        near_term_atm_iv=0.24,
        back_term_dte=25,
        back_term_atm_iv=0.28,
        near_back_iv_ratio=0.88,
        term_structure_slope=0.0024,
        near_term_implied_move_pct=5.9,
        near_term_implied_sigma_pct=5.9 * 1.2533141373155001,  # MAD × √(π/2), P-5a
        non_event_move_pct_har=1.2,
        event_implied_move_pct=5.78,
        event_move_share_of_total=0.88,
        historical_event_count=8,
        historical_median_move_pct=7.4,
        historical_avg_last4_move_pct=7.8,
        historical_p90_move_pct=10.2,
        historical_move_std_pct=2.0,
        historical_move_anchor_pct=7.6,
        historical_move_uncertainty_pct=0.8,
        historical_vs_implied_move_ratio=1.30,
        tail_vs_implied_move_ratio=1.72,
        smile_curvature=0.18,
        smile_concavity_flag=False,
        smile_points=6,
        near_term_spread_pct=2.4,
        near_term_liquidity_proxy=4200.0,
        atm_call_spread_pct=2.3,
        atm_put_spread_pct=2.5,
        atm_total_open_interest=2800.0,
        atm_total_volume=1400.0,
        liquidity_tier="high",
        iv_rv_yz=1.12,
        iv_rv_har=1.18,
        cheapness_score=0.64,
        event_risk_score=0.74,
        execution_score=0.86,
        timing_score=0.78,
        historical_move_source="earnings_history",
        null_reasons={},
    )
    payload.update(overrides)
    return VolSnapshot(**payload)


def _scorecard(
    structure: str,
    *,
    eligible: bool = True,
    eligibility_flags: list[str] | None = None,
    expected_edge_pct: float = 3.0,
    expected_return_pct: float = 6.0,
    expected_iv_contribution_pct: float = 2.0,
    expected_move_fit_score: float = 0.70,
    theta_drag_penalty: float = 0.02,
    execution_penalty: float = 0.03,
    crowding_penalty: float = 0.02,
    concavity_penalty: float = 0.01,
    sample_uncertainty_penalty: float = 0.02,
    sample_confidence: float = 0.72,
    walk_forward_history_count: int = 24,
    walk_forward_win_rate: float = 0.58,
    walk_forward_avg_return_pct: float = 6.0,
    walk_forward_rank_score: float = 0.70,
    composite_structure_score: float = 0.68,
) -> StructureScorecard:
    return StructureScorecard(
        structure=structure,
        eligible=eligible,
        eligibility_flags=eligibility_flags or [],
        expected_edge_pct=expected_edge_pct,
        expected_return_pct=expected_return_pct,
        expected_iv_contribution_pct=expected_iv_contribution_pct,
        expected_move_fit_score=expected_move_fit_score,
        theta_drag_penalty=theta_drag_penalty,
        execution_penalty=execution_penalty,
        crowding_penalty=crowding_penalty,
        concavity_penalty=concavity_penalty,
        sample_uncertainty_penalty=sample_uncertainty_penalty,
        sample_confidence=sample_confidence,
        walk_forward_history_count=walk_forward_history_count,
        walk_forward_win_rate=walk_forward_win_rate,
        walk_forward_avg_return_pct=walk_forward_avg_return_pct,
        walk_forward_rank_score=walk_forward_rank_score,
        composite_structure_score=composite_structure_score,
        rationale_bullets=[],
    )


class TestStructureSelector(unittest.TestCase):
    def test_clear_winner_returns_best_candidate(self):
        snapshot = _snapshot(historical_vs_implied_move_ratio=1.70, tail_vs_implied_move_ratio=1.95)
        cards = [
            _scorecard("atm_straddle", expected_edge_pct=6.2, expected_return_pct=9.0, composite_structure_score=0.82, execution_penalty=0.03, sample_confidence=0.74, walk_forward_history_count=28),
            _scorecard("otm_strangle", expected_edge_pct=2.8, composite_structure_score=0.61),
            _scorecard("call_calendar", expected_edge_pct=1.9, composite_structure_score=0.54),
            _scorecard("put_calendar", expected_edge_pct=1.5, composite_structure_score=0.49),
        ]

        output = select_best_structure(snapshot, cards)

        self.assertEqual(output.recommendation, RECOMMENDATION_BEST)
        self.assertEqual(output.best_structure, "atm_straddle")
        self.assertGreater(output.confidence_pct, 70.0)
        self.assertEqual(output.expected_edge_tier, "Positive")
        self.assertEqual(output.expected_return_signal, "Supportive")
        self.assertIn("not empirical return forecasts", output.model_output_note)
        self.assertTrue(
            any("score-derived diagnostics" in item for item in output.why_this_structure)
        )
        self.assertFalse(
            any("Expected edge after penalties" in item for item in output.why_this_structure)
        )

    def test_close_scores_return_watch(self):
        snapshot = _snapshot()
        cards = [
            _scorecard("atm_straddle", expected_edge_pct=2.5, composite_structure_score=0.63),
            _scorecard("otm_strangle", expected_edge_pct=2.4, composite_structure_score=0.60),
            _scorecard("call_calendar", expected_edge_pct=1.7, composite_structure_score=0.56),
        ]

        output = select_best_structure(snapshot, cards)

        self.assertEqual(output.recommendation, RECOMMENDATION_WATCH)
        self.assertEqual(output.best_structure, "atm_straddle")

    def test_negative_edge_returns_no_trade(self):
        snapshot = _snapshot()
        cards = [
            _scorecard("atm_straddle", expected_edge_pct=-0.4, composite_structure_score=0.70),
            _scorecard("otm_strangle", expected_edge_pct=-1.0, composite_structure_score=0.66),
            _scorecard("call_calendar", expected_edge_pct=-0.2, composite_structure_score=0.60),
        ]

        output = select_best_structure(snapshot, cards)

        self.assertEqual(output.recommendation, RECOMMENDATION_NO_TRADE)
        self.assertIsNone(output.best_structure)
        self.assertEqual(output.expected_edge_tier, "Negative / Unclear")
        self.assertTrue(
            any(
                "negative or unclear" in reason
                for reasons in output.why_not_others.values()
                for reason in reasons
            )
        )

    def test_execution_failure_returns_no_trade(self):
        snapshot = _snapshot(data_quality_score=0.88, near_term_spread_pct=11.0)
        cards = [
            _scorecard("atm_straddle", expected_edge_pct=4.0, composite_structure_score=0.76, execution_penalty=0.12),
            _scorecard("otm_strangle", expected_edge_pct=3.0, composite_structure_score=0.68, execution_penalty=0.11),
        ]

        output = select_best_structure(snapshot, cards)

        self.assertEqual(output.recommendation, RECOMMENDATION_NO_TRADE)
        self.assertIsNone(output.best_structure)

    def test_conflicting_signals_return_watch(self):
        snapshot = _snapshot(smile_concavity_flag=True, smile_curvature=0.80)
        cards = [
            _scorecard("atm_straddle", expected_edge_pct=3.8, composite_structure_score=0.74, expected_move_fit_score=0.78, concavity_penalty=0.08),
            _scorecard("otm_strangle", expected_edge_pct=2.7, composite_structure_score=0.58),
            _scorecard("call_calendar", expected_edge_pct=1.2, composite_structure_score=0.52),
        ]

        output = select_best_structure(snapshot, cards)

        self.assertEqual(output.recommendation, RECOMMENDATION_WATCH)
        self.assertEqual(output.best_structure, "atm_straddle")

    def test_selector_is_deterministic(self):
        snapshot = _snapshot()
        cards = [
            _scorecard("atm_straddle", expected_edge_pct=3.2, composite_structure_score=0.69),
            _scorecard("otm_strangle", expected_edge_pct=2.0, composite_structure_score=0.55),
            _scorecard("call_calendar", expected_edge_pct=1.8, composite_structure_score=0.51),
        ]

        first = select_best_structure(snapshot, cards).to_dict()
        second = select_best_structure(snapshot, cards).to_dict()
        self.assertEqual(first, second)

    # ── Phase 2.4 — no-trade gate property tests ──────────────────────────────
    # Each test toggles one gate condition across the threshold and confirms the
    # recommendation flips.  Three conditions had no existing test:
    #   (a) data_quality_score < MIN_DATA_QUALITY_FOR_ANY_TRADE
    #   (b) weak_history (walk_forward_history_count < MIN_WALK_FORWARD_HISTORY
    #         and gap < DOMINANT_SCORE_GAP)
    #   (c) execution cost dominates edge (exec_cost / gross_return >= 0.50)

    def test_low_data_quality_forces_no_trade(self):
        """data_quality_score below MIN_DATA_QUALITY_FOR_ANY_TRADE triggers NO_TRADE."""
        snap_low = _snapshot(data_quality_score=MIN_DATA_QUALITY_FOR_ANY_TRADE - 0.01)
        snap_ok = _snapshot(data_quality_score=MIN_DATA_QUALITY_FOR_ANY_TRADE + 0.10)
        cards = [
            _scorecard("atm_straddle", expected_edge_pct=5.0,
                       composite_structure_score=0.82, execution_penalty=0.02),
            _scorecard("otm_strangle", expected_edge_pct=2.5, composite_structure_score=0.58),
            _scorecard("call_calendar", expected_edge_pct=1.8, composite_structure_score=0.50),
            _scorecard("put_calendar", expected_edge_pct=1.2, composite_structure_score=0.42),
        ]
        self.assertEqual(
            select_best_structure(snap_low, cards).recommendation,
            RECOMMENDATION_NO_TRADE,
        )
        self.assertNotEqual(
            select_best_structure(snap_ok, cards).recommendation,
            RECOMMENDATION_NO_TRADE,
        )

    def test_weak_history_with_small_gap_forces_no_trade(self):
        """walk_forward_history_count below MIN_WALK_FORWARD_HISTORY and small gap → NO_TRADE."""
        gap = DOMINANT_SCORE_GAP / 2
        cards_thin = [
            _scorecard("atm_straddle", expected_edge_pct=3.5,
                       composite_structure_score=0.70,
                       walk_forward_history_count=MIN_WALK_FORWARD_HISTORY - 1),
            _scorecard("otm_strangle", expected_edge_pct=2.5,
                       composite_structure_score=0.70 - gap),
            _scorecard("call_calendar", expected_edge_pct=1.5, composite_structure_score=0.50),
            _scorecard("put_calendar", expected_edge_pct=1.0, composite_structure_score=0.40),
        ]
        cards_thick = [
            _scorecard("atm_straddle", expected_edge_pct=3.5,
                       composite_structure_score=0.70,
                       walk_forward_history_count=MIN_WALK_FORWARD_HISTORY),
            _scorecard("otm_strangle", expected_edge_pct=2.5,
                       composite_structure_score=0.70 - gap),
            _scorecard("call_calendar", expected_edge_pct=1.5, composite_structure_score=0.50),
            _scorecard("put_calendar", expected_edge_pct=1.0, composite_structure_score=0.40),
        ]
        snap = _snapshot()
        self.assertEqual(
            select_best_structure(snap, cards_thin).recommendation,
            RECOMMENDATION_NO_TRADE,
        )
        self.assertNotEqual(
            select_best_structure(snap, cards_thick).recommendation,
            RECOMMENDATION_NO_TRADE,
        )

    def test_execution_cost_dominates_edge_forces_no_trade(self):
        """When execution cost >= EXECUTION_COST_DOMINATES_EDGE_RATIO * gross return → NO_TRADE."""
        # gross_return = expected_edge_pct + exec_cost_pct; exec_cost_pct = 26 * execution_penalty
        # To trigger the veto: exec_cost >= 0.50 * gross_return
        # Choose execution_penalty so that 26 * penalty / (edge + 26 * penalty) >= 0.50
        # → 26 * penalty >= edge → penalty >= edge / 26
        edge_pct = 2.0
        penalty_triggers = (edge_pct / 26.0) + 0.01   # just above the ratio boundary
        penalty_safe = (edge_pct / 26.0) * 0.5         # well below

        snap = _snapshot()
        cards_veto = [
            _scorecard("atm_straddle", expected_edge_pct=edge_pct,
                       composite_structure_score=0.72, execution_penalty=penalty_triggers),
            _scorecard("otm_strangle", expected_edge_pct=1.0, composite_structure_score=0.55),
            _scorecard("call_calendar", expected_edge_pct=0.8, composite_structure_score=0.48),
            _scorecard("put_calendar", expected_edge_pct=0.5, composite_structure_score=0.40),
        ]
        cards_safe = [
            _scorecard("atm_straddle", expected_edge_pct=edge_pct,
                       composite_structure_score=0.72, execution_penalty=penalty_safe),
            _scorecard("otm_strangle", expected_edge_pct=1.0, composite_structure_score=0.55),
            _scorecard("call_calendar", expected_edge_pct=0.8, composite_structure_score=0.48),
            _scorecard("put_calendar", expected_edge_pct=0.5, composite_structure_score=0.40),
        ]
        self.assertEqual(
            select_best_structure(snap, cards_veto).recommendation,
            RECOMMENDATION_NO_TRADE,
        )
        self.assertNotEqual(
            select_best_structure(snap, cards_safe).recommendation,
            RECOMMENDATION_NO_TRADE,
        )


if __name__ == "__main__":
    unittest.main()
