import unittest
from datetime import date
from pathlib import Path
import time
from unittest.mock import patch

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import (
    WalkForwardPrior,
    build_structure_scorecards,
    reload_walk_forward_priors,
    score_atm_straddle,
)


def _base_snapshot(**overrides) -> VolSnapshot:
    payload = dict(
        symbol="AAPL",
        as_of_date=date(2026, 4, 20),
        earnings_date=date(2026, 4, 28),
        release_timing="after market close",
        days_to_earnings=8,
        underlying_price=100.0,
        option_source="provided",
        underlying_source="provided",
        price_staleness_minutes=5,
        chain_staleness_minutes=5,
        data_quality="high",
        data_quality_score=0.92,
        rv30_yang_zhang=0.21,
        rv30_estimator="yang_zhang",
        rv_har_forecast=0.20,
        rv_percentile_rank=48.0,
        vol_regime_label="Normal",
        iv30=0.24,
        iv45=0.27,
        near_term_dte=4,
        near_term_atm_iv=0.23,
        back_term_dte=25,
        back_term_atm_iv=0.27,
        near_back_iv_ratio=0.87,
        term_structure_slope=0.0025,
        near_term_implied_move_pct=5.8,
        near_term_implied_sigma_pct=5.8 * 1.2533141373155001,  # MAD × √(π/2), P-5a
        non_event_move_pct_har=1.2,
        event_implied_move_pct=5.67,
        event_move_share_of_total=0.88,
        historical_event_count=8,
        historical_median_move_pct=7.2,
        historical_avg_last4_move_pct=7.6,
        historical_p90_move_pct=10.0,
        historical_move_std_pct=2.1,
        historical_move_anchor_pct=7.4,
        historical_move_uncertainty_pct=0.85,
        historical_vs_implied_move_ratio=1.28,
        tail_vs_implied_move_ratio=1.72,
        smile_curvature=0.18,
        smile_concavity_flag=False,
        smile_points=7,
        near_term_spread_pct=2.5,
        near_term_liquidity_proxy=4200.0,
        atm_call_spread_pct=2.4,
        atm_put_spread_pct=2.6,
        atm_total_open_interest=3200.0,
        atm_total_volume=1600.0,
        liquidity_tier="high",
        iv_rv_yz=1.14,
        iv_rv_har=1.20,
        cheapness_score=0.62,
        event_risk_score=0.74,
        execution_score=0.86,
        timing_score=0.78,
        historical_move_source="earnings_history",
        null_reasons={},
    )
    payload.update(overrides)
    return VolSnapshot(**payload)


def _neutral_priors() -> dict[str, WalkForwardPrior]:
    return {
        structure: WalkForwardPrior(
            structure=structure,
            history_count=20,
            win_rate=0.52,
            avg_return_pct=2.0,
            rank_score=0.50,
            source="test_neutral",
        )
        for structure in ("atm_straddle", "otm_strangle", "call_calendar", "put_calendar")
    }


class TestStructureScorecards(unittest.TestCase):
    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_structure_differentiation_by_regime(self, _mock_priors):
        straddle_snapshot = _base_snapshot(
            historical_vs_implied_move_ratio=1.72,
            tail_vs_implied_move_ratio=1.95,
            historical_move_anchor_pct=8.8,
            event_implied_move_pct=6.2,
            cheapness_score=0.56,
            event_risk_score=0.70,
            term_structure_slope=0.0008,
            near_back_iv_ratio=0.95,
            iv_rv_yz=0.98,
            iv_rv_har=1.00,
        )
        straddle_cards = build_structure_scorecards(straddle_snapshot)
        self.assertEqual(
            max(straddle_cards, key=lambda card: card.composite_structure_score).structure,
            "atm_straddle",
        )

        calendar_snapshot = _base_snapshot(
            cheapness_score=0.95,
            term_structure_slope=0.0034,
            near_back_iv_ratio=0.82,
            event_move_share_of_total=0.58,
            historical_move_anchor_pct=5.6,
            historical_vs_implied_move_ratio=0.96,
            tail_vs_implied_move_ratio=1.05,
            event_risk_score=0.48,
            iv_rv_yz=0.84,
            iv_rv_har=0.86,
            atm_call_spread_pct=1.8,
            atm_put_spread_pct=1.9,
        )
        calendar_cards = build_structure_scorecards(calendar_snapshot)
        self.assertIn(
            max(calendar_cards, key=lambda card: card.composite_structure_score).structure,
            {"call_calendar", "put_calendar"},
        )

        strangle_snapshot = _base_snapshot(
            tail_vs_implied_move_ratio=2.30,
            event_risk_score=0.94,
            historical_move_anchor_pct=10.4,
            historical_vs_implied_move_ratio=1.08,
            cheapness_score=0.44,
            near_back_iv_ratio=0.92,
            term_structure_slope=0.0016,
            iv_rv_yz=1.02,
            iv_rv_har=1.05,
        )
        strangle_cards = build_structure_scorecards(strangle_snapshot)
        self.assertEqual(
            max(strangle_cards, key=lambda card: card.composite_structure_score).structure,
            "otm_strangle",
        )

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_poor_spreads_make_all_structures_ineligible(self, _mock_priors):
        snapshot = _base_snapshot(
            near_term_spread_pct=25.0,
            atm_call_spread_pct=22.0,
            atm_put_spread_pct=23.0,
            execution_score=0.20,
        )
        cards = build_structure_scorecards(snapshot)

        self.assertTrue(all(not card.eligible for card in cards))
        self.assertTrue(all(card.composite_structure_score == 0.0 for card in cards))
        self.assertTrue(all("spread_exceeds_absolute_threshold" in card.eligibility_flags for card in cards))

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_increasing_spread_worsens_execution_penalty(self, _mock_priors):
        tight = _base_snapshot(near_term_spread_pct=2.0, atm_call_spread_pct=2.0, atm_put_spread_pct=2.1, execution_score=0.90)
        wide = _base_snapshot(near_term_spread_pct=10.0, atm_call_spread_pct=9.8, atm_put_spread_pct=10.2, execution_score=0.55)

        tight_card = score_atm_straddle(tight)
        wide_card = score_atm_straddle(wide)

        self.assertGreater(wide_card.execution_penalty, tight_card.execution_penalty)
        self.assertLess(wide_card.composite_structure_score, tight_card.composite_structure_score)

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_increasing_iv_rv_worsens_straddle_score(self, _mock_priors):
        cheap = _base_snapshot(iv_rv_yz=0.90, iv_rv_har=0.92, cheapness_score=0.84)
        expensive = _base_snapshot(iv_rv_yz=1.55, iv_rv_har=1.58, cheapness_score=0.18)

        cheap_card = score_atm_straddle(cheap)
        expensive_card = score_atm_straddle(expensive)

        self.assertGreater(expensive_card.crowding_penalty, cheap_card.crowding_penalty)
        self.assertLess(expensive_card.composite_structure_score, cheap_card.composite_structure_score)

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_concavity_penalizes_straddle(self, _mock_priors):
        flat = _base_snapshot(smile_curvature=0.12, smile_concavity_flag=False)
        concave = _base_snapshot(smile_curvature=0.78, smile_concavity_flag=True)

        flat_card = score_atm_straddle(flat)
        concave_card = score_atm_straddle(concave)

        self.assertGreater(concave_card.concavity_penalty, flat_card.concavity_penalty)
        self.assertLess(concave_card.composite_structure_score, flat_card.composite_structure_score)

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_poor_liquidity_penalizes_all_structures(self, _mock_priors):
        liquid = _base_snapshot(near_term_liquidity_proxy=4200.0, atm_total_open_interest=3200.0, atm_total_volume=1600.0, execution_score=0.88)
        illiquid = _base_snapshot(near_term_liquidity_proxy=180.0, atm_total_open_interest=120.0, atm_total_volume=60.0, execution_score=0.30)

        liquid_cards = {card.structure: card for card in build_structure_scorecards(liquid)}
        illiquid_cards = {card.structure: card for card in build_structure_scorecards(illiquid)}

        for structure in liquid_cards:
            self.assertGreater(illiquid_cards[structure].execution_penalty, liquid_cards[structure].execution_penalty)
            self.assertLess(illiquid_cards[structure].composite_structure_score, liquid_cards[structure].composite_structure_score)

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_scorecards_are_deterministic_for_same_snapshot(self, _mock_priors):
        snapshot = _base_snapshot()
        first = [card.to_dict() for card in build_structure_scorecards(snapshot)]
        second = [card.to_dict() for card in build_structure_scorecards(snapshot)]
        self.assertEqual(first, second)


class TestWalkForwardPriorCache(unittest.TestCase):
    def _fake_prior(self, structure: str, counter: dict[str, int]) -> WalkForwardPrior:
        counter["calls"] += 1
        return WalkForwardPrior(
            structure=structure,
            history_count=counter["calls"],
            win_rate=0.50,
            avg_return_pct=1.0,
            rank_score=0.50,
            source="test_cache",
        )

    def test_mtime_cache_reuses_unchanged_file_and_reloads_on_change(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "marker.json"
            marker.write_text("{}")
            counter = {"calls": 0}

            def _signature(as_of_date=None):
                return ((str(marker), marker.stat().st_mtime),)

            def _calendar(structure: str) -> WalkForwardPrior:
                return self._fake_prior(structure, counter)

            def _straddle() -> WalkForwardPrior:
                return self._fake_prior("atm_straddle", counter)

            def _strangle() -> WalkForwardPrior:
                return self._fake_prior("otm_strangle", counter)

            reload_walk_forward_priors()
            with patch("services.structure_scorecard._walk_forward_prior_signature", side_effect=_signature), \
                 patch("services.structure_scorecard._load_calendar_prior_from_reports", side_effect=_calendar), \
                 patch("services.structure_scorecard._load_straddle_prior_from_reports", side_effect=_straddle), \
                 patch("services.structure_scorecard._load_strangle_prior_from_reports", side_effect=_strangle), \
                 patch("services.structure_prior_store.load_all_structure_priors", return_value={}):
                from services import structure_scorecard as sc

                first = sc._load_walk_forward_priors()
                second = sc._load_walk_forward_priors()
                self.assertIs(first, second)
                self.assertEqual(counter["calls"], 4)

                time.sleep(0.02)
                os.utime(marker, None)
                third = sc._load_walk_forward_priors()
                self.assertIsNot(first, third)
                self.assertEqual(counter["calls"], 8)

                sc.reload_walk_forward_priors()
                fourth = sc._load_walk_forward_priors()
                self.assertEqual(counter["calls"], 12)
                self.assertIsNot(third, fourth)


if __name__ == "__main__":
    unittest.main()
