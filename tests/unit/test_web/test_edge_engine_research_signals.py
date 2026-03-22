import unittest

import numpy as np
import pandas as pd
from datetime import datetime

from web.api.edge_engine import (
    _compute_move_uncertainty_pct,
    _compute_move_anchor,
    _historical_earnings_move_profile,
    _normalize_release_timing,
)


class TestEdgeEngineResearchSignals(unittest.TestCase):
    def test_compute_move_anchor_blends_recent_and_median(self):
        anchor = _compute_move_anchor(median_move_pct=4.0, avg_last4_move_pct=6.0)
        self.assertAlmostEqual(anchor, 5.3, places=6)

    def test_compute_move_anchor_fallbacks(self):
        self.assertAlmostEqual(_compute_move_anchor(np.nan, 5.0), 5.0, places=6)
        self.assertAlmostEqual(_compute_move_anchor(4.5, np.nan), 4.5, places=6)
        self.assertIsNone(_compute_move_anchor(np.nan, np.nan))

    def test_historical_profile_returns_recent_avg_field(self):
        """Profile with AMC-tagged events → source='earnings_history'."""
        dates = pd.bdate_range("2024-01-02", periods=140)
        prices = pd.Series(100.0 + np.linspace(0, 1.0, len(dates)), index=dates)

        earnings_positions = [20, 40, 60, 80, 100]
        target_moves = [4.0, 5.0, 6.0, 7.0, 8.0]
        earnings_events = [
            {"event_date": pd.Timestamp(dates[i]), "release_timing": "after market close"}
            for i in earnings_positions
        ]
        for pos, move in zip(earnings_positions, target_moves):
            pre_idx = pos
            post_idx = pos + 1
            pre_px = float(prices.iloc[pre_idx])
            prices.iloc[post_idx] = pre_px * (1.0 + move / 100.0)

        profile = _historical_earnings_move_profile(
            close=prices,
            earnings_events=earnings_events,
        )

        self.assertEqual(profile["source"], "earnings_history")
        self.assertEqual(profile["event_count"], 5)
        self.assertIn("avg_last4_move_pct", profile)
        self.assertIn("std_move_pct", profile)
        # Last 4 target moves are 5,6,7,8.
        self.assertAlmostEqual(profile["avg_last4_move_pct"], 6.5, places=1)
        self.assertGreater(profile["p90_move_pct"], profile["median_move_pct"])
        self.assertGreater(profile["std_move_pct"], 0.0)

    def test_historical_profile_bmo_alignment_uses_prev_to_event_close(self):
        dates = pd.bdate_range("2024-01-02", periods=80)
        prices = pd.Series(np.linspace(100.0, 104.0, len(dates)), index=dates)
        event_pos = 30
        target_move = 6.0
        pre_idx = event_pos - 1
        post_idx = event_pos
        pre_px = float(prices.iloc[pre_idx])
        prices.iloc[post_idx] = pre_px * (1.0 + target_move / 100.0)

        profile = _historical_earnings_move_profile(
            close=prices,
            earnings_events=[
                {"event_date": pd.Timestamp(dates[event_pos]), "release_timing": "before market open"}
            ],
        )
        self.assertEqual(profile["source"], "earnings_history")
        self.assertEqual(profile["event_count"], 1)
        self.assertAlmostEqual(profile["median_move_pct"], target_move, places=2)

    def test_historical_profile_daily_fallback_has_avg_last4(self):
        """Empty earnings_dates list → falls back to daily move profile."""
        dates = pd.bdate_range("2024-01-02", periods=80)
        prices = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)

        # Pass empty list to trigger daily_fallback path
        profile = _historical_earnings_move_profile(
            close=prices,
            earnings_events=[],
        )

        self.assertEqual(profile["source"], "daily_fallback")
        self.assertGreater(profile["event_count"], 0)
        self.assertIsNotNone(profile["avg_last4_move_pct"])
        self.assertIn("std_move_pct", profile)

    def test_move_uncertainty_scales_with_source_quality(self):
        earnings_uncertainty = _compute_move_uncertainty_pct(
            move_std_pct=2.0,
            sample_size=8,
            move_source="earnings_history",
        )
        fallback_uncertainty = _compute_move_uncertainty_pct(
            move_std_pct=2.0,
            sample_size=8,
            move_source="daily_fallback",
        )
        self.assertIsNotNone(earnings_uncertainty)
        self.assertIsNotNone(fallback_uncertainty)
        self.assertGreater(fallback_uncertainty, earnings_uncertainty)

    def test_normalize_release_timing_infers_bmo_from_timestamp(self):
        inferred = _normalize_release_timing(datetime(2024, 1, 15, 8, 0))
        self.assertEqual(inferred, "before market open")

    def test_normalize_release_timing_infers_amc_from_timestamp(self):
        inferred = _normalize_release_timing(datetime(2024, 1, 15, 16, 5))
        self.assertEqual(inferred, "after market close")

    def test_normalize_release_timing_infers_intraday_from_timestamp(self):
        inferred = _normalize_release_timing(datetime(2024, 1, 15, 12, 0))
        self.assertEqual(inferred, "during market hours")


if __name__ == "__main__":
    unittest.main()
