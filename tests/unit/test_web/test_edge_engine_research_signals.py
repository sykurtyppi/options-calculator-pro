import unittest

import numpy as np
import pandas as pd

from web.api.edge_engine import (
    _compute_move_anchor,
    _historical_earnings_move_profile,
)


class _TickerWithEarnings:
    def __init__(self, earnings_dates):
        self._earnings_dates = [pd.Timestamp(d) for d in earnings_dates]

    def get_earnings_dates(self, limit=24):
        _ = limit
        return pd.DataFrame({"eps": [0.0] * len(self._earnings_dates)}, index=pd.DatetimeIndex(self._earnings_dates))


class _TickerNoEarnings:
    def get_earnings_dates(self, limit=24):
        _ = limit
        return pd.DataFrame()


class TestEdgeEngineResearchSignals(unittest.TestCase):
    def test_compute_move_anchor_blends_recent_and_median(self):
        anchor = _compute_move_anchor(median_move_pct=4.0, avg_last4_move_pct=6.0)
        self.assertAlmostEqual(anchor, 5.3, places=6)

    def test_compute_move_anchor_fallbacks(self):
        self.assertAlmostEqual(_compute_move_anchor(np.nan, 5.0), 5.0, places=6)
        self.assertAlmostEqual(_compute_move_anchor(4.5, np.nan), 4.5, places=6)
        self.assertIsNone(_compute_move_anchor(np.nan, np.nan))

    def test_historical_profile_returns_recent_avg_field(self):
        dates = pd.bdate_range("2024-01-02", periods=140)
        prices = pd.Series(100.0 + np.linspace(0, 1.0, len(dates)), index=dates)

        earnings_positions = [20, 40, 60, 80, 100]
        target_moves = [4.0, 5.0, 6.0, 7.0, 8.0]
        earnings_dates = [dates[i] for i in earnings_positions]
        for pos, move in zip(earnings_positions, target_moves):
            pre_idx = pos - 1
            post_idx = pos + 1
            pre_px = float(prices.iloc[pre_idx])
            prices.iloc[post_idx] = pre_px * (1.0 + move / 100.0)

        profile = _historical_earnings_move_profile(
            ticker=_TickerWithEarnings(earnings_dates),
            close=prices,
        )

        self.assertEqual(profile["source"], "earnings_history")
        self.assertEqual(profile["event_count"], 5)
        self.assertIn("avg_last4_move_pct", profile)
        # Last 4 target moves are 5,6,7,8.
        self.assertAlmostEqual(profile["avg_last4_move_pct"], 6.5, places=1)
        self.assertGreater(profile["p90_move_pct"], profile["median_move_pct"])

    def test_historical_profile_daily_fallback_has_avg_last4(self):
        dates = pd.bdate_range("2024-01-02", periods=80)
        prices = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)

        profile = _historical_earnings_move_profile(
            ticker=_TickerNoEarnings(),
            close=prices,
        )

        self.assertEqual(profile["source"], "daily_fallback")
        self.assertGreater(profile["event_count"], 0)
        self.assertIsNotNone(profile["avg_last4_move_pct"])


if __name__ == "__main__":
    unittest.main()
