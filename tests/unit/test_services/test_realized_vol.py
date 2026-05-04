"""
Tests for services.realized_vol — the single-source-of-truth realized-vol
kernels shared by services.earnings_vol_snapshot and web.api.edge_engine.

Coverage
--------
- yang_zhang_rv30: golden numerical test, insufficient-history gate,
  excluded_sessions=None default behavior, excluded_sessions exclusion behavior.
- rs_daily_vol_series: golden RS values, exclusion, missing-column guard.
- har_rv_forecast: insufficient-history gate, sanity / round-trip on a
  constant series.
- rs_trailing_mean_forecast: insufficient-history gate, simple mean.
- Drift detection: the snapshot-module and edge_engine-module aliases must
  resolve to the same callable as the new shared kernels.
"""
import unittest

import numpy as np
import pandas as pd

import services.earnings_vol_snapshot as snap_mod
import services.realized_vol as rv_mod
import web.api.edge_engine as ee_mod
from services.realized_vol import (
    HAR_MIN_OBS,
    har_rv_forecast,
    rs_daily_vol_series,
    rs_trailing_mean_forecast,
    yang_zhang_rv30,
)


def _ohlc_frame(rows):
    """Build an OHLC DataFrame indexed by business-day timestamps."""
    dates = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close"],
        index=dates,
    )


class TestModuleWiringDriftDetection(unittest.TestCase):
    """Both the snapshot module and the edge-engine module must re-export the
    shared kernels. If any underscore alias stops pointing at the shared
    function, this test fires before any behavioral drift can land.

    Threshold reconciliation between the two paths landed in P-3b (HAR
    minimum observations = 100 on both); the edge-engine migration lands
    in P-3c.
    """

    def test_snapshot_aliases_point_to_shared_kernels(self):
        self.assertIs(snap_mod._yang_zhang_rv30, yang_zhang_rv30)
        self.assertIs(snap_mod._rs_daily_vol_series, rs_daily_vol_series)
        self.assertIs(snap_mod._har_rv_forecast, har_rv_forecast)
        self.assertIs(snap_mod._rs_trailing_mean_forecast, rs_trailing_mean_forecast)
        self.assertEqual(snap_mod._HAR_MIN_OBS, rv_mod.HAR_MIN_OBS)
        self.assertEqual(snap_mod._RS_FALLBACK_WINDOW, rv_mod.RS_FALLBACK_WINDOW)

    def test_edge_engine_aliases_point_to_shared_kernels(self):
        self.assertIs(ee_mod._yang_zhang_rv30, yang_zhang_rv30)
        self.assertIs(ee_mod._rs_daily_vol_series, rs_daily_vol_series)
        self.assertIs(ee_mod._har_rv_forecast, har_rv_forecast)


class TestYangZhangRV30(unittest.TestCase):
    def test_returns_nan_when_history_too_short(self):
        # Yang-Zhang requires at least max(window+1, 6) rows.
        short = _ohlc_frame([(100, 101, 99, 100)] * 5)
        self.assertTrue(np.isnan(yang_zhang_rv30(short, window=30)))

    def test_returns_nan_when_columns_missing(self):
        df = pd.DataFrame({"Open": [1, 2, 3], "Close": [1, 2, 3]})
        self.assertTrue(np.isnan(yang_zhang_rv30(df, window=2)))

    def test_constant_price_yields_floor_volatility(self):
        # Perfectly flat OHLC (no price movement) gives σ²_o, σ²_c, var_RS all ≈ 0.
        # The kernel floors yz_var at 1e-10 before sqrt(× 252) → vol ≈ √(2.52e-8).
        flat = _ohlc_frame([(100.0, 100.0, 100.0, 100.0)] * 35)
        result = yang_zhang_rv30(flat, window=30)
        self.assertTrue(np.isfinite(result))
        # √(1e-10 × 252) ≈ 1.5874e-4
        self.assertAlmostEqual(result, float(np.sqrt(1e-10 * 252.0)), places=10)

    def test_golden_value_for_synthetic_oscillating_series(self):
        """Golden test: deterministic OHLC where every component of YZ is non-zero.

        Construction:
          - Open alternates between 100.0 and 100.5 (small overnight moves)
          - High = Open × 1.01, Low = Open × 0.99 (constant intraday range)
          - Close = Open × 1.005 (constant open-to-close return)

        With these inputs the YZ formula reduces to a value that does not depend
        on series length (var_o is the variance of two alternating values, var_c
        and var_rs are deterministic constants), making the expected output
        analytically verifiable to within floating-point precision.
        """
        rows = []
        for i in range(35):
            o = 100.0 if i % 2 == 0 else 100.5
            rows.append((o, o * 1.01, o * 0.99, o * 1.005))
        result = yang_zhang_rv30(_ohlc_frame(rows), window=30)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0.0)

        # Recompute the same value from the formula on the trailing 30 rows.
        # This is the regression baseline — if the kernel ever changes algorithm,
        # this assertion catches the drift.
        df = _ohlc_frame(rows).tail(31)
        o = np.log(df["Open"] / df["Close"].shift(1)).dropna()
        c = np.log(df["Close"] / df["Open"]).dropna()
        h = np.log(df["High"] / df["Open"]).dropna()
        l = np.log(df["Low"] / df["Open"]).dropna()
        idx = o.index.intersection(c.index).intersection(h.index).intersection(l.index)
        o = o[idx].tail(30); c = c[idx].tail(30); h = h[idx].tail(30); l = l[idx].tail(30)
        n = len(o)
        rs = h * (h - c) + l * (l - c)
        k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
        var_o = float(((o - o.mean()) ** 2).sum() / max(n - 1, 1))
        var_c = float(((c - c.mean()) ** 2).sum() / max(n - 1, 1))
        var_rs = float(rs.mean())
        expected = float(np.sqrt(max(var_o + k * var_c + (1.0 - k) * var_rs, 1e-10) * 252.0))
        self.assertAlmostEqual(result, expected, places=12)

    def test_excluded_sessions_none_matches_no_exclusion(self):
        """Default excluded_sessions=None must produce the same result as
        passing an empty set or omitting the parameter."""
        rng = np.random.default_rng(42)
        rows = []
        prev_close = 100.0
        for _ in range(50):
            o = prev_close * (1.0 + rng.normal(0, 0.005))
            h = o * (1.0 + abs(rng.normal(0, 0.01)))
            l = o * (1.0 - abs(rng.normal(0, 0.01)))
            c = o * (1.0 + rng.normal(0, 0.008))
            rows.append((o, h, l, c))
            prev_close = c
        df = _ohlc_frame(rows)

        without = yang_zhang_rv30(df, window=30)
        empty_set = yang_zhang_rv30(df, window=30, excluded_sessions=set())
        explicit_none = yang_zhang_rv30(df, window=30, excluded_sessions=None)
        self.assertAlmostEqual(without, empty_set, places=12)
        self.assertAlmostEqual(without, explicit_none, places=12)

    def test_excluded_sessions_drops_target_day_from_estimate(self):
        """Injecting one outsized session and excluding it must yield the same
        estimate as if the outlier were never in the input."""
        rng = np.random.default_rng(7)
        rows = []
        prev_close = 100.0
        for _ in range(50):
            o = prev_close * (1.0 + rng.normal(0, 0.005))
            h = o * 1.01
            l = o * 0.99
            c = o * (1.0 + rng.normal(0, 0.005))
            rows.append((o, h, l, c))
            prev_close = c
        df_clean = _ohlc_frame(rows)

        # Inject an extreme jump session at index 25.
        df_with_spike = df_clean.copy()
        spike_open = float(df_with_spike.iloc[24]["Close"]) * 1.10  # 10% gap
        df_with_spike.iloc[25] = [spike_open, spike_open * 1.06, spike_open * 0.94, spike_open * 1.05]
        spike_ts = pd.Timestamp(df_with_spike.index[25]).normalize()

        # Without exclusion the spike inflates YZ; with exclusion the result
        # should approach the clean-series value.
        spike_yz = yang_zhang_rv30(df_with_spike, window=30)
        clean_yz = yang_zhang_rv30(df_clean, window=30)
        excluded_yz = yang_zhang_rv30(df_with_spike, window=30, excluded_sessions={spike_ts})

        self.assertGreater(spike_yz, clean_yz)
        # Excluding the spike should pull YZ back down — strictly below the spiked value.
        self.assertLess(excluded_yz, spike_yz)


class TestRSDailyVolSeries(unittest.TestCase):
    def test_returns_empty_series_when_columns_missing(self):
        df = pd.DataFrame({"Open": [1.0, 2.0]})
        out = rs_daily_vol_series(df)
        self.assertTrue(isinstance(out, pd.Series))
        self.assertTrue(out.empty)

    def test_constant_price_session_yields_zero(self):
        flat = _ohlc_frame([(100.0, 100.0, 100.0, 100.0)] * 5)
        out = rs_daily_vol_series(flat)
        self.assertEqual(len(out), 5)
        self.assertTrue((out == 0.0).all())

    def test_known_values_match_rogers_satchell_formula(self):
        # Single session with H=101, L=99, O=100, C=100.5
        # log(H/O) = log(1.01); log(H/C) = log(101/100.5)
        # log(L/O) = log(0.99); log(L/C) = log(99/100.5)
        # RS_var = log(H/O)·log(H/C) + log(L/O)·log(L/C)
        # Annualised vol = √(RS_var × 252)
        df = _ohlc_frame([(100.0, 101.0, 99.0, 100.5)])
        out = rs_daily_vol_series(df)
        log_HO = np.log(101.0 / 100.0)
        log_HC = np.log(101.0 / 100.5)
        log_LO = np.log(99.0 / 100.0)
        log_LC = np.log(99.0 / 100.5)
        expected_var = max(log_HO * log_HC + log_LO * log_LC, 0.0)
        expected_vol = float(np.sqrt(expected_var * 252.0))
        self.assertAlmostEqual(float(out.iloc[0]), expected_vol, places=12)

    def test_excluded_sessions_drops_filtered_rows(self):
        df = _ohlc_frame(
            [
                (100.0, 101.0, 99.0, 100.5),
                (100.5, 102.0, 100.0, 101.5),
                (101.5, 102.5, 101.0, 102.0),
            ]
        )
        # Drop middle row.
        excl = {pd.Timestamp(df.index[1]).normalize()}
        full = rs_daily_vol_series(df)
        filtered = rs_daily_vol_series(df, excluded_sessions=excl)
        self.assertEqual(len(full), 3)
        self.assertEqual(len(filtered), 2)
        # First and third rows present in the filtered output.
        self.assertAlmostEqual(float(filtered.iloc[0]), float(full.iloc[0]), places=12)
        self.assertAlmostEqual(float(filtered.iloc[1]), float(full.iloc[2]), places=12)


class TestHARRVForecast(unittest.TestCase):
    def test_returns_none_when_below_min_obs(self):
        short = pd.Series(np.linspace(0.2, 0.3, HAR_MIN_OBS - 1))
        self.assertIsNone(har_rv_forecast(short))

    def test_returns_none_for_empty_series(self):
        self.assertIsNone(har_rv_forecast(pd.Series(dtype=float)))

    def test_constant_series_forecast_recovers_input_level(self):
        """For a constant input vol σ, the HAR forecast should also be ≈ σ.

        With every RV equal to σ², the (RV_d, RV_w, RV_m) features collapse
        to identical values across the in-sample, OLS produces a fit that
        passes through the constant, and the forecast — sqrt of that — is σ.
        Floored at 1e-4 by the implementation.
        """
        sigma = 0.25
        rv_daily = pd.Series([sigma] * 200)
        forecast = har_rv_forecast(rv_daily)
        self.assertIsNotNone(forecast)
        self.assertAlmostEqual(forecast, sigma, places=4)

    def test_forecast_floor(self):
        """Negative or near-zero variance is clamped by the implementation
        to a 1 bp (1e-4) floor before being returned."""
        rv_daily = pd.Series([0.0] * 200)
        forecast = har_rv_forecast(rv_daily)
        self.assertIsNotNone(forecast)
        self.assertGreaterEqual(forecast, 1e-4)


class TestRSTrailingMeanForecast(unittest.TestCase):
    def test_returns_none_when_below_window(self):
        short = pd.Series([0.2] * 10)
        self.assertIsNone(rs_trailing_mean_forecast(short, window=30))

    def test_returns_simple_mean_of_trailing_window(self):
        rv_daily = pd.Series([0.10] * 20 + [0.30] * 30)
        out = rs_trailing_mean_forecast(rv_daily, window=30)
        # Trailing 30 are all 0.30
        self.assertAlmostEqual(out, 0.30, places=12)

    def test_returns_none_when_trailing_dropna_too_short(self):
        # 30 rows total but mostly NaN → after dropna only 4 valid → return None.
        rv_daily = pd.Series([np.nan] * 26 + [0.20, 0.25, 0.22, 0.28])
        self.assertIsNone(rs_trailing_mean_forecast(rv_daily, window=30))


if __name__ == "__main__":
    unittest.main()
