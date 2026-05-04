"""
P-5a — variance-additive event-vol decomposition.

Pins the new behaviour:
- near_term_implied_move_pct stays as the legacy MAD-form straddle premium.
- near_term_implied_sigma_pct is the 1σ-form percent move over [0, T_years],
  obtained by Brenner-Subrahmanyam: σ·√T = (call+put)/S · √(π/2).
- non_event_move_pct_har is computed on calendar time (denominator 365).
- event_implied_move_pct is the 1σ-form event component, derived from
  variance-additive subtraction in 1σ-form units.
- event_move_share_of_total uses the 1σ denominator
  (near_term_implied_sigma_pct), not the legacy MAD denominator.
- historical_vs_implied_move_ratio and tail_vs_implied_move_ratio use the
  1σ-form event_implied_move_pct denominator.

These tests exercise the decomposition through `build_vol_snapshot` so the
production code path is the one validated.
"""
from __future__ import annotations

import math
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from services.earnings_vol_snapshot import (
    VolSnapshotConfig,
    build_vol_snapshot,
)


# ── Test fixtures ─────────────────────────────────────────────────────────────


def _ohlc_history(
    n_days: int = 260,
    sigma_annual: float = 0.30,
    seed: int = 42,
    last_close: float = 100.0,
) -> pd.DataFrame:
    """Synthesize ~1y of business-day OHLC bars whose realized vol is
    approximately ``sigma_annual``. Used to drive the snapshot's RV pipeline
    so HAR-RV converges to ``sigma_annual``."""
    rng = np.random.default_rng(seed)
    # Anchor on a fixed historical start so test runs are deterministic and
    # bdate_range never collides with weekend-end edge cases.
    dates = pd.bdate_range(start="2024-01-02", periods=n_days)
    daily_sigma = sigma_annual / math.sqrt(252)
    log_rets = rng.normal(0.0, daily_sigma, size=n_days)
    close = np.empty(n_days, dtype=float)
    close[-1] = last_close
    for i in range(n_days - 2, -1, -1):
        close[i] = close[i + 1] / math.exp(log_rets[i + 1])
    open_ = np.empty(n_days, dtype=float)
    open_[0] = close[0] * math.exp(rng.normal(0.0, daily_sigma * 0.5))
    for i in range(1, n_days):
        open_[i] = close[i - 1] * math.exp(rng.normal(0.0, daily_sigma * 0.4))
    intraday_swing = np.abs(rng.normal(0.0, daily_sigma * 0.6, size=n_days))
    high = np.maximum(open_, close) * np.exp(intraday_swing)
    low = np.minimum(open_, close) * np.exp(-intraday_swing)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1_000_000.0},
        index=dates,
    )


def _option_chain(
    spot: float,
    expiry: date,
    sigma_implied: float,
    *,
    today: date,
) -> pd.DataFrame:
    """Synthesize an ATM-bracketed two-expiry chain whose ATM straddle premium
    matches Brenner-Subrahmanyam at the given implied σ:
        (C_atm + P_atm)/S = σ · √(2 · T_years / π)

    We split the premium evenly between call and put (ATM put-call symmetry
    near zero-rate); this is sufficient for `_expiry_atm_stats` to recover
    the same `near_term_implied_move_pct = (C+P)/S × 100`.
    """
    T_years = max((expiry - today).days, 0) / 365.0
    if T_years <= 0:
        atm_call_mid = atm_put_mid = 0.001
    else:
        atm_premium = spot * sigma_implied * math.sqrt(2.0 * T_years / math.pi)
        atm_call_mid = atm_put_mid = atm_premium / 2.0
    rows = []
    # Need at least two strikes per side; ATM and one OTM at +5%/-5%.
    for strike, call_price, put_price in (
        (round(spot * 0.95, 2), atm_call_mid * 1.5, atm_put_mid * 0.5),
        (round(spot, 2), atm_call_mid, atm_put_mid),
        (round(spot * 1.05, 2), atm_call_mid * 0.5, atm_put_mid * 1.5),
    ):
        for side, price in (("C", call_price), ("P", put_price)):
            spread = max(price * 0.04, 0.05)
            rows.append(
                {
                    "expiration_date": expiry.strftime("%Y-%m-%d"),
                    "strike": float(strike),
                    "call_put": side,
                    "iv": float(sigma_implied),
                    "mid": float(price),
                    "bid": float(price - spread / 2),
                    "ask": float(price + spread / 2),
                    "open_interest": 1500.0,
                    "volume": 800.0,
                    "underlying_price": float(spot),
                    "trade_date": today.strftime("%Y-%m-%d"),
                }
            )
    return pd.DataFrame(rows)


def _two_expiry_chain(
    spot: float,
    near_expiry: date,
    back_expiry: date,
    sigma_implied_near: float,
    sigma_implied_back: float,
    *,
    today: date,
) -> pd.DataFrame:
    return pd.concat(
        [
            _option_chain(spot, near_expiry, sigma_implied_near, today=today),
            _option_chain(spot, back_expiry, sigma_implied_back, today=today),
        ],
        ignore_index=True,
    )


def _build_snapshot(
    *,
    sigma_implied_near: float,
    sigma_implied_back: float,
    rv_target: float,
    near_dte: int,
    back_dte: int,
    spot: float = 100.0,
    seed: int = 42,
):
    today = date.today()
    near_expiry = today + timedelta(days=near_dte)
    back_expiry = today + timedelta(days=back_dte)
    chain = _two_expiry_chain(
        spot=spot,
        near_expiry=near_expiry,
        back_expiry=back_expiry,
        sigma_implied_near=sigma_implied_near,
        sigma_implied_back=sigma_implied_back,
        today=today,
    )
    history = _ohlc_history(n_days=260, sigma_annual=rv_target, seed=seed, last_close=spot)
    return build_vol_snapshot(
        symbol="TEST",
        as_of=today,
        option_chain_data=chain,
        earnings_metadata={
            "earnings_date": today + timedelta(days=max(near_dte // 2, 1)),
            "release_timing": "after market close",
            "prior_events": [],
        },
        price_data=history,
        config=VolSnapshotConfig(),
    )


# ── Brenner-Subrahmanyam direct-formula tests ────────────────────────────────


class TestBrennerSubrahmanyamConversion(unittest.TestCase):
    """The MAD → 1σ conversion is closed-form: σ_pct = MAD_pct · √(π/2).
    This test validates the conversion in isolation without going through
    the full snapshot pipeline.
    """

    def test_conversion_factor_is_sqrt_pi_over_two(self):
        # Pick an arbitrary MAD-form value; conversion must scale by √(π/2)
        mad_pct = 6.86  # representative ATM straddle ≈ 6.86% for σ=30%, T=30/365
        expected_sigma_pct = mad_pct * math.sqrt(math.pi / 2.0)
        self.assertAlmostEqual(expected_sigma_pct / mad_pct, 1.2533, places=4)

    def test_recovers_one_sigma_for_known_inputs(self):
        # σ = 0.30, T = 30/365. ATM 1σ pct move = σ·√T·100 = 30·√(30/365) ≈ 8.602%
        sigma = 0.30
        T_years = 30.0 / 365.0
        mad_pct = 100.0 * sigma * math.sqrt(2.0 * T_years / math.pi)
        sigma_pct = mad_pct * math.sqrt(math.pi / 2.0)
        expected_sigma_pct = 100.0 * sigma * math.sqrt(T_years)
        self.assertAlmostEqual(sigma_pct, expected_sigma_pct, places=10)


# ── Snapshot-level decomposition tests ───────────────────────────────────────


class TestEventDecomposition(unittest.TestCase):
    """End-to-end tests through `build_vol_snapshot`."""

    def test_near_term_implied_sigma_pct_is_populated(self):
        snap = _build_snapshot(
            sigma_implied_near=0.40,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=30,
            back_dte=60,
        )
        self.assertIsNotNone(snap.near_term_implied_sigma_pct)
        self.assertGreater(snap.near_term_implied_sigma_pct, 0.0)

    def test_near_term_implied_move_pct_unchanged_legacy_mad_form(self):
        """The legacy MAD-form field must continue to equal (call+put)/S · 100.
        For our synthetic chain the relation is exact within fp tolerance."""
        sigma = 0.40
        snap = _build_snapshot(
            sigma_implied_near=sigma,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=30,
            back_dte=60,
        )
        self.assertIsNotNone(snap.near_term_implied_move_pct)
        T_years = (snap.near_term_dte or 0) / 365.0
        expected_mad_pct = 100.0 * sigma * math.sqrt(2.0 * T_years / math.pi)
        # Synthetic chain uses sigma_implied_near directly; expected value is exact.
        self.assertAlmostEqual(
            snap.near_term_implied_move_pct, expected_mad_pct, places=4
        )

    def test_near_term_implied_sigma_pct_equals_mad_times_sqrt_pi_over_two(self):
        """The new field must be the MAD-form scaled by √(π/2) — independent of dte."""
        snap = _build_snapshot(
            sigma_implied_near=0.42,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=21,
            back_dte=60,
        )
        self.assertIsNotNone(snap.near_term_implied_move_pct)
        self.assertIsNotNone(snap.near_term_implied_sigma_pct)
        expected = snap.near_term_implied_move_pct * math.sqrt(math.pi / 2.0)
        self.assertAlmostEqual(snap.near_term_implied_sigma_pct, expected, places=10)

    def test_event_decomposition_near_zero_when_implied_equals_realized(self):
        """When σ_implied ≈ σ_HAR and there's no event premium, event_implied_move_pct
        should be small (≤ 1% of underlying). Threshold accounts for HAR forecast
        noise on a finite synthetic series."""
        snap = _build_snapshot(
            sigma_implied_near=0.30,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=30,
            back_dte=60,
        )
        self.assertIsNotNone(snap.event_implied_move_pct)
        self.assertLess(snap.event_implied_move_pct, 1.0)

    def test_event_decomposition_positive_when_implied_exceeds_realized(self):
        """σ_implied = 0.45 vs σ_HAR ≈ 0.30 over dte=30:
            σ_event² · 1d = σ_imp² · T - σ_HAR² · (T-1)
                          = 0.45² · 30/365 - 0.30² · 29/365
                          = 0.01664 - 0.00715 = 0.00949
            event_implied_move_pct = √0.00949 · 100 = 9.74
        (the ×100 here is pct-of-underlying scaled to the 1σ-form total over period).
        Allow loose bounds because synthetic HAR carries finite-sample noise.
        """
        snap = _build_snapshot(
            sigma_implied_near=0.45,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=30,
            back_dte=60,
        )
        self.assertIsNotNone(snap.event_implied_move_pct)
        # Lower bound: must clear the silent-zero pathology (P-5a's primary aim).
        self.assertGreater(snap.event_implied_move_pct, 4.0)
        # Upper bound: must not blow up.
        self.assertLess(snap.event_implied_move_pct, 15.0)

    def test_event_decomposition_clamps_at_zero_when_implied_below_realized(self):
        """If σ_implied < σ_HAR, total variance < non-event variance → clamp to 0,
        not NaN, not negative."""
        snap = _build_snapshot(
            sigma_implied_near=0.20,
            sigma_implied_back=0.40,
            rv_target=0.40,
            near_dte=30,
            back_dte=60,
        )
        self.assertIsNotNone(snap.event_implied_move_pct)
        self.assertEqual(snap.event_implied_move_pct, 0.0)

    def test_non_event_uses_calendar_time_denominator(self):
        """Pin the convention: non_event_move_pct_har = σ_HAR · √((dte-1)/365) · 100
        (calendar time), not √((dte-1)/252) (trading time).

        Validate by reconstructing the formula from `rv_har_forecast` (the σ_HAR
        the snapshot actually used) and asserting equality with the snapshot's
        published `non_event_move_pct_har` to fp tolerance.  This isolates the
        formula change from σ_HAR sample noise.
        """
        snap = _build_snapshot(
            sigma_implied_near=0.45,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=30,
            back_dte=60,
        )
        self.assertIsNotNone(snap.non_event_move_pct_har)
        self.assertIsNotNone(snap.rv_har_forecast)
        # Reconstruct under the calendar-time formula:
        non_event_T_years = max((snap.near_term_dte or 0) - 1, 0) / 365.0
        expected_calendar = snap.rv_har_forecast * math.sqrt(non_event_T_years) * 100.0
        self.assertAlmostEqual(snap.non_event_move_pct_har, expected_calendar, places=10)
        # And confirm that the trading-time formula would give a materially
        # different number (so this test would catch a regression to /252).
        non_event_T_trading = max((snap.near_term_dte or 0) - 1, 0) / 252.0
        wrong_trading = snap.rv_har_forecast * math.sqrt(non_event_T_trading) * 100.0
        # Calendar-time is strictly smaller than trading-time for the same dte > 1.
        self.assertLess(expected_calendar, wrong_trading)


class TestRatioDimensionalConsistency(unittest.TestCase):
    """The numerator and denominator of every ratio must be in matching units
    after P-5a."""

    def test_event_move_share_uses_one_sigma_denominator(self):
        snap = _build_snapshot(
            sigma_implied_near=0.45,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=30,
            back_dte=60,
        )
        self.assertIsNotNone(snap.event_move_share_of_total)
        self.assertIsNotNone(snap.near_term_implied_sigma_pct)
        self.assertIsNotNone(snap.event_implied_move_pct)
        expected = snap.event_implied_move_pct / snap.near_term_implied_sigma_pct
        self.assertAlmostEqual(snap.event_move_share_of_total, expected, places=10)
        # Ratio is bounded: event ≤ total in 1σ form.
        self.assertGreaterEqual(snap.event_move_share_of_total, 0.0)
        self.assertLessEqual(snap.event_move_share_of_total, 1.0 + 1e-9)

    def test_event_move_share_zero_when_no_event_premium(self):
        snap = _build_snapshot(
            sigma_implied_near=0.30,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=30,
            back_dte=60,
        )
        self.assertIsNotNone(snap.event_move_share_of_total)
        # Should be small (< 0.15) — not literally 0 because of HAR finite-sample noise.
        self.assertLess(snap.event_move_share_of_total, 0.15)

    def test_event_move_share_high_for_strong_event_premium(self):
        snap = _build_snapshot(
            sigma_implied_near=0.60,
            sigma_implied_back=0.30,
            rv_target=0.30,
            near_dte=14,
            back_dte=45,
        )
        self.assertIsNotNone(snap.event_move_share_of_total)
        self.assertGreater(snap.event_move_share_of_total, 0.40)


class TestEdgeEngineWiring(unittest.TestCase):
    """Drift detection: the legacy diagnostic block in
    web.api.edge_engine.analyze_single_ticker must read the decomposition
    from the snapshot, not recompute it locally."""

    def test_no_local_non_event_days_recomputation_in_edge_engine(self):
        import web.api.edge_engine as ee
        import inspect
        src = inspect.getsource(ee.analyze_single_ticker)
        # The pre-P-5a duplicate set non_event_days inside analyze_single_ticker
        # and recomputed event_implied_move_pct via local quadrature.  After
        # P-5a these names should not appear as locally-assigned tokens (the
        # canonical values come from snapshot_inputs).
        self.assertNotIn("non_event_days = max(int(near_term_dte) - 1, 0)", src)


if __name__ == "__main__":
    unittest.main()
