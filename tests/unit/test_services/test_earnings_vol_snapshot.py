import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

import web.api.edge_engine as edge_engine
from services.earnings_vol_snapshot import build_vol_snapshot


def _make_price_history(as_of: str = "2026-04-20") -> tuple[pd.DataFrame, list[dict[str, object]]]:
    dates = pd.bdate_range(end=as_of, periods=140)
    close = pd.Series(100.0 + np.linspace(0.0, 4.0, len(dates)), index=dates)

    specs = [
        (25, 4.5, "after market close"),
        (45, 5.0, "before market open"),
        (65, 6.0, "after market close"),
        (85, 4.0, "after market close"),
        (105, 7.0, "before market open"),
        (120, 5.5, "after market close"),
    ]
    prior_events: list[dict[str, object]] = []
    for idx, move_pct, timing in specs:
        event_date = pd.Timestamp(dates[idx])
        if timing == "after market close":
            pre_idx, post_idx = idx, idx + 1
        else:
            pre_idx, post_idx = idx - 1, idx
        pre_px = float(close.iloc[pre_idx])
        close.iloc[post_idx] = pre_px * (1.0 + move_pct / 100.0)
        prior_events.append({"event_date": event_date, "release_timing": timing})

    open_ = close.shift(1).fillna(close.iloc[0] / 1.002)
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.full(len(dates), 5_000_000, dtype=float)
    price_df = pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": volume,
        }
    )
    return price_df, prior_events


def _make_chain(
    as_of: str = "2026-04-20",
    *,
    spread_multiplier: float = 1.0,
    trade_date: str | None = None,
    strikes: tuple[float, ...] = (90.0, 95.0, 100.0, 105.0, 110.0),
    expiries: tuple[str, ...] = ("2026-04-24", "2026-05-15", "2026-06-19"),
    base_term_ivs: tuple[float, ...] = (0.24, 0.28, 0.31),
) -> pd.DataFrame:
    trade_date = trade_date or as_of
    rows: list[dict[str, object]] = []
    spot = 100.0
    for expiry, term_iv in zip(expiries, base_term_ivs):
        for strike in strikes:
            mny = abs((strike / spot) - 1.0)
            smile_bump = 0.06 * (mny ** 2) * 100.0
            iv = term_iv + smile_bump
            is_atm = abs(strike - spot) < 1e-9
            base_mid = 3.0 if is_atm and expiry == expiries[0] else 2.2 if is_atm else max(0.65, 1.8 - (mny * 6.0))
            width = base_mid * 0.04 * spread_multiplier
            for side in ("C", "P"):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "expiry": expiry,
                        "call_put": side,
                        "strike": strike,
                        "bid": round(base_mid - (width / 2.0), 4),
                        "ask": round(base_mid + (width / 2.0), 4),
                        "mid": round(base_mid, 4),
                        "iv": round(iv, 4),
                        "open_interest": 2500 if is_atm else 900,
                        "volume": 1200 if is_atm else 350,
                        "underlying_price": spot,
                    }
                )
    return pd.DataFrame(rows)


class TestEarningsVolSnapshot(unittest.TestCase):
    def _build_snapshot(
        self,
        *,
        price_df: pd.DataFrame | None = None,
        chain_df: pd.DataFrame | None = None,
        earnings_metadata: dict[str, object] | None = None,
        as_of: str = "2026-04-20",
    ):
        if price_df is None or earnings_metadata is None:
            base_prices, prior_events = _make_price_history(as_of=as_of)
            if price_df is None:
                price_df = base_prices
            if earnings_metadata is None:
                earnings_metadata = {
                    "earnings_date": "2026-04-28",
                    "release_timing": "after market close",
                    "prior_events": prior_events,
                }
        if chain_df is None:
            chain_df = _make_chain(as_of=as_of)
        return build_vol_snapshot(
            "AAPL",
            date.fromisoformat(as_of),
            option_chain_data=chain_df,
            earnings_metadata=earnings_metadata,
            price_data=price_df,
        )

    def test_snapshot_is_deterministic_for_same_inputs(self):
        snapshot_1 = self._build_snapshot()
        snapshot_2 = self._build_snapshot()
        self.assertEqual(snapshot_1.to_dict(), snapshot_2.to_dict())

    def test_snapshot_preserves_earnings_source_provenance(self):
        price_df, prior_events = _make_price_history()
        snapshot = self._build_snapshot(
            price_df=price_df,
            chain_df=_make_chain(),
            earnings_metadata={
                "earnings_date": "2026-04-28",
                "release_timing": "after market close",
                "prior_events": prior_events,
                "earnings_source_primary": "fmp_calendar",
                "earnings_source_confirmed": "sec_submissions",
                "earnings_source_confidence": 0.90,
                "release_timing_source": "sec_submissions",
                "earnings_source_stale": True,
                "earnings_source_notes": ["stale local cache used"],
            },
        )

        self.assertEqual(snapshot.earnings_source_primary, "fmp_calendar")
        self.assertEqual(snapshot.earnings_source_confirmed, "sec_submissions")
        self.assertAlmostEqual(snapshot.earnings_source_confidence or 0.0, 0.90, places=6)
        self.assertEqual(snapshot.release_timing_source, "sec_submissions")
        self.assertTrue(snapshot.earnings_source_stale)
        self.assertEqual(snapshot.earnings_source_notes, ["stale local cache used"])
        self.assertEqual(
            snapshot.null_reasons["earnings_source_stale"],
            "stale_earnings_cache_used_after_refresh_failure",
        )

    def test_earnings_source_confidence_affects_data_quality(self):
        price_df, prior_events = _make_price_history()
        base_metadata = {
            "earnings_date": "2026-04-28",
            "release_timing": "after market close",
            "prior_events": prior_events,
            "earnings_source_primary": "alpha_vantage_calendar",
        }
        high = self._build_snapshot(
            price_df=price_df,
            earnings_metadata={**base_metadata, "earnings_source_confidence": 0.95},
        )
        low = self._build_snapshot(
            price_df=price_df,
            earnings_metadata={**base_metadata, "earnings_source_confidence": 0.35},
        )
        stale = self._build_snapshot(
            price_df=price_df,
            earnings_metadata={
                **base_metadata,
                "earnings_source_confidence": 0.95,
                "earnings_source_stale": True,
            },
        )

        self.assertGreater(high.data_quality_score, low.data_quality_score)
        self.assertLess(stale.data_quality_score, high.data_quality_score)
        self.assertLessEqual(stale.data_quality_score, 0.72)

    def test_cheap_iv_setup_stays_neutral_and_scores_as_cheaper(self):
        snapshot = self._build_snapshot(chain_df=_make_chain(base_term_ivs=(0.21, 0.26, 0.29)))
        self.assertLess(snapshot.near_back_iv_ratio, 1.0)
        self.assertIsNotNone(snapshot.cheapness_score)
        self.assertGreater(snapshot.cheapness_score, 0.55)
        payload = snapshot.to_dict()
        self.assertNotIn("recommendation", payload)
        self.assertNotIn("best_structure", payload)
        self.assertNotIn("hard_no_trade", payload)

    def test_wider_spreads_only_degrade_execution_block(self):
        tight = self._build_snapshot(chain_df=_make_chain(spread_multiplier=1.0))
        wide = self._build_snapshot(chain_df=_make_chain(spread_multiplier=3.0))

        self.assertAlmostEqual(tight.iv30, wide.iv30, places=8)
        self.assertAlmostEqual(tight.term_structure_slope, wide.term_structure_slope, places=8)
        self.assertGreater(wide.near_term_spread_pct, tight.near_term_spread_pct)
        self.assertLess(wide.execution_score, tight.execution_score)

    def test_more_historical_events_improve_uncertainty_and_quality(self):
        price_df, prior_events = _make_price_history()
        short_history = {
            "earnings_date": "2026-04-28",
            "release_timing": "after market close",
            "prior_events": prior_events[:2],
        }
        long_history = {
            "earnings_date": "2026-04-28",
            "release_timing": "after market close",
            "prior_events": prior_events,
        }
        sparse = self._build_snapshot(price_df=price_df, earnings_metadata=short_history)
        rich = self._build_snapshot(price_df=price_df, earnings_metadata=long_history)

        self.assertEqual(sparse.historical_move_source, "earnings_history")
        self.assertEqual(rich.historical_move_source, "earnings_history")
        self.assertGreater(rich.historical_event_count, sparse.historical_event_count)
        self.assertGreaterEqual(rich.data_quality_score, sparse.data_quality_score)

    def test_missing_earnings_date_returns_null_timing_fields(self):
        price_df, prior_events = _make_price_history()
        snapshot = self._build_snapshot(
            price_df=price_df,
            earnings_metadata={"release_timing": "after market close", "prior_events": prior_events},
        )
        self.assertIsNone(snapshot.earnings_date)
        self.assertIsNone(snapshot.days_to_earnings)
        self.assertIsNone(snapshot.timing_score)
        self.assertEqual(snapshot.null_reasons["earnings_date"], "missing_earnings_metadata")

    def test_partial_chain_preserves_near_term_state_but_nulls_curve_interpolation(self):
        chain_df = _make_chain(expiries=("2026-04-24",), base_term_ivs=(0.24,))
        snapshot = self._build_snapshot(chain_df=chain_df)

        self.assertIsNotNone(snapshot.near_term_atm_iv)
        self.assertIsNotNone(snapshot.near_term_implied_move_pct)
        self.assertIsNone(snapshot.iv30)
        self.assertIsNone(snapshot.iv45)
        self.assertIsNone(snapshot.term_structure_slope)
        self.assertEqual(snapshot.null_reasons["iv30"], "insufficient_term_structure_points")

    def test_stale_prices_degrade_data_quality_without_changing_surface_state(self):
        fresh_price_df, prior_events = _make_price_history()
        stale_price_df = fresh_price_df[fresh_price_df["trade_date"] <= "2026-04-15"].copy()
        earnings = {
            "earnings_date": "2026-04-28",
            "release_timing": "after market close",
            "prior_events": prior_events,
        }

        fresh = self._build_snapshot(price_df=fresh_price_df, earnings_metadata=earnings)
        stale = self._build_snapshot(price_df=stale_price_df, earnings_metadata=earnings)

        self.assertEqual(fresh.iv30, stale.iv30)
        self.assertGreater(stale.price_staleness_minutes, fresh.price_staleness_minutes)
        self.assertLess(stale.data_quality_score, fresh.data_quality_score)

    def test_insufficient_smile_points_returns_null_smile_fields(self):
        chain_df = _make_chain(strikes=(95.0, 100.0, 105.0))
        snapshot = self._build_snapshot(chain_df=chain_df)

        self.assertIsNone(snapshot.smile_curvature)
        self.assertIsNone(snapshot.smile_concavity_flag)
        self.assertEqual(snapshot.smile_points, 0)
        self.assertEqual(snapshot.null_reasons["smile_curvature"], "insufficient_smile_points")

    def test_event_day_exclusion_reduces_baseline_rv_and_har_when_recent_event_is_large(self):
        # 140 periods → ~139 RS values, above _HAR_MIN_OBS=100, so true HAR fires
        dates = pd.bdate_range(end="2026-04-20", periods=140)
        close = pd.Series(100.0 + np.linspace(0.0, 2.0, len(dates)), index=dates)
        event_idx = len(dates) - 8
        pre_px = float(close.iloc[event_idx])
        close.iloc[event_idx + 1] = pre_px * 1.15
        open_ = close.shift(1).fillna(close.iloc[0] / 1.001)
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        price_df = pd.DataFrame(
            {
                "trade_date": dates,
                "open": open_.values,
                "high": high.values,
                "low": low.values,
                "close": close.values,
                "volume": np.full(len(dates), 4_000_000, dtype=float),
            }
        )
        prior_events = [{"event_date": dates[event_idx], "release_timing": "after market close"}]
        with_history = self._build_snapshot(
            price_df=price_df,
            earnings_metadata={
                "earnings_date": "2026-04-28",
                "release_timing": "after market close",
                "prior_events": prior_events,
            },
        )
        without_history = self._build_snapshot(
            price_df=price_df,
            earnings_metadata={
                "earnings_date": "2026-04-28",
                "release_timing": "after market close",
                "prior_events": [],
            },
        )

        self.assertIsNotNone(with_history.rv30_yang_zhang)
        self.assertIsNotNone(without_history.rv30_yang_zhang)
        self.assertLess(with_history.rv30_yang_zhang, without_history.rv30_yang_zhang)
        self.assertLess(with_history.rv_har_forecast, without_history.rv_har_forecast)

    def test_event_day_exclusion_does_not_fabricate_clean_history_when_window_too_sparse(self):
        price_df, prior_events = _make_price_history()
        # 25 rows → ~24 RS values, below both _HAR_MIN_OBS=100 and _RS_FALLBACK_WINDOW=30
        short_price_df = price_df.tail(25).copy()
        snapshot = self._build_snapshot(
            price_df=short_price_df,
            earnings_metadata={
                "earnings_date": "2026-04-28",
                "release_timing": "after market close",
                "prior_events": prior_events,
            },
        )
        self.assertIsNone(snapshot.rv_har_forecast)
        self.assertIn(snapshot.null_reasons["rv_har_forecast"], {
            "insufficient_clean_rs_history_after_event_exclusion",
            "insufficient_rs_history_for_har",
        })

    def test_snapshot_neutrality_does_not_leak_structure_fields(self):
        snapshot = self._build_snapshot()
        payload = snapshot.to_dict()
        forbidden = {
            "recommendation",
            "candidate",
            "best_structure",
            "runner_up_structures",
            "hard_no_trade",
            "analysis_mode",
        }
        self.assertTrue(forbidden.isdisjoint(payload.keys()))

    def test_service_matches_legacy_history_math_integration(self):
        price_df, prior_events = _make_price_history()
        snapshot = self._build_snapshot(
            price_df=price_df,
            earnings_metadata={
                "earnings_date": "2026-04-28",
                "release_timing": "after market close",
                "prior_events": prior_events,
            },
        )
        close = price_df.set_index("trade_date")["close"]
        with patch.object(edge_engine, "_utc_today_date", return_value=date.fromisoformat("2026-04-20")):
            legacy_profile = edge_engine._historical_earnings_move_profile(close=close, earnings_events=prior_events)
        legacy_anchor = edge_engine._compute_move_anchor(
            legacy_profile.get("median_move_pct"),
            legacy_profile.get("avg_last4_move_pct"),
        )
        legacy_uncertainty = edge_engine._compute_move_uncertainty_pct(
            move_std_pct=legacy_profile.get("std_move_pct"),
            sample_size=int(legacy_profile.get("event_count", 0) or 0),
            move_source=str(legacy_profile.get("source", "none")),
        )

        self.assertAlmostEqual(snapshot.historical_median_move_pct, legacy_profile["median_move_pct"], places=8)
        self.assertAlmostEqual(snapshot.historical_avg_last4_move_pct, legacy_profile["avg_last4_move_pct"], places=8)
        self.assertAlmostEqual(snapshot.historical_p90_move_pct, legacy_profile["p90_move_pct"], places=8)
        self.assertAlmostEqual(snapshot.historical_move_anchor_pct, legacy_anchor, places=8)
        self.assertAlmostEqual(snapshot.historical_move_uncertainty_pct, legacy_uncertainty, places=8)


class TestHARRVHardening(unittest.TestCase):
    """
    Verify HAR-RV minimum-sample hardening (audit precondition 2).

    HAR fires only at/above _HAR_MIN_OBS.  Below that threshold, the RS
    trailing-mean fallback is used when >= _RS_FALLBACK_WINDOW values are
    available, and None is returned when the series is too short for even
    the fallback.
    """

    def _price_df(self, periods: int) -> pd.DataFrame:
        dates = pd.bdate_range(end="2026-04-20", periods=periods)
        close = pd.Series(100.0 + np.linspace(0.0, 2.0, periods), index=dates)
        open_ = close.shift(1).fillna(close.iloc[0] / 1.001)
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        return pd.DataFrame(
            {
                "trade_date": dates,
                "open": open_.values,
                "high": high.values,
                "low": low.values,
                "close": close.values,
                "volume": np.full(periods, 4_000_000, dtype=float),
            }
        )

    def _snap(self, price_df: pd.DataFrame) -> "VolSnapshot":
        return build_vol_snapshot(
            "TEST",
            date.fromisoformat("2026-04-20"),
            option_chain_data=_make_chain(),
            earnings_metadata={
                "earnings_date": "2026-04-28",
                "release_timing": "after market close",
                "prior_events": [],
            },
            price_data=price_df,
        )

    def test_har_fires_at_or_above_min_obs_threshold(self):
        from services.earnings_vol_snapshot import _HAR_MIN_OBS

        # 140 periods → ~139 RS values, above _HAR_MIN_OBS=100
        snap = self._snap(self._price_df(140))
        self.assertIsNotNone(snap.rv_har_forecast)
        # No fallback note should be present when true HAR is used
        self.assertNotIn("rv_har_estimator", snap.null_reasons)

    def test_har_fallback_fires_below_min_obs_threshold(self):
        from services.earnings_vol_snapshot import _HAR_MIN_OBS

        # 80 periods → ~79 RS values: below HAR threshold, above fallback window
        snap = self._snap(self._price_df(80))
        self.assertIsNotNone(snap.rv_har_forecast)
        self.assertGreater(snap.rv_har_forecast, 0.0)
        # Fallback annotation must be present
        self.assertEqual(
            snap.null_reasons.get("rv_har_estimator"),
            f"rs_trailing_mean_fallback_n_lt_{_HAR_MIN_OBS}",
        )

    def test_both_har_and_fallback_return_none_when_too_sparse(self):
        # 25 periods → ~24 RS values: below both HAR (100) and fallback (30) thresholds
        snap = self._snap(self._price_df(25))
        self.assertIsNone(snap.rv_har_forecast)
        self.assertIn("rv_har_forecast", snap.null_reasons)
        self.assertIn(
            snap.null_reasons["rv_har_forecast"],
            {
                "insufficient_clean_rs_history_after_event_exclusion",
                "insufficient_rs_history_for_har",
            },
        )

    def test_fallback_produces_finite_positive_forecast(self):
        # 80 periods lands in fallback zone; result must be a positive finite number
        snap = self._snap(self._price_df(80))
        self.assertIsNotNone(snap.rv_har_forecast)
        self.assertTrue(
            snap.rv_har_forecast > 0 and not (snap.rv_har_forecast != snap.rv_har_forecast),
            "Fallback forecast must be finite and positive",
        )

    def test_event_implied_move_nonnegative_with_fallback(self):
        # Event decomposition must not produce negative values even on fallback path
        snap = self._snap(self._price_df(80))
        if snap.event_implied_move_pct is not None:
            self.assertGreaterEqual(snap.event_implied_move_pct, 0.0)

    def test_har_fallback_deterministic(self):
        # Same input → same output (no randomness in fallback)
        price_df = self._price_df(80)
        snap1 = self._snap(price_df)
        snap2 = self._snap(price_df)
        self.assertEqual(snap1.rv_har_forecast, snap2.rv_har_forecast)
