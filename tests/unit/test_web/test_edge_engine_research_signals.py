import unittest

import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

from services.earnings_vol_snapshot import VolSnapshot, build_vol_snapshot
from services.screener_service import compute_ranking_score
from services.structure_scorecard import StructureScorecard
from services.structure_selector import SelectorOutput
import web.api.edge_engine as edge_engine
from web.api.edge_engine import (
    _classify_move_risk,
    _compute_move_uncertainty_pct,
    _compute_move_anchor,
    _get_pricing_risk_free_rate,
    _historical_earnings_move_profile,
    _normalize_release_timing,
    _simulate_pre_earnings_calendar_trade,
    _summarize_pre_earnings_expansion,
)


class _StubExpansionFeatureStore:
    def __init__(self, mapping):
        self.mapping = mapping

    def query_chain(self, symbol, **kwargs):
        key = (symbol, kwargs["trade_date"])
        frame = self.mapping.get(key)
        if frame is None:
            return pd.DataFrame()
        return frame.copy()


def _calendar_chain(trade_date, rows):
    base_rows = []
    for row in rows:
        base_rows.append(
            {
                "trade_date": trade_date,
                "expiry": row["expiry"],
                "underlying_symbol": "AAPL",
                "option_symbol": row.get("option_symbol", f"AAPL{trade_date.replace('-', '')}{row['expiry'].replace('-', '')}{int(row['strike'] * 1000)}"),
                "call_put": "C",
                "dte": row.get("dte", 10),
                "volume": row.get("volume", 100),
                "open_interest": row.get("open_interest", 400),
                "strike": row["strike"],
                "bid": row["bid"],
                "ask": row["ask"],
                "mid": row["mid"],
                "underlying_price": row.get("underlying_price", 100.0),
                "iv": row["iv"],
                "spread_pct": row.get("spread_pct", 5.0),
                "liquidity_score": row.get("liquidity_score", 1_000.0),
            }
        )
    return pd.DataFrame(base_rows)


class TestEdgeEngineResearchSignals(unittest.TestCase):
    def test_analyze_single_ticker_reads_snapshot_backed_feature_state(self):
        hist_dates = pd.bdate_range("2024-01-02", periods=140)
        hist_df = pd.DataFrame(
            {
                "Open": np.linspace(100.0, 102.0, len(hist_dates)),
                "High": np.linspace(101.0, 103.0, len(hist_dates)),
                "Low": np.linspace(99.0, 101.0, len(hist_dates)),
                "Close": np.linspace(100.2, 102.2, len(hist_dates)),
                "Volume": np.full(len(hist_dates), 5_000_000.0),
            },
            index=hist_dates,
        )

        ticker = MagicMock()
        ticker.history.side_effect = [hist_df.iloc[-126:].copy(), hist_df.copy()]
        ticker.info = {}
        ticker.options = []
        ticker.get_earnings_dates.return_value = pd.DataFrame()
        ticker.earnings_dates = None

        snapshot = VolSnapshot(
            symbol="AAPL",
            as_of_date=datetime(2024, 7, 1).date(),
            earnings_date=datetime(2024, 7, 9).date(),
            release_timing="after market close",
            days_to_earnings=8,
            underlying_price=102.2,
            option_source="provided",
            underlying_source="provided",
            price_staleness_minutes=0,
            chain_staleness_minutes=0,
            data_quality="high",
            data_quality_score=0.91,
            rv30_yang_zhang=0.21,
            rv30_estimator="yang_zhang",
            rv_har_forecast=0.19,
            rv_percentile_rank=48.0,
            vol_regime_label="Normal",
            iv30=0.27,
            iv45=0.30,
            near_term_dte=4,
            near_term_atm_iv=0.26,
            back_term_dte=25,
            back_term_atm_iv=0.30,
            near_back_iv_ratio=0.8667,
            term_structure_slope=0.0015,
            near_term_implied_move_pct=5.7,
            near_term_implied_sigma_pct=5.7 * 1.2533141373155001,  # MAD × √(π/2), P-5a
            non_event_move_pct_har=1.0,
            event_implied_move_pct=5.61,
            event_move_share_of_total=0.98,
            historical_event_count=5,
            historical_median_move_pct=4.8,
            historical_avg_last4_move_pct=5.1,
            historical_p90_move_pct=6.9,
            historical_move_std_pct=1.4,
            historical_move_anchor_pct=4.995,
            historical_move_uncertainty_pct=0.80,
            historical_vs_implied_move_ratio=0.89,
            tail_vs_implied_move_ratio=1.23,
            smile_curvature=-0.62,
            smile_concavity_flag=True,
            smile_points=7,
            near_term_spread_pct=3.2,
            near_term_liquidity_proxy=2600.0,
            atm_call_spread_pct=3.0,
            atm_put_spread_pct=3.4,
            atm_total_open_interest=1700.0,
            atm_total_volume=900.0,
            liquidity_tier="low",
            iv_rv_yz=1.286,
            iv_rv_har=1.421,
            cheapness_score=0.41,
            event_risk_score=0.67,
            execution_score=0.72,
            timing_score=0.73,
            historical_move_source="earnings_history",
            null_reasons={},
            earnings_source_primary="fmp_calendar",
            earnings_source_confirmed="sec_submissions",
            earnings_source_confidence=0.90,
            release_timing_source="sec_submissions",
        )

        scorecards = [
            StructureScorecard(
                structure="atm_straddle",
                eligible=True,
                eligibility_flags=[],
                expected_edge_pct=2.4,
                expected_return_pct=4.8,
                expected_iv_contribution_pct=1.6,
                expected_move_fit_score=0.61,
                theta_drag_penalty=0.03,
                execution_penalty=0.05,
                crowding_penalty=0.06,
                concavity_penalty=0.03,
                sample_uncertainty_penalty=0.06,
                sample_confidence=0.22,
                walk_forward_history_count=4,
                walk_forward_win_rate=0.50,
                walk_forward_avg_return_pct=0.8,
                walk_forward_rank_score=0.42,
                composite_structure_score=0.58,
                rationale_bullets=["Shared selector lead."],
            )
        ]
        selector = SelectorOutput(
            symbol="AAPL",
            as_of=datetime(2024, 7, 1).date(),
            earnings_date=datetime(2024, 7, 9).date(),
            release_timing="after market close",
            recommendation="Watch",
            best_structure="atm_straddle",
            confidence_pct=47.0,
            expected_edge_pct=2.4,
            expected_return_pct=4.8,
            expected_edge_tier="Positive",
            expected_return_signal="Supportive",
            model_output_note="Expected-edge and expected-return fields are score-derived diagnostics. They are not empirical return forecasts and are not calibrated to live retail execution.",
            primary_thesis="Shared thesis.",
            primary_risks=["Execution risk."],
            why_this_structure=["Shared lead."],
            why_not_others={"otm_strangle": ["Trails winner."]},
            runner_up_structures=["otm_strangle"],
            data_quality="high",
            data_quality_score=0.91,
        )

        with patch.object(edge_engine.yf, "Ticker", return_value=ticker):
            with patch.object(edge_engine, "build_vol_snapshot", return_value=snapshot):
                with patch.object(edge_engine, "build_structure_scorecards", return_value=scorecards):
                    with patch.object(edge_engine, "select_best_structure", return_value=selector):
                        with patch.object(edge_engine, "_term_structure_points_yf", return_value=([4.0, 25.0], [0.26, 0.30], np.nan, np.nan, 0.0, 0.0, None, None, None, None, None, 4.0, 25.0)):
                            with patch.object(edge_engine, "_historical_earnings_move_profile", return_value={
                                "event_count": 5,
                                "median_move_pct": 4.8,
                                "p90_move_pct": 6.9,
                                "avg_last4_move_pct": 5.1,
                                "std_move_pct": 1.4,
                                "raw_moves_pct": [4.2, 4.9, 5.1, 5.4, 6.9],
                                "source": "earnings_history",
                            }):
                                with patch.object(edge_engine, "_summarize_pre_earnings_expansion", return_value={
                                    "available": False,
                                    "status": "no_simulations",
                                    "priceable_trades": 0,
                                    "ranking_score": None,
                                }):
                                    with patch.object(edge_engine, "_get_feature_store", return_value=None):
                                        with patch.object(edge_engine, "_get_pricing_risk_free_rate", return_value=(0.04, "test")):
                                            with patch("services.calibration_service.get_calibration") as get_calibration:
                                                get_calibration.return_value.diagnostics.return_value = {
                                                    "phase": "observational",
                                                    "n_observations": 58,
                                                    "min_for_observational": 40,
                                                    "min_for_fit": 120,
                                                }
                                                result = edge_engine.analyze_single_ticker("AAPL", mda_client=None)

        self.assertAlmostEqual(result.metrics["iv_rv30"], snapshot.iv30 / snapshot.rv30_yang_zhang, places=8)
        self.assertAlmostEqual(result.metrics["rv30_har_forecast"], snapshot.rv_har_forecast, places=8)
        self.assertAlmostEqual(result.metrics["near_back_iv_ratio"], snapshot.near_back_iv_ratio, places=8)
        self.assertAlmostEqual(result.metrics["smile_curvature"], snapshot.smile_curvature, places=8)
        self.assertAlmostEqual(result.metrics["earnings_move_median_pct"], snapshot.historical_median_move_pct, places=8)
        self.assertAlmostEqual(result.metrics["earnings_move_anchor_pct"], snapshot.historical_move_anchor_pct, places=8)
        self.assertAlmostEqual(result.metrics["near_term_spread_pct"], snapshot.near_term_spread_pct, places=8)
        self.assertEqual(result.metrics["days_to_earnings"], snapshot.days_to_earnings)
        self.assertEqual(result.recommendation, "Watch")
        self.assertAlmostEqual(result.confidence_pct, 47.0, places=8)
        self.assertAlmostEqual(result.metrics["sample_confidence"], 0.22, places=8)
        self.assertEqual(result.metrics["edge_quality"], "Positive")
        self.assertEqual(result.metrics["confidence_score_source"], "shared_selector_confidence_pct")
        self.assertEqual(result.metrics["calibration_phase"], "observational")
        self.assertIn("raw bucket observations only", result.metrics["calibration_phase_note"])
        self.assertEqual(result.metrics["data_sources"]["earnings_source_primary"], "fmp_calendar")
        self.assertEqual(result.metrics["data_sources"]["earnings_source_confirmed"], "sec_submissions")
        self.assertEqual(result.metrics["data_sources"]["release_timing_source"], "sec_submissions")
        self.assertAlmostEqual(result.metrics["earnings_source_confidence"], 0.90, places=8)
        self.assertAlmostEqual(
            result.setup_score,
            compute_ranking_score(
                iv_rv_ratio=snapshot.iv_rv_yz,
                ts_ratio=snapshot.near_back_iv_ratio,
                median_earnings_move_pct=snapshot.historical_median_move_pct,
                sample_size=snapshot.historical_event_count,
                dte=snapshot.days_to_earnings,
                spread_pct=snapshot.near_term_spread_pct,
            ),
            places=8,
        )

    def test_selector_recommendation_overrides_legacy_duplicate_ivrv_opinion(self):
        hist_dates = pd.bdate_range("2024-01-02", periods=140)
        hist_df = pd.DataFrame(
            {
                "Open": np.linspace(100.0, 102.0, len(hist_dates)),
                "High": np.linspace(101.0, 103.0, len(hist_dates)),
                "Low": np.linspace(99.0, 101.0, len(hist_dates)),
                "Close": np.linspace(100.2, 102.2, len(hist_dates)),
                "Volume": np.full(len(hist_dates), 5_000_000.0),
            },
            index=hist_dates,
        )
        ticker = MagicMock()
        ticker.history.side_effect = [hist_df.iloc[-126:].copy(), hist_df.copy()]
        ticker.info = {}
        ticker.options = []
        ticker.get_earnings_dates.return_value = pd.DataFrame()
        ticker.earnings_dates = None

        snapshot = VolSnapshot(
            symbol="AAPL",
            as_of_date=datetime(2024, 7, 1).date(),
            earnings_date=datetime(2024, 7, 9).date(),
            release_timing="after market close",
            days_to_earnings=8,
            underlying_price=102.2,
            option_source="provided",
            underlying_source="provided",
            price_staleness_minutes=0,
            chain_staleness_minutes=0,
            data_quality="high",
            data_quality_score=0.91,
            rv30_yang_zhang=0.18,
            rv30_estimator="yang_zhang",
            rv_har_forecast=0.17,
            rv_percentile_rank=32.0,
            vol_regime_label="Normal",
            iv30=0.30,
            iv45=0.31,
            near_term_dte=4,
            near_term_atm_iv=0.30,
            back_term_dte=25,
            back_term_atm_iv=0.31,
            near_back_iv_ratio=0.97,
            term_structure_slope=0.0003,
            near_term_implied_move_pct=5.7,
            near_term_implied_sigma_pct=5.7 * 1.2533141373155001,  # MAD × √(π/2), P-5a
            non_event_move_pct_har=1.0,
            event_implied_move_pct=5.61,
            event_move_share_of_total=0.98,
            historical_event_count=5,
            historical_median_move_pct=4.8,
            historical_avg_last4_move_pct=5.1,
            historical_p90_move_pct=6.9,
            historical_move_std_pct=1.4,
            historical_move_anchor_pct=4.995,
            historical_move_uncertainty_pct=0.80,
            historical_vs_implied_move_ratio=0.89,
            tail_vs_implied_move_ratio=1.23,
            smile_curvature=-0.62,
            smile_concavity_flag=True,
            smile_points=7,
            near_term_spread_pct=3.2,
            near_term_liquidity_proxy=2600.0,
            atm_call_spread_pct=3.0,
            atm_put_spread_pct=3.4,
            atm_total_open_interest=1700.0,
            atm_total_volume=900.0,
            liquidity_tier="low",
            iv_rv_yz=1.67,
            iv_rv_har=1.76,
            cheapness_score=0.05,
            event_risk_score=0.67,
            execution_score=0.72,
            timing_score=0.73,
            historical_move_source="daily_fallback",
            null_reasons={},
        )
        scorecards = [
            StructureScorecard(
                structure="atm_straddle",
                eligible=True,
                eligibility_flags=[],
                expected_edge_pct=0.6,
                expected_return_pct=1.2,
                expected_iv_contribution_pct=0.4,
                expected_move_fit_score=0.15,
                theta_drag_penalty=0.03,
                execution_penalty=0.05,
                crowding_penalty=0.12,
                concavity_penalty=0.03,
                sample_uncertainty_penalty=0.10,
                sample_confidence=0.15,
                walk_forward_history_count=3,
                walk_forward_win_rate=0.50,
                walk_forward_avg_return_pct=0.4,
                walk_forward_rank_score=0.36,
                composite_structure_score=0.41,
                rationale_bullets=["Daily fallback degraded."],
            )
        ]
        selector = SelectorOutput(
            symbol="AAPL",
            as_of=datetime(2024, 7, 1).date(),
            earnings_date=datetime(2024, 7, 9).date(),
            release_timing="after market close",
            recommendation="No Trade",
            best_structure=None,
            confidence_pct=28.0,
            expected_edge_pct=0.0,
            expected_return_pct=0.0,
            expected_edge_tier="Negative / Unclear",
            expected_return_signal="Weak",
            model_output_note="Expected-edge and expected-return fields are score-derived diagnostics. They are not empirical return forecasts and are not calibrated to live retail execution.",
            primary_thesis="No trade.",
            primary_risks=["Crowded."],
            why_this_structure=["Abstain."],
            why_not_others={},
            runner_up_structures=[],
            data_quality="high",
            data_quality_score=0.91,
        )

        with patch.object(edge_engine.yf, "Ticker", return_value=ticker):
            with patch.object(edge_engine, "build_vol_snapshot", return_value=snapshot):
                with patch.object(edge_engine, "build_structure_scorecards", return_value=scorecards):
                    with patch.object(edge_engine, "select_best_structure", return_value=selector):
                        with patch.object(edge_engine, "_term_structure_points_yf", return_value=([4.0, 25.0], [0.30, 0.31], np.nan, np.nan, 0.0, 0.0, None, None, None, None, None, 4.0, 25.0)):
                            with patch.object(edge_engine, "_historical_earnings_move_profile", return_value={
                                "event_count": 0,
                                "median_move_pct": 4.8,
                                "p90_move_pct": 6.9,
                                "avg_last4_move_pct": 5.1,
                                "std_move_pct": 1.4,
                                "raw_moves_pct": [],
                                "source": "daily_fallback",
                            }):
                                with patch.object(edge_engine, "_summarize_pre_earnings_expansion", return_value={
                                    "available": False,
                                    "status": "no_simulations",
                                    "priceable_trades": 0,
                                    "ranking_score": None,
                                }):
                                    with patch.object(edge_engine, "_get_feature_store", return_value=None):
                                        with patch.object(edge_engine, "_get_pricing_risk_free_rate", return_value=(0.04, "test")):
                                            result = edge_engine.analyze_single_ticker("AAPL", mda_client=None)

        self.assertEqual(result.recommendation, "No Trade")
        self.assertAlmostEqual(result.confidence_pct, 28.0, places=8)
        self.assertAlmostEqual(result.metrics["sample_confidence"], 0.15, places=8)
        self.assertEqual(result.metrics["edge_quality"], "Negative / Unclear")
        self.assertEqual(result.metrics["iv_rv30"], snapshot.iv30 / snapshot.rv30_yang_zhang)

    def test_snapshot_service_matches_legacy_history_fields(self):
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

        price_df = pd.DataFrame(
            {
                "trade_date": dates,
                "open": prices.shift(1).fillna(prices.iloc[0] / 1.001).values,
                "high": (prices * 1.01).values,
                "low": (prices * 0.99).values,
                "close": prices.values,
            }
        )
        chain_df = pd.DataFrame(
            [
                {"trade_date": "2024-07-01", "expiry": "2024-07-12", "call_put": "C", "strike": 100.0, "bid": 2.9, "ask": 3.1, "mid": 3.0, "iv": 0.24, "open_interest": 1000, "volume": 500, "underlying_price": 100.0},
                {"trade_date": "2024-07-01", "expiry": "2024-07-12", "call_put": "P", "strike": 100.0, "bid": 2.8, "ask": 3.0, "mid": 2.9, "iv": 0.24, "open_interest": 1000, "volume": 500, "underlying_price": 100.0},
                {"trade_date": "2024-07-01", "expiry": "2024-08-16", "call_put": "C", "strike": 100.0, "bid": 3.4, "ask": 3.6, "mid": 3.5, "iv": 0.28, "open_interest": 1000, "volume": 500, "underlying_price": 100.0},
                {"trade_date": "2024-07-01", "expiry": "2024-08-16", "call_put": "P", "strike": 100.0, "bid": 3.3, "ask": 3.5, "mid": 3.4, "iv": 0.28, "open_interest": 1000, "volume": 500, "underlying_price": 100.0},
            ]
        )

        with patch.object(edge_engine, "_utc_today_date", return_value=datetime(2024, 7, 1).date()):
            legacy_profile = _historical_earnings_move_profile(close=prices, earnings_events=earnings_events)

        snapshot = build_vol_snapshot(
            "AAPL",
            datetime(2024, 7, 1),
            option_chain_data=chain_df,
            earnings_metadata={
                "earnings_date": "2024-07-25",
                "release_timing": "after market close",
                "prior_events": earnings_events,
            },
            price_data=price_df,
        )

        self.assertAlmostEqual(snapshot.historical_median_move_pct, legacy_profile["median_move_pct"], places=6)
        self.assertAlmostEqual(snapshot.historical_avg_last4_move_pct, legacy_profile["avg_last4_move_pct"], places=6)
        self.assertAlmostEqual(snapshot.historical_p90_move_pct, legacy_profile["p90_move_pct"], places=6)
        self.assertEqual(snapshot.historical_event_count, legacy_profile["event_count"])

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

    def test_pricing_risk_free_rate_env_override(self):
        import os

        edge_engine._rf_rate_cache.update({"ts": 0.0, "rate": None, "source": None})
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": "0.0415"}, clear=False):
            rate, source = _get_pricing_risk_free_rate(cache_ttl_seconds=0.0)

        self.assertAlmostEqual(rate, 0.0415, places=6)
        self.assertEqual(source, "env")

    def test_pricing_risk_free_rate_falls_back_when_irx_fetch_fails(self):
        import os

        edge_engine._rf_rate_cache.update({"ts": 0.0, "rate": None, "source": None})
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": ""}, clear=False):
            with patch.object(edge_engine.yf, "Ticker", side_effect=RuntimeError("network down")):
                rate, source = _get_pricing_risk_free_rate(cache_ttl_seconds=0.0)

        self.assertAlmostEqual(rate, 0.0450, places=6)
        self.assertEqual(source, "fallback_static")

    def test_pricing_risk_free_rate_uses_cache_within_ttl(self):
        import os

        ticker_mock = MagicMock()
        history_df = pd.DataFrame({"Close": [3.62]})
        ticker_mock.history.return_value = history_df

        edge_engine._rf_rate_cache.update({"ts": 0.0, "rate": None, "source": None})
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": ""}, clear=False):
            with patch.object(edge_engine.yf, "Ticker", return_value=ticker_mock) as ticker_ctor:
                rate_1, source_1 = _get_pricing_risk_free_rate(cache_ttl_seconds=3600.0)
                rate_2, source_2 = _get_pricing_risk_free_rate(cache_ttl_seconds=3600.0)

        self.assertAlmostEqual(rate_1, 0.0362, places=6)
        self.assertEqual(source_1, "yfinance_^IRX")
        self.assertAlmostEqual(rate_2, 0.0362, places=6)
        self.assertEqual(source_2, "yfinance_^IRX")
        ticker_ctor.assert_called_once_with("^IRX")

    def test_classify_move_risk_elevated(self):
        level, ratio = _classify_move_risk(9.2, 7.0, sample_size=8)
        self.assertEqual(level, "elevated")
        self.assertAlmostEqual(ratio, 9.2 / 7.0, places=6)

    def test_classify_move_risk_moderate(self):
        level, ratio = _classify_move_risk(7.0, 7.0, sample_size=8)
        self.assertEqual(level, "moderate")
        self.assertAlmostEqual(ratio, 1.0, places=6)

    def test_classify_move_risk_low(self):
        level, ratio = _classify_move_risk(5.5, 7.0, sample_size=8)
        self.assertEqual(level, "low")
        self.assertAlmostEqual(ratio, 5.5 / 7.0, places=6)

    def test_classify_move_risk_low_downgrades_on_thin_sample(self):
        level, ratio = _classify_move_risk(5.5, 7.0, sample_size=4)
        self.assertEqual(level, "moderate")
        self.assertAlmostEqual(ratio, 5.5 / 7.0, places=6)

    def test_classify_move_risk_unknown_without_inputs(self):
        level, ratio = _classify_move_risk(None, 7.0, sample_size=8)
        self.assertEqual(level, "unknown")
        self.assertIsNone(ratio)

    def test_max_near_term_spread_threshold_matches_scorecard_eligibility(self):
        """Edge-engine hard gate must mirror scorecard eligibility threshold.

        Drift between MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE and
        ABSOLUTE_SPREAD_THRESHOLD_PCT produces the contradictory state where the
        selector returns NO_TRADE on ineligibility while the edge-engine gate
        does not fire (or vice versa).
        """
        from services.structure_scorecard import ABSOLUTE_SPREAD_THRESHOLD_PCT
        self.assertEqual(
            edge_engine.MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE,
            ABSOLUTE_SPREAD_THRESHOLD_PCT,
            msg=(
                "Edge-engine spread hard gate must equal scorecard eligibility "
                "threshold; otherwise setups in the gap produce contradictory "
                "selector vs gate output."
            ),
        )

    def test_max_near_term_spread_pct_for_trade_is_twelve(self):
        """Pin the absolute value at 12.0 (De Silva 2025 calibration)."""
        self.assertAlmostEqual(
            edge_engine.MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE, 12.0, places=6
        )

    def test_spread_thirteen_pct_triggers_hard_gate_condition(self):
        """A 13% near-term spread must satisfy the hard-gate trigger condition.

        The gate at the analyze_single_ticker path fires when
            spread_val > MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE.
        13.0 > 12.0 must hold (previously 13.0 > 18.0 was False, leaving the
        scorecard ineligibility unaccompanied by a hard gate trigger).
        """
        self.assertGreater(13.0, edge_engine.MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE)

    def test_spread_twelve_pct_does_not_trigger_hard_gate(self):
        """At exactly 12.0% the gate must not fire (condition is strictly greater).

        Mirrors test_spread_exactly_twelve_is_eligible in test_hardening_pass.py
        so the edge-engine gate and the scorecard agree on the boundary.
        """
        self.assertFalse(12.0 > edge_engine.MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE)

    def test_simulate_pre_earnings_calendar_trade_prices_exact_contracts(self):
        store = _StubExpansionFeatureStore(
            {
                ("AAPL", "2024-01-10"): _calendar_chain(
                    "2024-01-10",
                    [
                        {"expiry": "2024-01-19", "strike": 100.0, "bid": 2.0, "ask": 2.2, "mid": 2.1, "iv": 0.40},
                        {"expiry": "2024-02-16", "strike": 100.0, "bid": 3.5, "ask": 3.7, "mid": 3.6, "iv": 0.45},
                        {"expiry": "2024-01-19", "strike": 105.0, "bid": 0.8, "ask": 1.0, "mid": 0.9, "iv": 0.39},
                        {"expiry": "2024-02-16", "strike": 105.0, "bid": 1.6, "ask": 1.8, "mid": 1.7, "iv": 0.43},
                    ],
                ),
                ("AAPL", "2024-01-16"): _calendar_chain(
                    "2024-01-16",
                    [
                        {"expiry": "2024-01-19", "strike": 100.0, "bid": 2.5, "ask": 2.7, "mid": 2.6, "iv": 0.52},
                        {"expiry": "2024-02-16", "strike": 100.0, "bid": 4.5, "ask": 4.7, "mid": 4.6, "iv": 0.50},
                    ],
                ),
            }
        )

        trade = _simulate_pre_earnings_calendar_trade(
            "AAPL",
            store,
            event_date=pd.Timestamp("2024-01-17"),
            entry_date=pd.Timestamp("2024-01-10"),
            exit_date=pd.Timestamp("2024-01-16"),
            entry_offset=5,
        )

        self.assertEqual(trade["status"], "priceable")
        self.assertEqual(trade["strike"], 100.0)
        self.assertEqual(trade["front_expiry"], "2024-01-19")
        self.assertEqual(trade["back_expiry"], "2024-02-16")
        self.assertAlmostEqual(trade["entry_debit_mid"], 1.5, places=6)
        self.assertAlmostEqual(trade["exit_value_mid"], 2.0, places=6)
        self.assertAlmostEqual(trade["mid_pnl"], 0.5, places=6)
        self.assertAlmostEqual(trade["adjusted_pnl"], 0.1, places=6)

    def test_simulate_pre_earnings_calendar_trade_flags_missing_exact_exit_quote(self):
        store = _StubExpansionFeatureStore(
            {
                ("AAPL", "2024-01-10"): _calendar_chain(
                    "2024-01-10",
                    [
                        {"expiry": "2024-01-19", "strike": 100.0, "bid": 2.0, "ask": 2.2, "mid": 2.1, "iv": 0.40},
                        {"expiry": "2024-02-16", "strike": 100.0, "bid": 3.5, "ask": 3.7, "mid": 3.6, "iv": 0.45},
                    ],
                ),
                ("AAPL", "2024-01-16"): _calendar_chain(
                    "2024-01-16",
                    [
                        {"expiry": "2024-01-19", "strike": 105.0, "bid": 1.1, "ask": 1.3, "mid": 1.2, "iv": 0.41},
                    ],
                ),
            }
        )

        trade = _simulate_pre_earnings_calendar_trade(
            "AAPL",
            store,
            event_date=pd.Timestamp("2024-01-17"),
            entry_date=pd.Timestamp("2024-01-10"),
            exit_date=pd.Timestamp("2024-01-16"),
            entry_offset=5,
        )

        self.assertEqual(trade["status"], "missing_exit_quote")

    def test_summarize_pre_earnings_expansion_returns_selected_window_metrics(self):
        dates = pd.bdate_range("2024-01-02", periods=40)
        prices = pd.Series(np.linspace(100.0, 104.0, len(dates)), index=dates)
        event_date = pd.Timestamp("2024-01-17")

        store = _StubExpansionFeatureStore(
            {
                ("AAPL", "2024-01-10"): _calendar_chain(
                    "2024-01-10",
                    [
                        {"expiry": "2024-01-19", "strike": 100.0, "bid": 2.0, "ask": 2.2, "mid": 2.1, "iv": 0.40},
                        {"expiry": "2024-02-16", "strike": 100.0, "bid": 3.5, "ask": 3.7, "mid": 3.6, "iv": 0.45},
                    ],
                ),
                ("AAPL", "2024-01-16"): _calendar_chain(
                    "2024-01-16",
                    [
                        {"expiry": "2024-01-19", "strike": 100.0, "bid": 2.5, "ask": 2.7, "mid": 2.6, "iv": 0.52},
                        {"expiry": "2024-02-16", "strike": 100.0, "bid": 4.5, "ask": 4.7, "mid": 4.6, "iv": 0.50},
                    ],
                ),
            }
        )

        summary = _summarize_pre_earnings_expansion(
            "AAPL",
            prices,
            [{"event_date": event_date, "release_timing": "after market close"}],
            store,
            entry_offsets=(5,),
            exit_offset=1,
        )

        self.assertTrue(summary["available"])
        self.assertEqual(summary["selected_entry_offset_days"], 5)
        self.assertEqual(summary["priceable_trades"], 1)
        self.assertAlmostEqual(summary["expected_pnl_mid"], 0.5, places=6)
        self.assertAlmostEqual(summary["expected_pnl_adjusted"], 0.1, places=6)
        self.assertGreater(summary["ranking_score"], 0.0)


# ──────────────────────────────────────────────────────────────────────────
# PR #68 Codex follow-up: real analyze_single_ticker call-chain regression
# ──────────────────────────────────────────────────────────────────────────


class TestAnalyzeSingleTickerDividendYieldCallChain(unittest.TestCase):
    """Codex P2 follow-up on PR #68.

    The original PR #68 dividend-yield tests verified the BSM helpers in
    isolation and patched ``get_dividend_yield`` at the module level —
    but never proved that ``analyze_single_ticker`` actually invokes
    ``get_dividend_yield`` and threads the resolved ``q`` into the BSM
    helpers. A future refactor that accidentally drops
    ``q=pricing_dividend_yield`` from a call site would silently revert
    to the legacy no-dividend behavior with no test failure.

    This test closes that gap. Strategy:

      1. Mock the heavy upstream dependencies (yfinance, vol snapshot,
         scorecards, selector) so ``analyze_single_ticker`` can run
         far enough to reach the first BSM call.
      2. Patch ``get_dividend_yield`` to return a known non-fallback
         value: ``(0.025, "test_mock")``.
      3. Patch ``_bsm_greeks`` with a MagicMock that records its
         call args.
      4. After the call, assert:
         - ``_bsm_greeks`` was invoked with ``q=0.025`` (call-chain
           threading is intact).
         - ``result.metrics["pricing_dividend_yield"] == 0.025`` and
           ``result.metrics["pricing_dividend_yield_source"] == "test_mock"``
           (the operator-visible diagnostic surface).

    Without this test, the call chain could silently break and only
    surface as wrong prices on dividend names in production.
    """

    def _build_minimal_snapshot(self):
        """A minimal VolSnapshot sufficient to reach the BSM call site
        in analyze_single_ticker. Borrows the shape from the existing
        test_analyze_single_ticker_reads_snapshot_backed_feature_state
        fixture but trims to the field set actually consumed before
        ``_bsm_greeks`` runs."""
        return VolSnapshot(
            symbol="AAPL",
            as_of_date=datetime(2024, 7, 1).date(),
            earnings_date=datetime(2024, 7, 9).date(),
            release_timing="after market close",
            days_to_earnings=8,
            underlying_price=102.2,
            option_source="provided",
            underlying_source="provided",
            price_staleness_minutes=0,
            chain_staleness_minutes=0,
            data_quality="high",
            data_quality_score=0.91,
            rv30_yang_zhang=0.21,
            rv30_estimator="yang_zhang",
            rv_har_forecast=0.19,
            rv_percentile_rank=48.0,
            vol_regime_label="Normal",
            iv30=0.27,
            iv45=0.30,
            near_term_dte=4,
            near_term_atm_iv=0.26,
            back_term_dte=25,
            back_term_atm_iv=0.30,
            near_back_iv_ratio=0.8667,
            term_structure_slope=0.0015,
            near_term_implied_move_pct=5.7,
            near_term_implied_sigma_pct=5.7 * 1.2533141373155001,
            non_event_move_pct_har=1.0,
            event_implied_move_pct=5.61,
            event_move_share_of_total=0.98,
            historical_event_count=5,
            historical_median_move_pct=4.8,
            historical_avg_last4_move_pct=5.1,
            historical_p90_move_pct=6.9,
            historical_move_std_pct=1.4,
            historical_move_anchor_pct=4.995,
            historical_move_uncertainty_pct=0.80,
            historical_vs_implied_move_ratio=0.89,
            tail_vs_implied_move_ratio=1.23,
            smile_curvature=-0.62,
            smile_concavity_flag=True,
            smile_points=7,
            near_term_spread_pct=3.2,
            near_term_liquidity_proxy=2600.0,
            atm_call_spread_pct=3.0,
            atm_put_spread_pct=3.4,
            atm_total_open_interest=1700.0,
            atm_total_volume=900.0,
            liquidity_tier="low",
            iv_rv_yz=1.286,
            iv_rv_har=1.421,
            cheapness_score=0.41,
            event_risk_score=0.67,
            execution_score=0.72,
            timing_score=0.73,
            historical_move_source="earnings_history",
            null_reasons={},
            earnings_source_primary="fmp_calendar",
            earnings_source_confirmed="sec_submissions",
            earnings_source_confidence=0.90,
            release_timing_source="sec_submissions",
        )

    def _build_minimal_scorecards(self):
        return [
            StructureScorecard(
                structure="atm_straddle",
                eligible=True,
                eligibility_flags=[],
                expected_edge_pct=2.4,
                expected_return_pct=4.8,
                expected_iv_contribution_pct=1.6,
                expected_move_fit_score=0.61,
                theta_drag_penalty=0.03,
                execution_penalty=0.05,
                crowding_penalty=0.06,
                concavity_penalty=0.03,
                sample_uncertainty_penalty=0.06,
                sample_confidence=0.22,
                walk_forward_history_count=4,
                walk_forward_win_rate=0.50,
                walk_forward_avg_return_pct=0.8,
                walk_forward_rank_score=0.42,
                composite_structure_score=0.58,
                rationale_bullets=["Test fixture lead."],
            )
        ]

    def _build_minimal_selector(self):
        return SelectorOutput(
            symbol="AAPL",
            as_of=datetime(2024, 7, 1).date(),
            earnings_date=datetime(2024, 7, 9).date(),
            release_timing="after market close",
            recommendation="Watch",
            best_structure="atm_straddle",
            confidence_pct=47.0,
            expected_edge_pct=2.4,
            expected_return_pct=4.8,
            expected_edge_tier="Positive",
            expected_return_signal="Supportive",
            model_output_note="test fixture note",
            primary_thesis="Test thesis.",
            primary_risks=["Test risk."],
            why_this_structure=["Test lead."],
            why_not_others={},
            runner_up_structures=[],
            data_quality="high",
            data_quality_score=0.91,
        )

    def test_q_resolved_and_threaded_into_bsm_greeks(self):
        """The load-bearing assertion: get_dividend_yield is called,
        the value is threaded into _bsm_greeks, AND the resolved
        yield + source appear in the metrics response."""
        hist_dates = pd.bdate_range("2024-01-02", periods=140)
        hist_df = pd.DataFrame(
            {
                "Open": np.linspace(100.0, 102.0, len(hist_dates)),
                "High": np.linspace(101.0, 103.0, len(hist_dates)),
                "Low": np.linspace(99.0, 101.0, len(hist_dates)),
                "Close": np.linspace(100.2, 102.2, len(hist_dates)),
                "Volume": np.full(len(hist_dates), 5_000_000.0),
            },
            index=hist_dates,
        )

        ticker = MagicMock()
        ticker.history.side_effect = [hist_df.iloc[-126:].copy(), hist_df.copy()]
        ticker.info = {}
        ticker.options = []
        ticker.get_earnings_dates.return_value = pd.DataFrame()
        ticker.earnings_dates = None

        snapshot = self._build_minimal_snapshot()
        scorecards = self._build_minimal_scorecards()
        selector = self._build_minimal_selector()

        # MagicMock for _bsm_greeks. Returns a sensible dict so the
        # downstream code (which destructures greeks values) doesn't
        # NaN-cascade. The call-args inspection is what we actually
        # care about.
        bsm_mock = MagicMock(return_value={
            "delta_call": 0.50, "delta_put": -0.50,
            "gamma": 0.01, "vega": 0.10,
            "theta_call": -0.02, "theta_put": -0.02,
        })

        with patch.object(edge_engine.yf, "Ticker", return_value=ticker), \
             patch.object(edge_engine, "build_vol_snapshot", return_value=snapshot), \
             patch.object(edge_engine, "build_structure_scorecards", return_value=scorecards), \
             patch.object(edge_engine, "select_best_structure", return_value=selector), \
             patch.object(
                 edge_engine, "_term_structure_points_yf",
                 return_value=(
                     [4.0, 25.0], [0.26, 0.30], np.nan, np.nan, 0.0, 0.0,
                     None, None, None, None, None, 4.0, 25.0,
                 ),
             ), \
             patch.object(
                 edge_engine, "_historical_earnings_move_profile",
                 return_value={
                     "event_count": 5,
                     "median_move_pct": 4.8,
                     "p90_move_pct": 6.9,
                     "avg_last4_move_pct": 5.1,
                     "std_move_pct": 1.4,
                     "raw_moves_pct": [4.2, 4.9, 5.1, 5.4, 6.9],
                     "source": "earnings_history",
                 },
             ), \
             patch.object(
                 edge_engine, "_summarize_pre_earnings_expansion",
                 return_value={
                     "available": False,
                     "status": "no_simulations",
                     "priceable_trades": 0,
                     "ranking_score": None,
                 },
             ), \
             patch.object(edge_engine, "_get_feature_store", return_value=None), \
             patch.object(
                 edge_engine, "_get_pricing_risk_free_rate",
                 return_value=(0.04, "test"),
             ), \
             patch.object(
                 edge_engine, "get_dividend_yield",
                 return_value=(0.025, "test_mock"),
             ) as div_mock, \
             patch.object(edge_engine, "_bsm_greeks", bsm_mock), \
             patch("services.calibration_service.get_calibration") as get_calibration:
            get_calibration.return_value.diagnostics.return_value = {
                "phase": "observational",
                "n_observations": 58,
                "min_for_observational": 40,
                "min_for_fit": 120,
            }
            result = edge_engine.analyze_single_ticker("AAPL", mda_client=None)

        # 1. get_dividend_yield was actually called from
        #    analyze_single_ticker with the clean symbol.
        div_mock.assert_called_once_with("AAPL")

        # 2. _bsm_greeks received the resolved q as a keyword arg.
        #    The pre-PR-#68 code path passed no `q` (defaulted to 0.0).
        #    A future refactor that drops the threading would fail
        #    this assertion immediately.
        self.assertTrue(bsm_mock.called, "_bsm_greeks must be invoked")
        bsm_call_kwargs = bsm_mock.call_args.kwargs
        self.assertIn("q", bsm_call_kwargs, (
            "_bsm_greeks must be called with q= as a keyword arg; "
            "a regression that drops the threading would surface here."
        ))
        self.assertEqual(bsm_call_kwargs["q"], 0.025)

        # 3. The metrics surface exposes the resolved dividend yield
        #    + source so operators can see whether real data drove
        #    pricing or the fallback_zero path was taken.
        self.assertEqual(result.metrics["pricing_dividend_yield"], 0.025)
        self.assertEqual(
            result.metrics["pricing_dividend_yield_source"], "test_mock",
        )


if __name__ == "__main__":
    unittest.main()
