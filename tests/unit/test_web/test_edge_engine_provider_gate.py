"""Regression guard for the provider-gate (Gap A from the deep audit).

build_analysis_inputs once gated the injected market-data client with
`isinstance(client, MarketDataClient)`. The yfinance adapter
(YFinanceMarketDataClient) does NOT subclass MarketDataClient, so under the
yfinance-default provider the adapter was silently bypassed on the live
/api/edge/analyze path — the engine fell back to the cruder raw-yfinance term
structure and never used the adapter's normalized chain / surface-quality.

The gate is now duck-typed. These tests prove (1) a non-MarketDataClient client
with is_available()=True is actually USED (its get_option_chain is called and
the honest provider label propagates), and (2) the isinstance anti-pattern does
not return.
"""
from __future__ import annotations

import inspect
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from web.api import edge_engine
from services.structure_scorecard import StructureScorecard
from services.structure_selector import SelectorOutput
from services.earnings_vol_snapshot import VolSnapshot


def _minimal_snapshot() -> VolSnapshot:
    return VolSnapshot(
        symbol="AAPL", as_of_date=datetime(2024, 7, 1).date(), earnings_date=datetime(2024, 7, 9).date(),
        release_timing="after market close", days_to_earnings=8, underlying_price=102.2,
        option_source="provided", underlying_source="provided", price_staleness_minutes=0,
        chain_staleness_minutes=0, data_quality="high", data_quality_score=0.91,
        rv30_yang_zhang=0.21, rv30_estimator="yang_zhang", rv_har_forecast=0.19, rv_percentile_rank=48.0,
        vol_regime_label="Normal", iv30=0.27, iv45=0.30, near_term_dte=4, near_term_atm_iv=0.26,
        back_term_dte=25, back_term_atm_iv=0.30, near_back_iv_ratio=0.8667, term_structure_slope=0.0015,
        near_term_implied_move_pct=5.7, near_term_implied_sigma_pct=7.14,
        non_event_move_pct_har=1.0, event_implied_move_pct=5.61, event_move_share_of_total=0.98,
        historical_event_count=5, historical_median_move_pct=4.8, historical_avg_last4_move_pct=5.1,
        historical_p90_move_pct=6.9, historical_move_std_pct=1.4, historical_move_anchor_pct=4.995,
        historical_move_uncertainty_pct=0.80, historical_vs_implied_move_ratio=0.89,
        tail_vs_implied_move_ratio=1.23, smile_curvature=-0.62, smile_concavity_flag=True, smile_points=7,
        near_term_spread_pct=3.2, near_term_liquidity_proxy=2600.0, atm_call_spread_pct=3.0,
        atm_put_spread_pct=3.4, atm_total_open_interest=1700.0, atm_total_volume=900.0, liquidity_tier="low",
        iv_rv_yz=1.286, iv_rv_har=1.421, cheapness_score=0.41, event_risk_score=0.67, execution_score=0.72,
        timing_score=0.73, historical_move_source="earnings_history", null_reasons={},
        earnings_source_primary="fmp_calendar", earnings_source_confirmed="sec_submissions",
        earnings_source_confidence=0.90, release_timing_source="sec_submissions",
    )


def _minimal_scorecards():
    return [StructureScorecard(
        structure="atm_straddle", eligible=True, eligibility_flags=[], expected_edge_pct=2.4,
        expected_return_pct=4.8, expected_iv_contribution_pct=1.6, expected_move_fit_score=0.61,
        theta_drag_penalty=0.03, execution_penalty=0.05, crowding_penalty=0.06, concavity_penalty=0.03,
        sample_uncertainty_penalty=0.06, sample_confidence=0.22, walk_forward_history_count=4,
        walk_forward_win_rate=0.50, walk_forward_avg_return_pct=0.8, walk_forward_rank_score=0.42,
        composite_structure_score=0.58, rationale_bullets=["lead"])]


def _minimal_selector():
    return SelectorOutput(
        symbol="AAPL", as_of=datetime(2024, 7, 1).date(), earnings_date=datetime(2024, 7, 9).date(),
        release_timing="after market close", recommendation="Watch", best_structure="atm_straddle",
        confidence_pct=47.0, expected_edge_pct=2.4, expected_return_pct=4.8, expected_edge_tier="Positive",
        expected_return_signal="Supportive", model_output_note="note", primary_thesis="t",
        primary_risks=["r"], why_this_structure=["w"], why_not_others={}, runner_up_structures=[],
        data_quality="high", data_quality_score=0.91)


def _chain_df() -> pd.DataFrame:
    # A non-empty normalized chain so the provider path keeps chain_df and the
    # data_source label stays at the provider (not the fallback).
    rows = []
    for exp, dte in [("2024-07-12", 4), ("2024-08-02", 25)]:
        for side in ("call", "put"):
            for strike in (100.0, 102.0, 104.0):
                rows.append({
                    "expiration_date": exp, "strike": strike, "side": side,
                    "impliedVolatility": 0.27, "bid": 1.0, "ask": 1.2, "mid": 1.1,
                    "openInterest": 1000, "volume": 200, "dte": dte,
                    "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan,
                    "underlyingPrice": 102.2,
                })
    return pd.DataFrame(rows)


class TestProviderGate(unittest.TestCase):
    def test_duck_typed_yfinance_client_is_used_not_bypassed(self):
        hist = pd.DataFrame(
            {"Open": np.linspace(100, 102, 140), "High": np.linspace(101, 103, 140),
             "Low": np.linspace(99, 101, 140), "Close": np.linspace(100.2, 102.2, 140),
             "Volume": np.full(140, 5e6)},
            index=pd.bdate_range("2024-01-02", periods=140),
        )
        ticker = MagicMock()
        ticker.history.side_effect = [hist.iloc[-126:].copy(), hist.copy()]
        ticker.info = {}
        ticker.options = []
        ticker.get_earnings_dates.return_value = pd.DataFrame()

        # A duck-typed provider client that is NOT a MarketDataClient.
        client = MagicMock()
        client.provider_name = "yfinance"
        client.is_available.return_value = True
        client.get_option_chain.return_value = _chain_df()
        client.get_earnings.return_value = pd.DataFrame()

        ts_tuple = ([4.0, 25.0], [0.26, 0.30], np.nan, np.nan, 0.0, 0.0,
                    None, None, None, None, None, 4.0, 25.0)
        with patch.object(edge_engine.yf, "Ticker", return_value=ticker), \
             patch.object(edge_engine, "build_vol_snapshot", return_value=_minimal_snapshot()), \
             patch.object(edge_engine, "build_structure_scorecards", return_value=_minimal_scorecards()), \
             patch.object(edge_engine, "select_best_structure", return_value=_minimal_selector()), \
             patch.object(edge_engine, "_term_structure_from_mda_chain", return_value=ts_tuple), \
             patch.object(edge_engine, "_term_structure_points_yf", return_value=ts_tuple), \
             patch.object(edge_engine, "_historical_earnings_move_profile", return_value={"event_count": 5, "median_move_pct": 4.8, "p90_move_pct": 6.9, "avg_last4_move_pct": 5.1, "std_move_pct": 1.4, "raw_moves_pct": [4.2, 4.9, 5.1, 5.4, 6.9], "source": "earnings_history"}), \
             patch.object(edge_engine, "_summarize_pre_earnings_expansion", return_value={"available": False, "status": "no_simulations", "priceable_trades": 0, "ranking_score": None}), \
             patch.object(edge_engine, "_get_feature_store", return_value=None), \
             patch.object(edge_engine, "_get_pricing_risk_free_rate", return_value=(0.04, "test")), \
             patch.object(edge_engine, "get_dividend_yield", return_value=(0.0, "fallback_zero")), \
             patch("services.calibration_service.get_calibration") as gc:
            gc.return_value.diagnostics.return_value = {"phase": "observational", "n_observations": 58, "min_for_observational": 40, "min_for_fit": 120}
            result = edge_engine.analyze_single_ticker("AAPL", mda_client=client, record_to_ledger=False)

        # The gate accepted the duck-typed client → its chain/earnings were fetched.
        self.assertTrue(client.get_option_chain.called, "duck-typed provider client must be USED, not bypassed")
        self.assertTrue(client.get_earnings.called)
        # Honest provenance label (not 'marketdata_app', not the fallback).
        self.assertEqual(result.metrics["data_source"], "yfinance")
        if "options_source" in result.metrics:
            self.assertEqual(result.metrics["options_source"], "yfinance")

    def test_gate_is_duck_typed_not_isinstance(self):
        src = inspect.getsource(edge_engine.build_analysis_inputs)
        assert "isinstance(client, MarketDataClient)" not in src, (
            "the client gate must be duck-typed — an isinstance check makes the "
            "yfinance adapter dead code on the web path"
        )
        assert 'hasattr(client, "get_option_chain")' in src or "client.is_available()" in src


if __name__ == "__main__":
    unittest.main()
