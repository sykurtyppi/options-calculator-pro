"""
Tests for services.pricing_rates — single source of truth for the BSM
pricing risk-free rate.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import services.pricing_rates as pr
from services.pricing_rates import (
    DEFAULT_CACHE_TTL_SECONDS,
    FALLBACK_STATIC_RATE,
    MAX_VALID_RATE,
    MIN_VALID_RATE,
    get_pricing_risk_free_rate,
    reset_cache,
)


class TestPricingRatesEnvOverride(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_env_override_takes_precedence(self):
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": "0.0415"}, clear=False):
            rate, source = get_pricing_risk_free_rate(cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, 0.0415, places=6)
        self.assertEqual(source, "env")

    def test_env_override_outside_bounds_is_ignored(self):
        """Rates ≥ MAX_VALID_RATE or ≤ MIN_VALID_RATE must be skipped, not used."""
        # 30% is above the 25% ceiling — should fall through to static fallback.
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": "0.30"}, clear=False):
            with patch.object(pr.yf, "Ticker", side_effect=RuntimeError("no network")):
                rate, source = get_pricing_risk_free_rate(cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, FALLBACK_STATIC_RATE, places=6)
        self.assertEqual(source, "fallback_static")


class TestPricingRatesIRX(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_uses_yfinance_irx_when_env_absent(self):
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame({"Close": [3.62]})
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": ""}, clear=False):
            with patch.object(pr.yf, "Ticker", return_value=ticker_mock):
                rate, source = get_pricing_risk_free_rate(cache_ttl_seconds=0.0)
        # ^IRX quotes in percent → rate = close / 100
        self.assertAlmostEqual(rate, 0.0362, places=6)
        self.assertEqual(source, "yfinance_^IRX")

    def test_falls_back_when_irx_returns_empty_history(self):
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame({"Close": []})
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": ""}, clear=False):
            with patch.object(pr.yf, "Ticker", return_value=ticker_mock):
                rate, source = get_pricing_risk_free_rate(cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, FALLBACK_STATIC_RATE, places=6)
        self.assertEqual(source, "fallback_static")

    def test_falls_back_when_irx_raises(self):
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": ""}, clear=False):
            with patch.object(pr.yf, "Ticker", side_effect=RuntimeError("network down")):
                rate, source = get_pricing_risk_free_rate(cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, FALLBACK_STATIC_RATE, places=6)
        self.assertEqual(source, "fallback_static")

    def test_irx_outside_bounds_is_ignored(self):
        """If ^IRX returns a nonsensical value (e.g. 30% short-rate),
        the resolver must skip it and fall back to the static rate."""
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame({"Close": [30.5]})  # 30.5%
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": ""}, clear=False):
            with patch.object(pr.yf, "Ticker", return_value=ticker_mock):
                rate, source = get_pricing_risk_free_rate(cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, FALLBACK_STATIC_RATE, places=6)
        self.assertEqual(source, "fallback_static")


class TestPricingRatesCache(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_uses_cache_within_ttl(self):
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame({"Close": [3.62]})
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": ""}, clear=False):
            with patch.object(pr.yf, "Ticker", return_value=ticker_mock) as ctor:
                rate1, _ = get_pricing_risk_free_rate(cache_ttl_seconds=3600.0)
                rate2, _ = get_pricing_risk_free_rate(cache_ttl_seconds=3600.0)
        self.assertAlmostEqual(rate1, 0.0362, places=6)
        self.assertAlmostEqual(rate2, 0.0362, places=6)
        # Constructor only called once — second call hit cache.
        ctor.assert_called_once_with("^IRX")

    def test_zero_ttl_bypasses_cache(self):
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame({"Close": [3.62]})
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": ""}, clear=False):
            with patch.object(pr.yf, "Ticker", return_value=ticker_mock) as ctor:
                get_pricing_risk_free_rate(cache_ttl_seconds=0.0)
                get_pricing_risk_free_rate(cache_ttl_seconds=0.0)
        # cache_ttl_seconds=0.0 means the cached value is never fresh enough,
        # so each call goes through the env / yfinance path again.
        self.assertEqual(ctor.call_count, 2)

    def test_reset_cache_clears_state(self):
        with patch.dict(os.environ, {"OPTIONS_PRICING_RISK_FREE_RATE": "0.0415"}, clear=False):
            get_pricing_risk_free_rate(cache_ttl_seconds=3600.0)
        self.assertEqual(pr._rf_rate_cache.get("source"), "env")
        reset_cache()
        self.assertIsNone(pr._rf_rate_cache.get("rate"))
        self.assertIsNone(pr._rf_rate_cache.get("source"))


class TestEdgeEngineWiring(unittest.TestCase):
    """Drift detection: the edge_engine module must re-export the shared
    pricing-rate names so existing callers and patch-based tests keep working."""

    def test_edge_engine_aliases_point_to_shared_module(self):
        import web.api.edge_engine as ee
        self.assertIs(ee._get_pricing_risk_free_rate, get_pricing_risk_free_rate)
        self.assertIs(ee._rf_rate_cache, pr._rf_rate_cache)
        self.assertIs(ee._rf_rate_lock, pr._rf_rate_lock)


class TestConstants(unittest.TestCase):
    def test_default_cache_ttl_is_twelve_hours(self):
        self.assertEqual(DEFAULT_CACHE_TTL_SECONDS, 43_200.0)

    def test_fallback_rate_within_bounds(self):
        self.assertGreater(FALLBACK_STATIC_RATE, MIN_VALID_RATE)
        self.assertLess(FALLBACK_STATIC_RATE, MAX_VALID_RATE)


if __name__ == "__main__":
    unittest.main()
