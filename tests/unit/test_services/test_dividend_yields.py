"""Tests for services.dividend_yields — per-symbol continuous dividend yield
resolution used by BSM pricing.

The resolution chain mirrors services.pricing_rates: env override → yfinance
→ fallback zero. Tests cover each layer in isolation plus the bounds.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import services.dividend_yields as dy
from services.dividend_yields import (
    DEFAULT_CACHE_TTL_SECONDS,
    MAX_VALID_YIELD,
    MIN_VALID_YIELD,
    SOURCE_ENV,
    SOURCE_FALLBACK_ZERO,
    SOURCE_YFINANCE,
    get_dividend_yield,
    reset_cache,
)


class TestEnvOverride(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_per_symbol_env_override_is_used(self):
        with patch.dict(
            os.environ, {"OPTIONS_DIVIDEND_YIELD_AAPL": "0.0050"}, clear=False
        ):
            rate, source = get_dividend_yield("AAPL", cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, 0.0050, places=6)
        self.assertEqual(source, SOURCE_ENV)

    def test_env_override_normalises_symbol_with_dot(self):
        """BRK.B → OPTIONS_DIVIDEND_YIELD_BRK_B (non-identifier chars → _)."""
        with patch.dict(
            os.environ, {"OPTIONS_DIVIDEND_YIELD_BRK_B": "0.0123"}, clear=False
        ):
            rate, source = get_dividend_yield("BRK.B", cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, 0.0123, places=6)
        self.assertEqual(source, SOURCE_ENV)

    def test_env_override_outside_bounds_falls_through(self):
        # 50% yield is above the 20% ceiling — must be rejected and fall
        # through to yfinance (mocked to fail) then to fallback_zero.
        with patch.dict(
            os.environ, {"OPTIONS_DIVIDEND_YIELD_AAPL": "0.50"}, clear=False
        ):
            with patch.object(dy, "_dividend_cache", {}):
                with patch("yfinance.Ticker", side_effect=RuntimeError("blocked")):
                    rate, source = get_dividend_yield("AAPL", cache_ttl_seconds=0.0)
        self.assertEqual(rate, 0.0)
        self.assertEqual(source, SOURCE_FALLBACK_ZERO)

    def test_negative_env_override_is_rejected(self):
        with patch.dict(
            os.environ, {"OPTIONS_DIVIDEND_YIELD_AAPL": "-0.01"}, clear=False
        ):
            with patch.object(dy, "_dividend_cache", {}):
                with patch("yfinance.Ticker", side_effect=RuntimeError("blocked")):
                    rate, source = get_dividend_yield("AAPL", cache_ttl_seconds=0.0)
        self.assertEqual(rate, 0.0)
        self.assertEqual(source, SOURCE_FALLBACK_ZERO)


class TestYfinanceFetch(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_yfinance_trailing_yield_is_a_decimal(self):
        """trailingAnnualDividendYield is always a decimal (e.g. 0.0085)."""
        ticker_mock = MagicMock()
        ticker_mock.info = {"trailingAnnualDividendYield": 0.0085}
        with patch.object(dy, "_dividend_cache", {}):
            with patch("yfinance.Ticker", return_value=ticker_mock):
                rate, source = get_dividend_yield("KO", cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, 0.0085, places=6)
        self.assertEqual(source, SOURCE_YFINANCE)

    def test_dividend_rate_over_price_wins(self):
        """Annual cash dividend / price has no unit ambiguity."""
        ticker_mock = MagicMock()
        ticker_mock.info = {"dividendRate": 1.94, "regularMarketPrice": 70.0,
                            "trailingAnnualDividendYield": 0.5, "dividendYield": 2.77}
        with patch.object(dy, "_dividend_cache", {}):
            with patch("yfinance.Ticker", return_value=ticker_mock):
                rate, source = get_dividend_yield("KO", cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, 1.94 / 70.0, places=9)
        self.assertEqual(source, SOURCE_YFINANCE)

    def test_ambiguous_dividend_yield_alone_is_not_trusted(self):
        """0.15 could be 15% or 0.15%: without an unambiguous field the yield
        falls back to 0 (flagged in the UI) instead of guessing a unit."""
        for raw in (0.02, 0.15, 0.44, 2.6):
            reset_cache()
            ticker_mock = MagicMock()
            ticker_mock.info = {"dividendYield": raw}
            with patch.object(dy, "_dividend_cache", {}):
                with patch("yfinance.Ticker", return_value=ticker_mock):
                    self.assertEqual(get_dividend_yield("KO", cache_ttl_seconds=0.0),
                                     (0.0, SOURCE_FALLBACK_ZERO))

    def test_low_yields_are_not_scaled_by_100(self):
        """A 0.15% payer (dividendRate 0.27 at $180) stays 0.0015, not 15%."""
        ticker_mock = MagicMock()
        ticker_mock.info = {"dividendRate": 0.27, "currentPrice": 180.0, "dividendYield": 0.15}
        with patch.object(dy, "_dividend_cache", {}):
            with patch("yfinance.Ticker", return_value=ticker_mock):
                rate, _ = get_dividend_yield("XYZ", cache_ttl_seconds=0.0)
        self.assertAlmostEqual(rate, 0.0015, places=9)

    def test_yfinance_missing_dividend_yields_zero(self):
        """No dividend fields → fall through to zero. Standard case for
        most growth names (TSLA, AMZN, GOOG, etc.)."""
        ticker_mock = MagicMock()
        ticker_mock.info = {"symbol": "TSLA"}  # no dividend fields
        with patch.object(dy, "_dividend_cache", {}):
            with patch("yfinance.Ticker", return_value=ticker_mock):
                rate, source = get_dividend_yield("TSLA", cache_ttl_seconds=0.0)
        self.assertEqual(rate, 0.0)
        self.assertEqual(source, SOURCE_FALLBACK_ZERO)

    def test_yfinance_failure_yields_zero(self):
        """Exception inside yfinance lookup → fall through to zero (safe default,
        matches legacy BSM behavior of ignoring dividends)."""
        with patch.object(dy, "_dividend_cache", {}):
            with patch("yfinance.Ticker", side_effect=RuntimeError("network down")):
                rate, source = get_dividend_yield("AAPL", cache_ttl_seconds=0.0)
        self.assertEqual(rate, 0.0)
        self.assertEqual(source, SOURCE_FALLBACK_ZERO)


class TestCaching(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_second_call_within_ttl_uses_cache(self):
        ticker_mock = MagicMock()
        ticker_mock.info = {"trailingAnnualDividendYield": 0.0042}
        with patch("yfinance.Ticker", return_value=ticker_mock) as ticker_patch:
            first_rate, first_source = get_dividend_yield("MSFT")
            second_rate, second_source = get_dividend_yield("MSFT")
        self.assertAlmostEqual(first_rate, second_rate, places=6)
        self.assertEqual(first_source, SOURCE_YFINANCE)
        # Served from cache, reporting its original source.
        self.assertEqual(second_source, SOURCE_YFINANCE)
        ticker_patch.assert_called_once()  # second lookup served from cache

    def test_cached_fallback_still_reads_as_fallback(self):
        with patch("yfinance.Ticker", side_effect=RuntimeError("network down")) as ticker_patch:
            first = get_dividend_yield("KO")
            second = get_dividend_yield("KO")
        self.assertEqual(first, (0.0, SOURCE_FALLBACK_ZERO))
        self.assertEqual(second, (0.0, SOURCE_FALLBACK_ZERO))
        ticker_patch.assert_called_once()


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_empty_symbol_returns_zero_fallback(self):
        rate, source = get_dividend_yield("")
        self.assertEqual(rate, 0.0)
        self.assertEqual(source, SOURCE_FALLBACK_ZERO)

    def test_symbol_is_uppercased_before_cache_lookup(self):
        """Calling get_dividend_yield('aapl') then 'AAPL' must hit the cache."""
        ticker_mock = MagicMock()
        ticker_mock.info = {"trailingAnnualDividendYield": 0.0050}
        with patch("yfinance.Ticker", return_value=ticker_mock) as ticker_patch:
            get_dividend_yield("aapl")
            get_dividend_yield("AAPL")
        ticker_patch.assert_called_once()
