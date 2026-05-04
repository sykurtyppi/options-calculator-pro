"""
Tests for web.api.screener_engine.

These tests cover the yfinance-side chain normalization. The MDApp path is
already covered by tests/unit/test_services/test_market_data_client.py.
"""
import unittest

import numpy as np
import pandas as pd

from web.api.screener_engine import _normalize_yfinance_chain


class TestNormalizeYfinanceChain(unittest.TestCase):
    """Regression coverage for the H-05 fix: `expiration_date` must hold the
    queried expiry, not the per-row `lastTradeDate` timestamp."""

    def _make_yfinance_like_frame(self) -> pd.DataFrame:
        """Frame shaped like yfinance ticker.option_chain(expiry).calls."""
        return pd.DataFrame(
            {
                "contractSymbol": ["TEST250516C00100000", "TEST250516C00105000"],
                "strike": [100.0, 105.0],
                "lastPrice": [4.20, 1.30],
                "bid": [4.10, 1.25],
                "ask": [4.30, 1.35],
                # lastTradeDate is intentionally NOT the expiry — these are
                # arbitrary stale timestamps the previous implementation copied
                # into expiration_date by mistake.
                "lastTradeDate": [
                    pd.Timestamp("2025-04-30 17:59:00"),
                    pd.Timestamp("2025-04-29 14:12:00"),
                ],
                "openInterest": [1200, 800],
                "volume": [400, 200],
                "impliedVolatility": [0.31, 0.34],
            }
        )

    def test_writes_actual_expiry_into_expiration_date(self):
        """H-05: expiration_date must equal the queried expiry, not lastTradeDate."""
        frame = self._make_yfinance_like_frame()
        out = _normalize_yfinance_chain(frame, "call", expiry="2025-05-16")

        self.assertFalse(out.empty)
        # All rows must show the queried expiry
        self.assertTrue((out["expiration_date"] == "2025-05-16").all())
        # The per-row lastTradeDate must NOT have leaked into expiration_date
        self.assertFalse((out["expiration_date"] == out["lastTradeDate"].astype(str)).any())

    def test_expiration_date_does_not_carry_intraday_timestamp(self):
        """The queried expiry is a calendar date; output must not include time-of-day."""
        frame = self._make_yfinance_like_frame()
        out = _normalize_yfinance_chain(frame, "call", expiry="2025-05-16")

        for value in out["expiration_date"]:
            # Must be a YYYY-MM-DD string, not a tz-aware Timestamp string
            self.assertEqual(len(str(value)), 10)
            self.assertEqual(str(value)[4], "-")
            self.assertEqual(str(value)[7], "-")

    def test_preserves_other_columns(self):
        """Regression: only `side`, `optionSymbol`, `expiration_date`,
        and `openInterest` (when missing) should be set/added; existing pricing
        and quote columns must pass through unchanged."""
        frame = self._make_yfinance_like_frame()
        out = _normalize_yfinance_chain(frame, "call", expiry="2025-05-16")

        for col in ("strike", "bid", "ask", "openInterest", "volume", "lastPrice"):
            self.assertTrue((out[col] == frame[col]).all(), f"{col} altered")

    def test_side_is_set_for_call(self):
        out = _normalize_yfinance_chain(self._make_yfinance_like_frame(), "call", expiry="2025-05-16")
        self.assertTrue((out["side"] == "call").all())

    def test_side_is_set_for_put(self):
        out = _normalize_yfinance_chain(self._make_yfinance_like_frame(), "put", expiry="2025-05-16")
        self.assertTrue((out["side"] == "put").all())

    def test_open_interest_filled_when_missing(self):
        """When the upstream frame lacks openInterest, it must be filled with NaN
        so downstream `_safe_int` returns None rather than KeyError."""
        frame = self._make_yfinance_like_frame().drop(columns=["openInterest"])
        out = _normalize_yfinance_chain(frame, "call", expiry="2025-05-16")
        self.assertIn("openInterest", out.columns)
        self.assertTrue(out["openInterest"].isna().all())

    def test_empty_frame_returns_empty(self):
        """Early return on None / empty frame must not raise."""
        self.assertTrue(_normalize_yfinance_chain(None, "call", expiry="2025-05-16").empty)
        self.assertTrue(
            _normalize_yfinance_chain(pd.DataFrame(), "put", expiry="2025-05-16").empty
        )


if __name__ == "__main__":
    unittest.main()
