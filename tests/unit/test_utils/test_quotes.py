"""Tests for the canonical safe-mid helpers (utils/quotes.py)."""
import numpy as np
import pandas as pd
import pytest

from utils.quotes import is_executable_quote, safe_mid, safe_mid_series


class TestScalarSafeMid:
    def test_valid_two_sided_quote(self):
        assert safe_mid(1.8, 2.2) == pytest.approx(2.0)
        assert is_executable_quote(1.8, 2.2) is True

    def test_zero_bid_is_not_executable(self):
        assert safe_mid(0.0, 2.0) is None          # nobody bidding -> not 1.0
        assert is_executable_quote(0.0, 2.0) is False

    def test_negative_and_zero_ask(self):
        assert safe_mid(-1.0, 2.0) is None
        assert safe_mid(1.0, 0.0) is None

    def test_crossed_quote(self):
        assert safe_mid(2.5, 2.0) is None
        assert is_executable_quote(2.5, 2.0) is False

    def test_missing_or_nonnumeric(self):
        assert safe_mid(None, 2.0) is None
        assert safe_mid(1.0, None) is None
        assert safe_mid("x", 2.0) is None
        assert safe_mid(float("nan"), 2.0) is None


class TestVectorSafeMid:
    def test_masks_invalid_rows_to_nan(self):
        bid = pd.Series([1.8, 0.0, 2.5, np.nan, 1.0])
        ask = pd.Series([2.2, 2.0, 2.0, 2.0, np.nan])
        out = safe_mid_series(bid, ask)
        assert out.iloc[0] == pytest.approx(2.0)   # valid
        assert pd.isna(out.iloc[1])                 # zero bid
        assert pd.isna(out.iloc[2])                 # crossed
        assert pd.isna(out.iloc[3])                 # nan bid
        assert pd.isna(out.iloc[4])                 # nan ask

    def test_all_nan_series_is_safe(self):
        out = safe_mid_series(pd.Series([np.nan, np.nan]), pd.Series([np.nan, np.nan]))
        assert out.isna().all()
        assert out.dtype.kind == "f"

    def test_non_numeric_coerced(self):
        out = safe_mid_series(pd.Series(["1.0", "bad"]), pd.Series(["2.0", "2.0"]))
        assert out.iloc[0] == pytest.approx(1.5)
        assert pd.isna(out.iloc[1])
