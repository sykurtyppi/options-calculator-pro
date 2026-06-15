"""Unit tests for the yfinance options/earnings provider (no network).

A fake ticker factory feeds crafted yfinance-shaped data so we exercise the
normalization to the MarketDataClient schema without hitting Yahoo.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from services.yfinance_market_data_client import YFinanceMarketDataClient


class _FakeChain:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame) -> None:
        self.calls = calls
        self.puts = puts


class _FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.options = ("2026-07-17", "2026-08-21")
        self.fast_info = {"last_price": 100.0}
        self.info = {}  # no earningsTimestamp → upcoming reportTime None

    def option_chain(self, exp: str) -> _FakeChain:
        def _leg(strike, bid, ask, iv, oi, vol, itm):
            return {
                "contractSymbol": f"{self.symbol}{exp}C{strike}",
                "strike": strike, "bid": bid, "ask": ask,
                "impliedVolatility": iv, "openInterest": oi, "volume": vol,
                "lastPrice": (bid + ask) / 2, "inTheMoney": itm,
            }
        calls = pd.DataFrame([
            _leg(95, 6.0, 6.4, 0.30, 500, 100, True),
            _leg(100, 3.0, 3.4, 0.28, 800, 250, False),
            _leg(105, 1.2, 1.5, 0.32, 300, 80, False),
        ])
        puts = pd.DataFrame([
            _leg(95, 1.1, 1.4, 0.31, 400, 90, False),
            _leg(100, 2.9, 3.3, 0.29, 700, 220, False),
            _leg(105, 5.8, 6.2, 0.33, 250, 60, True),
        ])
        return _FakeChain(calls, puts)

    def get_earnings_dates(self, limit: int = 24) -> pd.DataFrame:
        idx = pd.to_datetime(["2026-07-30", "2026-04-30", "2026-01-29"])
        return pd.DataFrame(
            {"EPS Estimate": [1.5, 1.4, 1.3], "Reported EPS": [np.nan, 1.48, 1.41],
             "Surprise(%)": [np.nan, 5.7, 4.6]},
            index=idx,
        )


def _client() -> YFinanceMarketDataClient:
    return YFinanceMarketDataClient(ticker_factory=_FakeTicker)


def test_is_available_true_without_token():
    assert _client().is_available() is True


def test_get_expirations_returns_iso_strings():
    assert _client().get_expirations("aapl") == ["2026-07-17", "2026-08-21"]


def test_get_quote_returns_last_price():
    assert _client().get_quote("AAPL") == 100.0


def test_chain_normalized_to_expected_schema():
    df = _client().get_option_chain("AAPL", expiration="2026-07-17")
    # all MarketDataClient-schema columns present
    for col in ("strike", "side", "bid", "ask", "mid", "impliedVolatility",
                "openInterest", "volume", "dte", "expiration_date",
                "delta", "gamma", "theta", "vega", "underlyingPrice"):
        assert col in df.columns, col
    assert set(df["side"]) == {"call", "put"}
    assert (df["expiration_date"] == "2026-07-17").all()
    # mid is computed from bid/ask
    row = df[(df["side"] == "call") & (df["strike"] == 100)].iloc[0]
    assert row["mid"] == pytest.approx((3.0 + 3.4) / 2)


def test_chain_greeks_are_nan_not_zero():
    # Honesty: yfinance has no greeks. They must be NaN, never silently 0.
    df = _client().get_option_chain("AAPL", expiration="2026-07-17")
    for greek in ("delta", "gamma", "theta", "vega"):
        assert df[greek].isna().all(), greek


def test_chain_attaches_surface_quality_for_m5_gate():
    df = _client().get_option_chain("AAPL", expiration="2026-07-17")
    assert "surface_quality" in df.attrs
    assert "status" in df.attrs["surface_quality"]


def test_side_filter():
    df = _client().get_option_chain("AAPL", expiration="2026-07-17", side="call")
    assert set(df["side"]) == {"call"}


def test_range_filter_otm():
    df = _client().get_option_chain("AAPL", expiration="2026-07-17", range_filter="otm")
    # OTM calls have strike > spot(100); OTM puts have strike < spot
    for _, r in df.iterrows():
        if r["side"] == "call":
            assert r["strike"] > 100
        else:
            assert r["strike"] < 100


def test_historical_date_returns_empty():
    # yfinance cannot serve historical chains — empty, not a substituted today-chain.
    df = _client().get_option_chain("AAPL", expiration="2026-07-17", date="2025-01-02")
    assert df.empty


def test_unlisted_expiration_returns_empty():
    df = _client().get_option_chain("AAPL", expiration="1999-01-01")
    assert df.empty


def test_get_earnings_schema_and_timing():
    df = _client().get_earnings("AAPL", countback=12)
    for col in ("symbol", "report_date", "event_date", "reportTime",
                "reportedEPS", "estimatedEPS", "surpriseEPS"):
        assert col in df.columns, col
    # sorted newest first
    dates = [d for d in df["report_date"] if d is not None]
    assert dates == sorted(dates, reverse=True)
    # historical rows (no upcoming timestamp here) carry reportTime None, never a false BMO/AMC
    assert df["reportTime"].isna().all() or (df["reportTime"] == None).all()  # noqa: E711
    # surpriseEPS computed where both EPS present
    past = df[df["reportedEPS"].notna() & df["estimatedEPS"].notna()].iloc[0]
    assert past["surpriseEPS"] == pytest.approx(past["reportedEPS"] - past["estimatedEPS"])
