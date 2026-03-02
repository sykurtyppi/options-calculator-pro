"""
Shared pytest fixtures for Options Calculator Pro tests.
"""
import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ── Sample DataFrames ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_chain_df():
    """MDApp-normalized option chain DataFrame with calls and puts."""
    today = datetime.date.today()
    near_exp = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    far_exp  = (today + datetime.timedelta(days=42)).strftime("%Y-%m-%d")

    rows = []
    for exp, dte in [(near_exp, 7), (far_exp, 42)]:
        for side in ("call", "put"):
            for strike, delta_sign in [(490.0, 1), (500.0, 1), (510.0, 1)]:
                delta = 0.5 if strike == 500.0 else (0.65 if strike == 490.0 else 0.35)
                if side == "put":
                    delta = -delta
                rows.append({
                    "expiration_date": exp,
                    "strike": strike,
                    "side": side,
                    "impliedVolatility": 0.28 + 0.02 * (abs(strike - 500.0) / 10.0),
                    "delta": delta,
                    "lastPrice": 5.0,
                    "bid": 4.8,
                    "ask": 5.2,
                    "openInterest": 1200,
                    "volume": 400,
                    "dte": dte,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_earnings_df():
    """MDApp-normalized earnings DataFrame."""
    today = datetime.date.today()
    rows = [
        {
            "report_date": today + datetime.timedelta(days=8),
            "reportTime": "after market close",
            "fiscal_year": 2025,
            "fiscal_quarter": 1,
            "eps_estimate": 1.50,
            "eps_actual": None,
        },
        {
            "report_date": today - datetime.timedelta(days=84),
            "reportTime": "after market close",
            "fiscal_year": 2024,
            "fiscal_quarter": 4,
            "eps_estimate": 1.42,
            "eps_actual": 1.48,
        },
        {
            "report_date": today - datetime.timedelta(days=175),
            "reportTime": "before market open",
            "fiscal_year": 2024,
            "fiscal_quarter": 3,
            "eps_estimate": 1.35,
            "eps_actual": 1.41,
        },
    ]
    return pd.DataFrame(rows)


# ── Mock MDApp client ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_mda_client(sample_chain_df, sample_earnings_df):
    """Returns a MagicMock that mimics MarketDataClient's public interface."""
    client = MagicMock()
    client.is_available.return_value = True

    today = datetime.date.today()
    near_exp = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    far_exp  = (today + datetime.timedelta(days=42)).strftime("%Y-%m-%d")

    client.get_expirations.return_value = [near_exp, far_exp]
    client.get_option_chain.return_value = sample_chain_df
    client.get_earnings.return_value = sample_earnings_df
    client.get_quote.return_value = 500.0

    return client


@pytest.fixture
def unavailable_mda_client():
    """MDApp client stub with no token — simulates unconfigured state."""
    client = MagicMock()
    client.is_available.return_value = False
    return client
