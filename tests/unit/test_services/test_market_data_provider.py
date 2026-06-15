"""Provider-flag factory tests."""
from __future__ import annotations

from services.market_data_provider import (
    DEFAULT_PROVIDER,
    build_market_data_client,
    get_options_provider_name,
)
from services.yfinance_market_data_client import YFinanceMarketDataClient


def test_default_provider_is_yfinance():
    assert DEFAULT_PROVIDER == "yfinance"
    assert get_options_provider_name({}) == "yfinance"


def test_env_flag_selects_provider():
    assert get_options_provider_name({"OPTIONS_CALCULATOR_OPTIONS_PROVIDER": "marketdata_app"}) == "marketdata_app"
    assert get_options_provider_name({"OPTIONS_CALCULATOR_OPTIONS_PROVIDER": "  YFinance "}) == "yfinance"
    assert get_options_provider_name({"OPTIONS_CALCULATOR_OPTIONS_PROVIDER": ""}) == "yfinance"


def test_build_defaults_to_yfinance_client():
    client = build_market_data_client(provider="yfinance")
    assert isinstance(client, YFinanceMarketDataClient)
    assert client.is_available() is True


def test_build_marketdata_app_returns_marketdata_client():
    from services.market_data_client import MarketDataClient
    client = build_market_data_client(provider="marketdata_app")
    assert isinstance(client, MarketDataClient)


def test_unknown_provider_falls_back_to_default_not_raise():
    # A misconfigured flag must degrade to free data, not crash the pipeline.
    client = build_market_data_client(provider="totally_bogus")
    assert isinstance(client, YFinanceMarketDataClient)


def test_both_providers_share_the_interface():
    from services.market_data_client import MarketDataClient
    methods = ("is_available", "get_expirations", "get_option_chain", "get_earnings", "get_quote")
    yf_client = build_market_data_client(provider="yfinance")
    md_client = build_market_data_client(provider="marketdata_app")
    for m in methods:
        assert callable(getattr(yf_client, m)), m
        assert callable(getattr(md_client, m)), m
