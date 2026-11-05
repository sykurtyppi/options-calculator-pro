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


def test_ib_gateway_aliases_build_the_ib_client():
    from services.ib_gateway_market_data_client import IBGatewayMarketDataClient
    for alias in ("ib_gateway", "ib", "ibkr", "interactive_brokers", "tws"):
        client = build_market_data_client(provider=alias)
        assert isinstance(client, IBGatewayMarketDataClient), alias
        assert client.provider_name == "ib_gateway"
        for m in ("is_available", "get_expirations", "get_option_chain", "get_earnings", "get_quote"):
            assert callable(getattr(client, m)), m


def test_provider_io_category_matches_the_client_built():
    from services.external_io_gate import Category
    from services.market_data_provider import provider_io_category
    assert provider_io_category("yfinance") is Category.YFINANCE
    assert provider_io_category("marketdata_app") is Category.MARKETDATA
    assert provider_io_category("mda") is Category.MARKETDATA
    assert provider_io_category("ib_gateway") is Category.IB_GATEWAY
    assert provider_io_category("IBKR") is Category.IB_GATEWAY
    # Unknown names fall back to yfinance, like build_market_data_client does.
    assert provider_io_category("totally_bogus") is Category.YFINANCE
    assert provider_io_category(None) is Category.YFINANCE
