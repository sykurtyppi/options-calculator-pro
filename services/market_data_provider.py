"""Options/earnings provider selection.

A single factory both injection points (the live web API's _get_mda_client and
the forward loop's _get_marketdata_client) call, so the active provider is
governed in one place by an env flag:

    OPTIONS_CALCULATOR_OPTIONS_PROVIDER = yfinance | marketdata_app

Default is **yfinance** (free) — the paid MarketData.app plan is deferred until
the forward collector proves the edge is worth the spend. Flip the flag to
"marketdata_app" (and keep MARKETDATA_TOKEN set) to switch back; no code change.

Both providers implement the same 5-method interface
(is_available / get_expirations / get_option_chain / get_earnings / get_quote),
so consumers are provider-agnostic.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

OPTIONS_PROVIDER_ENV = "OPTIONS_CALCULATOR_OPTIONS_PROVIDER"
DEFAULT_PROVIDER = "yfinance"

_YFINANCE_ALIASES = {"yfinance", "yahoo", "yf"}
_MARKETDATA_ALIASES = {"marketdata_app", "marketdata", "mdapp", "mda"}


def get_options_provider_name(environ: Optional[Mapping[str, str]] = None) -> str:
    """Return the configured provider name (lowercased), defaulting to yfinance."""
    env = environ if environ is not None else os.environ
    raw = str(env.get(OPTIONS_PROVIDER_ENV, "") or "").strip().lower()
    return raw or DEFAULT_PROVIDER


def build_market_data_client(*, provider: Optional[str] = None, **marketdata_kwargs: Any):
    """Construct the active options/earnings client.

    `provider` overrides the env flag (handy for tests). Unknown values fall back
    to the default provider with a warning rather than raising — a misconfigured
    flag should degrade to free data, not crash the pipeline.
    """
    name = (provider or get_options_provider_name()).lower()

    if name in _YFINANCE_ALIASES:
        from services.yfinance_market_data_client import YFinanceMarketDataClient
        return YFinanceMarketDataClient()

    if name in _MARKETDATA_ALIASES:
        from services.market_data_client import MarketDataClient
        return MarketDataClient(**marketdata_kwargs)

    logger.warning(
        "Unknown %s=%r; falling back to default provider %r.",
        OPTIONS_PROVIDER_ENV, name, DEFAULT_PROVIDER,
    )
    from services.yfinance_market_data_client import YFinanceMarketDataClient
    return YFinanceMarketDataClient()
