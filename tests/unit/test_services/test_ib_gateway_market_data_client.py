"""Unit tests for the IB Gateway options provider (no gateway, no network).

A fake ``ib_async.IB`` stands in for the gateway so the normalization to the
MarketDataClient schema, the greeks pass-through, the delayed/live provenance
rule and the unavailable-gateway degradation are exercised without a socket.
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd
import pytest
from ib_async import Option, OptionChain, Stock

from services.ib_gateway_market_data_client import (
    _CHAIN_COLUMNS,
    IBGatewayMarketDataClient,
    _select_strikes,
)

_LISTED_STRIKES = {90.0, 95.0, 100.0, 105.0, 110.0, 115.0}
_UNLISTED_STRIKE = 105.0   # listed in secdef params but not qualifiable (delisted contract)
_QUOTE_TIME = _dt.datetime(2026, 7, 1, 14, 30, 5, tzinfo=_dt.timezone.utc)


class _Event:
    def __init__(self) -> None:
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def emit(self, *args):
        for h in self.handlers:
            h(*args)


class _Greeks:
    def __init__(self, strike: float, right: str) -> None:
        self.impliedVol = 0.30 + strike / 10_000.0
        self.delta = 0.5 if right == "C" else -0.5
        self.gamma = 0.02
        self.theta = -0.05
        self.vega = 0.11
        self.undPrice = 100.0


class _OptionTicker:
    def __init__(self, contract: Option) -> None:
        self.contract = contract
        # 115 strike: a zero bid (far OTM) — mid must be NaN, not (0 + ask) / 2.
        self.bid = 0.0 if contract.strike == 115.0 else 2.0
        self.ask = 2.4
        self.last = 2.2
        self.volume = 120
        self.callOpenInterest = 800 if contract.right == "C" else math.nan
        self.putOpenInterest = 700 if contract.right == "P" else math.nan
        self.modelGreeks = _Greeks(contract.strike, contract.right)
        self.time = _QUOTE_TIME


class _StockTicker:
    def __init__(self) -> None:
        self.last = 100.0
        self.close = 99.5

    def marketPrice(self) -> float:
        return 100.0


class _FakeIB:
    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.connected = False
        self.connect_calls = []
        self.market_data_type = None
        self.requested = []
        self.stock_requests = []
        self.cancelled = []
        self.errorEvent = _Event()

    def isConnected(self) -> bool:
        return self.connected

    async def connectAsync(self, host, port, clientId, timeout, readonly):
        self.connect_calls.append({"host": host, "port": port, "clientId": clientId, "readonly": readonly})
        if self.refuse:
            raise ConnectionRefusedError("connection refused")
        self.connected = True

    def reqMarketDataType(self, kind: int) -> None:
        self.market_data_type = kind

    async def qualifyContractsAsync(self, *contracts):
        out = []
        for c in contracts:
            if isinstance(c, Stock):
                if c.symbol == "ACME":
                    c.conId = 1
                    out.append(c)
            elif isinstance(c, Option):
                if c.strike in _LISTED_STRIKES and c.strike != _UNLISTED_STRIKE:
                    c.conId = int(c.strike * 10) + (1 if c.right == "C" else 2)
                    c.localSymbol = f"ACME {c.lastTradeDateOrContractMonth}{c.right}{int(c.strike):05d}000"
                    out.append(c)
        return out

    async def reqSecDefOptParamsAsync(self, symbol, fut_fop_exchange, sec_type, con_id):
        return [
            OptionChain("CBOE", con_id, "ACME", "100", {"20260717"}, {100.0}),
            OptionChain("SMART", con_id, "ACME", "100", {"20260821", "20260717"}, set(_LISTED_STRIKES)),
        ]

    async def reqTickersAsync(self, *contracts):
        return [_StockTicker() for _ in contracts]

    def reqMktData(self, contract, generic_ticks, snapshot, regulatory_snapshot):
        if isinstance(contract, Stock):
            self.stock_requests.append(contract)
            return _StockTicker()
        self.requested.append((contract, generic_ticks, snapshot))
        return _OptionTicker(contract)

    def cancelMktData(self, contract):
        self.cancelled.append(contract)


def _client(fake: _FakeIB, **kwargs) -> IBGatewayMarketDataClient:
    kwargs.setdefault("market_data_type", 3)
    return IBGatewayMarketDataClient(ib_factory=lambda: fake, settle_seconds=0.0, timeout=2.0,
                                     port=4002, client_id=17, **kwargs)


def test_chain_is_normalized_with_greeks_from_ib():
    fake = _FakeIB()
    client = _client(fake)

    chain = client.get_option_chain("acme", expiration="2026-07-17", strike_limit=4)

    # Delayed feed: no `updated` quote time is invented.
    assert list(chain.columns) == _CHAIN_COLUMNS
    # 4 strikes nearest spot 100: 100, 95, 105 and (lower strike wins the 90/110 tie) 90;
    # 105 is not qualifiable → 3 strikes × 2 sides.
    assert sorted(chain["strike"].unique().tolist()) == [90.0, 95.0, 100.0]
    assert len(chain) == 6
    assert set(chain["side"]) == {"call", "put"}
    assert (chain["expiration_date"] == "2026-07-17").all()
    assert chain["delta"].notna().all() and chain["vega"].notna().all()
    assert (chain.loc[chain["side"] == "call", "delta"] > 0).all()
    assert (chain.loc[chain["side"] == "put", "delta"] < 0).all()
    assert chain["impliedVolatility"].between(0.29, 0.32).all()
    assert (chain["underlyingPrice"] == 100.0).all()
    assert chain["optionSymbol"].str.startswith("ACME ").all()
    call_100 = chain[(chain["side"] == "call") & (chain["strike"] == 100.0)].iloc[0]
    assert call_100["openInterest"] == 800
    assert call_100["mid"] == pytest.approx(2.2)
    assert call_100["inTheMoney"] is False
    assert chain["dte"].notna().all()
    assert chain.attrs["market_data_type"] == "delayed"
    assert chain.attrs["provider"] == "ib_gateway"
    assert "surface_quality" in chain.attrs

    # Connection posture and housekeeping.
    assert fake.connect_calls == [{"host": "127.0.0.1", "port": 4002, "clientId": 17, "readonly": True}]
    assert fake.market_data_type == 3
    assert all(ticks == "100,101,106" and snapshot is False for _, ticks, snapshot in fake.requested)
    # Every option line and the streamed underlying are cancelled again.
    assert len(fake.requested) == 6 and len(fake.stock_requests) == 1
    assert len(fake.cancelled) == len(fake.requested) + len(fake.stock_requests)


def test_zero_bid_gives_nan_mid_not_half_ask():
    chain = _client(_FakeIB()).get_option_chain("ACME", expiration="2026-07-17", strike_limit=6)
    far_otm = chain[chain["strike"] == 115.0]
    assert not far_otm.empty
    assert far_otm["bid"].eq(0.0).all()
    assert far_otm["mid"].isna().all()
    assert far_otm["extrinsicValue"].isna().all()


def test_live_feed_stamps_provider_quote_time():
    chain = _client(_FakeIB(), market_data_type=1).get_option_chain("ACME", expiration="2026-07-17")
    assert "updated" in chain.columns
    assert (chain["updated"] == _QUOTE_TIME.timestamp()).all()
    assert chain.attrs["market_data_type"] == "live"


def test_default_expiration_is_the_front_month_and_all_is_capped():
    fake = _FakeIB()
    client = _client(fake)
    front = client.get_option_chain("ACME", strike_limit=2)
    assert set(front["expiration_date"]) == {"2026-07-17"}
    both = client.get_option_chain("ACME", expiration="all", strike_limit=2)
    assert set(both["expiration_date"]) == {"2026-07-17", "2026-08-21"}


def test_unlisted_expiration_and_historical_date_return_empty():
    client = _client(_FakeIB())
    assert client.get_option_chain("ACME", expiration="2030-01-01").empty
    assert client.get_option_chain("ACME", date="2026-01-02").empty


def test_side_filter_requests_only_that_right():
    fake = _FakeIB()
    chain = _client(fake).get_option_chain("ACME", expiration="2026-07-17", side="put", strike_limit=3)
    assert set(chain["side"]) == {"put"}
    assert all(c.right == "P" for c, _, _ in fake.requested)


def test_expirations_and_quote():
    client = _client(_FakeIB())
    assert client.get_expirations("acme") == ["2026-07-17", "2026-08-21"]
    assert client.get_quote("acme") == 100.0


def test_unknown_symbol_returns_empty_not_error():
    client = _client(_FakeIB())
    assert client.get_option_chain("NOPE", expiration="2026-07-17").empty
    assert client.get_expirations("NOPE") == []
    assert client.get_quote("NOPE") is None


def test_gateway_down_degrades_and_backs_off():
    fake = _FakeIB(refuse=True)
    client = _client(fake)
    assert client.is_available() is False
    assert client.get_option_chain("ACME", expiration="2026-07-17").empty
    assert client.get_quote("ACME") is None
    # The cooldown stops is_available() from re-dialling a stopped gateway on every call.
    attempts = len(fake.connect_calls)
    assert client.is_available() is False
    assert len(fake.connect_calls) == attempts
    assert attempts >= 1


def test_available_when_gateway_accepts():
    fake = _FakeIB()
    client = _client(fake)
    assert client.is_available() is True
    assert fake.connected is True


def test_earnings_delegate_to_yfinance_and_are_labelled_so():
    class _Earnings:
        provider_name = "yfinance"

        def get_earnings(self, symbol, countback=24):
            return pd.DataFrame({"symbol": [symbol], "report_date": [_dt.date(2026, 7, 30)], "countback": [countback]})

    client = _client(_FakeIB(), earnings_client=_Earnings())
    df = client.get_earnings("ACME", countback=8)
    assert df.iloc[0]["symbol"] == "ACME" and df.iloc[0]["countback"] == 8
    assert client.earnings_provider_name == "yfinance"
    assert client.provider_name == "ib_gateway"


def test_env_settings_are_read(monkeypatch):
    monkeypatch.setenv("IB_GATEWAY_HOST", "10.0.0.5")
    monkeypatch.setenv("IB_GATEWAY_PORT", "7497")
    monkeypatch.setenv("IB_GATEWAY_CLIENT_ID", "3")
    monkeypatch.setenv("IB_GATEWAY_MARKET_DATA_TYPE", "1")
    monkeypatch.setenv("IB_GATEWAY_TIMEOUT_SECONDS", "2.5")
    client = IBGatewayMarketDataClient(ib_factory=_FakeIB)
    assert (client.host, client.port, client.client_id) == ("10.0.0.5", 7497, 3)
    assert client.market_data_type == 1 and client.market_data_type_label == "live"
    assert client.timeout == 2.5


def test_bad_market_data_type_falls_back_to_delayed(monkeypatch):
    monkeypatch.setenv("IB_GATEWAY_MARKET_DATA_TYPE", "9")
    assert IBGatewayMarketDataClient(ib_factory=_FakeIB).market_data_type == 3


def test_select_strikes_nearest_spot():
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    assert _select_strikes(strikes, 101.0, 3, None) == [95.0, 100.0, 105.0]
    assert _select_strikes(strikes, 101.0, 0, None) == strikes
    # No spot: the middle of the listed range, never an arbitrary end.
    assert _select_strikes(strikes, None, 3, None) == [95.0, 100.0, 105.0]
    assert np.isnan(float("nan"))  # sanity: numpy imported for callers above
