"""Unit tests for the IB Gateway options provider (no gateway, no network).

A fake ``ib_async.IB`` stands in for the gateway so the normalization to the
MarketDataClient schema, the greeks pass-through, the delayed/live provenance
rule and the unavailable-gateway degradation are exercised without a socket.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import fcntl
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from ib_async import Option, OptionChain, Stock

import services.ib_gateway_market_data_client as ib_module
from services import provider_reliability
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
                    # IB pads the root to six characters, as the real gateway does.
                    c.localSymbol = f"ACME  {c.lastTradeDateOrContractMonth[2:]}{c.right}{int(c.strike * 1000):08d}"
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


@pytest.fixture(autouse=True)
def _no_stand_back(monkeypatch):
    """Back-to-back chains pause half a second for other processes; the one test
    about that pause turns it back on, the rest should not pay for it."""
    monkeypatch.setattr(ib_module, "_LINE_LOCK_STAND_BACK_SECONDS", 0.0)


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
    # Compact OCC symbols (no IB padding), as the claim contract and ledger expect.
    assert chain["optionSymbol"].str.fullmatch(r"ACME260717[CP]\d{8}").all()
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


def test_frozen_feed_stamps_no_quote_time():
    # Frozen replays the last close while the tick arrives now: no stamp.
    chain = _client(_FakeIB(), market_data_type=2).get_option_chain("ACME", expiration="2026-07-17")
    assert "updated" not in chain.columns
    assert chain.attrs["market_data_type"] == "frozen"


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


# ── chain queue: waiting for a turn is not the gateway being slow ─────────────

class _SlowIB(_FakeIB):
    """A gateway whose option-contract qualification takes ``work_seconds``.

    ``slow_calls`` limits the slowness to the first N chains; the rest are instant.
    """

    def __init__(self, *, work_seconds: float, slow_calls: int = 10**6) -> None:
        super().__init__()
        self.work_seconds = work_seconds
        self.slow_calls = slow_calls
        self.chains_started = 0
        self.running = 0
        self.max_running = 0

    async def qualifyContractsAsync(self, *contracts):
        if contracts and isinstance(contracts[0], Option):
            self.chains_started += 1
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            try:
                if self.chains_started <= self.slow_calls:
                    await asyncio.sleep(self.work_seconds)
            finally:
                self.running -= 1
        return await super().qualifyContractsAsync(*contracts)


@pytest.fixture
def telemetry(monkeypatch):
    events = []
    monkeypatch.setattr(ib_module, "record_provider_telemetry", lambda **kwargs: events.append(kwargs))
    return events


def _chain_events(events):
    return [e for e in events if e.get("request_kind") == "request"]


def test_requests_queued_past_one_chain_budget_all_succeed(telemetry):
    """The regression: a screener asks for many chains at once, they run one at a
    time, and the ones at the back used to time out while still waiting in line.
    Five chains of 0.5 s each cannot all finish within one 1.5 s budget; they
    must all succeed because each gets the budget for its own work."""
    fake = _SlowIB(work_seconds=0.5)
    client = _client(fake, chain_timeout=1.5, chain_queue_timeout=30.0)
    with ThreadPoolExecutor(max_workers=5) as pool:
        chains = list(pool.map(lambda _: client.get_option_chain("ACME", expiration="2026-07-17"), range(5)))

    assert all(not chain.empty for chain in chains)
    assert fake.max_running == 1                      # still strictly one chain at a time
    events = _chain_events(telemetry)
    assert len(events) == 5 and all(e["success"] for e in events)
    # The last one waited longer than a whole chain budget and was still served.
    assert max(e["latency_ms"] for e in events) > 1500.0
    assert provider_reliability.get_breaker("ib_gateway").consecutive_failures == 0


def test_chain_slower_than_its_budget_fails_and_frees_the_queue(telemetry):
    fake = _SlowIB(work_seconds=3.0, slow_calls=1)
    client = _client(fake, chain_timeout=1.0, chain_queue_timeout=30.0)

    assert client.get_option_chain("ACME", expiration="2026-07-17").empty
    failed = _chain_events(telemetry)[-1]
    assert failed["success"] is False and failed["error_category"] == "timeout"
    assert failed["response_quality_note"] == "fetch_error"
    # A chain the gateway could not finish is gateway trouble: it counts.
    assert provider_reliability.get_breaker("ib_gateway").consecutive_failures == 1
    assert 900.0 < failed["latency_ms"] < 2500.0      # cut off at its budget, not at 3 s

    # The turn was handed on: the next chain runs straight away and succeeds.
    assert not client.get_option_chain("ACME", expiration="2026-07-17").empty
    assert _chain_events(telemetry)[-1]["success"] is True


def test_request_that_waits_past_the_queue_limit_is_not_a_gateway_failure(telemetry):
    fake = _SlowIB(work_seconds=2.0, slow_calls=1)
    client = _client(fake, chain_timeout=10.0, chain_queue_timeout=1.0)
    first_started = threading.Event()
    original = fake.qualifyContractsAsync

    async def _signalling(*contracts):
        if contracts and isinstance(contracts[0], Option):
            first_started.set()
        return await original(*contracts)

    fake.qualifyContractsAsync = _signalling
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.get_option_chain, "ACME", expiration="2026-07-17")
        assert first_started.wait(5.0)
        second = pool.submit(client.get_option_chain, "ACME", expiration="2026-07-17")
        assert not first.result().empty
        assert second.result().empty

    queued = [e for e in _chain_events(telemetry) if not e["success"]]
    assert len(queued) == 1
    assert queued[0]["response_quality_note"] == "queue_timeout"
    assert queued[0]["error_category"] == "timeout"
    # The queue was long; the gateway did nothing wrong. The breaker must not move.
    assert provider_reliability.get_breaker("ib_gateway").consecutive_failures == 0
    # The refused request never touched the gateway.
    assert fake.chains_started == 1


def test_five_queue_timeouts_do_not_open_the_circuit(telemetry):
    fake = _SlowIB(work_seconds=1.6, slow_calls=1)
    client = _client(fake, chain_timeout=10.0, chain_queue_timeout=1.0)
    with ThreadPoolExecutor(max_workers=7) as pool:
        chains = list(pool.map(lambda _: client.get_option_chain("ACME", expiration="2026-07-17"), range(7)))

    assert sum(1 for chain in chains if not chain.empty) == 1
    breaker = provider_reliability.get_breaker("ib_gateway")
    assert breaker.state() == "closed" and breaker.allow()


def test_success_telemetry_says_where_the_seconds_went(telemetry):
    client = _client(_FakeIB())
    assert not client.get_option_chain("ACME", expiration="2026-07-17").empty
    note = _chain_events(telemetry)[-1]["response_quality_note"]
    assert note.startswith("market_data_type=delayed; queued=0.0s; fetch=")
    # Five listed strikes, calls and puts, every one with a quote and greeks.
    assert "; ready=10/10; " in note and "; settle=0.0s" in note


def test_settle_reports_contracts_the_feed_never_fills():
    class _Ticker:
        def __init__(self, greeks):
            self.bid, self.ask, self.modelGreeks = 1.0, 1.2, greeks

    client = _client(_FakeIB())
    client.settle_seconds = 0.3
    stats = asyncio.run(client._settle([_Ticker(object()), _Ticker(object()), _Ticker(None)]))
    assert (stats["ready"], stats["total"]) == (2, 3)
    assert stats["seconds"] >= 0.3                    # waited the whole cap for the third
    assert stats["last_arrival"] < 0.1                # though nothing arrived after the start


def test_chain_budgets_are_read_from_env(monkeypatch):
    monkeypatch.setenv("IB_GATEWAY_CHAIN_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("IB_GATEWAY_CHAIN_QUEUE_SECONDS", "300")
    client = IBGatewayMarketDataClient(ib_factory=_FakeIB)
    assert (client.chain_timeout, client.chain_queue_timeout) == (45.0, 300.0)
    monkeypatch.delenv("IB_GATEWAY_CHAIN_TIMEOUT_SECONDS")
    monkeypatch.setenv("IB_GATEWAY_CHAIN_QUEUE_SECONDS", "soon")
    client = IBGatewayMarketDataClient(ib_factory=_FakeIB)
    assert (client.chain_timeout, client.chain_queue_timeout) == (120.0, 900.0)


# ── one gateway, several processes ────────────────────────────────────────────

class _BusyIdsIB(_FakeIB):
    """A gateway where some client ids are already held by other processes.

    Mirrors the real one: it announces error 326 at once, then leaves the caller
    waiting until its connect timeout.
    """

    def __init__(self, *, taken) -> None:
        super().__init__()
        self.taken = set(taken)

    async def connectAsync(self, host, port, clientId, timeout, readonly):
        if clientId in self.taken:
            self.connect_calls.append({"host": host, "port": port, "clientId": clientId, "readonly": readonly})
            self.errorEvent.emit(-1, 326, "Unable to connect as the client id is already in use.", None)
            await asyncio.sleep(timeout)
            raise TimeoutError()
        await super().connectAsync(host, port, clientId, timeout, readonly)

    def disconnect(self) -> None:
        self.connected = False


def _tried_ids(fake):
    return [call["clientId"] for call in fake.connect_calls]


def test_taken_client_id_moves_to_the_next_free_one_without_waiting_out_the_timeout():
    """The API server holds client id 17; a scheduled job reading the same
    setting used to be refused and fall back to another provider."""
    fake = _BusyIdsIB(taken={17, 18})
    client = _client(fake)
    started = time.monotonic()
    assert client.is_available() is True
    # Two refusals at a 2 s connect timeout each would be 4 s if waited out.
    assert time.monotonic() - started < 1.5
    assert _tried_ids(fake) == [17, 18, 19]
    assert client.connected_client_id == 19
    assert not client.get_option_chain("ACME", expiration="2026-07-17").empty


def test_free_client_id_is_used_as_configured():
    fake = _FakeIB()
    client = _client(fake)
    assert client.connected_client_id is None            # nothing dialled yet
    assert client.is_available() is True
    assert _tried_ids(fake) == [17] and client.connected_client_id == 17


def test_every_client_id_taken_is_reported_as_such(caplog):
    fake = _BusyIdsIB(taken=set(range(17, 17 + ib_module.MAX_CLIENT_ID_ATTEMPTS)))
    client = _client(fake)
    with caplog.at_level("WARNING"):
        assert client.is_available() is False
    assert _tried_ids(fake) == list(range(17, 25))
    assert "refused client ids 17-24" in caplog.text and "already in use" in caplog.text
    assert client.connected_client_id is None


def test_stopped_gateway_is_not_mistaken_for_a_taken_client_id():
    fake = _FakeIB(refuse=True)
    client = _client(fake)
    assert client.is_available() is False
    assert _tried_ids(fake) == [17]                       # no walk through 18, 19, ...


def test_reconnect_starts_again_from_the_configured_client_id():
    fake = _BusyIdsIB(taken={17})
    client = _client(fake)
    assert client.is_available() and client.connected_client_id == 18
    # The other process went away and the gateway restarted overnight.
    fake.taken.clear()
    fake.connected = False
    assert client.is_available() and client.connected_client_id == 17


def _hold_line_lock(client):
    """Take the chain turn the way another process would: its own open file."""
    client._line_lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(client._line_lock_path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_chain_waits_for_another_process_to_finish_its_chain(telemetry):
    fake = _FakeIB()
    client = _client(fake)
    other = _hold_line_lock(client)
    threading.Timer(0.7, other.close).start()             # the other process finishes

    assert not client.get_option_chain("ACME", expiration="2026-07-17").empty
    event = _chain_events(telemetry)[-1]
    assert event["success"] is True and event["latency_ms"] >= 600.0
    queued = float(event["response_quality_note"].split("queued=")[1].split("s")[0])
    assert queued >= 0.6                                  # counted as waiting, not as gateway time
    # And the turn is handed back: a second chain does not wait.
    assert not client.get_option_chain("ACME", expiration="2026-07-17").empty
    assert _chain_events(telemetry)[-1]["latency_ms"] < 500.0


def test_a_burst_of_chains_lets_a_waiting_process_in_between(telemetry, monkeypatch):
    """A screener burst is a dozen chains queued back to back. Another process
    polling for the turn must get it after the chain in progress, not after the
    whole burst."""
    monkeypatch.setattr(ib_module, "_LINE_LOCK_STAND_BACK_SECONDS", 2 * ib_module._LINE_LOCK_POLL_SECONDS)
    fake = _SlowIB(work_seconds=0.3)
    client = _client(fake)
    seen = {}

    def _other_process():
        handle = open(client._line_lock_path, "a+")
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(ib_module._LINE_LOCK_POLL_SECONDS)
        seen["chains_started_before_its_turn"] = fake.chains_started
        time.sleep(0.2)
        handle.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        chains = [pool.submit(client.get_option_chain, "ACME", expiration="2026-07-17") for _ in range(4)]
        while fake.chains_started == 0:
            time.sleep(0.01)
        other = threading.Thread(target=_other_process)
        other.start()
        assert all(not chain.result().empty for chain in chains)
        other.join(5.0)

    assert seen["chains_started_before_its_turn"] < 4


def test_other_process_holding_the_turn_too_long_is_a_queue_timeout(telemetry):
    fake = _FakeIB()
    client = _client(fake, chain_queue_timeout=1.0)
    other = _hold_line_lock(client)
    try:
        assert client.get_option_chain("ACME", expiration="2026-07-17").empty
    finally:
        other.close()
    event = _chain_events(telemetry)[-1]
    assert (event["response_quality_note"], event["error_category"]) == ("queue_timeout", "timeout")
    assert fake.requested == []                           # no market-data line was opened
    assert provider_reliability.get_breaker("ib_gateway").consecutive_failures == 0


def test_two_clients_on_one_gateway_share_one_turn():
    a = _client(_FakeIB())
    b = _client(_FakeIB())
    assert a._line_lock_path == b._line_lock_path
    assert a._line_lock_path.name == "ib_gateway_chain_127_0_0_1_4002.lock"
    live = IBGatewayMarketDataClient(ib_factory=_FakeIB, port=4001)
    assert live._line_lock_path != a._line_lock_path


def test_unusable_lock_file_does_not_stop_the_fetch(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("")
    client = _client(_FakeIB(), line_lock_path=blocker / "chain.lock")
    assert not client.get_option_chain("ACME", expiration="2026-07-17").empty
