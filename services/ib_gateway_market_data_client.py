"""Interactive Brokers Gateway options provider — a drop-in for MarketDataClient.

Implements the same 5-method interface (is_available / get_expirations /
get_option_chain / get_earnings / get_quote) on top of a locally running IB
Gateway (or TWS) reached over its socket API via ``ib_async``. Selected via
``OPTIONS_CALCULATOR_OPTIONS_PROVIDER=ib_gateway`` (see build_market_data_client).

What IB gives that yfinance cannot:
  - Model greeks (delta/gamma/theta/vega) and model implied volatility per
    contract, straight from IB's option computation ticks.
  - Exchange bid/ask with a quote time, so quote provenance can carry a real
    provider timestamp when the feed is live.

Data-fidelity caveats, stated rather than hidden:
  - No earnings calendar. IB's socket API has no free earnings-date feed, so
    ``get_earnings`` delegates to the yfinance provider and the rows are labelled
    as such (``earnings_provider_name``), never as IB data.
  - No historical chains. ``get_option_chain(date=...)`` returns an empty frame.
  - Without a live OPRA subscription the gateway serves DELAYED quotes
    (``IB_GATEWAY_MARKET_DATA_TYPE=3``, the default). Delayed rows carry no
    ``updated`` quote time — a 15-minute-old quote stamped "now" would be an
    invented timestamp — and the frame is tagged
    ``attrs["market_data_type"] = "delayed"`` so consumers can see it.
  - Market-data lines are capped per account (100 by default), so chains are
    requested in batches and a wide ``expiration="all"`` fetch is slow, not free.

Connection settings (all optional):
  IB_GATEWAY_HOST                 default 127.0.0.1
  IB_GATEWAY_PORT                 default 4002 (paper gateway; live gateway 4001,
                                  TWS paper 7497, TWS live 7496)
  IB_GATEWAY_CLIENT_ID            default 17
  IB_GATEWAY_MARKET_DATA_TYPE     1 live | 2 frozen | 3 delayed | 4 delayed-frozen
  IB_GATEWAY_TIMEOUT_SECONDS      socket connect timeout, default 10
  IB_GATEWAY_QUOTE_SETTLE_SECONDS max wait for streamed quotes to arrive, default 15
                                  (delayed feeds need ~10 s; the wait ends early once
                                  every contract has a quote and greeks)

The gateway must be running and logged in; the client never enters credentials.
It connects read-only and never places, modifies or cancels orders.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from utils.quotes import safe_mid_series

from services import provider_reliability as reliability
from services.market_calendar import in_regular_session
from services.option_surface_quality import diagnose_option_surface_quality
from services.provider_telemetry import classify_error, record_provider_telemetry

logger = logging.getLogger(__name__)

HOST_ENV = "IB_GATEWAY_HOST"
PORT_ENV = "IB_GATEWAY_PORT"
CLIENT_ID_ENV = "IB_GATEWAY_CLIENT_ID"
MARKET_DATA_TYPE_ENV = "IB_GATEWAY_MARKET_DATA_TYPE"
TIMEOUT_ENV = "IB_GATEWAY_TIMEOUT_SECONDS"
SETTLE_ENV = "IB_GATEWAY_QUOTE_SETTLE_SECONDS"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_CLIENT_ID = 17
DEFAULT_MARKET_DATA_TYPE = 3
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_SETTLE_SECONDS = 15.0

MARKET_DATA_TYPE_LABELS = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed_frozen"}
# Only a live or frozen feed stamps quotes the gateway itself timed; a delayed
# feed's ticks arrive ~15 minutes after the fact with no original time.
_REAL_TIME_MARKET_DATA_TYPES = {1, 2}

# IB's default market-data line cap is 100 concurrent subscriptions per account;
# stay under it with headroom for anything else the user runs against the gateway.
MAX_CONCURRENT_LINES = 90
# Option volume (100), open interest (101) and model IV/greeks (106).
_OPTION_GENERIC_TICKS = "100,101,106"
# How long a failed connection attempt keeps is_available() False before retrying.
UNAVAILABLE_COOLDOWN_SECONDS = 60.0
MAX_EXPIRATIONS_FOR_ALL = 12

# Normalized chain columns expected by edge_engine / screener / institutional_ml_db
# (mirrors MarketDataClient._chain_response_to_df). ``updated`` (unix seconds) is
# appended only when the feed is real-time; see quote_provenance.
_CHAIN_COLUMNS = [
    "optionSymbol", "underlying", "side", "strike", "bid", "ask", "mid",
    "lastPrice", "volume", "openInterest", "impliedVolatility",
    "delta", "gamma", "theta", "vega", "dte", "underlyingPrice",
    "inTheMoney", "intrinsicValue", "extrinsicValue", "expiration_date",
]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(str(raw).strip()) if str(raw).strip() else default
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(str(raw).strip()) if str(raw).strip() else default
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %s", name, raw, default)
        return default


def _default_ib_factory():
    from ib_async import IB
    return IB()


class IBGatewayConnectionError(RuntimeError):
    """The gateway could not be reached or refused the API connection."""


class _IBSession:
    """Owns one asyncio loop on a daemon thread and the IB connection living on it.

    ib_async is asyncio-based; FastAPI handlers and the batch scripts are
    synchronous and may already sit inside (or outside) an event loop. Running
    every IB coroutine on a private loop thread via run_coroutine_threadsafe
    makes the client safe to call from either, and serialises access to the one
    socket connection.
    """

    def __init__(self, *, ib_factory: Callable[[], Any], host: str, port: int, client_id: int,
                 market_data_type: int, timeout: float) -> None:
        self._ib_factory = ib_factory
        self._host = host
        self._port = port
        self._client_id = client_id
        self._market_data_type = market_data_type
        self._timeout = timeout
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ib: Any = None
        self._start_lock = threading.Lock()
        self._chain_lock: Optional[asyncio.Lock] = None
        self.recent_errors: List[Dict[str, Any]] = []

    # ── loop plumbing ─────────────────────────────────────────────────────────

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._start_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=loop.run_forever, name="ib-gateway-loop", daemon=True)
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def run(self, coro_fn: Callable[[Any], Any], *, timeout: float) -> Any:
        """Run ``await coro_fn(ib)`` on the session loop with a connected ``ib``."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._call(coro_fn), loop)
        try:
            return future.result(timeout=timeout)
        except Exception:
            future.cancel()
            raise

    async def _call(self, coro_fn: Callable[[Any], Any]) -> Any:
        ib = await self._connected()
        return await coro_fn(ib)

    async def _connected(self) -> Any:
        if self._ib is None:
            # Construct on the loop thread: ib_async binds to the running loop.
            self._ib = self._ib_factory()
            error_event = getattr(self._ib, "errorEvent", None)
            if error_event is not None:
                error_event += self._on_error
        if not self._ib.isConnected():
            try:
                await self._ib.connectAsync(
                    self._host, self._port, clientId=self._client_id,
                    timeout=self._timeout, readonly=True,
                )
            except Exception as exc:
                raise IBGatewayConnectionError(
                    f"IB Gateway not reachable at {self._host}:{self._port} "
                    f"(clientId={self._client_id}): {type(exc).__name__}: {exc}"
                ) from exc
            self._ib.reqMarketDataType(self._market_data_type)
        return self._ib

    def _on_error(self, reqId: Any, errorCode: Any, errorString: Any, contract: Any = None, *rest: Any) -> None:
        # Keep a short ring of the gateway's error messages so an all-NaN chain
        # can be explained (e.g. 354 = market data not subscribed).
        self.recent_errors.append({
            "reqId": reqId, "code": errorCode, "message": str(errorString),
            "contract": getattr(contract, "localSymbol", None) or getattr(contract, "symbol", None),
            "at": time.time(),
        })
        del self.recent_errors[:-50]

    def chain_lock(self) -> asyncio.Lock:
        if self._chain_lock is None:
            self._chain_lock = asyncio.Lock()
        return self._chain_lock

    def close(self) -> None:
        loop, ib = self._loop, self._ib
        if loop is None:
            return
        if ib is not None and ib.isConnected():
            asyncio.run_coroutine_threadsafe(self._disconnect(), loop).result(timeout=self._timeout)
        loop.call_soon_threadsafe(loop.stop)

    async def _disconnect(self) -> None:
        self._ib.disconnect()


class IBGatewayMarketDataClient:
    """IB Gateway-backed implementation of the MarketDataClient interface."""

    provider_name = "ib_gateway"
    # Earnings rows come from the yfinance provider, not from IB (see module doc).
    earnings_provider_name = "yfinance"

    def __init__(
        self,
        *,
        ib_factory: Optional[Callable[[], Any]] = None,
        earnings_client: Any = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        market_data_type: Optional[int] = None,
        timeout: Optional[float] = None,
        settle_seconds: Optional[float] = None,
    ) -> None:
        self.host = host or os.environ.get(HOST_ENV, "").strip() or DEFAULT_HOST
        self.port = port if port is not None else _env_int(PORT_ENV, DEFAULT_PORT)
        self.client_id = client_id if client_id is not None else _env_int(CLIENT_ID_ENV, DEFAULT_CLIENT_ID)
        mdt = market_data_type if market_data_type is not None else _env_int(MARKET_DATA_TYPE_ENV, DEFAULT_MARKET_DATA_TYPE)
        if mdt not in MARKET_DATA_TYPE_LABELS:
            logger.warning("Unknown %s=%r; using delayed (3)", MARKET_DATA_TYPE_ENV, mdt)
            mdt = DEFAULT_MARKET_DATA_TYPE
        self.market_data_type = mdt
        self.timeout = timeout if timeout is not None else _env_float(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS)
        self.settle_seconds = settle_seconds if settle_seconds is not None else _env_float(SETTLE_ENV, DEFAULT_SETTLE_SECONDS)
        self._session = _IBSession(
            ib_factory=ib_factory or _default_ib_factory,
            host=self.host, port=self.port, client_id=self.client_id,
            market_data_type=self.market_data_type, timeout=self.timeout,
        )
        self._earnings_client = earnings_client
        self._unavailable_until = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def market_data_type_label(self) -> str:
        return MARKET_DATA_TYPE_LABELS[self.market_data_type]

    def _earnings(self) -> Any:
        if self._earnings_client is None:
            from services.yfinance_market_data_client import YFinanceMarketDataClient
            self._earnings_client = YFinanceMarketDataClient()
        return self._earnings_client

    def _run(self, coro_fn: Callable[[Any], Any], *, timeout: Optional[float] = None) -> Any:
        return self._session.run(coro_fn, timeout=timeout if timeout is not None else self.timeout + 5.0)

    @property
    def recent_gateway_errors(self) -> List[Dict[str, Any]]:
        return list(self._session.recent_errors)

    def close(self) -> None:
        try:
            self._session.close()
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            logger.debug("IB Gateway close failed: %s", exc)

    # ── availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when the gateway accepts an API connection.

        A failed attempt is remembered for UNAVAILABLE_COOLDOWN_SECONDS so a
        stopped gateway costs one connect timeout per minute, not one per call;
        callers then fall back exactly as they do for a missing token.
        """
        now = time.monotonic()
        if now < self._unavailable_until:
            return False
        try:
            async def _probe(ib: Any) -> bool:
                return bool(ib.isConnected())
            return bool(self._run(_probe))
        except Exception as exc:
            logger.warning(
                "IB Gateway unavailable at %s:%s (%s). Start IB Gateway and log in, "
                "or set %s to another provider.",
                self.host, self.port, exc, "OPTIONS_CALCULATOR_OPTIONS_PROVIDER",
            )
            self._unavailable_until = now + UNAVAILABLE_COOLDOWN_SECONDS
            return False

    # ── quote ─────────────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Optional[float]:
        symbol = symbol.upper()
        try:
            async def _quote(ib: Any) -> Optional[float]:
                stock = await self._qualified_stock(ib, symbol)
                if stock is None:
                    return None
                return await self._spot(ib, stock)
            return self._run(_quote)
        except Exception as exc:
            logger.warning("IB Gateway get_quote(%s) failed: %s", symbol, exc)
            return None

    # ── expirations ───────────────────────────────────────────────────────────

    def get_expirations(self, symbol: str) -> List[str]:
        symbol = symbol.upper()
        try:
            async def _exps(ib: Any) -> List[str]:
                stock = await self._qualified_stock(ib, symbol)
                if stock is None:
                    return []
                params = await self._option_params(ib, stock)
                return [] if params is None else sorted(_iso_date(e) for e in params.expirations)
            return self._run(_exps)
        except Exception as exc:
            logger.warning("IB Gateway get_expirations(%s) failed: %s", symbol, exc)
            return []

    # ── option chain ──────────────────────────────────────────────────────────

    def get_option_chain(
        self,
        symbol: str,
        *,
        expiration: Optional[str] = None,
        strike_limit: int = 20,
        side: Optional[str] = None,
        date: Optional[str] = None,
        range_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Current option chain, normalized to the MarketDataClient schema.

        Returns an empty DataFrame on error or for historical `date` requests
        (the socket API serves no historical chains).
        """
        symbol = symbol.upper()
        start = time.perf_counter()

        if date:
            return pd.DataFrame()

        breaker = reliability.get_breaker(self.provider_name)
        if not breaker.allow():
            record_provider_telemetry(
                provider_name=self.provider_name,
                endpoint_type="options_chain_quality",
                request_kind="pipeline",
                symbol=symbol,
                success=False,
                error_category=reliability.SKIP_CIRCUIT_OPEN,
            )
            reliability.note_skip(provider=self.provider_name, endpoint_type="options_chain",
                                  reason=reliability.SKIP_CIRCUIT_OPEN, symbol=symbol)
            return pd.DataFrame()

        try:
            result = self._run(
                lambda ib: self._fetch_chain(ib, symbol, expiration, strike_limit, side, range_filter),
                timeout=max(120.0, self.timeout + 30.0),
            )
        except Exception as exc:
            logger.warning("IB Gateway get_option_chain(%s) failed: %s", symbol, exc)
            category = classify_error(f"{type(exc).__name__}: {exc}")
            if isinstance(exc, (IBGatewayConnectionError, asyncio.TimeoutError, TimeoutError)) or \
                    category in {"timeout", "rate_limited", "http_error"}:
                breaker.record_failure()
            else:
                breaker.release_probe()
            return self._empty_chain_telemetry(symbol, start, "fetch_error", error_category=category)

        frame, note = result
        breaker.record_success()
        if frame.empty:
            self._log_gateway_hint(symbol)
            return self._empty_chain_telemetry(symbol, start, note or "empty_after_normalize")

        underlying_price = None
        vals = pd.to_numeric(frame.get("underlyingPrice"), errors="coerce").dropna()
        if not vals.empty:
            underlying_price = float(vals.iloc[0])
        frame.attrs["surface_quality"] = diagnose_option_surface_quality(
            frame, underlying_price=underlying_price
        ).to_dict()
        frame.attrs["market_data_type"] = self.market_data_type_label
        frame.attrs["provider"] = self.provider_name

        if frame["bid"].isna().all() and frame["ask"].isna().all():
            self._log_gateway_hint(symbol)

        record_provider_telemetry(
            provider_name=self.provider_name,
            endpoint_type="options_chain_quality",
            request_kind="request",
            symbol=symbol,
            success=True,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            response_quality_note=f"market_data_type={self.market_data_type_label}",
        )
        return frame

    async def _fetch_chain(self, ib: Any, symbol: str, expiration: Optional[str], strike_limit: int,
                           side: Optional[str], range_filter: Optional[str]):
        async with self._session.chain_lock():
            stock = await self._qualified_stock(ib, symbol)
            if stock is None:
                return pd.DataFrame(), "unknown_symbol"
            params = await self._option_params(ib, stock)
            if params is None or not params.expirations:
                return pd.DataFrame(), "no_expirations"

            all_exps = sorted(_iso_date(e) for e in params.expirations)
            exp_arg = str(expiration).lower() if expiration else ""
            if exp_arg and exp_arg != "all":
                target_exps = [str(expiration)] if str(expiration) in all_exps else []
            elif exp_arg == "all":
                target_exps = all_exps[:MAX_EXPIRATIONS_FOR_ALL]
            else:
                target_exps = all_exps[:1]
            if not target_exps:
                return pd.DataFrame(), "expiration_not_listed"

            spot = await self._spot(ib, stock)
            strikes = sorted(float(s) for s in params.strikes if _finite(s))
            if not strikes:
                return pd.DataFrame(), "no_strikes"

            wanted = str(side).lower() if side else None
            rights = [r for r, name in (("C", "call"), ("P", "put")) if wanted not in ("call", "put") or wanted == name]

            contracts = []
            for exp in target_exps:
                chosen = _select_strikes(strikes, spot, strike_limit, range_filter)
                for strike in chosen:
                    for right in rights:
                        contracts.append(self._option_contract(symbol, exp, strike, right, params))
            if not contracts:
                return pd.DataFrame(), "no_contracts"

            qualified = await ib.qualifyContractsAsync(*contracts)
            qualified = [c for c in (qualified or []) if getattr(c, "conId", 0)]
            if not qualified:
                return pd.DataFrame(), "no_listed_contracts"

            rows: List[Dict[str, Any]] = []
            for batch in _batched(qualified, MAX_CONCURRENT_LINES):
                tickers = [ib.reqMktData(c, _OPTION_GENERIC_TICKS, False, False) for c in batch]
                try:
                    await self._settle(tickers)
                    for contract, ticker in zip(batch, tickers):
                        rows.append(self._row(symbol, contract, ticker, spot))
                finally:
                    for c in batch:
                        try:
                            ib.cancelMktData(c)
                        except Exception:  # pragma: no cover - cancel is best-effort
                            pass
            return self._frame(rows), None

    async def _settle(self, tickers: Iterable[Any]) -> None:
        """Wait until every ticker shows a two-sided quote and greeks, or the settle cap."""
        tickers = list(tickers)
        deadline = time.monotonic() + max(0.0, self.settle_seconds)
        while True:
            ready = all(
                (_finite(getattr(t, "bid", None)) or _finite(getattr(t, "ask", None)))
                and getattr(t, "modelGreeks", None) is not None
                for t in tickers
            )
            if ready or time.monotonic() >= deadline:
                return
            await asyncio.sleep(0.25)

    async def _qualified_stock(self, ib: Any, symbol: str) -> Any:
        from ib_async import Stock
        stock = Stock(symbol, "SMART", "USD")
        qualified = await ib.qualifyContractsAsync(stock)
        stock = qualified[0] if qualified else stock
        return stock if getattr(stock, "conId", 0) else None

    async def _option_params(self, ib: Any, stock: Any) -> Any:
        chains = await ib.reqSecDefOptParamsAsync(stock.symbol, "", stock.secType, stock.conId)
        chains = list(chains or [])
        if not chains:
            return None
        smart = [c for c in chains if getattr(c, "exchange", "") == "SMART"]
        pool = smart or chains
        same_class = [c for c in pool if getattr(c, "tradingClass", "") == stock.symbol]
        return (same_class or pool)[0]

    async def _spot(self, ib: Any, stock: Any) -> Optional[float]:
        """Underlying price: stream the stock and stop at the first usable tick.

        A snapshot request would wait for IB's snapshot-end marker, which on a
        delayed feed arrives only after ~11 s; streaming returns in a second.
        """
        ticker = ib.reqMktData(stock, "", False, False)
        try:
            deadline = time.monotonic() + max(1.0, self.settle_seconds)
            while True:
                price = _ticker_price(ticker)
                if price is not None or time.monotonic() >= deadline:
                    return price
                await asyncio.sleep(0.25)
        finally:
            try:
                ib.cancelMktData(stock)
            except Exception:  # pragma: no cover - cancel is best-effort
                pass

    @staticmethod
    def _option_contract(symbol: str, exp_iso: str, strike: float, right: str, params: Any) -> Any:
        from ib_async import Option
        return Option(
            symbol, exp_iso.replace("-", ""), strike, right, "SMART",
            tradingClass=getattr(params, "tradingClass", "") or symbol, currency="USD",
        )

    def _row(self, symbol: str, contract: Any, ticker: Any, spot: Optional[float]) -> Dict[str, Any]:
        side = "call" if str(getattr(contract, "right", "")).upper().startswith("C") else "put"
        strike = _to_float(getattr(contract, "strike", None))
        exp_iso = _iso_date(getattr(contract, "lastTradeDateOrContractMonth", ""))
        greeks = getattr(ticker, "modelGreeks", None)
        bid = _quote_value(getattr(ticker, "bid", None))
        ask = _quote_value(getattr(ticker, "ask", None))
        iv = _to_float(getattr(greeks, "impliedVol", None)) if greeks is not None else None
        if iv is None:
            iv = _to_float(getattr(ticker, "impliedVolatility", None))
        und = _to_float(getattr(greeks, "undPrice", None)) if greeks is not None else None
        if und is None or und <= 0:
            und = spot
        oi_attr = "callOpenInterest" if side == "call" else "putOpenInterest"
        try:
            dte = (_dt.date.fromisoformat(exp_iso) - _dt.date.today()).days
        except Exception:
            dte = np.nan
        intrinsic = np.nan
        if strike is not None and und is not None:
            intrinsic = max(0.0, und - strike) if side == "call" else max(0.0, strike - und)
        row = {
            "optionSymbol": getattr(contract, "localSymbol", None) or None,
            "underlying": symbol,
            "side": side,
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "lastPrice": _quote_value(getattr(ticker, "last", None)),
            "volume": _to_float(getattr(ticker, "volume", None)),
            "openInterest": _to_float(getattr(ticker, oi_attr, None)),
            "impliedVolatility": iv,
            "delta": _to_float(getattr(greeks, "delta", None)) if greeks is not None else None,
            "gamma": _to_float(getattr(greeks, "gamma", None)) if greeks is not None else None,
            "theta": _to_float(getattr(greeks, "theta", None)) if greeks is not None else None,
            "vega": _to_float(getattr(greeks, "vega", None)) if greeks is not None else None,
            "dte": dte,
            "underlyingPrice": und if und is not None else np.nan,
            "inTheMoney": (bool(intrinsic > 0) if _finite(intrinsic) else None),
            "intrinsicValue": intrinsic,
            "expiration_date": exp_iso,
        }
        if self.market_data_type in _REAL_TIME_MARKET_DATA_TYPES:
            stamp = getattr(ticker, "time", None)
            row["updated"] = stamp.timestamp() if isinstance(stamp, _dt.datetime) else np.nan
        return row

    def _frame(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility",
                    "delta", "gamma", "theta", "vega", "underlyingPrice", "intrinsicValue"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Canonical mid rule (utils.quotes.safe_mid_series): NaN for a zero/absent
        # bid or crossed quote, never a fabricated (0 + ask)/2.
        df["mid"] = safe_mid_series(df["bid"], df["ask"])
        df["extrinsicValue"] = df["mid"] - df["intrinsicValue"]
        df["inTheMoney"] = df["inTheMoney"].astype(object)
        columns = list(_CHAIN_COLUMNS) + (["updated"] if "updated" in df.columns else [])
        df = df[columns]
        df = df.sort_values(["expiration_date", "strike", "side"]).reset_index(drop=True)
        return df

    def _log_gateway_hint(self, symbol: str) -> None:
        """Explain an all-NaN chain from what the gateway said and what time it is."""
        codes = {str(e.get("code")) for e in self._session.recent_errors[-20:]}
        reasons = []
        if not in_regular_session(_dt.datetime.now(_dt.timezone.utc)):
            reasons.append("outside the regular NYSE session, when delayed feeds publish no option bid/ask")
        subscription_codes = sorted(codes & {"354", "10089", "10091", "10168"})
        if subscription_codes:
            reasons.append(
                "this account has no live OPRA market-data subscription for the API "
                f"(IB codes {', '.join(subscription_codes)}), so only delayed data is served"
            )
        if not reasons:
            reasons.append("no quotes arrived within the settle window; raise "
                           f"{SETTLE_ENV} or check the gateway's Market Data Connections dialog")
        logger.warning("IB Gateway returned no option quotes for %s: %s.", symbol, "; ".join(reasons))

    def _empty_chain_telemetry(self, symbol: str, start: float, note: str,
                               error_category: str = "empty_response") -> pd.DataFrame:
        record_provider_telemetry(
            provider_name=self.provider_name,
            endpoint_type="options_chain_quality",
            request_kind="request",
            symbol=symbol,
            success=False,
            error_category=error_category,
            response_quality_note=note,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
        return pd.DataFrame()

    # ── earnings ──────────────────────────────────────────────────────────────

    def get_earnings(self, symbol: str, countback: int = 24) -> pd.DataFrame:
        """Earnings history from the yfinance provider (IB has no earnings feed)."""
        try:
            return self._earnings().get_earnings(symbol, countback=countback)
        except Exception as exc:
            logger.warning("IB Gateway get_earnings(%s) via yfinance failed: %s", symbol, exc)
            return pd.DataFrame()


# ── module helpers ────────────────────────────────────────────────────────────

def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> Optional[float]:
    return float(value) if _finite(value) else None


def _quote_value(value: Any) -> Optional[float]:
    """IB sends -1 (or NaN) for "no quote"; zero is a real, empty bid."""
    f = _to_float(value)
    return None if f is None or f < 0 else f


def _call_or_attr(obj: Any, name: str) -> Any:
    attr = getattr(obj, name, None)
    try:
        return attr() if callable(attr) else attr
    except Exception:
        return None


def _ticker_price(ticker: Any) -> Optional[float]:
    """First usable price on a ticker: market price, then last, then prior close."""
    for candidate in (
        _call_or_attr(ticker, "marketPrice"),
        getattr(ticker, "last", None),
        getattr(ticker, "close", None),
    ):
        if _finite(candidate) and float(candidate) > 0:
            return float(candidate)
    return None


def _iso_date(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def _select_strikes(strikes: List[float], spot: Optional[float], strike_limit: int,
                    range_filter: Optional[str]) -> List[float]:
    """The ``strike_limit`` listed strikes nearest spot, honouring an itm/otm filter.

    Without a spot there is no "nearest", so the middle of the listed range is
    used rather than an arbitrary end.
    """
    pool = list(strikes)
    if spot is not None and range_filter:
        rf = str(range_filter).lower()
        if rf == "itm":
            # Both legs of an ITM request straddle spot; keep the strikes around it.
            pool = [s for s in pool if abs(s - spot) <= abs(spot) * 0.5]
        elif rf == "otm":
            pool = [s for s in pool if abs(s - spot) <= abs(spot) * 0.5]
    if not strike_limit or len(pool) <= strike_limit:
        return pool
    if spot is None:
        mid = len(pool) // 2
        half = strike_limit // 2
        return pool[max(0, mid - half): max(0, mid - half) + strike_limit]
    # Ties (equidistant strikes) resolve to the lower strike, deterministically.
    return sorted(sorted(pool, key=lambda s: (abs(s - spot), s))[:strike_limit])


def _batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), max(1, size)):
        yield items[i:i + size]
