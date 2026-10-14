"""
MarketData.app API Client
=========================
Primary options data source for IV, Greeks, option chains, and earnings timing.
yfinance remains the source for price history and realized volatility.

Credit budget (100k/day):
  - Real-time chain (expiration=all, strikeLimit=10): ~150-250 credits per analysis
  - Historical chain (date=YYYY-MM-DD): 1 credit per 1,000 contracts
  - Earnings endpoint: 1 credit per call
  - Expirations endpoint: 1 credit per call
"""

from __future__ import annotations

import math
import os
import random
import time
import logging
from collections.abc import Mapping
from datetime import datetime, timezone, date as date_type
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from services import provider_reliability as reliability
from services.provider_telemetry import classify_error, record_provider_telemetry
from services.option_surface_quality import diagnose_option_surface_quality

logger = logging.getLogger(__name__)

_MARKETDATA_BASE_URL = "https://api.marketdata.app/v1"
_MARKETDATA_TOKEN_ENV = "MARKETDATA_TOKEN"

# Column name mapping: MarketData.app → normalized (compatible with existing code)
_CHAIN_COLUMN_MAP = {
    "iv": "impliedVolatility",
    "last": "lastPrice",
}

# Columns to include from the MDApp chain response
_CHAIN_COLUMNS = [
    "optionSymbol",
    "underlying",
    "expiration",         # unix timestamp → converted to expiration_date
    "side",
    "strike",
    "bid",
    "ask",
    "mid",
    "lastPrice",          # normalized from "last"
    "volume",
    "openInterest",
    "impliedVolatility",  # normalized from "iv"
    "delta",
    "gamma",
    "theta",
    "vega",
    "dte",
    "underlyingPrice",
    "inTheMoney",
    "intrinsicValue",
    "extrinsicValue",
    "updated",
]


class MarketDataError(Exception):
    """Raised for non-retryable MarketData.app API errors."""


class MarketDataClient:
    """
    Thin, cache-enabled wrapper around the MarketData.app REST API.

    The cache is an in-process dict keyed on (url, sorted params), backed for
    historical chains, earnings and expirations by a shared on-disk cache
    (services.provider_reliability). TTL varies by endpoint:
      - Live option chains: 60 s
      - Historical option chains: 24 h (immutable after settlement)
      - Earnings calendar: 1 h
      - Expirations: 5 min
      - Quotes: 30 s
    """

    # Stable provider label (parity with YFinanceMarketDataClient.provider_name)
    # so consumers can report the active provider without an isinstance check.
    provider_name = "marketdata_app"

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        response_cache: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ):
        # Explicit token argument takes precedence.
        # Passing token="" should intentionally disable API usage (useful for tests/fallback paths).
        self.token = os.environ.get(_MARKETDATA_TOKEN_ENV, "") if token is None else token
        self._cache: Dict[str, Tuple[float, Any]] = {}
        # Shared on-disk cache for slow-moving endpoints: None uses the default
        # store, False disables it.
        self._response_cache = response_cache
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._earnings_endpoint_blocked: bool = False
        # Per-symbol oldest historical date that returned 402 for chain endpoint.
        # Requests at or before this date are skipped for the current process.
        self._historical_chain_402_cutoff_by_symbol: Dict[str, date_type] = {}
        self._historical_chain_warned_symbols: set[str] = set()

    # ------------------------------------------------------------------
    # Public API — availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if an API token is configured."""
        return bool(self.token and self.token.strip())

    # ------------------------------------------------------------------
    # Expirations
    # ------------------------------------------------------------------

    def get_expirations(self, symbol: str) -> List[str]:
        """
        Return sorted list of available option expiration dates as 'YYYY-MM-DD'
        strings for *symbol*.  Returns [] on any error.
        """
        url = f"{_MARKETDATA_BASE_URL}/options/expirations/{symbol.upper()}/"
        try:
            data = self._get(url, cache_ttl=300.0)
        except Exception as exc:
            logger.warning("MDApp get_expirations(%s) failed: %s", symbol, exc)
            return []

        if data.get("s") != "ok":
            return []

        raw = data.get("expirations", []) or []
        result: List[str] = []
        for item in raw:
            try:
                if isinstance(item, (int, float)):
                    result.append(
                        datetime.fromtimestamp(float(item), tz=timezone.utc).strftime("%Y-%m-%d")
                    )
                else:
                    # Already a date string
                    result.append(str(item)[:10])
            except Exception:
                continue
        return sorted(result)

    # ------------------------------------------------------------------
    # Option chain
    # ------------------------------------------------------------------

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
        """
        Fetch an option chain and return a normalized DataFrame.

        Column normalization for backward-compatibility with existing engine code:
          - 'iv'   → 'impliedVolatility'
          - 'last' → 'lastPrice'
          - unix 'expiration' timestamps → 'expiration_date' (YYYY-MM-DD string)

        Parameters
        ----------
        symbol       : Underlying ticker symbol.
        expiration   : 'all' for full chain, or specific 'YYYY-MM-DD'.
        strike_limit : Max strikes returned per expiration (caps credit usage).
        side         : 'call', 'put', or None for both.
        date         : Historical date 'YYYY-MM-DD'. If provided, cache for 24 h.
        range_filter : 'itm', 'otm', or 'all'.

        Returns empty DataFrame on any error.
        """
        symbol = symbol.upper()
        url = f"{_MARKETDATA_BASE_URL}/options/chain/{symbol}/"
        params: Dict[str, Any] = {"strikeLimit": strike_limit}
        if expiration:
            params["expiration"] = expiration
        if side:
            params["side"] = side
        if date:
            params["date"] = date
        if range_filter:
            params["range"] = range_filter

        cache_ttl = 86_400.0 if date else 60.0

        if date:
            try:
                req_date = datetime.strptime(str(date), "%Y-%m-%d").date()
                cutoff = self._historical_chain_402_cutoff_by_symbol.get(symbol)
                if cutoff is not None and req_date <= cutoff:
                    # Known unsupported historical range for this symbol on this plan.
                    return pd.DataFrame()
            except Exception:
                req_date = None  # type: ignore
        else:
            req_date = None  # type: ignore

        try:
            data = self._get(url, params=params, cache_ttl=cache_ttl)
        except Exception as exc:
            message = str(exc)
            if date and "402" in message:
                # Historical depth likely exceeded current entitlement.
                if req_date is not None:
                    prev = self._historical_chain_402_cutoff_by_symbol.get(symbol)
                    if prev is None or req_date > prev:
                        self._historical_chain_402_cutoff_by_symbol[symbol] = req_date
                if symbol not in self._historical_chain_warned_symbols:
                    self._historical_chain_warned_symbols.add(symbol)
                    logger.warning(
                        "MDApp historical chain entitlement limit detected for %s (first 402 at %s). "
                        "Skipping older historical dates for this symbol in this run.",
                        symbol,
                        date,
                    )
            else:
                logger.warning("MDApp get_option_chain(%s) failed: %s", symbol, exc)
            return pd.DataFrame()

        if data.get("s") != "ok":
            record_provider_telemetry(
                provider_name="marketdata_app",
                endpoint_type="options_chain_quality",
                request_kind="quality",
                symbol=symbol,
                success=False,
                error_category="empty_response",
                response_quality_note=f"status={data.get('s')}",
            )
            logger.debug("MDApp chain no data for %s: s=%s", symbol, data.get("s"))
            return pd.DataFrame()

        frame = self._chain_response_to_df(data)
        if not frame.empty:
            underlying_price = None
            if "underlyingPrice" in frame.columns:
                values = pd.to_numeric(frame["underlyingPrice"], errors="coerce").dropna()
                if not values.empty:
                    underlying_price = float(values.iloc[0])
            frame.attrs["surface_quality"] = diagnose_option_surface_quality(
                frame,
                underlying_price=underlying_price,
            ).to_dict()
        if frame.empty:
            record_provider_telemetry(
                provider_name="marketdata_app",
                endpoint_type="options_chain_quality",
                request_kind="quality",
                symbol=symbol,
                success=False,
                error_category="empty_response",
                response_quality_note="chain payload converted to empty frame",
            )
        return frame

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------

    def get_earnings(
        self,
        symbol: str,
        countback: int = 24,
    ) -> pd.DataFrame:
        """
        Return earnings history for *symbol* with BMO/AMC release timing.

        Key columns in returned DataFrame:
          - report_date     : datetime.date — the calendar date of the report
          - reportTime      : str — 'before market open' | 'after market close' | 'during market hours'
          - fiscalYear      : int
          - fiscalQuarter   : int
          - reportedEPS     : float
          - estimatedEPS    : float
          - surpriseEPS     : float
          - surpriseEPSpct  : float

        Returns empty DataFrame on any error.
        """
        if self._earnings_endpoint_blocked:
            return pd.DataFrame()

        url = f"{_MARKETDATA_BASE_URL}/stocks/earnings/{symbol.upper()}/"
        params: Dict[str, Any] = {"countback": countback}

        try:
            data = self._get(url, params=params, cache_ttl=3_600.0)
        except Exception as exc:
            message = str(exc)
            if "402" in message:
                if not self._earnings_endpoint_blocked:
                    logger.warning(
                        "MDApp earnings endpoint unavailable (HTTP 402). "
                        "Falling back to yfinance/cached earnings events for this run."
                    )
                self._earnings_endpoint_blocked = True
            else:
                logger.warning("MDApp get_earnings(%s) failed: %s", symbol, exc)
            return pd.DataFrame()

        if data.get("s") not in ("ok",):
            record_provider_telemetry(
                provider_name="marketdata_app",
                endpoint_type="earnings_quality",
                request_kind="quality",
                symbol=symbol,
                success=False,
                error_category="empty_response",
                response_quality_note=f"status={data.get('s')}",
            )
            return pd.DataFrame()

        n = max(
            len(data.get("date", [])),
            len(data.get("reportDate", [])),
        )
        if n == 0:
            record_provider_telemetry(
                provider_name="marketdata_app",
                endpoint_type="earnings_quality",
                request_kind="quality",
                symbol=symbol,
                success=False,
                error_category="empty_response",
                response_quality_note="earnings payload had zero rows",
            )
            return pd.DataFrame()

        def _col(key: str, n: int) -> list:
            v = data.get(key, [])
            return list(v) + [None] * max(0, n - len(v))

        df = pd.DataFrame({
            "symbol": _col("symbol", n),
            "fiscalYear": _col("fiscalYear", n),
            "fiscalQuarter": _col("fiscalQuarter", n),
            "date_unix": _col("date", n),
            "reportDate_unix": _col("reportDate", n),
            "reportTime": _col("reportTime", n),
            "currency": _col("currency", n),
            "reportedEPS": _col("reportedEPS", n),
            "estimatedEPS": _col("estimatedEPS", n),
            "surpriseEPS": _col("surpriseEPS", n),
            "surpriseEPSpct": _col("surpriseEPSpct", n),
        })

        # Parse mixed unix-timestamp / ISO-date payloads into date objects.
        for raw_col, date_col in (("date_unix", "event_date"), ("reportDate_unix", "report_date")):
            df[date_col] = df[raw_col].map(_parse_marketdata_date)

        df.drop(columns=["date_unix", "reportDate_unix"], inplace=True)
        if "report_date" in df.columns:
            df["report_date"] = df["report_date"].where(df["report_date"].notna(), df["event_date"])

        # Normalise reportTime to short form for convenience
        def _normalise_rt(v: Any) -> Optional[str]:
            if v is None:
                return None
            s = str(v).lower().strip()
            if "before" in s:
                return "before market open"
            if "after" in s:
                return "after market close"
            if "during" in s:
                return "during market hours"
            return s or None

        df["reportTime"] = df["reportTime"].map(_normalise_rt)

        # Sort newest → oldest
        df.sort_values("report_date", ascending=False, inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)

        # pandas sort_values converts datetime.date objects to pd.Timestamp internally;
        # re-normalize both date columns back to datetime.date so callers get consistent types.
        import datetime as _dt
        for _dc in ("event_date", "report_date"):
            if _dc in df.columns:
                df[_dc] = df[_dc].apply(
                    lambda x: x.date() if isinstance(x, pd.Timestamp) else x
                )

        return df

    # ------------------------------------------------------------------
    # Stock quote (current price)
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Optional[float]:
        """
        Return the most recent trade price for *symbol*.
        Used only when yfinance history is not available. 30-second cache.
        """
        if not self.is_available():
            return None
        url = f"{_MARKETDATA_BASE_URL}/stocks/quotes/{symbol.upper()}/"
        try:
            data = self._get(url, cache_ttl=30.0)
        except Exception as exc:
            logger.warning("MDApp get_quote(%s) failed: %s", symbol, exc)
            return None

        if data.get("s") != "ok":
            record_provider_telemetry(
                provider_name="marketdata_app",
                endpoint_type="quote_quality",
                request_kind="quality",
                symbol=symbol,
                success=False,
                error_category="empty_response",
                response_quality_note=f"status={data.get('s')}",
            )
            return None

        last_list = data.get("last", [])
        if last_list:
            try:
                return float(last_list[0])
            except (TypeError, ValueError):
                pass
        record_provider_telemetry(
            provider_name="marketdata_app",
            endpoint_type="quote_quality",
            request_kind="quality",
            symbol=symbol,
            success=False,
            error_category="empty_response",
            response_quality_note="quote payload missing parseable last price",
        )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Token {self.token}"}

    def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Issue a GET request with caching, the daily credit budget, a circuit
        breaker, and retries with jittered backoff on transient failures
        (429, 5xx, timeouts, connection errors). Other HTTP errors are not
        retried. Raises MarketDataError on failure or when the call is skipped.
        Logs one telemetry row per logical request.
        """
        cache_key = url + "|" + str(sorted((params or {}).items()))
        now = time.monotonic()
        endpoint_type, symbol = _marketdata_endpoint_from_url(url)

        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if now - ts < cache_ttl:
                return cached

        disk_cache = self._disk_cache() if _persistable(endpoint_type, params) else None
        if disk_cache is not None:
            stored = disk_cache.get(cache_key)
            if isinstance(stored, dict):
                self._cache[cache_key] = (now, stored)
                return stored

        provider = self.provider_name
        priority = reliability.current_priority()
        estimate = _estimated_credits(endpoint_type, params)
        breaker = reliability.get_breaker(provider)
        skip_reason = None if breaker.allow() else reliability.SKIP_CIRCUIT_OPEN
        reservation: Optional[str] = reliability.UNMETERED
        if skip_reason is None:
            try:
                # Admit and hold the estimated cost atomically across processes.
                reservation = reliability.get_credit_budget().reserve(provider, priority, estimate)
            except Exception:  # noqa: BLE001 - an unreadable ledger must not stop data collection
                logger.debug("MDApp budget unavailable", exc_info=True)
                reservation = reliability.UNMETERED
            if reservation is None:
                breaker.release_probe()
                skip_reason = reliability.SKIP_BUDGET
        if skip_reason is not None:
            # Not sent, so not a provider request: a pipeline row.
            record_provider_telemetry(
                provider_name=provider,
                endpoint_type=endpoint_type,
                request_kind="pipeline",
                symbol=symbol,
                success=False,
                error_category=skip_reason,
                response_quality_note=f"priority={priority}",
            )
            reliability.note_skip(provider=provider, endpoint_type=endpoint_type, reason=skip_reason, symbol=symbol)
            try:
                reliability.get_credit_budget().record_skip(provider, priority)
            except Exception:  # noqa: BLE001
                logger.debug("MDApp budget skip not recorded", exc_info=True)
            raise MarketDataError(f"MDApp call skipped: {skip_reason}")

        max_retries = 3
        start = time.perf_counter()
        last_category = "unknown_error"
        settled = False
        health_reported = False

        def _fail(category: str, note: Optional[str] = None) -> None:
            record_provider_telemetry(
                provider_name=provider,
                endpoint_type=endpoint_type,
                symbol=symbol,
                success=False,
                error_category=category,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                response_quality_note=note,
            )

        try:
            for attempt in range(max_retries):
                retry_after: Any = None
                try:
                    resp = requests.get(
                        url,
                        headers=self._headers(),
                        params=params,
                        timeout=15,
                    )
                except requests.exceptions.Timeout:
                    last_category = "timeout"
                    logger.warning("MDApp request timeout (attempt %d): %s", attempt + 1, url)
                except requests.exceptions.ConnectionError as exc:
                    last_category = "connection_error"
                    logger.warning("MDApp connection error (attempt %d): %s", attempt + 1, exc)
                except requests.exceptions.RequestException as exc:
                    _fail(classify_error(str(exc)))
                    raise MarketDataError(f"MDApp request error: {exc}") from exc
                else:
                    # What the provider says about the quota counts on every
                    # response, errors included.
                    self._record_reported_quota(resp)
                    status = resp.status_code if isinstance(resp.status_code, int) else 200
                    if status == 429:
                        kind = _classify_429(resp)
                        if kind == QUOTA_EXHAUSTED:
                            # Retrying cannot help before the quota resets, and
                            # it is the budget's job, not the breaker's.
                            _fail(QUOTA_EXHAUSTED, note="provider reports the credit quota is used up")
                            try:
                                # The budget sees the exhaustion even without headers.
                                reliability.get_credit_budget().record_reported_quota(
                                    provider, reported_remaining=0.0)
                            except Exception:  # noqa: BLE001
                                logger.debug("MDApp quota exhaustion not recorded", exc_info=True)
                            raise MarketDataError(f"MDApp credit quota exhausted: {url}")
                        last_category = kind
                        retry_after = _header_map(resp).get("retry-after")
                        logger.warning("MDApp HTTP 429 %s (attempt %d): %s", kind, attempt + 1, url)
                    elif status >= 500:
                        last_category = "server_error"
                        retry_after = _header_map(resp).get("retry-after")
                        logger.warning("MDApp HTTP %d (attempt %d): %s", status, attempt + 1, url)
                    else:
                        try:
                            resp.raise_for_status()
                            data: Dict[str, Any] = resp.json()
                        except requests.exceptions.HTTPError as exc:
                            # A client error (auth, entitlement, not found) will not
                            # change on retry and says nothing about provider health.
                            _fail(classify_error(str(exc)))
                            raise MarketDataError(f"MDApp HTTP error: {exc}") from exc
                        except Exception as exc:
                            _fail(classify_error(str(exc)))
                            raise
                        breaker.record_success()
                        health_reported = True
                        self._charge(resp, data, endpoint_type, params, priority, estimate, reservation)
                        settled = True
                        self._cache[cache_key] = (now, data)
                        if disk_cache is not None:
                            disk_cache.put(cache_key, data, cache_ttl)
                        record_provider_telemetry(
                            provider_name=provider,
                            endpoint_type=endpoint_type,
                            symbol=symbol,
                            success=True,
                            latency_ms=(time.perf_counter() - start) * 1000.0,
                            response_quality_note=f"retried:{attempt}" if attempt else None,
                        )
                        return data
                if attempt < max_retries - 1:
                    self._sleep(reliability.retry_delay(attempt, retry_after=retry_after, rng=self._rng))

            breaker.record_failure()
            health_reported = True
            _fail(last_category, note=f"failed after {max_retries} attempts")
            raise MarketDataError(f"MDApp {last_category} after {max_retries} attempts: {url}")
        finally:
            if not settled:
                # Not served, so not charged: free the held estimate.
                try:
                    reliability.get_credit_budget().release(reservation)
                except Exception:  # noqa: BLE001
                    logger.debug("MDApp reservation not released", exc_info=True)
            if not health_reported:
                breaker.release_probe()

    def _record_reported_quota(self, resp: Any) -> None:
        headers = _header_map(resp)
        limit = _header_number(headers, "x-api-ratelimit-limit")
        remaining = _header_number(headers, "x-api-ratelimit-remaining")
        reset = _header_number(headers, "x-api-ratelimit-reset")
        if limit is None and remaining is None:
            return
        try:
            reliability.get_credit_budget().record_reported_quota(
                self.provider_name, reported_limit=limit, reported_remaining=remaining,
                reported_reset_at=datetime.fromtimestamp(reset, tz=timezone.utc) if reset else None,
            )
        except Exception:  # noqa: BLE001
            logger.debug("MDApp reported quota not recorded", exc_info=True)

    def _disk_cache(self) -> Optional["reliability.ResponseCache"]:
        if self._response_cache is False:
            return None
        return self._response_cache or reliability.get_response_cache()

    def _charge(self, resp: Any, data: Dict[str, Any], endpoint_type: str,
                params: Optional[Dict[str, Any]], priority: str, estimate: float,
                reservation: Optional[str] = None) -> None:
        """Settle the reservation at the call's actual cost (reconciliation)."""
        headers = _header_map(resp)
        consumed = _header_number(headers, "x-api-ratelimit-consumed")
        credits = consumed if consumed is not None else _actual_credits(endpoint_type, params, data, estimate)
        try:
            reliability.get_credit_budget().settle(reservation, self.provider_name, priority, credits)
        except Exception:  # noqa: BLE001 - accounting must not lose a good response
            logger.debug("MDApp credit charge not recorded", exc_info=True)

    @staticmethod
    def _chain_response_to_df(data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert the columnar MDApp chain response to a normalized DataFrame.
        Column names are mapped for backward-compatibility with edge_engine.py.
        """
        # Determine length from any present array field
        ref_col = next(
            (k for k in ("optionSymbol", "strike", "iv", "bid") if data.get(k)),
            None,
        )
        if ref_col is None:
            return pd.DataFrame()

        n = len(data[ref_col])

        def _get_col(src_key: str, n: int) -> list:
            v = data.get(src_key, [])
            return list(v) + [None] * max(0, n - len(v))

        raw: Dict[str, list] = {
            "optionSymbol": _get_col("optionSymbol", n),
            "underlying": _get_col("underlying", n),
            "_expiration_unix": _get_col("expiration", n),
            "side": _get_col("side", n),
            "strike": _get_col("strike", n),
            "bid": _get_col("bid", n),
            "ask": _get_col("ask", n),
            "mid": _get_col("mid", n),
            "lastPrice": _get_col("last", n),        # normalized
            "volume": _get_col("volume", n),
            "openInterest": _get_col("openInterest", n),
            "impliedVolatility": _get_col("iv", n),  # normalized
            "delta": _get_col("delta", n),
            "gamma": _get_col("gamma", n),
            "theta": _get_col("theta", n),
            "vega": _get_col("vega", n),
            "dte": _get_col("dte", n),
            "underlyingPrice": _get_col("underlyingPrice", n),
            "inTheMoney": _get_col("inTheMoney", n),
            "intrinsicValue": _get_col("intrinsicValue", n),
            "extrinsicValue": _get_col("extrinsicValue", n),
        }

        df = pd.DataFrame(raw)

        # Convert unix expiration to date string
        df["expiration_date"] = pd.to_datetime(
            pd.to_numeric(df["_expiration_unix"], errors="coerce"),
            unit="s",
            utc=True,
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        df.drop(columns=["_expiration_unix"], inplace=True)

        # Coerce numeric columns
        for col in ("strike", "bid", "ask", "mid", "lastPrice", "volume",
                    "openInterest", "impliedVolatility", "delta", "gamma",
                    "theta", "vega", "dte", "underlyingPrice",
                    "intrinsicValue", "extrinsicValue"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


def _parse_marketdata_date(value: Any) -> Any:
    """Parse MarketData date fields that may arrive as unix timestamps or ISO strings."""
    if value is None or value == "":
        return pd.NaT

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return pd.NaT
        if stripped.isdigit():
            value = int(stripped)
        else:
            parsed = pd.to_datetime(stripped, utc=True, errors="coerce")
            return parsed.date() if not pd.isna(parsed) else pd.NaT

    if isinstance(value, (int, float, np.integer, np.floating)):
        parsed = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
        return parsed.date() if not pd.isna(parsed) else pd.NaT

    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return parsed.date() if not pd.isna(parsed) else pd.NaT


def _marketdata_endpoint_from_url(url: str) -> Tuple[str, Optional[str]]:
    parts = [part for part in str(url).split("/") if part]
    if "chain" in parts:
        idx = parts.index("chain")
        return "options_chain", parts[idx + 1].upper() if idx + 1 < len(parts) else None
    if "earnings" in parts:
        idx = parts.index("earnings")
        return "earnings", parts[idx + 1].upper() if idx + 1 < len(parts) else None
    if "quotes" in parts:
        idx = parts.index("quotes")
        return "quote", parts[idx + 1].upper() if idx + 1 < len(parts) else None
    if "expirations" in parts:
        idx = parts.index("expirations")
        return "options_expirations", parts[idx + 1].upper() if idx + 1 < len(parts) else None
    return "unknown", None


# Endpoints whose payload does not move within its TTL. Live chains and quotes
# are never shared across processes (a later run must not price the present
# with an older quote).
def _persistable(endpoint_type: str, params: Optional[Dict[str, Any]]) -> bool:
    if endpoint_type in {"earnings", "options_expirations"}:
        return True
    return endpoint_type == "options_chain" and bool((params or {}).get("date"))


def _estimated_credits(endpoint_type: str, params: Optional[Dict[str, Any]]) -> float:
    """Upper-bound cost of a call, used to admit it under the budget."""
    params = params or {}
    if endpoint_type != "options_chain" or params.get("date"):
        return 1.0
    # Live chains cost one credit per contract returned.
    strikes = max(1, int(params.get("strikeLimit") or 20))
    sides = 1 if params.get("side") else 2
    expirations = 12 if str(params.get("expiration") or "").lower() == "all" else 1
    return float(strikes * sides * expirations)


def _actual_credits(endpoint_type: str, params: Optional[Dict[str, Any]], data: Dict[str, Any],
                    estimate: float) -> float:
    """Cost of a served call when the provider did not report it."""
    if endpoint_type != "options_chain":
        return 1.0
    contracts = data.get("optionSymbol") if isinstance(data, dict) else None
    if not isinstance(contracts, list):
        return 1.0 if (params or {}).get("date") else estimate
    if (params or {}).get("date"):
        return float(max(1, math.ceil(len(contracts) / 1000)))
    return float(max(1, len(contracts)))


QUOTA_EXHAUSTED = "quota_exhausted"
CONCURRENCY_THROTTLED = "concurrency_throttled"
RATE_LIMITED = "rate_limited"


def _classify_429(resp: Any) -> str:
    """Why the provider throttled: the day's credits are gone (not retryable
    before the reset), too many requests in flight, or a transient rate limit."""
    headers = _header_map(resp)
    if _header_number(headers, "x-api-ratelimit-remaining") == 0:
        return QUOTA_EXHAUSTED
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = None
    text = str((body or {}).get("errmsg") or "" if isinstance(body, dict) else body or "").lower()
    if "concurren" in text or "simultaneous" in text:
        return CONCURRENCY_THROTTLED
    if any(word in text for word in ("credit", "quota", "daily limit", "plan limit")):
        return QUOTA_EXHAUSTED
    return RATE_LIMITED


def _header_map(resp: Any) -> Dict[str, Any]:
    headers = getattr(resp, "headers", None)
    if not isinstance(headers, Mapping):
        return {}
    return {str(key).lower(): value for key, value in headers.items()}


def _header_number(headers: Dict[str, Any], name: str) -> Optional[float]:
    try:
        value = float(headers[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None
