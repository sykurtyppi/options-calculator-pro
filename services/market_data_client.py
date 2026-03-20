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

import os
import time
import logging
from datetime import datetime, timezone, date as date_type
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

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

    The cache is an in-process dict keyed on (url, sorted params).
    TTL varies by endpoint:
      - Live option chains: 60 s
      - Historical option chains: 24 h (immutable after settlement)
      - Earnings calendar: 1 h
      - Expirations: 5 min
      - Quotes: 30 s
    """

    def __init__(self, token: Optional[str] = None):
        # Explicit token argument takes precedence.
        # Passing token="" should intentionally disable API usage (useful for tests/fallback paths).
        self.token = os.environ.get(_MARKETDATA_TOKEN_ENV, "") if token is None else token
        self._cache: Dict[str, Tuple[float, Any]] = {}
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
            logger.debug("MDApp chain no data for %s: s=%s", symbol, data.get("s"))
            return pd.DataFrame()

        return self._chain_response_to_df(data)

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
            return pd.DataFrame()

        n = max(
            len(data.get("date", [])),
            len(data.get("reportDate", [])),
        )
        if n == 0:
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
            return None

        last_list = data.get("last", [])
        if last_list:
            try:
                return float(last_list[0])
            except (TypeError, ValueError):
                pass
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
        Issue a GET request with caching and retry on 429 (rate-limit).
        Raises MarketDataError on persistent failure.
        """
        cache_key = url + "|" + str(sorted((params or {}).items()))
        now = time.monotonic()

        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if now - ts < cache_ttl:
                return cached

        max_retries = 3
        backoff = 1.5
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=15,
                )
                if resp.status_code == 429:
                    wait = backoff * (attempt + 1)
                    logger.warning("MDApp rate-limited, waiting %.1fs (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data: Dict[str, Any] = resp.json()
                self._cache[cache_key] = (now, data)
                return data
            except requests.exceptions.Timeout:
                logger.warning("MDApp request timeout (attempt %d): %s", attempt + 1, url)
                if attempt == max_retries - 1:
                    raise MarketDataError(f"MDApp timeout after {max_retries} attempts: {url}")
            except requests.exceptions.HTTPError as exc:
                raise MarketDataError(f"MDApp HTTP error: {exc}") from exc

        raise MarketDataError(f"MDApp max retries exceeded: {url}")

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
