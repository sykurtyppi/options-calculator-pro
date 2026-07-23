"""Yahoo Finance options/earnings provider — a drop-in for MarketDataClient.

Implements the same 5-method interface (is_available / get_expirations /
get_option_chain / get_earnings / get_quote) on top of yfinance, so the live
web API and evidence cycle can run on free data while the paid MarketData.app
plan is deferred. Selected via the OPTIONS_CALCULATOR_OPTIONS_PROVIDER flag (see
build_market_data_client); flip back to "marketdata_app" once a paid sub lands.

Data-fidelity caveats (the price of "free"):
  - No greeks. yfinance does not expose delta/gamma/theta/vega/rho — those
    columns are emitted as NaN. Downstream metrics that need greeks degrade;
    after the M5 evidence-quality fix they degrade HONESTLY (missing surface
    fields are flagged, never read as zero).
  - IV is present but flakier than a paid feed (sparse/￶0 on illiquid strikes).
  - No historical chains. get_option_chain(date=...) returns an empty frame —
    yfinance only serves the current chain.
  - Earnings BMO/AMC timing is best-effort (only the upcoming event has a
    reliable timestamp); historical rows carry reportTime=None ("unknown").
"""
from __future__ import annotations

import datetime as _dt
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.quotes import safe_mid_series

from services.earnings_move_profile import (
    normalize_release_timing as _normalize_release_timing_shared,
)
from services.option_surface_quality import diagnose_option_surface_quality
from services.provider_telemetry import record_provider_telemetry

logger = logging.getLogger(__name__)

# Normalized chain columns expected by edge_engine / screener / institutional_ml_db
# (mirrors MarketDataClient._chain_response_to_df). Greeks are emitted as NaN.
_CHAIN_COLUMNS = [
    "optionSymbol", "underlying", "side", "strike", "bid", "ask", "mid",
    "lastPrice", "volume", "openInterest", "impliedVolatility",
    "delta", "gamma", "theta", "vega", "dte", "underlyingPrice",
    "inTheMoney", "intrinsicValue", "extrinsicValue", "expiration_date",
]


class YFinanceMarketDataClient:
    """yfinance-backed implementation of the MarketDataClient interface."""

    provider_name = "yfinance"

    def __init__(self, *, ticker_factory=None) -> None:
        # ticker_factory is injectable for testing (defaults to yfinance.Ticker).
        if ticker_factory is None:
            import yfinance as yf
            ticker_factory = yf.Ticker
        self._ticker_factory = ticker_factory

    # ── availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        # yfinance needs no token; it is always "available" (individual calls may
        # still fail and return empty, exactly like the paid client).
        return True

    # ── expirations ───────────────────────────────────────────────────────────

    def get_expirations(self, symbol: str) -> List[str]:
        try:
            exps = self._ticker_factory(symbol.upper()).options
            return [str(e) for e in (exps or [])]
        except Exception as exc:  # pragma: no cover - network/parse guard
            logger.warning("yfinance get_expirations(%s) failed: %s", symbol, exc)
            return []

    # ── quote ─────────────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Optional[float]:
        try:
            t = self._ticker_factory(symbol.upper())
            price = float(t.fast_info["last_price"]) if hasattr(t, "fast_info") else None
            if price and np.isfinite(price) and price > 0:
                return price
        except Exception as exc:
            logger.warning("yfinance get_quote(%s) failed: %s", symbol, exc)
        return None

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
        (yfinance cannot serve historical chains).
        """
        symbol = symbol.upper()
        start = time.perf_counter()

        if date:
            # yfinance has no historical-chain endpoint. Return empty rather than
            # silently substituting today's chain for a past date.
            return pd.DataFrame()

        try:
            t = self._ticker_factory(symbol)
            all_exps = list(t.options or [])
            if not all_exps:
                return self._empty_chain_telemetry(symbol, start, "no_expirations")

            if expiration and str(expiration).lower() != "all":
                target_exps = [str(expiration)] if str(expiration) in all_exps else []
            elif str(expiration).lower() == "all":
                target_exps = all_exps[:12]  # bound credit-free but heavy fetches
            else:
                target_exps = all_exps[:1]
            if not target_exps:
                return self._empty_chain_telemetry(symbol, start, "expiration_not_listed")

            spot = self.get_quote(symbol)
            frames: List[pd.DataFrame] = []
            for exp in target_exps:
                frames.append(self._one_expiry_frame(t, symbol, exp, spot, side, strike_limit, range_filter))
            # Guard on the FILTERED list, not `frames` — when every per-expiry
            # frame is empty, `frames` is truthy but the comprehension yields []
            # and pd.concat([]) raises (mis-recording a legitimately-empty chain
            # as a fetch error).
            nonempty = [f for f in frames if not f.empty]
            frame = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
        except Exception as exc:
            logger.warning("yfinance get_option_chain(%s) failed: %s", symbol, exc)
            return self._empty_chain_telemetry(symbol, start, "fetch_error")

        if frame.empty:
            return self._empty_chain_telemetry(symbol, start, "empty_after_normalize")

        # Attach surface-quality diagnostics exactly like MarketDataClient, so the
        # M5 evidence-quality gate sees a real (possibly degraded) surface status.
        underlying_price = None
        vals = pd.to_numeric(frame.get("underlyingPrice"), errors="coerce").dropna()
        if not vals.empty:
            underlying_price = float(vals.iloc[0])
        frame.attrs["surface_quality"] = diagnose_option_surface_quality(
            frame, underlying_price=underlying_price
        ).to_dict()

        record_provider_telemetry(
            provider_name=self.provider_name,
            endpoint_type="options_chain_quality",
            symbol=symbol,
            success=True,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
        return frame

    def _one_expiry_frame(self, t, symbol, exp, spot, side, strike_limit, range_filter) -> pd.DataFrame:
        chain = t.option_chain(exp)
        parts: List[pd.DataFrame] = []
        wanted = None if not side else str(side).lower()
        for leg_side, legs in (("call", chain.calls), ("put", chain.puts)):
            if wanted in ("call", "put") and leg_side != wanted:
                continue
            parts.append(self._normalize_legs(legs, leg_side, symbol, exp, spot))
        if not parts:
            return pd.DataFrame()
        nonempty = [p for p in parts if not p.empty]
        df = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
        if df.empty:
            return df
        if range_filter and spot:
            df = self._apply_range_filter(df, range_filter, float(spot))
        # Cap to the N STRIKES nearest spot (MDApp's strikeLimit is per-strike,
        # not per-row), keeping both legs of each kept strike. This avoids
        # doubling the strike count when a single side is requested and never
        # splits a call/put pair at the cutoff.
        if strike_limit and spot and not df.empty:
            strikes = df["strike"].dropna().drop_duplicates()
            if len(strikes) > strike_limit:
                keep = set(
                    strikes.to_frame()
                    .assign(_d=(strikes - float(spot)).abs().values)
                    .nsmallest(strike_limit, "_d")["strike"]
                )
                df = df[df["strike"].isin(keep)]
        return df

    def _normalize_legs(self, legs: pd.DataFrame, side: str, symbol: str, exp: str, spot) -> pd.DataFrame:
        if legs is None or legs.empty:
            return pd.DataFrame()
        n = len(legs)
        exp_date = str(exp)
        try:
            dte = (_dt.date.fromisoformat(exp_date) - _dt.date.today()).days
        except Exception:
            dte = np.nan
        bid = pd.to_numeric(legs.get("bid"), errors="coerce")
        ask = pd.to_numeric(legs.get("ask"), errors="coerce")
        # Canonical mid rule (utils.quotes.safe_mid_series): NaN for a zero/absent
        # bid or crossed quote, never a fabricated (0 + ask)/2.
        mid = safe_mid_series(bid, ask)
        out = pd.DataFrame({
            "optionSymbol": legs.get("contractSymbol", pd.Series([None] * n)).astype(object),
            "underlying": [symbol] * n,
            "side": [side] * n,
            "strike": pd.to_numeric(legs.get("strike"), errors="coerce"),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "lastPrice": pd.to_numeric(legs.get("lastPrice"), errors="coerce"),
            "volume": pd.to_numeric(legs.get("volume"), errors="coerce"),
            "openInterest": pd.to_numeric(legs.get("openInterest"), errors="coerce"),
            "impliedVolatility": pd.to_numeric(legs.get("impliedVolatility"), errors="coerce"),
            # yfinance exposes no greeks — emit NaN (honest, not zero).
            "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan,
            "dte": dte,
            "underlyingPrice": float(spot) if spot else np.nan,
            "inTheMoney": legs.get("inTheMoney", pd.Series([None] * n)),
            "intrinsicValue": np.nan,
            "extrinsicValue": np.nan,
            "expiration_date": [exp_date] * n,
        })
        return out[_CHAIN_COLUMNS]

    @staticmethod
    def _apply_range_filter(df: pd.DataFrame, range_filter: str, spot: float) -> pd.DataFrame:
        rf = str(range_filter).lower()
        if rf == "itm":
            return df[((df["side"] == "call") & (df["strike"] <= spot)) | ((df["side"] == "put") & (df["strike"] >= spot))]
        if rf == "otm":
            return df[((df["side"] == "call") & (df["strike"] > spot)) | ((df["side"] == "put") & (df["strike"] < spot))]
        return df

    def _empty_chain_telemetry(self, symbol: str, start: float, note: str) -> pd.DataFrame:
        record_provider_telemetry(
            provider_name=self.provider_name,
            endpoint_type="options_chain_quality",
            symbol=symbol,
            success=False,
            error_category="empty_response",
            response_quality_note=note,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
        return pd.DataFrame()

    # ── earnings ──────────────────────────────────────────────────────────────

    def get_earnings(self, symbol: str, countback: int = 24) -> pd.DataFrame:
        """Earnings history normalized to the MarketDataClient schema.

        reportTime (BMO/AMC) for HISTORICAL rows is derived from the yfinance
        index timestamp's time-of-day (16:00 ET → AMC, 07:00 ET → BMO) via the
        canonical ``normalize_release_timing`` — the same inference the
        screener's event collector already uses. Rows Yahoo stamps at midnight
        (date-only) get reportTime=None → "unknown" downstream, and the shared
        earnings-move profile now DROPS unknown-timing events from measurement
        rather than guessing a window (DD-1). Residual risk, accepted: a
        nonzero-but-wrong placeholder stamp (e.g. 10:00 on a true AMC report)
        classifies confidently wrong and no downstream backstop can catch it.
        The upcoming event keeps its .info-derived timing, which takes
        precedence over the index stamp when present.
        """
        symbol = symbol.upper()
        try:
            t = self._ticker_factory(symbol)
            raw = t.get_earnings_dates(limit=int(countback))
        except Exception as exc:
            logger.warning("yfinance get_earnings(%s) failed: %s", symbol, exc)
            return pd.DataFrame()

        if raw is None or getattr(raw, "empty", True):
            return pd.DataFrame()

        upcoming_time = self._upcoming_report_time(t)
        today = _dt.date.today()
        rows: List[Dict[str, Any]] = []
        for idx, row in raw.iterrows():
            # DD-1 (audit finding 1): derive the calendar date from the SAME
            # America/New_York-normalized timestamp the timing derivation uses,
            # not the raw index. If yfinance ever returns a UTC-tz index, a
            # late-AMC stamp (e.g. 02:00 UTC = 21:00 ET prior day... or the
            # reverse) would otherwise put report_date one session off from the
            # AMC/BMO classification, and the two must agree.
            report_date = _ny_normalized_ts(idx)
            report_date = report_date.date() if report_date is not None else None
            estimated = _to_float(row.get("EPS Estimate"))
            reported = _to_float(row.get("Reported EPS"))
            surprise_pct = _to_float(row.get("Surprise(%)"))
            surprise = (reported - estimated) if (reported is not None and estimated is not None) else None
            report_time = _report_time_from_index_ts(idx)
            if report_date is not None and report_date >= today and upcoming_time is not None:
                report_time = upcoming_time
            rows.append({
                "symbol": symbol,
                "fiscalYear": np.nan,      # yfinance does not provide fiscal labels
                "fiscalQuarter": np.nan,
                "reportTime": report_time,
                "currency": None,
                "reportedEPS": reported,
                "estimatedEPS": estimated,
                "surpriseEPS": surprise,
                "surpriseEPSpct": surprise_pct,
                "event_date": report_date,
                "report_date": report_date,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # A missing historical reportTime is a genuine None (→ "unknown"
        # downstream), NOT a float NaN. DataFrame construction can coerce None
        # to NaN in an object column on some pandas versions, so normalize the
        # column contract explicitly and version-independently. (Both None and
        # NaN normalize to "unknown", so this is a contract/stability fix, not
        # a classification change.)
        df["reportTime"] = df["reportTime"].astype(object).where(df["reportTime"].notna(), None)
        df.sort_values("report_date", ascending=False, inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _upcoming_report_time(t) -> Optional[str]:
        """Map the upcoming earnings timestamp to BMO/AMC via .info, best-effort."""
        try:
            info = t.info or {}
            ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
            if not ts:
                return None
            import pytz
            ny = pytz.timezone("America/New_York")
            dt_ny = _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).astimezone(ny)
            if dt_ny.hour >= 16:
                return "after market close"
            if dt_ny.hour < 12:
                return "before market open"
            return "during market hours"
        except Exception:
            return None


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _ny_normalized_ts(idx: Any) -> Optional[pd.Timestamp]:
    """Normalize a yfinance index timestamp to America/New_York wall-clock.

    A tz-aware stamp is converted to NY then made naive; a naive stamp is
    assumed to already be exchange wall-clock (yfinance's behavior after
    tz_localize(None)). Both the calendar date and the BMO/AMC classification
    read off THIS value so they can never disagree on which session an event
    belongs to (audit finding 1). Returns None for non-timestamps.
    """
    if not isinstance(idx, pd.Timestamp):
        return None
    try:
        return idx.tz_convert("America/New_York").tz_localize(None) if idx.tzinfo is not None else idx
    except Exception:
        return None


def _report_time_from_index_ts(idx: Any) -> Optional[str]:
    """Derive BMO/AMC from a yfinance earnings-dates index timestamp.

    The timezone is pinned via :func:`_ny_normalized_ts` before the wall-clock
    hour is read, so classification does not depend on whatever tz yfinance
    happens to return (a UTC-shaped 16:00-ET stamp would otherwise land near
    20:00 and misclassify).

    Midnight (date-only) stamps carry no timing signal → None, which
    downstream maps to "unknown" and the earnings-move profile excludes from
    measurement (DD-1).
    """
    ts = _ny_normalized_ts(idx)
    if ts is None:
        return None
    if not any((ts.hour, ts.minute, ts.second, ts.microsecond)):
        return None
    timing = _normalize_release_timing_shared(ts)
    return timing if timing != "unknown" else None
