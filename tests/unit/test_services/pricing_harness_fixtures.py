"""A small synthetic archive for the pricing-harness tests.

Chains are priced with Black-Scholes on a total variance of BASE_VOL^2 per
252 trading sessions to expiry plus the event variance for expiries after the
event, so a straddle carries the event premium the snapshot expects and the
two-expiry estimate recovers the event variance exactly. The event variance
can grow into the event (``event_growth`` per symbol). Deterministic.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.event_variance import sessions_between
from services.market_calendar import is_trading_day
from services.pricing_harness import EarningsEvent

BASE_VOL = 0.30
EVENT_MOVE = 0.06  # one-day event standard deviation


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t: float, vol: float, kind: str) -> float:
    if t <= 0:
        return max(0.0, spot - strike) if kind == "C" else max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if kind == "C":
        return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def trading_days(start: date, end: date) -> List[date]:
    out, day = [], start
    while day <= end:
        if is_trading_day(day):
            out.append(day)
        day += timedelta(days=1)
    return out


class SyntheticArchive:
    """Implements services.pricing_harness.Archive; records every read."""

    def __init__(self, events: Dict[str, List[EarningsEvent]], *, start: date = date(2023, 9, 1),
                 end: date = date(2024, 6, 28), seed: int = 3, spread_frac: float = 0.03,
                 vol_bump_after: Optional[date] = None, missing_days: Optional[Dict[str, set]] = None,
                 expiry_horizon_days: int = 70, event_growth: Optional[Dict[str, float]] = None):
        self._events = events
        self._days = trading_days(start, end)
        self.spread_frac = spread_frac
        self.vol_bump_after = vol_bump_after
        self.missing_days = missing_days or {}
        self.expiry_horizon_days = expiry_horizon_days
        self.event_growth = event_growth or {}
        self.reads: List[tuple] = []
        self._spot: Dict[str, Dict[date, float]] = {}
        for i, symbol in enumerate(sorted(events)):
            rng = np.random.default_rng(seed + i)
            path = 100.0 * np.exp(np.cumsum(rng.normal(0.0, BASE_VOL / math.sqrt(252), len(self._days))))
            self._spot[symbol] = dict(zip(self._days, path.tolist()))

    def sessions(self, symbol: str) -> List[date]:
        missing = self.missing_days.get(symbol, set())
        return [d for d in self._days if d not in missing]

    def events(self, symbol: str) -> List[EarningsEvent]:
        return list(self._events.get(symbol, []))

    def prices(self, symbol: str, through: date) -> pd.DataFrame:
        self.reads.append(("prices", symbol, through))
        rows = [{"trade_date": pd.Timestamp(d), "open": s, "high": s * 1.01, "low": s * 0.99, "close": s,
                 "volume": 1_000_000} for d, s in self._spot[symbol].items() if d <= through]
        return pd.DataFrame(rows)

    def event_variance(self, symbol: str, event_date: date, day: date) -> float:
        """The event variance priced on ``day``: grows linearly over the last 14 days."""
        growth = self.event_growth.get(symbol, 0.0)
        return EVENT_MOVE ** 2 * (1.0 + growth * max(0, 14 - (event_date - day).days) / 14.0)

    def _vol(self, day: date) -> float:
        if self.vol_bump_after is not None and day > self.vol_bump_after:
            return BASE_VOL * 1.6
        return BASE_VOL

    def chain(self, symbol: str, day: date) -> pd.DataFrame:
        self.reads.append(("chain", symbol, day))
        if day in self.missing_days.get(symbol, set()) or day not in self._spot[symbol]:
            return pd.DataFrame()
        spot = self._spot[symbol][day]
        vol = self._vol(day)
        fridays = [day + timedelta(days=k) for k in range(1, self.expiry_horizon_days) if (day + timedelta(days=k)).weekday() == 4]
        center = round(spot / 2.5) * 2.5
        strikes = [center + 2.5 * k for k in range(-12, 13)]
        rows = []
        for expiry in fridays:
            t = (expiry - day).days / 365.0
            event_var = sum(self.event_variance(symbol, e.event_date, day) for e in self._events.get(symbol, [])
                            if day < e.event_date <= expiry)
            total_var = vol * vol * sessions_between(day, expiry) / 252.0 + event_var
            total_vol = math.sqrt(total_var / t)
            for strike in strikes:
                for kind in ("C", "P"):
                    mid = bs_price(spot, strike, t, total_vol, kind)
                    half = max(0.025, self.spread_frac * mid)
                    bid = round(max(0.0, mid - half), 4)
                    ask = round(mid + half, 4)
                    rows.append({
                        "trade_date": pd.Timestamp(day), "expiry": pd.Timestamp(expiry), "call_put": kind,
                        "strike": float(strike), "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4),
                        "iv": total_vol, "open_interest": 500, "volume": 100, "underlying_price": spot,
                        "spread_pct": (ask - bid) / max(mid, 1e-9) * 100.0,
                        "option_symbol": f"{symbol}{expiry:%y%m%d}{kind}{int(strike * 1000):08d}",
                    })
        return pd.DataFrame(rows)


def quarterly_events(symbol: str, dates: List[date], timing: str = "AMC") -> List[EarningsEvent]:
    return [EarningsEvent(symbol=symbol, event_date=d, release_timing=timing, source="synthetic",
                          source_confidence=0.9) for d in dates]
