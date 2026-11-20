"""The market's event variance, measured from the term structure, and a forecast of its change.

Every structure exits the session before earnings, so the earnings move never
reaches the P&L. What a straddle earns between entry and exit is mostly the
change in the event variance the market prices (E), plus ordinary daily moves
against the base variance, less costs. This module handles the first term:

* ``implied_event_variance`` reads E from two expiries that both span the
  event. Each expiry's total implied variance is W = IV^2 x (calendar days /
  365), the convention the archive's IV is quoted in. Writing W = b x T + E,
  with T the NYSE sessions from the as-of date through the expiry and b the
  base variance per session, the two expiries give
      E = (W1 x T2 - W2 x T1) / (T2 - T1),    b = (W2 - W1) / (T2 - T1).
  This needs no realized-vol forecast (the snapshot's split subtracts HAR-RV
  from the front expiry instead, so its E mixes implied and realized), and it
  counts trading sessions, so a long weekend is not read as base variance.
* ``EventVarianceForecaster`` predicts log(E_exit / E_entry) for a holding
  period from earlier events only: the pooled mean for that holding-period
  bucket, plus a group offset and a symbol offset, each shrunk toward zero by
  n / (n + k) with n the number of events behind it. Only observations whose
  exit came before the forecast date are used.
* ``forecast_quality`` judges the forecast against the two obvious baselines:
  no change (log ratio 0) and the pooled bucket mean alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from services.market_calendar import is_trading_day
from services.release_timing import reaction_session

SESSIONS_PER_YEAR = 252
# Events behind an offset at which it gets half its raw weight.
DEFAULT_SHRINKAGE_EVENTS = 8.0
# Holding periods in sessions, pooled separately: E grows into the event, so
# a three-session hold and a ten-session hold are different questions.
HOLD_BUCKETS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10_000))


def sessions_between(start: date, end: date) -> int:
    """NYSE sessions d with start < d <= end."""
    count, day = 0, start + timedelta(days=1)
    while day <= end:
        if is_trading_day(day):
            count += 1
        day += timedelta(days=1)
    return count


def _finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _interpolated_iv(side: pd.DataFrame, spot: float) -> Optional[float]:
    frame = pd.DataFrame({
        "strike": pd.to_numeric(side["strike"], errors="coerce"),
        "iv": pd.to_numeric(side["iv"], errors="coerce"),
    }).dropna()
    frame = frame[frame["iv"] > 0].groupby("strike", as_index=False)["iv"].mean().sort_values("strike")
    if frame.empty:
        return None
    below = frame[frame["strike"] <= spot]
    above = frame[frame["strike"] >= spot]
    if below.empty or above.empty:
        return None  # never extrapolate the smile
    lo, hi = below.iloc[-1], above.iloc[0]
    if hi["strike"] == lo["strike"]:
        return float(lo["iv"])
    weight = (spot - lo["strike"]) / (hi["strike"] - lo["strike"])
    return float(lo["iv"] + weight * (hi["iv"] - lo["iv"]))


def atm_iv(chain: pd.DataFrame, expiry: date, spot: float) -> Optional[float]:
    """IV at the spot strike for one expiry: linear in strike, calls and puts averaged."""
    if chain is None or chain.empty or "iv" not in chain.columns:
        return None
    expiries = pd.to_datetime(chain["expiry"], errors="coerce").dt.date
    rows = chain[expiries == expiry]
    sides = rows["call_put"].astype(str).str.upper()
    values = [v for v in (_interpolated_iv(rows[sides == s], spot) for s in ("C", "P")) if v is not None]
    return float(np.mean(values)) if values else None


@dataclass(frozen=True)
class ImpliedEventVariance:
    as_of: date
    expiries: tuple = ()
    sessions: tuple = ()
    total_variances: tuple = ()
    event_variance: Optional[float] = None
    base_variance_per_session: Optional[float] = None
    reason: Optional[str] = None

    @property
    def event_move_pct(self) -> Optional[float]:
        """The 1-sigma event move in percent, when E is positive."""
        if self.event_variance is None or self.event_variance <= 0:
            return None
        return math.sqrt(self.event_variance) * 100.0

    @property
    def usable(self) -> bool:
        return self.reason is None and self.event_variance is not None and self.event_variance > 0


def chain_spot(chain: pd.DataFrame) -> Optional[float]:
    if chain is None or chain.empty or "underlying_price" not in chain.columns:
        return None
    spot = _finite(pd.to_numeric(chain["underlying_price"], errors="coerce").median())
    return spot if spot is not None and spot > 0 else None


def implied_event_variance(
    chain: pd.DataFrame,
    as_of: date,
    event_date: date,
    release_timing: Any,
    *,
    expiries: Optional[Sequence[date]] = None,
) -> ImpliedEventVariance:
    """E from the two nearest expiries spanning the event (or the ``expiries`` given).

    An expiry spans the event when it is on or after the first session the
    reaction can print on (unknown timing is treated as after the close).
    Passing the entry's ``expiries`` at exit measures both ends on the same
    pair, so the change is not partly a change of expiries.
    """
    spot = chain_spot(chain)
    if spot is None:
        return ImpliedEventVariance(as_of=as_of, reason="no_spot")
    if expiries is None:
        first = reaction_session(event_date, release_timing)
        listed = sorted({d for d in pd.to_datetime(chain["expiry"], errors="coerce").dt.date.dropna() if d >= first})
        expiries = listed[:2]
    pair = tuple(sorted(expiries))
    if len(pair) < 2:
        return ImpliedEventVariance(as_of=as_of, expiries=pair, reason="fewer_than_two_spanning_expiries")
    sessions = tuple(sessions_between(as_of, e) for e in pair)
    if sessions[0] <= 0 or sessions[1] <= sessions[0]:
        return ImpliedEventVariance(as_of=as_of, expiries=pair, sessions=sessions, reason="expiries_not_separable")
    totals = []
    for expiry in pair:
        iv = atm_iv(chain, expiry, spot)
        if iv is None:
            return ImpliedEventVariance(as_of=as_of, expiries=pair, sessions=sessions,
                                        reason=f"atm_iv_unavailable:{expiry.isoformat()}")
        totals.append(iv * iv * (expiry - as_of).days / 365.0)
    (w1, w2), (t1, t2) = totals, sessions
    event = (w1 * t2 - w2 * t1) / (t2 - t1)
    base = (w2 - w1) / (t2 - t1)
    return ImpliedEventVariance(
        as_of=as_of, expiries=pair, sessions=sessions, total_variances=tuple(totals),
        event_variance=event, base_variance_per_session=base,
        reason=None if event > 0 else "non_positive_event_variance",
    )


# ── the forecast ─────────────────────────────────────────────────────────────

def hold_bucket(sessions_held: int) -> int:
    for i, (lo, hi) in enumerate(HOLD_BUCKETS):
        if lo <= sessions_held <= hi:
            return i
    raise ValueError(f"sessions_held must be >= 1, got {sessions_held}")


@dataclass(frozen=True)
class VarianceObservation:
    symbol: str
    event_date: date
    entry_date: date
    exit_date: date
    sessions_held: int
    log_ratio: float


@dataclass(frozen=True)
class VarianceForecast:
    log_ratio: float
    pooled: float
    group_offset: float
    symbol_offset: float
    n_pooled: int
    n_group_events: int
    n_symbol_events: int


def _shrink(mean: float, n: int, k: float) -> float:
    return mean * n / (n + k) if n > 0 else 0.0


@dataclass
class EventVarianceForecaster:
    shrinkage_events: float = DEFAULT_SHRINKAGE_EVENTS
    group_of: Optional[Callable[[str], Optional[str]]] = None
    observations: List[VarianceObservation] = field(default_factory=list)

    def add(self, observation: VarianceObservation) -> None:
        if math.isfinite(observation.log_ratio):
            self.observations.append(observation)

    def known_on(self, day: date) -> List[VarianceObservation]:
        """Observations whose exit came before ``day``: all a forecast may use."""
        return [o for o in self.observations if o.exit_date < day]

    def predict(self, symbol: str, sessions_held: int, day: date) -> VarianceForecast:
        known = self.known_on(day)
        if not known:
            return VarianceForecast(0.0, 0.0, 0.0, 0.0, 0, 0, 0)
        overall = float(np.mean([o.log_ratio for o in known]))
        bucket_means: Dict[int, float] = {}
        for b in range(len(HOLD_BUCKETS)):
            in_bucket = [o.log_ratio for o in known if hold_bucket(o.sessions_held) == b]
            bucket_means[b] = float(np.mean(in_bucket)) if in_bucket else overall

        def residual(o: VarianceObservation) -> float:
            return o.log_ratio - bucket_means[hold_bucket(o.sessions_held)]

        def per_event_mean(obs: Iterable[VarianceObservation], adjust: float = 0.0) -> tuple[float, int]:
            events: Dict[tuple, List[float]] = {}
            for o in obs:
                events.setdefault((o.symbol, o.event_date), []).append(residual(o) - adjust)
            if not events:
                return 0.0, 0
            return float(np.mean([np.mean(v) for v in events.values()])), len(events)

        group = self.group_of(symbol) if self.group_of else None
        group_offset, n_group = 0.0, 0
        if group is not None:
            mean, n_group = per_event_mean(o for o in known if self.group_of(o.symbol) == group)
            group_offset = _shrink(mean, n_group, self.shrinkage_events)
        mean, n_symbol = per_event_mean((o for o in known if o.symbol == symbol), adjust=group_offset)
        symbol_offset = _shrink(mean, n_symbol, self.shrinkage_events)
        pooled = bucket_means[hold_bucket(sessions_held)]
        return VarianceForecast(
            log_ratio=pooled + group_offset + symbol_offset, pooled=pooled, group_offset=group_offset,
            symbol_offset=symbol_offset, n_pooled=len(known), n_group_events=n_group, n_symbol_events=n_symbol,
        )


# ── how good is it ───────────────────────────────────────────────────────────

def _round(value: Optional[float], digits: int = 5) -> Optional[float]:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def forecast_quality(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The forecast against no change and the pooled mean, on rows with a realized ratio.

    ``r2_vs_*`` is 1 - SSE(forecast) / SSE(baseline): above 0 beats the
    baseline out of sample, below 0 is worse than it.
    """
    usable = [r for r in rows if r.get("realized_log_ratio") is not None and r.get("forecast_log_ratio") is not None]
    if not usable:
        return {"n": 0}
    real = np.asarray([r["realized_log_ratio"] for r in usable], dtype=float)
    pred = np.asarray([r["forecast_log_ratio"] for r in usable], dtype=float)
    pooled = np.asarray([r["pooled_log_ratio"] for r in usable], dtype=float)
    sse = float(np.sum((real - pred) ** 2))
    sse_zero = float(np.sum(real ** 2))
    sse_pooled = float(np.sum((real - pooled) ** 2))
    from services.pricing_harness import spearman  # local: the harness imports this module

    by_bucket = {}
    for b, (lo, hi) in enumerate(HOLD_BUCKETS):
        mask = np.asarray([hold_bucket(int(r["sessions_held"])) == b for r in usable])
        if mask.any():
            label = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
            by_bucket[label] = {"n": int(mask.sum()), "mean_realized": _round(float(real[mask].mean())),
                                "mean_forecast": _round(float(pred[mask].mean()))}
    return {
        "n": len(usable),
        "events": len({(r["symbol"], r["event_date"]) for r in usable}),
        "mean_realized_log_ratio": _round(float(real.mean())),
        "rmse": _round(math.sqrt(sse / len(usable))),
        "rmse_no_change": _round(math.sqrt(sse_zero / len(usable))),
        "rmse_pooled": _round(math.sqrt(sse_pooled / len(usable))),
        "r2_vs_no_change": _round(1.0 - sse / sse_zero) if sse_zero > 0 else None,
        "r2_vs_pooled": _round(1.0 - sse / sse_pooled) if sse_pooled > 0 else None,
        "spearman": _round(spearman(list(pred), list(real))) if len(usable) >= 3 else None,
        "by_hold_sessions": by_bucket,
    }


__all__ = [
    "DEFAULT_SHRINKAGE_EVENTS", "EventVarianceForecaster", "HOLD_BUCKETS", "ImpliedEventVariance",
    "VarianceForecast", "VarianceObservation", "atm_iv", "chain_spot", "forecast_quality", "hold_bucket",
    "implied_event_variance", "sessions_between",
]
