"""Ordinary daily variance during the hold: forecast, and compared with what the stock did.

A T-1 straddle bought at entry and sold before the event gains when the stock
moves more on ordinary days than the base variance the options price, and loses
when it moves less. So, per entry, this module asks how good each forecast of
that variance is, in variance per NYSE session:

* ``implied``: the market's base variance per session, b, from the
  two-expiry fit (services/event_variance.py).
* ``har_rs``: what production uses today, the snapshot's HAR forecast on
  Rogers-Satchell daily variance. Rogers-Satchell is open-to-close only, so it
  leaves out the overnight gap; an option is paid on close-to-close moves.
* ``har_cc``: the same HAR regression on close-to-close log returns, with the
  sessions earlier earnings moved excluded (the snapshot's own rule).
* ``calibrated``: ``har_cc`` times a walk-forward scale learned from earlier
  holds only (``DailyVarianceCalibrator``): the pooled ratio of realized to
  forecast variance, then group and symbol factors shrunk toward 1.

The target is the realized close-to-close variance over the hold: the mean
squared log return of the sessions after entry through the exit. The hold ends
before the event, so the event is never in it.

A few sessions' squared returns are a very noisy, skewed estimate of the true
variance, so nothing here works in logs: by Jensen's inequality the mean of
log(realized / forecast) is negative even for the true variance, so a log
loss rewards forecasts that are too low, and a log-ratio calibration learns
that bias. The losses are the two that rank forecasts correctly against a
noisy unbiased target (Patton 2011): QLIKE (realized / forecast -
log(realized / forecast) - 1) and squared error in levels. The calibration is
a ratio of sums.
"""
from __future__ import annotations

import math
from datetime import date
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from services.earnings_vol_snapshot import _event_contaminated_session_dates, _normalize_price_frame
from services.realized_vol import har_rv_forecast

SESSIONS_PER_YEAR = 252
FORECASTS = ("implied", "har_rs", "har_cc", "calibrated")


def close_to_close_returns(price_data: Any) -> pd.Series:
    """Daily log returns of the close, indexed by session date."""
    frame, _ = _normalize_price_frame(price_data)
    if frame.empty or "Close" not in frame.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    close = close[close > 0].dropna()
    return np.log(close / close.shift(1)).dropna()


def realized_hold_variance(price_data: Any, entry: date, exit_day: date,
                           expected_sessions: int) -> Tuple[Optional[float], Optional[str]]:
    """Mean squared close-to-close log return over the sessions after ``entry`` through ``exit_day``.

    Every session of the hold must have a close (``expected_sessions`` of
    them, counted on the NYSE calendar); a gap is not averaged over.
    """
    returns = close_to_close_returns(price_data)
    if returns.empty:
        return None, "no_closes"
    held = returns[(returns.index > pd.Timestamp(entry)) & (returns.index <= pd.Timestamp(exit_day))]
    if expected_sessions <= 0:
        return None, "no_sessions_held"
    if len(held) != expected_sessions:
        return None, "hold_closes_missing"
    value = float(np.mean(held.to_numpy() ** 2))
    if value <= 0:
        return None, "zero_realized_variance"
    return value, None


def har_close_to_close(price_data: Any, past_events: Sequence[Mapping[str, Any]] | None) -> Optional[float]:
    """HAR forecast of next-session close-to-close variance, earnings sessions excluded.

    The squared returns go through the same Corsi regression as production
    (services.realized_vol.har_rv_forecast, which takes annualised vols).
    """
    frame, _ = _normalize_price_frame(price_data)
    returns = close_to_close_returns(frame)
    if returns.empty:
        return None
    excluded = _event_contaminated_session_dates(frame, list(past_events or []))
    if excluded:
        returns = returns[~returns.index.normalize().isin(list(excluded))]
    # har_rv_forecast returns None below its own minimum history (HAR_MIN_OBS).
    vol = har_rv_forecast(np.sqrt(returns ** 2 * SESSIONS_PER_YEAR))
    return None if vol is None else float(vol) ** 2 / SESSIONS_PER_YEAR


def per_session(annual_vol: Optional[float]) -> Optional[float]:
    if annual_vol is None:
        return None
    try:
        value = float(annual_vol)
    except (TypeError, ValueError):
        return None
    return value * value / SESSIONS_PER_YEAR if math.isfinite(value) and value > 0 else None


DEFAULT_SHRINKAGE_EVENTS = 8.0


@dataclass(frozen=True)
class Calibration:
    factor: float
    pooled: float
    group: float
    symbol: float
    n_pooled: int
    n_group_events: int
    n_symbol_events: int


@dataclass
class DailyVarianceCalibrator:
    """Walk-forward scale for a variance forecast: realized / forecast, in levels.

    The pooled scale is the ratio of sums over every known hold. A group's (and
    then a symbol's) factor is the mean, over its events, of each event's
    realized / (scaled forecast) ratio of sums, shrunk toward 1 by
    n / (n + k) with n its number of events. Only holds that exited before the
    forecast date are known.
    """

    shrinkage_events: float = DEFAULT_SHRINKAGE_EVENTS
    group_of: Optional[Callable[[str], Optional[str]]] = None
    observations: List[tuple] = field(default_factory=list)  # (symbol, event_date, exit_date, realized, base)

    def add(self, symbol: str, event_date: date, exit_date: date, realized: float, base: float) -> None:
        if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (realized, base)):
            self.observations.append((symbol, event_date, exit_date, float(realized), float(base)))

    def known_on(self, day: date) -> List[tuple]:
        return [o for o in self.observations if o[2] < day]

    @staticmethod
    def _factor(obs: Sequence[tuple], scale: float, k: float) -> Tuple[float, int]:
        events: Dict[tuple, List[float]] = {}
        for symbol, event_date, _x, realized, base in obs:
            events.setdefault((symbol, event_date), [0.0, 0.0])
            events[(symbol, event_date)][0] += realized
            events[(symbol, event_date)][1] += base * scale
        if not events:
            return 1.0, 0
        raw = float(np.mean([r / b for r, b in events.values()]))
        n = len(events)
        return 1.0 + (raw - 1.0) * n / (n + k), n

    def calibration(self, symbol: str, day: date) -> Calibration:
        known = self.known_on(day)
        if not known:
            return Calibration(1.0, 1.0, 1.0, 1.0, 0, 0, 0)
        pooled = sum(o[3] for o in known) / sum(o[4] for o in known)
        group_factor, n_group = 1.0, 0
        group = self.group_of(symbol) if self.group_of else None
        if group is not None:
            group_factor, n_group = self._factor([o for o in known if self.group_of(o[0]) == group],
                                                 pooled, self.shrinkage_events)
        symbol_factor, n_symbol = self._factor([o for o in known if o[0] == symbol],
                                               pooled * group_factor, self.shrinkage_events)
        return Calibration(pooled * group_factor * symbol_factor, pooled, group_factor, symbol_factor,
                           len(known), n_group, n_symbol)


def _round(value: Optional[float], digits: int = 5) -> Optional[float]:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def qlike(realized: float, forecast: float) -> float:
    ratio = realized / forecast
    return ratio - math.log(ratio) - 1.0


def forecast_quality(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Each forecast against the realized hold variance, on the rows where all are present.

    Comparing on the common rows keeps one forecast's coverage from flattering
    it. ``level_ratio`` is mean(realized) / mean(forecast): 1 is unbiased,
    above 1 the forecast is too low. ``qlike_skill_vs_implied`` and
    ``mse_skill_vs_implied`` are 1 - loss / loss(implied): above 0 beats the
    market's own base variance. ``spread_spearman`` asks whether a forecast
    knows when the market is rich or cheap: the rank correlation of
    forecast / implied with realized / implied.
    """
    keys = [f"daily_forecast_{name}" for name in FORECASTS]
    common = [r for r in rows if r.get("daily_realized_variance") is not None
              and all(r.get(k) is not None and r.get(k) > 0 for k in keys)]
    out: Dict[str, Any] = {"n": len(common), "rows_with_realized": sum(
        1 for r in rows if r.get("daily_realized_variance") is not None)}
    if not common:
        return out
    realized = np.asarray([r["daily_realized_variance"] for r in common], dtype=float)
    implied = np.asarray([r["daily_forecast_implied"] for r in common], dtype=float)
    qlike_implied = float(np.mean([qlike(a, b) for a, b in zip(realized, implied)]))
    mse_implied = float(np.mean((realized - implied) ** 2))
    from services.pricing_harness import spearman  # local: the harness imports this module

    out["mean_realized_vol_annual"] = _round(float(np.sqrt(realized.mean() * SESSIONS_PER_YEAR)))
    per: Dict[str, Any] = {}
    for name, key in zip(FORECASTS, keys):
        f = np.asarray([r[key] for r in common], dtype=float)
        loss = float(np.mean([qlike(a, b) for a, b in zip(realized, f)]))
        mse = float(np.mean((realized - f) ** 2))
        block = {
            "mean_forecast_vol_annual": _round(float(np.sqrt(f.mean() * SESSIONS_PER_YEAR))),
            "level_ratio": _round(float(realized.mean() / f.mean())),
            "qlike": _round(loss),
            "rmse_variance": float(f"{math.sqrt(mse):.6g}"),
        }
        if name != "implied":
            block["qlike_skill_vs_implied"] = _round(1.0 - loss / qlike_implied) if qlike_implied > 0 else None
            block["mse_skill_vs_implied"] = _round(1.0 - mse / mse_implied) if mse_implied > 0 else None
            if len(common) >= 3:
                block["spread_spearman"] = _round(spearman(list(f / implied), list(realized / implied)))
        per[name] = block
    out["forecasts"] = per
    return out


__all__ = [
    "Calibration", "DailyVarianceCalibrator", "FORECASTS", "close_to_close_returns", "forecast_quality", "har_close_to_close", "per_session", "qlike",
    "realized_hold_variance",
]
