"""
Realized-volatility kernels — single source of truth.

Centralizes the OHLC-based realized-vol estimators used by both the canonical
volatility snapshot (services.earnings_vol_snapshot) and the legacy edge-engine
diagnostics path (web.api.edge_engine), so improvements and bug fixes land in
one place instead of two.

Functions
---------
yang_zhang_rv30
    Yang-Zhang (2000) OHLC realized volatility over a rolling window.
rs_daily_vol_series
    Per-day Rogers-Satchell annualised vol series — input to HAR.
har_rv_forecast
    Corsi (2009) Heterogeneous Autoregressive Realized Variance forecast.
rs_trailing_mean_forecast
    Trailing-mean fallback used when the series is too short for stable HAR OLS.

Conventions
-----------
- All inputs are OHLC DataFrames indexed by date with columns Open, High, Low,
  Close. Indexes are expected to be sorted ascending.
- Returns are annualised (× sqrt(252)) so the output is comparable to implied
  volatility quoted by option pricers.
- ``excluded_sessions`` (where supported) is a set of normalized
  ``pd.Timestamp`` values whose corresponding rows should be removed before
  the estimator runs. Used to keep earnings-day jumps out of the non-event
  baseline. When ``None`` (default), no rows are excluded — preserving the
  legacy edge-engine behavior.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# Minimum RS-daily observations required for stable HAR OLS (4-param regression).
# Below this, the OLS has too few degrees of freedom; use the simpler RS fallback instead.
HAR_MIN_OBS: int = 100
# Trailing window for the RS mean fallback used when n < HAR_MIN_OBS.
RS_FALLBACK_WINDOW: int = 30


def yang_zhang_rv30(
    hist: pd.DataFrame,
    window: int = 30,
    excluded_sessions: Optional[set[pd.Timestamp]] = None,
) -> float:
    """
    Yang-Zhang (2000) OHLC realized volatility over *window* trading days.

        σ²_YZ = σ²_o + k·σ²_c + (1-k)·σ²_RS

    where
      σ²_o  = sample variance of overnight log-returns  log(Open_t / Close_{t-1})
      σ²_c  = sample variance of open-to-close returns  log(Close_t / Open_t)
      σ²_RS = mean Rogers-Satchell per-day variance (direction-free intraday)
      k     = 0.34 / (1.34 + (n+1)/(n-1))  — paper-optimal weight

    Returns annualised volatility (× √252), or ``np.nan`` if data is insufficient.

    When ``excluded_sessions`` is provided, those sessions are masked out of the
    overnight/intraday/RS series before the variance is computed (the
    earnings-day jump exclusion). ``None`` preserves the unmasked behavior of
    the legacy edge-engine implementation.
    """
    needed = ("Open", "High", "Low", "Close")
    if not all(col in hist.columns for col in needed):
        return np.nan
    df = hist[list(needed)].copy().dropna()
    if len(df) < max(window + 1, 6):
        return np.nan

    df = df.sort_index()
    overnight = np.log(df["Open"] / df["Close"].shift(1))
    intraday = np.log(df["Close"] / df["Open"])
    high = np.log(df["High"] / df["Open"])
    low = np.log(df["Low"] / df["Open"])

    if excluded_sessions:
        normalized_index = pd.Index([pd.Timestamp(ts).normalize() for ts in df.index])
        excluded_mask = normalized_index.isin(list(excluded_sessions))
        overnight = overnight.mask(excluded_mask)
        intraday = intraday.mask(excluded_mask)
        high = high.mask(excluded_mask)
        low = low.mask(excluded_mask)

    idx = overnight.index.intersection(intraday.index).intersection(high.index).intersection(low.index)
    overnight = overnight[idx].dropna().tail(window)
    intraday = intraday[idx].dropna().tail(window)
    high = high[idx].dropna().tail(window)
    low = low[idx].dropna().tail(window)
    idx = overnight.index.intersection(intraday.index).intersection(high.index).intersection(low.index)
    overnight = overnight[idx]
    intraday = intraday[idx]
    high = high[idx]
    low = low[idx]
    n = len(overnight)
    if n < 5:
        return np.nan

    rs = high * (high - intraday) + low * (low - intraday)
    k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
    var_o = float(((overnight - overnight.mean()) ** 2).sum() / max(n - 1, 1))
    var_c = float(((intraday - intraday.mean()) ** 2).sum() / max(n - 1, 1))
    var_rs = float(rs.mean())
    yz_var = var_o + k * var_c + (1.0 - k) * var_rs
    return float(np.sqrt(max(yz_var, 1e-10) * 252.0))


def rs_daily_vol_series(
    hist: pd.DataFrame,
    excluded_sessions: Optional[set[pd.Timestamp]] = None,
) -> pd.Series:
    """
    Per-day Rogers-Satchell annualised volatility series — used as input to HAR.

        RS_t = log(H_t/O_t)·log(H_t/C_t) + log(L_t/O_t)·log(L_t/C_t)

    Each RS_t is an unbiased, direction-free estimate of daily variance that
    requires no window.  Annualising: vol_t = √(252 · RS_t).

    Negative RS variance values (which can occur for narrow-range sessions) are
    clipped to zero before the sqrt to keep the daily series in vol units.

    When ``excluded_sessions`` is provided, those rows are filtered out of
    ``hist`` before the per-day computation.
    """
    needed = ("Open", "High", "Low", "Close")
    if not all(col in hist.columns for col in needed):
        return pd.Series(dtype=float)
    df = hist[list(needed)].copy().dropna()
    if excluded_sessions:
        normalized_index = pd.Index([pd.Timestamp(ts).normalize() for ts in df.index])
        df = df[~normalized_index.isin(list(excluded_sessions))]
    high = np.log(df["High"] / df["Open"])
    low = np.log(df["Low"] / df["Open"])
    close = np.log(df["Close"] / df["Open"])
    rs_var = (high * (high - close) + low * (low - close)).clip(lower=0.0)
    return np.sqrt(rs_var * 252.0).dropna()


def har_rv_forecast(rv_daily: pd.Series, horizon: int = 1) -> Optional[float]:
    """
    Corsi (2009) Heterogeneous Autoregressive Realized Variance.

        RV_{t+h} = β₀ + β_d·RV_t + β_w·RV^(w)_t + β_m·RV^(m)_t + ε

    RV^(w)_t = 5-day average of daily RV ending at t  (weekly persistence)
    RV^(m)_t = 22-day average of daily RV ending at t (monthly persistence)

    Fitted by OLS on the in-sample daily series.  Returns a 1-day-ahead
    forecast (annualised vol), floored at 1 bp.  Returns ``None`` when fewer
    than ``HAR_MIN_OBS`` observations are available, when too few rows are
    formable into the (RV, RV_w, RV_m) regression matrix (< 10), or when OLS
    raises.

    Note: input is the annualised vol series from :func:`rs_daily_vol_series`;
    we square to variance for HAR fitting (Corsi operates on RV, not σ) and
    convert the forecast back to vol via sqrt.  This avoids the
    Jensen's-inequality bias that would arise from regressing directly on σ.
    """
    n = len(rv_daily)
    if n < HAR_MIN_OBS:
        return None
    rv = (rv_daily.values.astype(float)) ** 2
    y: List[float] = []
    x: List[List[float]] = []
    for idx in range(22, n - horizon):
        rv_d = rv[idx]
        rv_w = rv[max(idx - 4, 0): idx + 1].mean()
        rv_m = rv[max(idx - 21, 0): idx + 1].mean()
        y.append(rv[idx + horizon])
        x.append([1.0, rv_d, rv_w, rv_m])
    if len(y) < 10:
        return None
    try:
        beta, _, _, _ = np.linalg.lstsq(np.array(x, dtype=float), np.array(y, dtype=float), rcond=None)
    except Exception:
        return None
    last_rv_d = rv[-1]
    last_rv_w = rv[-5:].mean() if n >= 5 else rv[-1]
    last_rv_m = rv[-22:].mean() if n >= 22 else rv[-1]
    forecast_var = float(beta[0] + beta[1] * last_rv_d + beta[2] * last_rv_w + beta[3] * last_rv_m)
    return max(np.sqrt(max(forecast_var, 0.0)), 1e-4)


def rs_trailing_mean_forecast(
    rv_daily: pd.Series,
    window: int = RS_FALLBACK_WINDOW,
) -> Optional[float]:
    """
    Trailing-mean fallback used when n < HAR_MIN_OBS and HAR is unavailable.

    Returns the simple mean of the trailing ``window`` Rogers-Satchell annualised
    vol values.  No regression; numerically stable at low sample sizes.  Serves
    as a conservative baseline so event decomposition does not lose a non-event
    vol estimate entirely when price history is short.

    The calling site records ``null_reasons["rv_har_estimator"]`` to indicate
    that this path was taken, so consumers can distinguish HAR from fallback.
    """
    if len(rv_daily) < window:
        return None
    trailing = rv_daily.tail(window).dropna()
    if len(trailing) < 5:
        return None
    mean_val = float(trailing.mean())
    return mean_val if np.isfinite(mean_val) and mean_val > 0 else None


__all__ = [
    "HAR_MIN_OBS",
    "RS_FALLBACK_WINDOW",
    "yang_zhang_rv30",
    "rs_daily_vol_series",
    "har_rv_forecast",
    "rs_trailing_mean_forecast",
]
