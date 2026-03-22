"""
Edge Engine — IV Crush / Calendar Spread Signal
================================================
Data layer:  MarketData.app (primary) + yfinance (fallback / price history)
Scoring:     unchanged from v0.1 — all thresholds and formulas are preserved
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ─── ML crush classifier (optional — loaded once at startup) ──────────────────
# Trained by /api/ml/train.  If absent, inference is skipped gracefully.

_ML_MODEL_DIR = Path.home() / ".options_calculator_pro" / "models"
_crush_clf = None
_crush_scaler = None
_crush_model_loaded: bool = False
_ml_model_lock = threading.Lock()
_rf_rate_lock = threading.Lock()
_rf_rate_cache: Dict[str, Any] = {"ts": 0.0, "rate": None, "source": None}


def _try_load_crush_model() -> None:
    global _crush_clf, _crush_scaler, _crush_model_loaded
    with _ml_model_lock:
        _crush_model_loaded = True  # set first so repeated failures are suppressed
        try:
            import joblib  # type: ignore[import]
            clf_path = _ML_MODEL_DIR / "crush_classifier.pkl"
            scaler_path = _ML_MODEL_DIR / "crush_scaler.pkl"
            if clf_path.exists() and scaler_path.exists():
                _crush_clf = joblib.load(clf_path)
                _crush_scaler = joblib.load(scaler_path)
                logger.info("ML crush classifier loaded from %s", _ML_MODEL_DIR)
            else:
                logger.debug("ML crush classifier not found at %s — skipping", _ML_MODEL_DIR)
        except Exception as exc:
            logger.debug("ML crush classifier load failed: %s", exc)


def reload_crush_model() -> None:
    """Re-read classifier from disk (called after /api/ml/train completes)."""
    global _crush_model_loaded
    with _ml_model_lock:
        _crush_model_loaded = False
    _try_load_crush_model()


_try_load_crush_model()


def _ml_crush_probability(
    near_iv: float,
    near_back_ratio: float,
    iv_rv: float,
) -> Tuple[Optional[float], float]:
    """Return (crush_prob, confidence_multiplier).

    Features must match training order: [near_back_ratio, log_front_iv, iv_rv_approx].
    If the model is unavailable, returns (None, 1.0) so the rules-based score is
    unchanged.

    Multiplier formula: 0.85 + 0.30 * P
      P=0.50 → 1.00x (neutral)   P=0.75 → 1.08x (modest boost)
      P=0.25 → 0.93x (modest haircut)  range: [0.85, 1.15]
    """
    with _ml_model_lock:
        clf, scaler = _crush_clf, _crush_scaler
    if clf is None or scaler is None:
        return None, 1.0
    try:
        nb = float(near_back_ratio) if np.isfinite(float(near_back_ratio)) else 1.10
        lniv = float(np.log(max(float(near_iv), 0.01)))
        ivr = float(iv_rv) if np.isfinite(float(iv_rv)) else 1.10
        feat = np.array([[nb, lniv, ivr]])
        feat_scaled = scaler.transform(feat)
        prob = float(clf.predict_proba(feat_scaled)[0][1])
        mult = float(np.clip(0.85 + 0.30 * prob, 0.85, 1.15))
        return prob, mult
    except Exception as exc:
        logger.debug("ML crush inference error: %s", exc)
        return None, 1.0


# ─── Constants (unchanged) ────────────────────────────────────────────────────

MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL = 8
HARD_NO_TRADE_CONFIDENCE_CAP_PCT = 55.0
TS_SLOPE_TARGET = -0.004
TS_SLOPE_BAND = 0.025
MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE = 18.0
MIN_SHORT_LEG_DTE = 2
MIN_NEAR_BACK_IV_RATIO_FOR_EVENT = 1.02
MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_TRADE = 400.0
MOVE_UNCERTAINTY_Z_SCORE = 1.28


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class EdgeSnapshot:
    symbol: str
    recommendation: str
    confidence_pct: float
    setup_score: float
    metrics: Dict[str, Any]
    rationale: List[str]


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _utc_today_date():
    return datetime.now(timezone.utc).date()


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return float(default)
        parsed = float(value)
        if not np.isfinite(parsed):
            return float(default)
        return parsed
    except (TypeError, ValueError):
        return float(default)


def _get_pricing_risk_free_rate(cache_ttl_seconds: float = 43_200.0) -> Tuple[float, str]:
    """
    Resolve the pricing risk-free rate for BSM-style calculations.

    Preference order:
      1. Explicit env override `OPTIONS_PRICING_RISK_FREE_RATE`
      2. Latest available 13-week T-bill yield via yfinance `^IRX`
      3. Conservative static fallback
    """
    now = time.time()
    with _rf_rate_lock:
        cached_rate = _safe_float(_rf_rate_cache.get("rate"), np.nan)
        if (
            np.isfinite(cached_rate)
            and cached_rate > 0
            and (now - float(_rf_rate_cache.get("ts") or 0.0)) < cache_ttl_seconds
        ):
            return float(cached_rate), str(_rf_rate_cache.get("source") or "cache")

    env_raw = os.getenv("OPTIONS_PRICING_RISK_FREE_RATE", "").strip()
    env_rate = _safe_float(env_raw, np.nan)
    if np.isfinite(env_rate) and 0.0 < env_rate < 0.25:
        with _rf_rate_lock:
            _rf_rate_cache.update({"ts": now, "rate": float(env_rate), "source": "env"})
        return float(env_rate), "env"

    try:
        irx_hist = yf.Ticker("^IRX").history(period="7d", auto_adjust=False)
        irx_close = pd.to_numeric(irx_hist.get("Close"), errors="coerce").dropna()
        if not irx_close.empty:
            irx_rate = float(irx_close.iloc[-1]) / 100.0
            if np.isfinite(irx_rate) and 0.0 < irx_rate < 0.25:
                with _rf_rate_lock:
                    _rf_rate_cache.update({"ts": now, "rate": irx_rate, "source": "yfinance_^IRX"})
                return irx_rate, "yfinance_^IRX"
    except Exception as exc:
        logger.debug("Risk-free rate fetch via ^IRX failed: %s", exc)

    fallback_rate = 0.0525
    with _rf_rate_lock:
        _rf_rate_cache.update({"ts": now, "rate": fallback_rate, "source": "fallback_static"})
    return fallback_rate, "fallback_static"


# ─── ATM option stats (unchanged — works on both yfinance and MDApp DataFrames) ──

def _nearest_atm_option_stats(
    chain_df: pd.DataFrame, current_price: float
) -> Dict[str, Optional[float]]:
    """
    Find the nearest-ATM strike in *chain_df* and return its IV, OI, volume,
    mid price, and bid-ask spread %.

    Works with both yfinance DataFrames and MarketData.app-normalized DataFrames.
    Both share the same column names after normalization:
      strike, impliedVolatility, openInterest, volume, bid, ask, lastPrice
    """
    empty = {"iv": None, "oi": None, "volume": None, "mid": None, "spread_pct": None}
    if chain_df is None or chain_df.empty:
        return empty

    df = chain_df.copy()
    for col in ("strike", "impliedVolatility", "openInterest", "volume", "bid", "ask", "lastPrice"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df = df.dropna(subset=["strike"])
    if df.empty:
        return empty

    df["_distance"] = (df["strike"] - float(current_price)).abs()
    row = df.sort_values("_distance", ascending=True).iloc[0]

    iv = _safe_float(row.get("impliedVolatility"), np.nan)
    oi = _safe_float(row.get("openInterest"), 0.0)
    vol = _safe_float(row.get("volume"), 0.0)
    bid = _safe_float(row.get("bid"), np.nan)
    ask = _safe_float(row.get("ask"), np.nan)
    last = _safe_float(row.get("lastPrice"), np.nan)

    mid = np.nan
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
    elif np.isfinite(last) and last > 0:
        mid = float(last)

    spread_pct = np.nan
    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid and np.isfinite(mid) and mid > 0:
        spread_pct = ((ask - bid) / mid) * 100.0

    if not np.isfinite(iv) or iv <= 0:
        iv = np.nan

    return {
        "iv": float(iv) if np.isfinite(iv) else None,
        "oi": float(max(oi, 0.0)),
        "volume": float(max(vol, 0.0)),
        "mid": float(mid) if np.isfinite(mid) else None,
        "spread_pct": float(spread_pct) if np.isfinite(spread_pct) else None,
    }


def _normalize_release_timing(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if any((value.hour, value.minute, value.second, value.microsecond)):
            total_minutes = (value.hour * 60) + value.minute
            if total_minutes < (9 * 60 + 30):
                return "before market open"
            if total_minutes >= (16 * 60):
                return "after market close"
            return "during market hours"
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "before" in text or text in {"bmo", "pre", "am"}:
        return "before market open"
    if "after" in text or text in {"amc", "post", "pm"}:
        return "after market close"
    if "during" in text or "intraday" in text:
        return "during market hours"
    return "unknown"


def _nearest_common_strike_pair_stats(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    current_price: float,
) -> Dict[str, Optional[float]]:
    """
    Return ATM call/put stats from the same strike.

    This avoids mixing call and put mids from different strikes when estimating
    implied move from the ATM straddle.
    """
    empty = {
        "strike": None,
        "call_iv": None,
        "put_iv": None,
        "call_mid": None,
        "put_mid": None,
        "call_spread_pct": None,
        "put_spread_pct": None,
        "oi_total": None,
        "volume_total": None,
    }
    if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
        return empty

    def _prep(frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        for col in ("strike", "impliedVolatility", "openInterest", "volume", "bid", "ask", "lastPrice"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        return df.dropna(subset=["strike"])

    calls = _prep(calls_df)
    puts = _prep(puts_df)
    if calls.empty or puts.empty:
        return empty

    common_strikes = sorted(set(calls["strike"].dropna().tolist()) & set(puts["strike"].dropna().tolist()))
    if not common_strikes:
        return empty

    chosen_strike = min(common_strikes, key=lambda s: abs(float(s) - float(current_price)))

    def _select_row(df: pd.DataFrame, strike: float) -> pd.Series:
        tmp = df.copy()
        tmp["_dist"] = (tmp["strike"] - float(strike)).abs()
        return tmp.sort_values("_dist", ascending=True).iloc[0]

    call_row = _select_row(calls, chosen_strike)
    put_row = _select_row(puts, chosen_strike)

    def _mid_and_spread(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        bid = _safe_float(row.get("bid"), np.nan)
        ask = _safe_float(row.get("ask"), np.nan)
        last = _safe_float(row.get("lastPrice"), np.nan)
        mid = np.nan
        if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
            mid = (bid + ask) / 2.0
        elif np.isfinite(last) and last > 0:
            mid = last
        spread_pct = np.nan
        if np.isfinite(bid) and np.isfinite(ask) and np.isfinite(mid) and mid > 0 and ask >= bid:
            spread_pct = ((ask - bid) / mid) * 100.0
        return (
            float(mid) if np.isfinite(mid) else None,
            float(spread_pct) if np.isfinite(spread_pct) else None,
        )

    call_mid, call_spread = _mid_and_spread(call_row)
    put_mid, put_spread = _mid_and_spread(put_row)
    call_iv = _safe_float(call_row.get("impliedVolatility"), np.nan)
    put_iv = _safe_float(put_row.get("impliedVolatility"), np.nan)
    call_oi = _safe_float(call_row.get("openInterest"), 0.0)
    put_oi = _safe_float(put_row.get("openInterest"), 0.0)
    call_vol = _safe_float(call_row.get("volume"), 0.0)
    put_vol = _safe_float(put_row.get("volume"), 0.0)

    return {
        "strike": float(chosen_strike),
        "call_iv": float(call_iv) if np.isfinite(call_iv) and call_iv > 0 else None,
        "put_iv": float(put_iv) if np.isfinite(put_iv) and put_iv > 0 else None,
        "call_mid": call_mid,
        "put_mid": put_mid,
        "call_spread_pct": call_spread,
        "put_spread_pct": put_spread,
        "oi_total": float(max(call_oi, 0.0) + max(put_oi, 0.0)),
        "volume_total": float(max(call_vol, 0.0) + max(put_vol, 0.0)),
    }


# ─── Scoring (unchanged) ─────────────────────────────────────────────────────

def _score_setup(iv_rv: float, ts_slope: float, dte: Optional[int], liquidity: float) -> float:
    iv_component = float(np.clip((iv_rv - 0.90) / 0.70, 0.0, 1.0)) if np.isfinite(iv_rv) else 0.0
    ts_component = (
        float(np.clip(np.exp(-0.5 * ((ts_slope - TS_SLOPE_TARGET) / TS_SLOPE_BAND) ** 2), 0.0, 1.0))
        if np.isfinite(ts_slope) else 0.0
    )
    if dte is None:
        timing_component = 0.45
    else:
        timing_component = float(np.clip(np.exp(-0.5 * ((float(dte) - 7.0) / 6.0) ** 2), 0.0, 1.0))
    liq_component = float(np.clip(
        (np.log1p(max(liquidity, 1.0)) - np.log1p(500.0)) / (np.log1p(25000.0) - np.log1p(500.0)),
        0.0, 1.0,
    ))
    score = 0.36 * iv_component + 0.24 * ts_component + 0.24 * timing_component + 0.16 * liq_component
    return float(np.clip(score, 0.0, 1.0))


def _score_expectancy(net_edge_pct: float, drawdown_risk_pct: float, sample_size: int) -> float:
    if not np.isfinite(net_edge_pct):
        return 0.45
    edge_component = float(np.clip((net_edge_pct + 0.60) / 2.40, 0.0, 1.0))
    ratio = float(net_edge_pct / max(drawdown_risk_pct, 0.25))
    ratio_component = float(np.clip((ratio + 0.25) / 0.90, 0.0, 1.0))
    sample_component = float(np.clip(float(sample_size) / 8.0, 0.0, 1.0))
    return float(np.clip(0.55 * edge_component + 0.35 * ratio_component + 0.10 * sample_component, 0.0, 1.0))


def _estimate_transaction_cost_pct(liquidity: float, spread_pct: Optional[float]) -> float:
    liq_component = 1.0 - float(np.clip(
        (np.log1p(max(liquidity, 1.0)) - np.log1p(500.0)) / (np.log1p(25000.0) - np.log1p(500.0)),
        0.0, 1.0,
    ))
    spread_component = float(np.clip((_safe_float(spread_pct, 18.0)) / 25.0, 0.20, 2.0))
    cost_pct = 0.18 + 0.55 * liq_component + 0.45 * spread_component
    return float(np.clip(cost_pct, 0.15, 2.25))


def _compute_move_anchor(median_move_pct: float, avg_last4_move_pct: float) -> Optional[float]:
    median_val = _safe_float(median_move_pct, np.nan)
    avg4_val = _safe_float(avg_last4_move_pct, np.nan)
    if np.isfinite(median_val) and np.isfinite(avg4_val):
        return float(0.65 * avg4_val + 0.35 * median_val)
    if np.isfinite(avg4_val):
        return float(avg4_val)
    if np.isfinite(median_val):
        return float(median_val)
    return None


def _compute_move_uncertainty_pct(
    move_std_pct: float, sample_size: int, move_source: str
) -> Optional[float]:
    std_val = _safe_float(move_std_pct, np.nan)
    if not np.isfinite(std_val) or std_val <= 0 or sample_size <= 0:
        return None
    source = str(move_source or "none")
    if source == "earnings_history":
        n_eff = float(sample_size)
    elif source == "daily_fallback":
        n_eff = float(np.clip(sample_size * 0.35, 1.0, 10.0))
    else:
        n_eff = float(np.clip(sample_size * 0.25, 1.0, 6.0))
    uncertainty = MOVE_UNCERTAINTY_Z_SCORE * std_val / np.sqrt(max(n_eff, 1.0))
    return float(np.clip(uncertainty, 0.03, 6.0))


# ─── MarketData.app data layer ───────────────────────────────────────────────

def _term_structure_from_mda_chain(
    chain_df: pd.DataFrame,
    current_price: float,
    max_expiries: int = 6,
) -> Tuple[
    List[float], List[float], float, float, float, float,
    Optional[float], Optional[float], Optional[int],
    Optional[float], Optional[float],
]:
    """
    Build the term structure from a MarketData.app chain DataFrame.

    One chain fetch covers all expirations simultaneously, saving API credits
    vs. the yfinance approach (which fetches one expiry at a time).

    Returns the same 11-element tuple as the legacy _term_structure_points().
    """
    empty_return = ([], [], np.nan, np.nan, 0.0, 0.0, None, None, None, None, None)

    if chain_df is None or chain_df.empty or "expiration_date" not in chain_df.columns:
        return empty_return

    today = _utc_today_date()
    days: List[float] = []
    ivs: List[float] = []
    oi_values: List[float] = []
    vol_values: List[float] = []
    near_term_implied_move_pct: Optional[float] = None
    near_term_spread_pct: Optional[float] = None
    near_term_dte: Optional[int] = None
    near_term_liquidity_proxy: Optional[float] = None

    # Group by expiration and iterate in DTE order
    expiry_groups = chain_df.groupby("expiration_date")
    sorted_expiries = sorted(expiry_groups.groups.keys())

    processed = 0
    for exp_str in sorted_expiries:
        if processed >= max_expiries:
            break
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        dte = float((exp_date - today).days)
        if dte <= 0:
            continue

        grp = expiry_groups.get_group(exp_str)
        calls = grp[grp["side"].str.lower() == "call"] if "side" in grp.columns else grp
        puts = grp[grp["side"].str.lower() == "put"] if "side" in grp.columns else pd.DataFrame()

        call_stats = _nearest_atm_option_stats(calls, current_price)
        put_stats = _nearest_atm_option_stats(puts, current_price) if not puts.empty else {
            "iv": None, "oi": None, "volume": None, "mid": None, "spread_pct": None
        }
        pair_stats = _nearest_common_strike_pair_stats(calls, puts, current_price) if not puts.empty else {}

        iv_candidates = [
            v for v in (call_stats["iv"], put_stats["iv"])
            if v is not None and np.isfinite(v) and v > 0
        ]
        pair_iv_candidates = [
            v for v in (pair_stats.get("call_iv"), pair_stats.get("put_iv"))
            if v is not None and np.isfinite(v) and v > 0
        ]
        if pair_iv_candidates:
            iv_candidates = pair_iv_candidates
        if not iv_candidates:
            continue

        iv_atm = float(np.mean(iv_candidates))
        days.append(dte)
        ivs.append(iv_atm)
        oi_total = _safe_float(pair_stats.get("oi_total"), np.nan)
        vol_total = _safe_float(pair_stats.get("volume_total"), np.nan)
        if np.isfinite(oi_total) and np.isfinite(vol_total):
            oi_values.append(float(max(oi_total, 0.0)))
            vol_values.append(float(max(vol_total, 0.0)))
        else:
            oi_values.append(float(max(call_stats["oi"] or 0.0, 0.0) + max(put_stats["oi"] or 0.0, 0.0)))
            vol_values.append(float(max(call_stats["volume"] or 0.0, 0.0) + max(put_stats["volume"] or 0.0, 0.0)))

        if near_term_implied_move_pct is None:
            call_mid = _safe_float(pair_stats.get("call_mid"), np.nan)
            put_mid = _safe_float(pair_stats.get("put_mid"), np.nan)
            if not (np.isfinite(call_mid) and np.isfinite(put_mid)):
                call_mid = _safe_float(call_stats["mid"], np.nan)
                put_mid = _safe_float(put_stats["mid"], np.nan)
            if np.isfinite(call_mid) and np.isfinite(put_mid) and current_price > 0:
                near_term_implied_move_pct = float(((call_mid + put_mid) / current_price) * 100.0)
                near_term_dte = int(dte)
                pair_liq = _safe_float(pair_stats.get("oi_total"), np.nan) + _safe_float(pair_stats.get("volume_total"), np.nan)
                if np.isfinite(pair_liq):
                    near_term_liquidity_proxy = float(max(pair_liq, 0.0))
                else:
                    near_term_liquidity_proxy = float(
                        max(call_stats["oi"] or 0.0, 0.0)
                        + max(put_stats["oi"] or 0.0, 0.0)
                        + max(call_stats["volume"] or 0.0, 0.0)
                        + max(put_stats["volume"] or 0.0, 0.0)
                    )

            spread_candidates = [
                float(v) for v in (call_stats.get("spread_pct"), put_stats.get("spread_pct"))
                if v is not None and np.isfinite(v) and v >= 0
            ]
            pair_spread_candidates = [
                float(v) for v in (pair_stats.get("call_spread_pct"), pair_stats.get("put_spread_pct"))
                if v is not None and np.isfinite(v) and v >= 0
            ]
            if pair_spread_candidates:
                spread_candidates = pair_spread_candidates
            if spread_candidates:
                near_term_spread_pct = float(np.mean(spread_candidates))

        processed += 1

    if len(days) < 2:
        return (
            days, ivs, np.nan, np.nan, 0.0, 0.0,
            near_term_implied_move_pct, near_term_spread_pct,
            near_term_dte, None, near_term_liquidity_proxy,
        )

    order = np.argsort(np.array(days, dtype=float))
    days_arr = np.array(days, dtype=float)[order]
    ivs_arr = np.array(ivs, dtype=float)[order]

    iv30 = float(np.interp(30.0, days_arr, ivs_arr))
    iv45 = float(np.interp(45.0, days_arr, ivs_arr))
    # FIX 1: slope between first real listed tenor and tenor nearest 45d.
    # np.interp(0.0, ...) clamps to shortest expiry — calling it "0-45d" is wrong.
    _ts_near_dte = float(days_arr[0])
    _ts_far_idx  = int(np.argmin(np.abs(days_arr - 45.0)))
    _ts_far_dte  = float(days_arr[_ts_far_idx])
    slope_0_45   = float((ivs_arr[_ts_far_idx] - ivs_arr[0]) / max(_ts_far_dte - _ts_near_dte, 1.0))
    near_back_iv_ratio = (
        float(ivs_arr[0] / ivs_arr[1])
        if len(ivs_arr) >= 2 and np.isfinite(ivs_arr[0]) and np.isfinite(ivs_arr[1]) and ivs_arr[1] > 0
        else None
    )
    avg_oi = float(np.mean(oi_values)) if oi_values else 0.0
    avg_opt_vol = float(np.mean(vol_values)) if vol_values else 0.0

    return (
        days_arr.tolist(),
        ivs_arr.tolist(),
        iv30,
        iv45,
        slope_0_45,
        max(avg_oi, avg_opt_vol),
        near_term_implied_move_pct,
        near_term_spread_pct,
        near_term_dte,
        near_back_iv_ratio,
        near_term_liquidity_proxy,
        _ts_near_dte,
        _ts_far_dte,
    )


def _smile_curvature_from_mda_chain(
    chain_df: pd.DataFrame, current_price: float
) -> Dict[str, Any]:
    """
    Estimate near-term IV smile curvature from an MDApp chain DataFrame.
    Identical maths to the legacy yfinance version; only the data source changes.
    """
    empty = {"curvature": None, "concave": False, "points": 0}
    if chain_df is None or chain_df.empty or "expiration_date" not in chain_df.columns:
        return empty

    today = _utc_today_date()
    sorted_expiries = sorted(chain_df["expiration_date"].dropna().unique())

    for exp_str in sorted_expiries[:3]:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if (exp_date - today).days <= 0:
                continue
        except (ValueError, TypeError):
            continue

        grp = chain_df[chain_df["expiration_date"] == exp_str].copy()
        grp["strike"] = pd.to_numeric(grp.get("strike"), errors="coerce")
        grp["impliedVolatility"] = pd.to_numeric(grp.get("impliedVolatility"), errors="coerce")
        grp = grp.dropna(subset=["strike", "impliedVolatility"])
        grp = grp[grp["impliedVolatility"] > 0.0]
        if grp.empty:
            continue

        iv_by_strike: Dict[float, List[float]] = {}
        for _, row in grp.iterrows():
            strike = _safe_float(row.get("strike"), np.nan)
            iv = _safe_float(row.get("impliedVolatility"), np.nan)
            if not np.isfinite(strike) or strike <= 0 or not np.isfinite(iv) or iv <= 0:
                continue
            iv_by_strike.setdefault(float(strike), []).append(float(iv))

        rows: List[Tuple[float, float]] = []
        for strike, values in iv_by_strike.items():
            iv_mean = float(np.mean(values))
            moneyness = float((strike / max(current_price, 1e-6)) - 1.0)
            if abs(moneyness) <= 0.20:
                rows.append((moneyness, iv_mean))

        if len(rows) < 5:
            continue

        rows.sort(key=lambda x: x[0])
        x = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        try:
            a, _, _ = np.polyfit(x, y, 2)
        except Exception:
            continue

        curvature = float(a)
        concave = bool(curvature < -0.5)
        return {"curvature": curvature, "concave": concave, "points": int(len(rows))}

    return empty


def _next_earnings_from_mda(
    earnings_df: pd.DataFrame,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Return (days_to_next_earnings, report_time) from an MDApp earnings DataFrame.
    report_time = 'before market open' | 'after market close' | 'during market hours' | None
    """
    if earnings_df is None or earnings_df.empty:
        return None, None

    today = _utc_today_date()
    for _, row in earnings_df.iterrows():
        rd = row.get("report_date")
        if rd is None:
            continue
        try:
            event_date = rd if isinstance(rd, type(today)) else pd.Timestamp(rd).date()
        except Exception:
            continue

        days = (event_date - today).days
        if days >= 0:
            rt = row.get("reportTime")
            return int(days), (str(rt) if rt is not None else None)

    return None, None


def _earnings_dates_from_mda(earnings_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Extract past earnings events from an MDApp earnings DataFrame.

    Each row contains:
      - event_date: pandas Timestamp (normalized to date)
      - release_timing: normalized timing token
    """
    if earnings_df is None or earnings_df.empty:
        return []

    today = pd.Timestamp(_utc_today_date())
    event_by_date: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for _, row in earnings_df.iterrows():
        rd = row.get("report_date")
        if rd is None:
            continue
        try:
            ts = pd.Timestamp(rd).normalize()
            if ts <= today:
                release_timing = _normalize_release_timing(row.get("reportTime"))
                existing = event_by_date.get(ts)
                if existing is None or existing.get("release_timing") == "unknown":
                    event_by_date[ts] = {
                        "event_date": ts,
                        "release_timing": release_timing,
                    }
        except Exception:
            continue
    return [event_by_date[key] for key in sorted(event_by_date.keys())]


# ─── yfinance fallback layer ─────────────────────────────────────────────────

def _term_structure_points_yf(
    ticker: yf.Ticker,
    current_price: float,
    max_expiries: int = 6,
) -> Tuple[
    List[float], List[float], float, float, float, float,
    Optional[float], Optional[float], Optional[int],
    Optional[float], Optional[float],
]:
    """yfinance fallback for term structure — unchanged logic from v0.1."""
    days: List[float] = []
    ivs: List[float] = []
    oi_values: List[float] = []
    vol_values: List[float] = []
    near_term_implied_move_pct: Optional[float] = None
    near_term_spread_pct: Optional[float] = None
    near_term_dte: Optional[int] = None
    near_term_liquidity_proxy: Optional[float] = None

    today = _utc_today_date()
    expirations = list(getattr(ticker, "options", []) or [])
    for exp in expirations[:max_expiries]:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = float((exp_date - today).days)
            if dte <= 0:
                continue
            chain = ticker.option_chain(exp)
            call_stats = _nearest_atm_option_stats(chain.calls, current_price)
            put_stats = _nearest_atm_option_stats(chain.puts, current_price)
            pair_stats = _nearest_common_strike_pair_stats(chain.calls, chain.puts, current_price)
            iv_candidates = [
                v for v in (call_stats["iv"], put_stats["iv"])
                if v is not None and np.isfinite(v) and v > 0
            ]
            pair_iv_candidates = [
                v for v in (pair_stats.get("call_iv"), pair_stats.get("put_iv"))
                if v is not None and np.isfinite(v) and v > 0
            ]
            if pair_iv_candidates:
                iv_candidates = pair_iv_candidates
            if not iv_candidates:
                continue
            iv_atm = float(np.mean(iv_candidates))
            days.append(dte)
            ivs.append(iv_atm)
            oi_total = _safe_float(pair_stats.get("oi_total"), np.nan)
            vol_total = _safe_float(pair_stats.get("volume_total"), np.nan)
            if np.isfinite(oi_total) and np.isfinite(vol_total):
                oi_values.append(float(max(oi_total, 0.0)))
                vol_values.append(float(max(vol_total, 0.0)))
            else:
                oi_values.append(float(max(call_stats["oi"] or 0.0, 0.0) + max(put_stats["oi"] or 0.0, 0.0)))
                vol_values.append(float(max(call_stats["volume"] or 0.0, 0.0) + max(put_stats["volume"] or 0.0, 0.0)))
            if near_term_implied_move_pct is None:
                call_mid = _safe_float(pair_stats.get("call_mid"), np.nan)
                put_mid = _safe_float(pair_stats.get("put_mid"), np.nan)
                if not (np.isfinite(call_mid) and np.isfinite(put_mid)):
                    call_mid = _safe_float(call_stats["mid"], np.nan)
                    put_mid = _safe_float(put_stats["mid"], np.nan)
                if np.isfinite(call_mid) and np.isfinite(put_mid) and current_price > 0:
                    near_term_implied_move_pct = float(((call_mid + put_mid) / current_price) * 100.0)
                    near_term_dte = int(dte)
                    pair_liq = _safe_float(pair_stats.get("oi_total"), np.nan) + _safe_float(pair_stats.get("volume_total"), np.nan)
                    if np.isfinite(pair_liq):
                        near_term_liquidity_proxy = float(max(pair_liq, 0.0))
                    else:
                        near_term_liquidity_proxy = float(
                            max(call_stats["oi"] or 0.0, 0.0) + max(put_stats["oi"] or 0.0, 0.0)
                            + max(call_stats["volume"] or 0.0, 0.0) + max(put_stats["volume"] or 0.0, 0.0)
                        )
                spread_candidates = [
                    float(v) for v in (call_stats.get("spread_pct"), put_stats.get("spread_pct"))
                    if v is not None and np.isfinite(v) and v >= 0
                ]
                pair_spread_candidates = [
                    float(v) for v in (pair_stats.get("call_spread_pct"), pair_stats.get("put_spread_pct"))
                    if v is not None and np.isfinite(v) and v >= 0
                ]
                if pair_spread_candidates:
                    spread_candidates = pair_spread_candidates
                if spread_candidates:
                    near_term_spread_pct = float(np.mean(spread_candidates))
        except Exception:
            continue

    if len(days) < 2:
        return (days, ivs, np.nan, np.nan, 0.0, 0.0, near_term_implied_move_pct,
                near_term_spread_pct, near_term_dte, None, near_term_liquidity_proxy)

    order = np.argsort(np.array(days, dtype=float))
    days_arr = np.array(days, dtype=float)[order]
    ivs_arr = np.array(ivs, dtype=float)[order]
    iv30 = float(np.interp(30.0, days_arr, ivs_arr))
    iv45 = float(np.interp(45.0, days_arr, ivs_arr))
    # FIX 1: slope between first real listed tenor and tenor nearest 45d.
    _ts_near_dte = float(days_arr[0])
    _ts_far_idx  = int(np.argmin(np.abs(days_arr - 45.0)))
    _ts_far_dte  = float(days_arr[_ts_far_idx])
    slope_0_45   = float((ivs_arr[_ts_far_idx] - ivs_arr[0]) / max(_ts_far_dte - _ts_near_dte, 1.0))
    near_back_iv_ratio = (
        float(ivs_arr[0] / ivs_arr[1])
        if len(ivs_arr) >= 2 and np.isfinite(ivs_arr[0]) and np.isfinite(ivs_arr[1]) and ivs_arr[1] > 0
        else None
    )
    avg_oi = float(np.mean(oi_values)) if oi_values else 0.0
    avg_opt_vol = float(np.mean(vol_values)) if vol_values else 0.0
    return (
        days_arr.tolist(), ivs_arr.tolist(), iv30, iv45, slope_0_45,
        max(avg_oi, avg_opt_vol), near_term_implied_move_pct, near_term_spread_pct,
        near_term_dte, near_back_iv_ratio, near_term_liquidity_proxy,
        _ts_near_dte, _ts_far_dte,
    )


def _smile_curvature_yf(ticker: yf.Ticker, current_price: float) -> Dict[str, Any]:
    """yfinance fallback for smile curvature — unchanged logic from v0.1."""
    empty = {"curvature": None, "concave": False, "points": 0}
    expirations = list(getattr(ticker, "options", []) or [])
    if not expirations:
        return empty

    for exp in expirations[:3]:
        try:
            chain = ticker.option_chain(exp)
        except Exception:
            continue

        prepared_frames: List[pd.DataFrame] = []
        for frame in (chain.calls, chain.puts):
            clean = frame.copy()
            clean["strike"] = pd.to_numeric(clean.get("strike"), errors="coerce")
            clean["impliedVolatility"] = pd.to_numeric(clean.get("impliedVolatility"), errors="coerce")
            clean.dropna(subset=["strike", "impliedVolatility"], inplace=True)
            clean = clean[clean["impliedVolatility"] > 0.0]
            if not clean.empty:
                prepared_frames.append(clean)

        if not prepared_frames:
            continue

        iv_by_strike: Dict[float, List[float]] = {}
        for frame in prepared_frames:
            for _, row in frame.iterrows():
                strike = _safe_float(row.get("strike"), np.nan)
                iv = _safe_float(row.get("impliedVolatility"), np.nan)
                if not np.isfinite(strike) or strike <= 0 or not np.isfinite(iv) or iv <= 0:
                    continue
                iv_by_strike.setdefault(float(strike), []).append(float(iv))

        rows: List[Tuple[float, float]] = []
        for strike, values in iv_by_strike.items():
            moneyness = float((strike / max(current_price, 1e-6)) - 1.0)
            if abs(moneyness) <= 0.20:
                rows.append((moneyness, float(np.mean(values))))

        if len(rows) < 5:
            continue

        rows.sort(key=lambda x: x[0])
        x = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        try:
            a, _, _ = np.polyfit(x, y, 2)
        except Exception:
            continue

        return {"curvature": float(a), "concave": bool(float(a) < -0.5), "points": int(len(rows))}

    return empty


def _next_earnings_days_yf(ticker: yf.Ticker) -> Optional[int]:
    """yfinance fallback for next earnings DTE — unchanged logic from v0.1."""
    today = _utc_today_date()
    get_dates = getattr(ticker, "get_earnings_dates", None)
    if callable(get_dates):
        try:
            earnings_df = get_dates(limit=8)
            if earnings_df is not None and not earnings_df.empty:
                for idx in earnings_df.index:
                    days = (pd.Timestamp(idx).date() - today).days
                    if days >= 0:
                        return int(days)
        except Exception as exc:
            logger.debug("yfinance get_earnings_dates failed: %s", exc)

    fallback = getattr(ticker, "earnings_dates", None)
    if fallback is not None and not fallback.empty:
        try:
            for idx in fallback.index:
                days = (pd.Timestamp(idx).date() - today).days
                if days >= 0:
                    return int(days)
        except Exception as exc:
            logger.debug("yfinance earnings_dates fallback failed: %s", exc)
    return None


# ─── Earnings move profile (uses yfinance price history + MDApp/yfinance dates) ──

def _historical_earnings_move_profile(
    close: pd.Series,
    earnings_events: List[Any],
) -> Dict[str, Any]:
    """
    Compute historical earnings move statistics from close-price series and
    past earnings events.

    close          : yfinance 6-month OHLCV close series (no API cost, free)
    earnings_events: list of either:
                     - pandas timestamps (legacy compatibility), or
                     - dicts with {"event_date": Timestamp, "release_timing": str}
    """
    empty = {
        "event_count": 0,
        "median_move_pct": None,
        "p90_move_pct": None,
        "avg_last4_move_pct": None,
        "std_move_pct": None,
        "source": "none",
    }

    if close.empty:
        return empty

    price_series = close.copy()
    if isinstance(price_series.index, pd.DatetimeIndex):
        idx = price_series.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        price_series.index = idx.normalize()
    price_series = price_series[~price_series.index.duplicated(keep="last")].sort_index()
    index_arr = price_series.index.to_numpy()
    if len(index_arr) < 10:
        return empty

    today = pd.Timestamp(_utc_today_date())
    parsed_events: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for item in earnings_events or []:
        try:
            if isinstance(item, dict):
                event_ts_raw = item.get("event_date")
                timing = _normalize_release_timing(item.get("release_timing"))
            else:
                # Raw timestamp (not a dict) — infer BMO/AMC from time-of-day
                # instead of silently collapsing to "unknown".
                event_ts_raw = item
                timing = _normalize_release_timing(item)
            if event_ts_raw is None:
                continue
            event_ts = pd.Timestamp(event_ts_raw).normalize()
            if event_ts > today:
                continue
            existing = parsed_events.get(event_ts)
            if existing is None or existing.get("release_timing") == "unknown":
                parsed_events[event_ts] = {
                    "event_date": event_ts,
                    "release_timing": timing,
                }
        except Exception:
            continue

    past_events = [parsed_events[key] for key in sorted(parsed_events.keys())]
    event_moves: List[float] = []

    for event in past_events[-24:]:
        event_ts = pd.Timestamp(event["event_date"]).normalize()
        release_timing = _normalize_release_timing(event.get("release_timing"))
        event_loc = int(index_arr.searchsorted(event_ts.to_datetime64(), side="left"))
        if event_loc >= len(index_arr):
            continue
        matched_event_session = pd.Timestamp(index_arr[event_loc]).normalize() == event_ts

        if release_timing == "after market close":
            if matched_event_session:
                pre_loc = event_loc
                post_loc = event_loc + 1
            else:
                pre_loc = event_loc - 1
                post_loc = event_loc
        else:
            pre_loc = event_loc - 1
            post_loc = event_loc

        if pre_loc < 0 or post_loc < 0 or pre_loc >= len(index_arr) or post_loc >= len(index_arr):
            continue

        pre_px = _safe_float(price_series.iloc[pre_loc], np.nan)
        post_px = _safe_float(price_series.iloc[post_loc], np.nan)
        if not np.isfinite(pre_px) or not np.isfinite(post_px) or pre_px <= 0:
            continue
        move_pct = abs((post_px - pre_px) / pre_px) * 100.0
        if np.isfinite(move_pct):
            event_moves.append(float(move_pct))

    if not event_moves:
        # Fallback to recent daily absolute moves
        daily_moves = (
            price_series.pct_change().abs().dropna().tail(126).to_numpy(dtype=float) * 100.0
        )
        if daily_moves.size == 0:
            return empty
        if daily_moves.size >= 5:
            low_clip, high_clip = np.percentile(daily_moves, [1.0, 99.0])
            daily_moves = np.clip(daily_moves, low_clip, high_clip)
        avg_last4 = float(np.mean(daily_moves[-4:]))
        std_move = float(np.std(daily_moves, ddof=1)) if daily_moves.size > 1 else 0.0
        return {
            "event_count": int(daily_moves.size),
            "median_move_pct": float(np.median(daily_moves)),
            "p90_move_pct": float(np.percentile(daily_moves, 90)),
            "avg_last4_move_pct": avg_last4,
            "std_move_pct": std_move,
            "source": "daily_fallback",
        }

    moves = np.array(event_moves, dtype=float)
    if moves.size >= 5:
        low_clip, high_clip = np.percentile(moves, [1.0, 99.0])
        moves = np.clip(moves, low_clip, high_clip)
    avg_last4 = float(np.mean(moves[-4:]))
    std_move = float(np.std(moves, ddof=1)) if moves.size > 1 else 0.0
    return {
        "event_count": int(moves.size),
        "median_move_pct": float(np.median(moves)),
        "p90_move_pct": float(np.percentile(moves, 90)),
        "avg_last4_move_pct": avg_last4,
        "std_move_pct": std_move,
        "raw_moves_pct": [float(x) for x in moves.tolist()],  # for kurtosis + crush-rate
        "source": "earnings_history",
    }


# ─── Main analysis entry point ────────────────────────────────────────────────

# ─── RV estimators: Yang-Zhang (2000) + HAR-RV (Corsi 2009) ─────────────────
#
# Rationale:
#   Close-to-close HV biases the IV/RV ratio because it misses intraday moves
#   and is noisy (high variance).  Yang-Zhang uses OHLC data (free via yfinance)
#   and is ~5-7x more statistically efficient.  HAR-RV then decomposes the
#   Yang-Zhang daily series into daily / weekly / monthly persistence components
#   to produce a forward-looking RV forecast — the proper denominator for
#   comparing against implied volatility (which is itself forward-looking).

def _yang_zhang_rv30(hist: pd.DataFrame, window: int = 30) -> float:
    """
    Yang-Zhang (2000) OHLC realized volatility over *window* trading days.

        σ²_YZ = σ²_o + k·σ²_c + (1-k)·σ²_RS

    where
      σ²_o  = sample variance of overnight log-returns  log(Open_t / Close_{t-1})
      σ²_c  = sample variance of open-to-close returns  log(Close_t / Open_t)
      σ²_RS = mean Rogers-Satchell per-day variance (direction-free intraday)
      k     = 0.34 / (1.34 + (n+1)/(n-1))  — paper-optimal weight

    Returns annualised volatility, or np.nan if data is insufficient.
    """
    needed = ("Open", "High", "Low", "Close")
    if not all(c in hist.columns for c in needed):
        return np.nan

    df = hist[list(needed)].copy().dropna()
    if len(df) < max(window + 1, 6):
        return np.nan

    df = df.tail(window + 1)  # extra row for overnight return denominator

    o = np.log(df["Open"] / df["Close"].shift(1)).dropna()
    c = np.log(df["Close"] / df["Open"]).dropna()
    h = np.log(df["High"] / df["Open"]).dropna()
    l = np.log(df["Low"] / df["Open"]).dropna()

    idx = o.index.intersection(c.index).intersection(h.index).intersection(l.index)
    o = o[idx].tail(window)
    c = c[idx].tail(window)
    h = h[idx].tail(window)
    l = l[idx].tail(window)

    n = len(o)
    if n < 5:
        return np.nan

    # FIX 9: do NOT clip individual RS values — negative RS is mathematically
    # valid (price traded below open all day) and clipping introduces upward
    # bias in the YZ estimator. Clip only the final combined variance before sqrt.
    rs = h * (h - c) + l * (l - c)  # Rogers-Satchell daily variance (unclipped)
    k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))

    # Paper specifies sample variance (1/(n-1)), not population variance (1/n)
    var_o = float(((o - o.mean()) ** 2).sum() / max(n - 1, 1))
    var_c = float(((c - c.mean()) ** 2).sum() / max(n - 1, 1))
    var_rs = float(rs.mean())

    yz_var = var_o + k * var_c + (1.0 - k) * var_rs
    return float(np.sqrt(max(yz_var, 1e-10) * 252))


def _rs_daily_vol_series(hist: pd.DataFrame) -> pd.Series:
    """
    Per-day Rogers-Satchell annualised volatility series — used as input to HAR.

    RS_t = log(H_t/O_t)·log(H_t/C_t) + log(L_t/O_t)·log(L_t/C_t)

    Each RS_t is an unbiased, direction-free estimate of daily variance that
    requires no window.  Annualising: vol_t = √(252 · RS_t).
    """
    needed = ("Open", "High", "Low", "Close")
    if not all(c in hist.columns for c in needed):
        return pd.Series(dtype=float)

    df = hist[list(needed)].copy().dropna()
    h = np.log(df["High"] / df["Open"])
    l = np.log(df["Low"] / df["Open"])
    c = np.log(df["Close"] / df["Open"])

    rs_var = (h * (h - c) + l * (l - c)).clip(lower=0.0)
    return np.sqrt(rs_var * 252).dropna()


def _har_rv_forecast(rv_daily: pd.Series, horizon: int = 1) -> Optional[float]:
    """
    Corsi (2009) Heterogeneous Autoregressive Realized Variance.

        RV_{t+h} = β₀ + β_d·RV_t + β_w·RV^(w)_t + β_m·RV^(m)_t + ε

    RV^(w)_t = 5-day average of daily RV ending at t  (weekly persistence)
    RV^(m)_t = 22-day average of daily RV ending at t (monthly persistence)

    Fitted by OLS on the in-sample daily series.  Returns a 1-day-ahead
    forecast (annualised vol), floored at 1 bp.  Returns None if the series
    is too short for a reliable monthly component (< 35 observations).

    Note: Input is annualised vol series from _rs_daily_vol_series(); we square
    to variance for HAR fitting (Corsi operates on RV, not σ) and convert the
    forecast back to vol via sqrt.  This avoids Jensen's-inequality bias that
    arises from regressing directly on σ.
    """
    n = len(rv_daily)
    if n < 35:
        return None

    # Convert vol → variance (Corsi 2009 operates on realised variance)
    rv = (rv_daily.values.astype(float)) ** 2
    y: List[float] = []
    X: List[List[float]] = []

    for i in range(22, n - horizon):
        rv_d = rv[i]
        rv_w = rv[max(i - 4, 0): i + 1].mean()
        rv_m = rv[max(i - 21, 0): i + 1].mean()
        y.append(rv[i + horizon])
        X.append([1.0, rv_d, rv_w, rv_m])

    if len(y) < 10:
        return None

    try:
        beta, _, _, _ = np.linalg.lstsq(
            np.array(X, dtype=float), np.array(y, dtype=float), rcond=None
        )
    except Exception as exc:
        logger.debug("HAR-RV OLS fitting failed: %s", exc)
        return None

    last_rv_d = rv[-1]
    last_rv_w = rv[-5:].mean() if n >= 5 else rv[-1]
    last_rv_m = rv[-22:].mean() if n >= 22 else rv[-1]

    forecast_var = float(beta[0] + beta[1] * last_rv_d + beta[2] * last_rv_w + beta[3] * last_rv_m)
    # Variance → vol; clamp negative variance to zero before sqrt, floor at 1 bp
    return max(np.sqrt(max(forecast_var, 0.0)), 1e-4)


def _bsm_greeks(
    S: float, T_days: float, sigma: float, r: float = 0.0525
) -> Dict[str, Optional[float]]:
    """
    Black-Scholes-Merton ATM option greeks (K = S).

    Parameters
    ----------
    S      : underlying price
    T_days : calendar days to option expiry (use earnings DTE for calendar spread context)
    sigma  : annualised implied vol (iv30)
    r      : risk-free rate (default ≈ current SOFR, ~5.25%)

    Returns greeks for a single ATM option leg:
      delta_call/put  — directional sensitivity (call ∈ [0,1], put ∈ [-1,0])
      gamma           — convexity per $1 move (identical for call and put ATM)
      vega            — $ sensitivity per 1 pp IV move (÷100 applied)
      theta_call/put  — daily time decay ($, negative = cost to holder)
    """
    empty: Dict[str, Optional[float]] = {
        k: None for k in ("delta_call", "delta_put", "gamma", "vega", "theta_call", "theta_put")
    }
    try:
        if T_days <= 0 or sigma <= 0 or S <= 0:
            return empty
        T = T_days / 365.0
        sq_T = math.sqrt(T)

        # ATM: K = S  →  log(S/K) = 0
        d1 = ((r + 0.5 * sigma ** 2) * T) / (sigma * sq_T)
        d2 = d1 - sigma * sq_T

        def _ncdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        def _npdf(x: float) -> float:
            return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

        N_d1 = _ncdf(d1)
        N_d2 = _ncdf(d2)
        n_d1 = _npdf(d1)

        delta_call = float(N_d1)
        delta_put  = float(N_d1 - 1.0)
        gamma      = float(n_d1 / (S * sigma * sq_T))
        vega       = float(S * n_d1 * sq_T * 0.01)          # per 1 pp vol move
        disc       = math.exp(-r * T)
        theta_call = float(
            (-S * n_d1 * sigma / (2.0 * sq_T) - r * S * disc * N_d2) / 365.0
        )
        theta_put  = float(
            (-S * n_d1 * sigma / (2.0 * sq_T) + r * S * disc * (1.0 - N_d2)) / 365.0
        )
        return {
            "delta_call": delta_call, "delta_put": delta_put,
            "gamma": gamma, "vega": vega,
            "theta_call": theta_call, "theta_put": theta_put,
        }
    except Exception as exc:
        logger.debug("BSM greeks computation failed: %s", exc)
        return empty


def _calendar_spread_payoff(
    S: float,
    iv_near: float,
    iv_back: float,
    T_near_days: float,
    T_back_days: float,
    r: float = 0.0525,
    n_points: int = 41,
) -> Optional[Dict[str, Any]]:
    """
    ATM call calendar spread payoff at near-leg expiry.

    Setup (long calendar for IV crush):
      SHORT near-term ATM call (expires around earnings, high IV)
      LONG  back-month ATM call (same strike, lower IV, more time)
      Net entry = debit = C_back_entry − C_near_entry

    At near-leg expiry with stock at S_t and post-earnings IV crushed:
      P&L = BSM_call(S_t, K=S, T_remaining, σ_crushed) − max(S_t−S, 0) − debit

    Returns payoff_scenarios as ($ P&L per share) across ±20% stock moves
    under four IV scenarios, plus approximate flat-IV breakeven moves.
    """
    try:
        if (T_near_days <= 0 or T_back_days <= T_near_days
                or iv_near <= 0 or iv_back <= 0 or S <= 0):
            return None

        T_near  = T_near_days / 365.0
        T_back  = T_back_days / 365.0
        T_rem   = max((T_back_days - T_near_days), 1.0) / 365.0
        K       = S  # ATM calendar — strike equals current underlying price

        def _ncdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        def _bsm_call(spot: float, T_yr: float, sigma: float) -> float:
            """BSM call price for arbitrary spot vs fixed strike K=S."""
            if T_yr <= 0 or sigma <= 0 or spot <= 0:
                return max(spot - K, 0.0)
            sq_T = math.sqrt(T_yr)
            d1 = (math.log(spot / K) + (r + 0.5 * sigma ** 2) * T_yr) / (sigma * sq_T)
            d2 = d1 - sigma * sq_T
            return spot * _ncdf(d1) - K * math.exp(-r * T_yr) * _ncdf(d2)

        # Entry premiums (at current stock price S)
        c_near_entry  = _bsm_call(S, T_near, iv_near)
        c_back_entry  = _bsm_call(S, T_back, iv_back)
        entry_debit   = c_back_entry - c_near_entry

        if entry_debit <= 0:
            return None  # degenerate: near leg worth more than back leg

        # Post-crush IV scenarios applied to the surviving back leg
        iv_scenarios = {
            "iv_expand_20":  iv_back * 1.20,
            "iv_flat":       iv_back,
            "iv_crush_25":   iv_back * 0.75,
            "iv_crush_45":   iv_back * 0.55,
        }

        # Price grid: −20 % to +20 % in equal steps
        n  = max(21, min(int(n_points), 81))
        moves = [round(-0.20 + i * (0.40 / (n - 1)), 4) for i in range(n)]

        payoff_rows: List[Dict[str, Any]] = []
        for move in moves:
            S_t = S * (1.0 + move)
            short_intrinsic = max(S_t - K, 0.0)   # cost of short near call at expiry
            row: Dict[str, Any] = {"move_pct": round(move * 100, 1), "price": round(S_t, 2)}
            for label, iv_post in iv_scenarios.items():
                c_back_post = _bsm_call(S_t, T_rem, iv_post)
                row[label] = round(c_back_post - short_intrinsic - entry_debit, 3)
            payoff_rows.append(row)

        # Breakeven detection on iv_flat scenario (sign change → linear interpolation)
        breakevens: List[float] = []
        for i in range(len(payoff_rows) - 1):
            p1 = payoff_rows[i]["iv_flat"]
            p2 = payoff_rows[i + 1]["iv_flat"]
            if p1 * p2 < 0:
                m1 = payoff_rows[i]["move_pct"]
                m2 = payoff_rows[i + 1]["move_pct"]
                be = m1 + (m2 - m1) * (-p1) / (p2 - p1)
                breakevens.append(round(be, 1))

        # FIX 8: per-contract values (1 contract = 100 shares)
        payoff_rows_per_contract = [
            {k: (round(v * 100, 2) if k not in ("move_pct", "price") else v)
             for k, v in row.items()}
            for row in payoff_rows
        ]
        return {
            "entry_debit":                round(entry_debit, 4),
            "entry_debit_per_contract":   round(entry_debit * 100, 2),   # FIX 8
            "entry_near_premium":         round(c_near_entry, 4),
            "entry_back_premium":         round(c_back_entry, 4),
            "t_near_days":                T_near_days,
            "t_back_days":                T_back_days,
            "t_remaining_days":           T_back_days - T_near_days,
            "iv_near":                    iv_near,
            "iv_back":                    iv_back,
            "breakeven_moves_pct":        breakevens,
            "payoff_scenarios":           payoff_rows,
            "payoff_scenarios_per_contract": payoff_rows_per_contract,  # FIX 8
            # FIX 2: make the synthetic nature explicit so the UI can warn users
            "calendar_is_theoretical":    True,
            "calendar_note":              (
                "Priced from interpolated IV30/IV45; back leg = near + 28d. "
                "Not guaranteed to match a live quoted chain."
            ),
        }
    except Exception as exc:
        logger.debug("Calendar payoff computation failed: %s", exc)
        return None


def _kelly_sizing(
    expected_net_edge_pct: float,
    drawdown_risk_pct: float,
    sample_confidence: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Conservative Kelly-based position sizing heuristic.

    Returns (full_kelly_pct, half_kelly_pct) as % of portfolio capital to risk.

    Derivation:
      raw_kelly = (edge / risk) × sample_confidence
      portfolio_kelly = raw_kelly × 3  (assumes option position ≈ 3% of portfolio per trade)
      half_kelly = portfolio_kelly / 2  — standard "fractional Kelly" for fat-tail risk

    Both values are capped conservatively for retail options sizing.
    """
    if not (np.isfinite(expected_net_edge_pct) and expected_net_edge_pct > 0):
        return None, None
    if not (np.isfinite(drawdown_risk_pct) and drawdown_risk_pct > 0.1):
        return None, None
    raw = (expected_net_edge_pct / drawdown_risk_pct) * float(np.clip(sample_confidence, 0.1, 1.0))
    full_kelly = float(np.clip(raw * 3.0, 0.10, 5.0))   # max 5% of portfolio
    half_kelly = float(np.clip(raw * 1.5, 0.05, 2.5))   # max 2.5% of portfolio
    return round(full_kelly, 2), round(half_kelly, 2)


def _rv_percentile_and_regime(
    hist_long: pd.DataFrame, current_rv30: float, window_days: int = 252
) -> Tuple[Optional[float], str]:
    """
    Percentile rank of *current_rv30* within its trailing *window_days* history.

    Uses the Rogers-Satchell daily vol series (same estimator as HAR-RV input)
    averaged over rolling 30-day windows.  Consistent estimator family throughout.

    Returns (percentile_rank: 0–100, regime_label: Low/Normal/Elevated/High).
    Low < 25th ≤ Normal < 50th ≤ Elevated < 75th ≤ High.
    """
    rs = _rs_daily_vol_series(hist_long)
    if len(rs) < 60:
        return None, "unknown"

    rolling_rv30 = rs.rolling(window=30, min_periods=20).mean()
    hist_vals = rolling_rv30.dropna().tail(window_days).values

    if len(hist_vals) < 30 or not np.isfinite(current_rv30) or current_rv30 <= 0:
        return None, "unknown"

    pct_rank = float(np.mean(hist_vals <= current_rv30) * 100.0)

    if pct_rank >= 75:
        regime = "High"
    elif pct_rank >= 50:
        regime = "Elevated"
    elif pct_rank >= 25:
        regime = "Normal"
    else:
        regime = "Low"

    return round(pct_rank, 1), regime


def _excess_kurtosis(arr: List[float]) -> float:
    """
    Excess kurtosis (Fisher definition: normal = 0) using numpy only.
    Requires at least 4 data points; returns 0.0 otherwise.
    """
    a = np.array(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 4:
        return 0.0
    mu  = np.mean(a)
    std = np.std(a, ddof=1)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((a - mu) / std) ** 4)) - 3.0


def _kurtosis_confidence_mult(raw_moves: List[float], min_events: int = 6) -> Tuple[float, float]:
    """
    Confidence multiplier from excess kurtosis of historical earnings moves.

    High kurtosis = fat tails = stock has history of extreme post-earnings gaps.
    Penalises confidence because the normal-distribution edge assumption breaks down.

    Returns (multiplier [0.55, 1.0], excess_kurtosis).
    """
    if len(raw_moves) < min_events:
        return 1.0, 0.0
    ek = _excess_kurtosis(raw_moves)
    if ek < 1.0:
        mult = 1.0                               # normal-ish distribution, no penalty
    elif ek < 3.0:
        mult = max(0.78, 1.0 - (ek - 1.0) * 0.11)  # 11pp per unit above 1.0
    else:
        mult = max(0.55, 0.78 - (ek - 3.0) * 0.046)  # heavier penalty for extreme fat tails
    return round(float(mult), 3), round(ek, 3)


def _classify_ticker_tier(market_cap_usd: Optional[float]) -> Tuple[str, float]:
    """
    Classify ticker by market cap into a tier with corresponding confidence multiplier.

    Institutional IV-crush patterns are strongest and most predictable in
    mega/large-cap names where earnings are well-modelled by analysts. Speculative
    small-caps have erratic IV dynamics and binary event risk that inflate false signals.

    Returns (tier_label, confidence_multiplier).
    """
    if market_cap_usd is None or not np.isfinite(market_cap_usd) or market_cap_usd <= 0:
        return "unknown", 0.88  # conservative default when market cap unavailable

    B = 1e9  # billion
    if market_cap_usd >= 200 * B:
        return "mega_cap",   1.00
    elif market_cap_usd >= 20 * B:
        return "large_cap",  0.95
    elif market_cap_usd >= 2 * B:
        return "mid_cap",    0.85
    elif market_cap_usd >= 300e6:
        return "small_cap",  0.72
    else:
        return "micro_cap",  0.55


def _crush_calibration_mult(
    raw_moves: List[float],
    implied_move_pct: float,
    min_events: int = 6,
) -> Tuple[float, float]:
    """
    Calibrate confidence using the historical crush realisation rate.

    Proxy: for each past earnings event, did the stock move LESS than the current
    implied move?  If options have historically overstated the move, that's evidence
    of genuine IV crush edge. If stocks have frequently blown past the implied move,
    the crush edge is suspect.

    Uses Laplace smoothing (Beta prior α=β=1) for stability with small samples.

    Returns (calibration_multiplier, historical_crush_rate [0–1]).
    """
    if not raw_moves or len(raw_moves) < min_events or not np.isfinite(implied_move_pct):
        return 1.0, float("nan")

    hits   = sum(1 for m in raw_moves if m < implied_move_pct)
    total  = len(raw_moves)
    # Laplace-smoothed rate
    rate   = (hits + 1) / (total + 2)

    if rate >= 0.72:
        mult = min(1.10, 0.95 + (rate - 0.72) * 0.55)  # historical crush confirms signal
    elif rate >= 0.50:
        mult = 1.0                                        # neutral — mixed or slight tendency
    elif rate >= 0.35:
        mult = max(0.83, 1.0 - (0.50 - rate) * 1.13)    # stock regularly exceeds implied
    else:
        mult = max(0.68, 0.83 - (0.35 - rate) * 1.0)    # stock consistently blows through IV

    return round(float(mult), 3), round(float(rate), 3)


def analyze_single_ticker(
    symbol: str,
    mda_client: Any = None,  # MarketDataClient | None
) -> EdgeSnapshot:
    """
    Full single-ticker IV crush edge analysis.

    Data strategy:
      - yfinance  : 6-month OHLCV for RV + 5-year close history for earnings-move anchors
      - MDApp     : live option chains (IV, Greeks, spread), earnings dates + BMO/AMC timing
      - Fallback  : if MDApp unavailable or returns empty, yfinance used for options/earnings too
    """
    from services.market_data_client import MarketDataClient  # late import to avoid circular

    clean_symbol = str(symbol or "").strip().upper()
    if not clean_symbol:
        raise ValueError("Symbol is required")

    # Determine data source availability
    client: Optional[MarketDataClient] = mda_client
    use_mda = isinstance(client, MarketDataClient) and client.is_available()
    data_source = "marketdata_app" if use_mda else "yfinance_fallback"

    # ── 1. Price history via yfinance (free, reliable for OHLCV) ─────────────
    ticker = yf.Ticker(clean_symbol)
    hist = ticker.history(period="6mo", auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"No market data for {clean_symbol}")

    close = pd.to_numeric(hist.get("Close"), errors="coerce").dropna()
    if close.empty:
        raise ValueError(f"No close prices for {clean_symbol}")

    # ── Staleness detection: warn if most recent bar is too old ──
    _price_stale = False
    _price_age_days: Optional[int] = None
    try:
        _last_bar_dt = hist.index[-1]
        if hasattr(_last_bar_dt, "tz") and _last_bar_dt.tz is not None:
            _now = datetime.now(timezone.utc)
        else:
            _now = datetime.utcnow()
        _price_age_days = (pd.Timestamp(_now) - pd.Timestamp(_last_bar_dt)).days
        if _price_age_days > 3:
            _price_stale = True
            logger.warning(
                "yfinance price data for %s is %d days stale (last bar: %s)",
                clean_symbol, _price_age_days, _last_bar_dt.strftime("%Y-%m-%d"),
            )
    except Exception as exc:
        logger.debug("Staleness check failed for %s: %s", clean_symbol, exc)

    # Ticker-tier classification — fetch market cap from yfinance (fast, cached)
    _market_cap: Optional[float] = None
    try:
        _info = ticker.info or {}
        _mc_raw = _info.get("marketCap") or _info.get("market_cap")
        if _mc_raw is not None:
            _market_cap = float(_mc_raw)
    except Exception as exc:
        logger.debug("Market cap fetch failed for %s: %s", clean_symbol, exc)
    ticker_tier, ticker_tier_mult = _classify_ticker_tier(_market_cap)

    current_price = float(close.iloc[-1])
    close_for_profile = close
    hist_long: Optional[pd.DataFrame] = None
    try:
        hist_long = ticker.history(period="5y", auto_adjust=True)
        if hist_long is not None and not hist_long.empty:
            close_long = pd.to_numeric(hist_long.get("Close"), errors="coerce").dropna()
            if not close_long.empty:
                close_for_profile = close_long
    except Exception as exc:
        logger.warning("5-year history fetch failed for %s (vol regime degraded): %s", clean_symbol, exc)
        hist_long = None

    # Yang-Zhang 30-day RV (primary — ~5x more efficient than close-to-close)
    rv30_yz = _yang_zhang_rv30(hist, window=30)
    rv30_cc = float(close.pct_change().dropna().tail(30).std(ddof=1) * np.sqrt(252))
    rv30 = rv30_yz if (np.isfinite(rv30_yz) and rv30_yz > 0.0) else rv30_cc
    rv30 = max(rv30, 1e-4)
    rv_estimator = "yang_zhang" if (np.isfinite(rv30_yz) and rv30_yz > 0.0) else "close_to_close"

    # HAR-RV 1-day-ahead forecast (Corsi 2009) — forward-looking RV for IV/RV comparison
    _rs_series = _rs_daily_vol_series(hist)
    rv_har_forecast: Optional[float] = _har_rv_forecast(_rs_series)

    # RV percentile rank + vol regime (uses 5y history for statistical depth)
    _rv_hist = hist_long if (hist_long is not None and not hist_long.empty) else hist
    rv_percentile_rank, vol_regime = _rv_percentile_and_regime(_rv_hist, rv30)

    # ── 2. Option chain: MDApp (one call, reused for term structure + curvature) ──
    chain_df: Optional[pd.DataFrame] = None
    if use_mda:
        try:
            chain_df = client.get_option_chain(
                clean_symbol,
                expiration="all",
                strike_limit=10,
            )
            if chain_df is None or chain_df.empty:
                logger.info("MDApp chain empty for %s, falling back to yfinance", clean_symbol)
                chain_df = None
                data_source = "yfinance_fallback"
        except Exception as exc:
            logger.warning("MDApp chain fetch failed for %s: %s", clean_symbol, exc)
            chain_df = None
            data_source = "yfinance_fallback"

    # ── 3. Term structure + smile curvature ──────────────────────────────────
    if chain_df is not None:
        (
            days, ivs, iv30, iv45, ts_slope_0_45, liquidity,
            implied_move_pct, near_term_spread_pct, near_term_dte,
            near_back_iv_ratio, near_term_liquidity_proxy,
            ts_slope_near_dte, ts_slope_far_dte,
        ) = _term_structure_from_mda_chain(chain_df, current_price)
        smile_state = _smile_curvature_from_mda_chain(chain_df, current_price)
    else:
        (
            days, ivs, iv30, iv45, ts_slope_0_45, liquidity,
            implied_move_pct, near_term_spread_pct, near_term_dte,
            near_back_iv_ratio, near_term_liquidity_proxy,
            ts_slope_near_dte, ts_slope_far_dte,
        ) = _term_structure_points_yf(ticker, current_price)
        smile_state = _smile_curvature_yf(ticker, current_price)
    options_source = "marketdata_app" if chain_df is not None else "yfinance"

    # ── 4. Earnings dates + BMO/AMC timing ───────────────────────────────────
    dte: Optional[int] = None
    earnings_release_time: Optional[str] = None
    earnings_events_for_profile: List[Dict[str, Any]] = []

    if use_mda:
        try:
            earnings_df = client.get_earnings(clean_symbol, countback=24)
            if earnings_df is not None and not earnings_df.empty:
                dte, earnings_release_time = _next_earnings_from_mda(earnings_df)
                earnings_events_for_profile = _earnings_dates_from_mda(earnings_df)
        except Exception as exc:
            logger.warning("MDApp earnings fetch failed for %s: %s", clean_symbol, exc)

    # yfinance fallback — two independent checks:
    #   1. If DTE not yet resolved (MDApp earnings returned nothing), use yfinance for next date.
    #   2. If historical events list is empty (MDApp only returns the next FUTURE date, not past
    #      history on this plan tier), always try yfinance for the move-profile events regardless
    #      of whether MDApp already provided the next-DTE.
    if dte is None:
        dte = _next_earnings_days_yf(ticker)
    if not earnings_events_for_profile:
        # Build earnings-event list from yfinance for the move profile
        get_dates_fn = getattr(ticker, "get_earnings_dates", None)
        if callable(get_dates_fn):
            try:
                edf = get_dates_fn(limit=24)
                if edf is not None and not edf.empty:
                    today_ts = pd.Timestamp(_utc_today_date())
                    parsed_events: List[Dict[str, Any]] = []
                    for ts in edf.index:
                        ts_raw = pd.Timestamp(ts).tz_localize(None)
                        ts_norm = ts_raw.normalize()
                        if ts_norm <= today_ts:
                            parsed_events.append(
                                {
                                    "event_date": ts_norm,
                                    "release_timing": _normalize_release_timing(ts_raw),
                                }
                            )
                    earnings_events_for_profile = parsed_events
            except Exception as exc:
                logger.warning("MDApp earnings event parsing failed for %s: %s", clean_symbol, exc)
        if not earnings_events_for_profile:
            fallback_edates = getattr(ticker, "earnings_dates", None)
            if fallback_edates is not None and not fallback_edates.empty:
                try:
                    today_ts = pd.Timestamp(_utc_today_date())
                    parsed_events = []
                    for ts in fallback_edates.index:
                        ts_raw = pd.Timestamp(ts).tz_localize(None)
                        ts_norm = ts_raw.normalize()
                        if ts_norm <= today_ts:
                            parsed_events.append(
                                {
                                    "event_date": ts_norm,
                                    "release_timing": _normalize_release_timing(ts_raw),
                                }
                            )
                    earnings_events_for_profile = parsed_events
                except Exception as exc:
                    logger.warning("yfinance earnings_dates parsing failed for %s: %s", clean_symbol, exc)

    # ATM BSM greeks — placed here because we now have dte resolved from section 4
    pricing_risk_free_rate, pricing_risk_free_rate_source = _get_pricing_risk_free_rate()
    _bsm_T = float(dte) if dte is not None and dte > 0 else np.nan
    greeks = (
        _bsm_greeks(
            S=current_price,
            T_days=_bsm_T,
            sigma=float(iv30) if np.isfinite(iv30) else np.nan,
            r=pricing_risk_free_rate,
        )
        if np.isfinite(_bsm_T) and np.isfinite(iv30)
        else {k: None for k in ("delta_call", "delta_put", "gamma", "vega", "theta_call", "theta_put")}
    )

    # ── 5. Historical earnings move profile (uses yfinance price series) ─────
    move_profile = _historical_earnings_move_profile(close_for_profile, earnings_events_for_profile)
    median_earnings_move_pct = _safe_float(move_profile.get("median_move_pct"), np.nan)
    p90_earnings_move_pct = _safe_float(move_profile.get("p90_move_pct"), np.nan)
    avg_last4_move_pct = _safe_float(move_profile.get("avg_last4_move_pct"), np.nan)
    move_std_pct = _safe_float(move_profile.get("std_move_pct"), np.nan)
    move_anchor_pct = _compute_move_anchor(median_earnings_move_pct, avg_last4_move_pct)
    move_anchor_pct_val = _safe_float(move_anchor_pct, np.nan)
    move_sample_size = int(move_profile.get("event_count", 0) or 0)
    move_source = str(move_profile.get("source", "none"))

    if move_source == "earnings_history":
        sample_confidence = float(np.clip(move_sample_size / 8.0, 0.20, 1.0))
    elif move_source == "daily_fallback":
        sample_confidence = float(np.clip(move_sample_size / 40.0, 0.10, 0.60))
    else:
        sample_confidence = 0.10

    move_uncertainty_pct = _compute_move_uncertainty_pct(
        move_std_pct=move_std_pct,
        sample_size=move_sample_size,
        move_source=move_source,
    )

    # ── 6. Scoring (unchanged formulas) ──────────────────────────────────────
    tx_cost_pct = _estimate_transaction_cost_pct(
        liquidity=liquidity, spread_pct=near_term_spread_pct
    )
    iv_rv = float(iv30 / rv30) if np.isfinite(iv30) and np.isfinite(rv30) and rv30 > 0 else np.nan
    setup_base = _score_setup(iv_rv=iv_rv, ts_slope=ts_slope_0_45, dte=dte, liquidity=liquidity)
    smile_curvature = _safe_float(smile_state.get("curvature"), np.nan)
    concavity_component = (
        float(np.clip((-smile_curvature - 0.5) / 2.5, 0.0, 1.0))
        if np.isfinite(smile_curvature) else 0.50
    )
    setup_score = float(np.clip(0.88 * setup_base + 0.12 * concavity_component, 0.0, 1.0))

    implied_move_total_pct = _safe_float(implied_move_pct, np.nan)
    rv_for_event = _safe_float(rv_har_forecast, rv30)
    non_event_move_pct = np.nan
    event_implied_move_pct = implied_move_total_pct
    if (
        np.isfinite(implied_move_total_pct)
        and near_term_dte is not None
        and np.isfinite(rv_for_event)
        and rv_for_event > 0
    ):
        non_event_days = max(int(near_term_dte) - 1, 0)
        non_event_move_pct = float(rv_for_event * np.sqrt(non_event_days / 252.0) * 100.0)
        event_implied_move_pct = float(
            np.sqrt(max(implied_move_total_pct * implied_move_total_pct - non_event_move_pct * non_event_move_pct, 0.0))
        )

    raw_gross_edge_pct = (
        float(event_implied_move_pct - move_anchor_pct_val)
        if np.isfinite(event_implied_move_pct) and np.isfinite(move_anchor_pct_val)
        else np.nan
    )
    implied_vs_anchor_ratio = (
        float(event_implied_move_pct / move_anchor_pct_val)
        if np.isfinite(event_implied_move_pct)
        and np.isfinite(move_anchor_pct_val) and move_anchor_pct_val > 0
        else np.nan
    )
    confidence_adjusted_gross_edge_pct = (
        float(raw_gross_edge_pct * sample_confidence)
        if np.isfinite(raw_gross_edge_pct) else np.nan
    )
    uncertainty_penalty_pct = (
        float((move_uncertainty_pct or 0.0) * (0.60 + 0.40 * (1.0 - sample_confidence)))
        if move_uncertainty_pct is not None
        else (0.35 if move_source != "earnings_history" else 0.0)
    )
    expected_gross_edge_pct = (
        float(confidence_adjusted_gross_edge_pct - uncertainty_penalty_pct)
        if np.isfinite(confidence_adjusted_gross_edge_pct) else np.nan
    )
    expected_net_edge_pct = (
        float(expected_gross_edge_pct - tx_cost_pct)
        if np.isfinite(expected_gross_edge_pct) else np.nan
    )
    base_drawdown_risk_pct = (
        float(max(p90_earnings_move_pct - event_implied_move_pct, 0.0) + tx_cost_pct)
        if np.isfinite(p90_earnings_move_pct) and np.isfinite(event_implied_move_pct)
        else float(0.75 + tx_cost_pct)
    )
    concavity_risk_surcharge_pct = float(
        (0.55 * concavity_component) if bool(smile_state.get("concave"))
        else max(0.0, (concavity_component - 0.60) * 0.20)
    )
    drawdown_risk_pct = float(
        base_drawdown_risk_pct + (1.0 - sample_confidence) * 2.50 + concavity_risk_surcharge_pct
    )
    expectancy_ratio = (
        float(expected_net_edge_pct / max(drawdown_risk_pct, 0.25))
        if np.isfinite(expected_net_edge_pct) else np.nan
    )
    expectancy_score = _score_expectancy(
        net_edge_pct=expected_net_edge_pct,
        drawdown_risk_pct=drawdown_risk_pct,
        sample_size=int(move_profile.get("event_count", 0) or 0),
    )
    composite_score = float(np.clip(0.60 * setup_score + 0.40 * expectancy_score, 0.0, 1.0))
    confidence_pct_raw = float(np.clip(30.0 + 70.0 * composite_score, 0.0, 100.0))

    # ── Grading refinements: apply calibration multipliers to confidence_pct ──

    # Fix 2: kurtosis penalty — fat-tailed movers lose confidence
    _raw_moves: List[float] = move_profile.get("raw_moves_pct") or []
    kurtosis_conf_mult, move_kurtosis = _kurtosis_confidence_mult(_raw_moves)

    # Fix 5: historical crush calibration — did the stock historically stay inside IV?
    _implied_for_crush = (
        float(implied_move_total_pct)
        if np.isfinite(_safe_float(implied_move_total_pct, np.nan))
        else float(move_anchor_pct_val) if np.isfinite(_safe_float(move_anchor_pct_val, np.nan))
        else float("nan")
    )
    crush_calibration_mult, hist_crush_rate = _crush_calibration_mult(_raw_moves, _implied_for_crush)

    # Fix 1: ticker tier multiplier — speculative small-caps get confidence haircut
    # ML: optional ±15% nudge from calibrated logistic regression (crush probability)
    _ml_crush_prob, _ml_mult = _ml_crush_probability(
        near_iv=_safe_float(iv30, 0.30),
        near_back_ratio=_safe_float(near_back_iv_ratio, 1.10),
        iv_rv=_safe_float(iv_rv, 1.10),
    )
    # Combined calibration factor — include ML multiplier when model is loaded
    _calibration_mult = float(np.clip(
        ticker_tier_mult * kurtosis_conf_mult * crush_calibration_mult * _ml_mult,
        0.40, 1.15,
    ))
    confidence_pct = float(np.clip(confidence_pct_raw * _calibration_mult, 0.0, 100.0))

    # Calendar spread P&L diagram (ATM call calendar — short near, long back+28d)
    _cal_iv_back = (
        float(iv45) if (np.isfinite(iv45) and iv45 > 0)
        else (float(iv30) * 0.88 if (np.isfinite(iv30) and iv30 > 0) else np.nan)
    )
    _cal_T_near = float(near_term_dte) if (near_term_dte is not None and near_term_dte > 0) else np.nan
    _cal_T_back = _cal_T_near + 28.0 if np.isfinite(_cal_T_near) else np.nan
    calendar_payoff: Optional[Dict[str, Any]] = (
        _calendar_spread_payoff(
            S=current_price,
            iv_near=float(iv30),
            iv_back=float(_cal_iv_back),
            T_near_days=float(_cal_T_near),
            T_back_days=float(_cal_T_back),
            r=pricing_risk_free_rate,
        )
        if (np.isfinite(_cal_T_near) and np.isfinite(_cal_T_back)
            and np.isfinite(iv30) and iv30 > 0
            and np.isfinite(_cal_iv_back) and _cal_iv_back > 0)
        else None
    )

    # Fix 4: Calendar spread viability based on near/back IV ratio + implied-move vs breakevens
    _nb_ratio = _safe_float(near_back_iv_ratio, np.nan)
    if not np.isfinite(_nb_ratio):
        calendar_spread_quality = "unknown"
    elif _nb_ratio >= 1.25:
        calendar_spread_quality = "Strong"
    elif _nb_ratio >= 1.15:
        calendar_spread_quality = "Moderate"
    elif _nb_ratio >= 1.05:
        calendar_spread_quality = "Weak"
    else:
        calendar_spread_quality = "Poor"

    # How much of the implied move do the calendar breakevens cover?
    _cal_be_vs_implied: Optional[float] = None
    if (
        calendar_payoff is not None
        and calendar_payoff.get("breakeven_moves_pct")
        and np.isfinite(_safe_float(implied_move_total_pct, np.nan))
    ):
        _be_list = calendar_payoff["breakeven_moves_pct"]
        _be_half_width = float(np.mean([abs(b) for b in _be_list]))
        _impl = float(implied_move_total_pct)
        if _impl > 0:
            _cal_be_vs_implied = round(_be_half_width / _impl, 3)

    # Kelly position sizing (computed after all edge/risk values are finalised)
    kelly_full_pct, kelly_half_pct = _kelly_sizing(
        expected_net_edge_pct=expected_net_edge_pct,
        drawdown_risk_pct=drawdown_risk_pct,
        sample_confidence=sample_confidence,
    )

    # ── 7. Hard gates (unchanged) ─────────────────────────────────────────────
    hard_gate_reasons: List[str] = []
    implied_move_val = _safe_float(implied_move_total_pct, np.nan)
    if dte is None:
        hard_gate_reasons.append("Days to earnings unavailable.")
    elif dte < 1:
        hard_gate_reasons.append("Entry window closed (DTE < 1).")
    if near_term_dte is None:
        hard_gate_reasons.append("Near-term option expiry unavailable.")
    elif near_term_dte < MIN_SHORT_LEG_DTE:
        hard_gate_reasons.append(f"Near-term option expiry too close ({near_term_dte} DTE < {MIN_SHORT_LEG_DTE}).")
    if dte is not None and near_term_dte is not None and dte >= near_term_dte:
        hard_gate_reasons.append("Earnings event is not before the short-leg expiry.")
    if not np.isfinite(iv30):
        hard_gate_reasons.append("IV30 unavailable — no option chain data.")
    if not np.isfinite(implied_move_val):
        hard_gate_reasons.append("Near-term implied move unavailable from quotes.")
    spread_val = _safe_float(near_term_spread_pct, np.nan)
    if np.isfinite(spread_val) and spread_val > MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE:
        hard_gate_reasons.append(
            f"Near-term spread too wide ({spread_val:.1f}% > {MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE:.1f}%)."
        )
    near_back_ratio_val = _safe_float(near_back_iv_ratio, np.nan)
    if np.isfinite(near_back_ratio_val) and near_back_ratio_val < MIN_NEAR_BACK_IV_RATIO_FOR_EVENT:
        hard_gate_reasons.append(
            f"Near/back IV premium too weak ({near_back_ratio_val:.2f}x < {MIN_NEAR_BACK_IV_RATIO_FOR_EVENT:.2f}x)."
        )
    near_term_liquidity_val = _safe_float(near_term_liquidity_proxy, np.nan)
    if np.isfinite(near_term_liquidity_val) and near_term_liquidity_val < MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_TRADE:
        hard_gate_reasons.append(
            f"Near-term liquidity too low ({near_term_liquidity_val:.0f} < {MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_TRADE:.0f})."
        )
    if move_source != "earnings_history":
        hard_gate_reasons.append("Earnings move profile is not from true earnings history.")
    if move_sample_size < MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL:
        hard_gate_reasons.append(
            f"Insufficient earnings events ({move_sample_size}/{MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL})."
        )

    hard_no_trade = len(hard_gate_reasons) > 0
    confidence_capped = False
    if hard_no_trade and confidence_pct > HARD_NO_TRADE_CONFIDENCE_CAP_PCT:
        confidence_pct = HARD_NO_TRADE_CONFIDENCE_CAP_PCT
        confidence_capped = True

    # ── 8. Recommendation (unchanged thresholds) ──────────────────────────────
    recommendation = "Pass"
    if hard_no_trade:
        recommendation = "No Trade"
    elif (
        np.isfinite(expected_net_edge_pct) and expected_net_edge_pct >= 0.25
        and np.isfinite(expectancy_ratio) and expectancy_ratio >= 0.20
        and np.isfinite(implied_vs_anchor_ratio) and implied_vs_anchor_ratio >= 1.05
        and sample_confidence >= 0.30
        and np.isfinite(iv_rv) and iv_rv >= 1.05
        and dte is not None and 1 <= dte <= 21
        and setup_score >= 0.56
        and confidence_pct >= 62.0
    ):
        recommendation = "Consider"
    elif setup_score >= 0.50 and (not np.isfinite(expected_net_edge_pct) or expected_net_edge_pct > 0.0):
        recommendation = "Watchlist"

    edge_quality = "Insufficient pricing context"
    if hard_no_trade:
        edge_quality = "Hard no-trade gate"
    elif np.isfinite(expected_net_edge_pct):
        if sample_confidence < 0.30:
            edge_quality = "Low-confidence edge"
        elif expected_net_edge_pct >= 0.50 and np.isfinite(expectancy_ratio) and expectancy_ratio >= 0.30:
            edge_quality = "Positive edge"
        elif expected_net_edge_pct > 0.0:
            edge_quality = "Marginal edge"
        else:
            edge_quality = "Negative expectancy"

    # ── 9. Rationale strings ──────────────────────────────────────────────────
    rt_label = ""
    if earnings_release_time:
        rt_map = {
            "before market open": " (BMO)",
            "after market close": " (AMC)",
            "during market hours": " (intraday)",
        }
        rt_label = rt_map.get(earnings_release_time, f" ({earnings_release_time})")

    iv_rv_har = (
        float(iv30 / rv_har_forecast)
        if rv_har_forecast is not None and np.isfinite(iv30) and rv_har_forecast > 0
        else None
    )

    rationale = [
        (
            f"IV/RV30={iv_rv:.2f} [YZ] / IV/RV(HAR fwd)={iv_rv_har:.2f}"
            if np.isfinite(iv_rv) and iv_rv_har is not None
            else f"IV/RV30={iv_rv:.2f}" if np.isfinite(iv_rv) else "IV/RV30 unavailable from chain snapshot."
        ),
        f"Term slope(0-45)={ts_slope_0_45:.4f}" if np.isfinite(ts_slope_0_45) else "Term slope unavailable.",
        f"Days to earnings={dte}{rt_label}" if dte is not None else "Days to earnings unavailable.",
        f"Near-term option DTE={near_term_dte}" if near_term_dte is not None else "Near-term option DTE unavailable.",
        (
            f"Near-term smile curvature={smile_curvature:.3f} "
            f"({'concave' if smile_state.get('concave') else 'not concave'})."
            if np.isfinite(smile_curvature) else "Near-term smile curvature unavailable."
        ),
        (
            f"Near-term spread={near_term_spread_pct:.2f}% "
            f"(gate <= {MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE:.1f}%)."
            if np.isfinite(_safe_float(near_term_spread_pct, np.nan))
            else "Near-term spread unavailable."
        ),
        (
            f"Near/back IV premium={near_back_iv_ratio:.2f}x "
            f"(gate >= {MIN_NEAR_BACK_IV_RATIO_FOR_EVENT:.2f}x)."
            if np.isfinite(_safe_float(near_back_iv_ratio, np.nan))
            else "Near/back IV premium unavailable."
        ),
        (
            f"Near-term liquidity proxy={near_term_liquidity_proxy:.0f} "
            f"(gate >= {MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_TRADE:.0f})."
            if np.isfinite(_safe_float(near_term_liquidity_proxy, np.nan))
            else "Near-term liquidity proxy unavailable."
        ),
        (
            f"Hard gate status=FAIL ({'; '.join(hard_gate_reasons)})"
            if hard_no_trade else "Hard gate status=PASS."
        ),
        (
            f"Expected net edge={expected_net_edge_pct:+.2f}% "
            f"(gross {expected_gross_edge_pct:+.2f}% after sample/uncertainty adjustment - cost {tx_cost_pct:.2f}%; "
            f"event-implied={event_implied_move_pct:.2f}% from total-implied={implied_move_total_pct:.2f}%"
            + (
                f" minus non-event={non_event_move_pct:.2f}%; "
                if np.isfinite(non_event_move_pct)
                else "; "
            )
            + f"implied/anchor={implied_vs_anchor_ratio:.2f}x)."
            if np.isfinite(expected_net_edge_pct)
            and np.isfinite(expected_gross_edge_pct)
            and np.isfinite(implied_vs_anchor_ratio)
            else f"Expected net edge unavailable; cost baseline={tx_cost_pct:.2f}%."
        ),
        (
            f"Anchor uncertainty penalty={uncertainty_penalty_pct:.2f}% "
            f"(move std={move_std_pct:.2f}%, source={move_source})."
            if np.isfinite(_safe_float(move_std_pct, np.nan))
            else "Anchor uncertainty unavailable."
        ),
        (
            f"Drawdown risk estimate={drawdown_risk_pct:.2f}% "
            f"(expectancy ratio={expectancy_ratio:.2f}; concavity surcharge={concavity_risk_surcharge_pct:.2f}%)."
            if np.isfinite(expectancy_ratio)
            else f"Drawdown risk estimate={drawdown_risk_pct:.2f}%."
        ),
        f"Evidence confidence={sample_confidence:.2f} ({move_sample_size} moves, source={move_source}).",
        f"Edge quality: {edge_quality}.",
        f"Liquidity proxy={liquidity:.0f} (ATM OI/vol aggregate).",
        f"Data source: {data_source}.",
        (
            f"RV model: {rv_estimator} rv30={rv30:.4f}; "
            f"HAR fwd={rv_har_forecast:.4f} (Corsi 2009 on Rogers-Satchell daily series)."
            if rv_har_forecast is not None
            else f"RV model: {rv_estimator} rv30={rv30:.4f}; HAR forecast unavailable (< 35 days history)."
        ),
    ]

    # ── 10. Metrics dict ──────────────────────────────────────────────────────
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        # FIX 4: granular source provenance — options and price/RV may differ
        "data_sources": {
            "options_source":  options_source,
            "price_rv_source": "yfinance",
        },
        "price_data_stale": _price_stale,
        "price_data_age_days": _price_age_days,
        "current_price": current_price,
        "pricing_risk_free_rate": float(pricing_risk_free_rate),
        "pricing_risk_free_rate_source": pricing_risk_free_rate_source,
        "days_to_earnings": dte,
        "earnings_release_time": earnings_release_time,
        "rv30": rv30,
        "iv30": float(iv30) if np.isfinite(iv30) else None,
        "iv45": float(iv45) if np.isfinite(iv45) else None,
        "iv_rv30": float(iv_rv) if np.isfinite(iv_rv) else None,
        "setup_base_score": float(setup_base),
        "concavity_component": float(concavity_component),
        "smile_curvature": float(smile_curvature) if np.isfinite(smile_curvature) else None,
        "smile_concave": bool(smile_state.get("concave", False)),
        "smile_points": int(smile_state.get("points", 0) or 0),
        "term_structure_slope_0_45": float(ts_slope_0_45) if np.isfinite(ts_slope_0_45) else None,
        # FIX 1: actual tenors used for slope — 0-45 label was misleading
        "term_structure_slope":   float(ts_slope_0_45) if np.isfinite(ts_slope_0_45) else None,
        "ts_slope_near_dte":      float(ts_slope_near_dte) if ts_slope_near_dte is not None else None,
        "ts_slope_far_dte":       float(ts_slope_far_dte) if ts_slope_far_dte is not None else None,
        "near_back_iv_ratio": float(near_back_iv_ratio) if np.isfinite(_safe_float(near_back_iv_ratio, np.nan)) else None,
        "min_near_back_iv_ratio_for_event": MIN_NEAR_BACK_IV_RATIO_FOR_EVENT,
        "near_term_dte": near_term_dte,
        "near_term_liquidity_proxy": (
            float(near_term_liquidity_proxy)
            if np.isfinite(_safe_float(near_term_liquidity_proxy, np.nan)) else None
        ),
        "min_near_term_liquidity_proxy_for_trade": MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_TRADE,
        "implied_move_pct": float(implied_move_total_pct) if np.isfinite(implied_move_total_pct) else None,
        "event_implied_move_pct": float(event_implied_move_pct) if np.isfinite(event_implied_move_pct) else None,
        "non_event_move_pct": float(non_event_move_pct) if np.isfinite(non_event_move_pct) else None,
        "earnings_move_median_pct": float(median_earnings_move_pct) if np.isfinite(median_earnings_move_pct) else None,
        "earnings_move_p90_pct": float(p90_earnings_move_pct) if np.isfinite(p90_earnings_move_pct) else None,
        "earnings_move_avg_last4_pct": float(avg_last4_move_pct) if np.isfinite(avg_last4_move_pct) else None,
        "earnings_move_std_pct": float(move_std_pct) if np.isfinite(move_std_pct) else None,
        "earnings_move_anchor_pct": float(move_anchor_pct_val) if np.isfinite(move_anchor_pct_val) else None,
        "move_uncertainty_pct": float(move_uncertainty_pct) if move_uncertainty_pct is not None else None,
        "implied_vs_anchor_ratio": float(implied_vs_anchor_ratio) if np.isfinite(implied_vs_anchor_ratio) else None,
        "earnings_move_sample_size": move_sample_size,
        "earnings_move_source": move_source,
        "sample_confidence": sample_confidence,
        "min_required_earnings_events": MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL,
        "min_short_leg_dte": MIN_SHORT_LEG_DTE,
        "max_near_term_spread_pct_for_trade": MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE,
        "hard_no_trade": hard_no_trade,
        "hard_gate_reasons": hard_gate_reasons,
        "confidence_capped": confidence_capped,
        "confidence_cap_pct": HARD_NO_TRADE_CONFIDENCE_CAP_PCT,
        "tx_cost_estimate_pct": tx_cost_pct,
        "raw_gross_edge_pct": float(raw_gross_edge_pct) if np.isfinite(raw_gross_edge_pct) else None,
        "confidence_adjusted_gross_edge_pct": (
            float(confidence_adjusted_gross_edge_pct)
            if np.isfinite(confidence_adjusted_gross_edge_pct) else None
        ),
        "uncertainty_penalty_pct": float(uncertainty_penalty_pct),
        "expected_gross_edge_pct": float(expected_gross_edge_pct) if np.isfinite(expected_gross_edge_pct) else None,
        "expected_net_edge_pct": float(expected_net_edge_pct) if np.isfinite(expected_net_edge_pct) else None,
        "base_drawdown_risk_pct": float(base_drawdown_risk_pct),
        "concavity_risk_surcharge_pct": float(concavity_risk_surcharge_pct),
        "drawdown_risk_pct": float(drawdown_risk_pct),
        "expectancy_ratio": float(expectancy_ratio) if np.isfinite(expectancy_ratio) else None,
        "expectancy_score": float(expectancy_score),
        "composite_score": float(composite_score),
        "edge_quality": edge_quality,
        "liquidity_proxy": float(liquidity),
        "near_term_spread_pct": float(near_term_spread_pct) if np.isfinite(_safe_float(near_term_spread_pct, np.nan)) else None,
        "term_structure_days": [float(v) for v in days],
        "term_structure_ivs": [float(v) for v in ivs],
        # ── RV model (Yang-Zhang + HAR-RV) ───────────────────────────────────
        "rv_estimator": rv_estimator,
        "rv30_har_forecast": float(rv_har_forecast) if rv_har_forecast is not None else None,
        "iv_rv_har": float(iv_rv_har) if iv_rv_har is not None else None,
        # ── Vol regime (percentile rank within 252-day rolling history) ───────
        "rv_percentile_rank": rv_percentile_rank,
        "vol_regime": vol_regime,
        # ── BSM ATM greeks (tenor = earnings DTE, σ = iv30) ──────────────────
        "atm_delta_call": greeks.get("delta_call"),
        "atm_delta_put": greeks.get("delta_put"),
        "atm_gamma": greeks.get("gamma"),
        "atm_vega": greeks.get("vega"),
        "atm_theta_call": greeks.get("theta_call"),
        "atm_theta_put": greeks.get("theta_put"),
        # ── Kelly position sizing (% of portfolio to risk) ───────────────────
        "kelly_full_pct": kelly_full_pct,
        "kelly_half_pct": kelly_half_pct,
        # ── Calendar spread P&L diagram ────────────────────────────────────
        "calendar_payoff": calendar_payoff,
        # Fix 4: calendar spread viability
        "calendar_spread_quality": calendar_spread_quality,
        "calendar_be_vs_implied": _cal_be_vs_implied,
        # Fix 1: ticker tier
        "ticker_tier": ticker_tier,
        "ticker_tier_mult": ticker_tier_mult,
        "market_cap_usd": _market_cap,
        # Fix 2: kurtosis penalty
        "move_kurtosis": move_kurtosis if move_kurtosis != 0.0 else None,
        "kurtosis_conf_mult": kurtosis_conf_mult,
        # Fix 5: historical crush calibration
        "hist_crush_rate": hist_crush_rate if np.isfinite(hist_crush_rate) else None,
        "crush_calibration_mult": crush_calibration_mult,
        # ML crush probability (None when model not yet trained)
        "ml_crush_prob": round(_ml_crush_prob, 3) if _ml_crush_prob is not None else None,
        # Combined confidence calibration
        "confidence_calibration_mult": round(_calibration_mult, 3),
        "confidence_pct_raw": round(confidence_pct_raw, 2),
    }

    return EdgeSnapshot(
        symbol=clean_symbol,
        recommendation=recommendation,
        confidence_pct=confidence_pct,
        setup_score=setup_score,
        metrics=metrics,
        rationale=rationale,
    )
