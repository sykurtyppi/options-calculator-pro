from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL = 8
HARD_NO_TRADE_CONFIDENCE_CAP_PCT = 55.0
TS_SLOPE_TARGET = -0.004
TS_SLOPE_BAND = 0.025
MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE = 18.0
MIN_SHORT_LEG_DTE = 2


@dataclass
class EdgeSnapshot:
    symbol: str
    recommendation: str
    confidence_pct: float
    setup_score: float
    metrics: Dict[str, Any]
    rationale: List[str]


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


def _nearest_atm_option_stats(chain_df: pd.DataFrame, current_price: float) -> Dict[str, Optional[float]]:
    if chain_df is None or chain_df.empty:
        return {"iv": None, "oi": None, "volume": None, "mid": None, "spread_pct": None}
    df = chain_df.copy()
    df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
    df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility"), errors="coerce")
    df["openInterest"] = pd.to_numeric(df.get("openInterest"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce")
    df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce")
    df["lastPrice"] = pd.to_numeric(df.get("lastPrice"), errors="coerce")
    df = df.dropna(subset=["strike"])
    if df.empty:
        return {"iv": None, "oi": None, "volume": None, "mid": None, "spread_pct": None}

    df["distance"] = (df["strike"] - float(current_price)).abs()
    row = df.sort_values("distance", ascending=True).iloc[0]
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


def _term_structure_points(
    ticker: yf.Ticker,
    current_price: float,
    max_expiries: int = 6,
) -> tuple[List[float], List[float], float, float, float, float, Optional[float], Optional[float], Optional[int]]:
    days: List[float] = []
    ivs: List[float] = []
    oi_values: List[float] = []
    vol_values: List[float] = []
    near_term_implied_move_pct: Optional[float] = None
    near_term_spread_pct: Optional[float] = None
    near_term_dte: Optional[int] = None

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
            call_iv = call_stats["iv"]
            put_iv = put_stats["iv"]

            iv_candidates = [v for v in (call_iv, put_iv) if v is not None and np.isfinite(v) and v > 0]
            if not iv_candidates:
                continue
            iv_atm = float(np.mean(iv_candidates))

            days.append(dte)
            ivs.append(iv_atm)
            oi_values.append(float(max(call_stats["oi"] or 0.0, 0.0) + max(put_stats["oi"] or 0.0, 0.0)))
            vol_values.append(float(max(call_stats["volume"] or 0.0, 0.0) + max(put_stats["volume"] or 0.0, 0.0)))

            if near_term_implied_move_pct is None:
                call_mid = _safe_float(call_stats["mid"], np.nan)
                put_mid = _safe_float(put_stats["mid"], np.nan)
                if np.isfinite(call_mid) and np.isfinite(put_mid) and current_price > 0:
                    near_term_implied_move_pct = float(((call_mid + put_mid) / current_price) * 100.0)
                    near_term_dte = int(dte)

                spread_candidates = [call_stats.get("spread_pct"), put_stats.get("spread_pct")]
                spread_candidates = [float(v) for v in spread_candidates if v is not None and np.isfinite(v) and v >= 0]
                if spread_candidates:
                    near_term_spread_pct = float(np.mean(spread_candidates))
        except Exception:
            continue

    if len(days) < 2:
        return days, ivs, np.nan, np.nan, 0.0, 0.0, near_term_implied_move_pct, near_term_spread_pct, near_term_dte

    order = np.argsort(np.array(days, dtype=float))
    days_arr = np.array(days, dtype=float)[order]
    ivs_arr = np.array(ivs, dtype=float)[order]

    iv30 = float(np.interp(30.0, days_arr, ivs_arr))
    iv45 = float(np.interp(45.0, days_arr, ivs_arr))
    iv0 = float(np.interp(0.0, days_arr, ivs_arr))
    slope_0_45 = float((iv45 - iv0) / 45.0)

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
    )


def _next_earnings_days(ticker: yf.Ticker) -> Optional[int]:
    today = _utc_today_date()
    get_dates = getattr(ticker, "get_earnings_dates", None)
    if callable(get_dates):
        try:
            earnings_df = get_dates(limit=8)
            if earnings_df is not None and not earnings_df.empty:
                for idx in earnings_df.index:
                    event_date = pd.Timestamp(idx).date()
                    days = (event_date - today).days
                    if days >= 0:
                        return int(days)
        except Exception:
            pass

    fallback = getattr(ticker, "earnings_dates", None)
    if fallback is not None and not fallback.empty:
        try:
            for idx in fallback.index:
                event_date = pd.Timestamp(idx).date()
                days = (event_date - today).days
                if days >= 0:
                    return int(days)
        except Exception:
            pass
    return None


def _score_setup(iv_rv: float, ts_slope: float, dte: Optional[int], liquidity: float) -> float:
    iv_component = float(np.clip((iv_rv - 0.90) / 0.70, 0.0, 1.0)) if np.isfinite(iv_rv) else 0.40
    ts_component = (
        float(np.clip(np.exp(-0.5 * ((ts_slope - TS_SLOPE_TARGET) / TS_SLOPE_BAND) ** 2), 0.0, 1.0))
        if np.isfinite(ts_slope) else 0.50
    )
    if dte is None:
        timing_component = 0.45
    else:
        timing_component = float(np.clip(np.exp(-0.5 * ((float(dte) - 7.0) / 6.0) ** 2), 0.0, 1.0))
    liq_component = float(np.clip((np.log1p(max(liquidity, 1.0)) - np.log1p(500.0)) / (np.log1p(25000.0) - np.log1p(500.0)), 0.0, 1.0))

    score = 0.36 * iv_component + 0.24 * ts_component + 0.24 * timing_component + 0.16 * liq_component
    return float(np.clip(score, 0.0, 1.0))


def _short_expiry_smile_curvature(ticker: yf.Ticker, current_price: float) -> Dict[str, Any]:
    """
    Estimate near-term IV smile curvature around ATM.
    Positive curvature => convex smile (usual case).
    Negative curvature => concave smile (event-risk pattern).
    """
    expirations = list(getattr(ticker, "options", []) or [])
    if not expirations:
        return {"curvature": None, "concave": False, "points": 0}

    for exp in expirations[:3]:
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
        except Exception:
            continue

        prepared_frames: List[pd.DataFrame] = []
        for frame in (calls, puts):
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

        if not iv_by_strike:
            continue

        rows: List[tuple[float, float]] = []
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
        # Quadratic fit y = a x^2 + b x + c, curvature sign is driven by a.
        try:
            a, _, _ = np.polyfit(x, y, 2)
        except Exception:
            continue

        curvature = float(a)
        concave = bool(curvature < -0.5)
        return {"curvature": curvature, "concave": concave, "points": int(len(rows))}

    return {"curvature": None, "concave": False, "points": 0}


def _historical_earnings_move_profile(ticker: yf.Ticker, close: pd.Series) -> Dict[str, Any]:
    if close.empty:
        return {
            "event_count": 0,
            "median_move_pct": None,
            "p90_move_pct": None,
            "avg_last4_move_pct": None,
            "source": "none",
        }

    price_series = close.copy()
    if isinstance(price_series.index, pd.DatetimeIndex):
        idx = price_series.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        price_series.index = idx.normalize()
    price_series = price_series[~price_series.index.duplicated(keep="last")].sort_index()
    index_arr = price_series.index.to_numpy()
    if len(index_arr) < 10:
        return {
            "event_count": 0,
            "median_move_pct": None,
            "p90_move_pct": None,
            "avg_last4_move_pct": None,
            "source": "none",
        }

    earnings_dates: List[pd.Timestamp] = []
    get_dates = getattr(ticker, "get_earnings_dates", None)
    if callable(get_dates):
        try:
            frame = get_dates(limit=24)
            if frame is not None and not frame.empty:
                earnings_dates = [pd.Timestamp(ts).tz_localize(None).normalize() for ts in frame.index]
        except Exception:
            pass
    if not earnings_dates:
        fallback = getattr(ticker, "earnings_dates", None)
        if fallback is not None and not fallback.empty:
            try:
                earnings_dates = [pd.Timestamp(ts).tz_localize(None).normalize() for ts in fallback.index]
            except Exception:
                earnings_dates = []

    today = pd.Timestamp(_utc_today_date())
    past_events = sorted({ts for ts in earnings_dates if ts <= today})
    event_moves: List[float] = []

    for event_ts in past_events[-16:]:
        pre_loc = int(index_arr.searchsorted(event_ts.to_datetime64(), side="left")) - 1
        post_loc = int(index_arr.searchsorted((event_ts + timedelta(days=1)).to_datetime64(), side="left"))
        if pre_loc < 0 or post_loc >= len(index_arr):
            continue
        pre_px = _safe_float(price_series.iloc[pre_loc], np.nan)
        post_px = _safe_float(price_series.iloc[post_loc], np.nan)
        if not np.isfinite(pre_px) or not np.isfinite(post_px) or pre_px <= 0:
            continue
        move_pct = abs((post_px - pre_px) / pre_px) * 100.0
        if np.isfinite(move_pct):
            event_moves.append(float(move_pct))

    if not event_moves:
        # Fallback to recent daily absolute moves if earnings history is unavailable.
        daily_moves = (
            price_series.pct_change()
            .abs()
            .dropna()
            .tail(126)
            .to_numpy(dtype=float)
            * 100.0
        )
        if daily_moves.size == 0:
            return {
                "event_count": 0,
                "median_move_pct": None,
                "p90_move_pct": None,
                "avg_last4_move_pct": None,
                "source": "none",
            }
        if daily_moves.size >= 5:
            low_clip, high_clip = np.percentile(daily_moves, [1.0, 99.0])
            daily_moves = np.clip(daily_moves, low_clip, high_clip)
        avg_last4 = float(np.mean(daily_moves[-4:]))
        return {
            "event_count": int(daily_moves.size),
            "median_move_pct": float(np.median(daily_moves)),
            "p90_move_pct": float(np.percentile(daily_moves, 90)),
            "avg_last4_move_pct": avg_last4,
            "source": "daily_fallback",
        }

    moves = np.array(event_moves, dtype=float)
    if moves.size >= 5:
        low_clip, high_clip = np.percentile(moves, [1.0, 99.0])
        moves = np.clip(moves, low_clip, high_clip)
    avg_last4 = float(np.mean(moves[-4:]))
    return {
        "event_count": int(moves.size),
        "median_move_pct": float(np.median(moves)),
        "p90_move_pct": float(np.percentile(moves, 90)),
        "avg_last4_move_pct": avg_last4,
        "source": "earnings_history",
    }


def _compute_move_anchor(
    median_move_pct: float,
    avg_last4_move_pct: float,
) -> Optional[float]:
    median_val = _safe_float(median_move_pct, np.nan)
    avg4_val = _safe_float(avg_last4_move_pct, np.nan)

    if np.isfinite(median_val) and np.isfinite(avg4_val):
        # Recent events get more weight, but keep a longer-window stabilizer.
        return float(0.65 * avg4_val + 0.35 * median_val)
    if np.isfinite(avg4_val):
        return float(avg4_val)
    if np.isfinite(median_val):
        return float(median_val)
    return None


def _estimate_transaction_cost_pct(liquidity: float, spread_pct: Optional[float]) -> float:
    liq_component = 1.0 - float(
        np.clip(
            (np.log1p(max(liquidity, 1.0)) - np.log1p(500.0))
            / (np.log1p(25000.0) - np.log1p(500.0)),
            0.0,
            1.0,
        )
    )
    spread_component = float(np.clip((_safe_float(spread_pct, 18.0)) / 25.0, 0.20, 2.0))
    cost_pct = 0.18 + 0.55 * liq_component + 0.45 * spread_component
    return float(np.clip(cost_pct, 0.15, 2.25))


def _score_expectancy(net_edge_pct: float, drawdown_risk_pct: float, sample_size: int) -> float:
    if not np.isfinite(net_edge_pct):
        return 0.45
    edge_component = float(np.clip((net_edge_pct + 0.60) / 2.40, 0.0, 1.0))
    ratio = float(net_edge_pct / max(drawdown_risk_pct, 0.25))
    ratio_component = float(np.clip((ratio + 0.25) / 0.90, 0.0, 1.0))
    sample_component = float(np.clip(float(sample_size) / 8.0, 0.0, 1.0))
    return float(np.clip(0.55 * edge_component + 0.35 * ratio_component + 0.10 * sample_component, 0.0, 1.0))


def analyze_single_ticker(symbol: str) -> EdgeSnapshot:
    clean_symbol = str(symbol or "").strip().upper()
    if not clean_symbol:
        raise ValueError("Symbol is required")

    ticker = yf.Ticker(clean_symbol)
    hist = ticker.history(period="6mo", auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"No market data for {clean_symbol}")

    close = pd.to_numeric(hist.get("Close"), errors="coerce").dropna()
    if close.empty:
        raise ValueError(f"No close prices for {clean_symbol}")

    current_price = float(close.iloc[-1])
    rv30 = float(close.pct_change().dropna().tail(30).std(ddof=1) * np.sqrt(252))
    rv30 = max(rv30, 1e-4)
    smile_state = _short_expiry_smile_curvature(ticker, current_price)

    dte = _next_earnings_days(ticker)
    (
        days,
        ivs,
        iv30,
        iv45,
        ts_slope_0_45,
        liquidity,
        implied_move_pct,
        near_term_spread_pct,
        near_term_dte,
    ) = _term_structure_points(ticker, current_price=current_price)
    move_profile = _historical_earnings_move_profile(ticker, close)
    median_earnings_move_pct = _safe_float(move_profile.get("median_move_pct"), np.nan)
    p90_earnings_move_pct = _safe_float(move_profile.get("p90_move_pct"), np.nan)
    avg_last4_move_pct = _safe_float(move_profile.get("avg_last4_move_pct"), np.nan)
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
    tx_cost_pct = _estimate_transaction_cost_pct(liquidity=liquidity, spread_pct=near_term_spread_pct)

    iv_rv = float(iv30 / rv30) if np.isfinite(iv30) and np.isfinite(rv30) and rv30 > 0 else np.nan
    setup_base = _score_setup(iv_rv=iv_rv, ts_slope=ts_slope_0_45, dte=dte, liquidity=liquidity)
    smile_curvature = _safe_float(smile_state.get("curvature"), np.nan)
    if np.isfinite(smile_curvature):
        concavity_component = float(np.clip((-smile_curvature - 0.5) / 2.5, 0.0, 1.0))
    else:
        concavity_component = 0.50
    setup_score = float(np.clip(0.88 * setup_base + 0.12 * concavity_component, 0.0, 1.0))

    raw_gross_edge_pct = (
        float(implied_move_pct - move_anchor_pct_val)
        if np.isfinite(_safe_float(implied_move_pct, np.nan)) and np.isfinite(move_anchor_pct_val)
        else np.nan
    )
    implied_vs_anchor_ratio = (
        float(implied_move_pct / move_anchor_pct_val)
        if np.isfinite(_safe_float(implied_move_pct, np.nan)) and np.isfinite(move_anchor_pct_val) and move_anchor_pct_val > 0
        else np.nan
    )
    expected_gross_edge_pct = (
        float(raw_gross_edge_pct * sample_confidence)
        if np.isfinite(raw_gross_edge_pct)
        else np.nan
    )
    expected_net_edge_pct = (
        float(expected_gross_edge_pct - tx_cost_pct)
        if np.isfinite(expected_gross_edge_pct)
        else np.nan
    )
    base_drawdown_risk_pct = (
        float(max(p90_earnings_move_pct - implied_move_pct, 0.0) + tx_cost_pct)
        if np.isfinite(p90_earnings_move_pct) and np.isfinite(_safe_float(implied_move_pct, np.nan))
        else float(0.75 + tx_cost_pct)
    )
    concavity_risk_surcharge_pct = float(
        (0.55 * concavity_component) if bool(smile_state.get("concave")) else max(0.0, (concavity_component - 0.60) * 0.20)
    )
    drawdown_risk_pct = float(
        base_drawdown_risk_pct + (1.0 - sample_confidence) * 2.50 + concavity_risk_surcharge_pct
    )
    expectancy_ratio = (
        float(expected_net_edge_pct / max(drawdown_risk_pct, 0.25))
        if np.isfinite(expected_net_edge_pct)
        else np.nan
    )
    expectancy_score = _score_expectancy(
        net_edge_pct=expected_net_edge_pct,
        drawdown_risk_pct=drawdown_risk_pct,
        sample_size=int(move_profile.get("event_count", 0) or 0),
    )
    composite_score = float(np.clip(0.60 * setup_score + 0.40 * expectancy_score, 0.0, 1.0))
    confidence_pct = float(np.clip(30.0 + 70.0 * composite_score, 0.0, 100.0))

    hard_gate_reasons: List[str] = []
    implied_move_val = _safe_float(implied_move_pct, np.nan)
    if dte is None:
        hard_gate_reasons.append("Days to earnings unavailable.")
    elif dte < 1:
        hard_gate_reasons.append("Entry window closed (DTE < 1).")
    if near_term_dte is None:
        hard_gate_reasons.append("Near-term option expiry unavailable.")
    elif near_term_dte < MIN_SHORT_LEG_DTE:
        hard_gate_reasons.append(
            f"Near-term option expiry too close ({near_term_dte} DTE < {MIN_SHORT_LEG_DTE})."
        )
    if dte is not None and near_term_dte is not None and dte >= near_term_dte:
        hard_gate_reasons.append("Earnings event is not before the short-leg expiry.")
    if not np.isfinite(implied_move_val):
        hard_gate_reasons.append("Near-term implied move unavailable from quotes.")
    spread_val = _safe_float(near_term_spread_pct, np.nan)
    if np.isfinite(spread_val) and spread_val > MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE:
        hard_gate_reasons.append(
            f"Near-term spread too wide ({spread_val:.1f}% > {MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE:.1f}%)."
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

    recommendation = "Pass"
    if hard_no_trade:
        recommendation = "No Trade"
    else:
        if (
            np.isfinite(expected_net_edge_pct)
            and expected_net_edge_pct >= 0.25
            and np.isfinite(expectancy_ratio)
            and expectancy_ratio >= 0.20
            and np.isfinite(implied_vs_anchor_ratio)
            and implied_vs_anchor_ratio >= 1.05
            and sample_confidence >= 0.30
            and np.isfinite(iv_rv)
            and iv_rv >= 1.05
            and dte is not None
            and 1 <= dte <= 21
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

    rationale = [
        f"IV/RV30={iv_rv:.2f}" if np.isfinite(iv_rv) else "IV/RV30 unavailable from chain snapshot.",
        f"Term slope(0-45)={ts_slope_0_45:.4f}" if np.isfinite(ts_slope_0_45) else "Term slope unavailable.",
        f"Days to earnings={dte}" if dte is not None else "Days to earnings unavailable.",
        f"Near-term option DTE={near_term_dte}" if near_term_dte is not None else "Near-term option DTE unavailable.",
        (
            f"Near-term smile curvature={smile_curvature:.3f} "
            f"({'concave' if smile_state.get('concave') else 'not concave'})."
            if np.isfinite(smile_curvature)
            else "Near-term smile curvature unavailable."
        ),
        (
            f"Near-term spread={near_term_spread_pct:.2f}% "
            f"(gate <= {MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE:.1f}%)."
            if np.isfinite(_safe_float(near_term_spread_pct, np.nan))
            else "Near-term spread unavailable."
        ),
        (
            f"Hard gate status=FAIL ({'; '.join(hard_gate_reasons)})"
            if hard_no_trade
            else "Hard gate status=PASS."
        ),
        (
            f"Expected net edge={expected_net_edge_pct:+.2f}% "
            f"(gross {expected_gross_edge_pct:+.2f}% after sample-adjustment - cost {tx_cost_pct:.2f}%; "
            f"implied/anchor={implied_vs_anchor_ratio:.2f}x)."
            if np.isfinite(expected_net_edge_pct)
            and np.isfinite(expected_gross_edge_pct)
            and np.isfinite(implied_vs_anchor_ratio)
            else f"Expected net edge unavailable; cost baseline={tx_cost_pct:.2f}%."
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
    ]

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_price": current_price,
        "days_to_earnings": dte,
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
        "near_term_dte": near_term_dte,
        "implied_move_pct": float(implied_move_pct) if np.isfinite(_safe_float(implied_move_pct, np.nan)) else None,
        "earnings_move_median_pct": float(median_earnings_move_pct) if np.isfinite(median_earnings_move_pct) else None,
        "earnings_move_p90_pct": float(p90_earnings_move_pct) if np.isfinite(p90_earnings_move_pct) else None,
        "earnings_move_avg_last4_pct": float(avg_last4_move_pct) if np.isfinite(avg_last4_move_pct) else None,
        "earnings_move_anchor_pct": float(move_anchor_pct_val) if np.isfinite(move_anchor_pct_val) else None,
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
    }

    return EdgeSnapshot(
        symbol=clean_symbol,
        recommendation=recommendation,
        confidence_pct=confidence_pct,
        setup_score=setup_score,
        metrics=metrics,
        rationale=rationale,
    )
