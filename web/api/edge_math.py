"""
Pure-math leaves extracted from web/api/edge_engine.py (Phase 3.1).

These functions are stateless (no edge_engine module state beyond constants)
and were moved here verbatim — no logic changes. edge_engine re-exports every
name, so existing `from web.api.edge_engine import <fn>` paths and monkeypatch
targets keep working. Internal call edges stay within this module
(_derive_iv_scenarios <- payoffs; _excess_kurtosis <- _kurtosis_confidence_mult).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.candidate_shadow_outcome import safe_float as _safe_float
from services.realized_vol import rs_daily_vol_series as _rs_daily_vol_series
from web.api.edge_constants import (
    MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL,
    MOVE_UNCERTAINTY_Z_SCORE,
    _HEURISTIC_THRESHOLDS,
)

logger = logging.getLogger(__name__)


def _classify_move_risk(
    p90_earnings_move_pct: Optional[float],
    event_implied_move_pct: Optional[float],
    sample_size: int,
) -> Tuple[str, Optional[float]]:
    """
    Soft advisory for how stressed the historical earnings tail is relative to the
    event-implied move currently priced by the market.
    """
    p90_val = _safe_float(p90_earnings_move_pct, np.nan)
    impl_val = _safe_float(event_implied_move_pct, np.nan)
    if not np.isfinite(p90_val) or not np.isfinite(impl_val) or impl_val <= 0:
        return "unknown", None

    ratio = float(p90_val / impl_val)
    if ratio > 1.15:
        level = "elevated"
    elif ratio >= 0.90:
        level = "moderate"
    else:
        level = "low"

    if int(sample_size or 0) < 5 and level == "low":
        level = "moderate"

    return level, ratio


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
    _w4 = _HEURISTIC_THRESHOLDS["move_anchor_avg_last4_weight"]["value"]  # 0.65; recency bias
    median_val = _safe_float(median_move_pct, np.nan)
    avg4_val = _safe_float(avg_last4_move_pct, np.nan)
    if np.isfinite(median_val) and np.isfinite(avg4_val):
        return float(_w4 * avg4_val + (1.0 - _w4) * median_val)
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


def _bsm_greeks(
    S: float, T_days: float, sigma: float, r: float = 0.045, q: float = 0.0,
) -> Dict[str, Optional[float]]:
    """
    Black-Scholes-Merton ATM option greeks (K = S).

    Parameters
    ----------
    S      : underlying price
    T_days : calendar days to option expiry (use earnings DTE for calendar spread context)
    sigma  : annualised implied vol (iv30)
    r      : risk-free rate (default fallback only; callers normally inject a live rate)
    q      : continuous dividend yield (PR #68 — was missing pre-fix). Default 0.0
             matches legacy no-dividend behavior byte-for-byte for callers that
             don't yet resolve a per-symbol q. The live ``analyze_single_ticker``
             path now resolves q via ``get_dividend_yield(symbol)``.

    Returns greeks for a single ATM option leg:
      delta_call/put  — directional sensitivity (call ∈ [0,1], put ∈ [-1,0])
      gamma           — convexity per $1 move (identical for call and put ATM)
      vega            — $ sensitivity per 1 pp IV move (÷100 applied)
      theta_call/put  — daily time decay ($, negative = cost to holder)

    Merton-extension form (PR #68):
      d1 = (ln(S/K) + (r - q + σ²/2)·T) / (σ·√T)
      delta_call = e^(−qT)·N(d1)
      delta_put  = e^(−qT)·(N(d1) − 1)
      gamma      = e^(−qT)·φ(d1) / (S·σ·√T)
      vega       = S·e^(−qT)·φ(d1)·√T
      theta_call = −S·e^(−qT)·φ(d1)·σ/(2√T) − r·K·e^(−rT)·N(d2) + q·S·e^(−qT)·N(d1)
      theta_put  = −S·e^(−qT)·φ(d1)·σ/(2√T) + r·K·e^(−rT)·N(−d2) − q·S·e^(−qT)·N(−d1)

    Setting q = 0 recovers the original no-dividend formulas exactly — the
    q = 0 regression test in test_bsm_dividend_yield.py pins this.
    """
    empty: Dict[str, Optional[float]] = {
        k: None for k in ("delta_call", "delta_put", "gamma", "vega", "theta_call", "theta_put")
    }
    try:
        if T_days <= 0 or sigma <= 0 or S <= 0:
            return empty
        T = T_days / 365.0
        sq_T = math.sqrt(T)

        # ATM: K = S  →  log(S/K) = 0. Carry cost is (r − q).
        d1 = ((r - q + 0.5 * sigma ** 2) * T) / (sigma * sq_T)
        d2 = d1 - sigma * sq_T

        def _ncdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        def _npdf(x: float) -> float:
            return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

        N_d1 = _ncdf(d1)
        N_d2 = _ncdf(d2)
        n_d1 = _npdf(d1)
        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        delta_call = float(disc_q * N_d1)
        delta_put  = float(disc_q * (N_d1 - 1.0))
        gamma      = float(disc_q * n_d1 / (S * sigma * sq_T))
        vega       = float(S * disc_q * n_d1 * sq_T * 0.01)  # per 1 pp vol move
        # Merton theta:
        #   call: −S·e^(−qT)·φ(d1)·σ/(2√T) − r·K·e^(−rT)·N(d2) + q·S·e^(−qT)·N(d1)
        #   put : −S·e^(−qT)·φ(d1)·σ/(2√T) + r·K·e^(−rT)·N(−d2) − q·S·e^(−qT)·N(−d1)
        # ATM: K = S. N(−d2) = 1 − N(d2); N(−d1) = 1 − N(d1).
        theta_call = float(
            (
                -S * disc_q * n_d1 * sigma / (2.0 * sq_T)
                - r * S * disc_r * N_d2
                + q * S * disc_q * N_d1
            ) / 365.0
        )
        theta_put  = float(
            (
                -S * disc_q * n_d1 * sigma / (2.0 * sq_T)
                + r * S * disc_r * (1.0 - N_d2)
                - q * S * disc_q * (1.0 - N_d1)
            ) / 365.0
        )
        return {
            "delta_call": delta_call, "delta_put": delta_put,
            "gamma": gamma, "vega": vega,
            "theta_call": theta_call, "theta_put": theta_put,
        }
    except Exception as exc:
        logger.debug("BSM greeks computation failed: %s", exc)
        return empty


def _derive_iv_scenarios(
    iv_back: float,
    raw_moves_pct: Optional[List[float]],
    implied_move_pct: Optional[float],
) -> Dict[str, float]:
    """
    Fix 5: Derive symbol-specific post-earnings IV scenarios from historical
    move distribution instead of hardcoded ±20%/−25%/−45% multipliers.

    Logic:
      - If we have ≥4 historical moves AND a valid implied_move_pct, compute
        the distribution of how moves compared to the implied move.
        • 'crush fraction' per event = max(0, 1 − |move| / implied_move)
          i.e. how far inside the implied move the stock settled.
        • median and 25th-pctile crush fraction drive mild/severe crush
          scenario multipliers respectively.
        • Expansion multiplier uses the 75th-pctile move vs implied_move.
      - Otherwise fall back to the hardcoded heuristic values documented in
        _HEURISTIC_THRESHOLDS.

    All multipliers are applied to iv_back (the surviving back leg at expiry).
    """
    # ── Fallback (hardcoded heuristics) ──────────────────────────────────────
    fallback = {
        "iv_expand":      iv_back * _HEURISTIC_THRESHOLDS["calendar_scenario_fallback_expand"]["value"],
        "iv_flat":        iv_back,
        "iv_crush_mild":  iv_back * _HEURISTIC_THRESHOLDS["calendar_scenario_fallback_crush_mild"]["value"],
        "iv_crush_severe": iv_back * _HEURISTIC_THRESHOLDS["calendar_scenario_fallback_crush_severe"]["value"],
    }

    if (
        not raw_moves_pct
        or len(raw_moves_pct) < 4
        or implied_move_pct is None
        or not np.isfinite(implied_move_pct)
        or implied_move_pct <= 0
    ):
        return {**fallback, "_source": "heuristic_fallback"}

    impl = float(implied_move_pct)
    moves = [abs(float(m)) for m in raw_moves_pct if np.isfinite(float(m))]
    if len(moves) < 4:
        return {**fallback, "_source": "heuristic_fallback"}

    # Source tier: ≥8 events = full calibration; 4-7 = small-sample estimate.
    # Threshold mirrors MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL (8 events).
    _n_events = len(moves)
    if _n_events >= MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL:
        _scenario_source = "historical_symbol_calibrated"
    else:
        _scenario_source = "small_sample_estimate"  # 4-7 events: usable but thin

    # Crush fraction per event: how far inside the implied move did the stock land?
    crush_fracs = [max(0.0, 1.0 - m / impl) for m in moves]
    crush_median = float(np.percentile(crush_fracs, 50))
    crush_p25    = float(np.percentile(crush_fracs, 25))  # more severe crush

    # Expansion: how much did the stock overshoot the implied move in the worst cases?
    overshoot_p75 = float(np.percentile(moves, 75))
    expand_ratio = float(np.clip(overshoot_p75 / impl, 1.0, 2.0))

    # IV scenarios are multiplicative on iv_back.
    # Mild crush ≈ median historical resolution inside implied move.
    # Severe crush ≈ 25th-pctile resolution (better-than-median crush).
    # Expansion ≈ stock blew through implied, IV may spike on residual uncertainty.
    mild_crush_mult   = float(np.clip(1.0 - crush_median * 0.50, 0.45, 0.95))
    severe_crush_mult = float(np.clip(1.0 - crush_p25   * 0.70, 0.30, 0.85))
    expand_mult       = float(np.clip(expand_ratio * 0.80 + 0.20, 1.05, 1.40))

    return {
        "iv_expand":       iv_back * expand_mult,
        "iv_flat":         iv_back,
        "iv_crush_mild":   iv_back * mild_crush_mult,
        "iv_crush_severe": iv_back * severe_crush_mult,
        "_source":            _scenario_source,
        "_n_events":          _n_events,
        "_crush_median_pct":  round(crush_median * 100, 1),
        "_crush_p25_pct":     round(crush_p25 * 100, 1),
        "_expand_ratio":      round(expand_ratio, 3),
    }


def _calendar_spread_payoff(
    S: float,
    iv_near: float,
    iv_back: float,
    T_near_days: float,
    T_back_days: float,
    r: float = 0.045,
    n_points: int = 41,
    raw_moves_pct: Optional[List[float]] = None,
    implied_move_pct: Optional[float] = None,
    side: str = "call",
    q: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    ATM calendar spread payoff at near-leg expiry.

    side='call' (default): short near ATM call / long back ATM call
    side='put':            short near ATM put  / long back ATM put

    Both legs are priced at K = S (ATM). Put prices derived via the
    Merton-extension put-call parity:
        P = C − S·exp(−qT) + K·exp(−rT)

    PR #68 added the ``q`` parameter (continuous dividend yield). Default
    ``q=0.0`` matches the legacy no-dividend formulas byte-for-byte. The
    live ``analyze_single_ticker`` path now resolves q via
    ``get_dividend_yield(symbol)`` so dividend-paying names get correct
    put-side debit / payoff math. Pre-fix, dividend names like AAPL/MSFT
    were systematically mispriced on the put-side calendar.

    P&L at near-leg expiry, stock at S_t:
      call: BSM_call(S_t, T_rem, σ_post) − max(S_t−K,0) − debit
      put:  BSM_put (S_t, T_rem, σ_post) − max(K−S_t,0) − debit

    Returns payoff_scenarios ($/share) across ±20% moves under four
    IV scenarios plus IV-flat breakeven moves.
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
            if T_yr <= 0 or sigma <= 0 or spot <= 0:
                return max(spot - K, 0.0)
            sq_T = math.sqrt(T_yr)
            # Merton: spot discounted by e^(−qT), carry is (r − q)
            d1 = (math.log(spot / K) + (r - q + 0.5 * sigma ** 2) * T_yr) / (sigma * sq_T)
            d2 = d1 - sigma * sq_T
            return spot * math.exp(-q * T_yr) * _ncdf(d1) - K * math.exp(-r * T_yr) * _ncdf(d2)

        def _bsm_put(spot: float, T_yr: float, sigma: float) -> float:
            # Merton-extension put-call parity: P = C − S·e^(−qT) + K·e^(−rT)
            return (
                _bsm_call(spot, T_yr, sigma)
                - spot * math.exp(-q * T_yr)
                + K * math.exp(-r * T_yr)
            )

        # Entry premiums
        c_near = _bsm_call(S, T_near, iv_near)
        c_back = _bsm_call(S, T_back, iv_back)
        if side == "put":
            near_entry = _bsm_put(S, T_near, iv_near)
            back_entry = _bsm_put(S, T_back, iv_back)
        else:
            near_entry = c_near
            back_entry = c_back

        entry_debit = back_entry - near_entry
        if entry_debit <= 0:
            return None  # degenerate: near leg worth more than back leg

        _iv_scenario_def = _derive_iv_scenarios(iv_back, raw_moves_pct, implied_move_pct)
        _scenario_source = str(_iv_scenario_def.get("_source", "heuristic_fallback"))
        iv_scenarios = {
            "iv_expand_20": _iv_scenario_def["iv_expand"],
            "iv_flat":      _iv_scenario_def["iv_flat"],
            "iv_crush_25":  _iv_scenario_def["iv_crush_mild"],
            "iv_crush_45":  _iv_scenario_def["iv_crush_severe"],
        }

        n  = max(21, min(int(n_points), 81))
        moves = [round(-0.20 + i * (0.40 / (n - 1)), 4) for i in range(n)]

        payoff_rows: List[Dict[str, Any]] = []
        for move in moves:
            S_t = S * (1.0 + move)
            if side == "put":
                short_intrinsic = max(K - S_t, 0.0)
            else:
                short_intrinsic = max(S_t - K, 0.0)
            row: Dict[str, Any] = {"move_pct": round(move * 100, 1), "price": round(S_t, 2)}
            for label, iv_post in iv_scenarios.items():
                if side == "put":
                    back_post = _bsm_put(S_t, T_rem, iv_post)
                else:
                    back_post = _bsm_call(S_t, T_rem, iv_post)
                row[label] = round(back_post - short_intrinsic - entry_debit, 3)
            payoff_rows.append(row)

        breakevens: List[float] = []
        for i in range(len(payoff_rows) - 1):
            p1 = payoff_rows[i]["iv_flat"]
            p2 = payoff_rows[i + 1]["iv_flat"]
            if p1 * p2 < 0:
                m1 = payoff_rows[i]["move_pct"]
                m2 = payoff_rows[i + 1]["move_pct"]
                be = m1 + (m2 - m1) * (-p1) / (p2 - p1)
                breakevens.append(round(be, 1))

        payoff_rows_per_contract = [
            {k: (round(v * 100, 2) if k not in ("move_pct", "price") else v)
             for k, v in row.items()}
            for row in payoff_rows
        ]
        return {
            "structure":                  "put_calendar" if side == "put" else "call_calendar",
            "entry_debit":                round(entry_debit, 4),
            "entry_debit_per_contract":   round(entry_debit * 100, 2),
            "entry_near_premium":         round(near_entry, 4),
            "entry_back_premium":         round(back_entry, 4),
            "t_near_days":                T_near_days,
            "t_back_days":                T_back_days,
            "t_remaining_days":           T_back_days - T_near_days,
            "iv_near":                    iv_near,
            "iv_back":                    iv_back,
            "breakeven_moves_pct":        breakevens,
            "payoff_scenarios":           payoff_rows,
            "payoff_scenarios_per_contract": payoff_rows_per_contract,
            "calendar_is_theoretical":    True,
            "calendar_note":              (
                "Priced from interpolated IV30/IV45; back leg = near + 28d. "
                "Not guaranteed to match a live quoted chain."
            ),
            "calendar_scenario_source":   _scenario_source,
        }
    except Exception as exc:
        logger.debug("Calendar payoff computation failed: %s", exc)
        return None


def _straddle_payoff(
    S: float,
    iv: float,
    T_near_days: float,
    r: float = 0.045,
    n_points: int = 41,
    raw_moves_pct: Optional[List[float]] = None,
    implied_move_pct: Optional[float] = None,
    q: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Long ATM straddle payoff for an earnings play.

    Entry: buy ATM call + buy ATM put (K = S, T = T_near_days, σ = iv).
    P&L evaluated 1 day post-event: stock has moved, IV has shifted per scenario.

    PR #68 added the ``q`` parameter (continuous dividend yield). Default
    ``q=0.0`` preserves the legacy no-dividend behavior byte-for-byte. With
    q > 0, both call and put prices follow Merton-extension BSM and the
    put-call parity used internally is ``P = C − S·exp(−qT) + K·exp(−rT)``.

    The 1-day residual convention (rather than full expiry) captures the
    dominant risk for earnings straddles — IV crush kills the position even
    when the stock moves correctly. IV scenarios come from the same
    _derive_iv_scenarios() calibration used by the calendar diagram.
    """
    try:
        if not (np.isfinite(S) and S > 0 and np.isfinite(iv) and iv > 0
                and np.isfinite(T_near_days) and T_near_days >= 1):
            return None

        T_entry = T_near_days / 365.0
        T_rem   = 1.0 / 365.0   # 1-day post-event residual
        K       = S              # ATM

        def _ncdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        def _bsm_call(spot: float, T_yr: float, sigma: float) -> float:
            if T_yr <= 0 or sigma <= 0 or spot <= 0:
                return max(spot - K, 0.0)
            sq_T = math.sqrt(T_yr)
            d1 = (math.log(spot / K) + (r - q + 0.5 * sigma ** 2) * T_yr) / (sigma * sq_T)
            d2 = d1 - sigma * sq_T
            return spot * math.exp(-q * T_yr) * _ncdf(d1) - K * math.exp(-r * T_yr) * _ncdf(d2)

        def _bsm_put(spot: float, T_yr: float, sigma: float) -> float:
            return (
                _bsm_call(spot, T_yr, sigma)
                - spot * math.exp(-q * T_yr)
                + K * math.exp(-r * T_yr)
            )

        c_entry = _bsm_call(S, T_entry, iv)
        p_entry = _bsm_put(S, T_entry, iv)
        entry_debit = c_entry + p_entry

        _iv_scenario_def = _derive_iv_scenarios(iv, raw_moves_pct, implied_move_pct)
        _scenario_source = str(_iv_scenario_def.get("_source", "heuristic_fallback"))
        iv_scenarios = {
            "iv_expand_20": _iv_scenario_def["iv_expand"],
            "iv_flat":      _iv_scenario_def["iv_flat"],
            "iv_crush_25":  _iv_scenario_def["iv_crush_mild"],
            "iv_crush_45":  _iv_scenario_def["iv_crush_severe"],
        }

        n  = max(21, min(int(n_points), 81))
        moves = [round(-0.20 + i * (0.40 / (n - 1)), 4) for i in range(n)]

        payoff_rows: List[Dict[str, Any]] = []
        for move in moves:
            S_t = S * (1.0 + move)
            row: Dict[str, Any] = {"move_pct": round(move * 100, 1), "price": round(S_t, 2)}
            for label, iv_post in iv_scenarios.items():
                c_post = _bsm_call(S_t, T_rem, iv_post)
                p_post = _bsm_put(S_t, T_rem, iv_post)
                row[label] = round(c_post + p_post - entry_debit, 3)
            payoff_rows.append(row)

        breakevens: List[float] = []
        for i in range(len(payoff_rows) - 1):
            p1 = payoff_rows[i]["iv_flat"]
            p2 = payoff_rows[i + 1]["iv_flat"]
            if p1 * p2 < 0:
                m1 = payoff_rows[i]["move_pct"]
                m2 = payoff_rows[i + 1]["move_pct"]
                be = m1 + (m2 - m1) * (-p1) / (p2 - p1)
                breakevens.append(round(be, 1))

        payoff_rows_per_contract = [
            {k: (round(v * 100, 2) if k not in ("move_pct", "price") else v)
             for k, v in row.items()}
            for row in payoff_rows
        ]
        return {
            "structure":                  "atm_straddle",
            "entry_debit":               round(entry_debit, 4),
            "entry_debit_per_contract":  round(entry_debit * 100, 2),
            "entry_call":                round(c_entry, 4),
            "entry_put":                 round(p_entry, 4),
            "strike":                    round(K, 2),
            "iv_entry":                  round(iv, 4),
            "T_near_days":               T_near_days,
            "T_remain_days":             1,
            "breakeven_moves_pct":       breakevens,
            "payoff_scenarios":          payoff_rows,
            "payoff_scenarios_per_contract": payoff_rows_per_contract,
            "is_theoretical":            True,
            "scenario_source":           _scenario_source,
            "note": (
                f"Long ATM straddle (K={K:.2f}). P&L shown 1-day post-event "
                "with IV scenarios. BSM entry at market IV; breakevens on IV-flat scenario."
            ),
        }
    except Exception as exc:
        logger.debug("Straddle payoff computation failed: %s", exc)
        return None


def _strangle_payoff(
    S: float,
    iv: float,
    T_near_days: float,
    wing_pct: float,
    r: float = 0.045,
    n_points: int = 41,
    raw_moves_pct: Optional[List[float]] = None,
    implied_move_pct: Optional[float] = None,
    q: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Long OTM strangle payoff for an earnings play.

    Wings placed at ±wing_pct% from ATM (typically = implied_move_pct so the
    strangle only profits when the stock exceeds the market's priced move —
    a pure earnings-surprise bet).

    Entry: buy OTM call at K_c = S*(1+wing/100) and OTM put at K_p = S*(1-wing/100).
    P&L evaluated 1 day post-event with IV scenarios (same as straddle).

    PR #68 added the ``q`` parameter (continuous dividend yield). Default
    ``q=0.0`` preserves legacy behavior; q > 0 routes through Merton-extension
    BSM with put-call parity ``P = C − S·exp(−qT) + K·exp(−rT)``.
    """
    try:
        if not (np.isfinite(S) and S > 0 and np.isfinite(iv) and iv > 0
                and np.isfinite(T_near_days) and T_near_days >= 1
                and np.isfinite(wing_pct) and wing_pct > 0):
            return None

        T_entry = T_near_days / 365.0
        T_rem   = 1.0 / 365.0
        K_c     = S * (1.0 + wing_pct / 100.0)   # OTM call strike
        K_p     = S * (1.0 - wing_pct / 100.0)   # OTM put strike

        def _ncdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        def _bsm_call(spot: float, K: float, T_yr: float, sigma: float) -> float:
            if T_yr <= 0 or sigma <= 0 or spot <= 0:
                return max(spot - K, 0.0)
            sq_T = math.sqrt(T_yr)
            d1 = (math.log(spot / K) + (r - q + 0.5 * sigma ** 2) * T_yr) / (sigma * sq_T)
            d2 = d1 - sigma * sq_T
            return spot * math.exp(-q * T_yr) * _ncdf(d1) - K * math.exp(-r * T_yr) * _ncdf(d2)

        def _bsm_put(spot: float, K: float, T_yr: float, sigma: float) -> float:
            return (
                _bsm_call(spot, K, T_yr, sigma)
                - spot * math.exp(-q * T_yr)
                + K * math.exp(-r * T_yr)
            )

        c_entry = _bsm_call(S, K_c, T_entry, iv)
        p_entry = _bsm_put(S, K_p, T_entry, iv)
        entry_debit = c_entry + p_entry

        _iv_scenario_def = _derive_iv_scenarios(iv, raw_moves_pct, implied_move_pct)
        _scenario_source = str(_iv_scenario_def.get("_source", "heuristic_fallback"))
        iv_scenarios = {
            "iv_expand_20": _iv_scenario_def["iv_expand"],
            "iv_flat":      _iv_scenario_def["iv_flat"],
            "iv_crush_25":  _iv_scenario_def["iv_crush_mild"],
            "iv_crush_45":  _iv_scenario_def["iv_crush_severe"],
        }

        n  = max(21, min(int(n_points), 81))
        moves = [round(-0.20 + i * (0.40 / (n - 1)), 4) for i in range(n)]

        payoff_rows: List[Dict[str, Any]] = []
        for move in moves:
            S_t = S * (1.0 + move)
            row: Dict[str, Any] = {"move_pct": round(move * 100, 1), "price": round(S_t, 2)}
            for label, iv_post in iv_scenarios.items():
                c_post = _bsm_call(S_t, K_c, T_rem, iv_post)
                p_post = _bsm_put(S_t, K_p, T_rem, iv_post)
                row[label] = round(c_post + p_post - entry_debit, 3)
            payoff_rows.append(row)

        breakevens: List[float] = []
        for i in range(len(payoff_rows) - 1):
            p1 = payoff_rows[i]["iv_flat"]
            p2 = payoff_rows[i + 1]["iv_flat"]
            if p1 * p2 < 0:
                m1 = payoff_rows[i]["move_pct"]
                m2 = payoff_rows[i + 1]["move_pct"]
                be = m1 + (m2 - m1) * (-p1) / (p2 - p1)
                breakevens.append(round(be, 1))

        payoff_rows_per_contract = [
            {k: (round(v * 100, 2) if k not in ("move_pct", "price") else v)
             for k, v in row.items()}
            for row in payoff_rows
        ]
        return {
            "structure":                 "otm_strangle",
            "entry_debit":               round(entry_debit, 4),
            "entry_debit_per_contract":  round(entry_debit * 100, 2),
            "entry_call":                round(c_entry, 4),
            "entry_put":                 round(p_entry, 4),
            "strike_call":               round(K_c, 2),
            "strike_put":                round(K_p, 2),
            "wing_pct":                  round(wing_pct, 2),
            "iv_entry":                  round(iv, 4),
            "T_near_days":               T_near_days,
            "T_remain_days":             1,
            "breakeven_moves_pct":       breakevens,
            "payoff_scenarios":          payoff_rows,
            "payoff_scenarios_per_contract": payoff_rows_per_contract,
            "is_theoretical":            True,
            "scenario_source":           _scenario_source,
            "note": (
                f"Long OTM strangle (call K={K_c:.2f} / put K={K_p:.2f}, "
                f"±{wing_pct:.1f}% wings at implied move). "
                "P&L shown 1-day post-event with IV scenarios."
            ),
        }
    except Exception as exc:
        logger.debug("Strangle payoff computation failed: %s", exc)
        return None


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
