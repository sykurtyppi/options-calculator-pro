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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from services.earnings_event_service import resolve_upcoming_earnings_event
from services.earnings_vol_snapshot import build_vol_snapshot
from services.provider_telemetry import classify_error, record_provider_telemetry
from services.pricing_rates import (
    _rf_rate_cache,
    _rf_rate_lock,
    get_pricing_risk_free_rate as _get_pricing_risk_free_rate,
)
from services.realized_vol import (
    har_rv_forecast as _har_rv_forecast,
    rs_daily_vol_series as _rs_daily_vol_series,
    yang_zhang_rv30 as _yang_zhang_rv30,
)
from services.structure_scorecard import build_structure_scorecards
from services.structure_selector import select_best_structure
from web.api.edge_legacy_adapter import (
    calibration_phase_note as _calibration_phase_note,
    legacy_recommendation_from_selector as _legacy_recommendation_from_selector,
    selector_lead_scorecard as _selector_lead_scorecard,
    shared_setup_score_from_snapshot as _shared_setup_score_from_snapshot,
    snapshot_to_edge_inputs as _snapshot_to_edge_inputs,
)

logger = logging.getLogger(__name__)

# ─── ML crush classifier (optional — loaded once at startup) ──────────────────
# Trained by /api/ml/train.  If absent, inference is skipped gracefully.

_ML_MODEL_DIR = Path.home() / ".options_calculator_pro" / "models"
_crush_clf = None
_crush_scaler = None
_crush_model_loaded: bool = False
_ml_model_lock = threading.Lock()
_feature_store_lock = threading.Lock()
_feature_store_cache: Dict[str, Any] = {"loaded": False, "store": None}


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
# Mirrors services/structure_scorecard.ABSOLUTE_SPREAD_THRESHOLD_PCT (12.0).
# De Silva, Smith & So (2025, Review of Finance): at 12% spread, round-trip cost
# ≈ 24% of option value, at or above the ceiling of typical pre-earnings IV
# expansion. Keeping this aligned with the scorecard threshold prevents the
# contradictory state where the selector returns NO_TRADE on ineligibility
# while the edge-engine hard gate does not fire.
MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE = 12.0
MIN_SHORT_LEG_DTE = 2
MIN_NEAR_BACK_IV_RATIO_FOR_EVENT = 1.02
MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_TRADE = 400.0
MOVE_UNCERTAINTY_Z_SCORE = 1.28
IV_RV_CROWDING_WARNING_THRESHOLD = 1.60  # above this, IV may already price the event (soft warning only)
MIN_TS_SLOPE_FOR_EXPANSION = 0.0005
MIN_RV_PERCENTILE_FOR_EXPANSION = 60.0
MAX_NEAR_TERM_SPREAD_PCT_FOR_EXPANSION = 12.0
MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION = 600.0
MIN_PRICEABLE_EXPANSION_TRADES = 3
EXPANSION_ENTRY_OFFSETS = tuple(range(5, 11))
EXPANSION_EXIT_OFFSET = 1
EXPANSION_BACK_EXPIRY_GAP_DAYS = 14

# ─── Stale-data enforcement ───────────────────────────────────────────────────
# IV is live (from option chain); RV is computed from yfinance price bars.
# If the price bars are stale, IV/RV is not contemporaneous — this violates a
# core assumption of the edge model.  Cap confidence rather than hard-reject
# so signal information is preserved but the output is clearly degraded.
# Cap = 40%: well below the 62% "Consider" floor, signals clear data quality issue.
STALE_DATA_THRESHOLD_DAYS = 2           # bars older than N calendar days → stale
STALE_DATA_CONFIDENCE_CAP_PCT = 40.0   # hard ceiling when IV/RV non-contemporaneous

# ─── Soft gates replacing hard-gates 9 & 10 ──────────────────────────────────
# Principle: degraded evidence → degraded confidence, NOT hard rejection.
# A ticker with 4 earnings events and a consistent crush pattern still has
# information.  Capping at 55% keeps it off "Consider" (requires ≥62%) while
# exposing the signal for watchlist/monitoring.  Low-event-count cap is looser
# (60%) because the history is thin but at least real earnings history.
# If both flags are active the stricter cap wins (fallback takes elif precedence).
FALLBACK_MODEL_CONFIDENCE_CAP_PCT = 55.0   # move_source != "earnings_history"
LOW_EVENT_COUNT_CONFIDENCE_CAP_PCT = 60.0  # 1–7 earnings events (real but thin)

# ─── Fix 7: Heuristic threshold registry ─────────────────────────────────────
# Every scoring threshold that is NOT empirically derived is tagged here so
# the system can be audited.  Values are intentionally NOT changed — they are
# working priors that require a calibration pass on the backtest corpus to
# justify or revise.  Label: assumption=True means "prior belief, unvalidated."
_HEURISTIC_THRESHOLDS: Dict[str, Any] = {
    # ── Move anchor blend ─────────────────────────────────────────────────────
    "move_anchor_avg_last4_weight": {
        "value": 0.65,
        "assumption": True,
        "rationale": "Weight on avg(last 4 earnings moves) vs median. "
                     "Recency bias intentional but magnitude (65/35) is subjective.",
    },
    # ── Ticker tier breakpoints ───────────────────────────────────────────────
    "ticker_tier_mega_cap_usd": {
        "value": 200e9,
        "assumption": True,
        "rationale": "Market cap cutoff for mega-cap tier (1.00x multiplier). "
                     "No empirical derivation; standard institutional convention.",
    },
    # ── Kurtosis penalty breakpoints ──────────────────────────────────────────
    "kurtosis_penalty_ek_threshold_mild": {
        "value": 1.0,
        "assumption": True,
        "rationale": "Excess kurtosis above which mild penalty begins. "
                     "Based on normal distribution EK=0 plus 1σ buffer. Not calibrated.",
    },
    "kurtosis_penalty_ek_threshold_severe": {
        "value": 3.0,
        "assumption": True,
        "rationale": "Excess kurtosis above which severe penalty applies. "
                     "Matches leptokurtic threshold; not empirically derived.",
    },
    # ── Crush calibration rate thresholds ─────────────────────────────────────
    "crush_rate_boost_threshold": {
        "value": 0.72,
        "assumption": True,
        "rationale": "Historical crush rate above which confidence is boosted. "
                     "Represents 'stock stays inside IV ~3 out of 4 times'. Not calibrated.",
    },
    "crush_rate_penalise_threshold": {
        "value": 0.35,
        "assumption": True,
        "rationale": "Historical crush rate below which strong penalty applied. "
                     "Represents 'stock blows through IV majority of the time'. Not calibrated.",
    },
    # ── Transaction cost heuristic ────────────────────────────────────────────
    "tx_cost_base_pct": {
        "value": 0.18,
        "assumption": True,
        "rationale": "Base friction estimate (18bp). "
                     "Represents ~retail broker commission + minimal slippage on liquid names. "
                     "Does not account for notional size or multi-leg fills.",
    },
    # ── Calendar scenario fallbacks ────────────────────────────────────────────
    "calendar_scenario_fallback_expand": {
        "value": 1.20,
        "assumption": True,
        "rationale": "IV back-leg expansion multiplier when no historical data. "
                     "+20% represents a moderate IV spike scenario. Arbitrary.",
    },
    "calendar_scenario_fallback_crush_mild": {
        "value": 0.75,
        "assumption": True,
        "rationale": "Back-leg IV after mild crush (−25%). "
                     "Based on typical observed post-earnings IV crush range. Not calibrated.",
    },
    "calendar_scenario_fallback_crush_severe": {
        "value": 0.55,
        "assumption": True,
        "rationale": "Back-leg IV after severe crush (−45%). "
                     "Represents extreme crush seen in mega-cap earnings. Not calibrated.",
    },
    # ── Ticker tier market-cap cutoffs ────────────────────────────────────────
    "ticker_tier_large_cap_usd": {
        "value": 20e9,
        "assumption": True,
        "rationale": "Market cap cutoff for large-cap tier (20B). Standard convention; not empirically calibrated.",
    },
    "ticker_tier_mid_cap_usd": {
        "value": 2e9,
        "assumption": True,
        "rationale": "Market cap cutoff for mid-cap tier (2B). Standard convention; not empirically calibrated.",
    },
    "ticker_tier_small_cap_usd": {
        "value": 300e6,
        "assumption": True,
        "rationale": "Market cap cutoff for small-cap tier (300M). Below this = micro-cap.",
    },
    # ── Ticker tier confidence multipliers ────────────────────────────────────
    "ticker_tier_large_cap_mult": {
        "value": 0.95,
        "assumption": True,
        "rationale": "Large-cap confidence multiplier (5% haircut vs mega-cap). Not empirically derived.",
    },
    "ticker_tier_mid_cap_mult": {
        "value": 0.85,
        "assumption": True,
        "rationale": "Mid-cap confidence multiplier (15% haircut). Larger discount for less-covered names.",
    },
    "ticker_tier_small_cap_mult": {
        "value": 0.72,
        "assumption": True,
        "rationale": "Small-cap confidence multiplier (28% haircut). IV dynamics more erratic.",
    },
    "ticker_tier_micro_cap_mult": {
        "value": 0.55,
        "assumption": True,
        "rationale": "Micro-cap confidence multiplier (45% haircut). Binary event risk; thin option markets.",
    },
    # ── Kurtosis penalty shape parameters ────────────────────────────────────
    "kurtosis_penalty_slope_mild": {
        "value": 0.11,
        "assumption": True,
        "rationale": "Confidence multiplier decrease per unit excess kurtosis above mild threshold. "
                     "~11pp reduction per EK unit. Not calibrated.",
    },
    "kurtosis_penalty_slope_severe": {
        "value": 0.046,
        "assumption": True,
        "rationale": "Confidence multiplier decrease per unit EK above severe threshold. "
                     "Flatter slope because confidence floor is already low. Not calibrated.",
    },
    "kurtosis_penalty_mild_floor": {
        "value": 0.78,
        "assumption": True,
        "rationale": "Minimum multiplier in the mild regime (EK 1-3). "
                     "Floor prevents over-penalising moderately fat-tailed names.",
    },
    "kurtosis_penalty_severe_floor": {
        "value": 0.55,
        "assumption": True,
        "rationale": "Absolute minimum kurtosis confidence multiplier (EK ≥3). "
                     "45% haircut for extreme tail-risk names.",
    },
    # ── Crush calibration rate shape parameters ───────────────────────────────
    "crush_rate_boost_coeff": {
        "value": 0.55,
        "assumption": True,
        "rationale": "Sensitivity of confidence boost per unit crush rate above boost threshold. Not calibrated.",
    },
    "crush_rate_penalty_coeff_moderate": {
        "value": 1.13,
        "assumption": True,
        "rationale": "Sensitivity of confidence penalty for crush rates 35-50%. Not calibrated.",
    },
    "crush_rate_penalty_coeff_severe": {
        "value": 1.0,
        "assumption": True,
        "rationale": "Sensitivity of confidence penalty for crush rates below 35%. Not calibrated.",
    },
}



# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class EdgeSnapshot:
    symbol: str
    recommendation: str
    confidence_pct: float
    setup_score: float
    metrics: Dict[str, Any]
    rationale: List[str]
    selector_output: Optional[Dict[str, Any]] = None
    structure_scorecards: Optional[List[Dict[str, Any]]] = None
    vol_snapshot: Optional[Dict[str, Any]] = None


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


def _get_feature_store() -> Optional[Any]:
    with _feature_store_lock:
        if _feature_store_cache["loaded"]:
            return _feature_store_cache["store"]
        _feature_store_cache["loaded"] = True
        try:
            from services.options_feature_store import OptionsFeatureStore

            store = OptionsFeatureStore()
            if store.is_available():
                _feature_store_cache["store"] = store
            else:
                logger.info("Options feature store unavailable for IV expansion analysis.")
        except Exception as exc:
            logger.warning("Options feature store load failed: %s", exc)
        return _feature_store_cache["store"]


def _normalize_close_index(close: pd.Series) -> pd.Series:
    normalized = close.copy()
    if isinstance(normalized.index, pd.DatetimeIndex):
        idx = normalized.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        normalized.index = idx.normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    return normalized


def _business_days_before(sessions: pd.DatetimeIndex, target_date: pd.Timestamp, sessions_before: int) -> Optional[pd.Timestamp]:
    if sessions.empty or sessions_before < 1:
        return None
    target = pd.Timestamp(target_date).normalize()
    pos = int(sessions.searchsorted(target.to_datetime64(), side="left"))
    idx = pos - sessions_before
    if idx < 0 or idx >= len(sessions):
        return None
    return pd.Timestamp(sessions[idx]).normalize()


def _prepare_feature_chain(chain_df: pd.DataFrame, call_put: str = "C") -> pd.DataFrame:
    if chain_df is None or chain_df.empty:
        return pd.DataFrame()
    df = chain_df.copy()
    if "call_put" in df.columns:
        df["call_put"] = df["call_put"].astype(str).str.upper()
        df = df[df["call_put"] == call_put.upper()]
    for col in ("trade_date", "expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    for col in ("strike", "bid", "ask", "mid", "iv", "open_interest", "volume", "liquidity_score", "underlying_price", "spread_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"bid", "ask"}.issubset(df.columns):
        df = df[(df["bid"] >= 0) & (df["ask"] >= df["bid"])]
    if "mid" in df.columns:
        df = df[np.isfinite(df["mid"]) & (df["mid"] > 0)]
    if "strike" in df.columns:
        df = df[np.isfinite(df["strike"])]
    return df


def _best_contract_row(frame: pd.DataFrame) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    ranked = frame.copy()
    ranked["_liq"] = pd.to_numeric(ranked.get("liquidity_score"), errors="coerce").fillna(0.0)
    ranked["_oi"] = pd.to_numeric(ranked.get("open_interest"), errors="coerce").fillna(0.0)
    ranked["_vol"] = pd.to_numeric(ranked.get("volume"), errors="coerce").fillna(0.0)
    ranked["_spread_pct"] = pd.to_numeric(ranked.get("spread_pct"), errors="coerce").fillna(np.inf)
    ranked = ranked.sort_values(
        ["_liq", "_oi", "_vol", "_spread_pct"],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def _select_pre_earnings_calendar_contracts(
    chain_df: pd.DataFrame,
    event_date: pd.Timestamp,
    back_gap_days: int = EXPANSION_BACK_EXPIRY_GAP_DAYS,
) -> Optional[Dict[str, Any]]:
    df = _prepare_feature_chain(chain_df, call_put="C")
    if df.empty or "expiry" not in df.columns:
        return None

    event_ts = pd.Timestamp(event_date).normalize()
    expiries = sorted({pd.Timestamp(v).normalize() for v in df["expiry"].dropna().tolist()})
    eligible_front = [exp for exp in expiries if exp > event_ts]
    if not eligible_front:
        return None
    front_expiry = eligible_front[0]
    eligible_back = [exp for exp in expiries if exp > front_expiry and (exp - front_expiry).days >= back_gap_days]
    if not eligible_back:
        eligible_back = [exp for exp in expiries if exp > front_expiry]
    if not eligible_back:
        return None
    back_expiry = eligible_back[0]

    front_df = df[df["expiry"] == front_expiry]
    back_df = df[df["expiry"] == back_expiry]
    common_strikes = sorted(set(front_df["strike"].dropna().tolist()) & set(back_df["strike"].dropna().tolist()))
    if not common_strikes:
        return None

    underlying_price = _safe_float(df.get("underlying_price", pd.Series(dtype=float)).dropna().median(), np.nan)
    if not np.isfinite(underlying_price):
        underlying_price = float(np.median(common_strikes))
    chosen_strike = min(common_strikes, key=lambda strike: abs(float(strike) - underlying_price))

    front_row = _best_contract_row(front_df[front_df["strike"] == chosen_strike])
    back_row = _best_contract_row(back_df[back_df["strike"] == chosen_strike])
    if front_row is None or back_row is None:
        return None

    return {
        "option_type": "C",
        "strike": float(chosen_strike),
        "front_expiry": pd.Timestamp(front_expiry).normalize(),
        "back_expiry": pd.Timestamp(back_expiry).normalize(),
        "front_row": front_row,
        "back_row": back_row,
    }


_DUAL_PICKER_EXPERIMENTAL_NOTE = (
    "candidate_selection uses an in-sample +14d-min-front-DTE rule "
    "(PR-AB). Not out-of-sample validated. Do not execute against "
    "candidate_selection without independent review."
)


def _empty_dual_picker(shadow_status: str) -> Dict[str, Any]:
    """Canonical empty dual-picker block.

    Used for every code path where the dual-picker cannot produce a
    selection (missing entry chain, picker exception, etc.). Ensures
    every return shape carries the SAME keys — required so downstream
    consumers can rely on stable schema regardless of which failure
    path produced the record.
    """
    from services.calendar_leg_picker import CANDIDATE_FRONT_MIN_DTE_DAYS
    return {
        "legacy_selection": None,
        "candidate_selection": None,
        "pickers_diverged": False,
        "shadow_status": shadow_status,
        "candidate_min_front_dte_days": int(CANDIDATE_FRONT_MIN_DTE_DAYS),
        "experimental_note": _DUAL_PICKER_EXPERIMENTAL_NOTE,
    }


def _dual_picker_calendar_selection(
    chain_df: pd.DataFrame,
    event_date: pd.Timestamp,
    *,
    side: str = "call",
) -> Dict[str, Any]:
    """Run the calendar_leg_picker module with BOTH variants (legacy +
    candidate) and report what each would have selected on this chain.

    This is shadow-mode logging. It does NOT replace the existing
    `_select_pre_earnings_calendar_contracts` path — that function
    remains the canonical selector for the historical simulation. The
    output here is for forward-validation evidence only: which picker
    rule would have chosen which contracts on each event.

    The candidate variant uses ``CANDIDATE_FRONT_MIN_DTE_DAYS = 14``
    (PR-AB finding, in-sample, NOT out-of-sample validated). Treat the
    `candidate_selection` field as experimental. The promotion criteria
    that would let the candidate become canonical are forthcoming in
    PR-AC commit 5; until then, see the PR-AC pull request description.

    NOTE: this helper is parameterized by ``side`` but the calling
    simulation currently only queries the calls chain (a known call/put
    correctness gap flagged in the PR-AC review). Wiring put-side
    dual-pick requires also pulling the put chain from the feature
    store; that is deferred to a later commit on this same PR.

    Returns a JSON-safe dict with stable shape — both selection slots are
    always present (None when the picker returned nothing), plus the
    `pickers_diverged` boolean and a labeled experimental note.
    """
    from services.calendar_leg_picker import (
        CANDIDATE_FRONT_MIN_DTE_DAYS,
        PICKER_CANDIDATE,
        PICKER_LEGACY,
        select_calendar_contracts,
    )

    ev = pd.Timestamp(event_date).date() if event_date is not None else None
    try:
        legacy = select_calendar_contracts(
            chain_df,
            event_date=ev,
            side=side,
            picker_variant=PICKER_LEGACY,
        )
        candidate = select_calendar_contracts(
            chain_df,
            event_date=ev,
            side=side,
            picker_variant=PICKER_CANDIDATE,
        )
    except Exception as exc:  # defensive: shadow logging must never break sim
        logger.debug("dual-picker shadow logging failed: %s", exc)
        return _empty_dual_picker(f"error:{type(exc).__name__}")

    diverged = False
    if legacy is not None and candidate is not None:
        diverged = (
            legacy.front_expiry != candidate.front_expiry
            or legacy.back_expiry != candidate.back_expiry
            or float(legacy.strike) != float(candidate.strike)
        )

    return {
        "legacy_selection": legacy.to_metadata_dict() if legacy is not None else None,
        "candidate_selection": candidate.to_metadata_dict() if candidate is not None else None,
        "pickers_diverged": bool(diverged),
        "shadow_status": "ok",
        "candidate_min_front_dte_days": int(CANDIDATE_FRONT_MIN_DTE_DAYS),
        "experimental_note": _DUAL_PICKER_EXPERIMENTAL_NOTE,
    }


def _lookup_exact_contract_row(
    chain_df: pd.DataFrame,
    *,
    expiry: pd.Timestamp,
    strike: float,
    call_put: str = "C",
) -> Optional[pd.Series]:
    df = _prepare_feature_chain(chain_df, call_put=call_put)
    if df.empty:
        return None
    subset = df[(df["expiry"] == pd.Timestamp(expiry).normalize()) & np.isclose(df["strike"], float(strike), atol=1e-8)]
    return _best_contract_row(subset)


def _simulate_pre_earnings_calendar_trade(
    symbol: str,
    store: Any,
    *,
    event_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_offset: int,
) -> Dict[str, Any]:
    base = {
        "symbol": symbol,
        "event_date": pd.Timestamp(event_date).date().isoformat(),
        "entry_date": pd.Timestamp(entry_date).date().isoformat(),
        "exit_date": pd.Timestamp(exit_date).date().isoformat(),
        "entry_offset_days": int(entry_offset),
        "exit_offset_days": int(EXPANSION_EXIT_OFFSET),
    }

    try:
        entry_chain = store.query_chain(symbol, trade_date=base["entry_date"], call_put="C", limit=5000)
    except Exception as exc:
        # No entry chain → no dual-picker possible. Use the canonical
        # empty-block factory so the shape matches every other return.
        return {
            **base,
            "status": "missing_entry_chain",
            "reason": str(exc),
            "dual_picker": _empty_dual_picker("skipped:missing_entry_chain"),
        }

    # Shadow-mode dual-picker logging. Computed ONCE from the entry chain,
    # attached to every subsequent return so per-event records have a
    # uniform shape. The existing simulation below continues to use the
    # legacy `_select_pre_earnings_calendar_contracts` selector — this is
    # comparison data, NOT a behavior change.
    base["dual_picker"] = _dual_picker_calendar_selection(
        entry_chain, event_date=event_date, side="call",
    )

    selection = _select_pre_earnings_calendar_contracts(entry_chain, event_date=pd.Timestamp(event_date))
    if selection is None:
        return {**base, "status": "structural_invalid", "reason": "No same-strike pre-earnings calendar pair"}

    front_row = selection["front_row"]
    back_row = selection["back_row"]
    strike = float(selection["strike"])
    front_expiry = selection["front_expiry"]
    back_expiry = selection["back_expiry"]

    entry_front_mid = _safe_float(front_row.get("mid"), np.nan)
    entry_back_mid = _safe_float(back_row.get("mid"), np.nan)
    entry_front_bid = _safe_float(front_row.get("bid"), np.nan)
    entry_front_ask = _safe_float(front_row.get("ask"), np.nan)
    entry_back_bid = _safe_float(back_row.get("bid"), np.nan)
    entry_back_ask = _safe_float(back_row.get("ask"), np.nan)
    entry_debit_mid = float(entry_back_mid - entry_front_mid)
    entry_debit_adjusted = float(entry_back_ask - entry_front_bid)

    if not np.isfinite(entry_debit_mid) or entry_debit_mid <= 0:
        return {
            **base,
            "status": "negative_debit",
            "reason": "Entry debit is non-positive",
            "front_expiry": front_expiry.date().isoformat(),
            "back_expiry": back_expiry.date().isoformat(),
            "strike": strike,
        }

    try:
        exit_chain = store.query_chain(symbol, trade_date=base["exit_date"], call_put="C", limit=5000)
    except Exception as exc:
        return {**base, "status": "missing_exit_chain", "reason": str(exc)}

    exit_front = _lookup_exact_contract_row(exit_chain, expiry=front_expiry, strike=strike, call_put="C")
    exit_back = _lookup_exact_contract_row(exit_chain, expiry=back_expiry, strike=strike, call_put="C")
    if exit_front is None or exit_back is None:
        return {
            **base,
            "status": "missing_exit_quote",
            "reason": "Exact exit quote unavailable",
            "front_expiry": front_expiry.date().isoformat(),
            "back_expiry": back_expiry.date().isoformat(),
            "strike": strike,
        }

    exit_front_mid = _safe_float(exit_front.get("mid"), np.nan)
    exit_back_mid = _safe_float(exit_back.get("mid"), np.nan)
    exit_front_bid = _safe_float(exit_front.get("bid"), np.nan)
    exit_front_ask = _safe_float(exit_front.get("ask"), np.nan)
    exit_back_bid = _safe_float(exit_back.get("bid"), np.nan)
    exit_back_ask = _safe_float(exit_back.get("ask"), np.nan)
    exit_value_mid = float(exit_back_mid - exit_front_mid)
    exit_value_adjusted = float(exit_back_bid - exit_front_ask)

    if not (np.isfinite(exit_value_mid) and np.isfinite(exit_value_adjusted)):
        return {
            **base,
            "status": "missing_exit_quote",
            "reason": "Exit quotes incomplete",
            "front_expiry": front_expiry.date().isoformat(),
            "back_expiry": back_expiry.date().isoformat(),
            "strike": strike,
        }

    return {
        **base,
        "status": "priceable",
        "option_type": "C",
        "strike": strike,
        "front_expiry": front_expiry.date().isoformat(),
        "back_expiry": back_expiry.date().isoformat(),
        "entry_front_iv": _safe_float(front_row.get("iv"), np.nan),
        "entry_back_iv": _safe_float(back_row.get("iv"), np.nan),
        "exit_front_iv": _safe_float(exit_front.get("iv"), np.nan),
        "exit_back_iv": _safe_float(exit_back.get("iv"), np.nan),
        "entry_debit_mid": entry_debit_mid,
        "entry_debit_adjusted": entry_debit_adjusted,
        "exit_value_mid": exit_value_mid,
        "exit_value_adjusted": exit_value_adjusted,
        "mid_pnl": float(exit_value_mid - entry_debit_mid),
        "adjusted_pnl": float(exit_value_adjusted - entry_debit_adjusted),
        "iv_change_front": float(_safe_float(exit_front.get("iv"), np.nan) - _safe_float(front_row.get("iv"), np.nan)),
        "iv_change_back": float(_safe_float(exit_back.get("iv"), np.nan) - _safe_float(back_row.get("iv"), np.nan)),
        "entry_front_spread_pct": _safe_float(front_row.get("spread_pct"), np.nan),
        "entry_back_spread_pct": _safe_float(back_row.get("spread_pct"), np.nan),
        "exit_front_spread_pct": _safe_float(exit_front.get("spread_pct"), np.nan),
        "exit_back_spread_pct": _safe_float(exit_back.get("spread_pct"), np.nan),
    }


def _summarize_pre_earnings_expansion(
    symbol: str,
    close: pd.Series,
    earnings_events: List[Any],
    feature_store: Optional[Any],
    *,
    entry_offsets: Tuple[int, ...] = EXPANSION_ENTRY_OFFSETS,
    exit_offset: int = EXPANSION_EXIT_OFFSET,
) -> Dict[str, Any]:
    summary = {
        "available": False,
        "structure": "same_strike_call_calendar",
        "entry_offsets_evaluated": list(entry_offsets),
        "exit_offset_days": int(exit_offset),
        "selected_entry_offset_days": None,
        "historical_sample_size": 0,
        "priceable_trades": 0,
        "missing_exit_quotes": 0,
        "excluded_negative_debit": 0,
        "excluded_structural_invalid": 0,
        "expected_iv_change": None,
        "expected_pnl_mid": None,
        "expected_pnl_adjusted": None,
        "expected_return_mid_pct": None,
        "expected_return_adjusted_pct": None,
        "ranking_score": None,
        "trade_examples": [],
        "status": "unavailable",
    }
    if feature_store is None:
        summary["status"] = "feature_store_unavailable"
        return summary

    sessions_series = _normalize_close_index(close)
    sessions = pd.DatetimeIndex(sessions_series.index)
    if sessions.empty:
        summary["status"] = "no_price_sessions"
        return summary

    parsed_events: List[pd.Timestamp] = []
    today = pd.Timestamp(_utc_today_date())
    for item in earnings_events or []:
        event_raw = item.get("event_date") if isinstance(item, dict) else item
        if event_raw is None:
            continue
        event_ts = pd.Timestamp(event_raw).normalize()
        if event_ts <= today:
            parsed_events.append(event_ts)
    parsed_events = sorted(set(parsed_events))[-16:]
    if not parsed_events:
        summary["status"] = "no_historical_events"
        return summary

    offset_summaries: List[Dict[str, Any]] = []
    # Flat collection across all offsets — used by the experimental
    # candidate evidence aggregator after the main offset loop.
    all_trades: List[Dict[str, Any]] = []
    for offset in entry_offsets:
        trades: List[Dict[str, Any]] = []
        for event_ts in parsed_events:
            entry_date = _business_days_before(sessions, event_ts, offset)
            exit_date = _business_days_before(sessions, event_ts, exit_offset)
            if entry_date is None or exit_date is None or entry_date >= exit_date:
                continue
            simulated = _simulate_pre_earnings_calendar_trade(
                symbol,
                feature_store,
                event_date=event_ts,
                entry_date=entry_date,
                exit_date=exit_date,
                entry_offset=offset,
            )
            trades.append(simulated)
            all_trades.append(simulated)

        if not trades:
            continue
        priceable = [trade for trade in trades if trade["status"] == "priceable"]
        missing_exit = sum(1 for trade in trades if trade["status"] == "missing_exit_quote")
        negative_debit = sum(1 for trade in trades if trade["status"] == "negative_debit")
        structural_invalid = sum(1 for trade in trades if trade["status"] == "structural_invalid")
        if priceable:
            avg_entry_debit = float(np.mean([trade["entry_debit_mid"] for trade in priceable]))
            avg_mid = float(np.mean([trade["mid_pnl"] for trade in priceable]))
            avg_adj = float(np.mean([trade["adjusted_pnl"] for trade in priceable]))
            avg_iv_change = float(np.mean([
                np.nanmean([trade["iv_change_front"], trade["iv_change_back"]]) for trade in priceable
            ]))
            expected_mid_return = float((avg_mid / avg_entry_debit) * 100.0) if avg_entry_debit > 0 else np.nan
            expected_adj_return = float((avg_adj / avg_entry_debit) * 100.0) if avg_entry_debit > 0 else np.nan
            win_rate_adj = float(np.mean([1.0 if trade["adjusted_pnl"] > 0 else 0.0 for trade in priceable]))
            score = float(np.clip(
                0.45 * np.clip((expected_adj_return + 10.0) / 25.0, 0.0, 1.0)
                + 0.25 * np.clip((expected_mid_return + 10.0) / 25.0, 0.0, 1.0)
                + 0.15 * np.clip(win_rate_adj, 0.0, 1.0)
                + 0.15 * np.clip(len(priceable) / 8.0, 0.0, 1.0),
                0.0,
                1.0,
            ))
        else:
            avg_mid = np.nan
            avg_adj = np.nan
            avg_iv_change = np.nan
            expected_mid_return = np.nan
            expected_adj_return = np.nan
            score = 0.0

        offset_summaries.append(
            {
                "entry_offset_days": int(offset),
                "historical_sample_size": len(trades),
                "priceable_trades": len(priceable),
                "missing_exit_quotes": missing_exit,
                "excluded_negative_debit": negative_debit,
                "excluded_structural_invalid": structural_invalid,
                "expected_iv_change": float(avg_iv_change) if np.isfinite(avg_iv_change) else None,
                "expected_pnl_mid": float(avg_mid) if np.isfinite(avg_mid) else None,
                "expected_pnl_adjusted": float(avg_adj) if np.isfinite(avg_adj) else None,
                "expected_return_mid_pct": float(expected_mid_return) if np.isfinite(expected_mid_return) else None,
                "expected_return_adjusted_pct": float(expected_adj_return) if np.isfinite(expected_adj_return) else None,
                "ranking_score": float(score),
                "trade_examples": priceable[:3],
            }
        )

    if not offset_summaries:
        summary["status"] = "no_simulations"
        # Even with no canonical simulations, attach an empty experimental
        # evidence block for stable response shape.
        summary["experimental_candidate_evidence"] = _aggregate_experimental_candidate_evidence([])
        return summary

    best = max(
        offset_summaries,
        key=lambda item: (
            item["priceable_trades"] >= MIN_PRICEABLE_EXPANSION_TRADES,
            _safe_float(item["expected_pnl_adjusted"], -np.inf),
            _safe_float(item["ranking_score"], 0.0),
        ),
    )
    summary.update(best)
    summary["selected_entry_offset_days"] = best["entry_offset_days"]
    summary["available"] = bool(best["priceable_trades"] > 0)
    summary["status"] = "ok" if summary["available"] else "insufficient_priceable_trades"

    # Shadow-mode evidence aggregation. Collected from every simulated
    # trade across every offset, attached as a SEPARATE top-level field
    # so it cannot be confused with the canonical legacy expectation
    # numbers. This is forward-validation evidence — diverged-event
    # counts, not promotion-ready performance claims.
    summary["experimental_candidate_evidence"] = _aggregate_experimental_candidate_evidence(
        all_trades
    )

    return summary


def _aggregate_experimental_candidate_evidence(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate dual-picker shadow logs into an evidence block.

    Each input trade dict is expected to carry a ``dual_picker`` field
    populated by ``_dual_picker_calendar_selection``. The aggregation
    distinguishes two count axes:

      * trade_counts: per simulated trade record. The same earnings
        event evaluated at N different entry-offsets produces N
        records, so this is inflated by the offset cardinality.
      * event_counts: deduplicated by (symbol, event_date). This is
        the count that should drive any promotion criterion — independent
        earnings events, not entry-offset replication.

    Picker outcome counts (legacy_succeeded, candidate_succeeded, etc.)
    only include trades whose ``shadow_status`` == "ok". Data-availability
    failures (``skipped:missing_entry_chain``, ``error:*``) are tallied
    separately under ``shadow_status_counts`` so they cannot be
    mistaken for picker failures."""
    # Trade-level (per offset × event)
    n_total = 0
    n_legacy_only = 0
    n_candidate_only = 0
    n_both = 0
    n_neither = 0
    n_diverged = 0

    # Shadow-status breakdown across ALL trades (incl. non-ok)
    shadow_status_counts: Dict[str, int] = {}

    # Event-level (deduplicated on (symbol, event_date))
    unique_events: set = set()
    unique_events_with_legacy: set = set()
    unique_events_with_candidate: set = set()
    unique_events_diverged: set = set()
    unique_events_picker_evaluated: set = set()  # at least one offset had shadow_status == "ok"

    diverged_sample: List[Dict[str, Any]] = []

    for trade in trades:
        dp = trade.get("dual_picker") or {}
        status = dp.get("shadow_status")
        shadow_status_counts[status or "missing"] = shadow_status_counts.get(status or "missing", 0) + 1

        event_key = (trade.get("symbol"), trade.get("event_date"))
        if event_key != (None, None):
            unique_events.add(event_key)

        # Outcome bucketing only applies when shadow logging actually ran.
        if status != "ok":
            continue

        n_total += 1
        unique_events_picker_evaluated.add(event_key)
        legacy_sel = dp.get("legacy_selection")
        candidate_sel = dp.get("candidate_selection")
        legacy_present = legacy_sel is not None
        candidate_present = candidate_sel is not None
        if legacy_present:
            unique_events_with_legacy.add(event_key)
        if candidate_present:
            unique_events_with_candidate.add(event_key)
        if legacy_present and candidate_present:
            n_both += 1
        elif legacy_present:
            n_legacy_only += 1
        elif candidate_present:
            n_candidate_only += 1
        else:
            n_neither += 1
        if dp.get("pickers_diverged"):
            n_diverged += 1
            unique_events_diverged.add(event_key)
            if len(diverged_sample) < 5:
                diverged_sample.append(
                    {
                        "event_date": trade.get("event_date"),
                        "entry_date": trade.get("entry_date"),
                        "legacy_front_expiry": (legacy_sel or {}).get("front_expiry"),
                        "candidate_front_expiry": (candidate_sel or {}).get("front_expiry"),
                        "legacy_back_expiry": (legacy_sel or {}).get("back_expiry"),
                        "candidate_back_expiry": (candidate_sel or {}).get("back_expiry"),
                        "legacy_strike": (legacy_sel or {}).get("strike"),
                        "candidate_strike": (candidate_sel or {}).get("strike"),
                        "legacy_front_dte": (legacy_sel or {}).get("front_dte_days"),
                        "candidate_front_dte": (candidate_sel or {}).get("front_dte_days"),
                    }
                )

    return {
        "note": (
            "Shadow comparison of two calendar leg-picker rules. The "
            "candidate rule (candidate_min_dte, +14d min front DTE) is "
            "in-sample on PR-AB's 10-symbol/139-event dataset and NOT "
            "out-of-sample validated. Do not interpret divergence counts "
            "as performance claims; promotion criteria are defined in the "
            "PR-AC pull request description.\n\n"
            "IMPORTANT: trade_counts.* are inflated by entry-offset "
            "replication — the same earnings event evaluated at multiple "
            "entry offsets produces multiple trade records. Use "
            "event_counts.unique_events_* for promotion-criterion "
            "thresholds (n_diverged, n_total) to avoid double-counting."
        ),
        "picker_legacy": "legacy_first_expiry",
        "picker_candidate": "candidate_min_dte",
        # Per-trade-record (offset-inflated). Useful for understanding
        # the simulation's overall picker behavior but NOT for evidence-
        # threshold checks.
        "trade_counts": {
            "total_picker_evaluated": int(n_total),
            "both_succeeded": int(n_both),
            "legacy_only": int(n_legacy_only),
            "candidate_only": int(n_candidate_only),
            "neither_succeeded": int(n_neither),
            "pickers_diverged": int(n_diverged),
        },
        # Per-earnings-event (deduplicated). THIS is the count that
        # promotion criteria should use.
        "event_counts": {
            "unique_events_observed": len(unique_events),
            "unique_events_picker_evaluated": len(unique_events_picker_evaluated),
            "unique_events_with_legacy_selection": len(unique_events_with_legacy),
            "unique_events_with_candidate_selection": len(unique_events_with_candidate),
            "unique_events_diverged": len(unique_events_diverged),
        },
        # Data-availability accounting (entry chain missing, helper
        # exceptions, etc.) — separated so it cannot be confused with
        # picker outcomes.
        "shadow_status_counts": {
            k: int(v) for k, v in sorted(shadow_status_counts.items())
        },
        "diverged_sample": diverged_sample,
    }


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


# _score_setup() was removed.
# The function was defined here but never called — the pipeline uses
# _shared_setup_score_from_snapshot() from web.api.edge_legacy_adapter,
# which delegates to the screener's shared ranking score. Keeping a dead
# parallel scoring path created auditable confusion about which weights
# were actually in use. The registry entries (setup_iv_rv_baseline etc.)
# have been removed from _HEURISTIC_THRESHOLDS for the same reason.


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

        # Fix 4: collect OI and bid-ask alongside IV to build liquidity weights.
        # Equal-weight polyfit lets illiquid far-wing strikes (often garbage quotes)
        # distort the curvature estimate.  We weight by sqrt(OI) so liquid ATM
        # strikes dominate the fit.  Fallback: 1/(1+spread_pct) when OI is absent.
        iv_by_strike: Dict[float, List[float]] = {}
        oi_by_strike: Dict[float, float] = {}
        spread_by_strike: Dict[float, float] = {}
        for _, row in grp.iterrows():
            strike = _safe_float(row.get("strike"), np.nan)
            iv = _safe_float(row.get("impliedVolatility"), np.nan)
            if not np.isfinite(strike) or strike <= 0 or not np.isfinite(iv) or iv <= 0:
                continue
            k = float(strike)
            iv_by_strike.setdefault(k, []).append(float(iv))
            oi = _safe_float(row.get("openInterest"), np.nan)
            if np.isfinite(oi) and oi >= 0:
                oi_by_strike[k] = oi_by_strike.get(k, 0.0) + float(oi)
            bid = _safe_float(row.get("bid"), np.nan)
            ask = _safe_float(row.get("ask"), np.nan)
            if np.isfinite(bid) and np.isfinite(ask) and ask > 0 and ask > bid:
                mid = (bid + ask) / 2.0
                sp = (ask - bid) / mid * 100.0
                # average spread across call/put at the same strike
                spread_by_strike[k] = (spread_by_strike.get(k, sp) + sp) / 2.0

        rows: List[Tuple[float, float, float]] = []  # (moneyness, iv, weight)
        for strike, values in iv_by_strike.items():
            iv_mean = float(np.mean(values))
            moneyness = float((strike / max(current_price, 1e-6)) - 1.0)
            if abs(moneyness) <= 0.20:
                oi_val = oi_by_strike.get(strike, np.nan)
                sp_val = spread_by_strike.get(strike, np.nan)
                if np.isfinite(oi_val) and oi_val > 0:
                    w = float(np.sqrt(oi_val))
                elif np.isfinite(sp_val) and sp_val > 0:
                    w = float(1.0 / (1.0 + sp_val))
                else:
                    w = 1.0  # uniform fallback
                rows.append((moneyness, iv_mean, w))

        if len(rows) < 5:
            continue

        rows.sort(key=lambda x: x[0])
        x = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        w = np.array([r[2] for r in rows], dtype=float)
        # Normalise weights to sum to len(rows) so the effective sample size is
        # unchanged and the curvature coefficient remains on the same scale.
        w = w / (w.sum() / len(w)) if w.sum() > 0 else np.ones_like(w)
        try:
            a, _, _ = np.polyfit(x, y, 2, w=w)
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

        # Fix 4 (yfinance path): same liquidity-weighted polyfit as MDA path.
        iv_by_strike: Dict[float, List[float]] = {}
        oi_by_strike: Dict[float, float] = {}
        spread_by_strike: Dict[float, float] = {}
        for frame in prepared_frames:
            for _, row in frame.iterrows():
                strike = _safe_float(row.get("strike"), np.nan)
                iv = _safe_float(row.get("impliedVolatility"), np.nan)
                if not np.isfinite(strike) or strike <= 0 or not np.isfinite(iv) or iv <= 0:
                    continue
                k = float(strike)
                iv_by_strike.setdefault(k, []).append(float(iv))
                oi = _safe_float(row.get("openInterest"), np.nan)
                if np.isfinite(oi) and oi >= 0:
                    oi_by_strike[k] = oi_by_strike.get(k, 0.0) + float(oi)
                bid = _safe_float(row.get("bid"), np.nan)
                ask = _safe_float(row.get("ask"), np.nan)
                if np.isfinite(bid) and np.isfinite(ask) and ask > 0 and ask > bid:
                    mid = (bid + ask) / 2.0
                    sp = (ask - bid) / mid * 100.0
                    spread_by_strike[k] = (spread_by_strike.get(k, sp) + sp) / 2.0

        rows: List[Tuple[float, float, float]] = []
        for strike, values in iv_by_strike.items():
            moneyness = float((strike / max(current_price, 1e-6)) - 1.0)
            if abs(moneyness) <= 0.20:
                oi_val = oi_by_strike.get(strike, np.nan)
                sp_val = spread_by_strike.get(strike, np.nan)
                if np.isfinite(oi_val) and oi_val > 0:
                    w = float(np.sqrt(oi_val))
                elif np.isfinite(sp_val) and sp_val > 0:
                    w = float(1.0 / (1.0 + sp_val))
                else:
                    w = 1.0
                rows.append((moneyness, float(np.mean(values)), w))

        if len(rows) < 5:
            continue

        rows.sort(key=lambda x: x[0])
        x = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        w = np.array([r[2] for r in rows], dtype=float)
        w = w / (w.sum() / len(w)) if w.sum() > 0 else np.ones_like(w)
        try:
            a, _, _ = np.polyfit(x, y, 2, w=w)
        except Exception:
            continue

        return {"curvature": float(a), "concave": bool(float(a) < -0.5), "points": int(len(rows))}

    return empty


def _next_earnings_days_yf(ticker: yf.Ticker) -> Optional[int]:
    """Resolve next earnings DTE through the shared event service, then fall back safely."""
    today = _utc_today_date()
    symbol = getattr(ticker, "ticker", None) or getattr(ticker, "symbol", None)
    if symbol:
        try:
            resolved = resolve_upcoming_earnings_event(
                str(symbol),
                today,
                today + timedelta(days=120),
                ticker=ticker,
            )
            if resolved.earnings_date is not None:
                return int((resolved.earnings_date - today).days)
        except Exception as exc:
            logger.debug("Shared earnings event resolution failed for %s: %s", symbol, exc)
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

def _bsm_greeks(
    S: float, T_days: float, sigma: float, r: float = 0.045
) -> Dict[str, Optional[float]]:
    """
    Black-Scholes-Merton ATM option greeks (K = S).

    Parameters
    ----------
    S      : underlying price
    T_days : calendar days to option expiry (use earnings DTE for calendar spread context)
    sigma  : annualised implied vol (iv30)
    r      : risk-free rate (default fallback only; callers normally inject a live rate)

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
) -> Optional[Dict[str, Any]]:
    """
    ATM calendar spread payoff at near-leg expiry.

    side='call' (default): short near ATM call / long back ATM call
    side='put':            short near ATM put  / long back ATM put

    Both legs are priced at K = S (ATM).  Put prices derived via
    put-call parity: P = C − S + K·exp(−rT).

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
            d1 = (math.log(spot / K) + (r + 0.5 * sigma ** 2) * T_yr) / (sigma * sq_T)
            d2 = d1 - sigma * sq_T
            return spot * _ncdf(d1) - K * math.exp(-r * T_yr) * _ncdf(d2)

        def _bsm_put(spot: float, T_yr: float, sigma: float) -> float:
            return _bsm_call(spot, T_yr, sigma) - spot + K * math.exp(-r * T_yr)

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
) -> Optional[Dict[str, Any]]:
    """
    Long ATM straddle payoff for an earnings play.

    Entry: buy ATM call + buy ATM put (K = S, T = T_near_days, σ = iv).
    P&L evaluated 1 day post-event: stock has moved, IV has shifted per scenario.

    The 1-day residual convention (rather than full expiry) captures the
    dominant risk for earnings straddles — IV crush kills the position even
    when the stock moves correctly.  IV scenarios come from the same
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
            d1 = (math.log(spot / K) + (r + 0.5 * sigma ** 2) * T_yr) / (sigma * sq_T)
            d2 = d1 - sigma * sq_T
            return spot * _ncdf(d1) - K * math.exp(-r * T_yr) * _ncdf(d2)

        def _bsm_put(spot: float, T_yr: float, sigma: float) -> float:
            return _bsm_call(spot, T_yr, sigma) - spot + K * math.exp(-r * T_yr)

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
) -> Optional[Dict[str, Any]]:
    """
    Long OTM strangle payoff for an earnings play.

    Wings placed at ±wing_pct% from ATM (typically = implied_move_pct so the
    strangle only profits when the stock exceeds the market's priced move —
    a pure earnings-surprise bet).

    Entry: buy OTM call at K_c = S*(1+wing/100) and OTM put at K_p = S*(1-wing/100).
    P&L evaluated 1 day post-event with IV scenarios (same as straddle).
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
            d1 = (math.log(spot / K) + (r + 0.5 * sigma ** 2) * T_yr) / (sigma * sq_T)
            d2 = d1 - sigma * sq_T
            return spot * _ncdf(d1) - K * math.exp(-r * T_yr) * _ncdf(d2)

        def _bsm_put(spot: float, K: float, T_yr: float, sigma: float) -> float:
            return _bsm_call(spot, K, T_yr, sigma) - spot + K * math.exp(-r * T_yr)

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


# Kelly sizing has been removed.
# Reason: the formula requires a calibrated edge estimate. The expected_return_signal_pct
# formula (18× multiplier) overstates empirical edge by 10–20× (ORATS best-filtered
# straddle: +0.91% gross; our formula: up to +9%). Feeding an overestimated edge into
# Kelly produces 2–10× Kelly, which targets zero or negative long-run growth.
# Position sizing guidance will be re-introduced once isotonic calibration reaches N≥500
# and edge estimates are validated against a live out-of-sample period.


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


def _collect_yf_option_chain_frame(
    ticker: yf.Ticker,
    *,
    as_of_date: date,
    max_expiries: int = 6,
) -> pd.DataFrame:
    expirations = list(getattr(ticker, "options", []) or [])
    rows: List[Dict[str, Any]] = []
    for expiry in expirations[:max_expiries]:
        start = time.perf_counter()
        try:
            chain = ticker.option_chain(expiry)
            record_provider_telemetry(
                provider_name="yfinance",
                endpoint_type="options_chain",
                symbol=getattr(ticker, "ticker", None),
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                fallback_used=True,
                response_quality_note="research-grade option-chain fallback",
            )
        except Exception as exc:
            record_provider_telemetry(
                provider_name="yfinance",
                endpoint_type="options_chain",
                symbol=getattr(ticker, "ticker", None),
                success=False,
                error_category=classify_error(str(exc)),
                latency_ms=(time.perf_counter() - start) * 1000.0,
                fallback_used=True,
                response_quality_note="research-grade option-chain fallback",
            )
            continue
        for side, frame in (("C", chain.calls), ("P", chain.puts)):
            if frame is None or frame.empty:
                continue
            working = frame.copy()
            for col in ("strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"):
                if col in working.columns:
                    working[col] = pd.to_numeric(working.get(col), errors="coerce")
            for _, row in working.iterrows():
                bid = row.get("bid")
                ask = row.get("ask")
                mid = np.nan
                if pd.notna(bid) and pd.notna(ask) and ask >= bid and ask > 0:
                    mid = (float(bid) + float(ask)) / 2.0
                else:
                    last_price = row.get("lastPrice")
                    if pd.notna(last_price):
                        mid = float(last_price)
                rows.append(
                    {
                        "trade_date": as_of_date.isoformat(),
                        "expiry": str(expiry)[:10],
                        "call_put": side,
                        "strike": row.get("strike"),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "iv": row.get("impliedVolatility"),
                        "open_interest": row.get("openInterest"),
                        "volume": row.get("volume"),
                    }
                )
    return pd.DataFrame(rows)


def analyze_single_ticker(
    symbol: str,
    mda_client: Any = None,  # MarketDataClient | None
    record_to_ledger: bool = True,
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
    _hist_start = time.perf_counter()
    try:
        hist = ticker.history(period="6mo", auto_adjust=True)
        record_provider_telemetry(
            provider_name="yfinance",
            endpoint_type="price_history",
            symbol=clean_symbol,
            success=hist is not None and not hist.empty,
            error_category=None if hist is not None and not hist.empty else "empty_response",
            latency_ms=(time.perf_counter() - _hist_start) * 1000.0,
            response_quality_note="6mo OHLCV for RV baseline",
        )
    except Exception as exc:
        record_provider_telemetry(
            provider_name="yfinance",
            endpoint_type="price_history",
            symbol=clean_symbol,
            success=False,
            error_category=classify_error(str(exc)),
            latency_ms=(time.perf_counter() - _hist_start) * 1000.0,
            response_quality_note="6mo OHLCV for RV baseline",
        )
        raise
    if hist is None or hist.empty:
        raise ValueError(f"No market data for {clean_symbol}")

    close = pd.to_numeric(hist.get("Close"), errors="coerce").dropna()
    if close.empty:
        raise ValueError(f"No close prices for {clean_symbol}")

    _price_stale = False
    _price_age_days: Optional[int] = None

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
        _hist_long_start = time.perf_counter()
        hist_long = ticker.history(period="5y", auto_adjust=True)
        record_provider_telemetry(
            provider_name="yfinance",
            endpoint_type="price_history_long",
            symbol=clean_symbol,
            success=hist_long is not None and not hist_long.empty,
            error_category=None if hist_long is not None and not hist_long.empty else "empty_response",
            latency_ms=(time.perf_counter() - _hist_long_start) * 1000.0,
            response_quality_note="5y OHLCV for earnings-move history",
        )
        if hist_long is not None and not hist_long.empty:
            close_long = pd.to_numeric(hist_long.get("Close"), errors="coerce").dropna()
            if not close_long.empty:
                close_for_profile = close_long
    except Exception as exc:
        record_provider_telemetry(
            provider_name="yfinance",
            endpoint_type="price_history_long",
            symbol=clean_symbol,
            success=False,
            error_category=classify_error(str(exc)),
            latency_ms=(time.perf_counter() - _hist_long_start) * 1000.0 if "_hist_long_start" in locals() else None,
            response_quality_note="5y OHLCV for earnings-move history",
        )
        logger.warning("5-year history fetch failed for %s (vol regime degraded): %s", clean_symbol, exc)
        hist_long = None

    # Yang-Zhang 30-day RV (primary — ~5x more efficient than close-to-close)
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

    # ── 3. Earnings dates + BMO/AMC timing ───────────────────────────────────
    dte: Optional[int] = None
    earnings_release_time: Optional[str] = None
    earnings_events_for_profile: List[Dict[str, Any]] = []
    earnings_metadata_for_snapshot: Optional[Any] = None
    resolved_earnings_event = None

    if use_mda:
        try:
            earnings_df = client.get_earnings(clean_symbol, countback=24)
            if earnings_df is not None and not earnings_df.empty:
                dte, earnings_release_time = _next_earnings_from_mda(earnings_df)
                earnings_events_for_profile = _earnings_dates_from_mda(earnings_df)
        except Exception as exc:
            logger.warning("MDApp earnings fetch failed for %s: %s", clean_symbol, exc)

    try:
        resolved_earnings_event = resolve_upcoming_earnings_event(
            clean_symbol,
            _utc_today_date(),
            _utc_today_date() + timedelta(days=120),
            ticker=ticker,
            mda_client=client if use_mda else None,
        )
    except Exception as exc:
        logger.debug("Shared earnings event resolution failed for %s: %s", clean_symbol, exc)

    # yfinance fallback — two independent checks:
    #   1. If DTE not yet resolved (MDApp earnings returned nothing), use yfinance for next date.
    #   2. If historical events list is empty (MDApp only returns the next FUTURE date, not past
    #      history on this plan tier), always try yfinance for the move-profile events regardless
    #      of whether MDApp already provided the next-DTE.
    if dte is None and resolved_earnings_event is not None and resolved_earnings_event.earnings_date is not None:
        dte = int((resolved_earnings_event.earnings_date - _utc_today_date()).days)
    if dte is None:
        dte = _next_earnings_days_yf(ticker)
    if (not earnings_release_time or earnings_release_time == "unknown") and resolved_earnings_event is not None:
        earnings_release_time = _normalize_release_timing(resolved_earnings_event.release_timing)
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

    earnings_date: Optional[date] = (
        resolved_earnings_event.earnings_date
        if resolved_earnings_event is not None and resolved_earnings_event.earnings_date is not None
        else (
            _utc_today_date() + timedelta(days=int(dte))
            if dte is not None and dte >= 0
            else None
        )
    )
    earnings_metadata_for_snapshot = {
        "earnings_date": earnings_date,
        "release_timing": earnings_release_time,
        "prior_events": earnings_events_for_profile,
        "earnings_source_primary": getattr(resolved_earnings_event, "primary_source", None),
        "earnings_source_confirmed": getattr(resolved_earnings_event, "confirmed_source", None),
        "earnings_source_confidence": getattr(resolved_earnings_event, "source_confidence", None),
        "release_timing_source": getattr(resolved_earnings_event, "release_timing_source", None),
        "earnings_source_stale": getattr(resolved_earnings_event, "source_stale", False),
        "earnings_source_notes": getattr(resolved_earnings_event, "source_notes", []),
    }

    price_snapshot_frame = (hist_long if (hist_long is not None and not hist_long.empty) else hist).reset_index().rename(
        columns={
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    option_snapshot_frame = chain_df if chain_df is not None else _collect_yf_option_chain_frame(
        ticker, as_of_date=_utc_today_date()
    )
    vol_snapshot = build_vol_snapshot(
        clean_symbol,
        _utc_today_date(),
        option_chain_data=option_snapshot_frame,
        earnings_metadata=earnings_metadata_for_snapshot,
        price_data=price_snapshot_frame,
    )
    structure_scorecards = build_structure_scorecards(vol_snapshot)
    selector_output = select_best_structure(vol_snapshot, structure_scorecards)
    lead_scorecard = _selector_lead_scorecard(structure_scorecards, selector_output)
    snapshot_inputs = _snapshot_to_edge_inputs(vol_snapshot)

    # Legacy chart arrays are still populated from chain data for the current UI.
    # Recommendation logic below reads canonical feature state from vol_snapshot.
    if chain_df is not None:
        (
            days, ivs, _, _, _, _,
            _, _, _,
            _, _,
            ts_slope_near_dte, ts_slope_far_dte,
        ) = _term_structure_from_mda_chain(chain_df, current_price)
    else:
        (
            days, ivs, _, _, _, _,
            _, _, _,
            _, _,
            ts_slope_near_dte, ts_slope_far_dte,
        ) = _term_structure_points_yf(ticker, current_price)
    options_source = "marketdata_app" if chain_df is not None else "yfinance"

    rv30_yz = _safe_float(snapshot_inputs["rv30_yz"], np.nan)
    rv30_cc = float(close.pct_change().dropna().tail(30).std(ddof=1) * np.sqrt(252))
    rv30 = rv30_yz if np.isfinite(rv30_yz) and rv30_yz > 0.0 else rv30_cc
    rv30 = max(rv30, 1e-4)
    rv_estimator = snapshot_inputs["rv_estimator"] if snapshot_inputs["rv_estimator"] else "close_to_close"
    rv_har_forecast = snapshot_inputs["rv_har_forecast"]
    rv_percentile_rank = snapshot_inputs["rv_percentile_rank"]
    vol_regime = snapshot_inputs["vol_regime"]
    iv30 = _safe_float(snapshot_inputs["iv30"], np.nan)
    iv45 = _safe_float(snapshot_inputs["iv45"], np.nan)
    ts_slope_0_45 = _safe_float(snapshot_inputs["ts_slope_0_45"], np.nan)
    implied_move_pct = _safe_float(snapshot_inputs["implied_move_pct"], np.nan)
    near_term_spread_pct = snapshot_inputs["near_term_spread_pct"]
    near_term_dte = snapshot_inputs["near_term_dte"]
    near_back_iv_ratio = snapshot_inputs["near_back_iv_ratio"]
    near_term_liquidity_proxy = snapshot_inputs["near_term_liquidity_proxy"]
    liquidity = float(snapshot_inputs["liquidity"] or 0.0)
    smile_curvature = _safe_float(snapshot_inputs["smile_curvature"], np.nan)
    smile_state = snapshot_inputs["smile_state"]
    dte = snapshot_inputs["dte"]
    earnings_release_time = snapshot_inputs["earnings_release_time"]
    median_earnings_move_pct = _safe_float(snapshot_inputs["median_earnings_move_pct"], np.nan)
    p90_earnings_move_pct = _safe_float(snapshot_inputs["p90_earnings_move_pct"], np.nan)
    avg_last4_move_pct = _safe_float(snapshot_inputs["avg_last4_move_pct"], np.nan)
    move_std_pct = _safe_float(snapshot_inputs["move_std_pct"], np.nan)
    move_anchor_pct = snapshot_inputs["move_anchor_pct"]
    move_anchor_pct_val = _safe_float(move_anchor_pct, np.nan)
    move_sample_size = int(snapshot_inputs["move_sample_size"] or 0)
    move_source = str(snapshot_inputs["move_source"] or "none")
    move_uncertainty_pct = snapshot_inputs["move_uncertainty_pct"]
    event_implied_move_pct = _safe_float(snapshot_inputs["event_implied_move_pct"], np.nan)
    non_event_move_pct = _safe_float(snapshot_inputs["non_event_move_pct"], np.nan)
    iv_rv = _safe_float(snapshot_inputs["iv_rv"], np.nan)
    iv_rv_har = snapshot_inputs["iv_rv_har"]
    _price_age_days = (
        int(vol_snapshot.price_staleness_minutes // 1440)
        if vol_snapshot.price_staleness_minutes is not None
        else None
    )
    _price_stale = bool(_price_age_days is not None and _price_age_days > STALE_DATA_THRESHOLD_DAYS)

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

    # ── 4. Historical earnings move profile used by legacy crush calibration ──
    # The canonical snapshot already provides the scalar move features used by
    # recommendations. We keep this legacy profile only for the raw move list
    # needed by crush-rate / kurtosis calibration and UI diagnostics.
    move_profile = _historical_earnings_move_profile(close_for_profile, earnings_events_for_profile)

    sample_confidence = float(
        getattr(lead_scorecard, "sample_confidence", 0.10)
        if lead_scorecard is not None
        else (0.10 if move_source != "earnings_history" else float(np.clip(move_sample_size / 8.0, 0.20, 1.0)))
    )

    expansion_summary = _summarize_pre_earnings_expansion(
        clean_symbol,
        close_for_profile,
        earnings_events_for_profile,
        _get_feature_store(),
    )

    # ── 6. Shared setup + legacy diagnostics ─────────────────────────────────
    tx_cost_pct = _estimate_transaction_cost_pct(
        liquidity=liquidity, spread_pct=near_term_spread_pct
    )
    iv_rv = float(iv30 / rv30) if np.isfinite(iv30) and np.isfinite(rv30) and rv30 > 0 else np.nan
    setup_base = _shared_setup_score_from_snapshot(vol_snapshot)
    smile_curvature = _safe_float(smile_state.get("curvature"), np.nan)
    concavity_component = (
        float(np.clip((-smile_curvature - 0.5) / 2.5, 0.0, 1.0))
        if np.isfinite(smile_curvature) else 0.50
    )
    setup_score = float(setup_base)

    implied_move_total_pct = _safe_float(implied_move_pct, np.nan)
    # P-5a: read the canonical event-vol decomposition from the VolSnapshot
    # produced by services.earnings_vol_snapshot.build_vol_snapshot.  The
    # local recomputation that previously lived here used the same
    # dimensionally-incoherent quadrature subtraction and is now obsolete;
    # eliminating it keeps the decomposition in a single source of truth.
    event_implied_move_pct = _safe_float(snapshot_inputs.get("event_implied_move_pct"), np.nan)
    non_event_move_pct = _safe_float(snapshot_inputs.get("non_event_move_pct"), np.nan)

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
    expectancy_score = float(
        getattr(lead_scorecard, "walk_forward_rank_score", np.nan)
        if lead_scorecard is not None
        else _score_expectancy(
            net_edge_pct=expected_net_edge_pct,
            drawdown_risk_pct=drawdown_risk_pct,
            sample_size=int(move_profile.get("event_count", 0) or 0),
        )
    )
    composite_score = float(
        getattr(lead_scorecard, "composite_structure_score", np.nan)
        if lead_scorecard is not None
        else np.clip(0.60 * setup_score + 0.40 * expectancy_score, 0.0, 1.0)
    )
    confidence_pct_raw = float(getattr(selector_output, "confidence_pct", np.clip(30.0 + 70.0 * composite_score, 0.0, 100.0)))

    # ── Move-risk level: soft advisory — how dangerous is the tail relative to the priced-in move? ──
    # Ratio = p90 historical move / event-implied move.
    # > 1.15 → market is under-pricing the historical tail (elevated)
    # 0.90–1.15 → broadly aligned (moderate)
    # < 0.90 → historical tail inside implied move (low)
    # Thin-sample caveat: degrade to "moderate" floor if fewer than 5 events.
    move_risk_level, move_risk_ratio = _classify_move_risk(
        p90_earnings_move_pct=(p90_earnings_move_pct if np.isfinite(p90_earnings_move_pct) else None),
        event_implied_move_pct=(event_implied_move_pct if np.isfinite(event_implied_move_pct) and event_implied_move_pct > 0 else None),
        sample_size=move_sample_size,
    )

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
    confidence_pct = float(getattr(selector_output, "confidence_pct", np.clip(confidence_pct_raw * _calibration_mult, 0.0, 100.0)))

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
            # Fix 5: pass historical moves so scenarios are symbol-calibrated
            raw_moves_pct=_raw_moves if _raw_moves else None,
            implied_move_pct=float(implied_move_total_pct) if np.isfinite(implied_move_total_pct) else None,
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

    # ── Structure-specific payoff for the recommended structure ──────────────
    # Drives the main-view diagram; distinct from calendar_payoff (legacy panel).
    _best_structure = getattr(selector_output, "best_structure", None)
    _sp_near_dte = float(near_term_dte) if (near_term_dte is not None and near_term_dte > 0) else None
    _sp_iv30_ok  = np.isfinite(iv30) and iv30 > 0
    _sp_move_ok  = np.isfinite(implied_move_total_pct) and implied_move_total_pct > 0
    _sp_raw      = _raw_moves if _raw_moves else None
    _sp_impl     = float(implied_move_total_pct) if _sp_move_ok else None

    if _best_structure == "call_calendar":
        structure_payoff: Optional[Dict[str, Any]] = calendar_payoff
    elif _best_structure == "put_calendar" and _sp_iv30_ok and _sp_near_dte and np.isfinite(_cal_iv_back) and _cal_iv_back > 0:
        structure_payoff = _calendar_spread_payoff(
            S=current_price, iv_near=float(iv30), iv_back=float(_cal_iv_back),
            T_near_days=_sp_near_dte, T_back_days=_sp_near_dte + 28.0,
            r=pricing_risk_free_rate, raw_moves_pct=_sp_raw,
            implied_move_pct=_sp_impl, side="put",
        )
    elif _best_structure == "atm_straddle" and _sp_iv30_ok and _sp_near_dte:
        structure_payoff = _straddle_payoff(
            S=current_price, iv=float(iv30), T_near_days=_sp_near_dte,
            r=pricing_risk_free_rate, raw_moves_pct=_sp_raw, implied_move_pct=_sp_impl,
        )
    elif _best_structure == "otm_strangle" and _sp_iv30_ok and _sp_near_dte and _sp_move_ok:
        structure_payoff = _strangle_payoff(
            S=current_price, iv=float(iv30), T_near_days=_sp_near_dte,
            wing_pct=float(implied_move_total_pct),
            r=pricing_risk_free_rate, raw_moves_pct=_sp_raw, implied_move_pct=_sp_impl,
        )
    else:
        structure_payoff = None

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
    # Fix 3: gates 9 & 10 converted from hard-reject to soft confidence caps.
    # Principle: "insufficient evidence" ≠ "no signal".  A stock with 4 earnings
    # events and consistent crush still has information; it deserves a degraded
    # confidence score, not complete suppression.  The caps are applied further
    # down after hard_no_trade is evaluated.
    fallback_move_model_flag: bool = (move_source != "earnings_history")
    low_event_count_flag: bool = (move_sample_size < MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL)

    hard_no_trade = len(hard_gate_reasons) > 0
    confidence_pct_uncapped = confidence_pct
    confidence_capped = False

    expansion_gate_reasons: List[str] = []
    if dte is None:
        expansion_gate_reasons.append("Days to earnings unavailable for pre-earnings timing.")
    elif dte not in EXPANSION_ENTRY_OFFSETS:
        expansion_gate_reasons.append(
            f"Current entry window inactive ({dte} DTE; target {min(EXPANSION_ENTRY_OFFSETS)}-{max(EXPANSION_ENTRY_OFFSETS)})."
        )
    # Low IV/RV is *favorable* for long-vol entry (cheap IV relative to realised vol).
    # Hard-rejecting it contradicted the screener, snapshot cheapness_score, and
    # structure scorecard crowding_intensity — all of which reward low IV/RV.
    # We only warn (softly) when IV/RV is very high, meaning the event is likely
    # already priced in and the position faces crowding risk.
    if np.isfinite(iv_rv) and iv_rv > IV_RV_CROWDING_WARNING_THRESHOLD:
        iv_rv_text = f"{iv_rv:.2f}"
        expansion_gate_reasons.append(
            f"IV/RV elevated ({iv_rv_text} > {IV_RV_CROWDING_WARNING_THRESHOLD:.2f}): event premium may already be priced in (crowding risk)."
        )
    if not np.isfinite(ts_slope_0_45) or ts_slope_0_45 <= MIN_TS_SLOPE_FOR_EXPANSION:
        ts_text = f"{ts_slope_0_45:.4f}" if np.isfinite(ts_slope_0_45) else "n/a"
        expansion_gate_reasons.append(
            f"Term structure slope not sufficiently positive ({ts_text})."
        )
    rv_pct_val = _safe_float(rv_percentile_rank, np.nan)
    if not np.isfinite(rv_pct_val) or rv_pct_val < MIN_RV_PERCENTILE_FOR_EXPANSION:
        expansion_gate_reasons.append(
            f"Volatility regime percentile too low ({rv_percentile_rank if rv_percentile_rank is not None else 'n/a'})."
        )
    if np.isfinite(spread_val) and spread_val > MAX_NEAR_TERM_SPREAD_PCT_FOR_EXPANSION:
        expansion_gate_reasons.append(
            f"Near-term spread too wide for expansion ({spread_val:.1f}% > {MAX_NEAR_TERM_SPREAD_PCT_FOR_EXPANSION:.1f}%)."
        )
    if np.isfinite(near_term_liquidity_val) and near_term_liquidity_val < MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION:
        expansion_gate_reasons.append(
            f"Near-term liquidity too low for expansion ({near_term_liquidity_val:.0f} < {MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION:.0f})."
        )
    if not expansion_summary.get("available"):
        expansion_gate_reasons.append(
            f"Historical pre-earnings simulation unavailable ({expansion_summary.get('status')})."
        )
    elif int(expansion_summary.get("priceable_trades") or 0) < MIN_PRICEABLE_EXPANSION_TRADES:
        expansion_gate_reasons.append(
            f"Too few priceable pre-earnings trades ({expansion_summary.get('priceable_trades', 0)}/{MIN_PRICEABLE_EXPANSION_TRADES})."
        )
    exp_adj = _safe_float(expansion_summary.get("expected_pnl_adjusted"), np.nan)
    if np.isfinite(exp_adj) and exp_adj <= 0:
        expansion_gate_reasons.append(f"Expected adjusted pre-earnings P&L is not positive ({exp_adj:+.2f}).")
    # Cheapness component: score is highest when IV/RV is low (cheap IV = good long-vol entry).
    # Anchored at 0.80 (very cheap) → 1.60 (expensive/crowded), consistent with screener and snapshot.
    exp_signal_component = float(np.clip(1.0 - (iv_rv - 0.80) / 0.80, 0.0, 1.0)) if np.isfinite(iv_rv) else 0.0
    exp_slope_component = (
        float(np.clip((ts_slope_0_45 - MIN_TS_SLOPE_FOR_EXPANSION) / 0.010, 0.0, 1.0))
        if np.isfinite(ts_slope_0_45)
        else 0.0
    )
    exp_regime_component = float(np.clip((rv_pct_val - MIN_RV_PERCENTILE_FOR_EXPANSION) / 40.0, 0.0, 1.0)) if np.isfinite(rv_pct_val) else 0.0
    exp_liq_component = float(np.clip(
        (np.log1p(max(near_term_liquidity_val, 1.0)) - np.log1p(MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION))
        / (np.log1p(25_000.0) - np.log1p(MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION)),
        0.0,
        1.0,
    )) if np.isfinite(near_term_liquidity_val) else 0.0
    expansion_live_score = float(np.clip(
        0.32 * exp_signal_component
        + 0.24 * exp_slope_component
        + 0.22 * exp_regime_component
        + 0.22 * exp_liq_component,
        0.0,
        1.0,
    ))
    expansion_ranking_score = float(np.clip(
        0.55 * expansion_live_score + 0.45 * _safe_float(expansion_summary.get("ranking_score"), 0.0),
        0.0,
        1.0,
    ))
    expansion_candidate = len(expansion_gate_reasons) == 0

    # ── 8. Recommendation / edge interpretation from shared selector ─────────
    analysis_mode = "post_earnings_crush"
    selected_hard_gate_reasons = hard_gate_reasons
    selected_hard_no_trade = hard_no_trade
    recommendation = _legacy_recommendation_from_selector(selector_output)
    if getattr(selector_output, "best_structure", None):
        analysis_mode = "selector_first"
    elif expansion_candidate:
        analysis_mode = "iv_expansion"
    elif hard_no_trade:
        recommendation = "No Trade"
        analysis_mode = "post_earnings_crush"

    confidence_pct = float(getattr(selector_output, "confidence_pct", confidence_pct_uncapped))
    _confidence_cap_reason: Optional[str] = None
    # Entries collected here are spliced into rationale[] once it is built below.
    _deferred_rationale: List[str] = []

    confidence_capped = False
    edge_quality = str(getattr(selector_output, "expected_edge_tier", "Negative / Unclear"))

    # ── 9. Rationale strings ──────────────────────────────────────────────────
    rt_label = ""
    if earnings_release_time:
        rt_map = {
            "before market open": " (BMO)",
            "after market close": " (AMC)",
            "during market hours": " (intraday)",
        }
        rt_label = rt_map.get(earnings_release_time, f" ({earnings_release_time})")

    iv_rv_har = _safe_float(snapshot_inputs["iv_rv_har"], np.nan)
    iv_rv_har = float(iv_rv_har) if np.isfinite(iv_rv_har) else None

    # Fix 8: event/non-event decomposition warning in high-vol regimes.
    # The decomposition assumes independence between event and non-event vol:
    #   event_move = sqrt(total² − non_event²)
    # In High regimes the two components are correlated — broad market vol
    # inflates both simultaneously, understating the event component.
    _decomp_regime_warning = (
        vol_regime == "High"
        and np.isfinite(non_event_move_pct)
        and non_event_move_pct > 0
    )
    try:
        from services.calibration_service import get_calibration

        calibration_diag = get_calibration().diagnostics()
    except Exception:
        calibration_diag = None
    calibration_phase_note = _calibration_phase_note(calibration_diag)

    rationale = [
        f"Selector-backed recommendation={recommendation}; best structure={getattr(selector_output, 'best_structure', None) or 'none'}.",
        f"Selector thesis: {getattr(selector_output, 'primary_thesis', 'No shared thesis available.')}",
        *[
            f"Selector evidence: {item}"
            for item in (getattr(selector_output, 'why_this_structure', []) or [])[:2]
        ],
        (
            f"IV/RV30={iv_rv:.2f} [YZ] / IV/RV(HAR fwd)={iv_rv_har:.2f}"
            if np.isfinite(iv_rv) and iv_rv_har is not None
            else f"IV/RV30={iv_rv:.2f}" if np.isfinite(iv_rv) else "IV/RV30 unavailable from chain snapshot."
        ),
        f"Term slope(0-45)={ts_slope_0_45:.4f}" if np.isfinite(ts_slope_0_45) else "Term slope unavailable.",
        f"Days to earnings={dte}{rt_label}" if dte is not None else "Days to earnings unavailable.",
        (
            f"Shared timing score={vol_snapshot.timing_score:.2f} (snapshot-derived pre-earnings entry quality)."
            if vol_snapshot.timing_score is not None
            else "Shared timing score unavailable."
        ),
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
            f"Active mode={analysis_mode}; gate status=FAIL ({'; '.join(selected_hard_gate_reasons)})"
            if selected_hard_no_trade
            else f"Active mode={analysis_mode}; gate status=PASS."
        ),
        (
            "Pre-earnings expansion unavailable."
            if not expansion_summary.get("available")
            else (
                f"Pre-earnings calendar ({expansion_summary.get('structure')}) selects "
                f"T-{expansion_summary.get('selected_entry_offset_days')} entry / T-{EXPANSION_EXIT_OFFSET} exit; "
                f"expected IV change={_safe_float(expansion_summary.get('expected_iv_change'), 0.0) * 100.0:+.2f} pts; "
                f"expected P&L mid={_safe_float(expansion_summary.get('expected_pnl_mid'), 0.0):+.2f}, "
                f"adjusted={_safe_float(expansion_summary.get('expected_pnl_adjusted'), 0.0):+.2f} per spread."
            )
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
        f"Evidence confidence={sample_confidence:.2f} ({move_sample_size} moves, source={move_source})."
        + (" [SOFT GATE: fallback move model — confidence capped at "
           f"{FALLBACK_MODEL_CONFIDENCE_CAP_PCT:.0f}%]" if fallback_move_model_flag else "")
        + (" [SOFT GATE: thin earnings history "
           f"({move_sample_size}/{MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL} events) — "
           f"confidence capped at {LOW_EVENT_COUNT_CONFIDENCE_CAP_PCT:.0f}%]"
           if low_event_count_flag and not fallback_move_model_flag else ""),
        f"Edge quality: {edge_quality}.",
        getattr(selector_output, "model_output_note", "Model-output note unavailable."),
        f"Liquidity proxy={liquidity:.0f} (ATM OI/vol aggregate).",
        f"Data source: {data_source}.",
        *([calibration_phase_note] if calibration_phase_note else []),
        # Fix 8: regime warning appended when decomposition is unreliable.
        *(
            ["⚠ High vol regime: event/non-event volatility decomposition may be "
             "unreliable due to correlation effects. Event-implied move "
             f"({event_implied_move_pct:.2f}%) could be understated."]
            if _decomp_regime_warning else []
        ),
        (
            f"RV model: {rv_estimator} rv30={rv30:.4f}; "
            f"HAR fwd={rv_har_forecast:.4f} (Corsi 2009 on Rogers-Satchell daily series)."
            if rv_har_forecast is not None
            else f"RV model: {rv_estimator} rv30={rv30:.4f}; HAR forecast unavailable (< 30d hist; HAR needs >=100d)."
        ),
        # Deferred entries from pre-rationale cap/downgrade logic.
        *_deferred_rationale,
    ]

    # ── 10. Metrics dict ──────────────────────────────────────────────────────
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        # FIX 4: granular source provenance — options and price/RV may differ
        "data_sources": {
            "options_source":  options_source,
            "price_rv_source": "yfinance",
            "earnings_source_primary": vol_snapshot.earnings_source_primary,
            "earnings_source_confirmed": vol_snapshot.earnings_source_confirmed,
            "earnings_source_stale": vol_snapshot.earnings_source_stale,
            "release_timing_source": vol_snapshot.release_timing_source,
        },
        "earnings_source_confidence": (
            float(vol_snapshot.earnings_source_confidence)
            if vol_snapshot.earnings_source_confidence is not None
            else None
        ),
        "price_data_stale": _price_stale,
        "price_data_age_days": _price_age_days,
        "data_quality": vol_snapshot.data_quality,
        "data_quality_score": float(vol_snapshot.data_quality_score),
        "price_staleness_minutes": vol_snapshot.price_staleness_minutes,
        "chain_staleness_minutes": vol_snapshot.chain_staleness_minutes,
        "earnings_source_notes": list(vol_snapshot.earnings_source_notes),
        "timing_score": float(vol_snapshot.timing_score) if vol_snapshot.timing_score is not None else None,
        "cheapness_score": float(vol_snapshot.cheapness_score) if vol_snapshot.cheapness_score is not None else None,
        "event_risk_score": float(vol_snapshot.event_risk_score) if vol_snapshot.event_risk_score is not None else None,
        "execution_score": float(vol_snapshot.execution_score) if vol_snapshot.execution_score is not None else None,
        # Fix 1: confidence cap applied when price bars are stale.
        "stale_data_confidence_cap_pct": STALE_DATA_CONFIDENCE_CAP_PCT if _price_stale else None,
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
        "setup_score_source": "shared_screener_ranking_score",
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
        "analysis_mode": analysis_mode,
        "legacy_orchestration_mode": "shared_snapshot_selector_adapter",
        "hard_no_trade": selected_hard_no_trade,
        "hard_gate_reasons": selected_hard_gate_reasons,
        "crush_hard_no_trade": hard_no_trade,
        "crush_hard_gate_reasons": hard_gate_reasons,
        "expansion_candidate": expansion_candidate,
        "expansion_gate_reasons": expansion_gate_reasons,
        "expansion_live_score": expansion_live_score,
        "expansion_ranking_score": expansion_ranking_score,
        "iv_rv_crowding_warning_threshold": IV_RV_CROWDING_WARNING_THRESHOLD,
        "min_term_structure_slope_for_expansion": MIN_TS_SLOPE_FOR_EXPANSION,
        "min_rv_percentile_for_expansion": MIN_RV_PERCENTILE_FOR_EXPANSION,
        "max_near_term_spread_pct_for_expansion": MAX_NEAR_TERM_SPREAD_PCT_FOR_EXPANSION,
        "min_near_term_liquidity_proxy_for_expansion": MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION,
        "expansion_structure": expansion_summary.get("structure"),
        "expansion_selected_entry_offset_days": expansion_summary.get("selected_entry_offset_days"),
        "expansion_exit_offset_days": expansion_summary.get("exit_offset_days"),
        "expansion_historical_sample_size": expansion_summary.get("historical_sample_size"),
        "expansion_priceable_trades": expansion_summary.get("priceable_trades"),
        "expansion_missing_exit_quotes": expansion_summary.get("missing_exit_quotes"),
        "expansion_excluded_negative_debit": expansion_summary.get("excluded_negative_debit"),
        "expansion_excluded_structural_invalid": expansion_summary.get("excluded_structural_invalid"),
        "expansion_expected_iv_change": expansion_summary.get("expected_iv_change"),
        "expansion_expected_pnl_mid": expansion_summary.get("expected_pnl_mid"),
        "expansion_expected_pnl_adjusted": expansion_summary.get("expected_pnl_adjusted"),
        "expansion_expected_return_mid_pct": expansion_summary.get("expected_return_mid_pct"),
        "expansion_expected_return_adjusted_pct": expansion_summary.get("expected_return_adjusted_pct"),
        "expansion_history_status": expansion_summary.get("status"),
        "confidence_capped": confidence_capped,
        "confidence_cap_pct": HARD_NO_TRADE_CONFIDENCE_CAP_PCT,
        # Explicit cap values and human-readable reason — consumers must not re-derive from flags.
        "confidence_cap_reason": _confidence_cap_reason,
        "fallback_model_confidence_cap_pct": FALLBACK_MODEL_CONFIDENCE_CAP_PCT if fallback_move_model_flag else None,
        "low_event_count_confidence_cap_pct": (
            LOW_EVENT_COUNT_CONFIDENCE_CAP_PCT
            if (low_event_count_flag and not fallback_move_model_flag) else None
        ),
        # Fix 3: soft gate disclosure flags — present in output so the UI and
        # downstream consumers can show explicit disclosure without re-deriving.
        "low_event_count_flag": low_event_count_flag,
        "fallback_move_model_flag": fallback_move_model_flag,
        "tx_cost_estimate_pct": tx_cost_pct,
        "raw_gross_edge_pct": float(raw_gross_edge_pct) if np.isfinite(raw_gross_edge_pct) else None,
        "confidence_adjusted_gross_edge_pct": (
            float(confidence_adjusted_gross_edge_pct)
            if np.isfinite(confidence_adjusted_gross_edge_pct) else None
        ),
        "uncertainty_penalty_pct": float(uncertainty_penalty_pct),
        "expected_gross_edge_pct": float(expected_gross_edge_pct) if np.isfinite(expected_gross_edge_pct) else None,
        "expected_net_edge_pct": float(expected_net_edge_pct) if np.isfinite(expected_net_edge_pct) else None,
        "selector_expected_edge_tier": getattr(selector_output, "expected_edge_tier", None),
        "selector_expected_return_signal": getattr(selector_output, "expected_return_signal", None),
        "model_output_note": getattr(selector_output, "model_output_note", None),
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
        "rv_har_is_fallback": vol_snapshot.null_reasons.get("rv_har_estimator", "").startswith("rs_trailing_mean_fallback"),
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
        # ── Position sizing ───────────────────────────────────────────────────
        # Kelly sizing removed — requires calibrated edge (see comment near _kelly_sizing).
        "position_sizing_note": (
            "Position sizing guidance not available yet — requires calibrated win rate "
            "and empirical edge estimate."
        ),
        # ── Structure-specific payoff (matches the recommended structure) ────
        "structure_payoff": structure_payoff,
        # ── Calendar spread P&L diagram (legacy panel) ─────────────────────
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
        "confidence_score_source": "shared_selector_confidence_pct",
        "calibration_phase": calibration_diag.get("phase") if calibration_diag else None,
        "calibration_n_observations": calibration_diag.get("n_observations") if calibration_diag else None,
        "calibration_min_for_observational": calibration_diag.get("min_for_observational") if calibration_diag else None,
        "calibration_min_for_fit": calibration_diag.get("min_for_fit") if calibration_diag else None,
        "calibration_phase_note": calibration_phase_note,
        # Fix 8: decomposition reliability flag — surfaced so the UI can show
        # the warning without re-deriving vol_regime on the client side.
        "decomp_regime_warning": _decomp_regime_warning,
        # ── Move-risk advisory (soft signal — not a hard gate) ────────────────
        "move_risk_level":  move_risk_level,
        "move_risk_ratio":  round(move_risk_ratio, 3) if move_risk_ratio is not None else None,
        "move_risk_sample_size": move_sample_size,
    }

    selector_output_dict = selector_output.to_dict()
    structure_scorecard_dicts = [card.to_dict() for card in structure_scorecards]
    vol_snapshot_dict = vol_snapshot.to_dict()
    edge_snapshot = EdgeSnapshot(
        symbol=clean_symbol,
        recommendation=recommendation,
        confidence_pct=confidence_pct,
        setup_score=setup_score,
        metrics=metrics,
        rationale=rationale,
        selector_output=selector_output_dict,
        structure_scorecards=structure_scorecard_dicts,
        vol_snapshot=vol_snapshot_dict,
    )
    if record_to_ledger:
        try:
            from services.recommendation_ledger import record_recommendation

            recommendation_id = record_recommendation(
                edge_snapshot,
                metadata={"source": "edge_engine.analyze_single_ticker"},
            )
            edge_snapshot.metrics["recommendation_id"] = recommendation_id
        except Exception as exc:
            logger.warning("Recommendation ledger write failed for %s: %s", clean_symbol, exc)
    return edge_snapshot
