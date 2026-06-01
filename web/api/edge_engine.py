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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from services.dividend_yields import get_dividend_yield
from services.iv_term_structure import bounded_interp
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


# ─── Constants ────────────────────────────────────────────────────────────────
# Extracted to web/api/edge_constants.py (Phase 3.1). Re-exported here so existing
# `from web.api.edge_engine import <CONST>` paths and all internal references work
# unchanged. Values/comments live in edge_constants.py; this is a pure relocation.
from web.api.edge_constants import (  # noqa: E402,F401
    MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL,
    HARD_NO_TRADE_CONFIDENCE_CAP_PCT,
    TS_SLOPE_TARGET,
    TS_SLOPE_BAND,
    MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE,
    MIN_SHORT_LEG_DTE,
    MIN_NEAR_BACK_IV_RATIO_FOR_EVENT,
    MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_TRADE,
    MOVE_UNCERTAINTY_Z_SCORE,
    IV_RV_CROWDING_WARNING_THRESHOLD,
    MIN_TS_SLOPE_FOR_EXPANSION,
    MIN_RV_PERCENTILE_FOR_EXPANSION,
    MAX_NEAR_TERM_SPREAD_PCT_FOR_EXPANSION,
    MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION,
    MIN_PRICEABLE_EXPANSION_TRADES,
    EXPANSION_ENTRY_OFFSETS,
    EXPANSION_EXIT_OFFSET,
    EXPANSION_BACK_EXPIRY_GAP_DAYS,
    STALE_DATA_THRESHOLD_DAYS,
    STALE_DATA_CONFIDENCE_CAP_PCT,
    FALLBACK_MODEL_CONFIDENCE_CAP_PCT,
    LOW_EVENT_COUNT_CONFIDENCE_CAP_PCT,
    _HEURISTIC_THRESHOLDS,
)



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
    # PR-AE C3: live-forward observation fields. ``analyze_single_ticker``
    # populates both before ledger write so the row lands with
    # ``sample_provenance = forward_post_freeze`` and an empty-but-stable
    # ``candidate_shadow_outcome`` block carrying status
    # "awaiting_exit_resolver". The PR-AE C4 resolver picks up these
    # rows after the earnings event and fills the candidate outcome.
    #
    # build_record_from_analysis (services/recommendation_ledger.py)
    # already reads both attributes via the _get helper, so the
    # ledger-side wiring is automatic — no additional code change is
    # required for these fields to flow into the recommendations row.
    #
    # NB: historical-replay records do NOT flow through EdgeSnapshot.
    # _simulate_pre_earnings_calendar_trade builds a plain dict with
    # sample_provenance=HISTORICAL_REPLAY directly. The boundary test
    # TestForwardProvenanceAssignmentBoundary enforces that fact via
    # source-grep.
    sample_provenance: Optional[str] = None
    candidate_shadow_outcome: Dict[str, Any] = field(default_factory=dict)


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _utc_today_date():
    return datetime.now(timezone.utc).date()


# PR-AE C2: extracted to services/candidate_shadow_outcome.py so the
# upcoming resolver (services/candidate_exit_resolver.py, PR-AE C4)
# does not need to import upward from web/. The private name
# _safe_float is preserved here as an alias so every existing call
# site below works unchanged.
from services.candidate_shadow_outcome import safe_float as _safe_float  # noqa: E402


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


# PR-AE C2: extracted to services/candidate_shadow_outcome.py. Aliased
# back to the private names here so every existing call site (legacy
# historical simulator in _simulate_pre_earnings_calendar_trade,
# _select_pre_earnings_calendar_contracts, etc.) continues to use the
# same identifiers byte-for-byte.
from services.candidate_shadow_outcome import (  # noqa: E402
    best_contract_row as _best_contract_row,
    prepare_feature_chain as _prepare_feature_chain,
)


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


# PR-AE C2: extracted to services/candidate_shadow_outcome.py. Aliased
# under the private name here so legacy historical-replay call sites
# (lines 709-724 in the previous simulator body, plus 883-884 in
# _simulate_pre_earnings_calendar_trade's exit-chain lookup) work
# unchanged.
from services.candidate_shadow_outcome import (  # noqa: E402
    lookup_exact_contract_row as _lookup_exact_contract_row,
)


# ── PR-AD commit 1: candidate shadow outcome simulator ────────────────────
#
# PR-AC commit 2 added shadow-mode dual-picker logging that records WHICH
# contracts the legacy and candidate pickers would have selected. That
# alone is not enough to answer "did the candidate make more money than
# legacy?" — for that we also need the candidate's exit pricing. This
# simulator closes the gap: given the entry chain, exit chain, and
# dual_picker block, it resolves the candidate's contracts through
# entry+exit pricing and computes candidate PnL alongside legacy.
#
# CRITICAL: outcomes produced here on the historical-backtest events are
# IN-SAMPLE for the +14d rule (the rule was discovered on this exact
# data in PR-AB). Per docs/CALENDAR_PICKER_PROMOTION_2026-05-27.md, the
# candidate_realized_return_pct from these events must NOT be used as
# out-of-sample evidence. Promotion criteria reference forward
# (post-PR-AD-merge) events only.
# Provenance taxonomy, shape constants, and pure helpers moved to
# services/candidate_shadow_provenance.py in PR-AD commit 4 (Codex
# review): the storage / service layer should not depend on the
# web/API layer. Re-imported here for module-local use AND re-exported
# as part of edge_engine's public surface so existing test imports
# (and any downstream callers) keep working without changes.
from services.candidate_shadow_provenance import (  # noqa: E402
    _CANDIDATE_SHADOW_LABELS,
    _CANDIDATE_SHADOW_OUTCOME_FIELDS,
    _CANDIDATE_SHADOW_SCENARIO_BLOCK_FIELDS,
    PROMOTION_ELIGIBLE_PROVENANCES,
    SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
    SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED,
    SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
    SAMPLE_PROVENANCE_UNKNOWN,
    VALID_SAMPLE_PROVENANCES,
    _empty_candidate_shadow_outcome,
    _tag_live_forward_observation,
    candidate_return_for_gate,
    is_promotion_eligible,
    is_promotion_eligible_for_scenario,
    normalize_sample_provenance,
)


# PR-AE C2: simulator extracted to services/candidate_shadow_outcome.py
# (Codex design-review required change). The full body — including the
# status enum, the calendar P&L convention, and the IV-change handling
# — lives there now. This alias preserves the private name so existing
# call sites in this module (in particular line ~877 inside
# _simulate_pre_earnings_calendar_trade) keep their existing identifier
# byte-for-byte.
#
# Upcoming PR-AE C4 (services/candidate_exit_resolver.py) imports the
# PUBLIC name `simulate_candidate_shadow_outcome` from the new module.
from services.candidate_shadow_outcome import (  # noqa: E402
    simulate_candidate_shadow_outcome as _simulate_candidate_shadow_outcome,
)


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
        # PR-AD commit 1b (Codex review): every record produced by the
        # historical-backtest path is tagged in-sample regardless of
        # when the script ran. The +14d rule was discovered on this
        # exact code path; re-running it tomorrow does not produce
        # out-of-sample evidence. Promotion criteria filter on
        # PROMOTION_ELIGIBLE_PROVENANCES, which does NOT include this
        # value.
        "sample_provenance": SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
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

    # PR-AD commit 1: placeholder candidate shadow outcome. The real
    # outcome can only be computed once the exit chain is loaded; for
    # every return path that fails before that point, this placeholder
    # preserves a uniform record shape. Status overwritten below the
    # moment the exit chain succeeds.
    base["candidate_shadow_outcome"] = _empty_candidate_shadow_outcome(
        "skipped:trade_failed_before_exit_chain"
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

    # PR-AD commit 1: with both chains in hand, resolve the candidate
    # picker's PnL too. Independent of whether legacy succeeds below —
    # we want candidate outcomes even on events where legacy fails on
    # its exit quote (the candidate may have selected different
    # contracts that ARE quoted at exit). Pure shadow logging.
    base["candidate_shadow_outcome"] = _simulate_candidate_shadow_outcome(
        entry_chain=entry_chain,
        exit_chain=exit_chain,
        dual_picker=base.get("dual_picker"),
    )

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


def _build_experimental_contract_selection(
    *,
    best_structure: Optional[str],
    chain_df: Optional[pd.DataFrame],
    earnings_event_date: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Live-API shadow surface for candidate calendar contracts.

    PR-AC commit 3. Strict shadow-mode: never replaces or modifies any
    existing recommendation field; returned as a separate, clearly-labeled
    `experimental_contract_selection` block.

    Coverage matrix:
      best_structure       | behavior
      ---------------------|------------------------------------------
      call_calendar        | runs _dual_picker_calendar_selection on
                           | the live calls chain; surfaces both
                           | legacy and candidate selections.
      put_calendar         | returns put_side_not_yet_supported
                           | placeholder. The historical backtest path
                           | only queries the calls chain, so we have
                           | NO put-side forward-validation evidence
                           | yet. Surfacing call-side contracts under a
                           | put_calendar response would falsely imply
                           | we know what we're doing on the put side.
                           | This is a deliberate Codex review hard
                           | requirement.
      anything else        | returns None (no calendar contracts to
                           | recommend).
    """
    if best_structure not in ("call_calendar", "put_calendar"):
        return None

    base = {
        "structure": best_structure,
        "labels": {
            "experimental": True,
            "shadow_mode": True,
            "not_execution_guidance": True,
            "out_of_sample_validated": False,
        },
        "note": (
            "Experimental candidate contract selection. The candidate "
            "picker uses an in-sample +14d min-front-DTE rule from "
            "PR-AB; not yet out-of-sample validated. Treat as research "
            "input, not as an order recommendation."
        ),
    }

    if best_structure == "put_calendar":
        # CODEX HARD REQUIREMENT: must not silently route call-side
        # data here. Put-side dual-picker requires querying the puts
        # chain through the historical backtest path, which is a
        # separate correctness change.
        return {
            **base,
            "status": "put_side_not_yet_supported",
            "reason": (
                "The historical backtest only queries the calls chain, "
                "so no put-side forward-validation evidence exists yet. "
                "Surfacing call-side contracts here would falsely imply "
                "put-side validation. Wiring puts through the historical "
                "simulation is a separate change."
            ),
            "candidate_contracts": None,
        }

    # call_calendar branch — invoke the dual picker on the live chain.
    if chain_df is None or earnings_event_date is None:
        return {
            **base,
            "status": "skipped:missing_inputs",
            "reason": "Live chain or earnings event date unavailable.",
            "candidate_contracts": None,
        }

    # The live MDApp chain uses `expiration_date`; the picker module
    # expects `expiry`. Normalize here per the picker docstring's
    # "callers must normalize column names" requirement.
    normalized = chain_df
    if "expiration_date" in chain_df.columns and "expiry" not in chain_df.columns:
        normalized = chain_df.copy()
        normalized["expiry"] = normalized["expiration_date"]

    try:
        dp = _dual_picker_calendar_selection(
            normalized,
            event_date=pd.Timestamp(earnings_event_date),
            side="call",
        )
    except Exception as exc:
        logger.debug("experimental_contract_selection failed: %s", exc)
        return {
            **base,
            "status": f"error:{type(exc).__name__}",
            "candidate_contracts": None,
        }

    return {
        **base,
        "status": dp.get("shadow_status", "unknown"),
        "candidate_contracts": dp,
    }


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

    # PR-AD commit 2 — provenance + candidate-outcome accounting.
    # Codex hard rules:
    #   - Show all records in diagnostics, including unknown / replay /
    #     skipped / malformed.
    #   - Compute promotion stats ONLY on records that pass
    #     is_promotion_eligible(record).
    #   - Never mix historical replay and forward evidence into one
    #     "candidate performance" number.
    provenance_counts: Dict[str, int] = {}
    candidate_outcome_status_counts: Dict[str, int] = {}

    # Per-event candidate PnL by provenance bucket. The outer key is
    # the sample_provenance string; the inner dict maps event_key to
    # the list of mid_realized_return_pct values seen at that event
    # (one per ok-status candidate outcome across entry offsets).
    candidate_pnl_by_provenance_and_event: Dict[str, Dict[Any, List[float]]] = {}

    # Same shape, restricted to records that pass is_promotion_eligible.
    # Used for the promotion_eligible_candidate_stats block. Kept
    # separate so a downstream typo in the aggregator can't accidentally
    # widen the filter.
    promotion_eligible_event_pnls: Dict[Any, List[float]] = {}

    # PR #66: scenario-aware visibility infrastructure. These counters
    # answer the operational question "what fraction of
    # promotion-eligible events have execution-cost-aware evidence?"
    # — the data needed to justify a future gate refactor to scenario-
    # aware promotion criteria.
    #
    # The default gate scenario is the same baseline structure_scorecard
    # uses for historical priors (cross_25), so the visibility metric
    # answers a uniform question across the whole pipeline.
    #
    # CRITICAL: this aggregator NEVER substitutes mid for a missing
    # scenario value. `allow_mid_fallback=False` is hardcoded below.
    # PR #66 ships visibility, not policy — the `fallback_events`
    # counter exists so a future PR that flips a caller to
    # `allow_mid_fallback=True` automatically gets the substitution
    # surface in the daily diagnostic output. Until then, the count
    # stays at 0 by construction.
    _GATE_SCENARIO_FOR_VISIBILITY = "cross_25"
    # Per-event scenario observations. We collect one bool per
    # (eligible offset, event) — True iff cross_25 was priced for
    # that offset — then reduce to one classification per event
    # after the trade loop. Doing the classification per trade
    # (instead of per event) caused a Codex P1 on PR #66 first
    # round: a single earnings event evaluated at N entry offsets
    # could land in BOTH the priced and missing sets simultaneously
    # if any offset had bid/ask coverage and any other didn't —
    # breaking the documented invariant
    # `scenario_priced_events + missing_scenario_events == n_events`.
    # The fix moves classification to event-level reduction with
    # the "any priced offset → event priced" rule (see comment
    # below the loop).
    per_event_scenario_priced_flags: Dict[Any, List[bool]] = {}
    scenario_missing_reason_counts: Dict[str, int] = {}

    # Event-level (deduplicated on (symbol, event_date))
    unique_events: set = set()
    unique_events_with_legacy: set = set()
    unique_events_with_candidate: set = set()
    unique_events_diverged: set = set()
    unique_events_picker_evaluated: set = set()  # at least one offset had shadow_status == "ok"

    # Event-level outcome buckets (Codex follow-up to commit 2b). These
    # are derived after the trade loop because a single event seen at
    # multiple offsets can have inconsistent per-trade outcomes — we
    # want a stable event-level classification, not "max over offsets".
    # Rule: an event is in `unique_events_both_succeeded` iff EVERY
    # ok-status trade for that event had both selections. Conservative.
    per_event_trade_outcomes: Dict[Any, List[str]] = {}

    diverged_sample: List[Dict[str, Any]] = []

    for trade in trades:
        dp = trade.get("dual_picker") or {}
        status = dp.get("shadow_status")
        shadow_status_counts[status or "missing"] = shadow_status_counts.get(status or "missing", 0) + 1

        # Codex P2a: dedupe key includes option_type so call_calendar
        # and put_calendar outcomes for the same earnings event do NOT
        # collapse into a single bucket once put-side support lands.
        # Today historical replay is calls-only ("C") so this has no
        # effect on current data — but future-proofing now means the
        # put-side commit doesn't need to touch the aggregator at all.
        side_key = trade.get("option_type") or (
            ((trade.get("dual_picker") or {}).get("candidate_selection") or {}).get("side")
            or "unknown"
        )
        # Normalize "call"/"put" → "C"/"P" so both sources line up
        if side_key == "call":
            side_key = "C"
        elif side_key == "put":
            side_key = "P"
        event_key = (trade.get("symbol"), trade.get("event_date"), side_key)
        if (trade.get("symbol") is not None
                and trade.get("event_date") is not None):
            unique_events.add(event_key)

        # PR-AD commit 2 accounting — observability first, gating second.
        prov = trade.get("sample_provenance") or "missing"
        provenance_counts[prov] = provenance_counts.get(prov, 0) + 1

        outcome = trade.get("candidate_shadow_outcome") or {}
        outcome_status = outcome.get("status") or "missing"
        candidate_outcome_status_counts[outcome_status] = (
            candidate_outcome_status_counts.get(outcome_status, 0) + 1
        )

        # Collect candidate PnL by provenance for diagnostics. The
        # in_sample_diagnostic block reads from this; the promotion
        # block uses is_promotion_eligible() instead so the strict
        # checks apply uniformly.
        candidate_pnl = outcome.get("mid_realized_return_pct")
        if (
            outcome_status == "ok"
            and isinstance(candidate_pnl, (int, float))
            and not isinstance(candidate_pnl, bool)
            and np.isfinite(candidate_pnl)
            and trade.get("symbol") is not None
            and trade.get("event_date") is not None
        ):
            candidate_pnl_by_provenance_and_event.setdefault(prov, {}).setdefault(
                event_key, []
            ).append(float(candidate_pnl))

        # Strict gate for promotion-eligible bucket. Routes through the
        # single source of truth so any future change to the eligibility
        # definition automatically tightens this aggregator too.
        if (
            is_promotion_eligible(trade)
            and trade.get("symbol") is not None
            and trade.get("event_date") is not None
        ):
            promotion_eligible_event_pnls.setdefault(event_key, []).append(
                float(outcome["mid_realized_return_pct"])
            )

            # PR #66 visibility: among the promotion-eligible records,
            # observe whether the gate-baseline scenario (cross_25
            # today) was priced. Fail-closed — `allow_mid_fallback=False`
            # is the only mode this PR uses in any production call
            # site. A missing cross_25 contributes a False to this
            # event's flag list; per-trade dedup is NOT done here.
            # The flags are reduced to one classification per event
            # AFTER the loop (see "event-level scenario classification"
            # block below). Codex P1 reproducer (multi-offset event
            # with mixed coverage) is pinned by a test in
            # tests/unit/test_web/test_calendar_dual_picker.py.
            _scenario_value, _scenario_meta = candidate_return_for_gate(
                outcome,
                _GATE_SCENARIO_FOR_VISIBILITY,
                allow_mid_fallback=False,
            )
            per_event_scenario_priced_flags.setdefault(event_key, []).append(
                _scenario_value is not None
            )
            if _scenario_value is None:
                # Reason breakdown is per-trade (we want to see, e.g.,
                # "5 offsets failed because of NaN bid/ask" vs "2
                # offsets failed because the candidate selection was
                # skipped"). Event-level reduction below sums these
                # into the n_events denominator, but the breakdown
                # keeps trade-level granularity because that's the
                # actionable diagnostic.
                _reason = _scenario_meta.get("missing_reason") or "unknown"
                scenario_missing_reason_counts[_reason] = (
                    scenario_missing_reason_counts.get(_reason, 0) + 1
                )
            # `fallback_events` is intentionally NOT computed here.
            # By construction every is_promotion_eligible(record)
            # record carries a finite mid (check #5 of the existing
            # gate), so `missing_scenario_events` would be identical
            # to "events that would have been rescued by mid fallback."
            # That makes a second `allow_mid_fallback=True` call
            # redundant — and crucially, including such a call would
            # violate the PR #66 invariant "no production call site
            # passes allow_mid_fallback=True." A future gate-flip PR
            # will bump `fallback_events` from its own code path
            # (typically by recording fallback_used decisions through
            # its own counter) at which point the field becomes
            # non-zero through actual policy use, not aggregator
            # speculation.

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
            per_event_trade_outcomes.setdefault(event_key, []).append("both")
        elif legacy_present:
            n_legacy_only += 1
            per_event_trade_outcomes.setdefault(event_key, []).append("legacy_only")
        elif candidate_present:
            n_candidate_only += 1
            per_event_trade_outcomes.setdefault(event_key, []).append("candidate_only")
        else:
            n_neither += 1
            per_event_trade_outcomes.setdefault(event_key, []).append("neither")
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

    # PR-AD commit 2: compute event-level candidate PnL summary stats.
    # Reduces to per-event mean PnL first (so multiple eligible offsets
    # of the same event don't get extra weight), then aggregates across
    # events. Returns None on empty input rather than NaN or 0.0 — a
    # missing stat must be visibly missing, never zero-coded.
    def _summarize_event_level_pnls(event_pnls: Dict[Any, List[float]]) -> Dict[str, Any]:
        if not event_pnls:
            return {
                "n_events": 0,
                "candidate_mid_realized_return_pct_mean": None,
                "candidate_mid_realized_return_pct_median": None,
                "candidate_mid_realized_return_pct_win_rate": None,
            }
        per_event_mean = [
            sum(pnls) / len(pnls) for pnls in event_pnls.values() if pnls
        ]
        n = len(per_event_mean)
        if n == 0:
            return {
                "n_events": 0,
                "candidate_mid_realized_return_pct_mean": None,
                "candidate_mid_realized_return_pct_median": None,
                "candidate_mid_realized_return_pct_win_rate": None,
            }
        per_event_mean_sorted = sorted(per_event_mean)
        if n % 2 == 1:
            median = per_event_mean_sorted[n // 2]
        else:
            median = 0.5 * (per_event_mean_sorted[n // 2 - 1] + per_event_mean_sorted[n // 2])
        wins = sum(1 for v in per_event_mean if v > 0)
        return {
            "n_events": int(n),
            "candidate_mid_realized_return_pct_mean": float(sum(per_event_mean) / n),
            "candidate_mid_realized_return_pct_median": float(median),
            "candidate_mid_realized_return_pct_win_rate": float(wins / n),
        }

    # In-sample diagnostic stats — historical replay subset ONLY.
    # Codex hard rule: never mix replay and forward evidence into one
    # number. We compute stats from the HISTORICAL_REPLAY bucket and
    # label the block exhaustively so a downstream reader cannot
    # mistake these for promotion evidence.
    in_sample_summary = _summarize_event_level_pnls(
        candidate_pnl_by_provenance_and_event.get(
            SAMPLE_PROVENANCE_HISTORICAL_REPLAY, {}
        )
    )

    # Promotion-eligible stats — gated by is_promotion_eligible(record).
    # On the historical replay code path this is empty by construction
    # (HISTORICAL_REPLAY is not in PROMOTION_ELIGIBLE_PROVENANCES). It
    # becomes non-empty only when forward observations accumulate via
    # the live API path.
    promotion_eligible_summary = _summarize_event_level_pnls(promotion_eligible_event_pnls)

    # PR #66 event-level scenario classification.
    #
    # Reduces per-event flag lists to one classification each. Rule:
    # an event is "priced" if AT LEAST ONE of its eligible offsets
    # had a finite cross_25; "missing" only if NO offset did. This
    # matches the existing `promotion_eligible_event_pnls`
    # convention (which averages over offsets so any single eligible
    # offset gives the event a representative number), and keeps the
    # invariant `scenario_priced_events + missing_scenario_events
    # == n_events` true on every input shape including the
    # multi-offset mixed-coverage case Codex flagged on PR #66
    # first round.
    #
    # Stricter alternatives considered + rejected:
    #   - "ALL offsets must price" — would understate execution
    #     evidence for events where the chain has missing bid/ask
    #     on some offsets but the simulator ran fine on others.
    #   - "MAJORITY of offsets must price" — arbitrary threshold,
    #     unstable under offset count changes.
    # The "any priced" rule is the minimum useful coverage and is
    # explicit in the test pinning this invariant.
    scenario_priced_event_keys: set = set()
    missing_scenario_event_keys: set = set()
    for ev_key, flags in per_event_scenario_priced_flags.items():
        if any(flags):
            scenario_priced_event_keys.add(ev_key)
        else:
            missing_scenario_event_keys.add(ev_key)

    # Event-level outcome buckets. Conservative classification: an event
    # is in `both_succeeded` ONLY when every ok-status trade for that
    # event had both selections. If a single offset failed on either
    # side, the event drops to the partial-success or mixed bucket.
    unique_events_both_succeeded: set = set()
    unique_events_legacy_only: set = set()
    unique_events_candidate_only: set = set()
    unique_events_neither_succeeded: set = set()
    unique_events_mixed_outcomes: set = set()
    for event_key, outcomes in per_event_trade_outcomes.items():
        unique = set(outcomes)
        if unique == {"both"}:
            unique_events_both_succeeded.add(event_key)
        elif unique == {"legacy_only"}:
            unique_events_legacy_only.add(event_key)
        elif unique == {"candidate_only"}:
            unique_events_candidate_only.add(event_key)
        elif unique == {"neither"}:
            unique_events_neither_succeeded.add(event_key)
        else:
            # Mixed across offsets (e.g. some offsets had both, others
            # only legacy). Tallied separately so promotion criteria
            # don't accidentally fold mixed events into the cleaner
            # buckets.
            unique_events_mixed_outcomes.add(event_key)

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
            "entry offsets produces multiple trade records. Promotion-"
            "criterion thresholds must reference the NARROWEST event-"
            "level fields: `event_counts.unique_events_diverged` (the "
            "events where the rule actually changed contract choice) "
            "and `event_counts.unique_events_picker_evaluated` (the "
            "events where both pickers actually ran). The broader "
            "`unique_events_observed` includes data-availability "
            "failures and should not be used as a denominator."
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
            # Outcome buckets (event-level, conservative per-event
            # classification — see _per_event_trade_outcomes logic).
            "unique_events_both_succeeded": len(unique_events_both_succeeded),
            "unique_events_legacy_only": len(unique_events_legacy_only),
            "unique_events_candidate_only": len(unique_events_candidate_only),
            "unique_events_neither_succeeded": len(unique_events_neither_succeeded),
            "unique_events_mixed_outcomes": len(unique_events_mixed_outcomes),
        },
        # Data-availability accounting (entry chain missing, helper
        # exceptions, etc.) — separated so it cannot be confused with
        # picker outcomes.
        "shadow_status_counts": {
            k: int(v) for k, v in sorted(shadow_status_counts.items())
        },
        # PR-AD commit 2 — provenance + candidate-outcome accounting.
        # Counts every record regardless of provenance or outcome
        # status (Codex hard rule: "show all records in diagnostics").
        # Empty buckets are still present with count 0 if seen.
        "provenance_counts": {
            k: int(v) for k, v in sorted(provenance_counts.items())
        },
        "candidate_outcome_status_counts": {
            k: int(v) for k, v in sorted(candidate_outcome_status_counts.items())
        },
        # In-sample diagnostic candidate stats. Computed from the
        # HISTORICAL_REPLAY provenance bucket and labeled explicitly
        # as in-sample so a downstream reader CANNOT mistake this for
        # validation evidence. Exists for sanity-checking simulator
        # math (e.g., the in-sample mean should roughly match PR-AB
        # reported numbers); it is NOT promotion evidence.
        "in_sample_diagnostic_candidate_stats": {
            "note": (
                "IN-SAMPLE diagnostic only. The +14d rule was discovered "
                "on this exact dataset (PR-AB). Use this block to verify "
                "the candidate shadow simulator is mathematically correct "
                "— never as out-of-sample evidence or promotion criterion "
                "input."
            ),
            "sample_provenance": SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
            **in_sample_summary,
        },
        # Promotion-eligible candidate stats. Gated by
        # is_promotion_eligible(record) — the SINGLE source of truth
        # for promotion-eligibility. Empty (n_events=0) on any historical
        # replay aggregation by construction; populated only once
        # forward observations accumulate via the live API path.
        # Codex hard rule: this is the ONLY block promotion criteria
        # may reference.
        "promotion_eligible_candidate_stats": {
            "note": (
                "Computed ONLY from records that pass "
                "is_promotion_eligible(record). Required properties: "
                "sample_provenance in PROMOTION_ELIGIBLE_PROVENANCES "
                "(today only forward_post_freeze), candidate "
                "outcome.status == 'ok', finite numeric "
                "mid_realized_return_pct, and intact research_mid / "
                "shadow_only / not_execution_grade labels.\n\n"
                "AGGREGATION CONVENTION (Codex review P3): when a "
                "single earnings event is evaluated at N entry "
                "offsets, the N per-offset PnLs are first reduced to a "
                "single per-event mean, and only then is the "
                "cross-event statistic computed. This prevents "
                "entry-offset replication from inflating denominator "
                "weights. As a side effect, comparing these stats to "
                "live forward observations (where each event has only "
                "ONE entry) is not strictly apples-to-apples — the "
                "live forward path produces one PnL per event, no "
                "averaging step. Document this when reporting."
            ),
            **promotion_eligible_summary,
        },
        # PR #66 visibility infrastructure. ADDITIVE — does NOT replace
        # `promotion_eligible_candidate_stats` above. Answers the
        # operational question "what fraction of promotion-eligible
        # events have execution-cost-aware evidence (cross_25 priced)?"
        # — the data needed to justify a future scenario-aware gate
        # refactor.
        #
        # CRITICAL INVARIANTS pinned by tests (see PR #66):
        #   * `n_events` == promotion_eligible_candidate_stats.n_events
        #     (the existing mid-based eligibility check is the
        #     denominator).
        #   * `scenario_priced_events + missing_scenario_events
        #     == n_events`.
        #   * Missing cross_25 values are NEVER counted as priced via
        #     mid — `allow_mid_fallback=False` is hardcoded at this
        #     call site.
        #   * `fallback_events` stays 0 in PR #66 (no production
        #     caller enables the flag). A future gate-flip PR that
        #     opts into `allow_mid_fallback=True` will bump this from
        #     its own code path, making the substitution visible.
        "promotion_eligible_execution_scenario_stats": {
            "note": (
                "Visibility-only block (PR #66). Reports how many "
                "promotion-eligible events have execution-cost-aware "
                "evidence (cross_25 priced) versus mid-only. Fail-"
                "closed: missing cross_25 reduces scenario_priced_events "
                "and is NEVER rescued by mid here. The current "
                "promotion-eligibility gate still routes through "
                "is_promotion_eligible(record), which is mid-based; "
                "this block exists so a future scenario-aware gate "
                "refactor can be justified by data showing what "
                "fraction of forward observations would qualify."
            ),
            "scenario": _GATE_SCENARIO_FOR_VISIBILITY,
            "n_events": int(len(promotion_eligible_event_pnls)),
            "scenario_priced_events": int(len(scenario_priced_event_keys)),
            "missing_scenario_events": int(len(missing_scenario_event_keys)),
            # Stays 0 in PR #66 by construction. A future PR that
            # flips a real gate caller to `allow_mid_fallback=True`
            # will increment this from that caller's own counter.
            "fallback_events": 0,
            # Per-reason breakdown of why cross_25 was missing —
            # operationally most useful diagnostic for whether the
            # gap is "outcome.status not ok" (no candidate selection
            # was made), "scenario_value_absent" (bid/ask missing in
            # the chain), or something else. Keys are the
            # `missing_reason` codes documented on
            # candidate_return_for_gate.
            "missing_reason_counts": {
                k: int(v) for k, v in sorted(scenario_missing_reason_counts.items())
            },
        },
        "diverged_sample": diverged_sample,
    }


# ─── Pure-math leaves ─────────────────────────────────────────────────────────
# Extracted to web/api/edge_math.py (Phase 3.1). Re-exported here so existing
# `from web.api.edge_engine import <fn>` paths and monkeypatch targets work
# unchanged. Internal callers (e.g. analyze_single_ticker) resolve these names
# through this module's namespace via the re-export below.
from web.api.edge_math import (  # noqa: E402,F401
    _classify_move_risk,
    _score_expectancy,
    _estimate_transaction_cost_pct,
    _compute_move_anchor,
    _compute_move_uncertainty_pct,
    _bsm_greeks,
    _derive_iv_scenarios,
    _calendar_spread_payoff,
    _straddle_payoff,
    _strangle_payoff,
    _rv_percentile_and_regime,
    _excess_kurtosis,
    _kurtosis_confidence_mult,
    _classify_ticker_tier,
    _crush_calibration_mult,
)


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

    # PR #72: bounded interpolation — return np.nan (the legacy
    # "missing" sentinel for this fallback path) when the target
    # tenor isn't bracketed. Pre-fix used np.interp directly which
    # silently clamped to the nearest endpoint.
    _iv30_value, _iv30_status = bounded_interp(30.0, days_arr, ivs_arr)
    _iv45_value, _iv45_status = bounded_interp(45.0, days_arr, ivs_arr)
    iv30 = float(_iv30_value) if _iv30_value is not None else np.nan
    iv45 = float(_iv45_value) if _iv45_value is not None else np.nan
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
    # PR #72: bounded interpolation (same fix as the sibling
    # fallback above). np.nan on out-of-bracket so downstream
    # `np.isfinite(iv30)` checks behave as before.
    _iv30_value, _iv30_status = bounded_interp(30.0, days_arr, ivs_arr)
    _iv45_value, _iv45_status = bounded_interp(45.0, days_arr, ivs_arr)
    iv30 = float(_iv30_value) if _iv30_value is not None else np.nan
    iv45 = float(_iv45_value) if _iv45_value is not None else np.nan
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











# Kelly sizing has been removed.
# Reason: the formula requires a calibrated edge estimate. The expected_return_signal_pct
# formula (18× multiplier) overstates empirical edge by 10–20× (ORATS best-filtered
# straddle: +0.91% gross; our formula: up to +9%). Feeding an overestimated edge into
# Kelly produces 2–10× Kelly, which targets zero or negative long-run growth.
# Position sizing guidance will be re-introduced once isotonic calibration reaches N≥500
# and edge estimates are validated against a live out-of-sample period.












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


def _attach_live_forward_provenance(edge_snapshot: "EdgeSnapshot") -> None:
    """Tag *edge_snapshot* as a live forward observation (PR-AE C3).

    Assigns both:
      - ``sample_provenance`` → ``SAMPLE_PROVENANCE_FORWARD_POST_FREEZE``
        via the SOLE authorized assigner
        ``_tag_live_forward_observation`` (the underscore-prefixed
        helper that lives in
        ``services/candidate_shadow_provenance.py``).
      - ``candidate_shadow_outcome`` → initialized to the stable
        empty-block factory output with status
        ``"awaiting_exit_resolver"`` ONLY when the snapshot does not
        already carry an outcome. Any prior outcome — including
        ``ok``, ``permanently_failed:*``, ``retrying``,
        ``awaiting_chain_data``, or even the
        ``awaiting_exit_resolver`` stub itself — is preserved
        verbatim (PR-AE C3b, Codex audit follow-up).

    Mutates the snapshot in place; returns None.

    Why this exists as a helper instead of inline:
      1. **Testability**: this is the entire surface of PR-AE C3's
         behavior change. Extracting it lets unit tests assert the
         live-tagging contract without standing up the full
         analyze_single_ticker dependency stack (yfinance, MDApp,
         VolSnapshot, structure scorecards, selector, …).
      2. **Boundary preservation**: the only call site of
         ``_tag_live_forward_observation`` inside this module is
         here. The source-grep regression
         ``TestForwardProvenanceAssignmentBoundary`` continues to
         pass because edge_engine.py is on its allowlist.
      3. **No literal exposure**: the string
         ``"forward_post_freeze"`` is NEVER written in this module.
         It flows through the helper from
         ``services/candidate_shadow_provenance.py`` only.

    Defensive non-overwrite (PR-AE C3b — Codex audit P1): the
    candidate_shadow_outcome guard exists so a future caller that
    re-runs this helper on a snapshot already carrying a resolved
    outcome cannot clobber the resolution. In normal flow
    analyze_single_ticker hands the helper a fresh snapshot whose
    candidate_shadow_outcome is the default empty dict, but the
    guard makes the helper safe under any call pattern. The
    sample_provenance assignment is intentionally unconditional —
    the live API path always knows it is producing a forward
    observation, and the underscore-prefixed tagger is the SOLE
    authorized writer of that constant regardless of any prior
    value.

    Historical replay does NOT flow through this helper.
    ``_simulate_pre_earnings_calendar_trade`` builds a plain dict
    with ``sample_provenance = SAMPLE_PROVENANCE_HISTORICAL_REPLAY``
    directly. Behavioral tests verify both paths.
    """
    live_record: Dict[str, Any] = {}
    _tag_live_forward_observation(live_record)
    edge_snapshot.sample_provenance = live_record["sample_provenance"]

    # PR-AE C3b non-overwrite guard. Treat the snapshot's current
    # candidate_shadow_outcome as authoritative whenever it carries a
    # non-empty status string. Empty (no status, or {}/None) means
    # "the live path just created this snapshot," and we initialize
    # the stub. Any non-empty status — including the very stub we
    # would otherwise write — is preserved.
    existing = edge_snapshot.candidate_shadow_outcome
    existing_status = (
        str(existing.get("status") or "")
        if isinstance(existing, dict)
        else ""
    )
    if not existing_status:
        edge_snapshot.candidate_shadow_outcome = _empty_candidate_shadow_outcome(
            "awaiting_exit_resolver"
        )


@dataclass(frozen=True)
class AnalysisInputs:
    """Frozen bundle of non-derivable inputs for a single-ticker analysis (Phase 4.1).

    Produced by build_analysis_inputs() — the gathering phase of analyze_single_ticker.
    Snapshot-derived scalars (iv30, smile_curvature, …) are intentionally NOT duplicated
    here; the orchestrator unpacks them from vol_snapshot via _snapshot_to_edge_inputs.
    """
    clean_symbol: str
    ticker: Any
    close: Any
    market_cap: Optional[float]
    ticker_tier: str
    ticker_tier_mult: float
    current_price: float
    close_for_profile: Any
    chain_df: Any
    data_source: str
    dte: Optional[int]
    earnings_release_time: Optional[str]
    resolved_earnings_event: Any
    earnings_events_for_profile: List[Dict[str, Any]]
    vol_snapshot: Any
    structure_scorecards: Any
    selector_output: Any


def build_analysis_inputs(symbol: str, mda_client: Any = None) -> AnalysisInputs:
    """Gather all non-derivable inputs for analyze_single_ticker (Phase 4.1 seam).

    Extracted verbatim from the orchestrator's gathering phase. This is the clean seam a
    backtest harness can call to obtain the vol snapshot + scorecards + selector output
    for a symbol without running the downstream scoring/gates/metrics phases.
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
    return AnalysisInputs(
        clean_symbol=clean_symbol,
        ticker=ticker,
        close=close,
        market_cap=_market_cap,
        ticker_tier=ticker_tier,
        ticker_tier_mult=ticker_tier_mult,
        current_price=current_price,
        close_for_profile=close_for_profile,
        chain_df=chain_df,
        data_source=data_source,
        dte=dte,
        earnings_release_time=earnings_release_time,
        resolved_earnings_event=resolved_earnings_event,
        earnings_events_for_profile=earnings_events_for_profile,
        vol_snapshot=vol_snapshot,
        structure_scorecards=structure_scorecards,
        selector_output=selector_output,
    )


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
    inputs = build_analysis_inputs(symbol, mda_client)
    clean_symbol = inputs.clean_symbol
    ticker = inputs.ticker
    close = inputs.close
    _market_cap = inputs.market_cap
    ticker_tier = inputs.ticker_tier
    ticker_tier_mult = inputs.ticker_tier_mult
    current_price = inputs.current_price
    close_for_profile = inputs.close_for_profile
    chain_df = inputs.chain_df
    data_source = inputs.data_source
    dte = inputs.dte
    earnings_release_time = inputs.earnings_release_time
    resolved_earnings_event = inputs.resolved_earnings_event
    earnings_events_for_profile = inputs.earnings_events_for_profile
    vol_snapshot = inputs.vol_snapshot
    structure_scorecards = inputs.structure_scorecards
    selector_output = inputs.selector_output
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

    # PR #68: resolve continuous dividend yield q once and thread it
    # through every live BSM call site. Without this, calls/puts/greeks
    # for dividend-paying names (AAPL, MSFT, JPM, KO, …) were computed
    # under the wrong forward — ATM put delta biased toward 0, calendar
    # put-side debit systematically wrong, etc. The `services/dividend_yields`
    # module (PR-J) was already shipped but its consumer wiring stopped
    # at the replay path; the live decision path is what this PR adds.
    #
    # `get_dividend_yield` fails closed to `(0.0, "fallback_zero")` on
    # any lookup failure, so non-dividend names and yfinance hiccups
    # both produce byte-identical-to-legacy pricing.
    pricing_dividend_yield, pricing_dividend_yield_source = get_dividend_yield(clean_symbol)
    _bsm_T = float(dte) if dte is not None and dte > 0 else np.nan
    greeks = (
        _bsm_greeks(
            S=current_price,
            T_days=_bsm_T,
            sigma=float(iv30) if np.isfinite(iv30) else np.nan,
            r=pricing_risk_free_rate,
            q=pricing_dividend_yield,
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
    # SelectorOutput.confidence_pct is a required float field — the getattr fallback was dead code.
    confidence_pct_raw = float(selector_output.confidence_pct)

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

    # ── Grading refinements: compute calibration multipliers (informational; selector owns confidence_pct) ──

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
    # Combined calibration factor — informational only; selector owns the final confidence value.
    # Exposed as "confidence_calibration_mult" in metrics so the components can be audited,
    # but NOT applied to confidence_pct (selector_output.confidence_pct is the authoritative output).
    _calibration_mult = float(np.clip(
        ticker_tier_mult * kurtosis_conf_mult * crush_calibration_mult * _ml_mult,
        0.40, 1.15,
    ))
    confidence_pct = float(selector_output.confidence_pct)

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
            q=pricing_dividend_yield,  # PR #68
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
            r=pricing_risk_free_rate, q=pricing_dividend_yield,  # PR #68
            raw_moves_pct=_sp_raw,
            implied_move_pct=_sp_impl, side="put",
        )
    elif _best_structure == "atm_straddle" and _sp_iv30_ok and _sp_near_dte:
        structure_payoff = _straddle_payoff(
            S=current_price, iv=float(iv30), T_near_days=_sp_near_dte,
            r=pricing_risk_free_rate, q=pricing_dividend_yield,  # PR #68
            raw_moves_pct=_sp_raw, implied_move_pct=_sp_impl,
        )
    elif _best_structure == "otm_strangle" and _sp_iv30_ok and _sp_near_dte and _sp_move_ok:
        structure_payoff = _strangle_payoff(
            S=current_price, iv=float(iv30), T_near_days=_sp_near_dte,
            wing_pct=float(implied_move_total_pct),
            r=pricing_risk_free_rate, q=pricing_dividend_yield,  # PR #68
            raw_moves_pct=_sp_raw, implied_move_pct=_sp_impl,
        )
    else:
        structure_payoff = None

    # ── Experimental: candidate calendar contract selection ──────────────────
    # PR-AC commit 3 — strict shadow surface. Only call_calendar gets live
    # contract recommendations; put_calendar receives an explicit placeholder
    # because we have no put-side forward-validation evidence yet. See
    # _build_experimental_contract_selection docstring for the coverage matrix.
    _earnings_event_date_for_picker = (
        resolved_earnings_event.earnings_date
        if resolved_earnings_event is not None
        and getattr(resolved_earnings_event, "earnings_date", None) is not None
        else None
    )
    experimental_contract_selection = _build_experimental_contract_selection(
        best_structure=_best_structure,
        chain_df=chain_df,
        earnings_event_date=_earnings_event_date_for_picker,
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

    # confidence_pct already equals selector_output.confidence_pct (set above). No reassignment needed.
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
        # PR #68 Codex follow-up: expose the resolved dividend yield
        # alongside the risk-free rate so operators can tell whether
        # pricing used real dividend data ("yfinance" / "env" / "cache")
        # or fell back to the no-dividend default ("fallback_zero").
        # Without this, a yfinance hiccup on a dividend-paying name
        # silently degraded the pricing to the legacy no-dividend
        # path with no visible signal.
        "pricing_dividend_yield": float(pricing_dividend_yield),
        "pricing_dividend_yield_source": pricing_dividend_yield_source,
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
        # ── Experimental candidate contract selection (PR-AC commit 3) ──────
        # Shadow surface. None for non-calendar structures. For call_calendar,
        # carries the dual-picker output for the live chain. For put_calendar,
        # carries an explicit "not yet supported" placeholder. Never replaces
        # any existing field; intended as a research diagnostic only.
        "experimental_contract_selection": experimental_contract_selection,
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

    # PR-AE C3: tag this live recommendation with the canonical
    # forward-post-freeze provenance and attach the pre-resolution
    # candidate_shadow_outcome stub. The logic lives in the small
    # helper below so it can be unit-tested in isolation without
    # running the full analyze_single_ticker pipeline.
    _attach_live_forward_provenance(edge_snapshot)

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
