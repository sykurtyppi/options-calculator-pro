"""Compatibility adapters for the legacy edge-analysis response path.

The canonical modeling surfaces now live in VolSnapshot, structure scorecards,
and selector output. This module keeps the old edge endpoint's variable names
and presentation glue out of ``edge_engine.py`` so that file can focus on
orchestration and legacy diagnostics rather than becoming another feature
definition source.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from services.earnings_vol_snapshot import VolSnapshot
from services.screener_service import compute_ranking_score


def snapshot_to_edge_inputs(snapshot: VolSnapshot) -> Dict[str, Any]:
    """Map canonical snapshot fields to legacy edge-engine variable names."""
    return {
        "current_price": snapshot.underlying_price,
        "dte": snapshot.days_to_earnings,
        "earnings_release_time": None if snapshot.release_timing == "unknown" else snapshot.release_timing,
        "rv30_yz": snapshot.rv30_yang_zhang,
        "rv30": snapshot.rv30_yang_zhang,
        "rv_estimator": snapshot.rv30_estimator,
        "rv_har_forecast": snapshot.rv_har_forecast,
        "rv_percentile_rank": snapshot.rv_percentile_rank,
        "vol_regime": snapshot.vol_regime_label,
        "iv30": snapshot.iv30,
        "iv45": snapshot.iv45,
        "ts_slope_0_45": snapshot.term_structure_slope,
        "implied_move_pct": snapshot.near_term_implied_move_pct,
        "near_term_spread_pct": snapshot.near_term_spread_pct,
        "near_term_dte": snapshot.near_term_dte,
        "near_back_iv_ratio": snapshot.near_back_iv_ratio,
        "near_term_liquidity_proxy": snapshot.near_term_liquidity_proxy,
        "liquidity": float(snapshot.near_term_liquidity_proxy or 0.0),
        "smile_curvature": snapshot.smile_curvature,
        "smile_state": {
            "curvature": snapshot.smile_curvature,
            "concave": bool(snapshot.smile_concavity_flag) if snapshot.smile_concavity_flag is not None else False,
            "points": int(snapshot.smile_points or 0),
        },
        "median_earnings_move_pct": snapshot.historical_median_move_pct,
        "p90_earnings_move_pct": snapshot.historical_p90_move_pct,
        "avg_last4_move_pct": snapshot.historical_avg_last4_move_pct,
        "move_std_pct": snapshot.historical_move_std_pct,
        "move_anchor_pct": snapshot.historical_move_anchor_pct,
        "move_sample_size": int(snapshot.historical_event_count or 0),
        "move_source": snapshot.historical_move_source,
        "move_uncertainty_pct": snapshot.historical_move_uncertainty_pct,
        "event_implied_move_pct": snapshot.event_implied_move_pct,
        "non_event_move_pct": snapshot.non_event_move_pct_har,
        "iv_rv": snapshot.iv_rv_yz,
        "iv_rv_har": snapshot.iv_rv_har,
        "data_quality": snapshot.data_quality,
        "data_quality_score": snapshot.data_quality_score,
        "price_staleness_minutes": snapshot.price_staleness_minutes,
        "chain_staleness_minutes": snapshot.chain_staleness_minutes,
    }


def selector_lead_scorecard(
    scorecards: List[Any],
    selector_output: Any,
) -> Optional[Any]:
    requested = getattr(selector_output, "best_structure", None)
    if requested:
        for card in scorecards:
            if getattr(card, "structure", None) == requested:
                return card
    eligible = [card for card in scorecards if getattr(card, "eligible", False)]
    if eligible:
        return max(
            eligible,
            key=lambda card: (
                float(getattr(card, "composite_structure_score", 0.0) or 0.0),
                float(getattr(card, "expected_edge_pct", 0.0) or 0.0),
            ),
        )
    return scorecards[0] if scorecards else None


def shared_setup_score_from_snapshot(snapshot: VolSnapshot) -> float:
    """Legacy setup score backed by the screener's shared ranking definition."""
    return float(
        np.clip(
            compute_ranking_score(
                iv_rv_ratio=snapshot.iv_rv_yz,
                ts_ratio=snapshot.near_back_iv_ratio,
                median_earnings_move_pct=snapshot.historical_median_move_pct,
                sample_size=int(snapshot.historical_event_count or 0),
                dte=snapshot.days_to_earnings,
                spread_pct=snapshot.near_term_spread_pct,
            ),
            0.0,
            1.0,
        )
    )


def legacy_recommendation_from_selector(selector_output: Any) -> str:
    recommendation = getattr(selector_output, "recommendation", None)
    return str(recommendation) if recommendation else "No Trade"


def calibration_phase_note(diag: Optional[Dict[str, Any]]) -> Optional[str]:
    if not diag:
        return None
    phase = str(diag.get("phase") or "unknown")
    n = int(diag.get("n_observations") or 0)
    min_obs = int(diag.get("min_for_observational") or 0)
    min_fit = int(diag.get("min_for_fit") or 0)
    if phase == "bootstrap_prior":
        return (
            f"Calibration phase={phase} ({n}/{min_obs} observations): research prior only, "
            "not an empirical fit."
        )
    if phase == "observational":
        return (
            f"Calibration phase={phase} (N={n}): raw bucket observations only; "
            f"fitted calibration held back until N={min_fit}."
        )
    return f"Calibration phase={phase} (N={n})."
