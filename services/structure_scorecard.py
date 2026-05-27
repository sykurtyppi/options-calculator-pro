from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import logging
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from services.earnings_vol_snapshot import VolSnapshot

logger = logging.getLogger(__name__)


SUPPORTED_STRUCTURES: tuple[str, ...] = (
    "atm_straddle",
    "otm_strangle",
    "call_calendar",
    "put_calendar",
)


@dataclass(frozen=True)
class StructureScorecard:
    structure: str
    eligible: bool
    eligibility_flags: List[str]

    expected_edge_pct: float
    expected_return_pct: float
    expected_iv_contribution_pct: float

    expected_move_fit_score: float

    theta_drag_penalty: float
    execution_penalty: float
    crowding_penalty: float
    concavity_penalty: float
    sample_uncertainty_penalty: float

    sample_confidence: float
    walk_forward_history_count: int
    walk_forward_win_rate: float
    walk_forward_avg_return_pct: float
    walk_forward_rank_score: float

    composite_structure_score: float

    rationale_bullets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WalkForwardPrior:
    structure: str
    history_count: int
    win_rate: float
    avg_return_pct: float
    rank_score: float
    source: str


@dataclass(frozen=True)
class _StructureContext:
    move_fit_score: float
    cheapness_score: float
    timing_score: float
    execution_score: float
    data_quality_score: float
    expected_iv_contribution_pct: float
    expected_return_pct: float
    expected_edge_pct: float
    theta_drag_penalty: float
    execution_penalty: float
    crowding_penalty: float
    concavity_penalty: float
    sample_uncertainty_penalty: float
    sample_confidence: float
    rationale_bullets: List[str]


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = REPO_ROOT / "exports" / "reports"
# Research basis: De Silva, Smith & So (2025, Review of Finance).
# For individual equity options, retail effective half-spread ≈ 7-8% of option midpoint.
# For a 2-leg structure with round-trip (entry + exit): 4 × half-spread ≈ 2× bid-ask spread.
# At spread_pct = 12%: round-trip ≈ 24% of option value.
# ORATS 2021-2025: best-filtered pre-earnings ATM straddle IV expansion ≈ 8-18% of option value.
# When round-trip cost (24%) ≥ typical IV expansion ceiling (18%): structurally unprofitable.
# The old threshold (18%) allowed setups with 36% round-trip cost — empirically indefensible.
ABSOLUTE_SPREAD_THRESHOLD_PCT = 12.0

# Persisted backtest priors are filtered to the 25%-of-bid-ask-cross
# execution scenario when loading. The previous name for this constant
# was `CONSERVATIVE_EXECUTION_SCENARIO`, which was misleading on two
# counts:
#
#   1. services/execution_scenarios.SCENARIO_LEVELS reserves the label
#      "conservative" for a 50%-of-spread cross — twice as harsh as the
#      25% level this filter actually selects. Reading the old name and
#      then looking up execution_scenarios would have made it appear
#      the priors were stricter than they actually are.
#   2. The upstream backtest scripts that emit these rows
#      (scripts/backtest_iv_expansion_study.py:98 and
#      scripts/backtest_pre_earnings_otm_strangle.py:90) only produce
#      "cross_25" and "cross_50" rows. There are no "conservative"
#      rows on disk to filter to; the loader had to map "conservative"
#      back to "cross_25" anyway. Renaming makes that explicit.
#
# PR #65 (planned) will refactor _evaluate_forward_quality_gate and
# _compute_rank_score to consume per-scenario returns directly from
# candidate_shadow_outcome blocks; at that point this single-scenario
# filter becomes unnecessary. For PR #64 (foundations) we only fix the
# naming so the new scenario-labeled fields downstream can use
# "conservative" without overloading the term.
PROMOTION_BASELINE_SCENARIO = "cross_25"
_PRIORS_CACHE_LOCK = threading.Lock()
# Bounded dict keyed on the full signature tuple (includes file mtimes + as_of_date).
# Each distinct as_of_date value gets its own slot, so backtest loops don't thrash
# the production (no-filter) entry.  FIFO eviction after _PRIORS_CACHE_MAX slots.
_PRIORS_CACHE: Dict[tuple, Dict[str, WalkForwardPrior]] = {}
_PRIORS_CACHE_MAX: int = 32


def build_structure_scorecards(
    snapshot: VolSnapshot,
    as_of_date: Optional[date] = None,
) -> List[StructureScorecard]:
    """
    Build scorecards for all four supported structures.

    Parameters
    ----------
    snapshot : VolSnapshot
    as_of_date : date, optional
        If provided, the walk-forward priors are filtered to observations
        recorded on or before this date.  Use snapshot.as_of_date from the
        backtest evaluation loop (#18).  None preserves current production
        behavior (all observations included, O(1) aggregate cache).

    Raises
    ------
    BacktestLeakageError
        If as_of_date is provided and the persistent prior store contains
        paper or live observations dated after as_of_date.  Replay
        observations are exempt (they are the sanctioned backtest signal).
    """
    if as_of_date is not None:
        from services.structure_prior_store import (
            BacktestLeakageError,
            get_structure_prior_store,
        )
        try:
            get_structure_prior_store().check_for_leakage(as_of_date)
        except BacktestLeakageError:
            # The sentinel itself — re-raise unchanged so callers can match it.
            raise
        except Exception as exc:
            # Phase 1.5 contract: if the store cannot be loaded or scanned we
            # cannot guarantee no leakage, so we must NOT silently proceed.
            # Convert into BacktestLeakageError so leakage-aware callers handle
            # both paths uniformly; preserve the underlying cause via `from exc`.
            logger.exception(
                "Leakage sentinel could not run for as_of_date=%s", as_of_date
            )
            raise BacktestLeakageError(
                f"Cannot verify leakage for as_of_date={as_of_date}: "
                f"prior store unavailable ({type(exc).__name__}: {exc})"
            ) from exc

    priors = _load_walk_forward_priors(as_of_date=as_of_date)
    return [
        score_atm_straddle(snapshot, prior=priors["atm_straddle"]),
        score_otm_strangle(snapshot, prior=priors["otm_strangle"]),
        score_call_calendar(snapshot, prior=priors["call_calendar"]),
        score_put_calendar(snapshot, prior=priors["put_calendar"]),
    ]


def score_atm_straddle(snapshot: VolSnapshot, *, prior: Optional[WalkForwardPrior] = None) -> StructureScorecard:
    prior = prior or _load_walk_forward_priors()["atm_straddle"]
    flags = _base_eligibility_flags(snapshot)
    if snapshot.atm_call_spread_pct is None or snapshot.atm_put_spread_pct is None:
        flags.append("cannot_form_atm_straddle")
    leg_spread_pct = _average([snapshot.atm_call_spread_pct, snapshot.atm_put_spread_pct])
    if leg_spread_pct is not None and leg_spread_pct > ABSOLUTE_SPREAD_THRESHOLD_PCT:
        flags.append("structure_spread_exceeds_absolute_threshold")

    # P-5c v2: ratio bounds anchored on the post-Finding-A real-corpus
    # distribution (743 distinct events, 2024-01-01 → 2025-06-30) generated
    # with σ_HAR computed over earnings-excluded sessions (production
    # parity). Using p10/p90 of:
    #   historical_vs_implied_move_ratio: p10=0.12, p90=1.14
    #   tail_vs_implied_move_ratio:       p10=0.24, p90=1.53
    # calibration_basis="post_p5c_v2_real_corpus_743_events_2024_2025_h1"
    move_ratio_score = _score_high_good(snapshot.historical_vs_implied_move_ratio, 0.12, 1.14)
    tail_score = _score_high_good(snapshot.tail_vs_implied_move_ratio, 0.24, 1.53)
    # P-5b: peak bounds anchored on the post-P-5a real-corpus distribution
    # (747 distinct events, 2024-01-01 → 2025-06-30). Using p10/p50/p90 from
    # the canonical /tmp/p5_post_iv_expansion_v2 trade log:
    #   event_implied_move_pct: p10=3.31, p50=5.70, p90=10.44
    # calibration_basis="post_p5a_real_corpus_747_events_2024_2025"
    moderate_event_score = _score_peak(snapshot.event_implied_move_pct, lower=3.31, center=5.70, upper=10.44)
    anchor_score = _score_high_good(snapshot.historical_move_anchor_pct, 3.0, 9.0)
    _using_daily_fallback = (
        getattr(snapshot, "historical_move_source", "earnings_history") == "daily_fallback"
    )
    if _using_daily_fallback:
        # Daily-return history cannot reproduce the earnings-day jump distribution.
        # Sentinel 0.15 prevents the move-fit component from driving the scorecard
        # while still allowing execution/cheapness/timing to contribute.
        move_fit = 0.15
    else:
        move_fit = _clamp01(
            0.35 * move_ratio_score
            + 0.30 * tail_score
            + 0.20 * moderate_event_score
            + 0.15 * anchor_score
        )

    cheapness = _coalesce_unit(snapshot.cheapness_score)
    timing = _coalesce_unit(snapshot.timing_score)
    execution = _coalesce_unit(snapshot.execution_score)
    quality = _coalesce_unit(snapshot.data_quality_score)
    # P-5b: event_move_share band anchored on post-P-5a real corpus (747
    # events). Using p25/p90 of event_move_share_of_total = (0.78, 0.94).
    # Pre-P-5b bounds (0.70, 0.98) sat at p17/p95 — too low and too high for
    # the post-P-5a 1σ-form distribution.
    # calibration_basis="post_p5a_real_corpus_747_events_2024_2025"
    crowding_intensity = _clamp01(
        0.55 * _score_high_good(_first_finite(snapshot.iv_rv_har, snapshot.iv_rv_yz), 1.00, 1.60)
        + 0.25 * _score_high_good(snapshot.event_move_share_of_total, 0.78, 0.94)
        + 0.20 * _score_high_good(snapshot.near_back_iv_ratio, 1.00, 1.15)
    )
    concavity_intensity = _clamp01(
        max(
            1.0 if snapshot.smile_concavity_flag else 0.0,
            _score_high_good(abs(snapshot.smile_curvature) if snapshot.smile_curvature is not None else None, 0.22, 0.70),
        )
    )
    theta_penalty = 0.10 * (1.0 - timing)
    execution_penalty = _execution_penalty(snapshot, leg_spread_pct=leg_spread_pct, structure="atm_straddle")
    crowding_penalty = 0.10 * crowding_intensity
    concavity_penalty = 0.08 * concavity_intensity
    sample_confidence, sample_penalty = _sample_confidence_and_penalty(snapshot, prior, max_penalty=0.12)

    expected_iv_contribution_pct = 12.0 * _clamp01(0.45 * cheapness + 0.35 * move_fit + 0.20 * timing) - 5.0 * crowding_penalty
    expected_return_signal_pct = 18.0 * (0.55 * move_fit + 0.20 * cheapness + 0.15 * timing + 0.10 * execution - 0.50)
    expected_return_pct = _blend_expected_return(prior, expected_return_signal_pct + expected_iv_contribution_pct * 0.35)
    expected_edge_pct = expected_return_pct - _penalty_pct(
        execution_penalty=execution_penalty,
        theta_penalty=theta_penalty,
        crowding_penalty=crowding_penalty,
        concavity_penalty=concavity_penalty,
        sample_penalty=sample_penalty,
    )

    rationale = [
        _ratio_rationale("historical/implied move", snapshot.historical_vs_implied_move_ratio, favorable="above 1 favors long gamma", cautious="below 1 dampens move capture"),
        _ratio_rationale("tail/implied move", snapshot.tail_vs_implied_move_ratio, favorable="fat historical tails support convex payoff", cautious="tails do not clear current implied move"),
        _execution_rationale(snapshot, execution_penalty),
        f"Walk-forward prior: {prior.history_count} observations, {prior.win_rate:.0%} win rate, source={prior.source}.",
    ]
    if _using_daily_fallback:
        rationale.append(
            "Warning: historical move profile uses daily returns (no earnings history available). "
            "Move-fit estimate set to sentinel 0.15 — unreliable for earnings-event sizing."
        )
    context = _StructureContext(
        move_fit_score=move_fit,
        cheapness_score=cheapness,
        timing_score=timing,
        execution_score=execution,
        data_quality_score=quality,
        expected_iv_contribution_pct=expected_iv_contribution_pct,
        expected_return_pct=expected_return_pct,
        expected_edge_pct=expected_edge_pct,
        theta_drag_penalty=theta_penalty,
        execution_penalty=execution_penalty,
        crowding_penalty=crowding_penalty,
        concavity_penalty=concavity_penalty,
        sample_uncertainty_penalty=sample_penalty,
        sample_confidence=sample_confidence,
        rationale_bullets=rationale,
    )
    return _finalize_scorecard("atm_straddle", snapshot, prior, context, flags)


def score_otm_strangle(snapshot: VolSnapshot, *, prior: Optional[WalkForwardPrior] = None) -> StructureScorecard:
    prior = prior or _load_walk_forward_priors()["otm_strangle"]
    flags = _base_eligibility_flags(snapshot)
    if snapshot.smile_points < 3 or snapshot.near_term_atm_iv is None:
        flags.append("cannot_form_otm_strangle")
    if snapshot.near_term_spread_pct is not None and snapshot.near_term_spread_pct > ABSOLUTE_SPREAD_THRESHOLD_PCT:
        flags.append("structure_spread_exceeds_absolute_threshold")

    # P-5c v2: tail ratio bound for OTM strangle uses (p25, p90) of the
    # post-Finding-A real-corpus distribution (743 events, 2024-01-01 →
    # 2025-06-30) generated with earnings-excluded σ_HAR:
    #   tail_vs_implied_move_ratio: p25=0.42, p90=1.53
    # Tighter lower bound than calendar (which uses p10) preserves the
    # prior relative ordering — this structure was always less permissive
    # on tail support.
    # calibration_basis="post_p5c_v2_real_corpus_743_events_2024_2025_h1"
    tail_score = _score_high_good(snapshot.tail_vs_implied_move_ratio, 0.42, 1.53)
    event_risk = _coalesce_unit(snapshot.event_risk_score)
    move_anchor = _score_high_good(snapshot.historical_move_anchor_pct, 4.0, 10.0)
    execution = _coalesce_unit(snapshot.execution_score)
    cheapness = _coalesce_unit(snapshot.cheapness_score)
    timing = _coalesce_unit(snapshot.timing_score)
    quality = _coalesce_unit(snapshot.data_quality_score)
    _using_daily_fallback = (
        getattr(snapshot, "historical_move_source", "earnings_history") == "daily_fallback"
    )
    if _using_daily_fallback:
        move_fit = 0.15
    else:
        move_fit = _clamp01(
            0.45 * tail_score
            + 0.30 * event_risk
            + 0.15 * move_anchor
            + 0.10 * execution
        )

    weak_move_penalty = 0.04 * (1.0 - move_anchor)
    wing_liquidity_penalty = 0.06 * _clamp01(
        0.65 * (1.0 - _score_high_good(snapshot.near_term_liquidity_proxy, 800.0, 3500.0))
        + 0.35 * (1.0 - _score_high_good(_first_finite(snapshot.atm_total_open_interest, snapshot.atm_total_volume), 600.0, 3500.0))
    )
    theta_penalty = 0.11 * (1.0 - timing)
    execution_penalty = _execution_penalty(snapshot, leg_spread_pct=_average([snapshot.atm_call_spread_pct, snapshot.atm_put_spread_pct]), structure="otm_strangle")
    # P-5b: event_move_share band anchored on post-P-5a real corpus (747
    # events). Using p25/p90 of event_move_share_of_total = (0.78, 0.94).
    # calibration_basis="post_p5a_real_corpus_747_events_2024_2025"
    crowding_penalty = 0.08 * _clamp01(
        0.60 * _score_high_good(_first_finite(snapshot.iv_rv_har, snapshot.iv_rv_yz), 1.05, 1.70)
        + 0.40 * _score_high_good(snapshot.event_move_share_of_total, 0.78, 0.94)
    )
    concavity_penalty = 0.05 * _clamp01(
        max(
            1.0 if snapshot.smile_concavity_flag else 0.0,
            _score_high_good(abs(snapshot.smile_curvature) if snapshot.smile_curvature is not None else None, 0.30, 0.80),
        )
    )
    sample_confidence, sample_penalty = _sample_confidence_and_penalty(snapshot, prior, max_penalty=0.11)
    sample_penalty = _clamp01(sample_penalty + weak_move_penalty + wing_liquidity_penalty)

    expected_iv_contribution_pct = 10.0 * _clamp01(0.45 * event_risk + 0.35 * tail_score + 0.20 * cheapness) - 4.0 * crowding_penalty
    expected_return_signal_pct = 22.0 * (0.45 * move_fit + 0.20 * event_risk + 0.15 * timing + 0.10 * execution + 0.10 * cheapness - 0.50)
    expected_return_pct = _blend_expected_return(prior, expected_return_signal_pct + expected_iv_contribution_pct * 0.30)
    expected_edge_pct = expected_return_pct - _penalty_pct(
        execution_penalty=execution_penalty,
        theta_penalty=theta_penalty,
        crowding_penalty=crowding_penalty,
        concavity_penalty=concavity_penalty,
        sample_penalty=sample_penalty,
    )

    rationale = [
        _ratio_rationale("tail/implied move", snapshot.tail_vs_implied_move_ratio, favorable="historical tails support wing exposure", cautious="tails do not justify the wider payoff shape"),
        f"Event-risk score {event_risk:.2f} {'supports' if event_risk >= 0.6 else 'does not strongly support'} a convex wings structure.",
        _execution_rationale(snapshot, execution_penalty),
        f"Walk-forward prior: {prior.history_count} observations, {prior.win_rate:.0%} win rate, source={prior.source}.",
    ]
    if _using_daily_fallback:
        rationale.append(
            "Warning: historical move profile uses daily returns (no earnings history available). "
            "Move-fit estimate set to sentinel 0.15 — unreliable for earnings-event sizing."
        )
    context = _StructureContext(
        move_fit_score=move_fit,
        cheapness_score=cheapness,
        timing_score=timing,
        execution_score=execution,
        data_quality_score=quality,
        expected_iv_contribution_pct=expected_iv_contribution_pct,
        expected_return_pct=expected_return_pct,
        expected_edge_pct=expected_edge_pct,
        theta_drag_penalty=theta_penalty,
        execution_penalty=execution_penalty,
        crowding_penalty=crowding_penalty,
        concavity_penalty=concavity_penalty,
        sample_uncertainty_penalty=sample_penalty,
        sample_confidence=sample_confidence,
        rationale_bullets=rationale,
    )
    return _finalize_scorecard("otm_strangle", snapshot, prior, context, flags)


def score_call_calendar(snapshot: VolSnapshot, *, prior: Optional[WalkForwardPrior] = None) -> StructureScorecard:
    prior = prior or _load_walk_forward_priors()["call_calendar"]
    return _score_calendar(snapshot, prior=prior, structure="call_calendar", leg_spread_pct=snapshot.atm_call_spread_pct)


def score_put_calendar(snapshot: VolSnapshot, *, prior: Optional[WalkForwardPrior] = None) -> StructureScorecard:
    prior = prior or _load_walk_forward_priors()["put_calendar"]
    return _score_calendar(snapshot, prior=prior, structure="put_calendar", leg_spread_pct=snapshot.atm_put_spread_pct)


def _score_calendar(
    snapshot: VolSnapshot,
    *,
    prior: WalkForwardPrior,
    structure: str,
    leg_spread_pct: Optional[float],
) -> StructureScorecard:
    flags = _base_eligibility_flags(snapshot)
    if snapshot.back_term_dte is None or snapshot.back_term_atm_iv is None or snapshot.near_term_atm_iv is None:
        flags.append("cannot_form_calendar")
    if leg_spread_pct is None:
        flags.append("missing_structure_leg_spread")
    if snapshot.near_term_spread_pct is not None and snapshot.near_term_spread_pct > ABSOLUTE_SPREAD_THRESHOLD_PCT:
        flags.append("structure_spread_exceeds_absolute_threshold")

    cheapness = _coalesce_unit(snapshot.cheapness_score)
    timing = _coalesce_unit(snapshot.timing_score)
    execution = _coalesce_unit(snapshot.execution_score)
    quality = _coalesce_unit(snapshot.data_quality_score)
    term_slope_score = _score_high_good(snapshot.term_structure_slope, 0.0000, 0.0035)
    moderate_anchor = _score_peak(snapshot.historical_move_anchor_pct, lower=2.5, center=5.5, upper=8.5)
    _using_daily_fallback = (
        getattr(snapshot, "historical_move_source", "earnings_history") == "daily_fallback"
    )
    if _using_daily_fallback:
        # Calendar is less move-dependent than straddle/strangle, but moderate_anchor
        # still comes from earnings history. Use sentinel to prevent false confidence.
        move_fit = 0.15
    else:
        move_fit = _clamp01(0.45 * cheapness + 0.35 * term_slope_score + 0.20 * moderate_anchor)

    # P-5b: event_move_share band anchored on post-P-5a real corpus (747
    # events). Using p25/p90 of event_move_share_of_total = (0.78, 0.94).
    # calibration_basis="post_p5a_real_corpus_747_events_2024_2025"
    elevated_front_end_penalty = 0.07 * _clamp01(
        0.65 * _score_high_good(snapshot.near_back_iv_ratio, 1.00, 1.15)
        + 0.35 * _score_high_good(snapshot.event_move_share_of_total, 0.78, 0.94)
    )
    # P-5c v2: tail-risk penalty uses (p75, p95) of the post-Finding-A
    # real-corpus distribution (743 events, 2024-01-01 → 2025-06-30)
    # generated with earnings-excluded σ_HAR:
    #   tail_vs_implied_move_ratio: p75=1.22, p95=1.80
    # Penalty engages only when tail ratio sits in the upper quartile of
    # the empirical distribution — preserves the prior intent that this
    # term fires for unusually fat historical tails.
    # calibration_basis="post_p5c_v2_real_corpus_743_events_2024_2025_h1"
    tail_risk_penalty = 0.06 * _score_high_good(snapshot.tail_vs_implied_move_ratio, 1.22, 1.80)
    theta_penalty = 0.05 * (1.0 - timing)
    execution_penalty = _execution_penalty(snapshot, leg_spread_pct=leg_spread_pct, structure=structure)
    crowding_penalty = elevated_front_end_penalty
    concavity_penalty = 0.03 * _clamp01(
        _score_high_good(abs(snapshot.smile_curvature) if snapshot.smile_curvature is not None else None, 0.35, 0.85)
    )
    sample_confidence, sample_penalty = _sample_confidence_and_penalty(snapshot, prior, max_penalty=0.10)
    sample_penalty = _clamp01(sample_penalty + tail_risk_penalty)

    expected_iv_contribution_pct = 8.0 * _clamp01(0.50 * cheapness + 0.30 * term_slope_score + 0.20 * timing) - 5.0 * crowding_penalty
    expected_return_signal_pct = 14.0 * (0.40 * move_fit + 0.25 * cheapness + 0.15 * timing + 0.10 * execution + 0.10 * quality - 0.50)
    expected_return_pct = _blend_expected_return(prior, expected_return_signal_pct + expected_iv_contribution_pct * 0.30)
    expected_edge_pct = expected_return_pct - _penalty_pct(
        execution_penalty=execution_penalty,
        theta_penalty=theta_penalty,
        crowding_penalty=crowding_penalty,
        concavity_penalty=concavity_penalty,
        sample_penalty=sample_penalty,
    )

    rationale = [
        f"Cheapness score {cheapness:.2f} {'supports' if cheapness >= 0.6 else 'does not strongly support'} a calendar entry.",
        (
            f"Term slope {snapshot.term_structure_slope:.4f} "
            f"{'is favorable' if term_slope_score >= 0.6 else 'is not clearly favorable'} for carrying long back-end vega."
            if snapshot.term_structure_slope is not None
            else "Term-structure slope is unavailable; that component stays neutral."
        ),
        _execution_rationale(snapshot, execution_penalty),
        f"Walk-forward prior: {prior.history_count} observations, {prior.win_rate:.0%} win rate, source={prior.source}.",
    ]
    if _using_daily_fallback:
        rationale.append(
            "Warning: historical move profile uses daily returns (no earnings history available). "
            "Move-fit estimate set to sentinel 0.15 — unreliable for earnings-event sizing."
        )
    context = _StructureContext(
        move_fit_score=move_fit,
        cheapness_score=cheapness,
        timing_score=timing,
        execution_score=execution,
        data_quality_score=quality,
        expected_iv_contribution_pct=expected_iv_contribution_pct,
        expected_return_pct=expected_return_pct,
        expected_edge_pct=expected_edge_pct,
        theta_drag_penalty=theta_penalty,
        execution_penalty=execution_penalty,
        crowding_penalty=crowding_penalty,
        concavity_penalty=concavity_penalty,
        sample_uncertainty_penalty=sample_penalty,
        sample_confidence=sample_confidence,
        rationale_bullets=rationale,
    )
    return _finalize_scorecard(structure, snapshot, prior, context, flags)


def _finalize_scorecard(
    structure: str,
    snapshot: VolSnapshot,
    prior: WalkForwardPrior,
    context: _StructureContext,
    flags: Sequence[str],
) -> StructureScorecard:
    flags = sorted(set(flags))
    eligible = len(flags) == 0
    expected_edge_score = _score_high_good(context.expected_edge_pct, -6.0, 18.0)
    composite = _clamp01(
        0.30 * prior.rank_score
        + 0.20 * expected_edge_score
        + 0.15 * context.move_fit_score
        + 0.10 * context.cheapness_score
        + 0.10 * context.timing_score
        + 0.10 * context.execution_score
        + 0.05 * context.data_quality_score
        - context.theta_drag_penalty
        - context.crowding_penalty
        - context.concavity_penalty
        - context.sample_uncertainty_penalty
    )
    if not eligible:
        composite = 0.0

    rationale = list(context.rationale_bullets)
    if flags:
        rationale.insert(0, f"Ineligible: {', '.join(flags)}.")

    return StructureScorecard(
        structure=structure,
        eligible=eligible,
        eligibility_flags=list(flags),
        expected_edge_pct=float(context.expected_edge_pct),
        expected_return_pct=float(context.expected_return_pct),
        expected_iv_contribution_pct=float(context.expected_iv_contribution_pct),
        expected_move_fit_score=float(context.move_fit_score),
        theta_drag_penalty=float(context.theta_drag_penalty),
        execution_penalty=float(context.execution_penalty),
        crowding_penalty=float(context.crowding_penalty),
        concavity_penalty=float(context.concavity_penalty),
        sample_uncertainty_penalty=float(context.sample_uncertainty_penalty),
        sample_confidence=float(context.sample_confidence),
        walk_forward_history_count=int(prior.history_count),
        walk_forward_win_rate=float(prior.win_rate),
        walk_forward_avg_return_pct=float(prior.avg_return_pct),
        walk_forward_rank_score=float(prior.rank_score),
        composite_structure_score=float(composite),
        rationale_bullets=rationale,
    )


def _base_eligibility_flags(snapshot: VolSnapshot) -> List[str]:
    flags: List[str] = []
    if snapshot.earnings_date is None:
        flags.append("missing_earnings_date")
    if snapshot.option_source is None or snapshot.near_term_atm_iv is None:
        flags.append("missing_option_chain")
    if getattr(snapshot, "surface_quality_status", None) == "record_only":
        flags.append("surface_quality_record_only")
    if snapshot.near_term_dte is None or snapshot.near_term_dte < 1:
        flags.append("invalid_near_term_dte")
    if snapshot.near_term_spread_pct is not None and snapshot.near_term_spread_pct > ABSOLUTE_SPREAD_THRESHOLD_PCT:
        flags.append("spread_exceeds_absolute_threshold")
    return sorted(set(flags))


def _sample_confidence_and_penalty(
    snapshot: VolSnapshot,
    prior: WalkForwardPrior,
    *,
    max_penalty: float,
) -> tuple[float, float]:
    hist_conf = _score_high_good(float(snapshot.historical_event_count), 2.0, 10.0)
    wf_conf = _score_high_good(float(prior.history_count), 8.0, 60.0)
    quality_conf = _coalesce_unit(snapshot.data_quality_score)
    sample_conf = _clamp01(0.45 * hist_conf + 0.40 * wf_conf + 0.15 * quality_conf)

    # If the historical move profile was built from daily returns rather than actual
    # earnings events, the move estimates are unreliable (daily returns do not capture
    # the earnings-day jump distribution). Severely discount confidence in that case.
    if getattr(snapshot, "historical_move_source", "earnings_history") == "daily_fallback":
        sample_conf = _clamp01(sample_conf * 0.15)

    return sample_conf, max_penalty * (1.0 - sample_conf)


def _execution_penalty(snapshot: VolSnapshot, *, leg_spread_pct: Optional[float], structure: str) -> float:
    # Soft scoring boundary updated from 14.0 → 10.0.
    # De Silva 2025: at 10% spread, half-spread = 5% — already 2× the median retail rate (7-8%).
    # Using 10% as the "very wide" ceiling ensures the scoring degrades appropriately before
    # the hard gate fires at 12%, rather than holding near-neutral all the way to the old 14% cap.
    spread_penalty = 1.0 - _score_low_good(_first_finite(leg_spread_pct, snapshot.near_term_spread_pct), 1.5, 10.0)
    proxy_penalty = 1.0 - _score_high_good(snapshot.near_term_liquidity_proxy, 500.0, 4000.0)
    oi_penalty = 1.0 - _score_high_good(_first_finite(snapshot.atm_total_open_interest, snapshot.atm_total_volume), 500.0, 4000.0)
    execution_score = _coalesce_unit(snapshot.execution_score)

    structure_scale = {
        "atm_straddle": 0.12,
        "otm_strangle": 0.14,
        "call_calendar": 0.12,
        "put_calendar": 0.12,
    }[structure]
    return structure_scale * _clamp01(
        0.45 * spread_penalty
        + 0.25 * (1.0 - execution_score)
        + 0.20 * proxy_penalty
        + 0.10 * oi_penalty
    )


def _penalty_pct(
    *,
    execution_penalty: float,
    theta_penalty: float,
    crowding_penalty: float,
    concavity_penalty: float,
    sample_penalty: float,
) -> float:
    return (
        26.0 * execution_penalty
        + 18.0 * theta_penalty
        + 16.0 * crowding_penalty
        + 12.0 * concavity_penalty
        + 10.0 * sample_penalty
    )


def _blend_expected_return(prior: WalkForwardPrior, signal_return_pct: float) -> float:
    wf_weight = _clamp01(prior.history_count / 40.0) * 0.55
    return (wf_weight * prior.avg_return_pct) + ((1.0 - wf_weight) * signal_return_pct)


def _ratio_rationale(label: str, value: Optional[float], *, favorable: str, cautious: str) -> str:
    if value is None or not np.isfinite(value):
        return f"{label} unavailable; this scorecard keeps that component neutral."
    if value >= 1.0:
        return f"{label} at {value:.2f}; {favorable}."
    return f"{label} at {value:.2f}; {cautious}."


def _execution_rationale(snapshot: VolSnapshot, execution_penalty: float) -> str:
    spread = snapshot.near_term_spread_pct
    if spread is None:
        return "Execution spread is unavailable; execution stays near neutral until live quotes are present."
    if execution_penalty >= 0.08:
        return f"Near-term spread {spread:.2f}% creates a material execution drag."
    if execution_penalty >= 0.04:
        return f"Near-term spread {spread:.2f}% is manageable but not trivial."
    return f"Near-term spread {spread:.2f}% is supportive for execution."


def _coalesce_unit(value: Optional[float], *, neutral: float = 0.50) -> float:
    if value is None or not np.isfinite(value):
        return neutral
    return _clamp01(float(value))


def _score_high_good(value: Optional[float], low: float, high: float, *, neutral: float = 0.50) -> float:
    if value is None or not np.isfinite(value):
        return neutral
    if high <= low:
        return neutral
    return _clamp01((float(value) - low) / (high - low))


def _score_low_good(value: Optional[float], low: float, high: float, *, neutral: float = 0.50) -> float:
    if value is None or not np.isfinite(value):
        return neutral
    if high <= low:
        return neutral
    return _clamp01(1.0 - ((float(value) - low) / (high - low)))


def _score_peak(value: Optional[float], *, lower: float, center: float, upper: float, neutral: float = 0.50) -> float:
    if value is None or not np.isfinite(value):
        return neutral
    val = float(value)
    if val <= lower or val >= upper:
        return 0.0
    if val == center:
        return 1.0
    if val < center:
        return _clamp01((val - lower) / (center - lower))
    return _clamp01((upper - val) / (upper - center))


def _average(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(val) for val in values if val is not None and np.isfinite(val)]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _first_finite(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _neutral_prior(structure: str, *, source: str) -> WalkForwardPrior:
    return WalkForwardPrior(
        structure=structure,
        history_count=0,
        win_rate=0.50,
        avg_return_pct=0.0,
        rank_score=0.50,
        source=source,
    )


def _compute_rank_score(*, win_rate: Optional[float], avg_return_pct: Optional[float], history_count: int) -> float:
    return_score = _score_high_good(avg_return_pct, -10.0, 15.0)
    win_score = _score_high_good(win_rate, 0.35, 0.70)
    history_score = _score_high_good(float(history_count), 5.0, 50.0)
    return _clamp01(0.45 * return_score + 0.30 * win_score + 0.25 * history_score)


def _avg_return_proxy_from_frame(frame: pd.DataFrame) -> Optional[float]:
    if "avg_return_pct" in frame.columns:
        series = pd.to_numeric(frame["avg_return_pct"], errors="coerce").dropna()
        if not series.empty:
            return float(series.mean())

    quantile_cols = [col for col in ("p10_trade_return_pct", "p50_trade_return_pct", "p90_trade_return_pct") if col in frame.columns]
    if len(quantile_cols) == 3:
        p10 = pd.to_numeric(frame["p10_trade_return_pct"], errors="coerce")
        p50 = pd.to_numeric(frame["p50_trade_return_pct"], errors="coerce")
        p90 = pd.to_numeric(frame["p90_trade_return_pct"], errors="coerce")
        proxy = ((p10 + (2.0 * p50) + p90) / 4.0).dropna()
        if not proxy.empty:
            return float(proxy.mean())
    return None


def _latest_report_file(glob_pattern: str) -> Optional[Path]:
    matches = sorted(REPORTS_ROOT.glob(glob_pattern))
    return matches[-1] if matches else None


def _load_calendar_prior_from_reports(structure: str) -> WalkForwardPrior:
    path = _latest_report_file("iv_expansion_study_*/iv_expansion_simulated_scoreboard.csv")
    if path is None:
        return _neutral_prior(structure, source="neutral_missing_iv_expansion_report")

    frame = pd.read_csv(path)
    subset = frame[(frame["structure"] == structure) & (frame["scenario"] == PROMOTION_BASELINE_SCENARIO)].copy()
    if subset.empty:
        return _neutral_prior(structure, source=f"neutral_missing_{structure}_scenario")

    history_count = int(pd.to_numeric(subset["simulated_priceable_count"], errors="coerce").fillna(0).max())
    win_rate = float(pd.to_numeric(subset["win_rate"], errors="coerce").dropna().mean()) if subset["win_rate"].notna().any() else 0.50
    avg_return = _avg_return_proxy_from_frame(subset)
    if avg_return is None:
        avg_return = 0.0
    rank = _compute_rank_score(win_rate=win_rate, avg_return_pct=avg_return, history_count=history_count)
    return WalkForwardPrior(
        structure=structure,
        history_count=history_count,
        win_rate=win_rate,
        avg_return_pct=float(avg_return),
        rank_score=rank,
        source=f"{path.parent.name}:{PROMOTION_BASELINE_SCENARIO}",
    )


def _load_straddle_prior_from_reports() -> WalkForwardPrior:
    path = _latest_report_file("pre_earnings_otm_strangle_*/pre_earnings_otm_strangle_scoreboard.csv")
    if path is None:
        return _neutral_prior("atm_straddle", source="neutral_missing_strangle_report")

    frame = pd.read_csv(path)
    subset = frame[(frame["structure"] == "atm_straddle") & (frame["execution_scenario"] == PROMOTION_BASELINE_SCENARIO)].copy()
    if subset.empty:
        return _neutral_prior("atm_straddle", source=f"neutral_missing_atm_straddle_{PROMOTION_BASELINE_SCENARIO}")

    weights = pd.to_numeric(subset["realized_count"], errors="coerce").fillna(0.0)
    valid_weights = weights.where(weights > 0.0, 1.0)
    wins = pd.to_numeric(subset["win_rate"], errors="coerce").fillna(0.50)
    history_count = int(pd.to_numeric(subset["realized_count"], errors="coerce").fillna(0).median())
    win_rate = float(np.average(wins, weights=valid_weights))
    avg_return = _avg_return_proxy_from_frame(subset)
    if avg_return is None:
        avg_return = 0.0
    rank = _compute_rank_score(win_rate=win_rate, avg_return_pct=avg_return, history_count=history_count)
    return WalkForwardPrior(
        structure="atm_straddle",
        history_count=history_count,
        win_rate=win_rate,
        avg_return_pct=float(avg_return),
        rank_score=rank,
        source=f"{path.parent.name}:{PROMOTION_BASELINE_SCENARIO}",
    )


def _load_strangle_prior_from_reports() -> WalkForwardPrior:
    summary_path = _latest_report_file("pre_earnings_otm_strangle_*/pre_earnings_otm_strangle_summary.json")
    if summary_path is not None:
        import json

        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        baseline_rows = payload.get("baseline_cost_table") or []
        match = next(
            (
                row for row in baseline_rows
                if str(row.get("execution_scenario")) == PROMOTION_BASELINE_SCENARIO
            ),
            None,
        )
        if match is not None:
            avg_return = _avg_return_proxy_from_frame(pd.DataFrame([match]))
            if avg_return is None:
                avg_return = 0.0
            history_count = int(match.get("realized_count", 0) or 0)
            win_rate = float(match.get("win_rate", 0.50) or 0.50)
            rank = _compute_rank_score(win_rate=win_rate, avg_return_pct=avg_return, history_count=history_count)
            return WalkForwardPrior(
                structure="otm_strangle",
                history_count=history_count,
                win_rate=win_rate,
                avg_return_pct=float(avg_return),
                rank_score=rank,
                source=f"{summary_path.parent.name}:baseline:{PROMOTION_BASELINE_SCENARIO}",
            )

    path = _latest_report_file("pre_earnings_otm_strangle_*/pre_earnings_otm_strangle_scoreboard.csv")
    if path is None:
        return _neutral_prior("otm_strangle", source="neutral_missing_strangle_report")

    frame = pd.read_csv(path)
    subset = frame[(frame["structure"] == "otm_strangle_3pct") & (frame["execution_scenario"] == PROMOTION_BASELINE_SCENARIO)].copy()
    if subset.empty:
        return _neutral_prior("otm_strangle", source=f"neutral_missing_otm_strangle_{PROMOTION_BASELINE_SCENARIO}")

    weights = pd.to_numeric(subset["realized_count"], errors="coerce").fillna(0.0)
    valid_weights = weights.where(weights > 0.0, 1.0)
    history_count = int(pd.to_numeric(subset["realized_count"], errors="coerce").fillna(0).median())
    wins = pd.to_numeric(subset["win_rate"], errors="coerce").fillna(0.50)
    win_rate = float(np.average(wins, weights=valid_weights))
    avg_return = _avg_return_proxy_from_frame(subset)
    if avg_return is None:
        avg_return = 0.0
    rank = _compute_rank_score(win_rate=win_rate, avg_return_pct=avg_return, history_count=history_count)
    return WalkForwardPrior(
        structure="otm_strangle",
        history_count=history_count,
        win_rate=win_rate,
        avg_return_pct=float(avg_return),
        rank_score=rank,
        source=f"{path.parent.name}:{PROMOTION_BASELINE_SCENARIO}",
    )


def _walk_forward_prior_signature(
    as_of_date: Optional[date] = None,
) -> tuple[tuple[str, Optional[float]], ...]:
    tracked_paths: list[Path] = []
    calendar_path = _latest_report_file("iv_expansion_study_*/iv_expansion_simulated_scoreboard.csv")
    if calendar_path is not None:
        tracked_paths.append(calendar_path)
    for pattern in (
        "pre_earnings_otm_strangle_*/pre_earnings_otm_strangle_summary.json",
        "pre_earnings_otm_strangle_*/pre_earnings_otm_strangle_scoreboard.csv",
    ):
        path = _latest_report_file(pattern)
        if path is not None:
            tracked_paths.append(path)
    try:
        from services.structure_prior_store import _DEFAULT_STORE as _STRUCTURE_PRIOR_STORE

        tracked_paths.append(_STRUCTURE_PRIOR_STORE)
    except Exception:
        pass

    signature: list[tuple[str, Optional[float]]] = []
    for path in sorted({p.resolve() for p in tracked_paths}):
        try:
            mtime = path.stat().st_mtime if path.exists() else None
        except OSError:
            mtime = None
        signature.append((str(path), mtime))
    # Include as_of_date in the cache key so as-of calls don't collide with
    # the production (no-filter) cache or with each other (#18).
    as_of_key: tuple[str, Optional[float]] = (
        "as_of_date", float(as_of_date.toordinal()) if as_of_date is not None else None
    )
    return tuple(signature) + (as_of_key,)


def _load_walk_forward_priors(
    as_of_date: Optional[date] = None,
) -> Dict[str, WalkForwardPrior]:
    """Load walk-forward priors from reports and the persistent store.

    Parameters
    ----------
    as_of_date : date, optional
        If provided, the persistent-store overlay is filtered to observations
        recorded on or before this date (#18).  None preserves current
        production behavior (unfiltered, uses the O(1) aggregate cache).
    """
    signature = _walk_forward_prior_signature(as_of_date=as_of_date)
    with _PRIORS_CACHE_LOCK:
        if signature in _PRIORS_CACHE:
            return _PRIORS_CACHE[signature]

    base: Dict[str, WalkForwardPrior] = {
        "call_calendar": _load_calendar_prior_from_reports("call_calendar"),
        "put_calendar": _load_calendar_prior_from_reports("put_calendar"),
        "atm_straddle": _load_straddle_prior_from_reports(),
        "otm_strangle": _load_strangle_prior_from_reports(),
    }
    # Overlay with the durable persistent store for any structure that has
    # accumulated >= MIN_OBS_FOR_OVERRIDE (5) real observations.
    # Below that threshold the report-based prior continues to govern.
    # The persistent store is imported lazily to avoid a circular dependency
    # (structure_prior_store does not import from structure_scorecard).
    try:
        from services.structure_prior_store import load_all_structure_priors

        for structure, d in load_all_structure_priors(as_of_date=as_of_date).items():
            if structure in base:
                base[structure] = WalkForwardPrior(
                    structure=d["structure"],
                    history_count=d["history_count"],
                    win_rate=d["win_rate"],
                    avg_return_pct=d["avg_return_pct"],
                    rank_score=d["rank_score"],
                    source=d["source"],
                )
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "_load_walk_forward_priors: persistent store overlay failed (%s)", exc
        )
    with _PRIORS_CACHE_LOCK:
        if len(_PRIORS_CACHE) >= _PRIORS_CACHE_MAX:
            _PRIORS_CACHE.pop(next(iter(_PRIORS_CACHE)))
        _PRIORS_CACHE[signature] = base
        return base


def _clear_walk_forward_priors_cache() -> None:
    with _PRIORS_CACHE_LOCK:
        _PRIORS_CACHE.clear()


def reload_walk_forward_priors() -> None:
    """
    Invalidate the walk-forward prior cache.

    Call this after updating the structure prior store so that the next
    call to build_structure_scorecards() picks up the updated priors.
    """
    _clear_walk_forward_priors_cache()


_load_walk_forward_priors.cache_clear = _clear_walk_forward_priors_cache  # type: ignore[attr-defined]  # noqa: E501
