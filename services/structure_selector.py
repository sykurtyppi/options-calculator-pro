from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import StructureScorecard


RECOMMENDATION_NO_TRADE = "No Trade"
RECOMMENDATION_WATCH = "Watch"
RECOMMENDATION_CANDIDATE = "Candidate"
RECOMMENDATION_BEST = "Best Candidate"

HIGH_EXECUTION_PENALTY_THRESHOLD = 0.10
WATCH_EXECUTION_PENALTY_THRESHOLD = 0.07
MIN_DATA_QUALITY_FOR_ANY_TRADE = 0.45
MIN_DATA_QUALITY_FOR_BEST = 0.75
MIN_WALK_FORWARD_HISTORY = 8
STRONG_WALK_FORWARD_HISTORY = 20
DOMINANT_SCORE_GAP = 0.12
MODERATE_SCORE_GAP = 0.06
MARGINAL_EDGE_THRESHOLD_PCT = 1.50
STRONG_EDGE_THRESHOLD_PCT = 4.00
HIGH_CONFLICT_PENALTY = 0.07

# Research basis: De Silva, Smith & So (2025, Review of Finance) show that when effective
# transaction costs consume ≥50% of gross theoretical return, realized net returns converge
# to zero or below after accounting for realistic trading conditions: bid-ask widening during
# fill, partial fills at unfavorable prices, and time-of-day impact.
# The 50% threshold provides a conservative margin of safety and fires before the absolute
# execution_penalty >= 0.10 gate catches extreme cases.
#
# Derivation (internally consistent with structure_scorecard._penalty_pct):
#   execution_cost_in_scorecard_pct = 26.0 × execution_penalty
#   gross_return_pct = expected_edge_pct + execution_cost_in_scorecard_pct  (reverse-engineer pre-cost)
#   veto fires when: execution_cost / gross_return >= EXECUTION_COST_DOMINATES_EDGE_RATIO
EXECUTION_COST_DOMINATES_EDGE_RATIO = 0.50


@dataclass(frozen=True)
class SelectorOutput:
    symbol: str
    as_of: date
    earnings_date: Optional[date]
    release_timing: str

    recommendation: str
    best_structure: Optional[str]

    confidence_pct: float
    expected_edge_pct: float
    expected_return_pct: float
    expected_edge_tier: str
    expected_return_signal: str
    model_output_note: str

    primary_thesis: str
    primary_risks: List[str]

    why_this_structure: List[str]
    why_not_others: Dict[str, List[str]]

    runner_up_structures: List[str]

    data_quality: str
    data_quality_score: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def select_best_structure(
    snapshot: VolSnapshot,
    scorecards: List[StructureScorecard],
) -> SelectorOutput:
    eligible = [card for card in scorecards if card.eligible]
    ranked = sorted(eligible, key=lambda card: (card.composite_structure_score, card.expected_edge_pct), reverse=True)
    runner_ups = [card.structure for card in ranked[1:3]]
    why_not = {card.structure: _explain_not_selected(card) for card in scorecards if not ranked or card.structure != ranked[0].structure}

    if not ranked:
        return SelectorOutput(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of_date,
            earnings_date=snapshot.earnings_date,
            release_timing=snapshot.release_timing,
            recommendation=RECOMMENDATION_NO_TRADE,
            best_structure=None,
            confidence_pct=_confidence_pct(composite=0.0, gap=0.0, sample_confidence=0.0, data_quality_score=snapshot.data_quality_score),
            expected_edge_pct=0.0,
            expected_return_pct=0.0,
            expected_edge_tier="Negative / Unclear",
            expected_return_signal="Weak",
            model_output_note=_model_output_note(),
            primary_thesis="No structure cleared the minimum eligibility checks for this earnings snapshot.",
            primary_risks=_common_risks(snapshot, None),
            why_this_structure=["Every supported structure failed a hard eligibility rule, so the engine abstains."],
            why_not_others=why_not,
            runner_up_structures=[],
            data_quality=snapshot.data_quality,
            data_quality_score=float(snapshot.data_quality_score),
        )

    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    gap = max(top.composite_structure_score - runner_up.composite_structure_score, 0.0) if runner_up is not None else top.composite_structure_score
    conflicting = _has_conflicting_signals(snapshot, top)
    weak_history = top.walk_forward_history_count < MIN_WALK_FORWARD_HISTORY and gap < DOMINANT_SCORE_GAP
    top_penalty = max(top.execution_penalty, top.crowding_penalty, top.concavity_penalty, top.theta_drag_penalty)

    # Execution-cost-dominates-edge veto (De Silva et al. 2025).
    # Compute gross return BEFORE the execution cost was deducted, then check the ratio.
    # The multiplier 26.0 matches structure_scorecard._penalty_pct(execution_penalty * 26.0),
    # keeping both sides in the same internal scorecard units.
    _exec_cost_pct = 26.0 * float(top.execution_penalty)
    _gross_return_pct = float(top.expected_edge_pct) + _exec_cost_pct
    _cost_dominates_edge = (
        _gross_return_pct > 0.0
        and _exec_cost_pct >= EXECUTION_COST_DOMINATES_EDGE_RATIO * _gross_return_pct
    )

    if (
        top.expected_edge_pct <= 0.0
        or top.execution_penalty >= HIGH_EXECUTION_PENALTY_THRESHOLD
        or snapshot.data_quality_score < MIN_DATA_QUALITY_FOR_ANY_TRADE
        or weak_history
        or _cost_dominates_edge
    ):
        recommendation = RECOMMENDATION_NO_TRADE
    elif (
        gap < MODERATE_SCORE_GAP
        or top.expected_edge_pct < MARGINAL_EDGE_THRESHOLD_PCT
        or top_penalty >= WATCH_EXECUTION_PENALTY_THRESHOLD
        or conflicting
    ):
        recommendation = RECOMMENDATION_WATCH
    elif (
        gap >= DOMINANT_SCORE_GAP
        and top.expected_edge_pct >= STRONG_EDGE_THRESHOLD_PCT
        and top.execution_penalty <= 0.04
        and top.walk_forward_history_count >= STRONG_WALK_FORWARD_HISTORY
        and snapshot.data_quality_score >= MIN_DATA_QUALITY_FOR_BEST
        and top.sample_confidence >= 0.60
    ):
        recommendation = RECOMMENDATION_BEST
    else:
        recommendation = RECOMMENDATION_CANDIDATE

    confidence = _confidence_pct(
        composite=top.composite_structure_score,
        gap=gap,
        sample_confidence=top.sample_confidence,
        data_quality_score=snapshot.data_quality_score,
    )

    return SelectorOutput(
        symbol=snapshot.symbol,
        as_of=snapshot.as_of_date,
        earnings_date=snapshot.earnings_date,
        release_timing=snapshot.release_timing,
        recommendation=recommendation,
        best_structure=top.structure if recommendation != RECOMMENDATION_NO_TRADE else None,
        confidence_pct=confidence,
        expected_edge_pct=float(top.expected_edge_pct) if recommendation != RECOMMENDATION_NO_TRADE else 0.0,
        expected_return_pct=float(top.expected_return_pct) if recommendation != RECOMMENDATION_NO_TRADE else 0.0,
        expected_edge_tier=_expected_edge_tier(float(top.expected_edge_pct) if recommendation != RECOMMENDATION_NO_TRADE else 0.0),
        expected_return_signal=_expected_return_signal(float(top.expected_return_pct) if recommendation != RECOMMENDATION_NO_TRADE else 0.0),
        model_output_note=_model_output_note(),
        primary_thesis=_primary_thesis(snapshot, top, recommendation),
        primary_risks=_common_risks(snapshot, top),
        why_this_structure=_explain_selected(snapshot, top, gap, recommendation),
        why_not_others=why_not,
        runner_up_structures=runner_ups,
        data_quality=snapshot.data_quality,
        data_quality_score=float(snapshot.data_quality_score),
    )


def _confidence_pct(
    *,
    composite: float,
    gap: float,
    sample_confidence: float,
    data_quality_score: float,
) -> float:
    gap_component = float(np.clip(gap / 0.20, 0.0, 1.0))
    blended = (
        0.35 * float(np.clip(composite, 0.0, 1.0))
        + 0.25 * gap_component
        + 0.20 * float(np.clip(sample_confidence, 0.0, 1.0))
        + 0.20 * float(np.clip(data_quality_score, 0.0, 1.0))
    )
    return float(np.clip(blended * 100.0, 0.0, 100.0))


def _expected_edge_tier(expected_edge_pct: float) -> str:
    if not np.isfinite(expected_edge_pct) or expected_edge_pct <= 0.0:
        return "Negative / Unclear"
    if expected_edge_pct < MARGINAL_EDGE_THRESHOLD_PCT:
        return "Marginal"
    return "Positive"


def _expected_return_signal(expected_return_pct: float) -> str:
    if not np.isfinite(expected_return_pct) or expected_return_pct <= 0.0:
        return "Weak"
    if expected_return_pct < 3.0:
        return "Mixed"
    return "Supportive"


def _model_output_note() -> str:
    return (
        "Expected-edge and expected-return fields are score-derived diagnostics. "
        "They are not empirical return forecasts and are not calibrated to live retail execution."
    )


def _has_conflicting_signals(snapshot: VolSnapshot, scorecard: StructureScorecard) -> bool:
    high_move_fit = scorecard.expected_move_fit_score >= 0.65
    elevated_penalty = (
        scorecard.concavity_penalty >= HIGH_CONFLICT_PENALTY
        or scorecard.crowding_penalty >= HIGH_CONFLICT_PENALTY
        or scorecard.execution_penalty >= WATCH_EXECUTION_PENALTY_THRESHOLD
    )
    thin_evidence = scorecard.walk_forward_history_count < MIN_WALK_FORWARD_HISTORY and scorecard.sample_confidence < 0.55
    return bool(high_move_fit and elevated_penalty) or thin_evidence


def _primary_thesis(snapshot: VolSnapshot, scorecard: StructureScorecard, recommendation: str) -> str:
    if recommendation == RECOMMENDATION_NO_TRADE:
        return (
            f"{scorecard.structure.replace('_', ' ').title()} ranks first on this snapshot, "
            f"but the edge does not clear the engine's conservative execution and evidence thresholds."
        )
    if scorecard.structure == "atm_straddle":
        return (
            f"ATM straddle leads because realized earnings moves and tails still outrun the current implied move, "
            f"while execution remains manageable."
        )
    if scorecard.structure == "otm_strangle":
        return (
            f"OTM strangle leads because the snapshot shows elevated event-tail risk that is better monetized by wings "
            f"than by an ATM or calendar structure."
        )
    return (
        f"{scorecard.structure.replace('_', ' ').title()} leads because front-end pricing still looks relatively cheap "
        f"versus the back term, with a term structure that supports carrying longer-dated vega."
    )


def _common_risks(snapshot: VolSnapshot, scorecard: Optional[StructureScorecard]) -> List[str]:
    risks: List[str] = []
    if scorecard is None or scorecard.execution_penalty >= 0.04 or (snapshot.near_term_spread_pct or 0.0) >= 6.0:
        risks.append("Execution risk remains meaningful if live spreads widen or fills miss mid-market pricing.")
    if scorecard is None or scorecard.crowding_penalty >= 0.05 or (snapshot.iv_rv_yz or 0.0) >= 1.35:
        risks.append("The setup could already be partially priced, reducing realized edge versus the modelled expectation.")
    if snapshot.data_quality_score < 0.75 or (snapshot.price_staleness_minutes or 0) > 0:
        risks.append("Data quality is not perfect, so the snapshot should be treated as an auditable estimate rather than ground truth.")
    if scorecard is None or scorecard.structure in {"atm_straddle", "otm_strangle"}:
        risks.append("A muted earnings reaction can still overwhelm IV expansion if theta decay dominates before the event.")
    else:
        risks.append("Large historical tails still matter because a calendar can lag if the front-end premium is already elevated.")
    return risks[:4]


def _explain_selected(
    snapshot: VolSnapshot,
    scorecard: StructureScorecard,
    gap: float,
    recommendation: str,
) -> List[str]:
    bullets = [
        f"Composite score {scorecard.composite_structure_score:.2f} leads the structure set with a score gap of {gap:.2f}.",
        (
            f"Edge quality is {_expected_edge_tier(scorecard.expected_edge_pct).lower()} and return support is "
            f"{_expected_return_signal(scorecard.expected_return_pct).lower()}; these are score-derived diagnostics, "
            "not empirical return forecasts."
        ),
        f"Walk-forward support is {scorecard.walk_forward_history_count} observations with rank score {scorecard.walk_forward_rank_score:.2f}.",
    ]
    if scorecard.structure in {"call_calendar", "put_calendar"}:
        bullets.append(
            f"Cheapness {snapshot.cheapness_score if snapshot.cheapness_score is not None else 0.5:.2f} and term slope "
            f"{snapshot.term_structure_slope if snapshot.term_structure_slope is not None else 0.0:.4f} fit a calendar profile."
        )
    elif scorecard.structure == "atm_straddle":
        bullets.append(
            f"Move-fit is supported by historical/implied ratio {snapshot.historical_vs_implied_move_ratio if snapshot.historical_vs_implied_move_ratio is not None else 0.0:.2f} "
            f"and tail/implied ratio {snapshot.tail_vs_implied_move_ratio if snapshot.tail_vs_implied_move_ratio is not None else 0.0:.2f}."
        )
    else:
        bullets.append(
            f"Tail/implied ratio {snapshot.tail_vs_implied_move_ratio if snapshot.tail_vs_implied_move_ratio is not None else 0.0:.2f} "
            f"and event-risk score {snapshot.event_risk_score if snapshot.event_risk_score is not None else 0.5:.2f} favor wing exposure."
        )
    if recommendation == RECOMMENDATION_WATCH:
        bullets.append("The engine still labels this a watch because the edge is real but not yet separated enough from the alternatives.")
    if recommendation == RECOMMENDATION_NO_TRADE:
        bullets.append("Despite ranking first, the structure does not clear the conservative decision rules.")
    return bullets


def _explain_not_selected(scorecard: StructureScorecard) -> List[str]:
    reasons: List[str] = []
    if not scorecard.eligible:
        reasons.append(f"Ineligible due to: {', '.join(scorecard.eligibility_flags)}.")
        return reasons
    if scorecard.expected_edge_pct <= 0:
        reasons.append("Edge quality is negative or unclear after scorecard penalties.")
    if scorecard.execution_penalty >= WATCH_EXECUTION_PENALTY_THRESHOLD:
        reasons.append(f"Execution penalty is elevated at {scorecard.execution_penalty:.2f}.")
    if scorecard.crowding_penalty >= HIGH_CONFLICT_PENALTY:
        reasons.append(f"Crowding penalty is elevated at {scorecard.crowding_penalty:.2f}.")
    if scorecard.concavity_penalty >= HIGH_CONFLICT_PENALTY:
        reasons.append(f"Concavity penalty is elevated at {scorecard.concavity_penalty:.2f}.")
    if scorecard.walk_forward_history_count < MIN_WALK_FORWARD_HISTORY:
        reasons.append(f"Walk-forward history is thin at {scorecard.walk_forward_history_count} observations.")
    if not reasons:
        reasons.append(
            f"Composite score trails the winner ({scorecard.composite_structure_score:.2f}) without a strong enough edge advantage."
        )
    return reasons
