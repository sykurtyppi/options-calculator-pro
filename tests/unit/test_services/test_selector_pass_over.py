"""The selector decides on the best structure that clears the hard gates.

It used to rank by composite score and apply the no-trade gates to the winner
only: a winner with negative edge made the event No Trade even when a lower
ranked structure cleared every gate (composite 0.8 / edge -1% beside 0.7 /
+5% was No Trade; the +5% structure alone was Best Candidate). Now a
structure failing only economic gates (PASS_OVER_GATES) is passed over and
recorded; broken inputs and snapshot-level gates still fail closed, and when
no structure clears the gates the abstention is unchanged.
"""
from __future__ import annotations

import json
import math

import pytest

from services.structure_selector import (
    GATE_DATA_QUALITY_BELOW_FLOOR,
    GATE_EXECUTION_PENALTY_HIGH,
    GATE_NON_FINITE_EXPECTED_RETURN,
    GATE_NON_POSITIVE_EDGE,
    GATE_WATCH_SCORE_GAP_SMALL,
    GATE_WEAK_WALK_FORWARD_HISTORY,
    PASS_OVER_GATES,
    RECOMMENDATION_BEST,
    RECOMMENDATION_NO_TRADE,
    RECOMMENDATION_WATCH,
    select_best_structure,
)
from tests.unit.test_services.test_structure_selector import _scorecard, _snapshot

STRONG = dict(execution_penalty=0.01, walk_forward_history_count=40, sample_confidence=0.80)


def _a(**overrides):
    return _scorecard("atm_straddle", **{**STRONG, "expected_edge_pct": -1.0, "composite_structure_score": 0.80,
                                         **overrides})


def _b(**overrides):
    return _scorecard("otm_strangle", **{**STRONG, "expected_edge_pct": 5.0, "composite_structure_score": 0.70,
                                         **overrides})


def _decision(out):
    return (out.recommendation, out.best_structure, out.binding_gates, out.expected_edge_pct,
            out.expected_return_pct, round(out.confidence_pct, 9))


def test_a_rejected_winner_no_longer_hides_an_acceptable_runner_up() -> None:
    together = select_best_structure(_snapshot(), [_a(), _b()])
    alone = select_best_structure(_snapshot(), [_b()])

    assert alone.recommendation == RECOMMENDATION_BEST
    # The structure it passed over changes nothing about the decision.
    assert _decision(together) == _decision(alone)
    assert together.passed_over_structures == {"atm_straddle": [GATE_NON_POSITIVE_EDGE]}
    assert together.runner_up_structures == []  # a rejected structure is not a runner-up
    assert any("atm_straddle ranked higher" in line for line in together.why_this_structure)
    assert "atm_straddle" in together.why_not_others and "otm_strangle" not in together.why_not_others


def test_card_order_does_not_matter() -> None:
    assert _decision(select_best_structure(_snapshot(), [_a(), _b()])) == \
        _decision(select_best_structure(_snapshot(), [_b(), _a()]))


def test_an_execution_penalty_rejection_is_passed_over_too() -> None:
    out = select_best_structure(_snapshot(), [_a(expected_edge_pct=6.0, execution_penalty=0.12), _b()])
    assert out.best_structure == "otm_strangle"
    assert out.passed_over_structures["atm_straddle"][0] == GATE_EXECUTION_PENALTY_HIGH


def test_several_rejected_structures_are_passed_over_in_rank_order() -> None:
    worse = _scorecard("call_calendar", **{**STRONG, "expected_edge_pct": -0.5, "composite_structure_score": 0.75})
    out = select_best_structure(_snapshot(), [_a(), worse, _b()])
    assert out.best_structure == "otm_strangle"
    assert list(out.passed_over_structures) == ["atm_straddle", "call_calendar"]


def test_when_no_structure_clears_the_gates_the_abstention_is_as_before() -> None:
    out = select_best_structure(_snapshot(), [_a(), _b(expected_edge_pct=-2.0)])
    assert out.recommendation == RECOMMENDATION_NO_TRADE and out.best_structure is None
    # The binding gates are the top-ranked structure's, exactly as recorded before.
    assert out.binding_gate == GATE_NON_POSITIVE_EDGE
    assert out.passed_over_structures == {}
    assert out.runner_up_structures == ["otm_strangle"]


def test_a_broken_card_fails_closed_and_is_never_skipped() -> None:
    out = select_best_structure(_snapshot(), [_a(expected_edge_pct=6.0, expected_return_pct=float("nan")), _b()])
    assert out.recommendation == RECOMMENDATION_NO_TRADE
    assert out.binding_gate == GATE_NON_FINITE_EXPECTED_RETURN and out.passed_over_structures == {}


def test_a_broken_card_with_an_economic_failure_still_fails_closed() -> None:
    out = select_best_structure(_snapshot(), [_a(expected_return_pct=float("inf")), _b()])
    assert out.recommendation == RECOMMENDATION_NO_TRADE
    assert GATE_NON_FINITE_EXPECTED_RETURN in out.binding_gates and out.passed_over_structures == {}


def test_a_snapshot_gate_stops_every_structure() -> None:
    out = select_best_structure(_snapshot(data_quality_score=0.30), [_a(), _b()])
    assert out.recommendation == RECOMMENDATION_NO_TRADE
    assert GATE_DATA_QUALITY_BELOW_FLOOR in out.binding_gates and out.passed_over_structures == {}


def test_decision_gates_still_apply_to_the_structure_chosen() -> None:
    weak = dict(walk_forward_history_count=3, sample_confidence=0.80, execution_penalty=0.01)
    low = _scorecard("call_calendar", **{**weak, "expected_edge_pct": 2.0, "composite_structure_score": 0.66})
    out = select_best_structure(_snapshot(), [_a(), _b(**weak), low])
    assert out.recommendation == RECOMMENDATION_NO_TRADE
    assert out.binding_gates == [GATE_WEAK_WALK_FORWARD_HISTORY]
    assert out.passed_over_structures == {"atm_straddle": [GATE_NON_POSITIVE_EDGE]}


def test_a_rejected_structure_ranked_below_still_counts_against_the_gap() -> None:
    # Unchanged rule: only structures ranked ABOVE the chosen one are passed
    # over. A close rejected runner-up still makes the call ambiguous.
    out = select_best_structure(_snapshot(), [_b(composite_structure_score=0.80),
                                              _a(composite_structure_score=0.78)])
    assert (out.recommendation, out.best_structure) == (RECOMMENDATION_WATCH, "otm_strangle")
    assert GATE_WATCH_SCORE_GAP_SMALL in out.binding_gates and out.passed_over_structures == {}


def test_only_economic_gates_allow_a_pass_over() -> None:
    assert PASS_OVER_GATES == {"non_positive_edge", "execution_penalty_high", "execution_cost_dominates_edge"}


def test_the_record_serializes_as_strict_json() -> None:
    payload = select_best_structure(_snapshot(), [_a(), _b()]).to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False, default=str))["passed_over_structures"] == {
        "atm_straddle": [GATE_NON_POSITIVE_EDGE]}
    assert all(math.isfinite(payload[k]) for k in ("confidence_pct", "expected_edge_pct", "expected_return_pct"))


@pytest.mark.parametrize("edge", [0.0, -0.01])
def test_zero_edge_is_rejected_like_negative_edge(edge) -> None:
    out = select_best_structure(_snapshot(), [_a(expected_edge_pct=edge), _b()])
    assert out.best_structure == "otm_strangle"
