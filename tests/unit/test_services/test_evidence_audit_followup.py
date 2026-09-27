"""Audit follow-up (#4-#9): paid-beta gate, degraded surfaces, bootstrap
order-invariance, duplicate pairing keys, non-finite and boolean values."""
from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from services.baseline_evidence_store import BaselineEvidenceStore
from services.evidence_report import (
    _commercialization_gate,
    _uncertainty,
    build_evidence_report,
)
from services.evidence_statistics import (
    mean_with_ci,
    paired_difference,
    summarize_returns,
    two_sample_difference,
)
from services.outcome_recorder import OutcomeStore, finalize_trade_and_update_learning
from services.structure_scorecard import SUPPORTED_STRUCTURES
from services.structure_selector import (
    RECOMMENDATION_BEST,
    RECOMMENDATION_WATCH,
    select_best_structure,
)
from tests.unit.test_services.test_evidence_report_uncertainty import _baseline, _selector
from tests.unit.test_services.test_structure_selector import _scorecard, _snapshot

EXIT = date(2026, 5, 4)
ACTIONABLE = {"Candidate", "Best Candidate"}

MATURE = {
    "edge_quality_label_allowed": True,
    "benchmark_comparison_meaningful": True,
    "bucket_interpretation_allowed": True,
    "maturity_label": "Mature evidence",
}


# ── #4 paid-beta gate ──────────────────────────────────────────────────────────


def test_gate_opens_only_when_every_condition_holds():
    assert _commercialization_gate(active_days=90, selector_n=80, maturity=MATURE)["ready_for_paid_beta"] is True


@pytest.mark.parametrize(
    "flags",
    list(itertools.product([True, False], repeat=3)),
)
def test_gate_is_never_ready_while_claims_are_withheld(flags):
    edge, benchmark, buckets = flags
    maturity = {
        "edge_quality_label_allowed": edge,
        "benchmark_comparison_meaningful": benchmark,
        "bucket_interpretation_allowed": buckets,
        "maturity_label": "Mature evidence",
    }
    # Days and sample far past the minimums must not open the gate by themselves.
    gate = _commercialization_gate(active_days=365, selector_n=1000, maturity=maturity)
    assert gate["ready_for_paid_beta"] is all(flags)
    assert len(gate["blocking_reasons"]) == flags.count(False)


@pytest.mark.parametrize("label", ["Early observational", "Insufficient evidence", None])
def test_gate_blocks_immature_label(label):
    gate = _commercialization_gate(active_days=365, selector_n=1000, maturity={**MATURE, "maturity_label": label})
    assert gate["ready_for_paid_beta"] is False
    assert any("maturity" in reason for reason in gate["blocking_reasons"])


def test_gate_blocks_short_or_small_evidence_even_when_mature():
    short = _commercialization_gate(active_days=59, selector_n=1000, maturity=MATURE)
    small = _commercialization_gate(active_days=365, selector_n=29, maturity=MATURE)
    assert short["ready_for_paid_beta"] is False and small["ready_for_paid_beta"] is False
    assert any("days" in reason for reason in short["blocking_reasons"])
    assert any("resolved" in reason for reason in small["blocking_reasons"])


def test_report_gate_is_closed_without_claimable_evidence(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    for i in range(40):
        _selector(outcomes, i, 5.0)
    report = build_evidence_report(baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), outcome_store=outcomes)
    gate = report["commercialization_gate"]
    assert gate["ready_for_paid_beta"] is False
    assert gate["blocking_reasons"]


# ── #5 degraded surfaces are demoted to Watch ─────────────────────────────────


def _winner_for(structure):
    cards = [_scorecard(structure, expected_edge_pct=6.2, expected_return_pct=9.0,
                        composite_structure_score=0.82, sample_confidence=0.74,
                        walk_forward_history_count=40)]
    cards += [_scorecard(other, expected_edge_pct=1.5, composite_structure_score=0.45)
              for other in SUPPORTED_STRUCTURES if other != structure]
    return cards


@pytest.mark.parametrize("structure", SUPPORTED_STRUCTURES)
def test_degraded_surface_is_never_actionable(structure):
    base = _snapshot(historical_vs_implied_move_ratio=2.2526, tail_vs_implied_move_ratio=1.1855)
    degraded = replace(base, surface_quality_status="degraded_surface")
    output = select_best_structure(degraded, _winner_for(structure))
    assert output.recommendation not in ACTIONABLE
    if output.recommendation == RECOMMENDATION_WATCH:
        assert any("surface is degraded" in bullet for bullet in output.why_this_structure)


def test_degraded_surface_demotes_a_best_candidate():
    base = _snapshot(historical_vs_implied_move_ratio=2.2526, tail_vs_implied_move_ratio=1.1855)
    clean = select_best_structure(replace(base, surface_quality_status="ok"), _winner_for("atm_straddle"))
    degraded = select_best_structure(replace(base, surface_quality_status="degraded_surface"), _winner_for("atm_straddle"))
    assert clean.recommendation == RECOMMENDATION_BEST
    assert degraded.recommendation == RECOMMENDATION_WATCH
    assert degraded.best_structure == clean.best_structure  # demoted, still recorded


@pytest.mark.parametrize("status", [None, "ok", "record_only"])
def test_non_degraded_surface_statuses_are_not_demoted(status):
    base = _snapshot(historical_vs_implied_move_ratio=2.2526, tail_vs_implied_move_ratio=1.1855)
    output = select_best_structure(replace(base, surface_quality_status=status), _winner_for("atm_straddle"))
    assert output.recommendation == RECOMMENDATION_BEST


# ── #6 bootstrap results do not depend on row order ───────────────────────────


def test_statistics_are_invariant_to_row_order():
    rng = random.Random(7)
    values = [rng.gauss(1.0, 5.0) for _ in range(40)]
    other = [rng.gauss(0.0, 5.0) for _ in range(35)]
    reference = (summarize_returns(values), two_sample_difference(values, other))
    for _ in range(5):
        shuffled_a, shuffled_b = values[:], other[:]
        rng.shuffle(shuffled_a)
        rng.shuffle(shuffled_b)
        assert (summarize_returns(shuffled_a), two_sample_difference(shuffled_a, shuffled_b)) == reference


def test_paired_difference_is_invariant_to_key_insertion_order():
    left = {f"k{i}": float(i) for i in range(20)}
    right = {f"k{i}": float(i % 4) for i in range(20)}
    reversed_left = dict(reversed(list(left.items())))
    assert paired_difference(left, right) == paired_difference(reversed_left, right)


# ── #7 duplicate pairing keys are excluded, not overwritten ───────────────────


def _row(rec, ret, **extra):
    return {"recommendation_id": rec, "realized_return_pct": ret, **extra}


def test_duplicate_selector_key_is_excluded_and_reported():
    selected = [_row(f"r{i}", 5.0) for i in range(12)] + [_row("r0", -500.0)]
    baselines = [_row(f"r{i}", 1.0, baseline_name="always_atm_straddle") for i in range(12)]
    paired = _uncertainty(selected, baselines, [])["selector_minus_baseline"]["always_atm_straddle"]
    assert paired["pairs"] == 11
    assert paired["mean"] == 4.0
    assert paired["excluded_duplicate_keys"] == ["r0"]


def test_duplicate_baseline_key_result_does_not_depend_on_order():
    selected = [_row(f"r{i}", 5.0) for i in range(12)]
    baselines = [_row(f"r{i}", 1.0, baseline_name="b") for i in range(12)] + [_row("r3", 90.0, baseline_name="b")]
    forward = _uncertainty(selected, baselines, [])["selector_minus_baseline"]["b"]
    backward = _uncertainty(selected, list(reversed(baselines)), [])["selector_minus_baseline"]["b"]
    assert forward == backward
    assert forward["excluded_duplicate_keys"] == ["r3"]
    assert forward["pairs"] == 11


# ── #8 non-finite values: refused at writes, sanitized in reports ─────────────


BAD_VALUES = [float("nan"), float("inf"), float("-inf"), True, np.bool_(False), "abc"]


@pytest.mark.parametrize("bad", BAD_VALUES, ids=repr)
@pytest.mark.parametrize("field", ["realized_return_pct", "realized_expansion_pct", "realized_pnl", "exit_mid"])
def test_update_exit_refuses_non_finite_values(tmp_path, field, bad):
    store = OutcomeStore(tmp_path / "o.sqlite")
    store.insert_entry(trade_id="T", symbol="S", structure="otm_strangle", entry_date=EXIT - timedelta(days=5),
                       setup_score=0.6, source_type="paper")
    values = {"exit_mid": 1.0, "realized_return_pct": 1.0, "realized_expansion_pct": 1.0, "realized_pnl": 1.0, field: bad}
    with pytest.raises(ValueError, match=field):
        store.update_exit(trade_id="T", exit_date=EXIT, **values)
    row = store.get_trade("T")
    assert row["status"] == "open" and row["realized_return_pct"] is None


@pytest.mark.parametrize("bad", BAD_VALUES[:4], ids=repr)
def test_finalize_refuses_non_finite_before_claiming(tmp_path, bad):
    store = OutcomeStore(tmp_path / "o.sqlite")
    store.insert_entry(trade_id="T", symbol="S", structure="otm_strangle", entry_date=EXIT - timedelta(days=5),
                       setup_score=0.6, source_type="paper")
    with pytest.raises(ValueError, match="realized_return_pct"):
        finalize_trade_and_update_learning(trade_id="T", exit_date=EXIT, realized_return_pct=bad,
                                           realized_expansion_pct=1.0, store=store)
    row = store.get_trade("T")
    assert row["status"] == "open"
    assert row.get("finalizing_owner") is None


def test_update_exit_accepts_numpy_and_int_values(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    store.insert_entry(trade_id="T", symbol="S", structure="otm_strangle", entry_date=EXIT - timedelta(days=5),
                       setup_score=0.6, source_type="paper")
    assert store.update_exit(trade_id="T", exit_date=EXIT, exit_mid=np.float64(2.0),
                             realized_return_pct=3, realized_expansion_pct=np.float32(1.5))
    assert store.get_trade("T")["realized_return_pct"] == 3.0


def _force_return(store, trade_id, value):
    # Simulates a legacy row written before the write boundary refused NaN/inf.
    with store._conn:
        store._conn.execute("UPDATE outcome_trades SET realized_return_pct = ? WHERE trade_id = ?", (value, trade_id))


def test_report_excludes_and_counts_legacy_non_finite_rows_and_is_strict_json(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    for i in range(12):
        _selector(outcomes, i, 2.0)
        _baseline(baselines, rec=f"rec-{i}", ret=1.0)
    _selector(outcomes, 50, 1.0)
    _force_return(outcomes, "T50", float("inf"))
    _selector(outcomes, 51, 1.0)
    _force_return(outcomes, "T51", float("-inf"))

    report = build_evidence_report(baseline_store=baselines, outcome_store=outcomes)

    assert report["selector_summary"]["n"] == 12
    assert report["selector_summary"]["avg_realized_return_pct"] == 2.0
    assert report["uncertainty"]["selector"]["n"] == 12
    assert report["non_finite_outcomes"]["selector_n"] == 2
    json.dumps(report, allow_nan=False)  # raises on any NaN/inf left anywhere


def test_report_sanitizes_nested_non_finite_values():
    from services.evidence_report import _json_safe

    payload = {"a": [1.0, float("nan"), {"b": np.float64("inf")}], "c": (float("-inf"),), "d": "x", "e": 3}
    assert _json_safe(payload) == {"a": [1.0, None, {"b": None}], "c": [None], "d": "x", "e": 3}


# ── #9 booleans are not returns ───────────────────────────────────────────────


def test_statistics_ignore_booleans():
    values = [1.0] * 10 + [True, np.bool_(True), False]
    result = mean_with_ci(values)
    assert result["n"] == 10
    assert result["mean"] == 1.0
    assert two_sample_difference([True] * 12, [0.0] * 12)["n_first"] == 0
    assert paired_difference({"a": True}, {"a": 0.0})["pairs"] == 0
    assert all(math.isfinite(v) for v in (result["ci_low"], result["ci_high"]))
