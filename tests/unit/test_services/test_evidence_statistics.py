"""Bootstrap intervals and outlier checks for forward evidence."""
from __future__ import annotations

import math

import numpy as np
import pytest

from services.evidence_statistics import (
    MIN_CI_SAMPLE,
    mean_with_ci,
    outlier_dependence,
    paired_difference,
    summarize_returns,
    two_sample_difference,
)


def test_no_interval_below_minimum_sample():
    # The live sample today: MU, NFLX, NVDA.
    result = mean_with_ci([8.96, -23.81, -10.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(-8.2833, abs=1e-4)
    assert (result["ci_low"], result["ci_high"]) == (None, None)
    assert result["verdict"] == "insufficient_sample"


def test_interval_is_deterministic_and_contains_the_mean():
    values = [5, -3, 12, 1, -8, 4, 7, -2, 9, 0, 3, -1]
    first, second = mean_with_ci(values), mean_with_ci(values)
    assert first == second
    assert first["ci_low"] < first["mean"] < first["ci_high"]


def test_interval_width_matches_normal_theory():
    rng = np.random.default_rng(1)
    values = rng.normal(1.0, 1.0, 400)
    result = mean_with_ci(values)
    half_width = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    assert result["ci_high"] - result["ci_low"] == pytest.approx(2 * half_width, rel=0.15)
    assert result["verdict"] == "above_zero"


@pytest.mark.parametrize(
    ("values", "verdict"),
    [
        ([10.0 + i % 3 for i in range(12)], "above_zero"),
        ([-10.0 - i % 3 for i in range(12)], "below_zero"),
        ([(-1) ** i * 5.0 for i in range(12)], "includes_zero"),
    ],
)
def test_verdicts(values, verdict):
    assert mean_with_ci(values)["verdict"] == verdict


def test_non_numeric_and_non_finite_values_are_dropped():
    result = mean_with_ci([1.0, None, "x", float("nan"), float("inf"), 3.0])
    assert result["n"] == 2
    assert result["mean"] == 2.0


def test_outlier_dependence_flags_a_result_carried_by_five_events():
    # Hermes's historical calendar case: five events carried ~85% of profit.
    values = [40.0] * 5 + [1.0] * 10 + [-2.0] * 41
    result = outlier_dependence(values)
    assert np.mean(values) > 0
    assert result["fragile"] is True
    assert result["mean_without_top_k"] < 0
    assert result["top_k_share_of_profit"] == pytest.approx(200 / 210)


def test_outlier_dependence_is_not_fragile_for_broad_gains():
    result = outlier_dependence([2.0] * 30 + [-1.0] * 10)
    assert result["fragile"] is False
    assert result["mean_without_top_k"] > 0


def test_outlier_dependence_needs_more_than_k_values():
    assert outlier_dependence([1, 2, 3])["fragile"] is None


def test_paired_difference_uses_only_shared_events():
    selector = {f"e{i}": 5.0 + i for i in range(12)} | {"only_selector": 100.0, "bad": None}
    baseline = {f"e{i}": 2.0 + i for i in range(12)} | {"only_baseline": -100.0, "bad": 1.0}
    result = paired_difference(selector, baseline)
    assert result["pairs"] == 12
    assert result["mean"] == pytest.approx(3.0)
    # Every pair differs by exactly 3: the interval collapses onto it.
    assert result["ci_low"] == pytest.approx(3.0)
    assert result["ci_high"] == pytest.approx(3.0)
    assert result["verdict"] == "above_zero"


def test_pairing_removes_shared_event_noise():
    rng = np.random.default_rng(7)
    shared = rng.normal(0, 20, 40)  # the market move both positions saw
    selector = {str(i): shared[i] + 1.0 for i in range(40)}
    baseline = {str(i): shared[i] for i in range(40)}
    paired = paired_difference(selector, baseline)
    unpaired = two_sample_difference(list(selector.values()), list(baseline.values()))
    assert paired["verdict"] == "above_zero"
    assert unpaired["verdict"] == "includes_zero"


def test_two_sample_difference_needs_both_groups():
    result = two_sample_difference([1.0] * 20, [0.0] * 5)
    assert result["mean"] == 1.0
    assert result["ci_low"] is None
    assert result["verdict"] == "insufficient_sample"
    full = two_sample_difference([1.0 + (i % 2) for i in range(20)], [-(i % 3) for i in range(20)])
    assert full["verdict"] == "above_zero"
    assert full["n_first"] == full["n_second"] == 20


def test_summary_combines_interval_and_outliers():
    summary = summarize_returns(range(MIN_CI_SAMPLE))
    assert summary["n"] == MIN_CI_SAMPLE
    assert summary["ci_low"] is not None
    assert summary["outlier_dependence"]["k"] == 5
