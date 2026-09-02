"""services.move_statistics — the F1 unit-conversion contract."""
from __future__ import annotations

import math

import numpy as np
import pytest

from services import move_statistics as ms


def test_shape_factors():
    assert ms.SIGMA_TO_EXPECTED_ABS_MOVE == pytest.approx(0.7978845608, abs=1e-9)
    assert ms.SIGMA_TO_P90_ABS_MOVE == pytest.approx(1.6448536270, abs=1e-9)
    assert ms.MEDIAN_TO_MEAN_ABS_MOVE == pytest.approx(0.845348, abs=1e-6)
    assert ms.ANCHOR_RATIO_UNIT_SCALE == pytest.approx(0.754696, abs=1e-6)
    assert ms.TAIL_RATIO_UNIT_SCALE == ms.SIGMA_TO_P90_ABS_MOVE


def test_fairly_priced_event_reads_exactly_one():
    """The whole point: implied σ == realized σ  ⇒  both ratios are 1.0."""
    sigma = 8.0
    mean_abs = sigma * ms.SIGMA_TO_EXPECTED_ABS_MOVE
    median_abs = mean_abs * ms.MEDIAN_TO_MEAN_ABS_MOVE
    w = ms.MOVE_ANCHOR_AVG_LAST4_WEIGHT
    anchor = w * mean_abs + (1 - w) * median_abs            # what _compute_move_anchor builds
    p90 = sigma * ms.SIGMA_TO_P90_ABS_MOVE
    assert ms.historical_vs_implied_ratio(anchor, sigma) == pytest.approx(1.0, abs=1e-12)
    assert ms.tail_vs_implied_ratio(p90, sigma) == pytest.approx(1.0, abs=1e-12)
    # And the legacy computation for the same fair event was NOT 1.0.
    assert anchor / sigma == pytest.approx(ms.ANCHOR_RATIO_UNIT_SCALE, abs=1e-12)
    assert p90 / sigma == pytest.approx(ms.TAIL_RATIO_UNIT_SCALE, abs=1e-12)


def test_corrected_ratio_is_legacy_ratio_over_a_constant():
    rng = np.random.default_rng(7)
    for _ in range(200):
        anchor, p90, sigma = rng.uniform(0.5, 20, 3)
        assert ms.historical_vs_implied_ratio(anchor, sigma) == pytest.approx(
            (anchor / sigma) / ms.ANCHOR_RATIO_UNIT_SCALE, rel=1e-12)
        assert ms.tail_vs_implied_ratio(p90, sigma) == pytest.approx(
            (p90 / sigma) / ms.TAIL_RATIO_UNIT_SCALE, rel=1e-12)


def test_percentile_bounds_rescale_exactly_and_scores_are_invariant():
    """A corpus-calibrated (lo, hi) bound divided by the scale gives a
    byte-identical linear score for the corrected ratio — the mechanism that
    lets the scorecard keep its calibration without a corpus regeneration."""
    from services.structure_scorecard import _score_high_good

    rng = np.random.default_rng(11)
    legacy = rng.uniform(0.01, 3.0, 500)
    for lo, hi, c in ((0.12, 1.14, ms.ANCHOR_RATIO_UNIT_SCALE),
                      (0.24, 1.53, ms.TAIL_RATIO_UNIT_SCALE),
                      (0.42, 1.53, ms.TAIL_RATIO_UNIT_SCALE),
                      (1.22, 1.80, ms.TAIL_RATIO_UNIT_SCALE)):
        for r in legacy:
            assert _score_high_good(r / c, lo / c, hi / c) == pytest.approx(
                _score_high_good(r, lo, hi), abs=1e-12)


def test_none_and_degenerate_inputs():
    assert ms.historical_vs_implied_ratio(None, 5.0) is None
    assert ms.historical_vs_implied_ratio(5.0, None) is None
    assert ms.historical_vs_implied_ratio(5.0, 0.0) is None
    assert ms.tail_vs_implied_ratio(float("nan"), 5.0) is None
    assert ms.sigma_to_expected_abs_move(float("inf")) is None


def test_weight_is_shared_with_edge_constants_registry():
    from web.api.edge_constants import _HEURISTIC_THRESHOLDS

    assert _HEURISTIC_THRESHOLDS["move_anchor_avg_last4_weight"]["value"] == ms.MOVE_ANCHOR_AVG_LAST4_WEIGHT
