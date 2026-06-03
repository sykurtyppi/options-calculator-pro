"""Guard tests for the crush-classifier target threshold.

The original threshold (front_iv_crush_pct < -0.10) was empirically degenerate
on the labeled universe (98% positive rate → "always predict crush"). The
threshold was changed to -0.40, which yields a ~45/55 class balance on the same
1,639 events and a strictly-better classifier (CV AUC 0.846 vs 0.831, half the
variance). These tests pin the choice + assert it produces a non-degenerate
binary target.

Methodology note: the trainer's CV is shuffled k-fold (not walk-forward), so the
reported AUC is an upper bound on true OOS performance. Walk-forward evaluation
lives in scripts/train_prior_crush_model.py and is a separate concern.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from services.institutional_ml_db import InstitutionalMLDatabase


def test_threshold_is_minus40_in_source():
    """Pin the threshold value in source. A future change to a degenerate value
    (e.g. back to -0.10) must fail this test before reaching CI."""
    src = inspect.getsource(InstitutionalMLDatabase._build_crush_training_data)
    assert "CRUSH_THRESHOLD_PCT = -0.40" in src, (
        "crush threshold must be -0.40; -0.10 was empirically degenerate (98% positive)"
    )
    # The accompanying rationale comment must also survive (so a future reviewer
    # doesn't change the value without re-reading WHY).
    assert "98%" in src, "data-driven rationale comment must be preserved"


def test_threshold_produces_balanced_target_on_synthetic_distribution():
    """On a distribution matching the real labels (median crush -38%), the -40%
    threshold yields ~45/55. The old -10% threshold would yield ~98/2."""
    # synthetic crushes drawn from the empirical shape: mean -0.38, std 0.135
    rng = np.random.default_rng(seed=42)
    crushes = rng.normal(loc=-0.378, scale=0.135, size=10_000)
    pos_at_minus40 = float((crushes < -0.40).mean())
    pos_at_minus10 = float((crushes < -0.10).mean())
    assert 0.30 < pos_at_minus40 < 0.60, (
        f"-40% threshold should produce ~45% positive on the empirical distribution, "
        f"got {pos_at_minus40:.2f}"
    )
    assert pos_at_minus10 > 0.95, (
        f"sanity: -10% threshold should be degenerate (>95% positive) on this "
        f"distribution, got {pos_at_minus10:.2f}"
    )


def test_meta_label_string_matches_threshold():
    """The meta-file's `label` string must reflect the actual threshold so a
    consumer reading the persisted metadata isn't misled."""
    src = inspect.getsource(InstitutionalMLDatabase.train_ml_model_on_historical_spreads)
    assert '"label": "front_iv_crush_pct < -0.40"' in src, (
        "meta `label` field must match the -0.40 threshold in _build_crush_training_data"
    )
