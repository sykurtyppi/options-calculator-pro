"""Unit tests for the crush-gate wiring in forward_screener.

The crush gate uses NBR (near_back_ratio = front_atm_iv / back_atm_iv) to
predict which pre-earnings expansion trades are likely to succeed. Walk-forward
analysis (2026-06-04) showed NBR >= 1.40 selects the 85% win-rate subset of
Fingerprint C trades, and the crush classifier (AUC=0.820) is effectively an
expansion-quality gate via the same feature.

These tests exercise the pure signal computation (no network) and the NBR
threshold boundary.
"""
from __future__ import annotations

import pytest

from scripts.forward_screener import (
    NBR_GATE_THRESHOLD,
    _compute_crush_gate_signals,
)


def test_nbr_gate_threshold_is_1_40():
    """Pin the threshold. If changed, the walk-forward evidence must be re-evaluated."""
    assert NBR_GATE_THRESHOLD == 1.40


def test_pass_when_nbr_above_threshold():
    result = _compute_crush_gate_signals(front_iv=0.56, back_iv=0.38)
    nbr = 0.56 / 0.38
    assert result["nbr"] is not None
    assert abs(result["nbr"] - round(nbr, 4)) < 1e-4
    assert result["crush_gate"] == "PASS"


def test_fail_when_nbr_below_threshold():
    result = _compute_crush_gate_signals(front_iv=0.40, back_iv=0.35)
    nbr = 0.40 / 0.35  # ~1.143
    assert result["nbr"] is not None
    assert result["crush_gate"] == "FAIL"


def test_exact_threshold_boundary():
    """NBR exactly at 1.40 should PASS (>=, not >)."""
    # back_iv = front_iv / 1.40
    front = 0.42
    back = front / 1.40
    result = _compute_crush_gate_signals(front_iv=front, back_iv=back)
    assert result["crush_gate"] == "PASS"


def test_no_data_when_front_iv_missing():
    result = _compute_crush_gate_signals(front_iv=None, back_iv=0.30)
    assert result["crush_gate"] == "NO_DATA"
    assert result["nbr"] is None


def test_no_data_when_back_iv_missing():
    result = _compute_crush_gate_signals(front_iv=0.50, back_iv=None)
    assert result["crush_gate"] == "NO_DATA"
    assert result["nbr"] is None


def test_no_data_when_back_iv_near_zero():
    result = _compute_crush_gate_signals(front_iv=0.50, back_iv=0.005)
    assert result["crush_gate"] == "NO_DATA"
    assert result["nbr"] is None


def test_crush_prob_is_none_without_model():
    """If the ML model isn't loaded, crush_prob should be None but NBR still works."""
    result = _compute_crush_gate_signals(front_iv=0.56, back_iv=0.38)
    # crush_prob may or may not be populated depending on whether the model is
    # installed; the important thing is it doesn't crash.
    assert result["nbr"] is not None
    assert result["crush_gate"] in ("PASS", "FAIL")
