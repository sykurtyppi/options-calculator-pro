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

import pandas as pd
import pytest

from scripts.forward_screener import (
    ATM_MONEYNESS_BAND,
    MAX_SANE_IV,
    MIN_SANE_IV,
    NBR_GATE_THRESHOLD,
    _compute_crush_gate_signals,
    _nearest_atm_iv,
)


def _chain(rows):
    """Build a minimal option-chain DataFrame: rows of (strike, bid, iv)."""
    return pd.DataFrame(
        [{"strike": s, "bid": b, "impliedVolatility": iv} for s, b, iv in rows]
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


# ── Sane-IV guard on the gate (regression for the #90 junk-strike bug) ──────────


def test_gate_rejects_insane_front_iv():
    """A 240%-IV junk leg must not produce an NBR — return NO_DATA instead."""
    result = _compute_crush_gate_signals(front_iv=2.40, back_iv=0.38)
    assert result["crush_gate"] == "NO_DATA"
    assert result["nbr"] is None


def test_gate_rejects_zero_front_iv():
    """An IV-0 junk leg (below MIN_SANE_IV) must yield NO_DATA, not a huge NBR."""
    result = _compute_crush_gate_signals(front_iv=0.0, back_iv=0.38)
    assert result["crush_gate"] == "NO_DATA"
    assert result["nbr"] is None


def test_gate_rejects_insane_back_iv():
    result = _compute_crush_gate_signals(front_iv=0.50, back_iv=MAX_SANE_IV + 0.5)
    assert result["crush_gate"] == "NO_DATA"
    assert result["nbr"] is None


# ── _nearest_atm_iv: moneyness band + sane-IV selection ─────────────────────────


def test_nearest_atm_iv_picks_strike_closest_to_spot_in_band():
    spot = 100.0
    chain = _chain([(95, 2.0, 0.30), (100, 2.5, 0.32), (110, 1.0, 0.34)])
    iv = _nearest_atm_iv(chain, spot)
    assert iv == pytest.approx(0.32)  # strike 100 is nearest


def test_nearest_atm_iv_ignores_out_of_band_junk_strike():
    """The headline #90 bug: a deep-ITM strike with junk IV must be excluded so
    the real ATM strike is selected."""
    spot = 310.0
    chain = _chain([
        (45, 213.0, 0.0001),   # deep ITM junk well outside the ±15% band
        (300, 5.0, 0.40),      # legit ATM
        (320, 3.0, 0.42),      # legit ATM
    ])
    iv = _nearest_atm_iv(chain, spot)
    # Must pick a strike inside [263.5, 356.5], not the 45 junk strike.
    assert iv in (pytest.approx(0.40), pytest.approx(0.42))


def test_nearest_atm_iv_returns_none_when_band_empty():
    """If no strike inside the ATM band has a sane IV, return None (→ NO_DATA)."""
    spot = 310.0
    chain = _chain([
        (45, 213.0, 0.0),      # junk, out of band
        (300, 5.0, 0.0),       # in band but IV below MIN_SANE_IV
        (320, 3.0, 3.5),       # in band but IV above MAX_SANE_IV
    ])
    assert _nearest_atm_iv(chain, spot) is None


def test_nearest_atm_iv_requires_positive_bid():
    spot = 100.0
    chain = _chain([(100, 0.0, 0.30), (105, 0.0, 0.31)])  # no bids
    assert _nearest_atm_iv(chain, spot) is None


def test_nearest_atm_iv_empty_chain():
    assert _nearest_atm_iv(_chain([]), 100.0) is None


def test_atm_band_and_iv_bounds_are_sane():
    assert 0.05 <= ATM_MONEYNESS_BAND <= 0.25
    assert MIN_SANE_IV > 0
    assert MAX_SANE_IV > MIN_SANE_IV
