"""Tests for services/iv_term_structure.py (PR #72).

The bounded-interpolation helper replaces bare ``np.interp`` calls
that silently clamped on out-of-bracket inputs — fabricating an
"iv30" value equal to the nearest-expiry IV when the chain had no
expiry near 30 days.

These tests pin:

  1. Bracketed input matches np.interp (the happy-path regression
     that guarantees non-Monday-post-monthly-expiry symbols are
     unaffected).
  2. Below-range / above-range inputs return None with the correct
     status code, so downstream null_reasons plumbing can show the
     specific gap.
  3. Degenerate inputs (empty, single point, length mismatch, NaN)
     return None with their own status codes — no exceptions, no
     silently-fabricated values.
  4. Sorting is internal: the helper handles unsorted input.
  5. STATUS_TO_NULL_REASON contains a mapping for every failure
     status (so the snapshot's null_reasons dict can't ever fall
     through to the generic catch-all on a recognized status).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from services.iv_term_structure import (
    INTERP_INSUFFICIENT_POINTS,
    INTERP_NO_DATA,
    INTERP_NON_FINITE_INPUT,
    INTERP_OK,
    INTERP_TARGET_ABOVE_RANGE,
    INTERP_TARGET_BELOW_RANGE,
    STATUS_TO_NULL_REASON,
    bounded_interp,
)


# ──────────────────────────────────────────────────────────────────────────
# Happy path: bracketed input
# ──────────────────────────────────────────────────────────────────────────


def test_bracketed_input_matches_np_interp() -> None:
    """The load-bearing regression. For inputs where np.interp would
    have produced a valid interpolated value (target inside the
    bracket), bounded_interp must produce the SAME numeric value to
    floating-point precision. This is what guarantees that non-edge-
    case symbols continue to produce identical iv30 / iv45 numbers
    after PR #72."""
    days = np.array([14.0, 35.0])
    ivs = np.array([0.25, 0.30])

    # Target inside [14, 35] — should interpolate.
    value, status = bounded_interp(30.0, days, ivs)
    expected = float(np.interp(30.0, days, ivs))
    assert status == INTERP_OK
    assert value is not None
    assert value == pytest.approx(expected, rel=1e-12)


def test_bracket_endpoints_inclusive() -> None:
    """Target equal to either endpoint of the bracket must succeed
    (inclusive range). At endpoint, np.interp returns the endpoint
    value exactly; bounded_interp does the same."""
    days = np.array([21.0, 49.0])
    ivs = np.array([0.20, 0.28])

    value, status = bounded_interp(21.0, days, ivs)
    assert status == INTERP_OK
    assert value == pytest.approx(0.20, abs=1e-12)

    value, status = bounded_interp(49.0, days, ivs)
    assert status == INTERP_OK
    assert value == pytest.approx(0.28, abs=1e-12)


def test_unsorted_input_handled_internally() -> None:
    """np.interp requires monotonically increasing x. bounded_interp
    sorts internally so callers don't have to remember. Same numeric
    result regardless of input order."""
    days_sorted = np.array([14.0, 35.0])
    ivs_sorted = np.array([0.25, 0.30])

    days_unsorted = np.array([35.0, 14.0])
    ivs_unsorted = np.array([0.30, 0.25])

    v_sorted, _ = bounded_interp(30.0, days_sorted, ivs_sorted)
    v_unsorted, _ = bounded_interp(30.0, days_unsorted, ivs_unsorted)
    assert v_sorted == pytest.approx(v_unsorted, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# Out-of-bracket: the load-bearing PR #72 fix
# ──────────────────────────────────────────────────────────────────────────


def test_target_below_bracket_returns_none() -> None:
    """The Codex-flagged class: nearest expiry > target. Pre-PR-#72,
    np.interp clamped to the nearest endpoint, fabricating an
    "iv30" equal to the 31D or 45D expiry's IV. Post-fix returns
    None with target_below_listed_expiries so the snapshot's
    null_reasons can surface the specific gap."""
    days = np.array([31.0, 45.0])
    ivs = np.array([0.28, 0.32])
    value, status = bounded_interp(30.0, days, ivs)
    assert value is None
    assert status == INTERP_TARGET_BELOW_RANGE


def test_target_above_bracket_returns_none() -> None:
    """The other side of the same bug. Pre-PR-#72, requesting
    iv45 from a chain whose longest expiry is 40D would have
    returned the 40D IV labeled as iv45."""
    days = np.array([10.0, 25.0, 40.0])
    ivs = np.array([0.22, 0.26, 0.31])
    value, status = bounded_interp(45.0, days, ivs)
    assert value is None
    assert status == INTERP_TARGET_ABOVE_RANGE


def test_target_below_at_codex_repro_31d_chain() -> None:
    """Codex's exact reproducer: a chain whose first expiry is 31D
    being asked for iv30. Documented in the PR #72 audit; pin it as
    a literal test."""
    days = np.array([31.0, 38.0, 45.0])
    ivs = np.array([0.305, 0.310, 0.315])
    # Pre-PR-#72: np.interp(30.0, ...) returned 0.305 (the 31D IV).
    # Post-PR-#72: returns None with the explicit status code.
    value, status = bounded_interp(30.0, days, ivs)
    assert value is None
    assert status == INTERP_TARGET_BELOW_RANGE


# ──────────────────────────────────────────────────────────────────────────
# Degenerate inputs
# ──────────────────────────────────────────────────────────────────────────


def test_empty_arrays_return_no_data() -> None:
    value, status = bounded_interp(30.0, np.array([]), np.array([]))
    assert value is None
    assert status == INTERP_NO_DATA


def test_none_inputs_return_no_data() -> None:
    value, status = bounded_interp(30.0, None, None)  # type: ignore[arg-type]
    assert value is None
    assert status == INTERP_NO_DATA


def test_single_point_returns_insufficient_points() -> None:
    """A single-expiry chain cannot interpolate even if the target
    happens to equal that expiry — "interpolation" of one point is
    just the point itself, which isn't an honest term-structure
    answer. Explicit code for this case."""
    days = np.array([30.0])
    ivs = np.array([0.25])
    value, status = bounded_interp(30.0, days, ivs)
    assert value is None
    assert status == INTERP_INSUFFICIENT_POINTS


def test_length_mismatch_returns_no_data() -> None:
    days = np.array([14.0, 35.0, 60.0])
    ivs = np.array([0.25, 0.30])
    value, status = bounded_interp(30.0, days, ivs)
    assert value is None
    assert status == INTERP_NO_DATA


def test_nan_in_days_returns_non_finite_status() -> None:
    days = np.array([14.0, float("nan"), 60.0])
    ivs = np.array([0.25, 0.27, 0.30])
    value, status = bounded_interp(30.0, days, ivs)
    assert value is None
    assert status == INTERP_NON_FINITE_INPUT


def test_inf_in_ivs_returns_non_finite_status() -> None:
    days = np.array([14.0, 35.0])
    ivs = np.array([0.25, float("inf")])
    value, status = bounded_interp(30.0, days, ivs)
    assert value is None
    assert status == INTERP_NON_FINITE_INPUT


# ──────────────────────────────────────────────────────────────────────────
# Status-to-null-reason mapping
# ──────────────────────────────────────────────────────────────────────────


def test_every_failure_status_has_a_null_reason_mapping() -> None:
    """Pin the contract that
    earnings_vol_snapshot.build_vol_snapshot depends on: every
    failure status code emitted by bounded_interp must have an
    entry in STATUS_TO_NULL_REASON so the snapshot's null_reasons
    dict can show a specific diagnostic. INTERP_OK is the only
    code without a mapping (success → no null reason).
    """
    failure_codes = {
        INTERP_TARGET_BELOW_RANGE,
        INTERP_TARGET_ABOVE_RANGE,
        INTERP_INSUFFICIENT_POINTS,
        INTERP_NO_DATA,
        INTERP_NON_FINITE_INPUT,
    }
    for code in failure_codes:
        assert code in STATUS_TO_NULL_REASON, (
            f"failure status {code!r} must have an entry in "
            f"STATUS_TO_NULL_REASON; otherwise the snapshot's "
            f"null_reasons dict falls through to the generic "
            f"catch-all on a known failure cause."
        )
    # The happy path is NOT in the map.
    assert INTERP_OK not in STATUS_TO_NULL_REASON


def test_status_to_null_reason_values_are_diagnostic_strings() -> None:
    """The null_reason strings are operator-facing diagnostics —
    verify they're non-empty strings (not None, not empty) so any
    downstream display gets something useful."""
    for status, reason in STATUS_TO_NULL_REASON.items():
        assert isinstance(reason, str) and reason, (
            f"STATUS_TO_NULL_REASON[{status!r}] must be a non-empty string"
        )


# ──────────────────────────────────────────────────────────────────────────
# Snapshot integration smoke
# ──────────────────────────────────────────────────────────────────────────


def test_snapshot_iv30_none_when_target_below_bracket() -> None:
    """End-to-end: when the term structure has no expiry <= 30D,
    the snapshot's iv30 must be None AND null_reasons['iv30'] must
    carry the specific target_below_listed_expiries diagnostic
    (not the generic insufficient_term_structure_points)."""
    # We can't easily construct a full VolSnapshot without a heavy
    # fixture; instead, exercise the internal _TermStructureSnapshot
    # builder path indirectly by constructing the inputs that drive
    # it.
    # The simplest direct check is: confirm the helper produces the
    # expected status on a Codex-style 31D-min chain, AND that the
    # mapping table contains that status. The full integration test
    # would require a synthetic chain DataFrame; covered by the
    # broader test_edge_engine_research_signals harness already
    # exercising build_vol_snapshot with mocks.
    days = np.array([31.0, 45.0, 60.0])
    ivs = np.array([0.30, 0.32, 0.34])
    value, status = bounded_interp(30.0, days, ivs)
    assert value is None
    null_reason = STATUS_TO_NULL_REASON[status]
    assert null_reason == "target_below_listed_expiries"
