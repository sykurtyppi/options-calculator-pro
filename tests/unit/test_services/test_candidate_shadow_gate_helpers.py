"""Tests for the scenario-aware gate helpers in
services/candidate_shadow_provenance.py (PR #66 visibility infrastructure).

The helpers under test:

  * ``candidate_return_for_gate(outcome, scenario, *, allow_mid_fallback=False)``
  * ``is_promotion_eligible_for_scenario(record, scenario, *, allow_mid_fallback=False)``
  * The ``conservative -> cross_50`` alias mapping.

These tests pin the PR #66 invariants the Codex review explicitly
called out:

  1. ``is_promotion_eligible()`` default behavior is byte-for-byte
     identical to pre-PR-#66.
  2. ``allow_mid_fallback`` defaults to ``False`` for every public
     entry point.
  3. Missing scenario values reduce ``scenario_priced_events`` —
     they are NEVER counted as priced via the mid fallback unless
     the caller explicitly opts in.
  4. ``conservative`` resolves to ``cross_50`` so it never becomes
     an independent fourth evidence dimension.

These invariants are operationally important: they prevent the
research-leakage failure mode where missing-bid/ask events get
silently re-injected into the promotion pool as mid-priced
research-grade evidence.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from services.candidate_shadow_provenance import (
    SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
    SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
    _SCENARIO_ALIASES,
    candidate_return_for_gate,
    is_promotion_eligible,
    is_promotion_eligible_for_scenario,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _ok_outcome(
    *,
    mid: float = 12.5,
    cross_25: float | None = 8.0,
    cross_50: float | None = 3.5,
    conservative: float | None = 3.5,
) -> Dict[str, Any]:
    """Build a candidate_shadow_outcome dict mimicking the live
    simulator output. All scenarios populated by default; pass None
    to drop a particular scenario from the dict's value space."""
    return {
        "status": "ok",
        "labels": {
            "research_mid": True,
            "shadow_only": True,
            "not_execution_grade": True,
        },
        "mid_realized_return_pct": mid,
        "execution_scenario_returns_pct": {
            "mid": mid,
            "cross_25": cross_25,
            "cross_50": cross_50,
            "conservative": conservative,
        },
    }


def _forward_record(outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap an outcome into a fully-formed forward-validation record
    that the pre-PR-#66 `is_promotion_eligible` would accept."""
    return {
        "sample_provenance": SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
        "candidate_shadow_outcome": outcome,
    }


# ──────────────────────────────────────────────────────────────────────────
# candidate_return_for_gate — strict mode (fail-closed default)
# ──────────────────────────────────────────────────────────────────────────


def test_returns_finite_value_and_clean_metadata_on_happy_path() -> None:
    outcome = _ok_outcome(cross_25=8.0)
    value, meta = candidate_return_for_gate(outcome, "cross_25")
    assert value == pytest.approx(8.0)
    assert meta == {
        "scenario_requested": "cross_25",
        "scenario_used": "cross_25",
        "fallback_used": False,
        "missing_reason": None,
    }


def test_strict_mode_returns_none_when_scenario_value_is_none() -> None:
    """The realistic "missing bid/ask" case: simulator emitted the
    outcome block with all four scenario keys present, but the
    cross_25 value is None because bid/ask was unavailable on the
    chain. Fail-closed default refuses to substitute mid."""
    outcome = _ok_outcome(cross_25=None)
    value, meta = candidate_return_for_gate(outcome, "cross_25")
    assert value is None
    assert meta["missing_reason"] == "scenario_value_not_finite"
    assert meta["scenario_used"] is None
    assert meta["fallback_used"] is False


def test_strict_mode_returns_none_when_scenario_value_is_nan() -> None:
    outcome = _ok_outcome(cross_25=math.nan)
    value, meta = candidate_return_for_gate(outcome, "cross_25")
    assert value is None
    assert meta["missing_reason"] == "scenario_value_not_finite"


def test_strict_mode_rejects_bool_scenario_values() -> None:
    """Bool is a numeric subtype in Python (`True == 1`). Without
    explicit bool exclusion a misconfigured pipeline could write
    `cross_25: True` and have it sail through. Fail closed."""
    outcome = _ok_outcome(cross_25=True)  # type: ignore[arg-type]
    value, meta = candidate_return_for_gate(outcome, "cross_25")
    assert value is None
    assert meta["missing_reason"] == "scenario_value_not_finite"


def test_strict_mode_returns_none_when_status_not_ok() -> None:
    outcome = _ok_outcome()
    outcome["status"] = "skipped:missing_entry_quote"
    value, meta = candidate_return_for_gate(outcome, "cross_25")
    assert value is None
    assert meta["missing_reason"] == "outcome_status_not_ok"


def test_strict_mode_returns_none_when_scenario_block_missing() -> None:
    """Pre-PR-#64 outcome blocks (legacy historical data) don't
    carry execution_scenario_returns_pct. The helper must handle
    them without crashing."""
    outcome: Dict[str, Any] = {
        "status": "ok",
        "labels": {"research_mid": True, "shadow_only": True, "not_execution_grade": True},
        "mid_realized_return_pct": 5.0,
        # No execution_scenario_returns_pct.
    }
    value, meta = candidate_return_for_gate(outcome, "cross_25")
    assert value is None
    assert meta["missing_reason"] == "scenario_block_missing"


def test_strict_mode_returns_none_when_outcome_is_not_dict() -> None:
    value, meta = candidate_return_for_gate(None, "cross_25")
    assert value is None
    assert meta["missing_reason"] == "outcome_not_dict"

    value, meta = candidate_return_for_gate("not a dict", "cross_25")  # type: ignore[arg-type]
    assert value is None
    assert meta["missing_reason"] == "outcome_not_dict"


def test_strict_mode_returns_none_when_scenario_label_unknown() -> None:
    """Codex P2 (PR #66): a typo'd or future scenario label that is
    NOT in SCENARIO_LEVELS or _SCENARIO_ALIASES resolves to
    ``unknown_scenario_label`` — distinct from ``scenario_value_absent``,
    which is reserved for the case where the label IS recognized but
    its value is missing from the specific outcome dict. The
    distinction matters for the fallback contract (see the next
    test): unknown labels must NEVER fall back to mid."""
    outcome = _ok_outcome()
    value, meta = candidate_return_for_gate(outcome, "cross_99")
    assert value is None
    assert meta["missing_reason"] == "unknown_scenario_label"


# ──────────────────────────────────────────────────────────────────────────
# candidate_return_for_gate — fallback opt-in
# ──────────────────────────────────────────────────────────────────────────


def test_fallback_opt_in_routes_to_mid_when_scenario_missing() -> None:
    """The future-gate transition contract: when the caller opts
    into `allow_mid_fallback=True` AND the requested scenario is
    missing AND mid is finite, return the mid value with
    `fallback_used=True` so the substitution is auditable."""
    outcome = _ok_outcome(cross_25=None, mid=12.5)
    value, meta = candidate_return_for_gate(
        outcome, "cross_25", allow_mid_fallback=True,
    )
    assert value == pytest.approx(12.5)
    assert meta["scenario_used"] == "mid"
    assert meta["fallback_used"] is True
    # The original missing_reason for the cross_25 lookup must
    # still be reported so observability sees the cause, not just
    # the consequence.
    assert meta["missing_reason"] == "scenario_value_not_finite"


def test_fallback_does_not_fire_when_scenario_actually_priced() -> None:
    """When the requested scenario IS priced, fallback must not
    activate — `fallback_used` stays False regardless of the
    `allow_mid_fallback=True` opt-in."""
    outcome = _ok_outcome(cross_25=8.0)
    value, meta = candidate_return_for_gate(
        outcome, "cross_25", allow_mid_fallback=True,
    )
    assert value == pytest.approx(8.0)
    assert meta["scenario_used"] == "cross_25"
    assert meta["fallback_used"] is False
    assert meta["missing_reason"] is None


def test_fallback_does_not_rescue_unknown_scenario_labels() -> None:
    """Codex P2 regression on PR #66: a typo'd scenario label
    (``cross_99``) with ``allow_mid_fallback=True`` MUST still fail
    closed. The dangerous prior behavior was returning the mid value
    with ``fallback_used=True`` — i.e. a misspelled execution scenario
    would silently become research-mid evidence.

    The fix is to validate the requested label against
    ``SCENARIO_LEVELS + _SCENARIO_ALIASES`` BEFORE entering the
    fallback path. Unknown labels return ``(None, meta)`` with
    ``missing_reason=\"unknown_scenario_label\"`` regardless of the
    fallback flag.
    """
    outcome = _ok_outcome(mid=12.5)
    value, meta = candidate_return_for_gate(
        outcome, "cross_99", allow_mid_fallback=True,
    )
    assert value is None, (
        "Unknown scenario label must NEVER be rescued by mid "
        "fallback — that would let typos in gate configuration "
        "silently re-introduce research-mid evidence into "
        "execution-aware gates."
    )
    assert meta["missing_reason"] == "unknown_scenario_label"
    assert meta["scenario_used"] is None
    assert meta["fallback_used"] is False


def test_fallback_returns_none_when_mid_also_unusable() -> None:
    """Fallback opted in but mid itself is NaN — the helper must
    leave the original missing_reason intact rather than mask it
    with a misleading 'mid_also_missing' code."""
    outcome = _ok_outcome(cross_25=None, mid=math.nan)
    value, meta = candidate_return_for_gate(
        outcome, "cross_25", allow_mid_fallback=True,
    )
    assert value is None
    # Original reason preserved — the cross_25 lookup is still the
    # primary diagnostic.
    assert meta["missing_reason"] == "scenario_value_not_finite"
    assert meta["fallback_used"] is False


# ──────────────────────────────────────────────────────────────────────────
# Alias contract: conservative -> cross_50
# ──────────────────────────────────────────────────────────────────────────


def test_conservative_resolves_to_cross_50_via_alias_map() -> None:
    """Requesting `conservative` must return the SAME value as
    requesting `cross_50` on the same outcome. This prevents a
    downstream caller from accidentally treating them as
    independent stress dimensions."""
    outcome = _ok_outcome(cross_25=8.0, cross_50=3.5, conservative=3.5)
    cross_50_value, _ = candidate_return_for_gate(outcome, "cross_50")
    conservative_value, conservative_meta = candidate_return_for_gate(
        outcome, "conservative",
    )
    assert cross_50_value == pytest.approx(3.5)
    assert conservative_value == pytest.approx(3.5)
    # Metadata distinguishes the requested label from the canonical
    # label that produced the value — useful for audit trails.
    assert conservative_meta["scenario_requested"] == "conservative"
    assert conservative_meta["scenario_used"] == "cross_50"


def test_conservative_alias_is_explicit_in_the_module() -> None:
    """Pin the alias mapping directly so a future commit that drops
    or repoints it has to update this test in the same change."""
    assert _SCENARIO_ALIASES == {"conservative": "cross_50"}


# ──────────────────────────────────────────────────────────────────────────
# is_promotion_eligible — byte-equivalence regression
# ──────────────────────────────────────────────────────────────────────────


def test_is_promotion_eligible_unchanged_for_records_with_scenario_block() -> None:
    """PR #66 must NOT change which records the existing gate
    accepts. Records that pass today must pass after, records that
    fail today must fail after."""
    # Accepted today (mid finite, all labels intact, forward
    # provenance): still accepted.
    accepted = _forward_record(_ok_outcome())
    assert is_promotion_eligible(accepted) is True

    # Rejected today (historical replay provenance): still rejected.
    rejected_provenance = {
        "sample_provenance": SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
        "candidate_shadow_outcome": _ok_outcome(),
    }
    assert is_promotion_eligible(rejected_provenance) is False

    # Rejected today (mid NaN): still rejected.
    rejected_nan = _forward_record(_ok_outcome(mid=math.nan))
    assert is_promotion_eligible(rejected_nan) is False


def test_is_promotion_eligible_accepts_records_missing_scenario_block() -> None:
    """Mid-only legacy records (no execution_scenario_returns_pct)
    must still be eligible under the existing gate — PR #66 does
    NOT make the new block a prerequisite for the default path."""
    legacy_outcome: Dict[str, Any] = {
        "status": "ok",
        "labels": {
            "research_mid": True,
            "shadow_only": True,
            "not_execution_grade": True,
        },
        "mid_realized_return_pct": 12.5,
        # Deliberately no execution_scenario_returns_pct.
    }
    record = _forward_record(legacy_outcome)
    assert is_promotion_eligible(record) is True


# ──────────────────────────────────────────────────────────────────────────
# is_promotion_eligible_for_scenario — strictly tighter than the default
# ──────────────────────────────────────────────────────────────────────────


def test_scenario_gate_strictly_tighter_when_scenario_missing() -> None:
    """A record that passes the existing mid gate but lacks the
    scenario value must FAIL the new scenario gate in strict mode."""
    outcome = _ok_outcome(cross_25=None)
    record = _forward_record(outcome)
    assert is_promotion_eligible(record) is True
    assert is_promotion_eligible_for_scenario(record, "cross_25") is False


def test_scenario_gate_accepts_when_scenario_priced() -> None:
    outcome = _ok_outcome(cross_25=8.0)
    record = _forward_record(outcome)
    assert is_promotion_eligible_for_scenario(record, "cross_25") is True


def test_scenario_gate_fallback_opt_in_rescues_missing_scenario() -> None:
    """With `allow_mid_fallback=True` AND the existing mid gate
    passing, a missing scenario is rescued — gate accepts. This
    is the transition-mode contract; no production caller uses
    it in PR #66."""
    outcome = _ok_outcome(cross_25=None, mid=12.5)
    record = _forward_record(outcome)
    assert is_promotion_eligible_for_scenario(
        record, "cross_25", allow_mid_fallback=True,
    ) is True


def test_scenario_gate_routes_through_existing_gate_first() -> None:
    """The new helper depends on the existing gate. A record that
    fails the existing gate must fail the new gate even if the
    scenario value would otherwise be acceptable."""
    outcome = _ok_outcome(cross_25=8.0, mid=math.nan)
    record = _forward_record(outcome)
    # Mid is NaN — existing gate rejects on check #5.
    assert is_promotion_eligible(record) is False
    # New gate inherits the rejection regardless of cross_25.
    assert is_promotion_eligible_for_scenario(record, "cross_25") is False


# ──────────────────────────────────────────────────────────────────────────
# Public API defaults: allow_mid_fallback=False everywhere
# ──────────────────────────────────────────────────────────────────────────


def test_public_helpers_default_to_strict_mode() -> None:
    """Pin the keyword-only default value at the API boundary so a
    future signature change can't quietly flip the default."""
    import inspect

    sig = inspect.signature(candidate_return_for_gate)
    assert sig.parameters["allow_mid_fallback"].default is False

    sig2 = inspect.signature(is_promotion_eligible_for_scenario)
    assert sig2.parameters["allow_mid_fallback"].default is False
