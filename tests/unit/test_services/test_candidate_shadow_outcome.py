"""Tests for services/candidate_shadow_outcome.py (PR-AE C2).

Two responsibilities:

  1. Lock in the PUBLIC surface of the extracted module so the
     upcoming resolver (services/candidate_exit_resolver.py, PR-AE C4)
     can rely on these names being stable.

  2. Prove byte-equivalence: outputs of the new public function are
     identical to outputs of the legacy private alias still exposed
     via web/api/edge_engine.py. Catches any accidental divergence
     during the extraction or future drift.

The existing simulator behavior is already exhaustively tested via
tests/unit/test_web/test_calendar_dual_picker.py (which exercises
_simulate_pre_earnings_calendar_trade end-to-end, which in turn calls
the simulator under test here). Those tests continuing to pass after
PR-AE C2 is the primary behavioral proof. This file adds the
extraction-specific contract checks.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────
# Public-surface contract
# ──────────────────────────────────────────────────────────────────────────


def test_module_exposes_expected_public_names() -> None:
    """The new module's public API: five functions the resolver (C4)
    and the legacy edge_engine compatibility shim depend on. Any
    rename / removal must be deliberate, never accidental."""
    from services import candidate_shadow_outcome as mod
    for name in (
        "simulate_candidate_shadow_outcome",
        "lookup_exact_contract_row",
        "prepare_feature_chain",
        "best_contract_row",
        "safe_float",
    ):
        assert hasattr(mod, name), f"missing public name: {name}"
        assert callable(getattr(mod, name))


def test_module_import_graph_has_no_web_edge() -> None:
    """services/candidate_shadow_outcome.py is the neutral simulator
    home. It MUST NOT import from web/. The PR-AE design rev 2 makes
    this an explicit invariant — a lazy import would re-introduce
    the services -> web edge we just removed in PR-AD C4.

    Inspecting the module's source for any web. or web/ string is a
    cheap static guard against accidental re-introduction.
    """
    import inspect
    from services import candidate_shadow_outcome as mod
    source = inspect.getsource(mod)
    # Allow doc-comment references to web/api in narrative text but
    # forbid actual import statements.
    for forbidden in (
        "from web.",
        "from web ",
        "import web.",
    ):
        assert forbidden not in source, (
            f"services/candidate_shadow_outcome.py must not import from web/. "
            f"Found forbidden pattern: {forbidden!r}"
        )


def test_edge_engine_keeps_private_aliases_pointing_at_public_functions() -> None:
    """edge_engine.py's compatibility shim aliases each new public
    function under its legacy private name. Verifying the alias is
    actually wired to the new module proves the extraction did not
    leave a stale copy behind in edge_engine.py.
    """
    from services import candidate_shadow_outcome as mod
    from web.api import edge_engine

    assert (
        edge_engine._simulate_candidate_shadow_outcome
        is mod.simulate_candidate_shadow_outcome
    )
    assert edge_engine._lookup_exact_contract_row is mod.lookup_exact_contract_row
    assert edge_engine._prepare_feature_chain is mod.prepare_feature_chain
    assert edge_engine._best_contract_row is mod.best_contract_row
    assert edge_engine._safe_float is mod.safe_float


# ──────────────────────────────────────────────────────────────────────────
# Behavior parity (byte-equivalence with the pre-extraction simulator)
# ──────────────────────────────────────────────────────────────────────────


def _toy_chain(
    *,
    trade_date: str,
    front_expiry: str,
    back_expiry: str,
    strike: float,
    front_mid: float,
    back_mid: float,
    front_iv: float = 0.30,
    back_iv: float = 0.30,
    call_put: str = "C",
) -> pd.DataFrame:
    """Build a minimal two-row chain with the columns the simulator
    needs: call_put, trade_date, expiry, strike, bid/ask/mid, iv,
    open_interest, volume, spread_pct, liquidity_score.
    """
    return pd.DataFrame([
        {
            "call_put": call_put,
            "trade_date": trade_date,
            "expiry": front_expiry,
            "strike": strike,
            "bid": front_mid - 0.05,
            "ask": front_mid + 0.05,
            "mid": front_mid,
            "iv": front_iv,
            "open_interest": 500,
            "volume": 50,
            "spread_pct": 10.0,
            "liquidity_score": 0.8,
        },
        {
            "call_put": call_put,
            "trade_date": trade_date,
            "expiry": back_expiry,
            "strike": strike,
            "bid": back_mid - 0.05,
            "ask": back_mid + 0.05,
            "mid": back_mid,
            "iv": back_iv,
            "open_interest": 800,
            "volume": 80,
            "spread_pct": 5.0,
            "liquidity_score": 0.9,
        },
    ])


def _happy_path_inputs():
    entry_chain = _toy_chain(
        trade_date="2026-04-23",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=200.0,
        front_mid=1.0,
        back_mid=2.0,
        front_iv=0.30,
        back_iv=0.30,
    )
    exit_chain = _toy_chain(
        trade_date="2026-04-25",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=200.0,
        front_mid=0.4,
        back_mid=1.7,
        front_iv=0.20,
        back_iv=0.25,
    )
    dual_picker = {
        "shadow_status": "ok",
        "candidate_selection": {
            "side": "call",
            "strike": 200.0,
            "front_expiry": "2026-05-08",
            "back_expiry": "2026-05-22",
        },
    }
    return entry_chain, exit_chain, dual_picker


def test_simulate_happy_path_returns_ok_with_resolved_pnl() -> None:
    """The simulator resolves PnL when entry + exit chains carry the
    candidate's exact (expiry, strike) — calendar P&L convention is
    long back, short front; debit = back_mid - front_mid."""
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    entry, exit_, dp = _happy_path_inputs()
    out = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    assert out["status"] == "ok"
    # entry debit = 2.0 - 1.0 = 1.0; exit value = 1.7 - 0.4 = 1.3
    # mid_pnl = 1.3 - 1.0 = 0.3; mid_realized_return_pct = 30.0
    assert out["entry_debit_mid"] == pytest.approx(1.0)
    assert out["exit_value_mid"] == pytest.approx(1.3)
    assert out["mid_pnl"] == pytest.approx(0.3)
    assert out["mid_realized_return_pct"] == pytest.approx(30.0)
    assert out["iv_change_front"] == pytest.approx(-0.10)
    assert out["iv_change_back"] == pytest.approx(-0.05)
    # Research labels always present and all True
    for label in ("research_mid", "shadow_only", "not_execution_grade"):
        assert out["labels"][label] is True


@pytest.mark.parametrize(
    "dual_picker,expected_status_prefix",
    [
        (None, "skipped:no_dual_picker"),
        ({"shadow_status": "missing_chain"},
         "skipped:dual_picker_status:missing_chain"),
        ({"shadow_status": "ok", "candidate_selection": None},
         "skipped:no_candidate_selection"),
        ({"shadow_status": "ok",
          "candidate_selection": {"side": "call"}},  # missing strike/expiries
         "skipped:malformed_candidate_selection"),
        ({"shadow_status": "ok",
          "candidate_selection": {"side": "call", "strike": 200.0,
                                  "front_expiry": "not-a-date",
                                  "back_expiry": "2026-05-22"}},
         "skipped:bad_expiry_parse"),
    ],
)
def test_simulate_skipped_dual_picker_branches(
    dual_picker, expected_status_prefix,
) -> None:
    """Branches that exit before chain inspection. Each must return
    its expected status while keeping the full stable-shape dict
    (every field present, None where not applicable)."""
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    entry, exit_, _ = _happy_path_inputs()
    out = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dual_picker,
    )
    assert out["status"] == expected_status_prefix
    # Labels still present even on skipped paths
    for label in ("research_mid", "shadow_only", "not_execution_grade"):
        assert out["labels"][label] is True
    # PnL fields are None on skipped paths
    assert out["mid_pnl"] is None
    assert out["mid_realized_return_pct"] is None


def test_simulate_missing_chain_returns_skipped() -> None:
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    _, _, dp = _happy_path_inputs()
    # Entry chain missing
    out = simulate_candidate_shadow_outcome(
        entry_chain=None, exit_chain=_toy_chain(
            trade_date="x", front_expiry="2026-05-08",
            back_expiry="2026-05-22", strike=200.0,
            front_mid=0.4, back_mid=1.7,
        ),
        dual_picker=dp,
    )
    assert out["status"] == "skipped:missing_chain"

    # Exit chain missing
    out = simulate_candidate_shadow_outcome(
        entry_chain=_toy_chain(
            trade_date="x", front_expiry="2026-05-08",
            back_expiry="2026-05-22", strike=200.0,
            front_mid=1.0, back_mid=2.0,
        ),
        exit_chain=None,
        dual_picker=dp,
    )
    assert out["status"] == "skipped:missing_chain"


def test_simulate_missing_entry_quote_when_strike_not_in_entry() -> None:
    """Candidate names strike=200 but entry chain only carries 195."""
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    entry = _toy_chain(
        trade_date="2026-04-23",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=195.0,  # wrong strike
        front_mid=1.0, back_mid=2.0,
    )
    exit_ = _toy_chain(
        trade_date="2026-04-25",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=200.0,  # right strike on exit side
        front_mid=0.4, back_mid=1.7,
    )
    dp = {
        "shadow_status": "ok",
        "candidate_selection": {
            "side": "call",
            "strike": 200.0,
            "front_expiry": "2026-05-08",
            "back_expiry": "2026-05-22",
        },
    }
    out = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    assert out["status"] == "skipped:missing_entry_quote"


def test_simulate_missing_exit_quote_when_strike_not_in_exit() -> None:
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    entry = _toy_chain(
        trade_date="2026-04-23",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    exit_ = _toy_chain(
        trade_date="2026-04-25",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=195.0,  # wrong strike
        front_mid=0.4, back_mid=1.7,
    )
    dp = {
        "shadow_status": "ok",
        "candidate_selection": {
            "side": "call",
            "strike": 200.0,
            "front_expiry": "2026-05-08",
            "back_expiry": "2026-05-22",
        },
    }
    out = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    assert out["status"] == "skipped:missing_exit_quote"


def test_simulate_negative_debit_returns_skipped() -> None:
    """When entry_back_mid <= entry_front_mid, the calendar would
    pay credit rather than debit — the legacy simulator gates this
    out as an unrealistic/degenerate setup. The simulator under test
    must mirror the same behavior."""
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    # Inverted: front_mid > back_mid → negative debit
    entry = _toy_chain(
        trade_date="2026-04-23",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=200.0,
        front_mid=2.5,  # inverted
        back_mid=1.0,
    )
    exit_ = _toy_chain(
        trade_date="2026-04-25",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    dp = {
        "shadow_status": "ok",
        "candidate_selection": {
            "side": "call",
            "strike": 200.0,
            "front_expiry": "2026-05-08",
            "back_expiry": "2026-05-22",
        },
    }
    out = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    assert out["status"] == "skipped:negative_debit"


def test_simulate_byte_equivalent_via_legacy_alias() -> None:
    """The whole point of PR-AE C2: the new public function and the
    legacy private alias must produce IDENTICAL outputs for the
    same inputs. Tested on the happy path (most assertion surface)
    plus one skipped path so both branches of the dict are
    exercised."""
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    from web.api.edge_engine import _simulate_candidate_shadow_outcome

    entry, exit_, dp = _happy_path_inputs()
    via_public = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    via_alias = _simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    assert via_public == via_alias

    # Skipped path: equivalence holds on stable-shape outputs too
    skipped_dp = {"shadow_status": "ok", "candidate_selection": None}
    s_public = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=skipped_dp,
    )
    s_alias = _simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=skipped_dp,
    )
    assert s_public == s_alias


# ──────────────────────────────────────────────────────────────────────────
# Helper function parity
# ──────────────────────────────────────────────────────────────────────────


def test_safe_float_handles_none_nan_and_strings() -> None:
    from services.candidate_shadow_outcome import safe_float
    assert np.isnan(safe_float(None))
    assert np.isnan(safe_float(np.nan))
    assert np.isnan(safe_float(np.inf))
    assert np.isnan(safe_float("not a number"))
    assert safe_float(3.14) == pytest.approx(3.14)
    assert safe_float("2.5") == pytest.approx(2.5)
    # Custom default propagates
    assert safe_float(None, default=0.0) == 0.0


def test_prepare_feature_chain_filters_by_call_put_and_drops_bad_rows() -> None:
    from services.candidate_shadow_outcome import prepare_feature_chain
    df = pd.DataFrame([
        {"call_put": "C", "trade_date": "2026-04-23", "expiry": "2026-05-08",
         "strike": 100.0, "bid": 1.0, "ask": 1.1, "mid": 1.05},
        {"call_put": "P", "trade_date": "2026-04-23", "expiry": "2026-05-08",
         "strike": 100.0, "bid": 1.0, "ask": 1.1, "mid": 1.05},
        {"call_put": "C", "trade_date": "2026-04-23", "expiry": "2026-05-08",
         "strike": 100.0, "bid": 1.2, "ask": 1.0, "mid": 1.1},  # crossed
        {"call_put": "C", "trade_date": "2026-04-23", "expiry": "2026-05-08",
         "strike": 100.0, "bid": 1.0, "ask": 1.1, "mid": 0.0},  # zero mid
        {"call_put": "C", "trade_date": "2026-04-23", "expiry": "2026-05-08",
         "strike": np.nan, "bid": 1.0, "ask": 1.1, "mid": 1.05},  # bad strike
    ])
    out = prepare_feature_chain(df, call_put="C")
    # Only the first call row survives all the filters
    assert len(out) == 1
    assert out["call_put"].iloc[0] == "C"


def test_prepare_feature_chain_empty_input_returns_empty() -> None:
    from services.candidate_shadow_outcome import prepare_feature_chain
    assert prepare_feature_chain(None).empty
    assert prepare_feature_chain(pd.DataFrame()).empty


def test_best_contract_row_picks_highest_quality() -> None:
    """best_contract_row sorts by (liquidity desc, OI desc, volume desc,
    spread_pct asc). High-liquidity row wins."""
    from services.candidate_shadow_outcome import best_contract_row
    frame = pd.DataFrame([
        {"strike": 100.0, "liquidity_score": 0.5, "open_interest": 100,
         "volume": 10, "spread_pct": 20.0, "mid": 1.0},
        {"strike": 100.0, "liquidity_score": 0.9, "open_interest": 500,
         "volume": 50, "spread_pct": 5.0, "mid": 1.05},  # the winner
        {"strike": 100.0, "liquidity_score": 0.7, "open_interest": 200,
         "volume": 30, "spread_pct": 12.0, "mid": 1.02},
    ])
    row = best_contract_row(frame)
    assert row is not None
    assert row["mid"] == pytest.approx(1.05)


def test_best_contract_row_handles_missing_optional_columns() -> None:
    """REGRESSION (PR-AC commit 1b pattern): when liquidity_score or
    other optional columns are absent, the function must NOT crash.
    The PR-AC fix uses index-aligned fallback Series; this test
    confirms the parallel behavior survived the C2 extraction."""
    from services.candidate_shadow_outcome import best_contract_row
    # Bare-minimum frame with just strike and mid
    frame = pd.DataFrame([
        {"strike": 100.0, "mid": 1.05},
        {"strike": 100.0, "mid": 1.02},
    ])
    # Must not raise AttributeError on missing liquidity_score etc.
    row = best_contract_row(frame)
    assert row is not None


def test_best_contract_row_empty_returns_none() -> None:
    from services.candidate_shadow_outcome import best_contract_row
    assert best_contract_row(None) is None
    assert best_contract_row(pd.DataFrame()) is None


def test_lookup_exact_contract_row_finds_matching_row() -> None:
    from services.candidate_shadow_outcome import lookup_exact_contract_row
    chain = _toy_chain(
        trade_date="2026-04-23",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    row = lookup_exact_contract_row(
        chain, expiry=pd.Timestamp("2026-05-08"), strike=200.0, call_put="C",
    )
    assert row is not None
    assert row["mid"] == pytest.approx(1.0)


def test_lookup_exact_contract_row_returns_none_when_no_match() -> None:
    from services.candidate_shadow_outcome import lookup_exact_contract_row
    chain = _toy_chain(
        trade_date="2026-04-23",
        front_expiry="2026-05-08",
        back_expiry="2026-05-22",
        strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    # Strike not present
    row = lookup_exact_contract_row(
        chain, expiry=pd.Timestamp("2026-05-08"), strike=195.0, call_put="C",
    )
    assert row is None
    # Expiry not present
    row = lookup_exact_contract_row(
        chain, expiry=pd.Timestamp("2026-06-01"), strike=200.0, call_put="C",
    )
    assert row is None


# ──────────────────────────────────────────────────────────────────────────
# PR #64: execution realism foundations
# ──────────────────────────────────────────────────────────────────────────
#
# These tests pin three properties of the PR #64 change:
#
#   1. `mid_realized_return_pct` is byte-identical to its pre-PR-#64
#      value. The promotion gate still reads this field and we have
#      explicitly deferred the gate refactor to PR #65; any drift here
#      would change promotion behavior under the guise of "just adding
#      new fields."
#
#   2. New labeled scenario blocks appear when bid/ask is available,
#      and degrade gracefully to all-None when bid/ask is missing.
#
#   3. The naming collision between
#      `services.execution_scenarios.SCENARIO_LEVELS` (which reserves
#      "conservative" as a 50%-cross label) and the old
#      `CONSERVATIVE_EXECUTION_SCENARIO` in structure_scorecard.py
#      (which was actually "cross_25") is resolved — the new
#      `PROMOTION_BASELINE_SCENARIO` constant uses an unambiguous
#      name and "conservative" is preserved as a distinct scenario
#      label.


def test_mid_realized_return_pct_unchanged_after_pr_64() -> None:
    """Regression: the canonical research-grade promotion-evidence
    number must not have drifted. Use the same _happy_path_inputs
    fixture as test_simulate_happy_path_returns_ok_with_resolved_pnl
    — the numerically expected value is 30.0 (debit 1.0 → exit 1.3 →
    +30%)."""
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    entry, exit_, dp = _happy_path_inputs()
    out = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    assert out["status"] == "ok"
    assert out["mid_realized_return_pct"] == pytest.approx(30.0)
    # The new labeled-returns block must agree with the scalar on
    # the mid path — both are definitionally the same number.
    assert out["execution_scenario_returns_pct"]["mid"] == pytest.approx(30.0)


def test_simulate_emits_per_scenario_returns_when_bid_ask_present() -> None:
    """Happy path emits a dict-shaped `execution_scenario_returns_pct`
    keyed by every label in SCENARIO_LEVELS, with finite numbers for
    scenarios the simulator can price. The toy chain uses bid =
    mid-0.05, ask = mid+0.05 on every leg, so the 25%/50% crosses
    produce determinable per-scenario debits/exits and finite
    returns."""
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    from services.execution_scenarios import SCENARIO_LEVELS

    entry, exit_, dp = _happy_path_inputs()
    out = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    assert out["status"] == "ok"

    expected_labels = {name for name, _d in SCENARIO_LEVELS}
    for block_name in (
        "entry_scenario_values",
        "exit_scenario_values",
        "execution_scenario_returns_pct",
        "execution_scenario_pnl",
    ):
        block = out[block_name]
        assert isinstance(block, dict), f"{block_name} must be dict-shaped"
        assert set(block.keys()) == expected_labels, (
            f"{block_name} keys {set(block.keys())} != {expected_labels}"
        )

    # cross_25 and cross_50 must produce finite numbers — the toy
    # chain has well-formed bid/ask on every leg.
    for label in ("mid", "cross_25", "cross_50", "conservative"):
        assert out["execution_scenario_returns_pct"][label] is not None
        assert np.isfinite(out["execution_scenario_returns_pct"][label])

    # Sanity: harsher crosses should produce a worse (lower) return
    # than the mid path on a profitable trade. The happy-path trade
    # is +30% on mid; cross_50 (50% spread cross on each leg in each
    # direction) must be strictly worse.
    assert (
        out["execution_scenario_returns_pct"]["cross_50"]
        < out["execution_scenario_returns_pct"]["mid"]
    )


def test_simulate_degrades_gracefully_when_bid_ask_missing() -> None:
    """A chain whose rows have NaN bid/ask must still produce a
    populated mid path (the mid is independent of bid/ask once
    `prepare_feature_chain` has normalized the row). The per-scenario
    non-mid blocks degrade to None for the cross labels but the dict
    shape is preserved.

    Why this matters: pre-PR-#64 historical chains and any provider
    with sparse bid/ask coverage must continue to contribute mid-only
    promotion-evidence rows. PR #64 must not silently shrink the
    eligibility pool.
    """
    from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome
    from services.execution_scenarios import SCENARIO_LEVELS

    entry, exit_, dp = _happy_path_inputs()

    # Drop bid/ask columns entirely; leave mid intact. The
    # `prepare_feature_chain` upstream filter only enforces
    # `bid >= 0 and ask >= bid` when BOTH columns are present
    # (`{"bid", "ask"}.issubset(df.columns)` gate). Dropping the
    # columns is the right model for a provider whose schema simply
    # doesn't carry bid/ask, which is the realistic missing-quote
    # case in historical chains.
    entry = entry.drop(columns=["bid", "ask"])
    exit_ = exit_.drop(columns=["bid", "ask"])

    out = simulate_candidate_shadow_outcome(
        entry_chain=entry, exit_chain=exit_, dual_picker=dp,
    )
    # Mid path survives: the simulator gets to "ok" with the same
    # mid_realized_return_pct as the bid/ask-populated case.
    assert out["status"] == "ok"
    assert out["mid_realized_return_pct"] == pytest.approx(30.0)
    assert out["execution_scenario_returns_pct"]["mid"] == pytest.approx(30.0)

    # Non-mid scenarios collapse to None.
    expected_labels = {name for name, _d in SCENARIO_LEVELS}
    block = out["execution_scenario_returns_pct"]
    assert set(block.keys()) == expected_labels
    for label in expected_labels - {"mid"}:
        assert block[label] is None, (
            f"scenario {label} should be None when bid/ask missing, got {block[label]}"
        )


def test_empty_outcome_block_has_dict_shaped_scenario_fields() -> None:
    """`_empty_candidate_shadow_outcome` is called on every skipped
    path (no dual_picker, malformed selection, missing chain, etc.).
    The new scenario-block fields must be dict-shaped (with every
    SCENARIO_LEVELS label mapped to None), NOT scalar None.

    Downstream consumers in PR #65 will do
    `outcome["execution_scenario_returns_pct"].get("conservative")`
    on every record — including skipped ones — and that must not
    AttributeError because the block is None."""
    from services.candidate_shadow_provenance import _empty_candidate_shadow_outcome
    from services.execution_scenarios import SCENARIO_LEVELS

    out = _empty_candidate_shadow_outcome("skipped:no_dual_picker")
    expected_labels = {name for name, _d in SCENARIO_LEVELS}

    for block_name in (
        "entry_scenario_values",
        "exit_scenario_values",
        "execution_scenario_returns_pct",
        "execution_scenario_pnl",
    ):
        block = out[block_name]
        assert isinstance(block, dict), (
            f"{block_name} must be dict-shaped on empty outcomes (got {type(block).__name__})"
        )
        assert set(block.keys()) == expected_labels
        for label in expected_labels:
            assert block[label] is None


def test_promotion_eligibility_unchanged_by_new_scenario_fields() -> None:
    """PR #64 is foundations only — it must NOT change which records
    `is_promotion_eligible` accepts. The gate still keys on
    `mid_realized_return_pct`; the new labeled fields are evidence
    that PR #65's gate refactor will consume.

    Concretely: a record with the new fields populated remains
    promotion-eligible IFF it would have been pre-PR-#64. A record
    whose `mid_realized_return_pct` is finite but whose non-mid
    scenarios are all None (the realistic case for old historical
    chains) is still eligible.
    """
    from services.candidate_shadow_provenance import (
        SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
        is_promotion_eligible,
    )

    # Record with the new fields populated → still eligible.
    record_with_scenarios = {
        "sample_provenance": SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
        "candidate_shadow_outcome": {
            "status": "ok",
            "mid_realized_return_pct": 12.5,
            "labels": {
                "research_mid": True,
                "shadow_only": True,
                "not_execution_grade": True,
            },
            "execution_scenario_returns_pct": {
                "mid": 12.5, "cross_25": 8.0, "cross_50": 3.5, "conservative": 3.5,
            },
            "execution_scenario_pnl": {
                "mid": 12.5, "cross_25": 8.0, "cross_50": 3.5, "conservative": 3.5,
            },
            "entry_scenario_values": {
                "mid": 1.0, "cross_25": 1.05, "cross_50": 1.10, "conservative": 1.10,
            },
            "exit_scenario_values": {
                "mid": 1.125, "cross_25": 1.13, "cross_50": 1.14, "conservative": 1.14,
            },
        },
    }
    assert is_promotion_eligible(record_with_scenarios) is True

    # Record with non-mid scenarios all None (historical chain) →
    # still eligible. The new fields are NOT gating criteria.
    record_mid_only = {
        "sample_provenance": SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
        "candidate_shadow_outcome": {
            "status": "ok",
            "mid_realized_return_pct": 12.5,
            "labels": {
                "research_mid": True,
                "shadow_only": True,
                "not_execution_grade": True,
            },
            "execution_scenario_returns_pct": {
                "mid": 12.5, "cross_25": None, "cross_50": None, "conservative": None,
            },
        },
    }
    assert is_promotion_eligible(record_mid_only) is True


def test_promotion_baseline_scenario_and_conservative_no_longer_collide() -> None:
    """Regression for the PR #64 naming-collision fix.

    Before PR #64:
      - structure_scorecard.CONSERVATIVE_EXECUTION_SCENARIO = "cross_25"
      - execution_scenarios.SCENARIO_LEVELS contained "conservative" at
        a 50%-of-spread cross.
    Reading the structure_scorecard name and then the
    execution_scenarios labels would have made it appear that priors
    were filtered to a stricter scenario than they actually were.

    After PR #64:
      - structure_scorecard.PROMOTION_BASELINE_SCENARIO = "cross_25" (renamed)
      - "conservative" is preserved in SCENARIO_LEVELS, distinct from
        the baseline.

    This test pins both: the renamed constant exists and holds the
    historical baseline label, and "conservative" is still a
    SCENARIO_LEVELS member but is NOT the baseline.
    """
    from services.structure_scorecard import PROMOTION_BASELINE_SCENARIO
    from services.execution_scenarios import SCENARIO_LEVELS

    scenario_labels = {name for name, _d in SCENARIO_LEVELS}
    assert "conservative" in scenario_labels, (
        "SCENARIO_LEVELS must still expose the 'conservative' label so "
        "downstream consumers can reference it explicitly."
    )
    assert PROMOTION_BASELINE_SCENARIO in scenario_labels, (
        f"PROMOTION_BASELINE_SCENARIO={PROMOTION_BASELINE_SCENARIO!r} must "
        f"be a valid SCENARIO_LEVELS label (got labels {scenario_labels})."
    )
    assert PROMOTION_BASELINE_SCENARIO != "conservative", (
        "The historical promotion baseline is 'cross_25' — it must NOT "
        "be aliased to the 'conservative' label, which is reserved for "
        "the stricter 50%-of-spread cross in execution_scenarios."
    )
    # And it must NOT be the old misleading name.
    import services.structure_scorecard as sc
    assert not hasattr(sc, "CONSERVATIVE_EXECUTION_SCENARIO"), (
        "Old `CONSERVATIVE_EXECUTION_SCENARIO` name must be fully removed "
        "to prevent accidental reintroduction of the naming collision."
    )
