"""
Candidate shadow outcome simulator — neutral service module.

Extracted from ``web/api/edge_engine.py`` in PR-AE commit 2 (Codex
design-review required change). Before the extraction, this code
lived in the web layer and would have forced any future caller in
``services/`` (the upcoming candidate exit resolver in PR-AE C4) to
import upward from ``web``. That would violate the architectural
invariant established in PR-AD commit 4: ``services`` must not
depend on ``web/api``.

This module is the second extraction in that series:

  - PR-AD C4 lifted the provenance taxonomy and label constants into
    ``services/candidate_shadow_provenance.py``.
  - PR-AE C2 (this module) lifts the simulator function and the
    chain-quote helpers it requires.

``web/api/edge_engine.py`` keeps the legacy private names
(`_simulate_candidate_shadow_outcome`, `_lookup_exact_contract_row`,
`_prepare_feature_chain`, `_best_contract_row`, `_safe_float`) by
importing the new public functions from here with private aliases.
Every existing edge-engine call site stays identical, byte-for-byte.

Public API
----------

``simulate_candidate_shadow_outcome(*, entry_chain, exit_chain, dual_picker)``
    The shadow simulator itself. Resolves the candidate picker's
    contracts through entry + exit pricing and returns a
    stable-shape dict with a ``status`` enum. See the function
    docstring for the full enum.

``lookup_exact_contract_row(chain_df, *, expiry, strike, call_put)``
    Lookup helper that returns the best-quality row in *chain_df*
    matching the (expiry, strike, call_put) tuple, or ``None``.

``prepare_feature_chain(chain_df, call_put)``
    Normalizes a raw chain DataFrame: enforces dtypes, drops
    invalid rows, filters by call/put.

``best_contract_row(frame)``
    Quality-rank a frame's rows and return the top one. Used as the
    tie-breaker when multiple rows match a (expiry, strike) lookup.

``safe_float(value, default=np.nan)``
    Defensive numeric coercion that returns *default* for None /
    non-numeric / non-finite inputs.

All five functions are PURE — no I/O, no global state, no
side-effects on inputs. They are safe to call from any thread or
process.

PR-AE C2 invariants (enforced by tests in
tests/unit/test_services/test_candidate_shadow_outcome.py):
  - simulate_candidate_shadow_outcome's output for any (entry_chain,
    exit_chain, dual_picker) triple is byte-identical to the
    pre-extraction ``_simulate_candidate_shadow_outcome`` output.
  - The five legacy private names in edge_engine.py remain importable
    so existing call sites work unchanged.
  - This module's import graph is services + numpy + pandas. NO
    edges into ``web/`` or ``scripts/``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from services.candidate_shadow_provenance import (
    _CANDIDATE_SHADOW_LABELS,
    _empty_candidate_shadow_outcome,
)
from services.execution_scenarios import (
    SCENARIO_LEVELS,
    build_execution_scenarios,
    compare_execution_scenarios,
)


# ──────────────────────────────────────────────────────────────────────────
# Numeric and chain helpers
# ──────────────────────────────────────────────────────────────────────────

def safe_float(value: Any, default: float = np.nan) -> float:
    """Coerce *value* to a finite float; return *default* otherwise.

    None, non-numeric strings, NaN, ±inf, and unparseable inputs all
    collapse to *default*. The default is ``np.nan`` which the
    simulator then treats as a missing-quote signal.
    """
    try:
        if value is None:
            return float(default)
        parsed = float(value)
        if not np.isfinite(parsed):
            return float(default)
        return parsed
    except (TypeError, ValueError):
        return float(default)


def prepare_feature_chain(chain_df: pd.DataFrame, call_put: str = "C") -> pd.DataFrame:
    """Return *chain_df* filtered to *call_put* with normalized dtypes.

    Drops rows with non-positive mid, missing strikes, or
    crossed/negative bid-ask quotes. Used as the first step inside
    ``lookup_exact_contract_row`` so all consumers see a clean,
    consistent chain shape.
    """
    if chain_df is None or chain_df.empty:
        return pd.DataFrame()
    df = chain_df.copy()
    if "call_put" in df.columns:
        df["call_put"] = df["call_put"].astype(str).str.upper()
        df = df[df["call_put"] == call_put.upper()]
    for col in ("trade_date", "expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    for col in ("strike", "bid", "ask", "mid", "iv", "open_interest", "volume", "liquidity_score", "underlying_price", "spread_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"bid", "ask"}.issubset(df.columns):
        df = df[(df["bid"] >= 0) & (df["ask"] >= df["bid"])]
    if "mid" in df.columns:
        df = df[np.isfinite(df["mid"]) & (df["mid"] > 0)]
    if "strike" in df.columns:
        df = df[np.isfinite(df["strike"])]
    return df


def best_contract_row(frame: pd.DataFrame) -> Optional[pd.Series]:
    """Return the highest-quality row from *frame*, or None if empty.

    Sorts by (liquidity_score desc, open_interest desc, volume desc,
    spread_pct asc) and returns the first row. When an optional
    column is absent, fills with an index-aligned zero/inf Series
    so a missing column never crashes the call (PR-AC commit 1b
    pattern, mirrored from services/calendar_leg_picker.py).
    """
    if frame is None or frame.empty:
        return None
    ranked = frame.copy()
    # Parallel to the PR-AC commit-1b fix in services/calendar_leg_picker
    # `_best_row`: when an optional column is absent, `frame.get("col")`
    # returns a scalar (None), then `pd.to_numeric(scalar).fillna(...)`
    # raises AttributeError because the scalar has no .fillna. Use an
    # index-aligned fallback Series instead. Production FeatureStore
    # chains always carry these columns, but synthetic test chains and
    # any future provider with a sparser schema would otherwise crash.
    zero = pd.Series(0.0, index=ranked.index)
    inf_series = pd.Series(np.inf, index=ranked.index)
    def _coerce(col: str, fallback: pd.Series) -> pd.Series:
        if col in ranked.columns:
            return pd.to_numeric(ranked[col], errors="coerce").fillna(fallback.iloc[0])
        return fallback
    ranked["_liq"] = _coerce("liquidity_score", zero)
    ranked["_oi"] = _coerce("open_interest", zero)
    ranked["_vol"] = _coerce("volume", zero)
    ranked["_spread_pct"] = _coerce("spread_pct", inf_series)
    ranked = ranked.sort_values(
        ["_liq", "_oi", "_vol", "_spread_pct"],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def _empty_scenario_dict() -> Dict[str, Optional[float]]:
    """Return a fresh {label: None} dict covering every scenario in
    SCENARIO_LEVELS. Used as the default when bid/ask quotes are
    incomplete and per-scenario evaluation cannot run.

    Stable shape: downstream consumers (PR #65 promotion gate refactor,
    ledger readers) get the same key set on every simulator output —
    `None` values communicate "couldn't price this scenario," missing
    keys would communicate "this scenario doesn't exist," and those
    are very different. We always emit the keys.
    """
    return {name: None for name, _distance in SCENARIO_LEVELS}


def _quote_payload_from_leg_rows(
    front_row: pd.Series,
    back_row: pd.Series,
) -> Dict[str, Any]:
    """Adapt the simulator's two leg row Series into the
    ``quote_payload`` shape that :func:`build_execution_scenarios`
    expects. Returns a payload like ``{"bid_ask_mid": {"legs":
    {"front": {bid, ask, mid}, "back": {bid, ask, mid}}}}``.

    Why an adapter: the chain row carries bid/ask/mid as columns,
    while build_execution_scenarios was designed against the live
    quote-snapshot payload shape used elsewhere in the engine. Rather
    than duplicate the scenario math here we adapt the inputs so the
    same code-path produces the same numbers regardless of upstream
    quote source.

    A leg whose bid or ask is missing (None / NaN) flows through
    cleanly: build_execution_scenarios returns ``None`` for every
    scenario value at that leg, and compare_execution_scenarios then
    propagates ``None`` for the per-scenario realized return. The mid
    path is unaffected because the simulator computes
    ``mid_realized_return_pct`` directly from the row mids, not from
    the scenario block.
    """
    return {
        "bid_ask_mid": {
            "legs": {
                "front": {
                    "bid": safe_float(front_row.get("bid"), np.nan),
                    "ask": safe_float(front_row.get("ask"), np.nan),
                    "mid": safe_float(front_row.get("mid"), np.nan),
                },
                "back": {
                    "bid": safe_float(back_row.get("bid"), np.nan),
                    "ask": safe_float(back_row.get("ask"), np.nan),
                    "mid": safe_float(back_row.get("mid"), np.nan),
                },
            }
        }
    }


def lookup_exact_contract_row(
    chain_df: pd.DataFrame,
    *,
    expiry: pd.Timestamp,
    strike: float,
    call_put: str = "C",
) -> Optional[pd.Series]:
    """Return the chain row matching (expiry, strike, call_put), or None.

    Normalizes *chain_df* via :py:func:`prepare_feature_chain`, then
    filters to rows where expiry equals the normalized timestamp
    and strike is within 1e-8 of *strike*. When multiple rows match
    the filter (rare, but possible with split-quotes feeds), the
    quality tie-breaker from :py:func:`best_contract_row` picks one.
    """
    df = prepare_feature_chain(chain_df, call_put=call_put)
    if df.empty:
        return None
    subset = df[(df["expiry"] == pd.Timestamp(expiry).normalize()) & np.isclose(df["strike"], float(strike), atol=1e-8)]
    return best_contract_row(subset)


# ──────────────────────────────────────────────────────────────────────────
# PR-AD commit 1: candidate shadow outcome simulator
# ──────────────────────────────────────────────────────────────────────────
#
# PR-AC commit 2 added shadow-mode dual-picker logging that records WHICH
# contracts the legacy and candidate pickers would have selected. That
# alone is not enough to answer "did the candidate make more money than
# legacy?" — for that we also need the candidate's exit pricing. This
# simulator closes the gap: given the entry chain, exit chain, and
# dual_picker block, it resolves the candidate's contracts through
# entry+exit pricing and computes candidate PnL alongside legacy.
#
# CRITICAL: outcomes produced here on the historical-backtest events are
# IN-SAMPLE for the +14d rule (the rule was discovered on this exact
# data in PR-AB). Per docs/CALENDAR_PICKER_PROMOTION_2026-05-27.md, the
# candidate_realized_return_pct from these events must NOT be used as
# out-of-sample evidence. Promotion criteria reference forward
# (post-PR-AD-merge) events only.
#
# Function body is byte-for-byte equivalent to the pre-PR-AE-C2
# `_simulate_candidate_shadow_outcome` from web/api/edge_engine.py.

def simulate_candidate_shadow_outcome(
    *,
    entry_chain: Optional[pd.DataFrame],
    exit_chain: Optional[pd.DataFrame],
    dual_picker: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve the candidate picker's contracts through entry + exit
    pricing. Pure shadow simulation — no selector/scoring side effects.

    Inputs are the entry chain DataFrame, the exit chain DataFrame, and
    the dual_picker block produced by ``_dual_picker_calendar_selection``
    (PR-AC commit 2). The candidate selection inside dual_picker carries
    the (front_expiry, back_expiry, strike, side) needed to look up the
    actual contracts.

    Returns a stable-shape dict — every key in
    ``_CANDIDATE_SHADOW_OUTCOME_FIELDS`` is always present with None
    when not applicable. Status enum:

      "ok"                                  — full PnL resolved
      "skipped:no_dual_picker"              — dual_picker arg was None
      "skipped:dual_picker_status:*"        — dual picker itself failed
      "skipped:no_candidate_selection"      — candidate picker abstained
      "skipped:malformed_candidate_selection" — missing required fields
      "skipped:bad_expiry_parse"            — expiry strings unparseable
      "skipped:missing_chain"               — entry/exit chain was None
      "skipped:missing_entry_quote"         — candidate's leg not in
                                              entry chain
      "skipped:missing_exit_quote"          — leg not in exit chain
      "skipped:non_finite_quote"            — quote mid was NaN/inf
      "skipped:negative_debit"              — entry debit <= 0 (mirrors
                                              the legacy simulator's
                                              negative-debit gate)
    """
    if dual_picker is None:
        return _empty_candidate_shadow_outcome("skipped:no_dual_picker")
    if dual_picker.get("shadow_status") != "ok":
        return _empty_candidate_shadow_outcome(
            f"skipped:dual_picker_status:{dual_picker.get('shadow_status') or 'unknown'}"
        )

    candidate = dual_picker.get("candidate_selection")
    if candidate is None:
        return _empty_candidate_shadow_outcome("skipped:no_candidate_selection")

    side = candidate.get("side")
    call_put = "C" if side == "call" else "P" if side == "put" else None
    strike = candidate.get("strike")
    front_expiry_str = candidate.get("front_expiry")
    back_expiry_str = candidate.get("back_expiry")
    if call_put is None or strike is None or not front_expiry_str or not back_expiry_str:
        return _empty_candidate_shadow_outcome("skipped:malformed_candidate_selection")

    try:
        front_expiry = pd.Timestamp(front_expiry_str)
        back_expiry = pd.Timestamp(back_expiry_str)
    except (ValueError, TypeError):
        return _empty_candidate_shadow_outcome("skipped:bad_expiry_parse")

    if entry_chain is None or exit_chain is None:
        return _empty_candidate_shadow_outcome("skipped:missing_chain")

    entry_front = lookup_exact_contract_row(
        entry_chain, expiry=front_expiry, strike=float(strike), call_put=call_put,
    )
    entry_back = lookup_exact_contract_row(
        entry_chain, expiry=back_expiry, strike=float(strike), call_put=call_put,
    )
    if entry_front is None or entry_back is None:
        return _empty_candidate_shadow_outcome("skipped:missing_entry_quote")

    exit_front = lookup_exact_contract_row(
        exit_chain, expiry=front_expiry, strike=float(strike), call_put=call_put,
    )
    exit_back = lookup_exact_contract_row(
        exit_chain, expiry=back_expiry, strike=float(strike), call_put=call_put,
    )
    if exit_front is None or exit_back is None:
        return _empty_candidate_shadow_outcome("skipped:missing_exit_quote")

    entry_front_mid = safe_float(entry_front.get("mid"), np.nan)
    entry_back_mid = safe_float(entry_back.get("mid"), np.nan)
    exit_front_mid = safe_float(exit_front.get("mid"), np.nan)
    exit_back_mid = safe_float(exit_back.get("mid"), np.nan)
    if not (np.isfinite(entry_front_mid) and np.isfinite(entry_back_mid)
            and np.isfinite(exit_front_mid) and np.isfinite(exit_back_mid)):
        return _empty_candidate_shadow_outcome("skipped:non_finite_quote")

    # Calendar P&L: long back, short front. Entry debit = back - front.
    # Mirrors the legacy simulator's convention so legacy and candidate
    # PnLs are directly comparable.
    entry_debit_mid = float(entry_back_mid - entry_front_mid)
    if entry_debit_mid <= 0:
        return _empty_candidate_shadow_outcome("skipped:negative_debit")
    exit_value_mid = float(exit_back_mid - exit_front_mid)
    mid_pnl = float(exit_value_mid - entry_debit_mid)
    mid_realized_return_pct = float((mid_pnl / entry_debit_mid) * 100.0)

    entry_front_iv = safe_float(entry_front.get("iv"), np.nan)
    entry_back_iv = safe_float(entry_back.get("iv"), np.nan)
    exit_front_iv = safe_float(exit_front.get("iv"), np.nan)
    exit_back_iv = safe_float(exit_back.get("iv"), np.nan)
    iv_change_front = (
        float(exit_front_iv - entry_front_iv)
        if np.isfinite(entry_front_iv) and np.isfinite(exit_front_iv) else None
    )
    iv_change_back = (
        float(exit_back_iv - entry_back_iv)
        if np.isfinite(entry_back_iv) and np.isfinite(exit_back_iv) else None
    )

    # PR #64 (execution realism foundations): emit per-scenario debit /
    # exit / realized-return alongside the existing mid-only fields.
    # The new fields are ADDITIVE — `mid_realized_return_pct` is still
    # computed directly from the row mids above (not via
    # `compare_execution_scenarios`) so its value is byte-identical to
    # pre-PR-#64 outputs and the existing aggregator + resolver paths
    # are unaffected.
    #
    # Promotion eligibility (`is_promotion_eligible` in
    # candidate_shadow_provenance.py) still gates on
    # `mid_realized_return_pct`; PR #65 will be the policy refactor
    # that consumes these new labeled returns. PR #64 only makes the
    # evidence substrate honest.
    #
    # Bid/ask gaps degrade gracefully: missing leg quotes flow through
    # `build_execution_scenarios` as None scenario values, and
    # `compare_execution_scenarios` propagates None into the per-
    # scenario realized return. The mid column is independent of
    # bid/ask in the existing simulator (mid was either provided
    # directly or computed as (bid+ask)/2 by `prepare_feature_chain`
    # upstream), so the mid path always survives. This is by design:
    # historical chains with sparse bid/ask coverage should still
    # produce promotion-evidence rows for the mid-priced
    # research-grade label.
    structure_for_scenarios = "call_calendar" if call_put == "C" else "put_calendar"
    try:
        entry_scenarios = build_execution_scenarios(
            structure=structure_for_scenarios,
            quote_payload=_quote_payload_from_leg_rows(entry_front, entry_back),
            phase="entry",
        )
        exit_scenarios = build_execution_scenarios(
            structure=structure_for_scenarios,
            quote_payload=_quote_payload_from_leg_rows(exit_front, exit_back),
            phase="exit",
        )
        scenario_compare = compare_execution_scenarios(
            entry=entry_scenarios.to_dict(),
            exit=exit_scenarios.to_dict(),
        )
        entry_scenario_values = dict(entry_scenarios.scenario_values)
        exit_scenario_values = dict(exit_scenarios.scenario_values)
        execution_scenario_returns_pct = dict(scenario_compare.get("realized_return_pct", {}))
        execution_scenario_pnl = dict(scenario_compare.get("realized_pnl", {}))
    except Exception:
        # Defense in depth: any helper failure must not break the mid
        # path. Stamp empty scenario dicts and continue. We surface
        # this only via the structure of the returned scenario blocks
        # (all-None) rather than a separate status, because the
        # legacy `status` enum is part of the resolver's stable
        # contract.
        entry_scenario_values = _empty_scenario_dict()
        exit_scenario_values = _empty_scenario_dict()
        execution_scenario_returns_pct = _empty_scenario_dict()
        execution_scenario_pnl = _empty_scenario_dict()

    # `mid` scenario in the labeled returns must equal
    # `mid_realized_return_pct` (definitionally — both are
    # ((exit_value - entry_debit) / entry_debit) * 100). Anchor it
    # explicitly rather than trusting the upstream helper, so the
    # "mid" key here remains the canonical research-grade number even
    # if the helper's mid path drifts in some future refactor.
    execution_scenario_returns_pct["mid"] = mid_realized_return_pct
    execution_scenario_pnl["mid"] = mid_pnl
    entry_scenario_values["mid"] = entry_debit_mid
    exit_scenario_values["mid"] = exit_value_mid

    return {
        "status": "ok",
        "labels": dict(_CANDIDATE_SHADOW_LABELS),
        "side": side,
        "front_expiry": front_expiry_str,
        "back_expiry": back_expiry_str,
        "strike": float(strike),
        "entry_front_mid": float(entry_front_mid),
        "entry_back_mid": float(entry_back_mid),
        "exit_front_mid": float(exit_front_mid),
        "exit_back_mid": float(exit_back_mid),
        "entry_front_iv": float(entry_front_iv) if np.isfinite(entry_front_iv) else None,
        "entry_back_iv": float(entry_back_iv) if np.isfinite(entry_back_iv) else None,
        "exit_front_iv": float(exit_front_iv) if np.isfinite(exit_front_iv) else None,
        "exit_back_iv": float(exit_back_iv) if np.isfinite(exit_back_iv) else None,
        "entry_debit_mid": entry_debit_mid,
        "exit_value_mid": exit_value_mid,
        "mid_pnl": mid_pnl,
        "mid_realized_return_pct": mid_realized_return_pct,
        "iv_change_front": iv_change_front,
        "iv_change_back": iv_change_back,
        # PR #64: per-scenario evidence. Same stable-shape contract as
        # the rest of the outcome block — every scenario in
        # SCENARIO_LEVELS is present; values are None when bid/ask was
        # missing or the scenario couldn't be priced.
        "entry_scenario_values": entry_scenario_values,
        "exit_scenario_values": exit_scenario_values,
        "execution_scenario_returns_pct": execution_scenario_returns_pct,
        "execution_scenario_pnl": execution_scenario_pnl,
    }
