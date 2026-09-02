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
# Real call-chain integration tests (Codex P2 on PR #72 first review)
#
# Codex caught that the prior "snapshot integration smoke" only
# called bounded_interp again — it never actually invoked
# build_vol_snapshot. A future refactor that accidentally dropped
# the call-site rewiring would not be caught. These tests close
# that gap by exercising the real call chain end-to-end with
# synthetic-but-realistic fixtures.
# ──────────────────────────────────────────────────────────────────────────


def test_build_vol_snapshot_iv30_none_with_specific_null_reason() -> None:
    """End-to-end call-chain pin for the production snapshot path.

    Constructs a chain with expiries 31D and 45D from the as_of
    date, calls the REAL ``build_vol_snapshot`` (not the helper),
    and asserts:

      1. ``snapshot.iv30 is None`` — the target 30D tenor falls
         BELOW the listed bracket [31D, 45D], so bounded_interp
         refused to extrapolate.
      2. ``snapshot.null_reasons["iv30"] == "target_below_listed_expiries"``
         — the SPECIFIC bounded-interp status, not the generic
         insufficient_term_structure_points catch-all. This proves
         the status code threads correctly from helper → snapshot
         dataclass → null_reasons dict.
      3. ``snapshot.iv45 is not None`` — the target 45D tenor IS
         bracketed (45 == max(days)) so it should still interpolate.

    A future refactor that dropped the bounded_interp call from
    earnings_vol_snapshot.py:892 would silently restore the
    np.interp clamping behavior; this test fires immediately.
    """
    from datetime import datetime, timedelta

    import pandas as pd

    from services.earnings_vol_snapshot import build_vol_snapshot

    as_of = datetime(2024, 7, 1)
    # 140 days of price history so RV calculations have data
    dates = pd.bdate_range("2024-01-02", periods=140)
    prices = pd.Series(100.0 + np.linspace(0, 1.0, len(dates)), index=dates)
    price_df = pd.DataFrame({
        "trade_date": dates,
        "open": prices.shift(1).fillna(prices.iloc[0] / 1.001).values,
        "high": (prices * 1.01).values,
        "low": (prices * 0.99).values,
        "close": prices.values,
    })

    # Chain: nearest expiry is 31D out (2024-08-01), next is 45D
    # out (2024-08-15). target=30D for iv30 lies BELOW [31, 45].
    expiry_31d = (as_of + timedelta(days=31)).strftime("%Y-%m-%d")
    expiry_45d = (as_of + timedelta(days=45)).strftime("%Y-%m-%d")
    as_of_str = as_of.strftime("%Y-%m-%d")
    chain_df = pd.DataFrame([
        {"trade_date": as_of_str, "expiry": expiry_31d, "call_put": "C",
         "strike": 100.0, "bid": 3.4, "ask": 3.6, "mid": 3.5, "iv": 0.28,
         "open_interest": 1000, "volume": 500, "underlying_price": 100.0},
        {"trade_date": as_of_str, "expiry": expiry_31d, "call_put": "P",
         "strike": 100.0, "bid": 3.3, "ask": 3.5, "mid": 3.4, "iv": 0.28,
         "open_interest": 1000, "volume": 500, "underlying_price": 100.0},
        {"trade_date": as_of_str, "expiry": expiry_45d, "call_put": "C",
         "strike": 100.0, "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": 0.32,
         "open_interest": 1000, "volume": 500, "underlying_price": 100.0},
        {"trade_date": as_of_str, "expiry": expiry_45d, "call_put": "P",
         "strike": 100.0, "bid": 3.9, "ask": 4.1, "mid": 4.0, "iv": 0.32,
         "open_interest": 1000, "volume": 500, "underlying_price": 100.0},
    ])

    snapshot = build_vol_snapshot(
        "TEST",
        as_of,
        option_chain_data=chain_df,
        earnings_metadata={
            "earnings_date": "2024-07-25",
            "release_timing": "after market close",
            "prior_events": [],
        },
        price_data=price_df,
    )

    # The load-bearing assertions.
    assert snapshot.iv30 is None, (
        f"Snapshot iv30 must be None when chain has no expiry "
        f"<= 30D. Got {snapshot.iv30!r}. If this returns a float, "
        f"the bounded_interp call at earnings_vol_snapshot.py was "
        f"reverted to np.interp."
    )
    assert snapshot.null_reasons.get("iv30") == "target_below_listed_expiries", (
        f"Snapshot null_reasons['iv30'] must carry the specific "
        f"bounded-interp status, not the generic "
        f"'insufficient_term_structure_points'. Got "
        f"{snapshot.null_reasons.get('iv30')!r}. If this is the "
        f"generic string, the STATUS_TO_NULL_REASON routing in "
        f"build_vol_snapshot was reverted."
    )
    # And iv45 IS bracketed (45 == max(days)) so it interpolates.
    assert snapshot.iv45 is not None, (
        f"Snapshot iv45 must interpolate when 45D is at the bracket "
        f"endpoint. Got {snapshot.iv45!r}."
    )


def test_backtest_signal_snapshot_signal_ready_false_when_iv30_unbracketed() -> None:
    """End-to-end call-chain pin for the backtest study path.

    Pre-PR-#72, ``build_signal_snapshot`` in
    scripts/backtest_iv_expansion_study.py used the same np.interp
    pattern, fabricating an iv30 that propagated into the candidate
    calendar prior the selector consumes.

    Post-PR-#72, the function returns
    ``{"signal_ready": False, "signal_fail_reasons": ["target_below_listed_expiries"]}``
    when the chain doesn't bracket the target tenor. This test
    constructs that exact scenario and asserts the signal_fail
    short-circuit fires.

    A future refactor that dropped the bounded-interp guard from
    backtest_iv_expansion_study.py:797 would silently re-introduce
    contaminated rows into the prior; this test fires immediately.
    """
    from datetime import timedelta

    import pandas as pd

    from scripts.backtest_iv_expansion_study import build_signal_snapshot

    entry_date = pd.Timestamp("2024-07-01")
    exit_date = pd.Timestamp("2024-07-26")
    event_date = pd.Timestamp("2024-07-25")

    # Price history sufficient for RV calculation.
    history_dates = pd.bdate_range("2024-01-02", periods=140)
    history_close = 100.0 + np.linspace(0, 1.0, len(history_dates))
    price_history = pd.DataFrame({
        "trade_date": history_dates,
        "open": history_close,
        "high": history_close * 1.01,
        "low": history_close * 0.99,
        "close": history_close,
        "volume": [5_000_000.0] * len(history_dates),
    })
    # Make sure entry_date is in the history.
    if entry_date not in set(price_history["trade_date"]):
        # Append the entry date row to the end with the last close.
        last_close = float(price_history["close"].iloc[-1])
        price_history = pd.concat([
            price_history,
            pd.DataFrame([{
                "trade_date": entry_date,
                "open": last_close, "high": last_close,
                "low": last_close, "close": last_close,
                "volume": 5_000_000.0,
            }]),
        ], ignore_index=True)

    # Day chain: expiries at 31D and 45D from entry_date — target
    # 30D is BELOW the bracket.
    expiry_31d = entry_date + timedelta(days=31)
    expiry_45d = entry_date + timedelta(days=45)
    day_chain = pd.DataFrame([
        {"expiry": expiry_31d, "call_put": "C", "strike": 100.0,
         "bid": 3.4, "ask": 3.6, "mid": 3.5, "iv": 0.28,
         "open_interest": 1000, "volume": 500, "spread_pct": 3.0},
        {"expiry": expiry_31d, "call_put": "P", "strike": 100.0,
         "bid": 3.3, "ask": 3.5, "mid": 3.4, "iv": 0.28,
         "open_interest": 1000, "volume": 500, "spread_pct": 3.0},
        {"expiry": expiry_45d, "call_put": "C", "strike": 100.0,
         "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": 0.32,
         "open_interest": 1000, "volume": 500, "spread_pct": 3.0},
        {"expiry": expiry_45d, "call_put": "P", "strike": 100.0,
         "bid": 3.9, "ask": 4.1, "mid": 4.0, "iv": 0.32,
         "open_interest": 1000, "volume": 500, "spread_pct": 3.0},
    ])

    event = pd.Series({"event_date": event_date, "release_timing": "AMC"})

    result = build_signal_snapshot(
        symbol="TEST",
        event=event,
        entry_date=entry_date,
        exit_date=exit_date,
        day_chain=day_chain,
        price_history=price_history,
    )

    assert result.get("signal_ready") is False, (
        f"Backtest signal_snapshot must return signal_ready=False "
        f"when target 30D tenor isn't bracketed. Got "
        f"signal_ready={result.get('signal_ready')!r}, "
        f"keys={list(result.keys())}. If True, the bounded-interp "
        f"short-circuit at backtest_iv_expansion_study.py:797 was "
        f"reverted, allowing fabricated iv30 to flow into the prior."
    )
    fail_reasons = result.get("signal_fail_reasons") or []
    assert "target_below_listed_expiries" in fail_reasons, (
        f"Expected 'target_below_listed_expiries' in "
        f"signal_fail_reasons; got {fail_reasons!r}."
    )


# ──────────────────────────────────────────────────────────────────────────
# Expiry-window coverage: the term structure must BRACKET 30D/45D when the
# chain actually lists those expiries.
#
# Regression for the weekly-heavy starvation bug. bounded_interp was doing
# its job correctly, but the caller only ever handed it the first
# `max_term_expiries` (6) expiries BY DATE. On weekly-heavy names that
# window closes inside ~2 weeks, so 30D/45D fell above the bracket and
# iv30/iv45 came back None — even though the same single already-fetched
# chain (expiration="all") carried 30D, 44D and 58D expiries. Observed live
# on NVDA: first 6 future expiries at DTE 2/5/9/12/14/16, killing iv_rv,
# cheapness_score and the 32%-weight IV-entry signal.
# ──────────────────────────────────────────────────────────────────────────

def _weekly_chain(as_of, dtes, iv_by_dte=None):
    """Build a minimal ATM chain listing one expiry per entry in *dtes*."""
    from datetime import timedelta

    import pandas as pd

    as_of_str = as_of.strftime("%Y-%m-%d")
    rows = []
    for dte in dtes:
        expiry = (as_of + timedelta(days=int(dte))).strftime("%Y-%m-%d")
        iv = float(iv_by_dte(dte)) if iv_by_dte else 0.30
        for side, mid in (("C", 3.5), ("P", 3.4)):
            rows.append({
                "trade_date": as_of_str, "expiry": expiry, "call_put": side,
                "strike": 100.0, "bid": mid - 0.1, "ask": mid + 0.1, "mid": mid,
                "iv": iv, "open_interest": 1000, "volume": 500,
                "spread_pct": 3.0, "underlying_price": 100.0,
            })
    return pd.DataFrame(rows)


def _price_history(as_of):
    import pandas as pd

    dates = pd.bdate_range("2024-01-02", periods=140)
    prices = pd.Series(100.0 + np.linspace(0, 1.0, len(dates)), index=dates)
    return pd.DataFrame({
        "trade_date": dates,
        "open": prices.shift(1).fillna(prices.iloc[0] / 1.001).values,
        "high": (prices * 1.01).values,
        "low": (prices * 0.99).values,
        "close": prices.values,
    })


def _snapshot_for(chain_df, as_of):
    from services.earnings_vol_snapshot import build_vol_snapshot

    return build_vol_snapshot(
        "TEST",
        as_of,
        option_chain_data=chain_df,
        earnings_metadata={
            "earnings_date": (as_of.date().replace(day=min(as_of.day + 7, 28))).strftime("%Y-%m-%d"),
            "release_timing": "after market close",
            "prior_events": [],
        },
        price_data=_price_history(as_of),
    )


def test_weekly_heavy_chain_still_reaches_iv30_and_iv45():
    """The live NVDA shape: 6 expiries inside 16D, then 30D/44D/58D."""
    from datetime import datetime

    as_of = datetime(2024, 7, 1)
    # Mirrors the observed NVDA ladder (DTE 2/5/9/12/14/16 then monthlies).
    dtes = [2, 5, 9, 12, 14, 16, 23, 30, 37, 44, 58, 93]
    chain = _weekly_chain(as_of, dtes, iv_by_dte=lambda d: 0.45 - 0.001 * d)
    snapshot = _snapshot_for(chain, as_of)

    assert snapshot.iv30 is not None, (
        "iv30 must interpolate when the chain lists a 30D expiry. Got None "
        f"with null_reason={snapshot.null_reasons.get('iv30')!r}. The expiry "
        "window in _build_term_structure_snapshot stopped short of the "
        "interpolation target again."
    )
    assert snapshot.iv45 is not None, (
        "iv45 must interpolate when the chain brackets 45D (44D and 58D are "
        f"both listed). Got None with "
        f"null_reason={snapshot.null_reasons.get('iv45')!r}."
    )
    assert "iv30" not in snapshot.null_reasons
    assert "iv45" not in snapshot.null_reasons
    # Near-term density is preserved: the nearest expiry still leads.
    assert snapshot.near_term_dte == 2


def test_short_chain_still_refuses_to_extrapolate():
    """PR #72 discipline preserved: no far expiries => honest None."""
    from datetime import datetime

    as_of = datetime(2024, 7, 1)
    # Genuinely short ladder — nothing at or beyond 30D exists.
    chain = _weekly_chain(as_of, [2, 5, 9, 12, 14, 16])
    snapshot = _snapshot_for(chain, as_of)

    assert snapshot.iv30 is None, (
        "iv30 must stay None when the chain genuinely lists nothing near 30D "
        f"— extending the window must not start extrapolating. Got {snapshot.iv30!r}."
    )
    assert snapshot.null_reasons.get("iv30") == "target_above_listed_expiries"
    assert snapshot.iv45 is None
    assert snapshot.null_reasons.get("iv45") == "target_above_listed_expiries"


def test_expiry_window_stops_once_target_is_covered():
    """Extension is bounded — it must not drag in the LEAPS tail."""
    from datetime import datetime

    from services.earnings_vol_snapshot import VolSnapshotConfig

    from services.earnings_vol_snapshot import _build_term_structure_snapshot

    as_of = datetime(2024, 7, 1)
    dtes = [2, 5, 9, 12, 14, 16, 23, 30, 37, 44, 58, 93, 121, 302, 485, 849]
    chain = _weekly_chain(as_of, dtes)
    cfg = VolSnapshotConfig()
    term = _build_term_structure_snapshot(chain, 100.0, as_of.date(), cfg)

    assert term.iv30 is not None and term.iv45 is not None
    # Stops at the first expiry at/beyond the 45D target (58D here), so the
    # 93D..849D LEAPS tail is never processed.
    assert term.point_count == 11, (
        f"Expected to stop right after covering 45D (11 expiries: 2..58), "
        f"got point_count={term.point_count}."
    )
    assert term.point_count <= cfg.max_term_expiries_extended, (
        "Expiry extension must respect max_term_expiries_extended."
    )


# ──────────────────────────────────────────────────────────────────────────
# select_tenor_spanning_expiries — the FETCH-side half of the same bug.
#
# The yfinance collectors pay one HTTP round-trip per expiry, so they cap the
# count. Taking "the first N by date" spends that budget entirely inside two
# weeks on weekly-heavy names, so the chain handed to the snapshot layer
# physically cannot bracket 30D/45D. Same budget, spanning selection.
# ──────────────────────────────────────────────────────────────────────────

_NVDA_LADDER = [
    "2026-08-19", "2026-08-21", "2026-08-24", "2026-08-28", "2026-08-31",
    "2026-09-02", "2026-09-04", "2026-09-11", "2026-09-18", "2026-09-25",
    "2026-10-02", "2026-10-16", "2026-11-20", "2027-06-17",
]


def _dtes(selected, as_of):
    from datetime import date as _d
    return [(_d.fromisoformat(e) - as_of).days for e in selected]


def test_spanning_selection_brackets_targets_within_same_budget():
    from datetime import date

    from services.iv_term_structure import select_tenor_spanning_expiries

    as_of = date(2026, 8, 19)
    selected = select_tenor_spanning_expiries(_NVDA_LADDER, as_of, max_expiries=6)

    assert len(selected) == 6, "must not increase the fetch budget"
    dtes = _dtes(selected, as_of)
    # The naive window would have been [2, 5, 9, 12, 14, 16] — nothing >= 30.
    assert max(dtes) >= 45, f"selection must reach the 45D target; got {dtes}"
    assert any(d >= 30 for d in dtes), f"selection must bracket 30D; got {dtes}"
    # Near-term density is still present for the near/back ratio.
    assert min(dtes) == 2, f"nearest expiry must survive; got {dtes}"


def test_spanning_selection_drops_past_and_current_expiries():
    from datetime import date

    from services.iv_term_structure import select_tenor_spanning_expiries

    as_of = date(2026, 8, 19)
    selected = select_tenor_spanning_expiries(_NVDA_LADDER, as_of, max_expiries=6)
    # 2026-08-19 is as_of itself (DTE 0) and must not be fetched.
    assert "2026-08-19" not in selected
    assert all(d > 0 for d in _dtes(selected, as_of))


def test_spanning_selection_handles_short_and_malformed_input():
    from datetime import date

    from services.iv_term_structure import select_tenor_spanning_expiries

    as_of = date(2026, 8, 19)
    # Nothing reaches the targets — take what exists, never raise.
    short = select_tenor_spanning_expiries(
        ["2026-08-21", "2026-08-24"], as_of, max_expiries=6
    )
    assert short == ["2026-08-21", "2026-08-24"]
    # Malformed entries cost one tenor, not the whole snapshot.
    mixed = select_tenor_spanning_expiries(
        ["not-a-date", "2026-08-21", None, "2026-09-18"], as_of, max_expiries=6
    )
    assert mixed == ["2026-08-21", "2026-09-18"]
    assert select_tenor_spanning_expiries([], as_of) == []


# ──────────────────────────────────────────────────────────────────────────
# Implausible ATM IV must ABSTAIN, not read as "cheap vol".
#
# Regression for the ORCL 2026-09-02 pre-open alert. yfinance returns
# bid=ask=0 and an impliedVolatility placeholder (~1e-5..4e-3) for every
# contract before the US open. A bare `> 0` filter let that through, so
# iv30 came out at 0.0039, IV/RV at 0.002, and _iv_entry_score clipped ANY
# ratio below 0.80 to its maximum — ranking a chain with no quotes at all as
# the single most attractive long-vega entry (rank 0.71, alert fired).
# ──────────────────────────────────────────────────────────────────────────

def test_placeholder_iv_is_rejected_rather_than_scored_as_cheap():
    from datetime import datetime

    from services.earnings_vol_snapshot import (
        MIN_PLAUSIBLE_ATM_IV,
        _build_term_structure_snapshot,
        VolSnapshotConfig,
    )

    as_of = datetime(2026, 9, 2)
    # The exact shape observed live: quotes absent, IV a placeholder.
    chain = _weekly_chain(as_of, [2, 9, 16, 23, 30, 79], iv_by_dte=lambda d: 0.0039)
    term = _build_term_structure_snapshot(chain, 100.0, as_of.date(), VolSnapshotConfig())

    assert term.iv30 is None, (
        f"A 0.39% ATM IV is an absent quote, not cheap vol; iv30 must abstain. "
        f"Got {term.iv30!r} — MIN_PLAUSIBLE_ATM_IV ({MIN_PLAUSIBLE_ATM_IV}) was bypassed."
    )
    assert term.near_term_atm_iv is None
    assert term.point_count == 0

    # A real vol surface at the same tenors still works.
    good = _weekly_chain(as_of, [2, 9, 16, 23, 30, 79], iv_by_dte=lambda d: 0.35)
    good_term = _build_term_structure_snapshot(good, 100.0, as_of.date(), VolSnapshotConfig())
    assert good_term.iv30 is not None and good_term.point_count >= 5


def test_expiry_walk_is_bounded_when_every_atm_iv_is_rejected():
    """The break conditions only engage after max_term_expiries usable points."""
    from datetime import datetime
    from unittest.mock import patch

    import services.earnings_vol_snapshot as evs

    as_of = datetime(2026, 9, 2)
    many = list(range(2, 200, 2))  # 99 listed expiries
    chain = _weekly_chain(as_of, many, iv_by_dte=lambda d: 0.0001)  # all rejected
    cfg = evs.VolSnapshotConfig()

    real = evs._expiry_atm_stats
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    with patch.object(evs, "_expiry_atm_stats", side_effect=counting):
        evs._build_term_structure_snapshot(chain, 100.0, as_of.date(), cfg)

    assert calls["n"] <= cfg.max_term_expiries_extended, (
        f"Walked {calls['n']} expiries with no usable stats; must stay bounded by "
        f"max_term_expiries_extended ({cfg.max_term_expiries_extended})."
    )
