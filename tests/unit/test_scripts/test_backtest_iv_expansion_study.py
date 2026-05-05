"""
Tests for scripts.backtest_iv_expansion_study earnings-session exclusion.

The audit (Finding A) established that the backtest must exclude past
earnings sessions from the RV/HAR baseline so that event_implied_move_pct
matches what services.earnings_vol_snapshot produces in production. These
tests pin the two contracts the backtest now relies on:

  1. _event_contaminated_session_dates returns the right session for a
     given (price-history, past-event) pair.
  2. yang_zhang_rv30 with excluded_sessions actually produces a smaller
     RV when a high-variance event session is masked out — i.e. the
     wiring is real, not a no-op.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from services.earnings_vol_snapshot import _event_contaminated_session_dates
from services.realized_vol import yang_zhang_rv30


def _synthetic_ohlc(n_sessions: int = 60, jump_offset: int = -10) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Build n_sessions of calm OHLC data plus one outsized jump session at
    ``jump_offset`` from the end. Returns (frame, jump_session_date).

    Calm baseline: 0.4 % daily moves; jump session: ±8 % open-to-close.
    """
    base_dates = pd.bdate_range(end="2025-06-30", periods=n_sessions)
    rng = np.random.default_rng(seed=20260505)
    # Calm sessions: small log returns ~ N(0, 0.004).
    log_returns = rng.normal(loc=0.0, scale=0.004, size=n_sessions)
    log_returns[jump_offset] = 0.08  # 8 % event-day jump
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    opens = np.r_[100.0, closes[:-1]]
    # Calm intraday range, but the event day expands wider.
    span = np.where(np.arange(n_sessions) == (n_sessions + jump_offset), 0.10, 0.008)
    highs = np.maximum(opens, closes) * (1.0 + span / 2.0)
    lows = np.minimum(opens, closes) * (1.0 - span / 2.0)
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=base_dates,
    )
    df.index.name = "trade_date"
    jump_date = base_dates[jump_offset]
    return df, jump_date


def test_event_contaminated_session_dates_marks_after_market_close_next_session():
    """
    For an event with timing=after market close, the session marked
    contaminated is the *next* trading session (where the gap-open
    materialises).
    """
    hist, jump_date = _synthetic_ohlc(n_sessions=60, jump_offset=-10)
    # Place the earnings event on the *previous* trading session, with
    # release after market close, so the contamination should land on
    # jump_date itself (the next session = where the gap opens).
    prior_event_date = hist.index[hist.index.get_loc(jump_date) - 1]
    events = [{"event_date": prior_event_date, "release_timing": "After market close"}]
    excluded = _event_contaminated_session_dates(hist, events)
    assert pd.Timestamp(jump_date).normalize() in excluded, (
        f"expected {jump_date.date()} in excluded set, got {sorted(excluded)}"
    )


def test_event_contaminated_session_dates_marks_before_market_open_event_session():
    """
    For timing=before market open (or unknown/intra-day), the contamination
    is the event session itself.
    """
    hist, jump_date = _synthetic_ohlc(n_sessions=60, jump_offset=-10)
    events = [{"event_date": jump_date, "release_timing": "Before market open"}]
    excluded = _event_contaminated_session_dates(hist, events)
    assert pd.Timestamp(jump_date).normalize() in excluded


def test_yang_zhang_rv30_drops_meaningfully_when_event_session_excluded():
    """
    YZ RV computed over a window that contains a single 8 % event session
    must be measurably smaller when that session is masked out. Confirms
    that excluded_sessions wiring is doing real work and not a no-op.
    """
    hist, jump_date = _synthetic_ohlc(n_sessions=60, jump_offset=-10)
    rv_with = yang_zhang_rv30(hist, window=30, excluded_sessions=None)
    rv_without = yang_zhang_rv30(
        hist, window=30, excluded_sessions={pd.Timestamp(jump_date).normalize()}
    )
    assert math.isfinite(rv_with) and rv_with > 0
    assert math.isfinite(rv_without) and rv_without > 0
    # The included-event RV must be at least 25 % larger; a single 8 % jump
    # in a 30-day window of 0.4 %/day base vol is overwhelmingly the
    # dominant variance contribution.
    assert rv_with > rv_without * 1.25, (
        f"exclusion did not lower RV meaningfully: with={rv_with:.4f} without={rv_without:.4f}"
    )
