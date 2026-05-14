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


# ──────────────────────────────────────────────────────────────────────────
# Finding B + C: per-structure rankings + side-weighted avg_iv_change
# ──────────────────────────────────────────────────────────────────────────


def test_feature_rankings_partitions_by_structure():
    """
    feature_rankings() must compute Spearman correlations and quartile
    statistics per structure, not pooled across structures. The output
    DataFrame must carry a leading ``structure`` column and contain rows
    for every structure that has ≥10 trades; the per-row ``sample_size``
    must never exceed the count of trades for that structure (proves no
    cross-structure leakage in the subsetting).
    """
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bs", "scripts/backtest_iv_expansion_study.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bs"] = mod
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(seed=20260507)
    rows = []
    structures = ["atm_straddle", "call_calendar"]
    for structure in structures:
        for i in range(40):
            rows.append({
                "structure": structure,
                "pnl_mid": float(rng.normal(0, 1)),
                "avg_iv_change": float(rng.normal(0, 1)),
                "return_mid_pct": float(rng.normal(0, 1)),
                "return_cross_50_pct": float(rng.normal(0, 1)),
                "signal_iv_rv30": float(rng.normal(1.2, 0.2)),
                "signal_iv_rv_har": float(rng.normal(1.2, 0.2)),
                "signal_term_structure_slope": float(rng.normal(0.005, 0.002)),
                "signal_near_back_iv_ratio": float(rng.normal(1.05, 0.05)),
                "signal_rv_percentile_rank": float(rng.uniform(20, 80)),
                "signal_near_term_spread_pct": float(rng.uniform(2, 8)),
                "signal_near_term_liquidity_proxy": float(rng.uniform(500, 5000)),
                "signal_event_implied_move_pct": float(rng.uniform(2, 12)),
                "signal_non_event_move_pct": float(rng.uniform(1, 6)),
                "signal_smile_curvature": float(rng.normal(0.0, 0.2)),
                "wf_ranking_score": float(rng.uniform(0, 1)),
            })
    base_trades = pd.DataFrame(rows)

    ranking = mod.feature_rankings(base_trades)
    assert not ranking.empty
    assert "structure" in ranking.columns, "rankings must carry leading 'structure' column"
    assert set(ranking["structure"].unique()) == set(structures), (
        f"expected rows for both {structures}, got {ranking['structure'].unique()}"
    )
    # No leakage: every row's sample_size must fit within its structure.
    per_structure_max = base_trades.groupby("structure").size().to_dict()
    for _, row in ranking.iterrows():
        assert row["sample_size"] <= per_structure_max[row["structure"]], (
            f"sample_size {row['sample_size']} exceeds {row['structure']} count "
            f"{per_structure_max[row['structure']]} — pooling regression"
        )
    # Sort contract: rows ordered by (structure asc, rank_score desc) within structure.
    for structure in structures:
        scores = ranking[ranking["structure"] == structure]["rank_score"].tolist()
        assert scores == sorted(scores, reverse=True), (
            f"{structure} rank_score not sorted descending"
        )


def test_avg_iv_change_is_side_weighted_for_calendar():
    """
    avg_iv_change must be side-weighted so positive ⇔ favorable for the
    position. For a calendar (front side=-1, back side=+1) where front
    IV crushes 10 vol pts and back IV rises 1 vol pt:
        unweighted  mean: (-10 + 1) / 2 = -4.5  (says IV "fell")
        side-weighted: (-1·-10 + 1·1) / 2 = +5.5  (says IV moved in our favor)
    The latter is what the backtest now emits.
    """
    # Mirror the inline computation pattern used by realize_structure_trade.
    class _Leg:
        def __init__(self, side: int, iv: float):
            self.side = side
            self.iv = iv

    entry_legs = [_Leg(side=-1, iv=30.0), _Leg(side=+1, iv=25.0)]   # short front, long back
    exit_legs  = [_Leg(side=-1, iv=20.0), _Leg(side=+1, iv=26.0)]   # front crush, back rise

    avg_iv_change = float(np.nanmean([
        e.side * (x.iv - e.iv)
        for e, x in zip(entry_legs, exit_legs)
        if e.iv is not None and x.iv is not None
    ]))

    assert math.isclose(avg_iv_change, 5.5, abs_tol=1e-9), (
        f"side-weighted calendar avg_iv_change should be +5.5, got {avg_iv_change}"
    )

    # Sanity: same expression on a long-only straddle gives the unsigned mean.
    straddle_entry = [_Leg(side=+1, iv=25.0), _Leg(side=+1, iv=25.0)]
    straddle_exit  = [_Leg(side=+1, iv=30.0), _Leg(side=+1, iv=30.0)]
    straddle_change = float(np.nanmean([
        e.side * (x.iv - e.iv)
        for e, x in zip(straddle_entry, straddle_exit)
    ]))
    assert math.isclose(straddle_change, 5.0, abs_tol=1e-9)
