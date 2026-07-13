"""Unit tests for the shared earnings-move computation.

This is the single source of truth that both web/api/edge_engine.py and
services/earnings_vol_snapshot.py delegate to (previously each carried a
byte-for-byte duplicate). The two call-site wrappers are covered transitively
by the golden master + reconciliation tests; this suite pins the canonical
module contract directly.
"""
import numpy as np
import pandas as pd

from services.earnings_move_profile import (
    EarningsMoveProfile,
    compute_earnings_move_profile,
    normalize_release_timing,
)


def _price_series(periods=140, start="2024-01-02"):
    dates = pd.bdate_range(start, periods=periods)
    return pd.Series(100.0 + np.linspace(0.0, 1.0, len(dates)), index=dates)


def _amc_events_with_moves(prices, positions, moves):
    """Inject an AMC move at each position and return the matching event dicts."""
    dates = prices.index
    events = [
        {"event_date": pd.Timestamp(dates[i]), "release_timing": "after market close"}
        for i in positions
    ]
    for pos, move in zip(positions, moves):
        prices.iloc[pos + 1] = float(prices.iloc[pos]) * (1.0 + move / 100.0)
    return events


class TestNormalizeReleaseTiming:
    def test_maps_free_strings_and_codes(self):
        assert normalize_release_timing("before market open") == "before market open"
        assert normalize_release_timing("BMO") == "before market open"
        assert normalize_release_timing("after market close") == "after market close"
        assert normalize_release_timing("amc") == "after market close"
        assert normalize_release_timing("intraday") == "during market hours"

    def test_unknown_and_empty(self):
        assert normalize_release_timing("") == "unknown"
        assert normalize_release_timing(None) == "unknown"
        assert normalize_release_timing("garbage") == "unknown"

    def test_infers_from_timestamp_time_of_day(self):
        assert normalize_release_timing(pd.Timestamp("2024-07-25 08:00")) == "before market open"
        assert normalize_release_timing(pd.Timestamp("2024-07-25 16:30")) == "after market close"
        assert normalize_release_timing(pd.Timestamp("2024-07-25 12:00")) == "during market hours"


class TestComputeEarningsMoveProfile:
    def test_empty_series_returns_none_source(self):
        profile = compute_earnings_move_profile(
            close=pd.Series(dtype=float), earnings_events=[], as_of_date=pd.Timestamp("2024-07-01").date()
        )
        assert isinstance(profile, EarningsMoveProfile)
        assert profile.source == "none"
        assert profile.earnings_event_count == 0
        assert profile.sample_size == 0
        assert profile.median_move_pct is None
        assert profile.raw_events == []

    def test_earnings_history_path(self):
        prices = _price_series()
        events = _amc_events_with_moves(prices, [20, 40, 60, 80, 100], [4.0, 5.0, 6.0, 7.0, 8.0])
        profile = compute_earnings_move_profile(
            close=prices, earnings_events=events, as_of_date=prices.index[-1].date()
        )
        assert profile.source == "earnings_history"
        assert profile.earnings_event_count == 5
        assert profile.sample_size == 5
        # last four injected moves are 5,6,7,8 → mean ≈ 6.5 (minor drift from
        # the linspace base trend + p1/p99 winsorization)
        assert abs(profile.avg_last4_move_pct - 6.5) < 0.2
        assert profile.p90_move_pct > profile.median_move_pct
        # raw_events are dated, chronological, and one per event
        assert len(profile.raw_events) == 5
        assert len(profile.raw_moves_pct) == 5
        dates = [e["date"] for e in profile.raw_events]
        assert dates == sorted(dates)

    def test_daily_fallback_when_no_events(self):
        prices = _price_series()
        profile = compute_earnings_move_profile(
            close=prices, earnings_events=[], as_of_date=prices.index[-1].date()
        )
        assert profile.source == "daily_fallback"
        assert profile.earnings_event_count == 0
        assert profile.sample_size > 0
        # the fallback path never populates the per-event UI records
        assert profile.raw_events == []
        assert profile.raw_moves_pct == []

    def test_as_of_date_cutoff_excludes_future_events(self):
        prices = _price_series()
        events = _amc_events_with_moves(prices, [20, 40, 60, 80, 100], [4.0, 5.0, 6.0, 7.0, 8.0])
        # Cut off before any event date → no earnings observations survive, so
        # the computation must fall back to daily moves.
        cutoff = prices.index[10].date()
        profile = compute_earnings_move_profile(close=prices, earnings_events=events, as_of_date=cutoff)
        assert profile.source == "daily_fallback"
        assert profile.earnings_event_count == 0


class TestAsOfCutoffLeakage:
    """F1: the as_of cutoff must bound the PRICE series, not just the events."""

    def test_daily_fallback_ignores_post_cutoff_prices(self):
        # Calm early regime (~0.02%/day), then a volatile post-cutoff regime
        # (alternating +/-5%/day). The daily-fallback p90 must reflect only the
        # calm pre-cutoff regime when as_of precedes the volatility.
        calm = np.full(140, 0.0002)
        volatile = np.tile([0.05, -0.05], 10)  # 20 sessions of large moves
        rets = np.concatenate([calm, volatile])
        idx = pd.bdate_range("2024-01-02", periods=len(rets))
        prices = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)

        cutoff = idx[130].date()  # strictly before the volatile regime
        # No usable earnings events -> daily fallback path.
        pre = compute_earnings_move_profile(close=prices, earnings_events=[], as_of_date=cutoff)
        full = compute_earnings_move_profile(close=prices, earnings_events=[], as_of_date=idx[-1].date())

        assert pre.source == "daily_fallback"
        # The calm pre-cutoff regime must not reflect the ~5% post-cutoff moves.
        assert pre.p90_move_pct is not None and pre.p90_move_pct < 1.0
        # Sanity: including the volatile tail (as_of at series end) is materially
        # larger, proving the truncation is what suppressed the leak.
        assert full.p90_move_pct > 2.0
        assert full.p90_move_pct > pre.p90_move_pct

    def test_amc_event_on_cutoff_does_not_consume_next_session_close(self):
        dates = pd.bdate_range("2024-01-02", periods=60)
        prices = pd.Series(np.linspace(100.0, 101.0, len(dates)), index=dates)
        event_i = 40
        event_date = dates[event_i].date()
        # Make the post-event session (D+1) a large, distinctive move.
        prices.iloc[event_i + 1] = prices.iloc[event_i] * 1.20  # +20% the day AFTER the AMC event
        events = [{"event_date": pd.Timestamp(event_date), "release_timing": "after market close"}]

        # cutoff == the event date: the AMC post-close is the NEXT session, which is
        # after the cutoff and must not be read -> the event forms no move.
        at_event = compute_earnings_move_profile(close=prices, earnings_events=events, as_of_date=event_date)
        assert at_event.earnings_event_count == 0

        # cutoff == D+1: the post-close is now legitimately available -> the event counts.
        next_day = dates[event_i + 1].date()
        at_next = compute_earnings_move_profile(close=prices, earnings_events=events, as_of_date=next_day)
        assert at_next.earnings_event_count == 1
        assert at_next.source == "earnings_history"
