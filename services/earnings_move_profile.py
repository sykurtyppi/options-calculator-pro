"""Shared historical earnings-move computation — single source of truth.

Computes "what the stock actually did on its past earnings" statistics from a
free close-price series plus a list of past earnings events. Two call sites
consume it:

  * ``web/api/edge_engine.py``       — the live web analysis engine
  * ``services/earnings_vol_snapshot.py`` — the neutral vol snapshot

Both previously carried a byte-for-byte duplicate of this ~60-line loop (audit:
code-health duplication). They now delegate here and map the canonical
:class:`EarningsMoveProfile` to their own return shapes, so the math cannot
drift between the two paths.

The computation is deliberately pure and provider-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from services.candidate_shadow_outcome import safe_float


def normalize_release_timing(value: Any) -> str:
    """Canonical BMO / AMC / intraday classifier for an earnings release time.

    Accepts a datetime/Timestamp (inferred from time-of-day) or a free string
    ("before market open", "amc", "post", …). Returns one of
    "before market open" / "after market close" / "during market hours" /
    "unknown".
    """
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if any((value.hour, value.minute, value.second, value.microsecond)):
            total_minutes = (value.hour * 60) + value.minute
            if total_minutes < (9 * 60 + 30):
                return "before market open"
            if total_minutes >= (16 * 60):
                return "after market close"
            return "during market hours"
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "before" in text or text in {"bmo", "pre", "am"}:
        return "before market open"
    if "after" in text or text in {"amc", "post", "pm"}:
        return "after market close"
    if "during" in text or "intraday" in text:
        return "during market hours"
    return "unknown"


@dataclass(frozen=True)
class EarningsMoveProfile:
    """Canonical output of :func:`compute_earnings_move_profile`.

    ``source`` is one of "earnings_history" | "daily_fallback" | "none".
    ``earnings_event_count`` counts REAL earnings observations (0 for the
    daily fallback / empty cases); ``sample_size`` counts the observations
    actually behind the stats (earnings moves, or fallback daily moves).
    ``raw_moves_pct`` is the clipped array the stats are computed over;
    ``raw_events`` are the UNCLIPPED dated per-event records for the
    "last N earnings" UI panel (populated only on the earnings_history path).
    """

    source: str
    earnings_event_count: int
    sample_size: int
    median_move_pct: Optional[float]
    p90_move_pct: Optional[float]
    avg_last4_move_pct: Optional[float]
    std_move_pct: Optional[float]
    raw_moves_pct: List[float] = field(default_factory=list)
    raw_events: List[Dict[str, Any]] = field(default_factory=list)


_EMPTY = EarningsMoveProfile(
    source="none",
    earnings_event_count=0,
    sample_size=0,
    median_move_pct=None,
    p90_move_pct=None,
    avg_last4_move_pct=None,
    std_move_pct=None,
)


def compute_earnings_move_profile(
    close: pd.Series,
    earnings_events: List[Any],
    as_of_date: date,
) -> EarningsMoveProfile:
    """Compute historical earnings-move stats.

    close          : yfinance close series (free OHLCV, no API cost)
    earnings_events: list of either pandas timestamps (legacy) or dicts with
                     {"event_date": Timestamp, "release_timing": str}
    as_of_date     : cutoff — events strictly after this date are ignored (lets
                     the backtest evaluate historically; the live engine passes
                     today)
    """
    if close is None or close.empty:
        return _EMPTY

    price_series = close.copy()
    if isinstance(price_series.index, pd.DatetimeIndex):
        idx = price_series.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        price_series.index = idx.normalize()
    price_series = price_series[~price_series.index.duplicated(keep="last")].sort_index()
    # F1: enforce the as_of cutoff on the PRICE series, not just the events.
    # Without this the daily fallback's .tail(126) and an AMC event on the cutoff
    # date (whose post-event close is the following session) can read closes
    # dated after as_of_date — future-data leakage into a historical/backtest
    # profile. Truncating here makes the helper's contract leak-safe regardless
    # of whether the caller pre-truncated its frame.
    cutoff = pd.Timestamp(as_of_date).normalize()
    if isinstance(price_series.index, pd.DatetimeIndex):
        price_series = price_series[price_series.index <= cutoff]
    index_arr = price_series.index.to_numpy()
    if len(index_arr) < 10:
        return _EMPTY

    parsed_events: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for item in earnings_events or []:
        try:
            if isinstance(item, Mapping):
                event_ts_raw = item.get("event_date")
                timing = normalize_release_timing(item.get("release_timing"))
            else:
                # Raw timestamp (not a mapping) — infer BMO/AMC from time-of-day
                # instead of silently collapsing to "unknown".
                event_ts_raw = item
                timing = normalize_release_timing(item)
            if event_ts_raw is None:
                continue
            event_ts = pd.Timestamp(event_ts_raw).normalize()
            if event_ts > cutoff:
                continue
            existing = parsed_events.get(event_ts)
            if existing is None or existing.get("release_timing") == "unknown":
                parsed_events[event_ts] = {
                    "event_date": event_ts,
                    "release_timing": timing,
                }
        except Exception:
            continue

    past_events = [parsed_events[key] for key in sorted(parsed_events.keys())]
    event_moves: List[float] = []
    # Per-event records (date + actual move) for the "last N earnings" UI panel.
    # Kept UNCLIPPED — this is what the stock actually did, not a stat input.
    event_records: List[Dict[str, Any]] = []

    for event in past_events[-24:]:
        event_ts = pd.Timestamp(event["event_date"]).normalize()
        release_timing = normalize_release_timing(event.get("release_timing"))
        event_loc = int(index_arr.searchsorted(event_ts.to_datetime64(), side="left"))
        if event_loc >= len(index_arr):
            continue
        matched_event_session = pd.Timestamp(index_arr[event_loc]).normalize() == event_ts

        if release_timing == "after market close":
            if matched_event_session:
                pre_loc = event_loc
                post_loc = event_loc + 1
            else:
                pre_loc = event_loc - 1
                post_loc = event_loc
        else:
            pre_loc = event_loc - 1
            post_loc = event_loc

        if pre_loc < 0 or post_loc < 0 or pre_loc >= len(index_arr) or post_loc >= len(index_arr):
            continue

        pre_px = safe_float(price_series.iloc[pre_loc], np.nan)
        post_px = safe_float(price_series.iloc[post_loc], np.nan)
        if not np.isfinite(pre_px) or not np.isfinite(post_px) or pre_px <= 0:
            continue
        move_pct = abs((post_px - pre_px) / pre_px) * 100.0
        if np.isfinite(move_pct):
            event_moves.append(float(move_pct))
            event_records.append({
                "date": event_ts.strftime("%Y-%m-%d"),
                "move_pct": float(move_pct),
                "release_timing": release_timing,
            })

    if event_moves:
        moves = np.array(event_moves, dtype=float)
        if moves.size >= 5:
            low_clip, high_clip = np.percentile(moves, [1.0, 99.0])
            moves = np.clip(moves, low_clip, high_clip)
        return EarningsMoveProfile(
            source="earnings_history",
            earnings_event_count=int(len(event_moves)),
            sample_size=int(moves.size),
            median_move_pct=float(np.median(moves)),
            p90_move_pct=float(np.percentile(moves, 90)),
            avg_last4_move_pct=float(np.mean(moves[-4:])),
            std_move_pct=float(np.std(moves, ddof=1)) if moves.size > 1 else 0.0,
            raw_moves_pct=[float(x) for x in moves.tolist()],
            raw_events=event_records,
        )

    # Fallback to recent daily absolute moves.
    daily_moves = (
        price_series.pct_change().abs().dropna().tail(126).to_numpy(dtype=float) * 100.0
    )
    if daily_moves.size == 0:
        return _EMPTY
    if daily_moves.size >= 5:
        low_clip, high_clip = np.percentile(daily_moves, [1.0, 99.0])
        daily_moves = np.clip(daily_moves, low_clip, high_clip)
    return EarningsMoveProfile(
        source="daily_fallback",
        earnings_event_count=0,
        sample_size=int(daily_moves.size),
        median_move_pct=float(np.median(daily_moves)),
        p90_move_pct=float(np.percentile(daily_moves, 90)),
        avg_last4_move_pct=float(np.mean(daily_moves[-4:])),
        std_move_pct=float(np.std(daily_moves, ddof=1)) if daily_moves.size > 1 else 0.0,
    )
