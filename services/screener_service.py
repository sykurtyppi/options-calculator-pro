"""
Pre-earnings long-vega setup quality screener.

Strategy context
----------------
Enter long-vega position (long call, put, or straddle) 3-10 DTE before earnings.
Exit the day before the event to capture IV expansion and avoid IV crush.
This screener ranks relative setup quality; it does not claim a fixed IV-expansion magnitude.

This screener ranks candidates by setup quality for that specific strategy.
It does NOT score post-earnings outcomes, strangle P&L, or calendar spread payoffs.

Ranking score
-------------
ranking_score ∈ [0, 1] — higher means a better pre-earnings long-vega entry candidate.

Components (weights sum to 1.00):
  iv_entry_score     0.32  IV cheap vs recent realized vol → primary buy signal
  move_history_score 0.25  large historical moves → more IV expansion into event
  ts_score           0.18  near-term not yet elevated vs back → room to expand
  dte_score          0.12  DTE in T-4 to T-8 sweet spot
  sample_score       0.08  rich earnings history → more reliable move estimate
  liquidity_score    0.05  tight spread → executable

The formula is documented in compute_ranking_score().

NOT a calibrated win probability.
See services/calibration_service.py for the score → expected IV expansion mapping.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from services.earnings_event_service import _timing_label_to_full, resolve_upcoming_earnings_event
from services.earnings_vol_snapshot import VolSnapshot, build_vol_snapshot

logger = logging.getLogger(__name__)

# ── Strategy constants ────────────────────────────────────────────────────────

#: Universe screened when none is supplied by the caller.
DEFAULT_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    "ORCL", "CRM", "ADBE", "NFLX", "INTU", "NOW", "QCOM", "KLAC", "LRCX",
    "CDNS", "SNPS", "AMAT", "PANW", "FTNT", "ZS", "DDOG", "NET", "TEAM",
    "WDAY", "SNOW", "OKTA", "TTD", "PYPL", "UBER", "COST", "SBUX", "DIS",
    "LULU", "NKE", "EBAY",
]

#: DTE range targeted by the strategy (entry window).
DTE_MIN_DEFAULT = 3
DTE_MAX_DEFAULT = 14

#: Weeks forward to look for upcoming earnings events.
WEEKS_FORWARD_DEFAULT = 4

#: Per-symbol timeout in seconds when running parallel fetches.
_SYMBOL_TIMEOUT_S = 20

#: Workers for parallel data fetching.
_MAX_WORKERS = 8

# ── Ranking weights ───────────────────────────────────────────────────────────
# Document every weight so changes are explicit.

_W_IV_ENTRY     = 0.32   # IV/RV ratio — lower IV/RV → IV cheap → better entry
_W_MOVE_HISTORY = 0.25   # median historical earnings move → bigger = more room for buildup
_W_TS           = 0.18   # near/back IV ratio — near < back → near not yet elevated
_W_DTE          = 0.12   # DTE alignment with T-4 to T-8 optimal window
_W_SAMPLE       = 0.08   # sample size quality — penalize thin histories
_W_LIQUIDITY    = 0.05   # bid-ask spread — penalize illiquid strikes

assert abs((_W_IV_ENTRY + _W_MOVE_HISTORY + _W_TS + _W_DTE + _W_SAMPLE + _W_LIQUIDITY) - 1.0) < 1e-9


# ── Scoring sub-functions (pure, unit-testable) ───────────────────────────────

def _iv_entry_score(iv_rv_ratio: Optional[float]) -> float:
    """Reward low IV/RV (IV cheap relative to recent realized vol).

    iv_rv = 0.80  → 1.00  (IV significantly below RV — ideal long-vol entry)
    iv_rv = 1.00  → 0.75  (IV at par with RV — decent)
    iv_rv = 1.20  → 0.50  (IV modestly elevated — neutral)
    iv_rv = 1.60  → 0.00  (IV well above RV — avoid long vol here)
    """
    if iv_rv_ratio is None or not np.isfinite(iv_rv_ratio) or iv_rv_ratio <= 0:
        return 0.25  # neutral fallback when data missing
    # Linear: 1.0 at iv_rv=0.80, 0 at iv_rv=1.60; clamped to [0, 1].
    score = 1.0 - (iv_rv_ratio - 0.80) / 0.80
    return float(np.clip(score, 0.0, 1.0))


def _move_history_score(median_move_pct: Optional[float]) -> float:
    """Reward large historical earnings moves.

    Larger historical moves → options market bids up near-term IV more aggressively
    ahead of the event → more expansion potential for long-vega holder.

    median_move =  2 % → 0.00
    median_move =  7 % → 0.50
    median_move = 12 %+→ 1.00
    """
    if median_move_pct is None or not np.isfinite(median_move_pct):
        return 0.20  # neutral fallback
    score = (median_move_pct - 2.0) / 10.0
    return float(np.clip(score, 0.0, 1.0))


def _ts_score(ts_ratio: Optional[float]) -> float:
    """Reward near-term IV not yet elevated relative to back-term.

    ts_ratio = near_iv / back_iv.
    < 1.0 means near cheaper than back → normal term structure, room to build.
    > 1.20 means near already elevated → IV has been bid up, less room.

    ts_ratio = 0.80  → 1.00  (steep normal structure — ideal)
    ts_ratio = 1.00  → 0.60  (flat — OK)
    ts_ratio = 1.30  → 0.00  (inverted — near already expensive)
    """
    if ts_ratio is None or not np.isfinite(ts_ratio) or ts_ratio <= 0:
        return 0.30  # neutral fallback
    score = (1.30 - ts_ratio) / 0.50
    return float(np.clip(score, 0.0, 1.0))


def _dte_score(dte: Optional[int]) -> float:
    """Reward DTE aligned with T-4 to T-8 optimal entry window.

    Uses a Gaussian centered at 6 DTE with σ=3.5.
    DTE= 6 → 1.00
    DTE= 4 → 0.89
    DTE= 3 → 0.75
    DTE=10 → 0.71
    DTE=14 → 0.34
    DTE= 1 → 0.39
    """
    if dte is None or dte <= 0:
        return 0.0
    score = np.exp(-0.5 * ((float(dte) - 6.0) / 3.5) ** 2)
    return float(np.clip(score, 0.0, 1.0))


def _sample_score(sample_size: int) -> float:
    """Reward having ≥ 8 historical earnings events.

    0 events → 0.00
    4 events → 0.50
    8+events → 1.00
    """
    return float(np.clip(sample_size / 8.0, 0.0, 1.0))


def _liquidity_score(spread_pct: Optional[float]) -> float:
    """Reward tight ATM spread (better executability).

    spread = 0 %  → 1.00
    spread = 8 %  → 0.47
    spread = 15%+ → 0.00
    """
    if spread_pct is None or not np.isfinite(spread_pct) or spread_pct < 0:
        return 0.30  # neutral fallback
    score = 1.0 - spread_pct / 15.0
    return float(np.clip(score, 0.0, 1.0))


def compute_ranking_score(
    iv_rv_ratio: Optional[float],
    ts_ratio: Optional[float],
    median_earnings_move_pct: Optional[float],
    sample_size: int,
    dte: Optional[int],
    spread_pct: Optional[float],
) -> float:
    """Compute the pre-earnings long-vega ranking score.

    All inputs are optional; missing values fall back to neutral scores so the
    function always returns a usable float rather than NaN.

    Args:
        iv_rv_ratio: IV / realized-vol ratio from the shared snapshot layer.
            Current consumer uses the snapshot's Yang-Zhang RV baseline.
        ts_ratio: near_expiry_ATM_IV / back_expiry_ATM_IV.  < 1.0 = upward term structure.
        median_earnings_move_pct: median absolute post-earnings move (%) across history.
        sample_size: number of historical earnings events with usable move data.
        dte: calendar days to next earnings announcement.
        spread_pct: ATM call bid-ask spread as % of mid price.

    Returns:
        Scalar in [0, 1]. Higher = stronger setup for pre-earnings long-vega entry.
    """
    return (
        _W_IV_ENTRY     * _iv_entry_score(iv_rv_ratio)
        + _W_MOVE_HISTORY * _move_history_score(median_earnings_move_pct)
        + _W_TS           * _ts_score(ts_ratio)
        + _W_DTE          * _dte_score(dte)
        + _W_SAMPLE       * _sample_score(sample_size)
        + _W_LIQUIDITY    * _liquidity_score(spread_pct)
    )


# ── Per-symbol data helpers ───────────────────────────────────────────────────


def _get_next_earnings(
    ticker: yf.Ticker,
    today: date,
    cutoff: date,
) -> Tuple[Optional[int], str, Optional[date]]:
    """Return (dte, release_timing, earnings_date) for the next upcoming event."""
    try:
        get_dates = getattr(ticker, "get_earnings_dates", None)
        if callable(get_dates):
            edf = get_dates(limit=12)
            if edf is not None and not edf.empty:
                for idx in edf.index:
                    event_date = pd.Timestamp(idx).normalize().date()
                    if today <= event_date <= cutoff:
                        dte = (event_date - today).days
                        # infer timing from timestamp hour
                        ts_raw = pd.Timestamp(idx)
                        total_min = ts_raw.hour * 60 + ts_raw.minute if ts_raw.hour else 0
                        if total_min < 9 * 60 + 30 and total_min > 0:
                            timing = "BMO"
                        elif total_min >= 16 * 60:
                            timing = "AMC"
                        else:
                            timing = "UNKNOWN"
                        return dte, timing, event_date
    except Exception:
        pass

    # Fallback: earnings_dates attribute
    try:
        fallback = getattr(ticker, "earnings_dates", None)
        if fallback is not None and not fallback.empty:
            for idx in fallback.index:
                event_date = pd.Timestamp(idx).normalize().date()
                if today <= event_date <= cutoff:
                    dte = (event_date - today).days
                    return dte, "UNKNOWN", event_date
    except Exception:
        pass

    return None, "UNKNOWN", None


def _release_timing_label_to_full(label: str) -> str:
    norm = str(label or "").upper()
    if norm == "AMC":
        return "after market close"
    if norm == "BMO":
        return "before market open"
    return "unknown"


def _release_timing_full_to_label(value: str) -> str:
    norm = str(value or "").strip().lower()
    if norm == "after market close":
        return "AMC"
    if norm == "before market open":
        return "BMO"
    return "UNKNOWN"


def _collect_yf_past_earnings_events(ticker: yf.Ticker, today: date) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    get_dates = getattr(ticker, "get_earnings_dates", None)
    if callable(get_dates):
        try:
            edf = get_dates(limit=24)
            if edf is not None and not edf.empty:
                for idx in edf.index:
                    ts = pd.Timestamp(idx).tz_localize(None)
                    event_date = ts.normalize().date()
                    if event_date < today:
                        release_timing = _release_timing_label_to_full("UNKNOWN")
                        if ts.hour or ts.minute:
                            if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30):
                                release_timing = "before market open"
                            elif ts.hour >= 16:
                                release_timing = "after market close"
                            else:
                                release_timing = "during market hours"
                        events.append({"event_date": ts.normalize(), "release_timing": release_timing})
        except Exception:
            pass
    if events:
        return events

    fallback = getattr(ticker, "earnings_dates", None)
    if fallback is not None and not fallback.empty:
        try:
            for idx in fallback.index:
                ts = pd.Timestamp(idx).tz_localize(None)
                event_date = ts.normalize().date()
                if event_date < today:
                    events.append({"event_date": ts.normalize(), "release_timing": "unknown"})
        except Exception:
            pass
    return events


def _collect_yf_option_chain_frame(
    ticker: yf.Ticker,
    *,
    as_of: date,
    max_expiries: int = 6,
) -> pd.DataFrame:
    expirations = list(getattr(ticker, "options", []) or [])
    rows: List[Dict[str, Any]] = []
    for expiry in expirations[:max_expiries]:
        try:
            chain = ticker.option_chain(expiry)
        except Exception:
            continue
        for side, frame in (("C", chain.calls), ("P", chain.puts)):
            if frame is None or frame.empty:
                continue
            working = frame.copy()
            for col in ("strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"):
                if col in working.columns:
                    working[col] = pd.to_numeric(working.get(col), errors="coerce")
            for _, row in working.iterrows():
                bid = row.get("bid")
                ask = row.get("ask")
                mid = np.nan
                if pd.notna(bid) and pd.notna(ask) and ask >= bid and ask > 0:
                    mid = (float(bid) + float(ask)) / 2.0
                else:
                    last_price = row.get("lastPrice")
                    if pd.notna(last_price):
                        mid = float(last_price)
                rows.append(
                    {
                        "trade_date": as_of.isoformat(),
                        "expiry": str(expiry)[:10],
                        "call_put": side,
                        "strike": row.get("strike"),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "iv": row.get("impliedVolatility"),
                        "open_interest": row.get("openInterest"),
                        "volume": row.get("volume"),
                    }
                )
    return pd.DataFrame(rows)


def _build_ranked_snapshot(
    symbol: str,
    ticker: yf.Ticker,
    today: date,
    cutoff: date,
    resolved_event: Optional[Any] = None,
) -> Tuple[Optional[VolSnapshot], Optional[str], Optional[date], Optional[str]]:
    # resolved_event may be supplied by the caller to avoid resolving earnings
    # twice (the caller resolves cheaply first to decide whether the expensive
    # snapshot below is even needed). Falls back to resolving here for callers
    # that don't pass it (and the existing unit test).
    if resolved_event is None:
        resolved_event = resolve_upcoming_earnings_event(symbol, today, cutoff, ticker=ticker)
    earnings_date = resolved_event.earnings_date
    release_timing = resolved_event.release_timing
    if earnings_date is None:
        return None, None, None, None

    hist = ticker.history(period="5y", auto_adjust=True)
    if hist is None or hist.empty or len(hist) < 15:
        return None, release_timing, earnings_date, "insufficient price history"

    price_df = hist.reset_index().rename(
        columns={
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    option_chain_df = _collect_yf_option_chain_frame(ticker, as_of=today)
    past_events = _collect_yf_past_earnings_events(ticker, today)
    snapshot = build_vol_snapshot(
        symbol,
        today,
        option_chain_data=option_chain_df,
        earnings_metadata={
            "earnings_date": earnings_date,
            "release_timing": _timing_label_to_full(release_timing),
            "prior_events": past_events,
            "earnings_source_primary": resolved_event.primary_source,
            "earnings_source_confirmed": resolved_event.confirmed_source,
            "earnings_source_confidence": resolved_event.source_confidence,
            "earnings_source_stale": resolved_event.source_stale,
            "release_timing_source": resolved_event.release_timing_source,
        },
        price_data=price_df,
    )
    return snapshot, release_timing, earnings_date, None


def _upcoming_pipeline_row(
    symbol: str, earnings_date: date, dte: int, release_timing: str
) -> Dict[str, Any]:
    """A lightweight row for an upcoming-but-not-yet-enterable earnings event.

    Carries only the cheap-to-resolve fields (symbol, earnings date, DTE,
    timing); scored/IV fields are null. These are the screener's forward
    pipeline — what's coming and when it enters the window — so the table is
    never just a blank '0 setups' between earnings seasons.
    """
    return {
        "symbol": symbol,
        "earnings_date": str(earnings_date),
        "days_to_earnings": dte,
        "release_timing": release_timing,
        "iv30": None,
        "rv30": None,
        "iv_rv_ratio": None,
        "term_structure_ratio": None,
        "median_earnings_move_pct": None,
        "p90_earnings_move_pct": None,
        "sample_size": None,
        "avg_spread_pct": None,
        "ranking_score": None,
        "in_entry_window": False,
        "upcoming": True,
        "earnings_source_primary": None,
        "earnings_source_confirmed": None,
        "earnings_source_confidence": None,
        "earnings_source_stale": None,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


# ── Per-symbol orchestration ──────────────────────────────────────────────────

def _screen_one_symbol_ranked(
    symbol: str,
    today: date,
    cutoff: date,
    dte_max: int = DTE_MAX_DEFAULT,
) -> Optional[Dict[str, Any]]:
    """Fetch data and compute ranking signal for a single symbol.

    Returns None if no upcoming earnings found within the cutoff window.
    Returns a lightweight 'upcoming' row (no score, no IV) for symbols whose
    earnings are beyond dte_max — the pipeline of what's coming — WITHOUT paying
    for the expensive 5y-history + option-chain snapshot. Returns a dict with an
    'error' key if data is partial.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Resolve earnings cheaply first; only entry-window symbols (dte <=
        # dte_max) get the heavy snapshot. Scanning the full universe with the
        # heavy per-symbol fetch was getting yfinance-rate-limited to ~0 results.
        resolved_event = resolve_upcoming_earnings_event(symbol, today, cutoff, ticker=ticker)
        earnings_date = resolved_event.earnings_date
        if earnings_date is None:
            return None
        dte = (earnings_date - today).days
        if dte > dte_max:
            # resolved_event.release_timing is already a short label (AMC/BMO/UNKNOWN),
            # so normalize directly rather than via the full-form converter.
            timing = (resolved_event.release_timing or "UNKNOWN").upper()
            if timing not in ("AMC", "BMO"):
                timing = "UNKNOWN"
            return _upcoming_pipeline_row(symbol, earnings_date, dte, timing)

        snapshot, release_timing, earnings_date, snapshot_error = _build_ranked_snapshot(
            symbol, ticker, today, cutoff, resolved_event=resolved_event
        )
        if snapshot is None and earnings_date is None:
            return None
        if snapshot_error:
            return {
                "symbol": symbol,
                "earnings_date": str(earnings_date),
                "days_to_earnings": (earnings_date - today).days if earnings_date is not None else None,
                "release_timing": release_timing or "UNKNOWN",
                "ranking_score": 0.0,
                "error": snapshot_error,
            }
        if snapshot is None:
            return None

        dte = snapshot.days_to_earnings
        if dte is None:
            return None
        release_timing = _release_timing_full_to_label(snapshot.release_timing or release_timing or "UNKNOWN")
        iv_front = snapshot.near_term_atm_iv
        rv30 = snapshot.rv30_yang_zhang
        iv_rv = round(snapshot.iv_rv_yz, 3) if snapshot.iv_rv_yz is not None else None
        ts_ratio = round(snapshot.near_back_iv_ratio, 4) if snapshot.near_back_iv_ratio is not None else None
        spread_pct = round(snapshot.near_term_spread_pct, 2) if snapshot.near_term_spread_pct is not None else None
        median_move_pct = (
            round(snapshot.historical_median_move_pct, 2)
            if snapshot.historical_median_move_pct is not None
            else None
        )
        p90_move_pct = (
            round(snapshot.historical_p90_move_pct, 2)
            if snapshot.historical_p90_move_pct is not None
            else None
        )
        sample_size = int(snapshot.historical_event_count or 0)

        # Ranking score
        ranking_score = compute_ranking_score(
            iv_rv_ratio=iv_rv,
            ts_ratio=ts_ratio,
            median_earnings_move_pct=median_move_pct,
            sample_size=sample_size,
            dte=dte,
            spread_pct=spread_pct,
        )

        return {
            "symbol": symbol,
            "earnings_date": str(earnings_date),
            "days_to_earnings": dte,
            "release_timing": release_timing,
            "iv30": round(float(iv_front), 4) if iv_front is not None else None,
            "rv30": round(float(rv30), 4) if rv30 is not None and np.isfinite(rv30) else None,
            "iv_rv_ratio": iv_rv,
            "term_structure_ratio": ts_ratio,
            "median_earnings_move_pct": median_move_pct,
            "p90_earnings_move_pct": p90_move_pct,
            "sample_size": sample_size,
            "avg_spread_pct": spread_pct,
            "ranking_score": round(ranking_score, 4),
            "in_entry_window": (DTE_MIN_DEFAULT <= dte <= DTE_MAX_DEFAULT),
            "earnings_source_primary": snapshot.earnings_source_primary,
            "earnings_source_confirmed": snapshot.earnings_source_confirmed,
            "earnings_source_confidence": snapshot.earnings_source_confidence,
            "earnings_source_stale": snapshot.earnings_source_stale,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }

    except Exception as exc:
        logger.debug("Ranked screener: %s failed — %s", symbol, exc)
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def build_ranked_screener(
    *,
    symbols: Optional[List[str]] = None,
    dte_min: int = DTE_MIN_DEFAULT,
    dte_max: int = DTE_MAX_DEFAULT,
    min_sample_size: int = 0,
    release_filter: Optional[str] = None,   # "AMC" | "BMO" | None = all
    weeks: int = WEEKS_FORWARD_DEFAULT,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Build a ranked pre-earnings long-vega screener table.

    Runs per-symbol fetches in parallel with a per-symbol timeout.
    Returns all symbols with upcoming earnings; filters and ranking applied.

    Args:
        symbols:        Ticker universe. Defaults to DEFAULT_UNIVERSE.
        dte_min:        Minimum DTE to include in active window filter.
        dte_max:        Maximum DTE to include in active window filter.
        min_sample_size: Exclude rows with fewer earnings history events.
        release_filter: Filter to "AMC" or "BMO" only; None = show all.
        weeks:          Forward window for upcoming earnings search.
        today:          Reference date for DTE calculation (defaults to date.today()).

    Returns:
        Dict with 'rows' sorted by ranking_score descending, plus summary stats.
    """
    as_of = today or date.today()
    cutoff = as_of + timedelta(weeks=weeks)
    universe = [s.upper() for s in (symbols or DEFAULT_UNIVERSE)]

    raw_rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_screen_one_symbol_ranked, sym, as_of, cutoff, dte_max): sym
            for sym in universe
        }
        for future in as_completed(futures, timeout=_SYMBOL_TIMEOUT_S * len(universe)):
            try:
                result = future.result(timeout=_SYMBOL_TIMEOUT_S)
                if result is not None:
                    raw_rows.append(result)
            except (FuturesTimeout, Exception):
                pass

    # Apply filters
    filtered: List[Dict[str, Any]] = []
    for row in raw_rows:
        if row.get("error"):
            # Include errored rows but at the bottom so the user sees data coverage
            filtered.append(row)
            continue
        dte_val = row.get("days_to_earnings")
        if dte_val is None:
            continue
        # min-sample filter applies only to SCORED rows; upcoming-pipeline rows
        # carry no sample (sample_size is None) and must not be dropped by it.
        if (
            not row.get("upcoming")
            and min_sample_size > 0
            and (row.get("sample_size") or 0) < min_sample_size
        ):
            continue
        if release_filter and row.get("release_timing") not in (release_filter, "UNKNOWN"):
            continue
        filtered.append(row)

    # Sort order, top to bottom:
    #   1. Scored rows (in/near the entry window) by ranking_score desc
    #   2. Upcoming-pipeline rows (no score yet) by DTE asc — "what's coming, soonest first"
    #   3. Errored rows last, so the user still sees data-coverage gaps
    # Symbol is the deterministic final tie-breaker; without it, rows with equal
    # keys sort by thread-completion order and screener output is non-reproducible.
    scored = [r for r in filtered if not r.get("error") and r.get("ranking_score") is not None]
    upcoming = [r for r in filtered if not r.get("error") and r.get("ranking_score") is None]
    errored = [r for r in filtered if r.get("error")]

    scored.sort(key=lambda r: (-r["ranking_score"], r.get("days_to_earnings", 9999), r.get("symbol", "")))
    upcoming.sort(key=lambda r: (r.get("days_to_earnings", 9999), r.get("symbol", "")))
    errored.sort(key=lambda r: str(r.get("symbol", "")))

    rows = scored + upcoming + errored

    # Label entry window (scored rows only — upcoming rows are by definition beyond it)
    in_window = sum(
        1 for r in scored
        if r.get("days_to_earnings") is not None
        and dte_min <= r["days_to_earnings"] <= dte_max
    )
    upcoming_count = len(upcoming)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(as_of),
        "universe_size": len(universe),
        "rows_returned": len(rows),
        "in_entry_window": in_window,
        "upcoming_count": upcoming_count,
        "ranking_weights": {
            "iv_entry": _W_IV_ENTRY,
            "move_history": _W_MOVE_HISTORY,
            "term_structure": _W_TS,
            "dte_alignment": _W_DTE,
            "sample_quality": _W_SAMPLE,
            "liquidity": _W_LIQUIDITY,
        },
        "strategy_note": (
            "Pre-earnings long-vega setup quality ranking. "
            "Enter 3-10 DTE, exit before event. "
            "Ranking score is NOT a calibrated win rate."
        ),
        "rows": rows,
    }
