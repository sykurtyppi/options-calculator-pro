#!/usr/bin/env python3
"""
replay_multi_structure_smoke.py
===============================
Multi-structure replay across all 4 supported structures, directly from
T9 chain data.

This is the natural extension of replay_otm_strangle_smoke.py (PR-Z). It
loads each event's entry/exit chains + VolSnapshot + scorecards exactly
once, then tries to construct and price each of:

    atm_straddle    long  ATM call + long  ATM put  (near-term expiry)
    otm_strangle    long  OTM call + long  OTM put  (near-term expiry)
    call_calendar   short ATM call front + long ATM call back  (same strike)
    put_calendar    short ATM put  front + long ATM put  back  (same strike)

Each successful structure becomes one observation written idempotently to:

    structure_prior_store    (all 4 structures)
    calibration_service      (long-vol structures only: straddle, strangle)

The calibration service is intentionally long-vega-only — its semantic is
"score → realized IV expansion for long-vol setups." Calendars don't fit
that mapping (their P&L is driven by IV crush of the FRONT leg, opposite
sign), so they only update the per-structure prior store.

Safety
------
- --target=tmp (default) routes ALL writes to tmp/replay_run_<utc>/
- Idempotent: observation_id = f"replay_{symbol}_{event}_{structure}"
- --dry-run prints the resolved plan without touching chain data or stores
- Each structure attempt is wrapped in try/skip — one structure failing on
  an event does not affect the others

Usage
-----
::

    # Inspect plan:
    python scripts/replay_multi_structure_smoke.py --dry-run

    # Default (4 symbols × 4 structures × 14 events ≈ 224 attempts):
    python scripts/replay_multi_structure_smoke.py

    # Subset of structures:
    python scripts/replay_multi_structure_smoke.py --structures atm_straddle,call_calendar

    # Wider universe:
    python scripts/replay_multi_structure_smoke.py --symbols all
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING)

T9_FEATURES_ROOT = Path(
    "/Volumes/T9/market_data/research/options_features_eod"
)
T9_EARNINGS_DB = Path(
    "/Volumes/T9/market_data/research/options_calculator_pro/earnings_calendar/historical_earnings.sqlite"
)

EARNINGS_SYMBOLS = ["AAPL", "AMD", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "ORCL", "TSLA"]
DEFAULT_SMOKE_SYMBOLS = ["AAPL", "AMZN", "MSFT", "NVDA"]

SUPPORTED_STRUCTURES = ("atm_straddle", "otm_strangle", "call_calendar", "put_calendar")
LONG_VOL_STRUCTURES = {"atm_straddle", "otm_strangle"}  # write to calibration store

ENTRY_LOOKBACK_DAYS = 2
ENTRY_MAX_SEARCH = 5
EXIT_FORWARD_DAYS = 1
EXIT_MAX_SEARCH = 4

OTM_PCT_DEFAULT = 0.05  # ±5% OTM wings for strangle

# Minimum entry capital for which a return % is meaningful.
# Calendars can have very small (or even negative) entry values when front IV
# is pumped above back IV; we skip those rather than report huge swings.
MIN_ENTRY_DEBIT = 0.10  # dollars per share — i.e. 10c per leg combined


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    side: str          # "long" or "short"
    expiry: datetime.date
    call_put: str      # "C" or "P"
    strike: float


@dataclass
class StructureAttempt:
    """One (event × structure) attempt. status is 'ok' or 'skipped:<reason>'."""
    structure: str
    status: str = ""
    setup_score: Optional[float] = None
    legs: Optional[List[Leg]] = None
    entry_value: Optional[float] = None
    exit_value: Optional[float] = None
    realized_return_pct: Optional[float] = None
    entry_iv_avg: Optional[float] = None
    exit_iv_avg: Optional[float] = None
    realized_expansion_pct: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventResult:
    symbol: str
    event_date: datetime.date
    release_timing: str
    status: str = ""  # event-level status: "ok"|"skipped:no_entry_chain"|...
    entry_date: Optional[datetime.date] = None
    exit_date: Optional[datetime.date] = None
    selector_pick: Optional[str] = None
    attempts: Dict[str, StructureAttempt] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# T9 / yfinance / chain helpers — copied/aligned with replay_otm_strangle_smoke
# ---------------------------------------------------------------------------

def _load_earnings_events(db_path: Path, symbols: List[str]) -> List[Tuple[str, datetime.date, str]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Earnings DB not found at {db_path}")
    placeholders = ",".join("?" * len(symbols))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""SELECT symbol, event_date, release_timing
                FROM historical_earnings_events
                WHERE symbol IN ({placeholders})
                ORDER BY symbol, event_date""",
            symbols,
        ).fetchall()
    return [(sym, datetime.date.fromisoformat(d), tm) for sym, d, tm in rows]


def _chain_glob(symbol: str) -> str:
    return f"{T9_FEATURES_ROOT}/underlying_symbol={symbol}/**/*.parquet"


def _find_nearest_chain_date(
    con,
    symbol: str,
    target: datetime.date,
    direction: str,
    max_days: int,
) -> Optional[datetime.date]:
    pattern = _chain_glob(symbol)
    if direction == "before":
        q = f"""SELECT DISTINCT trade_date FROM read_parquet('{pattern}', hive_partitioning=true)
                WHERE trade_date < ? AND trade_date >= ?
                ORDER BY trade_date DESC LIMIT 1"""
        params = [target, target - datetime.timedelta(days=max_days)]
    elif direction == "after":
        q = f"""SELECT DISTINCT trade_date FROM read_parquet('{pattern}', hive_partitioning=true)
                WHERE trade_date > ? AND trade_date <= ?
                ORDER BY trade_date ASC LIMIT 1"""
        params = [target, target + datetime.timedelta(days=max_days)]
    else:
        raise ValueError(direction)
    row = con.execute(q, params).fetchone()
    if not row:
        return None
    val = row[0]
    return val.date() if hasattr(val, "date") else val


def _load_chain(con, symbol: str, trade_date: datetime.date):
    import pandas as pd
    pattern = _chain_glob(symbol)
    df = con.execute(
        f"""SELECT trade_date, expiry, call_put, strike, bid, ask, mid, iv,
                   underlying_price, dte, delta, open_interest, volume, spread_pct
            FROM read_parquet('{pattern}', hive_partitioning=true)
            WHERE trade_date = ?
              AND mid IS NOT NULL AND mid > 0
              AND iv IS NOT NULL AND iv > 0""",
        [trade_date],
    ).fetchdf()
    if not df.empty:
        if hasattr(df["trade_date"].iloc[0], "date"):
            df["trade_date"] = df["trade_date"].dt.date
        if hasattr(df["expiry"].iloc[0], "date"):
            df["expiry"] = df["expiry"].dt.date
    return df


def _fetch_yfinance_ohlc(symbol: str, end_date: datetime.date, days: int = 120):
    import pandas as pd
    import yfinance as yf
    start = end_date - datetime.timedelta(days=days)
    end_exclusive = end_date + datetime.timedelta(days=1)
    hist = yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=False,
    )
    if hist.empty:
        return None
    hist = hist.reset_index()
    if pd.api.types.is_datetime64_any_dtype(hist["Date"]):
        try:
            hist["Date"] = hist["Date"].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    return hist.rename(columns={"Date": "trade_date"})


# ---------------------------------------------------------------------------
# Leg construction (one per structure)
# ---------------------------------------------------------------------------

def _strikes_in_both_chains(entry_chain, exit_chain, expiry, call_put):
    """Return the set of strikes for (expiry, call_put) present in BOTH chains."""
    e = entry_chain[
        (entry_chain["expiry"] == expiry) & (entry_chain["call_put"] == call_put)
    ]["strike"]
    x = exit_chain[
        (exit_chain["expiry"] == expiry) & (exit_chain["call_put"] == call_put)
    ]["strike"]
    return set(e) & set(x)


def _pick_strike_closest_to(target: float, candidates) -> Optional[float]:
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s - target))


def _common_expiries_after(entry_chain, exit_chain, event_date: datetime.date) -> List[datetime.date]:
    """Expiries > event_date present in BOTH chains, sorted ascending."""
    entry_e = {e for e in entry_chain["expiry"] if e > event_date}
    exit_e = {e for e in exit_chain["expiry"]}
    return sorted(entry_e & exit_e)


def build_atm_straddle_legs(entry_chain, exit_chain, event_date, exit_date, underlying):
    """Long ATM call + long ATM put at the same near-term expiry."""
    expiries = [e for e in _common_expiries_after(entry_chain, exit_chain, event_date) if e >= exit_date]
    for expiry in expiries:
        call_strikes = _strikes_in_both_chains(entry_chain, exit_chain, expiry, "C")
        put_strikes = _strikes_in_both_chains(entry_chain, exit_chain, expiry, "P")
        shared = call_strikes & put_strikes
        if not shared:
            continue
        atm = _pick_strike_closest_to(underlying, shared)
        if atm is None:
            continue
        return [
            Leg(side="long", expiry=expiry, call_put="C", strike=atm),
            Leg(side="long", expiry=expiry, call_put="P", strike=atm),
        ], {"near_term_expiry": str(expiry), "atm_strike": atm}
    return None, {"reason": "no_atm_expiry_with_shared_strike"}


def build_otm_strangle_legs(entry_chain, exit_chain, event_date, exit_date, underlying, otm_pct):
    """Long OTM call (+otm_pct) + long OTM put (-otm_pct), same near-term expiry."""
    target_call = underlying * (1 + otm_pct)
    target_put = underlying * (1 - otm_pct)
    expiries = [e for e in _common_expiries_after(entry_chain, exit_chain, event_date) if e >= exit_date]
    for expiry in expiries:
        call_strikes = _strikes_in_both_chains(entry_chain, exit_chain, expiry, "C")
        put_strikes = _strikes_in_both_chains(entry_chain, exit_chain, expiry, "P")
        if not call_strikes or not put_strikes:
            continue
        call_k = _pick_strike_closest_to(target_call, call_strikes)
        put_k = _pick_strike_closest_to(target_put, put_strikes)
        if call_k is None or put_k is None:
            continue
        return [
            Leg(side="long", expiry=expiry, call_put="C", strike=call_k),
            Leg(side="long", expiry=expiry, call_put="P", strike=put_k),
        ], {"near_term_expiry": str(expiry), "call_strike": call_k, "put_strike": put_k}
    return None, {"reason": "no_strangle_expiry_with_shared_strikes"}


def _build_calendar_legs(entry_chain, exit_chain, event_date, exit_date, underlying, call_put: str):
    """Short front, long back. Same strike at both expiries; both expiries in
    both chains; strike present in BOTH expiries in BOTH chains.

    Picks:
      front = smallest expiry > event_date
      back  = first expiry > front + 14 days  (to give meaningful term-structure)
    """
    expiries = [e for e in _common_expiries_after(entry_chain, exit_chain, event_date) if e >= exit_date]
    if len(expiries) < 2:
        return None, {"reason": "fewer_than_two_common_expiries"}
    for i, front in enumerate(expiries):
        for back in expiries[i + 1:]:
            if (back - front).days < 14:
                continue
            front_strikes = _strikes_in_both_chains(entry_chain, exit_chain, front, call_put)
            back_strikes = _strikes_in_both_chains(entry_chain, exit_chain, back, call_put)
            shared = front_strikes & back_strikes
            if not shared:
                continue
            atm = _pick_strike_closest_to(underlying, shared)
            if atm is None:
                continue
            return [
                Leg(side="short", expiry=front, call_put=call_put, strike=atm),
                Leg(side="long", expiry=back, call_put=call_put, strike=atm),
            ], {
                "front_expiry": str(front),
                "back_expiry": str(back),
                "atm_strike": atm,
                "front_to_back_days": (back - front).days,
            }
    return None, {"reason": "no_valid_front_back_pair_with_shared_strike"}


def build_call_calendar_legs(entry_chain, exit_chain, event_date, exit_date, underlying):
    return _build_calendar_legs(entry_chain, exit_chain, event_date, exit_date, underlying, "C")


def build_put_calendar_legs(entry_chain, exit_chain, event_date, exit_date, underlying):
    return _build_calendar_legs(entry_chain, exit_chain, event_date, exit_date, underlying, "P")


# ---------------------------------------------------------------------------
# Leg pricing
# ---------------------------------------------------------------------------

def _lookup_leg(chain, leg: Leg):
    row = chain[
        (chain["expiry"] == leg.expiry)
        & (chain["call_put"] == leg.call_put)
        & (chain["strike"] == leg.strike)
    ]
    if row.empty:
        return None
    return row.iloc[0]


def price_legs(legs: List[Leg], chain) -> Optional[Dict[str, float]]:
    """Look up each leg in *chain*. Return signed net value (long − short)
    and the LONG-side IV average (for long-vol expansion measurement) plus
    the per-leg IV breakdown.

    Returns None if any leg can't be looked up.
    """
    long_ivs = []
    short_ivs = []
    net_value = 0.0
    for leg in legs:
        row = _lookup_leg(chain, leg)
        if row is None:
            return None
        mid = float(row["mid"])
        iv = float(row["iv"])
        if leg.side == "long":
            net_value += mid
            long_ivs.append(iv)
        elif leg.side == "short":
            net_value -= mid
            short_ivs.append(iv)
        else:
            raise ValueError(f"unknown side: {leg.side}")
    return {
        "net_value": net_value,
        "long_iv_avg": sum(long_ivs) / len(long_ivs) if long_ivs else 0.0,
        "short_iv_avg": sum(short_ivs) / len(short_ivs) if short_ivs else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-structure replay
# ---------------------------------------------------------------------------

def attempt_structure(
    structure: str,
    scorecard,
    entry_chain,
    exit_chain,
    event_date,
    exit_date,
    underlying,
    otm_pct,
) -> StructureAttempt:
    attempt = StructureAttempt(structure=structure)

    if not scorecard.eligible:
        flags = ",".join(list(scorecard.eligibility_flags)[:3])
        attempt.status = f"skipped:ineligible:{flags}"
        return attempt
    attempt.setup_score = float(scorecard.composite_structure_score)

    # Build legs per structure
    if structure == "atm_straddle":
        legs, details = build_atm_straddle_legs(entry_chain, exit_chain, event_date, exit_date, underlying)
    elif structure == "otm_strangle":
        legs, details = build_otm_strangle_legs(entry_chain, exit_chain, event_date, exit_date, underlying, otm_pct)
    elif structure == "call_calendar":
        legs, details = build_call_calendar_legs(entry_chain, exit_chain, event_date, exit_date, underlying)
    elif structure == "put_calendar":
        legs, details = build_put_calendar_legs(entry_chain, exit_chain, event_date, exit_date, underlying)
    else:
        attempt.status = f"skipped:unknown_structure:{structure}"
        return attempt

    if legs is None:
        attempt.status = f"skipped:no_legs:{details.get('reason', 'unknown')}"
        attempt.details = details
        return attempt
    attempt.legs = legs
    attempt.details = dict(details)

    entry_priced = price_legs(legs, entry_chain)
    if entry_priced is None:
        attempt.status = "skipped:entry_pricing_failed"
        return attempt
    exit_priced = price_legs(legs, exit_chain)
    if exit_priced is None:
        attempt.status = "skipped:exit_pricing_failed"
        return attempt

    entry_value = entry_priced["net_value"]
    exit_value = exit_priced["net_value"]
    if entry_value < MIN_ENTRY_DEBIT:
        # For calendars this means front IV ≥ back IV (event so close that
        # front premium swamps back). Skip rather than report misleading %.
        attempt.status = f"skipped:entry_below_min_debit:{entry_value:+.3f}"
        return attempt

    attempt.entry_value = entry_value
    attempt.exit_value = exit_value
    attempt.realized_return_pct = (exit_value - entry_value) / entry_value * 100.0

    # IV expansion: use LONG-leg IV avg for long-vol structures (straddle,
    # strangle). For calendars, use SHORT-leg (front-month) IV change since
    # that's the IV that actually moves through the event.
    if structure in LONG_VOL_STRUCTURES:
        entry_iv = entry_priced["long_iv_avg"]
        exit_iv = exit_priced["long_iv_avg"]
    else:
        entry_iv = entry_priced["short_iv_avg"]
        exit_iv = exit_priced["short_iv_avg"]

    if entry_iv <= 0:
        attempt.status = "skipped:zero_entry_iv"
        return attempt

    attempt.entry_iv_avg = entry_iv
    attempt.exit_iv_avg = exit_iv
    attempt.realized_expansion_pct = (exit_iv - entry_iv) / entry_iv * 100.0

    attempt.details.update({
        "entry_value": entry_value,
        "exit_value": exit_value,
        "entry_iv": entry_iv,
        "exit_iv": exit_iv,
    })
    attempt.status = "ok"
    return attempt


def process_event(
    con,
    symbol: str,
    event_date: datetime.date,
    release_timing: str,
    otm_pct: float,
    structures: List[str],
) -> EventResult:
    from services.earnings_vol_snapshot import build_vol_snapshot
    from services.structure_scorecard import build_structure_scorecards
    from services.structure_selector import select_best_structure

    result = EventResult(symbol=symbol, event_date=event_date, release_timing=release_timing)

    target_entry = event_date - datetime.timedelta(days=ENTRY_LOOKBACK_DAYS)
    entry_date = _find_nearest_chain_date(con, symbol, target_entry, "before", ENTRY_MAX_SEARCH)
    if entry_date is None:
        entry_date = _find_nearest_chain_date(con, symbol, target_entry + datetime.timedelta(days=1), "before", ENTRY_MAX_SEARCH)
    if entry_date is None:
        result.status = "skipped:no_entry_chain"
        return result

    target_exit = event_date + datetime.timedelta(days=EXIT_FORWARD_DAYS)
    exit_date = _find_nearest_chain_date(con, symbol, target_exit, "after", EXIT_MAX_SEARCH)
    if exit_date is None:
        exit_date = _find_nearest_chain_date(con, symbol, target_exit - datetime.timedelta(days=1), "after", EXIT_MAX_SEARCH)
    if exit_date is None:
        result.status = "skipped:no_exit_chain"
        return result

    result.entry_date = entry_date
    result.exit_date = exit_date

    entry_chain = _load_chain(con, symbol, entry_date)
    if entry_chain.empty:
        result.status = "skipped:empty_entry_chain"
        return result
    exit_chain = _load_chain(con, symbol, exit_date)
    if exit_chain.empty:
        result.status = "skipped:empty_exit_chain"
        return result

    ohlc = _fetch_yfinance_ohlc(symbol, entry_date)
    if ohlc is None or ohlc.empty:
        result.status = "skipped:no_ohlc"
        return result

    earnings_metadata = {
        "symbol": symbol,
        "earnings_date": event_date.isoformat(),
        "release_timing": release_timing,
    }

    try:
        snapshot = build_vol_snapshot(
            symbol, entry_date,
            option_chain_data=entry_chain,
            earnings_metadata=earnings_metadata,
            price_data=ohlc,
        )
    except Exception as exc:
        result.status = f"skipped:vol_snapshot_failed:{type(exc).__name__}"
        return result

    try:
        scorecards = build_structure_scorecards(snapshot)
        selected = select_best_structure(snapshot, scorecards)
        result.selector_pick = selected.best_structure
    except Exception as exc:
        result.status = f"skipped:selector_failed:{type(exc).__name__}"
        return result

    scorecards_by_structure = {c.structure: c for c in scorecards}
    underlying = float(entry_chain["underlying_price"].iloc[0])

    for structure in structures:
        card = scorecards_by_structure.get(structure)
        if card is None:
            result.attempts[structure] = StructureAttempt(
                structure=structure, status="skipped:no_scorecard"
            )
            continue
        result.attempts[structure] = attempt_structure(
            structure=structure,
            scorecard=card,
            entry_chain=entry_chain,
            exit_chain=exit_chain,
            event_date=event_date,
            exit_date=exit_date,
            underlying=underlying,
            otm_pct=otm_pct,
        )

    result.status = "ok"
    return result


# ---------------------------------------------------------------------------
# Target / stores
# ---------------------------------------------------------------------------

def _apply_target_env(target: str, run_timestamp: str) -> Path:
    if target == "production":
        return Path.home() / ".options_calculator_pro"
    tmp_root = REPO_ROOT / "tmp" / f"replay_multi_{run_timestamp}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / "outcomes").mkdir(exist_ok=True)
    (tmp_root / "calibration").mkdir(exist_ok=True)
    (tmp_root / "priors").mkdir(exist_ok=True)
    os.environ["OPTIONS_CALCULATOR_OUTCOMES_PATH"] = str(tmp_root / "outcomes" / "outcome_store.sqlite")
    os.environ["OPTIONS_CALCULATOR_CALIBRATION_PATH"] = str(tmp_root / "calibration" / "iv_expansion.json")
    os.environ["OPTIONS_CALCULATOR_PRIORS_PATH"] = str(tmp_root / "priors" / "structure_priors.json")
    return tmp_root


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SMOKE_SYMBOLS),
        help=f"Comma-separated symbols, or 'all' for {EARNINGS_SYMBOLS}. Default: {','.join(DEFAULT_SMOKE_SYMBOLS)}",
    )
    parser.add_argument(
        "--structures",
        default=",".join(SUPPORTED_STRUCTURES),
        help=f"Subset of {SUPPORTED_STRUCTURES} to replay. Default: all 4.",
    )
    parser.add_argument(
        "--target", choices=["tmp", "production"], default="tmp",
        help="Where to write store updates (default: tmp)",
    )
    parser.add_argument("--otm-pct", type=float, default=OTM_PCT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.symbols == "all":
        symbols = list(EARNINGS_SYMBOLS)
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    structures = [s.strip() for s in args.structures.split(",") if s.strip()]
    unknown = [s for s in structures if s not in SUPPORTED_STRUCTURES]
    if unknown:
        print(f"ERROR: unknown structures {unknown}. Valid: {SUPPORTED_STRUCTURES}", file=sys.stderr)
        return 2

    if not T9_FEATURES_ROOT.exists():
        print(f"ERROR: T9 features root not found: {T9_FEATURES_ROOT}", file=sys.stderr)
        return 2
    if not T9_EARNINGS_DB.exists():
        print(f"ERROR: Earnings DB not found: {T9_EARNINGS_DB}", file=sys.stderr)
        return 2

    events = _load_earnings_events(T9_EARNINGS_DB, symbols)
    if args.limit is not None:
        events = events[: args.limit]

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = _apply_target_env(args.target, timestamp)

    print()
    print("=" * 72)
    print("  MULTI-STRUCTURE REPLAY — direct T9 chains → calibration/prior")
    print("=" * 72)
    print(f"  Symbols:        {len(symbols)} — {','.join(symbols)}")
    print(f"  Structures:     {','.join(structures)}")
    print(f"  Events:         {len(events)}  (× {len(structures)} structures = up to {len(events)*len(structures)} attempts)")
    print(f"  OTM target:     ±{args.otm_pct*100:.1f}%")
    print(f"  Target:         {args.target}")
    print(f"  Target dir:     {target_dir}")
    print(f"  Dry run:        {args.dry_run}")
    print()

    if args.dry_run:
        seen = set()
        print("Sample events (one per symbol):")
        for sym, ed, tm in events:
            if sym in seen:
                continue
            seen.add(sym)
            print(f"  {sym}  {ed.isoformat()}  {tm}")
        print()
        print("DRY RUN — exiting before any chain query or write.")
        return 0

    import duckdb
    con = duckdb.connect(":memory:")

    from services.calibration_service import get_calibration
    from services.structure_prior_store import get_structure_prior_store
    calibration = get_calibration()
    prior_store = get_structure_prior_store()

    print(f"  Calibration store path: {calibration._path}")
    print(f"  Prior store path:       {prior_store._path}")
    print()

    results: List[EventResult] = []
    # status_counts[structure] = {status: n}
    status_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    event_status_counts: Dict[str, int] = defaultdict(int)

    for idx, (symbol, event_date, release_timing) in enumerate(events, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(events):
            print(f"  [{idx}/{len(events)}] processing {symbol} {event_date}…")
        result = process_event(con, symbol, event_date, release_timing, args.otm_pct, structures)
        results.append(result)
        event_status_counts[result.status] += 1

        # If the event itself failed pre-snapshot, mark all structure
        # attempts as inherited-skipped for accounting.
        if result.status != "ok":
            for s in structures:
                status_counts[s][f"event_{result.status}"] += 1
            continue

        for structure, attempt in result.attempts.items():
            status_counts[structure][attempt.status] += 1
            if attempt.status != "ok":
                continue
            observation_id = f"replay_{symbol}_{event_date.isoformat()}_{structure}"
            try:
                if structure in LONG_VOL_STRUCTURES:
                    calibration.update(
                        attempt.setup_score or 0.0,
                        attempt.realized_expansion_pct,
                        observation_id=observation_id,
                        source_type="replay",
                        observation_date=result.exit_date,
                    )
                prior_store.update(
                    structure=structure,
                    realized_return_pct=attempt.realized_return_pct,
                    realized_expansion_pct=attempt.realized_expansion_pct,
                    source_type="replay",
                    observation_date=result.exit_date,
                    observation_id=observation_id,
                )
            except Exception as exc:
                attempt.status = f"write_failed:{type(exc).__name__}"
                attempt.details["write_error"] = str(exc)
                status_counts[structure][attempt.status] += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  REPLAY SUMMARY")
    print("=" * 72)
    print(f"  Total events processed: {len(results)}")
    print(f"  Event-level outcomes:")
    for status, n in sorted(event_status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:35s} {n}")
    print()

    summary_per_structure: Dict[str, Dict[str, Any]] = {}
    for structure in structures:
        oks = [
            r.attempts[structure]
            for r in results
            if structure in r.attempts and r.attempts[structure].status == "ok"
        ]
        print(f"  ── {structure} ──")
        sc = status_counts[structure]
        for status, n in sorted(sc.items(), key=lambda x: -x[1]):
            print(f"    {status:45s} {n}")
        if oks:
            mean_return = sum(a.realized_return_pct for a in oks) / len(oks)
            mean_expansion = sum(a.realized_expansion_pct for a in oks) / len(oks)
            wins = sum(1 for a in oks if a.realized_return_pct > 0)
            summary_per_structure[structure] = {
                "n": len(oks),
                "mean_return_pct": mean_return,
                "mean_expansion_pct": mean_expansion,
                "win_rate": wins / len(oks),
            }
            print(f"    observations: {len(oks)}    mean return: {mean_return:+7.2f}%    "
                  f"mean expansion: {mean_expansion:+7.2f}%    win rate: {wins}/{len(oks)} = {100*wins/len(oks):.0f}%")
        else:
            summary_per_structure[structure] = {"n": 0}
            print(f"    observations: 0 — nothing written for this structure")
        print()

    print(f"  Calibration store n after: {calibration._n()}")
    print(f"  Calibration store path:    {calibration._path}")
    print()

    summary_dir = target_dir if args.target == "tmp" else REPO_ROOT / "exports" / "reports" / "replay_multi_structure"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"replay_multi_structure_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "symbols": symbols,
                "structures": structures,
                "events_total": len(results),
                "event_status_counts": dict(event_status_counts),
                "structure_status_counts": {k: dict(v) for k, v in status_counts.items()},
                "summary_per_structure": summary_per_structure,
                "events": [
                    {
                        "symbol": r.symbol,
                        "event_date": r.event_date.isoformat(),
                        "status": r.status,
                        "entry_date": r.entry_date.isoformat() if r.entry_date else None,
                        "exit_date": r.exit_date.isoformat() if r.exit_date else None,
                        "selector_pick": r.selector_pick,
                        "attempts": {
                            structure: {
                                "status": a.status,
                                "setup_score": a.setup_score,
                                "realized_return_pct": a.realized_return_pct,
                                "realized_expansion_pct": a.realized_expansion_pct,
                                "entry_value": a.entry_value,
                                "exit_value": a.exit_value,
                                "details": a.details,
                            }
                            for structure, a in r.attempts.items()
                        },
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"  Summary JSON: {summary_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
