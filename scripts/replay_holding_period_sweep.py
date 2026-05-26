#!/usr/bin/env python3
"""
replay_holding_period_sweep.py
==============================
Holding-period sensitivity for the multi-structure replay.

Same picker + pricing logic as replay_multi_structure_smoke.py (PR-AA),
but for each event computes realized P&L at MULTIPLE exit horizons:
T+1 (the previous default), T+3, T+7.

Hypothesis being tested
-----------------------
The previous T+1-only result captured the IV crush but not the theta decay
on the short front leg of calendars. Real calendar traders hold past T+1
specifically to harvest theta on the short. If T+7 returns are materially
better than T+1, the system should recommend longer holds.

Risks of holding longer
-----------------------
- Front-month expiry: if the short front leg expires ITM, assignment
  risk materializes. We skip events where holding period would extend
  past the picked front expiry minus 2 trading days.
- IV reversal: post-crush IV often drifts back up; the long-vol structures
  could partially recover their value.

Output
------
One JSON per run with the schema:
    events[i].attempts[structure].horizons[days_held] = {
        exit_date, entry_value, exit_value,
        realized_return_pct, realized_expansion_pct, ...
    }

Idempotency
-----------
observation_id = f"replay_{symbol}_{event}_{structure}_t{days_held}"
so the same horizon can be re-run and overwrites are no-ops, but DIFFERENT
horizons coexist as separate observations.

Usage
-----
::

    # Default: 4 symbols × all 4 structures × [1, 3, 7] horizons:
    python scripts/replay_holding_period_sweep.py

    # All 10 symbols:
    python scripts/replay_holding_period_sweep.py --symbols all

    # Custom horizons:
    python scripts/replay_holding_period_sweep.py --horizons 1,2,5,10
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
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING)

T9_FEATURES_ROOT = Path("/Volumes/T9/market_data/research/options_features_eod")
T9_EARNINGS_DB = Path(
    "/Volumes/T9/market_data/research/options_calculator_pro/earnings_calendar/historical_earnings.sqlite"
)

EARNINGS_SYMBOLS = ["AAPL", "AMD", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "ORCL", "TSLA"]
DEFAULT_SMOKE_SYMBOLS = ["AAPL", "AMZN", "MSFT", "NVDA"]
SUPPORTED_STRUCTURES = ("atm_straddle", "otm_strangle", "call_calendar", "put_calendar")
LONG_VOL_STRUCTURES = {"atm_straddle", "otm_strangle"}

ENTRY_LOOKBACK_DAYS = 2
ENTRY_MAX_SEARCH = 5
EXIT_MAX_SEARCH_FLOOR = 5  # always at least 5 days past target

# How close to front expiry we allow holding before bailing on assignment risk.
# For example, if front expiry is 2024-02-09 and we hold past 2024-02-07, skip.
FRONT_EXPIRY_BUFFER_DAYS = 2

OTM_PCT_DEFAULT = 0.05
MIN_ENTRY_DEBIT = 0.10
DEFAULT_HORIZONS = (1, 3, 7)


@dataclass
class Leg:
    side: str
    expiry: datetime.date
    call_put: str
    strike: float


@dataclass
class HorizonResult:
    days_held: int
    status: str = ""
    exit_date: Optional[datetime.date] = None
    entry_value: Optional[float] = None
    exit_value: Optional[float] = None
    realized_return_pct: Optional[float] = None
    entry_iv: Optional[float] = None
    exit_iv: Optional[float] = None
    realized_expansion_pct: Optional[float] = None


@dataclass
class StructureAttempt:
    structure: str
    status: str = ""
    setup_score: Optional[float] = None
    legs: Optional[List[Leg]] = None
    leg_details: Dict[str, Any] = field(default_factory=dict)
    horizons: Dict[int, HorizonResult] = field(default_factory=dict)


@dataclass
class EventResult:
    symbol: str
    event_date: datetime.date
    release_timing: str
    status: str = ""
    entry_date: Optional[datetime.date] = None
    selector_pick: Optional[str] = None
    attempts: Dict[str, StructureAttempt] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chain helpers (same shape as PR-AA replay)
# ---------------------------------------------------------------------------

def _load_earnings_events(db_path: Path, symbols: List[str]):
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    placeholders = ",".join("?" * len(symbols))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""SELECT symbol, event_date, release_timing FROM historical_earnings_events
                WHERE symbol IN ({placeholders}) ORDER BY symbol, event_date""",
            symbols,
        ).fetchall()
    return [(s, datetime.date.fromisoformat(d), t) for s, d, t in rows]


def _chain_glob(symbol):
    return f"{T9_FEATURES_ROOT}/underlying_symbol={symbol}/**/*.parquet"


def _find_nearest_chain_date(con, symbol, target, direction, max_days):
    pattern = _chain_glob(symbol)
    if direction == "before":
        q = f"""SELECT DISTINCT trade_date FROM read_parquet('{pattern}', hive_partitioning=true)
                WHERE trade_date < ? AND trade_date >= ?
                ORDER BY trade_date DESC LIMIT 1"""
        params = [target, target - datetime.timedelta(days=max_days)]
    else:
        q = f"""SELECT DISTINCT trade_date FROM read_parquet('{pattern}', hive_partitioning=true)
                WHERE trade_date > ? AND trade_date <= ?
                ORDER BY trade_date ASC LIMIT 1"""
        params = [target, target + datetime.timedelta(days=max_days)]
    row = con.execute(q, params).fetchone()
    if not row:
        return None
    v = row[0]
    return v.date() if hasattr(v, "date") else v


def _trading_days_after(con, symbol, event_date, max_count=15) -> List[datetime.date]:
    """Return up to *max_count* distinct trade_dates strictly after event_date,
    ordered ascending. This lets callers pick the N-th trading day (T+N)."""
    pattern = _chain_glob(symbol)
    rows = con.execute(
        f"""SELECT DISTINCT trade_date FROM read_parquet('{pattern}', hive_partitioning=true)
            WHERE trade_date > ? ORDER BY trade_date ASC LIMIT {max_count}""",
        [event_date],
    ).fetchall()
    out = []
    for (v,) in rows:
        out.append(v.date() if hasattr(v, "date") else v)
    return out


def _load_chain(con, symbol, trade_date):
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


def _fetch_yfinance_ohlc(symbol, end_date, days=120):
    import pandas as pd
    import yfinance as yf
    start = end_date - datetime.timedelta(days=days)
    end_exclusive = end_date + datetime.timedelta(days=1)
    hist = yf.Ticker(symbol).history(start=start.isoformat(), end=end_exclusive.isoformat(), auto_adjust=False)
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
# Leg construction — picks expiries that survive the LONGEST horizon we'll
# need, so the same legs price at all horizons.
# ---------------------------------------------------------------------------

def _strikes_in_chain(chain, expiry, call_put):
    return set(chain[(chain["expiry"] == expiry) & (chain["call_put"] == call_put)]["strike"])


def _pick_strike_closest_to(target, candidates):
    return min(candidates, key=lambda s: abs(s - target)) if candidates else None


def _common_expiries_after_in_all_chains(
    entry_chain, exit_chains: Dict[int, Any], event_date: datetime.date, max_exit_date: datetime.date
) -> List[datetime.date]:
    """Expiries > event_date and >= max_exit_date that appear in entry AND every exit chain."""
    entry_e = {e for e in entry_chain["expiry"] if e > event_date and e >= max_exit_date}
    common = entry_e
    for exit_chain in exit_chains.values():
        common = common & set(exit_chain["expiry"])
    return sorted(common)


def _strikes_in_all_chains(entry_chain, exit_chains: Dict[int, Any], expiry, call_put):
    s = _strikes_in_chain(entry_chain, expiry, call_put)
    for ec in exit_chains.values():
        s = s & _strikes_in_chain(ec, expiry, call_put)
    return s


def build_atm_straddle_legs(entry_chain, exit_chains, event_date, max_exit_date, underlying):
    expiries = _common_expiries_after_in_all_chains(entry_chain, exit_chains, event_date, max_exit_date)
    for expiry in expiries:
        call_strikes = _strikes_in_all_chains(entry_chain, exit_chains, expiry, "C")
        put_strikes = _strikes_in_all_chains(entry_chain, exit_chains, expiry, "P")
        shared = call_strikes & put_strikes
        atm = _pick_strike_closest_to(underlying, shared)
        if atm is None:
            continue
        return [
            Leg("long", expiry, "C", atm),
            Leg("long", expiry, "P", atm),
        ], {"near_term_expiry": str(expiry), "atm_strike": atm}
    return None, {"reason": "no_atm_expiry_with_shared_strike"}


def build_otm_strangle_legs(entry_chain, exit_chains, event_date, max_exit_date, underlying, otm_pct):
    target_call = underlying * (1 + otm_pct)
    target_put = underlying * (1 - otm_pct)
    expiries = _common_expiries_after_in_all_chains(entry_chain, exit_chains, event_date, max_exit_date)
    for expiry in expiries:
        call_strikes = _strikes_in_all_chains(entry_chain, exit_chains, expiry, "C")
        put_strikes = _strikes_in_all_chains(entry_chain, exit_chains, expiry, "P")
        call_k = _pick_strike_closest_to(target_call, call_strikes)
        put_k = _pick_strike_closest_to(target_put, put_strikes)
        if call_k is None or put_k is None:
            continue
        return [
            Leg("long", expiry, "C", call_k),
            Leg("long", expiry, "P", put_k),
        ], {"near_term_expiry": str(expiry), "call_strike": call_k, "put_strike": put_k}
    return None, {"reason": "no_strangle_expiry_with_shared_strikes"}


def _build_calendar_legs(entry_chain, exit_chains, event_date, max_exit_date, underlying, call_put):
    expiries = _common_expiries_after_in_all_chains(entry_chain, exit_chains, event_date, max_exit_date)
    if len(expiries) < 2:
        return None, {"reason": "fewer_than_two_common_expiries"}
    for i, front in enumerate(expiries):
        for back in expiries[i + 1:]:
            if (back - front).days < 14:
                continue
            front_strikes = _strikes_in_all_chains(entry_chain, exit_chains, front, call_put)
            back_strikes = _strikes_in_all_chains(entry_chain, exit_chains, back, call_put)
            shared = front_strikes & back_strikes
            atm = _pick_strike_closest_to(underlying, shared)
            if atm is None:
                continue
            return [
                Leg("short", front, call_put, atm),
                Leg("long", back, call_put, atm),
            ], {
                "front_expiry": str(front),
                "back_expiry": str(back),
                "atm_strike": atm,
                "front_to_back_days": (back - front).days,
            }
    return None, {"reason": "no_valid_front_back_pair_with_shared_strike"}


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def _lookup_leg(chain, leg: Leg):
    row = chain[
        (chain["expiry"] == leg.expiry) & (chain["call_put"] == leg.call_put) & (chain["strike"] == leg.strike)
    ]
    return None if row.empty else row.iloc[0]


def price_legs(legs: List[Leg], chain):
    long_ivs, short_ivs = [], []
    net = 0.0
    for leg in legs:
        row = _lookup_leg(chain, leg)
        if row is None:
            return None
        mid, iv = float(row["mid"]), float(row["iv"])
        if leg.side == "long":
            net += mid
            long_ivs.append(iv)
        else:
            net -= mid
            short_ivs.append(iv)
    return {
        "net_value": net,
        "long_iv_avg": sum(long_ivs) / len(long_ivs) if long_ivs else 0.0,
        "short_iv_avg": sum(short_ivs) / len(short_ivs) if short_ivs else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-event
# ---------------------------------------------------------------------------

def process_event(
    con,
    symbol: str,
    event_date: datetime.date,
    release_timing: str,
    otm_pct: float,
    structures: List[str],
    horizons: List[int],
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
    result.entry_date = entry_date

    entry_chain = _load_chain(con, symbol, entry_date)
    if entry_chain.empty:
        result.status = "skipped:empty_entry_chain"
        return result

    # Resolve exit dates for every horizon using *trading-day* offsets.
    # T+1 = first trading day after event, T+3 = third, T+7 = seventh.
    # This is semantically what options traders mean by "hold for N days"
    # (a Friday→Monday transition is one trading day, not three).
    max_h = max(horizons)
    available_tds = _trading_days_after(con, symbol, event_date, max_count=max(15, max_h + 5))
    exit_chains: Dict[int, Any] = {}
    exit_dates: Dict[int, datetime.date] = {}
    for h in horizons:
        if h - 1 >= len(available_tds):
            continue
        d = available_tds[h - 1]
        ch = _load_chain(con, symbol, d)
        if ch.empty:
            continue
        exit_dates[h] = d
        exit_chains[h] = ch

    if not exit_chains:
        result.status = "skipped:no_exit_chains"
        return result

    max_exit_date = max(exit_dates.values())

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
        scorecards = build_structure_scorecards(snapshot)
        selected = select_best_structure(snapshot, scorecards)
        result.selector_pick = selected.best_structure
    except Exception as exc:
        result.status = f"skipped:snapshot_or_score_failed:{type(exc).__name__}"
        return result

    scorecards_by_structure = {c.structure: c for c in scorecards}
    underlying = float(entry_chain["underlying_price"].iloc[0])

    for structure in structures:
        card = scorecards_by_structure.get(structure)
        attempt = StructureAttempt(structure=structure)
        if card is None:
            attempt.status = "skipped:no_scorecard"
            result.attempts[structure] = attempt
            continue
        if not card.eligible:
            flags = ",".join(list(card.eligibility_flags)[:3])
            attempt.status = f"skipped:ineligible:{flags}"
            result.attempts[structure] = attempt
            continue
        attempt.setup_score = float(card.composite_structure_score)

        # Build legs ONCE using strikes/expiries available in ALL exit chains.
        if structure == "atm_straddle":
            legs, details = build_atm_straddle_legs(entry_chain, exit_chains, event_date, max_exit_date, underlying)
        elif structure == "otm_strangle":
            legs, details = build_otm_strangle_legs(entry_chain, exit_chains, event_date, max_exit_date, underlying, otm_pct)
        elif structure == "call_calendar":
            legs, details = _build_calendar_legs(entry_chain, exit_chains, event_date, max_exit_date, underlying, "C")
        elif structure == "put_calendar":
            legs, details = _build_calendar_legs(entry_chain, exit_chains, event_date, max_exit_date, underlying, "P")
        else:
            attempt.status = f"skipped:unknown_structure"
            result.attempts[structure] = attempt
            continue

        if legs is None:
            attempt.status = f"skipped:no_legs:{details.get('reason')}"
            attempt.leg_details = details
            result.attempts[structure] = attempt
            continue
        attempt.legs = legs
        attempt.leg_details = dict(details)

        # Price entry once
        entry_priced = price_legs(legs, entry_chain)
        if entry_priced is None:
            attempt.status = "skipped:entry_pricing_failed"
            result.attempts[structure] = attempt
            continue
        entry_value = entry_priced["net_value"]
        if entry_value < MIN_ENTRY_DEBIT:
            attempt.status = f"skipped:entry_below_min_debit:{entry_value:+.3f}"
            result.attempts[structure] = attempt
            continue
        if structure in LONG_VOL_STRUCTURES:
            entry_iv_baseline = entry_priced["long_iv_avg"]
        else:
            entry_iv_baseline = entry_priced["short_iv_avg"]

        # Determine front expiry for calendar assignment-risk gate
        front_expiry = None
        if structure in {"call_calendar", "put_calendar"}:
            front_expiry = legs[0].expiry  # short front leg

        # Price each horizon
        for h in horizons:
            hr = HorizonResult(days_held=h)
            if h not in exit_chains:
                hr.status = "skipped:no_exit_chain_for_horizon"
                attempt.horizons[h] = hr
                continue
            ex_date = exit_dates[h]
            # Calendar assignment-risk: don't hold past (front_expiry - buffer)
            if front_expiry is not None:
                cutoff = front_expiry - datetime.timedelta(days=FRONT_EXPIRY_BUFFER_DAYS)
                if ex_date > cutoff:
                    hr.status = f"skipped:past_front_expiry_buffer:{front_expiry}"
                    attempt.horizons[h] = hr
                    continue
            ex_priced = price_legs(legs, exit_chains[h])
            if ex_priced is None:
                hr.status = "skipped:exit_pricing_failed"
                attempt.horizons[h] = hr
                continue
            hr.exit_date = ex_date
            hr.entry_value = entry_value
            hr.exit_value = ex_priced["net_value"]
            hr.realized_return_pct = (hr.exit_value - entry_value) / entry_value * 100.0
            hr.entry_iv = entry_iv_baseline
            if structure in LONG_VOL_STRUCTURES:
                hr.exit_iv = ex_priced["long_iv_avg"]
            else:
                hr.exit_iv = ex_priced["short_iv_avg"]
            if entry_iv_baseline > 0:
                hr.realized_expansion_pct = (hr.exit_iv - entry_iv_baseline) / entry_iv_baseline * 100.0
            hr.status = "ok"
            attempt.horizons[h] = hr

        # Attempt status = "ok" if at least one horizon priced
        if any(hr.status == "ok" for hr in attempt.horizons.values()):
            attempt.status = "ok"
        else:
            attempt.status = "skipped:no_horizons_priced"
        result.attempts[structure] = attempt

    result.status = "ok"
    return result


# ---------------------------------------------------------------------------
# Target/env + main
# ---------------------------------------------------------------------------

def _apply_target_env(target, ts):
    if target == "production":
        return Path.home() / ".options_calculator_pro"
    tmp_root = REPO_ROOT / "tmp" / f"replay_horizons_{ts}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / "outcomes").mkdir(exist_ok=True)
    (tmp_root / "calibration").mkdir(exist_ok=True)
    (tmp_root / "priors").mkdir(exist_ok=True)
    os.environ["OPTIONS_CALCULATOR_OUTCOMES_PATH"] = str(tmp_root / "outcomes" / "outcome_store.sqlite")
    os.environ["OPTIONS_CALCULATOR_CALIBRATION_PATH"] = str(tmp_root / "calibration" / "iv_expansion.json")
    os.environ["OPTIONS_CALCULATOR_PRIORS_PATH"] = str(tmp_root / "priors" / "structure_priors.json")
    return tmp_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SMOKE_SYMBOLS))
    parser.add_argument("--structures", default=",".join(SUPPORTED_STRUCTURES))
    parser.add_argument("--horizons", default=",".join(str(d) for d in DEFAULT_HORIZONS),
                        help=f"Comma-separated exit horizons in days from event. Default {','.join(str(d) for d in DEFAULT_HORIZONS)}")
    parser.add_argument("--target", choices=["tmp", "production"], default="tmp")
    parser.add_argument("--otm-pct", type=float, default=OTM_PCT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-store-writes", action="store_true",
                        help="Skip calibration/prior store writes — analysis-only run.")
    args = parser.parse_args(argv)

    symbols = list(EARNINGS_SYMBOLS) if args.symbols == "all" else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    structures = [s.strip() for s in args.structures.split(",") if s.strip()]
    bad = [s for s in structures if s not in SUPPORTED_STRUCTURES]
    if bad:
        print(f"ERROR: unknown structures {bad}. Valid: {SUPPORTED_STRUCTURES}", file=sys.stderr)
        return 2
    horizons = sorted({int(h) for h in args.horizons.split(",") if h.strip()})

    if not T9_FEATURES_ROOT.exists() or not T9_EARNINGS_DB.exists():
        print("ERROR: T9 root or earnings DB not found.", file=sys.stderr)
        return 2

    events = _load_earnings_events(T9_EARNINGS_DB, symbols)
    if args.limit is not None:
        events = events[: args.limit]

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = _apply_target_env(args.target, ts)

    print()
    print("=" * 78)
    print("  HOLDING-PERIOD SWEEP — multi-structure replay across multiple exit horizons")
    print("=" * 78)
    print(f"  Symbols:        {len(symbols)} — {','.join(symbols)}")
    print(f"  Structures:     {','.join(structures)}")
    print(f"  Horizons:       {horizons} days")
    print(f"  Events:         {len(events)}  (× {len(structures)} struct × {len(horizons)} horizons = up to {len(events)*len(structures)*len(horizons)} obs)")
    print(f"  Target:         {args.target}    dir: {target_dir}")
    print(f"  Store writes:   {'NO (--no-store-writes)' if args.no_store_writes else 'YES'}")
    print(f"  Dry run:        {args.dry_run}")
    print()

    if args.dry_run:
        seen = set()
        for sym, ed, tm in events:
            if sym in seen:
                continue
            seen.add(sym)
            print(f"  {sym}  {ed.isoformat()}  {tm}")
        print()
        print("DRY RUN — exiting before chain queries.")
        return 0

    import duckdb
    con = duckdb.connect(":memory:")

    if args.no_store_writes:
        calibration = None
        prior_store = None
    else:
        from services.calibration_service import get_calibration
        from services.structure_prior_store import get_structure_prior_store
        calibration = get_calibration()
        prior_store = get_structure_prior_store()
        print(f"  Calibration store path: {calibration._path}")
        print(f"  Prior store path:       {prior_store._path}")
        print()

    results: List[EventResult] = []
    event_status_counts = defaultdict(int)
    # Per-(structure, horizon) status counts
    horizon_status: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for idx, (symbol, event_date, release_timing) in enumerate(events, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(events):
            print(f"  [{idx}/{len(events)}] processing {symbol} {event_date}…")
        r = process_event(con, symbol, event_date, release_timing, args.otm_pct, structures, horizons)
        results.append(r)
        event_status_counts[r.status] += 1

        if r.status != "ok":
            continue

        for structure, attempt in r.attempts.items():
            for h, hr in attempt.horizons.items():
                horizon_status[(structure, h)][hr.status] += 1
                if hr.status == "ok" and not args.no_store_writes:
                    obs_id = f"replay_{symbol}_{event_date.isoformat()}_{structure}_t{h}"
                    try:
                        if structure in LONG_VOL_STRUCTURES and calibration is not None:
                            calibration.update(
                                attempt.setup_score or 0.0,
                                hr.realized_expansion_pct,
                                observation_id=obs_id,
                                source_type="replay",
                                observation_date=hr.exit_date,
                            )
                        if prior_store is not None:
                            prior_store.update(
                                structure=structure,
                                realized_return_pct=hr.realized_return_pct,
                                realized_expansion_pct=hr.realized_expansion_pct,
                                source_type="replay",
                                observation_date=hr.exit_date,
                                observation_id=obs_id,
                            )
                    except Exception as exc:
                        hr.status = f"write_failed:{type(exc).__name__}"

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  HOLDING-PERIOD SWEEP SUMMARY")
    print("=" * 78)
    print("  Event-level outcomes:")
    for k, v in sorted(event_status_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:35s} {v}")
    print()

    summary_per_structure_horizon: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for structure in structures:
        print(f"  ── {structure} ──")
        summary_per_structure_horizon[structure] = {}
        for h in horizons:
            oks = [
                r.attempts[structure].horizons[h]
                for r in results
                if structure in r.attempts
                and h in r.attempts[structure].horizons
                and r.attempts[structure].horizons[h].status == "ok"
            ]
            sc = horizon_status[(structure, h)]
            if oks:
                m_ret = sum(hr.realized_return_pct for hr in oks) / len(oks)
                m_exp = sum(hr.realized_expansion_pct for hr in oks) / len(oks)
                wins = sum(1 for hr in oks if hr.realized_return_pct > 0)
                summary_per_structure_horizon[structure][h] = {
                    "n": len(oks),
                    "mean_return_pct": m_ret,
                    "mean_expansion_pct": m_exp,
                    "win_rate": wins / len(oks),
                }
                print(f"    T+{h}: n={len(oks):3d}  mean ret {m_ret:+7.2f}%  mean exp {m_exp:+7.2f}%  "
                      f"win {wins}/{len(oks)} = {100*wins/len(oks):.0f}%")
            else:
                summary_per_structure_horizon[structure][h] = {"n": 0}
                print(f"    T+{h}: n=0  (no observations)")
            # Skip reasons that are not 'ok'
            skips = [(k, v) for k, v in sc.items() if k != "ok"]
            if skips:
                top = sorted(skips, key=lambda x: -x[1])[:2]
                print(f"          skips: {', '.join(f'{k}={v}' for k, v in top)}")
        print()

    if calibration is not None:
        print(f"  Calibration store n after: {calibration._n()}")
        print(f"  Calibration store path:    {calibration._path}")
        print()

    summary_dir = target_dir if args.target == "tmp" else REPO_ROOT / "exports" / "reports" / "replay_horizons"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"replay_holding_period_sweep_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": ts,
                "symbols": symbols,
                "structures": structures,
                "horizons": horizons,
                "events_total": len(results),
                "event_status_counts": dict(event_status_counts),
                "horizon_status": {f"{s}|{h}": dict(v) for (s, h), v in horizon_status.items()},
                "summary_per_structure_horizon": summary_per_structure_horizon,
                "events": [
                    {
                        "symbol": r.symbol,
                        "event_date": r.event_date.isoformat(),
                        "status": r.status,
                        "entry_date": r.entry_date.isoformat() if r.entry_date else None,
                        "selector_pick": r.selector_pick,
                        "attempts": {
                            structure: {
                                "status": a.status,
                                "setup_score": a.setup_score,
                                "leg_details": a.leg_details,
                                "horizons": {
                                    str(h): {
                                        "status": hr.status,
                                        "exit_date": hr.exit_date.isoformat() if hr.exit_date else None,
                                        "entry_value": hr.entry_value,
                                        "exit_value": hr.exit_value,
                                        "realized_return_pct": hr.realized_return_pct,
                                        "realized_expansion_pct": hr.realized_expansion_pct,
                                    } for h, hr in a.horizons.items()
                                },
                            } for structure, a in r.attempts.items()
                        },
                    } for r in results
                ],
            }, f, indent=2, default=str,
        )
    print(f"  Summary JSON: {summary_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
