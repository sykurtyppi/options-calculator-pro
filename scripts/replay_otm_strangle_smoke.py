#!/usr/bin/env python3
"""
replay_otm_strangle_smoke.py
============================
Direct T9-chains → calibration/prior replay for OTM strangle ONLY.

Skips the institutional_ml_db middle layer entirely. For each historical
earnings event in /Volumes/T9/.../earnings_calendar/historical_earnings.sqlite,
this script:

  1. Reads the pre-event and post-event option chain snapshots directly
     from /Volumes/T9/.../options_features_eod/ via DuckDB.
  2. Fetches the symbol's recent OHLC history from yfinance for the
     VolSnapshot's realized-vol calculation.
  3. Builds the canonical VolSnapshot (same code path as live /api/edge/analyze).
  4. Runs build_structure_scorecards + select_best_structure.
  5. If select_best_structure picks 'otm_strangle', identifies the closest-
     to-target-OTM-pct call + put strikes at the near-term expiry, computes
     entry debit + exit credit using the actual chain mids, and derives
     realized_return_pct + realized_expansion_pct.
  6. Writes the result directly to calibration_service + structure_prior_store
     with source_type='replay'.

This is the path the Tier-4 audit ultimately recommended (Option B):
real chain IV in, observations land where the live engine reads from,
no synthetic IV proxy.

Why OTM strangle only
---------------------
The user's live forward-performance shows all 5 paper losses came from
OTM strangle selections. That's where the calibration most needs
historical evidence. Other structures get their own P&L formulas in
follow-up work.

Safety
------
- --target=tmp (default) routes ALL writes to tmp/replay_run_<utc>/
  via env-var overrides on the three store paths. Production stores are
  never touched without --target=production.
- Idempotent: each observation gets observation_id = f"replay_{symbol}_
  {event_date}_otm_strangle". Repeated runs are no-ops.
- --dry-run prints the resolved plan + sampled chain queries but writes
  nothing and never imports the writeable services.

Usage
-----
::

    # 1. Inspect plan:
    python scripts/replay_otm_strangle_smoke.py --dry-run

    # 2. Smoke (default: 4 symbols, target=tmp):
    python scripts/replay_otm_strangle_smoke.py

    # 3. Single-symbol verification:
    python scripts/replay_otm_strangle_smoke.py --symbols AAPL

    # 4. Wider universe (all 10 symbols with earnings data):
    python scripts/replay_otm_strangle_smoke.py --symbols all

    # 5. Promote to production calibration store:
    python scripts/replay_otm_strangle_smoke.py --target production
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

# Suppress duckdb's verbose info logging
logging.basicConfig(level=logging.WARNING)

T9_FEATURES_ROOT = Path(
    "/Volumes/T9/market_data/research/options_features_eod"
)
T9_EARNINGS_DB = Path(
    "/Volumes/T9/market_data/research/options_calculator_pro/earnings_calendar/historical_earnings.sqlite"
)

# Symbols with earnings data + T9 chains. AAPL/AMZN/MSFT/NVDA additionally
# have T9 daily_features (we don't use them; yfinance is the OHLC source).
EARNINGS_SYMBOLS = ["AAPL", "AMD", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "ORCL", "TSLA"]
DEFAULT_SMOKE_SYMBOLS = ["AAPL", "AMZN", "MSFT", "NVDA"]

# Pre/post-event timing — calendar days. Replay finds the nearest available
# chain trade_date within these bounds.
ENTRY_LOOKBACK_DAYS = 2   # T-2 ideal, search back up to 5
ENTRY_MAX_SEARCH = 5
EXIT_FORWARD_DAYS = 1     # T+1 ideal, search forward up to 4
EXIT_MAX_SEARCH = 4

OTM_PCT_DEFAULT = 0.05  # 5% OTM target for both wings


@dataclass
class EventResult:
    symbol: str
    event_date: datetime.date
    release_timing: str
    status: str = ""             # "ok" | "skipped:<reason>" | "selected_other:<structure>"
    selected_structure: Optional[str] = None
    setup_score: Optional[float] = None
    entry_date: Optional[datetime.date] = None
    exit_date: Optional[datetime.date] = None
    realized_return_pct: Optional[float] = None
    realized_expansion_pct: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


def _load_earnings_events(db_path: Path, symbols: List[str]) -> List[Tuple[str, datetime.date, str]]:
    """Pull all (symbol, event_date, release_timing) tuples for our universe."""
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
    """Find the chain trade_date closest to target in the requested direction."""
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
        raise ValueError(f"direction must be before|after, got {direction}")
    row = con.execute(q, params).fetchone()
    if not row:
        return None
    val = row[0]
    return val.date() if hasattr(val, "date") else val


def _load_chain(con, symbol: str, trade_date: datetime.date):
    """Return chain DataFrame at exactly this trade_date."""
    import pandas as pd  # local import keeps top-level fast
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
    # Normalize types: parquet timestamps → date for downstream code
    if not df.empty:
        if hasattr(df["trade_date"].iloc[0], "date"):
            df["trade_date"] = df["trade_date"].dt.date
        if hasattr(df["expiry"].iloc[0], "date"):
            df["expiry"] = df["expiry"].dt.date
    return df


def _fetch_yfinance_ohlc(symbol: str, end_date: datetime.date, days: int = 120):
    """OHLC history ending on end_date, formatted for build_vol_snapshot."""
    import pandas as pd
    import yfinance as yf

    start = end_date - datetime.timedelta(days=days)
    # yfinance treats end as exclusive
    end_exclusive = end_date + datetime.timedelta(days=1)
    hist = yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        auto_adjust=False,
    )
    if hist.empty:
        return None
    hist = hist.reset_index()
    # yfinance returns tz-aware; strip
    if pd.api.types.is_datetime64_any_dtype(hist["Date"]):
        try:
            hist["Date"] = hist["Date"].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
    # Rename to lowercase as build_vol_snapshot expects
    return hist.rename(columns={
        "Date": "trade_date",
        "Open": "Open",   # build_vol_snapshot's _normalize_price_frame accepts Open/High/Low/Close
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    })


def _pick_strangle_wings(
    entry_chain,
    exit_chain,
    near_term_expiry,
    underlying_price: float,
    otm_pct: float,
):
    """Find OTM call + put rows at the same near-term expiry, RESTRICTED to
    strikes that appear in BOTH the entry and exit chain at that expiry.

    Returns (call_row, put_row) or None.

    Restricting to the intersection of strikes guarantees we can compute
    a P&L roundtrip without "exit contracts missing" failures from the
    iVolatility feed's intermittent strike-level coverage gaps.
    """
    target_call = underlying_price * (1 + otm_pct)
    target_put = underlying_price * (1 - otm_pct)

    entry_near = entry_chain[entry_chain["expiry"] == near_term_expiry]
    exit_near = exit_chain[exit_chain["expiry"] == near_term_expiry]

    entry_calls = entry_near[entry_near["call_put"] == "C"]
    entry_puts = entry_near[entry_near["call_put"] == "P"]
    exit_call_strikes = set(exit_near[exit_near["call_put"] == "C"]["strike"])
    exit_put_strikes = set(exit_near[exit_near["call_put"] == "P"]["strike"])

    # Filter entry candidates to strikes present in exit
    callable_calls = entry_calls[entry_calls["strike"].isin(exit_call_strikes)]
    callable_puts = entry_puts[entry_puts["strike"].isin(exit_put_strikes)]
    if callable_calls.empty or callable_puts.empty:
        return None

    call_row = callable_calls.iloc[(callable_calls["strike"] - target_call).abs().argsort().iloc[0]]
    put_row = callable_puts.iloc[(callable_puts["strike"] - target_put).abs().argsort().iloc[0]]
    return call_row, put_row


def _exit_legs_at_same_contracts(exit_chain, expiry, call_strike, put_strike):
    """Look up the SAME contracts in the exit chain by (expiry, call_put, strike)."""
    exit_call = exit_chain[
        (exit_chain["expiry"] == expiry)
        & (exit_chain["call_put"] == "C")
        & (exit_chain["strike"] == call_strike)
    ]
    exit_put = exit_chain[
        (exit_chain["expiry"] == expiry)
        & (exit_chain["call_put"] == "P")
        & (exit_chain["strike"] == put_strike)
    ]
    if exit_call.empty or exit_put.empty:
        return None
    return exit_call.iloc[0], exit_put.iloc[0]


def process_event(
    con,
    symbol: str,
    event_date: datetime.date,
    release_timing: str,
    otm_pct: float,
) -> EventResult:
    """Process one event end-to-end. Returns an EventResult."""
    from services.earnings_vol_snapshot import build_vol_snapshot
    from services.structure_scorecard import build_structure_scorecards
    from services.structure_selector import select_best_structure

    result = EventResult(symbol=symbol, event_date=event_date, release_timing=release_timing)

    target_entry = event_date - datetime.timedelta(days=ENTRY_LOOKBACK_DAYS)
    entry_date = _find_nearest_chain_date(con, symbol, target_entry, "before", ENTRY_MAX_SEARCH)
    if entry_date is None:
        # Look at-or-before the target instead of strictly before
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
        vol_snapshot = build_vol_snapshot(
            symbol,
            entry_date,
            option_chain_data=entry_chain,
            earnings_metadata=earnings_metadata,
            price_data=ohlc,
        )
    except Exception as exc:
        result.status = f"skipped:vol_snapshot_failed:{type(exc).__name__}"
        result.details["error"] = str(exc)
        return result

    try:
        scorecards = build_structure_scorecards(vol_snapshot)
        selected = select_best_structure(vol_snapshot, scorecards)
    except Exception as exc:
        result.status = f"skipped:selector_failed:{type(exc).__name__}"
        result.details["error"] = str(exc)
        return result

    # For CALIBRATION purposes, we record the (otm_strangle.score, realized
    # otm_strangle outcome) pair for every event where otm_strangle is
    # eligible — regardless of whether the live selector would have traded
    # it. Rationale:
    #   - calibration_service maps a structure's score → realized expansion.
    #     The mapping has to be learned independently of the trade-or-not
    #     gate, otherwise it can never improve (selector rejects → no
    #     observations → calibration stays prior → selector rejects).
    #   - structure_prior_store learns per-structure win rate / avg return
    #     for use when that structure IS picked. Same chicken-and-egg.
    #
    # We still capture the selector's actual decision in the result for the
    # JSON summary, so we can report e.g. "selector would have picked X in
    # only Y of Z events" alongside the calibration observations.
    selector_decision = selected.recommendation
    selector_pick = selected.best_structure
    result.details["selector_recommendation"] = selector_decision
    result.details["selector_pick"] = selector_pick
    result.details["structure_scores"] = {
        c.structure: {
            "eligible": bool(c.eligible),
            "composite": float(c.composite_structure_score),
            "flags": list(c.eligibility_flags)[:5],
        }
        for c in scorecards
    }

    strangle_card = next((c for c in scorecards if c.structure == "otm_strangle"), None)
    if strangle_card is None:
        result.status = "skipped:no_otm_strangle_scorecard"
        return result
    if not strangle_card.eligible:
        result.status = f"skipped:strangle_ineligible:{','.join(strangle_card.eligibility_flags[:3])}"
        return result

    result.selected_structure = "otm_strangle"
    result.setup_score = float(strangle_card.composite_structure_score)

    # OTM strangle selected. Find wings + compute P&L.
    # IMPORTANT: pick an expiry that (a) is after earnings (b) survives
    # until exit_date AND (c) EXISTS in both the entry and exit chains.
    # The iVolatility parquet feed has intermittent gaps in weekly-expiry
    # coverage day-to-day: an expiry can be present on the entry trade_date
    # but missing from the exit trade_date (or vice versa). Picking the
    # smallest entry-side expiry without checking the exit side leaves us
    # with contracts that vanish on the exit lookup.
    entry_expiries = set(e for e in entry_chain["expiry"] if e > event_date and e >= exit_date)
    exit_expiries = set(exit_chain["expiry"])
    common_expiries = sorted(entry_expiries & exit_expiries)
    if not common_expiries:
        result.status = "skipped:no_common_expiry"
        result.details["entry_expiries"] = sorted(str(e) for e in entry_expiries)[:5]
        result.details["exit_expiries"] = sorted(str(e) for e in exit_expiries)[:5]
        return result
    near_term_expiry = common_expiries[0]

    # Normalize near_term_expiry to a date object
    if hasattr(near_term_expiry, "date"):
        near_term_expiry = near_term_expiry.date()

    underlying = float(entry_chain["underlying_price"].iloc[0])
    wings = _pick_strangle_wings(entry_chain, exit_chain, near_term_expiry, underlying, otm_pct)
    if wings is None:
        result.status = "skipped:no_strangle_wings"
        return result
    call_row, put_row = wings

    exit_legs = _exit_legs_at_same_contracts(
        exit_chain,
        near_term_expiry,
        float(call_row["strike"]),
        float(put_row["strike"]),
    )
    if exit_legs is None:
        # Should be unreachable now since wing picker filters to strikes
        # present in exit_chain, but defensive in case of float-precision.
        result.status = "skipped:exit_contracts_missing"
        return result
    exit_call, exit_put = exit_legs

    entry_debit = float(call_row["mid"]) + float(put_row["mid"])
    exit_credit = float(exit_call["mid"]) + float(exit_put["mid"])
    if entry_debit <= 0:
        result.status = "skipped:zero_debit"
        return result

    realized_return_pct = (exit_credit - entry_debit) / entry_debit * 100.0

    entry_iv = (float(call_row["iv"]) + float(put_row["iv"])) / 2.0
    exit_iv = (float(exit_call["iv"]) + float(exit_put["iv"])) / 2.0
    if entry_iv <= 0:
        result.status = "skipped:zero_entry_iv"
        return result
    realized_expansion_pct = (exit_iv - entry_iv) / entry_iv * 100.0

    result.realized_return_pct = realized_return_pct
    result.realized_expansion_pct = realized_expansion_pct
    result.details = {
        "call_strike": float(call_row["strike"]),
        "put_strike": float(put_row["strike"]),
        "near_term_expiry": str(near_term_expiry),
        "underlying": underlying,
        "entry_debit": entry_debit,
        "exit_credit": exit_credit,
        "entry_iv": entry_iv,
        "exit_iv": exit_iv,
    }
    result.status = "ok"
    return result


def _apply_target_env(target: str, run_timestamp: str) -> Path:
    """Point the store singletons at tmp or production via env vars.

    Returns the directory where stores will live (so the caller can show it).
    The store modules read these env vars at import time (PR-K design).
    """
    if target == "production":
        # Don't override anything — default _DEFAULT_STORE paths win.
        return Path.home() / ".options_calculator_pro"

    tmp_root = REPO_ROOT / "tmp" / f"replay_run_{run_timestamp}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / "outcomes").mkdir(exist_ok=True)
    (tmp_root / "calibration").mkdir(exist_ok=True)
    (tmp_root / "priors").mkdir(exist_ok=True)

    os.environ["OPTIONS_CALCULATOR_OUTCOMES_PATH"] = str(tmp_root / "outcomes" / "outcome_store.sqlite")
    os.environ["OPTIONS_CALCULATOR_CALIBRATION_PATH"] = str(tmp_root / "calibration" / "iv_expansion.json")
    os.environ["OPTIONS_CALCULATOR_PRIORS_PATH"] = str(tmp_root / "priors" / "structure_priors.json")
    return tmp_root


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("Why OTM strangle only")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SMOKE_SYMBOLS),
        help=f"Comma-separated symbols, or 'all' for all 10 earnings symbols. Default: {','.join(DEFAULT_SMOKE_SYMBOLS)}",
    )
    parser.add_argument(
        "--target",
        choices=["tmp", "production"],
        default="tmp",
        help="Where to write calibration + prior updates (default: tmp)",
    )
    parser.add_argument(
        "--otm-pct",
        type=float,
        default=OTM_PCT_DEFAULT,
        help=f"Target OTM percentage for the strangle wings (default: {OTM_PCT_DEFAULT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan + sample one event per symbol; no writes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N events (for quick smoke); default unlimited",
    )
    args = parser.parse_args(argv)

    # Resolve symbol list
    if args.symbols == "all":
        symbols = list(EARNINGS_SYMBOLS)
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

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
    print(f"  OTM STRANGLE REPLAY — direct T9 chains → calibration/prior")
    print("=" * 72)
    print(f"  Symbols:        {len(symbols)} — {','.join(symbols)}")
    print(f"  Events:         {len(events)}")
    print(f"  OTM target:     ±{args.otm_pct*100:.1f}%")
    print(f"  Target:         {args.target}")
    print(f"  Target dir:     {target_dir}")
    print(f"  Dry run:        {args.dry_run}")
    print()

    if args.dry_run:
        print("Sample events (one per symbol):")
        seen = set()
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

    # Lazy import the writeable services AFTER env vars are set, so the
    # singletons pick up tmp paths (or production defaults if --target=production).
    from services.calibration_service import get_calibration
    from services.structure_prior_store import get_structure_prior_store

    calibration = get_calibration()
    prior_store = get_structure_prior_store()

    print(f"  Calibration store path: {calibration._path}")
    print(f"  Prior store path:       {prior_store._path}")
    print()

    results: List[EventResult] = []
    status_counts: Dict[str, int] = defaultdict(int)
    structure_counts: Dict[str, int] = defaultdict(int)

    for idx, (symbol, event_date, release_timing) in enumerate(events, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(events):
            print(f"  [{idx}/{len(events)}] processing {symbol} {event_date}…")
        result = process_event(con, symbol, event_date, release_timing, args.otm_pct)
        results.append(result)
        status_counts[result.status] += 1
        if result.selected_structure:
            structure_counts[result.selected_structure] += 1

        if result.status == "ok":
            observation_id = f"replay_{symbol}_{event_date.isoformat()}_otm_strangle"
            try:
                calibration.update(
                    result.setup_score or 0.0,
                    result.realized_expansion_pct,
                    observation_id=observation_id,
                    source_type="replay",
                    observation_date=result.exit_date,
                )
                prior_store.update(
                    structure="otm_strangle",
                    realized_return_pct=result.realized_return_pct,
                    realized_expansion_pct=result.realized_expansion_pct,
                    source_type="replay",
                    observation_date=result.exit_date,
                    observation_id=observation_id,
                )
            except Exception as exc:
                result.status = f"write_failed:{type(exc).__name__}"
                result.details["write_error"] = str(exc)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  REPLAY SUMMARY")
    print("=" * 72)
    print(f"  Total events processed:    {len(results)}")
    print()
    print("  By selector decision:")
    for struct, n in sorted(structure_counts.items(), key=lambda x: -x[1]):
        print(f"    {struct:20s} {n}")
    if not structure_counts:
        print("    (none — likely all events skipped pre-selector)")
    print()
    print("  By outcome:")
    for status, n in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:35s} {n}")
    print()

    ok_results = [r for r in results if r.status == "ok"]
    if ok_results:
        avg_return = sum(r.realized_return_pct for r in ok_results) / len(ok_results)
        avg_expansion = sum(r.realized_expansion_pct for r in ok_results) / len(ok_results)
        wins = sum(1 for r in ok_results if r.realized_return_pct > 0)
        print(f"  OTM strangle observations written: {len(ok_results)}")
        print(f"    Mean realized return:   {avg_return:+.2f}%")
        print(f"    Mean IV expansion:      {avg_expansion:+.2f}%")
        print(f"    Win rate:               {wins}/{len(ok_results)} = {100*wins/len(ok_results):.0f}%")
    print()
    print(f"  Calibration store n after: {calibration._n()}")
    print(f"  Calibration store path:    {calibration._path}")
    print()

    # Persist a JSON summary alongside the stores
    summary_dir = target_dir if args.target == "tmp" else REPO_ROOT / "exports" / "reports" / "replay_otm_strangle"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"replay_otm_strangle_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "symbols": symbols,
                "events_total": len(results),
                "structure_counts": dict(structure_counts),
                "status_counts": dict(status_counts),
                "ok_count": len(ok_results),
                "mean_return_pct": (sum(r.realized_return_pct for r in ok_results) / len(ok_results)) if ok_results else None,
                "mean_expansion_pct": (sum(r.realized_expansion_pct for r in ok_results) / len(ok_results)) if ok_results else None,
                "events": [
                    {
                        "symbol": r.symbol,
                        "event_date": r.event_date.isoformat(),
                        "status": r.status,
                        "selected_structure": r.selected_structure,
                        "setup_score": r.setup_score,
                        "entry_date": r.entry_date.isoformat() if r.entry_date else None,
                        "exit_date": r.exit_date.isoformat() if r.exit_date else None,
                        "realized_return_pct": r.realized_return_pct,
                        "realized_expansion_pct": r.realized_expansion_pct,
                        "details": r.details,
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
