#!/usr/bin/env python3
"""
seed_replay_snapshots.py
========================
Seed the earnings_option_snapshots table with real historical IV data from
MarketData.app so the replay-based backtest path has actual option economics
to work with instead of synthetic return proxies.

This script does ONE thing clearly:
  1. Fetches ATM front/back IV snapshots at T-5 (pre-earnings) and T+1 (post-earnings)
     for each historical earnings event in the lookback window.
  2. Stores them in earnings_option_snapshots via capture_historical_iv_snapshots_mda().
  3. Prints a pairing-progress report so you can see how much of the walk-forward
     backtest is now replay-ready vs. still synthetic.

Usage
-----
  # Dry run — show what would be captured, estimate credits, touch nothing:
  python scripts/seed_replay_snapshots.py --dry-run

  # Seed 10 liquid names, 2-year lookback (good first run, ~15 credits):
  python scripts/seed_replay_snapshots.py --tier top10 --years 2

  # Seed 25 names:
  python scripts/seed_replay_snapshots.py --tier top25 --years 2

  # Custom symbol list:
  python scripts/seed_replay_snapshots.py --symbols AAPL,MSFT,NVDA --years 3

  # Full institutional universe (run overnight, ~250 credits):
  python scripts/seed_replay_snapshots.py --tier full --years 2

Prerequisites
-------------
  export MARKETDATA_TOKEN=your_token_here
  # or add MARKETDATA_TOKEN=... to your .env file

Credit budget note
------------------
MarketData.app historical chains cost ~1 credit per 1,000 contracts.
Each earnings event needs 2 chain pulls (T-5 pre, T+1 post) at ~10 ATM
strikes × 2 expiries = ~40 contracts per pull → ~0.08 credits per event.
At 8 events/year × 2 years × 10 symbols = 160 events → roughly 13 credits.
Running the full 100-symbol universe over 2 years stays well under 200 credits.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env so MARKETDATA_TOKEN is available when running outside FastAPI.
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    # python-dotenv not installed — fall back to manual export.
    # Fix: pip3 install python-dotenv  OR  export MARKETDATA_TOKEN=your_token
    print(
        "  ⚠  python-dotenv is not installed. "
        ".env will not be loaded automatically.\n"
        "     Run:  pip3 install python-dotenv\n"
        "     Or:   export MARKETDATA_TOKEN=your_token_here\n"
    )
except Exception:
    pass

# ── Tier definitions (subset of INSTITUTIONAL_UNIVERSE) ──────────────────────
TIER_TOP10 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "AMD", "NFLX", "CRM",
]

TIER_TOP25 = TIER_TOP10 + [
    "JPM", "BAC", "GS", "V", "MA",
    "UNH", "LLY", "ABBV", "JNJ", "PFE",
    "HD", "WMT", "COST", "ADBE", "NOW",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_credits(symbols: List[str], years: int) -> float:
    """
    Rough credit estimate: 2 chain pulls per event × ~40 contracts each
    = ~0.08 credits per event.  ~8 earnings events per symbol per year.
    """
    events_per_symbol_per_year = 8
    contracts_per_pull = 40          # 10 ATM strikes × 2 expiries × 2 sides
    credits_per_1k_contracts = 1.0
    pulls_per_event = 2              # T-5 pre + T+1 post
    total_events = len(symbols) * years * events_per_symbol_per_year
    total_contracts = total_events * pulls_per_event * contracts_per_pull
    return total_contracts / 1_000.0


def _print_pairing_report(progress: dict) -> None:
    total      = int(progress.get("total_snapshots", 0))
    events     = int(progress.get("total_events", 0))
    pairable   = int(progress.get("pairable_events", 0))
    pct        = float(progress.get("pairable_event_pct", 0.0))
    pre_only   = int(progress.get("pending_pre_only_events", 0))
    post_only  = int(progress.get("pending_post_only_events", 0))
    unqualified = int(progress.get("unqualified_events", 0))

    print()
    print("=" * 60)
    print("  SNAPSHOT PAIRING REPORT")
    print("=" * 60)
    print(f"  Total snapshots stored : {total:>6}")
    print(f"  Distinct events        : {events:>6}")
    print(f"  Fully paired (replay-  : {pairable:>6}  ({pct:.1f}%)")
    print(f"    ready)               ")
    print(f"  Pre-only (missing post): {pre_only:>6}")
    print(f"  Post-only (missing pre): {post_only:>6}")
    print(f"  Unqualified (no IV)    : {unqualified:>6}")
    print("=" * 60)

    if pairable == 0:
        print()
        print("  ⚠  NO paired events yet. The replay backtest path will")
        print("     still fall through to synthetic returns. Re-run with")
        print("     a wider --years window or check MARKETDATA_TOKEN.")
    elif pct < 50.0:
        print()
        print(f"  ⚠  Only {pct:.0f}% of events are replay-ready. Consider")
        print("     extending --years or adding more symbols.")
    else:
        print()
        print(f"  ✓  {pct:.0f}% of events are replay-ready. The walk-forward")
        print("     backtest will use real option economics for those trades.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed historical IV snapshots for replay-based backtest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tier",
        choices=["top10", "top25", "full"],
        default="top10",
        help="Predefined symbol tier to seed (default: top10).",
    )
    group.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (overrides --tier).",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Lookback window in years (default: 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be captured and estimated credits; do not call the API.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite DB (default: ~/.options_calculator_pro/institutional_ml.db).",
    )

    args = parser.parse_args()

    # ── Resolve symbol list ───────────────────────────────────────────────────
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.tier == "top25":
        symbols = TIER_TOP25
    elif args.tier == "full":
        from services.institutional_ml_db import INSTITUTIONAL_UNIVERSE
        symbols = list(INSTITUTIONAL_UNIVERSE)
    else:
        symbols = TIER_TOP10

    # ── Check token ──────────────────────────────────────────────────────────
    token = os.environ.get("MARKETDATA_TOKEN", "").strip()
    if not token:
        print()
        print("  ✗  MARKETDATA_TOKEN is not set.")
        print("     Set it in your environment or in .env before running.")
        print()
        return 1

    # ── Dry run ───────────────────────────────────────────────────────────────
    estimated_credits = _estimate_credits(symbols, args.years)
    print()
    print("=" * 60)
    print("  REPLAY SNAPSHOT SEEDING")
    print("=" * 60)
    print(f"  Symbols        : {len(symbols)}  ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"  Lookback       : {args.years} year(s)")
    print(f"  Estimated events: ~{len(symbols) * args.years * 8}")
    print(f"  Estimated credits: ~{estimated_credits:.0f}  (of 100k daily budget)")
    print(f"  Token          : ...{token[-6:]}")
    print("=" * 60)
    print()

    if args.dry_run:
        print("  DRY RUN — no API calls made. Remove --dry-run to execute.")
        print()
        return 0

    # ── Imports (deferred so --dry-run doesn't need heavy deps) ─────────────
    try:
        from services.market_data_client import MarketDataClient
        from services.institutional_ml_db import InstitutionalMLDatabase
    except ImportError as exc:
        print(f"  ✗  Import error: {exc}")
        print("     Run from the project root: python scripts/seed_replay_snapshots.py")
        return 1

    # ── DB path guard ─────────────────────────────────────────────────────────
    _default_db = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
    _test_db    = _ROOT / "tmp" / "institutional_ml_test.db"
    if args.db_path is None and not _default_db.exists():
        print()
        print("  ⚠  WARNING: default DB does not exist at:")
        print(f"       {_default_db}")
        if _test_db.exists():
            print()
            print(f"  Found a populated DB at:  {_test_db}")
            print("  If that is your working DB, re-run with:")
            print(f"    --db-path {_test_db}")
            print()
        else:
            print()
            print("  No existing DB found. A new one will be created at the default path.")
            print("  If you intended to seed an existing DB, pass --db-path explicitly.")
            print()

    # ── Initialise client and DB ──────────────────────────────────────────────
    print("  Initialising MarketData.app client …")
    mda = MarketDataClient(token=token)
    if not mda.is_available():
        print("  ✗  MarketDataClient reports unavailable. Check your token.")
        return 1

    print("  Initialising database …")
    db = InstitutionalMLDatabase(db_path=args.db_path, mda_client=mda)

    # ── Snapshot pairing status BEFORE capture ───────────────────────────────
    print()
    print("  Snapshot coverage BEFORE this run:")
    before = db.summarize_snapshot_pairing_progress()
    _print_pairing_report(before)

    # ── Run capture ───────────────────────────────────────────────────────────
    print(f"  Starting capture for {len(symbols)} symbol(s) …")
    print("  (This may take several minutes for large symbol lists.)")
    print()

    t0 = time.time()
    result = db.capture_historical_iv_snapshots_mda(
        symbols=symbols,
        lookback_years=args.years,
    )
    elapsed = time.time() - t0

    captured = int(result.get("captured", 0))
    skipped  = int(result.get("skipped", 0))
    errors   = int(result.get("errors", 0))
    reason   = result.get("reason", "")

    print()
    print(f"  Capture complete in {elapsed:.1f}s")
    print(f"    Captured : {captured}")
    print(f"    Skipped  : {skipped}")
    print(f"    Errors   : {errors}")
    if reason:
        print(f"    Reason   : {reason}")

    if reason == "no_mda_client":
        print()
        print("  ✗  MarketData.app client not available during capture.")
        print("     This should not happen if the token check above passed.")
        print("     Check that InstitutionalMLDatabase received the mda_client argument.")
        return 1

    # ── Snapshot pairing status AFTER capture ────────────────────────────────
    print()
    print("  Snapshot coverage AFTER this run:")
    after = db.summarize_snapshot_pairing_progress()
    _print_pairing_report(after)

    # ── Final verdict ─────────────────────────────────────────────────────────
    pairable_after = int(after.get("pairable_events", 0))
    pairable_before = int(before.get("pairable_events", 0))
    new_pairs = pairable_after - pairable_before

    print("  NEXT STEPS")
    print("  ----------")
    if pairable_after == 0:
        print("  The snapshot table is still empty or has no paired events.")
        print("  Check MARKETDATA_TOKEN permissions and try --years 1 first.")
    else:
        print(f"  +{new_pairs} new paired events added this run.")
        print(f"  {pairable_after} total paired events available for replay.")
        print()
        print("  Run a walk-forward backtest now. For each event where a")
        print("  pre/post snapshot pair exists, the backtest will use real")
        print("  option economics. All remaining events fall back to synthetic.")
        print()
        print("  To check what the backtest is using, look for log lines from:")
        print("    _simulate_snapshot_replay_trade  ← real replay")
        print("    _simulate_walk_forward_trade     ← synthetic fallback")
    print()

    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
