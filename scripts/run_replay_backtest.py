#!/usr/bin/env python3
"""
run_replay_backtest.py
======================
Run a walk-forward backtest in hybrid pricing mode so replay-ready events
use real option economics and the rest fall back to synthetic.

After the run, prints:
  - How many trades used REPLAY vs SYNTHETIC (from logs)
  - Session ID for looking up full results in the DB
  - Top-level P&L summary

Usage
-----
  # Default: top10 symbols, hybrid mode, 2-year window
  .venv_arm64/bin/python3 scripts/run_replay_backtest.py --db-path tmp/institutional_ml_test.db

  # Force snapshot-only (skips trades with no replay pair — pure validation)
  .venv_arm64/bin/python3 scripts/run_replay_backtest.py --db-path tmp/institutional_ml_test.db --mode snapshot_replay

  # Custom symbol list
  .venv_arm64/bin/python3 scripts/run_replay_backtest.py --db-path tmp/institutional_ml_test.db --symbols AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


# ── Intercept REPLAY vs SYNTHETIC log lines ───────────────────────────────────

class _ReplayCounter(logging.Handler):
    """Counts how many trades went through replay vs synthetic paths."""
    def __init__(self):
        super().__init__()
        self.replay   = 0
        self.synthetic = 0
        self.skipped   = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "📸 REPLAY" in msg:
            self.replay += 1
        elif "🔮 SYNTHETIC" in msg:
            self.synthetic += 1
        elif "⏭  SKIP" in msg:
            self.skipped += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run replay-aware walk-forward backtest")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to SQLite DB (default: ~/.options_calculator_pro/institutional_ml.db)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: top10 institutional universe)")
    parser.add_argument("--years", type=int, default=2,
                        help="Lookback window in years (default: 2)")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["hybrid", "snapshot_replay", "synthetic"],
                        help="Pricing mode: hybrid=replay when available, "
                             "snapshot_replay=replay only (skips others), "
                             "synthetic=always use proxy (baseline comparison)")
    parser.add_argument("--max-debit", type=float, default=None,
                        help="Max entry debit per contract in dollars. "
                             "Trades exceeding this are skipped. "
                             "Suggested calibrated value: 600 (filters NFLX Apr-2025 outlier).")
    args = parser.parse_args()

    # ── DB path guard ─────────────────────────────────────────────────────────
    _default_db = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
    _test_db    = _ROOT / "tmp" / "institutional_ml_test.db"
    if args.db_path is None and not _default_db.exists():
        print()
        print("  ⚠  Default DB not found at:")
        print(f"       {_default_db}")
        if _test_db.exists():
            print(f"  Found populated DB at: {_test_db}")
            print("  Re-run with:  --db-path tmp/institutional_ml_test.db")
        print()
        return 1

    # ── Imports ────────────────────────────────────────────────────────────────
    try:
        from services.institutional_ml_db import InstitutionalMLDatabase, INSTITUTIONAL_UNIVERSE
    except ImportError as exc:
        print(f"  ✗  Import error: {exc}")
        print("     Use .venv_arm64/bin/python3 and run from project root.")
        return 1

    # ── Wire up the replay counter before DB init (so it catches all logs) ───
    counter = _ReplayCounter()
    counter.setLevel(logging.DEBUG)
    logging.getLogger("services.institutional_ml_db").addHandler(counter)

    # ── Initialise DB ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  REPLAY-AWARE WALK-FORWARD BACKTEST")
    print("=" * 60)

    db = InstitutionalMLDatabase(db_path=args.db_path)

    # ── Build symbol list ─────────────────────────────────────────────────────
    if args.symbols:
        universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        universe = list(INSTITUTIONAL_UNIVERSE)[:10]

    import datetime
    end_date   = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=args.years * 365)

    strategy_params = {
        "universe":        universe,
        "pricing_mode":    args.mode,
        "start_date":      start_date.strftime("%Y-%m-%d"),
        "end_date":        end_date.strftime("%Y-%m-%d"),
        "lookback_days":   args.years * 365,
        "max_symbols":     len(universe),
    }
    if args.max_debit is not None:
        strategy_params["max_entry_debit_per_contract"] = args.max_debit

    print(f"  Symbols      : {len(universe)}  ({', '.join(universe[:5])}{'...' if len(universe) > 5 else ''})")
    print(f"  Window       : {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"  Pricing mode : {args.mode}")
    if args.max_debit:
        print(f"  Debit cap    : ${args.max_debit:.0f}/contract")
    print(f"  DB           : {args.db_path or _default_db}")
    print()
    print("  Running … (this takes 30–120 seconds)")
    print()

    # ── Run ───────────────────────────────────────────────────────────────────
    session_id = db.run_calendar_spread_backtest(strategy_params)

    # ── Results ───────────────────────────────────────────────────────────────
    total_trades = counter.replay + counter.synthetic + counter.skipped

    print()
    print("=" * 60)
    print("  BACKTEST COMPLETE")
    print("=" * 60)
    print(f"  Session ID   : {session_id}")
    print()
    print("  Trade routing breakdown:")
    print(f"    📸 REPLAY    : {counter.replay:>5}  (real option economics)")
    print(f"    🔮 SYNTHETIC : {counter.synthetic:>5}  (proxy fallback)")
    print(f"    ⏭  SKIPPED   : {counter.skipped:>5}  (no pair, mode=snapshot_replay)")
    if total_trades > 0:
        replay_pct = counter.replay / total_trades * 100
        print()
        print(f"  Replay coverage this run: {replay_pct:.1f}%")
        if replay_pct < 15:
            print("  ⚠  Low replay coverage — most P&L still from synthetic proxy.")
            print("     Run seed_replay_snapshots.py first, or upgrade MarketData.app tier.")
        elif replay_pct < 50:
            print("  ⚠  Partial replay coverage. Results are a mix of real and synthetic.")
            print("     Treat replay-only session (--mode snapshot_replay) as ground truth.")
        else:
            print("  ✓  Majority of trades priced from real option data.")
    print()
    print("  To query full results:")
    print(f"    sqlite3 {args.db_path or _default_db} \\")
    print(f'    "SELECT symbol, COUNT(*) trades, ROUND(AVG(net_return_pct)*100,2) avg_ret_pct,')
    print(f'     ROUND(SUM(pnl_per_contract * contracts),2) total_pnl')
    print(f'     FROM backtest_trades WHERE session_id=\\"{session_id}\\"')
    print(f'     GROUP BY symbol ORDER BY total_pnl DESC;"')
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
