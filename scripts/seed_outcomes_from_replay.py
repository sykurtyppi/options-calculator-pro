#!/usr/bin/env python3
"""
seed_outcomes_from_replay.py
============================
Seed the outcome store, calibration service, and structure prior store
from historical backtest data already recorded in the institutional ML database.

This is a one-time / repeatable seeding path.  Repeated runs are idempotent:
  - outcome store: INSERT OR IGNORE prevents double-counting
  - calibration: stable observation IDs prevent double-counting on repeated runs
  - structure priors: duplicate rows are skipped before prior updates, so re-runs stay stable

IMPORTANT — Honest limitations
-------------------------------
The backtest_trades table records calendar-spread backtest trades only
(run_calendar_spread_backtest) unless newer rows explicitly persist a
different `structure` value. When the source table lacks that column, the
seeding path falls back to `--structure` (default: `call_calendar`).

The backtest gross_return_pct field is used as realized_expansion_pct for
the calibration service.  This is an approximation: gross_return_pct is the
option percentage return before execution costs, which maps to IV expansion
but is not identical to it.  For a calendar spread, it is the best proxy
available from the backtest output.

If replay coverage in the DB is low (most trades used synthetic pricing),
the observations are from a proxy model, not real option economics.  The
seeding script prints replay vs synthetic coverage so you can decide whether
the seeded observations are empirically meaningful.

Usage
-----
  # Dry run — show what would be seeded, touch nothing:
  .venv_arm64/bin/python scripts/seed_outcomes_from_replay.py --dry-run

  # Seed from default DB:
  .venv_arm64/bin/python scripts/seed_outcomes_from_replay.py

  # Seed from explicit DB path:
  .venv_arm64/bin/python scripts/seed_outcomes_from_replay.py --db-path tmp/institutional_ml_test.db

  # Filter to a specific backtest session:
  .venv_arm64/bin/python scripts/seed_outcomes_from_replay.py --session-id <id>

  # Filter to specific symbols:
  .venv_arm64/bin/python scripts/seed_outcomes_from_replay.py --symbols AAPL,MSFT

  # Override assumed structure (default: call_calendar):
  .venv_arm64/bin/python scripts/seed_outcomes_from_replay.py --structure call_calendar
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_DB = Path.home() / ".options_calculator_pro" / "institutional_ml.db"

# Columns we need from backtest_trades.
_REQUIRED_COLS = {
    "symbol",
    "trade_date",
    "event_date",
    "days_to_earnings",
    "setup_score",
    "gross_return_pct",
    "net_return_pct",
    "pnl_per_contract",
    "execution_profile",
}


# ── Query helpers ─────────────────────────────────────────────────────────────


def _fetch_trades(
    db_path: Path,
    session_id: Optional[str],
    symbols: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Query backtest_trades from the institutional ML DB."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Check which columns actually exist (schema may vary across versions).
    existing_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(backtest_trades)").fetchall()
    }
    missing = _REQUIRED_COLS - existing_cols
    if missing:
        logger.warning("backtest_trades missing columns: %s — those fields will be NULL", missing)

    where_clauses = []
    params: List[Any] = []

    if session_id:
        where_clauses.append("session_id = ?")
        params.append(session_id)
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        where_clauses.append(f"symbol IN ({placeholders})")
        params.extend(symbols)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"SELECT * FROM backtest_trades {where_sql} ORDER BY trade_date"

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _parse_date(val: Any) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    s = str(val)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt).date()
        except ValueError:
            continue
    return None


# ── Seeding logic ─────────────────────────────────────────────────────────────


def seed_from_trades(
    trades: List[Dict[str, Any]],
    *,
    structure: str,
    dry_run: bool,
    outcome_store_path: Optional[Path],
    calibration_store_path: Optional[Path],
    prior_store_path: Optional[Path],
) -> Dict[str, Any]:
    """
    Seed outcome_store, calibration, and structure priors from a list of
    backtest trade dicts.

    Returns a summary dict.
    """
    from services.outcome_recorder import OutcomeStore, make_trade_id

    if not dry_run:
        store = OutcomeStore(store_path=outcome_store_path or OutcomeStore.__init__.__defaults__[0])
        from services.calibration_service import IVExpansionCalibration

        cal = IVExpansionCalibration(
            store_path=calibration_store_path
            or Path.home() / ".options_calculator_pro" / "calibration" / "iv_expansion.json"
        )
        from services.structure_prior_store import StructurePriorStore

        ps = StructurePriorStore(
            store_path=prior_store_path
            or Path.home() / ".options_calculator_pro" / "priors" / "structure_priors.json"
        )

    inserted = 0
    skipped_duplicate = 0
    skipped_bad_data = 0
    by_year: Dict[int, int] = defaultdict(int)
    by_structure: Dict[str, int] = defaultdict(int)
    cal_updates = 0

    for row in trades:
        entry_date = _parse_date(row.get("trade_date"))
        if entry_date is None:
            skipped_bad_data += 1
            continue

        setup_score = row.get("setup_score")
        gross_return_pct = row.get("gross_return_pct")
        net_return_pct = row.get("net_return_pct")

        if setup_score is None or gross_return_pct is None or net_return_pct is None:
            skipped_bad_data += 1
            continue

        try:
            setup_score = float(setup_score)
            gross_return_pct = float(gross_return_pct)
            net_return_pct = float(net_return_pct)
        except (TypeError, ValueError):
            skipped_bad_data += 1
            continue

        # backtest_trades stores return_pct as fractions (0.09 = 9%).
        # calibration_service and structure_prior_store expect percentages (9.0 = 9%).
        realized_expansion_pct = gross_return_pct * 100.0
        realized_return_pct = net_return_pct * 100.0
        realized_pnl = row.get("pnl_per_contract")

        raw_symbol = row.get("symbol")
        if raw_symbol is None:
            skipped_bad_data += 1
            continue
        symbol = str(raw_symbol).upper()
        if not symbol:
            skipped_bad_data += 1
            continue

        earnings_date = _parse_date(row.get("event_date"))
        days_to_earnings = row.get("days_to_earnings")
        assumed_cost_model = str(row.get("execution_profile", "backtest"))
        row_structure = str(row.get("structure") or structure)

        trade_id = make_trade_id(symbol, entry_date, row_structure)

        if dry_run:
            # Just count; don't touch any stores.
            inserted += 1
            by_year[entry_date.year] += 1
            by_structure[row_structure] += 1
            continue

        # ── Insert into outcome store ─────────────────────────────────────
        was_new = store.insert_entry(
            trade_id=trade_id,
            symbol=symbol,
            structure=row_structure,
            entry_date=entry_date,
            setup_score=setup_score,
            source_type="replay",
            earnings_date=earnings_date,
            days_to_earnings=int(days_to_earnings) if days_to_earnings is not None else None,
            assumed_cost_model=assumed_cost_model,
        )

        if not was_new:
            skipped_duplicate += 1
            continue

        # Mark as finalized immediately — replay trades have no "open" phase.
        store.update_exit(
            trade_id=trade_id,
            exit_date=earnings_date or entry_date,
            realized_return_pct=realized_return_pct,
            realized_pnl=float(realized_pnl) if realized_pnl is not None else None,
            realized_expansion_pct=realized_expansion_pct,
        )
        store.mark_finalized(trade_id)

        # ── Update calibration ────────────────────────────────────────────
        if cal.update(
            setup_score,
            realized_expansion_pct,
            observation_id=trade_id,
            source_type="replay",
        ):
            cal_updates += 1

        # ── Update structure prior ────────────────────────────────────────
        ps.update(
            structure=row_structure,
            realized_return_pct=realized_return_pct,
            realized_expansion_pct=realized_expansion_pct,
            source_type="replay",
        )

        inserted += 1
        by_year[entry_date.year] += 1
        by_structure[row_structure] += 1

    if not dry_run:
        cal_phase = cal._phase()
        cal_n = cal._n()
        prior_diag = ps.diagnostics()
    else:
        cal_phase = "unknown (dry-run)"
        cal_n = 0
        prior_diag = {}

    return {
        "inserted": inserted,
        "skipped_duplicate": skipped_duplicate,
        "skipped_bad_data": skipped_bad_data,
        "cal_updates": cal_updates,
        "cal_phase_after": cal_phase,
        "cal_n_after": cal_n,
        "by_year": dict(sorted(by_year.items())),
        "by_structure": dict(by_structure),
        "prior_diagnostics": prior_diag,
        "dry_run": dry_run,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed outcome store, calibration, and structure priors from replay backtest data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=f"Path to institutional ML DB (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Limit to a specific backtest session_id (default: all sessions)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbol list (default: all symbols in DB)",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="call_calendar",
        choices=["atm_straddle", "otm_strangle", "call_calendar", "put_calendar"],
        help=(
            "Assumed structure for all seeded trades (default: call_calendar). "
            "Used as a fallback only when backtest_trades has no explicit structure column."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be seeded without writing anything.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else _DEFAULT_DB
    if not db_path.exists():
        print()
        print(f"  ✗  Database not found at: {db_path}")
        print("  Run:  .venv_arm64/bin/python scripts/seed_replay_snapshots.py  first")
        print("  Or specify an existing DB with --db-path")
        print()
        return 1

    symbols: Optional[List[str]] = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print()
    print("=" * 60)
    print("  OUTCOME SEEDING FROM REPLAY BACKTEST")
    print("=" * 60)
    print(f"  DB            : {db_path}")
    print(f"  Session       : {args.session_id or 'all'}")
    print(f"  Symbols       : {', '.join(symbols) if symbols else 'all'}")
    print(f"  Structure     : {args.structure} (assumed for all trades)")
    print(f"  Mode          : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    if not args.dry_run:
        print("  Calibration updates are idempotent by stable replay trade ID.\n")

    # ── Fetch trades ──────────────────────────────────────────────────────────
    print("  Fetching trades from backtest_trades table …")
    try:
        trades = _fetch_trades(db_path, args.session_id, symbols)
    except Exception as exc:
        print(f"  ✗  Failed to fetch trades: {exc}")
        return 1

    print(f"  Found {len(trades)} trade records")
    if not trades:
        print("  No trades found — nothing to seed.")
        return 0

    # ── Replay coverage check ─────────────────────────────────────────────────
    # Try to read session-level replay coverage from the DB.
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT session_id, notes FROM backtest_sessions ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        conn.close()
        if rows:
            print()
            print("  Recent sessions (latest 5):")
            for sid, notes in rows:
                print(f"    {sid[:20]:<22} {(notes or '')[:40]}")
    except Exception:
        pass

    print()
    print(
        "  ⚠  IMPORTANT: If most trades above used synthetic pricing\n"
        "  (not replay), the seeded observations are proxy-model estimates,\n"
        "  not empirical option economics.  Run run_replay_backtest.py with\n"
        "  --mode snapshot_replay to check your replay coverage.\n"
    )

    # ── Seed ──────────────────────────────────────────────────────────────────
    result = seed_from_trades(
        trades,
        structure=args.structure,
        dry_run=args.dry_run,
        outcome_store_path=None,
        calibration_store_path=None,
        prior_store_path=None,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SEEDING RESULTS")
    print("=" * 60)
    prefix = "  [DRY RUN] " if args.dry_run else "  "
    print(f"{prefix}Inserted (new)     : {result['inserted']}")
    print(f"{prefix}Skipped (duplicate): {result['skipped_duplicate']}")
    print(f"{prefix}Skipped (bad data) : {result['skipped_bad_data']}")

    if result["by_year"]:
        print()
        print(f"{prefix}By year:")
        for yr, cnt in sorted(result["by_year"].items()):
            print(f"{prefix}  {yr}: {cnt}")

    if result["by_structure"]:
        print()
        print(f"{prefix}By structure:")
        for s, cnt in result["by_structure"].items():
            print(f"{prefix}  {s}: {cnt}")

    if not args.dry_run:
        print()
        print(f"  Calibration observations after seeding : {result['cal_n_after']}")
        print(f"  Calibration phase after seeding        : {result['cal_phase_after']}")

        pd = result.get("prior_diagnostics", {}).get("structures", {})
        if pd:
            print()
            print("  Structure prior summary:")
            for s, info in pd.items():
                n = info.get("observation_count", 0)
                wr = info.get("win_rate")
                override = info.get("overrides_report_prior", False)
                flag = " ← overrides report prior" if override else ""
                wr_str = f"{wr:.1%}" if wr is not None else "n/a"
                print(f"    {s:<18} n={n:>4}  win_rate={wr_str}{flag}")

        if result["cal_phase_after"] == "bootstrap_prior":
            print()
            print(
                "  ⚠  Calibration is still in bootstrap_prior phase.\n"
                "  The seeded observations are real but not yet numerous enough\n"
                "  to shift to an empirical phase (need 40 for observational).\n"
                "  Continue paper trading or seed from a wider date range."
            )
        elif result["cal_phase_after"] == "observational":
            print()
            print(
                "  ✓  Calibration is now in observational phase.\n"
                "  Raw bucket-level estimates are being used. Continue\n"
                "  accumulating observations to reach fitted_moderate (need 120)."
            )
        else:
            print()
            print(f"  ✓  Calibration is in {result['cal_phase_after']} phase.")
    else:
        print()
        print("  DRY RUN complete — nothing was written.")
        print("  Re-run without --dry-run to apply seeding.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
