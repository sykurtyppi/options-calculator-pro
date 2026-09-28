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

IMPORTANT — Store isolation (Phase 1.4)
----------------------------------------
By default this script writes to an ISOLATED temp directory under tmp/seed_run_<utc>/,
NOT the production store.  Inspect the dry-run output and the temp results before
promoting anything to the live store.

To write to the production store you must pass --target=production explicitly.
Omitting --target (or passing --target=tmp) is always safe.

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

  # Seed into an isolated temp directory (default, safe):
  .venv_arm64/bin/python scripts/seed_outcomes_from_replay.py

  # Seed into production stores (requires explicit flag):
  .venv_arm64/bin/python scripts/seed_outcomes_from_replay.py --target=production

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
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
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


def _is_unfinished_replay_row(existing: Dict[str, Any]) -> bool:
    """A replay row this script inserted but never gave an exit (interrupted run)."""
    from services.outcome_recorder import is_outcome_evidence_valid

    return (
        existing.get("source_type") == "replay"
        and existing.get("status") == "open"
        and not existing.get("exit_date")
        and is_outcome_evidence_valid(existing)
    )


def _needs_replay_relearn(existing: Dict[str, Any]) -> bool:
    """A seeded replay row that finalized but never finished its learning."""
    from services.outcome_recorder import is_outcome_evidence_valid

    return (
        existing.get("source_type") == "replay"
        and existing.get("status") == "finalized"
        and existing.get("learning_update_status") != "complete"
        and is_outcome_evidence_valid(existing)
        and all(
            existing.get(key) is not None
            for key in ("setup_score", "realized_return_pct", "realized_expansion_pct")
        )
    )


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
    skipped_conflict = 0
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
        try:
            realized_pnl = float(realized_pnl) if realized_pnl is not None else None
        except (TypeError, ValueError):
            realized_pnl = float("nan")
        # The outcome store refuses NaN/inf; checking only at the exit write
        # would stop the run after the entry was inserted, leaving an open row
        # that no later run repairs.
        if not all(
            math.isfinite(value)
            for value in (setup_score, realized_expansion_pct, realized_return_pct)
            + ((realized_pnl,) if realized_pnl is not None else ())
        ):
            skipped_bad_data += 1
            continue

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

        trade_id = make_trade_id(symbol, entry_date, row_structure, earnings_date=earnings_date)

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

        existing = {} if was_new else (store.get_trade(trade_id) or {})
        if not was_new and _is_unfinished_replay_row(existing):
            # An earlier run was interrupted between the insert and the exit:
            # complete it now, like a fresh insert.
            was_new = True
        if not was_new:
            skipped_duplicate += 1
            # Re-apply learning only to a row THIS script seeded and finalized
            # whose learning never completed (e.g. a failed store write). A
            # paper trade can share the id format; learning replay numbers
            # under its id would make its real outcome a duplicate later.
            if not _needs_replay_relearn(existing):
                continue
            # Learn what the row records, not this run's replay numbers.
            setup_score = float(existing["setup_score"])
            realized_return_pct = float(existing["realized_return_pct"])
            realized_expansion_pct = float(existing["realized_expansion_pct"])
        else:
            # Mark as finalized immediately — replay trades have no "open" phase.
            # Learn only if both writes landed: a row something else changed
            # meanwhile (e.g. invalidated) must not reach the learning stores.
            exit_written = store.update_exit(
                trade_id=trade_id,
                exit_date=earnings_date or entry_date,
                realized_return_pct=realized_return_pct,
                realized_pnl=float(realized_pnl) if realized_pnl is not None else None,
                realized_expansion_pct=realized_expansion_pct,
            )
            if not (exit_written and store.mark_finalized(trade_id)):
                logger.warning("seed: trade_id=%s could not be finalized; learning skipped", trade_id)
                skipped_conflict += 1
                continue

        # ── Update calibration ────────────────────────────────────────────
        obs_date = earnings_date or entry_date
        if obs_date is None:
            # First live exercise of this guard is the first real seed run — watch the log for
            # skip warnings; any skipped row means the replay source is missing both dates.
            logger.warning(
                "seed: trade_id=%s has no earnings_date or entry_date — skipping observation",
                trade_id,
            )
            continue
        # Learning writes can raise (PersistenceError, ValueError); record the
        # outcome per trade so a failure is retryable instead of the trade
        # being finalized with no learning and no way to find it again.
        calibration_ok = prior_ok = False
        try:
            if cal.update(
                setup_score,
                realized_expansion_pct,
                observation_id=trade_id,
                source_type="replay",
                observation_date=obs_date,
            ):
                cal_updates += 1
            calibration_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.error("seed: calibration update failed for %s (%s)", trade_id, exc)

        # ── Update structure prior ────────────────────────────────────────
        try:
            ps.update(
                structure=row_structure,
                realized_return_pct=realized_return_pct,
                realized_expansion_pct=realized_expansion_pct,
                source_type="replay",
                observation_date=obs_date,
                observation_id=trade_id,
            )
            prior_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.error("seed: structure prior update failed for %s (%s)", trade_id, exc)

        store.set_learning_update_status(
            trade_id,
            "complete" if calibration_ok and prior_ok
            else "both_failed" if not (calibration_ok or prior_ok)
            else "calibration_failed" if not calibration_ok
            else "prior_failed",
        )
        if not was_new:
            continue

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
        "skipped_conflict": skipped_conflict,
        "cal_updates": cal_updates,
        "cal_phase_after": cal_phase,
        "cal_n_after": cal_n,
        "by_year": dict(sorted(by_year.items())),
        "by_structure": dict(by_structure),
        "prior_diagnostics": prior_diag,
        "dry_run": dry_run,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


_PRODUCTION_OUTCOME_STORE = (
    Path.home() / ".options_calculator_pro" / "outcomes" / "outcome_store.sqlite"
)
_PRODUCTION_CALIBRATION_STORE = (
    Path.home() / ".options_calculator_pro" / "calibration" / "iv_expansion.json"
)
_PRODUCTION_PRIOR_STORE = (
    Path.home() / ".options_calculator_pro" / "priors" / "structure_priors.json"
)


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
    parser.add_argument(
        "--target",
        type=str,
        default="tmp",
        choices=["tmp", "production"],
        help=(
            "Where to write seeded observations.  "
            "'tmp' (default): isolated directory under tmp/seed_run_<utc>/, safe to inspect and discard.  "
            "'production': the live stores under ~/.options_calculator_pro/ — requires explicit opt-in."
        ),
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

    # ── Resolve store paths based on --target ─────────────────────────────────
    if args.dry_run:
        outcome_store_path: Optional[Path] = None
        calibration_store_path: Optional[Path] = None
        prior_store_path: Optional[Path] = None
        target_label = "DRY RUN (no writes)"
    elif args.target == "production":
        outcome_store_path = _PRODUCTION_OUTCOME_STORE
        calibration_store_path = _PRODUCTION_CALIBRATION_STORE
        prior_store_path = _PRODUCTION_PRIOR_STORE
        target_label = f"PRODUCTION — {Path.home() / '.options_calculator_pro'}"
    else:
        # datetime.utcnow() is deprecated in Python 3.12+; use the
        # timezone-aware equivalent. The 'Z' suffix in the format string
        # documents that the timestamp is UTC.
        utc_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tmp_root = _ROOT / "tmp" / f"seed_run_{utc_tag}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        outcome_store_path = tmp_root / "outcome_store.sqlite"
        calibration_store_path = tmp_root / "iv_expansion.json"
        prior_store_path = tmp_root / "structure_priors.json"
        target_label = f"ISOLATED TEMP — {tmp_root}"

    print()
    print("=" * 60)
    print("  OUTCOME SEEDING FROM REPLAY BACKTEST")
    print("=" * 60)
    print(f"  DB            : {db_path}")
    print(f"  Session       : {args.session_id or 'all'}")
    print(f"  Symbols       : {', '.join(symbols) if symbols else 'all'}")
    print(f"  Structure     : {args.structure} (assumed for all trades)")
    print(f"  Target        : {target_label}")
    print()

    if args.target == "production" and not args.dry_run:
        print("  ⚠  Writing to PRODUCTION stores.  This modifies live learning state.")
        print("  Calibration updates are idempotent by stable replay trade ID.\n")
    elif not args.dry_run:
        print("  Writing to isolated temp directory.  Pass --target=production to promote.\n")

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
        outcome_store_path=outcome_store_path,
        calibration_store_path=calibration_store_path,
        prior_store_path=prior_store_path,
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
    print(f"{prefix}Skipped (conflict) : {result['skipped_conflict']}")

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

    if not args.dry_run and args.target == "tmp":
        print()
        print(f"  Results written to: {outcome_store_path.parent}")
        print("  Inspect and promote to production with --target=production if satisfied.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
