"""
Backfill per-observation timestamps into the structure prior store and
calibration store from the outcome_trades SQLite table.

Context / motivation
--------------------
Prior to Phase 1.1-1.2 (issue #18), StructurePriorStore and
IVExpansionCalibration did not record per-observation dates.  The schema
migration on load synthesises a single aggregate placeholder dated at
last_updated, but individual trade dates are lost.

This script reconstructs the observations list from outcome_trades
(which has per-trade entry_date and exit_date) and rewrites the stores
with proper per-observation timestamps.  After running it, as_of_date
filtering will reflect the actual trade dates rather than a single
aggregate placeholder.

Usage
-----
  # Show what would be written, write nothing:
  python scripts/backfill_prior_store_timestamps.py --dry-run

  # Write to the default production stores:
  python scripts/backfill_prior_store_timestamps.py --target=production

Safety
------
- Default is --dry-run.  You must pass --target=production to touch files.
- The existing JSON files are backed up to <name>.pre_migration_backup
  before being overwritten.
- The outcome_trades table is read-only; this script never writes to SQLite.
- Run with --dry-run first and inspect the output before using
  --target=production.

Requirements
------------
The outcome_trades table must have these columns (added in earlier schema):
  trade_id, structure, source_type, entry_date, exit_date,
  realized_return_pct, realized_expansion_pct, setup_score
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTCOME_DB = Path.home() / ".options_calculator_pro" / "outcomes" / "outcome_store.sqlite"
_DEFAULT_PRIOR_STORE = Path.home() / ".options_calculator_pro" / "priors" / "structure_priors.json"
_DEFAULT_CAL_STORE = Path.home() / ".options_calculator_pro" / "calibration" / "iv_expansion.json"

SUPPORTED_STRUCTURES = ("atm_straddle", "otm_strangle", "call_calendar", "put_calendar")


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


def _fetch_trades(db_path: Path) -> List[Dict[str, Any]]:
    """Read finalized outcome_trades rows from the SQLite store."""
    if not db_path.exists():
        raise FileNotFoundError(f"outcome_store not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_id, structure, source_type, entry_date, exit_date,
                   realized_return_pct, realized_expansion_pct, setup_score
            FROM outcome_trades
            WHERE finalized = 1
              AND realized_return_pct IS NOT NULL
              AND realized_expansion_pct IS NOT NULL
            ORDER BY exit_date, entry_date, trade_id
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise RuntimeError(f"Failed to query outcome_trades: {exc}") from exc
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _backup(path: Path, dry_run: bool) -> None:
    backup = path.with_suffix(path.suffix + ".pre_migration_backup")
    if path.exists():
        if not dry_run:
            shutil.copy2(str(path), str(backup))
            logger.info("Backed up %s → %s", path, backup)
        else:
            logger.info("[DRY RUN] Would back up %s → %s", path, backup)


def _rebuild_prior_store(
    trades: List[Dict[str, Any]],
    existing_path: Path,
    dry_run: bool,
) -> None:
    """Rebuild the structure prior store observations list from outcome_trades."""
    logger.info("Rebuilding structure prior store …")

    # Load existing JSON to preserve schema_version and aggregate fields
    existing: Dict[str, Any] = {}
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
        except Exception as exc:
            logger.warning("Could not read existing prior store: %s", exc)

    structures: Dict[str, Any] = existing.get("structures", {})

    # Reset observations lists; aggregates are recomputed below
    obs_by_structure: Dict[str, List[Dict]] = {s: [] for s in SUPPORTED_STRUCTURES}

    skipped = 0
    for row in trades:
        structure = str(row.get("structure") or "")
        if structure not in SUPPORTED_STRUCTURES:
            skipped += 1
            continue

        exit_d = _parse_date(row.get("exit_date"))
        entry_d = _parse_date(row.get("entry_date"))
        obs_date = exit_d or entry_d
        if obs_date is None:
            skipped += 1
            continue

        realized_return = row.get("realized_return_pct")
        realized_expansion = row.get("realized_expansion_pct")
        if realized_return is None or realized_expansion is None:
            skipped += 1
            continue

        obs_by_structure[structure].append({
            "observation_id": str(row.get("trade_id", "")),
            "observation_date": obs_date.isoformat(),
            "source_type": str(row.get("source_type") or "paper"),
            "realized_return_pct": float(realized_return),
            "realized_expansion_pct": float(realized_expansion),
        })

    if skipped:
        logger.warning("Skipped %d trades (missing structure, dates, or return data)", skipped)

    from services.structure_prior_store import (
        _compute_rank_score, _recompute_aggregates, LAPLACE_SMOOTHING_THRESHOLD,
        MIN_OBS_FOR_OVERRIDE, _SCHEMA_VERSION,
    )

    new_structures: Dict[str, Any] = {}
    for s in SUPPORTED_STRUCTURES:
        obs_list = obs_by_structure[s]
        existing_entry = structures.get(s, {})

        if not obs_list:
            # Keep existing entry as-is if it has data; else initialize empty
            new_structures[s] = dict(existing_entry) if existing_entry else {
                "structure": s,
                "schema_version": _SCHEMA_VERSION,
                "observations": [],
                "observation_count": 0,
                "positive_count": 0,
                "sum_return_pct": 0.0,
                "sum_expansion_pct": 0.0,
                "win_rate": 0.50,
                "avg_return_pct": 0.0,
                "avg_realized_expansion_pct": 0.0,
                "rank_score": 0.50,
                "source_types": {},
                "last_updated": None,
            }
            if "observations" not in new_structures[s]:
                new_structures[s]["observations"] = []
            continue

        agg = _recompute_aggregates(obs_list)
        entry: Dict[str, Any] = {
            "structure": s,
            "schema_version": _SCHEMA_VERSION,
            "observations": obs_list,
            **agg,
            "last_updated": existing_entry.get("last_updated"),
        }
        new_structures[s] = entry
        logger.info(
            "  %-20s n=%3d  win_rate=%.2f  avg_ret=%.2f%%",
            s, agg["observation_count"], agg["win_rate"], agg["avg_return_pct"],
        )

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "structures": new_structures,
    }

    if dry_run:
        print("\n[DRY RUN] Would write prior store:")
        print(json.dumps(payload, indent=2)[:2000], "…" if len(json.dumps(payload)) > 2000 else "")
    else:
        _backup(existing_path, dry_run=False)
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote prior store → %s", existing_path)


def _rebuild_calibration_store(
    trades: List[Dict[str, Any]],
    existing_path: Path,
    dry_run: bool,
) -> None:
    """Rebuild the calibration store timestamps list from outcome_trades."""
    logger.info("Rebuilding calibration store …")

    existing: Dict[str, Any] = {}
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
        except Exception as exc:
            logger.warning("Could not read existing calibration store: %s", exc)

    scores_out: List[float] = []
    expansions_out: List[float] = []
    sources_out: List[str] = []
    timestamps_out: List[str] = []
    obs_ids_out: List[str] = []

    skipped = 0
    for row in trades:
        setup_score = row.get("setup_score")
        realized_expansion = row.get("realized_expansion_pct")
        if setup_score is None or realized_expansion is None:
            skipped += 1
            continue

        exit_d = _parse_date(row.get("exit_date"))
        entry_d = _parse_date(row.get("entry_date"))
        obs_date = exit_d or entry_d
        if obs_date is None:
            skipped += 1
            continue

        scores_out.append(float(setup_score))
        expansions_out.append(float(realized_expansion))
        sources_out.append(str(row.get("source_type") or "paper"))
        timestamps_out.append(obs_date.isoformat())
        obs_ids_out.append(str(row.get("trade_id", "")))

    if skipped:
        logger.warning("Skipped %d trades from calibration rebuild", skipped)

    payload = {
        "schema_version": 2,
        "scores": scores_out,
        "expansions": expansions_out,
        "sources": sources_out,
        "timestamps": timestamps_out,
        "observation_ids": sorted(set(obs_ids_out)),
        "n": len(scores_out),
    }
    logger.info("Calibration: %d observations", len(scores_out))

    if dry_run:
        print("\n[DRY RUN] Would write calibration store:")
        print(json.dumps(payload, indent=2)[:2000], "…" if len(json.dumps(payload)) > 2000 else "")
    else:
        _backup(existing_path, dry_run=False)
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote calibration store → %s", existing_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=f"Path to outcome_store SQLite (default: {_DEFAULT_OUTCOME_DB})",
    )
    parser.add_argument(
        "--prior-store",
        default=None,
        help=f"Path to structure prior store JSON (default: {_DEFAULT_PRIOR_STORE})",
    )
    parser.add_argument(
        "--cal-store",
        default=None,
        help=f"Path to calibration store JSON (default: {_DEFAULT_CAL_STORE})",
    )
    parser.add_argument(
        "--target",
        default=None,
        choices=["production"],
        help=(
            "Pass --target=production to actually write files.  "
            "Without this flag the script is a dry-run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be written without touching any files (default).",
    )
    args = parser.parse_args()

    dry_run = args.target != "production"
    if dry_run:
        logger.info("DRY RUN — pass --target=production to write files")

    db_path = Path(args.db_path) if args.db_path else _DEFAULT_OUTCOME_DB
    prior_path = Path(args.prior_store) if args.prior_store else _DEFAULT_PRIOR_STORE
    cal_path = Path(args.cal_store) if args.cal_store else _DEFAULT_CAL_STORE

    logger.info("Outcome DB  : %s", db_path)
    logger.info("Prior store : %s", prior_path)
    logger.info("Cal store   : %s", cal_path)

    try:
        trades = _fetch_trades(db_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Fetched %d finalized trades from outcome_trades", len(trades))
    if not trades:
        logger.warning("No finalized trades found — nothing to backfill.")
        return 0

    _rebuild_prior_store(trades, prior_path, dry_run=dry_run)
    _rebuild_calibration_store(trades, cal_path, dry_run=dry_run)

    if dry_run:
        print("\nDRY RUN complete.  Re-run with --target=production to apply.")
    else:
        logger.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
