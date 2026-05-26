#!/usr/bin/env python3
"""
train_from_t9.py
================
End-to-end orchestrator for the empirical-learning pipeline. Walks the
T9 historical option chains through three stages and lands the result
in calibration + structure priors so the selector stops running on its
research-prior bootstrap.

Stages
------
1. **Earnings calendar extension** — runs
   ``scripts/integrate_historical_replay_inputs.py --fetch-earnings`` so
   ``historical_earnings.sqlite`` covers every symbol with T9 option-chain
   data, not just the 4-symbol default set.
2. **Replay backtest** — runs ``scripts/run_replay_backtest.py`` over
   the chosen scale window, populating ``backtest_trades`` in the
   institutional ML DB with realistic pre/post-earnings trade economics.
3. **Seed outcomes** — runs ``scripts/seed_outcomes_from_replay.py``
   to convert those backtest trades into observations that flow into
   the live calibration store + structure priors (or an isolated tmp
   directory when ``--target=tmp``, the default).

This script is **wiring**, not new analytics. Every stage shells out to
an existing entry point. The orchestrator adds:
- A canonical symbol set: ``INSTITUTIONAL_UNIVERSE ∩ T9 chains_eod`` (181 symbols today).
- Scale presets (smoke / small / full) so re-runs at different sizes are one flag.
- Per-stage logs into ``exports/reports/train_from_t9/<utc>/`` for debuggable failures.
- A safety default of ``--target=tmp`` so a smoke run never touches live state.
- Pre-flight checks (T9 mounted, DB path writable, stage scripts exist).

Idempotency
-----------
Each underlying stage is already idempotent (INSERT OR IGNORE on outcomes,
stable observation IDs on calibration, dedup on prior store). Re-running
the orchestrator with the same scale + target should be a no-op once the
pipeline has converged for that snapshot of T9 data.

Usage examples
--------------
::

    # 1. Inspect what would happen — no commands actually run:
    python scripts/train_from_t9.py --dry-run

    # 2. Smoke run (~30 minutes): top-10 symbols, 1yr, writes to tmp/
    python scripts/train_from_t9.py --scale smoke

    # 3. Small run (~2 hours): top-25, 2yr, writes to tmp/
    python scripts/train_from_t9.py --scale small

    # 4. Full run (~6+ hours): 181 symbols, 3yr, COMMITS to production
    python scripts/train_from_t9.py --scale full --target production

    # 5. Skip earnings re-fetch if you've already extended the calendar
    python scripts/train_from_t9.py --scale full --skip-earnings-fetch
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REPORT_DIR = REPO_ROOT / "exports" / "reports" / "train_from_t9"
DEFAULT_DB_PATH = pathlib.Path.home() / ".options_calculator_pro" / "institutional_ml.db"
T9_CHAINS_ROOT = pathlib.Path("/Volumes/T9/market_data/normalized/options/chains_eod")
DEFAULT_EARNINGS_DB = pathlib.Path(
    "/Volumes/T9/market_data/research/options_calculator_pro/earnings_calendar/historical_earnings.sqlite"
)

# Scale presets — each is (symbol_limit, years, runtime_label, description)
SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "limit_symbols": 10,
        "years": 1,
        "runtime": "~30 min",
        "label": "top-10 / 1yr",
    },
    "small": {
        "limit_symbols": 25,
        "years": 2,
        "runtime": "~2 hr",
        "label": "top-25 / 2yr",
    },
    "full": {
        "limit_symbols": None,  # full intersection
        "years": 3,
        "runtime": "~6 hr",
        "label": "181 symbols / 3yr",
    },
}


def _t9_chain_symbols() -> set[str]:
    """Discover which symbols actually have T9 chain data."""
    if not T9_CHAINS_ROOT.exists():
        return set()
    return {
        p.name.replace("underlying_symbol=", "")
        for p in T9_CHAINS_ROOT.iterdir()
        if p.name.startswith("underlying_symbol=")
    }


def _resolve_symbols(limit: Optional[int]) -> List[str]:
    """Return symbols that exist in BOTH the institutional universe AND T9.

    Falls back to the T9-only set if the institutional universe is
    unavailable for some reason (defensive — should never happen since
    services.institutional_ml_db is in-tree).
    """
    t9 = _t9_chain_symbols()
    try:
        from services.institutional_ml_db import INSTITUTIONAL_UNIVERSE

        usable = [s for s in INSTITUTIONAL_UNIVERSE if s in t9]
    except Exception as exc:
        print(
            f"WARNING: could not import INSTITUTIONAL_UNIVERSE ({exc}); "
            "falling back to T9 chain list as-is."
        )
        usable = sorted(t9)
    return usable if limit is None else usable[:limit]


def _stage_banner(stage_num: int, description: str) -> None:
    print()
    print("=" * 72)
    print(f"  STAGE {stage_num}: {description}")
    print("=" * 72)


def _run_subprocess(cmd: List[str], log_path: pathlib.Path) -> Tuple[int, float]:
    """Run a subprocess, tee stdout/stderr to a log file, return (exit, duration).

    Subprocess output also streams to the orchestrator's stdout so the
    user sees progress in real time. The log file captures everything
    for post-mortem.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(cmd)}")
    print(f"  Log: {log_path}")

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"# Command: {' '.join(cmd)}\n")
        log_f.write(f"# Started: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        log_f.write("# --- output follows ---\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        proc.wait()
        log_f.write(
            f"# --- end output ---\n"
            f"# Ended: {datetime.datetime.now(datetime.timezone.utc).isoformat()} "
            f"(exit {proc.returncode})\n"
        )
    duration = time.time() - start
    return proc.returncode, duration


def _preflight_checks(args: argparse.Namespace, symbols: List[str]) -> List[str]:
    """Return a list of human-readable failures. Empty list = OK to proceed."""
    failures: List[str] = []

    if not T9_CHAINS_ROOT.exists():
        failures.append(
            f"T9 chains root not found at {T9_CHAINS_ROOT} — is the drive mounted?"
        )
    elif not list(T9_CHAINS_ROOT.iterdir()):
        failures.append(f"T9 chains root is empty: {T9_CHAINS_ROOT}")

    if not symbols:
        failures.append(
            "Resolved symbol set is empty — neither T9 chains nor "
            "INSTITUTIONAL_UNIVERSE produced a usable list."
        )

    for stage_script in (
        "scripts/integrate_historical_replay_inputs.py",
        "scripts/run_replay_backtest.py",
        "scripts/seed_outcomes_from_replay.py",
    ):
        if not (REPO_ROOT / stage_script).exists():
            failures.append(f"Missing stage script: {stage_script}")

    db_path = pathlib.Path(args.db_path)
    if not db_path.parent.exists():
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            failures.append(f"Cannot create DB parent dir {db_path.parent}: {exc}")

    if args.target == "production" and args.scale == "smoke":
        failures.append(
            "Refusing to write smoke-scale results into production stores — "
            "use --scale {small,full} when --target=production, or drop --target."
        )

    return failures


def _print_run_plan(args: argparse.Namespace, preset: Dict[str, Any],
                    symbols: List[str], report_dir: pathlib.Path) -> None:
    print()
    print("RUN PLAN")
    print("-" * 72)
    print(f"  Scale:           {args.scale}   ({preset['label']}, est. {preset['runtime']})")
    print(f"  Symbols:         {len(symbols)} ({', '.join(symbols[:8])}{'...' if len(symbols) > 8 else ''})")
    print(f"  Date range:      {args.start_date} → {args.end_date}")
    print(f"  ML DB path:      {args.db_path}")
    print(f"  Earnings DB:     {args.earnings_db}")
    print(f"  Target:          {args.target}{'  ← WRITES TO LIVE STORES' if args.target == 'production' else ''}")
    print(f"  Max debit/contract: ${args.max_debit}")
    print(f"  Report dir:      {report_dir}")
    print(f"  Skip earnings:   {args.skip_earnings_fetch}")
    print(f"  Skip replay:     {args.skip_replay}")
    print(f"  Dry run:         {args.dry_run}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("Usage examples")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scale",
        choices=["smoke", "small", "full"],
        default="smoke",
        help="Pipeline scale (default: smoke = top-10 symbols, 1 year)",
    )
    parser.add_argument(
        "--target",
        choices=["tmp", "production"],
        default="tmp",
        help=(
            "Where stage 3 writes seeded observations. 'tmp' (default) is "
            "an isolated directory under tmp/seed_run_<utc>/. 'production' "
            "writes to the live stores under ~/.options_calculator_pro/."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and exit without invoking any stage.",
    )
    parser.add_argument(
        "--skip-earnings-fetch",
        action="store_true",
        help="Skip stage 1 (use the existing historical_earnings.sqlite as-is).",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip stage 2 (use existing backtest_trades in the ML DB).",
    )
    parser.add_argument(
        "--max-debit",
        type=int,
        default=600,
        help="Replay backtest filter: skip trades with debit > $N per contract.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Institutional ML DB path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--earnings-db",
        default=str(DEFAULT_EARNINGS_DB),
        help=f"Historical earnings DB path (default: T9 path)",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Replay window start (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.date.today().isoformat(),
        help="Replay window end (default: today)",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Where per-stage logs land (default: exports/reports/train_from_t9/)",
    )
    args = parser.parse_args(argv)

    preset = SCALE_PRESETS[args.scale]
    symbols = _resolve_symbols(preset["limit_symbols"])

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = pathlib.Path(args.report_dir) / timestamp

    _print_run_plan(args, preset, symbols, report_dir)

    failures = _preflight_checks(args, symbols)
    if failures:
        print("PRE-FLIGHT FAILURES")
        print("-" * 72)
        for f in failures:
            print(f"  • {f}")
        print()
        print("Aborting before any stage runs.")
        return 2

    if args.dry_run:
        print("DRY RUN — exiting before stage 1.")
        return 0

    if args.target == "production":
        print("⚠  --target=production: stage 3 will commit to the LIVE")
        print("   calibration + prior stores under ~/.options_calculator_pro/.")
        print("   Re-running with --target=tmp first is the safer pattern.")
        print()

    report_dir.mkdir(parents=True, exist_ok=True)
    symbols_arg = ",".join(symbols)
    stage_results: List[Tuple[str, int, float]] = []

    # ── Stage 1: earnings calendar extension ─────────────────────────────────
    if args.skip_earnings_fetch:
        print()
        print("Stage 1 SKIPPED (--skip-earnings-fetch). Using existing earnings DB.")
    else:
        _stage_banner(1, f"Earnings calendar extension for {len(symbols)} symbols")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "integrate_historical_replay_inputs.py"),
            "--symbols", symbols_arg,
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--fetch-earnings",
            "--earnings-db", args.earnings_db,
            "--report-dir", str(report_dir / "stage1_earnings"),
        ]
        code, duration = _run_subprocess(cmd, report_dir / "stage1_earnings.log")
        stage_results.append(("stage1_earnings", code, duration))
        if code != 0:
            print(f"  Stage 1 returned exit code {code}. Continuing to stage 2 "
                  "with whatever earnings rows already exist.")

    # ── Stage 2: replay backtest at scale ────────────────────────────────────
    if args.skip_replay:
        print()
        print("Stage 2 SKIPPED (--skip-replay). Using existing backtest_trades in ML DB.")
    else:
        _stage_banner(2, f"Replay backtest ({len(symbols)} symbols, {preset['years']}yr)")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_replay_backtest.py"),
            "--symbols", symbols_arg,
            "--years", str(preset["years"]),
            "--mode", "hybrid",
            "--max-debit", str(args.max_debit),
            "--db-path", args.db_path,
        ]
        code, duration = _run_subprocess(cmd, report_dir / "stage2_replay.log")
        stage_results.append(("stage2_replay", code, duration))
        if code != 0:
            print(f"  Stage 2 returned exit code {code}. Stage 3 may produce "
                  "less data than expected, or none.")

    # ── Stage 3: seed outcomes from replay ───────────────────────────────────
    _stage_banner(3, f"Seeding outcomes into {args.target} stores")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "seed_outcomes_from_replay.py"),
        "--db-path", args.db_path,
        "--target", args.target,
    ]
    code, duration = _run_subprocess(cmd, report_dir / "stage3_seed.log")
    stage_results.append(("stage3_seed", code, duration))

    # ── Final summary ────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  PIPELINE SUMMARY")
    print("=" * 72)
    overall_failure = False
    for stage, exit_code, duration in stage_results:
        status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
        if exit_code != 0:
            overall_failure = True
        mins, secs = divmod(int(duration), 60)
        print(f"  {stage:24s} {status:24s} {mins:3d}m {secs:02d}s")
    print()
    print(f"  Logs: {report_dir}")
    if overall_failure:
        print()
        print("  One or more stages failed. Inspect the per-stage logs before re-running.")
        return 1
    if args.target == "tmp":
        print()
        print("  Stage 3 wrote to tmp/. Inspect the temp directory and re-run with")
        print("  --target=production --skip-replay --skip-earnings-fetch to promote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
