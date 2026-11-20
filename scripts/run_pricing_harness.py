#!/usr/bin/env python3
"""Run the walk-forward pricing harness on the historical archive.

Replays every earnings event in the archive point in time and compares the
selector's predictions with the conservative-fill returns the structures went
on to earn (services/pricing_harness.py). Read-only: it writes only its own
report directory.

  .venv_arm64/bin/python3 scripts/run_pricing_harness.py --symbols AAPL,MSFT --start 2023-01-01
  .venv_arm64/bin/python3 scripts/run_pricing_harness.py --prior-mode neutral

Outputs, under --out-dir/<UTC timestamp>/: summary.json (the report),
rows.csv (one row per event, entry session and structure), variance_rows.csv
(per entry: the event variance at entry and exit and its forecast, and the
ordinary daily variance of the hold against each of its forecasts)
and report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.pricing_harness import (  # noqa: E402
    PRIOR_MODES,
    PRIOR_WALK_FORWARD,
    ParquetArchive,
    run_harness,
    summarize,
)
from services.daily_vol import DailyVarianceCalibrator  # noqa: E402
from services.event_variance import EventVarianceForecaster  # noqa: E402
from services.structure_legs import STRUCTURES  # noqa: E402

DEFAULT_DATA_ROOT = Path(os.environ.get("MARKET_DATA_ROOT") or "/Volumes/T9/market_data")
DEFAULT_OUT_DIR = REPO_ROOT / "exports" / "reports" / "pricing_harness"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--options-root", type=Path, default=None,
                        help="default: <data-root>/research/options_features_eod")
    parser.add_argument("--ohlc-root", type=Path, default=None,
                        help="default: <data-root>/research/underlyings/daily_features")
    parser.add_argument("--earnings-db", type=Path, default=None,
                        help="default: <data-root>/research/options_calculator_pro/earnings_calendar/"
                             "historical_earnings.sqlite")
    parser.add_argument("--symbols", type=str, default=None,
                        help="comma-separated; default: every symbol in the options archive")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="default: today")
    parser.add_argument("--prior-mode", choices=PRIOR_MODES, default=PRIOR_WALK_FORWARD)
    parser.add_argument("--require-lookahead-safe", action="store_true",
                        help="only entries on which the event date was verifiably known")
    parser.add_argument("--group-by", choices=("sector", "none"), default="sector",
                        help="pool the event-variance forecast by sector before the symbol "
                             "(scripts/stress_test_expansion.SECTOR_MAP; unmapped symbols get no group)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance only
        return None


def _archive_symbols(options_root: Path) -> List[str]:
    prefix = "underlying_symbol="
    if not options_root.exists():
        return []
    return sorted(p.name[len(prefix):].upper() for p in options_root.iterdir()
                  if p.is_dir() and p.name.startswith(prefix))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(summary: Dict[str, Any]) -> str:
    cfg = summary["config"]
    lines = [
        "# Pricing harness report",
        "",
        f"- Harness: `{cfg['harness_version']}`, code `{cfg.get('git_sha') or 'unknown'}`",
        f"- Events dated {cfg['start']} to {cfg['end']}, {len(cfg['symbols'])} symbols, "
        f"priors `{cfg['prior_mode']}`",
        f"- Entry window {cfg['entry_window_days'][0]}-{cfg['entry_window_days'][1]} days before the event; "
        f"exit: {cfg['exit_rule']}; judged on `{cfg['judged_return']}`",
        f"- First-look rows with the event date known beforehand: "
        f"{summary['event_dates']['known_beforehand_rows']} of {summary['event_dates']['first_look_rows']}",
        "",
        "## Prediction quality (first look, one row per event)",
        "",
        "| Structure | n | priced share | mean predicted edge | mean realized | bias | Spearman (edge) | "
        "Spearman (composite) | sign hit rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for structure in STRUCTURES:
        block = summary["per_structure"][structure]
        edge = block["first_look"]["expected_edge_pct"]
        comp = block["first_look"]["composite_structure_score"]
        lines.append(
            f"| {structure} | {edge.get('n', 0)} | {_fmt(block['priced_share'])} | "
            f"{_fmt(edge.get('mean_predicted'))} | {_fmt(edge.get('mean_realized'))} | {_fmt(edge.get('bias'))} | "
            f"{_fmt(edge.get('spearman'))} | {_fmt(comp.get('spearman'))} | {_fmt(edge.get('sign_hit_rate'))} |"
        )
    pol = summary["policies"]
    lines += [
        "",
        "## Policies (one trade per event)",
        "",
        "| Policy | events | mean return | median | win rate | 95% CI of mean |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("selector", "straddle_same_entry", "selector_minus_straddle_same_entry",
                 "straddle_first_look", "rank_pick_first_look"):
        block = pol[name]
        lines.append(f"| {name} | {block.get('n_events', 0)} | {_fmt(block.get('mean_return'))} | "
                     f"{_fmt(block.get('median_return'))} | {_fmt(block.get('win_rate'))} | "
                     f"{_fmt(block.get('mean_ci95'))} |")
    lines += ["", f"Selector trades without an exit price: {pol['selector'].get('traded_but_unpriced', 0)}"]
    ev = summary.get("event_variance") or {}
    lines += ["", "## Event variance: forecast of log(E at exit / E at entry)", "",
              f"Measured on {ev.get('measured_entries', 0)} of {ev.get('entries', 0)} entries "
              "(two expiries spanning the event, same pair at entry and exit).", "",
              "| Sample | n | events | mean realized | RMSE | RMSE no change | RMSE pooled | "
              "R2 vs no change | R2 vs pooled | Spearman |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("first_look", "all_entries"):
        q = ev.get(name) or {}
        lines.append(f"| {name} | {q.get('n', 0)} | {_fmt(q.get('events'))} | {_fmt(q.get('mean_realized_log_ratio'))} | "
                     f"{_fmt(q.get('rmse'))} | {_fmt(q.get('rmse_no_change'))} | {_fmt(q.get('rmse_pooled'))} | "
                     f"{_fmt(q.get('r2_vs_no_change'))} | {_fmt(q.get('r2_vs_pooled'))} | {_fmt(q.get('spearman'))} |")
    dv = (summary.get("daily_variance") or {}).get("all_entries") or {}
    lines += ["", "## Ordinary daily variance during the hold (all entries)", "",
              f"Rows with every forecast and a realized hold variance: {dv.get('n', 0)} "
              f"(realized measured on {dv.get('rows_with_realized', 0)}); realized vol "
              f"{_fmt(dv.get('mean_realized_vol_annual'))} annualised. Skill is 1 - loss / loss(implied): "
              "above 0 beats the market's base variance.", "",
              "| Forecast | mean vol | level ratio | QLIKE | QLIKE skill | MSE skill | spread Spearman |",
              "|---|---|---|---|---|---|---|"]
    for name, block in (dv.get("forecasts") or {}).items():
        lines.append(f"| {name} | {_fmt(block.get('mean_forecast_vol_annual'))} | {_fmt(block.get('level_ratio'))} | "
                     f"{_fmt(block.get('qlike'))} | {_fmt(block.get('qlike_skill_vs_implied'))} | "
                     f"{_fmt(block.get('mse_skill_vs_implied'))} | {_fmt(block.get('spread_spearman'))} |")
    lines += ["", "## Counts", ""]
    lines += [f"- `{key}`: {value}" for key, value in summary["counts"].items()]
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in summary["caveats"]]
    return "\n".join(lines) + "\n"


def write_outputs(result: Any, summary: Dict[str, Any], out_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False, default=str))
    fields = list(result.rows[0].keys()) if result.rows else ["symbol"]
    with (run_dir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result.rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})
    variance_fields = list(result.variance_rows[0].keys()) if result.variance_rows else ["symbol"]
    with (run_dir / "variance_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=variance_fields)
        writer.writeheader()
        for row in result.variance_rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})
    (run_dir / "report.md").write_text(render_markdown(summary))
    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    options_root = args.options_root or args.data_root / "research" / "options_features_eod"
    ohlc_root = args.ohlc_root or args.data_root / "research" / "underlyings" / "daily_features"
    earnings_db = args.earnings_db or (args.data_root / "research" / "options_calculator_pro"
                                       / "earnings_calendar" / "historical_earnings.sqlite")
    missing = [str(p) for p in (options_root, ohlc_root, earnings_db) if not p.exists()]
    if missing:
        print("Archive input not found: " + ", ".join(missing), file=sys.stderr)
        return 2
    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else _archive_symbols(options_root))
    if not symbols:
        print("No symbols to replay.", file=sys.stderr)
        return 2
    end = args.end or date.today()
    archive = ParquetArchive(options_root, ohlc_root, earnings_db)
    group_of = None
    if args.group_by == "sector":
        from scripts.stress_test_expansion import SECTOR_MAP

        group_of = SECTOR_MAP.get
    result = run_harness(archive, symbols, start=args.start, end=end, prior_mode=args.prior_mode,
                         require_lookahead_safe=args.require_lookahead_safe,
                         variance_forecaster=EventVarianceForecaster(group_of=group_of),
                         daily_calibrator=DailyVarianceCalibrator(group_of=group_of))
    summary = summarize(result)
    summary["config"]["git_sha"] = _git_sha()
    run_dir = write_outputs(result, summary, args.out_dir)
    print(f"Pricing harness: {result.counts.get('entries_scored', 0)} entries scored, "
          f"{result.counts.get('events_replayed', 0)} events. Report: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
