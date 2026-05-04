#!/usr/bin/env python3
"""
Replay failed requests from an iVolatility download manifest.

This is useful when a large backfill mostly succeeds but a handful of request
plans fail due to transient network timeouts. The retry uses the exact request
parameters captured in the original manifest instead of broad symbol/date
ranges, which keeps follow-up runs small and deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.ivol_options_backfill import (  # noqa: E402
    IVolatilityBackfillRunner,
    RequestPlan,
    _require_api_key,
    _resolve_data_root,
)
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _matches_filters(entry: dict[str, Any], symbols: Optional[set[str]], statuses: set[str]) -> bool:
    if entry.get("status") not in statuses:
        return False
    if symbols and str(entry.get("symbol", "")).upper() not in symbols:
        return False
    return True


def _plan_from_manifest_entry(entry: dict[str, Any]) -> RequestPlan:
    return RequestPlan(
        symbol=str(entry["symbol"]).upper(),
        trade_date=str(entry["trade_date"]),
        endpoint=str(entry["endpoint"]),
        params=dict(entry["params"]),
        output_path=Path(str(entry["output_path"])),
        request_kind=str(entry["request_kind"]),
        expected_market_open=bool(entry.get("expected_market_open", True)),
    )


def load_retry_plans(
    manifest_path: Path,
    *,
    symbols: Optional[set[str]] = None,
    statuses: Optional[set[str]] = None,
) -> list[RequestPlan]:
    manifest = _load_manifest(manifest_path)
    selected_statuses = statuses or {"failed"}
    plans: list[RequestPlan] = []
    for entry in manifest.get("results", []):
        if _matches_filters(entry, symbols=symbols, statuses=selected_statuses):
            plans.append(_plan_from_manifest_entry(entry))
    return plans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retry failed iVolatility requests from an existing manifest.")
    parser.add_argument("--manifest-path", required=True, help="Path to the original download manifest JSON")
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbol filter")
    parser.add_argument(
        "--statuses",
        default="failed",
        help="Comma-separated manifest statuses to replay. Default: failed",
    )
    parser.add_argument("--api-key", default=None, help="Optional iVolatility API key override")
    parser.add_argument("--data-root", default=None, help="Optional market data root override")
    parser.add_argument("--run-label", default="ivol_manifest_retry")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip retry plans whose output file already exists. Default is to overwrite/retry.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    symbols = (
        {item.strip().upper() for item in args.symbols.split(",") if item.strip()}
        if args.symbols
        else None
    )

    plans = load_retry_plans(manifest_path, symbols=symbols, statuses=statuses)
    if not plans:
        logger.info("No matching manifest entries found in %s", manifest_path)
        return 0

    api_key = _require_api_key(args.api_key)
    data_root = _resolve_data_root(args.data_root)
    runner = IVolatilityBackfillRunner(
        api_key=api_key,
        data_root=data_root,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        skip_existing=args.skip_existing,
    )

    logger.info(
        "Retrying %d manifest requests from %s",
        len(plans),
        manifest_path,
    )
    manifest = runner.execute(plans=plans, run_label=args.run_label, dry_run=args.dry_run)
    logger.info("Retry run complete: %s", json.dumps(manifest["counts"], sort_keys=True))
    return 0 if manifest["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
