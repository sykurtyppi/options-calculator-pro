#!/usr/bin/env python3
"""
Audit the iVolatility raw-to-Parquet warehouse.

Produces a compact JSON report for a symbol covering:
- raw vs normalized storage footprint
- compression ratio
- option/underlying partition counts
- row counts from Parquet
- catalog-style date coverage and monthly row counts
- core data quality checks
- compaction manifest summary
- skipped/unsupported/error metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import duckdb

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")
except Exception:
    pass

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _resolve_data_root(explicit_root: Optional[str]) -> Path:
    configured = explicit_root or os.environ.get("MARKET_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("/Volumes/T9/market_data")


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _find_latest_manifest(manifest_root: Path, run_label: str) -> Optional[Path]:
    matches = sorted(manifest_root.rglob(f"{run_label}_*.json"))
    if not matches:
        return None
    return matches[-1]


def _load_manifest(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _discover_symbols(normalized_options_root: Path) -> list[str]:
    if not normalized_options_root.exists():
        return []

    symbols: list[str] = []
    for path in sorted(normalized_options_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        prefix = "underlying_symbol="
        if name.startswith(prefix):
            symbol = name[len(prefix) :].strip().upper()
            if symbol:
                symbols.append(symbol)
    return symbols


def _count_parquet_rows(base_dir: Path) -> int:
    parquet_glob = str(base_dir / "**" / "*.parquet")
    if not base_dir.exists():
        return 0
    with duckdb.connect(database=":memory:") as con:
        return int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [parquet_glob]).fetchone()[0])


def _fetch_single_row(query: str, parquet_glob: str) -> tuple[Any, ...]:
    with duckdb.connect(database=":memory:") as con:
        result = con.execute(query, [parquet_glob]).fetchone()
    return result or tuple()


def _fetch_df_rows(query: str, parquet_glob: str) -> list[dict[str, Any]]:
    with duckdb.connect(database=":memory:") as con:
        df = con.execute(query, [parquet_glob]).fetch_df()
    return df.to_dict(orient="records")


def _count_partitions(base_dir: Path) -> int:
    if not base_dir.exists():
        return 0
    return len(list(base_dir.rglob("*.parquet")))


def _sample_partition_paths(base_dir: Path, limit: int = 5) -> list[str]:
    if not base_dir.exists():
        return []
    return [str(path) for path in sorted(base_dir.rglob("*.parquet"))[:limit]]


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _quality_grade(
    duplicate_rows: int,
    bid_ask_inversions: int,
    zero_bid_positive_ask: int,
    null_critical_fields: int,
) -> str:
    if duplicate_rows == 0 and bid_ask_inversions == 0 and zero_bid_positive_ask == 0 and null_critical_fields == 0:
        return "excellent"
    if duplicate_rows < 100 and bid_ask_inversions < 100 and null_critical_fields == 0:
        return "good"
    if duplicate_rows < 1000 and bid_ask_inversions < 1000:
        return "needs_review"
    return "high_risk"


def _merge_date_ranges(ranges: list[dict[str, Optional[str]]]) -> dict[str, Optional[str]]:
    mins = sorted(value for entry in ranges for value in [entry.get("min")] if value)
    maxes = sorted(value for entry in ranges for value in [entry.get("max")] if value)
    return {
        "min": mins[0] if mins else None,
        "max": maxes[-1] if maxes else None,
    }


def _symbol_report(
    symbol: str,
    data_root: Path,
    manifest_root: Path,
    run_label: str,
) -> dict[str, Any]:
    raw_options_root = data_root / "raw" / "ivolatility" / "options_chains"
    raw_reference_root = data_root / "raw" / "ivolatility" / "reference"
    normalized_options_root = data_root / "normalized" / "options" / "chains_eod" / f"underlying_symbol={symbol}"
    normalized_underlying_root = (
        data_root / "normalized" / "underlyings" / "daily_ohlcv" / f"underlying_symbol={symbol}"
    )

    manifest_path = _find_latest_manifest(manifest_root, run_label)
    manifest = _load_manifest(manifest_path)

    raw_option_files = [path for path in raw_options_root.rglob("*.json") if f"_{symbol.lower()}_" in path.name.lower()]
    raw_underlying_files = [
        path for path in raw_reference_root.rglob("*.json") if f"stock_prices_{symbol.lower()}" in path.name.lower()
    ]

    raw_options_size = sum(path.stat().st_size for path in raw_option_files if path.exists())
    raw_underlying_size = sum(path.stat().st_size for path in raw_underlying_files if path.exists())
    normalized_options_size = _dir_size_bytes(normalized_options_root)
    normalized_underlying_size = _dir_size_bytes(normalized_underlying_root)

    option_row_count = _count_parquet_rows(normalized_options_root)
    underlying_row_count = _count_parquet_rows(normalized_underlying_root)
    options_glob = str(normalized_options_root / "**" / "*.parquet")
    underlyings_glob = str(normalized_underlying_root / "**" / "*.parquet")

    total_raw = raw_options_size + raw_underlying_size
    total_normalized = normalized_options_size + normalized_underlying_size

    option_date_bounds = (
        _fetch_single_row(
            """
            SELECT
                CAST(MIN(trade_date) AS VARCHAR),
                CAST(MAX(trade_date) AS VARCHAR),
                CAST(MIN(expiry) AS VARCHAR),
                CAST(MAX(expiry) AS VARCHAR)
            FROM read_parquet(?)
            """,
            options_glob,
        )
        if normalized_options_root.exists()
        else tuple()
    )

    underlying_date_bounds = (
        _fetch_single_row(
            """
            SELECT
                CAST(MIN(trade_date) AS VARCHAR),
                CAST(MAX(trade_date) AS VARCHAR)
            FROM read_parquet(?)
            """,
            underlyings_glob,
        )
        if normalized_underlying_root.exists()
        else tuple()
    )

    monthly_option_rows = (
        _fetch_df_rows(
            """
            SELECT
                year(trade_date) AS year,
                month(trade_date) AS month,
                COUNT(*) AS row_count,
                COUNT(DISTINCT trade_date) AS distinct_trade_dates
            FROM read_parquet(?)
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            options_glob,
        )
        if normalized_options_root.exists()
        else []
    )

    duplicate_rows = (
        int(
            _fetch_single_row(
                """
                SELECT COALESCE(SUM(cnt - 1), 0)
                FROM (
                    SELECT trade_date, option_id, COUNT(*) AS cnt
                    FROM read_parquet(?)
                    GROUP BY 1, 2
                    HAVING COUNT(*) > 1
                )
                """,
                options_glob,
            )[0]
        )
        if normalized_options_root.exists()
        else 0
    )

    bid_ask_inversions = (
        int(
            _fetch_single_row(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                WHERE bid_10000 IS NOT NULL
                  AND ask_10000 IS NOT NULL
                  AND bid_10000 > ask_10000
                """,
                options_glob,
            )[0]
        )
        if normalized_options_root.exists()
        else 0
    )

    zero_bid_positive_ask = (
        int(
            _fetch_single_row(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                WHERE bid_10000 = 0
                  AND ask_10000 > 0
                """,
                options_glob,
            )[0]
        )
        if normalized_options_root.exists()
        else 0
    )

    null_critical_fields = (
        int(
            _fetch_single_row(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                WHERE trade_date IS NULL
                   OR option_symbol IS NULL
                   OR option_id IS NULL
                   OR bid_10000 IS NULL
                   OR ask_10000 IS NULL
                   OR iv_1000000 IS NULL
                """,
                options_glob,
            )[0]
        )
        if normalized_options_root.exists()
        else 0
    )

    densest_option_months = sorted(monthly_option_rows, key=lambda row: row["row_count"], reverse=True)[:5]
    sparsest_option_months = sorted(monthly_option_rows, key=lambda row: row["row_count"])[:5]
    quality_grade = _quality_grade(
        duplicate_rows=duplicate_rows,
        bid_ask_inversions=bid_ask_inversions,
        zero_bid_positive_ask=zero_bid_positive_ask,
        null_critical_fields=null_critical_fields,
    )

    return {
        "symbol": symbol,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "raw": {
            "option_files": len(raw_option_files),
            "underlying_files": len(raw_underlying_files),
            "options_bytes": raw_options_size,
            "underlyings_bytes": raw_underlying_size,
            "total_bytes": total_raw,
        },
        "normalized": {
            "option_partitions": _count_partitions(normalized_options_root),
            "underlying_partitions": _count_partitions(normalized_underlying_root),
            "options_bytes": normalized_options_size,
            "underlyings_bytes": normalized_underlying_size,
            "total_bytes": total_normalized,
            "option_rows": option_row_count,
            "underlying_rows": underlying_row_count,
            "sample_option_partitions": _sample_partition_paths(normalized_options_root),
            "sample_underlying_partitions": _sample_partition_paths(normalized_underlying_root),
        },
        "catalog": {
            "option_trade_date_range": {
                "min": option_date_bounds[0] if len(option_date_bounds) >= 1 else None,
                "max": option_date_bounds[1] if len(option_date_bounds) >= 2 else None,
            },
            "option_expiry_range": {
                "min": option_date_bounds[2] if len(option_date_bounds) >= 3 else None,
                "max": option_date_bounds[3] if len(option_date_bounds) >= 4 else None,
            },
            "underlying_trade_date_range": {
                "min": underlying_date_bounds[0] if len(underlying_date_bounds) >= 1 else None,
                "max": underlying_date_bounds[1] if len(underlying_date_bounds) >= 2 else None,
            },
            "monthly_option_rows": monthly_option_rows,
            "densest_option_months": densest_option_months,
            "sparsest_option_months": sparsest_option_months,
        },
        "compression": {
            "normalized_vs_raw_ratio": _safe_ratio(total_normalized, total_raw),
            "space_saved_ratio": None if total_raw <= 0 else 1.0 - (float(total_normalized) / float(total_raw)),
        },
        "data_quality": {
            "grade": quality_grade,
            "duplicate_rows": duplicate_rows,
            "bid_ask_inversions": bid_ask_inversions,
            "zero_bid_positive_ask": zero_bid_positive_ask,
            "null_critical_fields": null_critical_fields,
        },
        "compaction_manifest_summary": {
            "option_files_found": manifest.get("option_files_found"),
            "option_files_materialized_via_details": manifest.get("option_files_materialized_via_details"),
            "option_files_skipped": manifest.get("option_files_skipped"),
            "option_files_unsupported_schema": manifest.get("option_files_unsupported_schema"),
            "option_errors_count": len(manifest.get("option_errors", []) or []),
        },
    }


def _aggregate_reports(symbol_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not symbol_reports:
        return {
            "symbol_count": 0,
            "symbols": [],
            "raw": {},
            "normalized": {},
            "compression": {},
            "catalog": {},
            "data_quality": {},
        }

    total_raw_bytes = sum(report["raw"]["total_bytes"] for report in symbol_reports)
    total_normalized_bytes = sum(report["normalized"]["total_bytes"] for report in symbol_reports)
    grade_counts: dict[str, int] = {}
    for report in symbol_reports:
        grade = report["data_quality"]["grade"]
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    largest_by_rows = sorted(
        (
            {
                "symbol": report["symbol"],
                "option_rows": report["normalized"]["option_rows"],
                "normalized_bytes": report["normalized"]["total_bytes"],
            }
            for report in symbol_reports
        ),
        key=lambda row: row["option_rows"],
        reverse=True,
    )[:10]

    return {
        "symbol_count": len(symbol_reports),
        "symbols": [report["symbol"] for report in symbol_reports],
        "raw": {
            "option_files": sum(report["raw"]["option_files"] for report in symbol_reports),
            "underlying_files": sum(report["raw"]["underlying_files"] for report in symbol_reports),
            "options_bytes": sum(report["raw"]["options_bytes"] for report in symbol_reports),
            "underlyings_bytes": sum(report["raw"]["underlyings_bytes"] for report in symbol_reports),
            "total_bytes": total_raw_bytes,
        },
        "normalized": {
            "option_partitions": sum(report["normalized"]["option_partitions"] for report in symbol_reports),
            "underlying_partitions": sum(report["normalized"]["underlying_partitions"] for report in symbol_reports),
            "options_bytes": sum(report["normalized"]["options_bytes"] for report in symbol_reports),
            "underlyings_bytes": sum(report["normalized"]["underlyings_bytes"] for report in symbol_reports),
            "total_bytes": total_normalized_bytes,
            "option_rows": sum(report["normalized"]["option_rows"] for report in symbol_reports),
            "underlying_rows": sum(report["normalized"]["underlying_rows"] for report in symbol_reports),
            "largest_symbols_by_option_rows": largest_by_rows,
        },
        "compression": {
            "normalized_vs_raw_ratio": _safe_ratio(total_normalized_bytes, total_raw_bytes),
            "space_saved_ratio": None
            if total_raw_bytes <= 0
            else 1.0 - (float(total_normalized_bytes) / float(total_raw_bytes)),
        },
        "catalog": {
            "option_trade_date_range": _merge_date_ranges(
                [report["catalog"]["option_trade_date_range"] for report in symbol_reports]
            ),
            "option_expiry_range": _merge_date_ranges([report["catalog"]["option_expiry_range"] for report in symbol_reports]),
            "underlying_trade_date_range": _merge_date_ranges(
                [report["catalog"]["underlying_trade_date_range"] for report in symbol_reports]
            ),
        },
        "data_quality": {
            "grade_counts": grade_counts,
            "duplicate_rows": sum(report["data_quality"]["duplicate_rows"] for report in symbol_reports),
            "bid_ask_inversions": sum(report["data_quality"]["bid_ask_inversions"] for report in symbol_reports),
            "zero_bid_positive_ask": sum(report["data_quality"]["zero_bid_positive_ask"] for report in symbol_reports),
            "null_critical_fields": sum(report["data_quality"]["null_critical_fields"] for report in symbol_reports),
            "symbols_requiring_review": [
                report["symbol"]
                for report in symbol_reports
                if report["data_quality"]["grade"] in {"needs_review", "high_risk"}
            ],
        },
    }


def _resolve_symbols(args: argparse.Namespace, data_root: Path) -> list[str]:
    requested: list[str] = []
    if args.symbol:
        requested.extend([args.symbol])
    if args.symbols:
        requested.extend(args.symbols.split(","))

    cleaned = sorted({symbol.strip().upper() for symbol in requested if symbol.strip()})
    if cleaned:
        return cleaned

    return _discover_symbols(data_root / "normalized" / "options" / "chains_eod")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit iVolatility Parquet store for one or more symbols.")
    parser.add_argument("--symbol", default=None, help="Single symbol to audit, e.g. SPY")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol list for multi-symbol catalog runs")
    parser.add_argument("--data-root", default=None, help="Override MARKET_DATA_ROOT")
    parser.add_argument(
        "--compaction-run-label",
        default=None,
        help="Compaction run label to resolve the latest manifest. Defaults to <symbol>_5y_parquet_compaction lowercased.",
    )
    parser.add_argument("--output-json", default=None, help="Optional explicit output path for the audit report JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = _resolve_data_root(args.data_root)
    manifest_root = data_root / "manifests" / "coverage"
    symbols = _resolve_symbols(args, data_root)
    if not symbols:
        raise SystemExit("No symbols were provided or discovered in the normalized warehouse.")

    symbol_reports = []
    for symbol in symbols:
        run_label = args.compaction_run_label or f"{symbol.lower()}_5y_parquet_compaction"
        symbol_reports.append(_symbol_report(symbol=symbol, data_root=data_root, manifest_root=manifest_root, run_label=run_label))

    created_at = datetime.now(UTC).isoformat()
    if len(symbol_reports) == 1:
        report = {
            **symbol_reports[0],
            "created_at": created_at,
            "data_root": str(data_root),
        }
    else:
        report = {
            "mode": "multi_symbol_catalog",
            "created_at": created_at,
            "data_root": str(data_root),
            "overall": _aggregate_reports(symbol_reports),
            "symbols_detail": symbol_reports,
        }

    output_path = None
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
    else:
        output_dir = data_root / "manifests" / "coverage" / f"{datetime.now(UTC):%Y}" / f"{datetime.now(UTC):%m}"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        if len(symbol_reports) == 1:
            output_path = output_dir / f"{symbol_reports[0]['symbol'].lower()}_warehouse_audit_{timestamp}.json"
        else:
            output_path = output_dir / f"warehouse_catalog_{len(symbol_reports)}symbols_{timestamp}.json"

    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote audit report to %s", output_path)
    if len(symbol_reports) == 1:
        total_raw = report["raw"]["total_bytes"]
        total_normalized = report["normalized"]["total_bytes"]
        logger.info(
            "Audit summary %s: raw=%.2f MB normalized=%.2f MB option_partitions=%d option_rows=%d",
            symbol_reports[0]["symbol"],
            total_raw / (1024 * 1024),
            total_normalized / (1024 * 1024),
            report["normalized"]["option_partitions"],
            report["normalized"]["option_rows"],
        )
    else:
        logger.info(
            "Catalog summary: symbols=%d raw=%.2f MB normalized=%.2f MB option_rows=%d",
            report["overall"]["symbol_count"],
            report["overall"]["raw"]["total_bytes"] / (1024 * 1024),
            report["overall"]["normalized"]["total_bytes"] / (1024 * 1024),
            report["overall"]["normalized"]["option_rows"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
