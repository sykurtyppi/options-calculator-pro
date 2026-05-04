#!/usr/bin/env python3
"""
Build research-ready option features from the normalized iVolatility warehouse.

This script leaves the normalized fixed-point Parquet source-of-truth untouched
and writes decoded, analysis-friendly feature Parquet under:

    /Volumes/T9/market_data/research/options_features_eod

The output is still partitioned by symbol/year/month and compressed with ZSTD.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

PRICE_SCALE = 10_000.0
RISK_SCALE = 1_000_000.0
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,15}$")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_data_root(explicit_root: Optional[str]) -> Path:
    configured = explicit_root or os.environ.get("MARKET_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("/Volumes/T9/market_data")


def _month_partition(base_dir: Path) -> Path:
    now = datetime.now(UTC)
    return base_dir / f"{now:%Y}" / f"{now:%m}"


def _discover_symbols(normalized_options_root: Path) -> list[str]:
    if not normalized_options_root.exists():
        return []
    symbols: list[str] = []
    for path in sorted(normalized_options_root.iterdir()):
        prefix = "underlying_symbol="
        if path.is_dir() and path.name.startswith(prefix):
            symbol = path.name[len(prefix) :].upper()
            if _is_safe_symbol(symbol):
                symbols.append(symbol)
    return symbols


def _parse_symbols(symbols: Optional[str], normalized_options_root: Path) -> list[str]:
    if not symbols:
        discovered = _discover_symbols(normalized_options_root)
        if not discovered:
            raise ValueError(f"No symbols provided and none discovered under {normalized_options_root}")
        return discovered

    parsed = sorted({item.strip().upper() for item in symbols.split(",") if item.strip()})
    unsafe = [symbol for symbol in parsed if not _is_safe_symbol(symbol)]
    if unsafe:
        raise ValueError(f"Unsafe symbol values: {unsafe}")
    return parsed


def _is_safe_symbol(symbol: str) -> bool:
    return bool(SYMBOL_RE.match(symbol))


def _sql_date_filter(start_date: Optional[str], end_date: Optional[str]) -> str:
    clauses: list[str] = []
    if start_date:
        clauses.append(f"trade_date >= DATE '{date_literal(start_date)}'")
    if end_date:
        clauses.append(f"trade_date <= DATE '{date_literal(end_date)}'")
    if not clauses:
        return ""
    return " AND " + " AND ".join(clauses)


def date_literal(value: str) -> str:
    # Validate ISO date and return the original normalized value for SQL literal use.
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


def _dte_bucket_expr() -> str:
    return """
        CASE
            WHEN dte IS NULL THEN 'unknown'
            WHEN dte BETWEEN 0 AND 6 THEN 'dte_000_006'
            WHEN dte BETWEEN 7 AND 14 THEN 'dte_007_014'
            WHEN dte BETWEEN 15 AND 19 THEN 'dte_015_019'
            WHEN dte BETWEEN 20 AND 45 THEN 'dte_020_045'
            WHEN dte BETWEEN 46 AND 90 THEN 'dte_046_090'
            ELSE 'dte_091_plus'
        END
    """


def _feature_select_sql(source_glob: str, start_date: Optional[str], end_date: Optional[str]) -> str:
    date_filter = _sql_date_filter(start_date=start_date, end_date=end_date)
    return f"""
    WITH decoded AS (
        SELECT
            trade_date,
            expiry,
            underlying_symbol,
            option_symbol,
            option_id,
            stock_id,
            call_put,
            call_put = 'C' AS is_call,
            call_put = 'P' AS is_put,
            dte,
            {_dte_bucket_expr()} AS dte_bucket,
            volume,
            open_interest,
            is_settlement,
            strike_10000 / {PRICE_SCALE} AS strike,
            bid_10000 / {PRICE_SCALE} AS bid,
            ask_10000 / {PRICE_SCALE} AS ask,
            COALESCE(mid_10000 / {PRICE_SCALE}, (bid_10000 + ask_10000) / {PRICE_SCALE * 2.0}) AS mid,
            last_10000 / {PRICE_SCALE} AS last,
            price_open_10000 / {PRICE_SCALE} AS price_open,
            price_high_10000 / {PRICE_SCALE} AS price_high,
            price_low_10000 / {PRICE_SCALE} AS price_low,
            underlying_price_10000 / {PRICE_SCALE} AS underlying_price,
            moneyness_pct_10000 / {PRICE_SCALE} AS moneyness_pct,
            iv_1000000 / {RISK_SCALE} AS iv,
            preiv_1000000 / {RISK_SCALE} AS preiv,
            delta_1000000 / {RISK_SCALE} AS delta,
            gamma_1000000 / {RISK_SCALE} AS gamma,
            theta_1000000 / {RISK_SCALE} AS theta,
            vega_1000000 / {RISK_SCALE} AS vega,
            rho_1000000 / {RISK_SCALE} AS rho,
            vendor,
            source_path,
            ingested_at
        FROM read_parquet('{source_glob}')
        WHERE trade_date IS NOT NULL
          {date_filter}
    ),
    featured AS (
        SELECT
            *,
            ABS(delta) AS abs_delta,
            ask - bid AS spread,
            CASE WHEN mid IS NULL OR mid = 0 THEN NULL ELSE (ask - bid) / mid END AS spread_pct,
            CASE
                WHEN bid IS NULL OR ask IS NULL THEN 'missing_quote'
                WHEN bid > ask THEN 'inverted_quote'
                WHEN bid = 0 AND ask > 0 THEN 'zero_bid'
                WHEN mid IS NULL OR mid <= 0 THEN 'non_positive_mid'
                ELSE 'ok'
            END AS quote_quality_flag,
            LN(1 + COALESCE(volume, 0)) AS log_volume,
            LN(1 + COALESCE(open_interest, 0)) AS log_open_interest,
            (
                LN(1 + COALESCE(volume, 0))
                + LN(1 + COALESCE(open_interest, 0))
                - 10 * COALESCE(CASE WHEN mid IS NULL OR mid = 0 THEN NULL ELSE (ask - bid) / mid END, 1)
            ) AS liquidity_score
        FROM decoded
    )
    SELECT
        trade_date,
        expiry,
        underlying_symbol,
        option_symbol,
        option_id,
        stock_id,
        call_put,
        is_call,
        is_put,
        dte,
        dte_bucket,
        volume,
        open_interest,
        is_settlement,
        strike,
        bid,
        ask,
        mid,
        last,
        price_open,
        price_high,
        price_low,
        underlying_price,
        moneyness_pct,
        iv,
        preiv,
        delta,
        abs_delta,
        gamma,
        theta,
        vega,
        rho,
        spread,
        spread_pct,
        quote_quality_flag,
        log_volume,
        log_open_interest,
        liquidity_score,
        vendor,
        source_path,
        ingested_at
    FROM featured
    """


def _copy_symbol_month(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    year: int,
    month: int,
    source_glob: str,
    output_root: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    overwrite: bool,
) -> Optional[dict[str, Any]]:
    partition_dir = output_root / f"underlying_symbol={symbol}" / f"year={year}" / f"month={month:02d}"
    output_path = partition_dir / f"{symbol.lower()}_options_features_eod_{year}-{month:02d}.parquet"
    if output_path.exists() and not overwrite:
        return {
            "symbol": symbol,
            "year": year,
            "month": month,
            "path": str(output_path),
            "rows": None,
            "status": "skipped_existing",
        }

    query = _feature_select_sql(source_glob=source_glob, start_date=start_date, end_date=end_date)
    month_filter = f"year(trade_date) = {year} AND month(trade_date) = {month}"
    row_count = int(con.execute(f"SELECT COUNT(*) FROM ({query}) WHERE {month_filter}").fetchone()[0])
    if row_count <= 0:
        return None

    partition_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM ({query})
            WHERE {month_filter}
            ORDER BY trade_date, option_id
        ) TO ? (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 250000
        )
        """,
        [str(output_path)],
    )
    return {
        "symbol": symbol,
        "year": year,
        "month": month,
        "path": str(output_path),
        "rows": row_count,
        "status": "written",
    }


def build_feature_layer(
    symbols: Sequence[str],
    data_root: Path,
    output_root: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    overwrite: bool,
) -> dict[str, Any]:
    normalized_root = data_root / "normalized" / "options" / "chains_eod"
    partitions: list[dict[str, Any]] = []
    missing_symbols: list[str] = []

    with duckdb.connect(database=":memory:") as con:
        for symbol in symbols:
            source_dir = normalized_root / f"underlying_symbol={symbol}"
            if not source_dir.exists():
                logger.warning("Skipping %s because normalized options directory is missing: %s", symbol, source_dir)
                missing_symbols.append(symbol)
                continue

            source_glob = str(source_dir / "**" / "*.parquet")
            query = _feature_select_sql(source_glob=source_glob, start_date=start_date, end_date=end_date)
            months = con.execute(
                f"""
                SELECT DISTINCT year(trade_date)::INTEGER AS year, month(trade_date)::INTEGER AS month
                FROM ({query})
                ORDER BY 1, 2
                """
            ).fetchall()
            logger.info("Building %s feature partitions for %s", len(months), symbol)
            for year, month in months:
                partition = _copy_symbol_month(
                    con=con,
                    symbol=symbol,
                    year=int(year),
                    month=int(month),
                    source_glob=source_glob,
                    output_root=output_root,
                    start_date=start_date,
                    end_date=end_date,
                    overwrite=overwrite,
                )
                if partition:
                    partitions.append(partition)

    written_rows = sum(int(item["rows"] or 0) for item in partitions)
    return {
        "symbols_requested": list(symbols),
        "missing_symbols": missing_symbols,
        "output_root": str(output_root),
        "partitions": partitions,
        "partition_count": len(partitions),
        "written_partition_count": sum(1 for item in partitions if item["status"] == "written"),
        "skipped_partition_count": sum(1 for item in partitions if item["status"] == "skipped_existing"),
        "written_rows": written_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build research-ready option feature Parquet from normalized iVol data.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols. Defaults to all normalized symbols.")
    parser.add_argument("--data-root", default=None, help="Override MARKET_DATA_ROOT")
    parser.add_argument("--output-root", default=None, help="Override output root for feature Parquet")
    parser.add_argument("--start-date", default=None, help="Optional inclusive trade-date lower bound YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Optional inclusive trade-date upper bound YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature partitions")
    parser.add_argument("--run-label", default="options_feature_layer")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = _resolve_data_root(args.data_root)
    normalized_root = data_root / "normalized" / "options" / "chains_eod"
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else data_root / "research" / "options_features_eod"
    )

    start_date = date_literal(args.start_date) if args.start_date else None
    end_date = date_literal(args.end_date) if args.end_date else None
    symbols = _parse_symbols(args.symbols, normalized_options_root=normalized_root)

    result = build_feature_layer(
        symbols=symbols,
        data_root=data_root,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        overwrite=args.overwrite,
    )

    manifest_dir = _month_partition(data_root / "manifests" / "research")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.run_label}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    manifest = {
        "run_label": args.run_label,
        "created_at": _utc_now_iso(),
        "data_root": str(data_root),
        "start_date": start_date,
        "end_date": end_date,
        **result,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Feature layer complete: symbols=%d partitions=%d written=%d skipped=%d rows=%d",
        len(symbols),
        result["partition_count"],
        result["written_partition_count"],
        result["skipped_partition_count"],
        result["written_rows"],
    )
    logger.info("Wrote feature manifest to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
