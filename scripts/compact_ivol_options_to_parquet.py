#!/usr/bin/env python3
"""
Compact raw iVolatility JSON snapshots into partitioned Parquet + ZSTD datasets.

This script preserves raw ingestion files as the audit layer and writes a
normalized, reversible, compressed analytics layer to the market-data warehouse.
It supports:
- option-chain snapshots from stock-opts-by-param payloads
- underlying EOD price payloads
- monthly partitioning by symbol/year/month
- fixed-point reversible scaling for prices, IV, and Greeks
- Parquet + ZSTD output with schema metadata
- best-effort materialization of deferred iVol results via urlForDetails
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import duckdb
import numpy as np
import pandas as pd
import requests

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")
except Exception:
    pass

from utils.logger import setup_logger

logger = setup_logger(__name__)

PRICE_SCALE = 10_000
RISK_SCALE = 1_000_000

OPTION_FLOAT_SCALE_MAP: dict[str, int] = {
    "strike": PRICE_SCALE,
    "bid": PRICE_SCALE,
    "ask": PRICE_SCALE,
    "mid": PRICE_SCALE,
    "last": PRICE_SCALE,
    "price_open": PRICE_SCALE,
    "price_high": PRICE_SCALE,
    "price_low": PRICE_SCALE,
    "underlying_price": PRICE_SCALE,
    "moneyness_pct": PRICE_SCALE,
    "iv": RISK_SCALE,
    "preiv": RISK_SCALE,
    "delta": RISK_SCALE,
    "gamma": RISK_SCALE,
    "theta": RISK_SCALE,
    "vega": RISK_SCALE,
    "rho": RISK_SCALE,
}

UNDERLYING_FLOAT_SCALE_MAP: dict[str, int] = {
    "open": PRICE_SCALE,
    "high": PRICE_SCALE,
    "low": PRICE_SCALE,
    "close": PRICE_SCALE,
    "adjusted_close": PRICE_SCALE,
}


@dataclass(frozen=True)
class PayloadLoadResult:
    frame: pd.DataFrame
    source_format: str
    deferred_materialized: bool
    records_found: int
    source_path: str


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _require_api_key(explicit_value: Optional[str]) -> str:
    api_key = explicit_value or os.environ.get("IVOLATILITY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("IVOLATILITY_API_KEY is not configured. Set it in .env or pass --api-key.")
    return api_key


def _resolve_data_root(explicit_root: Optional[str]) -> Path:
    configured = explicit_root or os.environ.get("MARKET_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("/Volumes/T9/market_data")


def _month_partition(base_dir: Path) -> Path:
    now = datetime.now(UTC)
    return base_dir / f"{now:%Y}" / f"{now:%m}"


def _scale_nullable_series(series: pd.Series, scale: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    scaled = np.rint(numeric * scale)
    return pd.Series(scaled, index=series.index, dtype="Float64").astype("Int64")


def _normalize_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _with_api_key(url: str, api_key: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("apiKey", api_key)
    return urlunparse(parsed._replace(query=urlencode(query)))


def _materialize_details_frame(details_url: str, api_key: str, session: requests.Session) -> tuple[pd.DataFrame, str]:
    resolved_url = _with_api_key(details_url, api_key)
    response = session.get(resolved_url, timeout=60)
    response.raise_for_status()
    content = response.content

    if not content:
        return pd.DataFrame(), "empty_details"

    if content[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
            csv_bytes = gz.read()
        return pd.read_csv(io.BytesIO(csv_bytes)), "csv_gzip"

    stripped = content.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        payload = json.loads(content.decode("utf-8"))
        if isinstance(payload, dict) and "data" in payload:
            data = payload.get("data", [])
            if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("urlForDownload"):
                return _materialize_details_frame(str(data[0]["urlForDownload"]), api_key=api_key, session=session)
            return pd.DataFrame(data), "json_details"
        if isinstance(payload, list):
            if (
                payload
                and isinstance(payload[0], dict)
                and isinstance(payload[0].get("data"), list)
                and payload[0]["data"]
                and isinstance(payload[0]["data"][0], dict)
                and payload[0]["data"][0].get("urlForDownload")
            ):
                return _materialize_details_frame(
                    str(payload[0]["data"][0]["urlForDownload"]),
                    api_key=api_key,
                    session=session,
                )
            return pd.DataFrame(payload), "json_list"

    return pd.read_csv(io.BytesIO(content)), "csv"


def _load_json_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_ivol_option_payload(path: Path, api_key: str, session: requests.Session) -> PayloadLoadResult:
    payload = _load_json_payload(path)
    status = payload.get("status", {}) if isinstance(payload, dict) else {}
    data = payload.get("data", []) if isinstance(payload, dict) else []
    records_found = int(status.get("recordsFound") or 0)

    if data:
        return PayloadLoadResult(
            frame=pd.DataFrame(data),
            source_format="json_inline",
            deferred_materialized=False,
            records_found=records_found,
            source_path=str(path),
        )

    details_url = status.get("urlForDetails")
    if records_found > 0 and details_url:
        frame, source_format = _materialize_details_frame(str(details_url), api_key=api_key, session=session)
        return PayloadLoadResult(
            frame=frame,
            source_format=source_format,
            deferred_materialized=True,
            records_found=records_found,
            source_path=str(path),
        )

    return PayloadLoadResult(
        frame=pd.DataFrame(),
        source_format="json_inline",
        deferred_materialized=False,
        records_found=records_found,
        source_path=str(path),
    )


def _load_ivol_underlying_payload(path: Path, api_key: str, session: requests.Session) -> PayloadLoadResult:
    payload = _load_json_payload(path)
    status = payload.get("status", {}) if isinstance(payload, dict) else {}
    data = payload.get("data", []) if isinstance(payload, dict) else []
    records_found = int(status.get("recordsFound") or 0)

    if data:
        return PayloadLoadResult(
            frame=pd.DataFrame(data),
            source_format="json_inline",
            deferred_materialized=False,
            records_found=records_found,
            source_path=str(path),
        )

    details_url = status.get("urlForDetails")
    if records_found > 0 and details_url:
        frame, source_format = _materialize_details_frame(str(details_url), api_key=api_key, session=session)
        return PayloadLoadResult(
            frame=frame,
            source_format=source_format,
            deferred_materialized=True,
            records_found=records_found,
            source_path=str(path),
        )

    return PayloadLoadResult(
        frame=pd.DataFrame(),
        source_format="json_inline",
        deferred_materialized=False,
        records_found=records_found,
        source_path=str(path),
    )


def _normalize_option_frame(frame: pd.DataFrame, source_path: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    if "c_date" not in frame.columns and "trade_date" not in frame.columns:
        logger.info("Skipping unsupported option payload without trade-date field: %s", source_path)
        return pd.DataFrame()

    if "option_symbol" not in frame.columns and "OptionSymbol" not in frame.columns:
        logger.info("Skipping unsupported option payload without option-symbol field: %s", source_path)
        return pd.DataFrame()

    renamed = frame.rename(
        columns={
            "c_date": "trade_date",
            "expiration_date": "expiry",
            "call_put": "call_put",
            "price_strike": "strike",
            "openinterest": "open_interest",
            "Ask": "ask",
            "Bid": "bid",
            "price": "last",
            "calc_OTM": "moneyness_pct",
            "stocks_id": "stock_id",
        }
    ).copy()

    renamed["trade_date"] = _normalize_datetime_series(renamed["trade_date"])
    renamed["expiry"] = _normalize_datetime_series(renamed["expiry"])
    renamed["underlying_symbol"] = renamed.get("option_symbol", "").astype(str).str.split().str[0]
    renamed["vendor"] = "ivolatility"
    renamed["source_path"] = source_path
    renamed["ingested_at"] = _utc_now_iso()
    renamed["mid"] = (
        pd.to_numeric(renamed.get("bid"), errors="coerce")
        + pd.to_numeric(renamed.get("ask"), errors="coerce")
    ) / 2.0

    for col in OPTION_FLOAT_SCALE_MAP:
        if col in renamed.columns:
            renamed[f"{col}_{OPTION_FLOAT_SCALE_MAP[col]}"] = _scale_nullable_series(
                renamed[col], OPTION_FLOAT_SCALE_MAP[col]
            )

    int_like_cols = {
        "option_id": "Int64",
        "stock_id": "Int64",
        "dte": "Int32",
        "volume": "Int64",
        "open_interest": "Int64",
        "is_settlement": "Int8",
    }
    for col, dtype in int_like_cols.items():
        if col in renamed.columns:
            renamed[col] = pd.to_numeric(renamed[col], errors="coerce").astype(dtype)

    ordered_cols = [
        "trade_date",
        "expiry",
        "underlying_symbol",
        "option_symbol",
        "option_id",
        "stock_id",
        "call_put",
        "dte",
        "volume",
        "open_interest",
        "is_settlement",
        "vendor",
        "source_path",
        "ingested_at",
    ]
    ordered_cols.extend(
        [f"{col}_{OPTION_FLOAT_SCALE_MAP[col]}" for col in OPTION_FLOAT_SCALE_MAP if f"{col}_{OPTION_FLOAT_SCALE_MAP[col]}" in renamed.columns]
    )
    existing_cols = [col for col in ordered_cols if col in renamed.columns]
    normalized = renamed[existing_cols].copy()
    normalized = normalized.dropna(subset=["trade_date", "expiry", "option_symbol"], how="any")
    normalized = normalized.drop_duplicates(subset=["trade_date", "option_id"], keep="last")
    return normalized


def _normalize_underlying_frame(frame: pd.DataFrame, source_path: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    if "date" not in frame.columns and "trade_date" not in frame.columns:
        logger.info("Skipping unsupported underlying payload without trade-date field: %s", source_path)
        return pd.DataFrame()

    if "symbol" not in frame.columns and "underlying_symbol" not in frame.columns:
        logger.info("Skipping unsupported underlying payload without symbol field: %s", source_path)
        return pd.DataFrame()

    renamed = frame.rename(
        columns={
            "date": "trade_date",
            "symbol": "underlying_symbol",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjClose": "adjusted_close",
            "adj_close": "adjusted_close",
        }
    ).copy()
    renamed["trade_date"] = _normalize_datetime_series(renamed["trade_date"])
    renamed["vendor"] = "ivolatility"
    renamed["source_path"] = source_path
    renamed["ingested_at"] = _utc_now_iso()

    for col in UNDERLYING_FLOAT_SCALE_MAP:
        if col in renamed.columns:
            renamed[f"{col}_{UNDERLYING_FLOAT_SCALE_MAP[col]}"] = _scale_nullable_series(
                renamed[col], UNDERLYING_FLOAT_SCALE_MAP[col]
            )

    if "volume" in renamed.columns:
        renamed["volume"] = pd.to_numeric(renamed["volume"], errors="coerce").astype("Int64")

    ordered_cols = [
        "trade_date",
        "underlying_symbol",
        "volume",
        "vendor",
        "source_path",
        "ingested_at",
    ]
    ordered_cols.extend(
        [
            f"{col}_{UNDERLYING_FLOAT_SCALE_MAP[col]}"
            for col in UNDERLYING_FLOAT_SCALE_MAP
            if f"{col}_{UNDERLYING_FLOAT_SCALE_MAP[col]}" in renamed.columns
        ]
    )
    existing_cols = [col for col in ordered_cols if col in renamed.columns]
    normalized = renamed[existing_cols].copy()
    normalized = normalized.dropna(subset=["trade_date", "underlying_symbol"], how="any")
    normalized = normalized.drop_duplicates(subset=["trade_date", "underlying_symbol"], keep="last")
    return normalized


def _iter_files(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(root.rglob(pattern))


def _filter_symbol_paths(paths: Iterable[Path], symbols: Optional[set[str]]) -> list[Path]:
    if not symbols:
        return list(paths)
    filtered: list[Path] = []
    for path in paths:
        lower_name = path.name.lower()
        if any(f"_{symbol.lower()}_" in lower_name for symbol in symbols):
            filtered.append(path)
    return filtered


def _append_existing_rows(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        return pd.DataFrame()
    with duckdb.connect(database=":memory:") as con:
        return con.execute("SELECT * FROM read_parquet(?)", [str(parquet_path)]).fetch_df()


def _canonicalize_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical = df.copy()
    if "trade_date" in canonical.columns:
        canonical["trade_date"] = _normalize_datetime_series(canonical["trade_date"])
    if "expiry" in canonical.columns:
        canonical["expiry"] = _normalize_datetime_series(canonical["expiry"])
    return canonical


def _write_partitioned_parquet(
    df: pd.DataFrame,
    dataset_type: str,
    base_root: Path,
    schema_metadata: Dict[str, Any],
) -> list[str]:
    if df.empty:
        return []

    output_paths: list[str] = []

    for (symbol, year, month), group in df.groupby(
        [
            "underlying_symbol",
            pd.to_datetime(df["trade_date"]).dt.year,
            pd.to_datetime(df["trade_date"]).dt.month,
        ]
    ):
        partition_dir = base_root / f"underlying_symbol={symbol}" / f"year={year}" / f"month={month:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / f"{symbol.lower()}_{dataset_type}_{year}-{month:02d}.parquet"

        existing = _canonicalize_temporal_columns(_append_existing_rows(parquet_path))
        combined = _canonicalize_temporal_columns(pd.concat([existing, group], ignore_index=True))
        primary_key = ["trade_date", "underlying_symbol"]
        if dataset_type == "options_eod":
            primary_key = ["trade_date", "option_id"]
        combined = combined.drop_duplicates(subset=primary_key, keep="last").sort_values(by=primary_key)
        with duckdb.connect(database=":memory:") as con:
            con.register("combined_df", combined)
            con.execute(
                """
                COPY combined_df TO ? (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD,
                    ROW_GROUP_SIZE 250000
                )
                """,
                [str(parquet_path)],
            )
        output_paths.append(str(parquet_path))
    return output_paths


def _write_schema_sidecar(data_root: Path) -> Path:
    schema_dir = data_root / "manifests" / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema_path = schema_dir / "ivol_parquet_fixed_point_v1.json"
    payload = {
        "version": 1,
        "created_at": _utc_now_iso(),
        "options_scale_map": OPTION_FLOAT_SCALE_MAP,
        "underlyings_scale_map": UNDERLYING_FLOAT_SCALE_MAP,
        "compression": "zstd",
        "format": "parquet",
    }
    schema_path.write_text(json.dumps(payload, indent=2))
    return schema_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact iVolatility raw data into Parquet + ZSTD.")
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbol filter. Default: SPY")
    parser.add_argument("--data-root", default=None, help="Override MARKET_DATA_ROOT")
    parser.add_argument("--api-key", default=None, help="Optional iVolatility API key override")
    parser.add_argument("--include-underlyings", action="store_true", help="Also compact stock-prices payloads")
    parser.add_argument("--options-only", action="store_true", help="Compact only option payloads")
    parser.add_argument("--underlyings-only", action="store_true", help="Compact only underlying stock-price payloads")
    parser.add_argument("--source-year", default=None, help="Limit raw source directory year, e.g. 2026")
    parser.add_argument("--source-month", default=None, help="Limit raw source directory month, e.g. 04")
    parser.add_argument("--run-label", default="ivol_parquet_compaction")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = _resolve_data_root(args.data_root)
    api_key = _require_api_key(args.api_key)
    session = requests.Session()
    symbols = {item.strip().upper() for item in args.symbols.split(",") if item.strip()}

    if args.options_only and args.underlyings_only:
        raise ValueError("Use either --options-only or --underlyings-only, not both.")

    source_suffix = Path()
    if args.source_year:
        source_suffix /= args.source_year
    if args.source_month:
        source_suffix /= args.source_month

    raw_options_root = data_root / "raw" / "ivolatility" / "options_chains" / source_suffix
    raw_reference_root = data_root / "raw" / "ivolatility" / "reference" / source_suffix

    normalized_root = data_root / "normalized" / "options" / "chains_eod"
    written_option_paths: list[str] = []
    option_paths: list[Path] = []
    option_materialized = 0
    option_skipped = 0
    option_unsupported = 0
    option_errors: list[dict[str, str]] = []
    if not args.underlyings_only:
        option_paths = _filter_symbol_paths(_iter_files(raw_options_root, "*.json"), symbols)
        logger.info("Found %d raw option files under %s", len(option_paths), raw_options_root)

        option_frames: list[pd.DataFrame] = []
        for path in option_paths:
            try:
                payload = _load_ivol_option_payload(path, api_key=api_key, session=session)
            except Exception as exc:
                logger.warning("Skipping %s due to materialization error: %s", path, exc)
                option_errors.append({"path": str(path), "error": str(exc)})
                continue
            if payload.frame.empty:
                option_skipped += 1
                continue
            normalized = _normalize_option_frame(payload.frame, source_path=payload.source_path)
            if normalized.empty:
                option_skipped += 1
                option_unsupported += 1
                continue
            option_frames.append(normalized)
            if payload.deferred_materialized:
                option_materialized += 1

        if option_frames:
            option_df = pd.concat(option_frames, ignore_index=True)
            written_option_paths = _write_partitioned_parquet(
                df=option_df,
                dataset_type="options_eod",
                base_root=normalized_root,
                schema_metadata=OPTION_FLOAT_SCALE_MAP,
            )

    written_underlying_paths: list[str] = []
    underlying_materialized = 0
    if args.include_underlyings or args.underlyings_only:
        underlying_paths = _filter_symbol_paths(_iter_files(raw_reference_root, "*.json"), symbols)
        logger.info("Found %d raw underlying files under %s", len(underlying_paths), raw_reference_root)
        underlying_frames: list[pd.DataFrame] = []
        for path in underlying_paths:
            payload = _load_ivol_underlying_payload(path, api_key=api_key, session=session)
            if payload.frame.empty:
                continue
            normalized = _normalize_underlying_frame(payload.frame, source_path=payload.source_path)
            if normalized.empty:
                continue
            underlying_frames.append(normalized)
            if payload.deferred_materialized:
                underlying_materialized += 1
        if underlying_frames:
            underlying_df = pd.concat(underlying_frames, ignore_index=True)
            written_underlying_paths = _write_partitioned_parquet(
                df=underlying_df,
                dataset_type="underlyings_eod",
                base_root=data_root / "normalized" / "underlyings" / "daily_ohlcv",
                schema_metadata=UNDERLYING_FLOAT_SCALE_MAP,
            )

    schema_path = _write_schema_sidecar(data_root)
    manifest_dir = _month_partition(data_root / "manifests" / "coverage")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.run_label}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    manifest = {
        "run_label": args.run_label,
        "created_at": _utc_now_iso(),
        "symbols": sorted(symbols),
        "raw_options_root": str(raw_options_root),
        "raw_reference_root": str(raw_reference_root),
        "option_files_found": len(option_paths),
        "option_files_materialized_via_details": option_materialized,
        "option_files_skipped": option_skipped,
        "option_files_unsupported_schema": option_unsupported,
        "option_errors": option_errors,
        "normalized_option_partitions_written": written_option_paths,
        "normalized_underlying_partitions_written": written_underlying_paths,
        "underlying_files_materialized_via_details": underlying_materialized,
        "schema_path": str(schema_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote compaction manifest to %s", manifest_path)
    logger.info(
        "Compaction complete: option_partitions=%d underlying_partitions=%d option_detail_materializations=%d underlying_detail_materializations=%d",
        len(written_option_paths),
        len(written_underlying_paths),
        option_materialized,
        underlying_materialized,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
