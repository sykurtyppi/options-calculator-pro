#!/usr/bin/env python3
"""Collect iVolatility intraday stock price snapshots and optional pivot bars.

The iVolatility intraday stock-prices endpoint returns timestamped price
snapshots (bid/ask/last), not native OHLC candles. For PIVOT_QUANT, this script
can aggregate those snapshots into bar_data rows using lastPrice as the source
price. That makes the approximation explicit and keeps raw snapshots preserved.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import duckdb
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

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

logger = setup_logger(__name__)

BASE_URL = "https://restapi.ivolatility.com"
ENDPOINT = "/equities/intraday/stock-prices"
NY_TZ = ZoneInfo("America/New_York") if ZoneInfo else UTC
SUPPORTED_MINUTE_TYPES = {"MINUTE_1", "MINUTE_5", "MINUTE_15", "MINUTE_30", "HOUR"}


@dataclass(frozen=True)
class IntradayFetchResult:
    raw_path: Path
    frame: pd.DataFrame
    records_found: int
    deferred_materialized: bool


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_data_root(explicit_root: str | None) -> Path:
    configured = explicit_root or os.environ.get("MARKET_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("/Volumes/T9/market_data")


def _require_api_key(explicit_key: str | None) -> str:
    api_key = explicit_key or os.environ.get("IVOLATILITY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("IVOLATILITY_API_KEY is not configured.")
    return api_key


def _iter_weekdays(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def _with_api_key(url: str, api_key: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("apiKey", api_key)
    return urlunparse(parsed._replace(query=urlencode(query)))


def _materialize_details(details_url: str, api_key: str, session: requests.Session) -> pd.DataFrame:
    response = session.get(_with_api_key(details_url, api_key), timeout=120)
    response.raise_for_status()
    content = response.content
    if not content:
        return pd.DataFrame()
    if content[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
            return pd.read_csv(io.BytesIO(gz.read()))
    stripped = content.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        payload = json.loads(content.decode("utf-8"))
        if isinstance(payload, dict):
            data = payload.get("data", [])
            if data and isinstance(data[0], dict) and data[0].get("urlForDownload"):
                return _materialize_details(str(data[0]["urlForDownload"]), api_key, session)
            return pd.DataFrame(data)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
    return pd.read_csv(io.BytesIO(content))


def _fetch_day(
    *,
    symbol: str,
    trade_date: date,
    minute_type: str,
    api_key: str,
    data_root: Path,
    session: requests.Session,
    run_timestamp: str,
) -> IntradayFetchResult:
    params = {
        "apiKey": api_key,
        "symbol": symbol,
        "date": trade_date.isoformat(),
        "minuteType": minute_type,
    }
    response = session.get(BASE_URL + ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    raw_dir = data_root / "raw" / "ivolatility" / "intraday_stock_prices" / run_timestamp[:4] / run_timestamp[4:6]
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"ivol_{symbol.lower()}_{minute_type.lower()}_{trade_date.isoformat()}_{run_timestamp}.json"
    raw_path.write_text(json.dumps(payload, indent=2))

    status = payload.get("status", {}) if isinstance(payload, dict) else {}
    records_found = int(status.get("recordsFound") or 0)
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if data:
        return IntradayFetchResult(raw_path, pd.DataFrame(data), records_found, False)

    details_url = status.get("urlForDetails")
    if records_found > 0 and details_url:
        frame = _materialize_details(str(details_url), api_key, session)
        return IntradayFetchResult(raw_path, frame, records_found, True)

    return IntradayFetchResult(raw_path, pd.DataFrame(), records_found, False)


def _normalize_snapshots(frame: pd.DataFrame, source_path: str, ingested_at: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    renamed = frame.rename(
        columns={
            "timestamp": "timestamp_et",
            "stockSymbol": "underlying_symbol",
            "bidPrice": "bid",
            "askPrice": "ask",
            "lastPrice": "last",
            "bidSize": "bid_size",
            "askSize": "ask_size",
            "lastSize": "last_size",
            "calcTimestamp": "calc_timestamp_et",
        }
    ).copy()
    if "timestamp_et" not in renamed.columns or "underlying_symbol" not in renamed.columns:
        return pd.DataFrame()

    ts_local = pd.to_datetime(renamed["timestamp_et"], errors="coerce")
    renamed["timestamp_utc"] = ts_local.map(
        lambda value: value.replace(tzinfo=NY_TZ).astimezone(UTC) if pd.notnull(value) else pd.NaT
    )
    renamed["ts_ms"] = renamed["timestamp_utc"].map(
        lambda value: int(value.timestamp() * 1000) if pd.notnull(value) else None
    )
    for col in ("bid", "ask", "last", "bid_size", "ask_size", "last_size", "volume"):
        if col in renamed.columns:
            renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
    renamed["vendor"] = "ivolatility"
    renamed["source_path"] = source_path
    renamed["ingested_at"] = ingested_at
    columns = [
        "ts_ms",
        "timestamp_utc",
        "timestamp_et",
        "underlying_symbol",
        "type",
        "currency",
        "bid",
        "ask",
        "last",
        "bid_size",
        "ask_size",
        "last_size",
        "volume",
        "calc_timestamp_et",
        "vendor",
        "source_path",
        "ingested_at",
    ]
    existing = [col for col in columns if col in renamed.columns]
    normalized = renamed[existing].dropna(subset=["ts_ms", "underlying_symbol"]).copy()
    return normalized.sort_values(["underlying_symbol", "ts_ms"]).drop_duplicates(["underlying_symbol", "ts_ms"], keep="last")


def _write_monthly_snapshots(df: pd.DataFrame, data_root: Path, symbol: str, minute_type: str) -> list[Path]:
    if df.empty:
        return []
    out = df.copy()
    out["year"] = pd.to_datetime(out["timestamp_utc"]).dt.year.astype(int)
    out["month"] = pd.to_datetime(out["timestamp_utc"]).dt.strftime("%m")
    written: list[Path] = []
    con = duckdb.connect()
    try:
        for (year, month), group in out.groupby(["year", "month"], sort=True):
            out_dir = (
                data_root
                / "normalized"
                / "underlyings"
                / "intraday_stock_prices"
                / f"underlying_symbol={symbol.upper()}"
                / f"minute_type={minute_type}"
                / f"year={int(year)}"
                / f"month={month}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{symbol.lower()}_intraday_stock_prices_{minute_type.lower()}_{int(year)}-{month}.parquet"
            existing = pd.DataFrame()
            if out_path.exists():
                existing = con.execute("SELECT * FROM read_parquet(?)", [str(out_path)]).fetch_df()
            combined = pd.concat([existing, group], ignore_index=True).drop_duplicates(
                ["underlying_symbol", "ts_ms"], keep="last"
            ).sort_values(["underlying_symbol", "ts_ms"])
            con.register("snapshots", combined)
            con.execute(f"COPY snapshots TO '{out_path}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD')")
            con.unregister("snapshots")
            written.append(out_path)
    finally:
        con.close()
    return written


def _snapshot_price(row: pd.Series) -> float | None:
    last = row.get("last")
    if pd.notnull(last) and float(last) > 0:
        return float(last)
    bid = row.get("bid")
    ask = row.get("ask")
    if pd.notnull(bid) and pd.notnull(ask) and float(bid) > 0 and float(ask) > 0:
        return (float(bid) + float(ask)) / 2.0
    return None


def _snapshots_to_bars(df: pd.DataFrame, interval_sec: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["price"] = work.apply(_snapshot_price, axis=1)
    work = work.dropna(subset=["price", "ts_ms", "underlying_symbol"])
    if work.empty:
        return pd.DataFrame()
    work["bar_ts_ms"] = (work["ts_ms"].astype("int64") // (interval_sec * 1000)) * (interval_sec * 1000)

    rows: list[dict[str, Any]] = []
    for (symbol, bar_ts), group in work.groupby(["underlying_symbol", "bar_ts_ms"], sort=True):
        prices = group.sort_values("ts_ms")["price"]
        volumes = pd.to_numeric(group.get("volume"), errors="coerce") if "volume" in group else pd.Series(dtype=float)
        volume = 0.0
        if not volumes.empty and volumes.notna().any():
            if len(volumes.dropna()) > 1 and volumes.is_monotonic_increasing:
                volume = float(max(volumes.max() - volumes.min(), 0))
            else:
                volume = float(volumes.fillna(0).sum())
        rows.append(
            {
                "symbol": symbol,
                "ts": int(bar_ts),
                "open": float(prices.iloc[0]),
                "high": float(prices.max()),
                "low": float(prices.min()),
                "close": float(prices.iloc[-1]),
                "volume": volume,
                "bar_interval_sec": int(interval_sec),
            }
        )
    return pd.DataFrame(rows)


def _ensure_pivot_bar_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bar_data (
            symbol TEXT NOT NULL,
            ts INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL,
            bar_interval_sec INTEGER,
            PRIMARY KEY (symbol, ts, bar_interval_sec)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bar_symbol_ts ON bar_data(symbol, ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bar_symbol_interval ON bar_data(symbol, bar_interval_sec, ts);")
    conn.commit()


def _write_pivot_bars(db_path: Path, bars: pd.DataFrame) -> int:
    if bars.empty:
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_pivot_bar_table(conn)
        rows = [
            (
                row.symbol,
                int(row.ts),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume or 0),
                int(row.bar_interval_sec),
            )
            for row in bars.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO bar_data
            (symbol, ts, open, high, low, close, volume, bar_interval_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect iVol intraday stock snapshots.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--minute-type", default="MINUTE_1", choices=sorted(SUPPORTED_MINUTE_TYPES))
    parser.add_argument("--data-root")
    parser.add_argument("--api-key")
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--run-label", default="ivol_intraday_stock_prices")
    parser.add_argument("--pivot-db", help="Optional PIVOT_QUANT SQLite DB to receive derived bar_data rows")
    parser.add_argument("--pivot-interval-sec", type=int, default=60, choices=[60, 300, 900, 1800, 3600])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbol = args.symbol.upper()
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    if start_date > end_date:
        raise ValueError("--start-date must be on or before --end-date")
    api_key = _require_api_key(args.api_key)
    data_root = _resolve_data_root(args.data_root)
    run_timestamp = _utc_timestamp()
    ingested_at = _utc_now_iso()
    session = requests.Session()

    normalized_frames: list[pd.DataFrame] = []
    raw_paths: list[str] = []
    deferred_materializations = 0
    records_found = 0
    failed_dates: list[str] = []

    for trade_date in _iter_weekdays(start_date, end_date):
        try:
            result = _fetch_day(
                symbol=symbol,
                trade_date=trade_date,
                minute_type=args.minute_type,
                api_key=api_key,
                data_root=data_root,
                session=session,
                run_timestamp=run_timestamp,
            )
            raw_paths.append(str(result.raw_path))
            records_found += result.records_found
            if result.deferred_materialized:
                deferred_materializations += 1
            normalized = _normalize_snapshots(result.frame, str(result.raw_path), ingested_at)
            if not normalized.empty:
                normalized_frames.append(normalized)
            logger.info(
                "Fetched %s %s %s records_found=%s normalized=%s",
                symbol,
                trade_date.isoformat(),
                args.minute_type,
                result.records_found,
                len(normalized),
            )
        except Exception as exc:
            failed_dates.append(trade_date.isoformat())
            logger.error("Failed %s %s %s: %s", symbol, trade_date.isoformat(), args.minute_type, exc)
        time.sleep(args.sleep_seconds)

    snapshots = pd.concat(normalized_frames, ignore_index=True) if normalized_frames else pd.DataFrame()
    snapshot_paths = _write_monthly_snapshots(snapshots, data_root, symbol, args.minute_type)
    pivot_rows_written = 0
    if args.pivot_db:
        bars = _snapshots_to_bars(snapshots, args.pivot_interval_sec)
        pivot_rows_written = _write_pivot_bars(Path(args.pivot_db), bars)
        logger.info(
            "Wrote %d derived %ss PIVOT bar_data rows from iVol lastPrice snapshots",
            pivot_rows_written,
            args.pivot_interval_sec,
        )

    manifest_dir = data_root / "manifests" / "research" / run_timestamp[:4] / run_timestamp[4:6]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.run_label}_{run_timestamp}.json"
    manifest = {
        "run_label": args.run_label,
        "created_at": _utc_now_iso(),
        "symbol": symbol,
        "endpoint": ENDPOINT,
        "minute_type": args.minute_type,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "records_found": records_found,
        "normalized_rows": int(len(snapshots)),
        "raw_paths": raw_paths,
        "normalized_partitions": [str(path) for path in snapshot_paths],
        "deferred_materializations": deferred_materializations,
        "failed_dates": failed_dates,
        "pivot_db": args.pivot_db,
        "pivot_interval_sec": args.pivot_interval_sec if args.pivot_db else None,
        "pivot_rows_written": pivot_rows_written,
        "bar_construction": "OHLC aggregated from iVol lastPrice snapshots",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest to %s", manifest_path)
    return 1 if failed_dates else 0


if __name__ == "__main__":
    raise SystemExit(main())
