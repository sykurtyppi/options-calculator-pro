#!/usr/bin/env python3
"""Collect daily underlying OHLCV history from Yahoo Finance.

This complements the iVolatility option-chain warehouse with long-run price
action data for regime research and cross-tool backtests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger

logger = setup_logger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class PriceHistoryOutputs:
    raw_path: Path
    normalized_paths: list[Path]
    feature_paths: list[Path]
    manifest_path: Path


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve_data_root(explicit_root: str | None) -> Path:
    configured = explicit_root or os.environ.get("MARKET_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("/Volumes/T9/market_data")


def _safe_symbol(symbol: str) -> str:
    return symbol.upper().replace("/", "_").replace(" ", "")


def _default_start(end_date: date, years: int) -> date:
    try:
        return end_date.replace(year=end_date.year - years)
    except ValueError:
        return end_date - timedelta(days=365 * years)


def _fetch_yahoo_chart(symbol: str, start_date: date, end_date: date) -> dict[str, Any]:
    period1 = int(datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC).timestamp())
    # Yahoo's period2 is exclusive, so request one day past the desired end.
    period2 = int(
        datetime(end_date.year, end_date.month, end_date.day, tzinfo=UTC).timestamp()
        + SECONDS_PER_DAY
    )
    params = urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    url = f"{YAHOO_CHART_URL.format(symbol=symbol.upper())}?{params}"
    req = Request(url, headers={"User-Agent": "OptionsCalculatorPro/price-history"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chart_to_frame(payload: dict[str, Any], symbol: str, source_path: Path, ingested_at: str) -> pd.DataFrame:
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo chart error for {symbol}: {error}")
    result = (chart.get("result") or [{}])[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []

    rows: list[dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        try:
            open_ = quote["open"][idx]
            high = quote["high"][idx]
            low = quote["low"][idx]
            close = quote["close"][idx]
            if any(value is None for value in (open_, high, low, close)):
                continue
            volume = quote.get("volume", [None] * len(timestamps))[idx]
            adj = adjclose[idx] if idx < len(adjclose) else close
            trade_date = datetime.fromtimestamp(int(ts), tz=UTC).date()
            rows.append(
                {
                    "trade_date": trade_date,
                    "underlying_symbol": symbol.upper(),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "adj_close": float(adj) if adj is not None else float(close),
                    "volume": int(volume or 0),
                    "vendor": "yahoo",
                    "source_path": str(source_path),
                    "ingested_at": ingested_at,
                }
            )
        except (IndexError, KeyError, TypeError, ValueError):
            continue

    if not rows:
        raise ValueError(f"Yahoo returned no usable daily OHLCV rows for {symbol}")

    df = pd.DataFrame(rows).sort_values("trade_date").drop_duplicates(["trade_date"])
    return df.reset_index(drop=True)


def _scaled_price(value: float) -> int | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return int(round(float(value) * 10000))


def _normalized_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[f"{col}_10000"] = out[col].map(_scaled_price)
    out["year"] = pd.to_datetime(out["trade_date"]).dt.year.astype(int)
    out["month"] = pd.to_datetime(out["trade_date"]).dt.strftime("%m")
    columns = [
        "trade_date",
        "underlying_symbol",
        "volume",
        "vendor",
        "source_path",
        "ingested_at",
        "open_10000",
        "high_10000",
        "low_10000",
        "close_10000",
        "year",
        "month",
    ]
    return out[columns]


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    log_return = (close / prev_close).map(lambda v: math.log(v) if pd.notnull(v) and v > 0 else None)

    out["return_1d"] = close.pct_change()
    out["log_return_1d"] = log_return
    out["true_range"] = true_range
    out["atr14"] = true_range.rolling(14, min_periods=5).mean()
    out["rv20_ann"] = out["log_return_1d"].rolling(20, min_periods=10).std() * math.sqrt(252)
    out["rv60_ann"] = out["log_return_1d"].rolling(60, min_periods=20).std() * math.sqrt(252)
    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()
    out["close_to_ema50_pct"] = (close / out["ema50"]) - 1.0
    out["drawdown_from_252_high_pct"] = (close / close.rolling(252, min_periods=20).max()) - 1.0
    out["regime_vol"] = pd.cut(
        out["rv20_ann"],
        bins=[-math.inf, 0.12, 0.22, math.inf],
        labels=["low_vol", "normal_vol", "high_vol"],
    ).astype("string")
    out["regime_trend"] = "range"
    out.loc[(close > out["ema50"]) & (out["ema50"] > out["ema200"]), "regime_trend"] = "trend_up"
    out.loc[(close < out["ema50"]) & (out["ema50"] < out["ema200"]), "regime_trend"] = "trend_down"
    out["year"] = pd.to_datetime(out["trade_date"]).dt.year.astype(int)
    out["month"] = pd.to_datetime(out["trade_date"]).dt.strftime("%m")
    return out


def _write_monthly_parquet(df: pd.DataFrame, root: Path, symbol: str, suffix: str) -> list[Path]:
    written: list[Path] = []
    con = duckdb.connect()
    try:
        for (year, month), group in df.groupby(["year", "month"], sort=True):
            out_dir = root / f"underlying_symbol={symbol.upper()}" / f"year={int(year)}" / f"month={month}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{symbol.lower()}_{suffix}_{int(year)}-{month}.parquet"
            con.register("monthly_df", group.reset_index(drop=True))
            con.execute(f"COPY monthly_df TO '{out_path}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD')")
            con.unregister("monthly_df")
            written.append(out_path)
    finally:
        con.close()
    return written


def collect_price_history(
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    data_root: Path,
    run_label: str,
) -> PriceHistoryOutputs:
    safe_symbol = _safe_symbol(symbol)
    timestamp = _utc_timestamp()
    ingested_at = datetime.now(UTC).isoformat()

    raw_dir = data_root / "raw" / "yahoo" / "ohlcv" / "1d" / f"{timestamp[:4]}" / f"{timestamp[4:6]}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / (
        f"yahoo_{safe_symbol.lower()}_1d_{start_date.isoformat()}_{end_date.isoformat()}_{timestamp}.json"
    )
    payload = _fetch_yahoo_chart(safe_symbol, start_date, end_date)
    raw_path.write_text(json.dumps(payload, indent=2))

    df = _chart_to_frame(payload, safe_symbol, raw_path, ingested_at)
    normalized = _normalized_frame(df)
    features = _feature_frame(df)

    normalized_paths = _write_monthly_parquet(
        normalized,
        data_root / "normalized" / "underlyings" / "daily_ohlcv",
        safe_symbol,
        "daily_ohlcv_yahoo",
    )
    feature_paths = _write_monthly_parquet(
        features,
        data_root / "research" / "underlyings" / "daily_features",
        safe_symbol,
        "daily_features",
    )

    manifest_dir = data_root / "manifests" / "research" / f"{timestamp[:4]}" / f"{timestamp[4:6]}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{run_label}_{timestamp}.json"
    manifest = {
        "run_label": run_label,
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": safe_symbol,
        "source": "yahoo_chart",
        "interval": "1d",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "rows": int(len(df)),
        "raw_path": str(raw_path),
        "normalized_root": str(data_root / "normalized" / "underlyings" / "daily_ohlcv"),
        "feature_root": str(data_root / "research" / "underlyings" / "daily_features"),
        "normalized_partitions": [str(path) for path in normalized_paths],
        "feature_partitions": [str(path) for path in feature_paths],
        "first_trade_date": str(df["trade_date"].min()),
        "last_trade_date": str(df["trade_date"].max()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return PriceHistoryOutputs(raw_path, normalized_paths, feature_paths, manifest_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Yahoo daily OHLCV for underlying regime research.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--data-root")
    parser.add_argument("--run-label", default="spy_10y_daily_yahoo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    end_date = date.fromisoformat(args.end_date) if args.end_date else datetime.now(UTC).date()
    start_date = date.fromisoformat(args.start_date) if args.start_date else _default_start(end_date, args.years)
    if start_date > end_date:
        raise ValueError("--start-date must be on or before --end-date")
    data_root = _resolve_data_root(args.data_root)
    outputs = collect_price_history(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        data_root=data_root,
        run_label=args.run_label,
    )
    logger.info("Wrote raw Yahoo payload to %s", outputs.raw_path)
    logger.info("Wrote %d normalized partitions", len(outputs.normalized_paths))
    logger.info("Wrote %d feature partitions", len(outputs.feature_paths))
    logger.info("Wrote manifest to %s", outputs.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
