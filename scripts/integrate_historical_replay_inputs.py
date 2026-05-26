#!/usr/bin/env python3
"""Integrate and validate historical earnings/OHLC inputs for T9 replay.

This script prepares data needed for faithful VolSnapshot reconstruction.  It
does not run a backtest, compute PnL, or tune selector/scoring logic.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.earnings_vol_snapshot import build_vol_snapshot
from services.structure_scorecard import build_structure_scorecards
from services.structure_selector import select_best_structure
from scripts.collect_yahoo_underlying_history import collect_price_history


DEFAULT_DATA_ROOT = Path("/Volumes/T9/market_data")
DEFAULT_OPTIONS_ROOT = DEFAULT_DATA_ROOT / "research" / "options_features_eod"
DEFAULT_OHLC_ROOT = DEFAULT_DATA_ROOT / "research" / "underlyings" / "daily_features"
DEFAULT_EARNINGS_DB = DEFAULT_DATA_ROOT / "research" / "options_calculator_pro" / "earnings_calendar" / "historical_earnings.sqlite"
DEFAULT_REPORT_DIR = Path("exports/reports/t9_replay_readiness")
DEFAULT_SYMBOLS = ("AAPL", "AMZN", "MSFT", "NVDA")
DEFAULT_START = date(2023, 1, 1)
DEFAULT_END = date(2026, 4, 24)
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_CONFIRMATION_FORMS = {"8-K", "10-Q", "10-K", "6-K", "20-F", "40-F"}


EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS historical_earnings_events (
    symbol TEXT NOT NULL,
    event_date TEXT NOT NULL,
    release_timing TEXT NOT NULL DEFAULT 'UNKNOWN',
    primary_source TEXT NOT NULL,
    confirmed_source TEXT,
    source_confidence REAL NOT NULL DEFAULT 0.0,
    lookahead_safe_for_pre_event INTEGER NOT NULL DEFAULT 0,
    known_as_of TEXT,
    known_as_of_source TEXT,
    source_event_time_utc TEXT,
    provider_record_json TEXT NOT NULL DEFAULT '{}',
    fetched_at TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (symbol, event_date)
)
"""

SOURCES_DDL = """
CREATE TABLE IF NOT EXISTS historical_earnings_event_sources (
    symbol TEXT NOT NULL,
    event_date TEXT NOT NULL,
    source TEXT NOT NULL,
    release_timing TEXT NOT NULL DEFAULT 'UNKNOWN',
    known_as_of TEXT,
    lookahead_safe_for_pre_event INTEGER NOT NULL DEFAULT 0,
    source_confidence REAL NOT NULL DEFAULT 0.0,
    provider_record_json TEXT NOT NULL DEFAULT '{}',
    fetched_at TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (symbol, event_date, source)
)
"""


@dataclass(frozen=True)
class EarningsRecord:
    symbol: str
    event_date: date
    release_timing: str
    source: str
    source_confidence: float
    known_as_of: str | None
    lookahead_safe_for_pre_event: bool
    source_event_time_utc: str | None
    provider_record: dict[str, Any]
    notes: str


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _normalize_timing(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "UNKNOWN"
    if text in {"amc", "after market close", "after-hours", "after hours"} or "after" in text:
        return "AMC"
    if text in {"bmo", "before market open", "pre-market", "pre market"} or "before" in text:
        return "BMO"
    if text in {"dmh", "during market hours", "during market"} or "during" in text:
        return "DMH"
    return "UNKNOWN"


def _infer_timing_from_timestamp(ts: Any) -> tuple[str, str | None]:
    parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(parsed):
        return "UNKNOWN", None
    hour = int(parsed.hour)
    if hour >= 20:
        return "AMC", parsed.isoformat()
    if hour < 14:
        return "BMO", parsed.isoformat()
    return "UNKNOWN", parsed.isoformat()


def _fetch_yfinance_events(symbol: str, start: date, end: date, *, limit: int = 100) -> list[EarningsRecord]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = None
    get_dates = getattr(ticker, "get_earnings_dates", None)
    if callable(get_dates):
        try:
            frame = get_dates(limit=limit)
        except Exception:
            frame = None
    if frame is None:
        frame = getattr(ticker, "earnings_dates", None)
    if frame is None or getattr(frame, "empty", True):
        return []

    fetched_at = datetime.now(UTC).isoformat()
    records: list[EarningsRecord] = []
    for idx, event_ts in enumerate(pd.to_datetime(frame.index, utc=True, errors="coerce")):
        if pd.isna(event_ts):
            continue
        event_date = event_ts.date()
        if not (start <= event_date <= end):
            continue
        row = frame.iloc[idx].to_dict() if idx < len(frame) else {}
        timing = "UNKNOWN"
        for key, value in row.items():
            if "time" in str(key).lower():
                timing = _normalize_timing(value)
                break
        if timing == "UNKNOWN":
            timing, _ = _infer_timing_from_timestamp(event_ts)
        records.append(
            EarningsRecord(
                symbol=symbol,
                event_date=event_date,
                release_timing=timing,
                source="yfinance_historical_earnings_dates",
                source_confidence=0.55 if timing != "UNKNOWN" else 0.45,
                known_as_of=None,
                lookahead_safe_for_pre_event=False,
                source_event_time_utc=event_ts.isoformat(),
                provider_record=_jsonable(row),
                notes="Historical actual earnings date from yfinance; no original calendar-publication timestamp supplied.",
            )
        )
    return _dedupe_records(records)


def _fetch_alpha_vantage_events(symbol: str, start: date, end: date, api_key: str) -> list[EarningsRecord]:
    params = urlencode({"function": "EARNINGS", "symbol": symbol, "apikey": api_key})
    req = Request(f"{ALPHA_VANTAGE_BASE_URL}?{params}", headers={"User-Agent": "OptionsCalculatorPro/replay-inputs"})
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    fetched_at = datetime.now(UTC).isoformat()
    rows = payload.get("quarterlyEarnings") or []
    records: list[EarningsRecord] = []
    for row in rows:
        event_date = _parse_date(row.get("reportedDate"))
        if event_date is None or not (start <= event_date <= end):
            continue
        records.append(
            EarningsRecord(
                symbol=symbol,
                event_date=event_date,
                release_timing="UNKNOWN",
                source="alpha_vantage_earnings_reported_date",
                source_confidence=0.60,
                known_as_of=None,
                lookahead_safe_for_pre_event=False,
                source_event_time_utc=None,
                provider_record=_jsonable(row),
                notes="Alpha Vantage historical EARNINGS reportedDate; no announcement timing or original known-as-of timestamp supplied.",
            )
        )
    return _dedupe_records(records)


def _fetch_json(url: str, *, user_agent: str) -> Any:
    req = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    with urlopen(req, timeout=30) as resp:
        body = resp.read()
        if body.startswith(b"\x1f\x8b"):
            body = gzip.decompress(body)
        return json.loads(body.decode("utf-8"))


def _lookup_sec_cik(symbol: str, *, user_agent: str) -> str | None:
    payload = _fetch_json(SEC_COMPANY_TICKERS_URL, user_agent=user_agent)
    if not isinstance(payload, dict):
        return None
    target = symbol.upper()
    for item in payload.values():
        if str(item.get("ticker", "")).upper() != target:
            continue
        cik = item.get("cik_str")
        if cik is None:
            return None
        return f"{int(cik):010d}"
    return None


def _sec_confirmation_records(
    symbol: str,
    seed_events: list[EarningsRecord],
    *,
    user_agent: str,
) -> list[EarningsRecord]:
    if not seed_events:
        return []
    cik = _lookup_sec_cik(symbol, user_agent=user_agent)
    if not cik:
        return []
    payload = _fetch_json(SEC_SUBMISSIONS_URL.format(cik=cik), user_agent=user_agent)
    recent = (payload or {}).get("filings", {}).get("recent", {})
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    acceptance_times = recent.get("acceptanceDateTime") or []
    accession_numbers = recent.get("accessionNumber") or []
    primary_documents = recent.get("primaryDocument") or []
    rows: list[dict[str, Any]] = []
    for idx, form in enumerate(forms):
        if str(form).upper() not in SEC_CONFIRMATION_FORMS:
            continue
        filing_date = _parse_date(filing_dates[idx] if idx < len(filing_dates) else None)
        if filing_date is None:
            continue
        rows.append(
            {
                "form": str(form).upper(),
                "filing_date": filing_date,
                "acceptanceDateTime": acceptance_times[idx] if idx < len(acceptance_times) else None,
                "accessionNumber": accession_numbers[idx] if idx < len(accession_numbers) else None,
                "primaryDocument": primary_documents[idx] if idx < len(primary_documents) else None,
                "cik": cik,
            }
        )

    confirmations: list[EarningsRecord] = []
    for event in seed_events:
        candidates = [
            row for row in rows
            if abs((row["filing_date"] - event.event_date).days) <= 1
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (abs((row["filing_date"] - event.event_date).days), row["filing_date"]))
        selected = candidates[0]
        timing, source_time = _infer_timing_from_timestamp(selected.get("acceptanceDateTime"))
        confirmations.append(
            EarningsRecord(
                symbol=symbol,
                event_date=event.event_date,
                release_timing=timing if timing != "UNKNOWN" else event.release_timing,
                source="sec_edgar_filing_confirmation",
                source_confidence=0.50,
                known_as_of=source_time or selected["filing_date"].isoformat(),
                lookahead_safe_for_pre_event=False,
                source_event_time_utc=source_time,
                provider_record=_jsonable(selected),
                notes=(
                    "SEC filing matched within +/-1 calendar day of an external earnings date. "
                    "Confirmation-only; filing timestamps are not pre-event calendar availability."
                ),
            )
        )
    return confirmations


def _dedupe_records(records: Iterable[EarningsRecord]) -> list[EarningsRecord]:
    best: dict[tuple[str, date, str], EarningsRecord] = {}
    for record in records:
        key = (record.symbol.upper(), record.event_date, record.source)
        best[key] = record
    return sorted(best.values(), key=lambda item: (item.symbol, item.event_date, item.source))


def _ensure_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(EVENTS_DDL)
        conn.execute(SOURCES_DDL)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hee_symbol_date ON historical_earnings_events(symbol, event_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hees_symbol_date ON historical_earnings_event_sources(symbol, event_date)")
        conn.commit()


def _persist_records(path: Path, records: list[EarningsRecord]) -> None:
    if not records:
        return
    _ensure_db(path)
    fetched_at = datetime.now(UTC).isoformat()
    with sqlite3.connect(str(path)) as conn:
        for record in records:
            conn.execute(
                """
                INSERT OR REPLACE INTO historical_earnings_event_sources (
                    symbol, event_date, source, release_timing, known_as_of,
                    lookahead_safe_for_pre_event, source_confidence,
                    provider_record_json, fetched_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.symbol.upper(),
                    record.event_date.isoformat(),
                    record.source,
                    record.release_timing,
                    record.known_as_of,
                    1 if record.lookahead_safe_for_pre_event else 0,
                    float(record.source_confidence),
                    json.dumps(record.provider_record, sort_keys=True),
                    fetched_at,
                    record.notes,
                ),
            )
        rows = conn.execute(
            """
            SELECT symbol, event_date, source, release_timing, known_as_of,
                   lookahead_safe_for_pre_event, source_confidence,
                   provider_record_json, notes
            FROM historical_earnings_event_sources
            """
        ).fetchall()
        grouped: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
        for row in rows:
            grouped.setdefault((str(row[0]), str(row[1])), []).append(row)
        for (symbol, event_date), items in grouped.items():
            items.sort(key=lambda row: (int(row[5]), float(row[6])), reverse=True)
            primary = items[0]
            confirmed_sources = [str(row[2]) for row in items[1:]]
            known_as_of = primary[4]
            known_as_of_source = primary[2] if known_as_of else None
            if known_as_of is None:
                confirmation_time = next((row for row in items[1:] if row[4]), None)
                if confirmation_time is not None:
                    known_as_of = confirmation_time[4]
                    known_as_of_source = confirmation_time[2]
            lookahead_safe = int(primary[5])
            notes = "; ".join(sorted({str(row[8] or "") for row in items if row[8]}))
            conn.execute(
                """
                INSERT OR REPLACE INTO historical_earnings_events (
                    symbol, event_date, release_timing, primary_source,
                    confirmed_source, source_confidence, lookahead_safe_for_pre_event,
                    known_as_of, known_as_of_source, source_event_time_utc,
                    provider_record_json, fetched_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    event_date,
                    primary[3],
                    primary[2],
                    ",".join(confirmed_sources) if confirmed_sources else None,
                    float(primary[6]),
                    lookahead_safe,
                    known_as_of,
                    known_as_of_source,
                    None,
                    primary[7],
                    fetched_at,
                    notes,
                ),
            )
        conn.commit()


def _load_events(path: Path, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM historical_earnings_events
            WHERE symbol = ?
              AND event_date >= ?
              AND event_date <= ?
            ORDER BY event_date
            """,
            (symbol.upper(), start.isoformat(), end.isoformat()),
        ).fetchall()
    return [dict(row) for row in rows]


def _option_dates(con: duckdb.DuckDBPyConnection, options_root: Path, symbol: str) -> list[date]:
    source = options_root / f"underlying_symbol={symbol}" / "**" / "*.parquet"
    if not (options_root / f"underlying_symbol={symbol}").exists():
        return []
    rows = con.execute(
        f"SELECT DISTINCT trade_date FROM read_parquet('{source}', hive_partitioning=true) ORDER BY trade_date"
    ).fetchall()
    # The parquet schema stores trade_date as TIMESTAMP, which DuckDB returns
    # as datetime.datetime. Coerce to datetime.date so downstream subtraction
    # against earnings event_dates (which are date.fromisoformat outputs)
    # produces a homogeneous timedelta operation. Without this, the comparison
    # raises `TypeError: unsupported operand type(s) for -: 'datetime.date'
    # and 'datetime.datetime'` and the entire earnings-fetch stage aborts.
    return [
        row[0].date() if hasattr(row[0], "date") else row[0]
        for row in rows
    ]


def _load_ohlc(con: duckdb.DuckDBPyConnection, ohlc_root: Path, symbol: str) -> pd.DataFrame:
    source = ohlc_root / f"underlying_symbol={symbol}" / "**" / "*.parquet"
    if not (ohlc_root / f"underlying_symbol={symbol}").exists():
        return pd.DataFrame()
    frame = con.execute(
        f"""
        SELECT trade_date, underlying_symbol, open, high, low, close, volume
        FROM read_parquet('{source}', hive_partitioning=true)
        ORDER BY trade_date
        """
    ).fetchdf()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    return frame.drop_duplicates(["trade_date"], keep="last")


def _load_chain_for_day(con: duckdb.DuckDBPyConnection, options_root: Path, symbol: str, as_of: date) -> pd.DataFrame:
    source = options_root / f"underlying_symbol={symbol}" / "**" / "*.parquet"
    frame = con.execute(
        f"""
        SELECT trade_date, expiry, call_put, strike, bid, ask, mid, iv,
               open_interest, volume, underlying_price, spread_pct
        FROM read_parquet('{source}', hive_partitioning=true)
        WHERE trade_date = DATE '{as_of.isoformat()}'
        ORDER BY expiry, call_put, strike
        """
    ).fetchdf()
    return frame


def _validate_ohlc(
    con: duckdb.DuckDBPyConnection,
    *,
    options_root: Path,
    ohlc_root: Path,
    symbols: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for symbol in symbols:
        opt_dates = _option_dates(con, options_root, symbol)
        ohlc = _load_ohlc(con, ohlc_root, symbol)
        ohlc_dates = {ts.date() for ts in pd.to_datetime(ohlc["trade_date"]).tolist()} if not ohlc.empty else set()
        opt_set = set(opt_dates)
        missing = sorted(opt_set - ohlc_dates)
        extra = sorted(ohlc_dates - opt_set)
        duplicate_count = 0
        outlier_count = 0
        if not ohlc.empty:
            duplicate_count = int(ohlc["trade_date"].duplicated().sum())
            close = pd.to_numeric(ohlc["close"], errors="coerce")
            returns = close.pct_change().abs()
            outlier_count = int((returns > 0.35).sum())
        result[symbol] = {
            "option_days": len(opt_dates),
            "ohlc_days": len(ohlc_dates),
            "full_ohlc_day_pct": round((1.0 - len(missing) / len(opt_dates)) * 100.0, 3) if opt_dates else None,
            "missing_ohlc_days": [item.isoformat() for item in missing[:20]],
            "missing_ohlc_day_count": len(missing),
            "extra_ohlc_day_count": len(extra),
            "duplicate_ohlc_timestamps": duplicate_count,
            "extreme_close_to_close_jump_count": outlier_count,
        }
    return result


def _validate_earnings(
    *,
    db_path: Path,
    options_root: Path,
    symbols: tuple[str, ...],
    start: date,
    end: date,
) -> dict[str, Any]:
    con = duckdb.connect(":memory:")
    result: dict[str, Any] = {}
    for symbol in symbols:
        events = _load_events(db_path, symbol, start, end)
        opt_dates = _option_dates(con, options_root, symbol)
        safe_events = [row for row in events if int(row.get("lookahead_safe_for_pre_event") or 0) == 1]
        actual_event_dates = {date.fromisoformat(row["event_date"]) for row in events}
        pre_event_option_days = 0
        for opt_date in opt_dates:
            if any(timedelta(days=1) <= (event_date - opt_date) <= timedelta(days=14) for event_date in actual_event_dates):
                pre_event_option_days += 1
        result[symbol] = {
            "event_count": len(events),
            "lookahead_safe_event_count": len(safe_events),
            "known_as_of_missing_count": sum(1 for row in events if not row.get("known_as_of")),
            "timing_counts": _counts(row.get("release_timing") or "UNKNOWN" for row in events),
            "pre_event_option_days_identified_posthoc": pre_event_option_days,
            "valid_no_lookahead_earnings_mapping_pct": 0.0 if opt_dates else None,
            "events": [
                {
                    "event_date": row["event_date"],
                    "release_timing": row["release_timing"],
                    "primary_source": row["primary_source"],
                    "confirmed_source": row["confirmed_source"],
                    "lookahead_safe_for_pre_event": bool(row["lookahead_safe_for_pre_event"]),
                    "known_as_of": row["known_as_of"],
                }
                for row in events
            ],
        }
    return result


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value or "UNKNOWN")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _prior_events(events: list[dict[str, Any]], as_of: date) -> list[dict[str, Any]]:
    prior = []
    for row in events:
        event_date = date.fromisoformat(row["event_date"])
        if event_date <= as_of:
            prior.append({"event_date": event_date.isoformat(), "release_timing": row.get("release_timing") or "UNKNOWN"})
    return prior


def _snapshot_reconstruction_check(
    con: duckdb.DuckDBPyConnection,
    *,
    options_root: Path,
    ohlc_root: Path,
    earnings_db: Path,
    symbols: tuple[str, ...],
    start: date,
    end: date,
    max_samples: int,
    allow_posthoc_earnings_smoke: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for symbol in symbols:
        events = _load_events(earnings_db, symbol, start, end)
        ohlc = _load_ohlc(con, ohlc_root, symbol)
        attempted = valid = no_trade = failed = 0
        missing_reasons: dict[str, int] = {}
        samples: list[dict[str, Any]] = []
        if ohlc.empty or not events:
            results[symbol] = {
                "attempted": 0,
                "valid": 0,
                "failed": 0,
                "status": "blocked_missing_ohlc_or_earnings",
                "missing_reasons": {"missing_ohlc_or_earnings": 1},
            }
            continue
        for event in events:
            if not allow_posthoc_earnings_smoke and not bool(event.get("lookahead_safe_for_pre_event")):
                continue
            event_date = date.fromisoformat(event["event_date"])
            candidate_dates = [
                event_date - timedelta(days=offset)
                for offset in (10, 7, 5, 3, 1)
                if start <= event_date - timedelta(days=offset) <= end
            ]
            for as_of in candidate_dates:
                if attempted >= max_samples:
                    break
                chain = _load_chain_for_day(con, options_root, symbol, as_of)
                if chain.empty:
                    continue
                price_frame = ohlc[ohlc["trade_date"] <= pd.Timestamp(as_of)].copy()
                if len(price_frame) < 80:
                    continue
                attempted += 1
                try:
                    metadata = {
                        "earnings_date": event["event_date"],
                        "release_timing": event.get("release_timing") or "UNKNOWN",
                        "prior_events": _prior_events(events, as_of),
                        "earnings_source_primary": event.get("primary_source"),
                        "earnings_source_confirmed": event.get("confirmed_source"),
                        "earnings_source_confidence": float(event.get("source_confidence") or 0.0),
                        "earnings_source_notes": [
                            "Historical earnings input is post-hoc unless lookahead_safe_for_pre_event=true."
                        ],
                    }
                    snapshot = build_vol_snapshot(
                        symbol,
                        as_of,
                        option_chain_data=chain,
                        price_data=price_frame,
                        earnings_metadata=metadata,
                    )
                    cards = build_structure_scorecards(snapshot)
                    selected = select_best_structure(snapshot, cards)
                    required = {
                        "rv30_yang_zhang": snapshot.rv30_yang_zhang,
                        "rv_har_forecast": snapshot.rv_har_forecast,
                        "iv30": snapshot.iv30,
                        "iv45": snapshot.iv45,
                        "near_term_implied_move_pct": snapshot.near_term_implied_move_pct,
                        "timing_score": snapshot.timing_score,
                    }
                    missing = [key for key, value in required.items() if value is None]
                    if missing:
                        for key in missing:
                            missing_reasons[key] = missing_reasons.get(key, 0) + 1
                        failed += 1
                    else:
                        valid += 1
                        if selected.recommendation == "No Trade":
                            no_trade += 1
                    samples.append(
                        {
                            "as_of": as_of.isoformat(),
                            "earnings_date": event["event_date"],
                            "days_to_earnings": snapshot.days_to_earnings,
                            "data_quality": snapshot.data_quality,
                            "historical_move_source": snapshot.historical_move_source,
                            "recommendation": selected.recommendation,
                            "missing_required_inputs": missing,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - audit records partial-data failures.
                    failed += 1
                    key = type(exc).__name__
                    missing_reasons[key] = missing_reasons.get(key, 0) + 1
            if attempted >= max_samples:
                break
        results[symbol] = {
            "attempted": attempted,
            "valid": valid,
            "failed": failed,
            "valid_pct": round(valid / attempted * 100.0, 3) if attempted else None,
            "no_trade": no_trade,
            "status": "posthoc_smoke_only" if allow_posthoc_earnings_smoke else "no_lookahead_only",
            "missing_reasons": missing_reasons,
            "samples": samples,
        }
    return results


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    _load_dotenv(REPO_ROOT / ".env")
    symbols = tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip())
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    db_path = Path(args.earnings_db)
    _ensure_db(db_path)

    ohlc_collection: dict[str, Any] = {}
    if args.collect_ohlc:
        data_root = Path(args.data_root)
        for symbol in symbols:
            try:
                outputs = collect_price_history(
                    symbol=symbol,
                    start_date=start,
                    end_date=end,
                    data_root=data_root,
                    run_label=f"h15_{symbol.lower()}_ohlcv",
                )
                ohlc_collection[symbol] = {
                    "raw_path": str(outputs.raw_path),
                    "normalized_partitions": len(outputs.normalized_paths),
                    "feature_partitions": len(outputs.feature_paths),
                    "manifest_path": str(outputs.manifest_path),
                }
            except Exception as exc:  # noqa: BLE001 - integration report should preserve failures.
                ohlc_collection[symbol] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }

    fetched_records: list[EarningsRecord] = []
    fetch_errors: dict[str, list[str]] = {}
    if args.fetch_earnings:
        alpha_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
        sec_user_agent = os.environ.get("SEC_API_USER_AGENT", "").strip()
        for symbol in symbols:
            symbol_records: list[EarningsRecord] = []
            try:
                symbol_records.extend(_fetch_yfinance_events(symbol, start, end, limit=args.yfinance_limit))
            except Exception as exc:  # noqa: BLE001
                fetch_errors.setdefault(symbol, []).append(f"yfinance: {type(exc).__name__}: {exc}")
            if alpha_key:
                time.sleep(args.provider_sleep_sec)
                try:
                    symbol_records.extend(_fetch_alpha_vantage_events(symbol, start, end, alpha_key))
                except Exception as exc:  # noqa: BLE001
                    fetch_errors.setdefault(symbol, []).append(f"alpha_vantage: {type(exc).__name__}: {exc}")
            if sec_user_agent:
                time.sleep(args.provider_sleep_sec)
                try:
                    symbol_records.extend(_sec_confirmation_records(symbol, symbol_records, user_agent=sec_user_agent))
                except Exception as exc:  # noqa: BLE001
                    fetch_errors.setdefault(symbol, []).append(f"sec_edgar: {type(exc).__name__}: {exc}")
            fetched_records.extend(symbol_records)
            time.sleep(args.provider_sleep_sec)
        _persist_records(db_path, fetched_records)

    con = duckdb.connect(":memory:")
    ohlc = _validate_ohlc(
        con,
        options_root=Path(args.options_root),
        ohlc_root=Path(args.ohlc_root),
        symbols=symbols,
    )
    earnings = _validate_earnings(
        db_path=db_path,
        options_root=Path(args.options_root),
        symbols=symbols,
        start=start,
        end=end,
    )
    snapshots = _snapshot_reconstruction_check(
        con,
        options_root=Path(args.options_root),
        ohlc_root=Path(args.ohlc_root),
        earnings_db=db_path,
        symbols=tuple(item for item in ("AAPL", "MSFT") if item in symbols),
        start=start,
        end=end,
        max_samples=args.max_snapshot_samples,
        allow_posthoc_earnings_smoke=args.allow_posthoc_earnings_smoke,
    )

    ohlc_ready = all((row.get("full_ohlc_day_pct") or 0.0) >= 99.0 for row in ohlc.values())
    no_lookahead_ready = all((row.get("lookahead_safe_event_count") or 0) > 0 for row in earnings.values())
    snapshot_ready = all((row.get("valid") or 0) > 0 and row.get("failed") == 0 for row in snapshots.values()) if snapshots else False
    full_ready = bool(ohlc_ready and no_lookahead_ready and snapshot_ready and not args.allow_posthoc_earnings_smoke)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {
            "symbols": list(symbols),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "options_root": str(args.options_root),
            "ohlc_root": str(args.ohlc_root),
            "earnings_db": str(db_path),
        },
        "earnings_integration": {
            "fetched_records": len(fetched_records),
            "fetch_errors": fetch_errors,
            "database": str(db_path),
            "schema": {
                "historical_earnings_events": [
                    "symbol",
                    "event_date",
                    "release_timing",
                    "primary_source",
                    "confirmed_source",
                    "source_confidence",
                    "lookahead_safe_for_pre_event",
                    "known_as_of",
                    "known_as_of_source",
                    "provider_record_json",
                ],
                "historical_earnings_event_sources": [
                    "symbol",
                    "event_date",
                    "source",
                    "release_timing",
                    "known_as_of",
                    "lookahead_safe_for_pre_event",
                    "source_confidence",
                    "provider_record_json",
                ],
            },
            "lookahead_policy": (
                "Events without an original provider known_as_of timestamp are stored for post-hoc alignment "
                "but are not marked safe for no-lookahead pre-event selection."
            ),
        },
        "ohlc_collection": ohlc_collection,
        "ohlc_coverage": ohlc,
        "earnings_alignment": earnings,
        "volsnapshot_reconstruction": snapshots,
        "go_no_go": {
            "full_replay_ready": full_ready,
            "decision": "GO" if full_ready else "NO-GO",
            "reason": (
                "Full no-lookahead replay is ready."
                if full_ready
                else "OHLC may be usable after collection, but earnings records are not no-lookahead safe unless known_as_of provenance exists before the historical entry date."
            ),
        },
        "failure_modes": [
            "Historical actual earnings dates from yfinance/Alpha Vantage are post-hoc unless the source supplies original calendar-publication timestamps.",
            "UNKNOWN announcement timing requires conservative BMO/AMC handling before event-window replay.",
            "OHLC and option trade dates must match exactly; missing OHLC dates block faithful RV/Yang-Zhang/HAR reconstruction.",
        ],
    }
    return _jsonable(report)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# H1.5 Earnings/OHLC Replay Input Integration",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Decision",
        "",
        f"**Full replay readiness:** {report['go_no_go']['decision']}",
        "",
        report["go_no_go"]["reason"],
        "",
        "## OHLC Coverage",
        "",
        "| Symbol | Option days | OHLC days | Full OHLC % | Missing OHLC | Duplicates | Extreme jumps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol, row in report["ohlc_coverage"].items():
        lines.append(
            f"| {symbol} | {row['option_days']} | {row['ohlc_days']} | {row['full_ohlc_day_pct']} | "
            f"{row['missing_ohlc_day_count']} | {row['duplicate_ohlc_timestamps']} | {row['extreme_close_to_close_jump_count']} |"
        )
    lines.extend(["", "## Earnings Alignment", ""])
    for symbol, row in report["earnings_alignment"].items():
        lines.append(
            f"- {symbol}: events={row['event_count']}, lookahead_safe={row['lookahead_safe_event_count']}, "
            f"known_as_of_missing={row['known_as_of_missing_count']}, timing={row['timing_counts']}"
        )
    lines.extend(["", "## VolSnapshot Reconstruction", ""])
    for symbol, row in report["volsnapshot_reconstruction"].items():
        lines.append(
            f"- {symbol}: attempted={row['attempted']}, valid={row['valid']}, failed={row['failed']}, "
            f"status={row['status']}, missing={row['missing_reasons']}"
        )
    lines.extend(["", "## Failure Modes", ""])
    for item in report["failure_modes"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrate H1.5 earnings/OHLC inputs for historical replay readiness.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", default=DEFAULT_START.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END.isoformat())
    parser.add_argument("--options-root", type=Path, default=DEFAULT_OPTIONS_ROOT)
    parser.add_argument("--ohlc-root", type=Path, default=DEFAULT_OHLC_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--earnings-db", type=Path, default=DEFAULT_EARNINGS_DB)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--fetch-earnings", action="store_true")
    parser.add_argument("--collect-ohlc", action="store_true")
    parser.add_argument("--yfinance-limit", type=int, default=100)
    parser.add_argument("--provider-sleep-sec", type=float, default=0.25)
    parser.add_argument("--max-snapshot-samples", type=int, default=8)
    parser.add_argument(
        "--allow-posthoc-earnings-smoke",
        action="store_true",
        help="Run VolSnapshot smoke reconstruction with actual historical dates even when not lookahead-safe.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(args)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "h15_replay_inputs_latest.json"
    md_path = args.report_dir / "h15_replay_inputs_latest.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path), "decision": report["go_no_go"]["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
