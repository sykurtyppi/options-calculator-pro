#!/usr/bin/env python3
"""
Week 1 — Earnings IV Crush Label Backfill Pipeline
====================================================

Extracts real pre/post earnings IV snapshot pairs directly from the EOD parquet
options feature store and writes calibrated IV decay labels to SQLite.

No live API calls required. All data is sourced from the local T9 parquet store.
Earnings dates are fetched from yfinance (with SQLite caching; one fetch per symbol).

Design principles enforced throughout:
  - No silent assumptions — every major decision is logged and flagged
  - No future-data leakage — pre_event_date is always strictly < earnings_date
  - No synthetic fills masquerading as real labels
  - Every label row carries full traceability (expiry, DTE, moneyness, fallback flags)
  - Idempotent by default (INSERT OR IGNORE); explicit --overwrite for re-runs
  - Dry-run mode produces identical logging without any DB writes
  - Quality scoring is deterministic and explainable per label row

Earnings date source decision:
  yfinance is used (not MarketData.app) because:
    1. MDApp requires credentials that may not be present in all environments
    2. yfinance provides ~3-4 years of historical earnings dates for all S&P 500 names
    3. Release timing (BMO/AMC) is inferred via the same heuristic already in the codebase
  Dates are cached in the `earnings_events` SQLite table on first fetch to avoid
  redundant network calls on reruns.

Pre-event date selection:
  Conservative rule: last trade_date in parquet strictly < earnings_date, regardless
  of BMO/AMC timing. This avoids any same-day leakage for UNKNOWN-timed events.
  For BMO events, the same-day snapshot *would* be valid (pre-open), but we cannot
  verify from yfinance timing with sufficient confidence. Safety > marginal data gain.

Front-leg DTE target: [2, 14] on pre_event_date (captures event-bearing near-term IV)
  Fallback: [15, 21] — acceptable with quality downgrade
  Rationale: this expiry expires after the earnings catalyst, so its IV embeds event
  uncertainty that collapses post-event.

Back-leg DTE target: [20, 45] on pre_event_date, prefer closest to DTE=30
  Fallback: [15, 19] — acceptable with quality downgrade
  Rationale: the "back month" hedge leg in a calendar spread; maintains long-vega
  exposure that does not collapse as sharply post-event.

Post-event expiry matching:
  Primary: same expiry as selected pre-event legs (exact match)
  Fallback: nearest available expiry with the same relative DTE ± 7 days
  Consequence: non-exact-match labels receive a quality penalty and exact_expiry_match=0

Label version: LABEL_VERSION constant below — increment when selection logic changes.

Usage:
    # Dry run on 5 symbols (inspect before writing)
    python scripts/build_earnings_iv_labels.py \\
        --symbols AAPL,MSFT,NVDA,AMZN,META --dry-run

    # Full backfill, all symbols in parquet store
    python scripts/build_earnings_iv_labels.py

    # Restricted date range
    python scripts/build_earnings_iv_labels.py \\
        --start-date 2023-01-01 --end-date 2024-12-31

    # Re-run with overwrite (replaces existing labels for same symbol+event_date)
    python scripts/build_earnings_iv_labels.py --overwrite

    # Specific symbols with verbose output
    python scripts/build_earnings_iv_labels.py \\
        --symbols AAPL,TSLA --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

# ─── Constants ────────────────────────────────────────────────────────────────

# Increment this string whenever the label selection logic changes materially.
# Stored in every label row for auditability.
LABEL_VERSION: str = "parquet_v1"

# Default parquet store root (resolved via env var MARKET_DATA_ROOT or T9 path)
DEFAULT_PARQUET_ROOT: Path = Path("/Volumes/T9/market_data/research/options_features_eod")

# Default SQLite DB path (same as InstitutionalMLDatabase)
DEFAULT_DB_PATH: Path = Path.home() / ".options_calculator_pro" / "institutional_ml.db"

# Front-leg DTE targeting (the near-term expiry that carries event IV)
FRONT_DTE_PRIMARY_MIN: int = 2    # inclusive lower bound of target range
FRONT_DTE_PRIMARY_MAX: int = 14   # inclusive upper bound of target range
FRONT_DTE_FALLBACK_MIN: int = 15  # fallback lower bound
FRONT_DTE_FALLBACK_MAX: int = 21  # fallback upper bound

# Back-leg DTE targeting (the hedge/spread back month)
BACK_DTE_PRIMARY_MIN: int = 20    # inclusive lower bound
BACK_DTE_PRIMARY_MAX: int = 45    # inclusive upper bound
BACK_DTE_TARGET_IDEAL: int = 30   # preferred DTE within primary range
BACK_DTE_FALLBACK_MIN: int = 15   # fallback lower bound
BACK_DTE_FALLBACK_MAX: int = 19   # fallback upper bound

# Post-event date: number of trading days to look forward for post snapshot
POST_MAX_DAYS_FORWARD: int = 5    # search up to 5 calendar days post-event

# Expiry fallback tolerance: post expiry may differ from pre by at most this many days
EXPIRY_FALLBACK_TOLERANCE_DAYS: int = 7

# IV validity bounds (filter out clearly bad or unreliable IV values)
IV_MIN: float = 0.01
IV_MAX: float = 5.00

# ATM moneyness ceiling (contracts further than this from spot are excluded)
ATM_MONEYNESS_MAX_PCT: float = 5.0

# yfinance fetch: seconds to sleep between symbol fetches to avoid rate limiting
YFINANCE_SLEEP_SEC: float = 0.5

# Earnings dates: how far back and forward to fetch
EARNINGS_LOOKBACK_YEARS: int = 4
EARNINGS_LOOKAHEAD_DAYS: int = 90

# Symbols to exclude from earnings processing (index products, ETFs, etc.)
NON_EARNINGS_SYMBOLS: frozenset[str] = frozenset({"SPY", "QQQ", "IWM", "SPX", "VIX"})

# ─── Quality scoring parameters ───────────────────────────────────────────────

# Quality tiers based on final quality_score
QUALITY_TIER_A_THRESHOLD: float = 0.80
QUALITY_TIER_B_THRESHOLD: float = 0.60
QUALITY_TIER_C_THRESHOLD: float = 0.40
# Below C_THRESHOLD → tier D (kept, but excluded from training by default)

# Penalty magnitudes (applied to base score of 1.0)
QUALITY_PENALTY_FRONT_DTE_FALLBACK: float = 0.15   # front leg used fallback DTE range
QUALITY_PENALTY_BACK_DTE_FALLBACK: float = 0.10    # back leg used fallback DTE range
QUALITY_PENALTY_NO_EXPIRY_MATCH: float = 0.25      # post expiry differs from pre
QUALITY_PENALTY_UNKNOWN_TIMING: float = 0.05       # release_timing is UNKNOWN
QUALITY_PENALTY_HIGH_MONEYNESS_FRONT: float = 0.05  # front ATM >2% moneyness
QUALITY_PENALTY_HIGH_MONEYNESS_BACK: float = 0.05   # back ATM >2% moneyness
QUALITY_PENALTY_PER_EXTRA_FRONT_DTE: float = 0.03  # each day front DTE exceeds target
QUALITY_PENALTY_PER_BACK_DTE_MISS: float = 0.02   # each 5d back DTE deviates from 30

# ─── Logging setup ────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger = logging.getLogger("build_earnings_iv_labels")
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class BackfillConfig:
    """All configurable parameters for a single backfill run."""
    parquet_root: Path
    db_path: Path
    symbols: Optional[List[str]]          # None = all symbols in parquet store
    start_date: Optional[date]
    end_date: Optional[date]
    dry_run: bool
    overwrite: bool
    verbose: bool
    label_version: str = LABEL_VERSION


@dataclass
class EarningsEvent:
    """A single known earnings event for a symbol."""
    symbol: str
    event_date: date
    release_timing: str                   # 'AMC', 'BMO', 'UNKNOWN'
    source: str                           # 'yfinance', 'cached'


@dataclass
class IVLeg:
    """Near-ATM IV snapshot for a single expiry on a single trade_date."""
    trade_date: date
    expiry: date
    dte: int
    atm_iv: float
    underlying_price: float
    abs_moneyness_pct: float
    open_interest: int
    liquidity_score: float
    dte_range: str                        # 'primary_front', 'fallback_front', 'primary_back', 'fallback_back'


@dataclass
class LabelRow:
    """A fully assembled label row ready for SQLite insertion."""
    symbol: str
    event_date: str                       # ISO date string YYYY-MM-DD
    release_timing: str
    pre_capture_date: str
    post_capture_date: str
    pre_front_iv: float
    post_front_iv: float
    pre_back_iv: float
    post_back_iv: float
    front_iv_crush_pct: float
    back_iv_crush_pct: float
    term_ratio_change: float
    underlying_move_pct: float
    quality_score: float
    source: str
    # Extended traceability
    front_expiry: str
    back_expiry: str
    front_dte_pre: int
    back_dte_pre: int
    front_dte_post: int
    back_dte_post: int
    exact_expiry_match: int               # 0 or 1
    fallback_used: int                    # 0 or 1
    fallback_reason: str
    pre_front_atm_moneyness: float
    pre_back_atm_moneyness: float
    pre_front_oi: int
    pre_back_oi: int
    quality_tier: str                     # 'A', 'B', 'C', 'D'
    label_version: str
    parquet_store_root: str
    pre_underlying_price: float
    post_underlying_price: float


@dataclass
class BackfillStats:
    """Aggregated statistics for a complete backfill run."""
    symbols_attempted: int = 0
    symbols_with_earnings: int = 0
    symbols_with_parquet: int = 0
    events_attempted: int = 0
    events_labeled: int = 0
    events_failed_no_pre: int = 0
    events_failed_no_front: int = 0
    events_failed_no_back: int = 0
    events_failed_no_post: int = 0
    events_failed_iv_invalid: int = 0
    exact_expiry_matches: int = 0
    fallback_expiry_matches: int = 0
    tier_a: int = 0
    tier_b: int = 0
    tier_c: int = 0
    tier_d: int = 0
    front_dtes_primary: int = 0
    front_dtes_fallback: int = 0
    back_dtes_primary: int = 0
    back_dtes_fallback: int = 0
    front_crush_pcts: List[float] = field(default_factory=list)
    back_crush_pcts: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    outlier_rows: List[Dict[str, Any]] = field(default_factory=list)


# ─── Schema management ────────────────────────────────────────────────────────

_EARNINGS_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS earnings_events (
    symbol TEXT NOT NULL,
    event_date TEXT NOT NULL,
    source TEXT NOT NULL,
    release_timing TEXT NOT NULL DEFAULT 'UNKNOWN',
    PRIMARY KEY (symbol, event_date)
)
"""

_EARNINGS_LABELS_DDL = """
CREATE TABLE IF NOT EXISTS earnings_iv_decay_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    event_date TEXT NOT NULL,
    release_timing TEXT NOT NULL DEFAULT 'UNKNOWN',
    pre_capture_date TEXT NOT NULL,
    post_capture_date TEXT NOT NULL,
    pre_front_iv REAL NOT NULL,
    post_front_iv REAL NOT NULL,
    pre_back_iv REAL NOT NULL,
    post_back_iv REAL NOT NULL,
    front_iv_crush_pct REAL NOT NULL,
    back_iv_crush_pct REAL NOT NULL,
    term_ratio_change REAL NOT NULL,
    underlying_move_pct REAL NOT NULL,
    quality_score REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'parquet_backfill_v1',
    front_expiry TEXT,
    back_expiry TEXT,
    front_dte_pre INTEGER,
    back_dte_pre INTEGER,
    front_dte_post INTEGER,
    back_dte_post INTEGER,
    exact_expiry_match INTEGER DEFAULT 1,
    fallback_used INTEGER DEFAULT 0,
    fallback_reason TEXT,
    pre_front_atm_moneyness REAL,
    pre_back_atm_moneyness REAL,
    pre_front_oi INTEGER,
    pre_back_oi INTEGER,
    quality_tier TEXT,
    label_version TEXT DEFAULT 'parquet_v1',
    parquet_store_root TEXT,
    pre_underlying_price REAL,
    post_underlying_price REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, event_date)
)
"""

# Extended columns that the existing InstitutionalMLDatabase schema may not have.
# Each tuple: (column_name, column_definition)
_EXTENDED_COLUMNS: List[Tuple[str, str]] = [
    ("front_expiry",             "TEXT"),
    ("back_expiry",              "TEXT"),
    ("front_dte_pre",            "INTEGER"),
    ("back_dte_pre",             "INTEGER"),
    ("front_dte_post",           "INTEGER"),
    ("back_dte_post",            "INTEGER"),
    ("exact_expiry_match",       "INTEGER DEFAULT 1"),
    ("fallback_used",            "INTEGER DEFAULT 0"),
    ("fallback_reason",          "TEXT"),
    ("pre_front_atm_moneyness",  "REAL"),
    ("pre_back_atm_moneyness",   "REAL"),
    ("pre_front_oi",             "INTEGER"),
    ("pre_back_oi",              "INTEGER"),
    ("quality_tier",             "TEXT"),
    ("label_version",            "TEXT DEFAULT 'parquet_v1'"),
    ("parquet_store_root",       "TEXT"),
    ("pre_underlying_price",     "REAL"),
    ("post_underlying_price",    "REAL"),
]


def _ensure_earnings_cache(db_path: Path) -> None:
    """Create the earnings_events cache table if it does not already exist.

    Safe to call unconditionally — used in dry-run mode so that yfinance results
    can still be cached without touching the labels table.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_EARNINGS_EVENTS_DDL)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ee_symbol_date "
            "ON earnings_events (symbol, event_date)"
        )


def _ensure_schema(db_path: Path, logger: logging.Logger) -> None:
    """
    Create or migrate the earnings_events and earnings_iv_decay_labels tables.

    Safe to call on an existing DB — uses CREATE TABLE IF NOT EXISTS and
    ALTER TABLE ADD COLUMN (column-existence checked before each ALTER).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(_EARNINGS_EVENTS_DDL)
        conn.execute(_EARNINGS_LABELS_DDL)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eidl_symbol_date "
            "ON earnings_iv_decay_labels (symbol, event_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ee_symbol_date "
            "ON earnings_events (symbol, event_date)"
        )

        # Add extended columns if missing (handles existing DB with base schema)
        existing = {row[1] for row in conn.execute(
            "PRAGMA table_info(earnings_iv_decay_labels)"
        ).fetchall()}
        added: List[str] = []
        for col_name, col_def in _EXTENDED_COLUMNS:
            if col_name not in existing:
                conn.execute(
                    f"ALTER TABLE earnings_iv_decay_labels ADD COLUMN {col_name} {col_def}"
                )
                added.append(col_name)
        conn.commit()

    if added:
        logger.info("Schema migrated — added %d extended column(s): %s", len(added), added)
    else:
        logger.debug("Schema up-to-date — no migrations needed")


# ─── Earnings date fetching ────────────────────────────────────────────────────

def _infer_release_timing(event_dt: datetime, row: Optional[pd.Series]) -> str:
    """
    Infer release timing from yfinance row metadata and timestamp hour.
    Returns 'AMC', 'BMO', or 'UNKNOWN'.

    This mirrors the heuristic in InstitutionalMLDatabase._infer_release_timing_from_earnings_row.
    """
    if row is not None:
        for key, value in row.items():
            key_text = str(key).lower()
            if "time" in key_text or "hour" in key_text:
                text = str(value or "").strip().upper()
                if "AMC" in text or "AFTER" in text or text in {"POST", "PM"}:
                    return "AMC"
                if "BMO" in text or "BEFORE" in text or text in {"PRE", "AM"}:
                    return "BMO"

    hour = int(event_dt.hour)
    if hour >= 15:
        return "AMC"
    if 0 <= hour <= 10:
        return "BMO"
    return "UNKNOWN"


def _load_cached_earnings(
    db_path: Path,
    symbol: str,
    start_date: date,
    end_date: date,
) -> List[EarningsEvent]:
    """Read earnings dates for symbol from the SQLite cache."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT event_date, source, release_timing
                FROM earnings_events
                WHERE symbol = ?
                  AND event_date >= ?
                  AND event_date <= ?
                ORDER BY event_date
                """,
                (symbol, start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
    except sqlite3.Error:
        return []

    events: List[EarningsEvent] = []
    for event_date_str, source, timing in rows:
        try:
            events.append(EarningsEvent(
                symbol=symbol,
                event_date=date.fromisoformat(event_date_str),
                release_timing=str(timing or "UNKNOWN").upper(),
                source="cached",
            ))
        except ValueError:
            continue
    return events


def _save_earnings_to_cache(db_path: Path, events: List[EarningsEvent]) -> None:
    """Persist fetched earnings dates to SQLite cache (INSERT OR IGNORE — never overwrites)."""
    if not events:
        return
    with sqlite3.connect(str(db_path)) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO earnings_events (symbol, event_date, source, release_timing)
            VALUES (?, ?, ?, ?)
            """,
            [
                (ev.symbol, ev.event_date.isoformat(), ev.source, ev.release_timing)
                for ev in events
            ],
        )
        conn.commit()


def _fetch_earnings_from_yfinance(
    symbol: str,
    start_date: date,
    end_date: date,
    logger: logging.Logger,
) -> List[EarningsEvent]:
    """
    Fetch earnings dates from yfinance for the given symbol and date window.

    Returns an empty list if yfinance has no data or raises an exception.
    Release timing is inferred via heuristic — marked 'UNKNOWN' when uncertain.
    """
    try:
        ticker = yf.Ticker(symbol)
        earnings_df = None

        get_dates = getattr(ticker, "get_earnings_dates", None)
        if callable(get_dates):
            try:
                earnings_df = get_dates(limit=60)
            except Exception:
                earnings_df = None

        if earnings_df is None:
            earnings_df = getattr(ticker, "earnings_dates", None)

        if earnings_df is None or (hasattr(earnings_df, "empty") and earnings_df.empty):
            return []

    except Exception as exc:
        logger.debug("yfinance fetch failed for %s: %s", symbol, exc)
        return []

    events: List[EarningsEvent] = []
    for idx, ts in enumerate(pd.to_datetime(earnings_df.index)):
        ts_obj = pd.Timestamp(ts)
        if ts_obj.tzinfo is not None:
            ts_obj = ts_obj.tz_convert(None)
        dt = ts_obj.to_pydatetime()
        # Normalise to midnight for date comparison
        event_date = dt.replace(hour=0, minute=0, second=0, microsecond=0).date()
        if not (start_date <= event_date <= end_date):
            continue
        row = earnings_df.iloc[idx] if idx < len(earnings_df) else None
        timing = _infer_release_timing(dt, row)
        events.append(EarningsEvent(
            symbol=symbol,
            event_date=event_date,
            release_timing=timing,
            source="yfinance",
        ))

    # Deduplicate: keep first seen per event_date
    seen: dict[date, EarningsEvent] = {}
    for ev in events:
        if ev.event_date not in seen:
            seen[ev.event_date] = ev
    return sorted(seen.values(), key=lambda e: e.event_date)


def get_earnings_events(
    symbol: str,
    start_date: date,
    end_date: date,
    db_path: Path,
    logger: logging.Logger,
) -> List[EarningsEvent]:
    """
    Return earnings events for a symbol, using the SQLite cache if available,
    otherwise fetching from yfinance and caching results.
    """
    cached = _load_cached_earnings(db_path, symbol, start_date, end_date)
    if cached:
        logger.debug("%s: loaded %d earnings dates from cache", symbol, len(cached))
        return cached

    logger.debug("%s: fetching earnings dates from yfinance ...", symbol)
    time.sleep(YFINANCE_SLEEP_SEC)

    fetched = _fetch_earnings_from_yfinance(symbol, start_date, end_date, logger)
    if fetched:
        _save_earnings_to_cache(db_path, fetched)
        logger.debug("%s: cached %d earnings dates", symbol, len(fetched))
    else:
        logger.debug("%s: no earnings dates found in yfinance", symbol)

    return fetched


# ─── Parquet IV extraction ─────────────────────────────────────────────────────

def _symbol_parquet_glob(parquet_root: Path, symbol: str) -> str:
    """Return the glob path for a symbol's parquet files."""
    return str(parquet_root / f"underlying_symbol={symbol}" / "**" / "*.parquet")


def _symbol_has_parquet(parquet_root: Path, symbol: str) -> bool:
    """Check whether any parquet partition exists for this symbol."""
    return (parquet_root / f"underlying_symbol={symbol}").is_dir()


def _query_atm_iv_for_dates(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    symbol: str,
    trade_dates: List[date],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    For each (trade_date, expiry) combination where trade_date is in the given list,
    return the nearest-ATM call IV.

    Uses QUALIFY ROW_NUMBER() to select one row per (trade_date, expiry) — the
    call with minimum |moneyness_pct|.

    Returns a DataFrame with columns:
        trade_date, expiry, dte, underlying_price, atm_iv,
        abs_moneyness, open_interest, liquidity_score
    """
    if not trade_dates:
        return pd.DataFrame()

    # Format dates for SQL IN clause — use ISO strings, safe against injection
    # since all values come from our own date arithmetic, not user input
    date_literals = ", ".join(f"DATE '{d.isoformat()}'" for d in trade_dates)

    sql = f"""
    SELECT
        trade_date,
        expiry,
        dte,
        underlying_price,
        iv           AS atm_iv,
        ABS(moneyness_pct) AS abs_moneyness,
        open_interest,
        liquidity_score
    FROM read_parquet('{glob}')
    WHERE trade_date IN ({date_literals})
      AND dte BETWEEN 1 AND 60
      AND iv > {IV_MIN}
      AND iv < {IV_MAX}
      AND call_put = 'C'
      AND quote_quality_flag = 'ok'
      AND ABS(moneyness_pct) < {ATM_MONEYNESS_MAX_PCT}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY trade_date, expiry
        ORDER BY ABS(moneyness_pct)
    ) = 1
    ORDER BY trade_date, expiry
    """
    try:
        df = con.execute(sql).fetchdf()
    except Exception as exc:
        logger.warning("DuckDB query failed for %s: %s", symbol, exc)
        return pd.DataFrame()

    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["dte"] = df["dte"].astype(int)
    df["open_interest"] = df["open_interest"].fillna(0).astype(int)
    df["liquidity_score"] = df["liquidity_score"].fillna(0.0).astype(float)
    return df


def _get_available_trade_dates(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    symbol: str,
    window_start: date,
    window_end: date,
    logger: logging.Logger,
) -> List[date]:
    """
    Return all distinct trade_dates for a symbol between window_start and window_end.
    Used to determine which pre/post dates are available in the parquet store.
    """
    sql = f"""
    SELECT DISTINCT trade_date
    FROM read_parquet('{glob}')
    WHERE trade_date BETWEEN DATE '{window_start.isoformat()}'
                         AND DATE '{window_end.isoformat()}'
    ORDER BY trade_date
    """
    try:
        df = con.execute(sql).fetchdf()
    except Exception as exc:
        logger.warning("Trade-date query failed for %s: %s", symbol, exc)
        return []
    if df.empty:
        return []
    return [pd.Timestamp(d).date() for d in df["trade_date"]]


# ─── Leg selection ────────────────────────────────────────────────────────────

def _select_front_leg(
    iv_df: pd.DataFrame,
    trade_date: date,
    event_date: date,
) -> Tuple[Optional[IVLeg], str]:
    """
    Select the front (short-dated) IV leg for a given trade_date.

    Rules:
      1. Filter to trade_date rows with expiry strictly after event_date
         (front leg must not expire before the earnings catalyst)
      2. Prefer DTE in primary range [FRONT_DTE_PRIMARY_MIN, FRONT_DTE_PRIMARY_MAX]
         → select minimum DTE in that range (shortest-dated, highest event IV premium)
      3. Fallback: DTE in [FRONT_DTE_FALLBACK_MIN, FRONT_DTE_FALLBACK_MAX]
         → select minimum DTE in fallback range
      4. If neither range has a valid contract → return (None, reason)

    Returns (IVLeg, dte_range_used) or (None, failure_reason).
    """
    rows = iv_df[iv_df["trade_date"] == trade_date].copy()
    # Front leg must expire AFTER the earnings event to carry event IV premium
    rows = rows[rows["expiry"] > event_date]

    if rows.empty:
        return None, "no_contracts_with_expiry_after_event"

    # Primary range
    primary = rows[
        (rows["dte"] >= FRONT_DTE_PRIMARY_MIN) &
        (rows["dte"] <= FRONT_DTE_PRIMARY_MAX)
    ]
    if not primary.empty:
        chosen = primary.loc[primary["dte"].idxmin()]
        return _row_to_iv_leg(chosen, trade_date, "primary_front"), "primary_front"

    # Fallback range
    fallback = rows[
        (rows["dte"] >= FRONT_DTE_FALLBACK_MIN) &
        (rows["dte"] <= FRONT_DTE_FALLBACK_MAX)
    ]
    if not fallback.empty:
        chosen = fallback.loc[fallback["dte"].idxmin()]
        return _row_to_iv_leg(chosen, trade_date, "fallback_front"), "fallback_front"

    return None, f"no_front_contract_in_dte_range_1_to_{FRONT_DTE_FALLBACK_MAX}"


def _select_back_leg(
    iv_df: pd.DataFrame,
    trade_date: date,
    event_date: date,
) -> Tuple[Optional[IVLeg], str]:
    """
    Select the back (medium-dated) IV leg for a given trade_date.

    Rules:
      1. Filter to trade_date rows with expiry strictly after event_date
      2. Prefer DTE in primary range [BACK_DTE_PRIMARY_MIN, BACK_DTE_PRIMARY_MAX]
         → select DTE closest to BACK_DTE_TARGET_IDEAL (30)
      3. Fallback: DTE in [BACK_DTE_FALLBACK_MIN, BACK_DTE_FALLBACK_MAX]
         → select DTE closest to BACK_DTE_TARGET_IDEAL
      4. If neither → (None, reason)

    Returns (IVLeg, dte_range_used) or (None, failure_reason).
    """
    rows = iv_df[iv_df["trade_date"] == trade_date].copy()
    rows = rows[rows["expiry"] > event_date]

    if rows.empty:
        return None, "no_contracts_with_expiry_after_event"

    # Primary range
    primary = rows[
        (rows["dte"] >= BACK_DTE_PRIMARY_MIN) &
        (rows["dte"] <= BACK_DTE_PRIMARY_MAX)
    ]
    if not primary.empty:
        chosen = primary.loc[(primary["dte"] - BACK_DTE_TARGET_IDEAL).abs().idxmin()]
        return _row_to_iv_leg(chosen, trade_date, "primary_back"), "primary_back"

    # Fallback range
    fallback = rows[
        (rows["dte"] >= BACK_DTE_FALLBACK_MIN) &
        (rows["dte"] <= BACK_DTE_FALLBACK_MAX)
    ]
    if not fallback.empty:
        chosen = fallback.loc[(fallback["dte"] - BACK_DTE_TARGET_IDEAL).abs().idxmin()]
        return _row_to_iv_leg(chosen, trade_date, "fallback_back"), "fallback_back"

    return None, f"no_back_contract_in_dte_range_{BACK_DTE_FALLBACK_MIN}_to_{BACK_DTE_PRIMARY_MAX}"


def _match_post_leg(
    iv_df: pd.DataFrame,
    post_date: date,
    pre_leg: IVLeg,
) -> Tuple[Optional[IVLeg], bool, str]:
    """
    Match a post-event leg for the same expiry as the pre-event leg.

    Primary: exact expiry match on post_date
    Fallback: nearest available expiry within ±EXPIRY_FALLBACK_TOLERANCE_DAYS

    Returns (IVLeg, exact_match: bool, fallback_reason: str).
    """
    rows = iv_df[iv_df["trade_date"] == post_date].copy()
    if rows.empty:
        return None, False, "no_contracts_on_post_date"

    # Primary: exact expiry match
    exact = rows[rows["expiry"] == pre_leg.expiry]
    if not exact.empty:
        chosen = exact.loc[exact["dte"].idxmin()]
        return _row_to_iv_leg(chosen, post_date, pre_leg.dte_range), True, ""

    # Fallback: nearest expiry within tolerance window
    tolerance = timedelta(days=EXPIRY_FALLBACK_TOLERANCE_DAYS)
    near = rows[
        (rows["expiry"] >= pre_leg.expiry - tolerance) &
        (rows["expiry"] <= pre_leg.expiry + tolerance)
    ]
    if not near.empty:
        # Pick the expiry closest to the pre-event expiry
        nearest_idx = (near["expiry"] - pre_leg.expiry).abs().idxmin() if hasattr(near["expiry"] - pre_leg.expiry, "abs") else near.index[0]
        # expiry subtraction works with date objects via timedelta
        nearest_idx = min(near.index, key=lambda i: abs((near.loc[i, "expiry"] - pre_leg.expiry).days))
        chosen = near.loc[nearest_idx]
        reason = (
            f"expiry_fallback: pre={pre_leg.expiry.isoformat()} "
            f"post={chosen['expiry'].isoformat() if isinstance(chosen['expiry'], date) else chosen['expiry']}"
        )
        return _row_to_iv_leg(chosen, post_date, pre_leg.dte_range), False, reason

    return None, False, f"no_post_expiry_within_{EXPIRY_FALLBACK_TOLERANCE_DAYS}d_of_{pre_leg.expiry.isoformat()}"


def _row_to_iv_leg(row: pd.Series, trade_date: date, dte_range: str) -> IVLeg:
    """Convert a DataFrame row to an IVLeg dataclass."""
    expiry = row["expiry"] if isinstance(row["expiry"], date) else pd.Timestamp(row["expiry"]).date()
    return IVLeg(
        trade_date=trade_date,
        expiry=expiry,
        dte=int(row["dte"]),
        atm_iv=float(row["atm_iv"]),
        underlying_price=float(row["underlying_price"]),
        abs_moneyness_pct=float(row["abs_moneyness"]),
        open_interest=int(row["open_interest"]),
        liquidity_score=float(row["liquidity_score"]),
        dte_range=dte_range,
    )


# ─── Quality scoring ──────────────────────────────────────────────────────────

def _compute_quality_score(
    pre_front: IVLeg,
    pre_back: IVLeg,
    post_front: IVLeg,
    post_back: IVLeg,
    exact_expiry_match: bool,
    release_timing: str,
) -> Tuple[float, str]:
    """
    Compute a quality score in [0.0, 1.0] and a human-readable explanation.

    Starting score: 1.0
    Penalties are applied for each quality degradation condition.
    Score is clipped to [0.10, 1.00].

    Returns (quality_score, explanation_string).
    """
    score = 1.0
    reasons: List[str] = []

    # Front leg DTE quality
    if pre_front.dte_range == "fallback_front":
        score -= QUALITY_PENALTY_FRONT_DTE_FALLBACK
        reasons.append(f"front_dte_fallback(dte={pre_front.dte})")

    # Per-day penalty for front DTE exceeding primary range upper bound
    front_excess = max(0, pre_front.dte - FRONT_DTE_PRIMARY_MAX)
    if front_excess > 0:
        penalty = front_excess * QUALITY_PENALTY_PER_EXTRA_FRONT_DTE
        score -= penalty
        reasons.append(f"front_dte_excess_{front_excess}d(penalty={penalty:.2f})")

    # Back leg DTE quality
    if pre_back.dte_range == "fallback_back":
        score -= QUALITY_PENALTY_BACK_DTE_FALLBACK
        reasons.append(f"back_dte_fallback(dte={pre_back.dte})")

    # Per-5d penalty for back DTE deviation from ideal
    back_miss = abs(pre_back.dte - BACK_DTE_TARGET_IDEAL)
    if back_miss > 5:
        penalty = (back_miss // 5) * QUALITY_PENALTY_PER_BACK_DTE_MISS
        score -= penalty
        reasons.append(f"back_dte_miss_{back_miss}d_from_{BACK_DTE_TARGET_IDEAL}(penalty={penalty:.2f})")

    # Expiry match
    if not exact_expiry_match:
        score -= QUALITY_PENALTY_NO_EXPIRY_MATCH
        reasons.append("no_exact_expiry_match")

    # Release timing uncertainty
    if release_timing == "UNKNOWN":
        score -= QUALITY_PENALTY_UNKNOWN_TIMING
        reasons.append("release_timing_unknown")

    # ATM moneyness quality
    if pre_front.abs_moneyness_pct > 2.0:
        score -= QUALITY_PENALTY_HIGH_MONEYNESS_FRONT
        reasons.append(f"front_atm_moneyness_{pre_front.abs_moneyness_pct:.2f}pct")

    if pre_back.abs_moneyness_pct > 2.0:
        score -= QUALITY_PENALTY_HIGH_MONEYNESS_BACK
        reasons.append(f"back_atm_moneyness_{pre_back.abs_moneyness_pct:.2f}pct")

    score = float(np.clip(score, 0.10, 1.00))
    explanation = "; ".join(reasons) if reasons else "all_primary"
    return score, explanation


def _quality_tier(score: float) -> str:
    if score >= QUALITY_TIER_A_THRESHOLD:
        return "A"
    if score >= QUALITY_TIER_B_THRESHOLD:
        return "B"
    if score >= QUALITY_TIER_C_THRESHOLD:
        return "C"
    return "D"


# ─── Label assembly ───────────────────────────────────────────────────────────

def _build_label_row(
    event: EarningsEvent,
    pre_date: date,
    post_date: date,
    pre_front: IVLeg,
    pre_back: IVLeg,
    post_front: IVLeg,
    post_back: IVLeg,
    exact_expiry_match: bool,
    fallback_reason: str,
    front_range: str,
    back_range: str,
    parquet_root: Path,
    label_version: str,
) -> LabelRow:
    """
    Assemble a complete LabelRow from paired IV legs.

    All arithmetic follows the convention already in calibrate_earnings_iv_decay_labels():
      front_iv_crush_pct  = (post_front_iv - pre_front_iv) / pre_front_iv
      back_iv_crush_pct   = (post_back_iv  - pre_back_iv)  / pre_back_iv
      term_ratio_change   = (post_front/post_back) - (pre_front/pre_back)
      underlying_move_pct = (post_price - pre_price) / pre_price
    """
    pre_front_iv = pre_front.atm_iv
    post_front_iv = post_front.atm_iv
    pre_back_iv = pre_back.atm_iv
    post_back_iv = post_back.atm_iv

    # Crush percentages (negative = IV declined = crush happened)
    front_iv_crush_pct = (post_front_iv - pre_front_iv) / pre_front_iv
    back_iv_crush_pct = (post_back_iv - pre_back_iv) / pre_back_iv

    # Term ratio: front/back. Change is post minus pre.
    pre_term_ratio = pre_front_iv / pre_back_iv if pre_back_iv > 0 else 0.0
    post_term_ratio = post_front_iv / post_back_iv if post_back_iv > 0 else 0.0
    term_ratio_change = post_term_ratio - pre_term_ratio

    # Underlying move
    pre_price = pre_front.underlying_price
    post_price = post_front.underlying_price
    underlying_move_pct = (
        (post_price - pre_price) / pre_price
        if pre_price > 1e-6
        else 0.0
    )

    # Quality
    quality_score, fallback_details = _compute_quality_score(
        pre_front=pre_front,
        pre_back=pre_back,
        post_front=post_front,
        post_back=post_back,
        exact_expiry_match=exact_expiry_match,
        release_timing=event.release_timing,
    )
    tier = _quality_tier(quality_score)

    # Consolidate fallback reason (selection fallbacks + quality details)
    all_fallback_parts = [p for p in [fallback_reason, fallback_details] if p]
    full_fallback_reason = " | ".join(all_fallback_parts) if all_fallback_parts else ""

    fallback_used = int(
        front_range == "fallback_front"
        or back_range == "fallback_back"
        or not exact_expiry_match
    )

    return LabelRow(
        symbol=event.symbol,
        event_date=event.event_date.isoformat(),
        release_timing=event.release_timing,
        pre_capture_date=pre_date.isoformat(),
        post_capture_date=post_date.isoformat(),
        pre_front_iv=pre_front_iv,
        post_front_iv=post_front_iv,
        pre_back_iv=pre_back_iv,
        post_back_iv=post_back_iv,
        front_iv_crush_pct=front_iv_crush_pct,
        back_iv_crush_pct=back_iv_crush_pct,
        term_ratio_change=term_ratio_change,
        underlying_move_pct=underlying_move_pct,
        quality_score=quality_score,
        source=f"parquet_{label_version}",
        front_expiry=pre_front.expiry.isoformat(),
        back_expiry=pre_back.expiry.isoformat(),
        front_dte_pre=pre_front.dte,
        back_dte_pre=pre_back.dte,
        front_dte_post=post_front.dte,
        back_dte_post=post_back.dte,
        exact_expiry_match=int(exact_expiry_match),
        fallback_used=fallback_used,
        fallback_reason=full_fallback_reason,
        pre_front_atm_moneyness=pre_front.abs_moneyness_pct,
        pre_back_atm_moneyness=pre_back.abs_moneyness_pct,
        pre_front_oi=pre_front.open_interest,
        pre_back_oi=pre_back.open_interest,
        quality_tier=tier,
        label_version=label_version,
        parquet_store_root=str(parquet_root),
        pre_underlying_price=pre_price,
        post_underlying_price=post_price,
    )


# ─── SQLite writing ───────────────────────────────────────────────────────────

_INSERT_IGNORE = """
INSERT OR IGNORE INTO earnings_iv_decay_labels (
    symbol, event_date, release_timing, pre_capture_date, post_capture_date,
    pre_front_iv, post_front_iv, pre_back_iv, post_back_iv,
    front_iv_crush_pct, back_iv_crush_pct, term_ratio_change, underlying_move_pct,
    quality_score, source,
    front_expiry, back_expiry,
    front_dte_pre, back_dte_pre, front_dte_post, back_dte_post,
    exact_expiry_match, fallback_used, fallback_reason,
    pre_front_atm_moneyness, pre_back_atm_moneyness,
    pre_front_oi, pre_back_oi,
    quality_tier, label_version, parquet_store_root,
    pre_underlying_price, post_underlying_price
) VALUES (
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?,
    ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?,
    ?, ?,
    ?, ?, ?,
    ?, ?
)
"""

_INSERT_REPLACE = _INSERT_IGNORE.replace("INSERT OR IGNORE", "INSERT OR REPLACE")


def _label_row_params(row: LabelRow) -> Tuple:
    return (
        row.symbol, row.event_date, row.release_timing,
        row.pre_capture_date, row.post_capture_date,
        row.pre_front_iv, row.post_front_iv,
        row.pre_back_iv, row.post_back_iv,
        row.front_iv_crush_pct, row.back_iv_crush_pct,
        row.term_ratio_change, row.underlying_move_pct,
        row.quality_score, row.source,
        row.front_expiry, row.back_expiry,
        row.front_dte_pre, row.back_dte_pre,
        row.front_dte_post, row.back_dte_post,
        row.exact_expiry_match, row.fallback_used, row.fallback_reason,
        row.pre_front_atm_moneyness, row.pre_back_atm_moneyness,
        row.pre_front_oi, row.pre_back_oi,
        row.quality_tier, row.label_version, row.parquet_store_root,
        row.pre_underlying_price, row.post_underlying_price,
    )


def _write_labels(
    db_path: Path,
    rows: List[LabelRow],
    overwrite: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> int:
    """
    Write label rows to SQLite. Returns number of rows actually inserted.

    dry_run=True: logs what would be written without touching the DB.
    overwrite=False: INSERT OR IGNORE (default, safe)
    overwrite=True: INSERT OR REPLACE (re-runs with updated logic)
    """
    if not rows:
        return 0

    if dry_run:
        logger.info("[DRY RUN] Would write %d label rows (overwrite=%s)", len(rows), overwrite)
        return 0

    sql = _INSERT_REPLACE if overwrite else _INSERT_IGNORE
    params = [_label_row_params(r) for r in rows]

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        before = conn.execute(
            "SELECT COUNT(*) FROM earnings_iv_decay_labels"
        ).fetchone()[0]
        conn.executemany(sql, params)
        conn.commit()
        after = conn.execute(
            "SELECT COUNT(*) FROM earnings_iv_decay_labels"
        ).fetchone()[0]

    inserted = after - before
    return inserted


# ─── Per-symbol pipeline ──────────────────────────────────────────────────────

def _process_symbol(
    symbol: str,
    config: BackfillConfig,
    con: duckdb.DuckDBPyConnection,
    stats: BackfillStats,
    logger: logging.Logger,
) -> List[LabelRow]:
    """
    Full label extraction pipeline for a single symbol.

    Steps:
      1. Fetch earnings events (yfinance → cache)
      2. For each event, determine pre_date and post_date from parquet trade_dates
      3. Query ATM IV for those dates from parquet
      4. Select front/back legs and pair pre/post
      5. Compute label and quality score
      6. Return assembled LabelRow list (not yet written to DB)
    """
    stats.symbols_attempted += 1

    # Check parquet coverage
    if not _symbol_has_parquet(config.parquet_root, symbol):
        logger.debug("%s: no parquet partition — skipping", symbol)
        return []
    stats.symbols_with_parquet += 1

    # Determine fetch window
    parquet_start = config.start_date or date(2021, 1, 1)
    parquet_end = config.end_date or date.today()
    earnings_start = parquet_start - timedelta(days=30)
    earnings_end = min(parquet_end + timedelta(days=EARNINGS_LOOKAHEAD_DAYS), date.today())

    # Fetch earnings events
    events = get_earnings_events(
        symbol=symbol,
        start_date=earnings_start,
        end_date=earnings_end,
        db_path=config.db_path,
        logger=logger,
    )
    if not events:
        logger.debug("%s: no earnings events found", symbol)
        return []
    stats.symbols_with_earnings += 1

    # Filter events to within parquet date range (need data both before and after)
    events = [
        ev for ev in events
        if parquet_start <= ev.event_date <= parquet_end
    ]
    if not events:
        logger.debug("%s: no earnings events within parquet date window", symbol)
        return []

    logger.debug("%s: %d earnings events to process", symbol, len(events))

    # Get available trade_dates from parquet for this symbol
    glob = _symbol_parquet_glob(config.parquet_root, symbol)
    # Window: earliest_event - 20 days to latest_event + 10 days
    window_start = min(ev.event_date for ev in events) - timedelta(days=20)
    window_end = max(ev.event_date for ev in events) + timedelta(days=10)
    if config.start_date:
        window_start = max(window_start, config.start_date - timedelta(days=5))
    if config.end_date:
        window_end = min(window_end, config.end_date + timedelta(days=5))

    trade_dates = _get_available_trade_dates(con, glob, symbol, window_start, window_end, logger)
    if not trade_dates:
        logger.warning("%s: no trade_dates in parquet for window %s → %s", symbol, window_start, window_end)
        return []

    trade_date_set = set(trade_dates)

    # Determine all unique pre/post dates needed across all events
    needed_dates: set[date] = set()
    event_date_info: List[Tuple[EarningsEvent, Optional[date], Optional[date]]] = []

    for event in events:
        stats.events_attempted += 1

        # Pre-event date: last trade_date strictly before event_date
        pre_candidates = [d for d in trade_dates if d < event.event_date]
        if not pre_candidates:
            logger.debug(
                "%s %s: no pre-event trade_date available — skipping",
                symbol, event.event_date,
            )
            stats.events_failed_no_pre += 1
            stats.failure_reasons.append(
                f"{symbol} {event.event_date}: no_pre_event_trade_date"
            )
            event_date_info.append((event, None, None))
            continue
        pre_date = max(pre_candidates)

        # Post-event date: first trade_date strictly after event_date
        # (for BMO events the same day would also be valid, but we use the conservative
        # rule to avoid AMC/BMO ambiguity for UNKNOWN-timed events)
        post_candidates = [
            d for d in trade_dates
            if d > event.event_date
            and d <= event.event_date + timedelta(days=POST_MAX_DAYS_FORWARD)
        ]
        if not post_candidates:
            logger.debug(
                "%s %s: no post-event trade_date within %d days — skipping",
                symbol, event.event_date, POST_MAX_DAYS_FORWARD,
            )
            stats.events_failed_no_post += 1
            stats.failure_reasons.append(
                f"{symbol} {event.event_date}: no_post_event_trade_date"
            )
            event_date_info.append((event, None, None))
            continue
        post_date = min(post_candidates)

        needed_dates.add(pre_date)
        needed_dates.add(post_date)
        event_date_info.append((event, pre_date, post_date))

    if not needed_dates:
        return []

    # Single DuckDB query for all needed dates for this symbol
    iv_df = _query_atm_iv_for_dates(con, glob, symbol, sorted(needed_dates), logger)
    if iv_df.empty:
        logger.warning("%s: DuckDB returned no IV rows for needed dates", symbol)
        return []

    # Assemble label rows
    label_rows: List[LabelRow] = []
    for event, pre_date, post_date in event_date_info:
        if pre_date is None or post_date is None:
            continue  # already counted in stats above

        # --- Front leg ---
        pre_front, front_range = _select_front_leg(iv_df, pre_date, event.event_date)
        if pre_front is None:
            logger.debug(
                "%s %s: no valid front leg on pre_date=%s (%s)",
                symbol, event.event_date, pre_date, front_range,
            )
            stats.events_failed_no_front += 1
            stats.failure_reasons.append(
                f"{symbol} {event.event_date}: {front_range}"
            )
            continue

        # --- Back leg ---
        pre_back, back_range = _select_back_leg(iv_df, pre_date, event.event_date)
        if pre_back is None:
            logger.debug(
                "%s %s: no valid back leg on pre_date=%s (%s)",
                symbol, event.event_date, pre_date, back_range,
            )
            stats.events_failed_no_back += 1
            stats.failure_reasons.append(
                f"{symbol} {event.event_date}: {back_range}"
            )
            continue

        # --- Post-event front leg ---
        post_front, front_exact, front_fallback_reason = _match_post_leg(
            iv_df, post_date, pre_front
        )
        if post_front is None:
            logger.debug(
                "%s %s: no post-event front leg on post_date=%s (%s)",
                symbol, event.event_date, post_date, front_fallback_reason,
            )
            stats.events_failed_no_post += 1
            stats.failure_reasons.append(
                f"{symbol} {event.event_date}: post_front_{front_fallback_reason}"
            )
            continue

        # --- Post-event back leg ---
        post_back, back_exact, back_fallback_reason = _match_post_leg(
            iv_df, post_date, pre_back
        )
        if post_back is None:
            logger.debug(
                "%s %s: no post-event back leg on post_date=%s (%s)",
                symbol, event.event_date, post_date, back_fallback_reason,
            )
            stats.events_failed_no_post += 1
            stats.failure_reasons.append(
                f"{symbol} {event.event_date}: post_back_{back_fallback_reason}"
            )
            continue

        # --- IV sanity checks ---
        exact_match = front_exact and back_exact
        fallback_parts = [p for p in [front_fallback_reason, back_fallback_reason] if p]
        fallback_reason_str = " | ".join(fallback_parts)

        for iv_val, label in [
            (pre_front.atm_iv, "pre_front_iv"),
            (post_front.atm_iv, "post_front_iv"),
            (pre_back.atm_iv, "pre_back_iv"),
            (post_back.atm_iv, "post_back_iv"),
        ]:
            if not (IV_MIN < iv_val < IV_MAX):
                logger.debug(
                    "%s %s: invalid %s=%.4f — skipping",
                    symbol, event.event_date, label, iv_val,
                )
                stats.events_failed_iv_invalid += 1
                stats.failure_reasons.append(
                    f"{symbol} {event.event_date}: invalid_{label}={iv_val:.4f}"
                )
                break
        else:
            # All IV values are valid — build the label row
            row = _build_label_row(
                event=event,
                pre_date=pre_date,
                post_date=post_date,
                pre_front=pre_front,
                pre_back=pre_back,
                post_front=post_front,
                post_back=post_back,
                exact_expiry_match=exact_match,
                fallback_reason=fallback_reason_str,
                front_range=front_range,
                back_range=back_range,
                parquet_root=config.parquet_root,
                label_version=config.label_version,
            )
            label_rows.append(row)
            stats.events_labeled += 1

            # Update stats
            if exact_match:
                stats.exact_expiry_matches += 1
            else:
                stats.fallback_expiry_matches += 1
            if front_range == "primary_front":
                stats.front_dtes_primary += 1
            else:
                stats.front_dtes_fallback += 1
            if back_range == "primary_back":
                stats.back_dtes_primary += 1
            else:
                stats.back_dtes_fallback += 1
            tier = row.quality_tier
            if tier == "A":
                stats.tier_a += 1
            elif tier == "B":
                stats.tier_b += 1
            elif tier == "C":
                stats.tier_c += 1
            else:
                stats.tier_d += 1
            stats.front_crush_pcts.append(row.front_iv_crush_pct)
            stats.back_crush_pcts.append(row.back_iv_crush_pct)
            stats.quality_scores.append(row.quality_score)

            # Flag outliers for manual inspection (extreme crush values)
            if abs(row.front_iv_crush_pct) > 0.80 or abs(row.back_iv_crush_pct) > 0.80:
                stats.outlier_rows.append({
                    "symbol": symbol,
                    "event_date": row.event_date,
                    "front_crush_pct": f"{row.front_iv_crush_pct:.3f}",
                    "back_crush_pct": f"{row.back_iv_crush_pct:.3f}",
                    "quality_score": f"{row.quality_score:.2f}",
                    "quality_tier": row.quality_tier,
                    "fallback_used": row.fallback_used,
                    "fallback_reason": row.fallback_reason,
                    "pre_front_iv": f"{row.pre_front_iv:.4f}",
                    "post_front_iv": f"{row.post_front_iv:.4f}",
                    "front_dte_pre": row.front_dte_pre,
                    "note": "outlier: |crush| > 80%",
                })
            logger.debug(
                "%s %s: labeled | front_crush=%.1f%% back_crush=%.1f%% "
                "quality=%.2f(%s) exact=%s front_dte=%d(%s) back_dte=%d(%s)",
                symbol, event.event_date,
                row.front_iv_crush_pct * 100,
                row.back_iv_crush_pct * 100,
                row.quality_score,
                row.quality_tier,
                bool(exact_match),
                pre_front.dte, front_range,
                pre_back.dte, back_range,
            )

    if label_rows:
        logger.info(
            "%s: %d/%d events labeled (exact_match=%d fallback=%d tier_A=%d B=%d C=%d D=%d)",
            symbol,
            len(label_rows),
            len([x for x in event_date_info if x[1] is not None]),
            sum(1 for r in label_rows if r.exact_expiry_match),
            sum(1 for r in label_rows if not r.exact_expiry_match),
            sum(1 for r in label_rows if r.quality_tier == "A"),
            sum(1 for r in label_rows if r.quality_tier == "B"),
            sum(1 for r in label_rows if r.quality_tier == "C"),
            sum(1 for r in label_rows if r.quality_tier == "D"),
        )

    return label_rows


# ─── Validation report ────────────────────────────────────────────────────────

def _print_validation_report(
    stats: BackfillStats,
    config: BackfillConfig,
    all_rows: List[LabelRow],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Print a structured validation report and return a dict for JSON export.
    """
    total = stats.events_labeled
    front_crush = stats.front_crush_pcts
    back_crush = stats.back_crush_pcts
    qs = stats.quality_scores

    def _pct(num: int, denom: int) -> str:
        return f"{100 * num / denom:.1f}%" if denom else "n/a"

    logger.info("=" * 72)
    logger.info("VALIDATION REPORT — Earnings IV Label Backfill")
    logger.info("=" * 72)
    logger.info("Config:")
    logger.info("  label_version    : %s", config.label_version)
    logger.info("  parquet_root     : %s", config.parquet_root)
    logger.info("  db_path          : %s", config.db_path)
    logger.info("  start_date       : %s", config.start_date)
    logger.info("  end_date         : %s", config.end_date)
    logger.info("  dry_run          : %s", config.dry_run)
    logger.info("  overwrite        : %s", config.overwrite)
    logger.info("")
    logger.info("Coverage:")
    logger.info("  symbols attempted    : %d", stats.symbols_attempted)
    logger.info("  symbols w/ parquet   : %d", stats.symbols_with_parquet)
    logger.info("  symbols w/ earnings  : %d", stats.symbols_with_earnings)
    logger.info("")
    logger.info("Events:")
    logger.info("  events attempted     : %d", stats.events_attempted)
    logger.info("  events labeled       : %d  (%s of attempted)", total, _pct(total, stats.events_attempted))
    logger.info("  failed: no pre-date  : %d", stats.events_failed_no_pre)
    logger.info("  failed: no front leg : %d", stats.events_failed_no_front)
    logger.info("  failed: no back leg  : %d", stats.events_failed_no_back)
    logger.info("  failed: no post-date : %d", stats.events_failed_no_post)
    logger.info("  failed: IV invalid   : %d", stats.events_failed_iv_invalid)
    logger.info("")
    logger.info("Label quality:")
    logger.info("  exact expiry match   : %d  (%s)", stats.exact_expiry_matches, _pct(stats.exact_expiry_matches, total))
    logger.info("  fallback expiry      : %d  (%s)", stats.fallback_expiry_matches, _pct(stats.fallback_expiry_matches, total))
    logger.info("  front DTE primary    : %d  (%s)", stats.front_dtes_primary, _pct(stats.front_dtes_primary, total))
    logger.info("  front DTE fallback   : %d  (%s)", stats.front_dtes_fallback, _pct(stats.front_dtes_fallback, total))
    logger.info("  back DTE primary     : %d  (%s)", stats.back_dtes_primary, _pct(stats.back_dtes_primary, total))
    logger.info("  back DTE fallback    : %d  (%s)", stats.back_dtes_fallback, _pct(stats.back_dtes_fallback, total))
    logger.info("")
    logger.info("Quality tiers:")
    logger.info("  A (>=%.0f%%)          : %d  (%s)", QUALITY_TIER_A_THRESHOLD * 100, stats.tier_a, _pct(stats.tier_a, total))
    logger.info("  B (%.0f%%–%.0f%%)      : %d  (%s)", QUALITY_TIER_B_THRESHOLD * 100, QUALITY_TIER_A_THRESHOLD * 100, stats.tier_b, _pct(stats.tier_b, total))
    logger.info("  C (%.0f%%–%.0f%%)      : %d  (%s)", QUALITY_TIER_C_THRESHOLD * 100, QUALITY_TIER_B_THRESHOLD * 100, stats.tier_c, _pct(stats.tier_c, total))
    logger.info("  D (<%.0f%%)           : %d  (%s)", QUALITY_TIER_C_THRESHOLD * 100, stats.tier_d, _pct(stats.tier_d, total))

    if total > 0:
        logger.info("")
        logger.info("Crush distributions (front / back):")
        logger.info("  median  : %.1f%% / %.1f%%",
                    np.median(front_crush) * 100, np.median(back_crush) * 100)
        logger.info("  mean    : %.1f%% / %.1f%%",
                    np.mean(front_crush) * 100, np.mean(back_crush) * 100)
        logger.info("  p10     : %.1f%% / %.1f%%",
                    np.percentile(front_crush, 10) * 100, np.percentile(back_crush, 10) * 100)
        logger.info("  p90     : %.1f%% / %.1f%%",
                    np.percentile(front_crush, 90) * 100, np.percentile(back_crush, 90) * 100)
        logger.info("  crush rate (front <-10%%) : %.1f%%",
                    100 * sum(1 for x in front_crush if x < -0.10) / total)
        logger.info("")
        logger.info("Quality score:")
        logger.info("  median  : %.3f", np.median(qs))
        logger.info("  mean    : %.3f", np.mean(qs))
        logger.info("  p10/p90 : %.3f / %.3f", np.percentile(qs, 10), np.percentile(qs, 90))

    if stats.outlier_rows:
        logger.info("")
        logger.info("Outliers (|crush| > 80%%) — %d rows for manual inspection:", len(stats.outlier_rows))
        for row in stats.outlier_rows[:20]:
            logger.info("  %s %s front=%.1f%% back=%.1f%% tier=%s fallback=%s",
                        row["symbol"], row["event_date"],
                        float(row["front_crush_pct"]) * 100,
                        float(row["back_crush_pct"]) * 100,
                        row["quality_tier"],
                        row["fallback_reason"] or "none")

    if stats.failure_reasons[:20]:
        logger.info("")
        logger.info("Sample failure reasons (first 20):")
        for reason in stats.failure_reasons[:20]:
            logger.info("  %s", reason)

    # Per-symbol breakdown
    if all_rows:
        logger.info("")
        logger.info("Per-symbol label counts (top 30):")
        by_symbol: Dict[str, int] = {}
        for r in all_rows:
            by_symbol[r.symbol] = by_symbol.get(r.symbol, 0) + 1
        for sym, cnt in sorted(by_symbol.items(), key=lambda x: -x[1])[:30]:
            logger.info("  %-8s  %d", sym, cnt)

        logger.info("")
        logger.info("Per-year label counts:")
        by_year: Dict[str, int] = {}
        for r in all_rows:
            yr = r.event_date[:4]
            by_year[yr] = by_year.get(yr, 0) + 1
        for yr in sorted(by_year):
            logger.info("  %s  %d", yr, by_year[yr])

    logger.info("=" * 72)

    return {
        "label_version": config.label_version,
        "dry_run": config.dry_run,
        "overwrite": config.overwrite,
        "symbols_attempted": stats.symbols_attempted,
        "symbols_with_parquet": stats.symbols_with_parquet,
        "symbols_with_earnings": stats.symbols_with_earnings,
        "events_attempted": stats.events_attempted,
        "events_labeled": total,
        "events_failed_no_pre": stats.events_failed_no_pre,
        "events_failed_no_front": stats.events_failed_no_front,
        "events_failed_no_back": stats.events_failed_no_back,
        "events_failed_no_post": stats.events_failed_no_post,
        "events_failed_iv_invalid": stats.events_failed_iv_invalid,
        "exact_expiry_match_count": stats.exact_expiry_matches,
        "fallback_expiry_count": stats.fallback_expiry_matches,
        "exact_expiry_rate_pct": round(100 * stats.exact_expiry_matches / total, 1) if total else 0,
        "front_dte_primary_count": stats.front_dtes_primary,
        "front_dte_fallback_count": stats.front_dtes_fallback,
        "back_dte_primary_count": stats.back_dtes_primary,
        "back_dte_fallback_count": stats.back_dtes_fallback,
        "tier_A": stats.tier_a,
        "tier_B": stats.tier_b,
        "tier_C": stats.tier_c,
        "tier_D": stats.tier_d,
        "front_crush_median": round(float(np.median(front_crush)) if front_crush else 0, 4),
        "front_crush_mean": round(float(np.mean(front_crush)) if front_crush else 0, 4),
        "front_crush_rate_10pct": round(
            100 * sum(1 for x in front_crush if x < -0.10) / total, 1
        ) if total else 0,
        "quality_score_median": round(float(np.median(qs)) if qs else 0, 3),
        "outlier_count": len(stats.outlier_rows),
        "outliers": stats.outlier_rows,
        "sample_failures": stats.failure_reasons[:50],
    }


# ─── Main pipeline ────────────────────────────────────────────────────────────

def _discover_parquet_symbols(parquet_root: Path) -> List[str]:
    """Return all symbol names present in the parquet store, excluding non-earnings symbols."""
    if not parquet_root.exists():
        return []
    symbols: List[str] = []
    for path in sorted(parquet_root.iterdir()):
        prefix = "underlying_symbol="
        if path.is_dir() and path.name.startswith(prefix):
            symbol = path.name[len(prefix):].upper()
            if symbol not in NON_EARNINGS_SYMBOLS and re.match(r"^[A-Z][A-Z0-9.-]{0,15}$", symbol):
                symbols.append(symbol)
    return symbols


def run_backfill(config: BackfillConfig, logger: logging.Logger) -> Dict[str, Any]:
    """
    Execute the full earnings IV label backfill pipeline.

    Returns a report dict (also printed to stdout).
    """
    logger.info("━" * 72)
    logger.info("Earnings IV Label Backfill — label_version=%s dry_run=%s overwrite=%s",
                config.label_version, config.dry_run, config.overwrite)
    logger.info("━" * 72)

    # 1. Ensure schema exists (creates or migrates tables).
    # The earnings_events cache table is always created (it is read-only from the
    # label perspective and needed even during dry-run to avoid yfinance hammering).
    # The labels table and extended-column migrations are skipped in dry-run mode.
    if not config.dry_run:
        _ensure_schema(config.db_path, logger)
    else:
        logger.info("[DRY RUN] Label schema check skipped (no label DB writes)")
        _ensure_earnings_cache(config.db_path)

    # 2. Resolve symbol list
    if config.symbols:
        symbols = [s.upper() for s in config.symbols if s.strip()]
        logger.info("Processing %d requested symbols", len(symbols))
    else:
        symbols = _discover_parquet_symbols(config.parquet_root)
        logger.info("Discovered %d symbols in parquet store", len(symbols))

    if not symbols:
        logger.error("No symbols to process — check parquet_root: %s", config.parquet_root)
        return {"error": "no_symbols"}

    stats = BackfillStats()
    all_rows: List[LabelRow] = []
    total_inserted = 0

    # 3. Process symbols — single DuckDB connection shared across all
    with duckdb.connect(database=":memory:") as con:
        for i, symbol in enumerate(symbols, 1):
            logger.info("[%d/%d] Processing %s ...", i, len(symbols), symbol)
            rows = _process_symbol(symbol, config, con, stats, logger)
            if rows:
                all_rows.extend(rows)
                inserted = _write_labels(
                    db_path=config.db_path,
                    rows=rows,
                    overwrite=config.overwrite,
                    dry_run=config.dry_run,
                    logger=logger,
                )
                total_inserted += inserted
                if not config.dry_run:
                    logger.info(
                        "  → wrote %d/%d new rows (inserted=%d, already_existed=%d)",
                        inserted, len(rows), inserted, len(rows) - inserted,
                    )

    logger.info("")
    logger.info("Backfill complete: %d symbols → %d labels assembled → %d inserted to DB",
                len(symbols), len(all_rows), total_inserted)

    # 4. Validation report
    report = _print_validation_report(stats, config, all_rows, logger)
    report["total_inserted"] = total_inserted

    # 5. Save JSON report
    if not config.dry_run:
        report_dir = Path.home() / ".options_calculator_pro" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report_path = report_dir / f"earnings_iv_labels_{config.label_version}_{ts}.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Validation report saved to %s", report_path)

    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build real earnings IV crush labels from the EOD parquet store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run — inspect without writing
  python scripts/build_earnings_iv_labels.py --symbols AAPL,MSFT --dry-run

  # Full backfill
  python scripts/build_earnings_iv_labels.py

  # Date-restricted backfill
  python scripts/build_earnings_iv_labels.py --start-date 2023-01-01 --end-date 2024-12-31

  # Re-run with overwrite
  python scripts/build_earnings_iv_labels.py --overwrite
        """,
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol list (e.g. AAPL,MSFT). Default: all symbols in parquet store.",
    )
    p.add_argument(
        "--start-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only process earnings events on or after this date.",
    )
    p.add_argument(
        "--end-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only process earnings events on or before this date.",
    )
    p.add_argument(
        "--parquet-root",
        default=None,
        help=f"Override parquet store root. Default: {DEFAULT_PARQUET_ROOT}",
    )
    p.add_argument(
        "--db-path",
        default=None,
        help=f"Override SQLite DB path. Default: {DEFAULT_DB_PATH}",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be written without touching the database.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing rows for the same (symbol, event_date). "
            "Default is INSERT OR IGNORE (safe for incremental runs)."
        ),
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def _parse_date(value: Optional[str], name: str) -> Optional[date]:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise SystemExit(f"Invalid {name}: {value!r} — expected YYYY-MM-DD")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logger = _setup_logging(verbose=args.verbose)

    # Resolve env-var override for parquet root
    parquet_root_str = (
        args.parquet_root
        or os.environ.get("MARKET_DATA_ROOT", "").strip()
    )
    parquet_root = (
        Path(parquet_root_str).expanduser().resolve() / "research" / "options_features_eod"
        if parquet_root_str
        else DEFAULT_PARQUET_ROOT
    )
    # If MARKET_DATA_ROOT points directly to the research/options_features_eod dir, use as-is
    if parquet_root_str and (Path(parquet_root_str) / "underlying_symbol=AAPL").exists():
        parquet_root = Path(parquet_root_str).expanduser().resolve()

    db_path = Path(args.db_path).expanduser().resolve() if args.db_path else DEFAULT_DB_PATH
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )

    config = BackfillConfig(
        parquet_root=parquet_root,
        db_path=db_path,
        symbols=symbols,
        start_date=_parse_date(args.start_date, "--start-date"),
        end_date=_parse_date(args.end_date, "--end-date"),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        verbose=args.verbose,
        label_version=LABEL_VERSION,
    )

    logger.info("Parquet root : %s (exists=%s)", parquet_root, parquet_root.exists())
    logger.info("DB path      : %s", db_path)

    if not parquet_root.exists():
        logger.error(
            "Parquet root does not exist: %s\n"
            "Check that the T9 drive is mounted and the path is correct.\n"
            "Override with --parquet-root or MARKET_DATA_ROOT env var.",
            parquet_root,
        )
        return 1

    report = run_backfill(config, logger)

    if report.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
