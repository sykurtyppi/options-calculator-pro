"""
database.py — SQLite schema, connection management, and CRUD helpers.

All system state lives here. The DB is the source of truth.
Every signal evaluated, every trade, every daily mark, every risk snapshot
is written here for full auditability.

Tables:
  signals          — every candidate evaluated (traded or not)
  trades           — full trade lifecycle: entry → MTM → exit
  daily_marks      — end-of-day MTM for open positions
  risk_snapshots   — daily risk state for limit enforcement
  performance_log  — daily P&L series for Sharpe / drawdown

Schema is additive (new columns added with ALTER TABLE, never dropped).
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

log = logging.getLogger(__name__)

SYSTEM_DB = Path.home() / ".options_calculator_pro" / "trading_system.db"


# ── connection ─────────────────────────────────────────────────────────────────

@contextmanager
def get_conn(db_path: Path = SYSTEM_DB) -> Generator[sqlite3.Connection, None, None]:
    """Context manager: auto-commit on success, rollback on exception."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── schema ─────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- ─────────────────────────────────────────────────────────────────
-- signals: every candidate evaluated by the signal engine each day
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date               TEXT NOT NULL,   -- date this was evaluated (YYYY-MM-DD)
    symbol                  TEXT NOT NULL,
    event_date              TEXT NOT NULL,   -- earnings release date
    pre_capture_date        TEXT NOT NULL,   -- T-1
    post_capture_date       TEXT NOT NULL,   -- T+1
    front_expiry            TEXT NOT NULL,
    back_expiry             TEXT NOT NULL,
    -- Signal values at screening (T-1)
    nbr                     REAL,
    nbr_threshold           REAL,
    t1_front_oi             INTEGER,
    t1_back_oi              INTEGER,
    t1_front_spread_pct     REAL,
    t1_back_spread_pct      REAL,
    t1_front_iv             REAL,
    t1_back_iv              REAL,
    -- Filter outcome
    passed_nbr              INTEGER DEFAULT 0,
    passed_oi               INTEGER DEFAULT 0,
    passed_spread           INTEGER DEFAULT 0,
    passed_all_filters      INTEGER DEFAULT 0,
    filter_reject_reason    TEXT,
    -- Risk gate outcome
    passed_risk_gate        INTEGER DEFAULT 0,
    risk_reject_reason      TEXT,
    -- Trading decision
    traded                  INTEGER DEFAULT 0,
    trade_id                INTEGER REFERENCES trades(id),
    -- Metadata
    created_at              TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signals_scan ON signals(scan_date);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, event_date);

-- ─────────────────────────────────────────────────────────────────
-- trades: full lifecycle from entry to exit
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id               INTEGER REFERENCES signals(id),
    symbol                  TEXT NOT NULL,
    sector                  TEXT,
    event_date              TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'open',  -- open / closed / cancelled
    n_contracts             INTEGER NOT NULL,
    -- Entry
    entry_date              TEXT,
    entry_front_bid         REAL,
    entry_front_ask         REAL,
    entry_front_mid         REAL,
    entry_front_fill        REAL,
    entry_back_bid          REAL,
    entry_back_ask          REAL,
    entry_back_mid          REAL,
    entry_back_fill         REAL,
    entry_cal_mid           REAL,      -- back_mid - front_mid at entry
    entry_cal_fill          REAL,      -- actual calendar debit paid
    entry_slippage_cost     REAL,      -- extra cost vs mid (both legs)
    entry_front_dte         INTEGER,
    entry_nbr               REAL,
    -- Expected exit
    expected_exit_date      TEXT,      -- T+1 from post_capture_date
    -- Exit
    exit_date               TEXT,
    exit_reason             TEXT,      -- 'scheduled' / 'stop_loss' / 'manual'
    exit_front_bid          REAL,
    exit_front_ask          REAL,
    exit_front_mid          REAL,
    exit_front_fill         REAL,
    exit_back_bid           REAL,
    exit_back_ask           REAL,
    exit_back_mid           REAL,
    exit_back_fill          REAL,
    exit_cal_mid            REAL,
    exit_cal_fill           REAL,
    exit_slippage_cost      REAL,
    -- P&L (realised on close)
    gross_pnl               REAL,      -- (exit_cal_fill - entry_cal_fill) * 100 * n
    total_transaction_cost  REAL,      -- realistic fill cost (slippage model)
    net_pnl                 REAL,      -- gross - costs
    capital_deployed        REAL,      -- entry_cal_fill * 100 * n_contracts (max risk)
    -- Metadata / audit
    notes                   TEXT,
    created_at              TEXT DEFAULT (datetime('now')),
    updated_at              TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_date);

-- ─────────────────────────────────────────────────────────────────
-- daily_marks: EOD mark-to-market for each open position
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS daily_marks (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id                INTEGER NOT NULL REFERENCES trades(id),
    mark_date               TEXT NOT NULL,
    front_bid               REAL,
    front_ask               REAL,
    front_mid               REAL,
    back_bid                REAL,
    back_ask                REAL,
    back_mid                REAL,
    cal_mid                 REAL,       -- back_mid - front_mid
    unrealized_pnl          REAL,       -- (cal_mid - entry_cal_fill) * 100 * n
    mark_source             TEXT DEFAULT 'parquet',  -- 'parquet' / 'broker' / 'manual'
    created_at              TEXT DEFAULT (datetime('now')),
    UNIQUE(trade_id, mark_date)
);

-- ─────────────────────────────────────────────────────────────────
-- risk_snapshots: daily risk state captured after each run
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS risk_snapshots (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date           TEXT NOT NULL UNIQUE,
    -- Position state
    n_open_trades           INTEGER DEFAULT 0,
    open_trade_ids          TEXT,    -- JSON list
    total_capital_deployed  REAL DEFAULT 0,
    -- P&L state
    unrealized_pnl          REAL DEFAULT 0,
    realized_pnl_mtd        REAL DEFAULT 0,
    realized_pnl_ytd        REAL DEFAULT 0,
    daily_net_pnl           REAL DEFAULT 0,
    -- Concentration
    top_symbol              TEXT,
    top_symbol_pct          REAL,
    tech_comm_pct           REAL,
    -- Limit breach flags
    at_position_limit       INTEGER DEFAULT 0,
    at_monthly_loss_limit   INTEGER DEFAULT 0,
    -- Metadata
    created_at              TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────────
-- performance_log: daily P&L series for analytics
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS performance_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    log_date                TEXT NOT NULL UNIQUE,
    realized_pnl_today      REAL DEFAULT 0,  -- closed trades today
    unrealized_pnl_today    REAL DEFAULT 0,  -- open position MTM change
    total_pnl_today         REAL DEFAULT 0,  -- realized + MTM change
    cumulative_realized     REAL DEFAULT 0,  -- running total of closed trades
    n_open_trades           INTEGER DEFAULT 0,
    n_trades_closed_ytd     INTEGER DEFAULT 0,
    n_signals_evaluated     INTEGER DEFAULT 0,
    created_at              TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────────
-- system_config: key-value store for runtime parameters
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_config (
    key                     TEXT PRIMARY KEY,
    value                   TEXT NOT NULL,
    description             TEXT,
    updated_at              TEXT DEFAULT (datetime('now'))
);
"""

DEFAULT_CONFIG = {
    # Universe filters
    "min_front_oi":             "75",
    "min_back_oi":              "37",
    "max_spread_pct":           "0.25",
    "nbr_q4_pctile":            "0.60",
    # Execution
    "entry_bdays_before_pre":   "2",
    "exit_bdays_after_post":    "1",
    "fill_fraction":            "0.75",   # pay 75% of half-spread per leg
    # Position limits
    "max_contracts_per_trade":  "3",
    "max_simultaneous_trades":  "3",
    # Concentration limits
    "max_pct_one_symbol":       "0.25",  # 25% of deployed capital
    "max_pct_tech_comm":        "0.80",  # 80% of deployed capital
    # Loss controls
    "per_trade_stop_multiple":  "2.0",   # stop at -2× avg expected loss
    "monthly_loss_limit_pct":   "0.10",  # -10% of total capital
    "starting_capital":         "5000",  # $ capital base for risk calculations
    # Scaling gates
    "scale_min_trades":         "20",    # minimum trades before scaling
    "scale_target_contracts":   "2",
    "scale_max_contracts":      "3",
    # Backtest reference (for vs-live comparison)
    "backtest_avg_net_per_trade": "289.92",
    "backtest_win_rate":          "0.842",
    "backtest_avg_cost_per_trade": "56.03",
}

_TABLE_COLUMNS = {
    "signals": {
        "scan_date", "symbol", "event_date", "pre_capture_date", "post_capture_date",
        "front_expiry", "back_expiry", "nbr", "nbr_threshold", "t1_front_oi",
        "t1_back_oi", "t1_front_spread_pct", "t1_back_spread_pct", "t1_front_iv",
        "t1_back_iv", "passed_nbr", "passed_oi", "passed_spread", "passed_all_filters",
        "filter_reject_reason", "passed_risk_gate", "risk_reject_reason", "traded",
        "trade_id", "created_at",
    },
    "trades": {
        "signal_id", "symbol", "sector", "event_date", "status", "n_contracts",
        "entry_date", "entry_front_bid", "entry_front_ask", "entry_front_mid",
        "entry_front_fill", "entry_back_bid", "entry_back_ask", "entry_back_mid",
        "entry_back_fill", "entry_cal_mid", "entry_cal_fill", "entry_slippage_cost",
        "entry_front_dte", "entry_nbr", "expected_exit_date", "exit_date", "exit_reason",
        "exit_front_bid", "exit_front_ask", "exit_front_mid", "exit_front_fill",
        "exit_back_bid", "exit_back_ask", "exit_back_mid", "exit_back_fill",
        "exit_cal_mid", "exit_cal_fill", "exit_slippage_cost", "gross_pnl",
        "total_transaction_cost", "net_pnl", "capital_deployed", "notes", "created_at",
        "updated_at",
    },
    "risk_snapshots": {
        "snapshot_date", "n_open_trades", "open_trade_ids", "total_capital_deployed",
        "unrealized_pnl", "realized_pnl_mtd", "realized_pnl_ytd", "daily_net_pnl",
        "top_symbol", "top_symbol_pct", "tech_comm_pct", "at_position_limit",
        "at_monthly_loss_limit", "created_at",
    },
    "performance_log": {
        "log_date", "realized_pnl_today", "unrealized_pnl_today", "total_pnl_today",
        "cumulative_realized", "n_open_trades", "n_trades_closed_ytd",
        "n_signals_evaluated", "created_at",
    },
}


def _validate_columns(table: str, payload: Dict[str, Any]) -> None:
    allowed = _TABLE_COLUMNS.get(table)
    if allowed is None:
        raise ValueError(f"Unknown table for column validation: {table}")
    invalid = sorted(set(payload) - allowed)
    if invalid:
        raise ValueError(f"Invalid {table} columns: {', '.join(invalid)}")


def init_db(db_path: Path = SYSTEM_DB) -> None:
    """Create schema and seed default config if not exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        for k, v in DEFAULT_CONFIG.items():
            conn.execute(
                "INSERT OR IGNORE INTO system_config(key, value) VALUES (?,?)",
                (k, v),
            )
    log.info("Database initialised at %s", db_path)


def get_config(key: str, db_path: Path = SYSTEM_DB) -> Optional[str]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM system_config WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None


def set_config(key: str, value: str, db_path: Path = SYSTEM_DB) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO system_config(key, value, updated_at) "
            "VALUES (?, ?, datetime('now'))",
            (key, value),
        )


def get_all_config(db_path: Path = SYSTEM_DB) -> Dict[str, str]:
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM system_config").fetchall()
        return {r["key"]: r["value"] for r in rows}


# ── signal helpers ─────────────────────────────────────────────────────────────

def insert_signal(signal: Dict[str, Any], db_path: Path = SYSTEM_DB) -> int:
    _validate_columns("signals", signal)
    cols = ", ".join(signal.keys())
    placeholders = ", ".join("?" * len(signal))
    with get_conn(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO signals ({cols}) VALUES ({placeholders})",
            list(signal.values()),
        )
        return cur.lastrowid


def update_signal(signal_id: int, updates: Dict[str, Any],
                  db_path: Path = SYSTEM_DB) -> None:
    _validate_columns("signals", updates)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with get_conn(db_path) as conn:
        conn.execute(
            f"UPDATE signals SET {set_clause} WHERE id=?",
            list(updates.values()) + [signal_id],
        )


# ── trade helpers ──────────────────────────────────────────────────────────────

def insert_trade(trade: Dict[str, Any], db_path: Path = SYSTEM_DB) -> int:
    _validate_columns("trades", trade)
    cols = ", ".join(trade.keys())
    placeholders = ", ".join("?" * len(trade))
    with get_conn(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO trades ({cols}) VALUES ({placeholders})",
            list(trade.values()),
        )
        return cur.lastrowid


def update_trade(trade_id: int, updates: Dict[str, Any],
                 db_path: Path = SYSTEM_DB) -> None:
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    _validate_columns("trades", updates)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with get_conn(db_path) as conn:
        conn.execute(
            f"UPDATE trades SET {set_clause} WHERE id=?",
            list(updates.values()) + [trade_id],
        )


def get_open_trades(db_path: Path = SYSTEM_DB) -> List[Dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='open' ORDER BY entry_date"
        ).fetchall()
        return [dict(r) for r in rows]


def get_trade(trade_id: int, db_path: Path = SYSTEM_DB) -> Optional[Dict]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None


def get_closed_trades(year: Optional[int] = None,
                      month: Optional[int] = None,
                      db_path: Path = SYSTEM_DB) -> List[Dict]:
    clauses = ["status='closed'"]
    params: List[Any] = []
    if year:
        clauses.append("strftime('%Y', exit_date)=?")
        params.append(str(year))
    if month:
        clauses.append("strftime('%m', exit_date)=?")
        params.append(f"{month:02d}")
    where = " AND ".join(clauses)
    with get_conn(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY exit_date",
            params,
        ).fetchall()
        return [dict(r) for r in rows]


# ── mark helpers ───────────────────────────────────────────────────────────────

def upsert_daily_mark(mark: Dict[str, Any], db_path: Path = SYSTEM_DB) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_marks
              (trade_id, mark_date, front_bid, front_ask, front_mid,
               back_bid, back_ask, back_mid, cal_mid, unrealized_pnl, mark_source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                mark["trade_id"], mark["mark_date"],
                mark.get("front_bid"), mark.get("front_ask"), mark.get("front_mid"),
                mark.get("back_bid"), mark.get("back_ask"), mark.get("back_mid"),
                mark.get("cal_mid"), mark.get("unrealized_pnl"),
                mark.get("mark_source", "parquet"),
            ),
        )


# ── risk snapshot helpers ──────────────────────────────────────────────────────

def upsert_risk_snapshot(snap: Dict[str, Any],
                          db_path: Path = SYSTEM_DB) -> None:
    _validate_columns("risk_snapshots", snap)
    with get_conn(db_path) as conn:
        cols = ", ".join(snap.keys())
        placeholders = ", ".join("?" * len(snap))
        conn.execute(
            f"INSERT OR REPLACE INTO risk_snapshots ({cols}) VALUES ({placeholders})",
            list(snap.values()),
        )


def upsert_performance_log(entry: Dict[str, Any],
                            db_path: Path = SYSTEM_DB) -> None:
    _validate_columns("performance_log", entry)
    with get_conn(db_path) as conn:
        cols = ", ".join(entry.keys())
        placeholders = ", ".join("?" * len(entry))
        conn.execute(
            f"INSERT OR REPLACE INTO performance_log ({cols}) VALUES ({placeholders})",
            list(entry.values()),
        )


def get_performance_series(db_path: Path = SYSTEM_DB) -> List[Dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM performance_log ORDER BY log_date"
        ).fetchall()
        return [dict(r) for r in rows]
