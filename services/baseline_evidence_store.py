"""Shadow baseline evidence store for forward paper comparisons.

This store records hypothetical baseline trades beside selector paper trades.
It is intentionally separate from OutcomeStore so baseline rows never update
calibration or structure priors.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Optional

_DEFAULT_STORE = Path.home() / ".options_calculator_pro" / "evidence" / "baseline_evidence.sqlite"
_WRITE_LOCK = threading.Lock()

BASELINE_STRUCTURES = {
    "always_atm_straddle": "atm_straddle",
    "always_otm_strangle": "otm_strangle",
}

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS baseline_trades (
    baseline_id                  TEXT PRIMARY KEY,
    recommendation_id            TEXT NOT NULL,
    symbol                       TEXT NOT NULL,
    baseline_name                TEXT NOT NULL,
    structure                    TEXT NOT NULL,
    entry_date                   TEXT NOT NULL,
    exit_date                    TEXT,
    earnings_date                TEXT,
    selector_structure           TEXT,
    entry_mid                    REAL,
    exit_mid                     REAL,
    realized_return_pct          REAL,
    realized_expansion_pct       REAL,
    modeled_cost_pct             REAL,
    execution_penalty_at_entry   REAL,
    data_quality_score_at_entry  REAL,
    iv_rv_har_at_entry           REAL,
    iv_rv_yz_at_entry            REAL,
    quote_source_at_entry        TEXT,
    quote_quality_at_entry       TEXT,
    entry_bid_ask_mid_json       TEXT,
    evidence_quality_status      TEXT,
    evidence_quality_reasons_json TEXT,
    claim_allowed                INTEGER,
    execution_grade              INTEGER,
    entry_execution_scenarios_json TEXT,
    surface_quality_status       TEXT,
    surface_quality_reasons_json TEXT,
    surface_quality_json         TEXT,
    surface_crossed_quote_count  INTEGER,
    surface_zero_bid_count       INTEGER,
    surface_extreme_spread_count INTEGER,
    surface_sparse_atm_count     INTEGER,
    surface_iv_anomaly_count     INTEGER,
    quote_source_at_exit         TEXT,
    quote_quality_at_exit        TEXT,
    exit_bid_ask_mid_json        TEXT,
    exit_execution_scenarios_json TEXT,
    status                       TEXT NOT NULL DEFAULT 'open',
    skip_reason                  TEXT,
    metadata_json                TEXT,
    created_at                   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_baseline_recommendation_id ON baseline_trades (recommendation_id);
CREATE INDEX IF NOT EXISTS idx_baseline_due ON baseline_trades (earnings_date, status);
CREATE INDEX IF NOT EXISTS idx_baseline_name ON baseline_trades (baseline_name);
CREATE INDEX IF NOT EXISTS idx_baseline_symbol ON baseline_trades (symbol);
"""

_MIGRATION_COLUMNS: Dict[str, str] = {
    "metadata_json": "TEXT",
    "evidence_quality_status": "TEXT",
    "evidence_quality_reasons_json": "TEXT",
    "claim_allowed": "INTEGER",
    "execution_grade": "INTEGER",
    "entry_execution_scenarios_json": "TEXT",
    "exit_execution_scenarios_json": "TEXT",
    "surface_quality_status": "TEXT",
    "surface_quality_reasons_json": "TEXT",
    "surface_quality_json": "TEXT",
    "surface_crossed_quote_count": "INTEGER",
    "surface_zero_bid_count": "INTEGER",
    "surface_extreme_spread_count": "INTEGER",
    "surface_sparse_atm_count": "INTEGER",
    "surface_iv_anomaly_count": "INTEGER",
}


def make_baseline_id(recommendation_id: str, baseline_name: str) -> str:
    return f"{recommendation_id}|baseline|{baseline_name}"


class BaselineEvidenceStore:
    def __init__(self, store_path: Path = _DEFAULT_STORE) -> None:
        self._path = store_path
        self._conn = _open_db(store_path)

    @property
    def path(self) -> Path:
        return self._path

    def insert_entry(
        self,
        *,
        recommendation_id: str,
        symbol: str,
        baseline_name: str,
        structure: str,
        entry_date: date,
        earnings_date: Optional[date],
        selector_structure: Optional[str],
        entry_mid: Optional[float],
        modeled_cost_pct: Optional[float],
        execution_penalty_at_entry: Optional[float],
        data_quality_score_at_entry: Optional[float],
        iv_rv_har_at_entry: Optional[float],
        iv_rv_yz_at_entry: Optional[float],
        quote_source_at_entry: Optional[str],
        quote_quality_at_entry: Optional[str],
        entry_bid_ask_mid: Optional[Dict[str, Any]] = None,
        evidence_quality_status: Optional[str] = None,
        evidence_quality_reasons: Optional[list[str]] = None,
        claim_allowed: Optional[bool] = None,
        execution_grade: Optional[bool] = None,
        entry_execution_scenarios: Optional[Dict[str, Any]] = None,
        surface_quality: Optional[Dict[str, Any]] = None,
        status: str = "open",
        skip_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        baseline_id = make_baseline_id(recommendation_id, baseline_name)
        sql = """
            INSERT OR IGNORE INTO baseline_trades (
                baseline_id, recommendation_id, symbol, baseline_name, structure,
                entry_date, earnings_date, selector_structure, entry_mid,
                modeled_cost_pct, execution_penalty_at_entry,
                data_quality_score_at_entry, iv_rv_har_at_entry, iv_rv_yz_at_entry,
                quote_source_at_entry, quote_quality_at_entry, entry_bid_ask_mid_json,
                evidence_quality_status, evidence_quality_reasons_json,
                claim_allowed, execution_grade, entry_execution_scenarios_json,
                surface_quality_status, surface_quality_reasons_json, surface_quality_json,
                surface_crossed_quote_count, surface_zero_bid_count,
                surface_extreme_spread_count, surface_sparse_atm_count,
                surface_iv_anomaly_count,
                status, skip_reason, metadata_json
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """
        surface_quality = surface_quality or {}
        params = (
            baseline_id,
            recommendation_id,
            str(symbol).upper(),
            baseline_name,
            structure,
            _fmt_date(entry_date),
            _fmt_date(earnings_date),
            selector_structure,
            entry_mid,
            modeled_cost_pct,
            execution_penalty_at_entry,
            data_quality_score_at_entry,
            iv_rv_har_at_entry,
            iv_rv_yz_at_entry,
            quote_source_at_entry,
            quote_quality_at_entry,
            _json(entry_bid_ask_mid or {}),
            evidence_quality_status,
            _json(evidence_quality_reasons or []),
            _bool_int(claim_allowed),
            _bool_int(execution_grade),
            _json(entry_execution_scenarios or {}),
            surface_quality.get("status"),
            _json(surface_quality.get("warning_flags") or []),
            _json(surface_quality),
            int(surface_quality.get("crossed_quote_count") or 0),
            int(surface_quality.get("zero_bid_count") or 0),
            int(surface_quality.get("extreme_spread_count") or 0),
            int(surface_quality.get("sparse_atm_expiration_count") or 0),
            int(surface_quality.get("missing_iv_count") or 0) + int(surface_quality.get("iv_outlier_count") or 0),
            status,
            skip_reason,
            _json(metadata or {}),
        )
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                cur.execute(sql, params)
                return cur.rowcount > 0

    def baselines_due_for_exit(self, as_of_date: date) -> list[Dict[str, Any]]:
        target = _fmt_date(as_of_date + timedelta(days=1))
        rows = self._conn.execute(
            """
            SELECT *
            FROM baseline_trades
            WHERE earnings_date = ?
              AND status = 'open'
            ORDER BY entry_date, symbol, baseline_name
            """,
            (target,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def update_exit(
        self,
        *,
        baseline_id: str,
        exit_date: date,
        exit_mid: Optional[float],
        realized_return_pct: Optional[float],
        realized_expansion_pct: Optional[float],
        quote_source_at_exit: Optional[str],
        quote_quality_at_exit: Optional[str],
        exit_bid_ask_mid: Optional[Dict[str, Any]] = None,
        exit_execution_scenarios: Optional[Dict[str, Any]] = None,
        status: str = "resolved",
        skip_reason: Optional[str] = None,
    ) -> bool:
        sql = """
            UPDATE baseline_trades
            SET exit_date = ?,
                exit_mid = ?,
                realized_return_pct = ?,
                realized_expansion_pct = ?,
                quote_source_at_exit = ?,
                quote_quality_at_exit = ?,
                exit_bid_ask_mid_json = ?,
                exit_execution_scenarios_json = ?,
                status = ?,
                skip_reason = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE baseline_id = ?
              AND status IN ('open', 'exit_skipped')
        """
        params = (
            _fmt_date(exit_date),
            exit_mid,
            realized_return_pct,
            realized_expansion_pct,
            quote_source_at_exit,
            quote_quality_at_exit,
            _json(exit_bid_ask_mid or {}),
            _json(exit_execution_scenarios or {}),
            status,
            skip_reason,
            baseline_id,
        )
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                cur.execute(sql, params)
                return cur.rowcount > 0

    def list_for_diagnostics(self, *, limit: int = 10_000) -> list[Dict[str, Any]]:
        capped = max(1, min(int(limit or 10_000), 50_000))
        rows = self._conn.execute(
            """
            SELECT *
            FROM baseline_trades
            ORDER BY COALESCE(exit_date, updated_at, created_at) DESC, created_at DESC
            LIMIT ?
            """,
            (capped,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM baseline_trades").fetchone()[0])

    def close(self) -> None:
        self._conn.close()


def _open_db(path: Path) -> sqlite3.Connection:
    # PR #73 P1/P2 family: shared sqlite_helpers.open_db_conn applies
    # WAL journal mode and sets busy_timeout=5000 explicitly. Pre-fix
    # used SQLite's default DELETE journal, which holds an exclusive
    # file lock per write transaction — concurrent launchd writers
    # would contend on that lock and rows could be lost if the caller
    # swallowed the resulting OperationalError.
    from services.sqlite_helpers import open_db_conn
    conn = open_db_conn(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_TABLE_DDL)
    _migrate(conn)
    conn.executescript(_INDEX_DDL)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(baseline_trades)").fetchall()}
    for column, ddl in _MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE baseline_trades ADD COLUMN {column} {ddl}")


@contextmanager
def _tx(conn: sqlite3.Connection) -> Generator[sqlite3.Cursor, None, None]:
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    result = dict(row)
    for key in (
        "entry_bid_ask_mid_json",
        "exit_bid_ask_mid_json",
        "entry_execution_scenarios_json",
        "exit_execution_scenarios_json",
        "surface_quality_reasons_json",
        "surface_quality_json",
        "metadata_json",
    ):
        if key in result:
            result[key] = _loads(result.get(key))
    return result


def _loads(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _bool_int(value: Optional[bool]) -> Optional[int]:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _fmt_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


_store: Optional[BaselineEvidenceStore] = None
_store_lock = threading.Lock()


def get_baseline_evidence_store(store_path: Optional[Path] = None) -> BaselineEvidenceStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = BaselineEvidenceStore(store_path=store_path or _DEFAULT_STORE)
    return _store
