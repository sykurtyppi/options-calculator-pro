"""Shadow baseline evidence store for forward paper comparisons.

This store records hypothetical baseline trades beside selector paper trades.
It is intentionally separate from OutcomeStore so baseline rows never update
calibration or structure priors.

Two cohorts share the table and must never be pooled:

* ``paired`` - entered on the same day as a selector paper trade, so it answers
  "did the selector's pick beat the naive structure on the SAME events?".
* ``universe`` - entered once per eligible earnings event (first day inside the
  DTE window) regardless of what the selector said, so it answers "what did
  every event pay, including the ones the selector passed on?". Without it the
  selector's No Trade calls have no counterfactual and can never be graded.
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
    # Short-vol control. Exit is T-1 (the day before earnings), so this measures
    # shorting the pre-earnings IV run-up, NOT selling the event crush.
    "always_iron_condor": "iron_condor",
}

COHORT_PAIRED = "paired"
COHORT_UNIVERSE = "universe"

# How the exit quote was obtained. Only EXIT_REPRICING_BOOKED rows are
# comparable: every leg of the exit quote was checked against the contracts
# stored at entry. Everything else is reported separately:
# * rediscovered_legacy - entered before the entry context was stored, so the
#   exit re-discovered strikes (a different contract than the one bought).
# * booked_strikes - the #143 label, applied WITHOUT checking the exit legs;
#   an ATM straddle exit could pair a different call and put strike.
# * unverifiable_entry_context - see EXIT_REPRICING_UNVERIFIABLE.
EXIT_REPRICING_BOOKED = "booked_entry_contracts"
EXIT_REPRICING_LEGACY = "rediscovered_legacy"
EXIT_REPRICING_UNVERIFIED = "booked_strikes"
# Priced from a stored entry context that lacks a leg strike or the expiry
# (e.g. straddles entered before the put strike was recorded separately), so
# the exit cannot be proven to be the same position.
EXIT_REPRICING_UNVERIFIABLE = "unverifiable_entry_context"

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
CREATE INDEX IF NOT EXISTS idx_baseline_universe_event ON baseline_trades (cohort, symbol, earnings_date);
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
    "cohort": "TEXT",
    "entry_pricing_context_json": "TEXT",
    "capital_at_risk": "REAL",
    "exit_repricing": "TEXT",
    "selector_recommendation": "TEXT",
    "days_to_earnings_at_entry": "INTEGER",
    "entry_attempt_count": "INTEGER",
    "failed_entry_attempts_json": "TEXT",
}


def make_baseline_id(recommendation_id: str, baseline_name: str) -> str:
    return f"{recommendation_id}|baseline|{baseline_name}"


def make_universe_baseline_id(symbol: str, earnings_date: Any, baseline_name: str) -> str:
    # Keyed on the EVENT, not the daily recommendation id, so an event is entered
    # exactly once no matter how many days it sits inside the DTE window.
    return f"universe|{str(symbol).upper()}|{_fmt_date(earnings_date)}|{baseline_name}"


def baseline_cohort(row: Dict[str, Any]) -> str:
    # Rows written before cohorts existed were all paired with a selector entry.
    return str(row.get("cohort") or COHORT_PAIRED)


def is_booked_strike_exit(row: Dict[str, Any]) -> bool:
    return str(row.get("exit_repricing") or "") == EXIT_REPRICING_BOOKED


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
        cohort: str = COHORT_PAIRED,
        baseline_id: Optional[str] = None,
        entry_pricing_context: Optional[Dict[str, Any]] = None,
        capital_at_risk: Optional[float] = None,
        selector_recommendation: Optional[str] = None,
        days_to_earnings_at_entry: Optional[int] = None,
    ) -> bool:
        baseline_id = baseline_id or make_baseline_id(recommendation_id, baseline_name)
        sql = """
            INSERT INTO baseline_trades (
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
                status, skip_reason, metadata_json,
                cohort, entry_pricing_context_json, capital_at_risk,
                selector_recommendation, days_to_earnings_at_entry,
                entry_attempt_count, failed_entry_attempts_json
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?
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
            cohort,
            _json(entry_pricing_context) if entry_pricing_context else None,
            capital_at_risk,
            selector_recommendation,
            int(days_to_earnings_at_entry) if days_to_earnings_at_entry is not None else None,
        )
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                prior = cur.execute(
                    "SELECT status, skip_reason, entry_date, entry_attempt_count, failed_entry_attempts_json"
                    " FROM baseline_trades WHERE baseline_id = ?",
                    (baseline_id,),
                ).fetchone()
                failed_attempts: list[Dict[str, Any]] = []
                attempt_count = 1
                if prior is not None:
                    # Only a failed universe entry may be retried. Anything that
                    # entered keeps its first entry; paired rows never retry.
                    if cohort != COHORT_UNIVERSE or str(prior["status"]) != "entry_skipped":
                        return False
                    failed_attempts = _loads_list(prior["failed_entry_attempts_json"])
                    if not failed_attempts:
                        # Rows skipped before attempts were logged: the row
                        # itself is the only record of the first attempt.
                        failed_attempts = [{"date": prior["entry_date"], "reason": prior["skip_reason"]}]
                    attempt_count = int(prior["entry_attempt_count"] or len(failed_attempts)) + 1
                    cur.execute("DELETE FROM baseline_trades WHERE baseline_id = ?", (baseline_id,))
                if status == "entry_skipped":
                    failed_attempts = failed_attempts + [{"date": _fmt_date(entry_date), "reason": skip_reason}]
                cur.execute(sql, params + (attempt_count, _json(failed_attempts)))
                return cur.rowcount > 0

    def recorded_universe_baselines(self, symbol: str, earnings_date: Any) -> set[str]:
        """Baseline names already ENTERED for this event in the universe cohort.

        Checked BEFORE quoting so an event that stays in the DTE window for ten
        days costs one set of quotes, not ten. A failed entry (entry_skipped) is
        not counted, so it is retried on the event's next day in the window;
        otherwise a one-day provider outage would silently drop the event and
        the resolved sample would be selected by first-day data availability.
        """
        rows = self._conn.execute(
            """
            SELECT baseline_name
            FROM baseline_trades
            WHERE cohort = ? AND symbol = ? AND earnings_date = ?
              AND status != 'entry_skipped'
            """,
            (COHORT_UNIVERSE, str(symbol).upper(), _fmt_date(earnings_date)),
        ).fetchall()
        return {str(row["baseline_name"]) for row in rows}

    def baselines_due_for_exit(self, as_of_date: date) -> list[Dict[str, Any]]:
        target = _fmt_date(as_of_date + timedelta(days=1))
        rows = self._conn.execute(
            """
            SELECT *
            FROM baseline_trades
            WHERE earnings_date = ?
              AND (
                status = 'open'
                -- A failed exit is retried if the loop runs again on the
                -- same T-1 day; never on a later day (different horizon).
                OR (status = 'exit_skipped' AND exit_date = ?)
              )
            ORDER BY entry_date, symbol, baseline_name
            """,
            (target, _fmt_date(as_of_date)),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def mark_missing_exits(self, as_of_date: date) -> list[Dict[str, Any]]:
        """Move open baselines whose T-1 exit day has passed to 'exit_missing'.

        Covers rows whose exit was never attempted (e.g. the loop did not run
        on T-1). No outcome is fabricated.
        """
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                due = cur.execute(
                    """
                    SELECT baseline_id, symbol, baseline_name, structure, earnings_date
                    FROM baseline_trades
                    WHERE status = 'open'
                      AND earnings_date IS NOT NULL
                      AND earnings_date <= ?
                    """,
                    (_fmt_date(as_of_date),),
                ).fetchall()
                cur.execute(
                    """
                    UPDATE baseline_trades
                    SET status = 'exit_missing',
                        skip_reason = 'no_exit_attempt_recorded',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'open'
                      AND earnings_date IS NOT NULL
                      AND earnings_date <= ?
                    """,
                    (_fmt_date(as_of_date),),
                )
        return [dict(row) for row in due]

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
        exit_repricing: Optional[str] = None,
    ) -> bool:
        sql = """
            UPDATE baseline_trades
            SET exit_date = ?,
                exit_repricing = ?,
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
            exit_repricing,
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
        "entry_pricing_context_json",
    ):
        if key in result:
            result[key] = _loads(result.get(key))
    if "failed_entry_attempts_json" in result:
        result["failed_entry_attempts_json"] = _loads_list(result.get("failed_entry_attempts_json"))
    return result


def _loads(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _loads_list(value: Any) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


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
