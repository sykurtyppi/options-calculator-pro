"""
Outcome recorder — durable trade journal for the earnings-volatility decision engine.

Provides a SQLite-backed store that records:
  1. Trade entry state — decision context, market state, structure pricing
  2. Trade exit outcome — realized return, realized IV expansion
  3. Feedback dispatch — updates calibration and structure priors on finalization

Storage
-------
~/.options_calculator_pro/outcomes/outcome_store.sqlite

Schema is append-safe: entries are INSERTed (with NULL exit fields),
exit fields are UPDATEd, and finalized is the terminal state.

Duplicate trade_id on INSERT is a silent no-op (INSERT OR IGNORE), making
repeated seed runs idempotent.

Thread safety
-------------
All writes acquire a module-level threading.Lock.
Read-only methods (get_trade, exists, diagnostics) do not acquire it.

Unit conventions
----------------
All percentage fields (realized_return_pct, realized_expansion_pct,
expected_edge_pct, etc.) are stored in "percent" format: 9.0 = 9%.
This is consistent with calibration_service which uses the same convention.
If your data source uses fractions (0.09 = 9%), multiply by 100 before
calling record_trade_entry / finalize_trade_and_update_learning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE = (
    Path.home() / ".options_calculator_pro" / "outcomes" / "outcome_store.sqlite"
)
_WRITE_LOCK = threading.Lock()

# ── Schema ─────────────────────────────────────────────────────────────────────

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS outcome_trades (
    -- Identity / timing
    trade_id                        TEXT PRIMARY KEY,
    symbol                          TEXT NOT NULL,
    structure                       TEXT NOT NULL,
    release_timing                  TEXT,
    entry_date                      TEXT NOT NULL,
    exit_date                       TEXT,
    earnings_date                   TEXT,
    as_of_date_at_entry             TEXT,
    recommendation_id               TEXT,

    -- Decision state at entry (captured from SelectorOutput)
    selector_recommendation         TEXT,
    selector_confidence_pct         REAL,
    setup_score                     REAL NOT NULL,
    expected_edge_pct               REAL,
    expected_return_pct             REAL,
    best_structure_at_entry         TEXT,
    runner_up_structure_at_entry    TEXT,
    data_quality_score_at_entry     REAL,

    -- Market state at entry (captured from VolSnapshot)
    days_to_earnings                INTEGER,
    iv_rv_yz                        REAL,
    iv_rv_har                       REAL,
    historical_vs_implied_move_ratio REAL,
    term_structure_slope            REAL,
    near_term_spread_pct            REAL,
    liquidity_tier                  TEXT,
    calibration_phase_at_entry      TEXT,

    -- Structure pricing / trade fields
    entry_mid                       REAL,
    exit_mid                        REAL,
    realized_return_pct             REAL,   -- net after costs, in % (9.0 = 9%)
    realized_pnl                    REAL,   -- dollar P&L
    realized_expansion_pct          REAL,   -- gross option value change, in %
    execution_penalty_at_entry      REAL,
    assumed_cost_model              TEXT,
    evidence_quality_status         TEXT,
    evidence_quality_reasons_json   TEXT,
    claim_allowed                   INTEGER,
    execution_grade                 INTEGER,
    entry_quote_source              TEXT,
    entry_quote_quality             TEXT,
    entry_quote_timestamp           TEXT,
    entry_bid_ask_mid_json          TEXT,
    entry_execution_scenarios_json  TEXT,
    exit_quote_source               TEXT,
    exit_quote_quality              TEXT,
    exit_quote_timestamp            TEXT,
    exit_bid_ask_mid_json           TEXT,
    exit_execution_scenarios_json   TEXT,
    surface_quality_status          TEXT,
    surface_quality_reasons_json    TEXT,
    surface_quality_json            TEXT,
    surface_crossed_quote_count     INTEGER,
    surface_zero_bid_count          INTEGER,
    surface_extreme_spread_count    INTEGER,
    surface_sparse_atm_count        INTEGER,
    surface_iv_anomaly_count        INTEGER,

    -- Meta / provenance
    source_type                     TEXT NOT NULL DEFAULT 'paper',
    snapshot_hash                   TEXT,
    notes                           TEXT,

    -- Lifecycle
    status                          TEXT NOT NULL DEFAULT 'open',
    created_at                      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_outcome_symbol_date  ON outcome_trades (symbol, entry_date);
CREATE INDEX IF NOT EXISTS idx_outcome_structure     ON outcome_trades (structure);
CREATE INDEX IF NOT EXISTS idx_outcome_source_type   ON outcome_trades (source_type);
CREATE INDEX IF NOT EXISTS idx_outcome_status        ON outcome_trades (status);
CREATE INDEX IF NOT EXISTS idx_outcome_recommendation_id ON outcome_trades (recommendation_id);
"""

_MIGRATION_COLUMNS: Dict[str, str] = {
    "recommendation_id": "TEXT",
    "evidence_quality_status": "TEXT",
    "evidence_quality_reasons_json": "TEXT",
    "claim_allowed": "INTEGER",
    "execution_grade": "INTEGER",
    "entry_quote_source": "TEXT",
    "entry_quote_quality": "TEXT",
    "entry_quote_timestamp": "TEXT",
    "entry_bid_ask_mid_json": "TEXT",
    "entry_execution_scenarios_json": "TEXT",
    "exit_quote_source": "TEXT",
    "exit_quote_quality": "TEXT",
    "exit_quote_timestamp": "TEXT",
    "exit_bid_ask_mid_json": "TEXT",
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

# ── DB helpers ─────────────────────────────────────────────────────────────────


def _open_db(store_path: Path) -> sqlite3.Connection:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_TABLE_DDL)
    _migrate_schema(conn)
    conn.executescript(_INDEX_DDL)
    conn.commit()
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(outcome_trades)").fetchall()}
    for column, ddl_type in _MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE outcome_trades ADD COLUMN {column} {ddl_type}")


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


# ── OutcomeStore ───────────────────────────────────────────────────────────────


class OutcomeStore:
    """
    Durable trade journal.

    Parameters
    ----------
    store_path : Path, optional
        Where to persist the SQLite file.
    """

    def __init__(self, store_path: Path = _DEFAULT_STORE) -> None:
        self._path = store_path
        self._conn = _open_db(store_path)
        logger.info("outcome_store: opened at %s", store_path)

    # ── Insert entry ──────────────────────────────────────────────────────────

    def insert_entry(
        self,
        *,
        trade_id: str,
        symbol: str,
        structure: str,
        entry_date: date,
        setup_score: float,
        source_type: str,
        # Optional fields follow — all default to None / not set
        recommendation_id: Optional[str] = None,
        release_timing: Optional[str] = None,
        earnings_date: Optional[date] = None,
        as_of_date_at_entry: Optional[date] = None,
        selector_recommendation: Optional[str] = None,
        selector_confidence_pct: Optional[float] = None,
        expected_edge_pct: Optional[float] = None,
        expected_return_pct: Optional[float] = None,
        best_structure_at_entry: Optional[str] = None,
        runner_up_structure_at_entry: Optional[str] = None,
        data_quality_score_at_entry: Optional[float] = None,
        days_to_earnings: Optional[int] = None,
        iv_rv_yz: Optional[float] = None,
        iv_rv_har: Optional[float] = None,
        historical_vs_implied_move_ratio: Optional[float] = None,
        term_structure_slope: Optional[float] = None,
        near_term_spread_pct: Optional[float] = None,
        liquidity_tier: Optional[str] = None,
        calibration_phase_at_entry: Optional[str] = None,
        entry_mid: Optional[float] = None,
        execution_penalty_at_entry: Optional[float] = None,
        assumed_cost_model: Optional[str] = None,
        evidence_quality_status: Optional[str] = None,
        evidence_quality_reasons: Optional[list[str]] = None,
        claim_allowed: Optional[bool] = None,
        execution_grade: Optional[bool] = None,
        entry_quote_source: Optional[str] = None,
        entry_quote_quality: Optional[str] = None,
        entry_quote_timestamp: Optional[str] = None,
        entry_bid_ask_mid: Optional[Dict[str, Any]] = None,
        entry_execution_scenarios: Optional[Dict[str, Any]] = None,
        surface_quality: Optional[Dict[str, Any]] = None,
        snapshot_hash: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Insert a new trade entry record.

        Returns True on successful insert, False if trade_id already exists
        (the existing row is left untouched — safe to call repeatedly).
        """
        sql = """
            INSERT OR IGNORE INTO outcome_trades (
                trade_id, symbol, structure, release_timing, recommendation_id,
                entry_date, earnings_date, as_of_date_at_entry,
                selector_recommendation, selector_confidence_pct,
                setup_score, expected_edge_pct, expected_return_pct,
                best_structure_at_entry, runner_up_structure_at_entry,
                data_quality_score_at_entry,
                days_to_earnings, iv_rv_yz, iv_rv_har,
                historical_vs_implied_move_ratio, term_structure_slope,
                near_term_spread_pct, liquidity_tier, calibration_phase_at_entry,
                entry_mid, execution_penalty_at_entry, assumed_cost_model,
                evidence_quality_status, evidence_quality_reasons_json,
                claim_allowed, execution_grade,
                entry_quote_source, entry_quote_quality, entry_quote_timestamp,
                entry_bid_ask_mid_json, entry_execution_scenarios_json,
                surface_quality_status, surface_quality_reasons_json, surface_quality_json,
                surface_crossed_quote_count, surface_zero_bid_count,
                surface_extreme_spread_count, surface_sparse_atm_count,
                surface_iv_anomaly_count,
                source_type, snapshot_hash, notes, status
            ) VALUES (
                ?,?,?,?,?,
                ?,?,?,
                ?,?,
                ?,?,?,
                ?,?,
                ?,
                ?,?,?,
                ?,?,
                ?,?,?,
                ?,?,?,
                ?,?,
                ?,?,
                ?,?,?,
                ?,?,
                ?,?,?,
                ?,?,
                ?,?,?,
                ?,?,?,'open'
            )
        """
        surface_quality = surface_quality or {}
        params = (
            trade_id, symbol, structure, release_timing,
            recommendation_id,
            _fmt_date(entry_date), _fmt_date(earnings_date), _fmt_date(as_of_date_at_entry),
            selector_recommendation, selector_confidence_pct,
            float(setup_score), expected_edge_pct, expected_return_pct,
            best_structure_at_entry, runner_up_structure_at_entry,
            data_quality_score_at_entry,
            days_to_earnings, iv_rv_yz, iv_rv_har,
            historical_vs_implied_move_ratio, term_structure_slope,
            near_term_spread_pct, liquidity_tier, calibration_phase_at_entry,
            entry_mid, execution_penalty_at_entry, assumed_cost_model,
            evidence_quality_status, _json_payload(evidence_quality_reasons or []),
            _bool_int(claim_allowed), _bool_int(execution_grade),
            entry_quote_source, entry_quote_quality, entry_quote_timestamp,
            _json_payload(entry_bid_ask_mid or {}),
            _json_payload(entry_execution_scenarios or {}),
            surface_quality.get("status"),
            _json_payload(surface_quality.get("warning_flags") or []),
            _json_payload(surface_quality),
            int(surface_quality.get("crossed_quote_count") or 0),
            int(surface_quality.get("zero_bid_count") or 0),
            int(surface_quality.get("extreme_spread_count") or 0),
            int(surface_quality.get("sparse_atm_expiration_count") or 0),
            int(surface_quality.get("missing_iv_count") or 0) + int(surface_quality.get("iv_outlier_count") or 0),
            source_type, snapshot_hash, notes,
        )
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                cur.execute(sql, params)
                inserted = cur.rowcount > 0
        if inserted:
            logger.debug("outcome_store: inserted entry trade_id=%s", trade_id)
        else:
            logger.debug("outcome_store: skipped duplicate trade_id=%s", trade_id)
        return inserted

    # ── Update exit ───────────────────────────────────────────────────────────

    def update_exit(
        self,
        *,
        trade_id: str,
        exit_date: date,
        exit_mid: Optional[float] = None,
        realized_return_pct: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        realized_expansion_pct: Optional[float] = None,
        exit_quote_source: Optional[str] = None,
        exit_quote_quality: Optional[str] = None,
        exit_quote_timestamp: Optional[str] = None,
        exit_bid_ask_mid: Optional[Dict[str, Any]] = None,
        exit_execution_scenarios: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update exit fields on an open trade.

        Sets status = 'exited'.  Does NOT trigger learning updates —
        call finalize_trade_and_update_learning() for that.

        Returns True if a row was updated, False if trade_id not found
        or already finalized.
        """
        sql = """
            UPDATE outcome_trades
            SET exit_date              = ?,
                exit_mid               = ?,
                realized_return_pct    = ?,
                realized_pnl           = ?,
                realized_expansion_pct = ?,
                exit_quote_source       = ?,
                exit_quote_quality      = ?,
                exit_quote_timestamp    = ?,
                exit_bid_ask_mid_json   = ?,
                exit_execution_scenarios_json = ?,
                status                 = 'exited',
                updated_at             = CURRENT_TIMESTAMP
            WHERE trade_id = ?
              AND status IN ('open', 'exited')
        """
        params = (
            _fmt_date(exit_date),
            exit_mid,
            realized_return_pct,
            realized_pnl,
            realized_expansion_pct,
            exit_quote_source,
            exit_quote_quality,
            exit_quote_timestamp,
            _json_payload(exit_bid_ask_mid or {}),
            _json_payload(exit_execution_scenarios or {}),
            trade_id,
        )
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                cur.execute(sql, params)
                updated = cur.rowcount > 0
        if updated:
            logger.debug("outcome_store: updated exit trade_id=%s", trade_id)
        else:
            logger.warning(
                "outcome_store: no open/exited row found for trade_id=%s", trade_id
            )
        return updated

    # ── Mark finalized ────────────────────────────────────────────────────────

    def mark_finalized(self, trade_id: str) -> bool:
        """
        Mark a trade as finalized (learning updates already applied externally).

        Returns True if status changed, False if not found or already finalized.
        """
        sql = """
            UPDATE outcome_trades
            SET status     = 'finalized',
                updated_at = CURRENT_TIMESTAMP
            WHERE trade_id = ?
              AND status != 'finalized'
        """
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                cur.execute(sql, (trade_id,))
                return cur.rowcount > 0

    # ── Read helpers ──────────────────────────────────────────────────────────

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Return the full trade record as a dict, or None if not found."""
        cur = self._conn.execute(
            "SELECT * FROM outcome_trades WHERE trade_id = ?", (trade_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def exists(self, trade_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM outcome_trades WHERE trade_id = ?", (trade_id,)
        )
        return cur.fetchone() is not None

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM outcome_trades"
        ).fetchone()[0]

    def count_by_source(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT source_type, COUNT(*) n FROM outcome_trades GROUP BY source_type"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def count_by_structure(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT structure, COUNT(*) n FROM outcome_trades GROUP BY structure"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def count_finalized(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM outcome_trades WHERE status = 'finalized'"
        ).fetchone()[0]

    def find_active_trade_for_event(
        self,
        *,
        symbol: str,
        structure: str,
        earnings_date: date,
    ) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM outcome_trades
            WHERE symbol = ?
              AND structure = ?
              AND earnings_date = ?
              AND status != 'finalized'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (str(symbol).upper(), str(structure), _fmt_date(earnings_date)),
        ).fetchone()
        return dict(row) if row else None

    def find_by_recommendation_id(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT *
            FROM outcome_trades
            WHERE recommendation_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (recommendation_id,),
        ).fetchone()
        return dict(row) if row else None

    def trades_due_for_exit(self, as_of_date: date) -> list[Dict[str, Any]]:
        target_earnings_date = _fmt_date(as_of_date + timedelta(days=1))
        rows = self._conn.execute(
            """
            SELECT *
            FROM outcome_trades
            WHERE earnings_date = ?
              AND status IN ('open', 'exited')
            ORDER BY entry_date, symbol
            """,
            (target_earnings_date,),
        ).fetchall()
        return [dict(row) for row in rows]

    def diagnostics(self) -> Dict[str, Any]:
        total = self.count()
        return {
            "total": total,
            "by_source": self.count_by_source(),
            "by_status": dict(
                self._conn.execute(
                    "SELECT status, COUNT(*) n FROM outcome_trades GROUP BY status"
                ).fetchall()
            ),
            "by_structure": self.count_by_structure(),
            "store_path": str(self._path),
        }

    def list_for_diagnostics(self, *, limit: int = 10_000) -> list[Dict[str, Any]]:
        """Return recent outcome rows for read-only diagnostics aggregation."""
        capped_limit = max(1, min(int(limit or 10_000), 50_000))
        rows = self._conn.execute(
            """
            SELECT *
            FROM outcome_trades
            ORDER BY COALESCE(exit_date, updated_at, created_at) DESC, created_at DESC
            LIMIT ?
            """,
            (capped_limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()


# ── Utility functions ──────────────────────────────────────────────────────────


def make_trade_id(symbol: str, entry_date: date, structure: str) -> str:
    """
    Generate a deterministic, stable trade_id from the three identity fields.

    The same inputs always produce the same ID, which makes repeated calls
    to seed_outcomes_from_replay idempotent (INSERT OR IGNORE catches dupes).

    Format: ``{SYMBOL}|{YYYY-MM-DD}|{structure}``
    """
    return f"{symbol.upper()}|{_fmt_date(entry_date)}|{structure}"


def make_snapshot_hash(snapshot_dict: Dict[str, Any]) -> str:
    """
    Stable SHA-256 prefix (first 16 hex chars) of a snapshot dict.

    Useful for provenance: two calls with the same snapshot produce the same
    hash, so you can detect whether the decision context changed between runs.
    """
    canonical = json.dumps(snapshot_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _fmt_date(d: Any) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)


def _json_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _bool_int(value: Optional[bool]) -> Optional[int]:
    if value is None:
        return None
    return 1 if bool(value) else 0


# ── High-level workflow functions ──────────────────────────────────────────────


def record_trade_entry(
    *,
    symbol: str,
    structure: str,
    entry_date: date,
    setup_score: float,
    source_type: str,
    store: Optional[OutcomeStore] = None,
    **kwargs: Any,
) -> str:
    """
    Record a new trade entry.  Returns the trade_id.

    Parameters
    ----------
    symbol, structure, entry_date, setup_score, source_type : required
    **kwargs : any additional insert_entry() keyword arguments
    store : OutcomeStore, optional — uses singleton if not provided

    Returns
    -------
    trade_id : str
        Stable identifier for this trade.
    """
    trade_id = kwargs.pop("trade_id", make_trade_id(symbol, entry_date, structure))
    s = store or get_outcome_store()
    s.insert_entry(
        trade_id=trade_id,
        symbol=symbol,
        structure=structure,
        entry_date=entry_date,
        setup_score=setup_score,
        source_type=source_type,
        **kwargs,
    )
    return trade_id


def record_trade_exit(
    *,
    trade_id: str,
    exit_date: date,
    exit_mid: Optional[float] = None,
    realized_return_pct: Optional[float] = None,
    realized_pnl: Optional[float] = None,
    realized_expansion_pct: Optional[float] = None,
    exit_quote_source: Optional[str] = None,
    exit_quote_quality: Optional[str] = None,
    exit_quote_timestamp: Optional[str] = None,
    exit_bid_ask_mid: Optional[Dict[str, Any]] = None,
    exit_execution_scenarios: Optional[Dict[str, Any]] = None,
    store: Optional[OutcomeStore] = None,
) -> bool:
    """
    Record exit prices and outcome for an open trade.

    Does not trigger learning updates — call finalize_trade_and_update_learning()
    when you are ready to commit this trade to calibration and structure priors.
    """
    s = store or get_outcome_store()
    return s.update_exit(
        trade_id=trade_id,
        exit_date=exit_date,
        exit_mid=exit_mid,
        realized_return_pct=realized_return_pct,
        realized_pnl=realized_pnl,
        realized_expansion_pct=realized_expansion_pct,
        exit_quote_source=exit_quote_source,
        exit_quote_quality=exit_quote_quality,
        exit_quote_timestamp=exit_quote_timestamp,
        exit_bid_ask_mid=exit_bid_ask_mid,
        exit_execution_scenarios=exit_execution_scenarios,
    )


def finalize_trade_and_update_learning(
    *,
    trade_id: str,
    exit_date: Optional[date] = None,
    exit_mid: Optional[float] = None,
    realized_return_pct: float,
    realized_pnl: Optional[float] = None,
    realized_expansion_pct: float,
    exit_quote_source: Optional[str] = None,
    exit_quote_quality: Optional[str] = None,
    exit_quote_timestamp: Optional[str] = None,
    exit_bid_ask_mid: Optional[Dict[str, Any]] = None,
    exit_execution_scenarios: Optional[Dict[str, Any]] = None,
    store: Optional[OutcomeStore] = None,
    source_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Finalize a trade and dispatch learning updates to calibration and priors.

    This is the single function to call once a trade outcome is known.
    It coordinates all four update steps:
      1. Update exit fields in the outcome store
      2. Update IVExpansionCalibration
      3. Update StructurePriorStore
      4. Mark the trade as finalized
      5. Invalidate the walk-forward prior cache so the scorecard picks
         up updated priors on the next call

    Parameters
    ----------
    trade_id : str
    realized_return_pct : float
        Net return after costs, in % (9.0 = 9%).  Positive = profitable.
    realized_expansion_pct : float
        Gross option value change from entry to exit, in %.
        Positive = IV expansion occurred.  Negative = crush.
    exit_date / exit_mid / realized_pnl : optional metadata

    Returns
    -------
    dict with keys:
        trade_id, structure, setup_score,
        calibration_n_before, calibration_n_after, calibration_phase,
        prior_observation_count, prior_win_rate,
        status  (always "finalized"),
        warnings  (list of any non-fatal issues)
    """
    s = store or get_outcome_store()
    warnings: list = []

    # ── 1. Fetch the entry record ─────────────────────────────────────────────
    row = s.get_trade(trade_id)
    if row is None:
        raise ValueError(f"finalize_trade_and_update_learning: trade_id={trade_id!r} not found")

    structure = row["structure"]
    setup_score = float(row["setup_score"])
    _source_type = source_type or row.get("source_type", "paper")

    # ── 2. Update exit fields ─────────────────────────────────────────────────
    if exit_date is not None or exit_mid is not None:
        s.update_exit(
            trade_id=trade_id,
            exit_date=exit_date or date.today(),
            exit_mid=exit_mid,
            realized_return_pct=realized_return_pct,
            realized_pnl=realized_pnl,
            realized_expansion_pct=realized_expansion_pct,
            exit_quote_source=exit_quote_source,
            exit_quote_quality=exit_quote_quality,
            exit_quote_timestamp=exit_quote_timestamp,
            exit_bid_ask_mid=exit_bid_ask_mid,
            exit_execution_scenarios=exit_execution_scenarios,
        )

    # ── 3. Update calibration ─────────────────────────────────────────────────
    try:
        from services.calibration_service import get_calibration

        cal = get_calibration()
        n_before = cal._n()
        calibration_recorded = cal.update(
            setup_score,
            realized_expansion_pct,
            observation_id=trade_id,
            source_type=_source_type,
            observation_date=exit_date or date.today(),
        )
        n_after = cal._n()
        cal_phase = cal._phase()
        if not calibration_recorded:
            warnings.append(
                f"calibration observation already present for trade_id={trade_id}"
            )
    except Exception as exc:
        logger.error("finalize: calibration update failed (%s)", exc)
        warnings.append(f"calibration update failed: {exc}")
        n_before = n_after = 0
        cal_phase = "unknown"

    # ── 4. Update structure prior ─────────────────────────────────────────────
    prior_obs_count = 0
    prior_win_rate: Optional[float] = None
    try:
        from services.structure_prior_store import get_structure_prior_store

        ps = get_structure_prior_store()
        ps.update(
            structure=structure,
            realized_return_pct=realized_return_pct,
            realized_expansion_pct=realized_expansion_pct,
            source_type=_source_type,
            observation_date=exit_date or date.today(),
            observation_id=trade_id,
        )
        diag = ps.diagnostics()
        struct_entry = diag["structures"].get(structure, {})
        prior_obs_count = struct_entry.get("observation_count", 0)
        prior_win_rate = struct_entry.get("win_rate")
    except Exception as exc:
        logger.error("finalize: structure prior update failed (%s)", exc)
        warnings.append(f"structure prior update failed: {exc}")

    # ── 5. Invalidate scorecard cache ─────────────────────────────────────────
    try:
        from services.structure_scorecard import reload_walk_forward_priors

        reload_walk_forward_priors()
    except Exception as exc:
        warnings.append(f"scorecard cache invalidation failed: {exc}")

    # ── 6. Mark finalized ─────────────────────────────────────────────────────
    s.mark_finalized(trade_id)

    return {
        "trade_id": trade_id,
        "structure": structure,
        "setup_score": round(setup_score, 4),
        "realized_return_pct": round(realized_return_pct, 4),
        "realized_expansion_pct": round(realized_expansion_pct, 4),
        "calibration_n_before": n_before,
        "calibration_n_after": n_after,
        "calibration_phase": cal_phase,
        "prior_observation_count": prior_obs_count,
        "prior_win_rate": round(prior_win_rate, 4) if prior_win_rate is not None else None,
        "status": "finalized",
        "warnings": warnings,
    }


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[OutcomeStore] = None
_store_lock = threading.Lock()


def get_outcome_store(store_path: Optional[Path] = None) -> OutcomeStore:
    """Return the process-level singleton OutcomeStore."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = OutcomeStore(store_path=store_path or _DEFAULT_STORE)
    return _store
