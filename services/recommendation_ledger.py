"""Durable recommendation ledger for selector auditability.

The ledger records the exact evidence surface that produced a recommendation:
VolSnapshot, structure scorecards, selector output, provider/quote provenance,
and schema/version metadata. It is intentionally append-safe and local-first so
research runs, API analyses, and forward paper trades can be audited later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
ENGINE_VERSION = "event_vol_selector_v1"
_DEFAULT_LEDGER = Path.home() / ".options_calculator_pro" / "recommendations" / "recommendation_ledger.sqlite"
_WRITE_LOCK = threading.Lock()

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id             TEXT PRIMARY KEY,
    created_at                    TEXT NOT NULL,
    symbol                        TEXT NOT NULL,
    as_of_date                    TEXT,
    earnings_date                 TEXT,
    earnings_source               TEXT,
    earnings_source_confidence    REAL,
    earnings_source_stale         INTEGER NOT NULL DEFAULT 0,
    recommendation                TEXT,
    selected_structure            TEXT,
    no_trade_reason               TEXT,
    data_quality_score            REAL,
    option_source                 TEXT,
    underlying_source             TEXT,
    provider_names_json           TEXT,
    quote_timestamp               TEXT,
    quote_source                  TEXT,
    quote_quality                 TEXT,
    bid_ask_mid_json              TEXT,
    surface_quality_status        TEXT,
    surface_quality_reasons_json  TEXT,
    surface_quality_json          TEXT,
    surface_crossed_quote_count   INTEGER,
    surface_zero_bid_count        INTEGER,
    surface_extreme_spread_count  INTEGER,
    surface_sparse_atm_count      INTEGER,
    surface_iv_anomaly_count      INTEGER,
    vol_snapshot_json             TEXT NOT NULL,
    structure_scorecards_json     TEXT NOT NULL,
    selector_output_json          TEXT NOT NULL,
    explanation_json              TEXT,
    metadata_json                 TEXT,
    schema_version                INTEGER NOT NULL,
    engine_version                TEXT NOT NULL
);
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_recommendations_symbol_date
    ON recommendations (symbol, as_of_date);
CREATE INDEX IF NOT EXISTS idx_recommendations_earnings_date
    ON recommendations (earnings_date);
CREATE INDEX IF NOT EXISTS idx_recommendations_selected_structure
    ON recommendations (selected_structure);
CREATE INDEX IF NOT EXISTS idx_recommendations_recommendation
    ON recommendations (recommendation);
"""

_MIGRATION_COLUMNS: Dict[str, str] = {
    "created_at": "TEXT",
    "symbol": "TEXT",
    "as_of_date": "TEXT",
    "earnings_date": "TEXT",
    "earnings_source": "TEXT",
    "earnings_source_confidence": "REAL",
    "earnings_source_stale": "INTEGER NOT NULL DEFAULT 0",
    "recommendation": "TEXT",
    "selected_structure": "TEXT",
    "no_trade_reason": "TEXT",
    "data_quality_score": "REAL",
    "option_source": "TEXT",
    "underlying_source": "TEXT",
    "provider_names_json": "TEXT",
    "quote_timestamp": "TEXT",
    "quote_source": "TEXT",
    "quote_quality": "TEXT",
    "bid_ask_mid_json": "TEXT",
    "surface_quality_status": "TEXT",
    "surface_quality_reasons_json": "TEXT",
    "surface_quality_json": "TEXT",
    "surface_crossed_quote_count": "INTEGER",
    "surface_zero_bid_count": "INTEGER",
    "surface_extreme_spread_count": "INTEGER",
    "surface_sparse_atm_count": "INTEGER",
    "surface_iv_anomaly_count": "INTEGER",
    "vol_snapshot_json": "TEXT",
    "structure_scorecards_json": "TEXT",
    "selector_output_json": "TEXT",
    "explanation_json": "TEXT",
    "metadata_json": "TEXT",
    "schema_version": "INTEGER NOT NULL DEFAULT 1",
    "engine_version": "TEXT NOT NULL DEFAULT 'event_vol_selector_v1'",
}


@dataclass(frozen=True)
class RecommendationRecord:
    recommendation_id: str
    created_at: str
    symbol: str
    as_of_date: Optional[str]
    earnings_date: Optional[str]
    earnings_source: Optional[str]
    earnings_source_confidence: Optional[float]
    earnings_source_stale: bool
    recommendation: Optional[str]
    selected_structure: Optional[str]
    no_trade_reason: Optional[str]
    data_quality_score: Optional[float]
    option_source: Optional[str]
    underlying_source: Optional[str]
    provider_names: Dict[str, Any] = field(default_factory=dict)
    quote_timestamp: Optional[str] = None
    quote_source: Optional[str] = None
    quote_quality: Optional[str] = None
    bid_ask_mid: Dict[str, Any] = field(default_factory=dict)
    surface_quality: Dict[str, Any] = field(default_factory=dict)
    vol_snapshot: Dict[str, Any] = field(default_factory=dict)
    structure_scorecards: list[Dict[str, Any]] = field(default_factory=list)
    selector_output: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    engine_version: str = ENGINE_VERSION


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_TABLE_DDL)
    _migrate(conn)
    conn.executescript(_INDEX_DDL)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(recommendations)").fetchall()
    existing = {str(row["name"]) for row in rows}
    for column, ddl_type in _MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE recommendations ADD COLUMN {column} {ddl_type}")


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


class RecommendationLedger:
    def __init__(self, ledger_path: Path = _DEFAULT_LEDGER) -> None:
        self._path = ledger_path
        self._conn = _open_db(ledger_path)

    @property
    def path(self) -> Path:
        return self._path

    def record(self, record: RecommendationRecord) -> bool:
        sql = """
            INSERT OR IGNORE INTO recommendations (
                recommendation_id, created_at, symbol, as_of_date, earnings_date,
                earnings_source, earnings_source_confidence, earnings_source_stale,
                recommendation, selected_structure, no_trade_reason, data_quality_score,
                option_source, underlying_source, provider_names_json,
                quote_timestamp, quote_source, quote_quality, bid_ask_mid_json,
                surface_quality_status, surface_quality_reasons_json, surface_quality_json,
                surface_crossed_quote_count, surface_zero_bid_count,
                surface_extreme_spread_count, surface_sparse_atm_count,
                surface_iv_anomaly_count,
                vol_snapshot_json, structure_scorecards_json, selector_output_json,
                explanation_json, metadata_json, schema_version, engine_version
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            ON CONFLICT(recommendation_id) DO UPDATE SET
                created_at = excluded.created_at,
                symbol = excluded.symbol,
                as_of_date = excluded.as_of_date,
                earnings_date = excluded.earnings_date,
                earnings_source = excluded.earnings_source,
                earnings_source_confidence = excluded.earnings_source_confidence,
                earnings_source_stale = excluded.earnings_source_stale,
                recommendation = excluded.recommendation,
                selected_structure = excluded.selected_structure,
                no_trade_reason = excluded.no_trade_reason,
                data_quality_score = excluded.data_quality_score,
                option_source = excluded.option_source,
                underlying_source = excluded.underlying_source,
                provider_names_json = excluded.provider_names_json,
                quote_timestamp = excluded.quote_timestamp,
                quote_source = excluded.quote_source,
                quote_quality = excluded.quote_quality,
                bid_ask_mid_json = excluded.bid_ask_mid_json,
                surface_quality_status = excluded.surface_quality_status,
                surface_quality_reasons_json = excluded.surface_quality_reasons_json,
                surface_quality_json = excluded.surface_quality_json,
                surface_crossed_quote_count = excluded.surface_crossed_quote_count,
                surface_zero_bid_count = excluded.surface_zero_bid_count,
                surface_extreme_spread_count = excluded.surface_extreme_spread_count,
                surface_sparse_atm_count = excluded.surface_sparse_atm_count,
                surface_iv_anomaly_count = excluded.surface_iv_anomaly_count,
                vol_snapshot_json = excluded.vol_snapshot_json,
                structure_scorecards_json = excluded.structure_scorecards_json,
                selector_output_json = excluded.selector_output_json,
                explanation_json = excluded.explanation_json,
                metadata_json = excluded.metadata_json,
                schema_version = excluded.schema_version,
                engine_version = excluded.engine_version
        """
        params = (
            record.recommendation_id,
            record.created_at,
            record.symbol.upper(),
            record.as_of_date,
            record.earnings_date,
            record.earnings_source,
            record.earnings_source_confidence,
            1 if record.earnings_source_stale else 0,
            record.recommendation,
            record.selected_structure,
            record.no_trade_reason,
            record.data_quality_score,
            record.option_source,
            record.underlying_source,
            _json(record.provider_names),
            record.quote_timestamp,
            record.quote_source,
            record.quote_quality,
            _json(record.bid_ask_mid),
            record.surface_quality.get("status"),
            _json(record.surface_quality.get("warning_flags") or []),
            _json(record.surface_quality),
            int(record.surface_quality.get("crossed_quote_count") or 0),
            int(record.surface_quality.get("zero_bid_count") or 0),
            int(record.surface_quality.get("extreme_spread_count") or 0),
            int(record.surface_quality.get("sparse_atm_expiration_count") or 0),
            int(record.surface_quality.get("missing_iv_count") or 0) + int(record.surface_quality.get("iv_outlier_count") or 0),
            _json(record.vol_snapshot),
            _json(record.structure_scorecards),
            _json(record.selector_output),
            _json(record.explanation),
            _json(record.metadata),
            int(record.schema_version),
            record.engine_version,
        )
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                cur.execute(sql, params)
                return cur.rowcount > 0

    def get(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM recommendations WHERE recommendation_id = ?",
            (recommendation_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def list_recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        symbol: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        capped_limit = max(1, min(int(limit or 50), 500))
        safe_offset = max(0, int(offset or 0))
        if symbol:
            rows = self._conn.execute(
                """
                SELECT *
                FROM recommendations
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
                OFFSET ?
                """,
                (str(symbol).upper(), capped_limit, safe_offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT *
                FROM recommendations
                ORDER BY created_at DESC
                LIMIT ?
                OFFSET ?
                """,
                (capped_limit, safe_offset),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_for_diagnostics(self, *, limit: int = 10_000) -> list[Dict[str, Any]]:
        capped_limit = max(1, min(int(limit or 10_000), 50_000))
        rows = self._conn.execute(
            """
            SELECT *
            FROM recommendations
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (capped_limit,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def summarize(self) -> Dict[str, Any]:
        def _counts(column: str) -> Dict[str, int]:
            rows = self._conn.execute(
                f"""
                SELECT COALESCE(NULLIF({column}, ''), 'unknown') AS key, COUNT(*) AS n
                FROM recommendations
                GROUP BY COALESCE(NULLIF({column}, ''), 'unknown')
                ORDER BY n DESC, key
                """
            ).fetchall()
            return {str(row["key"]): int(row["n"]) for row in rows}

        stale_rows = self._conn.execute(
            """
            SELECT earnings_source_stale AS stale, COUNT(*) AS n
            FROM recommendations
            GROUP BY earnings_source_stale
            """
        ).fetchall()
        stale_counts = {
            "stale": sum(int(row["n"]) for row in stale_rows if int(row["stale"] or 0) == 1),
            "fresh_or_unknown": sum(int(row["n"]) for row in stale_rows if int(row["stale"] or 0) == 0),
        }
        return {
            "total": self.count(),
            "by_selected_structure": _counts("selected_structure"),
            "by_no_trade_reason": _counts("no_trade_reason"),
            "by_earnings_source": _counts("earnings_source"),
            "by_recommendation": _counts("recommendation"),
            "by_stale_source_flag": stale_counts,
        }

    def count(self, *, symbol: Optional[str] = None) -> int:
        if symbol:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM recommendations WHERE symbol = ?",
                    (str(symbol).upper(),),
                ).fetchone()[0]
            )
        return int(self._conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0])

    def close(self) -> None:
        self._conn.close()


def make_recommendation_id(
    *,
    symbol: str,
    as_of_date: Any,
    earnings_date: Any,
    selected_structure: Any,
    salt: Optional[str] = None,
) -> str:
    """Stable when salt is omitted; unique when a timestamp/nonce salt is passed."""
    base = "|".join(
        [
            str(symbol or "").upper(),
            _fmt_date(as_of_date) or "unknown_asof",
            _fmt_date(earnings_date) or "unknown_earnings",
            str(selected_structure or "no_structure"),
            str(salt or ""),
        ]
    )
    return "rec_" + hashlib.sha256(base.encode()).hexdigest()[:20]


def build_record_from_analysis(
    analysis: Any,
    *,
    recommendation_id: Optional[str] = None,
    quote_payload: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RecommendationRecord:
    vol_snapshot = _as_dict(_get(analysis, "vol_snapshot", {}) or {})
    selector_output = _as_dict(_get(analysis, "selector_output", {}) or {})
    scorecards = [_as_dict(card) for card in (_get(analysis, "structure_scorecards", []) or [])]
    metrics = _as_dict(_get(analysis, "metrics", {}) or {})
    quote = quote_payload or {}
    surface_quality = quote.get("surface_quality") or vol_snapshot.get("surface_quality") or {}
    created_at = datetime.now(timezone.utc).isoformat()

    symbol = str(vol_snapshot.get("symbol") or _get(analysis, "symbol") or "").upper()
    as_of_date = vol_snapshot.get("as_of_date") or selector_output.get("as_of")
    earnings_date = vol_snapshot.get("earnings_date") or selector_output.get("earnings_date")
    selected_structure = selector_output.get("best_structure")
    recommendation = selector_output.get("recommendation") or _get(analysis, "recommendation")
    if recommendation_id is None:
        recommendation_id = make_recommendation_id(
            symbol=symbol,
            as_of_date=as_of_date,
            earnings_date=earnings_date,
            selected_structure=selected_structure,
            salt=created_at,
        )

    provider_names = {
        "option_source": vol_snapshot.get("option_source") or (metrics.get("data_sources") or {}).get("options_source"),
        "underlying_source": vol_snapshot.get("underlying_source") or (metrics.get("data_sources") or {}).get("price_rv_source"),
        "earnings_source": vol_snapshot.get("earnings_source_primary"),
        "quote_source": quote.get("quote_source"),
    }
    explanation = {
        "primary_thesis": selector_output.get("primary_thesis"),
        "primary_risks": selector_output.get("primary_risks") or [],
        "why_this_structure": selector_output.get("why_this_structure") or [],
        "why_not_others": selector_output.get("why_not_others") or {},
        "rationale": _get(analysis, "rationale", []) or [],
    }

    no_trade_reason = None
    if str(recommendation or "") == "No Trade":
        reasons = selector_output.get("primary_risks") or selector_output.get("why_this_structure") or []
        no_trade_reason = "; ".join(str(item) for item in reasons[:3]) or "selector_abstained"

    return RecommendationRecord(
        recommendation_id=recommendation_id,
        created_at=created_at,
        symbol=symbol,
        as_of_date=_fmt_date(as_of_date),
        earnings_date=_fmt_date(earnings_date),
        earnings_source=vol_snapshot.get("earnings_source_primary"),
        earnings_source_confidence=_safe_float_or_none(vol_snapshot.get("earnings_source_confidence")),
        earnings_source_stale=bool(vol_snapshot.get("earnings_source_stale", False)),
        recommendation=str(recommendation) if recommendation is not None else None,
        selected_structure=str(selected_structure) if selected_structure is not None else None,
        no_trade_reason=no_trade_reason,
        data_quality_score=_safe_float_or_none(vol_snapshot.get("data_quality_score")),
        option_source=provider_names["option_source"],
        underlying_source=provider_names["underlying_source"],
        provider_names=provider_names,
        quote_timestamp=quote.get("quote_timestamp"),
        quote_source=quote.get("quote_source"),
        quote_quality=quote.get("quote_quality"),
        bid_ask_mid=quote.get("bid_ask_mid") or quote.get("legs") or {},
        surface_quality=surface_quality,
        vol_snapshot=vol_snapshot,
        structure_scorecards=scorecards,
        selector_output=selector_output,
        explanation=explanation,
        metadata={
            "schema_source": "build_record_from_analysis",
            **(metadata or {}),
        },
    )


def record_recommendation(
    analysis: Any,
    *,
    ledger: Optional[RecommendationLedger] = None,
    recommendation_id: Optional[str] = None,
    quote_payload: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    record = build_record_from_analysis(
        analysis,
        recommendation_id=recommendation_id,
        quote_payload=quote_payload,
        metadata=metadata,
    )
    (ledger or get_recommendation_ledger()).record(record)
    return record.recommendation_id


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    payload = dict(row)
    for key in (
        "provider_names_json",
        "bid_ask_mid_json",
        "surface_quality_reasons_json",
        "surface_quality_json",
        "vol_snapshot_json",
        "structure_scorecards_json",
        "selector_output_json",
        "explanation_json",
        "metadata_json",
    ):
        payload[key] = _loads(payload.get(key))
    payload["earnings_source_stale"] = bool(payload.get("earnings_source_stale"))
    return payload


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, default=str)


def _loads(value: Any) -> Any:
    if value in (None, ""):
        return {}
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {}


def _safe_float_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else None
    except Exception:
        return None


def _fmt_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10] if len(str(value)) >= 10 else str(value)


_ledger: Optional[RecommendationLedger] = None
_ledger_lock = threading.Lock()


def get_recommendation_ledger(ledger_path: Optional[Path] = None) -> RecommendationLedger:
    global _ledger
    if _ledger is None or ledger_path is not None:
        with _ledger_lock:
            if _ledger is None or ledger_path is not None:
                _ledger = RecommendationLedger(ledger_path or _DEFAULT_LEDGER)
    return _ledger
