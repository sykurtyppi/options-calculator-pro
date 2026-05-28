"""Best-effort provider request telemetry.

Telemetry must never break analysis. All write helpers catch and log failures so
provider observability remains strictly non-invasive.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE = Path.home() / ".options_calculator_pro" / "telemetry" / "provider_telemetry.sqlite"
_MAX_AGE_DAYS_ENV = "OPTIONS_CALCULATOR_TELEMETRY_MAX_AGE_DAYS"
_MAX_ROWS_ENV = "OPTIONS_CALCULATOR_TELEMETRY_MAX_ROWS"
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_MAX_ROWS = 50_000
_WRITE_LOCK = threading.Lock()

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS provider_telemetry (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    request_timestamp      TEXT NOT NULL,
    provider_name          TEXT NOT NULL,
    endpoint_type          TEXT NOT NULL,
    symbol                 TEXT,
    success                INTEGER NOT NULL,
    error_category         TEXT,
    latency_ms             REAL,
    stale_used             INTEGER NOT NULL DEFAULT 0,
    fallback_used          INTEGER NOT NULL DEFAULT 0,
    response_quality_note  TEXT
);
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_provider_telemetry_timestamp
    ON provider_telemetry (request_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_provider_telemetry_provider
    ON provider_telemetry (provider_name, endpoint_type);
CREATE INDEX IF NOT EXISTS idx_provider_telemetry_symbol
    ON provider_telemetry (symbol);
"""


@dataclass(frozen=True)
class ProviderTelemetryEvent:
    provider_name: str
    endpoint_type: str
    symbol: Optional[str] = None
    success: bool = True
    error_category: Optional[str] = None
    latency_ms: Optional[float] = None
    stale_used: bool = False
    fallback_used: bool = False
    response_quality_note: Optional[str] = None
    request_timestamp: str = ""


def _open_db(path: Path) -> sqlite3.Connection:
    # PR #73 P1: route through the shared sqlite_helpers helper
    # that adds busy_timeout=5000. Pre-fix this had WAL but NO
    # busy_timeout — so two concurrent writers (two launchd jobs
    # both calling record_provider_telemetry within the same write
    # window) would race on the WAL writer lock and the loser
    # silently dropped its row with SQLITE_BUSY. The caller's
    # exception handling swallowed the failure (telemetry is
    # best-effort by design), but the rows were gone.
    #
    # With busy_timeout=5000 the second writer waits up to 5s for
    # the first to finish, which is more than enough for a
    # single-row INSERT.
    from services.sqlite_helpers import open_db_conn
    conn = open_db_conn(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_TABLE_DDL)
    conn.executescript(_INDEX_DDL)
    conn.commit()
    return conn


class ProviderTelemetryStore:
    def __init__(self, store_path: Path = _DEFAULT_STORE) -> None:
        self._path = store_path
        self._conn = _open_db(store_path)

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: ProviderTelemetryEvent) -> None:
        ts = event.request_timestamp or datetime.now(timezone.utc).isoformat()
        params = (
            ts,
            str(event.provider_name or "unknown"),
            str(event.endpoint_type or "unknown"),
            str(event.symbol).upper() if event.symbol else None,
            1 if event.success else 0,
            _clean_text(event.error_category),
            float(event.latency_ms) if event.latency_ms is not None else None,
            1 if event.stale_used else 0,
            1 if event.fallback_used else 0,
            _clean_text(event.response_quality_note),
        )
        with _WRITE_LOCK:
            self._conn.execute(
                """
                INSERT INTO provider_telemetry (
                    request_timestamp, provider_name, endpoint_type, symbol,
                    success, error_category, latency_ms, stale_used, fallback_used,
                    response_quality_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
            self._conn.commit()
        self._best_effort_retention_cleanup()

    def recent(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        failures_only: bool = False,
        provider: Optional[str] = None,
        endpoint_type: Optional[str] = None,
        symbol: Optional[str] = None,
        success: Optional[bool] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        capped = max(1, min(int(limit or 100), 1_000))
        safe_offset = max(0, int(offset or 0))
        where_sql, params = _build_filter_clause(
            provider=provider,
            endpoint_type=endpoint_type,
            symbol=symbol,
            success=False if failures_only else success,
            since=since,
            until=until,
        )
        rows = self._conn.execute(
            f"""
            SELECT * FROM provider_telemetry
            {where_sql}
            ORDER BY request_timestamp DESC, id DESC
            LIMIT ?
            OFFSET ?
            """,
            (*params, capped, safe_offset),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def summary_by_provider(
        self,
        *,
        provider: Optional[str] = None,
        endpoint_type: Optional[str] = None,
        symbol: Optional[str] = None,
        success: Optional[bool] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        where_sql, params = _build_filter_clause(
            provider=provider,
            endpoint_type=endpoint_type,
            symbol=symbol,
            success=success,
            since=since,
            until=until,
        )
        rows = self._conn.execute(
            f"""
            SELECT
                provider_name,
                COUNT(*) AS total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failures,
                AVG(latency_ms) AS avg_latency_ms,
                SUM(CASE WHEN fallback_used = 1 THEN 1 ELSE 0 END) AS fallback_count,
                SUM(CASE WHEN stale_used = 1 THEN 1 ELSE 0 END) AS stale_count
            FROM provider_telemetry
            {where_sql}
            GROUP BY provider_name
            ORDER BY total DESC, provider_name
            """,
            params,
        ).fetchall()
        summary: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            total = int(row["total"] or 0)
            failures = int(row["failures"] or 0)
            summary[str(row["provider_name"])] = {
                "total": total,
                "successes": int(row["successes"] or 0),
                "failures": failures,
                "failure_rate": round(failures / total, 6) if total else 0.0,
                "avg_latency_ms": round(float(row["avg_latency_ms"]), 3) if row["avg_latency_ms"] is not None else None,
                "fallback_count": int(row["fallback_count"] or 0),
                "stale_count": int(row["stale_count"] or 0),
            }
        return summary

    def count(
        self,
        *,
        provider: Optional[str] = None,
        endpoint_type: Optional[str] = None,
        symbol: Optional[str] = None,
        success: Optional[bool] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> int:
        where_sql, params = _build_filter_clause(
            provider=provider,
            endpoint_type=endpoint_type,
            symbol=symbol,
            success=success,
            since=since,
            until=until,
        )
        return int(self._conn.execute(f"SELECT COUNT(*) FROM provider_telemetry {where_sql}", params).fetchone()[0])

    def safe_cleanup(
        self,
        *,
        max_age_days: Optional[int] = None,
        max_rows: Optional[int] = None,
    ) -> Dict[str, int]:
        age_days = _resolve_max_age_days(max_age_days)
        row_cap = _resolve_max_rows(max_rows)
        deleted_by_age = 0
        deleted_overflow = 0
        with _WRITE_LOCK:
            if age_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=age_days)
                cur = self._conn.execute(
                    "DELETE FROM provider_telemetry WHERE request_timestamp < ?",
                    (cutoff.isoformat(),),
                )
                deleted_by_age = int(cur.rowcount or 0)
            if row_cap > 0:
                total = self.count()
                overflow = max(0, total - row_cap)
                if overflow > 0:
                    cur = self._conn.execute(
                        """
                        DELETE FROM provider_telemetry
                        WHERE id IN (
                            SELECT id
                            FROM provider_telemetry
                            ORDER BY request_timestamp ASC, id ASC
                            LIMIT ?
                        )
                        """,
                        (overflow,),
                    )
                    deleted_overflow = int(cur.rowcount or 0)
            self._conn.commit()
        return {
            "deleted_by_age": deleted_by_age,
            "deleted_overflow": deleted_overflow,
            "remaining_rows": self.count(),
        }

    def operational_metadata(self) -> Dict[str, Any]:
        size_bytes = self._path.stat().st_size if self._path.exists() else 0
        return {
            "db_path": str(self._path),
            "db_size_bytes": int(size_bytes),
            "row_count": self.count(),
            "retention": {
                "max_age_days": _resolve_max_age_days(None),
                "max_rows": _resolve_max_rows(None),
            },
        }

    def _best_effort_retention_cleanup(self) -> None:
        try:
            self.safe_cleanup()
        except Exception as exc:  # pragma: no cover - telemetry cleanup must never interrupt users
            logger.debug("Provider telemetry retention cleanup failed: %s", exc)

    def close(self) -> None:
        self._conn.close()


def record_provider_telemetry(
    *,
    provider_name: str,
    endpoint_type: str,
    symbol: Optional[str] = None,
    success: bool = True,
    error_category: Optional[str] = None,
    latency_ms: Optional[float] = None,
    stale_used: bool = False,
    fallback_used: bool = False,
    response_quality_note: Optional[str] = None,
    store: Optional[ProviderTelemetryStore] = None,
) -> None:
    try:
        event = ProviderTelemetryEvent(
            provider_name=provider_name,
            endpoint_type=endpoint_type,
            symbol=symbol,
            success=success,
            error_category=classify_error(error_category) if error_category else None,
            latency_ms=latency_ms,
            stale_used=stale_used,
            fallback_used=fallback_used,
            response_quality_note=response_quality_note,
        )
        (store or get_provider_telemetry_store()).record(event)
    except Exception as exc:  # pragma: no cover - telemetry must never interrupt users
        logger.debug("Provider telemetry write failed: %s", exc)


def classify_error(value: Any) -> str:
    text = str(value or "").lower()
    if "timeout" in text:
        return "timeout"
    if "rate" in text or "429" in text:
        return "rate_limited"
    if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
        return "auth"
    if "402" in text or "payment" in text or "entitlement" in text:
        return "entitlement"
    if "404" in text or "not found" in text:
        return "not_found"
    if "empty" in text or "no data" in text:
        return "empty_response"
    if "http" in text:
        return "http_error"
    if "cache" in text:
        return "cache_fallback"
    return "unknown_error"


def build_provider_telemetry_diagnostics(
    *,
    store: Optional[ProviderTelemetryStore] = None,
    limit: int = 100,
    offset: int = 0,
    provider: Optional[str] = None,
    endpoint_type: Optional[str] = None,
    symbol: Optional[str] = None,
    success: Optional[bool] = None,
    failures_only: bool = False,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> Dict[str, Any]:
    store_obj = store or get_provider_telemetry_store()
    recent_events = store_obj.recent(
        limit=limit,
        offset=offset,
        provider=provider,
        endpoint_type=endpoint_type,
        symbol=symbol,
        success=success,
        failures_only=failures_only,
        since=since,
        until=until,
    )
    recent_failures = store_obj.recent(
        limit=min(int(limit or 100), 100),
        offset=0,
        provider=provider,
        endpoint_type=endpoint_type,
        symbol=symbol,
        failures_only=True,
        since=since,
        until=until,
    )
    summary = store_obj.summary_by_provider(
        provider=provider,
        endpoint_type=endpoint_type,
        symbol=symbol,
        success=False if failures_only else success,
        since=since,
        until=until,
    )
    filtered_count = store_obj.count(
        provider=provider,
        endpoint_type=endpoint_type,
        symbol=symbol,
        success=False if failures_only else success,
        since=since,
        until=until,
    )
    totals = {
        "events": filtered_count,
        "failures": sum(item["failures"] for item in summary.values()),
        "fallback_count": sum(item["fallback_count"] for item in summary.values()),
        "stale_count": sum(item["stale_count"] for item in summary.values()),
    }
    totals["failure_rate"] = round(totals["failures"] / totals["events"], 6) if totals["events"] else 0.0
    avg_latency_values = [
        item["avg_latency_ms"]
        for item in summary.values()
        if item.get("avg_latency_ms") is not None
    ]
    operational_health = _operational_health(
        totals=totals,
        avg_latency_ms=(sum(avg_latency_values) / len(avg_latency_values) if avg_latency_values else None),
        metadata=store_obj.operational_metadata(),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
        "summary_by_provider": summary,
        "recent_events": recent_events,
        "recent_failures": recent_failures,
        "limit": max(1, min(int(limit or 100), 1_000)),
        "offset": max(0, int(offset or 0)),
        "has_more": max(0, int(offset or 0)) + len(recent_events) < filtered_count,
        "filters": {
            "provider": provider,
            "endpoint_type": endpoint_type,
            "symbol": symbol.upper() if symbol else None,
            "success": success,
            "failures_only": failures_only,
            "since": since,
            "until": until,
        },
        "operational_health": operational_health,
    }


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    payload = dict(row)
    payload["success"] = bool(payload.get("success"))
    payload["stale_used"] = bool(payload.get("stale_used"))
    payload["fallback_used"] = bool(payload.get("fallback_used"))
    return payload


def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).replace("\n", " ").replace("\r", " ")[:240]


def _build_filter_clause(
    *,
    provider: Optional[str] = None,
    endpoint_type: Optional[str] = None,
    symbol: Optional[str] = None,
    success: Optional[bool] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []
    if provider:
        clauses.append("provider_name = ?")
        params.append(str(provider))
    if endpoint_type:
        clauses.append("endpoint_type = ?")
        params.append(str(endpoint_type))
    if symbol:
        clauses.append("symbol = ?")
        params.append(str(symbol).upper())
    if success is not None:
        clauses.append("success = ?")
        params.append(1 if success else 0)
    if since:
        clauses.append("request_timestamp >= ?")
        params.append(str(since))
    if until:
        clauses.append("request_timestamp <= ?")
        params.append(str(until))
    return ("WHERE " + " AND ".join(clauses), tuple(params)) if clauses else ("", tuple())


def _operational_health(*, totals: Dict[str, Any], avg_latency_ms: Optional[float], metadata: Dict[str, Any]) -> Dict[str, Any]:
    warnings: list[str] = []
    if totals.get("events", 0) == 0:
        warnings.append("No provider telemetry events recorded yet.")
    if totals.get("failure_rate", 0.0) >= 0.20:
        warnings.append("Provider failure rate is elevated.")
    if totals.get("fallback_count", 0) > 0:
        warnings.append("Fallback provider paths were used.")
    if totals.get("stale_count", 0) > 0:
        warnings.append("Stale provider/cache data was used.")
    if avg_latency_ms is not None and avg_latency_ms >= 2_000:
        warnings.append("Average provider latency is elevated.")
    return {
        "warning_flags": warnings,
        "avg_latency_ms": round(float(avg_latency_ms), 3) if avg_latency_ms is not None else None,
        "db_size_bytes": metadata["db_size_bytes"],
        "row_count": metadata["row_count"],
        "retention": metadata["retention"],
    }


def _resolve_max_age_days(value: Optional[int]) -> int:
    if value is not None:
        return max(0, int(value))
    try:
        return max(0, int(os.getenv(_MAX_AGE_DAYS_ENV, str(DEFAULT_MAX_AGE_DAYS))))
    except ValueError:
        return DEFAULT_MAX_AGE_DAYS


def _resolve_max_rows(value: Optional[int]) -> int:
    if value is not None:
        return max(0, int(value))
    try:
        return max(0, int(os.getenv(_MAX_ROWS_ENV, str(DEFAULT_MAX_ROWS))))
    except ValueError:
        return DEFAULT_MAX_ROWS


_store: Optional[ProviderTelemetryStore] = None
_store_lock = threading.Lock()


def get_provider_telemetry_store(store_path: Optional[Path] = None) -> ProviderTelemetryStore:
    global _store
    if _store is None or store_path is not None:
        with _store_lock:
            if _store is None or store_path is not None:
                _store = ProviderTelemetryStore(store_path or _DEFAULT_STORE)
    return _store
