"""Best-effort provider request telemetry.

Telemetry must never break analysis. All write helpers catch and log failures so
provider observability remains strictly non-invasive.

Row kinds (``request_kind``)
----------------------------
* ``request``  one row per provider call (HTTP request / library fetch). Only
  these rows feed provider failure rates.
* ``quality``  a verdict on a response that already has its own ``request``
  row (e.g. an empty chain payload). Counting it as a failure double-counts
  one call, which inflated the MarketData.app failure rate.
* ``pipeline`` a derived outcome of our own code (e.g. "no usable option
  quote for this structure"), not a provider call.

Rows written before the column existed are classified by endpoint name:
``*_quality`` -> quality (except yfinance's ``options_chain_quality``, which
is that client's only row per chain fetch, so a request) and
``forward_loop_option_quote`` -> pipeline.

``cycle_run_id`` / ``candidate_key`` tie a row to a forward-loop run and the
candidate it served (see :func:`telemetry_context`).
"""

from __future__ import annotations

import contextvars
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE = Path.home() / ".options_calculator_pro" / "telemetry" / "provider_telemetry.sqlite"
_MAX_AGE_DAYS_ENV = "OPTIONS_CALCULATOR_TELEMETRY_MAX_AGE_DAYS"
_MAX_ROWS_ENV = "OPTIONS_CALCULATOR_TELEMETRY_MAX_ROWS"
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_MAX_ROWS = 50_000
_WRITE_LOCK = threading.Lock()

REQUEST_KIND_REQUEST = "request"
REQUEST_KIND_QUALITY = "quality"
REQUEST_KIND_PIPELINE = "pipeline"
REQUEST_KINDS = (REQUEST_KIND_REQUEST, REQUEST_KIND_QUALITY, REQUEST_KIND_PIPELINE)
DEFAULT_FAILURE_WINDOW_DAYS = 7

# Kind of a row, inferring it for rows written before the column existed.
_KIND_SQL = (
    "COALESCE(request_kind, CASE "
    "WHEN endpoint_type LIKE '%\\_quality' ESCAPE '\\' AND provider_name != 'yfinance' THEN 'quality' "
    "WHEN endpoint_type = 'forward_loop_option_quote' THEN 'pipeline' "
    "ELSE 'request' END)"
)

_MIGRATION_COLUMNS = {
    "request_kind": "TEXT",
    "cycle_run_id": "TEXT",
    "candidate_key": "TEXT",
}

_context: contextvars.ContextVar[Dict[str, Optional[str]]] = contextvars.ContextVar(
    "provider_telemetry_context", default={}
)


def bind_candidate(candidate_key: Optional[str]) -> None:
    """Set the candidate for rows written from here on, inside the enclosing
    :func:`telemetry_context` (which restores the previous context on exit)."""
    current = dict(_context.get())
    current["candidate_key"] = candidate_key
    _context.set(current)


@contextmanager
def telemetry_context(*, cycle_run_id: Optional[str] = None, candidate_key: Optional[str] = None) -> Iterator[None]:
    """Tag every telemetry row written inside the block with the run and the
    candidate it serves. Nested blocks inherit unset fields."""
    current = dict(_context.get())
    if cycle_run_id is not None:
        current["cycle_run_id"] = cycle_run_id
    if candidate_key is not None:
        current["candidate_key"] = candidate_key
    token = _context.set(current)
    try:
        yield
    finally:
        _context.reset(token)

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
    response_quality_note  TEXT,
    request_kind           TEXT,
    cycle_run_id           TEXT,
    candidate_key          TEXT
);
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_provider_telemetry_timestamp
    ON provider_telemetry (request_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_provider_telemetry_provider
    ON provider_telemetry (provider_name, endpoint_type);
CREATE INDEX IF NOT EXISTS idx_provider_telemetry_symbol
    ON provider_telemetry (symbol);
CREATE INDEX IF NOT EXISTS idx_provider_telemetry_cycle
    ON provider_telemetry (cycle_run_id);
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
    request_kind: str = REQUEST_KIND_REQUEST
    cycle_run_id: Optional[str] = None
    candidate_key: Optional[str] = None


def _open_db(path: Path) -> sqlite3.Connection:
    # PR #73 P1: route through the shared sqlite_helpers helper so
    # this DB gets WAL journal mode. Pre-fix used SQLite's default
    # DELETE journal, which holds an exclusive file lock for each
    # write transaction — two concurrent launchd jobs calling
    # record_provider_telemetry in the same write window would
    # contend on that lock. WAL removes the exclusive hold and allows
    # one writer to proceed while the other waits on the short commit
    # phase. The caller's exception handling swallows failures
    # (telemetry is best-effort by design), so lost rows from lock
    # contention were invisible without this fix.
    #
    # busy_timeout is set explicitly (5000 ms) for clarity; Python's
    # sqlite3.connect() already defaults to timeout=5.0 s.
    from services.sqlite_helpers import open_db_conn
    conn = open_db_conn(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_TABLE_DDL)
    existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(provider_telemetry)").fetchall()}
    for column, ddl in _MIGRATION_COLUMNS.items():
        if column not in existing:
            try:
                conn.execute(f"ALTER TABLE provider_telemetry ADD COLUMN {column} {ddl}")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
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
            event.request_kind if event.request_kind in REQUEST_KINDS else REQUEST_KIND_REQUEST,
            _clean_text(event.cycle_run_id),
            _clean_text(event.candidate_key),
        )
        with _WRITE_LOCK:
            self._conn.execute(
                """
                INSERT INTO provider_telemetry (
                    request_timestamp, provider_name, endpoint_type, symbol,
                    success, error_category, latency_ms, stale_used, fallback_used,
                    response_quality_note, request_kind, cycle_run_id, candidate_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        request_kind: Optional[str] = REQUEST_KIND_REQUEST,
    ) -> Dict[str, Dict[str, Any]]:
        """Per-provider totals. Failure rates count provider calls only
        (``request_kind="request"``) unless another kind (or None = all) is
        asked for."""
        where_sql, params = _build_filter_clause(
            provider=provider,
            endpoint_type=endpoint_type,
            symbol=symbol,
            success=success,
            since=since,
            until=until,
            request_kind=request_kind,
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
        request_kind: Optional[str] = None,
    ) -> int:
        where_sql, params = _build_filter_clause(
            provider=provider,
            endpoint_type=endpoint_type,
            symbol=symbol,
            success=success,
            since=since,
            until=until,
            request_kind=request_kind,
        )
        return int(self._conn.execute(f"SELECT COUNT(*) FROM provider_telemetry {where_sql}", params).fetchone()[0])

    def failure_breakdown(self, *, since: Optional[str] = None) -> Dict[str, Any]:
        """Failures since ``since``: provider-call failure rates by provider
        and endpoint, failures by error category, and the quality / pipeline
        rows reported apart (they are not provider call failures)."""
        where_sql, params = _build_filter_clause(since=since)
        by_endpoint = self._conn.execute(
            f"""
            SELECT provider_name, endpoint_type, COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failures
            FROM provider_telemetry
            {where_sql}{" AND" if where_sql else " WHERE"} {_KIND_SQL} = ?
            GROUP BY provider_name, endpoint_type
            ORDER BY provider_name, endpoint_type
            """,
            (*params, REQUEST_KIND_REQUEST),
        ).fetchall()
        by_category = self._conn.execute(
            f"""
            SELECT provider_name, COALESCE(error_category, 'unknown_error') AS category, COUNT(*) AS n
            FROM provider_telemetry
            {where_sql}{" AND" if where_sql else " WHERE"} {_KIND_SQL} = ? AND success = 0
            GROUP BY provider_name, category
            ORDER BY provider_name, n DESC, category
            """,
            (*params, REQUEST_KIND_REQUEST),
        ).fetchall()
        by_kind = self._conn.execute(
            f"""
            SELECT {_KIND_SQL} AS kind, COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failures
            FROM provider_telemetry
            {where_sql}
            GROUP BY kind
            """,
            params,
        ).fetchall()

        providers: Dict[str, Dict[str, Any]] = {}
        for row in by_endpoint:
            total, failures = int(row["total"] or 0), int(row["failures"] or 0)
            entry = providers.setdefault(
                str(row["provider_name"]), {"requests": 0, "failures": 0, "endpoints": {}, "error_categories": {}}
            )
            entry["requests"] += total
            entry["failures"] += failures
            entry["endpoints"][str(row["endpoint_type"])] = {
                "requests": total,
                "failures": failures,
                "failure_rate": round(failures / total, 6) if total else 0.0,
            }
        for entry in providers.values():
            entry["failure_rate"] = round(entry["failures"] / entry["requests"], 6) if entry["requests"] else 0.0
        for row in by_category:
            providers.setdefault(
                str(row["provider_name"]), {"requests": 0, "failures": 0, "failure_rate": 0.0,
                                            "endpoints": {}, "error_categories": {}}
            )["error_categories"][str(row["category"])] = int(row["n"])
        kinds = {kind: {"total": 0, "failures": 0} for kind in REQUEST_KINDS}
        for row in by_kind:
            kinds[str(row["kind"])] = {"total": int(row["total"] or 0), "failures": int(row["failures"] or 0)}
        requests = kinds[REQUEST_KIND_REQUEST]
        return {
            "since": since,
            "request_failure_rate": (
                round(requests["failures"] / requests["total"], 6) if requests["total"] else 0.0
            ),
            "providers": providers,
            "by_kind": kinds,
        }

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
    request_kind: str = REQUEST_KIND_REQUEST,
    cycle_run_id: Optional[str] = None,
    candidate_key: Optional[str] = None,
) -> None:
    """Record one row. ``request_kind`` says what the row is (see the module
    docs); ``cycle_run_id`` / ``candidate_key`` default to the active
    :func:`telemetry_context`."""
    try:
        context = _context.get()
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
            request_kind=request_kind,
            cycle_run_id=cycle_run_id or context.get("cycle_run_id"),
            candidate_key=candidate_key or context.get("candidate_key"),
        )
        (store or get_provider_telemetry_store()).record(event)
    except Exception as exc:  # pragma: no cover - telemetry must never interrupt users
        logger.debug("Provider telemetry write failed: %s", exc)


# Categories pass through unchanged, so classifying twice (a caller's
# classify_error, then record_provider_telemetry's) keeps them.
ERROR_CATEGORIES = frozenset({
    "timeout", "rate_limited", "auth", "entitlement", "not_found", "empty_response", "http_error",
    "cache_fallback", "unknown_error", "server_error", "connection_error",
    "provider_budget_exhausted", "provider_circuit_open",
})


def classify_error(value: Any) -> str:
    text = str(value or "").lower()
    if text in ERROR_CATEGORIES:
        return text
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
    # Provider-call totals: quality verdicts and pipeline outcomes are rows
    # too, but not provider calls, so they are counted apart (by_kind).
    totals = {
        "events": sum(item["total"] for item in summary.values()),
        "rows": filtered_count,
        "failures": sum(item["failures"] for item in summary.values()),
        "fallback_count": sum(item["fallback_count"] for item in summary.values()),
        "stale_count": sum(item["stale_count"] for item in summary.values()),
    }
    totals["failure_rate"] = round(totals["failures"] / totals["events"], 6) if totals["events"] else 0.0
    window_since = (datetime.now(timezone.utc) - timedelta(days=DEFAULT_FAILURE_WINDOW_DAYS)).isoformat()
    failure_window = store_obj.failure_breakdown(since=since or window_since)
    failure_window["window_days"] = None if since else DEFAULT_FAILURE_WINDOW_DAYS
    avg_latency_values = [
        item["avg_latency_ms"]
        for item in summary.values()
        if item.get("avg_latency_ms") is not None
    ]
    operational_health = _operational_health(
        totals={**totals, "failure_rate": failure_window["request_failure_rate"]},
        avg_latency_ms=(sum(avg_latency_values) / len(avg_latency_values) if avg_latency_values else None),
        metadata=store_obj.operational_metadata(),
    )
    from services.provider_reliability import reliability_status

    reliability = reliability_status()
    budget_level = (reliability.get("marketdata_budget") or {}).get("alert_level")
    if budget_level in {"warning", "critical", "exhausted"}:
        operational_health["warning_flags"].append(f"MarketData credit budget {budget_level}.")
    for name, breaker in reliability.get("circuit_breakers", {}).items():
        if breaker.get("state") != "closed":
            operational_health["warning_flags"].append(f"{name} circuit breaker is {breaker['state']}.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
        "summary_by_provider": summary,
        "failure_window": failure_window,
        "reliability": reliability,
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
    request_kind: Optional[str] = None,
) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []
    if request_kind:
        clauses.append(f"{_KIND_SQL} = ?")
        params.append(str(request_kind))
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
        warnings.append(
            f"Provider request failure rate over the last {DEFAULT_FAILURE_WINDOW_DAYS} days is elevated."
        )
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
