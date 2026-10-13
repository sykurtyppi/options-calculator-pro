"""
Evidence funnel: where each forward candidate stopped, and why.

The forward loop reports daily counters (discovered, analyzed, recorded, ...)
but most drop points left no per-candidate trace: the DTE window filter, an
analyzer exception, a no-trade gate. This store records one row per candidate
per stage per cycle, with an outcome and a reason from a fixed vocabulary, so
throughput can be measured stage by stage:

    discovered -> data_qualified -> analyzed -> eligible -> recorded
               -> exit_due -> resolved -> claimable

Outcomes:
    advanced  the candidate passed this stage
    dropped   it stopped here for good (reason says why)
    deferred  it stopped for now and will be looked at again (e.g. too early)
    error     the stage failed (reason names the error type)

This is measurement only: recording never changes what the loop trades, and
a recording failure never interrupts the loop (callers use ``safe_record``).

Module state:
    ``_store`` / ``_store_lock``: lazily created default store.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE = Path.home() / ".options_calculator_pro" / "evidence" / "evidence_funnel.sqlite"

STAGE_DISCOVERED = "discovered"
STAGE_DATA_QUALIFIED = "data_qualified"
STAGE_ANALYZED = "analyzed"
STAGE_ELIGIBLE = "eligible"
STAGE_RECORDED = "recorded"
STAGE_EXIT_DUE = "exit_due"
STAGE_RESOLVED = "resolved"
STAGE_CLAIMABLE = "claimable"
STAGES = (
    STAGE_DISCOVERED,
    STAGE_DATA_QUALIFIED,
    STAGE_ANALYZED,
    STAGE_ELIGIBLE,
    STAGE_RECORDED,
    STAGE_EXIT_DUE,
    STAGE_RESOLVED,
    STAGE_CLAIMABLE,
)

OUTCOME_ADVANCED = "advanced"
OUTCOME_DROPPED = "dropped"
OUTCOME_DEFERRED = "deferred"
OUTCOME_ERROR = "error"
OUTCOMES = (OUTCOME_ADVANCED, OUTCOME_DROPPED, OUTCOME_DEFERRED, OUTCOME_ERROR)

# Reason vocabulary. Parameterised reasons append ":<detail>".
REASON_MISSING_SYMBOL = "missing_symbol"
REASON_MISSING_DTE = "missing_dte"
REASON_TOO_EARLY = "outside_dte_window:too_early"
REASON_TOO_LATE = "outside_dte_window:too_late"
REASON_ANALYZER_ERROR = "analyzer_error"          # :<ExceptionType>
REASON_NO_TRADE = "no_trade"                      # :<binding_gate>
REASON_NOT_ACTIONABLE = "not_actionable"          # :<recommendation>
REASON_NO_STRUCTURE = "no_structure"
REASON_NO_EARNINGS_DATE = "no_earnings_date"
REASON_DUPLICATE_ACTIVE_TRADE = "duplicate_active_trade"
REASON_DUPLICATE_TRADE_ID = "duplicate_trade_id"
REASON_MISSING_ENTRY_QUOTE = "missing_entry_quote"  # :<quote reason>
REASON_EXIT_FAILED = "exit_failed"                # :<reason>
REASON_EXIT_MISSING = "exit_missing"              # :<reason>
REASON_NOT_CLAIMABLE = "not_claimable"            # :<evidence quality reasons>

_DDL = """
CREATE TABLE IF NOT EXISTS funnel_cycles (
    cycle_run_id      TEXT PRIMARY KEY,
    as_of_date        TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    universe_size     INTEGER,
    discovered        INTEGER,
    discovery_source  TEXT
);
CREATE TABLE IF NOT EXISTS funnel_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_run_id       TEXT NOT NULL,
    as_of_date         TEXT NOT NULL,
    stage              TEXT NOT NULL,
    outcome            TEXT NOT NULL,
    subject            TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    earnings_date      TEXT,
    reason             TEXT,
    recommendation_id  TEXT,
    trade_id           TEXT,
    detail_json        TEXT,
    recorded_at        TEXT NOT NULL,
    UNIQUE(cycle_run_id, stage, subject)
);
CREATE INDEX IF NOT EXISTS idx_funnel_events_as_of ON funnel_events (as_of_date, stage);
CREATE INDEX IF NOT EXISTS idx_funnel_events_symbol ON funnel_events (symbol, earnings_date);
"""


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return str(value)[:10]


def new_cycle_run_id(as_of: date) -> str:
    """A unique id for one forward-loop run (several runs per day are distinct)."""
    import uuid

    return f"{as_of.isoformat()}-{uuid.uuid4().hex[:10]}"


class EvidenceFunnelStore:
    def __init__(self, store_path: Path = _DEFAULT_STORE) -> None:
        from services.sqlite_helpers import open_db_conn

        self._path = Path(store_path)
        self._lock = threading.Lock()
        self._conn = open_db_conn(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._conn.close()

    def start_cycle(
        self,
        *,
        cycle_run_id: str,
        as_of: date,
        universe_size: Optional[int],
        discovered: int,
        discovery_source: Optional[str],
    ) -> None:
        """One row per run: how many symbols were scanned and how many surfaced.

        universe_size - discovered is what discovery dropped (no upcoming
        earnings in the window, screener errors or timeouts)."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO funnel_cycles
                   (cycle_run_id, as_of_date, started_at, universe_size, discovered, discovery_source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cycle_run_id, _iso(as_of), datetime.now(timezone.utc).isoformat(),
                 None if universe_size is None else int(universe_size), int(discovered), discovery_source),
            )
            self._conn.commit()

    def record(
        self,
        *,
        cycle_run_id: str,
        as_of: date,
        stage: str,
        outcome: str,
        symbol: str,
        reason: Optional[str] = None,
        earnings_date: Any = None,
        recommendation_id: Optional[str] = None,
        trade_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown funnel stage: {stage}")
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown funnel outcome: {outcome}")
        if outcome != OUTCOME_ADVANCED and not reason:
            raise ValueError("a non-advancing funnel event needs a reason")
        # A trade is the subject once one exists (exit stages), else the symbol.
        subject = str(trade_id) if trade_id else str(symbol).upper()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO funnel_events
                   (cycle_run_id, as_of_date, stage, outcome, subject, symbol, earnings_date, reason,
                    recommendation_id, trade_id, detail_json, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cycle_run_id, _iso(as_of), stage, outcome, subject, str(symbol).upper(), _iso(earnings_date),
                 reason, recommendation_id, trade_id,
                 json.dumps(detail, default=str, sort_keys=True) if detail else None,
                 datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def events(self, cycle_run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM funnel_events"
        params: list = []
        if cycle_run_id is not None:
            sql += " WHERE cycle_run_id = ?"
            params.append(cycle_run_id)
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql + " ORDER BY id", params).fetchall()]

    def summarize(self, *, since: Optional[date] = None, until: Optional[date] = None) -> Dict[str, Any]:
        """Stage counts, drop reasons and stage-to-stage conversion over a window."""
        clauses, params = [], []
        if since is not None:
            clauses.append("as_of_date >= ?")
            params.append(_iso(since))
        if until is not None:
            clauses.append("as_of_date <= ?")
            params.append(_iso(until))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            stage_rows = self._conn.execute(
                f"SELECT stage, outcome, COUNT(*) AS n FROM funnel_events{where} GROUP BY stage, outcome",
                params,
            ).fetchall()
            reason_rows = self._conn.execute(
                f"SELECT stage, reason, COUNT(*) AS n FROM funnel_events{where}"
                f"{' AND' if where else ' WHERE'} outcome != ? GROUP BY stage, reason",
                params + [OUTCOME_ADVANCED],
            ).fetchall()
            cycle_rows = self._conn.execute(
                f"SELECT COUNT(*) AS cycles, SUM(universe_size) AS scanned, SUM(discovered) AS discovered"
                f" FROM funnel_cycles{where}",
                params,
            ).fetchone()

        stages = {stage: {outcome: 0 for outcome in OUTCOMES} for stage in STAGES}
        for row in stage_rows:
            stages[row["stage"]][row["outcome"]] = int(row["n"])
        for counts in stages.values():
            counts["total"] = sum(counts[outcome] for outcome in OUTCOMES)
        drop_reasons: Dict[str, Dict[str, int]] = {}
        for row in reason_rows:
            drop_reasons.setdefault(row["stage"], {})[str(row["reason"])] = int(row["n"])
        for stage, reasons in drop_reasons.items():
            drop_reasons[stage] = dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))

        conversion = {}
        for upstream, downstream in zip(STAGES, STAGES[1:]):
            reached = stages[upstream]["advanced"]
            conversion[f"{upstream}->{downstream}"] = (
                None if reached == 0 else round(stages[downstream]["advanced"] / reached, 4)
            )
        return {
            "window": {"since": _iso(since), "until": _iso(until)},
            "cycles": int(cycle_rows["cycles"] or 0),
            "symbols_scanned": int(cycle_rows["scanned"] or 0),
            "discovered": int(cycle_rows["discovered"] or 0),
            "stages": stages,
            "drop_reasons": drop_reasons,
            "conversion": conversion,
        }

    def cycles_without_analysis(self, *, lookback: int = 10) -> int:
        """How many of the most recent cycles in a row discovered candidates but
        analyzed none (a stalled entry pipeline)."""
        with self._lock:
            cycles = self._conn.execute(
                "SELECT cycle_run_id, discovered FROM funnel_cycles ORDER BY as_of_date DESC, started_at DESC LIMIT ?",
                (int(lookback),),
            ).fetchall()
            streak = 0
            for cycle in cycles:
                analyzed = self._conn.execute(
                    "SELECT COUNT(*) FROM funnel_events WHERE cycle_run_id = ? AND stage = ? AND outcome = ?",
                    (cycle["cycle_run_id"], STAGE_ANALYZED, OUTCOME_ADVANCED),
                ).fetchone()[0]
                if int(cycle["discovered"] or 0) > 0 and int(analyzed) == 0:
                    streak += 1
                else:
                    break
            return streak


def safe_record(store: Optional[EvidenceFunnelStore], **kwargs: Any) -> None:
    """Record a funnel event; never raise into the caller (measurement only)."""
    if store is None:
        return
    try:
        store.record(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence funnel: could not record %s/%s: %s", kwargs.get("stage"), kwargs.get("symbol"), exc)


def safe_start_cycle(store: Optional[EvidenceFunnelStore], **kwargs: Any) -> None:
    if store is None:
        return
    try:
        store.start_cycle(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence funnel: could not record cycle %s: %s", kwargs.get("cycle_run_id"), exc)


DEFAULT_STALLED_CYCLES_THRESHOLD = 3


def build_evidence_funnel_report(
    *,
    as_of: date,
    store: Optional[EvidenceFunnelStore] = None,
    stalled_threshold: int = DEFAULT_STALLED_CYCLES_THRESHOLD,
) -> Dict[str, Any]:
    """Funnel section of the daily evidence report: today, 7 and 30 days."""
    from datetime import timedelta

    try:
        funnel = store or get_evidence_funnel()
        stalled = funnel.cycles_without_analysis()
        report = {
            "today": funnel.summarize(since=as_of, until=as_of),
            "last_7_days": funnel.summarize(since=as_of - timedelta(days=6), until=as_of),
            "last_30_days": funnel.summarize(since=as_of - timedelta(days=29), until=as_of),
            "cycles_without_analysis": stalled,
            "warnings": [],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "warnings": ["evidence_funnel_unavailable"]}
    if stalled >= int(stalled_threshold):
        report["warnings"].append("entry_pipeline_stalled")
    return report


def build_evidence_funnel_health(
    *,
    store: Optional[EvidenceFunnelStore] = None,
    stalled_threshold: int = DEFAULT_STALLED_CYCLES_THRESHOLD,
) -> Dict[str, Any]:
    """Watchdog issues for the funnel, in the evidence-health issue shape."""
    try:
        stalled = (store or get_evidence_funnel()).cycles_without_analysis()
    except Exception as exc:  # noqa: BLE001
        return {
            "summary": {"error": f"{type(exc).__name__}: {exc}"},
            "issues": [{"severity": "WARN", "alertable": False,
                        "message": f"Evidence funnel unreadable: {type(exc).__name__}"}],
        }
    issues = []
    if stalled >= int(stalled_threshold):
        issues.append({
            "severity": "WARN",
            "alertable": True,
            "message": (
                f"Entry pipeline stalled: the last {stalled} cycles discovered candidates "
                "but analyzed none (see the evidence funnel drop reasons)."
            ),
        })
    return {"summary": {"cycles_without_analysis": stalled, "stalled_threshold": int(stalled_threshold)},
            "issues": issues}


_store: Optional[EvidenceFunnelStore] = None
_store_lock = threading.Lock()


def get_evidence_funnel(store_path: Optional[Path] = None) -> EvidenceFunnelStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = EvidenceFunnelStore(store_path=store_path or _DEFAULT_STORE)
    return _store


__all__ = [
    "EvidenceFunnelStore",
    "build_evidence_funnel_health",
    "build_evidence_funnel_report",
    "OUTCOMES",
    "STAGES",
    "get_evidence_funnel",
    "new_cycle_run_id",
    "safe_record",
    "safe_start_cycle",
]
