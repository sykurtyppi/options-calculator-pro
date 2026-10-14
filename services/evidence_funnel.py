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
neither an unopenable store (``open_evidence_funnel``) nor a failed write
(``safe_record``) interrupts the loop. Failed writes are not silent, though:
they are counted per cycle and persisted when the store allows
(``finish_cycle``), and the report and watchdog show them as degraded health.

Conversion is cohort-valid: a transition's rate is, of the candidates (entry
stages, one pass per cycle) or trades (exit stages) that advanced through the
upstream stage in the window, the share that advanced through the next one.
It is never one window's downstream total over an unrelated upstream total
(newly recorded trades and older trades reaching exit are different cohorts).

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
REASON_RECHECK_EXPIRED = "recheck_expired"        # :<last deferral reason>

# Re-check schedule (F4): a deferred event is looked at again on its recheck
# date even if discovery does not surface it that day, and closes as done
# (it reached a decision) or expired (its entry window passed first).
RECHECK_PENDING = "pending"
RECHECK_DONE = "done"
RECHECK_EXPIRED = "expired"
# The event's earnings date moved: the re-check lives on under the new date.
RECHECK_SUPERSEDED = "superseded"
RECHECK_STATUSES = (RECHECK_PENDING, RECHECK_DONE, RECHECK_EXPIRED, RECHECK_SUPERSEDED)

# Stages after a trade exists: their subject is the trade, which can move
# through them across cycles. Earlier stages are one candidate's pass in one
# cycle (a deferred candidate starts a new pass in a later cycle).
TRADE_STAGES = (STAGE_RECORDED, STAGE_EXIT_DUE, STAGE_RESOLVED, STAGE_CLAIMABLE)

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
CREATE TABLE IF NOT EXISTS candidate_rechecks (
    symbol             TEXT NOT NULL,
    earnings_date      TEXT NOT NULL,
    recheck_on         TEXT NOT NULL,
    reason             TEXT NOT NULL,
    status             TEXT NOT NULL,
    attempts_json      TEXT NOT NULL,
    first_deferred_on  TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    PRIMARY KEY (symbol, earnings_date)
);
CREATE INDEX IF NOT EXISTS idx_candidate_rechecks_due ON candidate_rechecks (status, recheck_on);
CREATE TABLE IF NOT EXISTS funnel_write_health (
    cycle_run_id       TEXT PRIMARY KEY,
    as_of_date         TEXT NOT NULL,
    dropped_writes     INTEGER NOT NULL,
    updated_at         TEXT NOT NULL
);
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
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_DDL)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(candidate_rechecks)")}
            if "superseded_by" not in columns:
                self._conn.execute("ALTER TABLE candidate_rechecks ADD COLUMN superseded_by TEXT")
            self._conn.commit()
        except BaseException:
            self._conn.close()
            raise
        # Writes that failed this process, per cycle (see safe_record).
        self._dropped: Dict[str, Dict[str, Any]] = {}

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
        subject: Optional[str] = None,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown funnel stage: {stage}")
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown funnel outcome: {outcome}")
        if outcome != OUTCOME_ADVANCED and not reason:
            raise ValueError("a non-advancing funnel event needs a reason")
        # A trade is the subject once one exists (exit stages), else the
        # symbol; a row without a symbol (malformed discovery) names its own.
        subject = str(subject) if subject else (str(trade_id) if trade_id else str(symbol).upper())
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

        transitions = self._cohort_transitions(since=since, until=until)
        conversion = {name: item["rate"] for name, item in transitions.items()}
        with self._lock:
            dropped = self._conn.execute(
                f"SELECT COALESCE(SUM(dropped_writes), 0) FROM funnel_write_health{where}", params
            ).fetchone()[0]
        return {
            "window": {"since": _iso(since), "until": _iso(until)},
            "cycles": int(cycle_rows["cycles"] or 0),
            "symbols_scanned": int(cycle_rows["scanned"] or 0),
            "discovered": int(cycle_rows["discovered"] or 0),
            "stages": stages,
            "drop_reasons": drop_reasons,
            "conversion": conversion,
            "transitions": transitions,
            "dropped_writes": int(dropped or 0),
        }

    def _cohort_transitions(self, *, since: Optional[date], until: Optional[date]) -> Dict[str, Dict[str, Any]]:
        """Per stage transition: the upstream cohort in the window and where it went.

        Entry stages pair rows of the same candidate pass (cycle, symbol);
        trade stages pair rows of the same trade, whenever (up to ``until``)
        the trade reached the next stage. A member's downstream outcome is its
        best one: advanced, else dropped/error, else deferred; members with
        none yet are open (an exit still in the future, say).
        """
        until_iso = _iso(until)
        clauses, params = [], []
        if until_iso is not None:
            clauses.append("as_of_date <= ?")
            params.append(until_iso)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT cycle_run_id, as_of_date, stage, outcome, subject, symbol, trade_id FROM funnel_events{where}",
                params,
            ).fetchall()
        since_iso = _iso(since)

        def _member(row: sqlite3.Row, trade_transition: bool) -> Any:
            if trade_transition:
                return str(row["trade_id"] or row["subject"])
            return (row["cycle_run_id"], str(row["symbol"]) or str(row["subject"]))

        rank = {OUTCOME_ADVANCED: 3, OUTCOME_DROPPED: 2, OUTCOME_ERROR: 2, OUTCOME_DEFERRED: 1}
        out: Dict[str, Dict[str, Any]] = {}
        for upstream, downstream in zip(STAGES, STAGES[1:]):
            trade_transition = upstream in TRADE_STAGES
            cohort: Dict[Any, str] = {}
            for row in rows:
                if row["stage"] != upstream or row["outcome"] != OUTCOME_ADVANCED:
                    continue
                if since_iso is not None and str(row["as_of_date"]) < since_iso:
                    continue
                member = _member(row, trade_transition)
                cohort[member] = min(cohort.get(member, str(row["as_of_date"])), str(row["as_of_date"]))
            reached: Dict[Any, str] = {}
            for row in rows:
                if row["stage"] != downstream:
                    continue
                member = _member(row, trade_transition)
                if member not in cohort or str(row["as_of_date"]) < cohort[member]:
                    continue
                if rank[row["outcome"]] > rank.get(reached.get(member), 0):
                    reached[member] = row["outcome"]
            advanced = sum(1 for outcome in reached.values() if outcome == OUTCOME_ADVANCED)
            stopped = sum(1 for outcome in reached.values() if outcome in (OUTCOME_DROPPED, OUTCOME_ERROR))
            deferred = sum(1 for outcome in reached.values() if outcome == OUTCOME_DEFERRED)
            out[f"{upstream}->{downstream}"] = {
                "basis": "trade" if trade_transition else "candidate_pass",
                "cohort": len(cohort),
                "advanced": advanced,
                "stopped": stopped,
                "deferred": deferred,
                "open": len(cohort) - advanced - stopped - deferred,
                "rate": None if not cohort else round(advanced / len(cohort), 4),
            }
        return out

    # ── dropped writes ──────────────────────────────────────────────────────

    def note_dropped_write(self, *, cycle_run_id: Optional[str], as_of: Any = None) -> None:
        """Count a write this cycle lost (persisted by ``finish_cycle``)."""
        key = str(cycle_run_id or "unknown")
        entry = self._dropped.setdefault(key, {"count": 0, "as_of": None})
        entry["count"] += 1
        entry["as_of"] = entry["as_of"] or _iso(as_of) or (key[:10] if cycle_run_id else None)

    def dropped_writes(self, cycle_run_id: Optional[str] = None) -> int:
        if cycle_run_id is not None:
            return int(self._dropped.get(str(cycle_run_id), {}).get("count", 0))
        return sum(int(entry["count"]) for entry in self._dropped.values())

    def finish_cycle(self, *, cycle_run_id: str, as_of: date) -> int:
        """Persist how many of this cycle's writes were lost; returns the count.

        Raises if the store is still unwritable (callers use safe_finish_cycle);
        the count stays in memory and in the run's summary either way.
        """
        dropped = self.dropped_writes(cycle_run_id)
        if dropped:
            with self._lock:
                self._conn.execute(
                    """INSERT INTO funnel_write_health (cycle_run_id, as_of_date, dropped_writes, updated_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(cycle_run_id) DO UPDATE SET dropped_writes = excluded.dropped_writes,
                                                               updated_at = excluded.updated_at""",
                    (cycle_run_id, _iso(as_of), dropped, datetime.now(timezone.utc).isoformat()),
                )
                self._conn.commit()
        return dropped

    def recent_dropped_writes(self, *, since: date) -> int:
        with self._lock:
            value = self._conn.execute(
                "SELECT COALESCE(SUM(dropped_writes), 0) FROM funnel_write_health WHERE as_of_date >= ?",
                (_iso(since),),
            ).fetchone()[0]
        return int(value or 0)

    # ── re-check schedule ───────────────────────────────────────────────────

    def schedule_recheck(self, *, symbol: str, earnings_date: Any, recheck_on: date, reason: str,
                         as_of: date) -> None:
        """Defer an event to ``recheck_on``, keeping its attempt history."""
        key = (str(symbol).upper(), _iso(earnings_date))
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            prior = self._conn.execute(
                "SELECT attempts_json, first_deferred_on, status, recheck_on, reason FROM candidate_rechecks"
                " WHERE symbol = ? AND earnings_date = ?",
                key,
            ).fetchone()
            if (prior is not None and prior["status"] == RECHECK_PENDING and prior["reason"] == reason
                    and prior["recheck_on"] == _iso(recheck_on)):
                return  # already scheduled (e.g. a too-early event seen again)
            attempts = json.loads(prior["attempts_json"]) if prior else []
            attempts.append({"date": _iso(as_of), "reason": reason})
            self._conn.execute(
                """INSERT OR REPLACE INTO candidate_rechecks
                   (symbol, earnings_date, recheck_on, reason, status, attempts_json, first_deferred_on, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (*key, _iso(recheck_on), reason, RECHECK_PENDING, json.dumps(attempts),
                 prior["first_deferred_on"] if prior else _iso(as_of), now),
            )
            self._conn.commit()

    def close_recheck(self, *, symbol: str, earnings_date: Any, status: str = RECHECK_DONE) -> Optional[Dict[str, Any]]:
        """Close a pending re-check; returns it (with its attempts) or None if there was none."""
        key = (str(symbol).upper(), _iso(earnings_date))
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM candidate_rechecks WHERE symbol = ? AND earnings_date = ? AND status = ?",
                (*key, RECHECK_PENDING),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE candidate_rechecks SET status = ?, updated_at = ? WHERE symbol = ? AND earnings_date = ?",
                (status, datetime.now(timezone.utc).isoformat(), *key),
            )
            self._conn.commit()
        return _recheck_dict(row)

    def supersede_rechecks(self, *, symbol: str, earnings_date: Any, as_of: date) -> List[Dict[str, Any]]:
        """The event's earnings date moved to ``earnings_date``: close the
        symbol's pending re-checks filed under its earlier date.

        The same corporate event, by the ledger's rule (see
        ``recommendation_ledger.is_moved_event_date``): another date within
        the move window that had not yet passed at ``as_of``. Returns the
        superseded re-checks.
        """
        from services.recommendation_ledger import is_moved_event_date

        new_date = _iso(earnings_date)
        if not new_date:
            return []
        name = str(symbol).upper()
        closed: List[Dict[str, Any]] = []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM candidate_rechecks WHERE symbol = ? AND status = ? AND earnings_date != ?",
                (name, RECHECK_PENDING, new_date),
            ).fetchall()
            for row in rows:
                if not is_moved_event_date(row["earnings_date"], new_date, as_of=as_of):
                    continue
                self._conn.execute(
                    "UPDATE candidate_rechecks SET status = ?, superseded_by = ?, updated_at = ?"
                    " WHERE symbol = ? AND earnings_date = ?",
                    (RECHECK_SUPERSEDED, new_date, datetime.now(timezone.utc).isoformat(), name, row["earnings_date"]),
                )
                closed.append(_recheck_dict(row))
            if closed:
                self._conn.commit()
        return closed

    def pending_rechecks(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM candidate_rechecks WHERE status = ? ORDER BY recheck_on, symbol", (RECHECK_PENDING,)
            ).fetchall()
        return [_recheck_dict(row) for row in rows]

    def recheck_counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM candidate_rechecks GROUP BY status").fetchall()
        counts = {status: 0 for status in RECHECK_STATUSES}
        counts.update({str(row["status"]): int(row["n"]) for row in rows})
        return counts

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


def _recheck_dict(row: sqlite3.Row) -> Dict[str, Any]:
    out = dict(row)
    out["attempts"] = json.loads(out.pop("attempts_json") or "[]")
    return out


def note_dropped_write(store: Optional[EvidenceFunnelStore], *, cycle_run_id: Optional[str], as_of: Any = None) -> None:
    if store is None:
        return
    try:
        store.note_dropped_write(cycle_run_id=cycle_run_id, as_of=as_of)
    except Exception:  # noqa: BLE001
        logger.debug("evidence funnel: could not count a dropped write", exc_info=True)


def safe_record(store: Optional[EvidenceFunnelStore], **kwargs: Any) -> None:
    """Record a funnel event; never raise into the caller (measurement only).

    A failed write is counted against its cycle, so it shows up as degraded
    funnel health instead of vanishing."""
    if store is None:
        return
    try:
        store.record(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence funnel: could not record %s/%s: %s", kwargs.get("stage"), kwargs.get("symbol"), exc)
        note_dropped_write(store, cycle_run_id=kwargs.get("cycle_run_id"), as_of=kwargs.get("as_of"))


def safe_start_cycle(store: Optional[EvidenceFunnelStore], **kwargs: Any) -> None:
    if store is None:
        return
    try:
        store.start_cycle(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence funnel: could not record cycle %s: %s", kwargs.get("cycle_run_id"), exc)
        note_dropped_write(store, cycle_run_id=kwargs.get("cycle_run_id"), as_of=kwargs.get("as_of"))


def safe_finish_cycle(store: Optional[EvidenceFunnelStore], *, cycle_run_id: str, as_of: date) -> Dict[str, Any]:
    """The cycle's funnel status for its run summary: ok, degraded (writes
    were lost) or unavailable (no store); persists the lost-write count."""
    if store is None:
        return {"status": "unavailable", "dropped_writes": None}
    dropped = store.dropped_writes(cycle_run_id)
    persisted = True
    try:
        store.finish_cycle(cycle_run_id=cycle_run_id, as_of=as_of)
    except Exception as exc:  # noqa: BLE001
        persisted = False
        logger.warning("evidence funnel: could not persist %d dropped writes for %s: %s", dropped, cycle_run_id, exc)
    status = "degraded" if dropped else "ok"
    return {"status": status, "dropped_writes": dropped, "dropped_writes_persisted": persisted if dropped else None}


def open_evidence_funnel(store: Optional[EvidenceFunnelStore] = None) -> Optional[EvidenceFunnelStore]:
    """The funnel store, or None if it cannot be opened (locked, unwritable).

    The funnel is measurement only: a store that cannot be opened must not
    stop the loop it measures. The caller reports the cycle's funnel as
    unavailable."""
    if store is not None:
        return store
    try:
        return get_evidence_funnel()
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence funnel: store unavailable, this cycle is not measured: %s", exc)
        return None


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
            "rechecks": funnel.recheck_counts(),
            "warnings": [],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "warnings": ["evidence_funnel_unavailable"]}
    if stalled >= int(stalled_threshold):
        report["warnings"].append("entry_pipeline_stalled")
    if report["last_7_days"]["dropped_writes"]:
        # Some of the week's stage events were lost: its counts are a floor.
        report["warnings"].append("evidence_funnel_writes_dropped")
    # F6: exit-due -> resolved per event, across the paper, baseline and
    # resolver tracks (each keeps its own per-trade stages above).
    try:
        from services.event_resolution import load_event_resolutions, summarize_event_resolutions

        report["event_exits"] = summarize_event_resolutions(load_event_resolutions(as_of=as_of))
    except Exception as exc:  # noqa: BLE001
        report["event_exits"] = {"error": f"{type(exc).__name__}: {exc}"}
        report["warnings"].append("event_exits_unavailable")
    return report


def build_evidence_funnel_health(
    *,
    store: Optional[EvidenceFunnelStore] = None,
    stalled_threshold: int = DEFAULT_STALLED_CYCLES_THRESHOLD,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Watchdog issues for the funnel, in the evidence-health issue shape."""
    from datetime import timedelta

    try:
        funnel = store or get_evidence_funnel()
        stalled = funnel.cycles_without_analysis()
        dropped = funnel.recent_dropped_writes(since=(as_of or date.today()) - timedelta(days=6))
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
    if dropped:
        issues.append({
            "severity": "WARN",
            "alertable": True,
            "message": (
                f"Evidence funnel degraded: {dropped} stage writes were lost in the last 7 days "
                "(locked or unwritable store); funnel counts for those cycles are a floor."
            ),
        })
    return {"summary": {"cycles_without_analysis": stalled, "stalled_threshold": int(stalled_threshold),
                        "dropped_writes_7d": dropped},
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
    "note_dropped_write",
    "open_evidence_funnel",
    "safe_finish_cycle",
    "safe_record",
    "safe_start_cycle",
]
