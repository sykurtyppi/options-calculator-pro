"""Durable recommendation ledger for selector auditability.

The ledger records the exact evidence surface that produced a recommendation:
VolSnapshot, structure scorecards, selector output, provider/quote provenance,
and schema/version metadata. It is intentionally **immutable** — once a row
lands in ``recommendations``, it is never updated. Any subsequent ``record()``
call for the same ``recommendation_id`` writes a new row into the
``recommendation_revisions`` table instead, so the original evidence is
preserved and enrichment / re-runs are auditable as a separate history.

This is the durable backbone of the evidence loop. If the original row could
be silently overwritten (as it could before PR-K), then "we recommended X on
date Y" stops being a defensible claim.

Tables
------
``recommendations``
    Append-only. ``INSERT OR IGNORE`` semantics — the first write per
    ``recommendation_id`` wins forever.

``recommendation_revisions``
    Captures subsequent ``record()`` calls. Each revision stores the full new
    payload as JSON plus a content hash; identical-content re-runs are deduped
    via the ``UNIQUE(recommendation_id, content_hash)`` constraint, so paper
    loops re-recording the same evidence don't pile up no-op rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional

from services.candidate_shadow_provenance import (
    SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
    normalize_sample_provenance,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2  # Bumped: introduces recommendation_revisions
ENGINE_VERSION = "event_vol_selector_v1"
_DEFAULT_LEDGER = Path.home() / ".options_calculator_pro" / "recommendations" / "recommendation_ledger.sqlite"
_WRITE_LOCK = threading.Lock()

RecordStatus = Literal["inserted", "revision", "duplicate"]

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
    -- PR-AC commit 4: experimental_contract_selection block from the
    -- live API response, serialized verbatim. Captures picker_variant,
    -- shadow_mode label, front/back expiries, strike, pickers_diverged,
    -- and the candidate_min_front_dte_days threshold. Required for any
    -- future promotion-criterion analysis that compares forward
    -- observations between legacy and candidate picker rules. Schema
    -- documented in the PR-AC commit 3 response surface.
    picker_provenance_json        TEXT,
    -- PR-AD commit 3: candidate_shadow_outcome block serialized
    -- verbatim. On historical replay records it carries the candidate
    -- picker's resolved PnL (entry_debit_mid, exit_value_mid, mid_pnl,
    -- mid_realized_return_pct, IV changes) plus labels (research_mid /
    -- shadow_only / not_execution_grade). On live forward records
    -- produced before an exit resolver runs, this column is empty {}
    -- — see docs/CALENDAR_PICKER_PROMOTION_2026-05-27.md prerequisites
    -- for the exit-resolver gap.
    candidate_shadow_outcome_json TEXT,
    -- PR-AD commit 3: sample provenance for promotion-eligibility
    -- filtering. Denormalized so SQL queries can filter without
    -- parsing JSON. Values come from SAMPLE_PROVENANCE_* constants
    -- in services/candidate_shadow_provenance.py (canonical source
    -- after PR-AD commit 4 extracted them out of web/api/edge_engine.py
    -- for services -> web edge cleanup); promotion criteria
    -- reference only the PROMOTION_ELIGIBLE_PROVENANCES set
    -- defined in that same module.
    sample_provenance             TEXT,
    -- PR-AE commit 1: live-exit-resolver attempt counter. Process
    -- bookkeeping, NOT part of the recommendation evidence — that is
    -- why it lives directly on the immutable recommendations row
    -- rather than inside a revision payload. The resolver increments
    -- this only on terminal-failure revisions (retrying /
    -- permanently_failed:*), never on `ok` (success is not a retry)
    -- and never on `awaiting_chain_data` (data-not-ready is not a
    -- failed attempt). Defaults to 0 for existing rows on migration.
    candidate_exit_resolver_attempts INTEGER NOT NULL DEFAULT 0,
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
-- PR-AE commit 1: composite index covers the resolver's eligibility
-- query (sample_provenance = forward_post_freeze AND earnings_date
-- <= cutoff). Without it the query would full-scan every time the
-- daily resolver tick runs.
CREATE INDEX IF NOT EXISTS idx_recommendations_provenance_earnings
    ON recommendations (sample_provenance, earnings_date);
"""

_REVISIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS recommendation_revisions (
    revision_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id    TEXT NOT NULL,
    revised_at           TEXT NOT NULL,
    content_hash         TEXT NOT NULL,
    record_json          TEXT NOT NULL,
    UNIQUE(recommendation_id, content_hash)
);
"""

_REVISIONS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_revisions_recommendation_id
    ON recommendation_revisions (recommendation_id);
CREATE INDEX IF NOT EXISTS idx_revisions_revised_at
    ON recommendation_revisions (revised_at);
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
    # PR-AC commit 4 — picker provenance for forward-validation.
    "picker_provenance_json": "TEXT",
    # PR-AD commit 3 — candidate shadow outcome + provenance tag.
    "candidate_shadow_outcome_json": "TEXT",
    "sample_provenance": "TEXT",
    # PR-AE commit 1 — live-exit-resolver attempt counter (process
    # bookkeeping; never part of recommendation evidence or
    # content-hash).
    "candidate_exit_resolver_attempts": "INTEGER NOT NULL DEFAULT 0",
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
    # PR-AC commit 4: experimental_contract_selection block from the
    # live API response. Empty dict when the response did not include
    # one (most pre-PR-AC recommendations and any structure that isn't
    # call/put_calendar). Required for downstream promotion-criterion
    # analysis to attribute forward observations to their picker rule.
    picker_provenance: Dict[str, Any] = field(default_factory=dict)
    # PR-AD commit 3: candidate_shadow_outcome block (from the
    # historical replay simulator) and sample_provenance tag. For
    # forward live observations both are populated when the live API
    # path supplies them; the candidate outcome will be empty until a
    # live exit resolver runs (see docs/CALENDAR_PICKER_PROMOTION
    # prerequisites).
    candidate_shadow_outcome: Dict[str, Any] = field(default_factory=dict)
    sample_provenance: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    engine_version: str = ENGINE_VERSION


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # PR-T: consistency with trading_system/database.py. The module-level
    # _WRITE_LOCK serializes in-process writers and WAL handles concurrent
    # readers, but cross-process writers (a future worker pool or a manual
    # sqlite3 CLI session against the file) would otherwise fail with
    # OperationalError after SQLite's tiny default 5s busy timeout.
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_TABLE_DDL)
    _migrate(conn)
    conn.executescript(_INDEX_DDL)
    conn.executescript(_REVISIONS_TABLE_DDL)
    conn.executescript(_REVISIONS_INDEX_DDL)
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

    def record(self, record: RecommendationRecord) -> RecordStatus:
        """Record a recommendation, preserving original evidence.

        The first call with a given ``recommendation_id`` inserts into the
        ``recommendations`` table and returns ``"inserted"``. Subsequent calls
        for the same id never modify the original row — they go to the
        ``recommendation_revisions`` table with the full new payload, and
        return ``"revision"``. Calls with byte-identical content (same
        content_hash) are skipped and return ``"duplicate"``.

        This replaces the prior ``INSERT OR IGNORE … ON CONFLICT DO UPDATE``
        behavior, which silently overwrote the original evidence on every
        subsequent call — defeating the whole point of an audit ledger.
        """
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                return self._write_record_within_tx(cur, record)

    # Internal critical-section helper. Operates on an already-open
    # cursor inside an already-acquired lock + transaction. Two public
    # paths reuse it:
    #
    #   - record(): single-record write.
    #   - record_resolution_and_attempt(): single-record write PLUS an
    #     atomic resolver-counter UPDATE in the same transaction.
    #
    # Keeping the SQL + params in one place prevents drift between the
    # two call sites and is the linchpin of the PR-AE C1b atomicity
    # guarantee — the resolver cannot land a revision while leaving the
    # counter stale, because both writes commit (or roll back) together.
    def _write_record_within_tx(
        self,
        cur: sqlite3.Cursor,
        record: RecommendationRecord,
    ) -> RecordStatus:
        insert_sql = """
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
                explanation_json, metadata_json, picker_provenance_json,
                candidate_shadow_outcome_json, sample_provenance,
                schema_version, engine_version
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
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
            _json(record.picker_provenance),
            _json(record.candidate_shadow_outcome),
            record.sample_provenance,
            int(record.schema_version),
            record.engine_version,
        )
        content_hash = _record_content_hash(record)
        record_json = _json(_record_to_dict(record))
        revised_at = datetime.now(timezone.utc).isoformat()

        cur.execute(insert_sql, params)
        first_time = cur.rowcount > 0

        # Always stamp the current content into recommendation_revisions.
        # On first record, this writes the "version 1" row so the full
        # history table is complete. On subsequent calls, the
        # UNIQUE(recommendation_id, content_hash) constraint dedupes
        # byte-identical re-records (e.g. paper-loop replays of the
        # same evidence) into a no-op.
        cur.execute(
            """
            INSERT OR IGNORE INTO recommendation_revisions
                (recommendation_id, revised_at, content_hash, record_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                record.recommendation_id,
                revised_at,
                content_hash,
                record_json,
            ),
        )
        inserted_revision = cur.rowcount > 0

        if first_time:
            return "inserted"
        return "revision" if inserted_revision else "duplicate"

    def get(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM recommendations WHERE recommendation_id = ?",
            (recommendation_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_revisions(self, recommendation_id: str) -> List[Dict[str, Any]]:
        """Return the full version history of a recommendation, oldest first.

        Every successful ``record()`` call leaves a row here, including the
        first one (the "v1" snapshot). Byte-identical re-records are deduped
        via the ``UNIQUE(recommendation_id, content_hash)`` constraint, so
        the list contains one entry per distinct content state, in
        chronological order.

        For "show me only the changes after the original," use
        ``get_revisions(...)[1:]``.

        Pairs cleanly with ``get(recommendation_id)``: ``get()`` returns the
        immutable first-write payload, ``get_revisions()`` returns the full
        history including enrichment.
        """
        rows = self._conn.execute(
            """
            SELECT revision_id, recommendation_id, revised_at, content_hash, record_json
            FROM recommendation_revisions
            WHERE recommendation_id = ?
            ORDER BY revised_at ASC, revision_id ASC
            """,
            (recommendation_id,),
        ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            try:
                record_payload = json.loads(row["record_json"])
            except (TypeError, ValueError):
                record_payload = None
            result.append(
                {
                    "revision_id": int(row["revision_id"]),
                    "recommendation_id": str(row["recommendation_id"]),
                    "revised_at": str(row["revised_at"]),
                    "content_hash": str(row["content_hash"]),
                    "record": record_payload,
                }
            )
        return result

    # ── PR-AE commit 1: live-exit-resolver support ────────────────────────────
    #
    # The four helpers below are the ledger-side surface the resolver
    # (services/candidate_exit_resolver.py, commit 4) consumes. They are
    # deliberately small and self-contained:
    #
    #   - get_with_latest_resolution: merged "current state" view that
    #     overlays the latest revision's candidate_shadow_outcome onto
    #     the immutable v1 row. Everything else stays at v1.
    #
    #   - record_resolution_payload: convenience around record() that
    #     reconstructs a full RecommendationRecord from the v1 row and
    #     applies only the new candidate_shadow_outcome. Routes the
    #     write through the existing PR-K revisions table; the
    #     UNIQUE(recommendation_id, content_hash) constraint dedupes
    #     byte-identical re-resolutions automatically.
    #
    #   - increment_resolver_attempts: single-statement UPDATE on the
    #     immutable row. This is the ONLY mutation path PR-AE adds to
    #     the v1 row; justified because the counter is process
    #     bookkeeping, not part of the recommendation evidence.
    #
    #   - list_pending_candidate_exit_resolutions: eligibility query
    #     scoped to forward_post_freeze + call_calendar + outside the
    #     post-event window + under MAX_ATTEMPTS. Returns merged-view
    #     dicts (one row per eligible recommendation).
    #
    # Full design contract lives in docs/PR_AE_LIVE_EXIT_RESOLVER_DESIGN
    # _2026-05-27.md "Idempotent update behavior" section.

    def get_with_latest_resolution(
        self, recommendation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the merged "current state" view of a recommendation.

        Loads the immutable v1 row from ``recommendations`` and overlays
        the latest revision's ``candidate_shadow_outcome`` onto it.
        Every other field stays at v1 — in particular
        ``sample_provenance`` (immutable post-tagging) and
        ``picker_provenance_json`` (immutable selection record).

        Aggregators consuming the resolver's output should call this
        helper (not ``get()``) so they see post-event resolved
        outcomes. ``get()`` continues to return strictly the
        first-write payload for audit purposes.

        Returns ``None`` if the recommendation_id is unknown.
        """
        base = self.get(recommendation_id)
        if base is None:
            return None
        revisions = self.get_revisions(recommendation_id)
        if not revisions:
            return base
        latest_payload = revisions[-1].get("record")
        if not isinstance(latest_payload, dict):
            return base
        latest_outcome = latest_payload.get("candidate_shadow_outcome")
        if not isinstance(latest_outcome, dict):
            return base
        merged = dict(base)
        # _row_to_dict exposes the parsed JSON under the *_json key,
        # so we mirror that convention here for consistency with
        # everything else get() returns.
        merged["candidate_shadow_outcome_json"] = latest_outcome
        return merged

    def increment_resolver_attempts(self, recommendation_id: str) -> int:
        """Atomically increment ``candidate_exit_resolver_attempts``.

        Returns the new counter value. Raises ``KeyError`` if no row
        with the given id exists.

        The resolver calls this ONLY when writing a terminal-failure
        revision (``retrying`` or ``permanently_failed:*``). It must
        not be called on ``ok`` (success is not a retry) or
        ``awaiting_chain_data`` (data-not-ready is not a failed
        attempt). The full no-op-run rule table lives in the design
        doc.
        """
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                cur.execute(
                    """
                    UPDATE recommendations
                       SET candidate_exit_resolver_attempts =
                           candidate_exit_resolver_attempts + 1
                     WHERE recommendation_id = ?
                    """,
                    (recommendation_id,),
                )
                if cur.rowcount == 0:
                    raise KeyError(
                        f"recommendation_id not found: {recommendation_id}"
                    )
                row = cur.execute(
                    """
                    SELECT candidate_exit_resolver_attempts
                      FROM recommendations
                     WHERE recommendation_id = ?
                    """,
                    (recommendation_id,),
                ).fetchone()
                return int(row[0])

    def record_resolution_payload(
        self,
        *,
        recommendation_id: str,
        candidate_shadow_outcome: Dict[str, Any],
    ) -> RecordStatus:
        """Write a new revision overlaying only ``candidate_shadow_outcome``.

        Convenience wrapper around
        :py:meth:`record_resolution_and_attempt` with
        ``increment_attempts=False``. Use this when the resolver lands
        a non-counter-incrementing status (currently: ``ok`` or
        ``awaiting_chain_data``).

        For terminal-failure statuses (``retrying`` or
        ``permanently_failed:*``) call ``record_resolution_and_attempt``
        directly with ``increment_attempts=True`` so the revision write
        and the counter UPDATE happen inside a single transaction.

        Returns the ``RecordStatus`` from the underlying ``record()``:
          - ``"inserted"`` should never appear in resolver context
            (the v1 row must already exist). It does on a defensive
            path if the row was somehow deleted between read and
            write; the resolver logs at WARNING.
          - ``"revision"`` — new content_hash, new revision row
            written.
          - ``"duplicate"`` — content_hash matched a prior revision;
            no new row written, idempotent re-resolution.

        Raises ``KeyError`` if ``recommendation_id`` is unknown.
        """
        return self.record_resolution_and_attempt(
            recommendation_id=recommendation_id,
            candidate_shadow_outcome=candidate_shadow_outcome,
            increment_attempts=False,
        )

    def record_resolution_and_attempt(
        self,
        *,
        recommendation_id: str,
        candidate_shadow_outcome: Dict[str, Any],
        increment_attempts: bool,
    ) -> RecordStatus:
        """Atomic "write resolver outcome + maybe bump counter" path.

        PR-AE Codex review (C1 audit, P1): the resolver MUST NOT call
        ``record_resolution_payload(...)`` and
        ``increment_resolver_attempts(...)`` as two independent
        operations. A crash in the window between them would leave a
        ``retrying`` revision on disk while the attempt counter stayed
        at its old value — and since ``retrying`` is non-terminal, the
        eligibility filter would keep returning the row forever
        without ever escalating to ``permanently_failed:no_post_event_chain``.

        This helper takes the lock once, opens a single transaction,
        and inside it:

          1. Reconstructs a full ``RecommendationRecord`` from the v1
             row preserving everything except
             ``candidate_shadow_outcome``.
          2. Calls the shared internal ``_write_record_within_tx``
             helper to do the recommendations-table no-op insert and
             the revisions-table conditional insert.
          3. If ``increment_attempts`` is True, runs the counter
             UPDATE in the same cursor / same transaction.

        Either all three writes commit, or none do. There is no
        partial-state failure mode.

        The resolver's call rule (see design doc no-op-run table):

          status == "ok"                     → increment_attempts=False
          status == "awaiting_chain_data"    → increment_attempts=False
          status == "retrying"               → increment_attempts=True
          status startswith "permanently_failed:" → increment_attempts=True

        Raises ``KeyError`` if ``recommendation_id`` is unknown.

        Revision overlay semantics (Codex C5 audit, P3 — IMPORTANT):
        =============================================================
        This helper rebuilds the revision payload from the IMMUTABLE
        v1 row and overlays ONLY ``candidate_shadow_outcome``. It is
        NOT a cumulative full-state merge across prior revisions.

        Concretely: if a future enrichment path (some PR-AF, PR-AG, …)
        writes a revision that modifies ``picker_provenance`` or
        ``surface_quality`` or any other field, and then the resolver
        runs against the same row, the resolver's new revision will:

          - carry v1's ``picker_provenance`` / ``surface_quality`` —
            NOT the most-recent enriched values from the intervening
            revision.
          - carry the resolver's new ``candidate_shadow_outcome``.

        This is intentional for PR-AE: the resolver only mutates the
        candidate outcome, and v1 already carries the right values for
        every other field. If a future PR needs a true cumulative
        overlay (merge latest revision's payload + this revision's
        changes), it must:

          - Read the latest revision via ``get_revisions(...)`` rather
            than only v1 inside this transaction.
          - Or add a new dedicated helper with merge-from-latest
            semantics, leaving this one for "PR-AE-style outcome-only
            overlays."

        The "PR-AE-only" scope is enforced by the
        ``test_pr_ae_c5b_revisions_are_overlay_not_cumulative_merge``
        regression test below; any future helper that introduces true
        cumulative semantics must add its own test under a different
        name.
        """
        with _WRITE_LOCK:
            with _tx(self._conn) as cur:
                # Read v1 inside the transaction so a concurrent
                # delete cannot race a stale base into the
                # reconstruction. SQLite's WAL mode plus the
                # _WRITE_LOCK + busy_timeout make this safe.
                base_row = cur.execute(
                    "SELECT * FROM recommendations WHERE recommendation_id = ?",
                    (recommendation_id,),
                ).fetchone()
                if base_row is None:
                    raise KeyError(
                        f"recommendation_id not found: {recommendation_id}"
                    )
                base = _row_to_dict(base_row)

                scorecards_payload = base.get("structure_scorecards_json")
                structure_scorecards: List[Dict[str, Any]] = (
                    list(scorecards_payload)
                    if isinstance(scorecards_payload, list)
                    else []
                )

                def _dict_or_empty(value: Any) -> Dict[str, Any]:
                    return dict(value) if isinstance(value, dict) else {}

                record = RecommendationRecord(
                    recommendation_id=str(base["recommendation_id"]),
                    created_at=str(base["created_at"]),
                    symbol=str(base["symbol"]),
                    as_of_date=base.get("as_of_date"),
                    earnings_date=base.get("earnings_date"),
                    earnings_source=base.get("earnings_source"),
                    earnings_source_confidence=base.get("earnings_source_confidence"),
                    earnings_source_stale=bool(base.get("earnings_source_stale", False)),
                    recommendation=base.get("recommendation"),
                    selected_structure=base.get("selected_structure"),
                    no_trade_reason=base.get("no_trade_reason"),
                    data_quality_score=base.get("data_quality_score"),
                    option_source=base.get("option_source"),
                    underlying_source=base.get("underlying_source"),
                    provider_names=_dict_or_empty(base.get("provider_names_json")),
                    quote_timestamp=base.get("quote_timestamp"),
                    quote_source=base.get("quote_source"),
                    quote_quality=base.get("quote_quality"),
                    bid_ask_mid=_dict_or_empty(base.get("bid_ask_mid_json")),
                    surface_quality=_dict_or_empty(base.get("surface_quality_json")),
                    vol_snapshot=_dict_or_empty(base.get("vol_snapshot_json")),
                    structure_scorecards=structure_scorecards,
                    selector_output=_dict_or_empty(base.get("selector_output_json")),
                    explanation=_dict_or_empty(base.get("explanation_json")),
                    metadata=_dict_or_empty(base.get("metadata_json")),
                    picker_provenance=_dict_or_empty(base.get("picker_provenance_json")),
                    candidate_shadow_outcome=dict(candidate_shadow_outcome),
                    sample_provenance=base.get("sample_provenance"),
                    schema_version=int(base.get("schema_version") or SCHEMA_VERSION),
                    engine_version=str(base.get("engine_version") or ENGINE_VERSION),
                )

                status = self._write_record_within_tx(cur, record)

                if increment_attempts:
                    cur.execute(
                        """
                        UPDATE recommendations
                           SET candidate_exit_resolver_attempts =
                               candidate_exit_resolver_attempts + 1
                         WHERE recommendation_id = ?
                        """,
                        (recommendation_id,),
                    )
                    if cur.rowcount == 0:
                        # Defensive — should not happen since we
                        # read v1 inside the same transaction. If it
                        # does, raise to roll back the whole tx via
                        # the _tx context manager.
                        raise KeyError(
                            f"recommendation_id vanished mid-transaction: "
                            f"{recommendation_id}"
                        )

                return status

    def list_pending_candidate_exit_resolutions(
        self,
        *,
        now: Optional[date] = None,
        min_days_after_event: int = 3,
        max_attempts: int = 6,
        sample_provenance: str = SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
        selected_structure: str = "call_calendar",
    ) -> List[Dict[str, Any]]:
        """Return merged-view dicts for rows ready for exit resolution.

        Eligibility (all must hold):
          - ``sample_provenance`` matches the provided value (default:
            forward_post_freeze).
          - ``selected_structure`` matches the provided value
            (default: call_calendar; put_calendar is deferred).
          - ``earnings_date`` is set and ``<= now - min_days_after_event``
            so we land on a real post-event trade_date.
          - ``candidate_exit_resolver_attempts < max_attempts``.
          - The merged-view's ``candidate_shadow_outcome.status`` is
            not ``"ok"`` and does not start with ``"permanently_failed:"``.
            Rows already in those terminal states are not retried.

        Returns merged-view dicts in ``earnings_date ASC`` order so
        older events are resolved first.

        Caller (the resolver) is responsible for further
        fail-closed behavior on malformed ``picker_provenance_json``
        — that becomes ``permanently_failed:no_picker_provenance``
        terminal status written via ``record_resolution_payload``.

        **Calendar-day vs trading-day semantics (PR-AE Codex P2 audit):**
        ``min_days_after_event`` is measured in CALENDAR days, not
        trading days, because (a) ``earnings_date`` is stored as a plain
        ISO date string that supports cheap lexicographic SQL comparison
        and (b) the ledger has no exchange-calendar dependency. The
        default of 3 covers the common edge case where earnings land on
        a Friday: 3 calendar days later is Monday, by which time the
        post-event chain should have settled.

        The PRECISE chain trade-date is selected by the resolver layer
        using ``pandas.tseries.offsets.BDay`` lookahead within
        ``MAX_LOOKAHEAD_TRADE_DAYS`` (default 5 business days). This
        SQL filter is intentionally a coarse "post-event window" gate;
        any false positives (e.g. earnings on Friday, ``now``
        Sunday + 3 calendar days = Monday but no chain yet) are absorbed
        by the resolver writing ``awaiting_chain_data`` and NOT
        burning retry budget — see the design doc.
        """
        today = now or datetime.now(timezone.utc).date()
        cutoff = today - timedelta(days=int(min_days_after_event))
        cutoff_str = cutoff.isoformat()

        rows = self._conn.execute(
            """
            SELECT *
              FROM recommendations
             WHERE sample_provenance = ?
               AND selected_structure = ?
               AND earnings_date IS NOT NULL
               AND earnings_date <= ?
               AND candidate_exit_resolver_attempts < ?
             ORDER BY earnings_date ASC, created_at ASC
            """,
            (
                sample_provenance,
                selected_structure,
                cutoff_str,
                int(max_attempts),
            ),
        ).fetchall()

        eligible: List[Dict[str, Any]] = []
        for row in rows:
            merged = self.get_with_latest_resolution(str(row["recommendation_id"]))
            if merged is None:
                continue
            outcome = merged.get("candidate_shadow_outcome_json")
            outcome_status = (
                str(outcome.get("status") or "")
                if isinstance(outcome, dict)
                else ""
            )
            if outcome_status == "ok":
                continue
            if outcome_status.startswith("permanently_failed:"):
                continue
            eligible.append(merged)
        return eligible

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

    # PR-AC commit 4: capture the experimental_contract_selection block
    # if the API response carried one (commit 3 surface). Lives in metrics
    # since the analysis payload places it alongside structure_payoff /
    # calendar_payoff. Empty dict when absent — backward-compatible with
    # any earlier analysis payloads.
    picker_provenance = _as_dict(metrics.get("experimental_contract_selection") or {})

    # PR-AD commit 3: capture the candidate_shadow_outcome block and
    # the top-level sample_provenance tag. The analysis is typically an
    # EdgeSnapshot (live API) or a SimpleNamespace (historical replay
    # wrapper). For the live API today the candidate outcome is empty
    # — exit resolver is a pending prerequisite per the promotion doc.
    # For historical replay records the outcome carries the resolved
    # candidate PnL produced by _simulate_candidate_shadow_outcome.
    candidate_shadow_outcome = _as_dict(
        _get(analysis, "candidate_shadow_outcome", {}) or {}
    )
    raw_sample_provenance = _get(analysis, "sample_provenance", None)
    # Codex round-4 review fix: normalize to "unknown" (observable in
    # diagnostics) rather than silently dropping to None. Invalid /
    # typo / non-canonical values become SAMPLE_PROVENANCE_UNKNOWN so
    # they appear in provenance_counts under the unknown bucket
    # instead of disappearing as missing keys. Routes through the
    # neutral service module — no inverted dependency on web/api.
    if raw_sample_provenance is None:
        sample_provenance = None
    else:
        sample_provenance = normalize_sample_provenance(raw_sample_provenance)

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
        picker_provenance=picker_provenance,
        candidate_shadow_outcome=candidate_shadow_outcome,
        sample_provenance=sample_provenance,
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
        "picker_provenance_json",
        "candidate_shadow_outcome_json",
    ):
        payload[key] = _loads(payload.get(key))
    payload["earnings_source_stale"] = bool(payload.get("earnings_source_stale"))
    return payload


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, default=str)


def _record_to_dict(record: RecommendationRecord) -> Dict[str, Any]:
    """Serialize a RecommendationRecord into a JSON-safe dict.

    Used both for the revision payload (full original re-record) and for
    content hashing. Uses ``dataclasses.asdict`` which recursively turns
    the dataclass into nested dict/list primitives. Non-JSON-safe leaf
    types (e.g. ``date``) are handled by ``_json``'s ``default=str``.
    """
    return asdict(record)


def _record_content_hash(record: RecommendationRecord) -> str:
    """Stable SHA-256 of the record's logical content.

    Excludes ``created_at`` (timestamp drift on re-record should not
    register as a content change) and the literal ``recommendation_id``
    (a hash keyed by id alone is uninformative). Includes everything
    else, sorted for stability.
    """
    payload = _record_to_dict(record)
    payload.pop("created_at", None)
    payload.pop("recommendation_id", None)
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
