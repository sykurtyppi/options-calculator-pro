"""Durable alert outbox: send each alert at most once, and say what is known.

The premarket alert used to send first and record the "already sent" digest
afterwards in a JSON file. A failed state write, an unreadable state file
(read as empty), two runs at once, or an automatic retry after a timeout
(the first attempt may already have gone out) could each send the same alert
twice, and a timeout was audited as a plain failure.

Every alert key now moves through one row of this SQLite outbox:

    prepared            reserved by one run; nothing handed to the provider yet
    provider_submitted  about to be / being handed to the provider
    accepted            the provider took it (osascript returned success; this
                        is acceptance by Messages, not a delivery receipt)
    delivery_unknown    the hand-off was interrupted (timeout): it may or may
                        not have gone out, so it is never resent automatically
    failed              the provider refused it: nothing went out, a later run
                        may try again

The reservation is a single exclusive transaction, so two runs racing on the
same key cannot both send. A run that crashed after reserving but before
submitting leaves a ``prepared`` row, which a later run may reclaim after
RESERVATION_LEASE_SECONDS; a crash after ``provider_submitted`` is treated
like a timeout (never resent). An unreadable outbox is never treated as
empty: callers quarantine it and send nothing.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

PREPARED = "prepared"
PROVIDER_SUBMITTED = "provider_submitted"
ACCEPTED = "accepted"
DELIVERY_UNKNOWN = "delivery_unknown"
FAILED = "failed"
STATES = (PREPARED, PROVIDER_SUBMITTED, ACCEPTED, DELIVERY_UNKNOWN, FAILED)

_TRANSITIONS = {
    PREPARED: {PROVIDER_SUBMITTED, FAILED},
    PROVIDER_SUBMITTED: {ACCEPTED, DELIVERY_UNKNOWN, FAILED},
}

RESERVATION_LEASE_SECONDS = 600

_DDL = """
CREATE TABLE IF NOT EXISTS alert_outbox (
    alert_key     TEXT PRIMARY KEY,
    event_id      TEXT NOT NULL,
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    last_error    TEXT,
    detail_json   TEXT
);
"""


class OutboxCorrupt(RuntimeError):
    """The outbox file is damaged; it must not be treated as empty."""


class OutboxUnavailable(RuntimeError):
    """The outbox could not be opened just now (locked, I/O). Not damage:
    never quarantined, and nothing is sent."""


# Messages SQLite uses for a damaged file, as opposed to a busy or
# temporarily unreadable one.
_CORRUPTION_MARKERS = ("not a database", "malformed", "corrupt", "unsupported file format")
_BUSY_MARKERS = ("locked", "busy")
OPEN_ATTEMPTS = 20


class InvalidTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    acquired: bool
    status: Optional[str]
    attempts: int = 0


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AlertOutbox:
    def __init__(self, path: Path, *, clock=_now) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        # Several runs opening (or first creating) the outbox at once can meet
        # a briefly locked file: that is contention, not damage, so it is
        # retried, and only a genuinely damaged file is reported as corrupt.
        for attempt in range(OPEN_ATTEMPTS):
            try:
                self._conn = self._open()
                return
            except OutboxCorrupt:
                raise
            except sqlite3.DatabaseError as exc:
                message = str(exc).lower()
                if any(marker in message for marker in _CORRUPTION_MARKERS):
                    raise OutboxCorrupt(str(exc)) from exc
                if not any(marker in message for marker in _BUSY_MARKERS):
                    raise OutboxUnavailable(str(exc)) from exc
                time.sleep(0.05 * (attempt + 1))
        raise OutboxUnavailable(f"{self.path} stayed locked")

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            check = conn.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise OutboxCorrupt(f"integrity check failed: {check[0] if check else 'no result'}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_DDL)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(alert_outbox)")}
            if not {"alert_key", "status", "event_id", "updated_at"} <= columns:
                raise OutboxCorrupt("alert_outbox table has an unexpected shape")
            return conn
        except BaseException:
            conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def get(self, key: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM alert_outbox WHERE alert_key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def is_stale(self, key: str, lease_seconds: float = RESERVATION_LEASE_SECONDS) -> bool:
        """Has the row sat in its state longer than a live run would take?"""
        row = self.get(key)
        if row is None:
            return False
        age = self._clock() - datetime.fromisoformat(row["updated_at"])
        return age > timedelta(seconds=lease_seconds)

    def reserve(self, key: str, *, event_id: str,
                lease_seconds: float = RESERVATION_LEASE_SECONDS) -> Reservation:
        """Claim ``key`` for one send, or say why it is not this run's to send."""
        now = self._clock()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT status, attempts, updated_at FROM alert_outbox WHERE alert_key = ?", (key,)
                ).fetchone()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO alert_outbox (alert_key, event_id, status, attempts, created_at, updated_at)"
                        " VALUES (?, ?, ?, 1, ?, ?)",
                        (key, event_id, PREPARED, now.isoformat(), now.isoformat()),
                    )
                    self._conn.execute("COMMIT")
                    return Reservation(True, None, 1)
                status, attempts = str(row["status"]), int(row["attempts"])
                stale = (
                    status == PREPARED
                    and now - datetime.fromisoformat(row["updated_at"]) > timedelta(seconds=lease_seconds)
                )
                if status == FAILED or stale:
                    # Nothing went out: a refused send, or a run that died
                    # before handing anything to the provider.
                    self._conn.execute(
                        "UPDATE alert_outbox SET event_id = ?, status = ?, attempts = attempts + 1,"
                        " updated_at = ?, last_error = NULL WHERE alert_key = ?",
                        (event_id, PREPARED, now.isoformat(), key),
                    )
                    self._conn.execute("COMMIT")
                    return Reservation(True, status, attempts + 1)
                self._conn.execute("COMMIT")
                return Reservation(False, status, attempts)
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def transition(self, key: str, *, event_id: str, to: str, error: Optional[str] = None,
                   detail: Optional[Mapping[str, Any]] = None) -> None:
        """Move this run's reservation forward; refuses anything out of order."""
        if to not in STATES:
            raise InvalidTransition(f"unknown outbox state {to!r}")
        now = self._clock()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT status, event_id FROM alert_outbox WHERE alert_key = ?", (key,)
                ).fetchone()
                if row is None or row["event_id"] != event_id:
                    raise InvalidTransition(f"{key!r} is not reserved by this run")
                if to not in _TRANSITIONS.get(str(row["status"]), set()):
                    raise InvalidTransition(f"{key!r}: {row['status']} -> {to} is not allowed")
                self._conn.execute(
                    "UPDATE alert_outbox SET status = ?, updated_at = ?, last_error = ?, detail_json = ?"
                    " WHERE alert_key = ?",
                    (to, now.isoformat(), error, json.dumps(dict(detail or {}), default=str), key),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise


def quarantine(path: Path) -> Optional[Path]:
    """Move an unreadable state file aside (kept for inspection); returns where."""
    path = Path(path)
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.corrupt-{_now().strftime('%Y%m%dT%H%M%S%fZ')}")
    path.replace(target)
    for suffix in ("-wal", "-shm"):
        side = path.with_name(path.name + suffix)
        if side.exists():
            side.replace(target.with_name(target.name + suffix))
    return target


__all__ = [
    "ACCEPTED", "DELIVERY_UNKNOWN", "FAILED", "PREPARED", "PROVIDER_SUBMITTED", "AlertOutbox",
    "InvalidTransition", "OutboxCorrupt", "OutboxUnavailable", "Reservation", "quarantine",
]
