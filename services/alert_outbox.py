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

Quarantining moves the only record of what was already sent, so a fresh
outbox must not become a new send authority on its own: ``quarantine_outbox``
first writes a durable marker next to the outbox (``tombstone_path``), and
while it exists no alert is sent until an operator has reconciled the
quarantined file and cleared it. If the marker cannot be written, the damaged
outbox is left in place, which keeps every run from sending.

Runs serialize on a sibling lock file (``outbox_lock``) from the marker check
until their reservation commits, and quarantining takes the same lock: a run
can never check "no marker", lose the race to a quarantine, and then create a
fresh outbox that would resend. The outbox is created whole (schema written
under a temporary name, then linked into place), so an empty file or one
without the outbox table is damage, never a fresh start.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from services.sqlite_helpers import enable_wal_mode

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
        _create_outbox_file(self.path)
        if self.path.stat().st_size == 0:
            # A file that exists but holds nothing lost the record it held (or
            # was never written): opening it would start an empty outbox.
            raise OutboxCorrupt("outbox file is empty")
        conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            check = conn.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise OutboxCorrupt(f"integrity check failed: {check[0] if check else 'no result'}")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(alert_outbox)")}
            if not columns:
                raise OutboxCorrupt("outbox file has no alert_outbox table")
            if not {"alert_key", "status", "event_id", "updated_at"} <= columns:
                raise OutboxCorrupt("alert_outbox table has an unexpected shape")
            enable_wal_mode(conn)
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


    def resolve_unknown(self, key: str, *, delivered: bool, note: Optional[str] = None) -> str:
        """An operator checked whether an alert of unknown delivery arrived.

        ``delivered`` marks it accepted (never resent); otherwise failed, so
        the next run may send it. Only a ``delivery_unknown`` row, or a
        ``provider_submitted`` one whose run died (stale), can be resolved.
        Returns the new status.
        """
        now = self._clock()
        to = ACCEPTED if delivered else FAILED
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT status, updated_at FROM alert_outbox WHERE alert_key = ?", (key,)).fetchone()
                if row is None:
                    raise InvalidTransition(f"{key!r} is not in the outbox")
                status = str(row["status"])
                stale_submit = (status == PROVIDER_SUBMITTED and now - datetime.fromisoformat(row["updated_at"])
                                > timedelta(seconds=RESERVATION_LEASE_SECONDS))
                if status != DELIVERY_UNKNOWN and not stale_submit:
                    raise InvalidTransition(f"{key!r} is {status}, not of unknown delivery")
                self._conn.execute(
                    "UPDATE alert_outbox SET status = ?, updated_at = ?, last_error = ?, detail_json = ?"
                    " WHERE alert_key = ?",
                    (to, now.isoformat(), None if delivered else "operator: not delivered",
                     json.dumps({"resolved_by_operator": True, "from": status, "note": note}), key),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return to

    def unresolved(self) -> list[dict]:
        """Rows whose delivery is unknown (and stale submissions), for the operator."""
        cutoff = (self._clock() - timedelta(seconds=RESERVATION_LEASE_SECONDS)).isoformat()
        rows = self._conn.execute(
            "SELECT alert_key, status, updated_at, last_error FROM alert_outbox WHERE status = ?"
            " OR (status = ? AND updated_at < ?) ORDER BY updated_at",
            (DELIVERY_UNKNOWN, PROVIDER_SUBMITTED, cutoff),
        ).fetchall()
        return [dict(row) for row in rows]


def _create_outbox_file(path: Path) -> None:
    """Create the outbox with its schema in one step, if it does not exist.

    The schema is written under a temporary name and hard-linked into place,
    which fails if a file appeared meanwhile: the outbox path never holds a
    half-made database.
    """
    if path.exists():
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.new")
    try:
        with contextlib.closing(sqlite3.connect(str(tmp))) as conn:
            conn.executescript(_DDL)
            conn.commit()
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        with contextlib.suppress(FileNotFoundError):
            tmp.with_name(tmp.name + "-journal").unlink()


def lock_path(outbox_path: Path) -> Path:
    outbox_path = Path(outbox_path)
    return outbox_path.with_name(outbox_path.name + ".lock")


@contextlib.contextmanager
def outbox_lock(outbox_path: Path, *, timeout_seconds: float = 60.0) -> Iterator[None]:
    """Hold the outbox's sibling lock (exclusive, across processes).

    Taken by a run from its marker check until its reservation commits, and
    by quarantine. Raises OutboxUnavailable if it cannot be had in time.
    """
    path = lock_path(outbox_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES) or time.monotonic() >= deadline:
                    raise OutboxUnavailable(f"outbox lock {path} not acquired: {exc}") from exc
                time.sleep(0.02)
        yield
    finally:
        os.close(fd)


class QuarantineIncomplete(OSError):
    """The marker was written but moving the damaged outbox aside failed
    part-way; sends stay blocked by the marker."""


def tombstone_path(outbox_path: Path) -> Path:
    """Where the marker of a quarantined outbox is kept (next to the outbox)."""
    outbox_path = Path(outbox_path)
    return outbox_path.with_name(outbox_path.name + ".quarantined.json")


def read_tombstone(outbox_path: Path) -> Optional[dict]:
    """The quarantine marker, or None if there is none. An unreadable marker
    still blocks sending (returned with ``unreadable``)."""
    marker = tombstone_path(outbox_path)
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"unreadable": f"{type(exc).__name__}: {exc}"}
    return payload if isinstance(payload, dict) else {"unreadable": "not an object"}


def quarantine_outbox(outbox_path: Path, *, reason: str, expected_inode: Optional[int] = None) -> Optional[Path]:
    """Move a damaged outbox aside, leaving a durable marker that blocks sends.

    Call it holding ``outbox_lock``. The marker is written (and fsynced)
    before the move, and created exclusively: if another run already
    quarantined, nothing more is done (None). ``expected_inode`` is the file
    found damaged; if the outbox at the path is no longer that file, it is
    not moved. If the marker cannot be written, nothing is moved and OSError
    propagates: the damaged outbox then keeps blocking every run. If the move
    fails part-way, the marker records the step and QuarantineIncomplete is
    raised.
    """
    outbox_path = Path(outbox_path)
    marker = tombstone_path(outbox_path)
    if marker.exists():
        return None
    try:
        current_inode = outbox_path.stat().st_ino
    except FileNotFoundError:
        current_inode = None
    stamp = _now()
    target = outbox_path.with_name(f"{outbox_path.name}.corrupt-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}")
    move = current_inode is not None and (expected_inode is None or current_inode == expected_inode)
    payload = {"quarantined_at": stamp.isoformat(), "outbox": str(outbox_path),
               "moved_to": str(target) if move else None, "reason": str(reason)[:500]}
    _write_marker(marker, payload, exclusive=True)
    if not move:
        return None
    try:
        return quarantine(outbox_path, target=target)
    except QuarantineIncomplete as exc:
        payload.update(move_failed=str(exc))
        with contextlib.suppress(OSError):
            _write_marker(marker, payload, exclusive=False)
        raise


def _write_marker(marker: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    tmp = marker.with_name(marker.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if exclusive:
            os.link(tmp, marker)          # fails if a marker appeared meanwhile
        else:
            os.replace(tmp, marker)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def clear_tombstone(outbox_path: Path) -> bool:
    """An operator has reconciled the quarantined outbox: allow sends again."""
    marker = tombstone_path(outbox_path)
    if not marker.exists():
        return False
    marker.unlink()
    return True


def quarantine(path: Path, *, target: Optional[Path] = None) -> Optional[Path]:
    """Move an unreadable file aside (kept for inspection); returns where.

    SQLite side files move first: a -wal left behind would be replayed into
    whatever file next appears at ``path``. A failed step raises
    QuarantineIncomplete naming it.
    """
    path = Path(path)
    if not path.exists():
        return None
    target = target or path.with_name(f"{path.name}.corrupt-{_now().strftime('%Y%m%dT%H%M%S%fZ')}")
    for suffix in ("-wal", "-shm", ""):
        source = path.with_name(path.name + suffix)
        if suffix and not source.exists():
            continue
        try:
            source.replace(target.with_name(target.name + suffix))
        except OSError as exc:
            raise QuarantineIncomplete(f"moving {source.name} failed: {type(exc).__name__}: {exc}") from exc
    return target


__all__ = [
    "ACCEPTED", "DELIVERY_UNKNOWN", "FAILED", "PREPARED", "PROVIDER_SUBMITTED", "AlertOutbox",
    "InvalidTransition", "OutboxCorrupt", "OutboxUnavailable", "QuarantineIncomplete", "Reservation",
    "clear_tombstone", "lock_path", "outbox_lock", "quarantine", "quarantine_outbox", "read_tombstone",
    "tombstone_path",
]
