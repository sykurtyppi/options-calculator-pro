"""Shared SQLite connection helper for the project.

Why this module exists
----------------------
Across the codebase, many sites called ``sqlite3.connect(path)``
with no PRAGMA configuration. That default produces TWO related
silent-data-loss failure modes under concurrent writers:

  1. **No WAL mode.** SQLite's default ``rollback`` journal
     serializes all writers AND blocks readers during write
     transactions. Two writers contending for the same DB file
     can stack up arbitrarily — including across processes
     (launchd jobs that share the institutional_ml or telemetry
     DBs).

  2. **No busy_timeout.** SQLite's default is ~0ms. When a writer
     hits a lock held by another writer, the call IMMEDIATELY
     raises ``sqlite3.OperationalError: database is locked`` —
     and many call sites just propagate that exception, which
     means the second writer's data is silently dropped on the
     floor.

For the launchd jobs sharing ``telemetry.sqlite`` and
``institutional_ml.db``, the overlap windows on the 12:30 /
21:30 / 22:15 cycle make this concrete: rows go missing whenever
two jobs land in the same write window.

This helper provides ``open_db_conn`` which centralizes the
WAL + busy_timeout configuration used by ``recommendation_ledger``
(set up under Hardening P1-5) and applies it consistently
everywhere.

Audit ref: PR #73 (P1: provider_telemetry, P2: institutional_ml_db).

See also: ``services/recommendation_ledger._open_db`` for the
prior art that this module generalizes; both use the same 5000ms
busy timeout and per-connection WAL pragma.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

# 5-second busy timeout matches the existing recommendation_ledger
# convention (Hardening P1-5 set this in
# services/recommendation_ledger.py:265).
#
# Why 5 seconds: long enough to weather a concurrent writer's
# typical single-row INSERT (microseconds), a multi-row batch
# (single-digit milliseconds), or a brief CHECKPOINT pass; short
# enough that an actually-stuck writer surfaces as a real error
# rather than the connection hanging forever. The recommendation
# ledger picked this value after measuring real launchd contention
# windows; sharing it here keeps the cross-DB story uniform.
DEFAULT_BUSY_TIMEOUT_MS = 5000


def open_db_conn(
    db_path: Union[str, Path],
    *,
    check_same_thread: bool = False,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    enable_wal: bool = True,
) -> sqlite3.Connection:
    """Open a SQLite connection with the project's standard
    configuration (WAL journal mode + busy_timeout).

    Parameters
    ----------
    db_path
        Path to the SQLite file. Parent directory is created if it
        doesn't exist (matches the existing per-module ``_open``
        conventions).

    check_same_thread
        Matches ``sqlite3.connect``'s semantic. ``False`` (the
        default) allows the connection to be shared across threads,
        with the caller responsible for serializing concurrent
        writes via an external lock. Set to ``True`` only when the
        caller wants the sqlite3 module's per-thread safety check
        AND has no need for cross-thread sharing.

    busy_timeout_ms
        Milliseconds to wait for a conflicting writer before raising
        ``SQLITE_BUSY``. Default ``DEFAULT_BUSY_TIMEOUT_MS`` (5000)
        matches the existing recommendation_ledger.

    enable_wal
        Enable WAL journal mode. Default ``True``. WAL is a
        persistent on-disk format change, so the first connection
        to set it converts the file; subsequent opens inherit. Set
        to ``False`` only for special read-only consumers of DBs
        that legitimately need to stay in the default rollback
        journal (rare; not used in this codebase as of PR #73).

    Returns
    -------
    sqlite3.Connection
        A connection with both pragmas applied. The caller is
        responsible for closing it (or using it as a context
        manager, which the sqlite3 module supports for transaction
        commit/rollback but NOT for connection close — explicit
        ``conn.close()`` is still required).
    """
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=check_same_thread)
    if enable_wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    return conn


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "open_db_conn",
]
