"""Tests for services/sqlite_helpers.py and services/jsonl_helpers.py
(PR #73).

Two related classes of bug, two shared helpers:

  1. Many SQLite call sites opened with no WAL and no busy_timeout.
     Concurrent writers raced on the writer lock; the loser raised
     SQLITE_BUSY and (because many call sites only logged + swallowed)
     silently dropped its row. `open_db_conn` centralizes the
     WAL + busy_timeout=5000 convention.

  2. Two JSONL appenders (run_evidence_cycle, run_forward_loop) wrote
     without fcntl.LOCK_EX. Two concurrent processes could interleave
     a partial line and corrupt the JSONL stream the watchdog parses.
     `append_jsonl_locked` extracts the resolver's working pattern
     for shared use.

Tests pin:
  * `open_db_conn` applies WAL + busy_timeout correctly.
  * Concurrent writers don't lose rows (no SQLITE_BUSY exceptions).
  * `append_jsonl_locked` produces well-formed JSONL under concurrent
    writers (every line parses; row count is exact).
  * `append_jsonl_locked` is a no-op on empty input (doesn't touch
    the file).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from services.jsonl_helpers import append_jsonl_locked
from services.sqlite_helpers import DEFAULT_BUSY_TIMEOUT_MS, open_db_conn


# ──────────────────────────────────────────────────────────────────────────
# open_db_conn — pragma application
# ──────────────────────────────────────────────────────────────────────────


def test_open_db_conn_applies_wal_journal_mode(tmp_path: Path) -> None:
    """Verify WAL is actually set (not just attempted). PRAGMA
    journal_mode is persistent on the DB file once applied."""
    db = tmp_path / "test.db"
    conn = open_db_conn(db)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal", (
        f"Expected WAL journal mode; got {mode!r}. If this is "
        f"'delete' or anything else, open_db_conn's WAL pragma was "
        f"reverted or the file is read-only."
    )


def test_open_db_conn_applies_busy_timeout(tmp_path: Path) -> None:
    """busy_timeout pin. SQLite stores the value as milliseconds;
    PRAGMA busy_timeout returns the current value."""
    db = tmp_path / "test.db"
    conn = open_db_conn(db)
    try:
        timeout = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
    finally:
        conn.close()
    assert timeout == DEFAULT_BUSY_TIMEOUT_MS


def test_open_db_conn_creates_parent_directory(tmp_path: Path) -> None:
    """Parent dir auto-creation is part of the contract — every per-
    module _open_db helper in the project relies on it."""
    db = tmp_path / "nested" / "dirs" / "test.db"
    assert not db.parent.exists()
    conn = open_db_conn(db)
    try:
        assert db.parent.is_dir()
        assert db.exists()
    finally:
        conn.close()


def test_open_db_conn_with_wal_disabled(tmp_path: Path) -> None:
    """The enable_wal=False escape hatch (not used in production but
    documented in the helper signature) leaves the journal in
    default mode."""
    db = tmp_path / "test.db"
    conn = open_db_conn(db, enable_wal=False)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    # Default journal mode is "delete" on newly-created DBs.
    assert mode.lower() == "delete"


# ──────────────────────────────────────────────────────────────────────────
# Concurrent-writer SQLite regression (the load-bearing test)
# ──────────────────────────────────────────────────────────────────────────


def test_concurrent_writers_do_not_lose_rows(tmp_path: Path) -> None:
    """The load-bearing PR #73 SQLite regression. Pre-fix, two
    concurrent threads INSERTing rows would race on the writer
    lock and one would hit SQLITE_BUSY immediately, losing its row.
    With WAL + busy_timeout=5000, the second writer waits up to 5s
    for the first to finish — plenty of time for a single-row
    INSERT.

    Setup: open one connection per thread (each gets its own
    file-handle lock), fire N threads each doing M INSERTs, then
    count rows. Pre-fix this test would fail with row_count < N*M
    due to SQLITE_BUSY exceptions. Post-fix the count is exact.
    """
    db = tmp_path / "test.db"
    # Initialize schema with one connection, then close.
    conn = open_db_conn(db)
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.commit()
    finally:
        conn.close()

    n_threads = 8
    rows_per_thread = 25
    errors: list[BaseException] = []

    def writer(worker_id: int) -> None:
        try:
            wconn = open_db_conn(db)
            try:
                for i in range(rows_per_thread):
                    wconn.execute(
                        "INSERT INTO t (val) VALUES (?)",
                        (f"worker={worker_id}_row={i}",),
                    )
                    wconn.commit()
            finally:
                wconn.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive(), "writer thread did not finish"

    assert not errors, (
        f"Concurrent writers must not raise. Got {len(errors)} "
        f"errors, first: {errors[0]!r}. If SQLITE_BUSY appears, "
        f"the busy_timeout pragma was reverted."
    )

    rconn = open_db_conn(db)
    try:
        n = rconn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    finally:
        rconn.close()
    expected = n_threads * rows_per_thread
    assert n == expected, (
        f"Expected {expected} rows from {n_threads} writers × "
        f"{rows_per_thread} rows each; got {n}. Missing rows mean "
        f"writers hit SQLITE_BUSY without retrying — i.e. the "
        f"busy_timeout pragma isn't being applied."
    )


# ──────────────────────────────────────────────────────────────────────────
# append_jsonl_locked — basic shape
# ──────────────────────────────────────────────────────────────────────────


def test_append_jsonl_locked_writes_expected_lines(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    append_jsonl_locked(path, [{"a": 1}, {"b": 2}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}


def test_append_jsonl_locked_appends_not_truncates(tmp_path: Path) -> None:
    """Second call appends rather than overwriting."""
    path = tmp_path / "out.jsonl"
    append_jsonl_locked(path, [{"a": 1}])
    append_jsonl_locked(path, [{"b": 2}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}


def test_append_jsonl_locked_empty_input_is_noop(tmp_path: Path) -> None:
    """Empty iterable shouldn't even touch the file (the parent
    dir won't get created either)."""
    path = tmp_path / "nested" / "out.jsonl"
    append_jsonl_locked(path, [])
    assert not path.exists()
    assert not path.parent.exists()


def test_append_jsonl_locked_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"
    assert not path.parent.exists()
    append_jsonl_locked(path, [{"a": 1}])
    assert path.parent.is_dir()
    assert path.exists()


# ──────────────────────────────────────────────────────────────────────────
# Concurrent-writer JSONL regression
# ──────────────────────────────────────────────────────────────────────────


def test_concurrent_jsonl_writers_produce_well_formed_lines(
    tmp_path: Path,
) -> None:
    """The load-bearing PR #73 JSONL regression. Pre-fix, two
    concurrent processes appending to the same JSONL could
    interleave a partial line mid-write — producing malformed
    JSON that breaks any tail-reader that parses lines into
    objects (services/evidence_health.py does this).

    With fcntl.LOCK_EX serializing the writes, every line is
    atomic relative to other writers. This test fires N threads
    each writing M rows and asserts every resulting line is
    parseable as JSON AND that the total count matches.
    """
    path = tmp_path / "concurrent.jsonl"

    n_threads = 6
    rows_per_thread = 50
    errors: list[BaseException] = []

    def writer(worker_id: int) -> None:
        try:
            for i in range(rows_per_thread):
                # Each writer makes its OWN append call (so the
                # lock cycles N*M times across threads). The
                # payload includes a thread-unique key so we can
                # verify all rows survive.
                append_jsonl_locked(path, [{
                    "worker": worker_id,
                    "row": i,
                    # Long string to make the write large enough
                    # that an unlocked write would have a real
                    # chance of interleaving mid-line.
                    "filler": "x" * 200,
                }])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive()

    assert not errors, (
        f"Concurrent JSONL writers raised: {errors[0]!r}"
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    expected_count = n_threads * rows_per_thread
    assert len(lines) == expected_count, (
        f"Expected {expected_count} lines from {n_threads} writers × "
        f"{rows_per_thread} rows; got {len(lines)}. Mismatch means "
        f"lock acquisition was bypassed and some writes were lost."
    )

    # Every line must parse — no interleaved/malformed JSON.
    seen_pairs = set()
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"Line {i} is not valid JSON (mid-write interleave "
                f"would produce this). Error: {exc}. Line content: "
                f"{line[:200]!r}"
            )
        seen_pairs.add((obj["worker"], obj["row"]))

    # Every (worker, row) pair is unique and accounted for.
    assert len(seen_pairs) == expected_count, (
        f"Expected {expected_count} unique (worker, row) pairs; "
        f"got {len(seen_pairs)}. Some rows were duplicated or lost."
    )
