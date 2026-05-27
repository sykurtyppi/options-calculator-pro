"""Tests for scripts/backup_state.py and scripts/restore_state.py.

What's covered:

  * Cold roundtrip: backup → restore → bit-for-bit equivalence on
    every non-SQLite file, plus integrity_check pass on every
    restored SQLite.

  * **Hot backup** (the load-bearing test): a concurrent writer
    thread INSERTs rows into a SQLite DB while backup_state runs.
    The restored DB must (a) be openable, (b) pass integrity_check,
    (c) contain a row count somewhere between the snapshot's start
    and end states (i.e. produce a consistent point-in-time view,
    not a torn write). This is the test that proves we actually
    need the sqlite3 Online Backup API instead of cp/shutil.copy.

  * Lock-dir exclusion: state/*.lock directories present at backup
    time are NOT restored.

  * Retention: backup --retention=N keeps newest N archives.

  * Restore safety: refuses non-empty target without --force;
    with --force, preserves prior contents to .pre-restore-...
    sibling before extracting.

  * Integrity failure path: corrupt SQLite blob inside an archive
    is detected at restore time and returns exit code 2 with the
    restored state left in place.

  * Archive schema version mismatch: restore refuses an archive
    whose filename advertises a different schema major version.

Conventions:
  * tmp_path everywhere; never touches the real ~/.options_calculator_pro/.
  * scripts/backup_state.py and restore_state.py are imported as
    modules (not subprocess'd) so failures produce real tracebacks.
"""
from __future__ import annotations

import sqlite3
import tarfile
import threading
import time
from pathlib import Path

import pytest

from scripts import backup_state, restore_state


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _populate_state_dir(state_dir: Path, *, ledger_rows: int = 100) -> None:
    """Create a synthetic state dir matching the production layout:
    logs/, state/, recommendations/recommendation_ledger.sqlite."""
    (state_dir / "logs").mkdir(parents=True)
    (state_dir / "logs" / "evidence_cycle_launchd.log").write_text(
        "fake log content\n", encoding="utf-8"
    )
    (state_dir / "logs" / "candidate_exit_resolutions.jsonl").write_text(
        '{"trade_id": "abc", "status": "ok"}\n', encoding="utf-8"
    )

    (state_dir / "state").mkdir()
    (state_dir / "state" / "alert_dedup.json").write_text(
        '{"last_alert": "2026-05-27"}', encoding="utf-8"
    )

    recs = state_dir / "recommendations"
    recs.mkdir()
    db_path = recs / "recommendation_ledger.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE recs (id INTEGER PRIMARY KEY, symbol TEXT)"
        )
        for i in range(ledger_rows):
            conn.execute("INSERT INTO recs (symbol) VALUES (?)", (f"SYM{i}",))


# ──────────────────────────────────────────────────────────────────────────
# Cold roundtrip
# ──────────────────────────────────────────────────────────────────────────


def test_cold_backup_restore_roundtrip(tmp_path: Path) -> None:
    """A static state dir backs up, restores into a fresh location,
    and produces bit-for-bit equivalent files (plus an integrity-
    clean SQLite)."""
    src = tmp_path / "state"
    _populate_state_dir(src, ledger_rows=50)

    output_dir = tmp_path / "backups"
    rc = backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
    ])
    assert rc == 0

    archives = list(output_dir.glob("*.tar.gz"))
    assert len(archives) == 1

    target = tmp_path / "restored"
    rc = restore_state.main([
        str(archives[0]),
        "--target-dir", str(target),
    ])
    assert rc == 0

    # Non-SQLite files round-trip byte-for-byte.
    assert (target / "logs" / "evidence_cycle_launchd.log").read_text() == \
        (src / "logs" / "evidence_cycle_launchd.log").read_text()
    assert (target / "state" / "alert_dedup.json").read_text() == \
        (src / "state" / "alert_dedup.json").read_text()

    # SQLite row count survives.
    restored_db = target / "recommendations" / "recommendation_ledger.sqlite"
    assert restored_db.exists()
    with sqlite3.connect(str(restored_db)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM recs").fetchone()[0]
        assert n == 50


# ──────────────────────────────────────────────────────────────────────────
# Hot backup — concurrent writer
# ──────────────────────────────────────────────────────────────────────────


def test_hot_backup_under_concurrent_writer_restores_intact(
    tmp_path: Path,
) -> None:
    """The load-bearing test for the Online Backup API.

    Setup:
      * Pre-populate a SQLite DB with 1000 rows.
      * Start a writer thread that inserts rows in a tight loop
        (one INSERT per iteration, no batching) for ~2 seconds.
      * Run backup_state.main while the writer is still going.
      * Stop the writer, restore the archive, integrity_check.

    Expected:
      * Restore succeeds (rc == 0 — integrity_check passes).
      * Restored row count is between the writer's start count
        (1000) and end count (whatever it climbed to), proving the
        snapshot is a transaction-consistent point-in-time view
        rather than a torn read of a half-written page.

    If we used shutil.copy here instead of sqlite3.backup, this
    test would fail intermittently with integrity errors on the
    restored DB. The Online Backup API specifically prevents that.
    """
    src = tmp_path / "state"
    src.mkdir()
    recs = src / "recommendations"
    recs.mkdir()
    db_path = recs / "recommendation_ledger.sqlite"

    # Pre-populate. Using WAL mode mimics the real ledger's
    # configuration (recommendation_ledger.py enables WAL).
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE recs (id INTEGER PRIMARY KEY, symbol TEXT)"
        )
        for i in range(1000):
            conn.execute("INSERT INTO recs (symbol) VALUES (?)", (f"SYM{i}",))

    initial_count = 1000
    stop_event = threading.Event()
    writer_error: list[BaseException] = []

    def _writer() -> None:
        try:
            wconn = sqlite3.connect(str(db_path), timeout=10.0)
            wconn.execute("PRAGMA journal_mode=WAL")
            i = 0
            while not stop_event.is_set():
                wconn.execute(
                    "INSERT INTO recs (symbol) VALUES (?)",
                    (f"LIVE{i}",),
                )
                wconn.commit()
                i += 1
                # Small sleep keeps the test fast while still
                # ensuring the writer is genuinely interleaved
                # with the backup's read cursor.
                time.sleep(0.001)
            wconn.close()
        except BaseException as exc:  # noqa: BLE001
            writer_error.append(exc)

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    # Let the writer establish a few inserts before backup begins.
    time.sleep(0.1)

    output_dir = tmp_path / "backups"
    rc = backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
    ])
    assert rc == 0, "backup must succeed even under concurrent write load"

    # Let the writer continue briefly so the post-backup row count
    # is strictly greater than the snapshot count.
    time.sleep(0.2)
    stop_event.set()
    writer.join(timeout=5.0)
    assert not writer.is_alive(), "writer thread did not stop"
    assert not writer_error, (
        f"writer thread crashed during backup: {writer_error[0]!r}"
    )

    # Final row count after writer stopped — this is the upper
    # bound the restored count must NOT exceed.
    with sqlite3.connect(str(db_path)) as conn:
        final_count = conn.execute("SELECT COUNT(*) FROM recs").fetchone()[0]
    assert final_count > initial_count, (
        "writer should have inserted at least one row during the test"
    )

    # Restore and verify.
    archives = list(output_dir.glob("*.tar.gz"))
    assert len(archives) == 1
    target = tmp_path / "restored"
    rc = restore_state.main([
        str(archives[0]),
        "--target-dir", str(target),
    ])
    assert rc == 0, "restore must succeed AND pass integrity_check"

    restored_db = target / "recommendations" / "recommendation_ledger.sqlite"
    with sqlite3.connect(str(restored_db)) as conn:
        restored_count = conn.execute("SELECT COUNT(*) FROM recs").fetchone()[0]
        # Independent integrity_check on the restored DB.
        integrity = conn.execute("PRAGMA integrity_check").fetchall()
        assert integrity == [("ok",)], (
            f"restored DB failed integrity_check: {integrity}"
        )

    # The snapshot was taken sometime between the writer's start
    # (count==1000) and the writer's stop (count==final_count). Any
    # value in that range is a valid transaction-consistent
    # snapshot. A torn read would produce either a count outside
    # this range OR an integrity_check failure above.
    assert initial_count <= restored_count <= final_count, (
        f"restored row count {restored_count} outside the snapshot "
        f"window [{initial_count}, {final_count}] — likely a torn "
        f"read, not a transaction-consistent snapshot."
    )


# ──────────────────────────────────────────────────────────────────────────
# Lock-dir exclusion
# ──────────────────────────────────────────────────────────────────────────


def test_state_lock_dirs_excluded_from_backup(tmp_path: Path) -> None:
    """Lock dirs (mkdir-as-lock guards) must NOT be restored, since
    a stale lock would block the next wrapper run after restore."""
    src = tmp_path / "state"
    _populate_state_dir(src)
    # Add a synthetic lock directory matching the production
    # convention (state/*.lock).
    lock_dir = src / "state" / "evidence_cycle.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text("12345", encoding="utf-8")

    output_dir = tmp_path / "backups"
    backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
    ])
    archives = list(output_dir.glob("*.tar.gz"))
    target = tmp_path / "restored"
    restore_state.main([
        str(archives[0]),
        "--target-dir", str(target),
    ])
    assert not (target / "state" / "evidence_cycle.lock").exists(), (
        "stale lock dir must not survive a restore — it would "
        "block the next wrapper invocation."
    )
    # But the rest of state/ DID survive.
    assert (target / "state" / "alert_dedup.json").exists()


def test_backups_subdir_excluded_from_backup(tmp_path: Path) -> None:
    """The default output_dir lives INSIDE the state dir
    (~/.options_calculator_pro/backups/). The backup must not
    recursively include itself."""
    src = tmp_path / "state"
    _populate_state_dir(src)
    # Existing prior backup file simulating a previous run.
    prior_backups = src / "backups"
    prior_backups.mkdir()
    (prior_backups / "old-archive.tar.gz").write_bytes(b"x" * 1024)

    output_dir = src / "backups"
    backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
    ])

    # Read the new archive and inspect its contents.
    new_archives = sorted(output_dir.glob("options_calculator_pro-state-*.tar.gz"))
    assert len(new_archives) == 1
    with tarfile.open(new_archives[0]) as tf:
        names = tf.getnames()
    assert not any("backups/" in n for n in names), (
        "backup archive must not contain its own output dir"
    )


# ──────────────────────────────────────────────────────────────────────────
# Retention
# ──────────────────────────────────────────────────────────────────────────


def test_retention_keeps_newest_n_archives(tmp_path: Path) -> None:
    """Pre-seed the output dir with 5 dummy archives, then run a
    real backup with --retention=3. Expect to keep the 3 newest
    archives (2 dummies + the new one), prune the 2 oldest."""
    src = tmp_path / "state"
    _populate_state_dir(src)
    output_dir = tmp_path / "backups"
    output_dir.mkdir()

    # Lexicographically-ordered timestamps so we can assert which
    # archives survive without depending on real time.
    for ts in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z",
               "20260104T000000Z", "20260105T000000Z"):
        (
            output_dir / f"options_calculator_pro-state-{ts}-v1.tar.gz"
        ).write_bytes(b"dummy")

    rc = backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
        "--retention", "3",
    ])
    assert rc == 0

    remaining = sorted(p.name for p in output_dir.glob("*.tar.gz"))
    # 3 newest archives kept; the new archive (timestamp > 2026-01-05)
    # is one of them. So we expect 20260103, 20260104, 20260105 to
    # have been pruned... actually no: with --retention=3 and 6
    # total archives (5 dummies + new), prune the 3 oldest. The
    # new archive is the newest by virtue of its timestamp.
    assert len(remaining) == 3, f"expected 3 archives, got {remaining}"
    # The two oldest dummies should be gone.
    assert "options_calculator_pro-state-20260101T000000Z-v1.tar.gz" not in remaining
    assert "options_calculator_pro-state-20260102T000000Z-v1.tar.gz" not in remaining
    # The two newest dummies AND the new archive should remain.
    assert "options_calculator_pro-state-20260104T000000Z-v1.tar.gz" in remaining
    assert "options_calculator_pro-state-20260105T000000Z-v1.tar.gz" in remaining


def test_retention_zero_disables_pruning(tmp_path: Path) -> None:
    src = tmp_path / "state"
    _populate_state_dir(src)
    output_dir = tmp_path / "backups"
    output_dir.mkdir()
    for ts in ("20260101T000000Z", "20260102T000000Z"):
        (
            output_dir / f"options_calculator_pro-state-{ts}-v1.tar.gz"
        ).write_bytes(b"dummy")

    backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
        "--retention", "0",
    ])
    # All 3 archives present (2 dummies + the new one).
    archives = list(output_dir.glob("*.tar.gz"))
    assert len(archives) == 3


# ──────────────────────────────────────────────────────────────────────────
# Restore safety
# ──────────────────────────────────────────────────────────────────────────


def test_restore_refuses_non_empty_target_without_force(tmp_path: Path) -> None:
    src = tmp_path / "state"
    _populate_state_dir(src)
    output_dir = tmp_path / "backups"
    backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
    ])
    archive = next(output_dir.glob("*.tar.gz"))

    target = tmp_path / "restored"
    target.mkdir()
    (target / "preexisting").write_text("important data\n", encoding="utf-8")

    rc = restore_state.main([str(archive), "--target-dir", str(target)])
    assert rc == 1
    # Preexisting content untouched.
    assert (target / "preexisting").read_text() == "important data\n"


def test_restore_force_preserves_existing_target(tmp_path: Path) -> None:
    """--force is the "I really meant it" escape hatch, but the
    safety contract is that prior contents are moved aside to a
    sibling .pre-restore-... directory BEFORE the archive
    overwrites the target. The operator can still recover."""
    src = tmp_path / "state"
    _populate_state_dir(src)
    output_dir = tmp_path / "backups"
    backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
    ])
    archive = next(output_dir.glob("*.tar.gz"))

    target = tmp_path / "restored"
    target.mkdir()
    (target / "preexisting").write_text("important data\n", encoding="utf-8")

    rc = restore_state.main([
        str(archive), "--target-dir", str(target), "--force",
    ])
    assert rc == 0

    # Restored archive contents present.
    assert (target / "logs" / "evidence_cycle_launchd.log").exists()
    # Prior contents preserved at a sibling .pre-restore-... dir.
    preserve_candidates = list(
        tmp_path.glob("restored.pre-restore-*")
    )
    assert len(preserve_candidates) == 1
    assert (
        preserve_candidates[0] / "preexisting"
    ).read_text() == "important data\n"


# ──────────────────────────────────────────────────────────────────────────
# Integrity failure path
# ──────────────────────────────────────────────────────────────────────────


def test_restore_fails_when_sqlite_corrupt(tmp_path: Path) -> None:
    """Build a hand-crafted archive containing a bogus .sqlite file
    (not a real DB). Restore must extract the tree but exit code 2
    after integrity_check fails."""
    # Build a fake state tree with a corrupt sqlite.
    fake_state = tmp_path / "build" / ".options_calculator_pro"
    (fake_state / "recommendations").mkdir(parents=True)
    (fake_state / "recommendations" / "recommendation_ledger.sqlite").write_bytes(
        b"this is not a sqlite file at all"
    )

    # Tar it up with the schema-matching filename.
    archive_path = tmp_path / (
        "options_calculator_pro-state-20260527T120000Z-v1.tar.gz"
    )
    with tarfile.open(archive_path, mode="w:gz") as tf:
        tf.add(fake_state, arcname=".options_calculator_pro")

    target = tmp_path / "restored"
    rc = restore_state.main([str(archive_path), "--target-dir", str(target)])
    # Exit 2 specifically — distinguishes "extraction worked but
    # integrity failed" from "the whole restore couldn't start"
    # (which is exit 1).
    assert rc == 2
    # The restored tree is LEFT IN PLACE so the operator can
    # inspect. Verified by checking the corrupt file is there.
    corrupt_db = target / "recommendations" / "recommendation_ledger.sqlite"
    assert corrupt_db.exists()


# ──────────────────────────────────────────────────────────────────────────
# Archive schema-version mismatch
# ──────────────────────────────────────────────────────────────────────────


def test_restore_rejects_archive_with_wrong_schema_version(
    tmp_path: Path,
) -> None:
    """Archives produced by a future or past version of backup_state
    that uses a different schema MUST be rejected at the filename
    level — before any extraction happens — so the operator can't
    accidentally restore an incompatible layout into the live
    state dir."""
    # Filename advertises v999 — well outside the current schema.
    bogus_archive = tmp_path / (
        "options_calculator_pro-state-20260527T120000Z-v999.tar.gz"
    )
    with tarfile.open(bogus_archive, mode="w:gz") as tf:
        # Empty archive, but the filename version is what we test.
        pass

    target = tmp_path / "restored"
    rc = restore_state.main([str(bogus_archive), "--target-dir", str(target)])
    assert rc == 1
    # Target not even created (we bailed before extraction).
    assert not target.exists() or not any(target.iterdir())


def test_restore_rejects_archive_with_bad_filename_pattern(
    tmp_path: Path,
) -> None:
    """A tarball with a name that doesn't match the project's
    archive pattern at all (e.g., handed someone's random backup)
    is rejected at the filename level."""
    bogus = tmp_path / "random_backup.tar.gz"
    with tarfile.open(bogus, mode="w:gz"):
        pass
    target = tmp_path / "restored"
    rc = restore_state.main([str(bogus), "--target-dir", str(target)])
    assert rc == 1


def test_restore_rejects_tar_with_path_traversal(tmp_path: Path) -> None:
    """Defense against a malicious archive that uses ``..`` paths
    to write outside the target dir. The restore script's
    extraction loop must reject such members."""
    archive_path = tmp_path / (
        "options_calculator_pro-state-20260527T120000Z-v1.tar.gz"
    )
    # Build a malicious archive manually.
    bad_payload = tmp_path / "evil.txt"
    bad_payload.write_text("pwned", encoding="utf-8")
    with tarfile.open(archive_path, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = bad_payload.stat().st_size
        with bad_payload.open("rb") as f:
            tf.addfile(info, f)

    target = tmp_path / "restored"
    rc = restore_state.main([str(archive_path), "--target-dir", str(target)])
    assert rc == 1
    # The escape file MUST NOT have been written outside the
    # restore target.
    assert not (tmp_path / "escape.txt").exists()


# ──────────────────────────────────────────────────────────────────────────
# Empty/missing source handling
# ──────────────────────────────────────────────────────────────────────────


def test_backup_missing_state_dir_returns_one(tmp_path: Path) -> None:
    rc = backup_state.main([
        "--state-dir", str(tmp_path / "nonexistent"),
        "--output-dir", str(tmp_path / "backups"),
    ])
    assert rc == 1


def test_backup_empty_state_dir_returns_one(tmp_path: Path) -> None:
    """An empty state dir (or one containing only the backups/
    subdir) is treated as 'nothing to back up' — exit 1, not a
    crash. This is the realistic state immediately after a
    restore-from-archive cycle where someone re-runs backup before
    any new state has accumulated."""
    src = tmp_path / "state"
    src.mkdir()
    (src / "backups").mkdir()  # Only the self-excluded subdir.
    rc = backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(src / "backups"),
    ])
    assert rc == 1


# ──────────────────────────────────────────────────────────────────────────
# Multi-extension SQLite + sidecar handling
# ──────────────────────────────────────────────────────────────────────────


def test_dot_db_files_routed_through_online_backup_api(tmp_path: Path) -> None:
    """The live state dir surfaced a real .db file
    (institutional_ml.db) alongside the .sqlite ledger. Both
    extensions are recognized SQLite primary file shapes and both
    must be hot-snapshotted, not raw-copied.

    Smoke that .db files restore intact with a passing
    integrity_check. The Online Backup API path is what makes this
    safe under concurrent writers — without the extension routing,
    .db files would shutil.copy2 and could capture torn writes.
    """
    src = tmp_path / "state"
    src.mkdir()
    (src / "analytics").mkdir()
    db_path = src / "analytics" / "institutional_ml.db"

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (x INT)")
        for i in range(100):
            conn.execute("INSERT INTO t VALUES (?)", (i,))

    output_dir = tmp_path / "backups"
    rc = backup_state.main([
        "--state-dir", str(src),
        "--output-dir", str(output_dir),
    ])
    assert rc == 0

    archive = next(output_dir.glob("*.tar.gz"))
    target = tmp_path / "restored"
    rc = restore_state.main([str(archive), "--target-dir", str(target)])
    assert rc == 0, "restore must pass integrity_check on the .db file"

    restored = target / "analytics" / "institutional_ml.db"
    assert restored.exists()
    with sqlite3.connect(str(restored)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert n == 100


def test_wal_shm_sidecar_files_skipped_in_backup(tmp_path: Path) -> None:
    """SQLite WAL/SHM sidecar files (``*-wal``, ``*-shm``) must NOT
    appear in the archive. Including them would let SQLite at
    restore time read stale WAL state from before the snapshot,
    producing an inconsistent view. The Online Backup API produces
    a single self-contained primary DB; sidecars regenerate
    automatically on first write to the restored DB.
    """
    src = tmp_path / "state"
    src.mkdir()
    (src / "recommendations").mkdir()
    db_path = src / "recommendations" / "recommendation_ledger.sqlite"

    # Force WAL sidecar files to materialize by writing in WAL
    # mode and leaving an open transaction uncheckpointed.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (x INT)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    # Don't close conn yet — sidecars stay materialized.

    try:
        # Verify sidecars exist on disk before backup runs.
        sidecars_before = list(src.rglob("*-wal")) + list(src.rglob("*-shm"))
        assert sidecars_before, (
            "test setup: WAL/SHM sidecars should exist after WAL-mode "
            "write with active connection"
        )

        output_dir = tmp_path / "backups"
        rc = backup_state.main([
            "--state-dir", str(src),
            "--output-dir", str(output_dir),
        ])
        assert rc == 0
    finally:
        conn.close()

    # Inspect archive contents — no sidecar files should be present.
    archive = next(output_dir.glob("*.tar.gz"))
    with tarfile.open(archive) as tf:
        names = tf.getnames()
    sidecars_in_archive = [
        n for n in names if n.endswith("-wal") or n.endswith("-shm")
    ]
    assert not sidecars_in_archive, (
        f"backup must skip WAL/SHM sidecar files, but archive contains: "
        f"{sidecars_in_archive}"
    )

    # The primary DB IS in the archive and restores cleanly.
    target = tmp_path / "restored"
    rc = restore_state.main([str(archive), "--target-dir", str(target)])
    assert rc == 0
    restored = target / "recommendations" / "recommendation_ledger.sqlite"
    assert restored.exists()
    with sqlite3.connect(str(restored)) as rconn:
        n = rconn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert n == 1


def test_backup_negative_retention_rejected(tmp_path: Path) -> None:
    """argparse rejects --retention < 0 with exit code 2 (argparse
    convention)."""
    with pytest.raises(SystemExit) as exc:
        backup_state.main([
            "--state-dir", str(tmp_path),
            "--retention", "-1",
        ])
    assert exc.value.code == 2
