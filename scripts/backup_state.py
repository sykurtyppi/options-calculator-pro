#!/usr/bin/env python3
"""Hot, consistent snapshot of the options_calculator_pro state dir.

Closes the data-loss recovery gap surfaced by PR #62 (deployment
runbook). Until this script, the only way to recover from a corrupt
SQLite, an `rm -rf` typo, or a disk failure was "you can't." This
script + the companion `restore_state.py` give the operator a
documented, tested path.

What gets backed up
-------------------
Everything under ``~/.options_calculator_pro/`` (or the path passed
to ``--state-dir``), EXCEPT:

  * The ``backups/`` subdirectory itself — avoids recursive backups
    of backups blowing up disk usage.
  * ``state/*.lock`` directories — ephemeral mkdir-as-lock guards.
    They get recreated by the next wrapper invocation; backing them
    up would just confuse a restore by leaving stale locks behind.

SQLite files (``*.sqlite``) are copied via the **Online Backup API**
(``sqlite3.Connection.backup``), not raw ``cp``. The Online Backup
API produces a consistent snapshot even while another process is
actively writing to the source DB — exactly what we need for a
backup running on a cron schedule while wrapper jobs may be
executing.

All other files are copied byte-for-byte (logs, JSONL telemetry,
JSON state).

Archive format
--------------
A timestamped ``.tar.gz`` written to the output directory. The
filename includes a UTC ISO-ish timestamp and the schema version of
this script, so a future restore can reject archives produced by an
incompatible writer.

Default output location: ``~/.options_calculator_pro/backups/``.

For real disaster recovery, that's NOT a safe location — a disk
failure that destroys the state dir would destroy the backups
alongside it. The runbook (docs/DEPLOYMENT.md) recommends pointing
``--output-dir`` at an external sync folder (Dropbox / iCloud
Drive / external mount).

Retention
---------
After writing the new archive, the script prunes archives in the
output directory keeping only the newest ``--retention`` (default
14). Prune is by filename order — since timestamps are
lexicographically sortable, newest-N retention is unambiguous.

Exit codes
----------
  0 — backup written and (if applicable) pruned successfully.
  1 — state directory missing or empty (nothing to back up).
  2 — fatal error during backup (SQLite snapshot failed, tar write
      failed, etc.). Details on stderr.

Concurrency
-----------
Safe to run while wrapper jobs are executing. The Online Backup API
holds a shared read lock on each SQLite file for the duration of
the snapshot; concurrent writers may experience momentary BUSY
retries but will not deadlock or corrupt.

NOT safe to run two backups simultaneously into the same output
directory — they would race on the temp dir and possibly produce
malformed archives. The runbook's cron schedule should not overlap
itself.
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

# Bumped whenever the on-disk archive layout changes in a way that
# breaks restore. The restore script will refuse to extract an
# archive with a different major version. Today there's only one
# layout so the constant is mostly forward-looking.
ARCHIVE_SCHEMA_VERSION = 1

DEFAULT_STATE_DIR = Path.home() / ".options_calculator_pro"
DEFAULT_RETENTION = 14

# Paths inside the state dir we DO NOT include in the backup.
# Relative to the state dir root.
_SKIP_PATH_NAMES = frozenset({
    # Self-skip: backups subdir, otherwise we'd recursively include
    # every prior archive on each run.
    "backups",
})


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    state_dir: Path = args.state_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not state_dir.exists():
        print(
            f"[backup_state] state dir does not exist: {state_dir}",
            file=sys.stderr,
        )
        return 1
    if not state_dir.is_dir():
        print(
            f"[backup_state] state dir is not a directory: {state_dir}",
            file=sys.stderr,
        )
        return 2
    # Treat a state dir whose only content is the backups/ subdir
    # itself as "nothing to back up" — common immediately after a
    # restore-from-archive flow where someone re-runs backup before
    # any new state has accumulated.
    payload_entries = [
        p for p in state_dir.iterdir()
        if p.name not in _SKIP_PATH_NAMES
    ]
    if not payload_entries:
        print(
            f"[backup_state] state dir is empty (nothing outside "
            f"backups/): {state_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    archive_path = _next_available_archive_path(
        output_dir=output_dir, timestamp=timestamp,
    )

    try:
        _write_archive(state_dir=state_dir, archive_path=archive_path)
    except Exception as exc:  # noqa: BLE001 — surface everything to operator
        print(
            f"[backup_state] FAILED writing archive {archive_path.name}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        # Best-effort cleanup of partially-written archive.
        if archive_path.exists():
            try:
                archive_path.unlink()
            except OSError:
                pass
        return 2

    size_bytes = archive_path.stat().st_size
    print(
        f"[backup_state] wrote {archive_path.name} ({size_bytes:,} bytes)"
    )

    pruned = _prune_old_archives(output_dir, retention=args.retention)
    for old in pruned:
        print(f"[backup_state] pruned old archive {old.name}")

    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help=(
            f"Directory to back up. Default: {DEFAULT_STATE_DIR}. "
            f"Override for tests or non-default installs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STATE_DIR / "backups",
        help=(
            "Where to write the archive. Defaults to "
            "~/.options_calculator_pro/backups/. For real disaster "
            "recovery, point this at an external sync folder so a "
            "disk failure that destroys the state dir doesn't take "
            "the backups with it."
        ),
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=DEFAULT_RETENTION,
        help=(
            f"Keep this many newest archives in the output dir; "
            f"prune the rest. Default: {DEFAULT_RETENTION}. Set to "
            f"0 to disable pruning entirely."
        ),
    )
    args = parser.parse_args(argv)
    if args.retention < 0:
        parser.error("--retention must be >= 0")
    return args


def _next_available_archive_path(*, output_dir: Path, timestamp: str) -> Path:
    """Return an archive path that doesn't already exist in
    *output_dir*.

    The base name is::

        options_calculator_pro-state-<timestamp>-v<schema>.tar.gz

    If that path is already taken (two backups firing in the same
    UTC second, typically from a script that double-fires by
    mistake), append a numeric counter so the second backup writes
    a distinct archive instead of overwriting the first::

        ...-state-<timestamp>-v<schema>-2.tar.gz
        ...-state-<timestamp>-v<schema>-3.tar.gz
        ...

    Codex P2 (non-blocking) on PR #67: without this, two backups
    in the same second silently overwrite each other. The doc says
    simultaneous backups aren't safe (they race on the staging
    dir and tar write), but the second-collision case is just bad
    UX, not an actual race — a counter suffix turns a destructive
    foot-gun into a recoverable double-write.
    """
    base = (
        f"options_calculator_pro-state-{timestamp}"
        f"-v{ARCHIVE_SCHEMA_VERSION}.tar.gz"
    )
    candidate = output_dir / base
    if not candidate.exists():
        return candidate
    # Collision — find the next free counter suffix. Start at 2
    # because the original (un-suffixed) archive is implicitly "1".
    counter = 2
    while True:
        suffixed = (
            f"options_calculator_pro-state-{timestamp}"
            f"-v{ARCHIVE_SCHEMA_VERSION}-{counter}.tar.gz"
        )
        candidate = output_dir / suffixed
        if not candidate.exists():
            return candidate
        counter += 1
        if counter > 9999:
            # Practically unreachable — if we hit 10000 collisions
            # in a single second something is very wrong. Surface
            # as an exception rather than spin forever.
            raise RuntimeError(
                f"too many archive name collisions in {output_dir} "
                f"for timestamp {timestamp} — investigate "
                f"runaway double-fires of backup_state.py"
            )


def _write_archive(*, state_dir: Path, archive_path: Path) -> None:
    """Materialize the backup into a staging directory, then tar+gz it.

    Two-phase write so a crash during SQLite snapshot doesn't leave
    a half-written tarball at the destination. The staging dir
    auto-cleans via TemporaryDirectory context.
    """
    with tempfile.TemporaryDirectory(prefix="ocp-backup-") as staging_str:
        staging = Path(staging_str)
        staged_root = staging / state_dir.name
        staged_root.mkdir()

        _populate_staging(source=state_dir, dest=staged_root)

        # Write tar.gz from the staging tree. Using a tar.gz (not
        # just .tar) keeps backup archives small enough to be
        # reasonable for cron-scheduled writes to an external sync
        # folder.
        with tarfile.open(archive_path, mode="w:gz") as tf:
            tf.add(staged_root, arcname=state_dir.name)


def _populate_staging(*, source: Path, dest: Path) -> None:
    """Copy the source tree into `dest`, applying SQLite hot backup
    for *.sqlite files and skipping the exclusions documented at the
    top of this module."""
    for entry in source.iterdir():
        if entry.name in _SKIP_PATH_NAMES:
            continue
        target = dest / entry.name
        if entry.is_dir():
            target.mkdir()
            _populate_staging_dir(source_dir=entry, dest_dir=target)
        elif entry.is_file():
            _copy_one_file(src=entry, dst=target)


def _populate_staging_dir(*, source_dir: Path, dest_dir: Path) -> None:
    """Recursive copy of a subdirectory. Same skip rules as the top
    level, plus the lock-directory exclusion.

    Lock directories live under ``state/`` and are conventionally
    named ``*.lock`` (see the wrapper scripts in
    scripts/automation/run_*.sh). Skipping them prevents the backup
    from capturing a stale lock that would block the next wrapper
    run after restore.
    """
    for entry in source_dir.iterdir():
        # State-level lock dirs are mkdir-as-lock guards. Backing
        # them up would carry a stale lock into the restored state.
        if entry.is_dir() and entry.name.endswith(".lock"):
            continue
        target = dest_dir / entry.name
        if entry.is_dir():
            target.mkdir()
            _populate_staging_dir(source_dir=entry, dest_dir=target)
        elif entry.is_file():
            _copy_one_file(src=entry, dst=target)


# SQLite primary database extensions we recognize. Each of these
# triggers a hot snapshot via the Online Backup API. The codebase
# uses `.sqlite` for application-managed stores (recommendation
# ledger, candidate exit resolver) and `.db` for legacy
# institutional analytics DBs.
_SQLITE_PRIMARY_SUFFIXES = frozenset({".sqlite", ".sqlite3", ".db"})


def _is_sqlite_sidecar(name: str) -> bool:
    """Identify SQLite WAL / SHM sidecar files by name suffix.

    SQLite in WAL mode maintains two sidecar files alongside the
    primary DB:
      * ``<db>-wal`` — the write-ahead log
      * ``<db>-shm`` — shared memory index for WAL

    The Online Backup API produces a single consistent primary DB
    that is internally complete — no sidecar needed at restore
    time. If we ALSO copied the sidecars into the backup, restoring
    them alongside the primary could leave SQLite reading stale
    WAL state from before the snapshot, producing an inconsistent
    view. Skip them entirely; they get regenerated automatically
    on the next write to the restored DB.
    """
    return name.endswith("-wal") or name.endswith("-shm")


def _copy_one_file(*, src: Path, dst: Path) -> None:
    """Route SQLite primary files through the Online Backup API;
    everything else through shutil.copy2 (which preserves mtime,
    useful for log rotation's age-based behavior post-restore).

    SQLite WAL/SHM sidecar files are skipped entirely — see
    `_is_sqlite_sidecar` for the rationale.
    """
    if _is_sqlite_sidecar(src.name):
        # Caller's responsibility to not re-create dst; we just
        # silently skip.
        return
    if src.suffix in _SQLITE_PRIMARY_SUFFIXES:
        _hot_backup_sqlite(src=src, dst=dst)
    else:
        shutil.copy2(src, dst)


def _hot_backup_sqlite(*, src: Path, dst: Path) -> None:
    """Use sqlite3's Online Backup API to produce a consistent
    snapshot of *src* at *dst*.

    Why not ``cp`` or ``shutil.copy``: those produce a byte-for-byte
    copy of the file as it sits on disk. If a writer transaction is
    mid-flight (write-ahead log not yet checkpointed), the resulting
    copy may be inconsistent — restoring from it could leave the
    DB in a state that fails ``PRAGMA integrity_check``.

    The Online Backup API copies pages from the source DB to the
    destination DB while holding a shared read lock that ensures
    the snapshot is transaction-consistent. Concurrent writers may
    see brief ``SQLITE_BUSY`` retries but the source database
    stays usable.

    Reference: https://www.sqlite.org/backup.html
    """
    # check_same_thread=False isn't needed here (this code is
    # single-threaded), but matches the convention used elsewhere
    # in the codebase (recommendation_ledger.py:257) so the open
    # call shape is consistent.
    src_conn = sqlite3.connect(str(src))
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            with dst_conn:
                src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


# Regex matching every archive filename shape this script can
# write, including collision-suffixed ones. Used by the retention
# pruner so newly-suffixed archives don't slip past the cleanup
# pass.
#
# Matches:
#   options_calculator_pro-state-<ts>-v<schema>.tar.gz
#   options_calculator_pro-state-<ts>-v<schema>-<counter>.tar.gz
import re as _re  # local alias so the top-of-module import block
                  # stays focused on the public-API imports.
_ARCHIVE_FILENAME_PATTERN = _re.compile(
    r"^options_calculator_pro-state-"
    r"(?P<ts>\d{8}T\d{6}Z)"
    r"-v(?P<version>\d+)"
    r"(?:-(?P<counter>\d+))?"
    r"\.tar\.gz$"
)


def _prune_old_archives(output_dir: Path, *, retention: int) -> list[Path]:
    """Keep the newest *retention* archives in *output_dir*; delete
    the rest. Returns the list of pruned paths so the caller can
    log them.

    `retention=0` disables pruning entirely. A negative value is
    rejected at arg-parse time.

    Eligibility: an archive in *output_dir* is a prune candidate
    iff its filename matches the canonical pattern AND its schema
    version equals the current ``ARCHIVE_SCHEMA_VERSION``. Files
    that don't match — operator-renamed archives, archives from a
    future schema version, unrelated tarballs the operator dropped
    into the folder — are left untouched. This protects manual
    "DEFINITELY_KEEP_<name>.tar.gz" style hold-aside copies and
    makes a future schema-version bump non-destructive to old
    archives.

    Sort: lexicographic on filename. The canonical pattern places
    the timestamp before the counter, so this correctly orders
    archives across timestamps. Within a single same-second
    collision burst the un-suffixed archive sorts AFTER its
    counter-suffixed siblings — a minor ordering quirk that
    doesn't matter for retention since same-second archives are
    operationally equivalent.
    """
    if retention == 0:
        return []
    candidates: list[Path] = []
    for path in output_dir.glob("options_calculator_pro-state-*.tar.gz"):
        match = _ARCHIVE_FILENAME_PATTERN.match(path.name)
        if match is None:
            continue
        if int(match.group("version")) != ARCHIVE_SCHEMA_VERSION:
            continue
        candidates.append(path)
    archives = sorted(candidates)
    if len(archives) <= retention:
        return []
    to_delete = archives[: len(archives) - retention]
    pruned: list[Path] = []
    for old in to_delete:
        try:
            old.unlink()
            pruned.append(old)
        except OSError as exc:
            # Don't fail the whole backup just because we couldn't
            # delete one old archive; the new backup is already
            # safely written.
            print(
                f"[backup_state] WARNING: could not prune {old.name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    return pruned


if __name__ == "__main__":
    raise SystemExit(main())
