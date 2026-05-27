#!/usr/bin/env python3
"""Restore the options_calculator_pro state dir from a backup archive.

Companion to ``backup_state.py``. Refuses to overwrite a non-empty
target dir unless ``--force`` is passed. After extraction, runs
``PRAGMA integrity_check`` on every restored ``*.sqlite`` file and
fails loudly if any DB does not return the literal string ``ok``.

Usage
-----
    restore_state.py <archive.tar.gz>
        Restore to the default state dir
        (``~/.options_calculator_pro/``). Refuses if it's non-empty.

    restore_state.py <archive.tar.gz> --target-dir /path/to/dir
        Restore to a non-default location. Useful for testing a
        backup's contents without clobbering the live state.

    restore_state.py <archive.tar.gz> --force
        Allow overwriting a non-empty target. Backs the existing
        target up to ``<target>.pre-restore-<timestamp>/`` first so
        an "oh no I picked the wrong archive" mistake is reversible.

Exit codes
----------
  0 — restore succeeded AND every restored SQLite passed
      integrity_check.
  1 — archive missing or unreadable, schema version mismatch, or
      target dir non-empty without --force.
  2 — extraction completed but integrity_check FAILED on at least
      one SQLite file. The restored state is left in place so the
      operator can inspect it; the archive may be corrupt or the
      backup itself may have captured a torn write.

Safety
------
- The pre-restore backup directory created under ``--force`` is
  a sibling of the target dir, NOT a child. So if the restored
  state immediately overwrites a child path of the target with the
  same name, the operator's prior state is still safe.
- Integrity checks happen AFTER extraction so a torn archive
  doesn't strand the operator with no usable state — at worst they
  have to inspect the restored tree manually.
"""
from __future__ import annotations

import argparse
import datetime
import re
import shutil
import sqlite3
import sys
import tarfile
from pathlib import Path
from typing import Iterable

# Must match backup_state.ARCHIVE_SCHEMA_VERSION at the time the
# archive was written. Restore refuses archives with a different
# major version — keeps a future archive format change from
# silently restoring with the wrong assumptions.
ARCHIVE_SCHEMA_VERSION = 1

DEFAULT_TARGET_DIR = Path.home() / ".options_calculator_pro"

# Archive filename shape:
#   options_calculator_pro-state-YYYYMMDDTHHMMSSZ-vN.tar.gz
_ARCHIVE_FILENAME_RE = re.compile(
    r"^options_calculator_pro-state-(?P<ts>\d{8}T\d{6}Z)-v(?P<version>\d+)\.tar\.gz$"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    archive_path: Path = args.archive.expanduser().resolve()
    target_dir: Path = args.target_dir.expanduser().resolve()

    if not archive_path.exists():
        print(
            f"[restore_state] archive does not exist: {archive_path}",
            file=sys.stderr,
        )
        return 1
    if not archive_path.is_file():
        print(
            f"[restore_state] archive is not a regular file: {archive_path}",
            file=sys.stderr,
        )
        return 1

    schema_check = _check_archive_schema(archive_path)
    if schema_check is not None:
        print(f"[restore_state] {schema_check}", file=sys.stderr)
        return 1

    if target_dir.exists() and any(target_dir.iterdir()):
        if not args.force:
            print(
                f"[restore_state] target dir is not empty: {target_dir}\n"
                f"  Pass --force to allow overwrite (the existing "
                f"contents will be moved aside to "
                f"<target>.pre-restore-<timestamp>/ before extraction).",
                file=sys.stderr,
            )
            return 1
        try:
            preserve_path = _move_aside_existing(target_dir)
        except OSError as exc:
            print(
                f"[restore_state] could not preserve existing target "
                f"dir before --force restore: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        print(
            f"[restore_state] existing target preserved at {preserve_path}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        _extract(archive_path=archive_path, target_dir=target_dir)
    except (tarfile.TarError, OSError) as exc:
        print(
            f"[restore_state] FAILED extracting archive: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    integrity_failures = _verify_sqlite_integrity(target_dir)
    if integrity_failures:
        print(
            f"[restore_state] integrity_check FAILED on "
            f"{len(integrity_failures)} SQLite file(s):",
            file=sys.stderr,
        )
        for db_path, message in integrity_failures:
            print(f"  {db_path}: {message}", file=sys.stderr)
        print(
            "[restore_state] Restored state is left in place at "
            f"{target_dir} so you can inspect the damage. If you "
            f"--force'd this restore, your prior state is preserved "
            f"at the .pre-restore-... sibling directory.",
            file=sys.stderr,
        )
        return 2

    print(
        f"[restore_state] restored {archive_path.name} to {target_dir} "
        f"(integrity_check passed)"
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "archive",
        type=Path,
        help="Path to the .tar.gz archive produced by backup_state.py.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=(
            f"Directory to restore into. Default: {DEFAULT_TARGET_DIR}. "
            f"Refuses to overwrite a non-empty dir without --force."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow overwriting a non-empty target dir. Existing "
            "contents are moved to a sibling .pre-restore-<timestamp>/ "
            "directory BEFORE extraction, so the operator can roll "
            "back manually if they realize they picked the wrong "
            "archive."
        ),
    )
    return parser.parse_args(argv)


def _check_archive_schema(archive_path: Path) -> str | None:
    """Verify the archive filename matches the expected schema.

    Returns ``None`` on success; an error message on failure.

    We check the FILENAME, not the archive contents, because that's
    cheap (no extraction needed) and the filename is what
    backup_state writes deliberately. An attacker could fake the
    filename, but this isn't a security boundary — it's a sanity
    check that catches "you handed me a tarball from a different
    project" mistakes.
    """
    match = _ARCHIVE_FILENAME_RE.match(archive_path.name)
    if match is None:
        return (
            f"archive filename does not match expected pattern "
            f"options_calculator_pro-state-<timestamp>-v<n>.tar.gz: "
            f"{archive_path.name}"
        )
    version = int(match.group("version"))
    if version != ARCHIVE_SCHEMA_VERSION:
        return (
            f"archive schema version v{version} does not match this "
            f"restore script (v{ARCHIVE_SCHEMA_VERSION}). If the "
            f"version is newer, upgrade this codebase. If older, "
            f"use a compatible restore_state.py from that era."
        )
    return None


def _move_aside_existing(target_dir: Path) -> Path:
    """Atomically rename the existing target dir to a sibling
    ``.pre-restore-<timestamp>/`` so --force restores are
    recoverable.

    Returns the preservation path. Raises OSError on rename
    failure (caller surfaces).
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    preserve_path = target_dir.with_name(
        f"{target_dir.name}.pre-restore-{timestamp}"
    )
    # If by some chance the preserve path already exists (two
    # restores within the same second), append a counter so we
    # don't clobber the previous preservation.
    counter = 0
    while preserve_path.exists():
        counter += 1
        preserve_path = target_dir.with_name(
            f"{target_dir.name}.pre-restore-{timestamp}-{counter}"
        )
    target_dir.rename(preserve_path)
    return preserve_path


def _extract(*, archive_path: Path, target_dir: Path) -> None:
    """Extract the archive's CONTENTS into target_dir.

    The archive's top-level entry is the state-dir basename
    at the time of backup (e.g. ``.options_calculator_pro/``). We
    strip that top-level entry during extraction so the contents
    land directly inside *target_dir*. This decouples the restore
    target's name from whatever basename was used at backup time —
    the operator can restore a ``~/.options_calculator_pro/``
    backup into ``/tmp/restored_state/`` and find the contents at
    ``/tmp/restored_state/logs/...``, not
    ``/tmp/restored_state/.options_calculator_pro/logs/...``.

    Defensive validation per member:
      * Absolute paths rejected (would write outside target).
      * Any ``..`` component rejected (path traversal).
      * Empty paths or root-only entries skipped.
    """
    with tarfile.open(archive_path, mode="r:gz") as tf:
        members = tf.getmembers()
        # First pass: reject the whole archive if ANY member tries
        # to escape via absolute paths or `..` traversal. Bailing
        # before extraction means no partial-write left to clean
        # up.
        for member in members:
            # Absolute paths.
            if member.name.startswith("/"):
                raise tarfile.TarError(
                    f"unsafe path in archive: {member.name!r}"
                )
            # `..` anywhere in the path (including a leading
            # `..` like "../escape.txt", which has Path parts
            # ("..", "escape.txt")).
            if ".." in Path(member.name).parts:
                raise tarfile.TarError(
                    f"unsafe path in archive: {member.name!r}"
                )

        # Strip the leading directory component from each member's
        # name so contents land inside target_dir at the same
        # relative depth they had inside the original state dir.
        for member in members:
            parts = Path(member.name).parts
            if len(parts) <= 1:
                # Top-level dir entry itself (the basename) —
                # contents arrive via its children, so skip.
                continue
            stripped = Path(*parts[1:])
            # Reassign the member's name in-place; tarfile.extract
            # respects the rewritten name.
            member.name = str(stripped)
            tf.extract(member, target_dir)


# Must match backup_state._SQLITE_PRIMARY_SUFFIXES exactly. Kept
# separate so this module remains stand-alone for restore-only
# distribution (someone restoring on a fresh machine doesn't need
# the backup module installed).
_SQLITE_PRIMARY_SUFFIXES = frozenset({".sqlite", ".sqlite3", ".db"})


def _verify_sqlite_integrity(
    target_dir: Path,
) -> list[tuple[Path, str]]:
    """Run ``PRAGMA integrity_check`` on every SQLite primary file
    (``*.sqlite``, ``*.sqlite3``, ``*.db``) under *target_dir*.
    Returns a list of ``(db_path, error_message)`` for every DB
    that did NOT return the literal string ``ok``.

    Empty return value means all DBs are clean.

    integrity_check is the standard SQLite consistency probe. It
    reads every page and verifies internal cross-references; if
    the backup captured a torn write or the archive was corrupt
    during transit, this is where we catch it.

    WAL/SHM sidecar files are not checked directly because they
    aren't standalone databases — SQLite reads them only as
    auxiliaries to a primary DB, and `PRAGMA integrity_check` run
    against a primary DB transitively validates the sidecar state.
    """
    failures: list[tuple[Path, str]] = []
    candidates: list[Path] = []
    for suffix in _SQLITE_PRIMARY_SUFFIXES:
        candidates.extend(target_dir.rglob(f"*{suffix}"))
    for db_path in sorted(set(candidates)):
        try:
            conn = sqlite3.connect(str(db_path))
        except sqlite3.Error as exc:
            failures.append((db_path, f"could not open: {exc}"))
            continue
        try:
            cursor = conn.execute("PRAGMA integrity_check")
            rows = cursor.fetchall()
        except sqlite3.Error as exc:
            failures.append((db_path, f"integrity_check raised: {exc}"))
            conn.close()
            continue
        conn.close()
        # PRAGMA integrity_check returns the single string "ok" on
        # success, or one row per detected problem otherwise.
        if rows == [("ok",)]:
            continue
        msg = "; ".join(str(row[0]) for row in rows) or "no rows returned"
        failures.append((db_path, msg))
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
