#!/usr/bin/env python3
"""Rotate launchd-generated log files by size, gzip, and retention.

These launchd-produced ``*_launchd*.log`` files don't go through
Python's logging ``RotatingFileHandler`` — they're written directly via
shell ``>>`` redirection from the wrapper scripts. Without rotation
they grow until disk fills (the daily evidence cycle log was already
780 KB after one month of runs; 90 days unattended would push 2–3 MB,
and a verbose run can spike).

Targets:

    *_launchd.log
    *_launchd_stderr.log
    *_launchd_stdout.log
    *_launchd.stderr.log
    *_launchd.stdout.log

under ``~/.options_calculator_pro/logs/``. The Python-logger files
(``__main__.log``, ``services.*.log``) handle their own rotation via
``RotatingFileHandler(backupCount=N)`` and are explicitly NOT touched
by this script.

Algorithm (per matched file ``foo.log``):

  1. Pick up any orphan ``foo.log.<ts>`` from a prior interrupted run
     and gzip it.
  2. If ``foo.log`` size ≥ ``--max-bytes`` (default 5 MB), rename to
     ``foo.log.<UTC-YYYYMMDDhhmmss>`` and gzip → ``.gz``. launchd
     wrappers append via shell ``>>``, which creates a fresh empty
     file on next write — no need to touch the original handle.
  3. Keep the most recent ``--keep`` rotated ``.gz`` archives per
     file; ``unlink`` the rest.

Crash safety: rename → gzip → unlink-original is the standard
crash-safe rotation sequence. An interrupt between rename and gzip
leaves an uncompressed ``.log.<ts>`` orphan that step 1 picks up next
run. The threshold check in step 2 makes the script idempotent on
repeated runs with no new log writes.

**Self-logs are skipped by default.** When the launchd-driven rotation
job is running, the wrapper holds an fd on ``log_rotation_launchd.log``
(via shell ``>>`` redirection) and launchd holds fds on the plist's
``StandardOutPath`` / ``StandardErrorPath`` siblings. Renaming +
unlinking any of those would land the wrapper's completion marker —
written after Python exits — in a deleted inode. Concurrent writes
during the gzip step could also corrupt the archive. So we skip the
``log_rotation_launchd*`` set by default. Self-logs grow at ~hundreds
of bytes per run; a year of unattended growth is ~70 KB. If that ever
becomes a problem, an operator can pass ``--include-self`` when
invoking the script manually (NOT from within the launchd job).

Run daily via launchd at 03:00 local — far from the daily evidence
cycle (~22:15 local) and resolver (~12:30 local) so no other job has
an active handle.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_LOG_DIR = Path.home() / ".options_calculator_pro" / "logs"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_KEEP = 7

# Launchd outputs we own. Targeting "launchd" in the filename is
# deliberate — it avoids the Python-logger files, which manage their
# own rotation.
_LAUNCHD_LOG_GLOBS: tuple[str, ...] = (
    "*_launchd.log",
    "*_launchd_stderr.log",
    "*_launchd_stdout.log",
    "*_launchd.stderr.log",
    "*_launchd.stdout.log",
)

# Codex web-audit follow-up P1: when the rotator is itself running, the
# shell wrapper holds a write fd on ``log_rotation_launchd.log`` via
# ``>>`` redirection, and launchd holds fds on the plist's
# ``StandardOutPath`` / ``StandardErrorPath`` (the two ``_stdout/stderr``
# siblings). If the rotator renames+unlinks any of those, the wrapper's
# completion marker — written AFTER Python exits but before the shell's
# redirection block closes — lands in a deleted inode and is lost. Worse,
# concurrent writes during the gzip step can corrupt the archive.
#
# The fix here is the simplest one that works: skip the rotator's own
# logs by default. They're operational logs for a fast script (rotation
# completes in seconds), each run writes a few hundred bytes, and
# realistic growth caps at ~70 KB/year. If one ever grows large enough
# to need rotation, an operator can either pass ``--include-self`` to
# this script when invoking manually (NOT from the launchd job; that
# self-rotation race is the whole bug) or use ``gzip`` + truncate on
# the file directly.
_SELF_LOG_BASENAMES: frozenset[str] = frozenset({
    "log_rotation_launchd.log",
    "log_rotation_launchd_stderr.log",
    "log_rotation_launchd_stdout.log",
    "log_rotation_launchd.stderr.log",
    "log_rotation_launchd.stdout.log",
})


def _utc_timestamp() -> str:
    """Compact UTC timestamp suitable for the rotated filename."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _gzip_in_place(src: Path) -> Path:
    """Compress *src* into ``<src>.gz`` and remove the original.

    Atomic in the failure sense: writes to a sibling ``.tmp.gz`` first,
    then ``os.replace``\\ s into the final name, then unlinks the
    source. An interrupt between gzip and unlink leaves the .gz next
    to the original; the next run sees the source still present and
    re-gzips it (overwriting via the same ``.tmp.gz`` path).
    """
    dst = src.with_suffix(src.suffix + ".gz")
    tmp = src.with_suffix(src.suffix + ".tmp.gz")
    try:
        with src.open("rb") as fin, gzip.open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    src.unlink()
    return dst


def _find_orphan_uncompressed(base: Path) -> list[Path]:
    """Return uncompressed ``base.log.<ts>`` files left from an
    interrupted rotation. These are the dot-suffixed siblings of
    *base* that don't end in ``.gz``."""
    parent = base.parent
    pattern = f"{base.name}.*"
    out: list[Path] = []
    for p in parent.glob(pattern):
        if p == base or p.name.endswith(".gz") or p.name.endswith(".tmp.gz"):
            continue
        # Skip *_launchd.stderr.log siblings of foo_launchd.log — the
        # glob ``foo_launchd.log.*`` matches them too, but they're a
        # different file we'll process in their own pass.
        if p.suffix in {".log"}:
            continue
        out.append(p)
    return out


def _enforce_retention(base: Path, keep: int) -> list[Path]:
    """Delete the oldest ``base.log.*.gz`` files so that at most
    *keep* remain. Returns the list of deleted paths."""
    parent = base.parent
    archives = sorted(
        (p for p in parent.glob(f"{base.name}.*.gz") if not p.name.endswith(".tmp.gz")),
        key=lambda p: p.stat().st_mtime,
    )
    deleted: list[Path] = []
    if len(archives) <= keep:
        return deleted
    for old in archives[: len(archives) - keep]:
        try:
            old.unlink()
            deleted.append(old)
        except OSError as exc:
            logger.warning("rotate_launchd_logs: failed to delete %s: %s", old, exc)
    return deleted


def rotate_one(
    base: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    keep: int = DEFAULT_KEEP,
) -> dict:
    """Rotate a single log file. Returns a summary dict."""
    summary: dict = {
        "path": str(base),
        "exists": base.exists(),
        "size_bytes": 0,
        "rotated": False,
        "rotated_to": None,
        "orphans_gzipped": [],
        "archives_dropped": [],
    }
    if not base.exists() or not base.is_file():
        return summary

    summary["size_bytes"] = base.stat().st_size

    # Step 1: clean up any orphan uncompressed rotations from a
    # previous interrupted run.
    for orphan in _find_orphan_uncompressed(base):
        try:
            gz = _gzip_in_place(orphan)
            summary["orphans_gzipped"].append(str(gz))
        except OSError as exc:
            logger.warning(
                "rotate_launchd_logs: failed to gzip orphan %s: %s", orphan, exc
            )

    # Step 2: rotate the active log if it's over threshold.
    if summary["size_bytes"] >= max_bytes:
        ts = _utc_timestamp()
        rotated = base.with_suffix(base.suffix + f".{ts}")
        # Defensive: if a same-second collision exists (operator
        # invoked the script twice within one second), bump suffix.
        n = 1
        while rotated.exists():
            rotated = base.with_suffix(base.suffix + f".{ts}-{n}")
            n += 1
        os.rename(base, rotated)
        try:
            gz = _gzip_in_place(rotated)
            summary["rotated"] = True
            summary["rotated_to"] = str(gz)
        except OSError as exc:
            # The rename succeeded but gzip failed. Leave the orphan
            # in place — the next run will pick it up via step 1.
            logger.warning(
                "rotate_launchd_logs: rename succeeded but gzip failed for %s: %s",
                rotated,
                exc,
            )

    # Step 3: enforce retention regardless of whether we rotated this
    # call — even on a no-op run, old archives should be GC'd.
    dropped = _enforce_retention(base, keep)
    summary["archives_dropped"] = [str(p) for p in dropped]
    return summary


def rotate_launchd_logs(
    log_dir: Path = DEFAULT_LOG_DIR,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    keep: int = DEFAULT_KEEP,
    include_self: bool = False,
) -> list[dict]:
    """Rotate every launchd log under *log_dir*. Returns a list of
    per-file summaries (in the order files were processed).

    Self-logs (the rotator's own ``log_rotation_launchd*`` files) are
    skipped by default — the rotator's wrapper holds an fd on them
    while it's running, and rename+unlink would lose the completion
    marker (Codex web-audit follow-up P1). Pass ``include_self=True``
    only when invoking the script manually from outside the launchd
    job; the launchd-driven runs MUST leave it false.
    """
    if not log_dir.exists():
        return []
    seen: set[Path] = set()
    targets: list[Path] = []
    for pattern in _LAUNCHD_LOG_GLOBS:
        for p in log_dir.glob(pattern):
            if p in seen:
                continue
            if not include_self and p.name in _SELF_LOG_BASENAMES:
                continue
            seen.add(p)
            targets.append(p)
    targets.sort()
    return [rotate_one(t, max_bytes=max_bytes, keep=keep) for t in targets]


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory to scan for launchd logs (default: {DEFAULT_LOG_DIR}).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Size threshold for rotation (default: {DEFAULT_MAX_BYTES}).",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=DEFAULT_KEEP,
        help=f"Number of rotated .gz archives to keep per file (default: {DEFAULT_KEEP}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be rotated, but don't modify any files.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help=(
            "Override the self-log skip and rotate this script's own logs. "
            "Safe ONLY when invoking manually from outside the launchd job; "
            "the launchd-driven run holds fds on the self-logs and would "
            "lose its completion marker. See rotate_launchd_logs.py "
            "module docstring for the rationale."
        ),
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        # Just report sizes; never call rotate_one.
        log_dir: Path = args.log_dir
        if not log_dir.exists():
            print(f"log dir does not exist: {log_dir}")
            return 0
        seen: set[Path] = set()
        for pattern in _LAUNCHD_LOG_GLOBS:
            for p in log_dir.glob(pattern):
                if p in seen:
                    continue
                seen.add(p)
                size = p.stat().st_size
                is_self = p.name in _SELF_LOG_BASENAMES
                if is_self and not args.include_self:
                    marker = "skip(self)"
                elif size >= args.max_bytes:
                    marker = "WOULD ROTATE"
                else:
                    marker = "ok"
                print(f"  [{marker}] {p.name}: {_format_size(size)}")
        return 0

    started = time.perf_counter()
    summaries = rotate_launchd_logs(
        args.log_dir,
        max_bytes=args.max_bytes,
        keep=args.keep,
        include_self=args.include_self,
    )
    rotated = sum(1 for s in summaries if s["rotated"])
    orphans = sum(len(s["orphans_gzipped"]) for s in summaries)
    dropped = sum(len(s["archives_dropped"]) for s in summaries)
    elapsed = time.perf_counter() - started
    print(
        f"rotate_launchd_logs: scanned={len(summaries)} rotated={rotated} "
        f"orphans_gzipped={orphans} archives_dropped={dropped} "
        f"elapsed={elapsed:.2f}s"
    )
    for s in summaries:
        if s["rotated"] or s["orphans_gzipped"] or s["archives_dropped"]:
            print(
                f"  {Path(s['path']).name}: "
                f"size={_format_size(s['size_bytes'])} "
                f"rotated={s['rotated']} "
                f"orphans={len(s['orphans_gzipped'])} "
                f"dropped={len(s['archives_dropped'])}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
