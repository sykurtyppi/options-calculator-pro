"""Unit tests for scripts/rotate_launchd_logs.py.

The rotator is small but has several invariants worth pinning down so
a future "just one more refactor" doesn't quietly break the
crash-safety semantics:

  - Files under the threshold MUST NOT be rotated (idempotency).
  - Files at or over the threshold ARE rotated AND gzipped, AND the
    original file is removed (or recreated empty by the next writer).
  - Orphan ``foo.log.<ts>`` (an interrupt between rename and gzip)
    MUST be picked up and gzipped on the next run.
  - Retention drops the OLDEST .gz archives once the per-file count
    exceeds ``--keep``.
  - The rotator only targets ``*_launchd*.log`` shapes; Python-logger
    files (``__main__.log`` etc.) MUST NOT be touched.
"""
from __future__ import annotations

import gzip
import os
import time
from pathlib import Path

import pytest

from scripts import rotate_launchd_logs as r


def _make_log(path: Path, size_bytes: int) -> Path:
    """Create *path* with exactly *size_bytes* bytes of filler."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``b'x' * size_bytes`` is fine for the sizes we test (≤ few MB);
    # gzip-friendly content (highly compressible) keeps the test fast.
    path.write_bytes(b"x" * size_bytes)
    return path


# ── Threshold / no-op semantics ──────────────────────────────────────────


def test_file_under_threshold_is_not_rotated(tmp_path: Path) -> None:
    """The whole point of size-based rotation: small files stay put.
    Without this, every daily run would gzip every log regardless of
    size, churning disk and inflating the archive count."""
    log = _make_log(tmp_path / "demo_launchd.log", 1024)
    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    assert len(summaries) == 1
    assert summaries[0]["rotated"] is False
    assert summaries[0]["rotated_to"] is None
    assert log.exists()
    assert log.stat().st_size == 1024
    # No .gz archives should appear for an under-threshold file.
    assert list(tmp_path.glob("*.gz")) == []


def test_repeat_runs_on_quiescent_dir_are_idempotent(tmp_path: Path) -> None:
    """Running the rotator three times in a row with no writes between
    them must produce the same on-disk state as one run. Catches a
    class of bug where ``--keep`` mis-counts on subsequent passes."""
    _make_log(tmp_path / "demo_launchd.log", 6_000_000)
    r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=3)
    snapshot_after_first = sorted(p.name for p in tmp_path.iterdir())
    for _ in range(3):
        r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=3)
    assert sorted(p.name for p in tmp_path.iterdir()) == snapshot_after_first


# ── Rotation + gzip ──────────────────────────────────────────────────────


def test_file_over_threshold_is_rotated_and_gzipped(tmp_path: Path) -> None:
    """A file at/over threshold must produce exactly one ``.log.<ts>.gz``
    archive containing the original bytes, and the original .log must
    be gone (the next writer will create a fresh empty one via ``>>``).
    """
    log = _make_log(tmp_path / "demo_launchd.log", 6_000_000)
    original_bytes = log.read_bytes()

    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    assert summaries[0]["rotated"] is True
    assert summaries[0]["rotated_to"] is not None
    assert not log.exists()  # original gone
    archives = list(tmp_path.glob("demo_launchd.log.*.gz"))
    assert len(archives) == 1
    # Round-trip through gzip recovers the original payload.
    with gzip.open(archives[0], "rb") as fh:
        assert fh.read() == original_bytes


def test_rotation_targets_all_launchd_log_shapes(tmp_path: Path) -> None:
    """Verify the glob set covers the actual shapes launchd produces:
    ``foo_launchd.log``, ``foo_launchd_stderr.log`` (current resolver),
    and ``foo_launchd.stderr.log`` (legacy)."""
    shapes = [
        "demo_launchd.log",
        "demo_launchd_stderr.log",
        "demo_launchd_stdout.log",
        "demo_launchd.stderr.log",
        "demo_launchd.stdout.log",
    ]
    for name in shapes:
        _make_log(tmp_path / name, 6_000_000)

    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    rotated_names = {Path(s["path"]).name for s in summaries if s["rotated"]}
    assert rotated_names == set(shapes), (
        f"Some launchd-log shape isn't covered by _LAUNCHD_LOG_GLOBS: "
        f"got {rotated_names}, expected {set(shapes)}"
    )


def test_python_logger_files_are_not_rotated(tmp_path: Path) -> None:
    """The Python-logger files manage their own rotation via
    RotatingFileHandler(backupCount=N). The launchd rotator must NEVER
    touch them — doing so would interleave two rotation schemes and
    leak file handles."""
    _make_log(tmp_path / "__main__.log", 11_000_000)
    _make_log(tmp_path / "services.institutional_ml_db.log", 11_000_000)
    _make_log(tmp_path / "scripts.institutional_backfill.log", 11_000_000)

    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    assert summaries == []
    # And the files are still there at their original size.
    assert (tmp_path / "__main__.log").stat().st_size == 11_000_000
    assert (tmp_path / "services.institutional_ml_db.log").stat().st_size == 11_000_000


# ── Crash recovery ───────────────────────────────────────────────────────


def test_orphan_uncompressed_rotation_is_gzipped_on_next_run(
    tmp_path: Path,
) -> None:
    """Simulate an interrupt between rename and gzip: a
    ``foo.log.<ts>`` sits next to ``foo.log`` with no ``.gz``. The next
    rotator run must gzip it and clean up the orphan."""
    base = _make_log(tmp_path / "demo_launchd.log", 1024)  # under-threshold
    orphan = tmp_path / "demo_launchd.log.20260101000000"
    orphan.write_bytes(b"recovered bytes")

    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    # Active log was under threshold, so not rotated this run.
    assert summaries[0]["rotated"] is False
    # But the orphan was picked up and gzipped.
    assert orphan.exists() is False
    gz = tmp_path / "demo_launchd.log.20260101000000.gz"
    assert gz.exists()
    with gzip.open(gz, "rb") as fh:
        assert fh.read() == b"recovered bytes"
    # And the original .log is still there for future writes.
    assert base.exists()


# ── Retention ────────────────────────────────────────────────────────────


def test_retention_drops_oldest_archives_beyond_keep(tmp_path: Path) -> None:
    """When more than ``keep`` archives exist, the OLDEST (by mtime)
    are dropped. The newest ``keep`` survive."""
    base = tmp_path / "demo_launchd.log"
    # Six pre-existing archives with mtimes spaced apart.
    archives = []
    for i in range(6):
        gz = tmp_path / f"demo_launchd.log.2026010{i}000000.gz"
        with gzip.open(gz, "wb") as fh:
            fh.write(f"archive-{i}".encode())
        # Spread mtimes 60 s apart so sorting is unambiguous.
        ts = time.time() - (6 - i) * 60
        os.utime(gz, (ts, ts))
        archives.append(gz)

    # Create a small base file so the rotator has something to process.
    _make_log(base, 1024)
    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=3)

    # Only the 3 newest archives survive.
    surviving = sorted(tmp_path.glob("demo_launchd.log.*.gz"))
    assert len(surviving) == 3
    assert {p.name for p in surviving} == {p.name for p in archives[-3:]}
    assert len(summaries[0]["archives_dropped"]) == 3


def test_retention_does_not_drop_when_count_is_within_keep(
    tmp_path: Path,
) -> None:
    """When archives ≤ keep, nothing is deleted even if the rotator
    runs again. Pairs with the retention-drops test to nail the
    boundary."""
    base = tmp_path / "demo_launchd.log"
    for i in range(2):
        gz = tmp_path / f"demo_launchd.log.2026010{i}000000.gz"
        with gzip.open(gz, "wb") as fh:
            fh.write(f"archive-{i}".encode())

    _make_log(base, 1024)
    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    assert len(list(tmp_path.glob("demo_launchd.log.*.gz"))) == 2
    assert summaries[0]["archives_dropped"] == []


# ── Edge cases ───────────────────────────────────────────────────────────


def test_empty_log_dir_returns_empty_summary(tmp_path: Path) -> None:
    """Common steady-state on a fresh install: log dir is empty.
    The rotator must not raise, just return an empty list."""
    assert r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7) == []


def test_missing_log_dir_returns_empty_summary(tmp_path: Path) -> None:
    """Defensive: a missing log dir (operator hasn't installed launchd
    jobs yet) must not raise."""
    missing = tmp_path / "does-not-exist"
    assert r.rotate_launchd_logs(missing, max_bytes=5_000_000, keep=7) == []


def test_threshold_exactly_at_max_bytes_rotates(tmp_path: Path) -> None:
    """Boundary: a file whose size equals --max-bytes IS rotated.
    The condition is ``>=``, not ``>``."""
    _make_log(tmp_path / "demo_launchd.log", 5_000_000)
    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)
    assert summaries[0]["rotated"] is True


# ── Codex follow-up P1: self-log skip ─────────────────────────────────────


def test_self_log_over_threshold_is_skipped_by_default(tmp_path: Path) -> None:
    """The rotator's own ``log_rotation_launchd.log`` MUST NOT be
    rotated by default — the launchd-driven run holds an fd on it and
    would lose its completion marker. Regression for Codex web-audit
    follow-up P1.
    """
    self_log = _make_log(tmp_path / "log_rotation_launchd.log", 6_000_000)
    original_bytes = self_log.read_bytes()

    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    # The self-log was filtered out before processing; no summary, no
    # archive, original file untouched.
    assert all(
        Path(s["path"]).name != "log_rotation_launchd.log" for s in summaries
    ), f"self-log appeared in summaries: {summaries}"
    assert self_log.exists()
    assert self_log.read_bytes() == original_bytes
    assert list(tmp_path.glob("log_rotation_launchd.log.*.gz")) == []


def test_all_self_log_shapes_are_skipped(tmp_path: Path) -> None:
    """The plist's StandardOut/ErrPath siblings (stdout/stderr in both
    legacy ``.stderr.log`` and current ``_stderr.log`` conventions)
    must also be excluded — launchd holds fds on those too."""
    self_logs = [
        "log_rotation_launchd.log",
        "log_rotation_launchd_stderr.log",
        "log_rotation_launchd_stdout.log",
        "log_rotation_launchd.stderr.log",
        "log_rotation_launchd.stdout.log",
    ]
    for name in self_logs:
        _make_log(tmp_path / name, 6_000_000)

    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    assert summaries == []
    for name in self_logs:
        assert (tmp_path / name).exists()
    assert list(tmp_path.glob("log_rotation_launchd.*.gz")) == []


def test_include_self_flag_overrides_skip(tmp_path: Path) -> None:
    """An operator running the rotator manually (NOT from within the
    launchd job) can pass include_self=True to clean up a self-log
    that somehow grew. Verify the override works."""
    self_log = _make_log(tmp_path / "log_rotation_launchd.log", 6_000_000)

    summaries = r.rotate_launchd_logs(
        tmp_path, max_bytes=5_000_000, keep=7, include_self=True
    )

    assert any(
        Path(s["path"]).name == "log_rotation_launchd.log" and s["rotated"]
        for s in summaries
    )
    assert not self_log.exists()  # was rotated
    archives = list(tmp_path.glob("log_rotation_launchd.log.*.gz"))
    assert len(archives) == 1


def test_self_log_skip_does_not_affect_other_jobs(tmp_path: Path) -> None:
    """Critical contract: skipping self-logs MUST NOT affect rotation
    of the other launchd jobs' logs that happen to live in the same
    directory."""
    _make_log(tmp_path / "log_rotation_launchd.log", 6_000_000)
    _make_log(tmp_path / "daily_evidence_cycle_launchd.log", 6_000_000)

    summaries = r.rotate_launchd_logs(tmp_path, max_bytes=5_000_000, keep=7)

    # The cycle log IS rotated; the self-log is NOT.
    rotated_names = {Path(s["path"]).name for s in summaries if s["rotated"]}
    assert "daily_evidence_cycle_launchd.log" in rotated_names
    assert "log_rotation_launchd.log" not in rotated_names
