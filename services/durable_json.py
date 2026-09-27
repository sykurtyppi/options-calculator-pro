"""Durable, cross-process-safe JSON persistence for the learning stores.

The calibration and structure-prior stores used to keep state in memory, append
to it, and write the WHOLE in-memory state back. Two processes (or two store
instances) each holding a stale copy therefore overwrote each other's
observations, and a failed write was logged and swallowed after memory had
already changed, so a trade could be recorded as "learned" while nothing
reached disk.

This module gives them the three pieces that fix both:

* ``exclusive_lock`` - an advisory ``fcntl.flock`` on a sidecar lock file, held
  across the whole reload -> dedupe -> update -> write cycle so concurrent
  writers serialize instead of racing.
* ``read_json_strict`` - re-read the latest on-disk state inside the lock; a
  corrupt file raises instead of being treated as empty (which would overwrite
  it with a partial state).
* ``atomic_write_json`` - temp file + fsync + ``os.replace`` + directory
  fsync; every failure raises ``PersistenceError``.

POSIX-only (macOS/Linux), like ``services.jsonl_helpers``.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class PersistenceError(RuntimeError):
    """A learning-store update could not be made durable."""


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock for *path* (blocking)."""
    lock_path = _lock_path(Path(path))
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")
    except OSError as exc:
        raise PersistenceError(f"cannot open lock file {lock_path}: {exc}") from exc
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        # Closing the descriptor releases the flock.
        handle.close()


def read_json_strict(path: Path) -> Optional[Dict[str, Any]]:
    """The JSON object at *path*, or None if the file does not exist.

    Raises PersistenceError for an unreadable or non-object file: treating it
    as empty would let the next write replace real evidence with less.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PersistenceError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PersistenceError(f"{path} does not contain a JSON object")
    return raw


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Replace *path* with *payload* atomically, or raise PersistenceError."""
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # allow_nan=False: a NaN/inf would make the store unreadable as JSON.
        text = json.dumps(payload, indent=2, allow_nan=False)
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, ValueError, TypeError) as exc:
        raise PersistenceError(f"cannot write {path}: {exc}") from exc
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
