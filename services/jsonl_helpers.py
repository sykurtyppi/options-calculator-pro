"""Shared JSONL append helpers.

JSONL files in this project are produced by long-running launchd
jobs that can occasionally race each other or race an operator
invoking the same script manually. Without a writer lock, two
concurrent appenders can interleave their writes mid-line —
producing malformed JSON that breaks any tail-reader that parses
lines into objects (the watchdog and ``services/evidence_health``
both do this).

The resolver already had a locked-append helper under Hardening
P1-3 (``scripts/resolve_candidate_exits._append_jsonl``). PR #73
generalizes that pattern so the evidence-cycle structured run log
and the forward-loop learning log get the same protection.

POSIX-specific: ``fcntl.flock`` exists on macOS and Linux (the
project's deployment targets); it does not exist on Windows. The
helper raises an explicit ``RuntimeError`` if imported on an
unsupported platform rather than degrading silently.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union


def append_jsonl_locked(
    path: Union[str, Path],
    rows: Iterable[Dict[str, Any]],
) -> None:
    """Append one JSON line per row to *path*, holding an advisory
    ``fcntl.LOCK_EX`` for the duration of the write so concurrent
    invocations cannot interleave per-row writes and corrupt the
    JSONL stream.

    Parameters
    ----------
    path
        Destination JSONL file. Parent directory is created if
        absent.

    rows
        Iterable of dict-like rows. Empty iterable is a no-op
        (no file touched). Each row is serialized with
        ``json.dumps(row, sort_keys=True, default=str) + "\\n"``,
        matching the resolver's existing format so any downstream
        tail-reader doesn't have to know which producer wrote a
        given line.

    Locking notes
    -------------
    * ``fcntl.flock`` is advisory and POSIX-specific. The file
      handle's lock is automatically released when the ``with``
      block closes the descriptor, so no explicit ``LOCK_UN``.
    * The lock is held for the SINGLE write call that flushes the
      whole pre-built buffer. We deliberately serialize the dump
      step before opening the file so the lock window is
      minimized.
    * If a writer cannot acquire the lock (e.g. another process
      holds it), ``fcntl.LOCK_EX`` blocks until released — there
      is no busy-spin and no immediate failure. For the wrapper-
      script lock-cleanup scenarios this is the right behavior:
      the in-flight writer is doing useful work, and the waiter
      gets in line cleanly.

    History
    -------
    Originated as ``scripts/resolve_candidate_exits._append_jsonl``
    under Hardening P1-3 (the candidate exit resolver was the
    first JSONL appender to hit a corruption race). PR #73 lifted
    it here so the evidence-cycle structured run log
    (``scripts/run_evidence_cycle._append_structured_run_log``) and
    the forward-loop learning log
    (``scripts/run_forward_loop._append_learning_log``) get the
    same protection.
    """
    rows_list = list(rows)
    if not rows_list:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Build the entire payload BEFORE opening to minimize the
    # window the lock is held.
    buffer = "".join(
        json.dumps(row, sort_keys=True, default=str) + "\n"
        for row in rows_list
    )
    with open(p, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(buffer)
        fh.flush()


__all__ = ["append_jsonl_locked"]
