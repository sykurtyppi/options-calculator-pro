"""Tests for the OOS cooperative-cancellation path in web.api.app.

PR-N: ``_run_oos_with_timeout`` previously started a daemon thread that
kept running with all its pandas/duckdb memory after the timeout fired,
stacking memory on repeated timeouts. The fix passes a ``cancel_event``
that ``run_oos_validation`` polls between rolling-OOS splits.

These tests verify:
1. The timeout path actually signals cancellation (sets the event).
2. The kwargs threading is correct (the event reaches the worker).
3. The kwargs threading is correct even when the worker completes
   without the event being set (no leftover state).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import web.api.app as app_module


class _Capture:
    """Tiny stand-in for InstitutionalDataCollector that records the cancel_event."""

    def __init__(self, *, behavior: str = "complete_immediately"):
        self.behavior = behavior
        self.received_kwargs: Dict[str, Any] = {}
        self.was_cancelled = False

    def run_oos_validation(self, **kwargs):
        self.received_kwargs = kwargs
        event = kwargs.get("cancel_event")
        if self.behavior == "complete_immediately":
            return {"report_card": {"ready": True}, "splits": 1}
        if self.behavior == "loop_until_cancelled":
            # Simulate a long-running OOS loop that polls the event.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if event is not None and event.is_set():
                    self.was_cancelled = True
                    return None
                time.sleep(0.05)
            return None  # would have completed normally
        raise RuntimeError(f"Unknown behavior: {self.behavior}")


def test_run_oos_with_timeout_passes_cancel_event_to_worker() -> None:
    capture = _Capture(behavior="complete_immediately")
    result = app_module._run_oos_with_timeout(capture, kwargs={}, timeout_seconds=2.0)
    assert result is not None
    event = capture.received_kwargs.get("cancel_event")
    assert isinstance(event, threading.Event)
    assert not event.is_set()  # not set on the happy path


def test_run_oos_with_timeout_sets_cancel_event_on_timeout() -> None:
    """The whole point of PR-N: if the worker is still running when the
    wall-clock timeout fires, we must signal cancellation so the worker
    can clean up instead of being orphaned."""
    capture = _Capture(behavior="loop_until_cancelled")
    result = app_module._run_oos_with_timeout(
        capture, kwargs={}, timeout_seconds=0.2
    )
    assert result is None  # the timeout path returns None

    # Give the worker a moment to observe the cancel.
    deadline = time.time() + 2.0
    while time.time() < deadline and not capture.was_cancelled:
        time.sleep(0.05)
    assert capture.was_cancelled, (
        "Worker should have observed the cancel_event and exited cleanly"
    )


def test_run_oos_with_timeout_propagates_kwargs_to_worker() -> None:
    """Other kwargs must reach the worker unchanged; the cancel_event
    addition must not clobber existing arguments."""
    capture = _Capture(behavior="complete_immediately")
    app_module._run_oos_with_timeout(
        capture,
        kwargs={"lookback_days": 1095, "max_backtest_symbols": 50},
        timeout_seconds=2.0,
    )
    assert capture.received_kwargs["lookback_days"] == 1095
    assert capture.received_kwargs["max_backtest_symbols"] == 50
    assert "cancel_event" in capture.received_kwargs


def test_run_oos_validation_signature_accepts_cancel_event() -> None:
    """Static signature check: the chain from web.api → script wrapper →
    services rolling-OOS all accept the kwarg without TypeError."""
    from inspect import signature
    from scripts.institutional_backfill import InstitutionalDataCollector
    from services.institutional_ml_db import InstitutionalMLDatabase

    wrapper_sig = signature(InstitutionalDataCollector.run_oos_validation)
    assert "cancel_event" in wrapper_sig.parameters

    rolling_sig = signature(InstitutionalMLDatabase.run_rolling_oos_validation)
    assert "cancel_event" in rolling_sig.parameters
