"""Tests for utils.logger.get_logger handler lifecycle.

PR-O: previously every `get_logger(name)` call ran the full
`setup_logger(name)` path, which closed and re-opened handlers (including
RotatingFileHandlers). Long-running processes leaked file descriptors and
re-emitted the init line. Now `get_logger` is idempotent — subsequent
calls return the existing logger untouched.
"""
from __future__ import annotations

import logging

from utils.logger import get_logger, setup_logger


def _logger_for_test(name: str) -> logging.Logger:
    """Force a clean state for the named logger between tests."""
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    return logger


def test_get_logger_returns_same_instance_across_calls(tmp_path) -> None:
    name = "ops_test_logger.lifecycle.same_instance"
    _logger_for_test(name)

    first = get_logger(name)
    second = get_logger(name)

    assert first is second


def test_get_logger_does_not_grow_handler_count(tmp_path) -> None:
    """The core PR-O regression test: repeated get_logger() calls must not
    accumulate handlers. Before PR-O, this assertion failed within a few
    calls because every invocation re-ran setup_logger() and re-added
    console + file handlers."""
    name = "ops_test_logger.lifecycle.handler_count"
    _logger_for_test(name)

    get_logger(name)  # first call installs handlers
    handler_count_after_first = len(logging.getLogger(name).handlers)
    assert handler_count_after_first > 0

    for _ in range(10):
        get_logger(name)

    assert len(logging.getLogger(name).handlers) == handler_count_after_first


def test_setup_logger_still_resets_handlers(tmp_path) -> None:
    """The explicit setup_logger entry point keeps its reset-and-configure
    semantics — callers explicitly asking for a re-configuration get one.

    This guards against the PR-O guard from accidentally short-circuiting
    intentional re-init via setup_logger().
    """
    name = "ops_test_logger.lifecycle.setup_resets"
    _logger_for_test(name)

    setup_logger(name)
    first_handlers = list(logging.getLogger(name).handlers)
    setup_logger(name)
    second_handlers = list(logging.getLogger(name).handlers)

    # Handler count stays equal (reset + re-create yields the same number)
    # but the actual handler instances are fresh objects (closed + reopened).
    assert len(first_handlers) == len(second_handlers)
    for old in first_handlers:
        assert old not in second_handlers
