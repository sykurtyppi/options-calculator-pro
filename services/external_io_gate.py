"""External-IO gate. Stdlib only. Keep this module boring.

A module-level allowlist of `Category` members controls whether external
network/IPC operations are permitted. Default: every category allowed
(production behaviour). Tests disable everything in a session-autouse conftest
fixture and opt back in per-category via `enable(...)` as needed.

DO NOT add internal project imports here. DO NOT add logging or telemetry.
Every external-client call site depends on this module, so it must stay
trivially auditable. `install_yfinance_guard` imports yfinance lazily inside
the function body to preserve a stdlib-only top level.
"""
from __future__ import annotations

import enum
from typing import Optional


class Category(enum.Enum):
    MARKETDATA = "marketdata"
    EARNINGS_FMP = "earnings_fmp"
    EARNINGS_ALPHA_VANTAGE = "earnings_alpha_vantage"
    EARNINGS_SEC_EDGAR = "earnings_sec_edgar"
    YFINANCE = "yfinance"
    IMESSAGE = "imessage"


class ExternalIOBlocked(RuntimeError):
    pass


_ALLOWED: set[Category] = set(Category)


def assert_allowed(category: Category) -> None:
    if category not in _ALLOWED:
        raise ExternalIOBlocked(
            f"{category.value} external IO blocked during test session"
        )


def is_allowed(category: Category) -> bool:
    return category in _ALLOWED


def disable_all() -> None:
    _ALLOWED.clear()


def enable(category: Optional[Category] = None) -> None:
    if category is None:
        _ALLOWED.update(Category)
    else:
        _ALLOWED.add(category)


def disable(category: Category) -> None:
    _ALLOWED.discard(category)


def install_yfinance_guard() -> None:
    """Replace `yfinance.Ticker` with a version that consults the gate.

    Idempotent. Called once from the session-autouse conftest fixture so the
    16 inline `yf.Ticker(...)` call sites are protected without per-site edits.
    """
    import yfinance as yf

    if getattr(yf.Ticker, "__external_io_guarded__", False):
        return
    original = yf.Ticker

    def _guarded_ticker(*args, **kwargs):
        assert_allowed(Category.YFINANCE)
        return original(*args, **kwargs)

    _guarded_ticker.__external_io_guarded__ = True
    _guarded_ticker.__wrapped__ = original
    yf.Ticker = _guarded_ticker
