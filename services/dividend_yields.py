"""
Continuous dividend yield — single source of truth for BSM pricing.

Black-Scholes ignores dividends in its plain form, which biases call IVs down
and put IVs up for dividend-paying names and breaks put-call parity in
calendar-spread valuation. This module resolves a per-symbol continuous
dividend yield ``q`` for use in the BSM extension (Merton 1973):

    d1 = (ln(S/K) + (r - q + σ²/2)·T) / (σ·√T)
    Call = S·exp(-q·T)·N(d1) − K·exp(-r·T)·N(d2)

Resolution order
----------------
1. Per-symbol env override ``OPTIONS_DIVIDEND_YIELD_<SYMBOL>`` (decimal,
   e.g. 0.0085 for 0.85%). Useful for tests and known-good corrections.
2. ``yfinance.Ticker(symbol).info['dividendYield']`` if available.
3. Static fallback of 0.0 (the BSM default — same as the legacy behavior).

Bounds
------
A resolved yield is accepted only when 0.0 <= q < 0.20 (decimal). Anything
outside that band is treated as a malformed source and skipped. Non-paying
names correctly resolve to 0.0 with source ``"fallback_zero"`` (not an error).

yfinance quirk
--------------
yfinance returns ``dividendYield`` in two forms across versions: sometimes a
decimal (0.012), sometimes a percent (1.2). This module accepts either by
treating values >= 1.0 as percent and dividing by 100. A genuine 100% yield
would be discarded by the upper bound (0.20) — acceptable trade-off.

Module state
------------
- ``_dividend_cache``: symbol -> {ts, rate, source}.
- ``_dividend_lock``: thread lock guarding cache writes.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_dividend_lock = threading.Lock()
_dividend_cache: Dict[str, Dict[str, Any]] = {}

# 24 hours — dividend policy changes are slow and infrequent compared to
# the option-pricing horizons we care about. Refreshing daily is plenty.
DEFAULT_CACHE_TTL_SECONDS: float = 86_400.0

# Bounds (decimal). Reject obviously malformed values.
MIN_VALID_YIELD: float = 0.0
MAX_VALID_YIELD: float = 0.20

# Source identifiers used as the second element of the return tuple.
SOURCE_CACHE = "cache"
SOURCE_ENV = "env"
SOURCE_YFINANCE = "yfinance"
SOURCE_FALLBACK_ZERO = "fallback_zero"


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        parsed = float(value)
        if not np.isfinite(parsed):
            return float("nan")
        return parsed
    except (TypeError, ValueError):
        return float("nan")


def _normalize_yfinance_yield(raw: Any) -> float:
    """Coerce a yfinance dividendYield value into a decimal in [0, 0.20].

    yfinance reports inconsistently across versions: sometimes 0.0123,
    sometimes 1.23 (for 1.23%). Treat >= 1.0 as percent and divide.
    """
    parsed = _safe_float(raw)
    if not np.isfinite(parsed) or parsed < 0:
        return float("nan")
    if parsed >= 1.0:
        parsed = parsed / 100.0
    return parsed


def _resolve_symbol_env_override(symbol: str) -> float:
    """Look up `OPTIONS_DIVIDEND_YIELD_<SYMBOL>` if present.

    Symbol is uppercased and stripped of non-identifier characters so that
    e.g. ``BRK.B`` maps to ``OPTIONS_DIVIDEND_YIELD_BRK_B``.
    """
    safe_symbol = "".join(c if c.isalnum() else "_" for c in symbol.upper())
    env_key = f"OPTIONS_DIVIDEND_YIELD_{safe_symbol}"
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return float("nan")
    return _safe_float(raw)


def get_dividend_yield(
    symbol: str,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
) -> Tuple[float, str]:
    """Resolve a per-symbol continuous dividend yield ``q`` for BSM pricing.

    Returns ``(q, source)`` where ``q`` is a decimal in [0, 0.20] and
    ``source`` is one of ``"cache"``, ``"env"``, ``"yfinance"``, or
    ``"fallback_zero"``. Non-dividend-paying names resolve to ``(0.0,
    "fallback_zero")`` — this is the correct BSM-default behavior, not an
    error.

    Lookups are cached in-process for ``cache_ttl_seconds`` (default 24h).
    yfinance is queried lazily to keep this module importable without it.
    """
    sym = symbol.strip().upper()
    if not sym:
        return 0.0, SOURCE_FALLBACK_ZERO

    now = time.time()
    with _dividend_lock:
        cached = _dividend_cache.get(sym)
        if cached is not None:
            ts = float(cached.get("ts") or 0.0)
            if (now - ts) < cache_ttl_seconds:
                rate = _safe_float(cached.get("rate"))
                if np.isfinite(rate) and MIN_VALID_YIELD <= rate < MAX_VALID_YIELD:
                    return float(rate), SOURCE_CACHE

    env_rate = _resolve_symbol_env_override(sym)
    if np.isfinite(env_rate) and MIN_VALID_YIELD <= env_rate < MAX_VALID_YIELD:
        with _dividend_lock:
            _dividend_cache[sym] = {"ts": now, "rate": float(env_rate), "source": SOURCE_ENV}
        return float(env_rate), SOURCE_ENV

    try:
        import yfinance as yf  # lazy import — keep module usable without yfinance
        info = yf.Ticker(sym).info or {}
        raw_yield = info.get("dividendYield")
        normalized = _normalize_yfinance_yield(raw_yield)
        if np.isfinite(normalized) and MIN_VALID_YIELD <= normalized < MAX_VALID_YIELD:
            with _dividend_lock:
                _dividend_cache[sym] = {
                    "ts": now,
                    "rate": float(normalized),
                    "source": SOURCE_YFINANCE,
                }
            if normalized > 0.005:
                logger.debug(
                    "Dividend yield for %s = %.4f (yfinance) — BSM pricing will use q",
                    sym,
                    normalized,
                )
            return float(normalized), SOURCE_YFINANCE
    except Exception as exc:
        logger.debug("Dividend yield fetch for %s via yfinance failed: %s", sym, exc)

    # Fallback: assume non-paying (q = 0). This matches legacy BSM behavior
    # exactly, so the caller is no worse off than before.
    with _dividend_lock:
        _dividend_cache[sym] = {"ts": now, "rate": 0.0, "source": SOURCE_FALLBACK_ZERO}
    return 0.0, SOURCE_FALLBACK_ZERO


def reset_cache() -> None:
    """Drop all cached yields. Primarily for tests."""
    with _dividend_lock:
        _dividend_cache.clear()


__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "MAX_VALID_YIELD",
    "MIN_VALID_YIELD",
    "SOURCE_CACHE",
    "SOURCE_ENV",
    "SOURCE_FALLBACK_ZERO",
    "SOURCE_YFINANCE",
    "_dividend_cache",
    "_dividend_lock",
    "get_dividend_yield",
    "reset_cache",
]
