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
2. yfinance ``info``, from fields whose units are unambiguous:
   ``dividendRate`` (annual cash dividend) / the current price, else
   ``trailingAnnualDividendYield`` (a decimal).
3. Static fallback of 0.0 (the BSM default — same as the legacy behavior).

Bounds
------
A resolved yield is accepted only when 0.0 <= q < 0.20 (decimal). Anything
outside that band is treated as a malformed source and skipped. Non-paying
names correctly resolve to 0.0 with source ``"fallback_zero"`` (not an error).

yfinance quirk
--------------
``info['dividendYield']`` has been reported both as a decimal (0.012) and as
a percent (1.2) across Yahoo/yfinance versions, and no threshold can tell
them apart: 0.15 is either 15% or 0.15%. It is therefore never used. A
payer without an unambiguous field falls back to 0.0, which the UI flags.

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
from datetime import date, timedelta
from typing import Any, Dict, Optional, Tuple

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


def _yield_from_info(info: Dict[str, Any]) -> float:
    """Decimal dividend yield from yfinance ``info``, using unambiguous fields.

    ``dividendRate`` is dollars per share per year, so dividing it by the
    current price cannot be off by a unit. ``trailingAnnualDividendYield``
    is always a decimal. ``dividendYield`` is ignored (see module docs).
    """
    rate = _safe_float(info.get("dividendRate"))
    for key in ("regularMarketPrice", "currentPrice", "previousClose"):
        price = _safe_float(info.get(key))
        if np.isfinite(rate) and rate >= 0 and np.isfinite(price) and price > 0:
            return rate / price
    trailing = _safe_float(info.get("trailingAnnualDividendYield"))
    if np.isfinite(trailing) and trailing >= 0:
        return trailing
    return float("nan")


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
                    # Report where the cached value came from (like the rate
                    # service): a cached fallback must still read as a fallback.
                    return float(rate), str(cached.get("source") or SOURCE_CACHE)

    env_rate = _resolve_symbol_env_override(sym)
    if np.isfinite(env_rate) and MIN_VALID_YIELD <= env_rate < MAX_VALID_YIELD:
        with _dividend_lock:
            _dividend_cache[sym] = {"ts": now, "rate": float(env_rate), "source": SOURCE_ENV}
        return float(env_rate), SOURCE_ENV

    try:
        import yfinance as yf  # lazy import — keep module usable without yfinance
        info = yf.Ticker(sym).info or {}
        normalized = _yield_from_info(info)
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
        _dividend_history_cache.clear()


SOURCE_TRAILING_HISTORICAL = "yfinance_trailing_12m_dividends_historical"
_dividend_history_cache: Dict[str, Any] = {}


def get_historical_dividend_yield(
    symbol: str,
    as_of: date,
    underlying_price: float,
) -> Tuple[Optional[float], str]:
    """Continuous dividend yield as knowable ON ``as_of`` (point in time).

    Annualised cash dividends with ex-dates on or before ``as_of``, divided by
    the as-traded underlying price then:

    * Only dividends already paid by ``as_of`` count, so a past valuation
      cannot see a later dividend regime.
    * yfinance reports dividends split-adjusted to today's share basis; they
      are restated in the share basis of ``as_of`` (the basis of the price
      passed in) by undoing only the splits AFTER ``as_of``. A later split
      cannot shrink (or a reverse split inflate) the yield, and a split
      between an ex-date and ``as_of`` is already in both the dividend and
      the price.
    * The payout frequency comes from the gaps between recent ex-dates and
      the latest year's worth of payments is summed, so ex-date drift cannot
      pull a fifth quarterly dividend into a 365-day window.
    * One-off special dividends are not a recurring yield and are excluded:
      a payment more than twice the median of the OTHER payments, with no
      other payment of a similar size (a repeated higher amount is a raise,
      not a special).
    * ``underlying_price`` must be the as-traded price on ``as_of``, not a
      split- or dividend-adjusted close.
    * No dividend yet, or a dividend overdue by more than a cycle
      (suspended), is 0.0: a known fact at ``as_of``.

    Returns ``(None, "unavailable")`` if the history cannot be fetched;
    callers must then refuse to price. Still a continuous-yield
    approximation: discrete dividends and upcoming ex-dates are not modelled.
    """
    sym = symbol.strip().upper()
    if not sym or not np.isfinite(underlying_price) or underlying_price <= 0:
        return None, "unavailable"
    now = time.time()
    with _dividend_lock:
        cached = _dividend_history_cache.get(sym)
    history = None
    if cached is not None and (now - cached[0]) < DEFAULT_CACHE_TTL_SECONDS:
        history = cached[1]
    if history is None:
        try:
            import yfinance as yf  # lazy import — keep module usable without yfinance
            ticker = yf.Ticker(sym)
            dividends = ticker.dividends
            splits = ticker.splits
        except Exception as exc:
            logger.debug("Dividend/split history fetch for %s failed: %s", sym, exc)
            return None, "unavailable"
        if dividends is None or splits is None:
            return None, "unavailable"
        history = (_dated_values(dividends), _dated_values(splits))
        with _dividend_lock:
            # Refreshed daily: a long-running process capturing "today" must
            # see new ex-dates and splits.
            _dividend_history_cache[sym] = (now, history)
    dividends, splits = history

    # Today's basis -> as_of's basis: undo the splits after as_of only.
    basis_factor = 1.0
    for split_day, ratio in splits:
        if split_day > as_of and ratio > 0:
            basis_factor *= ratio

    paid = sorted(
        (day, amount * basis_factor) for day, amount in dividends
        if day <= as_of and amount > 0 and (as_of - day).days <= 800
    )
    paid = _without_specials(paid)
    q = 0.0
    if paid:
        per_year = _payments_per_year([day for day, _ in paid])
        last_day = paid[-1][0]
        if (as_of - last_day).days <= 365.0 / per_year + 45:
            q = sum(amount for _, amount in paid[-per_year:]) / float(underlying_price)
    if not (MIN_VALID_YIELD <= q < MAX_VALID_YIELD):
        return None, "unavailable"
    return float(q), SOURCE_TRAILING_HISTORICAL


def _without_specials(paid: list) -> list:
    """``paid`` without isolated outsized (special) payments.

    Leave-one-out, so a special cannot set its own yardstick: with one
    regular payment and one special, the median of all payments sits
    between them and would keep the special.
    """
    kept = []
    for i, (day, amount) in enumerate(paid):
        others = [other for j, (_, other) in enumerate(paid) if j != i]
        if others:
            typical = float(np.median(others))
            repeated = any(abs(other - amount) <= 0.25 * amount for other in others)
            if amount > 2.0 * typical and not repeated:
                continue
        kept.append((day, amount))
    return kept


def _dated_values(series: Any) -> list:
    """(date, value) pairs from a yfinance date-indexed Series."""
    pairs = []
    for when, value in getattr(series, "items", lambda: [])():
        day = when.date() if hasattr(when, "date") else when
        parsed = _safe_float(value)
        if isinstance(day, date) and np.isfinite(parsed):
            pairs.append((day, float(parsed)))
    return pairs


def _payments_per_year(ex_dates: list) -> int:
    """Payout frequency (1, 2, 4 or 12) from the median gap between ex-dates."""
    recent = ex_dates[-9:]
    gaps = [(b - a).days for a, b in zip(recent, recent[1:]) if (b - a).days > 0]
    if not gaps:
        return 1
    per_year = 365.0 / float(np.median(gaps))
    return min((1, 2, 4, 12), key=lambda candidate: abs(candidate - per_year))


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
    "SOURCE_TRAILING_HISTORICAL",
    "get_dividend_yield",
    "get_historical_dividend_yield",
    "reset_cache",
]
