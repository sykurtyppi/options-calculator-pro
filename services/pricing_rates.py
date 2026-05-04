"""
Pricing risk-free rate — single source of truth.

Centralizes risk-free rate resolution for all BSM-style pricing in the codebase
(live edge-engine analysis, institutional replay/calibration, snapshot-based
calendar pricing). One in-process cache, one fetch path, one set of bounds.

Resolution order
----------------
1. Explicit env override `OPTIONS_PRICING_RISK_FREE_RATE` (decimal, e.g. 0.0445).
2. Latest available 13-week T-bill yield via yfinance `^IRX` (price quoted in
   percent — divided by 100).
3. Conservative static fallback near the current short-rate regime.

Bounds
------
A resolved rate is accepted only when 0 < rate < 0.25 (decimal). Anything
outside that band is treated as a malformed source and skipped.

Module state
------------
- ``_rf_rate_cache``: dict storing last resolved (rate, source, timestamp).
- ``_rf_rate_lock``: thread lock guarding cache writes.

Both are module-level so a single instance is shared across importers. Callers
in other modules (``web.api.edge_engine`` historically) re-export them under
underscore aliases so existing tests that patch the cache continue to work.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Lock + in-process cache. Single source of truth across importers.
_rf_rate_lock = threading.Lock()
_rf_rate_cache: Dict[str, Any] = {"ts": 0.0, "rate": None, "source": None}

# 12 hours — short-rate moves are slow relative to the trading day; refresh
# twice a day is sufficient for BSM pricing accuracy at the bp level.
DEFAULT_CACHE_TTL_SECONDS: float = 43_200.0

# Static fallback used only if env override is absent and yfinance ^IRX fails.
# Chosen near the current US 13-week T-bill regime.
FALLBACK_STATIC_RATE: float = 0.0450

# Sanity bounds (decimal). Accept rates strictly inside this band.
MIN_VALID_RATE: float = 0.0
MAX_VALID_RATE: float = 0.25


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        parsed = float(value)
        if not np.isfinite(parsed):
            return float(default)
        return parsed
    except (TypeError, ValueError):
        return float(default)


def get_pricing_risk_free_rate(
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
) -> Tuple[float, str]:
    """
    Resolve the pricing risk-free rate for BSM-style calculations.

    Returns ``(rate, source)`` where ``source`` is one of:
      - ``"cache"`` (last resolved value still within TTL)
      - ``"env"`` (env override active)
      - ``"yfinance_^IRX"`` (live 13-week T-bill yield)
      - ``"fallback_static"`` (final fallback)

    The rate is a decimal (e.g. 0.0445 for 4.45%).
    """
    now = time.time()
    with _rf_rate_lock:
        cached_rate = _safe_float(_rf_rate_cache.get("rate"), float("nan"))
        if (
            np.isfinite(cached_rate)
            and cached_rate > MIN_VALID_RATE
            and (now - float(_rf_rate_cache.get("ts") or 0.0)) < cache_ttl_seconds
        ):
            return float(cached_rate), str(_rf_rate_cache.get("source") or "cache")

    env_raw = os.getenv("OPTIONS_PRICING_RISK_FREE_RATE", "").strip()
    env_rate = _safe_float(env_raw, float("nan"))
    if np.isfinite(env_rate) and MIN_VALID_RATE < env_rate < MAX_VALID_RATE:
        with _rf_rate_lock:
            _rf_rate_cache.update({"ts": now, "rate": float(env_rate), "source": "env"})
        return float(env_rate), "env"

    try:
        irx_hist = yf.Ticker("^IRX").history(period="7d", auto_adjust=False)
        irx_close = pd.to_numeric(irx_hist.get("Close"), errors="coerce").dropna()
        if not irx_close.empty:
            irx_rate = float(irx_close.iloc[-1]) / 100.0
            if np.isfinite(irx_rate) and MIN_VALID_RATE < irx_rate < MAX_VALID_RATE:
                with _rf_rate_lock:
                    _rf_rate_cache.update(
                        {"ts": now, "rate": irx_rate, "source": "yfinance_^IRX"}
                    )
                return irx_rate, "yfinance_^IRX"
    except Exception as exc:
        logger.debug("Risk-free rate fetch via ^IRX failed: %s", exc)

    with _rf_rate_lock:
        _rf_rate_cache.update(
            {"ts": now, "rate": FALLBACK_STATIC_RATE, "source": "fallback_static"}
        )
    return FALLBACK_STATIC_RATE, "fallback_static"


def reset_cache() -> None:
    """Drop any cached rate. Primarily for tests."""
    with _rf_rate_lock:
        _rf_rate_cache.update({"ts": 0.0, "rate": None, "source": None})


__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "FALLBACK_STATIC_RATE",
    "MIN_VALID_RATE",
    "MAX_VALID_RATE",
    "_rf_rate_cache",
    "_rf_rate_lock",
    "get_pricing_risk_free_rate",
    "reset_cache",
]
