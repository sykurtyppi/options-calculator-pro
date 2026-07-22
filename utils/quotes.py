"""Canonical option-quote helpers.

Single source of truth for the invariant: **a midpoint may only be derived from
an executable two-sided market.** A zero (or absent) bid means nobody is bidding
— it is not a tradeable quote, and `(0 + ask) / 2` is a fabricated price that
must never reach implied-move / straddle / execution math. A crossed quote
(ask < bid) is likewise invalid.

This module exists so that rule is defined in exactly ONE place. Prior audits
repeatedly found the guard applied at some call sites and missing at others
(`bid > 0` present in one mid computation, absent in a sibling); routing every
`(bid + ask) / 2` through these helpers makes that class of bug structurally
impossible. Use `safe_mid` for scalars and `safe_mid_series` for pandas frames.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def is_executable_quote(bid: Any, ask: Any) -> bool:
    """True iff (bid, ask) is a valid two-sided market.

    Requires both sides finite, ``bid > 0``, ``ask > 0`` and ``ask >= bid``.
    A zero/absent bid or a crossed quote is NOT executable.
    """
    try:
        b = float(bid)
        a = float(ask)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(b) and np.isfinite(a) and b > 0.0 and a > 0.0 and a >= b)


def safe_mid(bid: Any, ask: Any) -> Optional[float]:
    """Midpoint of a valid two-sided quote, else ``None``.

    Never fabricates a mid from a zero/absent bid (not an executable market) or
    a crossed quote — those return ``None`` so callers omit the strike rather
    than pricing off a phantom quote.
    """
    if not is_executable_quote(bid, ask):
        return None
    return (float(bid) + float(ask)) / 2.0


def safe_mid_series(bid: Any, ask: Any) -> pd.Series:
    """Vectorized :func:`safe_mid` over pandas Series / array-likes.

    Returns a float Series that is ``NaN`` wherever the quote is not an
    executable two-sided market (non-finite, ``bid <= 0``, ``ask <= 0`` or
    crossed). Non-numeric inputs are coerced to ``NaN`` first.
    """
    # Wrap in pd.Series so array-likes (list / ndarray) work too, not just
    # Series — pd.to_numeric returns an ndarray for those, which lacks `.where`.
    b = pd.Series(pd.to_numeric(bid, errors="coerce"))
    a = pd.Series(pd.to_numeric(ask, errors="coerce"))
    valid = np.isfinite(b) & np.isfinite(a) & (b > 0.0) & (a > 0.0) & (a >= b)
    return ((b + a) / 2.0).where(valid)
