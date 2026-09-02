"""Bounded interpolation helpers for the option-chain IV term structure.

Why this module exists
----------------------
Multiple call sites used to read an IV at a target tenor (30D, 45D)
via ``numpy.interp(target_days, days_arr, ivs_arr)``. ``np.interp``
SILENTLY CLAMPS the result to the endpoint when the target lies
outside the bracket — meaning a chain whose nearest expiry is 31D
or 45D would still return a number labeled "iv30," even though no
real 30-day expiry exists. That fabricated value then flowed into:

  * ``ranking_score`` (32% weight via ``_iv_entry_score``)
  * ``cheapness_score``
  * ``term_structure_slope``
  * Walk-forward priors (the backtest study used the same pattern)

This module replaces the bare ``np.interp`` calls with
``bounded_interp``, which returns ``(value, status)``. ``value`` is
``None`` unless the target lies INSIDE the bracket
``[min(days_arr), max(days_arr)]``. Callers are expected to
propagate the ``None`` into their existing ``null_reasons`` /
``signal_fail_reasons`` plumbing so a missing 30-day expiry surfaces
as an explicit gap rather than a silently fabricated value.

The audit finding driving this rewrite is PR #72 (calculations
audit, item M2). The behavior change is: on Monday-post-monthly-
expiry days (and other rare term-structure gaps) some symbols will
lose ``iv30`` / ``iv45`` until a closer expiry materializes —
exactly the right outcome, since there is no honest 30-day vol to
report.

See also: ``docs/PR72_BOUNDED_IV_TENOR_INTERPOLATION_*.md`` if a
design note ever gets written for this PR.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


# Status codes. Exported so callers can compare against named
# constants instead of magic strings, and so the test suite can
# assert on the exact code that fired.
INTERP_OK = "interpolated"
INTERP_TARGET_BELOW_RANGE = "target_below_listed_expiries"
INTERP_TARGET_ABOVE_RANGE = "target_above_listed_expiries"
INTERP_INSUFFICIENT_POINTS = "insufficient_term_structure_points"
INTERP_NO_DATA = "no_term_structure_data"
INTERP_NON_FINITE_INPUT = "non_finite_term_structure_input"

# Map the status codes to the ``null_reasons`` strings that
# downstream consumers (``earnings_vol_snapshot``,
# ``backtest_iv_expansion_study``) already understand. The
# ``INTERP_OK`` case has no null_reason because the result is
# valid. Keeping this map in the helper module (rather than at the
# call sites) means any future addition of a new status code lands
# the diagnostic string in one place.
STATUS_TO_NULL_REASON: dict[str, str] = {
    INTERP_TARGET_BELOW_RANGE: "target_below_listed_expiries",
    INTERP_TARGET_ABOVE_RANGE: "target_above_listed_expiries",
    INTERP_INSUFFICIENT_POINTS: "insufficient_term_structure_points",
    INTERP_NO_DATA: "no_term_structure_data",
    INTERP_NON_FINITE_INPUT: "non_finite_term_structure_input",
}


def bounded_interp(
    target_days: float,
    days_arr: np.ndarray,
    ivs_arr: np.ndarray,
) -> Tuple[Optional[float], str]:
    """Interpolate an IV value at *target_days* from a term structure.

    Returns ``(value, status)`` where:

      * ``value`` is the linearly-interpolated IV at ``target_days``
        IFF the target is bracketed by real expiries
        (``min(days_arr) <= target_days <= max(days_arr)``). Otherwise
        ``None``.

      * ``status`` is one of the ``INTERP_*`` constants describing
        either the success path (``INTERP_OK``) or the precise
        reason for falling back to ``None``.

    Inputs:
      * ``days_arr``: ``np.ndarray[float]`` of expiry DTEs.
      * ``ivs_arr``: ``np.ndarray[float]`` of corresponding ATM IV
        values. Must align by index with ``days_arr``.
      * ``target_days``: scalar float, the tenor we want IV for.

    Guards:
      * Empty array → ``INTERP_NO_DATA``.
      * Fewer than 2 points → ``INTERP_INSUFFICIENT_POINTS`` (we
        refuse to interpolate from a single observation even when
        the target equals it — the "interpolation" wouldn't be one).
      * Length mismatch → ``INTERP_NO_DATA``.
      * Any non-finite value in either array → ``INTERP_NON_FINITE_INPUT``.
      * Target below or above the bracket → respective range code.

    NOT guarded (caller's contract):
      * ``days_arr`` need not be sorted; ``np.interp`` requires it.
        We sort internally so callers don't have to remember.
      * Negative ``target_days`` are accepted as input — they'll
        usually trigger ``INTERP_TARGET_BELOW_RANGE`` since no real
        expiry has a negative DTE, but the helper doesn't reject
        the input itself.
    """
    if days_arr is None or ivs_arr is None:
        return None, INTERP_NO_DATA

    days = np.asarray(days_arr, dtype=float)
    ivs = np.asarray(ivs_arr, dtype=float)

    if days.size == 0 or ivs.size == 0:
        return None, INTERP_NO_DATA
    if days.size != ivs.size:
        return None, INTERP_NO_DATA
    if days.size < 2:
        return None, INTERP_INSUFFICIENT_POINTS

    if not (np.all(np.isfinite(days)) and np.all(np.isfinite(ivs))):
        return None, INTERP_NON_FINITE_INPUT

    # Sort internally — np.interp's contract requires monotonically
    # increasing x. Most call sites pre-sort, but defensive sorting
    # here removes a footgun and the cost is negligible.
    order = np.argsort(days)
    days_sorted = days[order]
    ivs_sorted = ivs[order]

    d_min = float(days_sorted[0])
    d_max = float(days_sorted[-1])

    if target_days < d_min:
        return None, INTERP_TARGET_BELOW_RANGE
    if target_days > d_max:
        return None, INTERP_TARGET_ABOVE_RANGE

    return float(np.interp(target_days, days_sorted, ivs_sorted)), INTERP_OK


def select_tenor_spanning_expiries(
    expirations: Sequence[Any],
    as_of_date: date,
    *,
    max_expiries: int = 6,
    targets: Sequence[float] = (30.0, 45.0),
) -> List[Any]:
    """Pick up to *max_expiries* expiries that BRACKET the interpolation targets.

    "The first N expiries by date" is the wrong window for weekly-heavy
    names. NVDA lists its first 6 expiries inside 16 days, so a 30D/45D
    interpolation has nothing to bracket it: ``bounded_interp`` honestly
    returns ``None`` and ``iv30``/``iv45`` — and with them ``iv_rv``,
    ``cheapness_score`` and the 32%-weight IV-entry rank component — go dark
    on exactly the most liquid, most tradeable symbols.

    This spends the SAME budget differently. The yfinance collectors pay one
    HTTP round-trip per expiry, so the fetch cost is the length of the
    returned list, not how far out it reaches. Every target therefore gets
    the first expiry at or beyond it (bracketing from above), and whatever
    budget remains is filled with the nearest-dated expiries, which supply
    the near/back ratio and the slope origin.

    Expiries at or before ``as_of_date`` are dropped. Unparseable entries are
    skipped rather than raising — a malformed expiry string from a provider
    should cost one tenor, not the whole snapshot. Returns the surviving
    entries in their ORIGINAL form (callers pass them straight back to the
    provider), ordered by expiry date.
    """
    parsed: List[Tuple[int, Any]] = []
    for raw in expirations or []:
        try:
            if isinstance(raw, datetime):
                exp = raw.date()
            elif isinstance(raw, date):
                exp = raw
            else:
                exp = datetime.strptime(str(raw), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        dte = (exp - as_of_date).days
        if dte <= 0:
            continue
        parsed.append((dte, raw))

    if not parsed or max_expiries <= 0:
        return []

    parsed.sort(key=lambda item: item[0])

    keep: set[int] = set()
    # Targets first — they are the whole point of the selection and must not
    # be crowded out by near-dated fill.
    for target in targets:
        idx = next((i for i, (dte, _) in enumerate(parsed) if float(dte) >= float(target)), None)
        if idx is not None and len(keep) < max_expiries:
            keep.add(idx)
    # Then fill with the nearest-dated expiries.
    for i in range(len(parsed)):
        if len(keep) >= max_expiries:
            break
        keep.add(i)

    return [parsed[i][1] for i in sorted(keep)]


__all__ = [
    "bounded_interp",
    "select_tenor_spanning_expiries",
    "INTERP_OK",
    "INTERP_TARGET_BELOW_RANGE",
    "INTERP_TARGET_ABOVE_RANGE",
    "INTERP_INSUFFICIENT_POINTS",
    "INTERP_NO_DATA",
    "INTERP_NON_FINITE_INPUT",
    "STATUS_TO_NULL_REASON",
]
