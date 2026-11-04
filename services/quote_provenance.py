"""When and from whom an analysis's option quotes came, and whether to trust them.

The recommendation ledger's quote_source / quote_timestamp columns were empty
for every forward analysis, so a stop caused by stale quotes could not be told
from one caused by the market. This records, for the chain an analysis read:

* ``quote_source``: the provider that supplied it;
* ``fetched_at``: when it was fetched (UTC);
* ``quote_timestamp``: the provider's own quote time when it has one
  (MarketData's per-contract ``updated``; the median across the chain), else
  the fetch time, with ``timestamp_basis`` saying which;
* ``in_regular_session`` / ``quote_session``: whether NYSE was open at the
  fetch, and which session's quotes were current.

yfinance gives no quote time at all, so for it only the fetch time and the
session are known; that is what is recorded, never an invented quote time.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from services.market_calendar import in_regular_session, quote_session

PROVENANCE_VERSION = 1
BASIS_PROVIDER = "provider_quote_time"
BASIS_FETCH = "fetch_time_only"

# Older provider quotes than this, at the fetch, are stale. Generous enough for
# a 15-minute delayed feed.
MAX_QUOTE_AGE_MINUTES = 30.0

PROBLEM_MISSING = "missing_quote_provenance"
PROBLEM_OUTSIDE_SESSION = "outside_session"
PROBLEM_STALE = "stale_quotes"


def _provider_quote_times(frame: Any) -> list[datetime]:
    """Per-contract quote times a provider stamped (MarketData ``updated``, unix s)."""
    columns = getattr(frame, "columns", None)
    if frame is None or columns is None or "updated" not in columns:
        return []
    times = []
    for value in frame["updated"].tolist():
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(seconds) and seconds > 0:
            times.append(datetime.fromtimestamp(seconds, tz=timezone.utc))
    return times


def chain_quote_provenance(frame: Any, *, provider: Optional[str], fetched_at: datetime) -> dict[str, Any]:
    """Provenance of the option chain an analysis read, fetched at ``fetched_at`` (aware)."""
    if fetched_at.tzinfo is None:
        raise ValueError("chain_quote_provenance needs a timezone-aware fetched_at")
    fetched_at = fetched_at.astimezone(timezone.utc)
    times = sorted(_provider_quote_times(frame))
    if times:
        quote_time = statistics.median_low(times)
        basis = BASIS_PROVIDER
        age = max(0.0, (fetched_at - quote_time).total_seconds() / 60.0)
        oldest = times[0]
    else:
        quote_time, basis, age, oldest = fetched_at, BASIS_FETCH, None, None
    return {
        "version": PROVENANCE_VERSION,
        "quote_source": provider,
        "fetched_at": fetched_at.isoformat(),
        "quote_timestamp": quote_time.isoformat(),
        "timestamp_basis": basis,
        "quote_age_minutes": None if age is None else round(age, 1),
        "oldest_quote_timestamp": oldest.isoformat() if oldest else None,
        "contracts_with_quote_time": len(times),
        "in_regular_session": in_regular_session(fetched_at),
        "quote_session": quote_session(fetched_at).isoformat(),
    }


def ledger_quote_payload(provenance: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The recommendation ledger's quote columns, filled from a provenance record."""
    if not provenance:
        return {}
    return {
        "quote_source": provenance.get("quote_source"),
        "quote_timestamp": provenance.get("quote_timestamp"),
        "quote_quality": provenance.get("timestamp_basis"),
    }


def quote_provenance_problem(
    provenance: Optional[Mapping[str, Any]],
    *,
    max_age_minutes: float = MAX_QUOTE_AGE_MINUTES,
) -> Optional[str]:
    """Why these quotes must not price forward evidence, or None.

    Unknown provenance, a fetch outside the regular session, or provider quote
    times older than ``max_age_minutes`` at the fetch all disqualify.
    """
    if not provenance:
        return PROBLEM_MISSING
    if provenance.get("in_regular_session") is not True:
        return PROBLEM_OUTSIDE_SESSION
    age = provenance.get("quote_age_minutes")
    if isinstance(age, (int, float)) and math.isfinite(age) and age > max_age_minutes:
        return f"{PROBLEM_STALE}:{age:.0f}m"
    return None


__all__ = [
    "BASIS_FETCH",
    "BASIS_PROVIDER",
    "MAX_QUOTE_AGE_MINUTES",
    "PROBLEM_MISSING",
    "PROBLEM_OUTSIDE_SESSION",
    "PROBLEM_STALE",
    "PROVENANCE_VERSION",
    "chain_quote_provenance",
    "ledger_quote_payload",
    "quote_provenance_problem",
]
