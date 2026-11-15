"""When an earnings release lands relative to the NYSE session: one rule for everyone.

Before this module, about eleven functions in eight files decided "before the
open" vs "after the close", each with its own cutoffs and time zone: some read
the UTC hour (>= 20 AMC, < 16 BMO, so a 15:30 ET release was AMC in winter
and a 10:00 ET one was BMO), one stripped a New York timestamp to naive UTC
and then applied 15:00 / 10:00 to it (a 07:00 ET release became unknown), one
used noon ET as the BMO cutoff, and several read a date-only midnight stamp as
"before the open". The same event could get a different label, and with it a
different reaction session, expiry and exit anchor, depending on which module
ran.

The rule here:

* Codes are ``BMO`` (before 09:30 ET), ``DMH`` (during the session), ``AMC``
  (at or after the session close, 16:00 ET, or 13:00 ET on an early-close
  day) and ``UNKNOWN``. ``to_label`` gives the long form some callers store.
* An instant is judged on the New York wall clock (``zoneinfo``, so daylight
  saving is handled). Aware values are converted; naive values are read as
  New York wall-clock time; unix seconds are UTC.
* A value with no time of day (a ``date``, or a timestamp at exactly midnight
  in New York or in UTC, which providers use for "date only") is ``UNKNOWN``,
  never "before the open".
* ``reaction_session`` is the first session on which the market can react:
  the event day's session (or the next one, if it is a weekend or holiday) for
  BMO and DMH; the next session after the event day for AMC and UNKNOWN
  (unknown is treated as AMC: never claim coverage that may not exist).
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional

from services.market_calendar import EXCHANGE_TZ, SESSION_OPEN, next_session_on_or_after, session_close

BMO = "BMO"
DMH = "DMH"
AMC = "AMC"
UNKNOWN = "UNKNOWN"
CODES = (BMO, DMH, AMC, UNKNOWN)

LABELS = {
    BMO: "before market open",
    DMH: "during market hours",
    AMC: "after market close",
    UNKNOWN: "unknown",
}

_EXACT = {
    "bmo": BMO, "before market open": BMO, "before the open": BMO, "pre-market": BMO, "pre market": BMO,
    "premarket": BMO, "pre": BMO, "am": BMO, "before open": BMO,
    "amc": AMC, "after market close": AMC, "after the close": AMC, "after-hours": AMC, "after hours": AMC,
    "afterhours": AMC, "post-market": AMC, "post market": AMC, "post": AMC, "pm": AMC, "after close": AMC,
    "dmh": DMH, "during market hours": DMH, "during market": DMH, "intraday": DMH, "during": DMH,
    "unknown": UNKNOWN, "tbd": UNKNOWN, "tba": UNKNOWN, "time not supplied": UNKNOWN, "--": UNKNOWN,
    "n/a": UNKNOWN, "none": UNKNOWN, "": UNKNOWN,
}
# "4:05 PM", "16:05", "7:00am ET", "09:30:00"
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\s*(et|est|edt)?$")


def _wall_clock_code(moment: time, day: Optional[date] = None) -> str:
    close = session_close(day) if day is not None else time(16, 0)
    if moment < SESSION_OPEN:
        return BMO
    if moment >= close:
        return AMC
    return DMH


def _is_midnight(moment: datetime) -> bool:
    return (moment.hour, moment.minute, moment.second, moment.microsecond) == (0, 0, 0, 0)


def _to_market_time(value: Any) -> Optional[datetime]:
    """An aware New York datetime, or None when the value carries no time of day."""
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        try:
            if value != value:  # NaT
                return None
        except Exception:  # noqa: BLE001
            pass
        value = value.to_pydatetime()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or value <= 0:
            return None
        try:
            value = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        if _is_midnight(value):
            return None
        return value.replace(tzinfo=EXCHANGE_TZ)
    if _is_midnight(value) or _is_midnight(value.astimezone(timezone.utc)):
        return None
    return value.astimezone(EXCHANGE_TZ)


def classify_instant(value: Any) -> str:
    """BMO / DMH / AMC for a release instant; UNKNOWN when it has no time of day."""
    moment = _to_market_time(value)
    if moment is None:
        return UNKNOWN
    return _wall_clock_code(moment.timetz().replace(tzinfo=None), moment.date())


def market_date(value: Any) -> Optional[date]:
    """The New York calendar date of a release stamp.

    A stamp with a time of day is converted to New York first, so a 20:30 ET
    release (01:30 UTC the next day) keeps its own date. A date-only stamp
    (midnight in its own zone) keeps the date it was given.
    """
    if hasattr(value, "to_pydatetime"):
        try:
            if value != value:  # NaT
                return None
        except Exception:  # noqa: BLE001
            pass
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None or _is_midnight(value):
            return value.date()
        return value.astimezone(EXCHANGE_TZ).date()
    if isinstance(value, date):
        return value
    return None


def _clock_code(text: str) -> Optional[str]:
    match = _CLOCK.match(text)
    if not match:
        return None
    hour, minute, second = int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)
    meridiem = match.group(4)
    if meridiem:
        if not 1 <= hour <= 12:
            return UNKNOWN
        hour = (hour % 12) + (12 if meridiem == "pm" else 0)
    if hour > 23 or minute > 59 or second > 59:
        return UNKNOWN
    return _wall_clock_code(time(hour, minute, second))


def to_code(value: Any) -> str:
    """The timing code for anything a provider, a store or a caller might hold."""
    if value is None or isinstance(value, bool):
        return UNKNOWN
    if isinstance(value, (datetime, int, float)) or hasattr(value, "to_pydatetime"):
        return classify_instant(value)
    text = " ".join(str(value).strip().lower().replace("_", " ").split())
    if text.upper() in CODES:
        return text.upper()
    if text in _EXACT:
        return _EXACT[text]
    clock = _clock_code(text)
    if clock is not None:
        return clock
    if text.startswith("intraday") or "during" in text:
        return DMH
    if "before" in text or "pre-market" in text or "premarket" in text:
        return BMO
    if "after" in text or "post-market" in text or "after-hours" in text:
        return AMC
    return UNKNOWN


def to_label(value: Any) -> str:
    """The long label ("before market open", ...) for any timing value."""
    return LABELS[to_code(value)]


def classify_sec_acceptance(value: Any) -> str:
    """Timing from an EDGAR ``acceptanceDateTime`` such as ``2025-10-30T16:30:05.000Z``.

    Whether that trailing ``Z`` means UTC or Eastern wall-clock time is not
    verified here, so the value is read both ways and used only when both
    readings agree (an early-morning or late-evening filing); otherwise it is
    UNKNOWN. A filing's acceptance time is also not the press release's time.
    """
    if value is None or isinstance(value, bool):
        return UNKNOWN
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return UNKNOWN
    as_utc = classify_instant(parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc))
    as_eastern = classify_instant(parsed.replace(tzinfo=None))
    return as_utc if as_utc == as_eastern else UNKNOWN


def reaction_session(earnings_date: date, timing: Any) -> date:
    """First NYSE session on which the earnings reaction can print."""
    if to_code(timing) in (BMO, DMH):
        return next_session_on_or_after(earnings_date)
    return next_session_on_or_after(earnings_date + timedelta(days=1))


__all__ = [
    "AMC", "BMO", "CODES", "DMH", "LABELS", "UNKNOWN",
    "classify_instant", "classify_sec_acceptance", "market_date", "reaction_session", "to_code", "to_label",
]
