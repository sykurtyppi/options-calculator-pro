"""
US-equity (NYSE) trading sessions from the exchange's holiday rules.

Weekday arithmetic mistakes a holiday for a session: earnings after the close
on Thursday 2026-07-02 react on Monday 07-06, not on the Independence Day
holiday (Friday 07-03). This module resolves sessions from the published
NYSE rules, with no external calendar dependency:

* New Year's Day (Sunday -> Monday; a Saturday holiday is not observed on
  the prior Friday, which closes a trading year).
* Martin Luther King Jr. Day (3rd Monday of January, from 1998).
* Washington's Birthday (3rd Monday of February).
* Good Friday.
* Memorial Day (last Monday of May).
* Juneteenth (June 19, from 2022).
* Independence Day, Christmas and Juneteenth: Saturday -> Friday,
  Sunday -> Monday.
* Labor Day (1st Monday of September), Thanksgiving (4th Thursday of
  November).
* Past one-off closures (national days of mourning, 9/11, Hurricane Sandy)
  are listed explicitly. Future one-off closures cannot be known in advance.

Early closes (1pm sessions) are still full sessions here.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import FrozenSet

# Unscheduled full-day closures since 1990.
_SPECIAL_CLOSURES: FrozenSet[date] = frozenset({
    date(1994, 4, 27),   # President Nixon, national day of mourning
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11),   # President Reagan
    date(2007, 1, 2),    # President Ford
    date(2012, 10, 29), date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),   # President G. H. W. Bush
    date(2025, 1, 9),    # President Carter
})


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th ``weekday`` (Mon=0) of a month."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    following = date(year + month // 12, month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(day: date) -> date:
    """Saturday holidays close the Friday before, Sunday ones the Monday after."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=None)
def nyse_holidays(year: int) -> FrozenSet[date]:
    """Full-day NYSE holidays observed in ``year``."""
    days = set()
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        days.add(new_year + timedelta(days=1))
    elif new_year.weekday() != 5:
        days.add(new_year)
    if year >= 1998:
        days.add(_nth_weekday(year, 1, 0, 3))   # Martin Luther King Jr. Day
    days.add(_nth_weekday(year, 2, 0, 3))       # Washington's Birthday
    days.add(_easter(year) - timedelta(days=2))  # Good Friday
    days.add(_last_weekday(year, 5, 0))         # Memorial Day
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    days.add(_observed(date(year, 7, 4)))       # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))       # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))      # Thanksgiving
    days.add(_observed(date(year, 12, 25)))     # Christmas
    days.update(day for day in _SPECIAL_CLOSURES if day.year == year)
    return frozenset(days)


def is_trading_day(day: date) -> bool:
    """True when NYSE holds a regular (full or early-close) session on ``day``."""
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


def next_session_on_or_after(day: date) -> date:
    """The first NYSE session on or after ``day``."""
    while not is_trading_day(day):
        day += timedelta(days=1)
    return day


__all__ = ["is_trading_day", "next_session_on_or_after", "nyse_holidays"]
