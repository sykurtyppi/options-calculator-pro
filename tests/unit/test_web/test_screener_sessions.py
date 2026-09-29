"""Screener dates follow the NYSE calendar (post-#159 deep audit).

1. The T-3 entry date counted weekdays, so it could land on a holiday.
2. "Next monthly opex" took the first expiry dated the 15th or later, which
   is often a weekly.
"""
from __future__ import annotations

from datetime import date

import pytest

from services.market_calendar import monthly_option_expiry
from web.api.screener_engine import _business_days_before, _pick_next_monthly_opex


@pytest.mark.parametrize("event, expected", [
    (date(2026, 7, 8), date(2026, 7, 2)),     # skips Fri 07-03 (Independence Day observed)
    (date(2026, 4, 8), date(2026, 4, 2)),     # skips Good Friday 04-03
    (date(2026, 1, 22), date(2026, 1, 16)),   # skips MLK Day 01-19
    (date(2026, 11, 30), date(2026, 11, 24)), # skips Thanksgiving 11-26
    (date(2026, 3, 12), date(2026, 3, 9)),    # an ordinary week
])
def test_entry_is_three_sessions_before_the_event(event, expected):
    assert _business_days_before(event, 3) == expected


def test_next_monthly_is_the_third_friday_not_a_late_weekly():
    expirations = ["2026-09-18", "2026-09-25", "2026-10-02", "2026-10-09", "2026-10-16", "2026-10-23"]
    assert _pick_next_monthly_opex(expirations, date(2026, 9, 19)) == "2026-10-16"


def test_holiday_monthly_expires_the_thursday_before():
    # April 2025: the third Friday (04-18) was Good Friday.
    assert monthly_option_expiry(2025, 4) == date(2025, 4, 17)
    assert _pick_next_monthly_opex(["2025-04-11", "2025-04-17", "2025-04-25"], date(2025, 4, 10)) == "2025-04-17"


def test_no_listed_monthly_is_reported_as_none():
    assert _pick_next_monthly_opex(["2026-09-25", "2026-10-02"], date(2026, 9, 19)) is None
