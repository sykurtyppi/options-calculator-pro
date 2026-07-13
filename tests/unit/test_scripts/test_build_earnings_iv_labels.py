"""Leakage tests for scripts/build_earnings_iv_labels.py.

The label pipeline's core validity guarantee — "pre_event_date is always
strictly < earnings_date" — was previously asserted only in the module
docstring. These tests pin it to the extracted, pure
`_select_pre_post_event_dates` so a future trade_date can never bleed into a
pre-event snapshot without a red test.
"""
from datetime import date, timedelta

from scripts.build_earnings_iv_labels import _select_pre_post_event_dates


def _trading_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_pre_date_is_strictly_before_event_never_on_or_after():
    event = date(2026, 5, 1)  # Friday
    # Include the event day itself and several days AFTER it in the pool.
    dates = _trading_days(date(2026, 4, 20), 20)
    assert event in dates and any(d > event for d in dates)

    pre, post, reason = _select_pre_post_event_dates(dates, event)
    assert reason is None
    # The leakage invariant: pre is strictly before the event, post strictly after.
    assert pre < event, "pre_date leaked onto/after the earnings date"
    assert post > event
    # Even though the event day and later days are present, pre is never one of them.
    assert pre == max(d for d in dates if d < event)
    assert post == min(d for d in dates if d > event)


def test_event_day_trade_date_is_excluded_from_both_legs():
    # A trade_date exactly on the event date must be neither pre nor post.
    event = date(2026, 5, 1)
    dates = [date(2026, 4, 30), event, date(2026, 5, 4)]
    pre, post, reason = _select_pre_post_event_dates(dates, event)
    assert reason is None
    assert pre == date(2026, 4, 30)
    assert post == date(2026, 5, 4)
    assert event not in (pre, post)


def test_no_pre_event_date_when_all_dates_are_on_or_after_event():
    event = date(2026, 5, 1)
    dates = [event, date(2026, 5, 2), date(2026, 5, 5)]  # nothing strictly before
    pre, post, reason = _select_pre_post_event_dates(dates, event)
    assert reason == "no_pre_event_trade_date"
    assert pre is None and post is None


def test_no_post_event_date_when_nothing_within_forward_window():
    event = date(2026, 5, 1)
    # Post candidate exists but is beyond the forward window → rejected.
    dates = [date(2026, 4, 30), event + timedelta(days=99)]
    pre, post, reason = _select_pre_post_event_dates(dates, event, post_max_days_forward=10)
    assert reason == "no_post_event_trade_date"
    assert pre is None and post is None


def test_post_date_respects_the_forward_window_boundary():
    event = date(2026, 5, 1)
    inside = event + timedelta(days=10)
    outside = event + timedelta(days=11)
    dates = [date(2026, 4, 30), outside, inside]
    pre, post, reason = _select_pre_post_event_dates(dates, event, post_max_days_forward=10)
    assert reason is None
    assert post == inside  # nearest within window, not the out-of-window one
