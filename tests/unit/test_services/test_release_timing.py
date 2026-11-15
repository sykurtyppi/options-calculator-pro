"""One rule for when an earnings release lands relative to the NYSE session.

About eleven functions in eight files used to decide this with different
cutoffs and time zones, so the same release could be labelled differently,
and get a different reaction session and exit anchor, depending on which
module ran. Each regression below is a case one of them got wrong.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from services import release_timing as rt

NY = ZoneInfo("America/New_York")


def _ny(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=NY)


# ── instants ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("moment,code", [
    (_ny(2026, 11, 10, 7, 0), rt.BMO),
    (_ny(2026, 11, 10, 9, 29), rt.BMO),
    (_ny(2026, 11, 10, 9, 30), rt.DMH),
    (_ny(2026, 11, 10, 15, 59), rt.DMH),
    (_ny(2026, 11, 10, 16, 0), rt.AMC),
    (_ny(2026, 11, 10, 16, 5), rt.AMC),
    (_ny(2026, 7, 14, 16, 5), rt.AMC),     # summer (EDT)
    (_ny(2026, 1, 14, 7, 30), rt.BMO),     # winter (EST)
])
def test_an_instant_is_judged_on_the_new_york_clock(moment, code) -> None:
    assert rt.classify_instant(moment) == code
    assert rt.classify_instant(moment.astimezone(timezone.utc)) == code  # same instant in UTC
    assert rt.classify_instant(pd.Timestamp(moment)) == code
    assert rt.classify_instant(moment.timestamp()) == code              # unix seconds


def test_daylight_saving_moves_the_utc_hour_not_the_label() -> None:
    # 16:05 ET is 21:05 UTC in winter and 20:05 UTC in summer; both are AMC.
    assert rt.classify_instant(datetime(2026, 1, 14, 21, 5, tzinfo=timezone.utc)) == rt.AMC
    assert rt.classify_instant(datetime(2026, 7, 14, 20, 5, tzinfo=timezone.utc)) == rt.AMC
    # Regression: "UTC hour >= 20 is AMC" made a 15:30 ET winter release (20:30 UTC) AMC.
    assert rt.classify_instant(datetime(2026, 1, 14, 20, 30, tzinfo=timezone.utc)) == rt.DMH


def test_an_early_close_day_ends_the_session_at_one() -> None:
    # 2026-11-27 (day after Thanksgiving) closes at 13:00 ET.
    assert rt.classify_instant(_ny(2026, 11, 27, 13, 30)) == rt.AMC
    assert rt.classify_instant(_ny(2026, 11, 27, 12, 30)) == rt.DMH


@pytest.mark.parametrize("value", [
    date(2026, 11, 10),
    datetime(2026, 11, 10),                                  # naive midnight
    _ny(2026, 11, 10, 0, 0),                                 # New York midnight
    datetime(2026, 11, 10, tzinfo=timezone.utc),             # UTC midnight (19:00 ET the day before)
    pd.Timestamp("2026-11-10"),
    pd.Timestamp("2026-11-10", tz="America/New_York"),
    # A UTC date-only stamp shown in New York time (19:00 the evening before).
    pd.Timestamp("2026-11-10T00:00:00Z").tz_convert("America/New_York"),
    pd.NaT, None, float("nan"), 0, -5, True,
])
def test_a_value_with_no_time_of_day_is_unknown_not_before_the_open(value) -> None:
    assert rt.classify_instant(value) == rt.UNKNOWN
    assert rt.to_code(value) == rt.UNKNOWN


def test_naive_times_are_new_york_wall_clock() -> None:
    assert rt.classify_instant(datetime(2026, 11, 10, 7, 0)) == rt.BMO
    assert rt.classify_instant(datetime(2026, 11, 10, 16, 30)) == rt.AMC


# ── tokens ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("token,code", [
    ("AMC", rt.AMC), ("amc", rt.AMC), ("after market close", rt.AMC), ("After-Hours", rt.AMC), ("post", rt.AMC),
    ("PM", rt.AMC), ("After the close", rt.AMC),
    ("BMO", rt.BMO), ("before market open", rt.BMO), ("Pre-Market", rt.BMO), ("pre", rt.BMO), ("am", rt.BMO),
    ("DMH", rt.DMH), ("during market hours", rt.DMH), ("INTRADAY (11:30 ET)", rt.DMH), ("intraday", rt.DMH),
    ("unknown", rt.UNKNOWN), ("TBD", rt.UNKNOWN), ("Time Not Supplied", rt.UNKNOWN), ("", rt.UNKNOWN),
    ("  ", rt.UNKNOWN), ("gibberish", rt.UNKNOWN),
    ("4:05 PM", rt.AMC), ("07:00 AM", rt.BMO), ("9:30 am", rt.DMH), ("16:00", rt.AMC), ("08:15:00", rt.BMO),
    ("4:05 pm ET", rt.AMC), ("13:61", rt.UNKNOWN), ("13:00 pm", rt.UNKNOWN),
    ("12:30 PM", rt.DMH), ("12:15 AM", rt.BMO), ("12:00 pm", rt.DMH),
])
def test_tokens(token, code) -> None:
    assert rt.to_code(token) == code


def test_labels_round_trip() -> None:
    for code, label in rt.LABELS.items():
        assert rt.to_label(code) == label and rt.to_code(label) == code


# ── SEC acceptance times ────────────────────────────────────────────────────


@pytest.mark.parametrize("value,code", [
    ("2026-10-29T06:01:36.000Z", rt.BMO),   # BMO whether the Z is UTC or Eastern
    ("2026-10-29T21:15:00.000Z", rt.AMC),   # AMC either way
    ("2026-10-29T16:30:05.000Z", rt.UNKNOWN),  # AMC if Eastern, mid-session if UTC: no guess
    ("2026-10-29T13:00:00.000Z", rt.UNKNOWN),
    ("not a time", rt.UNKNOWN), (None, rt.UNKNOWN),
])
def test_sec_acceptance_is_used_only_when_both_time_zone_readings_agree(value, code) -> None:
    assert rt.classify_sec_acceptance(value) == code


# ── the reaction session ────────────────────────────────────────────────────


@pytest.mark.parametrize("earnings_date,timing,session", [
    (date(2026, 11, 10), "BMO", date(2026, 11, 10)),
    (date(2026, 11, 10), "during market hours", date(2026, 11, 10)),
    (date(2026, 11, 10), "AMC", date(2026, 11, 11)),
    (date(2026, 11, 10), "UNKNOWN", date(2026, 11, 11)),     # unknown is treated as AMC
    (date(2026, 11, 13), "AMC", date(2026, 11, 16)),         # Friday close -> Monday, not Saturday
    (date(2026, 11, 25), "AMC", date(2026, 11, 27)),         # before Thanksgiving -> the early-close Friday
    (date(2026, 4, 2), "after market close", date(2026, 4, 6)),  # before Good Friday -> Monday
    (date(2026, 11, 14), "BMO", date(2026, 11, 16)),         # a Saturday date rolls to the next session
])
def test_reaction_session(earnings_date, timing, session) -> None:
    assert rt.reaction_session(earnings_date, timing) == session


# ── every caller applies the same rule ──────────────────────────────────────


CASES = [
    _ny(2026, 1, 14, 7, 0),    # was UNKNOWN in the research DB (naive-UTC stamp, 0-10h window)
    _ny(2026, 1, 14, 10, 0),   # was BMO under the UTC rule and the noon-ET rules
    _ny(2026, 1, 14, 11, 30),  # was BMO under the noon-ET rules
    _ny(2026, 1, 14, 15, 30),  # was AMC under the UTC rule (20:30 UTC)
    _ny(2026, 1, 14, 16, 5),
    _ny(2026, 7, 14, 16, 5),
]


@pytest.mark.parametrize("moment", CASES)
def test_every_wrapper_agrees_with_the_rule(moment) -> None:
    from scripts import build_earnings_iv_labels as labels
    from scripts import forward_screener
    from scripts import integrate_historical_replay_inputs as integrate
    from services import earnings_event_service as ees
    from services.earnings_move_profile import normalize_release_timing
    from services.institutional_ml_db import InstitutionalMLDatabase
    from web.api import screener_engine

    code = rt.classify_instant(moment)
    unix = {"earningsTimestamp": moment.timestamp()}
    stamp = pd.Timestamp(moment)

    assert ees._infer_release_timing_from_info(unix) == code
    assert screener_engine._infer_release_timing(unix) == code
    fs = forward_screener.get_release_timing(unix)
    assert (rt.DMH if fs.startswith("INTRADAY") else fs) == code
    assert normalize_release_timing(stamp) == rt.LABELS[code]
    assert normalize_release_timing(stamp.tz_convert("UTC")) == rt.LABELS[code]
    assert labels._infer_release_timing(stamp, None) == code
    assert InstitutionalMLDatabase._infer_release_timing_from_earnings_row(None, stamp, None) == code
    assert integrate._infer_timing_from_timestamp(stamp)[0] == code


def test_wrappers_return_unknown_for_a_date_only_stamp() -> None:
    from scripts import build_earnings_iv_labels as labels
    from services.earnings_move_profile import normalize_release_timing
    from services.institutional_ml_db import InstitutionalMLDatabase

    midnight = pd.Timestamp("2026-11-10", tz="America/New_York")
    assert labels._infer_release_timing(midnight, None) == rt.UNKNOWN
    assert InstitutionalMLDatabase._infer_release_timing_from_earnings_row(None, midnight, None) == rt.UNKNOWN
    assert normalize_release_timing(midnight) == "unknown"


def test_an_explicit_time_column_still_wins_over_the_stamp() -> None:
    from scripts import build_earnings_iv_labels as labels

    row = pd.Series({"Event Time": "After Market Close"})
    assert labels._infer_release_timing(_ny(2026, 11, 10, 7, 0), row) == rt.AMC


def test_the_snapshot_and_payoff_reaction_days_use_the_rule() -> None:
    from services.earnings_vol_snapshot import _reaction_session_for_event
    from web.api.edge_math import post_event_valuation_days

    assert _reaction_session_for_event(date(2026, 11, 13), "after market close") == date(2026, 11, 16)
    # Regression: a short code ("BMO") was not recognised and got the AMC day.
    assert post_event_valuation_days(date(2026, 11, 9), 1, "BMO") == 1
    assert post_event_valuation_days(date(2026, 11, 9), 4, "after market close") == 7  # Fri close -> Mon


def test_the_rule_uses_the_market_calendars_session_times() -> None:
    from services.market_calendar import SESSION_OPEN, session_close

    assert SESSION_OPEN == time(9, 30) and session_close(date(2026, 11, 10)) == time(16, 0)


# ── the research fetch loops classify the provider's stamp, not naive UTC ───


class _FakeTicker:
    def __init__(self, stamps):
        index = pd.DatetimeIndex(stamps)
        self._frame = pd.DataFrame({"EPS Estimate": [1.0] * len(stamps)}, index=index)

    def get_earnings_dates(self, limit=60):  # noqa: ARG002
        return self._frame


# 07:00 ET is 12:00 UTC: read as naive UTC it fell in no window (UNKNOWN);
# 16:05 ET is 21:05 UTC.
# 20:30 ET is 01:30 UTC the next day: read as UTC it landed on the wrong date.
STAMPS = [pd.Timestamp("2026-01-14 07:00", tz="America/New_York"),
          pd.Timestamp("2026-01-21 16:05", tz="America/New_York"),
          pd.Timestamp("2026-01-27 20:30", tz="America/New_York")]


def test_the_label_builder_fetch_classifies_on_the_new_york_clock(monkeypatch) -> None:
    import logging

    from scripts import build_earnings_iv_labels as labels

    monkeypatch.setattr(labels.yf, "Ticker", lambda symbol: _FakeTicker(STAMPS))
    events = labels._fetch_earnings_from_yfinance("AAA", date(2026, 1, 1), date(2026, 1, 31),
                                                  logging.getLogger("test"))
    assert [(e.event_date, e.release_timing) for e in events] == [
        (date(2026, 1, 14), rt.BMO), (date(2026, 1, 21), rt.AMC), (date(2026, 1, 27), rt.AMC)]


def test_the_research_db_fetch_classifies_on_the_new_york_clock(tmp_path, monkeypatch) -> None:
    import services.institutional_ml_db as imldb

    db = imldb.InstitutionalMLDatabase(db_path=str(tmp_path / "ml.db"))
    monkeypatch.setattr(imldb.yf, "Ticker", lambda symbol: _FakeTicker(STAMPS))
    rows = db._fetch_and_cache_earnings_dates("AAA", datetime(2026, 1, 1), datetime(2026, 1, 31))
    assert [(r["event_date"].date(), r["release_timing"]) for r in rows] == [
        (date(2026, 1, 14), rt.BMO), (date(2026, 1, 21), rt.AMC), (date(2026, 1, 27), rt.AMC)]


@pytest.mark.parametrize("value,day", [
    (pd.Timestamp("2026-01-27 20:30", tz="America/New_York"), date(2026, 1, 27)),
    (pd.Timestamp("2026-01-28 01:30", tz="UTC"), date(2026, 1, 27)),
    (pd.Timestamp("2026-01-27", tz="UTC"), date(2026, 1, 27)),   # date-only keeps its date
    (datetime(2026, 1, 27, 20, 30), date(2026, 1, 27)),          # naive: New York wall clock
    (date(2026, 1, 27), date(2026, 1, 27)),
    (pd.NaT, None), (None, None), ("x", None),
])
def test_market_date(value, day) -> None:
    assert rt.market_date(value) == day


def test_the_replay_integrator_fetch_uses_new_york_dates_and_times(monkeypatch) -> None:
    import yfinance

    from scripts import integrate_historical_replay_inputs as integrate

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: _FakeTicker(STAMPS))
    events = integrate._fetch_yfinance_events("AAA", date(2026, 1, 1), date(2026, 1, 31))
    assert [(e.event_date, e.release_timing) for e in events] == [
        (date(2026, 1, 14), rt.BMO), (date(2026, 1, 21), rt.AMC), (date(2026, 1, 27), rt.AMC)]
