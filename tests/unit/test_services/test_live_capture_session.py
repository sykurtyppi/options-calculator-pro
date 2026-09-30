"""Live snapshots are dated by the NYSE session their quotes belong to.

The capture used the machine's local date. Run in the New York evening from a
machine ahead of UTC (e.g. Iceland), an after-close report's pre-reaction
close was stamped the next day and labelled "post"; run before the open on a
before-open report day, the previous close was labelled "post" too.

Also pins that a swallowed yfinance history failure stays "unavailable"
(never a known zero dividend).
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import services.dividend_yields as dividend_yields
from services.institutional_ml_db import InstitutionalMLDatabase
from services.market_calendar import quote_session


def _utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


@pytest.mark.parametrize("moment, session", [
    (_utc(2026, 7, 29, 1, 0), date(2026, 7, 28)),    # Tue 21:00 ET, Iceland already Wed
    (_utc(2026, 7, 28, 12, 0), date(2026, 7, 27)),   # Tue 08:00 ET, before the open
    (_utc(2026, 7, 28, 13, 30), date(2026, 7, 28)),  # Tue 09:30 ET, the open
    (_utc(2026, 8, 1, 15, 0), date(2026, 7, 31)),    # Saturday -> Friday
    (_utc(2026, 7, 6, 12, 0), date(2026, 7, 2)),     # Mon pre-open after the Fri 07-03 holiday
])
def test_quote_session(moment, session):
    assert quote_session(moment) == session


def test_quote_session_rejects_a_naive_clock():
    with pytest.raises(ValueError):
        quote_session(datetime(2026, 7, 28, 12, 0))


def _capture(tmp_path, now, event_day, timing):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, *args, **kwargs):
            idx = pd.bdate_range(end=pd.Timestamp("2026-07-28"), periods=300)
            return pd.DataFrame({"Close": np.full(len(idx), 60.0)}, index=idx)

        options = ["2026-08-07", "2026-08-14"]

        def option_chain(self, expiry):
            return object()

    db = InstitutionalMLDatabase(db_path=str(tmp_path / "live.db"))
    db._utc_now = lambda: now
    db._get_symbol_earnings_dates = lambda **_: [{"event_date": event_day, "release_timing": timing}]
    db._extract_atm_iv_from_chain = lambda chain, price: (0.40, 60.0)
    with mock.patch("yfinance.Ticker", FakeTicker), \
            mock.patch("services.pricing_rates.get_historical_risk_free_rate", return_value=(0.05, "stub")), \
            mock.patch("services.dividend_yields.get_historical_dividend_yield", return_value=(0.0, "stub")):
        db.capture_earnings_option_snapshots(symbols=["XYZ"])
    with sqlite3.connect(db.db_path) as conn:
        return conn.execute(
            "SELECT capture_date, relative_day, snapshot_phase, pricing_inputs_observed_on "
            "FROM earnings_option_snapshots"
        ).fetchall()


def test_new_york_evening_run_from_ahead_of_utc_is_still_pre_reaction(tmp_path):
    # AMC report Tue 07-28; run at Tue 21:00 ET = Wed 01:00 UTC.
    rows = _capture(tmp_path, _utc(2026, 7, 29, 1, 0), date(2026, 7, 28), "AMC")
    assert rows == [("2026-07-28", 0, "pre", "2026-07-28")]


def test_pre_open_run_on_a_before_open_report_day_is_pre_reaction(tmp_path):
    # BMO report Tue 07-28; run at 08:00 ET, quotes are Monday's close.
    rows = _capture(tmp_path, _utc(2026, 7, 28, 12, 0), date(2026, 7, 28), "BMO")
    assert rows == [("2026-07-27", -1, "pre", "2026-07-27")]


def test_after_the_open_on_a_before_open_report_day_is_post(tmp_path):
    rows = _capture(tmp_path, _utc(2026, 7, 28, 15, 0), date(2026, 7, 28), "BMO")
    assert rows == [("2026-07-28", 0, "post", "2026-07-28")]


def test_weekend_run_uses_fridays_session(tmp_path):
    # AMC report Fri 07-31; run Saturday: quotes are Friday's close, pre-reaction.
    rows = _capture(tmp_path, _utc(2026, 8, 1, 15, 0), date(2026, 7, 31), "AMC")
    assert rows == [("2026-07-31", 0, "pre", "2026-07-31")]


def test_swallowed_yfinance_history_failure_is_unavailable_not_zero():
    """yfinance hides a failed history request (empty prices, "possibly
    delisted"); its dividends accessor then raises. That must stay
    "unavailable", never become a known zero dividend."""
    import yfinance as yf
    from yfinance.scrapers import history as yf_history

    dividend_yields.reset_cache()
    with mock.patch.object(yf_history.PriceHistory, "tz", "America/New_York", create=True), \
            mock.patch("yfinance.data.YfData.cache_get", side_effect=RuntimeError("HTTP 500")), \
            mock.patch("yfinance.data.YfData.get", side_effect=RuntimeError("HTTP 500")):
        assert dividend_yields.get_historical_dividend_yield("KO", date(2024, 3, 1), 60.0) == (None, "unavailable")
    dividend_yields.reset_cache()
