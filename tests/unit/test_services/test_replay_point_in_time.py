"""Point-in-time pricing inputs for historical snapshots and replay (Hermes review #2).

Replay used to price stored historical snapshots with TODAY's risk-free rate
and dividend yield, so the same stored pair replayed to different results on
different days. Snapshots now store the inputs known on their capture date;
replay uses only those, and refuses legacy pairs that lack them.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from unittest import mock

import pandas as pd
import pytest

import services.dividend_yields as dividend_yields
import services.pricing_rates as pricing_rates
from services.institutional_ml_db import (
    UNPRICEABLE_MISSING_HISTORICAL_INPUTS,
    ExecutionCostModel,
    InstitutionalMLDatabase,
    SnapshotReplayPair,
)


@pytest.fixture(autouse=True)
def _fresh_caches():
    pricing_rates.reset_cache()
    dividend_yields.reset_cache()
    yield
    pricing_rates.reset_cache()
    dividend_yields.reset_cache()


# ── historical inputs never look past as_of ──────────────────────────────────


def _irx_history(rows):
    frame = pd.DataFrame({"Close": [value for _, value in rows]},
                         index=pd.DatetimeIndex([pd.Timestamp(day) for day, _ in rows]))
    ticker = mock.MagicMock()
    ticker.history.return_value = frame
    return ticker


def test_historical_rate_is_the_last_close_on_or_before_as_of():
    ticker = _irx_history([("2024-03-04", 5.20), ("2024-03-05", 5.25), ("2024-03-06", 3.00)])
    with mock.patch.object(pricing_rates.yf, "Ticker", return_value=ticker):
        rate, source = pricing_rates.get_historical_risk_free_rate(date(2024, 3, 5))
    assert rate == pytest.approx(0.0525)  # not the later 3.00 close
    assert source == pricing_rates.SOURCE_HISTORICAL_IRX


def test_historical_rate_unavailable_is_not_substituted():
    ticker = mock.MagicMock()
    ticker.history.side_effect = RuntimeError("network down")
    with mock.patch.object(pricing_rates.yf, "Ticker", return_value=ticker):
        assert pricing_rates.get_historical_risk_free_rate(date(2024, 3, 5)) == (None, "unavailable")


def _dividend_ticker(rows):
    ticker = mock.MagicMock()
    ticker.dividends = pd.Series(
        [amount for _, amount in rows],
        index=pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day, _ in rows]),
    )
    return ticker


def test_historical_dividend_yield_counts_only_dividends_paid_by_as_of():
    ticker = _dividend_ticker([
        ("2023-02-10", 0.50),   # outside the trailing year
        ("2023-05-10", 0.50), ("2023-08-10", 0.50), ("2023-11-10", 0.50), ("2024-02-09", 0.50),
        ("2024-05-10", 5.00),   # after as_of: a later regime, must be ignored
    ])
    with mock.patch("yfinance.Ticker", return_value=ticker):
        q, source = dividend_yields.get_historical_dividend_yield("XYZ", date(2024, 3, 1), 100.0)
    assert q == pytest.approx(0.02)
    assert source == dividend_yields.SOURCE_TRAILING_HISTORICAL


def test_historical_dividend_yield_non_payer_and_failure():
    with mock.patch("yfinance.Ticker", return_value=_dividend_ticker([])):
        assert dividend_yields.get_historical_dividend_yield("NOPAY", date(2024, 3, 1), 50.0)[0] == 0.0
    dividend_yields.reset_cache()
    with mock.patch("yfinance.Ticker", side_effect=RuntimeError("down")):
        assert dividend_yields.get_historical_dividend_yield("XYZ", date(2024, 3, 1), 50.0) == (None, "unavailable")


# ── schema migration ─────────────────────────────────────────────────────────


def test_existing_database_gains_the_pricing_columns(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE earnings_option_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, event_date TEXT NOT NULL,
                capture_date TEXT NOT NULL, relative_day INTEGER NOT NULL, release_timing TEXT NOT NULL,
                snapshot_phase TEXT NOT NULL, short_expiry TEXT, long_expiry TEXT, atm_strike REAL,
                front_iv REAL NOT NULL, back_iv REAL NOT NULL, term_ratio REAL NOT NULL,
                underlying_price REAL NOT NULL, source TEXT NOT NULL DEFAULT 'yfinance_live',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, event_date, capture_date, short_expiry, long_expiry))"""
        )
    InstitutionalMLDatabase(db_path=str(path))
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(earnings_option_snapshots)")}
    assert {"pricing_risk_free_rate", "pricing_risk_free_rate_source", "pricing_dividend_yield",
            "pricing_dividend_yield_source", "pricing_inputs_observed_on"} <= columns


# ── historical backfill stores point-in-time inputs ──────────────────────────


EVENT = date(2026, 6, 10)  # a Wednesday, after the close


class _FakeMarketData:
    def is_available(self):
        return True

    def get_option_chain(self, symbol, expiration="all", strike_limit=5, date=None):
        rows = []
        for expiry, iv in (("2026-06-12", 0.60), ("2026-06-19", 0.45)):
            for strike in (95.0, 100.0, 105.0):
                rows.append({"expiration_date": expiry, "strike": strike, "impliedVolatility": iv,
                             "underlyingPrice": 100.0})
        return pd.DataFrame(rows)


def _backfill_db(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "inst.db"), mda_client=_FakeMarketData())
    with sqlite3.connect(db.db_path) as conn:
        for day in pd.bdate_range("2026-05-25", "2026-06-20"):
            conn.execute(
                "INSERT INTO daily_prices (symbol, date, open_price, high_price, low_price, close_price, "
                "volume, adj_close) VALUES ('XYZ', ?, 100, 101, 99, 100, 1000000, 100)",
                (day.strftime("%Y-%m-%d"),),
            )
    db._get_cached_earnings_dates = lambda **_: [(datetime.combine(EVENT, datetime.min.time()), "mdapp", "AMC")]
    return db


def _live_services_forbidden():
    boom = mock.Mock(side_effect=AssertionError("today's rate/dividend must not be used"))
    return (mock.patch.object(pricing_rates, "get_pricing_risk_free_rate", boom),
            mock.patch.object(dividend_yields, "get_dividend_yield", boom))


def test_backfill_stores_the_inputs_known_on_each_capture_date(tmp_path):
    db = _backfill_db(tmp_path)
    rates = {date(2026, 6, 5): 0.0501, date(2026, 6, 11): 0.0499}
    hist_rate = mock.Mock(side_effect=lambda day: (rates[day], pricing_rates.SOURCE_HISTORICAL_IRX))
    hist_q = mock.Mock(return_value=(0.012, dividend_yields.SOURCE_TRAILING_HISTORICAL))
    live_rate, live_q = _live_services_forbidden()
    with mock.patch.object(pricing_rates, "get_historical_risk_free_rate", hist_rate), \
            mock.patch.object(dividend_yields, "get_historical_dividend_yield", hist_q), live_rate, live_q:
        summary = db.capture_historical_iv_snapshots_mda(symbols=["XYZ"], lookback_years=1)

    assert summary["captured"] == 2 and summary["diagnostics"]["no_point_in_time_inputs"] == 0
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute(
            "SELECT capture_date, pricing_risk_free_rate, pricing_risk_free_rate_source, "
            "pricing_dividend_yield, pricing_inputs_observed_on FROM earnings_option_snapshots ORDER BY capture_date"
        ).fetchall()
    assert rows == [
        ("2026-06-05", 0.0501, pricing_rates.SOURCE_HISTORICAL_IRX, 0.012, "2026-06-05"),
        ("2026-06-11", 0.0499, pricing_rates.SOURCE_HISTORICAL_IRX, 0.012, "2026-06-11"),
    ]


def test_backfill_fails_closed_without_point_in_time_inputs(tmp_path):
    db = _backfill_db(tmp_path)
    live_rate, live_q = _live_services_forbidden()
    with mock.patch.object(pricing_rates, "get_historical_risk_free_rate", return_value=(None, "unavailable")), \
            mock.patch.object(dividend_yields, "get_historical_dividend_yield",
                              return_value=(0.0, dividend_yields.SOURCE_TRAILING_HISTORICAL)), live_rate, live_q:
        summary = db.capture_historical_iv_snapshots_mda(symbols=["XYZ"], lookback_years=1)

    assert summary["captured"] == 0 and summary["diagnostics"]["no_point_in_time_inputs"] == 2
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM earnings_option_snapshots").fetchone()[0] == 0


# ── replay uses only stored inputs ───────────────────────────────────────────


def _insert_pair(db, *, with_inputs):
    rows = [
        ("XYZ", "2026-03-01", "2026-02-25", -4, "AMC", "pre", "2026-03-06", "2026-03-13", 200.0, 0.65, 0.55, 1.18,
         205.0, "unit_test", 0.0520, "stored", 0.004, "stored", "2026-02-25"),
        ("XYZ", "2026-03-01", "2026-03-02", 1, "AMC", "post", "2026-03-06", "2026-03-13", 200.0, 0.48, 0.50, 0.96,
         202.0, "unit_test", 0.0515, "stored", 0.004, "stored", "2026-03-02"),
    ]
    if not with_inputs:
        rows = [row[:14] + (None, None, None, None, None) for row in rows]
    with sqlite3.connect(db.db_path) as conn:
        conn.executemany(
            """INSERT INTO earnings_option_snapshots
               (symbol, event_date, capture_date, relative_day, release_timing, snapshot_phase, short_expiry,
                long_expiry, atm_strike, front_iv, back_iv, term_ratio, underlying_price, source,
                pricing_risk_free_rate, pricing_risk_free_rate_source, pricing_dividend_yield,
                pricing_dividend_yield_source, pricing_inputs_observed_on)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def _replay(db, pair):
    return db._simulate_snapshot_replay_trade(
        session_id="s", setup_score=0.7, contracts=1, execution_profile="institutional",
        execution_cost_model=ExecutionCostModel("institutional"), snapshot_pair=pair,
        crush_context=None, daily_share_volume=5_000_000, volume_ratio=1.0,
    )


def test_replay_is_the_same_whatever_todays_rates_are(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "inst.db"))
    _insert_pair(db, with_inputs=True)
    pair = db._load_snapshot_replay_pair("XYZ", datetime(2026, 3, 1))
    assert (pair.pre_risk_free_rate, pair.post_risk_free_rate) == (0.0520, 0.0515)
    assert (pair.pre_dividend_yield, pair.post_dividend_yield) == (0.004, 0.004)

    results = []
    for today_rate, today_q in ((0.01, 0.0), (0.09, 0.05)):
        with mock.patch.object(pricing_rates, "get_pricing_risk_free_rate", return_value=(today_rate, "x")), \
                mock.patch.object(dividend_yields, "get_dividend_yield", return_value=(today_q, "x")):
            trade = _replay(db, pair)
        results.append((trade.debit_per_contract, trade.pnl_per_contract))
    assert results[0] == results[1]


def test_replay_prices_entry_and_exit_with_their_own_stored_inputs(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "inst.db"))
    _insert_pair(db, with_inputs=True)
    pair = db._load_snapshot_replay_pair("XYZ", datetime(2026, 3, 1))
    trade = _replay(db, pair)
    expected_entry = db._calendar_spread_market_value_from_snapshot(
        underlying_price=205.0, strike=200.0, as_of_date=datetime(2026, 2, 25), short_expiry="2026-03-06",
        long_expiry="2026-03-13", front_iv=0.65, back_iv=0.55, risk_free_rate=0.0520, dividend_yield=0.004,
    )
    assert trade.debit_per_contract == pytest.approx(expected_entry)


def test_legacy_pair_without_stored_inputs_is_unpriceable(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "inst.db"))
    _insert_pair(db, with_inputs=False)
    pair = db._load_snapshot_replay_pair("XYZ", datetime(2026, 3, 1))
    assert pair is not None and pair.pre_risk_free_rate is None

    live_rate, live_q = _live_services_forbidden()
    with live_rate, live_q:
        assert _replay(db, pair) is None
    assert db.replay_unpriceable == {UNPRICEABLE_MISSING_HISTORICAL_INPUTS: 1}


def test_partially_stored_inputs_are_unpriceable(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "inst.db"))
    pair = SnapshotReplayPair(
        symbol="XYZ", event_date=datetime(2026, 3, 1), release_timing="AMC",
        pre_capture_date=datetime(2026, 2, 25), post_capture_date=datetime(2026, 3, 2),
        short_expiry="2026-03-06", long_expiry="2026-03-13", atm_strike=200.0,
        pre_front_iv=0.65, pre_back_iv=0.55, post_front_iv=0.48, post_back_iv=0.50,
        pre_underlying_price=205.0, post_underlying_price=202.0,
        pre_risk_free_rate=0.05, pre_dividend_yield=0.0, post_risk_free_rate=None, post_dividend_yield=0.0,
    )
    assert _replay(db, pair) is None
