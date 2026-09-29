"""Regressions for the deep audit of #156 (point-in-time replay inputs).

1. split-adjusted dividends were divided by an as-traded price, so a later
   split skewed the dividend yield by the split ratio;
2. live capture stored fallback defaults (4.5%, 0%) as observed inputs;
3. an older legacy snapshot hid a priceable one for the same event;
4. the unpriceable counter accumulated across runs and refused pairs were
   still logged as replays;
plus: frequency-based annualisation (no fifth quarterly dividend), a 0.00%
T-bill close is a valid rate, and the pairing report counts priceable events.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import services.dividend_yields as dividend_yields
import services.pricing_rates as pricing_rates
from services.institutional_ml_db import (
    UNPRICEABLE_MISSING_HISTORICAL_INPUTS,
    BacktestSession,
    InstitutionalMLDatabase,
)


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.delenv("OPTIONS_PRICING_RISK_FREE_RATE", raising=False)
    pricing_rates.reset_cache()
    dividend_yields.reset_cache()
    yield
    pricing_rates.reset_cache()
    dividend_yields.reset_cache()


def _series(rows):
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day, _ in rows]),
        dtype=float,
    )


def _ticker(dividends, splits=()):
    ticker = mock.MagicMock()
    ticker.dividends = _series(dividends)
    ticker.splits = _series(splits)
    return ticker


def _q(dividends, splits=(), as_of=date(2024, 3, 1), price=100.0):
    dividend_yields.reset_cache()
    with mock.patch("yfinance.Ticker", return_value=_ticker(dividends, splits)):
        return dividend_yields.get_historical_dividend_yield("XYZ", as_of, price)[0]


QUARTERLY = [("2023-05-10", 0.5), ("2023-08-10", 0.5), ("2023-11-10", 0.5), ("2024-02-09", 0.5)]


# ── 1. splits after as_of ────────────────────────────────────────────────────


def test_a_later_split_does_not_shrink_the_yield():
    # yfinance reports these /10 after a 10:1 split in 2024-07; the price on
    # 2024-03-01 is as-traded (1000), so the true yield is 20/1000 = 2%.
    adjusted = [(day, amount / 10) for day, amount in [(d, a * 10) for d, a in QUARTERLY]]
    assert _q(adjusted, splits=[("2024-07-15", 10.0)], price=1000.0 / 10 * 10) == pytest.approx(
        sum(a for _, a in QUARTERLY) * 10 / 1000.0
    )


def test_a_later_reverse_split_does_not_inflate_or_drop_the_yield():
    # 1:20 reverse split later: yfinance multiplies past dividends by 20.
    adjusted = [(day, amount * 20) for day, amount in [(d, 0.05) for d, _ in QUARTERLY]]
    q = _q(adjusted, splits=[("2024-06-03", 0.05)], price=8.0)
    assert q == pytest.approx(0.20 / 8.0)


def test_a_split_before_as_of_is_already_in_the_as_traded_amounts():
    assert _q(QUARTERLY, splits=[("2022-01-10", 4.0)]) == pytest.approx(0.02)


# ── frequency-based annualisation ────────────────────────────────────────────


def test_ex_date_drift_does_not_add_a_fifth_quarterly_dividend():
    drifting = [("2023-03-02", 0.5)] + QUARTERLY  # 364 days before as_of
    assert _q(drifting) == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("dividends", "expected"),
    [
        ([("2023-06-01", 1.0), ("2023-12-01", 1.0)], 0.02),                      # semi-annual
        ([("2023-03-10", 2.0)], 0.02),                                             # annual
        ([(f"2023-{m:02d}-15", 0.1) for m in range(3, 13)]
         + [("2024-01-15", 0.1), ("2024-02-15", 0.1)], 0.012),                    # monthly
        ([], 0.0),                                                                  # non-payer
        ([("2023-01-10", 0.5), ("2023-04-10", 0.5), ("2023-07-10", 0.5)], 0.0),    # suspended
    ],
)
def test_annualised_yield(dividends, expected):
    assert _q(dividends) == pytest.approx(expected)


def test_history_fetch_failure_is_unavailable():
    with mock.patch("yfinance.Ticker", side_effect=RuntimeError("down")):
        assert dividend_yields.get_historical_dividend_yield("XYZ", date(2024, 3, 1), 100.0) == (None, "unavailable")


def test_zero_rate_close_is_a_valid_historical_rate():
    ticker = mock.MagicMock()
    ticker.history.return_value = pd.DataFrame(
        {"Close": [0.01, 0.00]}, index=pd.DatetimeIndex(["2021-03-04", "2021-03-05"])
    )
    with mock.patch.object(pricing_rates.yf, "Ticker", return_value=ticker):
        assert pricing_rates.get_historical_risk_free_rate(date(2021, 3, 5)) == (0.0, pricing_rates.SOURCE_HISTORICAL_IRX)


# ── 2. live capture never stores fallbacks as observed inputs ────────────────


def _live_db(tmp_path, *, irx_fails):
    today = datetime.now().date()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, *args, **kwargs):
            if self.symbol == "^IRX":
                if irx_fails:
                    raise RuntimeError("429 Too Many Requests")
                idx = pd.DatetimeIndex([pd.Timestamp(today - timedelta(days=1)), pd.Timestamp(today)])
                return pd.DataFrame({"Close": [5.10, 5.12]}, index=idx)
            idx = pd.bdate_range(end=pd.Timestamp(today), periods=300)
            return pd.DataFrame({"Close": np.full(len(idx), 60.0)}, index=idx)

        @property
        def info(self):
            raise RuntimeError("429 Too Many Requests")

        @property
        def dividends(self):
            return _series([((today - timedelta(days=90 * k)).isoformat(), 0.51) for k in range(4, 0, -1)])

        @property
        def splits(self):
            return _series([])

        options = [(today + timedelta(days=d)).strftime("%Y-%m-%d") for d in (5, 12)]

        def option_chain(self, expiry):
            return object()

    db = InstitutionalMLDatabase(db_path=str(tmp_path / "i.db"))
    db._get_symbol_earnings_dates = lambda **_: [{"event_date": today + timedelta(days=3), "release_timing": "AMC"}]
    db._extract_atm_iv_from_chain = lambda chain, price: (0.40, 60.0)
    with mock.patch("yfinance.Ticker", FakeTicker):
        db.capture_earnings_option_snapshots(symbols=["KO"])
    with sqlite3.connect(db.db_path) as conn:
        return conn.execute(
            "SELECT pricing_risk_free_rate, pricing_risk_free_rate_source, pricing_dividend_yield, "
            "pricing_dividend_yield_source, pricing_inputs_observed_on FROM earnings_option_snapshots"
        ).fetchall()


def test_live_capture_stores_point_in_time_inputs(tmp_path):
    rows = _live_db(tmp_path, irx_fails=False)
    assert len(rows) == 1
    rate, rate_source, q, q_source, observed = rows[0]
    assert rate == pytest.approx(0.0512) and rate_source == pricing_rates.SOURCE_HISTORICAL_IRX
    assert q == pytest.approx(4 * 0.51 / 60.0) and q_source == dividend_yields.SOURCE_TRAILING_HISTORICAL
    assert observed == datetime.now().date().isoformat()


def test_live_capture_stores_no_inputs_when_a_feed_fails(tmp_path):
    rows = _live_db(tmp_path, irx_fails=True)
    assert len(rows) == 1  # the snapshot's IVs are still kept
    rate, rate_source, q, _, _ = rows[0]
    assert rate is None and rate_source == "unavailable"
    assert rate != pricing_rates.FALLBACK_STATIC_RATE


# ── 3. pair selection prefers priced snapshots ───────────────────────────────


def _insert(db, rows):
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


def _row(capture, rel, phase, *, priced, iv=0.6):
    inputs = (0.05, "stored", 0.004, "stored", capture) if priced else (None, None, None, None, None)
    return ("XYZ", "2026-03-02", capture, rel, "AMC", phase, "2026-03-06", "2026-03-13", 200.0, iv, 0.5,
            1.2, 205.0, "test") + inputs


def test_a_legacy_snapshot_does_not_hide_a_priced_one(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "i.db"))
    _insert(db, [
        _row("2026-02-25", -5, "pre", priced=True, iv=0.61),    # backfilled, priceable
        _row("2026-02-28", -2, "pre", priced=False, iv=0.70),   # later legacy live row
        _row("2026-03-03", 1, "post", priced=True, iv=0.45),
    ])
    pair = db._load_snapshot_replay_pair("XYZ", datetime(2026, 3, 2))
    assert pair.pre_capture_date.date() == date(2026, 2, 25)
    assert pair.pre_risk_free_rate == 0.05 and pair.post_risk_free_rate == 0.05


def test_legacy_rows_are_still_used_when_nothing_is_priced(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "i.db"))
    _insert(db, [_row("2026-02-28", -2, "pre", priced=False), _row("2026-03-03", 1, "post", priced=False)])
    pair = db._load_snapshot_replay_pair("XYZ", datetime(2026, 3, 2))
    assert pair is not None and pair.pre_risk_free_rate is None


def test_pairing_report_counts_priceable_events(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "i.db"))
    _insert(db, [
        _row("2026-02-25", -5, "pre", priced=True), _row("2026-03-03", 1, "post", priced=True),
    ])
    legacy = [r[:1] + ("2026-05-02",) + r[2:] for r in (
        _row("2026-04-27", -5, "pre", priced=False), _row("2026-05-04", 1, "post", priced=False))]
    _insert(db, legacy)
    progress = db.summarize_snapshot_pairing_progress()
    assert progress["pairable_events"] == 2 and progress["priceable_events"] == 1


# ── 4. counter per run; REPLAY logged only when priced ──────────────────────


def _walk_forward_db(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "inst.db"))
    _insert(db, [_row("2026-02-25", -4, "pre", priced=False), _row("2026-03-03", 1, "post", priced=False)])
    row = SimpleNamespace(symbol="XYZ", iv30_rv30_ratio=1.2, price_momentum_5d=0.0, volume=5e6, volume_ratio_10d=1.0)
    db._load_walk_forward_dataset = lambda *a, **k: pd.DataFrame({"date": [pd.Timestamp("2026-02-25")], "symbol": ["XYZ"]})
    db._load_iv_crush_profiles = lambda *a, **k: {}
    db._build_earnings_event_candidates = lambda **k: {pd.Timestamp("2026-02-25"): [
        {"row": row, "hold_days": 5, "event_date": datetime(2026, 3, 2), "days_to_earnings": 5}]}
    db._derive_crush_signal_context = lambda **k: {"profile_source": "symbol", "confidence": 1, "magnitude": 1,
                                                   "edge_score": 1}
    db._score_setup_quality = lambda **k: 0.9
    db._rank_candidate_for_alpha = lambda **k: 0.9
    return db


def test_counter_is_per_run_and_refused_pairs_are_not_logged_as_replays(tmp_path):
    db = _walk_forward_db(tmp_path)
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture()
    db.logger.addHandler(handler)
    try:
        for i in range(3):
            session = BacktestSession(f"s{i}", "x", datetime(2026, 1, 1), datetime(2026, 4, 1), ["XYZ"], {},
                                      0, 0, 0, 0, 0, 0, datetime.now())
            assert db._run_walk_forward_backtest(session, {"pricing_mode": "hybrid"}) == []
            assert db.replay_unpriceable == {UNPRICEABLE_MISSING_HISTORICAL_INPUTS: 1}
    finally:
        db.logger.removeHandler(handler)
    assert not any("📸 REPLAY" in message for message in messages)
    assert sum("⛔ UNPRICEABLE" in message for message in messages) == 3


def test_replay_script_reports_unpriceable_pairs():
    import scripts.run_replay_backtest as script

    db = SimpleNamespace(replay_routing={"snapshot_replay": 1, "synthetic_proxy": 1},
                         replay_unpriceable={UNPRICEABLE_MISSING_HISTORICAL_INPUTS: 1})
    summary = script.routing_summary(db)
    assert (summary["unpriceable"], summary["replay"], summary["synthetic"]) == (1, 1, 1)
