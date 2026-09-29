"""Regressions for the post-#158 pricing audit.

1. Replay paired a pre-event snapshot with a post-event snapshot of OTHER
   expiries and priced the post-event IVs as the entry's contracts.
2. The "first post-event session" skipped weekends but not exchange
   holidays (earnings after the close on Thu 2026-07-02 react on Mon 07-06,
   not on the Independence Day holiday).
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from services.institutional_ml_db import (
    UNPRICEABLE_MISMATCHED_CONTRACTS,
    BacktestSession,
    InstitutionalMLDatabase,
    SnapshotReplayPair,
    SnapshotReplayRefusal,
)
from services.market_calendar import is_trading_day, next_session_on_or_after, nyse_holidays
from web.api.edge_math import post_event_valuation_days

# ── 1. replay pairs only the same contracts ─────────────────────────────────

NEAR = ("2026-03-06", "2026-03-13")
FAR = ("2026-04-17", "2026-05-15")


def _row(capture, rel, phase, expiries, *, iv=0.6, priced=True):
    inputs = (0.05, "stored", 0.004, "stored", capture) if priced else (None, None, None, None, None)
    return ("XYZ", "2026-03-02", capture, rel, "AMC", phase, expiries[0], expiries[1], 200.0, iv, 0.5,
            1.2, 205.0, "test") + inputs


def _db(tmp_path, rows):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "i.db"))
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
    return db


def test_post_event_snapshot_of_other_expiries_is_refused(tmp_path):
    db = _db(tmp_path, [_row("2026-02-27", -3, "pre", NEAR), _row("2026-03-03", 1, "post", FAR, iv=0.3)])
    result = db._load_snapshot_replay_pair("XYZ", datetime(2026, 3, 2))
    assert isinstance(result, SnapshotReplayRefusal)
    assert result.reason == UNPRICEABLE_MISMATCHED_CONTRACTS


def test_an_earlier_pre_event_snapshot_of_the_same_contracts_is_used(tmp_path):
    db = _db(tmp_path, [
        _row("2026-02-24", -6, "pre", FAR, iv=0.62),
        _row("2026-02-27", -3, "pre", NEAR, iv=0.70),   # latest, but no post-event match
        _row("2026-03-03", 1, "post", FAR, iv=0.41),
    ])
    pair = db._load_snapshot_replay_pair("XYZ", datetime(2026, 3, 2))
    assert isinstance(pair, SnapshotReplayPair)
    assert (pair.short_expiry, pair.long_expiry) == FAR
    assert (pair.pre_front_iv, pair.post_front_iv) == (0.62, 0.41)
    assert pair.pre_capture_date.date() == date(2026, 2, 24)


def test_matching_prefers_priced_snapshots_on_both_sides(tmp_path):
    db = _db(tmp_path, [
        _row("2026-02-24", -6, "pre", NEAR, iv=0.61),
        _row("2026-02-27", -3, "pre", NEAR, iv=0.70, priced=False),
        _row("2026-03-03", 1, "post", NEAR, iv=0.44, priced=False),
        _row("2026-03-04", 2, "post", NEAR, iv=0.42),
    ])
    pair = db._load_snapshot_replay_pair("XYZ", datetime(2026, 3, 2))
    assert (pair.pre_front_iv, pair.post_front_iv) == (0.61, 0.42)
    assert pair.pre_risk_free_rate == 0.05 and pair.post_risk_free_rate == 0.05


def test_pairing_report_counts_only_same_contract_pairs_as_priceable(tmp_path):
    db = _db(tmp_path, [_row("2026-02-27", -3, "pre", NEAR), _row("2026-03-03", 1, "post", FAR)])
    progress = db.summarize_snapshot_pairing_progress()
    assert progress["pairable_events"] == 1 and progress["priceable_events"] == 0


def test_hybrid_replay_refuses_a_mismatched_pair_instead_of_using_the_proxy(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", NEAR), _row("2026-03-03", 1, "post", FAR)])
    row = SimpleNamespace(symbol="XYZ", iv30_rv30_ratio=1.2, price_momentum_5d=0.0, volume=5e6, volume_ratio_10d=1.0)
    db._load_walk_forward_dataset = lambda *a, **k: pd.DataFrame({"date": [pd.Timestamp("2026-02-25")], "symbol": ["XYZ"]})
    db._load_iv_crush_profiles = lambda *a, **k: {}
    db._build_earnings_event_candidates = lambda **k: {pd.Timestamp("2026-02-25"): [
        {"row": row, "hold_days": 5, "event_date": datetime(2026, 3, 2), "days_to_earnings": 5}]}
    db._derive_crush_signal_context = lambda **k: {"profile_source": "symbol", "confidence": 1, "magnitude": 1,
                                                   "edge_score": 1}
    db._score_setup_quality = lambda **k: 0.9
    db._rank_candidate_for_alpha = lambda **k: 0.9
    db._simulate_walk_forward_trade = lambda **k: pytest.fail("mismatched pair fell back to the proxy")
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture()
    db.logger.addHandler(handler)
    try:
        session = BacktestSession("s", "x", datetime(2026, 1, 1), datetime(2026, 4, 1), ["XYZ"], {},
                                  0, 0, 0, 0, 0, 0, datetime.now())
        assert db._run_walk_forward_backtest(session, {"pricing_mode": "hybrid"}) == []
    finally:
        db.logger.removeHandler(handler)
    assert db.replay_unpriceable == {UNPRICEABLE_MISMATCHED_CONTRACTS: 1}
    assert sum("⛔ UNPRICEABLE" in message for message in messages) == 1
    assert not any("📸 REPLAY" in message or "🔮 SYNTHETIC" in message for message in messages)


# ── 2. sessions come from the NYSE calendar ─────────────────────────────────


def test_after_close_before_a_holiday_reacts_after_it():
    # Thu 2026-07-02 after the close; Fri 07-03 is the observed Independence Day.
    assert post_event_valuation_days(date(2026, 7, 2), 0, "after market close") == 4


@pytest.mark.parametrize("as_of, timing, expected", [
    (date(2026, 11, 26), "before market open", 1),   # Thanksgiving BMO -> Friday
    (date(2026, 4, 2), "after market close", 4),     # before Good Friday -> Monday
    (date(2026, 1, 16), "after market close", 4),    # Friday before MLK Day -> Tuesday
    (date(2026, 7, 1), "after market close", 1),     # an ordinary Thursday session
])
def test_holidays_roll_the_reaction_session(as_of, timing, expected):
    assert post_event_valuation_days(as_of, 0, timing) == expected


# Published NYSE full-day closures.
_KNOWN = {
    2020: ["01-01", "01-20", "02-17", "04-10", "05-25", "07-03", "09-07", "11-26", "12-25"],
    2021: ["01-01", "01-18", "02-15", "04-02", "05-31", "07-05", "09-06", "11-25", "12-24"],
    2022: ["01-17", "02-21", "04-15", "05-30", "06-20", "07-04", "09-05", "11-24", "12-26"],
    2023: ["01-02", "01-16", "02-20", "04-07", "05-29", "06-19", "07-04", "09-04", "11-23", "12-25"],
    2024: ["01-01", "01-15", "02-19", "03-29", "05-27", "06-19", "07-04", "09-02", "11-28", "12-25"],
    2025: ["01-01", "01-09", "01-20", "02-17", "04-18", "05-26", "06-19", "07-04", "09-01", "11-27", "12-25"],
    2026: ["01-01", "01-19", "02-16", "04-03", "05-25", "06-19", "07-03", "09-07", "11-26", "12-25"],
    2027: ["01-01", "01-18", "02-15", "03-26", "05-31", "06-18", "07-05", "09-06", "11-25", "12-24"],
}


@pytest.mark.parametrize("year", sorted(_KNOWN))
def test_holidays_match_the_published_nyse_schedule(year):
    assert sorted(nyse_holidays(year)) == [date.fromisoformat(f"{year}-{md}") for md in _KNOWN[year]]


def test_session_resolution():
    assert not is_trading_day(date(2026, 7, 3)) and not is_trading_day(date(2026, 7, 4))
    assert next_session_on_or_after(date(2026, 7, 3)) == date(2026, 7, 6)
    assert next_session_on_or_after(date(2012, 10, 29)) == date(2012, 10, 31)   # Hurricane Sandy
    assert is_trading_day(date(2021, 12, 31))   # Saturday New Year's Day is not observed
