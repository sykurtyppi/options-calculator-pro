"""Regressions for the post-#159 deep audit (replay and evidence integrity).

1. proxy (synthetic) trades were seeded into the learning stores as "replay";
2. the historical backfill could store a pre-event chain as "post" (and a
   post-event chain as "pre") when price history ended or started near the
   event, with a fixed relative_day hiding it;
3. a split between a dividend's ex-date and as_of overstated the yield;
4. replay could price the entry before the date that decided the trade;
5. IV-crush labels were still built from mismatched contracts;
6. crush profiles used post-event IVs observed after as_of;
7. the loader picked an unpriceable pair when a priceable one existed;
8. routing counts were parsed from logs (skips missed, debit-gated trades
   counted as replays);
plus: special dividends, adjusted-price fallback, entry strike look-ahead,
missing expiries counted as priceable, and the seed script's percentage.
"""
from __future__ import annotations

import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

import services.dividend_yields as dividend_yields
import services.pricing_rates as pricing_rates
from services.institutional_ml_db import (
    PRICING_SOURCE_SNAPSHOT_REPLAY,
    PRICING_SOURCE_SYNTHETIC_PROXY,
    UNPRICEABLE_MISMATCHED_CONTRACTS,
    UNPRICEABLE_MISSING_ENTRY_STRIKE,
    UNPRICEABLE_OUTSIDE_TRADE_WINDOW,
    BacktestSession,
    BacktestTrade,
    InstitutionalMLDatabase,
    SnapshotReplayPair,
    SnapshotReplayRefusal,
)


@pytest.fixture(autouse=True)
def _fresh_caches():
    pricing_rates.reset_cache()
    dividend_yields.reset_cache()
    yield
    pricing_rates.reset_cache()
    dividend_yields.reset_cache()


NEAR = ("2026-03-06", "2026-03-13")
FAR = ("2026-04-17", "2026-05-15")
EVENT = datetime(2026, 3, 2)


def _row(capture, rel, phase, expiries, *, iv=0.6, priced=True, strike=200.0):
    inputs = (0.05, "stored", 0.004, "stored", capture) if priced else (None, None, None, None, None)
    return ("XYZ", "2026-03-02", capture, rel, "AMC", phase, expiries[0], expiries[1], strike, iv, 0.5,
            1.2, 205.0, "test") + inputs


def _db(tmp_path, rows=()):
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


def _wire(db, trade_date, days_to_earnings, symbols=("XYZ",), hold_days=5):
    rows = {s: SimpleNamespace(symbol=s, iv30_rv30_ratio=1.2, price_momentum_5d=0.0, volume=5e6,
                               volume_ratio_10d=1.0) for s in symbols}
    db._load_walk_forward_dataset = lambda *a, **k: pd.DataFrame({"date": [pd.Timestamp(trade_date)], "symbol": ["XYZ"]})
    db._load_iv_crush_profiles = lambda *a, **k: {}
    db._build_earnings_event_candidates = lambda **k: {pd.Timestamp(trade_date): [
        {"row": rows[s], "hold_days": hold_days, "event_date": EVENT, "days_to_earnings": days_to_earnings}
        for s in symbols]}
    db._derive_crush_signal_context = lambda **k: {"profile_source": "symbol", "confidence": 1, "magnitude": 1,
                                                   "edge_score": 1}
    db._score_setup_quality = lambda **k: 0.9
    db._rank_candidate_for_alpha = lambda **k: 0.9


def _session():
    return BacktestSession("s", "x", datetime(2026, 1, 1), datetime(2026, 4, 1), ["XYZ"], {},
                           0, 0, 0, 0, 0, 0, datetime.now())


def _proxy_trade(**kwargs):
    return BacktestTrade(
        session_id=kwargs["session_id"], symbol="XYZ", trade_date=datetime(2026, 2, 20), event_date=EVENT,
        days_to_earnings=10, contracts=1, hold_days=5, setup_score=0.9, debit_per_contract=200.0,
        transaction_cost_per_contract=5.0, gross_return_pct=0.30, net_return_pct=0.275, pnl_per_contract=55.0,
        underlying_return=0.0, expected_move=0.05, move_ratio=0.0, predicted_front_iv_crush_pct=-0.2,
        crush_confidence=1, crush_edge_score=1, crush_profile_sample_size=1, execution_profile="institutional",
        pricing_source=PRICING_SOURCE_SYNTHETIC_PROXY,
    )


# ── 1. only snapshot-priced trades are seeded ───────────────────────────────


def _seed(tmp_path, trades):
    from scripts.seed_outcomes_from_replay import seed_from_trades

    return seed_from_trades(
        trades, structure="call_calendar", dry_run=False, outcome_store_path=tmp_path / "o.sqlite",
        calibration_store_path=tmp_path / "c.json", prior_store_path=tmp_path / "p.json",
    )


def test_proxy_trades_are_stored_as_proxy_and_never_seeded(tmp_path):
    from scripts.seed_outcomes_from_replay import _fetch_trades

    db = _db(tmp_path)
    _wire(db, "2026-02-20", 10)
    db._simulate_walk_forward_trade = _proxy_trade
    session_id = db.run_calendar_spread_backtest(
        {"pricing_mode": "hybrid", "start_date": "2026-01-01", "end_date": "2026-04-01", "universe": ["XYZ"]})
    trades = _fetch_trades(Path(db.db_path), session_id, None)
    assert [t["pricing_source"] for t in trades] == [PRICING_SOURCE_SYNTHETIC_PROXY]

    result = _seed(tmp_path, trades)
    assert result["inserted"] == 0 and result["skipped_not_snapshot_priced"] == 1
    with sqlite3.connect(tmp_path / "o.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM outcome_trades").fetchone()[0] == 0


def test_the_proxy_simulator_labels_its_trades_synthetic(tmp_path):
    db = _db(tmp_path)
    _wire(db, "2026-02-20", 10)
    db._build_earnings_event_candidates = lambda **k: {pd.Timestamp("2026-02-20"): [{
        "row": SimpleNamespace(symbol="XYZ", iv30_rv30_ratio=1.2, price_momentum_5d=0.0, volume=5e6,
                               volume_ratio_10d=1.0, underlying_price=100.0, realized_vol_30d=0.25,
                               forward_return_5d=0.01),
        "hold_days": 5, "event_date": EVENT, "days_to_earnings": 10}]}
    trades = db._run_walk_forward_backtest(_session(), {"pricing_mode": "hybrid"})
    assert [t.pricing_source for t in trades] == [PRICING_SOURCE_SYNTHETIC_PROXY]
    assert db.replay_routing[PRICING_SOURCE_SYNTHETIC_PROXY] == 1


def _trade_row(pricing_source):
    row = {"symbol": "XYZ", "trade_date": "2026-02-25", "event_date": "2026-03-02", "days_to_earnings": 5,
           "setup_score": 0.8, "gross_return_pct": 0.2, "net_return_pct": 0.18, "pnl_per_contract": 30.0,
           "execution_profile": "institutional", "structure": "call_calendar"}
    if pricing_source is not None:
        row["pricing_source"] = pricing_source
    return row


def test_only_snapshot_replay_rows_are_seeded_and_legacy_rows_are_refused(tmp_path):
    result = _seed(tmp_path, [_trade_row(PRICING_SOURCE_SNAPSHOT_REPLAY), _trade_row(None),
                              _trade_row(PRICING_SOURCE_SYNTHETIC_PROXY)])
    assert result["inserted"] == 1 and result["skipped_not_snapshot_priced"] == 2


def test_replayed_trade_is_labelled_snapshot_replay(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", NEAR), _row("2026-03-03", 1, "post", NEAR, iv=0.4)])
    _wire(db, "2026-02-20", 10)
    trades = db._run_walk_forward_backtest(_session(), {"pricing_mode": "snapshot_replay"})
    assert [t.pricing_source for t in trades] == [PRICING_SOURCE_SNAPSHOT_REPLAY]


# ── 2. backfill never crosses the event ─────────────────────────────────────


class _FakeMDA:
    def __init__(self, event, underlying_price=100.0):
        self.event = event
        self.underlying_price = underlying_price
        self.requested = []

    def is_available(self):
        return True

    def get_option_chain(self, symbol, expiration, strike_limit, date):
        self.requested.append(date)
        rows = []
        for exp in [(self.event + timedelta(days=17)).isoformat(), (self.event + timedelta(days=45)).isoformat()]:
            for strike in (95.0, 100.0, 105.0):
                rows.append(dict(expiration_date=exp, strike=strike, side="call", impliedVolatility=0.6,
                                 underlyingPrice=self.underlying_price))
        return pd.DataFrame(rows)


def _backfill(tmp_path, event, price_days, *, underlying_price=100.0):
    mda = _FakeMDA(event, underlying_price)
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "b.db"), mda_client=mda)
    db._get_cached_earnings_dates = lambda symbol, window_start, window_end: [
        (datetime.combine(event, datetime.min.time()), "marketdata_app", "AMC")]
    with sqlite3.connect(db.db_path) as conn:
        conn.executemany(
            "INSERT INTO daily_prices(symbol,date,open_price,high_price,low_price,close_price,volume,adj_close) "
            "VALUES('XYZ',?,10,10,10,10,1,10)",
            [(d.isoformat(),) for d in price_days if d.weekday() < 5],
        )
    with mock.patch("services.pricing_rates.get_historical_risk_free_rate", return_value=(0.05, "stub")), \
            mock.patch("services.dividend_yields.get_historical_dividend_yield", return_value=(0.0, "stub")):
        db.capture_historical_iv_snapshots_mda(symbols=["XYZ"], lookback_years=1)
    with sqlite3.connect(db.db_path) as conn:
        stored = conn.execute(
            "SELECT capture_date, relative_day, snapshot_phase FROM earnings_option_snapshots ORDER BY capture_date"
        ).fetchall()
    return stored, mda


def _recent_tuesday(weekday=1):
    event = date.today() - timedelta(days=60)
    while event.weekday() != weekday:
        event -= timedelta(days=1)
    return event


def test_backfill_stores_the_day_actually_captured(tmp_path):
    # Thursday after the close: the T-5 target is a Saturday, captured on the
    # Friday before (T-6); the post snapshot is Friday's session (T+1).
    event = _recent_tuesday(weekday=3)
    stored, _ = _backfill(tmp_path, event, [event - timedelta(days=i) for i in range(-40, 40)])
    assert sorted((rel, phase) for _, rel, phase in stored) == [(-6, "pre"), (1, "post")]


@pytest.mark.parametrize("history", ["ends_before_event", "starts_after_event"])
def test_backfill_snapshots_stay_on_their_side_of_the_event(tmp_path, history):
    event = _recent_tuesday()
    if history == "ends_before_event":
        days = [event - timedelta(days=i) for i in range(3, 40)]
    else:
        days = [event + timedelta(days=i) for i in range(2, 40)]
    stored, _ = _backfill(tmp_path, event, days)
    by_phase = {phase: (date.fromisoformat(capture), rel) for capture, rel, phase in stored}
    pre_day, pre_rel = by_phase["pre"]
    post_day, post_rel = by_phase["post"]
    assert pre_day < event < post_day   # an after-close report: post is the next session
    assert pre_rel == (pre_day - event).days and post_rel == (post_day - event).days


def test_backfill_refuses_the_adjusted_close_when_the_chain_has_no_price(tmp_path):
    event = _recent_tuesday()
    stored, mda = _backfill(tmp_path, event, [event - timedelta(days=i) for i in range(-40, 40)],
                            underlying_price=float("nan"))
    assert mda.requested and stored == []


# ── 3. dividends: split basis, specials, cache refresh ──────────────────────


def _series(rows):
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day, _ in rows]),
        dtype=float,
    )


def _q(dividends, splits, as_of, price):
    ticker = mock.MagicMock()
    ticker.dividends = _series(dividends)
    ticker.splits = _series(splits)
    dividend_yields.reset_cache()
    with mock.patch("yfinance.Ticker", return_value=ticker):
        return dividend_yields.get_historical_dividend_yield("XYZ", as_of, price)


def test_split_between_ex_date_and_as_of_does_not_inflate_the_yield():
    # 4:1 split on 2020-08-31; yfinance amounts are in today's (post-split) basis.
    dividends = [("2019-11-07", 0.1925), ("2020-02-07", 0.1925), ("2020-05-08", 0.205), ("2020-08-07", 0.205)]
    q, _ = _q(dividends, [("2020-08-31", 4.0)], date(2020, 10, 1), 116.79)
    assert q == pytest.approx(sum(a for _, a in dividends) / 116.79)


def test_split_after_as_of_is_undone():
    dividends = [("2023-03-01", 0.01), ("2023-06-01", 0.01), ("2023-09-01", 0.01), ("2023-12-01", 0.01)]
    q, _ = _q(dividends, [("2024-06-10", 10.0)], date(2024, 1, 15), 500.0)
    assert q == pytest.approx(0.4 / 500.0)


def test_special_dividend_is_not_annualised():
    regular = [("2023-02-09", 0.90), ("2023-04-27", 0.90), ("2023-07-27", 1.02), ("2023-10-26", 1.02)]
    q, _ = _q(regular + [("2023-12-27", 15.0)], [], date(2024, 3, 1), 700.0)
    assert q == pytest.approx(3.84 / 700.0)


def test_dividend_history_cache_refreshes_after_a_day():
    first = mock.MagicMock()
    first.dividends = _series([])
    first.splits = _series([])
    later = mock.MagicMock()
    later.dividends = _series([("2024-01-10", 0.5)])
    later.splits = _series([])
    dividend_yields.reset_cache()
    with mock.patch("yfinance.Ticker", side_effect=[first, later]), mock.patch("time.time") as clock:
        clock.return_value = 1_000_000.0
        assert dividend_yields.get_historical_dividend_yield("XYZ", date(2024, 1, 11), 100.0)[0] == 0.0
        clock.return_value += dividend_yields.DEFAULT_CACHE_TTL_SECONDS + 1
        assert dividend_yields.get_historical_dividend_yield("XYZ", date(2024, 1, 11), 100.0)[0] == pytest.approx(0.005)


# ── 4. / 7. pairing: decision date, hold window, priceable first, strike ─────


def test_entry_snapshot_before_the_decision_date_is_refused(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", NEAR), _row("2026-03-03", 1, "post", NEAR, iv=0.4)])
    result = db._load_snapshot_replay_pair("XYZ", EVENT, decided_on=datetime(2026, 2, 27))
    assert isinstance(result, SnapshotReplayRefusal) and result.reason == UNPRICEABLE_OUTSIDE_TRADE_WINDOW

    _wire(db, "2026-02-27", 3)
    assert db._run_walk_forward_backtest(_session(), {"pricing_mode": "snapshot_replay"}) == []
    assert db.replay_unpriceable == {UNPRICEABLE_OUTSIDE_TRADE_WINDOW: 1}


def test_exit_after_the_hold_window_is_refused(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", NEAR), _row("2026-03-06", 4, "post", NEAR, iv=0.4)])
    # Entry Wed 02-25 + 5 sessions = Wed 03-04; the only exit is Fri 03-06.
    result = db._load_snapshot_replay_pair("XYZ", EVENT, decided_on=datetime(2026, 2, 25), max_hold_sessions=5)
    assert isinstance(result, SnapshotReplayRefusal) and result.reason == UNPRICEABLE_OUTSIDE_TRADE_WINDOW


def test_entry_on_or_after_the_decision_date_is_priced(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", NEAR), _row("2026-03-03", 1, "post", NEAR, iv=0.4)])
    _wire(db, "2026-02-23", 7)
    trades = db._run_walk_forward_backtest(_session(), {"pricing_mode": "snapshot_replay"})
    assert len(trades) == 1 and trades[0].trade_date.date() == date(2026, 2, 25)


def test_a_priceable_pair_beats_a_later_unpriceable_one(tmp_path):
    db = _db(tmp_path, [
        _row("2026-02-25", -5, "pre", FAR), _row("2026-03-03", 1, "post", FAR, priced=False),
        _row("2026-02-24", -6, "pre", NEAR), _row("2026-03-04", 2, "post", NEAR),
    ])
    pair = db._load_snapshot_replay_pair("XYZ", EVENT)
    assert isinstance(pair, SnapshotReplayPair)
    assert (pair.short_expiry, pair.long_expiry) == NEAR and pair.post_risk_free_rate == 0.05
    assert db.summarize_snapshot_pairing_progress()["priceable_events"] == 1


def test_entry_strike_is_never_taken_from_the_post_event_snapshot(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", NEAR, strike=None),
                        _row("2026-03-03", 1, "post", NEAR, strike=240.0)])
    result = db._load_snapshot_replay_pair("XYZ", EVENT)
    assert isinstance(result, SnapshotReplayRefusal) and result.reason == UNPRICEABLE_MISSING_ENTRY_STRIKE
    assert db.summarize_snapshot_pairing_progress()["priceable_events"] == 0


def test_missing_expiries_are_neither_paired_nor_counted_priceable(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", (None, None)), _row("2026-03-03", 1, "post", (None, None))])
    result = db._load_snapshot_replay_pair("XYZ", EVENT)
    assert isinstance(result, SnapshotReplayRefusal) and result.reason == UNPRICEABLE_MISMATCHED_CONTRACTS
    assert db.summarize_snapshot_pairing_progress()["priceable_events"] == 0


# ── 5. / 6. crush labels and profiles ───────────────────────────────────────


def test_crush_labels_use_same_contract_pairs_only(tmp_path):
    db = _db(tmp_path, [_row("2026-02-27", -3, "pre", NEAR), _row("2026-03-03", 1, "post", FAR, iv=0.3)])
    assert db.calibrate_earnings_iv_decay_labels().empty
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM earnings_iv_decay_labels").fetchone()[0] == 0


def test_rebuilding_labels_drops_a_stale_mismatched_label(tmp_path):
    db = _db(tmp_path, [_row("2026-02-27", -3, "pre", NEAR), _row("2026-03-03", 1, "post", FAR, iv=0.3)])
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            """INSERT INTO earnings_iv_decay_labels
               (symbol, event_date, release_timing, pre_capture_date, post_capture_date, pre_front_iv,
                post_front_iv, pre_back_iv, post_back_iv, front_iv_crush_pct, back_iv_crush_pct,
                term_ratio_change, underlying_move_pct, quality_score, source)
               VALUES ('XYZ', '2026-03-02', 'AMC', '2026-02-27', '2026-03-03', 0.6, 0.3, 0.5, 0.5,
                       -0.5, 0.0, 0.0, 0.0, 0.6, 'snapshot_pair')""")
    db.calibrate_earnings_iv_decay_labels()
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM earnings_iv_decay_labels").fetchone()[0] == 0


def test_same_contract_label_uses_an_earlier_matching_entry(tmp_path):
    db = _db(tmp_path, [_row("2026-02-24", -6, "pre", FAR, iv=0.62), _row("2026-02-27", -3, "pre", NEAR),
                        _row("2026-03-03", 1, "post", FAR, iv=0.31)])
    labels = db.calibrate_earnings_iv_decay_labels()
    assert labels["pre_capture_date"].tolist() == ["2026-02-24"]
    assert labels["front_iv_crush_pct"].tolist() == [pytest.approx((0.31 - 0.62) / 0.62)]


def test_crush_profile_waits_for_the_post_event_observation(tmp_path):
    db = _db(tmp_path, [_row("2026-02-27", -3, "pre", NEAR), _row("2026-03-06", 4, "post", NEAR, iv=0.3)])
    db.calibrate_earnings_iv_decay_labels()
    before = db._load_iv_crush_profiles(["XYZ"], as_of_date="2026-03-04")
    after = db._load_iv_crush_profiles(["XYZ"], as_of_date="2026-03-09")
    assert "XYZ" not in before and after["XYZ"]["sample_size"] == 1


# ── 8. routing counts come from the backtest ────────────────────────────────


def test_routing_counts_recorded_trades_after_the_debit_gate(tmp_path):
    import scripts.run_replay_backtest as script

    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", NEAR), _row("2026-03-03", 1, "post", NEAR, iv=0.4)])
    _wire(db, "2026-02-20", 10, symbols=("XYZ", "ABC"))
    trades = db._run_walk_forward_backtest(
        _session(), {"pricing_mode": "snapshot_replay", "max_entry_debit_per_contract": 1.0})
    assert trades == []
    assert script.routing_summary(db) == {
        "replay": 0, "synthetic": 0, "skipped": 1, "unpriceable": 0, "debit_gated": 1}


def test_seed_script_prints_the_pairing_percentage(capsys):
    import scripts.seed_replay_snapshots as script

    script._print_pairing_report({"total_snapshots": 20, "total_events": 10, "pairable_events": 10,
                                  "pairable_event_pct": 1.0, "priceable_events": 10})
    out = capsys.readouterr().out
    assert "(100.0%)" in out and "Only" not in out and "100% of events are paired" in out
