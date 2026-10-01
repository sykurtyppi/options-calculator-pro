"""Event provenance and entry semantics (post-#162 review).

1. Snapshots did not record where their EVENT date came from: live capture
   (allow_proxy_earnings by default) and the historical backfill (when only
   proxy dates were cached) stored snapshots of modelled proxy dates that
   replay, labels and the pairing report then treated as reported events.
2. Replay entries were chosen by relative day alone: the pre/post label was
   ignored and an after-close report's event-day "pre" close (relative day
   0, the last pre-reaction observation) could never be an entry.
3. A special dividend in a sparse history (one regular payment) set its own
   yardstick and was annualised as recurring.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import services.dividend_yields as dividend_yields
from services.institutional_ml_db import InstitutionalMLDatabase, SnapshotReplayPair
from tests.unit.test_services.test_replay_evidence_integrity import _FakeMDA, _recent_tuesday

NEAR = ("2026-03-06", "2026-03-13")
EVENT = datetime(2026, 3, 2)


def _row(capture, rel, phase, *, source="marketdata_app", iv=0.6, event="2026-03-02"):
    return ("XYZ", event, capture, rel, "AMC", phase, NEAR[0], NEAR[1], 200.0, iv, 0.5, 1.2, 205.0, "test",
            0.05, "stored", 0.004, "stored", capture, source)


def _db(tmp_path, rows):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "p.db"))
    with sqlite3.connect(db.db_path) as conn:
        conn.executemany(
            """INSERT INTO earnings_option_snapshots
               (symbol, event_date, capture_date, relative_day, release_timing, snapshot_phase, short_expiry,
                long_expiry, atm_strike, front_iv, back_iv, term_ratio, underlying_price, source,
                pricing_risk_free_rate, pricing_risk_free_rate_source, pricing_dividend_yield,
                pricing_dividend_yield_source, pricing_inputs_observed_on, event_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return db


# ── 1. provenance is recorded and proxy events are excluded ─────────────────


def _live_capture(tmp_path, event_source):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, *args, **kwargs):
            idx = pd.bdate_range(end=pd.Timestamp("2026-03-04"), periods=300)
            return pd.DataFrame({"Close": np.full(len(idx), 60.0)}, index=idx)

        options = ["2026-03-13", "2026-03-20"]

        def option_chain(self, expiry):
            return object()

    db = InstitutionalMLDatabase(db_path=str(tmp_path / "live.db"))
    db._utc_now = lambda: datetime(2026, 3, 4, 17, 0, tzinfo=timezone.utc)
    db._get_symbol_earnings_dates = lambda **_: [
        {"event_date": datetime(2026, 3, 10), "release_timing": "AMC", "source": event_source}]
    db._extract_atm_iv_from_chain = lambda chain, price: (0.40, 60.0)
    with mock.patch("yfinance.Ticker", FakeTicker), \
            mock.patch("services.pricing_rates.get_historical_risk_free_rate", return_value=(0.05, "stub")), \
            mock.patch("services.dividend_yields.get_historical_dividend_yield", return_value=(0.0, "stub")):
        db.capture_earnings_option_snapshots(symbols=["XYZ"])
    with sqlite3.connect(db.db_path) as conn:
        return [r[0] for r in conn.execute("SELECT event_source FROM earnings_option_snapshots")]


@pytest.mark.parametrize("source", ["proxy", "marketdata_app"])
def test_live_capture_records_the_event_source(tmp_path, source):
    assert _live_capture(tmp_path, source) == [source]


def _backfill_with_cached(tmp_path, cached):
    event = _recent_tuesday()
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "b.db"), mda_client=_FakeMDA(event))
    db._get_cached_earnings_dates = lambda symbol, window_start, window_end: [
        (datetime.combine(event, datetime.min.time()), source, "AMC") for source in cached]
    with mock.patch("services.pricing_rates.get_historical_risk_free_rate", return_value=(0.05, "stub")), \
            mock.patch("services.dividend_yields.get_historical_dividend_yield", return_value=(0.0, "stub")):
        db.capture_historical_iv_snapshots_mda(symbols=["XYZ"], lookback_years=1)
    with sqlite3.connect(db.db_path) as conn:
        return sorted({r[0] for r in conn.execute("SELECT event_source FROM earnings_option_snapshots")})


def test_backfill_of_a_proxy_only_event_is_recorded_as_proxy(tmp_path):
    assert _backfill_with_cached(tmp_path, ["proxy"]) == ["proxy"]


def test_backfill_keeps_the_reported_source(tmp_path):
    assert _backfill_with_cached(tmp_path, ["proxy", "marketdata_app"]) == ["marketdata_app"]


def test_cached_true_events_keep_their_own_source(tmp_path):
    db = InstitutionalMLDatabase(db_path=str(tmp_path / "c.db"))
    db._get_cached_earnings_dates = lambda *a, **k: [(datetime(2026, 3, 2), "marketdata_app", "AMC")]
    events = db._get_symbol_earnings_dates(
        symbol="XYZ", start_date=datetime(2026, 2, 1), end_date=datetime(2026, 4, 1),
        trading_dates=pd.Series(dtype="datetime64[ns]"), require_true_earnings=False, allow_proxy_earnings=True)
    assert [e["source"] for e in events] == ["marketdata_app"]


def test_proxy_event_snapshots_are_not_evidence_by_default(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", source="proxy"),
                        _row("2026-03-03", 1, "post", source="proxy", iv=0.4)])
    assert db._load_snapshot_replay_pair("XYZ", EVENT) is None
    assert isinstance(db._load_snapshot_replay_pair("XYZ", EVENT, include_proxy_events=True), SnapshotReplayPair)
    assert db.calibrate_earnings_iv_decay_labels().empty
    assert len(db.calibrate_earnings_iv_decay_labels(include_proxy_events=True)) == 1
    progress = db.summarize_snapshot_pairing_progress()
    assert progress["priceable_events"] == 0 and progress["proxy_events"] == 1
    assert db.summarize_snapshot_pairing_progress(include_proxy_events=True)["priceable_events"] == 1


def test_rebuilding_labels_drops_a_proxy_events_old_label(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", source="proxy"),
                        _row("2026-03-03", 1, "post", source="proxy", iv=0.4)])
    db.calibrate_earnings_iv_decay_labels(include_proxy_events=True)
    db.calibrate_earnings_iv_decay_labels()
    with sqlite3.connect(db.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM earnings_iv_decay_labels").fetchone()[0] == 0


def test_legacy_database_gains_the_column_and_keeps_its_rows(tmp_path):
    path = tmp_path / "legacy.db"
    InstitutionalMLDatabase(db_path=str(path))
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE earnings_option_snapshots DROP COLUMN event_source")
        conn.execute(
            """INSERT INTO earnings_option_snapshots
               (symbol, event_date, capture_date, relative_day, release_timing, snapshot_phase, short_expiry,
                long_expiry, atm_strike, front_iv, back_iv, term_ratio, underlying_price, source)
               VALUES ('XYZ', '2026-03-02', '2026-02-25', -5, 'AMC', 'pre', '2026-03-06', '2026-03-13',
                       200, 0.6, 0.5, 1.2, 205, 'legacy')""")
    InstitutionalMLDatabase(db_path=str(path))
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT symbol, event_source FROM earnings_option_snapshots").fetchall() == [("XYZ", None)]


# ── 2. entries are pre-reaction snapshots, including the event-day close ────


def test_an_after_close_event_day_close_is_an_entry(tmp_path):
    db = _db(tmp_path, [_row("2026-02-25", -5, "pre", iv=0.61), _row("2026-03-02", 0, "pre", iv=0.70),
                        _row("2026-03-03", 1, "post", iv=0.40)])
    pair = db._load_snapshot_replay_pair("XYZ", EVENT)
    assert pair.pre_capture_date.date() == date(2026, 3, 2) and pair.pre_front_iv == 0.70


def test_a_post_reaction_snapshot_is_never_an_entry(tmp_path):
    # Mislabelled history: a "post" row before the event must not be an entry.
    db = _db(tmp_path, [_row("2026-02-27", -3, "post", iv=0.30), _row("2026-03-03", 1, "post", iv=0.40)])
    assert db._load_snapshot_replay_pair("XYZ", EVENT) is None
    assert db.summarize_snapshot_pairing_progress()["pairable_events"] == 0


def test_a_before_open_event_day_snapshot_is_an_exit_not_an_entry(tmp_path):
    rows = [_row("2026-02-27", -3, "pre", iv=0.61), _row("2026-03-02", 0, "post", iv=0.35)]
    rows = [r[:4] + ("BMO",) + r[5:] for r in rows]
    db = _db(tmp_path, rows)
    pair = db._load_snapshot_replay_pair("XYZ", EVENT)
    assert pair.pre_capture_date.date() == date(2026, 2, 27)
    assert pair.post_capture_date.date() == date(2026, 3, 2)


# ── 3. specials in a sparse dividend history ─────────────────────────────────


def _series(rows):
    return pd.Series([v for _, v in rows],
                     index=pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") for d, _ in rows]), dtype=float)


def _q(dividends):
    ticker = mock.MagicMock()
    ticker.dividends = _series(dividends)
    ticker.splits = _series([])
    dividend_yields.reset_cache()
    with mock.patch("yfinance.Ticker", return_value=ticker):
        return dividend_yields.get_historical_dividend_yield("XYZ", date(2024, 3, 1), 100.0)[0]


def test_a_special_in_a_sparse_history_is_excluded():
    assert _q([("2023-11-15", 0.5), ("2024-01-15", 15.0)]) == pytest.approx(0.005)


def test_a_repeated_dividend_raise_is_kept():
    q = _q([("2023-04-15", 0.5), ("2023-07-15", 0.5), ("2023-10-15", 1.2), ("2024-01-15", 1.2)])
    assert q == pytest.approx(3.4 / 100.0)
