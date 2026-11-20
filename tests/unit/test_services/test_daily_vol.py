"""Ordinary daily variance during the hold (services/daily_vol.py) and its place in the harness."""
from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import scripts.run_pricing_harness as cli
from services import daily_vol as dv
from services import pricing_harness as ph
from services.event_variance import sessions_between
from services.market_calendar import is_trading_day
from services.realized_vol import har_rv_forecast
from tests.unit.test_services.pricing_harness_fixtures import BASE_VOL, SyntheticArchive, quarterly_events


def _frame(closes, start=date(2024, 1, 2)):
    days, day = [], start
    while len(days) < len(closes):
        if is_trading_day(day):
            days.append(day)
        day += timedelta(days=1)
    return pd.DataFrame({"trade_date": pd.to_datetime(days), "open": closes, "high": closes,
                         "low": closes, "close": closes, "volume": 1})


# ── realized ────────────────────────────────────────────────────────────────

def test_close_to_close_returns_are_log_returns_by_session() -> None:
    frame = _frame([100.0, 110.0, 99.0, 0.0, 99.0])
    r = dv.close_to_close_returns(frame)
    assert list(r.round(10)) == [round(math.log(1.1), 10), round(math.log(0.9), 10), 0.0]
    assert r.index[0] == pd.Timestamp("2024-01-03")


def test_the_hold_variance_is_the_sessions_after_entry_through_the_exit() -> None:
    frame = _frame([100.0, 101.0, 99.0, 102.0, 150.0])  # 01-02 .. 01-08
    entry, exit_day = date(2024, 1, 3), date(2024, 1, 5)
    held = sessions_between(entry, exit_day)
    value, reason = dv.realized_hold_variance(frame, entry, exit_day, held)
    expected = np.mean([math.log(99 / 101) ** 2, math.log(102 / 99) ** 2])
    assert reason is None and held == 2 and value == pytest.approx(expected)


@pytest.mark.parametrize("frame, expected_sessions, reason", [
    (_frame([100.0, 101.0, 99.0]), 2, "hold_closes_missing"),
    (_frame([100.0, 100.0, 100.0, 100.0]), 2, "zero_realized_variance"),
    (_frame([100.0, 101.0, 99.0, 102.0]), 0, "no_sessions_held"),
    (pd.DataFrame(columns=["trade_date", "close"]), 2, "no_closes"),
])
def test_an_unmeasurable_hold_says_why(frame, expected_sessions, reason) -> None:
    value, why = dv.realized_hold_variance(frame, date(2024, 1, 3), date(2024, 1, 5), expected_sessions)
    assert value is None and why == reason


def test_a_gap_in_the_hold_is_not_averaged_over() -> None:
    frame = _frame([100.0, 101.0, 99.0, 102.0]).drop(index=2)  # 01-04 missing
    value, reason = dv.realized_hold_variance(frame, date(2024, 1, 3), date(2024, 1, 5), 2)
    assert value is None and reason == "hold_closes_missing"


# ── forecasts ───────────────────────────────────────────────────────────────

def _random_path(n=260, seed=1, jump_on=None):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, n)
    frame = _frame(list(100.0 * np.exp(np.cumsum(rets))), start=date(2023, 1, 3))
    if jump_on is not None:
        i = frame.index[frame["trade_date"] == pd.Timestamp(jump_on)][0]
        frame.loc[i:, ["open", "high", "low", "close"]] *= 1.25
    return frame


def test_close_to_close_har_is_production_har_on_squared_returns() -> None:
    frame = _random_path()
    r = dv.close_to_close_returns(frame)
    expected = har_rv_forecast(np.sqrt(r ** 2 * 252)) ** 2 / 252
    assert dv.har_close_to_close(frame, []) == pytest.approx(expected)


def test_earlier_earnings_reactions_are_kept_out_of_the_baseline() -> None:
    reaction = date(2023, 8, 3)  # a Thursday reaction to a Wednesday after-close report
    clean = _random_path()
    jumped = _random_path(jump_on=reaction)
    events = [{"event_date": "2023-08-02", "release_timing": "AMC"}]
    assert dv.har_close_to_close(jumped, events) == pytest.approx(dv.har_close_to_close(clean, events))
    assert dv.har_close_to_close(jumped, []) != pytest.approx(dv.har_close_to_close(clean, []))


def test_close_to_close_har_needs_enough_history() -> None:
    assert dv.har_close_to_close(_random_path(n=90), []) is None
    assert dv.har_close_to_close(pd.DataFrame(), []) is None


@pytest.mark.parametrize("vol, expected", [(0.30, 0.09 / 252), (None, None), (0.0, None), (float("nan"), None),
                                           ("x", None)])
def test_annual_vol_to_per_session_variance(vol, expected) -> None:
    got = dv.per_session(vol)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_qlike_is_zero_at_the_truth() -> None:
    assert dv.qlike(2.0, 2.0) == 0.0
    assert dv.qlike(2.0, 1.0) == pytest.approx(2.0 - math.log(2.0) - 1.0)


def test_qlike_ranks_the_true_variance_first_where_a_log_loss_does_not() -> None:
    # Five-session holds of a known variance: the realized target is noisy and skewed.
    rng = np.random.default_rng(11)
    true = 0.0004
    realized = (rng.normal(0.0, math.sqrt(true), (4000, 5)) ** 2).mean(axis=1)
    low = 0.7 * true
    q_true = np.mean([dv.qlike(r, true) for r in realized])
    q_low = np.mean([dv.qlike(r, low) for r in realized])
    log_true = np.mean(np.log(realized / true) ** 2)
    log_low = np.mean(np.log(realized / low) ** 2)
    assert q_true < q_low
    assert log_low < log_true  # why the report has no log loss


# ── calibration ─────────────────────────────────────────────────────────────

def test_an_empty_calibrator_leaves_the_forecast_alone() -> None:
    assert dv.DailyVarianceCalibrator().calibration("AAA", date(2024, 3, 1)) == dv.Calibration(
        1.0, 1.0, 1.0, 1.0, 0, 0, 0)


def test_the_calibration_knows_only_exits_before_its_date() -> None:
    c = dv.DailyVarianceCalibrator()
    c.add("AAA", date(2024, 2, 8), date(2024, 2, 7), 2.0, 1.0)
    c.add("AAA", date(2024, 2, 9), date(2024, 2, 8), float("nan"), 1.0)
    c.add("AAA", date(2024, 2, 9), date(2024, 2, 8), 1.0, 0.0)
    assert c.calibration("AAA", date(2024, 2, 7)).n_pooled == 0
    assert c.calibration("AAA", date(2024, 2, 8)).n_pooled == 1
    assert c.calibration("AAA", date(2024, 3, 1)).n_pooled == 1


def test_the_scale_is_a_ratio_of_sums_with_shrunk_group_and_symbol_factors() -> None:
    k = 4.0
    c = dv.DailyVarianceCalibrator(shrinkage_events=k, group_of={"AAA": "T", "BBB": "T", "CCC": "F"}.get)
    x = date(2024, 1, 10)
    c.add("AAA", date(2024, 1, 11), x, 3.0, 1.0)   # event 1, two holds
    c.add("AAA", date(2024, 1, 11), x, 1.0, 1.0)
    c.add("AAA", date(2024, 1, 12), x, 2.0, 2.0)   # event 2
    c.add("BBB", date(2024, 1, 13), x, 1.0, 1.0)
    c.add("CCC", date(2024, 1, 14), x, 0.5, 1.0)
    out = c.calibration("AAA", date(2024, 2, 1))
    pooled = (3 + 1 + 2 + 1 + 0.5) / (1 + 1 + 2 + 1 + 1)
    group_raw = np.mean([4.0 / (2 * pooled), 2.0 / (2 * pooled), 1.0 / pooled])
    group = 1 + (group_raw - 1) * 3 / (3 + k)
    symbol_raw = np.mean([4.0 / (2 * pooled * group), 2.0 / (2 * pooled * group)])
    symbol = 1 + (symbol_raw - 1) * 2 / (2 + k)
    assert out.pooled == pytest.approx(pooled)
    assert (out.group, out.n_group_events) == (pytest.approx(group), 3)
    assert (out.symbol, out.n_symbol_events) == (pytest.approx(symbol), 2)
    assert out.factor == pytest.approx(pooled * group * symbol)
    loner = c.calibration("ZZZ", date(2024, 2, 1))
    assert (loner.group, loner.symbol, loner.factor) == (1.0, 1.0, pytest.approx(pooled))


# ── quality ─────────────────────────────────────────────────────────────────

def _q(realized, implied, rs, cc, cal):
    return {"daily_realized_variance": realized, "daily_forecast_implied": implied, "daily_forecast_har_rs": rs,
            "daily_forecast_har_cc": cc, "daily_forecast_calibrated": cal}


def test_forecast_quality_compares_on_common_rows() -> None:
    rows = [_q(2.0, 1.0, 1.0, 2.0, 1.5), _q(1.0, 1.0, 0.5, 1.0, 1.0), _q(3.0, 2.5, 1.0, 3.0, 2.0),
            _q(1.0, 1.0, None, 1.0, 1.0), _q(None, 1.0, 1.0, 1.0, 1.0)]
    out = dv.forecast_quality(rows)
    assert out["n"] == 3 and out["rows_with_realized"] == 4
    implied, cc = out["forecasts"]["implied"], out["forecasts"]["har_cc"]
    q_implied = np.mean([dv.qlike(2, 1), dv.qlike(1, 1), dv.qlike(3, 2.5)])
    assert implied["qlike"] == pytest.approx(q_implied, abs=1e-5)
    assert implied["level_ratio"] == pytest.approx(6 / 4.5, abs=1e-5)  # a ratio of means, not a mean of ratios
    assert cc["qlike"] == 0.0 and cc["qlike_skill_vs_implied"] == 1.0 and cc["mse_skill_vs_implied"] == 1.0
    assert cc["level_ratio"] == 1.0
    assert "qlike_skill_vs_implied" not in implied
    rs = out["forecasts"]["har_rs"]
    mse_rs, mse_implied = np.mean([1.0, 0.25, 4.0]), np.mean([1.0, 0.0, 0.25])
    assert rs["mse_skill_vs_implied"] == pytest.approx(1 - mse_rs / mse_implied, abs=1e-5)
    # Against the market's level, not raw: forecast/implied 1, .5, .4 vs realized/implied 2, 1, 1.2.
    assert rs["spread_spearman"] == pytest.approx(ph.spearman([1.0, 0.5, 0.4], [2.0, 1.0, 1.2]), abs=1e-5)
    assert rs["spread_spearman"] != pytest.approx(ph.spearman([1.0, 0.5, 1.0], [2.0, 1.0, 3.0]), abs=1e-3)
    assert dv.forecast_quality([]) == {"n": 0, "rows_with_realized": 0}


# ── in the harness ──────────────────────────────────────────────────────────

SYMBOLS = ("AAA", "BBB")
EVENTS = {"AAA": quarterly_events("AAA", [date(2024, 2, 8), date(2024, 5, 2)]),
          "BBB": quarterly_events("BBB", [date(2024, 2, 6), date(2024, 4, 30)])}


@pytest.fixture(scope="module")
def replay():
    archive = SyntheticArchive(EVENTS, start=date(2023, 3, 1))
    snapshots = {}

    def recording(symbol, as_of, **kwargs):
        snap = ph.build_vol_snapshot(symbol, as_of, **kwargs)
        snapshots[(symbol, as_of)] = snap
        return snap

    result = ph.run_harness(archive, list(SYMBOLS), start=date(2024, 1, 1), end=date(2024, 6, 30),
                            prior_mode=ph.PRIOR_NEUTRAL, snapshot_builder=recording)
    return archive, result, snapshots


def test_every_entry_carries_the_four_forecasts_and_what_it_realized(replay) -> None:
    archive, result, snapshots = replay
    assert result.variance_rows and result.counts["daily:measured"] == len(result.variance_rows)
    for row in result.variance_rows:
        entry, exit_ = date.fromisoformat(row["entry_date"]), date.fromisoformat(row["exit_date"])
        prices = archive.prices(row["symbol"], exit_)
        expected, _ = dv.realized_hold_variance(prices, entry, exit_, row["sessions_held"])
        assert row["daily_realized_variance"] == pytest.approx(expected)
        assert row["daily_forecast_implied"] == pytest.approx(row["entry_base_variance_per_session"])
        assert row["daily_forecast_implied"] == pytest.approx(BASE_VOL ** 2 / 252)
        snap = snapshots[(row["symbol"], entry)]
        assert row["daily_forecast_har_rs"] == pytest.approx(snap.rv_har_forecast ** 2 / 252)
        assert row["daily_forecast_har_cc"] == pytest.approx(dv.har_close_to_close(
            archive.prices(row["symbol"], entry), ph._earnings_metadata(
                next(e for e in EVENTS[row["symbol"]] if e.event_date.isoformat() == row["event_date"]),
                EVENTS[row["symbol"]], entry)["prior_events"]))
        assert row["daily_forecast_calibrated"] == pytest.approx(
            row["daily_forecast_har_cc"] * row["daily_calibration_factor"])


def test_the_calibration_only_knows_holds_that_exited_before_the_entry(replay) -> None:
    _, result, _ = replay
    measured = [r["exit_date"] for r in result.variance_rows if r["daily_realized_variance"] is not None]
    for row in result.variance_rows:
        assert row["daily_calibration_n_pooled"] == sum(1 for x in measured if x < row["entry_date"])
    assert any(r["daily_calibration_n_pooled"] > 0 for r in result.variance_rows)
    assert any(r["entry_date"] < x < r["exit_date"] for r in result.variance_rows for x in measured)


def test_the_summary_ranks_the_forecasts(replay) -> None:
    _, result, _ = replay
    section = ph.summarize(result)["daily_variance"]["all_entries"]
    assert section["n"] == len(result.variance_rows)
    forecasts = section["forecasts"]
    assert set(forecasts) == set(dv.FORECASTS)
    # The fixture's options price the true variance; open-to-close HAR misses the moves between closes.
    assert forecasts["har_rs"]["level_ratio"] > 1.3
    assert forecasts["har_rs"]["qlike"] > forecasts["implied"]["qlike"]


def test_an_unmeasured_hold_is_counted_and_never_calibrates() -> None:
    class Gappy(SyntheticArchive):
        def prices(self, symbol, through):
            frame = super().prices(symbol, through)
            return frame[frame["trade_date"] != pd.Timestamp("2024-02-06")]

    result = ph.run_harness(Gappy({"AAA": EVENTS["AAA"][:1]}, start=date(2023, 3, 1)), ["AAA"],
                            start=date(2024, 1, 1), end=date(2024, 6, 30), prior_mode=ph.PRIOR_NEUTRAL)
    rows = result.variance_rows
    assert rows and all(r["daily_unmeasured_reason"] == "hold_closes_missing" for r in rows
                        if r["entry_date"] < "2024-02-06")
    assert all(r["daily_calibration_n_pooled"] == 0 for r in rows)


def test_the_report_has_the_daily_section(tmp_path, monkeypatch) -> None:
    archive = SyntheticArchive({"AAA": EVENTS["AAA"][:1]}, start=date(2023, 3, 1))
    seen = []

    def fake_run(_archive, symbols, **kwargs):
        seen.append(kwargs["daily_calibrator"])
        return ph.run_harness(archive, symbols, start=kwargs["start"], end=kwargs["end"],
                              daily_calibrator=kwargs["daily_calibrator"])

    monkeypatch.setattr(cli, "run_harness", fake_run)
    for path in ("options/underlying_symbol=AAA", "ohlc"):
        (tmp_path / path).mkdir(parents=True)
    (tmp_path / "e.sqlite").write_text("")
    assert cli.main(["--options-root", str(tmp_path / "options"), "--ohlc-root", str(tmp_path / "ohlc"),
                     "--earnings-db", str(tmp_path / "e.sqlite"), "--start", "2024-01-01", "--end", "2024-06-30",
                     "--out-dir", str(tmp_path / "out")]) == 0
    assert isinstance(seen[0], dv.DailyVarianceCalibrator) and seen[0].group_of("AAPL") == "Tech"
    (run_dir,) = list((tmp_path / "out").iterdir())
    report = (run_dir / "report.md").read_text()
    assert "## Ordinary daily variance during the hold" in report and "| har_rs |" in report
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["config"]["harness_version"] == "pricing_harness_v2"
    assert "daily_forecast_har_cc" in pd.read_csv(run_dir / "variance_rows.csv").columns
