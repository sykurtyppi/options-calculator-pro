"""Event variance from the term structure, and the walk-forward forecast of its change."""
from __future__ import annotations

import json
import math
from datetime import date

import pandas as pd
import pytest

import scripts.run_pricing_harness as cli
from services import event_variance as ev
from services import pricing_harness as ph
from tests.unit.test_services.pricing_harness_fixtures import (
    BASE_VOL,
    EVENT_MOVE,
    SyntheticArchive,
    quarterly_events,
)

ENTRY = date(2024, 1, 29)
EVENT = date(2024, 2, 8)  # Thursday


# ── measurement ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("start, end, n", [
    (date(2024, 1, 29), date(2024, 1, 29), 0),
    (date(2024, 1, 26), date(2024, 1, 29), 1),     # Friday -> Monday
    (date(2024, 1, 12), date(2024, 1, 16), 1),     # over MLK day
    (date(2024, 1, 29), date(2024, 2, 9), 9),
])
def test_sessions_are_counted_after_start_through_end(start, end, n) -> None:
    assert ev.sessions_between(start, end) == n


def _rows(expiry, side, points):
    return [{"expiry": pd.Timestamp(expiry), "call_put": side, "strike": k, "iv": v, "underlying_price": 101.0}
            for k, v in points]


def test_atm_iv_is_linear_in_strike_and_averages_calls_and_puts() -> None:
    chain = pd.DataFrame(_rows("2024-02-09", "C", [(100.0, 0.40), (102.5, 0.30)])
                         + _rows("2024-02-09", "P", [(100.0, 0.50), (102.5, 0.40)])
                         + _rows("2024-02-16", "C", [(100.0, 9.0), (102.5, 9.0)]))
    # spot 101 is 40% of the way from 100 to 102.5
    assert ev.atm_iv(chain, date(2024, 2, 9), 101.0) == pytest.approx(((0.36) + (0.46)) / 2)
    assert ev.atm_iv(chain, date(2024, 2, 9), 100.0) == pytest.approx(0.45)


def test_atm_iv_never_extrapolates_and_skips_bad_values() -> None:
    chain = pd.DataFrame(_rows("2024-02-09", "C", [(100.0, 0.40), (102.5, 0.0), (105.0, None), (95.0, 0.42)]))
    assert ev.atm_iv(chain, date(2024, 2, 9), 101.0) is None  # nothing valid above spot
    assert ev.atm_iv(chain, date(2024, 2, 9), 99.0) == pytest.approx(0.42 + 0.8 * (0.40 - 0.42))
    assert ev.atm_iv(chain, date(2024, 2, 23), 99.0) is None
    assert ev.atm_iv(chain.drop(columns=["iv"]), date(2024, 2, 9), 99.0) is None
    dup = pd.DataFrame(_rows("2024-02-09", "C", [(100.0, 0.40), (100.0, 0.50)]))
    assert ev.atm_iv(dup, date(2024, 2, 9), 100.0) == pytest.approx(0.45)


def test_the_two_expiry_estimate_recovers_the_event_and_base_variance() -> None:
    archive = SyntheticArchive({"AAA": quarterly_events("AAA", [EVENT])})
    out = ev.implied_event_variance(archive.chain("AAA", ENTRY), ENTRY, EVENT, "AMC")
    assert out.usable and out.reason is None
    assert out.expiries == (date(2024, 2, 9), date(2024, 2, 16))
    assert out.sessions == (9, 14)
    assert out.event_variance == pytest.approx(EVENT_MOVE ** 2, rel=1e-9)
    assert out.base_variance_per_session == pytest.approx(BASE_VOL ** 2 / 252, rel=1e-9)
    assert out.event_move_pct == pytest.approx(EVENT_MOVE * 100)


def test_a_growing_event_variance_is_tracked_day_by_day() -> None:
    archive = SyntheticArchive({"AAA": quarterly_events("AAA", [EVENT])}, event_growth={"AAA": 1.5})
    for day in (date(2024, 1, 25), ENTRY, date(2024, 2, 5), date(2024, 2, 7)):
        out = ev.implied_event_variance(archive.chain("AAA", day), day, EVENT, "AMC")
        assert out.event_variance == pytest.approx(archive.event_variance("AAA", EVENT, day), rel=1e-9), day


def _pair_chain(front_iv, back_iv, *, front="2024-02-09", back="2024-02-16"):
    return pd.DataFrame(_rows(front, "C", [(100.0, front_iv), (102.5, front_iv)])
                        + _rows(back, "C", [(100.0, back_iv), (102.5, back_iv)]))


def test_only_expiries_on_or_after_the_reaction_session_span_the_event() -> None:
    friday = date(2024, 2, 9)
    chain = _pair_chain(0.5, 0.4, front="2024-02-09", back="2024-02-16")
    chain = pd.concat([chain, pd.DataFrame(_rows("2024-02-23", "C", [(100.0, 0.35), (102.5, 0.35)]))])
    # Before the open on Friday: the Friday expiry spans it.
    assert ev.implied_event_variance(chain, ENTRY, friday, "BMO").expiries == (date(2024, 2, 9), date(2024, 2, 16))
    # After the close (or unknown): the reaction is Monday, so Friday's expiry does not.
    for timing in ("AMC", "UNKNOWN"):
        assert ev.implied_event_variance(chain, ENTRY, friday, timing).expiries == (
            date(2024, 2, 16), date(2024, 2, 23))


def test_the_entry_pair_can_be_forced_at_exit() -> None:
    chain = pd.concat([_pair_chain(0.5, 0.4), pd.DataFrame(_rows("2024-02-23", "C", [(100.0, 0.3), (102.5, 0.3)]))])
    out = ev.implied_event_variance(chain, date(2024, 2, 7), EVENT, "AMC",
                                    expiries=[date(2024, 2, 23), date(2024, 2, 9)])
    assert out.expiries == (date(2024, 2, 9), date(2024, 2, 23))


@pytest.mark.parametrize("chain, reason", [
    (pd.DataFrame(_rows("2024-02-09", "C", [(100.0, 0.5)])).drop(columns=["underlying_price"]), "no_spot"),
    (pd.DataFrame(_rows("2024-02-09", "C", [(100.0, 0.5), (102.5, 0.5)])), "fewer_than_two_spanning_expiries"),
    (_pair_chain(0.5, None), "atm_iv_unavailable:2024-02-16"),
    (_pair_chain(0.10, 0.60), "non_positive_event_variance"),
])
def test_an_unmeasurable_event_variance_says_why(chain, reason) -> None:
    out = ev.implied_event_variance(chain, ENTRY, EVENT, "AMC")
    assert out.reason == reason and not out.usable and out.event_move_pct is None


def test_expiries_without_distinct_sessions_are_not_separable() -> None:
    # 2024-03-29 is Good Friday: the Thursday and Friday expiries end on the same session count.
    chain = _pair_chain(0.5, 0.4, front="2024-03-28", back="2024-03-29")
    out = ev.implied_event_variance(chain, date(2024, 3, 20), date(2024, 3, 27), "AMC")
    assert out.reason == "expiries_not_separable"


# ── forecast ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("held, bucket", [(1, 0), (2, 0), (3, 1), (6, 2), (8, 3), (9, 4), (40, 4)])
def test_hold_buckets(held, bucket) -> None:
    assert ev.hold_bucket(held) == bucket


def test_a_hold_must_be_at_least_one_session() -> None:
    with pytest.raises(ValueError):
        ev.hold_bucket(0)


def _obs(symbol, event_day, exit_day, held, ratio, entry=date(2024, 1, 2)):
    return ev.VarianceObservation(symbol, event_day, entry, exit_day, held, ratio)


def test_with_no_history_the_forecast_is_no_change() -> None:
    f = ev.EventVarianceForecaster()
    assert f.predict("AAA", 5, date(2024, 3, 1)) == ev.VarianceForecast(0.0, 0.0, 0.0, 0.0, 0, 0, 0)


def test_the_forecast_uses_only_exits_before_its_date() -> None:
    f = ev.EventVarianceForecaster()
    f.add(_obs("AAA", date(2024, 2, 8), date(2024, 2, 7), 5, 0.4))
    f.add(_obs("BBB", date(2024, 2, 9), date(2024, 2, 8), 5, float("nan")))
    assert f.predict("CCC", 5, date(2024, 2, 7)).n_pooled == 0
    assert f.predict("CCC", 5, date(2024, 2, 8)).n_pooled == 1
    assert f.predict("CCC", 5, date(2024, 2, 9)).n_pooled == 1  # the NaN was never kept


def test_the_forecast_is_the_bucket_mean_plus_shrunk_offsets() -> None:
    k = 8.0
    f = ev.EventVarianceForecaster(shrinkage_events=k, group_of={"AAA": "Tech", "BBB": "Tech", "CCC": "Fin"}.get)
    x = date(2024, 1, 10)
    data = [("AAA", date(2024, 1, 11), 5, 0.6), ("AAA", date(2024, 1, 12), 5, 0.4),
            ("BBB", date(2024, 1, 13), 6, 0.1), ("CCC", date(2024, 1, 14), 5, -0.1),
            ("CCC", date(2024, 1, 15), 1, 0.05)]
    for sym, event_day, held, r in data:
        f.add(_obs(sym, event_day, x, held, r))
    out = f.predict("AAA", 5, date(2024, 2, 1))
    bucket = (0.6 + 0.4 + 0.1 - 0.1) / 4  # sessions 5 and 6 share a bucket
    res = {"AAA1": 0.6 - bucket, "AAA2": 0.4 - bucket, "BBB": 0.1 - bucket}
    group_mean = (res["AAA1"] + res["AAA2"] + res["BBB"]) / 3
    group = group_mean * 3 / (3 + k)
    symbol_mean = ((res["AAA1"] - group) + (res["AAA2"] - group)) / 2
    symbol = symbol_mean * 2 / (2 + k)
    assert out.pooled == pytest.approx(bucket)
    assert out.group_offset == pytest.approx(group) and out.n_group_events == 3
    assert out.symbol_offset == pytest.approx(symbol) and out.n_symbol_events == 2
    assert out.log_ratio == pytest.approx(bucket + group + symbol)
    assert out.n_pooled == 5
    # An empty bucket falls back to the overall mean; an ungrouped symbol has no group offset.
    other = f.predict("ZZZ", 9, date(2024, 2, 1))
    assert other.pooled == pytest.approx((0.6 + 0.4 + 0.1 - 0.1 + 0.05) / 5)
    assert other.group_offset == 0.0 and other.symbol_offset == 0.0 and other.n_group_events == 0


def test_each_event_weighs_the_same_in_an_offset_however_many_entries_it_has() -> None:
    f = ev.EventVarianceForecaster(shrinkage_events=2.0)
    for held in (9, 9, 9):
        f.add(_obs("AAA", date(2024, 1, 20), date(2024, 1, 19), held, 1.0))
    f.add(_obs("AAA", date(2024, 1, 25), date(2024, 1, 24), 9, 0.0))
    f.add(_obs("BBB", date(2024, 1, 26), date(2024, 1, 24), 9, 0.2))
    out = f.predict("AAA", 9, date(2024, 2, 1))
    bucket = (3 * 1.0 + 0.0 + 0.2) / 5
    per_event = ((1.0 - bucket) + (0.0 - bucket)) / 2  # not the per-observation mean
    assert out.n_symbol_events == 2
    assert out.symbol_offset == pytest.approx(per_event * 2 / (2 + 2.0))


def test_entries_of_one_event_count_once_in_its_offset() -> None:
    f = ev.EventVarianceForecaster(shrinkage_events=1.0)
    for held, r in ((9, 1.0), (5, 1.0), (2, 1.0)):
        f.add(_obs("AAA", date(2024, 1, 20), date(2024, 1, 19), held, r))
    f.add(_obs("BBB", date(2024, 1, 21), date(2024, 1, 19), 5, 0.0))
    out = f.predict("AAA", 5, date(2024, 2, 1))
    assert out.n_symbol_events == 1
    # Residuals 0, 0.5, 0 (buckets 9+, 5-6, 1-2) average to 1/6; one event at k=1 keeps half.
    assert out.symbol_offset == pytest.approx((1 / 6) * 0.5)


def test_forecast_quality_scores_against_both_baselines() -> None:
    rows = [{"symbol": "A", "event_date": "e1", "sessions_held": 5, "realized_log_ratio": r,
             "forecast_log_ratio": p, "pooled_log_ratio": q}
            for r, p, q in [(0.2, 0.1, 0.0), (0.4, 0.3, 0.1), (-0.1, 0.0, 0.1), (0.3, 0.2, 0.2)]]
    rows.append({"symbol": "A", "event_date": "e2", "sessions_held": 2, "realized_log_ratio": None,
                 "forecast_log_ratio": 0.1, "pooled_log_ratio": 0.1})
    q = ev.forecast_quality(rows)
    sse, sse0, ssep = 0.04, 0.30, 0.04 + 0.09 + 0.04 + 0.01
    assert q["n"] == 4 and q["events"] == 1
    assert q["rmse"] == pytest.approx(math.sqrt(sse / 4), abs=1e-5)
    assert q["rmse_no_change"] == pytest.approx(math.sqrt(sse0 / 4), abs=1e-5)
    assert q["r2_vs_no_change"] == pytest.approx(1 - sse / sse0, abs=1e-5)
    assert q["r2_vs_pooled"] == pytest.approx(1 - sse / ssep, abs=1e-5)
    assert q["by_hold_sessions"] == {"5-6": {"n": 4, "mean_realized": 0.2, "mean_forecast": 0.15}}
    assert ev.forecast_quality([]) == {"n": 0}


# ── in the harness ──────────────────────────────────────────────────────────

SYMBOLS = ("AAA", "BBB", "CCC", "DDD")
# CCC and DDD report two days earlier, so their exits fall inside AAA's and
# BBB's holding periods: a forecast dated at the exit would see them.
EVENTS = {s: quarterly_events(s, [date(2024, 2, 8), date(2024, 5, 2)] if s in ("AAA", "BBB")
                              else [date(2024, 2, 6), date(2024, 4, 30)]) for s in SYMBOLS}
GROWTH = {"AAA": 1.5, "BBB": 1.5, "CCC": 0.0, "DDD": 0.0}


@pytest.fixture(scope="module")
def replay():
    archive = SyntheticArchive(EVENTS, event_growth=GROWTH)
    return archive, ph.run_harness(archive, list(SYMBOLS), start=date(2024, 1, 1), end=date(2024, 6, 30),
                                   prior_mode=ph.PRIOR_NEUTRAL)


def test_every_scored_entry_has_a_variance_row_measured_on_the_entry_pair(replay) -> None:
    archive, result = replay
    assert len(result.variance_rows) == result.counts["entries_scored"] == result.counts["variance:measured"]
    for row in result.variance_rows:
        entry, exit_ = date.fromisoformat(row["entry_date"]), date.fromisoformat(row["exit_date"])
        event = date.fromisoformat(row["event_date"])
        assert row["entry_event_variance"] == pytest.approx(archive.event_variance(row["symbol"], event, entry))
        exit_chain = archive.chain(row["symbol"], exit_)
        forced = ev.implied_event_variance(exit_chain, exit_, event, "AMC",
                                           expiries=[date.fromisoformat(e) for e in row["expiries"]])
        assert row["exit_event_variance"] == pytest.approx(forced.event_variance)
        assert row["realized_log_ratio"] == pytest.approx(
            math.log(row["exit_event_variance"] / row["entry_event_variance"]))
        assert row["sessions_held"] == ev.sessions_between(entry, exit_)
        assert row["exit_expiries"] == row["expiries"]


def test_the_variance_forecast_only_knows_exits_before_the_entry(replay) -> None:
    _, result = replay
    measured = [(r["exit_date"], r["symbol"], r["event_date"]) for r in result.variance_rows
                if r["realized_log_ratio"] is not None]
    for row in result.variance_rows:
        assert row["forecast_n_pooled"] == sum(1 for x, _, _ in measured if x < row["entry_date"])
        assert row["forecast_n_symbol_events"] == len({e for x, s, e in measured
                                                       if s == row["symbol"] and x < row["entry_date"]})
    assert any(r["forecast_n_pooled"] > 0 for r in result.variance_rows)
    # Some exits land strictly inside another entry's holding period.
    assert any(r["entry_date"] < x < r["exit_date"] for r in result.variance_rows for x, _, _ in measured)


def test_the_exit_is_measured_on_the_entry_pair_even_when_a_nearer_expiry_lists() -> None:
    event = date(2024, 2, 8)
    exit_day = ph.exit_session(event)

    class LateListing(SyntheticArchive):
        def chain(self, symbol, day):
            frame = super().chain(symbol, day)
            if day != exit_day and not frame.empty:
                frame = frame[pd.to_datetime(frame["expiry"]).dt.date != date(2024, 2, 16)]
            return frame

    result = ph.run_harness(LateListing({"AAA": quarterly_events("AAA", [event])}), ["AAA"],
                            start=date(2024, 1, 1), end=date(2024, 6, 30), prior_mode=ph.PRIOR_NEUTRAL)
    assert result.variance_rows
    for row in result.variance_rows:
        assert row["expiries"] == ["2024-02-09", "2024-02-23"]
        assert row["exit_expiries"] == row["expiries"]


def test_it_learns_which_names_run_up(replay) -> None:
    _, result = replay
    may = [r for r in result.variance_rows if r["event_date"][:7] in ("2024-04", "2024-05") and r["first_look"]]
    by_symbol = {r["symbol"]: r for r in may}
    assert by_symbol["AAA"]["symbol_offset"] > 0 > by_symbol["CCC"]["symbol_offset"]
    summary = ph.summarize(result)["event_variance"]
    assert summary["measured_entries"] == len(result.variance_rows)
    assert summary["all_entries"]["r2_vs_no_change"] > 0
    assert summary["all_entries"]["rmse"] < summary["all_entries"]["rmse_pooled"]


def test_an_unmeasurable_entry_is_counted_and_never_learned() -> None:
    exits = {ph.exit_session(e.event_date) for e in EVENTS["AAA"]}

    class NoExitIv(SyntheticArchive):
        def chain(self, symbol, day):
            frame = super().chain(symbol, day)
            return frame.assign(iv=float("nan")) if day in exits and not frame.empty else frame

    result = ph.run_harness(NoExitIv({"AAA": EVENTS["AAA"]}), ["AAA"], start=date(2024, 1, 1),
                            end=date(2024, 6, 30), prior_mode=ph.PRIOR_NEUTRAL)
    assert result.variance_rows
    assert all(r["entry_event_variance"] is not None for r in result.variance_rows)
    assert all(r["realized_log_ratio"] is None and r["unmeasured_reason"].startswith("atm_iv_unavailable")
               for r in result.variance_rows)
    assert result.counts["variance:atm_iv_unavailable"] == len(result.variance_rows)
    assert all(r["forecast_n_pooled"] == 0 for r in result.variance_rows)
    assert not any(k == "variance:measured" for k in result.counts)


def test_the_script_writes_the_variance_rows_and_section(tmp_path, monkeypatch) -> None:
    archive = SyntheticArchive({"AAA": EVENTS["AAA"][:1]})
    grouped = []

    def fake_run(_archive, symbols, **kwargs):
        grouped.append(kwargs["variance_forecaster"].group_of)
        return ph.run_harness(archive, symbols, start=kwargs["start"], end=kwargs["end"],
                              variance_forecaster=kwargs["variance_forecaster"])

    monkeypatch.setattr(cli, "run_harness", fake_run)
    for path in ("options/underlying_symbol=AAA", "ohlc"):
        (tmp_path / path).mkdir(parents=True)
    (tmp_path / "e.sqlite").write_text("")
    args = ["--options-root", str(tmp_path / "options"), "--ohlc-root", str(tmp_path / "ohlc"),
            "--earnings-db", str(tmp_path / "e.sqlite"), "--start", "2024-01-01", "--end", "2024-06-30"]
    assert cli.main(args + ["--out-dir", str(tmp_path / "a")]) == 0
    assert cli.main(args + ["--out-dir", str(tmp_path / "b"), "--group-by", "none"]) == 0
    assert grouped[0] is not None and grouped[0]("AAPL") == "Tech" and grouped[0]("ZZZZ") is None
    assert grouped[1] is None
    (run_dir,) = list((tmp_path / "a").iterdir())
    rows = pd.read_csv(run_dir / "variance_rows.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert len(rows) == summary["event_variance"]["entries"] > 0
    assert summary["config"]["variance_grouping"] == "by group"
    assert "## Event variance" in (run_dir / "report.md").read_text()
