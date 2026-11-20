"""The walk-forward pricing harness (services/pricing_harness.py).

What must hold for its numbers to mean anything: every read is point in time,
priors only ever hold outcomes that exited before the entry, the exit re-prices
exactly the booked contracts at the claim contract's conservative fill, and a
leg that cannot be priced is counted, never filled in.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date

import duckdb
import pandas as pd
import pytest
from scipy import stats

import scripts.run_forward_loop as forward_loop
import scripts.run_pricing_harness as cli
from services import pricing_harness as ph
from services import screener_service
from services import structure_scorecard as sc
from services.claim_contract import fee_per_share, slippage_per_share
from services.execution_scenarios import compare_execution_scenarios
from services.structure_legs import STRUCTURES
from tests.unit.test_services.pricing_harness_fixtures import SyntheticArchive, quarterly_events
from tests.unit.test_services.test_structure_scorecard import _base_snapshot

# AAA reports Thursday 2024-02-08 (exit Wed 02-07); BBB reports Monday
# 2024-02-12, so its entry window (01-29..02-08) contains AAA's exit day.
EVENTS = {
    "AAA": quarterly_events("AAA", [date(2024, 2, 8), date(2024, 5, 2)]),
    "BBB": quarterly_events("BBB", [date(2024, 2, 12), date(2024, 5, 13)], timing="BMO"),
}
START, END = date(2024, 1, 1), date(2024, 6, 30)


@pytest.fixture(scope="module")
def replay():
    archive = SyntheticArchive(EVENTS)
    seen = []

    def recording_builder(symbol, as_of, **kwargs):
        seen.append((symbol, as_of, kwargs))
        return ph.build_vol_snapshot(symbol, as_of, **kwargs)

    result = ph.run_harness(archive, ["AAA", "BBB"], start=START, end=END, snapshot_builder=recording_builder)
    return archive, result, seen


# ── the rules it shares with the forward loop ───────────────────────────────

def test_entry_window_and_traded_decisions_are_the_loops() -> None:
    assert (ph.DTE_MIN, ph.DTE_MAX) == (screener_service.DTE_MIN_DEFAULT, screener_service.DTE_MAX_DEFAULT)
    assert ph.TRADED_RECOMMENDATIONS == forward_loop.ALLOWED_RECOMMENDATIONS


@pytest.mark.parametrize("event, exit_day", [
    (date(2024, 2, 8), date(2024, 2, 7)),     # Thursday -> Wednesday
    (date(2024, 2, 12), date(2024, 2, 9)),    # Monday -> Friday
    (date(2024, 1, 16), date(2024, 1, 12)),   # Tuesday after MLK day -> Friday
    (date(2024, 7, 5), date(2024, 7, 3)),     # Friday after July 4 -> the early-close Wednesday
])
def test_the_exit_is_the_last_session_before_the_event(event, exit_day) -> None:
    assert ph.exit_session(event) == exit_day


def test_the_loops_calendar_exit_day_is_not_always_a_session() -> None:
    assert ph.loop_exit_day(date(2024, 2, 8)) == ph.exit_session(date(2024, 2, 8))
    assert ph.loop_exit_day(date(2024, 2, 12)) == date(2024, 2, 11)  # a Sunday


def test_entry_sessions_are_the_window_and_strictly_before_the_exit() -> None:
    event = date(2024, 2, 22)  # Thursday; exit Wednesday 02-21
    days = [date(2024, 2, d) for d in range(5, 22)]
    picked = ph.entry_sessions(event, days)
    assert picked[0] == date(2024, 2, 8)           # 14 days out
    assert date(2024, 2, 7) not in picked           # 15 days out
    assert picked[-1] == date(2024, 2, 19)          # 3 days out
    assert date(2024, 2, 20) not in picked          # 2 days out
    # A Monday event: 3 days out is the Friday exit itself, which is no entry.
    monday = date(2024, 2, 12)
    assert ph.entry_sessions(monday, [date(2024, 2, 9), date(2024, 2, 8)]) == [date(2024, 2, 8)]


def test_an_event_is_known_only_strictly_after_its_known_as_of_date() -> None:
    event = ph.EarningsEvent("AAA", date(2024, 2, 8), "AMC", known_as_of=date(2024, 1, 30))
    assert not ph.event_known_on(event, date(2024, 1, 30))
    assert ph.event_known_on(event, date(2024, 1, 31))
    assert ph.event_known_on(ph.EarningsEvent("AAA", date(2024, 2, 8), "AMC", lookahead_safe=True), date(2024, 1, 2))
    assert not ph.event_known_on(ph.EarningsEvent("AAA", date(2024, 2, 8), "AMC"), date(2024, 1, 2))


# ── point in time ───────────────────────────────────────────────────────────

def test_every_read_is_the_entry_session_or_the_exit_session(replay) -> None:
    archive, result, seen = replay
    entries = {(r["symbol"], date.fromisoformat(r["entry_date"])) for r in result.rows}
    exits = {(r["symbol"], date.fromisoformat(r["exit_date"])) for r in result.rows}
    chain_reads = {(s, d) for kind, s, d in archive.reads if kind == "chain"}
    price_reads = {(s, d) for kind, s, d in archive.reads if kind == "prices"}
    assert chain_reads == entries | exits
    # Prices through the exit are the outcome (the hold's realized variance);
    # the snapshot only ever gets the entry read (checked below).
    assert price_reads == entries | exits


def test_the_snapshot_sees_nothing_after_its_session(replay) -> None:
    _, result, seen = replay
    assert seen and len(seen) == result.counts["entries_scored"]
    for symbol, as_of, kwargs in seen:
        chain = kwargs["option_chain_data"]
        assert set(pd.to_datetime(chain["trade_date"]).dt.date) == {as_of}
        assert pd.to_datetime(kwargs["price_data"]["trade_date"]).max().date() <= as_of
        assert all(date.fromisoformat(e["event_date"]) < as_of
                   for e in kwargs["earnings_metadata"]["prior_events"])
        assert kwargs["structure_chain"]["fetched_on"] == as_of.isoformat()
        assert all(date.fromisoformat(d) > as_of for d in kwargs["structure_chain"]["listing"])


def test_the_full_chain_gives_complete_leg_coverage(replay) -> None:
    _, result, _ = replay
    assert not any(key.startswith("entries_legs_not_complete") for key in result.counts)
    assert all(r["unpriced_reason"] is None for r in result.rows)


def test_priors_only_hold_outcomes_that_exited_before_the_entry(replay) -> None:
    _, result, _ = replay
    first_exits = [(r["structure"], r["exit_date"]) for r in result.rows
                   if r["first_look"] and r["return_conservative_fill"] is not None]
    coincident = 0
    for row in result.rows:
        expected = sum(1 for s, x in first_exits if s == row["structure"] and x < row["entry_date"])
        coincident += sum(1 for s, x in first_exits if s == row["structure"] and x == row["entry_date"])
        assert row["prior_history_count"] == expected, row
    # The fixture must exercise both edges: an entry on another event's exit
    # day (excluded) and entries after earlier exits (included).
    assert coincident > 0
    assert max(r["prior_history_count"] for r in result.rows) > 0


def test_neutral_priors_never_learn() -> None:
    archive = SyntheticArchive(EVENTS)
    result = ph.run_harness(archive, ["AAA"], start=START, end=END, prior_mode=ph.PRIOR_NEUTRAL)
    assert result.rows and all(r["prior_history_count"] == 0 for r in result.rows)


def test_outcome_history_is_strictly_before_and_skips_non_finite() -> None:
    h = ph.OutcomeHistory()
    h.add(date(2024, 2, 7), "atm_straddle", 5.0)
    h.add(date(2024, 2, 8), "atm_straddle", float("nan"))
    h.add(date(2024, 2, 8), "atm_straddle", None)
    assert h.returns_before("atm_straddle", date(2024, 2, 7)) == []
    assert h.returns_before("atm_straddle", date(2024, 2, 8)) == [5.0]
    assert h.returns_before("otm_strangle", date(2024, 3, 1)) == []
    with pytest.raises(ValueError):
        h.priors(date(2024, 3, 1), "report")


def test_only_entries_on_which_the_date_was_known_are_kept_when_required() -> None:
    known = {"AAA": [ph.EarningsEvent("AAA", date(2024, 2, 8), "AMC", source_confidence=0.9,
                                      known_as_of=date(2024, 1, 31))]}
    result = ph.run_harness(SyntheticArchive(known), ["AAA"], start=START, end=END, prior_mode=ph.PRIOR_NEUTRAL,
                            require_lookahead_safe=True)
    entries = sorted({r["entry_date"] for r in result.rows})
    assert entries[0] == "2024-02-01" and all(r["event_date_known_beforehand"] for r in result.rows)
    unknown = ph.run_harness(SyntheticArchive(EVENTS), ["AAA"], start=START, end=END,
                             require_lookahead_safe=True)
    assert unknown.rows == [] and unknown.counts["events_skipped:no_entry_session"] == 2


# ── pricing ─────────────────────────────────────────────────────────────────

def _q(bid, ask, strike=100.0):
    return {"strike": strike, "bid": bid, "ask": ask, "mid": (bid + ask) / 2}


def _cf_buy(bid, ask):
    return (bid + ask) / 2 + 0.5 * (ask - bid) + slippage_per_share(bid, ask) + fee_per_share()


def _cf_sell(bid, ask):
    return (bid + ask) / 2 - 0.5 * (ask - bid) - slippage_per_share(bid, ask) - fee_per_share()


def test_a_straddle_round_trip_is_priced_at_the_conservative_fill() -> None:
    entry = {"call": _q(4.9, 5.1), "put": _q(3.9, 4.1)}
    exit_ = {"call": _q(6.0, 6.2), "put": _q(2.0, 2.2)}
    returns, reason = ph.price_round_trip("atm_straddle", entry, exit_)
    paid = _cf_buy(4.9, 5.1) + _cf_buy(3.9, 4.1)
    got = _cf_sell(6.0, 6.2) + _cf_sell(2.0, 2.2)
    assert reason is None
    assert returns["conservative_fill"] == pytest.approx((got - paid) / paid * 100, abs=1e-4)
    assert returns["conservative_fill"] == pytest.approx(-14.1224, abs=1e-3)
    assert returns["mid"] == pytest.approx((8.2 - 9.0) / 9.0 * 100, abs=1e-4)
    assert returns["conservative_fill"] < returns["natural"] < returns["mid"]


def test_a_condor_is_returned_on_its_capital_at_risk() -> None:
    entry = {"short_call": _q(2.0, 2.2, 105), "long_call": _q(1.0, 1.2, 108),
             "short_put": _q(2.0, 2.2, 95), "long_put": _q(1.0, 1.2, 92)}
    exit_ = {"short_call": _q(1.0, 1.2, 105), "long_call": _q(0.4, 0.6, 108),
             "short_put": _q(1.0, 1.2, 95), "long_put": _q(0.4, 0.6, 92)}
    assert ph.condor_capital_at_risk(entry) == pytest.approx(1.0)
    returns, reason = ph.price_round_trip("iron_condor", entry, exit_)
    credit = 2 * _cf_sell(2.0, 2.2) - 2 * _cf_buy(1.0, 1.2)
    cost = 2 * _cf_buy(1.0, 1.2) - 2 * _cf_sell(0.4, 0.6)
    assert reason is None
    assert returns["conservative_fill"] == pytest.approx((credit - cost) / 1.0 * 100, abs=1e-4)
    assert returns["conservative_fill"] == pytest.approx(-21.36, abs=1e-2)


def test_a_condor_without_defined_risk_is_not_priced() -> None:
    entry = {"short_call": _q(2.0, 2.2, 105), "long_call": _q(1.0, 1.2, 107),
             "short_put": _q(2.0, 2.2, 95), "long_put": _q(1.0, 1.2, 93)}
    returns, reason = ph.price_round_trip("iron_condor", entry, entry)
    assert reason == "condor_risk_undefined" and returns["conservative_fill"] is None


def test_a_calendar_is_short_the_front_and_long_the_back() -> None:
    entry = {"front": _q(2.0, 2.1), "back": _q(3.0, 3.1)}
    exit_ = {"front": _q(0.5, 0.6), "back": _q(2.6, 2.7)}
    returns, _ = ph.price_round_trip("call_calendar", entry, exit_)
    paid = _cf_buy(3.0, 3.1) - _cf_sell(2.0, 2.1)
    got = _cf_sell(2.6, 2.7) - _cf_buy(0.5, 0.6)
    assert returns["conservative_fill"] == pytest.approx((got - paid) / paid * 100, abs=1e-4)


def _exit_chain(rows):
    return pd.DataFrame([{"expiry": pd.Timestamp(e), "call_put": cp, "strike": k, "bid": b, "ask": a}
                         for e, cp, k, b, a in rows])


def test_exit_legs_match_expiry_side_and_strike() -> None:
    chain = _exit_chain([("2024-02-09", "C", 100.0, 1.0, 1.1), ("2024-02-09", "P", 100.0, 2.0, 2.1),
                         ("2024-02-23", "C", 100.0, 3.0, 3.1), ("2024-02-09", "C", 102.5, 0.5, 0.6)])
    quotes, reason = ph.exit_legs("call_calendar", {"front": _q(1, 1.1), "back": _q(3, 3.1)}, chain,
                                  trade_expiry="2024-02-09", back_expiry="2024-02-23")
    assert reason is None and quotes == {"front": {"strike": 100.0, "bid": 1.0, "ask": 1.1},
                                         "back": {"strike": 100.0, "bid": 3.0, "ask": 3.1}}
    quotes, _ = ph.exit_legs("put_calendar", {"front": _q(1, 1.1)}, chain,
                             trade_expiry="2024-02-09", back_expiry=None)
    assert quotes["front"]["bid"] == 2.0


@pytest.mark.parametrize("rows, reason", [
    ([("2024-02-09", "C", 100.0, 1.0, 1.1)], "exit_contract_missing:put"),
    ([("2024-02-09", "C", 100.0, 1.0, 1.1), ("2024-02-09", "C", 100.0, 1.0, 1.2),
      ("2024-02-09", "P", 100.0, 2.0, 2.1)], "exit_contract_ambiguous:call"),
    ([("2024-02-09", "C", 100.0, 0.0, 0.1), ("2024-02-09", "P", 100.0, 2.0, 2.1)], "exit_leg_zero_bid:call"),
    ([("2024-02-09", "C", 100.0, 1.2, 1.1), ("2024-02-09", "P", 100.0, 2.0, 2.1)], "exit_leg_unquoted:call"),
    ([("2024-02-09", "C", 100.0, None, 1.1), ("2024-02-09", "P", 100.0, 2.0, 2.1)], "exit_leg_unquoted:call"),
    ([], "exit_chain_missing"),
])
def test_an_unpriceable_exit_is_named_and_never_filled(rows, reason) -> None:
    chain = _exit_chain(rows) if rows else pd.DataFrame()
    quotes, why = ph.exit_legs("atm_straddle", {"call": _q(1, 1.1), "put": _q(2, 2.1)}, chain,
                               trade_expiry="2024-02-09", back_expiry=None)
    assert quotes is None and why == reason


def test_a_missing_exit_contract_leaves_the_row_unpriced() -> None:
    class Thin(SyntheticArchive):
        def chain(self, symbol, day):
            frame = super().chain(symbol, day)
            if day == date(2024, 2, 7) and not frame.empty:
                frame = frame[frame["call_put"] == "C"]  # the exit lost its puts
            return frame

    result = ph.run_harness(Thin({"AAA": EVENTS["AAA"][:1]}), ["AAA"], start=START, end=END)
    straddles = [r for r in result.rows if r["structure"] == "atm_straddle"]
    assert straddles and all(r["unpriced_reason"] == "exit_contract_missing:put" for r in straddles)
    assert all(r[f"return_{lvl}"] is None for r in straddles for lvl in ph.RETURN_LEVELS)
    assert result.counts["rows:atm_straddle:exit_contract_missing:put"] == len(straddles)
    # Nothing unpriced reaches the priors.
    assert all(r["prior_history_count"] == 0 for r in result.rows if r["structure"] == "atm_straddle")


def test_an_event_whose_exit_session_is_missing_is_skipped_and_counted() -> None:
    archive = SyntheticArchive({"AAA": EVENTS["AAA"][:1]}, missing_days={"AAA": {date(2024, 2, 7)}})
    result = ph.run_harness(archive, ["AAA"], start=START, end=END)
    assert result.rows == [] and result.counts["events_skipped:exit_session_not_in_archive"] == 1


def test_a_failing_snapshot_is_counted_not_hidden() -> None:
    def broken(*_a, **_k):
        raise RuntimeError("boom")

    result = ph.run_harness(SyntheticArchive({"AAA": EVENTS["AAA"][:1]}), ["AAA"], start=START, end=END,
                            snapshot_builder=broken)
    assert result.rows == [] and result.counts["entries_failed:RuntimeError"] == result.counts["entries_attempted"]


def test_short_price_history_is_skipped_and_counted() -> None:
    archive = SyntheticArchive({"AAA": EVENTS["AAA"][:1]}, start=date(2023, 12, 1))
    result = ph.run_harness(archive, ["AAA"], start=START, end=END)
    assert result.rows == [] and result.counts["entries_skipped:short_price_history"] > 0


def test_an_unknown_prior_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        ph.run_harness(SyntheticArchive(EVENTS), ["AAA"], start=START, end=END, prior_mode="report")


# ── caller-supplied priors in the scorecards ────────────────────────────────

def test_supplied_priors_never_read_the_reports_or_the_store(monkeypatch) -> None:
    def forbidden(*_a, **_k):
        raise AssertionError("must not be read")

    monkeypatch.setattr(sc, "_load_walk_forward_priors", forbidden)
    import services.structure_prior_store as store
    monkeypatch.setattr(store, "get_structure_prior_store", forbidden)
    priors = {s: sc.prior_from_returns(s, [10.0] * 30, source="test") for s in STRUCTURES}
    cards = sc.build_structure_scorecards(_base_snapshot(), as_of_date=date(2024, 1, 2), priors=priors)
    assert {c.structure for c in cards} == set(STRUCTURES)
    assert all(c.walk_forward_history_count == 30 for c in cards)


def test_supplied_priors_score_exactly_like_the_scorers_given_them() -> None:
    priors = {s: sc.prior_from_returns(s, [4.0, -2.0, 6.0], source="test") for s in STRUCTURES}
    snap = _base_snapshot()
    cards = {c.structure: c for c in sc.build_structure_scorecards(snap, priors=priors)}
    assert cards["atm_straddle"] == sc.score_atm_straddle(snap, prior=priors["atm_straddle"])
    assert cards["iron_condor"] == sc.score_iron_condor(snap, prior=priors["iron_condor"])


def test_priors_must_cover_every_structure() -> None:
    with pytest.raises(ValueError, match="iron_condor"):
        sc.build_structure_scorecards(_base_snapshot(), priors={s: sc.prior_from_returns(s, [], source="t")
                                                                 for s in STRUCTURES if s != "iron_condor"})


def test_a_prior_from_returns_is_what_production_shrinks_into_the_neutral_baseline() -> None:
    from services.structure_prior_store import _recompute_aggregates

    assert sc.prior_from_returns("atm_straddle", [], source="t") == sc._neutral_prior("atm_straddle", source="t")
    returns = [5.0, -3.0, 8.0, float("nan")]
    agg = _recompute_aggregates([{"realized_return_pct": r} for r in returns[:3]])
    expected = sc._shrink_live_into_report_prior(sc._neutral_prior("atm_straddle", source="t"), {
        "history_count": 3, "win_rate": agg["win_rate"], "avg_return_pct": agg["avg_return_pct"]})
    got = sc.prior_from_returns("atm_straddle", returns, source="t")
    assert (got.history_count, got.win_rate, got.avg_return_pct, got.rank_score) == (
        expected.history_count, expected.win_rate, expected.avg_return_pct, expected.rank_score)
    assert got.source.startswith("t:shrinkage_blend")


# ── metrics ─────────────────────────────────────────────────────────────────

def test_spearman_matches_scipy_with_ties() -> None:
    x = [1.0, 2.0, 2.0, 3.0, 5.0, 4.0]
    y = [2.0, 1.0, 4.0, 4.0, 6.0, 3.0]
    assert ph.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)
    assert ph.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    with pytest.raises(ValueError):
        ph.spearman([1.0], [1.0, 2.0])


def _row(pred, realized, **extra):
    return {"expected_edge_pct": pred, "return_conservative_fill": realized, **extra}


def test_prediction_quality_reports_bias_sign_and_calibration() -> None:
    rows = [_row(p, r) for p, r in [(5, 3), (4, -1), (-2, -4), (-1, 2), (3, 6), (0, 1), (None, 9), (2, None)]]
    q = ph.prediction_quality(rows)
    assert q["n"] == 6
    assert q["mean_predicted"] == pytest.approx(1.5) and q["mean_realized"] == pytest.approx(7 / 6, abs=1e-4)
    assert q["bias"] == pytest.approx(1.5 - 7 / 6, abs=1e-4)
    assert q["sign_hit_rate"] == pytest.approx(3 / 5)  # the zero prediction is not signed
    assert q["mean_realized_when_predicted_positive"] == pytest.approx(8 / 3, abs=1e-4)
    assert q["mean_realized_when_predicted_non_positive"] == pytest.approx(-1 / 3, abs=1e-4)
    assert [b["n"] for b in q["calibration"]] == [2, 1, 1, 1, 1]
    assert q["calibration"][0]["mean_predicted"] == pytest.approx(-1.5)
    assert ph.prediction_quality([]) == {"n": 0}


def test_the_bootstrap_interval_is_deterministic_and_brackets_the_mean() -> None:
    values = [1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 0.5]
    ci = ph.bootstrap_mean_ci(values)
    assert ci == ph.bootstrap_mean_ci(values)
    assert ci[0] < sum(values) / len(values) < ci[1]
    assert ph.bootstrap_mean_ci([1.0, 2.0]) is None


def _policy_row(symbol, entry, structure, ret, *, traded=False, edge=0.0):
    return {"symbol": symbol, "event_date": "2024-02-08", "entry_date": entry, "structure": structure,
            "return_conservative_fill": ret, "traded": traded, "expected_edge_pct": edge}


def test_policies_trade_the_first_traded_session_against_the_same_day_straddle() -> None:
    rows = [
        _policy_row("AAA", "2024-01-25", "atm_straddle", -5.0, edge=1.0),
        _policy_row("AAA", "2024-01-25", "otm_strangle", -9.0, edge=2.0),
        _policy_row("AAA", "2024-01-29", "atm_straddle", 4.0),
        _policy_row("AAA", "2024-01-29", "otm_strangle", 10.0, traded=True),
        _policy_row("AAA", "2024-01-30", "otm_strangle", 99.0, traded=True),  # a later decision is not booked
        _policy_row("BBB", "2024-01-25", "atm_straddle", 2.0),
        _policy_row("BBB", "2024-01-25", "iron_condor", None, traded=True),
    ]
    out = ph.policy_comparison(rows)
    assert out["events"] == 2
    assert out["selector"]["n_events"] == 1 and out["selector"]["mean_return"] == 10.0
    assert out["selector"]["traded_but_unpriced"] == 1
    assert out["straddle_same_entry"]["mean_return"] == 4.0
    assert out["selector_minus_straddle_same_entry"]["mean_return"] == 6.0
    assert out["straddle_first_look"]["n_events"] == 2 and out["straddle_first_look"]["mean_return"] == -1.5
    assert out["rank_pick_first_look"]["mean_return"] == pytest.approx((-9.0 + 2.0) / 2)


def test_the_summary_states_its_basis_and_caveats(replay) -> None:
    _, result, _ = replay
    summary = ph.summarize(result)
    json.dumps(summary, allow_nan=False)
    assert summary["config"]["judged_return"] == "conservative_fill"
    assert summary["config"]["prior_mode"] == ph.PRIOR_WALK_FORWARD
    assert summary["event_dates"]["posthoc_rows"] == summary["event_dates"]["first_look_rows"] > 0
    assert summary["counts"]["events_loop_exit_day_not_a_session"] == 2  # BBB's Monday events
    for structure in STRUCTURES:
        block = summary["per_structure"][structure]
        assert block["first_look"]["expected_edge_pct"]["n"] == 4
        assert block["priced_share"] == 1.0
    assert any("claim contract" in c for c in summary["caveats"])
    assert any("post-hoc" in c for c in summary["caveats"])


# ── the parquet archive and the script ──────────────────────────────────────

def _write_parquet(frame: pd.DataFrame, root) -> None:
    con = duckdb.connect(":memory:")
    con.register("frame", frame)
    con.execute(f"COPY frame TO '{root}' (FORMAT PARQUET, PARTITION_BY (underlying_symbol))")


@pytest.fixture(scope="module")
def parquet_archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("t9")
    events = {"AAA": EVENTS["AAA"][:1]}
    synthetic = SyntheticArchive(events)
    event = events["AAA"][0].event_date
    days = ph.entry_sessions(event, synthetic.sessions("AAA")) + [ph.exit_session(event)]
    chain = pd.concat([synthetic.chain("AAA", d) for d in days], ignore_index=True).assign(underlying_symbol="AAA")
    _write_parquet(chain, root / "options")
    ohlc = synthetic.prices("AAA", date(2024, 6, 28)).assign(underlying_symbol="AAA")
    _write_parquet(ohlc, root / "ohlc")
    db = root / "earnings.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE historical_earnings_events (symbol TEXT, event_date TEXT, release_timing TEXT, "
                     "primary_source TEXT, source_confidence REAL, lookahead_safe_for_pre_event INTEGER, "
                     "known_as_of TEXT)")
        conn.execute("INSERT INTO historical_earnings_events VALUES "
                     "('AAA', '2024-02-08', 'after market close', 'synthetic', 0.9, 0, NULL)")
        conn.execute("INSERT INTO historical_earnings_events VALUES "
                     "('AAA', '2023-11-02', 'amc', 'synthetic', 0.9, 0, '2023-10-01T12:00:00')")
    return root, synthetic, days


def test_the_parquet_archive_reads_the_t9_layout(parquet_archive) -> None:
    root, synthetic, days = parquet_archive
    archive = ph.ParquetArchive(root / "options", root / "ohlc", root / "earnings.sqlite")
    assert archive.sessions("AAA") == days
    assert archive.sessions("ZZZ") == []
    chain = archive.chain("AAA", days[0])
    assert len(chain) == len(synthetic.chain("AAA", days[0]))
    assert set(pd.to_datetime(chain["trade_date"]).dt.date) == {days[0]}
    prices = archive.prices("AAA", days[0])
    assert pd.to_datetime(prices["trade_date"]).max().date() == days[0]
    events = archive.events("AAA")
    assert [(e.event_date, e.release_timing, e.known_as_of) for e in events] == [
        (date(2023, 11, 2), "AMC", date(2023, 10, 1)), (date(2024, 2, 8), "AMC", None)]
    with pytest.raises(ValueError):
        archive.sessions("A'; DROP")


def test_the_parquet_archive_replays_like_the_in_memory_one(parquet_archive) -> None:
    root, _, _ = parquet_archive
    archive = ph.ParquetArchive(root / "options", root / "ohlc", root / "earnings.sqlite")
    from_parquet = ph.run_harness(archive, ["AAA"], start=START, end=END)
    # The same events the database holds (its 2023 event is history for both).
    same_events = SyntheticArchive({"AAA": archive.events("AAA")})
    in_memory = ph.run_harness(same_events, ["AAA"], start=START, end=END)
    assert from_parquet.rows and len(from_parquet.rows) == len(in_memory.rows)
    for a, b in zip(from_parquet.rows, in_memory.rows):
        for key in ("entry_date", "structure", "expected_edge_pct", "return_conservative_fill", "recommendation"):
            va, vb = a[key], b[key]
            assert (va == pytest.approx(vb, abs=1e-6)) if isinstance(va, float) else va == vb, key


def test_the_script_writes_the_report(parquet_archive, tmp_path, capsys) -> None:
    root, _, _ = parquet_archive
    code = cli.main(["--options-root", str(root / "options"), "--ohlc-root", str(root / "ohlc"),
                     "--earnings-db", str(root / "earnings.sqlite"), "--start", "2024-01-01",
                     "--end", "2024-06-30", "--out-dir", str(tmp_path)])
    assert code == 0
    (run_dir,) = list(tmp_path.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["config"]["symbols"] == ["AAA"]
    assert summary["config"]["git_sha"] is None or len(summary["config"]["git_sha"]) == 40
    report = (run_dir / "report.md").read_text()
    assert "| atm_straddle |" in report and "## Caveats" in report
    rows = pd.read_csv(run_dir / "rows.csv")
    assert len(rows) == summary["counts"]["entries_scored"] * len(STRUCTURES)
    assert "Report:" in capsys.readouterr().out


def test_the_script_refuses_a_missing_archive(tmp_path, capsys) -> None:
    assert cli.main(["--data-root", str(tmp_path / "nowhere"), "--out-dir", str(tmp_path)]) == 2
    assert "Archive input not found" in capsys.readouterr().err


def test_only_events_reported_before_the_session_are_history() -> None:
    events = [ph.EarningsEvent("AAA", date(2023, 11, 2), "AMC"), ph.EarningsEvent("AAA", date(2024, 1, 25), "BMO"),
              ph.EarningsEvent("AAA", date(2024, 2, 8), "AMC")]
    meta = ph._earnings_metadata(events[2], events, date(2024, 1, 25))
    assert meta["prior_events"] == [{"event_date": "2023-11-02", "release_timing": "AMC"}]
    assert meta["earnings_date"] == "2024-02-08" and meta["release_timing"] == "AMC"


def test_first_look_policies_use_only_the_first_session() -> None:
    rows = [
        _policy_row("AAA", "2024-01-29", "atm_straddle", 40.0, edge=0.0),
        _policy_row("AAA", "2024-01-29", "otm_strangle", 30.0, edge=9.0),
        _policy_row("AAA", "2024-01-25", "atm_straddle", -5.0, edge=1.0),
        _policy_row("AAA", "2024-01-25", "otm_strangle", -9.0, edge=2.0),
    ]
    out = ph.policy_comparison(rows)
    assert out["straddle_first_look"]["mean_return"] == -5.0
    assert out["rank_pick_first_look"]["mean_return"] == -9.0


def test_a_structure_the_chain_cannot_form_is_unpriced_with_its_reason() -> None:
    # Expiries at most 15 days out: the calendars never have a back month.
    archive = SyntheticArchive({"AAA": EVENTS["AAA"][:1]}, expiry_horizon_days=16)
    result = ph.run_harness(archive, ["AAA"], start=START, end=END)
    calendars = [r for r in result.rows if r["structure"] == "call_calendar"]
    assert calendars and all(r["unpriced_reason"] == "unformable:no_back_expiry_for_calendar" for r in calendars)
    assert all(r["return_conservative_fill"] is None for r in calendars)
    assert all(r["unpriced_reason"] is None for r in result.rows if r["structure"] == "atm_straddle")


def test_sampled_legs_are_never_priced() -> None:
    def without_structure_chain(symbol, as_of, **kwargs):
        kwargs.pop("structure_chain")
        return ph.build_vol_snapshot(symbol, as_of, **kwargs)

    result = ph.run_harness(SyntheticArchive({"AAA": EVENTS["AAA"][:1]}), ["AAA"], start=START, end=END,
                            snapshot_builder=without_structure_chain)
    assert result.rows and all(r["unpriced_reason"] == "legs_not_complete" for r in result.rows)
    assert result.counts["entries_legs_not_complete:sampled"] == result.counts["entries_scored"]


def test_the_structure_listing_holds_only_expiries_after_the_session() -> None:
    chain = pd.DataFrame({"expiry": pd.to_datetime(["2024-01-26", "2024-02-02", "2024-01-26", "2024-01-19"])})
    out = ph._structure_chain(chain, date(2024, 1, 26))
    assert out["listing"] == ["2024-02-02"] and out["fetched_on"] == "2024-01-26"
