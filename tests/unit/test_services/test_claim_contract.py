"""Claimability (F5): the frozen claim contract and the conservative fill.

Before this, claim_allowed required a quote labelled "execution_grade", which
no forward quote ever carried, so nothing could be claimed; outcomes were
reported at the mid and at "conservative" (which was only the natural price);
and the exit side of a trade was never checked at all.
"""
from __future__ import annotations

import copy
import json
from datetime import date, timedelta

import pytest

import scripts.run_forward_loop as forward_loop
from services import claim_contract as cc
from services import evidence_funnel as ef
from services.baseline_evidence_store import BaselineEvidenceStore
from services.evidence_quality import evaluate_evidence_quality
from services.execution_scenarios import build_execution_scenarios, compare_execution_scenarios
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore, make_trade_id
from services.recommendation_ledger import RecommendationLedger

TODAY = date(2026, 4, 21)                       # a Tuesday
CAPTURED = "2026-04-21T20:30:00+00:00"          # 16:30 New York, after the close
CLOSE_QUOTE = "2026-04-21T19:59:40+00:00"       # 15:59:40 New York
CALL = "AAPL260424C00100000"
PUT = "AAPL260424P00100000"
CLEAN_SURFACE = {"status": "clean_surface", "warning_flags": []}


def _quote(*, source="marketdata_app", captured=CAPTURED, legs=None, quality="marketdata_app_paper_research_mid_not_execution_grade"):
    legs = legs if legs is not None else {
        "call": {"contract": CALL, "bid": 2.40, "ask": 2.60, "quote_time": CLOSE_QUOTE},
        "put": {"contract": PUT, "bid": 1.90, "ask": 2.10, "quote_time": "2026-04-21T19:59:55+00:00"},
    }
    for leg in legs.values():
        if leg.get("bid") is not None and leg.get("ask") is not None:
            leg.setdefault("mid", (leg["bid"] + leg["ask"]) / 2.0)
    return {
        "symbol": "AAPL",
        "quote_source": source, "quote_quality": quality, "quote_timestamp": captured,
        "surface_quality": CLEAN_SURFACE, "bid_ask_mid": {"legs": legs},
        "mid": sum(leg.get("mid") or 0 for leg in legs.values()), "context": {},
    }


def _with_leg(quote, leg, **changes):
    out = copy.deepcopy(quote)
    out["bid_ask_mid"]["legs"][leg].update(changes)
    return out


# ── the quote contract ──────────────────────────────────────────────────────


def test_a_clean_closing_marketdata_quote_passes():
    check = cc.evaluate_quote_contract(_quote())
    assert check.passed, check.reasons
    assert check.leg_contracts == {"call": CALL, "put": PUT}
    assert check.max_leg_skew_seconds == 15.0 and check.quote_age_seconds == 20.0


@pytest.mark.parametrize("change, reason", [
    (lambda q: {**q, "quote_source": "yfinance"}, "claim_source_not_live:yfinance"),
    (lambda q: _with_leg(q, "call", contract=None), "claim_call_missing_contract"),
    (lambda q: _with_leg(q, "call", contract="AAPL1260424C00100000"), "claim_call_non_standard_contract"),
    (lambda q: _with_leg(q, "put", bid=0.04, ask=0.10, mid=0.07), "claim_put_bid_below_minimum"),
    (lambda q: _with_leg(q, "put", bid=2.2, ask=2.1), "claim_put_crossed_quote"),
    (lambda q: _with_leg(q, "put", bid=1.0, ask=1.3), "claim_put_spread_too_wide"),
    (lambda q: _with_leg(q, "call", bid=None), "claim_call_missing_bid_ask"),
    (lambda q: _with_leg(q, "call", quote_time=None), "claim_call_missing_quote_time"),
    (lambda q: _with_leg(q, "call", quote_time="2026-04-21T13:00:00+00:00"), "claim_call_quote_outside_session"),
    (lambda q: _with_leg(q, "call", quote_time="2026-04-21T19:58:00+00:00"), "claim_legs_not_synchronized"),
    (lambda q: {**_with_leg(_with_leg(q, "call", quote_time="2026-04-21T19:50:00+00:00"), "put",
                            quote_time="2026-04-21T19:50:10+00:00")}, "claim_quote_stale"),
    (lambda q: {**q, "quote_timestamp": None}, "claim_missing_capture_time"),
    (lambda q: {**q, "bid_ask_mid": {"legs": {}}}, "claim_missing_legs"),
])
def test_each_rule_fails_on_its_own(change, reason):
    check = cc.evaluate_quote_contract(change(_quote()))
    assert not check.passed and reason in check.reasons


def test_quote_age_runs_from_the_capture_during_the_session():
    intraday = _quote(captured="2026-04-21T17:00:00+00:00", legs={
        "call": {"contract": CALL, "bid": 2.4, "ask": 2.6, "quote_time": "2026-04-21T16:57:00+00:00"},
        "put": {"contract": PUT, "bid": 1.9, "ask": 2.1, "quote_time": "2026-04-21T16:57:30+00:00"},
    })
    assert cc.evaluate_quote_contract(intraday).passed
    late = _with_leg(_with_leg(intraday, "call", quote_time="2026-04-21T16:54:00+00:00"), "put",
                     quote_time="2026-04-21T16:54:30+00:00")
    assert "claim_quote_stale" in cc.evaluate_quote_contract(late).reasons


def test_a_quote_from_an_earlier_session_is_not_today_s():
    # Captured on the next morning before the open: the session is the prior
    # day, so yesterday's closing quotes are current; two days back are not.
    morning = _quote(captured="2026-04-22T12:00:00+00:00")
    assert cc.evaluate_quote_contract(morning).passed
    stale_day = _quote(captured="2026-04-23T12:00:00+00:00")
    assert "claim_quote_not_from_session" in cc.evaluate_quote_contract(stale_day).reasons
    weekend = _with_leg(_quote(), "call", quote_time="2026-04-25T15:00:00+00:00")     # Saturday
    assert "claim_call_quote_outside_session" in cc.evaluate_quote_contract(weekend).reasons


# ── entry claim (evidence quality) ──────────────────────────────────────────


def test_a_passing_entry_is_claim_eligible_despite_the_research_label():
    result = evaluate_evidence_quality(quote_payload=_quote(), vol_snapshot={"data_quality_score": 0.9})
    assert result.claim_allowed and result.claim_contract_reasons == []
    assert "quote_not_execution_grade" not in result.evidence_quality_reasons
    # Chain-wide surface warnings degrade the record but not the fill.
    degraded = {**_quote(), "surface_quality": {"status": "degraded_surface", "warning_flags": ["iv_outliers"]}}
    result = evaluate_evidence_quality(quote_payload=degraded)
    assert result.evidence_quality_status == "degraded_evidence" and result.claim_allowed


@pytest.mark.parametrize("snapshot, surface, reason", [
    ({"earnings_source_stale": True}, CLEAN_SURFACE, "stale_earnings_source"),
    ({"earnings_source_confidence": 0.4}, CLEAN_SURFACE, "low_earnings_source_confidence"),
    ({}, {"status": "record_only"}, "surface_quality_record_only"),
])
def test_doubt_about_the_event_blocks_the_claim(snapshot, surface, reason):
    result = evaluate_evidence_quality(quote_payload={**_quote(), "surface_quality": surface}, vol_snapshot=snapshot)
    assert not result.claim_allowed and reason in result.evidence_quality_reasons


def test_a_failed_contract_says_why_and_keeps_the_research_label():
    result = evaluate_evidence_quality(quote_payload=_quote(source="yfinance"))
    assert not result.claim_allowed
    assert {"quote_not_execution_grade", "claim_source_not_live:yfinance"} <= set(result.evidence_quality_reasons)
    assert result.quote_issue_count == len(set(result.evidence_quality_reasons) - set(result.claim_contract_reasons))


# ── the conservative fill ───────────────────────────────────────────────────


def test_three_prices_per_structure():
    entry = build_execution_scenarios(structure="atm_straddle", quote_payload=_quote(), phase="entry").scenario_values
    fee = (0.65 + 0.02) / 100
    assert entry["mid"] == pytest.approx(4.5)
    assert entry["natural"] == entry["cross_50"] == entry["conservative"] == pytest.approx(4.7)
    # Each leg: ask + max(0.01, 10% of a 0.20 spread) = ask + 0.02, plus fees.
    assert entry["conservative_fill"] == pytest.approx(4.7 + 0.04 + 2 * fee)
    exit_ = build_execution_scenarios(structure="atm_straddle", quote_payload=_quote(), phase="exit").scenario_values
    assert exit_["natural"] == pytest.approx(4.3)
    assert exit_["conservative_fill"] == pytest.approx(4.3 - 0.04 - 2 * fee)
    returns = compare_execution_scenarios(
        entry={"scenario_values": entry}, exit={"scenario_values": exit_}, structure="atm_straddle",
    )["realized_return_pct"]
    assert returns["mid"] > returns["natural"] > returns["conservative_fill"]


def test_the_conservative_fill_is_adverse_for_a_credit_structure_too():
    legs = {name: {"contract": f"AAPL260424{'C' if 'call' in name else 'P'}001{i}0000", "bid": 1.0 + i / 10,
                   "ask": 1.1 + i / 10, "quote_time": CLOSE_QUOTE}
            for i, name in enumerate(("short_call", "long_call", "short_put", "long_put"))}
    entry = build_execution_scenarios(structure="iron_condor", quote_payload=_quote(legs=legs), phase="entry")
    values = entry.scenario_values
    # Reported as a positive credit: the conservative fill receives the least.
    assert values["mid"] > values["natural"] > values["conservative_fill"]


# ── the exit side ───────────────────────────────────────────────────────────


def _entry_legs():
    return build_execution_scenarios(structure="atm_straddle", quote_payload=_quote(), phase="entry").legs


def test_the_exit_must_pass_and_reprice_the_booked_contracts():
    assert cc.evaluate_exit_claim(entry_quote_legs=_entry_legs(), exit_quote_payload=_quote())["passed"]
    rolled = _with_leg(_quote(), "call", contract="AAPL260424C00105000")
    verdict = cc.evaluate_exit_claim(entry_quote_legs=_entry_legs(), exit_quote_payload=rolled)
    assert not verdict["passed"] and verdict["reasons"] == ["claim_exit_contracts_differ_from_entry"]
    stale = _quote(captured="2026-04-23T20:30:00+00:00")
    verdict = cc.evaluate_exit_claim(entry_quote_legs=_entry_legs(), exit_quote_payload=stale)
    assert "exit_claim_quote_not_from_session" in verdict["reasons"]
    assert not cc.evaluate_exit_claim(entry_quote_legs={}, exit_quote_payload=_quote())["passed"]


V = cc.CONTRACT_VERSION
ENTRY_VERDICT = {"passed": True, "reasons": [], "contract_version": V}


def _insert(store, trade_id, *, claim_allowed=True, version=cc.CONTRACT_VERSION, entry_verdict=ENTRY_VERDICT):
    store.insert_entry(
        trade_id=trade_id, symbol="AAPL", structure="atm_straddle", entry_date=TODAY - timedelta(days=4),
        setup_score=0.6, source_type="paper", earnings_date=TODAY + timedelta(days=1), entry_mid=4.5,
        execution_penalty_at_entry=0.03, claim_allowed=claim_allowed, evidence_quality_reasons=["entry_reason"],
        claim_contract_version=version, entry_execution_scenarios={"claim": entry_verdict},
    )


@pytest.mark.parametrize("verdict, expected_claim, expected_reasons", [
    ({"passed": True, "reasons": [], "contract_version": V}, 1, ["entry_reason"]),
    ({"passed": False, "reasons": ["exit_claim_quote_stale"], "contract_version": V}, 0,
     ["entry_reason", "exit_claim_quote_stale"]),
    # Fails closed: no verdict, a malformed one, or another version's.
    (None, 0, ["claim_exit_verdict_missing", "entry_reason"]),
    ({"reasons": []}, 0, ["claim_exit_verdict_malformed", "entry_reason"]),
    ({"passed": "yes", "contract_version": V}, 0, ["claim_exit_verdict_malformed", "entry_reason"]),
    ({"passed": True, "reasons": [], "contract_version": "conservative_fill_v1"}, 0,
     ["claim_contract_version_mismatch", "entry_reason"]),
])
def test_the_outcome_store_folds_the_exit_verdict(tmp_path, verdict, expected_claim, expected_reasons):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    _insert(store, "T1")
    scenarios = {"scenario_values": {}} | ({"claim": verdict} if verdict else {})
    for _ in range(2):                                        # idempotent on a recovery re-write
        store.update_exit(trade_id="T1", exit_date=TODAY, exit_mid=5.0, realized_return_pct=11.0,
                          exit_execution_scenarios=scenarios)
    row = store.get_trade("T1")
    assert row["claim_allowed"] == expected_claim
    assert json.loads(row["evidence_quality_reasons_json"]) == expected_reasons


def test_an_ineligible_entry_is_not_made_claimable_by_its_exit(tmp_path):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    _insert(store, "T1", claim_allowed=False)
    store.update_exit(trade_id="T1", exit_date=TODAY, exit_mid=5.0, realized_return_pct=11.0,
                      exit_execution_scenarios={"claim": {"passed": True, "reasons": [], "contract_version": V}})
    assert store.get_trade("T1")["claim_allowed"] == 0


def test_an_entry_checked_under_another_version_is_not_claimable(tmp_path):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    _insert(store, "T1", version=None)                      # e.g. entered before versions were stored
    store.update_exit(trade_id="T1", exit_date=TODAY, exit_mid=5.0, realized_return_pct=11.0,
                      exit_execution_scenarios={"claim": {"passed": True, "reasons": [], "contract_version": V}})
    row = store.get_trade("T1")
    assert row["claim_allowed"] == 0
    assert "claim_contract_version_mismatch" in json.loads(row["evidence_quality_reasons_json"])


def test_the_baseline_store_folds_the_exit_verdict(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    store.insert_entry(
        recommendation_id="rec", symbol="AAPL", baseline_name="always_straddle", structure="atm_straddle",
        entry_date=TODAY - timedelta(days=4), earnings_date=TODAY + timedelta(days=1), selector_structure=None,
        entry_mid=4.5, modeled_cost_pct=None, execution_penalty_at_entry=None, data_quality_score_at_entry=None,
        iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None, quote_source_at_entry="marketdata_app",
        quote_quality_at_entry=None, claim_allowed=True, baseline_id="B1", claim_contract_version=V,
        entry_execution_scenarios={"claim": ENTRY_VERDICT},
    )
    store.update_exit(baseline_id="B1", exit_date=TODAY, exit_mid=5.0, realized_return_pct=11.0,
                      realized_expansion_pct=None, quote_source_at_exit="marketdata_app", quote_quality_at_exit=None,
                      exit_execution_scenarios={"claim": {"passed": False, "reasons": ["exit_claim_quote_stale"]}})
    (row,) = [r for r in store.list_for_diagnostics() if r["baseline_id"] == "B1"]
    assert not row["claim_allowed"]
    assert "exit_claim_quote_stale" in row["evidence_quality_reasons_json"]


# ── end to end: entry, exit, funnel and report ──────────────────────────────


def _finalizer(*, trade_id, store, exit_date, exit_mid, realized_return_pct, exit_execution_scenarios, **_):
    store.update_exit(trade_id=trade_id, exit_date=exit_date, exit_mid=exit_mid,
                      realized_return_pct=realized_return_pct, exit_execution_scenarios=exit_execution_scenarios)
    return {"status": "finalized"}


@pytest.mark.parametrize("exit_quote, claimable", [
    (_quote(), True),
    (_with_leg(_quote(), "put", contract="AAPL260424P00095000"), False),
    (_quote(source="yfinance"), False),
])
def test_a_round_trip_is_claimable_only_when_both_ends_pass(tmp_path, exit_quote, claimable, monkeypatch):
    monkeypatch.setattr(forward_loop, "_today", lambda: TODAY)
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    funnel = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    earnings = TODAY + timedelta(days=1)
    entry_quote = _quote()
    evidence = evaluate_evidence_quality(quote_payload=entry_quote).to_dict()
    trade_id = make_trade_id("AAPL", TODAY - timedelta(days=4), "atm_straddle", earnings_date=earnings)
    store.insert_entry(
        trade_id=trade_id, symbol="AAPL", structure="atm_straddle", entry_date=TODAY - timedelta(days=4),
        setup_score=0.6, source_type="paper", earnings_date=earnings, entry_mid=4.5,
        execution_penalty_at_entry=0.03, notes=json.dumps({"pricing_context": {}}),
        claim_allowed=evidence["claim_allowed"], evidence_quality_reasons=evidence["evidence_quality_reasons"],
        claim_contract_version=evidence["claim_contract_version"],
        entry_execution_scenarios={**build_execution_scenarios(
            structure="atm_straddle", quote_payload=entry_quote, phase="entry").to_dict(),
            "claim": evidence["claim_verdict"]},        # as the forward loop stores it
    )
    assert evidence["claim_allowed"]

    forward_loop.run_exit_detection(
        today=TODAY, store=store, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {**exit_quote, "mid": 5.0}, finalizer=_finalizer,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), funnel=funnel, cycle_run_id="c1",
    )
    (claim_event,) = [e for e in funnel.events("c1") if e["stage"] == "claimable"]
    assert (claim_event["outcome"] == "advanced") is claimable
    row = store.get_trade(trade_id)
    assert bool(row["claim_allowed"]) is claimable
    if not claimable:
        assert "claim_" in claim_event["reason"]

    report = build_forward_performance_diagnostics(
        ledger=RecommendationLedger(ledger_path=tmp_path / "r.sqlite"), outcome_store=store,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )["claimable_evidence"]
    assert report["claim_contract_version"] == cc.CONTRACT_VERSION
    assert report["claimable_count"] == int(claimable)
    if claimable:
        returns = report["claimable_returns"]
        assert set(returns) == {"mid", "natural", "conservative_fill"}
        assert returns["mid"]["avg_realized_return_pct"] > returns["conservative_fill"]["avg_realized_return_pct"]
    funnel.close()


def test_leg_quote_times_are_read_from_the_chain():
    assert forward_loop._leg_quote_time(1776801580) == "2026-04-21T19:59:40+00:00"
    assert forward_loop._leg_quote_time("1776801580") == "2026-04-21T19:59:40+00:00"
    assert forward_loop._leg_quote_time("2026-04-21T15:59:40-04:00") == "2026-04-21T19:59:40+00:00"
    for missing in (None, float("nan"), 0, "", "2026-04-21T15:59:40", True):
        assert forward_loop._leg_quote_time(missing) is None


def test_a_marketdata_chain_quote_carries_leg_contracts_and_times(monkeypatch):
    import pandas as pd

    from tests.unit.test_scripts.test_run_forward_loop import _FakeMarketDataClient

    chain = pd.DataFrame([
        {"optionSymbol": "AAPL260424C00103000", "side": "call", "strike": 103.0, "bid": 1.0, "ask": 1.1,
         "mid": 1.05, "lastPrice": 1.05, "underlyingPrice": 100.0, "updated": 1776801580},
        {"optionSymbol": "AAPL260424P00097000", "side": "put", "strike": 97.0, "bid": 0.9, "ask": 1.0,
         "mid": 0.95, "lastPrice": 0.95, "underlyingPrice": 100.0, "updated": 1776801585},
    ])
    monkeypatch.setattr(forward_loop.yf, "Ticker", lambda _symbol: (_ for _ in ()).throw(AssertionError("no yf")))
    client = _FakeMarketDataClient(chain)
    quote = forward_loop.fetch_structure_quote(symbol="AAPL", structure="otm_strangle",
                                               earnings_date=date(2026, 4, 30), as_of_date=date(2026, 4, 24),
                                               mda_client=client)
    legs = quote["bid_ask_mid"]["legs"]
    assert {leg["contract"] for leg in legs.values()} == {"AAPL260424C00103000", "AAPL260424P00097000"}
    assert sorted(leg["quote_time"] for leg in legs.values()) == [
        "2026-04-21T19:59:40+00:00", "2026-04-21T19:59:45+00:00"]
    assert isinstance(pd.Timestamp(quote["quote_timestamp"]), pd.Timestamp)
    assert quote["symbol"] == "AAPL"                  # the underlying the claim contract binds to
