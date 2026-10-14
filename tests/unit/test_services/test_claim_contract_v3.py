"""PR K: claimable evidence cannot bypass the contract (conservative_fill_v3).

Regression tests for the audit of main at 469c6ff:
- an exit verdict with ``passed=True`` stayed claimable while listing a
  blocking reason (``exit_claim_quote_stale``);
- a string ``reasons`` was consumed character by character;
- the outcome store persisted a caller's ``claim_allowed=True`` with no entry
  verdict at all;
- AAPL contracts quoted four days after they expired passed;
- an AAPL call combined with an MSFT put passed;
- a quote labelled "execution grade" skipped the contract entirely.
"""
from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import scripts.run_forward_loop as forward_loop
from services import claim_contract as cc
from services.baseline_evidence_store import BaselineEvidenceStore
from services.evidence_quality import evaluate_evidence_quality
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger

V = cc.CONTRACT_VERSION
TODAY = date(2026, 4, 21)
CAPTURED = "2026-04-21T20:30:00+00:00"
CALL, PUT = "AAPL260424C00100000", "AAPL260424P00100000"
PASSED = {"passed": True, "reasons": [], "contract_version": V}


def _quote(*, symbol="AAPL", call=CALL, put=PUT, captured=CAPTURED, day="2026-04-21"):
    return {
        "symbol": symbol, "quote_source": "marketdata_app", "quote_timestamp": captured,
        "quote_quality": "marketdata_app_paper_research_mid_not_execution_grade",
        "surface_quality": {"status": "clean_surface", "warning_flags": []},
        "bid_ask_mid": {"legs": {
            "call": {"contract": call, "bid": 2.4, "ask": 2.6, "mid": 2.5, "quote_time": f"{day}T19:59:40+00:00"},
            "put": {"contract": put, "bid": 1.9, "ask": 2.1, "mid": 2.0, "quote_time": f"{day}T19:59:50+00:00"},
        }},
        "mid": 4.5, "context": {},
    }


# ── verdicts are typed and consistent ───────────────────────────────────────


def test_a_passed_verdict_that_lists_a_blocking_reason_is_not_a_pass():
    verdict = {"passed": True, "reasons": ["exit_claim_quote_stale"], "contract_version": V}
    allowed, reasons = cc.apply_exit_claim(
        claim_allowed=True, evidence_reasons=[], exit_execution_scenarios={"claim": verdict},
        entry_contract_version=V, entry_execution_scenarios={"claim": PASSED})
    assert allowed is False
    assert {"claim_exit_verdict_inconsistent", "exit_claim_quote_stale"} <= set(reasons)


def test_a_failed_verdict_with_no_reason_is_inconsistent_too():
    assert cc.verdict_problems({"passed": False, "reasons": [], "contract_version": V}, side="exit") == [
        "claim_exit_verdict_inconsistent"]


@pytest.mark.parametrize("verdict", [
    {"passed": False, "reasons": "exit_claim_quote_stale", "contract_version": V},   # a string, not a list
    {"passed": True, "reasons": [1], "contract_version": V},                         # not strings
    {"passed": "true", "reasons": [], "contract_version": V},
    {"passed": True, "contract_version": V},                                         # no reasons at all
])
def test_malformed_verdicts_fail_closed_without_reading_strings_as_lists(verdict):
    allowed, reasons = cc.apply_exit_claim(
        claim_allowed=True, evidence_reasons=[], exit_execution_scenarios={"claim": verdict},
        entry_contract_version=V, entry_execution_scenarios={"claim": PASSED})
    assert allowed is False and "claim_exit_verdict_malformed" in reasons
    assert all(len(reason) > 1 for reason in reasons)          # never one character per "reason"


def test_a_clean_round_trip_still_passes():
    allowed, reasons = cc.apply_exit_claim(
        claim_allowed=True, evidence_reasons=["kept"], exit_execution_scenarios={"claim": PASSED},
        entry_contract_version=V, entry_execution_scenarios={"claim": PASSED})
    assert (allowed, reasons) == (True, ["kept"])


@pytest.mark.parametrize("entry, problem", [
    (None, "claim_entry_verdict_missing"),
    ({"passed": True, "reasons": ["claim_put_spread_too_wide"], "contract_version": V},
     "claim_entry_verdict_inconsistent"),
    ({"passed": False, "reasons": ["claim_put_spread_too_wide"], "contract_version": V}, "claim_put_spread_too_wide"),
    ({"passed": True, "reasons": [], "contract_version": "conservative_fill_v2"}, "claim_contract_version_mismatch"),
])
def test_the_exit_fold_needs_a_sound_entry_verdict_too(entry, problem):
    allowed, reasons = cc.apply_exit_claim(
        claim_allowed=True, evidence_reasons=[], exit_execution_scenarios={"claim": PASSED},
        entry_contract_version=V, entry_execution_scenarios={"claim": entry} if entry else {})
    assert allowed is False and problem in reasons


# ── the stores derive the claim from the verdict ────────────────────────────


def _insert_outcome(store, trade_id, **kwargs):
    store.insert_entry(trade_id=trade_id, symbol="AAPL", structure="atm_straddle", entry_date=TODAY,
                       setup_score=0.6, source_type="paper", earnings_date=TODAY + timedelta(days=2),
                       entry_mid=4.5, **kwargs)
    return store.get_trade(trade_id)


def test_the_outcome_store_does_not_take_a_callers_claim_on_trust(tmp_path):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    bare = _insert_outcome(store, "T_BARE", claim_allowed=True, claim_contract_version=V)
    assert bare["claim_allowed"] == 0
    assert "claim_entry_verdict_missing" in json.loads(bare["evidence_quality_reasons_json"])

    failing = _insert_outcome(store, "T_FAIL", claim_allowed=True, claim_contract_version=V,
                              evidence_quality_reasons=["x"],
                              entry_execution_scenarios={"claim": {"passed": False, "reasons": ["claim_quote_stale"],
                                                                   "contract_version": V}})
    assert failing["claim_allowed"] == 0
    assert {"x", "claim_quote_stale"} <= set(json.loads(failing["evidence_quality_reasons_json"]))

    old = _insert_outcome(store, "T_OLD", claim_allowed=True, claim_contract_version="conservative_fill_v2",
                          entry_execution_scenarios={"claim": PASSED})
    assert old["claim_allowed"] == 0

    sound = _insert_outcome(store, "T_OK", claim_allowed=True, claim_contract_version=V,
                            entry_execution_scenarios={"claim": PASSED})
    assert sound["claim_allowed"] == 1

    # Not claim-eligible stays as the caller said (False, or never classified).
    assert _insert_outcome(store, "T_NO", claim_allowed=False)["claim_allowed"] == 0
    assert _insert_outcome(store, "T_NONE")["claim_allowed"] is None


def test_the_baseline_store_does_not_take_a_callers_claim_on_trust(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    for baseline_id, scenarios in (("B_BARE", None), ("B_OK", {"claim": PASSED})):
        store.insert_entry(
            recommendation_id=baseline_id, symbol="AAPL", baseline_name="always_straddle", structure="atm_straddle",
            entry_date=TODAY, earnings_date=TODAY + timedelta(days=2), selector_structure=None, entry_mid=4.5,
            modeled_cost_pct=None, execution_penalty_at_entry=None, data_quality_score_at_entry=None,
            iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None, quote_source_at_entry="marketdata_app",
            quote_quality_at_entry=None, claim_allowed=True, baseline_id=baseline_id, claim_contract_version=V,
            entry_execution_scenarios=scenarios,
        )
    rows = {row["baseline_id"]: row for row in store.list_for_diagnostics()}
    assert {key: row["claim_allowed"] for key, row in rows.items()} == {"B_BARE": 0, "B_OK": 1}
    reasons = rows["B_BARE"]["evidence_quality_reasons_json"]
    assert "claim_entry_verdict_missing" in (reasons if isinstance(reasons, list) else json.loads(reasons))


def test_a_stored_entry_verdict_that_cannot_be_read_fails_the_exit(tmp_path):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    _insert_outcome(store, "T1", claim_allowed=True, claim_contract_version=V,
                    entry_execution_scenarios={"claim": PASSED})
    store._conn.execute("UPDATE outcome_trades SET entry_execution_scenarios_json = '{broken' WHERE trade_id = 'T1'")
    store._conn.commit()
    store.update_exit(trade_id="T1", exit_date=TODAY + timedelta(days=3), exit_mid=5.0, realized_return_pct=10.0,
                      exit_execution_scenarios={"claim": PASSED})
    row = store.get_trade("T1")
    assert row["claim_allowed"] == 0
    assert "claim_entry_verdict_missing" in json.loads(row["evidence_quality_reasons_json"])


def test_a_baseline_whose_stored_entry_verdict_is_unreadable_fails_its_exit(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    store.insert_entry(
        recommendation_id="rec", symbol="AAPL", baseline_name="always_straddle", structure="atm_straddle",
        entry_date=TODAY, earnings_date=TODAY + timedelta(days=2), selector_structure=None, entry_mid=4.5,
        modeled_cost_pct=None, execution_penalty_at_entry=None, data_quality_score_at_entry=None,
        iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None, quote_source_at_entry="marketdata_app",
        quote_quality_at_entry=None, claim_allowed=True, baseline_id="B1", claim_contract_version=V,
        entry_execution_scenarios={"claim": PASSED},
    )
    store._conn.execute("UPDATE baseline_trades SET entry_execution_scenarios_json = '{broken' WHERE baseline_id = 'B1'")
    store._conn.commit()
    store.update_exit(baseline_id="B1", exit_date=TODAY + timedelta(days=3), exit_mid=5.0, realized_return_pct=10.0,
                      realized_expansion_pct=None, quote_source_at_exit="marketdata_app", quote_quality_at_exit=None,
                      exit_execution_scenarios={"claim": PASSED})
    (row,) = store.list_for_diagnostics()
    assert row["claim_allowed"] == 0


# ── contract identity is bound to the trade ─────────────────────────────────


def test_contracts_quoted_after_they_expired_fail():
    # The audit's repro: AAPL contracts expiring 04-24, quoted four days later.
    late = cc.evaluate_quote_contract(_quote(captured="2026-04-28T20:30:00+00:00", day="2026-04-28"))
    assert not late.passed
    assert {"claim_call_contract_expired", "claim_put_contract_expired"} <= set(late.reasons)
    # Expired the day before the quote's session.
    yesterday = cc.evaluate_quote_contract(_quote(call="AAPL260423C00100000", put="AAPL260423P00100000",
                                                  captured="2026-04-24T20:30:00+00:00", day="2026-04-24"))
    assert "claim_call_contract_expired" in yesterday.reasons
    # Expiration day itself is still a live contract.
    on_expiry = cc.evaluate_quote_contract(_quote(captured="2026-04-24T20:30:00+00:00", day="2026-04-24"))
    assert on_expiry.passed, on_expiry.reasons


def test_legs_on_another_underlying_fail():
    mixed = cc.evaluate_quote_contract(_quote(put="MSFT260424P00100000"))
    assert not mixed.passed and "claim_put_contract_not_on_underlying" in mixed.reasons
    # The recommendation's symbol wins over whatever the quote says it is.
    other = cc.evaluate_quote_contract(_quote(symbol="MSFT"), expected_underlying="AAPL")
    assert other.passed
    wrong = cc.evaluate_quote_contract(_quote(), expected_underlying="MSFT")
    assert {"claim_call_contract_not_on_underlying", "claim_put_contract_not_on_underlying"} <= set(wrong.reasons)


def test_a_quote_with_no_underlying_cannot_be_bound_and_fails():
    quote = _quote()
    del quote["symbol"]
    check = cc.evaluate_quote_contract(quote)
    assert not check.passed and "claim_missing_underlying" in check.reasons


def test_occ_roots_drop_share_class_punctuation():
    assert cc.occ_root("brk.b") == "BRKB"
    check = cc.evaluate_quote_contract(_quote(call="BRKB260424C00100000", put="BRKB260424P00100000"),
                                       expected_underlying="BRK.B")
    assert check.passed, check.reasons


def test_the_exit_is_bound_to_the_trades_underlying():
    msft_legs = _quote(call="MSFT260424C00100000", put="MSFT260424P00100000")["bid_ask_mid"]["legs"]
    exit_quote = _quote(symbol="MSFT", call="MSFT260424C00100000", put="MSFT260424P00100000")
    verdict = cc.evaluate_exit_claim(entry_quote_legs=msft_legs, exit_quote_payload=exit_quote,
                                     expected_underlying="AAPL")
    assert not verdict["passed"] and "exit_claim_call_contract_not_on_underlying" in verdict["reasons"]


# ── evidence quality: no label bypass, and the verdict travels ──────────────


def test_an_execution_grade_label_does_not_skip_the_contract():
    labelled = {**_quote(put="MSFT260424P00100000"), "quote_quality": "execution_grade"}
    result = evaluate_evidence_quality(quote_payload=labelled, symbol="AAPL")
    assert result.execution_grade and not result.claim_allowed
    assert result.claim_verdict == {
        "passed": False, "reasons": ["claim_put_contract_not_on_underlying"], "contract_version": V,
        "leg_contracts": {"call": CALL, "put": "MSFT260424P00100000"}}


def test_evidence_quality_binds_the_named_symbol():
    msft = _quote(symbol=None, call="MSFT260424C00100000", put="MSFT260424P00100000")
    assert evaluate_evidence_quality(quote_payload=msft, symbol="MSFT").claim_allowed
    assert not evaluate_evidence_quality(quote_payload=msft, symbol="AAPL").claim_allowed
    # Without a named symbol, the snapshot's is used.
    assert evaluate_evidence_quality(quote_payload=msft, vol_snapshot={"symbol": "MSFT"}).claim_allowed


def test_the_forward_loop_stores_the_entry_verdict_with_the_trade(tmp_path, monkeypatch):
    monkeypatch.setattr(forward_loop, "_today", lambda: TODAY)
    earnings = TODAY + timedelta(days=6)
    analysis = SimpleNamespace(
        symbol="AAPL", recommendation="Candidate", confidence_pct=70.0, setup_score=0.6, metrics={},
        rationale=["ok"],
        selector_output={"recommendation": "Candidate", "best_structure": "atm_straddle",
                         "earnings_date": earnings.isoformat(), "binding_gate": None},
        structure_scorecards=[{"structure": "atm_straddle", "execution_penalty": 0.03}],
        vol_snapshot={"as_of_date": TODAY.isoformat(), "earnings_date": earnings.isoformat(),
                      "data_quality_score": 0.9},
    )
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    quote = _quote()
    del quote["symbol"]                              # bound by the loop, not by the payload
    summary = forward_loop.run_forward_screener(
        today=TODAY, store=store, ledger=RecommendationLedger(ledger_path=tmp_path / "r.sqlite"),
        baseline_store=baselines, log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": [{"symbol": "AAPL", "dte": 6, "earnings_date": earnings.isoformat()}]},
        analyzer=lambda symbol, mda_client=None: analysis,
        price_fetcher=lambda **_: copy.deepcopy(quote), symbols=["AAPL"], cycle_run_id="c1",
    )
    assert summary["entries"] == 1
    (row,) = store.list_for_diagnostics(limit=10)
    assert row["claim_allowed"] == 1 and row["claim_contract_version"] == V
    scenarios = row["entry_execution_scenarios_json"]
    scenarios = json.loads(scenarios) if isinstance(scenarios, str) else scenarios
    assert scenarios["claim"]["passed"] is True and scenarios["claim"]["reasons"] == []
    # The shadow baselines priced from the same quote carry their verdict too.
    shadow = baselines.list_for_diagnostics()
    assert summary["baseline_entries"] >= 1 and shadow
    assert all(row["claim_allowed"] == 1 for row in shadow), [row["evidence_quality_reasons_json"] for row in shadow]


def _finalizer(*, trade_id, store, exit_date, exit_mid, realized_return_pct, exit_execution_scenarios, **_):
    store.update_exit(trade_id=trade_id, exit_date=exit_date, exit_mid=exit_mid,
                      realized_return_pct=realized_return_pct, exit_execution_scenarios=exit_execution_scenarios)
    return {"status": "finalized"}


@pytest.mark.parametrize("exit_symbol, legs_root, claimable", [
    (None, "AAPL", True),          # the payload names no symbol: the trade's binds it
    ("MSFT", "MSFT", False),       # a quote for another underlying never passes for an AAPL trade
])
def test_the_exit_pass_binds_the_trades_symbol(tmp_path, monkeypatch, exit_symbol, legs_root, claimable):
    from services.outcome_recorder import make_trade_id

    monkeypatch.setattr(forward_loop, "_today", lambda: TODAY)
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    earnings = TODAY + timedelta(days=1)          # exits the session before the event
    entry_legs = _quote(call=f"{legs_root}260424C00100000", put=f"{legs_root}260424P00100000")["bid_ask_mid"]["legs"]
    trade_id = make_trade_id("AAPL", TODAY - timedelta(days=4), "atm_straddle", earnings_date=earnings)
    store.insert_entry(
        trade_id=trade_id, symbol="AAPL", structure="atm_straddle", entry_date=TODAY - timedelta(days=4),
        setup_score=0.6, source_type="paper", earnings_date=earnings, entry_mid=4.5, execution_penalty_at_entry=0.03,
        notes=json.dumps({"pricing_context": {}}), claim_allowed=True, claim_contract_version=V,
        entry_execution_scenarios={"legs": entry_legs, "claim": PASSED},
    )
    exit_quote = _quote(symbol=exit_symbol, call=f"{legs_root}260424C00100000", put=f"{legs_root}260424P00100000")
    if exit_symbol is None:
        del exit_quote["symbol"]
    forward_loop.run_exit_detection(
        today=TODAY, store=store, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {**exit_quote, "mid": 5.0}, finalizer=_finalizer,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), cycle_run_id="c1",
    )
    row = store.get_trade(trade_id)
    assert row["exit_date"] is not None
    assert bool(row["claim_allowed"]) is claimable
