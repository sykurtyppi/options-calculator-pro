"""Claim-contract findings from the independent audit of main at bfc48f5.

1. A claimable iron condor whose conservative fill opens for a net debit had
   its conservative return withheld, so it dropped out of the claimed figure
   and flattered it.
2. A baseline whose first T-1 exit attempt was skipped lost its claim for good,
   even when the same-day retry resolved with a passing exit verdict.
3. The report counted open rows as claimable evidence, hiding the "no
   claimable evidence yet" warning.
4. The funnel's claimable stage used the raw flag, not the reports' rule.
5. The stores recorded a claim next to a claim-disqualifying evidence reason,
   or from a verdict that was not made for the booked legs and source.
6. An out-of-range epoch quote time raised instead of failing the contract.
7. ``True``/``False`` were read as prices 1.0/0.0.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

import scripts.run_forward_loop as forward_loop
from services.baseline_evidence_store import BaselineEvidenceStore
from services.claim_contract import (
    CONTRACT_VERSION,
    claimable_under_current_contract,
    entry_claim_status,
    evaluate_exit_claim,
    evaluate_quote_contract,
)
from services.evidence_funnel import EvidenceFunnelStore
from services.evidence_quality import evaluate_evidence_quality
from services.evidence_report import _evidence_quality_summary
from services.execution_scenarios import build_execution_scenarios, compare_execution_scenarios
from services.forward_performance_diagnostics import _claimable_returns
from services.outcome_recorder import OutcomeStore

CAPTURE = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, a Wednesday


def _leg(contract, bid, ask, t=CAPTURE - timedelta(seconds=30)):
    return {"contract": contract, "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "quote_time": t.isoformat()}


def _quote(legs, source="marketdata_app"):
    return {"symbol": "AAPL", "quote_source": source, "quote_timestamp": CAPTURE.isoformat(),
            "quote_quality": "live", "surface_quality": {"status": "clean_surface"},
            "bid_ask_mid": {"legs": legs}}


def _calendar():
    return _quote({"front": _leg("AAPL261016C00250000", 2.00, 2.20),
                   "back": _leg("AAPL261120C00250000", 5.00, 5.30)})


def _thin_condor():
    return _quote({"short_call": _leg("AAPL261120C00260000", 0.50, 0.60),
                   "long_call": _leg("AAPL261120C00262500", 0.40, 0.48),
                   "short_put": _leg("AAPL261120P00240000", 0.50, 0.60),
                   "long_put": _leg("AAPL261120P00237500", 0.40, 0.48)})


# ── 1. debit-fill condors stay in the claimed conservative figure ───────────

def test_a_condor_whose_conservative_fill_is_a_debit_keeps_its_return():
    quote = _thin_condor()
    assert evaluate_quote_contract(quote, expected_underlying="AAPL").passed
    entry = build_execution_scenarios(structure="iron_condor", quote_payload=quote, phase="entry").to_dict()
    exit_ = build_execution_scenarios(structure="iron_condor", quote_payload=quote, phase="exit").to_dict()
    assert entry["scenario_values"]["conservative_fill"] < 0          # the fill pays to open
    out = compare_execution_scenarios(entry=entry, exit=exit_, structure="iron_condor", capital_at_risk=2.28)
    conservative = out["realized_return_pct"]["conservative_fill"]
    # Risked: the mid-fill max loss plus the credit the fill gave up.
    base = 2.28 + entry["scenario_values"]["mid"] - entry["scenario_values"]["conservative_fill"]
    pnl = entry["scenario_values"]["conservative_fill"] - exit_["scenario_values"]["conservative_fill"]
    assert conservative == pytest.approx(pnl / base * 100.0, abs=1e-5)
    assert conservative < out["realized_return_pct"]["natural"] < out["realized_return_pct"]["mid"]

    good = {"claim_allowed": True,
            "execution_scenario_returns": {"mid": 5.0, "natural": 2.0, "conservative_fill": 1.0}}
    thin = {"claim_allowed": True, "execution_scenario_returns": out["realized_return_pct"]}
    stats = _claimable_returns([good, thin])
    assert {level: stats[level]["n"] for level in stats} == {level: 2 for level in stats}
    assert all(stats[level]["complete"] and stats[level]["withheld_count"] == 0 for level in stats)


def test_a_level_missing_for_a_claimable_row_is_reported_withheld():
    rows = [{"claim_allowed": True, "execution_scenario_returns": {"mid": 5.0, "natural": 2.0}},
            {"claim_allowed": True,
             "execution_scenario_returns": {"mid": 1.0, "natural": 0.5, "conservative_fill": -3.0}}]
    stats = _claimable_returns(rows)
    assert stats["conservative_fill"]["n"] == 1
    assert stats["conservative_fill"]["withheld_count"] == 1 and stats["conservative_fill"]["complete"] is False
    assert stats["mid"]["complete"] is True


def test_a_debit_fill_without_a_mid_still_withholds_rather_than_guess():
    entry = {"scenario_values": {"conservative_fill": -0.05}}
    exit_ = {"scenario_values": {"conservative_fill": 0.30}}
    out = compare_execution_scenarios(entry=entry, exit=exit_, structure="iron_condor", capital_at_risk=2.0)
    assert out["realized_return_pct"]["conservative_fill"] is None
    assert out["realized_pnl"]["conservative_fill"] == pytest.approx(-35.0)


# ── 2. a skipped baseline exit does not burn the entry claim ────────────────

def _baseline_entry(store):
    quote = _calendar()
    quality = evaluate_evidence_quality(quote_payload=quote, symbol="AAPL")
    assert quality.claim_allowed
    entry = build_execution_scenarios(structure="call_calendar", quote_payload=quote, phase="entry").to_dict()
    entry["claim"] = quality.claim_verdict
    store.insert_entry(recommendation_id="r1", symbol="AAPL", baseline_name="b", structure="call_calendar",
                       entry_date=date(2026, 10, 14), earnings_date=date(2026, 10, 16), selector_structure=None,
                       entry_mid=3.05, modeled_cost_pct=None, execution_penalty_at_entry=None,
                       data_quality_score_at_entry=None, iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None,
                       quote_source_at_entry="marketdata_app", quote_quality_at_entry="live",
                       claim_allowed=quality.claim_allowed, claim_contract_version=quality.claim_contract_version,
                       entry_execution_scenarios=entry)
    return "r1|baseline|b", entry


def _skip(store, baseline_id):
    store.update_exit(baseline_id=baseline_id, exit_date=date(2026, 10, 15), exit_mid=None,
                      realized_return_pct=None, realized_expansion_pct=None,
                      quote_source_at_exit="marketdata_app", quote_quality_at_exit="live",
                      exit_execution_scenarios={}, status="exit_skipped", skip_reason="missing_exit_mid")


def test_a_same_day_retry_after_a_skipped_baseline_exit_stays_claimable(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    baseline_id, entry = _baseline_entry(store)
    _skip(store, baseline_id)
    assert store.list_for_diagnostics()[0]["claim_allowed"] == 1
    verdict = evaluate_exit_claim(entry_quote_legs=entry["legs"], exit_quote_payload=_calendar(),
                                  expected_underlying="AAPL")
    store.update_exit(baseline_id=baseline_id, exit_date=date(2026, 10, 15), exit_mid=3.4,
                      realized_return_pct=10.0, realized_expansion_pct=10.0,
                      quote_source_at_exit="marketdata_app", quote_quality_at_exit="live",
                      exit_execution_scenarios={"claim": verdict}, status="resolved",
                      exit_repricing="booked_strike")
    row = store.list_for_diagnostics()[0]
    assert (row["status"], row["claim_allowed"], json.loads(row["evidence_quality_reasons_json"])) == (
        "resolved", 1, [])


def test_a_retry_that_resolves_without_an_exit_verdict_still_loses_the_claim(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    baseline_id, _ = _baseline_entry(store)
    _skip(store, baseline_id)
    store.update_exit(baseline_id=baseline_id, exit_date=date(2026, 10, 15), exit_mid=3.4,
                      realized_return_pct=10.0, realized_expansion_pct=10.0,
                      quote_source_at_exit="marketdata_app", quote_quality_at_exit="live",
                      exit_execution_scenarios={}, status="resolved", exit_repricing="booked_strike")
    row = store.list_for_diagnostics()[0]
    assert row["claim_allowed"] == 0
    assert "claim_exit_verdict_missing" in json.loads(row["evidence_quality_reasons_json"])


# ── 3. open rows are not claimable evidence ─────────────────────────────────

def test_the_report_counts_claims_only_on_resolved_rows():
    claimed = {"claim_allowed": 1, "claim_contract_version": CONTRACT_VERSION}
    open_outcome = {**claimed, "status": "open", "realized_return_pct": None}
    open_baseline = {**claimed, "status": "open"}
    skipped_baseline = {**claimed, "status": "exit_skipped"}
    summary = _evidence_quality_summary([open_outcome], [open_baseline, skipped_baseline])
    assert summary["claim_allowed_count"] == 0
    assert summary["claim_eligible_unresolved_count"] == 3

    resolved_outcome = {**claimed, "status": "finalized", "realized_return_pct": 3.0}
    resolved_baseline = {**claimed, "status": "resolved"}
    blocked = {"claim_allowed": 0, "claim_contract_version": CONTRACT_VERSION, "status": "resolved"}
    summary = _evidence_quality_summary([resolved_outcome, open_outcome], [resolved_baseline, blocked])
    assert (summary["claim_allowed_count"], summary["claim_blocked_count"],
            summary["claim_eligible_unresolved_count"]) == (2, 1, 1)


# ── 4. the funnel's claimable stage follows the reports' rule ───────────────

def test_the_funnel_does_not_count_a_claim_made_under_an_older_contract(tmp_path):
    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    funnel = EvidenceFunnelStore(tmp_path / "f.sqlite")
    store.insert_entry(trade_id="t1", symbol="AAPL", structure="call_calendar", entry_date=date(2026, 10, 12),
                       setup_score=1.0, source_type="paper", earnings_date=date(2026, 10, 15), entry_mid=3.0)
    store._conn.execute(
        "UPDATE outcome_trades SET claim_allowed=1, claim_contract_version='conservative_fill_v2',"
        " exit_date='2026-10-14', exit_mid=3.3, realized_return_pct=10.0, realized_expansion_pct=10.0,"
        " status='exited' WHERE trade_id='t1'")
    store._conn.commit()
    assert claimable_under_current_contract(store.get_trade("t1")) is False
    summary = forward_loop._run_exit_detection(
        today=date(2026, 10, 19), store=store, funnel=funnel,
        finalizer=lambda **_: {"status": "finalized", "learning_update_status": "complete"},
        price_fetcher=lambda **_: {"mid": None}, baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
        log_path=tmp_path / "log.jsonl")
    assert summary["refinalized"] == 1
    stage = funnel.summarize()["stages"]["claimable"]
    assert (stage["advanced"], stage["dropped"]) == (0, 1)
    reasons = [row[0] for row in funnel._conn.execute(
        "SELECT reason FROM funnel_events WHERE stage = 'claimable'")]
    assert reasons == ["not_claimable:claim_contract_version_mismatch"]


# ── 5. the stores bind a claim to the evidence they hold ────────────────────

def _calendar_entry():
    quote = _calendar()
    entry = build_execution_scenarios(structure="call_calendar", quote_payload=quote, phase="entry").to_dict()
    entry["claim"] = evaluate_evidence_quality(quote_payload=quote, symbol="AAPL").claim_verdict
    return entry


def _insert_outcome(store, **overrides):
    kwargs = dict(trade_id="t1", symbol="AAPL", structure="call_calendar", entry_date=date(2026, 10, 14),
                  setup_score=1.0, source_type="paper", earnings_date=date(2026, 10, 16), entry_mid=3.05,
                  claim_allowed=True, claim_contract_version=CONTRACT_VERSION,
                  entry_quote_source="marketdata_app", entry_execution_scenarios=_calendar_entry())
    kwargs.update(overrides)
    store.insert_entry(**kwargs)
    return store.get_trade("t1")


def test_a_clean_bound_entry_is_recorded_claimable(tmp_path):
    row = _insert_outcome(OutcomeStore(store_path=tmp_path / "o.sqlite"))
    assert row["claim_allowed"] == 1


@pytest.mark.parametrize("reason", ["stale_earnings_source", "low_earnings_source_confidence",
                                    "provider_research_grade_yfinance", "missing_quote_source"])
def test_a_disqualifying_evidence_reason_withdraws_the_claim(tmp_path, reason):
    row = _insert_outcome(OutcomeStore(store_path=tmp_path / "o.sqlite"), evidence_quality_reasons=[reason])
    assert row["claim_allowed"] == 0 and reason in json.loads(row["evidence_quality_reasons_json"])


def test_a_baseline_with_a_disqualifying_reason_is_not_claimable(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    store.insert_entry(recommendation_id="r1", symbol="AAPL", baseline_name="b", structure="call_calendar",
                       entry_date=date(2026, 10, 14), earnings_date=date(2026, 10, 16), selector_structure=None,
                       entry_mid=3.05, modeled_cost_pct=None, execution_penalty_at_entry=None,
                       data_quality_score_at_entry=None, iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None,
                       quote_source_at_entry="marketdata_app", quote_quality_at_entry="live",
                       evidence_quality_reasons=["stale_earnings_source"], claim_allowed=True,
                       claim_contract_version=CONTRACT_VERSION, entry_execution_scenarios=_calendar_entry())
    assert store.list_for_diagnostics()[0]["claim_allowed"] == 0


def test_a_hand_written_verdict_for_a_non_live_quote_is_refused(tmp_path):
    forged = {"passed": True, "reasons": [], "contract_version": CONTRACT_VERSION}
    row = _insert_outcome(OutcomeStore(store_path=tmp_path / "o.sqlite"), entry_quote_source="yfinance",
                          entry_execution_scenarios={"claim": forged})
    reasons = json.loads(row["evidence_quality_reasons_json"])
    assert row["claim_allowed"] == 0
    assert {"claim_entry_verdict_not_for_booked_legs", "claim_source_not_live:yfinance"} <= set(reasons)


def test_a_verdict_for_other_contracts_than_the_booked_legs_is_refused():
    entry = _calendar_entry()
    entry["claim"] = {**entry["claim"], "leg_contracts": {"front": "AAPL261016C00255000",
                                                          "back": "AAPL261120C00250000"}}
    allowed, problems = entry_claim_status(claim_allowed=True, claim_contract_version=CONTRACT_VERSION,
                                           entry_execution_scenarios=entry)
    assert allowed is False and problems == ["claim_entry_verdict_not_for_booked_legs"]
    entry.pop("legs")
    entry["claim"] = _calendar_entry()["claim"]
    assert entry_claim_status(claim_allowed=True, claim_contract_version=CONTRACT_VERSION,
                              entry_execution_scenarios=entry)[0] is False


# ── 6/7. malformed times and bools fail the contract instead of raising ─────

@pytest.mark.parametrize("field, value", [
    ("leg", int(CAPTURE.timestamp() * 1000)),     # unix milliseconds
    ("leg", 1e20),
    ("capture", int(CAPTURE.timestamp() * 1000)),
])
def test_an_out_of_range_epoch_fails_the_contract(field, value):
    quote = _calendar()
    if field == "leg":
        quote["bid_ask_mid"]["legs"]["front"]["quote_time"] = value
    else:
        quote["quote_timestamp"] = value
    check = evaluate_quote_contract(quote, expected_underlying="AAPL")
    assert check.passed is False
    expected = "claim_front_missing_quote_time" if field == "leg" else "claim_missing_capture_time"
    assert expected in check.reasons
    verdict = evaluate_exit_claim(entry_quote_legs=_calendar()["bid_ask_mid"]["legs"], exit_quote_payload=quote,
                                  expected_underlying="AAPL")
    assert verdict["passed"] is False


def test_bool_prices_are_not_prices():
    quote = _calendar()
    quote["bid_ask_mid"]["legs"]["front"].update(bid=True, ask=True)
    check = evaluate_quote_contract(quote, expected_underlying="AAPL")
    assert "claim_front_missing_bid_ask" in check.reasons
    out = compare_execution_scenarios(entry={"scenario_values": {"mid": True}},
                                      exit={"scenario_values": {"mid": 2.0}})
    assert out["realized_return_pct"]["mid"] is None and out["realized_pnl"]["mid"] is None


def test_the_exit_fold_withdraws_a_legacy_claim_stored_beside_a_disqualifying_reason():
    # Rows stored before the entry check carried both; the exit fold must not pass them.
    from services.claim_contract import apply_exit_claim

    entry = _calendar_entry()
    verdict = evaluate_exit_claim(entry_quote_legs=entry["legs"], exit_quote_payload=_calendar(),
                                  expected_underlying="AAPL")
    fold = dict(claim_allowed=True, exit_execution_scenarios={"claim": verdict},
                entry_contract_version=CONTRACT_VERSION, entry_execution_scenarios=entry)
    assert apply_exit_claim(evidence_reasons=[], **fold) == (True, [])
    assert apply_exit_claim(evidence_reasons=["stale_earnings_source"], **fold) == (False, ["stale_earnings_source"])
