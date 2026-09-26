"""Follow-up evidence-integrity regressions (review of #144 at 3acc626).

1. Partial / legacy entry contexts are never certified as booked exits.
2. A T-1 exit that cannot be priced becomes terminal 'exit_missing' attrition.
3. Invalidation is enforced at every outcome mutation, race-free.
4. Invalid rows are filtered before the report row limit.
5. Nested notes invalidation reasons survive into audit output.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

import pytest

import scripts.run_forward_loop as forward_loop
import services.outcome_recorder as outcome_recorder
from scripts.invalidate_outcome import main as invalidate_cli
from services.baseline_evidence_store import (
    EXIT_REPRICING_BOOKED,
    EXIT_REPRICING_UNVERIFIABLE,
    BaselineEvidenceStore,
    make_baseline_id,
)
from services.evidence_report import build_evidence_report
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import (
    _VALID_EVIDENCE_SQL,
    OutcomeStore,
    finalize_trade_and_update_learning,
    is_outcome_evidence_valid,
    outcome_invalidation_reason,
    record_trade_exit,
)
from services.recommendation_ledger import RecommendationLedger

EXPIRY = "2026-05-01"
T_MINUS_1 = date(2026, 4, 27)
EARNINGS = T_MINUS_1 + timedelta(days=1)
NESTED = {
    "evidence_invalidated": {
        "invalidated_at": "2026-05-02T00:00:00Z",
        "reason_code": "exit_repriced_unheld_strike",
        "reason": "exit priced the 265 call; the 260 was booked",
    }
}

STRANGLE = {"call_strike": 103.0, "put_strike": 97.0, "front_expiry": EXPIRY}
CONDOR = {
    "short_call_strike": 105.0, "long_call_strike": 110.0,
    "short_put_strike": 95.0, "long_put_strike": 90.0, "front_expiry": EXPIRY,
}
STRADDLE = {"strike": 100.0, "straddle_put_strike": 100.0, "front_expiry": EXPIRY}


# ── 1. contract identity ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("structure", "entry", "exit_"),
    [
        # Reviewer probes: one strike present is no longer enough.
        ("otm_strangle", {"call_strike": 103.0, "front_expiry": EXPIRY}, {**STRANGLE, "put_strike": 90.0}),
        ("iron_condor", {"short_call_strike": 105.0, "front_expiry": EXPIRY}, {**CONDOR, "long_put_strike": 1.0}),
        ("otm_strangle", {"call_strike": 103.0, "put_strike": 97.0}, {**STRANGLE, "front_expiry": "2026-05-08"}),
        # Legacy straddle: C100 / P105 booked, only the call strike recorded.
        (
            "atm_straddle",
            {"strike": 100.0, "call_contract": "C100", "put_contract": "P105", "front_expiry": EXPIRY},
            {**STRADDLE, "call_contract": "C100", "put_contract": "P100"},
        ),
    ],
)
def test_partial_or_legacy_entry_context_is_never_certified(structure, entry, exit_):
    assert forward_loop._entry_context_verifiable(structure, entry) is False
    assert forward_loop._booked_contracts_match(structure, entry, exit_) is False


def test_contract_symbol_mismatch_is_rejected_when_both_sides_recorded_it():
    entry = {**STRADDLE, "call_contract": "AAPL260501C00100000", "put_contract": "AAPL260501P00100000"}
    same = {**entry, "put_contract": " aapl260501p00100000 "}
    other = {**entry, "put_contract": "AAPL260508P00100000"}
    no_symbol = {**STRADDLE, "put_contract": "None"}

    assert forward_loop._booked_contracts_match("atm_straddle", entry, same)
    assert not forward_loop._booked_contracts_match("atm_straddle", entry, other)
    # A symbol only one side recorded cannot contradict the strikes.
    assert forward_loop._booked_contracts_match("atm_straddle", entry, no_symbol)


def _baseline(store, baseline_name, structure, context, *, entry_mid=5.0):
    store.insert_entry(
        recommendation_id=f"rec-{baseline_name}",
        symbol="AAPL",
        baseline_name=baseline_name,
        structure=structure,
        entry_date=EARNINGS - timedelta(days=6),
        earnings_date=EARNINGS,
        selector_structure=None,
        entry_mid=entry_mid,
        modeled_cost_pct=0.0,
        execution_penalty_at_entry=0.0,
        data_quality_score_at_entry=0.9,
        iv_rv_har_at_entry=1.0,
        iv_rv_yz_at_entry=1.0,
        quote_source_at_entry="yfinance",
        quote_quality_at_entry="paper",
        entry_pricing_context=context,
    )
    return make_baseline_id(f"rec-{baseline_name}", baseline_name)


def _finalize_baselines(tmp_path, store, fetcher, as_of=T_MINUS_1):
    return forward_loop._finalize_baseline_exits(
        baseline_store=store,
        price_fetcher=fetcher,
        as_of=as_of,
        log_path=tmp_path / "log.jsonl",
        dry_run=False,
        mda_client=None,
    )


def test_legacy_straddle_baseline_resolves_as_unverifiable_and_is_not_compared(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    legacy = {"strike": 100.0, "call_contract": "C100", "put_contract": "P105", "front_expiry": EXPIRY}
    _baseline(store, "always_atm_straddle", "atm_straddle", legacy)

    def fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):
        return {"mid": 6.0, "context": {**STRADDLE, "call_contract": "C100", "put_contract": "P100"}}

    _finalize_baselines(tmp_path, store, fetcher)

    row = store.list_for_diagnostics()[0]
    assert row["status"] == "resolved"
    assert row["exit_repricing"] == EXIT_REPRICING_UNVERIFIABLE
    report = build_evidence_report(baseline_store=store, outcome_store=OutcomeStore(tmp_path / "o.sqlite"))
    assert "always_atm_straddle" not in report["baseline_comparison"]
    assert report["legacy_repriced_baselines"]["by_exit_repricing"] == {EXIT_REPRICING_UNVERIFIABLE: 1}


def test_selector_exit_on_other_contracts_fails_closed_and_records_reason(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _open_trade(store, "NVDA", "otm_strangle", context=STRANGLE)
    captured = []

    def fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):
        return {"mid": 3.0, "context": {**STRANGLE, "call_strike": 104.0}}

    summary = forward_loop.run_exit_detection(
        today=T_MINUS_1, store=store, log_path=tmp_path / "log.jsonl", price_fetcher=fetcher,
        finalizer=lambda **kw: captured.append(kw) or {},
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )

    assert captured == []
    assert summary["skipped"] == 1
    row = store.get_trade("NVDA")
    assert row["status"] == "open"
    assert row["last_exit_attempt_reason"] == "booked_contract_mismatch"


def test_selector_exit_records_contract_verification(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _open_trade(store, "NVDA", "otm_strangle", context=STRANGLE)
    captured = []

    def fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):
        return {"mid": 3.0, "context": dict(STRANGLE)}

    forward_loop.run_exit_detection(
        today=T_MINUS_1, store=store, log_path=tmp_path / "log.jsonl", price_fetcher=fetcher,
        finalizer=lambda **kw: captured.append(kw) or {},
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )

    assert captured[0]["exit_execution_scenarios"]["contract_verification"] == "verified"


# ── 2. exit attrition ─────────────────────────────────────────────────────────


def _open_trade(store, trade_id, structure="otm_strangle", *, context=None, notes_extra=None):
    notes = {"pricing_context": context or {}}
    notes.update(notes_extra or {})
    store.insert_entry(
        trade_id=trade_id,
        symbol=trade_id,
        structure=structure,
        entry_date=EARNINGS - timedelta(days=6),
        earnings_date=EARNINGS,
        setup_score=0.6,
        source_type="paper",
        entry_mid=2.0,
        execution_penalty_at_entry=0.0,
        notes=json.dumps(notes),
    )


def test_unpriced_t_minus_1_exit_becomes_terminal_exit_missing(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    _open_trade(store, "AMD", context=STRANGLE)
    calls = []

    def failing(*, symbol, structure, earnings_date, as_of_date, context=None):
        calls.append(as_of_date)
        return {"mid": None, "reason": "booked_contract_unavailable"}

    kwargs = dict(store=store, log_path=tmp_path / "log.jsonl", price_fetcher=failing,
                  finalizer=lambda **kw: pytest.fail("must not finalize"), baseline_store=baselines)
    # Two runs on T-1 = two attempts at the same valuation date.
    forward_loop.run_exit_detection(today=T_MINUS_1, **kwargs)
    forward_loop.run_exit_detection(today=T_MINUS_1, **kwargs)
    assert store.get_trade("AMD")["status"] == "open"
    summary = forward_loop.run_exit_detection(today=EARNINGS, **kwargs)

    assert calls == [T_MINUS_1, T_MINUS_1]  # never re-priced on a later date
    assert summary["exit_missing"] == 1
    row = store.get_trade("AMD")
    assert row["status"] == "exit_missing"
    assert row["exit_missing_reason"] == "booked_contract_unavailable"
    assert row["realized_return_pct"] is None
    logs = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(item["event_type"] == "exit_missing" and item["trade_id"] == "AMD" for item in logs)


def test_historical_orphans_are_classified_without_fabricating_outcomes(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _open_trade(store, "PYPL")  # earnings long past, never attempted
    _open_trade(store, "BAD", notes_extra=NESTED)  # invalidated: left alone

    moved = store.mark_missing_exits(EARNINGS + timedelta(days=120))

    assert [row["trade_id"] for row in moved] == ["PYPL"]
    assert store.get_trade("PYPL")["exit_missing_reason"] == "no_exit_attempt_recorded"
    assert store.get_trade("BAD")["status"] == "open"
    # Idempotent.
    assert store.mark_missing_exits(EARNINGS + timedelta(days=121)) == []


def test_baseline_exit_retries_same_day_then_goes_terminal(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    straddle_id = _baseline(store, "always_atm_straddle", "atm_straddle", STRADDLE)
    strangle_id = _baseline(store, "always_otm_strangle", "otm_strangle", STRANGLE)
    attempts = {"n": 0}

    def flaky(*, symbol, structure, earnings_date, as_of_date, context=None):
        if structure == "otm_strangle":
            return {"mid": None, "reason": "no_option_expiries"}
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"mid": None, "reason": "no_option_expiries"}
        return {"mid": 6.0, "context": dict(STRADDLE)}

    _finalize_baselines(tmp_path, store, flaky)
    _finalize_baselines(tmp_path, store, flaky)  # same T-1 day: retried
    rows = {row["baseline_id"]: row for row in store.list_for_diagnostics()}
    assert rows[straddle_id]["status"] == "resolved"
    assert rows[straddle_id]["exit_repricing"] == EXIT_REPRICING_BOOKED
    assert rows[strangle_id]["status"] == "exit_skipped"

    _finalize_baselines(tmp_path, store, flaky, as_of=EARNINGS)  # window closed: not retried
    assert attempts["n"] == 2
    assert {row["baseline_id"]: row["status"] for row in store.list_for_diagnostics()}[strangle_id] == "exit_skipped"


def test_never_attempted_baseline_becomes_exit_missing(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    _baseline(store, "always_iron_condor", "iron_condor", CONDOR, entry_mid=1.0)

    summary = _finalize_baselines(tmp_path, store, lambda **_: pytest.fail("must not quote"), as_of=EARNINGS)

    assert summary["baseline_exit_missing"] == 1
    row = store.list_for_diagnostics()[0]
    assert (row["status"], row["skip_reason"]) == ("exit_missing", "no_exit_attempt_recorded")


def test_report_counts_exit_attrition_by_reason_and_structure(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(outcomes, "AMD")
    outcomes.record_exit_attempt_failure("AMD", reason="booked_contract_unavailable", attempted_on=T_MINUS_1)
    outcomes.mark_missing_exits(EARNINGS)
    _resolved(outcomes, "MU", 9.0)
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    _baseline(baselines, "always_iron_condor", "iron_condor", CONDOR, entry_mid=1.0)
    baselines.mark_missing_exits(EARNINGS)

    attrition = build_evidence_report(baseline_store=baselines, outcome_store=outcomes)["exit_attrition"]

    assert attrition["selector"]["exit_missing"] == 1
    assert attrition["selector"]["resolved"] == 1
    assert attrition["selector"]["attrition_rate"] == 0.5
    assert attrition["selector"]["by_reason"] == {"booked_contract_unavailable": 1}
    assert attrition["selector"]["by_structure"] == {"otm_strangle": 1}
    assert attrition["baselines"]["by_reason"] == {"no_exit_attempt_recorded": 1}


# ── 3. invalidation at the mutation boundary ──────────────────────────────────


def _resolved(store, trade_id, ret, *, exit_date=T_MINUS_1, notes=None):
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure="otm_strangle",
        entry_date=exit_date - timedelta(days=5), earnings_date=exit_date + timedelta(days=1),
        setup_score=0.6, source_type="paper", entry_mid=2.0,
    )
    store.update_exit(trade_id=trade_id, exit_date=exit_date, exit_mid=2.0,
                      realized_return_pct=ret, realized_expansion_pct=ret)
    store.mark_finalized(trade_id)
    if notes:
        # Invalidated by hand AFTER resolving, as AMZN/TTD were: the guarded
        # writes above would (correctly) refuse an already-invalid row.
        store._conn.execute("UPDATE outcome_trades SET notes = ? WHERE trade_id = ?", (json.dumps(notes), trade_id))
        store._conn.commit()
    return trade_id


def test_every_outcome_mutation_refuses_an_invalidated_row(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(store, "TTD", notes_extra=NESTED)
    assert not is_outcome_evidence_valid(store.get_trade("TTD"))

    assert record_trade_exit(trade_id="TTD", exit_date=T_MINUS_1, exit_mid=4.0,
                             realized_return_pct=100.0, realized_expansion_pct=100.0, store=store) is False
    assert store.mark_finalized("TTD") is False
    assert store.set_learning_update_status("TTD", "complete") is False
    assert store.claim_for_finalization("TTD") is False
    row = store.get_trade("TTD")
    assert (row["status"], row["realized_return_pct"], row["learning_update_status"]) == ("open", None, None)


def test_invalidation_racing_finalization_cannot_reach_learning_stores(tmp_path, monkeypatch):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(store, "AMZN")
    learned = []
    import services.calibration_service as calibration_service
    import services.structure_prior_store as structure_prior_store
    monkeypatch.setattr(calibration_service, "get_calibration", lambda: learned.append("cal") or pytest.fail("cal"))
    monkeypatch.setattr(structure_prior_store, "get_structure_prior_store", lambda: pytest.fail("prior"))

    # The finalizer has already read the row as valid when another process
    # invalidates it, before the exit write.
    real_update_exit = store.update_exit

    def update_exit_after_invalidation(**kwargs):
        store.invalidate("AMZN", reason="strike never booked")
        return real_update_exit(**kwargs)

    monkeypatch.setattr(store, "update_exit", update_exit_after_invalidation)

    with pytest.raises(ValueError, match="could not be claimed"):
        finalize_trade_and_update_learning(
            trade_id="AMZN", exit_date=T_MINUS_1,
            realized_return_pct=50.0, realized_expansion_pct=50.0, store=store,
        )
    assert learned == []
    row = store.get_trade("AMZN")
    assert (row["status"], row["realized_return_pct"]) == ("open", None)


def test_invalidate_is_refused_while_learning_updates_run(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(store, "AMZN")
    store.update_exit(trade_id="AMZN", exit_date=T_MINUS_1, exit_mid=3.0,
                      realized_return_pct=10.0, realized_expansion_pct=10.0)

    assert store.claim_for_finalization("AMZN") is True
    with pytest.raises(ValueError, match="being finalized"):
        store.invalidate("AMZN", reason="late")
    assert is_outcome_evidence_valid(store.get_trade("AMZN"))
    assert store.mark_finalized("AMZN") is True
    assert store.invalidate("AMZN", reason="late")["learning_already_applied"] is False


def test_refinalizing_a_finalized_trade_stays_idempotent(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _resolved(store, "MU", 9.0)
    assert store.claim_for_finalization("MU") is True
    assert store.get_trade("MU")["status"] == "finalized"


@pytest.mark.parametrize(
    "notes",
    [
        None, "not json", "[1, 2]", "7", json.dumps({}),
        json.dumps({"evidence_invalidated": True}), json.dumps({"evidence_invalidated": False}),
        json.dumps({"evidence_invalidated": None}), json.dumps({"evidence_invalidated": 0}),
        json.dumps({"evidence_invalidated": 1}), json.dumps({"evidence_invalidated": "yes"}),
        json.dumps({"evidence_invalidated": ""}), json.dumps({"evidence_invalidated": {}}),
        json.dumps({"evidence_invalidated": []}), json.dumps(NESTED),
    ],
)
@pytest.mark.parametrize("evidence_valid", [None, 0, 1])
def test_sql_and_python_validity_agree(notes, evidence_valid):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE outcome_trades (evidence_valid INTEGER, notes TEXT)")
    conn.execute("INSERT INTO outcome_trades VALUES (?, ?)", (evidence_valid, notes))
    sql_valid = conn.execute(f"SELECT {_VALID_EVIDENCE_SQL} AS v FROM outcome_trades").fetchone()["v"]
    row = dict(conn.execute("SELECT * FROM outcome_trades").fetchone())
    assert bool(sql_valid) is is_outcome_evidence_valid(row)


# ── 4. filter before LIMIT ────────────────────────────────────────────────────


def test_invalidated_rows_cannot_displace_valid_evidence_at_the_row_cap(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    _resolved(outcomes, "OLD", 5.0, exit_date=date(2026, 4, 1))
    _resolved(outcomes, "NEW", 900.0, exit_date=date(2026, 4, 20))
    outcomes.invalidate("NEW", reason="x")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")

    report = build_evidence_report(baseline_store=baselines, outcome_store=outcomes, max_rows=1)
    forward = build_forward_performance_diagnostics(
        outcome_store=outcomes, baseline_store=baselines,
        ledger=RecommendationLedger(ledger_path=tmp_path / "l.sqlite"), max_rows=1,
    )

    assert report["selector_summary"]["n"] == 1
    assert report["selector_summary"]["avg_realized_return_pct"] == 5.0
    assert report["invalidated_outcomes"]["n"] == 1
    assert forward["resolved_outcome_count"] == 1
    assert forward["invalidated_outcome_count"] == 1
    with pytest.raises(ValueError):
        outcomes.list_for_diagnostics(evidence="bogus")


# ── 5. nested reasons ─────────────────────────────────────────────────────────


def test_nested_notes_reason_reaches_report_and_cli(tmp_path, capsys):
    path = tmp_path / "o.sqlite"
    outcomes = OutcomeStore(path)
    _resolved(outcomes, "AMZN", -40.0, notes=NESTED)
    _resolved(outcomes, "TTD", -20.0, notes={"evidence_invalidated": True})
    expected = "exit_repriced_unheld_strike: exit priced the 265 call; the 260 was booked"

    assert outcome_invalidation_reason(outcomes.get_trade("AMZN")) == expected
    assert outcome_invalidation_reason(outcomes.get_trade("TTD")) == "notes.evidence_invalidated"
    report = build_evidence_report(baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), outcome_store=outcomes)
    assert report["invalidated_outcomes"]["by_reason"] == {expected: 1, "notes.evidence_invalidated": 1}
    outcomes.close()

    assert invalidate_cli(["--store", str(path), "--list"]) == 0
    listed = {json.loads(line)["trade_id"]: json.loads(line) for line in capsys.readouterr().out.splitlines()}
    assert listed["AMZN"]["invalidation_reason"] == expected
    assert listed["AMZN"]["evidence_valid"] is False


def test_valid_rows_have_no_invalidation_reason(tmp_path):
    outcomes = OutcomeStore(tmp_path / "o.sqlite")
    _resolved(outcomes, "MU", 9.0)
    assert outcome_invalidation_reason(outcomes.get_trade("MU")) is None
    assert outcome_recorder.is_outcome_evidence_valid(outcomes.get_trade("MU"))
