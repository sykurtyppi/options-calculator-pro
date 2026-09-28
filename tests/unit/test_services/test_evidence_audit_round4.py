"""Regressions for the independent audit of main after #153.

1. a same-day T-1 retry re-quoted a trade whose exit was already recorded, so
   the row, calibration and priors disagreed;
2. the pending-finalization sweep wiped the stored exit details;
3. seeded replay/backtest rows counted as forward evidence (gate, maturity);
4. a past as_of priced exits with present-day live quotes;
5. the backfill could not remove an invalidated calibration observation once
   no valid trades remained;
6. the invalidated-after-learning health check read a status column, not the
   stores (never cleared after repair; missed partial learning);
7. failed learning updates were never retried or alerted;
8. a legacy NaN exit stayed 'exited' forever;
9. the seed script learned rows it failed to finalize and could strand rows
   the loop then counted as attrition;
10. the health check used the UTC date;
11. an 'exited' row counted as both open and resolved;
plus: a claim was not released when the final status write raised.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.backfill_prior_store_timestamps as backfill
import scripts.run_forward_loop as forward_loop
from scripts.seed_outcomes_from_replay import seed_from_trades
from services.baseline_evidence_store import BaselineEvidenceStore
from services.calibration_service import IVExpansionCalibration
from services.evidence_health import EvidenceHealthConfig, _check_evidence_integrity
from services.evidence_report import build_evidence_report
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore, finalize_trade_and_update_learning, make_trade_id
from services.recommendation_ledger import RecommendationLedger
from services.structure_prior_store import StructurePriorStore

T_MINUS_1 = date(2026, 4, 27)
EARNINGS = T_MINUS_1 + timedelta(days=1)
QUOTE = {"mid": 3.0, "context": {}, "quote_source": "marketdata", "quote_quality": "live",
         "quote_timestamp": "2026-04-27T15:59:00", "bid_ask_mid": {"bid": 2.9, "ask": 3.1}}


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    import services.calibration_service as calibration_service
    import services.structure_prior_store as structure_prior_store

    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    priors = StructurePriorStore(tmp_path / "priors.json")
    monkeypatch.setattr(calibration_service, "get_calibration", lambda: cal)
    monkeypatch.setattr(structure_prior_store, "get_structure_prior_store", lambda: priors)
    return cal, priors


def _open(store, trade_id, *, source="paper"):
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure="otm_strangle",
        entry_date=EARNINGS - timedelta(days=6), earnings_date=EARNINGS,
        setup_score=0.6, source_type=source, entry_mid=2.0, execution_penalty_at_entry=0.0,
        notes=json.dumps({"pricing_context": {}}),
    )


def _run(tmp_path, store, day, quote=None, **kwargs):
    return forward_loop.run_exit_detection(
        today=day, store=store, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: dict(quote or QUOTE),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), **kwargs,
    )


def _cal_expansions(tmp_path):
    return json.loads((tmp_path / "cal.json").read_text())["expansions"]


def _prior_returns(tmp_path):
    raw = json.loads((tmp_path / "priors.json").read_text())
    return [o["realized_return_pct"] for e in raw["structures"].values() for o in e["observations"]]


def _cfg(tmp_path):
    return EvidenceHealthConfig(
        expected_date=EARNINGS, outcome_store_path=tmp_path / "o.sqlite",
        baseline_store_path=tmp_path / "b.sqlite",
        calibration_store_path=tmp_path / "cal.json", prior_store_path=tmp_path / "priors.json",
    )


def _raise(*_args, **_kwargs):
    raise RuntimeError("database is locked")


# ── 1 / 2. a recorded exit is final; nothing wipes it ─────────────────────────


def test_same_day_retry_after_lost_claim_does_not_requote(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    real_renew = store.renew_finalization_claim
    calls = {"n": 0}

    def renew(*args, **kwargs):
        calls["n"] += 1  # the lease lapses after the calibration write
        return real_renew(*args, **kwargs) if calls["n"] == 1 else False

    store.renew_finalization_claim = renew
    _run(tmp_path, store, T_MINUS_1)
    del store.renew_finalization_claim

    summary = _run(tmp_path, store, T_MINUS_1, quote={**QUOTE, "mid": 4.0})  # same-day retry, new price

    # Not re-quoted: completed from the recorded exit by the sweep.
    assert (summary["exits"], summary["refinalized"]) == (0, 1)
    row = store.get_trade("AAA")
    assert row["status"] == "finalized"
    assert row["realized_expansion_pct"] == 50.0  # (3 - 2) / 2, the first quote
    assert _cal_expansions(tmp_path) == [50.0] and _prior_returns(tmp_path) == [50.0]


def test_finalize_error_after_learning_then_retry_stays_consistent(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.mark_finalized = _raise
    _run(tmp_path, store, T_MINUS_1)
    del store.mark_finalized
    assert store.get_trade("AAA")["status"] == "exited"  # the claim was released

    _run(tmp_path, store, T_MINUS_1, quote={**QUOTE, "mid": 4.0})

    row = store.get_trade("AAA")
    assert row["status"] == "finalized" and row["realized_return_pct"] == 50.0
    assert _cal_expansions(tmp_path) == [50.0] and _prior_returns(tmp_path) == [50.0]


def test_sweep_keeps_every_stored_exit_detail(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.mark_finalized = _raise
    _run(tmp_path, store, T_MINUS_1)
    del store.mark_finalized
    keys = ["exit_date", "exit_mid", "realized_return_pct", "realized_pnl", "exit_quote_source",
            "exit_quote_quality", "exit_quote_timestamp", "exit_bid_ask_mid_json", "exit_execution_scenarios_json"]
    before = {key: store.get_trade("AAA")[key] for key in keys}
    assert before["exit_quote_source"] == "marketdata" and before["exit_mid"] == 3.0

    summary = _run(tmp_path, store, EARNINGS)

    assert summary["refinalized"] == 1
    row = store.get_trade("AAA")
    assert row["status"] == "finalized"
    assert {key: row[key] for key in keys} == before


def test_partial_finalize_call_keeps_recorded_exit_fields(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, exit_mid=2.5, exit_quote_source="marketdata",
                      exit_bid_ask_mid={"bid": 2.4, "ask": 2.6})  # record_trade_exit, no realized values yet

    finalize_trade_and_update_learning(trade_id="AAA", realized_return_pct=25.0, realized_expansion_pct=25.0,
                                       store=store)

    row = store.get_trade("AAA")
    assert (row["exit_mid"], row["exit_quote_source"]) == (2.5, "marketdata")
    assert json.loads(row["exit_bid_ask_mid_json"]) == {"bid": 2.4, "ask": 2.6}
    assert row["realized_return_pct"] == 25.0 and row["status"] == "finalized"


def test_due_list_never_offers_a_recorded_exit_for_requoting(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "RECORDED")
    store.update_exit(trade_id="RECORDED", exit_date=T_MINUS_1, realized_return_pct=1.0, realized_expansion_pct=1.0)
    _open(store, "OPEN")

    assert [row["trade_id"] for row in store.trades_due_for_exit(T_MINUS_1)] == ["OPEN"]


def test_finalize_learns_the_recorded_exit_not_the_callers_numbers(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, exit_mid=3.0, realized_return_pct=50.0,
                      realized_expansion_pct=50.0, exit_quote_source="marketdata")

    result = finalize_trade_and_update_learning(trade_id="AAA", exit_date=T_MINUS_1, exit_mid=4.0,
                                                realized_return_pct=100.0, realized_expansion_pct=100.0,
                                                exit_quote_source="other", store=store)

    row = store.get_trade("AAA")
    assert (row["realized_return_pct"], row["exit_mid"], row["exit_quote_source"]) == (50.0, 3.0, "marketdata")
    assert _cal_expansions(tmp_path) == [50.0] and _prior_returns(tmp_path) == [50.0]
    assert any("already recorded" in warning for warning in result["warnings"])


# ── 3. replay rows are not forward evidence ──────────────────────────────────


def test_replay_rows_are_excluded_from_report_and_gate(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    for i in range(40):
        trade_id = f"R{i}"
        store.insert_entry(trade_id=trade_id, symbol=trade_id, structure="otm_strangle",
                           entry_date=date(2024, 1, 1) + timedelta(days=i * 15),
                           earnings_date=date(2024, 1, 3) + timedelta(days=i * 15),
                           setup_score=0.6, source_type="replay", claim_allowed=True)
        store.update_exit(trade_id=trade_id, exit_date=date(2024, 1, 3) + timedelta(days=i * 15),
                          realized_return_pct=8.0, realized_expansion_pct=8.0)
        store.mark_finalized(trade_id)

    report = build_evidence_report(baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
                                   outcome_store=store)

    assert report["selector_summary"]["n"] == 0
    inputs = report["maturity"]["inputs"]
    assert inputs["resolved_selector_outcomes"] == 0 and inputs["max_bucket_sample_size"] == 0
    assert inputs["active_evidence_days"] == 0
    assert report["commercialization_gate"]["ready_for_paid_beta"] is False
    assert report["excluded_replay_outcomes"] == {**report["excluded_replay_outcomes"], "n": 40, "resolved_n": 40}
    forward = build_forward_performance_diagnostics(
        ledger=RecommendationLedger(ledger_path=tmp_path / "l.sqlite"), outcome_store=store,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )
    assert forward["resolved_outcome_count"] == 0


def test_loop_never_touches_replay_rows(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "REPLAY", source="replay")  # e.g. an interrupted seed run

    assert store.trades_due_for_exit(T_MINUS_1) == []
    _run(tmp_path, store, EARNINGS + timedelta(days=3))

    assert store.get_trade("REPLAY")["status"] == "open"


# ── 4. point-in-time exits ───────────────────────────────────────────────────


def test_live_fetcher_is_refused_for_a_past_as_of(tmp_path, stores, monkeypatch):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    called = []
    monkeypatch.setattr(forward_loop, "_today", lambda: T_MINUS_1 + timedelta(days=1))

    def live(**kwargs):
        called.append(kwargs)
        return dict(QUOTE)

    monkeypatch.setattr(forward_loop, "fetch_structure_quote", live)
    forward_loop.run_exit_detection(
        today=T_MINUS_1, store=store, log_path=tmp_path / "log.jsonl", price_fetcher=live,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )

    assert called == []
    row = store.get_trade("AAA")
    assert row["status"] == "open" and row["exit_date"] is None
    assert row["last_exit_attempt_reason"] == forward_loop.PAST_AS_OF_REASON


def test_quote_from_another_day_is_not_an_exit(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")

    _run(tmp_path, store, T_MINUS_1, quote={**QUOTE, "quote_timestamp": "2026-04-28T10:00:00"})

    row = store.get_trade("AAA")
    assert row["status"] == "open" and row["exit_date"] is None
    assert row["last_exit_attempt_reason"] == forward_loop.QUOTE_DATE_MISMATCH_REASON


def test_utc_timestamp_is_compared_as_a_local_date(monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        # 21:30 in New York is 01:30 UTC the next day: still the same local day.
        assert forward_loop._quote_local_date("2026-04-28T01:30:00+00:00") == date(2026, 4, 27)
        assert forward_loop._quote_local_date(None) is None
        assert forward_loop._quote_local_date("not a time") is None
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


# ── 5. backfill and invalidated calibration ──────────────────────────────────


def _backfill(tmp_path):
    return backfill.main(["--db-path", str(tmp_path / "o.sqlite"), "--prior-store", str(tmp_path / "priors.json"),
                          "--cal-store", str(tmp_path / "cal.json"), "--target", "production"])


def _finalized(store, trade_id, value=50.0):
    _open(store, trade_id)
    finalize_trade_and_update_learning(trade_id=trade_id, exit_date=T_MINUS_1, realized_return_pct=value,
                                       realized_expansion_pct=value, store=store)


def test_backfill_empties_calibration_holding_only_invalidated_observations(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _finalized(store, "AAA")
    store.invalidate("AAA", reason="bad quote")

    assert _backfill(tmp_path) == 0

    cal = json.loads((tmp_path / "cal.json").read_text())
    assert cal["observation_ids"] == [] and cal["expansions"] == []


def test_backfill_refuses_to_guess_when_calibration_mixes_unknown_and_invalid(tmp_path, stores):
    cal, _ = stores
    store = OutcomeStore(tmp_path / "o.sqlite")
    _finalized(store, "AAA")
    store.invalidate("AAA", reason="bad quote")
    cal.update(0.5, 10.0, observation_id="EXTERNAL", source_type="replay", observation_date=T_MINUS_1)

    assert _backfill(tmp_path) == 1
    assert sorted(json.loads((tmp_path / "cal.json").read_text())["observation_ids"]) == ["AAA", "EXTERNAL"]


# ── 6. health reads the stores ───────────────────────────────────────────────


NOW = datetime(2026, 5, 10, 12, tzinfo=timezone.utc)


def test_health_warning_clears_after_repair(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _finalized(store, "AAA")
    _finalized(store, "BBB")
    store.invalidate("AAA", reason="bad quote")
    assert _check_evidence_integrity(_cfg(tmp_path), NOW)["summary"]["selector"]["invalidated_after_learning"] == ["AAA"]

    assert _backfill(tmp_path) == 0

    result = _check_evidence_integrity(_cfg(tmp_path), NOW)
    assert result["summary"]["selector"]["invalidated_after_learning"] == []
    assert not any("already reached" in issue["message"] for issue in result["issues"])


def test_health_and_invalidate_see_partial_learning(tmp_path, stores):
    _, priors = stores
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    priors.update = _raise  # prior write fails: calibration only
    finalize_trade_and_update_learning(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=5.0,
                                       realized_expansion_pct=5.0, store=store)
    assert store.get_trade("AAA")["learning_update_status"] == "prior_failed"

    assert store.invalidate("AAA", reason="x")["learning_already_applied"] is True
    assert _check_evidence_integrity(_cfg(tmp_path), NOW)["summary"]["selector"]["invalidated_after_learning"] == ["AAA"]


# ── 7. failed learning is retried and alerted ────────────────────────────────


def test_failed_learning_is_retried_by_the_loop(tmp_path, stores):
    _, priors = stores
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    real_update = priors.update
    priors.update = _raise
    _run(tmp_path, store, T_MINUS_1)
    assert store.get_trade("AAA")["learning_update_status"] == "prior_failed"
    assert _check_evidence_integrity(_cfg(tmp_path), NOW)["summary"]["selector"]["learning_failed"] == []  # < 24h

    priors.update = real_update
    summary = _run(tmp_path, store, EARNINGS)

    assert summary["learning_retried"] == 1
    assert store.get_trade("AAA")["learning_update_status"] == "complete"
    assert _prior_returns(tmp_path) == [50.0] and _cal_expansions(tmp_path) == [50.0]


def test_health_alerts_on_learning_that_keeps_failing(tmp_path, stores):
    _, priors = stores
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    priors.update = _raise
    _run(tmp_path, store, T_MINUS_1)
    with store._conn:
        store._conn.execute("UPDATE outcome_trades SET updated_at = datetime('now', '-3 days')")

    result = _check_evidence_integrity(_cfg(tmp_path), datetime.now(timezone.utc))

    assert result["summary"]["selector"]["learning_failed"] == ["AAA"]
    assert any("failed calibration/prior update" in issue["message"] for issue in result["issues"])


# ── 8. an exit without a value becomes terminal ──────────────────────────────


def test_legacy_nan_exit_becomes_terminal(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "NANROW")
    store.update_exit(trade_id="NANROW", exit_date=T_MINUS_1, realized_return_pct=1.0, realized_expansion_pct=1.0)
    with store._conn:  # a legacy NaN reads back as NULL
        store._conn.execute("UPDATE outcome_trades SET realized_return_pct = NULL WHERE trade_id = 'NANROW'")

    summary = _run(tmp_path, store, EARNINGS)

    row = store.get_trade("NANROW")
    assert summary["exit_missing"] == 1
    assert (row["status"], row["exit_missing_reason"]) == ("exit_missing", forward_loop.EXIT_WITHOUT_VALUE_REASON)


def test_exit_without_value_is_left_alone_on_its_exit_day(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, exit_mid=2.5)  # step 1 of a manual exit
    assert store.exits_without_value(T_MINUS_1) == []
    assert [row["trade_id"] for row in store.exits_without_value(EARNINGS)] == ["AAA"]


# ── 9. seed script ───────────────────────────────────────────────────────────


def _replay(symbol):
    return {"trade_date": "2025-03-03", "event_date": "2025-03-10", "symbol": symbol, "setup_score": 0.6,
            "gross_return_pct": 0.5, "net_return_pct": 0.4, "structure": "otm_strangle"}


def _seed(tmp_path, rows):
    return seed_from_trades(rows, structure="otm_strangle", dry_run=False, outcome_store_path=tmp_path / "o.sqlite",
                            calibration_store_path=tmp_path / "cal.json", prior_store_path=tmp_path / "priors.json")


def test_seed_completes_a_row_an_interrupted_run_left_open(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    trade_id = make_trade_id("ZZZ", date(2025, 3, 3), "otm_strangle", earnings_date=date(2025, 3, 10))
    store.insert_entry(trade_id=trade_id, symbol="ZZZ", structure="otm_strangle", entry_date=date(2025, 3, 3),
                       earnings_date=date(2025, 3, 10), setup_score=0.6, source_type="replay")

    summary = _seed(tmp_path, [_replay("ZZZ")])

    row = OutcomeStore(tmp_path / "o.sqlite").get_trade(trade_id)
    assert summary["inserted"] == 1
    assert (row["status"], row["realized_return_pct"], row["learning_update_status"]) == ("finalized", 40.0, "complete")


def test_seed_does_not_learn_a_row_it_could_not_finalize(tmp_path, monkeypatch):
    monkeypatch.setattr(OutcomeStore, "mark_finalized", lambda *_args, **_kwargs: False)

    summary = _seed(tmp_path, [_replay("YYY")])

    assert summary["skipped_conflict"] == 1 and summary["cal_updates"] == 0
    assert not (tmp_path / "cal.json").exists() or json.loads((tmp_path / "cal.json").read_text())["observation_ids"] == []


# ── 10. health uses the local date ───────────────────────────────────────────


def test_health_does_not_flag_a_normal_evening_run(tmp_path, stores, monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        store = OutcomeStore(tmp_path / "o.sqlite")
        _open(store, "AAA")
        _run(tmp_path, store, T_MINUS_1, quote={"mid": None, "reason": "missing_exit_mid", "context": {}})
        # 22:15 New York on T-1 is already the earnings day in UTC.
        result = _check_evidence_integrity(_cfg(tmp_path), datetime(2026, 4, 28, 2, 15, tzinfo=timezone.utc))
        assert result["summary"]["selector"]["open_past_exit_day"] == []
        # 20:30 New York on earnings day (00:30 UTC the next day), before
        # that evening's run: not yet overdue locally.
        evening = _check_evidence_integrity(_cfg(tmp_path), datetime(2026, 4, 29, 0, 30, tzinfo=timezone.utc))
        assert evening["summary"]["selector"]["open_past_exit_day"] == []
        # Still open two local days later: the loop is not running.
        late = _check_evidence_integrity(_cfg(tmp_path), datetime(2026, 4, 29, 16, 0, tzinfo=timezone.utc))
        assert late["summary"]["selector"]["open_past_exit_day"] == ["AAA"]
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


# ── 11. exited rows are not also open ────────────────────────────────────────


def test_exited_row_is_resolved_not_open(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=1.0, realized_expansion_pct=1.0)
    _open(store, "BBB")

    forward = build_forward_performance_diagnostics(
        ledger=RecommendationLedger(ledger_path=tmp_path / "l.sqlite"), outcome_store=store,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )

    assert (forward["open_outcome_count"], forward["resolved_outcome_count"]) == (1, 1)
