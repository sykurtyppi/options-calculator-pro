"""PR J: the evidence funnel says what happened, and moved earnings dates are one event.

Regression tests for the audit of #165 and #169:
- the funnel store was opened before the safe wrappers, so a locked database
  aborted the forward loop it only measures;
- ``safe_record`` lost locked writes silently while funnel health stayed green;
- stage conversion divided unrelated populations (``recorded -> exit_due``
  read 200% when older trades exited in a window that recorded fewer);
- discovery rows without a symbol vanished, so ``missing_symbol`` was never
  recorded;
- a re-check expired through the discovery path was not counted;
- a corrected earnings date opened a second event and left the old re-check
  pending (and the old ledger key counted as its own event).
"""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import scripts.run_forward_loop as forward_loop
from services import evidence_funnel as ef
from services.baseline_evidence_store import BaselineEvidenceStore
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import (
    EVENT_MOVE_WINDOW_DAYS,
    RecommendationLedger,
    build_record_from_analysis,
    is_moved_event_date,
)

TODAY = date(2026, 4, 21)


@pytest.fixture
def funnel(tmp_path):
    store = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    yield store
    store.close()


def _analysis(symbol, earnings, as_of=TODAY):
    return SimpleNamespace(
        symbol=symbol, recommendation="Candidate", confidence_pct=70.0, setup_score=0.6, metrics={},
        rationale=["ok"],
        selector_output={"recommendation": "Candidate", "best_structure": "atm_straddle",
                         "earnings_date": earnings.isoformat(), "binding_gate": None},
        structure_scorecards=[{"structure": "atm_straddle", "execution_penalty": 0.03}],
        vol_snapshot={"as_of_date": as_of.isoformat(), "earnings_date": earnings.isoformat(),
                      "data_quality_score": 0.9},
    )


def _cycle(tmp_path, funnel, today, rows, *, earnings=None, quotes=None, cycle="c", dry_run=False):
    analyzed = []

    def analyzer(symbol, mda_client=None):  # noqa: ARG001
        analyzed.append(symbol)
        return _analysis(symbol, (earnings or {})[symbol], as_of=today)

    def price_fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
        mid = (quotes or {}).get(symbol)
        return {"mid": None, "reason": "no_spot_price"} if mid is None else {"mid": mid, "context": {}}

    summary = forward_loop.run_forward_screener(
        today=today, dry_run=dry_run, store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        ledger=RecommendationLedger(ledger_path=tmp_path / "r.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": rows}, analyzer=analyzer, price_fetcher=price_fetcher,
        funnel=funnel, cycle_run_id=cycle, symbols=["A"],
    )
    return summary, analyzed


def _locked(*_args, **_kwargs):
    raise sqlite3.OperationalError("database is locked")


# ── a funnel that cannot be opened or written ──────────────────────────────


def test_an_unopenable_funnel_does_not_stop_the_entry_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "get_evidence_funnel", _locked)
    earnings = TODAY + timedelta(days=6)
    rows = [{"symbol": "AAA", "dte": 6, "earnings_date": earnings.isoformat()}]
    summary = forward_loop.run_forward_screener(
        today=TODAY, store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        ledger=RecommendationLedger(ledger_path=tmp_path / "r.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": rows},
        analyzer=lambda symbol, mda_client=None: _analysis(symbol, earnings),
        price_fetcher=lambda **_: {"mid": 4.0, "context": {}},
        cycle_run_id="c1", symbols=["A"],
    )
    assert summary["analyzed"] == 1 and summary["entries"] == 1
    assert summary["evidence_funnel"] == {"status": "unavailable", "dropped_writes": None}


def test_the_exit_pass_also_survives_an_unopenable_funnel(tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "get_evidence_funnel", _locked)
    summary = forward_loop.run_exit_detection(
        today=TODAY, store=OutcomeStore(store_path=tmp_path / "o.sqlite"),
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {"mid": None, "reason": "x"}, cycle_run_id="c1",
    )
    assert summary["evidence_funnel"]["status"] == "unavailable"


def test_a_really_locked_store_is_reported_unavailable_not_raised(tmp_path, monkeypatch):
    path = tmp_path / "funnel.sqlite"
    ef.EvidenceFunnelStore(path).close()
    holder = sqlite3.connect(str(path), isolation_level=None)
    holder.execute("PRAGMA journal_mode=DELETE")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        import services.sqlite_helpers as helpers

        real_open = helpers.open_db_conn
        monkeypatch.setattr(helpers, "open_db_conn", lambda p, **kw: real_open(p, **{**kw, "busy_timeout_ms": 50}))
        monkeypatch.setattr(ef, "_store", None)
        monkeypatch.setattr(ef, "_DEFAULT_STORE", path)
        assert ef.open_evidence_funnel() is None
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_lost_writes_are_counted_persisted_and_shown_as_degraded(tmp_path, funnel, monkeypatch):
    earnings = TODAY + timedelta(days=6)
    rows = [{"symbol": "AAA", "dte": 6, "earnings_date": earnings.isoformat()}]
    real_record = funnel.record
    lost = {"n": 0}

    def flaky_record(**kwargs):
        if kwargs["stage"] in (ef.STAGE_ANALYZED, ef.STAGE_ELIGIBLE):
            lost["n"] += 1
            raise sqlite3.OperationalError("database is locked")
        return real_record(**kwargs)

    monkeypatch.setattr(funnel, "record", flaky_record)
    summary, analyzed = _cycle(tmp_path, funnel, TODAY, rows, earnings={"AAA": earnings},
                               quotes={"AAA": 4.0}, cycle="c1")
    assert analyzed == ["AAA"] and summary["entries"] == 1          # the loop itself was untouched
    assert lost["n"] == 2
    assert summary["evidence_funnel"] == {"status": "degraded", "dropped_writes": 2,
                                          "dropped_writes_persisted": True}
    assert funnel.summarize(since=TODAY, until=TODAY)["dropped_writes"] == 2

    report = ef.build_evidence_funnel_report(as_of=TODAY, store=funnel)
    assert "evidence_funnel_writes_dropped" in report["warnings"]
    health = ef.build_evidence_funnel_health(store=funnel, as_of=TODAY)
    assert health["summary"]["dropped_writes_7d"] == 2
    assert any("2 stage writes were lost" in issue["message"] and issue["alertable"] for issue in health["issues"])


def test_a_clean_cycle_is_ok_and_raises_no_warning(tmp_path, funnel):
    earnings = TODAY + timedelta(days=6)
    rows = [{"symbol": "AAA", "dte": 6, "earnings_date": earnings.isoformat()}]
    summary, _ = _cycle(tmp_path, funnel, TODAY, rows, earnings={"AAA": earnings}, quotes={"AAA": 4.0})
    assert summary["evidence_funnel"] == {"status": "ok", "dropped_writes": 0, "dropped_writes_persisted": None}
    assert "evidence_funnel_writes_dropped" not in ef.build_evidence_funnel_report(as_of=TODAY, store=funnel)["warnings"]


def test_a_dry_run_reports_its_funnel_as_dry_run(tmp_path, funnel):
    summary, _ = _cycle(tmp_path, funnel, TODAY, [], dry_run=True)
    assert summary["evidence_funnel"]["status"] == "dry_run"


def test_the_dropped_count_survives_a_store_that_stays_unwritable(tmp_path, funnel, monkeypatch):
    funnel.note_dropped_write(cycle_run_id="c1", as_of=TODAY)
    monkeypatch.setattr(funnel, "finish_cycle", _locked)
    status = ef.safe_finish_cycle(funnel, cycle_run_id="c1", as_of=TODAY)
    # Kept beside the store (which is what failed) and absorbed once it is writable.
    assert status == {"status": "degraded", "dropped_writes": 1, "dropped_writes_persisted": True}
    monkeypatch.undo()
    assert ef.build_evidence_funnel_health(store=funnel, as_of=TODAY)["summary"]["dropped_writes_7d"] == 1


# ── cohort-valid conversion ─────────────────────────────────────────────────


def _rec(funnel, cycle, day, stage, *, symbol="S", trade=None, outcome=ef.OUTCOME_ADVANCED, reason=None):
    funnel.record(cycle_run_id=cycle, as_of=day, stage=stage, outcome=outcome, symbol=symbol,
                  trade_id=trade, reason=reason or (None if outcome == ef.OUTCOME_ADVANCED else "x"))


def test_recorded_to_exit_due_follows_the_recorded_trades_not_older_ones(funnel):
    # The audit's repro: two older trades reach exit in a window that records one new trade.
    day0, day1 = TODAY - timedelta(days=10), TODAY
    _rec(funnel, "c0", day0, ef.STAGE_RECORDED, symbol="OLD1", trade="t_old1")
    _rec(funnel, "c0", day0, ef.STAGE_RECORDED, symbol="OLD2", trade="t_old2")
    _rec(funnel, "c1", day1, ef.STAGE_RECORDED, symbol="NEW", trade="t_new")
    _rec(funnel, "c1", day1, ef.STAGE_EXIT_DUE, symbol="OLD1", trade="t_old1")
    _rec(funnel, "c1", day1, ef.STAGE_EXIT_DUE, symbol="OLD2", trade="t_old2")

    today = funnel.summarize(since=day1, until=day1)
    assert today["stages"]["exit_due"]["advanced"] == 2 and today["stages"]["recorded"]["advanced"] == 1
    move = today["transitions"]["recorded->exit_due"]
    assert move == {"basis": "trade", "cohort": 1, "advanced": 0, "stopped": 0, "deferred": 0, "open": 1,
                    "rate": 0.0}
    assert today["conversion"]["recorded->exit_due"] == 0.0

    # Over a window that holds their recording, the old trades did convert.
    month = funnel.summarize(since=day0, until=day1)["transitions"]["recorded->exit_due"]
    assert (month["cohort"], month["advanced"], month["open"]) == (3, 2, 1)


def test_a_trade_converts_in_a_later_cycle_but_not_after_the_window_ends(funnel):
    _rec(funnel, "c1", TODAY, ef.STAGE_EXIT_DUE, trade="t1")
    _rec(funnel, "c1", TODAY, ef.STAGE_RESOLVED, trade="t1", outcome=ef.OUTCOME_DEFERRED)
    _rec(funnel, "c2", TODAY + timedelta(days=1), ef.STAGE_RESOLVED, trade="t1")
    first_day = funnel.summarize(since=TODAY, until=TODAY)["transitions"]["exit_due->resolved"]
    assert (first_day["advanced"], first_day["deferred"]) == (0, 1)
    later = funnel.summarize(since=TODAY, until=TODAY + timedelta(days=1))["transitions"]["exit_due->resolved"]
    assert (later["advanced"], later["deferred"], later["rate"]) == (1, 0, 1.0)


def test_entry_conversion_pairs_stages_within_one_candidate_pass(funnel):
    # Seen twice (deferred, then in window): two passes, one of which qualified.
    _rec(funnel, "c1", TODAY, ef.STAGE_DISCOVERED, symbol="A")
    _rec(funnel, "c1", TODAY, ef.STAGE_DATA_QUALIFIED, symbol="A", outcome=ef.OUTCOME_DEFERRED)
    _rec(funnel, "c2", TODAY, ef.STAGE_DISCOVERED, symbol="A")
    _rec(funnel, "c2", TODAY, ef.STAGE_DATA_QUALIFIED, symbol="A")
    # Qualified in a pass where it was never discovered: does not count for the discovered cohort.
    _rec(funnel, "c3", TODAY, ef.STAGE_DATA_QUALIFIED, symbol="B")
    move = funnel.summarize(since=TODAY, until=TODAY)["transitions"]["discovered->data_qualified"]
    assert (move["basis"], move["cohort"], move["advanced"], move["deferred"], move["rate"]) == (
        "candidate_pass", 2, 1, 1, 0.5)


def test_no_conversion_ever_exceeds_one(funnel):
    rng = random.Random(7)
    stages = list(ef.STAGES)
    for i in range(400):
        day = TODAY - timedelta(days=rng.randrange(20))
        stage = rng.choice(stages)
        outcome = rng.choice(ef.OUTCOMES)
        _rec(funnel, f"c{rng.randrange(25)}", day, stage, symbol=f"S{rng.randrange(8)}",
             trade=f"t{rng.randrange(10)}" if stage in ef.TRADE_STAGES else None, outcome=outcome)
    for since in (TODAY, TODAY - timedelta(days=6), TODAY - timedelta(days=29)):
        summary = funnel.summarize(since=since, until=TODAY)
        for name, item in summary["transitions"].items():
            assert item["rate"] is None or 0.0 <= item["rate"] <= 1.0, name
            assert item["advanced"] + item["stopped"] + item["deferred"] + item["open"] == item["cohort"]


# ── malformed discovery rows ────────────────────────────────────────────────


def test_discovery_rows_without_a_symbol_are_recorded_not_dropped(tmp_path, funnel):
    rows = [{"symbol": "", "dte": 6, "earnings_date": "2026-04-27"}, {"dte": 5}, "garbage",
            {"symbol": "OK", "dte": 30, "earnings_date": "2026-05-21"}]
    summary, _ = _cycle(tmp_path, funnel, TODAY, rows, cycle="c1")
    assert summary["malformed_discovery_rows"] == 3 and summary["discovered"] == 1
    malformed = [e for e in funnel.events("c1") if e["reason"] == ef.REASON_MISSING_SYMBOL]
    assert sorted(e["subject"] for e in malformed) == [
        "malformed:ranked_screener:0", "malformed:ranked_screener:1", "malformed:ranked_screener:2"]
    assert all(e["stage"] == "discovered" and e["outcome"] == "dropped" for e in malformed)
    assert json.loads(malformed[0]["detail_json"])["fields"] == ["dte", "earnings_date", "symbol"]
    assert funnel.summarize(since=TODAY, until=TODAY)["drop_reasons"]["discovered"] == {ef.REASON_MISSING_SYMBOL: 3}


# ── re-check bookkeeping ────────────────────────────────────────────────────


def test_a_recheck_expired_through_discovery_is_counted(tmp_path, funnel):
    earnings = TODAY + timedelta(days=2)
    funnel.schedule_recheck(symbol="LATE", earnings_date=earnings, recheck_on=TODAY - timedelta(days=1),
                            reason="missing_entry_quote:no_spot_price", as_of=TODAY - timedelta(days=2))
    summary, _ = _cycle(tmp_path, funnel, TODAY,
                        [{"symbol": "LATE", "dte": 2, "earnings_date": earnings.isoformat()}], cycle="c1")
    assert summary["rechecks_expired"] == 1
    (late,) = [e for e in funnel.events("c1") if e["symbol"] == "LATE" and e["stage"] == "data_qualified"]
    assert json.loads(late["detail_json"])["recheck_expired"]["reason"] == "missing_entry_quote:no_spot_price"


def test_too_late_without_a_recheck_counts_no_expiry(tmp_path, funnel):
    summary, _ = _cycle(tmp_path, funnel, TODAY,
                        [{"symbol": "LATE", "dte": 2, "earnings_date": (TODAY + timedelta(days=2)).isoformat()}])
    assert summary["rechecks_expired"] == 0


# ── moved earnings dates ────────────────────────────────────────────────────


def test_the_moved_date_rule():
    seen = date(2026, 4, 10)
    assert is_moved_event_date("2026-04-30", "2026-05-07", as_of=seen)
    assert is_moved_event_date("2026-05-07", "2026-04-30", as_of=seen)            # moved earlier
    assert not is_moved_event_date("2026-04-30", "2026-04-30", as_of=seen)        # same date
    far = date(2026, 4, 30) + timedelta(days=EVENT_MOVE_WINDOW_DAYS + 1)
    assert not is_moved_event_date("2026-04-30", far, as_of=seen)                 # next quarter
    assert not is_moved_event_date("2026-04-05", "2026-04-20", as_of=seen)        # old date had passed
    assert not is_moved_event_date(None, "2026-04-20", as_of=seen)
    # Boundaries: exactly the window apart, and an old date that is the day the move is seen.
    assert is_moved_event_date("2026-04-30", date(2026, 4, 30) + timedelta(days=EVENT_MOVE_WINDOW_DAYS), as_of=seen)
    assert is_moved_event_date("2026-04-10", "2026-04-14", as_of=seen)            # postponed on the day


def test_discovery_with_a_moved_date_supersedes_the_old_recheck(tmp_path, funnel):
    old, new = TODAY + timedelta(days=20), TODAY + timedelta(days=27)
    _cycle(tmp_path, funnel, TODAY, [{"symbol": "MOVE", "dte": 20, "earnings_date": old.isoformat()}], cycle="c1")
    assert [r["earnings_date"] for r in funnel.pending_rechecks()] == [old.isoformat()]

    summary, _ = _cycle(tmp_path, funnel, TODAY + timedelta(days=1),
                        [{"symbol": "MOVE", "dte": 26, "earnings_date": new.isoformat()}], cycle="c2")
    assert summary["rechecks_superseded"] == 1
    (pending,) = funnel.pending_rechecks()
    assert pending["earnings_date"] == new.isoformat()
    assert funnel.recheck_counts()["superseded"] == 1
    row = funnel._conn.execute("SELECT status, superseded_by FROM candidate_rechecks WHERE earnings_date = ?",
                               (old.isoformat(),)).fetchone()
    assert tuple(row) == ("superseded", new.isoformat())
    (discovered,) = [e for e in funnel.events("c2") if e["stage"] == "discovered"]
    assert json.loads(discovered["detail_json"])["superseded_rechecks"][0]["earnings_date"] == old.isoformat()


def test_an_analysis_with_a_moved_date_supersedes_the_old_recheck(tmp_path, funnel):
    old, new = TODAY + timedelta(days=12), TODAY + timedelta(days=9)
    funnel.schedule_recheck(symbol="MOVE", earnings_date=old, recheck_on=TODAY + timedelta(days=1), reason="x",
                            as_of=TODAY - timedelta(days=1))
    # Discovery gives no date; the analysis finds the corrected one.
    summary, _ = _cycle(tmp_path, funnel, TODAY, [{"symbol": "MOVE", "dte": 9}],
                        earnings={"MOVE": new}, quotes={"MOVE": 4.0}, cycle="c1")
    assert summary["entries"] == 1 and summary["rechecks_superseded"] == 1
    assert funnel.pending_rechecks() == []
    assert funnel.recheck_counts()["superseded"] == 1
    # Discovery and analysis agreeing on a date supersede nothing.
    summary, _ = _cycle(tmp_path, funnel, TODAY + timedelta(days=1),
                        [{"symbol": "MOVE", "dte": 8, "earnings_date": new.isoformat()}],
                        earnings={"MOVE": new}, cycle="c2")
    assert summary["rechecks_superseded"] == 0


def test_a_recheck_for_a_distinct_event_is_not_superseded(tmp_path, funnel):
    funnel.schedule_recheck(symbol="Q", earnings_date="2026-07-30", recheck_on=date(2026, 7, 16), reason="x",
                            as_of=TODAY)
    assert funnel.supersede_rechecks(symbol="Q", earnings_date="2026-04-30", as_of=TODAY) == []
    assert len(funnel.pending_rechecks()) == 1


def _record(symbol, earnings, as_of, provenance="forward_post_freeze"):
    analysis = SimpleNamespace(
        symbol=symbol, recommendation="Candidate", metrics={},
        selector_output={"recommendation": "Candidate", "best_structure": "call_calendar",
                         "earnings_date": earnings},
        structure_scorecards=[], rationale=[], sample_provenance=provenance,
        vol_snapshot={"symbol": symbol, "as_of_date": as_of, "earnings_date": earnings},
    )
    return build_record_from_analysis(analysis)


@pytest.fixture
def ledger(tmp_path):
    store = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    yield store
    store.close()


def test_a_corrected_date_supersedes_the_old_event_in_the_ledger(ledger):
    ledger.record(_record("MOVE", "2026-04-28", "2026-04-20"))
    ledger.record(_record("MOVE", "2026-05-05", "2026-04-22"))     # corrected two days later
    assert ledger.superseded_event_keys() == {"MOVE|2026-04-28": "MOVE|2026-05-05"}
    events = ledger.summarize()["events"]
    assert events["total"] == 1 and events["superseded"] == 1
    assert [row["event_key"] for row in ledger.latest_event_rows()] == ["MOVE|2026-05-05"]
    # The stale date is never resolved as if the event happened then.
    pending = ledger.list_pending_candidate_exit_resolutions(now=date(2026, 5, 2))
    assert pending == []
    assert [r["event_key"] for r in ledger.list_pending_candidate_exit_resolutions(now=date(2026, 5, 9))] == [
        "MOVE|2026-05-05"]


def test_a_past_date_is_its_own_event(ledger):
    ledger.record(_record("PAST", "2026-04-20", "2026-04-15"))
    ledger.record(_record("PAST", "2026-05-12", "2026-04-25"))     # old date had passed by then
    assert ledger.superseded_event_keys() == {}
    assert ledger.summarize()["events"]["total"] == 2


def test_two_dates_on_one_day_are_ambiguous_and_both_stand(ledger):
    ledger.record(_record("AMB", "2026-04-28", "2026-04-20"))
    ledger.record(_record("AMB", "2026-05-05", "2026-04-20"))
    assert ledger.superseded_event_keys() == {}


def test_older_information_recorded_late_does_not_override_newer(ledger):
    ledger.record(_record("LATE", "2026-05-05", "2026-04-22"))
    ledger.record(_record("LATE", "2026-04-28", "2026-04-20"))     # a backfill of an earlier analysis
    assert ledger.superseded_event_keys() == {"LATE|2026-04-28": "LATE|2026-05-05"}
    assert ledger.summarize()["events"]["total"] == 1


def test_a_date_that_moves_back_is_reinstated(ledger):
    ledger.record(_record("FLIP", "2026-04-28", "2026-04-20"))
    ledger.record(_record("FLIP", "2026-05-05", "2026-04-21"))
    ledger.record(_record("FLIP", "2026-04-28", "2026-04-22"))
    assert ledger.superseded_event_keys() == {"FLIP|2026-05-05": "FLIP|2026-04-28"}
    assert [row["event_key"] for row in ledger.latest_event_rows()] == ["FLIP|2026-04-28"]


def test_diagnostics_count_a_moved_event_once(tmp_path, ledger):
    no_trade = _record("NT", "2026-04-28", "2026-04-20")
    ledger.record(no_trade.__class__(**{**no_trade.__dict__, "recommendation": "No Trade"}))
    moved = _record("NT", "2026-05-05", "2026-04-22")
    ledger.record(moved.__class__(**{**moved.__dict__, "recommendation": "No Trade"}))
    diagnostics = build_forward_performance_diagnostics(
        outcome_store=OutcomeStore(store_path=tmp_path / "o.sqlite"), ledger=ledger,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"))
    assert (diagnostics["total_events"], diagnostics["no_trade_event_count"]) == (1, 1)


def test_the_run_log_carries_the_worst_funnel_status_and_malformed_rows():
    from datetime import datetime, timezone

    from scripts.run_evidence_cycle import _run_log_payload

    payload = {"forward_loop": {
        "entries": {"malformed_discovery_rows": 2, "evidence_funnel": {"status": "ok", "dropped_writes": 0}},
        "exits": {"evidence_funnel": {"status": "degraded", "dropped_writes": 3}},
    }}
    log = _run_log_payload(started_at=datetime.now(timezone.utc), status="success", cycle_date=TODAY,
                           dry_run=False, payload=payload)
    assert (log["candidates_malformed"], log["evidence_funnel_status"], log["evidence_funnel_dropped_writes"]) == (
        2, "degraded", 3)
    empty = _run_log_payload(started_at=datetime.now(timezone.utc), status="success", cycle_date=TODAY,
                             dry_run=False, payload={})
    assert (empty["evidence_funnel_status"], empty["evidence_funnel_dropped_writes"]) == (None, 0)


def test_a_members_best_downstream_outcome_counts_and_earlier_rows_do_not(funnel):
    # Resolved, then a stray later deferral of the same trade: still resolved.
    _rec(funnel, "c1", TODAY, ef.STAGE_EXIT_DUE, trade="t1")
    _rec(funnel, "c1", TODAY, ef.STAGE_RESOLVED, trade="t1")
    _rec(funnel, "c2", TODAY + timedelta(days=1), ef.STAGE_RESOLVED, trade="t1", outcome=ef.OUTCOME_DEFERRED)
    # A downstream row dated before the member entered the cohort is not its conversion.
    _rec(funnel, "c0", TODAY - timedelta(days=1), ef.STAGE_RESOLVED, trade="t2")
    _rec(funnel, "c1", TODAY, ef.STAGE_EXIT_DUE, trade="t2")
    move = funnel.summarize(since=TODAY, until=TODAY + timedelta(days=1))["transitions"]["exit_due->resolved"]
    assert (move["cohort"], move["advanced"], move["deferred"], move["open"]) == (2, 1, 0, 1)
