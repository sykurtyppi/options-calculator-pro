"""Unified exits and frozen cohorts (F6).

Before this, an event's exit was tracked three times on three schedules (the
paper trade at T-1, the baseline trades at T-1, the candidate resolver after
the event) with three status vocabularies, and nothing said whether the event
as a whole had resolved. Nor was there a way to fix a sample's membership
before reading its results.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import scripts.freeze_evidence_cohort as freeze_cli
from services import event_resolution as er
from services import evidence_cohort as ec
from services.baseline_evidence_store import BaselineEvidenceStore
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger, build_record_from_analysis

EARNINGS = date(2026, 4, 30)


def _paper(trade_id, status, *, earnings=EARNINGS, realized=None, symbol="AAPL"):
    return {"trade_id": trade_id, "symbol": symbol, "earnings_date": earnings.isoformat(), "status": status,
            "realized_return_pct": realized}


def _baseline(baseline_id, status, *, earnings=EARNINGS, symbol="AAPL"):
    return {"baseline_id": baseline_id, "symbol": symbol, "earnings_date": earnings.isoformat(), "status": status}


def _resolver(rec_id, status, *, earnings=EARNINGS, symbol="AAPL"):
    return {"recommendation_id": rec_id, "symbol": symbol, "earnings_date": earnings.isoformat(),
            "candidate_shadow_outcome_json": {"status": status} if status else {}}


# ── each track's vocabulary maps onto one ───────────────────────────────────


@pytest.mark.parametrize("row, as_of, state", [
    (_paper("t", "open"), EARNINGS - timedelta(days=2), er.STATE_OPEN),
    (_paper("t", "open"), EARNINGS - timedelta(days=1), er.STATE_EXIT_DUE),
    (_paper("t", "finalizing"), EARNINGS, er.STATE_EXIT_DUE),
    (_paper("t", "finalized", realized=12.0), EARNINGS, er.STATE_RESOLVED),
    (_paper("t", "exited", realized=-3.0), EARNINGS, er.STATE_RESOLVED),
    (_paper("t", "exited"), EARNINGS, er.STATE_EXIT_DUE),                 # no outcome yet
    (_paper("t", "exit_missing"), EARNINGS, er.STATE_MISSING),
])
def test_paper_states(row, as_of, state):
    assert er.paper_state(row, as_of) == state


@pytest.mark.parametrize("row, as_of, state", [
    (_baseline("b", "entry_skipped"), EARNINGS, None),                   # never entered
    (_baseline("b", "open"), EARNINGS - timedelta(days=3), er.STATE_OPEN),
    (_baseline("b", "open"), EARNINGS - timedelta(days=1), er.STATE_EXIT_DUE),
    (_baseline("b", "exit_skipped"), EARNINGS - timedelta(days=1), er.STATE_EXIT_DUE),   # retried that day
    (_baseline("b", "exit_skipped"), EARNINGS, er.STATE_MISSING),
    (_baseline("b", "resolved"), EARNINGS, er.STATE_RESOLVED),
    (_baseline("b", "exit_missing"), EARNINGS, er.STATE_MISSING),
])
def test_baseline_states(row, as_of, state):
    assert er.baseline_state(row, as_of) == state


@pytest.mark.parametrize("row, as_of, state", [
    (_resolver("r", None), EARNINGS + timedelta(days=2), er.STATE_OPEN),
    (_resolver("r", "awaiting_chain_data"), EARNINGS + timedelta(days=3), er.STATE_EXIT_DUE),
    (_resolver("r", "retrying"), EARNINGS + timedelta(days=9), er.STATE_EXIT_DUE),
    (_resolver("r", "ok"), EARNINGS + timedelta(days=4), er.STATE_RESOLVED),
    (_resolver("r", "permanently_failed:no_post_event_chain"), EARNINGS + timedelta(days=9), er.STATE_MISSING),
])
def test_resolver_states(row, as_of, state):
    assert er.resolver_state(row, as_of) == state


# ── one state per event ─────────────────────────────────────────────────────


def test_an_event_is_resolved_only_when_every_track_is():
    as_of = EARNINGS + timedelta(days=5)
    events = er.build_event_resolutions(
        as_of=as_of,
        paper_rows=[_paper("T1", "finalized", realized=8.0), _paper("T2", "finalized", realized=1.0, symbol="MSFT")],
        baseline_rows=[_baseline("B1", "resolved"), _baseline("B2", "entry_skipped"),
                       _baseline("B3", "resolved", symbol="MSFT"), _baseline("B4", "exit_missing", symbol="NVDA")],
        resolver_rows=[_resolver("R1", "retrying"), _resolver("R2", "ok", symbol="MSFT")],
    )
    aapl, msft, nvda = events["AAPL|2026-04-30"], events["MSFT|2026-04-30"], events["NVDA|2026-04-30"]
    assert aapl["track_states"] == {"paper": "resolved", "baseline": "resolved", "resolver": "exit_due"}
    assert aapl["state"] == er.STATE_EXIT_DUE
    assert [item["id"] for item in aapl["tracks"]["baseline"]] == ["B1"]     # the skipped entry is not a row
    assert msft["state"] == er.STATE_RESOLVED and nvda["state"] == er.STATE_MISSING

    summary = er.summarize_event_resolutions(events)
    assert summary["by_state"] == {"open": 0, "exit_due": 1, "resolved": 1, "missing": 1}
    assert summary["resolution_rate"] == 0.5 and summary["missing_events"] == ["NVDA|2026-04-30"]
    assert summary["by_track"]["resolver"] == {"open": 0, "exit_due": 1, "resolved": 1, "missing": 0}


@pytest.mark.parametrize("states, combined", [
    ([], er.STATE_OPEN),
    (["resolved", "open"], er.STATE_OPEN),
    (["resolved", "missing"], er.STATE_MISSING),
    (["open", "exit_due"], er.STATE_EXIT_DUE),
    (["resolved", "resolved"], er.STATE_RESOLVED),
])
def test_the_least_finished_track_decides(states, combined):
    assert er.combine_states(states) == combined


def _ledger_record(symbol, as_of, structure="call_calendar"):
    analysis = SimpleNamespace(
        symbol=symbol, recommendation="Candidate", metrics={}, structure_scorecards=[], rationale=[],
        selector_output={"recommendation": "Candidate", "best_structure": structure,
                         "earnings_date": EARNINGS.isoformat()},
        sample_provenance="forward_post_freeze",
        vol_snapshot={"symbol": symbol, "as_of_date": as_of, "earnings_date": EARNINGS.isoformat()},
    )
    return build_record_from_analysis(analysis)


def test_the_stores_are_read_into_one_view(tmp_path):
    ledger = RecommendationLedger(ledger_path=tmp_path / "r.sqlite")
    first = _ledger_record("AAPL", "2026-04-20")
    latest = _ledger_record("AAPL", "2026-04-24")
    ledger.record(first)
    ledger.record(latest)
    ledger.record(_ledger_record("MSFT", "2026-04-24", structure="atm_straddle"))   # not resolver-tracked
    ledger.record_resolution_payload(recommendation_id=latest.recommendation_id,
                                     candidate_shadow_outcome={"status": "ok"})
    outcomes = OutcomeStore(store_path=tmp_path / "o.sqlite")
    outcomes.insert_entry(trade_id="T1", symbol="AAPL", structure="atm_straddle", entry_date=date(2026, 4, 24),
                          setup_score=0.6, source_type="paper", earnings_date=EARNINGS, entry_mid=4.0,
                          execution_penalty_at_entry=0.03)
    events = er.load_event_resolutions(as_of=EARNINGS + timedelta(days=4), ledger=ledger, outcome_store=outcomes,
                                       baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"))
    assert list(events) == ["AAPL|2026-04-30"]
    aapl = events["AAPL|2026-04-30"]
    assert aapl["tracks"]["resolver"] == [{"id": latest.recommendation_id, "state": "resolved"}]
    assert aapl["tracks"]["paper"] == [{"id": "T1", "state": "exit_due"}]
    assert aapl["state"] == er.STATE_EXIT_DUE


def test_the_funnel_report_carries_event_exits(tmp_path):
    from services import evidence_funnel as ef

    funnel = ef.EvidenceFunnelStore(tmp_path / "f.sqlite")
    report = ef.build_evidence_funnel_report(as_of=EARNINGS, store=funnel)
    # Default stores: other tests may have written to them, so only the shape is checked.
    assert set(report["event_exits"]) >= {"events", "by_state", "by_track", "resolution_rate"}
    assert "event_exits_unavailable" not in report["warnings"]
    funnel.close()


# ── frozen cohorts ──────────────────────────────────────────────────────────


def _events(as_of=EARNINGS + timedelta(days=10)):
    return er.build_event_resolutions(
        as_of=as_of,
        paper_rows=[_paper("T1", "finalized", realized=8.0),
                    _paper("T9", "open", symbol="LATER", earnings=EARNINGS + timedelta(days=60))],
        baseline_rows=[_baseline("B1", "resolved"), _baseline("B2", "exit_missing", symbol="MSFT")],
        resolver_rows=[_resolver("R1", "ok")],
    )


def test_a_closed_window_freezes_its_members_once(tmp_path):
    manifest = ec.build_cohort_manifest(name="cohort-1", since=date(2026, 4, 1), until=EARNINGS,
                                        as_of=EARNINGS + timedelta(days=3), events=_events())
    assert [m["event_key"] for m in manifest["members"]] == ["AAPL|2026-04-30", "MSFT|2026-04-30"]
    assert manifest["members"][0]["tracks"] == {"baseline": ["B1"], "paper": ["T1"], "resolver": ["R1"]}
    path = ec.write_cohort(manifest, directory=tmp_path)
    with pytest.raises(ec.CohortError, match="already frozen"):
        ec.write_cohort({**manifest, "member_count": 0}, directory=tmp_path)
    assert json.loads(path.read_text())["members_sha256"] == manifest["members_sha256"]
    assert ec.load_cohort("cohort-1", directory=tmp_path) == manifest


@pytest.mark.parametrize("kwargs, message", [
    ({"until": EARNINGS, "as_of": EARNINGS + timedelta(days=2)}, "still open"),
    ({"until": EARNINGS - timedelta(days=40)}, "on or before"),
    ({"name": "Cohort 1"}, "lowercase"),
])
def test_an_open_or_malformed_window_cannot_be_frozen(kwargs, message):
    args = {"name": "c", "since": date(2026, 4, 1), "until": EARNINGS, "as_of": EARNINGS + timedelta(days=30),
            "events": _events(), **kwargs}
    with pytest.raises(ec.CohortError, match=message):
        ec.build_cohort_manifest(**args)


def test_an_edited_cohort_is_refused(tmp_path):
    manifest = ec.build_cohort_manifest(name="c", since=date(2026, 4, 1), until=EARNINGS,
                                        as_of=EARNINGS + timedelta(days=3), events=_events())
    path = ec.write_cohort(manifest, directory=tmp_path)
    edited = json.loads(path.read_text())
    edited["members"] = edited["members"][:1]
    path.write_text(json.dumps(edited))
    with pytest.raises(ec.CohortError, match="was edited"):
        ec.load_cohort("c", directory=tmp_path)
    with pytest.raises(ec.CohortError, match="no frozen cohort"):
        ec.load_cohort("absent", directory=tmp_path)


def test_a_cohort_report_counts_frozen_members_only():
    manifest = ec.build_cohort_manifest(name="c", since=date(2026, 4, 1), until=EARNINGS,
                                        as_of=EARNINGS + timedelta(days=3), events=_events())
    later = er.build_event_resolutions(
        as_of=EARNINGS + timedelta(days=20),
        paper_rows=[],                                                      # T1 was invalidated
        baseline_rows=[_baseline("B1", "resolved"), _baseline("B7", "resolved"), _baseline("B2", "exit_missing", symbol="MSFT")],
        resolver_rows=[_resolver("R1", "ok")],
    )
    report = ec.cohort_report(manifest, later)
    assert report["frozen_rows_missing"] == [{"event_key": "AAPL|2026-04-30", "track": "paper", "id": "T1"}]
    assert report["rows_added_after_freeze"] == [{"event_key": "AAPL|2026-04-30", "track": "baseline", "id": "B7"}]
    assert report["summary"]["by_state"] == {"open": 0, "exit_due": 0, "resolved": 1, "missing": 1}


def test_the_cli_freezes_and_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(freeze_cli, "load_event_resolutions", lambda **kwargs: _events())
    args = ["--name", "c1", "--since", "2026-04-01", "--until", "2026-04-30", "--as-of", "2026-05-10",
            "--directory", str(tmp_path)]
    assert freeze_cli.main(["freeze", *args, "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["member_count"] == 2 and not list(tmp_path.iterdir())
    assert freeze_cli.main(["freeze", *args]) == 0
    assert json.loads(capsys.readouterr().out)["member_count"] == 2
    assert freeze_cli.main(["freeze", *args]) == 2 and "already frozen" in capsys.readouterr().err
    assert freeze_cli.main(["report", "--name", "c1", "--as-of", "2026-05-10", "--directory", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["events"] == 2


def test_the_view_can_be_limited_to_given_events():
    events = er.build_event_resolutions(
        as_of=EARNINGS, paper_rows=[_paper("T1", "open"), _paper("T2", "open", symbol="MSFT")],
        baseline_rows=[], resolver_rows=[], event_keys={"MSFT|2026-04-30"},
    )
    assert list(events) == ["MSFT|2026-04-30"]


def test_events_before_the_window_are_not_members():
    events = er.build_event_resolutions(
        as_of=EARNINGS + timedelta(days=10),
        paper_rows=[_paper("T1", "finalized", realized=1.0),
                    _paper("T0", "finalized", realized=1.0, symbol="EARLY", earnings=date(2026, 3, 31))],
        baseline_rows=[], resolver_rows=[],
    )
    manifest = ec.build_cohort_manifest(name="c", since=date(2026, 4, 1), until=EARNINGS,
                                        as_of=EARNINGS + timedelta(days=3), events=events)
    assert [m["event_key"] for m in manifest["members"]] == ["AAPL|2026-04-30"]
