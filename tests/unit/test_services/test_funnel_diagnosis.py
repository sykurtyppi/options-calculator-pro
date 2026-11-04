"""Per-event funnel diagnosis: final stops, binding stage, claimable throughput."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts import diagnose_evidence_funnel as cli
from services import evidence_funnel as ef
from services.evidence_report import MIN_RESOLVED_SAMPLE
from services.funnel_diagnosis import (
    GATE_SAMPLE,
    diagnose_funnel,
    read_funnel_rows,
    read_recommendation_rows,
    render_text,
)

AS_OF = date(2026, 11, 3)
A, D, X = ef.OUTCOME_ADVANCED, ef.OUTCOME_DEFERRED, ef.OUTCOME_DROPPED
ENTRY = (ef.STAGE_DISCOVERED, ef.STAGE_DATA_QUALIFIED, ef.STAGE_ANALYZED, ef.STAGE_ELIGIBLE, ef.STAGE_RECORDED)


def _d(s: str) -> date:
    return date.fromisoformat(s)


class _Funnel:
    def __init__(self, path: Path) -> None:
        self.store = ef.EvidenceFunnelStore(path)
        self.cycles: set[str] = set()

    def row(self, day: str, stage: str, outcome: str, symbol: str, earnings: str,
            reason: str | None = None, trade_id: str | None = None,
            recommendation_id: str | None = None) -> None:
        cycle = f"c-{day}"
        if cycle not in self.cycles:
            self.store.start_cycle(cycle_run_id=cycle, as_of=_d(day), universe_size=500, discovered=10,
                                   discovery_source="test")
            self.cycles.add(cycle)
        self.store.record(cycle_run_id=cycle, as_of=_d(day), stage=stage, outcome=outcome, symbol=symbol,
                          earnings_date=_d(earnings), reason=reason, trade_id=trade_id,
                          recommendation_id=recommendation_id)

    def entry(self, day: str, symbol: str, earnings: str, *, upto: str, trade_id: str | None = None) -> None:
        for stage in ENTRY[: ENTRY.index(upto) + 1]:
            self.row(day, stage, A, symbol, earnings,
                     trade_id=trade_id if stage == ef.STAGE_RECORDED else None)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    f = _Funnel(tmp_path / "funnel.sqlite")
    # AAPL: too early, then entered, exited and claimable.
    f.row("2026-10-01", ef.STAGE_DISCOVERED, A, "AAPL", "2026-10-30")
    f.row("2026-10-01", ef.STAGE_DATA_QUALIFIED, D, "AAPL", "2026-10-30", ef.REASON_TOO_EARLY)
    f.entry("2026-10-20", "AAPL", "2026-10-30", upto=ef.STAGE_RECORDED, trade_id="T-AAPL")
    f.row("2026-10-29", ef.STAGE_EXIT_DUE, A, "AAPL", "2026-10-30", trade_id="T-AAPL")
    f.row("2026-10-31", ef.STAGE_RESOLVED, A, "AAPL", "2026-10-30", trade_id="T-AAPL")
    f.row("2026-10-31", ef.STAGE_CLAIMABLE, A, "AAPL", "2026-10-30", trade_id="T-AAPL")
    # MSFT and TSLA: analysed, the IV/RV gate says no trade (MSFT on two cycles).
    for day in ("2026-10-20", "2026-10-21"):
        f.entry(day, "MSFT", "2026-10-28", upto=ef.STAGE_ANALYZED)
        f.row(day, ef.STAGE_ELIGIBLE, X, "MSFT", "2026-10-28", "no_trade:iv_rv_gate")
    f.entry("2026-10-20", "TSLA", "2026-10-25", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "TSLA", "2026-10-25", "no_trade:iv_rv_gate")
    # NVDA: too early, event still ahead. AMD: too early, event passed, never picked up.
    f.row("2026-10-20", ef.STAGE_DATA_QUALIFIED, D, "NVDA", "2026-11-20", ef.REASON_TOO_EARLY)
    f.row("2026-10-01", ef.STAGE_DATA_QUALIFIED, D, "AMD", "2026-10-15", ef.REASON_TOO_EARLY)
    # UBER: entered, exit ahead. SHOP: entered, event passed, nothing after.
    f.entry("2026-10-28", "UBER", "2026-11-10", upto=ef.STAGE_RECORDED, trade_id="T-UBER")
    f.entry("2026-10-05", "SHOP", "2026-10-10", upto=ef.STAGE_RECORDED, trade_id="T-SHOP")
    # AMZN: entered, the exit could not be priced.
    f.entry("2026-10-15", "AMZN", "2026-10-22", upto=ef.STAGE_RECORDED, trade_id="T-AMZN")
    f.row("2026-10-22", ef.STAGE_RESOLVED, X, "AMZN", "2026-10-22", "exit_missing:no_quote", trade_id="T-AMZN")
    f.store.close()
    return tmp_path / "funnel.sqlite"


def _diag(path: Path, **kw):
    rows = read_funnel_rows(path)
    return diagnose_funnel(rows["events"], rows["cycles"], as_of=kw.pop("as_of", AS_OF), **kw)


def test_each_event_ends_in_exactly_one_final_state(store_path: Path) -> None:
    ev = _diag(store_path)["events"]
    assert ev["total"] == 8
    assert ev["by_final"] == {"claimable": 1, "in_flight": 1, "no_further_record": 1, "stopped": 3,
                              "deferred": 1, "deferred_then_lost": 1}
    assert ev["stopped_by_stage"] == {
        "eligible": {"events": 2, "top_reasons": [{"reason": "no_trade:iv_rv_gate", "events": 2}]},
        "resolved": {"events": 1, "top_reasons": [{"reason": "exit_missing:no_quote", "events": 1}]},
    }
    assert ev["deferred_then_lost_by_reason"] == [{"reason": ef.REASON_TOO_EARLY, "events": 1}]
    assert ev["in_flight"] == {"in_flight:recorded": 1, "no_further_record:recorded": 1}


def test_an_event_analysed_on_many_cycles_counts_once(store_path: Path) -> None:
    # MSFT was analysed twice; it is still one stopped event, not two.
    ev = _diag(store_path)["events"]
    assert ev["stopped_by_stage"]["eligible"]["events"] == 2
    assert ev["mean_analysis_passes"] == pytest.approx(7 / 8, abs=0.01)  # MSFT 2; AAPL TSLA UBER SHOP AMZN 1; NVDA AMD 0


def test_binding_constraint_is_the_stage_that_ends_the_most_events(store_path: Path) -> None:
    b = _diag(store_path)["binding_constraint"]
    assert b["stage"] == "eligible" and b["events_lost"] == 2 and b["share_of_events"] == 0.25
    assert b["top_reasons"][0] == {"reason": "no_trade:iv_rv_gate", "events": 2}


def test_lost_deferrals_count_against_their_stage(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "f.sqlite")
    for sym in ("A", "B"):
        f.row("2026-10-01", ef.STAGE_DATA_QUALIFIED, D, sym, "2026-10-10", ef.REASON_TOO_EARLY)
    f.entry("2026-10-01", "C", "2026-10-10", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-01", ef.STAGE_ELIGIBLE, X, "C", "2026-10-10", "no_trade:x")
    f.store.close()
    b = _diag(tmp_path / "f.sqlite")["binding_constraint"]
    assert b["stage"] == "data_qualified" and b["events_lost"] == 2
    assert b["top_reasons"] == [{"reason": f"deferred_then_lost:{ef.REASON_TOO_EARLY}", "events": 2}]


def test_throughput_and_weeks_to_gate(store_path: Path) -> None:
    d = _diag(store_path)
    tp = d["throughput"]
    assert tp["claimable_trades_total"] == 1
    assert tp["claimable_per_week"] == pytest.approx(1 / 8, abs=1e-3)
    assert tp["remaining_to_gate"] == GATE_SAMPLE - 1
    assert tp["weeks_to_gate_at_current_rate"] == (GATE_SAMPLE - 1) * 8
    week = {w["week"]: w for w in d["weekly"]}
    assert week["2026-10-26"]["claimable"] == 1
    assert sum(w["events_first_seen"] for w in d["weekly"]) == 8
    assert sum(w["cycles"] for w in d["weekly"]) == len({"2026-10-01", "2026-10-05", "2026-10-15", "2026-10-20",
                                                        "2026-10-21", "2026-10-22", "2026-10-28", "2026-10-29",
                                                        "2026-10-31"})


def test_no_recent_claimable_means_never_and_a_met_sample_means_zero(store_path: Path) -> None:
    later = _diag(store_path, as_of=AS_OF + timedelta(weeks=12))["throughput"]
    assert later["claimable_per_week"] == 0 and later["weeks_to_gate_at_current_rate"] is None
    met = _diag(store_path, gate_sample=1)["throughput"]
    assert met["remaining_to_gate"] == 0 and met["weeks_to_gate_at_current_rate"] == 0


def test_window_and_as_of_bound_what_is_counted(store_path: Path) -> None:
    since = _diag(store_path, since=date(2026, 10, 15))["events"]
    assert since["total"] == 5  # MSFT, TSLA, NVDA, UBER, AMZN: first seen on or after Oct 15
    before = _diag(store_path, as_of=date(2026, 10, 21))
    assert before["events"]["by_final"]["claimable"] == 0  # AAPL's claim came later
    assert before["throughput"]["claimable_trades_total"] == 0


def test_reading_leaves_the_store_untouched(store_path: Path) -> None:
    before = hashlib.sha256(store_path.read_bytes()).hexdigest()
    _diag(store_path)
    assert hashlib.sha256(store_path.read_bytes()).hexdigest() == before


def test_gate_sample_matches_the_report_gate() -> None:
    assert GATE_SAMPLE == MIN_RESOLVED_SAMPLE


def test_cli_prints_text_and_json(store_path: Path, capsys) -> None:
    assert cli.main(["--store", str(store_path), "--as-of", "2026-11-03"]) == 0
    out = capsys.readouterr().out
    assert "Binding constraint: eligible ends 2 events (25%)" in out
    assert "Claimable trades so far: 1 of 30" in out and "about 232 more weeks" in out
    assert cli.main(["--store", str(store_path), "--as-of", "2026-11-03", "--json"]) == 0
    assert '"binding_constraint"' in capsys.readouterr().out
    assert cli.main(["--store", str(store_path.with_name("missing.sqlite"))]) == 1


def test_empty_store_renders(tmp_path: Path) -> None:
    ef.EvidenceFunnelStore(tmp_path / "e.sqlite").close()
    d = _diag(tmp_path / "e.sqlite")
    assert d["events"]["total"] == 0 and d["binding_constraint"] is None
    assert "Binding constraint: none" in render_text(d)


def test_an_event_on_the_as_of_day_has_not_passed(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "f.sqlite")
    f.row("2026-11-01", ef.STAGE_DATA_QUALIFIED, D, "A", AS_OF.isoformat(), ef.REASON_TOO_EARLY)
    f.store.close()
    assert _diag(tmp_path / "f.sqlite")["events"]["by_final"]["deferred"] == 1


def test_a_later_duplicate_drop_does_not_undo_an_entered_trade(tmp_path: Path) -> None:
    # Entered on day 1; re-analysed on day 2 and dropped as a duplicate of
    # the active trade. The event is in flight, not stopped.
    f = _Funnel(tmp_path / "f.sqlite")
    f.entry("2026-11-01", "A", "2026-11-20", upto=ef.STAGE_RECORDED, trade_id="T-A")
    f.entry("2026-11-02", "A", "2026-11-20", upto=ef.STAGE_ELIGIBLE)
    f.row("2026-11-02", ef.STAGE_RECORDED, X, "A", "2026-11-20", ef.REASON_DUPLICATE_ACTIVE_TRADE)
    f.store.close()
    ev = _diag(tmp_path / "f.sqlite")["events"]
    assert ev["by_final"]["in_flight"] == 1 and ev["by_final"]["stopped"] == 0


def test_follow_up_uses_the_actual_last_recorded_date(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "f.sqlite")
    f.entry("2026-10-01", "A", "2026-10-20", upto=ef.STAGE_RECORDED, trade_id="T-A")
    f.row("2026-10-02", ef.STAGE_RECORDED, X, "A", "2026-10-20",
          ef.REASON_DUPLICATE_ACTIVE_TRADE)
    f.store.close()

    event = _diag(tmp_path / "f.sqlite")["events"]["no_further_record_events"][0]
    assert event["last_recorded_as_of"] == "2026-10-02"


def test_a_trade_counts_once_in_the_week_it_first_became_claimable(store_path: Path) -> None:
    f = _Funnel(store_path)
    f.cycles.update({"c-2026-11-02"})
    f.store.start_cycle(cycle_run_id="c-2026-11-02", as_of=date(2026, 11, 2), universe_size=1, discovered=0,
                        discovery_source="test")
    f.row("2026-11-02", ef.STAGE_CLAIMABLE, A, "AAPL", "2026-10-30", trade_id="T-AAPL")
    f.store.close()
    d = _diag(store_path)
    week = {w["week"]: w for w in d["weekly"]}
    assert week["2026-10-26"]["claimable"] == 1 and week["2026-11-02"]["claimable"] == 0
    assert d["throughput"]["claimable_trades_total"] == 1
    # Claims before the window are not this window's weekly throughput.
    assert all(w["claimable"] == 0 for w in _diag(store_path, since=date(2026, 11, 1))["weekly"])


def test_a_met_sample_is_zero_weeks_even_with_no_recent_claims(store_path: Path) -> None:
    tp = _diag(store_path, as_of=AS_OF + timedelta(weeks=12), gate_sample=1)["throughput"]
    assert tp["claimable_per_week"] == 0 and tp["weeks_to_gate_at_current_rate"] == 0


def test_reading_a_missing_store_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(Exception):
        read_funnel_rows(missing)
    assert not missing.exists()


def test_no_eligible_structure_reports_underlying_eligibility_causes(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "f.sqlite")
    for symbol in ("A", "B"):
        f.entry("2026-10-20", symbol, "2026-10-28", upto=ef.STAGE_ANALYZED)
        f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, symbol, "2026-10-28", "no_eligible_structure")
    f.store.close()

    recommendations = [
        {
            "symbol": "A", "earnings_date": "2026-10-28", "as_of_date": "2026-10-20",
            "created_at": "2026-10-20T21:00:00+00:00", "binding_gate": "no_eligible_structure",
            "structure_scorecards_json": [
                {"structure": "atm_straddle", "eligible": False,
                 "eligibility_flags": ["spread_exceeds_absolute_threshold"], "expected_edge_pct": -3.0},
                {"structure": "call_calendar", "eligible": False,
                 "eligibility_flags": ["missing_structure_leg_spread", "surface_quality_record_only"],
                 "expected_edge_pct": -2.0},
            ],
        },
        {
            "symbol": "B", "earnings_date": "2026-10-28", "as_of_date": "2026-10-20",
            "created_at": "2026-10-20T21:01:00+00:00", "binding_gate": "no_eligible_structure",
            "structure_scorecards_json": [
                {"structure": "atm_straddle", "eligible": False,
                 "eligibility_flags": ["cannot_form_atm_straddle"], "expected_edge_pct": -1.0},
            ],
        },
    ]
    rows = read_funnel_rows(tmp_path / "f.sqlite")
    d = diagnose_funnel(rows["events"], rows["cycles"], as_of=AS_OF, recommendations=recommendations)

    detail = d["events"]["stop_reason_details"]["no_eligible_structure"]
    assert detail["events"] == 2 and detail["matched_recommendations"] == 2
    assert detail["eligibility_categories"] == [
        {"category": "missing_legs", "events": 2},
        {"category": "spread_or_liquidity", "events": 1},
        {"category": "surface_quality", "events": 1},
    ]
    assert detail["eligibility_flags"] == [
        {"flag": "cannot_form_atm_straddle", "events": 1},
        {"flag": "missing_structure_leg_spread", "events": 1},
        {"flag": "spread_exceeds_absolute_threshold", "events": 1},
        {"flag": "surface_quality_record_only", "events": 1},
    ]
    assert "Eligibility causes (2/2 usable; 2 ledger matches): missing_legs (2), spread_or_liquidity (1), surface_quality (1)" in render_text(d)


def test_stop_diagnostics_match_production_no_trade_reason_prefix(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "funnel.sqlite")
    f.entry("2026-10-20", "A", "2026-10-28", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28",
          "no_trade:no_eligible_structure")
    f.store.close()
    recommendations = [{
        "symbol": "A", "earnings_date": "2026-10-28", "as_of_date": "2026-10-20",
        "created_at": "2026-10-20T21:00:00Z", "binding_gate": "no_eligible_structure",
        "structure_scorecards_json": '[{"structure":"atm_straddle","eligible":false,'
                                      '"eligibility_flags":["spread_exceeds_absolute_threshold"]}]',
    }]

    details = diagnose_funnel(read_funnel_rows(tmp_path / "funnel.sqlite")["events"], [],
                              as_of=date(2026, 11, 3), recommendations=recommendations)["events"]["stop_reason_details"]

    assert details["no_eligible_structure"]["matched_recommendations"] == 1


def test_non_positive_edge_reports_best_eligible_candidate(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "f.sqlite")
    f.entry("2026-10-20", "A", "2026-10-28", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28", "non_positive_edge")
    f.store.close()

    recommendations = [{
        "symbol": "A", "earnings_date": "2026-10-28", "as_of_date": "2026-10-20",
        "created_at": "2026-10-20T21:00:00+00:00", "binding_gate": "non_positive_edge",
        "structure_scorecards_json": [
            {"structure": "atm_straddle", "eligible": True, "eligibility_flags": [],
             "composite_structure_score": 85.0, "expected_edge_pct": -2.25},
            {"structure": "iron_condor", "eligible": True, "eligibility_flags": [],
             "composite_structure_score": 70.0, "expected_edge_pct": -0.15},
            {"structure": "bad_quotes", "eligible": False, "eligibility_flags": ["spread_too_wide"],
             "expected_edge_pct": 9.0},
        ],
        "selected_structure": "atm_straddle",
    }]
    rows = read_funnel_rows(tmp_path / "f.sqlite")
    d = diagnose_funnel(rows["events"], rows["cycles"], as_of=AS_OF, recommendations=recommendations)

    detail = d["events"]["stop_reason_details"]["non_positive_edge"]
    assert detail["events"] == 1 and detail["matched_recommendations"] == 1
    assert detail["best_eligible_edge_pct"] == {"events": 1, "min": -2.25, "median": -2.25, "max": -2.25}
    assert detail["event_details"] == [{
        "symbol": "A", "earnings_date": "2026-10-28", "as_of_date": "2026-10-20",
        "match_status": "usable", "match_method": "event_key",
        "best_structure": "atm_straddle", "best_eligible_edge_pct": -2.25,
    }]
    assert "Selected eligible edge at non-positive stops (1/1 usable): median -2.25%, range -2.25% to -2.25%" in render_text(d)


def test_eligibility_categories_do_not_mislabel_event_metadata(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "f.sqlite")
    f.entry("2026-10-20", "A", "2026-10-28", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28", "no_eligible_structure")
    f.store.close()
    recommendations = [{
        "symbol": "A", "earnings_date": "2026-10-28", "as_of_date": "2026-10-20",
        "created_at": "2026-10-20T21:00:00Z", "binding_gate": "no_eligible_structure",
        "structure_scorecards_json": [{
            "structure": "atm_straddle", "eligible": False,
            "eligibility_flags": ["missing_earnings_date", "missing_option_chain", "invalid_near_term_dte",
                                  "structure_spread_exceeds_absolute_threshold",
                                  "cannot_form_otm_strangle", "cannot_form_calendar"],
        }],
    }]

    detail = diagnose_funnel(
        read_funnel_rows(tmp_path / "f.sqlite")["events"], [], as_of=AS_OF,
        recommendations=recommendations,
    )["events"]["stop_reason_details"]["no_eligible_structure"]
    assert detail["eligibility_categories"] == [
        {"category": "dte", "events": 1},
        {"category": "event_metadata", "events": 1},
        {"category": "missing_legs", "events": 1},
        {"category": "option_chain", "events": 1},
        {"category": "spread_or_liquidity", "events": 1},
    ]
    assert detail["events_with_eligibility_causes"] == 1
    assert detail["event_details"][0]["match_status"] == "usable"


def test_lost_and_orphaned_events_are_identified_for_follow_up(store_path: Path) -> None:
    ev = _diag(store_path)["events"]

    assert ev["deferred_then_lost_events"] == [{
        "symbol": "AMD", "earnings_date": "2026-10-15", "stage": "data_qualified",
        "reason": ef.REASON_TOO_EARLY, "last_recorded_as_of": "2026-10-01", "analysis_passes": 0,
    }]
    assert ev["no_further_record_events"] == [{
        "symbol": "SHOP", "earnings_date": "2026-10-10", "stage": "recorded",
        "reason": None, "last_recorded_as_of": "2026-10-05", "analysis_passes": 1,
    }]
    text = render_text(_diag(store_path))
    assert (f"Deferred/lost event: AMD 2026-10-15 at data_qualified "
            f"({ef.REASON_TOO_EARLY}; last 2026-10-01)") in text
    assert "No-further-record event: SHOP 2026-10-10 after recorded (last 2026-10-05)" in text


def test_recommendation_rows_are_read_only_and_missing_ledger_is_optional(tmp_path: Path) -> None:
    ledger = tmp_path / "recommendations.sqlite"
    conn = sqlite3.connect(ledger)
    conn.execute("CREATE TABLE recommendations (symbol TEXT, earnings_date TEXT, as_of_date TEXT, "
                 "created_at TEXT, binding_gate TEXT, structure_scorecards_json TEXT)")
    conn.execute("INSERT INTO recommendations VALUES (?, ?, ?, ?, ?, ?)",
                 ("A", "2026-10-28", "2026-10-20", "2026-10-20T21:00:00Z", "non_positive_edge", "[]"))
    conn.commit()
    conn.close()
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()

    assert read_recommendation_rows(ledger) == [{
        "symbol": "A", "earnings_date": "2026-10-28", "as_of_date": "2026-10-20",
        "created_at": "2026-10-20T21:00:00Z", "binding_gate": "non_positive_edge",
        "structure_scorecards_json": "[]",
    }]
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before
    missing = tmp_path / "missing.sqlite"
    assert read_recommendation_rows(missing) == [] and not missing.exists()


def test_incompatible_recommendation_ledger_is_optional(tmp_path: Path) -> None:
    ledger = tmp_path / "legacy.sqlite"
    sqlite3.connect(ledger).close()
    status: dict = {}

    assert read_recommendation_rows(ledger, status=status) == []
    assert status["status"] == "unavailable"
    assert "recommendations table" in status["error"]


def test_cli_discloses_missing_optional_recommendation_ledger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    f = _Funnel(tmp_path / "funnel.sqlite")
    f.entry("2026-10-20", "A", "2026-10-28", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28", "non_positive_edge")
    f.store.close()
    ledger = tmp_path / "missing.sqlite"

    assert cli.main(["--store", str(tmp_path / "funnel.sqlite"),
                     "--recommendations", str(ledger), "--as-of", "2026-11-03"]) == 0
    captured = capsys.readouterr()
    assert "Recommendation enrichment missing:" in captured.out
    assert "recommendation enrichment missing:" in captured.err
    assert not ledger.exists()


def test_cli_continues_when_optional_recommendation_ledger_is_incompatible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    f = _Funnel(tmp_path / "funnel.sqlite")
    f.entry("2026-10-20", "A", "2026-10-28", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28", "non_positive_edge")
    f.store.close()
    ledger = tmp_path / "legacy.sqlite"
    sqlite3.connect(ledger).close()

    assert cli.main(["--store", str(tmp_path / "funnel.sqlite"),
                     "--recommendations", str(ledger), "--as-of", "2026-11-03",
                     "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["recommendation_ledger"]["status"] == "unavailable"
    assert payload["events"]["stop_reason_details"]["non_positive_edge"][
        "events_with_selected_edge"] == 0
    assert "recommendation enrichment unavailable" in captured.err


def test_latest_recommendation_revision_is_used(tmp_path: Path) -> None:
    ledger = tmp_path / "recommendations.sqlite"
    conn = sqlite3.connect(ledger)
    conn.execute("CREATE TABLE recommendations (recommendation_id TEXT, symbol TEXT, earnings_date TEXT, "
                 "as_of_date TEXT, created_at TEXT, binding_gate TEXT, selected_structure TEXT, "
                 "structure_scorecards_json TEXT, selector_output_json TEXT)")
    conn.execute("CREATE TABLE recommendation_revisions (revision_id INTEGER PRIMARY KEY, "
                 "recommendation_id TEXT, revised_at TEXT, content_hash TEXT, record_json TEXT)")
    conn.execute("INSERT INTO recommendations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        "r1", "A", "2026-10-28", "2026-10-20", "2026-10-20T20:00:00Z", "non_positive_edge",
        "atm_straddle", '[{"structure":"atm_straddle","eligible":true,"expected_edge_pct":-3.0}]', "{}"))
    conn.execute("INSERT INTO recommendation_revisions VALUES (?, ?, ?, ?, ?)", (
        1, "r1", "2026-10-20T21:00:00Z", "hash", '{"recommendation_id":"r1","symbol":"A",'
        '"earnings_date":"2026-10-28","as_of_date":"2026-10-20",'
        '"created_at":"2026-10-20T20:00:00Z","binding_gate":"non_positive_edge",'
        '"selected_structure":"iron_condor","structure_scorecards":['
        '{"structure":"iron_condor","eligible":true,"expected_edge_pct":-0.5}]}'))
    conn.commit()
    conn.close()

    row = read_recommendation_rows(ledger)[0]
    assert row["selected_structure"] == "iron_condor"
    assert row["structure_scorecards_json"][0]["expected_edge_pct"] == -0.5


def test_latest_revision_effective_gate_controls_inclusion(tmp_path: Path) -> None:
    ledger = tmp_path / "recommendations.sqlite"
    conn = sqlite3.connect(ledger)
    conn.execute("CREATE TABLE recommendations (recommendation_id TEXT, symbol TEXT, earnings_date TEXT, "
                 "as_of_date TEXT, created_at TEXT, binding_gate TEXT, selected_structure TEXT, "
                 "structure_scorecards_json TEXT, selector_output_json TEXT)")
    conn.execute("CREATE TABLE recommendation_revisions (revision_id INTEGER PRIMARY KEY, "
                 "recommendation_id TEXT, revised_at TEXT, content_hash TEXT, record_json TEXT)")
    conn.execute("INSERT INTO recommendations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        "r1", "A", "2026-10-28", "2026-10-20", "2026-10-20T20:00:00Z", "iv_rv_gate",
        None, "[]", "{}"))
    conn.execute("INSERT INTO recommendation_revisions VALUES (?, ?, ?, ?, ?)", (
        1, "r1", "2026-10-20T21:00:00Z", "hash", '{"recommendation_id":"r1","symbol":"A",'
        '"earnings_date":"2026-10-28","as_of_date":"2026-10-20",'
        '"created_at":"2026-10-20T20:00:00Z","binding_gate":"non_positive_edge",'
        '"selected_structure":"iron_condor","structure_scorecards":['
        '{"structure":"iron_condor","eligible":true,"expected_edge_pct":-0.5}]}'))
    conn.commit()
    conn.close()

    rows = read_recommendation_rows(ledger)
    assert len(rows) == 1
    assert rows[0]["binding_gate"] == "non_positive_edge"


def test_cli_enriches_stops_from_recommendation_ledger(tmp_path: Path, capsys) -> None:
    f = _Funnel(tmp_path / "funnel.sqlite")
    f.entry("2026-10-20", "A", "2026-10-28", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28", "no_eligible_structure")
    f.store.close()
    ledger = tmp_path / "recommendations.sqlite"
    conn = sqlite3.connect(ledger)
    conn.execute("CREATE TABLE recommendations (symbol TEXT, earnings_date TEXT, as_of_date TEXT, "
                 "created_at TEXT, binding_gate TEXT, structure_scorecards_json TEXT)")
    conn.execute("INSERT INTO recommendations VALUES (?, ?, ?, ?, ?, ?)", (
        "A", "2026-10-28", "2026-10-20", "2026-10-20T21:00:00Z", "no_eligible_structure",
        '[{"structure":"atm_straddle","eligible":false,'
        '"eligibility_flags":["spread_exceeds_absolute_threshold"]}]'))
    conn.commit()
    conn.close()

    assert cli.main(["--store", str(tmp_path / "funnel.sqlite"), "--recommendations", str(ledger),
                     "--as-of", "2026-11-03"]) == 0
    assert "Eligibility causes (1/1 usable; 1 ledger matches): spread_or_liquidity (1)" in capsys.readouterr().out


# ── matching a stop to its ledger row ───────────────────────────────────────


def _card(structure: str, edge: float, composite: float = 80.0) -> dict:
    return {"structure": structure, "eligible": True, "eligibility_flags": [],
            "composite_structure_score": composite, "expected_edge_pct": edge}


def _ledger_row(recommendation_id: str, created_at: str, gate: str, cards: list) -> dict:
    return {"recommendation_id": recommendation_id, "symbol": "A", "earnings_date": "2026-10-28",
            "as_of_date": "2026-10-20", "created_at": created_at, "binding_gate": gate,
            "structure_scorecards_json": cards}


def _edge_detail(tmp_path: Path, recommendation_id: str | None, recommendations: list) -> dict:
    f = _Funnel(tmp_path / "f.sqlite")
    f.entry("2026-10-20", "A", "2026-10-28", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28", "no_trade:non_positive_edge",
          recommendation_id=recommendation_id)
    f.store.close()
    rows = read_funnel_rows(tmp_path / "f.sqlite")
    return diagnose_funnel(rows["events"], rows["cycles"], as_of=AS_OF,
                           recommendations=recommendations)["events"]["stop_reason_details"]["non_positive_edge"]


def test_funnel_rows_carry_the_recommendation_id(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "f.sqlite")
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28", "no_trade:x", recommendation_id="rec-1")
    f.store.close()
    assert read_funnel_rows(tmp_path / "f.sqlite")["events"][0]["recommendation_id"] == "rec-1"


def test_a_stop_is_explained_by_its_own_ledger_row_not_a_later_ui_analysis(tmp_path: Path) -> None:
    # The forward loop's row, then a UI analysis of the same symbol and day
    # with the same gate but different numbers, written later.
    loop = _ledger_row("rec-loop", "2026-10-20T21:30:00+00:00", "non_positive_edge", [_card("atm_straddle", -2.0)])
    ui = _ledger_row("rec-ui", "2026-10-20T23:00:00+00:00", "non_positive_edge", [_card("iron_condor", -0.1)])
    detail = _edge_detail(tmp_path, "rec-loop", [loop, ui])
    [event] = detail["event_details"]
    assert (event["match_method"], event["best_structure"], event["best_eligible_edge_pct"]) == \
        ("recommendation_id", "atm_straddle", -2.0)


def test_a_missing_or_mismatched_ledger_row_is_not_borrowed_from_another_analysis(tmp_path: Path) -> None:
    ui = _ledger_row("rec-ui", "2026-10-20T23:00:00+00:00", "non_positive_edge", [_card("iron_condor", -0.1)])
    [event] = _edge_detail(tmp_path, "rec-loop", [ui])["event_details"]
    assert (event["match_status"], event["match_method"], event["best_structure"]) == \
        ("unmatched", "recommendation_id", None)
    # Its own row exists but records another gate: still not this stop's evidence.
    loop = _ledger_row("rec-loop", "2026-10-20T21:30:00+00:00", "execution_penalty_high", [_card("x", -1.0)])
    [event] = _edge_detail(tmp_path / "b", "rec-loop", [loop, ui])["event_details"]
    assert event["match_status"] == "unmatched"


def test_a_stop_recorded_without_an_id_falls_back_to_the_event_key(tmp_path: Path) -> None:
    older = _ledger_row("rec-1", "2026-10-20T21:30:00+00:00", "non_positive_edge", [_card("atm_straddle", -2.0)])
    newer = _ledger_row("rec-2", "2026-10-20T23:00:00+00:00", "non_positive_edge", [_card("iron_condor", -0.1)])
    [event] = _edge_detail(tmp_path, None, [older, newer])["event_details"]
    assert (event["match_method"], event["best_structure"]) == ("event_key", "iron_condor")


def test_no_eligible_stops_match_by_id_too(tmp_path: Path) -> None:
    f = _Funnel(tmp_path / "f.sqlite")
    f.entry("2026-10-20", "A", "2026-10-28", upto=ef.STAGE_ANALYZED)
    f.row("2026-10-20", ef.STAGE_ELIGIBLE, X, "A", "2026-10-28", "no_trade:no_eligible_structure",
          recommendation_id="rec-loop")
    f.store.close()
    bad = {"structure": "atm_straddle", "eligible": False}
    loop = _ledger_row("rec-loop", "2026-10-20T21:30:00+00:00", "no_eligible_structure",
                       [{**bad, "eligibility_flags": ["spread_exceeds_absolute_threshold"]}])
    ui = _ledger_row("rec-ui", "2026-10-20T23:00:00+00:00", "no_eligible_structure",
                     [{**bad, "eligibility_flags": ["missing_option_chain"]}])
    rows = read_funnel_rows(tmp_path / "f.sqlite")
    detail = diagnose_funnel(rows["events"], rows["cycles"], as_of=AS_OF,
                             recommendations=[loop, ui])["events"]["stop_reason_details"]["no_eligible_structure"]
    [event] = detail["event_details"]
    assert event["match_method"] == "recommendation_id"
    assert event["eligibility_flags"] == ["spread_exceeds_absolute_threshold"]
