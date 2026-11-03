"""Per-event funnel diagnosis: final stops, binding stage, claimable throughput."""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts import diagnose_evidence_funnel as cli
from services import evidence_funnel as ef
from services.evidence_report import MIN_RESOLVED_SAMPLE
from services.funnel_diagnosis import GATE_SAMPLE, diagnose_funnel, read_funnel_rows, render_text

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
            reason: str | None = None, trade_id: str | None = None) -> None:
        cycle = f"c-{day}"
        if cycle not in self.cycles:
            self.store.start_cycle(cycle_run_id=cycle, as_of=_d(day), universe_size=500, discovered=10,
                                   discovery_source="test")
            self.cycles.add(cycle)
        self.store.record(cycle_run_id=cycle, as_of=_d(day), stage=stage, outcome=outcome, symbol=symbol,
                          earnings_date=_d(earnings), reason=reason, trade_id=trade_id)

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
