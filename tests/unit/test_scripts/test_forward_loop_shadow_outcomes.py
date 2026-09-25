"""Shadow outcome cohorts: universe entries, booked-strike exits, condor baseline."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_forward_loop as forward_loop
from services.baseline_evidence_store import (
    BASELINE_STRUCTURES,
    EXIT_REPRICING_BOOKED,
    EXIT_REPRICING_LEGACY,
    BaselineEvidenceStore,
    _TABLE_DDL,
    make_baseline_id,
    make_universe_baseline_id,
)
from services.evidence_report import build_evidence_report
from services.outcome_recorder import OutcomeStore
from services.recommendation_ledger import RecommendationLedger

TODAY = date(2026, 4, 21)
EARNINGS = TODAY + timedelta(days=6)


def _analysis(symbol: str, recommendation: str, structure, as_of: date = TODAY) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        recommendation=recommendation,
        confidence_pct=55.0,
        setup_score=0.4,
        metrics={"calibration_phase": "bootstrap_prior"},
        rationale=["ok"],
        selector_output={
            "recommendation": recommendation,
            "best_structure": structure,
            "earnings_date": EARNINGS.isoformat(),
        },
        structure_scorecards=[
            {"structure": "atm_straddle", "execution_penalty": 0.03},
            {"structure": "otm_strangle", "execution_penalty": 0.04},
            {"structure": "iron_condor", "execution_penalty": 0.05},
        ],
        vol_snapshot={
            "as_of_date": as_of.isoformat(),
            "earnings_date": EARNINGS.isoformat(),
            "days_to_earnings": (EARNINGS - as_of).days,
            "data_quality_score": 0.9,
            "iv_rv_har": 1.1,
            "iv_rv_yz": 1.05,
        },
    )


ENTRY_CONTEXTS = {
    "atm_straddle": {"strike": 100.0, "straddle_put_strike": 100.0, "front_expiry": "2026-05-01"},
    "otm_strangle": {"call_strike": 103.0, "put_strike": 97.0, "front_expiry": "2026-05-01"},
    "iron_condor": {
        "short_call_strike": 105.0, "long_call_strike": 110.0,
        "short_put_strike": 95.0, "long_put_strike": 90.0,
        "front_expiry": "2026-05-01", "max_loss_per_unit": 4.0,
    },
}


class _Fetcher:
    """Records every quote request; returns a structure-specific context.

    At exit a real quote echoes the booked context back; ``exit_context``
    overrides that per structure to simulate a quote on different contracts.
    """

    def __init__(self, exit_mids: dict | None = None, exit_context: dict | None = None, fail_entry: set | None = None) -> None:
        self.calls: list[dict] = []
        self.exit_mids = exit_mids or {}
        self.exit_context = exit_context or {}
        self.fail_entry = fail_entry if fail_entry is not None else set()

    def __call__(self, *, symbol, structure, earnings_date, as_of_date, context=None):
        self.calls.append({"symbol": symbol, "structure": structure, "context": context, "as_of": as_of_date})
        if context is not None or as_of_date == EARNINGS - timedelta(days=1):
            mid = self.exit_mids.get(structure, 1.0)
            quoted = self.exit_context.get(structure, context or {"rediscovered": True})
            return {"mid": mid, "context": quoted}
        if as_of_date in self.fail_entry:
            return {"mid": None, "reason": "no_option_expiries"}
        mid = 1.0 if structure == "iron_condor" else 5.0
        return {"mid": mid, "context": dict(ENTRY_CONTEXTS[structure])}


def _run(tmp_path: Path, *, recommendation: str, structure, fetcher, baseline_store, as_of=TODAY, dry_run=False):
    return forward_loop.run_forward_screener(
        today=as_of,
        dry_run=dry_run,
        store=OutcomeStore(store_path=tmp_path / "outcomes.sqlite"),
        ledger=RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite"),
        baseline_store=baseline_store,
        log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": [{"symbol": "AAPL", "status": "ranked", "dte": (EARNINGS - as_of).days}]},
        analyzer=lambda symbol, mda_client=None: _analysis(symbol, recommendation, structure, as_of),  # noqa: ARG005
        price_fetcher=fetcher,
    )


def test_no_trade_event_is_shadow_entered_once_without_touching_selector_stores(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    fetcher = _Fetcher()

    summary = _run(tmp_path, recommendation="No Trade", structure=None, fetcher=fetcher, baseline_store=baselines)

    assert summary["entries"] == 0
    assert summary["skip_reasons"] == {"recommendation_not_actionable": 1}
    assert summary["universe_shadow_entries"] == len(BASELINE_STRUCTURES) == 3
    rows = baselines.list_for_diagnostics()
    assert {row["cohort"] for row in rows} == {"universe"}
    assert {row["baseline_name"] for row in rows} == set(BASELINE_STRUCTURES)
    assert {row["selector_recommendation"] for row in rows} == {"No Trade"}
    assert {row["days_to_earnings_at_entry"] for row in rows} == {6}
    condor = next(row for row in rows if row["baseline_name"] == "always_iron_condor")
    assert condor["capital_at_risk"] == 4.0
    assert condor["entry_pricing_context_json"]["short_call_strike"] == 105.0
    assert condor["baseline_id"] == make_universe_baseline_id("AAPL", EARNINGS, "always_iron_condor")
    # Shadow rows never become selector paper trades (which feed priors/calibration).
    assert OutcomeStore(store_path=tmp_path / "outcomes.sqlite").count() == 0


def test_event_in_window_on_later_days_costs_no_extra_quotes(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    fetcher = _Fetcher()

    for offset in range(3):
        _run(
            tmp_path,
            recommendation="Watchlist",
            structure=None,
            fetcher=fetcher,
            baseline_store=baselines,
            as_of=TODAY + timedelta(days=offset),
        )

    assert len(fetcher.calls) == 3
    assert baselines.count() == 3
    # First eligible day is the entry day, whatever happened later.
    assert {row["entry_date"] for row in baselines.list_for_diagnostics()} == {TODAY.isoformat()}


def test_actionable_event_gets_paired_and_universe_rows(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    summary = _run(tmp_path, recommendation="Candidate", structure="atm_straddle", fetcher=_Fetcher(), baseline_store=baselines)

    assert summary["entries"] == 1
    assert summary["baseline_entries"] == 3
    assert summary["universe_shadow_entries"] == 3
    rows = baselines.list_for_diagnostics()
    assert sorted(row["cohort"] for row in rows) == ["paired"] * 3 + ["universe"] * 3
    assert all(row["entry_pricing_context_json"] for row in rows)


def test_dry_run_records_no_shadow_rows(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    fetcher = _Fetcher()

    _run(tmp_path, recommendation="No Trade", structure=None, fetcher=fetcher, baseline_store=baselines, dry_run=True)

    assert baselines.count() == 0
    assert fetcher.calls == []


def test_shadow_failure_does_not_block_selector_entry(tmp_path):
    class _Broken(BaselineEvidenceStore):
        def recorded_universe_baselines(self, symbol, earnings_date):
            raise sqlite3.OperationalError("database is locked")

    baselines = _Broken(tmp_path / "baselines.sqlite")

    summary = _run(tmp_path, recommendation="Candidate", structure="atm_straddle", fetcher=_Fetcher(), baseline_store=baselines)

    assert summary["universe_shadow_failures"] == 1
    assert summary["entries"] == 1
    assert summary["baseline_entries"] == 3


def test_exit_reprices_booked_contracts_and_condor_uses_return_on_risk(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    _run(tmp_path, recommendation="No Trade", structure=None, fetcher=_Fetcher(), baseline_store=baselines)

    # Straddle 5.0 -> 6.0; condor sold for 1.0 and bought back for 0.0.
    exit_fetcher = _Fetcher(exit_mids={"atm_straddle": 6.0, "otm_strangle": 4.0, "iron_condor": 0.0})
    summary = forward_loop._finalize_baseline_exits(
        baseline_store=baselines,
        price_fetcher=exit_fetcher,
        as_of=EARNINGS - timedelta(days=1),
        log_path=tmp_path / "log.jsonl",
        dry_run=False,
        mda_client=None,
    )

    assert summary == {"baseline_exits": 3, "baseline_skipped": 0}
    by_structure = {call["structure"]: call["context"] for call in exit_fetcher.calls}
    assert by_structure["atm_straddle"] == ENTRY_CONTEXTS["atm_straddle"]
    assert by_structure["iron_condor"]["short_call_strike"] == 105.0

    rows = {row["baseline_name"]: row for row in baselines.list_for_diagnostics()}
    assert {row["exit_repricing"] for row in rows.values()} == {EXIT_REPRICING_BOOKED}
    # Debit: (6 - 5) / 5 = 20% gross, minus 26 * 0.03 modeled cost.
    assert rows["always_atm_straddle"]["realized_return_pct"] == pytest.approx(20.0 - 0.78)
    # Credit: (1.0 - 0.0) / 4.0 max loss = 25% gross, minus 26 * 0.05. A zero
    # buy-back is a full win, not a missing quote.
    assert rows["always_iron_condor"]["status"] == "resolved"
    assert rows["always_iron_condor"]["realized_return_pct"] == pytest.approx(25.0 - 1.3)
    assert rows["always_iron_condor"]["realized_expansion_pct"] == pytest.approx(-100.0)


def test_legacy_row_without_booked_context_is_flagged(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    baselines.insert_entry(
        recommendation_id="rec-old",
        symbol="MSFT",
        baseline_name="always_atm_straddle",
        structure="atm_straddle",
        entry_date=TODAY,
        earnings_date=EARNINGS,
        selector_structure="otm_strangle",
        entry_mid=5.0,
        modeled_cost_pct=0.0,
        execution_penalty_at_entry=0.0,
        data_quality_score_at_entry=0.9,
        iv_rv_har_at_entry=1.0,
        iv_rv_yz_at_entry=1.0,
        quote_source_at_entry="yfinance",
        quote_quality_at_entry="paper",
    )
    fetcher = _Fetcher(exit_mids={"atm_straddle": 5.5})

    forward_loop._finalize_baseline_exits(
        baseline_store=baselines,
        price_fetcher=fetcher,
        as_of=EARNINGS - timedelta(days=1),
        log_path=tmp_path / "log.jsonl",
        dry_run=False,
        mda_client=None,
    )

    assert fetcher.calls[0]["context"] is None
    row = baselines.list_for_diagnostics()[0]
    assert row["exit_repricing"] == EXIT_REPRICING_LEGACY
    assert row["status"] == "resolved"


def test_pre_migration_database_opens_and_reads_as_paired(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_TABLE_DDL)
    conn.execute(
        "INSERT INTO baseline_trades (baseline_id, recommendation_id, symbol, baseline_name, structure, entry_date, status)"
        " VALUES ('rec|baseline|always_atm_straddle', 'rec', 'AAPL', 'always_atm_straddle', 'atm_straddle', '2026-04-01', 'open')"
    )
    conn.commit()
    conn.close()

    store = BaselineEvidenceStore(path)

    row = store.list_for_diagnostics()[0]
    assert row["cohort"] is None
    assert row["entry_pricing_context_json"] == {}
    assert store.recorded_universe_baselines("AAPL", "2026-04-02") == set()


def _resolved(store, *, baseline_id, recommendation_id, cohort, repricing, ret, selector_rec=None):
    store.insert_entry(
        recommendation_id=recommendation_id,
        baseline_id=baseline_id,
        symbol="AAPL",
        baseline_name="always_iron_condor",
        structure="iron_condor",
        entry_date=TODAY,
        earnings_date=EARNINGS,
        selector_structure=None,
        entry_mid=1.0,
        modeled_cost_pct=0.0,
        execution_penalty_at_entry=0.0,
        data_quality_score_at_entry=0.9,
        iv_rv_har_at_entry=1.0,
        iv_rv_yz_at_entry=1.0,
        quote_source_at_entry="yfinance",
        quote_quality_at_entry="paper",
        cohort=cohort,
        selector_recommendation=selector_rec,
    )
    store.update_exit(
        baseline_id=baseline_id,
        exit_date=EARNINGS - timedelta(days=1),
        exit_mid=0.5,
        realized_return_pct=ret,
        realized_expansion_pct=-50.0,
        quote_source_at_exit="yfinance",
        quote_quality_at_exit="paper",
        exit_repricing=repricing,
    )


def test_report_keeps_cohorts_and_legacy_rows_apart(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    _resolved(baselines, baseline_id=make_baseline_id("p1", "always_iron_condor"), recommendation_id="p1",
              cohort="paired", repricing=EXIT_REPRICING_BOOKED, ret=10.0)
    _resolved(baselines, baseline_id=make_baseline_id("p2", "always_iron_condor"), recommendation_id="p2",
              cohort="paired", repricing=EXIT_REPRICING_LEGACY, ret=-90.0)
    _resolved(baselines, baseline_id="u1", recommendation_id="u1", cohort="universe",
              repricing=EXIT_REPRICING_BOOKED, ret=6.0, selector_rec="Candidate")
    _resolved(baselines, baseline_id="u2", recommendation_id="u2", cohort="universe",
              repricing=EXIT_REPRICING_BOOKED, ret=8.0, selector_rec="No Trade")
    _resolved(baselines, baseline_id="u3", recommendation_id="u3", cohort="universe",
              repricing=EXIT_REPRICING_BOOKED, ret=12.0, selector_rec="Watchlist")

    report = build_evidence_report(
        baseline_store=baselines,
        outcome_store=OutcomeStore(store_path=tmp_path / "outcomes.sqlite"),
    )

    paired = report["baseline_comparison"]["always_iron_condor"]
    assert paired["n"] == 1 and paired["avg_realized_return_pct"] == 10.0
    assert report["legacy_repriced_baselines"]["n"] == 1
    universe = report["universe_shadow"]["by_baseline"]["always_iron_condor"]
    assert universe["all_events"]["n"] == 3
    assert universe["selector_actionable"]["avg_realized_return_pct"] == 6.0
    assert universe["selector_not_actionable"]["n"] == 2
    assert universe["selector_not_actionable"]["avg_realized_return_pct"] == 10.0
    assert report["universe_shadow"]["entries"] == 3


def _enter_all(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    _run(tmp_path, recommendation="No Trade", structure=None, fetcher=_Fetcher(), baseline_store=baselines)
    return baselines


def _exit(tmp_path, baselines, fetcher):
    return forward_loop._finalize_baseline_exits(
        baseline_store=baselines,
        price_fetcher=fetcher,
        as_of=EARNINGS - timedelta(days=1),
        log_path=tmp_path / "log.jsonl",
        dry_run=False,
        mda_client=None,
    )


def test_exit_quote_on_other_strikes_fails_closed(tmp_path):
    baselines = _enter_all(tmp_path)
    # The quote comes back "successful" but on a 95 call + 105 put.
    wrong = {"atm_straddle": {"strike": 95.0, "straddle_put_strike": 105.0, "front_expiry": "2026-05-01"}}

    summary = _exit(tmp_path, baselines, _Fetcher(exit_mids={"atm_straddle": 22.0}, exit_context=wrong))

    rows = {row["baseline_name"]: row for row in baselines.list_for_diagnostics()}
    straddle = rows["always_atm_straddle"]
    assert straddle["status"] == "exit_skipped"
    assert straddle["skip_reason"] == "booked_contract_mismatch"
    assert straddle["realized_return_pct"] is None
    assert summary == {"baseline_exits": 2, "baseline_skipped": 1}


@pytest.mark.parametrize(
    ("structure", "exit_context"),
    [
        ("otm_strangle", {"call_strike": 104.0, "put_strike": 97.0, "front_expiry": "2026-05-01"}),
        ("iron_condor", {**ENTRY_CONTEXTS["iron_condor"], "long_put_strike": 85.0}),
        ("atm_straddle", {**ENTRY_CONTEXTS["atm_straddle"], "front_expiry": "2026-05-08"}),
        ("atm_straddle", {"strike": 100.0, "front_expiry": "2026-05-01"}),  # put leg unverifiable
    ],
)
def test_booked_contract_check_rejects_any_leg_or_expiry_change(structure, exit_context):
    assert forward_loop._booked_contracts_match(structure, ENTRY_CONTEXTS[structure], ENTRY_CONTEXTS[structure])
    assert not forward_loop._booked_contracts_match(structure, ENTRY_CONTEXTS[structure], exit_context)
    assert not forward_loop._booked_contracts_match("call_calendar", {"strike": 100.0}, {"strike": 100.0})


def test_straddle_reprice_requires_booked_strike_on_both_legs(monkeypatch):
    from tests.unit.test_scripts.test_run_forward_loop import _install_fake_option_chain, _option_frame

    calls = _option_frame([{"contractSymbol": "C95", "strike": 95.0, "bid": 9.0, "ask": 10.0, "lastPrice": 9.5}])
    puts = _option_frame([{"contractSymbol": "P105", "strike": 105.0, "bid": 12.0, "ask": 13.0, "lastPrice": 12.5}])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=100.0)

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="atm_straddle",
        earnings_date=EARNINGS,
        as_of_date=EARNINGS - timedelta(days=1),
        context={"strike": 100.0, "front_expiry": "2026-05-01"},
    )

    assert quote["mid"] is None
    assert quote["reason"] == "booked_contract_unavailable"


def test_straddle_discovery_pairs_call_and_put_at_one_strike(monkeypatch):
    from tests.unit.test_scripts.test_run_forward_loop import _install_fake_option_chain, _option_frame

    calls = _option_frame([{"contractSymbol": "C100", "strike": 100.0, "bid": 2.0, "ask": 2.2, "lastPrice": 2.1}])
    puts = _option_frame([
        {"contractSymbol": "P100", "strike": 100.0, "bid": 1.9, "ask": 2.1, "lastPrice": 2.0},
        {"contractSymbol": "P101", "strike": 101.0, "bid": 2.4, "ask": 2.6, "lastPrice": 2.5},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=100.6)

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="atm_straddle", earnings_date=EARNINGS, as_of_date=TODAY,
    )

    assert quote["context"]["strike"] == quote["context"]["straddle_put_strike"] == 100.0


def test_failed_universe_entry_is_retried_and_attempts_are_kept(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    fetcher = _Fetcher(fail_entry={TODAY, TODAY + timedelta(days=1)})

    for offset in range(4):
        _run(
            tmp_path,
            recommendation="No Trade",
            structure=None,
            fetcher=fetcher,
            baseline_store=baselines,
            as_of=TODAY + timedelta(days=offset),
        )

    rows = baselines.list_for_diagnostics()
    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"open"}
    assert {row["entry_date"] for row in rows} == {(TODAY + timedelta(days=2)).isoformat()}
    assert {row["entry_attempt_count"] for row in rows} == {3}
    for row in rows:
        assert row["failed_entry_attempts_json"] == [
            {"date": TODAY.isoformat(), "reason": "no_option_expiries"},
            {"date": (TODAY + timedelta(days=1)).isoformat(), "reason": "no_option_expiries"},
        ]
    # 3 structures x (2 failed days + 1 success); day 4 costs nothing.
    assert len(fetcher.calls) == 9

    report = build_evidence_report(
        baseline_store=baselines,
        outcome_store=OutcomeStore(store_path=tmp_path / "outcomes.sqlite"),
    )
    attrition = report["universe_shadow"]["entry_attrition"]
    assert attrition["entered_after_retry"] == 3
    assert attrition["failed_attempts_by_reason"] == {"no_option_expiries": 6}


def test_event_that_never_quotes_is_counted_as_attrition(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    fetcher = _Fetcher(fail_entry={TODAY + timedelta(days=offset) for offset in range(2)})
    for offset in range(2):
        _run(tmp_path, recommendation="No Trade", structure=None, fetcher=fetcher,
             baseline_store=baselines, as_of=TODAY + timedelta(days=offset))

    report = build_evidence_report(
        baseline_store=baselines,
        outcome_store=OutcomeStore(store_path=tmp_path / "outcomes.sqlite"),
    )
    attrition = report["universe_shadow"]["entry_attrition"]
    assert attrition["not_entered"] == 3
    assert attrition["not_entered_by_reason"] == {"no_option_expiries": 3}


def test_paired_rows_are_never_replaced(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    kwargs = dict(
        recommendation_id="rec", symbol="AAPL", baseline_name="always_atm_straddle", structure="atm_straddle",
        entry_date=TODAY, earnings_date=EARNINGS, selector_structure="atm_straddle", modeled_cost_pct=0.0,
        execution_penalty_at_entry=0.0, data_quality_score_at_entry=0.9, iv_rv_har_at_entry=1.0,
        iv_rv_yz_at_entry=1.0, quote_source_at_entry="yfinance", quote_quality_at_entry="paper",
    )
    assert baselines.insert_entry(entry_mid=None, status="entry_skipped", skip_reason="x", **kwargs) is True
    assert baselines.insert_entry(entry_mid=5.0, **kwargs) is False
    assert baselines.list_for_diagnostics()[0]["status"] == "entry_skipped"


def test_unverified_143_label_is_not_comparable(tmp_path):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    _resolved(baselines, baseline_id=make_baseline_id("p1", "always_iron_condor"), recommendation_id="p1",
              cohort="paired", repricing="booked_strikes", ret=50.0)

    report = build_evidence_report(
        baseline_store=baselines,
        outcome_store=OutcomeStore(store_path=tmp_path / "outcomes.sqlite"),
    )

    assert "always_iron_condor" not in report["baseline_comparison"]
    assert report["legacy_repriced_baselines"]["by_exit_repricing"] == {"booked_strikes": 1}
