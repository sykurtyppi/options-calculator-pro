"""PR G: the claim contract fails closed, and event identity holds under concurrency.

Each test reproduces a failure found in review of #169/#170:
- a leg quoted after the capture time passed as claimable (age -300 s);
- an entry with no exit verdict stayed claimable;
- quotes after an early close (13:00) passed as in-session;
- impossible OCC expiration dates passed the symbol check;
- 24 processes recording one event got 2 sequence numbers between them;
- 24 processes opening a pre-F4 ledger: 13 crashed on "duplicate column".
"""
from __future__ import annotations

import multiprocessing
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import claim_contract as cc
from services.evidence_quality import evaluate_evidence_quality
from services.market_calendar import nyse_early_closes, session_close
from services.recommendation_ledger import RecommendationLedger, build_record_from_analysis

CALL = "AAPL260424C00100000"
PUT = "AAPL260424P00100000"


def _quote(*, captured, call_time, put_time=None, call=CALL, put=PUT):
    return {
        "symbol": "AAPL", "quote_source": "marketdata_app", "quote_timestamp": captured,
        "bid_ask_mid": {"legs": {
            "call": {"contract": call, "bid": 2.4, "ask": 2.6, "mid": 2.5, "quote_time": call_time},
            "put": {"contract": put, "bid": 1.9, "ask": 2.1, "mid": 2.0, "quote_time": put_time or call_time},
        }},
    }


# ── point-in-time quotes ────────────────────────────────────────────────────


def test_a_leg_quoted_after_the_capture_is_look_ahead():
    future = cc.evaluate_quote_contract(_quote(captured="2026-04-21T17:00:00+00:00",
                                               call_time="2026-04-21T17:05:00+00:00"))
    assert not future.passed and "claim_quote_after_capture" in future.reasons
    # Within the clock-skew allowance it is the same moment.
    skewed = cc.evaluate_quote_contract(_quote(captured="2026-04-21T17:00:00+00:00",
                                               call_time="2026-04-21T17:00:04+00:00"))
    assert skewed.passed, skewed.reasons


def test_an_early_close_ends_the_session_at_13_00():
    assert session_close(date(2026, 11, 27)) .hour == 13 and session_close(date(2026, 11, 25)).hour == 16
    assert date(2026, 12, 24) in nyse_early_closes(2026) and date(2026, 7, 3) not in nyse_early_closes(2026)
    # Black Friday 2026, captured 13:30 ET (18:30 UTC): quotes at 14:00 ET are after the close.
    late = cc.evaluate_quote_contract(_quote(captured="2026-11-27T18:30:00+00:00",
                                             call_time="2026-11-27T19:00:00+00:00"))
    assert "claim_call_quote_outside_session" in late.reasons
    # The closing quotes (12:59 ET) are fresh against the 13:00 close, not a 16:00 one.
    closing = cc.evaluate_quote_contract(_quote(captured="2026-11-27T18:30:00+00:00",
                                                call_time="2026-11-27T17:59:30+00:00",
                                                call="AAPL261204C00100000", put="AAPL261204P00100000"))
    assert closing.passed, closing.reasons
    stale = cc.evaluate_quote_contract(_quote(captured="2026-11-27T21:00:00+00:00",
                                              call_time="2026-11-27T17:40:00+00:00"))
    assert "claim_quote_stale" in stale.reasons


@pytest.mark.parametrize("contract", [
    "AAPL261332C00100000",     # month 13
    "AAPL260231C00100000",     # 31 February
    "AAPL260424C00000000",     # zero strike
])
def test_impossible_contracts_fail_identity(contract):
    check = cc.evaluate_quote_contract(_quote(captured="2026-04-21T20:30:00+00:00",
                                              call_time="2026-04-21T19:59:40+00:00", call=contract))
    assert "claim_call_non_standard_contract" in check.reasons


# ── claims fail closed ──────────────────────────────────────────────────────

V = cc.CONTRACT_VERSION
PASSED = {"legs": {"call": {"contract": "AAPL260424C00100000"}},
          "claim": {"passed": True, "reasons": [], "contract_version": V,
                    "leg_contracts": {"call": "AAPL260424C00100000"}}}


@pytest.mark.parametrize("claim_allowed, scenarios, entry_version, expected", [
    (True, PASSED, V, True),
    (True, None, V, False),                                           # no exit verdict
    (True, {"scenario_values": {}}, V, False),                        # verdict absent
    (True, {"claim": "passed"}, V, False),                            # malformed
    (True, {"claim": {"passed": 1, "contract_version": V}}, V, False),  # passed must be a bool
    (True, PASSED, "conservative_fill_v1", False),                    # entry checked under v1
    (True, {"claim": {"passed": True, "contract_version": "conservative_fill_v1"}}, V, False),
    (False, PASSED, V, False),                                        # an exit never upgrades an entry
    (None, PASSED, V, None),                                          # never classified
])
def test_the_exit_verdict_is_required(claim_allowed, scenarios, entry_version, expected):
    result, _ = cc.apply_exit_claim(claim_allowed=claim_allowed, evidence_reasons=[],
                                    exit_execution_scenarios=scenarios, entry_contract_version=entry_version,
                                    entry_execution_scenarios=PASSED)
    assert result is expected


def test_reports_count_claims_under_the_current_version_only():
    assert cc.claimable_under_current_contract({"claim_allowed": 1, "claim_contract_version": V}) is True
    assert cc.claimable_under_current_contract({"claim_allowed": 1, "claim_contract_version": None}) is False
    assert cc.claimable_under_current_contract({"claim_allowed": 1,
                                                "claim_contract_version": "conservative_fill_v1"}) is False
    assert cc.claimable_under_current_contract({"claim_allowed": 0, "claim_contract_version": V}) is False
    assert cc.claimable_under_current_contract({"claim_allowed": None}) is None


def test_entries_carry_the_contract_version():
    quote = {**_quote(captured="2026-04-21T20:30:00+00:00", call_time="2026-04-21T19:59:40+00:00"),
             "surface_quality": {"status": "clean_surface"}}
    result = evaluate_evidence_quality(quote_payload=quote)
    assert result.claim_allowed and result.claim_contract_version == V == "conservative_fill_v3"


# ── event identity under concurrency ────────────────────────────────────────


def _analysis(day: int):
    return SimpleNamespace(
        symbol="AAPL", recommendation="Candidate", metrics={}, structure_scorecards=[], rationale=[],
        selector_output={"recommendation": "Candidate", "best_structure": "call_calendar",
                         "earnings_date": "2026-05-01"},
        vol_snapshot={"symbol": "AAPL", "as_of_date": f"2026-04-{day:02d}", "earnings_date": "2026-05-01"},
    )


def _record_three(args):
    path, worker = args
    ledger = RecommendationLedger(ledger_path=Path(path))
    for i in range(3):
        ledger.record(build_record_from_analysis(_analysis(1 + (worker * 3 + i) % 28)))
    ledger.close()
    return "ok"


def _open(path):
    try:
        RecommendationLedger(ledger_path=Path(path)).close()
        return "ok"
    except Exception as exc:  # noqa: BLE001 - the failure is what is being counted
        return f"{type(exc).__name__}: {exc}"


def _sequences(path):
    with sqlite3.connect(path) as conn:
        return [row[0] for row in conn.execute("SELECT event_sequence FROM recommendations ORDER BY 1")]


def test_concurrent_writers_get_unique_event_sequences(tmp_path):
    path = str(tmp_path / "ledger.sqlite")
    RecommendationLedger(ledger_path=Path(path)).close()
    with multiprocessing.get_context("fork").Pool(8) as pool:
        assert pool.map(_record_three, [(path, worker) for worker in range(8)]) == ["ok"] * 8
    assert _sequences(path) == list(range(1, 25))


def _legacy_ledger(path):
    """A ledger as written before event identity existed."""
    ledger = RecommendationLedger(ledger_path=Path(path))
    for day in (20, 21, 22):
        ledger.record(build_record_from_analysis(_analysis(day)))
    ledger.close()
    with sqlite3.connect(path) as conn:
        for index in ("idx_recommendations_event", "ux_recommendations_event_sequence"):
            conn.execute(f"DROP INDEX IF EXISTS {index}")
        conn.execute("ALTER TABLE recommendations DROP COLUMN event_key")
        conn.execute("ALTER TABLE recommendations DROP COLUMN event_sequence")


def test_concurrent_startup_upgrades_an_old_ledger_once(tmp_path):
    path = str(tmp_path / "ledger.sqlite")
    _legacy_ledger(path)
    with multiprocessing.get_context("fork").Pool(12) as pool:
        assert pool.map(_open, [path] * 12) == ["ok"] * 12
    assert _sequences(path) == [1, 2, 3]


def test_duplicate_sequences_already_written_are_renumbered_and_then_refused(tmp_path):
    path = str(tmp_path / "ledger.sqlite")
    ledger = RecommendationLedger(ledger_path=Path(path))
    for day in (20, 21, 22):
        ledger.record(build_record_from_analysis(_analysis(day)))
    ledger.close()
    with sqlite3.connect(path) as conn:     # what the race left behind before this fix
        conn.execute("DROP INDEX ux_recommendations_event_sequence")
        conn.execute("UPDATE recommendations SET event_sequence = 1")
    RecommendationLedger(ledger_path=Path(path)).close()
    assert _sequences(path) == [1, 2, 3]
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE recommendations SET event_sequence = 1")


# ── stores and reports keep versions apart ──────────────────────────────────


def test_a_baseline_exit_needs_the_entry_version_too(tmp_path):
    from services.baseline_evidence_store import BaselineEvidenceStore

    store = BaselineEvidenceStore(tmp_path / "b.sqlite")
    for baseline_id, version in (("B_V2", V), ("B_OLD", None)):
        store.insert_entry(
            recommendation_id=baseline_id, symbol="AAPL", baseline_name="always_straddle", structure="atm_straddle",
            entry_date=date(2026, 4, 20), earnings_date=date(2026, 4, 24), selector_structure=None, entry_mid=4.5,
            modeled_cost_pct=None, execution_penalty_at_entry=None, data_quality_score_at_entry=None,
            iv_rv_har_at_entry=None, iv_rv_yz_at_entry=None, quote_source_at_entry="marketdata_app",
            quote_quality_at_entry=None, claim_allowed=True, baseline_id=baseline_id, claim_contract_version=version,
            entry_execution_scenarios=PASSED,
        )
        store.update_exit(baseline_id=baseline_id, exit_date=date(2026, 4, 23), exit_mid=5.0, realized_return_pct=11.0,
                          realized_expansion_pct=None, quote_source_at_exit="marketdata_app",
                          quote_quality_at_exit=None, exit_execution_scenarios=PASSED)
    claims = {row["baseline_id"]: bool(row["claim_allowed"]) for row in store.list_for_diagnostics()}
    assert claims == {"B_V2": True, "B_OLD": False}


def test_reports_ignore_claims_resolved_under_an_older_contract(tmp_path):
    from services.baseline_evidence_store import BaselineEvidenceStore
    from services.evidence_report import _claimable_resolved_count
    from services.forward_performance_diagnostics import build_forward_performance_diagnostics
    from services.outcome_recorder import OutcomeStore

    store = OutcomeStore(store_path=tmp_path / "o.sqlite")
    for trade_id, symbol in (("T_OLD", "AAPL"), ("T_V2", "MSFT")):      # two events
        store.insert_entry(trade_id=trade_id, symbol=symbol, structure="atm_straddle", entry_date=date(2026, 4, 20),
                           setup_score=0.6, source_type="paper", earnings_date=date(2026, 4, 24), entry_mid=4.0,
                           claim_allowed=True, claim_contract_version=V, entry_execution_scenarios=PASSED)
        store.update_exit(trade_id=trade_id, exit_date=date(2026, 4, 23), exit_mid=4.4, realized_return_pct=10.0,
                          realized_expansion_pct=5.0, exit_execution_scenarios=PASSED)
        store.mark_finalized(trade_id)
    # A row claimed under v1, before versions were stored.
    store._conn.execute("UPDATE outcome_trades SET claim_contract_version = NULL WHERE trade_id = 'T_OLD'")
    store._conn.commit()
    rows = store.list_for_diagnostics(evidence="valid", source="forward")
    assert _claimable_resolved_count(rows) == 1
    report = build_forward_performance_diagnostics(
        ledger=RecommendationLedger(ledger_path=tmp_path / "r.sqlite"), outcome_store=store,
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )["claimable_evidence"]
    assert (report["claimable_count"], report["non_claimable_count"]) == (1, 1)
