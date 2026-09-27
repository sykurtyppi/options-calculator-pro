"""Health alerts for evidence that degrades quietly instead of failing loudly."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

from services.baseline_evidence_store import BaselineEvidenceStore
from services.evidence_health import (
    EvidenceHealthConfig,
    _check_evidence_integrity,
    build_evidence_health_status,
)
from services.outcome_recorder import OutcomeStore

NOW = datetime(2026, 9, 27, 22, 30, tzinfo=timezone.utc)
TODAY = NOW.date()


def _cfg(tmp_path, **overrides):
    return EvidenceHealthConfig(
        expected_date=TODAY,
        outcome_store_path=tmp_path / "outcomes.sqlite",
        baseline_store_path=tmp_path / "baselines.sqlite",
        **overrides,
    )


def _trade(store, trade_id, *, earnings, status="finalized", reason=None):
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure="otm_strangle",
        entry_date=earnings - timedelta(days=6), earnings_date=earnings,
        setup_score=0.6, source_type="paper", entry_mid=2.0,
    )
    if status in {"finalized", "exited"}:
        store.update_exit(trade_id=trade_id, exit_date=earnings - timedelta(days=1), exit_mid=2.2,
                          realized_return_pct=5.0, realized_expansion_pct=5.0)
        if status == "finalized":
            store.mark_finalized(trade_id)
    elif status == "exit_missing":
        store.record_exit_attempt_failure(trade_id, reason=reason or "booked_contract_unavailable",
                                          attempted_on=earnings - timedelta(days=1))
        store.mark_missing_exits(earnings)


def _checks(result):
    return [issue["message"] for issue in result["issues"]]


def test_clean_stores_raise_nothing(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    for i in range(6):
        _trade(store, f"T{i}", earnings=TODAY - timedelta(days=10 + i))
    BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    result = _check_evidence_integrity(_cfg(tmp_path), NOW)

    assert result["issues"] == []
    assert result["summary"]["selector"]["exit_attrition_rate"] == 0.0


def test_missing_stores_are_left_to_the_store_check(tmp_path):
    result = _check_evidence_integrity(_cfg(tmp_path), NOW)
    assert result == {"summary": {"window_days": 90}, "issues": []}


def test_pre_migration_store_is_skipped_not_alarmed(tmp_path):
    conn = sqlite3.connect(tmp_path / "outcomes.sqlite")
    conn.execute("CREATE TABLE outcome_trades (trade_id TEXT, status TEXT)")
    conn.commit()
    conn.close()

    assert _check_evidence_integrity(_cfg(tmp_path), NOW)["issues"] == []


def test_unreadable_store_warns_instead_of_going_silent(tmp_path):
    (tmp_path / "outcomes.sqlite").write_bytes(b"this is not a sqlite database" * 100)

    result = _check_evidence_integrity(_cfg(tmp_path), NOW)

    assert any("Could not read the selector store" in message for message in _checks(result))


def test_stuck_finalizing_trade_warns(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _trade(store, "STUCK", earnings=TODAY - timedelta(days=3), status="exited")
    assert store.claim_for_finalization("STUCK")
    store._conn.execute("UPDATE outcome_trades SET updated_at = '2026-09-20 00:00:00' WHERE trade_id = 'STUCK'")
    store._conn.commit()
    _trade(store, "FRESH", earnings=TODAY - timedelta(days=2), status="exited")
    assert store.claim_for_finalization("FRESH")  # just claimed: not stuck

    result = _check_evidence_integrity(_cfg(tmp_path), NOW)

    assert result["summary"]["selector"]["stuck_finalizing"] == ["STUCK"]
    assert any("stuck in 'finalizing'" in message for message in _checks(result))


def test_open_trade_past_its_exit_day_warns(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _trade(store, "AMD", earnings=TODAY - timedelta(days=100), status="open")
    _trade(store, "FUTURE", earnings=TODAY + timedelta(days=5), status="open")

    result = _check_evidence_integrity(_cfg(tmp_path), NOW)

    assert result["summary"]["selector"]["open_past_exit_day"] == ["AMD"]
    assert any("still open after their T-1 exit day" in message for message in _checks(result))


def test_high_selector_exit_attrition_warns_only_with_enough_sample(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _trade(store, "R1", earnings=TODAY - timedelta(days=5))
    _trade(store, "M1", earnings=TODAY - timedelta(days=6), status="exit_missing")
    _trade(store, "M2", earnings=TODAY - timedelta(days=7), status="exit_missing", reason="booked_contract_mismatch")

    small = _check_evidence_integrity(_cfg(tmp_path), NOW)
    assert small["summary"]["selector"]["exit_attrition_rate"] == 2 / 3
    assert small["issues"] == []  # 3 < min sample of 5

    _trade(store, "R2", earnings=TODAY - timedelta(days=8))
    _trade(store, "R3", earnings=TODAY - timedelta(days=9))
    result = _check_evidence_integrity(_cfg(tmp_path), NOW)

    assert result["summary"]["selector"]["exit_missing_by_reason"] == {
        "booked_contract_mismatch": 1, "booked_contract_unavailable": 1,
    }
    assert any("Selector exit attrition is 40%" in message for message in _checks(result))
    assert _check_evidence_integrity(_cfg(tmp_path, max_exit_attrition_rate=0.5), NOW)["issues"] == []


def test_attrition_outside_the_window_is_ignored(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    for i in range(5):
        _trade(store, f"OLD{i}", earnings=TODAY - timedelta(days=200 + i), status="exit_missing")

    assert _check_evidence_integrity(_cfg(tmp_path), NOW)["issues"] == []


def test_invalidated_outcome_that_reached_learning_stores_warns(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _trade(store, "TTD", earnings=TODAY - timedelta(days=30))
    store.set_learning_update_status("TTD", "complete")
    store._conn.execute(
        "UPDATE outcome_trades SET notes = ? WHERE trade_id = 'TTD'",
        (json.dumps({"evidence_invalidated": {"reason_code": "x"}}),),
    )
    store._conn.commit()

    result = _check_evidence_integrity(_cfg(tmp_path), NOW)

    assert result["summary"]["selector"]["invalidated_after_learning"] == ["TTD"]
    assert any("backfill_prior_store_timestamps.py" in issue["fix"] for issue in result["issues"])


def _baseline(store, bid, *, earnings, status, cohort="universe", skip_reason=None):
    store.insert_entry(
        recommendation_id=bid, baseline_id=bid, symbol=bid, baseline_name="always_atm_straddle",
        structure="atm_straddle", entry_date=earnings - timedelta(days=6), earnings_date=earnings,
        selector_structure=None, entry_mid=None if status == "entry_skipped" else 5.0,
        modeled_cost_pct=0.0, execution_penalty_at_entry=0.0, data_quality_score_at_entry=0.9,
        iv_rv_har_at_entry=1.0, iv_rv_yz_at_entry=1.0, quote_source_at_entry="yfinance",
        quote_quality_at_entry="paper", cohort=cohort,
        status="entry_skipped" if status == "entry_skipped" else "open", skip_reason=skip_reason,
    )
    if status in {"resolved", "exit_skipped"}:
        store.update_exit(baseline_id=bid, exit_date=earnings - timedelta(days=1),
                          exit_mid=6.0 if status == "resolved" else None,
                          realized_return_pct=10.0 if status == "resolved" else None,
                          realized_expansion_pct=None, quote_source_at_exit="yfinance",
                          quote_quality_at_exit="paper", status=status,
                          skip_reason=None if status == "resolved" else "booked_contract_unavailable")


def test_universe_entry_failures_and_baseline_attrition_warn(tmp_path):
    store = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    past = TODAY - timedelta(days=10)
    for i in range(3):
        _baseline(store, f"OK{i}", earnings=past, status="resolved")
    for i in range(3):
        _baseline(store, f"SKIP{i}", earnings=past, status="entry_skipped", skip_reason="no_option_expiries")
    _baseline(store, "XS1", earnings=past, status="exit_skipped")
    _baseline(store, "XS2", earnings=past, status="exit_skipped")
    # Window still open: a current failure is not final attrition yet.
    _baseline(store, "LIVE", earnings=TODAY + timedelta(days=4), status="entry_skipped", skip_reason="x")

    result = _check_evidence_integrity(_cfg(tmp_path), NOW)

    summary = result["summary"]["baselines"]
    assert summary["universe_entry_failures"] == 3
    assert summary["universe_entry_failures_by_reason"] == {"no_option_expiries": 3}
    assert summary["exit_attrition_rate"] == 2 / 5
    messages = _checks(result)
    assert any("Universe shadow entries failed for 38%" in message for message in messages)
    assert any("Baseline exit attrition is 40%" in message for message in messages)


def test_integrity_issues_reach_the_aggregate_status(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _trade(store, "AMD", earnings=TODAY - timedelta(days=100), status="open")

    status = build_evidence_health_status(config=_cfg(tmp_path, report_dir=tmp_path), now=NOW)

    assert status["evidence_integrity"]["selector"]["open_past_exit_day"] == ["AMD"]
    assert any(issue["check"] == "evidence_integrity" for issue in status["issues"])
