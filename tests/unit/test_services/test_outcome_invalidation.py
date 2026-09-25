"""Invalidated outcomes are kept for audit but never count as evidence."""
from __future__ import annotations

import json
from datetime import date

import pytest

from scripts.invalidate_outcome import main as invalidate_cli
from services.baseline_evidence_store import BaselineEvidenceStore
from services.evidence_report import build_evidence_report
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import (
    OutcomeStore,
    finalize_trade_and_update_learning,
    is_outcome_evidence_valid,
)
from services.recommendation_ledger import RecommendationLedger


def _resolved(store: OutcomeStore, symbol: str, ret: float, *, notes: dict | None = None) -> str:
    trade_id = f"{symbol}|2026-04-24|otm_strangle"
    store.insert_entry(
        trade_id=trade_id,
        recommendation_id=f"rec-{symbol}",
        symbol=symbol,
        structure="otm_strangle",
        entry_date=date(2026, 4, 24),
        earnings_date=date(2026, 4, 28),
        setup_score=0.7,
        source_type="paper",
        entry_mid=2.0,
        iv_rv_har=0.9,
        notes=json.dumps(notes) if notes else None,
    )
    store.update_exit(
        trade_id=trade_id,
        exit_date=date(2026, 4, 27),
        exit_mid=2.0,
        realized_return_pct=ret,
        realized_expansion_pct=ret,
    )
    store.mark_finalized(trade_id)
    return trade_id


def _reports(tmp_path, outcomes):
    baselines = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    report = build_evidence_report(baseline_store=baselines, outcome_store=outcomes)
    forward = build_forward_performance_diagnostics(
        outcome_store=outcomes,
        baseline_store=baselines,
        ledger=RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite"),
    )
    return report, forward


def test_extreme_invalidated_outcome_cannot_move_any_statistic(tmp_path):
    clean = OutcomeStore(tmp_path / "clean.sqlite")
    dirty = OutcomeStore(tmp_path / "dirty.sqlite")
    for store in (clean, dirty):
        for symbol, ret in (("MU", 8.96), ("NFLX", -23.81), ("NVDA", -10.0)):
            _resolved(store, symbol, ret)
    bad = _resolved(dirty, "AMZN", 900.0)
    dirty.invalidate(bad, reason="exit repriced a strike that was never booked")
    legacy = _resolved(dirty, "TTD", -500.0, notes={"evidence_invalidated": True})

    clean_report, clean_forward = _reports(tmp_path / "c", clean)
    dirty_report, dirty_forward = _reports(tmp_path / "d", dirty)

    assert dirty_report["selector_summary"] == clean_report["selector_summary"]
    assert dirty_report["selector_summary"]["n"] == 3
    assert dirty_report["selector_summary"]["avg_realized_return_pct"] == pytest.approx(-8.2833, abs=1e-4)
    assert dirty_report["simple_iv_rv_filter"] == clean_report["simple_iv_rv_filter"]
    assert dirty_report["maturity"] == clean_report["maturity"]
    assert dirty_report["evidence_quality"] == clean_report["evidence_quality"]
    assert dirty_report["invalidated_outcomes"]["n"] == 2
    assert dirty_report["invalidated_outcomes"]["by_reason"] == {
        "exit repriced a strike that was never booked": 1,
        "notes.evidence_invalidated": 1,
    }
    for key in ("performance_summary", "by_structure", "calibration_report", "benchmark_comparison"):
        assert dirty_forward[key] == clean_forward[key], key
    assert dirty_forward["resolved_outcome_count"] == 3
    assert dirty_forward["invalidated_outcome_count"] == 2
    assert not is_outcome_evidence_valid(dirty.get_trade(legacy))


def test_invalidated_open_trade_is_never_exited_or_finalized(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    store.insert_entry(
        trade_id="AMZN|open",
        symbol="AMZN",
        structure="otm_strangle",
        entry_date=date(2026, 4, 24),
        earnings_date=date(2026, 4, 28),
        setup_score=0.7,
        source_type="paper",
        entry_mid=2.0,
    )
    assert len(store.trades_due_for_exit(date(2026, 4, 27))) == 1

    result = store.invalidate("AMZN|open", reason="strikes not booked")

    assert result["learning_already_applied"] is False
    assert store.trades_due_for_exit(date(2026, 4, 27)) == []
    with pytest.raises(ValueError, match="invalidated evidence"):
        finalize_trade_and_update_learning(
            trade_id="AMZN|open",
            exit_date=date(2026, 4, 27),
            realized_return_pct=50.0,
            realized_expansion_pct=50.0,
            store=store,
        )
    assert store.get_trade("AMZN|open")["realized_return_pct"] is None


def test_invalidate_requires_reason_and_reports_applied_learning(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    trade_id = _resolved(store, "NVDA", 5.0)
    store.set_learning_update_status(trade_id, "complete")

    with pytest.raises(ValueError, match="reason"):
        store.invalidate(trade_id, reason="  ")
    with pytest.raises(ValueError, match="not found"):
        store.invalidate("missing", reason="x")
    result = store.invalidate(trade_id, reason="bad quote")

    assert result["learning_already_applied"] is True
    row = store.get_trade(trade_id)
    assert row["evidence_valid"] == 0
    assert row["invalidation_reason"] == "bad quote"
    assert row["invalidated_at"]


def test_cli_invalidates_and_lists(tmp_path, capsys):
    path = tmp_path / "outcomes.sqlite"
    store = OutcomeStore(path)
    trade_id = _resolved(store, "TTD", -40.0)
    store.set_learning_update_status(trade_id, "complete")
    store.close()

    assert invalidate_cli(["--store", str(path), "--trade-id", trade_id, "--reason", "strike not booked"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip())["invalidation_reason"] == "strike not booked"
    assert "already finalized" in captured.err

    assert invalidate_cli(["--store", str(path), "--list"]) == 0
    listed = json.loads(capsys.readouterr().out.strip())
    assert listed["evidence_valid"] is False
    assert invalidate_cli(["--store", str(path), "--trade-id", "nope", "--reason", "x"]) == 1
