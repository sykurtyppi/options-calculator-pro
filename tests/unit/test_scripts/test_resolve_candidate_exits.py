"""Tests for scripts/resolve_candidate_exits.py (PR-AE C5).

The script is a thin wrapper around the C4 resolver service. These
tests cover the wrapper's responsibilities:

  1. CLI runs end-to-end against a real ledger + stubbed store and
     exits 0 when the count-balance invariant holds.
  2. JSONL telemetry is APPENDED (not overwritten) so multiple
     daily ticks accumulate cleanly. One JSON object per row
     processed.
  3. Stdout summary contains the expected count taxonomy and the
     sub-reason breakdowns (Codex C5 watch item).
  4. days_in_awaiting_state is computed for awaiting rows by
     walking the row's revision history.
  5. CLI exits 1 when count_balance_holds is False (drift
     detection).
  6. --no-jsonl flag suppresses JSONL writes (smoke-test use).
  7. --quiet flag suppresses stdout summary.

Resolver semantic correctness is exhaustively covered in
tests/unit/test_services/test_candidate_exit_resolver.py; we
deliberately do NOT re-test resolver outcomes here.
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.resolve_candidate_exits import (
    _compute_days_in_awaiting_state,
    _format_summary,
    _outcome_to_jsonl_row,
    _run,
    main,
)
from services.candidate_exit_resolver import (
    RESOLVER_STATUS_AWAITING_CHAIN_DATA,
    RESOLVER_STATUS_OK,
    RESOLVER_STATUS_RETRYING,
    ResolverOutcome,
    ResolverRunSummary,
)
from services.recommendation_ledger import RecommendationLedger


# Reuse the seed/chain helpers from the resolver test module.
# Importing across test modules is fine for shared fixtures.
from tests.unit.test_services.test_candidate_exit_resolver import (
    _picker_provenance_block,
    _seed_forward_record,
    _StubFeatureStore,
    _toy_chain,
)


def _build_args(**overrides) -> argparse.Namespace:
    """Construct a Namespace matching the CLI's argparse output."""
    defaults = {
        "ledger_path": None,
        "feature_store_root": None,
        "now": None,
        "min_days_after_event": 3,
        "max_lookahead_trade_days": 5,
        "max_attempts": 6,
        "jsonl_path": None,
        "no_jsonl": False,
        "quiet": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ──────────────────────────────────────────────────────────────────────────
# Full pipeline: CLI end-to-end against a real ledger + stub store
# ──────────────────────────────────────────────────────────────────────────


def test_cli_run_writes_jsonl_and_returns_zero_on_clean_run(tmp_path: Path) -> None:
    """Happy path: seed an ok-resolvable forward row, run the CLI,
    verify exit code 0, JSONL file created with one row, stdout
    summary printed."""
    ledger_path = tmp_path / "ledger.sqlite"
    jsonl_path = tmp_path / "logs" / "resolutions.jsonl"

    ledger = RecommendationLedger(ledger_path=ledger_path)
    _seed_forward_record(
        ledger, recommendation_id="rec_cli_ok",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )
    ledger.close()

    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    exit_chain = _toy_chain(
        trade_date="2026-05-25", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    stub_store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        ("AAPL", "2026-05-25"): exit_chain,
    })

    args = _build_args(
        ledger_path=ledger_path,
        jsonl_path=jsonl_path,
        now=date(2026, 5, 27),
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch(
        "scripts.resolve_candidate_exits.OptionsFeatureStore",
        return_value=stub_store,
    ):
        exit_code = _run(args, stdout=stdout, stderr=stderr)

    assert exit_code == 0, f"stderr: {stderr.getvalue()}"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    # Required schema fields from the design doc
    for key in (
        "ts", "recommendation_id", "symbol", "earnings_date",
        "exit_trade_date", "attempt_number", "days_in_awaiting_state",
        "resolver_status", "simulator_status", "mid_realized_return_pct",
        "duration_ms",
    ):
        assert key in row, f"JSONL row missing required field: {key}"
    assert row["recommendation_id"] == "rec_cli_ok"
    assert row["resolver_status"] == "ok"
    assert row["mid_realized_return_pct"] == pytest.approx(30.0)
    # Awaiting-state field is null for non-awaiting outcomes
    assert row["days_in_awaiting_state"] is None

    # Stdout summary mentions the counts and exit
    summary_text = stdout.getvalue()
    assert "resolved_ok" in summary_text
    assert "scanned" in summary_text


def test_cli_jsonl_is_appended_not_overwritten_across_runs(tmp_path: Path) -> None:
    """Two consecutive CLI runs against different recommendations
    must accumulate JSONL rows, never overwrite. Daily ticks rely
    on this so historical runs stay readable for debugging."""
    ledger_path = tmp_path / "ledger.sqlite"
    jsonl_path = tmp_path / "logs" / "resolutions.jsonl"

    ledger = RecommendationLedger(ledger_path=ledger_path)
    _seed_forward_record(
        ledger, recommendation_id="rec_tick1",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )
    ledger.close()

    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    exit_chain = _toy_chain(
        trade_date="2026-05-25", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    stub_store_1 = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        ("AAPL", "2026-05-25"): exit_chain,
    })

    # Tick 1: resolves rec_tick1
    args1 = _build_args(
        ledger_path=ledger_path,
        jsonl_path=jsonl_path,
        now=date(2026, 5, 27),
        quiet=True,
    )
    with patch(
        "scripts.resolve_candidate_exits.OptionsFeatureStore",
        return_value=stub_store_1,
    ):
        assert _run(args1) == 0

    # Tick 2: seed a SECOND row and resolve it on a later "now"
    ledger2 = RecommendationLedger(ledger_path=ledger_path)
    _seed_forward_record(
        ledger2, recommendation_id="rec_tick2",
        symbol="MSFT", earnings_date="2026-05-23",
        as_of_date="2026-05-16",
    )
    ledger2.close()

    stub_store_2 = _StubFeatureStore({
        ("MSFT", "2026-05-16"): _toy_chain(
            trade_date="2026-05-16", front_expiry="2026-05-08",
            back_expiry="2026-05-22", strike=200.0,
            front_mid=1.0, back_mid=2.0,
        ),
        ("MSFT", "2026-05-26"): _toy_chain(
            trade_date="2026-05-26", front_expiry="2026-05-08",
            back_expiry="2026-05-22", strike=200.0,
            front_mid=0.4, back_mid=1.7,
        ),
    })

    args2 = _build_args(
        ledger_path=ledger_path,
        jsonl_path=jsonl_path,
        now=date(2026, 5, 28),
        quiet=True,
    )
    with patch(
        "scripts.resolve_candidate_exits.OptionsFeatureStore",
        return_value=stub_store_2,
    ):
        assert _run(args2) == 0

    # JSONL must now have BOTH rows, appended chronologically
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ids = [json.loads(line)["recommendation_id"] for line in lines]
    assert ids == ["rec_tick1", "rec_tick2"]


def test_cli_no_jsonl_flag_skips_jsonl_writes(tmp_path: Path) -> None:
    """--no-jsonl is the smoke-test/dry-run flag. Stdout summary
    still prints; JSONL file is NOT created."""
    ledger_path = tmp_path / "ledger.sqlite"
    jsonl_path = tmp_path / "logs" / "resolutions.jsonl"

    ledger = RecommendationLedger(ledger_path=ledger_path)
    _seed_forward_record(
        ledger, recommendation_id="rec_no_jsonl",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )
    ledger.close()

    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    exit_chain = _toy_chain(
        trade_date="2026-05-25", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    stub_store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        ("AAPL", "2026-05-25"): exit_chain,
    })

    args = _build_args(
        ledger_path=ledger_path,
        jsonl_path=jsonl_path,
        now=date(2026, 5, 27),
        no_jsonl=True,
    )
    with patch(
        "scripts.resolve_candidate_exits.OptionsFeatureStore",
        return_value=stub_store,
    ):
        assert _run(args) == 0
    assert not jsonl_path.exists()


def test_cli_quiet_flag_suppresses_stdout(tmp_path: Path) -> None:
    """--quiet suppresses the summary block but JSONL still writes."""
    ledger_path = tmp_path / "ledger.sqlite"
    jsonl_path = tmp_path / "logs" / "resolutions.jsonl"

    ledger = RecommendationLedger(ledger_path=ledger_path)
    _seed_forward_record(
        ledger, recommendation_id="rec_quiet",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )
    ledger.close()

    args = _build_args(
        ledger_path=ledger_path,
        jsonl_path=jsonl_path,
        now=date(2026, 5, 27),
        quiet=True,
    )
    stub_store = _StubFeatureStore({})  # nothing → all rows go awaiting
    stdout = io.StringIO()
    with patch(
        "scripts.resolve_candidate_exits.OptionsFeatureStore",
        return_value=stub_store,
    ):
        _run(args, stdout=stdout)
    assert stdout.getvalue() == ""
    # JSONL still written
    assert jsonl_path.exists()


# ──────────────────────────────────────────────────────────────────────────
# days_in_awaiting_state freshness field
# ──────────────────────────────────────────────────────────────────────────


def test_days_in_awaiting_state_returns_none_when_no_awaiting_revision(
    tmp_path: Path,
) -> None:
    """A row never marked awaiting → field is None. Defensive guard
    so the watchdog cannot mistakenly raise on a row that was never
    in the awaiting state."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger, recommendation_id="rec_never_awaiting",
    )
    result = _compute_days_in_awaiting_state(
        ledger, "rec_never_awaiting", today=date(2026, 6, 1),
    )
    assert result is None


def test_days_in_awaiting_state_computes_days_since_first_awaiting(
    tmp_path: Path,
) -> None:
    """Walk the revision history; pick the EARLIEST awaiting
    revision; return today - that revision's revised_at in calendar
    days. Pattern: a row goes awaiting today, the watchdog sees
    days_in_awaiting_state == 0 today, 5 in 5 days, and can alert
    above a threshold (e.g. 10 days)."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger, recommendation_id="rec_awaiting_freshness",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )

    # Stamp an awaiting_chain_data revision into the ledger directly
    # with a controlled revised_at so the days calculation is
    # deterministic.
    ledger.record_resolution_and_attempt(
        recommendation_id="rec_awaiting_freshness",
        candidate_shadow_outcome={
            "status": "awaiting_chain_data",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
        },
        increment_attempts=False,
    )

    # Patch the revision's revised_at to a fixed past date so the
    # delta is deterministic. We need to update the row in-place
    # because record_resolution_and_attempt stamps `datetime.now`.
    ledger._conn.execute(  # noqa: SLF001
        """
        UPDATE recommendation_revisions
           SET revised_at = ?
         WHERE recommendation_id = ?
        """,
        ("2026-05-23T00:00:00+00:00", "rec_awaiting_freshness"),
    )
    ledger._conn.commit()

    days = _compute_days_in_awaiting_state(
        ledger, "rec_awaiting_freshness", today=date(2026, 5, 30),
    )
    # 2026-05-23 → 2026-05-30 = 7 calendar days
    assert days == 7


def test_days_in_awaiting_state_uses_earliest_awaiting_revision(
    tmp_path: Path,
) -> None:
    """If a row went awaiting THEN ok THEN got re-awaiting somehow
    (edge case), the freshness field should reflect the EARLIEST
    awaiting revision so the watchdog sees the cumulative stuck
    time, not just the latest re-entry. (Today this won't happen in
    normal flow — ok is terminal — but the helper should be
    defensible against future state transitions.)"""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger, recommendation_id="rec_multi_awaiting",
    )

    # Two awaiting revisions, different revised_at — but the
    # UNIQUE(rec_id, content_hash) constraint dedupes identical
    # payloads. To get two awaiting revision rows we vary an
    # unrelated field. Use slightly different label values (this is
    # synthetic — would not occur in practice).
    ledger.record_resolution_and_attempt(
        recommendation_id="rec_multi_awaiting",
        candidate_shadow_outcome={
            "status": "awaiting_chain_data",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
            "_synthetic_variant": "first",
        },
        increment_attempts=False,
    )
    ledger.record_resolution_and_attempt(
        recommendation_id="rec_multi_awaiting",
        candidate_shadow_outcome={
            "status": "awaiting_chain_data",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
            "_synthetic_variant": "second",
        },
        increment_attempts=False,
    )

    # Force the two awaiting revisions to known revised_at values
    # in chronological order.
    ledger._conn.execute(  # noqa: SLF001
        """
        UPDATE recommendation_revisions
           SET revised_at = CASE
               WHEN record_json LIKE '%"_synthetic_variant": "first"%' THEN '2026-05-20T00:00:00+00:00'
               WHEN record_json LIKE '%"_synthetic_variant": "second"%' THEN '2026-05-27T00:00:00+00:00'
               ELSE revised_at
           END
         WHERE recommendation_id = ?
        """,
        ("rec_multi_awaiting",),
    )
    ledger._conn.commit()

    days = _compute_days_in_awaiting_state(
        ledger, "rec_multi_awaiting", today=date(2026, 5, 30),
    )
    # Earliest awaiting was 2026-05-20 → 10 days, NOT the later
    # 2026-05-27 revision's 3 days.
    assert days == 10


# ──────────────────────────────────────────────────────────────────────────
# Stdout summary formatting
# ──────────────────────────────────────────────────────────────────────────


def test_summary_includes_retrying_sub_reason_breakdown() -> None:
    """Codex C5 watch item: the summary must surface
    retrying_by_reason sub-counts. With both no_post_event_chain
    and no_entry_chain_replay retrying rows in the run, both must
    appear in the stdout block."""
    summary = ResolverRunSummary(
        scanned=3, retrying=3,
        retrying_by_reason={
            "no_post_event_chain": 2,
            "no_entry_chain_replay": 1,
        },
    )
    text = _format_summary(
        summary,
        now=date(2026, 5, 30),
        run_started_at=pd.Timestamp("2026-05-30T12:00:00", tz="UTC").to_pydatetime(),
        duration_seconds=1.5,
    )
    assert "retrying:" in text
    assert "no_post_event_chain" in text
    assert "no_entry_chain_replay" in text


def test_summary_includes_permanently_failed_sub_reason_breakdown() -> None:
    """Same contract for permanently_failed: sub-reasons surface in
    the stdout block, not just the rollup."""
    summary = ResolverRunSummary(
        scanned=2,
        permanently_failed_by_reason={
            "no_post_event_chain": 1,
            "simulator_skipped:negative_debit": 1,
        },
    )
    text = _format_summary(
        summary,
        now=date(2026, 5, 30),
        run_started_at=pd.Timestamp("2026-05-30T12:00:00", tz="UTC").to_pydatetime(),
        duration_seconds=0.5,
    )
    assert "permanently_failed:" in text
    assert "no_post_event_chain" in text
    assert "simulator_skipped:negative_debit" in text


def test_summary_reports_count_balance_status() -> None:
    """The summary surfaces count_balance_holds verbatim so log
    monitoring can grep for it. Default-true summary prints
    'true'."""
    text = _format_summary(
        ResolverRunSummary(),
        now=date(2026, 5, 30),
        run_started_at=pd.Timestamp("2026-05-30T12:00:00", tz="UTC").to_pydatetime(),
        duration_seconds=0.0,
    )
    assert "count_balance_holds:    true" in text


def test_summary_skipped_ineligible_is_not_misleadingly_zero() -> None:
    """REGRESSION (Codex C5 audit P2): the original summary printed
    `skipped_ineligible: 0` unconditionally, which suggested the
    resolver had measured zero ineligible rows when in fact those
    rows are filtered out at the SQL layer and never seen by the
    resolver. Replaced with an explicit `not_measured` indicator
    that names the SQL prefilter as the reason."""
    text = _format_summary(
        ResolverRunSummary(scanned=5, resolved_ok=5),
        now=date(2026, 5, 30),
        run_started_at=pd.Timestamp("2026-05-30T12:00:00", tz="UTC").to_pydatetime(),
        duration_seconds=0.0,
    )
    # The line must mention "not_measured" or the SQL prefilter
    # source — never a bare 0 that pretends to be a measurement.
    assert "skipped_ineligible:" in text
    assert (
        "not_measured" in text
        or "list_pending_candidate_exit_resolutions" in text
    ), (
        "skipped_ineligible must be labeled as not measured (SQL "
        "prefilter) rather than reported as a misleading 0."
    )


# ──────────────────────────────────────────────────────────────────────────
# Exit-code behavior
# ──────────────────────────────────────────────────────────────────────────


def test_cli_returns_one_when_count_balance_violated(tmp_path: Path) -> None:
    """If the resolver run produces a summary where
    count_balance_holds is False (a bug somewhere in the bucketing
    logic), the CLI must exit 1 so cron/launchd can flag the run."""
    ledger_path = tmp_path / "ledger.sqlite"
    ledger = RecommendationLedger(ledger_path=ledger_path)
    ledger.close()

    args = _build_args(
        ledger_path=ledger_path,
        jsonl_path=tmp_path / "logs" / "x.jsonl",
        now=date(2026, 5, 27),
        no_jsonl=True,
        quiet=True,
    )

    # Stub the resolver to return a deliberately-imbalanced summary.
    bogus_summary = ResolverRunSummary(scanned=5, resolved_ok=2)
    # 5 scanned but only 2 in any bucket → invariant violated.
    assert not bogus_summary.count_balance_holds

    with patch(
        "scripts.resolve_candidate_exits.resolve_pending_candidate_exits",
        return_value=([], bogus_summary),
    ), patch(
        "scripts.resolve_candidate_exits.OptionsFeatureStore",
        return_value=_StubFeatureStore({}),
    ):
        stderr = io.StringIO()
        exit_code = _run(args, stderr=stderr)
    assert exit_code == 1
    assert "count_balance_holds" in stderr.getvalue()


def test_cli_main_returns_two_on_fatal_pre_resolver_error(tmp_path: Path) -> None:
    """If the wrapper itself throws BEFORE the resolver runs (e.g.
    OptionsFeatureStore construction raises), main() must return 2
    so the exit code distinguishes "ran but had problems" (1) from
    "could not run" (2)."""
    args = [
        "--ledger-path", str(tmp_path / "nonexistent" / "ledger.sqlite"),
        "--feature-store-root", "/path/that/cannot/exist/12345",
        "--no-jsonl", "--quiet",
        "--now", "2026-05-27",
    ]

    # Force a fatal error inside _run by patching the resolver call
    # to raise an unexpected exception. The wrapper's outer try/except
    # must catch and return 2.
    with patch(
        "scripts.resolve_candidate_exits.resolve_pending_candidate_exits",
        side_effect=RuntimeError("synthetic fatal pre-resolver error"),
    ):
        exit_code = main(args)
    assert exit_code == 2


# ──────────────────────────────────────────────────────────────────────────
# Per-row JSONL schema spot-checks
# ──────────────────────────────────────────────────────────────────────────


def test_jsonl_row_for_ok_outcome_carries_pnl_and_no_failure_reason(
    tmp_path: Path,
) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    outcome = ResolverOutcome(
        recommendation_id="rec_ok_row",
        symbol="AAPL",
        earnings_date="2026-05-22",
        exit_trade_date="2026-05-25",
        attempt_number=0,
        resolver_status=RESOLVER_STATUS_OK,
        simulator_status="ok",
        mid_realized_return_pct=30.0,
    )
    row = _outcome_to_jsonl_row(
        outcome, ledger=ledger, today=date(2026, 5, 27), duration_ms=42.0,
    )
    assert row["resolver_status"] == "ok"
    assert row["mid_realized_return_pct"] == 30.0
    assert row["failure_reason"] is None
    assert row["days_in_awaiting_state"] is None
    assert "error_message" not in row


def test_jsonl_row_for_retrying_carries_failure_reason(tmp_path: Path) -> None:
    """Codex C5 watch item at the per-row layer: a retrying
    outcome must carry failure_reason in the JSONL row so
    operators can grep by reason without re-deriving from other
    fields."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    outcome = ResolverOutcome(
        recommendation_id="rec_retry",
        symbol="MSFT",
        earnings_date="2026-05-22",
        exit_trade_date=None,
        attempt_number=2,
        resolver_status=RESOLVER_STATUS_RETRYING,
        simulator_status=None,
        mid_realized_return_pct=None,
        failure_reason="no_post_event_chain",
    )
    row = _outcome_to_jsonl_row(
        outcome, ledger=ledger, today=date(2026, 6, 1), duration_ms=12.0,
    )
    assert row["resolver_status"] == "retrying"
    assert row["failure_reason"] == "no_post_event_chain"
    assert row["exit_trade_date"] is None
