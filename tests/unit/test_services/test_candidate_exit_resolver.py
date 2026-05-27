"""Tests for services/candidate_exit_resolver.py (PR-AE C4).

The resolver is the algorithmic core of PR-AE: it walks ledger rows
tagged forward_post_freeze, finds the post-event chain, calls the
shared shadow simulator, and writes the resolved candidate outcome
back atomically. This file covers:

  1. Eligibility filter integration — only forward_post_freeze
     rows are processed; historical replay rows must never appear.
  2. Picker-provenance fail-closed paths — empty / malformed /
     put_side_not_yet_supported produce permanently_failed:
     no_picker_provenance.
  3. Post-event chain selection — BDay lookahead picks the first
     non-empty chain in the window; missing entry chain →
     permanently_failed:no_entry_chain_replay.
  4. Window-elapsed three-way split — awaiting_chain_data (no
     counter bump), retrying (counter bump), permanently_failed:
     no_post_event_chain at MAX_ATTEMPTS (counter bump → terminal).
  5. Simulator status passthrough — ok flows through as resolver
     ok; skipped:* simulator statuses wrap as permanently_failed:
     simulator_<sim_status>.
  6. Idempotency on re-runs — duplicate ok outcomes don't pile up
     revisions, duplicate retrying outcomes still bump the counter
     (Codex C1c clarification).
  7. Counter rules — ok and awaiting_chain_data do NOT bump;
     retrying and permanently_failed:* do.
  8. Failure isolation — one row raising never aborts the run.
  9. Summary count balance — sum of buckets == scanned for every
     run (the CLI invariant).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from services.candidate_exit_resolver import (
    RESOLVER_STATUS_AWAITING_CHAIN_DATA,
    RESOLVER_STATUS_OK,
    RESOLVER_STATUS_PERMFAIL_NO_ENTRY_CHAIN_REPLAY,
    RESOLVER_STATUS_PERMFAIL_NO_PICKER_PROVENANCE,
    RESOLVER_STATUS_PERMFAIL_NO_POST_EVENT_CHAIN,
    RESOLVER_STATUS_PERMFAIL_SIMULATOR_ERROR,
    RESOLVER_STATUS_PERMFAIL_SIMULATOR_PREFIX,
    RESOLVER_STATUS_RETRYING,
    ResolverOutcome,
    ResolverRunSummary,
    _extract_candidate_selection,
    _post_event_bday_candidates,
    resolve_pending_candidate_exits,
)
from services.recommendation_ledger import RecommendationLedger


# ──────────────────────────────────────────────────────────────────────────
# Fixtures + chain builders (kept local for test independence)
# ──────────────────────────────────────────────────────────────────────────


def _toy_chain(
    *,
    trade_date: str,
    front_expiry: str,
    back_expiry: str,
    strike: float,
    front_mid: float,
    back_mid: float,
    front_iv: float = 0.30,
    back_iv: float = 0.30,
) -> pd.DataFrame:
    """Minimal two-row call chain matching the simulator's expected
    columns. Identical shape to the C2 test helper, kept independent
    here so resolver tests don't depend on simulator-test fixtures."""
    return pd.DataFrame([
        {"call_put": "C", "trade_date": trade_date, "expiry": front_expiry,
         "strike": strike, "bid": front_mid - 0.05, "ask": front_mid + 0.05,
         "mid": front_mid, "iv": front_iv, "open_interest": 500, "volume": 50,
         "spread_pct": 10.0, "liquidity_score": 0.8},
        {"call_put": "C", "trade_date": trade_date, "expiry": back_expiry,
         "strike": strike, "bid": back_mid - 0.05, "ask": back_mid + 0.05,
         "mid": back_mid, "iv": back_iv, "open_interest": 800, "volume": 80,
         "spread_pct": 5.0, "liquidity_score": 0.9},
    ])


def _picker_provenance_block(
    *,
    side: str = "call",
    strike: float = 200.0,
    front_expiry: str = "2026-05-08",
    back_expiry: str = "2026-05-22",
    status: str = "ok",
    structure: str = "call_calendar",
) -> Dict[str, Any]:
    """Mirror the shape produced by _build_experimental_contract_selection
    in web/api/edge_engine.py. The resolver pulls
    candidate_contracts.candidate_selection out of this block."""
    return {
        "structure": structure,
        "labels": {"experimental": True, "shadow_mode": True,
                   "not_execution_guidance": True,
                   "out_of_sample_validated": False},
        "status": status,
        "candidate_contracts": {
            "shadow_status": "ok",
            "candidate_selection": {
                "side": side,
                "strike": strike,
                "front_expiry": front_expiry,
                "back_expiry": back_expiry,
            },
            "legacy_selection": {
                "side": side,
                "strike": strike,
                "front_expiry": front_expiry,
                "back_expiry": back_expiry,
            },
            "pickers_diverged": False,
        },
    }


def _seed_forward_record(
    ledger: RecommendationLedger,
    *,
    recommendation_id: str,
    earnings_date: str = "2026-05-22",
    as_of_date: str = "2026-05-15",
    picker_provenance: Optional[Dict[str, Any]] = None,
    symbol: str = "AAPL",
) -> None:
    """Insert a forward_post_freeze ledger row ready for the resolver
    to pick up. Mirrors what analyze_single_ticker (post PR-AE C3)
    would write."""
    from types import SimpleNamespace
    from services.recommendation_ledger import build_record_from_analysis

    pp = picker_provenance if picker_provenance is not None else _picker_provenance_block()

    analysis = SimpleNamespace(
        symbol=symbol,
        recommendation="Candidate",
        setup_score=0.72,
        metrics={
            "data_sources": {"options_source": "marketdata_app",
                             "price_rv_source": "yfinance"},
            "experimental_contract_selection": pp,
        },
        rationale=[],
        selector_output={
            "recommendation": "Candidate",
            "best_structure": "call_calendar",
            "earnings_date": earnings_date,
            "primary_thesis": "thesis",
            "primary_risks": [],
            "why_this_structure": [],
            "why_not_others": {},
        },
        structure_scorecards=[
            {"structure": "call_calendar", "eligible": True,
             "composite_structure_score": 0.81},
        ],
        vol_snapshot={
            "symbol": symbol,
            "as_of_date": as_of_date,
            "earnings_date": earnings_date,
            "earnings_source_primary": "alpha_vantage",
            "earnings_source_confidence": 0.82,
            "earnings_source_stale": False,
            "option_source": "marketdata_app",
            "underlying_source": "yfinance",
            "data_quality_score": 0.77,
        },
        sample_provenance="forward_post_freeze",
        candidate_shadow_outcome={},
    )
    record = build_record_from_analysis(
        analysis,
        recommendation_id=recommendation_id,
        quote_payload={
            "quote_source": "yfinance",
            "quote_timestamp": f"{as_of_date}T12:00:00+00:00",
            "quote_quality": "paper_research_mid_not_execution_grade",
            "bid_ask_mid": {"legs": {"call": {"mid": 4.1}}},
            "surface_quality": {"status": "valid_evidence", "warning_flags": []},
        },
    )
    ledger.record(record)


class _StubFeatureStore:
    """In-memory query_chain stub. Maps (symbol, trade_date) -> DataFrame.

    Pass a dict at construction time; query_chain returns the matching
    frame (or an empty DataFrame for the "chain not yet landed" case).
    """

    def __init__(self, chains_by_key: Dict[tuple, pd.DataFrame]) -> None:
        self.chains_by_key = chains_by_key
        self.queries: List[tuple] = []  # call log for assertion in tests

    def query_chain(self, symbol: str, *, trade_date: str, **kwargs: Any) -> pd.DataFrame:
        key = (symbol.upper(), str(trade_date))
        self.queries.append(key)
        return self.chains_by_key.get(key, pd.DataFrame())


# ──────────────────────────────────────────────────────────────────────────
# 1. Eligibility filter integration
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_skips_historical_replay_rows(tmp_path: Path) -> None:
    """Rows tagged historical_replay_in_sample_or_research are never
    eligible. The resolver must not even examine them — the
    list_pending filter excludes them at the ledger level."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    # Plant a historical-replay row directly (the build_record path
    # would tag it forward; we want to verify the filter, not the
    # tagger).
    import json
    ledger._conn.execute(  # noqa: SLF001
        """
        INSERT INTO recommendations (
            recommendation_id, created_at, symbol, earnings_date,
            sample_provenance, selected_structure, picker_provenance_json,
            candidate_shadow_outcome_json, candidate_exit_resolver_attempts,
            schema_version, engine_version, vol_snapshot_json,
            structure_scorecards_json, selector_output_json, as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("rec_historical", "2026-04-01T00:00:00+00:00", "AAPL", "2026-04-01",
         "historical_replay_in_sample_or_research", "call_calendar",
         json.dumps(_picker_provenance_block()), "{}", 0, 2,
         "event_vol_selector_v1", "{}", "[]", "{}", "2026-03-25"),
    )
    ledger._conn.commit()

    store = _StubFeatureStore({})
    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-27").date(),
    )
    assert outcomes == []
    assert summary.scanned == 0
    assert summary.count_balance_holds


# ──────────────────────────────────────────────────────────────────────────
# 2. Picker-provenance fail-closed paths
# ──────────────────────────────────────────────────────────────────────────


def test_extract_candidate_selection_returns_none_for_empty_provenance() -> None:
    assert _extract_candidate_selection({}) is None
    assert _extract_candidate_selection(None) is None
    assert _extract_candidate_selection("not a dict") is None


def test_extract_candidate_selection_returns_none_for_put_side_not_supported() -> None:
    """PR-AC commit 3 placeholder for put_calendar — picker status is
    not 'ok' so the resolver must fail-closed."""
    block = _picker_provenance_block(
        structure="put_calendar",
        status="put_side_not_yet_supported",
    )
    # And the candidate_contracts is None in the real shape
    block["candidate_contracts"] = None
    assert _extract_candidate_selection(block) is None


def test_extract_candidate_selection_returns_none_for_missing_required_keys() -> None:
    """Defensive: a candidate_selection missing any of (side, strike,
    front_expiry, back_expiry) must fail-closed."""
    for missing_key in ("side", "strike", "front_expiry", "back_expiry"):
        block = _picker_provenance_block()
        del block["candidate_contracts"]["candidate_selection"][missing_key]
        assert _extract_candidate_selection(block) is None, (
            f"missing {missing_key} should fail-closed"
        )


def test_extract_candidate_selection_returns_selection_for_valid_block() -> None:
    block = _picker_provenance_block(
        side="call", strike=185.0,
        front_expiry="2026-05-15", back_expiry="2026-05-29",
    )
    sel = _extract_candidate_selection(block)
    assert sel == {
        "side": "call",
        "strike": 185.0,
        "front_expiry": "2026-05-15",
        "back_expiry": "2026-05-29",
    }


def test_resolver_marks_malformed_picker_provenance_terminal(tmp_path: Path) -> None:
    """A row with empty picker_provenance_json gets a
    permanently_failed:no_picker_provenance status and is bucketed
    under skipped_malformed in the summary (not permanently_failed —
    we never tried the simulator)."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_malformed",
        picker_provenance={},  # empty — no candidate to resolve
    )

    store = _StubFeatureStore({})
    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-27").date(),
    )
    assert len(outcomes) == 1
    assert outcomes[0].resolver_status == RESOLVER_STATUS_PERMFAIL_NO_PICKER_PROVENANCE
    assert summary.skipped_malformed == 1
    assert summary.permanently_failed_total == 0  # bucketed separately
    assert summary.count_balance_holds


# ──────────────────────────────────────────────────────────────────────────
# 3. Post-event chain selection
# ──────────────────────────────────────────────────────────────────────────


def test_post_event_bday_candidates_skips_weekends() -> None:
    """BDay arithmetic: Friday + 1 BDay = Monday (not Saturday)."""
    candidates = _post_event_bday_candidates(
        pd.Timestamp("2026-05-22").date(), max_lookahead_trade_days=5,
    )
    # 5/22 is Friday → 5/25 Mon, 5/26 Tue, 5/27 Wed, 5/28 Thu, 5/29 Fri
    assert candidates == [
        "2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29",
    ]


def test_resolver_picks_first_non_empty_chain_in_window(tmp_path: Path) -> None:
    """First two BDays return empty; the third has a chain. Resolver
    uses that one and the exit_trade_date in the outcome reflects it."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_first_nonempty",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )

    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    third_bday_chain = _toy_chain(
        trade_date="2026-05-27", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        # Mon and Tue return empty (key absent → default empty DF)
        ("AAPL", "2026-05-27"): third_bday_chain,
    })

    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-30").date(),
    )
    assert len(outcomes) == 1
    assert outcomes[0].resolver_status == RESOLVER_STATUS_OK
    assert outcomes[0].exit_trade_date == "2026-05-27"
    assert summary.resolved_ok == 1
    assert summary.count_balance_holds


def test_resolver_missing_entry_chain_first_miss_is_retrying(tmp_path: Path) -> None:
    """Codex C4 audit caution: the entry chain SHOULD be present
    (it was used at recommendation time), but the resolver mustn't
    treat the first miss as permanent. The store could be in a
    transient I/O failure or mid-reprocessing window. Behavior on
    first miss: write retrying (counter +1), eligible on next tick.
    """
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_no_entry",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )

    # Exit chain exists but entry chain does not
    exit_chain = _toy_chain(
        trade_date="2026-05-25", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-25"): exit_chain,
        # No entry chain at 2026-05-15
    })

    outcomes, _ = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-30").date(),
    )
    assert outcomes[0].resolver_status == RESOLVER_STATUS_RETRYING
    assert outcomes[0].attempt_number == 1


def test_resolver_missing_entry_chain_escalates_to_permfail_at_max(tmp_path: Path) -> None:
    """At MAX_ATTEMPTS - 1, a missing entry chain on the next tick
    escalates to permanently_failed:no_entry_chain_replay. Parallels
    the exit-chain escalation logic, so operators see a uniform
    escalation timeline regardless of which chain was missing."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_entry_max",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )
    # Push counter to MAX_ATTEMPTS - 1 = 5
    for _ in range(5):
        ledger.increment_resolver_attempts("rec_entry_max")

    exit_chain = _toy_chain(
        trade_date="2026-05-25", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-25"): exit_chain,
        # No entry chain
    })

    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-30").date(),
    )
    assert outcomes[0].resolver_status == RESOLVER_STATUS_PERMFAIL_NO_ENTRY_CHAIN_REPLAY
    assert outcomes[0].attempt_number == 6
    assert summary.permanently_failed_by_reason["no_entry_chain_replay"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. Window-elapsed three-way split
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_awaiting_chain_data_does_not_bump_counter(tmp_path: Path) -> None:
    """Earnings just happened, no chain landed yet, window not
    elapsed → awaiting_chain_data, counter unchanged."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_awaiting",
        earnings_date="2026-05-22",  # Friday
        as_of_date="2026-05-15",
    )

    # Entry chain exists (otherwise we'd hit no_entry_chain_replay first)
    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        # No exit chain on any BDay in the window
    })

    # "now" is only 3 calendar days past earnings: 5/22 Fri + 3 = 5/25 Mon.
    # 5 BDays past 5/22 Fri = 5/29 Fri. So 5/25 is well within the window.
    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-25").date(),
    )
    assert outcomes[0].resolver_status == RESOLVER_STATUS_AWAITING_CHAIN_DATA
    assert outcomes[0].attempt_number == 0  # counter unchanged
    assert summary.awaiting_chain_data == 1

    # Verify on disk: counter is still 0
    counter = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_awaiting",),
    ).fetchone()[0]
    assert int(counter) == 0


def test_resolver_retrying_bumps_counter(tmp_path: Path) -> None:
    """Window has elapsed (more than MAX_LOOKAHEAD_TRADE_DAYS BDays
    since earnings), chain still missing, attempts < MAX → retrying,
    counter +1."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_retrying",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )
    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
    })

    # "now" is well past the 5-BDay window past earnings 5/22.
    # 5/22 Fri + 5 BDays = 5/29 Fri. 6/01 Mon is past that.
    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-06-01").date(),
    )
    assert outcomes[0].resolver_status == RESOLVER_STATUS_RETRYING
    assert outcomes[0].attempt_number == 1  # bumped from 0 → 1
    assert summary.retrying == 1

    counter = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_retrying",),
    ).fetchone()[0]
    assert int(counter) == 1


def test_resolver_escalates_to_permfail_at_max_attempts(tmp_path: Path) -> None:
    """At MAX_ATTEMPTS-1 the next retry escalates to
    permanently_failed:no_post_event_chain. After this write the row
    is no longer eligible (counter == MAX_ATTEMPTS, terminal status).
    """
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_max",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )
    # Manually push counter to MAX_ATTEMPTS-1 (= 5)
    for _ in range(5):
        ledger.increment_resolver_attempts("rec_max")

    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
    })

    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-06-01").date(),
    )
    assert outcomes[0].resolver_status == RESOLVER_STATUS_PERMFAIL_NO_POST_EVENT_CHAIN
    assert outcomes[0].attempt_number == 6
    assert summary.permanently_failed_by_reason["no_post_event_chain"] == 1


def test_resolver_does_not_pick_up_already_at_max_attempts(tmp_path: Path) -> None:
    """Eligibility filter excludes rows at MAX_ATTEMPTS. Even if the
    resolver were invoked, the row never reaches _resolve_one_row —
    list_pending returns an empty list."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_already_maxed",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )
    for _ in range(6):
        ledger.increment_resolver_attempts("rec_already_maxed")

    store = _StubFeatureStore({})
    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-06-01").date(),
    )
    assert outcomes == []
    assert summary.scanned == 0


# ──────────────────────────────────────────────────────────────────────────
# 5. Simulator status passthrough
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_ok_path_writes_resolved_outcome_with_pnl(tmp_path: Path) -> None:
    """Happy path: simulator returns ok with PnL fields, resolver
    surfaces them on the outcome, ledger row's latest revision
    carries the resolved outcome."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_ok",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )

    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    exit_chain = _toy_chain(
        trade_date="2026-05-25", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
        front_iv=0.20, back_iv=0.25,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        ("AAPL", "2026-05-25"): exit_chain,
    })

    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-27").date(),
    )

    assert outcomes[0].resolver_status == RESOLVER_STATUS_OK
    assert outcomes[0].simulator_status == "ok"
    # Entry debit = 2.0 - 1.0 = 1.0, exit value = 1.7 - 0.4 = 1.3,
    # mid_pnl = 0.3, mid_realized_return_pct = 30.0
    assert outcomes[0].mid_realized_return_pct == pytest.approx(30.0)
    assert outcomes[0].attempt_number == 0  # OK does NOT bump counter
    assert summary.resolved_ok == 1

    # Counter unchanged on disk
    counter = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_ok",),
    ).fetchone()[0]
    assert int(counter) == 0

    # Latest revision carries the resolved outcome
    merged = ledger.get_with_latest_resolution("rec_ok")
    assert merged["candidate_shadow_outcome_json"]["status"] == "ok"
    assert merged["candidate_shadow_outcome_json"]["mid_realized_return_pct"] == pytest.approx(30.0)


def test_resolver_simulator_negative_debit_wraps_as_permfail(tmp_path: Path) -> None:
    """Simulator's skipped:negative_debit must wrap into the resolver
    status taxonomy as permanently_failed:simulator_skipped:negative_debit.
    Counter increments because this is a terminal-failure status."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_neg_debit",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )
    # Inverted: front_mid > back_mid → simulator returns negative_debit
    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=2.5, back_mid=1.0,  # inverted
    )
    exit_chain = _toy_chain(
        trade_date="2026-05-25", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        ("AAPL", "2026-05-25"): exit_chain,
    })

    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-27").date(),
    )
    assert outcomes[0].resolver_status == (
        f"{RESOLVER_STATUS_PERMFAIL_SIMULATOR_PREFIX}skipped:negative_debit"
    )
    assert outcomes[0].simulator_status == "skipped:negative_debit"
    assert outcomes[0].attempt_number == 1  # bumped
    assert summary.permanently_failed_by_reason[
        "simulator_skipped:negative_debit"
    ] == 1


# ──────────────────────────────────────────────────────────────────────────
# 6. Idempotency on re-runs
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_ok_row_not_re_picked_on_next_tick(tmp_path: Path) -> None:
    """After a successful resolution, the row's merged-view status is
    'ok' → list_pending excludes it. A subsequent resolver tick sees
    nothing to do for that row."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_ok_once",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )
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
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        ("AAPL", "2026-05-25"): exit_chain,
    })

    # First tick — resolves
    outcomes_1, summary_1 = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-27").date(),
    )
    assert summary_1.resolved_ok == 1

    # Second tick — eligibility filter excludes it
    outcomes_2, summary_2 = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-27").date(),
    )
    assert outcomes_2 == []
    assert summary_2.scanned == 0


def test_resolver_duplicate_retrying_still_bumps_counter(tmp_path: Path) -> None:
    """Codex C1c contract: duplicate retrying outcomes on the
    terminal-failure path still bump the counter. Without this, a
    row stuck in retrying with no upstream chain change would loop
    forever and never escalate to permanently_failed."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_dup_retry",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )
    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    # No exit chain anywhere; both ticks produce retrying.
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
    })

    # Two ticks well past the window — both produce retrying.
    outcomes_1, _ = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-06-01").date(),
    )
    outcomes_2, _ = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-06-02").date(),
    )
    assert outcomes_1[0].resolver_status == RESOLVER_STATUS_RETRYING
    assert outcomes_2[0].resolver_status == RESOLVER_STATUS_RETRYING

    counter = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_dup_retry",),
    ).fetchone()[0]
    assert int(counter) == 2, (
        "Duplicate retrying outcomes must each bump the counter so "
        "the row eventually escalates to permanently_failed."
    )


# ──────────────────────────────────────────────────────────────────────────
# 8. Failure isolation
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_one_bad_row_does_not_abort_run(tmp_path: Path) -> None:
    """A row that raises during resolution (e.g. malformed
    persisted state) must NOT abort the run — the other eligible
    rows still get processed and the bad row gets a
    permanently_failed:simulator_error outcome."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger,
        recommendation_id="rec_good",
        earnings_date="2026-05-22",
        as_of_date="2026-05-15",
    )
    _seed_forward_record(
        ledger,
        recommendation_id="rec_bad",
        earnings_date="2026-05-21",  # earlier earnings → processed first
        as_of_date="2026-05-14",
    )

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

    # Build a store whose query_chain raises ONLY for rec_bad's
    # as_of_date. Other queries pass through normally.
    class _RaisingStore:
        def __init__(self):
            self.real = _StubFeatureStore({
                ("AAPL", "2026-05-15"): entry_chain,
                ("AAPL", "2026-05-25"): exit_chain,
                ("AAPL", "2026-05-26"): exit_chain,
                ("AAPL", "2026-05-27"): exit_chain,
                ("AAPL", "2026-05-28"): exit_chain,
                ("AAPL", "2026-05-29"): exit_chain,
            })

        def query_chain(self, symbol, *, trade_date, **kwargs):
            if trade_date == "2026-05-14":
                raise RuntimeError("synthetic database fault for rec_bad")
            return self.real.query_chain(symbol, trade_date=trade_date, **kwargs)

    store = _RaisingStore()
    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-30").date(),
    )

    # Two rows scanned; one resolved ok, one bumped to retrying. The
    # RuntimeError on rec_bad's entry-chain query is caught inside
    # the explicit try/except around store.query_chain. Under the
    # C4b update, that maps to retrying (counter +1) rather than
    # immediate permfail — the resolver gives the store a few
    # attempts before declaring the row permanently unrecoverable.
    assert summary.scanned == 2
    by_id = {o.recommendation_id: o for o in outcomes}
    assert by_id["rec_good"].resolver_status == RESOLVER_STATUS_OK
    assert by_id["rec_bad"].resolver_status == RESOLVER_STATUS_RETRYING
    assert by_id["rec_bad"].attempt_number == 1


# ──────────────────────────────────────────────────────────────────────────
# 9. Summary count balance — the CLI invariant
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_summary_count_balance_holds_across_mixed_outcomes(
    tmp_path: Path,
) -> None:
    """The CLI in C5 will assert summary.count_balance_holds before
    exit. This test exercises a mixed run (ok + awaiting + retrying +
    skipped_malformed + permfail) and verifies the invariant."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")

    # 1) ok candidate
    _seed_forward_record(
        ledger, recommendation_id="r_ok",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )
    # 2) awaiting (no chain, still in window)
    _seed_forward_record(
        ledger, recommendation_id="r_awaiting",
        earnings_date="2026-05-24",  # closer to "now" so still in window
        as_of_date="2026-05-17",
    )
    # 3) retrying (window elapsed, no chain)
    _seed_forward_record(
        ledger, recommendation_id="r_retrying",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )
    # 4) malformed picker provenance
    _seed_forward_record(
        ledger, recommendation_id="r_malformed",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
        picker_provenance={},
    )
    # 5) simulator returns negative_debit → permfail:simulator_*
    _seed_forward_record(
        ledger, recommendation_id="r_neg",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )

    ok_entry = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    ok_exit = _toy_chain(
        trade_date="2026-05-25", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=0.4, back_mid=1.7,
    )
    neg_entry = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=2.5, back_mid=1.0,  # inverted — neg debit
    )

    # Both r_ok and r_neg share the same earnings_date / as_of_date,
    # so the stub keys collide. Use distinct symbols in seeding...
    # Actually for simplicity we'll seed r_neg with different earnings
    # to get a distinct exit query date. Easier path: re-build the
    # mixed test with distinct seeds per symbol.

    # Refactor: rebuild rows with distinct symbols so the chain stub
    # can map cleanly.
    ledger_path = tmp_path / "mixed.sqlite"
    ledger = RecommendationLedger(ledger_path=ledger_path)
    _seed_forward_record(
        ledger, recommendation_id="r_ok",
        symbol="AAPL", earnings_date="2026-05-22", as_of_date="2026-05-15",
    )
    _seed_forward_record(
        ledger, recommendation_id="r_awaiting",
        symbol="MSFT", earnings_date="2026-05-25", as_of_date="2026-05-18",
    )
    _seed_forward_record(
        ledger, recommendation_id="r_retrying",
        symbol="GOOG", earnings_date="2026-05-22", as_of_date="2026-05-15",
    )
    _seed_forward_record(
        ledger, recommendation_id="r_malformed",
        symbol="TSLA", earnings_date="2026-05-22", as_of_date="2026-05-15",
        picker_provenance={},
    )
    _seed_forward_record(
        ledger, recommendation_id="r_neg",
        symbol="NVDA", earnings_date="2026-05-22", as_of_date="2026-05-15",
    )

    store = _StubFeatureStore({
        # r_ok: full chain pair
        ("AAPL", "2026-05-15"): ok_entry,
        ("AAPL", "2026-05-25"): ok_exit,
        # r_awaiting: entry chain present, no exit chain anywhere in
        # the post-event window — and "now" still inside window
        ("MSFT", "2026-05-18"): _toy_chain(
            trade_date="2026-05-18", front_expiry="2026-05-08",
            back_expiry="2026-05-22", strike=200.0,
            front_mid=1.0, back_mid=2.0,
        ),
        # r_retrying: entry chain present, no exit chain — "now" past
        # the window
        ("GOOG", "2026-05-15"): _toy_chain(
            trade_date="2026-05-15", front_expiry="2026-05-08",
            back_expiry="2026-05-22", strike=200.0,
            front_mid=1.0, back_mid=2.0,
        ),
        # r_neg: entry chain with inverted mids, exit chain present
        ("NVDA", "2026-05-15"): neg_entry,
        ("NVDA", "2026-05-25"): ok_exit,
    })

    # "now" = 5/27 is past the window for r_ok/r_retrying/r_neg/r_malformed
    # (window end = 5/22 + 5 BDay = 5/29 Fri — wait, that puts "now"
    # 5/27 BEFORE the window end of 5/29). Recheck:
    # 5/22 Fri + 5 BDays = Mon 5/25, Tue 5/26, Wed 5/27, Thu 5/28, Fri 5/29.
    # So window_end = 5/29 (using BDay arithmetic) → 5/27 is INSIDE the
    # window. To get r_retrying behavior on 5/27, we'd need earnings
    # on a much earlier date. Use "now" = 6/01 instead so all rows
    # past the 5/22 window have elapsed, but r_awaiting (5/25 earnings,
    # window_end = 5/25 + 5 BDay = 6/01) is right at the boundary.
    # 6/01 Mon < window_end 6/01 Mon? Actually >= is the elapsed test.
    # To get r_awaiting still in window, use "now" = 5/30 — which is
    # 5 cal days past 5/25 (= eligible per min_days=3) but only 3 BDays
    # past, so before the BDay-5 window end of 6/01.

    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-30").date(),
    )

    assert summary.scanned == 5
    assert summary.count_balance_holds, (
        f"count balance failed: scanned={summary.scanned}, "
        f"resolved_ok={summary.resolved_ok}, "
        f"awaiting={summary.awaiting_chain_data}, "
        f"retrying={summary.retrying}, "
        f"skipped_malformed={summary.skipped_malformed}, "
        f"permfail={summary.permanently_failed_total}"
    )

    by_id = {o.recommendation_id: o for o in outcomes}
    assert by_id["r_ok"].resolver_status == RESOLVER_STATUS_OK
    assert by_id["r_awaiting"].resolver_status == RESOLVER_STATUS_AWAITING_CHAIN_DATA
    assert by_id["r_retrying"].resolver_status == RESOLVER_STATUS_RETRYING
    assert by_id["r_malformed"].resolver_status == \
        RESOLVER_STATUS_PERMFAIL_NO_PICKER_PROVENANCE
    assert by_id["r_neg"].resolver_status.startswith(
        RESOLVER_STATUS_PERMFAIL_SIMULATOR_PREFIX
    )


# ──────────────────────────────────────────────────────────────────────────
# Misc — module-level contract guards
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_module_has_no_web_edge() -> None:
    """Static guard: services/candidate_exit_resolver.py must not
    import from web/. The PR-AE design rev 2 makes this a hard
    invariant — the simulator was extracted to services/ in C2
    precisely so this module could call it without forming a
    services -> web edge."""
    import inspect
    from services import candidate_exit_resolver as mod
    source = inspect.getsource(mod)
    for forbidden in ("from web.", "from web ", "import web."):
        assert forbidden not in source, (
            f"services/candidate_exit_resolver.py must not import from "
            f"web/. Found forbidden pattern: {forbidden!r}"
        )


# The existing TestForwardProvenanceAssignmentBoundary regression
# (tests/unit/test_web/test_calendar_dual_picker.py) already enforces
# that this module never references _tag_live_forward_observation or
# the forward_post_freeze literal — its source-grep scans
# services/ + web/ + scripts/ and fails if either appears outside the
# two allowlisted files (services/candidate_shadow_provenance.py and
# web/api/edge_engine.py). A duplicate in-module guard would be
# redundant; the codebase-level test is the canonical defense.


def test_resolver_summary_balance_holds_for_empty_run() -> None:
    """Trivial sanity: a run that finds nothing still satisfies the
    invariant."""
    summary = ResolverRunSummary()
    assert summary.count_balance_holds
    assert summary.scanned == 0
    assert summary.permanently_failed_total == 0


# ──────────────────────────────────────────────────────────────────────────
# Codex C4 audit follow-ups (P1 and P2)
# ──────────────────────────────────────────────────────────────────────────


def test_pr_ae_c4b_end_to_end_promotion_eligibility_after_resolver(
    tmp_path: Path,
) -> None:
    """Codex C4 audit P1: full path from forward-tagged row →
    resolver ok → promotion-eligible aggregator block. The whole
    point of PR-AE is to make this path work; everything before this
    test is necessary but not sufficient.

    Pipeline verified:
      1. Seed a forward_post_freeze row with picker_provenance.
      2. Run the resolver against chains that produce a clean ok.
      3. get_with_latest_resolution(...) returns the merged view
         with status=ok and a finite mid_realized_return_pct.
      4. is_promotion_eligible(...) returns True on that merged
         dict (the EXACT shape it would check at aggregation time).
      5. The aggregator's promotion_eligible_candidate_stats.n is
         exactly 1 when fed this merged dict.

    If ANY of these links breaks, the resolver's "ok" status is
    operationally meaningless. This test guards the whole chain.
    """
    from services.candidate_shadow_provenance import is_promotion_eligible
    from web.api.edge_engine import _aggregate_experimental_candidate_evidence

    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger, recommendation_id="rec_e2e",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )

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
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
        ("AAPL", "2026-05-25"): exit_chain,
    })

    # Run the resolver
    outcomes, summary = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-27").date(),
    )
    assert summary.resolved_ok == 1

    # Step 3: merged view surfaces the resolved candidate outcome
    merged = ledger.get_with_latest_resolution("rec_e2e")
    assert merged is not None
    assert merged["candidate_shadow_outcome_json"]["status"] == "ok"
    assert merged["candidate_shadow_outcome_json"]["mid_realized_return_pct"] == pytest.approx(30.0)
    # sample_provenance stays at v1's tagged value
    assert merged["sample_provenance"] == "forward_post_freeze"

    # Step 4: is_promotion_eligible accepts this row.
    # The eligibility helper checks a dict with TOP-LEVEL keys
    # `sample_provenance` and `candidate_shadow_outcome`. The merged
    # view exposes the parsed outcome under `candidate_shadow_outcome_json`,
    # so we build the eligibility-shape dict from those two keys.
    eligibility_payload = {
        "sample_provenance": merged["sample_provenance"],
        "candidate_shadow_outcome": merged["candidate_shadow_outcome_json"],
    }
    assert is_promotion_eligible(eligibility_payload) is True, (
        "After resolver writes ok, the merged-view row MUST pass "
        "is_promotion_eligible. If this fails, the resolver is "
        "producing rows that look promotion-eligible by name but "
        "fail the strict-checks helper — every link in the PR-AE "
        "chain has to hold for forward evidence to count."
    )

    # Step 5: the aggregator counts this row in promotion_eligible_
    # candidate_stats. The aggregator expects a list of "trade" dicts
    # carrying symbol + event_date + sample_provenance +
    # candidate_shadow_outcome + dual_picker. Construct the full
    # shape so the per-event dedup keys (symbol, event_date, side_key)
    # are populated.
    aggregator_trade = {
        "symbol": "AAPL",
        "event_date": "2026-05-22",
        "sample_provenance": merged["sample_provenance"],
        "candidate_shadow_outcome": merged["candidate_shadow_outcome_json"],
        "dual_picker": {
            "shadow_status": "ok",
            "candidate_selection": {
                "side": "call",
                "strike": 200.0,
                "front_expiry": "2026-05-08",
                "back_expiry": "2026-05-22",
            },
        },
    }
    result = _aggregate_experimental_candidate_evidence([aggregator_trade])
    promotion_block = result["promotion_eligible_candidate_stats"]
    assert promotion_block["n_events"] == 1, (
        f"Aggregator's promotion_eligible_candidate_stats should "
        f"show exactly one promotion-eligible row after the "
        f"resolver lands an ok outcome. Got: {promotion_block}"
    )
    # The mid_realized_return_pct on the resolved outcome (30.0)
    # propagates through to the aggregator's mean.
    assert promotion_block["candidate_mid_realized_return_pct_mean"] == pytest.approx(30.0)


def test_pr_ae_c4b_repeated_awaiting_ticks_do_not_pile_up_revisions(
    tmp_path: Path,
) -> None:
    """Codex C4 audit P2: a row in awaiting_chain_data state should
    not accumulate identical revisions over many resolver ticks. The
    PR-K UNIQUE(recommendation_id, content_hash) constraint dedupes
    byte-identical payloads — three consecutive awaiting ticks must
    produce at most one new awaiting revision row, and the counter
    must stay at 0 throughout (awaiting doesn't burn retry budget).
    """
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    _seed_forward_record(
        ledger, recommendation_id="rec_dup_awaiting",
        earnings_date="2026-05-22", as_of_date="2026-05-15",
    )

    # Entry chain present; no exit chain anywhere. "Now" still inside
    # the post-event window so every tick lands awaiting_chain_data.
    entry_chain = _toy_chain(
        trade_date="2026-05-15", front_expiry="2026-05-08",
        back_expiry="2026-05-22", strike=200.0,
        front_mid=1.0, back_mid=2.0,
    )
    store = _StubFeatureStore({
        ("AAPL", "2026-05-15"): entry_chain,
    })

    # First tick — initial awaiting revision lands
    outcomes_1, _ = resolve_pending_candidate_exits(
        ledger, store, now=pd.Timestamp("2026-05-25").date(),
    )
    assert outcomes_1[0].resolver_status == RESOLVER_STATUS_AWAITING_CHAIN_DATA

    revisions_after_first = len(ledger.get_revisions("rec_dup_awaiting"))

    # Two more ticks — identical content → no new revisions
    for tick_day in ("2026-05-25", "2026-05-26"):
        outcomes, _ = resolve_pending_candidate_exits(
            ledger, store, now=pd.Timestamp(tick_day).date(),
        )
        assert outcomes[0].resolver_status == RESOLVER_STATUS_AWAITING_CHAIN_DATA

    revisions_after_three = len(ledger.get_revisions("rec_dup_awaiting"))
    assert revisions_after_three == revisions_after_first, (
        f"Repeated awaiting_chain_data ticks must dedupe via "
        f"content_hash. Expected {revisions_after_first} revisions, "
        f"got {revisions_after_three}. Without this dedupe, a row "
        f"sitting in awaiting state for weeks would accumulate one "
        f"revision per resolver tick."
    )

    # Counter never bumps for awaiting — verify on disk
    counter = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_dup_awaiting",),
    ).fetchone()[0]
    assert int(counter) == 0, (
        "awaiting_chain_data must NEVER bump the resolver attempt "
        "counter, regardless of how many ticks the row sits in that "
        "state."
    )
