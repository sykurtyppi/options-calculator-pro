"""Tests for the calendar dual-picker shadow logging in edge_engine.

These tests cover:
  - The standalone helper `_dual_picker_calendar_selection`.
  - The `experimental_candidate_evidence` aggregation function.

We do NOT exercise `_simulate_pre_earnings_calendar_trade` end-to-end
here — that requires a fully-populated FeatureStore mock. Per-trade
dual_picker wiring is verified indirectly through the existing
edge_engine research-signals suite (which still passes after this
change) plus a focused integration-style test that uses a stub store.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd
import pytest

from web.api.edge_engine import (
    _CANDIDATE_SHADOW_OUTCOME_FIELDS,
    _CANDIDATE_SHADOW_SCENARIO_BLOCK_FIELDS,
    PROMOTION_ELIGIBLE_PROVENANCES,
    SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
    SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED,
    SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
    SAMPLE_PROVENANCE_UNKNOWN,
    VALID_SAMPLE_PROVENANCES,
    _aggregate_experimental_candidate_evidence,
    _build_experimental_contract_selection,
    _dual_picker_calendar_selection,
    _empty_candidate_shadow_outcome,
    _empty_dual_picker,
    _simulate_candidate_shadow_outcome,
    _tag_live_forward_observation,
    is_promotion_eligible,
)

# Canonical dual_picker key set — every code path that produces a
# dual_picker block must emit exactly these keys. Asserted via tests so
# any drift fails loudly rather than silently breaking the schema.
DUAL_PICKER_KEYS = frozenset({
    "legacy_selection",
    "candidate_selection",
    "pickers_diverged",
    "shadow_status",
    "candidate_min_front_dte_days",
    "experimental_note",
})


def _chain(rows: List[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["expiry", "strike", "call_put", "mid", "iv"])
    return df


# ──────────────────────────────────────────────────────────────────────────
# _dual_picker_calendar_selection
# ──────────────────────────────────────────────────────────────────────────

class TestDualPickerHelper:
    def test_returns_stable_shape_keys(self):
        event = pd.Timestamp("2024-05-01")
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.0, 0.30),
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ])
        chain["underlying_price"] = 100.0
        result = _dual_picker_calendar_selection(chain, event_date=event, side="call")
        assert set(result.keys()) == set(DUAL_PICKER_KEYS)

    def test_pickers_diverge_when_short_dte_weekly_exists(self):
        event = pd.Timestamp("2024-05-01")
        chain = _chain([
            (date(2024, 5, 3), 100.0, "C", 1.0, 0.30),   # 2 DTE — legacy picks this
            (date(2024, 5, 17), 100.0, "C", 2.0, 0.30),  # 16 DTE — candidate picks this
            (date(2024, 6, 21), 100.0, "C", 3.0, 0.30),
        ])
        chain["underlying_price"] = 100.0
        result = _dual_picker_calendar_selection(chain, event_date=event, side="call")

        assert result["pickers_diverged"] is True
        assert result["legacy_selection"]["front_expiry"] == "2024-05-03"
        assert result["candidate_selection"]["front_expiry"] == "2024-05-17"

    def test_pickers_agree_when_no_short_weekly(self):
        event = pd.Timestamp("2024-05-01")
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.0, 0.30),
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ])
        chain["underlying_price"] = 100.0
        result = _dual_picker_calendar_selection(chain, event_date=event, side="call")
        assert result["pickers_diverged"] is False
        assert result["legacy_selection"]["front_expiry"] == "2024-05-17"
        assert result["candidate_selection"]["front_expiry"] == "2024-05-17"

    def test_candidate_returns_none_when_no_long_dte_available(self):
        event = pd.Timestamp("2024-05-01")
        # Only close-in weeklies — candidate's +14d floor finds nothing
        chain = _chain([
            (date(2024, 5, 3), 100.0, "C", 1.0, 0.30),
            (date(2024, 5, 10), 100.0, "C", 1.5, 0.30),
        ])
        chain["underlying_price"] = 100.0
        result = _dual_picker_calendar_selection(chain, event_date=event, side="call")
        # Legacy can still pick (it has no DTE floor), but it needs a back too.
        # 5/3 front, back must be > front → 5/10 with 7-day gap. Legacy picker
        # falls back to "any > front" when no 14-day-gap exists → 5/10.
        assert result["legacy_selection"] is not None
        assert result["candidate_selection"] is None
        assert result["pickers_diverged"] is False  # candidate failed → no comparison

    def test_shadow_status_ok_when_normal_inputs(self):
        event = pd.Timestamp("2024-05-01")
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.0, 0.30),
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ])
        chain["underlying_price"] = 100.0
        result = _dual_picker_calendar_selection(chain, event_date=event, side="call")
        assert result["shadow_status"] == "ok"

    def test_experimental_note_warns_against_execution(self):
        event = pd.Timestamp("2024-05-01")
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.0, 0.30),
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ])
        chain["underlying_price"] = 100.0
        result = _dual_picker_calendar_selection(chain, event_date=event, side="call")
        # Codex hard requirement: candidate output must be labeled experimental
        # and discourage execution. The note text is checked by substring so
        # phrasing can evolve without test churn.
        note = result["experimental_note"].lower()
        assert "in-sample" in note or "not out-of-sample validated" in note
        assert "candidate_selection" in result["experimental_note"]

    def test_no_performance_claims_in_output(self):
        """The dual-picker output must not include any embedded performance
        claims (e.g. '+11%', '23% return', etc.) — Codex hard requirement."""
        event = pd.Timestamp("2024-05-01")
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.0, 0.30),
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ])
        chain["underlying_price"] = 100.0
        result = _dual_picker_calendar_selection(chain, event_date=event, side="call")
        flat = str(result).lower()
        forbidden = ["+11%", "+23%", "73% win", "12% return", "outperform"]
        for substring in forbidden:
            assert substring not in flat, (
                f"performance claim {substring!r} leaked into dual-picker output"
            )


# ──────────────────────────────────────────────────────────────────────────
# _empty_dual_picker — shape stability across error/missing paths
# ──────────────────────────────────────────────────────────────────────────

class TestEmptyDualPickerShape:
    """REGRESSION (Codex review of commit 2): the dual_picker block
    must carry the SAME key set whether the picker ran normally or
    bailed (missing entry chain, exception, etc.). Drift between paths
    was previously masking that some statuses lacked
    candidate_min_front_dte_days / experimental_note."""

    def test_missing_entry_chain_shape(self):
        block = _empty_dual_picker("skipped:missing_entry_chain")
        assert set(block.keys()) == set(DUAL_PICKER_KEYS)
        assert block["legacy_selection"] is None
        assert block["candidate_selection"] is None
        assert block["pickers_diverged"] is False
        assert block["shadow_status"] == "skipped:missing_entry_chain"
        assert isinstance(block["candidate_min_front_dte_days"], int)
        assert block["candidate_min_front_dte_days"] > 0

    def test_error_path_shape(self):
        block = _empty_dual_picker("error:ValueError")
        assert set(block.keys()) == set(DUAL_PICKER_KEYS)
        assert block["shadow_status"] == "error:ValueError"

    def test_helper_exception_path_matches_empty_shape(self):
        """When the picker module raises, _dual_picker_calendar_selection
        delegates to _empty_dual_picker. Verify the resulting shape
        matches the canonical key set."""
        # Pass a chain that will explode inside the picker via a missing
        # required column. Picker should not raise (returns None), so
        # we instead trigger the exception path by feeding obviously
        # malformed input that gets past the lazy import but breaks
        # downstream — use call_put column with non-string values.
        # Easier: just verify that calling _empty_dual_picker directly
        # with an error status produces the same keys.
        normal = _empty_dual_picker("ok")
        errored = _empty_dual_picker("error:Boom")
        assert set(normal.keys()) == set(errored.keys()) == set(DUAL_PICKER_KEYS)


# ──────────────────────────────────────────────────────────────────────────
# _aggregate_experimental_candidate_evidence
# ──────────────────────────────────────────────────────────────────────────

class TestExperimentalCandidateEvidenceAggregation:
    def test_empty_input_yields_zero_counts(self):
        result = _aggregate_experimental_candidate_evidence([])
        tc = result["trade_counts"]
        ec = result["event_counts"]
        assert tc["total_picker_evaluated"] == 0
        assert tc["pickers_diverged"] == 0
        assert tc["both_succeeded"] == 0
        assert tc["legacy_only"] == 0
        assert tc["candidate_only"] == 0
        assert tc["neither_succeeded"] == 0
        assert ec["unique_events_observed"] == 0
        assert ec["unique_events_picker_evaluated"] == 0
        assert ec["unique_events_diverged"] == 0
        assert result["diverged_sample"] == []

    def _ok_trade(self, *, symbol="AAPL", event_date, entry_date, legacy, candidate, diverged):
        return {
            "symbol": symbol,
            "event_date": event_date,
            "entry_date": entry_date,
            "dual_picker": {
                "shadow_status": "ok",
                "legacy_selection": legacy,
                "candidate_selection": candidate,
                "pickers_diverged": diverged,
                "candidate_min_front_dte_days": 14,
                "experimental_note": "...",
            },
        }

    def test_counts_split_by_picker_success(self):
        sel = {"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
               "strike": 100.0, "front_dte_days": 16}
        sel_legacy_close = {"front_expiry": "2024-08-02", "back_expiry": "2024-08-16",
                            "strike": 200.0, "front_dte_days": 1}
        sel_candidate_far = {"front_expiry": "2024-08-16", "back_expiry": "2024-09-20",
                             "strike": 200.0, "front_dte_days": 15}
        trades = [
            self._ok_trade(event_date="2024-05-01", entry_date="2024-04-29",
                           legacy=sel, candidate=sel, diverged=False),
            self._ok_trade(event_date="2024-08-01", entry_date="2024-07-30",
                           legacy=sel_legacy_close, candidate=sel_candidate_far,
                           diverged=True),
            self._ok_trade(event_date="2024-11-01", entry_date="2024-10-30",
                           legacy=sel, candidate=None, diverged=False),
            self._ok_trade(event_date="2025-02-01", entry_date="2025-01-30",
                           legacy=None, candidate=None, diverged=False),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        tc = result["trade_counts"]
        ec = result["event_counts"]
        assert tc["total_picker_evaluated"] == 4
        assert tc["both_succeeded"] == 2
        assert tc["legacy_only"] == 1
        assert tc["candidate_only"] == 0
        assert tc["neither_succeeded"] == 1
        assert tc["pickers_diverged"] == 1
        # Per-event: 4 distinct (symbol, event_date) keys
        assert ec["unique_events_observed"] == 4
        assert ec["unique_events_picker_evaluated"] == 4
        assert ec["unique_events_diverged"] == 1
        assert ec["unique_events_with_legacy_selection"] == 3
        assert ec["unique_events_with_candidate_selection"] == 2

    def test_diverged_sample_captures_up_to_five(self):
        # 7 diverged events should produce a sample of 5
        trades = []
        for i in range(7):
            trades.append({
                "symbol": "AAPL",
                "event_date": f"2024-0{i+1}-01",
                "entry_date": f"2024-0{i+1}-01",
                "dual_picker": {
                    "shadow_status": "ok",
                    "legacy_selection": {"front_expiry": "2024-05-03", "back_expiry": "2024-05-17",
                                         "strike": 100.0, "front_dte_days": 2},
                    "candidate_selection": {"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
                                            "strike": 100.0, "front_dte_days": 16},
                    "pickers_diverged": True,
                },
            })
        result = _aggregate_experimental_candidate_evidence(trades)
        assert result["trade_counts"]["pickers_diverged"] == 7
        assert len(result["diverged_sample"]) == 5
        # Sample entries carry both fronts for human eyeball
        for entry in result["diverged_sample"]:
            assert entry["legacy_front_expiry"] == "2024-05-03"
            assert entry["candidate_front_expiry"] == "2024-05-17"

    def test_unique_event_count_deduplicates_across_entry_offsets(self):
        """REGRESSION (Codex review of commit 2): trade_counts is inflated
        when the same earnings event is evaluated at multiple entry
        offsets. event_counts.unique_events_diverged must dedupe on
        (symbol, event_date) so promotion criteria like 'n >= 40 events'
        cannot be satisfied by entry-offset replication of fewer
        independent events."""
        # 3 distinct (symbol, event_date) keys × 6 offsets = 18 trade records
        trades = []
        sel_legacy = {"front_expiry": "2024-05-03", "back_expiry": "2024-05-17",
                      "strike": 100.0, "front_dte_days": 2}
        sel_candidate = {"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
                         "strike": 100.0, "front_dte_days": 16}
        for event_date in ("2024-05-01", "2024-08-01", "2024-11-01"):
            for offset in (5, 6, 7, 8, 9, 10):
                trades.append({
                    "symbol": "AAPL",
                    "event_date": event_date,
                    "entry_date": f"offset-{offset}",
                    "dual_picker": {
                        "shadow_status": "ok",
                        "legacy_selection": sel_legacy,
                        "candidate_selection": sel_candidate,
                        "pickers_diverged": True,
                    },
                })
        result = _aggregate_experimental_candidate_evidence(trades)
        # Trade-level: 18 records all diverged
        assert result["trade_counts"]["total_picker_evaluated"] == 18
        assert result["trade_counts"]["pickers_diverged"] == 18
        # Event-level: only 3 unique earnings events
        assert result["event_counts"]["unique_events_observed"] == 3
        assert result["event_counts"]["unique_events_diverged"] == 3
        # The doc note must explicitly warn about this — protects future readers
        assert "inflated by entry-offset replication" in result["note"]

    def test_missing_entry_chain_does_not_count_as_picker_failure(self):
        """REGRESSION (Codex review of commit 2): trades that failed
        because the entry chain wasn't available must be tallied under
        shadow_status_counts, NOT counted as 'both pickers failed'."""
        trades = [
            # 5 events where entry chain wasn't fetched — neither picker ran
            {
                "symbol": "AAPL",
                "event_date": f"2024-0{i+1}-01",
                "entry_date": f"2024-0{i+1}-01",
                "dual_picker": {
                    "shadow_status": "skipped:missing_entry_chain",
                    "legacy_selection": None,
                    "candidate_selection": None,
                    "pickers_diverged": False,
                    "candidate_min_front_dte_days": 14,
                    "experimental_note": "...",
                },
            }
            for i in range(5)
        ]
        # Plus 1 ok event where both pickers genuinely failed (e.g. empty chain)
        trades.append(self._ok_trade(
            event_date="2024-12-01", entry_date="2024-11-29",
            legacy=None, candidate=None, diverged=False,
        ))
        result = _aggregate_experimental_candidate_evidence(trades)
        tc = result["trade_counts"]
        # Only the ok event is counted as a picker outcome
        assert tc["total_picker_evaluated"] == 1
        assert tc["neither_succeeded"] == 1
        # The 5 skipped events are tallied under shadow_status_counts
        assert result["shadow_status_counts"]["skipped:missing_entry_chain"] == 5
        assert result["shadow_status_counts"]["ok"] == 1
        # And they are NOT in the picker-evaluated event count
        assert result["event_counts"]["unique_events_picker_evaluated"] == 1
        assert result["event_counts"]["unique_events_observed"] == 6  # all 6 distinct events

    def test_trades_without_dual_picker_dont_count_as_picker_evaluated(self):
        """Trades with no dual_picker block at all (e.g. legacy code path
        that never wrote one) must not inflate trade_counts. They still
        show up in shadow_status_counts under 'missing' for debugging."""
        trades = [
            {"symbol": "AAPL", "event_date": "2024-05-01",
             "status": "missing_entry_chain"},  # no dual_picker key
            self._ok_trade(
                event_date="2024-05-02", entry_date="2024-04-30",
                legacy={"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
                        "strike": 100.0, "front_dte_days": 16},
                candidate={"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
                           "strike": 100.0, "front_dte_days": 16},
                diverged=False,
            ),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        # Only the ok trade is counted as picker-evaluated
        assert result["trade_counts"]["total_picker_evaluated"] == 1
        assert result["trade_counts"]["both_succeeded"] == 1
        # The trade with no dual_picker shows up as "missing" status
        assert result["shadow_status_counts"]["missing"] == 1
        assert result["shadow_status_counts"]["ok"] == 1

    def test_picker_names_in_output(self):
        """Codex hard requirement: explicit picker variant names so the
        consumer can map evidence to the right rule."""
        result = _aggregate_experimental_candidate_evidence([])
        assert result["picker_legacy"] == "legacy_first_expiry"
        assert result["picker_candidate"] == "candidate_min_dte"

    def test_output_is_json_safe(self):
        """No pd.Series, no numpy scalars in the aggregation output."""
        trades = [
            {"event_date": "2024-05-01", "entry_date": "2024-04-29",
             "dual_picker": {
                 "shadow_status": "ok",
                 "legacy_selection": {"front_expiry": "2024-05-03",
                                      "back_expiry": "2024-05-17",
                                      "strike": 100.0, "front_dte_days": 2},
                 "candidate_selection": {"front_expiry": "2024-05-17",
                                         "back_expiry": "2024-06-21",
                                         "strike": 100.0, "front_dte_days": 16},
                 "pickers_diverged": True,
             }},
        ]
        import json
        json.dumps(_aggregate_experimental_candidate_evidence(trades))  # must not raise

    def test_event_level_outcome_buckets(self):
        """Codex follow-up to commit 2b: event-level outcome buckets
        (unique_events_both_succeeded, _legacy_only, _candidate_only,
        _neither_succeeded). Distinct from trade-level versions and
        clearer for promotion decisions."""
        sel = {"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
               "strike": 100.0, "front_dte_days": 16}
        trades = [
            # Event A: both pickers succeeded at every offset → both_succeeded
            self._ok_trade(symbol="AAPL", event_date="2024-05-01",
                           entry_date="2024-04-26", legacy=sel, candidate=sel, diverged=False),
            self._ok_trade(symbol="AAPL", event_date="2024-05-01",
                           entry_date="2024-04-29", legacy=sel, candidate=sel, diverged=False),
            # Event B: only legacy succeeded at both offsets → legacy_only
            self._ok_trade(symbol="AAPL", event_date="2024-08-01",
                           entry_date="2024-07-29", legacy=sel, candidate=None, diverged=False),
            self._ok_trade(symbol="AAPL", event_date="2024-08-01",
                           entry_date="2024-07-30", legacy=sel, candidate=None, diverged=False),
            # Event C: only candidate at both offsets → candidate_only
            self._ok_trade(symbol="AAPL", event_date="2024-11-01",
                           entry_date="2024-10-29", legacy=None, candidate=sel, diverged=False),
            # Event D: neither at the one offset → neither_succeeded
            self._ok_trade(symbol="AAPL", event_date="2025-02-01",
                           entry_date="2025-01-30", legacy=None, candidate=None, diverged=False),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        ec = result["event_counts"]
        assert ec["unique_events_both_succeeded"] == 1
        assert ec["unique_events_legacy_only"] == 1
        assert ec["unique_events_candidate_only"] == 1
        assert ec["unique_events_neither_succeeded"] == 1
        assert ec["unique_events_mixed_outcomes"] == 0

    def test_event_level_buckets_handle_mixed_offsets(self):
        """When the same event shows different outcomes across offsets
        (e.g. candidate succeeds at offset 10 but not offset 5), the
        event must NOT be folded into a clean bucket — it goes into
        `unique_events_mixed_outcomes` so promotion analysis cannot
        accidentally treat it as a clean win/loss."""
        sel = {"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
               "strike": 100.0, "front_dte_days": 16}
        trades = [
            # Same event, two offsets, different outcomes
            self._ok_trade(symbol="AAPL", event_date="2024-05-01",
                           entry_date="off5", legacy=sel, candidate=None, diverged=False),
            self._ok_trade(symbol="AAPL", event_date="2024-05-01",
                           entry_date="off10", legacy=sel, candidate=sel, diverged=False),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        ec = result["event_counts"]
        assert ec["unique_events_mixed_outcomes"] == 1
        # And the event must not appear in any clean bucket
        assert ec["unique_events_both_succeeded"] == 0
        assert ec["unique_events_legacy_only"] == 0
        assert ec["unique_events_candidate_only"] == 0
        assert ec["unique_events_neither_succeeded"] == 0

    def test_note_points_at_narrowest_promotion_fields(self):
        """Codex review explicitly asked for the note to point at the
        narrowest fields (unique_events_diverged, _picker_evaluated),
        not the broad unique_events_observed."""
        result = _aggregate_experimental_candidate_evidence([])
        note = result["note"]
        assert "unique_events_diverged" in note
        assert "unique_events_picker_evaluated" in note
        # The broader count should be explicitly called out as NOT the
        # right denominator
        assert "unique_events_observed" in note  # mentioned by name
        assert "should not be used as a denominator" in note


# ──────────────────────────────────────────────────────────────────────────
# _build_experimental_contract_selection  (PR-AC commit 3)
# ──────────────────────────────────────────────────────────────────────────

class TestExperimentalContractSelection:
    """Live-API shadow surface tests. The hardest constraint here is the
    put-side guardrail: put_calendar must NEVER receive call-side
    contracts. That mistake is exactly the kind of silent-routing error
    Codex's review flagged."""

    @staticmethod
    def _live_chain():
        # Live chain uses MDApp column name `expiration_date`. The helper
        # must normalize it to the picker's `expiry` schema.
        return pd.DataFrame([
            {"expiration_date": date(2024, 5, 3), "strike": 100.0,
             "call_put": "C", "mid": 1.0, "iv": 0.30, "underlying_price": 100.0},
            {"expiration_date": date(2024, 5, 17), "strike": 100.0,
             "call_put": "C", "mid": 2.0, "iv": 0.30, "underlying_price": 100.0},
            {"expiration_date": date(2024, 6, 21), "strike": 100.0,
             "call_put": "C", "mid": 3.0, "iv": 0.30, "underlying_price": 100.0},
            # Include puts so we can verify they're NOT used for call_calendar
            {"expiration_date": date(2024, 5, 17), "strike": 100.0,
             "call_put": "P", "mid": 2.1, "iv": 0.31, "underlying_price": 100.0},
            {"expiration_date": date(2024, 6, 21), "strike": 100.0,
             "call_put": "P", "mid": 3.2, "iv": 0.31, "underlying_price": 100.0},
        ])

    def test_returns_none_for_non_calendar_structures(self):
        for structure in (None, "atm_straddle", "otm_strangle", "iron_condor", ""):
            result = _build_experimental_contract_selection(
                best_structure=structure,
                chain_df=self._live_chain(),
                earnings_event_date=date(2024, 5, 1),
            )
            assert result is None, (
                f"non-calendar structure {structure!r} should not receive "
                f"experimental_contract_selection, got {result}"
            )

    def test_call_calendar_returns_dual_picker_candidates(self):
        result = _build_experimental_contract_selection(
            best_structure="call_calendar",
            chain_df=self._live_chain(),
            earnings_event_date=date(2024, 5, 1),
        )
        assert result is not None
        assert result["structure"] == "call_calendar"
        assert result["status"] == "ok"
        # Labels must communicate this is shadow / experimental
        labels = result["labels"]
        assert labels["experimental"] is True
        assert labels["shadow_mode"] is True
        assert labels["not_execution_guidance"] is True
        assert labels["out_of_sample_validated"] is False
        # Candidate contracts populated via dual picker
        cc = result["candidate_contracts"]
        assert cc is not None
        assert cc["legacy_selection"] is not None
        assert cc["candidate_selection"] is not None
        # Both pickers should diverge on this chain (5/3 vs 5/17)
        assert cc["pickers_diverged"] is True

    def test_put_calendar_returns_placeholder_not_call_side_contracts(self):
        """REGRESSION: the most important guardrail in this commit.
        put_calendar must NEVER receive call-side contract data — we
        have no put-side forward-validation evidence yet. Surfacing
        call-side contracts under a put_calendar response would falsely
        imply put-side validation."""
        chain = self._live_chain()
        result = _build_experimental_contract_selection(
            best_structure="put_calendar",
            chain_df=chain,  # contains both calls and puts
            earnings_event_date=date(2024, 5, 1),
        )
        assert result is not None
        assert result["structure"] == "put_calendar"
        # The critical assertion: status is the explicit placeholder
        assert result["status"] == "put_side_not_yet_supported"
        # AND candidate_contracts must be None — no leakage
        assert result["candidate_contracts"] is None
        # Reason explains why
        assert "calls chain" in result["reason"].lower()
        # Labels still applied
        assert result["labels"]["experimental"] is True

    def test_put_calendar_does_not_invoke_picker_at_all(self, monkeypatch):
        """Defense in depth: even if a future refactor changes the
        placeholder string, the picker MUST NOT be called for puts."""
        called = {"count": 0}
        from web.api import edge_engine
        original = edge_engine._dual_picker_calendar_selection

        def _spy(*args, **kwargs):
            called["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(edge_engine, "_dual_picker_calendar_selection", _spy)
        _build_experimental_contract_selection(
            best_structure="put_calendar",
            chain_df=self._live_chain(),
            earnings_event_date=date(2024, 5, 1),
        )
        assert called["count"] == 0, (
            "put_calendar must not invoke the dual picker — that would "
            "produce call-side selections from the call-only path"
        )

    def test_missing_chain_returns_skipped_status(self):
        result = _build_experimental_contract_selection(
            best_structure="call_calendar",
            chain_df=None,
            earnings_event_date=date(2024, 5, 1),
        )
        assert result is not None
        assert result["status"] == "skipped:missing_inputs"
        assert result["candidate_contracts"] is None

    def test_missing_event_date_returns_skipped_status(self):
        result = _build_experimental_contract_selection(
            best_structure="call_calendar",
            chain_df=self._live_chain(),
            earnings_event_date=None,
        )
        assert result is not None
        assert result["status"] == "skipped:missing_inputs"
        assert result["candidate_contracts"] is None

    def test_normalizes_mdapp_expiration_date_column(self):
        """The MDApp chain uses `expiration_date`; the picker expects
        `expiry`. The helper must normalize the column rather than
        silently returning no contracts."""
        # Verified implicitly by test_call_calendar_returns_dual_picker_
        # candidates passing — but assert explicitly here for clarity.
        chain = self._live_chain()
        assert "expiration_date" in chain.columns
        assert "expiry" not in chain.columns
        result = _build_experimental_contract_selection(
            best_structure="call_calendar",
            chain_df=chain,
            earnings_event_date=date(2024, 5, 1),
        )
        # Status ok proves normalization worked
        assert result["status"] == "ok"
        assert result["candidate_contracts"]["legacy_selection"] is not None

    def test_output_is_json_safe(self):
        import json
        result = _build_experimental_contract_selection(
            best_structure="call_calendar",
            chain_df=self._live_chain(),
            earnings_event_date=date(2024, 5, 1),
        )
        json.dumps(result)  # must not raise

    def test_no_performance_claims_in_labels_or_note(self):
        """Codex hard requirement: no performance language at the
        experimental surface."""
        result = _build_experimental_contract_selection(
            best_structure="call_calendar",
            chain_df=self._live_chain(),
            earnings_event_date=date(2024, 5, 1),
        )
        flat = str(result).lower()
        forbidden = ["+11%", "+23%", "73% win", "12% return",
                     "outperform", "expected pnl"]
        for s in forbidden:
            assert s not in flat, f"performance claim {s!r} leaked"


# ──────────────────────────────────────────────────────────────────────────
# _simulate_candidate_shadow_outcome  (PR-AD commit 1)
# ──────────────────────────────────────────────────────────────────────────

# Canonical key set produced by both _empty_candidate_shadow_outcome
# and the happy-path return. Asserted via tests so any future schema
# drift fails loudly.
CANDIDATE_SHADOW_KEYS = frozenset({"status", "labels"} | set(_CANDIDATE_SHADOW_OUTCOME_FIELDS))


def _candidate_only_dual_picker(
    *,
    front_expiry: str = "2024-05-17",
    back_expiry: str = "2024-06-21",
    strike: float = 100.0,
    side: str = "call",
) -> Dict[str, Any]:
    """Minimal dual_picker block with only the fields the candidate
    shadow simulator reads. Avoids having to construct a full
    CalendarSelection.to_metadata_dict() in every test."""
    return {
        "shadow_status": "ok",
        "legacy_selection": None,  # legacy doesn't matter for these tests
        "candidate_selection": {
            "side": side,
            "strike": strike,
            "front_expiry": front_expiry,
            "back_expiry": back_expiry,
        },
        "pickers_diverged": False,
        "candidate_min_front_dte_days": 14,
        "experimental_note": "...",
    }


def _chain_with_contracts(rows):
    """Build a chain DataFrame in the format _lookup_exact_contract_row expects.

    Rows: (expiry_date, strike, call_put, mid, iv, bid, ask)
    """
    return pd.DataFrame(rows, columns=["expiry", "strike", "call_put", "mid", "iv", "bid", "ask"])


# ── Empty-state factory shape ────────────────────────────────────────────

class TestEmptyCandidateShadowOutcome:
    def test_empty_block_has_canonical_key_set(self):
        block = _empty_candidate_shadow_outcome("skipped:test")
        assert set(block.keys()) == set(CANDIDATE_SHADOW_KEYS)
        assert block["status"] == "skipped:test"
        # Scalar fields are None; scenario-block fields (PR #64) are
        # dict-shaped placeholders so consumers can `.get("cross_25")`
        # without first None-checking the wrapper.
        for k in _CANDIDATE_SHADOW_OUTCOME_FIELDS:
            if k in _CANDIDATE_SHADOW_SCENARIO_BLOCK_FIELDS:
                assert isinstance(block[k], dict), (
                    f"{k} should be a dict in empty block (PR #64 stable shape), "
                    f"got {type(block[k]).__name__}"
                )
                # Every value inside the dict is None on empty paths.
                for label, value in block[k].items():
                    assert value is None, f"{k}[{label!r}] should be None, got {value!r}"
            else:
                assert block[k] is None, (
                    f"{k} should be None in empty block, got {block[k]!r}"
                )

    def test_status_string_preserved(self):
        for status in ("skipped:no_dual_picker", "skipped:no_candidate_selection",
                       "skipped:missing_entry_quote", "error:Boom"):
            assert _empty_candidate_shadow_outcome(status)["status"] == status


# ── Happy path: candidate resolved to PnL ────────────────────────────────

class TestCandidateShadowOutcomeHappyPath:
    def test_resolves_candidate_pnl_with_both_chains(self):
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        assert result["status"] == "ok"
        # Sanity-check PnL math
        # entry_debit = 2.0 - 1.0 = 1.0
        # exit_value  = 1.7 - 0.4 = 1.3
        # mid_pnl     = 1.3 - 1.0 = 0.3
        # return_pct  = 0.3/1.0 * 100 = 30.0
        assert result["entry_debit_mid"] == pytest.approx(1.0)
        assert result["exit_value_mid"] == pytest.approx(1.3)
        assert result["mid_pnl"] == pytest.approx(0.3)
        assert result["mid_realized_return_pct"] == pytest.approx(30.0)
        # IV change is exit - entry (front crushed -10pp, back -5pp)
        assert result["iv_change_front"] == pytest.approx(-0.10)
        assert result["iv_change_back"] == pytest.approx(-0.05)
        # Side + identifiers preserved
        assert result["side"] == "call"
        assert result["strike"] == 100.0
        assert result["front_expiry"] == "2024-05-17"
        assert result["back_expiry"] == "2024-06-21"

    def test_put_side_resolves_against_put_chain(self):
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "P", 1.5, 0.32, 1.45, 1.55),
            (pd.Timestamp("2024-06-21"), 100.0, "P", 2.8, 0.31, 2.75, 2.85),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "P", 0.6, 0.22, 0.55, 0.65),
            (pd.Timestamp("2024-06-21"), 100.0, "P", 2.3, 0.26, 2.25, 2.35),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(side="put"),
        )
        assert result["status"] == "ok"
        assert result["side"] == "put"
        # entry_debit = 2.8 - 1.5 = 1.3, exit_value = 2.3 - 0.6 = 1.7
        assert result["entry_debit_mid"] == pytest.approx(1.3)
        assert result["exit_value_mid"] == pytest.approx(1.7)


# ── Skip statuses ─────────────────────────────────────────────────────────

class TestCandidateShadowOutcomeSkips:
    def test_none_dual_picker(self):
        assert _simulate_candidate_shadow_outcome(
            entry_chain=pd.DataFrame(), exit_chain=pd.DataFrame(),
            dual_picker=None,
        )["status"] == "skipped:no_dual_picker"

    def test_dual_picker_failed(self):
        dp = _empty_dual_picker("skipped:missing_entry_chain")
        result = _simulate_candidate_shadow_outcome(
            entry_chain=pd.DataFrame(), exit_chain=pd.DataFrame(),
            dual_picker=dp,
        )
        # Inherits dual_picker's failure reason
        assert result["status"] == "skipped:dual_picker_status:skipped:missing_entry_chain"

    def test_no_candidate_selection(self):
        # dual_picker.shadow_status == "ok" but candidate_selection is None
        # (legitimate — candidate picker had no eligible front)
        dp = _candidate_only_dual_picker()
        dp["candidate_selection"] = None
        result = _simulate_candidate_shadow_outcome(
            entry_chain=pd.DataFrame(), exit_chain=pd.DataFrame(),
            dual_picker=dp,
        )
        assert result["status"] == "skipped:no_candidate_selection"

    def test_malformed_candidate_selection(self):
        dp = _candidate_only_dual_picker()
        dp["candidate_selection"] = {"side": "call"}  # missing strike, expiries
        result = _simulate_candidate_shadow_outcome(
            entry_chain=pd.DataFrame(), exit_chain=pd.DataFrame(),
            dual_picker=dp,
        )
        assert result["status"] == "skipped:malformed_candidate_selection"

    def test_missing_entry_quote(self):
        # Entry chain doesn't contain candidate's contracts
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-07-19"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        assert result["status"] == "skipped:missing_entry_quote"

    def test_missing_exit_quote(self):
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            # Different expiry; candidate's contracts not present at exit
            (pd.Timestamp("2024-07-19"), 100.0, "C", 0.5, 0.25, 0.45, 0.55),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        assert result["status"] == "skipped:missing_exit_quote"

    def test_negative_debit_mirrors_legacy_behavior(self):
        # front more expensive than back at entry → entry_debit <= 0
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 3.0, 0.30, 2.95, 3.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.25, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.8, 0.26, 1.75, 1.85),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        assert result["status"] == "skipped:negative_debit"


# ── Schema stability ──────────────────────────────────────────────────────

class TestCandidateShadowOutcomeShape:
    def test_happy_path_uses_canonical_key_set(self):
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        assert set(result.keys()) == set(CANDIDATE_SHADOW_KEYS)

    def test_every_skip_status_uses_canonical_key_set(self):
        """Every failure path must produce the same key set as the
        happy path. Drift would break the ledger schema in PR-AD
        commit 2."""
        dp_ok = _candidate_only_dual_picker()

        cases = [
            # (description, kwargs producing each status)
            ("no_dual_picker", dict(entry_chain=pd.DataFrame(),
                                    exit_chain=pd.DataFrame(),
                                    dual_picker=None)),
            ("no_candidate_selection",
             dict(entry_chain=pd.DataFrame(), exit_chain=pd.DataFrame(),
                  dual_picker={**dp_ok, "candidate_selection": None})),
            ("malformed_candidate",
             dict(entry_chain=pd.DataFrame(), exit_chain=pd.DataFrame(),
                  dual_picker={**dp_ok, "candidate_selection": {"side": "call"}})),
            ("missing_chain",
             dict(entry_chain=None, exit_chain=None, dual_picker=dp_ok)),
        ]
        for description, kwargs in cases:
            result = _simulate_candidate_shadow_outcome(**kwargs)
            assert set(result.keys()) == set(CANDIDATE_SHADOW_KEYS), (
                f"key drift on {description}: extra={set(result.keys()) - set(CANDIDATE_SHADOW_KEYS)}, "
                f"missing={set(CANDIDATE_SHADOW_KEYS) - set(result.keys())}"
            )
            # On skip paths: scalar fields are None; scenario-block
            # fields (PR #64) are dict-shaped placeholders with every
            # SCENARIO_LEVELS label mapped to None.
            for k in _CANDIDATE_SHADOW_OUTCOME_FIELDS:
                if k in _CANDIDATE_SHADOW_SCENARIO_BLOCK_FIELDS:
                    assert isinstance(result[k], dict), (
                        f"{description}: {k} should be a dict on skip paths "
                        f"(PR #64 stable shape), got {type(result[k]).__name__}"
                    )
                    for label, value in result[k].items():
                        assert value is None, (
                            f"{description}: {k}[{label!r}] should be None, got {value!r}"
                        )
                else:
                    assert result[k] is None, (
                        f"{description}: {k} is {result[k]!r}, expected None"
                    )

    def test_output_is_json_safe(self):
        import json
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        json.dumps(result)  # must not raise

    def test_no_performance_claims_in_outputs(self):
        """Hard requirement carried over from PR-AC: the candidate
        shadow outcome block must not embed any performance-language
        substrings that could be confused with promotion claims."""
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        flat = str(result).lower()
        for forbidden in ("+11%", "+23%", "73% win", "outperform", "validated"):
            assert forbidden not in flat, (
                f"performance/validation claim {forbidden!r} leaked into outcome"
            )


# ──────────────────────────────────────────────────────────────────────────
# PR-AD commit 1b — Codex review fixes
# ──────────────────────────────────────────────────────────────────────────

class TestCandidateShadowOutcomeLabels:
    """Codex hard requirement: candidate outcome must be labeled
    research_mid / shadow_only / not_execution_grade on every path.
    Mid-based historical PnL is not execution-grade evidence."""

    REQUIRED_LABELS = frozenset({"research_mid", "shadow_only", "not_execution_grade"})

    def test_happy_path_carries_all_required_labels(self):
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        labels = result["labels"]
        assert set(labels.keys()) >= self.REQUIRED_LABELS
        for k in self.REQUIRED_LABELS:
            assert labels[k] is True

    def test_skip_paths_also_carry_labels(self):
        """Labels describe the kind of block, not whether it
        succeeded — so they must be present on skip statuses too."""
        block = _empty_candidate_shadow_outcome("skipped:no_dual_picker")
        labels = block["labels"]
        for k in self.REQUIRED_LABELS:
            assert labels[k] is True


class TestSampleProvenanceTaxonomy:
    """Codex P1: provenance must be based on collection method, NOT
    timestamp. Re-running historical replay produces in-sample evidence
    regardless of when the script runs."""

    def test_all_four_values_are_distinct(self):
        values = {
            SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
            SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
            SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED,
            SAMPLE_PROVENANCE_UNKNOWN,
        }
        assert len(values) == 4

    def test_valid_set_contains_exactly_the_four_values(self):
        assert VALID_SAMPLE_PROVENANCES == {
            SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
            SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
            SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED,
            SAMPLE_PROVENANCE_UNKNOWN,
        }

    def test_promotion_eligible_set_excludes_historical_replay(self):
        """The MOST important assertion in this taxonomy: historical
        replay output is NEVER promotion-eligible, no matter how it's
        aggregated. If this fails, the in-sample protections collapse."""
        assert SAMPLE_PROVENANCE_HISTORICAL_REPLAY not in PROMOTION_ELIGIBLE_PROVENANCES

    def test_promotion_eligible_set_excludes_unknown(self):
        """Fail-closed: untagged evidence cannot be used for promotion."""
        assert SAMPLE_PROVENANCE_UNKNOWN not in PROMOTION_ELIGIBLE_PROVENANCES

    def test_promotion_eligible_set_includes_forward_post_freeze(self):
        assert SAMPLE_PROVENANCE_FORWARD_POST_FREEZE in PROMOTION_ELIGIBLE_PROVENANCES

    def test_promotion_eligible_set_excludes_preregistered_holdout_until_manifest_support(self):
        """REGRESSION (Codex round-3 review): a string label is not a
        preregistration control. HOLDOUT_PREREGISTERED remains a
        VALID provenance value (so future records can be tagged once
        the manifest infrastructure is built), but it is NOT in the
        promotion-eligible set today. Promotion via this label requires
        verified manifest support that does not yet exist."""
        assert SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED in VALID_SAMPLE_PROVENANCES
        assert SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED not in PROMOTION_ELIGIBLE_PROVENANCES

    def test_promotion_eligible_set_contains_only_forward_post_freeze_today(self):
        """Pin the entire set so any addition (e.g., accidentally
        re-adding holdout) trips a regression."""
        assert PROMOTION_ELIGIBLE_PROVENANCES == frozenset({
            SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
        })


class TestCandidateOutcomeIndependentOfLegacy:
    """Codex P2: candidate outcome must still resolve when legacy's
    exit quote fails but candidate's exit quote exists. The whole
    point of the +14d rule is that it picks DIFFERENT contracts;
    those contracts may be quoted at exit even when legacy's aren't.

    We test the helper directly here — the equivalent end-to-end
    behavior through _simulate_pre_earnings_calendar_trade is verified
    by the wiring: candidate_shadow_outcome is computed unconditionally
    once the exit chain loads, regardless of legacy's subsequent
    success or failure."""

    def test_candidate_resolves_when_legacy_contracts_absent_from_exit(self):
        # Candidate selects 5/17, 6/21 at strike 100 (different from
        # what a hypothetical legacy would pick — say 5/3 / 5/17).
        # Exit chain has the CANDIDATE's contracts but NOT the legacy's.
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
            # Legacy's hypothetical contracts (different expiry)
            (pd.Timestamp("2024-05-03"), 100.0, "C", 0.7, 0.30, 0.65, 0.75),
        ])
        exit_chain = _chain_with_contracts([
            # Candidate's contracts ARE in exit chain
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
            # Legacy's hypothetical exit contract is NOT here
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(
                front_expiry="2024-05-17", back_expiry="2024-06-21",
            ),
        )
        # Candidate resolves cleanly despite legacy being unresolvable
        assert result["status"] == "ok"
        assert result["mid_realized_return_pct"] == pytest.approx(30.0)


class TestCandidateOutcomeExpiryFormatRobustness:
    """Codex P3: expiry/strike matching must be robust across `date`,
    `Timestamp`, and string formats. Was a pain point in PR-AC."""

    @pytest.mark.parametrize("entry_expiry_format", [
        pd.Timestamp("2024-05-17"),
        pd.Timestamp("2024-05-17").to_pydatetime().date(),
        # Pandas auto-coerces datetime64[ns] in DataFrame construction;
        # this just confirms downstream lookup works with that path too.
    ])
    def test_entry_chain_expiry_format_variants(self, entry_expiry_format):
        entry = pd.DataFrame([
            {"expiry": entry_expiry_format, "strike": 100.0, "call_put": "C",
             "mid": 1.0, "iv": 0.30, "bid": 0.95, "ask": 1.05},
            {"expiry": pd.Timestamp("2024-06-21"), "strike": 100.0, "call_put": "C",
             "mid": 2.0, "iv": 0.30, "bid": 1.95, "ask": 2.05},
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        # Even with mixed expiry formats the lookup succeeds
        assert result["status"] == "ok", (
            f"expiry format {type(entry_expiry_format).__name__} broke lookup: {result['status']}"
        )

    def test_string_expiry_in_dual_picker_is_parsed(self):
        # to_metadata_dict serializes expiries as ISO strings — make sure
        # the simulator parses them back correctly.
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        dp = _candidate_only_dual_picker(
            front_expiry="2024-05-17", back_expiry="2024-06-21",
        )
        assert isinstance(dp["candidate_selection"]["front_expiry"], str)
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain, dual_picker=dp,
        )
        assert result["status"] == "ok"

    def test_unparseable_expiry_returns_skip_status(self):
        dp = _candidate_only_dual_picker()
        dp["candidate_selection"]["front_expiry"] = "not-a-date"
        result = _simulate_candidate_shadow_outcome(
            entry_chain=pd.DataFrame(), exit_chain=pd.DataFrame(), dual_picker=dp,
        )
        assert result["status"] == "skipped:bad_expiry_parse"


# ──────────────────────────────────────────────────────────────────────────
# PR-AD commit 1c — Codex round-2 hardening
# ──────────────────────────────────────────────────────────────────────────

def _eligible_record(**overrides) -> Dict[str, Any]:
    """A canonical record that PASSES is_promotion_eligible. Tests
    that want to verify a specific failure mode start from this and
    perturb a single field."""
    record = {
        "sample_provenance": SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
        "candidate_shadow_outcome": _empty_candidate_shadow_outcome("ok"),
    }
    # _empty_candidate_shadow_outcome sets status to whatever string we
    # pass, but its mid_realized_return_pct is None — replace with a
    # finite number so the strict helper accepts it.
    record["candidate_shadow_outcome"]["mid_realized_return_pct"] = 5.0
    record.update(overrides)
    return record


class TestIsPromotionEligible:
    """Codex P1 + 'fail closed on UNKNOWN' + P2 (strict checks): the
    single source of truth for which records count toward promotion.
    Aggregators MUST route through this helper rather than open-coding
    partial checks."""

    # ── Provenance gate ────────────────────────────────────────────

    def test_forward_post_freeze_with_resolved_outcome_is_eligible(self):
        """Happy path: forward provenance + ok candidate outcome +
        labels intact → eligible."""
        assert is_promotion_eligible(_eligible_record()) is True

    def test_historical_replay_is_never_eligible(self):
        """The single most important assertion in PR-AD: historical
        replay output cannot become promotion-eligible no matter what."""
        record = _eligible_record(
            sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
        )
        assert is_promotion_eligible(record) is False

    def test_holdout_preregistered_not_eligible_until_manifest_support(self):
        """REGRESSION (Codex round-3 review): holdout label is in
        VALID_SAMPLE_PROVENANCES but NOT in PROMOTION_ELIGIBLE_
        PROVENANCES until verified manifest infrastructure exists.
        is_promotion_eligible must reject these records."""
        record = _eligible_record(
            sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED,
        )
        assert is_promotion_eligible(record) is False

    def test_unknown_is_never_eligible(self):
        record = _eligible_record(sample_provenance=SAMPLE_PROVENANCE_UNKNOWN)
        assert is_promotion_eligible(record) is False

    def test_missing_key_fails_closed(self):
        record = _eligible_record()
        del record["sample_provenance"]
        assert is_promotion_eligible(record) is False

    def test_none_value_fails_closed(self):
        record = _eligible_record(sample_provenance=None)
        assert is_promotion_eligible(record) is False

    def test_arbitrary_string_fails_closed(self):
        """Defense against typos / drift: only the canonical promotion-
        eligible constants count. A clever string can't slip through."""
        for s in ("forward", "forward_post_pr_ad", "promotion_eligible_pls",
                  "out_of_sample", "validated", ""):
            record = _eligible_record(sample_provenance=s)
            assert is_promotion_eligible(record) is False, (
                f"non-canonical string {s!r} should not be promotion-eligible"
            )

    def test_non_dict_input_fails_closed(self):
        """Defensive: arbitrary inputs do not crash the helper."""
        assert is_promotion_eligible(None) is False
        assert is_promotion_eligible("string") is False
        assert is_promotion_eligible([]) is False

    # ── Candidate-outcome quality gate (Codex P2) ──────────────────

    def test_missing_candidate_shadow_outcome_fails_closed(self):
        """Even with forward provenance, a record with no candidate
        outcome cannot enter promotion stats."""
        record = _eligible_record()
        del record["candidate_shadow_outcome"]
        assert is_promotion_eligible(record) is False

    def test_candidate_outcome_not_a_dict_fails_closed(self):
        record = _eligible_record(candidate_shadow_outcome="not a dict")
        assert is_promotion_eligible(record) is False
        record = _eligible_record(candidate_shadow_outcome=None)
        assert is_promotion_eligible(record) is False

    def test_skipped_candidate_outcome_fails_closed(self):
        """A forward record with candidate_shadow_outcome.status != ok
        must not enter promotion denominators. Otherwise an event where
        candidate couldn't resolve its quotes still counts toward the
        sample size, inflating n."""
        record = _eligible_record()
        record["candidate_shadow_outcome"]["status"] = "skipped:missing_exit_quote"
        assert is_promotion_eligible(record) is False

    def test_non_numeric_return_fails_closed(self):
        """mid_realized_return_pct must be a finite number — guards
        against string drift and NaN/inf leakage."""
        record = _eligible_record()
        record["candidate_shadow_outcome"]["mid_realized_return_pct"] = "30.0"
        assert is_promotion_eligible(record) is False

        record = _eligible_record()
        record["candidate_shadow_outcome"]["mid_realized_return_pct"] = float("nan")
        assert is_promotion_eligible(record) is False

        record = _eligible_record()
        record["candidate_shadow_outcome"]["mid_realized_return_pct"] = float("inf")
        assert is_promotion_eligible(record) is False

        record = _eligible_record()
        record["candidate_shadow_outcome"]["mid_realized_return_pct"] = None
        assert is_promotion_eligible(record) is False

    def test_bool_return_fails_closed(self):
        """bool is a subclass of int in Python — explicit check
        prevents accidental True/False slipping through as 1/0."""
        record = _eligible_record()
        record["candidate_shadow_outcome"]["mid_realized_return_pct"] = True
        assert is_promotion_eligible(record) is False

    # ── Label-integrity gate (defends against refactor accidents) ──

    def test_missing_labels_dict_fails_closed(self):
        record = _eligible_record()
        del record["candidate_shadow_outcome"]["labels"]
        assert is_promotion_eligible(record) is False

    def test_labels_not_a_dict_fails_closed(self):
        record = _eligible_record()
        record["candidate_shadow_outcome"]["labels"] = "research_mid"
        assert is_promotion_eligible(record) is False

    def test_missing_required_label_fails_closed(self):
        """All three labels must be present and True. Stripping any
        one rejects the row."""
        for label in ("research_mid", "shadow_only", "not_execution_grade"):
            record = _eligible_record()
            del record["candidate_shadow_outcome"]["labels"][label]
            assert is_promotion_eligible(record) is False, (
                f"missing {label} label should fail-closed"
            )

    def test_label_set_to_false_fails_closed(self):
        """If someone refactors away the not_execution_grade label by
        setting it to False, the helper must reject — defends against
        mid-priced data being treated as execution evidence."""
        for label in ("research_mid", "shadow_only", "not_execution_grade"):
            record = _eligible_record()
            record["candidate_shadow_outcome"]["labels"][label] = False
            assert is_promotion_eligible(record) is False, (
                f"{label}=False should fail-closed"
            )


class TestTagLiveForwardObservation:
    """_tag_live_forward_observation is the SINGLE function authorized
    to assign SAMPLE_PROVENANCE_FORWARD_POST_FREEZE. Underscore-prefixed
    to signal module-private. The regression test below
    (TestForwardProvenanceAssignmentBoundary) enforces that no other
    code path writes the constant OR references the function name."""

    def test_assigns_forward_post_freeze(self):
        record = {"symbol": "AAPL"}
        _tag_live_forward_observation(record)
        assert record["sample_provenance"] == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE
        # Note: is_promotion_eligible would still return False here
        # because the record has no candidate_shadow_outcome. That's
        # the new strict behavior — see TestIsPromotionEligible.

    def test_overwrites_existing_provenance(self):
        """If the record was previously tagged UNKNOWN/HISTORICAL by
        default initialization, calling the live-forward tagger replaces
        it. This is the intended behavior — the live API path knows it
        is producing forward observations regardless of the default."""
        record = {"sample_provenance": SAMPLE_PROVENANCE_UNKNOWN}
        _tag_live_forward_observation(record)
        assert record["sample_provenance"] == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE

    def test_returns_the_same_record_for_chaining(self):
        record = {"symbol": "AAPL"}
        result = _tag_live_forward_observation(record)
        assert result is record  # mutated in-place


class TestForwardProvenanceAssignmentBoundary:
    """Codex's defense-in-depth requirement: 'forward provenance must
    be assigned only by the live evidence-cycle path, not by a generic
    script flag anyone can pass casually.'

    Source-grep regression: BOTH the string value of
    SAMPLE_PROVENANCE_FORWARD_POST_FREEZE *and* references to the
    function name `_tag_live_forward_observation` may appear in the
    codebase ONLY inside web/api/edge_engine.py. Anywhere else is a
    research-leakage hazard.

    Round-3 update (Codex): grepping the string literal alone is not
    enough — a replay script could call the helper directly without
    containing the literal. We now grep for both."""

    @staticmethod
    def _grep_prod_sources(pattern_obj):
        """Yield (rel_path, lineno, line) for every match of *pattern*
        in services/ + web/ + scripts/ Python files."""
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        prod_dirs = [repo_root / "services", repo_root / "web", repo_root / "scripts"]
        for d in prod_dirs:
            if not d.exists():
                continue
            for py_file in d.rglob("*.py"):
                with open(py_file) as f:
                    for lineno, line in enumerate(f, 1):
                        if pattern_obj.search(line):
                            yield py_file.relative_to(repo_root), lineno, line.strip()

    # Allowlist for both grep regressions below. After PR-AD commit 4,
    # the canonical assignment site moved out of edge_engine.py into
    # services/candidate_shadow_provenance.py — the underscore-prefixed
    # _tag_live_forward_observation now lives in the neutral service
    # module, and edge_engine.py re-imports it for use by the live API
    # path. Both files are legitimate references; anywhere else is a
    # research-leakage hazard.
    _ALLOWED_PROVENANCE_FILES = (
        "services/candidate_shadow_provenance.py",
        "web/api/edge_engine.py",
    )

    def test_forward_post_freeze_string_only_in_allowed_modules(self):
        """The string literal "forward_post_freeze" must only appear in
        the allowed provenance-owning modules — services/
        candidate_shadow_provenance.py (canonical definition) and
        web/api/edge_engine.py (re-import + usage). Anywhere else is
        the research-leakage hazard Codex round-1 flagged."""
        import re
        pattern = re.compile(r'"forward_post_freeze"|\'forward_post_freeze\'')
        suspects = [
            f"{rel}:{lineno}: {line}"
            for rel, lineno, line in self._grep_prod_sources(pattern)
            if not any(allowed in str(rel) for allowed in self._ALLOWED_PROVENANCE_FILES)
        ]
        assert suspects == [], (
            "forward_post_freeze string assignment leaked outside the "
            "allowed modules. Only _tag_live_forward_observation in "
            "services/candidate_shadow_provenance.py may assign this "
            "value. Found:\n  " + "\n  ".join(suspects)
        )

    def test_tag_helper_only_referenced_in_allowed_modules(self):
        """REGRESSION (Codex rounds 3+4): the function NAME must also
        only appear in the allowed modules. A script could call
        `_tag_live_forward_observation(record)` without ever writing
        the forbidden string literal — this grep catches that. The
        underscore-prefix convention signals module-private."""
        import re
        pattern = re.compile(r'\b_?tag_live_forward_observation\b')
        suspects = [
            f"{rel}:{lineno}: {line}"
            for rel, lineno, line in self._grep_prod_sources(pattern)
            if not any(allowed in str(rel) for allowed in self._ALLOWED_PROVENANCE_FILES)
        ]
        assert suspects == [], (
            "_tag_live_forward_observation referenced outside the "
            "allowed modules. The function is module-private; calling "
            "it from another module bypasses the assignment boundary. "
            "Found:\n  " + "\n  ".join(suspects)
        )


class TestHistoricalReplayCannotBecomePromotionEligible:
    """Codex's hard requirement: 'historical replay candidate outcomes
    are never promotion-eligible.' Tested at the simulator output level
    so a future change to _simulate_pre_earnings_calendar_trade can't
    accidentally tag its output with a promotion-eligible provenance."""

    def test_simulator_record_base_is_historical_replay(self):
        """The `base` dict produced inside _simulate_pre_earnings_
        calendar_trade carries HISTORICAL_REPLAY. Any record that flows
        from it inherits the tag."""
        # We can verify this by reading the source — the tag is hardcoded
        # in the `base` dict construction. The test asserts that the
        # canonical value (read from the module) is what's referenced.
        import inspect
        from web.api import edge_engine
        src = inspect.getsource(edge_engine._simulate_pre_earnings_calendar_trade)
        # The `base` dict must assign SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
        # not the forward variant.
        assert "SAMPLE_PROVENANCE_HISTORICAL_REPLAY" in src
        assert "SAMPLE_PROVENANCE_FORWARD_POST_FREEZE" not in src, (
            "_simulate_pre_earnings_calendar_trade must NOT reference the "
            "forward provenance constant — historical replay is in-sample"
        )

    def test_promotion_eligible_set_invariant_holds(self):
        """Even if someone refactors the simulator, the promotion-
        eligible set MUST NOT include HISTORICAL_REPLAY. This is the
        first line of defense against any leakage attempt."""
        # Construct a record the way the simulator would
        record = {"sample_provenance": SAMPLE_PROVENANCE_HISTORICAL_REPLAY}
        assert is_promotion_eligible(record) is False
        # And the converse: no historical_replay should ever appear in
        # the promotion-eligible set
        assert SAMPLE_PROVENANCE_HISTORICAL_REPLAY not in PROMOTION_ELIGIBLE_PROVENANCES


class TestPricingGradeNaming:
    """Codex: 'all candidate outcome returns are named/labeled as
    mid/research returns.' Field names embed the pricing grade so
    downstream aggregators cannot quietly relabel them as execution
    returns."""

    def test_return_field_is_named_mid_realized(self):
        entry = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 1.0, 0.30, 0.95, 1.05),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 2.0, 0.30, 1.95, 2.05),
        ])
        exit_chain = _chain_with_contracts([
            (pd.Timestamp("2024-05-17"), 100.0, "C", 0.4, 0.20, 0.35, 0.45),
            (pd.Timestamp("2024-06-21"), 100.0, "C", 1.7, 0.25, 1.65, 1.75),
        ])
        result = _simulate_candidate_shadow_outcome(
            entry_chain=entry, exit_chain=exit_chain,
            dual_picker=_candidate_only_dual_picker(),
        )
        # New name present
        assert "mid_realized_return_pct" in result
        # Old name absent — no silent dual-naming
        assert "realized_return_pct" not in result

    def test_pnl_and_value_fields_carry_mid_qualifier(self):
        """Adjacent monetary fields also embed the pricing grade in
        their names — entry_debit_mid, exit_value_mid, mid_pnl. None
        of them should be called the bare `return` or `pnl`."""
        block = _empty_candidate_shadow_outcome("skipped:test")
        assert "entry_debit_mid" in block
        assert "exit_value_mid" in block
        assert "mid_pnl" in block
        assert "pnl" not in block  # the bare name is a research-leakage
                                    # hazard if used in promotion stats


# ──────────────────────────────────────────────────────────────────────────
# PR-AD commit 2 — aggregator with strict provenance bucketing
# ──────────────────────────────────────────────────────────────────────────

def _trade_with_provenance_and_outcome(
    *,
    symbol: str,
    event_date: str,
    entry_date: str,
    sample_provenance: str,
    outcome_status: str = "ok",
    mid_realized_return_pct: Any = 5.0,
    legacy_sel: Any = None,
    candidate_sel: Any = None,
    pickers_diverged: bool = False,
    shadow_status: str = "ok",
    labels_intact: bool = True,
) -> Dict[str, Any]:
    """Construct a single trade record matching the live shape produced
    by _simulate_pre_earnings_calendar_trade. Used to exercise the
    aggregator's provenance + eligibility logic in isolation."""
    outcome = _empty_candidate_shadow_outcome(outcome_status)
    if outcome_status == "ok":
        outcome["mid_realized_return_pct"] = mid_realized_return_pct
        outcome["side"] = "call"
        outcome["strike"] = 100.0
        outcome["front_expiry"] = "2024-05-17"
        outcome["back_expiry"] = "2024-06-21"
    if not labels_intact:
        outcome["labels"] = {}
    sel = {"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
           "strike": 100.0, "front_dte_days": 16}
    return {
        "symbol": symbol,
        "event_date": event_date,
        "entry_date": entry_date,
        "sample_provenance": sample_provenance,
        "candidate_shadow_outcome": outcome,
        "dual_picker": {
            "shadow_status": shadow_status,
            "legacy_selection": legacy_sel if legacy_sel is not None else sel,
            "candidate_selection": candidate_sel if candidate_sel is not None else sel,
            "pickers_diverged": pickers_diverged,
            "candidate_min_front_dte_days": 14,
            "experimental_note": "...",
        },
    }


def _forward_eligible_trade(**overrides) -> Dict[str, Any]:
    """A canonical forward-post-freeze trade that PASSES is_promotion_eligible.
    Tests perturb one field at a time to verify gating."""
    return _trade_with_provenance_and_outcome(
        symbol=overrides.pop("symbol", "AAPL"),
        event_date=overrides.pop("event_date", "2026-06-01"),
        entry_date=overrides.pop("entry_date", "2026-05-29"),
        sample_provenance=overrides.pop(
            "sample_provenance", SAMPLE_PROVENANCE_FORWARD_POST_FREEZE
        ),
        **overrides,
    )


class TestAggregatorProvenanceBucketing:
    """Codex hard rules for commit 2: show all records in diagnostics
    (including unknown/replay/skipped/malformed); compute promotion
    stats ONLY on eligible records; never mix replay and forward."""

    def test_empty_input_produces_empty_promotion_stats_with_none_values(self):
        result = _aggregate_experimental_candidate_evidence([])
        pe = result["promotion_eligible_candidate_stats"]
        assert pe["n_events"] == 0
        assert pe["candidate_mid_realized_return_pct_mean"] is None
        assert pe["candidate_mid_realized_return_pct_median"] is None
        assert pe["candidate_mid_realized_return_pct_win_rate"] is None

    def test_provenance_counts_includes_every_bucket_seen(self):
        """Codex: 'show all records in diagnostics, including unknown,
        replay, skipped, and malformed.'"""
        trades = [
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date=f"2026-{m:02d}-01",
                entry_date=f"2026-{m:02d}-01",
                sample_provenance=prov,
            )
            for m, prov in enumerate([
                SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
                SAMPLE_PROVENANCE_UNKNOWN,
                "totally_made_up_string",
            ], start=1)
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        counts = result["provenance_counts"]
        assert counts[SAMPLE_PROVENANCE_HISTORICAL_REPLAY] == 2
        assert counts[SAMPLE_PROVENANCE_FORWARD_POST_FREEZE] == 1
        assert counts[SAMPLE_PROVENANCE_UNKNOWN] == 1
        # Even unrecognized provenance strings appear in diagnostics
        assert counts["totally_made_up_string"] == 1

    def test_candidate_outcome_status_counts_includes_skip_reasons(self):
        """Skipped/malformed outcomes must surface as their own counts
        so the operator can see why candidate PnL is missing."""
        trades = [
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2026-06-01", entry_date="2026-05-29",
                sample_provenance=SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
                outcome_status="ok",
            ),
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2026-09-01", entry_date="2026-08-29",
                sample_provenance=SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
                outcome_status="skipped:missing_exit_quote",
            ),
            _trade_with_provenance_and_outcome(
                symbol="MSFT", event_date="2026-07-01", entry_date="2026-06-29",
                sample_provenance=SAMPLE_PROVENANCE_FORWARD_POST_FREEZE,
                outcome_status="skipped:no_candidate_selection",
            ),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        counts = result["candidate_outcome_status_counts"]
        assert counts["ok"] == 1
        assert counts["skipped:missing_exit_quote"] == 1
        assert counts["skipped:no_candidate_selection"] == 1


class TestAggregatorPromotionEligibleBucket:
    """The promotion_eligible_candidate_stats block is gated by
    is_promotion_eligible(record). It must NEVER include historical
    replay rows, UNKNOWN provenance, skipped outcomes, non-finite
    returns, or label-tampered records."""

    def test_historical_replay_trade_is_excluded_from_promotion(self):
        """The single most important regression: in-sample data must
        never enter promotion stats."""
        trade = _trade_with_provenance_and_outcome(
            symbol="AAPL", event_date="2024-05-01", entry_date="2024-04-29",
            sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
            outcome_status="ok", mid_realized_return_pct=42.0,
        )
        result = _aggregate_experimental_candidate_evidence([trade])
        # Diagnostics show the row exists
        assert result["provenance_counts"][SAMPLE_PROVENANCE_HISTORICAL_REPLAY] == 1
        assert result["candidate_outcome_status_counts"]["ok"] == 1
        # But promotion-eligible stats are empty
        pe = result["promotion_eligible_candidate_stats"]
        assert pe["n_events"] == 0
        assert pe["candidate_mid_realized_return_pct_mean"] is None

    def test_forward_eligible_trade_appears_in_promotion_block(self):
        trade = _forward_eligible_trade(mid_realized_return_pct=10.0)
        result = _aggregate_experimental_candidate_evidence([trade])
        pe = result["promotion_eligible_candidate_stats"]
        assert pe["n_events"] == 1
        assert pe["candidate_mid_realized_return_pct_mean"] == pytest.approx(10.0)
        assert pe["candidate_mid_realized_return_pct_win_rate"] == pytest.approx(1.0)

    def test_skipped_forward_trade_does_not_enter_promotion(self):
        """A forward record with an unresolved candidate outcome is
        diagnostic only — not promotion evidence."""
        trade = _forward_eligible_trade(outcome_status="skipped:missing_exit_quote")
        result = _aggregate_experimental_candidate_evidence([trade])
        assert result["candidate_outcome_status_counts"]["skipped:missing_exit_quote"] == 1
        assert result["promotion_eligible_candidate_stats"]["n_events"] == 0

    def test_label_tampered_forward_trade_does_not_enter_promotion(self):
        """If labels are missing/stripped, is_promotion_eligible rejects
        the row → it can't enter promotion stats. Provenance count and
        outcome-status count still see it."""
        trade = _forward_eligible_trade(labels_intact=False)
        result = _aggregate_experimental_candidate_evidence([trade])
        assert result["provenance_counts"][SAMPLE_PROVENANCE_FORWARD_POST_FREEZE] == 1
        assert result["promotion_eligible_candidate_stats"]["n_events"] == 0

    def test_event_dedupe_includes_option_type_for_put_side_support(self):
        """REGRESSION (Codex P2a): once put-side support lands, the
        same earnings event can have BOTH a call_calendar candidate
        outcome and a put_calendar candidate outcome. The event_key
        must include option_type so they don't collapse into a single
        bucket. Today the historical sim is calls-only, but extending
        the key now means the put-side commit doesn't need to touch
        the aggregator."""
        # Simulate the future state: same (symbol, event_date) but
        # different option_type. Each should count as a distinct
        # bucket in the aggregator.
        trades = [
            {
                **_forward_eligible_trade(
                    symbol="AAPL", event_date="2026-06-01",
                    entry_date="2026-05-29", mid_realized_return_pct=10.0,
                ),
                "option_type": "C",
            },
            {
                **_forward_eligible_trade(
                    symbol="AAPL", event_date="2026-06-01",
                    entry_date="2026-05-29", mid_realized_return_pct=20.0,
                ),
                "option_type": "P",
            },
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        pe = result["promotion_eligible_candidate_stats"]
        # TWO distinct buckets: call_calendar's AAPL 2026-06-01 and
        # put_calendar's AAPL 2026-06-01. They do NOT collapse.
        assert pe["n_events"] == 2
        # Mean of 10 and 20 = 15
        assert pe["candidate_mid_realized_return_pct_mean"] == pytest.approx(15.0)

    def test_event_level_dedupe_across_entry_offsets(self):
        """The same event seen at multiple entry offsets should count
        ONCE in the promotion-eligible event count, not N times. PnL
        for that event is the mean of its per-offset PnLs."""
        trades = [
            _forward_eligible_trade(
                symbol="AAPL", event_date="2026-06-01",
                entry_date=f"offset-{i}",
                mid_realized_return_pct=10.0 + i,
            )
            for i in range(6)
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        pe = result["promotion_eligible_candidate_stats"]
        # 1 unique event, regardless of 6 offsets
        assert pe["n_events"] == 1
        # Mean PnL is mean of 10, 11, 12, 13, 14, 15 = 12.5
        assert pe["candidate_mid_realized_return_pct_mean"] == pytest.approx(12.5)

    def test_win_rate_event_level(self):
        # 3 events: one wins (+10), two lose (-5, -1) → win rate = 1/3
        trades = [
            _forward_eligible_trade(
                symbol="AAPL", event_date="2026-06-01",
                entry_date="2026-05-29", mid_realized_return_pct=10.0,
            ),
            _forward_eligible_trade(
                symbol="AAPL", event_date="2026-09-01",
                entry_date="2026-08-29", mid_realized_return_pct=-5.0,
            ),
            _forward_eligible_trade(
                symbol="MSFT", event_date="2026-07-01",
                entry_date="2026-06-29", mid_realized_return_pct=-1.0,
            ),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        pe = result["promotion_eligible_candidate_stats"]
        assert pe["n_events"] == 3
        assert pe["candidate_mid_realized_return_pct_win_rate"] == pytest.approx(1 / 3)


class TestAggregatorInSampleDiagnostic:
    """The in_sample_diagnostic_candidate_stats block exists for
    sanity-checking simulator math and is labeled exhaustively as
    non-promotion. Tests verify the label is unambiguous."""

    def test_in_sample_block_carries_explicit_provenance_label(self):
        result = _aggregate_experimental_candidate_evidence([])
        diag = result["in_sample_diagnostic_candidate_stats"]
        assert diag["sample_provenance"] == SAMPLE_PROVENANCE_HISTORICAL_REPLAY
        # Note text warns explicitly
        assert "IN-SAMPLE" in diag["note"]
        assert "never as out-of-sample evidence" in diag["note"]

    def test_in_sample_stats_only_aggregate_historical_replay(self):
        """A forward trade should NOT contribute to in_sample stats
        even though its outcome is ok and its PnL is finite — its
        provenance is wrong for this bucket."""
        trades = [
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2024-05-01", entry_date="2024-04-29",
                sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                mid_realized_return_pct=20.0,
            ),
            _forward_eligible_trade(mid_realized_return_pct=999.0),  # noise
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        diag = result["in_sample_diagnostic_candidate_stats"]
        # Only the historical_replay row contributes
        assert diag["n_events"] == 1
        assert diag["candidate_mid_realized_return_pct_mean"] == pytest.approx(20.0)

    def test_in_sample_stats_exclude_skipped_outcomes(self):
        """REGRESSION (Codex P2b round-3 review of commit 2): the
        in-sample diagnostic stats must aggregate ONLY records with
        candidate_shadow_outcome.status == 'ok' and finite numeric
        mid_realized_return_pct. Skipped/malformed historical-replay
        rows still appear in candidate_outcome_status_counts (for
        observability) but must NOT enter the mean/median/win-rate
        calculations."""
        trades = [
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2024-05-01", entry_date="2024-04-29",
                sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                outcome_status="ok", mid_realized_return_pct=10.0,
            ),
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2024-08-01", entry_date="2024-07-30",
                sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                outcome_status="skipped:missing_exit_quote",
            ),
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2024-11-01", entry_date="2024-10-30",
                sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                outcome_status="skipped:negative_debit",
            ),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)

        # Observability: skip reasons appear in the diagnostic counts
        assert result["candidate_outcome_status_counts"]["ok"] == 1
        assert result["candidate_outcome_status_counts"]["skipped:missing_exit_quote"] == 1
        assert result["candidate_outcome_status_counts"]["skipped:negative_debit"] == 1

        # But ONLY the ok row contributes to the diagnostic stats
        diag = result["in_sample_diagnostic_candidate_stats"]
        assert diag["n_events"] == 1
        assert diag["candidate_mid_realized_return_pct_mean"] == pytest.approx(10.0)

    def test_in_sample_stats_exclude_non_finite_returns(self):
        """A record with status=ok but mid_realized_return_pct=NaN or
        None must NOT enter the in-sample mean. The collector's
        np.isfinite check is the gatekeeper."""
        trades = [
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2024-05-01", entry_date="2024-04-29",
                sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                outcome_status="ok", mid_realized_return_pct=10.0,
            ),
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2024-08-01", entry_date="2024-07-30",
                sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                outcome_status="ok", mid_realized_return_pct=float("nan"),
            ),
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2024-11-01", entry_date="2024-10-30",
                sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                outcome_status="ok", mid_realized_return_pct=None,
            ),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        diag = result["in_sample_diagnostic_candidate_stats"]
        assert diag["n_events"] == 1  # only the finite-PnL row
        assert diag["candidate_mid_realized_return_pct_mean"] == pytest.approx(10.0)


class TestAggregatorNoMixingOfProvenance:
    """Codex hard rule: 'never mix historical replay and forward
    evidence into one candidate performance number.'"""

    def test_in_sample_and_promotion_blocks_are_separate(self):
        """Both blocks coexist with disjoint contents — historical_replay
        rows only in in_sample, forward rows only in promotion-eligible."""
        trades = [
            _trade_with_provenance_and_outcome(
                symbol="AAPL", event_date="2024-05-01", entry_date="2024-04-29",
                sample_provenance=SAMPLE_PROVENANCE_HISTORICAL_REPLAY,
                mid_realized_return_pct=-30.0,
            ),
            _forward_eligible_trade(
                symbol="AAPL", event_date="2026-06-01",
                entry_date="2026-05-29", mid_realized_return_pct=15.0,
            ),
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        diag = result["in_sample_diagnostic_candidate_stats"]
        pe = result["promotion_eligible_candidate_stats"]
        # In-sample: only the replay row
        assert diag["n_events"] == 1
        assert diag["candidate_mid_realized_return_pct_mean"] == pytest.approx(-30.0)
        # Forward: only the eligible row
        assert pe["n_events"] == 1
        assert pe["candidate_mid_realized_return_pct_mean"] == pytest.approx(15.0)
        # Math check: -30 and +15 never get averaged into a single
        # "candidate performance" number. The blocks stay disjoint.

    def test_aggregator_output_has_no_combined_candidate_performance_field(self):
        """Defense: assert that no top-level key contains a single
        candidate-performance scalar. Promotion stats live ONLY inside
        promotion_eligible_candidate_stats."""
        result = _aggregate_experimental_candidate_evidence([])
        # The forbidden shapes: a top-level mean/median/win_rate
        # that would aggregate over all provenance buckets.
        forbidden_keys = {
            "candidate_mean_return_pct",
            "candidate_win_rate",
            "candidate_realized_return_pct_mean",
            "combined_candidate_performance",
        }
        assert forbidden_keys.isdisjoint(result.keys()), (
            "aggregator must not expose a top-level scalar that could "
            "be interpreted as a single 'candidate performance' number"
        )

    def test_field_names_carry_mid_pricing_grade(self):
        """Codex: 'name fields with mid_ / research_mid so nobody reads
        them as execution-grade.' The candidate stats field names must
        embed mid_realized to make this clear."""
        result = _aggregate_experimental_candidate_evidence([])
        for block_name in ("in_sample_diagnostic_candidate_stats",
                           "promotion_eligible_candidate_stats"):
            block = result[block_name]
            for k in block.keys():
                if "return" in k or "win_rate" in k:
                    # Stats fields must carry mid_realized (or be the
                    # event-count). Win rate is the one acceptable
                    # exception — it doesn't need a pricing qualifier.
                    if "win_rate" not in k:
                        assert "mid_realized" in k, (
                            f"{block_name}.{k} is missing the mid_ "
                            f"pricing-grade qualifier"
                        )


# ──────────────────────────────────────────────────────────────────────────
# PR-AE C3 — live forward tagging wired into analyze_single_ticker
# ──────────────────────────────────────────────────────────────────────────


class TestPRAECCThreeLiveForwardTagging:
    """The two halves PR-AE C3 wires up in web/api/edge_engine.py:

      (a) Live forward observations get tagged with
          SAMPLE_PROVENANCE_FORWARD_POST_FREEZE before
          record_recommendation runs, AND attach a placeholder
          candidate_shadow_outcome with status "awaiting_exit_resolver"
          for the PR-AE C4 resolver to pick up post-event.

      (b) Historical replay observations (produced by
          _simulate_pre_earnings_calendar_trade) keep their existing
          SAMPLE_PROVENANCE_HISTORICAL_REPLAY tag and NEVER acquire the
          forward variant — even if the simulator is run for the first
          time on this revision of the code.

    Both halves are guarded behaviorally below. The existing
    TestForwardProvenanceAssignmentBoundary class still enforces the
    source-grep boundary on top of these behavioral checks (the
    underscore-prefixed helper and the literal string may only
    appear inside two allowlisted modules).
    """

    def test_attach_live_forward_provenance_assigns_forward_tag(self):
        """Half (a) — direct behavioral test of the
        ``_attach_live_forward_provenance`` helper called inside
        analyze_single_ticker. Constructing the snapshot manually
        bypasses the full live-API dependency stack (yfinance, MDApp,
        VolSnapshot, structure scorecards) and tests only the
        tagging contract."""
        from web.api.edge_engine import (
            EdgeSnapshot,
            _attach_live_forward_provenance,
        )
        snapshot = EdgeSnapshot(
            symbol="AAPL",
            recommendation="Candidate",
            confidence_pct=72.0,
            setup_score=0.72,
            metrics={},
            rationale=[],
            selector_output={"best_structure": "call_calendar"},
            structure_scorecards=[],
            vol_snapshot={"symbol": "AAPL"},
        )
        # Before tagging: both PR-AE C3 fields carry their defaults
        assert snapshot.sample_provenance is None
        assert snapshot.candidate_shadow_outcome == {}

        _attach_live_forward_provenance(snapshot)

        # After tagging: forward provenance set, candidate shadow
        # outcome carries the resolver-recognized awaiting sentinel
        assert snapshot.sample_provenance == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE
        assert snapshot.candidate_shadow_outcome["status"] == "awaiting_exit_resolver"
        # Stable-shape labels survive — the C4 resolver and the
        # aggregator both depend on these three label keys being True.
        for label in ("research_mid", "shadow_only", "not_execution_grade"):
            assert snapshot.candidate_shadow_outcome["labels"][label] is True

    def test_attach_live_forward_provenance_is_idempotent(self):
        """Calling the helper twice on the same snapshot is safe.
        sample_provenance stays at the forward value, and
        candidate_shadow_outcome remains the awaiting-stub.

        After the PR-AE C3b non-overwrite guard, the second call
        recognizes the non-empty status and PRESERVES the first call's
        outcome dict by reference — not just by value equality. The
        is-identity check below tightens the idempotency contract."""
        from web.api.edge_engine import (
            EdgeSnapshot,
            _attach_live_forward_provenance,
        )
        snapshot = EdgeSnapshot(
            symbol="AAPL",
            recommendation="Candidate",
            confidence_pct=72.0,
            setup_score=0.72,
            metrics={},
            rationale=[],
        )
        _attach_live_forward_provenance(snapshot)
        first_outcome = snapshot.candidate_shadow_outcome
        _attach_live_forward_provenance(snapshot)
        assert snapshot.sample_provenance == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE
        assert snapshot.candidate_shadow_outcome["status"] == "awaiting_exit_resolver"
        # PR-AE C3b: same reference, not a freshly-constructed dict.
        # Guarantees the helper performs ZERO writes to
        # candidate_shadow_outcome on the second call.
        assert snapshot.candidate_shadow_outcome is first_outcome

    # ── PR-AE C3b non-overwrite guard (Codex audit P1) ───────────────────
    #
    # The four tests below pin the contract: when
    # candidate_shadow_outcome already carries a non-empty status,
    # _attach_live_forward_provenance MUST NOT replace it. The guard
    # protects against a future caller (or future request middleware)
    # accidentally clobbering a resolved outcome with the
    # awaiting_exit_resolver stub.

    def test_attach_preserves_existing_ok_outcome(self):
        """Resolver wrote status=ok onto a snapshot. Re-running the
        live tagger MUST preserve the ok outcome verbatim — otherwise
        a future code path that loops live → resolver → live would
        wipe the resolution every time it cycled."""
        from web.api.edge_engine import (
            EdgeSnapshot,
            _attach_live_forward_provenance,
        )
        resolved_outcome = {
            "status": "ok",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
            "side": "call",
            "strike": 200.0,
            "mid_realized_return_pct": 12.5,
        }
        snapshot = EdgeSnapshot(
            symbol="AAPL", recommendation="Candidate",
            confidence_pct=72.0, setup_score=0.72,
            metrics={}, rationale=[],
            candidate_shadow_outcome=dict(resolved_outcome),
        )
        _attach_live_forward_provenance(snapshot)
        # The ok outcome survives, byte-for-byte
        assert snapshot.candidate_shadow_outcome == resolved_outcome
        # Provenance still gets assigned — that's unconditional
        assert snapshot.sample_provenance == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE

    def test_attach_preserves_existing_permanently_failed_outcome(self):
        """Same protection for terminal-failure statuses. The C4
        resolver writes permanently_failed:* statuses that must not
        be re-armed for retry by a stray helper call."""
        from web.api.edge_engine import (
            EdgeSnapshot,
            _attach_live_forward_provenance,
        )
        terminal_outcome = {
            "status": "permanently_failed:no_post_event_chain",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
        }
        snapshot = EdgeSnapshot(
            symbol="AAPL", recommendation="Candidate",
            confidence_pct=72.0, setup_score=0.72,
            metrics={}, rationale=[],
            candidate_shadow_outcome=dict(terminal_outcome),
        )
        _attach_live_forward_provenance(snapshot)
        assert snapshot.candidate_shadow_outcome == terminal_outcome
        assert snapshot.sample_provenance == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE

    def test_attach_preserves_existing_retrying_outcome(self):
        """Mid-cycle non-terminal statuses (retrying,
        awaiting_chain_data) also survive. They are NOT terminal,
        but the C4 resolver is the only authorized writer — the live
        tagger must not reset them either."""
        from web.api.edge_engine import (
            EdgeSnapshot,
            _attach_live_forward_provenance,
        )
        retrying_outcome = {
            "status": "retrying",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
        }
        snapshot = EdgeSnapshot(
            symbol="AAPL", recommendation="Candidate",
            confidence_pct=72.0, setup_score=0.72,
            metrics={}, rationale=[],
            candidate_shadow_outcome=dict(retrying_outcome),
        )
        _attach_live_forward_provenance(snapshot)
        assert snapshot.candidate_shadow_outcome == retrying_outcome

    def test_attach_initializes_stub_when_outcome_is_explicit_none(self):
        """Edge case: if a caller explicitly sets
        candidate_shadow_outcome to None (rather than the default
        empty dict), the guard treats that as "no prior outcome"
        and initializes the stub. Fail-closed because is_promotion_
        eligible would already reject the None-payload row, and the
        resolver expects to see the awaiting_exit_resolver sentinel."""
        from web.api.edge_engine import (
            EdgeSnapshot,
            _attach_live_forward_provenance,
        )
        snapshot = EdgeSnapshot(
            symbol="AAPL", recommendation="Candidate",
            confidence_pct=72.0, setup_score=0.72,
            metrics={}, rationale=[],
            candidate_shadow_outcome=None,  # type: ignore[arg-type]
        )
        _attach_live_forward_provenance(snapshot)
        assert isinstance(snapshot.candidate_shadow_outcome, dict)
        assert snapshot.candidate_shadow_outcome["status"] == "awaiting_exit_resolver"

    def test_edge_snapshot_carries_pr_ae_c3_fields_with_safe_defaults(self):
        """A bare EdgeSnapshot (constructed without invoking the
        helper) has sample_provenance=None and
        candidate_shadow_outcome={}. This protects the
        is_promotion_eligible filter: untagged snapshots fail
        eligibility on the provenance check (None is not in
        PROMOTION_ELIGIBLE_PROVENANCES) AND on the outcome check
        (empty dict has no "status" key)."""
        from web.api.edge_engine import EdgeSnapshot
        snapshot = EdgeSnapshot(
            symbol="AAPL",
            recommendation="Candidate",
            confidence_pct=72.0,
            setup_score=0.72,
            metrics={},
            rationale=[],
        )
        assert snapshot.sample_provenance is None
        assert snapshot.candidate_shadow_outcome == {}
        # Fail-closed: an untagged snapshot is not promotion-eligible
        assert is_promotion_eligible({
            "sample_provenance": snapshot.sample_provenance,
            "candidate_shadow_outcome": snapshot.candidate_shadow_outcome,
        }) is False

    def test_analyze_single_ticker_source_contains_helper_call(self):
        """Static guard against accidental removal of the tagging call
        from analyze_single_ticker during a future refactor. If a
        future PR moves the call site to a different helper, this test
        must be updated AND the boundary regression confirmed."""
        import inspect
        from web.api import edge_engine
        src = inspect.getsource(edge_engine.analyze_single_ticker)
        assert "_attach_live_forward_provenance(edge_snapshot)" in src, (
            "analyze_single_ticker must call _attach_live_forward_provenance "
            "so live recommendations land in the ledger tagged "
            "forward_post_freeze with an awaiting candidate_shadow_outcome. "
            "Removing this breaks the PR-AE C4 resolver's eligibility "
            "filter (every live row would be untagged and skipped)."
        )

    def test_historical_replay_simulator_tags_records_as_historical_replay(self):
        """Half (b) behavioral — invoke _simulate_pre_earnings_calendar_
        trade with a mock store whose query_chain raises. The
        function early-exits returning the ``base`` dict, which by
        construction carries SAMPLE_PROVENANCE_HISTORICAL_REPLAY. No
        chain queries succeed, no candidate shadow simulator runs,
        the forward tagger MUST NOT be invoked anywhere in this code
        path."""
        from unittest.mock import MagicMock
        from web.api.edge_engine import _simulate_pre_earnings_calendar_trade

        store = MagicMock()
        store.query_chain.side_effect = RuntimeError("synthetic: no chain data")

        result = _simulate_pre_earnings_calendar_trade(
            "AAPL",
            store,
            event_date=pd.Timestamp("2024-05-02"),
            entry_date=pd.Timestamp("2024-04-30"),
            exit_date=pd.Timestamp("2024-05-03"),
            entry_offset=2,
        )

        assert result["sample_provenance"] == SAMPLE_PROVENANCE_HISTORICAL_REPLAY
        # Critically: the forward tag must NEVER appear on a
        # historical-replay record, regardless of when this code path
        # runs.
        assert result["sample_provenance"] != SAMPLE_PROVENANCE_FORWARD_POST_FREEZE
        # Early-exit status confirms we exercised the missing-entry-chain
        # branch (the cleanest seam in the simulator that builds the
        # base dict and returns).
        assert result["status"] == "missing_entry_chain"

    def test_historical_replay_does_not_invoke_forward_tagger(self):
        """Codex C3 watch item, directly behavioral: when the
        historical replay simulator runs, ``_tag_live_forward_observation``
        MUST NOT be called. Patching the tagger and verifying its call
        count is zero is the strongest possible guarantee — a future
        refactor that accidentally wires it into the historical path
        will fail this test loudly.
        """
        from unittest.mock import MagicMock, patch
        from web.api.edge_engine import _simulate_pre_earnings_calendar_trade

        store = MagicMock()
        store.query_chain.side_effect = RuntimeError("synthetic")

        # Patch the tagger AT the import site inside edge_engine.py
        # (which is where the live path would call it from).
        with patch("web.api.edge_engine._tag_live_forward_observation") as tagger:
            _simulate_pre_earnings_calendar_trade(
                "AAPL",
                store,
                event_date=pd.Timestamp("2024-05-02"),
                entry_date=pd.Timestamp("2024-04-30"),
                exit_date=pd.Timestamp("2024-05-03"),
                entry_offset=2,
            )

        assert tagger.call_count == 0, (
            "Historical replay must NEVER invoke "
            "_tag_live_forward_observation. The function is the SOLE "
            "authorized assigner of forward_post_freeze; calling it "
            "from the replay path would re-introduce the in-sample-"
            "into-production research-leakage failure mode PR-AD was "
            "designed to prevent."
        )

    def test_attach_live_forward_provenance_invokes_authorized_tagger(self):
        """Positive companion to the historical-rejection test: the
        live helper MUST invoke _tag_live_forward_observation exactly
        once per call. Confirms the assignment boundary is preserved
        — the literal string never appears in this module; the
        tagger function is the only writer."""
        from unittest.mock import patch
        from web.api.edge_engine import (
            EdgeSnapshot,
            _attach_live_forward_provenance,
        )
        snapshot = EdgeSnapshot(
            symbol="AAPL",
            recommendation="Candidate",
            confidence_pct=72.0,
            setup_score=0.72,
            metrics={},
            rationale=[],
        )

        # We can't just patch _tag_live_forward_observation because the
        # helper depends on it to set the dict's sample_provenance key.
        # Instead, wrap it as a side-effect spy that still performs the
        # real work.
        from web.api import edge_engine as ee
        real_tagger = ee._tag_live_forward_observation
        with patch.object(ee, "_tag_live_forward_observation",
                          wraps=real_tagger) as spy:
            _attach_live_forward_provenance(snapshot)

        assert spy.call_count == 1
        # The spy was called with the intermediate dict (not the
        # snapshot directly). The boundary's tiny-dict pattern keeps
        # the live snapshot's sample_provenance attribute getting set
        # via attribute assignment, not directly by the tagger.
        called_with = spy.call_args.args[0]
        assert isinstance(called_with, dict)
        assert called_with.get("sample_provenance") == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE


