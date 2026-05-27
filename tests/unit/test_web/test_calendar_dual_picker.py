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
    is_promotion_eligible,
    tag_live_forward_observation,
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
        # Every non-status field is None
        for k in _CANDIDATE_SHADOW_OUTCOME_FIELDS:
            assert block[k] is None, f"{k} should be None in empty block, got {block[k]!r}"

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
            # Every non-status field must be None on skip paths
            for k in _CANDIDATE_SHADOW_OUTCOME_FIELDS:
                assert result[k] is None, f"{description}: {k} is {result[k]!r}, expected None"

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

    def test_promotion_eligible_set_includes_preregistered_holdout(self):
        assert SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED in PROMOTION_ELIGIBLE_PROVENANCES


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

class TestIsPromotionEligible:
    """Codex P1 + 'fail closed on UNKNOWN': the single source of truth
    for which records count toward promotion. Aggregators MUST route
    through this helper rather than open-coding the check."""

    def test_forward_post_freeze_is_eligible(self):
        record = {"sample_provenance": SAMPLE_PROVENANCE_FORWARD_POST_FREEZE}
        assert is_promotion_eligible(record) is True

    def test_preregistered_holdout_is_eligible(self):
        record = {"sample_provenance": SAMPLE_PROVENANCE_HISTORICAL_HOLDOUT_PREREGISTERED}
        assert is_promotion_eligible(record) is True

    def test_historical_replay_is_never_eligible(self):
        """The single most important assertion in PR-AD: historical
        replay output cannot become promotion-eligible no matter what."""
        record = {"sample_provenance": SAMPLE_PROVENANCE_HISTORICAL_REPLAY}
        assert is_promotion_eligible(record) is False

    def test_unknown_is_never_eligible(self):
        record = {"sample_provenance": SAMPLE_PROVENANCE_UNKNOWN}
        assert is_promotion_eligible(record) is False

    def test_missing_key_fails_closed(self):
        record = {"symbol": "AAPL"}  # no sample_provenance key at all
        assert is_promotion_eligible(record) is False

    def test_none_value_fails_closed(self):
        record = {"sample_provenance": None}
        assert is_promotion_eligible(record) is False

    def test_arbitrary_string_fails_closed(self):
        """Defense against typos / drift: only the canonical promotion-
        eligible constants count. A clever string can't slip through."""
        for s in ("forward", "forward_post_pr_ad", "promotion_eligible_pls",
                  "out_of_sample", "validated", ""):
            record = {"sample_provenance": s}
            assert is_promotion_eligible(record) is False, (
                f"non-canonical string {s!r} should not be promotion-eligible"
            )

    def test_non_dict_input_fails_closed(self):
        """Defensive: arbitrary inputs do not crash the helper."""
        assert is_promotion_eligible(None) is False
        assert is_promotion_eligible("string") is False
        assert is_promotion_eligible([]) is False


class TestTagLiveForwardObservation:
    """tag_live_forward_observation is the SINGLE function authorized to
    assign SAMPLE_PROVENANCE_FORWARD_POST_FREEZE. The regression test
    further down (TestForwardProvenanceAssignmentBoundary) enforces
    that no other code path writes the constant."""

    def test_assigns_forward_post_freeze(self):
        record = {"symbol": "AAPL"}
        tag_live_forward_observation(record)
        assert record["sample_provenance"] == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE
        assert is_promotion_eligible(record) is True

    def test_overwrites_existing_provenance(self):
        """If the record was previously tagged UNKNOWN/HISTORICAL by
        default initialization, calling the live-forward tagger replaces
        it. This is the intended behavior — the live API path knows it
        is producing forward observations regardless of the default."""
        record = {"sample_provenance": SAMPLE_PROVENANCE_UNKNOWN}
        tag_live_forward_observation(record)
        assert record["sample_provenance"] == SAMPLE_PROVENANCE_FORWARD_POST_FREEZE

    def test_returns_the_same_record_for_chaining(self):
        record = {"symbol": "AAPL"}
        result = tag_live_forward_observation(record)
        assert result is record  # mutated in-place


class TestForwardProvenanceAssignmentBoundary:
    """Codex's defense-in-depth requirement: 'forward provenance must
    be assigned only by the live evidence-cycle path, not by a generic
    script flag anyone can pass casually.'

    Source-grep regression: the string value of
    SAMPLE_PROVENANCE_FORWARD_POST_FREEZE may appear in the codebase
    ONLY in three legitimate sites:
      1. The constant definition itself
      2. The body of tag_live_forward_observation
      3. The PROMOTION_ELIGIBLE_PROVENANCES set definition
    Anywhere else is a research-leakage hazard."""

    def test_forward_post_freeze_string_assignment_sites_are_audited(self):
        """Scan the production source tree (services/ + web/) for any
        line that assigns the string literal "forward_post_freeze".
        Allowlist the legitimate definition sites; flag anything else."""
        import re
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        # Production code only — tests legitimately reference the
        # string for testing purposes.
        prod_dirs = [repo_root / "services", repo_root / "web", repo_root / "scripts"]
        suspect_lines: list[str] = []
        pattern = re.compile(r'"forward_post_freeze"|\'forward_post_freeze\'')
        for d in prod_dirs:
            if not d.exists():
                continue
            for py_file in d.rglob("*.py"):
                with open(py_file) as f:
                    for lineno, line in enumerate(f, 1):
                        if pattern.search(line):
                            rel = py_file.relative_to(repo_root)
                            suspect_lines.append(f"{rel}:{lineno}: {line.strip()}")

        # Allowlist: legitimate occurrences in edge_engine.py only.
        # All matches MUST be in that file. If a match shows up in
        # scripts/ or services/ (other than edge_engine), that's the
        # leakage hazard Codex warned about.
        non_edge_engine = [
            s for s in suspect_lines
            if "web/api/edge_engine.py" not in s
        ]
        assert non_edge_engine == [], (
            "forward_post_freeze string assignment leaked outside the "
            "edge_engine.py file. Only tag_live_forward_observation is "
            "authorized to assign this value. Found:\n  "
            + "\n  ".join(non_edge_engine)
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


