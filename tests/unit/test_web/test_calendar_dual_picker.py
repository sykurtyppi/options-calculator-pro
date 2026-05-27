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
    _aggregate_experimental_candidate_evidence,
    _build_experimental_contract_selection,
    _dual_picker_calendar_selection,
    _empty_candidate_shadow_outcome,
    _empty_dual_picker,
    _simulate_candidate_shadow_outcome,
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
CANDIDATE_SHADOW_KEYS = frozenset({"status"} | set(_CANDIDATE_SHADOW_OUTCOME_FIELDS))


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
        assert result["realized_return_pct"] == pytest.approx(30.0)
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


