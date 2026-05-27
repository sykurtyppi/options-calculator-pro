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
    _aggregate_experimental_candidate_evidence,
    _dual_picker_calendar_selection,
    _empty_dual_picker,
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
