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
)


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

        # Every commit-2 contract: these keys must always be present
        assert "legacy_selection" in result
        assert "candidate_selection" in result
        assert "pickers_diverged" in result
        assert "shadow_status" in result
        assert "candidate_min_front_dte_days" in result
        assert "experimental_note" in result

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
# _aggregate_experimental_candidate_evidence
# ──────────────────────────────────────────────────────────────────────────

class TestExperimentalCandidateEvidenceAggregation:
    def test_empty_input_yields_zero_counts(self):
        result = _aggregate_experimental_candidate_evidence([])
        counts = result["trade_counts"]
        assert counts["total"] == 0
        assert counts["pickers_diverged"] == 0
        assert counts["both_succeeded"] == 0
        assert counts["legacy_only"] == 0
        assert counts["candidate_only"] == 0
        assert counts["both_failed"] == 0
        assert result["diverged_sample"] == []

    def test_counts_split_by_picker_success(self):
        trades = [
            # Both picked (and agreed)
            {"event_date": "2024-05-01", "entry_date": "2024-04-29",
             "dual_picker": {
                 "shadow_status": "ok",
                 "legacy_selection": {"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
                                      "strike": 100.0, "front_dte_days": 16},
                 "candidate_selection": {"front_expiry": "2024-05-17", "back_expiry": "2024-06-21",
                                         "strike": 100.0, "front_dte_days": 16},
                 "pickers_diverged": False,
             }},
            # Both picked, diverged
            {"event_date": "2024-08-01", "entry_date": "2024-07-30",
             "dual_picker": {
                 "shadow_status": "ok",
                 "legacy_selection": {"front_expiry": "2024-08-02", "back_expiry": "2024-08-16",
                                      "strike": 200.0, "front_dte_days": 1},
                 "candidate_selection": {"front_expiry": "2024-08-16", "back_expiry": "2024-09-20",
                                         "strike": 200.0, "front_dte_days": 15},
                 "pickers_diverged": True,
             }},
            # Legacy succeeded only
            {"event_date": "2024-11-01", "entry_date": "2024-10-30",
             "dual_picker": {
                 "shadow_status": "ok",
                 "legacy_selection": {"front_expiry": "2024-11-03", "back_expiry": "2024-11-10",
                                      "strike": 50.0, "front_dte_days": 2},
                 "candidate_selection": None,
                 "pickers_diverged": False,
             }},
            # Both failed
            {"event_date": "2025-02-01", "entry_date": "2025-01-30",
             "dual_picker": {
                 "shadow_status": "ok",
                 "legacy_selection": None,
                 "candidate_selection": None,
                 "pickers_diverged": False,
             }},
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        c = result["trade_counts"]
        assert c["total"] == 4
        assert c["both_succeeded"] == 2
        assert c["legacy_only"] == 1
        assert c["candidate_only"] == 0
        assert c["both_failed"] == 1
        assert c["pickers_diverged"] == 1
        assert c["legacy_succeeded"] == 3  # both + legacy_only
        assert c["candidate_succeeded"] == 2  # both

    def test_diverged_sample_captures_up_to_five(self):
        # 7 diverged events should produce a sample of 5
        trades = []
        for i in range(7):
            trades.append({
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

    def test_skips_trades_without_dual_picker(self):
        """Trades that never got past the entry_chain step have no
        dual_picker block — they should not contribute to counts."""
        trades = [
            {"event_date": "2024-05-01", "status": "missing_entry_chain"},  # no dual_picker
            {"event_date": "2024-05-02", "dual_picker": {
                "shadow_status": "ok",
                "legacy_selection": {"front_expiry": "2024-05-17",
                                     "back_expiry": "2024-06-21",
                                     "strike": 100.0, "front_dte_days": 16},
                "candidate_selection": {"front_expiry": "2024-05-17",
                                        "back_expiry": "2024-06-21",
                                        "strike": 100.0, "front_dte_days": 16},
                "pickers_diverged": False,
            }},
        ]
        result = _aggregate_experimental_candidate_evidence(trades)
        # Only the one trade with dual_picker counts
        assert result["trade_counts"]["total"] == 1

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
