"""Tests for services/calendar_leg_picker.py."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from services.calendar_leg_picker import (
    CANDIDATE_FRONT_MIN_DTE_DAYS,
    DEFAULT_BACK_GAP_DAYS,
    LEG_FIELDS,
    PICKER_CANDIDATE,
    PICKER_LEGACY,
    SIDE_CALL,
    SIDE_PUT,
    CalendarSelection,
    select_calendar_contracts,
)


def _chain(rows, *, underlying_price: float | None = None) -> pd.DataFrame:
    """Build a small synthetic chain from (expiry, strike, call_put, mid, iv) tuples."""
    df = pd.DataFrame(rows, columns=["expiry", "strike", "call_put", "mid", "iv"])
    if underlying_price is not None:
        df["underlying_price"] = underlying_price
    return df


# ──────────────────────────────────────────────────────────────────────────
# Picker variant — legacy
# ──────────────────────────────────────────────────────────────────────────

class TestPickerLegacy:
    def test_picks_first_expiry_strictly_after_event(self):
        event = date(2024, 5, 1)
        chain = _chain([
            # Short-DTE front (2 days)
            (date(2024, 5, 3), 100.0, "C", 1.0, 0.30),
            # Mid (16 days)
            (date(2024, 5, 17), 100.0, "C", 2.0, 0.30),
            # Far (51 days)
            (date(2024, 6, 21), 100.0, "C", 3.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_LEGACY,
        )
        assert result is not None
        assert result.front_expiry == date(2024, 5, 3)
        assert result.front_dte_days == 2
        assert result.picker_variant == PICKER_LEGACY

    def test_legacy_skips_event_day_itself(self):
        event = date(2024, 5, 1)
        chain = _chain([
            # An expiry ON the event day must be rejected — "strictly after"
            (date(2024, 5, 1), 100.0, "C", 1.0, 0.30),
            (date(2024, 5, 17), 100.0, "C", 2.0, 0.30),
            (date(2024, 6, 21), 100.0, "C", 3.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_LEGACY,
        )
        assert result is not None
        assert result.front_expiry == date(2024, 5, 17)


# ──────────────────────────────────────────────────────────────────────────
# Picker variant — candidate (≥14 DTE)
# ──────────────────────────────────────────────────────────────────────────

class TestPickerCandidate:
    def test_skips_short_dte_front_uses_default_floor(self):
        event = date(2024, 5, 1)
        chain = _chain([
            (date(2024, 5, 3), 100.0, "C", 1.0, 0.30),   # 2 DTE — skip
            (date(2024, 5, 17), 100.0, "C", 2.0, 0.30),  # 16 DTE — pick
            (date(2024, 6, 21), 100.0, "C", 3.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        assert result.front_expiry == date(2024, 5, 17)
        assert result.front_dte_days == 16
        assert result.picker_min_front_dte_days == CANDIDATE_FRONT_MIN_DTE_DAYS

    def test_exact_14_dte_is_accepted(self):
        event = date(2024, 5, 1)
        chain = _chain([
            (date(2024, 5, 15), 100.0, "C", 1.0, 0.30),  # exactly 14 DTE — accept
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        assert result.front_expiry == date(2024, 5, 15)

    def test_custom_min_dte_overrides_default(self):
        event = date(2024, 5, 1)
        chain = _chain([
            (date(2024, 5, 5), 100.0, "C", 1.0, 0.30),  # 4 DTE
            (date(2024, 5, 17), 100.0, "C", 2.0, 0.30),
            (date(2024, 6, 21), 100.0, "C", 3.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL,
            picker_variant=PICKER_CANDIDATE, min_front_dte_days=4,
        )
        assert result is not None
        assert result.front_expiry == date(2024, 5, 5)
        assert result.picker_min_front_dte_days == 4

    def test_returns_none_when_no_expiry_meets_floor(self):
        event = date(2024, 5, 1)
        chain = _chain([
            (date(2024, 5, 3), 100.0, "C", 1.0, 0.30),
            (date(2024, 5, 10), 100.0, "C", 1.5, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────
# Call/put symmetry — both sides pick consistently
# ──────────────────────────────────────────────────────────────────────────

class TestSideSymmetry:
    def test_calls_and_puts_pick_same_expiries_and_strike(self):
        event = date(2024, 5, 1)
        rows = []
        for expiry in [date(2024, 5, 17), date(2024, 6, 21)]:
            for strike in [95.0, 100.0, 105.0]:
                for cp in ["C", "P"]:
                    rows.append((expiry, strike, cp, 1.5, 0.30))
        chain = _chain(rows, underlying_price=100.0)

        result_c = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        result_p = select_calendar_contracts(
            chain, event_date=event, side=SIDE_PUT, picker_variant=PICKER_CANDIDATE,
        )
        assert result_c is not None and result_p is not None
        assert result_c.front_expiry == result_p.front_expiry
        assert result_c.back_expiry == result_p.back_expiry
        assert result_c.strike == result_p.strike
        assert result_c.side == "call"
        assert result_p.side == "put"

    def test_put_side_filters_out_calls_from_chain(self):
        """Pure correctness regression: if chain has calls only, picking
        side='put' must return None — not silently fall through to calls."""
        event = date(2024, 5, 1)
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.5, 0.30),
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_PUT, picker_variant=PICKER_CANDIDATE,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────
# Strike picking
# ──────────────────────────────────────────────────────────────────────────

class TestStrikePicking:
    def test_picks_strike_closest_to_underlying(self):
        event = date(2024, 5, 1)
        rows = []
        for expiry in [date(2024, 5, 17), date(2024, 6, 21)]:
            for strike in [90.0, 100.0, 110.0]:
                rows.append((expiry, strike, "C", 1.0, 0.30))
        chain = _chain(rows, underlying_price=102.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        assert result.strike == 100.0

    def test_falls_back_to_median_strike_without_underlying(self):
        event = date(2024, 5, 1)
        rows = []
        for expiry in [date(2024, 5, 17), date(2024, 6, 21)]:
            for strike in [95.0, 100.0, 105.0]:
                rows.append((expiry, strike, "C", 1.0, 0.30))
        chain = _chain(rows)  # no underlying_price column
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        assert result.strike == 100.0

    def test_strike_must_exist_at_both_expiries(self):
        event = date(2024, 5, 1)
        # 100 exists only at front; 105 exists only at back → no common strike
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.0, 0.30),
            (date(2024, 6, 21), 105.0, "C", 1.5, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────
# Back expiry gap
# ──────────────────────────────────────────────────────────────────────────

class TestBackExpiry:
    def test_back_gap_minimum_enforced(self):
        event = date(2024, 5, 1)
        # front = 5/17, back candidates: 5/24 (7d) and 6/14 (28d)
        rows = []
        for expiry in [date(2024, 5, 17), date(2024, 5, 24), date(2024, 6, 14)]:
            rows.append((expiry, 100.0, "C", 1.0, 0.30))
        chain = _chain(rows, underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL,
            picker_variant=PICKER_CANDIDATE, back_gap_days=14,
        )
        assert result is not None
        assert result.back_expiry == date(2024, 6, 14)
        assert result.back_minus_front_days == 28

    def test_back_falls_back_when_no_expiry_meets_gap(self):
        """If no expiry meets back_gap_days, fall back to the next available."""
        event = date(2024, 5, 1)
        rows = []
        for expiry in [date(2024, 5, 17), date(2024, 5, 24)]:
            rows.append((expiry, 100.0, "C", 1.0, 0.30))
        chain = _chain(rows, underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL,
            picker_variant=PICKER_CANDIDATE, back_gap_days=14,
        )
        assert result is not None
        assert result.back_expiry == date(2024, 5, 24)
        assert result.back_minus_front_days == 7


# ──────────────────────────────────────────────────────────────────────────
# No-result paths
# ──────────────────────────────────────────────────────────────────────────

class TestNoResultPaths:
    def test_empty_chain_returns_none(self):
        chain = pd.DataFrame(columns=["expiry", "strike", "call_put", "mid"])
        result = select_calendar_contracts(
            chain, event_date=date(2024, 5, 1), side=SIDE_CALL,
            picker_variant=PICKER_CANDIDATE,
        )
        assert result is None

    def test_missing_required_columns_returns_none(self):
        chain = pd.DataFrame([{"expiry": date(2024, 5, 17), "strike": 100.0}])
        result = select_calendar_contracts(
            chain, event_date=date(2024, 5, 1), side=SIDE_CALL,
            picker_variant=PICKER_CANDIDATE,
        )
        assert result is None

    def test_only_one_expiry_returns_none(self):
        event = date(2024, 5, 1)
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is None

    def test_zero_mid_filtered_out(self):
        """Rows with mid <= 0 should not contribute to expiry/strike sets."""
        event = date(2024, 5, 1)
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 0.0, 0.30),   # zero mid — filter
            (date(2024, 5, 17), 105.0, "C", 0.0, 0.30),   # zero mid — filter
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        # Only 2024-06-21 has usable mids, so no calendar pair
        assert result is None


# ──────────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_invalid_side_raises(self):
        chain = _chain([(date(2024, 5, 17), 100.0, "C", 1.0, 0.30)], underlying_price=100.0)
        with pytest.raises(ValueError, match="side"):
            select_calendar_contracts(
                chain, event_date=date(2024, 5, 1), side="straddle",
                picker_variant=PICKER_CANDIDATE,
            )

    def test_invalid_picker_raises(self):
        chain = _chain([(date(2024, 5, 17), 100.0, "C", 1.0, 0.30)], underlying_price=100.0)
        with pytest.raises(ValueError, match="picker_variant"):
            select_calendar_contracts(
                chain, event_date=date(2024, 5, 1), side=SIDE_CALL,
                picker_variant="optimized_v3",
            )

    def test_negative_min_dte_raises(self):
        chain = _chain([(date(2024, 5, 17), 100.0, "C", 1.0, 0.30)], underlying_price=100.0)
        with pytest.raises(ValueError, match="min_front_dte_days"):
            select_calendar_contracts(
                chain, event_date=date(2024, 5, 1), side=SIDE_CALL,
                picker_variant=PICKER_CANDIDATE, min_front_dte_days=-1,
            )

    def test_negative_back_gap_raises(self):
        chain = _chain([(date(2024, 5, 17), 100.0, "C", 1.0, 0.30)], underlying_price=100.0)
        with pytest.raises(ValueError, match="back_gap_days"):
            select_calendar_contracts(
                chain, event_date=date(2024, 5, 1), side=SIDE_CALL,
                picker_variant=PICKER_CANDIDATE, back_gap_days=-1,
            )


# ──────────────────────────────────────────────────────────────────────────
# Picker metadata — provenance preserved on output
# ──────────────────────────────────────────────────────────────────────────

class TestPickerMetadata:
    def test_metadata_preserved_in_output(self):
        event = date(2024, 5, 1)
        rows = []
        for expiry in [date(2024, 5, 17), date(2024, 6, 21)]:
            rows.append((expiry, 100.0, "C", 1.0, 0.30))
        chain = _chain(rows, underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL,
            picker_variant=PICKER_CANDIDATE, min_front_dte_days=10, back_gap_days=20,
        )
        assert result is not None
        assert result.picker_variant == PICKER_CANDIDATE
        assert result.picker_min_front_dte_days == 10
        assert result.picker_back_gap_days == 20
        assert result.front_dte_days == 16
        assert result.back_minus_front_days == 35

    def test_legacy_metadata_records_default_min_dte_unused(self):
        """Even when the legacy picker doesn't use min_front_dte_days, the
        metadata still records what was passed — so downstream auditing
        can tell the picker variant authoritatively from the variant
        field, not from the value."""
        event = date(2024, 5, 1)
        rows = [
            (date(2024, 5, 3), 100.0, "C", 1.0, 0.30),
            (date(2024, 5, 17), 100.0, "C", 2.0, 0.30),
        ]
        chain = _chain(rows, underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_LEGACY,
        )
        assert result is not None
        assert result.picker_variant == PICKER_LEGACY
        assert result.front_expiry == date(2024, 5, 3)


# ──────────────────────────────────────────────────────────────────────────
# Legacy vs candidate side-by-side
# ──────────────────────────────────────────────────────────────────────────

class TestLegacyVsCandidate:
    def test_pickers_diverge_when_close_in_weekly_exists(self):
        """The whole point of the dual-path: different fronts, different
        trade construction, different observations to record."""
        event = date(2024, 5, 1)
        rows = []
        for expiry in [date(2024, 5, 3), date(2024, 5, 17), date(2024, 6, 21)]:
            rows.append((expiry, 100.0, "C", 1.0, 0.30))
        chain = _chain(rows, underlying_price=100.0)

        legacy = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_LEGACY,
        )
        candidate = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert legacy is not None and candidate is not None
        assert legacy.front_expiry == date(2024, 5, 3)
        assert candidate.front_expiry == date(2024, 5, 17)
        assert legacy.front_expiry != candidate.front_expiry

    def test_pickers_agree_when_no_close_in_weekly(self):
        """When the chain has no short-DTE expiries, both pickers pick the
        same front. The dual-path is still safe to record — the variant
        field disambiguates."""
        event = date(2024, 5, 1)
        rows = []
        for expiry in [date(2024, 5, 17), date(2024, 6, 21)]:
            rows.append((expiry, 100.0, "C", 1.0, 0.30))
        chain = _chain(rows, underlying_price=100.0)

        legacy = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_LEGACY,
        )
        candidate = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert legacy is not None and candidate is not None
        assert legacy.front_expiry == candidate.front_expiry
        assert legacy.picker_variant != candidate.picker_variant


# ──────────────────────────────────────────────────────────────────────────
# Tie-breaking when multiple rows share (expiry, strike, side)
# ──────────────────────────────────────────────────────────────────────────

class TestBestRowTieBreaking:
    def test_picks_more_liquid_row_on_duplicate(self):
        event = date(2024, 5, 1)
        # Two front-row candidates at 100C/5-17: one with high OI, one low
        chain = pd.DataFrame([
            {"expiry": date(2024, 5, 17), "strike": 100.0, "call_put": "C", "mid": 2.0,
             "iv": 0.30, "open_interest": 10, "volume": 1, "spread_pct": 8.0},
            {"expiry": date(2024, 5, 17), "strike": 100.0, "call_put": "C", "mid": 2.0,
             "iv": 0.30, "open_interest": 5000, "volume": 200, "spread_pct": 1.5},
            {"expiry": date(2024, 6, 21), "strike": 100.0, "call_put": "C", "mid": 3.0,
             "iv": 0.30, "open_interest": 500, "volume": 50, "spread_pct": 2.0},
        ])
        chain["underlying_price"] = 100.0
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        assert result.front_row["open_interest"] == 5000

    def test_duplicate_rows_without_optional_liquidity_columns(self):
        """Regression for the Codex review finding: when duplicate rows exist
        but open_interest/volume/spread_pct columns are absent, the tie-breaker
        must not crash. Previous implementation used s.get('col', 0) which
        returns a scalar when the column is missing, then called .fillna()
        on it (AttributeError on scalar)."""
        event = date(2024, 5, 1)
        chain = pd.DataFrame([
            # Two identical rows, no OI/volume/spread columns at all
            {"expiry": date(2024, 5, 17), "strike": 100.0, "call_put": "C",
             "mid": 2.0, "iv": 0.30},
            {"expiry": date(2024, 5, 17), "strike": 100.0, "call_put": "C",
             "mid": 2.0, "iv": 0.30},
            {"expiry": date(2024, 6, 21), "strike": 100.0, "call_put": "C",
             "mid": 3.0, "iv": 0.30},
        ])
        chain["underlying_price"] = 100.0
        # Should NOT raise AttributeError
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        # First-row stable: both are equivalent, picker takes the first
        assert result.front_expiry == date(2024, 5, 17)

    def test_duplicate_rows_with_partial_optional_columns(self):
        """Mixed case: open_interest column present, volume + spread_pct absent."""
        event = date(2024, 5, 1)
        chain = pd.DataFrame([
            {"expiry": date(2024, 5, 17), "strike": 100.0, "call_put": "C",
             "mid": 2.0, "iv": 0.30, "open_interest": 100},
            {"expiry": date(2024, 5, 17), "strike": 100.0, "call_put": "C",
             "mid": 2.0, "iv": 0.30, "open_interest": 9999},
            {"expiry": date(2024, 6, 21), "strike": 100.0, "call_put": "C",
             "mid": 3.0, "iv": 0.30, "open_interest": 500},
        ])
        chain["underlying_price"] = 100.0
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        # Should still prefer the more liquid duplicate
        assert result.front_row["open_interest"] == 9999


# ──────────────────────────────────────────────────────────────────────────
# Serialization adapter (for ledger / API surfaces in commits 3-4)
# ──────────────────────────────────────────────────────────────────────────

class TestToMetadataDict:
    def test_returns_json_safe_scalars_only(self):
        event = date(2024, 5, 1)
        chain = pd.DataFrame([
            {"expiry": date(2024, 5, 17), "strike": 100.0, "call_put": "C",
             "mid": 2.0, "iv": 0.30, "bid": 1.95, "ask": 2.05,
             "open_interest": 500, "volume": 50, "spread_pct": 5.0},
            {"expiry": date(2024, 6, 21), "strike": 100.0, "call_put": "C",
             "mid": 3.0, "iv": 0.30, "bid": 2.95, "ask": 3.05,
             "open_interest": 800, "volume": 80, "spread_pct": 3.3},
        ])
        chain["underlying_price"] = 100.0
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        meta = result.to_metadata_dict()

        # ISO-format dates, not date objects
        assert isinstance(meta["front_expiry"], str)
        assert meta["front_expiry"] == "2024-05-17"
        assert meta["back_expiry"] == "2024-06-21"

        # Picker provenance preserved
        assert meta["picker_variant"] == PICKER_CANDIDATE
        assert meta["picker_min_front_dte_days"] == CANDIDATE_FRONT_MIN_DTE_DAYS
        assert meta["picker_back_gap_days"] == DEFAULT_BACK_GAP_DAYS

        # Leg fields are simple floats
        assert meta["front_leg"]["mid"] == 2.0
        assert meta["front_leg"]["bid"] == 1.95
        assert meta["front_leg"]["ask"] == 2.05
        assert meta["back_leg"]["mid"] == 3.0

        # Must be JSON-safe — no pd.Series, no numpy scalars that break json
        import json
        json.dumps(meta)  # would raise if not serializable

    def test_stable_shape_when_optional_leg_fields_missing(self):
        """Every LEG_FIELDS key must always be present in the serialized
        leg dict, with value None when the source row didn't carry the
        column. Stable shape simplifies downstream consumers (ledger
        schema, frontend types) which can rely on a fixed key set."""
        event = date(2024, 5, 1)
        chain = pd.DataFrame([
            {"expiry": date(2024, 5, 17), "strike": 100.0, "call_put": "C",
             "mid": 2.0, "iv": 0.30},
            {"expiry": date(2024, 6, 21), "strike": 100.0, "call_put": "C",
             "mid": 3.0, "iv": 0.30},
        ])
        chain["underlying_price"] = 100.0
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        meta = result.to_metadata_dict()
        # Present fields keep their values
        assert meta["front_leg"]["mid"] == 2.0
        assert meta["front_leg"]["iv"] == 0.30
        # Absent fields are present as None — not omitted
        assert meta["front_leg"]["bid"] is None
        assert meta["front_leg"]["ask"] is None
        assert meta["front_leg"]["spread_pct"] is None
        assert meta["front_leg"]["open_interest"] is None
        assert meta["front_leg"]["volume"] is None
        # Stable shape — exact key set matches LEG_FIELDS
        assert set(meta["front_leg"].keys()) == set(LEG_FIELDS)
        assert set(meta["back_leg"].keys()) == set(LEG_FIELDS)
        import json
        json.dumps(meta)


# ──────────────────────────────────────────────────────────────────────────
# Date coercion robustness
# ──────────────────────────────────────────────────────────────────────────

class TestDateCoercion:
    def test_pandas_timestamp_event_date(self):
        event = pd.Timestamp("2024-05-01")
        chain = _chain([
            (date(2024, 5, 17), 100.0, "C", 1.0, 0.30),
            (date(2024, 6, 21), 100.0, "C", 2.0, 0.30),
        ], underlying_price=100.0)
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        assert result.front_dte_days == 16

    def test_pandas_timestamp_expiries_in_chain(self):
        event = date(2024, 5, 1)
        chain = pd.DataFrame([
            {"expiry": pd.Timestamp("2024-05-17"), "strike": 100.0, "call_put": "C", "mid": 1.0, "iv": 0.30},
            {"expiry": pd.Timestamp("2024-06-21"), "strike": 100.0, "call_put": "C", "mid": 2.0, "iv": 0.30},
        ])
        chain["underlying_price"] = 100.0
        result = select_calendar_contracts(
            chain, event_date=event, side=SIDE_CALL, picker_variant=PICKER_CANDIDATE,
        )
        assert result is not None
        assert result.front_expiry == date(2024, 5, 17)
