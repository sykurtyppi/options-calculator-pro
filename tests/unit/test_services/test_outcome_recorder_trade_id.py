"""Tests for the `make_trade_id` format contract in services.outcome_recorder.

PR-L: the legacy 3-field format ``{SYMBOL}|{ENTRY}|{STRUCTURE}`` collided when
two recommendations existed for the same symbol/structure on the same entry
date around different earnings events. The new 4-field format includes
earnings_date to disambiguate; the legacy form is retained as a fallback so
existing trade_ids in storage remain addressable.
"""
from __future__ import annotations

from datetime import date

from services.outcome_recorder import make_trade_id


class TestLegacyFormat:
    """Calling without earnings_date preserves the original 3-field shape."""

    def test_legacy_format_without_earnings_date(self) -> None:
        tid = make_trade_id("AAPL", date(2026, 5, 1), "atm_straddle")
        assert tid == "AAPL|2026-05-01|atm_straddle"

    def test_legacy_format_uppercases_symbol(self) -> None:
        tid = make_trade_id("aapl", date(2026, 5, 1), "atm_straddle")
        assert tid == "AAPL|2026-05-01|atm_straddle"


class TestNewFormatWithEarningsDate:
    def test_new_format_includes_earnings_date(self) -> None:
        tid = make_trade_id(
            "AAPL",
            date(2026, 5, 1),
            "atm_straddle",
            earnings_date=date(2026, 5, 7),
        )
        assert tid == "AAPL|2026-05-01|2026-05-07|atm_straddle"

    def test_different_earnings_dates_produce_different_ids(self) -> None:
        """The whole point: same (symbol, entry_date, structure) but
        different earnings events no longer collide."""
        tid_a = make_trade_id(
            "AAPL", date(2026, 5, 1), "atm_straddle", earnings_date=date(2026, 5, 7)
        )
        tid_b = make_trade_id(
            "AAPL", date(2026, 5, 1), "atm_straddle", earnings_date=date(2026, 8, 5)
        )
        assert tid_a != tid_b

    def test_same_inputs_produce_same_id(self) -> None:
        """Determinism: identical inputs must yield identical IDs (so
        INSERT OR IGNORE dedup works on re-runs)."""
        tid_a = make_trade_id(
            "AAPL", date(2026, 5, 1), "atm_straddle", earnings_date=date(2026, 5, 7)
        )
        tid_b = make_trade_id(
            "AAPL", date(2026, 5, 1), "atm_straddle", earnings_date=date(2026, 5, 7)
        )
        assert tid_a == tid_b


class TestLegacyAndNewFormatsDoNotCollide:
    """Critical invariant for the read-side fallback strategy: a legacy
    3-field id can never accidentally equal a new 4-field id."""

    def test_legacy_and_new_form_are_distinct(self) -> None:
        legacy = make_trade_id("AAPL", date(2026, 5, 1), "atm_straddle")
        new = make_trade_id(
            "AAPL", date(2026, 5, 1), "atm_straddle", earnings_date=date(2026, 5, 7)
        )
        assert legacy != new
        assert legacy.count("|") == 2  # 3 fields → 2 separators
        assert new.count("|") == 3  # 4 fields → 3 separators


class TestRecordTradeEntryUsesEarningsDate:
    """record_trade_entry should auto-include earnings_date in the
    generated trade_id when it's available in kwargs."""

    def test_record_trade_entry_threads_earnings_date_into_trade_id(self, tmp_path) -> None:
        from services.outcome_recorder import OutcomeStore, record_trade_entry

        store = OutcomeStore(store_path=tmp_path / "outcomes.sqlite")
        trade_id = record_trade_entry(
            symbol="AAPL",
            structure="atm_straddle",
            entry_date=date(2026, 5, 1),
            setup_score=0.7,
            source_type="paper",
            earnings_date=date(2026, 5, 7),
            store=store,
        )
        assert trade_id == "AAPL|2026-05-01|2026-05-07|atm_straddle"

    def test_record_trade_entry_falls_back_to_legacy_without_earnings_date(
        self, tmp_path
    ) -> None:
        from services.outcome_recorder import OutcomeStore, record_trade_entry

        store = OutcomeStore(store_path=tmp_path / "outcomes.sqlite")
        trade_id = record_trade_entry(
            symbol="AAPL",
            structure="atm_straddle",
            entry_date=date(2026, 5, 1),
            setup_score=0.7,
            source_type="paper",
            store=store,
        )
        assert trade_id == "AAPL|2026-05-01|atm_straddle"

    def test_two_recommendations_same_day_different_earnings_dont_collide(
        self, tmp_path
    ) -> None:
        """The bug PR-L closes: two recommendations on the same
        (symbol, entry_date, structure) for different earnings events used
        to collide. They must produce distinct trade_ids now."""
        from services.outcome_recorder import OutcomeStore, record_trade_entry

        store = OutcomeStore(store_path=tmp_path / "outcomes.sqlite")
        common = dict(
            symbol="AAPL",
            structure="atm_straddle",
            entry_date=date(2026, 5, 1),
            setup_score=0.7,
            source_type="paper",
            store=store,
        )
        first = record_trade_entry(**common, earnings_date=date(2026, 5, 7))
        second = record_trade_entry(**common, earnings_date=date(2026, 8, 5))
        assert first != second
        # Both rows should land in the store — no collision-induced merge.
        assert store.count() == 2
