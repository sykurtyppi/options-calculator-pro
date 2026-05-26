"""
Tests for the outcome feedback loop:
  - OutcomeStore (insert, update, finalize, deduplication)
  - StructurePriorStore (update, persistence, Laplace smoothing)
  - finalize_trade_and_update_learning() (end-to-end dispatch)
  - Replay seeding (idempotency, count accuracy)
  - Persistence (restart/reload)
  - Provenance separation (replay vs paper vs live)

All tests use tmp_path-scoped SQLite / JSON files so they never touch the
user's live ~/.options_calculator_pro directory.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import List, Optional

import pytest

from services.outcome_recorder import (
    OutcomeStore,
    finalize_trade_and_update_learning,
    make_trade_id,
    make_snapshot_hash,
)
from services.structure_prior_store import (
    LAPLACE_SMOOTHING_THRESHOLD,
    MIN_OBS_FOR_OVERRIDE,
    StructurePriorStore,
)
from services.calibration_service import IVExpansionCalibration


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "outcomes" / "outcome_store.sqlite"


@pytest.fixture()
def prior_path(tmp_path: Path) -> Path:
    return tmp_path / "priors" / "structure_priors.json"


@pytest.fixture()
def cal_path(tmp_path: Path) -> Path:
    return tmp_path / "calibration" / "iv_expansion.json"


@pytest.fixture()
def store(store_path: Path) -> OutcomeStore:
    return OutcomeStore(store_path=store_path)


@pytest.fixture()
def prior_store(prior_path: Path) -> StructurePriorStore:
    return StructurePriorStore(store_path=prior_path)


@pytest.fixture()
def calibration(cal_path: Path) -> IVExpansionCalibration:
    return IVExpansionCalibration(store_path=cal_path)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _entry_date(year: int = 2025, month: int = 1, day: int = 15) -> date:
    return date(year, month, day)


def _insert_minimal(
    store: OutcomeStore,
    *,
    symbol: str = "AAPL",
    structure: str = "atm_straddle",
    entry_date: Optional[date] = None,
    setup_score: float = 0.65,
    source_type: str = "paper",
    trade_id: Optional[str] = None,
) -> str:
    ed = entry_date or _entry_date()
    tid = trade_id or make_trade_id(symbol, ed, structure)
    store.insert_entry(
        trade_id=tid,
        symbol=symbol,
        structure=structure,
        entry_date=ed,
        setup_score=setup_score,
        source_type=source_type,
    )
    return tid


# ══════════════════════════════════════════════════════════════════════════════
# 1. Outcome store — insert and update
# ══════════════════════════════════════════════════════════════════════════════


class TestOutcomeStoreInsert:
    def test_insert_entry_succeeds(self, store: OutcomeStore) -> None:
        tid = _insert_minimal(store)
        row = store.get_trade(tid)
        assert row is not None
        assert row["symbol"] == "AAPL"
        assert row["structure"] == "atm_straddle"
        assert row["status"] == "open"

    def test_insert_entry_returns_true_on_new(self, store: OutcomeStore) -> None:
        ed = _entry_date()
        tid = make_trade_id("MSFT", ed, "otm_strangle")
        result = store.insert_entry(
            trade_id=tid,
            symbol="MSFT",
            structure="otm_strangle",
            entry_date=ed,
            setup_score=0.70,
            source_type="paper",
        )
        assert result is True

    def test_duplicate_insert_returns_false(self, store: OutcomeStore) -> None:
        ed = _entry_date()
        tid = make_trade_id("AAPL", ed, "atm_straddle")
        store.insert_entry(
            trade_id=tid,
            symbol="AAPL",
            structure="atm_straddle",
            entry_date=ed,
            setup_score=0.65,
            source_type="paper",
        )
        # Second insert with same trade_id — should return False, not raise.
        result = store.insert_entry(
            trade_id=tid,
            symbol="AAPL",
            structure="atm_straddle",
            entry_date=ed,
            setup_score=0.65,
            source_type="paper",
        )
        assert result is False

    def test_duplicate_does_not_corrupt_existing_row(self, store: OutcomeStore) -> None:
        ed = _entry_date()
        tid = make_trade_id("AAPL", ed, "atm_straddle")
        store.insert_entry(
            trade_id=tid,
            symbol="AAPL",
            structure="atm_straddle",
            entry_date=ed,
            setup_score=0.65,
            source_type="paper",
            notes="original",
        )
        # Try to overwrite with different notes — should be silently ignored.
        store.insert_entry(
            trade_id=tid,
            symbol="AAPL",
            structure="atm_straddle",
            entry_date=ed,
            setup_score=0.99,  # different score
            source_type="paper",
            notes="overwrite_attempt",
        )
        row = store.get_trade(tid)
        assert row["setup_score"] == pytest.approx(0.65)
        assert row["notes"] == "original"

    def test_total_count_after_inserts(self, store: OutcomeStore) -> None:
        for i in range(5):
            _insert_minimal(store, entry_date=date(2025, 1, i + 1))
        assert store.count() == 5

    def test_insert_entry_persists_recommendation_id(self, store: OutcomeStore) -> None:
        tid = make_trade_id("AAPL", _entry_date(), "atm_straddle")
        store.insert_entry(
            trade_id=tid,
            recommendation_id="rec_abc123",
            symbol="AAPL",
            structure="atm_straddle",
            entry_date=_entry_date(),
            setup_score=0.65,
            source_type="paper",
        )
        row = store.get_trade(tid)
        assert row is not None
        assert row["recommendation_id"] == "rec_abc123"

    def test_schema_migration_adds_recommendation_id(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(store_path)
        conn.execute(
            """
            CREATE TABLE outcome_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                structure TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                setup_score REAL NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'paper',
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

        migrated = OutcomeStore(store_path=store_path)
        columns = {
            row[1]
            for row in migrated._conn.execute("PRAGMA table_info(outcome_trades)").fetchall()  # noqa: SLF001
        }
        assert "recommendation_id" in columns


class TestOutcomeStoreUpdate:
    def test_update_exit_sets_exit_fields(self, store: OutcomeStore) -> None:
        tid = _insert_minimal(store)
        result = store.update_exit(
            trade_id=tid,
            exit_date=date(2025, 1, 20),
            exit_mid=5.50,
            realized_return_pct=9.0,
            realized_pnl=90.0,
            realized_expansion_pct=10.0,
        )
        assert result is True
        row = store.get_trade(tid)
        assert row["exit_date"] == "2025-01-20"
        assert row["exit_mid"] == pytest.approx(5.50)
        assert row["realized_return_pct"] == pytest.approx(9.0)
        assert row["realized_expansion_pct"] == pytest.approx(10.0)
        assert row["status"] == "exited"

    def test_update_exit_on_missing_trade_returns_false(self, store: OutcomeStore) -> None:
        result = store.update_exit(
            trade_id="nonexistent_id",
            exit_date=date(2025, 1, 20),
        )
        assert result is False

    def test_mark_finalized_transitions_status(self, store: OutcomeStore) -> None:
        tid = _insert_minimal(store)
        store.update_exit(trade_id=tid, exit_date=date(2025, 1, 20))
        store.mark_finalized(tid)
        row = store.get_trade(tid)
        assert row["status"] == "finalized"

    def test_mark_finalized_twice_is_safe(self, store: OutcomeStore) -> None:
        tid = _insert_minimal(store)
        store.mark_finalized(tid)
        result = store.mark_finalized(tid)
        # Second call returns False (no row changed), but does not raise.
        assert result is False


class TestOutcomeStoreQuery:
    def test_exists_returns_true_after_insert(self, store: OutcomeStore) -> None:
        tid = _insert_minimal(store)
        assert store.exists(tid) is True

    def test_exists_returns_false_for_unknown(self, store: OutcomeStore) -> None:
        assert store.exists("NEVER_INSERTED") is False

    def test_count_by_source_type(self, store: OutcomeStore) -> None:
        _insert_minimal(store, symbol="AAPL", entry_date=date(2025, 1, 1), source_type="replay")
        _insert_minimal(store, symbol="MSFT", entry_date=date(2025, 1, 2), source_type="paper")
        _insert_minimal(store, symbol="GOOGL", entry_date=date(2025, 1, 3), source_type="live")
        counts = store.count_by_source()
        assert counts["replay"] == 1
        assert counts["paper"] == 1
        assert counts["live"] == 1

    def test_diagnostics_returns_expected_shape(self, store: OutcomeStore) -> None:
        _insert_minimal(store)
        diag = store.diagnostics()
        assert "total" in diag
        assert "by_source" in diag
        assert "by_status" in diag
        assert "by_structure" in diag
        assert "store_path" in diag


# ══════════════════════════════════════════════════════════════════════════════
# 2. Calibration feedback
# ══════════════════════════════════════════════════════════════════════════════


class TestCalibrationFeedback:
    def test_finalized_trade_updates_calibration_persistence(
        self,
        store_path: Path,
        cal_path: Path,
        prior_path: Path,
    ) -> None:
        """
        finalize_trade_and_update_learning() should append a calibration
        observation and persist it to disk.
        """
        store = OutcomeStore(store_path=store_path)
        cal_before = IVExpansionCalibration(store_path=cal_path)
        n_before = cal_before._n()

        tid = _insert_minimal(store, source_type="paper")
        # Inject isolated calibration and prior store instances via module-level singletons reset.
        # We exercise the function with real singletons but point them at tmp paths.
        import services.outcome_recorder as _or
        import services.calibration_service as _cs
        import services.structure_prior_store as _ps
        import services.structure_scorecard as _sc

        # Temporarily replace singletons for this test.
        orig_store = _or._store
        orig_cal = _cs._calibration
        orig_prior = _ps._store

        _or._store = store
        _cs._calibration = IVExpansionCalibration(store_path=cal_path)
        _ps._store = StructurePriorStore(store_path=prior_path)

        try:
            result = finalize_trade_and_update_learning(
                trade_id=tid,
                exit_date=date(2025, 1, 20),
                realized_return_pct=9.0,
                realized_expansion_pct=11.0,
            )
        finally:
            _or._store = orig_store
            _cs._calibration = orig_cal
            _ps._store = orig_prior
            _sc._load_walk_forward_priors.cache_clear()

        assert result["calibration_n_after"] == n_before + 1
        assert result["status"] == "finalized"

        # Verify persistence: reload calibration from the same path.
        cal_reloaded = IVExpansionCalibration(store_path=cal_path)
        assert cal_reloaded._n() == n_before + 1

    def test_finalize_marks_trade_as_finalized(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        store = OutcomeStore(store_path=store_path)
        tid = _insert_minimal(store, source_type="paper")

        import services.outcome_recorder as _or
        import services.calibration_service as _cs
        import services.structure_prior_store as _ps
        import services.structure_scorecard as _sc

        orig_store = _or._store
        orig_cal = _cs._calibration
        orig_prior = _ps._store

        _or._store = store
        _cs._calibration = IVExpansionCalibration(store_path=cal_path)
        _ps._store = StructurePriorStore(store_path=prior_path)

        try:
            finalize_trade_and_update_learning(
                trade_id=tid,
                exit_date=date(2025, 1, 20),
                realized_return_pct=5.0,
                realized_expansion_pct=7.0,
            )
        finally:
            _or._store = orig_store
            _cs._calibration = orig_cal
            _ps._store = orig_prior
            _sc._load_walk_forward_priors.cache_clear()

        row = store.get_trade(tid)
        assert row["status"] == "finalized"

    def test_finalize_persists_calibration_observation_id_and_source(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        store = OutcomeStore(store_path=store_path)
        tid = _insert_minimal(store, source_type="live")

        import services.outcome_recorder as _or
        import services.calibration_service as _cs
        import services.structure_prior_store as _ps
        import services.structure_scorecard as _sc

        orig_store = _or._store
        orig_cal = _cs._calibration
        orig_prior = _ps._store

        _or._store = store
        _cs._calibration = IVExpansionCalibration(store_path=cal_path)
        _ps._store = StructurePriorStore(store_path=prior_path)

        try:
            finalize_trade_and_update_learning(
                trade_id=tid,
                exit_date=date(2025, 1, 20),
                realized_return_pct=6.0,
                realized_expansion_pct=8.0,
            )
        finally:
            _or._store = orig_store
            _cs._calibration = orig_cal
            _ps._store = orig_prior
            _sc.reload_walk_forward_priors()

        cal = IVExpansionCalibration(store_path=cal_path)
        assert tid in cal._observation_ids
        assert cal.diagnostics()["n_live"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. Structure prior feedback
# ══════════════════════════════════════════════════════════════════════════════


class TestStructurePriorFeedback:
    def test_update_increments_observation_count(self, prior_store: StructurePriorStore) -> None:
        prior_store.update(
            structure="atm_straddle",
            realized_return_pct=8.0,
            realized_expansion_pct=10.0,
        )
        diag = prior_store.diagnostics()
        assert diag["structures"]["atm_straddle"]["observation_count"] == 1

    def test_update_increments_correct_structure_only(
        self, prior_store: StructurePriorStore
    ) -> None:
        prior_store.update(
            structure="atm_straddle",
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
        )
        prior_store.update(
            structure="otm_strangle",
            realized_return_pct=-3.0,
            realized_expansion_pct=-2.0,
        )
        diag = prior_store.diagnostics()
        assert diag["structures"]["atm_straddle"]["observation_count"] == 1
        assert diag["structures"]["otm_strangle"]["observation_count"] == 1
        assert diag["structures"]["call_calendar"]["observation_count"] == 0

    def test_win_uses_laplace_smoothing_below_threshold(
        self, prior_store: StructurePriorStore
    ) -> None:
        """With N=1 win, Laplace-1: (1+1)/(1+2) = 2/3 ≈ 0.667."""
        prior_store.update(
            structure="call_calendar",
            realized_return_pct=5.0,  # win
            realized_expansion_pct=6.0,
        )
        d = prior_store.diagnostics()["structures"]["call_calendar"]
        assert d["observation_count"] == 1
        # Laplace: (1+1)/(1+2) = 0.6667
        assert d["win_rate"] == pytest.approx(2 / 3, abs=1e-6)

    def test_win_uses_laplace_smoothing_1_loss(
        self, prior_store: StructurePriorStore
    ) -> None:
        """With N=1 loss, Laplace-1: (0+1)/(1+2) = 1/3 ≈ 0.333."""
        prior_store.update(
            structure="put_calendar",
            realized_return_pct=-3.0,  # loss
            realized_expansion_pct=-2.0,
        )
        d = prior_store.diagnostics()["structures"]["put_calendar"]
        assert d["win_rate"] == pytest.approx(1 / 3, abs=1e-6)

    def test_win_rate_no_laplace_at_or_above_threshold(
        self, prior_store: StructurePriorStore
    ) -> None:
        """For N >= LAPLACE_SMOOTHING_THRESHOLD, use raw win rate."""
        n = LAPLACE_SMOOTHING_THRESHOLD
        for i in range(n):
            # All wins
            prior_store.update(
                structure="atm_straddle",
                realized_return_pct=5.0,
                realized_expansion_pct=7.0,
                source_type="paper",
            )
        d = prior_store.diagnostics()["structures"]["atm_straddle"]
        # All wins at exactly threshold: raw win_rate = n/n = 1.0
        assert d["win_rate"] == pytest.approx(1.0, abs=1e-6)

    def test_get_prior_dict_returns_none_below_min_obs(
        self, prior_store: StructurePriorStore
    ) -> None:
        """Below MIN_OBS_FOR_OVERRIDE, get_prior_dict should return None."""
        for _ in range(MIN_OBS_FOR_OVERRIDE - 1):
            prior_store.update(
                structure="atm_straddle",
                realized_return_pct=5.0,
                realized_expansion_pct=7.0,
            )
        assert prior_store.get_prior_dict("atm_straddle") is None

    def test_get_prior_dict_returns_dict_at_min_obs(
        self, prior_store: StructurePriorStore
    ) -> None:
        """At exactly MIN_OBS_FOR_OVERRIDE, get_prior_dict should return a dict."""
        for _ in range(MIN_OBS_FOR_OVERRIDE):
            prior_store.update(
                structure="atm_straddle",
                realized_return_pct=5.0,
                realized_expansion_pct=7.0,
            )
        d = prior_store.get_prior_dict("atm_straddle")
        assert d is not None
        assert d["history_count"] == MIN_OBS_FOR_OVERRIDE
        assert "win_rate" in d
        assert "avg_return_pct" in d
        assert "rank_score" in d
        assert "source" in d

    def test_unknown_structure_ignored(self, prior_store: StructurePriorStore) -> None:
        """Updating an unknown structure should not raise, just warn."""
        prior_store.update(
            structure="not_a_real_structure",
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
        )
        # Nothing should be stored for the fake structure.
        assert prior_store.get_prior_dict("not_a_real_structure") is None


# ══════════════════════════════════════════════════════════════════════════════
# 4. Replay seeding (idempotency and count accuracy)
# ══════════════════════════════════════════════════════════════════════════════


class TestReplaySeeding:
    """
    Uses scripts.seed_outcomes_from_replay.seed_from_trades() with fixture data
    so no actual DB file is needed.
    """

    def _make_trade_rows(self, n: int = 5, symbol_prefix: str = "SYM") -> list:
        rows = []
        for i in range(n):
            rows.append(
                {
                    "symbol": f"{symbol_prefix}{i}",
                    "trade_date": f"2024-0{(i % 9) + 1}-10",
                    "event_date": f"2024-0{(i % 9) + 1}-15",
                    "days_to_earnings": 5,
                    "setup_score": 0.60 + i * 0.02,
                    "gross_return_pct": 0.08 + i * 0.01,  # fraction form
                    "net_return_pct": 0.05 + i * 0.01,    # fraction form
                    "pnl_per_contract": 50.0 + i * 5,
                    "execution_profile": "institutional",
                }
            )
        return rows

    def test_seed_inserts_correct_count(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        from scripts.seed_outcomes_from_replay import seed_from_trades

        rows = self._make_trade_rows(5)
        result = seed_from_trades(
            rows,
            structure="call_calendar",
            dry_run=False,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )
        assert result["inserted"] == 5
        assert result["skipped_duplicate"] == 0
        assert result["skipped_bad_data"] == 0

    def test_repeated_seed_does_not_double_count(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        """Second seed run with the same rows should produce 0 new inserts."""
        from scripts.seed_outcomes_from_replay import seed_from_trades

        rows = self._make_trade_rows(5)

        # First run
        seed_from_trades(
            rows,
            structure="call_calendar",
            dry_run=False,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )

        # Second run — same rows
        result2 = seed_from_trades(
            rows,
            structure="call_calendar",
            dry_run=False,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )

        assert result2["inserted"] == 0
        assert result2["skipped_duplicate"] == 5
        assert result2["cal_updates"] == 0

        # Outcome store should still have exactly 5 rows.
        store = OutcomeStore(store_path=store_path)
        assert store.count() == 5
        cal = IVExpansionCalibration(store_path=cal_path)
        assert cal._n() == 5

    def test_seeded_calibration_n_matches_inserted(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        from scripts.seed_outcomes_from_replay import seed_from_trades

        rows = self._make_trade_rows(7)
        result = seed_from_trades(
            rows,
            structure="call_calendar",
            dry_run=False,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )

        # All 7 inserted → calibration should have 7 observations.
        assert result["cal_updates"] == 7

        cal = IVExpansionCalibration(store_path=cal_path)
        assert cal._n() == 7

    def test_seeded_prior_count_matches_inserted(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        from scripts.seed_outcomes_from_replay import seed_from_trades

        rows = self._make_trade_rows(6)
        seed_from_trades(
            rows,
            structure="call_calendar",
            dry_run=False,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )

        ps = StructurePriorStore(store_path=prior_path)
        diag = ps.diagnostics()
        assert diag["structures"]["call_calendar"]["observation_count"] == 6

    def test_dry_run_touches_nothing(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        from scripts.seed_outcomes_from_replay import seed_from_trades

        rows = self._make_trade_rows(4)
        result = seed_from_trades(
            rows,
            structure="call_calendar",
            dry_run=True,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )

        # Dry run should report what it would do but write nothing.
        assert result["inserted"] == 4
        assert result["dry_run"] is True

        # No files should have been created.
        assert not store_path.exists()
        assert not cal_path.exists()
        assert not prior_path.exists()

    def test_bad_data_rows_are_skipped(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        from scripts.seed_outcomes_from_replay import seed_from_trades

        good = self._make_trade_rows(3)
        bad = [
            {"symbol": None, "trade_date": "2024-01-10", "setup_score": 0.5,
             "gross_return_pct": 0.05, "net_return_pct": 0.03, "pnl_per_contract": 30,
             "execution_profile": "x", "event_date": "2024-01-15", "days_to_earnings": 5},
            {"symbol": "AAPL", "trade_date": None, "setup_score": 0.5,
             "gross_return_pct": 0.05, "net_return_pct": 0.03, "pnl_per_contract": 30,
             "execution_profile": "x", "event_date": "2024-01-15", "days_to_earnings": 5},
            {"symbol": "AAPL", "trade_date": "2024-01-10", "setup_score": None,
             "gross_return_pct": None, "net_return_pct": None, "pnl_per_contract": 30,
             "execution_profile": "x", "event_date": "2024-01-15", "days_to_earnings": 5},
        ]
        result = seed_from_trades(
            good + bad,
            structure="call_calendar",
            dry_run=False,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )
        assert result["inserted"] == 3
        assert result["skipped_bad_data"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# 5. Persistence — restart / reload
# ══════════════════════════════════════════════════════════════════════════════


class TestPersistence:
    def test_outcome_store_survives_restart(self, store_path: Path) -> None:
        """Rows inserted in one OutcomeStore instance are visible after reload."""
        s1 = OutcomeStore(store_path=store_path)
        tid = _insert_minimal(s1)
        s1.close()

        s2 = OutcomeStore(store_path=store_path)
        row = s2.get_trade(tid)
        assert row is not None
        assert row["symbol"] == "AAPL"
        s2.close()

    def test_outcome_store_exit_update_survives_restart(self, store_path: Path) -> None:
        s1 = OutcomeStore(store_path=store_path)
        tid = _insert_minimal(s1)
        s1.update_exit(
            trade_id=tid,
            exit_date=date(2025, 2, 1),
            realized_return_pct=7.5,
            realized_expansion_pct=9.0,
        )
        s1.close()

        s2 = OutcomeStore(store_path=store_path)
        row = s2.get_trade(tid)
        assert row["realized_return_pct"] == pytest.approx(7.5)
        assert row["status"] == "exited"
        s2.close()

    def test_structure_prior_store_survives_restart(self, prior_path: Path) -> None:
        ps1 = StructurePriorStore(store_path=prior_path)
        for _ in range(MIN_OBS_FOR_OVERRIDE):
            ps1.update(
                structure="atm_straddle",
                realized_return_pct=8.0,
                realized_expansion_pct=10.0,
            )

        ps2 = StructurePriorStore(store_path=prior_path)
        d = ps2.diagnostics()["structures"]["atm_straddle"]
        assert d["observation_count"] == MIN_OBS_FOR_OVERRIDE

    def test_calibration_survives_restart(self, cal_path: Path) -> None:
        cal1 = IVExpansionCalibration(store_path=cal_path)
        cal1.update(0.72, 11.0)
        cal1.update(0.65, 8.5)

        cal2 = IVExpansionCalibration(store_path=cal_path)
        assert cal2._n() == 2

    def test_prior_store_json_is_valid(self, prior_path: Path) -> None:
        """The JSON file written by StructurePriorStore should be valid JSON
        with the expected top-level keys."""
        ps = StructurePriorStore(store_path=prior_path)
        ps.update(
            structure="call_calendar",
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
        )
        raw = json.loads(prior_path.read_text())
        assert "schema_version" in raw
        assert "structures" in raw
        assert "call_calendar" in raw["structures"]


# ══════════════════════════════════════════════════════════════════════════════
# 6. Provenance separation
# ══════════════════════════════════════════════════════════════════════════════


class TestProvenance:
    def test_source_types_are_stored_and_distinguishable(
        self, store: OutcomeStore
    ) -> None:
        for src, sym in [("replay", "AAPL"), ("paper", "MSFT"), ("live", "GOOGL")]:
            _insert_minimal(
                store,
                symbol=sym,
                entry_date=date(2025, 1, 1),
                source_type=src,
            )
        counts = store.count_by_source()
        assert counts.get("replay") == 1
        assert counts.get("paper") == 1
        assert counts.get("live") == 1

    def test_replay_source_is_stored_in_row(self, store: OutcomeStore) -> None:
        tid = _insert_minimal(store, source_type="replay")
        row = store.get_trade(tid)
        assert row["source_type"] == "replay"

    def test_prior_store_source_types_tracked_per_structure(
        self, prior_store: StructurePriorStore
    ) -> None:
        prior_store.update(
            structure="atm_straddle",
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
            source_type="replay",
        )
        prior_store.update(
            structure="atm_straddle",
            realized_return_pct=3.0,
            realized_expansion_pct=4.0,
            source_type="paper",
        )
        diag = prior_store.diagnostics()
        src_types = diag["structures"]["atm_straddle"]["source_types"]
        assert src_types.get("replay") == 1
        assert src_types.get("paper") == 1

    def test_replay_seeded_observations_tagged_as_replay(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        from scripts.seed_outcomes_from_replay import seed_from_trades

        rows = [
            {
                "symbol": "AAPL",
                "trade_date": "2024-02-10",
                "event_date": "2024-02-15",
                "days_to_earnings": 5,
                "setup_score": 0.65,
                "gross_return_pct": 0.09,
                "net_return_pct": 0.06,
                "pnl_per_contract": 60.0,
                "execution_profile": "institutional",
            }
        ]
        seed_from_trades(
            rows,
            structure="call_calendar",
            dry_run=False,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )
        store = OutcomeStore(store_path=store_path)
        # PR-L: seeding now folds earnings_date into the trade_id to prevent
        # collisions between same-(symbol/structure/entry) trades across
        # different earnings events. Look up with the matching 4-field form.
        tid = make_trade_id(
            "AAPL",
            date(2024, 2, 10),
            "call_calendar",
            earnings_date=date(2024, 2, 15),
        )
        row = store.get_trade(tid)
        assert row is not None
        assert row["source_type"] == "replay"
        assert row["status"] == "finalized"
        cal = IVExpansionCalibration(store_path=cal_path)
        assert cal.diagnostics()["n_replay"] == 1

    def test_live_and_paper_observations_remain_separate_from_replay(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        from scripts.seed_outcomes_from_replay import seed_from_trades

        # Seed one replay trade.
        rows = [
            {
                "symbol": "MSFT",
                "trade_date": "2024-03-01",
                "event_date": "2024-03-06",
                "days_to_earnings": 5,
                "setup_score": 0.70,
                "gross_return_pct": 0.07,
                "net_return_pct": 0.05,
                "pnl_per_contract": 50.0,
                "execution_profile": "institutional",
            }
        ]
        seed_from_trades(
            rows,
            structure="call_calendar",
            dry_run=False,
            outcome_store_path=store_path,
            calibration_store_path=cal_path,
            prior_store_path=prior_path,
        )

        # Manually add a paper trade.
        store = OutcomeStore(store_path=store_path)
        _insert_minimal(
            store,
            symbol="MSFT",
            entry_date=date(2024, 6, 1),
            source_type="paper",
        )
        counts = store.count_by_source()
        assert counts.get("replay") == 1
        assert counts.get("paper") == 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. Utility functions
# ══════════════════════════════════════════════════════════════════════════════


class TestUtilities:
    def test_make_trade_id_is_deterministic(self) -> None:
        d = date(2025, 3, 15)
        assert make_trade_id("AAPL", d, "atm_straddle") == make_trade_id(
            "AAPL", d, "atm_straddle"
        )

    def test_make_trade_id_symbol_normalised_to_uppercase(self) -> None:
        d = date(2025, 3, 15)
        assert make_trade_id("aapl", d, "atm_straddle") == make_trade_id(
            "AAPL", d, "atm_straddle"
        )

    def test_make_trade_id_differs_by_structure(self) -> None:
        d = date(2025, 3, 15)
        assert make_trade_id("AAPL", d, "atm_straddle") != make_trade_id(
            "AAPL", d, "otm_strangle"
        )

    def test_make_snapshot_hash_is_deterministic(self) -> None:
        snap = {"symbol": "AAPL", "iv_rv": 1.12, "dte": 7}
        assert make_snapshot_hash(snap) == make_snapshot_hash(snap)

    def test_make_snapshot_hash_differs_for_different_inputs(self) -> None:
        a = make_snapshot_hash({"x": 1})
        b = make_snapshot_hash({"x": 2})
        assert a != b


# ══════════════════════════════════════════════════════════════════════════════
# PR-M. learning_update_status — finalization no longer hides learning failures
# ══════════════════════════════════════════════════════════════════════════════


def _finalize_with_patched_services(
    *,
    store: "OutcomeStore",
    cal_path: Path,
    prior_path: Path,
    trade_id: str,
    exit_date: date,
    realized_return_pct: float,
    realized_expansion_pct: float,
    cal_update_side_effect=None,
    prior_update_side_effect=None,
):
    """Helper: run finalize_trade_and_update_learning with optional injected
    failures into the calibration or structure prior side. Returns the
    result dict. Mirrors the singleton-swap pattern used elsewhere in this file.
    """
    import services.outcome_recorder as _or
    import services.calibration_service as _cs
    import services.structure_prior_store as _ps
    import services.structure_scorecard as _sc
    from unittest.mock import patch

    orig_store = _or._store
    orig_cal = _cs._calibration
    orig_prior = _ps._store

    cal = IVExpansionCalibration(store_path=cal_path)
    prior = StructurePriorStore(store_path=prior_path)
    _or._store = store
    _cs._calibration = cal
    _ps._store = prior

    contexts = []
    if cal_update_side_effect is not None:
        contexts.append(patch.object(cal, "update", side_effect=cal_update_side_effect))
    if prior_update_side_effect is not None:
        contexts.append(patch.object(prior, "update", side_effect=prior_update_side_effect))

    try:
        with _stack_contexts(contexts):
            return finalize_trade_and_update_learning(
                trade_id=trade_id,
                exit_date=exit_date,
                realized_return_pct=realized_return_pct,
                realized_expansion_pct=realized_expansion_pct,
            )
    finally:
        _or._store = orig_store
        _cs._calibration = orig_cal
        _ps._store = orig_prior
        _sc._load_walk_forward_priors.cache_clear()


def _stack_contexts(contexts):
    """contextlib.ExitStack equivalent for a list of context managers."""
    import contextlib

    stack = contextlib.ExitStack()
    for ctx in contexts:
        stack.enter_context(ctx)
    return stack


class TestLearningUpdateStatus:
    def test_successful_finalize_marks_learning_complete(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        store = OutcomeStore(store_path=store_path)
        tid = _insert_minimal(store, source_type="paper")
        result = _finalize_with_patched_services(
            store=store,
            cal_path=cal_path,
            prior_path=prior_path,
            trade_id=tid,
            exit_date=date(2025, 1, 20),
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
        )
        assert result["learning_update_status"] == "complete"
        row = store.get_trade(tid)
        assert row["status"] == "finalized"
        assert row["learning_update_status"] == "complete"

    def test_calibration_failure_marks_calibration_failed(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        """Previously: calibration raised → caught, then mark_finalized ran
        anyway, leaving the trade silently incomplete. Now: the failure is
        recorded so a retry sweep can find it."""
        store = OutcomeStore(store_path=store_path)
        tid = _insert_minimal(store, source_type="paper")
        result = _finalize_with_patched_services(
            store=store,
            cal_path=cal_path,
            prior_path=prior_path,
            trade_id=tid,
            exit_date=date(2025, 1, 20),
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
            cal_update_side_effect=RuntimeError("simulated calibration outage"),
        )
        assert result["learning_update_status"] == "calibration_failed"
        # Trade is still finalized — its P&L is real.
        row = store.get_trade(tid)
        assert row["status"] == "finalized"
        assert row["learning_update_status"] == "calibration_failed"

    def test_prior_failure_marks_prior_failed(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        store = OutcomeStore(store_path=store_path)
        tid = _insert_minimal(store, source_type="paper")
        result = _finalize_with_patched_services(
            store=store,
            cal_path=cal_path,
            prior_path=prior_path,
            trade_id=tid,
            exit_date=date(2025, 1, 20),
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
            prior_update_side_effect=RuntimeError("simulated prior outage"),
        )
        assert result["learning_update_status"] == "prior_failed"
        assert store.get_trade(tid)["learning_update_status"] == "prior_failed"

    def test_both_failures_mark_both_failed(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        store = OutcomeStore(store_path=store_path)
        tid = _insert_minimal(store, source_type="paper")
        result = _finalize_with_patched_services(
            store=store,
            cal_path=cal_path,
            prior_path=prior_path,
            trade_id=tid,
            exit_date=date(2025, 1, 20),
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
            cal_update_side_effect=RuntimeError("cal down"),
            prior_update_side_effect=RuntimeError("prior down"),
        )
        assert result["learning_update_status"] == "both_failed"
        assert store.get_trade(tid)["learning_update_status"] == "both_failed"

    def test_list_failed_returns_only_non_complete(
        self, store_path: Path, cal_path: Path, prior_path: Path
    ) -> None:
        """A retry sweep should be able to find exactly the trades that
        need re-attempted learning propagation."""
        store = OutcomeStore(store_path=store_path)
        good_tid = _insert_minimal(store, symbol="AAPL", source_type="paper")
        cal_fail_tid = _insert_minimal(store, symbol="MSFT", source_type="paper")

        # Successful finalize.
        _finalize_with_patched_services(
            store=store, cal_path=cal_path, prior_path=prior_path,
            trade_id=good_tid, exit_date=date(2025, 1, 20),
            realized_return_pct=4.0, realized_expansion_pct=6.0,
        )
        # Failed finalize.
        _finalize_with_patched_services(
            store=store, cal_path=cal_path, prior_path=prior_path,
            trade_id=cal_fail_tid, exit_date=date(2025, 1, 21),
            realized_return_pct=-1.0, realized_expansion_pct=-3.0,
            cal_update_side_effect=RuntimeError("cal down"),
        )

        failed = store.list_trades_with_failed_learning_update()
        failed_tids = {row["trade_id"] for row in failed}
        assert cal_fail_tid in failed_tids
        assert good_tid not in failed_tids
