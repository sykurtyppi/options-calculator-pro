"""
Phase 2.2 — Learning loop round-trip integration tests.

Pins two properties that no test previously exercised:
  1. finalize_trade_and_update_learning() updates the prior store AND invalidates
     the scorecard cache so the next build_structure_scorecards() picks up the new
     win_rate — not a stale snapshot.
  2. Submitting the same trade_id twice increments prior observation count by 1,
     not 2 (deduplication via observation_id).

All I/O is isolated to tmp_path; the production ~/.options_calculator_pro paths
are never touched.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from services.calibration_service import IVExpansionCalibration
from services.outcome_recorder import OutcomeStore, finalize_trade_and_update_learning
from services.structure_prior_store import MIN_OBS_FOR_OVERRIDE, StructurePriorStore


# ── singleton-swap helper ─────────────────────────────────────────────────────


class _IsolatedLearningContext:
    """Context manager: redirects all singleton I/O to tmp paths for one test."""

    def __init__(self, store_path: Path, prior_path: Path, cal_path: Path) -> None:
        self._store_path = store_path
        self._prior_path = prior_path
        self._cal_path = cal_path

    def __enter__(self):
        import services.outcome_recorder as _or
        import services.calibration_service as _cs
        import services.structure_prior_store as _ps
        import services.structure_scorecard as _sc

        self._or = _or
        self._cs = _cs
        self._ps = _ps
        self._sc = _sc

        self._orig_store = _or._store
        self._orig_cal = _cs._calibration
        self._orig_prior = _ps._store

        self.store = OutcomeStore(store_path=self._store_path)
        self.cal = IVExpansionCalibration(store_path=self._cal_path)
        self.prior = StructurePriorStore(store_path=self._prior_path)

        _or._store = self.store
        _cs._calibration = self.cal
        _ps._store = self.prior
        # Clear the walk-forward prior cache on ENTER as well as exit: a prior
        # test may have warmed _PRIORS_CACHE from the global/prod store, which
        # would make the in-test priming read reflect global state instead of
        # the isolated tmp store (flaky once atm_straddle accrues >= 5 obs).
        _sc._load_walk_forward_priors.cache_clear()
        return self

    def __exit__(self, *_):
        self._or._store = self._orig_store
        self._cs._calibration = self._orig_cal
        self._ps._store = self._orig_prior
        self._sc._load_walk_forward_priors.cache_clear()


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def paths(tmp_path: Path):
    return (
        tmp_path / "outcomes.sqlite",
        tmp_path / "priors.json",
        tmp_path / "calibration.json",
    )


def _seed_trade(store: OutcomeStore, *, structure: str = "atm_straddle") -> str:
    from services.outcome_recorder import make_trade_id
    trade_id = make_trade_id("AAPL", date(2025, 1, 10), structure,
                             earnings_date=date(2025, 1, 15))
    store.insert_entry(
        trade_id=trade_id,
        symbol="AAPL",
        structure=structure,
        entry_date=date(2025, 1, 10),
        setup_score=0.72,
        source_type="paper",
        earnings_date=date(2025, 1, 15),
    )
    return trade_id


# ── Phase 2.2 tests ───────────────────────────────────────────────────────────


def test_finalize_updates_priors_and_invalidates_cache(paths) -> None:
    """After finalize(), the prior store has a new observation AND the scorecard
    cache is cleared so the next _load_walk_forward_priors() sees the update.

    The cache-invalidation contract is what this test exists to protect.
    It used to observe the update via the MIN_OBS_FOR_OVERRIDE cliff (prior
    flipping wholesale to the persistent store on the 5th observation). That
    cliff was replaced by a continuous shrinkage blend, so the update is now
    observed the way it actually manifests: the blended prior's live_n rises by
    one and its evidence count rises by one, on a NEW (non-cached) object.
    """
    store_path, prior_path, cal_path = paths

    with _IsolatedLearningContext(store_path, prior_path, cal_path) as ctx:
        import services.structure_scorecard as _sc

        # Seed MIN_OBS_FOR_OVERRIDE-1 observations directly into the prior store
        # (bypassing finalize so the cache is not yet cleared).
        for i in range(MIN_OBS_FOR_OVERRIDE - 1):
            ctx.prior.update(
                structure="atm_straddle",
                realized_return_pct=5.0,
                realized_expansion_pct=2.0,
                source_type="paper",
                observation_date=date(2025, 1, i + 1),
                observation_id=f"pre-seed-{i}",
            )

        # Prime the cache. The 4 seeded observations are blended into the report
        # prior (shrinkage weights them down; they no longer replace it).
        priors_before = _sc._load_walk_forward_priors()
        assert "live_n=4" in priors_before["atm_straddle"].source, (
            f"pre-finalize prior should blend the 4 seeded observations, "
            f"got source={priors_before['atm_straddle'].source}"
        )
        # Confirm it's actually cached — same dict object on repeated call.
        assert _sc._load_walk_forward_priors() is priors_before, (
            "_load_walk_forward_priors should return the cached object without re-reading"
        )

        # Finalize the threshold-crossing trade: adds obs #5 and must clear the cache.
        trade_id = _seed_trade(ctx.store)
        result = finalize_trade_and_update_learning(
            trade_id=trade_id,
            exit_date=date(2025, 1, 20),
            realized_return_pct=8.0,
            realized_expansion_pct=10.0,
        )

        # Cache-invalidation check: post-finalize call must return a NEW object …
        priors_after = _sc._load_walk_forward_priors()
        assert priors_after is not priors_before, (
            "finalize_trade_and_update_learning() must clear the prior cache; "
            "_load_walk_forward_priors() returned the stale cached object"
        )
        # … and that new object must reflect the newly persisted observation:
        # one more live observation in the blend, and one more unit of evidence.
        assert "live_n=5" in priors_after["atm_straddle"].source, (
            f"post-finalize prior must reflect the 5th observation, "
            f"got source={priors_after['atm_straddle'].source}"
        )
        assert (
            priors_after["atm_straddle"].history_count
            == priors_before["atm_straddle"].history_count + 1
        ), "post-finalize evidence count must rise by exactly the one new observation"

    assert result["status"] == "finalized"
    assert result["prior_observation_count"] == MIN_OBS_FOR_OVERRIDE


def test_calibration_observation_dedup_via_observation_id(paths) -> None:
    """Finalizing the same trade_id twice increments prior count by 1, not 2."""
    store_path, prior_path, cal_path = paths

    with _IsolatedLearningContext(store_path, prior_path, cal_path) as ctx:
        trade_id = _seed_trade(ctx.store)

        finalize_trade_and_update_learning(
            trade_id=trade_id,
            exit_date=date(2025, 1, 20),
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
        )
        # Second call with same trade_id — deduplication should fire on prior and calibration
        result2 = finalize_trade_and_update_learning(
            trade_id=trade_id,
            exit_date=date(2025, 1, 20),
            realized_return_pct=5.0,
            realized_expansion_pct=7.0,
        )

    assert result2["prior_observation_count"] == 1, (
        "Second finalize() must not double-count: prior should still show 1 observation"
    )
    assert result2["calibration_n_after"] == result2["calibration_n_before"], (
        "Second finalize() must not add a second calibration observation"
    )
