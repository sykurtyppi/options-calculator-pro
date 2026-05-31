"""
Phase 2.1 — Dedicated unit tests for StructurePriorStore.

Covers the six properties identified as untested in the architectural review:
  - Laplace-1 smoothing crossover at LAPLACE_SMOOTHING_THRESHOLD
  - MIN_OBS_FOR_OVERRIDE boundary (below → None; at threshold → dict)
  - source_types accumulator for mixed provenance
  - Round-trip save/load field preservation
  - Concurrent writes do not corrupt observation count
  - Unknown structure name is silently dropped
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path

import pytest

from services.structure_prior_store import (
    LAPLACE_SMOOTHING_THRESHOLD,
    MIN_OBS_FOR_OVERRIDE,
    StructurePriorStore,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _store(tmp_path: Path) -> StructurePriorStore:
    return StructurePriorStore(store_path=tmp_path / "priors.json")


def _add(store: StructurePriorStore, structure: str, *, n: int, win: bool = True,
         source: str = "replay", base_date: date = date(2024, 1, 1)) -> None:
    for i in range(n):
        store.update(
            structure=structure,
            realized_return_pct=5.0 if win else -2.0,
            realized_expansion_pct=2.0,
            source_type=source,
            observation_date=date(base_date.year, base_date.month, base_date.day + i),
            observation_id=f"{structure}-{source}-{i}",
        )


# ── Phase 2.1 tests ───────────────────────────────────────────────────────────


def test_laplace_smoothing_applies_below_threshold(tmp_path: Path) -> None:
    """win_rate uses Laplace-1 for N < LAPLACE_SMOOTHING_THRESHOLD; switches to raw at threshold."""
    store = _store(tmp_path)
    n_below = LAPLACE_SMOOTHING_THRESHOLD - 1  # 9 — all wins
    _add(store, "atm_straddle", n=n_below, win=True)

    d = store.get_prior_dict("atm_straddle", as_of_date=date(2025, 1, 1))
    assert d is not None, "9 >= MIN_OBS_FOR_OVERRIDE so dict should be returned"
    expected_smoothed = (n_below + 1) / (n_below + 2)
    assert abs(d["win_rate"] - expected_smoothed) < 1e-9, (
        f"expected Laplace-smoothed win_rate={expected_smoothed:.6f}, got {d['win_rate']:.6f}"
    )

    # Add the threshold-th observation (still a win) → raw rate kicks in
    store.update(
        structure="atm_straddle",
        realized_return_pct=5.0,
        realized_expansion_pct=2.0,
        source_type="replay",
        observation_date=date(2024, 1, LAPLACE_SMOOTHING_THRESHOLD),
        observation_id=f"atm_straddle-replay-{n_below}",
    )
    d2 = store.get_prior_dict("atm_straddle", as_of_date=date(2025, 1, 1))
    assert d2 is not None
    assert abs(d2["win_rate"] - 1.0) < 1e-9, (
        f"all {LAPLACE_SMOOTHING_THRESHOLD} wins → raw rate should be 1.0, got {d2['win_rate']}"
    )


def test_min_obs_for_override_boundary(tmp_path: Path) -> None:
    """get_prior_dict returns None for N < MIN_OBS_FOR_OVERRIDE; dict for N == threshold."""
    store = _store(tmp_path)

    # 4 observations → None
    _add(store, "otm_strangle", n=MIN_OBS_FOR_OVERRIDE - 1)
    assert store.get_prior_dict("otm_strangle") is None, (
        f"{MIN_OBS_FOR_OVERRIDE - 1} obs should return None"
    )

    # 5th observation → dict returned
    store.update(
        structure="otm_strangle",
        realized_return_pct=3.0,
        realized_expansion_pct=1.5,
        source_type="replay",
        observation_date=date(2024, 1, MIN_OBS_FOR_OVERRIDE),
        observation_id=f"otm_strangle-replay-{MIN_OBS_FOR_OVERRIDE - 1}",
    )
    d = store.get_prior_dict("otm_strangle")
    assert d is not None, f"{MIN_OBS_FOR_OVERRIDE} obs should return a dict"
    assert d["history_count"] == MIN_OBS_FOR_OVERRIDE


def test_source_types_accumulator_for_mixed_sources(tmp_path: Path) -> None:
    """source_types tracks per-provenance counts correctly across replay/paper/live."""
    store = _store(tmp_path)
    _add(store, "atm_straddle", n=3, source="replay")
    _add(store, "atm_straddle", n=2, source="paper",
         base_date=date(2024, 2, 1))
    store.update(
        structure="atm_straddle",
        realized_return_pct=2.0,
        realized_expansion_pct=1.0,
        source_type="live",
        observation_date=date(2024, 3, 1),
        observation_id="atm_straddle-live-0",
    )

    diag = store.diagnostics()
    src = diag["structures"]["atm_straddle"]["source_types"]
    assert src.get("replay") == 3
    assert src.get("paper") == 2
    assert src.get("live") == 1


def test_round_trip_save_load_preserves_all_fields(tmp_path: Path) -> None:
    """Saving then reloading from disk preserves observation_count, win_rate, avg_return, source_types."""
    store_path = tmp_path / "priors.json"
    store = StructurePriorStore(store_path=store_path)
    for i in range(6):
        store.update(
            structure="call_calendar",
            realized_return_pct=4.0 if i % 2 == 0 else -1.0,
            realized_expansion_pct=2.0,
            source_type="paper",
            observation_date=date(2024, 1, i + 1),
            observation_id=f"call_calendar-paper-{i}",
        )
    store.save()

    reloaded = StructurePriorStore(store_path=store_path)
    orig = store.diagnostics()["structures"]["call_calendar"]
    reloaded_d = reloaded.diagnostics()["structures"]["call_calendar"]

    assert reloaded_d["observation_count"] == orig["observation_count"]
    assert abs(reloaded_d["win_rate"] - orig["win_rate"]) < 1e-9
    assert abs(reloaded_d["avg_return_pct"] - orig["avg_return_pct"]) < 1e-9
    assert reloaded_d["source_types"] == orig["source_types"]


def test_concurrent_writes_do_not_corrupt_count(tmp_path: Path) -> None:
    """16 threads each writing a unique observation arrive at total count == 16."""
    store = _store(tmp_path)
    n_threads = 16

    def add_one(index: int) -> None:
        store.update(
            structure="put_calendar",
            realized_return_pct=1.0,
            realized_expansion_pct=0.5,
            source_type="replay",
            observation_date=date(2024, 1, 1),
            observation_id=f"concurrent-{index}",
        )

    threads = [threading.Thread(target=add_one, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    count = store.diagnostics()["structures"]["put_calendar"]["observation_count"]
    assert count == n_threads, f"expected {n_threads} observations, got {count}"


def test_unknown_structure_is_silently_dropped(tmp_path: Path) -> None:
    """update() with an unrecognised structure name does not raise and adds no entry."""
    store = _store(tmp_path)
    store.update(
        structure="iron_condor",
        realized_return_pct=5.0,
        realized_expansion_pct=2.0,
        source_type="paper",
        observation_date=date(2024, 1, 1),
    )
    assert "iron_condor" not in store.diagnostics()["structures"]
