"""
Phase 1.6 leakage-detection tests — walk-forward data contamination.

These tests document the specific missing APIs that caused future observations
to leak into walk-forward backtest evaluations.  They were originally written
as xfail evidence (commit e0daf1e) and un-xfailed once Phases 1.1-1.3 landed.

Fix tracked in: https://github.com/sykurtyppi/options-calculator-pro/issues/18
Related: issue #12 (production-side StructurePriorStore refactor)
"""

import datetime
import json
from datetime import date
from unittest.mock import patch

import pytest

from services.calibration_service import IVExpansionCalibration
from services.structure_prior_store import (
    BacktestLeakageError,
    MIN_OBS_FOR_OVERRIDE,
    StructurePriorStore,
)


# ── Test 1: StructurePriorStore.get_prior_dict() accepts as_of_date ───────────


def test_structure_prior_store_get_prior_dict_accepts_as_of_date(tmp_path):
    """get_prior_dict() accepts as_of_date so walk-forward evaluations can
    freeze the prior at a backtest cutoff date.

    Observations added with observation_date=past must not appear when
    as_of_date is before that date.  They must appear when as_of_date is
    on or after.
    """
    store = StructurePriorStore(store_path=tmp_path / "priors.json")
    obs_date = datetime.date(2024, 6, 1)
    for i in range(MIN_OBS_FOR_OVERRIDE):
        store.update(
            structure="atm_straddle",
            realized_return_pct=5.0,
            realized_expansion_pct=8.0,
            source_type="replay",
            observation_date=obs_date,
            observation_id=f"obs_{i}",
        )

    # as_of_date after all observations → data visible
    result_after = store.get_prior_dict("atm_straddle", as_of_date=datetime.date(2025, 1, 1))
    assert result_after is not None
    assert result_after["history_count"] == MIN_OBS_FOR_OVERRIDE

    # as_of_date before all observations → no data, returns None
    result_before = store.get_prior_dict("atm_straddle", as_of_date=datetime.date(2024, 1, 1))
    assert result_before is None


# ── Test 2: IVExpansionCalibration.apply() accepts as_of_date ─────────────────


def test_calibration_apply_accepts_as_of_date(tmp_path):
    """apply() accepts as_of_date so walk-forward evaluations can restrict the
    calibration curve to observations recorded before a backtest cutoff.

    Observations added with observation_date=past must not be counted when
    as_of_date is strictly before that date.
    """
    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    obs_date = datetime.date(2024, 6, 1)
    cal.update(0.70, 12.0, observation_id="trade_A", source_type="replay",
               observation_date=obs_date)

    # as_of_date after observation → includes it
    result_after = cal.apply(0.70, as_of_date=datetime.date(2025, 1, 1))
    assert result_after is not None
    assert result_after["n_observations"] == 1

    # as_of_date before observation → excludes it
    result_before = cal.apply(0.70, as_of_date=datetime.date(2024, 1, 1))
    assert result_before is not None
    assert result_before["n_observations"] == 0


# ── Test 3: StructurePriorStore persists per-observation timestamps ────────────


def test_structure_prior_store_persists_observation_timestamps(tmp_path):
    """StructurePriorStore must persist an individual observation_date alongside
    each observation so that as_of_date filtering can be implemented.
    """
    store_path = tmp_path / "priors.json"
    store = StructurePriorStore(store_path=store_path)
    obs_date = datetime.date(2024, 6, 1)
    for i in range(MIN_OBS_FOR_OVERRIDE):
        store.update(
            structure="atm_straddle",
            realized_return_pct=5.0,
            realized_expansion_pct=8.0,
            source_type="replay",
            observation_date=obs_date,
            observation_id=f"obs_{i}",
        )

    raw = json.loads(store_path.read_text())
    entry = raw["structures"]["atm_straddle"]
    assert "observations" in entry, (
        "StructurePriorStore must persist an 'observations' list in the JSON"
    )
    obs_list = entry["observations"]
    assert len(obs_list) == MIN_OBS_FOR_OVERRIDE
    for obs in obs_list:
        assert "observation_date" in obs, (
            f"Each observation must carry 'observation_date'; got keys: {list(obs.keys())}"
        )


# ── Test 4: IVExpansionCalibration persists per-observation timestamps ─────────


def test_calibration_persists_observation_timestamps(tmp_path):
    """IVExpansionCalibration must persist a timestamps list parallel to
    scores[] and expansions[] so that as_of_date filtering can be implemented.
    """
    store_path = tmp_path / "cal.json"
    cal = IVExpansionCalibration(store_path=store_path)
    cal.update(0.70, 12.0, observation_id="trade_A", source_type="replay",
               observation_date=datetime.date(2024, 6, 1))
    cal.update(0.55, 7.5, observation_id="trade_B", source_type="replay",
               observation_date=datetime.date(2024, 7, 1))

    raw = json.loads(store_path.read_text())
    assert "timestamps" in raw, (
        f"IVExpansionCalibration JSON must have a 'timestamps' list; "
        f"actual keys: {list(raw.keys())}"
    )
    assert len(raw["timestamps"]) == len(raw["scores"]), (
        "timestamps[] must be the same length as scores[]"
    )


# ── V3 Phase 1.1: schema_version=1 migration preserves aggregate ───────────────


def test_structure_prior_store_v1_migration_preserves_aggregate(tmp_path):
    """Loading a schema_version=1 JSON must produce identical aggregate results
    via the no-filter read path (no as_of_date passed).

    The migration must be backward-compatible so production callers that
    don't pass as_of_date see exactly the same values as before.
    """
    store_path = tmp_path / "priors.json"
    v1_payload = {
        "schema_version": 1,
        "structures": {
            "atm_straddle": {
                "structure": "atm_straddle",
                "observation_count": 7,
                "positive_count": 5,
                "sum_return_pct": 35.0,
                "sum_expansion_pct": 56.0,
                "win_rate": 0.7142857142857143,
                "avg_return_pct": 5.0,
                "avg_realized_expansion_pct": 8.0,
                "rank_score": 0.62,
                "source_types": {"replay": 7},
                "last_updated": "2024-01-15T10:00:00+00:00",
            }
        },
    }
    store_path.write_text(json.dumps(v1_payload, indent=2))

    store = StructurePriorStore(store_path=store_path)
    result = store.get_prior_dict("atm_straddle")
    assert result is not None
    assert result["history_count"] == 7
    assert abs(result["avg_return_pct"] - 5.0) < 0.001
    assert abs(result["win_rate"] - 0.7142857142857143) < 0.001


# ── V3 Phase 1.2: as_of_date filter actually changes the estimate ──────────────


def test_calibration_as_of_date_changes_behavior(tmp_path):
    """as_of_date filtering must actually exclude future observations and
    produce a different estimate when early vs. late observations differ.
    """
    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    early_date = datetime.date(2023, 1, 1)
    late_date = datetime.date(2025, 1, 1)

    # 50 observations: first 25 early with high expansion, last 25 late with low
    for i in range(50):
        obs_date = early_date if i < 25 else late_date
        cal.update(
            0.80,
            20.0 if i < 25 else 2.0,
            observation_id=f"obs_{i}",
            source_type="replay",
            observation_date=obs_date,
        )

    result_early = cal.apply(0.80, as_of_date=datetime.date(2024, 1, 1))
    result_late = cal.apply(0.80, as_of_date=datetime.date(2025, 12, 31))

    assert result_early["n_observations"] == 25
    assert result_late["n_observations"] == 50
    # Early-only estimate (expansion≈20) must exceed late+early mix (avg≈11)
    assert result_early["expected_expansion_pct"] > result_late["expected_expansion_pct"], (
        f"as_of filter should change estimate: "
        f"early={result_early['expected_expansion_pct']}, late={result_late['expected_expansion_pct']}"
    )


# ── V3 Phase 1.3: build_structure_scorecards plumbs as_of_date ────────────────


def test_build_structure_scorecards_as_of_date_uses_filtered_prior(tmp_path):
    """build_structure_scorecards with as_of_date must pass the cutoff through
    to walk-forward prior loading so only observations before the cutoff
    affect the scorecard's rank_score.
    """
    from services.structure_scorecard import build_structure_scorecards
    from services.earnings_vol_snapshot import VolSnapshot
    import services.structure_prior_store as _sps_module

    snapshot = VolSnapshot(
        symbol="AAPL",
        as_of_date=date(2024, 6, 15),
        earnings_date=date(2024, 6, 23),
        release_timing="after market close",
        days_to_earnings=8,
        underlying_price=100.0,
        option_source="provided",
        underlying_source="provided",
        price_staleness_minutes=5,
        chain_staleness_minutes=5,
        data_quality="high",
        data_quality_score=0.92,
        rv30_yang_zhang=0.21,
        rv30_estimator="yang_zhang",
        rv_har_forecast=0.20,
        rv_percentile_rank=48.0,
        vol_regime_label="Normal",
        iv30=0.24,
        iv45=0.27,
        near_term_dte=4,
        near_term_atm_iv=0.23,
        back_term_dte=25,
        back_term_atm_iv=0.27,
        near_back_iv_ratio=0.87,
        term_structure_slope=0.0025,
        near_term_implied_move_pct=5.8,
        near_term_implied_sigma_pct=7.27,
        non_event_move_pct_har=1.2,
        event_implied_move_pct=5.67,
        event_move_share_of_total=0.88,
        historical_event_count=8,
        historical_median_move_pct=7.2,
        historical_avg_last4_move_pct=7.6,
        historical_p90_move_pct=10.0,
        historical_move_std_pct=2.1,
        historical_move_anchor_pct=7.4,
        historical_move_uncertainty_pct=0.85,
        historical_vs_implied_move_ratio=1.28,
        tail_vs_implied_move_ratio=1.72,
        smile_curvature=0.18,
        smile_concavity_flag=False,
        smile_points=7,
        near_term_spread_pct=2.5,
        near_term_liquidity_proxy=4200.0,
        atm_call_spread_pct=2.4,
        atm_put_spread_pct=2.6,
        atm_total_open_interest=3200.0,
        atm_total_volume=1600.0,
        liquidity_tier="high",
        iv_rv_yz=1.14,
        iv_rv_har=1.20,
        cheapness_score=0.62,
        event_risk_score=0.74,
        execution_score=0.86,
        timing_score=0.78,
        historical_move_source="earnings_history",
        null_reasons={},
    )

    past_date = date(2024, 1, 1)
    future_date = date(2025, 1, 1)
    cutoff = date(2024, 3, 1)

    prior_store = StructurePriorStore(store_path=tmp_path / "priors.json")
    # Early (profitable) observations before cutoff
    for i in range(MIN_OBS_FOR_OVERRIDE):
        prior_store.update(
            structure="atm_straddle",
            realized_return_pct=10.0,
            realized_expansion_pct=15.0,
            source_type="replay",
            observation_date=past_date,
            observation_id=f"early_{i}",
        )
    # Late (unprofitable) observations after cutoff
    for i in range(MIN_OBS_FOR_OVERRIDE):
        prior_store.update(
            structure="atm_straddle",
            realized_return_pct=-5.0,
            realized_expansion_pct=-3.0,
            source_type="replay",
            observation_date=future_date,
            observation_id=f"late_{i}",
        )

    with patch.object(_sps_module, "get_structure_prior_store", return_value=prior_store):
        cards_historical = build_structure_scorecards(snapshot, as_of_date=cutoff)
        cards_full = build_structure_scorecards(snapshot, as_of_date=future_date)

    straddle_hist = next(c for c in cards_historical if c.structure == "atm_straddle")
    straddle_full = next(c for c in cards_full if c.structure == "atm_straddle")

    # Historical cutoff → only profitable observations visible → higher rank_score
    assert straddle_hist.walk_forward_rank_score > straddle_full.walk_forward_rank_score, (
        f"historical cutoff should yield higher rank_score than full view; "
        f"historical={straddle_hist.walk_forward_rank_score:.3f}, "
        f"full={straddle_full.walk_forward_rank_score:.3f}"
    )


# ── Phase 1.5: leakage sentinel in build_structure_scorecards ─────────────────


def _make_sentinel_store(tmp_path, source_type: str, obs_date: date) -> StructurePriorStore:
    """Build a store with one observation of the given source_type and date."""
    store = StructurePriorStore(store_path=tmp_path / "priors.json")
    store.update(
        structure="atm_straddle",
        realized_return_pct=5.0,
        realized_expansion_pct=8.0,
        source_type=source_type,
        observation_date=obs_date,
        observation_id=f"sentinel_{source_type}",
    )
    return store


def test_leakage_sentinel_fires_for_future_paper_observation(tmp_path):
    """check_for_leakage raises BacktestLeakageError for a paper observation
    dated after as_of_date.  Paper trades are real and must not influence a
    backtest set before they occurred.
    """
    future_obs = date(2025, 6, 1)
    as_of = date(2024, 1, 1)
    store = _make_sentinel_store(tmp_path, "paper", future_obs)
    with pytest.raises(BacktestLeakageError, match="future non-replay"):
        store.check_for_leakage(as_of)


def test_leakage_sentinel_fires_for_future_live_observation(tmp_path):
    """check_for_leakage raises BacktestLeakageError for a live observation
    dated after as_of_date.
    """
    future_obs = date(2025, 6, 1)
    as_of = date(2024, 1, 1)
    store = _make_sentinel_store(tmp_path, "live", future_obs)
    with pytest.raises(BacktestLeakageError, match="future non-replay"):
        store.check_for_leakage(as_of)


def test_leakage_sentinel_exempts_replay_observations(tmp_path):
    """check_for_leakage must NOT raise for replay observations dated after
    as_of_date.  Replays are the sanctioned backtest signal and are expected
    to cover the full date range.
    """
    future_obs = date(2025, 6, 1)
    as_of = date(2024, 1, 1)
    store = _make_sentinel_store(tmp_path, "replay", future_obs)
    store.check_for_leakage(as_of)  # must not raise


def test_leakage_sentinel_integrated_in_build_structure_scorecards(tmp_path):
    """build_structure_scorecards propagates BacktestLeakageError when called
    with as_of_date and the store has a future paper observation.
    """
    import services.structure_prior_store as _sps_module
    from services.structure_scorecard import build_structure_scorecards
    from services.earnings_vol_snapshot import VolSnapshot

    snapshot = VolSnapshot(
        symbol="TEST",
        as_of_date=date(2024, 1, 1),
        earnings_date=date(2024, 1, 15),
        release_timing="after market close",
        days_to_earnings=14,
        underlying_price=100.0,
        option_source="provided",
        underlying_source="provided",
        price_staleness_minutes=5,
        chain_staleness_minutes=5,
        data_quality="high",
        data_quality_score=0.90,
        rv30_yang_zhang=0.20,
        rv30_estimator="yang_zhang",
        rv_har_forecast=0.19,
        rv_percentile_rank=45.0,
        vol_regime_label="Normal",
        iv30=0.23,
        iv45=0.26,
        near_term_dte=4,
        near_term_atm_iv=0.22,
        back_term_dte=25,
        back_term_atm_iv=0.26,
        near_back_iv_ratio=0.86,
        term_structure_slope=0.0024,
        near_term_implied_move_pct=5.5,
        near_term_implied_sigma_pct=7.0,
        non_event_move_pct_har=1.1,
        event_implied_move_pct=5.4,
        event_move_share_of_total=0.87,
        historical_event_count=8,
        historical_median_move_pct=7.0,
        historical_avg_last4_move_pct=7.4,
        historical_p90_move_pct=9.8,
        historical_move_std_pct=2.0,
        historical_move_anchor_pct=7.2,
        historical_move_uncertainty_pct=0.84,
        historical_vs_implied_move_ratio=1.25,
        tail_vs_implied_move_ratio=1.70,
        smile_curvature=0.17,
        smile_concavity_flag=False,
        smile_points=7,
        near_term_spread_pct=2.4,
        near_term_liquidity_proxy=4000.0,
        atm_call_spread_pct=2.3,
        atm_put_spread_pct=2.5,
        atm_total_open_interest=3000.0,
        atm_total_volume=1500.0,
        liquidity_tier="high",
        iv_rv_yz=1.12,
        iv_rv_har=1.18,
        cheapness_score=0.60,
        event_risk_score=0.72,
        execution_score=0.84,
        timing_score=0.76,
        historical_move_source="earnings_history",
        null_reasons={},
    )

    future_obs = date(2025, 6, 1)
    store = _make_sentinel_store(tmp_path, "paper", future_obs)

    with patch.object(_sps_module, "get_structure_prior_store", return_value=store):
        with pytest.raises(BacktestLeakageError):
            build_structure_scorecards(snapshot, as_of_date=date(2024, 1, 1))
