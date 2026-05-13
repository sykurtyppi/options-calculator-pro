"""
Phase 1.6 leakage-detection tests — walk-forward data contamination.

Each test in this module documents a specific way that future observations
leak into walk-forward backtest evaluations today.  They are marked
xfail(strict=False) so the suite stays green while the bug is unfixed.
They will be un-xfailed as the corresponding Phase 1.x fix lands.

Fix tracked in: https://github.com/sykurtyppi/options-calculator-pro/issues/18
Related: issue #12 (production-side StructurePriorStore refactor)
"""

import datetime
import json

import pytest

from services.calibration_service import IVExpansionCalibration
from services.structure_prior_store import StructurePriorStore, MIN_OBS_FOR_OVERRIDE

_XFAIL_REASON = (
    "walk-forward leakage — fix tracked in Phase 1 of architectural review, see #18"
)


# ── Test 1: StructurePriorStore.get_prior_dict() must accept as_of_date ───────


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_structure_prior_store_get_prior_dict_accepts_as_of_date(tmp_path):
    """get_prior_dict() must accept an as_of_date parameter so walk-forward
    evaluations can freeze the prior at a backtest cutoff date.

    Today it raises TypeError because the parameter does not exist.
    """
    store = StructurePriorStore(store_path=tmp_path / "priors.json")
    for _ in range(MIN_OBS_FOR_OVERRIDE):
        store.update(
            structure="atm_straddle",
            realized_return_pct=5.0,
            realized_expansion_pct=8.0,
            source_type="replay",
        )

    cutoff = datetime.date(2024, 1, 1)
    # Raises TypeError today: unexpected keyword argument 'as_of_date'
    result = store.get_prior_dict("atm_straddle", as_of_date=cutoff)
    assert result is not None


# ── Test 2: IVExpansionCalibration.apply() must accept as_of_date ─────────────


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_calibration_apply_accepts_as_of_date(tmp_path):
    """apply() must accept an as_of_date parameter so walk-forward evaluations
    can restrict the calibration curve to observations recorded before a
    backtest cutoff date.

    Today it raises TypeError because the parameter does not exist.
    """
    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    cal.update(0.70, 12.0, observation_id="trade_A", source_type="replay")

    cutoff = datetime.date(2024, 1, 1)
    # Raises TypeError today: unexpected keyword argument 'as_of_date'
    result = cal.apply(0.70, as_of_date=cutoff)
    assert result is not None


# ── Test 3: StructurePriorStore must persist per-observation timestamps ────────


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_structure_prior_store_persists_observation_timestamps(tmp_path):
    """StructurePriorStore must store an individual timestamp alongside each
    observation so that as_of_date filtering (Phase 1.2) can be implemented.

    Today the JSON schema stores only aggregate accumulators
    (sum_return_pct, observation_count, …) with a single entry-level
    last_updated timestamp.  Individual observations are not retained, so
    as_of_date filtering is structurally impossible without a schema change.
    """
    store_path = tmp_path / "priors.json"
    store = StructurePriorStore(store_path=store_path)
    for _ in range(MIN_OBS_FOR_OVERRIDE):
        store.update(
            structure="atm_straddle",
            realized_return_pct=5.0,
            realized_expansion_pct=8.0,
            source_type="replay",
        )

    raw = json.loads(store_path.read_text())
    entry = raw["structures"]["atm_straddle"]
    # Must have a per-observation list with a timestamp on each entry.
    # Today this key does not exist — only aggregate fields are stored.
    assert "observations" in entry, (
        "StructurePriorStore must persist individual observations with timestamps "
        "so that as_of_date filtering is possible; only aggregate fields found"
    )
    obs_list = entry["observations"]
    assert len(obs_list) == MIN_OBS_FOR_OVERRIDE
    for obs in obs_list:
        assert "observation_date" in obs, (
            f"Each observation must carry an observation_date; got keys: {list(obs.keys())}"
        )


# ── Test 4: IVExpansionCalibration must persist per-observation timestamps ─────


@pytest.mark.xfail(strict=False, reason=_XFAIL_REASON)
def test_calibration_persists_observation_timestamps(tmp_path):
    """IVExpansionCalibration must store a timestamp alongside each
    (score, expansion) observation so that as_of_date filtering (Phase 1.4)
    can be implemented.

    Today the JSON schema stores parallel lists scores[], expansions[], and
    sources[] but no corresponding timestamps[] list.  Without timestamps,
    as_of_date filtering is structurally impossible.
    """
    store_path = tmp_path / "cal.json"
    cal = IVExpansionCalibration(store_path=store_path)
    cal.update(0.70, 12.0, observation_id="trade_A", source_type="replay")
    cal.update(0.55, 7.5, observation_id="trade_B", source_type="replay")

    raw = json.loads(store_path.read_text())
    # Must have a timestamps list parallel to scores[] and expansions[].
    # Today this key is absent from the persisted payload.
    assert "timestamps" in raw, (
        "IVExpansionCalibration must persist a timestamps[] list parallel to "
        "scores[] and expansions[] so that as_of_date filtering is possible; "
        f"actual keys: {list(raw.keys())}"
    )
    assert len(raw["timestamps"]) == len(raw["scores"]), (
        "timestamps[] must be the same length as scores[]"
    )
