"""The return contract: which return each evidence figure and each prior means.

``realized_return_pct`` is the mid return minus the selector's own modeled
cost. The claim contract prices a claim at the conservative fill, stored in
the exit scenarios. Before this, the evidence report's H1/H2 analysis, the
"Claimable evidence" table and the structure priors all read
``realized_return_pct``, so a claimable outcome was reported (and learned) at
a return the claim contract never made.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from scripts.seed_outcomes_from_replay import REPLAY_RETURN_BASIS, seed_from_trades
from services.claim_contract import CONTRACT_VERSION
from services.evidence_report import _claimed_evidence
from services.forward_performance_diagnostics import _claimable_performance
from services.outcome_recorder import OutcomeStore, finalize_trade_and_update_learning
from services.outcome_returns import (
    PRIOR_LEARNING_RETURN_BASIS,
    RETURN_CLAIM,
    RETURN_MID,
    RETURN_MODELED_NET,
    claimed_return_pct,
    outcome_returns,
    return_pct,
)
from services.structure_prior_store import StructurePriorStore
from tests.unit.outcome_fixtures import claimed_exit_scenarios

# ── reading the named returns ───────────────────────────────────────────────


def _row(modeled, conservative=None, *, mid=None, claimable=True, rec="rec-1", **extra):
    scenarios = {"scenario_outcomes": {"realized_return_pct": {"mid": mid, RETURN_CLAIM: conservative}}}
    return {
        "recommendation_id": rec,
        "realized_return_pct": modeled,
        "exit_execution_scenarios_json": scenarios,
        "claim_allowed": claimable,
        "claim_contract_version": CONTRACT_VERSION,
        "status": "finalized",
        **extra,
    }


def test_each_basis_reads_its_own_number() -> None:
    row = _row(12.0, -3.0, mid=15.0)
    assert outcome_returns(row) == {RETURN_MID: 15.0, RETURN_MODELED_NET: 12.0, RETURN_CLAIM: -3.0}


def test_stored_json_strings_and_the_unparsed_column_name_both_read() -> None:
    payload = {"scenario_outcomes": {"realized_return_pct": {RETURN_CLAIM: 4.5}}}
    assert claimed_return_pct({"exit_execution_scenarios_json": json.dumps(payload)}) == 4.5
    assert claimed_return_pct({"exit_execution_scenarios": payload}) == 4.5


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, "x"])
def test_an_unusable_conservative_fill_is_no_return(value) -> None:
    assert claimed_return_pct(_row(10.0, value)) is None


@pytest.mark.parametrize("stored", [None, "", "{not json", "[]", {"scenario_outcomes": "x"}])
def test_missing_or_malformed_scenarios_are_no_return(stored) -> None:
    assert claimed_return_pct({"exit_execution_scenarios_json": stored}) is None


def test_the_modeled_return_never_stands_in_for_the_claimed_one() -> None:
    assert claimed_return_pct(_row(25.0, None)) is None


def test_an_unknown_basis_is_refused() -> None:
    with pytest.raises(ValueError):
        return_pct(_row(1.0, 1.0), "net")


# ── the pre-registered analysis ─────────────────────────────────────────────


def test_claimed_evidence_uses_the_conservative_fill_of_claimable_rows_only() -> None:
    selected = [
        _row(20.0, -5.0, rec="a"),
        _row(30.0, 1.0, rec="b"),
        _row(50.0, 40.0, rec="c", claimable=False),          # observational only
        _row(9.0, 9.0, rec="d", claim_contract_version="old"),  # another contract
    ]
    claimed = _claimed_evidence(selected, [], [])

    assert claimed["return_basis"] == RETURN_CLAIM
    assert claimed["selector_claimable_n"] == 2
    assert claimed["selector"]["n"] == 2
    assert claimed["selector"]["mean"] == pytest.approx(-2.0)  # (-5 + 1) / 2, not (20 + 30) / 2


def test_a_claimable_row_without_a_conservative_fill_is_withheld_and_counted() -> None:
    claimed = _claimed_evidence([_row(20.0, 3.0, rec="a"), _row(80.0, None, rec="b")], [], [])
    assert claimed["selector_withheld_count"] == 1
    assert (claimed["selector"]["n"], claimed["selector"]["mean"]) == (1, 3.0)


def test_h2_pairs_conservative_fills_and_needs_both_sides_claimable() -> None:
    selected = [_row(20.0, 4.0, rec="a"), _row(20.0, 6.0, rec="b")]
    baselines = [
        _row(1.0, 1.0, rec="a", baseline_name="always_atm_straddle"),
        _row(1.0, 9.0, rec="b", baseline_name="always_atm_straddle", claimable=False),
    ]
    paired = _claimed_evidence(selected, baselines, [])["selector_minus_baseline"]["always_atm_straddle"]
    assert (paired["pairs"], paired["mean"]) == (1, 3.0)  # only event a: 4 - 1


def test_the_universe_comparison_uses_claimable_conservative_fills() -> None:
    universe = [
        _row(99.0, 2.0, rec="p", baseline_name="always_atm_straddle", cohort="universe",
             selector_recommendation="Candidate", status="resolved", exit_repricing="booked_entry_contracts"),
        _row(-99.0, -1.0, rec="s", baseline_name="always_atm_straddle", cohort="universe",
             selector_recommendation="No Trade", status="resolved", exit_repricing="booked_entry_contracts"),
    ]
    result = _claimed_evidence([], [], universe)["universe_picked_minus_skipped"]["always_atm_straddle"]
    assert result["mean"] == pytest.approx(3.0)


# ── the "Claimable evidence" table ──────────────────────────────────────────


def _diag_row(modeled, conservative, claimable):
    return {"realized_return_pct": modeled, "modeled_return_pct": 1.0, "claim_allowed": claimable,
            "execution_scenario_returns": {"mid": modeled, RETURN_CLAIM: conservative}}


def test_the_claimable_group_reports_the_conservative_fill_and_says_so() -> None:
    groups = _claimable_performance([
        _diag_row(10.0, -2.0, True),
        _diag_row(12.0, None, True),
        _diag_row(-4.0, -6.0, False),
    ])
    claimable = groups["claimable"]
    assert claimable["return_basis"] == RETURN_CLAIM
    assert (claimable["n"], claimable["withheld_count"], claimable["avg_realized_return_pct"]) == (1, 1, -2.0)
    assert claimable["wins"] == 0  # a 10% modeled return that lost money at the claimed fill
    assert groups["non_claimable"]["return_basis"] == RETURN_MODELED_NET
    assert groups["non_claimable"]["avg_realized_return_pct"] == -4.0


# ── what the priors learn ───────────────────────────────────────────────────


@pytest.fixture()
def learning(tmp_path, monkeypatch):
    import services.calibration_service as calibration_service
    import services.structure_prior_store as structure_prior_store
    from services.calibration_service import IVExpansionCalibration

    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    priors = StructurePriorStore(tmp_path / "priors.json")
    monkeypatch.setattr(calibration_service, "get_calibration", lambda: cal)
    monkeypatch.setattr(structure_prior_store, "get_structure_prior_store", lambda: priors)
    store = OutcomeStore(tmp_path / "o.sqlite")
    store.insert_entry(trade_id="AAA", symbol="AAA", structure="atm_straddle", entry_date=date(2026, 4, 20),
                       earnings_date=date(2026, 4, 28), setup_score=0.6, source_type="paper", entry_mid=2.0)
    return store, tmp_path / "priors.json"


def _observations(path):
    raw = json.loads(path.read_text())
    return [o for entry in raw["structures"].values() for o in entry["observations"]]


def test_priors_learn_the_conservative_fill_and_record_its_basis(learning) -> None:
    store, priors_path = learning
    result = finalize_trade_and_update_learning(
        trade_id="AAA", exit_date=date(2026, 4, 27), realized_return_pct=15.0, realized_expansion_pct=15.0,
        store=store, exit_execution_scenarios=claimed_exit_scenarios(15.0, conservative=-4.0),
    )
    [obs] = _observations(priors_path)
    assert (obs["realized_return_pct"], obs["return_basis"]) == (-4.0, PRIOR_LEARNING_RETURN_BASIS)
    assert store.get_trade("AAA")["realized_return_pct"] == 15.0  # the row keeps the modeled-net figure
    assert (result["prior_learning_return_pct"], result["prior_learning_withheld"]) == (-4.0, False)
    assert StructurePriorStore(priors_path).diagnostics()["structures"]["atm_straddle"]["return_bases"] == {
        RETURN_CLAIM: 1}


def test_no_conservative_fill_means_no_prior_observation(learning) -> None:
    store, priors_path = learning
    result = finalize_trade_and_update_learning(
        trade_id="AAA", exit_date=date(2026, 4, 27), realized_return_pct=15.0, realized_expansion_pct=15.0,
        store=store,
    )
    assert result["status"] == "finalized" and result["prior_learning_withheld"] is True
    assert any("structure prior not updated" in warning for warning in result["warnings"])
    assert not priors_path.exists() or _observations(priors_path) == []


def test_observations_without_a_recorded_basis_are_counted_as_unrecorded(tmp_path) -> None:
    priors = StructurePriorStore(tmp_path / "priors.json")
    priors.update(structure="atm_straddle", realized_return_pct=3.0, realized_expansion_pct=3.0,
                  observation_id="legacy")
    priors.update(structure="atm_straddle", realized_return_pct=1.0, realized_expansion_pct=1.0,
                  observation_id="new", return_basis=RETURN_CLAIM)
    bases = StructurePriorStore(tmp_path / "priors.json").diagnostics()["structures"]["atm_straddle"]["return_bases"]
    assert bases == {RETURN_CLAIM: 1, "unrecorded": 1}


def test_replay_seeds_record_their_own_basis(tmp_path) -> None:
    row = {"pricing_source": "snapshot_replay", "trade_date": "2026-04-21", "event_date": "2026-04-28",
           "setup_score": 0.6, "gross_return_pct": 0.9, "net_return_pct": 0.8, "symbol": "AAA",
           "structure": "otm_strangle"}
    seed_from_trades([row], structure="otm_strangle", dry_run=False, outcome_store_path=tmp_path / "o.sqlite",
                     calibration_store_path=tmp_path / "cal.json", prior_store_path=tmp_path / "priors.json")
    [obs] = _observations(tmp_path / "priors.json")
    assert obs["return_basis"] == REPLAY_RETURN_BASIS
