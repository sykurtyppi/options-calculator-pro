from __future__ import annotations

from pathlib import Path

from services.calibration_service import IVExpansionCalibration
from services.learning_diagnostics import build_learning_diagnostics
from services.structure_prior_store import StructurePriorStore


def _seed_prior(
    store: StructurePriorStore,
    *,
    structure: str,
    count: int,
    source_type: str,
    return_pct: float = 5.0,
    expansion_pct: float = 7.0,
) -> None:
    for _ in range(count):
        store.update(
            structure=structure,
            realized_return_pct=return_pct,
            realized_expansion_pct=expansion_pct,
            source_type=source_type,
        )


def test_empty_learning_diagnostics(monkeypatch, tmp_path: Path) -> None:
    cal = IVExpansionCalibration(store_path=tmp_path / "calibration.json")
    priors = StructurePriorStore(store_path=tmp_path / "priors.json")

    monkeypatch.setattr("services.learning_diagnostics.get_calibration", lambda: cal)
    monkeypatch.setattr("services.learning_diagnostics.get_structure_prior_store", lambda: priors)

    payload = build_learning_diagnostics()

    assert payload["calibration"]["n_total"] == 0
    assert payload["data_quality"]["has_real_data"] is False
    assert payload["data_quality"]["synthetic_ratio"] == 0.0
    assert payload["learning_health"]["sufficient_observations"] is False
    assert "Calibration in bootstrap phase" in payload["learning_health"]["warning_flags"]
    assert "No real forward observations yet" in payload["learning_health"]["warning_flags"]


def test_replay_only_learning_diagnostics(monkeypatch, tmp_path: Path) -> None:
    cal = IVExpansionCalibration(store_path=tmp_path / "calibration.json")
    priors = StructurePriorStore(store_path=tmp_path / "priors.json")
    for idx in range(30):
        cal.update(
            0.40 + idx * 0.01,
            4.0 + idx * 0.2,
            observation_id=f"replay-{idx}",
            source_type="replay",
        )
    _seed_prior(priors, structure="call_calendar", count=6, source_type="replay")

    monkeypatch.setattr("services.learning_diagnostics.get_calibration", lambda: cal)
    monkeypatch.setattr("services.learning_diagnostics.get_structure_prior_store", lambda: priors)

    payload = build_learning_diagnostics()

    assert payload["data_quality"]["has_real_data"] is True
    assert payload["data_quality"]["replay_dominant"] is True
    assert payload["calibration"]["n_replay"] == 30
    assert payload["calibration"]["is_prior_only"] is True
    assert payload["structure_priors"]["call_calendar"]["count"] == 6
    assert "No real forward observations yet" in payload["learning_health"]["warning_flags"]


def test_mixed_replay_and_paper_learning_diagnostics(monkeypatch, tmp_path: Path) -> None:
    cal = IVExpansionCalibration(store_path=tmp_path / "calibration.json")
    priors = StructurePriorStore(store_path=tmp_path / "priors.json")
    for idx in range(80):
        cal.update(0.30 + idx * 0.002, 3.0 + idx * 0.1, observation_id=f"replay-{idx}", source_type="replay")
    for idx in range(50):
        cal.update(0.45 + idx * 0.002, 5.0 + idx * 0.1, observation_id=f"paper-{idx}", source_type="paper")
    _seed_prior(priors, structure="atm_straddle", count=12, source_type="paper")
    _seed_prior(priors, structure="otm_strangle", count=9, source_type="replay")

    monkeypatch.setattr("services.learning_diagnostics.get_calibration", lambda: cal)
    monkeypatch.setattr("services.learning_diagnostics.get_structure_prior_store", lambda: priors)

    payload = build_learning_diagnostics()

    assert payload["calibration"]["n_total"] == 130
    assert payload["learning_health"]["sufficient_observations"] is True
    assert payload["data_quality"]["paper_ratio"] > 0.0
    assert payload["data_quality"]["replay_dominant"] is False
    assert payload["learning_health"]["calibration_stable"] is False
    assert "No real forward observations yet" not in payload["learning_health"]["warning_flags"]


def test_synthetic_heavy_learning_diagnostics(monkeypatch, tmp_path: Path) -> None:
    cal = IVExpansionCalibration(store_path=tmp_path / "calibration.json")
    priors = StructurePriorStore(store_path=tmp_path / "priors.json")
    for idx in range(70):
        cal.update(0.35 + idx * 0.002, 2.0 + idx * 0.05, observation_id=f"synthetic-{idx}", source_type="synthetic")
    for idx in range(10):
        cal.update(0.50 + idx * 0.003, 4.0 + idx * 0.1, observation_id=f"paper-{idx}", source_type="paper")
    _seed_prior(priors, structure="put_calendar", count=3, source_type="synthetic")

    monkeypatch.setattr("services.learning_diagnostics.get_calibration", lambda: cal)
    monkeypatch.setattr("services.learning_diagnostics.get_structure_prior_store", lambda: priors)

    payload = build_learning_diagnostics()

    assert payload["data_quality"]["synthetic_ratio"] > 0.5
    assert "Calibration dominated by synthetic data" in payload["learning_health"]["warning_flags"]
    assert payload["learning_health"]["sufficient_observations"] is False
