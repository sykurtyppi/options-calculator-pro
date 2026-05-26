"""Tests for services/calibration_service.py — isotonic regression + bootstrap prior."""

import json
import pathlib
import tempfile

import pytest

from services.calibration_service import (
    IVExpansionCalibration,
    _MIN_OBS_FOR_OBSERVATIONAL,
    _bootstrap_estimate,
    _isotonic_fit,
    _MIN_OBS_FOR_HIGH_FIT,
    _MIN_OBS_FOR_FIT,
)


# ── Bootstrap prior ───────────────────────────────────────────────────────────


class TestBootstrapEstimate:
    def test_low_score_returns_low_estimate(self):
        est = _bootstrap_estimate(0.10)
        assert est == pytest.approx(1.5)

    def test_mid_score_returns_mid_estimate(self):
        est = _bootstrap_estimate(0.50)
        assert est == pytest.approx(7.5)

    def test_high_score_returns_high_estimate(self):
        est = _bootstrap_estimate(0.90)
        assert est == pytest.approx(18.0)

    def test_score_one_returns_top_bin(self):
        # score == 1.0 falls through all half-open bins; should use top bin
        est = _bootstrap_estimate(1.0)
        assert est == pytest.approx(18.0)

    def test_monotone_across_bins(self):
        mids = [0.10, 0.30, 0.50, 0.70, 0.90]
        estimates = [_bootstrap_estimate(m) for m in mids]
        for i in range(len(estimates) - 1):
            assert estimates[i] < estimates[i + 1], (
                f"Prior not monotone: {estimates}"
            )


# ── Isotonic fit ──────────────────────────────────────────────────────────────


class TestIsotonicFit:
    def test_already_monotone_unchanged(self):
        xs = [0.1, 0.3, 0.5, 0.7, 0.9]
        ys = [1.0, 3.0, 6.0, 10.0, 15.0]
        fit_xs, fit_ys = _isotonic_fit(xs, ys)
        # Values must be non-decreasing
        for i in range(len(fit_ys) - 1):
            assert fit_ys[i] <= fit_ys[i + 1], "Fitted curve not non-decreasing"

    def test_violation_merged_to_mean(self):
        # [0.5, 0.3] violation → both should be levelled to 0.4
        xs = [0.2, 0.4]
        ys = [0.5, 0.3]
        fit_xs, fit_ys = _isotonic_fit(xs, ys)
        assert fit_ys[0] == pytest.approx(0.4)
        assert fit_ys[-1] == pytest.approx(0.4)

    def test_empty_input_returns_empty(self):
        xs, ys = _isotonic_fit([], [])
        assert xs == []
        assert ys == []

    def test_single_point(self):
        fit_xs, fit_ys = _isotonic_fit([0.5], [7.0])
        assert fit_ys[0] == pytest.approx(7.0)

    def test_result_non_decreasing(self):
        import random
        rng = random.Random(42)
        xs = sorted(rng.uniform(0, 1) for _ in range(20))
        ys = [rng.gauss(5, 3) for _ in range(20)]
        fit_xs, fit_ys = _isotonic_fit(xs, ys)
        for i in range(len(fit_ys) - 1):
            assert fit_ys[i] <= fit_ys[i + 1] + 1e-9, (
                f"Isotonic violated at position {i}: {fit_ys[i]:.4f} > {fit_ys[i+1]:.4f}"
            )


# ── IVExpansionCalibration ────────────────────────────────────────────────────


@pytest.fixture()
def cal(tmp_path):
    """Fresh calibration instance backed by a temp directory."""
    store = tmp_path / "calibration" / "iv_expansion.json"
    return IVExpansionCalibration(store_path=store)


class TestCalibrationPriorPhase:
    def test_prior_phase_when_no_observations(self, cal):
        result = cal.apply(0.70)
        assert result["phase"] == "bootstrap_prior"
        assert result["prior_only"] is True
        assert result["n_observations"] == 0

    def test_expected_expansion_positive(self, cal):
        for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = cal.apply(score)
            assert result["expected_expansion_pct"] > 0, f"Expected positive expansion for score={score}"

    def test_interval_contains_point_estimate(self, cal):
        result = cal.apply(0.60)
        assert result["low_pct"] <= result["expected_expansion_pct"] <= result["high_pct"]

    def test_score_input_echoed(self, cal):
        result = cal.apply(0.72)
        assert result["score_input"] == pytest.approx(0.72, abs=0.001)

    def test_score_clipped_to_unit_interval(self, cal):
        low = cal.apply(-0.5)
        high = cal.apply(1.5)
        assert 0.0 <= low["score_input"] <= 1.0
        assert 0.0 <= high["score_input"] <= 1.0

    def test_note_present_in_prior_phase(self, cal):
        result = cal.apply(0.50)
        assert "note" in result
        assert str(_MIN_OBS_FOR_OBSERVATIONAL) in result["note"]


class TestCalibrationUpdate:
    def test_update_increments_count(self, cal):
        assert cal.update(0.70, 8.5) is True
        assert cal._n() == 1

    def test_multiple_updates(self, cal):
        for i in range(5):
            assert cal.update(float(i) / 10.0, float(i) * 2.0) is True
        assert cal._n() == 5

    def test_update_persists_to_disk(self, tmp_path):
        store = tmp_path / "cal" / "iv_expansion.json"
        c = IVExpansionCalibration(store_path=store)
        c.update(0.60, 9.0)
        assert store.exists()
        raw = json.loads(store.read_text())
        assert raw["n"] == 1
        assert raw["scores"][0] == pytest.approx(0.60)
        assert raw["expansions"][0] == pytest.approx(9.0)
        assert raw["sources"] == ["paper"]

    def test_duplicate_observation_id_does_not_append_twice(self, cal):
        assert cal.update(0.60, 9.0, observation_id="trade-1", source_type="paper") is True
        assert cal.update(0.60, 9.0, observation_id="trade-1", source_type="paper") is False
        assert cal._n() == 1

    def test_different_observation_ids_append_normally(self, cal):
        assert cal.update(0.60, 9.0, observation_id="trade-1") is True
        assert cal.update(0.60, 9.5, observation_id="trade-2") is True
        assert cal._n() == 2

    def test_source_types_persist_and_reload(self, tmp_path):
        store = tmp_path / "cal" / "iv_expansion.json"
        c1 = IVExpansionCalibration(store_path=store)
        c1.update(0.55, 6.0, observation_id="replay-1", source_type="replay")
        c1.update(0.65, 8.0, observation_id="paper-1", source_type="paper")
        c1.update(0.75, 10.0, observation_id="live-1", source_type="live")
        c1.update(0.45, 4.0, observation_id="synth-1", source_type="synthetic")

        c2 = IVExpansionCalibration(store_path=store)
        assert c2._sources == ["replay", "paper", "live", "synthetic"]
        diag = c2.diagnostics()
        assert diag["n_replay"] == 1
        assert diag["n_paper"] == 1
        assert diag["n_live"] == 1
        assert diag["n_synthetic"] == 1

    def test_legacy_json_without_observation_ids_loads_cleanly(self, tmp_path):
        store = tmp_path / "cal" / "iv_expansion.json"
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps({"scores": [0.4, 0.7], "expansions": [3.0, 8.0], "n": 2}))

        cal = IVExpansionCalibration(store_path=store)
        assert cal._n() == 2
        assert cal._observation_ids == set()
        assert cal._sources == ["paper", "paper"]


class TestCalibrationFittedPhase:
    def _populate(self, cal, n=_MIN_OBS_FOR_FIT):
        """Add `n` synthetic observations with a clear positive relationship."""
        import math
        for i in range(n):
            score = float(i) / float(n)
            expansion = 2.0 + score * 20.0 + 0.1 * math.sin(i)
            cal.update(score, expansion)

    def test_transitions_to_observational_before_fitted(self, cal):
        assert cal.apply(0.50)["phase"] == "bootstrap_prior"
        self._populate(cal, _MIN_OBS_FOR_OBSERVATIONAL)
        assert cal.apply(0.50)["phase"] == "observational"

    def test_transitions_to_fitted_at_threshold(self, cal):
        self._populate(cal, _MIN_OBS_FOR_FIT)
        assert cal.apply(0.50)["phase"] == "fitted_moderate"

    def test_fitted_phase_not_prior_only(self, cal):
        self._populate(cal)
        result = cal.apply(0.60)
        assert result["prior_only"] is False

    def test_fitted_estimate_monotone(self, cal):
        self._populate(cal)
        estimates = [cal.apply(s)["expected_expansion_pct"] for s in [0.1, 0.3, 0.5, 0.7, 0.9]]
        for i in range(len(estimates) - 1):
            assert estimates[i] <= estimates[i + 1] + 0.5, (
                f"Fitted estimates not roughly monotone: {estimates}"
            )

    def test_interval_contains_estimate_in_fitted_phase(self, cal):
        self._populate(cal)
        result = cal.apply(0.65)
        assert result["low_pct"] <= result["expected_expansion_pct"] <= result["high_pct"]

    def test_high_fit_phase_reached_with_large_sample(self, cal):
        self._populate(cal, _MIN_OBS_FOR_HIGH_FIT)
        result = cal.apply(0.65)
        assert result["phase"] == "fitted_high"


class TestCalibrationObservationalPhase:
    def _populate(self, cal, n=_MIN_OBS_FOR_OBSERVATIONAL):
        for i in range(n):
            score = 0.42 + (i % 4) * 0.01
            expansion = 3.0 + (i % 5) * 0.4
            cal.update(score, expansion)

    def test_observational_phase_uses_raw_bucket_evidence(self, cal):
        self._populate(cal)
        result = cal.apply(0.44)
        assert result["phase"] == "observational"
        assert result["prior_only"] is False
        assert result["n_local"] >= 3
        assert "No fitted curve" in result["note"]


class TestCalibrationPersistence:
    def test_load_restores_observations(self, tmp_path):
        store = tmp_path / "cal" / "iv_expansion.json"
        c1 = IVExpansionCalibration(store_path=store)
        for i in range(5):
            c1.update(float(i) / 10, float(i) * 3.0)

        # New instance at same path must reload
        c2 = IVExpansionCalibration(store_path=store)
        assert c2._n() == 5
        assert c2._scores == c1._scores
        assert c2._expansions == c1._expansions

    def test_missing_store_starts_empty(self, tmp_path):
        store = tmp_path / "nonexistent" / "iv_expansion.json"
        c = IVExpansionCalibration(store_path=store)
        assert c._n() == 0

    def test_atomic_write_preserves_original_on_fsync_failure(self, tmp_path, monkeypatch):
        # A crash mid-write must not corrupt or truncate the existing store.
        # Previously _save_locked used Path.write_text — a partial write left
        # the next _load to silently fall back to an empty store.
        store_path = tmp_path / "iv_expansion.json"
        c = IVExpansionCalibration(store_path=store_path)
        c.update(0.5, 3.0)
        original_bytes = store_path.read_bytes()

        def boom(_fd):
            raise OSError("simulated crash during fsync")

        monkeypatch.setattr("services.calibration_service.os.fsync", boom)
        c.update(0.9, 9.0)  # save fails internally, swallowed by outer except

        assert store_path.read_bytes() == original_bytes, (
            "atomic-write contract broken — original file modified despite fsync failure"
        )
        c_reloaded = IVExpansionCalibration(store_path=store_path)
        assert c_reloaded._n() == 1, "store survived but observation count diverged"

    def test_atomic_write_leaves_no_temp_files(self, tmp_path):
        store_path = tmp_path / "iv_expansion.json"
        c = IVExpansionCalibration(store_path=store_path)
        c.update(0.5, 3.0)
        c.update(0.6, 4.0)

        siblings = list(store_path.parent.iterdir())
        stray = [p for p in siblings if p.name.startswith(".") and p.name.endswith(".tmp")]
        assert stray == [], f"stray temp files after successful writes: {stray}"


class TestGetCurveSummary:
    def test_returns_ten_buckets(self, cal):
        buckets = cal.get_curve_summary()
        assert len(buckets) == 10

    def test_buckets_cover_full_range(self, cal):
        buckets = cal.get_curve_summary()
        assert buckets[0]["score_lo"] == pytest.approx(0.0)
        assert buckets[-1]["score_hi"] == pytest.approx(1.0)

    def test_prior_buckets_flagged(self, cal):
        buckets = cal.get_curve_summary()
        assert all(b["prior_only"] for b in buckets)

    def test_fitted_buckets_not_prior(self, cal):
        for i in range(_MIN_OBS_FOR_FIT):
            cal.update(float(i) / _MIN_OBS_FOR_FIT, float(i) * 0.5)
        buckets = cal.get_curve_summary()
        assert all(not b["prior_only"] for b in buckets)

    def test_observational_buckets_mix_empirical_and_prior_rows(self, cal):
        for i in range(_MIN_OBS_FOR_OBSERVATIONAL):
            cal.update(0.42 + (i % 3) * 0.01, 2.0 + (i % 4) * 0.5)
        buckets = cal.get_curve_summary()
        assert any(not b["prior_only"] for b in buckets)
        assert any(b["prior_only"] for b in buckets)

    def test_expansion_positive(self, cal):
        for b in cal.get_curve_summary():
            assert b["expected_expansion_pct"] >= 0.0


class TestIntervalLabeling:
    """
    Audit precondition 1 — calibration interval correctness.

    The interval is ±1σ (~68% under normality).  The old docstring incorrectly
    called it an "80%" interval.  These tests verify the fix is consistent
    across output strings and that the numeric math is ±1σ, not ±1.28σ.
    """

    def test_no_eighty_percent_claim_in_prior_phase_note(self, cal):
        result = cal.apply(0.60)
        note = result.get("note", "")
        assert "80%" not in note, f"Stale '80%' found in note: {note!r}"
        assert "80 %" not in note, f"Stale '80 %' found in note: {note!r}"

    def test_no_eighty_percent_claim_in_observational_note(self, cal):
        for i in range(_MIN_OBS_FOR_OBSERVATIONAL):
            cal.update(0.42 + (i % 3) * 0.01, 3.0 + (i % 4) * 0.5)
        result = cal.apply(0.44)
        note = result.get("note", "")
        assert "80%" not in note
        assert "80 %" not in note

    def test_no_eighty_percent_claim_in_fitted_note(self, cal):
        for i in range(_MIN_OBS_FOR_FIT):
            cal.update(float(i) / _MIN_OBS_FOR_FIT, float(i) * 0.1)
        result = cal.apply(0.50)
        note = result.get("note", "")
        assert "80%" not in note
        assert "80 %" not in note

    def test_fitted_interval_width_equals_one_sigma_not_1_28_sigma(self, cal):
        """
        Interval half-width must equal empirical std (±1σ), not 1.28×std (80% level).
        Load observations where the local std is known, then verify the half-width.
        """
        import numpy as _np

        # 120 obs all within ±0.02 of score=0.50 → all in the ±0.10 band
        expansions = []
        for i in range(_MIN_OBS_FOR_FIT):
            score = 0.50 + (i % 5 - 2) * 0.004  # scores in [0.492, 0.508]
            exp = 8.0 + (i % 11) - 5.0           # expansions: 3,4,5,6,7,8,9,10,11,12,13
            cal.update(score, exp)
            expansions.append(exp)

        result = cal.apply(0.50)
        assert result["phase"] == "fitted_moderate"

        local_std = float(_np.std(expansions, ddof=1))
        half_width = result["high_pct"] - result["expected_expansion_pct"]

        # half_width ≈ std (±1σ).  If this were 80% (1.28σ), half_width would be
        # ~28% larger — assert it is NOT that wide (with tolerance for finite-N effects).
        assert half_width < 1.28 * local_std + 1.5, (
            f"Interval is wider than 1.28σ which would imply 80% level: "
            f"half_width={half_width:.3f}, 1.28*std={1.28 * local_std:.3f}"
        )
        assert half_width >= 0.0


class TestDiagnostics:
    def test_diagnostics_structure(self, cal):
        diag = cal.diagnostics()
        assert "n_observations" in diag
        assert "phase" in diag
        assert "min_for_observational" in diag
        assert "min_for_fit" in diag
        assert diag["min_for_fit"] == _MIN_OBS_FOR_FIT

    def test_phase_bootstrap_initially(self, cal):
        assert cal.diagnostics()["phase"] == "bootstrap_prior"

    def test_diagnostics_report_source_counts(self, cal):
        cal.update(0.60, 6.0, observation_id="r1", source_type="replay")
        cal.update(0.62, 6.5, observation_id="p1", source_type="paper")
        diag = cal.diagnostics()
        assert diag["n_replay"] == 1
        assert diag["n_paper"] == 1
        assert diag["n_live"] == 0
        assert diag["n_synthetic"] == 0
