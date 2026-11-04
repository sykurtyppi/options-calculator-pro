"""The pre-registration names the exact analysis the code runs.

If a constant it fixes changes in code, the document is out of date (and by
its own section 7, that change starts a new cohort): this fails until the
document is amended.
"""
from __future__ import annotations

from pathlib import Path

from services.baseline_evidence_store import BASELINE_STRUCTURES
from services.claim_contract import CONTRACT_VERSION
from services.evidence_health import DEFAULT_MAX_EXIT_ATTRITION_RATE
from services.evidence_report import MIN_RESOLVED_SAMPLE
from services.evidence_statistics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CONFIDENCE, MIN_CI_SAMPLE, OUTLIER_K

DOC = (Path(__file__).resolve().parents[3] / "docs" / "PREREGISTRATION.md").read_text()


def test_the_document_names_the_frozen_analysis_constants() -> None:
    assert f"`{CONTRACT_VERSION}`" in DOC
    assert f"{BOOTSTRAP_RESAMPLES} resamples, fixed seed {BOOTSTRAP_SEED}" in DOC
    assert f"{int(CONFIDENCE * 100)}% percentile bootstrap" in DOC
    assert f"no interval\n  below n = {MIN_CI_SAMPLE}" in DOC
    assert f"k = {OUTLIER_K}" in DOC
    assert f"{MIN_RESOLVED_SAMPLE} claimable selector outcomes (the paid-beta gate's sample)" in DOC
    assert f"exit attrition above {DEFAULT_MAX_EXIT_ATTRITION_RATE:.0%}" in DOC


def test_every_baseline_it_compares_against_exists() -> None:
    for name in ("always_atm_straddle", "always_otm_strangle", "always_iron_condor"):
        assert name in BASELINE_STRUCTURES
        assert name in DOC
