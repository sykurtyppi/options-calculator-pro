from services.evidence_maturity import build_evidence_maturity


def test_maturity_blocks_weak_samples_from_interpretation() -> None:
    maturity = build_evidence_maturity(
        active_evidence_days=5,
        resolved_selector_outcomes=2,
        resolved_baseline_outcomes=0,
        claimable_evidence_count=0,
        max_bucket_sample_size=2,
    )

    assert maturity["maturity_label"] == "Insufficient evidence"
    assert maturity["benchmark_comparison_meaningful"] is False
    assert maturity["edge_quality_label_allowed"] is False
    assert maturity["calibration_interpretation_allowed"] is False
    assert "Withheld" in maturity["edge_quality_label"]
    assert any("Bucket samples" in warning for warning in maturity["warning_flags"])


def test_maturity_progression_labels_developing_and_mature_without_strategy_changes() -> None:
    developing = build_evidence_maturity(
        active_evidence_days=65,
        resolved_selector_outcomes=35,
        resolved_baseline_outcomes=35,
        claimable_evidence_count=12,
        max_bucket_sample_size=31,
    )
    mature = build_evidence_maturity(
        active_evidence_days=95,
        resolved_selector_outcomes=125,
        resolved_baseline_outcomes=125,
        claimable_evidence_count=35,
        max_bucket_sample_size=45,
    )

    assert developing["maturity_label"] == "Developing evidence"
    assert developing["benchmark_comparison_meaningful"] is True
    assert developing["edge_quality_label_allowed"] is False
    assert mature["maturity_label"] == "Mature evidence"
    assert mature["edge_quality_label_allowed"] is True
    assert mature["edge_quality_label"] == "Claimable evidence sample available"
