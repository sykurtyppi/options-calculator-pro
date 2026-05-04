"""Evidence maturity guardrails for forward paper/research reporting.

This module deliberately does not change selection, scoring, or calibration.
It only decides how strongly diagnostics are allowed to be interpreted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class EvidenceMaturityThresholds:
    min_bucket_sample_size: int = 30
    min_benchmark_resolved_outcomes: int = 30
    min_claimable_evidence_count: int = 30
    min_calibration_date_span_days: int = 60
    early_observational_days: int = 14
    early_observational_outcomes: int = 10
    mature_date_span_days: int = 90
    mature_resolved_outcomes: int = 120


DEFAULT_THRESHOLDS = EvidenceMaturityThresholds()


def build_evidence_maturity(
    *,
    active_evidence_days: int,
    resolved_selector_outcomes: int,
    resolved_baseline_outcomes: int,
    claimable_evidence_count: int,
    max_bucket_sample_size: int = 0,
    thresholds: EvidenceMaturityThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Return conservative reporting guardrails for evidence interpretation."""
    active_days = max(0, int(active_evidence_days or 0))
    resolved = max(0, int(resolved_selector_outcomes or 0))
    baselines = max(0, int(resolved_baseline_outcomes or 0))
    claimable = max(0, int(claimable_evidence_count or 0))
    max_bucket = max(0, int(max_bucket_sample_size or 0))

    benchmark_meaningful = resolved >= thresholds.min_benchmark_resolved_outcomes and baselines >= thresholds.min_benchmark_resolved_outcomes
    claimable_ready = claimable >= thresholds.min_claimable_evidence_count
    calibration_ready = active_days >= thresholds.min_calibration_date_span_days and max_bucket >= thresholds.min_bucket_sample_size

    if (
        active_days >= thresholds.mature_date_span_days
        and resolved >= thresholds.mature_resolved_outcomes
        and benchmark_meaningful
        and claimable_ready
        and calibration_ready
    ):
        label = "Mature evidence"
    elif active_days >= thresholds.min_calibration_date_span_days and resolved >= thresholds.min_benchmark_resolved_outcomes:
        label = "Developing evidence"
    elif active_days >= thresholds.early_observational_days or resolved >= thresholds.early_observational_outcomes:
        label = "Early observational"
    else:
        label = "Insufficient evidence"

    warnings: list[str] = []
    if max_bucket < thresholds.min_bucket_sample_size:
        warnings.append(
            f"Bucket samples are below {thresholds.min_bucket_sample_size}; do not interpret per-bucket calibration yet."
        )
    if not benchmark_meaningful:
        warnings.append(
            f"Benchmark comparison needs at least {thresholds.min_benchmark_resolved_outcomes} selector and baseline outcomes."
        )
    if not claimable_ready:
        warnings.append(
            f"Edge-quality labels are withheld until at least {thresholds.min_claimable_evidence_count} claimable evidence rows exist."
        )
    if active_days < thresholds.min_calibration_date_span_days:
        warnings.append(
            f"Calibration interpretation is withheld until at least {thresholds.min_calibration_date_span_days} evidence days are collected."
        )

    return {
        "maturity_label": label,
        "thresholds": asdict(thresholds),
        "inputs": {
            "active_evidence_days": active_days,
            "resolved_selector_outcomes": resolved,
            "resolved_baseline_outcomes": baselines,
            "claimable_evidence_count": claimable,
            "max_bucket_sample_size": max_bucket,
        },
        "bucket_interpretation_allowed": max_bucket >= thresholds.min_bucket_sample_size,
        "benchmark_comparison_meaningful": benchmark_meaningful,
        "edge_quality_label_allowed": claimable_ready,
        "calibration_interpretation_allowed": calibration_ready,
        "edge_quality_label": "Claimable evidence sample available" if claimable_ready else "Withheld: insufficient claimable evidence",
        "warning_flags": warnings,
        "paper_research_label": "Maturity labels guard interpretation only; they are not trading advice or performance claims.",
    }
