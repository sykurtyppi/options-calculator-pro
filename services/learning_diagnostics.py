from __future__ import annotations

from typing import Any, Dict

from services.calibration_service import get_calibration
from services.structure_prior_store import SUPPORTED_STRUCTURES, get_structure_prior_store


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _normalize_structure_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "count": int(entry.get("observation_count", 0) or 0),
        "win_rate": float(entry.get("win_rate") or 0.0),
        "avg_return_pct": float(entry.get("avg_return_pct") or 0.0),
        "avg_expansion_pct": float(entry.get("avg_realized_expansion_pct") or 0.0),
    }


def build_learning_diagnostics() -> Dict[str, Any]:
    cal_diag = get_calibration().diagnostics()
    prior_diag = get_structure_prior_store().diagnostics()

    n_total = int(cal_diag.get("n_observations", 0) or 0)
    n_replay = int(cal_diag.get("n_replay", 0) or 0)
    n_synthetic = int(cal_diag.get("n_synthetic", 0) or 0)
    n_paper = int(cal_diag.get("n_paper", 0) or 0)
    n_live = int(cal_diag.get("n_live", 0) or 0)

    replay_ratio = _safe_ratio(n_replay, n_total)
    synthetic_ratio = _safe_ratio(n_synthetic, n_total)
    paper_ratio = _safe_ratio(n_paper, n_total)

    structures_raw = prior_diag.get("structures", {})
    structure_priors = {
        structure: _normalize_structure_entry(structures_raw.get(structure, {}))
        for structure in SUPPORTED_STRUCTURES
    }

    counts = [entry["count"] for entry in structure_priors.values()]
    nonzero_counts = [count for count in counts if count > 0]
    structure_imbalanced = False
    if nonzero_counts:
        min_count = min(nonzero_counts)
        max_count = max(nonzero_counts)
        structure_imbalanced = (
            len(nonzero_counts) < len(SUPPORTED_STRUCTURES)
            or (min_count > 0 and max_count >= (3 * min_count))
        )

    warning_flags: list[str] = []
    if n_total < 40:
        warning_flags.append("Calibration in bootstrap phase")
    if synthetic_ratio > 0.5:
        warning_flags.append("Calibration dominated by synthetic data")
    if structure_imbalanced:
        warning_flags.append("Structure priors imbalanced")
    if (n_paper + n_live) == 0:
        warning_flags.append("No real forward observations yet")

    return {
        "calibration": {
            "phase": cal_diag.get("phase", "bootstrap_prior"),
            "n_total": n_total,
            "n_replay": n_replay,
            "n_synthetic": n_synthetic,
            "n_paper": n_paper,
            "n_live": n_live,
            "is_prior_only": bool(cal_diag.get("is_prior_only", n_total < 40)),
            "score_distribution": cal_diag.get(
                "score_distribution",
                {"min": None, "max": None, "mean": None},
            ),
            "expansion_distribution": cal_diag.get(
                "expansion_distribution",
                {"min": None, "max": None, "mean": None},
            ),
        },
        "structure_priors": structure_priors,
        "data_quality": {
            "has_real_data": bool((n_replay + n_paper + n_live) > 0),
            "replay_dominant": replay_ratio > 0.7,
            "synthetic_ratio": round(synthetic_ratio, 4),
            "paper_ratio": round(paper_ratio, 4),
        },
        "learning_health": {
            "calibration_stable": bool(n_total >= 250 and replay_ratio >= 0.5),
            "sufficient_observations": bool(n_total >= 120),
            "warning_flags": warning_flags,
        },
    }
