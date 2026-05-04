"""Evidence report aggregation for the automated forward-learning loop.

The report is deliberately observational. It compares selected paper outcomes
against shadow baselines without presenting either as execution-grade live P&L.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from services.baseline_evidence_store import BaselineEvidenceStore, get_baseline_evidence_store
from services.data_quality_diagnostics import build_data_quality_diagnostics
from services.evidence_maturity import build_evidence_maturity
from services.forward_performance_diagnostics import build_forward_performance_diagnostics
from services.outcome_recorder import OutcomeStore, get_outcome_store
from services.provider_telemetry import build_provider_telemetry_diagnostics

MIN_COMMERCIAL_EVIDENCE_DAYS = 60
TARGET_COMMERCIAL_EVIDENCE_DAYS = 90
MIN_RESOLVED_SAMPLE = 30


def build_evidence_report(
    *,
    baseline_store: Optional[BaselineEvidenceStore] = None,
    outcome_store: Optional[OutcomeStore] = None,
    max_rows: int = 10_000,
    recent_limit: int = 25,
) -> Dict[str, Any]:
    baselines = (baseline_store or get_baseline_evidence_store()).list_for_diagnostics(limit=max_rows)
    outcome_obj = outcome_store or get_outcome_store()
    forward = build_forward_performance_diagnostics(
        outcome_store=outcome_obj,
        baseline_store=baseline_store or get_baseline_evidence_store(),
        max_rows=max_rows,
        recent_limit=recent_limit,
    )
    outcomes = outcome_obj.list_for_diagnostics(limit=max_rows)
    selected = [row for row in outcomes if _is_resolved(row)]
    resolved_baselines = [row for row in baselines if str(row.get("status") or "") == "resolved"]
    open_baselines = [row for row in baselines if str(row.get("status") or "") == "open"]
    skipped_baselines = [row for row in baselines if "skipped" in str(row.get("status") or "")]

    selector_stats = _outcome_stats(selected)
    baseline_stats = _group_by(resolved_baselines, lambda row: row.get("baseline_name") or "unknown_baseline")
    baseline_stats["no_trade"] = {
        "n": len(selected),
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "avg_realized_return_pct": 0.0 if selected else None,
        "avg_realized_expansion_pct": 0.0 if selected else None,
        "paper_research_label": "No-trade baseline assumes zero paper return and zero exposure.",
    }

    first_entry = _first_date([row.get("entry_date") for row in selected] + [row.get("entry_date") for row in baselines])
    active_days = _active_days(first_entry)
    evidence_quality = _evidence_quality_summary(outcomes, baselines)
    execution_realism = _execution_realism_summary(outcomes, baselines)
    surface_quality = _surface_quality_summary(outcomes, baselines)
    maturity = build_evidence_maturity(
        active_evidence_days=active_days,
        resolved_selector_outcomes=selector_stats["n"],
        resolved_baseline_outcomes=len(resolved_baselines),
        claimable_evidence_count=int(evidence_quality.get("claim_allowed_count") or 0),
        max_bucket_sample_size=_max_bucket_sample_size(forward.get("calibration_report", {})),
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "paper_research_not_execution_grade",
        "maturity": maturity,
        "commercialization_gate": {
            "active_evidence_days": active_days,
            "minimum_days": MIN_COMMERCIAL_EVIDENCE_DAYS,
            "target_days": TARGET_COMMERCIAL_EVIDENCE_DAYS,
            "resolved_selector_outcomes": selector_stats["n"],
            "minimum_resolved_sample": MIN_RESOLVED_SAMPLE,
            "ready_for_paid_beta": bool(
                active_days >= MIN_COMMERCIAL_EVIDENCE_DAYS
                and selector_stats["n"] >= MIN_RESOLVED_SAMPLE
            ),
        },
        "selector_summary": selector_stats,
        "baseline_comparison": baseline_stats,
        "structure_breakdown": forward.get("by_structure", {}),
        "data_quality_breakdown": forward.get("data_quality_comparison", {}),
        "stale_source_breakdown": forward.get("stale_source_comparison", {}),
        "simple_iv_rv_filter": _simple_iv_rv_filter(selected),
        "evidence_quality": evidence_quality,
        "execution_realism": execution_realism,
        "surface_quality": surface_quality,
        "quote_quality": {
            "baseline_entries": len(baselines),
            "baseline_open": len(open_baselines),
            "baseline_resolved": len(resolved_baselines),
            "baseline_skipped": len(skipped_baselines),
            "entry_sources": _count_by(baselines, lambda row: row.get("quote_source_at_entry") or "unknown"),
            "exit_sources": _count_by(resolved_baselines, lambda row: row.get("quote_source_at_exit") or "unknown"),
            "entry_quality": _count_by(baselines, lambda row: row.get("quote_quality_at_entry") or "unknown"),
        },
        "recent_resolved": forward.get("recent_resolved_outcomes", [])[:recent_limit],
        "warning_flags": _warnings(
            active_days=active_days,
            selector_n=selector_stats["n"],
            baseline_n=len(resolved_baselines),
            forward_warnings=forward.get("warning_flags", []),
            evidence_quality=evidence_quality,
            surface_quality=surface_quality,
            maturity=maturity,
        ),
        "notes": [
            "Selector outcomes are paper/research records, not live broker fills.",
            "Baseline structures are shadow evidence only and do not update calibration or priors.",
            "Simple IV/RV filter is an observational baseline over selected paper outcomes, not a separately traded strategy.",
        ],
    }


def build_weekly_evidence_report(
    *,
    baseline_store: Optional[BaselineEvidenceStore] = None,
    outcome_store: Optional[OutcomeStore] = None,
    data_quality_diagnostics: Optional[Dict[str, Any]] = None,
    provider_telemetry_diagnostics: Optional[Dict[str, Any]] = None,
    max_rows: int = 10_000,
    recent_limit: int = 50,
) -> Dict[str, Any]:
    """Build a weekly, export-friendly evidence packet.

    This is a cadence artifact, not a strategy output. It intentionally repeats
    sample-size and paper/research labels so exported reports cannot be detached
    from their maturity caveats.
    """
    outcome_obj = outcome_store or get_outcome_store()
    baseline_obj = baseline_store or get_baseline_evidence_store()
    evidence = build_evidence_report(
        baseline_store=baseline_obj,
        outcome_store=outcome_obj,
        max_rows=max_rows,
        recent_limit=recent_limit,
    )
    forward = build_forward_performance_diagnostics(
        outcome_store=outcome_obj,
        baseline_store=baseline_obj,
        max_rows=max_rows,
        recent_limit=recent_limit,
    )
    data_quality = data_quality_diagnostics or build_data_quality_diagnostics(max_rows=max_rows, recent_limit=recent_limit)
    provider = provider_telemetry_diagnostics or build_provider_telemetry_diagnostics(limit=recent_limit)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "weekly_evidence_report",
        "evidence_label": "paper_research_not_execution_grade",
        "maturity": evidence.get("maturity", {}),
        "forward_recommendations": {
            "total_recommendations": forward.get("total_recommendations", 0),
            "no_trade_count": forward.get("no_trade_count", 0),
            "open_outcome_count": forward.get("open_outcome_count", 0),
            "resolved_outcome_count": forward.get("resolved_outcome_count", 0),
        },
        "resolved_outcomes": {
            "selector_summary": evidence.get("selector_summary", {}),
            "recent_resolved": evidence.get("recent_resolved", []),
        },
        "claimable_vs_non_claimable": {
            "claimable_evidence": forward.get("claimable_evidence", {}),
            "claimable_performance": forward.get("claimable_performance", {}),
        },
        "evidence_quality_breakdown": evidence.get("evidence_quality", {}),
        "surface_quality_breakdown": evidence.get("surface_quality", {}),
        "execution_scenario_comparison": forward.get("execution_scenario_comparison", {}),
        "execution_realism": evidence.get("execution_realism", {}),
        "benchmark_comparison": forward.get("benchmark_comparison", {}),
        "provider_data_quality_warnings": {
            "provider_telemetry": (provider.get("operational_health") or {}).get("warning_flags", []),
            "data_quality": data_quality.get("warning_flags", []),
            "evidence": evidence.get("warning_flags", []),
        },
        "sample_size_warnings": _sample_size_warnings(evidence.get("maturity", {}), evidence.get("warning_flags", [])),
        "notes": [
            "This report is an operational evidence summary, not trading advice.",
            "Paper/research rows are not execution-grade broker fills.",
            "Maturity guardrails block interpretation when sample sizes or date spans are weak.",
        ],
    }


def _is_resolved(row: Dict[str, Any]) -> bool:
    return row.get("realized_return_pct") is not None and str(row.get("status") or "") in {"exited", "finalized"}


def _outcome_stats(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows)
    returns = [_num(row.get("realized_return_pct")) for row in items]
    expansion = [_num(row.get("realized_expansion_pct")) for row in items]
    wins = sum(1 for value in returns if value is not None and value > 0)
    clean_returns = [value for value in returns if value is not None]
    return {
        "n": len(clean_returns),
        "wins": wins,
        "losses": max(0, len(clean_returns) - wins),
        "win_rate": wins / len(clean_returns) if clean_returns else None,
        "avg_realized_return_pct": _avg(clean_returns),
        "avg_realized_expansion_pct": _avg(expansion),
        "paper_research_label": "paper/research outcomes, not execution-grade live fills",
    }


def _baseline_stats(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows)
    returns = [_num(row.get("realized_return_pct")) for row in items]
    expansion = [_num(row.get("realized_expansion_pct")) for row in items]
    wins = sum(1 for value in returns if value is not None and value > 0)
    clean_returns = [value for value in returns if value is not None]
    return {
        "n": len(clean_returns),
        "wins": wins,
        "losses": max(0, len(clean_returns) - wins),
        "win_rate": wins / len(clean_returns) if clean_returns else None,
        "avg_realized_return_pct": _avg(clean_returns),
        "avg_realized_expansion_pct": _avg(expansion),
        "paper_research_label": "shadow paper baseline, not execution-grade live fills",
    }


def _group_by(rows: Iterable[Dict[str, Any]], key_fn: Any) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return {key: _baseline_stats(items) for key, items in sorted(grouped.items())}


def _simple_iv_rv_filter(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    # Observational baseline: keep selected paper outcomes only when IV was not
    # already rich versus HAR RV. This tests a simple cheap-IV rule against the
    # selector without creating a second trade engine.
    kept = [
        row for row in rows
        if (_num(row.get("iv_rv_har")) is not None and float(row.get("iv_rv_har")) <= 1.05)
    ]
    skipped = [
        row for row in rows
        if not (_num(row.get("iv_rv_har")) is not None and float(row.get("iv_rv_har")) <= 1.05)
    ]
    stats = _outcome_stats(kept)
    stats["skipped_by_filter"] = len(skipped)
    stats["rule"] = "Keep selected paper outcomes only when iv_rv_har <= 1.05; otherwise no-trade."
    return stats


def _evidence_quality_summary(
    outcomes: Iterable[Dict[str, Any]],
    baselines: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    outcome_rows = list(outcomes)
    baseline_rows = list(baselines)
    all_rows = outcome_rows + baseline_rows
    return {
        "selector_status_counts": _count_by(
            outcome_rows,
            lambda row: row.get("evidence_quality_status") or "legacy_unclassified",
        ),
        "baseline_status_counts": _count_by(
            baseline_rows,
            lambda row: row.get("evidence_quality_status") or "legacy_unclassified",
        ),
        "claim_allowed_count": sum(1 for row in all_rows if _truthy(row.get("claim_allowed"))),
        "claim_blocked_count": sum(1 for row in all_rows if row.get("claim_allowed") is not None and not _truthy(row.get("claim_allowed"))),
        "execution_grade_count": sum(1 for row in all_rows if _truthy(row.get("execution_grade"))),
        "record_only_count": sum(1 for row in all_rows if row.get("evidence_quality_status") == "record_only"),
        "degraded_count": sum(1 for row in all_rows if row.get("evidence_quality_status") == "degraded_evidence"),
        "paper_research_label": "Recorded rows may be useful for audit/research while still blocked from performance claims.",
    }


def _execution_realism_summary(
    outcomes: Iterable[Dict[str, Any]],
    baselines: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    outcome_rows = list(outcomes)
    baseline_rows = list(baselines)
    all_rows = outcome_rows + baseline_rows
    spreads = [
        value
        for value in (_scenario_spread_pct(row, "entry") for row in all_rows)
        if value is not None
    ]
    return {
        "selector_entry_scenario_rows": sum(1 for row in outcome_rows if bool(row.get("entry_execution_scenarios_json"))),
        "baseline_entry_scenario_rows": sum(1 for row in baseline_rows if bool(row.get("entry_execution_scenarios_json"))),
        "exit_scenario_rows": sum(1 for row in all_rows if bool(row.get("exit_execution_scenarios_json"))),
        "avg_entry_spread_as_pct_of_premium": _avg(spreads),
        "paper_research_label": "Execution scenarios are modeled from bid/ask quotes, not broker fill records.",
    }


def _surface_quality_summary(
    outcomes: Iterable[Dict[str, Any]],
    baselines: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    outcome_rows = list(outcomes)
    baseline_rows = list(baselines)
    all_rows = outcome_rows + baseline_rows
    return {
        "selector_status_counts": _count_by(outcome_rows, lambda row: row.get("surface_quality_status") or "legacy_unclassified"),
        "baseline_status_counts": _count_by(baseline_rows, lambda row: row.get("surface_quality_status") or "legacy_unclassified"),
        "crossed_quote_count": sum(int(row.get("surface_crossed_quote_count") or 0) for row in all_rows),
        "zero_bid_count": sum(int(row.get("surface_zero_bid_count") or 0) for row in all_rows),
        "extreme_spread_count": sum(int(row.get("surface_extreme_spread_count") or 0) for row in all_rows),
        "sparse_atm_count": sum(int(row.get("surface_sparse_atm_count") or 0) for row in all_rows),
        "iv_anomaly_count": sum(int(row.get("surface_iv_anomaly_count") or 0) for row in all_rows),
        "paper_research_label": "Surface quality is an evidence gate, not a selector score.",
    }


def _scenario_spread_pct(row: Dict[str, Any], phase: str) -> Optional[float]:
    key = f"{phase}_execution_scenarios_json"
    value = row.get(key)
    if isinstance(value, dict):
        return _num(value.get("spread_as_pct_of_premium"))
    if not value:
        return None
    try:
        import json

        parsed = json.loads(str(value))
    except Exception:
        return None
    return _num(parsed.get("spread_as_pct_of_premium")) if isinstance(parsed, dict) else None


def _warnings(
    *,
    active_days: int,
    selector_n: int,
    baseline_n: int,
    forward_warnings: list[str],
    evidence_quality: Optional[Dict[str, Any]] = None,
    surface_quality: Optional[Dict[str, Any]] = None,
    maturity: Optional[Dict[str, Any]] = None,
) -> list[str]:
    warnings = list(forward_warnings or [])
    if active_days < MIN_COMMERCIAL_EVIDENCE_DAYS:
        warnings.append(f"Evidence window is short ({active_days} days); collect at least {MIN_COMMERCIAL_EVIDENCE_DAYS} days before paid beta claims.")
    if selector_n < MIN_RESOLVED_SAMPLE:
        warnings.append(f"Resolved selector sample is small (n={selector_n}); do not infer durable edge yet.")
    if baseline_n == 0:
        warnings.append("No resolved shadow baselines yet; selector-vs-baseline comparison is not mature.")
    if (evidence_quality or {}).get("claim_allowed_count", 0) == 0:
        warnings.append("No execution-grade claimable evidence yet; keep results labeled paper/research.")
    if (evidence_quality or {}).get("record_only_count", 0) > 0:
        warnings.append("Some rows are record-only because quote/data quality failed evidence gates.")
    if (surface_quality or {}).get("extreme_spread_count", 0) > 0:
        warnings.append("Some option surfaces contain extreme bid/ask spreads.")
    if (surface_quality or {}).get("sparse_atm_count", 0) > 0:
        warnings.append("Some option surfaces are sparse around ATM strikes.")
    warnings.extend((maturity or {}).get("warning_flags", []))
    return _dedupe(warnings)


def _max_bucket_sample_size(calibration_report: Dict[str, Any]) -> int:
    max_n = 0
    for groups in (calibration_report or {}).values():
        if not isinstance(groups, dict):
            continue
        for stats in groups.values():
            if isinstance(stats, dict):
                max_n = max(max_n, int(stats.get("n") or 0))
    return max_n


def _sample_size_warnings(maturity: Dict[str, Any], warnings: Iterable[str]) -> list[str]:
    tagged = [
        str(item)
        for item in warnings
        if "sample" in str(item).lower()
        or "evidence days" in str(item).lower()
        or "benchmark comparison" in str(item).lower()
        or "withheld" in str(item).lower()
    ]
    tagged.extend(str(item) for item in (maturity or {}).get("warning_flags", []))
    return _dedupe(tagged)


def _count_by(rows: Iterable[Dict[str, Any]], key_fn: Any) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(key_fn(row))] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _first_date(values: Iterable[Any]) -> Optional[str]:
    clean = sorted(str(value)[:10] for value in values if value)
    return clean[0] if clean else None


def _active_days(first_entry: Optional[str]) -> int:
    if not first_entry:
        return 0
    try:
        start = datetime.fromisoformat(first_entry).date()
    except ValueError:
        return 0
    return max(0, (datetime.now(timezone.utc).date() - start).days + 1)


def _avg(values: Iterable[Any]) -> Optional[float]:
    clean = [_num(value) for value in values]
    nums = [value for value in clean if value is not None]
    return sum(nums) / len(nums) if nums else None


def _num(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else None
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
