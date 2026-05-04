"""Forward paper-performance diagnostics for selector recommendations.

This module is intentionally read-only. It joins the durable recommendation
ledger to the paper outcome journal so the product can inspect whether
selector recommendations are improving without presenting paper results as
execution-grade live performance.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from services.baseline_evidence_store import BaselineEvidenceStore, get_baseline_evidence_store
from services.outcome_recorder import OutcomeStore, get_outcome_store
from services.recommendation_ledger import RecommendationLedger, get_recommendation_ledger

LOW_QUALITY_THRESHOLD = 0.60
HIGH_QUALITY_THRESHOLD = 0.75
MIN_SAMPLE_FOR_STABLE_READ = 30

_CONFIDENCE_BUCKETS = (
    (0.0, 40.0, "0-40"),
    (40.0, 60.0, "40-60"),
    (60.0, 75.0, "60-75"),
    (75.0, 90.0, "75-90"),
    (90.0, 100.000001, "90-100"),
)


def build_forward_performance_diagnostics(
    *,
    ledger: Optional[RecommendationLedger] = None,
    outcome_store: Optional[OutcomeStore] = None,
    baseline_store: Optional[BaselineEvidenceStore] = None,
    max_rows: int = 10_000,
    recent_limit: int = 25,
) -> Dict[str, Any]:
    """Aggregate paper/research outcomes by structure, quality, and score bucket."""
    ledger_obj = ledger or get_recommendation_ledger()
    outcome_obj = outcome_store or get_outcome_store()
    baseline_obj = baseline_store or get_baseline_evidence_store()

    ledger_rows = ledger_obj.list_for_diagnostics(limit=max_rows)
    outcomes = outcome_obj.list_for_diagnostics(limit=max_rows)
    baseline_rows = baseline_obj.list_for_diagnostics(limit=max_rows)
    ledger_by_id = {
        str(row.get("recommendation_id")): row
        for row in ledger_rows
        if row.get("recommendation_id")
    }

    resolved = [
        _merged_outcome(row, ledger_by_id.get(str(row.get("recommendation_id") or "")))
        for row in outcomes
        if _is_resolved(row)
    ]
    resolved_baselines = [_merged_baseline(row) for row in baseline_rows if _is_baseline_resolved(row)]

    no_trade_count = sum(1 for row in ledger_rows if str(row.get("recommendation") or "") == "No Trade")
    open_count = sum(1 for row in outcomes if str(row.get("status") or "") in {"open", "exited"})
    warning_flags = _warning_flags(resolved, no_trade_count=no_trade_count)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "paper_research_not_execution_grade",
        "notes": [
            "Forward results are paper/research diagnostics unless explicitly labeled live.",
            "Modeled P&L uses score-derived expected-return fields captured at recommendation time.",
            "Realized P&L is only as good as the recorded paper quote and cost assumptions.",
        ],
        "thresholds": {
            "low_quality_threshold": LOW_QUALITY_THRESHOLD,
            "high_quality_threshold": HIGH_QUALITY_THRESHOLD,
            "min_sample_for_stable_read": MIN_SAMPLE_FOR_STABLE_READ,
        },
        "total_recommendations": len(ledger_rows),
        "no_trade_count": no_trade_count,
        "open_outcome_count": open_count,
        "resolved_outcome_count": len(resolved),
        "performance_summary": _group_stats(resolved),
        "by_structure": _group_by(resolved, lambda item: item["structure"] or "unknown"),
        "stale_source_comparison": _group_by(
            resolved,
            lambda item: "stale_source" if item["earnings_source_stale"] else "fresh_or_unknown_source",
        ),
        "data_quality_comparison": _group_by(resolved, _quality_group),
        "evidence_quality_comparison": _group_by(resolved, lambda item: item.get("evidence_quality_status") or "legacy_unclassified"),
        "surface_quality_comparison": _group_by(resolved, lambda item: item.get("surface_quality_status") or "legacy_unclassified"),
        "spread_cost_buckets": _group_by(resolved, _spread_cost_bucket),
        "execution_scenario_comparison": _execution_scenario_stats(resolved),
        "calibration_report": _calibration_report(resolved),
        "benchmark_comparison": _benchmark_comparison(resolved, resolved_baselines),
        "claimable_performance": _group_by(resolved, _claimable_group),
        "claimable_evidence": {
            "claimable_count": sum(1 for row in resolved if bool(row.get("claim_allowed"))),
            "non_claimable_count": sum(1 for row in resolved if row.get("claim_allowed") is not None and not bool(row.get("claim_allowed"))),
            "execution_grade_count": sum(1 for row in resolved if bool(row.get("execution_grade"))),
            "paper_research_label": "Claimability requires execution-grade evidence; paper/research rows remain observational.",
        },
        "outcome_count_by_symbol": _count_by(resolved, lambda item: item["symbol"] or "UNKNOWN"),
        "calibration_buckets": _bucket_rows(resolved),
        "recent_resolved_outcomes": _recent_rows(resolved, limit=recent_limit),
        "warning_flags": warning_flags,
    }


def _is_resolved(row: Dict[str, Any]) -> bool:
    if row.get("realized_return_pct") is None:
        return False
    return str(row.get("status") or "") in {"exited", "finalized"}


def _is_baseline_resolved(row: Dict[str, Any]) -> bool:
    if row.get("realized_return_pct") is None:
        return False
    return str(row.get("status") or "") in {"resolved", "exited", "finalized"}


def _merged_outcome(outcome: Dict[str, Any], ledger_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    selector_output = _as_dict((ledger_row or {}).get("selector_output_json"))
    structure = (
        outcome.get("structure")
        or (ledger_row or {}).get("selected_structure")
        or selector_output.get("best_structure")
    )
    confidence_pct = _num(outcome.get("selector_confidence_pct"))
    if confidence_pct is None:
        confidence_pct = _num(selector_output.get("confidence_pct"))
    setup_score = _num(outcome.get("setup_score"))
    modeled = _num(outcome.get("expected_return_pct"))
    if modeled is None:
        modeled = _num(selector_output.get("expected_return_pct"))

    quality = _num(outcome.get("data_quality_score_at_entry"))
    if quality is None:
        quality = _num((ledger_row or {}).get("data_quality_score"))

    return {
        "trade_id": outcome.get("trade_id"),
        "recommendation_id": outcome.get("recommendation_id"),
        "symbol": str(outcome.get("symbol") or (ledger_row or {}).get("symbol") or "").upper(),
        "structure": str(structure or "unknown"),
        "status": outcome.get("status"),
        "source_type": outcome.get("source_type") or "paper",
        "entry_date": outcome.get("entry_date"),
        "exit_date": outcome.get("exit_date"),
        "earnings_date": outcome.get("earnings_date") or (ledger_row or {}).get("earnings_date"),
        "selector_recommendation": outcome.get("selector_recommendation") or (ledger_row or {}).get("recommendation"),
        "confidence_pct": confidence_pct,
        "setup_score": setup_score,
        "modeled_return_pct": modeled,
        "realized_return_pct": _num(outcome.get("realized_return_pct")),
        "realized_expansion_pct": _num(outcome.get("realized_expansion_pct")),
        "data_quality_score": quality,
        "earnings_source": (ledger_row or {}).get("earnings_source"),
        "earnings_source_confidence": _num((ledger_row or {}).get("earnings_source_confidence")),
        "earnings_source_stale": bool((ledger_row or {}).get("earnings_source_stale")),
        "quote_quality": (ledger_row or {}).get("quote_quality"),
        "iv_rv_har": _num(outcome.get("iv_rv_har")),
        "liquidity_tier": outcome.get("liquidity_tier"),
        "evidence_quality_status": outcome.get("evidence_quality_status"),
        "claim_allowed": _boolish(outcome.get("claim_allowed")),
        "execution_grade": _boolish(outcome.get("execution_grade")),
        "surface_quality_status": outcome.get("surface_quality_status") or (ledger_row or {}).get("surface_quality_status"),
        "surface_quality_reasons": _as_list(outcome.get("surface_quality_reasons_json") or (ledger_row or {}).get("surface_quality_reasons_json")),
        "surface_crossed_quote_count": int(outcome.get("surface_crossed_quote_count") or (ledger_row or {}).get("surface_crossed_quote_count") or 0),
        "surface_zero_bid_count": int(outcome.get("surface_zero_bid_count") or (ledger_row or {}).get("surface_zero_bid_count") or 0),
        "surface_extreme_spread_count": int(outcome.get("surface_extreme_spread_count") or (ledger_row or {}).get("surface_extreme_spread_count") or 0),
        "surface_sparse_atm_count": int(outcome.get("surface_sparse_atm_count") or (ledger_row or {}).get("surface_sparse_atm_count") or 0),
        "surface_iv_anomaly_count": int(outcome.get("surface_iv_anomaly_count") or (ledger_row or {}).get("surface_iv_anomaly_count") or 0),
        "entry_spread_cost_pct": _scenario_spread_pct(outcome.get("entry_execution_scenarios_json")),
        "execution_scenario_returns": _scenario_returns(outcome.get("exit_execution_scenarios_json")),
    }


def _merged_baseline(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "baseline_id": row.get("baseline_id"),
        "recommendation_id": row.get("recommendation_id"),
        "symbol": str(row.get("symbol") or "").upper(),
        "baseline_name": row.get("baseline_name") or "unknown_baseline",
        "structure": row.get("structure") or "unknown",
        "entry_date": row.get("entry_date"),
        "exit_date": row.get("exit_date"),
        "modeled_return_pct": _num(row.get("modeled_cost_pct")),
        "realized_return_pct": _num(row.get("realized_return_pct")),
        "realized_expansion_pct": _num(row.get("realized_expansion_pct")),
        "data_quality_score": _num(row.get("data_quality_score_at_entry")),
        "evidence_quality_status": row.get("evidence_quality_status"),
        "claim_allowed": _boolish(row.get("claim_allowed")),
        "execution_grade": _boolish(row.get("execution_grade")),
        "surface_quality_status": row.get("surface_quality_status"),
        "entry_spread_cost_pct": _scenario_spread_pct(row.get("entry_execution_scenarios_json")),
        "execution_scenario_returns": _scenario_returns(row.get("exit_execution_scenarios_json")),
    }


def _group_stats(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(items)
    n = len(rows)
    wins = sum(1 for row in rows if (_num(row.get("realized_return_pct")) or 0.0) > 0.0)
    losses = n - wins
    modeled = [_num(row.get("modeled_return_pct")) for row in rows]
    realized = [_num(row.get("realized_return_pct")) for row in rows]
    expansion = [_num(row.get("realized_expansion_pct")) for row in rows]
    model_errors = [
        r - m
        for r, m in zip(realized, modeled)
        if r is not None and m is not None
    ]
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n if n else None,
        "avg_modeled_return_pct": _avg(modeled),
        "avg_realized_return_pct": _avg(realized),
        "median_realized_return_pct": _median(realized),
        "avg_model_error_pct": _avg(model_errors),
        "avg_realized_expansion_pct": _avg(expansion),
        "claimable_count": sum(1 for row in rows if bool(row.get("claim_allowed"))),
        "small_sample_warning": bool(n and n < MIN_SAMPLE_FOR_STABLE_READ),
        "interpretation_allowed": bool(n >= MIN_SAMPLE_FOR_STABLE_READ),
        "warning": f"Small sample (n={n}); directional paper/research read only." if n and n < MIN_SAMPLE_FOR_STABLE_READ else None,
        "paper_research_warning": "Paper/research only; not execution-grade live performance.",
        "paper_research_label": "paper/research outcomes, not execution-grade live fills",
    }


def _group_by(items: Iterable[Dict[str, Any]], key_fn: Any) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(key_fn(item))].append(item)
    return {key: _group_stats(rows) for key, rows in sorted(grouped.items())}


def _count_by(items: Iterable[Dict[str, Any]], key_fn: Any) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        counts[str(key_fn(item))] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _quality_group(item: Dict[str, Any]) -> str:
    quality = _num(item.get("data_quality_score"))
    if quality is None:
        return "unknown_quality"
    if quality < LOW_QUALITY_THRESHOLD:
        return "low_quality"
    if quality >= HIGH_QUALITY_THRESHOLD:
        return "high_quality"
    return "middle_quality"


def _claimable_group(item: Dict[str, Any]) -> str:
    claim_allowed = item.get("claim_allowed")
    if claim_allowed is None:
        return "legacy_unknown_claimability"
    return "claimable" if bool(claim_allowed) else "non_claimable"


def _spread_cost_bucket(item: Dict[str, Any]) -> str:
    spread = _num(item.get("entry_spread_cost_pct"))
    if spread is None:
        return "unknown_spread_cost"
    if spread < 5:
        return "0-5pct_spread_cost"
    if spread < 15:
        return "5-15pct_spread_cost"
    if spread < 30:
        return "15-30pct_spread_cost"
    return "30pct_plus_spread_cost"


def _score_bucket(item: Dict[str, Any]) -> str:
    score = _num(item.get("setup_score"))
    if score is None:
        confidence = _num(item.get("confidence_pct"))
        score = confidence / 100.0 if confidence is not None else None
    if score is None:
        return "unknown_score"
    if score < 0.40:
        return "0.00-0.40"
    if score < 0.60:
        return "0.40-0.60"
    if score < 0.75:
        return "0.60-0.75"
    if score < 0.90:
        return "0.75-0.90"
    return "0.90-1.00"


def _data_quality_bucket(item: Dict[str, Any]) -> str:
    quality = _num(item.get("data_quality_score"))
    if quality is None:
        return "unknown_data_quality"
    if quality < 0.50:
        return "0.00-0.50"
    if quality < LOW_QUALITY_THRESHOLD:
        return "0.50-0.60"
    if quality < HIGH_QUALITY_THRESHOLD:
        return "0.60-0.75"
    if quality < 0.90:
        return "0.75-0.90"
    return "0.90-1.00"


def _earnings_confidence_bucket(item: Dict[str, Any]) -> str:
    confidence = _num(item.get("earnings_source_confidence"))
    if confidence is None:
        return "unknown_earnings_confidence"
    if confidence < 0.50:
        return "0.00-0.50"
    if confidence < 0.75:
        return "0.50-0.75"
    if confidence < 0.90:
        return "0.75-0.90"
    return "0.90-1.00"


def _execution_scenario_stats(items: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_scenario: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in items:
        scenario_returns = row.get("execution_scenario_returns") or {}
        for scenario, realized_return in scenario_returns.items():
            if _num(realized_return) is None:
                continue
            by_scenario[str(scenario)].append({**row, "realized_return_pct": _num(realized_return)})
    return {scenario: _group_stats(rows) for scenario, rows in sorted(by_scenario.items())}


def _calibration_report(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(items)
    return {
        "selector_score_bucket": _group_by(rows, _score_bucket),
        "selected_structure": _group_by(rows, lambda item: item.get("structure") or "unknown"),
        "evidence_quality_status": _group_by(rows, lambda item: item.get("evidence_quality_status") or "legacy_unclassified"),
        "surface_quality_status": _group_by(rows, lambda item: item.get("surface_quality_status") or "legacy_unclassified"),
        "data_quality_score_bucket": _group_by(rows, _data_quality_bucket),
        "spread_cost_bucket": _group_by(rows, _spread_cost_bucket),
        "execution_scenario": _execution_scenario_stats(rows),
        "earnings_source_confidence_bucket": _group_by(rows, _earnings_confidence_bucket),
        "paper_research_label": "Calibration buckets are empirical paper/research diagnostics, not model changes.",
    }


def _benchmark_comparison(
    selector_rows: Iterable[Dict[str, Any]],
    baseline_rows: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    selector = list(selector_rows)
    baselines = list(baseline_rows)
    baseline_by_name: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in baselines:
        baseline_by_name[str(row.get("baseline_name") or "unknown_baseline")].append(row)

    no_trade = _no_trade_baseline(selector)
    simple_filter_members = [
        row for row in selector
        if _num(row.get("iv_rv_har")) is not None and float(_num(row.get("iv_rv_har")) or 0.0) <= 1.05
    ]
    liquidity_members = [
        row for row in selector
        if _num(row.get("entry_spread_cost_pct")) is not None and float(_num(row.get("entry_spread_cost_pct")) or 0.0) <= 15.0
    ]
    clean_surface_members = [
        row for row in selector
        if str(row.get("surface_quality_status") or "") == "clean_surface"
    ]

    return {
        "selector": {**_group_stats(selector), "rule": "Actual selector-selected paper outcomes."},
        "always_atm_straddle": {
            **_group_stats(baseline_by_name.get("always_atm_straddle", [])),
            "rule": "Shadow baseline: enter ATM straddle whenever selector records a candidate baseline.",
        },
        "always_otm_strangle": {
            **_group_stats(baseline_by_name.get("always_otm_strangle", [])),
            "rule": "Shadow baseline: enter OTM strangle whenever selector records a candidate baseline.",
        },
        "no_trade": no_trade,
        "simple_iv_rv_filter": {
            **_group_stats(simple_filter_members),
            "rule": "Keep selected paper outcomes only when iv_rv_har <= 1.05; otherwise no-trade.",
            "skipped_by_filter": max(0, len(selector) - len(simple_filter_members)),
        },
        "liquidity_only_filter": {
            **_group_stats(liquidity_members),
            "rule": "Keep selected paper outcomes only when entry spread cost <= 15% of premium; otherwise no-trade.",
            "skipped_by_filter": max(0, len(selector) - len(liquidity_members)),
        },
        "clean_surface_only_filter": {
            **_group_stats(clean_surface_members),
            "rule": "Keep selected paper outcomes only when surface_quality_status is clean_surface; otherwise no-trade.",
            "skipped_by_filter": max(0, len(selector) - len(clean_surface_members)),
        },
        "paper_research_label": "Benchmarks are observational paper/research comparisons and are not optimized.",
    }


def _no_trade_baseline(selector_rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(selector_rows)
    return {
        "n": n,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "avg_modeled_return_pct": 0.0 if n else None,
        "avg_realized_return_pct": 0.0 if n else None,
        "median_realized_return_pct": 0.0 if n else None,
        "avg_model_error_pct": 0.0 if n else None,
        "avg_realized_expansion_pct": 0.0 if n else None,
        "claimable_count": 0,
        "small_sample_warning": bool(n and n < MIN_SAMPLE_FOR_STABLE_READ),
        "interpretation_allowed": bool(n >= MIN_SAMPLE_FOR_STABLE_READ),
        "warning": f"Small sample (n={n}); directional paper/research read only." if n and n < MIN_SAMPLE_FOR_STABLE_READ else None,
        "paper_research_warning": "No-trade baseline assumes zero paper return and zero exposure.",
        "paper_research_label": "No-trade baseline assumes zero paper return and zero exposure.",
        "rule": "No trade; zero exposure baseline.",
    }


def _bucket_rows(items: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    rows = list(items)
    bucketed: list[Dict[str, Any]] = []
    for low, high, label in _CONFIDENCE_BUCKETS:
        members = [
            row for row in rows
            if _bucket_value(row) is not None and low <= float(_bucket_value(row)) < high
        ]
        stats = _group_stats(members)
        bucketed.append(
            {
                "bucket": label,
                "bucket_type": "selector_confidence_pct_or_setup_score",
                "n": stats["n"],
                "win_rate": stats["win_rate"],
                "avg_confidence_pct": _avg([_num(row.get("confidence_pct")) for row in members]),
                "avg_setup_score": _avg([_num(row.get("setup_score")) for row in members]),
                "avg_modeled_return_pct": stats["avg_modeled_return_pct"],
                "avg_realized_return_pct": stats["avg_realized_return_pct"],
                "median_realized_return_pct": stats["median_realized_return_pct"],
                "avg_model_error_pct": stats["avg_model_error_pct"],
                "claimable_count": stats["claimable_count"],
                "small_sample_warning": stats["small_sample_warning"],
            }
        )
    return bucketed


def _bucket_value(row: Dict[str, Any]) -> Optional[float]:
    confidence = _num(row.get("confidence_pct"))
    if confidence is not None:
        return max(0.0, min(100.0, confidence))
    setup_score = _num(row.get("setup_score"))
    if setup_score is not None:
        return max(0.0, min(100.0, setup_score * 100.0))
    return None


def _recent_rows(items: Iterable[Dict[str, Any]], *, limit: int) -> list[Dict[str, Any]]:
    rows = sorted(
        list(items),
        key=lambda row: str(row.get("exit_date") or row.get("entry_date") or ""),
        reverse=True,
    )
    return [
        {
            "trade_id": row.get("trade_id"),
            "recommendation_id": row.get("recommendation_id"),
            "symbol": row.get("symbol"),
            "structure": row.get("structure"),
            "source_type": row.get("source_type"),
            "entry_date": row.get("entry_date"),
            "exit_date": row.get("exit_date"),
            "modeled_return_pct": row.get("modeled_return_pct"),
            "realized_return_pct": row.get("realized_return_pct"),
            "realized_expansion_pct": row.get("realized_expansion_pct"),
            "data_quality_score": row.get("data_quality_score"),
            "earnings_source_stale": row.get("earnings_source_stale"),
            "quote_quality": row.get("quote_quality"),
            "evidence_quality_status": row.get("evidence_quality_status"),
            "surface_quality_status": row.get("surface_quality_status"),
            "entry_spread_cost_pct": row.get("entry_spread_cost_pct"),
            "claim_allowed": row.get("claim_allowed"),
        }
        for row in rows[: max(1, min(int(limit or 25), 100))]
    ]


def _warning_flags(resolved: list[Dict[str, Any]], *, no_trade_count: int) -> list[str]:
    warnings: list[str] = ["Forward-performance results are paper/research diagnostics, not execution-grade live performance."]
    n = len(resolved)
    if n == 0:
        warnings.append("No resolved paper outcomes are available yet.")
    elif n < MIN_SAMPLE_FOR_STABLE_READ:
        warnings.append(f"Resolved sample is small (n={n}); treat performance reads as directional only.")
    if no_trade_count > 0 and n == 0:
        warnings.append("No-trade decisions are being recorded, but there are not enough resolved candidate outcomes yet.")
    if any(row.get("earnings_source_stale") for row in resolved):
        warnings.append("Some resolved outcomes were linked to stale earnings-source recommendations.")
    if any(_quality_group(row) == "low_quality" for row in resolved):
        warnings.append("Some resolved outcomes came from low data-quality recommendations.")
    return warnings


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    clean: list[float] = []
    for value in values:
        parsed = _num(value)
        if parsed is not None:
            clean.append(parsed)
    if not clean:
        return None
    return sum(clean) / len(clean)


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = sorted(parsed for value in values if (parsed := _num(value)) is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _num(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else None
    except Exception:
        return None


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            import json

            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        import json

        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _scenario_spread_pct(value: Any) -> Optional[float]:
    parsed = _json_dict(value)
    return _num(parsed.get("spread_as_pct_of_premium"))


def _scenario_returns(value: Any) -> Dict[str, Any]:
    parsed = _json_dict(value)
    outcomes = parsed.get("scenario_outcomes") if isinstance(parsed, dict) else {}
    returns = (outcomes or {}).get("realized_return_pct") if isinstance(outcomes, dict) else {}
    return returns if isinstance(returns, dict) else {}


def _boolish(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
