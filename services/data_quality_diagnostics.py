"""Read-only provider and data-quality diagnostics for recommendation records."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from services.recommendation_ledger import RecommendationLedger, get_recommendation_ledger

LOW_DATA_QUALITY_THRESHOLD = 0.50
STALE_RATE_WARN_THRESHOLD = 0.25
LOW_QUALITY_RATE_WARN_THRESHOLD = 0.20


def build_data_quality_diagnostics(
    *,
    ledger: Optional[RecommendationLedger] = None,
    max_rows: int = 10_000,
    recent_limit: int = 10,
) -> Dict[str, Any]:
    ledger_obj = ledger or get_recommendation_ledger()
    rows = ledger_obj.list_for_diagnostics(limit=max_rows)
    total = len(rows)

    stale_count = sum(1 for row in rows if bool(row.get("earnings_source_stale")))
    missing_option_chain_count = sum(1 for row in rows if _missing_option_chain(row))
    low_quality_count = sum(1 for row in rows if _is_low_quality(row))
    poor_surface_count = sum(1 for row in rows if _poor_surface(row))

    source_breakdown = {
        "option_source": _counter(row.get("option_source") for row in rows),
        "underlying_source": _counter(row.get("underlying_source") for row in rows),
        "earnings_source": _counter(row.get("earnings_source") for row in rows),
        "quote_source": _counter(row.get("quote_source") for row in rows),
    }
    provider_health = {
        "option_source": {
            "success": max(0, total - missing_option_chain_count),
            "failure": missing_option_chain_count,
        },
        "underlying_source": _source_health(rows, "underlying_source"),
        "earnings_source": {
            **_source_health(rows, "earnings_source"),
            "stale": stale_count,
        },
        "quote_source": _source_health(rows, "quote_source"),
    }

    recent_weak = [
        _weak_row_summary(row)
        for row in rows
        if _is_low_quality(row)
        or bool(row.get("earnings_source_stale"))
        or _missing_option_chain(row)
        or _poor_surface(row)
        or _safe_float(row.get("earnings_source_confidence"), default=1.0) < 0.50
    ][: max(1, int(recent_limit or 10))]

    stale_rate = _ratio(stale_count, total)
    low_quality_rate = _ratio(low_quality_count, total)
    warnings: List[str] = []
    if total == 0:
        warnings.append("No recommendation records available for data-quality diagnostics.")
    if stale_rate > STALE_RATE_WARN_THRESHOLD:
        warnings.append("Stale earnings-source rate is elevated.")
    if low_quality_rate > LOW_QUALITY_RATE_WARN_THRESHOLD:
        warnings.append("Low data-quality recommendation rate is elevated.")
    if missing_option_chain_count > 0:
        warnings.append("Some recommendations were recorded without a usable option-chain source.")
    if poor_surface_count > 0:
        warnings.append("Some recommendations were recorded with degraded option-surface quality.")
    if source_breakdown["option_source"].get("unknown", 0) > 0 or source_breakdown["earnings_source"].get("unknown", 0) > 0:
        warnings.append("Some provider/source fields are missing or unknown.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_recommendations": total,
        "provider_health": provider_health,
        "stale_earnings_source_count": stale_count,
        "stale_earnings_source_rate": stale_rate,
        "missing_option_chain_count": missing_option_chain_count,
        "low_data_quality_count": low_quality_count,
        "low_data_quality_rate": low_quality_rate,
        "poor_surface_quality_count": poor_surface_count,
        "surface_quality_buckets": _counter(row.get("surface_quality_status") for row in rows),
        "data_quality_buckets": _quality_buckets(rows),
        "source_breakdown": source_breakdown,
        "recent_weak_data_recommendations": recent_weak,
        "warning_flags": warnings,
        "thresholds": {
            "low_data_quality_score": LOW_DATA_QUALITY_THRESHOLD,
            "stale_rate_warning": STALE_RATE_WARN_THRESHOLD,
            "low_quality_rate_warning": LOW_QUALITY_RATE_WARN_THRESHOLD,
            "max_rows_scanned": max_rows,
        },
    }


def _counter(values: Iterable[Any]) -> Dict[str, int]:
    counts: Counter[str] = Counter(_source_key(value) for value in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _source_key(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _source_health(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    missing = sum(1 for row in rows if _source_key(row.get(key)) == "unknown")
    total = len(rows)
    return {"success": max(0, total - missing), "failure": missing}


def _quality_buckets(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    buckets = {
        "0.00-0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.75": 0,
        "0.75-1.00": 0,
        "unknown": 0,
    }
    for row in rows:
        score = _safe_float(row.get("data_quality_score"), default=None)
        if score is None:
            buckets["unknown"] += 1
        elif score < 0.25:
            buckets["0.00-0.25"] += 1
        elif score < 0.50:
            buckets["0.25-0.50"] += 1
        elif score < 0.75:
            buckets["0.50-0.75"] += 1
        else:
            buckets["0.75-1.00"] += 1
    return buckets


def _missing_option_chain(row: Dict[str, Any]) -> bool:
    if _source_key(row.get("option_source")) == "unknown":
        return True
    snapshot = row.get("vol_snapshot_json") or {}
    null_reasons = snapshot.get("null_reasons") or {}
    flags = snapshot.get("data_quality_flags") or []
    if "missing_option_chain" in flags:
        return True
    if null_reasons.get("option_chain") == "missing_option_chain":
        return True
    if snapshot.get("near_term_atm_iv") is None and snapshot.get("iv30") is None:
        return True
    return False


def _is_low_quality(row: Dict[str, Any]) -> bool:
    score = _safe_float(row.get("data_quality_score"), default=None)
    return score is not None and score < LOW_DATA_QUALITY_THRESHOLD


def _weak_row_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "recommendation_id": row.get("recommendation_id"),
        "created_at": row.get("created_at"),
        "symbol": row.get("symbol"),
        "recommendation": row.get("recommendation"),
        "selected_structure": row.get("selected_structure"),
        "no_trade_reason": row.get("no_trade_reason"),
        "data_quality_score": row.get("data_quality_score"),
        "earnings_source": row.get("earnings_source"),
        "earnings_source_confidence": row.get("earnings_source_confidence"),
        "earnings_source_stale": bool(row.get("earnings_source_stale")),
        "option_source": row.get("option_source"),
        "underlying_source": row.get("underlying_source"),
        "quote_source": row.get("quote_source"),
        "surface_quality_status": row.get("surface_quality_status"),
        "weak_data_reasons": _weak_reasons(row),
    }


def _weak_reasons(row: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if _is_low_quality(row):
        reasons.append("low_data_quality_score")
    if bool(row.get("earnings_source_stale")):
        reasons.append("stale_earnings_source")
    if _missing_option_chain(row):
        reasons.append("missing_option_chain")
    if _safe_float(row.get("earnings_source_confidence"), default=1.0) < 0.50:
        reasons.append("low_earnings_source_confidence")
    if _poor_surface(row):
        reasons.append("poor_option_surface_quality")
    return reasons


def _poor_surface(row: Dict[str, Any]) -> bool:
    return str(row.get("surface_quality_status") or "") in {"record_only", "degraded_surface"}


def _safe_float(value: Any, *, default: Optional[float]) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else default
    except Exception:
        return default


def _ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0
