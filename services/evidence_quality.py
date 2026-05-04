from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import math


VALID_EVIDENCE = "valid_evidence"
DEGRADED_EVIDENCE = "degraded_evidence"
RECORD_ONLY = "record_only"

EXTREME_SPREAD_PCT = 25.0
BLOCKING_SPREAD_PCT = 50.0
LOW_DATA_QUALITY_SCORE = 0.70
LOW_EARNINGS_CONFIDENCE = 0.70


@dataclass(frozen=True)
class EvidenceQualityResult:
    evidence_quality_status: str
    evidence_quality_reasons: list[str] = field(default_factory=list)
    claim_allowed: bool = False
    execution_grade: bool = False
    quote_issue_count: int = 0
    max_leg_spread_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_evidence_quality(
    *,
    quote_payload: Mapping[str, Any] | None,
    vol_snapshot: Mapping[str, Any] | None = None,
    surface_quality: Mapping[str, Any] | None = None,
    require_volume_open_interest: bool = False,
) -> EvidenceQualityResult:
    """Classify whether a paper record can support evidence claims.

    The gate is intentionally conservative. It never blocks storage; it marks
    whether a row is usable for claims versus useful only as an audit record.
    """
    quote = quote_payload or {}
    snapshot = vol_snapshot or {}
    surface = surface_quality or {}
    if not surface and isinstance(quote.get("surface_quality"), Mapping):
        surface = quote.get("surface_quality") or {}
    reasons: list[str] = []
    blocking = False
    degraded = False

    quote_source = str(quote.get("quote_source") or "").lower()
    quote_quality = str(quote.get("quote_quality") or "").lower()
    if not quote_source:
        reasons.append("missing_quote_source")
        degraded = True
    if quote_source == "yfinance":
        reasons.append("provider_research_grade_yfinance")
        degraded = True
    if "paper" in quote_quality or "research" in quote_quality or "not_execution" in quote_quality:
        reasons.append("quote_not_execution_grade")
        degraded = True

    bid_ask_mid = quote.get("bid_ask_mid") if isinstance(quote.get("bid_ask_mid"), Mapping) else {}
    legs = bid_ask_mid.get("legs") if isinstance(bid_ask_mid, Mapping) else {}
    if not isinstance(legs, Mapping) or not legs:
        reasons.append("missing_bid_ask_legs")
        blocking = True

    max_spread_pct: float | None = None
    for leg_name, raw_leg in (legs.items() if isinstance(legs, Mapping) else []):
        leg = raw_leg if isinstance(raw_leg, Mapping) else {}
        prefix = f"{leg_name}_"
        bid = _finite_float(leg.get("bid"))
        ask = _finite_float(leg.get("ask"))
        mid = _finite_float(leg.get("mid"))
        volume = _finite_float(leg.get("volume"))
        open_interest = _finite_float(leg.get("open_interest"))

        if bid is None or ask is None:
            reasons.append(prefix + "missing_bid_ask")
            blocking = True
            continue
        if bid < 0 or ask < 0 or ask < bid:
            reasons.append(prefix + "crossed_or_invalid_bid_ask")
            blocking = True
            continue
        if bid == 0:
            reasons.append(prefix + "zero_bid")
            degraded = True
        if mid is None or mid <= 0:
            reasons.append(prefix + "missing_or_non_positive_mid")
            blocking = True
            continue

        spread_pct = ((ask - bid) / mid) * 100.0 if mid > 0 else None
        if spread_pct is not None and math.isfinite(spread_pct):
            max_spread_pct = spread_pct if max_spread_pct is None else max(max_spread_pct, spread_pct)
            if spread_pct >= BLOCKING_SPREAD_PCT:
                reasons.append(prefix + "blocking_extreme_spread")
                blocking = True
            elif spread_pct >= EXTREME_SPREAD_PCT:
                reasons.append(prefix + "extreme_spread")
                degraded = True

        quality_label = str(leg.get("quote_quality_label") or "")
        if quality_label == "nearest_valid_wing_mid":
            reasons.append(prefix + "fallback_wing_selected")
            degraded = True
        elif quality_label == "partial_or_missing_mid":
            reasons.append(prefix + "partial_or_missing_mid")
            blocking = True

        if require_volume_open_interest and volume is None and open_interest is None:
            reasons.append(prefix + "missing_volume_open_interest")
            degraded = True

    data_quality_score = _finite_float(snapshot.get("data_quality_score"))
    if data_quality_score is not None and data_quality_score < LOW_DATA_QUALITY_SCORE:
        reasons.append("low_snapshot_data_quality")
        degraded = True
    if bool(snapshot.get("earnings_source_stale")):
        reasons.append("stale_earnings_source")
        degraded = True
    earnings_conf = _finite_float(snapshot.get("earnings_source_confidence"))
    if earnings_conf is not None and earnings_conf < LOW_EARNINGS_CONFIDENCE:
        reasons.append("low_earnings_source_confidence")
        degraded = True
    if str(snapshot.get("historical_move_source") or "") == "daily_fallback":
        reasons.append("historical_move_daily_fallback")
        degraded = True

    surface_status = str(surface.get("status") or "")
    if surface_status == "record_only":
        reasons.append("surface_quality_record_only")
        blocking = True
    elif surface_status and surface_status != "clean_surface":
        reasons.append(f"surface_quality_{surface_status}")
        degraded = True
    for flag in surface.get("warning_flags") or []:
        reasons.append(f"surface_{flag}")
        degraded = True
    if _finite_float(surface.get("crossed_quote_count")):
        reasons.append("surface_crossed_quotes")
        blocking = True
    if _finite_float(surface.get("extreme_spread_count")):
        reasons.append("surface_extreme_spreads")
        degraded = True
    if _finite_float(surface.get("sparse_atm_expiration_count")):
        reasons.append("surface_sparse_atm")
        degraded = True
    if _finite_float(surface.get("missing_iv_count")) or _finite_float(surface.get("iv_outlier_count")):
        reasons.append("surface_iv_anomalies")
        degraded = True

    status = RECORD_ONLY if blocking else DEGRADED_EVIDENCE if degraded else VALID_EVIDENCE
    execution_grade = (
        status == VALID_EVIDENCE
        and "execution_grade" in quote_quality
        and quote_source not in {"", "yfinance"}
    )
    claim_allowed = status == VALID_EVIDENCE and execution_grade
    return EvidenceQualityResult(
        evidence_quality_status=status,
        evidence_quality_reasons=sorted(set(reasons)),
        claim_allowed=bool(claim_allowed),
        execution_grade=bool(execution_grade),
        quote_issue_count=len(set(reasons)),
        max_leg_spread_pct=round(max_spread_pct, 6) if max_spread_pct is not None else None,
    )


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
