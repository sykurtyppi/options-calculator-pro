"""Versioned, content-addressed evidence contracts for workstation integrations."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from web.api.schemas import EdgeAnalyzeResponse

WORKSTATION_EVIDENCE_SCHEMA_VERSION = "workstation.evidence.v1"
EDGE_ANALYZE_NATIVE_SCHEMA = "options-calculator-pro.edge-analyze.v1"


class EvidenceProducer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: Literal["options-calculator-pro"] = "options-calculator-pro"
    version: str
    native_output_schema: Literal["options-calculator-pro.edge-analyze.v1"] = EDGE_ANALYZE_NATIVE_SCHEMA


class EvidenceSubject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: Literal["security"] = "security"
    local_identifier: str
    scheme: Literal["ticker"] = "ticker"
    canonical_identifier: None = None
    mapping_status: Literal["unresolved"] = "unresolved"


class EvidenceTemporalContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    as_of: date
    as_of_basis: str
    horizon: Literal["earnings_event"] = "earnings_event"
    event_date: Optional[date] = None


class EvidenceEpistemicContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_class: Literal["analytical_signal"] = "analytical_signal"
    methodology: str
    confidence: Dict[str, Any] = Field(default_factory=dict)
    uncertainty: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    invalidation_conditions: List[str] = Field(default_factory=list)
    blocking_quality_reasons: List[str] = Field(default_factory=list)


class WorkstationEvidenceEnvelope(BaseModel):
    """Cross-tool evidence envelope; never a human investment decision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["workstation.evidence.v1"] = WORKSTATION_EVIDENCE_SCHEMA_VERSION
    evidence_id: str
    producer: EvidenceProducer
    subject: EvidenceSubject
    temporal: EvidenceTemporalContext
    epistemic: EvidenceEpistemicContext
    native_payload_schema: Literal["options-calculator-pro.edge-analyze.v1"] = EDGE_ANALYZE_NATIVE_SCHEMA
    native_payload: EdgeAnalyzeResponse
    human_decision: None = None


def _first_date(candidates: List[tuple[str, Any]]) -> tuple[Optional[date], Optional[str]]:
    for basis, value in candidates:
        if value in (None, ""):
            continue
        try:
            if isinstance(value, datetime):
                return value.date(), basis
            if isinstance(value, date):
                return value, basis
            return date.fromisoformat(str(value)[:10]), basis
        except (TypeError, ValueError):
            continue
    return None, None


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _canonical_content(envelope: WorkstationEvidenceEnvelope) -> str:
    content = envelope.model_dump(mode="json", exclude={"evidence_id"})
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_workstation_evidence_envelope(
    native_payload: EdgeAnalyzeResponse,
    *,
    producer_version: str,
) -> WorkstationEvidenceEnvelope:
    """Wrap an edge analysis without changing or interpreting its native output."""
    metrics = native_payload.metrics or {}
    selector = native_payload.selector_output or {}
    vol = native_payload.vol_snapshot or {}

    as_of, as_of_basis = _first_date([
        ("native_payload.vol_snapshot.as_of_date", vol.get("as_of_date")),
        ("native_payload.selector_output.as_of", selector.get("as_of")),
        ("native_payload.metrics.as_of_date", metrics.get("as_of_date")),
    ])
    if as_of is None:
        as_of = native_payload.generated_at.date()
        as_of_basis = "native_payload.generated_at_date_fallback"
    assert as_of_basis is not None

    event_date, _ = _first_date([
        ("native_payload.vol_snapshot.earnings_date", vol.get("earnings_date")),
        ("native_payload.selector_output.earnings_date", selector.get("earnings_date")),
        ("native_payload.metrics.earnings_date", metrics.get("earnings_date")),
    ])

    confidence: Dict[str, Any] = {
        "confidence_pct": native_payload.confidence_pct,
        "setup_score": native_payload.setup_score,
    }
    for key in ("sample_confidence", "confidence_label", "confidence_reasons"):
        if key in metrics:
            confidence[key] = metrics[key]

    uncertainty = {
        key: value
        for key, value in metrics.items()
        if "uncertainty" in key.lower()
    }
    provenance = {
        key: metrics[key]
        for key in ("data_source", "data_sources", "provider", "providers", "source_provenance")
        if key in metrics
    }
    invalidation_conditions = _dedupe(
        _string_list(metrics.get("hard_gate_reasons"))
        + _string_list(metrics.get("invalidation_conditions"))
        + _string_list(selector.get("binding_gates"))
        + (
            [selector["binding_gate"]]
            if isinstance(selector.get("binding_gate"), str) and selector["binding_gate"]
            else []
        )
        + _string_list(selector.get("invalidation_conditions"))
    )
    blocking_quality_reasons = _dedupe(
        _string_list(vol.get("surface_quality_reasons"))
        + _string_list(vol.get("quality_reasons"))
        + _string_list(metrics.get("quality_reasons"))
        + (_string_list(metrics.get("hard_gate_reasons")) if metrics.get("hard_no_trade") else [])
    )

    methodology = selector.get("methodology")
    if not isinstance(methodology, str) or not methodology:
        methodology = "options-calculator-pro.edge-analysis"

    envelope = WorkstationEvidenceEnvelope(
        evidence_id="sha256:" + ("0" * 64),
        producer=EvidenceProducer(version=producer_version),
        subject=EvidenceSubject(local_identifier=native_payload.symbol),
        temporal=EvidenceTemporalContext(
            generated_at=native_payload.generated_at,
            as_of=as_of,
            as_of_basis=as_of_basis,
            event_date=event_date,
        ),
        epistemic=EvidenceEpistemicContext(
            methodology=methodology,
            confidence=confidence,
            uncertainty=uncertainty,
            provenance=provenance,
            invalidation_conditions=invalidation_conditions,
            blocking_quality_reasons=blocking_quality_reasons,
        ),
        native_payload=native_payload,
    )
    digest = hashlib.sha256(_canonical_content(envelope).encode("utf-8")).hexdigest()
    return envelope.model_copy(update={"evidence_id": f"sha256:{digest}"})
