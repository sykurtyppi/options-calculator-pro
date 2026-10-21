"""Versioned, content-addressed evidence contracts for workstation integrations."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from web.api.schemas import EdgeAnalyzeRequest, EdgeAnalyzeResponse

WORKSTATION_EVIDENCE_SCHEMA_VERSION = "workstation.evidence.v1"
EDGE_ANALYZE_NATIVE_SCHEMA = "options-calculator-pro.edge-analyze.v1"
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
Sha256Digest = Annotated[str, StringConstraints(pattern=_SHA256_PATTERN)]


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class EvidenceProducer(_FrozenStrictModel):
    system: str
    version: str


class EvidenceLocalIdentifier(_FrozenStrictModel):
    scheme: str
    value: str


class EvidenceSubject(_FrozenStrictModel):
    entity_type: Literal["issuer", "security"]
    local_identifier: EvidenceLocalIdentifier
    canonical_identifier: Optional[str]
    mapping_status: Literal["resolved", "unresolved"]


class EvidenceTemporalContext(_FrozenStrictModel):
    generated_at: datetime
    as_of: date
    as_of_basis: str
    horizon: str
    event_date: Optional[date] = None
    fiscal_labels: list[str] = Field(default_factory=list)
    first_period_end: Optional[date] = None
    last_period_end: Optional[date] = None

    @field_validator("generated_at")
    @classmethod
    def normalize_generated_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value.astimezone(timezone.utc)


class EvidenceMethodology(_FrozenStrictModel):
    label: str
    version: str


class EvidenceEpistemicContext(_FrozenStrictModel):
    evidence_class: Literal["analytical_signal"]
    methodology: EvidenceMethodology
    limitations: list[str] = Field(default_factory=list)
    confidence: dict[str, JsonValue] = Field(default_factory=dict)
    uncertainty: dict[str, JsonValue] = Field(default_factory=dict)
    provenance_references: list[dict[str, JsonValue]] = Field(default_factory=list)
    unavailable_metrics: list[dict[str, JsonValue]] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    blocking_quality_reasons: list[str] = Field(default_factory=list)


class WorkstationEvidenceEnvelope(_FrozenStrictModel):
    """Cross-tool evidence envelope; never a human investment decision."""

    schema_version: Literal["workstation.evidence.v1"] = WORKSTATION_EVIDENCE_SCHEMA_VERSION
    evidence_id: Sha256Digest
    input_digest: Sha256Digest
    producer: EvidenceProducer
    subject: EvidenceSubject
    temporal: EvidenceTemporalContext
    epistemic: EvidenceEpistemicContext
    native_payload_schema: Literal["options-calculator-pro.edge-analyze.v1"]
    native_payload: dict[str, JsonValue]
    human_decision: None = None

    @model_validator(mode="after")
    def validate_evidence_id(self) -> "WorkstationEvidenceEnvelope":
        expected = _sha256_digest(self.model_dump(mode="json", exclude={"evidence_id"}))
        if self.evidence_id != expected:
            raise ValueError("evidence_id does not match the serialized envelope content")
        return self


def _deterministic_json_bytes(value: JsonValue | dict[str, Any]) -> bytes:
    """Serialize for OCP-local hashing; this profile is not RFC 8785."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_digest(value: JsonValue | dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_deterministic_json_bytes(value)).hexdigest()


def _analysis_input_digest(request: EdgeAnalyzeRequest) -> str:
    # EdgeAnalyzeRequest currently contains only the non-secret analysis input.
    # Keep this explicit so credentials can never be absorbed into evidence IDs if
    # transport/auth fields are added to another request surface in the future.
    analysis_input: dict[str, JsonValue] = {"symbol": request.symbol}
    return _sha256_digest(analysis_input)


def _first_date(candidates: list[tuple[str, Any]]) -> tuple[Optional[date], Optional[str]]:
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _dict_list(value: Any) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def build_workstation_evidence_envelope(
    request: EdgeAnalyzeRequest,
    native_payload: EdgeAnalyzeResponse,
    *,
    producer_version: str,
) -> WorkstationEvidenceEnvelope:
    """Bind one non-secret analysis request to its complete native response."""
    metrics = native_payload.metrics or {}
    selector = native_payload.selector_output or {}
    vol = native_payload.vol_snapshot or {}
    generated_at = native_payload.generated_at
    if generated_at.tzinfo is not None and generated_at.utcoffset() is not None:
        generated_at = generated_at.astimezone(timezone.utc)

    as_of, as_of_basis = _first_date([
        ("native_payload.vol_snapshot.as_of_date", vol.get("as_of_date")),
        ("native_payload.selector_output.as_of", selector.get("as_of")),
        ("native_payload.metrics.as_of_date", metrics.get("as_of_date")),
    ])
    if as_of is None:
        as_of = generated_at.date()
        as_of_basis = "native_payload.generated_at_date_fallback"
    assert as_of_basis is not None

    event_date, _ = _first_date([
        ("native_payload.vol_snapshot.earnings_date", vol.get("earnings_date")),
        ("native_payload.selector_output.earnings_date", selector.get("earnings_date")),
        ("native_payload.metrics.earnings_date", metrics.get("earnings_date")),
    ])

    confidence: dict[str, JsonValue] = {"confidence_pct": native_payload.confidence_pct}
    for key in ("sample_confidence", "confidence_label", "confidence_reasons"):
        if key in metrics:
            confidence[key] = metrics[key]

    uncertainty: dict[str, JsonValue] = {
        key: value for key, value in metrics.items() if "uncertainty" in key.lower()
    }
    uncertainty["model_scores"] = {"setup_score": native_payload.setup_score}

    provenance = {
        key: metrics[key]
        for key in ("data_source", "data_sources", "provider", "providers", "source_provenance")
        if key in metrics
    }
    provenance_references = [provenance] if provenance else []
    limitations = _dedupe(
        _string_list(metrics.get("limitations"))
        + _string_list(selector.get("limitations"))
        + _string_list(vol.get("limitations"))
    )
    unavailable_metrics = (
        _dict_list(metrics.get("unavailable_metrics"))
        + _dict_list(selector.get("unavailable_metrics"))
        + _dict_list(vol.get("unavailable_metrics"))
    )
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

    methodology_label = selector.get("methodology")
    if not isinstance(methodology_label, str) or not methodology_label:
        methodology_label = "options-calculator-pro.edge-analysis"

    values: dict[str, Any] = {
        "schema_version": WORKSTATION_EVIDENCE_SCHEMA_VERSION,
        "input_digest": _analysis_input_digest(request),
        "producer": EvidenceProducer(system="options-calculator-pro", version=producer_version),
        "subject": EvidenceSubject(
            entity_type="security",
            local_identifier=EvidenceLocalIdentifier(scheme="ticker", value=native_payload.symbol),
            canonical_identifier=None,
            mapping_status="unresolved",
        ),
        "temporal": EvidenceTemporalContext(
            generated_at=generated_at,
            as_of=as_of,
            as_of_basis=as_of_basis,
            horizon="earnings_event",
            event_date=event_date,
        ),
        "epistemic": EvidenceEpistemicContext(
            evidence_class="analytical_signal",
            methodology=EvidenceMethodology(label=methodology_label, version=producer_version),
            limitations=limitations,
            confidence=confidence,
            uncertainty=uncertainty,
            provenance_references=provenance_references,
            unavailable_metrics=unavailable_metrics,
            invalidation_conditions=invalidation_conditions,
            blocking_quality_reasons=blocking_quality_reasons,
        ),
        "native_payload_schema": EDGE_ANALYZE_NATIVE_SCHEMA,
        "native_payload": native_payload.model_dump(mode="json"),
        "human_decision": None,
    }
    draft = WorkstationEvidenceEnvelope.model_construct(
        evidence_id="sha256:" + ("0" * 64),
        **values,
    )
    evidence_id = _sha256_digest(draft.model_dump(mode="json", exclude={"evidence_id"}))
    return WorkstationEvidenceEnvelope(evidence_id=evidence_id, **values)
