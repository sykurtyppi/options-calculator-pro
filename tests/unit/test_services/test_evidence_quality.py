from __future__ import annotations

from services.evidence_quality import (
    DEGRADED_EVIDENCE,
    RECORD_ONLY,
    VALID_EVIDENCE,
    evaluate_evidence_quality,
)


def _quote(*, source: str = "broker", quality: str = "execution_grade", legs: dict | None = None) -> dict:
    return {
        "quote_source": source,
        "quote_quality": quality,
        "bid_ask_mid": {"legs": legs or {"call": {"bid": 1.0, "ask": 1.2, "mid": 1.1}}},
    }


def _clean_surface() -> dict:
    # Mirrors diagnose_option_surface_quality's clean output: a recognized
    # status plus zeroed counts. The live snapshot pipeline always attaches one.
    return {
        "status": "clean_surface",
        "warning_flags": [],
        "crossed_quote_count": 0,
        "extreme_spread_count": 0,
        "missing_iv_count": 0,
        "iv_outlier_count": 0,
        "sparse_atm_expiration_count": 0,
    }


def test_execution_grade_quote_with_clean_surface_can_support_claims() -> None:
    result = evaluate_evidence_quality(
        quote_payload=_quote(),
        surface_quality=_clean_surface(),
    ).to_dict()

    assert result["evidence_quality_status"] == VALID_EVIDENCE
    assert result["execution_grade"] is True
    assert result["claim_allowed"] is True


def test_absent_surface_is_degraded_not_valid() -> None:
    # M5: a missing surface dict must NOT pass as valid evidence. Missing
    # surface fields were previously read as zero ("no problems"), so a
    # degraded/missing chain could look clean. Absence is now degraded.
    result = evaluate_evidence_quality(quote_payload=_quote()).to_dict()

    assert result["evidence_quality_status"] == DEGRADED_EVIDENCE
    assert result["claim_allowed"] is False
    assert "surface_quality_unavailable" in result["evidence_quality_reasons"]


def test_surface_present_without_status_is_degraded() -> None:
    # An incomplete surface dict (counts present, status missing) is "unknown",
    # not clean — the missing status must not be treated as clean_surface.
    result = evaluate_evidence_quality(
        quote_payload=_quote(),
        surface_quality={"crossed_quote_count": 0, "extreme_spread_count": 0},
    ).to_dict()

    assert result["evidence_quality_status"] == DEGRADED_EVIDENCE
    assert result["claim_allowed"] is False
    assert "surface_quality_status_unknown" in result["evidence_quality_reasons"]


def test_degraded_surface_status_is_degraded() -> None:
    result = evaluate_evidence_quality(
        quote_payload=_quote(),
        surface_quality={**_clean_surface(), "status": "degraded_surface"},
    ).to_dict()

    assert result["evidence_quality_status"] == DEGRADED_EVIDENCE
    assert "surface_quality_degraded_surface" in result["evidence_quality_reasons"]


def test_research_grade_provider_is_degraded_not_claimable() -> None:
    result = evaluate_evidence_quality(
        quote_payload=_quote(source="yfinance", quality="paper_research_mid_not_execution_grade")
    ).to_dict()

    assert result["evidence_quality_status"] == DEGRADED_EVIDENCE
    assert result["claim_allowed"] is False
    assert "provider_research_grade_yfinance" in result["evidence_quality_reasons"]
    assert "quote_not_execution_grade" in result["evidence_quality_reasons"]


def test_crossed_quote_is_record_only() -> None:
    result = evaluate_evidence_quality(
        quote_payload=_quote(legs={"call": {"bid": 1.4, "ask": 1.2, "mid": 1.3}})
    ).to_dict()

    assert result["evidence_quality_status"] == RECORD_ONLY
    assert "call_crossed_or_invalid_bid_ask" in result["evidence_quality_reasons"]


def test_zero_bid_with_untradeably_wide_spread_is_record_only() -> None:
    result = evaluate_evidence_quality(
        quote_payload=_quote(legs={"call": {"bid": 0.0, "ask": 0.2, "mid": 0.1}})
    ).to_dict()

    assert result["evidence_quality_status"] == RECORD_ONLY
    assert "call_zero_bid" in result["evidence_quality_reasons"]
    assert "call_blocking_extreme_spread" in result["evidence_quality_reasons"]


def test_blocking_extreme_spread_is_record_only() -> None:
    result = evaluate_evidence_quality(
        quote_payload=_quote(legs={"call": {"bid": 0.1, "ask": 1.9, "mid": 1.0}})
    ).to_dict()

    assert result["evidence_quality_status"] == RECORD_ONLY
    assert "call_blocking_extreme_spread" in result["evidence_quality_reasons"]


def test_snapshot_staleness_and_low_confidence_degrade_evidence() -> None:
    result = evaluate_evidence_quality(
        quote_payload=_quote(),
        vol_snapshot={
            "data_quality_score": 0.6,
            "earnings_source_stale": True,
            "earnings_source_confidence": 0.4,
            "historical_move_source": "daily_fallback",
        },
    ).to_dict()

    assert result["evidence_quality_status"] == DEGRADED_EVIDENCE
    assert "low_snapshot_data_quality" in result["evidence_quality_reasons"]
    assert "stale_earnings_source" in result["evidence_quality_reasons"]
    assert "low_earnings_source_confidence" in result["evidence_quality_reasons"]
    assert "historical_move_daily_fallback" in result["evidence_quality_reasons"]


def test_poor_surface_quality_blocks_claims_and_can_make_record_only() -> None:
    result = evaluate_evidence_quality(
        quote_payload=_quote(),
        surface_quality={
            "status": "record_only",
            "warning_flags": ["crossed_quotes", "sparse_strikes_around_atm"],
            "crossed_quote_count": 2,
            "extreme_spread_count": 3,
            "sparse_atm_expiration_count": 1,
        },
    ).to_dict()

    assert result["evidence_quality_status"] == RECORD_ONLY
    assert result["claim_allowed"] is False
    assert "surface_quality_record_only" in result["evidence_quality_reasons"]
    assert "surface_crossed_quotes" in result["evidence_quality_reasons"]
    assert "surface_sparse_atm" in result["evidence_quality_reasons"]
