from __future__ import annotations

from services.data_quality_diagnostics import build_data_quality_diagnostics
from services.recommendation_ledger import RecommendationLedger, RecommendationRecord


def _record(
    recommendation_id: str,
    *,
    symbol: str,
    score: float | None,
    earnings_source: str | None = "alpha_vantage",
    stale: bool = False,
    option_source: str | None = "marketdata_app",
    underlying_source: str | None = "yfinance",
    quote_source: str | None = "yfinance",
    earnings_confidence: float | None = 0.8,
    snapshot_extra: dict | None = None,
) -> RecommendationRecord:
    snapshot = {
        "symbol": symbol,
        "as_of_date": "2026-04-24",
        "earnings_date": "2026-05-01",
        "earnings_source_primary": earnings_source,
        "earnings_source_confidence": earnings_confidence,
        "earnings_source_stale": stale,
        "option_source": option_source,
        "underlying_source": underlying_source,
        "data_quality_score": score,
        "iv30": 0.42,
        "near_term_atm_iv": 0.44,
        **(snapshot_extra or {}),
    }
    return RecommendationRecord(
        recommendation_id=recommendation_id,
        created_at=f"2026-04-24T12:0{recommendation_id[-1]}:00+00:00",
        symbol=symbol,
        as_of_date="2026-04-24",
        earnings_date="2026-05-01",
        earnings_source=earnings_source,
        earnings_source_confidence=earnings_confidence,
        earnings_source_stale=stale,
        recommendation="Candidate" if score is None or score >= 0.5 else "No Trade",
        selected_structure="atm_straddle" if score is None or score >= 0.5 else None,
        no_trade_reason="Data quality too low." if score is not None and score < 0.5 else None,
        data_quality_score=score,
        option_source=option_source,
        underlying_source=underlying_source,
        provider_names={
            "option_source": option_source,
            "underlying_source": underlying_source,
            "earnings_source": earnings_source,
            "quote_source": quote_source,
        },
        quote_source=quote_source,
        quote_quality="paper_research_mid_not_execution_grade" if quote_source else None,
        vol_snapshot=snapshot,
        structure_scorecards=[],
        selector_output={"recommendation": "Candidate"},
    )


def test_data_quality_diagnostics_aggregates_provider_and_quality_counts(tmp_path):
    ledger = RecommendationLedger(tmp_path / "ledger.sqlite")
    ledger.record(_record("rec_1", symbol="AAPL", score=0.82))
    ledger.record(_record("rec_2", symbol="MSFT", score=0.42, stale=True, earnings_confidence=0.35))
    ledger.record(
        _record(
            "rec_3",
            symbol="TSLA",
            score=0.20,
            earnings_source=None,
            option_source=None,
            quote_source=None,
            snapshot_extra={"iv30": None, "near_term_atm_iv": None, "data_quality_flags": ["missing_option_chain"]},
        )
    )

    payload = build_data_quality_diagnostics(ledger=ledger)

    assert payload["total_recommendations"] == 3
    assert payload["stale_earnings_source_count"] == 1
    assert payload["missing_option_chain_count"] == 1
    assert payload["low_data_quality_count"] == 2
    assert payload["data_quality_buckets"]["0.00-0.25"] == 1
    assert payload["data_quality_buckets"]["0.25-0.50"] == 1
    assert payload["data_quality_buckets"]["0.75-1.00"] == 1
    assert payload["source_breakdown"]["option_source"]["marketdata_app"] == 2
    assert payload["source_breakdown"]["option_source"]["unknown"] == 1
    assert payload["provider_health"]["option_source"]["success"] == 2
    assert payload["provider_health"]["option_source"]["failure"] == 1
    assert payload["provider_health"]["earnings_source"]["stale"] == 1
    assert "Low data-quality recommendation rate is elevated." in payload["warning_flags"]
    assert any(row["symbol"] == "TSLA" for row in payload["recent_weak_data_recommendations"])


def test_data_quality_diagnostics_empty_system_is_safe(tmp_path):
    ledger = RecommendationLedger(tmp_path / "ledger.sqlite")

    payload = build_data_quality_diagnostics(ledger=ledger)

    assert payload["total_recommendations"] == 0
    assert payload["stale_earnings_source_rate"] == 0.0
    assert payload["low_data_quality_rate"] == 0.0
    assert payload["data_quality_buckets"]["unknown"] == 0
    assert payload["recent_weak_data_recommendations"] == []
    assert "No recommendation records available for data-quality diagnostics." in payload["warning_flags"]
