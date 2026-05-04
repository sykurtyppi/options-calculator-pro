from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from services.recommendation_ledger import (
    RecommendationLedger,
    make_recommendation_id,
    record_recommendation,
)


def _analysis(*, recommendation: str = "Candidate", stale: bool = False) -> SimpleNamespace:
    earnings_date = date(2026, 5, 1)
    return SimpleNamespace(
        symbol="AAPL",
        recommendation=recommendation,
        setup_score=0.72,
        metrics={"data_sources": {"options_source": "marketdata_app", "price_rv_source": "yfinance"}},
        rationale=["selector rationale"],
        selector_output={
            "recommendation": recommendation,
            "best_structure": "atm_straddle" if recommendation != "No Trade" else None,
            "earnings_date": earnings_date.isoformat(),
            "primary_thesis": "ATM straddle fits the evidence.",
            "primary_risks": ["Execution risk."],
            "why_this_structure": ["Move evidence is strongest."],
            "why_not_others": {"call_calendar": ["Lower score."]},
        },
        structure_scorecards=[
            {"structure": "atm_straddle", "eligible": True, "composite_structure_score": 0.81},
            {"structure": "call_calendar", "eligible": False, "eligibility_flags": ["cannot_form_required_structure"]},
        ],
        vol_snapshot={
            "symbol": "AAPL",
            "as_of_date": "2026-04-23",
            "earnings_date": earnings_date.isoformat(),
            "earnings_source_primary": "alpha_vantage",
            "earnings_source_confidence": 0.82,
            "earnings_source_stale": stale,
            "option_source": "marketdata_app",
            "underlying_source": "yfinance",
            "data_quality_score": 0.77,
            "iv_rv_yz": 0.92,
            "near_term_spread_pct": 2.4,
        },
    )


def test_recommendation_records_are_written_with_provenance_and_quotes(tmp_path: Path) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis(stale=True)
    recommendation_id = make_recommendation_id(
        symbol="AAPL",
        as_of_date="2026-04-23",
        earnings_date="2026-05-01",
        selected_structure="atm_straddle",
    )

    returned_id = record_recommendation(
        analysis,
        ledger=ledger,
        recommendation_id=recommendation_id,
        quote_payload={
            "quote_source": "yfinance",
            "quote_timestamp": "2026-04-23T12:00:00+00:00",
            "quote_quality": "paper_research_mid_not_execution_grade",
            "bid_ask_mid": {"legs": {"call": {"bid": 4.0, "ask": 4.2, "mid": 4.1}}},
            "surface_quality": {
                "status": "degraded_surface",
                "warning_flags": ["extreme_bid_ask_spreads"],
                "extreme_spread_count": 2,
            },
        },
        metadata={"source": "unit_test"},
    )

    assert returned_id == recommendation_id
    assert ledger.count() == 1

    row = ledger.get(recommendation_id)
    assert row is not None
    assert row["symbol"] == "AAPL"
    assert row["earnings_source"] == "alpha_vantage"
    assert row["earnings_source_confidence"] == 0.82
    assert row["earnings_source_stale"] is True
    assert row["quote_source"] == "yfinance"
    assert row["quote_quality"] == "paper_research_mid_not_execution_grade"
    assert row["bid_ask_mid_json"]["legs"]["call"]["mid"] == 4.1
    assert row["surface_quality_status"] == "degraded_surface"
    assert row["surface_extreme_spread_count"] == 2
    assert row["surface_quality_json"]["warning_flags"] == ["extreme_bid_ask_spreads"]
    assert row["vol_snapshot_json"]["iv_rv_yz"] == 0.92
    assert row["structure_scorecards_json"][0]["structure"] == "atm_straddle"


def test_no_trade_recommendations_are_recorded_with_abstain_reason(tmp_path: Path) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    recommendation_id = record_recommendation(
        _analysis(recommendation="No Trade"),
        ledger=ledger,
        recommendation_id="rec_no_trade",
    )

    row = ledger.get(recommendation_id)
    assert row is not None
    assert row["recommendation"] == "No Trade"
    assert row["selected_structure"] is None
    assert row["no_trade_reason"]


def test_schema_migration_initializes_missing_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_ledger.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE recommendations (recommendation_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO recommendations (recommendation_id) VALUES ('legacy')")
    conn.commit()
    conn.close()

    ledger = RecommendationLedger(ledger_path=db_path)
    columns = {
        row[1]
        for row in ledger._conn.execute("PRAGMA table_info(recommendations)").fetchall()  # noqa: SLF001
    }

    assert "earnings_source_confidence" in columns
    assert "surface_quality_status" in columns
    assert "structure_scorecards_json" in columns
    assert "schema_version" in columns
