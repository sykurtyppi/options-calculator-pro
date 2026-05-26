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


# ── PR-K: ledger immutability contract ────────────────────────────────────────


def _build_record(
    *,
    recommendation_id: str = "rec_immutable_test",
    recommendation: str = "Candidate",
    setup_score: float = 0.72,
    selected_structure: str = "atm_straddle",
    quote_quality: str = "paper_research_mid_not_execution_grade",
):
    from services.recommendation_ledger import build_record_from_analysis

    return build_record_from_analysis(
        SimpleNamespace(
            symbol="AAPL",
            recommendation=recommendation,
            setup_score=setup_score,
            metrics={
                "data_sources": {
                    "options_source": "marketdata_app",
                    "price_rv_source": "yfinance",
                }
            },
            rationale=["sel"],
            selector_output={
                "recommendation": recommendation,
                "best_structure": selected_structure,
                "earnings_date": "2026-05-01",
                "primary_thesis": "thesis",
                "primary_risks": ["r"],
                "why_this_structure": ["w"],
                "why_not_others": {},
            },
            structure_scorecards=[
                {
                    "structure": selected_structure,
                    "eligible": True,
                    "composite_structure_score": 0.81,
                }
            ],
            vol_snapshot={
                "symbol": "AAPL",
                "as_of_date": "2026-04-23",
                "earnings_date": "2026-05-01",
                "earnings_source_primary": "alpha_vantage",
                "earnings_source_confidence": 0.82,
                "earnings_source_stale": False,
                "option_source": "marketdata_app",
                "underlying_source": "yfinance",
                "data_quality_score": 0.77,
                "iv_rv_yz": 0.92,
                "near_term_spread_pct": 2.4,
            },
        ),
        recommendation_id=recommendation_id,
        quote_payload={
            "quote_source": "yfinance",
            "quote_timestamp": "2026-04-23T12:00:00+00:00",
            "quote_quality": quote_quality,
            "bid_ask_mid": {"legs": {"call": {"mid": 4.1}}},
            "surface_quality": {"status": "valid_evidence", "warning_flags": []},
        },
    )


def test_first_record_returns_inserted_and_persists_original(tmp_path: Path) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    status = ledger.record(_build_record(recommendation_id="rec_first"))
    assert status == "inserted"
    row = ledger.get("rec_first")
    assert row is not None
    assert row["symbol"] == "AAPL"

    # The revisions table also gets a "v1" entry on first record so the full
    # history is queryable from one place.
    revisions = ledger.get_revisions("rec_first")
    assert len(revisions) == 1


def test_identical_re_record_returns_duplicate(tmp_path: Path) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    record = _build_record(recommendation_id="rec_dup")
    assert ledger.record(record) == "inserted"
    assert ledger.record(record) == "duplicate"
    # Only one revision row (the original) — UNIQUE(rec_id, content_hash) dedup.
    assert len(ledger.get_revisions("rec_dup")) == 1


def test_content_change_records_revision_and_preserves_original(tmp_path: Path) -> None:
    """The critical contract: the original recommendations row must NEVER be
    overwritten. A subsequent record() with different content goes to the
    revisions table; the recommendations row remains the first-write payload.
    """
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    original = _build_record(
        recommendation_id="rec_evolves",
        quote_quality="paper_research_mid_not_execution_grade",
    )
    enriched = _build_record(
        recommendation_id="rec_evolves",
        quote_quality="execution_grade_bid_ask_mid",  # enrichment changes provenance
    )
    assert ledger.record(original) == "inserted"
    assert ledger.record(enriched) == "revision"

    # Original row in recommendations is untouched — this is the whole point.
    row = ledger.get("rec_evolves")
    assert row is not None
    assert row["quote_quality"] == "paper_research_mid_not_execution_grade"

    # Revisions table now has v1 (original) and v2 (enriched).
    revisions = ledger.get_revisions("rec_evolves")
    assert len(revisions) == 2
    assert revisions[0]["record"]["quote_quality"] == "paper_research_mid_not_execution_grade"
    assert revisions[1]["record"]["quote_quality"] == "execution_grade_bid_ask_mid"


def test_multiple_revisions_accumulate_in_order(tmp_path: Path) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_build_record(recommendation_id="rec_seq", quote_quality="v1"))
    ledger.record(_build_record(recommendation_id="rec_seq", quote_quality="v2"))
    ledger.record(_build_record(recommendation_id="rec_seq", quote_quality="v3"))

    revisions = ledger.get_revisions("rec_seq")
    assert len(revisions) == 3
    # Chronological order: v1 → v2 → v3.
    quote_qualities = [r["record"]["quote_quality"] for r in revisions]
    assert quote_qualities == ["v1", "v2", "v3"]


def test_byte_identical_revision_after_revision_is_deduped(tmp_path: Path) -> None:
    """The UNIQUE(recommendation_id, content_hash) constraint dedupes any
    re-record of an already-stored content_hash, regardless of how many
    revisions exist between."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    v1 = _build_record(recommendation_id="rec_dedup", quote_quality="v1")
    v2 = _build_record(recommendation_id="rec_dedup", quote_quality="v2")
    assert ledger.record(v1) == "inserted"
    assert ledger.record(v2) == "revision"
    # Re-record v2 → already in revisions table → duplicate.
    assert ledger.record(v2) == "duplicate"
    # Re-record v1 → still in revisions table → also a duplicate.
    assert ledger.record(v1) == "duplicate"
    assert len(ledger.get_revisions("rec_dedup")) == 2


def test_revisions_table_created_on_open(tmp_path: Path) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    tables = {
        row[0]
        for row in ledger._conn.execute(  # noqa: SLF001
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "recommendation_revisions" in tables
