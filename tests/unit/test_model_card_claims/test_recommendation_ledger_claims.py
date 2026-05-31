"""
Phase 2.3 — Model card claim audit: recommendation_ledger.

The card's Validation Approach section states:
  "Tests cover record writing, stale/provenance preservation, no-trade records,
   schema migration, exports, and linkage."

Each function below is named after one of those claims and exercises it directly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        metrics={"data_sources": {"options_source": "marketdata_app",
                                  "price_rv_source": "yfinance"}},
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
            {"structure": "atm_straddle", "eligible": True,
             "composite_structure_score": 0.81},
            {"structure": "call_calendar", "eligible": False,
             "eligibility_flags": ["cannot_form_required_structure"]},
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


def _rec_id(structure: str = "atm_straddle") -> str:
    return make_recommendation_id(
        symbol="AAPL",
        as_of_date="2026-04-23",
        earnings_date="2026-05-01",
        selected_structure=structure,
    )


def test_record_writing(tmp_path: Path) -> None:
    """record_recommendation() persists a row that is readable back from the ledger."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    rec_id = _rec_id()
    returned_id = record_recommendation(_analysis(), ledger=ledger,
                                        recommendation_id=rec_id)
    assert returned_id == rec_id
    rows = ledger.list_recent(limit=10)
    assert any(r["recommendation_id"] == rec_id for r in rows), (
        "written record must appear in list_recent()"
    )


def test_stale_provenance_preservation(tmp_path: Path) -> None:
    """earnings_source_stale=True is preserved exactly on write and round-trip read."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    rec_id = _rec_id()
    record_recommendation(_analysis(stale=True), ledger=ledger,
                          recommendation_id=rec_id)
    rows = ledger.list_recent(limit=10)
    row = next(r for r in rows if r["recommendation_id"] == rec_id)
    assert row["earnings_source_stale"] is True


def test_no_trade_records(tmp_path: Path) -> None:
    """No-trade recommendations are written and retrievable with abstain metadata."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure=None,
    )
    record_recommendation(_analysis(recommendation="No Trade"),
                          ledger=ledger, recommendation_id=rec_id)
    rows = ledger.list_recent(limit=10)
    row = next((r for r in rows if r["recommendation_id"] == rec_id), None)
    assert row is not None, "No Trade record must be persisted"
    assert row["recommendation"] == "No Trade"


def test_schema_migration(tmp_path: Path) -> None:
    """Opening a DB that is missing a column triggers migration and adds the column."""
    import sqlite3
    db_path = tmp_path / "ledger.sqlite"
    # Create a bare DB that lacks picker_provenance_json (added in a migration)
    con = sqlite3.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            recommendation_id TEXT PRIMARY KEY,
            created_at TEXT,
            symbol TEXT,
            as_of_date TEXT,
            earnings_date TEXT,
            earnings_source TEXT,
            earnings_source_confidence REAL,
            earnings_source_stale INTEGER NOT NULL DEFAULT 0,
            recommendation TEXT,
            setup_score REAL,
            rationale_json TEXT,
            metrics_json TEXT,
            selector_output_json TEXT,
            structure_scorecards_json TEXT,
            vol_snapshot_json TEXT
        )
    """)
    con.commit()
    con.close()

    # Opening via RecommendationLedger must not raise; migration adds missing columns
    ledger = RecommendationLedger(ledger_path=db_path)
    rec_id = _rec_id()
    returned_id = record_recommendation(_analysis(), ledger=ledger,
                                        recommendation_id=rec_id)
    assert returned_id == rec_id


def test_exports(tmp_path: Path) -> None:
    """list_recent() returns rows with the expected field set for export consumers."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    rec_id = _rec_id()
    record_recommendation(_analysis(), ledger=ledger, recommendation_id=rec_id)
    rows = ledger.list_recent(limit=100)
    assert len(rows) == 1
    row = rows[0]
    for field in ("recommendation_id", "symbol", "recommendation",
                  "as_of_date", "earnings_source_stale"):
        assert field in row, f"export field {field!r} missing from list_recent() row"


def test_linkage(tmp_path: Path) -> None:
    """candidate_shadow_outcome is written and read back when present in analysis."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    shadow = {"shadow_structure": "otm_strangle", "shadow_score": 0.61,
              "shadow_expected_edge_pct": 2.1}
    # record_recommendation reads candidate_shadow_outcome directly from analysis
    analysis.candidate_shadow_outcome = shadow
    rec_id = _rec_id()
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)

    rows = ledger.list_recent(limit=10)
    row = next(r for r in rows if r["recommendation_id"] == rec_id)
    # _row_to_dict parses candidate_shadow_outcome_json to a dict already
    stored = row.get("candidate_shadow_outcome_json") or {}
    assert stored.get("shadow_structure") == "otm_strangle"
