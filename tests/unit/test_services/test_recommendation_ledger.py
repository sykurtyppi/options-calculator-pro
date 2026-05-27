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


# ── PR-AC commit 4: picker_provenance round-trip ────────────────────────────


def _sample_experimental_contract_selection() -> dict:
    """A realistic experimental_contract_selection block shaped exactly
    like the one produced by web.api.edge_engine in commit 3. Kept here
    rather than imported so the ledger test stays decoupled from the
    edge_engine module."""
    return {
        "structure": "call_calendar",
        "labels": {
            "experimental": True,
            "shadow_mode": True,
            "not_execution_guidance": True,
            "out_of_sample_validated": False,
        },
        "note": "Experimental candidate contract selection.",
        "status": "ok",
        "candidate_contracts": {
            "shadow_status": "ok",
            "candidate_min_front_dte_days": 14,
            "experimental_note": "...",
            "legacy_selection": {
                "side": "call", "strike": 100.0,
                "front_expiry": "2026-05-03", "back_expiry": "2026-05-17",
                "front_dte_days": 2, "back_minus_front_days": 14,
                "picker_variant": "legacy_first_expiry",
                "picker_min_front_dte_days": 14, "picker_back_gap_days": 14,
                "front_leg": {"mid": 1.0, "bid": 0.95, "ask": 1.05,
                              "iv": 0.30, "spread_pct": 10.0,
                              "open_interest": 500, "volume": 50,
                              "delta": None, "dte": 2},
                "back_leg": {"mid": 2.0, "bid": 1.95, "ask": 2.05,
                             "iv": 0.30, "spread_pct": 5.0,
                             "open_interest": 800, "volume": 80,
                             "delta": None, "dte": 16},
            },
            "candidate_selection": {
                "side": "call", "strike": 100.0,
                "front_expiry": "2026-05-17", "back_expiry": "2026-06-21",
                "front_dte_days": 16, "back_minus_front_days": 35,
                "picker_variant": "candidate_min_dte",
                "picker_min_front_dte_days": 14, "picker_back_gap_days": 14,
                "front_leg": {"mid": 2.0, "bid": 1.95, "ask": 2.05,
                              "iv": 0.30, "spread_pct": 5.0,
                              "open_interest": 800, "volume": 80,
                              "delta": None, "dte": 16},
                "back_leg": {"mid": 3.0, "bid": 2.95, "ask": 3.05,
                             "iv": 0.30, "spread_pct": 3.3,
                             "open_interest": 1200, "volume": 120,
                             "delta": None, "dte": 51},
            },
            "pickers_diverged": True,
        },
    }


def test_picker_provenance_persists_when_present_in_analysis(tmp_path: Path) -> None:
    """Codex hard requirement for PR-AC commit 4: every recorded paper
    observation that displayed experimental picker metadata MUST persist
    picker_variant, shadow_mode, selected expiries, strike, and whether
    legacy/candidate diverged. Without this, the live API shows
    candidates but the system fails to accumulate auditable evidence."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    analysis.metrics["experimental_contract_selection"] = _sample_experimental_contract_selection()
    recommendation_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="atm_straddle",
        salt="picker-provenance-test",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=recommendation_id)
    row = ledger.get(recommendation_id)
    assert row is not None
    pp = row["picker_provenance_json"]
    # Round-trip preserves the labeled shadow surface
    assert pp["structure"] == "call_calendar"
    assert pp["labels"]["shadow_mode"] is True
    assert pp["labels"]["out_of_sample_validated"] is False
    # All the Codex-required fields are accessible
    cc = pp["candidate_contracts"]
    assert cc["legacy_selection"]["picker_variant"] == "legacy_first_expiry"
    assert cc["candidate_selection"]["picker_variant"] == "candidate_min_dte"
    assert cc["legacy_selection"]["front_expiry"] == "2026-05-03"
    assert cc["candidate_selection"]["front_expiry"] == "2026-05-17"
    assert cc["legacy_selection"]["strike"] == 100.0
    assert cc["candidate_selection"]["strike"] == 100.0
    assert cc["pickers_diverged"] is True
    assert cc["candidate_min_front_dte_days"] == 14


def test_picker_provenance_empty_when_absent_from_analysis(tmp_path: Path) -> None:
    """Backward-compatible: analyses produced before PR-AC commit 3 have
    no experimental_contract_selection field. The ledger must accept
    those records and store an empty dict for picker_provenance — never
    raise, never produce NULL JSON."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    # NO experimental_contract_selection key
    recommendation_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="atm_straddle",
        salt="no-experimental-block",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=recommendation_id)
    row = ledger.get(recommendation_id)
    assert row is not None
    assert row["picker_provenance_json"] == {}


def test_picker_provenance_handles_put_calendar_placeholder(tmp_path: Path) -> None:
    """put_calendar gets the explicit 'put_side_not_yet_supported'
    placeholder from commit 3. The ledger must persist that as-is,
    including candidate_contracts=None — never silently dropping it."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    analysis.metrics["experimental_contract_selection"] = {
        "structure": "put_calendar",
        "labels": {"experimental": True, "shadow_mode": True,
                   "not_execution_guidance": True,
                   "out_of_sample_validated": False},
        "note": "Experimental candidate contract selection.",
        "status": "put_side_not_yet_supported",
        "reason": "Historical backtest only queries calls.",
        "candidate_contracts": None,
    }
    recommendation_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="put_calendar",
        salt="put-placeholder-test",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=recommendation_id)
    row = ledger.get(recommendation_id)
    pp = row["picker_provenance_json"]
    assert pp["structure"] == "put_calendar"
    assert pp["status"] == "put_side_not_yet_supported"
    assert pp["candidate_contracts"] is None
    # The placeholder remains attributable — future analysis can
    # filter for "put_side_not_yet_supported" to see how often the
    # selector picked put_calendar without having put-side evidence.


def test_picker_provenance_field_location_contract(tmp_path: Path) -> None:
    """REGRESSION (Codex review of commit 4): build_record_from_analysis
    must read `experimental_contract_selection` from analysis.metrics —
    not from a top-level analysis attribute, not from selector_output,
    not from vol_snapshot. The live edge_engine assembles the block as
    a key inside the `metrics` dict it passes to EdgeSnapshot.

    This test plants the correct payload at metrics[...] AND sentinel
    decoy values at every plausible-wrong location, then asserts the
    captured picker_provenance is the metrics-level payload — not any
    of the decoys. If a future refactor moves the field out of
    `metrics`, this fails loudly rather than silently dropping picker
    provenance for every recorded paper trade."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    expected = _sample_experimental_contract_selection()
    analysis.metrics["experimental_contract_selection"] = expected
    # Decoys at every other plausible location
    analysis.selector_output["experimental_contract_selection"] = {"WRONG_LOCATION": "selector_output"}
    analysis.vol_snapshot["experimental_contract_selection"] = {"WRONG_LOCATION": "vol_snapshot"}
    # And as a top-level attribute on the namespace itself
    analysis.experimental_contract_selection = {"WRONG_LOCATION": "top_level"}

    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="field-location-contract",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)
    pp = ledger.get(rec_id)["picker_provenance_json"]
    # The metrics-level payload was captured
    assert pp["structure"] == "call_calendar"
    assert pp["candidate_contracts"]["pickers_diverged"] is True
    # None of the decoys leaked in
    pp_str = str(pp)
    assert "WRONG_LOCATION" not in pp_str
    assert "selector_output" not in pp_str or "selector_output" in str(expected)
    # (the second clause guards against false negatives if expected itself
    # mentions "selector_output" in some unrelated field text)


def test_picker_provenance_end_to_end_through_edge_snapshot(tmp_path: Path) -> None:
    """REGRESSION (Codex review of commit 4b): the field-location
    contract test exercises `build_record_from_analysis` directly with
    a SimpleNamespace. This test goes one layer further — constructs
    a real EdgeSnapshot exactly the way the live engine does
    (metrics dict carrying experimental_contract_selection), passes
    it through `record_recommendation`, and reads the ledger to
    confirm picker_provenance survived the full round trip. Catches
    adapter changes outside build_record_from_analysis."""
    from web.api.edge_engine import EdgeSnapshot

    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    expected = _sample_experimental_contract_selection()

    # Mirror the live engine's assembly: metrics dict with the
    # experimental block as a top-level key inside metrics.
    metrics = {
        "data_sources": {"options_source": "marketdata_app",
                         "price_rv_source": "yfinance"},
        "experimental_contract_selection": expected,
        # Other metrics fields the engine populates — included so the
        # shape matches reality even if our test doesn't read them
        "calendar_payoff": None,
        "structure_payoff": None,
    }
    edge_snapshot = EdgeSnapshot(
        symbol="AAPL",
        recommendation="Candidate",
        confidence_pct=72.0,
        setup_score=0.72,
        metrics=metrics,
        rationale=["selector rationale"],
        selector_output={
            "recommendation": "Candidate",
            "best_structure": "call_calendar",
            "earnings_date": "2026-05-01",
            "as_of": "2026-04-23",
            "primary_thesis": "thesis",
            "primary_risks": ["r"],
            "why_this_structure": ["w"],
            "why_not_others": {},
        },
        structure_scorecards=[
            {"structure": "call_calendar", "eligible": True,
             "composite_structure_score": 0.81},
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
        },
    )

    rec_id = record_recommendation(edge_snapshot, ledger=ledger)
    row = ledger.get(rec_id)
    assert row is not None
    pp = row["picker_provenance_json"]
    # Survived the full EdgeSnapshot → build_record → ledger round trip
    assert pp["structure"] == "call_calendar"
    assert pp["labels"]["shadow_mode"] is True
    assert pp["candidate_contracts"]["pickers_diverged"] is True
    assert pp["candidate_contracts"]["candidate_selection"]["picker_variant"] == "candidate_min_dte"


def test_picker_provenance_preserves_numeric_types_through_round_trip(tmp_path: Path) -> None:
    """REGRESSION (Codex review of commit 4b): _json uses `default=str`
    to never crash on dates/numpy/timestamps. That's operationally
    correct but can hide silent type drift — a numpy.float64 written as
    a JSON string would round-trip as a string, breaking any future
    SQL/JSON analytics that expects a number.

    Numeric fields persisted by `to_metadata_dict` (mid, bid, ask, iv,
    strike, spread_pct) are already coerced to native Python floats
    inside the picker module. This test confirms that contract holds
    end-to-end: write → read → numeric fields are still numbers."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    analysis.metrics["experimental_contract_selection"] = _sample_experimental_contract_selection()
    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="numeric-type-test",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)
    pp = ledger.get(rec_id)["picker_provenance_json"]

    legacy = pp["candidate_contracts"]["legacy_selection"]
    candidate = pp["candidate_contracts"]["candidate_selection"]

    # Strikes must be float, not string
    assert isinstance(legacy["strike"], (int, float))
    assert isinstance(candidate["strike"], (int, float))
    # DTE measurements must be int
    assert isinstance(legacy["front_dte_days"], int)
    assert isinstance(candidate["front_dte_days"], int)
    assert isinstance(legacy["back_minus_front_days"], int)
    # Leg quote fields must be numeric (or None when missing)
    for side_key in ("front_leg", "back_leg"):
        for field_key in ("mid", "bid", "ask", "iv", "spread_pct"):
            v_legacy = legacy[side_key][field_key]
            v_candidate = candidate[side_key][field_key]
            assert v_legacy is None or isinstance(v_legacy, (int, float)), (
                f"legacy.{side_key}.{field_key} drifted to non-numeric: {type(v_legacy).__name__}"
            )
            assert v_candidate is None or isinstance(v_candidate, (int, float)), (
                f"candidate.{side_key}.{field_key} drifted to non-numeric: {type(v_candidate).__name__}"
            )
    # Booleans stay booleans
    assert pp["candidate_contracts"]["pickers_diverged"] is True


def test_malformed_picker_provenance_json_yields_empty_dict_on_read(tmp_path: Path) -> None:
    """Codex hygiene check: a single row with malformed
    picker_provenance_json must NOT break the whole ledger read.
    _loads returns {} on JSONDecodeError; this confirms the contract
    holds end-to-end so one bad row never blocks the others."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")

    # Insert a valid record first
    analysis = _analysis()
    analysis.metrics["experimental_contract_selection"] = _sample_experimental_contract_selection()
    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="malformed-json-test",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)

    # Corrupt the column on this row
    ledger._conn.execute(  # noqa: SLF001
        "UPDATE recommendations SET picker_provenance_json = ? WHERE recommendation_id = ?",
        ("{not valid json", rec_id),
    )
    ledger._conn.commit()

    # Read must not raise; _loads returns {} for malformed JSON
    row = ledger.get(rec_id)
    assert row is not None
    assert row["picker_provenance_json"] == {}


def test_picker_provenance_column_added_by_migration(tmp_path: Path) -> None:
    """An existing ledger DB created BEFORE PR-AC commit 4 (no
    picker_provenance_json column) must gain the column on next open,
    not crash and not lose data."""
    db_path = tmp_path / "ledger.sqlite"
    # Create the old schema by hand — leave picker_provenance_json out
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE recommendations (
            recommendation_id TEXT PRIMARY KEY,
            created_at TEXT,
            symbol TEXT,
            schema_version INTEGER NOT NULL DEFAULT 1,
            engine_version TEXT NOT NULL DEFAULT 'event_vol_selector_v1'
        )"""
    )
    conn.execute(
        "INSERT INTO recommendations (recommendation_id, created_at, symbol) VALUES (?, ?, ?)",
        ("legacy_row_001", "2025-01-01T00:00:00+00:00", "AAPL"),
    )
    conn.commit()
    conn.close()

    # Open via the ledger — migration should add the missing column
    ledger = RecommendationLedger(ledger_path=db_path)
    cols = {row["name"] for row in ledger._conn.execute("PRAGMA table_info(recommendations)").fetchall()}  # noqa: SLF001
    assert "picker_provenance_json" in cols, "migration must add the new column"

    # Legacy row preserved (no data loss)
    row = ledger._conn.execute(  # noqa: SLF001
        "SELECT recommendation_id, symbol FROM recommendations WHERE recommendation_id = ?",
        ("legacy_row_001",),
    ).fetchone()
    assert row is not None
    assert row["symbol"] == "AAPL"

    # And new records can write picker_provenance_json without error
    analysis = _analysis()
    analysis.metrics["experimental_contract_selection"] = _sample_experimental_contract_selection()
    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="post-migration-write",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)
    assert ledger.get(rec_id)["picker_provenance_json"]["structure"] == "call_calendar"


# ── PR-AD commit 3: candidate_shadow_outcome + sample_provenance round-trip ──


def _sample_candidate_shadow_outcome() -> dict:
    """Realistic candidate_shadow_outcome block from PR-AD commit 1 +
    1b/1c hardening — happy-path resolved candidate PnL."""
    return {
        "status": "ok",
        "labels": {
            "research_mid": True,
            "shadow_only": True,
            "not_execution_grade": True,
        },
        "side": "call",
        "strike": 100.0,
        "front_expiry": "2024-05-17",
        "back_expiry": "2024-06-21",
        "entry_front_mid": 1.0,
        "entry_back_mid": 2.0,
        "exit_front_mid": 0.4,
        "exit_back_mid": 1.7,
        "entry_front_iv": 0.30,
        "entry_back_iv": 0.30,
        "exit_front_iv": 0.20,
        "exit_back_iv": 0.25,
        "entry_debit_mid": 1.0,
        "exit_value_mid": 1.3,
        "mid_pnl": 0.3,
        "mid_realized_return_pct": 30.0,
        "iv_change_front": -0.10,
        "iv_change_back": -0.05,
    }


def test_candidate_shadow_outcome_persists_on_historical_replay_record(tmp_path: Path) -> None:
    """Codex prerequisite: the ledger must persist the resolved
    candidate PnL block. On historical replay records the block
    carries the candidate outcome and the sample_provenance is
    historical_replay_in_sample_or_research."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    analysis.candidate_shadow_outcome = _sample_candidate_shadow_outcome()
    analysis.sample_provenance = "historical_replay_in_sample_or_research"

    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="candidate-shadow-roundtrip",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)
    row = ledger.get(rec_id)
    assert row is not None

    cso = row["candidate_shadow_outcome_json"]
    assert cso["status"] == "ok"
    assert cso["labels"]["research_mid"] is True
    assert cso["labels"]["shadow_only"] is True
    assert cso["labels"]["not_execution_grade"] is True
    assert cso["mid_realized_return_pct"] == 30.0
    assert cso["entry_debit_mid"] == 1.0
    assert cso["mid_pnl"] == 0.3
    # Denormalized provenance column round-trips too
    assert row["sample_provenance"] == "historical_replay_in_sample_or_research"


def test_candidate_shadow_outcome_empty_when_absent_from_analysis(tmp_path: Path) -> None:
    """Live API recommendations produced BEFORE an exit resolver runs
    have no candidate_shadow_outcome. Ledger persists empty dict +
    None provenance, never raises."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="atm_straddle",
        salt="no-candidate-outcome",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)
    row = ledger.get(rec_id)
    assert row["candidate_shadow_outcome_json"] == {}
    assert row["sample_provenance"] is None


def test_sample_provenance_invalid_value_normalizes_to_unknown(tmp_path: Path) -> None:
    """REGRESSION (Codex round-4 P2): invalid / typo / non-canonical
    sample_provenance values get normalized to "unknown" — NOT None.
    "unknown" is observable in downstream aggregator
    provenance_counts (it surfaces under the unknown bucket), so
    operators can see how many rows were rejected for invalid tags.
    None would silently disappear as a missing key, hurting
    observability. is_promotion_eligible still rejects "unknown"
    because it isn't in PROMOTION_ELIGIBLE_PROVENANCES."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    analysis.sample_provenance = "forward_post_pr_ad"  # non-canonical typo

    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="invalid-provenance",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)
    persisted = ledger.get(rec_id)["sample_provenance"]
    assert persisted == "unknown", (
        f"Invalid provenance should normalize to observable 'unknown', "
        f"got {persisted!r}"
    )

    # When the analysis has NO sample_provenance at all, the column
    # stays None (different signal — "caller didn't tag" rather than
    # "caller tagged with an invalid value").
    analysis2 = _analysis()
    rec_id2 = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="missing-provenance",
    )
    record_recommendation(analysis2, ledger=ledger, recommendation_id=rec_id2)
    assert ledger.get(rec_id2)["sample_provenance"] is None


def test_top_level_provenance_is_canonical_nested_does_not_override(tmp_path: Path) -> None:
    """REGRESSION (Codex round-4 P1#2): if a future shape change ever
    puts a `sample_provenance` field INSIDE the candidate_shadow_outcome
    dict, the ledger must treat the TOP-LEVEL record.sample_provenance
    as canonical. The nested duplicate is preserved verbatim in the
    candidate_shadow_outcome_json blob (for audit) but does NOT
    override the persisted top-level column.

    This policy means: if the two ever disagree, the analyst can see
    BOTH in the row (top-level column + nested JSON field), but
    is_promotion_eligible reads only the top-level column."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    analysis = _analysis()
    outcome = _sample_candidate_shadow_outcome()
    # Inject a divergent sample_provenance nested INSIDE the outcome.
    # This isn't a shape we ship today — but if a future refactor
    # adds it, the policy must be unambiguous.
    outcome["sample_provenance"] = "forward_post_pr_ad"  # bad / divergent
    analysis.candidate_shadow_outcome = outcome
    analysis.sample_provenance = "historical_replay_in_sample_or_research"  # canonical

    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="provenance-mismatch",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)
    row = ledger.get(rec_id)

    # Top-level column wins for filtering / promotion eligibility
    assert row["sample_provenance"] == "historical_replay_in_sample_or_research"
    # Nested duplicate is preserved verbatim in the JSON blob for audit
    # (the storage layer doesn't mutate the outcome shape)
    assert row["candidate_shadow_outcome_json"]["sample_provenance"] == "forward_post_pr_ad"


def test_candidate_shadow_outcome_column_added_by_migration(tmp_path: Path) -> None:
    """Pre-PR-AD ledger DB lacks the new columns. Opening it must add
    both candidate_shadow_outcome_json and sample_provenance without
    losing data."""
    db_path = tmp_path / "ledger.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE recommendations (
            recommendation_id TEXT PRIMARY KEY,
            created_at TEXT,
            symbol TEXT,
            picker_provenance_json TEXT,
            schema_version INTEGER NOT NULL DEFAULT 1,
            engine_version TEXT NOT NULL DEFAULT 'event_vol_selector_v1'
        )"""
    )
    conn.execute(
        "INSERT INTO recommendations (recommendation_id, created_at, symbol, picker_provenance_json)"
        " VALUES (?, ?, ?, ?)",
        ("legacy_pr_ac_row", "2026-01-01T00:00:00+00:00", "AAPL", "{}"),
    )
    conn.commit()
    conn.close()

    ledger = RecommendationLedger(ledger_path=db_path)
    cols = {row["name"] for row in ledger._conn.execute(  # noqa: SLF001
        "PRAGMA table_info(recommendations)"
    ).fetchall()}
    assert "candidate_shadow_outcome_json" in cols
    assert "sample_provenance" in cols

    # Legacy row preserved with no data loss
    row = ledger._conn.execute(  # noqa: SLF001
        "SELECT recommendation_id, symbol FROM recommendations WHERE recommendation_id = ?",
        ("legacy_pr_ac_row",),
    ).fetchone()
    assert row is not None
    assert row["symbol"] == "AAPL"

    # New records write the new fields cleanly
    analysis = _analysis()
    analysis.candidate_shadow_outcome = _sample_candidate_shadow_outcome()
    analysis.sample_provenance = "historical_replay_in_sample_or_research"
    rec_id = make_recommendation_id(
        symbol="AAPL", as_of_date="2026-04-23",
        earnings_date="2026-05-01", selected_structure="call_calendar",
        salt="post-pr-ad-migration",
    )
    record_recommendation(analysis, ledger=ledger, recommendation_id=rec_id)
    persisted = ledger.get(rec_id)
    assert persisted["candidate_shadow_outcome_json"]["status"] == "ok"
    assert persisted["sample_provenance"] == "historical_replay_in_sample_or_research"


# ── PR-AE commit 1: live-exit-resolver schema + ledger helpers ────────────────


def _forward_record(
    *,
    recommendation_id: str = "rec_fw_test",
    earnings_date: str = "2026-05-01",
    as_of_date: str = "2026-04-23",
    selected_structure: str = "call_calendar",
    sample_provenance: str = "forward_post_freeze",
    candidate_shadow_outcome: Optional[dict] = None,
):
    """Build a forward-post-freeze ledger record shaped the way the
    PR-AE live API path will produce one: candidate_shadow_outcome
    starts empty (the resolver will fill it later), sample_provenance
    is tagged, picker_provenance carries a valid candidate selection."""
    from services.recommendation_ledger import build_record_from_analysis

    analysis = SimpleNamespace(
        symbol="AAPL",
        recommendation="Candidate",
        setup_score=0.72,
        metrics={
            "data_sources": {
                "options_source": "marketdata_app",
                "price_rv_source": "yfinance",
            },
            "experimental_contract_selection": {
                "structure": "call_calendar",
                "labels": {"experimental": True, "shadow_mode": True,
                           "not_execution_guidance": True,
                           "out_of_sample_validated": False},
                "candidate_contracts": {
                    "candidate_selection": {
                        "side": "call",
                        "strike": 200.0,
                        "front_expiry": "2026-05-08",
                        "back_expiry": "2026-05-22",
                    },
                    "pickers_diverged": True,
                },
            },
        },
        rationale=["r"],
        selector_output={
            "recommendation": "Candidate",
            "best_structure": selected_structure,
            "earnings_date": earnings_date,
            "primary_thesis": "thesis",
            "primary_risks": [],
            "why_this_structure": [],
            "why_not_others": {},
        },
        structure_scorecards=[
            {"structure": selected_structure, "eligible": True,
             "composite_structure_score": 0.81},
        ],
        vol_snapshot={
            "symbol": "AAPL",
            "as_of_date": as_of_date,
            "earnings_date": earnings_date,
            "earnings_source_primary": "alpha_vantage",
            "earnings_source_confidence": 0.82,
            "earnings_source_stale": False,
            "option_source": "marketdata_app",
            "underlying_source": "yfinance",
            "data_quality_score": 0.77,
        },
        sample_provenance=sample_provenance,
        candidate_shadow_outcome=candidate_shadow_outcome or {},
    )
    return build_record_from_analysis(
        analysis,
        recommendation_id=recommendation_id,
        quote_payload={
            "quote_source": "yfinance",
            "quote_timestamp": f"{as_of_date}T12:00:00+00:00",
            "quote_quality": "paper_research_mid_not_execution_grade",
            "bid_ask_mid": {"legs": {"call": {"mid": 4.1}}},
            "surface_quality": {"status": "valid_evidence", "warning_flags": []},
        },
    )


def test_pr_ae_schema_adds_resolver_attempts_column(tmp_path: Path) -> None:
    """A fresh ledger DB created via the current ledger code MUST contain
    the candidate_exit_resolver_attempts column from the start."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    cols = {
        row["name"]
        for row in ledger._conn.execute(  # noqa: SLF001
            "PRAGMA table_info(recommendations)"
        ).fetchall()
    }
    assert "candidate_exit_resolver_attempts" in cols


def test_pr_ae_existing_db_migrates_resolver_attempts_to_zero(
    tmp_path: Path,
) -> None:
    """An existing pre-PR-AE ledger DB (no resolver-attempts column)
    must gain the column on next open, with default 0 for every
    existing row. No data loss; legacy rows readable."""
    db_path = tmp_path / "ledger.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE recommendations (
            recommendation_id TEXT PRIMARY KEY,
            created_at TEXT,
            symbol TEXT,
            sample_provenance TEXT,
            schema_version INTEGER NOT NULL DEFAULT 1,
            engine_version TEXT NOT NULL DEFAULT 'event_vol_selector_v1'
        )"""
    )
    conn.execute(
        "INSERT INTO recommendations (recommendation_id, created_at, symbol, sample_provenance)"
        " VALUES (?, ?, ?, ?)",
        ("legacy_pr_ad_row", "2026-05-01T00:00:00+00:00", "AAPL", "forward_post_freeze"),
    )
    conn.commit()
    conn.close()

    ledger = RecommendationLedger(ledger_path=db_path)
    cols = {
        row["name"]
        for row in ledger._conn.execute(  # noqa: SLF001
            "PRAGMA table_info(recommendations)"
        ).fetchall()
    }
    assert "candidate_exit_resolver_attempts" in cols

    legacy = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("legacy_pr_ad_row",),
    ).fetchone()
    # NOT NULL DEFAULT 0 applies to existing rows on ADD COLUMN
    assert legacy is not None
    assert int(legacy[0]) == 0


def test_pr_ae_record_initializes_resolver_attempts_to_zero(
    tmp_path: Path,
) -> None:
    """Every newly recorded row defaults to 0 attempts. The record()
    INSERT does not need to mention the column — the DEFAULT 0 applies."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_fresh"))
    row = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_fresh",),
    ).fetchone()
    assert int(row[0]) == 0


def test_pr_ae_increment_resolver_attempts_is_atomic_and_returns_new_value(
    tmp_path: Path,
) -> None:
    """Counter goes 0 → 1 → 2; returned value reflects the new state."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_inc"))
    assert ledger.increment_resolver_attempts("rec_inc") == 1
    assert ledger.increment_resolver_attempts("rec_inc") == 2
    row = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_inc",),
    ).fetchone()
    assert int(row[0]) == 2


def test_pr_ae_increment_resolver_attempts_raises_for_unknown_id(
    tmp_path: Path,
) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    import pytest
    with pytest.raises(KeyError):
        ledger.increment_resolver_attempts("rec_does_not_exist")


def test_pr_ae_get_with_latest_resolution_returns_v1_when_no_revision(
    tmp_path: Path,
) -> None:
    """First write only — get_with_latest_resolution returns v1 verbatim.
    Note: the PR-K revisions table always records a v1 entry on first
    record, but the latest_payload's candidate_shadow_outcome equals
    v1's (empty dict by construction), so the merged view == v1."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_no_resolve"))
    base = ledger.get("rec_no_resolve")
    merged = ledger.get_with_latest_resolution("rec_no_resolve")
    assert merged is not None
    # v1's candidate_shadow_outcome was empty {}, and the latest
    # revision (v1 snapshot) carries the same empty dict — so the
    # merged view's candidate_shadow_outcome_json is also {}.
    assert merged["candidate_shadow_outcome_json"] == {}
    # Every other field matches the immutable v1
    assert merged["sample_provenance"] == base["sample_provenance"]
    assert merged["selected_structure"] == base["selected_structure"]


def test_pr_ae_get_with_latest_resolution_overlays_latest_outcome(
    tmp_path: Path,
) -> None:
    """After a resolver write, get_with_latest_resolution surfaces the
    NEW candidate_shadow_outcome — get() still returns the immutable v1."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_overlay"))

    resolved = _sample_candidate_shadow_outcome()
    status = ledger.record_resolution_payload(
        recommendation_id="rec_overlay",
        candidate_shadow_outcome=resolved,
    )
    assert status == "revision"

    # get() returns the immutable v1 — candidate_shadow_outcome was empty
    v1 = ledger.get("rec_overlay")
    assert v1["candidate_shadow_outcome_json"] == {}

    # get_with_latest_resolution overlays the resolved outcome
    merged = ledger.get_with_latest_resolution("rec_overlay")
    assert merged is not None
    assert merged["candidate_shadow_outcome_json"]["status"] == "ok"
    assert merged["candidate_shadow_outcome_json"]["mid_realized_return_pct"] == 30.0
    # Sample provenance stays at v1 (immutable post-tagging)
    assert merged["sample_provenance"] == "forward_post_freeze"


def test_pr_ae_get_with_latest_resolution_returns_none_for_unknown_id(
    tmp_path: Path,
) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    assert ledger.get_with_latest_resolution("rec_does_not_exist") is None


def test_pr_ae_record_resolution_payload_writes_revision(tmp_path: Path) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_resolve"))

    # First resolver write differs from v1 (which had empty
    # candidate_shadow_outcome) — content_hash differs, new revision.
    status = ledger.record_resolution_payload(
        recommendation_id="rec_resolve",
        candidate_shadow_outcome=_sample_candidate_shadow_outcome(),
    )
    assert status == "revision"

    revisions = ledger.get_revisions("rec_resolve")
    # v1 (auto-recorded on first record()) + resolver revision = 2
    assert len(revisions) == 2
    # The latest revision carries the resolved candidate outcome
    assert revisions[-1]["record"]["candidate_shadow_outcome"]["status"] == "ok"


def test_pr_ae_record_resolution_payload_duplicate_outcome_is_idempotent(
    tmp_path: Path,
) -> None:
    """Codex idempotency contract: a second resolver run that lands the
    SAME candidate_shadow_outcome must return "duplicate" and not add a
    second revision row. The PR-K UNIQUE(rec_id, content_hash) constraint
    enforces this without any resolver-side bookkeeping."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_dup_resolve"))

    outcome = _sample_candidate_shadow_outcome()
    assert (
        ledger.record_resolution_payload(
            recommendation_id="rec_dup_resolve",
            candidate_shadow_outcome=outcome,
        )
        == "revision"
    )
    # Same outcome again → identical content_hash → duplicate
    assert (
        ledger.record_resolution_payload(
            recommendation_id="rec_dup_resolve",
            candidate_shadow_outcome=outcome,
        )
        == "duplicate"
    )
    # Still exactly 2 revision rows (v1 snapshot + first resolver write)
    assert len(ledger.get_revisions("rec_dup_resolve")) == 2


def test_pr_ae_record_resolution_payload_data_drift_yields_new_revision(
    tmp_path: Path,
) -> None:
    """Same row, different resolved outcome (e.g. T9 reprocessed a quote)
    → different content_hash → new revision. Audit trail must show the
    state changed because upstream data changed."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_drift"))

    outcome_a = _sample_candidate_shadow_outcome()
    outcome_b = dict(outcome_a)
    outcome_b["mid_realized_return_pct"] = 35.0  # drifted
    outcome_b["mid_pnl"] = 0.35

    assert ledger.record_resolution_payload(
        recommendation_id="rec_drift",
        candidate_shadow_outcome=outcome_a,
    ) == "revision"
    assert ledger.record_resolution_payload(
        recommendation_id="rec_drift",
        candidate_shadow_outcome=outcome_b,
    ) == "revision"

    # v1 + resolver_v1 + resolver_v2 = 3 revisions
    assert len(ledger.get_revisions("rec_drift")) == 3


def test_pr_ae_record_resolution_payload_raises_for_unknown_id(
    tmp_path: Path,
) -> None:
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    import pytest
    with pytest.raises(KeyError):
        ledger.record_resolution_payload(
            recommendation_id="rec_does_not_exist",
            candidate_shadow_outcome=_sample_candidate_shadow_outcome(),
        )


def test_pr_ae_record_resolution_payload_preserves_v1_picker_provenance(
    tmp_path: Path,
) -> None:
    """The reconstructed RecommendationRecord must carry v1's
    picker_provenance verbatim — the resolver does not regenerate it.
    Without this, the revision payload would lose the candidate
    contract identity and downstream merged views would be incomplete."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_preserves_pp"))

    ledger.record_resolution_payload(
        recommendation_id="rec_preserves_pp",
        candidate_shadow_outcome=_sample_candidate_shadow_outcome(),
    )
    revisions = ledger.get_revisions("rec_preserves_pp")
    latest_payload = revisions[-1]["record"]
    pp = latest_payload["picker_provenance"]
    # Same shape as v1 — candidate selection survived
    assert pp["structure"] == "call_calendar"
    assert pp["candidate_contracts"]["candidate_selection"]["front_expiry"] == "2026-05-08"
    assert pp["candidate_contracts"]["candidate_selection"]["strike"] == 200.0


def test_pr_ae_list_pending_filters_by_sample_provenance(tmp_path: Path) -> None:
    """Only forward_post_freeze rows are eligible by default. Historical
    replay rows are explicitly excluded — re-running the resolver on a
    backtest row would corrupt the audit trail."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    # Forward row — eligible
    ledger.record(_forward_record(
        recommendation_id="rec_fw",
        sample_provenance="forward_post_freeze",
        earnings_date="2026-04-01",  # well in the past
    ))
    # Historical replay row — NOT eligible
    ledger.record(_forward_record(
        recommendation_id="rec_hist",
        sample_provenance="historical_replay_in_sample_or_research",
        earnings_date="2026-04-01",
    ))
    # Unknown-provenance row — NOT eligible
    ledger.record(_forward_record(
        recommendation_id="rec_unknown",
        sample_provenance="unknown",
        earnings_date="2026-04-01",
    ))

    eligible = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 27),  # well past all earnings_dates
    )
    ids = {row["recommendation_id"] for row in eligible}
    assert ids == {"rec_fw"}


def test_pr_ae_list_pending_filters_by_selected_structure(tmp_path: Path) -> None:
    """Only call_calendar rows by default. Put-calendar deferred."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(
        recommendation_id="rec_call",
        selected_structure="call_calendar",
        earnings_date="2026-04-01",
    ))
    ledger.record(_forward_record(
        recommendation_id="rec_put",
        selected_structure="put_calendar",
        earnings_date="2026-04-01",
    ))
    ledger.record(_forward_record(
        recommendation_id="rec_straddle",
        selected_structure="atm_straddle",
        earnings_date="2026-04-01",
    ))

    eligible = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 27),
    )
    ids = {row["recommendation_id"] for row in eligible}
    assert ids == {"rec_call"}


def test_pr_ae_list_pending_filters_by_earnings_date_cutoff(tmp_path: Path) -> None:
    """A row whose earnings_date is too recent (within min_days_after_event
    of now) is NOT eligible yet. The post-event chain may not exist or may
    still be settling — better to wait than to fail.

    Default ``min_days_after_event=3`` covers the common edge case of
    Friday earnings: by Monday (3 calendar days later) the post-event
    chain should have settled.
    """
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    now = date(2026, 5, 27)
    # 5 days ago — eligible
    ledger.record(_forward_record(
        recommendation_id="rec_old",
        earnings_date="2026-05-22",
    ))
    # Same day as 'now' — NOT eligible (still inside the window)
    ledger.record(_forward_record(
        recommendation_id="rec_today",
        earnings_date="2026-05-27",
    ))
    # 1 day ago — NOT eligible (default min_days_after_event=3)
    ledger.record(_forward_record(
        recommendation_id="rec_yesterday",
        earnings_date="2026-05-26",
    ))
    # Future earnings — NOT eligible (defensive — should not happen
    # since live forward rows are written before the event, but the
    # filter guards against weird states)
    ledger.record(_forward_record(
        recommendation_id="rec_future",
        earnings_date="2026-06-15",
    ))

    eligible = ledger.list_pending_candidate_exit_resolutions(now=now)
    ids = {row["recommendation_id"] for row in eligible}
    assert ids == {"rec_old"}


def test_pr_ae_list_pending_default_min_days_is_three_calendar_days(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex C1 P2 audit): the default cutoff is 3 calendar
    days, not 2. Bumped from 2 to 3 so a Friday-earnings row is
    eligible no earlier than the following Monday — the day the
    post-event chain typically settles. The resolver layer then applies
    BDay-aware chain-date selection within the lookahead window.

    Boundary test: earnings_date == now - 3 calendar days is the
    smallest eligible gap; earnings_date == now - 2 calendar days is
    NOT eligible.
    """
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    now = date(2026, 5, 27)
    # Exactly 3 days back — eligible at the boundary
    ledger.record(_forward_record(
        recommendation_id="rec_at_boundary",
        earnings_date="2026-05-24",
    ))
    # 2 days back — NOT eligible (one day shy of cutoff)
    ledger.record(_forward_record(
        recommendation_id="rec_two_days",
        earnings_date="2026-05-25",
    ))

    eligible = ledger.list_pending_candidate_exit_resolutions(now=now)
    ids = {row["recommendation_id"] for row in eligible}
    assert ids == {"rec_at_boundary"}


def test_pr_ae_list_pending_friday_earnings_eligible_by_monday(tmp_path: Path) -> None:
    """The trading-day-aware rationale for default=3: Friday earnings
    on 2026-05-22 become eligible by Monday 2026-05-25 (3 calendar days
    later — covers the weekend). With the old default of 2, eligibility
    would land on Sunday with no post-event chain yet.
    """
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(
        recommendation_id="rec_fri_earnings",
        earnings_date="2026-05-22",  # Friday
    ))
    # Sunday: NOT yet eligible (would be at min_days=2; not at default=3)
    eligible_sun = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 24),
    )
    assert all(row["recommendation_id"] != "rec_fri_earnings" for row in eligible_sun)
    # Monday: eligible
    eligible_mon = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 25),
    )
    assert any(row["recommendation_id"] == "rec_fri_earnings" for row in eligible_mon)


def test_pr_ae_list_pending_filters_by_max_attempts(tmp_path: Path) -> None:
    """A row whose counter has reached max_attempts is permanently
    excluded. Tested with max_attempts=3 and a counter incremented to 3."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(
        recommendation_id="rec_maxed",
        earnings_date="2026-04-01",
    ))
    # Eligible row at 0 attempts
    ledger.record(_forward_record(
        recommendation_id="rec_fresh",
        earnings_date="2026-04-01",
    ))
    # Burn three attempts on rec_maxed
    ledger.increment_resolver_attempts("rec_maxed")
    ledger.increment_resolver_attempts("rec_maxed")
    ledger.increment_resolver_attempts("rec_maxed")

    eligible = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 27),
        max_attempts=3,
    )
    ids = {row["recommendation_id"] for row in eligible}
    assert ids == {"rec_fresh"}


def test_pr_ae_list_pending_excludes_rows_resolved_ok(tmp_path: Path) -> None:
    """A row whose latest revision has candidate_shadow_outcome.status == 'ok'
    is terminal — never re-resolved. (If a future PR-AF wants to
    intentionally re-resolve on data drift, that's a separate code path
    with an explicit override.)"""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(
        recommendation_id="rec_pending",
        earnings_date="2026-04-01",
    ))
    ledger.record(_forward_record(
        recommendation_id="rec_resolved",
        earnings_date="2026-04-01",
    ))
    # Resolve one with ok
    ledger.record_resolution_payload(
        recommendation_id="rec_resolved",
        candidate_shadow_outcome=_sample_candidate_shadow_outcome(),
    )

    eligible = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 27),
    )
    ids = {row["recommendation_id"] for row in eligible}
    assert ids == {"rec_pending"}


def test_pr_ae_list_pending_excludes_permanently_failed_rows(tmp_path: Path) -> None:
    """Any latest-revision status starting with 'permanently_failed:' is
    terminal — excluded forever from re-resolution attempts."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(
        recommendation_id="rec_active",
        earnings_date="2026-04-01",
    ))
    ledger.record(_forward_record(
        recommendation_id="rec_failed",
        earnings_date="2026-04-01",
    ))
    # Mark one with permanently_failed status
    failed_outcome = {
        "status": "permanently_failed:no_post_event_chain",
        "labels": {"research_mid": True, "shadow_only": True,
                   "not_execution_grade": True},
    }
    ledger.record_resolution_payload(
        recommendation_id="rec_failed",
        candidate_shadow_outcome=failed_outcome,
    )

    eligible = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 27),
    )
    ids = {row["recommendation_id"] for row in eligible}
    assert ids == {"rec_active"}


def test_pr_ae_list_pending_includes_retrying_and_awaiting_rows(tmp_path: Path) -> None:
    """retrying and awaiting_chain_data are NOT terminal — those rows
    must remain eligible so the next resolver tick can try again."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(
        recommendation_id="rec_retrying",
        earnings_date="2026-04-01",
    ))
    ledger.record(_forward_record(
        recommendation_id="rec_awaiting",
        earnings_date="2026-04-01",
    ))
    ledger.record_resolution_payload(
        recommendation_id="rec_retrying",
        candidate_shadow_outcome={
            "status": "retrying",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
        },
    )
    ledger.record_resolution_payload(
        recommendation_id="rec_awaiting",
        candidate_shadow_outcome={
            "status": "awaiting_chain_data",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
        },
    )

    eligible = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 27),
    )
    ids = {row["recommendation_id"] for row in eligible}
    assert ids == {"rec_retrying", "rec_awaiting"}


def test_pr_ae_list_pending_orders_by_earnings_date_ascending(tmp_path: Path) -> None:
    """Older earnings events get resolved first so the queue drains
    chronologically. Operators reading the JSONL telemetry see a
    natural progression."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(
        recommendation_id="rec_mid",
        earnings_date="2026-04-15",
    ))
    ledger.record(_forward_record(
        recommendation_id="rec_old",
        earnings_date="2026-03-15",
    ))
    ledger.record(_forward_record(
        recommendation_id="rec_new",
        earnings_date="2026-05-15",
    ))

    eligible = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 27),
    )
    ids = [row["recommendation_id"] for row in eligible]
    assert ids == ["rec_old", "rec_mid", "rec_new"]


def test_pr_ae_list_pending_returns_merged_view_not_v1(tmp_path: Path) -> None:
    """The returned dicts must reflect the LATEST resolver state
    (retrying / awaiting_chain_data), not v1's empty dict. Otherwise
    the resolver would re-examine "fresh" rows that have actually
    been touched."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(
        recommendation_id="rec_partial",
        earnings_date="2026-04-01",
    ))
    awaiting_outcome = {
        "status": "awaiting_chain_data",
        "labels": {"research_mid": True, "shadow_only": True,
                   "not_execution_grade": True},
    }
    ledger.record_resolution_payload(
        recommendation_id="rec_partial",
        candidate_shadow_outcome=awaiting_outcome,
    )

    eligible = ledger.list_pending_candidate_exit_resolutions(
        now=date(2026, 5, 27),
    )
    assert len(eligible) == 1
    # Merged view surfaces the resolver-applied status, NOT v1's empty dict
    assert eligible[0]["candidate_shadow_outcome_json"]["status"] == "awaiting_chain_data"


def test_pr_ae_c5b_revisions_are_overlay_not_cumulative_merge(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex C5 audit P3): record_resolution_and_attempt
    rebuilds the revision payload from the IMMUTABLE v1 row and
    overlays ONLY candidate_shadow_outcome. It does NOT cumulatively
    merge intermediate revisions' changes to other fields.

    This is the intentional PR-AE scope. The test guards against a
    future refactor that silently changes this to a cumulative
    merge — which would entangle the resolver with arbitrary other
    enrichment paths and break the "v1 is the canonical source for
    every non-candidate field" invariant.

    Scenario:
      1. Record a forward row with picker_provenance == A.
      2. Plant a synthetic intermediate revision that ALSO changes
         picker_provenance to A_PRIME (simulating some future
         enrichment path's revision).
      3. Run the resolver write (record_resolution_and_attempt).
      4. The new resolver revision must carry picker_provenance ==
         A (from v1), NOT A_PRIME (from the intermediate revision).
         This proves the overlay is from v1, not cumulative.

    If a future PR adds cumulative-merge semantics, it must do so in
    a separately-named helper and leave this contract intact for
    PR-AE.
    """
    from services.recommendation_ledger import build_record_from_analysis

    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")

    # Step 1: record v1 with picker_provenance carrying "A" marker.
    # _forward_record's signature is fixed for the common case; this
    # test needs to override picker_provenance, so we go through
    # build_record_from_analysis directly.
    pp_a = _sample_experimental_contract_selection()
    pp_a["candidate_contracts"]["pickers_diverged"] = True  # v1 marker

    analysis = SimpleNamespace(
        symbol="AAPL",
        recommendation="Candidate",
        setup_score=0.72,
        metrics={
            "data_sources": {"options_source": "marketdata_app",
                             "price_rv_source": "yfinance"},
            "experimental_contract_selection": pp_a,
        },
        rationale=[],
        selector_output={
            "recommendation": "Candidate",
            "best_structure": "call_calendar",
            "earnings_date": "2026-05-01",
            "primary_thesis": "thesis",
            "primary_risks": [],
            "why_this_structure": [],
            "why_not_others": {},
        },
        structure_scorecards=[
            {"structure": "call_calendar", "eligible": True,
             "composite_structure_score": 0.81},
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
        },
        sample_provenance="forward_post_freeze",
        candidate_shadow_outcome={},
    )
    rec = build_record_from_analysis(
        analysis,
        recommendation_id="rec_overlay_contract",
        quote_payload={
            "quote_source": "yfinance",
            "quote_timestamp": "2026-04-23T12:00:00+00:00",
            "quote_quality": "paper_research_mid_not_execution_grade",
            "bid_ask_mid": {"legs": {"call": {"mid": 4.1}}},
            "surface_quality": {"status": "valid_evidence", "warning_flags": []},
        },
    )
    ledger.record(rec)

    # Step 2: plant an intermediate revision that changes
    # picker_provenance to "A_PRIME" (simulating some future
    # enrichment path that we have not yet shipped — e.g. a
    # post-recommendation picker-provenance correction PR).
    import json
    pp_a_prime = _sample_experimental_contract_selection()
    pp_a_prime["candidate_contracts"]["pickers_diverged"] = False  # A_PRIME marker
    enriched_payload = {
        "recommendation_id": "rec_overlay_contract",
        "picker_provenance": pp_a_prime,
        "candidate_shadow_outcome": {},
        "sample_provenance": "forward_post_freeze",
    }
    ledger._conn.execute(  # noqa: SLF001
        """
        INSERT INTO recommendation_revisions
            (recommendation_id, revised_at, content_hash, record_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            "rec_overlay_contract",
            "2026-05-26T00:00:00+00:00",
            "synthetic_intermediate_hash",
            json.dumps(enriched_payload, sort_keys=True),
        ),
    )
    ledger._conn.commit()

    # Step 3: run the resolver write — overlays only the outcome.
    ledger.record_resolution_and_attempt(
        recommendation_id="rec_overlay_contract",
        candidate_shadow_outcome=_sample_candidate_shadow_outcome(),
        increment_attempts=False,
    )

    # Step 4: the LATEST revision (= the one record_resolution_and_attempt
    # just wrote) must carry v1's picker_provenance with pickers_diverged
    # == True, NOT the intermediate revision's pickers_diverged == False.
    revisions = ledger.get_revisions("rec_overlay_contract")
    latest_payload = revisions[-1]["record"]
    assert (
        latest_payload["picker_provenance"]["candidate_contracts"]["pickers_diverged"]
        is True
    ), (
        "record_resolution_and_attempt MUST overlay onto v1's "
        "picker_provenance (pickers_diverged=True), NOT the "
        "intermediate revision's (pickers_diverged=False). If this "
        "fails, the resolver has silently been changed to do a "
        "cumulative merge — which would entangle PR-AE with every "
        "other future enrichment path. Move the cumulative-merge "
        "behavior into a separately-named helper and restore the "
        "v1-overlay contract here."
    )


def test_pr_ae_composite_index_exists(tmp_path: Path) -> None:
    """The (sample_provenance, earnings_date) composite index must exist
    so the daily eligibility query doesn't full-scan a growing
    recommendations table."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    indexes = {
        row[1]
        for row in ledger._conn.execute(  # noqa: SLF001
            "SELECT type, name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_recommendations_provenance_earnings" in indexes


# ── PR-AE C1b: atomic combined helper (Codex P1 audit) ────────────────────────


def test_pr_ae_atomic_helper_increments_counter_only_when_requested(
    tmp_path: Path,
) -> None:
    """record_resolution_and_attempt(..., increment_attempts=True)
    bumps the counter by exactly one. With increment_attempts=False,
    the counter stays put. Both paths write the same kind of revision."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_inc_path"))
    ledger.record(_forward_record(recommendation_id="rec_no_inc_path"))

    # Path 1: increment_attempts=True (terminal-failure outcome)
    ledger.record_resolution_and_attempt(
        recommendation_id="rec_inc_path",
        candidate_shadow_outcome={"status": "retrying",
                                  "labels": {"research_mid": True,
                                             "shadow_only": True,
                                             "not_execution_grade": True}},
        increment_attempts=True,
    )
    counter_inc = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_inc_path",),
    ).fetchone()[0]
    assert int(counter_inc) == 1

    # Path 2: increment_attempts=False (ok or awaiting_chain_data outcome)
    ledger.record_resolution_and_attempt(
        recommendation_id="rec_no_inc_path",
        candidate_shadow_outcome={"status": "awaiting_chain_data",
                                  "labels": {"research_mid": True,
                                             "shadow_only": True,
                                             "not_execution_grade": True}},
        increment_attempts=False,
    )
    counter_no_inc = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_no_inc_path",),
    ).fetchone()[0]
    assert int(counter_no_inc) == 0


def test_pr_ae_record_resolution_payload_delegates_with_no_increment(
    tmp_path: Path,
) -> None:
    """record_resolution_payload is the convenience wrapper for the
    no-counter-increment path. Calling it must NOT bump the counter
    even when the outcome looks like a terminal failure — the caller
    has to use the atomic combined helper for that."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_wrapper"))

    # Even with a terminal-failure outcome, the wrapper does not
    # increment (callers must use record_resolution_and_attempt for
    # that). This is a guardrail: nothing in the wrapper's name
    # implies counter bookkeeping.
    ledger.record_resolution_payload(
        recommendation_id="rec_wrapper",
        candidate_shadow_outcome={
            "status": "permanently_failed:no_post_event_chain",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
        },
    )
    counter = ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_wrapper",),
    ).fetchone()[0]
    assert int(counter) == 0


def test_pr_ae_atomic_helper_rolls_back_both_on_mid_tx_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The core P1 atomicity contract (Codex C1 audit):
    record_resolution_and_attempt MUST commit both the revision row and
    the counter UPDATE in the same transaction. If anything raises
    mid-transaction, the _tx context manager rolls back BOTH — there
    must be no partial state where the revision lands but the counter
    stays stale (which would let retrying revisions never escalate to
    permanently_failed).

    This test forces a mid-transaction failure by patching
    _write_record_within_tx to raise AFTER the revision insert would
    have landed but BEFORE the counter UPDATE runs. After the raise,
    the row must look unchanged: same revisions count, same counter.
    """
    import pytest

    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_atomic"))

    revisions_before = len(ledger.get_revisions("rec_atomic"))
    counter_before = int(ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_atomic",),
    ).fetchone()[0])

    real = ledger._write_record_within_tx  # noqa: SLF001
    crash_marker = RuntimeError("simulated crash between revision and counter update")

    def patched_write(cur, record):
        # Run the real write so the revision INSERT OR IGNORE
        # statement executes (would have landed a row in the
        # revisions table absent rollback), then crash.
        real(cur, record)
        raise crash_marker

    monkeypatch.setattr(ledger, "_write_record_within_tx", patched_write)

    with pytest.raises(RuntimeError):
        ledger.record_resolution_and_attempt(
            recommendation_id="rec_atomic",
            candidate_shadow_outcome={
                "status": "retrying",
                "labels": {"research_mid": True, "shadow_only": True,
                           "not_execution_grade": True},
            },
            increment_attempts=True,
        )

    # Counter unchanged: rollback undid the (not-yet-executed) UPDATE
    counter_after = int(ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_atomic",),
    ).fetchone()[0])
    assert counter_after == counter_before

    # No new revision row: rollback also undid the revision insert
    assert len(ledger.get_revisions("rec_atomic")) == revisions_before


def test_pr_ae_atomic_helper_writes_revision_and_counter_in_one_tx(
    tmp_path: Path,
) -> None:
    """Positive companion to the rollback test: the happy path must
    leave BOTH the revision row AND the counter increment visible
    after a single call."""
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_happy"))

    status = ledger.record_resolution_and_attempt(
        recommendation_id="rec_happy",
        candidate_shadow_outcome={
            "status": "permanently_failed:no_post_event_chain",
            "labels": {"research_mid": True, "shadow_only": True,
                       "not_execution_grade": True},
        },
        increment_attempts=True,
    )
    assert status == "revision"

    # Counter bumped
    counter = int(ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_happy",),
    ).fetchone()[0])
    assert counter == 1

    # Revision present and carries the terminal status
    revisions = ledger.get_revisions("rec_happy")
    assert revisions[-1]["record"]["candidate_shadow_outcome"]["status"] == \
        "permanently_failed:no_post_event_chain"


def test_pr_ae_atomic_helper_raises_for_unknown_id(tmp_path: Path) -> None:
    """KeyError on unknown id; lock and tx context managers clean up."""
    import pytest

    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    with pytest.raises(KeyError):
        ledger.record_resolution_and_attempt(
            recommendation_id="rec_ghost",
            candidate_shadow_outcome={"status": "ok",
                                      "labels": {"research_mid": True,
                                                 "shadow_only": True,
                                                 "not_execution_grade": True}},
            increment_attempts=False,
        )


def test_pr_ae_atomic_helper_duplicate_outcome_still_bumps_counter(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex C1b audit clarification): on the
    terminal-failure write path (increment_attempts=True), a duplicate
    revision payload STILL increments the counter.

    Subtle case: if a retry lands the EXACT same payload as the most
    recent revision, the revisions UNIQUE constraint dedupes the new
    row (record() returns "duplicate"). The counter still increments —
    the resolver did do an attempt, even if its output happened to
    match the prior state.

    Rationale: if we did NOT increment on duplicate, a row stuck in
    "retrying" with no upstream chain change would never escalate to
    permanently_failed (each retry would be a duplicate). Incrementing
    on duplicate preserves the retry-budget semantics.

    Pairs with the rule table in the design doc:
      - retrying  → increment_attempts=True  → counter bumps (always)
      - permanently_failed:*  → same
      - ok  → increment_attempts=False  → counter unchanged
      - awaiting_chain_data  → increment_attempts=False  → counter unchanged
    """
    ledger = RecommendationLedger(ledger_path=tmp_path / "ledger.sqlite")
    ledger.record(_forward_record(recommendation_id="rec_stuck"))

    outcome = {
        "status": "retrying",
        "labels": {"research_mid": True, "shadow_only": True,
                   "not_execution_grade": True},
    }
    s1 = ledger.record_resolution_and_attempt(
        recommendation_id="rec_stuck",
        candidate_shadow_outcome=outcome,
        increment_attempts=True,
    )
    assert s1 == "revision"
    s2 = ledger.record_resolution_and_attempt(
        recommendation_id="rec_stuck",
        candidate_shadow_outcome=outcome,
        increment_attempts=True,
    )
    assert s2 == "duplicate"  # no new revision row (idempotent)

    counter = int(ledger._conn.execute(  # noqa: SLF001
        "SELECT candidate_exit_resolver_attempts FROM recommendations WHERE recommendation_id = ?",
        ("rec_stuck",),
    ).fetchone()[0])
    # Counter bumped twice — retry budget burns down even when output
    # repeats. The row escalates to permanently_failed after MAX_ATTEMPTS
    # retries regardless of content drift.
    assert counter == 2
