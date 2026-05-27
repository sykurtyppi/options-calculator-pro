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
