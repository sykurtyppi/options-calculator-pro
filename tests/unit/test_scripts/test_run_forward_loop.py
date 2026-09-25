from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd
import pytest

import scripts.run_forward_loop as forward_loop
from services.calibration_service import IVExpansionCalibration
from services.baseline_evidence_store import BaselineEvidenceStore
from services.outcome_recorder import OutcomeStore, make_trade_id
from services.recommendation_ledger import RecommendationLedger
from services.structure_prior_store import StructurePriorStore
from scripts.run_forward_loop import run_daily_cycle


def _analysis(symbol: str, earnings_date: date, structure: str, recommendation: str = "Candidate") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        recommendation=recommendation,
        confidence_pct=72.0,
        setup_score=0.66,
        metrics={"calibration_phase": "bootstrap_prior"},
        rationale=["ok"],
        selector_output={
            "recommendation": recommendation,
            "best_structure": structure,
            "earnings_date": earnings_date.isoformat(),
            "confidence_pct": 72.0,
            "expected_edge_pct": 3.2,
            "expected_return_pct": 5.1,
            "runner_up_structures": ["otm_strangle"],
        },
        structure_scorecards=[
            {"structure": structure, "execution_penalty": 0.03},
        ],
        vol_snapshot={
            "as_of_date": date(2026, 4, 21).isoformat(),
            "earnings_date": earnings_date.isoformat(),
            "release_timing": "after market close",
            "days_to_earnings": (earnings_date - date(2026, 4, 21)).days,
            "data_quality_score": 0.91,
            "iv_rv_yz": 1.02,
            "iv_rv_har": 1.01,
            "historical_vs_implied_move_ratio": 1.18,
            "term_structure_slope": 0.0022,
            "near_term_spread_pct": 2.1,
            "liquidity_tier": "high",
        },
    )


def _option_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["contractSymbol", "strike", "bid", "ask", "lastPrice"])


def _marketdata_option_frame(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "optionSymbol",
        "side",
        "strike",
        "bid",
        "ask",
        "mid",
        "lastPrice",
        "underlyingPrice",
    ]
    return pd.DataFrame(rows, columns=columns)


def _install_fake_option_chain(
    monkeypatch,
    *,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float = 100.0,
    expiries: Optional[list[str]] = None,
) -> None:
    class _FakeTicker:
        options = expiries or ["2026-05-01"]

        def option_chain(self, _expiry: str) -> SimpleNamespace:
            return SimpleNamespace(calls=calls, puts=puts)

    monkeypatch.setattr(forward_loop.yf, "Ticker", lambda _symbol: _FakeTicker())
    monkeypatch.setattr(forward_loop, "_latest_spot_price", lambda _symbol: spot)
    monkeypatch.setattr(forward_loop, "record_provider_telemetry", lambda **_kwargs: None)


class _FakeMarketDataClient:
    def __init__(self, chain: pd.DataFrame, *, expiries: Optional[list[str]] = None, spot: float = 100.0) -> None:
        self.chain = chain
        self.expiries = expiries or ["2026-05-01"]
        self.spot = spot

    def is_available(self) -> bool:
        return True

    def get_expirations(self, _symbol: str) -> list[str]:
        return list(self.expiries)

    def get_quote(self, _symbol: str) -> float:
        return self.spot

    def get_option_chain(self, _symbol: str, **_kwargs) -> pd.DataFrame:
        return self.chain.copy()


def test_run_daily_cycle_records_entries_finalizes_exits_and_skips_missing_data(tmp_path: Path) -> None:
    today = date(2026, 4, 21)
    store_path = tmp_path / "outcomes.sqlite"
    cal_path = tmp_path / "calibration.json"
    prior_path = tmp_path / "priors.json"
    log_path = tmp_path / "learning_log.jsonl"
    ledger = RecommendationLedger(ledger_path=tmp_path / "recommendations.sqlite")
    baseline_store = BaselineEvidenceStore(tmp_path / "baselines.sqlite")

    store = OutcomeStore(store_path=store_path)
    cal = IVExpansionCalibration(store_path=cal_path)
    priors = StructurePriorStore(store_path=prior_path)

    # Two already-open paper trades due to exit today (earnings tomorrow).
    due_earnings = today + timedelta(days=1)
    for symbol, structure, entry_mid, marker in [
        ("SHOP", "atm_straddle", 5.0, "exit-shop"),
        ("CRM", "call_calendar", 2.5, "exit-crm"),
    ]:
        trade_id = make_trade_id(symbol, today - timedelta(days=4), structure)
        store.insert_entry(
            trade_id=trade_id,
            symbol=symbol,
            structure=structure,
            entry_date=today - timedelta(days=4),
            setup_score=0.61,
            source_type="paper",
            earnings_date=due_earnings,
            entry_mid=entry_mid,
            execution_penalty_at_entry=0.03,
            notes=json.dumps({"pricing_context": {"marker": marker}}),
        )

    def fake_screener_builder(**_: object) -> dict:
        return {
            "rows": [
                {"symbol": "AAPL", "status": "ranked", "dte": 6},
                {"symbol": "MSFT", "status": "ranked", "dte": 5},
                {"symbol": "NVDA", "status": "ranked", "dte": 4},
                {"symbol": "TSLA", "status": "ranked", "dte": 7},
            ]
        }

    def fake_analyzer(symbol: str, mda_client=None):  # noqa: ARG001
        earnings_date = today + timedelta(days=6)
        mapping = {
            "AAPL": _analysis("AAPL", earnings_date, "atm_straddle", "Best Candidate"),
            "MSFT": _analysis("MSFT", earnings_date, "call_calendar"),
            "NVDA": _analysis("NVDA", earnings_date, "otm_strangle"),
            "TSLA": _analysis("TSLA", earnings_date, "put_calendar"),
        }
        return mapping[symbol]

    def fake_price_fetcher(*, symbol: str, structure: str, earnings_date: date, as_of_date: date, context=None):  # noqa: ARG001
        if context is None:
            mids = {
                ("AAPL", "atm_straddle"): 6.0,
                ("MSFT", "call_calendar"): 2.8,
                ("NVDA", "otm_strangle"): 7.2,
                ("TSLA", "put_calendar"): None,
            }
            mid = mids.get((symbol, structure), 3.3)
            if mid is None:
                return {"mid": None, "reason": "missing_entry_mid"}
            return {"mid": mid, "context": {"marker": f"entry-{symbol.lower()}"}}

        marker = context.get("marker")
        exit_mids = {
            "exit-shop": 5.6,
            "exit-crm": 2.9,
        }
        mid = exit_mids.get(marker)
        if mid is None:
            return {"mid": None, "reason": "missing_exit_mid"}
        return {"mid": mid, "context": context}

    import services.calibration_service as _cs
    import services.structure_prior_store as _ps
    import services.outcome_recorder as _or
    import services.structure_scorecard as _sc

    orig_cal = _cs._calibration
    orig_prior = _ps._store
    orig_store = _or._store
    _cs._calibration = cal
    _ps._store = priors
    _or._store = store

    try:
        result1 = run_daily_cycle(
            today=today,
            dry_run=False,
            store=store,
            ledger=ledger,
            log_path=log_path,
            screener_builder=fake_screener_builder,
            analyzer=fake_analyzer,
            price_fetcher=fake_price_fetcher,
            baseline_store=baseline_store,
        )
        result2 = run_daily_cycle(
            today=today,
            dry_run=False,
            store=store,
            ledger=ledger,
            log_path=log_path,
            screener_builder=fake_screener_builder,
            analyzer=fake_analyzer,
            price_fetcher=fake_price_fetcher,
            baseline_store=baseline_store,
        )
    finally:
        _cs._calibration = orig_cal
        _ps._store = orig_prior
        _or._store = orig_store
        _sc.reload_walk_forward_priors()

    assert result1["entries"]["entries"] == 3
    assert result1["exits"]["exits"] == 2
    assert result1["entries"]["skipped"] == 1
    assert result1["entries"]["skip_reasons"] == {"missing_entry_mid": 1}
    assert result1["entries"]["discovered"] == 4
    assert result1["entries"]["analyzed"] == 4
    assert result1["entries"]["discovery_source"] == "ranked_screener"
    assert result2["entries"]["entries"] == 0
    assert result2["entries"]["duplicates"] == 3
    assert result2["entries"]["skip_reasons"]["duplicate_active_trade"] == 3
    assert result2["exits"]["exits"] == 0

    assert store.count() == 5
    assert store.count_finalized() == 2
    assert cal._n() == 2
    assert ledger.count() == 4
    assert baseline_store.count() == 6

    aapl_trade = store.get_trade(
        make_trade_id("AAPL", today, "atm_straddle", earnings_date=today + timedelta(days=6))
    )
    assert aapl_trade is not None
    aapl_notes = json.loads(str(aapl_trade["notes"]))
    assert aapl_trade["recommendation_id"]
    assert aapl_notes["structure_comparison"][0]["structure"] == "atm_straddle"
    assert aapl_notes["discovery_context"]["source"] == "ranked_screener"
    assert aapl_notes["recommendation_id"] == aapl_trade["recommendation_id"]

    prior_diag = priors.diagnostics()["structures"]
    assert prior_diag["atm_straddle"]["observation_count"] == 1
    assert prior_diag["call_calendar"]["observation_count"] == 1

    logs = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any(item["event_type"] == "entry" and item["symbol"] == "AAPL" for item in logs)
    assert any(item["event_type"] == "exit" and item["symbol"] == "SHOP" for item in logs)
    assert any(item["event_type"] == "skip" and item["symbol"] == "TSLA" for item in logs)
    assert any(
        item["event_type"] == "entry"
        and item["symbol"] == "AAPL"
        and item["discovery_source"] == "ranked_screener"
        and item["structure_comparison"][0]["structure"] == "atm_straddle"
        for item in logs
    )


def test_run_daily_cycle_prefers_edge_screener_discovery_when_marketdata_is_available(tmp_path: Path) -> None:
    today = date(2026, 4, 23)
    store = OutcomeStore(store_path=tmp_path / "outcomes.sqlite")
    ledger = RecommendationLedger(ledger_path=tmp_path / "recommendations.sqlite")
    baseline_store = BaselineEvidenceStore(tmp_path / "baselines.sqlite")
    log_path = tmp_path / "learning_log.jsonl"

    class _FakeMDA:
        def is_available(self) -> bool:
            return True

    def fake_edge_screener(**_: object) -> dict:
        return {
            "qualified_count": 1,
            "marginal_count": 0,
            "excluded_count": 0,
            "rows": [
                {
                    "symbol": "MSFT",
                    "earnings_date": (today + timedelta(days=6)).isoformat(),
                    "release_timing": "AMC",
                    "status": "QUALIFIED",
                    "status_reason": "ok",
                    "detail_metrics": {"days_to_earnings": 6},
                }
            ],
        }

    def fake_analyzer(symbol: str, mda_client=None):  # noqa: ARG001
        assert symbol == "MSFT"
        return _analysis("MSFT", today + timedelta(days=6), "otm_strangle", "Candidate")

    def fake_price_fetcher(*, symbol: str, structure: str, earnings_date: date, as_of_date: date, context=None):  # noqa: ARG001
        assert symbol == "MSFT"
        return {"mid": 7.25, "context": {"marker": "entry-msft"}}

    original_edge_screener = forward_loop.build_edge_screener
    forward_loop.build_edge_screener = fake_edge_screener
    try:
        result = run_daily_cycle(
            today=today,
            dry_run=False,
            store=store,
            ledger=ledger,
            log_path=log_path,
            analyzer=fake_analyzer,
            price_fetcher=fake_price_fetcher,
            baseline_store=baseline_store,
            mda_client=_FakeMDA(),
        )
    finally:
        forward_loop.build_edge_screener = original_edge_screener

    assert result["entries"]["entries"] == 1
    assert result["entries"]["discovered"] == 1
    assert result["entries"]["analyzed"] == 1
    assert result["entries"]["discovery_source"] == "edge_screener"

    trade = store.get_trade(
        make_trade_id("MSFT", today, "otm_strangle", earnings_date=today + timedelta(days=6))
    )
    assert trade is not None
    assert trade["recommendation_id"]
    assert ledger.count() == 1
    notes = json.loads(str(trade["notes"]))
    assert notes["discovery_context"]["source"] == "edge_screener"
    assert notes["discovery_context"]["status"] == "QUALIFIED"
    assert notes["structure_comparison"][0]["structure"] == "otm_strangle"

    logs = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any(
        item["event_type"] == "entry"
        and item["symbol"] == "MSFT"
        and item["discovery_source"] == "edge_screener"
        and item["structure_comparison"][0]["structure"] == "otm_strangle"
        for item in logs
    )


def test_run_daily_cycle_dry_run_does_not_write_ledger_or_trades(tmp_path: Path) -> None:
    today = date(2026, 4, 24)
    store = OutcomeStore(store_path=tmp_path / "outcomes.sqlite")
    ledger = RecommendationLedger(ledger_path=tmp_path / "recommendations.sqlite")
    seen = {"record_to_ledger": None}

    def fake_screener_builder(**_: object) -> dict:
        return {"rows": [{"symbol": "AAPL", "status": "ranked", "dte": 5}]}

    def fake_analyzer(symbol: str, mda_client=None, record_to_ledger=True):  # noqa: ARG001
        seen["record_to_ledger"] = record_to_ledger
        return _analysis(symbol, today + timedelta(days=5), "atm_straddle", "Candidate")

    def fake_price_fetcher(*, symbol: str, structure: str, earnings_date: date, as_of_date: date, context=None):  # noqa: ARG001
        return {"mid": 4.2, "context": {"marker": "dry-run"}}

    result = run_daily_cycle(
        today=today,
        dry_run=True,
        store=store,
        ledger=ledger,
        log_path=tmp_path / "learning_log.jsonl",
        screener_builder=fake_screener_builder,
        analyzer=fake_analyzer,
        price_fetcher=fake_price_fetcher,
    )

    assert seen["record_to_ledger"] is False
    assert result["entries"]["entries"] == 1
    assert result["entries"]["ledger_records"] == 0
    assert ledger.count() == 0
    assert store.count() == 0


def test_otm_strangle_exact_target_quote_succeeds(monkeypatch) -> None:
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([
            {"contractSymbol": "AAPL_C_103", "strike": 103.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
        ]),
        puts=_option_frame([
            {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        ]),
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    assert quote["mid"] == 2.0
    provenance = quote["bid_ask_mid"]["provenance"]
    assert provenance["requested_call_wing_strike"] == 103.0
    assert provenance["selected_call_wing_strike"] == 103.0
    assert provenance["requested_put_wing_strike"] == 97.0
    assert provenance["selected_put_wing_strike"] == 97.0
    assert provenance["call_quote_quality_label"] == "exact_wing_mid"
    assert provenance["put_quote_quality_label"] == "exact_wing_mid"
    assert quote["bid_ask_mid"]["legs"]["call"]["quote_quality_label"] == "exact_wing_mid"
    assert quote["surface_quality"]["status"] == "record_only"
    assert "missing_expiration_depth" in quote["surface_quality"]["warning_flags"]


def test_forward_quote_prefers_marketdata_app_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        forward_loop.yf,
        "Ticker",
        lambda _symbol: (_ for _ in ()).throw(AssertionError("yfinance fallback should not be used")),
    )
    monkeypatch.setattr(forward_loop, "record_provider_telemetry", lambda **_kwargs: None)
    mda_client = _FakeMarketDataClient(
        _marketdata_option_frame([
            {
                "optionSymbol": "AAPL_C_103",
                "side": "call",
                "strike": 103.0,
                "bid": 1.0,
                "ask": 1.2,
                "mid": 1.1,
                "lastPrice": 1.1,
                "underlyingPrice": 100.0,
            },
            {
                "optionSymbol": "AAPL_P_97",
                "side": "put",
                "strike": 97.0,
                "bid": 0.8,
                "ask": 1.0,
                "mid": 0.9,
                "lastPrice": 0.9,
                "underlyingPrice": 100.0,
            },
        ])
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
        mda_client=mda_client,
    )

    assert quote["mid"] == 2.0
    assert quote["quote_source"] == "marketdata_app"
    assert quote["quote_quality"] == "marketdata_app_paper_research_mid_not_execution_grade"
    assert quote["bid_ask_mid"]["legs"]["call"]["contract"] == "AAPL_C_103"


def test_otm_strangle_target_call_missing_mid_uses_nearest_valid_call(monkeypatch) -> None:
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([
            {"contractSymbol": "AAPL_C_103_BAD", "strike": 103.0, "bid": 0.0, "ask": 0.0, "lastPrice": 0.0},
            {"contractSymbol": "AAPL_C_104", "strike": 104.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
        ]),
        puts=_option_frame([
            {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        ]),
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    provenance = quote["bid_ask_mid"]["provenance"]
    assert quote["mid"] == 2.0
    assert provenance["selected_call_wing_strike"] == 104.0
    assert provenance["call_quote_quality_label"] == "nearest_valid_wing_mid"
    assert provenance["call_fallback_distance"] == 1.0


def test_otm_strangle_target_put_missing_mid_uses_nearest_valid_put(monkeypatch) -> None:
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([
            {"contractSymbol": "AAPL_C_103", "strike": 103.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
        ]),
        puts=_option_frame([
            {"contractSymbol": "AAPL_P_97_BAD", "strike": 97.0, "bid": 0.0, "ask": 0.0, "lastPrice": 0.0},
            {"contractSymbol": "AAPL_P_96", "strike": 96.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        ]),
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    provenance = quote["bid_ask_mid"]["provenance"]
    assert quote["mid"] == 2.0
    assert provenance["selected_put_wing_strike"] == 96.0
    assert provenance["put_quote_quality_label"] == "nearest_valid_wing_mid"
    assert provenance["put_fallback_distance"] == 1.0


def test_otm_strangle_empty_call_chain_fails_with_exact_reason(monkeypatch) -> None:
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([]),
        puts=_option_frame([
            {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        ]),
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    assert quote["mid"] is None
    assert quote["reason"] == "empty_call_chain"
    assert quote["bid_ask_mid"]["final_reason"] == "empty_call_chain"


def test_otm_strangle_empty_put_chain_fails_with_exact_reason(monkeypatch) -> None:
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([
            {"contractSymbol": "AAPL_C_103", "strike": 103.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
        ]),
        puts=_option_frame([]),
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    assert quote["mid"] is None
    assert quote["reason"] == "empty_put_chain"
    assert quote["bid_ask_mid"]["final_reason"] == "empty_put_chain"


def test_otm_strangle_bad_bid_ask_fails_with_exact_reason(monkeypatch) -> None:
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([
            {"contractSymbol": "AAPL_C_103_BAD", "strike": 103.0, "bid": 2.0, "ask": 1.0, "lastPrice": 1.5},
        ]),
        puts=_option_frame([
            {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        ]),
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    assert quote["mid"] is None
    assert quote["reason"] == "bad_bid_ask"
    assert quote["bid_ask_mid"]["final_reason"] == "bad_bid_ask"


def test_otm_strangle_no_valid_strike_within_cap_fails(monkeypatch) -> None:
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([
            {"contractSymbol": "AAPL_C_112", "strike": 112.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
        ]),
        puts=_option_frame([
            {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        ]),
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    assert quote["mid"] is None
    assert quote["reason"] == "wing_strike_unavailable"
    assert quote["bid_ask_mid"]["provenance"]["max_fallback_distance"] == 2.5


def test_otm_strangle_quote_provenance_is_persisted_in_paper_entry(tmp_path: Path, monkeypatch) -> None:
    today = date(2026, 4, 24)
    store = OutcomeStore(store_path=tmp_path / "outcomes.sqlite")
    ledger = RecommendationLedger(ledger_path=tmp_path / "recommendations.sqlite")
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([
            {"contractSymbol": "AAPL_C_103_BAD", "strike": 103.0, "bid": 0.0, "ask": 0.0, "lastPrice": 0.0},
            {"contractSymbol": "AAPL_C_104", "strike": 104.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
        ]),
        puts=_option_frame([
            {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        ]),
    )

    def fake_screener_builder(**_: object) -> dict:
        return {"rows": [{"symbol": "AAPL", "status": "ranked", "dte": 5}]}

    def fake_analyzer(symbol: str, mda_client=None):  # noqa: ARG001
        return _analysis(symbol, today + timedelta(days=5), "otm_strangle", "Candidate")

    result = run_daily_cycle(
        today=today,
        dry_run=False,
        store=store,
        ledger=ledger,
        log_path=tmp_path / "learning_log.jsonl",
        screener_builder=fake_screener_builder,
        analyzer=fake_analyzer,
        price_fetcher=forward_loop.fetch_structure_quote,
    )

    assert result["entries"]["entries"] == 1
    trade = store.get_trade(
        make_trade_id("AAPL", today, "otm_strangle", earnings_date=today + timedelta(days=5))
    )
    assert trade is not None
    notes = json.loads(str(trade["notes"]))
    provenance = notes["entry_bid_ask_mid"]["provenance"]
    assert provenance["requested_call_wing_strike"] == 103.0
    assert provenance["selected_call_wing_strike"] == 104.0
    assert provenance["call_quote_quality_label"] == "nearest_valid_wing_mid"
    assert notes["entry_bid_ask_mid"]["legs"]["call"]["bid"] == 1.0
    assert trade["evidence_quality_status"] == "record_only"
    assert trade["claim_allowed"] == 0
    assert trade["execution_grade"] == 0
    reasons = json.loads(str(trade["evidence_quality_reasons_json"]))
    assert "call_fallback_wing_selected" in reasons
    assert "provider_research_grade_yfinance" in reasons
    assert "quote_not_execution_grade" in reasons
    assert "surface_quality_record_only" in reasons
    assert trade["surface_quality_status"] == "record_only"
    entry_scenarios = json.loads(str(trade["entry_execution_scenarios_json"]))
    assert entry_scenarios["scenario_values"]["mid"] == 2.0
    assert entry_scenarios["scenario_values"]["cross_50"] == 2.2


# ── Iron condor booking (defined-risk credit structure) ──────────────────────
#
# Geometry used by these tests (spot = 100):
#   short call 103 (mid 1.1)   long call 105 (mid 0.5)
#   short put   97 (mid 0.9)   long put   95 (mid 0.4)
#   net credit = (1.1 + 0.9) - (0.5 + 0.4) = 1.1
#   wing width = 2.0  →  max loss = 2.0 - 1.1 = 0.9


def _condor_chain():
    calls = _option_frame([
        {"contractSymbol": "AAPL_C_103", "strike": 103.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
        {"contractSymbol": "AAPL_C_105", "strike": 105.0, "bid": 0.4, "ask": 0.6, "lastPrice": 0.5},
    ])
    puts = _option_frame([
        {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        {"contractSymbol": "AAPL_P_95", "strike": 95.0, "bid": 0.3, "ask": 0.5, "lastPrice": 0.4},
    ])
    return calls, puts


def test_iron_condor_quote_forms_defined_risk_credit(monkeypatch) -> None:
    calls, puts = _condor_chain()
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts)

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="iron_condor",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    # `mid` is the NET CREDIT received, and it is positive so the entry gate
    # (entry_mid > 0) accepts it exactly like a debit structure's premium.
    assert quote["mid"] == pytest.approx(1.1)
    ctx = quote["context"]
    assert ctx["short_call_strike"] == 103.0
    assert ctx["long_call_strike"] == 105.0
    assert ctx["short_put_strike"] == 97.0
    assert ctx["long_put_strike"] == 95.0
    assert ctx["wing_width"] == pytest.approx(2.0)
    # Capital at risk persisted at entry, used as the exit return base.
    assert ctx["max_loss_per_unit"] == pytest.approx(0.9)
    # Leg names must carry the short_/long_ prefix so execution_scenarios can
    # derive fill direction.
    assert set(quote["bid_ask_mid"]["legs"]) == {"short_call", "long_call", "short_put", "long_put"}


def test_iron_condor_skips_when_no_protective_call_wing(monkeypatch) -> None:
    # Only the short call exists — there is no strike beyond it to buy, so the
    # structure would be a naked short. It must be refused, not booked.
    calls = _option_frame([
        {"contractSymbol": "AAPL_C_103", "strike": 103.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
    ])
    _, puts = _condor_chain()
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts)

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="iron_condor",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    assert quote["mid"] is None
    assert quote["reason"] == "no_condor_call_wing"


def test_iron_condor_rejects_non_positive_credit(monkeypatch) -> None:
    # Wings priced above the body → the "condor" would cost money to put on.
    calls = _option_frame([
        {"contractSymbol": "AAPL_C_103", "strike": 103.0, "bid": 0.1, "ask": 0.3, "lastPrice": 0.2},
        {"contractSymbol": "AAPL_C_105", "strike": 105.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
    ])
    puts = _option_frame([
        {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.1, "ask": 0.3, "lastPrice": 0.2},
        {"contractSymbol": "AAPL_P_95", "strike": 95.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts)

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="iron_condor",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    assert quote["mid"] is None
    assert quote["reason"] == "non_positive_condor_credit"


def test_realized_trade_math_is_unchanged_for_long_debit() -> None:
    # Regression guard: the refactor must reproduce the original
    # ((exit - entry) / entry) formula exactly for debit structures.
    gross, pnl, expansion = forward_loop._realized_trade_math(
        structure="atm_straddle", entry_mid=4.0, exit_mid=5.0,
    )
    assert gross == pytest.approx(25.0)
    assert pnl == pytest.approx(100.0)
    assert expansion == pytest.approx(25.0)
    # A max-loss hint must NOT change a debit structure's math.
    assert forward_loop._realized_trade_math(
        structure="atm_straddle", entry_mid=4.0, exit_mid=5.0, capital_at_risk=0.9,
    )[0] == pytest.approx(25.0)


def test_realized_trade_math_condor_win_flips_sign_and_uses_max_loss() -> None:
    # Sold for 1.1, bought back for 0.4 → profit 0.7 per share on 0.9 at risk.
    gross, pnl, expansion = forward_loop._realized_trade_math(
        structure="iron_condor", entry_mid=1.1, exit_mid=0.4, capital_at_risk=0.9,
    )
    assert pnl == pytest.approx(70.0)
    assert gross == pytest.approx((0.7 / 0.9) * 100.0)
    # Expansion keeps its vol meaning: the structure got CHEAPER, so negative.
    assert expansion == pytest.approx(((0.4 - 1.1) / 1.1) * 100.0)
    assert expansion < 0 < gross


def test_realized_trade_math_condor_loss_is_negative() -> None:
    # Sold for 1.1, costs 1.8 to close → a loss, even though the structure's
    # market value rose (which for a long structure would be a gain).
    gross, pnl, expansion = forward_loop._realized_trade_math(
        structure="iron_condor", entry_mid=1.1, exit_mid=1.8, capital_at_risk=0.9,
    )
    assert pnl == pytest.approx(-70.0)
    assert gross == pytest.approx((-0.7 / 0.9) * 100.0)
    assert expansion > 0 > gross


def test_realized_trade_math_condor_falls_back_to_credit_base() -> None:
    # Legacy/missing max_loss must still produce a correctly SIGNED return
    # rather than dividing by nothing.
    gross, pnl, _ = forward_loop._realized_trade_math(
        structure="iron_condor", entry_mid=1.1, exit_mid=0.4, capital_at_risk=None,
    )
    assert pnl == pytest.approx(70.0)
    assert gross == pytest.approx((0.7 / 1.1) * 100.0)


def test_exit_detection_books_condor_profit_when_premium_decays(tmp_path: Path) -> None:
    """End-to-end wiring: max_loss_per_unit recorded in the entry notes must
    reach the exit and produce a POSITIVE return when the condor cheapens."""
    as_of = date(2026, 4, 24)
    store = OutcomeStore(store_path=tmp_path / "outcomes.sqlite")
    store.insert_entry(
        trade_id="AAPL-condor",
        symbol="AAPL",
        structure="iron_condor",
        entry_date=as_of - timedelta(days=3),
        setup_score=0.7,
        source_type="paper",
        earnings_date=as_of + timedelta(days=1),  # due for exit on as_of
        entry_mid=1.1,
        execution_penalty_at_entry=0.0,
        notes=json.dumps({"pricing_context": {"max_loss_per_unit": 0.9, "wing_width": 2.0}}),
    )

    captured: dict = {}

    def fake_finalizer(**kwargs):
        captured.update(kwargs)
        return {}

    def fake_price_fetcher(**_kwargs):
        return {"mid": 0.4, "context": {}, "bid_ask_mid": {}}

    summary = forward_loop.run_exit_detection(
        today=as_of,
        store=store,
        log_path=tmp_path / "learning_log.jsonl",
        price_fetcher=fake_price_fetcher,
        finalizer=fake_finalizer,
        baseline_store=BaselineEvidenceStore(store_path=tmp_path / "baseline.sqlite"),
    )

    assert summary["exits"] == 1
    assert captured["realized_pnl"] == pytest.approx(70.0)
    assert captured["realized_return_pct"] == pytest.approx((0.7 / 0.9) * 100.0)
    # The structure got cheaper — the vol diagnostic is negative while the
    # position return is positive.
    assert captured["realized_expansion_pct"] < 0


def test_iron_condor_reprice_keeps_booked_strikes_after_spot_moves_through_them(monkeypatch) -> None:
    """The condor's loss case is spot moving THROUGH a short strike. Re-pricing
    must still quote the strikes actually held: the OTM-discovery filter would
    have dropped the booked 103 call once spot printed 106 and silently priced a
    different option, describing a trade that was never held."""
    calls, puts = _condor_chain()
    # Spot has rallied to 106 — above BOTH the short (103) and long (105) calls.
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=106.0)

    booked_context = {
        "front_expiry": "2026-05-01",
        "short_call_strike": 103.0,
        "long_call_strike": 105.0,
        "short_put_strike": 97.0,
        "long_put_strike": 95.0,
    }
    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="iron_condor",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 5, 1),
        context=booked_context,
    )

    assert quote["bid_ask_mid"]["provenance"]["reprice_of_booked_strikes"] is True
    ctx = quote["context"]
    assert ctx["short_call_strike"] == 103.0
    assert ctx["long_call_strike"] == 105.0
    assert ctx["short_put_strike"] == 97.0
    assert ctx["long_put_strike"] == 95.0
    # Same four legs → same credit as the entry chain prices.
    assert quote["mid"] == pytest.approx(1.1)


def test_iron_condor_reprice_reports_a_delisted_booked_strike(monkeypatch) -> None:
    # The booked long call is no longer in the chain — refuse rather than
    # substitute a different strike.
    calls = _option_frame([
        {"contractSymbol": "AAPL_C_103", "strike": 103.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
    ])
    _, puts = _condor_chain()
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts)

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="iron_condor",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 5, 1),
        context={
            "front_expiry": "2026-05-01",
            "short_call_strike": 103.0, "long_call_strike": 105.0,
            "short_put_strike": 97.0, "long_put_strike": 95.0,
        },
    )

    assert quote["mid"] is None
    assert quote["reason"] == "condor_strike_no_longer_listed"
    assert "long_call" in quote["bid_ask_mid"]["provenance"]["missing_booked_legs"]


def test_strangle_reprice_keeps_booked_strike_after_spot_moves_through_it(monkeypatch) -> None:
    """Regression for the live AMZN/TTD mis-pricing: once spot trades through the
    booked short strike, OTM discovery dropped it and the fallback substituted a
    further-OTM (cheaper) strike, understating the exit and the realized return.
    Re-pricing must quote the strike actually held."""
    calls = _option_frame([
        {"contractSymbol": "AMZN_C_260", "strike": 260.0, "bid": 4.3, "ask": 4.7, "lastPrice": 4.5},
        {"contractSymbol": "AMZN_C_265", "strike": 265.0, "bid": 2.8, "ask": 3.2, "lastPrice": 3.0},
    ])
    puts = _option_frame([
        {"contractSymbol": "AMZN_P_245", "strike": 245.0, "bid": 1.5, "ask": 1.9, "lastPrice": 1.7},
    ])
    # Spot has rallied through the booked 260 call, exactly the live case.
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=262.0)

    quote = forward_loop.fetch_structure_quote(
        symbol="AMZN",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 28),
        as_of_date=date(2026, 4, 29),
        context={"front_expiry": "2026-05-01", "call_strike": 260.0, "put_strike": 245.0},
    )

    provenance = quote["bid_ask_mid"]["provenance"]
    assert provenance["selected_call_wing_strike"] == 260.0, "must re-price the BOOKED strike, not the 265"
    assert provenance["call_mid"] == pytest.approx(4.5)
    # 4.5 + 1.7 — using the 265 call would have understated this by 1.5.
    assert quote["mid"] == pytest.approx(6.2)


def test_strangle_reprice_reports_a_delisted_booked_strike(monkeypatch) -> None:
    calls = _option_frame([
        {"contractSymbol": "AMZN_C_265", "strike": 265.0, "bid": 2.8, "ask": 3.2, "lastPrice": 3.0},
    ])
    puts = _option_frame([
        {"contractSymbol": "AMZN_P_245", "strike": 245.0, "bid": 1.5, "ask": 1.9, "lastPrice": 1.7},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=262.0)

    quote = forward_loop.fetch_structure_quote(
        symbol="AMZN",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 28),
        as_of_date=date(2026, 4, 29),
        context={"front_expiry": "2026-05-01", "call_strike": 260.0, "put_strike": 245.0},
    )

    assert quote["mid"] is None
    assert quote["reason"] == "booked_strike_no_longer_listed"


def test_strangle_discovery_still_uses_otm_selection(monkeypatch) -> None:
    """With no booked strikes in context this is a NEW structure, so wing
    discovery against current spot must be unchanged."""
    _install_fake_option_chain(
        monkeypatch,
        calls=_option_frame([
            {"contractSymbol": "AAPL_C_103", "strike": 103.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
        ]),
        puts=_option_frame([
            {"contractSymbol": "AAPL_P_97", "strike": 97.0, "bid": 0.8, "ask": 1.0, "lastPrice": 0.9},
        ]),
    )

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL",
        structure="otm_strangle",
        earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 4, 24),
    )

    assert quote["mid"] == pytest.approx(2.0)
    assert quote["bid_ask_mid"]["provenance"]["call_quote_quality_label"] == "exact_wing_mid"


def test_condor_reprice_books_a_max_loss_exit_instead_of_dropping_it(monkeypatch) -> None:
    """AUDIT HIGH: the net_credit/max_loss sanity gates are OPEN-time checks. They
    also fired when re-pricing an open position, so as a condor approached max
    loss (breached vertical worth ~the wing width) ordinary deep-ITM quote noise
    tripped them and the exit quote was dropped. Exits are one-shot
    (trades_due_for_exit matches only earnings_date == as_of + 1 day, and the skip
    path leaves status='open'), so that trade was orphaned forever — deleting the
    WORST outcomes from the learning ledger in an outcome-correlated way."""
    # Spot gapped far above the call spread: it is fully breached.
    # Call vertical: 9.50 - 7.52 = 1.98, inside its own 2.00 width (valid).
    calls = _option_frame([
        {"contractSymbol": "C_103", "strike": 103.0, "bid": 9.30, "ask": 9.70, "lastPrice": 9.5},
        {"contractSymbol": "C_105", "strike": 105.0, "bid": 7.42, "ask": 7.62, "lastPrice": 7.52},
    ])
    # Put vertical: 0.13 - 0.06 = 0.07 of residual time value on the UNTESTED
    # side. Total 2.05 therefore exceeds max(call_width, put_width) = 2.00 while
    # both verticals are individually valid — the legitimate early-close mark.
    puts = _option_frame([
        {"contractSymbol": "P_97", "strike": 97.0, "bid": 0.10, "ask": 0.16, "lastPrice": 0.13},
        {"contractSymbol": "P_95", "strike": 95.0, "bid": 0.04, "ask": 0.08, "lastPrice": 0.06},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=112.0)

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor",
        earnings_date=date(2026, 4, 30), as_of_date=date(2026, 5, 1),
        context={"front_expiry": "2026-05-01",
                 "short_call_strike": 103.0, "long_call_strike": 105.0,
                 "short_put_strike": 97.0, "long_put_strike": 95.0},
    )

    # It must produce a quote, not a skip.
    assert quote["mid"] is not None, f"max-loss exit was dropped: {quote.get('reason')}"
    # The mark is the TRUE value, not truncated. This system always closes before
    # expiration, so the untested side still carries time value and an early close
    # can legitimately cost more than the expiration-defined max loss. Clamping at
    # max(call_width, put_width) hid that as a suspiciously clean -100%; the real
    # no-arbitrage ceiling is call_width + put_width.
    assert quote["mid"] > 2.0, "real early-close cost must not be truncated to the expiry max loss"
    assert quote["mid"] <= 4.0 + 1e-9, "must stay inside the no-arbitrage ceiling"
    assert "reprice_value_clamped_from" not in quote["bid_ask_mid"]["provenance"]


def test_condor_discovery_still_refuses_a_broken_credit(monkeypatch) -> None:
    """The same gates must STILL protect the open decision."""
    calls = _option_frame([
        {"contractSymbol": "C_103", "strike": 103.0, "bid": 0.1, "ask": 0.3, "lastPrice": 0.2},
        {"contractSymbol": "C_105", "strike": 105.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
    ])
    puts = _option_frame([
        {"contractSymbol": "P_97", "strike": 97.0, "bid": 0.1, "ask": 0.3, "lastPrice": 0.2},
        {"contractSymbol": "P_95", "strike": 95.0, "bid": 1.0, "ask": 1.2, "lastPrice": 1.1},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts)
    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor",
        earnings_date=date(2026, 4, 30), as_of_date=date(2026, 4, 24),
    )
    assert quote["mid"] is None
    assert quote["reason"] == "non_positive_condor_credit"


def test_condor_realized_loss_cannot_exceed_defined_max_loss() -> None:
    """With the exit clamped to the wing width, return-on-risk bottoms out at
    -100% — a defined-risk structure must never book worse than its max loss."""
    gross, pnl, _ = forward_loop._realized_trade_math(
        structure="iron_condor", entry_mid=1.1, exit_mid=2.0, capital_at_risk=0.9,
    )
    assert gross == pytest.approx(-100.0)
    assert pnl == pytest.approx(-90.0)


def test_condor_max_win_exit_is_recorded_not_deleted(monkeypatch) -> None:
    """RE-AUDIT HIGH: every leg decaying to nothing is the MODAL success for a
    short-vol structure, and those legs go bid-less. _mid_from_row voids any leg
    with bid <= 0, so one bid-less leg out of four voided the whole exit and the
    trade was orphaned. Losers keep expensive two-sided ITM legs that always
    quote, so the ledger kept losers and deleted winners."""
    calls = _option_frame([
        {"contractSymbol": "C_103", "strike": 103.0, "bid": 0.00, "ask": 0.05, "lastPrice": 0.02},
        {"contractSymbol": "C_105", "strike": 105.0, "bid": 0.00, "ask": 0.02, "lastPrice": 0.01},
    ])
    puts = _option_frame([
        {"contractSymbol": "P_97", "strike": 97.0, "bid": 0.00, "ask": 0.05, "lastPrice": 0.02},
        {"contractSymbol": "P_95", "strike": 95.0, "bid": 0.00, "ask": 0.02, "lastPrice": 0.01},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=100.0)

    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor",
        earnings_date=date(2026, 4, 30), as_of_date=date(2026, 5, 1),
        context={"front_expiry": "2026-05-01",
                 "short_call_strike": 103.0, "long_call_strike": 105.0,
                 "short_put_strike": 97.0, "long_put_strike": 95.0},
    )
    assert quote["mid"] is not None, f"max-WIN exit was deleted: {quote.get('reason')}"
    # Shorts are bought back near the ask (~0.025 each); the unsellable long
    # wings mark at 0. Cost to close is tiny, i.e. nearly the full credit kept.
    assert quote["mid"] == pytest.approx(0.05)


def test_condor_entry_still_refuses_a_bidless_leg(monkeypatch) -> None:
    """The relaxation is exit-only. Refusing to OPEN on a phantom quote biases
    nothing, because no trade is booked."""
    calls, puts = _condor_chain()
    calls.loc[calls["strike"] == 103.0, "bid"] = 0.0
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts)
    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor",
        earnings_date=date(2026, 4, 30), as_of_date=date(2026, 4, 24),
    )
    assert quote["mid"] is None


def test_condor_reprice_refuses_an_arbitrage_violating_quote(monkeypatch) -> None:
    """RE-AUDIT MEDIUM: clamping a LARGE violation fabricated an outcome — a raw
    value of -0.80 was repaired to 0.0 and booked as the exact maximum win.
    Noise is repaired; unusable data is refused."""
    # Long wings quoted far above the shorts at the same widths: an arbitrage
    # violation, i.e. the chain is broken.
    calls = _option_frame([
        {"contractSymbol": "C_103", "strike": 103.0, "bid": 0.10, "ask": 0.20, "lastPrice": 0.15},
        {"contractSymbol": "C_105", "strike": 105.0, "bid": 0.90, "ask": 1.10, "lastPrice": 1.00},
    ])
    puts = _option_frame([
        {"contractSymbol": "P_97", "strike": 97.0, "bid": 0.10, "ask": 0.20, "lastPrice": 0.15},
        {"contractSymbol": "P_95", "strike": 95.0, "bid": 0.90, "ask": 1.10, "lastPrice": 1.00},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=100.0)
    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor",
        earnings_date=date(2026, 4, 30), as_of_date=date(2026, 5, 1),
        context={"front_expiry": "2026-05-01",
                 "short_call_strike": 103.0, "long_call_strike": 105.0,
                 "short_put_strike": 97.0, "long_put_strike": 95.0},
    )
    assert quote["mid"] is None
    assert quote["reason"] == "condor_quote_outside_no_arbitrage_bounds"


def test_closing_leg_mid_rejects_an_unknown_bid_but_accepts_a_real_zero() -> None:
    """RE-AUDIT CRITICAL: a missing/NaN bid means UNKNOWN, not zero. Coercing it
    to zero halved the cost of buying a short leg back — a leg quoted `NaN x 3.00`
    marked at 1.50 instead of ~2.95, converting a -95% loss into a +267% win that
    passed every downstream check because the number looked plausible."""
    import pandas as _pd

    def row(bid, ask):
        return _pd.Series({"strike": 103.0, "bid": bid, "ask": ask})

    # UNKNOWN bid in any form -> reject, never mark at ask/2.
    for bad in (float("nan"), None, -1.0, float("inf")):
        assert forward_loop._closing_leg_mid(row(bad, 3.00), action="buy_to_close") is None, bad
        assert forward_loop._closing_leg_mid(row(bad, 3.00), action="sell_to_close") is None, bad
    # A leg with no bid column at all is also unknown.
    assert forward_loop._closing_leg_mid(_pd.Series({"strike": 103.0, "ask": 3.0}), action="buy_to_close") is None
    # A PRESENT, VALID zero bid is the one relaxation: buying back is cheap,
    # selling gets nothing.
    assert forward_loop._closing_leg_mid(row(0.0, 0.05), action="buy_to_close") == pytest.approx(0.025)
    assert forward_loop._closing_leg_mid(row(0.0, 0.05), action="sell_to_close") == 0.0
    # Normal two-sided quotes are unchanged, and crossed quotes are refused.
    assert forward_loop._closing_leg_mid(row(2.90, 3.00), action="buy_to_close") == pytest.approx(2.95)
    assert forward_loop._closing_leg_mid(row(3.10, 3.00), action="buy_to_close") is None


def test_condor_reprice_refuses_a_single_broken_vertical(monkeypatch) -> None:
    """RE-AUDIT MEDIUM-HIGH: each short vertical is independently bounded by its
    OWN width. Checking only the aggregate let a call vertical marked 2.50
    against a 2.00 width hide behind a cheap put vertical (total 2.515 < the 4.00
    aggregate ceiling) and book a loss roughly double the structural maximum."""
    calls = _option_frame([
        {"contractSymbol": "C_103", "strike": 103.0, "bid": 9.90, "ask": 10.10, "lastPrice": 10.0},
        {"contractSymbol": "C_105", "strike": 105.0, "bid": 7.40, "ask": 7.60, "lastPrice": 7.5},
    ])  # call vertical = 2.50 vs width 2.00 -> impossible
    puts = _option_frame([
        {"contractSymbol": "P_97", "strike": 97.0, "bid": 0.01, "ask": 0.03, "lastPrice": 0.02},
        {"contractSymbol": "P_95", "strike": 95.0, "bid": 0.005, "ask": 0.01, "lastPrice": 0.007},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=112.0)
    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor",
        earnings_date=date(2026, 4, 30), as_of_date=date(2026, 5, 1),
        context={"front_expiry": "2026-05-01",
                 "short_call_strike": 103.0, "long_call_strike": 105.0,
                 "short_put_strike": 97.0, "long_put_strike": 95.0},
    )
    assert quote["mid"] is None
    assert quote["reason"] == "condor_quote_outside_no_arbitrage_bounds"
    assert quote["bid_ask_mid"]["provenance"]["violating_leg"] == "call_vertical"


def test_condor_repair_tolerance_scales_with_capital_at_risk(monkeypatch) -> None:
    """RE-AUDIT LOW-MEDIUM: a flat 0.02 is two ticks on a 0.40 risk base but 20
    percentage points of return on a thin 0.10 base."""
    calls = _option_frame([
        {"contractSymbol": "C_103", "strike": 103.0, "bid": 9.90, "ask": 10.10, "lastPrice": 10.0},
        {"contractSymbol": "C_105", "strike": 105.0, "bid": 8.00, "ask": 8.20, "lastPrice": 8.1},
    ])  # call vertical = 1.90; width 2.00 -> valid
    puts = _option_frame([
        {"contractSymbol": "P_97", "strike": 97.0, "bid": 0.01, "ask": 0.03, "lastPrice": 0.02},
        {"contractSymbol": "P_95", "strike": 95.0, "bid": 0.02, "ask": 0.05, "lastPrice": 0.035},
    ])  # put vertical = 0.02 - 0.035 = -0.015 -> a small negative violation
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=112.0)
    base_ctx = {"front_expiry": "2026-05-01",
                "short_call_strike": 103.0, "long_call_strike": 105.0,
                "short_put_strike": 97.0, "long_put_strike": 95.0}

    # Fat risk base: 0.015 is inside the flat tolerance -> repaired and booked.
    fat = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor", earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 5, 1), context={**base_ctx, "max_loss_per_unit": 0.90},
    )
    assert fat["mid"] is not None

    # Thin risk base: 5% of 0.10 = 0.005, so the same 0.015 error is now too
    # large a slice of the return to repair silently.
    thin = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor", earnings_date=date(2026, 4, 30),
        as_of_date=date(2026, 5, 1), context={**base_ctx, "max_loss_per_unit": 0.10},
    )
    assert thin["mid"] is None
    assert thin["reason"] == "condor_quote_outside_no_arbitrage_bounds"


def test_condor_repair_budget_is_total_not_per_leg(monkeypatch) -> None:
    """RE-AUDIT LOW: CONDOR_QUOTE_REPAIR_MAX_RETURN_IMPACT asserts a bound on the
    return error introduced by repair. Applied PER LEG, two verticals violating
    in the same direction each just under tolerance manufactured ~2x that — the
    auditor booked a +400% maximum win on a position whose true value was
    negative. The budget must cover the whole structure."""
    # Each vertical sits at -0.0195: individually under the 0.02 tolerance,
    # together 0.039 — over it.
    calls = _option_frame([
        {"contractSymbol": "C_103", "strike": 103.0, "bid": 0.10, "ask": 0.12, "lastPrice": 0.11},
        {"contractSymbol": "C_105", "strike": 105.0, "bid": 0.12, "ask": 0.139, "lastPrice": 0.1295},
    ])
    puts = _option_frame([
        {"contractSymbol": "P_97", "strike": 97.0, "bid": 0.10, "ask": 0.12, "lastPrice": 0.11},
        {"contractSymbol": "P_95", "strike": 95.0, "bid": 0.12, "ask": 0.139, "lastPrice": 0.1295},
    ])
    _install_fake_option_chain(monkeypatch, calls=calls, puts=puts, spot=100.0)
    quote = forward_loop.fetch_structure_quote(
        symbol="AAPL", structure="iron_condor",
        earnings_date=date(2026, 4, 30), as_of_date=date(2026, 5, 1),
        context={"front_expiry": "2026-05-01",
                 "short_call_strike": 103.0, "long_call_strike": 105.0,
                 "short_put_strike": 97.0, "long_put_strike": 95.0,
                 "max_loss_per_unit": 0.40},
    )
    assert quote["mid"] is None, "two sub-tolerance violations must not sum past the budget"
    prov = quote["bid_ask_mid"]["provenance"]
    assert quote["reason"] == "condor_quote_outside_no_arbitrage_bounds"
    assert prov["total_violation"] > prov["repair_tolerance"]
    # Each leg alone was inside tolerance — the TOTAL is what refused it.
    assert all(v <= prov["repair_tolerance"] for v in prov["leg_violations"].values())
