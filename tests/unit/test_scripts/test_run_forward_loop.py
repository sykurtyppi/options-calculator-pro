from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd

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
