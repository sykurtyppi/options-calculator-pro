from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.backtest_calendar_exact import (
    STATUS_EXCLUDED_NEGATIVE_DEBIT,
    STATUS_EXCLUDED_STRUCTURAL_INVALIDITY,
    STATUS_MISSING_EXIT_QUOTE,
    BacktestConfig,
    BacktestInvariantError,
    ContractIdentity,
    ContractSnapshot,
    EntrySelection,
    TradeOutcome,
    build_contract_snapshot,
    choose_same_strike_pair,
    lookup_exact_contract,
    realize_trade,
    select_entry_contracts,
    validate_trade_outcome,
)


def _chain(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    return frame


def _row(
    *,
    trade_date: str = "2025-01-02",
    expiry: str,
    strike: float,
    option_id: int,
    option_symbol: str,
    bid: float,
    ask: float,
    mid: float,
    spread: float,
    spread_pct: float,
    moneyness_pct: float,
    underlying_price: float = 100.0,
    option_type: str = "C",
) -> dict:
    return {
        "trade_date": trade_date,
        "expiry": expiry,
        "call_put": option_type,
        "strike": strike,
        "option_id": option_id,
        "option_symbol": option_symbol,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "spread_pct": spread_pct,
        "moneyness_pct": moneyness_pct,
        "volume": 10,
        "open_interest": 100,
        "underlying_price": underlying_price,
        "quote_quality_flag": "ok",
    }


def _config() -> BacktestConfig:
    return BacktestConfig(
        db_path=Path("/tmp/unused.db"),
        parquet_root=Path("/tmp/unused"),
        output_dir=Path("/tmp"),
    )


def _selection() -> EntrySelection:
    front = ContractSnapshot(
        identity=ContractIdentity(
            symbol="TEST",
            option_type="C",
            strike=100.0,
            expiry="2025-01-17",
            option_symbol="TEST 250117C00100000",
            option_id=101,
        ),
        quote_date="2025-01-13",
        bid=4.8,
        ask=5.2,
        mid=5.0,
        spread=0.4,
        spread_pct=0.08,
        volume=20,
        open_interest=200,
        moneyness_pct=0.1,
        underlying_price=100.0,
    )
    back = ContractSnapshot(
        identity=ContractIdentity(
            symbol="TEST",
            option_type="C",
            strike=100.0,
            expiry="2025-02-21",
            option_symbol="TEST 250221C00100000",
            option_id=202,
        ),
        quote_date="2025-01-13",
        bid=6.8,
        ask=7.2,
        mid=7.0,
        spread=0.4,
        spread_pct=0.06,
        volume=20,
        open_interest=200,
        moneyness_pct=0.1,
        underlying_price=100.0,
    )
    return EntrySelection(
        symbol="TEST",
        event_date="2025-01-15",
        release_timing="AMC",
        signal_nbr=1.5,
        threshold_value=1.4,
        entry_date="2025-01-13",
        exit_date="2025-01-16",
        front_expiry="2025-01-17",
        back_expiry="2025-02-21",
        front=front,
        back=back,
        entry_debit_mid=2.0,
        entry_debit_adjusted=2.4,
        entry_penalty=0.4,
        debit_capital=400.0,
        actual_entry_cashflow=400.0,
        entry_offset_bdays=2,
        exit_offset_bdays=1,
    )


def test_choose_same_strike_pair_enforces_common_strike():
    front = _chain(
        [
            _row(expiry="2025-01-17", strike=100.0, option_id=1, option_symbol="F100", bid=4.9, ask=5.1, mid=5.0, spread=0.2, spread_pct=0.04, moneyness_pct=0.1),
            _row(expiry="2025-01-17", strike=101.0, option_id=2, option_symbol="F101", bid=4.7, ask=4.9, mid=4.8, spread=0.2, spread_pct=0.04, moneyness_pct=0.6),
        ]
    )
    back = _chain(
        [
            _row(expiry="2025-02-21", strike=100.0, option_id=3, option_symbol="B100", bid=6.9, ask=7.1, mid=7.0, spread=0.2, spread_pct=0.03, moneyness_pct=0.2),
            _row(expiry="2025-02-21", strike=102.0, option_id=4, option_symbol="B102", bid=6.4, ask=6.6, mid=6.5, spread=0.2, spread_pct=0.03, moneyness_pct=0.8),
        ]
    )

    front_row, back_row, reason = choose_same_strike_pair(front, back)

    assert reason == ""
    assert float(front_row["strike"]) == 100.0
    assert float(back_row["strike"]) == 100.0


def test_lookup_exact_contract_uses_identity_not_fresh_atm():
    chain = _chain(
        [
            _row(expiry="2025-01-17", strike=100.0, option_id=101, option_symbol="TEST 250117C00100000", bid=4.5, ask=4.7, mid=4.6, spread=0.2, spread_pct=0.04, moneyness_pct=5.0),
            _row(expiry="2025-01-17", strike=105.0, option_id=105, option_symbol="TEST 250117C00105000", bid=1.0, ask=1.2, mid=1.1, spread=0.2, spread_pct=0.18, moneyness_pct=0.1),
        ]
    )
    identity = ContractIdentity(
        symbol="TEST",
        option_type="C",
        strike=100.0,
        expiry="2025-01-17",
        option_symbol="TEST 250117C00100000",
        option_id=101,
    )

    exact = lookup_exact_contract(chain, identity)

    assert exact is not None
    assert float(exact["strike"]) == 100.0
    assert int(exact["option_id"]) == 101


def test_missing_exit_quote_is_flagged(monkeypatch: pytest.MonkeyPatch):
    selection = _selection()
    config = _config()

    def fake_load_day_chain(*args, **kwargs):
        return None

    monkeypatch.setattr("scripts.backtest_calendar_exact.load_day_chain", fake_load_day_chain)

    outcome = realize_trade(selection, config, chain_cache={})

    assert outcome.status == STATUS_MISSING_EXIT_QUOTE
    assert outcome.exact_exit_quote_found is False


def test_select_entry_contracts_excludes_negative_debit(monkeypatch: pytest.MonkeyPatch):
    config = _config()
    candidate = pd.Series(
        {
            "symbol": "TEST",
            "event_date": pd.Timestamp("2025-01-15"),
            "release_timing": "AMC",
            "signal_nbr": 1.5,
            "threshold_value": 1.4,
            "front_expiry": pd.Timestamp("2025-01-17"),
            "back_expiry": pd.Timestamp("2025-02-21"),
        }
    )
    entry_chain = _chain(
        [
            _row(trade_date="2025-01-13", expiry="2025-01-17", strike=100.0, option_id=101, option_symbol="TEST 250117C00100000", bid=4.9, ask=5.1, mid=5.0, spread=0.2, spread_pct=0.04, moneyness_pct=0.1),
            _row(trade_date="2025-01-13", expiry="2025-02-21", strike=100.0, option_id=202, option_symbol="TEST 250221C00100000", bid=3.9, ask=4.1, mid=4.0, spread=0.2, spread_pct=0.05, moneyness_pct=0.1),
        ]
    )

    def fake_load_day_chain(*args, **kwargs):
        return entry_chain

    monkeypatch.setattr("scripts.backtest_calendar_exact.load_day_chain", fake_load_day_chain)

    selection, exclusion = select_entry_contracts(candidate, config, chain_cache={}, logger=None)  # type: ignore[arg-type]

    assert selection is None
    assert exclusion is not None
    assert exclusion.status == STATUS_EXCLUDED_NEGATIVE_DEBIT


def test_select_entry_contracts_excludes_no_same_strike(monkeypatch: pytest.MonkeyPatch):
    config = _config()
    candidate = pd.Series(
        {
            "symbol": "TEST",
            "event_date": pd.Timestamp("2025-01-15"),
            "release_timing": "AMC",
            "signal_nbr": 1.5,
            "threshold_value": 1.4,
            "front_expiry": pd.Timestamp("2025-01-17"),
            "back_expiry": pd.Timestamp("2025-02-21"),
        }
    )
    entry_chain = _chain(
        [
            _row(trade_date="2025-01-13", expiry="2025-01-17", strike=100.0, option_id=101, option_symbol="TEST 250117C00100000", bid=4.9, ask=5.1, mid=5.0, spread=0.2, spread_pct=0.04, moneyness_pct=0.1),
            _row(trade_date="2025-01-13", expiry="2025-02-21", strike=101.0, option_id=202, option_symbol="TEST 250221C00101000", bid=6.9, ask=7.1, mid=7.0, spread=0.2, spread_pct=0.03, moneyness_pct=0.1),
        ]
    )

    def fake_load_day_chain(*args, **kwargs):
        return entry_chain

    monkeypatch.setattr("scripts.backtest_calendar_exact.load_day_chain", fake_load_day_chain)

    selection, exclusion = select_entry_contracts(candidate, config, chain_cache={}, logger=None)  # type: ignore[arg-type]

    assert selection is None
    assert exclusion is not None
    assert exclusion.status == STATUS_EXCLUDED_STRUCTURAL_INVALIDITY


def test_validate_trade_outcome_enforces_timing_offsets():
    config = _config()
    valid = TradeOutcome(
        status="REALIZED",
        exclusion_reason="",
        symbol="TEST",
        event_date="2025-01-15",
        release_timing="AMC",
        signal_nbr=1.5,
        threshold_value=1.4,
        option_type="C",
        screen_date="2025-01-14",
        screen_front_oi=200,
        screen_back_oi=180,
        screen_front_spread_pct=0.04,
        screen_back_spread_pct=0.03,
        entry_date="2025-01-13",
        exit_date="2025-01-16",
        entry_offset_bdays=2,
        exit_offset_bdays=1,
        front_expiry="2025-01-17",
        back_expiry="2025-02-21",
        front_option_symbol="TEST 250117C00100000",
        back_option_symbol="TEST 250221C00100000",
        front_option_id=101,
        back_option_id=202,
        strike=100.0,
        front_bid_entry=4.8,
        front_ask_entry=5.2,
        front_mid_entry=5.0,
        back_bid_entry=6.8,
        back_ask_entry=7.2,
        back_mid_entry=7.0,
        entry_front_spread=0.4,
        entry_back_spread=0.4,
        front_bid_exit=4.0,
        front_ask_exit=4.2,
        front_mid_exit=4.1,
        back_bid_exit=6.0,
        back_ask_exit=6.2,
        back_mid_exit=6.1,
        exit_front_spread=0.2,
        exit_back_spread=0.2,
        entry_debit_mid=2.0,
        entry_debit_adjusted=2.4,
        exit_value_mid=2.0,
        exit_value_adjusted=1.8,
        realized_pnl_mid=0.0,
        realized_pnl_adjusted=-60.0,
        return_on_capital_mid=0.0,
        return_on_capital_adjusted=-0.15,
        debit_capital=400.0,
        actual_entry_cashflow=400.0,
        exact_exit_quote_found=True,
        front_exact_match=True,
        back_exact_match=True,
        timing_validated=True,
    )

    validate_trade_outcome(valid, config)

    invalid = valid
    invalid.entry_date = "2025-01-14"
    with pytest.raises(BacktestInvariantError):
        validate_trade_outcome(invalid, config)


def test_realize_trade_fails_on_exit_identity_mismatch(monkeypatch: pytest.MonkeyPatch):
    selection = _selection()
    config = _config()
    exit_chain = _chain(
        [
            _row(trade_date="2025-01-16", expiry="2025-01-17", strike=100.0, option_id=101, option_symbol="TEST 250117C00100000", bid=4.0, ask=4.2, mid=4.1, spread=0.2, spread_pct=0.05, moneyness_pct=0.2),
            _row(trade_date="2025-01-16", expiry="2025-02-21", strike=100.0, option_id=202, option_symbol="TEST 250221C00100000", bid=6.0, ask=6.2, mid=6.1, spread=0.2, spread_pct=0.03, moneyness_pct=0.2),
        ]
    )

    def fake_load_day_chain(*args, **kwargs):
        return exit_chain

    def fake_lookup(chain, identity):
        if identity.expiry == "2025-01-17":
            wrong = exit_chain.iloc[0].copy()
            wrong["strike"] = 101.0
            return wrong
        return exit_chain.iloc[1]

    monkeypatch.setattr("scripts.backtest_calendar_exact.load_day_chain", fake_load_day_chain)
    monkeypatch.setattr("scripts.backtest_calendar_exact.lookup_exact_contract", fake_lookup)

    with pytest.raises(BacktestInvariantError):
        realize_trade(selection, config, chain_cache={})
