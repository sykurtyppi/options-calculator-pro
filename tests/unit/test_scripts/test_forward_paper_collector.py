"""Unit tests for the forward paper-trade collector (pure logic, no network).

Exercises run_collection() with injected screen/exit-pricer fakes: entry on the
exact T-3 day, idempotency, T-0 exit resolution + P&L math, and the EXIT_MISSING
path. The live yfinance wrappers are not exercised here.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.forward_paper_collector import (
    load_log,
    realized,
    run_collection,
    _trade_id,
)

ENTRY_DAY = date(2026, 1, 20)
EVENT_DAY = date(2026, 1, 23)  # 3 business days after ENTRY_DAY


def _qualify_row(symbol, event_date, t3, debit=2.00):
    return {
        "symbol": symbol, "earnings_date": event_date, "t3_entry_date": t3,
        "status": "QUALIFY", "entry_debit_mid": debit, "front_expiry": "2026-01-30",
        "call_strike": 105.0, "put_strike": 95.0, "avg_sp_pct": 2.1,
        "call_OI": 500, "put_OI": 400,
    }


def _no_new(universe, weeks, today):
    return []


def test_realized_pnl_math():
    assert realized(2.00, 2.60) == {"pnl_per_contract": 60.0, "net_return_pct": 30.0}
    assert realized(2.00, 1.50) == {"pnl_per_contract": -50.0, "net_return_pct": -25.0}
    assert realized(0.0, 1.0)["net_return_pct"] == 0.0  # no divide-by-zero


def test_entry_only_on_exact_t3_day(tmp_path: Path):
    """Opens a position only when today == t3_entry_date; future candidates are skipped."""
    def screen(universe, weeks, today):
        return [
            _qualify_row("AAPL", EVENT_DAY, today),                       # T-3 == today
            _qualify_row("MSFT", date(2026, 2, 10), date(2026, 2, 5)),    # T-3 in the future
        ]
    log_path = tmp_path / "log.csv"
    stats = run_collection(universe=["AAPL", "MSFT"], weeks=6, today=ENTRY_DAY,
                           log_path=log_path, screen_fn=screen, exit_pricer_fn=lambda p: None)
    assert stats["opened"] == 1
    log = load_log(log_path)
    assert _trade_id("AAPL", EVENT_DAY) in log
    assert _trade_id("MSFT", date(2026, 2, 10)) not in log


def test_non_qualifying_candidates_not_recorded(tmp_path: Path):
    def screen(universe, weeks, today):
        r = _qualify_row("AAPL", EVENT_DAY, today)
        r["status"] = "MARGINAL"  # not a paper trade
        return [r]
    log_path = tmp_path / "log.csv"
    stats = run_collection(universe=["AAPL"], weeks=6, today=ENTRY_DAY,
                           log_path=log_path, screen_fn=screen, exit_pricer_fn=lambda p: None)
    assert stats["opened"] == 0
    assert load_log(log_path) == {}


def test_entry_is_idempotent(tmp_path: Path):
    def screen(universe, weeks, today):
        return [_qualify_row("AAPL", EVENT_DAY, today)]
    log_path = tmp_path / "log.csv"
    run_collection(universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
                   screen_fn=screen, exit_pricer_fn=lambda p: None)
    again = run_collection(universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
                           screen_fn=screen, exit_pricer_fn=lambda p: None)
    assert again["opened"] == 0
    assert len(load_log(log_path)) == 1


def test_exit_resolves_at_t0_with_pnl(tmp_path: Path):
    log_path = tmp_path / "log.csv"
    run_collection(universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
                   screen_fn=lambda u, w, t: [_qualify_row("AAPL", EVENT_DAY, t)],
                   exit_pricer_fn=lambda p: None)
    stats = run_collection(universe=["AAPL"], weeks=6, today=EVENT_DAY, log_path=log_path,
                           screen_fn=_no_new, exit_pricer_fn=lambda p: 2.60)
    assert stats["closed"] == 1
    row = load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]
    assert row["status"] == "CLOSED"
    assert float(row["net_return_pct"]) == 30.0
    assert float(row["pnl_per_contract"]) == 60.0
    assert row["exit_date"] == EVENT_DAY.isoformat()


def test_open_position_carried_before_t0(tmp_path: Path):
    log_path = tmp_path / "log.csv"
    run_collection(universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
                   screen_fn=lambda u, w, t: [_qualify_row("AAPL", EVENT_DAY, t)],
                   exit_pricer_fn=lambda p: None)
    mid_day = date(2026, 1, 21)  # before T-0
    stats = run_collection(universe=["AAPL"], weeks=6, today=mid_day, log_path=log_path,
                           screen_fn=_no_new, exit_pricer_fn=lambda p: 9.99)
    assert stats["closed"] == 0 and stats["open_carried"] == 1
    assert load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]["status"] == "OPEN"


def test_missing_exit_quote_flagged(tmp_path: Path):
    log_path = tmp_path / "log.csv"
    run_collection(universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
                   screen_fn=lambda u, w, t: [_qualify_row("AAPL", EVENT_DAY, t)],
                   exit_pricer_fn=lambda p: None)
    stats = run_collection(universe=["AAPL"], weeks=6, today=EVENT_DAY, log_path=log_path,
                           screen_fn=_no_new, exit_pricer_fn=lambda p: None)  # no quote
    assert stats["exit_missing"] == 1
    row = load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]
    assert row["status"] == "EXIT_MISSING"
    assert row["net_return_pct"] == ""  # excluded from realized P&L


def test_exit_is_idempotent(tmp_path: Path):
    log_path = tmp_path / "log.csv"
    run_collection(universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
                   screen_fn=lambda u, w, t: [_qualify_row("AAPL", EVENT_DAY, t)],
                   exit_pricer_fn=lambda p: None)
    run_collection(universe=["AAPL"], weeks=6, today=EVENT_DAY, log_path=log_path,
                   screen_fn=_no_new, exit_pricer_fn=lambda p: 2.60)
    again = run_collection(universe=["AAPL"], weeks=6, today=EVENT_DAY, log_path=log_path,
                           screen_fn=_no_new, exit_pricer_fn=lambda p: 5.00)  # would change pnl
    assert again["closed"] == 0
    row = load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]
    assert float(row["net_return_pct"]) == 30.0  # unchanged from first close
