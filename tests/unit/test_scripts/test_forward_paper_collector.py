"""Unit tests for the forward paper-trade collector (pure logic, no network).

Exercises run_collection() with injected screen/exit-pricer fakes: entry within
the [T-3, event) window (incl. catch-up after a missed run), idempotency, T-0
exit resolution + P&L math, the late-exit exclusion, and EXIT_MISSING. The live
yfinance wrappers are not exercised here.
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


def _qualify_row(symbol, event_date, t3, debit=2.00, nbr=1.52, crush_prob=0.73,
                 crush_gate="PASS"):
    return {
        "symbol": symbol, "earnings_date": event_date, "t3_entry_date": t3,
        "status": "QUALIFY", "entry_debit_mid": debit, "front_expiry": "2026-01-30",
        "call_strike": 105.0, "put_strike": 95.0, "avg_sp_pct": 2.1,
        "call_OI": 500, "put_OI": 400,
        "nbr": nbr, "crush_prob": crush_prob, "crush_gate": crush_gate,
    }


def _no_new(universe, weeks, today):
    return []


def test_realized_pnl_math():
    # net_return_pct is a FRACTION (0.30 = +30%), matching the tracker's derived
    # rows and every `.2%` report — NOT a percent (30.0), which was a 100x mismatch.
    assert realized(2.00, 2.60) == {"pnl_per_contract": 60.0, "net_return_pct": 0.3}
    assert realized(2.00, 1.50) == {"pnl_per_contract": -50.0, "net_return_pct": -0.25}
    assert realized(0.0, 1.0)["net_return_pct"] == 0.0  # no divide-by-zero


def test_entry_skips_future_t3_candidates(tmp_path: Path):
    """Enters within the [T-3, event) window, but NOT before T-3 (future entries)."""
    def screen(universe, weeks, today):
        return [
            _qualify_row("AAPL", EVENT_DAY, today),                       # T-3 == today → enter
            _qualify_row("MSFT", date(2026, 2, 10), date(2026, 2, 5)),    # T-3 in the future → skip
        ]
    log_path = tmp_path / "log.csv"
    stats = run_collection(universe=["AAPL", "MSFT"], weeks=6, today=ENTRY_DAY,
                           log_path=log_path, screen_fn=screen, exit_pricer_fn=lambda p: None)
    assert stats["opened"] == 1
    log = load_log(log_path)
    assert _trade_id("AAPL", EVENT_DAY) in log
    assert _trade_id("MSFT", date(2026, 2, 10)) not in log


def test_entry_catches_up_after_missed_t3(tmp_path: Path):
    """A missed exact-T-3 run must NOT drop the trade: entering the day after T-3
    (still before the event) opens it, tagged with the real (late) trade_date."""
    missed_day = date(2026, 1, 21)  # T-2: one day after t3, still before EVENT_DAY
    def screen(universe, weeks, today):
        # t3 is in the PAST (ENTRY_DAY) but the event hasn't happened yet.
        return [_qualify_row("AAPL", EVENT_DAY, ENTRY_DAY)]
    log_path = tmp_path / "log.csv"
    stats = run_collection(universe=["AAPL"], weeks=6, today=missed_day,
                           log_path=log_path, screen_fn=screen, exit_pricer_fn=lambda p: None)
    assert stats["opened"] == 1
    row = load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]
    assert row["status"] == "OPEN"
    assert row["trade_date"] == missed_day.isoformat()          # real (late) entry day
    assert row["t3_entry_date"] == ENTRY_DAY.isoformat()        # intended day, still visible


def test_late_exit_excluded_as_exit_late(tmp_path: Path):
    """A run that lands AFTER the event (missed T-0) must not price post-earnings
    into realized P&L — it's flagged EXIT_LATE and excluded, like EXIT_MISSING."""
    log_path = tmp_path / "log.csv"
    run_collection(universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
                   screen_fn=lambda u, w, t: [_qualify_row("AAPL", EVENT_DAY, t)],
                   exit_pricer_fn=lambda p: None)
    after_event = date(2026, 1, 26)  # T+ (past EVENT_DAY)
    stats = run_collection(universe=["AAPL"], weeks=6, today=after_event, log_path=log_path,
                           screen_fn=_no_new, exit_pricer_fn=lambda p: 9.99)  # would-be post-print quote
    assert stats["closed"] == 0 and stats["exit_late"] == 1
    row = load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]
    assert row["status"] == "EXIT_LATE"
    assert row["net_return_pct"] == ""       # excluded from realized P&L
    assert row["exit_debit_mid"] == ""       # no post-print price recorded


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
    assert float(row["net_return_pct"]) == 0.3
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
    assert float(row["net_return_pct"]) == 0.3  # unchanged from first close


# ── Crush gate ────────────────────────────────────────────────────────────────


def test_crush_gate_fields_persisted_on_entry(tmp_path: Path):
    """NBR, crush_prob, and crush_gate are captured at entry and survive round-trip."""
    log_path = tmp_path / "log.csv"
    run_collection(
        universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
        screen_fn=lambda u, w, t: [_qualify_row("AAPL", EVENT_DAY, t,
                                                nbr=1.52, crush_prob=0.73, crush_gate="PASS")],
        exit_pricer_fn=lambda p: None,
    )
    row = load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]
    assert row["nbr"] == "1.52", f"nbr not persisted: {row}"
    assert row["crush_prob"] == "0.73", f"crush_prob not persisted: {row}"
    assert row["crush_gate"] == "PASS", f"crush_gate not persisted: {row}"


def test_crush_gate_fail_also_recorded(tmp_path: Path):
    """Failing the gate is advisory — trade is still recorded with crush_gate=FAIL."""
    log_path = tmp_path / "log.csv"
    run_collection(
        universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
        screen_fn=lambda u, w, t: [_qualify_row("AAPL", EVENT_DAY, t,
                                                nbr=1.31, crush_prob=0.38, crush_gate="FAIL")],
        exit_pricer_fn=lambda p: None,
    )
    row = load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]
    assert row["crush_gate"] == "FAIL"
    assert row["status"] == "OPEN"  # still recorded — gate is advisory


def test_crush_gate_survives_close(tmp_path: Path):
    """Crush gate fields survive the exit pass (not overwritten)."""
    log_path = tmp_path / "log.csv"
    run_collection(
        universe=["AAPL"], weeks=6, today=ENTRY_DAY, log_path=log_path,
        screen_fn=lambda u, w, t: [_qualify_row("AAPL", EVENT_DAY, t,
                                                nbr=1.45, crush_prob=0.61, crush_gate="PASS")],
        exit_pricer_fn=lambda p: None,
    )
    run_collection(
        universe=["AAPL"], weeks=6, today=EVENT_DAY, log_path=log_path,
        screen_fn=_no_new, exit_pricer_fn=lambda p: 2.60,
    )
    row = load_log(log_path)[_trade_id("AAPL", EVENT_DAY)]
    assert row["status"] == "CLOSED"
    assert row["crush_gate"] == "PASS"
    assert row["nbr"] == "1.45"
    assert float(row["net_return_pct"]) == 0.3
