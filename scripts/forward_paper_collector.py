#!/usr/bin/env python3
"""Forward paper-trade collector for the validated pre-earnings OTM-strangle pocket.

Closes the gap between forward_screener.py (which only *identifies* T-3 entries) and
the forward paper tracker (which *consumes* a trade log). Run daily; it:

  1. ENTRY pass — on a symbol's T-3 business day, if its AMC 3% OTM strangle QUALIFIES
     (Fingerprint C: avg spread <= 3%, call & put OI >= 100), record an OPEN paper
     position capturing the exact contract identity + entry mid debit.
  2. EXIT  pass — on the earnings date (T-0, before the AMC print), reprice the SAME
     contracts and record realized P&L; mark CLOSED. Missing exact quotes are flagged
     EXIT_MISSING and excluded from realized P&L (same discipline as the backtest study).

The log is append/update-idempotent and column-compatible with
institutional_backfill.py --forward-trade-log-path, so accumulated rows feed straight
into `--forward-paper-tracker`.

Why this exists: the backtested edge (AMC T-3/T-0 3% OTM strangle) rests on only 42
trades and decays out of sample. The only way to learn whether it is real or decaying
is to accrue genuine forward samples on the *correct* config — which this does.

Usage:
    python scripts/forward_paper_collector.py                 # daily entry+exit pass
    python scripts/forward_paper_collector.py --weeks 8
    python scripts/forward_paper_collector.py --log-path PATH
    python scripts/forward_paper_collector.py --selftest      # offline logic test
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_LOG = _ROOT / "exports" / "reports" / "forward_paper_trades.csv"

# Persisted columns. trade_date/debit_per_contract/net_return_pct are the
# tracker-consumed fields; the rest are identity + diagnostics.
# nbr/crush_prob/crush_gate were added 2026-06-04 to test the NBR >= 1.40 gate
# hypothesis from the walk-forward expansion analysis. Both gated and ungated
# trades are recorded — the gate is ADVISORY so forward data can prove/disprove it.
FIELDNAMES = [
    "trade_id", "symbol", "trade_date", "event_date", "t3_entry_date",
    "front_expiry", "call_strike", "put_strike",
    "entry_debit_mid", "debit_per_contract", "contracts",
    "avg_sp_pct", "call_OI", "put_OI",
    "nbr", "crush_prob", "crush_gate",
    "exit_date", "exit_debit_mid", "pnl_per_contract", "net_return_pct",
    "status",  # OPEN | CLOSED | EXIT_MISSING
]


def _trade_id(symbol: str, event_date: Any) -> str:
    return f"{str(symbol).upper()}|{event_date}"


def load_log(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["trade_id"]] = row
    return out


def save_log(path: Path, log: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for tid in sorted(log):
            w.writerow({k: log[tid].get(k, "") for k in FIELDNAMES})
    tmp.replace(path)


def realized(entry_debit: float, exit_debit: float) -> Dict[str, float]:
    """Long-strangle P&L: buy at entry_debit, sell at exit_debit (per share)."""
    pnl_per_contract = (exit_debit - entry_debit) * 100.0
    net_return_pct = ((exit_debit - entry_debit) / entry_debit * 100.0) if entry_debit else 0.0
    return {"pnl_per_contract": round(pnl_per_contract, 2),
            "net_return_pct": round(net_return_pct, 4)}


def run_collection(
    *,
    universe: List[str],
    weeks: int,
    today: date,
    log_path: Path,
    screen_fn: Callable[[List[str], int, date], "Any"],
    exit_pricer_fn: Callable[[Dict[str, Any]], Optional[float]],
    dry_run: bool = False,
) -> Dict[str, int]:
    """Run one entry+exit pass. screen_fn/exit_pricer_fn are injected for testability."""
    log = load_log(log_path)
    stats = {"opened": 0, "closed": 0, "exit_missing": 0, "open_carried": 0}

    # ── ENTRY pass: open positions whose T-3 entry day is today and that QUALIFY ──
    screened = screen_fn(universe, weeks, today)
    rows = screened.to_dict("records") if hasattr(screened, "to_dict") else list(screened)
    for r in rows:
        if str(r.get("status")) != "QUALIFY":
            continue
        if r.get("t3_entry_date") != today:   # only enter on the exact T-3 business day
            continue
        tid = _trade_id(r["symbol"], r["earnings_date"])
        if tid in log:                         # idempotent — already recorded
            continue
        entry_debit = float(r["entry_debit_mid"])
        log[tid] = {
            "trade_id": tid, "symbol": str(r["symbol"]).upper(),
            "trade_date": today.isoformat(), "event_date": str(r["earnings_date"]),
            "t3_entry_date": str(r["t3_entry_date"]),
            "front_expiry": r.get("front_expiry", ""),
            "call_strike": r.get("call_strike", ""), "put_strike": r.get("put_strike", ""),
            "entry_debit_mid": entry_debit, "debit_per_contract": round(entry_debit * 100.0, 2),
            "contracts": 1, "avg_sp_pct": r.get("avg_sp_pct", ""),
            "call_OI": r.get("call_OI", ""), "put_OI": r.get("put_OI", ""),
            "nbr": r.get("nbr", ""), "crush_prob": r.get("crush_prob", ""),
            "crush_gate": r.get("crush_gate", "NO_DATA"),
            "exit_date": "", "exit_debit_mid": "", "pnl_per_contract": "",
            "net_return_pct": "", "status": "OPEN",
        }
        stats["opened"] += 1

    # ── EXIT pass: resolve OPEN positions whose earnings date (T-0) has arrived ──
    for tid, pos in log.items():
        if pos.get("status") != "OPEN":
            continue
        try:
            event_date = datetime.fromisoformat(str(pos["event_date"])).date()
        except ValueError:
            continue
        if today < event_date:                 # not yet at T-0
            stats["open_carried"] += 1
            continue
        exit_debit = exit_pricer_fn(pos)
        if exit_debit is None:                  # exact contracts not quotable
            pos["status"] = "EXIT_MISSING"
            pos["exit_date"] = today.isoformat()
            stats["exit_missing"] += 1
            continue
        pos.update({
            "exit_date": today.isoformat(), "exit_debit_mid": round(float(exit_debit), 4),
            **realized(float(pos["entry_debit_mid"]), float(exit_debit)),
            "status": "CLOSED",
        })
        stats["closed"] += 1

    if not dry_run:
        save_log(log_path, log)
    return stats


# ── Live (network) screen + exit pricer — thin wrappers over forward_screener/yfinance ──

def _live_screen(universe: List[str], weeks: int, today: date):
    from scripts.forward_screener import run_screener
    return run_screener(universe, weeks, today)


def _live_exit_pricer(pos: Dict[str, Any]) -> Optional[float]:
    """Reprice the exact entry contracts (front_expiry, call_strike, put_strike) at mid."""
    import yfinance as yf
    try:
        t = yf.Ticker(str(pos["symbol"]))
        exp = str(pos["front_expiry"])
        if not exp or exp not in (t.options or ()):
            return None
        ch = t.option_chain(exp)
        cs, ps = float(pos["call_strike"]), float(pos["put_strike"])
        cm = ch.calls.loc[ch.calls["strike"] == cs]
        pm = ch.puts.loc[ch.puts["strike"] == ps]
        if cm.empty or pm.empty:
            return None
        c = (float(cm.iloc[0]["bid"]) + float(cm.iloc[0]["ask"])) / 2.0
        p = (float(pm.iloc[0]["bid"]) + float(pm.iloc[0]["ask"])) / 2.0
        return c + p if (c > 0 and p > 0) else None
    except Exception:
        return None


def main() -> int:
    from scripts.forward_screener import DEFAULT_UNIVERSE
    ap = argparse.ArgumentParser(description="Forward paper-trade collector (AMC T-3/T-0 OTM strangle)")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--weeks", type=int, default=6)
    ap.add_argument("--log-path", default=str(DEFAULT_LOG))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run offline logic test and exit")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return 0

    today = datetime.now(timezone.utc).date()
    stats = run_collection(
        universe=args.symbols, weeks=args.weeks, today=today,
        log_path=Path(args.log_path), screen_fn=_live_screen,
        exit_pricer_fn=_live_exit_pricer, dry_run=args.dry_run,
    )
    print(f"[{today}] forward paper collector: {stats}")
    print(f"log: {args.log_path}")
    return 0


def _selftest() -> None:
    """Offline logic check (no network): entry idempotency, exit resolution, PnL math."""
    import tempfile
    assert realized(2.00, 2.60) == {"pnl_per_contract": 60.0, "net_return_pct": 30.0}, "PnL math"
    assert realized(2.00, 1.50)["pnl_per_contract"] == -50.0, "loss math"

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "log.csv"
        entry_day = date(2026, 1, 20)
        event_day = date(2026, 1, 23)  # T-3 from entry_day (Tue->Fri, all weekdays)

        def fake_screen(universe, weeks, today):
            # AAPL qualifies and its T-3 is today; MSFT qualifies but T-3 is not today
            return [
                {"symbol": "AAPL", "earnings_date": event_day, "t3_entry_date": today,
                 "status": "QUALIFY", "entry_debit_mid": 2.00, "front_expiry": "2026-01-30",
                 "call_strike": 105.0, "put_strike": 95.0, "avg_sp_pct": 2.1,
                 "call_OI": 500, "put_OI": 400,
                 "nbr": 1.52, "crush_prob": 0.73, "crush_gate": "PASS"},
                {"symbol": "MSFT", "earnings_date": date(2026, 2, 10), "t3_entry_date": date(2026, 2, 5),
                 "status": "QUALIFY", "entry_debit_mid": 3.0, "front_expiry": "2026-02-13",
                 "call_strike": 410.0, "put_strike": 390.0, "avg_sp_pct": 1.5,
                 "call_OI": 900, "put_OI": 800,
                 "nbr": 1.31, "crush_prob": 0.38, "crush_gate": "FAIL"},
            ]

        # Day 1: entry day — opens AAPL only (MSFT's T-3 != today)
        s1 = run_collection(universe=["AAPL", "MSFT"], weeks=6, today=entry_day, log_path=log_path,
                            screen_fn=fake_screen, exit_pricer_fn=lambda pos: None)
        assert s1["opened"] == 1, s1
        # Re-run same day — idempotent, no duplicate
        s1b = run_collection(universe=["AAPL", "MSFT"], weeks=6, today=entry_day, log_path=log_path,
                             screen_fn=fake_screen, exit_pricer_fn=lambda pos: None)
        assert s1b["opened"] == 0, s1b

        # Day 2 (T-0): exit AAPL at 2.60 -> +30%
        def no_new(universe, weeks, today): return []
        s2 = run_collection(universe=["AAPL"], weeks=6, today=event_day, log_path=log_path,
                            screen_fn=no_new, exit_pricer_fn=lambda pos: 2.60)
        assert s2["closed"] == 1, s2
        log = load_log(log_path)
        aapl = log[_trade_id("AAPL", event_day)]
        assert aapl["status"] == "CLOSED" and float(aapl["net_return_pct"]) == 30.0, aapl
        assert aapl["crush_gate"] == "PASS" and float(aapl["nbr"]) == 1.52, f"crush gate: {aapl}"

        # Re-run after close — idempotent (no re-exit)
        s2b = run_collection(universe=["AAPL"], weeks=6, today=event_day, log_path=log_path,
                             screen_fn=no_new, exit_pricer_fn=lambda pos: 2.60)
        assert s2b["closed"] == 0, s2b

        # Missing exit quote -> EXIT_MISSING, excluded from realized
        log2_path = Path(td) / "log2.csv"
        run_collection(universe=["AAPL"], weeks=6, today=entry_day, log_path=log2_path,
                       screen_fn=fake_screen, exit_pricer_fn=lambda pos: None)
        sm = run_collection(universe=["AAPL"], weeks=6, today=event_day, log_path=log2_path,
                            screen_fn=no_new, exit_pricer_fn=lambda pos: None)
        assert sm["exit_missing"] == 1, sm
    print("forward_paper_collector selftest: PASS")


if __name__ == "__main__":
    raise SystemExit(main())
