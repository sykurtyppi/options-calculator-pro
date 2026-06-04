#!/usr/bin/env python3
"""Pre-Earnings OTM Strangle Forward Screener.

Run this each morning to identify qualifying paper-trade candidates.

Usage:
    python scripts/forward_screener.py
    python scripts/forward_screener.py --symbols AMZN AAPL GOOGL
    python scripts/forward_screener.py --weeks 8
    python scripts/forward_screener.py --validate   # test against known historical trades

Qualifying rule (Fingerprint C from structural analysis):
    avg(call_spread_pct, put_spread_pct) <= 3.0%
    call OI >= 100  AND  put OI >= 100
    release_timing == AMC  (after-market close)
    structure: 3% OTM strangle, T-3 entry / T-0 exit
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────────

# Default universe: confirmed backtest names + high-liquidity adjacent names
DEFAULT_UNIVERSE = [
    # Core — confirmed in backtest (spread ≤3%, AMC, profitable)
    "AMZN", "AAPL", "GOOGL", "MSFT", "ADBE", "PANW", "NFLX", "FTNT", "NKE",
    # High-priority additions — deep option markets
    "META", "NVDA", "TSLA", "AVGO", "AMD", "ORCL", "CRM", "INTU", "NOW",
    "QCOM", "INTC", "LRCX", "KLAC", "CDNS", "SNPS", "AMAT",
    "TTD", "ZS", "OKTA", "TEAM", "WDAY", "SNOW", "DDOG", "NET",
    "PYPL", "EBAY", "DIS", "UBER", "COST", "SBUX", "LULU",
]

SPREAD_THRESHOLD_QUALIFY = 0.03   # Fingerprint C — proven edge
SPREAD_THRESHOLD_MARGINAL = 0.05  # Watch list — no paper trade without re-check
OI_MIN = 100
TARGET_OTM_PCT = 0.03             # 3% OTM strangle
NY_TZ = pytz.timezone("America/New_York")

# ── NBR crush-gate threshold ────────────────────────────────────────────────
# Walk-forward analysis (2026-06-04) showed NBR >= 1.40 filters to 85% win rate
# on backtested Fingerprint C trades. The crush classifier (AUC=0.820) uses NBR
# as its dominant feature, so high NBR ≈ "classifier predicts deep crush" ≈
# "term structure steep enough to support the pre-earnings IV expansion trade."
NBR_GATE_THRESHOLD = 1.40

# ── Helpers ───────────────────────────────────────────────────────────────────

def t3_entry_date(event_date: date) -> date:
    """Return the T-3 business-day entry date preceding event_date."""
    d, n = event_date, 0
    while n < 3:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


def get_release_timing(info: dict) -> str:
    ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
    if not ts:
        return "UNKNOWN"
    dt_ny = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(NY_TZ)
    if dt_ny.hour >= 16:
        return "AMC"
    if dt_ny.hour < 12:
        return "BMO"
    return f"INTRADAY ({dt_ny.strftime('%H:%M')} ET)"


def pick_front_expiry_after(expirations: tuple, event_date: date) -> Optional[str]:
    """First expiry strictly after event_date (front weekly — backtest methodology)."""
    for exp_str in sorted(expirations):
        if date.fromisoformat(exp_str) > event_date:
            return exp_str
    return None


def pick_next_monthly_opex(expirations: tuple, event_date: date) -> Optional[str]:
    """Next standard monthly OPEX after event_date.
    Monthly OPEX = third Friday of the month.  We approximate by taking the
    first expiry after event_date whose day-of-month is >= 15 (third Friday
    always falls 15-21).  Falls back to first available if none found.
    """
    for exp_str in sorted(expirations):
        exp = date.fromisoformat(exp_str)
        if exp > event_date and exp.day >= 15:
            return exp_str
    return pick_front_expiry_after(expirations, event_date)


def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0 or bid < 0 or ask < 0:
        return np.nan
    return (ask - bid) / mid


def implied_move_pct(ticker_obj, spot: float, exp_str: str) -> Optional[float]:
    """ATM straddle mid / spot — rough event-implied move estimate."""
    try:
        ch = ticker_obj.option_chain(exp_str)
        calls = ch.calls[ch.calls["bid"] > 0]
        puts  = ch.puts[ch.puts["bid"] > 0]
        atm_c = calls[calls["strike"] >= spot].sort_values("strike")
        atm_p = puts[puts["strike"] <= spot].sort_values("strike", ascending=False)
        if atm_c.empty or atm_p.empty:
            return None
        straddle_mid = (
            (atm_c.iloc[0]["bid"] + atm_c.iloc[0]["ask"]) / 2
            + (atm_p.iloc[0]["bid"] + atm_p.iloc[0]["ask"]) / 2
        )
        return straddle_mid / spot * 100.0
    except Exception:
        return None


def _atm_iv_for_expiry(ticker_obj, spot: float, exp_str: str) -> Optional[float]:
    """ATM implied vol for a given expiry — average of nearest call+put ATM IVs."""
    try:
        ch = ticker_obj.option_chain(exp_str)
        calls = ch.calls[(ch.calls["bid"] > 0) & (ch.calls["impliedVolatility"] > 0)]
        puts = ch.puts[(ch.puts["bid"] > 0) & (ch.puts["impliedVolatility"] > 0)]
        if calls.empty or puts.empty:
            return None
        atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
        atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
        c_iv = float(atm_c.iloc[0]["impliedVolatility"])
        p_iv = float(atm_p.iloc[0]["impliedVolatility"])
        return (c_iv + p_iv) / 2.0
    except Exception:
        return None


def _compute_crush_gate_signals(
    front_iv: Optional[float],
    back_iv: Optional[float],
) -> dict:
    """Compute NBR and crush classifier probability from front/back ATM IVs.

    Returns a dict with keys: nbr, crush_prob, crush_gate.
    crush_gate is 'PASS' if NBR >= NBR_GATE_THRESHOLD, else 'FAIL'.

    The crush classifier (walk-forward AUC=0.820 on 1,143 OOS events) predicts
    which events will see deep IV crush (>40%). The same NBR feature that drives
    this prediction also separates winning from losing pre-earnings expansion
    trades — high NBR means steep term structure means more vol to accumulate.
    """
    result: dict = {"nbr": None, "crush_prob": None, "crush_gate": "NO_DATA"}
    if front_iv is None or back_iv is None or back_iv < 0.01:
        return result

    nbr = front_iv / back_iv
    result["nbr"] = round(nbr, 4)
    result["crush_gate"] = "PASS" if nbr >= NBR_GATE_THRESHOLD else "FAIL"

    # Run the trained crush classifier if available
    try:
        from web.api.edge_engine import _ml_crush_probability
        prob, _mult = _ml_crush_probability(
            near_iv=front_iv,
            near_back_ratio=nbr,
            iv_rv=front_iv * 0.75,  # fallback iv_rv (same as training — RV data empty)
        )
        if prob is not None:
            result["crush_prob"] = round(prob, 4)
    except Exception:
        pass  # model unavailable — NBR gate still works standalone

    return result


def screen_one(sym: str, event_date: date, today: date) -> dict:
    """Return a screening result dict for one symbol / event date."""
    result = {
        "symbol": sym,
        "earnings_date": event_date,
        "t3_entry_date": t3_entry_date(event_date),
        "days_to_entry": (t3_entry_date(event_date) - today).days,
        "release_timing": "UNKNOWN",
        "spot": None,
        "front_expiry": None,
        "monthly_expiry": None,
        "call_strike": None, "put_strike": None,
        "call_sp_pct": None, "put_sp_pct": None, "avg_sp_pct": None,
        "call_OI": None, "put_OI": None,
        "call_IV": None, "put_IV": None,
        "back_atm_iv": None,
        "nbr": None, "crush_prob": None, "crush_gate": "NO_DATA",
        "impl_move_pct": None,
        "entry_debit_mid": None,
        "status": "ERROR",
        "note": "",
    }
    try:
        t = yf.Ticker(sym)
        info = t.info
        result["release_timing"] = get_release_timing(info)

        spot = float(t.fast_info.last_price)
        result["spot"] = round(spot, 2)

        exps = t.options
        exp_str = pick_front_expiry_after(exps, event_date)
        if exp_str is None:
            result["status"] = "NO_EXPIRY"
            return result
        result["front_expiry"] = exp_str
        result["monthly_expiry"] = pick_next_monthly_opex(exps, event_date)

        ch = t.option_chain(exp_str)
        calls = ch.calls[ch.calls["bid"] > 0].copy()
        puts  = ch.puts[ch.puts["bid"] > 0].copy()

        call_otm = calls[calls["strike"] >= spot * 1.01].copy()
        put_otm  = puts[puts["strike"] <= spot * 0.99].copy()
        if call_otm.empty or put_otm.empty:
            result["status"] = "NO_OTM_CHAIN"
            return result

        call_target, put_target = spot * (1 + TARGET_OTM_PCT), spot * (1 - TARGET_OTM_PCT)
        call_otm["dist"] = (call_otm["strike"] - call_target).abs()
        put_otm["dist"]  = (put_otm["strike"] - put_target).abs()
        cr = call_otm.sort_values("dist").iloc[0]
        pr = put_otm.sort_values("dist").iloc[0]

        c_sp = spread_pct(float(cr["bid"]), float(cr["ask"]))
        p_sp = spread_pct(float(pr["bid"]), float(pr["ask"]))
        avg_sp = float(np.nanmean([c_sp, p_sp]))

        result.update({
            "call_strike": float(cr["strike"]),
            "put_strike": float(pr["strike"]),
            "call_sp_pct": round(c_sp * 100, 2) if not np.isnan(c_sp) else None,
            "put_sp_pct": round(p_sp * 100, 2) if not np.isnan(p_sp) else None,
            "avg_sp_pct": round(avg_sp * 100, 2) if not np.isnan(avg_sp) else None,
            "call_OI": int(cr["openInterest"]) if "openInterest" in cr else None,
            "put_OI": int(pr["openInterest"]) if "openInterest" in pr else None,
            "call_IV": round(float(cr["impliedVolatility"]), 3) if "impliedVolatility" in cr else None,
            "put_IV": round(float(pr["impliedVolatility"]), 3) if "impliedVolatility" in pr else None,
            "entry_debit_mid": round(
                (float(cr["bid"]) + float(cr["ask"])) / 2
                + (float(pr["bid"]) + float(pr["ask"])) / 2, 2
            ),
        })

        result["impl_move_pct"] = round(implied_move_pct(t, spot, exp_str) or 0, 1) or None

        # ── Back-month IV + NBR + crush gate ───────────────────────────────
        # Front ATM IV: average the call/put IV from the OTM legs (proxy for
        # ATM since 3% OTM is close). Back ATM IV: look up the monthly expiry.
        front_atm_iv = None
        if result["call_IV"] is not None and result["put_IV"] is not None:
            front_atm_iv = (result["call_IV"] + result["put_IV"]) / 2.0

        back_atm_iv = None
        monthly = result.get("monthly_expiry")
        if monthly and monthly != exp_str:
            back_atm_iv = _atm_iv_for_expiry(t, spot, monthly)
            result["back_atm_iv"] = round(back_atm_iv, 4) if back_atm_iv else None

        gate_signals = _compute_crush_gate_signals(front_atm_iv, back_atm_iv)
        result.update(gate_signals)

        # Filter logic
        oi_ok = (
            result["call_OI"] is not None and result["call_OI"] >= OI_MIN
            and result["put_OI"] is not None and result["put_OI"] >= OI_MIN
        )
        spread_ok = not np.isnan(avg_sp) and avg_sp <= SPREAD_THRESHOLD_QUALIFY
        spread_marginal = not np.isnan(avg_sp) and avg_sp <= SPREAD_THRESHOLD_MARGINAL

        if result["release_timing"] != "AMC":
            result["status"] = "SKIP_BMO"
        elif not oi_ok:
            result["status"] = "FAIL_OI"
            result["note"] = f"call_OI={result['call_OI']}, put_OI={result['put_OI']}"
        elif spread_ok:
            result["status"] = "QUALIFY"
        elif spread_marginal:
            result["status"] = "MARGINAL"
            result["note"] = f"Re-check spread on entry day"
        else:
            result["status"] = "FAIL_SPREAD"

    except Exception as e:
        result["status"] = "ERROR"
        result["note"] = str(e)[:80]

    return result


def run_screener(symbols: list[str], weeks: int, today: date) -> pd.DataFrame:
    cutoff = today + timedelta(weeks=weeks)
    print(f"Fetching earnings dates for {len(symbols)} symbols ({today} → {cutoff})...\n")

    # Step 1: get earnings dates
    candidates: list[tuple[str, date]] = []
    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar
            if not isinstance(cal, dict):
                continue
            dates = cal.get("Earnings Date", [])
            if not isinstance(dates, list):
                dates = [dates]
            for d in dates:
                if d and today <= d <= cutoff:
                    candidates.append((sym, d))
        except Exception:
            pass

    candidates.sort(key=lambda x: x[1])
    print(f"Found {len(candidates)} upcoming events. Screening...\n")

    rows = [screen_one(sym, edate, today) for sym, edate in candidates]
    return pd.DataFrame(rows).sort_values("earnings_date").reset_index(drop=True)


def print_report(df: pd.DataFrame, today: date) -> None:
    width = 110
    print("=" * width)
    print(f"PRE-EARNINGS OTM STRANGLE SCREENER  |  Run: {today}  |  Fingerprint C: avg_sp ≤ 3%, OI ≥ 100, AMC")
    print("=" * width)

    qualify  = df[df["status"] == "QUALIFY"]
    marginal = df[df["status"] == "MARGINAL"]
    skip     = df[~df["status"].isin(["QUALIFY", "MARGINAL"])]

    def fmt_row(r):
        sp = f"{r['avg_sp_pct']:.2f}%" if r["avg_sp_pct"] is not None else "  N/A "
        oi = f"c={r['call_OI']}/p={r['put_OI']}" if r["call_OI"] else "OI=N/A"
        iv = f"c={r['call_IV']:.3f}/p={r['put_IV']:.3f}" if r["call_IV"] else ""
        im = f"impl={r['impl_move_pct']:.1f}%" if r["impl_move_pct"] else ""
        nbr_s = f"NBR={r['nbr']:.3f}" if r.get("nbr") else "NBR=N/A"
        gate = r.get("crush_gate", "NO_DATA")
        cp = f"P={r['crush_prob']:.2f}" if r.get("crush_prob") else ""
        gate_s = f"[{gate}]" if gate != "NO_DATA" else ""
        return (f"  {r['symbol']:6s}  {str(r['earnings_date']):<12}  "
                f"T-3={str(r['t3_entry_date']):<12}  days={r['days_to_entry']:>3}  "
                f"spot=${r['spot']:>8.2f}  avg_sp={sp}  {oi:<22}  "
                f"{nbr_s:11s} {cp:8s} {gate_s:6s}  {im}")

    if not qualify.empty:
        print(f"\n{'─'*width}")
        print(f"  ✓  QUALIFY  ({len(qualify)} trades)  —  paper-trade these")
        print(f"{'─'*width}")
        for _, r in qualify.iterrows():
            note = f"  ← {r['note']}" if r["note"] else ""
            print(fmt_row(r) + note)

    if not marginal.empty:
        print(f"\n{'─'*width}")
        print(f"  ~  MARGINAL  ({len(marginal)} trades)  —  re-check spread on entry day")
        print(f"{'─'*width}")
        for _, r in marginal.iterrows():
            print(fmt_row(r))

    skipped_bmo    = (df["status"] == "SKIP_BMO").sum()
    skipped_spread = (df["status"] == "FAIL_SPREAD").sum()
    skipped_oi     = (df["status"] == "FAIL_OI").sum()
    print(f"\n{'─'*width}")
    print(f"  ✗  Excluded: {skipped_bmo} BMO  |  {skipped_spread} spread too wide  |  {skipped_oi} OI too thin  |  {(df['status']=='ERROR').sum()} errors")
    print("=" * width)


# ── Validation against known historical trades ────────────────────────────────

KNOWN_TRADES = [
    # symbol, event_date, entry_date, spot, call_strike, put_strike,
    # call_sp_pct (decimal×100), put_sp_pct (decimal×100), entry_debit_mid, pnl_c100c
    ("AMZN",  "2023-02-02", "2023-01-30", 100.55, 104.0, 104.0, 3.03, 2.26,  9.925, 41.4),
    ("AAPL",  "2023-08-03", "2023-07-31", 196.45, 202.5, 202.5, 1.20, 1.99,  9.185, 265.4),
    ("PANW",  "2023-08-18", "2023-08-15", 215.66, 222.5, 222.5, 3.72, 1.91, 19.800, 217.4),
    ("GOOGL", "2024-04-25", "2024-04-22", 156.28, 160.0, 160.0, 4.14, 1.42, 10.675,  57.4),
    ("MSFT",  "2024-04-25", "2024-04-22", 400.96, 412.5, 412.5, 2.47, 1.78, 22.975, 217.4),
]


def run_validation() -> None:
    """Verify the screener logic reproduces known backtest values."""
    print("\n" + "="*80)
    print("SCREENER VALIDATION — comparing against known backtest trades")
    print("These are historical trades; live prices won't match — checking logic only.")
    print("="*80)

    cols = ["symbol", "event_date", "entry_date", "spot_bt",
            "call_strike_bt", "put_strike_bt",
            "c_sp_bt%", "p_sp_bt%", "debit_bt", "pnl_c100c_bt",
            "t3_correct", "oi_gate_logic", "spread_gate_logic", "note"]
    rows = []
    for sym, ed, en, spot, cs, ps, csp, psp, debit, pnl in KNOWN_TRADES:
        event_date = date.fromisoformat(ed)
        entry_date = date.fromisoformat(en)
        computed_entry = t3_entry_date(event_date)
        t3_ok = computed_entry == entry_date

        avg_sp_bt = (csp + psp) / 2.0
        spread_gate = "QUALIFY" if avg_sp_bt <= 3.0 else ("MARGINAL" if avg_sp_bt <= 5.0 else "FAIL")
        # OI not stored in this table — just flag as "check in trade log"
        rows.append({
            "symbol": sym, "event_date": ed, "entry_date": en,
            "spot_bt": spot, "call_strike_bt": cs, "put_strike_bt": ps,
            "c_sp_bt%": csp, "p_sp_bt%": psp, "debit_bt": debit,
            "pnl_c100c_bt": pnl,
            "t3_correct": "✓" if t3_ok else f"✗ got {computed_entry}",
            "oi_gate_logic": "check trade log",
            "spread_gate_logic": spread_gate,
            "note": f"avg_sp={avg_sp_bt:.2f}%",
        })
    vdf = pd.DataFrame(rows)
    print(vdf.to_string(index=False))
    print("\nAll T-3 date calculations:", "PASS" if all("✓" in str(r) for r in vdf["t3_correct"]) else "FAIL — check dates")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-earnings OTM strangle forward screener")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE,
                   help="Symbols to screen (default: full universe)")
    p.add_argument("--weeks", type=int, default=6,
                   help="Look-ahead window in weeks (default: 6)")
    p.add_argument("--validate", action="store_true",
                   help="Run validation against known historical trades and exit")
    p.add_argument("--save-csv", metavar="PATH",
                   help="Save results to CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.validate:
        run_validation()
        return 0

    today = date.today()
    df = run_screener(args.symbols, args.weeks, today)
    print_report(df, today)

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved to {args.save_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
