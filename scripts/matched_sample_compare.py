#!/usr/bin/env python3
"""
matched_sample_compare.py
=========================
Compare replay vs synthetic P&L on the EXACT SAME events.

This is the definitive test: same symbols, same event dates, different pricing
engines. Any gap is purely the synthetic model's error, not universe selection.

Usage
-----
  .venv_arm64/bin/python3 scripts/matched_sample_compare.py \
      --db-path tmp/institutional_ml_test.db \
      --replay-session  backtest_20260322_124301_249339 \
      --synth-session   backtest_20260322_124309_843708
"""

from __future__ import annotations
import argparse
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load_trades(con: sqlite3.Connection, session_id: str) -> dict:
    rows = con.execute(
        """
        SELECT symbol, event_date,
               debit_per_contract, net_return_pct,
               pnl_per_contract, contracts,
               underlying_return, predicted_front_iv_crush_pct
        FROM backtest_trades
        WHERE session_id = ?
        ORDER BY symbol, event_date
        """,
        (session_id,),
    ).fetchall()
    trades = {}
    for r in rows:
        key = (r[0], r[1])          # (symbol, event_date)
        trades[key] = {
            "debit":      r[2],
            "net_ret":    r[3],
            "pnl":        r[4],
            "contracts":  r[5],
            "uret":       r[6],
            "iv_crush":   r[7],
        }
    return trades


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",        required=True)
    parser.add_argument("--replay-session", required=True)
    parser.add_argument("--synth-session",  required=True)
    args = parser.parse_args()

    con = sqlite3.connect(args.db_path)
    replay = _load_trades(con, args.replay_session)
    synth  = _load_trades(con, args.synth_session)
    con.close()

    # ── Matched events (appear in BOTH sessions) ──────────────────────────────
    matched_keys = sorted(set(replay) & set(synth))
    replay_only  = sorted(set(replay) - set(synth))
    synth_only   = sorted(set(synth)  - set(replay))

    print()
    print("=" * 78)
    print("  MATCHED-SAMPLE: REPLAY vs SYNTHETIC  (same events, different pricing)")
    print("=" * 78)
    print(f"  Replay session : {args.replay_session}")
    print(f"  Synth session  : {args.synth_session}")
    print(f"  Matched events : {len(matched_keys)}  |  "
          f"Replay-only: {len(replay_only)}  |  Synth-only: {len(synth_only)}")
    print()

    hdr = (f"  {'Symbol':<6} {'Event':<12} {'R-Debit':>8} {'S-Debit':>8} "
           f"{'R-Ret%':>8} {'S-Ret%':>8} {'R-PnL':>8} {'S-PnL':>8} "
           f"{'UMove%':>7} {'IVCrush%':>9}  Verdict")
    print(hdr)
    print("  " + "-" * 96)

    r_total_pnl = 0.0
    s_total_pnl = 0.0
    agree_wins  = 0
    agree_loss  = 0
    disagree    = 0

    rows_out = []
    for key in matched_keys:
        sym, edate = key
        r = replay[key]
        s = synth[key]
        r_pnl = r["pnl"] * r["contracts"]
        s_pnl = s["pnl"] * s["contracts"]
        r_total_pnl += r_pnl
        s_total_pnl += s_pnl
        r_win = r_pnl > 0
        s_win = s_pnl > 0
        if r_win and s_win:
            verdict = "✓ both win"
            agree_wins += 1
        elif not r_win and not s_win:
            verdict = "✗ both lose"
            agree_loss += 1
        elif r_win and not s_win:
            verdict = "⚡ replay↑ synth↓"
            disagree += 1
        else:
            verdict = "⚠ replay↓ synth↑"   # synth says win, replay says lose
            disagree += 1

        uret_pct  = r["uret"]   * 100 if r["uret"]   is not None else float("nan")
        crush_pct = r["iv_crush"] * 100 if r["iv_crush"] is not None else float("nan")
        rows_out.append((
            sym, edate,
            r["debit"], s["debit"],
            r["net_ret"] * 100, s["net_ret"] * 100,
            r_pnl, s_pnl,
            uret_pct, crush_pct,
            verdict,
        ))

    # Sort: biggest absolute divergence first
    rows_out.sort(key=lambda x: abs(x[6] - x[7]), reverse=True)

    for row in rows_out:
        sym, edate, r_deb, s_deb, r_ret, s_ret, r_pnl, s_pnl, uret, crush, verdict = row
        print(
            f"  {sym:<6} {edate:<12} {r_deb:>8.0f} {s_deb:>8.0f} "
            f"{r_ret:>8.1f} {s_ret:>8.1f} {r_pnl:>8.0f} {s_pnl:>8.0f} "
            f"{uret:>7.1f} {crush:>9.1f}  {verdict}"
        )

    print("  " + "-" * 96)
    pnl_gap = r_total_pnl - s_total_pnl
    print(
        f"  {'TOTAL':<6} {'':<12} {'':>8} {'':>8} "
        f"{'':>8} {'':>8} {r_total_pnl:>8.0f} {s_total_pnl:>8.0f}"
    )
    print()
    print(f"  Synthetic overstatement on matched sample : ${abs(pnl_gap):.0f}  "
          f"({'synthetic too optimistic' if pnl_gap < 0 else 'synthetic too pessimistic'})")
    print(f"  Agreement breakdown : both-win={agree_wins}  both-lose={agree_loss}  "
          f"disagree={disagree}")
    print()

    # ── Underlying move analysis ───────────────────────────────────────────────
    print("  UNDERLYING MOVE ANALYSIS (key driver of calendar spread loss)")
    print("  " + "-" * 60)

    big_move_losers  = [(k, replay[k]) for k in matched_keys
                        if replay[k]["pnl"] < 0 and abs(replay[k]["uret"]) > 0.05]
    small_move_wins  = [(k, replay[k]) for k in matched_keys
                        if replay[k]["pnl"] > 0 and abs(replay[k]["uret"]) <= 0.05]
    big_move_wins    = [(k, replay[k]) for k in matched_keys
                        if replay[k]["pnl"] > 0 and abs(replay[k]["uret"]) > 0.05]

    print(f"  Losers with |underlying move| > 5% : {len(big_move_losers)}")
    for k, t in big_move_losers:
        print(f"    {k[0]:<6} {k[1]}  uret={t['uret']*100:+.1f}%  pnl={t['pnl']:>8.2f}")

    print(f"  Winners with |underlying move| <= 5%: {len(small_move_wins)}")
    for k, t in small_move_wins:
        print(f"    {k[0]:<6} {k[1]}  uret={t['uret']*100:+.1f}%  pnl={t['pnl']:>8.2f}")

    if big_move_wins:
        print(f"  Winners with |underlying move| > 5% : {len(big_move_wins)}  ← lucky breaks")
        for k, t in big_move_wins:
            print(f"    {k[0]:<6} {k[1]}  uret={t['uret']*100:+.1f}%  pnl={t['pnl']:>8.2f}")
    print()

    # ── High-debit analysis ────────────────────────────────────────────────────
    print("  HIGH-DEBIT OUTLIER ANALYSIS (replay debit > $400)")
    print("  " + "-" * 60)
    high_debit = sorted(
        [(k, replay[k]) for k in matched_keys if replay[k]["debit"] > 400],
        key=lambda x: x[1]["debit"], reverse=True
    )
    if high_debit:
        for k, t in high_debit:
            flag = "OUTLIER" if t["debit"] > 800 else ""
            print(f"  {k[0]:<6} {k[1]}  debit={t['debit']:>8.0f}  "
                  f"pnl={t['pnl']:>8.0f}  uret={t['uret']*100:+.1f}%  {flag}")
    else:
        print("  None.")
    print()

    # ── Events not in both sessions ────────────────────────────────────────────
    if replay_only:
        print("  IN REPLAY ONLY (no synthetic equivalent):")
        for k in replay_only:
            t = replay[k]
            print(f"    {k[0]:<6} {k[1]}  pnl={t['pnl']:>8.2f}")
        print()
    if synth_only:
        print("  IN SYNTHETIC ONLY (no replay pair — not in matched sample):")
        for k in synth_only:
            t = synth[k]
            print(f"    {k[0]:<6} {k[1]}  pnl={t['pnl']:>8.2f}")
        print()

    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
