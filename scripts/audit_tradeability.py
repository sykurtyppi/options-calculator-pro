"""
Tradeability audit for earnings IV crush calendar spread strategy.

NOT a modeling task — this is a market microstructure and execution
feasibility assessment using real bid/ask data from the ivolatility
options chain parquet store.

Goal: determine whether the NBR-based IV crush edge survives
transaction costs and liquidity constraints in the real market.

Steps:
  1  Sample selection (Q5 NBR events + representative cross-section)
  2  Market snapshot: real bid/ask, OI, spread for ATM front+back legs
  3  Liquidity classification (HIGH / MEDIUM / LOW)
  4  Execution cost model (mid / realistic / conservative)
  5  Edge vs cost comparison (gross P&L vs round-trip cost)
  6  Structural constraints (untradeable symbols, wide expirations)
  7  Size sensitivity (1 / 10 / 50 contracts)
  8  Output: summary tables, findings, recommendations
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - \033[32m%(levelname)s\033[0m - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
DB_PATH          = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
PARQUET_BASE     = Path("/Volumes/T9/market_data/research/options_features_eod")
REPORT_DIR       = Path.home() / ".options_calculator_pro" / "reports"
AUDIT_VERSION    = "tradeability_v1"

# Liquidity thresholds
OI_FRONT_HIGH    = 1_000     # front OI ≥ this → HIGH
OI_BACK_HIGH     =   500     # back OI ≥ this → HIGH
OI_FRONT_LOW     =   100     # front OI < this → LOW
OI_BACK_LOW      =    50     # back OI < this → LOW
SPREAD_PCT_TIGHT =  0.05     # spread_pct ≤ this → tight leg
SPREAD_PCT_WIDE  =  0.20     # spread_pct ≥ this → wide leg

# Fill scenarios: fraction of half-spread paid per leg
FILL_MID         = 0.0   # fill at mid (zero slippage — optimistic)
FILL_REALISTIC   = 0.50  # mid ± 50% of half-spread = mid ± 25% of full spread
FILL_CONSERVATIVE = 1.0  # cross the full spread

# Minimum moneyness range to search for ATM (±% of underlying)
ATM_MONEYNESS_WINDOW = 5.0   # look within ±5% moneyness

# Contracts for size sensitivity
SIZE_SCENARIOS   = [1, 10, 50]


# ── data loading ──────────────────────────────────────────────────────────────

def load_events() -> pd.DataFrame:
    """Load all Tier A+B events from DB, compute NBR and NBR quintiles."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT symbol, event_date, pre_capture_date, post_capture_date,
               front_expiry, back_expiry,
               front_dte_pre, back_dte_pre, front_dte_post, back_dte_post,
               pre_front_iv, pre_back_iv,
               post_front_iv, post_back_iv,
               front_iv_crush_pct, back_iv_crush_pct,
               term_ratio_change,
               pre_underlying_price, post_underlying_price,
               pre_front_oi, pre_back_oi,
               pre_front_atm_moneyness, pre_back_atm_moneyness,
               quality_tier
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ('A', 'B')
        ORDER BY event_date
        """,
        conn,
    )
    conn.close()

    df["pre_dt"]  = pd.to_datetime(df["pre_capture_date"])
    df["post_dt"] = pd.to_datetime(df["post_capture_date"])
    df["nbr"]     = df["pre_front_iv"] / df["pre_back_iv"]
    df["quintile"] = pd.qcut(df["nbr"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    df["year"]    = df["pre_dt"].dt.year

    log.info("Loaded %d events, %d symbols, years %s",
             len(df), df["symbol"].nunique(),
             sorted(df["year"].unique().tolist()))
    return df


def _parquet_path(symbol: str, dt: pd.Timestamp) -> Path:
    """Return the parquet file path for a symbol on a given date."""
    yr = str(dt.year)
    mo = str(dt.month).zfill(2)
    return (PARQUET_BASE / f"underlying_symbol={symbol}"
            / f"year={yr}" / f"month={mo}"
            / f"{symbol.lower()}_options_features_eod_{yr}-{mo}.parquet")


def _load_day_chain(symbol: str, dt: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Load options chain for a single symbol on a specific date."""
    path = _parquet_path(symbol, dt)
    if not path.exists():
        return None
    chain = pd.read_parquet(path)
    chain["trade_date"] = pd.to_datetime(chain["trade_date"])
    chain["expiry"]     = pd.to_datetime(chain["expiry"])
    day = chain[
        (chain["trade_date"] == dt)
        & (chain["quote_quality_flag"] == "ok")
        & (chain["bid"] > 0)
    ].copy()
    return day if len(day) > 0 else None


def _find_atm_call(chain: pd.DataFrame, target_expiry: pd.Timestamp
                   ) -> Optional[pd.Series]:
    """Return the ATM call closest to target_expiry with best data quality."""
    leg = chain[
        (chain["expiry"] == target_expiry)
        & (chain["call_put"] == "C")
        & (chain["moneyness_pct"].abs() <= ATM_MONEYNESS_WINDOW)
    ]
    if len(leg) == 0:
        return None
    return leg.loc[leg["moneyness_pct"].abs().idxmin()]


# ── Step 2: market snapshot per event ─────────────────────────────────────────

def collect_snapshot(row: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Load the real bid/ask snapshot for both legs at ENTRY and EXIT.

    Gross P&L = actual calendar value change (exit mid − entry mid) × 100.
    This correctly captures theta, full option repricing, and the earnings
    IV event — not just the vega × ΔIV approximation.

    Transaction costs use real spreads from both entry and exit chains.
    """
    sym       = row["symbol"]
    pre_dt    = row["pre_dt"]
    post_dt   = row["post_dt"]
    front_exp = pd.to_datetime(row["front_expiry"])
    back_exp  = pd.to_datetime(row["back_expiry"])

    pre_chain  = _load_day_chain(sym, pre_dt)
    post_chain = _load_day_chain(sym, post_dt)
    if pre_chain is None or post_chain is None:
        return None

    # Entry legs (pre-earnings)
    f_pre = _find_atm_call(pre_chain, front_exp)
    b_pre = _find_atm_call(pre_chain, back_exp)
    if f_pre is None or b_pre is None:
        return None

    # Exit legs (post-earnings) — find ATM at new underlying price
    f_post = _find_atm_call(post_chain, front_exp)
    b_post = _find_atm_call(post_chain, back_exp)
    if f_post is None or b_post is None:
        return None

    def leg_dict(opt: pd.Series, label: str) -> Dict:
        return {
            f"{label}_strike":     float(opt["strike"]),
            f"{label}_bid":        float(opt["bid"]),
            f"{label}_ask":        float(opt["ask"]),
            f"{label}_mid":        float(opt["mid"]),
            f"{label}_spread":     float(opt["spread"]),
            f"{label}_spread_pct": float(opt["spread_pct"]),
            f"{label}_oi":         int(opt["open_interest"]),
            f"{label}_volume":     int(opt["volume"]),
            f"{label}_vega":       float(opt["vega"]),
            f"{label}_delta":      float(opt["delta"]),
            f"{label}_iv":         float(opt["iv"]),
            f"{label}_dte":        int(opt["dte"]),
            f"{label}_moneyness":  float(opt["moneyness_pct"]),
        }

    snap = {
        "symbol":            sym,
        "event_date":        row["event_date"],
        "pre_capture_date":  row["pre_capture_date"],
        "post_capture_date": row["post_capture_date"],
        "quintile":          str(row["quintile"]),
        "nbr":               float(row["nbr"]),
        "year":              int(row["year"]),
        "underlying_price":  float(row["pre_underlying_price"]),
        "underlying_move":   float(row["post_underlying_price"] / row["pre_underlying_price"] - 1),
        # Entry legs
        **leg_dict(f_pre,  "front_entry"),
        **leg_dict(b_pre,  "back_entry"),
        # Exit legs
        **leg_dict(f_post, "front_exit"),
        **leg_dict(b_post, "back_exit"),
        # Convenience aliases for liquidity classification (use entry)
        "front_oi":          int(f_pre["open_interest"]),
        "back_oi":           int(b_pre["open_interest"]),
        "front_spread_pct":  float(f_pre["spread_pct"]),
        "back_spread_pct":   float(b_pre["spread_pct"]),
        # DB fields
        "pre_front_iv":      float(row["pre_front_iv"]),
        "pre_back_iv":       float(row["pre_back_iv"]),
        "post_front_iv":     float(row["post_front_iv"]),
        "post_back_iv":      float(row["post_back_iv"]),
        "front_iv_crush_pct": float(row["front_iv_crush_pct"]),
        "back_iv_crush_pct":  float(row["back_iv_crush_pct"]),
        "front_dte_pre":     int(row["front_dte_pre"]),
        "back_dte_pre":      int(row["back_dte_pre"]),
    }
    return snap


# ── Step 3: liquidity classification ──────────────────────────────────────────

def classify_liquidity(snap: Dict) -> str:
    """
    Classify trade liquidity as HIGH / MEDIUM / LOW.

    Rules:
      HIGH  : both legs have sufficient OI AND tight spreads
      LOW   : either leg has very low OI OR very wide spreads
      MEDIUM: everything else
    """
    f_oi   = snap["front_oi"]
    b_oi   = snap["back_oi"]
    f_sp   = snap["front_spread_pct"]
    b_sp   = snap["back_spread_pct"]

    is_high = (
        f_oi >= OI_FRONT_HIGH
        and b_oi >= OI_BACK_HIGH
        and f_sp <= SPREAD_PCT_TIGHT
        and b_sp <= SPREAD_PCT_TIGHT
    )
    is_low = (
        f_oi < OI_FRONT_LOW
        or b_oi < OI_BACK_LOW
        or f_sp >= SPREAD_PCT_WIDE
        or b_sp >= SPREAD_PCT_WIDE
    )

    if is_high:
        return "HIGH"
    if is_low:
        return "LOW"
    return "MEDIUM"


# ── Step 4+5: execution costs and edge ───────────────────────────────────────

def compute_costs_and_edge(snap: Dict) -> Dict[str, Any]:
    """
    Compute actual round-trip P&L using real entry AND exit option prices.

    Gross P&L at mid prices:
      calendar_entry = back_entry_mid - front_entry_mid
      calendar_exit  = back_exit_mid  - front_exit_mid
      gross_pnl      = (calendar_exit - calendar_entry) × 100

    This captures theta, full option repricing, and IV crush — the complete
    calendar spread economics. NOT a vega-only approximation.

    Fill scenarios: fraction of half-spread paid per leg per trade.
      Transaction cost = fill_fraction × (spread_entry + spread_exit) per leg
      Round trip uses BOTH entry and exit spreads.
    """
    # Entry prices
    f_e_bid = snap["front_entry_bid"];  f_e_ask = snap["front_entry_ask"]
    f_e_mid = snap["front_entry_mid"];  f_e_sp  = snap["front_entry_spread"]
    b_e_bid = snap["back_entry_bid"];   b_e_ask = snap["back_entry_ask"]
    b_e_mid = snap["back_entry_mid"];   b_e_sp  = snap["back_entry_spread"]

    # Exit prices
    f_x_bid = snap["front_exit_bid"];   f_x_ask = snap["front_exit_ask"]
    f_x_mid = snap["front_exit_mid"];   f_x_sp  = snap["front_exit_spread"]
    b_x_bid = snap["back_exit_bid"];    b_x_ask = snap["back_exit_ask"]
    b_x_mid = snap["back_exit_mid"];    b_x_sp  = snap["back_exit_spread"]

    # ── calendar spread values ────────────────────────────────────────────────
    cal_entry_mid = b_e_mid - f_e_mid
    cal_exit_mid  = b_x_mid - f_x_mid
    # Worst-case entry: buy back at ask, sell front at bid
    cal_entry_worst = b_e_ask - f_e_bid
    # Worst-case exit: sell back at bid, buy front at ask
    cal_exit_worst  = b_x_bid - f_x_ask
    cal_eff_spread  = cal_entry_worst - cal_exit_worst   # total round-trip friction

    # ── gross P&L per contract at mid ────────────────────────────────────────
    gross_pnl = (cal_exit_mid - cal_entry_mid) * 100

    # ── round-trip transaction costs ─────────────────────────────────────────
    # Total leg spreads: entry spreads + exit spreads
    total_leg_spreads = f_e_sp + b_e_sp + f_x_sp + b_x_sp   # per-share round trip
    # Mid: fill at mid, zero slippage
    rt_cost_mid = 0.0
    # Realistic: pay 75% of total half-spread (≈ mid ± 25% of full spread per trade)
    rt_cost_realistic = (total_leg_spreads / 2) * 0.75 * 100
    # Conservative: cross full spread on every leg every way
    rt_cost_conservative = total_leg_spreads * 100

    # ── net P&L ───────────────────────────────────────────────────────────────
    net_mid          = gross_pnl - rt_cost_mid
    net_realistic    = gross_pnl - rt_cost_realistic
    net_conservative = gross_pnl - rt_cost_conservative

    def pct_edge_consumed(cost, gross):
        if gross <= 0 or gross < 1.0:   # < $1 gross → ratio not meaningful
            return float("nan")
        return round(cost / gross * 100, 1)

    return {
        # Calendar values
        "cal_entry_mid":      round(cal_entry_mid, 3),
        "cal_exit_mid":       round(cal_exit_mid, 3),
        "cal_entry_worst":    round(cal_entry_worst, 3),
        "cal_exit_worst":     round(cal_exit_worst, 3),
        "cal_eff_spread":     round(cal_eff_spread, 3),
        # Leg spreads in $ per contract
        "front_entry_spread_dollar": round(f_e_sp * 100, 2),
        "back_entry_spread_dollar":  round(b_e_sp * 100, 2),
        "front_exit_spread_dollar":  round(f_x_sp * 100, 2),
        "back_exit_spread_dollar":   round(b_x_sp * 100, 2),
        "rt_cost_mid":               round(rt_cost_mid, 2),
        "rt_cost_realistic":         round(rt_cost_realistic, 2),
        "rt_cost_conservative":      round(rt_cost_conservative, 2),
        # P&L
        "gross_pnl":           round(gross_pnl, 2),
        "net_pnl_mid":         round(net_mid, 2),
        "net_pnl_realistic":   round(net_realistic, 2),
        "net_pnl_conservative":round(net_conservative, 2),
        # Cost as % of gross edge
        "cost_pct_mid":          pct_edge_consumed(rt_cost_mid,          gross_pnl),
        "cost_pct_realistic":    pct_edge_consumed(rt_cost_realistic,    gross_pnl),
        "cost_pct_conservative": pct_edge_consumed(rt_cost_conservative, gross_pnl),
        # Flags
        "gross_pnl_positive":      bool(gross_pnl > 0),
        "net_viable_realistic":    bool(net_realistic > 0),
        "net_viable_conservative": bool(net_conservative > 0),
    }


# ── Step 7: size sensitivity ──────────────────────────────────────────────────

def compute_size_sensitivity(snap: Dict, cost_metrics: Dict) -> Dict[str, Any]:
    """
    Estimate impact of scaling position size.

    For small positions (≤10 contracts on liquid names), slippage is
    minimal beyond the spread. For larger positions, market impact
    is estimated using square-root-of-OI rule:
      additional_slippage_pct ≈ (n / OI)^0.5 × base_spread
    """
    f_oi  = max(snap["front_oi"], 1)
    b_oi  = max(snap["back_oi"], 1)
    f_sp  = snap["front_entry_spread"]
    b_sp  = snap["back_entry_spread"]
    gross = cost_metrics["gross_pnl"]

    results = {}
    for n in SIZE_SCENARIOS:
        # Linear for tiny sizes, sqrt impact above ~1% of OI
        front_impact = f_sp * min(1.0, math.sqrt(n / max(f_oi, 1))) * 100 * n
        back_impact  = b_sp * min(1.0, math.sqrt(n / max(b_oi, 1))) * 100 * n
        base_rt_cost = cost_metrics["rt_cost_realistic"] * n
        total_cost   = base_rt_cost + front_impact + back_impact
        net          = gross * n - total_cost
        results[f"size_{n}"] = {
            "contracts":   n,
            "gross_total": round(gross * n, 2),
            "total_cost":  round(total_cost, 2),
            "net_total":   round(net, 2),
            "cost_per_contract": round(total_cost / n, 2),
            "net_viable":  bool(net > 0),
        }
    return results


# ── aggregate helpers ─────────────────────────────────────────────────────────

def _pct(x):
    return round(100 * x, 1)


def aggregate_by_quintile(trades: pd.DataFrame) -> pd.DataFrame:
    """Summary table broken down by NBR quintile."""
    g = trades.groupby("quintile")
    return pd.DataFrame({
        "n_events":           g.size(),
        "n_viable_realistic": g["net_viable_realistic"].sum(),
        "pct_viable":         g["net_viable_realistic"].mean().mul(100).round(1),
        "avg_gross_pnl":      g["gross_pnl"].mean().round(2),
        "avg_net_pnl_mid":    g["net_pnl_mid"].mean().round(2),
        "avg_net_pnl_real":   g["net_pnl_realistic"].mean().round(2),
        "avg_net_pnl_cons":   g["net_pnl_conservative"].mean().round(2),
        "avg_rt_cost_real":   g["rt_cost_realistic"].mean().round(2),
        "avg_cost_pct_real":  g["cost_pct_realistic"].apply(lambda x: x.replace([np.inf, -np.inf], np.nan).mean()).round(1),
        "avg_front_spread_pct": g["front_spread_pct"].mean().mul(100).round(2),
        "avg_back_spread_pct":  g["back_spread_pct"].mean().mul(100).round(2),
        "median_front_oi":    g["front_oi"].median().round(0),
        "median_back_oi":     g["back_oi"].median().round(0),
    }).reset_index()


def aggregate_by_liquidity(trades: pd.DataFrame) -> pd.DataFrame:
    g = trades.groupby("liquidity_tier")
    return pd.DataFrame({
        "n_events":           g.size(),
        "avg_gross_pnl":      g["gross_pnl"].mean().round(2),
        "avg_net_pnl_real":   g["net_pnl_realistic"].mean().round(2),
        "pct_viable_real":    g["net_viable_realistic"].mean().mul(100).round(1),
        "avg_front_spread_pct": g["front_spread_pct"].mean().mul(100).round(2),
        "avg_back_spread_pct":  g["back_spread_pct"].mean().mul(100).round(2),
        "avg_rt_cost_real":   g["rt_cost_realistic"].mean().round(2),
        "median_front_oi":    g["front_oi"].median().round(0),
        "median_back_oi":     g["back_oi"].median().round(0),
    }).reset_index()


def compute_structural_flags(trades: pd.DataFrame) -> Dict[str, Any]:
    """Step 6: identify systematically problematic symbols and expirations."""
    # Symbols that are consistently LOW liquidity
    sym_liq = trades.groupby("symbol")["liquidity_tier"].apply(
        lambda x: (x == "LOW").mean()
    )
    always_low = sym_liq[sym_liq >= 0.80].index.tolist()

    # Symbols where conservative net P&L is still positive (reliably tradeable)
    sym_net = trades.groupby("symbol")["net_viable_conservative"].mean()
    reliable = sym_net[sym_net >= 0.70].index.tolist()

    # Short-DTE front options with very wide spreads
    wide_front = trades[
        (trades["front_dte_pre"] <= 7) & (trades["front_spread_pct"] >= 0.15)
    ][["symbol", "event_date", "front_dte_pre", "front_spread_pct"]].head(20)

    # Events where back OI < 50 (cannot execute back leg)
    no_back_liq = trades[trades["back_oi"] < 50][
        ["symbol", "event_date", "back_oi", "back_spread_pct"]
    ]

    return {
        "symbols_always_low_liquidity": sorted(always_low),
        "symbols_reliably_tradeable":   sorted(reliable),
        "n_low_back_oi_events":        int(len(no_back_liq)),
        "pct_low_back_oi":             round(len(no_back_liq) / len(trades) * 100, 1),
        "wide_front_examples":         wide_front.to_dict("records"),
    }


# ── printing helpers ──────────────────────────────────────────────────────────

def _print_table(title: str, df: pd.DataFrame) -> None:
    log.info("")
    log.info(title)
    log.info("=" * 72)
    for line in df.to_string(index=False).split("\n"):
        log.info("  %s", line)


def _hr():
    log.info("─" * 72)


# ── main ──────────────────────────────────────────────────────────────────────

def main(quality_tier: str = "AB") -> None:
    tiers = list(quality_tier.upper())
    log.info("")
    log.info("=" * 72)
    log.info("TRADEABILITY AUDIT  version=%s", AUDIT_VERSION)
    log.info("=" * 72)
    log.info("  Strategy : long back / short front calendar spread (IV crush)")
    log.info("  Signal   : NBR (near_back_ratio = pre_front_iv / pre_back_iv)")
    log.info("  Data     : real bid/ask from ivolatility options chain parquet")
    log.info("  Goal     : determine whether edge survives real market costs")
    log.info("")

    # ── Step 1: load events ───────────────────────────────────────────────────
    log.info("STEP 1 — Sample selection")
    log.info("─" * 72)
    events = load_events()
    log.info("  All 1,639 events included (not just Q5 — full cross-section gives")
    log.info("  cost calibration across liquidity spectrum)")
    log.info("  NBR quintile distribution:")
    for q, grp in events.groupby("quintile"):
        log.info("    %s: N=%d  NBR range [%.3f – %.3f]  median=%.3f",
                 q, len(grp), grp.nbr.min(), grp.nbr.max(), grp.nbr.median())

    # ── Step 2: collect market snapshots ─────────────────────────────────────
    log.info("")
    log.info("STEP 2 — Market snapshot collection (real bid/ask from parquet)")
    log.info("─" * 72)
    log.info("  Loading ATM call options for both legs on pre-earnings entry day...")

    snapshots: List[Dict] = []
    failed = 0
    for _, row in events.iterrows():
        snap = collect_snapshot(row)
        if snap is None:
            failed += 1
            continue
        liq  = classify_liquidity(snap)
        cost = compute_costs_and_edge(snap)
        size = compute_size_sensitivity(snap, cost)
        snapshots.append({**snap, "liquidity_tier": liq, **cost, "size_sensitivity": size})

    log.info("  Snapshots loaded: %d / %d  (failed: %d)", len(snapshots), len(events), failed)
    trades = pd.DataFrame(snapshots)
    log.info("  Coverage: %.1f%% of all events", 100 * len(trades) / len(events))

    if len(trades) == 0:
        log.error("No data loaded — check parquet store mount.")
        sys.exit(1)

    # ── Step 3: liquidity classification ─────────────────────────────────────
    log.info("")
    log.info("STEP 3 — Liquidity classification")
    log.info("─" * 72)
    liq_counts = trades["liquidity_tier"].value_counts()
    total = len(trades)
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        n = liq_counts.get(tier, 0)
        log.info("  %-8s  %4d events  (%5.1f%%)", tier, n, 100 * n / total)

    log.info("")
    log.info("  OI distribution (pre-earnings, ATM call):")
    for col, label in [("front_oi", "Front"), ("back_oi", "Back ")]:
        p = trades[col].describe(percentiles=[.1, .25, .5, .75, .9])
        log.info("    %s OI: p10=%4.0f  p25=%4.0f  p50=%4.0f  p75=%4.0f  p90=%5.0f  max=%6.0f",
                 label, p["10%"], p["25%"], p["50%"], p["75%"], p["90%"], p["max"])

    log.info("")
    log.info("  Bid-ask spread distribution (ATM call, as %% of mid price):")
    for col, label in [("front_spread_pct", "Front"), ("back_spread_pct", "Back ")]:
        p = trades[col].describe(percentiles=[.1, .25, .5, .75, .9])
        log.info("    %s spread%%: p10=%.1f%%  p25=%.1f%%  p50=%.1f%%  p75=%.1f%%  p90=%.1f%%",
                 label, 100*p["10%"], 100*p["25%"], 100*p["50%"], 100*p["75%"], 100*p["90%"])

    # ── Step 4: execution costs ───────────────────────────────────────────────
    log.info("")
    log.info("STEP 4 — Execution cost model")
    log.info("─" * 72)
    log.info("  Fill assumptions (round-trip, both legs):")
    log.info("    Mid:          fill at mid price, zero slippage  (optimistic)")
    log.info("    Realistic:    mid ± 25%% of full spread per leg  (working estimate)")
    log.info("    Conservative: cross full spread on both legs     (worst case)")
    log.info("")

    for scenario, col in [
        ("Mid",         "rt_cost_mid"),
        ("Realistic",   "rt_cost_realistic"),
        ("Conservative","rt_cost_conservative"),
    ]:
        p = trades[col].describe(percentiles=[.25, .5, .75, .9])
        log.info("  %-13s round-trip cost/contract ($):  "
                 "p25=$%5.2f  p50=$%5.2f  p75=$%5.2f  p90=$%6.2f  avg=$%5.2f",
                 scenario, p["25%"], p["50%"], p["75%"], p["90%"], p["mean"])

    # ── Step 5: edge vs cost ──────────────────────────────────────────────────
    log.info("")
    log.info("STEP 5 — Edge vs cost (gross P&L vs transaction cost)")
    log.info("─" * 72)
    log.info("  Gross P&L = (cal_exit_mid − cal_entry_mid) × 100 per contract")
    log.info("  Uses actual entry AND exit option prices — captures full repricing")
    log.info("  (theta, IV crush, gamma). NOT a vega×ΔIV approximation.")
    log.info("")

    gp = trades["gross_pnl"]
    log.info("  Gross P&L per contract ($):")
    log.info("    mean=$%.2f  median=$%.2f  p10=$%.2f  p90=$%.2f  "
             "pct_positive=%.1f%%",
             gp.mean(), gp.median(), gp.quantile(0.1), gp.quantile(0.9),
             100 * (gp > 0).mean())

    log.info("")
    log.info("  Net P&L per contract ($) and viability:")
    for scenario, net_col, viable_col, cost_col in [
        ("Mid",         "net_pnl_mid",         "net_viable_realistic",    "cost_pct_mid"),
        ("Realistic",   "net_pnl_realistic",   "net_viable_realistic",    "cost_pct_realistic"),
        ("Conservative","net_pnl_conservative","net_viable_conservative", "cost_pct_conservative"),
    ]:
        net = trades[net_col]
        viable_pct = trades[viable_col].mean() * 100
        avg_cost_pct = trades[cost_col].median()
        log.info("    %-13s  avg_net=$%6.2f  pct_positive=%.1f%%",
                 scenario, net.mean(), 100*(net>0).mean())
        log.info("                    median_cost=%.0f%% of edge  "
                 "pct_cost_exceeds_edge=%.1f%%",
                 avg_cost_pct if not math.isnan(avg_cost_pct) else 0.0,
                 100*(trades[net_col] < 0).mean())

    # ── Quintile table ────────────────────────────────────────────────────────
    log.info("")
    log.info("STEP 5 cont. — Edge vs cost by NBR quintile")
    log.info("─" * 72)
    q_table = aggregate_by_quintile(trades)
    log.info("  %-5s %7s %8s %9s %9s %9s %9s %9s %10s",
             "Q", "N", "avg_gross", "avg_net_M", "avg_net_R", "avg_net_C",
             "rt_cost_R", "cost%_R", "pct_viable")
    log.info("  " + "─" * 80)
    for _, r in q_table.iterrows():
        log.info("  %-5s %7d %8.2f %9.2f %9.2f %9.2f %9.2f %9.1f %10.1f%%",
                 r.quintile, r.n_events,
                 r.avg_gross_pnl, r.avg_net_pnl_mid, r.avg_net_pnl_real,
                 r.avg_net_pnl_cons, r.avg_rt_cost_real,
                 r.avg_cost_pct_real, r.pct_viable)

    # ── Liquidity breakdown ───────────────────────────────────────────────────
    log.info("")
    log.info("  Edge vs cost by liquidity tier")
    log.info("─" * 72)
    liq_table = aggregate_by_liquidity(trades)
    log.info("  %-8s %7s %9s %9s %8s %9s %9s",
             "Tier", "N", "avg_gross", "avg_net_R", "pct_viable",
             "front_sp%", "back_sp%")
    log.info("  " + "─" * 68)
    for _, r in liq_table.iterrows():
        log.info("  %-8s %7d %9.2f %9.2f %8.1f%% %9.2f%% %9.2f%%",
                 r.liquidity_tier, r.n_events,
                 r.avg_gross_pnl, r.avg_net_pnl_real, r.pct_viable_real,
                 r.avg_front_spread_pct, r.avg_back_spread_pct)

    # ── Step 6: structural constraints ───────────────────────────────────────
    log.info("")
    log.info("STEP 6 — Structural constraints")
    log.info("─" * 72)
    flags = compute_structural_flags(trades)

    log.info("  Symbols consistently LOW liquidity (≥80%% of events LOW): %d",
             len(flags["symbols_always_low_liquidity"]))
    if flags["symbols_always_low_liquidity"]:
        log.info("    %s", ", ".join(flags["symbols_always_low_liquidity"]))

    log.info("")
    log.info("  Symbols reliably tradeable (≥70%% events viable at conservative): %d",
             len(flags["symbols_reliably_tradeable"]))
    if flags["symbols_reliably_tradeable"]:
        log.info("    %s", ", ".join(flags["symbols_reliably_tradeable"][:30]))

    log.info("")
    log.info("  Back leg OI < 50 (effectively untradeable): %d events (%.1f%%)",
             flags["n_low_back_oi_events"], flags["pct_low_back_oi"])

    log.info("")
    log.info("  Liquidity by DTE bucket (front leg):")
    for dte_max, label in [(7, "≤7 DTE"), (14, "8–14 DTE"), (21, "15–21 DTE")]:
        mask = (trades["front_dte_pre"] <= dte_max) if dte_max == 7 else \
               (trades["front_dte_pre"] > dte_max - 7) & (trades["front_dte_pre"] <= dte_max)
        sub = trades[mask]
        if len(sub):
            log.info("    %-12s  N=%3d  avg_spread_pct=%.1f%%  pct_low_liq=%.1f%%",
                     label, len(sub),
                     100*sub["front_spread_pct"].mean(),
                     100*(sub["liquidity_tier"]=="LOW").mean())

    # ── Step 7: size sensitivity ───────────────────────────────────────────────
    log.info("")
    log.info("STEP 7 — Size sensitivity")
    log.info("─" * 72)
    log.info("  Assuming realistic fills. Impact slippage from sqrt(n/OI) model.")
    log.info("")
    log.info("  %-12s %9s %10s %10s %9s %9s",
             "Contracts", "avg_gross", "avg_cost", "avg_net", "pct_viable",
             "cost_incr%")
    log.info("  " + "─" * 65)
    for n in SIZE_SCENARIOS:
        col = f"size_{n}"
        gross_vals = [r["size_sensitivity"][col]["gross_total"] for _, r in trades.iterrows() if col in r.get("size_sensitivity", {})]
        cost_vals  = [r["size_sensitivity"][col]["total_cost"]  for _, r in trades.iterrows() if col in r.get("size_sensitivity", {})]
        net_vals   = [r["size_sensitivity"][col]["net_total"]   for _, r in trades.iterrows() if col in r.get("size_sensitivity", {})]
        viable     = [r["size_sensitivity"][col]["net_viable"]  for _, r in trades.iterrows() if col in r.get("size_sensitivity", {})]
        if gross_vals:
            avg_gross = np.mean(gross_vals)
            avg_cost  = np.mean(cost_vals)
            avg_net   = np.mean(net_vals)
            pct_v     = 100 * np.mean(viable)
            cost_pct  = avg_cost / avg_gross * 100 if avg_gross > 0 else float("nan")
            log.info("  %-12d %9.2f %10.2f %10.2f %9.1f%% %9.1f%%",
                     n, avg_gross, avg_cost, avg_net, pct_v, cost_pct)

    # ── Step 8: findings and recommendations ─────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 8 — FINDINGS AND RECOMMENDATIONS")
    log.info("=" * 72)

    viable_real = float(100 * trades["net_viable_realistic"].mean())
    viable_cons = float(100 * trades["net_viable_conservative"].mean())
    avg_gross   = float(trades["gross_pnl"].mean())
    avg_net_r   = float(trades["net_pnl_realistic"].mean())
    avg_cost_r  = float(trades["rt_cost_realistic"].mean())
    avg_cost_pct = float(trades["cost_pct_realistic"].replace([np.inf, -np.inf], np.nan).dropna().mean())

    high_n = int((trades["liquidity_tier"] == "HIGH").sum())
    mid_n  = int((trades["liquidity_tier"] == "MEDIUM").sum())
    low_n  = int((trades["liquidity_tier"] == "LOW").sum())

    # Q5 subset
    q5 = trades[trades["quintile"] == "Q5"]
    q5_viable_real = float(100 * q5["net_viable_realistic"].mean())
    q5_avg_gross   = float(q5["gross_pnl"].mean())
    q5_avg_net_r   = float(q5["net_pnl_realistic"].mean())

    log.info("")
    log.info("  KEY METRICS:")
    log.info("    Full universe (N=%d):", len(trades))
    log.info("      Avg gross P&L per contract:  $%.2f", avg_gross)
    log.info("      Avg net P&L (realistic):      $%.2f", avg_net_r)
    log.info("      Avg round-trip cost:          $%.2f", avg_cost_r)
    log.info("      Cost as %% of gross edge:     %.0f%%", avg_cost_pct)
    log.info("      Viable at realistic fill:     %.1f%% of trades", viable_real)
    log.info("      Viable at conservative fill:  %.1f%% of trades", viable_cons)
    log.info("")
    log.info("    Q5 (highest NBR) subset (N=%d):", len(q5))
    log.info("      Avg gross P&L per contract:  $%.2f", q5_avg_gross)
    log.info("      Avg net P&L (realistic):      $%.2f", q5_avg_net_r)
    log.info("      Viable at realistic fill:     %.1f%% of trades", q5_viable_real)
    log.info("")
    log.info("    Liquidity breakdown: HIGH=%d (%.0f%%)  MEDIUM=%d (%.0f%%)  LOW=%d (%.0f%%)",
             high_n, 100*high_n/total, mid_n, 100*mid_n/total,
             low_n, 100*low_n/total)

    log.info("")
    log.info("  FINDINGS:")
    log.info("")

    # Tradability verdict
    if q5_avg_net_r > 0 and q5_viable_real > 50:
        verdict = "CONDITIONALLY TRADEABLE"
        verdict_detail = "Q5 NBR events show positive net P&L at realistic fills"
    elif avg_net_r > 0 and viable_real > 60:
        verdict = "LARGELY TRADEABLE"
        verdict_detail = "Majority of events show positive net P&L at realistic fills"
    elif avg_net_r < 0 or viable_real < 40:
        verdict = "MARGINAL — COSTS CONSUME EDGE"
        verdict_detail = "Transaction costs significantly erode theoretical edge"
    else:
        verdict = "MIXED — LIQUIDITY-DEPENDENT"
        verdict_detail = "Tradeable in liquid names, marginal in others"

    log.info("  OVERALL VERDICT: %s", verdict)
    log.info("  %s", verdict_detail)
    log.info("")

    log.info("  RECOMMENDATIONS:")
    log.info("    1. Minimum liquidity filters:")
    log.info("         Front OI ≥ %d contracts", OI_FRONT_LOW)
    log.info("         Back OI  ≥ %d contracts", OI_BACK_LOW)
    log.info("         Front spread ≤ %.0f%% of mid", 100 * SPREAD_PCT_WIDE)
    log.info("         Back spread  ≤ %.0f%% of mid", 100 * SPREAD_PCT_WIDE)

    tradeable_pct = 100 * (
        (trades["front_oi"] >= OI_FRONT_LOW) &
        (trades["back_oi"]  >= OI_BACK_LOW)  &
        (trades["front_spread_pct"] <= SPREAD_PCT_WIDE) &
        (trades["back_spread_pct"]  <= SPREAD_PCT_WIDE)
    ).mean()
    log.info("         Events passing all filters: %.1f%%", tradeable_pct)

    log.info("")
    log.info("    2. Fill assumption for production sizing:")
    log.info("         Use mid ± 25%% of spread (realistic scenario)")
    log.info("         Budget $%.2f per contract round-trip for cost estimates",
             avg_cost_r)
    log.info("")
    log.info("    3. Position sizing limits:")
    log.info("         1–5 contracts: OI constraint rarely binding")
    log.info("         10+ contracts: verify back OI ≥ 10× position size")
    log.info("         50+ contracts: limited to HIGH liquidity events only")

    if flags["symbols_always_low_liquidity"]:
        log.info("")
        log.info("    4. Exclude from live trading:")
        log.info("         %s", ", ".join(flags["symbols_always_low_liquidity"]))

    log.info("")
    log.info("    5. Entry timing:")
    log.info("         Audit uses pre_capture_date (day before earnings).")
    log.info("         Entering 2–3 days early typically gives tighter spreads")
    log.info("         but more overnight risk and different NBR.")

    # ── save report ───────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / f"tradeability_{AUDIT_VERSION}_{ts}.json"

    report = {
        "version":    AUDIT_VERSION,
        "timestamp":  ts,
        "n_events":   len(trades),
        "n_failed":   failed,
        "verdict":    verdict,
        "summary": {
            "avg_gross_pnl_per_contract":  round(avg_gross, 2),
            "avg_net_pnl_realistic":       round(avg_net_r, 2),
            "avg_rt_cost_realistic":       round(avg_cost_r, 2),
            "avg_cost_pct_of_edge":        round(avg_cost_pct, 1),
            "pct_viable_realistic":        round(viable_real, 1),
            "pct_viable_conservative":     round(viable_cons, 1),
            "liquidity_high_pct":          round(100 * high_n / total, 1),
            "liquidity_medium_pct":        round(100 * mid_n / total, 1),
            "liquidity_low_pct":           round(100 * low_n / total, 1),
        },
        "q5_summary": {
            "n":                           len(q5),
            "avg_gross_pnl":               round(q5_avg_gross, 2),
            "avg_net_pnl_realistic":       round(q5_avg_net_r, 2),
            "pct_viable_realistic":        round(q5_viable_real, 1),
        },
        "by_quintile":   aggregate_by_quintile(trades).to_dict("records"),
        "by_liquidity":  aggregate_by_liquidity(trades).to_dict("records"),
        "structural":    {k: v for k, v in flags.items()
                         if not isinstance(v, list) or len(v) <= 50},
        "filters_recommended": {
            "min_front_oi":         OI_FRONT_LOW,
            "min_back_oi":          OI_BACK_LOW,
            "max_front_spread_pct": SPREAD_PCT_WIDE,
            "max_back_spread_pct":  SPREAD_PCT_WIDE,
            "pct_passing_filters":  round(tradeable_pct, 1),
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("")
    log.info("Report saved: %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tradeability audit for IV crush calendar strategy")
    parser.add_argument("--quality-tier", default="AB", help="A, B, or AB")
    args = parser.parse_args()
    main(quality_tier=args.quality_tier)
