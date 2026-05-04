"""
Entry/exit timing optimization for earnings IV crush calendar spread strategy.

Tests entry at T-7, T-5, T-3, T-2, T-1 (business days before earnings)
and exit at T+0 EOD (post_capture_date) and T+1 EOD (next business day).

Tradeable universe: HIGH+MEDIUM liquidity at T-1, Q4-Q5 NBR,
applying hard filters: front OI ≥ 100, back OI ≥ 50, spread ≤ 20%.

Methodology:
  - "T-1" = pre_capture_date (the standard audit entry point)
  - Entry at T-N = pre_capture_date - (N-1) business days
  - "T+0" exit = post_capture_date (first full trading day post-earnings)
  - "T+1" exit = post_capture_date + 1 business day

Gross P&L = (cal_exit_mid - cal_entry_mid) × 100 per contract
            using actual option prices, not vega approximation.

Realistic cost = (entry_spread + exit_spread) / 2 × 0.75 × 100 per contract
  (same model as tradeability audit)
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
DB_PATH       = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
PARQUET_BASE  = Path("/Volumes/T9/market_data/research/options_features_eod")
REPORT_DIR    = Path.home() / ".options_calculator_pro" / "reports"
AUDIT_VERSION = "entry_exit_v1"

# Universe filters (applied at T-1 — the screening checkpoint)
MIN_FRONT_OI      = 100
MIN_BACK_OI       = 50
MAX_SPREAD_PCT    = 0.20
NBR_QUINTILE_KEEP = {"Q4", "Q5"}

# ATM moneyness search window
ATM_MONEYNESS_WINDOW = 5.0

# Entry timing: key is label, value is number of BDays subtracted from pre_capture_date
# pre_capture_date is already T-1, so T-7 = pre_capture_date - 6 BDays
ENTRY_OFFSETS: Dict[str, int] = {
    "T-7": 6,
    "T-5": 4,
    "T-3": 2,
    "T-2": 1,
    "T-1": 0,
}

# Exit timing: number of BDays added to post_capture_date
# post_capture_date = T+0 (first full trading day post-earnings)
EXIT_OFFSETS: Dict[str, int] = {
    "T+0": 0,
    "T+1": 1,
}

# Fill fraction for realistic cost model
FILL_REALISTIC = 0.75   # pay 75% of half-spread per leg


# ── data loading ──────────────────────────────────────────────────────────────

def load_events() -> pd.DataFrame:
    """Load Tier A+B events from DB with NBR and quintile."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT symbol, event_date, pre_capture_date, post_capture_date,
               front_expiry, back_expiry,
               front_dte_pre, back_dte_pre,
               pre_front_iv, pre_back_iv,
               post_front_iv, post_back_iv,
               front_iv_crush_pct, back_iv_crush_pct,
               term_ratio_change,
               pre_underlying_price, post_underlying_price,
               pre_front_oi, pre_back_oi,
               quality_tier
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ('A', 'B')
        ORDER BY event_date
        """,
        conn,
    )
    conn.close()
    df["pre_dt"]    = pd.to_datetime(df["pre_capture_date"])
    df["post_dt"]   = pd.to_datetime(df["post_capture_date"])
    df["nbr"]       = df["pre_front_iv"] / df["pre_back_iv"]
    df["quintile"]  = pd.qcut(df["nbr"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    df["year"]      = df["pre_dt"].dt.year
    return df


def _parquet_path(symbol: str, dt: pd.Timestamp) -> Path:
    yr = str(dt.year)
    mo = str(dt.month).zfill(2)
    return (PARQUET_BASE / f"underlying_symbol={symbol}"
            / f"year={yr}" / f"month={mo}"
            / f"{symbol.lower()}_options_features_eod_{yr}-{mo}.parquet")


def _load_day_chain(symbol: str, dt: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Load and filter the options chain for a symbol on a specific date."""
    path = _parquet_path(symbol, dt)
    if not path.exists():
        return None
    try:
        chain = pd.read_parquet(path)
    except Exception:
        return None
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
    """Find the ATM call closest to target_expiry."""
    leg = chain[
        (chain["expiry"] == target_expiry)
        & (chain["call_put"] == "C")
        & (chain["moneyness_pct"].abs() <= ATM_MONEYNESS_WINDOW)
    ]
    if len(leg) == 0:
        return None
    return leg.loc[leg["moneyness_pct"].abs().idxmin()]


def _leg_snap(opt: pd.Series) -> Dict[str, Any]:
    """Extract key fields from an ATM option row."""
    return {
        "strike":     float(opt["strike"]),
        "bid":        float(opt["bid"]),
        "ask":        float(opt["ask"]),
        "mid":        float(opt["mid"]),
        "spread":     float(opt["spread"]),
        "spread_pct": float(opt["spread_pct"]),
        "oi":         int(opt["open_interest"]),
        "volume":     int(opt["volume"]),
        "vega":       float(opt["vega"]),
        "iv":         float(opt["iv"]),
        "dte":        int(opt["dte"]),
        "moneyness":  float(opt["moneyness_pct"]),
    }


# ── T-1 universe filtering ────────────────────────────────────────────────────

def build_t1_universe(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load T-1 snapshots for all events, apply liquidity + NBR filters.
    Returns (tradeable_df, full_snap_df) where tradeable_df is the filtered universe.
    """
    log.info("Building T-1 universe (loading chains for all %d events)...", len(events))

    rows = []
    n_fail = 0
    for _, ev in events.iterrows():
        pre_chain  = _load_day_chain(ev["symbol"], ev["pre_dt"])
        post_chain = _load_day_chain(ev["symbol"], ev["post_dt"])
        if pre_chain is None or post_chain is None:
            n_fail += 1
            continue

        front_exp = pd.to_datetime(ev["front_expiry"])
        back_exp  = pd.to_datetime(ev["back_expiry"])

        f_pre  = _find_atm_call(pre_chain,  front_exp)
        b_pre  = _find_atm_call(pre_chain,  back_exp)
        f_post = _find_atm_call(post_chain, front_exp)
        b_post = _find_atm_call(post_chain, back_exp)
        if any(x is None for x in [f_pre, b_pre, f_post, b_post]):
            n_fail += 1
            continue

        row = {
            "symbol":      ev["symbol"],
            "event_date":  ev["event_date"],
            "year":        int(ev["year"]),
            "quintile":    str(ev["quintile"]),
            "nbr":         float(ev["nbr"]),
            "pre_dt":      ev["pre_dt"],
            "post_dt":     ev["post_dt"],
            "front_expiry": ev["front_expiry"],
            "back_expiry":  ev["back_expiry"],
            # T-1 entry data
            "t1_front_mid":    float(f_pre["mid"]),
            "t1_front_spread": float(f_pre["spread"]),
            "t1_front_sp_pct": float(f_pre["spread_pct"]),
            "t1_front_oi":     int(f_pre["open_interest"]),
            "t1_front_dte":    int(f_pre["dte"]),
            "t1_front_iv":     float(f_pre["iv"]),
            "t1_back_mid":     float(b_pre["mid"]),
            "t1_back_spread":  float(b_pre["spread"]),
            "t1_back_sp_pct":  float(b_pre["spread_pct"]),
            "t1_back_oi":      int(b_pre["open_interest"]),
            "t1_back_dte":     int(b_pre["dte"]),
            "t1_back_iv":      float(b_pre["iv"]),
            # T+0 exit data (post_capture_date)
            "t0x_front_mid":   float(f_post["mid"]),
            "t0x_front_spread":float(f_post["spread"]),
            "t0x_front_sp_pct":float(f_post["spread_pct"]),
            "t0x_back_mid":    float(b_post["mid"]),
            "t0x_back_spread": float(b_post["spread"]),
            "t0x_back_sp_pct": float(b_post["spread_pct"]),
            # DB IV values for reference
            "pre_front_iv":  float(ev["pre_front_iv"]),
            "pre_back_iv":   float(ev["pre_back_iv"]),
            "post_front_iv": float(ev["post_front_iv"]),
            "post_back_iv":  float(ev["post_back_iv"]),
            "front_iv_crush_pct": float(ev["front_iv_crush_pct"]),
        }
        rows.append(row)

    snap_df = pd.DataFrame(rows)
    log.info("  T-1 chains loaded: %d / %d  (failed: %d)",
             len(snap_df), len(events), n_fail)

    # Apply filters
    mask = (
        (snap_df["t1_front_oi"]     >= MIN_FRONT_OI)
        & (snap_df["t1_back_oi"]    >= MIN_BACK_OI)
        & (snap_df["t1_front_sp_pct"] <= MAX_SPREAD_PCT)
        & (snap_df["t1_back_sp_pct"]  <= MAX_SPREAD_PCT)
        & (snap_df["quintile"].isin(NBR_QUINTILE_KEEP))
    )
    tradeable = snap_df[mask].copy()
    log.info("  After filters (OI/spread + Q4-Q5): %d events (%.1f%% of loaded)",
             len(tradeable), 100 * len(tradeable) / max(len(snap_df), 1))
    log.info("  Quintile breakdown: %s",
             tradeable["quintile"].value_counts().to_dict())
    log.info("  Year breakdown: %s",
             tradeable["year"].value_counts().sort_index().to_dict())
    return tradeable, snap_df


# ── entry/exit chain loading ──────────────────────────────────────────────────

def load_entry_chain(ev_row: pd.Series, offset_bdays: int
                     ) -> Optional[Dict[str, Any]]:
    """
    Load the options chain at entry date = pre_dt - offset_bdays.
    Returns front/back leg snap or None if data unavailable.
    """
    if offset_bdays == 0:
        entry_dt = ev_row["pre_dt"]
    else:
        entry_dt = ev_row["pre_dt"] - pd.offsets.BDay(offset_bdays)

    front_exp = pd.to_datetime(ev_row["front_expiry"])
    back_exp  = pd.to_datetime(ev_row["back_expiry"])

    chain = _load_day_chain(ev_row["symbol"], entry_dt)
    if chain is None:
        return None

    f = _find_atm_call(chain, front_exp)
    b = _find_atm_call(chain, back_exp)
    if f is None or b is None:
        return None

    f_snap = _leg_snap(f)
    b_snap = _leg_snap(b)
    return {
        "entry_dt":          entry_dt,
        "front_mid":         f_snap["mid"],
        "front_bid":         f_snap["bid"],
        "front_ask":         f_snap["ask"],
        "front_spread":      f_snap["spread"],
        "front_sp_pct":      f_snap["spread_pct"],
        "front_oi":          f_snap["oi"],
        "front_dte":         f_snap["dte"],
        "front_iv":          f_snap["iv"],
        "back_mid":          b_snap["mid"],
        "back_bid":          b_snap["bid"],
        "back_ask":          b_snap["ask"],
        "back_spread":       b_snap["spread"],
        "back_sp_pct":       b_snap["spread_pct"],
        "back_oi":           b_snap["oi"],
        "back_dte":          b_snap["dte"],
        "back_iv":           b_snap["iv"],
        "cal_mid":           b_snap["mid"] - f_snap["mid"],
        "nbr_at_entry":      f_snap["iv"] / b_snap["iv"] if b_snap["iv"] > 0 else float("nan"),
    }


def load_exit_chain(ev_row: pd.Series, offset_bdays: int
                    ) -> Optional[Dict[str, Any]]:
    """
    Load the options chain at exit date = post_dt + offset_bdays.
    offset_bdays=0 → post_capture_date (T+0), =1 → T+1.
    """
    if offset_bdays == 0:
        exit_dt = ev_row["post_dt"]
    else:
        exit_dt = ev_row["post_dt"] + pd.offsets.BDay(offset_bdays)

    front_exp = pd.to_datetime(ev_row["front_expiry"])
    back_exp  = pd.to_datetime(ev_row["back_expiry"])

    chain = _load_day_chain(ev_row["symbol"], exit_dt)
    if chain is None:
        return None

    f = _find_atm_call(chain, front_exp)
    b = _find_atm_call(chain, back_exp)
    if f is None or b is None:
        return None

    f_snap = _leg_snap(f)
    b_snap = _leg_snap(b)
    return {
        "exit_dt":           exit_dt,
        "front_mid":         f_snap["mid"],
        "front_spread":      f_snap["spread"],
        "front_sp_pct":      f_snap["spread_pct"],
        "front_iv":          f_snap["iv"],
        "back_mid":          b_snap["mid"],
        "back_spread":       b_snap["spread"],
        "back_sp_pct":       b_snap["spread_pct"],
        "back_iv":           b_snap["iv"],
        "cal_mid":           b_snap["mid"] - f_snap["mid"],
        "nbr_at_exit":       f_snap["iv"] / b_snap["iv"] if b_snap["iv"] > 0 else float("nan"),
    }


# ── P&L computation ──────────────────────────────────────────────────────────

def compute_pnl(entry: Dict, exit_: Dict) -> Dict[str, float]:
    """
    Compute calendar spread P&L for one (entry, exit) pair.

    Gross P&L = (cal_exit_mid - cal_entry_mid) × 100
    Realistic cost = (entry_spread_total + exit_spread_total) / 2 × 0.75 × 100
    where spread_total = front_spread + back_spread (both legs)
    """
    cal_entry = entry["cal_mid"]
    cal_exit  = exit_["cal_mid"]
    gross_pnl = (cal_exit - cal_entry) * 100

    entry_leg_spread = entry["front_spread"] + entry["back_spread"]
    exit_leg_spread  = exit_["front_spread"] + exit_["back_spread"]
    total_spread     = entry_leg_spread + exit_leg_spread

    rt_cost_mid          = 0.0
    rt_cost_realistic    = (total_spread / 2) * FILL_REALISTIC * 100
    rt_cost_conservative = total_spread * 100

    return {
        "gross_pnl":           round(gross_pnl, 2),
        "rt_cost_realistic":   round(rt_cost_realistic, 2),
        "rt_cost_conservative":round(rt_cost_conservative, 2),
        "net_pnl_mid":         round(gross_pnl, 2),
        "net_pnl_realistic":   round(gross_pnl - rt_cost_realistic, 2),
        "net_pnl_conservative":round(gross_pnl - rt_cost_conservative, 2),
        "entry_front_sp_pct":  entry["front_sp_pct"],
        "entry_back_sp_pct":   entry["back_sp_pct"],
        "exit_front_sp_pct":   exit_["front_sp_pct"],
        "exit_back_sp_pct":    exit_["back_sp_pct"],
        "entry_front_dte":     entry["front_dte"],
        "entry_nbr":           entry["nbr_at_entry"],
        "entry_front_iv":      entry["front_iv"],
        "entry_back_iv":       entry["back_iv"],
        "exit_nbr":            exit_["nbr_at_exit"],
        "exit_front_iv":       exit_["front_iv"],
        "exit_back_iv":        exit_["back_iv"],
        "cal_entry_mid":       cal_entry,
        "cal_exit_mid":        cal_exit,
    }


# ── main sweep ───────────────────────────────────────────────────────────────

def run_timing_sweep(tradeable: pd.DataFrame) -> pd.DataFrame:
    """
    For each event × entry_timing × exit_timing, collect P&L data.
    Returns a long-format DataFrame with one row per (event, entry, exit).
    """
    # Pre-load exit chains: for each event, T+0 exit chain is already loaded
    # (we have t0x_* fields). For T+1, we need to load an extra chain.
    # Cache exit chains to avoid double-loading.
    exit_cache: Dict[Tuple, Optional[Dict]] = {}
    entry_cache: Dict[Tuple, Optional[Dict]] = {}

    records = []
    total = len(tradeable)

    for i, (_, ev) in enumerate(tradeable.iterrows()):
        if (i + 1) % 20 == 0:
            log.info("  Processing event %d/%d ...", i + 1, total)

        sym = ev["symbol"]
        front_exp = pd.to_datetime(ev["front_expiry"])
        back_exp  = pd.to_datetime(ev["back_expiry"])

        for entry_label, entry_bday_offset in ENTRY_OFFSETS.items():

            # Load entry chain (cached)
            e_key = (sym, ev["event_date"], entry_label)
            if e_key not in entry_cache:
                if entry_label == "T-1":
                    # Use pre-loaded T-1 data from tradeable df
                    entry_cache[e_key] = {
                        "entry_dt":      ev["pre_dt"],
                        "front_mid":     ev["t1_front_mid"],
                        "front_bid":     ev["t1_front_mid"] - ev["t1_front_spread"] / 2,
                        "front_ask":     ev["t1_front_mid"] + ev["t1_front_spread"] / 2,
                        "front_spread":  ev["t1_front_spread"],
                        "front_sp_pct":  ev["t1_front_sp_pct"],
                        "front_oi":      ev["t1_front_oi"],
                        "front_dte":     ev["t1_front_dte"],
                        "front_iv":      ev["t1_front_iv"],
                        "back_mid":      ev["t1_back_mid"],
                        "back_bid":      ev["t1_back_mid"] - ev["t1_back_spread"] / 2,
                        "back_ask":      ev["t1_back_mid"] + ev["t1_back_spread"] / 2,
                        "back_spread":   ev["t1_back_spread"],
                        "back_sp_pct":   ev["t1_back_sp_pct"],
                        "back_oi":       ev["t1_back_oi"],
                        "back_dte":      ev["t1_back_dte"],
                        "back_iv":       ev["t1_back_iv"],
                        "cal_mid":       ev["t1_back_mid"] - ev["t1_front_mid"],
                        "nbr_at_entry":  ev["nbr"],
                    }
                else:
                    entry_cache[e_key] = load_entry_chain(ev, entry_bday_offset)

            entry = entry_cache[e_key]
            if entry is None:
                continue

            for exit_label, exit_bday_offset in EXIT_OFFSETS.items():

                # Load exit chain (cached)
                x_key = (sym, ev["event_date"], exit_label)
                if x_key not in exit_cache:
                    if exit_label == "T+0":
                        # Use pre-loaded T+0 data
                        exit_cache[x_key] = {
                            "exit_dt":      ev["post_dt"],
                            "front_mid":    ev["t0x_front_mid"],
                            "front_spread": ev["t0x_front_spread"],
                            "front_sp_pct": ev["t0x_front_sp_pct"],
                            "front_iv":     ev.get("post_front_iv", float("nan")),
                            "back_mid":     ev["t0x_back_mid"],
                            "back_spread":  ev["t0x_back_spread"],
                            "back_sp_pct":  ev["t0x_back_sp_pct"],
                            "back_iv":      ev.get("post_back_iv", float("nan")),
                            "cal_mid":      ev["t0x_back_mid"] - ev["t0x_front_mid"],
                            "nbr_at_exit":  (ev.get("post_front_iv", 0) /
                                             ev.get("post_back_iv", 1)
                                             if ev.get("post_back_iv", 0) > 0
                                             else float("nan")),
                        }
                    else:
                        exit_cache[x_key] = load_exit_chain(ev, exit_bday_offset)

                exit_ = exit_cache[x_key]
                if exit_ is None:
                    continue

                pnl = compute_pnl(entry, exit_)
                records.append({
                    "symbol":      sym,
                    "event_date":  ev["event_date"],
                    "year":        ev["year"],
                    "quintile":    ev["quintile"],
                    "nbr_t1":      ev["nbr"],
                    "entry_label": entry_label,
                    "exit_label":  exit_label,
                    **pnl,
                })

    return pd.DataFrame(records)


# ── aggregation and reporting ─────────────────────────────────────────────────

def _summary_stats(df: pd.DataFrame, label: str) -> Dict:
    """Compute summary statistics for a P&L series."""
    net = df["net_pnl_realistic"]
    gross = df["gross_pnl"]
    cost = df["rt_cost_realistic"]
    return {
        "label":          label,
        "n":              len(df),
        "avg_gross":      round(gross.mean(), 2),
        "avg_cost":       round(cost.mean(), 2),
        "avg_net":        round(net.mean(), 2),
        "median_net":     round(net.median(), 2),
        "pct_positive":   round(100 * (net > 0).mean(), 1),
        "std_net":        round(net.std(), 2),
        "p10_net":        round(net.quantile(0.10), 2),
        "p90_net":        round(net.quantile(0.90), 2),
        "worst":          round(net.min(), 2),
        "best":           round(net.max(), 2),
        "avg_entry_front_sp": round(df["entry_front_sp_pct"].mean() * 100, 2),
        "avg_entry_back_sp":  round(df["entry_back_sp_pct"].mean() * 100, 2),
        "avg_exit_front_sp":  round(df["exit_front_sp_pct"].mean() * 100, 2),
        "avg_exit_back_sp":   round(df["exit_back_sp_pct"].mean() * 100, 2),
        "avg_entry_nbr":  round(df["entry_nbr"].mean(), 4),
        "avg_entry_dte":  round(df["entry_front_dte"].mean(), 1),
    }


def build_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """Build the main (entry, exit) comparison table."""
    rows = []
    entry_order = ["T-7", "T-5", "T-3", "T-2", "T-1"]
    exit_order  = ["T+0", "T+1"]
    for ent in entry_order:
        for ext in exit_order:
            sub = results[(results["entry_label"] == ent) & (results["exit_label"] == ext)]
            if len(sub) < 5:
                continue
            s = _summary_stats(sub, f"{ent}→{ext}")
            s["entry"] = ent
            s["exit"]  = ext
            # Q5-only subset
            q5 = sub[sub["quintile"] == "Q5"]
            s["q5_avg_net"]     = round(q5["net_pnl_realistic"].mean(), 2) if len(q5) > 0 else float("nan")
            s["q5_pct_pos"]     = round(100 * (q5["net_pnl_realistic"] > 0).mean(), 1) if len(q5) > 0 else float("nan")
            rows.append(s)
    return pd.DataFrame(rows)


def build_nbr_timing_table(results: pd.DataFrame) -> pd.DataFrame:
    """NBR level at each entry timing — shows IV inflation as you enter later."""
    rows = []
    entry_order = ["T-7", "T-5", "T-3", "T-2", "T-1"]
    for ent in entry_order:
        sub = results[results["entry_label"] == ent].drop_duplicates(
            subset=["symbol", "event_date"])
        if len(sub) == 0:
            continue
        rows.append({
            "entry":          ent,
            "n_events":       len(sub),
            "avg_nbr":        round(sub["entry_nbr"].mean(), 4),
            "median_nbr":     round(sub["entry_nbr"].median(), 4),
            "avg_front_sp_pct": round(sub["entry_front_sp_pct"].mean() * 100, 2),
            "avg_back_sp_pct":  round(sub["entry_back_sp_pct"].mean() * 100, 2),
            "avg_front_dte":  round(sub["entry_front_dte"].mean(), 1),
            "avg_rt_cost":    round(
                ((sub["entry_front_sp_pct"] * sub["t1_front_mid"].reindex(sub.index, fill_value=0)
                  if "t1_front_mid" in sub.columns else pd.Series(0, index=sub.index))).mean(), 2
            ),
        })
    # simpler version without sub-df merging
    rows2 = []
    for ent in entry_order:
        sub = results[results["entry_label"] == ent].drop_duplicates(
            subset=["symbol", "event_date"])
        if len(sub) == 0:
            continue
        rows2.append({
            "entry":              ent,
            "n_events":           len(sub),
            "avg_nbr":            round(sub["entry_nbr"].mean(), 4),
            "median_nbr":         round(sub["entry_nbr"].median(), 4),
            "avg_front_sp_pct%":  round(sub["entry_front_sp_pct"].mean() * 100, 2),
            "avg_back_sp_pct%":   round(sub["entry_back_sp_pct"].mean() * 100, 2),
            "avg_front_dte":      round(sub["entry_front_dte"].mean(), 1),
        })
    return pd.DataFrame(rows2)


def build_year_stability(results: pd.DataFrame) -> pd.DataFrame:
    """Per-year breakdown for each (entry, exit) combo to check stability."""
    rows = []
    entry_order = ["T-7", "T-5", "T-3", "T-2", "T-1"]
    exit_order  = ["T+0", "T+1"]
    for yr in sorted(results["year"].unique()):
        for ent in entry_order:
            for ext in exit_order:
                sub = results[
                    (results["year"] == yr)
                    & (results["entry_label"] == ent)
                    & (results["exit_label"] == ext)
                ]
                if len(sub) < 3:
                    continue
                rows.append({
                    "year":       yr,
                    "entry":      ent,
                    "exit":       ext,
                    "n":          len(sub),
                    "avg_gross":  round(sub["gross_pnl"].mean(), 2),
                    "avg_net":    round(sub["net_pnl_realistic"].mean(), 2),
                    "pct_pos":    round(100 * (sub["net_pnl_realistic"] > 0).mean(), 1),
                })
    return pd.DataFrame(rows)


def print_table(title: str, df: pd.DataFrame) -> None:
    log.info("")
    log.info(title)
    log.info("=" * 80)
    for line in df.to_string(index=False).split("\n"):
        log.info("  %s", line)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("")
    log.info("=" * 80)
    log.info("ENTRY/EXIT TIMING OPTIMIZATION  version=%s", AUDIT_VERSION)
    log.info("=" * 80)
    log.info("  Universe : HIGH+MEDIUM liquidity, Q4-Q5 NBR, OI/spread filters")
    log.info("  Entry    : T-7, T-5, T-3, T-2, T-1 (business days before earnings)")
    log.info("  Exit     : T+0 EOD, T+1 EOD (days after earnings)")
    log.info("  P&L      : actual option price change, realistic fill assumptions")
    log.info("")

    # ── Step 1: load and filter universe ─────────────────────────────────────
    log.info("STEP 1 — Universe definition")
    log.info("─" * 80)
    events = load_events()
    tradeable, _ = build_t1_universe(events)

    log.info("")
    log.info("  Tradeable universe: %d events across %d symbols",
             len(tradeable), tradeable["symbol"].nunique())
    log.info("  NBR range: [%.3f – %.3f]  median=%.3f",
             tradeable["nbr"].min(), tradeable["nbr"].max(), tradeable["nbr"].median())
    log.info("  Year distribution: %s",
             tradeable["year"].value_counts().sort_index().to_dict())

    # ── Step 2+3: sweep all entry/exit combos ─────────────────────────────────
    log.info("")
    log.info("STEP 2–3 — Entry/exit timing sweep")
    log.info("─" * 80)
    log.info("  Loading chains for all entry/exit offsets...")
    results = run_timing_sweep(tradeable)
    log.info("  Total (event × entry × exit) records: %d", len(results))

    # ── Step 2 detail: NBR and spread at each entry timing ────────────────────
    log.info("")
    log.info("STEP 2 — NBR and spread at each entry timing")
    log.info("─" * 80)
    nbr_table = build_nbr_timing_table(results)
    print_table("  NBR, spread, and DTE at each entry point", nbr_table)

    # Coverage at each entry
    log.info("")
    log.info("  Data coverage by entry timing (events with valid chains):")
    for ent in ["T-7", "T-5", "T-3", "T-2", "T-1"]:
        n = results[results["entry_label"] == ent]["event_date"].nunique()
        log.info("    %s : %d / %d events  (%.0f%%)",
                 ent, n, len(tradeable), 100 * n / max(len(tradeable), 1))

    # ── Step 4+5+6: P&L comparison table ─────────────────────────────────────
    log.info("")
    log.info("STEP 4–6 — P&L comparison: all (entry, exit) combinations")
    log.info("─" * 80)
    comp = build_comparison_table(results)

    log.info("")
    log.info("  %-10s %-8s %6s %8s %8s %8s %8s %9s %7s %8s %8s %8s",
             "Entry", "Exit", "N",
             "avg_grs", "avg_cst", "avg_net", "med_net",
             "pct_pos", "std",
             "p10_net", "q5_net", "q5_pos%")
    log.info("  " + "─" * 110)
    for _, r in comp.iterrows():
        log.info("  %-10s %-8s %6d %8.2f %8.2f %8.2f %8.2f %9.1f%% %7.2f %8.2f %8.2f %8.1f%%",
                 r["entry"], r["exit"], r["n"],
                 r["avg_gross"], r["avg_cost"], r["avg_net"], r["median_net"],
                 r["pct_positive"], r["std_net"],
                 r["p10_net"],
                 r.get("q5_avg_net", float("nan")),
                 r.get("q5_pct_pos", float("nan")))

    # ── Best combination highlight ────────────────────────────────────────────
    log.info("")
    log.info("  Best combinations by avg net P&L (realistic fills):")
    best = comp.nlargest(5, "avg_net")[["entry", "exit", "n", "avg_gross", "avg_cost",
                                         "avg_net", "pct_positive", "q5_avg_net"]]
    for _, r in best.iterrows():
        log.info("    %s→%s  N=%d  avg_net=$%.2f  pct_pos=%.1f%%  q5_net=$%.2f",
                 r["entry"], r["exit"], r["n"],
                 r["avg_net"], r["pct_positive"],
                 r.get("q5_avg_net", float("nan")))

    # ── Spread comparison across entry timings ────────────────────────────────
    log.info("")
    log.info("STEP 5 — Entry timing trade-off: NBR vs spread vs net P&L")
    log.info("─" * 80)
    log.info("  %-8s %-8s  %8s %8s %8s %8s %8s %9s",
             "Entry", "Exit",
             "avg_NBR", "f_sp%", "b_sp%",
             "avg_grs", "avg_net", "pct_pos")
    log.info("  " + "─" * 75)
    for _, r in comp.iterrows():
        log.info("  %-8s %-8s  %8.4f %8.2f %8.2f %8.2f %8.2f %9.1f%%",
                 r["entry"], r["exit"],
                 r["avg_entry_nbr"],
                 r["avg_entry_front_sp"], r["avg_entry_back_sp"],
                 r["avg_gross"], r["avg_net"],
                 r["pct_positive"])

    # ── Step 7: Year-by-year stability ────────────────────────────────────────
    log.info("")
    log.info("STEP 7 — Year-by-year stability (best 3 entry/exit combos)")
    log.info("─" * 80)
    yr_table = build_year_stability(results)

    # Find the top 3 combos by avg_net from comp
    top3 = comp.nlargest(3, "avg_net")[["entry", "exit"]].values.tolist()
    for entry, exit_ in top3:
        sub = yr_table[(yr_table["entry"] == entry) & (yr_table["exit"] == exit_)]
        if len(sub) == 0:
            continue
        log.info("")
        log.info("  %s → %s:", entry, exit_)
        log.info("  %-6s %5s %9s %9s %9s",
                 "Year", "N", "avg_gross", "avg_net", "pct_pos")
        for _, yr_row in sub.iterrows():
            log.info("  %-6s %5d %9.2f %9.2f %9.1f%%",
                     yr_row["year"], yr_row["n"],
                     yr_row["avg_gross"], yr_row["avg_net"], yr_row["pct_pos"])

    # ── Step 8+9: Final recommendation ────────────────────────────────────────
    log.info("")
    log.info("=" * 80)
    log.info("STEP 8–9 — OPTIMAL PROTOCOL AND STRATEGY DEFINITION")
    log.info("=" * 80)

    # Identify winner
    best_row = comp.loc[comp["avg_net"].idxmax()]

    log.info("")
    log.info("  OPTIMAL ENTRY: %s  (best avg net P&L at realistic fills)", best_row["entry"])
    log.info("  OPTIMAL EXIT : %s", best_row["exit"])
    log.info("")
    log.info("  Performance at optimal timing:")
    log.info("    N events:              %d", int(best_row["n"]))
    log.info("    Avg gross P&L:         $%.2f/contract", best_row["avg_gross"])
    log.info("    Avg realistic cost:    $%.2f/contract", best_row["avg_cost"])
    log.info("    Avg net P&L:           $%.2f/contract", best_row["avg_net"])
    log.info("    Median net P&L:        $%.2f/contract", best_row["median_net"])
    log.info("    Pct profitable:        %.1f%%", best_row["pct_positive"])
    log.info("    Std dev of net:        $%.2f", best_row["std_net"])
    log.info("    Worst case (p10):      $%.2f", best_row["p10_net"])
    log.info("    Worst single trade:    $%.2f", best_row["worst"])
    log.info("    Q5 avg net:            $%.2f", best_row.get("q5_avg_net", float("nan")))
    log.info("    Q5 pct profitable:     %.1f%%", best_row.get("q5_pct_pos", float("nan")))

    # Entry timing NBR vs T-1 comparison
    t1_row = comp[(comp["entry"] == "T-1") & (comp["exit"] == best_row["exit"])]
    if len(t1_row) > 0:
        t1_row = t1_row.iloc[0]
        if best_row["entry"] != "T-1":
            log.info("")
            log.info("  Comparison vs T-1 entry (same exit %s):", best_row["exit"])
            log.info("    T-1 avg net:   $%.2f  vs  %s avg net: $%.2f  (Δ = $%.2f)",
                     t1_row["avg_net"], best_row["entry"],
                     best_row["avg_net"],
                     best_row["avg_net"] - t1_row["avg_net"])
            log.info("    T-1 pct_pos:   %.1f%%  vs  %s pct_pos: %.1f%%",
                     t1_row["pct_positive"], best_row["entry"],
                     best_row["pct_positive"])
            log.info("    T-1 NBR:       %.4f  vs  %s NBR: %.4f",
                     t1_row["avg_entry_nbr"], best_row["entry"],
                     best_row["avg_entry_nbr"])
            log.info("    T-1 front_sp:  %.2f%%  vs  %s front_sp: %.2f%%",
                     t1_row["avg_entry_front_sp"], best_row["entry"],
                     best_row["avg_entry_front_sp"])

    log.info("")
    log.info("  FINAL TRADING RULE:")
    log.info("  ─" * 40)
    log.info("  Universe filter:")
    log.info("    - Earnings IV crush calendar spread (long back, short front)")
    log.info("    - Front OI ≥ %d contracts at T-1", MIN_FRONT_OI)
    log.info("    - Back OI  ≥ %d contracts at T-1", MIN_BACK_OI)
    log.info("    - Front and back spread ≤ %.0f%% of mid at T-1", MAX_SPREAD_PCT * 100)
    log.info("    - NBR quintile Q4 or Q5 (NBR ≥ 1.39 based on full universe)")
    log.info("  Signal threshold : NBR ≥ 1.39 at screening checkpoint (T-1)")
    log.info("  Entry timing     : %s (%d bdays before pre_capture_date)",
             best_row["entry"],
             ENTRY_OFFSETS[best_row["entry"]])
    log.info("  Exit timing      : %s post-earnings", best_row["exit"])
    log.info("  Position size    : 1–5 contracts (liquidity constrained)")
    log.info("  Exp. net P&L     : $%.2f/contract (realistic fills)",
             best_row["avg_net"])
    log.info("  Win rate         : %.0f%%", best_row["pct_positive"])
    log.info("  Trade frequency  : ~%d–%d qualifying events/year",
             max(1, int(best_row["n"] / 4 * 0.8)),
             int(best_row["n"] / 4 * 1.2))

    # ── save results ──────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / f"entry_exit_{AUDIT_VERSION}_{ts}.json"

    report = {
        "version":   AUDIT_VERSION,
        "timestamp": ts,
        "n_tradeable_universe": len(tradeable),
        "n_result_records": len(results),
        "comparison_table": comp.to_dict("records"),
        "nbr_timing_table": build_nbr_timing_table(results).to_dict("records"),
        "year_stability":   yr_table.to_dict("records"),
        "optimal": {
            "entry": best_row["entry"],
            "exit":  best_row["exit"],
            "avg_net_per_contract": round(float(best_row["avg_net"]), 2),
            "pct_profitable": round(float(best_row["pct_positive"]), 1),
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("")
    log.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
