"""
Stress-test and universe expansion for the earnings IV crush calendar spread.

Approach:
  1. Run one master scan at lenient filters (NBR ≥ 1.25, OI ≥ 50/25, spread ≤ 30%)
     — loads T-2 entry and T+1 exit chains for all candidates
  2. Post-process the master dataset with multiple filter configs
  3. Stress tests, sector breakdown, train/test split from the master trade log

Steps:
  1. Master scan + sector annotation
  2. Filter sensitivity (OI 75 / 100 / 150, spread 20% / 25%)
  3. Side-by-side comparison: original vs expanded universe
  4. Concentration analysis (top-N symbol contribution, sector P&L)
  5. Out-of-sample split: train 2023-2024 / test 2025
  6. Stress scenarios: remove top 3 / top 5 symbols
  7. Final verdict
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - \033[32m%(levelname)s\033[0m - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH      = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
PARQUET_BASE = Path("/Volumes/T9/market_data/research/options_features_eod")
REPORT_DIR   = Path.home() / ".options_calculator_pro" / "reports"
VERSION      = "stress_v1"

# Master scan parameters — intentionally lenient to capture full picture
MASTER_NBR_MIN      = 1.25    # below Q3 baseline; anything interesting above here
MASTER_FRONT_OI_MIN = 50
MASTER_BACK_OI_MIN  = 25
MASTER_SPREAD_MAX   = 0.30
ATM_WIN             = 5.0
ENTRY_BDAYS         = 2       # T-2 entry
EXIT_BDAYS          = 1       # T+1 exit
FILL_REALISTIC      = 0.75
NBR_Q4_PCTILE       = 0.60    # Q4 bottom boundary

# Filter configurations for sensitivity analysis
FILTER_CONFIGS = {
    "loose":    {"front_oi": 75,  "back_oi": 37,  "spread": 0.25, "label": "OI≥75/37 spread≤25%"},
    "base":     {"front_oi": 100, "back_oi": 50,  "spread": 0.20, "label": "OI≥100/50 spread≤20%"},
    "tight":    {"front_oi": 150, "back_oi": 75,  "spread": 0.15, "label": "OI≥150/75 spread≤15%"},
    "very_tight":{"front_oi":250, "back_oi": 125, "spread": 0.10, "label": "OI≥250/125 spread≤10%"},
}

# ── sector map ────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    # Technology
    "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","AMD":"Tech","AVGO":"Tech",
    "AMAT":"Tech","CSCO":"Tech","IBM":"Tech","ORCL":"Tech","CRM":"Tech",
    "INTU":"Tech","TXN":"Tech","LRCX":"Tech","KLAC":"Tech","MCHP":"Tech",
    "ADI":"Tech","QCOM":"Tech","NOW":"Tech","CDNS":"Tech","PANW":"Tech",
    "FTNT":"Tech","MSI":"Tech","CHTR":"Tech","CBOE":"Tech",
    # Communication Services
    "META":"Comm","GOOGL":"Comm","NFLX":"Comm","DIS":"Comm","CMCSA":"Comm",
    "T":"Comm","VZ":"Comm","BKNG":"Comm","UBER":"Comm","EA":"Comm",
    # Financials
    "BAC":"Fin","JPM":"Fin","GS":"Fin","MS":"Fin","C":"Fin","WFC":"Fin",
    "AXP":"Fin","V":"Fin","MA":"Fin","COF":"Fin","SCHW":"Fin","BLK":"Fin",
    "USB":"Fin","PNC":"Fin","MET":"Fin","AFL":"Fin","AIG":"Fin","AIZ":"Fin",
    "ICE":"Fin","CME":"Fin","MCO":"Fin","PGR":"Fin","ALL":"Fin","PRU":"Fin",
    "BR":"Fin",
    # Healthcare
    "JNJ":"Health","UNH":"Health","LLY":"Health","ABBV":"Health","MRK":"Health",
    "AMGN":"Health","GILD":"Health","REGN":"Health","BMY":"Health","ISRG":"Health",
    "TMO":"Health","DHR":"Health","GEHC":"Health","MDT":"Health","BAX":"Health",
    "BDX":"Health","IDXX":"Health","EW":"Health","BSX":"Health","ZTS":"Health",
    "BIIB":"Health","PFE":"Health","ABT":"Health","VRTX":"Health",
    # Consumer Discretionary
    "AMZN":"ConDisc","HD":"ConDisc","MCD":"ConDisc","SBUX":"ConDisc","NKE":"ConDisc",
    "TSLA":"ConDisc","LOW":"ConDisc","ROST":"ConDisc","BBY":"ConDisc","LULU":"ConDisc",
    "MAR":"ConDisc","AZO":"ConDisc","HLT":"ConDisc",
    # Consumer Staples
    "KO":"ConStap","PEP":"ConStap","WMT":"ConStap","PG":"ConStap","COST":"ConStap",
    "MO":"ConStap","PM":"ConStap","KHC":"ConStap","GIS":"ConStap","CL":"ConStap",
    "MNST":"ConStap",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","FCX":"Energy",
    "FANG":"Energy","OKE":"Energy",
    # Industrials
    "CAT":"Indust","DE":"Indust","HON":"Indust","GE":"Indust","GD":"Indust",
    "EMR":"Indust","RTX":"Indust","LMT":"Indust","NSC":"Indust","CSX":"Indust",
    "UPS":"Indust","FDX":"Indust","ETN":"Indust","HWM":"Indust","ITW":"Indust",
    "ROK":"Indust","CTAS":"Indust","NOC":"Indust","FAST":"Indust","DD":"Indust",
    "CARR":"Indust","LHX":"Indust","MMM":"Indust","DAL":"Indust","GIS":"Indust",
    # Materials
    "LIN":"Matl","NEM":"Matl","APD":"Matl",
    # Utilities
    "NEE":"Util","SO":"Util","DUK":"Util","AEP":"Util","EXC":"Util",
    "SRE":"Util","CEG":"Util",
    # REITs
    "EQIX":"REIT","CCI":"REIT","PSA":"REIT","AVB":"REIT","DGX":"REIT",
    # Catch-all
    "ACN":"Tech","ANET":"Tech",
}

def get_sector(sym: str) -> str:
    return SECTOR_MAP.get(sym, "Other")


# ── data helpers (identical to backtest_strategy.py) ─────────────────────────

def load_events() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT symbol, event_date, pre_capture_date, post_capture_date,
               front_expiry, back_expiry,
               front_dte_pre, pre_front_iv, pre_back_iv,
               post_front_iv, post_back_iv,
               front_iv_crush_pct, pre_underlying_price, post_underlying_price,
               quality_tier
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ('A', 'B')
        ORDER BY event_date
        """, conn,
    )
    conn.close()
    df["pre_dt"]  = pd.to_datetime(df["pre_capture_date"])
    df["post_dt"] = pd.to_datetime(df["post_capture_date"])
    df["nbr"]     = df["pre_front_iv"] / df["pre_back_iv"]
    df["year"]    = df["pre_dt"].dt.year
    df["sector"]  = df["symbol"].map(get_sector)
    return df


def _parquet_path(sym: str, dt: pd.Timestamp) -> Path:
    yr, mo = str(dt.year), str(dt.month).zfill(2)
    return (PARQUET_BASE / f"underlying_symbol={sym}"
            / f"year={yr}" / f"month={mo}"
            / f"{sym.lower()}_options_features_eod_{yr}-{mo}.parquet")


def _load_chain(sym: str, dt: pd.Timestamp) -> Optional[pd.DataFrame]:
    p = _parquet_path(sym, dt)
    if not p.exists():
        return None
    try:
        c = pd.read_parquet(p)
    except Exception:
        return None
    c["trade_date"] = pd.to_datetime(c["trade_date"])
    c["expiry"]     = pd.to_datetime(c["expiry"])
    day = c[(c["trade_date"] == dt) & (c["quote_quality_flag"] == "ok") & (c["bid"] > 0)]
    return day.copy() if len(day) > 0 else None


def _atm_call(chain: pd.DataFrame, exp: pd.Timestamp) -> Optional[pd.Series]:
    leg = chain[(chain["expiry"] == exp) & (chain["call_put"] == "C")
                & (chain["moneyness_pct"].abs() <= ATM_WIN)]
    if len(leg) == 0:
        return None
    return leg.loc[leg["moneyness_pct"].abs().idxmin()]


def _leg(opt: pd.Series) -> Dict:
    return {
        "mid": float(opt["mid"]), "bid": float(opt["bid"]), "ask": float(opt["ask"]),
        "spread": float(opt["spread"]), "sp_pct": float(opt["spread_pct"]),
        "oi": int(opt["open_interest"]), "iv": float(opt["iv"]), "dte": int(opt["dte"]),
    }


# ── walk-forward thresholds ────────────────────────────────────────────────────

def wf_thresholds(events: pd.DataFrame) -> Dict[int, float]:
    years = sorted(events["year"].unique())
    out = {}
    for yr in years:
        prior = events if yr == min(years) else events[events["year"] < yr]
        out[yr] = float(prior["nbr"].quantile(NBR_Q4_PCTILE))
    return out


# ── master scan ───────────────────────────────────────────────────────────────

def run_master_scan(events: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt every event above MASTER_NBR_MIN with lenient OI/spread filters.
    Loads T-1 (screening), T-2 (entry), T+1 (exit) chains.
    Returns a master trade record for all successfully loaded events.
    """
    thresh_map = wf_thresholds(events)
    records = []
    n_total = n_nbr_ok = n_t1_ok = n_entry_ok = n_exit_ok = 0

    for _, ev in events.iterrows():
        n_total += 1
        yr = int(ev["year"])
        nbr = float(ev["nbr"])

        if nbr < MASTER_NBR_MIN:
            continue
        n_nbr_ok += 1

        front_exp = pd.to_datetime(ev["front_expiry"])
        back_exp  = pd.to_datetime(ev["back_expiry"])

        # T-1 screening snapshot
        t1_chain = _load_chain(ev["symbol"], ev["pre_dt"])
        if t1_chain is None:
            continue
        f_t1 = _atm_call(t1_chain, front_exp)
        b_t1 = _atm_call(t1_chain, back_exp)
        if f_t1 is None or b_t1 is None:
            continue

        f1 = _leg(f_t1); b1 = _leg(b_t1)
        # Apply master (lenient) filters
        if f1["oi"] < MASTER_FRONT_OI_MIN or b1["oi"] < MASTER_BACK_OI_MIN:
            continue
        if f1["sp_pct"] > MASTER_SPREAD_MAX or b1["sp_pct"] > MASTER_SPREAD_MAX:
            continue
        n_t1_ok += 1

        # T-2 entry
        entry_dt = ev["pre_dt"] - pd.offsets.BDay(ENTRY_BDAYS)
        e_chain  = _load_chain(ev["symbol"], entry_dt)
        if e_chain is None:
            continue
        f_e = _atm_call(e_chain, front_exp)
        b_e = _atm_call(e_chain, back_exp)
        if f_e is None or b_e is None:
            continue
        fe = _leg(f_e); be = _leg(b_e)
        n_entry_ok += 1

        # T+1 exit
        exit_dt = ev["post_dt"] + pd.offsets.BDay(EXIT_BDAYS)
        x_chain = _load_chain(ev["symbol"], exit_dt)
        if x_chain is None:
            continue
        f_x = _atm_call(x_chain, front_exp)
        b_x = _atm_call(x_chain, back_exp)
        if f_x is None or b_x is None:
            continue
        fx = _leg(f_x); bx = _leg(b_x)
        n_exit_ok += 1

        # P&L at 1 contract (scale by n later)
        cal_entry = be["mid"] - fe["mid"]
        cal_exit  = bx["mid"] - fx["mid"]
        gross     = (cal_exit - cal_entry) * 100
        total_sp  = fe["spread"] + be["spread"] + fx["spread"] + bx["spread"]
        cost      = (total_sp / 2) * FILL_REALISTIC * 100
        net       = gross - cost
        cap       = max(cal_entry, 0.01) * 100   # max risk = debit paid

        records.append({
            "symbol":        ev["symbol"],
            "event_date":    ev["event_date"],
            "year":          yr,
            "sector":        ev["sector"],
            "nbr":           round(nbr, 4),
            "nbr_threshold": round(thresh_map.get(yr, 1.39), 4),
            # T-1 screening fields
            "t1_front_oi":   f1["oi"],
            "t1_back_oi":    b1["oi"],
            "t1_front_sp":   round(f1["sp_pct"], 4),
            "t1_back_sp":    round(b1["sp_pct"], 4),
            # Entry fields
            "entry_dt":      str(entry_dt.date()),
            "entry_front_sp":round(fe["sp_pct"], 4),
            "entry_back_sp": round(be["sp_pct"], 4),
            "cal_entry_mid": round(cal_entry, 3),
            # Exit fields
            "exit_dt":       str(exit_dt.date()),
            "exit_front_sp": round(fx["sp_pct"], 4),
            "exit_back_sp":  round(bx["sp_pct"], 4),
            "cal_exit_mid":  round(cal_exit, 3),
            # P&L (1 contract)
            "gross_1c":      round(gross, 2),
            "cost_1c":       round(cost, 2),
            "net_1c":        round(net, 2),
            "capital_1c":    round(cap, 2),
            "winner":        net > 0,
        })

    log.info("  Master scan: total=%d  nbr_ok=%d  t1_ok=%d  entry_ok=%d  exit_ok=%d",
             n_total, n_nbr_ok, n_t1_ok, n_entry_ok, n_exit_ok)
    return pd.DataFrame(records)


# ── filter application ────────────────────────────────────────────────────────

def apply_filters(master: pd.DataFrame, front_oi: int, back_oi: int,
                  spread: float, nbr_min: Optional[float] = None) -> pd.DataFrame:
    """Apply filter config to master scan. Returns qualifying subset."""
    m = master.copy()
    m = m[
        (m["t1_front_oi"] >= front_oi)
        & (m["t1_back_oi"] >= back_oi)
        & (m["t1_front_sp"] <= spread)
        & (m["t1_back_sp"] <= spread)
    ]
    if nbr_min is not None:
        m = m[m["nbr"] >= nbr_min]
    else:
        # Walk-forward Q4 threshold
        m = m[m["nbr"] >= m["nbr_threshold"]]
    return m


# ── portfolio metrics ─────────────────────────────────────────────────────────

def portfolio_metrics(trades: pd.DataFrame, n_contracts: int = 2,
                      label: str = "") -> Dict:
    """Compute full portfolio metrics from a filtered trade set."""
    if len(trades) == 0:
        return {"label": label, "n_trades": 0}

    net = trades["net_1c"] * n_contracts
    gross = trades["gross_1c"] * n_contracts
    cost = trades["cost_1c"] * n_contracts
    cap  = trades["capital_1c"] * n_contracts

    n_years = max(trades["year"].nunique(), 1)

    # Monthly P&L
    trades_cp = trades.copy()
    trades_cp["month"] = pd.to_datetime(trades_cp["event_date"]).dt.to_period("M")
    monthly = (net).groupby(trades_cp["month"]).sum()
    sharpe = (monthly.mean() / monthly.std() * math.sqrt(12)
              if len(monthly) > 1 and monthly.std() > 0 else float("nan"))

    # Max drawdown
    eq = monthly.cumsum().values
    if len(eq) > 0:
        peak = eq[0]; max_dd = 0.0
        for v in eq:
            peak = max(peak, v)
            max_dd = max(max_dd, peak - v)
    else:
        max_dd = 0.0

    # Top-5 symbol concentration
    sym_pnl = net.groupby(trades["symbol"]).sum().sort_values(ascending=False)
    total_pnl = net.sum()
    top5_pct = (sym_pnl.head(5).sum() / total_pnl * 100
                if total_pnl > 0 else float("nan"))
    top10_pct = (sym_pnl.head(10).sum() / total_pnl * 100
                 if total_pnl > 0 else float("nan"))

    # Peak capital
    trades_cp2 = trades.copy()
    trades_cp2["entry_dt_ts"] = pd.to_datetime(trades_cp2["entry_dt"])
    trades_cp2["exit_dt_ts"]  = pd.to_datetime(trades_cp2["exit_dt"])
    ev_list = (
        [(r["entry_dt_ts"], +r["capital_1c"] * n_contracts)
         for _, r in trades_cp2.iterrows()] +
        [(r["exit_dt_ts"],  -r["capital_1c"] * n_contracts)
         for _, r in trades_cp2.iterrows()]
    )
    ev_list.sort(key=lambda x: x[0])
    cur = peak_cap = 0.0
    for _, delta in ev_list:
        cur += delta
        peak_cap = max(peak_cap, cur)

    # Streaks
    results = (net > 0).astype(int).tolist()
    max_win = max_loss = cur_w = cur_l = 0
    for r in results:
        if r:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_win = max(max_win, cur_w); max_loss = max(max_loss, cur_l)

    return {
        "label":             label,
        "n_trades":          len(trades),
        "n_symbols":         trades["symbol"].nunique(),
        "trades_per_year":   round(len(trades) / n_years, 1),
        "n_years":           n_years,
        "total_net_pnl":     round(net.sum(), 2),
        "ann_net_pnl":       round(net.sum() / n_years, 2),
        "avg_net_trade":     round(net.mean(), 2),
        "median_net":        round(net.median(), 2),
        "win_rate":          round(100 * (net > 0).mean(), 1),
        "avg_winner":        round(net[net > 0].mean(), 2) if (net > 0).any() else 0,
        "avg_loser":         round(net[net <= 0].mean(), 2) if (net <= 0).any() else 0,
        "sharpe_monthly":    round(sharpe, 2) if not math.isnan(sharpe) else float("nan"),
        "max_drawdown":      round(max_dd, 2),
        "max_loss_streak":   max_loss,
        "worst_trade":       round(net.min(), 2),
        "best_trade":        round(net.max(), 2),
        "std_net":           round(net.std(), 2),
        "peak_capital":      round(peak_cap, 2),
        "roc_ann_pct":       round(net.sum() / n_years / max(peak_cap, 1) * 100, 1),
        "top5_pnl_pct":      round(top5_pct, 1),
        "top10_pnl_pct":     round(top10_pct, 1),
        "top_symbol":        sym_pnl.index[0] if len(sym_pnl) > 0 else "",
        "top_symbol_pct":    round(sym_pnl.iloc[0] / total_pnl * 100, 1)
                             if total_pnl > 0 and len(sym_pnl) > 0 else float("nan"),
    }


def _hr(n=80): log.info("─" * n)
def _eq(n=80): log.info("=" * n)

def _ptable(title: str, df: pd.DataFrame) -> None:
    log.info("")
    log.info(title)
    _hr()
    for line in df.to_string(index=False).split("\n"):
        log.info("  %s", line)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _eq()
    log.info("STRESS TEST & UNIVERSE EXPANSION  version=%s", VERSION)
    _eq()

    events = load_events()
    log.info("Loaded %d events, %d symbols, %d sectors",
             len(events), events["symbol"].nunique(), events["sector"].nunique())
    log.info("Sector distribution: %s",
             events["symbol"].map(get_sector).value_counts().to_dict())

    # ── Master scan ───────────────────────────────────────────────────────────
    log.info("")
    _eq()
    log.info("STEP 1 — Master scan (lenient filters, NBR ≥ %.2f)", MASTER_NBR_MIN)
    _eq()
    log.info("  Scanning all %d events with NBR ≥ %.2f ...", len(events), MASTER_NBR_MIN)
    master = run_master_scan(events)
    log.info("  Master dataset: %d tradeable events across %d symbols",
             len(master), master["symbol"].nunique())
    log.info("  Year breakdown: %s",
             master["year"].value_counts().sort_index().to_dict())
    log.info("  Sector breakdown: %s",
             master["sector"].value_counts().to_dict())

    # ── Base config (reproduces original backtest) ────────────────────────────
    base = apply_filters(master, front_oi=100, back_oi=50, spread=0.20)
    log.info("  Base config trades: %d  symbols: %d",
             len(base), base["symbol"].nunique())

    # ── Step 2: Filter sensitivity ────────────────────────────────────────────
    log.info("")
    _eq()
    log.info("STEP 2 — Filter sensitivity analysis")
    _eq()

    sensitivity_rows = []
    for cfg_name, cfg in FILTER_CONFIGS.items():
        subset = apply_filters(master,
                               front_oi=cfg["front_oi"], back_oi=cfg["back_oi"],
                               spread=cfg["spread"])
        m = portfolio_metrics(subset, n_contracts=2, label=cfg["label"])
        m["config"] = cfg_name
        sensitivity_rows.append(m)

    sens_df = pd.DataFrame(sensitivity_rows)[
        ["config", "label", "n_trades", "n_symbols", "trades_per_year",
         "ann_net_pnl", "avg_net_trade", "win_rate", "sharpe_monthly",
         "max_drawdown", "top5_pnl_pct", "peak_capital"]
    ]
    _ptable("  Filter sensitivity (2 contracts/trade, Q4-Q5 NBR, walk-forward thresholds):",
            sens_df)

    # ── Step 3: Side-by-side original vs expanded ─────────────────────────────
    log.info("")
    _eq()
    log.info("STEP 3 — Original vs expanded universe comparison")
    _eq()

    # "Expanded" = loose config
    loose_subset = apply_filters(master, front_oi=75, back_oi=37, spread=0.25)
    base_metrics = portfolio_metrics(base, n_contracts=2, label="Original (base filters)")
    exp_metrics  = portfolio_metrics(loose_subset, n_contracts=2, label="Expanded (loose filters)")

    compare_keys = [
        "n_trades", "n_symbols", "trades_per_year",
        "ann_net_pnl", "avg_net_trade", "win_rate",
        "sharpe_monthly", "max_drawdown", "max_loss_streak",
        "top5_pnl_pct", "peak_capital",
    ]
    log.info("")
    log.info("  %-28s %18s %18s", "Metric", "Original", "Expanded")
    log.info("  " + "─" * 68)
    for k in compare_keys:
        orig_v = base_metrics.get(k, "—")
        exp_v  = exp_metrics.get(k, "—")
        if isinstance(orig_v, float) and math.isnan(orig_v):
            orig_v = "nan"
        if isinstance(exp_v, float) and math.isnan(exp_v):
            exp_v = "nan"
        log.info("  %-28s %18s %18s", k, orig_v, exp_v)

    # ── Step 4: Concentration analysis ────────────────────────────────────────
    log.info("")
    _eq()
    log.info("STEP 4 — Concentration analysis")
    _eq()

    for label, subset in [("Base (100/50/20%)", base),
                           ("Expanded (75/37/25%)", loose_subset)]:
        net = subset["net_1c"] * 2
        total = net.sum()
        sym_pnl = net.groupby(subset["symbol"]).sum().sort_values(ascending=False)
        sec_pnl = net.groupby(subset["sector"]).sum().sort_values(ascending=False)
        cum_pct = 0.0

        log.info("")
        log.info("  %s  (N=%d trades, total=$%.2f):", label, len(subset), total)
        log.info("  %-8s  %7s  %9s  %8s  %9s",
                 "Symbol", "Trades", "Net P&L", "% total", "Cumul %")
        _hr(60)
        for sym, pnl in sym_pnl.head(15).items():
            n = (subset["symbol"] == sym).sum()
            pct = pnl / total * 100 if total else float("nan")
            cum_pct += pct
            log.info("  %-8s  %7d  %9.2f  %8.1f%%  %9.1f%%",
                     sym, n, pnl, pct, cum_pct)

        log.info("")
        log.info("  Sector P&L breakdown:")
        log.info("  %-12s  %7s  %9s  %8s",
                 "Sector", "Trades", "Net P&L", "% total")
        _hr(50)
        for sec, pnl in sec_pnl.items():
            n = (subset["sector"] == sec).sum()
            pct = pnl / total * 100 if total else float("nan")
            log.info("  %-12s  %7d  %9.2f  %8.1f%%", sec, n, pnl, pct)

    # ── Step 5: Out-of-sample split (train 2023-2024, test 2025) ──────────────
    log.info("")
    _eq()
    log.info("STEP 5 — Out-of-sample split: train 2023-2024 / test 2025")
    _eq()

    for label, subset in [("Base", base), ("Expanded", loose_subset)]:
        train = subset[subset["year"].isin([2023, 2024])]
        test  = subset[subset["year"] == 2025]

        train_m = portfolio_metrics(train, 2, f"{label} TRAIN 2023-2024")
        test_m  = portfolio_metrics(test,  2, f"{label} TEST  2025")

        log.info("")
        log.info("  %s:", label)
        log.info("  %-26s %8s %8s %8s %9s %8s %9s",
                 "Period", "Trades", "Ann P&L", "Avg/trd", "Win%", "Sharpe", "MaxDD")
        _hr(75)
        for m in [train_m, test_m]:
            log.info("  %-26s %8d %8.2f %8.2f %9.1f%% %8.2f %9.2f",
                     m["label"], m["n_trades"],
                     m["ann_net_pnl"], m["avg_net_trade"],
                     m["win_rate"],
                     m.get("sharpe_monthly", float("nan")),
                     m["max_drawdown"])

    # ── Step 6: Year-by-year detail ────────────────────────────────────────────
    log.info("")
    _eq()
    log.info("STEP 6 — Year-by-year detail (base vs expanded)")
    _eq()

    for label, subset in [("Base", base), ("Expanded", loose_subset)]:
        log.info("")
        log.info("  %s:", label)
        log.info("  %-6s %8s %8s %9s %9s %9s %8s",
                 "Year", "Trades", "Symbols", "GrossPnL", "NetPnL", "Avg/trd", "Win%")
        _hr(70)
        for yr in sorted(subset["year"].unique()):
            t = subset[subset["year"] == yr]
            net = t["net_1c"] * 2
            log.info("  %-6d %8d %8d %9.2f %9.2f %9.2f %8.1f%%",
                     yr, len(t), t["symbol"].nunique(),
                     (t["gross_1c"] * 2).sum(), net.sum(),
                     net.mean(), 100 * (net > 0).mean())

    # ── Step 7: Stress scenarios ───────────────────────────────────────────────
    log.info("")
    _eq()
    log.info("STEP 7 — Stress scenarios")
    _eq()

    # Identify top symbols in base config by total P&L
    base_net = base["net_1c"] * 2
    base_sym_pnl = base_net.groupby(base["symbol"]).sum().sort_values(ascending=False)
    top3 = list(base_sym_pnl.head(3).index)
    top5 = list(base_sym_pnl.head(5).index)

    log.info("  Top 5 symbols in base: %s", top5)
    log.info("  Their combined P&L: $%.2f / $%.2f total (%.1f%%)",
             base_sym_pnl.head(5).sum(), base_net.sum(),
             100 * base_sym_pnl.head(5).sum() / base_net.sum())

    scenarios = [
        ("Base (all symbols)",           base),
        (f"Remove top 3 ({', '.join(top3)})", base[~base["symbol"].isin(top3)]),
        (f"Remove top 5 ({', '.join(top5)})", base[~base["symbol"].isin(top5)]),
    ]

    # Also run on expanded for same stress
    exp_net = loose_subset["net_1c"] * 2
    exp_sym_pnl = exp_net.groupby(loose_subset["symbol"]).sum().sort_values(ascending=False)
    exp_top5 = list(exp_sym_pnl.head(5).index)

    log.info("")
    log.info("  Stress results (base config, 2 contracts/trade):")
    log.info("  %-42s %8s %9s %9s %9s %8s %9s",
             "Scenario", "Trades", "TotalPnL", "Ann PnL", "Avg/trd", "Win%", "Sharpe")
    _hr(100)
    for s_label, s_subset in scenarios:
        m = portfolio_metrics(s_subset, 2, s_label)
        log.info("  %-42s %8d %9.2f %9.2f %9.2f %8.1f%% %9.2f",
                 s_label, m["n_trades"],
                 m["total_net_pnl"], m["ann_net_pnl"],
                 m["avg_net_trade"], m["win_rate"],
                 m.get("sharpe_monthly", float("nan")))

    log.info("")
    log.info("  Per-year breakdown with top 5 removed (base config):")
    no_top5 = base[~base["symbol"].isin(top5)]
    log.info("  %-6s %8s %9s %9s %8s",
             "Year", "Trades", "Net PnL", "Avg/trd", "Win%")
    _hr(50)
    for yr in sorted(no_top5["year"].unique()):
        t = no_top5[no_top5["year"] == yr]
        net = t["net_1c"] * 2
        log.info("  %-6d %8d %9.2f %9.2f %8.1f%%",
                 yr, len(t), net.sum(), net.mean() if len(t) > 0 else 0,
                 100 * (net > 0).mean() if len(t) > 0 else 0)

    # Also stress on expanded universe
    log.info("")
    log.info("  Same stress on expanded universe (75/37/25%%):")
    exp_scenarios = [
        ("Expanded (all)",                   loose_subset),
        (f"Expanded − top 5 ({', '.join(exp_top5)})",
         loose_subset[~loose_subset["symbol"].isin(exp_top5)]),
    ]
    log.info("  %-46s %8s %9s %9s %8s %9s",
             "Scenario", "Trades", "Ann PnL", "Avg/trd", "Win%", "Sharpe")
    _hr(105)
    for s_label, s_subset in exp_scenarios:
        m = portfolio_metrics(s_subset, 2, s_label)
        log.info("  %-46s %8d %9.2f %9.2f %8.1f%% %9.2f",
                 s_label, m["n_trades"],
                 m["ann_net_pnl"], m["avg_net_trade"],
                 m["win_rate"], m.get("sharpe_monthly", float("nan")))

    # ── Step 8: Final verdict ─────────────────────────────────────────────────
    log.info("")
    _eq()
    log.info("STEP 8 — FINAL VERDICT")
    _eq()

    base_m  = portfolio_metrics(base, 2, "Base")
    exp_m   = portfolio_metrics(loose_subset, 2, "Expanded")
    no5_m   = portfolio_metrics(base[~base["symbol"].isin(top5)], 2, "Base−top5")
    exp_no5 = portfolio_metrics(
        loose_subset[~loose_subset["symbol"].isin(exp_top5)], 2, "Exp−top5")

    log.info("")
    log.info("  %-28s %10s %10s %10s %10s",
             "", "Base", "Expanded", "Base−top5", "Exp−top5")
    log.info("  " + "─" * 68)
    verdict_rows = [
        ("Trades/year",     "trades_per_year"),
        ("Ann. net P&L",    "ann_net_pnl"),
        ("Avg P&L/trade",   "avg_net_trade"),
        ("Win rate",        "win_rate"),
        ("Sharpe (monthly)","sharpe_monthly"),
        ("Max drawdown $",  "max_drawdown"),
        ("Max loss streak", "max_loss_streak"),
        ("Top-5 sym %",     "top5_pnl_pct"),
        ("N symbols",       "n_symbols"),
        ("Peak capital",    "peak_capital"),
    ]
    for row_label, key in verdict_rows:
        vals = [base_m, exp_m, no5_m, exp_no5]
        parts = []
        for m in vals:
            v = m.get(key, "—")
            if isinstance(v, float) and math.isnan(v):
                parts.append("  nan")
            elif isinstance(v, float):
                parts.append(f"{v:10.2f}")
            else:
                parts.append(f"{v:>10}")
        log.info("  %-28s %s", row_label, "".join(parts))

    log.info("")
    log.info("  KEY FINDINGS:")
    log.info("")

    # Finding 1: edge persistence
    if exp_m["win_rate"] >= 70 and exp_m["ann_net_pnl"] > 0:
        log.info("  1. EDGE PERSISTS in expanded universe (win rate %.1f%%, Sharpe %.2f)",
                 exp_m["win_rate"], exp_m.get("sharpe_monthly", float("nan")))
    else:
        log.info("  1. EDGE DEGRADES in expanded universe (win rate %.1f%%)",
                 exp_m["win_rate"])

    # Finding 2: concentration
    if no5_m["ann_net_pnl"] > 0 and no5_m["win_rate"] >= 60:
        log.info("  2. STRATEGY SURVIVES without top 5 symbols (ann=$%.2f, win=%.1f%%)",
                 no5_m["ann_net_pnl"], no5_m["win_rate"])
    else:
        log.info("  2. STRATEGY IS DEPENDENT on top 5 symbols (remainder: ann=$%.2f, win=%.1f%%)",
                 no5_m["ann_net_pnl"], no5_m["win_rate"])

    # Finding 3: scaling
    log.info("  3. TRADE FREQUENCY: base=%.1f/yr  expanded=%.1f/yr",
             base_m["trades_per_year"], exp_m["trades_per_year"])

    # Finding 4: recommendation
    log.info("")
    log.info("  RECOMMENDED UNIVERSE DEFINITION:")
    # Pick the config with best Sharpe that has >20 trades/year
    best_cfg = None
    best_sharpe = -999
    for cfg_name, cfg in FILTER_CONFIGS.items():
        sub = apply_filters(master, front_oi=cfg["front_oi"],
                            back_oi=cfg["back_oi"], spread=cfg["spread"])
        m = portfolio_metrics(sub, 2)
        if (m["trades_per_year"] >= 15
                and not math.isnan(m.get("sharpe_monthly", float("nan")))
                and m["sharpe_monthly"] > best_sharpe
                and m["win_rate"] >= 70):
            best_sharpe = m["sharpe_monthly"]
            best_cfg = (cfg_name, cfg, m)

    if best_cfg:
        cfg_name, cfg, m = best_cfg
        log.info("    Filter config:  %s", cfg["label"])
        log.info("    Trades/year:    %.1f", m["trades_per_year"])
        log.info("    Ann. P&L:       $%.2f", m["ann_net_pnl"])
        log.info("    Win rate:       %.1f%%", m["win_rate"])
        log.info("    Sharpe:         %.2f", m["sharpe_monthly"])
        log.info("    Top-5 conc.:    %.1f%%", m["top5_pnl_pct"])

    # ── save ──────────────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / f"stress_{VERSION}_{ts}.json"

    report = {
        "version":   VERSION,
        "timestamp": ts,
        "master_n":  len(master),
        "base_metrics":     base_m,
        "expanded_metrics": exp_m,
        "no_top5_metrics":  no5_m,
        "exp_no_top5":      exp_no5,
        "sensitivity": sensitivity_rows,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("")
    log.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
