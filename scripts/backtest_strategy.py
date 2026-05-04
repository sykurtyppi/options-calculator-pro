"""
Portfolio-level walk-forward backtest for the earnings IV crush calendar spread.

Protocol (from entry/exit optimization):
  - Screen at T-1:  front OI ≥ 100, back OI ≥ 50, spread ≤ 20%, NBR ≥ Q4 threshold
  - Entry at T-2:   enter calendar spread 2 business days before pre_capture_date
  - Exit  at T+1:   exit 1 business day after post_capture_date

Walk-forward discipline:
  - NBR quintile thresholds computed on PRIOR years only
  - 2023: use full-sample threshold (cold start — no prior data)
  - 2024: threshold from 2023 events
  - 2025: threshold from 2023+2024 events

P&L: actual option price change (entry and exit chains), realistic fills.
Capital: calendar entry debit × 100 × n_contracts per position.

Outputs:
  - Full trade log (every trade)
  - Monthly P&L series and equity curve
  - Annual performance summary (P&L, Sharpe, drawdown, win rate)
  - Simultaneous position analysis
  - Capital efficiency metrics
  - Sensitivity: 1, 2, 5 contracts per trade
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
VERSION       = "backtest_v1"

# Universe filters — screened at T-1
MIN_FRONT_OI   = 100
MIN_BACK_OI    = 50
MAX_SPREAD_PCT = 0.20
NBR_Q4_PCTILE  = 0.60    # Q4+Q5 = top 40% of NBR distribution

# Execution
ENTRY_OFFSET_BDAYS = 2    # T-2 entry: 2 BDays before pre_capture_date
EXIT_OFFSET_BDAYS  = 1    # T+1 exit:  1 BDay after post_capture_date
FILL_REALISTIC     = 0.75  # pay 75% of half-spread per leg
ATM_MONEYNESS_WIN  = 5.0

# Position sizing scenarios
SIZE_SCENARIOS = [1, 2, 5]
BASE_SIZE      = 2   # contracts per trade for primary analysis

# Risk-free rate for Sharpe (annualised)
RISK_FREE_ANNUAL = 0.045


# ── data loading ──────────────────────────────────────────────────────────────

def load_events() -> pd.DataFrame:
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
               quality_tier
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ('A', 'B')
        ORDER BY event_date
        """,
        conn,
    )
    conn.close()
    df["pre_dt"]   = pd.to_datetime(df["pre_capture_date"])
    df["post_dt"]  = pd.to_datetime(df["post_capture_date"])
    df["nbr"]      = df["pre_front_iv"] / df["pre_back_iv"]
    df["year"]     = df["pre_dt"].dt.year
    df["month"]    = df["pre_dt"].dt.to_period("M")
    return df


def _parquet_path(symbol: str, dt: pd.Timestamp) -> Path:
    yr = str(dt.year)
    mo = str(dt.month).zfill(2)
    return (PARQUET_BASE / f"underlying_symbol={symbol}"
            / f"year={yr}" / f"month={mo}"
            / f"{symbol.lower()}_options_features_eod_{yr}-{mo}.parquet")


def _load_day_chain(symbol: str, dt: pd.Timestamp) -> Optional[pd.DataFrame]:
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


def _find_atm_call(chain: pd.DataFrame, expiry: pd.Timestamp
                   ) -> Optional[pd.Series]:
    leg = chain[
        (chain["expiry"] == expiry)
        & (chain["call_put"] == "C")
        & (chain["moneyness_pct"].abs() <= ATM_MONEYNESS_WIN)
    ]
    if len(leg) == 0:
        return None
    return leg.loc[leg["moneyness_pct"].abs().idxmin()]


# ── walk-forward NBR thresholds ────────────────────────────────────────────────

def compute_wf_thresholds(events: pd.DataFrame) -> Dict[int, float]:
    """
    Return walk-forward Q4+ NBR threshold for each year.
    Year 2023 (cold start): computed on full sample (no prior data).
    Year Y (Y > 2023): computed on events from years < Y.
    """
    years = sorted(events["year"].unique())
    thresholds: Dict[int, float] = {}
    for yr in years:
        if yr == min(years):
            prior = events  # cold start: use all data
        else:
            prior = events[events["year"] < yr]
        thresholds[yr] = float(prior["nbr"].quantile(NBR_Q4_PCTILE))
    return thresholds


# ── single-event snapshot ─────────────────────────────────────────────────────

def get_t1_snapshot(ev: pd.Series) -> Optional[Dict]:
    """Load T-1 chains to screen liquidity and check filters."""
    front_exp = pd.to_datetime(ev["front_expiry"])
    back_exp  = pd.to_datetime(ev["back_expiry"])
    chain = _load_day_chain(ev["symbol"], ev["pre_dt"])
    if chain is None:
        return None
    f = _find_atm_call(chain, front_exp)
    b = _find_atm_call(chain, back_exp)
    if f is None or b is None:
        return None
    return {
        "front_oi":       int(f["open_interest"]),
        "back_oi":        int(b["open_interest"]),
        "front_sp_pct":   float(f["spread_pct"]),
        "back_sp_pct":    float(b["spread_pct"]),
        "front_iv":       float(f["iv"]),
        "back_iv":        float(b["iv"]),
        "nbr_t1":         float(f["iv"]) / float(b["iv"]),
    }


def get_entry_snapshot(ev: pd.Series) -> Optional[Dict]:
    """Load T-2 entry chain."""
    entry_dt  = ev["pre_dt"] - pd.offsets.BDay(ENTRY_OFFSET_BDAYS)
    front_exp = pd.to_datetime(ev["front_expiry"])
    back_exp  = pd.to_datetime(ev["back_expiry"])
    chain = _load_day_chain(ev["symbol"], entry_dt)
    if chain is None:
        return None
    f = _find_atm_call(chain, front_exp)
    b = _find_atm_call(chain, back_exp)
    if f is None or b is None:
        return None
    return {
        "entry_dt":       entry_dt,
        "front_mid":      float(f["mid"]),
        "front_spread":   float(f["spread"]),
        "front_sp_pct":   float(f["spread_pct"]),
        "front_dte":      int(f["dte"]),
        "front_iv":       float(f["iv"]),
        "back_mid":       float(b["mid"]),
        "back_spread":    float(b["spread"]),
        "back_sp_pct":    float(b["spread_pct"]),
        "back_dte":       int(b["dte"]),
        "back_iv":        float(b["iv"]),
        "cal_entry_mid":  float(b["mid"]) - float(f["mid"]),
        "nbr_entry":      float(f["iv"]) / float(b["iv"]),
    }


def get_exit_snapshot(ev: pd.Series) -> Optional[Dict]:
    """Load T+1 exit chain."""
    exit_dt   = ev["post_dt"] + pd.offsets.BDay(EXIT_OFFSET_BDAYS)
    front_exp = pd.to_datetime(ev["front_expiry"])
    back_exp  = pd.to_datetime(ev["back_expiry"])
    chain = _load_day_chain(ev["symbol"], exit_dt)
    if chain is None:
        return None
    f = _find_atm_call(chain, front_exp)
    b = _find_atm_call(chain, back_exp)
    if f is None or b is None:
        return None
    return {
        "exit_dt":        exit_dt,
        "front_mid":      float(f["mid"]),
        "front_spread":   float(f["spread"]),
        "front_sp_pct":   float(f["spread_pct"]),
        "front_iv":       float(f["iv"]),
        "back_mid":       float(b["mid"]),
        "back_spread":    float(b["spread"]),
        "back_sp_pct":    float(b["spread_pct"]),
        "back_iv":        float(b["iv"]),
        "cal_exit_mid":   float(b["mid"]) - float(f["mid"]),
    }


# ── P&L computation ───────────────────────────────────────────────────────────

def compute_trade_pnl(entry: Dict, exit_: Dict, n_contracts: int) -> Dict:
    gross_per = (exit_["cal_exit_mid"] - entry["cal_entry_mid"]) * 100
    total_spread = (entry["front_spread"] + entry["back_spread"]
                    + exit_["front_spread"] + exit_["back_spread"])
    cost_per  = (total_spread / 2) * FILL_REALISTIC * 100
    net_per   = gross_per - cost_per
    # Capital at risk = entry calendar debit (max loss on defined-risk spread)
    capital_per = max(entry["cal_entry_mid"], 0.01) * 100
    return {
        "gross_per_contract":   round(gross_per, 2),
        "cost_per_contract":    round(cost_per, 2),
        "net_per_contract":     round(net_per, 2),
        "gross_total":          round(gross_per * n_contracts, 2),
        "cost_total":           round(cost_per * n_contracts, 2),
        "net_total":            round(net_per * n_contracts, 2),
        "capital_deployed":     round(capital_per * n_contracts, 2),
        "winner":               gross_per > 0,
        "net_winner":           net_per > 0,
    }


# ── screening + execution ────────────────────────────────────────────────────

def run_backtest(events: pd.DataFrame, n_contracts: int = BASE_SIZE
                 ) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the full walk-forward backtest.
    Returns (trade_log_df, metadata_dict).
    """
    wf_thresholds = compute_wf_thresholds(events)
    log.info("  Walk-forward NBR thresholds: %s",
             {yr: f"{th:.4f}" for yr, th in wf_thresholds.items()})

    trades: List[Dict] = []
    n_total = n_screened = n_entry_ok = n_exit_ok = 0
    skip_reasons: Dict[str, int] = {}

    for _, ev in events.iterrows():
        n_total += 1
        yr    = int(ev["year"])
        nbr   = float(ev["nbr"])
        thresh = wf_thresholds.get(yr, wf_thresholds[min(wf_thresholds)])

        # ── T-1 screening ──────────────────────────────────────────────────
        if nbr < thresh:
            skip_reasons["nbr_below_threshold"] = skip_reasons.get("nbr_below_threshold", 0) + 1
            continue

        t1 = get_t1_snapshot(ev)
        if t1 is None:
            skip_reasons["no_t1_data"] = skip_reasons.get("no_t1_data", 0) + 1
            continue

        if t1["front_oi"] < MIN_FRONT_OI:
            skip_reasons["front_oi_low"] = skip_reasons.get("front_oi_low", 0) + 1
            continue
        if t1["back_oi"] < MIN_BACK_OI:
            skip_reasons["back_oi_low"] = skip_reasons.get("back_oi_low", 0) + 1
            continue
        if t1["front_sp_pct"] > MAX_SPREAD_PCT:
            skip_reasons["front_spread_wide"] = skip_reasons.get("front_spread_wide", 0) + 1
            continue
        if t1["back_sp_pct"] > MAX_SPREAD_PCT:
            skip_reasons["back_spread_wide"] = skip_reasons.get("back_spread_wide", 0) + 1
            continue
        n_screened += 1

        # ── T-2 entry ──────────────────────────────────────────────────────
        entry = get_entry_snapshot(ev)
        if entry is None:
            skip_reasons["no_entry_data"] = skip_reasons.get("no_entry_data", 0) + 1
            continue
        n_entry_ok += 1

        # ── T+1 exit ───────────────────────────────────────────────────────
        exit_ = get_exit_snapshot(ev)
        if exit_ is None:
            skip_reasons["no_exit_data"] = skip_reasons.get("no_exit_data", 0) + 1
            continue
        n_exit_ok += 1

        pnl = compute_trade_pnl(entry, exit_, n_contracts)

        trades.append({
            "symbol":          ev["symbol"],
            "event_date":      ev["event_date"],
            "year":            yr,
            "month":           str(ev["month"]),
            "nbr_t1":          round(float(t1["nbr_t1"]), 4),
            "nbr_entry":       round(float(entry["nbr_entry"]), 4),
            "nbr_threshold":   round(thresh, 4),
            "entry_dt":        str(entry["entry_dt"].date()),
            "exit_dt":         str(exit_["exit_dt"].date()),
            "entry_front_dte": entry["front_dte"],
            "entry_front_sp%": round(entry["front_sp_pct"] * 100, 2),
            "entry_back_sp%":  round(entry["back_sp_pct"] * 100, 2),
            "exit_front_sp%":  round(exit_["front_sp_pct"] * 100, 2),
            "exit_back_sp%":   round(exit_["back_sp_pct"] * 100, 2),
            "cal_entry_mid":   round(entry["cal_entry_mid"], 3),
            "cal_exit_mid":    round(exit_["cal_exit_mid"], 3),
            "front_iv_crush":  round(float(ev["front_iv_crush_pct"]), 4),
            "n_contracts":     n_contracts,
            **pnl,
        })

    log.info("  Pipeline: total=%d  nbr_pass=%d  screened=%d  entry_ok=%d  traded=%d",
             n_total, n_total - skip_reasons.get("nbr_below_threshold", 0),
             n_screened, n_entry_ok, n_exit_ok)
    log.info("  Skip breakdown: %s", skip_reasons)

    meta = {
        "n_total":        n_total,
        "n_screened":     n_screened,
        "n_traded":       n_exit_ok,
        "n_contracts":    n_contracts,
        "skip_reasons":   skip_reasons,
        "wf_thresholds":  {str(k): round(v, 4) for k, v in wf_thresholds.items()},
    }
    return pd.DataFrame(trades), meta


# ── portfolio-level analytics ────────────────────────────────────────────────

def annual_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in sorted(trades["year"].unique()):
        t = trades[trades["year"] == yr]
        net = t["net_total"]
        gross = t["gross_total"]
        rows.append({
            "year":           yr,
            "n_trades":       len(t),
            "n_symbols":      t["symbol"].nunique(),
            "gross_total":    round(gross.sum(), 2),
            "cost_total":     round(t["cost_total"].sum(), 2),
            "net_total":      round(net.sum(), 2),
            "avg_net_trade":  round(net.mean(), 2),
            "median_net":     round(net.median(), 2),
            "win_rate":       round(100 * t["net_winner"].mean(), 1),
            "gross_win_rate": round(100 * t["winner"].mean(), 1),
            "best_trade":     round(net.max(), 2),
            "worst_trade":    round(net.min(), 2),
            "std_net":        round(net.std(), 2),
        })
    return pd.DataFrame(rows)


def monthly_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Monthly P&L series — used for Sharpe and drawdown."""
    m = trades.groupby("month")["net_total"].sum().reset_index()
    m.columns = ["month", "net_pnl"]
    m["cumulative"] = m["net_pnl"].cumsum()
    return m


def compute_sharpe(monthly: pd.DataFrame) -> float:
    """Monthly Sharpe ratio, annualised."""
    rf_monthly = RISK_FREE_ANNUAL / 12
    excess = monthly["net_pnl"]  # dollar P&L, not return — Sharpe on raw P&L
    if len(excess) < 2 or excess.std() == 0:
        return float("nan")
    return float((excess.mean() - rf_monthly * excess.mean().clip(0)) /
                 excess.std() * math.sqrt(12))


def compute_sharpe_per_trade(trades: pd.DataFrame) -> float:
    """Per-trade Sharpe (annualised using trade count)."""
    net = trades["net_total"]
    if len(net) < 2 or net.std() == 0:
        return float("nan")
    trades_per_year = len(net) / max(trades["year"].nunique(), 1)
    return float(net.mean() / net.std() * math.sqrt(trades_per_year))


def compute_drawdown(monthly: pd.DataFrame) -> Dict:
    """Max drawdown from equity curve."""
    equity = monthly["cumulative"].values
    peak   = equity[0]
    max_dd = 0.0
    max_dd_start_idx = 0
    max_dd_end_idx   = 0
    cur_peak_idx     = 0

    for i, val in enumerate(equity):
        if val > peak:
            peak = val
            cur_peak_idx = i
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
            max_dd_start_idx = cur_peak_idx
            max_dd_end_idx   = i

    months = monthly["month"].tolist()
    return {
        "max_drawdown_dollars":   round(max_dd, 2),
        "dd_start_month":         str(months[max_dd_start_idx]) if months else "",
        "dd_end_month":           str(months[max_dd_end_idx])   if months else "",
        "dd_duration_months":     max_dd_end_idx - max_dd_start_idx,
        "final_equity":           round(float(equity[-1]), 2) if len(equity) > 0 else 0.0,
    }


def compute_streaks(trades: pd.DataFrame) -> Dict:
    """Longest winning and losing streaks (by trade)."""
    results = trades["net_winner"].astype(int).tolist()
    max_win = max_loss = cur_win = cur_loss = 0
    for r in results:
        if r == 1:
            cur_win  += 1
            cur_loss  = 0
        else:
            cur_loss += 1
            cur_win   = 0
        max_win  = max(max_win,  cur_win)
        max_loss = max(max_loss, cur_loss)
    return {"max_win_streak": max_win, "max_loss_streak": max_loss}


def simultaneous_positions(trades: pd.DataFrame) -> Dict:
    """
    Compute max number of simultaneously open positions and peak capital deployed.
    A position is 'open' from entry_dt to exit_dt inclusive.
    """
    events_list = []
    for _, t in trades.iterrows():
        events_list.append((pd.to_datetime(t["entry_dt"]), +1,
                            t["capital_deployed"]))
        events_list.append((pd.to_datetime(t["exit_dt"]), -1,
                            t["capital_deployed"]))
    events_list.sort(key=lambda x: x[0])

    max_open = 0
    cur_open = 0
    cur_cap  = 0.0
    max_cap  = 0.0

    for dt, delta, cap in events_list:
        cur_open += delta
        cur_cap  += delta * cap
        max_open  = max(max_open, cur_open)
        max_cap   = max(max_cap, cur_cap)

    return {
        "max_simultaneous_positions": max_open,
        "peak_capital_deployed":      round(max_cap, 2),
        "avg_capital_deployed":       round(trades["capital_deployed"].mean(), 2),
        "total_capital_deployed":     round(trades["capital_deployed"].sum(), 2),
    }


def capital_efficiency(trades: pd.DataFrame, sim: Dict) -> Dict:
    """Return on peak capital (key metric for sizing)."""
    net_total = trades["net_total"].sum()
    peak_cap  = sim["peak_capital_deployed"]
    avg_cap   = sim["avg_capital_deployed"]
    n_years   = trades["year"].nunique()
    if peak_cap > 0:
        return_on_peak = net_total / peak_cap * 100 / n_years  # annualised %
    else:
        return_on_peak = float("nan")
    return {
        "total_net_pnl":              round(net_total, 2),
        "annualised_net_pnl":         round(net_total / n_years, 2),
        "return_on_peak_capital_pct": round(return_on_peak, 1),
        "peak_capital":               round(peak_cap, 2),
        "n_years":                    n_years,
    }


# ── printing helpers ──────────────────────────────────────────────────────────

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
    log.info("PORTFOLIO BACKTEST  version=%s", VERSION)
    _eq()
    log.info("  Protocol : T-2 entry / T+1 exit / Q4-Q5 NBR / OI+spread filters")
    log.info("  P&L      : actual option price change, realistic fills (75%% half-spread)")
    log.info("  Base size: %d contracts per trade", BASE_SIZE)
    log.info("")

    events = load_events()
    log.info("Loaded %d events, %d symbols, years %s",
             len(events), events["symbol"].nunique(),
             sorted(events["year"].unique().tolist()))

    # ── primary backtest at BASE_SIZE ──────────────────────────────────────
    log.info("")
    _eq()
    log.info("STEP 1 — Walk-forward backtest (%d contracts/trade)", BASE_SIZE)
    _eq()
    trades, meta = run_backtest(events, n_contracts=BASE_SIZE)

    if len(trades) == 0:
        log.error("No trades executed — check data and filters.")
        return

    log.info("  Executed %d trades across %d symbols",
             len(trades), trades["symbol"].nunique())

    # ── per-year summary ───────────────────────────────────────────────────
    log.info("")
    log.info("STEP 2 — Annual performance summary")
    _hr()
    ann = annual_summary(trades)
    _ptable("  Year-by-year results:", ann)

    # ── monthly P&L and equity curve ───────────────────────────────────────
    log.info("")
    log.info("STEP 3 — Monthly P&L and equity curve")
    _hr()
    mon = monthly_pnl(trades)

    log.info("  Monthly P&L statistics:")
    log.info("    Profitable months:  %d / %d  (%.0f%%)",
             (mon["net_pnl"] > 0).sum(), len(mon),
             100 * (mon["net_pnl"] > 0).mean())
    log.info("    Avg monthly P&L:    $%.2f", mon["net_pnl"].mean())
    log.info("    Std monthly P&L:    $%.2f", mon["net_pnl"].std())
    log.info("    Best month:         $%.2f  (%s)",
             mon["net_pnl"].max(),
             mon.loc[mon["net_pnl"].idxmax(), "month"])
    log.info("    Worst month:        $%.2f  (%s)",
             mon["net_pnl"].min(),
             mon.loc[mon["net_pnl"].idxmin(), "month"])

    log.info("")
    log.info("  Equity curve (cumulative P&L by quarter):")
    trades["quarter"] = pd.to_datetime(trades["entry_dt"]).dt.to_period("Q")
    qtr = trades.groupby("quarter")["net_total"].sum().reset_index()
    qtr["cumulative"] = qtr["net_total"].cumsum()
    log.info("  %-10s %10s %12s %8s", "Quarter", "Period P&L", "Cumulative", "N trades")
    for _, r in qtr.iterrows():
        n_q = len(trades[trades["quarter"] == r["quarter"]])
        log.info("  %-10s %10.2f %12.2f %8d",
                 str(r["quarter"]), r["net_total"], r["cumulative"], n_q)

    # ── Sharpe ratio ───────────────────────────────────────────────────────
    log.info("")
    log.info("STEP 4 — Risk-adjusted performance")
    _hr()
    sharpe_monthly   = compute_sharpe(mon)
    sharpe_per_trade = compute_sharpe_per_trade(trades)
    dd               = compute_drawdown(mon)
    streaks          = compute_streaks(trades)

    log.info("  Sharpe (monthly, annualised):    %.2f", sharpe_monthly)
    log.info("  Sharpe (per-trade, annualised):  %.2f", sharpe_per_trade)
    log.info("")
    log.info("  Drawdown:")
    log.info("    Max drawdown:            $%.2f", dd["max_drawdown_dollars"])
    log.info("    Drawdown period:         %s → %s  (%d months)",
             dd["dd_start_month"], dd["dd_end_month"], dd["dd_duration_months"])
    log.info("    Final equity:            $%.2f", dd["final_equity"])
    log.info("")
    log.info("  Win/loss streaks:")
    log.info("    Max winning streak:  %d consecutive trades", streaks["max_win_streak"])
    log.info("    Max losing streak:   %d consecutive trades", streaks["max_loss_streak"])

    # Full trade win/loss stats
    net = trades["net_total"]
    winners = trades[trades["net_winner"]]
    losers  = trades[~trades["net_winner"]]
    log.info("")
    log.info("  Trade-level statistics:")
    log.info("    Total P&L:           $%.2f", net.sum())
    log.info("    Win rate:            %.1f%%  (%d/%d)",
             100 * trades["net_winner"].mean(), trades["net_winner"].sum(), len(trades))
    log.info("    Avg winner:          $%.2f", winners["net_total"].mean() if len(winners) > 0 else 0)
    log.info("    Avg loser:           $%.2f", losers["net_total"].mean() if len(losers) > 0 else 0)
    log.info("    Win/loss ratio:      %.2f",
             abs(winners["net_total"].mean() / losers["net_total"].mean())
             if len(losers) > 0 and losers["net_total"].mean() != 0 else float("nan"))
    log.info("    Largest single win:  $%.2f", net.max())
    log.info("    Largest single loss: $%.2f", net.min())
    log.info("    p10 net:             $%.2f", net.quantile(0.10))
    log.info("    p90 net:             $%.2f", net.quantile(0.90))

    # ── simultaneous positions and capital ────────────────────────────────
    log.info("")
    log.info("STEP 5 — Simultaneous positions and capital requirements")
    _hr()
    sim = simultaneous_positions(trades)
    cap = capital_efficiency(trades, sim)

    log.info("  Max simultaneous open positions:  %d",   sim["max_simultaneous_positions"])
    log.info("  Peak capital deployed:            $%.2f", sim["peak_capital_deployed"])
    log.info("  Avg capital per trade:            $%.2f", sim["avg_capital_deployed"])
    log.info("  Total net P&L (all years):        $%.2f", cap["total_net_pnl"])
    log.info("  Annualised net P&L:               $%.2f/year", cap["annualised_net_pnl"])
    log.info("  Return on peak capital (ann.):    %.1f%%",   cap["return_on_peak_capital_pct"])

    # ── breakdown by symbol (top contributors) ────────────────────────────
    log.info("")
    log.info("STEP 6 — Symbol-level breakdown")
    _hr()
    sym_grp = trades.groupby("symbol").agg(
        n_trades      = ("net_total", "count"),
        total_net     = ("net_total", "sum"),
        avg_net       = ("net_total", "mean"),
        win_rate      = ("net_winner", "mean"),
    ).sort_values("total_net", ascending=False).reset_index()

    log.info("  Top 10 contributors:")
    log.info("  %-8s %8s %10s %9s %8s",
             "Symbol", "Trades", "Total$", "Avg$", "Win%")
    for _, r in sym_grp.head(10).iterrows():
        log.info("  %-8s %8d %10.2f %9.2f %8.1f%%",
                 r["symbol"], r["n_trades"], r["total_net"],
                 r["avg_net"], 100 * r["win_rate"])

    log.info("")
    log.info("  Bottom 5 (worst symbols):")
    for _, r in sym_grp.tail(5).iterrows():
        log.info("  %-8s %8d %10.2f %9.2f %8.1f%%",
                 r["symbol"], r["n_trades"], r["total_net"],
                 r["avg_net"], 100 * r["win_rate"])

    # Concentration check: what % of P&L comes from top 5 symbols?
    top5_pnl = sym_grp.head(5)["total_net"].sum()
    total_pnl = trades["net_total"].sum()
    log.info("")
    log.info("  Top 5 symbols: %.1f%% of total P&L",
             100 * top5_pnl / total_pnl if total_pnl != 0 else float("nan"))
    log.info("  Unique symbols trading: %d", trades["symbol"].nunique())

    # ── size sensitivity ───────────────────────────────────────────────────
    log.info("")
    log.info("STEP 7 — Size sensitivity (1 / 2 / 5 contracts)")
    _hr()
    log.info("  %-12s %8s %10s %10s %10s %9s %8s",
             "Contracts", "Trades", "Total P&L", "Ann. P&L", "Peak Cap",
             "ROC%", "Sharpe")
    _hr(70)
    for n in SIZE_SCENARIOS:
        t_n, _ = run_backtest(events, n_contracts=n)
        if len(t_n) == 0:
            continue
        m_n   = monthly_pnl(t_n)
        dd_n  = compute_drawdown(m_n)
        sim_n = simultaneous_positions(t_n)
        cap_n = capital_efficiency(t_n, sim_n)
        sh_n  = compute_sharpe(m_n)
        log.info("  %-12d %8d %10.2f %10.2f %10.2f %9.1f%% %8.2f",
                 n, len(t_n),
                 t_n["net_total"].sum(),
                 cap_n["annualised_net_pnl"],
                 sim_n["peak_capital_deployed"],
                 cap_n["return_on_peak_capital_pct"],
                 sh_n)

    # ── NBR sub-group analysis ─────────────────────────────────────────────
    log.info("")
    log.info("STEP 8 — NBR sub-group performance")
    _hr()
    trades["nbr_bucket"] = pd.cut(trades["nbr_t1"],
                                   bins=[0, 1.39, 1.48, 1.60, 99],
                                   labels=["Q4-low", "Q4-high", "Q5-low", "Q5-high"])
    nbr_grp = trades.groupby("nbr_bucket").agg(
        n         = ("net_total", "count"),
        total_net = ("net_total", "sum"),
        avg_net   = ("net_total", "mean"),
        win_rate  = ("net_winner", "mean"),
    ).reset_index()
    _ptable("  P&L by NBR bucket:", nbr_grp)

    # ── final summary ──────────────────────────────────────────────────────
    log.info("")
    _eq()
    log.info("SUMMARY — COMPLETE STRATEGY PERFORMANCE")
    _eq()
    log.info("")
    log.info("  Universe:       Q4-Q5 NBR, OI/spread filtered, walk-forward thresholds")
    log.info("  Protocol:       T-2 entry → T+1 exit, %d contracts", BASE_SIZE)
    log.info("  Period:         %d – %d  (%d years)",
             int(trades["year"].min()), int(trades["year"].max()),
             int(trades["year"].nunique()))
    log.info("")
    log.info("  RETURNS:")
    log.info("    Total net P&L:                    $%.2f",   cap["total_net_pnl"])
    log.info("    Annualised net P&L:               $%.2f",   cap["annualised_net_pnl"])
    log.info("    Return on peak capital (ann.):    %.1f%%",  cap["return_on_peak_capital_pct"])
    log.info("")
    log.info("  RISK:")
    log.info("    Sharpe ratio (monthly):           %.2f",  sharpe_monthly)
    log.info("    Sharpe ratio (per-trade):         %.2f",  sharpe_per_trade)
    log.info("    Max drawdown:                     $%.2f", dd["max_drawdown_dollars"])
    log.info("    Max losing streak:                %d trades", streaks["max_loss_streak"])
    log.info("    Worst single trade:               $%.2f", net.min())
    log.info("")
    log.info("  EXECUTION:")
    log.info("    Total trades:                     %d", len(trades))
    log.info("    Trades per year:                  %.1f", len(trades) / trades["year"].nunique())
    log.info("    Win rate (net of costs):          %.1f%%", 100 * trades["net_winner"].mean())
    log.info("    Max simultaneous positions:       %d",   sim["max_simultaneous_positions"])
    log.info("    Peak capital deployed:            $%.2f", sim["peak_capital_deployed"])
    log.info("    Avg cost per contract:            $%.2f",
             trades["cost_per_contract"].mean())
    log.info("")

    # ── save report ────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / f"backtest_{VERSION}_{ts}.json"
    report = {
        "version":     VERSION,
        "timestamp":   ts,
        "meta":        meta,
        "annual":      ann.to_dict("records"),
        "monthly_pnl": mon.to_dict("records"),
        "quarterly":   qtr.assign(quarter=qtr["quarter"].astype(str)).to_dict("records"),
        "sharpe_monthly":   round(sharpe_monthly, 3) if not math.isnan(sharpe_monthly) else None,
        "sharpe_per_trade": round(sharpe_per_trade, 3) if not math.isnan(sharpe_per_trade) else None,
        "drawdown":    dd,
        "streaks":     streaks,
        "sim_positions": sim,
        "capital":     cap,
        "trade_log":   trades.drop(columns=["month","quarter"], errors="ignore").to_dict("records"),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("  Report saved: %s", report_path)


if __name__ == "__main__":
    main()
