"""
signal_engine.py — Scan upcoming earnings and compute NBR signals.

Daily workflow:
  1. Query institutional DB for events where pre_capture_date is TODAY
     (i.e., earnings are tomorrow — T-1 screening checkpoint)
  2. Load T-1 options chain from parquet store
  3. Find ATM call for front and back expiries
  4. Compute NBR = front_iv / back_iv
  5. Compute walk-forward NBR threshold from prior year data
  6. Return Signal objects for downstream filtering

Why T-1 screening:
  We screen on T-1 to confirm real-time NBR and liquidity.
  Entry actually happens at T-2 (2 business days before this screening date),
  but the signal gate is evaluated at T-1 when we have the clearest IV signal.

In live use: replace parquet loading with broker quote API.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

log = logging.getLogger(__name__)

INSTITUTIONAL_DB = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
PARQUET_BASE     = Path("/Volumes/T9/market_data/research/options_features_eod")
ATM_MONEYNESS_WIN = 5.0


@dataclass
class Signal:
    """Fully populated signal for one earnings event."""
    symbol:               str
    event_date:           str
    pre_capture_date:     str   # T-1 = screening date
    post_capture_date:    str   # expected exit day (T+1 post-earnings)
    front_expiry:         str
    back_expiry:          str
    sector:               str
    # T-1 market data
    nbr:                  float
    nbr_threshold:        float
    t1_front_oi:          int
    t1_back_oi:           int
    t1_front_spread_pct:  float
    t1_back_spread_pct:   float
    t1_front_iv:          float
    t1_back_iv:           float
    t1_front_mid:         float
    t1_back_mid:          float
    t1_front_dte:         int
    t1_back_dte:          int
    # Computed fields
    passed_nbr:           bool = False
    passed_oi:            bool = False
    passed_spread:        bool = False
    passed_all_filters:   bool = False
    filter_reject_reason: str  = ""
    # Set after risk gate
    passed_risk_gate:     bool = False
    risk_reject_reason:   str  = ""
    traded:               bool = False
    trade_id:             Optional[int] = None


SECTOR_MAP = {
    "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","AMD":"Tech","AVGO":"Tech",
    "AMAT":"Tech","CSCO":"Tech","IBM":"Tech","ORCL":"Tech","CRM":"Tech",
    "INTU":"Tech","TXN":"Tech","LRCX":"Tech","KLAC":"Tech","MCHP":"Tech",
    "ADI":"Tech","QCOM":"Tech","NOW":"Tech","CDNS":"Tech","PANW":"Tech",
    "FTNT":"Tech","MSI":"Tech","CHTR":"Tech","CBOE":"Tech","ACN":"Tech","ANET":"Tech",
    "META":"Comm","GOOGL":"Comm","NFLX":"Comm","DIS":"Comm","CMCSA":"Comm",
    "EA":"Comm","BKNG":"Comm","UBER":"Comm",
    "BAC":"Fin","JPM":"Fin","GS":"Fin","MS":"Fin","C":"Fin","WFC":"Fin",
    "AXP":"Fin","V":"Fin","MA":"Fin","COF":"Fin","SCHW":"Fin","BLK":"Fin",
    "USB":"Fin","PNC":"Fin","MET":"Fin","AFL":"Fin","AIG":"Fin",
    "JNJ":"Health","UNH":"Health","LLY":"Health","ABBV":"Health","MRK":"Health",
    "AMGN":"Health","GILD":"Health","REGN":"Health","BMY":"Health","ISRG":"Health",
    "TMO":"Health","DHR":"Health","MDT":"Health","BAX":"Health","BIIB":"Health","PFE":"Health",
    "AMZN":"ConDisc","HD":"ConDisc","MCD":"ConDisc","SBUX":"ConDisc","NKE":"ConDisc",
    "TSLA":"ConDisc","LOW":"ConDisc","ROST":"ConDisc","BBY":"ConDisc","LULU":"ConDisc",
    "MAR":"ConDisc",
    "KO":"ConStap","PEP":"ConStap","WMT":"ConStap","PG":"ConStap","COST":"ConStap",
    "MO":"ConStap","PM":"ConStap","KHC":"ConStap","CL":"ConStap","MNST":"ConStap",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","FCX":"Energy",
    "CAT":"Indust","DE":"Indust","HON":"Indust","GE":"Indust","GD":"Indust",
    "EMR":"Indust","RTX":"Indust","LMT":"Indust","NSC":"Indust","CSX":"Indust",
    "UPS":"Indust","FDX":"Indust","ETN":"Indust","HWM":"Indust","NOC":"Indust",
    "DAL":"Indust","MMM":"Indust","DD":"Indust",
    "NEE":"Util","SO":"Util","DUK":"Util","AEP":"Util","EXC":"Util","SRE":"Util",
}


def _parquet_path(symbol: str, dt: pd.Timestamp) -> Path:
    yr, mo = str(dt.year), str(dt.month).zfill(2)
    return (PARQUET_BASE / f"underlying_symbol={symbol}"
            / f"year={yr}" / f"month={mo}"
            / f"{symbol.lower()}_options_features_eod_{yr}-{mo}.parquet")


def _load_chain(symbol: str, dt: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Load EOD options chain from parquet. Replace with broker API in live."""
    p = _parquet_path(symbol, dt)
    if not p.exists():
        return None
    try:
        chain = pd.read_parquet(p)
    except Exception as e:
        log.warning("Failed to read parquet for %s on %s: %s", symbol, dt.date(), e)
        return None
    chain["trade_date"] = pd.to_datetime(chain["trade_date"])
    chain["expiry"]     = pd.to_datetime(chain["expiry"])
    day = chain[
        (chain["trade_date"] == dt)
        & (chain["quote_quality_flag"] == "ok")
        & (chain["bid"] > 0)
    ]
    return day.copy() if len(day) > 0 else None


def _atm_call(chain: pd.DataFrame, expiry: pd.Timestamp) -> Optional[pd.Series]:
    leg = chain[
        (chain["expiry"] == expiry)
        & (chain["call_put"] == "C")
        & (chain["moneyness_pct"].abs() <= ATM_MONEYNESS_WIN)
    ]
    if len(leg) == 0:
        return None
    return leg.loc[leg["moneyness_pct"].abs().idxmin()]


def _compute_wf_threshold(as_of_date: pd.Timestamp,
                           pctile: float = 0.60) -> float:
    """
    Walk-forward NBR threshold: use all events BEFORE as_of_date's year.
    Falls back to full-sample if no prior data.
    """
    yr = as_of_date.year
    conn = sqlite3.connect(INSTITUTIONAL_DB)
    df = pd.read_sql(
        "SELECT pre_front_iv, pre_back_iv, pre_capture_date "
        "FROM earnings_iv_decay_labels WHERE quality_tier IN ('A','B')",
        conn,
    )
    conn.close()
    df["year"] = pd.to_datetime(df["pre_capture_date"]).dt.year
    prior = df[df["year"] < yr] if (df["year"] < yr).any() else df
    nbr_series = prior["pre_front_iv"] / prior["pre_back_iv"]
    return float(nbr_series.quantile(pctile))


def scan_for_date(screening_date: pd.Timestamp,
                  nbr_pctile: float = 0.60) -> List[Signal]:
    """
    Main entry point: scan all earnings events where pre_capture_date == screening_date.

    In live use: screening_date = today.
    In paper trading: pass any historical date.

    Returns a list of Signal objects (both passed and failed filters),
    so the caller has full visibility into all candidates.
    """
    log.info("Scanning earnings for pre_capture_date = %s", screening_date.date())

    conn = sqlite3.connect(INSTITUTIONAL_DB)
    events = pd.read_sql(
        """
        SELECT symbol, event_date, pre_capture_date, post_capture_date,
               front_expiry, back_expiry, front_dte_pre, back_dte_pre,
               pre_front_iv, pre_back_iv, quality_tier
        FROM earnings_iv_decay_labels
        WHERE pre_capture_date = ?
          AND quality_tier IN ('A','B')
        ORDER BY symbol
        """,
        conn,
        params=[screening_date.strftime("%Y-%m-%d")],
    )
    conn.close()

    if len(events) == 0:
        log.info("  No earnings events on %s", screening_date.date())
        return []

    log.info("  Found %d earnings events on %s", len(events), screening_date.date())
    threshold = _compute_wf_threshold(screening_date, nbr_pctile)
    log.info("  Walk-forward NBR threshold: %.4f", threshold)

    signals: List[Signal] = []

    for _, ev in events.iterrows():
        sym       = ev["symbol"]
        pre_dt    = pd.to_datetime(ev["pre_capture_date"])
        front_exp = pd.to_datetime(ev["front_expiry"])
        back_exp  = pd.to_datetime(ev["back_expiry"])

        # Load T-1 chain (screening data)
        chain = _load_chain(sym, pre_dt)
        if chain is None:
            log.debug("  %s: no parquet data on %s", sym, pre_dt.date())
            continue

        f_opt = _atm_call(chain, front_exp)
        b_opt = _atm_call(chain, back_exp)
        if f_opt is None or b_opt is None:
            log.debug("  %s: ATM call not found for one or both expiries", sym)
            continue

        nbr = float(f_opt["iv"]) / float(b_opt["iv"])

        sig = Signal(
            symbol            = sym,
            event_date        = ev["event_date"],
            pre_capture_date  = ev["pre_capture_date"],
            post_capture_date = ev["post_capture_date"],
            front_expiry      = ev["front_expiry"],
            back_expiry       = ev["back_expiry"],
            sector            = SECTOR_MAP.get(sym, "Other"),
            nbr               = round(nbr, 4),
            nbr_threshold     = round(threshold, 4),
            t1_front_oi       = int(f_opt["open_interest"]),
            t1_back_oi        = int(b_opt["open_interest"]),
            t1_front_spread_pct = round(float(f_opt["spread_pct"]), 4),
            t1_back_spread_pct  = round(float(b_opt["spread_pct"]), 4),
            t1_front_iv       = round(float(f_opt["iv"]), 4),
            t1_back_iv        = round(float(b_opt["iv"]), 4),
            t1_front_mid      = round(float(f_opt["mid"]), 3),
            t1_back_mid       = round(float(b_opt["mid"]), 3),
            t1_front_dte      = int(f_opt["dte"]),
            t1_back_dte       = int(b_opt["dte"]),
        )
        signals.append(sig)

    log.info("  Signals with market data: %d / %d events", len(signals), len(events))
    return signals


def load_entry_chain_for_signal(sig: Signal) -> Optional[dict]:
    """
    Load T-2 options chain (entry date).
    Returns {front: Series, back: Series} or None.
    """
    pre_dt    = pd.to_datetime(sig.pre_capture_date)
    entry_dt  = pre_dt - pd.offsets.BDay(2)
    front_exp = pd.to_datetime(sig.front_expiry)
    back_exp  = pd.to_datetime(sig.back_expiry)

    chain = _load_chain(sig.symbol, entry_dt)
    if chain is None:
        return None
    f = _atm_call(chain, front_exp)
    b = _atm_call(chain, back_exp)
    if f is None or b is None:
        return None
    return {"front": f, "back": b, "entry_dt": entry_dt}


def load_exit_chain_for_trade(trade: dict) -> Optional[dict]:
    """
    Load T+1 exit chain for an open trade.
    Returns {front: Series, back: Series} or None.
    """
    post_dt   = pd.to_datetime(trade["post_capture_date"] if "post_capture_date" in trade
                                else trade["expected_exit_date"])
    # For exit: we load on expected_exit_date directly
    exit_dt   = pd.to_datetime(trade["expected_exit_date"])
    sym       = trade["symbol"]
    front_exp = pd.to_datetime(trade["front_expiry"]) if "front_expiry" in trade else None
    back_exp  = pd.to_datetime(trade["back_expiry"])  if "back_expiry" in trade else None

    if front_exp is None or back_exp is None:
        # Reconstruct from signal if missing
        conn = sqlite3.connect(INSTITUTIONAL_DB)
        row = pd.read_sql(
            "SELECT front_expiry, back_expiry FROM earnings_iv_decay_labels "
            "WHERE symbol=? AND event_date=?",
            conn,
            params=[sym, trade["event_date"]],
        )
        conn.close()
        if len(row) == 0:
            return None
        front_exp = pd.to_datetime(row.iloc[0]["front_expiry"])
        back_exp  = pd.to_datetime(row.iloc[0]["back_expiry"])

    chain = _load_chain(sym, exit_dt)
    if chain is None:
        return None
    f = _atm_call(chain, front_exp)
    b = _atm_call(chain, back_exp)
    if f is None or b is None:
        return None
    return {"front": f, "back": b, "exit_dt": exit_dt}


def load_mark_chain(symbol: str, mark_date: pd.Timestamp,
                    front_expiry: str, back_expiry: str) -> Optional[dict]:
    """Load current-day chain for MTM marking of an open position."""
    front_exp = pd.to_datetime(front_expiry)
    back_exp  = pd.to_datetime(back_expiry)
    chain = _load_chain(symbol, mark_date)
    if chain is None:
        return None
    f = _atm_call(chain, front_exp)
    b = _atm_call(chain, back_exp)
    if f is None or b is None:
        return None
    return {"front": f, "back": b}
