#!/usr/bin/env python3
"""
scripts/train_vix_regime_model.py
===================================
Week 3 — VIX Regime Feature Research

Skeptical evaluation: does VIX at the pre-event date provide REAL incremental
signal beyond the 3-feature baseline established in Week 2?

Pipeline steps
--------------
  Step 1  Load VIX series and compute trailing regime features (no leakage)
  Step 2  Regime analysis — VIX quintile vs crush outcomes (before any modeling)
  Step 3  Walk-forward comparison: baseline vs VIX-enhanced vs VIX-only
  Step 4  Ablation analysis — isolate VIX contribution
  Step 5  Economic bucket comparison (Q5-Q1 spread)
  Step 6  Failure mode analysis — energy expansion events, high/low VIX regimes
  Step 7  Verdict: does VIX belong in production?

Leakage guarantees
------------------
  - VIX value from pre_capture_date ONLY (never post-event)
  - Trailing 252-day percentile uses ONLY data up to and including pre_capture_date
  - Scaler fit on training fold only — never sees test data
  - Feature computation uses VIX series exclusively — no label information

Usage
-----
  python scripts/train_vix_regime_model.py [options]
  python scripts/train_vix_regime_model.py --quality-tier A
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
try:
    from utils.logger import setup_logger
    log = setup_logger(__name__)
except Exception:
    logging.basicConfig(format="%(asctime)s | %(levelname)-8s | %(message)s",
                        level=logging.INFO)
    log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH  = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
DEFAULT_REPORT_DIR = Path.home() / ".options_calculator_pro" / "reports"
RESEARCH_VERSION = "vix_regime_v1"

VIX_PARQUET_GLOB = (
    "/Volumes/T9/market_data/normalized/underlyings/"
    "daily_ohlcv/underlying_symbol=^VIX/**/*.parquet"
)
VIX_PRICE_SCALE = 10_000.0      # close_10000 / 10000 = VIX level
VIX_TRAILING_WINDOW = 252       # 1 trading year for rolling percentile

# Feature sets (all walk-forward runs use same hyperparams)
BASELINE_FEATURES = ["near_back_ratio", "log_front_iv", "log_back_iv"]
VIX_FEATURES      = ["log_vix", "vix_pct_252"]
COMBINED_FEATURES = BASELINE_FEATURES + VIX_FEATURES

# Walk-forward folds: (train_years, test_year) — identical to Week 2
WALK_FORWARD_FOLDS: List[Tuple[List[int], int]] = [
    ([2023],           2024),
    ([2023, 2024],     2025),
    ([2023, 2024, 2025], 2026),
]

# Regression target (primary) + binary target (secondary) — unchanged from Wk2
PRIMARY_REG_TARGET    = "term_ratio_change"
SECONDARY_CLF_TARGET  = "crush_lt_40pct"

# VIX regime thresholds for analysis
VIX_REGIME_BINS  = [0, 15, 20, 25, 35, 999]
VIX_REGIME_NAMES = ["<15 (calm)", "15–20 (normal)", "20–25 (elevated)",
                    "25–35 (stressed)", ">35 (crisis)"]

# Model config — fixed, no search (identical to Week 2)
LR_C        = 1.0
LR_SOLVER   = "lbfgs"
LR_MAX_ITER = 500
RIDGE_ALPHA = 1.0


# ── VIX data loading + feature computation ────────────────────────────────────

def load_vix_series() -> pd.DataFrame:
    """Load full VIX daily close from normalized parquet store.

    Returns DataFrame with columns: [trade_date (datetime), vix_close (float)]
    Sorted ascending by trade_date. No forward-looking data.
    """
    con = duckdb.connect(":memory:")
    vix = con.execute(f"""
        SELECT trade_date,
               close_10000 / {VIX_PRICE_SCALE} AS vix_close
        FROM read_parquet('{VIX_PARQUET_GLOB}')
        WHERE close_10000 IS NOT NULL AND close_10000 > 0
        ORDER BY trade_date
    """).df()
    vix["trade_date"] = pd.to_datetime(vix["trade_date"])
    con.close()
    log.info("VIX series: %d days  %s → %s",
             len(vix), vix["trade_date"].min().date(), vix["trade_date"].max().date())
    return vix


def compute_vix_features(vix: pd.DataFrame) -> pd.DataFrame:
    """Compute VIX regime features using TRAILING data only.

    For each date D:
      - log_vix:      log(vix_close[D])
      - vix_pct_252:  percentile of vix_close[D] within vix_close[D-251:D+1]
                      (trailing 252-day window, inclusive of D)

    The percentile is strictly backward-looking — it ranks D's VIX against
    the prior 252 trading days. No future VIX data is used at any point.
    """
    vix = vix.copy().sort_values("trade_date").reset_index(drop=True)
    vix["log_vix"] = np.log(vix["vix_close"])

    # Rolling trailing percentile: what fraction of past-252-day VIX values
    # is the current VIX ≥ to? rank(current) / window_size.
    pcts = np.full(len(vix), np.nan)
    for i in range(len(vix)):
        start = max(0, i - VIX_TRAILING_WINDOW + 1)
        window = vix["vix_close"].values[start : i + 1]
        pcts[i] = (window <= vix["vix_close"].values[i]).mean()
    vix["vix_pct_252"] = pcts

    # VIX 5-day change (trailing) — computed but not used in baseline unless justified
    vix["vix_chg_5d"] = vix["vix_close"].pct_change(5)

    valid = vix.dropna(subset=["vix_pct_252"])
    log.info("VIX features computed: %d dates with valid pct_252  "
             "(window requires min 1 obs, full 252 days available from %s)",
             len(valid), vix.loc[VIX_TRAILING_WINDOW - 1, "trade_date"].date()
             if len(vix) > VIX_TRAILING_WINDOW else "n/a")
    return vix


# ── label loading + feature engineering ──────────────────────────────────────

def load_labels(db_path: Path, quality_tiers: List[str]) -> pd.DataFrame:
    ph = ",".join("?" * len(quality_tiers))
    query = f"""
        SELECT id, symbol, event_date, pre_capture_date, post_capture_date,
               release_timing,
               pre_front_iv, post_front_iv, pre_back_iv, post_back_iv,
               front_iv_crush_pct, back_iv_crush_pct, term_ratio_change,
               underlying_move_pct, front_dte_pre, back_dte_pre,
               exact_expiry_match, quality_score, quality_tier,
               pre_front_atm_moneyness, pre_front_oi, pre_underlying_price
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ({ph})
          AND pre_front_iv > 0 AND pre_back_iv > 0
          AND term_ratio_change IS NOT NULL
          AND front_iv_crush_pct IS NOT NULL
        ORDER BY event_date
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=quality_tiers)
    df["event_date"]       = pd.to_datetime(df["event_date"])
    df["pre_capture_date"] = pd.to_datetime(df["pre_capture_date"])
    df["post_capture_date"] = pd.to_datetime(df["post_capture_date"])
    df["year"]    = df["event_date"].dt.year
    df["quarter"] = df["event_date"].dt.quarter
    log.info("Loaded %d label rows (quality_tiers=%s)", len(df), quality_tiers)
    return df


def engineer_features(df: pd.DataFrame, vix_feats: pd.DataFrame) -> pd.DataFrame:
    """Derive all features from pre-event data only.

    Baseline features (unchanged from Week 2):
      near_back_ratio = pre_front_iv / pre_back_iv
      log_front_iv    = log(pre_front_iv)
      log_back_iv     = log(pre_back_iv)

    VIX features (joined by pre_capture_date — STRICT leakage control):
      log_vix         = log(VIX at pre_capture_date)
      vix_pct_252     = trailing 252-day VIX percentile at pre_capture_date
    """
    df = df.copy()
    # Baseline
    df["near_back_ratio"] = df["pre_front_iv"] / df["pre_back_iv"]
    df["log_front_iv"]    = np.log(df["pre_front_iv"])
    df["log_back_iv"]     = np.log(df["pre_back_iv"])

    # Binary target
    df["crush_lt_40pct"]  = (df["front_iv_crush_pct"] < -0.40).astype(int)

    # VIX join: by pre_capture_date — ONLY pre-event VIX, never post-event
    vix_cols = vix_feats[["trade_date", "vix_close", "log_vix",
                           "vix_pct_252", "vix_chg_5d"]].copy()
    df = df.merge(vix_cols, left_on="pre_capture_date", right_on="trade_date", how="left")

    missing = df["log_vix"].isna().sum()
    if missing:
        log.warning("%d rows missing VIX after join — check date alignment", missing)
    else:
        log.info("VIX join: all %d rows matched (0 missing)", len(df))

    # Auxiliary for economic eval
    df["spread_differential"] = df["front_iv_crush_pct"] - df["back_iv_crush_pct"]
    return df


# ── Step 2: regime analysis ───────────────────────────────────────────────────

def regime_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Bucket events by VIX level and compute crush statistics per regime.

    This is the mandatory pre-modeling check: does VIX correlate with outcomes?
    If no clear pattern, VIX adds no value and we stop here.
    """
    df2 = df.dropna(subset=["vix_close", "front_iv_crush_pct"]).copy()

    # ── Named-bin analysis ────────────────────────────────────────────────────
    df2["vix_regime"] = pd.cut(df2["vix_close"], bins=VIX_REGIME_BINS,
                               labels=VIX_REGIME_NAMES, right=False)
    named_bins = df2.groupby("vix_regime", observed=True).agg(
        n=("front_iv_crush_pct", "count"),
        vix_mean=("vix_close", "mean"),
        front_crush_mean=("front_iv_crush_pct", "mean"),
        front_crush_std=("front_iv_crush_pct", "std"),
        back_crush_mean=("back_iv_crush_pct", "mean"),
        trc_mean=("term_ratio_change", "mean"),
        crush_rate_lt40=("crush_lt_40pct", "mean"),
    ).reset_index()

    # ── Quintile analysis (finer-grained) ────────────────────────────────────
    df2["vix_quintile"] = pd.qcut(df2["vix_close"], 5,
                                  labels=["Q1","Q2","Q3","Q4","Q5"])
    quintile_bins = df2.groupby("vix_quintile", observed=True).agg(
        n=("front_iv_crush_pct", "count"),
        vix_min=("vix_close", "min"),
        vix_max=("vix_close", "max"),
        vix_mean=("vix_close", "mean"),
        front_crush_mean=("front_iv_crush_pct", "mean"),
        back_crush_mean=("back_iv_crush_pct", "mean"),
        trc_mean=("term_ratio_change", "mean"),
        crush_rate_lt40=("crush_lt_40pct", "mean"),
        nbr_mean=("near_back_ratio", "mean"),
    ).reset_index()

    # ── Spearman: VIX vs crush ────────────────────────────────────────────────
    sp_front_vix, p_front_vix = spearmanr(df2["vix_close"], df2["front_iv_crush_pct"])
    sp_trc_vix,   p_trc_vix   = spearmanr(df2["vix_close"], df2["term_ratio_change"])
    sp_pct_front, p_pct_front = spearmanr(df2["vix_pct_252"], df2["front_iv_crush_pct"])
    sp_pct_trc,   p_pct_trc   = spearmanr(df2["vix_pct_252"], df2["term_ratio_change"])

    # ── Near-back ratio mean per quintile (to check VIX is not just a proxy) ──
    # If near_back_ratio is similar across VIX quintiles, VIX is independent signal.
    nbr_by_vixq = df2.groupby("vix_quintile", observed=True)["near_back_ratio"].mean()

    # ── Partial correlation: VIX vs TRC controlling for near_back_ratio ───────
    from sklearn.linear_model import LinearRegression
    X_nbr = df2[["near_back_ratio"]].values
    y_trc = df2["term_ratio_change"].values
    y_vix = df2["vix_close"].values
    # Residualise TRC on near_back_ratio
    res_trc_on_nbr = y_trc - LinearRegression().fit(X_nbr, y_trc).predict(X_nbr)
    # Residualise VIX on near_back_ratio
    res_vix_on_nbr = y_vix - LinearRegression().fit(X_nbr, y_vix).predict(X_nbr)
    sp_partial, p_partial = spearmanr(res_vix_on_nbr, res_trc_on_nbr)

    # ── VIX at IV expansion events ─────────────────────────────────────────────
    expansion = df2[df2["front_iv_crush_pct"] > 0][
        ["symbol", "event_date", "vix_close", "vix_pct_252",
         "front_iv_crush_pct", "underlying_move_pct"]
    ].copy()
    expansion["front_iv_crush_pct"] = (expansion["front_iv_crush_pct"] * 100).round(1)

    return {
        "named_bins": named_bins.to_dict("records"),
        "quintile_bins": quintile_bins.to_dict("records"),
        "correlations": {
            "spearman_vix_vs_front_crush": {
                "r": round(float(sp_front_vix), 4), "p": round(float(p_front_vix), 4)
            },
            "spearman_vix_vs_trc": {
                "r": round(float(sp_trc_vix), 4), "p": round(float(p_trc_vix), 4)
            },
            "spearman_vixpct_vs_front_crush": {
                "r": round(float(sp_pct_front), 4), "p": round(float(p_pct_front), 4)
            },
            "spearman_vixpct_vs_trc": {
                "r": round(float(sp_pct_trc), 4), "p": round(float(p_pct_trc), 4)
            },
            "partial_vix_vs_trc_controlling_nbr": {
                "r": round(float(sp_partial), 4), "p": round(float(p_partial), 4),
                "note": "Spearman(VIX residual, TRC residual) after removing near_back_ratio"
            },
        },
        "nbr_by_vix_quintile": nbr_by_vixq.round(4).to_dict(),
        "expansion_events": expansion.to_dict("records"),
    }


def print_regime_analysis(ra: Dict[str, Any]) -> None:
    log.info("=" * 72)
    log.info("STEP 2 — VIX REGIME ANALYSIS (pre-modeling)")
    log.info("=" * 72)
    log.info("")
    log.info("── VIX Level vs IV Crush (named regimes) ───────────────────")
    log.info("  %-22s  %5s  %6s  %13s  %12s  %10s  %13s",
             "Regime", "N", "VIX", "Front crush", "Back crush", "TRC mean", "Rate(<-40%)")
    log.info("  " + "-" * 80)
    for row in ra["named_bins"]:
        log.info("  %-22s  %5d  %5.1f  %+12.1f%%  %+11.1f%%  %+9.4f  %12.1f%%",
                 row["vix_regime"], row["n"], row["vix_mean"],
                 row["front_crush_mean"] * 100, row["back_crush_mean"] * 100,
                 row["trc_mean"], row["crush_rate_lt40"] * 100)

    log.info("")
    log.info("── VIX Quintile vs IV Crush ─────────────────────────────────")
    log.info("  %-4s  %5s  %8s  %13s  %12s  %10s  %13s  %8s",
             "Q", "N", "VIX rng", "Front crush", "Back crush", "TRC mean",
             "Rate(<-40%)", "NBR mean")
    log.info("  " + "-" * 86)
    for row in ra["quintile_bins"]:
        log.info("  %-4s  %5d  %4.1f–%4.1f  %+12.1f%%  %+11.1f%%  %+9.4f  %12.1f%%  %8.3f",
                 row["vix_quintile"], row["n"],
                 row["vix_min"], row["vix_max"],
                 row["front_crush_mean"] * 100, row["back_crush_mean"] * 100,
                 row["trc_mean"], row["crush_rate_lt40"] * 100, row["nbr_mean"])

    log.info("")
    log.info("── Correlation Analysis ─────────────────────────────────────")
    c = ra["correlations"]
    log.info("  Spearman(VIX level, front_crush):   r=%+.4f  p=%.4f",
             c["spearman_vix_vs_front_crush"]["r"], c["spearman_vix_vs_front_crush"]["p"])
    log.info("  Spearman(VIX level, TRC):           r=%+.4f  p=%.4f",
             c["spearman_vix_vs_trc"]["r"], c["spearman_vix_vs_trc"]["p"])
    log.info("  Spearman(VIX_pct252, front_crush):  r=%+.4f  p=%.4f",
             c["spearman_vixpct_vs_front_crush"]["r"], c["spearman_vixpct_vs_front_crush"]["p"])
    log.info("  Spearman(VIX_pct252, TRC):          r=%+.4f  p=%.4f",
             c["spearman_vixpct_vs_trc"]["r"], c["spearman_vixpct_vs_trc"]["p"])
    log.info("")
    log.info("  Partial correlation (key independence check):")
    pc = c["partial_vix_vs_trc_controlling_nbr"]
    log.info("  Spearman(VIX | NBR, TRC | NBR):    r=%+.4f  p=%.4f",
             pc["r"], pc["p"])
    log.info("  → %s", pc["note"])

    log.info("")
    log.info("── near_back_ratio by VIX Quintile (independence check) ──────")
    log.info("  (If NBR varies strongly with VIX, VIX may be a proxy, not new info)")
    for q, nbr in ra["nbr_by_vix_quintile"].items():
        log.info("  VIX %-4s  NBR mean = %.4f", q, nbr)

    log.info("")
    log.info("── IV Expansion Events (front crush > 0) ────────────────────")
    for ev in ra["expansion_events"]:
        log.info("  %-6s  %s  VIX=%.1f  pct=%.2f  crush=%+.1f%%",
                 ev["symbol"],
                 str(ev["event_date"])[:10],
                 ev["vix_close"], ev["vix_pct_252"],
                 ev["front_iv_crush_pct"])


# ── walk-forward engine (generic — accepts any feature list) ──────────────────

def _fold_data(df, train_years, test_year, feature_cols, target_col):
    tr = df[df["year"].isin(train_years)].dropna(subset=feature_cols + [target_col])
    te = df[df["year"] == test_year].dropna(subset=feature_cols + [target_col])
    if len(tr) == 0 or len(te) == 0:
        return None
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr[feature_cols].values)
    X_te = scaler.transform(te[feature_cols].values)
    return X_tr, tr[target_col].values, X_te, te[target_col].values, te.index


def wf_regression(df, target_col, feature_cols, label):
    """Walk-forward Ridge regression. Returns per-fold + pooled metrics."""
    preds = []
    folds = []
    for train_yrs, test_yr in WALK_FORWARD_FOLDS:
        fd = _fold_data(df, train_yrs, test_yr, feature_cols, target_col)
        if fd is None:
            continue
        X_tr, y_tr, X_te, y_te, te_idx = fd
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)
        naive = np.full_like(y_te, y_tr.mean())
        rmse  = float(np.sqrt(mean_squared_error(y_te, y_hat)))
        naive_rmse = float(np.sqrt(mean_squared_error(y_te, naive)))
        r2    = float(r2_score(y_te, y_hat))
        sp_r, sp_p = spearmanr(y_te, y_hat)
        coef  = {f: round(float(c), 5) for f, c in zip(feature_cols, m.coef_)}
        folds.append({
            "train_years": train_yrs, "test_year": test_yr,
            "n_train": len(X_tr), "n_test": len(X_te),
            "rmse": round(rmse, 6), "naive_rmse": round(naive_rmse, 6),
            "rmse_ratio": round(rmse / naive_rmse, 4),
            "r2": round(r2, 4),
            "spearman_r": round(float(sp_r), 4), "spearman_p": round(float(sp_p), 4),
            "coef": coef,
        })
        for idx, yt, yp in zip(te_idx.tolist(), y_te.tolist(), y_hat.tolist()):
            preds.append({"df_index": idx, "y_true": yt, "y_pred": yp})

    # Pooled
    pooled: Dict[str, Any] = {}
    if preds:
        yt_all  = np.array([x["y_true"] for x in preds])
        yp_all  = np.array([x["y_pred"] for x in preds])
        naive_all = np.full_like(yt_all, yt_all.mean())
        sp_r, sp_p = spearmanr(yt_all, yp_all)
        pooled = {
            "rmse": round(float(np.sqrt(mean_squared_error(yt_all, yp_all))), 6),
            "naive_rmse": round(float(np.sqrt(mean_squared_error(yt_all, naive_all))), 6),
            "rmse_ratio": round(float(np.sqrt(mean_squared_error(yt_all, yp_all))) /
                                float(np.sqrt(mean_squared_error(yt_all, naive_all))), 4),
            "r2": round(float(r2_score(yt_all, yp_all)), 4),
            "spearman_r": round(float(sp_r), 4),
            "spearman_p": round(float(sp_p), 4),
            "n": len(yt_all),
        }
    return {"label": label, "features": feature_cols, "target": target_col,
            "folds": folds, "pooled": pooled, "oos_preds": preds}


def wf_classification(df, target_col, feature_cols, label):
    """Walk-forward Logistic Regression. Returns per-fold + pooled metrics."""
    preds = []
    folds = []
    for train_yrs, test_yr in WALK_FORWARD_FOLDS:
        fd = _fold_data(df, train_yrs, test_yr, feature_cols, target_col)
        if fd is None:
            continue
        X_tr, y_tr, X_te, y_te, te_idx = fd
        if y_tr.mean() < 0.05 or y_tr.mean() > 0.95:
            log.warning("  %s fold train=%s: extreme imbalance pos=%.1f%%",
                        label, train_yrs, y_tr.mean() * 100)
        m = LogisticRegression(C=LR_C, solver=LR_SOLVER, max_iter=LR_MAX_ITER, random_state=42)
        m.fit(X_tr, y_tr)
        y_prob = m.predict_proba(X_te)[:, 1]
        auc   = float(roc_auc_score(y_te, y_prob)) if 0 < y_te.mean() < 1 else float("nan")
        brier = float(brier_score_loss(y_te, y_prob))
        coef  = {f: round(float(c), 5) for f, c in zip(feature_cols, m.coef_[0])}
        folds.append({
            "train_years": train_yrs, "test_year": test_yr,
            "n_train": len(X_tr), "n_test": len(X_te),
            "auc": round(auc, 4) if not math.isnan(auc) else None,
            "brier": round(brier, 4), "coef": coef,
        })
        for idx, yt, yp in zip(te_idx.tolist(), y_te.tolist(), y_prob.tolist()):
            preds.append({"df_index": idx, "y_true": yt, "y_prob": yp})

    pooled: Dict[str, Any] = {}
    if preds:
        yt_all  = np.array([x["y_true"] for x in preds])
        yp_all  = np.array([x["y_prob"] for x in preds])
        pooled = {
            "auc":   round(float(roc_auc_score(yt_all, yp_all)), 4) if 0 < yt_all.mean() < 1 else None,
            "brier": round(float(brier_score_loss(yt_all, yp_all)), 4),
            "n":     len(yt_all),
        }
    return {"label": label, "features": feature_cols, "target": target_col,
            "folds": folds, "pooled": pooled, "oos_preds": preds}


# ── economic bucket evaluation ─────────────────────────────────────────────────

def economic_eval(df, oos_preds, pred_col, label):
    """Quintile P&L proxy evaluation — same protocol as Week 2."""
    if not oos_preds:
        return {}
    pred_df = pd.DataFrame(oos_preds).set_index("df_index")
    merged = df[["front_iv_crush_pct", "back_iv_crush_pct",
                 "term_ratio_change", "spread_differential",
                 "vix_close", "year"]].join(pred_df, how="inner")
    if merged.empty:
        return {}
    merged["quintile"] = pd.qcut(merged[pred_col], 5, labels=["Q1","Q2","Q3","Q4","Q5"])
    agg = merged.groupby("quintile", observed=True).agg(
        n=("front_iv_crush_pct", "count"),
        avg_front=("front_iv_crush_pct", "mean"),
        avg_back=("back_iv_crush_pct", "mean"),
        avg_diff=("spread_differential", "mean"),
        avg_trc=("term_ratio_change", "mean"),
    ).reset_index()
    q1 = agg.loc[agg["quintile"] == "Q1", "avg_front"].values
    q5 = agg.loc[agg["quintile"] == "Q5", "avg_front"].values
    spread = float(q5[0] - q1[0]) * 100 if len(q1) and len(q5) else float("nan")
    merged["q_rank"] = merged["quintile"].cat.codes
    sp_r, sp_p = spearmanr(merged["q_rank"], merged["front_iv_crush_pct"])
    return {
        "label": label, "n_oos": len(merged),
        "q5_q1_spread_pp": round(spread, 1),
        "spearman_r": round(float(sp_r), 4), "spearman_p": round(float(sp_p), 4),
        "quintile_table": agg.assign(
            avg_front_pct=lambda x: (x["avg_front"]*100).round(1),
            avg_back_pct=lambda x:  (x["avg_back"]*100).round(1),
            avg_diff_pct=lambda x:  (x["avg_diff"]*100).round(1),
        )[["quintile","n","avg_front_pct","avg_back_pct","avg_diff_pct","avg_trc"]].to_dict("records"),
    }


# ── Step 5: comparison printing ───────────────────────────────────────────────

def print_model_comparison(results: List[Dict], metric_type: str) -> None:
    """Print side-by-side comparison of baseline vs VIX-enhanced vs VIX-only."""
    log.info("")
    if metric_type == "regression":
        log.info("  %-32s  %10s  %10s  %10s  %6s",
                 "Model", "RMSE/naive", "Spearman_r", "R²", "N")
        log.info("  " + "-" * 74)
        for r in results:
            p = r.get("pooled", {})
            log.info("  %-32s  %10.4f  %10.4f  %10.4f  %6d",
                     r["label"],
                     p.get("rmse_ratio", float("nan")),
                     p.get("spearman_r", float("nan")),
                     p.get("r2", float("nan")),
                     p.get("n", 0))
        log.info("")
        # Per-fold breakdown
        log.info("  Per-fold Spearman_r:")
        log.info("  %-32s  %10s  %10s  %10s",
                 "Model", "Fold1(2024)", "Fold2(2025)", "Fold3(2026)")
        log.info("  " + "-" * 66)
        for r in results:
            folds = r.get("folds", [])
            f_sp = [f.get("spearman_r", float("nan")) for f in folds]
            log.info("  %-32s  %10.4f  %10.4f  %10.4f",
                     r["label"], *f_sp[:3])
    else:
        log.info("  %-32s  %10s  %10s  %6s",
                 "Model", "AUC", "Brier", "N")
        log.info("  " + "-" * 62)
        for r in results:
            p = r.get("pooled", {})
            log.info("  %-32s  %10.4f  %10.4f  %6d",
                     r["label"],
                     p.get("auc") or float("nan"),
                     p.get("brier", float("nan")),
                     p.get("n", 0))


def print_economic_comparison(econs: List[Dict]) -> None:
    log.info("")
    log.info("  %-32s  %10s  %12s", "Model", "Q5-Q1 pp", "Spearman_r")
    log.info("  " + "-" * 58)
    for e in econs:
        if e:
            log.info("  %-32s  %+9.1f  %12.4f",
                     e.get("label",""), e.get("q5_q1_spread_pp", float("nan")),
                     e.get("spearman_r", float("nan")))


def print_quintile_table(econ: Dict) -> None:
    if not econ:
        return
    log.info("  %-4s  %5s  %14s  %13s  %14s  %10s",
             "Q", "N", "Avg front crush", "Avg back crush",
             "Avg differential", "Avg TRC")
    log.info("  " + "-" * 64)
    for row in econ.get("quintile_table", []):
        log.info("  %-4s  %5d  %+13.1f%%  %+12.1f%%  %+13.1f%%  %+10.4f",
                 row["quintile"], row["n"],
                 row["avg_front_pct"], row["avg_back_pct"],
                 row["avg_diff_pct"], row["avg_trc"])


# ── Step 6: failure mode analysis ────────────────────────────────────────────

def failure_mode_analysis(df: pd.DataFrame, combined_reg_preds: List[Dict]) -> Dict[str, Any]:
    """Analyze whether VIX regime reduces prediction failures.

    Specific focus:
      1. High-VIX events (>25): does the model distinguish IV expansion from crush?
      2. Low-VIX events (<15): baseline performance in calm markets
      3. Energy/Utilities sector: Aug-2024 anomaly cluster
    """
    pred_df = pd.DataFrame(combined_reg_preds).set_index("df_index") if combined_reg_preds else None

    analysis: Dict[str, Any] = {}

    # VIX regime performance buckets
    regime_results = []
    for lo, hi, name in zip(VIX_REGIME_BINS[:-1], VIX_REGIME_BINS[1:], VIX_REGIME_NAMES):
        mask = (df["vix_close"] >= lo) & (df["vix_close"] < hi)
        grp  = df[mask]
        if len(grp) < 5:
            continue
        entry = {
            "regime": name, "n": int(len(grp)),
            "vix_mean": round(float(grp["vix_close"].mean()), 1),
            "actual_front_crush_mean": round(float(grp["front_iv_crush_pct"].mean() * 100), 1),
            "actual_trc_mean": round(float(grp["term_ratio_change"].mean()), 4),
            "iv_expansion_count": int((grp["front_iv_crush_pct"] > 0).sum()),
        }
        if pred_df is not None:
            sub = grp.join(pred_df, how="inner")
            if len(sub) >= 5:
                sp_r, sp_p = spearmanr(sub["term_ratio_change"], sub["y_pred"])
                rmse = float(np.sqrt(mean_squared_error(sub["term_ratio_change"], sub["y_pred"])))
                entry["combined_spearman_r"] = round(float(sp_r), 4)
                entry["combined_rmse"] = round(rmse, 4)
                entry["n_oos"] = int(len(sub))
        regime_results.append(entry)
    analysis["by_regime"] = regime_results

    # Expansion events: VIX and near_back_ratio at time of IV expansion
    expansion = df[df["front_iv_crush_pct"] > 0].copy()
    expansion_summary = expansion[["symbol", "event_date", "vix_close", "vix_pct_252",
                                   "near_back_ratio", "front_iv_crush_pct",
                                   "underlying_move_pct"]].copy()
    expansion_summary["front_iv_crush_pct"] = (expansion_summary["front_iv_crush_pct"] * 100).round(1)
    expansion_summary["event_date"] = expansion_summary["event_date"].dt.strftime("%Y-%m-%d")
    analysis["expansion_events_detail"] = expansion_summary.to_dict("records")

    # Year-level model performance (from per-fold data)
    analysis["by_year_stats"] = df.groupby("year").agg(
        n=("term_ratio_change", "count"),
        vix_mean=("vix_close", "mean"),
        front_crush_mean=("front_iv_crush_pct", "mean"),
        trc_mean=("term_ratio_change", "mean"),
        crush_rate_lt40=("crush_lt_40pct", "mean"),
    ).round(4).to_dict()

    return analysis


def print_failure_analysis(fa: Dict[str, Any]) -> None:
    log.info("")
    log.info("── By VIX Regime ────────────────────────────────────────────")
    log.info("  %-22s  %5s  %6s  %13s  %10s  %6s  %10s",
             "Regime", "N", "VIX", "Front crush", "TRC mean",
             "IV exp", "OOS Spear")
    log.info("  " + "-" * 80)
    for r in fa.get("by_regime", []):
        log.info("  %-22s  %5d  %5.1f  %+12.1f%%  %+9.4f  %5d  %10s",
                 r["regime"], r["n"], r["vix_mean"],
                 r["actual_front_crush_mean"], r["actual_trc_mean"],
                 r["iv_expansion_count"],
                 f'{r["combined_spearman_r"]:.4f}' if "combined_spearman_r" in r else "n/a")

    log.info("")
    log.info("── IV Expansion Events (front crush > 0) ────────────────────")
    log.info("  %-6s  %-10s  %5s  %8s  %8s  %13s",
             "Symbol", "Date", "VIX", "VIX_pct", "NBR", "Front crush")
    for e in fa.get("expansion_events_detail", []):
        log.info("  %-6s  %-10s  %5.1f  %8.3f  %8.3f  %+12.1f%%",
                 e["symbol"], e["event_date"], e["vix_close"],
                 e["vix_pct_252"], e["near_back_ratio"], e["front_iv_crush_pct"])


# ── Step 7: verdict ────────────────────────────────────────────────────────────

def build_verdict(
    ra: Dict,
    reg_results: List[Dict],   # [baseline, vix_only, combined]
    clf_results: List[Dict],
    econ_results: List[Dict],
    fa: Dict,
) -> Dict[str, Any]:
    """Produce structured verdict: does VIX belong in production?"""
    baseline_reg = next((r for r in reg_results if "Baseline" in r["label"]), {})
    combined_reg = next((r for r in reg_results if "Combined" in r["label"]), {})
    vix_only_reg = next((r for r in reg_results if "VIX-only" in r["label"]), {})

    baseline_clf = next((r for r in clf_results if "Baseline" in r["label"]), {})
    combined_clf = next((r for r in clf_results if "Combined" in r["label"]), {})

    b_sp  = baseline_reg.get("pooled", {}).get("spearman_r", 0)
    c_sp  = combined_reg.get("pooled", {}).get("spearman_r", 0)
    v_sp  = vix_only_reg.get("pooled", {}).get("spearman_r", 0)
    b_r2  = baseline_reg.get("pooled", {}).get("r2", 0)
    c_r2  = combined_reg.get("pooled", {}).get("r2", 0)
    b_auc = (baseline_clf.get("pooled") or {}).get("auc", 0) or 0
    c_auc = (combined_clf.get("pooled") or {}).get("auc", 0) or 0

    delta_sp  = round(c_sp - b_sp, 4)
    delta_r2  = round(c_r2 - b_r2, 4)
    delta_auc = round(c_auc - b_auc, 4)

    # Partial correlation tells us if VIX is independent
    partial_r = ra["correlations"]["partial_vix_vs_trc_controlling_nbr"]["r"]
    partial_p = ra["correlations"]["partial_vix_vs_trc_controlling_nbr"]["p"]
    vix_is_independent = abs(partial_r) > 0.08 and partial_p < 0.05

    # Economic improvement
    econ_baseline = next((e for e in econ_results if e and "Baseline" in e.get("label","")), {})
    econ_combined = next((e for e in econ_results if e and "Combined" in e.get("label","")), {})
    b_spread = econ_baseline.get("q5_q1_spread_pp", 0)
    c_spread = econ_combined.get("q5_q1_spread_pp", 0)
    delta_spread = round(c_spread - b_spread, 1)

    # Fold consistency (does VIX help consistently across all folds?)
    b_fold_sp = [f.get("spearman_r", 0) for f in baseline_reg.get("folds", [])]
    c_fold_sp = [f.get("spearman_r", 0) for f in combined_reg.get("folds", [])]
    fold_improvements = [c - b for c, b in zip(c_fold_sp, b_fold_sp)]
    consistent_gain = all(d > 0 for d in fold_improvements) if fold_improvements else False

    # Final verdict
    meaningful_stat  = abs(delta_sp) >= 0.015 or abs(delta_r2) >= 0.01
    meaningful_econ  = abs(delta_spread) >= 1.5
    regime_signal    = abs(ra["correlations"]["spearman_vix_vs_trc"]["r"]) > 0.10

    vix_belongs_in_prod = (
        vix_is_independent and
        meaningful_stat and
        consistent_gain
    )

    verdict = {
        "baseline_pooled": {"spearman_r": b_sp, "r2": b_r2, "auc": b_auc},
        "combined_pooled": {"spearman_r": c_sp, "r2": c_r2, "auc": c_auc},
        "vix_only_pooled": {"spearman_r": v_sp},
        "delta": {"spearman_r": delta_sp, "r2": delta_r2, "auc": delta_auc},
        "q5_q1_spread_baseline": b_spread,
        "q5_q1_spread_combined": c_spread,
        "delta_spread_pp": delta_spread,
        "partial_vix_r": round(partial_r, 4),
        "partial_vix_p": round(partial_p, 4),
        "vix_is_independent_of_nbr": vix_is_independent,
        "meaningful_statistical_gain": meaningful_stat,
        "meaningful_economic_gain": meaningful_econ,
        "consistent_across_folds": consistent_gain,
        "fold_spearman_deltas": fold_improvements,
        "regime_signal_exists": regime_signal,
        "vix_belongs_in_production": vix_belongs_in_prod,
        "summary": (
            "INCLUDE: VIX provides real incremental signal beyond near_back_ratio."
            if vix_belongs_in_prod else
            "MARGINAL: VIX may add calibration value but does not reliably improve ranking."
            if (regime_signal and not consistent_gain) else
            "EXCLUDE: VIX does not provide incremental signal beyond existing features."
        ),
    }
    return verdict


def print_verdict(v: Dict) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 7 — VERDICT: DOES VIX BELONG IN PRODUCTION?")
    log.info("=" * 72)
    log.info("")
    log.info("  %-40s  %10s  %10s  %10s",
             "Model", "Spearman_r", "R²(TRC)", "AUC(cls)")
    log.info("  " + "-" * 74)
    log.info("  %-40s  %10.4f  %10.4f  %10.4f",
             "Baseline (3 features)",
             v["baseline_pooled"]["spearman_r"],
             v["baseline_pooled"]["r2"],
             v["baseline_pooled"]["auc"])
    log.info("  %-40s  %10.4f  %10s  %10s",
             "VIX-only (2 VIX features)",
             v["vix_only_pooled"]["spearman_r"], "—", "—")
    log.info("  %-40s  %10.4f  %10.4f  %10.4f",
             "Combined (5 features)",
             v["combined_pooled"]["spearman_r"],
             v["combined_pooled"]["r2"],
             v["combined_pooled"]["auc"])
    log.info("  %-40s  %+9.4f  %+9.4f  %+9.4f",
             "Δ Combined − Baseline",
             v["delta"]["spearman_r"], v["delta"]["r2"], v["delta"]["auc"])
    log.info("")
    log.info("  Economic (Q5-Q1 spread):  Baseline %+.1f pp → Combined %+.1f pp  (Δ %+.1f pp)",
             v["q5_q1_spread_baseline"], v["q5_q1_spread_combined"], v["delta_spread_pp"])
    log.info("")
    log.info("  Partial corr VIX|NBR vs TRC|NBR:  r=%+.4f  p=%.4f",
             v["partial_vix_r"], v["partial_vix_p"])
    log.info("  VIX independent of near_back_ratio: %s",
             "YES" if v["vix_is_independent_of_nbr"] else "NO")
    log.info("  Consistent gain across all folds:   %s  (deltas: %s)",
             "YES" if v["consistent_across_folds"] else "NO",
             [f"{d:+.4f}" for d in v["fold_spearman_deltas"]])
    log.info("  Meaningful statistical gain (≥0.015 Spearman): %s",
             "YES" if v["meaningful_statistical_gain"] else "NO")
    log.info("  Meaningful economic gain (≥1.5 pp spread):     %s",
             "YES" if v["meaningful_economic_gain"] else "NO")
    log.info("")
    log.info("  *** VERDICT: %s ***", v["summary"])


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Week 3 VIX regime research pipeline")
    p.add_argument("--db-path", default=None)
    p.add_argument("--quality-tier", default="AB", choices=["A","AB"])
    p.add_argument("--report-dir", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    db_path    = Path(args.db_path).expanduser() if args.db_path else DEFAULT_DB_PATH
    report_dir = Path(args.report_dir).expanduser() if args.report_dir else DEFAULT_REPORT_DIR
    quality_tiers = list(args.quality_tier)

    log.info("=" * 72)
    log.info("WEEK 3 VIX REGIME RESEARCH  version=%s", RESEARCH_VERSION)
    log.info("=" * 72)
    log.info("DB        : %s", db_path)
    log.info("Quality   : tiers %s", quality_tiers)
    log.info("VIX data  : %s", VIX_PARQUET_GLOB)
    log.info("Leakage guarantee: VIX joined on pre_capture_date ONLY")
    log.info("Trailing pct window: %d trading days", VIX_TRAILING_WINDOW)
    log.info("")

    # ── load data ─────────────────────────────────────────────────────────────
    log.info("── Loading data ─────────────────────────────────────────────")
    vix_raw  = load_vix_series()
    vix_feat = compute_vix_features(vix_raw)
    labels   = load_labels(db_path, quality_tiers)
    df       = engineer_features(labels, vix_feat)

    # Quick VIX summary for the dataset
    log.info("VIX at pre-event dates: mean=%.1f  min=%.1f  max=%.1f  std=%.1f",
             df["vix_close"].mean(), df["vix_close"].min(),
             df["vix_close"].max(), df["vix_close"].std())

    # ── Step 2: regime analysis ────────────────────────────────────────────────
    ra = regime_analysis(df)
    print_regime_analysis(ra)

    # ── Step 3: walk-forward models (3 feature sets × 2 targets) ─────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 3 — WALK-FORWARD MODEL COMPARISON")
    log.info("=" * 72)
    log.info("Three feature sets × two targets. Same folds and hyperparams as Week 2.")

    # Regression: term_ratio_change (primary)
    log.info("")
    log.info("── Regression target: term_ratio_change ─────────────────────")
    log.info("")
    log.info("  A) Baseline (3 features):")
    r_base = wf_regression(df, PRIMARY_REG_TARGET, BASELINE_FEATURES, "Baseline (3 feat)")
    for f in r_base["folds"]:
        log.info("    Fold train=%s→test=%d  Spearman=%.4f  R²=%.4f  RMSE/naive=%.4f",
                 f["train_years"], f["test_year"],
                 f["spearman_r"], f["r2"], f["rmse_ratio"])

    log.info("")
    log.info("  B) VIX-only (2 VIX features):")
    r_vix = wf_regression(df, PRIMARY_REG_TARGET, VIX_FEATURES, "VIX-only (2 feat)")
    for f in r_vix["folds"]:
        log.info("    Fold train=%s→test=%d  Spearman=%.4f  R²=%.4f  RMSE/naive=%.4f",
                 f["train_years"], f["test_year"],
                 f["spearman_r"], f["r2"], f["rmse_ratio"])

    log.info("")
    log.info("  C) Combined (5 features):")
    r_comb = wf_regression(df, PRIMARY_REG_TARGET, COMBINED_FEATURES, "Combined (5 feat)")
    for f in r_comb["folds"]:
        log.info("    Fold train=%s→test=%d  Spearman=%.4f  R²=%.4f  RMSE/naive=%.4f  coef_vix=%s",
                 f["train_years"], f["test_year"],
                 f["spearman_r"], f["r2"], f["rmse_ratio"],
                 {k: v for k, v in f["coef"].items() if "vix" in k})

    log.info("")
    log.info("── Regression comparison (term_ratio_change) ─────────────────")
    print_model_comparison([r_base, r_vix, r_comb], "regression")

    # Classification: crush_lt_40pct (secondary)
    log.info("")
    log.info("── Classification target: crush_lt_40pct ─────────────────────")
    c_base = wf_classification(df, SECONDARY_CLF_TARGET, BASELINE_FEATURES, "Baseline (3 feat)")
    c_vix  = wf_classification(df, SECONDARY_CLF_TARGET, VIX_FEATURES, "VIX-only (2 feat)")
    c_comb = wf_classification(df, SECONDARY_CLF_TARGET, COMBINED_FEATURES, "Combined (5 feat)")
    log.info("")
    log.info("── Classification comparison (crush_lt_40pct) ────────────────")
    print_model_comparison([c_base, c_vix, c_comb], "classification")

    # ── Step 4: ablation ────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 4 — ABLATION: ISOLATING VIX CONTRIBUTION")
    log.info("=" * 72)
    log.info("")
    log.info("  Baseline (NBR + log_front + log_back):  Spearman=%.4f  R²=%.4f",
             r_base["pooled"]["spearman_r"], r_base["pooled"]["r2"])
    log.info("  VIX-only (log_vix + vix_pct):           Spearman=%.4f  R²=%.4f",
             r_vix["pooled"]["spearman_r"], r_vix["pooled"]["r2"])
    log.info("  Combined (all 5):                       Spearman=%.4f  R²=%.4f",
             r_comb["pooled"]["spearman_r"], r_comb["pooled"]["r2"])
    log.info("")
    log.info("  Incremental gain (Combined - Baseline): ΔSpearman=%+.4f  ΔR²=%+.4f",
             r_comb["pooled"]["spearman_r"] - r_base["pooled"]["spearman_r"],
             r_comb["pooled"]["r2"] - r_base["pooled"]["r2"])
    log.info("  VIX-only vs Baseline: %s",
             "VIX alone is WEAKER than baseline (near_back_ratio dominates)"
             if r_vix["pooled"]["spearman_r"] < r_base["pooled"]["spearman_r"]
             else "VIX alone MATCHES baseline (strong independent regime signal)")
    log.info("")
    log.info("  Interpretation:")
    log.info("  If Combined >> Baseline but VIX-only << Baseline:")
    log.info("    → VIX adds calibration, not independent discriminative power.")
    log.info("  If VIX-only ~= Baseline and Combined > either:")
    log.info("    → VIX is an independent, complementary signal.")

    # ── Step 5: economic evaluation ───────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 5 — ECONOMIC BUCKET EVALUATION")
    log.info("=" * 72)
    log.info("Using Ridge → term_ratio_change predictions (regression model).")

    # Use negative of y_pred so Q5 = deepest predicted crush (most negative TRC)
    def flip_preds(preds):
        return [{**p, "y_pred": -p["y_pred"]} for p in preds]

    e_base = economic_eval(df, flip_preds(r_base["oos_preds"]), "y_pred", "Baseline (3 feat)")
    e_vix  = economic_eval(df, flip_preds(r_vix["oos_preds"]),  "y_pred", "VIX-only (2 feat)")
    e_comb = economic_eval(df, flip_preds(r_comb["oos_preds"]), "y_pred", "Combined (5 feat)")

    log.info("")
    log.info("── Economic comparison (Q5 = predicted deepest crush) ─────────")
    print_economic_comparison([e_base, e_vix, e_comb])

    log.info("")
    log.info("── Quintile table: Baseline ─────────────────────────────────")
    print_quintile_table(e_base)

    log.info("")
    log.info("── Quintile table: Combined ─────────────────────────────────")
    print_quintile_table(e_comb)

    # ── Step 6: failure mode analysis ────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 6 — FAILURE MODE ANALYSIS")
    log.info("=" * 72)
    fa = failure_mode_analysis(df, r_comb["oos_preds"])
    print_failure_analysis(fa)

    # ── Step 7: verdict ──────────────────────────────────────────────────────
    verdict = build_verdict(
        ra,
        [r_base, r_vix, r_comb],
        [c_base, c_vix, c_comb],
        [e_base, e_vix, e_comb],
        fa,
    )
    print_verdict(verdict)

    # Week 4 recommendation
    log.info("")
    log.info("── Week 4 Recommendation ────────────────────────────────────")
    if verdict["vix_belongs_in_production"]:
        log.info("  VIX included: proceed to Week 4 with 5-feature set.")
        log.info("  Next: add prior_crush_history per symbol (symbol-level prior)")
        log.info("  Next: test sector encoding (energy outlier suppression)")
        log.info("  Next: compare Ridge vs shallow GBM on 5-feature set")
    elif verdict["regime_signal_exists"]:
        log.info("  VIX has regime-level signal but inconsistent incremental gain.")
        log.info("  Recommendation: include VIX for calibration, not ranking.")
        log.info("  Consider: regime-conditional models (high-VIX vs low-VIX submodels)")
        log.info("  Next: add prior_crush_history (likely more impactful than VIX)")
    else:
        log.info("  VIX does not add incremental value. Keep 3-feature baseline.")
        log.info("  Next: add prior_crush_history per symbol")
        log.info("  Next: test whether sector or DTE features improve on baseline")

    # ── save report ───────────────────────────────────────────────────────────
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"vix_regime_{RESEARCH_VERSION}_{ts}.json"

    def _safe(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    report = {
        "research_version": RESEARCH_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "db_path": str(db_path), "quality_tiers": quality_tiers,
            "vix_trailing_window": VIX_TRAILING_WINDOW,
            "baseline_features": BASELINE_FEATURES,
            "vix_features": VIX_FEATURES,
            "combined_features": COMBINED_FEATURES,
            "leakage_note": "VIX joined on pre_capture_date only. Trailing pct uses data ≤ pre_capture_date.",
        },
        "regime_analysis": {
            "quintile_bins": ra["quintile_bins"],
            "correlations": ra["correlations"],
            "nbr_by_vix_quintile": ra["nbr_by_vix_quintile"],
        },
        "regression_results": {
            "baseline": {k: v for k, v in r_base.items() if k != "oos_preds"},
            "vix_only": {k: v for k, v in r_vix.items()  if k != "oos_preds"},
            "combined": {k: v for k, v in r_comb.items() if k != "oos_preds"},
        },
        "classification_results": {
            "baseline": {k: v for k, v in c_base.items() if k != "oos_preds"},
            "vix_only": {k: v for k, v in c_vix.items()  if k != "oos_preds"},
            "combined": {k: v for k, v in c_comb.items() if k != "oos_preds"},
        },
        "economic_eval": {
            "baseline": e_base, "vix_only": e_vix, "combined": e_comb
        },
        "failure_mode_analysis": fa,
        "verdict": verdict,
    }
    report_path.write_text(json.dumps(report, indent=2, default=_safe))
    log.info("")
    log.info("Research report saved to: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
