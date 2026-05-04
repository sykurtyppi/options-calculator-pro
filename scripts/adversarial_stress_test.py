#!/usr/bin/env python3
"""
scripts/adversarial_stress_test.py
====================================
Adversarial Validation — "Is the signal real?"

Assumes the model is WRONG until proven otherwise.

Steps
-----
  Step 1  Feature dominance & mechanical linkage
           - near_back_ratio alone vs 3-feature baseline vs VIX-only vs full 5-feature
           - Algebraic/mechanical decomposition of term_ratio_change
  Step 2  Permutation / placebo test (mandatory)
           - Shuffle targets within same year (N_PERMUTATIONS times)
           - Expected: performance collapses. Failure to collapse = leakage flag.
  Step 3  Symbol-level robustness
           - High-frequency vs low-frequency symbols
           - Leave-one-symbol-out sampling (top-5 symbols removed)
  Step 4  Regime stability
           - Per-year OOS performance comparison
           - Stability of Q5-Q1 economic spread across years
  Step 5  Economic realism check
           - Calendar spread payoff approximation per quintile
           - Front crush + back crush differential → realistic P&L proxy
  Step 6  Failure mode identification
           - Symbols with worst prediction error
           - High-confidence-wrong predictions
           - IV expansion events (model predicts crush, reality expands)
  Step 7  VIX contribution validation
           - Δ metrics: baseline → combined
           - Regime stability with/without VIX
           - VIX-only model
  Step 8  Verdict
           - Signal attribution breakdown
           - Fragility flags
           - Trading-readiness assessment

Engineering guarantees (identical to Week 2/3)
-----------------------------------------------
  - Walk-forward only: scaler fit on train fold, never test
  - No hyperparameter search
  - All feature derivation from pre-event data only
  - Permutation shuffles within year to respect temporal structure

Usage
-----
  python scripts/adversarial_stress_test.py [options]
  python scripts/adversarial_stress_test.py --quality-tier A --permutations 500
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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
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
DEFAULT_DB_PATH   = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
DEFAULT_REPORT_DIR = Path.home() / ".options_calculator_pro" / "reports"
RESEARCH_VERSION  = "adversarial_v1"

VIX_PARQUET_GLOB  = (
    "/Volumes/T9/market_data/normalized/underlyings/"
    "daily_ohlcv/underlying_symbol=^VIX/**/*.parquet"
)
VIX_PRICE_SCALE   = 10_000.0
VIX_TRAILING_WINDOW = 252

N_PERMUTATIONS_DEFAULT = 200   # permutation test iterations
RIDGE_ALPHA = 1.0
LR_C        = 1.0
LR_SOLVER   = "lbfgs"
LR_MAX_ITER = 500

PRIMARY_REG_TARGET   = "term_ratio_change"
SECONDARY_CLF_TARGET = "crush_lt_40pct"

WALK_FORWARD_FOLDS: List[Tuple[List[int], int]] = [
    ([2023],             2024),
    ([2023, 2024],       2025),
    ([2023, 2024, 2025], 2026),
]

# Feature set definitions
FS_NBR_ONLY  = ["near_back_ratio"]
FS_BASELINE  = ["near_back_ratio", "log_front_iv", "log_back_iv"]
FS_VIX_ONLY  = ["log_vix", "vix_pct_252"]
FS_COMBINED  = ["near_back_ratio", "log_front_iv", "log_back_iv", "log_vix", "vix_pct_252"]


# ── helpers: walk-forward engine ─────────────────────────────────────────────

def _fold_data(df, train_years, test_year, features, target):
    tr = df[df["year"].isin(train_years)].dropna(subset=features + [target])
    te = df[df["year"] == test_year].dropna(subset=features + [target])
    if len(tr) < 5 or len(te) < 5:
        return None
    sc = StandardScaler()
    X_tr = sc.fit_transform(tr[features].values)
    X_te = sc.transform(te[features].values)
    return X_tr, tr[target].values, X_te, te[target].values, te.index.tolist()


def wf_ridge(df, features, target="term_ratio_change", label=""):
    """Walk-forward Ridge. Returns pooled metrics + per-fold breakdown."""
    oos: List[Dict] = []
    folds: List[Dict] = []
    for tr_yrs, te_yr in WALK_FORWARD_FOLDS:
        fd = _fold_data(df, tr_yrs, te_yr, features, target)
        if fd is None:
            continue
        X_tr, y_tr, X_te, y_te, idx = fd
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)
        naive = np.full_like(y_te, y_tr.mean())
        sp_r, _ = spearmanr(y_te, y_hat)
        r2 = float(r2_score(y_te, y_hat))
        rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
        rmse_naive = float(np.sqrt(mean_squared_error(y_te, naive)))
        coef = {f: round(float(c), 5) for f, c in zip(features, m.coef_)}
        folds.append({
            "train_years": tr_yrs, "test_year": te_yr,
            "n_train": len(X_tr), "n_test": len(X_te),
            "spearman_r": round(float(sp_r), 4),
            "r2": round(r2, 4),
            "rmse": round(rmse, 6),
            "rmse_ratio": round(rmse / rmse_naive, 4),
            "coef": coef,
        })
        for i, yp in zip(idx, y_hat.tolist()):
            oos.append({"idx": i, "y_true": float(y_te[idx.index(i)]), "y_pred": yp})
    pooled = _pool_reg(oos)
    return {"label": label, "features": features, "folds": folds,
            "pooled": pooled, "oos": oos}


def _pool_reg(oos):
    if not oos:
        return {}
    yt = np.array([x["y_true"] for x in oos])
    yp = np.array([x["y_pred"] for x in oos])
    naive = np.full_like(yt, yt.mean())
    sp_r, sp_p = spearmanr(yt, yp)
    return {
        "n": len(yt),
        "spearman_r": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 6),
        "r2": round(float(r2_score(yt, yp)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 6),
        "rmse_ratio": round(float(np.sqrt(mean_squared_error(yt, yp))) /
                            float(np.sqrt(mean_squared_error(yt, naive))), 4),
    }


def wf_lr(df, features, target="crush_lt_40pct", label=""):
    """Walk-forward LR. Returns pooled AUC + Brier."""
    oos: List[Dict] = []
    folds: List[Dict] = []
    for tr_yrs, te_yr in WALK_FORWARD_FOLDS:
        fd = _fold_data(df, tr_yrs, te_yr, features, target)
        if fd is None:
            continue
        X_tr, y_tr, X_te, y_te, idx = fd
        if y_tr.mean() < 0.03 or y_tr.mean() > 0.97:
            continue
        m = LogisticRegression(C=LR_C, solver=LR_SOLVER, max_iter=LR_MAX_ITER,
                               random_state=42)
        m.fit(X_tr, y_tr)
        y_prob = m.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, y_prob)) if 0 < y_te.mean() < 1 else float("nan")
        brier = float(brier_score_loss(y_te, y_prob))
        folds.append({
            "train_years": tr_yrs, "test_year": te_yr,
            "n_train": len(X_tr), "n_test": len(X_te),
            "auc": round(auc, 4) if not math.isnan(auc) else None,
            "brier": round(brier, 4),
        })
        for i, yp in zip(idx, y_prob.tolist()):
            oos.append({"idx": i, "y_true": float(y_te[idx.index(i)]), "y_prob": yp})
    pooled = {}
    if oos:
        yt = np.array([x["y_true"] for x in oos])
        yp = np.array([x["y_prob"] for x in oos])
        pooled = {
            "n": len(yt),
            "auc": round(float(roc_auc_score(yt, yp)), 4) if 0 < yt.mean() < 1 else None,
            "brier": round(float(brier_score_loss(yt, yp)), 4),
        }
    return {"label": label, "features": features, "folds": folds,
            "pooled": pooled, "oos": oos}


def quintile_spread(df, oos, pred_col="y_pred"):
    """Q5-Q1 average front crush spread (pp). Q5 = deepest predicted crush."""
    if not oos:
        return float("nan"), []
    pred_df = pd.DataFrame(oos).rename(columns={"idx": "idx"}).set_index("idx")
    merged = df[["front_iv_crush_pct", "back_iv_crush_pct", "year",
                 "spread_differential"]].join(pred_df, how="inner")
    if len(merged) < 10:
        return float("nan"), []
    # Flip so Q5 = highest predicted crush magnitude
    merged["score"] = -merged[pred_col]
    try:
        merged["q"] = pd.qcut(merged["score"], 5, labels=["Q1","Q2","Q3","Q4","Q5"],
                              duplicates="drop")
    except ValueError:
        return float("nan"), []
    agg = merged.groupby("q", observed=True).agg(
        n=("front_iv_crush_pct", "count"),
        avg_front_pct=("front_iv_crush_pct", lambda x: round(x.mean() * 100, 1)),
        avg_back_pct=("back_iv_crush_pct",  lambda x: round(x.mean() * 100, 1)),
        avg_diff_pct=("spread_differential", lambda x: round(x.mean() * 100, 1)),
    ).reset_index()
    table = agg.to_dict("records")
    q1 = agg.loc[agg["q"] == "Q1", "avg_front_pct"].values
    q5 = agg.loc[agg["q"] == "Q5", "avg_front_pct"].values
    spread = float(q5[0] - q1[0]) if len(q1) and len(q5) else float("nan")
    return spread, table


# ── data loading ──────────────────────────────────────────────────────────────

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
    df["year"]             = df["event_date"].dt.year
    df["quarter"]          = df["event_date"].dt.quarter
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["near_back_ratio"] = df["pre_front_iv"] / df["pre_back_iv"]
    df["log_front_iv"]    = np.log(df["pre_front_iv"])
    df["log_back_iv"]     = np.log(df["pre_back_iv"])
    df["crush_lt_40pct"]  = (df["front_iv_crush_pct"] < -0.40).astype(int)
    df["spread_differential"] = df["front_iv_crush_pct"] - df["back_iv_crush_pct"]
    return df


def try_load_vix(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to attach VIX features. Returns df unchanged if VIX unavailable."""
    try:
        con = duckdb.connect(":memory:")
        vix = con.execute(f"""
            SELECT trade_date, close_10000 / {VIX_PRICE_SCALE} AS vix_close
            FROM read_parquet('{VIX_PARQUET_GLOB}')
            WHERE close_10000 IS NOT NULL AND close_10000 > 0
            ORDER BY trade_date
        """).df()
        con.close()
        vix["trade_date"] = pd.to_datetime(vix["trade_date"])
        vix = vix.sort_values("trade_date").reset_index(drop=True)
        # Trailing percentile
        pcts = np.full(len(vix), np.nan)
        for i in range(len(vix)):
            s = max(0, i - VIX_TRAILING_WINDOW + 1)
            w = vix["vix_close"].values[s:i + 1]
            pcts[i] = (w <= vix["vix_close"].values[i]).mean()
        vix["log_vix"]     = np.log(vix["vix_close"])
        vix["vix_pct_252"] = pcts
        df = df.merge(
            vix[["trade_date", "vix_close", "log_vix", "vix_pct_252"]],
            left_on="pre_capture_date", right_on="trade_date", how="left"
        )
        n_miss = df["log_vix"].isna().sum()
        log.info("VIX loaded: %d days. Missing after join: %d / %d",
                 len(vix), n_miss, len(df))
    except Exception as exc:
        log.warning("VIX data unavailable (%s) — VIX steps will be skipped.", exc)
        df["log_vix"] = np.nan
        df["vix_pct_252"] = np.nan
        df["vix_close"] = np.nan
    return df


# ── STEP 1: feature dominance & mechanical linkage ────────────────────────────

def step1_feature_dominance(df: pd.DataFrame) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 1 — FEATURE DOMINANCE & MECHANICAL LINKAGE")
    log.info("=" * 72)

    # 1a. Algebraic decomposition of term_ratio_change
    # term_ratio_change = post_TRC - pre_TRC
    #                   = (post_front/post_back) - (pre_front/pre_back)
    # near_back_ratio   = pre_front / pre_back
    # So: term_ratio_change = post_TRC - near_back_ratio
    # → partial algebraic link, NOT a tautology (post_TRC is the unknown)
    df2 = df.dropna(subset=["near_back_ratio", "term_ratio_change",
                             "front_iv_crush_pct"]).copy()
    pr_nbr_trc, pp_nbr_trc = pearsonr(df2["near_back_ratio"], df2["term_ratio_change"])
    sp_nbr_trc, _          = spearmanr(df2["near_back_ratio"], df2["term_ratio_change"])
    pr_nbr_fc,  _          = pearsonr(df2["near_back_ratio"], df2["front_iv_crush_pct"])
    sp_nbr_fc,  _          = spearmanr(df2["near_back_ratio"], df2["front_iv_crush_pct"])

    # Algebraic bound: if near_back_ratio alone explains TRC mechanically,
    # then TRC ≈ f(NBR) × constant. Check by regressing TRC on NBR alone.
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(df2[["near_back_ratio"]].values, df2["term_ratio_change"].values)
    trc_hat = lr.predict(df2[["near_back_ratio"]].values)
    r2_nbr_on_trc = float(r2_score(df2["term_ratio_change"].values, trc_hat))

    # OLS slope interpretation: 1-unit increase in NBR → coef change in TRC
    nbr_coef = float(lr.coef_[0])
    nbr_intercept = float(lr.intercept_)

    log.info("")
    log.info("── 1a. Algebraic / mechanical linkage analysis ──────────────────")
    log.info("  term_ratio_change = post_TRC - near_back_ratio (algebraic decomposition)")
    log.info("  → near_back_ratio = pre_front/pre_back is the PRE-event TRC")
    log.info("  → This means TRC = (post_front/post_back) - (pre_front/pre_back)")
    log.info("  → NOT a tautology: post values are unknown pre-event")
    log.info("")
    log.info("  Correlation(NBR, TRC):   Pearson r=%.4f  Spearman r=%.4f",
             pr_nbr_trc, sp_nbr_trc)
    log.info("  Correlation(NBR, front_crush): Pearson r=%.4f  Spearman r=%.4f",
             pr_nbr_fc, sp_nbr_fc)
    log.info("  OLS(TRC ~ NBR):  coef=%.4f  intercept=%.4f  R²=%.4f",
             nbr_coef, nbr_intercept, r2_nbr_on_trc)
    log.info("")
    if abs(pr_nbr_trc) > 0.8:
        log.info("  ⚠  WARNING: Pearson(NBR, TRC) > 0.8 — potential mechanical linkage.")
        log.info("     NBR appears on both sides of the equation (it IS pre_TRC).")
        log.info("     Model may be learning an accounting identity, not economics.")
    elif abs(pr_nbr_trc) > 0.5:
        log.info("  ⚡ MODERATE: Pearson(NBR, TRC) ∈ (0.5, 0.8) — partial mechanical link.")
        log.info("     Some of the signal may be algebraic; the rest is real information.")
    else:
        log.info("  ✓  NBR-TRC correlation is modest — signal is not purely mechanical.")

    # 1b. Walk-forward ablation: 4 model sizes
    has_vix = bool(df["log_vix"].notna().mean() > 0.3)
    feature_sets = [
        ("NBR-only (1 feat)",   FS_NBR_ONLY),
        ("Baseline (3 feat)",   FS_BASELINE),
    ]
    if has_vix:
        feature_sets += [
            ("VIX-only (2 feat)",   FS_VIX_ONLY),
            ("Combined (5 feat)",   FS_COMBINED),
        ]

    reg_results = []
    clf_results = []
    for lbl, feats in feature_sets:
        log.info("  Running walk-forward Ridge: %s …", lbl)
        r = wf_ridge(df, feats, PRIMARY_REG_TARGET, lbl)
        c = wf_lr(df, feats, SECONDARY_CLF_TARGET, lbl)
        reg_results.append(r)
        clf_results.append(c)

    log.info("")
    log.info("── 1b. Walk-forward ablation results ────────────────────────────")
    log.info("  %-32s  %10s  %10s  %10s",
             "Model", "Spearman_r", "R²(TRC)", "RMSE/naive")
    log.info("  " + "-" * 66)
    for r in reg_results:
        p = r["pooled"]
        log.info("  %-32s  %10.4f  %10.4f  %10.4f",
                 r["label"],
                 p.get("spearman_r", float("nan")),
                 p.get("r2", float("nan")),
                 p.get("rmse_ratio", float("nan")))

    log.info("")
    log.info("  %-32s  %10s  %10s", "Model", "AUC", "Brier")
    log.info("  " + "-" * 54)
    for c in clf_results:
        p = c["pooled"]
        log.info("  %-32s  %10s  %10.4f",
                 c["label"],
                 f'{p["auc"]:.4f}' if p.get("auc") else "n/a",
                 p.get("brier", float("nan")))

    # 1c. Coefficient stability (is NBR always dominant?)
    log.info("")
    log.info("── 1c. Coefficient stability (NBR-only) ─────────────────────────")
    nbr_only = next(r for r in reg_results if "NBR-only" in r["label"])
    base_reg  = next(r for r in reg_results if "Baseline" in r["label"])
    nbr_sp   = nbr_only["pooled"].get("spearman_r", float("nan"))
    base_sp  = base_reg["pooled"].get("spearman_r", float("nan"))
    pct_from_nbr = nbr_sp / base_sp * 100 if base_sp else float("nan")
    log.info("  NBR-only  Spearman=%.4f", nbr_sp)
    log.info("  Baseline  Spearman=%.4f  → NBR explains ~%.0f%% of baseline signal",
             base_sp, pct_from_nbr)
    if pct_from_nbr > 90:
        log.info("  ⚠  NBR captures ≥90%% of baseline signal — log_front + log_back are REDUNDANT")
    elif pct_from_nbr > 75:
        log.info("  ⡠  NBR captures >75%% of signal — other features add modest lift only")
    else:
        log.info("  ✓  NBR + other features contribute meaningfully beyond NBR alone")

    # 1d. Feature correlation matrix
    log.info("")
    log.info("── 1d. Feature × target correlation matrix ──────────────────────")
    all_feats = ["near_back_ratio", "log_front_iv", "log_back_iv",
                 "term_ratio_change", "front_iv_crush_pct", "crush_lt_40pct"]
    if has_vix:
        all_feats = ["near_back_ratio", "log_front_iv", "log_back_iv",
                     "log_vix", "vix_pct_252",
                     "term_ratio_change", "front_iv_crush_pct", "crush_lt_40pct"]
    corr_df = df[all_feats].dropna()
    corr_mat = corr_df.corr(method="spearman")

    log.info("  Spearman correlation matrix:")
    header = f"  {'':22s}" + "".join(f"  {c[:10]:>12s}" for c in all_feats)
    log.info(header)
    for r_name in all_feats:
        row_str = f"  {r_name:22s}"
        for c_name in all_feats:
            v = corr_mat.loc[r_name, c_name]
            row_str += f"  {v:+12.3f}"
        log.info(row_str)

    return {
        "algebraic": {
            "pearson_nbr_trc": round(pr_nbr_trc, 4),
            "spearman_nbr_trc": round(sp_nbr_trc, 4),
            "r2_nbr_on_trc": round(r2_nbr_on_trc, 4),
            "nbr_coef": round(nbr_coef, 4),
            "mechanical_link_flag": bool(abs(pr_nbr_trc) > 0.8),
        },
        "ablation": {
            r["label"]: r["pooled"] for r in reg_results
        },
        "clf_ablation": {
            c["label"]: c["pooled"] for c in clf_results
        },
        "nbr_pct_of_baseline": round(pct_from_nbr, 1),
        "has_vix": has_vix,
        "reg_results": reg_results,
        "clf_results": clf_results,
    }


# ── STEP 2: permutation / placebo test ───────────────────────────────────────

def step2_permutation(df: pd.DataFrame, n_perm: int,
                      features: List[str]) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 2 — PERMUTATION / PLACEBO TEST  (N=%d)", n_perm)
    log.info("=" * 72)
    log.info("  Procedure: shuffle term_ratio_change WITHIN each year,")
    log.info("  retrain identical pipeline, collect null distribution.")
    log.info("  Expected: Spearman → ~0 under null. Leakage if it does not.")

    rng = np.random.default_rng(seed=0)
    null_sp = []
    null_r2 = []
    null_auc = []

    for perm_i in range(n_perm):
        perm_df = df.copy()
        # Shuffle target within each year (preserves year-level marginal distribution)
        for yr, grp in perm_df.groupby("year"):
            shuffled = grp["term_ratio_change"].values.copy()
            rng.shuffle(shuffled)
            perm_df.loc[grp.index, "term_ratio_change"] = shuffled
            # Also shuffle binary target within year
            shuffled_bin = grp["crush_lt_40pct"].values.copy()
            rng.shuffle(shuffled_bin)
            perm_df.loc[grp.index, "crush_lt_40pct"] = shuffled_bin

        r = wf_ridge(perm_df, features, "term_ratio_change", "perm")
        c = wf_lr(perm_df, features, "crush_lt_40pct", "perm")
        null_sp.append(r["pooled"].get("spearman_r", float("nan")))
        null_r2.append(r["pooled"].get("r2", float("nan")))
        auc = (c["pooled"] or {}).get("auc")
        null_auc.append(auc if auc is not None else float("nan"))

    null_sp  = np.array([x for x in null_sp  if not math.isnan(x)])
    null_r2  = np.array([x for x in null_r2  if not math.isnan(x)])
    null_auc = np.array([x for x in null_auc if not math.isnan(x)])

    # Real performance for comparison (same feature set, real targets)
    real = wf_ridge(df, features, "term_ratio_change", "real")
    real_clf = wf_lr(df, features, "crush_lt_40pct", "real")
    real_sp  = real["pooled"].get("spearman_r", float("nan"))
    real_r2  = real["pooled"].get("r2", float("nan"))
    real_auc = (real_clf["pooled"] or {}).get("auc", float("nan"))

    # Empirical p-values: fraction of null >= real
    p_sp  = float((null_sp  >= real_sp).mean())  if len(null_sp)  else float("nan")
    p_r2  = float((null_r2  >= real_r2).mean())  if len(null_r2)  else float("nan")
    p_auc = float((null_auc >= real_auc).mean()) if len(null_auc) else float("nan")

    # Z-scores
    z_sp  = (real_sp  - null_sp.mean())  / null_sp.std()  if null_sp.std()  > 0 else float("nan")
    z_r2  = (real_r2  - null_r2.mean())  / null_r2.std()  if null_r2.std()  > 0 else float("nan")
    z_auc = (real_auc - null_auc.mean()) / null_auc.std() if null_auc.std() > 0 else float("nan")

    log.info("")
    log.info("  Results (N=%d permutations, shuffle within year):", n_perm)
    log.info("  %-22s  %10s  %10s  %12s  %8s",
             "Metric", "Real", "Null mean", "Null p95", "p-value")
    log.info("  " + "-" * 66)

    def _p95(arr): return float(np.percentile(arr, 95)) if len(arr) else float("nan")

    log.info("  %-22s  %10.4f  %10.4f  %12.4f  %8.4f  z=%.2f",
             "Spearman_r (TRC)", real_sp,
             null_sp.mean() if len(null_sp) else float("nan"), _p95(null_sp), p_sp, z_sp)
    log.info("  %-22s  %10.4f  %10.4f  %12.4f  %8.4f  z=%.2f",
             "R² (TRC)", real_r2,
             null_r2.mean() if len(null_r2) else float("nan"), _p95(null_r2), p_r2, z_r2)
    log.info("  %-22s  %10s  %10.4f  %12.4f  %8.4f  z=%.2f",
             "AUC (binary)",
             f"{real_auc:.4f}" if not math.isnan(real_auc) else "n/a",
             null_auc.mean() if len(null_auc) else float("nan"), _p95(null_auc), p_auc, z_auc)

    log.info("")
    if p_sp < 0.01 and p_auc < 0.01:
        log.info("  ✓  SIGNAL IS REAL: p < 0.01 for both Spearman and AUC.")
        log.info("     Null distribution does not overlap with real performance.")
        verdict = "REAL"
    elif p_sp < 0.05 or p_auc < 0.05:
        log.info("  ⡠  MARGINAL: p < 0.05 for at least one metric but not both.")
        log.info("     Signal is statistically present but not robustly separated.")
        verdict = "MARGINAL"
    else:
        log.info("  ✗  LEAKAGE OR NO SIGNAL: permutation p-value ≥ 0.05.")
        log.info("     Real performance does NOT exceed the null distribution.")
        log.info("     Investigate for leakage immediately.")
        verdict = "FAILED"

    return {
        "n_permutations": n_perm,
        "features": features,
        "real_spearman": round(real_sp, 4),
        "real_r2": round(real_r2, 4),
        "real_auc": round(real_auc, 4) if not math.isnan(real_auc) else None,
        "null_spearman_mean": round(float(null_sp.mean()), 4) if len(null_sp) else None,
        "null_spearman_p95":  round(_p95(null_sp), 4),
        "null_auc_mean":      round(float(null_auc.mean()), 4) if len(null_auc) else None,
        "null_auc_p95":       round(_p95(null_auc), 4),
        "p_value_spearman":   round(p_sp, 4),
        "p_value_r2":         round(p_r2, 4),
        "p_value_auc":        round(p_auc, 4),
        "z_spearman": round(z_sp, 2) if not math.isnan(z_sp) else None,
        "z_auc":      round(z_auc, 2) if not math.isnan(z_auc) else None,
        "verdict": verdict,
    }


# ── STEP 3: symbol-level robustness ──────────────────────────────────────────

def step3_symbol_robustness(df: pd.DataFrame,
                            features: List[str]) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 3 — SYMBOL-LEVEL ROBUSTNESS")
    log.info("=" * 72)

    # 3a. Count events per symbol
    symbol_counts = df.groupby("symbol").size().sort_values(ascending=False)
    top5 = symbol_counts.head(5).index.tolist()
    log.info("  Top-10 symbols by event count:")
    for sym, cnt in symbol_counts.head(10).items():
        log.info("    %-8s  %d events", sym, cnt)

    # 3b. Performance on high-frequency vs low-frequency symbols
    med_count = symbol_counts.median()
    hi_syms = symbol_counts[symbol_counts >= med_count].index.tolist()
    lo_syms = symbol_counts[symbol_counts < med_count].index.tolist()

    def oos_subset_metrics(idx_list, label):
        sub = df.loc[idx_list].dropna(subset=features + [PRIMARY_REG_TARGET])
        if len(sub) < 10:
            return None
        r = wf_ridge(sub, features, PRIMARY_REG_TARGET, label)
        return r

    hi_df = df[df["symbol"].isin(hi_syms)]
    lo_df = df[df["symbol"].isin(lo_syms)]
    log.info("  High-freq symbols (≥ median=%d events): %d symbols, %d rows",
             int(med_count), len(hi_syms), len(hi_df))
    log.info("  Low-freq symbols  (< median=%d events): %d symbols, %d rows",
             int(med_count), len(lo_syms), len(lo_df))

    r_hi = wf_ridge(hi_df, features, PRIMARY_REG_TARGET, "High-freq symbols")
    r_lo = wf_ridge(lo_df, features, PRIMARY_REG_TARGET, "Low-freq symbols")
    log.info("")
    log.info("  %-32s  %10s  %10s", "Subset", "Spearman_r", "R²(TRC)")
    log.info("  " + "-" * 54)
    for r in [r_hi, r_lo]:
        p = r["pooled"]
        log.info("  %-32s  %10.4f  %10.4f",
                 r["label"],
                 p.get("spearman_r", float("nan")),
                 p.get("r2", float("nan")))

    # 3c. Leave-top5-symbols-out
    df_no_top5 = df[~df["symbol"].isin(top5)]
    r_no_top5  = wf_ridge(df_no_top5, features, PRIMARY_REG_TARGET, "Without top-5 symbols")
    r_full     = wf_ridge(df, features, PRIMARY_REG_TARGET, "Full dataset")
    log.info("")
    log.info("── Leave-top-5-symbols-out sensitivity ──────────────────────────")
    log.info("  Full dataset:            Spearman=%.4f  R²=%.4f  N=%d",
             r_full["pooled"].get("spearman_r", float("nan")),
             r_full["pooled"].get("r2", float("nan")),
             r_full["pooled"].get("n", 0))
    log.info("  Without top-5 symbols:  Spearman=%.4f  R²=%.4f  N=%d",
             r_no_top5["pooled"].get("spearman_r", float("nan")),
             r_no_top5["pooled"].get("r2", float("nan")),
             r_no_top5["pooled"].get("n", 0))
    delta = (r_no_top5["pooled"].get("spearman_r", 0) -
             r_full["pooled"].get("spearman_r", 0))
    log.info("  Δ Spearman = %+.4f", delta)
    if abs(delta) > 0.05:
        log.info("  ⚠  Model is sensitive to top-5 symbols — results may be driven by a few names")
    else:
        log.info("  ✓  Removing top-5 symbols has minimal impact")

    # 3d. Per-symbol OOS error (identify worst symbols)
    r_for_error = wf_ridge(df, features, PRIMARY_REG_TARGET, "for_error_analysis")
    if r_for_error["oos"]:
        oos_df = pd.DataFrame(r_for_error["oos"]).set_index("idx")
        merged = df[["symbol", "year", "term_ratio_change",
                     "front_iv_crush_pct"]].join(oos_df, how="inner")
        merged["abs_err"] = (merged["y_true"] - merged["y_pred"]).abs()
        sym_err = merged.groupby("symbol").agg(
            n=("abs_err", "count"),
            mae=("abs_err", "mean"),
            spearman=("y_true", lambda x: spearmanr(
                x, merged.loc[x.index, "y_pred"])[0] if len(x) >= 3 else float("nan")),
        ).sort_values("mae", ascending=False)
        worst5 = sym_err.head(5)
        best5  = sym_err.sort_values("mae").head(5)
        log.info("")
        log.info("── Per-symbol OOS error (top 5 worst) ───────────────────────────")
        for sym, row in worst5.iterrows():
            log.info("  %-8s  N=%3d  MAE=%.4f  Spearman=%.3f",
                     sym, int(row["n"]), row["mae"], row["spearman"])
        log.info("── Per-symbol OOS error (top 5 best) ────────────────────────────")
        for sym, row in best5.iterrows():
            log.info("  %-8s  N=%3d  MAE=%.4f  Spearman=%.3f",
                     sym, int(row["n"]), row["mae"], row["spearman"])
    else:
        sym_err = pd.DataFrame()
        worst5 = pd.DataFrame()
        best5  = pd.DataFrame()

    return {
        "symbol_count_top10": symbol_counts.head(10).to_dict(),
        "top5_symbols": top5,
        "high_freq_pooled": r_hi["pooled"],
        "low_freq_pooled": r_lo["pooled"],
        "full_pooled": r_full["pooled"],
        "without_top5_pooled": r_no_top5["pooled"],
        "delta_spearman_top5_removed": round(delta, 4),
        "top5_sensitive": bool(abs(delta) > 0.05),
        "worst5_symbols": worst5.reset_index().to_dict("records") if not worst5.empty else [],
    }


# ── STEP 4: regime stability ──────────────────────────────────────────────────

def step4_regime_stability(df: pd.DataFrame,
                           features: List[str]) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 4 — REGIME STABILITY")
    log.info("=" * 72)
    log.info("  Each year is tested ONLY as the OOS fold, trained on preceding years.")

    per_year: List[Dict] = []
    for tr_yrs, te_yr in WALK_FORWARD_FOLDS:
        fd = _fold_data(df, tr_yrs, te_yr, features, PRIMARY_REG_TARGET)
        if fd is None:
            log.info("  Year %d: insufficient data", te_yr)
            continue
        X_tr, y_tr, X_te, y_te, idx = fd
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)

        sp_r, _ = spearmanr(y_te, y_hat)
        r2      = float(r2_score(y_te, y_hat))

        # OOS economy
        te_df = df.loc[idx].copy()
        te_df["y_pred"] = y_hat
        oos_for_q = [{"idx": i, "y_pred": yp, "y_true": yt}
                     for i, yp, yt in zip(idx, y_hat.tolist(), y_te.tolist())]
        spread, qtable = quintile_spread(te_df.assign(**{
            "front_iv_crush_pct": te_df["front_iv_crush_pct"],
            "back_iv_crush_pct": te_df["back_iv_crush_pct"],
            "spread_differential": te_df["spread_differential"],
            "year": te_df["year"],
        }), oos_for_q)

        yr_stats = df[df["year"] == te_yr]
        log.info("")
        log.info("  Year %d  (train=%s → test=%d):", te_yr, tr_yrs, te_yr)
        log.info("    N_test=%d  VIX_mean=%.1f  crush_rate_lt40=%.1f%%",
                 len(X_te),
                 yr_stats["vix_close"].mean() if "vix_close" in yr_stats.columns else float("nan"),
                 yr_stats["crush_lt_40pct"].mean() * 100)
        log.info("    Spearman=%.4f  R²=%.4f  Q5-Q1_spread=%.1f pp",
                 sp_r, r2, spread if not math.isnan(spread) else -999)

        per_year.append({
            "test_year": te_yr,
            "train_years": tr_yrs,
            "n_test": len(X_te),
            "spearman_r": round(float(sp_r), 4),
            "r2": round(r2, 4),
            "q5_q1_spread_pp": round(spread, 1) if not math.isnan(spread) else None,
        })

    # Stability check
    sp_vals = [r["spearman_r"] for r in per_year]
    if len(sp_vals) >= 2:
        sp_range = max(sp_vals) - min(sp_vals)
        sp_min   = min(sp_vals)
        log.info("")
        log.info("── Stability summary ─────────────────────────────────────────────")
        log.info("  Spearman range across years: %.4f  (min=%.4f  max=%.4f)",
                 sp_range, sp_min, max(sp_vals))
        if sp_range > 0.20:
            log.info("  ⚠  HIGH INSTABILITY: >0.20 Spearman range across OOS years.")
            log.info("     Model performance is regime-dependent.")
        elif sp_range > 0.10:
            log.info("  ⡠  MODERATE instability: 0.10–0.20 Spearman range.")
        else:
            log.info("  ✓  Stable across OOS years (range < 0.10).")

        if sp_min < 0.40:
            log.info("  ⚠  At least one year with Spearman < 0.40 — model can fail in specific regimes.")

    return {"per_year": per_year}


# ── STEP 5: economic realism ──────────────────────────────────────────────────

def step5_economic_realism(df: pd.DataFrame,
                           features: List[str]) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 5 — ECONOMIC REALISM CHECK")
    log.info("=" * 72)
    log.info("  Calendar spread P&L proxy per quintile.")
    log.info("  Buy back/sell front (calendar): profits from front IV crushing MORE than back.")
    log.info("  Payoff ≈ (front_crush - back_crush) = spread_differential")
    log.info("  Higher differential = better calendar spread outcome.")

    r = wf_ridge(df, features, PRIMARY_REG_TARGET, "econ_check")
    oos = r["oos"]
    if not oos:
        log.info("  No OOS predictions available.")
        return {}

    pred_df = pd.DataFrame(oos).set_index("idx")
    merged = df[["symbol", "year", "front_iv_crush_pct", "back_iv_crush_pct",
                 "spread_differential", "term_ratio_change",
                 "underlying_move_pct"]].join(pred_df, how="inner")

    # Q5 = predicted deepest crush (most negative TRC)
    merged["score"] = -merged["y_pred"]
    try:
        merged["q"] = pd.qcut(merged["score"], 5,
                              labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
    except ValueError:
        log.info("  Could not compute quintiles (insufficient data).")
        return {}

    agg = merged.groupby("q", observed=True).agg(
        n=("front_iv_crush_pct", "count"),
        avg_front_pct=("front_iv_crush_pct",   lambda x: round(x.mean() * 100, 1)),
        avg_back_pct=("back_iv_crush_pct",      lambda x: round(x.mean() * 100, 1)),
        avg_diff_pct=("spread_differential",    lambda x: round(x.mean() * 100, 1)),
        avg_trc=("term_ratio_change",           lambda x: round(x.mean(), 4)),
        avg_underlying=("underlying_move_pct",  lambda x: round(x.mean() * 100, 1)),
        pct_profitable=("spread_differential",  lambda x: round((x < 0).mean() * 100, 1)),
    ).reset_index()

    log.info("")
    log.info("  %-4s  %5s  %12s  %11s  %13s  %8s  %11s  %12s",
             "Q", "N", "Front crush", "Back crush", "Cal differential",
             "TRC", "Undly move", "% profitable")
    log.info("  " + "-" * 82)
    for _, row in agg.iterrows():
        log.info("  %-4s  %5d  %+11.1f%%  %+10.1f%%  %+12.1f%%  %+7.4f  %+10.1f%%  %11.1f%%",
                 row["q"], row["n"],
                 row["avg_front_pct"], row["avg_back_pct"],
                 row["avg_diff_pct"], row["avg_trc"],
                 row["avg_underlying"], row["pct_profitable"])

    q1 = agg.loc[agg["q"] == "Q1", "avg_diff_pct"].values
    q5 = agg.loc[agg["q"] == "Q5", "avg_diff_pct"].values
    cal_spread = float(q5[0] - q1[0]) if len(q1) and len(q5) else float("nan")

    q1_front = agg.loc[agg["q"] == "Q1", "avg_front_pct"].values
    q5_front = agg.loc[agg["q"] == "Q5", "avg_front_pct"].values
    front_spread = float(q5_front[0] - q1_front[0]) if len(q1_front) and len(q5_front) else float("nan")

    log.info("")
    log.info("  Q5-Q1 front crush spread:      %+.1f pp", front_spread)
    log.info("  Q5-Q1 calendar differential:   %+.1f pp", cal_spread)

    # Check for adversarial cases: Q5 high predicted but bad real outcome
    q5_events = merged[merged["q"] == "Q5"].copy()
    q5_bad = q5_events[q5_events["spread_differential"] > 0]  # spread went wrong way
    pct_q5_bad = len(q5_bad) / len(q5_events) * 100 if len(q5_events) > 0 else float("nan")
    log.info("")
    log.info("  Q5 adversarial cases (high predicted crush, spread_diff > 0): %d / %d  (%.1f%%)",
             len(q5_bad), len(q5_events), pct_q5_bad)
    if pct_q5_bad > 25:
        log.info("  ⚠  >25%% of Q5 events have wrong-direction spread differential")
    elif pct_q5_bad > 15:
        log.info("  ⡠  10–25%% Q5 adversarial rate — manageable but monitor")
    else:
        log.info("  ✓  Q5 adversarial rate is low (<15%%)")

    return {
        "quintile_table": agg.to_dict("records"),
        "q5_q1_front_spread_pp": round(front_spread, 1),
        "q5_q1_cal_differential_pp": round(cal_spread, 1),
        "q5_adversarial_pct": round(pct_q5_bad, 1),
    }


# ── STEP 6: failure mode identification ──────────────────────────────────────

def step6_failure_modes(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 6 — FAILURE MODE IDENTIFICATION")
    log.info("=" * 72)

    r = wf_ridge(df, features, PRIMARY_REG_TARGET, "failure_modes")
    if not r["oos"]:
        log.info("  No OOS predictions available.")
        return {}

    pred_df = pd.DataFrame(r["oos"]).set_index("idx")
    merged = df[["symbol", "year", "term_ratio_change", "front_iv_crush_pct",
                 "back_iv_crush_pct", "spread_differential",
                 "underlying_move_pct", "vix_close",
                 "near_back_ratio"]].join(pred_df, how="inner")

    merged["residual"]  = merged["y_true"] - merged["y_pred"]
    merged["abs_resid"] = merged["residual"].abs()

    # 6a. Worst quintile of residuals
    resid_thresh = merged["abs_resid"].quantile(0.90)
    worst = merged[merged["abs_resid"] >= resid_thresh].copy()
    log.info("")
    log.info("── 6a. Largest 10%% residuals (%d events) ────────────────────────", len(worst))
    log.info("  Characteristics vs overall population:")

    def _compare(col, subset, full):
        s = subset[col].dropna()
        f = full[col].dropna()
        if len(s) == 0 or len(f) == 0:
            return
        log.info("  %-20s  Worst-10%%: %+8.4f   All: %+8.4f   Δ=%+.4f",
                 col, s.mean(), f.mean(), s.mean() - f.mean())

    for col in ["near_back_ratio", "underlying_move_pct",
                "front_iv_crush_pct", "vix_close"]:
        if col in merged.columns:
            _compare(col, worst, merged)

    # 6b. IV expansion events (model expected crush, got expansion)
    expansions = merged[merged["front_iv_crush_pct"] > 0].copy()
    log.info("")
    log.info("── 6b. IV Expansion events (front crush > 0): %d events ──────────",
             len(expansions))
    if len(expansions) > 0:
        log.info("  Model predicted TRC median: %.4f  (negative = predicts crush)",
                 expansions["y_pred"].median())
        log.info("  VIX mean at expansion events: %.1f  (vs overall %.1f)",
                 expansions["vix_close"].mean() if "vix_close" in expansions.columns else float("nan"),
                 merged["vix_close"].mean() if "vix_close" in merged.columns else float("nan"))
        log.info("  Underlying move mean: %+.1f%%  (vs overall %+.1f%%)",
                 expansions["underlying_move_pct"].mean() * 100,
                 merged["underlying_move_pct"].mean() * 100)
        if len(expansions) >= 5:
            log.info("  Top expansion events by magnitude:")
            top_exp = expansions.nlargest(5, "front_iv_crush_pct")[
                ["symbol", "year", "front_iv_crush_pct", "y_pred",
                 "underlying_move_pct", "vix_close"]
            ]
            for _, row in top_exp.iterrows():
                log.info("    %-6s  %d  front_crush=%+.1f%%  pred_TRC=%.4f  undly=%+.1f%%  VIX=%.1f",
                         row.get("symbol","?"), int(row.get("year", 0)),
                         row["front_iv_crush_pct"] * 100,
                         row["y_pred"],
                         row["underlying_move_pct"] * 100,
                         row["vix_close"] if not math.isnan(row["vix_close"]) else -1)

    # 6c. High-confidence wrong predictions (model very confident, very wrong)
    # "Confident" = predicted TRC in bottom quintile (large negative = deep crush)
    pred_q20 = merged["y_pred"].quantile(0.20)
    confident_crush = merged[merged["y_pred"] <= pred_q20]
    # Among those, "wrong" = actual TRC in top half (shallow or positive)
    confident_wrong = confident_crush[confident_crush["y_true"] > merged["y_true"].median()]
    pct_conf_wrong = len(confident_wrong) / len(confident_crush) * 100 if len(confident_crush) > 0 else float("nan")
    log.info("")
    log.info("── 6c. High-confidence-wrong predictions ─────────────────────────")
    log.info("  Model bottom-quintile predicted TRC (deepest crush predicted): %d events",
             len(confident_crush))
    log.info("  Of those, actual TRC > median (wrong direction): %d  (%.1f%%)",
             len(confident_wrong), pct_conf_wrong)
    if pct_conf_wrong > 30:
        log.info("  ⚠  >30%% of high-confidence predictions are directionally wrong.")
    elif pct_conf_wrong > 20:
        log.info("  ⡠  20–30%% confidence-wrong rate — non-trivial.")
    else:
        log.info("  ✓  High-confidence-wrong rate < 20%%")

    # 6d. Year-level residual bias
    log.info("")
    log.info("── 6d. Residual bias by year ─────────────────────────────────────")
    yr_bias = merged.groupby("year")[["residual", "y_true", "y_pred"]].mean()
    for yr, row in yr_bias.iterrows():
        log.info("  %d  mean_residual=%+.4f  mean_y_true=%+.4f  mean_y_pred=%+.4f",
                 yr, row["residual"], row["y_true"], row["y_pred"])

    return {
        "n_expansion_events": int(len(expansions)),
        "expansion_event_vix_mean": round(float(expansions["vix_close"].mean()), 1)
            if "vix_close" in expansions.columns and len(expansions) > 0 else None,
        "pct_confident_wrong": round(pct_conf_wrong, 1),
        "worst_10pct_n": len(worst),
        "yr_residual_bias": yr_bias["residual"].round(4).to_dict(),
    }


# ── STEP 7: VIX contribution validation ──────────────────────────────────────

def step7_vix_contribution(df: pd.DataFrame,
                           step1_results: Dict) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 7 — VIX CONTRIBUTION VALIDATION")
    log.info("=" * 72)

    has_vix = step1_results.get("has_vix", False)
    if not has_vix:
        log.info("  VIX data unavailable — skipping this step.")
        return {"skipped": True, "reason": "VIX parquet not available"}

    reg_results = step1_results["reg_results"]
    clf_results = step1_results["clf_results"]

    def _get(results, substr):
        return next((r for r in results if substr in r["label"]), {})

    r_nbr      = _get(reg_results, "NBR-only")
    r_base     = _get(reg_results, "Baseline")
    r_vix_only = _get(reg_results, "VIX-only")
    r_comb     = _get(reg_results, "Combined")

    c_base = _get(clf_results, "Baseline")
    c_comb = _get(clf_results, "Combined")

    log.info("")
    log.info("── 7a. Regression performance comparison ─────────────────────────")
    log.info("  %-30s  %10s  %10s  %10s",
             "Model", "Spearman_r", "R²(TRC)", "RMSE/naive")
    log.info("  " + "-" * 64)
    for r in [r_nbr, r_base, r_vix_only, r_comb]:
        if r:
            p = r.get("pooled", {})
            log.info("  %-30s  %10.4f  %10.4f  %10.4f",
                     r.get("label",""),
                     p.get("spearman_r", float("nan")),
                     p.get("r2", float("nan")),
                     p.get("rmse_ratio", float("nan")))

    b_sp = (r_base.get("pooled") or {}).get("spearman_r", 0)
    c_sp = (r_comb.get("pooled") or {}).get("spearman_r", 0)
    v_sp = (r_vix_only.get("pooled") or {}).get("spearman_r", 0)
    delta_sp = round(c_sp - b_sp, 4)

    log.info("")
    log.info("  Δ Spearman (Combined - Baseline): %+.4f", delta_sp)
    log.info("  VIX-only vs Baseline: %s",
             "WEAKER" if v_sp < b_sp else "STRONGER OR COMPARABLE")

    log.info("")
    log.info("── 7b. Per-fold consistency ──────────────────────────────────────")
    b_folds = r_base.get("folds", [])
    c_folds = r_comb.get("folds", [])
    log.info("  %-30s  %10s  %10s  %10s",
             "Model", "Fold1(2024)", "Fold2(2025)", "Fold3(2026)")
    for r in [r_base, r_comb]:
        folds = r.get("folds", [])
        sp_vals = [f.get("spearman_r", float("nan")) for f in folds]
        while len(sp_vals) < 3:
            sp_vals.append(float("nan"))
        log.info("  %-30s  %10.4f  %10.4f  %10.4f",
                 r.get("label", ""), *sp_vals[:3])

    fold_deltas = [c.get("spearman_r", 0) - b.get("spearman_r", 0)
                   for c, b in zip(c_folds, b_folds)]
    consistent = all(d > 0 for d in fold_deltas) if fold_deltas else False
    log.info("  Fold deltas (C - B): %s  → consistent gain: %s",
             [f"{d:+.4f}" for d in fold_deltas], "YES" if consistent else "NO")

    # 7c. VIX-enhanced economic spread
    log.info("")
    log.info("── 7c. VIX economic contribution ────────────────────────────────")
    for r in [r_base, r_vix_only, r_comb]:
        if r and r.get("oos"):
            spread, _ = quintile_spread(df, r["oos"])
            log.info("  %-30s  Q5-Q1 spread = %+.1f pp",
                     r.get("label", ""), spread)

    vix_verdict = (
        "ADDITIVE"   if delta_sp >= 0.015 and consistent else
        "MARGINAL"   if delta_sp >= 0.005 else
        "REDUNDANT"
    )
    log.info("")
    log.info("  VIX contribution verdict: %s", vix_verdict)

    return {
        "baseline_spearman": round(b_sp, 4),
        "combined_spearman": round(c_sp, 4),
        "vix_only_spearman": round(v_sp, 4),
        "delta_spearman": delta_sp,
        "consistent_fold_gain": consistent,
        "fold_deltas": [round(d, 4) for d in fold_deltas],
        "vix_verdict": vix_verdict,
    }


# ── STEP 8: verdict ───────────────────────────────────────────────────────────

def step8_verdict(
    step1: Dict, step2: Dict, step3: Dict, step4: Dict,
    step5: Dict, step6: Dict, step7: Dict,
) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 8 — ADVERSARIAL VERDICT")
    log.info("=" * 72)

    # Evidence summary
    perm_verdict   = step2.get("verdict", "UNKNOWN")
    z_sp           = step2.get("z_spearman", 0) or 0
    top5_sensitive = step3.get("top5_sensitive", False)
    worst_yr_sp    = min((r["spearman_r"] for r in step4.get("per_year", [{}])), default=float("nan"))
    sp_range       = (max((r["spearman_r"] for r in step4.get("per_year", [{}])), default=0) -
                      min((r["spearman_r"] for r in step4.get("per_year", [{}])), default=0))
    pct_conf_wrong  = step6.get("pct_confident_wrong", 100)
    n_expansion    = step6.get("n_expansion_events", 0)
    q5_adv_pct     = step5.get("q5_adversarial_pct", 100)
    vix_verdict    = step7.get("vix_verdict", "N/A")
    nbr_pct        = step1.get("nbr_pct_of_baseline", 100)
    mech_link_flag = step1["algebraic"]["mechanical_link_flag"]
    r2_nbr_on_trc  = step1["algebraic"]["r2_nbr_on_trc"]

    # Regression target performance
    base_sp  = (step1.get("ablation", {}).get("Baseline (3 feat)", {}) or {}).get("spearman_r", 0)
    base_r2  = (step1.get("ablation", {}).get("Baseline (3 feat)", {}) or {}).get("r2", 0)

    log.info("")
    log.info("  SIGNAL PROVENANCE BREAKDOWN:")
    log.info("  ─────────────────────────────────────────────────────────────────")
    log.info("  near_back_ratio alone:   Spearman ~ %.4f  (%.0f%% of baseline signal)",
             step1.get("ablation",{}).get("NBR-only (1 feat)",{}).get("spearman_r", float("nan")),
             nbr_pct)
    log.info("  Full baseline (3 feat):  Spearman ~ %.4f  R² ~ %.4f", base_sp, base_r2)
    log.info("  VIX contribution:        %s", vix_verdict)
    log.info("  Permutation test:        %s  (z=%.2f)", perm_verdict, z_sp)
    log.info("  Regime stability range:  Spearman range = %.4f  (worst year = %.4f)",
             sp_range, worst_yr_sp)
    log.info("  Symbol concentration:    top-5 sensitive = %s", top5_sensitive)
    log.info("  Confidence-wrong rate:   %.1f%%", pct_conf_wrong)
    log.info("  IV expansion events:     %d", n_expansion)
    log.info("  Q5 adversarial rate:     %.1f%%", q5_adv_pct)
    log.info("  Mechanical link R²:      %.4f  (NBR → TRC algebraic fraction)", r2_nbr_on_trc)

    # Flags
    flags = []
    if mech_link_flag:
        flags.append("MECHANICAL_LINK: near_back_ratio is pre_TRC — algebraically related to target. "
                     "Part of R² may be accounting identity, not prediction.")
    if nbr_pct > 90:
        flags.append("SINGLE_FEATURE_DOMINANCE: NBR-only captures >90% of baseline signal. "
                     "log_front_iv + log_back_iv add minimal independent value.")
    if perm_verdict == "FAILED":
        flags.append("PERMUTATION_FAILED: performance does not significantly exceed null. "
                     "Investigate for leakage or structural bias.")
    elif perm_verdict == "MARGINAL":
        flags.append("PERMUTATION_MARGINAL: only one metric passes 0.05 threshold.")
    if top5_sensitive:
        flags.append("SYMBOL_CONCENTRATION: removing top-5 symbols degrades Spearman by >5pp. "
                     "Model may rely on a handful of highly-predictable names.")
    if sp_range > 0.20:
        flags.append(f"REGIME_FRAGILITY: Spearman range {sp_range:.3f} across OOS years. "
                     "Model performance varies materially by macro environment.")
    if not math.isnan(worst_yr_sp) and worst_yr_sp < 0.40:
        flags.append(f"WORST_YEAR_WEAK: minimum OOS Spearman = {worst_yr_sp:.4f}. "
                     "Model can fail in specific years.")
    if pct_conf_wrong > 30:
        flags.append(f"HIGH_CONFIDENCE_WRONG: {pct_conf_wrong:.1f}% of top-quartile predictions "
                     "are directionally wrong.")
    if q5_adv_pct > 25:
        flags.append(f"Q5_ADVERSARIAL: {q5_adv_pct:.1f}% of Q5 events have adverse spread direction. "
                     "Economic edge may not survive transaction costs.")

    log.info("")
    log.info("  FRAGILITY FLAGS:")
    if flags:
        for i, f in enumerate(flags, 1):
            log.info("  [%d] %s", i, f)
    else:
        log.info("  None — signal passed all adversarial checks.")

    # Final verdict
    critical_failure = perm_verdict == "FAILED"
    major_concerns   = len([f for f in flags if any(kw in f for kw in
                            ["MECHANICAL", "PERMUTATION", "REGIME_FRAGILITY",
                             "HIGH_CONFIDENCE", "SYMBOL_CONC"])]) >= 2
    minor_concerns   = len(flags) > 0 and not critical_failure

    if critical_failure:
        overall = "FAIL — Signal does not survive permutation test. Do not trade."
    elif major_concerns:
        overall = "CONDITIONAL — Signal is real but fragile. Requires additional validation."
    elif minor_concerns:
        overall = "PASS WITH CAVEATS — Signal appears real. Address flagged weaknesses before trading."
    else:
        overall = "PASS — Signal is real, robust, and economically significant."

    log.info("")
    log.info("  OVERALL VERDICT: %s", overall)
    log.info("")
    log.info("  TRADING READINESS ASSESSMENT:")
    log.info("  ─────────────────────────────────────────────────────────────────")
    if perm_verdict in ("REAL", "MARGINAL"):
        log.info("  ✓ Statistical signal confirmed")
    else:
        log.info("  ✗ Statistical signal NOT confirmed")
    if sp_range <= 0.15 and (math.isnan(worst_yr_sp) or worst_yr_sp >= 0.45):
        log.info("  ✓ Regime stability adequate")
    else:
        log.info("  ⚠ Regime stability concerns — model can degrade in specific environments")
    if q5_adv_pct < 20:
        log.info("  ✓ Economic edge (Q5) is reliable")
    else:
        log.info("  ⚠ Q5 economic edge has %.1f%% adverse rate", q5_adv_pct)
    if not mech_link_flag:
        log.info("  ✓ No mechanical linkage between feature and target")
    else:
        log.info("  ⚠ Partial mechanical linkage — quantify genuine vs algebraic R²")
    log.info("  → Recommended next step: prior_crush_history per symbol (Week 4)")
    log.info("    (expected to reduce symbol-level noise without adding data leakage)")

    return {
        "overall_verdict": overall,
        "flags": flags,
        "critical_failure": bool(critical_failure),
        "signal_real": bool(perm_verdict in ("REAL", "MARGINAL")),
        "nbr_pct_of_baseline": nbr_pct,
        "mechanical_link_flag": mech_link_flag,
        "r2_nbr_on_trc": round(r2_nbr_on_trc, 4),
        "regime_sp_range": round(sp_range, 4),
        "worst_year_spearman": round(worst_yr_sp, 4) if not math.isnan(worst_yr_sp) else None,
        "pct_confident_wrong": pct_conf_wrong,
        "q5_adversarial_pct": q5_adv_pct,
        "permutation_verdict": perm_verdict,
        "vix_verdict": vix_verdict,
        "n_flags": len(flags),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adversarial stress test for IV crush model")
    p.add_argument("--db-path",        default=None)
    p.add_argument("--quality-tier",   default="AB", choices=["A","AB"])
    p.add_argument("--permutations",   type=int, default=N_PERMUTATIONS_DEFAULT)
    p.add_argument("--report-dir",     default=None)
    p.add_argument("--skip-permutation", action="store_true",
                   help="Skip permutation test (fast mode for debugging)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    db_path    = Path(args.db_path).expanduser() if args.db_path else DEFAULT_DB_PATH
    report_dir = Path(args.report_dir).expanduser() if args.report_dir else DEFAULT_REPORT_DIR
    quality_tiers = list(args.quality_tier)
    n_perm     = 0 if args.skip_permutation else args.permutations

    log.info("=" * 72)
    log.info("ADVERSARIAL STRESS TEST  version=%s", RESEARCH_VERSION)
    log.info("=" * 72)
    log.info("DB         : %s", db_path)
    log.info("Quality    : tiers %s", quality_tiers)
    log.info("Permutations: %d", n_perm)
    log.info("Mindset    : Assume the model is WRONG until proven otherwise.")
    log.info("")

    # ── load + engineer ───────────────────────────────────────────────────────
    log.info("── Loading data ─────────────────────────────────────────────────")
    labels = load_labels(db_path, quality_tiers)
    df     = engineer_features(labels)
    df     = try_load_vix(df)
    has_vix = df["log_vix"].notna().mean() > 0.3
    # Use combined features if VIX available, else baseline
    primary_features = FS_COMBINED if has_vix else FS_BASELINE
    log.info("Primary feature set: %s (%d features)",
             "combined" if has_vix else "baseline", len(primary_features))
    log.info("")

    # ── run all steps ─────────────────────────────────────────────────────────
    s1 = step1_feature_dominance(df)
    s2 = (step2_permutation(df, n_perm, FS_BASELINE) if n_perm > 0
          else {"verdict": "SKIPPED", "skipped": True})
    s3 = step3_symbol_robustness(df, FS_BASELINE)
    s4 = step4_regime_stability(df, FS_BASELINE)
    s5 = step5_economic_realism(df, FS_BASELINE)
    s6 = step6_failure_modes(df, FS_BASELINE)
    s7 = step7_vix_contribution(df, s1)
    s8 = step8_verdict(s1, s2, s3, s4, s5, s6, s7)

    # ── save report ───────────────────────────────────────────────────────────
    def _safe(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    report = {
        "research_version": RESEARCH_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "db_path": str(db_path), "quality_tiers": quality_tiers,
            "n_permutations": n_perm,
            "primary_features": primary_features,
        },
        "step1_feature_dominance": {k: v for k, v in s1.items()
                                    if k not in ("reg_results", "clf_results")},
        "step2_permutation": s2,
        "step3_symbol_robustness": s3,
        "step4_regime_stability": s4,
        "step5_economic_realism": s5,
        "step6_failure_modes": s6,
        "step7_vix_contribution": s7,
        "step8_verdict": s8,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"adversarial_{RESEARCH_VERSION}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, default=_safe))
    log.info("")
    log.info("Adversarial report saved to: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
