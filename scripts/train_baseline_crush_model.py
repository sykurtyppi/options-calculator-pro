#!/usr/bin/env python3
"""
scripts/train_baseline_crush_model.py
=======================================
Week 2 Baseline — Earnings IV Crush Signal Research

Structured research pipeline:
  Step 1  Dataset audit (mandatory before any modeling)
  Step 2  Target definition analysis — correct framing of the problem
  Step 3  Feature engineering — 3-feature baseline set
  Step 4  Walk-forward LR + Ridge baselines (time-based, no leakage)
  Step 5  Statistical evaluation (AUC, RMSE, calibration)
  Step 6  Economic evaluation (quintile P&L proxy)
  Step 7  Baseline conclusion + Week 3 recommendation

Output
------
  - Console: full audit + result tables
  - ~/.options_calculator_pro/reports/  JSON research report

Usage
-----
  python scripts/train_baseline_crush_model.py [options]
  python scripts/train_baseline_crush_model.py --quality-tier A
  python scripts/train_baseline_crush_model.py --db-path /path/to/db

Engineering standards
---------------------
  NO random train/test splits.
  Scaler fit exclusively on the training fold — never seen test data.
  Fixed regularization — no hyperparameter search on test data.
  All assumptions explicitly logged.
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

# ── project root ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
try:
    from utils.logger import setup_logger
    log = setup_logger(__name__)
except Exception:
    logging.basicConfig(format="%(asctime)s | %(levelname)-8s | %(message)s",
                        level=logging.DEBUG)
    log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
DEFAULT_REPORT_DIR = Path.home() / ".options_calculator_pro" / "reports"
RESEARCH_VERSION = "baseline_v1"

# Baseline feature set (3 features matching edge_engine.py signature)
# iv_rv_approx: requires live realized-vol data not in labels table.
# We substitute log_back_iv (log of pre-event back IV) as a proxy for "baseline
# vol level ≈ long-run RV."  Document this deviation explicitly.
FEATURES: List[str] = [
    "near_back_ratio",   # pre_front_iv / pre_back_iv  — earnings premium multiplier
    "log_front_iv",      # log(pre_front_iv)           — absolute front IV level
    "log_back_iv",       # log(pre_back_iv)            — iv_rv proxy (back IV ≈ RV)
]
FEATURE_DESCRIPTIONS = {
    "near_back_ratio": "pre_front_iv / pre_back_iv (earnings term-structure steepness)",
    "log_front_iv":    "log(pre_front_iv) — absolute level of pre-earnings front vol",
    "log_back_iv":     "log(pre_back_iv)  — back vol proxy for iv_rv_approx",
}

# Walk-forward fold definitions (train years → test year)
WALK_FORWARD_FOLDS: List[Tuple[List[int], int]] = [
    ([2023],       2024),
    ([2023, 2024], 2025),
    ([2023, 2024, 2025], 2026),
]

# Binary classification threshold candidates
BINARY_THRESHOLDS = [-0.10, -0.20, -0.30, -0.40, -0.50]

# Primary binary target (closest to 50/50 split)
PRIMARY_BINARY_THRESHOLD = -0.40   # 45.2% positive rate (validated in audit)

# Logistic regression config (fixed — no search)
LR_C = 1.0           # inverse regularization strength (L2)
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 500

# Ridge regression config (fixed)
RIDGE_ALPHA = 1.0

# ── data loading ──────────────────────────────────────────────────────────────

def load_labels(db_path: Path, quality_tiers: List[str]) -> pd.DataFrame:
    """Load earnings_iv_decay_labels from SQLite, filter by quality tier."""
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    placeholders = ",".join("?" * len(quality_tiers))
    query = f"""
        SELECT
            symbol,
            event_date,
            release_timing,
            pre_front_iv,
            post_front_iv,
            pre_back_iv,
            post_back_iv,
            front_iv_crush_pct,
            back_iv_crush_pct,
            term_ratio_change,
            underlying_move_pct,
            front_dte_pre,
            back_dte_pre,
            exact_expiry_match,
            quality_score,
            quality_tier,
            pre_front_atm_moneyness,
            pre_back_atm_moneyness,
            pre_front_oi,
            pre_back_oi,
            pre_underlying_price,
            post_underlying_price
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ({placeholders})
          AND pre_front_iv > 0
          AND pre_back_iv > 0
          AND front_iv_crush_pct IS NOT NULL
          AND back_iv_crush_pct IS NOT NULL
        ORDER BY event_date
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=quality_tiers)

    df["event_date"] = pd.to_datetime(df["event_date"])
    df["year"] = df["event_date"].dt.year
    df["quarter"] = df["event_date"].dt.quarter
    log.info("Loaded %d rows from %s (quality_tiers=%s)", len(df), db_path, quality_tiers)
    return df


# ── feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the 3 baseline features from raw label columns.

    All features are computed exclusively from PRE-EVENT data to prevent leakage.
    """
    df = df.copy()
    df["near_back_ratio"] = df["pre_front_iv"] / df["pre_back_iv"]
    df["log_front_iv"] = np.log(df["pre_front_iv"])
    df["log_back_iv"] = np.log(df["pre_back_iv"])
    # Auxiliary: used in economic evaluation but not in baseline model
    df["spread_differential"] = df["front_iv_crush_pct"] - df["back_iv_crush_pct"]
    return df


# ── dataset audit ─────────────────────────────────────────────────────────────

def _decile_summary(series: pd.Series) -> Dict[str, float]:
    pcts = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
    return {f"p{p}": round(float(np.percentile(series.dropna(), p)), 4) for p in pcts}


def _text_histogram(series: pd.Series, bins: int = 12, width: int = 40) -> List[str]:
    """Return ASCII histogram lines for a numeric series."""
    counts, edges = np.histogram(series.dropna(), bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    lines = []
    for i, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)):
        bar = "█" * int(c / max_count * width)
        lines.append(f"  [{lo:+.3f},{hi:+.3f})  {bar:<{width}} {c:4d}")
    return lines


def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Full dataset audit. Must run before any modeling."""
    n = len(df)
    audit: Dict[str, Any] = {
        "total_rows": n,
        "symbols": int(df["symbol"].nunique()),
        "years": sorted(df["year"].unique().tolist()),
        "quality_tiers": df["quality_tier"].value_counts().to_dict(),
    }

    # --- crush distributions ---
    front = df["front_iv_crush_pct"] * 100   # convert to %
    back  = df["back_iv_crush_pct"] * 100
    audit["front_crush_pct"] = {
        "deciles": _decile_summary(front),
        "mean": round(float(front.mean()), 2),
        "std":  round(float(front.std()),  2),
    }
    audit["back_crush_pct"] = {
        "deciles": _decile_summary(back),
        "mean": round(float(back.mean()), 2),
        "std":  round(float(back.std()),  2),
    }

    # --- class balance at candidate thresholds ---
    balance: Dict[str, Any] = {}
    for t in BINARY_THRESHOLDS:
        pos = (df["front_iv_crush_pct"] < t).sum()
        balance[f"lt_{abs(int(t*100))}pct"] = {
            "n_positive": int(pos),
            "n_negative": int(n - pos),
            "positive_rate": round(pos / n, 4),
        }
    audit["class_balance"] = balance

    # --- per-year breakdown ---
    year_stats: Dict[str, Any] = {}
    for yr, grp in df.groupby("year"):
        pos40 = (grp["front_iv_crush_pct"] < PRIMARY_BINARY_THRESHOLD).sum()
        year_stats[str(yr)] = {
            "n": int(len(grp)),
            "mean_front_crush": round(float(grp["front_iv_crush_pct"].mean() * 100), 1),
            "positive_rate_lt40pct": round(float(pos40 / len(grp)), 3),
        }
    audit["by_year"] = year_stats

    # --- IV expansion events (crush is positive) ---
    iv_exp = df[df["front_iv_crush_pct"] > 0]
    audit["iv_expansion_events"] = {
        "count": int(len(iv_exp)),
        "pct_of_total": round(len(iv_exp) / n, 4),
        "examples": iv_exp[["symbol", "event_date", "front_iv_crush_pct"]]\
                        .head(5).assign(
                            event_date=lambda x: x["event_date"].dt.strftime("%Y-%m-%d"),
                            front_iv_crush_pct=lambda x: (x["front_iv_crush_pct"]*100).round(1)
                        ).to_dict("records"),
    }

    # --- symbols with very few observations ---
    sym_counts = df.groupby("symbol").size().sort_values()
    audit["thin_symbols"] = {
        "lt_5_obs": int((sym_counts < 5).sum()),
        "lt_3_obs": int((sym_counts < 3).sum()),
        "bottom_10": sym_counts.head(10).to_dict(),
    }

    # --- low-quality rows ---
    tier_b = df[df["quality_tier"] == "B"]
    audit["tier_b"] = {
        "count": int(len(tier_b)),
        "mean_front_crush": round(float(tier_b["front_iv_crush_pct"].mean() * 100), 2) if len(tier_b) else None,
    }

    # --- feature summary ---
    for col in FEATURES:
        if col in df.columns:
            audit[f"feature_{col}"] = {
                "mean": round(float(df[col].mean()), 4),
                "std":  round(float(df[col].std()),  4),
                "min":  round(float(df[col].min()),  4),
                "max":  round(float(df[col].max()),  4),
            }

    # --- raw near_back_ratio quintile crush (pre-modeling signal check) ---
    df2 = df.copy()
    df2["nbr_quintile"] = pd.qcut(df2["near_back_ratio"], 5, labels=["Q1","Q2","Q3","Q4","Q5"])
    raw_signal = df2.groupby("nbr_quintile", observed=True)["front_iv_crush_pct"].agg(
        ["mean", "std", "count"]
    ).round(4)
    raw_signal["mean_pct"] = (raw_signal["mean"] * 100).round(1)
    audit["raw_signal_nbr_quintile"] = raw_signal[["mean_pct","std","count"]].to_dict()

    return audit


def print_audit(audit: Dict[str, Any]) -> None:
    log.info("=" * 72)
    log.info("STEP 1 — DATASET AUDIT")
    log.info("=" * 72)
    log.info("Dataset: %d rows | %d symbols | years %s",
             audit["total_rows"], audit["symbols"], audit["years"])
    log.info("Quality tiers: %s", audit["quality_tiers"])

    log.info("")
    log.info("── Front IV Crush Distribution ──────────────────────────────")
    dc = audit["front_crush_pct"]
    d  = dc["deciles"]
    log.info("  mean=%.1f%%  std=%.1f%%", dc["mean"], dc["std"])
    log.info("  p10=%.1f%%  p25=%.1f%%  p50=%.1f%%  p75=%.1f%%  p90=%.1f%%",
             d["p10"], d["p25"], d["p50"], d["p75"], d["p90"])
    log.info("  range: [%.1f%%, %.1f%%]", d["p0"], d["p100"])

    log.info("")
    log.info("── Back IV Crush Distribution ───────────────────────────────")
    dc2 = audit["back_crush_pct"]
    d2  = dc2["deciles"]
    log.info("  mean=%.1f%%  std=%.1f%%", dc2["mean"], dc2["std"])
    log.info("  p10=%.1f%%  p25=%.1f%%  p50=%.1f%%  p75=%.1f%%  p90=%.1f%%",
             d2["p10"], d2["p25"], d2["p50"], d2["p75"], d2["p90"])

    log.info("")
    log.info("── Class Balance at Candidate Thresholds ────────────────────")
    log.info("  %-16s  %7s  %7s  %12s", "Threshold", "N_pos", "N_neg", "Positive%")
    log.info("  " + "-" * 50)
    for k, v in audit["class_balance"].items():
        thresh = k.replace("lt_","<-").replace("pct","%")
        marker = " ← PRIMARY TARGET" if "40" in k else ""
        log.info("  %-16s  %7d  %7d  %11.1f%%%s",
                 thresh, v["n_positive"], v["n_negative"], v["positive_rate"]*100, marker)

    log.info("")
    log.info("── IV Expansion Events (crush > 0) ──────────────────────────")
    ie = audit["iv_expansion_events"]
    log.info("  %d events (%.1f%% of dataset)", ie["count"], ie["pct_of_total"]*100)
    for ex in ie["examples"]:
        log.info("    %s  %s  %+.1f%%", ex["symbol"], ex["event_date"], ex["front_iv_crush_pct"])

    log.info("")
    log.info("── Per-Year Breakdown ───────────────────────────────────────")
    log.info("  %-6s  %5s  %14s  %20s", "Year", "N", "Mean crush", "Pos-rate(<-40%)")
    for yr, v in audit["by_year"].items():
        log.info("  %-6s  %5d  %+13.1f%%  %19.1f%%",
                 yr, v["n"], v["mean_front_crush"], v["positive_rate_lt40pct"]*100)

    log.info("")
    log.info("── Raw Signal Check — near_back_ratio Quintiles ─────────────")
    log.info("  (Non-temporal; for intuition only. Walk-forward results below.)")
    log.info("  %-4s  %12s  %7s", "Q", "Mean crush", "N")
    rq = audit["raw_signal_nbr_quintile"]
    for q in ["Q1","Q2","Q3","Q4","Q5"]:
        log.info("  %-4s  %+11.1f%%  %7d", q, rq["mean_pct"][q], int(rq["count"][q]))

    log.info("")
    log.info("── Feature Summary ──────────────────────────────────────────")
    for col in FEATURES:
        k = f"feature_{col}"
        if k in audit:
            v = audit[k]
            log.info("  %-18s  mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                     col, v["mean"], v["std"], v["min"], v["max"])
        else:
            log.info("  %-18s  NOT YET ENGINEERED", col)

    log.info("")
    log.info("── Data Quality Notes ───────────────────────────────────────")
    log.info("  Thin symbols (<5 obs): %d", audit["thin_symbols"]["lt_5_obs"])
    log.info("  Very thin (<3 obs): %d", audit["thin_symbols"]["lt_3_obs"])
    log.info("  Tier B rows: %d", audit["tier_b"]["count"])
    log.info("  iv_rv_approx NOTE: realized vol not in labels table.")
    log.info("    Substituting log_back_iv (back IV ≈ long-run vol proxy).")
    log.info("    This should be replaced by pre_front_iv/30d_RV when available.")


# ── target definition analysis ────────────────────────────────────────────────

def analyze_targets(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate candidate target definitions before any model training."""
    targets: Dict[str, Any] = {}

    # Binary: crush < threshold (several thresholds)
    for t in BINARY_THRESHOLDS:
        col = f"crush_lt_{abs(int(t*100))}pct"
        df[col] = (df["front_iv_crush_pct"] < t).astype(int)
        rate = df[col].mean()
        targets[col] = {
            "type": "binary",
            "threshold": t,
            "positive_rate": round(float(rate), 4),
            "base_accuracy": round(max(rate, 1-rate), 4),  # majority-class baseline
            "economically_relevant": t <= -0.20,
        }

    # Regression: front crush magnitude
    front_mean = float(df["front_iv_crush_pct"].mean())
    front_std  = float(df["front_iv_crush_pct"].std())
    naive_rmse = float(df["front_iv_crush_pct"].std())  # predict-mean baseline
    targets["front_iv_crush_pct"] = {
        "type": "regression",
        "mean": round(front_mean * 100, 2),
        "std":  round(front_std * 100, 2),
        "naive_rmse_pct": round(naive_rmse * 100, 2),
        "economically_relevant": True,
    }

    # Regression: term_ratio_change (most economically relevant for cal spreads)
    trc_mean = float(df["term_ratio_change"].mean())
    trc_std  = float(df["term_ratio_change"].std())
    targets["term_ratio_change"] = {
        "type": "regression",
        "mean": round(trc_mean, 4),
        "std":  round(trc_std, 4),
        "naive_rmse": round(trc_std, 4),
        "note": "Change in front/back IV ratio post-earnings. Direct calendar-spread proxy.",
        "economically_relevant": True,
    }

    # Regression: spread differential (front_crush - back_crush)
    df["spread_diff"] = df["front_iv_crush_pct"] - df["back_iv_crush_pct"]
    sd_mean = float(df["spread_diff"].mean())
    sd_std  = float(df["spread_diff"].std())
    targets["spread_differential"] = {
        "type": "regression",
        "mean": round(sd_mean * 100, 2),
        "std":  round(sd_std * 100, 2),
        "note": "front_crush − back_crush. Positive EV when front crushes more than back.",
        "economically_relevant": True,
    }

    return targets


def print_target_analysis(targets: Dict[str, Any]) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 2 — TARGET DEFINITION ANALYSIS")
    log.info("=" * 72)
    log.info("")
    log.info("CRITICAL: 98%% of events show front IV crush > 10%%.")
    log.info("Binary 'any crush' target is trivial (98%% base rate).")
    log.info("We model CRUSH MAGNITUDE, not just binary direction.")
    log.info("")
    log.info("── Binary Target Candidates ─────────────────────────────────")
    log.info("  %-28s  %12s  %12s  %s",
             "Target", "Pos rate", "Base acc.", "Economic?")
    log.info("  " + "-" * 60)
    for k, v in targets.items():
        if v["type"] == "binary":
            marker = "YES" if v["economically_relevant"] else "no"
            log.info("  %-28s  %11.1f%%  %11.1f%%  %s",
                     k, v["positive_rate"]*100, v["base_accuracy"]*100, marker)
    log.info("  → PRIMARY BINARY TARGET: crush_lt_40pct (45.2%% rate, ~balanced)")

    log.info("")
    log.info("── Regression Target Candidates ─────────────────────────────")
    for k, v in targets.items():
        if v["type"] == "regression":
            if k == "front_iv_crush_pct":
                log.info("  front_iv_crush_pct: mean=%.1f%%  std=%.1f%%  naive_RMSE=%.1f%%",
                         v["mean"], v["std"], v["naive_rmse_pct"])
            elif k == "term_ratio_change":
                log.info("  term_ratio_change:  mean=%.4f  std=%.4f  naive_RMSE=%.4f",
                         v["mean"], v["std"], v["naive_rmse"])
                log.info("    (%s)", v["note"])
            elif k == "spread_differential":
                log.info("  spread_differential: mean=%.1f%%  std=%.1f%%",
                         v["mean"], v["std"])
                log.info("    (%s)", v["note"])

    log.info("")
    log.info("Selected targets for baseline evaluation:")
    log.info("  1. crush_lt_40pct       [binary, ~balanced, calendar EV proxy]")
    log.info("  2. front_iv_crush_pct   [regression, magnitude directly]")
    log.info("  3. term_ratio_change    [regression, pure calendar-spread proxy]")


# ── walk-forward helpers ──────────────────────────────────────────────────────

def _get_fold_data(
    df: pd.DataFrame,
    train_years: List[int],
    test_year: int,
    feature_cols: List[str],
    target_col: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                     StandardScaler, pd.Series]]:
    """Extract and scale one walk-forward fold. Scaler fit on train only."""
    train_df = df[df["year"].isin(train_years)].dropna(subset=feature_cols + [target_col])
    test_df  = df[df["year"] == test_year].dropna(subset=feature_cols + [target_col])
    if len(train_df) == 0 or len(test_df) == 0:
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit on train ONLY
    X_test  = scaler.transform(X_test)         # apply to test

    return X_train, y_train, X_test, y_test, scaler, test_df.index


def walk_forward_classification(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Time-based walk-forward logistic regression.

    Returns per-fold metrics and pooled OOS predictions.
    """
    results: Dict[str, Any] = {
        "model": "LogisticRegression",
        "target": target_col,
        "features": feature_cols,
        "C": LR_C,
        "folds": [],
        "oos_predictions": [],   # list of (index, y_true, y_prob) for economic eval
    }

    for train_years, test_year in WALK_FORWARD_FOLDS:
        fold_data = _get_fold_data(df, train_years, test_year, feature_cols, target_col)
        if fold_data is None:
            log.warning("Fold train=%s test=%d: insufficient data, skipping", train_years, test_year)
            continue
        X_train, y_train, X_test, y_test, scaler, test_idx = fold_data
        n_train, n_test = len(X_train), len(X_test)

        # Check class balance in training fold
        pos_rate_train = float(y_train.mean())
        pos_rate_test  = float(y_test.mean())
        if pos_rate_train < 0.05 or pos_rate_train > 0.95:
            log.warning("  Fold train=%s: extreme class imbalance in training (pos=%.1f%%). "
                        "Results may be unreliable.", train_years, pos_rate_train * 100)

        model = LogisticRegression(C=LR_C, solver=LR_SOLVER, max_iter=LR_MAX_ITER,
                                   random_state=42)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        if pos_rate_test == 0 or pos_rate_test == 1:
            auc = float("nan")
            log.warning("  Fold test=%d: only one class in test set, AUC undefined.", test_year)
        else:
            auc = float(roc_auc_score(y_test, y_prob))

        brier = float(brier_score_loss(y_test, y_prob))
        coef  = {f: round(float(c), 4) for f, c in zip(feature_cols, model.coef_[0])}

        fold_result = {
            "train_years": train_years,
            "test_year": test_year,
            "n_train": n_train,
            "n_test": n_test,
            "pos_rate_train": round(pos_rate_train, 4),
            "pos_rate_test":  round(pos_rate_test, 4),
            "auc_roc": round(auc, 4) if not math.isnan(auc) else None,
            "brier_score": round(brier, 4),
            "intercept": round(float(model.intercept_[0]), 4),
            "coefficients": coef,
        }
        results["folds"].append(fold_result)

        # Store OOS predictions for pooled economic eval
        for idx, yt, yp in zip(test_idx.tolist(), y_test.tolist(), y_prob.tolist()):
            results["oos_predictions"].append({"df_index": idx, "y_true": yt, "y_prob": yp})

        log.info("  Fold train=%s → test=%d: n_train=%d n_test=%d AUC=%.4f Brier=%.4f",
                 train_years, test_year, n_train, n_test,
                 auc if not math.isnan(auc) else -1, brier)
        log.info("    Coefficients: %s", coef)

    # Pooled OOS AUC (across all folds)
    if results["oos_predictions"]:
        y_true_all = np.array([x["y_true"] for x in results["oos_predictions"]])
        y_prob_all = np.array([x["y_prob"] for x in results["oos_predictions"]])
        if y_true_all.mean() > 0 and y_true_all.mean() < 1:
            results["pooled_auc"] = round(float(roc_auc_score(y_true_all, y_prob_all)), 4)
            results["pooled_brier"] = round(float(brier_score_loss(y_true_all, y_prob_all)), 4)
            results["pooled_n"] = int(len(y_true_all))
        else:
            results["pooled_auc"] = None

    return results


def walk_forward_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Time-based walk-forward Ridge regression.

    Returns per-fold metrics and pooled OOS predictions.
    """
    results: Dict[str, Any] = {
        "model": "Ridge",
        "target": target_col,
        "features": feature_cols,
        "alpha": RIDGE_ALPHA,
        "folds": [],
        "oos_predictions": [],
    }

    for train_years, test_year in WALK_FORWARD_FOLDS:
        fold_data = _get_fold_data(df, train_years, test_year, feature_cols, target_col)
        if fold_data is None:
            continue
        X_train, y_train, X_test, y_test, scaler, test_idx = fold_data
        n_train, n_test = len(X_train), len(X_test)

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        naive_pred = np.full_like(y_test, y_train.mean())
        rmse      = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae       = float(mean_absolute_error(y_test, y_pred))
        r2        = float(r2_score(y_test, y_pred))
        naive_rmse = float(np.sqrt(mean_squared_error(y_test, naive_pred)))
        spearman_r, spearman_p = spearmanr(y_test, y_pred)
        coef = {f: round(float(c), 4) for f, c in zip(feature_cols, model.coef_)}

        fold_result = {
            "train_years": train_years,
            "test_year": test_year,
            "n_train": n_train,
            "n_test": n_test,
            "rmse": round(rmse, 6),
            "mae":  round(mae, 6),
            "r2":   round(r2, 4),
            "naive_rmse": round(naive_rmse, 6),
            "rmse_vs_naive": round(rmse / naive_rmse, 4),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 4),
            "intercept": round(float(model.intercept_), 4),
            "coefficients": coef,
        }
        results["folds"].append(fold_result)

        for idx, yt, yp in zip(test_idx.tolist(), y_test.tolist(), y_pred.tolist()):
            results["oos_predictions"].append({"df_index": idx, "y_true": yt, "y_pred": yp})

        log.info("  Fold train=%s → test=%d: n_train=%d n_test=%d "
                 "RMSE=%.4f (naive=%.4f ratio=%.3f) R²=%.4f Spearman_r=%.4f p=%.4f",
                 train_years, test_year, n_train, n_test,
                 rmse, naive_rmse, rmse/naive_rmse, r2, spearman_r, spearman_p)
        log.info("    Coefficients: %s", coef)

    # Pooled OOS metrics
    if results["oos_predictions"]:
        y_true_all = np.array([x["y_true"] for x in results["oos_predictions"]])
        y_pred_all = np.array([x["y_pred"] for x in results["oos_predictions"]])
        naive_all  = np.full_like(y_true_all, y_true_all.mean())
        pooled_rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
        pooled_naive = float(np.sqrt(mean_squared_error(y_true_all, naive_all)))
        sp_r, sp_p   = spearmanr(y_true_all, y_pred_all)
        results["pooled_rmse"] = round(pooled_rmse, 6)
        results["pooled_naive_rmse"] = round(pooled_naive, 6)
        results["pooled_rmse_ratio"] = round(pooled_rmse / pooled_naive, 4)
        results["pooled_r2"] = round(float(r2_score(y_true_all, y_pred_all)), 4)
        results["pooled_spearman_r"] = round(float(sp_r), 4)
        results["pooled_spearman_p"]  = round(float(sp_p), 4)
        results["pooled_n"] = int(len(y_true_all))

    return results


# ── economic evaluation ────────────────────────────────────────────────────────

def economic_evaluation(
    df: pd.DataFrame,
    oos_preds: List[Dict],
    pred_col: str,   # "y_prob" for clf, "y_pred" for reg
    label: str,
) -> Dict[str, Any]:
    """Quintile bucket analysis — the real economic evaluation.

    Pools all OOS predictions, buckets by predicted value, computes:
      - avg front crush per bucket
      - avg back crush per bucket
      - avg spread differential (front - back)
      - avg term_ratio_change
    If the model has signal, we expect a monotone relationship bucket → crush.
    """
    if not oos_preds:
        return {}

    pred_df = pd.DataFrame(oos_preds)
    pred_df = pred_df.set_index("df_index")
    pred_df[pred_col] = pred_df[pred_col]

    # Join back to original df for outcome columns
    merged = df[["front_iv_crush_pct", "back_iv_crush_pct",
                 "term_ratio_change", "spread_differential",
                 "underlying_move_pct", "year"]].join(pred_df, how="inner")
    if merged.empty:
        return {}

    merged["quintile"] = pd.qcut(merged[pred_col], 5, labels=["Q1","Q2","Q3","Q4","Q5"])

    agg = merged.groupby("quintile", observed=True).agg(
        n=("front_iv_crush_pct", "count"),
        avg_front_crush=("front_iv_crush_pct", "mean"),
        avg_back_crush=("back_iv_crush_pct", "mean"),
        avg_spread_diff=("spread_differential", "mean"),
        avg_trc=("term_ratio_change", "mean"),
        avg_move=("underlying_move_pct", "mean"),
        std_front=("front_iv_crush_pct", "std"),
    ).reset_index()

    agg["avg_front_crush_pct"] = (agg["avg_front_crush"] * 100).round(1)
    agg["avg_back_crush_pct"]  = (agg["avg_back_crush"] * 100).round(1)
    agg["avg_spread_diff_pct"] = (agg["avg_spread_diff"] * 100).round(1)
    agg["std_front_pct"]       = (agg["std_front"] * 100).round(1)

    # Spread between Q5 and Q1 (signal quality check)
    q1_crush = agg.loc[agg["quintile"] == "Q1", "avg_front_crush"].values
    q5_crush = agg.loc[agg["quintile"] == "Q5", "avg_front_crush"].values
    if len(q1_crush) and len(q5_crush):
        spread_pct = float(q5_crush[0] - q1_crush[0]) * 100
    else:
        spread_pct = float("nan")

    # Spearman rank correlation: quintile rank vs actual crush
    merged["q_rank"] = merged["quintile"].cat.codes
    sp_r, sp_p = spearmanr(merged["q_rank"], merged["front_iv_crush_pct"])

    result = {
        "label": label,
        "n_oos": int(len(merged)),
        "q5_vs_q1_spread_pct": round(spread_pct, 1),
        "spearman_r_qrank_vs_crush": round(float(sp_r), 4),
        "spearman_p": round(float(sp_p), 4),
        "quintile_table": agg[["quintile","n","avg_front_crush_pct","avg_back_crush_pct",
                                "avg_spread_diff_pct","avg_trc","std_front_pct"]].to_dict("records"),
    }
    return result


def print_economic_eval(econ: Dict[str, Any], title: str) -> None:
    if not econ:
        log.info("  (no OOS predictions available for economic eval)")
        return
    log.info("")
    log.info("── Economic Evaluation: %s ──────────────────────────────", title)
    log.info("  OOS sample: %d events", econ["n_oos"])
    log.info("  Q5 vs Q1 front-crush spread: %+.1f pp", econ["q5_vs_q1_spread_pct"])
    log.info("  Spearman(bucket_rank, actual_crush): r=%.4f  p=%.4f",
             econ["spearman_r_qrank_vs_crush"], econ["spearman_p"])
    log.info("")
    log.info("  %-4s  %5s  %14s  %13s  %14s  %11s",
             "Q", "N", "Avg front crush", "Avg back crush",
             "Avg differential", "Avg TRC")
    log.info("  " + "-" * 70)
    for row in econ["quintile_table"]:
        log.info("  %-4s  %5d  %+13.1f%%  %+12.1f%%  %+13.1f%%  %+10.4f",
                 row["quintile"], row["n"],
                 row["avg_front_crush_pct"], row["avg_back_crush_pct"],
                 row["avg_spread_diff_pct"], row["avg_trc"])


# ── calibration check ─────────────────────────────────────────────────────────

def calibration_check(oos_preds: List[Dict]) -> Dict[str, Any]:
    """Bin predicted probabilities into deciles and compare to actual rates."""
    if not oos_preds:
        return {}
    y_true = np.array([x["y_true"] for x in oos_preds])
    y_prob = np.array([x["y_prob"] for x in oos_preds])

    bins = np.linspace(0, 1, 11)
    result_bins = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        actual_rate = float(y_true[mask].mean())
        pred_mean   = float(y_prob[mask].mean())
        result_bins.append({
            "pred_range": f"[{lo:.1f},{hi:.1f})",
            "n": n,
            "mean_predicted": round(pred_mean, 3),
            "actual_rate": round(actual_rate, 3),
            "calibration_error": round(abs(pred_mean - actual_rate), 3),
        })

    overall_ece = float(np.mean([b["calibration_error"] for b in result_bins]))
    return {"bins": result_bins, "expected_calibration_error": round(overall_ece, 4)}


def print_calibration(cal: Dict[str, Any]) -> None:
    if not cal:
        return
    log.info("")
    log.info("── Calibration Check (predicted prob vs actual rate) ─────────")
    log.info("  %-14s  %5s  %14s  %12s  %12s",
             "Pred range", "N", "Mean pred prob", "Actual rate", "Calib error")
    log.info("  " + "-" * 62)
    for b in cal["bins"]:
        log.info("  %-14s  %5d  %14.3f  %12.3f  %12.3f",
                 b["pred_range"], b["n"],
                 b["mean_predicted"], b["actual_rate"], b["calibration_error"])
    log.info("  Expected Calibration Error (ECE): %.4f", cal["expected_calibration_error"])


# ── conclusion ────────────────────────────────────────────────────────────────

def build_conclusion(
    audit: Dict,
    clf_results: Dict,
    reg_front_results: Dict,
    reg_trc_results: Dict,
    econ_clf: Dict,
    econ_reg: Dict,
) -> Dict[str, Any]:
    """Produce a structured, honest research conclusion."""
    conclusion: Dict[str, Any] = {}

    # 1. Is there a real statistical signal?
    pooled_auc  = clf_results.get("pooled_auc")
    pooled_sp_r = reg_front_results.get("pooled_spearman_r")
    pooled_rmse_ratio = reg_front_results.get("pooled_rmse_ratio")

    has_clf_signal  = pooled_auc is not None and pooled_auc > 0.55
    has_reg_signal  = pooled_sp_r is not None and abs(pooled_sp_r) > 0.10
    has_any_signal  = has_clf_signal or has_reg_signal

    conclusion["has_statistical_signal"] = bool(has_any_signal)
    conclusion["signal_strength"] = (
        "strong"   if (pooled_auc or 0) > 0.65 or abs(pooled_sp_r or 0) > 0.30 else
        "moderate" if (pooled_auc or 0) > 0.58 or abs(pooled_sp_r or 0) > 0.15 else
        "weak"     if has_any_signal else
        "absent"
    )

    # 2. Economic signal
    q5_q1_spread = econ_clf.get("q5_vs_q1_spread_pct", 0)
    econ_sp_r    = econ_clf.get("spearman_r_qrank_vs_crush", 0)
    has_econ_signal = abs(q5_q1_spread) > 5.0 or abs(econ_sp_r) > 0.10
    conclusion["has_economic_signal"] = bool(has_econ_signal)
    conclusion["q5_vs_q1_spread_pp"] = q5_q1_spread

    # 3. Recommended target
    conclusion["recommended_target"] = (
        "front_iv_crush_pct (regression) — magnitude prediction is the right frame."
        " Binary crush/no-crush (>98%% base rate) is trivial."
        " crush_lt_40pct provides useful binary signal (~balanced classes)."
    )

    # 4. Where does it fail
    failures = []
    thin_sym = audit["thin_symbols"]["lt_5_obs"]
    if thin_sym > 10:
        failures.append(f"{thin_sym} symbols have <5 observations — unreliable per-symbol inference")
    fold_aucs = [f.get("auc_roc") for f in clf_results.get("folds", []) if f.get("auc_roc")]
    if fold_aucs and max(fold_aucs) - min(fold_aucs) > 0.10:
        failures.append("AUC varies materially across folds — regime sensitivity likely")
    if audit["by_year"]["2026"]["n"] < 200:
        failures.append("2026 test set is partial-year (small) — high uncertainty")
    conclusion["known_failures"] = failures

    # 5. Week 3 recommendation
    if has_any_signal:
        wk3 = [
            "Feature expansion is justified: add VIX-regime, sector IV, prior-crush.",
            "Add front_dte and exact_expiry_match as additional baseline features.",
            "Investigate whether regression on term_ratio_change outperforms binary cls.",
            "Consider per-sector or per-DTE-regime stratified models.",
        ]
    else:
        wk3 = [
            "Signal is weak with 3 features only. Examine whether additional features change this.",
            "Consider ensemble of sector-level models vs one global model.",
            "Examine whether outliers (IV expansion events) are driving the noise.",
        ]
    conclusion["week3_recommendation"] = wk3

    # 6. Metrics summary
    conclusion["metrics_summary"] = {
        "clf_crush_lt40pct": {
            "pooled_auc": pooled_auc,
            "pooled_brier": clf_results.get("pooled_brier"),
            "pooled_n": clf_results.get("pooled_n"),
        },
        "reg_front_crush": {
            "pooled_rmse_ratio_vs_naive": pooled_rmse_ratio,
            "pooled_spearman_r": pooled_sp_r,
            "pooled_r2": reg_front_results.get("pooled_r2"),
            "pooled_n": reg_front_results.get("pooled_n"),
        },
        "reg_term_ratio_change": {
            "pooled_rmse_ratio_vs_naive": reg_trc_results.get("pooled_rmse_ratio"),
            "pooled_spearman_r": reg_trc_results.get("pooled_spearman_r"),
            "pooled_r2": reg_trc_results.get("pooled_r2"),
            "pooled_n": reg_trc_results.get("pooled_n"),
        },
    }

    return conclusion


def print_conclusion(conclusion: Dict[str, Any]) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 7 — BASELINE CONCLUSION")
    log.info("=" * 72)
    log.info("")
    log.info("Statistical signal:  %s  (strength: %s)",
             "YES" if conclusion["has_statistical_signal"] else "NO",
             conclusion["signal_strength"])
    log.info("Economic signal:     %s  (Q5-Q1 spread: %+.1f pp)",
             "YES" if conclusion["has_economic_signal"] else "NO",
             conclusion["q5_vs_q1_spread_pp"])
    log.info("")
    log.info("── Metrics Summary ──────────────────────────────────────────")
    ms = conclusion["metrics_summary"]
    clf = ms["clf_crush_lt40pct"]
    rfr = ms["reg_front_crush"]
    trc = ms["reg_term_ratio_change"]
    log.info("  Classifier (crush_lt_40pct):    AUC=%-6s  Brier=%-6s  N=%s",
             f"{clf['pooled_auc']:.4f}" if clf["pooled_auc"] else "n/a",
             f"{clf['pooled_brier']:.4f}" if clf["pooled_brier"] else "n/a",
             clf["pooled_n"])
    log.info("  Regression (front_iv_crush_pct): RMSE/naive=%-6s  Spearman_r=%-6s  R²=%-6s  N=%s",
             f"{rfr['pooled_rmse_ratio_vs_naive']:.4f}" if rfr["pooled_rmse_ratio_vs_naive"] else "n/a",
             f"{rfr['pooled_spearman_r']:.4f}" if rfr["pooled_spearman_r"] else "n/a",
             f"{rfr['pooled_r2']:.4f}" if rfr["pooled_r2"] else "n/a",
             rfr["pooled_n"])
    log.info("  Regression (term_ratio_change):  RMSE/naive=%-6s  Spearman_r=%-6s  R²=%-6s  N=%s",
             f"{trc['pooled_rmse_ratio_vs_naive']:.4f}" if trc["pooled_rmse_ratio_vs_naive"] else "n/a",
             f"{trc['pooled_spearman_r']:.4f}" if trc["pooled_spearman_r"] else "n/a",
             f"{trc['pooled_r2']:.4f}" if trc["pooled_r2"] else "n/a",
             trc["pooled_n"])
    log.info("")
    log.info("Correct target frame: %s", conclusion["recommended_target"])
    log.info("")
    if conclusion["known_failures"]:
        log.info("Known failures / caveats:")
        for f in conclusion["known_failures"]:
            log.info("  • %s", f)
    log.info("")
    log.info("Week 3 recommendations:")
    for r in conclusion["week3_recommendation"]:
        log.info("  • %s", r)


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Week 2 baseline IV crush model research pipeline")
    p.add_argument("--db-path", default=None, help="Override path to institutional_ml.db")
    p.add_argument("--quality-tier", default="AB", choices=["A","AB"],
                   help="Which quality tiers to include (default: AB)")
    p.add_argument("--report-dir", default=None, help="Override report output directory")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db_path).expanduser() if args.db_path else DEFAULT_DB_PATH
    report_dir = Path(args.report_dir).expanduser() if args.report_dir else DEFAULT_REPORT_DIR
    quality_tiers = list(args.quality_tier)   # "A" → ["A"],  "AB" → ["A","B"]

    log.info("=" * 72)
    log.info("WEEK 2 BASELINE RESEARCH PIPELINE  version=%s", RESEARCH_VERSION)
    log.info("=" * 72)
    log.info("DB        : %s", db_path)
    log.info("Quality   : tiers %s", quality_tiers)
    log.info("Features  : %s", FEATURES)
    log.info("  NOTE: iv_rv_approx unavailable — using log_back_iv as proxy.")
    log.info("Walk-fwd  : %s", [(tr, te) for tr, te in WALK_FORWARD_FOLDS])
    log.info("")

    # ── Step 0: load and engineer features ───────────────────────────────────
    df = load_labels(db_path, quality_tiers)
    df = engineer_features(df)

    # ── Step 1: audit ─────────────────────────────────────────────────────────
    audit = audit_dataset(df)
    print_audit(audit)

    # ── Step 2: target analysis ───────────────────────────────────────────────
    targets = analyze_targets(df)
    print_target_analysis(targets)

    # ── Step 3–4: walk-forward models ─────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 3-4 — BASELINE MODELS + WALK-FORWARD VALIDATION")
    log.info("=" * 72)

    # A) Logistic regression: crush_lt_40pct
    log.info("")
    log.info("── A) Logistic Regression → crush_lt_40pct ─────────────────")
    clf_results = walk_forward_classification(df, "crush_lt_40pct", FEATURES)
    log.info("  Pooled OOS AUC:   %s", f"{clf_results.get('pooled_auc'):.4f}"
             if clf_results.get("pooled_auc") else "n/a")
    log.info("  Pooled OOS Brier: %s", f"{clf_results.get('pooled_brier'):.4f}"
             if clf_results.get("pooled_brier") else "n/a")

    # B) Ridge regression: front_iv_crush_pct magnitude
    log.info("")
    log.info("── B) Ridge Regression → front_iv_crush_pct ────────────────")
    reg_front = walk_forward_regression(df, "front_iv_crush_pct", FEATURES)
    log.info("  Pooled OOS RMSE/naive:   %s",
             f"{reg_front.get('pooled_rmse_ratio'):.4f}" if reg_front.get("pooled_rmse_ratio") else "n/a")
    log.info("  Pooled OOS Spearman_r:  %s  p=%s",
             f"{reg_front.get('pooled_spearman_r'):.4f}" if reg_front.get("pooled_spearman_r") else "n/a",
             f"{reg_front.get('pooled_spearman_p'):.4f}" if reg_front.get("pooled_spearman_p") else "n/a")

    # C) Ridge regression: term_ratio_change (calendar-spread proxy)
    log.info("")
    log.info("── C) Ridge Regression → term_ratio_change ─────────────────")
    reg_trc = walk_forward_regression(df, "term_ratio_change", FEATURES)
    log.info("  Pooled OOS RMSE/naive:   %s",
             f"{reg_trc.get('pooled_rmse_ratio'):.4f}" if reg_trc.get("pooled_rmse_ratio") else "n/a")
    log.info("  Pooled OOS Spearman_r:  %s  p=%s",
             f"{reg_trc.get('pooled_spearman_r'):.4f}" if reg_trc.get("pooled_spearman_r") else "n/a",
             f"{reg_trc.get('pooled_spearman_p'):.4f}" if reg_trc.get("pooled_spearman_p") else "n/a")

    # ── Step 5: calibration ────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 5 — EVALUATION")
    log.info("=" * 72)
    cal = calibration_check(clf_results.get("oos_predictions", []))
    print_calibration(cal)

    # ── Step 6: economic evaluation ────────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 6 — ECONOMIC EVALUATION")
    log.info("=" * 72)
    log.info("Quintile buckets of OOS predicted value vs actual crush.")
    log.info("Q5 = predicted 'deep crush', Q1 = predicted 'mild crush'.")
    log.info("A model with real edge should show monotone Q1→Q5 crush depth.")

    econ_clf = economic_evaluation(
        df, clf_results.get("oos_predictions", []), "y_prob",
        "LR → crush_lt_40pct"
    )
    print_economic_eval(econ_clf, "LR → crush_lt_40pct (prob buckets)")

    econ_reg = economic_evaluation(
        df, reg_front.get("oos_predictions", []), "y_pred",
        "Ridge → front_iv_crush_pct"
    )
    print_economic_eval(econ_reg, "Ridge → front_iv_crush_pct (pred magnitude buckets)")

    econ_trc = economic_evaluation(
        df, reg_trc.get("oos_predictions", []), "y_pred",
        "Ridge → term_ratio_change"
    )
    print_economic_eval(econ_trc, "Ridge → term_ratio_change")

    # ── Step 7: conclusion ─────────────────────────────────────────────────────
    conclusion = build_conclusion(audit, clf_results, reg_front, reg_trc, econ_clf, econ_reg)
    print_conclusion(conclusion)

    # ── Save report ────────────────────────────────────────────────────────────
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"baseline_research_{RESEARCH_VERSION}_{ts}.json"

    report = {
        "research_version": RESEARCH_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "db_path": str(db_path),
            "quality_tiers": quality_tiers,
            "features": FEATURES,
            "feature_descriptions": FEATURE_DESCRIPTIONS,
            "walk_forward_folds": [
                {"train_years": tr, "test_year": te} for tr, te in WALK_FORWARD_FOLDS
            ],
            "lr_C": LR_C,
            "ridge_alpha": RIDGE_ALPHA,
            "primary_binary_threshold": PRIMARY_BINARY_THRESHOLD,
            "iv_rv_note": (
                "iv_rv_approx not available without realized-vol data. "
                "Substituted log_back_iv (back IV ≈ long-run vol proxy). "
                "Should be replaced by pre_front_iv / 30d_RV when available."
            ),
        },
        "audit": audit,
        "target_analysis": targets,
        "clf_walk_forward": {k: v for k, v in clf_results.items() if k != "oos_predictions"},
        "reg_front_walk_forward": {k: v for k, v in reg_front.items() if k != "oos_predictions"},
        "reg_trc_walk_forward": {k: v for k, v in reg_trc.items() if k != "oos_predictions"},
        "calibration": cal,
        "economic_eval_clf": econ_clf,
        "economic_eval_reg_front": econ_reg,
        "economic_eval_reg_trc": econ_trc,
        "conclusion": conclusion,
    }

    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("")
    log.info("Research report saved to: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
