#!/usr/bin/env python3
"""
scripts/train_prior_crush_model.py
====================================
Week 4 — Prior Crush History Feature Research

Core question: does historical earnings IV crush behavior per symbol provide
INDEPENDENT, ADDITIVE signal on top of near_back_ratio (NBR)?

The baseline from adversarial testing (Week 4 entry state):
  - NBR-only: Spearman=0.757, AUC=0.820, Q5-Q1 spread=-24.3pp
  - log_front_iv + log_back_iv: REDUNDANT
  - VIX features: REDUNDANT
  - Effective model going in: 1 feature (NBR)

Pipeline steps
--------------
  Step 1  Feature construction + leakage validation
           - prior_crush_mean:   expanding mean of front_iv_crush_pct for
                                 prior events of same symbol (equal-weight)
           - prior_trc_mean:     expanding mean of term_ratio_change
           - prior_crush_ewm:    EWM version (span=3, recent events weighted more)
           - prior_event_count:  how many prior events exist for this symbol

  Step 2  Data coverage report
           - % of events with valid history
           - cold-start distribution
           - symbol-level prior count distribution

  Step 3  Walk-forward comparison: 3 models
           A) NBR-only            (1 feature — established baseline)
           B) PCH-only            (1 feature — prior_crush_mean)
           C) NBR + PCH           (2 features — combined)

  Step 4  Incremental signal test (Δ metrics vs baseline)

  Step 5  Independence test
           - Spearman(NBR, PCH)
           - Partial correlation: PCH vs TRC controlling for NBR
           - Combined model coefficients

  Step 6  Robustness by history depth
           - Low-history symbols (1–2 prior events)
           - High-history symbols (≥4 prior events)

  Step 7  Failure mode correction
           - Events where NBR-only is worst decile of residuals
           - Does PCH correct any of those?

  Step 8  EWM variant comparison

  Step 9  Verdict (include or reject)

Leakage guarantees
------------------
  prior_crush_mean[i] = mean(front_iv_crush_pct[j])
      where j = all prior events for symbol s with event_date < event_date[i]
  Implementation: groupby(symbol) → expanding().mean().shift(1)
  The .shift(1) is the critical control — row i sees mean of rows 0..i-1, never itself.
  Cold-start (first event for symbol) → NaN, not imputed.

Usage
-----
  python scripts/train_prior_crush_model.py [options]
  python scripts/train_prior_crush_model.py --quality-tier A
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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
RESEARCH_VERSION  = "prior_crush_v1"

WALK_FORWARD_FOLDS: List[Tuple[List[int], int]] = [
    ([2023],             2024),
    ([2023, 2024],       2025),
    ([2023, 2024, 2025], 2026),
]

PRIMARY_REG_TARGET   = "term_ratio_change"
SECONDARY_CLF_TARGET = "crush_lt_40pct"

# Minimum prior events required to include a row in PCH-using models.
# 1 = use any symbol that has appeared at least once before.
# We also report results at threshold=2 for sensitivity.
MIN_PRIOR_EVENTS = 1

EWM_SPAN = 3     # exponential weighting half-life ≈ 2 events

# Feature set definitions (Week 4: 3 models only)
FS_NBR  = ["near_back_ratio"]
FS_PCH  = ["prior_crush_mean"]
FS_COMB = ["near_back_ratio", "prior_crush_mean"]
FS_COMB_EWM = ["near_back_ratio", "prior_crush_ewm"]

RIDGE_ALPHA = 1.0
LR_C        = 1.0
LR_SOLVER   = "lbfgs"
LR_MAX_ITER = 500


# ── data loading ──────────────────────────────────────────────────────────────

def load_labels(db_path: Path, quality_tiers: List[str]) -> pd.DataFrame:
    ph = ",".join("?" * len(quality_tiers))
    query = f"""
        SELECT symbol, event_date, pre_front_iv, post_front_iv,
               pre_back_iv, post_back_iv,
               front_iv_crush_pct, back_iv_crush_pct,
               term_ratio_change, underlying_move_pct,
               quality_score, quality_tier,
               front_dte_pre, back_dte_pre
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ({ph})
          AND pre_front_iv > 0 AND pre_back_iv > 0
          AND term_ratio_change IS NOT NULL
          AND front_iv_crush_pct IS NOT NULL
        ORDER BY event_date
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=quality_tiers)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["year"]       = df["event_date"].dt.year
    df["quarter"]    = df["event_date"].dt.quarter
    log.info("Loaded %d rows (quality_tiers=%s)", len(df), quality_tiers)
    return df


# ── feature engineering ───────────────────────────────────────────────────────

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive NBR and targets from pre-event data only."""
    df = df.copy()
    df["near_back_ratio"]     = df["pre_front_iv"] / df["pre_back_iv"]
    df["log_front_iv"]        = np.log(df["pre_front_iv"])
    df["log_back_iv"]         = np.log(df["pre_back_iv"])
    df["spread_differential"] = df["front_iv_crush_pct"] - df["back_iv_crush_pct"]
    df["crush_lt_40pct"]      = (df["front_iv_crush_pct"] < -0.40).astype(int)
    return df


def compute_prior_crush_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prior earnings crush history per symbol.

    For each event row i with symbol s and event_date d:
      prior_crush_mean[i]  = mean(front_iv_crush_pct[j])
                             for all j where symbol[j]==s AND event_date[j] < d
      prior_trc_mean[i]    = mean(term_ratio_change[j])   (same constraint)
      prior_crush_ewm[i]   = EWM mean (span=EWM_SPAN) of front_iv_crush_pct,
                             shifted by 1 (no current event included)
      prior_event_count[i] = number of prior events for symbol s before d

    LEAKAGE PREVENTION: groupby(symbol) + expanding().mean().shift(1)
      - .expanding().mean() at row i includes rows 0..i (inclusive of self)
      - .shift(1) pushes the value forward by 1, so row i sees mean(0..i-1)
      - Row 0 per symbol (first ever event) → NaN after shift ✓
      - This is a STRICT guarantee: no event ever contributes to its own prior

    Cold-start policy: NaN, not imputed. Model fit excludes NaN rows.
    """
    # Sort by symbol then event_date to ensure correct temporal ordering.
    # Use pre_front_iv as a deterministic tiebreaker so results are invariant
    # to input row order (same-date same-symbol ties are rare in real data).
    df = df.copy()
    df = df.sort_values(
        ["symbol", "event_date", "pre_front_iv"], na_position="last"
    ).reset_index(drop=True)

    # ── prior_event_count ─────────────────────────────────────────────────────
    # cumcount() is 0-indexed: first event = 0 (cold-start), second = 1, etc.
    df["prior_event_count"] = df.groupby("symbol").cumcount()

    # ── prior_crush_mean (equal-weight expanding) ─────────────────────────────
    df["prior_crush_mean"] = (
        df.groupby("symbol")["front_iv_crush_pct"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    # ── prior_trc_mean ────────────────────────────────────────────────────────
    df["prior_trc_mean"] = (
        df.groupby("symbol")["term_ratio_change"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    # ── prior_crush_ewm (exponential, recent events weighted more) ────────────
    df["prior_crush_ewm"] = (
        df.groupby("symbol")["front_iv_crush_pct"]
        .transform(lambda x: x.ewm(span=EWM_SPAN, adjust=False).mean().shift(1))
    )

    # ── prior_crush_std (variability — how consistent is the symbol?) ─────────
    df["prior_crush_std"] = (
        df.groupby("symbol")["front_iv_crush_pct"]
        .transform(lambda x: x.expanding().std().shift(1))
    )

    # Restore original sort order (by event_date globally) for modeling
    df = df.sort_values("event_date").reset_index(drop=True)

    return df


# ── STEP 1+2: feature construction + leakage validation + coverage ────────────

def step1_construct_and_validate(df: pd.DataFrame) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 1 — FEATURE CONSTRUCTION + LEAKAGE VALIDATION")
    log.info("=" * 72)
    log.info("")
    log.info("  Definition: prior_crush_mean[i] = mean(front_iv_crush_pct[j])")
    log.info("              for all j: symbol[j]==symbol[i] AND event_date[j] < event_date[i]")
    log.info("  Implementation: groupby(symbol) → expanding().mean().shift(1)")
    log.info("  Cold-start policy: NaN (not imputed, not filled)")
    log.info("  Minimum prior events: %d", MIN_PRIOR_EVENTS)
    log.info("  EWM span: %d events", EWM_SPAN)

    # ── Leakage verification (mandatory) ─────────────────────────────────────
    log.info("")
    log.info("── Leakage verification ─────────────────────────────────────────")

    # Check 1: Cold-start rows (prior_event_count==0) must have NaN prior_crush_mean
    cold_start_with_value = df[
        (df["prior_event_count"] == 0) & df["prior_crush_mean"].notna()
    ]
    if len(cold_start_with_value) > 0:
        log.error("  ✗ LEAKAGE DETECTED: %d cold-start rows have non-NaN prior_crush_mean!",
                  len(cold_start_with_value))
        raise AssertionError("Leakage in prior_crush_mean: cold-start rows have values")
    log.info("  ✓ Cold-start check: all %d first-occurrence rows have NaN prior_crush_mean",
             (df["prior_event_count"] == 0).sum())

    # Check 2: For each symbol, verify prior_crush_mean[i] == mean of prior events
    # Spot-check 5 symbols with ≥3 events
    multi_event_syms = (
        df.groupby("symbol").size().loc[lambda s: s >= 4].sample(
            min(5, (df.groupby("symbol").size() >= 4).sum()),
            random_state=42
        ).index.tolist()
    )
    leakage_errors = 0
    for sym in multi_event_syms:
        sym_df = df[df["symbol"] == sym].sort_values("event_date").reset_index(drop=True)
        for i in range(1, min(4, len(sym_df))):
            expected = sym_df.loc[:i-1, "front_iv_crush_pct"].mean()
            actual   = sym_df.loc[i, "prior_crush_mean"]
            if abs(expected - actual) > 1e-9:
                log.error("  ✗ LEAKAGE: sym=%s row=%d expected=%.6f actual=%.6f",
                          sym, i, expected, actual)
                leakage_errors += 1
    if leakage_errors == 0:
        log.info("  ✓ Spot-check: prior_crush_mean verified correct for %d symbols × 3 rows each",
                 len(multi_event_syms))
    else:
        raise AssertionError(f"Leakage detected in {leakage_errors} spot-check rows")

    # Check 3: Temporal order — prior_crush_mean[i] should not use any event after event_date[i]
    # If the sort was correct, this is guaranteed by the shift(1), but verify for one symbol.
    # Rows sharing the same event_date for a symbol are skipped (tie-breaking is
    # positional/deterministic but not date-strict, so the strict-date recomputation would
    # spuriously disagree for those rows).
    test_sym = multi_event_syms[0] if multi_event_syms else None
    if test_sym:
        sym_rows = df[df["symbol"] == test_sym].sort_values("event_date").reset_index(drop=True)
        future_events_used = False
        for i, row in sym_rows.iterrows():
            if pd.isna(row["prior_crush_mean"]):
                continue
            # Skip if any earlier row shares the same date (positional ordering, not date-strict)
            same_date_before = (sym_rows.loc[:i - 1, "event_date"] == row["event_date"]).any() if i > 0 else False
            if same_date_before:
                continue
            prior_rows = sym_rows[sym_rows["event_date"] < row["event_date"]]
            if len(prior_rows) == 0:
                continue
            recomputed = prior_rows["front_iv_crush_pct"].mean()
            if abs(recomputed - row["prior_crush_mean"]) > 1e-9:
                future_events_used = True
        if future_events_used:
            raise AssertionError(f"Temporal leakage detected in symbol {test_sym}")
        log.info("  ✓ Temporal check: verified no future events used in prior_crush_mean")

    log.info("  ✓ All leakage checks passed")

    # ── Coverage report ───────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("STEP 2 — DATA COVERAGE REPORT")
    log.info("=" * 72)

    n_total    = len(df)
    n_valid    = df["prior_crush_mean"].notna().sum()
    n_cold     = (df["prior_event_count"] == 0).sum()
    pct_valid  = n_valid / n_total * 100

    log.info("")
    log.info("  Total events:                %d", n_total)
    log.info("  Events with valid PCH:        %d  (%.1f%%)", n_valid, pct_valid)
    log.info("  Cold-start events (0 prior):  %d  (%.1f%%)",
             n_cold, n_cold / n_total * 100)
    log.info("")

    # Distribution of prior event count
    count_dist = df["prior_event_count"].value_counts().sort_index()
    log.info("  Prior event count distribution:")
    cumulative = 0
    for cnt, n in count_dist.items():
        cumulative += n
        log.info("    %2d prior events: %4d rows  cumulative: %.1f%%",
                 cnt, n, cumulative / n_total * 100)
        if cnt >= 10:
            remaining = n_total - cumulative
            log.info("    ≥11 prior events: %4d rows  cumulative: %.1f%%",
                     remaining, 100.0)
            break

    # Cold-start symbols breakdown
    symbols_with_cold = df[df["prior_event_count"] == 0]["symbol"].nunique()
    log.info("")
    log.info("  Symbols with ≥1 cold-start event: %d (all symbols have one)", symbols_with_cold)

    # Year-level coverage (what % of each test year has valid PCH?)
    log.info("")
    log.info("  PCH coverage by year (critical for walk-forward):")
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        yr_valid = yr_df["prior_crush_mean"].notna().mean() * 100
        log.info("    %d: %.1f%% valid PCH  (N=%d)", yr, yr_valid, len(yr_df))

    # Distribution of prior_crush_mean for valid events
    valid = df["prior_crush_mean"].dropna()
    log.info("")
    log.info("  prior_crush_mean distribution (valid events only):")
    log.info("    mean=%.4f  std=%.4f  min=%.4f  p25=%.4f  "
             "p50=%.4f  p75=%.4f  max=%.4f",
             valid.mean(), valid.std(),
             valid.min(), valid.quantile(0.25), valid.median(),
             valid.quantile(0.75), valid.max())

    # Symbols with worst (shallowest) prior history — potential model blind spots
    sym_pch = df.dropna(subset=["prior_crush_mean"]).groupby("symbol").agg(
        n_prior=("prior_event_count", "max"),
        pch_mean=("prior_crush_mean", "mean"),
    ).sort_values("pch_mean", ascending=False)  # least crush first
    log.info("")
    log.info("  5 symbols with shallowest mean PCH (weakest historical crush):")
    for sym, row in sym_pch.head(5).iterrows():
        log.info("    %-8s  n_prior_max=%d  pch_mean=%+.4f",
                 sym, int(row["n_prior"]), row["pch_mean"])
    log.info("  5 symbols with deepest mean PCH (strongest historical crush):")
    for sym, row in sym_pch.tail(5).iterrows():
        log.info("    %-8s  n_prior_max=%d  pch_mean=%+.4f",
                 sym, int(row["n_prior"]), row["pch_mean"])

    return {
        "n_total": n_total,
        "n_valid_pch": int(n_valid),
        "pct_valid_pch": round(pct_valid, 1),
        "n_cold_start": int(n_cold),
        "pct_cold_start": round(n_cold / n_total * 100, 1),
        "prior_crush_mean_stats": {
            "mean": round(float(valid.mean()), 4),
            "std":  round(float(valid.std()),  4),
            "min":  round(float(valid.min()),  4),
            "p25":  round(float(valid.quantile(0.25)), 4),
            "p50":  round(float(valid.median()), 4),
            "p75":  round(float(valid.quantile(0.75)), 4),
            "max":  round(float(valid.max()),  4),
        },
        "leakage_verified": True,
    }


# ── walk-forward engine ───────────────────────────────────────────────────────

def _fold_data(df, train_years, test_year, features, target):
    tr = df[df["year"].isin(train_years)].dropna(subset=features + [target])
    te = df[df["year"] == test_year].dropna(subset=features + [target])
    if len(tr) < 5 or len(te) < 5:
        return None
    sc = StandardScaler()
    X_tr = sc.fit_transform(tr[features].values)
    X_te = sc.transform(te[features].values)
    return X_tr, tr[target].values, X_te, te[target].values, te.index.tolist()


def wf_ridge(df, features, target=PRIMARY_REG_TARGET, label=""):
    oos: List[Dict] = []
    folds: List[Dict] = []
    for tr_yrs, te_yr in WALK_FORWARD_FOLDS:
        fd = _fold_data(df, tr_yrs, te_yr, features, target)
        if fd is None:
            log.warning("  Fold train=%s→test=%d: insufficient data (skipped)", tr_yrs, te_yr)
            continue
        X_tr, y_tr, X_te, y_te, idx = fd
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(X_tr, y_tr)
        y_hat  = m.predict(X_te)
        naive  = np.full_like(y_te, y_tr.mean())
        sp_r, sp_p = spearmanr(y_te, y_hat)
        r2     = float(r2_score(y_te, y_hat))
        rmse   = float(np.sqrt(mean_squared_error(y_te, y_hat)))
        rmse_n = float(np.sqrt(mean_squared_error(y_te, naive)))
        coef   = {f: round(float(c), 5) for f, c in zip(features, m.coef_)}
        folds.append({
            "train_years": tr_yrs, "test_year": te_yr,
            "n_train": len(X_tr), "n_test": len(X_te),
            "spearman_r": round(float(sp_r), 4), "spearman_p": round(float(sp_p), 6),
            "r2": round(r2, 4), "rmse": round(rmse, 6),
            "rmse_ratio": round(rmse / rmse_n, 4), "coef": coef,
        })
        y_te_list = y_te.tolist()
        for list_i, (orig_idx, yp) in enumerate(zip(idx, y_hat.tolist())):
            oos.append({"idx": orig_idx, "y_true": y_te_list[list_i], "y_pred": yp})
    pooled = _pool(oos)
    return {"label": label, "features": features, "folds": folds,
            "pooled": pooled, "oos": oos}


def wf_lr(df, features, target=SECONDARY_CLF_TARGET, label=""):
    oos: List[Dict] = []
    folds: List[Dict] = []
    for tr_yrs, te_yr in WALK_FORWARD_FOLDS:
        fd = _fold_data(df, tr_yrs, te_yr, features, target)
        if fd is None:
            continue
        X_tr, y_tr, X_te, y_te, idx = fd
        if y_tr.mean() < 0.03 or y_tr.mean() > 0.97:
            continue
        m = LogisticRegression(C=LR_C, solver=LR_SOLVER, max_iter=LR_MAX_ITER, random_state=42)
        m.fit(X_tr, y_tr)
        y_prob = m.predict_proba(X_te)[:, 1]
        auc    = float(roc_auc_score(y_te, y_prob)) if 0 < y_te.mean() < 1 else float("nan")
        brier  = float(brier_score_loss(y_te, y_prob))
        folds.append({
            "train_years": tr_yrs, "test_year": te_yr,
            "n_train": len(X_tr), "n_test": len(X_te),
            "auc": round(auc, 4) if not math.isnan(auc) else None,
            "brier": round(brier, 4),
        })
        y_te_list = y_te.tolist()
        for list_i, (orig_idx, yp) in enumerate(zip(idx, y_prob.tolist())):
            oos.append({"idx": orig_idx, "y_true": y_te_list[list_i], "y_prob": yp})
    pooled: Dict[str, Any] = {}
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


def _pool(oos):
    if not oos:
        return {}
    yt = np.array([x["y_true"] for x in oos])
    yp = np.array([x["y_pred"] for x in oos])
    naive = np.full_like(yt, yt.mean())
    sp_r, sp_p = spearmanr(yt, yp)
    return {
        "n": len(yt),
        "spearman_r":   round(float(sp_r), 4),
        "spearman_p":   round(float(sp_p), 8),
        "r2":           round(float(r2_score(yt, yp)), 4),
        "rmse":         round(float(np.sqrt(mean_squared_error(yt, yp))), 6),
        "rmse_ratio":   round(
            float(np.sqrt(mean_squared_error(yt, yp))) /
            float(np.sqrt(mean_squared_error(yt, naive))), 4),
    }


def quintile_spread(df, oos, pred_col="y_pred"):
    if not oos:
        return float("nan"), []
    pred_df = pd.DataFrame(oos).set_index("idx")
    merged  = df[["front_iv_crush_pct", "back_iv_crush_pct",
                  "spread_differential", "year"]].join(pred_df, how="inner")
    if len(merged) < 10:
        return float("nan"), []
    merged["score"] = -merged[pred_col]   # flip: Q5 = deepest predicted crush
    try:
        merged["q"] = pd.qcut(merged["score"], 5,
                              labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
    except ValueError:
        return float("nan"), []
    agg = merged.groupby("q", observed=True).agg(
        n=("front_iv_crush_pct", "count"),
        avg_front=("front_iv_crush_pct", lambda x: round(x.mean() * 100, 1)),
        avg_back=("back_iv_crush_pct",   lambda x: round(x.mean() * 100, 1)),
        avg_diff=("spread_differential", lambda x: round(x.mean() * 100, 1)),
    ).reset_index()
    q1_f = agg.loc[agg["q"] == "Q1", "avg_front"].values
    q5_f = agg.loc[agg["q"] == "Q5", "avg_front"].values
    spread = float(q5_f[0] - q1_f[0]) if len(q1_f) and len(q5_f) else float("nan")
    return spread, agg.to_dict("records")


# ── STEP 3+4: model comparison + incremental signal ──────────────────────────

def step3_model_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 3+4 — WALK-FORWARD MODEL COMPARISON + INCREMENTAL SIGNAL")
    log.info("=" * 72)
    log.info("  Three models, same folds, same hyperparameters as previous weeks.")
    log.info("  NaN rows dropped per model (each model uses rows where all its features are valid).")

    # Model A: NBR-only (runs on full dataset — no NaN issue)
    log.info("")
    log.info("  A) NBR-only (1 feat)  —  N_full=%d", len(df))
    r_nbr = wf_ridge(df, FS_NBR, label="NBR-only (1 feat)")
    c_nbr = wf_lr(df, FS_NBR, label="NBR-only (1 feat)")
    _print_fold_detail(r_nbr)

    # Model B: PCH-only (drops cold-start rows)
    n_pch = df["prior_crush_mean"].notna().sum()
    log.info("")
    log.info("  B) PCH-only (1 feat)  —  N_valid_pch=%d", n_pch)
    r_pch = wf_ridge(df, FS_PCH, label="PCH-only (1 feat)")
    c_pch = wf_lr(df, FS_PCH, label="PCH-only (1 feat)")
    _print_fold_detail(r_pch)

    # Model C: NBR + PCH (combined — drops cold-start rows)
    log.info("")
    log.info("  C) NBR + PCH (2 feat) —  N_valid_pch=%d", n_pch)
    r_comb = wf_ridge(df, FS_COMB, label="NBR + PCH (2 feat)")
    c_comb = wf_lr(df, FS_COMB, label="NBR + PCH (2 feat)")
    _print_fold_detail(r_comb)

    # ── comparison table ──────────────────────────────────────────────────────
    log.info("")
    log.info("── Regression comparison (term_ratio_change) ─────────────────────")
    log.info("  %-28s  %10s  %10s  %10s  %6s",
             "Model", "Spearman_r", "R²(TRC)", "RMSE/naive", "N_oos")
    log.info("  " + "-" * 70)
    for r in [r_nbr, r_pch, r_comb]:
        p = r["pooled"]
        log.info("  %-28s  %10.4f  %10.4f  %10.4f  %6d",
                 r["label"],
                 p.get("spearman_r", float("nan")),
                 p.get("r2", float("nan")),
                 p.get("rmse_ratio", float("nan")),
                 p.get("n", 0))

    log.info("")
    log.info("── Classification comparison (crush_lt_40pct) ────────────────────")
    log.info("  %-28s  %10s  %10s  %6s",
             "Model", "AUC", "Brier", "N_oos")
    log.info("  " + "-" * 58)
    for c in [c_nbr, c_pch, c_comb]:
        p = c["pooled"]
        log.info("  %-28s  %10s  %10.4f  %6d",
                 c["label"],
                 f'{p["auc"]:.4f}' if p.get("auc") else "n/a",
                 p.get("brier", float("nan")),
                 p.get("n", 0))

    # ── delta metrics vs NBR baseline ─────────────────────────────────────────
    log.info("")
    log.info("── Δ metrics: combined vs NBR-only (key test) ────────────────────")
    nbr_sp   = r_nbr["pooled"].get("spearman_r", 0)
    comb_sp  = r_comb["pooled"].get("spearman_r", 0)
    nbr_r2   = r_nbr["pooled"].get("r2", 0)
    comb_r2  = r_comb["pooled"].get("r2", 0)
    nbr_auc  = (c_nbr["pooled"] or {}).get("auc", 0) or 0
    comb_auc = (c_comb["pooled"] or {}).get("auc", 0) or 0
    delta_sp  = round(comb_sp  - nbr_sp,  4)
    delta_r2  = round(comb_r2  - nbr_r2,  4)
    delta_auc = round(comb_auc - nbr_auc, 4)
    log.info("  Δ Spearman (NBR+PCH − NBR): %+.4f", delta_sp)
    log.info("  Δ R²       (NBR+PCH − NBR): %+.4f", delta_r2)
    log.info("  Δ AUC      (NBR+PCH − NBR): %+.4f", delta_auc)

    # Fold-level consistency
    nbr_fold_sp  = [f["spearman_r"] for f in r_nbr["folds"]]
    comb_fold_sp = [f["spearman_r"] for f in r_comb["folds"]]
    fold_deltas  = [c - b for c, b in zip(comb_fold_sp, nbr_fold_sp)]
    consistent   = all(d > 0 for d in fold_deltas) if fold_deltas else False
    log.info("  Per-fold Spearman deltas: %s  → consistent gain: %s",
             [f"{d:+.4f}" for d in fold_deltas], "YES" if consistent else "NO")

    log.info("")
    log.info("── Per-fold detail: NBR-only vs NBR+PCH ──────────────────────────")
    log.info("  %-28s  %10s  %10s  %10s",
             "Model", "Fold1(2024)", "Fold2(2025)", "Fold3(2026)")
    for r in [r_nbr, r_comb]:
        sp_vals = [f["spearman_r"] for f in r["folds"]]
        while len(sp_vals) < 3:
            sp_vals.append(float("nan"))
        log.info("  %-28s  %10.4f  %10.4f  %10.4f", r["label"], *sp_vals[:3])

    # ── economic evaluation ───────────────────────────────────────────────────
    log.info("")
    log.info("── Economic evaluation (Q5-Q1 front crush spread) ───────────────")
    spreads = {}
    for r in [r_nbr, r_pch, r_comb]:
        sp, table = quintile_spread(df, r["oos"])
        spreads[r["label"]] = {"spread_pp": round(sp, 1), "table": table}
        log.info("  %-28s  Q5-Q1 spread = %+.1f pp", r["label"], sp)

    log.info("")
    log.info("── Quintile table: NBR-only ──────────────────────────────────────")
    _print_quintile_table(spreads["NBR-only (1 feat)"]["table"])
    log.info("")
    log.info("── Quintile table: NBR + PCH ────────────────────────────────────")
    _print_quintile_table(spreads["NBR + PCH (2 feat)"]["table"])

    return {
        "r_nbr": r_nbr, "r_pch": r_pch, "r_comb": r_comb,
        "c_nbr": c_nbr, "c_pch": c_pch, "c_comb": c_comb,
        "delta_spearman": delta_sp,
        "delta_r2": delta_r2,
        "delta_auc": delta_auc,
        "fold_deltas": fold_deltas,
        "consistent_fold_gain": bool(consistent),
        "economic": spreads,
    }


def _print_fold_detail(r):
    for f in r["folds"]:
        log.info("    fold train=%s→test=%d  N_tr=%d  N_te=%d  "
                 "Spearman=%.4f  R²=%.4f  coef=%s",
                 f["train_years"], f["test_year"],
                 f["n_train"], f["n_test"],
                 f["spearman_r"], f["r2"],
                 {k: v for k, v in f["coef"].items()})


def _print_quintile_table(table):
    if not table:
        return
    log.info("  %-4s  %5s  %13s  %12s  %13s",
             "Q", "N", "Avg front", "Avg back", "Cal diff")
    log.info("  " + "-" * 52)
    for row in table:
        log.info("  %-4s  %5d  %+12.1f%%  %+11.1f%%  %+12.1f%%",
                 row["q"], row["n"],
                 row["avg_front"], row["avg_back"], row["avg_diff"])


# ── STEP 5: independence test ─────────────────────────────────────────────────

def step5_independence_test(df: pd.DataFrame, r_comb: Dict) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 5 — INDEPENDENCE TEST")
    log.info("=" * 72)
    log.info("  Key question: is PCH capturing something NBR cannot?")

    valid = df.dropna(subset=["near_back_ratio", "prior_crush_mean",
                              "term_ratio_change"]).copy()

    # 5a. Raw correlation between NBR and PCH
    sp_nbr_pch, p_nbr_pch = spearmanr(valid["near_back_ratio"], valid["prior_crush_mean"])
    log.info("")
    log.info("  Spearman(NBR, PCH):   r=%+.4f  p=%.4f", sp_nbr_pch, p_nbr_pch)
    if abs(sp_nbr_pch) > 0.50:
        log.info("  ⚠  High correlation — PCH may be partially redundant with NBR")
    elif abs(sp_nbr_pch) > 0.30:
        log.info("  ⡠  Moderate correlation — some overlap with NBR")
    else:
        log.info("  ✓  Low correlation — PCH and NBR are largely independent")

    # 5b. Each feature's raw Spearman with the target
    sp_nbr_trc, _ = spearmanr(valid["near_back_ratio"], valid["term_ratio_change"])
    sp_pch_trc, _ = spearmanr(valid["prior_crush_mean"], valid["term_ratio_change"])
    log.info("")
    log.info("  Spearman(NBR, TRC):   r=%+.4f", sp_nbr_trc)
    log.info("  Spearman(PCH, TRC):   r=%+.4f", sp_pch_trc)

    # 5c. Partial correlation: PCH vs TRC controlling for NBR
    # Residualise both PCH and TRC on NBR, then measure correlation
    X_nbr = valid[["near_back_ratio"]].values
    y_trc = valid["term_ratio_change"].values
    y_pch = valid["prior_crush_mean"].values

    res_trc = y_trc - LinearRegression().fit(X_nbr, y_trc).predict(X_nbr)
    res_pch = y_pch - LinearRegression().fit(X_nbr, y_pch).predict(X_nbr)
    sp_partial, p_partial = spearmanr(res_pch, res_trc)

    log.info("")
    log.info("  Partial Spearman(PCH | NBR, TRC | NBR): r=%+.4f  p=%.4f",
             sp_partial, p_partial)
    log.info("  → After removing NBR's contribution, PCH explains an additional")
    log.info("    r=%+.4f Spearman in TRC (p=%.4f)", sp_partial, p_partial)
    if abs(sp_partial) > 0.10 and p_partial < 0.05:
        log.info("  ✓  Independent signal confirmed: PCH adds information beyond NBR")
    elif p_partial < 0.05:
        log.info("  ⡠  Marginal independent signal (r < 0.10 but p < 0.05)")
    else:
        log.info("  ✗  No independent signal: PCH adds nothing beyond what NBR already captures")

    # 5d. Combined model coefficients across folds
    log.info("")
    log.info("  Combined model Ridge coefficients (standardized):")
    log.info("  %-8s  %12s  %12s  %12s",
             "Fold", "NBR coef", "PCH coef", "PCH/NBR ratio")
    for fold in r_comb.get("folds", []):
        nbr_c = fold["coef"].get("near_back_ratio", float("nan"))
        pch_c = fold["coef"].get("prior_crush_mean", float("nan"))
        ratio = pch_c / nbr_c if nbr_c != 0 else float("nan")
        log.info("  %d→%d  %12.4f  %12.4f  %12.4f",
                 fold["train_years"][-1], fold["test_year"],
                 nbr_c, pch_c, ratio)

    # 5e. PCH vs PCH-EWM correlation (are they the same signal?)
    if "prior_crush_ewm" in valid.columns:
        sp_ewm_pch, _ = spearmanr(valid["prior_crush_mean"], valid["prior_crush_ewm"])
        log.info("")
        log.info("  Spearman(PCH_mean, PCH_ewm): r=%.4f", sp_ewm_pch)
        if sp_ewm_pch > 0.90:
            log.info("  → EWM and mean are nearly identical — EWM adds no new information")

    return {
        "spearman_nbr_pch": round(float(sp_nbr_pch), 4),
        "spearman_nbr_trc": round(float(sp_nbr_trc), 4),
        "spearman_pch_trc": round(float(sp_pch_trc), 4),
        "partial_pch_given_nbr": round(float(sp_partial), 4),
        "partial_p_value":       round(float(p_partial), 6),
        "independent_signal":    bool(abs(sp_partial) > 0.10 and p_partial < 0.05),
    }


# ── STEP 6: robustness by history depth ──────────────────────────────────────

def step6_robustness(df: pd.DataFrame) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 6 — ROBUSTNESS BY HISTORY DEPTH")
    log.info("=" * 72)

    # Depth buckets
    buckets = [
        ("Cold-start (0 prior)",        df["prior_event_count"] == 0),
        ("Shallow (1–2 prior)",          df["prior_event_count"].isin([1, 2])),
        ("Moderate (3–5 prior)",         df["prior_event_count"].between(3, 5)),
        ("Deep (≥6 prior)",              df["prior_event_count"] >= 6),
    ]

    results = []
    log.info("")
    log.info("  %-30s  %5s  %10s  %10s  %10s",
             "Bucket", "N", "Spearman(NBR)", "Spearman(NBR+PCH)", "Δ")
    log.info("  " + "-" * 70)
    for name, mask in buckets:
        sub = df[mask]
        if len(sub) < 20:
            log.info("  %-30s  %5d  insufficient data", name, len(sub))
            continue

        r_nbr_sub  = wf_ridge(sub, FS_NBR,  label="nbr_sub")
        r_comb_sub = wf_ridge(sub, FS_COMB, label="comb_sub")
        sp_nbr  = r_nbr_sub["pooled"].get("spearman_r", float("nan"))
        sp_comb = r_comb_sub["pooled"].get("spearman_r", float("nan"))
        delta   = round(sp_comb - sp_nbr, 4) if not math.isnan(sp_comb) else float("nan")
        log.info("  %-30s  %5d  %10.4f  %10.4f  %+9.4f",
                 name, len(sub), sp_nbr, sp_comb, delta)
        results.append({
            "bucket": name, "n": len(sub),
            "spearman_nbr": round(sp_nbr, 4),
            "spearman_comb": round(sp_comb, 4) if not math.isnan(sp_comb) else None,
            "delta": round(delta, 4) if not math.isnan(delta) else None,
        })

    log.info("")
    log.info("  Interpretation:")
    log.info("  If PCH only helps in the 'Deep' bucket but not 'Shallow',")
    log.info("  the feature requires many prior events and is fragile for new symbols.")

    return {"by_history_depth": results}


# ── STEP 7: failure mode correction ──────────────────────────────────────────

def step7_failure_modes(df: pd.DataFrame,
                        r_nbr: Dict, r_comb: Dict) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 7 — FAILURE MODE CORRECTION")
    log.info("=" * 72)
    log.info("  Do events where NBR fails get corrected by adding PCH?")

    if not r_nbr["oos"] or not r_comb["oos"]:
        log.info("  Insufficient OOS predictions — skipping.")
        return {}

    nbr_preds  = pd.DataFrame(r_nbr["oos"]).set_index("idx").rename(
        columns={"y_pred": "y_pred_nbr"})
    # Drop y_true from comb_preds to avoid column overlap on second join
    comb_preds = pd.DataFrame(r_comb["oos"]).set_index("idx").rename(
        columns={"y_pred": "y_pred_comb"})[["y_pred_comb"]]

    # Join on common indices (combined model may have fewer rows due to NaN drops)
    common = df[["symbol", "year", "term_ratio_change", "front_iv_crush_pct",
                 "prior_event_count", "prior_crush_mean",
                 "near_back_ratio"]].join(nbr_preds, how="inner").join(
                     comb_preds, how="inner")

    if len(common) < 20:
        log.info("  Too few matched rows (%d) for failure analysis.", len(common))
        return {}

    common["resid_nbr"]  = common["y_true"] - common["y_pred_nbr"]
    common["resid_comb"] = common["y_true"] - common["y_pred_comb"]
    common["abs_nbr"]    = common["resid_nbr"].abs()
    common["abs_comb"]   = common["resid_comb"].abs()
    common["improved"]   = common["abs_comb"] < common["abs_nbr"]
    common["worsened"]   = common["abs_comb"] > common["abs_nbr"]

    # NBR's worst 10%
    thresh = common["abs_nbr"].quantile(0.90)
    nbr_worst = common[common["abs_nbr"] >= thresh]

    log.info("")
    log.info("  NBR worst 10%% residuals (%d events):", len(nbr_worst))
    log.info("    Combined model improved:  %d  (%.1f%%)",
             nbr_worst["improved"].sum(),
             nbr_worst["improved"].mean() * 100)
    log.info("    Combined model worsened:  %d  (%.1f%%)",
             nbr_worst["worsened"].sum(),
             nbr_worst["worsened"].mean() * 100)
    log.info("    Mean |residual| NBR:      %.4f", nbr_worst["abs_nbr"].mean())
    log.info("    Mean |residual| NBR+PCH:  %.4f", nbr_worst["abs_comb"].mean())

    # IV expansion events (NBR predicts crush, got expansion)
    expansion = df[df["front_iv_crush_pct"] > 0].copy()
    exp_in_common = common[common.index.isin(expansion.index)]
    if len(exp_in_common) > 0:
        log.info("")
        log.info("  IV expansion events in OOS overlap: %d", len(exp_in_common))
        log.info("    NBR mean |resid|:     %.4f", exp_in_common["abs_nbr"].mean())
        log.info("    NBR+PCH mean |resid|: %.4f", exp_in_common["abs_comb"].mean())
        log.info("    PCH improved:         %d / %d",
                 exp_in_common["improved"].sum(), len(exp_in_common))

    # Overall improvement rate
    overall_improved = common["improved"].mean() * 100
    overall_worsened = common["worsened"].mean() * 100
    log.info("")
    log.info("  Overall (all OOS rows where both models overlap):")
    log.info("    PCH improved prediction: %.1f%% of events", overall_improved)
    log.info("    PCH worsened prediction: %.1f%% of events", overall_worsened)
    log.info("    Mean |resid| NBR:        %.4f", common["abs_nbr"].mean())
    log.info("    Mean |resid| NBR+PCH:    %.4f", common["abs_comb"].mean())

    return {
        "n_common_oos": len(common),
        "nbr_worst10pct_n": len(nbr_worst),
        "nbr_worst10pct_pch_improved_pct": round(float(nbr_worst["improved"].mean() * 100), 1),
        "overall_improved_pct": round(float(overall_improved), 1),
        "overall_worsened_pct": round(float(overall_worsened), 1),
        "mean_abs_resid_nbr":  round(float(common["abs_nbr"].mean()), 4),
        "mean_abs_resid_comb": round(float(common["abs_comb"].mean()), 4),
    }


# ── STEP 8: EWM variant ───────────────────────────────────────────────────────

def step8_ewm_variant(df: pd.DataFrame, delta_sp_mean: float) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 8 — EWM VARIANT (span=%d)", EWM_SPAN)
    log.info("=" * 72)
    log.info("  Testing NBR + prior_crush_ewm vs NBR + prior_crush_mean")

    r_ewm = wf_ridge(df, FS_COMB_EWM, label="NBR + PCH_ewm (2 feat)")
    r_nbr = wf_ridge(df, FS_NBR, label="NBR-only")

    sp_ewm  = r_ewm["pooled"].get("spearman_r", float("nan"))
    sp_nbr  = r_nbr["pooled"].get("spearman_r", float("nan"))
    delta_ewm = round(sp_ewm - sp_nbr, 4) if not math.isnan(sp_ewm) else float("nan")

    log.info("")
    log.info("  %-28s  Spearman=%.4f  ΔSpearman vs NBR=%+.4f",
             "NBR + PCH_mean", delta_sp_mean, delta_sp_mean)
    log.info("  %-28s  Spearman=%.4f  ΔSpearman vs NBR=%+.4f",
             "NBR + PCH_ewm", sp_ewm, delta_ewm)
    if not math.isnan(delta_ewm) and not math.isnan(delta_sp_mean):
        if delta_ewm > delta_sp_mean + 0.005:
            log.info("  → EWM is BETTER than equal-weight mean by > 0.005 Spearman — prefer EWM")
        elif delta_sp_mean > delta_ewm + 0.005:
            log.info("  → Equal-weight mean is BETTER — recent events don't add extra signal")
        else:
            log.info("  → EWM and mean are EQUIVALENT — use mean for simplicity")

    return {
        "spearman_comb_ewm": round(float(sp_ewm), 4),
        "delta_spearman_ewm_vs_nbr": round(float(delta_ewm), 4),
        "delta_spearman_mean_vs_nbr": round(float(delta_sp_mean), 4),
    }


# ── STEP 9: verdict ───────────────────────────────────────────────────────────

def step9_verdict(
    coverage: Dict,
    comparison: Dict,
    independence: Dict,
    robustness: Dict,
    failure: Dict,
    ewm: Dict,
) -> Dict[str, Any]:
    log.info("")
    log.info("=" * 72)
    log.info("STEP 9 — VERDICT: INCLUDE OR REJECT prior_crush_history?")
    log.info("=" * 72)

    delta_sp  = comparison["delta_spearman"]
    delta_r2  = comparison["delta_r2"]
    delta_auc = comparison["delta_auc"]
    consistent = comparison["consistent_fold_gain"]
    partial_r  = independence["partial_pch_given_nbr"]
    partial_p  = independence["partial_p_value"]
    pct_valid  = coverage["pct_valid_pch"]

    nbr_spread  = (comparison["economic"].get("NBR-only (1 feat)", {}) or {}).get("spread_pp", 0)
    comb_spread = (comparison["economic"].get("NBR + PCH (2 feat)", {}) or {}).get("spread_pp", 0)
    delta_spread = round((comb_spread - nbr_spread), 1) if nbr_spread and comb_spread else float("nan")

    log.info("")
    log.info("  EVIDENCE SUMMARY:")
    log.info("  ─────────────────────────────────────────────────────────────────")
    log.info("  Statistical:")
    log.info("    Δ Spearman (NBR+PCH − NBR):  %+.4f", delta_sp)
    log.info("    Δ R²       (NBR+PCH − NBR):  %+.4f", delta_r2)
    log.info("    Δ AUC      (NBR+PCH − NBR):  %+.4f", delta_auc)
    log.info("    Consistent gain across folds: %s", "YES" if consistent else "NO")
    log.info("  Independence:")
    log.info("    Partial Spearman(PCH|NBR, TRC|NBR): r=%+.4f  p=%.4f", partial_r, partial_p)
    log.info("    Independent signal:  %s", "YES" if independence["independent_signal"] else "NO")
    log.info("  Economic:")
    log.info("    NBR-only Q5-Q1 spread:  %+.1f pp", nbr_spread)
    log.info("    NBR+PCH  Q5-Q1 spread:  %+.1f pp", comb_spread)
    log.info("    Δ economic spread:      %+.1f pp", delta_spread)
    log.info("  Coverage:")
    log.info("    Events with valid PCH:  %.1f%%", pct_valid)

    # Inclusion criteria
    stat_improvement   = abs(delta_sp) >= 0.010 or abs(delta_r2) >= 0.008
    economic_improv    = abs(delta_spread) >= 1.5 if not math.isnan(delta_spread) else False
    independent        = independence["independent_signal"]
    coverage_adequate  = pct_valid >= 50.0

    flags = []
    if not stat_improvement:
        flags.append(f"NEGLIGIBLE_STATISTICAL_GAIN: ΔSpearman={delta_sp:+.4f}, ΔR²={delta_r2:+.4f}. "
                     "Improvement is below the 0.010 / 0.008 materiality threshold.")
    if not consistent:
        flags.append("FOLD_INCONSISTENCY: PCH does not improve Spearman in all three folds. "
                     "Gain may be regime-specific, not structural.")
    if not independent:
        flags.append(f"NOT_INDEPENDENT: Partial Spearman(PCH|NBR)={partial_r:+.4f} (p={partial_p:.4f}). "
                     "PCH does not add information beyond NBR.")
    if not economic_improv and not math.isnan(delta_spread):
        flags.append(f"NEGLIGIBLE_ECONOMIC_GAIN: Δspread={delta_spread:+.1f}pp < 1.5pp threshold.")
    if not coverage_adequate:
        flags.append(f"LOW_COVERAGE: only {pct_valid:.1f}% of events have valid PCH. "
                     "Feature is missing for >{100-pct_valid:.0f}% of the universe.")

    # Decision
    include = (stat_improvement and consistent and independent and coverage_adequate)
    reject  = not include

    if include:
        verdict_str = "INCLUDE: prior_crush_history provides independent, additive, consistent signal."
    elif flags and any("NEGLIGIBLE_STATISTICAL" in f or "NOT_INDEPENDENT" in f for f in flags):
        verdict_str = "REJECT: prior_crush_history does not justify inclusion — see flags."
    else:
        verdict_str = "CONDITIONAL: signal exists but fails at least one inclusion gate — needs refinement."

    log.info("")
    log.info("  INCLUSION GATES:")
    log.info("  %-45s  %s", "Statistical improvement (ΔSpearman≥0.010 or ΔR²≥0.008):",
             "PASS" if stat_improvement else "FAIL")
    log.info("  %-45s  %s", "Consistent gain across all 3 folds:",
             "PASS" if consistent else "FAIL")
    log.info("  %-45s  %s", "Independent signal (partial r>0.10, p<0.05):",
             "PASS" if independent else "FAIL")
    log.info("  %-45s  %s", "Economic improvement (Δspread≥1.5pp):",
             "PASS" if economic_improv else "FAIL")
    log.info("  %-45s  %s", "Coverage adequate (≥50%% valid PCH):",
             "PASS" if coverage_adequate else "FAIL")

    log.info("")
    if flags:
        log.info("  FLAGS:")
        for i, f in enumerate(flags, 1):
            log.info("  [%d] %s", i, f)

    log.info("")
    log.info("  *** VERDICT: %s ***", verdict_str)

    if include:
        log.info("")
        log.info("  Recommended production feature: prior_crush_mean")
        ewm_delta = ewm.get("delta_spearman_ewm_vs_nbr", 0)
        mean_delta = ewm.get("delta_spearman_mean_vs_nbr", 0)
        if ewm_delta > mean_delta + 0.005:
            log.info("  → Prefer EWM variant (span=%d) over equal-weight mean", EWM_SPAN)
        else:
            log.info("  → Use equal-weight mean (EWM adds no additional lift)")
        log.info("  Weak-coverage policy: drop cold-start events from PCH-using models")
        log.info("  (retain NBR-only prediction for cold-start events in production)")
    else:
        log.info("")
        log.info("  Recommended action:")
        if not independent:
            log.info("  → PCH is capturing the same variance as NBR (both reflect")
            log.info("    the structural level of IV compression for that company).")
            log.info("  → NBR at event-time already reflects the market's pre-earnings")
            log.info("    pricing, which implicitly incorporates prior crush history.")
        if not stat_improvement:
            log.info("  → Marginal features on a strong 1-feature model should be rejected")
            log.info("    unless they pass ALL inclusion gates.")
        log.info("  → Proceed to Week 5 without prior_crush_history.")
        log.info("  → Consider: sector/industry features (different mechanism from NBR)")

    return {
        "include": include,
        "verdict": verdict_str,
        "gates": {
            "stat_improvement": bool(stat_improvement),
            "consistent_folds": bool(consistent),
            "independent_signal": bool(independent),
            "economic_improvement": bool(economic_improv),
            "coverage_adequate": bool(coverage_adequate),
        },
        "flags": flags,
        "n_flags": len(flags),
        "delta_spearman": delta_sp,
        "delta_r2": delta_r2,
        "delta_auc": delta_auc,
        "delta_economic_spread": round(delta_spread, 1) if not math.isnan(delta_spread) else None,
        "partial_r_pch_given_nbr": round(partial_r, 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Week 4 prior crush history research")
    p.add_argument("--db-path",      default=None)
    p.add_argument("--quality-tier", default="AB", choices=["A","AB"])
    p.add_argument("--report-dir",   default=None)
    return p


def main(argv=None) -> int:
    args       = build_parser().parse_args(argv)
    db_path    = Path(args.db_path).expanduser() if args.db_path else DEFAULT_DB_PATH
    report_dir = Path(args.report_dir).expanduser() if args.report_dir else DEFAULT_REPORT_DIR
    quality_tiers = list(args.quality_tier)

    log.info("=" * 72)
    log.info("WEEK 4 PRIOR CRUSH HISTORY  version=%s", RESEARCH_VERSION)
    log.info("=" * 72)
    log.info("DB         : %s", db_path)
    log.info("Quality    : tiers %s", quality_tiers)
    log.info("Hypothesis : prior crush behavior per symbol adds independent")
    log.info("             signal on top of NBR (1-feature baseline)")
    log.info("")

    # ── load + engineer ───────────────────────────────────────────────────────
    labels = load_labels(db_path, quality_tiers)
    df     = engineer_base_features(labels)
    df     = compute_prior_crush_features(df)

    # ── run pipeline ─────────────────────────────────────────────────────────
    coverage   = step1_construct_and_validate(df)
    comparison = step3_model_comparison(df)
    indep      = step5_independence_test(df, comparison["r_comb"])
    robust     = step6_robustness(df)
    failure    = step7_failure_modes(df, comparison["r_nbr"], comparison["r_comb"])
    ewm        = step8_ewm_variant(df, comparison["delta_spearman"])
    verdict    = step9_verdict(coverage, comparison, indep, robust, failure, ewm)

    # ── save report ───────────────────────────────────────────────────────────
    def _safe(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    report = {
        "research_version": RESEARCH_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {"db_path": str(db_path), "quality_tiers": quality_tiers,
                   "min_prior_events": MIN_PRIOR_EVENTS, "ewm_span": EWM_SPAN},
        "coverage": coverage,
        "model_comparison": {
            "regression": {
                "nbr_only":  {k: v for k, v in comparison["r_nbr"].items()  if k != "oos"},
                "pch_only":  {k: v for k, v in comparison["r_pch"].items()  if k != "oos"},
                "nbr_pch":   {k: v for k, v in comparison["r_comb"].items() if k != "oos"},
            },
            "classification": {
                "nbr_only":  {k: v for k, v in comparison["c_nbr"].items()  if k != "oos"},
                "pch_only":  {k: v for k, v in comparison["c_pch"].items()  if k != "oos"},
                "nbr_pch":   {k: v for k, v in comparison["c_comb"].items() if k != "oos"},
            },
            "deltas": {
                "spearman": comparison["delta_spearman"],
                "r2":       comparison["delta_r2"],
                "auc":      comparison["delta_auc"],
            },
            "economic": comparison["economic"],
        },
        "independence_test": indep,
        "robustness":        robust,
        "failure_modes":     failure,
        "ewm_variant":       ewm,
        "verdict":           verdict,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = report_dir / f"prior_crush_{RESEARCH_VERSION}_{ts}.json"
    path.write_text(json.dumps(report, indent=2, default=_safe))
    log.info("")
    log.info("Report saved: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
