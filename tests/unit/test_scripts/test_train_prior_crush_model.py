"""Unit tests for scripts/train_prior_crush_model.py.

Leakage tests are the highest priority — they verify the core
invariant: prior_crush_mean[i] uses only events before event_date[i].

Coverage:
  - compute_prior_crush_features  leakage, correctness, cold-start, EWM
  - step1_construct_and_validate  leakage checks pass, coverage report
  - step3_model_comparison        3-model structure, delta metrics
  - step5_independence_test       partial correlation structure
  - step6_robustness              history-depth buckets
  - step7_failure_modes           correction analysis
  - step9_verdict                 inclusion gate logic
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from scripts.train_prior_crush_model import (
    EWM_SPAN,
    FS_COMB,
    FS_NBR,
    FS_PCH,
    MIN_PRIOR_EVENTS,
    _pool,
    compute_prior_crush_features,
    engineer_base_features,
    quintile_spread,
    step1_construct_and_validate,
    step3_model_comparison,
    step5_independence_test,
    step6_robustness,
    step7_failure_modes,
    step9_verdict,
    wf_lr,
    wf_ridge,
)


# ── synthetic data ────────────────────────────────────────────────────────────

def _make_df(n: int = 200, n_symbols: int = 10,
             signal_strength: float = -0.15, seed: int = 0) -> pd.DataFrame:
    """Synthetic labeled earnings events with injected signal.

    Symbols repeat across years so PCH is non-trivial.
    signal_strength controls correlation between NBR and TRC.
    """
    rng   = np.random.default_rng(seed)
    syms  = [f"SYM{i:02d}" for i in range(n_symbols)]
    bdays = pd.date_range("2023-01-03", "2026-12-31", freq="B")
    idx   = rng.integers(1, len(bdays), size=n)
    dates = pd.DatetimeIndex(bdays[idx])

    front_iv = rng.uniform(0.30, 1.20, n)
    back_iv  = rng.uniform(0.20, 0.80, n)
    nbr      = front_iv / back_iv
    noise    = rng.normal(0, 0.04, n)
    trc      = signal_strength * nbr + 0.10 + noise
    fc       = -0.40 * nbr + 0.20 + rng.normal(0, 0.03, n)
    bc       = fc * rng.uniform(0.3, 0.7, n)

    df = pd.DataFrame({
        "symbol":            [syms[i % n_symbols] for i in range(n)],
        "event_date":        list(dates),
        "year":              dates.year,
        "quarter":           dates.quarter,
        "pre_front_iv":      front_iv,
        "post_front_iv":     front_iv * (1 + fc),
        "pre_back_iv":       back_iv,
        "post_back_iv":      back_iv  * (1 + bc),
        "front_iv_crush_pct": fc,
        "back_iv_crush_pct":  bc,
        "term_ratio_change":  trc,
        "spread_differential": fc - bc,
        "underlying_move_pct": rng.uniform(-0.10, 0.10, n),
        "near_back_ratio":    nbr,
        "log_front_iv":       np.log(front_iv),
        "log_back_iv":        np.log(back_iv),
        "crush_lt_40pct":     (fc < -0.40).astype(int),
        "quality_score":      rng.uniform(0.7, 1.0, n),
        "quality_tier":       "A",
        "front_dte_pre":      rng.integers(3, 10, size=n).astype(float),
        "back_dte_pre":       rng.integers(25, 50, size=n).astype(float),
    })
    return df


def _make_full_df(n: int = 300, n_symbols: int = 12, seed: int = 0) -> pd.DataFrame:
    df = _make_df(n, n_symbols=n_symbols, seed=seed)
    return compute_prior_crush_features(df)


# ── compute_prior_crush_features — leakage tests (highest priority) ───────────

class TestComputePriorCrushLeakage:

    def test_cold_start_rows_have_nan(self):
        """First occurrence of every symbol must have NaN prior_crush_mean."""
        df = compute_prior_crush_features(_make_df(200))
        cold = df[df["prior_event_count"] == 0]
        assert cold["prior_crush_mean"].isna().all(), \
            "Cold-start rows must have NaN — any non-NaN indicates leakage"

    def test_single_event_symbol_is_always_cold_start(self):
        """A symbol that appears exactly once must always be cold-start."""
        base = _make_df(100)
        # Add a unique symbol with a single event
        unique_row = base.iloc[[0]].copy()
        unique_row["symbol"] = "UNIQUE_XYZ"
        df = pd.concat([base, unique_row], ignore_index=True)
        df = compute_prior_crush_features(df)
        row = df[df["symbol"] == "UNIQUE_XYZ"]
        assert row["prior_crush_mean"].isna().all(), \
            "Single-event symbol must have NaN prior_crush_mean"

    def test_second_event_uses_only_first(self):
        """For a symbol with 2 events, the second's PCH must equal the first's crush."""
        rng = np.random.default_rng(42)
        crush1, crush2 = -0.35, -0.50
        df = pd.DataFrame({
            "symbol":             ["AAA", "AAA"],
            "event_date":         pd.to_datetime(["2024-01-10", "2024-04-10"]),
            "year":               [2024, 2024],
            "quarter":            [1, 2],
            "front_iv_crush_pct": [crush1, crush2],
            "term_ratio_change":  [-0.10, -0.20],
            "back_iv_crush_pct":  [-0.15, -0.20],
            "spread_differential": [0.05, 0.10],
            "pre_front_iv":       [0.50, 0.60],
            "pre_back_iv":        [0.30, 0.35],
            "underlying_move_pct": [0.01, -0.02],
        })
        out = compute_prior_crush_features(df)
        out = out.sort_values("event_date").reset_index(drop=True)
        assert math.isnan(out.loc[0, "prior_crush_mean"]), \
            "First event must have NaN PCH"
        assert abs(out.loc[1, "prior_crush_mean"] - crush1) < 1e-9, \
            f"Second event PCH={out.loc[1,'prior_crush_mean']:.6f} != first_crush={crush1}"

    def test_third_event_is_mean_of_first_two(self):
        """Third event's PCH must equal mean(crush1, crush2)."""
        crushes = [-0.30, -0.55, -0.40]
        df = pd.DataFrame({
            "symbol":             ["BBB"] * 3,
            "event_date":         pd.to_datetime(["2023-03-01", "2023-06-01", "2023-09-01"]),
            "year":               [2023, 2023, 2023],
            "quarter":            [1, 2, 3],
            "front_iv_crush_pct": crushes,
            "term_ratio_change":  [-0.10, -0.20, -0.15],
            "back_iv_crush_pct":  [-0.10, -0.20, -0.15],
            "spread_differential": [0.0, 0.0, 0.0],
            "pre_front_iv":       [0.5, 0.6, 0.55],
            "pre_back_iv":        [0.3, 0.35, 0.32],
            "underlying_move_pct": [0.0, 0.0, 0.0],
        })
        out = compute_prior_crush_features(df).sort_values("event_date").reset_index(drop=True)
        expected_pch2 = np.mean(crushes[:2])
        assert abs(out.loc[2, "prior_crush_mean"] - expected_pch2) < 1e-9, \
            f"Third event PCH should be mean of first two: {expected_pch2:.6f}"

    def test_no_future_data_used(self):
        """Verify: for every row, PCH equals mean of all EARLIER rows for that symbol."""
        df = compute_prior_crush_features(_make_df(150, n_symbols=5, seed=7))
        for sym in df["symbol"].unique():
            # Match the script's tiebreaker sort so positional comparison is consistent
            sym_df = df[df["symbol"] == sym].sort_values(
                ["event_date", "pre_front_iv"], na_position="last"
            ).reset_index(drop=True)
            for i in range(len(sym_df)):
                if i == 0:
                    assert math.isnan(sym_df.loc[i, "prior_crush_mean"])
                    continue
                expected = sym_df.loc[:i-1, "front_iv_crush_pct"].mean()
                actual   = sym_df.loc[i, "prior_crush_mean"]
                assert abs(expected - actual) < 1e-9, \
                    f"{sym} row {i}: expected={expected:.6f} actual={actual:.6f}"

    def test_prior_event_count_is_correct(self):
        """prior_event_count must be a permutation of 0..N-1 per symbol."""
        df = compute_prior_crush_features(_make_df(100, n_symbols=5))
        for sym in df["symbol"].unique():
            sym_df = df[df["symbol"] == sym]
            n = len(sym_df)
            counts = set(sym_df["prior_event_count"].tolist())
            assert counts == set(range(n)), \
                f"{sym}: prior_event_count is not a permutation of 0..{n-1}: {sorted(counts)}"

    def test_sort_invariance(self):
        """Result must be the same regardless of input sort order."""
        base = _make_df(100, n_symbols=5, seed=1)
        shuffled = base.sample(frac=1, random_state=99).reset_index(drop=True)
        out1 = compute_prior_crush_features(base).sort_values(
            ["symbol", "event_date"]).reset_index(drop=True)
        out2 = compute_prior_crush_features(shuffled).sort_values(
            ["symbol", "event_date"]).reset_index(drop=True)
        np.testing.assert_allclose(
            out1["prior_crush_mean"].fillna(999).values,
            out2["prior_crush_mean"].fillna(999).values,
            rtol=1e-9, err_msg="PCH must be sort-invariant",
        )

    def test_multiple_symbols_independent(self):
        """Events for symbol A must not influence symbol B's PCH."""
        df = pd.DataFrame({
            "symbol":             ["AAA", "BBB", "AAA", "BBB"],
            "event_date":         pd.to_datetime(["2023-01-01", "2023-01-01",
                                                   "2023-04-01", "2023-04-01"]),
            "year":               [2023, 2023, 2023, 2023],
            "quarter":            [1, 1, 2, 2],
            "front_iv_crush_pct": [-0.40, -0.20, -0.50, -0.30],
            "term_ratio_change":  [-0.10, -0.05, -0.15, -0.08],
            "back_iv_crush_pct":  [-0.10, -0.08, -0.12, -0.10],
            "spread_differential": [0.0, 0.0, 0.0, 0.0],
            "pre_front_iv":       [0.5, 0.4, 0.6, 0.45],
            "pre_back_iv":        [0.3, 0.25, 0.35, 0.28],
            "underlying_move_pct": [0.0, 0.0, 0.0, 0.0],
        })
        out = compute_prior_crush_features(df).sort_values(
            ["symbol", "event_date"]).reset_index(drop=True)
        # Second event of AAA should use only AAA's first event
        aaa2 = out[(out["symbol"] == "AAA") & (out["event_date"] == pd.Timestamp("2023-04-01"))]
        assert abs(aaa2["prior_crush_mean"].values[0] - (-0.40)) < 1e-9
        # Second event of BBB should use only BBB's first event
        bbb2 = out[(out["symbol"] == "BBB") & (out["event_date"] == pd.Timestamp("2023-04-01"))]
        assert abs(bbb2["prior_crush_mean"].values[0] - (-0.20)) < 1e-9


# ── compute_prior_crush_features — output structure ───────────────────────────

class TestComputePriorCrushStructure:

    def test_output_columns_present(self):
        df  = compute_prior_crush_features(_make_df(100))
        for col in ("prior_crush_mean", "prior_trc_mean", "prior_crush_ewm",
                    "prior_crush_std", "prior_event_count"):
            assert col in df.columns, f"Missing column: {col}"

    def test_output_length_unchanged(self):
        base = _make_df(100)
        out  = compute_prior_crush_features(base)
        assert len(out) == len(base)

    def test_ewm_is_nan_for_cold_start(self):
        df = compute_prior_crush_features(_make_df(150))
        cold = df[df["prior_event_count"] == 0]
        assert cold["prior_crush_ewm"].isna().all()

    def test_std_is_nan_for_first_two_events(self):
        """std needs at least 2 data points to be non-NaN (after shift, need ≥3 events)."""
        df = pd.DataFrame({
            "symbol":             ["S"] * 3,
            "event_date":         pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01"]),
            "year":               [2023] * 3, "quarter": [1, 2, 3],
            "front_iv_crush_pct": [-0.40, -0.50, -0.35],
            "term_ratio_change":  [-0.10, -0.15, -0.12],
            "back_iv_crush_pct":  [-0.10, -0.12, -0.11],
            "spread_differential": [0.0] * 3,
            "pre_front_iv": [0.5] * 3, "pre_back_iv": [0.3] * 3,
            "underlying_move_pct": [0.0] * 3,
        })
        out = compute_prior_crush_features(df).sort_values("event_date").reset_index(drop=True)
        assert math.isnan(out.loc[0, "prior_crush_std"])  # 0 prior
        assert math.isnan(out.loc[1, "prior_crush_std"])  # 1 prior — std needs ≥2

    def test_prior_crush_mean_monotone_coverage(self):
        """As prior_event_count increases, prior_crush_mean must not be NaN."""
        df = compute_prior_crush_features(_make_df(200, n_symbols=5))
        has_pch = df[df["prior_event_count"] >= 1]
        assert has_pch["prior_crush_mean"].notna().all(), \
            "All rows with ≥1 prior event must have valid prior_crush_mean"

    def test_ewm_and_mean_close_for_stable_symbol(self):
        """For a symbol with consistent crush, EWM and mean should be close."""
        n = 8
        crushes = [-0.40] * n   # perfectly constant
        df = pd.DataFrame({
            "symbol":             ["STABLE"] * n,
            "event_date":         pd.date_range("2023-01-01", periods=n, freq="QS"),
            "year":               [2023, 2023, 2024, 2024, 2025, 2025, 2026, 2026],
            "quarter":            list(range(1, 5)) * 2,
            "front_iv_crush_pct": crushes,
            "term_ratio_change":  [-0.10] * n,
            "back_iv_crush_pct":  [-0.10] * n,
            "spread_differential": [0.0] * n,
            "pre_front_iv": [0.5] * n, "pre_back_iv": [0.3] * n,
            "underlying_move_pct": [0.0] * n,
        })
        out = compute_prior_crush_features(df)
        valid = out.dropna(subset=["prior_crush_mean", "prior_crush_ewm"])
        if len(valid) > 0:
            diff = (valid["prior_crush_mean"] - valid["prior_crush_ewm"]).abs()
            assert diff.max() < 0.01, "EWM and mean should be near-identical for constant crush"


# ── step1_construct_and_validate ──────────────────────────────────────────────

class TestStep1:
    def test_leakage_checks_pass(self):
        df  = _make_full_df(200)
        cov = step1_construct_and_validate(df)
        assert cov["leakage_verified"] is True

    def test_returns_coverage_metrics(self):
        df  = _make_full_df(200)
        cov = step1_construct_and_validate(df)
        for k in ("n_total", "n_valid_pch", "pct_valid_pch",
                  "n_cold_start", "pct_cold_start"):
            assert k in cov

    def test_pct_valid_plus_pct_cold_le_100(self):
        df  = _make_full_df(200)
        cov = step1_construct_and_validate(df)
        # Cold-start events are a subset of all events; cold + valid = total
        assert cov["n_cold_start"] + cov["n_valid_pch"] == cov["n_total"]

    def test_coverage_increases_with_more_events(self):
        """More events per symbol → higher % valid PCH."""
        df_small = compute_prior_crush_features(_make_df(60, n_symbols=30))
        df_large = compute_prior_crush_features(_make_df(300, n_symbols=10))
        cov_s = step1_construct_and_validate(df_small)
        cov_l = step1_construct_and_validate(df_large)
        # Large dataset has more history per symbol → more valid PCH
        assert cov_l["pct_valid_pch"] >= cov_s["pct_valid_pch"]

    def test_n_total_matches_input(self):
        df  = _make_full_df(180)
        cov = step1_construct_and_validate(df)
        assert cov["n_total"] == 180

    def test_prior_crush_mean_stats_finite(self):
        df  = _make_full_df(200)
        cov = step1_construct_and_validate(df)
        stats = cov["prior_crush_mean_stats"]
        for k in ("mean", "std", "min", "p50", "max"):
            assert math.isfinite(stats[k])

    def test_raises_on_injected_leakage(self):
        """If we inject leakage (assign non-NaN to cold-start row), step1 must raise."""
        df = _make_full_df(100)
        # Corrupt: set a cold-start row's prior_crush_mean to a non-NaN value
        cold_idx = df[df["prior_event_count"] == 0].index[0]
        df.loc[cold_idx, "prior_crush_mean"] = -0.40
        with pytest.raises(AssertionError, match="Leakage"):
            step1_construct_and_validate(df)


# ── step3_model_comparison ────────────────────────────────────────────────────

class TestStep3:
    def setup_method(self):
        self.df = _make_full_df(300, n_symbols=12, seed=3)

    def test_returns_three_regression_results(self):
        res = step3_model_comparison(self.df)
        assert all(k in res for k in ("r_nbr", "r_pch", "r_comb"))

    def test_returns_three_clf_results(self):
        res = step3_model_comparison(self.df)
        assert all(k in res for k in ("c_nbr", "c_pch", "c_comb"))

    def test_nbr_only_has_three_folds(self):
        res = step3_model_comparison(self.df)
        assert len(res["r_nbr"]["folds"]) == 3

    def test_delta_metrics_present(self):
        res = step3_model_comparison(self.df)
        for k in ("delta_spearman", "delta_r2", "delta_auc",
                  "fold_deltas", "consistent_fold_gain"):
            assert k in res

    def test_consistent_fold_gain_is_boolean(self):
        res = step3_model_comparison(self.df)
        assert isinstance(res["consistent_fold_gain"], bool)

    def test_delta_spearman_finite(self):
        res = step3_model_comparison(self.df)
        assert math.isfinite(res["delta_spearman"])

    def test_nbr_only_spearman_positive(self):
        """NBR correlates with TRC by design — NBR-only must beat zero."""
        res = step3_model_comparison(self.df)
        assert res["r_nbr"]["pooled"]["spearman_r"] > 0

    def test_pch_uses_fewer_rows_than_nbr(self):
        """PCH models drop cold-start rows, so they must have ≤ NBR's N."""
        res = step3_model_comparison(self.df)
        nbr_n   = res["r_nbr"]["pooled"].get("n", 0)
        pch_n   = res["r_pch"]["pooled"].get("n", 0)
        comb_n  = res["r_comb"]["pooled"].get("n", 0)
        assert pch_n  <= nbr_n, "PCH model must have ≤ rows than NBR-only"
        assert comb_n <= nbr_n, "Combined model must have ≤ rows than NBR-only"

    def test_economic_keys_present(self):
        res = step3_model_comparison(self.df)
        for k in ("NBR-only (1 feat)", "PCH-only (1 feat)", "NBR + PCH (2 feat)"):
            assert k in res["economic"]


# ── step5_independence_test ───────────────────────────────────────────────────

class TestStep5:
    def setup_method(self):
        self.df = _make_full_df(300, n_symbols=12, seed=5)
        res = step3_model_comparison(self.df)
        self.r_comb = res["r_comb"]

    def test_returns_required_keys(self):
        r = step5_independence_test(self.df, self.r_comb)
        for k in ("spearman_nbr_pch", "spearman_pch_trc",
                  "partial_pch_given_nbr", "partial_p_value", "independent_signal"):
            assert k in r

    def test_partial_correlation_finite(self):
        r = step5_independence_test(self.df, self.r_comb)
        assert math.isfinite(r["partial_pch_given_nbr"])

    def test_partial_p_value_in_range(self):
        r = step5_independence_test(self.df, self.r_comb)
        assert 0.0 <= r["partial_p_value"] <= 1.0

    def test_independent_signal_is_boolean(self):
        r = step5_independence_test(self.df, self.r_comb)
        assert isinstance(r["independent_signal"], bool)

    def test_spearman_nbr_pch_finite(self):
        r = step5_independence_test(self.df, self.r_comb)
        assert math.isfinite(r["spearman_nbr_pch"])


# ── step6_robustness ──────────────────────────────────────────────────────────

class TestStep6:
    def test_returns_by_history_depth(self):
        df = _make_full_df(300, n_symbols=8)
        r  = step6_robustness(df)
        assert "by_history_depth" in r
        assert isinstance(r["by_history_depth"], list)

    def test_each_bucket_has_required_fields(self):
        df = _make_full_df(300, n_symbols=8)
        r  = step6_robustness(df)
        for bucket in r["by_history_depth"]:
            assert "bucket" in bucket and "n" in bucket and "spearman_nbr" in bucket

    def test_n_values_sum_to_at_most_total(self):
        df = _make_full_df(300, n_symbols=8)
        r  = step6_robustness(df)
        total_buckets = sum(b["n"] for b in r["by_history_depth"])
        assert total_buckets <= len(df)


# ── step7_failure_modes ───────────────────────────────────────────────────────

class TestStep7:
    def setup_method(self):
        self.df = _make_full_df(300, n_symbols=12, seed=7)
        res = step3_model_comparison(self.df)
        self.r_nbr  = res["r_nbr"]
        self.r_comb = res["r_comb"]

    def test_returns_required_keys(self):
        r = step7_failure_modes(self.df, self.r_nbr, self.r_comb)
        for k in ("n_common_oos", "overall_improved_pct", "overall_worsened_pct",
                  "mean_abs_resid_nbr", "mean_abs_resid_comb"):
            assert k in r

    def test_improved_plus_worsened_le_100(self):
        r = step7_failure_modes(self.df, self.r_nbr, self.r_comb)
        assert r["overall_improved_pct"] + r["overall_worsened_pct"] <= 100.0 + 1e-9

    def test_mean_abs_resid_positive(self):
        r = step7_failure_modes(self.df, self.r_nbr, self.r_comb)
        assert r["mean_abs_resid_nbr"] > 0
        assert r["mean_abs_resid_comb"] > 0

    def test_n_common_oos_positive(self):
        r = step7_failure_modes(self.df, self.r_nbr, self.r_comb)
        assert r["n_common_oos"] > 0

    def test_empty_oos_returns_empty(self):
        r_empty = {"oos": [], "folds": [], "pooled": {}, "label": "x", "features": []}
        r = step7_failure_modes(self.df, r_empty, r_empty)
        assert r == {}


# ── step9_verdict ─────────────────────────────────────────────────────────────

class TestStep9:
    def _run_verdict(self, df=None, delta_sp=0.005, consistent=False,
                     partial_r=0.05, partial_p=0.30, pct_valid=80.0,
                     delta_spread=-1.0):
        if df is None:
            df = _make_full_df(300)
        res = step3_model_comparison(df)

        # Override deltas for parametric testing
        res["delta_spearman"]     = delta_sp
        res["delta_r2"]           = delta_sp * 0.5
        res["delta_auc"]          = delta_sp * 0.3
        res["consistent_fold_gain"] = bool(consistent)
        res["economic"]["NBR-only (1 feat)"]  = {"spread_pp": -24.0, "table": []}
        res["economic"]["PCH-only (1 feat)"]  = {"spread_pp": -20.0, "table": []}
        res["economic"]["NBR + PCH (2 feat)"] = {
            "spread_pp": -24.0 + delta_spread, "table": []
        }

        coverage = {
            "n_total": 300, "n_valid_pch": int(300 * pct_valid / 100),
            "pct_valid_pch": pct_valid, "n_cold_start": int(300 * (1 - pct_valid / 100)),
            "pct_cold_start": 100.0 - pct_valid,
            "prior_crush_mean_stats": {
                "mean": -0.40, "std": 0.10, "min": -0.70,
                "p25": -0.50, "p50": -0.40, "p75": -0.30, "max": -0.10
            },
            "leakage_verified": True,
        }
        independence = {
            "spearman_nbr_pch": 0.2,
            "spearman_nbr_trc": -0.77,
            "spearman_pch_trc": -0.35,
            "partial_pch_given_nbr": partial_r,
            "partial_p_value": partial_p,
            "independent_signal": bool(abs(partial_r) > 0.10 and partial_p < 0.05),
        }
        robustness = {"by_history_depth": []}
        failure    = {"n_common_oos": 100, "overall_improved_pct": 55.0,
                      "overall_worsened_pct": 40.0,
                      "mean_abs_resid_nbr": 0.10, "mean_abs_resid_comb": 0.09,
                      "nbr_worst10pct_n": 20, "nbr_worst10pct_pch_improved_pct": 60.0}
        ewm        = {"delta_spearman_ewm_vs_nbr": delta_sp,
                      "delta_spearman_mean_vs_nbr": delta_sp,
                      "spearman_comb_ewm": 0.77}

        return step9_verdict(coverage, res, independence, robustness, failure, ewm)

    def test_returns_required_keys(self):
        v = self._run_verdict()
        for k in ("include", "verdict", "gates", "flags", "n_flags"):
            assert k in v

    def test_include_is_boolean(self):
        v = self._run_verdict()
        assert isinstance(v["include"], bool)

    def test_verdict_is_string(self):
        v = self._run_verdict()
        assert isinstance(v["verdict"], str)

    def test_all_gates_pass_gives_include(self):
        v = self._run_verdict(
            delta_sp=0.025, consistent=True,
            partial_r=0.18, partial_p=0.001,
            pct_valid=80.0, delta_spread=-3.0,
        )
        assert v["include"] is True
        assert "INCLUDE" in v["verdict"]

    def test_no_independent_signal_gives_reject(self):
        v = self._run_verdict(
            delta_sp=0.020, consistent=True,
            partial_r=0.03, partial_p=0.60,   # not independent
            pct_valid=80.0, delta_spread=-2.0,
        )
        assert v["include"] is False

    def test_inconsistent_folds_gives_reject(self):
        v = self._run_verdict(
            delta_sp=0.020, consistent=False,  # inconsistent folds
            partial_r=0.15, partial_p=0.01,
            pct_valid=80.0, delta_spread=-2.0,
        )
        assert v["include"] is False

    def test_negligible_stat_improvement_gives_reject(self):
        v = self._run_verdict(
            delta_sp=0.003, consistent=True,   # below 0.010 threshold
            partial_r=0.15, partial_p=0.01,
            pct_valid=80.0, delta_spread=-0.5,
        )
        assert v["include"] is False

    def test_low_coverage_gives_reject(self):
        v = self._run_verdict(
            delta_sp=0.020, consistent=True,
            partial_r=0.15, partial_p=0.01,
            pct_valid=30.0,                    # below 50% threshold
            delta_spread=-2.0,
        )
        assert v["include"] is False

    def test_n_flags_consistent_with_flags_list(self):
        v = self._run_verdict()
        assert v["n_flags"] == len(v["flags"])

    def test_gates_are_booleans(self):
        v = self._run_verdict()
        for key, val in v["gates"].items():
            assert isinstance(val, bool), f"gate '{key}' should be bool, got {type(val)}"
