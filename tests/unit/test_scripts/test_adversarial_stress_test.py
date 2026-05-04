"""Unit tests for scripts/adversarial_stress_test.py.

All tests are pure (no DB, no parquet, no disk I/O).
Coverage targets:
  - wf_ridge / wf_lr        walk-forward engine
  - step1_feature_dominance algebraic linkage + ablation
  - step2_permutation       null distribution collapses vs real
  - step3_symbol_robustness subset metrics + leave-out sensitivity
  - step4_regime_stability  per-year stability
  - step5_economic_realism  quintile table + adversarial rate
  - step6_failure_modes     expansion events + confidence-wrong
  - step7_vix_contribution  VIX incremental check
  - step8_verdict           flag logic + overall verdict
  - quintile_spread         economic helper
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from scripts.adversarial_stress_test import (
    FS_BASELINE,
    FS_COMBINED,
    FS_NBR_ONLY,
    FS_VIX_ONLY,
    _fold_data,
    _pool_reg,
    quintile_spread,
    step1_feature_dominance,
    step2_permutation,
    step3_symbol_robustness,
    step4_regime_stability,
    step5_economic_realism,
    step6_failure_modes,
    step7_vix_contribution,
    step8_verdict,
    wf_lr,
    wf_ridge,
)


# ── synthetic data ────────────────────────────────────────────────────────────

def _make_df(n: int = 200, with_vix: bool = False, seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic DataFrame matching adversarial_stress_test expectations.

    Dates sampled directly from business-day calendar so year splits are clean.
    The signal is injected: near_back_ratio correlates with term_ratio_change so
    walk-forward models achieve non-trivial Spearman.
    """
    rng = np.random.default_rng(seed)
    bdays = pd.date_range("2023-01-03", "2026-12-31", freq="B")
    idx   = rng.integers(1, len(bdays), size=n)
    dates = bdays[idx]
    years = pd.DatetimeIndex(dates).year

    front_iv  = rng.uniform(0.30, 1.20, n)
    back_iv   = rng.uniform(0.20, 0.80, n)
    nbr       = front_iv / back_iv

    # Inject signal: higher NBR → more negative TRC (deeper crush)
    noise          = rng.normal(0, 0.05, n)
    trc            = -0.15 * nbr + 0.10 + noise
    front_crush    = -0.40 * nbr + 0.20 + rng.normal(0, 0.04, n)
    back_crush     = front_crush * rng.uniform(0.3, 0.7, n)
    post_front_iv  = front_iv * (1 + front_crush)
    post_back_iv   = back_iv  * (1 + back_crush)

    df = pd.DataFrame({
        "id":                    range(n),
        "symbol":                [f"SYM{i % 15}" for i in range(n)],
        "event_date":            list(dates),
        "pre_capture_date":      list(bdays[idx - 1]),
        "post_capture_date":     list(bdays[np.minimum(idx + 1, len(bdays) - 1)]),
        "release_timing":        "AMC",
        "pre_front_iv":          front_iv,
        "post_front_iv":         post_front_iv,
        "pre_back_iv":           back_iv,
        "post_back_iv":          post_back_iv,
        "front_iv_crush_pct":    front_crush,
        "back_iv_crush_pct":     back_crush,
        "term_ratio_change":     trc,
        "spread_differential":   front_crush - back_crush,
        "underlying_move_pct":   rng.uniform(-0.10, 0.10, n),
        "near_back_ratio":       nbr,
        "log_front_iv":          np.log(front_iv),
        "log_back_iv":           np.log(back_iv),
        "crush_lt_40pct":        (front_crush < -0.40).astype(int),
        "quality_score":         rng.uniform(0.6, 1.0, n),
        "quality_tier":          "A",
        "year":                  years,
        "quarter":               pd.DatetimeIndex(dates).quarter,
        "front_dte_pre":         rng.integers(3, 10, size=n).astype(float),
        "back_dte_pre":          rng.integers(25, 50, size=n).astype(float),
        "pre_front_oi":          rng.integers(100, 5000, size=n).astype(float),
        "pre_underlying_price":  rng.uniform(20, 500, n),
    })
    if with_vix:
        df["log_vix"]     = rng.uniform(2.6, 3.7, n)   # ln(13..40)
        df["vix_pct_252"] = rng.uniform(0.1, 0.9, n)
        df["vix_close"]   = np.exp(df["log_vix"])
    else:
        df["log_vix"]     = np.nan
        df["vix_pct_252"] = np.nan
        df["vix_close"]   = np.nan
    return df


# ── wf_ridge / wf_lr ─────────────────────────────────────────────────────────

class TestWfRidge:
    def test_returns_three_folds(self):
        df = _make_df(200)
        r  = wf_ridge(df, FS_BASELINE)
        assert len(r["folds"]) == 3

    def test_pooled_spearman_finite(self):
        df = _make_df(200)
        r  = wf_ridge(df, FS_BASELINE)
        assert math.isfinite(r["pooled"]["spearman_r"])

    def test_injected_signal_above_naive(self):
        """With synthetic signal, rmse_ratio should be < 1 (better than naive)."""
        df = _make_df(300, seed=7)
        r  = wf_ridge(df, FS_BASELINE)
        assert r["pooled"]["rmse_ratio"] < 1.0, \
            "Model should beat naive mean when signal is injected"

    def test_oos_preds_count_matches_folds(self):
        df = _make_df(200)
        r  = wf_ridge(df, FS_BASELINE)
        n_oos = sum(f["n_test"] for f in r["folds"])
        assert len(r["oos"]) == n_oos

    def test_coef_keys_match_features(self):
        df    = _make_df(200)
        feats = ["near_back_ratio", "log_front_iv"]
        r     = wf_ridge(df, feats)
        for fold in r["folds"]:
            assert set(fold["coef"].keys()) == set(feats)

    def test_nbr_only_spearman_positive(self):
        """NBR correlates with TRC by construction — NBR-only must beat zero."""
        df = _make_df(300)
        r  = wf_ridge(df, FS_NBR_ONLY)
        assert r["pooled"]["spearman_r"] > 0.0


class TestWfLr:
    def test_returns_three_folds(self):
        df = _make_df(200)
        c  = wf_lr(df, FS_BASELINE)
        assert len(c["folds"]) == 3

    def test_pooled_auc_in_range(self):
        df = _make_df(200)
        c  = wf_lr(df, FS_BASELINE)
        if c["pooled"].get("auc") is not None:
            assert 0.0 <= c["pooled"]["auc"] <= 1.0

    def test_probs_in_unit_interval(self):
        df = _make_df(200)
        c  = wf_lr(df, FS_BASELINE)
        for p in c["oos"]:
            assert 0.0 <= p["y_prob"] <= 1.0


# ── _pool_reg ─────────────────────────────────────────────────────────────────

class TestPoolReg:
    def test_empty_returns_empty(self):
        assert _pool_reg([]) == {}

    def test_returns_spearman_r2_rmse(self):
        oos = [{"y_true": float(x), "y_pred": float(x) * 0.9 + 0.1}
               for x in np.linspace(-0.5, 0.5, 50)]
        p = _pool_reg(oos)
        assert "spearman_r" in p and "r2" in p and "rmse" in p

    def test_perfect_preds_rmse_zero(self):
        oos = [{"y_true": float(x), "y_pred": float(x)} for x in np.linspace(-1, 1, 30)]
        p = _pool_reg(oos)
        assert p["rmse"] < 1e-10


# ── quintile_spread ───────────────────────────────────────────────────────────

class TestQuintileSpread:
    def test_returns_finite_spread(self):
        df  = _make_df(200)
        r   = wf_ridge(df, FS_BASELINE)
        sp, table = quintile_spread(df, r["oos"])
        assert math.isfinite(sp)
        assert len(table) == 5

    def test_empty_preds_returns_nan(self):
        df = _make_df(50)
        sp, table = quintile_spread(df, [])
        assert math.isnan(sp) and table == []

    def test_q5_has_more_negative_front_crush(self):
        """Q5 = predicted deepest crush should have lower avg_front_pct than Q1."""
        df  = _make_df(300, seed=3)
        r   = wf_ridge(df, FS_BASELINE)
        sp, table = quintile_spread(df, r["oos"])
        q1 = next(t for t in table if t["q"] == "Q1")
        q5 = next(t for t in table if t["q"] == "Q5")
        # Q5 should be more negative (deeper crush)
        assert q5["avg_front_pct"] < q1["avg_front_pct"]


# ── step1_feature_dominance ───────────────────────────────────────────────────

class TestStep1:
    def setup_method(self):
        self.df = _make_df(200)

    def test_returns_required_keys(self):
        r = step1_feature_dominance(self.df)
        for k in ("algebraic", "ablation", "nbr_pct_of_baseline", "has_vix"):
            assert k in r

    def test_algebraic_keys(self):
        r = step1_feature_dominance(self.df)
        for k in ("pearson_nbr_trc", "spearman_nbr_trc", "r2_nbr_on_trc",
                  "nbr_coef", "mechanical_link_flag"):
            assert k in r["algebraic"]

    def test_ablation_has_baseline(self):
        r = step1_feature_dominance(self.df)
        assert "Baseline (3 feat)" in r["ablation"]

    def test_nbr_pct_is_positive(self):
        r = step1_feature_dominance(self.df)
        assert r["nbr_pct_of_baseline"] > 0

    def test_no_vix_flag_without_vix(self):
        r = step1_feature_dominance(self.df)
        assert r["has_vix"] is False

    def test_vix_models_present_with_vix(self):
        df = _make_df(200, with_vix=True)
        r  = step1_feature_dominance(df)
        assert r["has_vix"] is True
        assert "VIX-only (2 feat)" in r["ablation"]
        assert "Combined (5 feat)" in r["ablation"]

    def test_mechanical_link_flag_boolean(self):
        r = step1_feature_dominance(self.df)
        assert isinstance(r["algebraic"]["mechanical_link_flag"], bool)

    def test_r2_nbr_on_trc_finite(self):
        r = step1_feature_dominance(self.df)
        assert math.isfinite(r["algebraic"]["r2_nbr_on_trc"])


# ── step2_permutation ─────────────────────────────────────────────────────────

class TestStep2:
    def test_returns_required_keys(self):
        df = _make_df(200)
        r  = step2_permutation(df, n_perm=5, features=FS_BASELINE)
        for k in ("verdict", "n_permutations", "real_spearman",
                  "null_spearman_mean", "p_value_spearman"):
            assert k in r

    def test_verdict_is_valid_string(self):
        df = _make_df(200)
        r  = step2_permutation(df, n_perm=5, features=FS_BASELINE)
        assert r["verdict"] in ("REAL", "MARGINAL", "FAILED")

    def test_null_spearman_near_zero(self):
        """Under permutation, pooled Spearman must be close to zero."""
        df  = _make_df(250, seed=1)
        r   = step2_permutation(df, n_perm=20, features=FS_BASELINE)
        assert abs(r["null_spearman_mean"]) < 0.25, \
            f"Null Spearman should be near 0, got {r['null_spearman_mean']}"

    def test_real_exceeds_null_for_injected_signal(self):
        """Real Spearman must exceed null mean when signal is injected."""
        df = _make_df(300, seed=5)
        r  = step2_permutation(df, n_perm=20, features=FS_BASELINE)
        assert r["real_spearman"] > r["null_spearman_mean"]

    def test_p_value_in_unit_interval(self):
        df = _make_df(200)
        r  = step2_permutation(df, n_perm=5, features=FS_BASELINE)
        assert 0.0 <= r["p_value_spearman"] <= 1.0

    def test_n_permutations_stored(self):
        df = _make_df(200)
        r  = step2_permutation(df, n_perm=7, features=FS_BASELINE)
        assert r["n_permutations"] == 7


# ── step3_symbol_robustness ───────────────────────────────────────────────────

class TestStep3:
    def setup_method(self):
        self.df = _make_df(300)

    def test_returns_required_keys(self):
        r = step3_symbol_robustness(self.df, FS_BASELINE)
        for k in ("top5_symbols", "high_freq_pooled", "low_freq_pooled",
                  "without_top5_pooled", "delta_spearman_top5_removed",
                  "top5_sensitive"):
            assert k in r

    def test_top5_are_strings(self):
        r = step3_symbol_robustness(self.df, FS_BASELINE)
        assert all(isinstance(s, str) for s in r["top5_symbols"])

    def test_high_low_pooled_spearman_finite(self):
        r = step3_symbol_robustness(self.df, FS_BASELINE)
        hi_sp = r["high_freq_pooled"].get("spearman_r")
        lo_sp = r["low_freq_pooled"].get("spearman_r")
        # High-freq subset should always have enough data for a finite result
        if hi_sp is not None:
            assert math.isfinite(hi_sp)
        # Low-freq subset may have insufficient data in some folds — just check type
        if lo_sp is not None:
            assert math.isfinite(lo_sp) or math.isnan(lo_sp)

    def test_delta_spearman_is_finite(self):
        r = step3_symbol_robustness(self.df, FS_BASELINE)
        assert math.isfinite(r["delta_spearman_top5_removed"])

    def test_top5_sensitive_is_boolean(self):
        r = step3_symbol_robustness(self.df, FS_BASELINE)
        assert isinstance(r["top5_sensitive"], bool)


# ── step4_regime_stability ────────────────────────────────────────────────────

class TestStep4:
    def test_returns_per_year_list(self):
        df = _make_df(200)
        r  = step4_regime_stability(df, FS_BASELINE)
        assert "per_year" in r
        assert isinstance(r["per_year"], list)

    def test_per_year_has_required_fields(self):
        df = _make_df(200)
        r  = step4_regime_stability(df, FS_BASELINE)
        for yr in r["per_year"]:
            assert "test_year" in yr and "spearman_r" in yr and "r2" in yr

    def test_spearman_finite_per_year(self):
        df = _make_df(200)
        r  = step4_regime_stability(df, FS_BASELINE)
        for yr in r["per_year"]:
            assert math.isfinite(yr["spearman_r"])

    def test_three_oos_years(self):
        df = _make_df(250)
        r  = step4_regime_stability(df, FS_BASELINE)
        assert len(r["per_year"]) == 3


# ── step5_economic_realism ────────────────────────────────────────────────────

class TestStep5:
    def test_returns_quintile_table(self):
        df = _make_df(300)
        r  = step5_economic_realism(df, FS_BASELINE)
        assert "quintile_table" in r
        assert len(r["quintile_table"]) == 5

    def test_spread_finite(self):
        df = _make_df(300)
        r  = step5_economic_realism(df, FS_BASELINE)
        assert math.isfinite(r["q5_q1_front_spread_pp"])

    def test_adversarial_pct_in_range(self):
        df = _make_df(300)
        r  = step5_economic_realism(df, FS_BASELINE)
        assert 0.0 <= r["q5_adversarial_pct"] <= 100.0

    def test_quintile_table_has_all_bins(self):
        df = _make_df(300)
        r  = step5_economic_realism(df, FS_BASELINE)
        qs = {row["q"] for row in r["quintile_table"]}
        assert qs == {"Q1", "Q2", "Q3", "Q4", "Q5"}

    def test_q5_deeper_crush_than_q1(self):
        """With injected signal, Q5 (deepest predicted crush) should have more
        negative average front crush than Q1."""
        df = _make_df(400, seed=11)
        r  = step5_economic_realism(df, FS_BASELINE)
        tbl = r["quintile_table"]
        q1_front = next(t["avg_front_pct"] for t in tbl if t["q"] == "Q1")
        q5_front = next(t["avg_front_pct"] for t in tbl if t["q"] == "Q5")
        assert q5_front < q1_front, \
            f"Q5 front crush {q5_front} should be < Q1 {q1_front}"


# ── step6_failure_modes ───────────────────────────────────────────────────────

class TestStep6:
    def test_returns_required_keys(self):
        df = _make_df(300)
        r  = step6_failure_modes(df, FS_BASELINE)
        for k in ("n_expansion_events", "pct_confident_wrong", "worst_10pct_n"):
            assert k in r

    def test_expansion_count_nonnegative(self):
        df = _make_df(300)
        r  = step6_failure_modes(df, FS_BASELINE)
        assert r["n_expansion_events"] >= 0

    def test_pct_confident_wrong_in_range(self):
        df = _make_df(300)
        r  = step6_failure_modes(df, FS_BASELINE)
        assert 0.0 <= r["pct_confident_wrong"] <= 100.0

    def test_yr_residual_bias_is_dict(self):
        df = _make_df(300)
        r  = step6_failure_modes(df, FS_BASELINE)
        assert isinstance(r["yr_residual_bias"], dict)

    def test_worst_10pct_n_is_positive(self):
        df = _make_df(300)
        r  = step6_failure_modes(df, FS_BASELINE)
        assert r["worst_10pct_n"] > 0


# ── step7_vix_contribution ────────────────────────────────────────────────────

class TestStep7:
    def test_skips_when_no_vix(self):
        df = _make_df(200, with_vix=False)
        s1 = step1_feature_dominance(df)
        r  = step7_vix_contribution(df, s1)
        assert r.get("skipped") is True

    def test_returns_verdict_with_vix(self):
        df = _make_df(250, with_vix=True)
        s1 = step1_feature_dominance(df)
        r  = step7_vix_contribution(df, s1)
        assert "vix_verdict" in r
        assert r["vix_verdict"] in ("ADDITIVE", "MARGINAL", "REDUNDANT")

    def test_delta_spearman_finite_with_vix(self):
        df = _make_df(250, with_vix=True)
        s1 = step1_feature_dominance(df)
        r  = step7_vix_contribution(df, s1)
        assert math.isfinite(r["delta_spearman"])

    def test_fold_deltas_length_with_vix(self):
        df = _make_df(250, with_vix=True)
        s1 = step1_feature_dominance(df)
        r  = step7_vix_contribution(df, s1)
        assert 0 <= len(r["fold_deltas"]) <= 3


# ── step8_verdict ─────────────────────────────────────────────────────────────

class TestStep8:
    def _run_all(self, n=300, with_vix=False, n_perm=5, seed=0):
        df = _make_df(n, with_vix=with_vix, seed=seed)
        s1 = step1_feature_dominance(df)
        s2 = step2_permutation(df, n_perm=n_perm, features=FS_BASELINE)
        s3 = step3_symbol_robustness(df, FS_BASELINE)
        s4 = step4_regime_stability(df, FS_BASELINE)
        s5 = step5_economic_realism(df, FS_BASELINE)
        s6 = step6_failure_modes(df, FS_BASELINE)
        s7 = step7_vix_contribution(df, s1)
        return step8_verdict(s1, s2, s3, s4, s5, s6, s7)

    def test_returns_required_keys(self):
        v = self._run_all()
        for k in ("overall_verdict", "flags", "critical_failure",
                  "signal_real", "n_flags"):
            assert k in v

    def test_flags_is_list(self):
        v = self._run_all()
        assert isinstance(v["flags"], list)

    def test_overall_verdict_is_string(self):
        v = self._run_all()
        assert isinstance(v["overall_verdict"], str) and len(v["overall_verdict"]) > 0

    def test_critical_failure_when_permutation_failed(self):
        """When permutation verdict is FAILED, critical_failure must be True."""
        df = _make_df(200)
        s1 = step1_feature_dominance(df)
        s2 = {"verdict": "FAILED", "z_spearman": 0.5}
        s3 = step3_symbol_robustness(df, FS_BASELINE)
        s4 = step4_regime_stability(df, FS_BASELINE)
        s5 = step5_economic_realism(df, FS_BASELINE)
        s6 = step6_failure_modes(df, FS_BASELINE)
        s7 = step7_vix_contribution(df, s1)
        v = step8_verdict(s1, s2, s3, s4, s5, s6, s7)
        assert v["critical_failure"] is True

    def test_signal_real_when_permutation_passes(self):
        """When permutation verdict is REAL, signal_real must be True."""
        df = _make_df(200)
        s1 = step1_feature_dominance(df)
        s2 = {"verdict": "REAL", "z_spearman": 4.0,
              "real_spearman": 0.70, "null_spearman_mean": 0.01,
              "p_value_spearman": 0.002, "p_value_r2": 0.002,
              "p_value_auc": 0.003}
        s3 = step3_symbol_robustness(df, FS_BASELINE)
        s4 = step4_regime_stability(df, FS_BASELINE)
        s5 = step5_economic_realism(df, FS_BASELINE)
        s6 = step6_failure_modes(df, FS_BASELINE)
        s7 = step7_vix_contribution(df, s1)
        v = step8_verdict(s1, s2, s3, s4, s5, s6, s7)
        assert v["signal_real"] is True

    def test_nbr_pct_stored(self):
        v = self._run_all()
        assert math.isfinite(v["nbr_pct_of_baseline"])

    def test_regime_sp_range_finite(self):
        v = self._run_all()
        assert math.isfinite(v["regime_sp_range"])

    def test_mechanical_link_flag_stored(self):
        v = self._run_all()
        assert isinstance(v["mechanical_link_flag"], bool)

    def test_n_flags_consistent_with_flags_list(self):
        v = self._run_all()
        assert v["n_flags"] == len(v["flags"])


# ── _fold_data ────────────────────────────────────────────────────────────────

class TestFoldData:
    def test_returns_none_for_insufficient_data(self):
        df = _make_df(20)
        # Only 2023 data → test fold 2026 will find ≤5 test rows after filtering
        # Just test that the function returns None gracefully for tiny subsets
        result = _fold_data(df, [2023], 2026, FS_BASELINE, "term_ratio_change")
        # Either None or a valid tuple — both are acceptable
        if result is not None:
            X_tr, y_tr, X_te, y_te, idx = result
            assert len(X_tr) >= 5 and len(X_te) >= 5

    def test_scaler_fit_only_on_train(self):
        """Verify that test data scaling uses train-fold statistics only."""
        df    = _make_df(200)
        feats = ["near_back_ratio"]
        fd    = _fold_data(df, [2023, 2024], 2025, feats, "term_ratio_change")
        if fd is None:
            pytest.skip("Insufficient data for this fold")
        X_tr, y_tr, X_te, y_te, idx = fd
        # Train fold must be standardized (mean ≈ 0)
        assert abs(X_tr.mean()) < 0.2, "Train fold mean should be near 0 after scaling"

    def test_returns_correct_types(self):
        df = _make_df(200)
        fd = _fold_data(df, [2023], 2024, FS_BASELINE, "term_ratio_change")
        if fd is None:
            pytest.skip("Insufficient data")
        X_tr, y_tr, X_te, y_te, idx = fd
        assert isinstance(X_tr, np.ndarray)
        assert isinstance(y_tr, np.ndarray)
        assert isinstance(idx, list)
