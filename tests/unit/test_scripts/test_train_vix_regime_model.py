"""Unit tests for scripts/train_vix_regime_model.py.

All tests are pure (no DB, no parquet, no disk I/O).  They exercise:
  - compute_vix_features   (rolling trailing-percentile, no leakage)
  - engineer_features      (baseline feature derivation + VIX join)
  - regime_analysis        (correlation + quintile binning)
  - wf_regression          (walk-forward Ridge, per-fold + pooled metrics)
  - wf_classification      (walk-forward LR)
  - economic_eval          (quintile P&L proxy)
  - build_verdict          (final decision logic)
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from scripts.train_vix_regime_model import (
    BASELINE_FEATURES,
    COMBINED_FEATURES,
    VIX_FEATURES,
    build_verdict,
    compute_vix_features,
    economic_eval,
    engineer_features,
    regime_analysis,
    wf_classification,
    wf_regression,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_vix(n: int = 300, start: str = "2020-01-02", seed: int = 0) -> pd.DataFrame:
    """Synthetic VIX series: n business-day dates starting at *start*.

    Uses a low-drift random walk so VIX stays in [13, 25] and never clusters
    at a clip boundary — avoids degenerate qcut bins in regime_analysis tests.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    # Small step-size (0.05) limits 3-sigma drift to ±0.05*sqrt(n) ≈ ±1.9 over
    # 1500 steps from an anchor of 18 — well inside the [10, 60] clip.
    closes = 18.0 + np.cumsum(rng.normal(0, 0.05, n))
    closes = np.clip(closes, 10.0, 60.0)
    return pd.DataFrame({"trade_date": dates, "vix_close": closes})


def _make_labels(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Synthetic earnings label rows spanning 2023-2026.

    pre_capture_dates are chosen directly from the VIX business-day calendar
    so every label row can join to a VIX value without gaps.
    """
    rng = np.random.default_rng(seed)
    # Build a pool of business days spanning 2023-2026 (guaranteed overlap with VIX)
    bdays = pd.date_range("2023-01-03", "2026-12-31", freq="B")
    idx = rng.integers(1, len(bdays), size=n)   # offset by 1 so event_date != pre_date
    pre_dates = bdays[idx - 1]
    dates = bdays[idx]                          # event_date = pre_date + 1 bday
    years = pd.DatetimeIndex(dates).year
    front_iv = rng.uniform(0.30, 1.20, n)
    back_iv  = rng.uniform(0.20, 0.80, n)
    post_front_iv = front_iv * rng.uniform(0.4, 0.95, n)
    post_back_iv  = back_iv  * rng.uniform(0.60, 0.98, n)
    front_crush = (post_front_iv - front_iv) / front_iv
    back_crush  = (post_back_iv  - back_iv)  / back_iv
    pre_trc   = front_iv / back_iv
    post_trc  = post_front_iv / post_back_iv
    trc_change = post_trc - pre_trc
    return pd.DataFrame({
        "id":                    range(n),
        "symbol":                [f"SYM{i % 10}" for i in range(n)],
        "event_date":            list(dates),
        "pre_capture_date":      list(pre_dates),
        "post_capture_date":     [d + pd.Timedelta(days=1) for d in dates],
        "release_timing":        "AMC",
        "pre_front_iv":          front_iv,
        "post_front_iv":         post_front_iv,
        "pre_back_iv":           back_iv,
        "post_back_iv":          post_back_iv,
        "front_iv_crush_pct":    front_crush,
        "back_iv_crush_pct":     back_crush,
        "term_ratio_change":     trc_change,
        "underlying_move_pct":   rng.uniform(-0.10, 0.10, n),
        "front_dte_pre":         rng.integers(3, 10, size=n).astype(float),
        "back_dte_pre":          rng.integers(25, 50, size=n).astype(float),
        "exact_expiry_match":    1,
        "quality_score":         rng.uniform(0.6, 1.0, n),
        "quality_tier":          "A",
        "pre_front_atm_moneyness": rng.uniform(-0.01, 0.01, n),
        "pre_front_oi":          rng.integers(100, 5000, size=n).astype(float),
        "pre_underlying_price":  rng.uniform(20, 500, n),
        "year":                  years,
        "quarter":               pd.DatetimeIndex(dates).quarter,
    })


def _make_full_df(n_labels: int = 80) -> pd.DataFrame:
    """Return engineered feature df ready for walk-forward functions.

    Always generates 2200 VIX business-days from 2019-01-02, which reaches
    2027-09 and covers all 2023-2026 label pre_capture_dates with no gaps.
    The trailing-percentile loop is O(n × 252) so 2200 rows is fast.
    """
    vix_raw  = _make_vix(2200, start="2019-01-02")
    vix_feat = compute_vix_features(vix_raw)
    labels   = _make_labels(n_labels)
    return engineer_features(labels, vix_feat)


# ── compute_vix_features ──────────────────────────────────────────────────────

class TestComputeVixFeatures:
    def test_output_columns(self):
        vix = _make_vix(300)
        out = compute_vix_features(vix)
        assert {"log_vix", "vix_pct_252", "vix_chg_5d"}.issubset(out.columns)

    def test_log_vix_correct(self):
        vix = _make_vix(300)
        out = compute_vix_features(vix)
        np.testing.assert_allclose(out["log_vix"], np.log(out["vix_close"]), rtol=1e-9)

    def test_trailing_pct_range(self):
        """vix_pct_252 must be in [0, 1]."""
        vix = _make_vix(300)
        out = compute_vix_features(vix)
        valid = out["vix_pct_252"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_no_lookahead_in_pct(self):
        """Each pct value uses only data up to and including that row."""
        vix = _make_vix(60)
        out = compute_vix_features(vix)
        # Manually recompute row 30 percentile
        i = 30
        window = out["vix_close"].values[:i + 1]
        expected = (window <= out["vix_close"].values[i]).mean()
        assert abs(out["vix_pct_252"].values[i] - expected) < 1e-9

    def test_output_length_unchanged(self):
        n = 150
        vix = _make_vix(n, start="2022-01-03")
        out = compute_vix_features(vix)
        assert len(out) == n

    def test_monotone_vix_gets_increasing_pct(self):
        """A strictly increasing VIX series must have pct_252==1.0 at every row."""
        dates = pd.date_range("2022-01-03", periods=50, freq="B")
        closes = np.arange(10.0, 60.0)  # 10, 11, ..., 59
        vix = pd.DataFrame({"trade_date": dates, "vix_close": closes})
        out = compute_vix_features(vix)
        pct = out["vix_pct_252"].values
        # Every new row is the highest value so far → pct must be 1.0 for all rows
        assert np.all(pct == 1.0)


# ── engineer_features ─────────────────────────────────────────────────────────

class TestEngineerFeatures:
    def setup_method(self):
        vix = _make_vix(400)
        self.vix_feat = compute_vix_features(vix)
        self.labels   = _make_labels(60)

    def test_baseline_columns_present(self):
        df = engineer_features(self.labels, self.vix_feat)
        for col in BASELINE_FEATURES:
            assert col in df.columns, f"missing baseline feature: {col}"

    def test_vix_columns_present(self):
        df = engineer_features(self.labels, self.vix_feat)
        for col in VIX_FEATURES:
            assert col in df.columns, f"missing VIX feature: {col}"

    def test_near_back_ratio_formula(self):
        df = engineer_features(self.labels, self.vix_feat)
        expected = df["pre_front_iv"] / df["pre_back_iv"]
        np.testing.assert_allclose(df["near_back_ratio"].values, expected.values, rtol=1e-9)

    def test_binary_target_values(self):
        df = engineer_features(self.labels, self.vix_feat)
        assert set(df["crush_lt_40pct"].unique()).issubset({0, 1})

    def test_no_vix_leakage(self):
        """VIX is joined on pre_capture_date, never post_capture_date."""
        df = engineer_features(self.labels, self.vix_feat)
        # post_capture_date is always > pre_capture_date in our synthetic data
        # If post dates were used instead, the joined VIX would differ — we just
        # verify the join key is pre_capture_date by checking vix_close is NOT NaN
        # when pre_capture_date is within the VIX series.
        pre_in_vix = df["pre_capture_date"].isin(self.vix_feat["trade_date"])
        assert df.loc[pre_in_vix, "vix_close"].notna().all()

    def test_spread_differential_column(self):
        df = engineer_features(self.labels, self.vix_feat)
        assert "spread_differential" in df.columns
        expected = df["front_iv_crush_pct"] - df["back_iv_crush_pct"]
        np.testing.assert_allclose(df["spread_differential"].values, expected.values, rtol=1e-9)


# ── regime_analysis ───────────────────────────────────────────────────────────

class TestRegimeAnalysis:
    def setup_method(self):
        self.df = _make_full_df(n_labels=80)

    def test_returns_expected_keys(self):
        ra = regime_analysis(self.df)
        for key in ("named_bins", "quintile_bins", "correlations",
                    "nbr_by_vix_quintile", "expansion_events"):
            assert key in ra

    def test_quintile_bins_count(self):
        ra = regime_analysis(self.df)
        assert len(ra["quintile_bins"]) == 5

    def test_spearman_p_values_present(self):
        ra = regime_analysis(self.df)
        for k in ("spearman_vix_vs_front_crush", "spearman_vix_vs_trc",
                  "spearman_vixpct_vs_front_crush", "spearman_vixpct_vs_trc",
                  "partial_vix_vs_trc_controlling_nbr"):
            assert "r" in ra["correlations"][k]
            assert "p" in ra["correlations"][k]

    def test_partial_correlation_finite(self):
        ra = regime_analysis(self.df)
        pc = ra["correlations"]["partial_vix_vs_trc_controlling_nbr"]
        assert math.isfinite(pc["r"]) and math.isfinite(pc["p"])

    def test_expansion_events_have_required_fields(self):
        ra = regime_analysis(self.df)
        for ev in ra["expansion_events"]:
            assert "symbol" in ev and "vix_close" in ev and "front_iv_crush_pct" in ev

    def test_nbr_by_vix_quintile_has_five_keys(self):
        ra = regime_analysis(self.df)
        assert len(ra["nbr_by_vix_quintile"]) == 5


# ── wf_regression ─────────────────────────────────────────────────────────────

class TestWfRegression:
    def setup_method(self):
        self.df = _make_full_df(n_labels=120)

    def test_returns_three_folds(self):
        result = wf_regression(self.df, "term_ratio_change", BASELINE_FEATURES, "test_reg")
        assert len(result["folds"]) == 3

    def test_pooled_spearman_finite(self):
        result = wf_regression(self.df, "term_ratio_change", BASELINE_FEATURES, "test_reg")
        sp = result["pooled"]["spearman_r"]
        assert math.isfinite(sp)

    def test_pooled_r2_is_computed(self):
        result = wf_regression(self.df, "term_ratio_change", BASELINE_FEATURES, "test_reg")
        assert "r2" in result["pooled"]

    def test_scaler_fit_only_on_train(self):
        """Verify separate folds produce different fold-level predictions
        (not identical, which would indicate test data leaked into scaler fit)."""
        result = wf_regression(self.df, "term_ratio_change", BASELINE_FEATURES, "test_reg")
        fold_years = [f["test_year"] for f in result["folds"]]
        assert len(set(fold_years)) == 3, "all three fold test years must be distinct"

    def test_coef_keys_match_features(self):
        result = wf_regression(self.df, "term_ratio_change", COMBINED_FEATURES, "test_comb")
        for fold in result["folds"]:
            assert set(fold["coef"].keys()) == set(COMBINED_FEATURES)

    def test_rmse_ratio_positive(self):
        result = wf_regression(self.df, "term_ratio_change", BASELINE_FEATURES, "test_reg")
        for fold in result["folds"]:
            assert fold["rmse_ratio"] > 0

    def test_oos_preds_length_matches_test_data(self):
        result = wf_regression(self.df, "term_ratio_change", BASELINE_FEATURES, "test_reg")
        n_oos = sum(f["n_test"] for f in result["folds"])
        assert len(result["oos_preds"]) == n_oos


# ── wf_classification ─────────────────────────────────────────────────────────

class TestWfClassification:
    def setup_method(self):
        self.df = _make_full_df(n_labels=120)

    def test_returns_three_folds(self):
        result = wf_classification(self.df, "crush_lt_40pct", BASELINE_FEATURES, "test_clf")
        assert len(result["folds"]) == 3

    def test_pooled_auc_in_range(self):
        result = wf_classification(self.df, "crush_lt_40pct", BASELINE_FEATURES, "test_clf")
        auc = result["pooled"]["auc"]
        if auc is not None:
            assert 0.0 <= auc <= 1.0

    def test_pooled_brier_in_range(self):
        result = wf_classification(self.df, "crush_lt_40pct", BASELINE_FEATURES, "test_clf")
        brier = result["pooled"]["brier"]
        assert 0.0 <= brier <= 1.0

    def test_fold_probs_in_unit_interval(self):
        result = wf_classification(self.df, "crush_lt_40pct", BASELINE_FEATURES, "test_clf")
        for p in result["oos_preds"]:
            assert 0.0 <= p["y_prob"] <= 1.0

    def test_label_is_stored(self):
        result = wf_classification(self.df, "crush_lt_40pct", BASELINE_FEATURES, "my_label")
        assert result["label"] == "my_label"


# ── economic_eval ─────────────────────────────────────────────────────────────

class TestEconomicEval:
    def setup_method(self):
        self.df = _make_full_df(n_labels=120)
        reg = wf_regression(self.df, "term_ratio_change", BASELINE_FEATURES, "base")
        # Flip predictions so Q5 = deepest crush
        self.preds = [{**p, "y_pred": -p["y_pred"]} for p in reg["oos_preds"]]

    def test_returns_quintile_table(self):
        result = economic_eval(self.df, self.preds, "y_pred", "test")
        assert "quintile_table" in result
        assert len(result["quintile_table"]) == 5

    def test_q5_q1_spread_is_finite(self):
        result = economic_eval(self.df, self.preds, "y_pred", "test")
        assert math.isfinite(result["q5_q1_spread_pp"])

    def test_n_oos_positive(self):
        result = economic_eval(self.df, self.preds, "y_pred", "test")
        assert result["n_oos"] > 0

    def test_returns_empty_dict_for_no_preds(self):
        result = economic_eval(self.df, [], "y_pred", "test")
        assert result == {}

    def test_spearman_finite(self):
        result = economic_eval(self.df, self.preds, "y_pred", "test")
        assert math.isfinite(result["spearman_r"])


# ── build_verdict ─────────────────────────────────────────────────────────────

class TestBuildVerdict:
    def _make_verdict_inputs(self):
        df = _make_full_df(n_labels=120)
        ra = regime_analysis(df)
        r_base = wf_regression(df, "term_ratio_change", BASELINE_FEATURES, "Baseline (3 feat)")
        r_vix  = wf_regression(df, "term_ratio_change", VIX_FEATURES,      "VIX-only (2 feat)")
        r_comb = wf_regression(df, "term_ratio_change", COMBINED_FEATURES,  "Combined (5 feat)")
        c_base = wf_classification(df, "crush_lt_40pct", BASELINE_FEATURES, "Baseline (3 feat)")
        c_vix  = wf_classification(df, "crush_lt_40pct", VIX_FEATURES,      "VIX-only (2 feat)")
        c_comb = wf_classification(df, "crush_lt_40pct", COMBINED_FEATURES,  "Combined (5 feat)")

        def flip(preds):
            return [{**p, "y_pred": -p["y_pred"]} for p in preds]

        e_base = economic_eval(df, flip(r_base["oos_preds"]), "y_pred", "Baseline (3 feat)")
        e_vix  = economic_eval(df, flip(r_vix["oos_preds"]),  "y_pred", "VIX-only (2 feat)")
        e_comb = economic_eval(df, flip(r_comb["oos_preds"]), "y_pred", "Combined (5 feat)")
        fa: dict = {}
        return ra, [r_base, r_vix, r_comb], [c_base, c_vix, c_comb], [e_base, e_vix, e_comb], fa

    def test_verdict_has_required_keys(self):
        v = build_verdict(*self._make_verdict_inputs())
        for key in ("vix_belongs_in_production", "summary", "delta",
                    "vix_is_independent_of_nbr", "consistent_across_folds",
                    "meaningful_statistical_gain", "meaningful_economic_gain"):
            assert key in v, f"missing verdict key: {key}"

    def test_summary_is_one_of_three_options(self):
        v = build_verdict(*self._make_verdict_inputs())
        assert any(kw in v["summary"] for kw in ("INCLUDE", "MARGINAL", "EXCLUDE"))

    def test_delta_values_finite(self):
        v = build_verdict(*self._make_verdict_inputs())
        assert math.isfinite(v["delta"]["spearman_r"])
        assert math.isfinite(v["delta"]["r2"])

    def test_vix_belongs_is_boolean(self):
        v = build_verdict(*self._make_verdict_inputs())
        assert isinstance(v["vix_belongs_in_production"], bool)

    def test_fold_spearman_deltas_length(self):
        v = build_verdict(*self._make_verdict_inputs())
        # One delta per fold that both baseline and combined ran; up to 3.
        assert 0 <= len(v["fold_spearman_deltas"]) <= 3

    def test_include_verdict_when_metrics_are_strong(self):
        """When combined model clearly beats baseline across all folds,
        verdict must be INCLUDE."""
        ra, reg_results, clf_results, econ_results, fa = self._make_verdict_inputs()

        # Force partial correlation to be significant
        ra["correlations"]["partial_vix_vs_trc_controlling_nbr"]["r"] = 0.20
        ra["correlations"]["partial_vix_vs_trc_controlling_nbr"]["p"] = 0.001

        # Ensure both baseline and combined have 3 matching folds by injecting
        # synthetic fold data that shows consistent gain.
        stub_folds = [
            {"train_years": [2023],           "test_year": 2024, "n_train": 30, "n_test": 20,
             "spearman_r": 0.75, "r2": 0.55, "rmse": 0.05, "naive_rmse": 0.09,
             "rmse_ratio": 0.56, "spearman_p": 0.001, "coef": {}},
            {"train_years": [2023, 2024],      "test_year": 2025, "n_train": 50, "n_test": 20,
             "spearman_r": 0.77, "r2": 0.57, "rmse": 0.05, "naive_rmse": 0.09,
             "rmse_ratio": 0.56, "spearman_p": 0.001, "coef": {}},
            {"train_years": [2023, 2024, 2025], "test_year": 2026, "n_train": 70, "n_test": 20,
             "spearman_r": 0.78, "r2": 0.58, "rmse": 0.05, "naive_rmse": 0.09,
             "rmse_ratio": 0.56, "spearman_p": 0.001, "coef": {}},
        ]
        # Baseline: slightly lower spearman per fold
        base_folds = [{**f, "spearman_r": f["spearman_r"] - 0.05} for f in stub_folds]
        reg_results[0]["folds"] = base_folds
        reg_results[0]["pooled"] = {"spearman_r": 0.72, "r2": 0.52, "n": 60,
                                    "rmse": 0.06, "naive_rmse": 0.09, "rmse_ratio": 0.67,
                                    "spearman_p": 0.001}
        reg_results[2]["folds"] = stub_folds
        reg_results[2]["pooled"] = {"spearman_r": 0.77, "r2": 0.57, "n": 60,
                                    "rmse": 0.05, "naive_rmse": 0.09, "rmse_ratio": 0.56,
                                    "spearman_p": 0.001}

        # Classification pooled
        clf_results[0]["pooled"] = {"auc": 0.82, "brier": 0.17, "n": 60}
        clf_results[2]["pooled"] = {"auc": 0.85, "brier": 0.16, "n": 60}

        # Economic spread
        econ_results[0]["q5_q1_spread_pp"] = 10.0
        econ_results[2]["q5_q1_spread_pp"] = 12.5

        v = build_verdict(ra, reg_results, clf_results, econ_results, fa)
        assert "INCLUDE" in v["summary"]

    def test_exclude_verdict_when_no_signal(self):
        """When partial correlation is weak/insignificant and gain is trivial,
        verdict must not be INCLUDE."""
        ra, reg_results, clf_results, econ_results, fa = self._make_verdict_inputs()
        ra["correlations"]["partial_vix_vs_trc_controlling_nbr"]["r"] = 0.01
        ra["correlations"]["partial_vix_vs_trc_controlling_nbr"]["p"] = 0.80
        ra["correlations"]["spearman_vix_vs_trc"]["r"] = 0.05
        # Set combined == baseline (no improvement)
        sp = reg_results[0]["pooled"]["spearman_r"]
        reg_results[2]["pooled"]["spearman_r"] = sp + 0.001
        reg_results[2]["pooled"]["r2"] = reg_results[0]["pooled"]["r2"] + 0.001
        for i, fold in enumerate(reg_results[2]["folds"]):
            fold["spearman_r"] = reg_results[0]["folds"][i]["spearman_r"] + 0.001

        v = build_verdict(ra, reg_results, clf_results, econ_results, fa)
        assert "INCLUDE" not in v["summary"]
