"""Tests for the honest crush-model CV metrics helper (audit H3/H4).

_crush_oof_metrics must (a) split by SYMBOL so a ticker's events don't leak
across folds, (b) fall back to ungrouped stratified CV — and SAY so — when there
are too few symbols, and (c) return out-of-fold metrics for the calibrated
pipeline, not in-sample numbers. The helper uses no `self`, so we call it unbound.
"""
import unittest

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    _HAVE_SK = True
except ImportError:
    _HAVE_SK = False

from services.institutional_ml_db import InstitutionalMLDatabase


@unittest.skipUnless(_HAVE_SK, "scikit-learn / numpy not installed")
class TestCrushOofMetrics(unittest.TestCase):
    def _data(self, n_symbols, per_symbol=12):
        rng = np.random.RandomState(1)
        rows, groups = [], []
        for s in range(n_symbols):
            for _ in range(per_symbol):
                rows.append(rng.uniform([0.5, -3.0, 0.5], [4.0, 0.0, 5.0]))
                groups.append(f"SYM{s}")
        X = np.array(rows)
        y = (X[:, 0] > 1.8).astype(int)  # crush ~ high NBR
        return X, y, np.array(groups)

    def _run(self, X, y, groups, folds=5):
        base_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42,
                                     class_weight="balanced")
        return InstitutionalMLDatabase._crush_oof_metrics(
            None, X, y, groups, folds, base_lr, 0.50)

    def test_groups_by_symbol_when_enough_tickers(self):
        X, y, groups = self._data(n_symbols=15)
        m = self._run(X, y, groups)
        self.assertTrue(m["cv_scheme"].startswith("stratified_group_kfold_by_symbol"),
                        m["cv_scheme"])
        self.assertEqual(len(m["cv_auc"]), 5)
        for key in ("oof_auc", "oof_brier", "oof_precision", "oof_recall"):
            self.assertTrue(np.isfinite(m[key]), key)
        self.assertGreaterEqual(m["oof_auc"], 0.0)
        self.assertLessEqual(m["oof_auc"], 1.0)

    def test_falls_back_to_ungrouped_with_too_few_symbols(self):
        # 3 symbols but 5 folds → grouping impossible, must fall back and say so.
        X, y, groups = self._data(n_symbols=3, per_symbol=40)
        m = self._run(X, y, groups, folds=5)
        self.assertIn("ungrouped", m["cv_scheme"])


if __name__ == "__main__":
    unittest.main()
