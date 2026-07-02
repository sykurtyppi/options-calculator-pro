"""Tests for the shared crush-classifier feature transform (services.crush_features).

The whole point of this module is that TRAIN and SERVE apply the identical
transform. These tests pin the clips/fallbacks that removed the train/serve
skew (audit H1) and the no-RV ratio the forward screener must use (M5).
"""
import math
import unittest

from services import crush_features as cf


class TestCrushFeatureVector(unittest.TestCase):
    def test_feature_order_and_length(self):
        vec = cf.crush_feature_vector(near_iv=0.30, near_back_ratio=1.10, iv_rv=1.333)
        self.assertEqual(len(vec), 3)
        # [near_back_ratio, log_front_iv, iv_rv_approx]
        self.assertAlmostEqual(vec[0], 1.10, places=6)
        self.assertAlmostEqual(vec[1], math.log(0.30), places=6)
        self.assertAlmostEqual(vec[2], 1.333, places=6)

    def test_nbr_clipped_to_train_bounds(self):
        """The bug (H1): out-of-range NBR fed the scaler raw, saturating the model.
        NBR must clip to [0.50, 4.0] exactly as at train time."""
        self.assertEqual(cf.crush_feature_vector(0.30, 12.0, 1.333)[0], 4.0)
        self.assertEqual(cf.crush_feature_vector(0.30, 50.0, 1.333)[0], 4.0)
        self.assertEqual(cf.crush_feature_vector(0.30, 0.01, 1.333)[0], 0.50)

    def test_iv_rv_clipped_to_train_bounds(self):
        self.assertEqual(cf.crush_feature_vector(0.30, 1.10, 99.0)[2], 5.0)
        self.assertEqual(cf.crush_feature_vector(0.30, 1.10, 0.01)[2], 0.50)

    def test_front_iv_floor_before_log(self):
        # near_iv <= 0 must floor at IV_FLOOR, never log(0)/log(neg).
        self.assertAlmostEqual(cf.crush_feature_vector(0.0, 1.10, 1.333)[1],
                               math.log(cf.IV_FLOOR), places=6)
        self.assertAlmostEqual(cf.crush_feature_vector(-1.0, 1.10, 1.333)[1],
                               math.log(cf.IV_FLOOR), places=6)

    def test_nonfinite_inputs_fall_back(self):
        vec = cf.crush_feature_vector(float("nan"), float("nan"), float("inf"))
        # Non-finite inputs fall back to neutral defaults (never NaN/inf into the scaler):
        # near_iv NaN -> log(IV_FLOOR); NBR NaN -> NBR_FALLBACK; iv_rv inf -> no-RV ratio.
        self.assertAlmostEqual(vec[0], cf.NBR_FALLBACK, places=6)
        self.assertAlmostEqual(vec[1], math.log(cf.IV_FLOOR), places=6)
        self.assertAlmostEqual(vec[2], cf.NO_RV_IV_RV_RATIO, places=6)

    def test_no_rv_ratio_matches_training_fallback(self):
        """When RV is absent, training sets RV = front_iv * 0.75, so the RATIO is
        front_iv / (front_iv*0.75) = 1/0.75. The forward screener must pass this
        exact ratio, NOT front_iv*0.75 (a different quantity — the M5 bug)."""
        self.assertAlmostEqual(cf.NO_RV_IV_RV_RATIO, 1.0 / 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
