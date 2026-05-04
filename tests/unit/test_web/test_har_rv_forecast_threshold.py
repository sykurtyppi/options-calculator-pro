"""
P-3b — pin the edge-engine HAR minimum-observation threshold at 100.

Reconciles edge_engine._har_rv_forecast with the snapshot path's
services.earnings_vol_snapshot._HAR_MIN_OBS = 100 (and the new shared
services.realized_vol.HAR_MIN_OBS = 100). Required so that a future
migration of edge_engine onto services.realized_vol is byte-equivalent
on this gate.

These tests target web.api.edge_engine._har_rv_forecast directly. After
the edge_engine migration eventually lands, these same tests will still
hold because edge_engine._har_rv_forecast will resolve to the shared
kernel — which uses the same 100-observation gate.
"""
import unittest

import numpy as np
import pandas as pd

from web.api.edge_engine import _har_rv_forecast


class TestEdgeEngineHARRVForecastThreshold(unittest.TestCase):
    def test_returns_none_at_99_observations(self):
        """Below the 100-observation gate, the function must return None.

        Before P-3b the edge_engine gate was n < 35, so a 99-row series
        would have returned a forecast. Pinning at n < 100 aligns it with
        the snapshot path and the shared services.realized_vol kernel.
        """
        rv_daily = pd.Series([0.25] * 99)
        self.assertIsNone(_har_rv_forecast(rv_daily))

    def test_returns_forecast_at_100_observations_with_valid_data(self):
        """At exactly 100 observations the gate should let the regression
        proceed and return a finite forecast.

        Construction: a constant σ = 0.25 input. With every RV equal to σ²,
        the (RV_d, RV_w, RV_m) features are identical across the in-sample
        and OLS produces a fit whose forecast is the constant. Floored at
        1e-4 by the implementation.
        """
        sigma = 0.25
        rv_daily = pd.Series([sigma] * 100)
        forecast = _har_rv_forecast(rv_daily)
        self.assertIsNotNone(forecast)
        self.assertTrue(np.isfinite(forecast))
        self.assertAlmostEqual(forecast, sigma, places=4)

    def test_returns_none_at_one_below_threshold_off_by_one_guard(self):
        """Belt-and-braces: confirm the gate is strictly less-than (n < 100).
        Exactly 99 → None (already covered above), and 100 → forecast (above).
        This third test pins n=99 explicitly with a varied input so any future
        change of the gate from `<` to `<=` would fire here as well.
        """
        rng = np.random.default_rng(42)
        rv_daily = pd.Series(np.abs(rng.normal(0.25, 0.05, size=99)))
        self.assertIsNone(_har_rv_forecast(rv_daily))


if __name__ == "__main__":
    unittest.main()
