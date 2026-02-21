import sys
import types
import unittest
import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Ensure test imports resolve when run as a standalone file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep logger file writes sandbox-safe in local/dev CI environments.
os.environ.setdefault("HOME", tempfile.gettempdir())


def _install_market_data_stub() -> None:
    """Stub services.market_data to avoid optional runtime deps in unit tests."""
    if "services.market_data" in sys.modules:
        return
    stub_md = types.ModuleType("services.market_data")

    class MarketDataService:  # pragma: no cover - tiny test helper
        def get_vix(self):
            return 20.0

        def get_historical_data(self, symbol, period="1y", interval="1d"):
            return pd.DataFrame()

    stub_md.MarketDataService = MarketDataService
    sys.modules["services.market_data"] = stub_md


class TestQuantRegressions(unittest.TestCase):
    def test_yang_zhang_regression(self):
        _install_market_data_stub()

        from services.volatility_service import VolatilityService

        class DummyConfig:
            def get(self, key, default=None):
                return default

        class DummyMarket:
            def get_vix(self):
                return 20.0

            def get_historical_data(self, symbol, period="1y", interval="1d"):
                return pd.DataFrame()

        np.random.seed(123)
        n = 140
        rets = np.random.normal(0.0002, 0.014, size=n)
        close = 100 * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1] * np.exp(np.random.normal(0.0, 0.003, size=n - 1))]
        high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0.002, 0.0025, size=n)))
        low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0.002, 0.0025, size=n)))
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})

        service = VolatilityService(DummyConfig(), DummyMarket())
        window = 30
        yz_series = service._yang_zhang_volatility(df, window=window)
        yz_last = float(yz_series.iloc[-1])

        # Independent reference for last point.
        log_ho = np.log(df["High"] / df["Open"])
        log_lo = np.log(df["Low"] / df["Open"])
        log_co = np.log(df["Close"] / df["Open"])
        log_oc = np.log(df["Open"] / df["Close"].shift(1))
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        open_var = log_oc.rolling(window=window, min_periods=window).var(ddof=1)
        close_var = log_co.rolling(window=window, min_periods=window).var(ddof=1)
        rs_mean = rs.rolling(window=window, min_periods=window).mean()
        k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
        ref_var = (open_var + k * close_var + (1 - k) * rs_mean).clip(lower=0.0)
        ref_last = float(np.sqrt(ref_var.iloc[-1]) * np.sqrt(252.0))
        ref_last = float(np.clip(ref_last, service.min_volatility, service.max_volatility))

        self.assertTrue(np.isfinite(yz_last))
        self.assertGreaterEqual(yz_last, service.min_volatility)
        self.assertLessEqual(yz_last, service.max_volatility)
        self.assertAlmostEqual(yz_last, ref_last, places=10)

    def test_term_structure_slope_0_45_regression(self):
        _install_market_data_stub()

        from services.options_service import OptionsService, OptionChain, OptionContract, OptionType

        class DummyConfig:
            def get(self, key, default=None):
                return default

        class DummyMarket:
            pass

        service = OptionsService(DummyConfig(), DummyMarket())
        today = datetime.now().date()
        expirations = [
            (today + timedelta(days=14)).strftime("%Y-%m-%d"),
            (today + timedelta(days=30)).strftime("%Y-%m-%d"),
            (today + timedelta(days=45)).strftime("%Y-%m-%d"),
            (today + timedelta(days=60)).strftime("%Y-%m-%d"),
        ]
        ivs = {
            expirations[0]: 0.34,
            expirations[1]: 0.30,
            expirations[2]: 0.27,
            expirations[3]: 0.25,
        }

        service.get_available_expirations = lambda symbol: expirations  # type: ignore

        def _chain(symbol, exp):
            iv = float(ivs[exp])
            call = OptionContract(
                symbol=symbol,
                strike=100.0,
                expiration=exp,
                option_type=OptionType.CALL,
                bid=2.0,
                ask=2.2,
                last=2.1,
                volume=100,
                open_interest=200,
                implied_volatility=iv,
            )
            put = OptionContract(
                symbol=symbol,
                strike=100.0,
                expiration=exp,
                option_type=OptionType.PUT,
                bid=2.1,
                ask=2.3,
                last=2.2,
                volume=90,
                open_interest=180,
                implied_volatility=iv,
            )
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            return OptionChain(
                symbol=symbol,
                expiration=exp,
                underlying_price=100.0,
                calls=[call],
                puts=[put],
                days_to_expiration=dte,
            )

        service.get_option_chain = _chain  # type: ignore

        ts = service.build_iv_term_structure("TEST")
        self.assertIsNotNone(ts)
        slope = float(ts["slope_0_45"])
        iv_30d = float(ts["iv_30d"])
        iv_45d = float(ts["iv_45d"])

        self.assertLess(slope, 0.0)
        self.assertAlmostEqual(iv_30d, 0.30, places=6)
        self.assertAlmostEqual(iv_45d, 0.27, places=6)

    def test_term_structure_slope_interpolation_extrapolation(self):
        _install_market_data_stub()

        from services.volatility_service import VolatilityService

        class DummyConfig:
            def get(self, key, default=None):
                return default

        class DummyMarket:
            def get_vix(self):
                return 20.0

            def get_historical_data(self, symbol, period="1y", interval="1d"):
                return pd.DataFrame()

        service = VolatilityService(DummyConfig(), DummyMarket())
        days = [14, 30, 60]
        ivs = [0.36, 0.30, 0.24]

        slope = float(service._calculate_term_structure_slope(days, ivs, 0, 45))

        # Expected from linear extrapolation/interpolation:
        # IV(0) from [14, 30], IV(45) from [30, 60].
        iv0 = 0.36 + (0.0 - 14.0) * (0.30 - 0.36) / (30.0 - 14.0)
        iv45 = 0.30 + (45.0 - 30.0) * (0.24 - 0.30) / (60.0 - 30.0)
        expected = (iv45 - iv0) / 45.0

        self.assertTrue(np.isfinite(slope))
        self.assertLess(slope, 0.0)
        self.assertAlmostEqual(slope, expected, places=12)

    def test_monte_carlo_jump_earnings_sensitivity(self):
        from utils.monte_carlo import MonteCarloEngine

        np.random.seed(5)
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0, 0.01, 160)))
        hist = pd.DataFrame({"Close": close})
        metrics = {"iv30": 0.44, "rv30": 0.23, "yang_zhang_volatility": 0.29}

        engine = MonteCarloEngine()
        res_far = engine.run_simulation(
            "TEST",
            100.0,
            hist,
            metrics,
            simulations=1000,
            days_to_expiration=30,
            days_to_earnings=30,
        )
        res_near = engine.run_simulation(
            "TEST",
            100.0,
            hist,
            metrics,
            simulations=1000,
            days_to_expiration=30,
            days_to_earnings=2,
        )

        lam_far = float(res_far["jump_parameters"]["lambda"])
        lam_near = float(res_near["jump_parameters"]["lambda"])
        self.assertGreater(lam_near, lam_far)

        h = res_near["heston_parameters"]
        self.assertGreaterEqual(2.0 * float(h["kappa"]) * float(h["theta"]), float(h["sigma"]) ** 2 * 0.99)

    def test_monte_carlo_disable_jump_flag(self):
        from utils.monte_carlo import MonteCarloEngine

        np.random.seed(7)
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0, 0.01, 120)))
        hist = pd.DataFrame({"Close": close})
        metrics = {"iv30": 0.35, "rv30": 0.22, "yang_zhang_volatility": 0.26}

        engine = MonteCarloEngine()
        res = engine.run_simulation(
            "TEST",
            100.0,
            hist,
            metrics,
            simulations=1000,
            days_to_expiration=30,
            days_to_earnings=1,
            use_jump_diffusion=False,
        )

        self.assertIn("jump_parameters", res)
        self.assertFalse(bool(res["jump_parameters"]["enabled"]))


if __name__ == "__main__":
    unittest.main()
