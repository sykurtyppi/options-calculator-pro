import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional dependency in constrained environments
    TestClient = None

import web.api.app as app_module
from web.api.edge_engine import EdgeSnapshot


class _StubCollector:
    def __init__(self):
        self.db = type("StubDB", (), {"db_path": str(Path(tempfile.gettempdir()) / "stub_oos.sqlite")})()

    def run_oos_validation(self, **_kwargs):
        return {
            "splits": np.int64(5),
            "best_params": {
                "min_signal_score": np.float64(0.5),
            },
            "report_card": {
                "verdict": {"grade": "B", "overall_pass": True},
                "sample": {"total_test_trades": np.int64(42), "avg_trades_per_split": np.float64(8.4)},
                "metrics": {
                    "alpha": {"mean": np.float64(0.12), "low": np.float64(0.03)},
                    "sharpe": {"mean": np.float64(1.10), "low": np.float64(0.70)},
                    "win_rate": {"mean": np.float64(0.58), "low": np.float64(0.49)},
                    "pnl": {"mean": np.float64(122.5), "low": np.float64(24.0)},
                },
                "gates": {"min_splits": True},
            },
            "csv_path": Path("exports/reports/oos.csv"),
            "markdown_path": Path("exports/reports/oos.md"),
            "json_path": Path("exports/reports/oos.json"),
            "report_card_markdown_path": Path("exports/reports/oos_report_card.md"),
            "report_card_json_path": Path("exports/reports/oos_report_card.json"),
        }


class _AdaptiveStubCollector:
    def __init__(self):
        self.db = type("StubDB", (), {"db_path": str(Path(tempfile.gettempdir()) / "stub_oos_adaptive.sqlite")})()
        self.calls = 0

    def run_oos_validation(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "splits": np.int64(4),
                "best_params": {},
                "report_card": {
                    "ready": True,
                    "verdict": {"grade": "D", "overall_pass": False},
                    "sample": {"splits": np.int64(4), "total_test_trades": np.int64(21), "avg_trades_per_split": np.float64(5.25)},
                    "metrics": {"alpha": {"mean": np.float64(0.02), "low": np.float64(-0.20)}},
                    "gates": {
                        "min_splits": {"required": 8, "actual": 4, "passed": False},
                        "min_total_test_trades": {"required": 80, "actual": 21, "passed": False},
                        "min_trades_per_split": {"required": 5.0, "actual": 5.25, "passed": True},
                    },
                },
                "csv_path": Path("exports/reports/oos_fail.csv"),
                "markdown_path": Path("exports/reports/oos_fail.md"),
                "json_path": Path("exports/reports/oos_fail.json"),
                "report_card_markdown_path": Path("exports/reports/oos_fail_report_card.md"),
                "report_card_json_path": Path("exports/reports/oos_fail_report_card.json"),
            }
        return {
            "splits": np.int64(10),
            "best_params": {"min_signal_score": np.float64(0.55)},
            "report_card": {
                "ready": True,
                "verdict": {"grade": "B", "overall_pass": True},
                "sample": {"splits": np.int64(10), "total_test_trades": np.int64(144), "avg_trades_per_split": np.float64(14.4)},
                "metrics": {"alpha": {"mean": np.float64(0.25), "low": np.float64(0.11)}},
                "gates": {
                    "min_splits": {"required": 8, "actual": 10, "passed": True},
                    "min_total_test_trades": {"required": 80, "actual": 144, "passed": True},
                    "min_trades_per_split": {"required": 5.0, "actual": 14.4, "passed": True},
                },
            },
            "csv_path": Path("exports/reports/oos_pass.csv"),
            "markdown_path": Path("exports/reports/oos_pass.md"),
            "json_path": Path("exports/reports/oos_pass.json"),
            "report_card_markdown_path": Path("exports/reports/oos_pass_report_card.md"),
            "report_card_json_path": Path("exports/reports/oos_pass_report_card.json"),
        }


class _ProfileCaptureStubCollector:
    last_kwargs = None

    def __init__(self):
        self.db = type("StubDB", (), {"db_path": str(Path(tempfile.gettempdir()) / "stub_oos_profile.sqlite")})()

    def run_oos_validation(self, **kwargs):
        _ProfileCaptureStubCollector.last_kwargs = kwargs
        return {
            "splits": np.int64(8),
            "best_params": {},
            "report_card": {
                "ready": True,
                "verdict": {"grade": "C", "overall_pass": False},
                "sample": {"splits": np.int64(8), "total_test_trades": np.int64(80), "avg_trades_per_split": np.float64(10.0)},
                "metrics": {
                    "alpha": {"mean": np.float64(0.10), "low": np.float64(0.01)},
                    "sharpe": {"mean": np.float64(0.20), "low": np.float64(-0.20)},
                    "win_rate": {"mean": np.float64(0.55), "low": np.float64(0.48)},
                    "pnl": {"mean": np.float64(35.0), "low": np.float64(-12.0)},
                },
                "gates": {
                    "min_splits": {"required": 8, "actual": 8, "passed": True},
                    "min_total_test_trades": {"required": 80, "actual": 80, "passed": True},
                    "min_trades_per_split": {"required": 5.0, "actual": 10.0, "passed": True},
                },
            },
            "csv_path": Path("exports/reports/oos_profile.csv"),
            "markdown_path": Path("exports/reports/oos_profile.md"),
            "json_path": Path("exports/reports/oos_profile.json"),
            "report_card_markdown_path": Path("exports/reports/oos_profile_report_card.md"),
            "report_card_json_path": Path("exports/reports/oos_profile_report_card.json"),
        }


@unittest.skipIf(TestClient is None, "httpx/TestClient dependency not available")
class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("timestamp", payload)

    def test_edge_analyze_endpoint(self):
        snapshot = EdgeSnapshot(
            symbol="AAPL",
            recommendation="Watchlist",
            confidence_pct=77.7,
            setup_score=0.66,
            metrics={"iv_rv30": 1.31},
            rationale=["signal check"],
        )
        with patch.object(app_module, "analyze_single_ticker", return_value=snapshot):
            response = self.client.post("/api/edge/analyze", json={"symbol": "AAPL"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["recommendation"], "Watchlist")
        self.assertAlmostEqual(payload["confidence_pct"], 77.7, places=3)
        self.assertEqual(payload["metrics"]["iv_rv30"], 1.31)

    def test_oos_report_card_serializes_numpy(self):
        with patch.object(app_module, "InstitutionalDataCollector", _StubCollector):
            response = self.client.post("/api/oos/report-card", json={})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"]["grade"], "B")
        self.assertEqual(payload["summary"]["splits"], 5)
        self.assertEqual(payload["summary"]["sample"]["total_test_trades"], 42)
        self.assertEqual(payload["output_files"]["csv"], "exports/reports/oos.csv")

    def test_oos_report_card_adaptive_retry_improves_sample(self):
        with patch.object(app_module, "InstitutionalDataCollector", _AdaptiveStubCollector):
            response = self.client.post(
                "/api/oos/report-card",
                json={
                    "oos_stability_profile": "evidence_balanced",
                    "lookback_days": 730,
                    "max_backtest_symbols": 20,
                    "oos_train_days": 252,
                    "oos_test_days": 63,
                    "oos_step_days": 63,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["summary"]["adaptive_retry_used"])
        self.assertTrue(any("Adaptive OOS retry applied" in note for note in payload["summary"]["notes"]))
        self.assertEqual(payload["summary"]["grade"], "B")
        self.assertEqual(payload["summary"]["sample"]["total_test_trades"], 144)
        self.assertEqual(payload["summary"]["windows_used"]["max_backtest_symbols"], 50)
        self.assertEqual(payload["summary"]["windows_used"]["lookback_days"], 1095)

    def test_oos_variance_control_profile_applies_stricter_bounds(self):
        _ProfileCaptureStubCollector.last_kwargs = None
        with patch.object(app_module, "InstitutionalDataCollector", _ProfileCaptureStubCollector):
            response = self.client.post(
                "/api/oos/report-card",
                json={
                    "oos_stability_profile": "variance_control",
                    "min_signal_score": 0.40,
                    "min_crush_confidence": 0.20,
                    "min_crush_magnitude": 0.04,
                    "min_crush_edge": 0.01,
                    "entry_dte_band": 8,
                    "min_daily_share_volume": 100_000,
                    "max_abs_momentum_5d": 0.20,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        kwargs = _ProfileCaptureStubCollector.last_kwargs
        self.assertIsNotNone(kwargs)
        self.assertEqual(kwargs["execution_profiles"], ["institutional_tight", "institutional"])
        self.assertEqual(kwargs["trades_per_day_grid"], [1, 2])
        self.assertEqual(kwargs["entry_days_grid"], [3, 5])
        self.assertEqual(kwargs["signal_threshold_grid"], [0.55])
        self.assertAlmostEqual(kwargs["min_crush_confidence"], 0.40, places=6)
        self.assertAlmostEqual(kwargs["min_crush_magnitude"], 0.07, places=6)
        self.assertAlmostEqual(kwargs["min_crush_edge"], 0.03, places=6)
        self.assertEqual(kwargs["entry_dte_band"], 4)
        self.assertEqual(kwargs["min_daily_share_volume"], 2_500_000)
        self.assertAlmostEqual(kwargs["max_abs_momentum_5d"], 0.08, places=6)
        self.assertEqual(payload["summary"]["stability_profile_requested"], "variance_control")
        self.assertEqual(payload["summary"]["stability_profile_used"], "variance_control")


if __name__ == "__main__":
    unittest.main()
