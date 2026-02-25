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


if __name__ == "__main__":
    unittest.main()
