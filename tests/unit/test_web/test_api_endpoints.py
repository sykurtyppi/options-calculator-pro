import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional dependency in constrained environments
    TestClient = None

import web.api.app as app_module
from web.api.edge_engine import EdgeSnapshot


def _make_mock_mda_client():
    """Return a no-op MarketDataClient mock (is_available=False → pure yfinance path)."""
    m = MagicMock()
    m.is_available.return_value = False
    return m


class _StubFeatureStore:
    data_root = Path("/tmp/market_data")
    feature_root = Path("/tmp/market_data/research/options_features_eod")

    def is_available(self):
        return True

    def list_symbols(self):
        return ["SPY"]

    def coverage(self, symbol=None):
        return [
            {
                "symbol": symbol or "SPY",
                "start_date": "2025-01-02",
                "end_date": "2025-01-31",
                "rows": 123,
                "expiries": 4,
                "option_symbols": 120,
            }
        ]

    def query_chain_records(self, symbol, **kwargs):
        assert symbol == "SPY"
        assert kwargs["trade_date"] == "2025-01-02"
        return [
            {
                "trade_date": "2025-01-02",
                "expiry": "2025-01-17",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY250117C00600000",
                "call_put": "C",
                "dte": 15,
                "strike": 600.0,
                "bid": 4.0,
                "ask": 4.2,
                "mid": 4.1,
                "iv": 0.20,
                "delta": 0.51,
                "quote_quality_flag": "ok",
            }
        ]


def _stub_screener_payload(expiry_mode="front_after_earnings"):
    return {
        "generated_at": "2026-04-17T12:00:00Z",
        "expiry_mode": expiry_mode,
        "as_of_date": "2026-04-17",
        "universe_size": 2,
        "qualified_count": 1,
        "marginal_count": 0,
        "excluded_count": 1,
        "rows": [
            {
                "symbol": "AAPL",
                "earnings_date": "2026-04-23",
                "release_timing": "AMC",
                "entry_date": "2026-04-20",
                "entry_label": "T-3",
                "selected_expiry": "2026-04-24",
                "alternative_expiry": "2026-05-15",
                "expiry_mode": expiry_mode,
                "avg_spread_pct": 2.8,
                "previous_avg_spread_pct": 3.1,
                "spread_change_pct": -0.3,
                "spread_change_state": "improved",
                "call_oi": 420,
                "put_oi": 390,
                "implied_move_pct": 5.2,
                "call_strike": 210.0,
                "put_strike": 198.0,
                "call_iv": 0.41,
                "put_iv": 0.39,
                "entry_debit_mid": 8.4,
                "status": "QUALIFIED",
                "status_reason": "Inside spread and OI thresholds.",
                "compact_signal_summary": ["Avg spread 2.8%", "OI 420/390", "AMC timing"],
                "caveats": [],
                "checks": [
                    {
                        "label": "Release timing",
                        "threshold": "AMC only",
                        "actual": "AMC",
                        "passed": True,
                        "severity": "hard",
                        "note": None,
                    }
                ],
                "notes": ["Selected expiry uses front_after_earnings methodology."],
                "last_updated": "2026-04-17T12:00:00Z",
                "detail_metrics": {"spot": 203.5},
            },
            {
                "symbol": "JPM",
                "earnings_date": "2026-04-21",
                "release_timing": "BMO",
                "entry_date": "2026-04-16",
                "entry_label": "T-3",
                "selected_expiry": "2026-04-24",
                "alternative_expiry": "2026-05-15",
                "expiry_mode": expiry_mode,
                "avg_spread_pct": 4.9,
                "previous_avg_spread_pct": None,
                "spread_change_pct": None,
                "spread_change_state": "new",
                "call_oi": 150,
                "put_oi": 145,
                "implied_move_pct": 4.1,
                "call_strike": 252.0,
                "put_strike": 236.0,
                "call_iv": 0.28,
                "put_iv": 0.29,
                "entry_debit_mid": 4.6,
                "status": "EXCLUDED",
                "status_reason": "AMC-only timing is required.",
                "compact_signal_summary": ["BMO timing"],
                "caveats": [],
                "checks": [],
                "notes": [],
                "last_updated": "2026-04-17T12:00:00Z",
                "detail_metrics": {"spot": 244.1},
            },
        ],
    }


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


class _SampleBottleneckStubCollector:
    def __init__(self):
        self.db = type("StubDB", (), {"db_path": str(Path(tempfile.gettempdir()) / "stub_oos_bottleneck.sqlite")})()

    def run_oos_validation(self, **_kwargs):
        return {
            "splits": np.int64(6),
            "best_params": {},
            "report_card": {
                "ready": True,
                "verdict": {"grade": "D", "overall_pass": False},
                "sample": {"splits": np.int64(6), "total_test_trades": np.int64(32), "avg_trades_per_split": np.float64(5.33)},
                "metrics": {
                    "alpha": {"mean": np.float64(0.14), "low": np.float64(0.03)},
                    "sharpe": {"mean": np.float64(0.66), "low": np.float64(0.11)},
                    "win_rate": {"mean": np.float64(0.58), "low": np.float64(0.49)},
                    "pnl": {"mean": np.float64(42.0), "low": np.float64(8.0)},
                },
                "gates": {
                    "alpha_ci_positive": {"passed": True},
                    "sharpe_ci_positive": {"passed": True},
                    "pnl_ci_positive": {"passed": True},
                    "min_splits": {"required": 8, "actual": 6, "passed": False},
                    "min_total_test_trades": {"required": 80, "actual": 32, "passed": False},
                    "min_trades_per_split": {"required": 5.0, "actual": 5.33, "passed": True},
                },
            },
            "csv_path": Path("exports/reports/oos_bottleneck.csv"),
            "markdown_path": Path("exports/reports/oos_bottleneck.md"),
            "json_path": Path("exports/reports/oos_bottleneck.json"),
            "report_card_markdown_path": Path("exports/reports/oos_bottleneck_report_card.md"),
            "report_card_json_path": Path("exports/reports/oos_bottleneck_report_card.json"),
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

    def test_health_endpoint_remains_public_when_share_auth_enabled(self):
        with patch.object(app_module, "_SHARE_AUTH_ENABLED", True):
            with patch.object(app_module, "_SHARE_PASSWORD", "secret"):
                response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_api_requires_auth_when_share_auth_enabled(self):
        with patch.object(app_module, "_SHARE_AUTH_ENABLED", True):
            with patch.object(app_module, "_SHARE_PASSWORD", "secret"):
                response = self.client.post("/api/edge/analyze", json={"symbol": "AAPL"})

        self.assertEqual(response.status_code, 401)
        self.assertIn("Unauthorized", response.text)

    def test_hosted_auth_config_fails_fast_without_auth(self):
        with patch.object(app_module, "_HOSTED_MODE", True):
            with patch.object(app_module, "_SHARE_AUTH_ENABLED", False):
                with self.assertRaisesRegex(RuntimeError, "HOSTED_MODE"):
                    app_module._validate_auth_config()

    def test_auth_config_requires_strong_session_secret(self):
        with patch.object(app_module, "_HOSTED_MODE", False):
            with patch.object(app_module, "_SHARE_AUTH_ENABLED", True):
                with patch.object(app_module, "_SHARE_PASSWORD", "secret"):
                    with patch.object(app_module, "_SESSION_SECRET", "change-me-in-env"):
                        with self.assertRaisesRegex(RuntimeError, "SESSION_SECRET"):
                            app_module._validate_auth_config()

    def test_api_docs_can_be_protected_by_auth_middleware(self):
        with patch.object(app_module, "_SHARE_AUTH_ENABLED", True):
            with patch.object(app_module, "_SHARE_PASSWORD", "secret"):
                with patch.object(app_module, "_PROTECT_API_DOCS", True):
                    response = self.client.get("/docs")

        self.assertEqual(response.status_code, 401)
        self.assertIn("Unauthorized", response.text)

    def test_login_cookie_uses_secure_flag_when_configured(self):
        with patch.object(app_module, "_SHARE_AUTH_ENABLED", True):
            with patch.object(app_module, "_SHARE_PASSWORD", "secret"):
                with patch.object(app_module, "_SESSION_SECRET", "x" * 32):
                    with patch.object(app_module, "_SECURE_SESSION_COOKIE", True):
                        response = self.client.post(
                            "/login",
                            data={"password": "secret"},
                            follow_redirects=False,
                        )

        self.assertEqual(response.status_code, 303)
        set_cookie = response.headers.get("set-cookie", "")
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("Secure", set_cookie)

    def test_edge_analyze_error_is_sanitized(self):
        with patch.object(app_module, "_get_mda_client", return_value=_make_mock_mda_client()), \
             patch.object(app_module, "analyze_single_ticker", side_effect=RuntimeError("secret-token=/tmp/key")):
            response = self.client.post("/api/edge/analyze", json={"symbol": "AAPL"})

        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertEqual(
            detail,
            "Analysis failed. Check the ticker, data availability, and provider configuration.",
        )
        self.assertNotIn("secret-token", response.text)

    def test_edge_screener_endpoint_returns_rows(self):
        with patch.object(app_module, "build_edge_screener", return_value=_stub_screener_payload()), \
             patch.object(app_module, "_get_mda_client", return_value=_make_mock_mda_client()):
            response = self.client.get("/api/edge/screener")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["expiry_mode"], "front_after_earnings")
        self.assertEqual(payload["qualified_count"], 1)
        self.assertEqual(payload["rows"][0]["symbol"], "AAPL")
        self.assertEqual(payload["rows"][0]["spread_change_state"], "improved")

    def test_edge_screener_endpoint_honors_expiry_mode(self):
        with patch.object(
            app_module,
            "build_edge_screener",
            return_value=_stub_screener_payload(expiry_mode="next_monthly_opex"),
        ) as mock_build, patch.object(
            app_module, "_get_mda_client", return_value=_make_mock_mda_client()
        ):
            response = self.client.get("/api/edge/screener?expiry_mode=next_monthly_opex&weeks=8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["expiry_mode"], "next_monthly_opex")
        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        self.assertEqual(kwargs["expiry_mode"], "next_monthly_opex")
        self.assertEqual(kwargs["weeks"], 8)

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
            with patch.object(app_module, "_get_mda_client", return_value=_make_mock_mda_client()):
                response = self.client.post("/api/edge/analyze", json={"symbol": "AAPL"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["recommendation"], "Watchlist")
        self.assertAlmostEqual(payload["confidence_pct"], 77.7, places=3)
        self.assertEqual(payload["metrics"]["iv_rv30"], 1.31)

    def test_historical_options_symbols_endpoint(self):
        with patch.object(app_module, "_get_feature_store", return_value=_StubFeatureStore()):
            response = self.client.get("/api/historical/options/symbols")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["symbols"], ["SPY"])

    def test_historical_options_coverage_endpoint(self):
        with patch.object(app_module, "_get_feature_store", return_value=_StubFeatureStore()):
            response = self.client.get("/api/historical/options/SPY/coverage")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["coverage"][0]["symbol"], "SPY")
        self.assertEqual(payload["coverage"][0]["rows"], 123)

    def test_historical_options_chain_endpoint(self):
        with patch.object(app_module, "_get_feature_store", return_value=_StubFeatureStore()):
            response = self.client.get("/api/historical/options/SPY/chain?trade_date=2025-01-02&call_put=C")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["symbol"], "SPY")
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["rows"][0]["option_symbol"], "SPY250117C00600000")
        self.assertEqual(payload["rows"][0]["quote_quality_flag"], "ok")

    def test_edge_analyze_new_metrics_in_response(self):
        """Verify earnings_release_time and data_source are included when set."""
        snapshot = EdgeSnapshot(
            symbol="MSFT",
            recommendation="Strong Edge",
            confidence_pct=82.5,
            setup_score=0.78,
            metrics={
                "iv_rv30": 1.35,
                "earnings_release_time": "after market close",
                "data_source": "marketdata_app",
            },
            rationale=["MDApp data sourced"],
        )
        with patch.object(app_module, "analyze_single_ticker", return_value=snapshot):
            with patch.object(app_module, "_get_mda_client", return_value=_make_mock_mda_client()):
                response = self.client.post("/api/edge/analyze", json={"symbol": "MSFT"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["metrics"]["earnings_release_time"], "after market close")
        self.assertEqual(payload["metrics"]["data_source"], "marketdata_app")

    def test_edge_analyze_exposes_selector_contract(self):
        snapshot = EdgeSnapshot(
            symbol="AAPL",
            recommendation="Watchlist",
            confidence_pct=77.7,
            setup_score=0.66,
            metrics={"iv_rv30": 1.31},
            rationale=["signal check"],
            selector_output={
                "symbol": "AAPL",
                "as_of": "2026-04-20",
                "earnings_date": "2026-04-28",
                "release_timing": "after market close",
                "recommendation": "Candidate",
                "best_structure": "atm_straddle",
                "confidence_pct": 71.0,
                "expected_edge_pct": 3.4,
                "expected_return_pct": 6.2,
                "primary_thesis": "ATM straddle leads because realized moves still outrun current implied pricing.",
                "primary_risks": ["Execution risk remains meaningful if spreads widen."],
                "why_this_structure": ["Composite score leads the structure set."],
                "why_not_others": {"call_calendar": ["Composite score trails the winner."]},
                "runner_up_structures": ["otm_strangle", "call_calendar"],
                "data_quality": "high",
                "data_quality_score": 0.91,
            },
            structure_scorecards=[
                {
                    "structure": "atm_straddle",
                    "eligible": True,
                    "eligibility_flags": [],
                    "expected_edge_pct": 3.4,
                    "expected_return_pct": 6.2,
                    "expected_iv_contribution_pct": 2.1,
                    "expected_move_fit_score": 0.74,
                    "theta_drag_penalty": 0.02,
                    "execution_penalty": 0.03,
                    "crowding_penalty": 0.02,
                    "concavity_penalty": 0.01,
                    "sample_uncertainty_penalty": 0.02,
                    "sample_confidence": 0.70,
                    "walk_forward_history_count": 24,
                    "walk_forward_win_rate": 0.58,
                    "walk_forward_avg_return_pct": 6.2,
                    "walk_forward_rank_score": 0.71,
                    "composite_structure_score": 0.72,
                    "rationale_bullets": ["Composite score leads the structure set."],
                }
            ],
            vol_snapshot={
                "symbol": "AAPL",
                "as_of_date": "2026-04-20",
                "earnings_date": "2026-04-28",
            },
        )
        with patch.object(app_module, "analyze_single_ticker", return_value=snapshot):
            with patch.object(app_module, "_get_mda_client", return_value=_make_mock_mda_client()):
                response = self.client.post("/api/edge/analyze", json={"symbol": "AAPL"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["selector_output"]["best_structure"], "atm_straddle")
        self.assertEqual(payload["structure_scorecards"][0]["structure"], "atm_straddle")
        self.assertEqual(payload["vol_snapshot"]["symbol"], "AAPL")

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
        self.assertEqual(kwargs["trades_per_day_grid"], [2])
        self.assertEqual(kwargs["entry_days_grid"], [5, 7])
        self.assertEqual(kwargs["signal_threshold_grid"], [0.65])
        self.assertAlmostEqual(kwargs["min_crush_confidence"], 0.50, places=6)
        self.assertAlmostEqual(kwargs["min_crush_magnitude"], 0.09, places=6)
        self.assertAlmostEqual(kwargs["min_crush_edge"], 0.025, places=6)
        self.assertEqual(kwargs["entry_dte_band"], 4)
        self.assertEqual(kwargs["min_daily_share_volume"], 10_000_000)
        self.assertAlmostEqual(kwargs["max_abs_momentum_5d"], 0.09, places=6)
        self.assertEqual(payload["summary"]["stability_profile_requested"], "variance_control")
        self.assertEqual(payload["summary"]["stability_profile_used"], "variance_control")

    def test_oos_sample_expansion_profile_broadens_search_space(self):
        _ProfileCaptureStubCollector.last_kwargs = None
        with patch.object(app_module, "InstitutionalDataCollector", _ProfileCaptureStubCollector):
            response = self.client.post(
                "/api/oos/report-card",
                json={
                    "oos_stability_profile": "sample_expansion",
                    "min_signal_score": 0.30,
                    "min_crush_confidence": 0.10,
                    "min_crush_magnitude": 0.01,
                    "min_crush_edge": 0.001,
                    "entry_dte_band": 9,
                    "min_daily_share_volume": 100_000,
                    "max_abs_momentum_5d": 0.20,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        kwargs = _ProfileCaptureStubCollector.last_kwargs
        self.assertIsNotNone(kwargs)
        self.assertEqual(kwargs["execution_profiles"], ["institutional"])
        self.assertEqual(kwargs["hold_days_grid"], [1, 3])
        self.assertEqual(kwargs["trades_per_day_grid"], [2, 3])
        self.assertEqual(kwargs["entry_days_grid"], [3, 5, 7])
        self.assertEqual(kwargs["signal_threshold_grid"], [0.45])
        self.assertAlmostEqual(kwargs["min_crush_confidence"], 0.25, places=6)
        self.assertAlmostEqual(kwargs["min_crush_magnitude"], 0.05, places=6)
        self.assertAlmostEqual(kwargs["min_crush_edge"], 0.015, places=6)
        self.assertEqual(kwargs["entry_dte_band"], 6)
        self.assertEqual(kwargs["min_daily_share_volume"], 1_000_000)
        self.assertAlmostEqual(kwargs["max_abs_momentum_5d"], 0.11, places=6)
        self.assertEqual(payload["summary"]["stability_profile_requested"], "sample_expansion")
        self.assertEqual(payload["summary"]["stability_profile_used"], "sample_expansion")

    def test_oos_report_card_flags_sample_bottleneck_note(self):
        with patch.object(app_module, "InstitutionalDataCollector", _SampleBottleneckStubCollector):
            response = self.client.post(
                "/api/oos/report-card",
                json={"oos_stability_profile": "evidence_balanced"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        # Bottleneck note now goes to warnings, not notes
        warnings = payload["summary"].get("warnings", [])
        self.assertTrue(any("sample size is insufficient" in w.lower() for w in warnings))
        self.assertTrue(any("sample_expansion" in w for w in warnings))

    def test_oos_report_card_flags_sparse_alpha_note(self):
        """Sparse alpha rate < 0.5 across enough splits should surface a warning."""
        class _SparseAlphaStubCollector:
            def __init__(self):
                self.db = type("StubDB", (), {"db_path": str(Path(tempfile.gettempdir()) / "stub_oos_sparse.sqlite")})()

            def run_oos_validation(self, **_kwargs):
                return {
                    "splits": np.int64(10),
                    "best_params": {},
                    "report_card": {
                        "verdict": {"grade": "D", "overall_pass": False},
                        "sample": {"splits": np.int64(10), "total_test_trades": np.int64(30), "avg_trades_per_split": np.float64(3.0)},
                        "metrics": {
                            "alpha": {"mean": np.float64(0.30), "low": np.float64(0.05)},
                            "sharpe": {"mean": np.float64(2.1), "low": np.float64(-1.0)},
                            "pnl": {"mean": np.float64(10.0), "low": np.float64(-5.0)},
                            "win_rate": {"mean": np.float64(0.60), "low": np.float64(0.45)},
                            "positive_alpha_split_rate": np.float64(0.30),  # < 0.5 → warning
                            "positive_pnl_split_rate": np.float64(0.30),
                        },
                        "gates": {
                            "alpha_ci_positive": {"passed": True},
                            "sharpe_ci_positive": {"passed": False},
                            "pnl_ci_positive": {"passed": False},
                            "min_splits": {"required": 8, "actual": 10, "passed": True},
                            "min_total_test_trades": {"required": 80, "actual": 30, "passed": False},
                            "min_trades_per_split": {"required": 5.0, "actual": 3.0, "passed": False},
                        },
                    },
                    "csv_path": Path("exports/reports/oos_sparse.csv"),
                    "markdown_path": Path("exports/reports/oos_sparse.md"),
                    "json_path": Path("exports/reports/oos_sparse.json"),
                    "report_card_markdown_path": Path("exports/reports/oos_sparse_report_card.md"),
                    "report_card_json_path": Path("exports/reports/oos_sparse_report_card.json"),
                }

        with patch.object(app_module, "InstitutionalDataCollector", _SparseAlphaStubCollector):
            response = self.client.post(
                "/api/oos/report-card",
                json={"oos_stability_profile": "variance_control"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        warnings = payload["summary"].get("warnings", [])
        self.assertTrue(any("30%" in w or "positive" in w.lower() for w in warnings),
                        f"Expected sparse-alpha warning in {warnings}")

    def test_oos_validation_timeout_returns_gracefully(self):
        """A hung run_oos_validation should return a warning, not block the request."""
        import time

        class _TimeoutStubCollector:
            def __init__(self):
                self.db = type("StubDB", (), {"db_path": str(Path(tempfile.gettempdir()) / "stub_oos_timeout.sqlite")})()

            def run_oos_validation(self, **_kwargs):
                # Simulate a hung call by sleeping longer than the patched timeout
                time.sleep(10)
                return None  # pragma: no cover

        with patch.object(app_module, "_OOS_VALIDATION_TIMEOUT_SECONDS", 0.1):
            with patch.object(app_module, "InstitutionalDataCollector", _TimeoutStubCollector):
                response = self.client.post(
                    "/api/oos/report-card",
                    json={"oos_stability_profile": "variance_control"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"]["status"], "no_oos_rows")
        warnings = payload["summary"].get("warnings", [])
        self.assertTrue(any("did not complete" in w.lower() or "timeout" in w.lower() for w in warnings),
                        f"Expected timeout warning in {warnings}")

    def test_oos_submit_rejects_when_capacity_reached(self):
        with app_module._oos_jobs_lock:
            app_module._oos_jobs.clear()
            app_module._oos_jobs["existing"] = {
                "status": "running",
                "started_at": "2026-04-23T00:00:00+00:00",
                "data": None,
                "error": None,
            }
        try:
            with patch.object(app_module, "_MAX_OOS_RUNNING_JOBS", 1):
                response = self.client.post("/api/oos/submit", json={})
        finally:
            with app_module._oos_jobs_lock:
                app_module._oos_jobs.clear()

        self.assertEqual(response.status_code, 429)
        self.assertIn("capacity", response.json()["detail"].lower())

    def test_ml_train_rejects_when_capacity_reached(self):
        with app_module._ml_train_jobs_lock:
            app_module._ml_train_jobs.clear()
            app_module._ml_train_jobs["existing"] = {
                "status": "pending",
                "started_at": "2026-04-23T00:00:00+00:00",
                "data": None,
                "error": None,
            }
        try:
            with patch.object(app_module, "_MAX_ML_RUNNING_JOBS", 1):
                response = self.client.post("/api/ml/train")
        finally:
            with app_module._ml_train_jobs_lock:
                app_module._ml_train_jobs.clear()

        self.assertEqual(response.status_code, 429)
        self.assertIn("capacity", response.json()["detail"].lower())

    def test_background_job_errors_are_sanitized(self):
        with app_module._oos_jobs_lock:
            app_module._oos_jobs.clear()
            app_module._oos_jobs["job-1"] = {
                "status": "pending",
                "started_at": "2026-04-23T00:00:00+00:00",
                "data": None,
                "error": None,
            }

        try:
            with patch.object(app_module, "_execute_oos_logic", side_effect=RuntimeError("secret filesystem path")):
                app_module._oos_job_worker("job-1", app_module.OOSReportRequest())
            with app_module._oos_jobs_lock:
                job = dict(app_module._oos_jobs["job-1"])
        finally:
            with app_module._oos_jobs_lock:
                app_module._oos_jobs.clear()

        self.assertEqual(job["status"], "error")
        self.assertEqual(job["error"], "OOS job failed.")
        self.assertNotIn("secret", job["error"])


    # ── Ranked screener endpoint ──────────────────────────────────────────────

    def test_ranked_screener_endpoint_returns_response_shape(self):
        """GET /api/screener/ranked should return RankedScreenerResponse shape."""
        from datetime import date as _date

        stub_payload = {
            "generated_at": "2026-04-20T10:00:00Z",
            "as_of_date": str(_date.today()),
            "universe_size": 40,
            "rows_returned": 2,
            "in_entry_window": 2,
            "ranking_weights": {
                "iv_entry_score": 0.32,
                "move_history_score": 0.25,
                "ts_score": 0.18,
                "dte_score": 0.12,
                "sample_score": 0.08,
                "liquidity_score": 0.05,
            },
            "strategy_note": "Pre-earnings long-vega: enter 3-10 DTE, exit T-1 before event.",
            "rows": [
                {
                    "rank": 1,
                    "symbol": "NVDA",
                    "earnings_date": str(_date.today()),
                    "dte": 5,
                    "release_timing": "AMC",
                    "iv_rv_ratio": 0.92,
                    "atm_iv": 55.2,
                    "rv30": 60.0,
                    "ts_ratio": 0.88,
                    "median_earnings_move_pct": 9.5,
                    "sample_size": 10,
                    "spread_pct": 1.8,
                    "ranking_score": 0.74,
                    "score_components": {"iv_entry_score": 0.8, "dte_score": 0.9},
                    "status": "ranked",
                    "error_note": None,
                },
                {
                    "rank": 2,
                    "symbol": "META",
                    "earnings_date": str(_date.today()),
                    "dte": 7,
                    "release_timing": "AMC",
                    "iv_rv_ratio": 1.10,
                    "atm_iv": 42.0,
                    "rv30": 38.0,
                    "ts_ratio": 0.95,
                    "median_earnings_move_pct": 6.2,
                    "sample_size": 8,
                    "spread_pct": 2.1,
                    "ranking_score": 0.55,
                    "score_components": {},
                    "status": "ranked",
                    "error_note": None,
                },
            ],
        }

        with patch("services.screener_service.build_ranked_screener", return_value=stub_payload):
            response = self.client.get("/api/screener/ranked")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("rows", payload)
        self.assertEqual(payload["universe_size"], 40)
        self.assertEqual(payload["in_entry_window"], 2)
        self.assertIn("ranking_weights", payload)
        self.assertAlmostEqual(payload["ranking_weights"]["iv_entry_score"], 0.32)

    def test_calibration_curve_endpoint_returns_ten_buckets(self):
        """GET /api/calibration/curve should return 10 score buckets."""
        from services.calibration_service import IVExpansionCalibration
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            store = pathlib.Path(tmpdir) / "iv_expansion.json"
            fresh_cal = IVExpansionCalibration(store_path=store)

            with patch("services.calibration_service.get_calibration", return_value=fresh_cal):
                response = self.client.get("/api/calibration/curve")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["phase"], "bootstrap_prior")
        self.assertEqual(payload["n_observations"], 0)
        self.assertEqual(payload["min_for_observational"], 40)
        self.assertEqual(len(payload["buckets"]), 10)
        # All prior buckets should flag prior_only=True
        self.assertTrue(all(b["prior_only"] for b in payload["buckets"]))
        # Score range covers 0.0 to 1.0
        self.assertAlmostEqual(payload["buckets"][0]["score_lo"], 0.0)
        self.assertAlmostEqual(payload["buckets"][-1]["score_hi"], 1.0)

    def test_calibration_curve_prior_estimates_increase_with_score(self):
        """Bootstrap prior estimates must be monotone increasing across buckets."""
        from services.calibration_service import IVExpansionCalibration
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            store = pathlib.Path(tmpdir) / "iv_expansion.json"
            fresh_cal = IVExpansionCalibration(store_path=store)

            with patch("services.calibration_service.get_calibration", return_value=fresh_cal):
                response = self.client.get("/api/calibration/curve")

        buckets = response.json()["buckets"]
        exps = [b["expected_expansion_pct"] for b in buckets]
        # Allow ties but not reversals — monotone non-decreasing
        for i in range(len(exps) - 1):
            self.assertLessEqual(
                exps[i], exps[i + 1] + 0.01,
                f"Prior not monotone at bucket {i}: {exps[i]:.2f} > {exps[i+1]:.2f}",
            )

    def test_learning_diagnostics_endpoint_returns_expected_shape(self):
        stub = {
            "calibration": {
                "phase": "observational",
                "n_total": 55,
                "n_replay": 40,
                "n_synthetic": 5,
                "n_paper": 10,
                "n_live": 0,
                "is_prior_only": False,
                "score_distribution": {"min": 0.2, "max": 0.8, "mean": 0.54},
                "expansion_distribution": {"min": -1.0, "max": 12.0, "mean": 5.2},
            },
            "structure_priors": {
                "atm_straddle": {"count": 3, "win_rate": 0.5, "avg_return_pct": 1.2, "avg_expansion_pct": 2.1},
                "otm_strangle": {"count": 1, "win_rate": 0.0, "avg_return_pct": -0.8, "avg_expansion_pct": -0.4},
                "call_calendar": {"count": 9, "win_rate": 0.56, "avg_return_pct": 2.2, "avg_expansion_pct": 3.1},
                "put_calendar": {"count": 0, "win_rate": 0.0, "avg_return_pct": 0.0, "avg_expansion_pct": 0.0},
            },
            "data_quality": {
                "has_real_data": True,
                "replay_dominant": True,
                "synthetic_ratio": 0.0909,
                "paper_ratio": 0.1818,
            },
            "learning_health": {
                "calibration_stable": False,
                "sufficient_observations": False,
                "warning_flags": ["Structure priors imbalanced"],
            },
        }

        with patch("services.learning_diagnostics.build_learning_diagnostics", return_value=stub):
            response = self.client.get("/api/diagnostics/learning")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["calibration"]["phase"], "observational")
        self.assertEqual(payload["calibration"]["n_total"], 55)
        self.assertIn("call_calendar", payload["structure_priors"])
        self.assertTrue(payload["data_quality"]["has_real_data"])
        self.assertIn("warning_flags", payload["learning_health"])

    def test_data_quality_diagnostics_endpoint_returns_expected_shape(self):
        stub = {
            "generated_at": "2026-04-24T12:00:00+00:00",
            "total_recommendations": 3,
            "provider_health": {
                "option_source": {"success": 2, "failure": 1},
                "underlying_source": {"success": 3, "failure": 0},
                "earnings_source": {"success": 2, "failure": 1, "stale": 1},
                "quote_source": {"success": 2, "failure": 1},
            },
            "stale_earnings_source_count": 1,
            "stale_earnings_source_rate": 0.333333,
            "missing_option_chain_count": 1,
            "low_data_quality_count": 2,
            "low_data_quality_rate": 0.666667,
            "data_quality_buckets": {
                "0.00-0.25": 1,
                "0.25-0.50": 1,
                "0.50-0.75": 0,
                "0.75-1.00": 1,
                "unknown": 0,
            },
            "source_breakdown": {
                "option_source": {"marketdata_app": 2, "unknown": 1},
                "underlying_source": {"yfinance": 3},
                "earnings_source": {"alpha_vantage": 2, "unknown": 1},
                "quote_source": {"yfinance": 2, "unknown": 1},
            },
            "recent_weak_data_recommendations": [
                {
                    "recommendation_id": "rec_weak",
                    "symbol": "TSLA",
                    "data_quality_score": 0.2,
                    "earnings_source_stale": True,
                    "weak_data_reasons": ["low_data_quality_score", "stale_earnings_source"],
                }
            ],
            "warning_flags": ["Low data-quality recommendation rate is elevated."],
            "thresholds": {"low_data_quality_score": 0.5},
        }

        with patch("services.data_quality_diagnostics.build_data_quality_diagnostics", return_value=stub):
            response = self.client.get("/api/diagnostics/data-quality")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_recommendations"], 3)
        self.assertEqual(payload["provider_health"]["option_source"]["failure"], 1)
        self.assertEqual(payload["data_quality_buckets"]["0.00-0.25"], 1)
        self.assertEqual(payload["source_breakdown"]["option_source"]["unknown"], 1)
        self.assertIn("warning_flags", payload)

    def test_provider_telemetry_endpoint_returns_expected_shape(self):
        stub = {
            "generated_at": "2026-04-24T12:00:00+00:00",
            "totals": {
                "events": 3,
                "failures": 1,
                "fallback_count": 1,
                "stale_count": 1,
                "failure_rate": 0.333333,
            },
            "summary_by_provider": {
                "marketdata_app": {
                    "total": 2,
                    "successes": 1,
                    "failures": 1,
                    "failure_rate": 0.5,
                    "avg_latency_ms": 180.0,
                    "fallback_count": 1,
                    "stale_count": 0,
                }
            },
            "recent_events": [
                {
                    "provider_name": "marketdata_app",
                    "endpoint_type": "options_chain",
                    "symbol": "AAPL",
                    "success": True,
                    "latency_ms": 120.0,
                }
            ],
            "recent_failures": [
                {
                    "provider_name": "marketdata_app",
                    "endpoint_type": "earnings",
                    "symbol": "MSFT",
                    "success": False,
                    "error_category": "rate_limited",
                }
            ],
            "limit": 25,
            "offset": 5,
            "has_more": False,
            "filters": {
                "provider": "marketdata_app",
                "endpoint_type": "earnings",
                "symbol": "AAPL",
                "success": False,
                "failures_only": True,
                "since": "2026-04-24T00:00:00+00:00",
                "until": None,
            },
            "operational_health": {
                "warning_flags": ["Provider failure rate is elevated."],
                "avg_latency_ms": 180.0,
                "db_size_bytes": 4096,
                "row_count": 3,
                "retention": {"max_age_days": 30, "max_rows": 50000},
            },
        }

        with patch("services.provider_telemetry.build_provider_telemetry_diagnostics", return_value=stub) as mock_build:
            response = self.client.get(
                "/api/diagnostics/provider-telemetry"
                "?limit=25&offset=5&provider=marketdata_app&endpoint_type=earnings"
                "&symbol=AAPL&success=false&failures_only=true&since=2026-04-24T00:00:00%2B00:00"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["totals"]["events"], 3)
        self.assertEqual(payload["summary_by_provider"]["marketdata_app"]["failure_rate"], 0.5)
        self.assertEqual(payload["recent_failures"][0]["error_category"], "rate_limited")
        self.assertEqual(payload["filters"]["provider"], "marketdata_app")
        self.assertEqual(payload["operational_health"]["row_count"], 3)
        _, kwargs = mock_build.call_args
        self.assertEqual(kwargs["limit"], 25)
        self.assertEqual(kwargs["offset"], 5)
        self.assertEqual(kwargs["provider"], "marketdata_app")
        self.assertEqual(kwargs["endpoint_type"], "earnings")
        self.assertEqual(kwargs["symbol"], "AAPL")
        self.assertFalse(kwargs["success"])
        self.assertTrue(kwargs["failures_only"])

    def test_forward_performance_diagnostics_endpoint_returns_expected_shape(self):
        stub = {
            "generated_at": "2026-04-24T00:00:00+00:00",
            "evidence_label": "paper_research_not_execution_grade",
            "notes": ["Forward results are paper/research diagnostics unless explicitly labeled live."],
            "thresholds": {
                "low_quality_threshold": 0.6,
                "high_quality_threshold": 0.75,
                "min_sample_for_stable_read": 30,
            },
            "total_recommendations": 3,
            "no_trade_count": 1,
            "open_outcome_count": 0,
            "resolved_outcome_count": 2,
            "performance_summary": {
                "n": 2,
                "wins": 1,
                "losses": 1,
                "win_rate": 0.5,
                "avg_modeled_return_pct": 2.0,
                "avg_realized_return_pct": 0.75,
                "avg_model_error_pct": -1.25,
                "avg_realized_expansion_pct": 3.0,
                "paper_research_label": "paper/research outcomes, not execution-grade live fills",
            },
            "by_structure": {"atm_straddle": {"n": 2, "wins": 1, "losses": 1, "win_rate": 0.5}},
            "stale_source_comparison": {"fresh_or_unknown_source": {"n": 1}, "stale_source": {"n": 1}},
            "data_quality_comparison": {"high_quality": {"n": 1}, "low_quality": {"n": 1}},
            "evidence_quality_comparison": {"degraded_evidence": {"n": 2}},
            "surface_quality_comparison": {"clean_surface": {"n": 1}, "degraded_surface": {"n": 1}},
            "spread_cost_buckets": {"5-15pct_spread_cost": {"n": 1}},
            "execution_scenario_comparison": {"mid": {"n": 2}},
            "calibration_report": {
                "selector_score_bucket": {"0.75-0.90": {"n": 1}},
                "surface_quality_status": {"degraded_surface": {"n": 1}},
            },
            "benchmark_comparison": {
                "selector": {"n": 2},
                "always_atm_straddle": {"n": 1},
                "no_trade": {"n": 2, "avg_realized_return_pct": 0.0},
                "paper_research_label": "Benchmarks are observational paper/research comparisons and are not optimized.",
            },
            "claimable_performance": {"non_claimable": {"n": 2}},
            "claimable_evidence": {"claimable_count": 0, "non_claimable_count": 2, "execution_grade_count": 0},
            "outcome_count_by_symbol": {"AAPL": 1, "MSFT": 1},
            "calibration_buckets": [
                {
                    "bucket": "75-90",
                    "bucket_type": "selector_confidence_pct_or_setup_score",
                    "n": 1,
                    "win_rate": 1.0,
                    "avg_confidence_pct": 82.0,
                    "avg_setup_score": 0.82,
                    "avg_modeled_return_pct": 2.0,
                    "avg_realized_return_pct": 4.0,
                    "avg_model_error_pct": 2.0,
                }
            ],
            "recent_resolved_outcomes": [
                {
                    "trade_id": "AAPL|2026-04-24|atm_straddle",
                    "recommendation_id": "rec_aapl",
                    "symbol": "AAPL",
                    "structure": "atm_straddle",
                    "source_type": "paper",
                    "modeled_return_pct": 2.0,
                    "realized_return_pct": 4.0,
                }
            ],
            "warning_flags": ["Forward-performance results are paper/research diagnostics, not execution-grade live performance."],
        }

        with patch("services.forward_performance_diagnostics.build_forward_performance_diagnostics", return_value=stub) as mock_build:
            response = self.client.get("/api/diagnostics/forward-performance?limit=500&recent_limit=10")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["evidence_label"], "paper_research_not_execution_grade")
        self.assertEqual(payload["calibration_report"]["surface_quality_status"]["degraded_surface"]["n"], 1)
        self.assertEqual(payload["benchmark_comparison"]["no_trade"]["avg_realized_return_pct"], 0.0)
        self.assertEqual(payload["claimable_performance"]["non_claimable"]["n"], 2)
        mock_build.assert_called_once_with(max_rows=500, recent_limit=10)

    def test_evidence_report_endpoint_returns_expected_shape(self):
        stub = {
            "evidence_label": "paper_research_not_execution_grade",
            "maturity": {
                "maturity_label": "Insufficient evidence",
                "edge_quality_label_allowed": False,
                "benchmark_comparison_meaningful": False,
            },
            "commercialization_gate": {
                "active_evidence_days": 12,
                "ready_for_paid_beta": False,
            },
            "selector_summary": {"n": 2},
            "baseline_comparison": {"no_trade": {"n": 2}},
            "warning_flags": ["Resolved selector sample is small (n=2); do not infer durable edge yet."],
        }
        with patch("services.evidence_report.build_evidence_report", return_value=stub) as mock_build:
            response = self.client.get("/api/diagnostics/evidence-report?limit=500&recent_limit=10")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["evidence_label"], "paper_research_not_execution_grade")
        self.assertFalse(payload["commercialization_gate"]["ready_for_paid_beta"])
        self.assertEqual(payload["maturity"]["maturity_label"], "Insufficient evidence")
        mock_build.assert_called_once_with(max_rows=500, recent_limit=10)

    def test_recommendation_ledger_endpoints_expose_recent_detail_linkage_and_summary(self):
        from datetime import date as _date
        from services.outcome_recorder import OutcomeStore
        from services.recommendation_ledger import RecommendationLedger, record_recommendation
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            ledger = RecommendationLedger(root / "ledger.sqlite")
            outcomes = OutcomeStore(root / "outcomes.sqlite")

            analysis = EdgeSnapshot(
                symbol="AAPL",
                recommendation="Candidate",
                confidence_pct=72.0,
                setup_score=0.66,
                metrics={},
                rationale=["ledger rationale"],
                selector_output={
                    "recommendation": "Candidate",
                    "best_structure": "atm_straddle",
                    "earnings_date": "2026-05-01",
                    "primary_thesis": "Structure fits the snapshot.",
                    "primary_risks": ["Execution risk."],
                    "why_this_structure": ["Move evidence leads."],
                    "why_not_others": {"call_calendar": ["Lower score."]},
                },
                structure_scorecards=[
                    {"structure": "atm_straddle", "eligible": True, "composite_structure_score": 0.8}
                ],
                vol_snapshot={
                    "symbol": "AAPL",
                    "as_of_date": "2026-04-23",
                    "earnings_date": "2026-05-01",
                    "earnings_source_primary": "alpha_vantage",
                    "earnings_source_confidence": 0.82,
                    "earnings_source_stale": True,
                    "option_source": "marketdata_app",
                    "underlying_source": "yfinance",
                    "data_quality_score": 0.77,
                },
            )
            rec_id = record_recommendation(
                analysis,
                ledger=ledger,
                recommendation_id="rec_api_candidate",
                quote_payload={
                    "quote_source": "yfinance",
                    "quote_quality": "paper_research_mid_not_execution_grade",
                    "bid_ask_mid": {"legs": {"call": {"mid": 4.1}}},
                },
            )
            outcomes.insert_entry(
                trade_id="AAPL|2026-04-23|atm_straddle",
                recommendation_id=rec_id,
                symbol="AAPL",
                structure="atm_straddle",
                entry_date=_date(2026, 4, 23),
                setup_score=0.66,
                source_type="paper",
                earnings_date=_date(2026, 5, 1),
                entry_mid=4.1,
            )
            record_recommendation(
                EdgeSnapshot(
                    symbol="MSFT",
                    recommendation="No Trade",
                    confidence_pct=24.0,
                    setup_score=0.22,
                    metrics={},
                    rationale=["abstain"],
                    selector_output={
                        "recommendation": "No Trade",
                        "best_structure": None,
                        "earnings_date": "2026-05-08",
                        "primary_thesis": "Selector abstained.",
                        "primary_risks": ["Data quality too low."],
                        "why_this_structure": [],
                        "why_not_others": {},
                    },
                    structure_scorecards=[],
                    vol_snapshot={
                        "symbol": "MSFT",
                        "as_of_date": "2026-04-23",
                        "earnings_date": "2026-05-08",
                        "earnings_source_primary": "fmp",
                        "earnings_source_confidence": 0.35,
                        "earnings_source_stale": False,
                        "option_source": "marketdata_app",
                        "underlying_source": "yfinance",
                        "data_quality_score": 0.31,
                    },
                ),
                ledger=ledger,
                recommendation_id="rec_api_no_trade",
            )

            with patch("services.recommendation_ledger.get_recommendation_ledger", return_value=ledger):
                with patch("services.outcome_recorder.get_outcome_store", return_value=outcomes):
                    recent = self.client.get("/api/diagnostics/recommendations")
                    by_symbol = self.client.get("/api/diagnostics/recommendations/symbol/AAPL")
                    page_two = self.client.get("/api/diagnostics/recommendations?limit=1&offset=1")
                    detail = self.client.get(f"/api/diagnostics/recommendations/{rec_id}")
                    linkage = self.client.get(f"/api/diagnostics/recommendations/{rec_id}/linkage")
                    export_json = self.client.get("/api/diagnostics/recommendations/export?format=json&limit=2")
                    export_csv = self.client.get("/api/diagnostics/recommendations/export?format=csv&limit=2")
                    linkage_export = self.client.get("/api/diagnostics/recommendations/linkage/export?format=json&limit=2")
                    summary = self.client.get("/api/diagnostics/recommendations/summary")

        self.assertEqual(recent.status_code, 200)
        recent_rows = {row["recommendation_id"]: row for row in recent.json()["rows"]}
        self.assertEqual(recent.json()["count"], 2)
        self.assertEqual(recent.json()["total"], 2)
        self.assertFalse(recent.json()["has_more"])
        self.assertEqual(recent_rows[rec_id]["outcome_status"], "open")
        self.assertEqual(recent_rows["rec_api_no_trade"]["recommendation"], "No Trade")
        self.assertEqual(by_symbol.status_code, 200)
        self.assertEqual(by_symbol.json()["symbol"], "AAPL")
        self.assertEqual(by_symbol.json()["count"], 1)
        self.assertEqual(page_two.status_code, 200)
        self.assertEqual(page_two.json()["limit"], 1)
        self.assertEqual(page_two.json()["offset"], 1)
        self.assertEqual(page_two.json()["count"], 1)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["quote_provenance"]["quote_quality"], "paper_research_mid_not_execution_grade")
        self.assertTrue(detail.json()["recommendation"]["earnings_source_stale"])
        self.assertEqual(linkage.status_code, 200)
        self.assertTrue(linkage.json()["has_linked_outcome"])
        self.assertEqual(linkage.json()["paper_trade"]["recommendation_id"], rec_id)
        self.assertEqual(export_json.status_code, 200)
        self.assertEqual(export_json.json()["export_type"], "recommendations")
        self.assertEqual(export_json.json()["count"], 2)
        self.assertEqual(export_csv.status_code, 200)
        self.assertIn("recommendation_id", export_csv.text)
        self.assertIn("text/csv", export_csv.headers.get("content-type", ""))
        self.assertEqual(linkage_export.status_code, 200)
        self.assertEqual(linkage_export.json()["export_type"], "recommendation_linkages")
        self.assertTrue(any(row["has_linked_outcome"] for row in linkage_export.json()["rows"]))
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["summary"]["by_selected_structure"]["atm_straddle"], 1)
        self.assertEqual(summary.json()["summary"]["by_no_trade_reason"]["Data quality too low."], 1)
        self.assertEqual(summary.json()["summary"]["by_stale_source_flag"]["stale"], 1)
        self.assertEqual(summary.json()["summary"]["by_earnings_source"]["fmp"], 1)

    def test_recommendation_ledger_detail_returns_404_for_unknown_id(self):
        from services.recommendation_ledger import RecommendationLedger
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = RecommendationLedger(pathlib.Path(tmpdir) / "ledger.sqlite")
            with patch("services.recommendation_ledger.get_recommendation_ledger", return_value=ledger):
                response = self.client.get("/api/diagnostics/recommendations/does-not-exist")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
