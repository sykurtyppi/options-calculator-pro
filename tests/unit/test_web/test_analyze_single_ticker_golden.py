"""
Phase 4.0 — golden-master / characterization test for analyze_single_ticker.

Pins the FULL orchestrator output (recommendation, confidence_pct, setup_score,
sample_provenance, rationale, and all 150 non-volatile metrics keys) for a fixed,
fully-mocked scenario. Phase 4 decomposes analyze_single_ticker (build_analysis_inputs,
gates.py, rationale.py, metrics_builder.py, ...) — every step claims to be
behavior-preserving. This test makes that claim VERIFIABLE across the entire output
surface, not just the ~13 keys other tests happen to assert.

If a Phase 4 PR changes output INTENTIONALLY, regenerate the fixture and review the
JSON diff in the PR. An UNINTENTIONAL change fails here with the first differing path.

Excluded (non-deterministic, not computed output):
  - generated_at                  : wall-clock timestamp
  - pricing_dividend_yield_source : cache-warmth provenance label (value is stable)
"""
from __future__ import annotations

import json
import math
import pathlib
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import StructureScorecard
from services.structure_selector import SelectorOutput
import web.api.edge_engine as edge_engine

_FIXTURE = pathlib.Path(__file__).parent / "golden" / "analyze_watch.json"
VOLATILE = {"generated_at", "pricing_dividend_yield_source"}


def _canon(o):
    if isinstance(o, float):
        return "NaN" if math.isnan(o) else (round(o, 8) if math.isfinite(o) else str(o))
    if isinstance(o, dict):
        return {k: _canon(o[k]) for k in sorted(o, key=str)}
    if isinstance(o, (list, tuple)):
        return [_canon(x) for x in o]
    return o


def _to_golden(r):
    metrics = {k: v for k, v in r.metrics.items() if k not in VOLATILE}
    return {
        "recommendation": r.recommendation,
        "confidence_pct": _canon(r.confidence_pct),
        "setup_score": _canon(r.setup_score),
        "sample_provenance": _canon(getattr(r, "sample_provenance", None)),
        "rationale": _canon(r.rationale),
        "metrics": _canon(metrics),
    }


def _first_diff(a, b, path=""):
    """Return a human-readable description of the first structural difference, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                return f"{path}.{k}: missing in actual"
            if k not in b:
                return f"{path}.{k}: missing in golden"
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: list length {len(a)} (actual) != {len(b)} (golden)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b:
        return f"{path}: {a!r} (actual) != {b!r} (golden)"
    return None


def _run_watch_scenario():
    """Fully-mocked analyze_single_ticker call — deterministic, no network/provider."""
    hist_dates = pd.bdate_range("2024-01-02", periods=140)
    hist_df = pd.DataFrame({
        "Open": np.linspace(100.0, 102.0, len(hist_dates)),
        "High": np.linspace(101.0, 103.0, len(hist_dates)),
        "Low": np.linspace(99.0, 101.0, len(hist_dates)),
        "Close": np.linspace(100.2, 102.2, len(hist_dates)),
        "Volume": np.full(len(hist_dates), 5_000_000.0),
    }, index=hist_dates)
    ticker = MagicMock()
    ticker.history.side_effect = [hist_df.iloc[-126:].copy(), hist_df.copy()]
    ticker.info = {}
    ticker.options = []
    ticker.get_earnings_dates.return_value = pd.DataFrame()
    ticker.earnings_dates = None
    snapshot = VolSnapshot(
        symbol="AAPL", as_of_date=datetime(2024, 7, 1).date(), earnings_date=datetime(2024, 7, 9).date(),
        release_timing="after market close", days_to_earnings=8, underlying_price=102.2,
        option_source="provided", underlying_source="provided", price_staleness_minutes=0,
        chain_staleness_minutes=0, data_quality="high", data_quality_score=0.91,
        rv30_yang_zhang=0.21, rv30_estimator="yang_zhang", rv_har_forecast=0.19, rv_percentile_rank=48.0,
        vol_regime_label="Normal", iv30=0.27, iv45=0.30, near_term_dte=4, near_term_atm_iv=0.26,
        back_term_dte=25, back_term_atm_iv=0.30, near_back_iv_ratio=0.8667, term_structure_slope=0.0015,
        near_term_implied_move_pct=5.7, near_term_implied_sigma_pct=5.7 * 1.2533141373155001,
        non_event_move_pct_har=1.0, event_implied_move_pct=5.61, event_move_share_of_total=0.98,
        historical_event_count=5, historical_median_move_pct=4.8, historical_avg_last4_move_pct=5.1,
        historical_p90_move_pct=6.9, historical_move_std_pct=1.4, historical_move_anchor_pct=4.995,
        historical_move_uncertainty_pct=0.80, historical_vs_implied_move_ratio=0.89,
        tail_vs_implied_move_ratio=1.23, smile_curvature=-0.62, smile_concavity_flag=True, smile_points=7,
        near_term_spread_pct=3.2, near_term_liquidity_proxy=2600.0, atm_call_spread_pct=3.0,
        atm_put_spread_pct=3.4, atm_total_open_interest=1700.0, atm_total_volume=900.0, liquidity_tier="low",
        iv_rv_yz=1.286, iv_rv_har=1.421, cheapness_score=0.41, event_risk_score=0.67, execution_score=0.72,
        timing_score=0.73, historical_move_source="earnings_history", null_reasons={},
        earnings_source_primary="fmp_calendar", earnings_source_confirmed="sec_submissions",
        earnings_source_confidence=0.90, release_timing_source="sec_submissions",
    )
    scorecards = [StructureScorecard(
        structure="atm_straddle", eligible=True, eligibility_flags=[], expected_edge_pct=2.4,
        expected_return_pct=4.8, expected_iv_contribution_pct=1.6, expected_move_fit_score=0.61,
        theta_drag_penalty=0.03, execution_penalty=0.05, crowding_penalty=0.06, concavity_penalty=0.03,
        sample_uncertainty_penalty=0.06, sample_confidence=0.22, walk_forward_history_count=4,
        walk_forward_win_rate=0.50, walk_forward_avg_return_pct=0.8, walk_forward_rank_score=0.42,
        composite_structure_score=0.58, rationale_bullets=["Shared selector lead."])]
    selector = SelectorOutput(
        symbol="AAPL", as_of=datetime(2024, 7, 1).date(), earnings_date=datetime(2024, 7, 9).date(),
        release_timing="after market close", recommendation="Watch", best_structure="atm_straddle",
        confidence_pct=47.0, expected_edge_pct=2.4, expected_return_pct=4.8, expected_edge_tier="Positive",
        expected_return_signal="Supportive", model_output_note="note",
        primary_thesis="Shared thesis.", primary_risks=["Execution risk."], why_this_structure=["Shared lead."],
        why_not_others={"otm_strangle": ["Trails winner."]}, runner_up_structures=["otm_strangle"],
        data_quality="high", data_quality_score=0.91)
    with patch.object(edge_engine.yf, "Ticker", return_value=ticker), \
         patch.object(edge_engine, "build_vol_snapshot", return_value=snapshot), \
         patch.object(edge_engine, "build_structure_scorecards", return_value=scorecards), \
         patch.object(edge_engine, "select_best_structure", return_value=selector), \
         patch.object(edge_engine, "_term_structure_points_yf", return_value=([4.0, 25.0], [0.26, 0.30], np.nan, np.nan, 0.0, 0.0, None, None, None, None, None, 4.0, 25.0)), \
         patch.object(edge_engine, "_historical_earnings_move_profile", return_value={"event_count": 5, "median_move_pct": 4.8, "p90_move_pct": 6.9, "avg_last4_move_pct": 5.1, "std_move_pct": 1.4, "raw_moves_pct": [4.2, 4.9, 5.1, 5.4, 6.9], "source": "earnings_history"}), \
         patch.object(edge_engine, "_summarize_pre_earnings_expansion", return_value={"available": False, "status": "no_simulations", "priceable_trades": 0, "ranking_score": None}), \
         patch.object(edge_engine, "_get_feature_store", return_value=None), \
         patch.object(edge_engine, "_get_pricing_risk_free_rate", return_value=(0.04, "test")), \
         patch.object(edge_engine, "get_dividend_yield", return_value=(0.0, "fallback_zero")), \
         patch("services.calibration_service.get_calibration") as gc:
        gc.return_value.diagnostics.return_value = {"phase": "observational", "n_observations": 58, "min_for_observational": 40, "min_for_fit": 120}
        return edge_engine.analyze_single_ticker("AAPL", mda_client=None, record_to_ledger=False)


class TestAnalyzeSingleTickerGolden(unittest.TestCase):
    def test_watch_scenario_matches_golden(self):
        actual = _to_golden(_run_watch_scenario())
        golden = json.loads(_FIXTURE.read_text())
        diff = _first_diff(actual, golden)
        self.assertIsNone(
            diff,
            "analyze_single_ticker output diverged from the golden master.\n"
            f"First difference: {diff}\n"
            "If this change is INTENTIONAL, regenerate tests/unit/test_web/golden/"
            "analyze_watch.json and review the diff in your PR.",
        )

    def test_golden_pins_full_metrics_surface(self):
        """Guard against the golden silently shrinking — pins the key count."""
        golden = json.loads(_FIXTURE.read_text())
        self.assertEqual(len(golden["metrics"]), 150,
                         "golden metrics key count changed; confirm intentional")


if __name__ == "__main__":
    unittest.main()
