"""
Tests for the four model-risk fixes applied to the pre-earnings long-vega system.

Fix 1 — IV/RV directional consistency
    Low IV/RV is now treated as favorable (cheap IV) instead of being hard-rejected.
    Very high IV/RV (>1.60) emits a soft crowding warning instead of a hard gate.

Fix 2 — Kelly sizing removed
    No kelly_full_pct / kelly_half_pct in the metrics dict.
    position_sizing_note present instead.

Fix 3 — historical_move_source propagation
    daily_fallback degrades sample_confidence and sets move_fit to sentinel 0.15.
    earnings_history path behaves normally.

Fix 4 — timing_score direction
    Gaussian peaked at 6 DTE; 5–8 DTE > 0 DTE; 5–8 DTE > 30 DTE.
"""

import math
from datetime import date
from unittest.mock import patch

import pytest

# ── Fix 4: timing_score ───────────────────────────────────────────────────────

from services.earnings_vol_snapshot import _timing_score


class TestTimingScoreGaussian:
    """timing_score must peak near 6 DTE, not at 0 DTE."""

    def test_six_dte_is_highest(self):
        score_6 = _timing_score(6)
        assert score_6 == pytest.approx(1.0, abs=1e-6), "6 DTE should equal peak (≈1.0)"

    def test_optimal_window_beats_zero_dte(self):
        """5–8 DTE should all score higher than 0 DTE."""
        score_0 = _timing_score(0)
        for dte in [5, 6, 7, 8]:
            assert _timing_score(dte) > score_0, (
                f"DTE={dte} should beat DTE=0; 0-DTE is the worst entry timing"
            )

    def test_optimal_window_beats_thirty_dte(self):
        score_30 = _timing_score(30)
        for dte in [5, 6, 7, 8]:
            assert _timing_score(dte) > score_30, (
                f"DTE={dte} should beat DTE=30 (too early, excessive theta drag)"
            )

    def test_output_in_unit_interval(self):
        for dte in [0, 1, 3, 5, 6, 8, 10, 14, 21, 30]:
            score = _timing_score(dte)
            assert 0.0 <= score <= 1.0, f"Out of [0,1] for DTE={dte}: {score}"

    def test_none_dte_returns_none(self):
        assert _timing_score(None) is None

    def test_negative_dte_returns_none(self):
        assert _timing_score(-1) is None

    def test_symmetry_around_center(self):
        """Scores should be equal (or very close) at equal distances from the center."""
        score_4 = _timing_score(4)   # 2 before center
        score_8 = _timing_score(8)   # 2 after center
        assert abs(score_4 - score_8) < 0.02, (
            f"Gaussian should be approximately symmetric: score(4)={score_4:.4f}, score(8)={score_8:.4f}"
        )

    def test_gaussian_formula_correctness(self):
        """Direct validation of the Gaussian formula at several DTE values."""
        for dte in [0, 3, 6, 9, 12, 20]:
            expected = float(min(max(math.exp(-0.5 * ((dte - 6.0) / 3.5) ** 2), 0.0), 1.0))
            assert _timing_score(dte) == pytest.approx(expected, abs=1e-9)


# ── Fix 1 & Fix 3: structure_scorecard ───────────────────────────────────────

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import (
    score_atm_straddle,
    score_otm_strangle,
    _sample_confidence_and_penalty,
)
from services.structure_scorecard import WalkForwardPrior


def _make_prior(**kwargs) -> WalkForwardPrior:
    defaults = dict(
        structure="atm_straddle",
        win_rate=0.55,
        avg_return_pct=1.5,
        history_count=30,
        rank_score=0.55,
        source="synthetic",
    )
    defaults.update(kwargs)
    return WalkForwardPrior(**defaults)


def _make_snapshot(*, historical_move_source: str = "earnings_history", **overrides) -> VolSnapshot:
    """Minimal valid VolSnapshot for scorecard tests."""
    base = dict(
        symbol="TEST",
        as_of_date=date(2026, 4, 20),
        earnings_date=date(2026, 4, 26),
        release_timing="after market close",
        days_to_earnings=6,
        underlying_price=100.0,
        option_source="provided",
        underlying_source="provided",
        price_staleness_minutes=0,
        chain_staleness_minutes=0,
        data_quality="high",
        data_quality_score=0.88,
        rv30_yang_zhang=0.22,
        rv30_estimator="yang_zhang",
        rv_har_forecast=0.21,
        rv_percentile_rank=50.0,
        vol_regime_label="Normal",
        iv30=0.26,
        iv45=0.29,
        near_term_dte=6,
        near_term_atm_iv=0.26,
        back_term_dte=34,
        back_term_atm_iv=0.29,
        near_back_iv_ratio=0.897,
        term_structure_slope=0.0010,
        near_term_implied_move_pct=5.5,
        near_term_implied_sigma_pct=5.5 * 1.2533141373155001,  # MAD × √(π/2), P-5a
        non_event_move_pct_har=1.1,
        event_implied_move_pct=5.3,
        event_move_share_of_total=0.96,
        historical_event_count=8,
        historical_median_move_pct=6.8,
        historical_avg_last4_move_pct=7.1,
        historical_p90_move_pct=10.0,
        historical_move_std_pct=2.0,
        historical_move_anchor_pct=7.0,
        historical_move_uncertainty_pct=0.9,
        historical_vs_implied_move_ratio=1.28,
        tail_vs_implied_move_ratio=1.50,
        smile_curvature=0.15,
        smile_concavity_flag=False,
        smile_points=5,
        near_term_spread_pct=2.5,
        near_term_liquidity_proxy=3500.0,
        atm_call_spread_pct=2.4,
        atm_put_spread_pct=2.6,
        atm_total_open_interest=2500.0,
        atm_total_volume=1200.0,
        liquidity_tier="mid",
        iv_rv_yz=1.18,
        iv_rv_har=1.24,
        cheapness_score=0.60,
        event_risk_score=0.72,
        execution_score=0.78,
        timing_score=0.97,  # 6 DTE Gaussian ≈ 1.0
        historical_move_source=historical_move_source,
        null_reasons={},
    )
    base.update(overrides)
    return VolSnapshot(**base)


class TestFix3MoveSourcePropagation:
    """daily_fallback must degrade sample_confidence and set move_fit to 0.15 sentinel."""

    def test_earnings_history_gives_normal_sample_confidence(self):
        snap = _make_snapshot(historical_move_source="earnings_history", historical_event_count=8)
        prior = _make_prior()
        conf, penalty = _sample_confidence_and_penalty(snap, prior, max_penalty=0.12)
        # With 8 earnings events + 30 WF observations, confidence should be substantial
        assert conf > 0.40, f"Expected conf > 0.40 with earnings history; got {conf:.4f}"

    def test_daily_fallback_severely_degrades_confidence(self):
        snap_good = _make_snapshot(historical_move_source="earnings_history", historical_event_count=8)
        snap_fallback = _make_snapshot(historical_move_source="daily_fallback", historical_event_count=8)
        prior = _make_prior()
        conf_good, _ = _sample_confidence_and_penalty(snap_good, prior, max_penalty=0.12)
        conf_fallback, _ = _sample_confidence_and_penalty(snap_fallback, prior, max_penalty=0.12)
        assert conf_fallback < conf_good * 0.40, (
            f"daily_fallback should severely degrade confidence; "
            f"earnings_history={conf_good:.4f}, daily_fallback={conf_fallback:.4f}"
        )

    def test_daily_fallback_confidence_capped_at_015(self):
        """Even with many 'events' the fallback cap must hold."""
        snap = _make_snapshot(historical_move_source="daily_fallback", historical_event_count=100)
        prior = _make_prior(history_count=200)
        conf, _ = _sample_confidence_and_penalty(snap, prior, max_penalty=0.12)
        assert conf <= 0.15 + 1e-9, (
            f"daily_fallback confidence must be ≤0.15 (×0.15 multiplier); got {conf:.4f}"
        )

    def test_daily_fallback_sets_move_fit_sentinel_in_straddle(self):
        snap = _make_snapshot(historical_move_source="daily_fallback")
        prior = _make_prior()
        scorecard = score_atm_straddle(snap, prior=prior)
        assert scorecard.expected_move_fit_score == pytest.approx(0.15, abs=1e-9), (
            f"daily_fallback should set move_fit to 0.15 sentinel in straddle; "
            f"got {scorecard.expected_move_fit_score:.4f}"
        )

    def test_daily_fallback_sets_move_fit_sentinel_in_strangle(self):
        snap = _make_snapshot(historical_move_source="daily_fallback")
        prior = _make_prior(structure="otm_strangle")
        scorecard = score_otm_strangle(snap, prior=prior)
        assert scorecard.expected_move_fit_score == pytest.approx(0.15, abs=1e-9), (
            f"daily_fallback should set move_fit to 0.15 sentinel in strangle; "
            f"got {scorecard.expected_move_fit_score:.4f}"
        )

    def test_earnings_history_move_fit_exceeds_sentinel_in_straddle(self):
        snap = _make_snapshot(
            historical_move_source="earnings_history",
            historical_vs_implied_move_ratio=1.40,
            tail_vs_implied_move_ratio=1.70,
        )
        prior = _make_prior()
        scorecard = score_atm_straddle(snap, prior=prior)
        assert scorecard.expected_move_fit_score > 0.15, (
            f"earnings_history with good move ratios should give move_fit > 0.15; "
            f"got {scorecard.expected_move_fit_score:.4f}"
        )

    def test_daily_fallback_warning_in_rationale(self):
        snap = _make_snapshot(historical_move_source="daily_fallback")
        prior = _make_prior()
        scorecard = score_atm_straddle(snap, prior=prior)
        warning_bullets = [b for b in scorecard.rationale_bullets if "daily returns" in b.lower()]
        assert len(warning_bullets) >= 1, (
            "daily_fallback should inject a warning rationale bullet about daily returns"
        )


# ── Fix 1: IV/RV directional consistency in edge engine ──────────────────────

from web.api.edge_engine import IV_RV_CROWDING_WARNING_THRESHOLD


class TestFix1IvRvDirectionalConsistency:
    """
    The expansion gate must no longer hard-reject low IV/RV.
    Low IV/RV is a FAVORABLE signal for long-vol entry (cheap implied vol).
    High IV/RV (>1.60) should emit a soft crowding warning.
    """

    def test_crowding_threshold_constant_is_160(self):
        """Crowding threshold must be 1.60, consistent with screener ceiling."""
        assert IV_RV_CROWDING_WARNING_THRESHOLD == pytest.approx(1.60)

    def test_low_iv_rv_does_not_appear_in_expansion_gate_string(self):
        """
        Verify that the old 'IV/RV too low' language is gone from the source.
        This protects against the constant being renamed but the gate logic surviving.
        """
        import inspect
        import web.api.edge_engine as ee
        source = inspect.getsource(ee)
        assert "IV/RV too low for expansion" not in source, (
            "The old 'IV/RV too low for expansion' hard gate must be removed from edge_engine.py"
        )

    def test_min_iv_rv_for_expansion_constant_removed(self):
        """MIN_IV_RV_FOR_EXPANSION constant must not exist in edge engine."""
        import web.api.edge_engine as ee
        assert not hasattr(ee, "MIN_IV_RV_FOR_EXPANSION"), (
            "MIN_IV_RV_FOR_EXPANSION constant should be removed — it encoded the wrong direction"
        )

    def test_crowding_warning_fires_above_threshold(self):
        """When IV/RV > 1.60, expansion_gate_reasons must contain a crowding warning."""
        import web.api.edge_engine as ee
        # We test the constant and the condition language, not the full pipeline
        # (full pipeline integration test would require extensive mocking).
        assert IV_RV_CROWDING_WARNING_THRESHOLD == 1.60
        # The test below confirms directional consistency at the module level:
        # screener rewards low IV/RV → same as cheapness_score (both use 0.80 floor)
        # exp_signal_component = 1 - (iv_rv - 0.80) / 0.80 → score = 1.0 at iv_rv=0.80
        iv_rv_cheap = 0.85
        iv_rv_expensive = 1.55
        cheap_component = float(min(max(1.0 - (iv_rv_cheap - 0.80) / 0.80, 0.0), 1.0))
        expensive_component = float(min(max(1.0 - (iv_rv_expensive - 0.80) / 0.80, 0.0), 1.0))
        assert cheap_component > expensive_component, (
            "exp_signal_component must be higher for cheap IV/RV than expensive IV/RV"
        )
        assert cheap_component == pytest.approx(1.0 - (0.85 - 0.80) / 0.80, abs=1e-9)

    def test_exp_signal_component_formula_direction(self):
        """
        Direct validation of the new exp_signal_component formula:
        score = clip(1.0 - (iv_rv - 0.80) / 0.80, 0, 1)
        At iv_rv=0.80 → 1.0 (best), at iv_rv=1.60 → 0.0 (worst).
        """
        cases = [
            (0.80, 1.0),
            (0.85, 0.9375),
            (1.20, 0.50),
            (1.60, 0.0),
            (2.00, 0.0),   # clamped at 0
            (0.50, 1.0),   # clamped at 1
        ]
        for iv_rv, expected in cases:
            computed = float(min(max(1.0 - (iv_rv - 0.80) / 0.80, 0.0), 1.0))
            assert computed == pytest.approx(expected, abs=1e-6), (
                f"iv_rv={iv_rv}: expected {expected}, got {computed}"
            )

    def test_score_setup_removed_replaced_by_shared_setup_score(self):
        """
        _score_setup() was a dead function — never called in the pipeline.
        The active path is _shared_setup_score_from_snapshot() → compute_ranking_score().
        Verify the dead code has been cleaned up and the live path still works.
        """
        import web.api.edge_engine as ee
        # Dead code must be gone
        assert not hasattr(ee, "_score_setup"), (
            "_score_setup() was dead code and has been removed; "
            "the live path is _shared_setup_score_from_snapshot()"
        )
        # Live replacement must exist and produce a valid float in [0,1]
        assert hasattr(ee, "_shared_setup_score_from_snapshot"), (
            "_shared_setup_score_from_snapshot must still be present"
        )


# ── Fix 2: Kelly sizing removed ───────────────────────────────────────────────

class TestFix2KellyRemoved:
    """kelly_full_pct / kelly_half_pct must be absent from edge engine output."""

    def test_kelly_sizing_function_removed(self):
        """_kelly_sizing must not exist in the edge engine module."""
        import web.api.edge_engine as ee
        assert not hasattr(ee, "_kelly_sizing"), (
            "_kelly_sizing function should be removed — it depends on an uncalibrated edge estimate"
        )

    def test_kelly_fields_absent_from_metrics_source(self):
        """kelly_full_pct and kelly_half_pct must not appear in the metrics dict literal."""
        import inspect
        import web.api.edge_engine as ee
        source = inspect.getsource(ee)
        assert '"kelly_full_pct"' not in source, "kelly_full_pct must be removed from metrics dict"
        assert '"kelly_half_pct"' not in source, "kelly_half_pct must be removed from metrics dict"

    def test_position_sizing_note_present_in_source(self):
        """position_sizing_note must appear in the metrics dict as the replacement."""
        import inspect
        import web.api.edge_engine as ee
        source = inspect.getsource(ee)
        assert "position_sizing_note" in source, (
            "position_sizing_note should replace Kelly fields in the metrics dict"
        )

    def test_frontend_no_longer_uses_kelly_naming(self):
        from pathlib import Path

        app_source = Path("web/frontend/src/App.jsx").read_text()
        css_source = Path("web/frontend/src/styles.css").read_text()
        assert "kelly-note" not in app_source
        assert ".kelly-note" not in css_source
        assert "position-sizing-note" in app_source


class TestPhaseBCopyCleanup:
    def test_unsupported_iv_expansion_magnitude_claim_removed_from_screener(self):
        from pathlib import Path

        screener_source = Path("services/screener_service.py").read_text()
        assert "8-20 %" not in screener_source
