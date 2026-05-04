"""Tests for services/screener_service.py — ranking formula and determinism."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.screener_service import (
    _dte_score,
    _build_ranked_snapshot,
    _iv_entry_score,
    _liquidity_score,
    _move_history_score,
    _sample_score,
    _ts_score,
    _screen_one_symbol_ranked,
    compute_ranking_score,
)
from services.earnings_vol_snapshot import VolSnapshot
from services.earnings_event_service import ResolvedEarningsEvent


# ── Pure scoring functions ────────────────────────────────────────────────────


class TestIvEntryScore:
    def test_cheap_iv_gives_high_score(self):
        # iv_rv = 0.80 → score == 1.0 (at the left boundary)
        assert _iv_entry_score(0.80) == pytest.approx(1.0)

    def test_expensive_iv_gives_zero(self):
        # iv_rv = 1.60 → score == 0.0 (at the right boundary)
        assert _iv_entry_score(1.60) == pytest.approx(0.0)

    def test_midpoint(self):
        # iv_rv = 1.20 → halfway between 0.80 and 1.60 → score ≈ 0.50
        score = _iv_entry_score(1.20)
        assert 0.45 <= score <= 0.55

    def test_below_floor_clamped(self):
        # iv_rv < 0.80 → still clamped at 1.0 (very cheap IV is still max score)
        assert _iv_entry_score(0.50) == pytest.approx(1.0)

    def test_above_ceiling_clamped(self):
        assert _iv_entry_score(2.50) == pytest.approx(0.0)

    def test_output_in_unit_interval(self):
        for ratio in [0.0, 0.5, 1.0, 1.2, 1.6, 2.0]:
            score = _iv_entry_score(ratio)
            assert 0.0 <= score <= 1.0, f"Out of [0,1] for iv_rv={ratio}"


class TestMoveHistoryScore:
    def test_small_move_gives_low_score(self):
        # 2 % move → score == 0.0
        assert _move_history_score(2.0) == pytest.approx(0.0)

    def test_large_move_gives_high_score(self):
        # 12 %+ → score == 1.0
        assert _move_history_score(12.0) == pytest.approx(1.0)
        assert _move_history_score(20.0) == pytest.approx(1.0)

    def test_midpoint(self):
        score = _move_history_score(7.0)
        assert 0.45 <= score <= 0.55

    def test_output_in_unit_interval(self):
        for move in [0.0, 2.0, 5.0, 7.0, 12.0, 25.0]:
            s = _move_history_score(move)
            assert 0.0 <= s <= 1.0


class TestTsScore:
    def test_near_cheaper_gives_high_score(self):
        # ts_ratio = 0.80 → near IV is 20% cheaper than back → score == 1.0
        assert _ts_score(0.80) == pytest.approx(1.0)

    def test_near_elevated_gives_low_score(self):
        # ts_ratio = 1.30 → near IV is elevated vs back → score == 0.0
        assert _ts_score(1.30) == pytest.approx(0.0)

    def test_midpoint(self):
        score = _ts_score(1.05)
        assert 0.40 <= score <= 0.60

    def test_output_in_unit_interval(self):
        for ratio in [0.5, 0.80, 1.0, 1.10, 1.30, 1.80]:
            s = _ts_score(ratio)
            assert 0.0 <= s <= 1.0


class TestDteScore:
    def test_sweet_spot_near_six(self):
        # DTE 6 is the Gaussian center → highest score
        score_6 = _dte_score(6)
        score_1 = _dte_score(1)
        score_15 = _dte_score(15)
        assert score_6 > score_1
        assert score_6 > score_15

    def test_score_decreases_away_from_center(self):
        score_5 = _dte_score(5)
        score_8 = _dte_score(8)
        score_6 = _dte_score(6)
        assert score_6 >= score_5
        assert score_6 >= score_8

    def test_output_in_unit_interval(self):
        for dte in [0, 1, 3, 6, 8, 10, 14, 30]:
            s = _dte_score(dte)
            assert 0.0 <= s <= 1.0, f"Out of [0,1] for dte={dte}"


class TestSampleScore:
    def test_zero_sample_gives_zero(self):
        assert _sample_score(0) == pytest.approx(0.0)

    def test_eight_or_more_gives_max(self):
        assert _sample_score(8) == pytest.approx(1.0)
        assert _sample_score(20) == pytest.approx(1.0)

    def test_four_gives_half(self):
        score = _sample_score(4)
        assert 0.45 <= score <= 0.55

    def test_output_in_unit_interval(self):
        for n in [0, 1, 4, 8, 16]:
            s = _sample_score(n)
            assert 0.0 <= s <= 1.0


class TestLiquidityScore:
    def test_tight_spread_gives_high_score(self):
        # 0% spread → score == 1.0
        assert _liquidity_score(0.0) == pytest.approx(1.0)

    def test_wide_spread_gives_zero(self):
        # 15%+ spread → score == 0.0
        assert _liquidity_score(15.0) == pytest.approx(0.0)
        assert _liquidity_score(25.0) == pytest.approx(0.0)

    def test_midpoint(self):
        score = _liquidity_score(7.5)
        assert 0.45 <= score <= 0.55

    def test_output_in_unit_interval(self):
        for spread in [0, 2, 5, 8, 15, 30]:
            s = _liquidity_score(spread)
            assert 0.0 <= s <= 1.0


# ── compute_ranking_score ─────────────────────────────────────────────────────


class TestComputeRankingScore:
    def test_output_in_unit_interval(self):
        """Ranking score must always be in [0, 1]."""
        score = compute_ranking_score(
            iv_rv_ratio=1.10,
            ts_ratio=0.95,
            median_earnings_move_pct=8.0,
            sample_size=6,
            dte=5,
            spread_pct=3.0,
        )
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        """Same inputs must produce identical outputs across repeated calls."""
        kwargs = dict(
            iv_rv_ratio=1.05,
            ts_ratio=0.92,
            median_earnings_move_pct=9.5,
            sample_size=8,
            dte=6,
            spread_pct=2.0,
        )
        results = [compute_ranking_score(**kwargs) for _ in range(10)]
        assert all(r == results[0] for r in results), "Non-deterministic ranking score"

    def test_better_setup_scores_higher(self):
        """A clearly superior setup (cheap IV, good TS, large hist move, tight spread)
        must outrank a clearly inferior setup."""
        good = compute_ranking_score(
            iv_rv_ratio=0.85,    # IV cheap
            ts_ratio=0.88,       # near term not elevated
            median_earnings_move_pct=12.0,  # big moves
            sample_size=12,
            dte=6,
            spread_pct=1.5,
        )
        bad = compute_ranking_score(
            iv_rv_ratio=1.55,    # IV expensive
            ts_ratio=1.25,       # near elevated
            median_earnings_move_pct=2.5,   # tiny moves
            sample_size=2,
            dte=1,               # too close
            spread_pct=14.0,     # wide spread
        )
        assert good > bad, f"Good setup ({good:.4f}) did not beat bad setup ({bad:.4f})"

    def test_weights_sum_to_one(self):
        """Smoke-test: a perfect setup on every component gives score close to 1."""
        perfect = compute_ranking_score(
            iv_rv_ratio=0.80,
            ts_ratio=0.80,
            median_earnings_move_pct=15.0,
            sample_size=20,
            dte=6,
            spread_pct=0.0,
        )
        assert perfect == pytest.approx(1.0, abs=0.01)

    def test_iv_rv_ratio_dominance(self):
        """iv_rv_ratio has the largest weight (0.32).  Changing it from ideal to
        worst should lower the score by more than any other single component."""
        base = dict(
            iv_rv_ratio=0.80,
            ts_ratio=0.80,
            median_earnings_move_pct=15.0,
            sample_size=20,
            dte=6,
            spread_pct=0.0,
        )
        base_score = compute_ranking_score(**base)

        # Degrade iv_rv only
        iv_degraded = compute_ranking_score(**{**base, "iv_rv_ratio": 1.60})
        # Degrade ts only
        ts_degraded = compute_ranking_score(**{**base, "ts_ratio": 1.30})

        assert (base_score - iv_degraded) > (base_score - ts_degraded), (
            "iv_rv_ratio (weight 0.32) should produce larger score drop than ts (0.18)"
        )


def _stub_snapshot() -> VolSnapshot:
    return VolSnapshot(
        symbol="AAPL",
        as_of_date=date(2026, 4, 20),
        earnings_date=date(2026, 4, 28),
        release_timing="after market close",
        days_to_earnings=8,
        underlying_price=100.0,
        option_source="provided",
        underlying_source="provided",
        price_staleness_minutes=0,
        chain_staleness_minutes=0,
        data_quality="high",
        data_quality_score=0.92,
        rv30_yang_zhang=0.22,
        rv30_estimator="yang_zhang",
        rv_har_forecast=0.20,
        rv_percentile_rank=44.0,
        vol_regime_label="Normal",
        iv30=0.25,
        iv45=0.28,
        near_term_dte=4,
        near_term_atm_iv=0.24,
        back_term_dte=25,
        back_term_atm_iv=0.28,
        near_back_iv_ratio=0.8571,
        term_structure_slope=0.0018,
        near_term_implied_move_pct=6.0,
        non_event_move_pct_har=1.2,
        event_implied_move_pct=5.88,
        event_move_share_of_total=0.98,
        historical_event_count=6,
        historical_median_move_pct=7.25,
        historical_avg_last4_move_pct=7.75,
        historical_p90_move_pct=9.10,
        historical_move_std_pct=1.90,
        historical_move_anchor_pct=7.575,
        historical_move_uncertainty_pct=0.99,
        historical_vs_implied_move_ratio=1.288,
        tail_vs_implied_move_ratio=1.548,
        smile_curvature=0.18,
        smile_concavity_flag=False,
        smile_points=5,
        near_term_spread_pct=2.4,
        near_term_liquidity_proxy=4200.0,
        atm_call_spread_pct=2.3,
        atm_put_spread_pct=2.5,
        atm_total_open_interest=2800.0,
        atm_total_volume=1400.0,
        liquidity_tier="mid",
        iv_rv_yz=1.136,
        iv_rv_har=1.25,
        cheapness_score=0.63,
        event_risk_score=0.74,
        execution_score=0.81,
        timing_score=0.73,
        historical_move_source="earnings_history",
        null_reasons={},
        earnings_source_primary="alpha_vantage_calendar",
        earnings_source_confirmed=None,
        earnings_source_confidence=0.72,
        release_timing_source="yfinance_calendar",
    )


class TestSnapshotBackedRankedScreener:
    def test_build_ranked_snapshot_uses_shared_event_resolution(self):
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=20, freq="B"),
                "Open": [100.0] * 20,
                "High": [101.0] * 20,
                "Low": [99.0] * 20,
                "Close": [100.5] * 20,
                "Volume": [1_000_000] * 20,
            }
        )

        with patch(
            "services.screener_service.resolve_upcoming_earnings_event",
            return_value=ResolvedEarningsEvent(
                symbol="AAPL",
                earnings_date=date(2026, 4, 28),
                release_timing="AMC",
                primary_source="alpha_vantage_calendar",
                confirmed_source=None,
                source_confidence=0.72,
                source_notes=["cached alpha vantage"],
            ),
        ) as mock_resolve:
            with patch("services.screener_service._collect_yf_option_chain_frame", return_value=pd.DataFrame()):
                with patch("services.screener_service._collect_yf_past_earnings_events", return_value=[]):
                    with patch("services.screener_service.build_vol_snapshot", return_value=_stub_snapshot()) as mock_snapshot:
                        snapshot, release_timing, earnings_date, error = _build_ranked_snapshot(
                            "AAPL",
                            ticker,
                            date(2026, 4, 20),
                            date(2026, 5, 18),
                        )

        assert error is None
        assert snapshot is not None
        assert release_timing == "AMC"
        assert earnings_date == date(2026, 4, 28)
        mock_resolve.assert_called_once()
        mock_snapshot.assert_called_once()

    def test_ranked_symbol_maps_fields_from_snapshot(self):
        ticker = MagicMock()
        ticker.history.return_value = MagicMock()  # not used because helper is patched

        with patch("services.screener_service.yf.Ticker", return_value=ticker):
            with patch(
                "services.screener_service._build_ranked_snapshot",
                return_value=(_stub_snapshot(), "AMC", date(2026, 4, 28), None),
            ) as mock_snapshot:
                row = _screen_one_symbol_ranked("AAPL", date(2026, 4, 20), date(2026, 5, 18))

        assert row is not None
        mock_snapshot.assert_called_once()
        assert row["release_timing"] == "AMC"
        assert row["days_to_earnings"] == 8
        assert row["iv30"] == pytest.approx(0.24, abs=1e-4)
        assert row["rv30"] == pytest.approx(0.22, abs=1e-4)
        assert row["iv_rv_ratio"] == pytest.approx(1.136, abs=1e-3)
        assert row["term_structure_ratio"] == pytest.approx(0.8571, abs=1e-4)
        assert row["median_earnings_move_pct"] == pytest.approx(7.25, abs=1e-2)
        assert row["p90_earnings_move_pct"] == pytest.approx(9.10, abs=1e-2)
        assert row["sample_size"] == 6
        assert row["avg_spread_pct"] == pytest.approx(2.4, abs=1e-2)
        assert row["earnings_source_primary"] == "alpha_vantage_calendar"
        assert row["earnings_source_confirmed"] is None
        assert row["earnings_source_confidence"] == pytest.approx(0.72, abs=1e-6)
        assert row["error"] is None
