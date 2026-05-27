import unittest
from datetime import date
from pathlib import Path
import time
from unittest.mock import patch

from services.earnings_vol_snapshot import VolSnapshot
from services.structure_scorecard import (
    WalkForwardPrior,
    build_structure_scorecards,
    reload_walk_forward_priors,
    score_atm_straddle,
)


def _base_snapshot(**overrides) -> VolSnapshot:
    payload = dict(
        symbol="AAPL",
        as_of_date=date(2026, 4, 20),
        earnings_date=date(2026, 4, 28),
        release_timing="after market close",
        days_to_earnings=8,
        underlying_price=100.0,
        option_source="provided",
        underlying_source="provided",
        price_staleness_minutes=5,
        chain_staleness_minutes=5,
        data_quality="high",
        data_quality_score=0.92,
        rv30_yang_zhang=0.21,
        rv30_estimator="yang_zhang",
        rv_har_forecast=0.20,
        rv_percentile_rank=48.0,
        vol_regime_label="Normal",
        iv30=0.24,
        iv45=0.27,
        near_term_dte=4,
        near_term_atm_iv=0.23,
        back_term_dte=25,
        back_term_atm_iv=0.27,
        near_back_iv_ratio=0.87,
        term_structure_slope=0.0025,
        near_term_implied_move_pct=5.8,
        near_term_implied_sigma_pct=5.8 * 1.2533141373155001,  # MAD × √(π/2), P-5a
        non_event_move_pct_har=1.2,
        event_implied_move_pct=5.67,
        event_move_share_of_total=0.88,
        historical_event_count=8,
        historical_median_move_pct=7.2,
        historical_avg_last4_move_pct=7.6,
        historical_p90_move_pct=10.0,
        historical_move_std_pct=2.1,
        historical_move_anchor_pct=7.4,
        historical_move_uncertainty_pct=0.85,
        historical_vs_implied_move_ratio=1.28,
        tail_vs_implied_move_ratio=1.72,
        smile_curvature=0.18,
        smile_concavity_flag=False,
        smile_points=7,
        near_term_spread_pct=2.5,
        near_term_liquidity_proxy=4200.0,
        atm_call_spread_pct=2.4,
        atm_put_spread_pct=2.6,
        atm_total_open_interest=3200.0,
        atm_total_volume=1600.0,
        liquidity_tier="high",
        iv_rv_yz=1.14,
        iv_rv_har=1.20,
        cheapness_score=0.62,
        event_risk_score=0.74,
        execution_score=0.86,
        timing_score=0.78,
        historical_move_source="earnings_history",
        null_reasons={},
    )
    payload.update(overrides)
    return VolSnapshot(**payload)


def _neutral_priors() -> dict[str, WalkForwardPrior]:
    return {
        structure: WalkForwardPrior(
            structure=structure,
            history_count=20,
            win_rate=0.52,
            avg_return_pct=2.0,
            rank_score=0.50,
            source="test_neutral",
        )
        for structure in ("atm_straddle", "otm_strangle", "call_calendar", "put_calendar")
    }


class TestStructureScorecards(unittest.TestCase):
    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_structure_differentiation_by_regime(self, _mock_priors):
        straddle_snapshot = _base_snapshot(
            historical_vs_implied_move_ratio=1.72,
            tail_vs_implied_move_ratio=1.95,
            historical_move_anchor_pct=8.8,
            event_implied_move_pct=6.2,
            cheapness_score=0.56,
            event_risk_score=0.70,
            term_structure_slope=0.0008,
            near_back_iv_ratio=0.95,
            iv_rv_yz=0.98,
            iv_rv_har=1.00,
        )
        straddle_cards = build_structure_scorecards(straddle_snapshot)
        self.assertEqual(
            max(straddle_cards, key=lambda card: card.composite_structure_score).structure,
            "atm_straddle",
        )

        calendar_snapshot = _base_snapshot(
            cheapness_score=0.95,
            term_structure_slope=0.0034,
            near_back_iv_ratio=0.82,
            event_move_share_of_total=0.58,
            historical_move_anchor_pct=5.6,
            historical_vs_implied_move_ratio=0.96,
            tail_vs_implied_move_ratio=1.05,
            event_risk_score=0.48,
            iv_rv_yz=0.84,
            iv_rv_har=0.86,
            atm_call_spread_pct=1.8,
            atm_put_spread_pct=1.9,
        )
        calendar_cards = build_structure_scorecards(calendar_snapshot)
        self.assertIn(
            max(calendar_cards, key=lambda card: card.composite_structure_score).structure,
            {"call_calendar", "put_calendar"},
        )

        strangle_snapshot = _base_snapshot(
            tail_vs_implied_move_ratio=2.30,
            event_risk_score=0.94,
            historical_move_anchor_pct=10.4,
            historical_vs_implied_move_ratio=1.08,
            cheapness_score=0.44,
            near_back_iv_ratio=0.92,
            term_structure_slope=0.0016,
            iv_rv_yz=1.02,
            iv_rv_har=1.05,
        )
        strangle_cards = build_structure_scorecards(strangle_snapshot)
        self.assertEqual(
            max(strangle_cards, key=lambda card: card.composite_structure_score).structure,
            "otm_strangle",
        )

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_poor_spreads_make_all_structures_ineligible(self, _mock_priors):
        snapshot = _base_snapshot(
            near_term_spread_pct=25.0,
            atm_call_spread_pct=22.0,
            atm_put_spread_pct=23.0,
            execution_score=0.20,
        )
        cards = build_structure_scorecards(snapshot)

        self.assertTrue(all(not card.eligible for card in cards))
        self.assertTrue(all(card.composite_structure_score == 0.0 for card in cards))
        self.assertTrue(all("spread_exceeds_absolute_threshold" in card.eligibility_flags for card in cards))

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_record_only_surface_quality_makes_all_structures_ineligible(self, _mock_priors):
        snapshot = _base_snapshot(surface_quality_status="record_only")

        cards = build_structure_scorecards(snapshot)

        self.assertTrue(all(not card.eligible for card in cards))
        self.assertTrue(all(card.composite_structure_score == 0.0 for card in cards))
        self.assertTrue(all("surface_quality_record_only" in card.eligibility_flags for card in cards))

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_increasing_spread_worsens_execution_penalty(self, _mock_priors):
        tight = _base_snapshot(near_term_spread_pct=2.0, atm_call_spread_pct=2.0, atm_put_spread_pct=2.1, execution_score=0.90)
        wide = _base_snapshot(near_term_spread_pct=10.0, atm_call_spread_pct=9.8, atm_put_spread_pct=10.2, execution_score=0.55)

        tight_card = score_atm_straddle(tight)
        wide_card = score_atm_straddle(wide)

        self.assertGreater(wide_card.execution_penalty, tight_card.execution_penalty)
        self.assertLess(wide_card.composite_structure_score, tight_card.composite_structure_score)

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_increasing_iv_rv_worsens_straddle_score(self, _mock_priors):
        cheap = _base_snapshot(iv_rv_yz=0.90, iv_rv_har=0.92, cheapness_score=0.84)
        expensive = _base_snapshot(iv_rv_yz=1.55, iv_rv_har=1.58, cheapness_score=0.18)

        cheap_card = score_atm_straddle(cheap)
        expensive_card = score_atm_straddle(expensive)

        self.assertGreater(expensive_card.crowding_penalty, cheap_card.crowding_penalty)
        self.assertLess(expensive_card.composite_structure_score, cheap_card.composite_structure_score)

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_concavity_penalizes_straddle(self, _mock_priors):
        flat = _base_snapshot(smile_curvature=0.12, smile_concavity_flag=False)
        concave = _base_snapshot(smile_curvature=0.78, smile_concavity_flag=True)

        flat_card = score_atm_straddle(flat)
        concave_card = score_atm_straddle(concave)

        self.assertGreater(concave_card.concavity_penalty, flat_card.concavity_penalty)
        self.assertLess(concave_card.composite_structure_score, flat_card.composite_structure_score)

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_poor_liquidity_penalizes_all_structures(self, _mock_priors):
        liquid = _base_snapshot(near_term_liquidity_proxy=4200.0, atm_total_open_interest=3200.0, atm_total_volume=1600.0, execution_score=0.88)
        illiquid = _base_snapshot(near_term_liquidity_proxy=180.0, atm_total_open_interest=120.0, atm_total_volume=60.0, execution_score=0.30)

        liquid_cards = {card.structure: card for card in build_structure_scorecards(liquid)}
        illiquid_cards = {card.structure: card for card in build_structure_scorecards(illiquid)}

        for structure in liquid_cards:
            self.assertGreater(illiquid_cards[structure].execution_penalty, liquid_cards[structure].execution_penalty)
            self.assertLess(illiquid_cards[structure].composite_structure_score, liquid_cards[structure].composite_structure_score)

    @patch("services.structure_scorecard._load_walk_forward_priors", side_effect=lambda as_of_date=None: _neutral_priors())
    def test_scorecards_are_deterministic_for_same_snapshot(self, _mock_priors):
        snapshot = _base_snapshot()
        first = [card.to_dict() for card in build_structure_scorecards(snapshot)]
        second = [card.to_dict() for card in build_structure_scorecards(snapshot)]
        self.assertEqual(first, second)


class TestWalkForwardPriorCache(unittest.TestCase):
    def _fake_prior(self, structure: str, counter: dict[str, int]) -> WalkForwardPrior:
        counter["calls"] += 1
        return WalkForwardPrior(
            structure=structure,
            history_count=counter["calls"],
            win_rate=0.50,
            avg_return_pct=1.0,
            rank_score=0.50,
            source="test_cache",
        )

    def test_mtime_cache_reuses_unchanged_file_and_reloads_on_change(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "marker.json"
            marker.write_text("{}")
            counter = {"calls": 0}

            def _signature(as_of_date=None):
                return ((str(marker), marker.stat().st_mtime),)

            def _calendar(structure: str) -> WalkForwardPrior:
                return self._fake_prior(structure, counter)

            def _straddle() -> WalkForwardPrior:
                return self._fake_prior("atm_straddle", counter)

            def _strangle() -> WalkForwardPrior:
                return self._fake_prior("otm_strangle", counter)

            reload_walk_forward_priors()
            with patch("services.structure_scorecard._walk_forward_prior_signature", side_effect=_signature), \
                 patch("services.structure_scorecard._load_calendar_prior_from_reports", side_effect=_calendar), \
                 patch("services.structure_scorecard._load_straddle_prior_from_reports", side_effect=_straddle), \
                 patch("services.structure_scorecard._load_strangle_prior_from_reports", side_effect=_strangle), \
                 patch("services.structure_prior_store.load_all_structure_priors", return_value={}):
                from services import structure_scorecard as sc

                first = sc._load_walk_forward_priors()
                second = sc._load_walk_forward_priors()
                self.assertIs(first, second)
                self.assertEqual(counter["calls"], 4)

                time.sleep(0.02)
                os.utime(marker, None)
                third = sc._load_walk_forward_priors()
                self.assertIsNot(first, third)
                self.assertEqual(counter["calls"], 8)

                sc.reload_walk_forward_priors()
                fourth = sc._load_walk_forward_priors()
                self.assertEqual(counter["calls"], 12)
                self.assertIsNot(third, fourth)


# ──────────────────────────────────────────────────────────────────────────
# PR #70: Neutral prior rank consistency
# ──────────────────────────────────────────────────────────────────────────


class TestNeutralPriorRankConsistency(unittest.TestCase):
    """Pre-PR-#70, ``_neutral_prior()`` hardcoded ``rank_score=0.50``
    while ``_compute_rank_score(0.50, 0.0, 0)`` returned 0.308571 —
    same nominal inputs, different rank. A "no observations at all"
    structure could outrank an "observed-and-genuinely-neutral"
    structure by ~0.19, flipping selector decisions on thin-data
    names.

    PR #70 routes ``_neutral_prior``'s rank_score through
    ``_compute_rank_score`` so the consistency property (same
    input → same output) holds. These tests pin the property and
    the new numerical value.
    """

    def test_neutral_prior_rank_score_matches_compute_rank_score(self):
        """Load-bearing consistency property: a neutral prior and a
        real prior with the same (win_rate, avg_return, history_count)
        defaults MUST produce the same rank_score. This is the
        invariant the old hardcoded 0.50 violated."""
        from services.structure_scorecard import (
            _neutral_prior,
            _compute_rank_score,
            _NEUTRAL_PRIOR_WIN_RATE,
            _NEUTRAL_PRIOR_AVG_RETURN_PCT,
            _NEUTRAL_PRIOR_HISTORY_COUNT,
        )
        neutral = _neutral_prior("atm_straddle", source="test_consistency")
        observed = _compute_rank_score(
            win_rate=_NEUTRAL_PRIOR_WIN_RATE,
            avg_return_pct=_NEUTRAL_PRIOR_AVG_RETURN_PCT,
            history_count=_NEUTRAL_PRIOR_HISTORY_COUNT,
        )
        self.assertAlmostEqual(
            neutral.rank_score, observed, places=10,
            msg=(
                "Neutral prior's rank_score MUST equal "
                "_compute_rank_score(neutral defaults). If they "
                "diverge, the perverse-incentive bug returns."
            ),
        )

    def test_neutral_prior_rank_score_is_approximately_0_309(self):
        """Anchor the literal post-fix value. Catches an accidental
        revert of the routing to a hardcoded 0.50."""
        from services.structure_scorecard import _neutral_prior
        neutral = _neutral_prior("atm_straddle", source="test_anchor")
        # 0.45 * return_score(0,−10,15) + 0.30 * win_score(0.50,0.35,0.70)
        # + 0.25 * history_score(0,5,50)
        # = 0.45 * 0.40 + 0.30 * 0.428571 + 0.25 * 0
        # = 0.180 + 0.128571 + 0 = 0.308571
        self.assertAlmostEqual(neutral.rank_score, 0.308571, places=6)
        # And specifically: NOT 0.50 (the old hardcoded value).
        self.assertLess(
            neutral.rank_score, 0.40,
            "rank_score must be < 0.40 — if it's ~0.50, the "
            "PR #70 routing fix was reverted.",
        )

    def test_neutral_prior_other_fields_unchanged(self):
        """The fix touches ONLY rank_score. Every other field on the
        WalkForwardPrior dataclass must remain at its pre-PR-#70
        value so downstream consumers that inspect history_count,
        win_rate, avg_return_pct, source, or structure are
        unaffected."""
        from services.structure_scorecard import _neutral_prior
        neutral = _neutral_prior("call_calendar", source="test_other_fields")
        self.assertEqual(neutral.structure, "call_calendar")
        self.assertEqual(neutral.history_count, 0)
        self.assertEqual(neutral.win_rate, 0.50)
        self.assertEqual(neutral.avg_return_pct, 0.0)
        self.assertEqual(neutral.source, "test_other_fields")

    def test_observed_neutral_prior_ranks_identically_to_no_data(self):
        """The actual behavior change in plain language: a structure
        with one observation showing 50% win rate and 0% avg return
        and 0 history (an edge case but possible) ranks IDENTICALLY
        to a no-data structure. Pre-PR-#70 they ranked DIFFERENTLY
        (0.50 hardcoded vs. 0.308571 computed) — a structure with
        less information beat a structure with the same nominal
        statistics.

        This test pins the new equivalence.
        """
        from services.structure_scorecard import (
            _neutral_prior, _compute_rank_score,
        )
        neutral_rank = _neutral_prior(
            "otm_strangle", source="test_equivalence",
        ).rank_score
        observed_neutral_rank = _compute_rank_score(
            win_rate=0.50, avg_return_pct=0.0, history_count=0,
        )
        self.assertEqual(neutral_rank, observed_neutral_rank)

    def test_structures_with_observed_data_now_outrank_no_data(self):
        """The directional consequence the fix is meant to produce.

        A structure with even a small amount of observed positive
        evidence (win_rate=0.55, avg_return=+2%, history=5) should
        rank HIGHER than the no-data baseline. Pre-PR-#70 this was
        often false — the hardcoded 0.50 was a high bar to beat
        for thin-data structures. Now the no-data baseline sits at
        ~0.309, so any positive evidence cleanly outranks it.
        """
        from services.structure_scorecard import (
            _neutral_prior, _compute_rank_score,
        )
        no_data = _neutral_prior(
            "atm_straddle", source="test_direction",
        ).rank_score
        thin_positive_evidence = _compute_rank_score(
            win_rate=0.55, avg_return_pct=2.0, history_count=5,
        )
        self.assertGreater(thin_positive_evidence, no_data, (
            f"A structure with thin positive evidence "
            f"(win_rate=0.55, +2% avg return, 5 obs) must outrank "
            f"the no-data baseline. Got "
            f"thin={thin_positive_evidence:.4f} vs no_data={no_data:.4f}."
        ))


# ──────────────────────────────────────────────────────────────────────────
# PR #71: Trade-log prior loaders
# ──────────────────────────────────────────────────────────────────────────


class TestPriorLoadersFromTradeLog(unittest.TestCase):
    """PR #71 replaces the scoreboard-based straddle prior loader
    (which weighted by ``realized_count`` over multi-counted
    filter combinations) with a trade-log loader that counts each
    physical trade exactly once.

    Per Codex review: unit tests use SYNTHETIC fixtures only —
    live numerical pinning lives in the PR body manual smoke, not
    here, because the live report artifacts are not guaranteed
    stable across future backtest re-runs.

    The single live-file dependency is the schema-existence guard
    (test #5), which only asserts column NAMES — not values — so
    a backtest schema change surfaces immediately without
    coupling to specific data.
    """

    # Layout of a synthetic trade log directory. Tests build their
    # own fixtures so we never depend on the live report data.
    def _write_trade_log_fixture(
        self, reports_root: Path, rows: list[dict], suffix: str = "20260101T000000Z",
    ) -> Path:
        """Write a minimal synthetic trade log into
        ``reports_root/pre_earnings_otm_strangle_<suffix>/`` and
        return its path. ``rows`` is a list of dicts; each dict
        must carry the columns the loader inspects. Missing
        numeric columns default to NaN so the fixture stays
        compact."""
        from services.structure_scorecard import _TRADE_LOG_REQUIRED_COLUMNS
        run_dir = reports_root / f"pre_earnings_otm_strangle_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Make sure every required column is present even if the
        # caller didn't supply it — the loader's schema check
        # would otherwise short-circuit to None.
        normalized = []
        for r in rows:
            full = {col: r.get(col, float("nan")) for col in _TRADE_LOG_REQUIRED_COLUMNS}
            # Tag any caller-supplied extra columns through.
            for k, v in r.items():
                if k not in full:
                    full[k] = v
            normalized.append(full)
        import pandas as pd
        path = run_dir / "pre_earnings_otm_strangle_trade_log.csv"
        pd.DataFrame(normalized).to_csv(path, index=False)
        return path

    def _baseline_row(self, **overrides) -> dict:
        """A trade-log row that passes the baseline filter. Tests
        override individual fields to construct edge cases without
        re-declaring every column."""
        base = {
            "structure": "atm_straddle",
            "entry_offset_bdays": 3,
            "exit_offset_bdays": 0,
            "entry_pass_oi_100": True,
            "entry_pass_spread_10": True,
            "priceable_realized": True,
            "pnl_cross_25": 1.0,           # winner
            "return_cross_25_pct": 5.0,
        }
        base.update(overrides)
        return base

    def test_baseline_filter_selects_only_qualifying_rows(self):
        """Loader must include only rows that pass entry=3, exit=0,
        OI>=100, spread<=10%, priceable, finite cross_25. A row
        that fails any criterion does NOT contribute to the
        prior."""
        from services.structure_scorecard import _load_prior_from_trade_log
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                self._write_trade_log_fixture(Path(tmp), [
                    # 3 baseline winners
                    self._baseline_row(pnl_cross_25=1.0, return_cross_25_pct=5.0),
                    self._baseline_row(pnl_cross_25=2.0, return_cross_25_pct=10.0),
                    self._baseline_row(pnl_cross_25=0.5, return_cross_25_pct=2.0),
                    # 1 baseline loser
                    self._baseline_row(pnl_cross_25=-1.0, return_cross_25_pct=-5.0),
                    # excluded: wrong entry offset
                    self._baseline_row(entry_offset_bdays=2, pnl_cross_25=99.0),
                    # excluded: wrong exit offset
                    self._baseline_row(exit_offset_bdays=1, pnl_cross_25=99.0),
                    # excluded: failed OI filter
                    self._baseline_row(entry_pass_oi_100=False, pnl_cross_25=99.0),
                    # excluded: failed spread filter
                    self._baseline_row(entry_pass_spread_10=False, pnl_cross_25=99.0),
                    # excluded: not priceable
                    self._baseline_row(priceable_realized=False, pnl_cross_25=99.0),
                    # excluded: NaN PnL
                    self._baseline_row(pnl_cross_25=float("nan")),
                    # excluded: wrong structure
                    self._baseline_row(structure="otm_strangle_3pct", pnl_cross_25=99.0),
                ])
                prior = _load_prior_from_trade_log(
                    structure_kind="atm_straddle",
                    prior_structure_label="atm_straddle",
                )
        self.assertIsNotNone(prior)
        # 4 baseline rows, 3 winners → win_rate = 0.75.
        self.assertEqual(prior.history_count, 4)
        self.assertAlmostEqual(prior.win_rate, 0.75, places=10)
        # Excluded high-PnL rows must not contribute — if any leaked
        # in, win_rate would be 7/11 or higher.

    def test_no_realized_count_weighting(self):
        """Codex P1 concern: the trade-log loader must NOT
        re-introduce a weighting bias. Each physical trade gets
        equal weight, not weight-by-realized_count.

        Codex review of PR #71 first round asked for a stronger
        version of this test: inject a fake ``realized_count``
        column with EXTREME asymmetric values across winners vs
        losers, and assert the loader ignores it. If the loader
        ever accidentally re-introduces weighting (e.g., a future
        refactor that reads from a more aggregated source), the
        weighted win_rate would diverge sharply from the
        trade-level win_rate.

        Fixture:
          - 1 baseline WINNER with realized_count = 1 (tight filter)
          - 1 baseline LOSER  with realized_count = 999 (loose filter)
          A weighted-by-realized_count win_rate would be ≈ 0.001
          (1 / (1+999)). The trade-level (correct) answer is 0.50.
          The assertion at 0.50 catches any drift toward the
          weighted value.
        """
        from services.structure_scorecard import _load_prior_from_trade_log
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                rows = [
                    # WINNER with low scoreboard-style realized_count
                    self._baseline_row(
                        pnl_cross_25=1.0, return_cross_25_pct=5.0,
                        realized_count=1,  # would be weighted low
                    ),
                    # LOSER with very high realized_count — if a
                    # future refactor weighted by this column, the
                    # win_rate would collapse to ~0.
                    self._baseline_row(
                        pnl_cross_25=-1.0, return_cross_25_pct=-5.0,
                        realized_count=999,  # would be weighted high
                    ),
                ]
                # Plus 50 winners at wrong offsets — must NOT count,
                # regardless of any realized_count or weighting bug.
                rows.extend([
                    self._baseline_row(
                        entry_offset_bdays=2, pnl_cross_25=99.0,
                        return_cross_25_pct=99.0, realized_count=10000,
                    )
                    for _ in range(50)
                ])
                self._write_trade_log_fixture(Path(tmp), rows)
                prior = _load_prior_from_trade_log(
                    structure_kind="atm_straddle",
                    prior_structure_label="atm_straddle",
                )
        self.assertIsNotNone(prior)
        # 2 baseline trades, equal weight → 1 winner of 2 = 0.50.
        # If realized_count weighting leaked back in, this would
        # be ~0.001 instead.
        self.assertEqual(prior.history_count, 2)
        self.assertAlmostEqual(prior.win_rate, 0.50, places=10)

    def test_returns_none_when_trade_log_missing(self):
        """Empty reports dir → loader returns None so the caller
        can route to the scoreboard-legacy fallback."""
        from services.structure_scorecard import _load_prior_from_trade_log
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                prior = _load_prior_from_trade_log(
                    structure_kind="atm_straddle",
                    prior_structure_label="atm_straddle",
                )
        self.assertIsNone(prior)

    def test_returns_none_when_no_rows_match_baseline(self):
        """File present but no baseline-qualifying rows → returns
        None. Caller falls through to fallback (which may emit a
        neutral prior). Different from the "file missing" case but
        produces the same caller-level fallback behavior."""
        from services.structure_scorecard import _load_prior_from_trade_log
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                # All rows fail at least one baseline criterion.
                self._write_trade_log_fixture(Path(tmp), [
                    self._baseline_row(entry_offset_bdays=2),
                    self._baseline_row(entry_pass_oi_100=False),
                ])
                prior = _load_prior_from_trade_log(
                    structure_kind="atm_straddle",
                    prior_structure_label="atm_straddle",
                )
        self.assertIsNone(prior)

    def test_straddle_loader_fallback_chain_source_labels(self):
        """End-to-end fallback chain for straddle:

          * Trade log present + has baseline rows → source carries
            "trade_log:baseline".
          * Trade log MISSING + scoreboard present → falls back to
            scoreboard with "legacy_scoreboard_fallback" label.
          * Both missing → neutral prior with
            "neutral_missing_strangle_report" label.

        The label values are the contract downstream telemetry
        depends on to distinguish "clean primary path" from
        "contaminated fallback."
        """
        from services.structure_scorecard import (
            _load_straddle_prior_from_reports,
            _NEUTRAL_PRIOR_HISTORY_COUNT,
        )
        import tempfile
        from pathlib import Path

        # Case 1: primary path with valid trade log.
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                self._write_trade_log_fixture(Path(tmp), [
                    self._baseline_row(pnl_cross_25=1.0, return_cross_25_pct=5.0),
                    self._baseline_row(pnl_cross_25=-1.0, return_cross_25_pct=-5.0),
                ])
                prior = _load_straddle_prior_from_reports()
        self.assertIn("trade_log:baseline", prior.source)
        self.assertNotIn("legacy_scoreboard_fallback", prior.source)

        # Case 2: trade log missing, scoreboard present.
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                import pandas as pd
                run_dir = Path(tmp) / "pre_earnings_otm_strangle_20260101T000000Z"
                run_dir.mkdir()
                # Minimal scoreboard with one straddle/cross_25 row.
                pd.DataFrame([{
                    "structure": "atm_straddle",
                    "execution_scenario": "cross_25",
                    "realized_count": 10,
                    "win_rate": 0.55,
                    "avg_pnl_per_trade": 0.5,
                }]).to_csv(run_dir / "pre_earnings_otm_strangle_scoreboard.csv", index=False)
                prior = _load_straddle_prior_from_reports()
        self.assertIn("legacy_scoreboard_fallback", prior.source)

        # Case 3: both missing — neutral prior.
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                prior = _load_straddle_prior_from_reports()
        self.assertEqual(prior.history_count, _NEUTRAL_PRIOR_HISTORY_COUNT)
        self.assertIn("neutral_missing", prior.source)

    def test_strangle_primary_path_baseline_cost_table_unchanged(self):
        """Codex pinned this explicitly: PR #71 must NOT touch the
        strangle's primary path (the summary JSON
        ``baseline_cost_table``). The trade-log path is only the
        secondary fallback. Pin the source-label shape so a future
        refactor that accidentally reroutes the primary path fails
        this test."""
        from services.structure_scorecard import _load_strangle_prior_from_reports
        import tempfile
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                run_dir = Path(tmp) / "pre_earnings_otm_strangle_20260101T000000Z"
                run_dir.mkdir()
                (run_dir / "pre_earnings_otm_strangle_summary.json").write_text(
                    json.dumps({
                        "baseline_cost_table": [
                            {
                                "execution_scenario": "cross_25",
                                "realized_count": 42,
                                "win_rate": 0.6905,
                                "avg_return_pct": 9.0,
                            },
                        ],
                    }),
                    encoding="utf-8",
                )
                prior = _load_strangle_prior_from_reports()
        # Primary path source label is "<run>:baseline:<scenario>"
        # — no "trade_log" or "legacy_scoreboard_fallback".
        self.assertIn(":baseline:", prior.source)
        self.assertNotIn("trade_log", prior.source)
        self.assertNotIn("legacy_scoreboard_fallback", prior.source)
        self.assertEqual(prior.history_count, 42)
        self.assertAlmostEqual(prior.win_rate, 0.6905, places=4)

    def test_newer_run_without_trade_log_does_not_fall_back_to_older_trade_log(self):
        """Codex P2 regression on PR #71 first review.

        Pre-fix bug: ``_load_prior_from_trade_log`` called
        ``_latest_report_file("...trade_log.csv")`` — which returns
        the latest file matching the glob across ALL run
        directories. If a newer run had a scoreboard but no trade
        log, the straddle loader silently used the OLDER run's
        trade log instead of the newer run's scoreboard.

        Codex's exact repro: older run with trade log, newer run
        with scoreboard only. The loader should prefer the newer
        run (scoreboard fallback inside it) — NOT pick a stale
        trade log from a different point in time.

        Fix: every loader now scopes through
        ``_latest_strangle_run_dir()`` first, so the fallback chain
        operates within a single run snapshot.

        This test pins the contract by name AND by source label —
        the post-fix loader must report
        ``legacy_scoreboard_fallback`` for the NEWER run, not
        ``trade_log:baseline`` for the older one.
        """
        from services.structure_scorecard import _load_straddle_prior_from_reports
        import pandas as pd
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "services.structure_scorecard.REPORTS_ROOT", Path(tmp),
            ):
                # OLDER run — has trade log only.
                older_dir = Path(tmp) / "pre_earnings_otm_strangle_20260101T000000Z"
                older_dir.mkdir()
                self._write_trade_log_fixture(Path(tmp), [
                    self._baseline_row(pnl_cross_25=1.0, return_cross_25_pct=5.0),
                    self._baseline_row(pnl_cross_25=2.0, return_cross_25_pct=10.0),
                ], suffix="20260101T000000Z")

                # NEWER run — has scoreboard, no trade log.
                newer_dir = Path(tmp) / "pre_earnings_otm_strangle_20260201T000000Z"
                newer_dir.mkdir()
                pd.DataFrame([{
                    "structure": "atm_straddle",
                    "execution_scenario": "cross_25",
                    "realized_count": 20,
                    "win_rate": 0.42,
                    "avg_pnl_per_trade": -1.0,
                }]).to_csv(
                    newer_dir / "pre_earnings_otm_strangle_scoreboard.csv",
                    index=False,
                )

                prior = _load_straddle_prior_from_reports()

        # The loader must have picked the NEWER run, not the older
        # trade log. Source label proves which path fired AND which
        # run dir was scoped.
        self.assertIn(
            "20260201T000000Z", prior.source,
            f"loader must scope to the newer run; got source: {prior.source}",
        )
        self.assertIn(
            "legacy_scoreboard_fallback", prior.source,
            "newer run has only a scoreboard, so the fallback must "
            "be the scoreboard-legacy path inside that run.",
        )
        self.assertNotIn(
            "20260101T000000Z", prior.source,
            "older run's trade log MUST NOT leak in as a substitute "
            "for the newer run's missing trade log.",
        )
        self.assertNotIn(
            "trade_log", prior.source,
            "newer run has no trade log; the trade_log path must "
            "not fire.",
        )

    def test_live_trade_log_has_required_schema_columns(self):
        """Schema-stability guard (Codex correction: column
        EXISTENCE only, no win_rate value pinning).

        If a backtest refactor drops or renames any of the columns
        the trade-log loader depends on, this test fails
        immediately rather than silently returning bogus priors.

        Skipped if no live trade log exists in the repo (e.g., CI
        on a clean Ubuntu runner that has no exports/reports/
        artifacts). Codex CI provisions reports so the check fires
        in practice."""
        from services.structure_scorecard import (
            _TRADE_LOG_REQUIRED_COLUMNS,
            _latest_report_file,
        )
        path = _latest_report_file(
            "pre_earnings_otm_strangle_*/pre_earnings_otm_strangle_trade_log.csv"
        )
        if path is None:
            self.skipTest(
                "No live trade log present — schema guard cannot run. "
                "This is expected on a fresh-clone CI runner; the "
                "guard fires in environments that have backtest "
                "artifacts."
            )
        import pandas as pd
        # Read only the header so we don't pay the I/O cost of
        # loading the full file just for a column check.
        head = pd.read_csv(path, nrows=0)
        missing = [
            col for col in _TRADE_LOG_REQUIRED_COLUMNS if col not in head.columns
        ]
        self.assertFalse(missing, (
            f"Trade-log schema regression: columns {missing} are "
            f"required by services/structure_scorecard.py's "
            f"_load_prior_from_trade_log but absent from "
            f"{path.name}. If the backtest was refactored to rename "
            f"or drop these, update _TRADE_LOG_REQUIRED_COLUMNS in "
            f"lock-step."
        ))


if __name__ == "__main__":
    unittest.main()
