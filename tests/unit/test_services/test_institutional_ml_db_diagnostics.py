import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from services.institutional_ml_db import InstitutionalMLDatabase


class TestInstitutionalDiagnostics(unittest.TestCase):
    def _seed_backtest_data(self, db: InstitutionalMLDatabase, session_id: str = "session_diag_1") -> None:
        """Seed deterministic trade/feature/label rows for diagnostics testing."""
        start = datetime(2024, 1, 2)
        rng = np.random.default_rng(42)

        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO backtest_sessions
                (session_id, strategy_name, start_date, end_date, universe, parameters,
                 total_trades, win_rate, total_pnl, sharpe_ratio, max_drawdown, calmar_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    "Calendar_Spread_WalkForward",
                    "2024-01-02",
                    "2024-04-30",
                    '["AAPL","MSFT"]',
                    "{}",
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            )

            for i in range(80):
                symbol = "AAPL" if i % 2 == 0 else "MSFT"
                trade_date = (start + timedelta(days=i)).date()
                event_date = (trade_date + timedelta(days=20 + i))

                setup_score = float(np.clip(0.28 + 0.0085 * i, 0.20, 0.97))
                crush_edge = float(np.clip((setup_score - 0.45) * 0.28, 0.0, 0.18))
                crush_confidence = float(np.clip(0.30 + 0.010 * i, 0.20, 0.95))

                base_net_return = -0.045 + 0.145 * setup_score
                noise = float(rng.normal(0.0, 0.012))
                net_return_pct = float(np.clip(base_net_return + noise, -0.45, 0.45))
                debit_per_contract = float(280.0 + 40.0 * setup_score)
                tx_cost = float(3.0 + (1.0 - setup_score) * 3.0)
                pnl_per_contract = float(debit_per_contract * net_return_pct - tx_cost)

                days_to_earnings = int(2 + (i % 12))
                hold_days = int(5 + (i % 4))
                contracts = 1

                vix_level = [13.5, 19.0, 29.0][i % 3]
                iv30_rv30_ratio = float(np.clip(0.92 + 0.70 * setup_score + 0.03 * (i % 3), 0.8, 2.2))
                predicted_front_iv_crush = float(np.clip(-0.07 - 0.19 * setup_score, -0.45, -0.01))

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO ml_features
                    (symbol, date, underlying_price, iv_rank, iv_percentile, iv30_rv30_ratio, vix_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        trade_date.strftime("%Y-%m-%d"),
                        100.0 + i,
                        setup_score,
                        setup_score * 100.0,
                        iv30_rv30_ratio,
                        vix_level,
                    ),
                )

                cursor.execute(
                    """
                    INSERT INTO backtest_trades
                    (session_id, symbol, trade_date, event_date, days_to_earnings, contracts, hold_days,
                     setup_score, debit_per_contract, transaction_cost_per_contract, gross_return_pct,
                     net_return_pct, pnl_per_contract, underlying_return, expected_move, move_ratio,
                     predicted_front_iv_crush_pct, crush_confidence, crush_edge_score,
                     crush_profile_sample_size, execution_profile)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        symbol,
                        trade_date.strftime("%Y-%m-%d"),
                        event_date.strftime("%Y-%m-%d"),
                        days_to_earnings,
                        contracts,
                        hold_days,
                        setup_score,
                        debit_per_contract,
                        tx_cost,
                        net_return_pct + 0.01,
                        net_return_pct,
                        pnl_per_contract,
                        float(rng.normal(0.0, 0.025)),
                        float(np.clip(0.015 + 0.02 * setup_score, 0.01, 0.08)),
                        float(abs(rng.normal(0.7, 0.2))),
                        predicted_front_iv_crush,
                        crush_confidence,
                        crush_edge,
                        float(5 + (i % 10)),
                        "institutional",
                    ),
                )

                # Label most events so crush calibration metrics are non-empty.
                if i % 5 != 0:
                    realized_front = float(np.clip(predicted_front_iv_crush + rng.normal(0.0, 0.03), -0.5, 0.2))
                    pre_front = float(0.50 + 0.08 * setup_score)
                    post_front = float(pre_front * (1.0 + realized_front))
                    pre_back = float(0.40 + 0.05 * setup_score)
                    post_back = float(pre_back * (1.0 + realized_front * 0.35))
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO earnings_iv_decay_labels
                        (symbol, event_date, release_timing, pre_capture_date, post_capture_date,
                         pre_front_iv, post_front_iv, pre_back_iv, post_back_iv,
                         front_iv_crush_pct, back_iv_crush_pct, term_ratio_change,
                         underlying_move_pct, quality_score, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            event_date.strftime("%Y-%m-%d"),
                            "AMC",
                            (event_date - timedelta(days=2)).strftime("%Y-%m-%d"),
                            (event_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                            pre_front,
                            post_front,
                            pre_back,
                            post_back,
                            realized_front,
                            float((post_back - pre_back) / max(pre_back, 1e-6)),
                            float((post_front / max(post_back, 1e-6)) - (pre_front / max(pre_back, 1e-6))),
                            float(rng.normal(0.0, 0.025)),
                            float(np.clip(0.70 + rng.normal(0.0, 0.08), 0.20, 1.0)),
                            "unit_test",
                        ),
                    )

            conn.commit()

    def test_regime_diagnostics_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/inst_diag.db"
            db = InstitutionalMLDatabase(db_path=db_path)
            self._seed_backtest_data(db)

            regime_df = db.build_regime_diagnostics(
                session_id="session_diag_1",
                min_confidence=0.30,
            )
            self.assertFalse(regime_df.empty)
            self.assertIn("vix_regime", regime_df.columns)
            self.assertIn("alpha_proxy_score", regime_df.columns)
            self.assertTrue(np.isfinite(regime_df["alpha_proxy_score"]).all())

            summary = db.summarize_regime_diagnostics(
                session_id="session_diag_1",
                min_confidence=0.30,
                top_n=3,
            )
            self.assertGreater(summary["rows"], 0)
            self.assertGreater(len(summary["best_regimes"]), 0)
            self.assertGreater(len(summary["weak_regimes"]), 0)

    def test_decile_table_and_threshold_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/inst_tune.db"
            db = InstitutionalMLDatabase(db_path=db_path)
            self._seed_backtest_data(db)

            decile_df = db.build_signal_decile_table(
                session_id="session_diag_1",
                min_confidence=0.30,
                use_composite_signal=True,
            )
            self.assertFalse(decile_df.empty)
            self.assertIn("signal_decile", decile_df.columns)
            self.assertIn("mean_net_return_pct", decile_df.columns)

            recommendation = db.recommend_signal_threshold_from_deciles(
                session_id="session_diag_1",
                min_confidence=0.30,
                min_trades=8,
                use_composite_signal=True,
            )
            self.assertTrue(recommendation.get("recommended"))
            self.assertGreater(recommendation.get("candidate_count", 0), 0)
            threshold = float(recommendation.get("best_threshold", -1.0))
            self.assertGreaterEqual(threshold, 0.0)
            self.assertLessEqual(threshold, 1.0)

    def test_proxy_schedule_generation_requires_sufficient_history(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = InstitutionalMLDatabase(db_path=f"{tmp_dir}/inst_proxy.db")
            window_start = datetime(2024, 1, 15)
            window_end = datetime(2024, 3, 31)

            short_history = pd.Series(pd.bdate_range(start="2024-01-02", periods=20))
            self.assertEqual(
                db._build_proxy_earnings_schedule(short_history, window_start, window_end),
                [],
            )

            window_start = datetime(2024, 5, 1)
            window_end = datetime(2024, 12, 31)
            long_history = pd.Series(pd.bdate_range(start="2024-01-02", periods=220))
            proxy_events = db._build_proxy_earnings_schedule(long_history, window_start, window_end)
            self.assertGreater(len(proxy_events), 0)

    def test_proxy_schedule_projects_future_dates_when_window_extends_forward(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = InstitutionalMLDatabase(db_path=f"{tmp_dir}/inst_proxy_future.db")
            history = pd.Series(pd.bdate_range(start="2025-01-02", periods=250))
            history_last = pd.Timestamp(history.max()).to_pydatetime().date()

            window_start = datetime(2025, 9, 1)
            window_end = datetime(2026, 3, 31)
            proxy_events = db._build_proxy_earnings_schedule(history, window_start, window_end)
            self.assertGreater(len(proxy_events), 0)
            self.assertTrue(
                any(pd.Timestamp(item["event_date"]).date() > history_last for item in proxy_events)
            )

    def test_proxy_schedule_symbol_phase_offsets_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = InstitutionalMLDatabase(db_path=f"{tmp_dir}/inst_proxy_phase.db")
            history = pd.Series(pd.bdate_range(start="2025-01-02", periods=280))
            window_start = datetime(2025, 1, 1)
            window_end = datetime(2026, 3, 31)

            events_aapl = db._build_proxy_earnings_schedule(history, window_start, window_end, symbol="AAPL")
            events_msft = db._build_proxy_earnings_schedule(history, window_start, window_end, symbol="MSFT")
            self.assertGreater(len(events_aapl), 0)
            self.assertGreater(len(events_msft), 0)
            dates_aapl = {pd.Timestamp(item["event_date"]).date() for item in events_aapl}
            dates_msft = {pd.Timestamp(item["event_date"]).date() for item in events_msft}
            self.assertNotEqual(dates_aapl, dates_msft)

    def test_get_symbol_earnings_dates_falls_back_to_proxy_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = InstitutionalMLDatabase(db_path=f"{tmp_dir}/inst_proxy_fallback.db")
            trading_dates = pd.Series(pd.bdate_range(start="2024-01-02", periods=240))
            start_date = datetime(2024, 7, 1)
            end_date = datetime(2024, 12, 31)

            original_fetch = db._fetch_and_cache_earnings_dates
            db._fetch_and_cache_earnings_dates = lambda *_args, **_kwargs: []
            try:
                proxy_events = db._get_symbol_earnings_dates(
                    symbol="AAPL",
                    start_date=start_date,
                    end_date=end_date,
                    trading_dates=trading_dates,
                    require_true_earnings=False,
                    allow_proxy_earnings=True,
                )
                self.assertGreater(len(proxy_events), 0)

                cached = db._get_cached_earnings_dates(
                    symbol="AAPL",
                    window_start=start_date - timedelta(days=60),
                    window_end=end_date + timedelta(days=30),
                )
                self.assertGreater(len(cached), 0)
                self.assertTrue(any(source == "proxy" for _, source, _ in cached))

                true_only = db._get_symbol_earnings_dates(
                    symbol="AAPL",
                    start_date=start_date,
                    end_date=end_date,
                    trading_dates=trading_dates,
                    require_true_earnings=True,
                    allow_proxy_earnings=True,
                )
                self.assertEqual(true_only, [])
            finally:
                db._fetch_and_cache_earnings_dates = original_fetch

    def test_get_symbol_earnings_dates_refreshes_stale_proxy_cache_without_future_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = InstitutionalMLDatabase(db_path=f"{tmp_dir}/inst_proxy_refresh.db")
            today = datetime.now().date()
            trading_dates = pd.Series(pd.bdate_range(end=today, periods=260))
            start_date = datetime.combine(today - timedelta(days=30), datetime.min.time())
            end_date = datetime.combine(today + timedelta(days=120), datetime.min.time())
            stale_proxy_date = datetime.combine(today - timedelta(days=20), datetime.min.time())
            rebuilt_proxy_date = datetime.combine(today + timedelta(days=40), datetime.min.time())

            db._cache_earnings_dates(
                "AAPL",
                [{'event_date': stale_proxy_date, 'release_timing': 'UNKNOWN'}],
                source='proxy',
                release_timing='UNKNOWN',
            )

            original_fetch = db._fetch_and_cache_earnings_dates
            original_build = db._build_proxy_earnings_schedule
            db._fetch_and_cache_earnings_dates = lambda *_args, **_kwargs: []
            db._build_proxy_earnings_schedule = (
                lambda *_args, **_kwargs: [{'event_date': rebuilt_proxy_date, 'release_timing': 'UNKNOWN'}]
            )
            try:
                events = db._get_symbol_earnings_dates(
                    symbol="AAPL",
                    start_date=start_date,
                    end_date=end_date,
                    trading_dates=trading_dates,
                    require_true_earnings=False,
                    allow_proxy_earnings=True,
                )
            finally:
                db._fetch_and_cache_earnings_dates = original_fetch
                db._build_proxy_earnings_schedule = original_build

            event_dates = {pd.Timestamp(item['event_date']).date() for item in events}
            self.assertIn(rebuilt_proxy_date.date(), event_dates)
            self.assertNotIn(stale_proxy_date.date(), event_dates)

    def test_snapshot_pairing_progress_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = InstitutionalMLDatabase(db_path=f"{tmp_dir}/inst_snapshot_status.db")
            with sqlite3.connect(db.db_path) as conn:
                rows = [
                    ("AAPL", "2026-03-01", "2026-02-25", -4, "AMC", "pre", "2026-03-06", "2026-03-13", 200.0, 0.65, 0.55, 1.18, 205.0, "unit_test"),
                    ("AAPL", "2026-03-01", "2026-03-02", 1, "AMC", "post", "2026-03-06", "2026-03-13", 200.0, 0.48, 0.50, 0.96, 202.0, "unit_test"),
                    ("MSFT", "2026-03-05", "2026-03-01", -4, "BMO", "pre", "2026-03-06", "2026-03-13", 400.0, 0.52, 0.47, 1.11, 398.0, "unit_test"),
                    ("NVDA", "2026-03-07", "2026-03-08", 1, "AMC", "post", "2026-03-13", "2026-03-20", 900.0, 0.62, 0.59, 1.05, 910.0, "unit_test"),
                    ("TSLA", "2026-03-10", "2026-03-10", 0, "UNKNOWN", "neutral", "2026-03-13", "2026-03-20", 180.0, 0.55, 0.53, 1.04, 181.0, "unit_test"),
                ]
                conn.executemany(
                    """
                    INSERT INTO earnings_option_snapshots
                    (symbol, event_date, capture_date, relative_day, release_timing, snapshot_phase,
                     short_expiry, long_expiry, atm_strike, front_iv, back_iv, term_ratio,
                     underlying_price, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()

            status = db.summarize_snapshot_pairing_progress(
                min_pre_days=1,
                max_pre_days=12,
                min_post_days=0,
                max_post_days=5,
            )
            self.assertEqual(status["total_snapshots"], 5)
            self.assertEqual(status["total_events"], 4)
            self.assertEqual(status["events_with_pre"], 2)
            self.assertEqual(status["events_with_post"], 2)
            self.assertEqual(status["pairable_events"], 1)
            self.assertEqual(status["pending_pre_only_events"], 1)
            self.assertEqual(status["pending_post_only_events"], 1)
            self.assertEqual(status["unqualified_events"], 1)
            self.assertAlmostEqual(status["pairable_event_pct"], 0.25)
            self.assertEqual(status["capture_days"], 5)
            self.assertEqual(status["min_relative_day"], -4)
            self.assertEqual(status["max_relative_day"], 1)


if __name__ == "__main__":
    unittest.main()
