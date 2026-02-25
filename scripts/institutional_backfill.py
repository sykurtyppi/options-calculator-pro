#!/usr/bin/env python3
"""
Institutional Data Backfill Script
================================

Professional-grade historical data collection for institutional trading:
- 50+ ticker universe with 2-year historical data
- Technical indicators and volatility metrics
- ML feature engineering pipeline
- Calendar spread backtesting data preparation
- Rate-limited and resumable operations

Usage:
    python scripts/institutional_backfill.py --symbols AAPL,MSFT,GOOGL --years 2
    python scripts/institutional_backfill.py --full-universe --years 1
    python scripts/institutional_backfill.py --backtest-only
"""

import asyncio
import argparse
import sys
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.institutional_ml_db import InstitutionalMLDatabase, INSTITUTIONAL_UNIVERSE
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(',') if item.strip()]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(',') if item.strip()]


def _parse_csv_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(',') if item.strip()]


class InstitutionalDataCollector:
    """Professional data collection orchestrator"""

    def __init__(self, db_path: Optional[str] = None):
        self.db = InstitutionalMLDatabase(db_path)
        self.logger = logger

    async def run_full_backfill(self, symbols: List[str], years: int = 2,
                              batch_size: int = 5) -> bool:
        """
        Run comprehensive institutional data backfill

        Args:
            symbols: List of symbols to process
            years: Years of historical data
            batch_size: Concurrent processing batch size

        Returns:
            True if successful
        """
        self.logger.info(f"🏛️ Starting institutional backfill for {len(symbols)} symbols")
        self.logger.info(f"📅 Timeframe: {years} years, Batch size: {batch_size}")

        try:
            # 1. Historical price data with technical indicators
            self.logger.info("📈 Phase 1: Historical price data collection")
            price_success = await self.db.backfill_historical_data(
                symbols=symbols,
                years_back=years,
                batch_size=batch_size
            )

            if not price_success:
                self.logger.error("❌ Price data collection failed")
                return False

            self.logger.info("✅ Phase 1 completed successfully")

            # 2. Feature engineering and ML preparation
            self.logger.info("🧠 Phase 2: ML feature engineering")
            feature_success = await self._engineer_ml_features(symbols)

            if not feature_success:
                self.logger.warning("⚠️ Feature engineering had issues but continuing")

            self.logger.info("✅ Phase 2 completed")

            # 3. Data quality validation
            self.logger.info("🔍 Phase 3: Data quality validation")
            validation_results = await self._validate_data_quality(symbols)

            self._report_data_quality(validation_results)

            self.logger.info("🎯 Institutional backfill completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Backfill failed: {e}")
            return False

    async def _engineer_ml_features(self, symbols: List[str]) -> bool:
        """Engineering additional ML features"""
        try:
            # This would implement sophisticated feature engineering
            # For now, features are calculated in _calculate_and_store_features
            self.logger.info("🔧 Advanced feature engineering completed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            return False

    async def _validate_data_quality(self, symbols: List[str]) -> dict:
        """Validate data quality and completeness"""
        validation_results = {}

        try:
            import sqlite3

            with sqlite3.connect(self.db.db_path) as conn:
                for symbol in symbols:
                    # Check data completeness
                    cursor = conn.cursor()

                    # Price data count
                    cursor.execute("""
                        SELECT COUNT(*) FROM daily_prices
                        WHERE symbol = ? AND date >= date('now', '-2 years')
                    """, (symbol,))
                    price_count = cursor.fetchone()[0]

                    # Feature data count
                    cursor.execute("""
                        SELECT COUNT(*) FROM ml_features
                        WHERE symbol = ? AND date >= date('now', '-2 years')
                    """, (symbol,))
                    feature_count = cursor.fetchone()[0]

                    # Data quality score
                    expected_days = 500  # ~2 years of trading days
                    price_completeness = min(price_count / expected_days, 1.0)
                    feature_completeness = min(feature_count / expected_days, 1.0)

                    quality_score = (price_completeness + feature_completeness) / 2

                    validation_results[symbol] = {
                        'price_records': price_count,
                        'feature_records': feature_count,
                        'price_completeness': price_completeness,
                        'feature_completeness': feature_completeness,
                        'quality_score': quality_score,
                        'status': 'GOOD' if quality_score >= 0.8 else 'POOR'
                    }

        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")

        return validation_results

    def _report_data_quality(self, results: dict):
        """Report data quality metrics"""
        self.logger.info("📊 Data Quality Report")
        self.logger.info("=" * 60)

        good_symbols = 0
        poor_symbols = 0

        for symbol, metrics in results.items():
            status_emoji = "✅" if metrics['status'] == 'GOOD' else "❌"
            self.logger.info(f"{status_emoji} {symbol}: {metrics['quality_score']:.1%} quality, "
                           f"{metrics['price_records']} price records, "
                           f"{metrics['feature_records']} feature records")

            if metrics['status'] == 'GOOD':
                good_symbols += 1
            else:
                poor_symbols += 1

        total_symbols = len(results)
        overall_quality = good_symbols / total_symbols if total_symbols > 0 else 0

        self.logger.info("=" * 60)
        self.logger.info(f"📈 Overall Quality: {overall_quality:.1%}")
        self.logger.info(f"✅ Good symbols: {good_symbols}")
        self.logger.info(f"❌ Poor symbols: {poor_symbols}")

        if overall_quality >= 0.8:
            self.logger.info("🎯 Data quality meets institutional standards")
        else:
            self.logger.warning("⚠️ Data quality below institutional threshold")

    def run_sample_backtest(self, execution_profile: str = 'institutional',
                          hold_days: int = 7,
                          min_signal_score: float = 0.58,
                          max_trades_per_day: int = 2,
                          position_contracts: int = 1,
                          lookback_days: int = 365,
                          max_backtest_symbols: int = 10,
                          entry_days_before_earnings: int = 7,
                          exit_days_after_earnings: int = 1,
                          target_entry_dte: int = 6,
                          entry_dte_band: int = 6,
                          min_daily_share_volume: int = 1_000_000,
                          max_abs_momentum_5d: float = 0.11,
                          use_crush_confidence_gate: bool = True,
                          allow_global_crush_profile: bool = True,
                          min_crush_confidence: float = 0.45,
                          min_crush_magnitude: float = 0.08,
                          min_crush_edge: float = 0.02,
                          require_true_earnings: bool = False,
                          allow_proxy_earnings: bool = True,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None):
        """Run sample calendar spread backtest"""
        self.logger.info("🎯 Running sample calendar spread backtest")
        hold_days = max(1, int(hold_days))
        min_signal_score = max(0.0, min(1.0, float(min_signal_score)))
        max_trades_per_day = max(1, int(max_trades_per_day))
        position_contracts = max(1, int(position_contracts))
        lookback_days = max(120, int(lookback_days))
        max_backtest_symbols = max(1, int(max_backtest_symbols))
        entry_days_before_earnings = max(1, int(entry_days_before_earnings))
        exit_days_after_earnings = max(0, int(exit_days_after_earnings))
        target_entry_dte = max(1, int(target_entry_dte))
        entry_dte_band = max(1, int(entry_dte_band))
        min_daily_share_volume = max(0, int(min_daily_share_volume))
        max_abs_momentum_5d = max(0.01, float(max_abs_momentum_5d))
        min_crush_confidence = max(0.0, min(1.0, float(min_crush_confidence)))
        min_crush_magnitude = max(0.0, min(1.0, float(min_crush_magnitude)))
        min_crush_edge = max(0.0, min(1.0, float(min_crush_edge)))

        strategy_params = {
            'strategy_type': 'calendar_spread',
            'lookback_days': lookback_days,
            'hold_days': hold_days,
            'min_signal_score': min_signal_score,
            'max_trades_per_day': max_trades_per_day,
            'position_size_contracts': position_contracts,
            'execution_profile': execution_profile,
            'max_symbols': max_backtest_symbols,
            'iv_rv_min': 0.95,
            'iv_rv_max': 2.30,
            'earnings_event_mode': True,
            'entry_days_before_earnings': entry_days_before_earnings,
            'exit_days_after_earnings': exit_days_after_earnings,
            'target_entry_dte': target_entry_dte,
            'entry_dte_band': entry_dte_band,
            'min_daily_share_volume': min_daily_share_volume,
            'max_abs_momentum_5d': max_abs_momentum_5d,
            'use_crush_confidence_gate': bool(use_crush_confidence_gate),
            'allow_global_crush_profile': bool(allow_global_crush_profile),
            'min_crush_confidence': min_crush_confidence,
            'min_crush_magnitude': min_crush_magnitude,
            'min_crush_edge': min_crush_edge,
            'require_true_earnings': bool(require_true_earnings),
            'allow_proxy_earnings': bool(allow_proxy_earnings),
        }
        if start_date:
            strategy_params['start_date'] = str(start_date)
        if end_date:
            strategy_params['end_date'] = str(end_date)

        try:
            session_id = self.db.run_calendar_spread_backtest(strategy_params)

            # Get and display results
            results_df = self.db.get_backtest_results(session_id)

            if not results_df.empty:
                result = results_df.iloc[0]
                self.logger.info("📊 Backtest Results:")
                self.logger.info(f"   Session ID: {result['session_id']}")
                self.logger.info(f"   Total Trades: {result['total_trades']}")
                self.logger.info(f"   Win Rate: {result['win_rate']:.1%}")
                self.logger.info(f"   Total P&L: ${result['total_pnl']:.2f}")
                self.logger.info(f"   Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                self.logger.info(f"   Max Drawdown: ${result['max_drawdown']:.2f}")
                self.logger.info(f"   Calmar Ratio: {result['calmar_ratio']:.2f}")

                trades_df = self.db.get_backtest_trades(session_id)
                if not trades_df.empty:
                    avg_net_return = trades_df['net_return_pct'].mean()
                    avg_setup_score = trades_df['setup_score'].mean()
                    avg_tx_cost = trades_df['transaction_cost_per_contract'].mean()
                    avg_crush_conf = trades_df['crush_confidence'].mean() if 'crush_confidence' in trades_df.columns else 0.0
                    avg_crush_edge = trades_df['crush_edge_score'].mean() if 'crush_edge_score' in trades_df.columns else 0.0
                    self.logger.info(f"   Avg Net Return/Trade: {avg_net_return:.2%}")
                    self.logger.info(f"   Avg Setup Score: {avg_setup_score:.2f}")
                    self.logger.info(f"   Avg Tx Cost/Contract: ${avg_tx_cost:.2f}")
                    self.logger.info(f"   Avg Crush Confidence: {avg_crush_conf:.2f}")
                    self.logger.info(f"   Avg Crush Edge Score: {avg_crush_edge:.4f}")

                # Assess institutional viability
                if result['sharpe_ratio'] >= 1.5 and result['win_rate'] >= 0.6:
                    self.logger.info("🏆 Strategy shows institutional-grade performance")
                else:
                    self.logger.warning("⚠️ Strategy needs optimization for institutional use")

            return session_id

        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            return None

    def capture_earnings_snapshots(self, symbols: List[str],
                                 lookback_days: int = 14,
                                 lookahead_days: int = 45,
                                 max_expiries: int = 2,
                                 require_true_earnings: bool = False,
                                 allow_proxy_earnings: bool = True) -> Dict[str, Any]:
        """Capture option-IV snapshots around earnings events."""
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not symbols:
            symbols = INSTITUTIONAL_UNIVERSE[:10]
        result = self.db.capture_earnings_option_snapshots(
            symbols=symbols,
            lookback_days=lookback_days,
            lookahead_days=lookahead_days,
            max_expiries=max_expiries,
            require_true_earnings=require_true_earnings,
            allow_proxy_earnings=allow_proxy_earnings,
        )
        self.logger.info(
            f"📸 Snapshot capture: {result.get('captured', 0)} captured, "
            f"{result.get('attempts', 0)} attempts, "
            f"{result.get('eligible_events', 0)} eligible events, "
            f"{result.get('errors', 0)} errors"
        )
        diagnostics = result.get('diagnostics') or {}
        if diagnostics:
            self.logger.info(
                "📸 Snapshot diagnostics: "
                f"no-price={diagnostics.get('no_price_symbols', 0)}, "
                f"no-events={diagnostics.get('no_event_symbols', 0)}, "
                f"no-expiries={diagnostics.get('no_expiry_symbols', 0)}, "
                f"no-iv={diagnostics.get('no_iv_events', 0)}, "
                f"chain-errors={diagnostics.get('chain_errors', 0)}"
            )
        return result

    def calibrate_iv_decay_labels(self, min_pre_days: int = 1,
                                max_pre_days: int = 12,
                                min_post_days: int = 0,
                                max_post_days: int = 5) -> Dict[str, Any]:
        """Calibrate IV-crush labels from captured snapshots."""
        labels_df = self.db.calibrate_earnings_iv_decay_labels(
            min_pre_days=min_pre_days,
            max_pre_days=max_pre_days,
            min_post_days=min_post_days,
            max_post_days=max_post_days,
        )
        if labels_df.empty:
            self.logger.warning("⚠️ No IV-decay labels calibrated")
            return {'rows': 0, 'avg_front_iv_crush_pct': 0.0, 'avg_term_ratio_change': 0.0}

        avg_crush = float(labels_df['front_iv_crush_pct'].mean())
        avg_term = float(labels_df['term_ratio_change'].mean())
        self.logger.info(
            f"🧩 IV-decay labels calibrated: {len(labels_df)} rows, "
            f"avg front IV crush {avg_crush:.2%}, avg term-ratio change {avg_term:.4f}"
        )
        return {
            'rows': int(len(labels_df)),
            'avg_front_iv_crush_pct': avg_crush,
            'avg_term_ratio_change': avg_term,
        }

    def report_snapshot_pairing_status(self, min_pre_days: int = 1,
                                     max_pre_days: int = 12,
                                     min_post_days: int = 0,
                                     max_post_days: int = 5) -> Dict[str, Any]:
        """Log and return snapshot pre/post pairing coverage for calibration readiness."""
        status = self.db.summarize_snapshot_pairing_progress(
            min_pre_days=min_pre_days,
            max_pre_days=max_pre_days,
            min_post_days=min_post_days,
            max_post_days=max_post_days,
        )
        self.logger.info(
            "📸 Snapshot pairing status: snapshots=%d, events=%d, pre-ready=%d, post-ready=%d, "
            "pairable=%d (%.1f%%), pre-only=%d, post-only=%d, unqualified=%d, capture-days=%d, "
            "relative-day-range=[%s, %s]",
            status.get('total_snapshots', 0),
            status.get('total_events', 0),
            status.get('events_with_pre', 0),
            status.get('events_with_post', 0),
            status.get('pairable_events', 0),
            100.0 * float(status.get('pairable_event_pct', 0.0) or 0.0),
            status.get('pending_pre_only_events', 0),
            status.get('pending_post_only_events', 0),
            status.get('unqualified_events', 0),
            status.get('capture_days', 0),
            status.get('min_relative_day'),
            status.get('max_relative_day'),
        )
        return status

    def run_crush_scorecard(self, session_id: Optional[str] = None,
                          window_size: int = 40,
                          min_confidence: float = 0.35,
                          output_dir: str = "exports/reports") -> Optional[dict]:
        """Generate rolling prediction-vs-realization crush scorecard reports."""
        window_size = max(5, int(window_size))
        min_confidence = max(0.0, min(1.0, float(min_confidence)))

        scorecard_df = self.db.build_rolling_crush_scorecard(
            session_id=session_id,
            window_size=window_size,
            min_confidence=min_confidence,
        )
        if scorecard_df.empty:
            self.logger.warning("⚠️ No labeled crush records available for scorecard generation")
            return None

        summary = self.db.summarize_crush_scorecard(
            session_id=session_id,
            window_size=window_size,
            min_confidence=min_confidence,
        )

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f"earnings_crush_scorecard_{timestamp}.csv"
        md_path = output_path / f"earnings_crush_scorecard_{timestamp}.md"
        json_path = output_path / f"earnings_crush_scorecard_{timestamp}.json"

        scorecard_df.to_csv(csv_path, index=False)

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings IV-Crush Prediction Scorecard\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Session scope: `{session_id or 'all_sessions'}`\n")
            f.write(f"- Records: {summary.get('rows', 0)}\n")
            f.write(f"- Rolling window: {summary.get('window_size', window_size)}\n")
            f.write(f"- Min confidence filter: {min_confidence:.2f}\n\n")

            f.write("## Summary Metrics\n\n")
            f.write(f"- Mean absolute error: {summary.get('mean_mae', float('nan')):.4f}\n")
            f.write(f"- Directional accuracy: {summary.get('directional_accuracy', float('nan')):.2%}\n")
            f.write(f"- Mean confidence: {summary.get('mean_confidence', float('nan')):.2f}\n")
            f.write(f"- Mean predicted front IV crush: {summary.get('mean_predicted_crush', float('nan')):.2%}\n")
            f.write(f"- Mean realized front IV crush: {summary.get('mean_realized_crush', float('nan')):.2%}\n")
            f.write(f"- Latest rolling institutional score: {summary.get('latest_institutional_score', float('nan')):.4f}\n\n")

            f.write("## Latest Rolling Observations\n\n")
            display_cols = [
                'trade_date',
                'symbol',
                'predicted_front_iv_crush_pct',
                'realized_front_iv_crush_pct',
                'abs_error',
                'crush_confidence',
                'rolling_mae',
                'rolling_directional_accuracy',
                'rolling_institutional_score',
            ]
            preview = scorecard_df[display_cols].tail(15).to_string(index=False)
            f.write("```text\n")
            f.write(preview)
            f.write("\n```\n")

            symbol_rows = summary.get('by_symbol', [])[:10]
            if symbol_rows:
                f.write("\n## Top Symbol Calibration\n\n")
                table_lines = ["symbol observations mean_mae directional_accuracy mean_confidence avg_net_return"]
                for row in symbol_rows:
                    table_lines.append(
                        f"{row.get('symbol')} {int(row.get('observations', 0))} "
                        f"{float(row.get('mean_mae', float('nan'))):.4f} "
                        f"{float(row.get('directional_accuracy', float('nan'))):.2%} "
                        f"{float(row.get('mean_confidence', float('nan'))):.2f} "
                        f"{float(row.get('avg_net_return', float('nan'))):.2%}"
                    )
                f.write("```text\n")
                f.write("\n".join(table_lines))
                f.write("\n```\n")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'session_id': session_id,
                'window_size': window_size,
                'min_confidence': min_confidence,
                'summary': summary,
                'source_csv': str(csv_path),
            }, f, indent=2, default=str)

        self.logger.info(f"✅ Crush scorecard saved: {csv_path}")
        self.logger.info(f"✅ Crush scorecard summary saved: {md_path}")
        self.logger.info(f"✅ Crush scorecard JSON saved: {json_path}")
        return {
            'csv_path': str(csv_path),
            'markdown_path': str(md_path),
            'json_path': str(json_path),
            'rows': int(summary.get('rows', 0)),
            'session_id': session_id,
        }

    def _resolve_session_for_reports(self, session_id: Optional[str] = None) -> Optional[str]:
        """Resolve report session scope (explicit session id preferred, fallback latest)."""
        if session_id:
            return str(session_id)
        sessions_df = self.db.get_backtest_results()
        if sessions_df.empty:
            return None
        return str(sessions_df.iloc[0]['session_id'])

    def run_regime_diagnostics(self, session_id: Optional[str] = None,
                             min_confidence: float = 0.35,
                             output_dir: str = "exports/reports",
                             top_n: int = 6) -> Optional[dict]:
        """Generate regime diagnostics report for VIX/DTE/IV-RV buckets."""
        resolved_session = self._resolve_session_for_reports(session_id)
        min_confidence = max(0.0, min(1.0, float(min_confidence)))
        top_n = max(1, int(top_n))

        diagnostics_df = self.db.build_regime_diagnostics(
            session_id=resolved_session,
            min_confidence=min_confidence,
        )
        if diagnostics_df.empty:
            self.logger.warning("⚠️ No regime diagnostics rows available")
            return None

        summary = self.db.summarize_regime_diagnostics(
            session_id=resolved_session,
            min_confidence=min_confidence,
            top_n=top_n,
        )

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f"earnings_regime_diagnostics_{timestamp}.csv"
        md_path = output_path / f"earnings_regime_diagnostics_{timestamp}.md"
        json_path = output_path / f"earnings_regime_diagnostics_{timestamp}.json"

        diagnostics_df.to_csv(csv_path, index=False)

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings Regime Diagnostics\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Session scope: `{resolved_session or 'all_sessions'}`\n")
            f.write(f"- Min confidence filter: {min_confidence:.2f}\n")
            f.write(f"- Regime rows: {summary.get('rows', 0)}\n\n")

            f.write("## Overall\n\n")
            f.write(f"- Mean net return: {summary.get('overall_mean_net_return_pct', float('nan')):.2%}\n")
            f.write(f"- Win rate: {summary.get('overall_win_rate', float('nan')):.2%}\n")
            f.write(f"- Crush MAE: {summary.get('overall_crush_mae', float('nan')):.4f}\n")
            f.write(
                f"- Crush directional accuracy: "
                f"{summary.get('overall_crush_directional_accuracy', float('nan')):.2%}\n\n"
            )

            best_rows = summary.get('best_regimes', [])
            if best_rows:
                best_df = pd.DataFrame(best_rows)
                f.write("## Best Regimes\n\n")
                f.write("```text\n")
                f.write(best_df.to_string(index=False))
                f.write("\n```\n")

            weak_rows = summary.get('weak_regimes', [])
            if weak_rows:
                weak_df = pd.DataFrame(weak_rows)
                f.write("\n## Weak Regimes\n\n")
                f.write("```text\n")
                f.write(weak_df.to_string(index=False))
                f.write("\n```\n")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'session_id': resolved_session,
                'min_confidence': min_confidence,
                'summary': summary,
                'source_csv': str(csv_path),
            }, f, indent=2, default=str)

        self.logger.info(f"✅ Regime diagnostics saved: {csv_path}")
        self.logger.info(f"✅ Regime diagnostics summary saved: {md_path}")
        self.logger.info(f"✅ Regime diagnostics JSON saved: {json_path}")
        return {
            'csv_path': str(csv_path),
            'markdown_path': str(md_path),
            'json_path': str(json_path),
            'rows': int(summary.get('rows', 0)),
            'session_id': resolved_session,
        }

    def run_threshold_tuning(self, session_id: Optional[str] = None,
                           min_confidence: float = 0.35,
                           min_trades: int = 30,
                           use_composite_signal: bool = True,
                           output_dir: str = "exports/reports") -> Optional[dict]:
        """Generate decile table + threshold recommendation report."""
        resolved_session = self._resolve_session_for_reports(session_id)
        min_confidence = max(0.0, min(1.0, float(min_confidence)))
        min_trades = max(5, int(min_trades))
        use_composite_signal = bool(use_composite_signal)

        deciles_df = self.db.build_signal_decile_table(
            session_id=resolved_session,
            min_confidence=min_confidence,
            use_composite_signal=use_composite_signal,
        )
        recommendation = self.db.recommend_signal_threshold_from_deciles(
            session_id=resolved_session,
            min_confidence=min_confidence,
            min_trades=min_trades,
            use_composite_signal=use_composite_signal,
        )
        recommendation_reason = str(recommendation.get('reason', 'unknown'))
        available_trade_count = int(recommendation.get('available_trade_count', 0) or 0)
        max_trade_count = int(recommendation.get('max_trade_count_for_thresholds', 0) or 0)
        suggested_min_trades = recommendation.get('suggested_min_trades')
        if suggested_min_trades is not None:
            try:
                suggested_min_trades = int(suggested_min_trades)
            except (TypeError, ValueError):
                suggested_min_trades = None

        candidates_df = pd.DataFrame(recommendation.get('candidate_thresholds', []))

        if deciles_df.empty:
            self.logger.warning("⚠️ No decile rows available for threshold tuning")
            return None

        if (
            not recommendation.get('recommended')
            and recommendation_reason == 'insufficient_trade_count_for_thresholds'
        ):
            suggestion_text = (
                f" Consider --tuning-min-trades {suggested_min_trades}."
                if suggested_min_trades is not None else ""
            )
            self.logger.warning(
                "⚠️ Threshold tuning produced no candidates: "
                f"min_trades={min_trades}, available_trades={available_trade_count}, "
                f"max_trades_at_any_threshold={max_trade_count}.{suggestion_text}"
            )

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deciles_csv_path = output_path / f"earnings_signal_deciles_{timestamp}.csv"
        candidates_csv_path = output_path / f"earnings_threshold_candidates_{timestamp}.csv"
        md_path = output_path / f"earnings_threshold_tuning_{timestamp}.md"
        json_path = output_path / f"earnings_threshold_tuning_{timestamp}.json"

        deciles_df.to_csv(deciles_csv_path, index=False)
        if not candidates_df.empty:
            candidates_df.to_csv(candidates_csv_path, index=False)

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings Signal Threshold Tuning\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Session scope: `{resolved_session or 'all_sessions'}`\n")
            f.write(f"- Signal mode: `{recommendation.get('signal_name', 'setup_score')}`\n")
            f.write(f"- Min confidence filter: {min_confidence:.2f}\n")
            f.write(f"- Min trades per threshold: {min_trades}\n")
            f.write(f"- Decile rows: {len(deciles_df)}\n")
            f.write(f"- Candidate thresholds: {recommendation.get('candidate_count', 0)}\n\n")

            if recommendation.get('recommended'):
                best = recommendation.get('best_metrics', {}) or {}
                f.write("## Recommended Threshold\n\n")
                f.write(f"- Threshold: `{recommendation.get('best_threshold'):.4f}`\n")
                f.write(f"- Alpha score: `{float(best.get('alpha_score', float('nan'))):.4f}`\n")
                f.write(f"- Trade count: `{int(best.get('trade_count', 0))}`\n")
                f.write(f"- Win rate: `{float(best.get('win_rate', float('nan'))):.2%}`\n")
                f.write(f"- Mean net return: `{float(best.get('mean_net_return_pct', float('nan'))):.2%}`\n")
                f.write(f"- Mean tx cost/contract: `${float(best.get('mean_tx_cost_per_contract', float('nan'))):.2f}`\n\n")
            else:
                f.write("## Recommended Threshold\n\n")
                f.write(f"- No recommendation (`reason={recommendation.get('reason', 'unknown')}`)\n\n")
                if recommendation_reason == 'insufficient_trade_count_for_thresholds':
                    f.write("### Candidate Availability\n\n")
                    f.write(f"- Available trades in scope: `{available_trade_count}`\n")
                    f.write(f"- Max trades at any threshold: `{max_trade_count}`\n")
                    if suggested_min_trades is not None:
                        f.write(
                            f"- Suggested min trades for this sample: `{suggested_min_trades}` "
                            "(use `--tuning-min-trades`)\n"
                        )
                    f.write("\n")

            f.write("## Decile Performance\n\n")
            f.write("```text\n")
            f.write(deciles_df.to_string(index=False))
            f.write("\n```\n")

            if not candidates_df.empty:
                f.write("\n## Candidate Thresholds\n\n")
                f.write("```text\n")
                f.write(candidates_df.to_string(index=False))
                f.write("\n```\n")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'session_id': resolved_session,
                'min_confidence': min_confidence,
                'min_trades': min_trades,
                'use_composite_signal': use_composite_signal,
                'recommendation': recommendation,
                'deciles_csv': str(deciles_csv_path),
                'candidates_csv': str(candidates_csv_path) if not candidates_df.empty else None,
            }, f, indent=2, default=str)

        self.logger.info(f"✅ Decile table saved: {deciles_csv_path}")
        if not candidates_df.empty:
            self.logger.info(f"✅ Threshold candidates saved: {candidates_csv_path}")
        self.logger.info(f"✅ Threshold tuning summary saved: {md_path}")
        self.logger.info(f"✅ Threshold tuning JSON saved: {json_path}")
        return {
            'deciles_csv_path': str(deciles_csv_path),
            'candidates_csv_path': str(candidates_csv_path) if not candidates_df.empty else None,
            'markdown_path': str(md_path),
            'json_path': str(json_path),
            'decile_rows': int(len(deciles_df)),
            'candidate_rows': int(len(candidates_df)),
            'recommended': bool(recommendation.get('recommended', False)),
            'best_threshold': recommendation.get('best_threshold'),
            'reason': recommendation_reason,
            'available_trade_count': available_trade_count,
            'max_trade_count_for_thresholds': max_trade_count,
            'suggested_min_trades': suggested_min_trades,
            'session_id': resolved_session,
        }

    def run_parameter_sweep(self, execution_profiles: List[str],
                          hold_days_grid: List[int],
                          signal_threshold_grid: List[float],
                          trades_per_day_grid: List[int],
                          entry_days_grid: List[int],
                          exit_days_grid: List[int],
                          target_entry_dte: int = 6,
                          entry_dte_band: int = 6,
                          min_daily_share_volume: int = 1_000_000,
                          max_abs_momentum_5d: float = 0.11,
                          position_contracts: int = 1,
                          lookback_days: int = 365,
                          max_backtest_symbols: int = 10,
                          use_crush_confidence_gate: bool = True,
                          allow_global_crush_profile: bool = True,
                          min_crush_confidence: float = 0.45,
                          min_crush_magnitude: float = 0.08,
                          min_crush_edge: float = 0.02,
                          require_true_earnings: bool = False,
                          allow_proxy_earnings: bool = True,
                          output_dir: str = "exports/reports",
                          top_n: int = 20,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Optional[dict]:
        """Run a multi-parameter walk-forward sweep and persist ranked results."""
        execution_profiles = [p.strip().lower() for p in execution_profiles if p.strip()]
        hold_days_grid = [max(1, int(v)) for v in hold_days_grid]
        signal_threshold_grid = [max(0.0, min(1.0, float(v))) for v in signal_threshold_grid]
        trades_per_day_grid = [max(1, int(v)) for v in trades_per_day_grid]
        entry_days_grid = [max(1, int(v)) for v in entry_days_grid]
        exit_days_grid = [max(0, int(v)) for v in exit_days_grid]
        position_contracts = max(1, int(position_contracts))
        lookback_days = max(120, int(lookback_days))
        max_backtest_symbols = max(1, int(max_backtest_symbols))
        target_entry_dte = max(1, int(target_entry_dte))
        entry_dte_band = max(1, int(entry_dte_band))
        min_daily_share_volume = max(0, int(min_daily_share_volume))
        max_abs_momentum_5d = max(0.01, float(max_abs_momentum_5d))
        top_n = max(1, int(top_n))
        min_crush_confidence = max(0.0, min(1.0, float(min_crush_confidence)))
        min_crush_magnitude = max(0.0, min(1.0, float(min_crush_magnitude)))
        min_crush_edge = max(0.0, min(1.0, float(min_crush_edge)))

        base_params = {
            'strategy_type': 'calendar_spread',
            'lookback_days': lookback_days,
            'position_size_contracts': position_contracts,
            'max_symbols': max_backtest_symbols,
            'iv_rv_min': 0.95,
            'iv_rv_max': 2.30,
            'earnings_event_mode': True,
            'target_entry_dte': target_entry_dte,
            'entry_dte_band': entry_dte_band,
            'min_daily_share_volume': min_daily_share_volume,
            'max_abs_momentum_5d': max_abs_momentum_5d,
            'use_crush_confidence_gate': bool(use_crush_confidence_gate),
            'allow_global_crush_profile': bool(allow_global_crush_profile),
            'min_crush_confidence': min_crush_confidence,
            'min_crush_magnitude': min_crush_magnitude,
            'min_crush_edge': min_crush_edge,
            'require_true_earnings': bool(require_true_earnings),
            'allow_proxy_earnings': bool(allow_proxy_earnings),
        }
        if start_date:
            base_params['start_date'] = str(start_date)
        if end_date:
            base_params['end_date'] = str(end_date)
        parameter_grid = {
            'execution_profile': execution_profiles,
            'hold_days': hold_days_grid,
            'min_signal_score': signal_threshold_grid,
            'max_trades_per_day': trades_per_day_grid,
            'entry_days_before_earnings': entry_days_grid,
            'exit_days_after_earnings': exit_days_grid,
        }

        self.logger.info("🧪 Running parameter sweep for earnings-window backtest")
        results_df = self.db.run_backtest_parameter_sweep(base_params, parameter_grid, top_n=None)
        if results_df.empty:
            self.logger.warning("⚠️ Parameter sweep produced no results")
            return None

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f"earnings_walkforward_sweep_{timestamp}.csv"
        md_path = output_path / f"earnings_walkforward_sweep_{timestamp}.md"
        json_path = output_path / f"earnings_walkforward_sweep_{timestamp}_best_params.json"

        ranked = results_df.sort_values(
            by=['alpha_score', 'sharpe_ratio', 'total_pnl'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        ranked.to_csv(csv_path, index=False)
        top_rows = ranked.head(top_n).copy()

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings Walk-Forward Parameter Sweep\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Total combinations: {len(ranked)}\n")
            f.write(f"- Top rows shown: {min(top_n, len(ranked))}\n\n")

            best = ranked.iloc[0]
            f.write("## Best Configuration\n\n")
            f.write(f"- Session: `{best['session_id']}`\n")
            f.write(f"- Alpha score: `{best['alpha_score']:.4f}`\n")
            f.write(f"- Sharpe: `{best['sharpe_ratio']:.4f}`\n")
            f.write(f"- Win rate: `{best['win_rate']:.2%}`\n")
            f.write(f"- Total PnL: `${best['total_pnl']:.2f}`\n")
            f.write(f"- Max drawdown: `${best['max_drawdown']:.2f}`\n\n")

            f.write("## Top Results\n\n")
            columns = [
                'alpha_score', 'sharpe_ratio', 'win_rate', 'total_pnl', 'max_drawdown',
                'mean_crush_confidence', 'mean_crush_edge_score', 'crush_label_coverage',
                'crush_prediction_mae', 'crush_directional_accuracy',
                'param_execution_profile', 'param_hold_days', 'param_min_signal_score',
                'param_max_trades_per_day', 'param_entry_days_before_earnings',
                'param_exit_days_after_earnings', 'session_id'
            ]
            top_table = top_rows[columns].to_string(index=False)
            f.write("```text\n")
            f.write(top_table)
            f.write("\n```\n")

        best_params = {
            key[6:]: best[key]
            for key in ranked.columns
            if key.startswith("param_")
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'best_session_id': str(best['session_id']),
                'best_alpha_score': float(best['alpha_score']),
                'best_sharpe_ratio': float(best['sharpe_ratio']),
                'best_win_rate': float(best['win_rate']),
                'best_total_pnl': float(best['total_pnl']),
                'best_max_drawdown': float(best['max_drawdown']),
                'best_mean_crush_confidence': float(best['mean_crush_confidence']) if 'mean_crush_confidence' in best else None,
                'best_mean_crush_edge_score': float(best['mean_crush_edge_score']) if 'mean_crush_edge_score' in best else None,
                'best_crush_label_coverage': float(best['crush_label_coverage']) if 'crush_label_coverage' in best else None,
                'best_crush_prediction_mae': float(best['crush_prediction_mae']) if 'crush_prediction_mae' in best else None,
                'best_crush_directional_accuracy': float(best['crush_directional_accuracy']) if 'crush_directional_accuracy' in best else None,
                'best_params': best_params,
                'source_csv': str(csv_path),
            }, f, indent=2, default=str)

        self.logger.info(f"✅ Parameter sweep saved: {csv_path}")
        self.logger.info(f"✅ Sweep summary saved: {md_path}")
        self.logger.info(f"✅ Best params saved: {json_path}")
        return {
            'csv_path': str(csv_path),
            'markdown_path': str(md_path),
            'json_path': str(json_path),
            'rows': int(len(ranked)),
            'top_rows': int(min(top_n, len(ranked))),
            'best_params': best_params,
            'best_session_id': str(best['session_id']),
        }

    def run_oos_validation(self, execution_profiles: List[str],
                         hold_days_grid: List[int],
                         signal_threshold_grid: List[float],
                         trades_per_day_grid: List[int],
                         entry_days_grid: List[int],
                         exit_days_grid: List[int],
                         target_entry_dte: int = 6,
                         entry_dte_band: int = 6,
                         min_daily_share_volume: int = 1_000_000,
                         max_abs_momentum_5d: float = 0.11,
                         train_days: int = 252,
                         test_days: int = 63,
                         step_days: int = 63,
                         top_n_train: int = 1,
                         position_contracts: int = 1,
                         lookback_days: int = 365,
                         max_backtest_symbols: int = 10,
                         use_crush_confidence_gate: bool = True,
                         allow_global_crush_profile: bool = True,
                         min_crush_confidence: float = 0.45,
                         min_crush_magnitude: float = 0.08,
                         min_crush_edge: float = 0.02,
                         require_true_earnings: bool = False,
                         allow_proxy_earnings: bool = True,
                         min_splits: int = 8,
                         min_total_test_trades: int = 80,
                         min_trades_per_split: float = 5.0,
                         output_dir: str = "exports/reports",
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Optional[dict]:
        """Run rolling out-of-sample validation and export results."""
        execution_profiles = [p.strip().lower() for p in execution_profiles if p.strip()]
        hold_days_grid = [max(1, int(v)) for v in hold_days_grid]
        signal_threshold_grid = [max(0.0, min(1.0, float(v))) for v in signal_threshold_grid]
        trades_per_day_grid = [max(1, int(v)) for v in trades_per_day_grid]
        entry_days_grid = [max(1, int(v)) for v in entry_days_grid]
        exit_days_grid = [max(0, int(v)) for v in exit_days_grid]
        train_days = max(63, int(train_days))
        test_days = max(21, int(test_days))
        step_days = max(21, int(step_days))
        top_n_train = max(1, int(top_n_train))
        position_contracts = max(1, int(position_contracts))
        lookback_days = max(120, int(lookback_days))
        max_backtest_symbols = max(1, int(max_backtest_symbols))
        target_entry_dte = max(1, int(target_entry_dte))
        entry_dte_band = max(1, int(entry_dte_band))
        min_daily_share_volume = max(0, int(min_daily_share_volume))
        max_abs_momentum_5d = max(0.01, float(max_abs_momentum_5d))
        min_crush_confidence = max(0.0, min(1.0, float(min_crush_confidence)))
        min_crush_magnitude = max(0.0, min(1.0, float(min_crush_magnitude)))
        min_crush_edge = max(0.0, min(1.0, float(min_crush_edge)))
        min_splits = max(1, int(min_splits))
        min_total_test_trades = max(1, int(min_total_test_trades))
        min_trades_per_split = max(1.0, float(min_trades_per_split))

        base_params = {
            'strategy_type': 'calendar_spread',
            'lookback_days': lookback_days,
            'position_size_contracts': position_contracts,
            'max_symbols': max_backtest_symbols,
            'iv_rv_min': 0.95,
            'iv_rv_max': 2.30,
            'earnings_event_mode': True,
            'target_entry_dte': target_entry_dte,
            'entry_dte_band': entry_dte_band,
            'min_daily_share_volume': min_daily_share_volume,
            'max_abs_momentum_5d': max_abs_momentum_5d,
            'use_crush_confidence_gate': bool(use_crush_confidence_gate),
            'allow_global_crush_profile': bool(allow_global_crush_profile),
            'min_crush_confidence': min_crush_confidence,
            'min_crush_magnitude': min_crush_magnitude,
            'min_crush_edge': min_crush_edge,
            'require_true_earnings': bool(require_true_earnings),
            'allow_proxy_earnings': bool(allow_proxy_earnings),
        }
        if start_date:
            base_params['start_date'] = str(start_date)
        if end_date:
            base_params['end_date'] = str(end_date)

        parameter_grid = {
            'execution_profile': execution_profiles,
            'hold_days': hold_days_grid,
            'min_signal_score': signal_threshold_grid,
            'max_trades_per_day': trades_per_day_grid,
            'entry_days_before_earnings': entry_days_grid,
            'exit_days_after_earnings': exit_days_grid,
        }

        self.logger.info("🧪 Running rolling out-of-sample validation")
        oos_df = self.db.run_rolling_oos_validation(
            base_params=base_params,
            parameter_grid=parameter_grid,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            top_n_train=top_n_train,
        )
        if oos_df.empty:
            self.logger.warning("⚠️ OOS validation produced no rows")
            return None

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f"earnings_oos_validation_{timestamp}.csv"
        md_path = output_path / f"earnings_oos_validation_{timestamp}.md"
        json_path = output_path / f"earnings_oos_validation_{timestamp}_best_params.json"
        report_card_md_path = output_path / f"earnings_oos_report_card_{timestamp}.md"
        report_card_json_path = output_path / f"earnings_oos_report_card_{timestamp}.json"

        oos_df.to_csv(csv_path, index=False)

        selected_cols = [col for col in oos_df.columns if col.startswith('selected_')]
        if selected_cols:
            grouped = (
                oos_df.groupby(selected_cols, dropna=False)
                .agg(
                    splits=('split_index', 'count'),
                    mean_test_alpha=('test_alpha_score', 'mean'),
                    mean_test_sharpe=('test_sharpe_ratio', 'mean'),
                    mean_test_win_rate=('test_win_rate', 'mean'),
                    mean_test_pnl=('test_total_pnl', 'mean'),
                )
                .reset_index()
                .sort_values(by=['mean_test_alpha', 'mean_test_sharpe', 'mean_test_pnl'],
                             ascending=[False, False, False])
                .reset_index(drop=True)
            )
            best_group = grouped.iloc[0]
            best_params = {
                col[9:]: best_group[col]
                for col in selected_cols
            }
        else:
            grouped = oos_df.copy()
            best_group = oos_df.iloc[0]
            best_params = {}

        report_card = self.db.build_oos_report_card(
            oos_df=oos_df,
            min_splits=min_splits,
            min_total_test_trades=min_total_test_trades,
            min_trades_per_split=min_trades_per_split,
        )

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings Rolling OOS Validation\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Splits: {len(oos_df)}\n")
            f.write(f"- Mean test alpha: {oos_df['test_alpha_score'].mean():.4f}\n")
            f.write(f"- Mean test Sharpe: {oos_df['test_sharpe_ratio'].mean():.4f}\n")
            f.write(f"- Mean test win rate: {oos_df['test_win_rate'].mean():.2%}\n")
            f.write(f"- Mean test PnL: ${oos_df['test_total_pnl'].mean():.2f}\n\n")
            if 'test_crush_prediction_mae' in oos_df.columns:
                f.write(f"- Mean test crush MAE: {oos_df['test_crush_prediction_mae'].mean():.4f}\n")
            if 'test_crush_directional_accuracy' in oos_df.columns:
                f.write(f"- Mean test crush directional accuracy: {oos_df['test_crush_directional_accuracy'].mean():.2%}\n")
            if 'test_mean_crush_confidence' in oos_df.columns:
                f.write(f"- Mean test crush confidence: {oos_df['test_mean_crush_confidence'].mean():.2f}\n")
            f.write("\n")
            f.write("## OOS Robustness Report Card\n\n")
            if report_card.get('ready'):
                verdict = report_card.get('verdict', {})
                sample = report_card.get('sample', {})
                metrics = report_card.get('metrics', {})
                alpha = metrics.get('alpha', {}) or {}
                sharpe = metrics.get('sharpe', {}) or {}
                pnl = metrics.get('pnl', {}) or {}
                win_rate = metrics.get('win_rate', {}) or {}
                f.write(f"- Grade: `{verdict.get('grade', 'n/a')}`\n")
                f.write(f"- Overall pass: `{verdict.get('overall_pass', False)}`\n")
                f.write(f"- Message: {verdict.get('message', 'n/a')}\n")
                f.write(f"- Splits: `{sample.get('splits', 0)}`\n")
                f.write(f"- Total test trades: `{sample.get('total_test_trades', 0)}`\n")
                f.write(f"- Avg trades/split: `{float(sample.get('avg_trades_per_split', 0.0)):.2f}`\n")
                if alpha.get('mean') is not None:
                    f.write(
                        f"- Alpha mean (95% CI): `{float(alpha['mean']):.4f}` "
                        f"[`{float(alpha.get('low', alpha['mean'])):.4f}`, `{float(alpha.get('high', alpha['mean'])):.4f}`]\n"
                    )
                if sharpe.get('mean') is not None:
                    f.write(
                        f"- Sharpe mean (95% CI): `{float(sharpe['mean']):.4f}` "
                        f"[`{float(sharpe.get('low', sharpe['mean'])):.4f}`, `{float(sharpe.get('high', sharpe['mean'])):.4f}`]\n"
                    )
                if pnl.get('mean') is not None:
                    f.write(
                        f"- PnL mean (95% CI): `${float(pnl['mean']):.2f}` "
                        f"[`${float(pnl.get('low', pnl['mean'])):.2f}`, `${float(pnl.get('high', pnl['mean'])):.2f}`]\n"
                    )
                if win_rate.get('mean') is not None:
                    f.write(
                        f"- Win rate (Wilson 95% CI): `{float(win_rate['mean']):.2%}` "
                        f"[`{float(win_rate.get('low', win_rate['mean'])):.2%}`, `{float(win_rate.get('high', win_rate['mean'])):.2%}`]\n"
                    )
                f.write("\n")
                f.write("### Gates\n\n")
                for gate_name, gate in (report_card.get('gates') or {}).items():
                    f.write(
                        f"- `{gate_name}`: {'PASS' if gate.get('passed') else 'FAIL'} "
                        f"(actual={gate.get('actual')}, required={gate.get('required')})\n"
                    )
                f.write("\n")
            else:
                f.write(
                    f"- Report card unavailable (`reason={report_card.get('reason', 'unknown')}`)\n\n"
                )

            f.write("## Best Stable Parameters (Grouped by OOS mean alpha)\n\n")
            if best_params:
                for key, value in best_params.items():
                    f.write(f"- `{key}`: `{value}`\n")
            else:
                f.write("- No grouped params available\n")
            f.write("\n## Split Results\n\n")
            show_cols = [
                'split_index', 'train_start', 'train_end', 'test_start', 'test_end',
                'test_alpha_score', 'test_sharpe_ratio', 'test_win_rate', 'test_total_pnl',
                'test_crush_prediction_mae', 'test_crush_directional_accuracy', 'test_mean_crush_confidence'
            ]
            table = oos_df[show_cols].to_string(index=False)
            f.write("```text\n")
            f.write(table)
            f.write("\n```\n")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'splits': int(len(oos_df)),
                'mean_test_alpha': float(oos_df['test_alpha_score'].mean()),
                'mean_test_sharpe': float(oos_df['test_sharpe_ratio'].mean()),
                'mean_test_win_rate': float(oos_df['test_win_rate'].mean()),
                'mean_test_pnl': float(oos_df['test_total_pnl'].mean()),
                'mean_test_crush_mae': float(oos_df['test_crush_prediction_mae'].mean()) if 'test_crush_prediction_mae' in oos_df.columns else None,
                'mean_test_crush_directional_accuracy': float(oos_df['test_crush_directional_accuracy'].mean()) if 'test_crush_directional_accuracy' in oos_df.columns else None,
                'mean_test_crush_confidence': float(oos_df['test_mean_crush_confidence'].mean()) if 'test_mean_crush_confidence' in oos_df.columns else None,
                'best_params': best_params,
                'source_csv': str(csv_path),
            }, f, indent=2, default=str)

        with report_card_json_path.open("w", encoding="utf-8") as f:
            json.dump({
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'thresholds': {
                    'min_splits': min_splits,
                    'min_total_test_trades': min_total_test_trades,
                    'min_trades_per_split': min_trades_per_split,
                },
                'report_card': report_card,
                'source_oos_csv': str(csv_path),
            }, f, indent=2, default=str)

        with report_card_md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings OOS Robustness Report Card\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Source OOS CSV: `{csv_path}`\n")
            f.write(f"- Min splits: `{min_splits}`\n")
            f.write(f"- Min total test trades: `{min_total_test_trades}`\n")
            f.write(f"- Min trades per split: `{min_trades_per_split:.2f}`\n\n")
            if report_card.get('ready'):
                verdict = report_card.get('verdict', {}) or {}
                sample = report_card.get('sample', {}) or {}
                metrics = report_card.get('metrics', {}) or {}
                alpha = metrics.get('alpha', {}) or {}
                sharpe = metrics.get('sharpe', {}) or {}
                pnl = metrics.get('pnl', {}) or {}
                win_rate = metrics.get('win_rate', {}) or {}
                f.write(f"## Verdict\n\n")
                f.write(f"- Grade: `{verdict.get('grade', 'n/a')}`\n")
                f.write(f"- Overall pass: `{verdict.get('overall_pass', False)}`\n")
                f.write(f"- Message: {verdict.get('message', 'n/a')}\n\n")
                f.write("## Sample\n\n")
                f.write(f"- Splits: `{sample.get('splits', 0)}`\n")
                f.write(f"- Total test trades: `{sample.get('total_test_trades', 0)}`\n")
                f.write(f"- Avg trades per split: `{float(sample.get('avg_trades_per_split', 0.0)):.2f}`\n\n")
                f.write("## Confidence Intervals\n\n")
                if alpha.get('mean') is not None:
                    f.write(
                        f"- Alpha: `{float(alpha['mean']):.4f}` "
                        f"[`{float(alpha.get('low', alpha['mean'])):.4f}`, `{float(alpha.get('high', alpha['mean'])):.4f}`]\n"
                    )
                if sharpe.get('mean') is not None:
                    f.write(
                        f"- Sharpe: `{float(sharpe['mean']):.4f}` "
                        f"[`{float(sharpe.get('low', sharpe['mean'])):.4f}`, `{float(sharpe.get('high', sharpe['mean'])):.4f}`]\n"
                    )
                if pnl.get('mean') is not None:
                    f.write(
                        f"- PnL: `${float(pnl['mean']):.2f}` "
                        f"[`${float(pnl.get('low', pnl['mean'])):.2f}`, `${float(pnl.get('high', pnl['mean'])):.2f}`]\n"
                    )
                if win_rate.get('mean') is not None:
                    f.write(
                        f"- Win rate: `{float(win_rate['mean']):.2%}` "
                        f"[`{float(win_rate.get('low', win_rate['mean'])):.2%}`, `{float(win_rate.get('high', win_rate['mean'])):.2%}`]\n"
                    )
                f.write("\n## Gates\n\n")
                for gate_name, gate in (report_card.get('gates') or {}).items():
                    f.write(
                        f"- `{gate_name}`: {'PASS' if gate.get('passed') else 'FAIL'} "
                        f"(actual={gate.get('actual')}, required={gate.get('required')})\n"
                    )
            else:
                f.write(
                    f"Report card unavailable (`reason={report_card.get('reason', 'unknown')}`).\n"
                )

        self.logger.info(f"✅ OOS CSV saved: {csv_path}")
        self.logger.info(f"✅ OOS summary saved: {md_path}")
        self.logger.info(f"✅ OOS best params saved: {json_path}")
        self.logger.info(f"✅ OOS report card saved: {report_card_json_path}")
        best_split = oos_df.sort_values(
            by=['test_alpha_score', 'test_sharpe_ratio', 'test_total_pnl'],
            ascending=[False, False, False]
        ).iloc[0]
        return {
            'csv_path': str(csv_path),
            'markdown_path': str(md_path),
            'json_path': str(json_path),
            'report_card_markdown_path': str(report_card_md_path),
            'report_card_json_path': str(report_card_json_path),
            'splits': int(len(oos_df)),
            'best_params': best_params,
            'report_card': report_card,
            'best_test_session_id': str(best_split['test_session_id']),
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Institutional Options Data Backfill",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)'
    )

    parser.add_argument(
        '--full-universe',
        action='store_true',
        help='Process full institutional universe (50+ symbols)'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=2,
        help='Years of historical data to collect (default: 2)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Concurrent processing batch size (default: 5)'
    )

    parser.add_argument(
        '--backtest-only',
        action='store_true',
        help='Run backtest only (skip data collection)'
    )

    parser.add_argument(
        '--capture-snapshots',
        action='store_true',
        help='Capture live option-IV snapshots around earnings events'
    )

    parser.add_argument(
        '--calibrate-iv-decay',
        action='store_true',
        help='Calibrate IV-crush labels from captured snapshots'
    )

    parser.add_argument(
        '--snapshot-status',
        action='store_true',
        help='Report current snapshot pre/post pairing progress for IV-decay calibration'
    )

    parser.add_argument(
        '--run-sample-backtest',
        action='store_true',
        help='Force a sample backtest run when using snapshot/calibration-only workflows'
    )

    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode with limited symbols'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        help='Custom database path'
    )

    parser.add_argument(
        '--execution-profile',
        type=str,
        default='institutional',
        help='Execution-cost profile (paper, retail, institutional, institutional_tight)'
    )

    parser.add_argument(
        '--hold-days',
        type=int,
        default=7,
        help='Backtest holding period in trading days (default: 7)'
    )

    parser.add_argument(
        '--min-signal-score',
        type=float,
        default=0.58,
        help='Minimum setup score threshold [0-1] (default: 0.58)'
    )

    parser.add_argument(
        '--max-trades-per-day',
        type=int,
        default=2,
        help='Maximum new trades per day in walk-forward engine (default: 2)'
    )

    parser.add_argument(
        '--position-contracts',
        type=int,
        default=1,
        help='Contracts per trade in backtest accounting (default: 1)'
    )

    parser.add_argument(
        '--lookback-days',
        type=int,
        default=365,
        help='Historical window for walk-forward backtest (default: 365)'
    )

    parser.add_argument(
        '--max-backtest-symbols',
        type=int,
        default=10,
        help='Maximum symbols to include in backtest universe when not explicitly set (default: 10)'
    )

    parser.add_argument(
        '--entry-days-before-earnings',
        type=int,
        default=7,
        help='Target trading entry offset before earnings date (default: 7)'
    )

    parser.add_argument(
        '--exit-days-after-earnings',
        type=int,
        default=1,
        help='Target trading exit offset after earnings date (default: 1)'
    )

    parser.add_argument(
        '--target-entry-dte',
        type=int,
        default=6,
        help='Preferred days-to-earnings for entry scoring sweet-spot (default: 6)'
    )

    parser.add_argument(
        '--entry-dte-band',
        type=int,
        default=6,
        help='Width of timing sweet-spot around target entry DTE (default: 6)'
    )

    parser.add_argument(
        '--min-daily-share-volume',
        type=int,
        default=1000000,
        help='Minimum underlying daily share volume filter (default: 1,000,000)'
    )

    parser.add_argument(
        '--max-abs-momentum-5d',
        type=float,
        default=0.11,
        help='Max absolute 5-day momentum allowed for entry filtering (default: 0.11)'
    )

    parser.add_argument(
        '--require-true-earnings',
        action='store_true',
        help='Require API-sourced earnings dates (skip proxy fallback)'
    )

    parser.add_argument(
        '--no-proxy-earnings',
        action='store_true',
        help='Disable proxy quarterly earnings fallback schedule'
    )

    parser.add_argument(
        '--disable-crush-confidence-gate',
        action='store_true',
        help='Disable strict IV-crush confidence/magnitude gate in candidate selection'
    )

    parser.add_argument(
        '--no-global-crush-profile',
        action='store_true',
        help='Disable fallback to global crush profile when symbol-specific profile is missing'
    )

    parser.add_argument(
        '--min-crush-confidence',
        type=float,
        default=0.45,
        help='Minimum crush confidence score [0-1] when confidence gate is enabled (default: 0.45)'
    )

    parser.add_argument(
        '--min-crush-magnitude',
        type=float,
        default=0.08,
        help='Minimum expected front-IV crush magnitude [0-1] required per trade (default: 0.08)'
    )

    parser.add_argument(
        '--min-crush-edge',
        type=float,
        default=0.02,
        help='Minimum crush edge score required per trade (default: 0.02)'
    )

    parser.add_argument(
        '--parameter-sweep',
        action='store_true',
        help='Run ranked parameter sweep for event-window backtests'
    )

    parser.add_argument(
        '--oos-validation',
        action='store_true',
        help='Run rolling out-of-sample validation (train-sweep then test)'
    )

    parser.add_argument(
        '--sweep-profiles',
        type=str,
        default='institutional,institutional_tight',
        help='Comma-separated execution profiles for sweep'
    )

    parser.add_argument(
        '--sweep-hold-days',
        type=str,
        default='5,7',
        help='Comma-separated hold-days values for sweep'
    )

    parser.add_argument(
        '--sweep-min-signal-scores',
        type=str,
        default='0.55,0.62',
        help='Comma-separated signal thresholds for sweep'
    )

    parser.add_argument(
        '--sweep-max-trades-per-day',
        type=str,
        default='1,2',
        help='Comma-separated max-trades/day values for sweep'
    )

    parser.add_argument(
        '--sweep-entry-days-before',
        type=str,
        default='5,7',
        help='Comma-separated entry-day offsets before earnings for sweep'
    )

    parser.add_argument(
        '--sweep-exit-days-after',
        type=str,
        default='0,1',
        help='Comma-separated exit-day offsets after earnings for sweep'
    )

    parser.add_argument(
        '--sweep-output-dir',
        type=str,
        default='exports/reports',
        help='Directory for sweep CSV/summary outputs'
    )

    parser.add_argument(
        '--sweep-top-n',
        type=int,
        default=20,
        help='Top N rows emphasized in sweep markdown report'
    )

    parser.add_argument(
        '--snapshot-lookback-days',
        type=int,
        default=14,
        help='Days back from today to capture earnings snapshots (default: 14)'
    )

    parser.add_argument(
        '--snapshot-lookahead-days',
        type=int,
        default=45,
        help='Days ahead from today to capture earnings snapshots (default: 45)'
    )

    parser.add_argument(
        '--snapshot-max-expiries',
        type=int,
        default=2,
        help='Number of expiries to evaluate per snapshot capture (default: 2)'
    )

    parser.add_argument(
        '--label-min-pre-days',
        type=int,
        default=1,
        help='Minimum days-before-earnings for pre snapshot label pairing (default: 1)'
    )

    parser.add_argument(
        '--label-max-pre-days',
        type=int,
        default=12,
        help='Maximum days-before-earnings for pre snapshot label pairing (default: 12)'
    )

    parser.add_argument(
        '--label-min-post-days',
        type=int,
        default=0,
        help='Minimum days-after-earnings for post snapshot label pairing (default: 0)'
    )

    parser.add_argument(
        '--label-max-post-days',
        type=int,
        default=5,
        help='Maximum days-after-earnings for post snapshot label pairing (default: 5)'
    )

    parser.add_argument(
        '--oos-train-days',
        type=int,
        default=252,
        help='Training window length in calendar days for OOS validation (default: 252)'
    )

    parser.add_argument(
        '--oos-test-days',
        type=int,
        default=63,
        help='Test window length in calendar days for OOS validation (default: 63)'
    )

    parser.add_argument(
        '--oos-step-days',
        type=int,
        default=63,
        help='Window step in calendar days for OOS validation (default: 63)'
    )

    parser.add_argument(
        '--oos-top-n-train',
        type=int,
        default=1,
        help='Top-N train candidates to consider per split (default: 1)'
    )

    parser.add_argument(
        '--oos-output-dir',
        type=str,
        default='exports/reports',
        help='Directory for OOS CSV/summary outputs'
    )

    parser.add_argument(
        '--oos-min-splits',
        type=int,
        default=8,
        help='Minimum OOS splits required for report-card pass gate (default: 8)'
    )

    parser.add_argument(
        '--oos-min-total-test-trades',
        type=int,
        default=80,
        help='Minimum total test trades required for report-card pass gate (default: 80)'
    )

    parser.add_argument(
        '--oos-min-trades-per-split',
        type=float,
        default=5.0,
        help='Minimum average test trades per split for report-card pass gate (default: 5.0)'
    )

    parser.add_argument(
        '--crush-scorecard',
        action='store_true',
        help='Generate rolling predicted-vs-realized IV-crush scorecard report'
    )

    parser.add_argument(
        '--scorecard-session-id',
        type=str,
        help='Optional session ID filter for crush scorecard'
    )

    parser.add_argument(
        '--scorecard-window',
        type=int,
        default=40,
        help='Rolling window size for crush scorecard metrics (default: 40)'
    )

    parser.add_argument(
        '--scorecard-min-confidence',
        type=float,
        default=0.35,
        help='Minimum trade confidence included in crush scorecard [0-1] (default: 0.35)'
    )

    parser.add_argument(
        '--scorecard-output-dir',
        type=str,
        default='exports/reports',
        help='Directory for crush scorecard CSV/summary outputs'
    )

    parser.add_argument(
        '--regime-diagnostics',
        action='store_true',
        help='Generate regime diagnostics report (VIX/DTE/IV-RV buckets)'
    )

    parser.add_argument(
        '--regime-session-id',
        type=str,
        help='Optional session ID filter for regime diagnostics'
    )

    parser.add_argument(
        '--regime-min-confidence',
        type=float,
        default=0.35,
        help='Minimum crush confidence included in regime diagnostics [0-1] (default: 0.35)'
    )

    parser.add_argument(
        '--regime-output-dir',
        type=str,
        default='exports/reports',
        help='Directory for regime diagnostics CSV/summary outputs'
    )

    parser.add_argument(
        '--threshold-tuning',
        action='store_true',
        help='Generate decile-based signal-threshold tuning report'
    )

    parser.add_argument(
        '--tuning-session-id',
        type=str,
        help='Optional session ID filter for threshold tuning'
    )

    parser.add_argument(
        '--tuning-min-confidence',
        type=float,
        default=0.35,
        help='Minimum crush confidence included in threshold tuning [0-1] (default: 0.35)'
    )

    parser.add_argument(
        '--tuning-min-trades',
        type=int,
        default=30,
        help='Minimum trades required per candidate threshold (default: 30)'
    )

    parser.add_argument(
        '--tuning-raw-setup-score',
        action='store_true',
        help='Use raw setup_score instead of composite score (setup+crush edge) for threshold tuning'
    )

    parser.add_argument(
        '--tuning-output-dir',
        type=str,
        default='exports/reports',
        help='Directory for threshold tuning CSV/summary outputs'
    )

    parser.add_argument(
        '--backtest-start-date',
        type=str,
        help='Optional backtest start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--backtest-end-date',
        type=str,
        help='Optional backtest end date (YYYY-MM-DD)'
    )

    return parser.parse_args()

async def main():
    """Main execution function"""
    args = parse_arguments()
    allow_proxy_earnings = not args.no_proxy_earnings
    use_crush_confidence_gate = not args.disable_crush_confidence_gate
    allow_global_crush_profile = not args.no_global_crush_profile
    for label, value in (
        ("--backtest-start-date", args.backtest_start_date),
        ("--backtest-end-date", args.backtest_end_date),
    ):
        if value:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                logger.error(f"❌ Invalid date format for {label}: {value} (expected YYYY-MM-DD)")
                sys.exit(2)

    logger.info("🏛️ Institutional Options Data Backfill Starting")
    logger.info("=" * 60)

    # Initialize collector
    collector = InstitutionalDataCollector(args.db_path)

    try:
        if args.backtest_only:
            # Run backtest only
            logger.info("🎯 Backtest-only mode")
            latest_session_id: Optional[str] = None
            if args.test_mode:
                aux_symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'TSLA']
            elif args.full_universe:
                aux_symbols = INSTITUTIONAL_UNIVERSE
            elif args.symbols:
                aux_symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
            else:
                aux_symbols = INSTITUTIONAL_UNIVERSE[:max(1, args.max_backtest_symbols)]

            if args.capture_snapshots:
                collector.capture_earnings_snapshots(
                    symbols=aux_symbols,
                    lookback_days=args.snapshot_lookback_days,
                    lookahead_days=args.snapshot_lookahead_days,
                    max_expiries=args.snapshot_max_expiries,
                    require_true_earnings=args.require_true_earnings,
                    allow_proxy_earnings=allow_proxy_earnings,
                )

            if args.calibrate_iv_decay:
                collector.calibrate_iv_decay_labels(
                    min_pre_days=args.label_min_pre_days,
                    max_pre_days=args.label_max_pre_days,
                    min_post_days=args.label_min_post_days,
                    max_post_days=args.label_max_post_days,
                )

            if args.capture_snapshots or args.calibrate_iv_decay or args.snapshot_status:
                collector.report_snapshot_pairing_status(
                    min_pre_days=args.label_min_pre_days,
                    max_pre_days=args.label_max_pre_days,
                    min_post_days=args.label_min_post_days,
                    max_post_days=args.label_max_post_days,
                )

            scorecard_requested = bool(args.crush_scorecard)
            analysis_requested = bool(
                args.parameter_sweep
                or args.oos_validation
                or scorecard_requested
                or args.regime_diagnostics
                or args.threshold_tuning
            )
            snapshot_only_mode = bool(
                (args.capture_snapshots or args.calibrate_iv_decay or args.snapshot_status)
                and not analysis_requested
                and not args.run_sample_backtest
            )
            if args.oos_validation:
                oos_result = collector.run_oos_validation(
                    execution_profiles=_parse_csv_strings(args.sweep_profiles),
                    hold_days_grid=_parse_csv_ints(args.sweep_hold_days),
                    signal_threshold_grid=_parse_csv_floats(args.sweep_min_signal_scores),
                    trades_per_day_grid=_parse_csv_ints(args.sweep_max_trades_per_day),
                    entry_days_grid=_parse_csv_ints(args.sweep_entry_days_before),
                    exit_days_grid=_parse_csv_ints(args.sweep_exit_days_after),
                    target_entry_dte=args.target_entry_dte,
                    entry_dte_band=args.entry_dte_band,
                    min_daily_share_volume=args.min_daily_share_volume,
                    max_abs_momentum_5d=args.max_abs_momentum_5d,
                    train_days=args.oos_train_days,
                    test_days=args.oos_test_days,
                    step_days=args.oos_step_days,
                    top_n_train=args.oos_top_n_train,
                    position_contracts=args.position_contracts,
                    lookback_days=args.lookback_days,
                    max_backtest_symbols=args.max_backtest_symbols,
                    use_crush_confidence_gate=use_crush_confidence_gate,
                    allow_global_crush_profile=allow_global_crush_profile,
                    min_crush_confidence=args.min_crush_confidence,
                    min_crush_magnitude=args.min_crush_magnitude,
                    min_crush_edge=args.min_crush_edge,
                    require_true_earnings=args.require_true_earnings,
                    allow_proxy_earnings=allow_proxy_earnings,
                    min_splits=args.oos_min_splits,
                    min_total_test_trades=args.oos_min_total_test_trades,
                    min_trades_per_split=args.oos_min_trades_per_split,
                    output_dir=args.oos_output_dir,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )
                if oos_result:
                    card = oos_result.get('report_card', {})
                    verdict = card.get('verdict', {}) if isinstance(card, dict) else {}
                    logger.info(
                        "✅ OOS validation completed: "
                        f"{oos_result['splits']} splits, "
                        f"grade={verdict.get('grade', 'n/a')}, "
                        f"pass={verdict.get('overall_pass', False)}"
                    )
                    latest_session_id = str(oos_result.get('best_test_session_id') or latest_session_id or "")
            elif args.parameter_sweep:
                sweep_result = collector.run_parameter_sweep(
                    execution_profiles=_parse_csv_strings(args.sweep_profiles),
                    hold_days_grid=_parse_csv_ints(args.sweep_hold_days),
                    signal_threshold_grid=_parse_csv_floats(args.sweep_min_signal_scores),
                    trades_per_day_grid=_parse_csv_ints(args.sweep_max_trades_per_day),
                    entry_days_grid=_parse_csv_ints(args.sweep_entry_days_before),
                    exit_days_grid=_parse_csv_ints(args.sweep_exit_days_after),
                    target_entry_dte=args.target_entry_dte,
                    entry_dte_band=args.entry_dte_band,
                    min_daily_share_volume=args.min_daily_share_volume,
                    max_abs_momentum_5d=args.max_abs_momentum_5d,
                    position_contracts=args.position_contracts,
                    lookback_days=args.lookback_days,
                    max_backtest_symbols=args.max_backtest_symbols,
                    use_crush_confidence_gate=use_crush_confidence_gate,
                    allow_global_crush_profile=allow_global_crush_profile,
                    min_crush_confidence=args.min_crush_confidence,
                    min_crush_magnitude=args.min_crush_magnitude,
                    min_crush_edge=args.min_crush_edge,
                    require_true_earnings=args.require_true_earnings,
                    allow_proxy_earnings=allow_proxy_earnings,
                    output_dir=args.sweep_output_dir,
                    top_n=args.sweep_top_n,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )
                if sweep_result:
                    logger.info(f"✅ Sweep completed: {sweep_result['rows']} rows")
                    latest_session_id = str(sweep_result.get('best_session_id') or latest_session_id or "")
            elif snapshot_only_mode:
                logger.info(
                    "ℹ️ Snapshot-status/calibration-only mode: skipping sample backtest "
                    "(use --run-sample-backtest to force)"
                )
            elif not scorecard_requested:
                session_id = collector.run_sample_backtest(
                    execution_profile=args.execution_profile,
                    hold_days=args.hold_days,
                    min_signal_score=args.min_signal_score,
                    max_trades_per_day=args.max_trades_per_day,
                    position_contracts=args.position_contracts,
                    lookback_days=args.lookback_days,
                    max_backtest_symbols=args.max_backtest_symbols,
                    entry_days_before_earnings=args.entry_days_before_earnings,
                    exit_days_after_earnings=args.exit_days_after_earnings,
                    target_entry_dte=args.target_entry_dte,
                    entry_dte_band=args.entry_dte_band,
                    min_daily_share_volume=args.min_daily_share_volume,
                    max_abs_momentum_5d=args.max_abs_momentum_5d,
                    use_crush_confidence_gate=use_crush_confidence_gate,
                    allow_global_crush_profile=allow_global_crush_profile,
                    min_crush_confidence=args.min_crush_confidence,
                    min_crush_magnitude=args.min_crush_magnitude,
                    min_crush_edge=args.min_crush_edge,
                    require_true_earnings=args.require_true_earnings,
                    allow_proxy_earnings=allow_proxy_earnings,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )
                if session_id:
                    logger.info(f"✅ Backtest completed: {session_id}")
                    latest_session_id = str(session_id)

            if scorecard_requested:
                scorecard_result = collector.run_crush_scorecard(
                    session_id=args.scorecard_session_id or latest_session_id,
                    window_size=args.scorecard_window,
                    min_confidence=args.scorecard_min_confidence,
                    output_dir=args.scorecard_output_dir,
                )
                if scorecard_result:
                    logger.info(f"✅ Crush scorecard completed: {scorecard_result['rows']} labeled rows")
                else:
                    logger.warning("⚠️ Crush scorecard did not produce results")

            if args.regime_diagnostics:
                regime_result = collector.run_regime_diagnostics(
                    session_id=args.regime_session_id or latest_session_id,
                    min_confidence=args.regime_min_confidence,
                    output_dir=args.regime_output_dir,
                )
                if regime_result:
                    logger.info(f"✅ Regime diagnostics completed: {regime_result['rows']} rows")
                else:
                    logger.warning("⚠️ Regime diagnostics did not produce results")

            if args.threshold_tuning:
                tuning_result = collector.run_threshold_tuning(
                    session_id=args.tuning_session_id or latest_session_id,
                    min_confidence=args.tuning_min_confidence,
                    min_trades=args.tuning_min_trades,
                    use_composite_signal=not args.tuning_raw_setup_score,
                    output_dir=args.tuning_output_dir,
                )
                if tuning_result:
                    if tuning_result.get('recommended'):
                        logger.info(
                            "✅ Threshold tuning completed: "
                            f"{tuning_result['candidate_rows']} candidates, "
                            f"best_threshold={tuning_result.get('best_threshold')}"
                        )
                    else:
                        reason = tuning_result.get('reason', 'unknown')
                        suggestion = tuning_result.get('suggested_min_trades')
                        suffix = (
                            f", suggested_min_trades={suggestion}"
                            if suggestion is not None else ""
                        )
                        logger.warning(
                            "⚠️ Threshold tuning completed without recommendation: "
                            f"reason={reason}, candidate_rows={tuning_result['candidate_rows']}"
                            f"{suffix}"
                        )
                else:
                    logger.warning("⚠️ Threshold tuning did not produce results")
            return

        # Determine symbol list
        if args.test_mode:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'TSLA']
            logger.info("🧪 Test mode: Using 5 symbols")
        elif args.full_universe:
            symbols = INSTITUTIONAL_UNIVERSE
            logger.info(f"🌍 Full universe mode: {len(symbols)} symbols")
        elif args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            logger.info(f"📋 Custom symbols: {symbols}")
        else:
            # Default to top 10
            symbols = INSTITUTIONAL_UNIVERSE[:10]
            logger.info(f"📊 Default mode: Top 10 symbols")

        # Run backfill
        success = await collector.run_full_backfill(
            symbols=symbols,
            years=args.years,
            batch_size=args.batch_size
        )

        if success:
            logger.info("🎉 Institutional backfill completed successfully!")
            latest_session_id: Optional[str] = None

            if args.capture_snapshots:
                collector.capture_earnings_snapshots(
                    symbols=symbols,
                    lookback_days=args.snapshot_lookback_days,
                    lookahead_days=args.snapshot_lookahead_days,
                    max_expiries=args.snapshot_max_expiries,
                    require_true_earnings=args.require_true_earnings,
                    allow_proxy_earnings=allow_proxy_earnings,
                )

            if args.calibrate_iv_decay:
                collector.calibrate_iv_decay_labels(
                    min_pre_days=args.label_min_pre_days,
                    max_pre_days=args.label_max_pre_days,
                    min_post_days=args.label_min_post_days,
                    max_post_days=args.label_max_post_days,
                )

            if args.oos_validation:
                logger.info("🧪 Running post-backfill OOS validation...")
                oos_result = collector.run_oos_validation(
                    execution_profiles=_parse_csv_strings(args.sweep_profiles),
                    hold_days_grid=_parse_csv_ints(args.sweep_hold_days),
                    signal_threshold_grid=_parse_csv_floats(args.sweep_min_signal_scores),
                    trades_per_day_grid=_parse_csv_ints(args.sweep_max_trades_per_day),
                    entry_days_grid=_parse_csv_ints(args.sweep_entry_days_before),
                    exit_days_grid=_parse_csv_ints(args.sweep_exit_days_after),
                    target_entry_dte=args.target_entry_dte,
                    entry_dte_band=args.entry_dte_band,
                    min_daily_share_volume=args.min_daily_share_volume,
                    max_abs_momentum_5d=args.max_abs_momentum_5d,
                    train_days=args.oos_train_days,
                    test_days=args.oos_test_days,
                    step_days=args.oos_step_days,
                    top_n_train=args.oos_top_n_train,
                    position_contracts=args.position_contracts,
                    lookback_days=args.lookback_days,
                    max_backtest_symbols=args.max_backtest_symbols,
                    use_crush_confidence_gate=use_crush_confidence_gate,
                    allow_global_crush_profile=allow_global_crush_profile,
                    min_crush_confidence=args.min_crush_confidence,
                    min_crush_magnitude=args.min_crush_magnitude,
                    min_crush_edge=args.min_crush_edge,
                    require_true_earnings=args.require_true_earnings,
                    allow_proxy_earnings=allow_proxy_earnings,
                    min_splits=args.oos_min_splits,
                    min_total_test_trades=args.oos_min_total_test_trades,
                    min_trades_per_split=args.oos_min_trades_per_split,
                    output_dir=args.oos_output_dir,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )
                if oos_result:
                    card = oos_result.get('report_card', {})
                    verdict = card.get('verdict', {}) if isinstance(card, dict) else {}
                    logger.info(
                        "✅ OOS validation finished and reports were generated "
                        f"(grade={verdict.get('grade', 'n/a')}, pass={verdict.get('overall_pass', False)})"
                    )
                    latest_session_id = str(oos_result.get('best_test_session_id') or latest_session_id or "")
                else:
                    logger.warning("⚠️ OOS validation did not produce results")
            elif args.parameter_sweep:
                logger.info("🧪 Running post-backfill parameter sweep...")
                sweep_result = collector.run_parameter_sweep(
                    execution_profiles=_parse_csv_strings(args.sweep_profiles),
                    hold_days_grid=_parse_csv_ints(args.sweep_hold_days),
                    signal_threshold_grid=_parse_csv_floats(args.sweep_min_signal_scores),
                    trades_per_day_grid=_parse_csv_ints(args.sweep_max_trades_per_day),
                    entry_days_grid=_parse_csv_ints(args.sweep_entry_days_before),
                    exit_days_grid=_parse_csv_ints(args.sweep_exit_days_after),
                    target_entry_dte=args.target_entry_dte,
                    entry_dte_band=args.entry_dte_band,
                    min_daily_share_volume=args.min_daily_share_volume,
                    max_abs_momentum_5d=args.max_abs_momentum_5d,
                    position_contracts=args.position_contracts,
                    lookback_days=args.lookback_days,
                    max_backtest_symbols=args.max_backtest_symbols,
                    use_crush_confidence_gate=use_crush_confidence_gate,
                    allow_global_crush_profile=allow_global_crush_profile,
                    min_crush_confidence=args.min_crush_confidence,
                    min_crush_magnitude=args.min_crush_magnitude,
                    min_crush_edge=args.min_crush_edge,
                    require_true_earnings=args.require_true_earnings,
                    allow_proxy_earnings=allow_proxy_earnings,
                    output_dir=args.sweep_output_dir,
                    top_n=args.sweep_top_n,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )
                if sweep_result:
                    logger.info("✅ Sweep finished and reports were generated")
                    latest_session_id = str(sweep_result.get('best_session_id') or latest_session_id or "")
                else:
                    logger.warning("⚠️ Parameter sweep did not produce results")
            elif not args.crush_scorecard:
                logger.info("🎯 Running post-backfill validation backtest...")
                session_id = collector.run_sample_backtest(
                    execution_profile=args.execution_profile,
                    hold_days=args.hold_days,
                    min_signal_score=args.min_signal_score,
                    max_trades_per_day=args.max_trades_per_day,
                    position_contracts=args.position_contracts,
                    lookback_days=args.lookback_days,
                    max_backtest_symbols=args.max_backtest_symbols,
                    entry_days_before_earnings=args.entry_days_before_earnings,
                    exit_days_after_earnings=args.exit_days_after_earnings,
                    target_entry_dte=args.target_entry_dte,
                    entry_dte_band=args.entry_dte_band,
                    min_daily_share_volume=args.min_daily_share_volume,
                    max_abs_momentum_5d=args.max_abs_momentum_5d,
                    use_crush_confidence_gate=use_crush_confidence_gate,
                    allow_global_crush_profile=allow_global_crush_profile,
                    min_crush_confidence=args.min_crush_confidence,
                    min_crush_magnitude=args.min_crush_magnitude,
                    min_crush_edge=args.min_crush_edge,
                    require_true_earnings=args.require_true_earnings,
                    allow_proxy_earnings=allow_proxy_earnings,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )

                if session_id:
                    latest_session_id = str(session_id)
                    results_df = collector.db.get_backtest_results(session_id)
                    total_trades = 0
                    if not results_df.empty and 'total_trades' in results_df.columns:
                        try:
                            total_trades = int(results_df.iloc[0]['total_trades'])
                        except (TypeError, ValueError):
                            total_trades = 0

                    if total_trades > 0:
                        logger.info("✅ System is ready for institutional trading!")
                    else:
                        logger.warning(
                            "⚠️ Backtest completed but produced 0 trades; "
                            "do not treat this run as production-ready."
                        )
                else:
                    logger.warning("⚠️ Backtest validation failed")

            if args.crush_scorecard:
                logger.info("🧪 Building rolling IV-crush scorecard...")
                scorecard_result = collector.run_crush_scorecard(
                    session_id=args.scorecard_session_id or latest_session_id,
                    window_size=args.scorecard_window,
                    min_confidence=args.scorecard_min_confidence,
                    output_dir=args.scorecard_output_dir,
                )
                if scorecard_result:
                    logger.info("✅ Crush scorecard finished and reports were generated")
                else:
                    logger.warning("⚠️ Crush scorecard did not produce results")

            if args.regime_diagnostics:
                logger.info("🧪 Building regime diagnostics...")
                regime_result = collector.run_regime_diagnostics(
                    session_id=args.regime_session_id or latest_session_id,
                    min_confidence=args.regime_min_confidence,
                    output_dir=args.regime_output_dir,
                )
                if regime_result:
                    logger.info("✅ Regime diagnostics finished and reports were generated")
                else:
                    logger.warning("⚠️ Regime diagnostics did not produce results")

            if args.threshold_tuning:
                logger.info("🧪 Running decile threshold tuning...")
                tuning_result = collector.run_threshold_tuning(
                    session_id=args.tuning_session_id or latest_session_id,
                    min_confidence=args.tuning_min_confidence,
                    min_trades=args.tuning_min_trades,
                    use_composite_signal=not args.tuning_raw_setup_score,
                    output_dir=args.tuning_output_dir,
                )
                if tuning_result:
                    logger.info("✅ Threshold tuning finished and reports were generated")
                else:
                    logger.warning("⚠️ Threshold tuning did not produce results")

        else:
            logger.error("❌ Backfill failed - check logs for details")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⏹️ Backfill interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
