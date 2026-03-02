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
    python scripts/institutional_backfill.py --full-universe --capture-historical-mda-snapshots
"""

import asyncio
import argparse
import sys
import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env for CLI workflows (backfill/training scripts run outside FastAPI app).
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except Exception:
    pass

from services.institutional_ml_db import InstitutionalMLDatabase, INSTITUTIONAL_UNIVERSE
from utils.logger import setup_logger
from services.market_data_client import MarketDataClient

logger = setup_logger(__name__)


def _parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(',') if item.strip()]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(',') if item.strip()]


def _parse_csv_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(',') if item.strip()]


class InstitutionalDataCollector:
    """Professional data collection orchestrator"""

    def __init__(
        self,
        db_path: Optional[str] = None,
        marketdata_token: Optional[str] = None,
        enable_marketdata: bool = True,
    ):
        self.mda_client: Optional[MarketDataClient] = None
        if enable_marketdata:
            try:
                candidate = MarketDataClient(token=marketdata_token)
                if candidate.is_available():
                    self.mda_client = candidate
                else:
                    logger.info(
                        "ℹ️ MarketData.app client disabled (no token configured). "
                        "Set MARKETDATA_TOKEN or pass --marketdata-token."
                    )
            except Exception as exc:
                logger.warning("⚠️ Failed to initialize MarketData.app client: %s", exc)

        self.db = InstitutionalMLDatabase(db_path, mda_client=self.mda_client)
        self.logger = logger

    async def run_full_backfill(self, symbols: List[str], years: int = 2,
                              batch_size: int = 5,
                              capture_historical_mda_snapshots: bool = False,
                              mda_lookback_years: int = 2) -> bool:
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

            # 2b. Historical option-chain snapshots (MarketData.app)
            if capture_historical_mda_snapshots:
                self.logger.info("📸 Phase 2b: Historical MDApp IV snapshot backfill")
                snapshot_result = self.capture_historical_iv_snapshots_mda(
                    symbols=symbols,
                    lookback_years=mda_lookback_years,
                )
                self.logger.info(
                    "✅ Phase 2b completed: captured=%d, skipped=%d, errors=%d",
                    int(snapshot_result.get("captured", 0)),
                    int(snapshot_result.get("skipped", 0)),
                    int(snapshot_result.get("errors", 0)),
                )

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
            import sqlite3

            rebuilt_symbols = 0
            skipped_symbols = 0
            missing_price_history: List[str] = []

            with sqlite3.connect(self.db.db_path) as conn:
                for symbol in symbols:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM daily_prices WHERE symbol = ?",
                        (symbol,),
                    )
                    price_rows = int(cursor.fetchone()[0] or 0)

                    cursor.execute(
                        "SELECT COUNT(*) FROM ml_features WHERE symbol = ?",
                        (symbol,),
                    )
                    feature_rows = int(cursor.fetchone()[0] or 0)

                    if price_rows == 0:
                        missing_price_history.append(symbol)
                        continue

                    # Recompute features only when coverage is clearly incomplete.
                    coverage = float(feature_rows) / float(price_rows) if price_rows > 0 else 0.0
                    if feature_rows >= 120 and coverage >= 0.80:
                        skipped_symbols += 1
                        continue

                    hist_data = pd.read_sql_query(
                        """
                        SELECT
                            date,
                            open_price AS Open,
                            high_price AS High,
                            low_price AS Low,
                            close_price AS Close,
                            volume AS Volume,
                            adj_close AS "Adj Close",
                            rsi_14 AS RSI_14,
                            bb_position AS BB_Position
                        FROM daily_prices
                        WHERE symbol = ?
                        ORDER BY date ASC
                        """,
                        conn,
                        params=(symbol,),
                        parse_dates=["date"],
                    )

                    if hist_data.empty:
                        missing_price_history.append(symbol)
                        continue

                    hist_data.set_index("date", inplace=True)
                    for col in ("Open", "High", "Low", "Close", "Volume", "Adj Close"):
                        hist_data[col] = pd.to_numeric(hist_data[col], errors="coerce")
                    hist_data.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

                    if hist_data.empty:
                        missing_price_history.append(symbol)
                        continue

                    await self.db._calculate_and_store_features(symbol, hist_data)
                    rebuilt_symbols += 1

            self.logger.info(
                "🔧 ML feature engineering completed: rebuilt=%d, skipped=%d, missing_price_history=%d",
                rebuilt_symbols,
                skipped_symbols,
                len(missing_price_history),
            )
            if missing_price_history:
                self.logger.warning(
                    "⚠️ Symbols with no usable price history for feature engineering: %s",
                    ", ".join(missing_price_history[:10]),
                )
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
                                 lookback_days: int = 30,
                                 lookahead_days: int = 90,
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
                f"chain-errors={diagnostics.get('chain_errors', 0)}, "
                f"outside-window={diagnostics.get('events_outside_window', 0)}"
            )
        return result

    def capture_historical_iv_snapshots_mda(
        self,
        symbols: List[str],
        lookback_years: int = 2,
    ) -> Dict[str, Any]:
        """Backfill historical IV snapshots from MarketData.app chains."""
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not symbols:
            symbols = INSTITUTIONAL_UNIVERSE[:10]
        result = self.db.capture_historical_iv_snapshots_mda(
            symbols=symbols,
            lookback_years=max(1, int(lookback_years)),
        )
        self.logger.info(
            "📚 Historical MDApp snapshots: captured=%d, attempts=%d, events=%d, "
            "skipped=%d, errors=%d, symbols=%d",
            int(result.get("captured", 0)),
            int(result.get("attempts", 0)),
            int(result.get("events_considered", 0)),
            int(result.get("skipped", 0)),
            int(result.get("errors", 0)),
            int(result.get("symbols_processed", 0)),
        )
        diagnostics = result.get("diagnostics") or {}
        if diagnostics:
            self.logger.info(
                "📚 Historical MDApp diagnostics: no-chain=%d, no-expiry-pair=%d, "
                "no-underlying=%d, no-iv=%d",
                int(diagnostics.get("no_chain_rows", 0)),
                int(diagnostics.get("no_expiry_pairs", 0)),
                int(diagnostics.get("no_underlying", 0)),
                int(diagnostics.get("no_iv_values", 0)),
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

    def _normalize_session_filter(self, session_id: Optional[str]) -> Optional[str]:
        """Normalize optional session filter where all/all_sessions means no filter."""
        if not session_id:
            return None
        normalized = str(session_id).strip()
        if not normalized:
            return None
        if normalized.lower() in {"all", "all_sessions", "*"}:
            return None
        return normalized

    def _load_forward_tracker_trades(self, session_id: Optional[str],
                                   lookback_days: int,
                                   min_confidence: float) -> Tuple[pd.DataFrame, str]:
        """Load trade-level frame for forward/paper tracking diagnostics."""
        lookback_days = max(1, int(lookback_days))
        min_confidence = float(np.clip(float(min_confidence), 0.0, 1.0))
        scope_session = self._normalize_session_filter(session_id)
        scope_label = scope_session or "all_sessions"
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        query = """
            SELECT
                t.session_id,
                t.symbol,
                t.trade_date,
                t.event_date,
                t.contracts,
                t.debit_per_contract,
                t.transaction_cost_per_contract,
                t.gross_return_pct,
                t.net_return_pct,
                t.pnl_per_contract,
                t.crush_confidence,
                t.crush_edge_score,
                t.predicted_front_iv_crush_pct,
                l.front_iv_crush_pct AS realized_front_iv_crush_pct
            FROM backtest_trades t
            LEFT JOIN earnings_iv_decay_labels l
              ON t.symbol = l.symbol
             AND t.event_date = l.event_date
            WHERE t.trade_date >= ?
              AND t.crush_confidence >= ?
        """
        params: List[Any] = [cutoff_date, min_confidence]
        if scope_session:
            query += " AND t.session_id = ?"
            params.append(scope_session)
        query += " ORDER BY t.trade_date, t.symbol"

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as exc:
            self.logger.warning("⚠️ Forward tracker query failed: %s", exc)
            return pd.DataFrame(), scope_label

        if df.empty:
            return df, scope_label

        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce').dt.normalize()
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce').dt.normalize()
        df = df.dropna(subset=['trade_date']).copy()
        numeric_cols = [
            'contracts',
            'debit_per_contract',
            'transaction_cost_per_contract',
            'gross_return_pct',
            'net_return_pct',
            'pnl_per_contract',
            'crush_confidence',
            'crush_edge_score',
            'predicted_front_iv_crush_pct',
            'realized_front_iv_crush_pct',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'contracts' in df.columns:
            df['contracts'] = df['contracts'].fillna(1).clip(lower=1)
        return df, scope_label

    def _load_forward_fill_log(self, fill_log_path: Optional[str],
                             lookback_days: int) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
        """Load optional external fill/latency log and aggregate by trade_date."""
        empty_summary = {
            'fill_events': 0,
            'mean_slippage_bps': np.nan,
            'p95_slippage_bps': np.nan,
            'mean_fill_latency_seconds': np.nan,
            'p95_fill_latency_seconds': np.nan,
            'mean_fill_quality_score': np.nan,
        }
        if not fill_log_path:
            return pd.DataFrame(), empty_summary, "No forward fill log provided."

        source_path = Path(str(fill_log_path)).expanduser()
        if not source_path.is_absolute():
            source_path = project_root / source_path
        if not source_path.exists():
            return pd.DataFrame(), empty_summary, f"Fill log not found: {source_path}"

        try:
            raw = pd.read_csv(source_path)
        except Exception as exc:
            return pd.DataFrame(), empty_summary, f"Unable to parse fill log ({exc})"

        if raw.empty:
            return pd.DataFrame(), empty_summary, "Fill log is empty."

        df = raw.copy()
        cols_lower = {str(col).lower(): col for col in df.columns}

        def _pick_column(candidates: List[str]) -> Optional[str]:
            for name in candidates:
                if name in cols_lower:
                    return cols_lower[name]
            return None

        trade_date_col = _pick_column([
            'trade_date', 'date', 'fill_date', 'fill_timestamp', 'fill_ts',
            'signal_timestamp', 'signal_ts',
        ])
        if not trade_date_col:
            return pd.DataFrame(), empty_summary, "Fill log missing a trade-date/timestamp column."

        trade_date = pd.to_datetime(df[trade_date_col], errors='coerce')
        if trade_date.isna().all():
            return pd.DataFrame(), empty_summary, "Fill log has no parseable trade dates."
        df['trade_date'] = trade_date.dt.normalize()

        slippage_col = _pick_column(['slippage_bps', 'slippage_bp', 'slippage'])
        if slippage_col:
            slippage_series = pd.to_numeric(df[slippage_col], errors='coerce')
            # If values look like fractional pct, convert to bps.
            median_abs = float(np.nanmedian(np.abs(slippage_series.to_numpy(dtype=float))))
            if np.isfinite(median_abs) and median_abs <= 1.0:
                slippage_series = slippage_series * 10000.0
            df['slippage_bps'] = slippage_series
        else:
            df['slippage_bps'] = np.nan

        latency_col = _pick_column(['fill_latency_seconds', 'latency_seconds', 'fill_latency_sec'])
        signal_ts_col = _pick_column(['signal_timestamp', 'signal_ts'])
        fill_ts_col = _pick_column(['fill_timestamp', 'fill_ts'])
        if latency_col:
            df['fill_latency_seconds'] = pd.to_numeric(df[latency_col], errors='coerce')
        elif signal_ts_col and fill_ts_col:
            signal_ts = pd.to_datetime(df[signal_ts_col], errors='coerce')
            fill_ts = pd.to_datetime(df[fill_ts_col], errors='coerce')
            df['fill_latency_seconds'] = (fill_ts - signal_ts).dt.total_seconds()
        else:
            df['fill_latency_seconds'] = np.nan

        fill_quality_col = _pick_column(['fill_quality_score', 'fill_quality', 'fill_score'])
        if fill_quality_col:
            df['fill_quality_score'] = pd.to_numeric(df[fill_quality_col], errors='coerce')
        else:
            df['fill_quality_score'] = np.nan

        cutoff_date = pd.Timestamp((datetime.now() - timedelta(days=max(1, int(lookback_days)))).date())
        df = df[df['trade_date'] >= cutoff_date].copy()
        if df.empty:
            return pd.DataFrame(), empty_summary, "Fill log has no rows in selected lookback."

        daily = (
            df.groupby('trade_date', as_index=False)
            .agg(
                fill_events=('trade_date', 'count'),
                avg_live_slippage_bps=('slippage_bps', 'mean'),
                p95_live_slippage_bps=('slippage_bps', lambda x: float(np.nanpercentile(x, 95)) if len(x) else np.nan),
                avg_fill_latency_seconds=('fill_latency_seconds', 'mean'),
                p95_fill_latency_seconds=('fill_latency_seconds', lambda x: float(np.nanpercentile(x, 95)) if len(x) else np.nan),
                avg_fill_quality_score=('fill_quality_score', 'mean'),
            )
        )

        summary = {
            'fill_events': int(len(df)),
            'mean_slippage_bps': float(np.nanmean(df['slippage_bps'])) if df['slippage_bps'].notna().any() else np.nan,
            'p95_slippage_bps': float(np.nanpercentile(df['slippage_bps'].dropna(), 95))
            if df['slippage_bps'].notna().any() else np.nan,
            'mean_fill_latency_seconds': float(np.nanmean(df['fill_latency_seconds']))
            if df['fill_latency_seconds'].notna().any() else np.nan,
            'p95_fill_latency_seconds': float(np.nanpercentile(df['fill_latency_seconds'].dropna(), 95))
            if df['fill_latency_seconds'].notna().any() else np.nan,
            'mean_fill_quality_score': float(np.nanmean(df['fill_quality_score']))
            if df['fill_quality_score'].notna().any() else np.nan,
        }
        return daily, summary, None

    def _compute_forward_tracker_metrics(self, trades_df: pd.DataFrame,
                                       scope_label: str,
                                       lookback_days: int,
                                       min_confidence: float,
                                       fill_daily_df: Optional[pd.DataFrame] = None,
                                       fill_summary: Optional[Dict[str, Any]] = None,
                                       fill_note: Optional[str] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Compute daily and aggregate forward paper-trading diagnostics."""
        if trades_df is None or trades_df.empty:
            return {
                'session_scope': scope_label,
                'lookback_days': int(lookback_days),
                'min_confidence': float(min_confidence),
                'trade_count': 0,
                'labeled_trade_count': 0,
                'label_coverage': 0.0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'mean_predicted_front_iv_crush_pct': np.nan,
                'mean_realized_front_iv_crush_pct': np.nan,
                'crush_prediction_mae': np.nan,
                'crush_directional_accuracy': np.nan,
                'mean_predicted_crush_edge_score': np.nan,
                'mean_execution_drag_bps': np.nan,
                'p95_execution_drag_bps': np.nan,
                'mean_tx_cost_per_contract': np.nan,
                'mean_tx_cost_pct_of_debit': np.nan,
                'fill_metrics': fill_summary or {},
                'fill_note': fill_note,
                'status': 'insufficient_data',
            }, pd.DataFrame()

        frame = trades_df.copy()
        frame['contracts'] = pd.to_numeric(frame['contracts'], errors='coerce').fillna(1.0).clip(lower=1.0)
        frame['pnl_dollars'] = (
            pd.to_numeric(frame['pnl_per_contract'], errors='coerce').fillna(0.0)
            * frame['contracts']
        )
        frame['execution_drag_pct'] = (
            pd.to_numeric(frame['gross_return_pct'], errors='coerce').fillna(0.0)
            - pd.to_numeric(frame['net_return_pct'], errors='coerce').fillna(0.0)
        )
        frame['execution_drag_bps'] = frame['execution_drag_pct'] * 10000.0
        frame['tx_cost_pct_of_debit'] = (
            pd.to_numeric(frame['transaction_cost_per_contract'], errors='coerce').fillna(0.0)
            / pd.to_numeric(frame['debit_per_contract'], errors='coerce').replace(0, np.nan)
        )

        labeled = frame.dropna(subset=['realized_front_iv_crush_pct']).copy()
        if labeled.empty:
            crush_mae = np.nan
            crush_directional_accuracy = np.nan
        else:
            crush_mae = float(
                (labeled['predicted_front_iv_crush_pct'] - labeled['realized_front_iv_crush_pct']).abs().mean()
            )
            crush_directional_accuracy = float(
                ((labeled['predicted_front_iv_crush_pct'] < 0) == (labeled['realized_front_iv_crush_pct'] < 0)).mean()
            )

        daily = (
            frame.groupby('trade_date', as_index=False)
            .agg(
                trades=('symbol', 'count'),
                labeled_trades=('realized_front_iv_crush_pct', lambda x: int(np.isfinite(x).sum())),
                daily_pnl=('pnl_dollars', 'sum'),
                avg_gross_return_pct=('gross_return_pct', 'mean'),
                avg_net_return_pct=('net_return_pct', 'mean'),
                avg_execution_drag_bps=('execution_drag_bps', 'mean'),
                avg_tx_cost_per_contract=('transaction_cost_per_contract', 'mean'),
                avg_tx_cost_pct_of_debit=('tx_cost_pct_of_debit', 'mean'),
                avg_predicted_front_iv_crush_pct=('predicted_front_iv_crush_pct', 'mean'),
                avg_realized_front_iv_crush_pct=('realized_front_iv_crush_pct', 'mean'),
                avg_predicted_crush_edge_score=('crush_edge_score', 'mean'),
                avg_crush_confidence=('crush_confidence', 'mean'),
            )
            .sort_values('trade_date')
            .reset_index(drop=True)
        )
        daily['cumulative_pnl'] = daily['daily_pnl'].cumsum()
        daily['running_peak_pnl'] = daily['cumulative_pnl'].cummax()
        daily['drawdown'] = daily['cumulative_pnl'] - daily['running_peak_pnl']
        if fill_daily_df is not None and not fill_daily_df.empty:
            merged = fill_daily_df.copy()
            merged['trade_date'] = pd.to_datetime(merged['trade_date'], errors='coerce').dt.normalize()
            daily = daily.merge(merged, on='trade_date', how='left')

        trade_count = int(len(frame))
        max_drawdown = float(daily['drawdown'].min()) if not daily.empty else 0.0
        summary = {
            'session_scope': scope_label,
            'lookback_days': int(lookback_days),
            'min_confidence': float(min_confidence),
            'trade_count': trade_count,
            'labeled_trade_count': int(len(labeled)),
            'label_coverage': float(len(labeled) / trade_count) if trade_count > 0 else 0.0,
            'total_pnl': float(frame['pnl_dollars'].sum()),
            'mean_daily_pnl': float(daily['daily_pnl'].mean()) if not daily.empty else np.nan,
            'max_drawdown': max_drawdown,
            'mean_predicted_front_iv_crush_pct': float(frame['predicted_front_iv_crush_pct'].mean()),
            'mean_realized_front_iv_crush_pct': (
                float(labeled['realized_front_iv_crush_pct'].mean()) if not labeled.empty else np.nan
            ),
            'crush_prediction_mae': crush_mae,
            'crush_directional_accuracy': crush_directional_accuracy,
            'mean_predicted_crush_edge_score': float(frame['crush_edge_score'].mean()),
            'mean_execution_drag_bps': float(frame['execution_drag_bps'].mean()),
            'p95_execution_drag_bps': float(np.percentile(frame['execution_drag_bps'], 95)),
            'mean_tx_cost_per_contract': float(frame['transaction_cost_per_contract'].mean()),
            'mean_tx_cost_pct_of_debit': (
                float(frame['tx_cost_pct_of_debit'].mean())
                if frame['tx_cost_pct_of_debit'].notna().any() else np.nan
            ),
            'fill_metrics': fill_summary or {},
            'fill_note': fill_note,
            'status': 'ok' if trade_count >= 20 else 'insufficient_data',
        }
        return summary, daily

    @staticmethod
    def _grade_rank(grade: Optional[str]) -> float:
        """Convert grade string (A/B/C/...) into sortable numeric rank."""
        if not grade:
            return 0.0
        text = str(grade).strip().upper()
        if not text:
            return 0.0
        base = text[0]
        mapping = {'A': 5.0, 'B': 4.0, 'C': 3.0, 'D': 2.0, 'E': 1.0, 'F': 0.0}
        rank = float(mapping.get(base, 0.0))
        if '+' in text:
            rank += 0.25
        elif '-' in text:
            rank -= 0.25
        return rank

    def run_forward_paper_tracker(self, session_id: Optional[str] = None,
                                lookback_days: int = 120,
                                min_confidence: float = 0.35,
                                output_dir: str = "exports/reports",
                                fill_log_path: Optional[str] = None,
                                write_artifacts: bool = True) -> Optional[dict]:
        """Track forward paper performance: predicted vs realized crush edge and execution realism."""
        lookback_days = max(1, int(lookback_days))
        min_confidence = float(np.clip(float(min_confidence), 0.0, 1.0))
        trades_df, scope_label = self._load_forward_tracker_trades(
            session_id=session_id,
            lookback_days=lookback_days,
            min_confidence=min_confidence,
        )
        fill_daily, fill_summary, fill_note = self._load_forward_fill_log(
            fill_log_path=fill_log_path,
            lookback_days=lookback_days,
        )
        summary, daily = self._compute_forward_tracker_metrics(
            trades_df=trades_df,
            scope_label=scope_label,
            lookback_days=lookback_days,
            min_confidence=min_confidence,
            fill_daily_df=fill_daily,
            fill_summary=fill_summary,
            fill_note=fill_note,
        )

        if not write_artifacts:
            return {
                'summary': summary,
                'daily_rows': int(len(daily)),
                'session_scope': scope_label,
                'status': summary.get('status'),
            }

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f"forward_paper_tracker_{timestamp}.csv"
        md_path = output_path / f"forward_paper_tracker_{timestamp}.md"
        json_path = output_path / f"forward_paper_tracker_{timestamp}.json"

        if daily.empty:
            pd.DataFrame(columns=['trade_date']).to_csv(csv_path, index=False)
        else:
            daily_out = daily.copy()
            daily_out['trade_date'] = daily_out['trade_date'].dt.strftime('%Y-%m-%d')
            daily_out.to_csv(csv_path, index=False)

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Forward Paper Trading Tracker\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Session scope: `{summary.get('session_scope', 'all_sessions')}`\n")
            f.write(f"- Lookback days: `{summary.get('lookback_days')}`\n")
            f.write(f"- Min confidence filter: `{summary.get('min_confidence'):.2f}`\n")
            f.write(f"- Status: `{summary.get('status')}`\n\n")

            f.write("## Prediction vs Realization\n\n")
            f.write(f"- Trades: `{summary.get('trade_count', 0)}`\n")
            f.write(f"- Labeled trades: `{summary.get('labeled_trade_count', 0)}`\n")
            f.write(f"- Label coverage: `{summary.get('label_coverage', 0.0):.2%}`\n")
            f.write(
                f"- Mean predicted front IV crush: "
                f"`{summary.get('mean_predicted_front_iv_crush_pct', float('nan')):.2%}`\n"
            )
            f.write(
                f"- Mean realized front IV crush: "
                f"`{summary.get('mean_realized_front_iv_crush_pct', float('nan')):.2%}`\n"
            )
            f.write(f"- Crush MAE: `{summary.get('crush_prediction_mae', float('nan')):.4f}`\n")
            f.write(
                f"- Crush directional accuracy: "
                f"`{summary.get('crush_directional_accuracy', float('nan')):.2%}`\n"
            )
            f.write(
                f"- Mean predicted crush-edge score: "
                f"`{summary.get('mean_predicted_crush_edge_score', float('nan')):.4f}`\n\n"
            )

            f.write("## Execution Realism\n\n")
            f.write(f"- Total PnL: `${summary.get('total_pnl', 0.0):.2f}`\n")
            f.write(f"- Mean daily PnL: `${summary.get('mean_daily_pnl', float('nan')):.2f}`\n")
            f.write(f"- Max drawdown: `${summary.get('max_drawdown', 0.0):.2f}`\n")
            f.write(
                f"- Mean execution drag: "
                f"`{summary.get('mean_execution_drag_bps', float('nan')):.2f} bps`\n"
            )
            f.write(
                f"- P95 execution drag: "
                f"`{summary.get('p95_execution_drag_bps', float('nan')):.2f} bps`\n"
            )
            f.write(
                f"- Mean tx cost per contract: "
                f"`${summary.get('mean_tx_cost_per_contract', float('nan')):.2f}`\n"
            )
            f.write(
                f"- Mean tx cost / debit: "
                f"`{summary.get('mean_tx_cost_pct_of_debit', float('nan')):.2%}`\n"
            )
            fill_metrics = summary.get('fill_metrics', {}) or {}
            f.write(
                f"- Mean live slippage (fill log): "
                f"`{float(fill_metrics.get('mean_slippage_bps', np.nan)):.2f} bps`\n"
            )
            f.write(
                f"- Mean fill latency (fill log): "
                f"`{float(fill_metrics.get('mean_fill_latency_seconds', np.nan)):.2f}s`\n"
            )
            f.write(
                f"- Mean fill quality (fill log): "
                f"`{float(fill_metrics.get('mean_fill_quality_score', np.nan)):.4f}`\n"
            )
            if summary.get('fill_note'):
                f.write(f"- Fill log note: `{summary.get('fill_note')}`\n")

            if not daily.empty:
                preview_cols = [
                    'trade_date', 'trades', 'daily_pnl', 'cumulative_pnl', 'drawdown',
                    'avg_predicted_front_iv_crush_pct', 'avg_realized_front_iv_crush_pct',
                    'avg_execution_drag_bps', 'avg_tx_cost_pct_of_debit',
                ]
                available_cols = [col for col in preview_cols if col in daily.columns]
                preview_df = daily[available_cols].tail(20).copy()
                if 'trade_date' in preview_df.columns:
                    preview_df['trade_date'] = preview_df['trade_date'].dt.strftime('%Y-%m-%d')
                f.write("\n## Daily Tracker (Latest 20 rows)\n\n")
                f.write("```text\n")
                f.write(preview_df.to_string(index=False))
                f.write("\n```\n")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump({
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'summary': summary,
                'daily_rows': int(len(daily)),
                'source_fill_log': str(fill_log_path) if fill_log_path else None,
                'source_csv': str(csv_path),
            }, f, indent=2, default=str)

        self.logger.info(f"✅ Forward paper tracker saved: {csv_path}")
        self.logger.info(f"✅ Forward paper tracker summary saved: {md_path}")
        self.logger.info(f"✅ Forward paper tracker JSON saved: {json_path}")
        return {
            'csv_path': str(csv_path),
            'markdown_path': str(md_path),
            'json_path': str(json_path),
            'session_scope': scope_label,
            'summary': summary,
            'rows': int(len(daily)),
            'status': summary.get('status'),
        }

    def _resolve_session_for_reports(self, session_id: Optional[str] = None) -> Optional[str]:
        """Resolve report session scope (explicit session id preferred, fallback latest)."""
        if session_id:
            normalized = str(session_id).strip()
            if normalized.lower() in {"all", "all_sessions", "*"}:
                return None
            return normalized
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

    def run_trade_stress_test(self, session_id: Optional[str] = None,
                            simulations: int = 5000,
                            confidence_level: float = 0.95,
                            drawdown_limit: float = -600.0,
                            loss_limit: float = 0.0,
                            seed: Optional[int] = 42,
                            output_dir: str = "exports/reports",
                            min_reliable_trades: int = 20) -> Optional[dict]:
        """Run bootstrap Monte Carlo stress test on trade-level PnL paths."""
        requested_scope = str(session_id or "").strip().lower()
        use_all_sessions = requested_scope in {"all", "all_sessions", "*"}
        resolved_session = "all_sessions" if use_all_sessions else self._resolve_session_for_reports(session_id)
        if not resolved_session:
            self.logger.warning("⚠️ Stress test unavailable: no backtest session found")
            return None

        simulations = max(500, int(simulations))
        confidence_level = float(np.clip(float(confidence_level), 0.50, 0.99))
        drawdown_limit = float(drawdown_limit)
        loss_limit = float(loss_limit)
        min_reliable_trades = max(3, int(min_reliable_trades))
        rng = np.random.default_rng(seed if seed is not None else None)

        if use_all_sessions:
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    trades_df = pd.read_sql_query(
                        "SELECT * FROM backtest_trades",
                        conn,
                    )
            except Exception as exc:
                self.logger.warning("⚠️ Stress test unavailable: failed loading all-session trades (%s)", exc)
                return None
        else:
            trades_df = self.db.get_backtest_trades(resolved_session)
        if trades_df.empty:
            self.logger.warning("⚠️ Stress test unavailable: no trades for session %s", resolved_session)
            return None
        if 'pnl' in trades_df.columns:
            pnl_series = pd.to_numeric(trades_df['pnl'], errors='coerce').dropna().to_numpy(dtype=float)
        elif 'pnl_per_contract' in trades_df.columns:
            contracts_vec = (
                pd.to_numeric(trades_df['contracts'], errors='coerce').fillna(1.0).to_numpy(dtype=float)
                if 'contracts' in trades_df.columns
                else np.ones(len(trades_df), dtype=float)
            )
            pnl_per_contract = pd.to_numeric(trades_df['pnl_per_contract'], errors='coerce').to_numpy(dtype=float)
            pnl_series = (contracts_vec * pnl_per_contract)
            pnl_series = pnl_series[np.isfinite(pnl_series)]
        else:
            self.logger.warning(
                "⚠️ Stress test unavailable: trade records missing pnl and pnl_per_contract columns"
            )
            return None
        if pnl_series.size < 3:
            self.logger.warning(
                "⚠️ Stress test unavailable: insufficient trades for session %s (n=%d)",
                resolved_session,
                int(pnl_series.size),
            )
            return None

        returns_series = None
        if 'net_return_pct' in trades_df.columns:
            returns_series = pd.to_numeric(
                trades_df['net_return_pct'],
                errors='coerce',
            ).dropna().to_numpy(dtype=float)
            if returns_series.size != pnl_series.size:
                returns_series = None

        trade_count = int(pnl_series.size)
        simulation_horizon_trades = trade_count
        if use_all_sessions and trade_count > min_reliable_trades:
            simulation_horizon_trades = max(min_reliable_trades, 20)
        sample_reliability = float(min(1.0, trade_count / float(min_reliable_trades)))
        sample_size_note = (
            f"Trade sample underpowered: {trade_count} trades < {min_reliable_trades} reliable minimum."
            if trade_count < min_reliable_trades
            else None
        )
        sampled_idx = rng.integers(0, trade_count, size=(simulations, simulation_horizon_trades))
        sampled_pnl = pnl_series[sampled_idx]
        cumulative = np.cumsum(sampled_pnl, axis=1)
        peaks = np.maximum.accumulate(cumulative, axis=1)
        sampled_drawdown = np.min(cumulative - peaks, axis=1)
        sampled_total_pnl = np.sum(sampled_pnl, axis=1)
        sampled_win_rate = np.mean(sampled_pnl > 0.0, axis=1)

        def _max_drawdown(pnl_path: np.ndarray) -> float:
            if pnl_path.size == 0:
                return 0.0
            cum = np.cumsum(pnl_path)
            running_peak = np.maximum.accumulate(cum)
            return float(np.min(cum - running_peak))

        base_path = pnl_series[-simulation_horizon_trades:]
        base_total_pnl = float(np.sum(base_path))
        base_max_drawdown = _max_drawdown(base_path)
        base_win_rate = float(np.mean(base_path > 0.0))

        var_cutoff = float(np.percentile(sampled_total_pnl, (1.0 - confidence_level) * 100.0))
        tail_mask = sampled_total_pnl <= var_cutoff
        cvar_total_pnl = float(sampled_total_pnl[tail_mask].mean()) if bool(np.any(tail_mask)) else var_cutoff

        prob_loss = float(np.mean(sampled_total_pnl <= loss_limit))
        prob_drawdown_breach = float(np.mean(sampled_drawdown <= drawdown_limit))
        robustness_score = float(
            np.clip(
                1.0 - 0.60 * prob_loss - 0.40 * prob_drawdown_breach,
                0.0,
                1.0,
            )
        )
        if prob_loss > 0.50 or prob_drawdown_breach > 0.40:
            status = "fail"
        elif prob_loss > 0.35 or prob_drawdown_breach > 0.25:
            status = "advisory"
        else:
            status = "pass"
        if trade_count < min_reliable_trades:
            status = "advisory" if status == "pass" else status
            if trade_count < max(5, min_reliable_trades // 3):
                status = "fail"
            robustness_score = float(np.clip(robustness_score * sample_reliability, 0.0, 1.0))

        sampled_sharpe = None
        base_sharpe = None
        if returns_series is not None and returns_series.size > 2:
            hold_days_avg = 1.0
            if 'hold_days' in trades_df.columns:
                hold_days_vec = pd.to_numeric(trades_df['hold_days'], errors='coerce').dropna().to_numpy(dtype=float)
                if hold_days_vec.size > 0:
                    hold_days_avg = float(max(1.0, hold_days_vec.mean()))
            annualization = float(np.sqrt(252.0 / hold_days_avg))
            returns_sampled = returns_series[sampled_idx]
            sample_ret_mean = returns_sampled.mean(axis=1)
            sample_ret_std = returns_sampled.std(axis=1, ddof=0)
            sampled_sharpe = np.full(sample_ret_mean.shape, np.nan, dtype=float)
            stable_std = sample_ret_std > 1e-9
            sampled_sharpe[stable_std] = (
                sample_ret_mean[stable_std] / sample_ret_std[stable_std]
            ) * annualization

            base_returns = returns_series[-simulation_horizon_trades:]
            base_ret_std = float(np.std(base_returns, ddof=0))
            base_ret_mean = float(np.mean(base_returns))
            if base_ret_std > 1e-9:
                base_sharpe = float((base_ret_mean / base_ret_std) * annualization)

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f"earnings_trade_stress_test_{timestamp}.csv"
        md_path = output_path / f"earnings_trade_stress_test_{timestamp}.md"
        json_path = output_path / f"earnings_trade_stress_test_{timestamp}.json"

        summary_df = pd.DataFrame([{
            'session_id': resolved_session,
            'simulations': simulations,
            'trade_count': trade_count,
            'simulation_horizon_trades': simulation_horizon_trades,
            'base_total_pnl': base_total_pnl,
            'base_max_drawdown': base_max_drawdown,
            'base_win_rate': base_win_rate,
            'base_sharpe': base_sharpe if base_sharpe is not None else np.nan,
            'sim_p05_total_pnl': float(np.percentile(sampled_total_pnl, 5)),
            'sim_p50_total_pnl': float(np.percentile(sampled_total_pnl, 50)),
            'sim_p95_total_pnl': float(np.percentile(sampled_total_pnl, 95)),
            'sim_var_total_pnl': var_cutoff,
            'sim_cvar_total_pnl': cvar_total_pnl,
            'sim_mean_max_drawdown': float(np.mean(sampled_drawdown)),
            'sim_p95_max_drawdown': float(np.percentile(sampled_drawdown, 5)),
            'sim_mean_win_rate': float(np.mean(sampled_win_rate)),
            'sim_p05_win_rate': float(np.percentile(sampled_win_rate, 5)),
            'sim_p95_win_rate': float(np.percentile(sampled_win_rate, 95)),
            'prob_total_pnl_le_loss_limit': prob_loss,
            'prob_drawdown_breach': prob_drawdown_breach,
            'robustness_score': robustness_score,
            'sample_reliability': sample_reliability,
            'min_reliable_trades': min_reliable_trades,
            'status': status,
            'confidence_level': confidence_level,
            'drawdown_limit': drawdown_limit,
            'loss_limit': loss_limit,
        }])
        summary_df.to_csv(csv_path, index=False)

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings Trade Stress Test\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Session: `{resolved_session}`\n")
            f.write(f"- Simulations: `{simulations}`\n")
            f.write(f"- Trade count: `{trade_count}`\n")
            f.write(f"- Simulation horizon trades: `{simulation_horizon_trades}`\n")
            f.write(f"- Min reliable trades: `{min_reliable_trades}`\n")
            f.write(f"- Sample reliability: `{sample_reliability:.2f}`\n")
            f.write(f"- Status: `{status}`\n")
            f.write(f"- Robustness score: `{robustness_score:.3f}`\n\n")
            if sample_size_note:
                f.write(f"- Sample note: {sample_size_note}\n\n")
            f.write("## Risk Metrics\n\n")
            f.write(f"- Probability total PnL <= {loss_limit:.2f}: `{prob_loss:.2%}`\n")
            f.write(f"- Probability max drawdown <= {drawdown_limit:.2f}: `{prob_drawdown_breach:.2%}`\n")
            f.write(f"- Total PnL VaR({confidence_level:.0%}): `${var_cutoff:.2f}`\n")
            f.write(f"- Total PnL CVaR({confidence_level:.0%}): `${cvar_total_pnl:.2f}`\n")
            f.write(f"- Total PnL P05 / P50 / P95: `${np.percentile(sampled_total_pnl, 5):.2f}` / `${np.percentile(sampled_total_pnl, 50):.2f}` / `${np.percentile(sampled_total_pnl, 95):.2f}`\n")
            f.write(f"- Max drawdown mean / P05(worst): `${np.mean(sampled_drawdown):.2f}` / `${np.percentile(sampled_drawdown, 5):.2f}`\n")
            f.write("\n## Base Session\n\n")
            f.write(f"- Base total PnL: `${base_total_pnl:.2f}`\n")
            f.write(f"- Base max drawdown: `${base_max_drawdown:.2f}`\n")
            f.write(f"- Base win rate: `{base_win_rate:.2%}`\n")
            if base_sharpe is not None:
                f.write(f"- Base Sharpe: `{base_sharpe:.4f}`\n")

        def _finite_stat(values: Optional[np.ndarray], fn) -> Optional[float]:
            if values is None:
                return None
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                return None
            return float(fn(finite))

        payload = {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'session_id': resolved_session,
            'simulations': simulations,
            'confidence_level': confidence_level,
            'drawdown_limit': drawdown_limit,
            'loss_limit': loss_limit,
            'trade_count': trade_count,
            'simulation_horizon_trades': simulation_horizon_trades,
            'min_reliable_trades': min_reliable_trades,
            'sample_reliability': sample_reliability,
            'sample_size_note': sample_size_note,
            'status': status,
            'robustness_score': robustness_score,
            'base': {
                'total_pnl': base_total_pnl,
                'max_drawdown': base_max_drawdown,
                'win_rate': base_win_rate,
                'sharpe': base_sharpe,
            },
            'simulation': {
                'mean_total_pnl': float(np.mean(sampled_total_pnl)),
                'median_total_pnl': float(np.percentile(sampled_total_pnl, 50)),
                'p05_total_pnl': float(np.percentile(sampled_total_pnl, 5)),
                'p95_total_pnl': float(np.percentile(sampled_total_pnl, 95)),
                'var_total_pnl': var_cutoff,
                'cvar_total_pnl': cvar_total_pnl,
                'mean_max_drawdown': float(np.mean(sampled_drawdown)),
                'p05_max_drawdown': float(np.percentile(sampled_drawdown, 5)),
                'mean_win_rate': float(np.mean(sampled_win_rate)),
                'p05_win_rate': float(np.percentile(sampled_win_rate, 5)),
                'p95_win_rate': float(np.percentile(sampled_win_rate, 95)),
                'prob_total_pnl_le_loss_limit': prob_loss,
                'prob_drawdown_breach': prob_drawdown_breach,
                'mean_sharpe': _finite_stat(sampled_sharpe, np.mean),
                'p05_sharpe': _finite_stat(sampled_sharpe, lambda a: np.percentile(a, 5)),
                'p95_sharpe': _finite_stat(sampled_sharpe, lambda a: np.percentile(a, 95)),
            },
            'paths': {
                'csv': str(csv_path),
                'markdown': str(md_path),
                'json': str(json_path),
            },
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        self.logger.info(f"✅ Stress test summary saved: {md_path}")
        self.logger.info(f"✅ Stress test JSON saved: {json_path}")
        return {
            'session_id': resolved_session,
            'status': status,
            'robustness_score': robustness_score,
            'trade_count': trade_count,
            'simulation_horizon_trades': simulation_horizon_trades,
            'min_reliable_trades': min_reliable_trades,
            'sample_reliability': sample_reliability,
            'sample_size_note': sample_size_note,
            'prob_total_pnl_le_loss_limit': prob_loss,
            'prob_drawdown_breach': prob_drawdown_breach,
            'var_total_pnl': var_cutoff,
            'cvar_total_pnl': cvar_total_pnl,
            'csv_path': str(csv_path),
            'markdown_path': str(md_path),
            'json_path': str(json_path),
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
                          min_trades_for_ranking: int = 0,
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
        min_trades_for_ranking = max(0, int(min_trades_for_ranking))

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
        ranked_for_selection = ranked
        excluded_by_trade_count = 0
        if min_trades_for_ranking > 0 and 'total_trades' in ranked.columns:
            ranked_for_selection = ranked[ranked['total_trades'] >= min_trades_for_ranking].copy()
            excluded_by_trade_count = int(len(ranked) - len(ranked_for_selection))
            if ranked_for_selection.empty:
                self.logger.warning(
                    "⚠️ Sweep min-trades filter removed all rows (min_trades=%d); "
                    "falling back to full ranking",
                    min_trades_for_ranking,
                )
                ranked_for_selection = ranked
                excluded_by_trade_count = 0
            elif excluded_by_trade_count > 0:
                self.logger.info(
                    "🧮 Sweep min-trades filter: excluded %d/%d rows (min_trades=%d)",
                    excluded_by_trade_count,
                    int(len(ranked)),
                    min_trades_for_ranking,
                )
        top_rows = ranked_for_selection.head(top_n).copy()
        inactive_sweep_params: List[str] = []
        param_columns = [col for col in ranked.columns if col.startswith("param_")]
        for param_col in param_columns:
            if ranked[param_col].nunique(dropna=False) <= 1:
                continue
            grouping_cols = [col for col in param_columns if col != param_col]
            if not grouping_cols:
                continue
            pnl_std = ranked.groupby(grouping_cols, dropna=False)['total_pnl'].std(ddof=0).fillna(0.0)
            sharpe_std = ranked.groupby(grouping_cols, dropna=False)['sharpe_ratio'].std(ddof=0).fillna(0.0)
            if bool((pnl_std <= 1e-9).all()) and bool((sharpe_std <= 1e-9).all()):
                inactive_sweep_params.append(param_col[6:])
        if inactive_sweep_params:
            joined = ", ".join(inactive_sweep_params)
            self.logger.warning(
                "⚠️ Sweep dimensions with no observed impact on PnL/Sharpe: %s",
                joined,
            )

        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings Walk-Forward Parameter Sweep\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Total combinations: {len(ranked)}\n")
            f.write(f"- Top rows shown: {min(top_n, len(ranked_for_selection))}\n\n")

            best = ranked_for_selection.iloc[0]
            f.write("## Best Configuration\n\n")
            f.write(f"- Session: `{best['session_id']}`\n")
            f.write(f"- Alpha score: `{best['alpha_score']:.4f}`\n")
            f.write(f"- Sharpe: `{best['sharpe_ratio']:.4f}`\n")
            f.write(f"- Win rate: `{best['win_rate']:.2%}`\n")
            f.write(f"- Total PnL: `${best['total_pnl']:.2f}`\n")
            f.write(f"- Max drawdown: `${best['max_drawdown']:.2f}`\n\n")
            if min_trades_for_ranking > 0 and 'total_trades' in best.index:
                f.write(
                    f"- Min trades for ranking: `{min_trades_for_ranking}` "
                    f"(excluded rows: `{excluded_by_trade_count}`)\n\n"
                )

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
            if inactive_sweep_params:
                f.write("\n## Inactive Sweep Dimensions\n\n")
                f.write("No measurable PnL/Sharpe change detected for:\n")
                for name in inactive_sweep_params:
                    f.write(f"- `{name}`\n")

        best_params = {
            key[6:]: best[key]
            for key in ranked_for_selection.columns
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
                'inactive_sweep_params': inactive_sweep_params,
                'min_trades_for_ranking': min_trades_for_ranking,
                'excluded_rows_by_min_trades': excluded_by_trade_count,
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
            'top_rows': int(min(top_n, len(ranked_for_selection))),
            'best_params': best_params,
            'best_session_id': str(best['session_id']),
            'inactive_sweep_params': inactive_sweep_params,
            'min_trades_for_ranking': min_trades_for_ranking,
            'excluded_rows_by_min_trades': excluded_by_trade_count,
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
        # Build per-split timeseries for API consumers (chart data)
        def _safe_f(v, default=0.0):
            try:
                f = float(v)
                return f if (f == f) else default  # NaN check
            except (TypeError, ValueError):
                return default

        splits_detail = [
            {
                "split": int(row.get("split_index", i + 1)),
                "test_start": str(row.get("test_start", "")),
                "test_end": str(row.get("test_end", "")),
                "pnl": _safe_f(row.get("test_total_pnl")),
                "alpha": _safe_f(row.get("test_alpha_score")),
                "sharpe": _safe_f(row.get("test_sharpe_ratio")),
                "win_rate": _safe_f(row.get("test_win_rate")),
                "trades": int(row.get("test_total_trades") or 0),
            }
            for i, (_, row) in enumerate(oos_df.iterrows())
        ]

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
            'splits_detail': splits_detail,
        }

    def run_readiness_report(self, execution_profiles: List[str],
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
                           sweep_output_dir: str = "exports/reports",
                           oos_output_dir: str = "exports/reports",
                           tuning_output_dir: str = "exports/reports",
                           readiness_output_dir: str = "exports/reports",
                           sweep_top_n: int = 20,
                           sweep_min_trades: int = 12,
                           tuning_session_id: Optional[str] = None,
                           tuning_min_confidence: float = 0.35,
                           tuning_min_trades: int = 30,
                           tuning_use_composite_signal: bool = True,
                           run_stress_test: bool = True,
                           stress_session_id: Optional[str] = None,
                           stress_simulations: int = 5000,
                           stress_confidence_level: float = 0.95,
                           stress_drawdown_limit: float = -600.0,
                           stress_loss_limit: float = 0.0,
                           stress_seed: Optional[int] = 42,
                           stress_min_trades: int = 20,
                           stress_output_dir: str = "exports/reports",
                           promotion_min_oos_grade: str = "A",
                           promotion_min_live_trades: int = 50,
                           promotion_live_lookback_days: int = 120,
                           promotion_live_session_id: Optional[str] = None,
                           promotion_live_min_confidence: float = 0.35,
                           forward_fill_log_path: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Optional[dict]:
        """Run sweep + OOS + threshold tuning and emit a consolidated readiness report."""
        self.logger.info("🧪 Running consolidated readiness workflow")

        sweep_result = self.run_parameter_sweep(
            execution_profiles=execution_profiles,
            hold_days_grid=hold_days_grid,
            signal_threshold_grid=signal_threshold_grid,
            trades_per_day_grid=trades_per_day_grid,
            entry_days_grid=entry_days_grid,
            exit_days_grid=exit_days_grid,
            target_entry_dte=target_entry_dte,
            entry_dte_band=entry_dte_band,
            min_daily_share_volume=min_daily_share_volume,
            max_abs_momentum_5d=max_abs_momentum_5d,
            position_contracts=position_contracts,
            lookback_days=lookback_days,
            max_backtest_symbols=max_backtest_symbols,
            use_crush_confidence_gate=use_crush_confidence_gate,
            allow_global_crush_profile=allow_global_crush_profile,
            min_crush_confidence=min_crush_confidence,
            min_crush_magnitude=min_crush_magnitude,
            min_crush_edge=min_crush_edge,
            require_true_earnings=require_true_earnings,
            allow_proxy_earnings=allow_proxy_earnings,
            output_dir=sweep_output_dir,
            top_n=sweep_top_n,
            min_trades_for_ranking=sweep_min_trades,
            start_date=start_date,
            end_date=end_date,
        )
        if not sweep_result:
            self.logger.warning("⚠️ Readiness workflow stopped: parameter sweep produced no results")
            return None

        oos_result = self.run_oos_validation(
            execution_profiles=execution_profiles,
            hold_days_grid=hold_days_grid,
            signal_threshold_grid=signal_threshold_grid,
            trades_per_day_grid=trades_per_day_grid,
            entry_days_grid=entry_days_grid,
            exit_days_grid=exit_days_grid,
            target_entry_dte=target_entry_dte,
            entry_dte_band=entry_dte_band,
            min_daily_share_volume=min_daily_share_volume,
            max_abs_momentum_5d=max_abs_momentum_5d,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            top_n_train=top_n_train,
            position_contracts=position_contracts,
            lookback_days=lookback_days,
            max_backtest_symbols=max_backtest_symbols,
            use_crush_confidence_gate=use_crush_confidence_gate,
            allow_global_crush_profile=allow_global_crush_profile,
            min_crush_confidence=min_crush_confidence,
            min_crush_magnitude=min_crush_magnitude,
            min_crush_edge=min_crush_edge,
            require_true_earnings=require_true_earnings,
            allow_proxy_earnings=allow_proxy_earnings,
            min_splits=min_splits,
            min_total_test_trades=min_total_test_trades,
            min_trades_per_split=min_trades_per_split,
            output_dir=oos_output_dir,
            start_date=start_date,
            end_date=end_date,
        )
        if not oos_result:
            self.logger.warning("⚠️ Readiness workflow stopped: OOS validation produced no results")
            return None

        preferred_tuning_session = (
            tuning_session_id
            or str(oos_result.get('best_test_session_id') or "")
            or str(sweep_result.get('best_session_id') or "")
            or None
        )
        tuning_result = self.run_threshold_tuning(
            session_id=preferred_tuning_session,
            min_confidence=tuning_min_confidence,
            min_trades=tuning_min_trades,
            use_composite_signal=tuning_use_composite_signal,
            output_dir=tuning_output_dir,
        )
        if not tuning_result:
            self.logger.warning("⚠️ Readiness workflow: threshold tuning produced no decile report")
        elif (
            preferred_tuning_session
            and not tuning_result.get('recommended')
            and str(tuning_result.get('reason', '')) == 'insufficient_trade_count_for_thresholds'
        ):
            self.logger.info(
                "ℹ️ Threshold tuning had thin sample on scoped session; retrying across all sessions"
            )
            broader_tuning_result = self.run_threshold_tuning(
                session_id="all_sessions",
                min_confidence=tuning_min_confidence,
                min_trades=tuning_min_trades,
                use_composite_signal=tuning_use_composite_signal,
                output_dir=tuning_output_dir,
            )
            if broader_tuning_result and int(broader_tuning_result.get('candidate_rows', 0)) >= int(
                tuning_result.get('candidate_rows', 0)
            ):
                tuning_result = broader_tuning_result

        stress_result = None
        if run_stress_test:
            stress_result = self.run_trade_stress_test(
                session_id=stress_session_id or str(oos_result.get('best_test_session_id') or ""),
                simulations=stress_simulations,
                confidence_level=stress_confidence_level,
                drawdown_limit=stress_drawdown_limit,
                loss_limit=stress_loss_limit,
                seed=stress_seed,
                min_reliable_trades=stress_min_trades,
                output_dir=stress_output_dir,
            )
            if (
                not stress_session_id
                and stress_result
                and int(stress_result.get('trade_count', 0) or 0) < int(stress_min_trades)
            ):
                self.logger.info(
                    "ℹ️ Stress test sample was underpowered (n=%d < min=%d); retrying on all sessions",
                    int(stress_result.get('trade_count', 0) or 0),
                    int(stress_min_trades),
                )
                broader_stress_result = self.run_trade_stress_test(
                    session_id="all_sessions",
                    simulations=stress_simulations,
                    confidence_level=stress_confidence_level,
                    drawdown_limit=stress_drawdown_limit,
                    loss_limit=stress_loss_limit,
                    seed=stress_seed,
                    min_reliable_trades=stress_min_trades,
                    output_dir=stress_output_dir,
                )
                if broader_stress_result and int(broader_stress_result.get('trade_count', 0) or 0) >= int(
                    stress_result.get('trade_count', 0) or 0
                ):
                    stress_result = broader_stress_result

        promotion_min_live_trades = max(0, int(promotion_min_live_trades))
        promotion_live_lookback_days = max(1, int(promotion_live_lookback_days))
        promotion_live_min_confidence = float(np.clip(float(promotion_live_min_confidence), 0.0, 1.0))
        required_oos_grade = str(promotion_min_oos_grade or "A").strip().upper() or "A"
        forward_tracker_result = self.run_forward_paper_tracker(
            session_id=promotion_live_session_id or "all_sessions",
            lookback_days=promotion_live_lookback_days,
            min_confidence=promotion_live_min_confidence,
            output_dir=readiness_output_dir,
            fill_log_path=forward_fill_log_path,
            write_artifacts=True,
        )
        forward_summary = dict((forward_tracker_result or {}).get('summary') or {})
        live_trade_count = int(forward_summary.get('trade_count', 0) or 0)

        report_card = oos_result.get('report_card', {}) if isinstance(oos_result, dict) else {}
        verdict = report_card.get('verdict', {}) if isinstance(report_card, dict) else {}
        oos_grade = str(verdict.get('grade', 'n/a') if isinstance(verdict, dict) else 'n/a')
        oos_pass = bool(verdict.get('overall_pass', False))
        threshold_recommended = bool((tuning_result or {}).get('recommended', False))
        inactive_sweep_params = list(sweep_result.get('inactive_sweep_params', []) or [])
        stress_status = str((stress_result or {}).get('status', 'not_run'))
        stress_fail = stress_status in {'fail', 'insufficient_data'}
        stress_advisory = stress_status == 'advisory'
        if run_stress_test and stress_result is None:
            stress_fail = True

        if oos_pass and threshold_recommended and not inactive_sweep_params and not stress_advisory and not stress_fail:
            readiness_status = "ready"
        elif oos_pass and not stress_fail:
            readiness_status = "ready_with_advisories"
        else:
            readiness_status = "not_ready"

        grade_gate_pass = self._grade_rank(oos_grade) >= self._grade_rank(required_oos_grade)
        live_trade_gate_pass = live_trade_count >= promotion_min_live_trades
        promotion_gate_pass = bool(grade_gate_pass or live_trade_gate_pass)
        if readiness_status != "not_ready" and not promotion_gate_pass:
            readiness_status = "ready_with_advisories"
        readiness_pass = readiness_status != "not_ready"
        production_ready = readiness_pass and promotion_gate_pass

        advisories: List[str] = []
        if inactive_sweep_params:
            advisories.append(
                "Some sweep dimensions had no observed PnL/Sharpe impact: "
                + ", ".join(inactive_sweep_params)
            )
        if not threshold_recommended:
            reason = (tuning_result or {}).get('reason', 'no_tuning_result')
            advisories.append(f"No threshold recommendation available (reason={reason}).")
        if stress_status == 'advisory':
            sample_note = str((stress_result or {}).get('sample_size_note') or "").strip()
            if sample_note:
                advisories.append(f"Stress test advisory: {sample_note}")
            else:
                advisories.append(
                    "Stress test indicates elevated tail-risk "
                    f"(p_loss={(stress_result or {}).get('prob_total_pnl_le_loss_limit', float('nan')):.2%}, "
                    f"p_dd_breach={(stress_result or {}).get('prob_drawdown_breach', float('nan')):.2%})."
                )
        elif stress_status == 'fail':
            sample_note = str((stress_result or {}).get('sample_size_note') or "").strip()
            if sample_note:
                advisories.append(f"Stress test failed: {sample_note}")
            else:
                advisories.append(
                    "Stress test failed tail-risk thresholds "
                    f"(p_loss={(stress_result or {}).get('prob_total_pnl_le_loss_limit', float('nan')):.2%}, "
                    f"p_dd_breach={(stress_result or {}).get('prob_drawdown_breach', float('nan')):.2%})."
                )
        elif run_stress_test and stress_result is None:
            advisories.append(
                "Stress test could not run due to insufficient trade history for the selected session."
            )
        if not oos_pass:
            advisories.append(f"OOS report card did not pass (grade={oos_grade}).")
        if not promotion_gate_pass:
            advisories.append(
                "Promotion gate not met: require "
                f"OOS grade >= {required_oos_grade} OR >= {promotion_min_live_trades} live trades "
                f"in last {promotion_live_lookback_days} days "
                f"(actual grade={oos_grade}, live_trades={live_trade_count})."
            )
        fill_note = str(forward_summary.get('fill_note') or "").strip()
        if fill_note:
            advisories.append(f"Forward fill quality note: {fill_note}")

        readiness_output_path = Path(readiness_output_dir)
        if not readiness_output_path.is_absolute():
            readiness_output_path = project_root / readiness_output_path
        readiness_output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readiness_md_path = readiness_output_path / f"earnings_readiness_report_{timestamp}.md"
        readiness_json_path = readiness_output_path / f"earnings_readiness_report_{timestamp}.json"

        with readiness_md_path.open("w", encoding="utf-8") as f:
            f.write("# Earnings Strategy Readiness Report\n\n")
            f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"- Readiness pass: `{readiness_pass}`\n")
            f.write(f"- Readiness status: `{readiness_status}`\n")
            f.write(f"- Production-ready: `{production_ready}`\n")
            f.write(f"- OOS pass: `{oos_pass}`\n")
            f.write(f"- Threshold recommended: `{threshold_recommended}`\n")
            f.write(f"- Stress status: `{stress_status}`\n")
            f.write(f"- OOS grade: `{oos_grade}`\n")
            f.write(f"- Promotion gate pass: `{promotion_gate_pass}`\n\n")

            f.write("## Core Metrics\n\n")
            f.write(f"- Sweep rows: `{sweep_result.get('rows', 0)}`\n")
            f.write(f"- OOS splits: `{oos_result.get('splits', 0)}`\n")
            f.write(
                f"- OOS best test session: `{oos_result.get('best_test_session_id', 'n/a')}`\n"
            )
            if tuning_result:
                f.write(f"- Threshold candidates: `{tuning_result.get('candidate_rows', 0)}`\n")
                f.write(f"- Suggested threshold: `{tuning_result.get('best_threshold')}`\n")
            if stress_result:
                f.write(f"- Stress robustness score: `{stress_result.get('robustness_score', float('nan')):.3f}`\n")
                f.write(
                    f"- Stress P(loss<=limit): "
                    f"`{stress_result.get('prob_total_pnl_le_loss_limit', float('nan')):.2%}`\n"
                )
                f.write(
                    f"- Stress P(drawdown breach): "
                    f"`{stress_result.get('prob_drawdown_breach', float('nan')):.2%}`\n"
                )
                sample_note = str(stress_result.get('sample_size_note') or "").strip()
                if sample_note:
                    f.write(f"- Stress sample note: `{sample_note}`\n")
            f.write(
                f"- Forward live trades (last {promotion_live_lookback_days}d): "
                f"`{live_trade_count}`\n"
            )
            f.write(
                f"- Forward max drawdown: "
                f"`${float(forward_summary.get('max_drawdown', 0.0)):.2f}`\n"
            )
            f.write(
                f"- Forward mean execution drag: "
                f"`{float(forward_summary.get('mean_execution_drag_bps', float('nan'))):.2f} bps`\n"
            )
            f.write("\n")

            f.write("## Promotion Gate\n\n")
            f.write(f"- Required minimum OOS grade: `{required_oos_grade}`\n")
            f.write(f"- Grade gate pass: `{grade_gate_pass}`\n")
            f.write(
                f"- Required minimum live trades ({promotion_live_lookback_days}d): "
                f"`{promotion_min_live_trades}`\n"
            )
            f.write(f"- Live trade gate pass: `{live_trade_gate_pass}`\n")
            f.write(f"- Promotion gate pass (grade OR live trades): `{promotion_gate_pass}`\n\n")

            if advisories:
                f.write("## Advisories\n\n")
                for note in advisories:
                    f.write(f"- {note}\n")
                f.write("\n")

            f.write("## Artifact Paths\n\n")
            f.write(f"- Sweep CSV: `{sweep_result.get('csv_path', 'n/a')}`\n")
            f.write(f"- Sweep summary: `{sweep_result.get('markdown_path', 'n/a')}`\n")
            f.write(f"- OOS CSV: `{oos_result.get('csv_path', 'n/a')}`\n")
            f.write(f"- OOS summary: `{oos_result.get('markdown_path', 'n/a')}`\n")
            f.write(f"- OOS report card: `{oos_result.get('report_card_markdown_path', 'n/a')}`\n")
            f.write(
                f"- Threshold tuning summary: "
                f"`{(tuning_result or {}).get('markdown_path', 'n/a')}`\n"
            )
            f.write(
                f"- Stress summary: "
                f"`{(stress_result or {}).get('markdown_path', 'n/a')}`\n"
            )
            f.write(
                f"- Forward paper tracker summary: "
                f"`{(forward_tracker_result or {}).get('markdown_path', 'n/a')}`\n"
            )
            f.write(f"- Readiness JSON: `{readiness_json_path}`\n")

        with readiness_json_path.open("w", encoding="utf-8") as f:
            json.dump({
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'readiness_pass': readiness_pass,
                'readiness_status': readiness_status,
                'production_ready': production_ready,
                'promotion_gate_pass': promotion_gate_pass,
                'oos_pass': oos_pass,
                'threshold_recommended': threshold_recommended,
                'stress_status': stress_status,
                'oos_grade': oos_grade,
                'promotion_gate': {
                    'required_oos_grade': required_oos_grade,
                    'grade_gate_pass': grade_gate_pass,
                    'required_live_trades': promotion_min_live_trades,
                    'live_trade_count': live_trade_count,
                    'live_trade_lookback_days': promotion_live_lookback_days,
                    'live_trade_gate_pass': live_trade_gate_pass,
                    'promotion_gate_pass': promotion_gate_pass,
                },
                'inactive_sweep_params': inactive_sweep_params,
                'advisories': advisories,
                'sweep': sweep_result,
                'oos': oos_result,
                'threshold_tuning': tuning_result,
                'stress_test': stress_result,
                'forward_tracker': forward_tracker_result,
                'artifact_paths': {
                    'readiness_markdown': str(readiness_md_path),
                    'readiness_json': str(readiness_json_path),
                },
            }, f, indent=2, default=str)

        self.logger.info(f"✅ Readiness report saved: {readiness_md_path}")
        self.logger.info(f"✅ Readiness JSON saved: {readiness_json_path}")
        return {
            'readiness_pass': readiness_pass,
            'readiness_status': readiness_status,
            'production_ready': production_ready,
            'promotion_gate_pass': promotion_gate_pass,
            'oos_pass': oos_pass,
            'threshold_recommended': threshold_recommended,
            'stress_status': stress_status,
            'oos_grade': oos_grade,
            'inactive_sweep_params': inactive_sweep_params,
            'advisories': advisories,
            'markdown_path': str(readiness_md_path),
            'json_path': str(readiness_json_path),
            'sweep_result': sweep_result,
            'oos_result': oos_result,
            'tuning_result': tuning_result,
            'stress_result': stress_result,
            'forward_tracker_result': forward_tracker_result,
            'best_test_session_id': oos_result.get('best_test_session_id'),
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
        '--marketdata-token',
        type=str,
        help='Optional MarketData.app token override (otherwise uses MARKETDATA_TOKEN env var)'
    )

    parser.add_argument(
        '--disable-marketdata',
        action='store_true',
        help='Disable MarketData.app usage even if token is configured'
    )

    parser.add_argument(
        '--capture-historical-mda-snapshots',
        action='store_true',
        help='Backfill historical earnings option-IV snapshots from MarketData.app for ML labels'
    )

    parser.add_argument(
        '--mda-lookback-years',
        type=int,
        default=2,
        help='Years of earnings history to pull for MDApp historical snapshots (default: 2)'
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
        '--readiness-report',
        action='store_true',
        help='Run sweep + OOS + threshold tuning and export a consolidated readiness report'
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
        '--sweep-min-trades',
        type=int,
        default=12,
        help='Minimum trades required for a config to be eligible as sweep winner (default: 12)'
    )

    parser.add_argument(
        '--snapshot-lookback-days',
        type=int,
        default=30,
        help='Days back from today to capture earnings snapshots (default: 30)'
    )

    parser.add_argument(
        '--snapshot-lookahead-days',
        type=int,
        default=90,
        help='Days ahead from today to capture earnings snapshots (default: 90)'
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
        '--readiness-output-dir',
        type=str,
        default='exports/reports',
        help='Directory for consolidated readiness markdown/json outputs'
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
        '--stress-test',
        action='store_true',
        help='Run bootstrap Monte Carlo stress test for trade-level tail risk'
    )

    parser.add_argument(
        '--stress-session-id',
        type=str,
        help='Optional session ID for stress test (default: latest/best available)'
    )

    parser.add_argument(
        '--stress-simulations',
        type=int,
        default=5000,
        help='Number of Monte Carlo bootstrap simulations for stress test (default: 5000)'
    )

    parser.add_argument(
        '--stress-confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for VaR/CVaR in stress test [0.5-0.99] (default: 0.95)'
    )

    parser.add_argument(
        '--stress-drawdown-limit',
        type=float,
        default=-600.0,
        help='Drawdown breach threshold in dollars for stress test (default: -600)'
    )

    parser.add_argument(
        '--stress-loss-limit',
        type=float,
        default=0.0,
        help='Loss threshold in dollars for stress probability metric (default: 0)'
    )

    parser.add_argument(
        '--stress-seed',
        type=int,
        default=42,
        help='Random seed for stress simulation reproducibility (default: 42)'
    )

    parser.add_argument(
        '--stress-min-trades',
        type=int,
        default=20,
        help='Minimum trade count considered statistically reliable for stress verdicts (default: 20)'
    )

    parser.add_argument(
        '--stress-output-dir',
        type=str,
        default='exports/reports',
        help='Directory for stress-test CSV/summary outputs'
    )

    parser.add_argument(
        '--disable-readiness-stress-test',
        action='store_true',
        help='Disable automatic stress-test step inside --readiness-report'
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
    collector = InstitutionalDataCollector(
        db_path=args.db_path,
        marketdata_token=args.marketdata_token,
        enable_marketdata=not args.disable_marketdata,
    )

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

            if args.capture_historical_mda_snapshots:
                collector.capture_historical_iv_snapshots_mda(
                    symbols=aux_symbols,
                    lookback_years=args.mda_lookback_years,
                )

            if args.calibrate_iv_decay:
                collector.calibrate_iv_decay_labels(
                    min_pre_days=args.label_min_pre_days,
                    max_pre_days=args.label_max_pre_days,
                    min_post_days=args.label_min_post_days,
                    max_post_days=args.label_max_post_days,
                )

            if (
                args.capture_snapshots
                or args.capture_historical_mda_snapshots
                or args.calibrate_iv_decay
                or args.snapshot_status
            ):
                collector.report_snapshot_pairing_status(
                    min_pre_days=args.label_min_pre_days,
                    max_pre_days=args.label_max_pre_days,
                    min_post_days=args.label_min_post_days,
                    max_post_days=args.label_max_post_days,
                )

            scorecard_requested = bool(args.crush_scorecard)
            analysis_requested = bool(
                args.readiness_report
                or args.parameter_sweep
                or args.oos_validation
                or args.stress_test
                or scorecard_requested
                or args.regime_diagnostics
                or args.threshold_tuning
            )
            snapshot_only_mode = bool(
                (
                    args.capture_snapshots
                    or args.capture_historical_mda_snapshots
                    or args.calibrate_iv_decay
                    or args.snapshot_status
                )
                and not analysis_requested
                and not args.run_sample_backtest
            )
            if args.readiness_report:
                readiness_result = collector.run_readiness_report(
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
                    sweep_output_dir=args.sweep_output_dir,
                    oos_output_dir=args.oos_output_dir,
                    tuning_output_dir=args.tuning_output_dir,
                    readiness_output_dir=args.readiness_output_dir,
                    sweep_top_n=args.sweep_top_n,
                    sweep_min_trades=args.sweep_min_trades,
                    tuning_session_id=args.tuning_session_id,
                    tuning_min_confidence=args.tuning_min_confidence,
                    tuning_min_trades=args.tuning_min_trades,
                    tuning_use_composite_signal=not args.tuning_raw_setup_score,
                    run_stress_test=not args.disable_readiness_stress_test,
                    stress_session_id=args.stress_session_id,
                    stress_simulations=args.stress_simulations,
                    stress_confidence_level=args.stress_confidence_level,
                    stress_drawdown_limit=args.stress_drawdown_limit,
                    stress_loss_limit=args.stress_loss_limit,
                    stress_seed=args.stress_seed,
                    stress_min_trades=args.stress_min_trades,
                    stress_output_dir=args.stress_output_dir,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )
                if readiness_result:
                    logger.info(
                        "✅ Readiness workflow completed: "
                        f"pass={readiness_result.get('readiness_pass', False)}, "
                        f"oos_grade={readiness_result.get('oos_grade', 'n/a')}"
                    )
                    latest_session_id = str(
                        readiness_result.get('best_test_session_id') or latest_session_id or ""
                    )
                else:
                    logger.warning("⚠️ Readiness workflow did not produce results")
            elif args.oos_validation:
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
                    min_trades_for_ranking=args.sweep_min_trades,
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

            if args.threshold_tuning and not args.readiness_report:
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

            if args.stress_test and not args.readiness_report:
                stress_result = collector.run_trade_stress_test(
                    session_id=args.stress_session_id or latest_session_id,
                    simulations=args.stress_simulations,
                    confidence_level=args.stress_confidence_level,
                    drawdown_limit=args.stress_drawdown_limit,
                    loss_limit=args.stress_loss_limit,
                    seed=args.stress_seed,
                    min_reliable_trades=args.stress_min_trades,
                    output_dir=args.stress_output_dir,
                )
                if stress_result:
                    logger.info(
                        "✅ Stress test completed: "
                        f"status={stress_result.get('status')}, "
                        f"robustness={stress_result.get('robustness_score', float('nan')):.3f}"
                    )
                else:
                    logger.warning("⚠️ Stress test did not produce results")
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
            batch_size=args.batch_size,
            capture_historical_mda_snapshots=args.capture_historical_mda_snapshots,
            mda_lookback_years=args.mda_lookback_years,
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

            if args.readiness_report:
                logger.info("🧪 Running post-backfill consolidated readiness workflow...")
                readiness_result = collector.run_readiness_report(
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
                    sweep_output_dir=args.sweep_output_dir,
                    oos_output_dir=args.oos_output_dir,
                    tuning_output_dir=args.tuning_output_dir,
                    readiness_output_dir=args.readiness_output_dir,
                    sweep_top_n=args.sweep_top_n,
                    sweep_min_trades=args.sweep_min_trades,
                    tuning_session_id=args.tuning_session_id,
                    tuning_min_confidence=args.tuning_min_confidence,
                    tuning_min_trades=args.tuning_min_trades,
                    tuning_use_composite_signal=not args.tuning_raw_setup_score,
                    run_stress_test=not args.disable_readiness_stress_test,
                    stress_session_id=args.stress_session_id,
                    stress_simulations=args.stress_simulations,
                    stress_confidence_level=args.stress_confidence_level,
                    stress_drawdown_limit=args.stress_drawdown_limit,
                    stress_loss_limit=args.stress_loss_limit,
                    stress_seed=args.stress_seed,
                    stress_min_trades=args.stress_min_trades,
                    stress_output_dir=args.stress_output_dir,
                    start_date=args.backtest_start_date,
                    end_date=args.backtest_end_date,
                )
                if readiness_result:
                    logger.info(
                        "✅ Readiness workflow finished "
                        f"(pass={readiness_result.get('readiness_pass', False)}, "
                        f"oos_grade={readiness_result.get('oos_grade', 'n/a')})"
                    )
                    latest_session_id = str(
                        readiness_result.get('best_test_session_id') or latest_session_id or ""
                    )
                else:
                    logger.warning("⚠️ Readiness workflow did not produce results")
            elif args.oos_validation:
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
                    min_trades_for_ranking=args.sweep_min_trades,
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

            if args.threshold_tuning and not args.readiness_report:
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

            if args.stress_test and not args.readiness_report:
                logger.info("🧪 Running trade-level stress test...")
                stress_result = collector.run_trade_stress_test(
                    session_id=args.stress_session_id or latest_session_id,
                    simulations=args.stress_simulations,
                    confidence_level=args.stress_confidence_level,
                    drawdown_limit=args.stress_drawdown_limit,
                    loss_limit=args.stress_loss_limit,
                    seed=args.stress_seed,
                    min_reliable_trades=args.stress_min_trades,
                    output_dir=args.stress_output_dir,
                )
                if stress_result:
                    logger.info(
                        "✅ Stress test finished: "
                        f"status={stress_result.get('status')}, "
                        f"robustness={stress_result.get('robustness_score', float('nan')):.3f}"
                    )
                else:
                    logger.warning("⚠️ Stress test did not produce results")

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
