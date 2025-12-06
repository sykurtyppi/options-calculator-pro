
# Scanner worker for background stock scanning operations
# Professional Options Calculator - Scanner Worker

import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging

from .base_worker import BaseWorker
from .analysis_worker import AnalysisWorker

logger = logging.getLogger(__name__)


class ScannerWorker(BaseWorker):
    def __init__(self, symbols: List[str], scan_criteria: Dict[str, Any], contracts: int = 1, callback: Optional[Callable] = None):
        super().__init__()
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.scan_criteria = scan_criteria
        self.contracts = contracts
        self.callback = callback
        self.opportunities = []
        self.failed_symbols = []
        self.scan_stats = {
            'total_scanned': 0,
            'opportunities_found': 0,
            'failed_analyses': 0,
            'scan_start_time': None,
            'scan_end_time': None
        }

    def run(self):
        try:
            self.started.emit()
            self.scan_stats['scan_start_time'] = datetime.now()
            self.emit_status("Starting options scanner...")

            if not self.symbols:
                self.emit_error("Invalid Input", "No symbols provided for scanning")
                return

            total_symbols = len(self.symbols)
            self.emit_status(f"Scanning {total_symbols} symbols for opportunities...")

            min_confidence = self.scan_criteria.get('min_confidence', 50.0)
            max_iv_rv_ratio = self.scan_criteria.get('max_iv_rv_ratio', 2.0)
            min_days_to_earnings = self.scan_criteria.get('min_days_to_earnings', 1)
            max_days_to_earnings = self.scan_criteria.get('max_days_to_earnings', 14)
            min_volume = self.scan_criteria.get('min_volume', 50000000)

            for i, symbol in enumerate(self.symbols):
                if self.is_cancelled():
                    self.emit_status("Scanner cancelled by user")
                    break

                try:
                    progress = int((i / total_symbols) * 90)
                    self.emit_progress(progress, f"Scanning {symbol}...")

                    analysis_result = self._analyze_symbol(symbol)
                    self.scan_stats['total_scanned'] += 1

                    if analysis_result:
                        if self._meets_scan_criteria(analysis_result, min_confidence, max_iv_rv_ratio, min_days_to_earnings, max_days_to_earnings, min_volume):
                            opportunity = self._create_opportunity_record(symbol, analysis_result)
                            self.opportunities.append(opportunity)
                            self.scan_stats['opportunities_found'] += 1

                            self.emit_result({
                                'type': 'opportunity_found',
                                'symbol': symbol,
                                'opportunity': opportunity
                            })
                            self.emit_status(f"{symbol}: OPPORTUNITY FOUND! Confidence: {analysis_result.get('confidence', 0):.1f}%")
                        else:
                            self.emit_status(f"{symbol}: No opportunity (criteria not met)")
                    else:
                        self.failed_symbols.append(symbol)
                        self.scan_stats['failed_analyses'] += 1
                        self.emit_status(f"{symbol}: Analysis failed")

                except Exception as e:
                    self.failed_symbols.append(symbol)
                    self.scan_stats['failed_analyses'] += 1
                    logger.error(f"Error scanning {symbol}: {e}")
                    self.emit_status(f"{symbol}: Error - {str(e)}")

                if not self.is_cancelled():
                    time.sleep(0.2)

            self.emit_progress(95, "Finalizing scan results...")
            self._finalize_scan_results()
            self.opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            self.scan_stats['scan_end_time'] = datetime.now()
            duration = (self.scan_stats['scan_end_time'] - self.scan_stats['scan_start_time']).total_seconds()

            final_result = {
                'type': 'scan_complete',
                'opportunities': self.opportunities,
                'failed_symbols': self.failed_symbols,
                'stats': self.scan_stats,
                'criteria': self.scan_criteria,
                'scan_duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }

            self.emit_result(final_result)
            self.emit_progress(100, f"Scan complete: {len(self.opportunities)} opportunities found")

        except Exception as e:
            logger.error(f"Critical error in scanner worker: {e}")
            self.emit_error("Scanner Failed", f"Critical error: {str(e)}")
        finally:
            self.finished.emit()

    def _analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            worker = AnalysisWorker([symbol], self.contracts)
            return worker._analyze_symbol(symbol)
        except Exception as e:
            logger.error(f"Error analyzing {symbol} in scanner: {e}")
            return None

    def _meets_scan_criteria(self, analysis_result: Dict[str, Any], min_confidence: float,
                             max_iv_rv_ratio: float, min_days_to_earnings: int,
                             max_days_to_earnings: int, min_volume: float) -> bool:
        try:
            confidence = analysis_result.get('confidence', 0)
            vol_metrics = analysis_result.get('volatility_metrics', {})
            iv_rv_ratio = vol_metrics.get('iv_rv_ratio', 999)
            earnings_data = analysis_result.get('earnings_data', {})
            days_to_earnings = earnings_data.get('days_to_earnings', 999)

            volume_check = True  # Placeholder
            return all([
                confidence >= min_confidence,
                iv_rv_ratio <= max_iv_rv_ratio,
                min_days_to_earnings <= days_to_earnings <= max_days_to_earnings,
                volume_check
            ])
        except Exception as e:
            logger.error(f"Error checking scan criteria: {e}")
            return False

    def _create_opportunity_record(self, symbol: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            vol_metrics = analysis_result.get('volatility_metrics', {})
            earnings_data = analysis_result.get('earnings_data', {})
            market_data = analysis_result.get('market_data', {})

            return {
                'symbol': symbol,
                'confidence': analysis_result.get('confidence', 0),
                'underlying_price': analysis_result.get('underlying_price', 0),
                'iv_rv_ratio': vol_metrics.get('iv_rv_ratio', 0),
                'days_to_earnings': earnings_data.get('days_to_earnings', 0),
                'earnings_date': earnings_data.get('earnings_date'),
                'expected_move': analysis_result.get('expected_move', 'N/A'),
                'max_loss': analysis_result.get('max_loss', 'N/A'),
                'recommendation': analysis_result.get('recommendation', 'N/A'),
                'vix': market_data.get('vix', 0),
                'sector': 'Unknown',
                'scan_timestamp': datetime.now().isoformat(),
                'full_analysis': analysis_result
            }
        except Exception as e:
            logger.error(f"Error creating opportunity record for {symbol}: {e}")
            return {
                'symbol': symbol,
                'confidence': 0,
                'error': str(e)
            }

    def _finalize_scan_results(self):
        try:
            if self.opportunities:
                confidences = [opp['confidence'] for opp in self.opportunities]
                self.scan_stats['avg_confidence'] = sum(confidences) / len(confidences)
                self.scan_stats['max_confidence'] = max(confidences)
                self.scan_stats['min_confidence'] = min(confidences)
            else:
                self.scan_stats['avg_confidence'] = 0
                self.scan_stats['max_confidence'] = 0
                self.scan_stats['min_confidence'] = 0

            total_attempted = self.scan_stats['total_scanned'] + self.scan_stats['failed_analyses']
            if total_attempted > 0:
                self.scan_stats['success_rate'] = (self.scan_stats['total_scanned'] / total_attempted) * 100
            else:
                self.scan_stats['success_rate'] = 0

            if self.scan_stats['total_scanned'] > 0:
                self.scan_stats['opportunity_rate'] = (self.scan_stats['opportunities_found'] / self.scan_stats['total_scanned']) * 100
            else:
                self.scan_stats['opportunity_rate'] = 0
        except Exception as e:
            logger.error(f"Error finalizing scan results: {e}")
