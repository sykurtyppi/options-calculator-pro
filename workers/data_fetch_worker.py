"""
DataFetchWorker - Live Data Integration Worker
===========================================

Handles live data fetching for Calendar Spreads analysis using existing services.
Integrates with AnalysisController to provide threaded data operations.

Part of Professional Options Calculator v10.0
Optimized for live market data integration
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QApplication

from controllers.analysis_controller import AnalysisController
from services.market_data import MarketDataService
from services.options_service import OptionsService
from services.volatility_service import VolatilityService
from services.ml_service import MLService
from utils.greeks_calculator import GreeksCalculator
from utils.thread_manager import ThreadManager, TaskPriority, WorkerType
from utils.logger import get_logger

logger = get_logger(__name__)

class DataFetchWorker(QObject):
    """
    Live data fetching worker that integrates with existing services.
    Provides threaded data operations for Calendar Spreads analysis.
    """

    # Signals for GUI communication
    analysis_started = Signal(str)  # symbol
    analysis_progress = Signal(str, int, str)  # symbol, progress, status
    analysis_completed = Signal(str, dict)  # symbol, results_data
    analysis_error = Signal(str, str)  # symbol, error_message
    connection_status_changed = Signal(bool, str)  # is_connected, status_message
    market_data_updated = Signal(str, dict)  # symbol, market_data

    def __init__(self,
                 market_data_service: MarketDataService,
                 options_service: OptionsService,
                 volatility_service: VolatilityService,
                 ml_service: MLService,
                 greeks_calculator: GreeksCalculator,
                 thread_manager: ThreadManager,
                 parent=None):
        super().__init__(parent)

        self.logger = logger

        # Store service references
        self.market_data_service = market_data_service
        self.options_service = options_service
        self.volatility_service = volatility_service
        self.ml_service = ml_service
        self.greeks_calculator = greeks_calculator
        self.thread_manager = thread_manager

        # Initialize analysis controller for comprehensive analysis
        # We'll get config_manager from one of the services
        config_manager = (
            getattr(ml_service, 'config', None)
            or getattr(ml_service, 'config_manager', None)
            or getattr(market_data_service, 'config_manager', None)
        )

        self.analysis_controller = AnalysisController(
            market_data_service=market_data_service,
            config_manager=config_manager,
            ml_service=ml_service,
            thread_manager=thread_manager,
            parent=self
        )

        # Connect analysis controller signals
        self.analysis_controller.analysis_started.connect(self.analysis_started.emit)
        self.analysis_controller.analysis_progress.connect(self.analysis_progress.emit)
        self.analysis_controller.analysis_completed.connect(self._on_analysis_completed)
        self.analysis_controller.analysis_error.connect(self.analysis_error.emit)

        self.logger.info("DataFetchWorker initialized with live services")

    def set_config_manager(self, config_manager):
        """Set config manager after initialization"""
        self.analysis_controller.config_manager = config_manager
        self.analysis_controller.volatility_service = VolatilityService(config_manager, self.market_data_service)

    def analyze_calendar_spread(self, symbol: str, contracts: int = 1, debit_override: Optional[float] = None):
        """
        Start live calendar spread analysis for a symbol

        Args:
            symbol: Stock ticker symbol
            contracts: Number of contracts (default: 1)
            debit_override: Manual debit override (optional)
        """
        try:
            symbol = symbol.strip().upper()
            if not symbol:
                self.analysis_error.emit("", "Empty symbol provided")
                return

            self.logger.info(f"Starting calendar spread analysis for {symbol}")

            # Check market data connectivity first
            self._check_connectivity()

            # Use the analysis controller for comprehensive analysis
            request_id = self.analysis_controller.analyze_symbol(
                symbol=symbol,
                contracts=contracts,
                debit_override=debit_override,
                parameters={
                    'analysis_type': 'calendar_spread',
                    'use_ml': True,
                    'monte_carlo_sims': 10000
                }
            )

            if request_id:
                self.logger.info(f"Calendar spread analysis started with request ID: {request_id}")
            else:
                self.analysis_error.emit(symbol, "Failed to start analysis")

        except Exception as e:
            error_msg = f"Error starting calendar spread analysis: {e}"
            self.logger.error(error_msg)
            self.analysis_error.emit(symbol if 'symbol' in locals() else "", error_msg)

    def fetch_market_data(self, symbol: str):
        """
        Fetch live market data for a symbol in background thread

        Args:
            symbol: Stock ticker symbol
        """
        def fetch_data():
            """Background task to fetch market data"""
            try:
                # Get current price
                current_price = self.market_data_service.get_current_price(symbol)
                if not current_price:
                    raise ValueError(f"Unable to get current price for {symbol}")

                # Get historical data
                historical_data = self.market_data_service.get_historical_data(symbol, period="3mo")
                if historical_data.empty:
                    raise ValueError(f"Unable to get historical data for {symbol}")

                # Get basic volatility info
                returns = historical_data['Close'].pct_change().dropna()
                historical_vol = returns.std() * (252 ** 0.5)  # Annualized volatility

                market_data = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'historical_vol': historical_vol,
                    'last_close': historical_data['Close'].iloc[-1],
                    'volume': historical_data['Volume'].iloc[-1],
                    'timestamp': datetime.now().isoformat()
                }

                return market_data

            except Exception as e:
                self.logger.error(f"Error fetching market data for {symbol}: {e}")
                raise

        # Submit to thread manager
        task_id = self.thread_manager.submit_task(
            function=fetch_data,
            name=f"market_data_{symbol}",
            worker_type=WorkerType.DATA_FETCH,
            priority=TaskPriority.HIGH,
            callback=lambda data: self.market_data_updated.emit(symbol, data),
            error_callback=lambda error: self.analysis_error.emit(symbol, str(error))
        )

        self.logger.debug(f"Market data fetch task submitted: {task_id}")

    def check_options_availability(self, symbol: str):
        """
        Check if options are available for a symbol in background thread

        Args:
            symbol: Stock ticker symbol
        """
        def check_options():
            """Background task to check options availability"""
            try:
                expirations = self.options_service.get_available_expirations(symbol, timeout=10)

                if expirations:
                    # Filter for calendar spread suitable expirations
                    suitable_expirations = self.options_service.filter_expirations_for_calendar(expirations)

                    return {
                        'symbol': symbol,
                        'has_options': True,
                        'total_expirations': len(expirations),
                        'suitable_expirations': len(suitable_expirations),
                        'next_expirations': expirations[:3],  # First 3 expirations
                        'calendar_ready': len(suitable_expirations) >= 2
                    }
                else:
                    return {
                        'symbol': symbol,
                        'has_options': False,
                        'total_expirations': 0,
                        'suitable_expirations': 0,
                        'next_expirations': [],
                        'calendar_ready': False
                    }

            except Exception as e:
                self.logger.error(f"Error checking options for {symbol}: {e}")
                raise

        # Submit to thread manager
        task_id = self.thread_manager.submit_task(
            function=check_options,
            name=f"options_check_{symbol}",
            worker_type=WorkerType.DATA_FETCH,
            priority=TaskPriority.NORMAL,
            callback=lambda data: self.market_data_updated.emit(symbol, data),
            error_callback=lambda error: self.analysis_error.emit(symbol, str(error))
        )

        self.logger.debug(f"Options check task submitted: {task_id}")

    def _check_connectivity(self):
        """Check market data connectivity and emit status"""
        def check_connection():
            """Background task to check connectivity"""
            try:
                # Test connectivity with a known symbol
                test_price = self.market_data_service.get_current_price("AAPL")

                if test_price and test_price > 0:
                    return {'connected': True, 'status': 'Market data connected'}
                else:
                    return {'connected': False, 'status': 'Market data unavailable'}

            except Exception as e:
                return {'connected': False, 'status': f'Connection error: {str(e)}'}

        # Submit to thread manager
        task_id = self.thread_manager.submit_task(
            function=check_connection,
            name="connectivity_check",
            worker_type=WorkerType.DATA_FETCH,
            priority=TaskPriority.HIGH,
            callback=lambda data: self.connection_status_changed.emit(data['connected'], data['status']),
            error_callback=lambda error: self.connection_status_changed.emit(False, f"Connectivity check failed: {error}")
        )

        self.logger.debug(f"Connectivity check task submitted: {task_id}")

    @Slot(str, object)
    def _on_analysis_completed(self, symbol: str, analysis_result):
        """
        Handle analysis completion and convert to calendar spread data

        Args:
            symbol: Stock ticker symbol
            analysis_result: AnalysisResult object from analysis controller
        """
        try:
            # Convert AnalysisResult to calendar spread display data
            calendar_data = self._convert_to_calendar_data(analysis_result)

            # Emit the converted data
            self.analysis_completed.emit(symbol, calendar_data)

            self.logger.info(f"Calendar spread analysis completed for {symbol}")

        except Exception as e:
            error_msg = f"Error processing analysis results for {symbol}: {e}"
            self.logger.error(error_msg)
            self.analysis_error.emit(symbol, error_msg)

    def _convert_to_calendar_data(self, analysis_result) -> Dict[str, Any]:
        """
        Convert AnalysisResult to calendar spread display format

        Args:
            analysis_result: AnalysisResult object

        Returns:
            Dictionary with calendar spread data for GUI display
        """
        try:
            def _get_value(obj, key: str, default=None):
                if obj is None:
                    return default
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            # Extract key data from analysis result
            symbol = analysis_result.symbol
            current_price = analysis_result.current_price
            confidence = analysis_result.confidence_score
            volatility = analysis_result.volatility_metrics
            options_data = analysis_result.options_data
            greeks = analysis_result.greeks
            risk_metrics = analysis_result.risk_metrics
            monte_carlo = analysis_result.monte_carlo_result

            confidence_score = _get_value(confidence, 'overall_score', 50.0)
            confidence_factors = _get_value(confidence, 'factors', []) or []
            edge_grade = "C"
            for factor in confidence_factors:
                if isinstance(factor, (list, tuple)) and factor and str(factor[0]).startswith("Edge Grade"):
                    edge_grade = str(factor[0]).replace("Edge Grade", "").strip() or "C"
                    break
            rationale_lines = []
            for factor in confidence_factors:
                if not isinstance(factor, (list, tuple)) or len(factor) < 2:
                    continue
                label = str(factor[0])
                points = float(factor[1])
                if points == 0:
                    continue
                rationale_lines.append(f"{label}: {points:+.1f}")

            # Build calendar spread specific data
            calendar_data = {
                # Basic info
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': analysis_result.timestamp.isoformat(),
                'analysis_type': _get_value(analysis_result, 'analysis_type', 'calendar_spread'),
                'source': _get_value(analysis_result, 'source', 'analysis_controller'),

                # Calendar spread metrics
                'net_debit': risk_metrics.debit_paid if risk_metrics else 0.0,
                'max_profit': risk_metrics.max_profit_per_contract if risk_metrics else 0.0,
                'max_loss': risk_metrics.max_loss_per_contract if risk_metrics else 0.0,
                'prob_profit': risk_metrics.probability_of_profit * 100 if risk_metrics else 50.0,
                'transaction_cost': _get_value(risk_metrics, 'transaction_cost_per_contract', 0.0) if risk_metrics else 0.0,

                # Strike and expiration info
                'recommended_strike': _get_value(options_data, 'strike', current_price),
                'short_expiration': _get_value(options_data, 'short_expiration', _get_value(options_data, 'short_expiry', 'N/A')),
                'long_expiration': _get_value(options_data, 'long_expiration', _get_value(options_data, 'long_expiry', 'N/A')),
                'days_to_expiration': _get_value(options_data, 'days_to_expiration', 30),

                # Greeks
                'delta': _get_value(greeks, 'net_delta', 0.0),
                'gamma': _get_value(greeks, 'net_gamma', 0.0),
                'theta': _get_value(greeks, 'net_theta', 0.0),
                'vega': _get_value(greeks, 'net_vega', 0.0),

                # Volatility metrics
                'implied_vol': _get_value(volatility, 'implied_volatility', _get_value(volatility, 'iv30', 0.0)),
                'historical_vol': _get_value(volatility, 'historical_volatility', _get_value(volatility, 'rv30', 0.0)),
                'iv_rank': _get_value(volatility, 'iv_rank', 0.5),
                'term_structure_slope': _get_value(volatility, 'term_structure_slope', 0.0),
                'event_iv_premium': _get_value(options_data, 'event_iv_premium', 0.0),
                'implied_move_pct': _get_value(options_data, 'implied_move_pct', 0.0),
                'historical_move_pct': _get_value(_get_value(options_data, 'earnings_move_profile', {}), 'median_abs_move_pct', 0.0),
                'liquidity_score': _get_value(options_data, 'liquidity_score', 0.0) * 100.0,

                # Confidence and recommendation
                'confidence_score': confidence_score,
                'recommendation': _get_value(confidence, 'recommendation', 'ANALYZE MANUALLY'),
                'risk_level': _get_value(confidence, 'risk_level', 'MODERATE'),
                'edge_grade': edge_grade,
                'confidence_factors': confidence_factors,
                'edge_rationale': rationale_lines[:6],

                # Monte Carlo results
                'prob_50_profit': _get_value(monte_carlo, 'prob_exceed_50pct', _get_value(monte_carlo, 'upside_probability', 50.0)),
                'prob_100_profit': _get_value(monte_carlo, 'prob_exceed_1x', 30.0),
                'expected_move': _get_value(monte_carlo, 'expected_move', current_price * 0.05),

                # Analysis metadata
                'analysis_duration': analysis_result.analysis_duration,
                'ml_prediction': analysis_result.ml_prediction,
                'earnings_date': analysis_result.earnings_date,
                'days_to_earnings': analysis_result.days_to_earnings,

                # Status
                'status': 'completed',
                'error': None
            }

            return calendar_data

        except Exception as e:
            self.logger.error(f"Error converting analysis result: {e}")
            return {
                'symbol': analysis_result.symbol if hasattr(analysis_result, 'symbol') else 'Unknown',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def cancel_all_tasks(self):
        """Cancel all active data fetch tasks"""
        try:
            cancelled_count = self.thread_manager.cancel_all_tasks(WorkerType.DATA_FETCH)
            self.logger.info(f"Cancelled {cancelled_count} data fetch tasks")
            return cancelled_count
        except Exception as e:
            self.logger.error(f"Error cancelling data fetch tasks: {e}")
            return 0

    def get_status(self) -> Dict[str, Any]:
        """Get current worker status"""
        try:
            active_tasks = self.thread_manager.get_active_tasks(WorkerType.DATA_FETCH)

            return {
                'active_tasks': len(active_tasks),
                'thread_manager_stats': self.thread_manager.get_performance_stats(),
                'analysis_controller_stats': self.analysis_controller.get_statistics(),
                'services_available': {
                    'market_data': self.market_data_service is not None,
                    'options': self.options_service is not None,
                    'volatility': self.volatility_service is not None,
                    'ml': self.ml_service is not None,
                    'greeks': self.greeks_calculator is not None
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting worker status: {e}")
            return {'error': str(e)}
