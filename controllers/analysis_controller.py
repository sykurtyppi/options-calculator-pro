"""
Analysis Controller - Professional Options Calculator Pro
Handles all analysis business logic and coordinates between services
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal, QTimer, Slot

from services.market_data import MarketDataService
from services.ml_service import MLService
from services.options_service import OptionsService, OptionType as ServiceOptionType
from services.volatility_service import VolatilityService
from services.execution_cost_model import ExecutionCostModel
from models.analysis_result import AnalysisResult, ConfidenceScore, RiskMetrics
from utils.thread_manager import ThreadManager, TaskPriority
from utils.monte_carlo import MonteCarloEngine
from utils.greeks_calculator import GreeksCalculator
from utils.config_manager import ConfigManager


@dataclass
class AnalysisRequest:
    """Analysis request container"""
    symbol: str
    contracts: int = 1
    debit_override: Optional[float] = None
    parameters: Dict[str, Any] = None
    request_id: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not self.request_id:
            self.request_id = f"{self.symbol}_{int(self.timestamp.timestamp())}"


class AnalysisController(QObject):
    """
    Professional analysis controller that orchestrates the entire analysis process
    """

    # Signals for UI communication
    analysis_started = Signal(str)  # symbol
    analysis_progress = Signal(str, int, str)  # symbol, progress, status
    analysis_completed = Signal(str, object)  # symbol, AnalysisResult
    analysis_error = Signal(str, str)  # symbol, error_message
    batch_progress = Signal(int, int, str)  # completed, total, current_symbol

    def __init__(self, market_data_service: MarketDataService, config_manager: ConfigManager, ml_service: MLService, thread_manager: ThreadManager, parent=None):
        # CRITICAL FIX: Initialize QObject parent class first
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)

        # Services (use passed parameters, don't recreate ConfigManager)
        self.market_data_service = market_data_service
        self.config_manager = config_manager  # Don't create new instance
        self.ml_service = ml_service
        self.thread_manager = thread_manager

        # Additional services
        self.options_service = OptionsService(config_manager, market_data_service)
        self.volatility_service = VolatilityService(config_manager, market_data_service)

        # Analysis engines
        self.monte_carlo_engine = MonteCarloEngine()
        self.greeks_calculator = GreeksCalculator(self.config_manager)
        try:
            execution_profile = str(
                self.config_manager.get("trading.execution_profile", "institutional")
            ).strip().lower()
        except Exception:
            execution_profile = "institutional"
        self.execution_cost_model = ExecutionCostModel(execution_profile)

        # Active analysis tracking
        self.active_analyses: Dict[str, AnalysisRequest] = {}
        self.analysis_cache: Dict[str, AnalysisResult] = {}

        # Performance monitoring
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_duration': 0.0,
            'cache_hits': 0
        }

        # Cache cleanup timer
        self.cache_cleanup_timer = QTimer()
        self.cache_cleanup_timer.timeout.connect(self._cleanup_cache)
        self.cache_cleanup_timer.start(300000)  # 5 minutes

        self.logger.info("AnalysisController initialized with proper QObject inheritance")

    def analyze_symbol(self, symbol: str, contracts: int = 1,
                      debit_override: Optional[float] = None,
                      parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Start analysis for a single symbol

        Args:
            symbol: Stock symbol to analyze
            contracts: Number of contracts
            debit_override: Manual debit override
            parameters: Additional analysis parameters

        Returns:
            Request ID for tracking
        """
        try:
            # Validate symbol
            symbol = symbol.strip().upper()
            if not self._validate_symbol(symbol):
                error_msg = f"Invalid symbol: {symbol}"
                self._emit_error_signal(symbol, error_msg)
                return ""

            # Create analysis request
            request = AnalysisRequest(
                symbol=symbol,
                contracts=contracts,
                debit_override=debit_override,
                parameters=parameters or {}
            )

            # Check cache first
            cache_key = self._get_cache_key(request)
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]

                # Check if cache is still valid (15 minutes)
                if (datetime.now() - cached_result.timestamp).total_seconds() < 900:
                    self.logger.debug(f"Using cached result for {symbol}")
                    self.analysis_stats['cache_hits'] += 1

                    # Emit cached result
                    self._emit_completion_signal(symbol, cached_result)
                    return request.request_id

            # Add to active analyses
            self.active_analyses[request.request_id] = request

            # Emit analysis started
            self._emit_started_signal(symbol)

            # CRITICAL FIX: Get available expirations on main thread first
            try:
                self.logger.info(f"Fetching expirations for {symbol} on main thread...")
                available_expirations = self.options_service.get_available_expirations(symbol)
                if available_expirations:
                    # Store expirations in request parameters for background thread
                    request.parameters['available_expirations'] = available_expirations
                    self.logger.info(f"Found {len(available_expirations)} expirations for {symbol}: {available_expirations[:3]}...")
                else:
                    # Try with a fallback - use common monthly expirations
                    from datetime import datetime, timedelta
                    import calendar

                    # Generate next 3 monthly option expirations (3rd Friday of month)
                    today = datetime.now().date()
                    fallback_expirations = []
                    for i in range(1, 4):
                        # Get first day of target month
                        if today.month + i <= 12:
                            target_month = today.month + i
                            target_year = today.year
                        else:
                            target_month = (today.month + i) % 12
                            if target_month == 0:
                                target_month = 12
                            target_year = today.year + 1

                        # Find 3rd Friday
                        first_day = datetime(target_year, target_month, 1).date()
                        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
                        third_friday = first_friday + timedelta(days=14)
                        fallback_expirations.append(third_friday.strftime('%Y-%m-%d'))

                    request.parameters['available_expirations'] = fallback_expirations
                    self.logger.warning(f"No expirations from service, using fallback: {fallback_expirations}")

            except Exception as exp_error:
                self.logger.error(f"Failed to get expirations for {symbol}: {exp_error}")
                # Use fallback dates as above
                from datetime import datetime, timedelta
                today = datetime.now().date()
                fallback_expirations = []
                for weeks in [1, 2, 4]:  # 1 week, 2 weeks, 1 month
                    exp_date = today + timedelta(weeks=weeks)
                    # Adjust to Friday if not already
                    while exp_date.weekday() != 4:  # 4 = Friday
                        exp_date += timedelta(days=1)
                    fallback_expirations.append(exp_date.strftime('%Y-%m-%d'))

                request.parameters['available_expirations'] = fallback_expirations
                self.logger.warning(f"Exception getting expirations, using fallback: {fallback_expirations}")

            # Submit analysis task to thread manager with thread-safe callbacks
            task_id = self.thread_manager.submit_task(
                function=self._perform_analysis,
                args=(request,),
                priority=TaskPriority.HIGH,
                callback=self._create_safe_completion_callback(request.request_id),
                error_callback=self._create_safe_error_callback(request.request_id)
            )

            self.logger.info(f"Started analysis for {symbol} (request: {request.request_id}, task: {task_id})")
            return request.request_id

        except Exception as e:
            self.logger.error(f"Error starting analysis for {symbol}: {e}")
            self._emit_error_signal(symbol, str(e))
            return ""

    def analyze_batch(self, symbols: List[str], parameters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Start batch analysis for multiple symbols

        Args:
            symbols: List of stock symbols
            parameters: Analysis parameters

        Returns:
            List of request IDs
        """
        try:
            # Validate symbols
            valid_symbols = [s.strip().upper() for s in symbols if self._validate_symbol(s.strip())]

            if not valid_symbols:
                self.analysis_error.emit("BATCH", "No valid symbols provided")
                return []

            # Submit individual analyses
            request_ids = []
            for i, symbol in enumerate(valid_symbols):
                try:
                    request_id = self.analyze_symbol(
                        symbol=symbol,
                        contracts=parameters.get('contracts', 1) if parameters else 1,
                        debit_override=parameters.get('debit_override') if parameters else None,
                        parameters=parameters
                    )

                    if request_id:
                        request_ids.append(request_id)

                    # Emit batch progress
                    self.batch_progress.emit(i, len(valid_symbols), symbol)

                except Exception as e:
                    self.logger.error(f"Error starting analysis for {symbol} in batch: {e}")
                    continue

            self.logger.info(f"Started batch analysis for {len(request_ids)} symbols")
            return request_ids

        except Exception as e:
            self.logger.error(f"Error in batch analysis: {e}")
            self.analysis_error.emit("BATCH", str(e))
            return []

    def cancel_analysis(self, request_id: str) -> bool:
        """Cancel an active analysis"""
        try:
            if request_id in self.active_analyses:
                # Remove from active analyses
                request = self.active_analyses.pop(request_id)

                # Try to cancel the task (may not be possible if already running)
                self.thread_manager.cancel_task(request_id)

                self.logger.info(f"Cancelled analysis for {request.symbol}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error cancelling analysis {request_id}: {e}")
            return False

    def get_analysis_status(self, request_id: str) -> Optional[str]:
        """Get status of an analysis request"""
        if request_id in self.active_analyses:
            return "running"
        elif any(request_id in result.request_id for result in self.analysis_cache.values()):
            return "completed"
        else:
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis controller statistics"""
        stats = self.analysis_stats.copy()
        stats.update({
            'active_analyses': len(self.active_analyses),
            'cached_results': len(self.analysis_cache),
            'thread_manager_stats': self.thread_manager.get_statistics()
        })
        return stats

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or len(symbol) > 10:
            return False

        # Basic validation - letters, numbers, dots, hyphens
        import re
        return bool(re.match(r'^[A-Z0-9.\-]+$', symbol))

    def _get_cache_key(self, request: AnalysisRequest) -> str:
        """Generate cache key for analysis request"""
        params_str = "_".join(f"{k}:{v}" for k, v in sorted(request.parameters.items()))
        return f"{request.symbol}_{request.contracts}_{request.debit_override}_{hash(params_str)}"

    def _perform_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform the actual analysis (runs in background thread)

        This is the main analysis engine that orchestrates all calculations
        """
        start_time = datetime.now()
        symbol = request.symbol

        try:
            self.logger.info(f"Starting comprehensive analysis for {symbol}")

            # Step 1: Get current market data (10% progress)
            self._emit_progress(symbol, 10, "Fetching market data...")
            current_price = self.market_data_service.get_current_price(symbol)
            if not current_price or current_price <= 0:
                raise ValueError(f"Unable to get valid price for {symbol}")

            # Step 2: Get historical data (20% progress)
            self._emit_progress(symbol, 20, "Analyzing historical data...")
            historical_data = self.market_data_service.get_historical_data(symbol, period="6mo")
            if historical_data.empty:
                raise ValueError(f"Unable to get historical data for {symbol}")

            # Step 3: Get earnings information (30% progress)
            self._emit_progress(symbol, 30, "Checking earnings calendar...")
            earnings_date = self.market_data_service.get_next_earnings(symbol)
            days_to_earnings = self._calculate_days_to_earnings(earnings_date)
            earnings_move_profile = self._estimate_historical_earnings_move(symbol, historical_data)

            # Step 4: Get options data (40% progress)
            self._emit_progress(symbol, 40, "Fetching options data...")

            # Use pre-fetched expiration data (fetched on main thread)
            available_expirations = request.parameters.get('available_expirations', [])
            if not available_expirations:
                raise ValueError(f"No options expirations available for {symbol}")

            options_data = self._build_calendar_options_data(
                symbol=symbol,
                current_price=current_price,
                available_expirations=available_expirations,
                days_to_earnings=days_to_earnings
            )
            if not options_data:
                raise ValueError(f"Unable to build calendar options data for {symbol}")
            options_data['days_to_earnings_target'] = days_to_earnings
            options_data['earnings_alignment_error_days'] = abs(
                int(options_data.get('short_dte', 0)) - max(0, days_to_earnings)
            )

            # Step 5: Calculate volatility metrics (50% progress)
            self._emit_progress(symbol, 50, "Calculating volatility...")
            raw_vol_metrics = self.volatility_service.calculate_volatility_metrics(symbol)
            returns = historical_data['Close'].pct_change().dropna()
            historical_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.25
            volatility_metrics = self._normalize_volatility_metrics(raw_vol_metrics, historical_vol)

            # Blend term-structure signal derived from selected spread contracts
            if self._safe_get_options_value(options_data, 'term_structure_slope', None) is not None:
                volatility_metrics['term_structure_slope'] = float(
                    self._safe_get_options_value(options_data, 'term_structure_slope', 0.0)
                )
            self._augment_volatility_metrics_with_options_data(volatility_metrics, options_data)

            # Step 6: Calculate Greeks (60% progress)
            self._emit_progress(symbol, 60, "Calculating Greeks...")

            # Import required types for MarketParameters
            from utils.greeks_calculator import MarketParameters, OptionType

            # Extract values from options_data and volatility_metrics
            strike_price = self._safe_get_options_value(options_data, 'strike', current_price)
            short_dte = self._safe_get_options_value(options_data, 'short_dte', 30) / 365.25
            long_dte = self._safe_get_options_value(options_data, 'long_dte', 60) / 365.25
            risk_free_rate = 0.05  # Default 5% risk-free rate
            volatility = 0.25
            if isinstance(volatility_metrics, dict):
                for iv_key in ("implied_volatility", "implied_vol_30d", "iv30", "iv"):
                    iv_value = volatility_metrics.get(iv_key)
                    if isinstance(iv_value, (int, float)) and iv_value > 0:
                        volatility = float(iv_value)
                        break

            # Create MarketParameters for calendar spread (typically uses puts)
            short_params = MarketParameters(
                spot_price=current_price,
                strike_price=strike_price,
                time_to_expiry=max(short_dte, 1/365.25),  # Minimum 1 day
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=OptionType.PUT
            )

            long_params = MarketParameters(
                spot_price=current_price,
                strike_price=strike_price,
                time_to_expiry=max(long_dte, short_dte + 7/365.25),  # Minimum 7 days after short
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=OptionType.PUT
            )

            greeks = self.greeks_calculator.calculate_calendar_spread_greeks(
                short_params, long_params
            )

            # Step 7: Run Monte Carlo simulation (70% progress)
            self._emit_progress(symbol, 70, "Running Monte Carlo simulation...")
            monte_carlo_params = request.parameters.get('monte_carlo_sims', 10000)
            monte_carlo_result = self.monte_carlo_engine.run_simulation(
                symbol=symbol,
                current_price=current_price,
                historical_data=historical_data,
                volatility_metrics=volatility_metrics,
                simulations=monte_carlo_params,
                days_to_expiration=self._safe_get_options_value(options_data, 'days_to_expiration', 30),
                days_to_earnings=days_to_earnings
            )

            # Step 8: Calculate confidence score (80% progress)
            self._emit_progress(symbol, 80, "Calculating confidence score...")
            confidence_score = self._calculate_confidence_score(
                symbol, current_price, volatility_metrics, monte_carlo_result,
                days_to_earnings, options_data, greeks, earnings_move_profile
            )

            # Step 9: Machine learning prediction (90% progress)
            self._emit_progress(symbol, 90, "Running ML prediction...")
            ml_prediction = None
            if request.parameters.get('use_ml', True):
                try:
                    ml_features = self._prepare_ml_features(
                        symbol, current_price, volatility_metrics,
                        days_to_earnings, options_data
                    )
                    prediction = self.ml_service.predict_trade_outcome(ml_features)
                    ml_prediction = {
                        'probability': prediction.probability,
                        'confidence': prediction.confidence.value,
                        'risk_score': prediction.risk_score,
                        'recommendation': prediction.recommendation,
                        'model_version': prediction.model_version,
                        'contributing_factors': prediction.contributing_factors
                    }
                except Exception as ml_error:
                    self.logger.warning(f"ML prediction failed for {symbol}: {ml_error}")

            # Step 10: Calculate risk metrics and final assembly (100% progress)
            self._emit_progress(symbol, 100, "Finalizing analysis...")
            risk_metrics = self._calculate_risk_metrics(
                current_price, options_data, request.contracts,
                request.debit_override, volatility_metrics, monte_carlo_result
            )

            options_data['earnings_move_profile'] = earnings_move_profile
            options_data['calendar_edge_score'] = confidence_score.overall_score
            options_data['calendar_edge_grade'] = self._score_to_grade(confidence_score.overall_score)

            # Create analysis result
            analysis_result = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                request_id=request.request_id,
                current_price=current_price,
                confidence_score=confidence_score,
                volatility_metrics=volatility_metrics,
                options_data=options_data,
                greeks=greeks,
                monte_carlo_result=monte_carlo_result,
                ml_prediction=ml_prediction,
                risk_metrics=risk_metrics,
                earnings_date=earnings_date,
                days_to_earnings=days_to_earnings,
                analysis_duration=(datetime.now() - start_time).total_seconds(),
                analysis_type=str(request.parameters.get('analysis_type', 'comprehensive')),
                source=str(request.parameters.get('source', 'analysis_controller'))
            )

            # Update statistics
            self.analysis_stats['successful_analyses'] += 1
            self.analysis_stats['total_analyses'] += 1

            duration = (datetime.now() - start_time).total_seconds()
            total_duration = (self.analysis_stats['average_duration'] *
                            (self.analysis_stats['successful_analyses'] - 1) + duration)
            self.analysis_stats['average_duration'] = (
                total_duration / self.analysis_stats['successful_analyses']
            )

            self.logger.info(f"Completed analysis for {symbol} in {duration:.2f}s")
            return analysis_result

        except Exception as e:
            self.analysis_stats['failed_analyses'] += 1
            self.analysis_stats['total_analyses'] += 1
            self.logger.error(f"Analysis failed for {symbol}: {e}")
            raise

    def _emit_progress(self, symbol: str, progress: int, status: str):
        """Emit progress signal safely from background thread"""
        try:
            # Direct signal emission - Qt handles thread safety automatically
            self.analysis_progress.emit(symbol, progress, status)
        except Exception as e:
            self.logger.warning(f"Progress signal emit failed for {symbol}: {e}")


    def _emit_started_signal(self, symbol: str):
        """Emit analysis started signal safely from background thread"""
        try:
            # Direct signal emission - Qt handles thread safety automatically
            self.analysis_started.emit(symbol)
        except Exception as e:
            self.logger.warning(f"Started signal emit failed for {symbol}: {e}")


    def _emit_completion_signal(self, symbol: str, result):
        """Emit analysis completed signal safely from background thread"""
        try:
            # Direct signal emission - Qt handles thread safety automatically
            self.analysis_completed.emit(symbol, result)
        except Exception as e:
            self.logger.warning(f"Completion signal emit failed for {symbol}: {e}")


    def _emit_error_signal(self, symbol: str, error_msg: str):
        """Emit analysis error signal safely from background thread"""
        try:
            # Direct signal emission - Qt handles thread safety automatically
            self.analysis_error.emit(symbol, error_msg)
        except Exception as e:
            self.logger.warning(f"Error signal emit failed for {symbol}: {e}")


    def _calculate_days_to_earnings(self, earnings_date) -> int:
        """Calculate days until next earnings"""
        if not earnings_date:
            return 90  # Default assumption - quarterly earnings

        try:
            if isinstance(earnings_date, str):
                from datetime import datetime
                earnings_date = datetime.strptime(earnings_date, "%Y-%m-%d").date()
            elif hasattr(earnings_date, 'date'):
                earnings_date = earnings_date.date()

            today = datetime.now().date()
            return (earnings_date - today).days

        except Exception as e:
            self.logger.warning(f"Error calculating days to earnings: {e}")
            return 90

    def _normalize_volatility_metrics(self, raw_metrics: Any, historical_vol: float) -> Dict[str, Any]:
        """Normalize volatility service output to a consistent dictionary schema."""
        metrics: Dict[str, Any] = {}
        if hasattr(raw_metrics, '__dict__'):
            for field_name, field_value in raw_metrics.__dict__.items():
                metrics[field_name] = getattr(field_value, 'value', field_value)
        elif isinstance(raw_metrics, dict):
            metrics.update(raw_metrics)

        rv30 = float(metrics.get('rv30', metrics.get('realized_vol_30d', historical_vol)))
        iv30 = float(metrics.get('iv30', metrics.get('implied_vol_30d', max(rv30, 0.01))))
        iv_rank = float(metrics.get('iv_rank', metrics.get('vol_rank', 0.5)))
        iv_percentile = float(metrics.get('iv_percentile', metrics.get('vol_percentile', 50.0)))
        vix_level = float(metrics.get('vix_level', metrics.get('vix', self.market_data_service.get_vix())))

        metrics['rv30'] = max(0.01, rv30)
        metrics['iv30'] = max(0.01, iv30)
        metrics['iv_rv_ratio'] = self._safe_ratio(metrics['iv30'], metrics['rv30'], default=1.0)
        metrics['iv_rank'] = max(0.0, min(1.0, iv_rank))
        metrics['iv_percentile'] = max(0.0, min(100.0, iv_percentile))
        metrics['theta'] = float(metrics.get('theta', metrics['rv30']))
        metrics['historical_volatility'] = float(metrics.get('historical_volatility', metrics['rv30']))
        metrics['vix_level'] = vix_level
        metrics['vix'] = vix_level
        metrics['term_structure_slope'] = float(metrics.get('term_structure_slope', 0.0))
        return metrics

    def _augment_volatility_metrics_with_options_data(self, metrics: Dict[str, Any], options_data: Dict[str, Any]) -> None:
        """Inject option-surface context into volatility metrics for MC calibration."""
        try:
            metrics['short_iv'] = float(self._safe_get_options_value(options_data, 'short_iv', metrics.get('iv30', 0.25)))
            metrics['long_iv'] = float(self._safe_get_options_value(options_data, 'long_iv', metrics.get('iv30', 0.25)))
            metrics['short_dte'] = float(self._safe_get_options_value(options_data, 'short_dte', 30.0))
            metrics['long_dte'] = float(self._safe_get_options_value(options_data, 'long_dte', 60.0))
            metrics['event_iv_premium'] = float(self._safe_get_options_value(
                options_data, 'event_iv_premium', metrics['short_iv'] - metrics['long_iv']
            ))
            metrics['iv_term_ratio'] = float(self._safe_get_options_value(options_data, 'iv_term_ratio', 1.0))
            metrics['term_structure_slope_0_45'] = float(self._safe_get_options_value(
                options_data, 'term_structure_slope_0_45',
                self._safe_get_options_value(options_data, 'term_structure_slope', metrics.get('term_structure_slope', 0.0))
            ))
            metrics['iv_30d'] = float(self._safe_get_options_value(options_data, 'term_structure_iv_30d', metrics.get('iv30', 0.25)))
            metrics['iv_45d'] = float(self._safe_get_options_value(
                options_data, 'term_structure_iv_45d',
                self._safe_get_options_value(options_data, 'long_iv', metrics.get('iv30', 0.25))
            ))

            ts_days = self._safe_get_options_value(options_data, 'term_structure_days', [])
            ts_ivs = self._safe_get_options_value(options_data, 'term_structure_atm_ivs', [])
            if isinstance(ts_days, (list, tuple)) and isinstance(ts_ivs, (list, tuple)) and len(ts_days) == len(ts_ivs):
                clean_days = []
                clean_ivs = []
                for d, v in zip(ts_days, ts_ivs):
                    try:
                        d_val = float(d)
                        v_val = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(d_val) and np.isfinite(v_val) and d_val > 0 and 0 < v_val < 5:
                        clean_days.append(d_val)
                        clean_ivs.append(v_val)
                if len(clean_days) >= 2:
                    metrics['term_structure_days'] = clean_days
                    metrics['term_structure_atm_ivs'] = clean_ivs
        except Exception:
            return

    def _build_calendar_options_data(self, symbol: str, current_price: float,
                                   available_expirations: List[str],
                                   days_to_earnings: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Build calendar-spread-ready option metrics for a symbol."""
        try:
            expirations = self.options_service.filter_expirations_for_calendar(available_expirations)
            if len(expirations) < 2:
                expirations = available_expirations
            if len(expirations) < 2:
                return None

            pair = self._choose_calendar_expirations(expirations, days_to_earnings)
            if pair is None:
                return None
            short_exp, long_exp = pair
            short_chain = self.options_service.get_option_chain(symbol, short_exp)
            long_chain = self.options_service.get_option_chain(symbol, long_exp)
            if not short_chain or not long_chain:
                return None

            strike = float(short_chain.get_atm_strike())
            short_contract = short_chain.get_contract(strike, ServiceOptionType.PUT)
            long_contract = long_chain.get_contract(strike, ServiceOptionType.PUT)

            if short_contract is None or long_contract is None:
                put_short = short_contract or self._nearest_contract(short_chain.puts, strike)
                put_long = long_contract or self._nearest_contract(long_chain.puts, strike)
                if put_short and put_long:
                    short_contract, long_contract = put_short, put_long
                else:
                    short_contract = self._nearest_contract(short_chain.calls, strike)
                    long_contract = self._nearest_contract(long_chain.calls, strike)
            if not short_contract or not long_contract:
                return None

            short_dte = max(1, int(getattr(short_chain, 'days_to_expiration', 30)))
            long_dte = max(short_dte + 1, int(getattr(long_chain, 'days_to_expiration', short_dte + 30)))

            short_mid = float(getattr(short_contract, 'mid_price', 0.0) or 0.0)
            long_mid = float(getattr(long_contract, 'mid_price', 0.0) or 0.0)
            short_bid = float(getattr(short_contract, 'bid', short_mid) or short_mid)
            long_ask = float(getattr(long_contract, 'ask', long_mid) or long_mid)
            short_premium = max(0.0, short_mid)
            long_premium = max(0.0, long_mid)
            debit = max(0.01, long_ask - short_bid, long_premium - short_premium)

            short_iv = float(getattr(short_contract, 'implied_volatility', 0.0) or 0.0)
            long_iv = float(getattr(long_contract, 'implied_volatility', 0.0) or 0.0)
            event_iv_premium = short_iv - long_iv
            term_structure_slope = self._safe_ratio(long_iv - short_iv, long_dte - short_dte, default=0.0)
            iv_term_ratio = self._safe_ratio(short_iv, max(long_iv, 0.0001), default=1.0)

            short_spread = float(getattr(short_contract, 'spread', 0.0) or 0.0)
            long_spread = float(getattr(long_contract, 'spread', 0.0) or 0.0)
            short_spread_pct = self._safe_ratio(short_spread, max(short_mid, 0.01), default=0.0)
            long_spread_pct = self._safe_ratio(long_spread, max(long_mid, 0.01), default=0.0)
            bid_ask_spread_pct = float((short_spread_pct + long_spread_pct) / 2.0)

            short_volume = int(getattr(short_contract, 'volume', 0) or 0)
            long_volume = int(getattr(long_contract, 'volume', 0) or 0)
            short_oi = int(getattr(short_contract, 'open_interest', 0) or 0)
            long_oi = int(getattr(long_contract, 'open_interest', 0) or 0)
            average_option_volume = float((short_volume + long_volume) / 2.0)
            open_interest = float(min(short_oi, long_oi))
            option_volume = float(short_volume + long_volume)
            volume_turnover = self._safe_ratio(option_volume, max(open_interest, 1.0), default=0.0)

            underlying_avg_volume = 0.0
            underlying_last_volume = 0.0
            try:
                underlying_hist = self.market_data_service.get_historical_data(
                    symbol=symbol, period="3mo", interval="1d"
                )
                if not underlying_hist.empty and 'Volume' in underlying_hist.columns:
                    volume_series = pd.to_numeric(underlying_hist['Volume'], errors='coerce').dropna()
                    if not volume_series.empty:
                        underlying_last_volume = float(volume_series.iloc[-1])
                        underlying_avg_volume = float(volume_series.tail(20).mean())
            except Exception:
                underlying_avg_volume = 0.0
                underlying_last_volume = 0.0

            implied_move = float(short_chain.get_straddle_price(strike))
            implied_move_pct = self._safe_ratio(implied_move, current_price, default=0.0) * 100.0
            debit_to_implied_move = self._safe_ratio(debit, max(implied_move, 0.01), default=1.0)
            dte_gap = max(1, long_dte - short_dte)
            short_theta = float(getattr(short_contract, 'theta', -1.0) or -1.0)
            long_theta = float(getattr(long_contract, 'theta', -0.8) or -0.8)
            theta_ratio_estimate = self._safe_ratio(abs(short_theta), max(abs(long_theta), 1e-6), default=1.0)

            liquidity_score = self._calculate_liquidity_score(
                average_volume=average_option_volume,
                open_interest=open_interest,
                bid_ask_spread_pct=bid_ask_spread_pct
            )
            transaction_cost_per_contract = self._estimate_transaction_cost_per_contract(
                short_spread=short_spread,
                long_spread=long_spread,
                average_volume=average_option_volume,
                open_interest=open_interest
            )
            cost_to_debit_ratio = self._safe_ratio(transaction_cost_per_contract, max(debit, 0.01), default=0.0)
            debit_after_cost = debit + transaction_cost_per_contract

            ts_data = self.options_service.build_iv_term_structure(symbol)
            term_structure_slope_0_45 = term_structure_slope
            term_structure_iv_30d = short_iv
            term_structure_iv_45d = long_iv
            term_structure_days: List[float] = []
            term_structure_atm_ivs: List[float] = []
            if ts_data:
                term_structure_slope_0_45 = float(ts_data.get('slope_0_45', term_structure_slope))
                term_structure_iv_30d = float(ts_data.get('iv_30d', term_structure_iv_30d))
                term_structure_iv_45d = float(ts_data.get('iv_45d', term_structure_iv_45d))
                ts_days_raw = ts_data.get('days_to_expiration', [])
                ts_ivs_raw = ts_data.get('atm_ivs', [])
                if isinstance(ts_days_raw, (list, tuple)) and isinstance(ts_ivs_raw, (list, tuple)):
                    for d, v in zip(ts_days_raw, ts_ivs_raw):
                        try:
                            d_val = float(d)
                            v_val = float(v)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(d_val) and np.isfinite(v_val) and d_val > 0 and 0 < v_val < 5:
                            term_structure_days.append(d_val)
                            term_structure_atm_ivs.append(v_val)

            return {
                'symbol': symbol,
                'short_expiration': short_exp,
                'long_expiration': long_exp,
                'short_chain': short_chain,
                'long_chain': long_chain,
                'short_contract': short_contract,
                'long_contract': long_contract,
                'strike': strike,
                'short_dte': short_dte,
                'long_dte': long_dte,
                'days_to_expiration': short_dte,
                'short_premium': short_premium,
                'long_premium': long_premium,
                'debit': debit,
                'debit_after_cost': debit_after_cost,
                'average_volume': average_option_volume,
                'average_option_volume': average_option_volume,
                'option_volume': option_volume,
                'underlying_avg_volume': underlying_avg_volume,
                'underlying_last_volume': underlying_last_volume,
                'open_interest': open_interest,
                'volume_ratio': volume_turnover,
                'oi_ratio': self._safe_ratio(open_interest, max(average_option_volume, 1.0), default=1.0),
                'option_volume_share': self._safe_ratio(option_volume, max(underlying_avg_volume, 1.0), default=0.0),
                'bid_ask_spread': short_spread + long_spread,
                'bid_ask_spread_pct': bid_ask_spread_pct,
                'short_iv': short_iv,
                'long_iv': long_iv,
                'short_theta': short_theta,
                'long_theta': long_theta,
                'theta_ratio_estimate': theta_ratio_estimate,
                'event_iv_premium': event_iv_premium,
                'term_structure_slope': term_structure_slope,
                'term_structure_slope_0_45': term_structure_slope_0_45,
                'term_structure_iv_30d': term_structure_iv_30d,
                'term_structure_iv_45d': term_structure_iv_45d,
                'term_structure_days': term_structure_days,
                'term_structure_atm_ivs': term_structure_atm_ivs,
                'iv_term_ratio': iv_term_ratio,
                'dte_gap': dte_gap,
                'implied_move': implied_move,
                'implied_move_pct': implied_move_pct,
                'debit_to_implied_move': debit_to_implied_move,
                'liquidity_score': liquidity_score,
                'transaction_cost_per_contract': transaction_cost_per_contract,
                'cost_to_debit_ratio': cost_to_debit_ratio,
                'sector': 'Unknown',
                'put_call_ratio': 1.0
            }
        except Exception as e:
            self.logger.error(f"Failed to build calendar options data for {symbol}: {e}")
            return None

    def _choose_calendar_expirations(self, expirations: List[str],
                                   days_to_earnings: Optional[int]) -> Optional[Tuple[str, str]]:
        """Choose short/long expirations with preference for event-aligned short legs."""
        try:
            today = datetime.now().date()
            dated: List[Tuple[str, int]] = []
            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    dte = (exp_date - today).days
                    if dte > 0:
                        dated.append((exp_str, dte))
                except ValueError:
                    continue

            dated.sort(key=lambda x: x[1])
            if len(dated) < 2:
                return None

            if days_to_earnings is None or days_to_earnings <= 0:
                return dated[0][0], dated[1][0]

            target_short = max(3, int(days_to_earnings))
            best_pair: Optional[Tuple[str, str]] = None
            best_score = float("inf")

            for i in range(len(dated) - 1):
                short_exp, short_dte = dated[i]
                for j in range(i + 1, len(dated)):
                    long_exp, long_dte = dated[j]
                    gap = long_dte - short_dte
                    if gap <= 0:
                        continue

                    # Strongly prefer short expiry at or just after earnings.
                    if short_dte >= target_short:
                        score = abs(short_dte - target_short) * 1.2
                    else:
                        score = 120.0 + (target_short - short_dte) * 4.0

                    # Prefer long leg 2-10 weeks beyond short leg.
                    if 14 <= gap <= 70:
                        score += abs(gap - 35) * 0.6
                    else:
                        score += 35.0 + abs(gap - 35)

                    if score < best_score:
                        best_score = score
                        best_pair = (short_exp, long_exp)

            return best_pair or (dated[0][0], dated[1][0])
        except Exception as e:
            self.logger.warning(f"Error choosing calendar expirations: {e}")
            if len(expirations) >= 2:
                return expirations[0], expirations[1]
            return None

    def _estimate_transaction_cost_per_contract(self, short_spread: float, long_spread: float,
                                               average_volume: float, open_interest: float) -> float:
        """Estimate per-share execution drag for one calendar spread (2 legs)."""
        try:
            try:
                profile_name = str(
                    self.config_manager.get("trading.execution_profile", self.execution_cost_model.profile_name)
                ).strip().lower()
            except Exception:
                profile_name = self.execution_cost_model.profile_name
            model = self.execution_cost_model
            if profile_name != self.execution_cost_model.profile_name:
                model = ExecutionCostModel(profile_name)

            estimate = model.estimate_calendar_round_trip_cost(
                short_spread=float(short_spread),
                long_spread=float(long_spread),
                average_volume=float(average_volume),
                open_interest=float(open_interest),
                contracts=1,
            )
            return float(max(0.01, estimate.get("cost_per_contract", 0.02)))
        except Exception:
            return 0.02

    def _nearest_contract(self, contracts: List[Any], target_strike: float) -> Optional[Any]:
        """Return the contract with strike nearest to target strike."""
        if not contracts:
            return None
        return min(contracts, key=lambda contract: abs(float(getattr(contract, 'strike', target_strike)) - target_strike))

    def _calculate_liquidity_score(self, average_volume: float, open_interest: float,
                                 bid_ask_spread_pct: float) -> float:
        """Normalize liquidity quality to a 0-1 score."""
        volume_score = min(1.0, np.log1p(max(0.0, average_volume)) / 8.5)
        oi_score = min(1.0, np.log1p(max(0.0, open_interest)) / 9.0)
        spread_score = max(0.0, 1.0 - min(1.0, max(0.0, bid_ask_spread_pct) / 0.30))
        return float(0.4 * volume_score + 0.4 * oi_score + 0.2 * spread_score)

    def _estimate_historical_earnings_move(self, symbol: str, historical_data) -> Dict[str, Any]:
        """Estimate historical absolute earnings move from recent events."""
        fallback_move = 4.0
        profile = {
            'sample_size': 0,
            'mean_abs_move_pct': fallback_move,
            'median_abs_move_pct': fallback_move
        }
        try:
            if historical_data is None or historical_data.empty:
                return profile

            import pandas as pd
            import yfinance as yf

            hist = historical_data.copy()
            hist_index = pd.to_datetime(hist.index)
            if getattr(hist_index, 'tz', None) is not None:
                hist_index = hist_index.tz_localize(None)
            hist.index = hist_index
            ticker = yf.Ticker(symbol)
            earnings_dates = getattr(ticker, 'earnings_dates', None)
            if earnings_dates is None or earnings_dates.empty:
                return profile

            now = datetime.now()
            event_dates = []
            for ts in earnings_dates.index:
                ts_obj = pd.Timestamp(ts)
                if ts_obj.tzinfo is not None:
                    ts_obj = ts_obj.tz_localize(None)
                if ts_obj.to_pydatetime() < now:
                    event_dates.append(ts_obj)
            event_dates = sorted(event_dates)[-8:]
            moves: List[float] = []
            for event_date in event_dates:
                pos = int(hist.index.searchsorted(event_date))
                if pos <= 0 or pos + 1 >= len(hist):
                    continue
                pre_close = float(hist['Close'].iloc[pos - 1])
                post_close = float(hist['Close'].iloc[pos + 1])
                if pre_close > 0:
                    moves.append(abs((post_close / pre_close - 1.0) * 100.0))

            if moves:
                profile['sample_size'] = len(moves)
                profile['mean_abs_move_pct'] = float(np.mean(moves))
                profile['median_abs_move_pct'] = float(np.median(moves))
            return profile
        except Exception as e:
            self.logger.debug(f"Could not estimate earnings move profile for {symbol}: {e}")
            return profile

    def _calculate_calendar_edge_profile(self, volatility_metrics: dict, monte_carlo_result: dict,
                                       days_to_earnings: int, options_data: dict, greeks: Any,
                                       earnings_move_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Build normalized edge components for pre-earnings calendar setups."""
        short_dte = int(self._safe_get_options_value(options_data, 'short_dte', 30))
        long_dte = int(self._safe_get_options_value(options_data, 'long_dte', max(short_dte + 7, 60)))
        short_iv = float(self._safe_get_options_value(options_data, 'short_iv', volatility_metrics.get('iv30', 0.25)))
        long_iv = float(self._safe_get_options_value(options_data, 'long_iv', volatility_metrics.get('iv30', 0.25)))
        event_iv_premium = float(self._safe_get_options_value(options_data, 'event_iv_premium', short_iv - long_iv))

        implied_move_pct = float(self._safe_get_options_value(options_data, 'implied_move_pct', 0.0))
        if implied_move_pct <= 0:
            iv30 = float(volatility_metrics.get('iv30', 0.25))
            implied_move_pct = current_move = iv30 * np.sqrt(max(short_dte, 1) / 365.25) * 100.0
        else:
            current_move = implied_move_pct
        historical_move_pct = float(earnings_move_profile.get('median_abs_move_pct', 4.0))
        move_ratio = self._safe_ratio(current_move, max(historical_move_pct, 0.10), default=1.0)
        iv_term_ratio = float(self._safe_get_options_value(
            options_data, 'iv_term_ratio',
            self._safe_ratio(short_iv, max(long_iv, 0.0001), default=1.0)
        ))
        dte_gap = int(self._safe_get_options_value(options_data, 'dte_gap', max(1, long_dte - short_dte)))
        debit_to_move = float(self._safe_get_options_value(
            options_data, 'debit_to_implied_move',
            self._safe_ratio(
                self._safe_get_options_value(options_data, 'debit', 0.0),
                max(self._safe_get_options_value(options_data, 'implied_move', 0.01), 0.01),
                default=1.0
            )
        ))

        if days_to_earnings < 0:
            earnings_alignment = 25.0
        elif days_to_earnings <= short_dte:
            earnings_alignment = 92.0
        elif days_to_earnings <= long_dte:
            earnings_alignment = 58.0
        elif days_to_earnings <= long_dte + 14:
            earnings_alignment = 42.0
        else:
            earnings_alignment = 30.0

        if event_iv_premium >= 0.03:
            event_iv_score = 90.0
        elif event_iv_premium >= 0.01:
            event_iv_score = 76.0
        elif event_iv_premium >= -0.005:
            event_iv_score = 55.0
        else:
            event_iv_score = 30.0

        if 1.10 <= move_ratio <= 1.90:
            move_richness = 88.0
        elif 0.95 <= move_ratio < 1.10 or 1.90 < move_ratio <= 2.30:
            move_richness = 70.0
        elif 0.80 <= move_ratio < 0.95:
            move_richness = 55.0
        elif move_ratio < 0.80:
            move_richness = 35.0
        else:
            move_richness = 58.0

        if iv_term_ratio >= 1.12 and 14 <= dte_gap <= 65:
            term_structure_quality = 90.0
        elif iv_term_ratio >= 1.05 and 10 <= dte_gap <= 80:
            term_structure_quality = 78.0
        elif iv_term_ratio >= 0.98:
            term_structure_quality = 60.0
        else:
            term_structure_quality = 35.0

        if debit_to_move <= 0.45:
            debit_efficiency = 88.0
        elif debit_to_move <= 0.65:
            debit_efficiency = 72.0
        elif debit_to_move <= 0.85:
            debit_efficiency = 58.0
        elif debit_to_move <= 1.10:
            debit_efficiency = 45.0
        else:
            debit_efficiency = 32.0

        debit = float(self._safe_get_options_value(options_data, 'debit', 0.01))
        transaction_cost = float(self._safe_get_options_value(options_data, 'transaction_cost_per_contract', 0.02))
        cost_to_debit_ratio = float(self._safe_get_options_value(
            options_data, 'cost_to_debit_ratio',
            self._safe_ratio(transaction_cost, max(debit, 0.01), default=0.0)
        ))
        if cost_to_debit_ratio <= 0.08:
            execution_quality = 90.0
        elif cost_to_debit_ratio <= 0.14:
            execution_quality = 76.0
        elif cost_to_debit_ratio <= 0.22:
            execution_quality = 60.0
        elif cost_to_debit_ratio <= 0.32:
            execution_quality = 46.0
        else:
            execution_quality = 32.0

        liquidity_quality = float(self._safe_get_options_value(options_data, 'liquidity_score', 0.5)) * 100.0
        liquidity_quality = max(0.0, min(100.0, liquidity_quality))

        net_delta = float(getattr(greeks, 'net_delta', 0.0))
        net_vega = float(getattr(greeks, 'net_vega', 0.0))
        theta_ratio = float(getattr(greeks, 'time_decay_ratio', 1.0))
        delta_neutrality = max(0.0, 100.0 - abs(net_delta) * 240.0)
        vega_support = 80.0 if net_vega > 0 else 45.0
        theta_shape = 80.0 if theta_ratio >= 1.15 else 65.0 if theta_ratio >= 0.95 else 45.0
        greek_profile = 0.45 * delta_neutrality + 0.30 * vega_support + 0.25 * theta_shape

        vix_level = float(volatility_metrics.get('vix_level', volatility_metrics.get('vix', 20.0)))
        if 15.0 <= vix_level <= 28.0:
            market_regime = 80.0
        elif 12.0 <= vix_level < 15.0 or 28.0 < vix_level <= 35.0:
            market_regime = 64.0
        else:
            market_regime = 45.0

        components = {
            'earnings_alignment': earnings_alignment,
            'event_iv_premium': event_iv_score,
            'move_richness': move_richness,
            'term_structure_quality': term_structure_quality,
            'debit_efficiency': debit_efficiency,
            'execution_quality': execution_quality,
            'liquidity_quality': liquidity_quality,
            'greek_profile': greek_profile,
            'market_regime': market_regime
        }

        rationale = [
            f"Earnings in {days_to_earnings}d vs short leg {short_dte}d / long leg {long_dte}d.",
            f"Front-back IV premium: {event_iv_premium:.2%}.",
            f"Implied move {current_move:.2f}% vs historical earnings move {historical_move_pct:.2f}%.",
            f"Term ratio {iv_term_ratio:.2f} with DTE gap {dte_gap}d.",
            f"Debit/implied-move ratio: {debit_to_move:.2f}.",
            f"Execution drag {transaction_cost:.3f} ({cost_to_debit_ratio:.1%} of debit).",
            f"Liquidity score {liquidity_quality:.1f}/100 with spread {self._safe_get_options_value(options_data, 'bid_ask_spread_pct', 0.0):.2%}.",
        ]

        return {
            'components': components,
            'metrics': {
                'event_iv_premium': event_iv_premium,
                'implied_move_pct': current_move,
                'historical_move_pct': historical_move_pct,
                'move_ratio': move_ratio,
                'iv_term_ratio': iv_term_ratio,
                'dte_gap': dte_gap,
                'debit_to_implied_move': debit_to_move,
                'transaction_cost_per_contract': transaction_cost,
                'cost_to_debit_ratio': cost_to_debit_ratio,
                'vix_level': vix_level,
                'prob_exceed_1x': float(monte_carlo_result.get('prob_exceed_1x', 50.0))
            },
            'rationale': rationale
        }

    def _component_to_points(self, component_score: float, max_points: float) -> float:
        """Convert a 0-100 component score into signed point contribution."""
        component_score = max(0.0, min(100.0, float(component_score)))
        centered = (component_score - 50.0) / 50.0
        return float(max(-max_points, min(max_points, centered * max_points)))

    def _safe_ratio(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe numeric ratio helper."""
        try:
            numerator = float(numerator)
            denominator = float(denominator)
            if denominator == 0:
                return default
            return numerator / denominator
        except Exception:
            return default

    def _score_to_grade(self, score: float) -> str:
        """Convert confidence score to an easy-to-scan letter grade."""
        if score >= 80:
            return "A"
        if score >= 68:
            return "B"
        if score >= 55:
            return "C"
        if score >= 45:
            return "D"
        return "F"

    def _calculate_confidence_score(self, symbol: str, current_price: float,
                                  volatility_metrics: dict, monte_carlo_result: dict,
                                  days_to_earnings: int, options_data: dict,
                                  greeks: Any,
                                  earnings_move_profile: Optional[Dict[str, Any]] = None) -> ConfidenceScore:
        """Calculate explainable confidence score for pre-earnings calendar spreads."""
        try:
            base_score = 50.0
            factors: List[Tuple[str, float]] = []
            edge_profile = self._calculate_calendar_edge_profile(
                volatility_metrics=volatility_metrics,
                monte_carlo_result=monte_carlo_result,
                days_to_earnings=days_to_earnings,
                options_data=options_data,
                greeks=greeks,
                earnings_move_profile=earnings_move_profile or {}
            )

            component_weights = [
                ("Earnings Window Alignment", "earnings_alignment", 14.0),
                ("Event IV Premium", "event_iv_premium", 12.0),
                ("Implied vs Historical Move", "move_richness", 10.0),
                ("Term Structure Fit", "term_structure_quality", 9.0),
                ("Debit Efficiency", "debit_efficiency", 8.0),
                ("Execution Quality", "execution_quality", 6.0),
                ("Liquidity Quality", "liquidity_quality", 8.0),
                ("Greeks Structure", "greek_profile", 7.0),
                ("Market Regime Fit", "market_regime", 6.0),
            ]

            score = base_score
            for label, key, max_points in component_weights:
                component_score = float(edge_profile['components'].get(key, 50.0))
                points = self._component_to_points(component_score, max_points)
                factors.append((label, points))
                score += points

            prob_1x = float(monte_carlo_result.get('prob_exceed_1x', 50.0))
            if 38 <= prob_1x <= 62:
                mc_balance = 4.0
            elif 28 <= prob_1x <= 72:
                mc_balance = 1.5
            else:
                mc_balance = -4.5
            factors.append(("Path Risk Balance", mc_balance))
            score += mc_balance

            iv_rv_ratio = float(volatility_metrics.get('iv_rv_ratio', 1.0))
            if iv_rv_ratio >= 1.15:
                iv_rv_points = 3.5
            elif iv_rv_ratio >= 0.95:
                iv_rv_points = 1.0
            else:
                iv_rv_points = -3.5
            factors.append(("Volatility Risk Premium", iv_rv_points))
            score += iv_rv_points

            final_score = max(0.0, min(100.0, score))
            factors.append((f"Edge Grade {self._score_to_grade(final_score)}", 0.0))

            return ConfidenceScore(
                overall_score=final_score,
                factors=factors,
                recommendation=self._get_recommendation(final_score),
                risk_level=self._get_risk_level(final_score)
            )

        except Exception as e:
            self.logger.error(f"Error calculating confidence score for {symbol}: {e}")
            return ConfidenceScore(
                overall_score=50.0,
                factors=[('Calculation Error', 0.0)],
                recommendation="ANALYZE MANUALLY",
                risk_level="UNKNOWN"
            )

    def _get_recommendation(self, score: float) -> str:
        """Get trading recommendation based on confidence score"""
        if score >= 75:
            return "STRONG BUY"
        elif score >= 65:
            return "BUY"
        elif score >= 55:
            return "CONSIDER"
        elif score >= 45:
            return "WEAK"
        else:
            return "AVOID"

    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on confidence score"""
        if score >= 70:
            return "LOW"
        elif score >= 55:
            return "MODERATE"
        elif score >= 40:
            return "HIGH"
        else:
            return "VERY HIGH"

    def _prepare_ml_features(self, symbol: str, current_price: float,
                           volatility_metrics: dict, days_to_earnings: int,
                           options_data: dict) -> dict:
        """Prepare features for ML prediction"""
        short_iv = float(self._safe_get_options_value(options_data, 'short_iv', volatility_metrics.get('iv30', 0.25)))
        long_iv = float(self._safe_get_options_value(options_data, 'long_iv', volatility_metrics.get('iv30', 0.25)))
        avg_volume = float(self._safe_get_options_value(
            options_data,
            'underlying_avg_volume',
            self._safe_get_options_value(options_data, 'average_volume', 0.0)
        ))
        option_volume = float(self._safe_get_options_value(options_data, 'option_volume', avg_volume))
        open_interest = float(self._safe_get_options_value(options_data, 'open_interest', 0.0))
        bid_ask_spread = float(self._safe_get_options_value(options_data, 'bid_ask_spread_pct', 0.0))
        iv_rv_ratio = float(volatility_metrics.get('iv_rv_ratio', 1.0))
        vix_level = float(volatility_metrics.get('vix_level', volatility_metrics.get('vix', 20.0)))
        short_theta = float(self._safe_get_options_value(options_data, 'short_theta', -1.0))
        long_theta = float(self._safe_get_options_value(options_data, 'long_theta', -0.8))
        time_decay_ratio = self._safe_ratio(abs(short_theta), max(abs(long_theta), 1e-6), default=1.0)
        liquidity_score = float(self._safe_get_options_value(options_data, 'liquidity_score', 0.5))
        term_structure_slope = float(self._safe_get_options_value(
            options_data, 'term_structure_slope_0_45',
            self._safe_get_options_value(options_data, 'term_structure_slope',
            volatility_metrics.get('term_structure_slope', 0.0)
            )
        ))
        volatility_skew = float(self._safe_get_options_value(options_data, 'event_iv_premium', short_iv - long_iv))

        return {
            'iv30_rv30': iv_rv_ratio,
            'ts_slope_0_45': term_structure_slope,
            'days_to_earnings': days_to_earnings,
            'vix': vix_level,
            'avg_volume': avg_volume,
            'gamma': float(self._safe_get_options_value(options_data, 'gamma_exposure', 0.01)),
            'sector': str(self._safe_get_options_value(options_data, 'sector', 'Unknown')),
            'iv_rank': float(volatility_metrics.get('iv_rank', 0.5)),
            'iv_percentile': float(volatility_metrics.get('iv_percentile', 50.0)),
            'short_theta': short_theta,
            'long_theta': long_theta,
            'time_decay_ratio': time_decay_ratio,
            'liquidity_score': liquidity_score,
            'volatility_skew': volatility_skew,
            'option_volume': option_volume,
            'open_interest': open_interest,
            'bid_ask_spread': bid_ask_spread,
            'call_iv': max(short_iv, 0.01),
            'put_iv': max(long_iv, 0.01),
            'put_call_ratio': float(self._safe_get_options_value(options_data, 'put_call_ratio', 1.0))
        }

    def _calculate_risk_metrics(self, current_price: float, options_data: dict,
                              contracts: int, debit_override: Optional[float],
                              volatility_metrics: dict,
                              monte_carlo_result: Optional[dict] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Calculate debit
            if debit_override:
                debit = debit_override
            else:
                # Estimate debit from options data
                short_premium = self._safe_get_options_value(options_data, 'short_premium', current_price * 0.02)
                long_premium = self._safe_get_options_value(options_data, 'long_premium', current_price * 0.025)
                debit = max(0.01, float(long_premium - short_premium))
            short_premium = float(self._safe_get_options_value(options_data, 'short_premium', current_price * 0.02))
            long_premium = float(self._safe_get_options_value(options_data, 'long_premium', current_price * 0.025))
            short_theta = float(self._safe_get_options_value(options_data, 'short_theta', -1.0))
            long_theta = float(self._safe_get_options_value(options_data, 'long_theta', -0.8))
            theta_ratio = self._safe_ratio(abs(short_theta), max(abs(long_theta), 1e-6), default=1.0)
            event_iv_premium = float(self._safe_get_options_value(options_data, 'event_iv_premium', 0.0))
            transaction_cost_per_contract = float(self._safe_get_options_value(
                options_data, 'transaction_cost_per_contract', 0.0
            ))
            debit_effective = max(0.01, debit + transaction_cost_per_contract)

            # Maximum loss (debit paid)
            max_loss_per_contract = debit_effective
            max_loss_total = max_loss_per_contract * contracts * 100

            # Maximum profit estimate (calendar-specific heuristic with IV crush + theta edge).
            iv_capture = float(np.clip(0.30 + max(0.0, event_iv_premium) * 2.8, 0.20, 0.95))
            theta_capture = float(np.clip(0.35 + (theta_ratio - 1.0) * 0.15, 0.20, 0.85))
            short_decay_capture = short_premium * (0.55 * iv_capture + 0.45 * theta_capture)
            long_residual = long_premium * float(np.clip(0.18 + (theta_ratio - 1.0) * 0.10, 0.12, 0.45))
            max_profit_per_contract = max(0.01, short_decay_capture + long_residual - debit_effective)
            max_profit_total = max_profit_per_contract * contracts * 100

            # Break-even band around strike based on debit and expected move.
            strike = float(self._safe_get_options_value(options_data, 'strike', current_price))
            implied_move = float(self._safe_get_options_value(
                options_data,
                'implied_move',
                current_price * float(volatility_metrics.get('iv30', 0.25)) * np.sqrt(
                    max(float(self._safe_get_options_value(options_data, 'days_to_expiration', 30)), 1.0) / 365.25
                )
            ))
            breakeven_width = max(debit_effective, implied_move * 0.55)
            break_even_upper = strike + breakeven_width
            break_even_lower = max(0.01, strike - breakeven_width)

            # Probability of profit (calendar-aware, not directional).
            prob_profit = self._safe_get_options_value(options_data, 'prob_profit', 0.5)
            if isinstance(prob_profit, (int, float)) and float(prob_profit) > 1.0:
                prob_profit = float(prob_profit) / 100.0
            if monte_carlo_result:
                mc_exceed_1x = monte_carlo_result.get('prob_exceed_1x')
                if isinstance(mc_exceed_1x, (int, float)):
                    mc_inside_1x = max(0.0, min(100.0, 100.0 - float(mc_exceed_1x)))
                    prob_profit = 0.65 * (mc_inside_1x / 100.0) + 0.35 * float(prob_profit)
            iv_support = float(np.clip(0.50 + max(0.0, event_iv_premium) * 4.0, 0.35, 0.85))
            liquidity_support = float(np.clip(self._safe_get_options_value(options_data, 'liquidity_score', 0.5), 0.10, 1.00))
            prob_profit = 0.75 * float(prob_profit) + 0.15 * iv_support + 0.10 * liquidity_support
            prob_profit = max(0.01, min(0.99, float(prob_profit)))

            # Risk-reward ratio
            risk_reward_ratio = max_profit_per_contract / max_loss_per_contract if max_loss_per_contract > 0 else 0

            # Expected value calculation
            expected_value = (prob_profit * max_profit_per_contract -
                            (1 - prob_profit) * max_loss_per_contract)

            return RiskMetrics(
                max_loss_per_contract=max_loss_per_contract,
                max_loss_total=max_loss_total,
                max_profit_per_contract=max_profit_per_contract,
                max_profit_total=max_profit_total,
                break_even_upper=break_even_upper,
                break_even_lower=break_even_lower,
                probability_of_profit=prob_profit,
                risk_reward_ratio=risk_reward_ratio,
                expected_value=expected_value,
                debit_paid=debit_effective,
                contracts=contracts,
                transaction_cost_per_contract=transaction_cost_per_contract
            )

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                max_loss_per_contract=0.0,
                max_loss_total=0.0,
                max_profit_per_contract=0.0,
                max_profit_total=0.0,
                break_even_upper=current_price,
                break_even_lower=current_price,
                probability_of_profit=0.5,
                risk_reward_ratio=0.0,
                expected_value=0.0,
                debit_paid=0.0,
                contracts=contracts,
                transaction_cost_per_contract=0.0
            )

    def _create_safe_completion_callback(self, request_id: str):
        """Create thread-safe completion callback"""
        def safe_callback(result):
            try:
                QTimer.singleShot(0, lambda: self._on_analysis_complete(request_id, result))
            except RuntimeError as e:
                self.logger.error(f"Failed to invoke completion callback: {e}")
        return safe_callback

    def _create_safe_error_callback(self, request_id: str):
        """Create thread-safe error callback"""
        def safe_callback(error):
            try:
                QTimer.singleShot(0, lambda: self._on_analysis_error(request_id, str(error)))
            except RuntimeError as e:
                self.logger.error(f"Failed to invoke error callback: {e}")
        return safe_callback

    @Slot(str, object)
    def _on_analysis_complete(self, request_id: str, result: AnalysisResult):
        """Handle analysis completion (always called in main thread)"""
        try:
            if request_id in self.active_analyses:
                request = self.active_analyses.pop(request_id)

                # Cache the result
                cache_key = self._get_cache_key(request)
                self.analysis_cache[cache_key] = result

                # Emit completion signal safely
                try:
                    self._emit_completion_signal(request.symbol, result)
                    self.logger.info(f"Analysis completed for {request.symbol}")
                except RuntimeError as e:
                    self.logger.error(f"Failed to emit analysis_completed signal: {e}")

        except Exception as e:
            self.logger.error(f"Error handling analysis completion: {e}")

    @Slot(str, str)
    def _on_analysis_error(self, request_id: str, error_str: str):
        """Handle analysis error (always called in main thread)"""
        try:
            if request_id in self.active_analyses:
                request = self.active_analyses.pop(request_id)

                # Emit error signal safely
                try:
                    self._emit_error_signal(request.symbol, error_str)
                    self.logger.error(f"Analysis failed for {request.symbol}: {error_str}")
                except RuntimeError as e:
                    self.logger.error(f"Failed to emit analysis_error signal: {e}")

        except Exception as e:
            self.logger.error(f"Error handling analysis error: {e}")

    def _cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)

            old_keys = [
                key for key, result in self.analysis_cache.items()
                if result.timestamp < cutoff_time
            ]

            for key in old_keys:
                del self.analysis_cache[key]

            if old_keys:
                self.logger.debug(f"Cleaned up {len(old_keys)} old cache entries")

        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")

    def clear_cache(self):
        """Clear all cached results"""
        self.analysis_cache.clear()
        self.logger.info("Analysis cache cleared")

    def get_cached_result(self, symbol: str, parameters: dict = None) -> Optional[AnalysisResult]:
        """Get cached result if available"""
        try:
            # Create a dummy request to generate cache key
            dummy_request = AnalysisRequest(
                symbol=symbol,
                parameters=parameters or {}
            )
            cache_key = self._get_cache_key(dummy_request)

            if cache_key in self.analysis_cache:
                result = self.analysis_cache[cache_key]

                # Check if still valid (15 minutes)
                if (datetime.now() - result.timestamp).total_seconds() < 900:
                    return result
                else:
                    # Remove expired cache entry
                    del self.analysis_cache[cache_key]

            return None

        except Exception as e:
            self.logger.error(f"Error getting cached result: {e}")
            return None

    def _safe_get_options_value(self, options_data, key: str, default):
        """
        Safely get value from options_data whether it's a dictionary or OptionChain object

        Args:
            options_data: Dictionary or OptionChain object
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value from options_data or default
        """
        try:
            # If it's a dictionary, use .get()
            if hasattr(options_data, 'get') and callable(getattr(options_data, 'get')):
                return options_data.get(key, default)
            # If it's an object with the attribute, get it directly
            elif hasattr(options_data, key):
                return getattr(options_data, key)
            # Fall back to default
            else:
                return default
        except Exception as e:
            self.logger.debug(f"Error accessing options_data.{key}: {e}")
            return default
