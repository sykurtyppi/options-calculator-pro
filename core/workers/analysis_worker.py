"""
Analysis worker for background stock analysis processing
Professional Options Calculator - Analysis Worker
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from .base_worker import BaseWorker
from services.market_data import MarketDataService
from services.options_service import OptionsService
from services.volatility_service import VolatilityService
from services.ml_service import MLService
from utils.monte_carlo import MonteCarloEngine
from utils.greeks_calculator import GreeksCalculator

logger = logging.getLogger(__name__)

class AnalysisWorker(BaseWorker):
    """Worker for performing comprehensive options analysis"""
    
    def __init__(self, symbols: List[str], contracts: int = 1, debit: Optional[float] = None):
        super().__init__()
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.contracts = contracts
        self.debit = debit
        
        # Initialize services
        self.market_data = MarketDataService()
        self.options_service = OptionsService()
        self.volatility_service = VolatilityService()
        self.ml_service = MLService()
        self.monte_carlo = MonteCarloEngine()
        self.greeks_calc = GreeksCalculator()
        
        # Results storage
        self.results = {}
        self.errors = {}
    
    def run(self):
        """Main analysis execution"""
        try:
            self.started.emit()
            self.emit_status("Starting comprehensive analysis...")
            
            if not self.symbols:
                self.emit_error("Invalid Input", "No valid symbols provided")
                return
            
            total_symbols = len(self.symbols)
            successful_analyses = 0
            
            for i, symbol in enumerate(self.symbols):
                if self.is_cancelled():
                    self.emit_status("Analysis cancelled by user")
                    break
                
                try:
                    # Update progress
                    progress = int((i / total_symbols) * 90)  # Reserve 10% for final processing
                    self.emit_progress(progress, f"Analyzing {symbol}...")
                    
                    # Perform analysis for this symbol
                    result = self._analyze_symbol(symbol)
                    
                    if result:
                        self.results[symbol] = result
                        successful_analyses += 1
                        
                        # Emit individual result
                        self.emit_result({
                            'type': 'individual_result',
                            'symbol': symbol,
                            'result': result
                        })
                        
                        self.emit_status(f"{symbol}: Analysis complete - {result.get('confidence', 0):.1f}% confidence")
                    else:
                        self.errors[symbol] = "Analysis failed"
                        self.emit_status(f"{symbol}: Analysis failed")
                        
                except Exception as e:
                    error_msg = f"Error analyzing {symbol}: {str(e)}"
                    self.errors[symbol] = error_msg
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    self.emit_status(f"{symbol}: Error - {str(e)}")
                    
                # Small delay to prevent overwhelming APIs
                if not self.is_cancelled():
                    time.sleep(0.1)
            
            # Final processing
            self.emit_progress(95, "Finalizing results...")
            
            # Emit summary result
            summary = {
                'type': 'analysis_summary',
                'total_symbols': total_symbols,
                'successful': successful_analyses,
                'failed': len(self.errors),
                'results': self.results,
                'errors': self.errors,
                'timestamp': datetime.now().isoformat()
            }
            
            self.emit_result(summary)
            self.emit_progress(100, f"Analysis complete: {successful_analyses}/{total_symbols} successful")
            
        except Exception as e:
            logger.error(f"Critical error in analysis worker: {e}\n{traceback.format_exc()}")
            self.emit_error("Analysis Failed", f"Critical error: {str(e)}")
        finally:
            self.finished.emit()
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Perform comprehensive analysis for a single symbol"""
        try:
            # Step 1: Get basic market data
            self.emit_status(f"{symbol}: Fetching market data...")
            
            current_price = self.market_data.get_current_price(symbol)
            if not current_price or current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}")
                return None
            
            # Step 2: Get options data
            self.emit_status(f"{symbol}: Retrieving options data...")
            
            options_data = self._get_options_data(symbol)
            if not options_data:
                logger.warning(f"No options data for {symbol}")
                return None
            
            # Step 3: Calculate volatility metrics
            self.emit_status(f"{symbol}: Calculating volatility...")
            
            volatility_metrics = self._calculate_volatility_metrics(symbol, current_price)
            
            # Step 4: Run Monte Carlo simulation
            self.emit_status(f"{symbol}: Running Monte Carlo simulation...")
            
            monte_carlo_results = self._run_monte_carlo(symbol, current_price, volatility_metrics)
            
            # Step 5: Calculate Greeks
            self.emit_status(f"{symbol}: Calculating Greeks...")
            
            greeks_data = self._calculate_greeks(symbol, current_price, options_data, volatility_metrics)
            
            # Step 6: Get earnings and market data
            self.emit_status(f"{symbol}: Getting earnings and market data...")
            
            earnings_data = self._get_earnings_data(symbol)
            market_data = self._get_market_context()
            
            # Step 7: Calculate confidence score
            self.emit_status(f"{symbol}: Calculating confidence score...")
            
            confidence_data = self._calculate_confidence(
                symbol, current_price, volatility_metrics, 
                monte_carlo_results, earnings_data, market_data
            )
            
            # Step 8: Get ML prediction
            ml_prediction = self._get_ml_prediction(symbol, volatility_metrics, earnings_data, market_data)
            
            # Step 9: Compile final result
            result = self._compile_result(
                symbol=symbol,
                current_price=current_price,
                options_data=options_data,
                volatility_metrics=volatility_metrics,
                monte_carlo_results=monte_carlo_results,
                greeks_data=greeks_data,
                earnings_data=earnings_data,
                market_data=market_data,
                confidence_data=confidence_data,
                ml_prediction=ml_prediction
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in symbol analysis for {symbol}: {e}")
            return None
    
    def _get_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options chain data for analysis"""
        try:
            # Get available expirations
            expirations = self.options_service.get_available_expirations(symbol, timeout=30)
            if not expirations or len(expirations) < 2:
                return None
            
            # Filter to get near-term and next expiration
            filtered_exps = self.options_service.filter_expirations_for_calendar(expirations)
            if len(filtered_exps) < 2:
                return None
            
            # Get option chains for both expirations
            short_exp, long_exp = filtered_exps[:2]
            
            short_chain = self.options_service.get_option_chain(symbol, short_exp, timeout=30)
            long_chain = self.options_service.get_option_chain(symbol, long_exp, timeout=30)
            
            if not short_chain or not long_chain:
                return None
            
            return {
                'short_expiration': short_exp,
                'long_expiration': long_exp,
                'short_chain': short_chain,
                'long_chain': long_chain,
                'all_expirations': expirations
            }
            
        except Exception as e:
            logger.error(f"Error getting options data for {symbol}: {e}")
            return None
    
    def _calculate_volatility_metrics(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics"""
        try:
            # Get historical data
            hist_data = self.market_data.get_historical_data(symbol, period="1y")
            
            if hist_data.empty:
                # Use default values
                return {
                    'rv30': 0.25,
                    'iv30': 0.30,
                    'iv_rv_ratio': 1.2,
                    'iv_rank': 0.5,
                    'iv_percentile': 50.0
                }
            
            # Calculate realized volatility
            rv30 = self.volatility_service.calculate_realized_volatility(hist_data, window=30)
            
            # Estimate implied volatility (this would need actual IV data in production)
            iv30 = self.volatility_service.estimate_implied_volatility(symbol, hist_data)
            
            # Calculate metrics
            iv_rv_ratio = iv30 / rv30 if rv30 > 0 else 1.0
            iv_rank = self.volatility_service.calculate_iv_rank(symbol, iv30)
            iv_percentile = self.volatility_service.calculate_iv_percentile(symbol, iv30)
            
            return {
                'rv30': rv30,
                'iv30': iv30,
                'iv_rv_ratio': iv_rv_ratio,
                'iv_rank': iv_rank,
                'iv_percentile': iv_percentile,
                'historical_data': hist_data
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return {
                'rv30': 0.25,
                'iv30': 0.30,
                'iv_rv_ratio': 1.2,
                'iv_rank': 0.5,
                'iv_percentile': 50.0
            }
    
    def _run_monte_carlo(self, symbol: str, current_price: float, vol_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for price projections"""
        try:
            # Get parameters
            volatility = vol_metrics.get('rv30', 0.25)
            days_to_expiry = 30  # This should come from options data
            
            # Run simulation
            results = self.monte_carlo.run_heston_simulation(
                current_price=current_price,
                volatility=volatility,
                days=days_to_expiry,
                num_simulations=10000
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo for {symbol}: {e}")
            return {
                'prob_1x': 45.0,
                'prob_1_5x': 25.0,
                'prob_2x': 15.0,
                'upside_prob': 50.0,
                'downside_prob': 50.0,
                'expected_return': 0.0,
                'var_95': current_price * 0.9
            }
    
    def _calculate_greeks(self, symbol: str, current_price: float, options_data: Dict[str, Any], vol_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate option Greeks for the calendar spread"""
        try:
            if not options_data:
                return self._default_greeks()
            
            # Extract parameters
            strike = round(current_price / 5) * 5  # Round to nearest $5
            iv = vol_metrics.get('iv30', 0.25)
            
            # Calculate time to expiration
            short_exp = options_data['short_expiration']
            long_exp = options_data['long_expiration']
            
            short_dte = self._calculate_days_to_expiry(short_exp)
            long_dte = self._calculate_days_to_expiry(long_exp)
            
            # Calculate Greeks
            greeks = self.greeks_calc.calculate_calendar_spread_greeks(
                spot_price=current_price,
                strike_price=strike,
                short_dte=short_dte,
                long_dte=long_dte,
                volatility=iv,
                risk_free_rate=0.05
            )
            
            return greeks
            
        except Exception as e:
            logger.error(f"Error calculating Greeks for {symbol}: {e}")
            return self._default_greeks()
    
    def _get_earnings_data(self, symbol: str) -> Dict[str, Any]:
        """Get earnings-related data"""
        try:
            earnings_date = self.market_data.get_next_earnings_date(symbol)
            
            if earnings_date:
                days_to_earnings = (earnings_date - datetime.now().date()).days
            else:
                days_to_earnings = 90  # Default
            
            return {
                'earnings_date': earnings_date,
                'days_to_earnings': days_to_earnings,
                'earnings_expected': 0 <= days_to_earnings <= 14
            }
            
        except Exception as e:
            logger.error(f"Error getting earnings data for {symbol}: {e}")
            return {
                'earnings_date': None,
                'days_to_earnings': 90,
                'earnings_expected': False
            }
    
    def _get_market_context(self) -> Dict[str, Any]:
        """Get broader market context"""
        try:
            vix = self.market_data.get_vix()
            
            return {
                'vix': vix,
                'market_regime': self._determine_market_regime(vix),
                'volatility_environment': self._determine_vol_environment(vix)
            }
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {
                'vix': 20.0,
                'market_regime': 'NORMAL',
                'volatility_environment': 'MODERATE'
            }
    
    def _calculate_confidence(self, symbol: str, current_price: float, vol_metrics: Dict[str, Any], 
                            monte_carlo_results: Dict[str, Any], earnings_data: Dict[str, Any], 
                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence score and breakdown"""
        try:
            base_confidence = 50.0
            
            # IV/RV ratio component
            iv_rv_ratio = vol_metrics.get('iv_rv_ratio', 1.0)
            iv_rv_score = 15 if iv_rv_ratio >= 1.5 else 10 if iv_rv_ratio >= 1.25 else -10
            
            # Monte Carlo component
            prob_1x = monte_carlo_results.get('prob_1x', 45.0)
            mc_score = (prob_1x - 45.0) * 0.5
            
            # Earnings timing
            days_to_earnings = earnings_data.get('days_to_earnings', 90)
            earnings_score = 15 if 0 <= days_to_earnings <= 7 else 0
            
            # Market regime
            vix = market_data.get('vix', 20.0)
            market_score = -10 if vix > 30 else 5 if vix < 20 else 0
            
            # Final confidence
            confidence = base_confidence + iv_rv_score + mc_score + earnings_score + market_score
            confidence = max(0, min(100, confidence))
            
            return {
                'confidence': confidence,
                'score_breakdown': {
                    'base': base_confidence,
                    'iv_rv': iv_rv_score,
                    'monte_carlo': mc_score,
                    'earnings_timing': earnings_score,
                    'market_regime': market_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence for {symbol}: {e}")
            return {'confidence': 50.0, 'score_breakdown': {}}
    
    def _get_ml_prediction(self, symbol: str, vol_metrics: Dict[str, Any], 
                          earnings_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get machine learning prediction"""
        try:
            features = {
                'iv_rv_ratio': vol_metrics.get('iv_rv_ratio', 1.0),
                'days_to_earnings': earnings_data.get('days_to_earnings', 90),
                'vix': market_data.get('vix', 20.0),
                'iv_rank': vol_metrics.get('iv_rank', 0.5)
            }
            
            prediction = self.ml_service.predict_trade_outcome(features)
            
            return {
                'prediction_probability': prediction,
                'confidence_adjustment': (prediction - 0.5) * 20
            }
            
        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return {
                'prediction_probability': 0.5,
                'confidence_adjustment': 0
            }
    
    def _compile_result(self, **kwargs) -> Dict[str, Any]:
        """Compile final analysis result"""
        symbol = kwargs['symbol']
        current_price = kwargs['current_price']
        confidence_data = kwargs['confidence_data']
        
        # Build comprehensive result dictionary
        result = {
            'ticker': symbol,
            'underlying_price': current_price,
            'confidence': confidence_data.get('confidence', 50.0),
            'timestamp': datetime.now().isoformat(),
            
            # Include all analysis components
            'volatility_metrics': kwargs['volatility_metrics'],
            'monte_carlo_results': kwargs['monte_carlo_results'],
            'greeks_data': kwargs['greeks_data'],
            'earnings_data': kwargs['earnings_data'],
            'market_data': kwargs['market_data'],
            'ml_prediction': kwargs['ml_prediction'],
            
            # Trading parameters
            'contracts': self.contracts,
            'debit': self.debit,
            
            # Additional calculated fields for UI
            'expected_move': self._calculate_expected_move(kwargs),
            'max_loss': self._calculate_max_loss(kwargs),
            'profit_target': self._calculate_profit_target(kwargs),
            'recommendation': self._generate_recommendation(confidence_data.get('confidence', 50.0))
        }
        
        return result
    
    def _calculate_days_to_expiry(self, expiry_string: str) -> int:
        """Calculate days to expiry from date string"""
        try:
            expiry_date = datetime.strptime(expiry_string, "%Y-%m-%d").date()
            return (expiry_date - datetime.now().date()).days
        except:
            return 30
    
    def _determine_market_regime(self, vix: float) -> str:
        """Determine market regime based on VIX"""
        if vix < 16:
            return "LOW_VOLATILITY"
        elif vix < 24:
            return "NORMAL"
        elif vix < 32:
            return "ELEVATED"
        else:
            return "HIGH_VOLATILITY"
    
    def _determine_vol_environment(self, vix: float) -> str:
        """Determine volatility environment"""
        if vix < 18:
            return "LOW"
        elif vix < 28:
            return "MODERATE"
        else:
            return "HIGH"
    
    def _default_greeks(self) -> Dict[str, Any]:
        """Return default Greeks when calculation fails"""
        return {
            'net_delta': 0.0,
            'net_gamma': 0.0,
            'net_theta': 0.0,
            'net_vega': 0.0,
            'display': {
                'net_delta': 'Net Δ: N/A',
                'net_gamma': 'Net Γ: N/A',
                'net_theta': 'Net Θ: N/A',
                'net_vega': 'Net ν: N/A'
            }
        }
    
    def _calculate_expected_move(self, data: Dict[str, Any]) -> str:
        """Calculate expected move percentage"""
        try:
            vol_metrics = data['volatility_metrics']
            iv = vol_metrics.get('iv30', 0.25)
            move_pct = iv * (30 / 365) ** 0.5 * 100
            return f"{move_pct:.1f}%"
        except:
            return "N/A"
    
    def _calculate_max_loss(self, data: Dict[str, Any]) -> str:
        """Calculate maximum loss estimate"""
        try:
            if self.debit:
                max_loss = self.debit * 100 * self.contracts
                return f"${max_loss:.2f}"
            else:
                # Estimate based on price
                current_price = data['current_price']
                estimated_debit = current_price * 0.02  # 2% of stock price
                max_loss = estimated_debit * 100 * self.contracts
                return f"~${max_loss:.2f}"
        except:
            return "N/A"
    
    def _calculate_profit_target(self, data: Dict[str, Any]) -> str:
        """Calculate profit target estimate"""
        try:
            # Simplified profit target calculation
            if self.debit:
                profit_target = self.debit * 50 * self.contracts  # 50% of debit
                return f"${profit_target:.2f}"
            else:
                return "N/A"
        except:
            return "N/A"
    
    def _generate_recommendation(self, confidence: float) -> str:
        """Generate trading recommendation"""
        if confidence >= 75:
            return "STRONG BUY"
        elif confidence >= 60:
            return "BUY"
        elif confidence >= 45:
            return "CONSIDER"
        else:
            return "AVOID"