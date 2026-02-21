"""
Analysis worker for background stock analysis processing
Professional Options Calculator - Analysis Worker
"""

import time
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import numpy as np

from .base_worker import BaseWorker
from services.market_data import MarketDataService
from services.options_service import OptionsService
from services.volatility_service import VolatilityService
from services.ml_service import MLService
from utils.monte_carlo import MonteCarloEngine
from utils.greeks_calculator import GreeksCalculator, MarketParameters, OptionType

logger = logging.getLogger(__name__)

class AnalysisWorker(BaseWorker):
    """Worker for performing comprehensive options analysis"""
    
    def __init__(self, symbols: List[str], contracts: int = 1, debit: Optional[float] = None):
        super().__init__()
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.contracts = contracts
        self.debit = debit
        
        # Initialize services with proper parameters
        from utils.config_manager import ConfigManager
        self.config_manager = ConfigManager()

        self.market_data = MarketDataService()
        self.volatility_service = VolatilityService(self.config_manager, self.market_data)
        self.ml_service = MLService(self.config_manager)
        self.greeks_calc = GreeksCalculator(self.config_manager)
        self.options_service = OptionsService(self.config_manager, self.market_data)
        self.monte_carlo = MonteCarloEngine()
        
        # Results storage
        self.results = {}
        self.errors = {}

    def get_basic_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get basic stock data for a symbol"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period='1d')

            if history.empty:
                return None

            current_price = history['Close'].iloc[-1]

            return {
                'Symbol': symbol,
                'Current Price': f'${current_price:.2f}',
                'Volume': f'{history["Volume"].iloc[-1]:,}',
                'Market Cap': info.get('marketCap', 'N/A'),
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Beta': info.get('beta', 'N/A'),
                'Sector': info.get('sector', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error getting basic data for {symbol}: {e}")
            return None

    def run_heston_simulation(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Run Heston volatility model simulation using MonteCarloEngine"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='60d')

            if hist.empty:
                return None

            current_price = hist['Close'].iloc[-1]

            # CRITICAL FIX: Calculate actual historical volatility instead of hardcoded 25%
            returns = hist['Close'].pct_change().dropna()
            historical_vol = returns.std() * np.sqrt(252)  # Annualize volatility
            logger.info(f"🔧 HESTON THETA FIX: Using calculated historical volatility {historical_vol:.1%} instead of hardcoded 25% for {symbol}")

            # Use the MonteCarloEngine's run_simulation method
            simulation_results = self.monte_carlo.run_simulation(
                symbol=symbol,
                current_price=current_price,
                historical_data=hist,
                volatility_metrics={'rv30': historical_vol, 'theta': historical_vol, 'iv30': 0.30},
                simulations=5000,
                days_to_expiration=30
            )

            if simulation_results:
                heston_params = simulation_results.get('heston_parameters', {}) or {}
                return {
                    'Model': 'Heston Stochastic Volatility',
                    'Current Price': f'${current_price:.2f}',
                    'Kappa (mean reversion)': f'{float(heston_params.get("kappa", 2.0)):.3f}',
                    'Theta (long-term var)': f'{float(heston_params.get("theta", 0.04)):.3f}',
                    'Sigma (vol of vol)': f'{float(heston_params.get("sigma", 0.3)):.3f}',
                    'Rho (correlation)': f'{float(heston_params.get("rho", -0.7)):.3f}',
                    'Initial Variance': f'{float(heston_params.get("v0", 0.04)):.3f}',
                    'Expected Move': simulation_results.get('expected_move', 'N/A'),
                    'Simulations': simulation_results.get(
                        'simulations_run',
                        simulation_results.get('num_simulations', 'N/A')
                    )
                }
            else:
                return {
                    'Model': 'Heston Stochastic Volatility',
                    'Status': 'Fallback mode - insufficient data for calibration'
                }

        except Exception as e:
            logger.error(f"Error running Heston simulation for {symbol}: {e}")
            return None

    def run_monte_carlo_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Run Monte Carlo analysis using professional MonteCarloEngine"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='60d')

            if hist.empty:
                return None

            current_price = hist['Close'].iloc[-1]

            # CRITICAL FIX: Calculate actual historical volatility instead of hardcoded 25%
            returns = hist['Close'].pct_change().dropna()
            historical_vol = returns.std() * np.sqrt(252)  # Annualize volatility
            logger.debug(
                "Monte Carlo volatility calibration for %s: historical_vol=%.4f, theta_variance=%.6f",
                symbol,
                historical_vol,
                historical_vol**2,
            )

            # Use the MonteCarloEngine for professional analysis with REAL volatility
            simulation_results = self.monte_carlo.run_simulation(
                symbol=symbol,
                current_price=current_price,
                historical_data=hist,
                volatility_metrics={'rv30': historical_vol, 'theta': historical_vol, 'iv30': 0.30},
                simulations=10000,
                days_to_expiration=30
            )

            if simulation_results:
                price_range_95 = simulation_results.get('price_range_95', {}) or {}
                heston_params = simulation_results.get('heston_parameters', {}) or {}
                jump_params = simulation_results.get('jump_parameters', {}) or {}

                prob_1x = float(simulation_results.get('prob_exceed_1x', 45.0))
                prob_1_5x = float(simulation_results.get('prob_exceed_1_5x', 25.0))
                prob_2x = float(simulation_results.get('prob_exceed_2x', 12.0))
                upside = float(simulation_results.get('upside_probability', 50.0))
                downside = float(simulation_results.get('downside_probability', 50.0))
                var_95_price = float(
                    simulation_results.get('value_at_risk_95', price_range_95.get('lower', current_price * 0.9))
                )
                upper_95_price = float(price_range_95.get('upper', current_price * 1.1))
                long_run_vol = float(np.sqrt(max(float(heston_params.get('theta', 0.04)), 1e-8)))

                if long_run_vol < 0.18:
                    vol_regime = "Low"
                elif long_run_vol < 0.32:
                    vol_regime = "Normal"
                else:
                    vol_regime = "High"

                # Extract key metrics from the professional simulation
                return {
                    'Model': 'Monte Carlo (Heston)',
                    'Simulations': simulation_results.get('simulations_run', 10000),
                    'Current Price': f'${current_price:.2f}',
                    'Expected Move': simulation_results.get('expected_move', 'N/A'),
                    'Distribution Analysis': {
                        'Prob >= 1x move': f'{prob_1x:.1f}%',
                        'Prob >= 1.5x move': f'{prob_1_5x:.1f}%',
                        'Prob >= 2x move': f'{prob_2x:.1f}%',
                        'Upside Probability': f'{upside:.1f}%',
                        'Downside Probability': f'{downside:.1f}%',
                        'Skewness': f'{float(simulation_results.get("distribution_skewness", 0.0)):.3f}',
                        'Kurtosis': f'{float(simulation_results.get("distribution_kurtosis", 0.0)):.3f}',
                    },
                    'VaR 5%': f'${var_95_price:.2f}',
                    'VaR 95%': f'${upper_95_price:.2f}',
                    'Mean Price': f'${simulation_results.get("mean_final_price", current_price):.2f}',
                    'Volatility Regime': vol_regime,
                    'Tail Risk': {
                        'Jump Enabled': bool(jump_params.get('enabled', False)),
                        'Jump Lambda': float(jump_params.get('lambda', 0.0)),
                        'Distribution Kurtosis': float(simulation_results.get('distribution_kurtosis', 0.0)),
                    }
                }
            else:
                # Fallback to simple calculation if MonteCarloEngine fails
                return {
                    'Model': 'Monte Carlo (Fallback)',
                    'Status': 'Using fallback mode - insufficient data',
                    'Current Price': f'${current_price:.2f}',
                    'Note': 'Professional Heston simulation unavailable'
                }

        except Exception as e:
            logger.error(f"Error running Monte Carlo for {symbol}: {e}")
            return None

    def calculate_kelly_sizing(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Calculate Kelly Criterion position sizing"""
        try:
            import numpy as np
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='252d')  # 1 year

            if hist.empty:
                return None

            returns = hist['Close'].pct_change().dropna()

            # Kelly Criterion calculation
            win_rate = len(returns[returns > 0]) / len(returns)
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0

            if avg_loss == 0:
                kelly_fraction = 0
            else:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

            # Cap at reasonable levels
            kelly_fraction = max(0, min(0.25, kelly_fraction))

            return {
                'Model': 'Kelly Criterion',
                'Win Rate': f'{win_rate:.2%}',
                'Average Win': f'{avg_win:.2%}',
                'Average Loss': f'{avg_loss:.2%}',
                'Kelly Fraction': f'{kelly_fraction:.2%}',
                'Recommended Position': f'{kelly_fraction * 100:.1f}% of capital'
            }
        except Exception as e:
            logger.error(f"Error calculating Kelly sizing for {symbol}: {e}")
            return None

    def get_ml_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ML prediction using comprehensive features"""
        try:
            import yfinance as yf
            import numpy as np

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='90d')

            if hist.empty:
                return None

            # Calculate comprehensive features for ML model
            current_price = hist['Close'].iloc[-1]
            returns = hist['Close'].pct_change().dropna()

            # Volatility features
            volatility = returns.std() * np.sqrt(252)
            vol_10d = hist['Close'].pct_change().rolling(10).std().iloc[-1] * np.sqrt(252)
            vol_30d = hist['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252)

            # Price momentum features
            price_change_1d = (current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
            price_change_5d = (current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]
            price_change_20d = (current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]

            # Technical indicators
            rsi = self._calculate_rsi(hist['Close'], 14)
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            price_to_sma = current_price / sma_20

            # Volume indicators
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1

            earnings_date = self.market_data.get_next_earnings(symbol)
            days_to_earnings = (earnings_date - datetime.now().date()).days if earnings_date else 30
            info = ticker.info or {}

            # Prepare features for ML service (keys aligned with MLService.prepare_features)
            ml_features = {
                'iv30_rv30': 1.2,
                'ts_slope_0_45': 0.0,
                'days_to_earnings': max(0, int(days_to_earnings)),
                'vix': self.market_data.get_vix(),
                'avg_volume': float(avg_volume) if not np.isnan(avg_volume) else 0.0,
                'gamma': 0.01,
                'sector': info.get('sector', 'Unknown'),
                'iv_rank': 0.5,
                'iv_percentile': 50.0,
                'short_theta': -1.0,
                'long_theta': -0.5,
                'option_volume': float(hist['Volume'].iloc[-1]) if len(hist) else 0.0,
                'open_interest': 0.0,
                'bid_ask_spread': 0.1,
                'call_iv': max(0.01, float(vol_30d) if not np.isnan(vol_30d) else 0.25),
                'put_iv': max(0.01, float(vol_30d * 1.05) if not np.isnan(vol_30d) else 0.26),
                'put_call_ratio': 1.0
            }

            # Get prediction from ML service
            prediction_result = self.ml_service.predict_trade_outcome(ml_features)
            confidence_value = getattr(prediction_result.confidence, 'value', str(prediction_result.confidence))

            return {
                'Model': 'ML Prediction Engine',
                'Symbol': symbol,
                'Prediction Probability': f'{prediction_result.probability:.1%}',
                'Confidence': confidence_value,
                'Risk Score': f'{prediction_result.risk_score:.2f}',
                'Key Features': {
                    'Volatility': f'{volatility:.1%}',
                    'RSI': f'{rsi:.1f}',
                    'Price Momentum (5d)': f'{price_change_5d:.1%}',
                    'Volume Ratio': f'{volume_ratio:.2f}'
                },
                'Recommendation': self._generate_ml_recommendation(
                    prediction_result.probability,
                    confidence_value,
                    volatility
                ),
                'Model Version': 'Professional ML v3.0'
            }

        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            # Return a basic prediction based on available data
            return {
                'Model': 'ML Prediction (Fallback)',
                'Status': 'Limited analysis - using fallback mode',
                'Note': 'Professional ML model unavailable'
            }

    def _generate_ml_recommendation(self, probability: float, confidence: Any, volatility: float) -> str:
        """Generate trading recommendation based on ML output"""
        confidence_str = getattr(confidence, 'value', str(confidence)).lower()
        if probability > 0.7 and confidence_str in ['high', 'very_high']:
            return 'STRONG BUY'
        elif probability > 0.6:
            return 'BUY'
        elif probability > 0.4:
            if volatility > 0.3:  # High volatility
                return 'HOLD (High Vol)'
            else:
                return 'HOLD'
        elif probability > 0.3:
            return 'SELL'
        else:
            return 'STRONG SELL'

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except:
            return 50.0
    
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
                # CRITICAL FIX: Try direct yfinance fallback instead of hardcoded 25%
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    direct_hist = ticker.history(period='60d')
                    if not direct_hist.empty:
                        returns = direct_hist['Close'].pct_change().dropna()
                        historical_vol = returns.std() * np.sqrt(252)  # Annualize volatility
                        logger.info(f"🔧 FALLBACK THETA FIX: Using direct yfinance historical volatility {historical_vol:.1%} instead of hardcoded 25% for {symbol}")
                        return {
                            'rv30': historical_vol,
                            'iv30': historical_vol * 1.2,
                            'iv_rv_ratio': 1.2,
                            'iv_rank': 0.5,
                            'iv_percentile': 50.0,
                            'theta': historical_vol  # Add theta for Monte Carlo
                        }
                except Exception as e:
                    logger.warning(f"Direct yfinance fallback failed: {e}")

                # Use default values ONLY if yfinance also fails
                return {
                    'rv30': 0.28,  # More realistic default than 25%
                    'iv30': 0.34,  # 28% * 1.2
                    'iv_rv_ratio': 1.2,
                    'iv_rank': 0.5,
                    'iv_percentile': 50.0,
                    'theta': 0.28  # Add theta for Monte Carlo
                }
            
            # Calculate realized volatility - FIXED: Pass symbol string, not DataFrame
            rv30 = self.volatility_service.calculate_realized_volatility(symbol, window=30)
            
            # Estimate implied volatility (this would need actual IV data in production)
            try:
                if hasattr(self.volatility_service, 'estimate_implied_volatility'):
                    iv30 = self.volatility_service.estimate_implied_volatility(symbol, hist_data)
                elif hasattr(self.volatility_service, '_estimate_implied_volatility'):
                    iv30 = self.volatility_service._estimate_implied_volatility(symbol)
                else:
                    # Fallback: estimate as RV * 1.2 (realistic premium)
                    iv30 = rv30 * 1.2 if rv30 > 0 else 0.30
                    logger.warning(f"🔧 IV ESTIMATION FIX: Using fallback IV calculation (RV * 1.2 = {iv30:.1%}) for {symbol}")
            except Exception as e:
                # Robust fallback for IV estimation
                iv30 = rv30 * 1.2 if rv30 > 0 else 0.30
                logger.warning(f"🔧 IV ESTIMATION ERROR FIX: Using fallback IV calculation due to error: {e}")
            
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

            # Calculate real volatility as fallback using same method as analysis_view.py
            try:
                import yfinance as yf
                import numpy as np
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1y')
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    if len(returns) > 30:
                        # Use exact same calculation as analysis_view.py:149
                        historical_vol = returns.std() * np.sqrt(252)
                        logger.info(f"Calculated real historical volatility for {symbol}: {historical_vol:.1%}")
                        return {
                            'rv30': historical_vol,  # Use real calculation not 25% hardcode
                            'iv30': historical_vol * 1.2,
                            'iv_rv_ratio': 1.2,
                            'iv_rank': 0.5,
                            'iv_percentile': 50.0,
                            'theta': historical_vol  # Add theta for Monte Carlo
                        }
            except Exception as fallback_error:
                logger.warning(f"Fallback volatility calculation failed: {fallback_error}")

            # Ultimate fallback - use more realistic defaults than 25%
            logger.warning(f"🔧 ULTIMATE FALLBACK THETA FIX: Using realistic 28% default instead of hardcoded 25% for {symbol}")
            return {
                'rv30': 0.28,  # More realistic than 25%
                'iv30': 0.34,  # 28% * 1.2
                'iv_rv_ratio': 1.2,
                'iv_rank': 0.5,
                'iv_percentile': 50.0,
                'theta': 0.28  # Use 28% for Monte Carlo instead of 25%
            }
    
    def _run_monte_carlo(self, symbol: str, current_price: float, vol_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for price projections"""
        try:
            # Get historical data for Monte Carlo
            historical_data = self.market_data.get_historical_data(symbol, period="6mo")
            if historical_data.empty:
                # Use empty DataFrame with minimal required structure
                import pandas as pd
                historical_data = pd.DataFrame({'Close': [current_price]})

            # Get parameters - use theta if available, otherwise rv30
            volatility = vol_metrics.get('theta', vol_metrics.get('rv30', 0.25))
            days_to_expiry = 30  # This should come from options data

            # Log the volatility being used for debugging
            logger.info(f"Monte Carlo using volatility: {volatility:.1%} for {symbol}")

            # Run simulation with required parameters
            results = self.monte_carlo.run_simulation(
                symbol=symbol,
                current_price=current_price,
                historical_data=historical_data,
                volatility_metrics=vol_metrics,
                simulations=10000,
                days_to_expiration=days_to_expiry
            )
            
            # Normalize keys so downstream logic can use either legacy or new names
            if isinstance(results, dict):
                prob_exceed_1x = results.get('prob_exceed_1x', results.get('prob_1x', 45.0))
                prob_exceed_1_5x = results.get('prob_exceed_1_5x', results.get('prob_1_5x', 25.0))
                prob_exceed_2x = results.get('prob_exceed_2x', results.get('prob_2x', 15.0))
                results.setdefault('prob_exceed_1x', prob_exceed_1x)
                results.setdefault('prob_exceed_1_5x', prob_exceed_1_5x)
                results.setdefault('prob_exceed_2x', prob_exceed_2x)
                results.setdefault('prob_1x', prob_exceed_1x)
                results.setdefault('prob_1_5x', prob_exceed_1_5x)
                results.setdefault('prob_2x', prob_exceed_2x)

            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo for {symbol}: {e}")
            return {
                'prob_exceed_1x': 45.0,
                'prob_exceed_1_5x': 25.0,
                'prob_exceed_2x': 15.0,
                'prob_1x': 45.0,
                'prob_1_5x': 25.0,
                'prob_2x': 15.0,
                'upside_probability': 50.0,
                'downside_probability': 50.0,
                'expected_return': 0.0,
                'value_at_risk_95': current_price * 0.9
            }
    
    def _calculate_greeks(self, symbol: str, current_price: float, options_data: Dict[str, Any], vol_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate option Greeks for the calendar spread"""
        try:
            if not options_data:
                return self._default_greeks()
            
            # Extract parameters
            strike = round(current_price / 5) * 5  # Round to nearest $5
            iv = vol_metrics.get('iv30', 0.25)
            try:
                iv = float(iv)
            except (TypeError, ValueError):
                iv = 0.25
            
            # Calculate time to expiration
            short_exp = options_data['short_expiration']
            long_exp = options_data['long_expiration']
            
            short_dte = self._calculate_days_to_expiry(short_exp)
            long_dte = self._calculate_days_to_expiry(long_exp)
            
            short_time = max(short_dte / 365.25, 1 / 365.25)
            long_time = max(long_dte / 365.25, short_time + (7 / 365.25))

            short_params = MarketParameters(
                spot_price=current_price,
                strike_price=strike,
                time_to_expiry=short_time,
                risk_free_rate=0.05,
                volatility=max(iv, 0.01),
                option_type=OptionType.PUT
            )
            long_params = MarketParameters(
                spot_price=current_price,
                strike_price=strike,
                time_to_expiry=long_time,
                risk_free_rate=0.05,
                volatility=max(iv, 0.01),
                option_type=OptionType.PUT
            )

            spread_greeks = self.greeks_calc.calculate_calendar_spread_greeks(
                short_params,
                long_params
            )

            return {
                'net_delta': spread_greeks.net_delta,
                'net_gamma': spread_greeks.net_gamma,
                'net_theta': spread_greeks.net_theta,
                'net_vega': spread_greeks.net_vega,
                'net_rho': spread_greeks.net_rho,
                'time_decay_ratio': spread_greeks.time_decay_ratio,
                'theta_efficiency': spread_greeks.theta_efficiency,
                'display': {
                    'net_delta': f"Net Δ: {spread_greeks.net_delta:.4f}",
                    'net_gamma': f"Net Γ: {spread_greeks.net_gamma:.4f}",
                    'net_theta': f"Net Θ: {spread_greeks.net_theta:.4f}",
                    'net_vega': f"Net ν: {spread_greeks.net_vega:.4f}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks for {symbol}: {e}")
            return self._default_greeks()
    
    def _get_earnings_data(self, symbol: str) -> Dict[str, Any]:
        """Get earnings-related data"""
        try:
            earnings_date = self.market_data.get_next_earnings(symbol)
            
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
            prob_1x = monte_carlo_results.get('prob_exceed_1x', monte_carlo_results.get('prob_1x', 45.0))
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
                'iv30_rv30': vol_metrics.get('iv_rv_ratio', 1.0),
                'ts_slope_0_45': vol_metrics.get('term_structure_slope', 0.0),
                'days_to_earnings': earnings_data.get('days_to_earnings', 90),
                'vix': market_data.get('vix', 20.0),
                'avg_volume': vol_metrics.get('avg_volume', 0.0),
                'gamma': 0.01,
                'sector': 'Unknown',
                'iv_rank': vol_metrics.get('iv_rank', 0.5),
                'iv_percentile': vol_metrics.get('iv_percentile', 50.0),
                'short_theta': -1.0,
                'long_theta': -0.7,
                'option_volume': 0.0,
                'open_interest': 0.0,
                'bid_ask_spread': 0.1,
                'call_iv': vol_metrics.get('iv30', 0.25),
                'put_iv': vol_metrics.get('iv30', 0.25),
                'put_call_ratio': 1.0
            }
            
            prediction = self.ml_service.predict_trade_outcome(features)
            
            return {
                'prediction_probability': float(getattr(prediction, 'probability', 0.5)),
                'confidence': getattr(getattr(prediction, 'confidence', None), 'value', 'moderate'),
                'risk_score': float(getattr(prediction, 'risk_score', 0.5)),
                'model_recommendation': str(getattr(prediction, 'recommendation', 'HOLD')),
                'confidence_adjustment': (float(getattr(prediction, 'probability', 0.5)) - 0.5) * 20
            }
            
        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return {
                'prediction_probability': 0.5,
                'confidence': 'moderate',
                'risk_score': 0.5,
                'model_recommendation': 'HOLD',
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
