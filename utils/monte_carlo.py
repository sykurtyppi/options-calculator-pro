"""
Monte Carlo Engine - Professional Options Calculator Pro
Advanced Monte Carlo simulations using Heston model with optimizations
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class HestonParameters:
    """Heston model parameters"""
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    sigma: float  # Volatility of volatility
    rho: float    # Correlation between price and volatility
    v0: float     # Initial variance


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation"""
    num_simulations: int = 10000
    time_steps: int = 50
    use_antithetic: bool = True
    use_control_variate: bool = True
    random_seed: Optional[int] = None


class MonteCarloEngine:
    """
    Professional Monte Carlo engine with Heston stochastic volatility model
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Model cache for performance
        self._parameter_cache = {}
        self._simulation_cache = {}
        
        self.logger.info(f"Monte Carlo engine initialized (Numba: {NUMBA_AVAILABLE})")
    
    def run_simulation(self, symbol: str, current_price: float, 
                      historical_data: pd.DataFrame, volatility_metrics: dict,
                      simulations: int = 10000, days_to_expiration: int = 30,
                      **kwargs) -> Dict[str, Any]:
        """
        Run complete Monte Carlo simulation
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            historical_data: Historical price data
            volatility_metrics: Volatility analysis results
            simulations: Number of simulations
            days_to_expiration: Days to option expiration
            
        Returns:
            Dictionary with simulation results
        """
        try:
            start_time = datetime.now()
            
            # Calibrate Heston parameters
            heston_params = self._calibrate_heston_parameters(
                symbol, historical_data, volatility_metrics
            )
            
            # Setup simulation configuration
            config = SimulationConfig(
                num_simulations=simulations,
                time_steps=min(days_to_expiration, 50),
                use_antithetic=simulations >= 5000,
                use_control_variate=True,
                random_seed=42 if simulations <= 1000 else None  # Reproducible for small tests
            )
            
            # Run Heston simulation
            final_prices = self._run_heston_simulation(
                S0=current_price,
                T=days_to_expiration / 365.0,
                heston_params=heston_params,
                config=config
            )
            
            # Calculate expected move (straddle approximation)
            expected_move = self._estimate_expected_move(
                current_price, volatility_metrics, days_to_expiration
            )
            
            # Analyze price distribution
            results = self._analyze_price_distribution(
                final_prices, current_price, expected_move
            )
            
            # Add metadata
            duration = (datetime.now() - start_time).total_seconds()
            results.update({
                'symbol': symbol,
                'current_price': current_price,
                'expected_move': expected_move,
                'simulations_run': len(final_prices),
                'computation_time': duration,
                'heston_parameters': {
                    'kappa': heston_params.kappa,
                    'theta': heston_params.theta,
                    'sigma': heston_params.sigma,
                    'rho': heston_params.rho,
                    'v0': heston_params.v0
                },
                'price_distribution': final_prices.tolist() if len(final_prices) <= 1000 else None
            })
            
            self.logger.info(f"Monte Carlo simulation completed for {symbol} in {duration:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed for {symbol}: {e}")
            return self._create_fallback_result(current_price, simulations)
    
    def _calibrate_heston_parameters(self, symbol: str, historical_data: pd.DataFrame,
                                   volatility_metrics: dict) -> HestonParameters:
        """Calibrate Heston model parameters from historical data"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{len(historical_data)}_{hash(str(volatility_metrics))}"
            if cache_key in self._parameter_cache:
                return self._parameter_cache[cache_key]
            
            # Calculate returns
            returns = historical_data['Close'].pct_change().dropna()
            
            if len(returns) < 30:
                # Insufficient data, use default parameters
                params = self._get_default_heston_parameters(volatility_metrics)
            else:
                # Estimate parameters from data
                params = self._estimate_heston_parameters(returns, volatility_metrics)
            
            # Cache the parameters
            self._parameter_cache[cache_key] = params
            
            return params
            
        except Exception as e:
            self.logger.warning(f"Parameter calibration failed for {symbol}: {e}")
            return self._get_default_heston_parameters(volatility_metrics)
    
    def _estimate_heston_parameters(self, returns: pd.Series,
                                  volatility_metrics: dict) -> HestonParameters:
        """Estimate Heston parameters from return series"""
        # Calculate basic statistics
        mean_return = returns.mean() * 252  # Annualized
        return_vol = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Initial variance (current IV squared)
        iv30 = volatility_metrics.get('iv30', return_vol)
        v0 = iv30 ** 2
        
        # Long-term variance (use realized volatility as proxy)
        rv30 = volatility_metrics.get('rv30', return_vol)
        theta = rv30 ** 2
        
        # Mean reversion speed (calibrated to typical values)
        # Higher for more mean-reverting volatility
        vol_of_vol = returns.rolling(window=20).std().std() * np.sqrt(252)
        kappa = max(0.5, min(5.0, 2.0 / (1 + vol_of_vol)))
        
        # Volatility of volatility
        # Estimated from volatility clustering
        sigma = max(0.1, min(1.0, vol_of_vol * 2))
        
        # Correlation between returns and volatility
        # Calculate empirical correlation if possible
        if len(returns) >= 60:
            # Rolling volatility
            rolling_vol = returns.rolling(window=20).std()
            vol_changes = rolling_vol.pct_change().dropna()
            return_aligned = returns.loc[vol_changes.index]
            
            if len(vol_changes) > 10:
                correlation = return_aligned.corr(vol_changes)
                rho = max(-0.95, min(-0.1, correlation if not np.isnan(correlation) else -0.7))
            else:
                rho = -0.7  # Default negative correlation
        else:
            rho = -0.7  # Default
        
        return HestonParameters(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            v0=v0
        )
    
    def _get_default_heston_parameters(self, volatility_metrics: dict) -> HestonParameters:
        """Get default Heston parameters when calibration isn't possible"""
        iv30 = volatility_metrics.get('iv30', 0.25)
        vix = volatility_metrics.get('vix_level', 20.0) / 100
        
        return HestonParameters(
            kappa=2.0,  # Moderate mean reversion
            theta=max(0.01, min(0.09, vix ** 2)),  # Long-term variance based on VIX
            sigma=0.3,  # Moderate vol of vol
            rho=-0.7,   # Typical negative correlation
            v0=max(0.01, iv30 ** 2)  # Current variance
        )
    
    def _run_heston_simulation(self, S0: float, T: float,
                             heston_params: HestonParameters,
                             config: SimulationConfig) -> np.ndarray:
        """Run Heston stochastic volatility simulation"""
        try:
            if NUMBA_AVAILABLE and config.num_simulations >= 1000:
                # Use optimized Numba version for large simulations
                return self._run_heston_numba(S0, T, heston_params, config)
            else:
                # Use standard numpy version
                return self._run_heston_numpy(S0, T, heston_params, config)
                
        except Exception as e:
            self.logger.error(f"Heston simulation failed: {e}")
            # Fallback to simple geometric Brownian motion
            return self._run_gbm_fallback(S0, T, heston_params.v0, config)
    
    def _run_heston_numpy(self, S0: float, T: float,
                         params: HestonParameters,
                         config: SimulationConfig) -> np.ndarray:
        """Standard numpy implementation of Heston model"""
        # Simulation parameters
        n_sims = config.num_simulations
        n_steps = config.time_steps
        dt = T / n_steps
        r = 0.03  # Risk-free rate
        
        # Antithetic variates for variance reduction
        if config.use_antithetic:
            n_base_sims = n_sims // 2
        else:
            n_base_sims = n_sims
        
        # Initialize arrays
        S = np.zeros((n_base_sims, n_steps + 1))
        V = np.zeros((n_base_sims, n_steps + 1))
        
        # Initial conditions
        S[:, 0] = S0
        V[:, 0] = params.v0
        
        # Pre-generate random numbers for better performance
        if config.random_seed:
            np.random.seed(config.random_seed)
        
        Z1 = np.random.standard_normal((n_base_sims, n_steps))
        Z2 = np.random.standard_normal((n_base_sims, n_steps))
        
        # Apply correlation
        Z2_corr = params.rho * Z1 + np.sqrt(1 - params.rho**2) * Z2
        
        # Simulation loop
        for t in range(n_steps):
            # Variance process (with truncation to ensure non-negativity)
            V[:, t+1] = np.maximum(
                V[:, t] + params.kappa * (params.theta - V[:, t]) * dt +
                params.sigma * np.sqrt(V[:, t] * dt) * Z2_corr[:, t],
                0.001  # Minimum variance
            )
            
            # Price process
            S[:, t+1] = S[:, t] * np.exp(
                (r - 0.5 * V[:, t]) * dt + np.sqrt(V[:, t] * dt) * Z1[:, t]
            )
        
        # Final prices
        final_prices = S[:, -1]
        
        # Apply antithetic variates
        if config.use_antithetic:
            # Rerun with negated random numbers
            Z1_anti = -Z1
            Z2_anti = -Z2
            Z2_corr_anti = params.rho * Z1_anti + np.sqrt(1 - params.rho**2) * Z2_anti
            
            S_anti = np.zeros((n_base_sims, n_steps + 1))
            V_anti = np.zeros((n_base_sims, n_steps + 1))
            S_anti[:, 0] = S0
            V_anti[:, 0] = params.v0
            
            for t in range(n_steps):
                V_anti[:, t+1] = np.maximum(
                    V_anti[:, t] + params.kappa * (params.theta - V_anti[:, t]) * dt +
                    params.sigma * np.sqrt(V_anti[:, t] * dt) * Z2_corr_anti[:, t],
                    0.001
                )
                
                S_anti[:, t+1] = S_anti[:, t] * np.exp(
                    (r - 0.5 * V_anti[:, t]) * dt + np.sqrt(V_anti[:, t] * dt) * Z1_anti[:, t]
                )
            
            # Combine results
            final_prices = np.concatenate([final_prices, S_anti[:, -1]])
        
        return final_prices
    
    def _run_heston_numba(self, S0: float, T: float,
                         params: HestonParameters,
                         config: SimulationConfig) -> np.ndarray:
        """Numba-optimized Heston simulation"""
        if not NUMBA_AVAILABLE:
            return self._run_heston_numpy(S0, T, params, config)
        
        return _heston_simulation_numba(
            S0, T, params.kappa, params.theta, params.sigma,
            params.rho, params.v0, config.num_simulations, config.time_steps
        )
    
    def _run_gbm_fallback(self, S0: float, T: float, initial_var: float,
                         config: SimulationConfig) -> np.ndarray:
        """Fallback geometric Brownian motion simulation"""
        self.logger.warning("Using GBM fallback simulation")
        
        sigma = np.sqrt(initial_var)
        r = 0.03
        dt = T / config.time_steps
        
        # Generate random numbers
        if config.random_seed:
            np.random.seed(config.random_seed)
        
        Z = np.random.standard_normal((config.num_simulations, config.time_steps))
        
        # Simulate paths
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(log_prices[:, -1])
    
    def _estimate_expected_move(self, current_price: float, volatility_metrics: dict,
                              days_to_expiration: int) -> float:
        """Estimate expected move (approximates straddle price)"""
        # Use implied volatility if available, otherwise realized volatility
        iv = volatility_metrics.get('iv30', volatility_metrics.get('rv30', 0.25))
        
        # Convert days to years
        T = days_to_expiration / 365.0
        
        # Expected move formula (approximation)
        expected_move = current_price * iv * np.sqrt(T)
        
        return expected_move
    
    def _analyze_price_distribution(self, final_prices: np.ndarray, current_price: float,
                                  expected_move: float) -> Dict[str, Any]:
        """Analyze the simulated price distribution"""
        try:
            # Basic statistics
            price_changes = np.abs(final_prices - current_price)
            
            # Movement probabilities
            prob_exceed_1x = np.mean(price_changes >= expected_move) * 100
            prob_exceed_1_5x = np.mean(price_changes >= 1.5 * expected_move) * 100
            prob_exceed_2x = np.mean(price_changes >= 2.0 * expected_move) * 100
            
            # Directional probabilities
            upside_prob = np.mean(final_prices > current_price) * 100
            downside_prob = np.mean(final_prices < current_price) * 100
            
            # Expected return
            returns = (final_prices - current_price) / current_price
            expected_return = np.mean(returns) * 100
            
            # Value at Risk (5th percentile)
            var_95 = np.percentile(final_prices, 5)
            
            # Distribution characteristics
            price_mean = np.mean(final_prices)
            price_std = np.std(final_prices)
            skewness = self._calculate_skewness(final_prices)
            kurtosis = self._calculate_kurtosis(final_prices)
            
            return {
                'prob_exceed_1x': prob_exceed_1x,
                'prob_exceed_1_5x': prob_exceed_1_5x,
                'prob_exceed_2x': prob_exceed_2x,
                'upside_probability': upside_prob,
                'downside_probability': downside_prob,
                'expected_return': expected_return,
                'value_at_risk_95': var_95,
                'mean_final_price': price_mean,
                'price_volatility': price_std,
                'distribution_skewness': skewness,
                'distribution_kurtosis': kurtosis,
                'median_final_price': np.median(final_prices),
                'price_range_95': {
                    'lower': np.percentile(final_prices, 2.5),
                    'upper': np.percentile(final_prices, 97.5)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing price distribution: {e}")
            return self._create_fallback_distribution_result(current_price)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of distribution"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        except:
            return 0.0
    
    def _create_fallback_result(self, current_price: float, simulations: int) -> Dict[str, Any]:
        """Create fallback result when simulation fails"""
        return {
            'prob_exceed_1x': 45.0,
            'prob_exceed_1_5x': 25.0,
            'prob_exceed_2x': 12.0,
            'upside_probability': 50.0,
            'downside_probability': 50.0,
            'expected_return': 0.0,
            'value_at_risk_95': current_price * 0.9,
            'simulations_run': simulations,
            'computation_time': 0.0,
            'fallback_mode': True,
            'mean_final_price': current_price,
            'price_volatility': current_price * 0.2,
            'distribution_skewness': 0.0,
            'distribution_kurtosis': 0.0
        }
    
    def _create_fallback_distribution_result(self, current_price: float) -> Dict[str, Any]:
        """Create fallback distribution analysis"""
        return {
            'prob_exceed_1x': 45.0,
            'prob_exceed_1_5x': 25.0,
            'prob_exceed_2x': 12.0,
            'upside_probability': 50.0,
            'downside_probability': 50.0,
            'expected_return': 0.0,
            'value_at_risk_95': current_price * 0.9,
            'mean_final_price': current_price,
            'price_volatility': current_price * 0.2,
            'distribution_skewness': 0.0,
            'distribution_kurtosis': 0.0,
            'median_final_price': current_price,
            'price_range_95': {
                'lower': current_price * 0.85,
                'upper': current_price * 1.15
            }
        }
    
    def clear_cache(self):
        """Clear parameter and simulation caches"""
        self._parameter_cache.clear()
        self._simulation_cache.clear()
        self.logger.info("Monte Carlo cache cleared")


# Numba-optimized functions (if available)
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _heston_simulation_numba(S0, T, kappa, theta, sigma, rho, v0, n_sims, n_steps):
        """Numba-optimized Heston simulation"""
        dt = T / n_steps
        r = 0.03
        
        final_prices = np.zeros(n_sims)
        
        for i in prange(n_sims):
            S = S0
            V = v0
            
            for j in range(n_steps):
                # Generate correlated random numbers
                Z1 = np.random.standard_normal()
                Z2 = np.random.standard_normal()
                Z2_corr = rho * Z1 + np.sqrt(1 - rho*rho) * Z2
                
                # Update variance (with floor)
                V_new = V + kappa * (theta - V) * dt + sigma * np.sqrt(max(V * dt, 0)) * Z2_corr
                V = max(V_new, 0.001)
                
                # Update price
                S = S * np.exp((r - 0.5 * V) * dt + np.sqrt(max(V * dt, 0)) * Z1)
            
            final_prices[i] = S
        
        return final_prices
else:
    def _heston_simulation_numba(*args, **kwargs):
        """Placeholder when Numba is not available"""
        raise NotImplementedError("Numba not available")


# Performance testing utilities
class MonteCarloProfiler:
    """Profiling utilities for Monte Carlo performance"""
    
    @staticmethod
    def benchmark_simulation_speeds(engine: MonteCarloEngine, test_price: float = 100.0):
        """Benchmark different simulation configurations"""
        import time
        
        configurations = [
            (1000, "Small"),
            (5000, "Medium"), 
            (10000, "Large"),
            (25000, "Extra Large")
        ]
        
        # Create dummy data
        dummy_data = pd.DataFrame({
            'Close': [test_price * (1 + np.random.normal(0, 0.02, 100))[i] for i in range(100)]
        })
        
        dummy_vol_metrics = {
            'iv30': 0.25,
            'rv30': 0.20,
            'vix_level': 20.0
        }
        
        results = {}
        
        for num_sims, size_label in configurations:
            start_time = time.time()
            
            result = engine.run_simulation(
                symbol="TEST",
                current_price=test_price,
                historical_data=dummy_data,
                volatility_metrics=dummy_vol_metrics,
                simulations=num_sims,
                days_to_expiration=30
            )
            
            duration = time.time() - start_time
            sims_per_second = num_sims / duration
            
            results[size_label] = {
                'simulations': num_sims,
                'duration': duration,
                'sims_per_second': sims_per_second,
                'prob_exceed_1x': result['prob_exceed_1x']
            }
        
        return results


# Export main class
__all__ = ['MonteCarloEngine', 'HestonParameters', 'SimulationConfig', 'MonteCarloProfiler']