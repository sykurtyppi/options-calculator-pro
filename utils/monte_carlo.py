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
class MertonJumpParameters:
    """Merton jump-diffusion parameters for log-price jumps."""
    lambda_: float = 0.10   # Jump intensity (annualized)
    mu_j: float = 0.0       # Mean jump size in log space
    sigma_j: float = 0.08   # Std-dev of jump size in log space


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation"""
    num_simulations: int = 10000
    time_steps: int = 50
    use_antithetic: bool = True
    use_control_variate: bool = True
    random_seed: Optional[int] = None
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    use_jump_diffusion: bool = False


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
            risk_free_rate = float(kwargs.get('risk_free_rate', volatility_metrics.get('risk_free_rate', 0.05)))
            dividend_yield = float(kwargs.get('dividend_yield', volatility_metrics.get('dividend_yield', 0.0)))
            days_to_earnings = kwargs.get('days_to_earnings', None)
            use_jump_diffusion = bool(kwargs.get('use_jump_diffusion', True))
            
            # Calibrate Heston parameters
            heston_params = self._calibrate_heston_parameters(
                symbol, historical_data, volatility_metrics
            )
            jump_params = self._calibrate_jump_parameters(
                volatility_metrics=volatility_metrics,
                days_to_earnings=days_to_earnings,
            )
            
            # Setup simulation configuration
            config = SimulationConfig(
                num_simulations=simulations,
                time_steps=min(days_to_expiration, 50),
                use_antithetic=simulations >= 5000,
                use_control_variate=True,
                random_seed=42 if simulations <= 1000 else None,  # Reproducible for small tests
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                use_jump_diffusion=use_jump_diffusion,
            )
            
            # Run Heston simulation
            final_prices = self._run_heston_simulation(
                S0=current_price,
                T=days_to_expiration / 365.0,
                heston_params=heston_params,
                jump_params=jump_params,
                config=config
            )
            
            # Calculate expected move (straddle approximation)
            expected_move = self._estimate_expected_move(
                current_price, volatility_metrics, days_to_expiration, jump_params=jump_params
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
                'jump_parameters': {
                    'lambda': jump_params.lambda_,
                    'mu_j': jump_params.mu_j,
                    'sigma_j': jump_params.sigma_j,
                    'enabled': bool(config.use_jump_diffusion),
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

            # Refine Heston params using option term-structure points when available.
            params = self._fit_heston_to_term_structure_proxy(params, volatility_metrics)
            
            # Cache the parameters
            self._parameter_cache[cache_key] = params
            
            return params
            
        except Exception as e:
            self.logger.warning(f"Parameter calibration failed for {symbol}: {e}")
            return self._get_default_heston_parameters(volatility_metrics)

    def _extract_term_structure_points(self, volatility_metrics: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract (tenor_years, atm_iv) arrays from volatility metric payload."""
        days = volatility_metrics.get('term_structure_days')
        ivs = volatility_metrics.get('term_structure_atm_ivs')

        if isinstance(days, (list, tuple)) and isinstance(ivs, (list, tuple)):
            days_arr = np.asarray(days, dtype=float)
            ivs_arr = np.asarray(ivs, dtype=float)
        else:
            points = []
            if isinstance(volatility_metrics.get('iv_30d'), (int, float)):
                points.append((30.0, float(volatility_metrics.get('iv_30d'))))
            if isinstance(volatility_metrics.get('iv_45d'), (int, float)):
                points.append((45.0, float(volatility_metrics.get('iv_45d'))))
            if isinstance(volatility_metrics.get('iv30'), (int, float)):
                points.append((30.0, float(volatility_metrics.get('iv30'))))
            if isinstance(volatility_metrics.get('short_iv'), (int, float)) and isinstance(volatility_metrics.get('short_dte'), (int, float)):
                points.append((float(volatility_metrics.get('short_dte')), float(volatility_metrics.get('short_iv'))))
            if isinstance(volatility_metrics.get('long_iv'), (int, float)) and isinstance(volatility_metrics.get('long_dte'), (int, float)):
                points.append((float(volatility_metrics.get('long_dte')), float(volatility_metrics.get('long_iv'))))
            if not points:
                return np.array([], dtype=float), np.array([], dtype=float)
            points = [(d, v) for d, v in points if np.isfinite(d) and np.isfinite(v) and d > 0 and v > 0]
            if not points:
                return np.array([], dtype=float), np.array([], dtype=float)
            unique = {}
            for d, v in points:
                unique.setdefault(round(float(d), 6), []).append(float(v))
            days_arr = np.array(sorted(unique.keys()), dtype=float)
            ivs_arr = np.array([np.mean(unique[d]) for d in days_arr], dtype=float)

        mask = (
            np.isfinite(days_arr)
            & np.isfinite(ivs_arr)
            & (days_arr > 0.0)
            & (ivs_arr > 0.0)
            & (ivs_arr < 5.0)
        )
        if mask.sum() < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        days_arr = days_arr[mask]
        ivs_arr = ivs_arr[mask]
        sort_idx = np.argsort(days_arr)
        days_arr = days_arr[sort_idx]
        ivs_arr = ivs_arr[sort_idx]

        # Average duplicate days.
        unique_days, inverse = np.unique(days_arr, return_inverse=True)
        unique_ivs = np.array([float(np.mean(ivs_arr[inverse == i])) for i in range(len(unique_days))], dtype=float)
        tenor_years = unique_days / 365.25
        return tenor_years, unique_ivs

    def _fit_heston_to_term_structure_proxy(self, base_params: HestonParameters,
                                          volatility_metrics: dict) -> HestonParameters:
        """
        Lightweight Heston calibration to ATM term structure:
          sqrt(E[v_t]) ~= IV(t), where E[v_t] = theta + (v0-theta)exp(-kappa t)
        """
        tenors, iv_targets = self._extract_term_structure_points(volatility_metrics)
        if tenors.size < 2:
            return base_params

        v0 = max(1e-6, float(base_params.v0))
        iv_var = np.square(iv_targets)
        theta_min = float(max(1e-4, np.percentile(iv_var, 10) * 0.6))
        theta_max = float(max(theta_min + 1e-4, np.percentile(iv_var, 90) * 1.4))
        theta_grid = np.linspace(theta_min, theta_max, 28)
        kappa_grid = np.linspace(0.20, 7.50, 40)

        weights = 1.0 / np.sqrt(np.maximum(tenors, 1e-4))
        weights = weights / np.sum(weights)

        best_kappa = float(base_params.kappa)
        best_theta = float(base_params.theta)
        best_loss = float('inf')

        for kappa in kappa_grid:
            exp_term = np.exp(-kappa * tenors)
            for theta in theta_grid:
                expected_var = theta + (v0 - theta) * exp_term
                expected_iv = np.sqrt(np.clip(expected_var, 1e-8, 10.0))
                residual = expected_iv - iv_targets
                loss = float(np.sum(weights * residual * residual))
                if loss < best_loss:
                    best_loss = loss
                    best_kappa = float(kappa)
                    best_theta = float(theta)

        fitted = HestonParameters(
            kappa=best_kappa,
            theta=float(np.clip(best_theta, 1e-4, 4.0)),
            sigma=float(base_params.sigma),
            rho=float(base_params.rho),
            v0=v0,
        )

        # Adjust vol-of-vol by observed term-structure curvature signal.
        if tenors.size >= 3:
            coeffs = np.polyfit(tenors, iv_targets, 2)
            curvature = float(coeffs[0])
            fitted.sigma = float(np.clip(fitted.sigma * (1.0 + min(abs(curvature) * 2.0, 0.7)), 0.1, 1.2))

        # Preserve stability.
        feller_lhs = 2.0 * fitted.kappa * fitted.theta
        if feller_lhs > 0 and fitted.sigma ** 2 >= feller_lhs:
            fitted.sigma = float(np.sqrt(max(feller_lhs * 0.95, 0.01)))

        return fitted

    def _calibrate_jump_parameters(self, volatility_metrics: dict,
                                 days_to_earnings: Optional[int] = None) -> MertonJumpParameters:
        """
        Calibrate Merton jump parameters from volatility state.
        Uses IV-RV dislocation and earnings proximity to set jump intensity.
        """
        try:
            iv30 = float(volatility_metrics.get('iv30', volatility_metrics.get('implied_vol_30d', 0.25)))
            rv30 = float(volatility_metrics.get('rv30', volatility_metrics.get('realized_vol_30d', iv30)))
            yz_vol = float(volatility_metrics.get('yang_zhang_volatility', rv30))

            iv30 = max(0.01, min(3.0, iv30))
            rv30 = max(0.01, min(3.0, rv30))
            yz_vol = max(0.01, min(3.0, yz_vol))

            iv_rv_gap = max(0.0, iv30 - rv30)
            yz_excess = max(0.0, yz_vol - rv30)
            event_iv_premium = float(volatility_metrics.get('event_iv_premium', 0.0))
            iv_term_ratio = float(volatility_metrics.get('iv_term_ratio', 1.0))
            ts_slope_0_45 = float(volatility_metrics.get('term_structure_slope_0_45', volatility_metrics.get('term_structure_slope', 0.0)))
            term_structure_stress = max(
                0.0,
                event_iv_premium * 1.8
                + max(0.0, iv_term_ratio - 1.0) * 0.8
                + max(0.0, -ts_slope_0_45) * 35.0
            )

            # Base annualized jump intensity before earnings adjustment.
            base_lambda = 0.06 + 3.2 * iv_rv_gap + 1.8 * yz_excess + 1.6 * term_structure_stress

            if days_to_earnings is None:
                earnings_multiplier = 1.0
            else:
                dte = max(0, int(days_to_earnings))
                if dte <= 10:
                    earnings_multiplier = 1.0 + 4.5 * np.exp(-dte / 3.0)
                else:
                    earnings_multiplier = 1.0

            lambda_eff = float(np.clip(base_lambda * earnings_multiplier, 0.01, 3.0))

            # Jump size volatility linked to dislocation magnitude.
            sigma_j = float(np.clip(0.05 + 0.55 * (iv_rv_gap + yz_excess + 0.7 * term_structure_stress), 0.04, 0.45))

            # Keep jump mean centered; risk-neutral drift compensation handles expectation.
            mu_j = float(np.clip(volatility_metrics.get('jump_mean', 0.0), -0.20, 0.20))

            return MertonJumpParameters(lambda_=lambda_eff, mu_j=mu_j, sigma_j=sigma_j)
        except Exception:
            return MertonJumpParameters()
    
    def _estimate_heston_parameters(self, returns: pd.Series,
                                  volatility_metrics: dict) -> HestonParameters:
        """Estimate Heston parameters from return series"""
        # Calculate basic statistics
        return_vol = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Initial variance (current IV squared)
        iv30 = float(volatility_metrics.get('iv30', return_vol))
        iv30 = max(0.05, min(2.5, iv30))
        v0 = iv30 ** 2
        
        # Long-term variance (use realized volatility as proxy)
        rv30 = float(volatility_metrics.get('rv30', return_vol))
        rv30 = max(0.05, min(2.5, rv30))
        theta = rv30 ** 2
        
        # Mean reversion speed (calibrated to typical values)
        # Higher for more mean-reverting volatility
        rolling_vol = returns.rolling(window=20).std()
        vol_of_vol = float(rolling_vol.std() * np.sqrt(252))
        if not np.isfinite(vol_of_vol):
            vol_of_vol = 0.25
        kappa = max(0.5, min(5.0, 2.0 / (1 + vol_of_vol)))
        
        # Volatility of volatility
        # Estimated from volatility clustering
        sigma = max(0.1, min(1.2, vol_of_vol * 2))
        
        # Correlation between returns and volatility
        # Calculate empirical correlation if possible
        if len(returns) >= 60:
            # Rolling volatility
            vol_changes = rolling_vol.pct_change().dropna()
            return_aligned = returns.loc[vol_changes.index]
            
            if len(vol_changes) > 10:
                correlation = return_aligned.corr(vol_changes)
                rho = max(-0.95, min(-0.1, correlation if not np.isnan(correlation) else -0.7))
            else:
                rho = -0.7  # Default negative correlation
        else:
            rho = -0.7  # Default

        # Enforce near-Feller consistency for numerical stability.
        # If violated, reduce sigma to sit slightly below theoretical boundary.
        feller_lhs = 2.0 * kappa * theta
        if feller_lhs <= 0:
            theta = max(theta, 0.01)
            feller_lhs = 2.0 * kappa * theta
        if sigma ** 2 >= feller_lhs:
            sigma = max(0.1, np.sqrt(max(feller_lhs * 0.95, 0.01)))
        
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
        # Use actual historical volatility instead of VIX-based calculation
        historical_vol = volatility_metrics.get('rv30', volatility_metrics.get('theta', 0.25))

        # CRITICAL FIX: Use historical volatility squared for theta, not VIX squared
        theta_variance = historical_vol ** 2  # Convert volatility to variance

        self.logger.debug(
            "Default Heston parameters: historical_vol=%.4f, theta_variance=%.6f, iv30=%.4f",
            historical_vol,
            theta_variance,
            iv30,
        )

        result = HestonParameters(
            kappa=2.0,  # Moderate mean reversion
            theta=max(0.01, min(0.20, theta_variance)),  # Use REAL historical volatility
            sigma=0.3,  # Moderate vol of vol
            rho=-0.7,   # Typical negative correlation
            v0=max(0.01, iv30 ** 2)  # Current variance
        )
        if result.sigma ** 2 >= 2.0 * result.kappa * result.theta:
            result.sigma = float(np.sqrt(max(2.0 * result.kappa * result.theta * 0.95, 0.01)))
        return result
    
    def _run_heston_simulation(self, S0: float, T: float,
                             heston_params: HestonParameters,
                             jump_params: MertonJumpParameters,
                             config: SimulationConfig) -> np.ndarray:
        """Run Heston stochastic volatility simulation"""
        try:
            if NUMBA_AVAILABLE and config.num_simulations >= 1000 and not config.use_jump_diffusion:
                # Use optimized Numba version for large simulations
                return self._run_heston_numba(S0, T, heston_params, config)
            else:
                # Use standard numpy version
                return self._run_heston_numpy(S0, T, heston_params, jump_params, config)
                
        except Exception as e:
            self.logger.error(f"Heston simulation failed: {e}")
            # Fallback to simple geometric Brownian motion
            return self._run_gbm_fallback(
                S0, T, heston_params.v0, config, config.risk_free_rate, config.dividend_yield
            )

    def _run_heston_numpy(self, S0: float, T: float,
                         params: HestonParameters,
                         jump_params: MertonJumpParameters,
                         config: SimulationConfig) -> np.ndarray:
        """Standard numpy implementation of Heston model"""
        # Simulation parameters
        n_sims = config.num_simulations
        n_steps = config.time_steps
        dt = T / n_steps
        r = config.risk_free_rate
        q = config.dividend_yield
        use_jumps = bool(config.use_jump_diffusion)

        jump_lambda = max(0.0, float(jump_params.lambda_))
        jump_mu = float(jump_params.mu_j)
        jump_sigma = max(0.0, float(jump_params.sigma_j))
        jump_compensation = jump_lambda * (np.exp(jump_mu + 0.5 * jump_sigma**2) - 1.0)
        lambda_dt = jump_lambda * dt
        
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
            v_prev = np.maximum(V[:, t], 1e-8)
            V[:, t+1] = np.maximum(
                v_prev + params.kappa * (params.theta - v_prev) * dt +
                params.sigma * np.sqrt(v_prev * dt) * Z2_corr[:, t],
                0.001  # Minimum variance
            )

            if use_jumps and lambda_dt > 0.0:
                jump_counts = np.random.poisson(lambda_dt, n_base_sims)
                jump_noise = np.random.standard_normal(n_base_sims)
                jump_term = jump_counts * jump_mu + np.sqrt(jump_counts) * jump_sigma * jump_noise
            else:
                jump_term = 0.0
            
            # Price process
            S[:, t+1] = S[:, t] * np.exp(
                (r - q - jump_compensation - 0.5 * v_prev) * dt +
                np.sqrt(v_prev * dt) * Z1[:, t] +
                jump_term
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
                v_prev_anti = np.maximum(V_anti[:, t], 1e-8)
                V_anti[:, t+1] = np.maximum(
                    v_prev_anti + params.kappa * (params.theta - v_prev_anti) * dt +
                    params.sigma * np.sqrt(v_prev_anti * dt) * Z2_corr_anti[:, t],
                    0.001
                )

                if use_jumps and lambda_dt > 0.0:
                    jump_counts_anti = np.random.poisson(lambda_dt, n_base_sims)
                    jump_noise_anti = -np.random.standard_normal(n_base_sims)
                    jump_term_anti = (
                        jump_counts_anti * jump_mu +
                        np.sqrt(jump_counts_anti) * jump_sigma * jump_noise_anti
                    )
                else:
                    jump_term_anti = 0.0
                
                S_anti[:, t+1] = S_anti[:, t] * np.exp(
                    (r - q - jump_compensation - 0.5 * v_prev_anti) * dt +
                    np.sqrt(v_prev_anti * dt) * Z1_anti[:, t] +
                    jump_term_anti
                )
            
            # Combine results
            final_prices = np.concatenate([final_prices, S_anti[:, -1]])
        
        return final_prices
    
    def _run_heston_numba(self, S0: float, T: float,
                         params: HestonParameters,
                         config: SimulationConfig) -> np.ndarray:
        """Numba-optimized Heston simulation"""
        if not NUMBA_AVAILABLE:
            return self._run_heston_numpy(S0, T, params, MertonJumpParameters(), config)
        
        return _heston_simulation_numba(
            S0, T, params.kappa, params.theta, params.sigma,
            params.rho, params.v0, config.num_simulations, config.time_steps, config.risk_free_rate
        )
    
    def _run_gbm_fallback(self, S0: float, T: float, initial_var: float,
                         config: SimulationConfig, risk_free_rate: float,
                         dividend_yield: float = 0.0) -> np.ndarray:
        """Fallback geometric Brownian motion simulation"""
        self.logger.warning("Using GBM fallback simulation")
        
        sigma = np.sqrt(initial_var)
        r = risk_free_rate
        q = float(dividend_yield)
        dt = T / config.time_steps
        
        # Generate random numbers
        if config.random_seed:
            np.random.seed(config.random_seed)
        
        Z = np.random.standard_normal((config.num_simulations, config.time_steps))
        
        # Simulate paths
        log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(log_prices[:, -1])
    
    def _estimate_expected_move(self, current_price: float, volatility_metrics: dict,
                              days_to_expiration: int,
                              jump_params: Optional[MertonJumpParameters] = None) -> float:
        """Estimate expected move (approximates straddle price)"""
        # Use implied volatility if available, otherwise realized volatility
        iv = float(volatility_metrics.get('iv30', volatility_metrics.get('rv30', 0.25)))
        iv = max(0.01, min(3.0, iv))
        
        # Convert days to years
        T = max(float(days_to_expiration) / 365.0, 1.0 / 365.0)

        diffusion_var = (iv ** 2) * T
        jump_var = 0.0
        if jump_params is not None:
            jump_var = max(0.0, float(jump_params.lambda_)) * (
                float(jump_params.sigma_j) ** 2 + float(jump_params.mu_j) ** 2
            ) * T
        
        # Expected absolute move proxy from total variance.
        expected_move = current_price * np.sqrt(max(diffusion_var + jump_var, 0.0))
        
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
    def _heston_simulation_numba(S0, T, kappa, theta, sigma, rho, v0, n_sims, n_steps, r):
        """Numba-optimized Heston simulation"""
        dt = T / n_steps
        
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
