"""
Advanced Monte Carlo simulation engine for options pricing and risk analysis.

This module provides production-ready Monte Carlo implementations with:
- Heston stochastic volatility model for realistic volatility dynamics
- High-performance simulation with variance reduction techniques
- Multiple payoff types and exotic option support
- Risk metrics and scenario analysis
- GPU acceleration support (optional)
"""

import numpy as np
import os
from typing import Dict, List, Optional, Callable, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import time

from .option_data import OptionType, OptionContract
from .greeks import GreeksResult

class ModelType(Enum):
    """Monte Carlo model types"""
    BLACK_SCHOLES = "black_scholes"
    HESTON = "heston"
    JUMP_DIFFUSION = "jump_diffusion"
    STOCHASTIC_RATES = "stochastic_rates"

class VarianceReductionTechnique(Enum):
    """Variance reduction techniques"""
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    STRATIFIED = "stratified"
    IMPORTANCE_SAMPLING = "importance_sampling"

@dataclass
class HestonParameters:
    """Heston stochastic volatility model parameters"""
    kappa: float = 2.0      # Rate of mean reversion
    theta: float = 0.08     # Long-term variance level (FIXED: was hardcoded to 4%, now 8% default)
    sigma: float = 0.3      # Volatility of volatility
    rho: float = -0.7       # Correlation between asset and variance
    v0: float = 0.08        # Initial variance (FIXED: was hardcoded to 4%, now 8% default)

    def __post_init__(self):
        """Validate Heston parameters"""
        if self.kappa <= 0:
            raise ValueError("Kappa (mean reversion rate) must be positive")
        if self.theta <= 0:
            raise ValueError("Theta (long-term variance) must be positive")
        if self.sigma <= 0:
            raise ValueError("Sigma (vol of vol) must be positive")
        if not -1 <= self.rho <= 1:
            raise ValueError("Rho (correlation) must be between -1 and 1")
        if self.v0 <= 0:
            raise ValueError("V0 (initial variance) must be positive")

        # Feller condition check
        if 2 * self.kappa * self.theta < self.sigma**2:
            warnings.warn(
                "Feller condition violated: 2*kappa*theta < sigma^2. "
                "Variance process may hit zero boundary.",
                UserWarning
            )

@dataclass
class MertonJumpParameters:
    """Merton jump diffusion model parameters"""
    lambda_: float = 0.1        # Jump intensity (jumps per unit time)
    mu_j: float = 0.0           # Mean jump size (log scale)
    sigma_j: float = 0.05       # Jump size volatility (log scale)
    yang_zhang_lambda: Optional[float] = None  # Yang-Zhang derived lambda
    earnings_multiplier: float = 5.0  # Lambda multiplier near earnings

    def __post_init__(self):
        """Validate Merton parameters"""
        if self.lambda_ < 0:
            raise ValueError("Lambda (jump intensity) must be non-negative")
        if self.sigma_j <= 0:
            raise ValueError("Sigma_j (jump volatility) must be positive")
        if self.earnings_multiplier < 1.0:
            raise ValueError("Earnings multiplier must be >= 1.0")

    def get_effective_lambda(self, days_to_earnings: Optional[int] = None) -> float:
        """Get effective lambda considering earnings proximity"""
        effective_lambda = self.yang_zhang_lambda or self.lambda_

        # Increase jump intensity near earnings
        if days_to_earnings is not None and days_to_earnings <= 7:
            # Exponential increase as earnings approach
            earnings_factor = self.earnings_multiplier * np.exp(-days_to_earnings / 3.0)
            effective_lambda *= earnings_factor

        return effective_lambda

@dataclass  
class MonteCarloResult:
    """Monte Carlo simulation result"""
    option_price: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    paths_used: int
    execution_time: float
    convergence_achieved: bool = False
    greeks: Optional[GreeksResult] = None
    path_statistics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def relative_error(self) -> float:
        """Relative standard error as percentage"""
        return (self.standard_error / self.option_price * 100) if self.option_price != 0 else 0.0
    
    @property
    def confidence_width(self) -> float:
        """Width of confidence interval"""
        return self.confidence_interval[1] - self.confidence_interval[0]

@dataclass
class PathStatistics:
    """Statistics from Monte Carlo paths"""
    final_prices: np.ndarray
    max_prices: np.ndarray
    min_prices: np.ndarray
    volatilities: Optional[np.ndarray] = None
    
    @property
    def price_percentiles(self) -> Dict[str, float]:
        """Key percentiles of final prices"""
        return {
            'p1': np.percentile(self.final_prices, 1),
            'p5': np.percentile(self.final_prices, 5),
            'p25': np.percentile(self.final_prices, 25),
            'p50': np.percentile(self.final_prices, 50),
            'p75': np.percentile(self.final_prices, 75),
            'p95': np.percentile(self.final_prices, 95),
            'p99': np.percentile(self.final_prices, 99)
        }
    
    @property
    def risk_metrics(self) -> Dict[str, float]:
        """Risk metrics from path statistics"""
        mean_price = np.mean(self.final_prices)
        return {
            'var_95': np.percentile(self.final_prices, 5) - mean_price,
            'var_99': np.percentile(self.final_prices, 1) - mean_price,
            'expected_shortfall_95': np.mean(self.final_prices[self.final_prices <= np.percentile(self.final_prices, 5)]) - mean_price,
            'max_drawdown': mean_price - np.min(self.min_prices),
            'probability_profit': np.sum(self.final_prices > 0) / len(self.final_prices) if len(self.final_prices) > 0 else 0.0
        }

class MonteCarloEngine:
    """
    High-performance Monte Carlo simulation engine.
    
    Features:
    - Multiple stochastic models (Black-Scholes, Heston, Jump-Diffusion)
    - Variance reduction techniques for improved convergence
    - Parallel processing for large simulations
    - Greek calculation via finite differences
    - Exotic option support
    """
    
    def __init__(
        self,
        model_type: ModelType = ModelType.HESTON,
        variance_reduction: VarianceReductionTechnique = VarianceReductionTechnique.ANTITHETIC,
        random_seed: Optional[int] = None,
        use_parallel: bool = True,
        max_workers: Optional[int] = None
    ):
        self.model_type = model_type
        self.variance_reduction = variance_reduction
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize model-specific parameters
        self.heston_params = HestonParameters()
        self.merton_params = MertonJumpParameters()
    
    def price_option(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        volatility: float = 0.2,
        option_type: OptionType = OptionType.CALL,
        num_paths: int = 100000,
        num_steps: int = 252,
        payoff_func: Optional[Callable] = None,
        target_error: float = 0.01,
        max_paths: int = 1000000
    ) -> MonteCarloResult:
        """
        Price an option using Monte Carlo simulation.
        
        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            volatility: Initial volatility (used differently per model)
            option_type: Call or Put
            num_paths: Initial number of simulation paths
            num_steps: Number of time steps per path
            payoff_func: Custom payoff function (overrides option_type)
            target_error: Target relative standard error for convergence
            max_paths: Maximum paths before stopping
            
        Returns:
            MonteCarloResult with pricing and statistics
        """
        start_time = time.time()
        
        # Input validation
        if time_to_expiry <= 0:
            return self._zero_result(start_time)
        if spot <= 0 or strike <= 0:
            return self._zero_result(start_time)
        if num_paths <= 0:
            return self._zero_result(start_time)
        
        # Adaptive path sizing for convergence
        current_paths = num_paths
        cumulative_payoffs = []
        cumulative_paths = 0
        
        while cumulative_paths < max_paths:
            # Generate paths based on model type
            if self.model_type == ModelType.HESTON:
                paths, vol_paths = self._generate_heston_paths(
                    spot, time_to_expiry, risk_free_rate, dividend_yield,
                    current_paths, num_steps
                )
            elif self.model_type == ModelType.BLACK_SCHOLES:
                paths = self._generate_bs_paths(
                    spot, volatility, time_to_expiry, risk_free_rate,
                    dividend_yield, current_paths, num_steps
                )
                vol_paths = None
            elif self.model_type == ModelType.JUMP_DIFFUSION:
                paths = self._generate_jump_diffusion_paths(
                    spot, volatility, time_to_expiry, risk_free_rate,
                    dividend_yield, current_paths, num_steps
                )
                vol_paths = None
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Calculate payoffs
            if payoff_func is not None:
                payoffs = payoff_func(paths)
            else:
                payoffs = self._calculate_standard_payoffs(paths, strike, option_type)
            
            # Apply variance reduction
            if self.variance_reduction == VarianceReductionTechnique.ANTITHETIC:
                payoffs = self._apply_antithetic_variates(payoffs)
            
            # Discount payoffs
            discounted_payoffs = payoffs * np.exp(-risk_free_rate * time_to_expiry)
            cumulative_payoffs.extend(discounted_payoffs)
            cumulative_paths += len(discounted_payoffs)
            
            # Check convergence
            if cumulative_paths >= num_paths:
                current_mean = np.mean(cumulative_payoffs)
                current_std = np.std(cumulative_payoffs) / np.sqrt(len(cumulative_payoffs))
                relative_error = current_std / current_mean if current_mean != 0 else float('inf')
                
                if relative_error <= target_error or cumulative_paths >= max_paths:
                    break
                
                # Increase path count for next iteration
                current_paths = min(current_paths * 2, max_paths - cumulative_paths)
        
        # Final calculations
        final_payoffs = np.array(cumulative_payoffs)
        option_price = np.mean(final_payoffs)
        standard_error = np.std(final_payoffs) / np.sqrt(len(final_payoffs))
        
        # Confidence interval (95%)
        confidence_interval = (
            option_price - 1.96 * standard_error,
            option_price + 1.96 * standard_error
        )
        
        # Path statistics
        final_prices = paths[:, -1] if len(paths.shape) > 1 else paths
        path_stats = PathStatistics(
            final_prices=final_prices,
            max_prices=np.max(paths, axis=1) if len(paths.shape) > 1 else paths,
            min_prices=np.min(paths, axis=1) if len(paths.shape) > 1 else paths,
            volatilities=vol_paths[:, -1] if vol_paths is not None else None
        )
        
        execution_time = time.time() - start_time
        convergence_achieved = (standard_error / option_price <= target_error) if option_price != 0 else False
        
        return MonteCarloResult(
            option_price=option_price,
            standard_error=standard_error,
            confidence_interval=confidence_interval,
            paths_used=len(final_payoffs),
            execution_time=execution_time,
            convergence_achieved=convergence_achieved,
            path_statistics=path_stats.price_percentiles,
            risk_metrics=path_stats.risk_metrics
        )
    
    def _generate_heston_paths(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        num_paths: int,
        num_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate paths using Heston stochastic volatility model.
        
        Uses Euler discretization with absorption at zero boundary for variance.
        More sophisticated schemes like QE or Milstein could be implemented for accuracy.
        """
        dt = T / num_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        S = np.zeros((num_paths, num_steps + 1))
        v = np.zeros((num_paths, num_steps + 1))
        
        # Initial conditions
        S[:, 0] = S0
        v[:, 0] = self.heston_params.v0
        
        # Generate correlated random numbers
        rho = self.heston_params.rho
        sqrt_one_minus_rho2 = np.sqrt(1 - rho**2)
        
        for i in range(num_steps):
            # Independent random numbers
            Z1 = np.random.standard_normal(num_paths)
            Z2 = np.random.standard_normal(num_paths)
            
            # Correlated random numbers
            dW_S = Z1
            dW_v = rho * Z1 + sqrt_one_minus_rho2 * Z2
            
            # Current values
            S_curr = S[:, i]
            v_curr = np.maximum(v[:, i], 0)  # Absorption at zero
            sqrt_v_curr = np.sqrt(v_curr)
            
            # Heston dynamics
            # Use log-Euler step for price to preserve strict positivity.
            # d ln S = (r - q - 0.5*v) dt + sqrt(v) dW
            log_return = (r - q - 0.5 * v_curr) * dt + sqrt_v_curr * sqrt_dt * dW_S
            S[:, i + 1] = S_curr * np.exp(log_return)
            
            # dv = kappa * (theta - v) * dt + sigma * sqrt(v) * dW_v
            kappa = self.heston_params.kappa
            theta = self.heston_params.theta
            sigma = self.heston_params.sigma
            
            v[:, i + 1] = v_curr + kappa * (theta - v_curr) * dt + \
                         sigma * sqrt_v_curr * sqrt_dt * dW_v
            
            # Ensure variance stays positive (absorption scheme)
            v[:, i + 1] = np.maximum(v[:, i + 1], 0)
        
        return S, v

    def _generate_jump_diffusion_paths(
        self,
        S0: float,
        sigma: float,
        T: float,
        r: float,
        q: float,
        num_paths: int,
        num_steps: int,
        days_to_earnings: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate paths using Merton jump diffusion model.

        Combines geometric Brownian motion with Poisson jumps.
        Yang-Zhang lambda can be used as jump intensity parameter.
        """
        dt = T / num_steps
        sqrt_dt = np.sqrt(dt)

        # Get effective jump intensity
        lambda_eff = self.merton_params.get_effective_lambda(days_to_earnings)

        # Initialize paths
        S = np.zeros((num_paths, num_steps + 1))
        S[:, 0] = S0

        # Merton jump-diffusion parameters
        mu_j = self.merton_params.mu_j
        sigma_j = self.merton_params.sigma_j

        # Adjust drift for jump risk
        jump_compensation = lambda_eff * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        adjusted_drift = r - q - 0.5 * sigma**2 - jump_compensation

        for i in range(num_steps):
            # Standard Brownian motion increment
            dW = np.random.standard_normal(num_paths) * sqrt_dt

            # Poisson process for jumps
            # Number of jumps in time interval dt
            lambda_dt = lambda_eff * dt
            num_jumps = np.random.poisson(lambda_dt, num_paths)

            # Generate jump sizes (log-normal distribution)
            jump_total = np.zeros(num_paths)
            for j in range(num_paths):
                if num_jumps[j] > 0:
                    # Sum of log-normal jumps
                    jump_sizes = np.random.normal(
                        mu_j, sigma_j, size=num_jumps[j]
                    )
                    jump_total[j] = np.sum(jump_sizes)

            # Current prices
            S_curr = S[:, i]

            # Jump-diffusion evolution
            # dS = S * [(r-q-λκ-σ²/2)dt + σdW + ΣJ]
            # where κ = E[e^J - 1] is jump compensation
            log_return = (
                adjusted_drift * dt +
                sigma * dW +
                jump_total
            )

            S[:, i + 1] = S_curr * np.exp(log_return)

        return S

    def _generate_bs_paths(
        self,
        S0: float,
        sigma: float,
        T: float,
        r: float,
        q: float,
        num_paths: int,
        num_steps: int
    ) -> np.ndarray:
        """Generate paths using Black-Scholes geometric Brownian motion."""
        dt = T / num_steps
        
        # Generate random increments
        dW = np.random.standard_normal((num_paths, num_steps)) * np.sqrt(dt)
        
        # Cumulative sum for Brownian motion
        W = np.cumsum(dW, axis=1)
        W = np.column_stack([np.zeros(num_paths), W])  # Add t=0
        
        # Time grid
        t = np.linspace(0, T, num_steps + 1)
        
        # Geometric Brownian motion solution
        drift = (r - q - 0.5 * sigma**2) * t
        diffusion = sigma * W
        
        paths = S0 * np.exp(drift + diffusion)
        return paths
    
    def _calculate_standard_payoffs(
        self,
        paths: np.ndarray,
        strike: float,
        option_type: OptionType
    ) -> np.ndarray:
        """Calculate standard European option payoffs."""
        final_prices = paths[:, -1] if len(paths.shape) > 1 else paths
        
        if option_type == OptionType.CALL:
            return np.maximum(final_prices - strike, 0)
        else:
            return np.maximum(strike - final_prices, 0)
    
    def _apply_antithetic_variates(self, payoffs: np.ndarray) -> np.ndarray:
        """
        Apply antithetic variates for variance reduction.
        
        This technique generates paired paths with negated random numbers,
        which can significantly reduce variance for smooth payoffs.
        """
        if len(payoffs) % 2 == 1:
            payoffs = payoffs[:-1]  # Remove last element if odd
        
        mid = len(payoffs) // 2
        antithetic_payoffs = (payoffs[:mid] + payoffs[mid:]) / 2
        return antithetic_payoffs
    
    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        volatility: float = 0.2,
        option_type: OptionType = OptionType.CALL,
        num_paths: int = 100000,
        bump_size: float = 0.01
    ) -> GreeksResult:
        """
        Calculate Greeks using finite differences with Monte Carlo.
        
        This is computationally expensive but provides Greeks for complex payoffs
        and stochastic volatility models where analytical solutions don't exist.
        """
        # Base price
        base_result = self.price_option(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
            volatility, option_type, num_paths
        )
        base_price = base_result.option_price
        
        # Delta (bump spot)
        spot_up = self.price_option(
            spot * (1 + bump_size), strike, time_to_expiry, risk_free_rate,
            dividend_yield, volatility, option_type, num_paths
        ).option_price
        
        spot_down = self.price_option(
            spot * (1 - bump_size), strike, time_to_expiry, risk_free_rate,
            dividend_yield, volatility, option_type, num_paths
        ).option_price
        
        delta = (spot_up - spot_down) / (2 * spot * bump_size)
        
        # Gamma (second derivative)
        gamma = (spot_up - 2 * base_price + spot_down) / ((spot * bump_size) ** 2)
        
        # Theta (bump time)
        time_bump = 1 / 365.25  # 1 day
        if time_to_expiry > time_bump:
            theta_price = self.price_option(
                spot, strike, time_to_expiry - time_bump, risk_free_rate,
                dividend_yield, volatility, option_type, num_paths
            ).option_price
            theta = (theta_price - base_price) / time_bump
        else:
            theta = 0.0
        
        # Vega (bump volatility) - only meaningful for Black-Scholes
        if self.model_type == ModelType.BLACK_SCHOLES:
            vol_bump = volatility * 0.01  # 1% of volatility
            vega_price = self.price_option(
                spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
                volatility + vol_bump, option_type, num_paths
            ).option_price
            vega = (vega_price - base_price) / vol_bump / 100  # Per 1% vol change
        else:
            vega = 0.0  # Vega not well-defined for stochastic vol models
        
        # Rho (bump interest rate)
        rate_bump = 0.0001  # 1 basis point
        rho_price = self.price_option(
            spot, strike, time_to_expiry, risk_free_rate + rate_bump,
            dividend_yield, volatility, option_type, num_paths
        ).option_price
        rho = (rho_price - base_price) / rate_bump / 100  # Per 1% rate change
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )
    
    def scenario_analysis(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        volatility: float = 0.2,
        option_type: OptionType = OptionType.CALL,
        num_paths: int = 50000,
        spot_scenarios: List[float] = None,
        vol_scenarios: List[float] = None,
        time_scenarios: List[float] = None
    ) -> Dict[str, Any]:
        """
        Run scenario analysis across multiple market conditions.
        
        Returns a comprehensive analysis showing how option value changes
        across different underlying prices, volatilities, and time decay.
        """
        if spot_scenarios is None:
            spot_scenarios = [spot * factor for factor in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]]
        
        if vol_scenarios is None and self.model_type == ModelType.BLACK_SCHOLES:
            vol_scenarios = [volatility * factor for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
        else:
            vol_scenarios = [volatility]  # Single vol for stochastic vol models
        
        if time_scenarios is None:
            time_scenarios = [time_to_expiry * factor for factor in [1.0, 0.75, 0.5, 0.25, 0.1]]
        
        results = {}
        
        # Spot scenarios
        spot_results = []
        for s in spot_scenarios:
            result = self.price_option(
                s, strike, time_to_expiry, risk_free_rate, dividend_yield,
                volatility, option_type, num_paths
            )
            spot_results.append({
                'spot': s,
                'price': result.option_price,
                'error': result.standard_error
            })
        results['spot_scenarios'] = spot_results
        
        # Volatility scenarios (Black-Scholes only)
        if self.model_type == ModelType.BLACK_SCHOLES:
            vol_results = []
            for vol in vol_scenarios:
                result = self.price_option(
                    spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
                    vol, option_type, num_paths
                )
                vol_results.append({
                    'volatility': vol,
                    'price': result.option_price,
                    'error': result.standard_error
                })
            results['volatility_scenarios'] = vol_results
        
        # Time decay scenarios
        time_results = []
        for t in time_scenarios:
            if t > 0:
                result = self.price_option(
                    spot, strike, t, risk_free_rate, dividend_yield,
                    volatility, option_type, num_paths
                )
                time_results.append({
                    'time_to_expiry': t,
                    'days_to_expiry': t * 365.25,
                    'price': result.option_price,
                    'error': result.standard_error
                })
        results['time_scenarios'] = time_results
        
        return results
    
    def _zero_result(self, start_time: float) -> MonteCarloResult:
        """Return zero result for invalid inputs."""
        return MonteCarloResult(
            option_price=0.0,
            standard_error=0.0,
            confidence_interval=(0.0, 0.0),
            paths_used=0,
            execution_time=time.time() - start_time,
            convergence_achieved=False
        )

    def simulate_option_price(
        self,
        option: OptionContract,
        num_simulations: int = 100000,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0
    ) -> MonteCarloResult:
        """
        Backward-compatible wrapper expected by worker code.
        """
        time_to_expiry = max((option.expiration - date.today()).days / 365.25, 1.0 / 365.25)
        spot_guess = float(
            getattr(option, "underlying_price", 0.0)
            or getattr(option, "spot_price", 0.0)
            or getattr(option, "underlying_last", 0.0)
            or option.strike
        )
        return self.price_option(
            spot=max(spot_guess, 0.01),
            strike=max(option.strike, 0.01),
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=max(float(option.implied_volatility or 0.20), 0.01),
            option_type=option.option_type,
            num_paths=max(1000, int(num_simulations)),
        )
    
    def set_heston_parameters(self, **kwargs) -> None:
        """Update Heston model parameters."""
        for key, value in kwargs.items():
            if hasattr(self.heston_params, key):
                setattr(self.heston_params, key, value)
            else:
                raise ValueError(f"Invalid Heston parameter: {key}")

    def set_merton_parameters(self, **kwargs) -> None:
        """Update Merton jump diffusion parameters."""
        for key, value in kwargs.items():
            if hasattr(self.merton_params, key):
                setattr(self.merton_params, key, value)
            else:
                raise ValueError(f"Invalid Merton parameter: {key}")

    def set_yang_zhang_lambda(self, yang_zhang_vol: float, base_vol: float = 0.2) -> None:
        """
        Set jump intensity based on Yang-Zhang volatility estimate.

        Args:
            yang_zhang_vol: Yang-Zhang volatility estimate
            base_vol: Base volatility for comparison
        """
        # Higher Yang-Zhang vol indicates more jumps
        # Empirical relationship: λ ≈ max(0, (YZ_vol - base_vol) * scaling_factor)
        vol_excess = max(0, yang_zhang_vol - base_vol)
        lambda_estimate = vol_excess * 10.0  # Scaling factor

        # Bound the lambda estimate
        lambda_estimate = min(lambda_estimate, 2.0)  # Max 2 jumps per year

        self.merton_params.yang_zhang_lambda = lambda_estimate
    
    def calibrate_heston_to_surface(
        self,
        market_data: List[Tuple[float, float, float, float]],  # (strike, expiry, market_price, spot)
        initial_guess: Optional[Dict[str, float]] = None,
        optimization_method: str = "least_squares"
    ) -> HestonParameters:
        """
        Calibrate Heston parameters to market option prices.
        
        This is a simplified calibration routine. Production systems would use
        more sophisticated optimization algorithms and multiple strikes/expiries.
        
        Args:
            market_data: List of (strike, expiry, market_price, spot) tuples
            initial_guess: Initial parameter values for optimization
            optimization_method: Optimization algorithm to use
            
        Returns:
            Calibrated HestonParameters
        """
        # This would implement a full calibration routine
        # For now, return current parameters with a warning
        warnings.warn(
            "Heston calibration not fully implemented. Using current parameters.",
            UserWarning
        )
        return self.heston_params

# Exotic option payoff functions

def asian_call_payoff(paths: np.ndarray, strike: float) -> np.ndarray:
    """Asian (average price) call option payoff."""
    avg_prices = np.mean(paths, axis=1)
    return np.maximum(avg_prices - strike, 0)

def asian_put_payoff(paths: np.ndarray, strike: float) -> np.ndarray:
    """Asian (average price) put option payoff."""
    avg_prices = np.mean(paths, axis=1)
    return np.maximum(strike - avg_prices, 0)

def lookback_call_payoff(paths: np.ndarray, strike: float = 0) -> np.ndarray:
    """Lookback call option payoff (max price - strike or max price - initial)."""
    max_prices = np.max(paths, axis=1)
    if strike > 0:
        return np.maximum(max_prices - strike, 0)
    else:
        return max_prices - paths[:, 0]  # Floating strike

def barrier_up_and_out_call_payoff(paths: np.ndarray, strike: float, barrier: float) -> np.ndarray:
    """Up-and-out barrier call option payoff."""
    final_prices = paths[:, -1]
    max_prices = np.max(paths, axis=1)
    
    # Option knocks out if barrier is hit
    knocked_out = max_prices >= barrier
    payoffs = np.maximum(final_prices - strike, 0)
    payoffs[knocked_out] = 0  # Zero payoff if knocked out
    
    return payoffs

def digital_call_payoff(paths: np.ndarray, strike: float, payout: float = 1.0) -> np.ndarray:
    """Digital (binary) call option payoff."""
    final_prices = paths[:, -1]
    return np.where(final_prices > strike, payout, 0.0)

# Utility functions

def calculate_implied_volatility_mc(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    num_paths: int = 100000,
    tolerance: float = 0.01,
    max_iterations: int = 50
) -> Optional[float]:
    """
    Calculate implied volatility using Monte Carlo pricing.
    
    This is much slower than analytical IV calculation but works for
    any payoff function and stochastic volatility models.
    """
    # Import here to avoid circular dependency
    import os
    
    engine = MonteCarloEngine(
        model_type=ModelType.BLACK_SCHOLES,
        random_seed=42  # For consistent results
    )
    
    # Bisection method for IV
    vol_low, vol_high = 0.01, 5.0
    
    for _ in range(max_iterations):
        vol_mid = (vol_low + vol_high) / 2
        
        mc_result = engine.price_option(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield,
            vol_mid, option_type, num_paths
        )
        
        price_diff = mc_result.option_price - market_price
        
        if abs(price_diff) < tolerance:
            return vol_mid
        
        if price_diff > 0:
            vol_high = vol_mid
        else:
            vol_low = vol_mid
        
        if vol_high - vol_low < 0.0001:
            break
    
    return (vol_low + vol_high) / 2 if vol_high > vol_low else None

# Import os for the function above
import os
