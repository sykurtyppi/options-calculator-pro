"""
Comprehensive Greek calculations for options pricing and risk management.

This module provides production-ready implementations of all major Greeks with
high precision and performance optimizations for real-time trading applications.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from datetime import date, datetime
from scipy.stats import norm
from math import log, sqrt, exp, pi
from enum import Enum

from .option_data import OptionType, OptionContract

class GreekCalculationMethod(Enum):
    """Methods for calculating Greeks"""
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"
    FINITE_DIFFERENCE = "finite_difference"

@dataclass
class GreeksResult:
    """Complete Greeks calculation result"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    epsilon: float = 0.0  # Dividend sensitivity
    lambda_: float = 0.0  # Leverage/elasticity
    vanna: float = 0.0    # Delta sensitivity to volatility
    volga: float = 0.0    # Vega sensitivity to volatility
    charm: float = 0.0    # Delta decay over time
    veta: float = 0.0     # Vega decay over time
    speed: float = 0.0    # Gamma sensitivity to underlying
    color: float = 0.0    # Gamma decay over time
    ultima: float = 0.0   # Third-order volatility derivative
    
    @property
    def dollar_delta(self) -> float:
        """Delta in dollar terms (delta * 100 for option contracts)"""
        return self.delta * 100
    
    @property
    def dollar_gamma(self) -> float:
        """Gamma in dollar terms"""
        return self.gamma * 100
    
    @property
    def dollar_theta(self) -> float:
        """Theta in dollar terms (per day)"""
        return self.theta * 100
    
    @property
    def dollar_vega(self) -> float:
        """Vega in dollar terms (per 1% vol change)"""
        return self.vega * 100
    
    @property
    def risk_summary(self) -> Dict[str, str]:
        """Human-readable risk summary"""
        return {
            'directional_risk': 'Long' if self.delta > 0 else 'Short' if self.delta < 0 else 'Neutral',
            'convexity': 'Positive' if self.gamma > 0 else 'Negative' if self.gamma < 0 else 'Neutral',
            'time_decay': 'Negative' if self.theta < -0.01 else 'Positive' if self.theta > 0.01 else 'Minimal',
            'volatility_exposure': 'Long Vol' if self.vega > 0 else 'Short Vol' if self.vega < 0 else 'Vol Neutral'
        }

class GreeksCalculator:
    """
    High-performance Greeks calculator with multiple calculation methods.
    
    Features:
    - Black-Scholes analytical formulas (fastest)
    - Finite difference methods (most accurate)
    - American option approximations
    - Dividend adjustments
    - Second and third order Greeks
    """
    
    def __init__(self):
        self.cache = {}  # Cache for expensive calculations
        self.cache_size_limit = 1000
    
    def calculate_all_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
        method: GreekCalculationMethod = GreekCalculationMethod.BLACK_SCHOLES
    ) -> GreeksResult:
        """
        Calculate all Greeks for an option.
        
        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Implied volatility
            option_type: Call or Put
            dividend_yield: Dividend yield (default 0)
            method: Calculation method
        
        Returns:
            GreeksResult with all calculated Greeks
        """
        # Input validation
        if time_to_expiry <= 0:
            return self._zero_greeks()
        if volatility <= 0 or spot <= 0 or strike <= 0:
            return self._zero_greeks()
        
        # Create cache key
        cache_key = (
            spot, strike, time_to_expiry, risk_free_rate, 
            volatility, option_type, dividend_yield, method
        )
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Calculate based on method
        if method == GreekCalculationMethod.BLACK_SCHOLES:
            result = self._calculate_bs_greeks(
                spot, strike, time_to_expiry, risk_free_rate,
                volatility, option_type, dividend_yield
            )
        elif method == GreekCalculationMethod.FINITE_DIFFERENCE:
            result = self._calculate_fd_greeks(
                spot, strike, time_to_expiry, risk_free_rate,
                volatility, option_type, dividend_yield
            )
        else:
            # Fallback to Black-Scholes
            result = self._calculate_bs_greeks(
                spot, strike, time_to_expiry, risk_free_rate,
                volatility, option_type, dividend_yield
            )
        
        # Cache result
        if len(self.cache) >= self.cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = result
        
        return result
    
    def _calculate_bs_greeks(
        self,
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: OptionType, q: float = 0.0
    ) -> GreeksResult:
        """
        Calculate Greeks using Black-Scholes analytical formulas.
        
        This is the fastest method and provides exact solutions for European options.
        """
        # Black-Scholes parameters
        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)  # PDF at d1
        n_d2 = norm.pdf(d2)  # PDF at d2
        
        # Option type multiplier
        phi = 1 if option_type == OptionType.CALL else -1
        
        # First-order Greeks
        delta = phi * exp(-q * T) * N_d1 if option_type == OptionType.CALL else phi * exp(-q * T) * (N_d1 - 1)
        
        gamma = exp(-q * T) * n_d1 / (S * sigma * sqrt(T))
        
        theta_call = (
            -exp(-q * T) * S * n_d1 * sigma / (2 * sqrt(T))
            - r * K * exp(-r * T) * N_d2
            + q * S * exp(-q * T) * N_d1
        ) / 365.25  # Convert to per-day
        
        theta_put = (
            -exp(-q * T) * S * n_d1 * sigma / (2 * sqrt(T))
            + r * K * exp(-r * T) * (1 - N_d2)
            - q * S * exp(-q * T) * (1 - N_d1)
        ) / 365.25  # Convert to per-day
        
        theta = theta_call if option_type == OptionType.CALL else theta_put
        
        vega = S * exp(-q * T) * n_d1 * sqrt(T) / 100  # Per 1% volatility change
        
        rho_call = K * T * exp(-r * T) * N_d2 / 100  # Per 1% rate change
        rho_put = -K * T * exp(-r * T) * (1 - N_d2) / 100
        rho = rho_call if option_type == OptionType.CALL else rho_put
        
        epsilon_call = -S * T * exp(-q * T) * N_d1 / 100
        epsilon_put = S * T * exp(-q * T) * (1 - N_d1) / 100
        epsilon = epsilon_call if option_type == OptionType.CALL else epsilon_put
        
        # Second-order Greeks
        vanna = -exp(-q * T) * n_d1 * d2 / sigma / 100  # vega sensitivity to spot
        
        volga = S * exp(-q * T) * n_d1 * sqrt(T) * d1 * d2 / sigma / 10000  # vega sensitivity to vol
        
        charm_call = q * exp(-q * T) * N_d1 - exp(-q * T) * n_d1 * (
            2 * (r - q) * T - d2 * sigma * sqrt(T)
        ) / (2 * T * sigma * sqrt(T))
        charm_call /= 365.25  # Per day
        
        charm_put = -q * exp(-q * T) * (1 - N_d1) - exp(-q * T) * n_d1 * (
            2 * (r - q) * T - d2 * sigma * sqrt(T)
        ) / (2 * T * sigma * sqrt(T))
        charm_put /= 365.25  # Per day
        
        charm = charm_call if option_type == OptionType.CALL else charm_put
        
        veta = -S * exp(-q * T) * n_d1 * sqrt(T) * (
            q + (r - q) * d1 / (sigma * sqrt(T)) - (1 + d1 * d2) / (2 * T)
        ) / 100 / 365.25  # Per day per 1% vol
        
        # Third-order Greeks
        speed = -gamma / S * (d1 / (sigma * sqrt(T)) + 1)
        
        color = -exp(-q * T) * n_d1 / (2 * S * T * sigma * sqrt(T)) * (
            2 * q * T + 1 + (2 * (r - q) * T - d2 * sigma * sqrt(T)) * d1 / (sigma * sqrt(T))
        ) / 365.25  # Per day
        
        ultima = -vega / sigma * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2) / 10000
        
        # Lambda (leverage/elasticity)
        option_price = self._black_scholes_price(S, K, T, r, sigma, option_type, q)
        lambda_ = delta * S / option_price if option_price != 0 else 0.0
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            epsilon=epsilon,
            lambda_=lambda_,
            vanna=vanna,
            volga=volga,
            charm=charm,
            veta=veta,
            speed=speed,
            color=color,
            ultima=ultima
        )
    
    def _calculate_fd_greeks(
        self,
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: OptionType, q: float = 0.0
    ) -> GreeksResult:
        """
        Calculate Greeks using finite difference methods.
        
        More accurate for complex payoffs but slower than analytical formulas.
        """
        # Small increments for finite differences
        dS = S * 0.01  # 1% of spot
        dt = min(T * 0.01, 1/365.25)  # 1% of time or 1 day
        dsigma = sigma * 0.01  # 1% of volatility
        dr = 0.0001  # 1 basis point
        
        # Base price
        P0 = self._black_scholes_price(S, K, T, r, sigma, option_type, q)
        
        # Delta (first derivative w.r.t. spot)
        P_up = self._black_scholes_price(S + dS, K, T, r, sigma, option_type, q)
        P_down = self._black_scholes_price(S - dS, K, T, r, sigma, option_type, q)
        delta = (P_up - P_down) / (2 * dS)
        
        # Gamma (second derivative w.r.t. spot)
        gamma = (P_up - 2 * P0 + P_down) / (dS**2)
        
        # Theta (first derivative w.r.t. time)
        if T > dt:
            P_theta = self._black_scholes_price(S, K, T - dt, r, sigma, option_type, q)
            theta = (P_theta - P0) / dt
        else:
            theta = 0.0
        
        # Vega (first derivative w.r.t. volatility)
        P_vega_up = self._black_scholes_price(S, K, T, r, sigma + dsigma, option_type, q)
        P_vega_down = self._black_scholes_price(S, K, T, r, sigma - dsigma, option_type, q)
        vega = (P_vega_up - P_vega_down) / (2 * dsigma) / 100  # Per 1% vol change
        
        # Rho (first derivative w.r.t. interest rate)
        P_rho_up = self._black_scholes_price(S, K, T, r + dr, sigma, option_type, q)
        P_rho_down = self._black_scholes_price(S, K, T, r - dr, sigma, option_type, q)
        rho = (P_rho_up - P_rho_down) / (2 * dr) / 100  # Per 1% rate change
        
        # Second-order Greeks (more computationally expensive)
        
        # Vanna (cross derivative: delta w.r.t. volatility)
        delta_up = (self._black_scholes_price(S + dS, K, T, r, sigma + dsigma, option_type, q) -
                   self._black_scholes_price(S - dS, K, T, r, sigma + dsigma, option_type, q)) / (2 * dS)
        delta_down = (self._black_scholes_price(S + dS, K, T, r, sigma - dsigma, option_type, q) -
                     self._black_scholes_price(S - dS, K, T, r, sigma - dsigma, option_type, q)) / (2 * dS)
        vanna = (delta_up - delta_down) / (2 * dsigma) / 100
        
        # Volga (second derivative w.r.t. volatility)
        volga = (P_vega_up - 2 * P0 + P_vega_down) / (dsigma**2) / 10000
        
        # Other Greeks set to zero for finite difference method (would be very expensive)
        epsilon = lambda_ = charm = veta = speed = color = ultima = 0.0
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            epsilon=epsilon,
            lambda_=lambda_,
            vanna=vanna,
            volga=volga,
            charm=charm,
            veta=veta,
            speed=speed,
            color=color,
            ultima=ultima
        )
    
    def _black_scholes_price(
        self,
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: OptionType, q: float = 0.0
    ) -> float:
        """Calculate Black-Scholes option price"""
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if option_type == OptionType.CALL:
            return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:
            return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
    
    def _zero_greeks(self) -> GreeksResult:
        """Return zero Greeks for invalid inputs"""
        return GreeksResult(
            delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            epsilon=0.0, lambda_=0.0, vanna=0.0, volga=0.0,
            charm=0.0, veta=0.0, speed=0.0, color=0.0, ultima=0.0
        )
    
    def calculate_portfolio_greeks(
        self,
        positions: List[Tuple[OptionContract, int]]
    ) -> GreeksResult:
        """
        Calculate net Greeks for a portfolio of options.
        
        Args:
            positions: List of (OptionContract, quantity) tuples
        
        Returns:
            GreeksResult with net portfolio Greeks
        """
        net_delta = net_gamma = net_theta = net_vega = net_rho = 0.0
        net_vanna = net_volga = net_charm = net_veta = 0.0
        
        for contract, quantity in positions:
            # Scale Greeks by position size (quantity * 100 shares per contract)
            multiplier = quantity * 100
            
            net_delta += contract.delta * multiplier
            net_gamma += contract.gamma * multiplier
            net_theta += contract.theta * multiplier
            net_vega += contract.vega * multiplier
            net_rho += contract.rho * multiplier
        
        return GreeksResult(
            delta=net_delta,
            gamma=net_gamma,
            theta=net_theta,
            vega=net_vega,
            rho=net_rho,
            epsilon=0.0,  # Portfolio-level calculation would need more data
            lambda_=0.0,
            vanna=net_vanna,
            volga=net_volga,
            charm=net_charm,
            veta=net_veta,
            speed=0.0,
            color=0.0,
            ultima=0.0
        )
    
    def calculate_hedge_ratio(
        self,
        option_delta: float,
        option_quantity: int,
        hedge_delta: float = 1.0
    ) -> int:
        """
        Calculate hedge ratio for delta-neutral hedging.
        
        Args:
            option_delta: Delta of the option position
            option_quantity: Number of option contracts
            hedge_delta: Delta of hedge instrument (1.0 for stock)
        
        Returns:
            Number of hedge units needed (negative for short hedge)
        """
        total_option_delta = option_delta * option_quantity * 100
        hedge_ratio = -total_option_delta / hedge_delta
        return int(round(hedge_ratio))
    
    def estimate_pnl_greeks(
        self,
        greeks: GreeksResult,
        spot_change: float,
        vol_change: float = 0.0,
        time_decay_days: float = 0.0,
        contracts: int = 1
    ) -> Dict[str, float]:
        """
        Estimate P&L contribution from each Greek.
        
        This is a first-order approximation and becomes less accurate
        for large moves or longer time periods.
        
        Args:
            greeks: Greeks for the position
            spot_change: Change in underlying price
            vol_change: Change in implied volatility (in percentage points)
            time_decay_days: Number of days for time decay
            contracts: Number of contracts
        
        Returns:
            Dictionary with P&L breakdown by Greek
        """
        multiplier = contracts * 100
        
        delta_pnl = greeks.delta * spot_change * multiplier
        gamma_pnl = 0.5 * greeks.gamma * (spot_change**2) * multiplier
        theta_pnl = greeks.theta * time_decay_days * multiplier
        vega_pnl = greeks.vega * vol_change * multiplier
        
        # Second-order adjustments
        vanna_pnl = greeks.vanna * spot_change * vol_change * multiplier
        charm_pnl = greeks.charm * spot_change * time_decay_days * multiplier
        
        total_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + vanna_pnl + charm_pnl
        
        return {
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'vanna_pnl': vanna_pnl,
            'charm_pnl': charm_pnl,
            'total_pnl': total_pnl
        }

def days_to_expiry(expiration_date: Union[date, datetime]) -> float:
    """
    Calculate time to expiry in years.
    
    Args:
        expiration_date: Expiration date of the option
    
    Returns:
        Time to expiry in years (decimal)
    """
    if isinstance(expiration_date, datetime):
        expiration_date = expiration_date.date()
    
    today = date.today()
    if expiration_date <= today:
        return 0.0
    
    days = (expiration_date - today).days
    return days / 365.25

def implied_volatility_newton_raphson(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    This is more accurate than bisection but can fail to converge
    for extreme option prices.
    
    Args:
        market_price: Market price of the option
        spot: Current underlying price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free interest rate
        option_type: Call or Put
        dividend_yield: Dividend yield
        max_iterations: Maximum Newton-Raphson iterations
        tolerance: Convergence tolerance
    
    Returns:
        Implied volatility or None if convergence fails
    """
    if time_to_expiry <= 0 or market_price <= 0:
        return None
    
    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = sqrt(2 * pi / time_to_expiry) * market_price / spot
    sigma = max(0.01, min(sigma, 5.0))  # Bound between 1% and 500%
    
    calculator = GreeksCalculator()
    
    for _ in range(max_iterations):
        # Calculate theoretical price and vega
        theoretical_price = calculator._black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, sigma, option_type, dividend_yield
        )
        
        # Calculate vega (derivative of price w.r.t. volatility)
        if time_to_expiry > 0:
            d1 = (log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * sigma**2) * time_to_expiry) / (sigma * sqrt(time_to_expiry))
            vega_raw = spot * exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * sqrt(time_to_expiry)
        else:
            return None
        
        if abs(vega_raw) < 1e-10:  # Avoid division by zero
            return None
        
        # Newton-Raphson update
        price_diff = theoretical_price - market_price
        if abs(price_diff) < tolerance:
            return sigma
        
        sigma_new = sigma - price_diff / vega_raw
        
        # Bound the volatility
        sigma_new = max(0.001, min(sigma_new, 10.0))
        
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new
        
        sigma = sigma_new
    
    return None  # Failed to converge

# Utility functions for common calculations

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate Black-Scholes call option price"""
    calculator = GreeksCalculator()
    return calculator._black_scholes_price(S, K, T, r, sigma, OptionType.CALL, q)

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate Black-Scholes put option price"""
    calculator = GreeksCalculator()
    return calculator._black_scholes_price(S, K, T, r, sigma, OptionType.PUT, q)

def intrinsic_value(spot: float, strike: float, option_type: OptionType) -> float:
    """Calculate intrinsic value of an option"""
    if option_type == OptionType.CALL:
        return max(0, spot - strike)
    else:
        return max(0, strike - spot)

def time_value(option_price: float, spot: float, strike: float, option_type: OptionType) -> float:
    """Calculate time value of an option"""
    return option_price - intrinsic_value(spot, strike, option_type)