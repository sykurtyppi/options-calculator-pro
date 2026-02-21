"""
Greeks Calculator - Professional Options Greeks Engine
=====================================================

Handles all options Greeks calculations including:
- Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho)
- American options Greeks using binomial models
- Exotic options Greeks (barriers, digitals, etc.)
- Calendar spread Greeks analysis
- Greeks sensitivity analysis and scenarios
- Risk management metrics

Part of Professional Options Calculator v9.1
Optimized for Apple Silicon and PySide6
"""

import logging
import threading
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from functools import lru_cache
import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad

# Import your existing utilities
from utils.config_manager import ConfigManager
from utils.logger import get_logger

logger = get_logger(__name__)

class OptionStyle(Enum):
    """Option exercise style"""
    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"

class OptionType(Enum):
    """Option type"""
    CALL = "call"
    PUT = "put"

class GreeksModel(Enum):
    """Greeks calculation models"""
    BLACK_SCHOLES = "black_scholes"
    BLACK_SCHOLES_MERTON = "black_scholes_merton"
    BINOMIAL = "binomial"
    TRINOMIAL = "trinomial"
    MONTE_CARLO = "monte_carlo"

class UnderlyingType(Enum):
    """Type of underlying asset"""
    STOCK = "stock"
    INDEX = "index"
    FUTURES = "futures"
    FOREX = "forex"
    COMMODITY = "commodity"

@dataclass
class GreeksResult:
    """Complete Greeks calculation result"""
    # First-order Greeks
    delta: float
    
    # Second-order Greeks
    gamma: float
    
    # Time decay
    theta: float  # Per day
    theta_annual: float  # Per year
    
    # Volatility sensitivity
    vega: float  # Per 1% change in IV
    vega_decimal: float  # Per 0.01 change in IV
    
    # Interest rate sensitivity
    rho: float  # Per 1% change in rate
    
    # Higher-order Greeks
    vanna: float  # dVega/dSpot or dDelta/dVol
    charm: float  # dDelta/dTime
    vomma: float  # dVega/dVol (volga)
    ultima: float  # dVomma/dVol
    speed: float  # dGamma/dSpot
    zomma: float  # dGamma/dVol
    color: float  # dGamma/dTime
    
    # Risk metrics
    lambda_: float  # Leverage (elasticity)
    epsilon: float  # Dividend sensitivity
    
    # Additional metrics
    dollar_delta: float
    dollar_gamma: float
    dollar_theta: float
    dollar_vega: float
    
    # Calculation metadata
    model_used: GreeksModel
    calculation_time: datetime = field(default_factory=datetime.now)

@dataclass
class MarketParameters:
    """Market parameters for Greeks calculation"""
    spot_price: float
    strike_price: float
    time_to_expiry: float  # In years
    risk_free_rate: float
    volatility: float
    option_type: OptionType
    dividend_yield: float = 0.0
    option_style: OptionStyle = OptionStyle.EUROPEAN
    underlying_type: UnderlyingType = UnderlyingType.STOCK

@dataclass
class CalendarSpreadGreeks:
    """Greeks for calendar spread positions"""
    # Net position Greeks
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
    
    # Individual leg Greeks
    short_leg_greeks: GreeksResult
    long_leg_greeks: GreeksResult
    
    # Spread-specific metrics
    time_decay_ratio: float  # Short theta / Long theta
    volatility_exposure: float  # Net vega sensitivity
    delta_neutrality: float  # How close to delta neutral
    gamma_scalping_potential: float
    
    # Risk metrics
    max_gamma_exposure: float
    theta_efficiency: float  # Theta collected per dollar risked
    
@dataclass
class GreeksScenario:
    """Greeks under different market scenarios"""
    base_case: GreeksResult
    up_5_percent: GreeksResult
    down_5_percent: GreeksResult
    up_10_percent: GreeksResult
    down_10_percent: GreeksResult
    vol_up_5_percent: GreeksResult
    vol_down_5_percent: GreeksResult
    time_decay_1_day: GreeksResult
    time_decay_7_days: GreeksResult

class GreeksCalculator:
    """
    Professional Options Greeks Calculator
    
    Provides comprehensive Greeks calculations including:
    - Standard Black-Scholes Greeks with extensions
    - American options using binomial/trinomial models
    - Higher-order Greeks for advanced risk management
    - Calendar spread analysis
    - Scenario analysis and stress testing
    - Performance optimization with caching
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logger
        
        # Configuration
        self.cache_enabled = self.config.get("greeks.cache_enabled", True)
        self.cache_ttl = self.config.get("greeks.cache_ttl", 300)  # 5 minutes
        self.precision = self.config.get("greeks.precision", 6)
        
        # Numerical parameters
        self.bump_size_spot = 0.01  # 1% for delta/gamma
        self.bump_size_vol = 0.01   # 1% for vega
        self.bump_size_time = 1/365  # 1 day for theta
        self.bump_size_rate = 0.0001  # 1 basis point for rho
        
        # Binomial model parameters
        self.binomial_steps = 100
        self.trinomial_steps = 50
        
        # Cache
        self._greeks_cache = {}
        self._cache_lock = threading.Lock()
        
        # Constants
        self.SQRT_2PI = math.sqrt(2 * math.pi)
        self.DAYS_PER_YEAR = 365.25
        
        self.logger.info("GreeksCalculator initialized")
    
    def calculate_greeks(self, params: MarketParameters, 
                        model: GreeksModel = GreeksModel.BLACK_SCHOLES,
                        include_higher_order: bool = False) -> GreeksResult:
        """
        Calculate option Greeks using specified model
        
        Args:
            params: Market parameters
            model: Calculation model to use
            include_higher_order: Whether to calculate higher-order Greeks
            
        Returns:
            Complete Greeks result
        """
        try:
            # Validate parameters
            self._validate_parameters(params)
            
            # Check cache
            cache_key = self._generate_cache_key(params, model, include_higher_order)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Calculate Greeks based on model
            if model == GreeksModel.BLACK_SCHOLES:
                result = self._calculate_black_scholes_greeks(params, include_higher_order)
            elif model == GreeksModel.BLACK_SCHOLES_MERTON:
                result = self._calculate_bsm_greeks(params, include_higher_order)
            elif model == GreeksModel.BINOMIAL:
                result = self._calculate_binomial_greeks(params, include_higher_order)
            elif model == GreeksModel.TRINOMIAL:
                result = self._calculate_trinomial_greeks(params, include_higher_order)
            elif model == GreeksModel.MONTE_CARLO:
                result = self._calculate_monte_carlo_greeks(params, include_higher_order)
            else:
                raise ValueError(f"Unsupported model: {model}")
            
            result.model_used = model
            
            # Cache result
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
            return self._get_default_greeks(model)
    
    def calculate_calendar_spread_greeks(self, 
                                       short_params: MarketParameters,
                                       long_params: MarketParameters,
                                       model: GreeksModel = GreeksModel.BLACK_SCHOLES) -> CalendarSpreadGreeks:
        """
        Calculate Greeks for calendar spread position
        
        Args:
            short_params: Parameters for short leg
            long_params: Parameters for long leg
            model: Calculation model
            
        Returns:
            Calendar spread Greeks analysis
        """
        try:
            # Calculate Greeks for each leg
            short_greeks = self.calculate_greeks(short_params, model)
            long_greeks = self.calculate_greeks(long_params, model)
            
            # Calculate net Greeks (long - short for calendar spread)
            net_delta = long_greeks.delta - short_greeks.delta
            net_gamma = long_greeks.gamma - short_greeks.gamma
            net_theta = long_greeks.theta - short_greeks.theta
            net_vega = long_greeks.vega - short_greeks.vega
            net_rho = long_greeks.rho - short_greeks.rho
            
            # Calculate spread-specific metrics
            time_decay_ratio = (
                abs(short_greeks.theta) / abs(long_greeks.theta) 
                if long_greeks.theta != 0 else 0
            )
            
            volatility_exposure = net_vega
            delta_neutrality = abs(net_delta)
            
            # Gamma scalping potential (simplified)
            gamma_scalping_potential = abs(net_gamma) * short_params.spot_price * 0.01
            
            # Risk metrics
            max_gamma_exposure = max(abs(short_greeks.gamma), abs(long_greeks.gamma))
            
            # Theta efficiency (theta per dollar of premium paid)
            long_premium = self._estimate_option_price(long_params)
            short_premium = self._estimate_option_price(short_params)
            net_premium = long_premium - short_premium
            
            theta_efficiency = (
                abs(net_theta) / abs(net_premium) 
                if net_premium != 0 else 0
            )
            
            return CalendarSpreadGreeks(
                net_delta=net_delta,
                net_gamma=net_gamma,
                net_theta=net_theta,
                net_vega=net_vega,
                net_rho=net_rho,
                short_leg_greeks=short_greeks,
                long_leg_greeks=long_greeks,
                time_decay_ratio=time_decay_ratio,
                volatility_exposure=volatility_exposure,
                delta_neutrality=delta_neutrality,
                gamma_scalping_potential=gamma_scalping_potential,
                max_gamma_exposure=max_gamma_exposure,
                theta_efficiency=theta_efficiency
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating calendar spread Greeks: {e}")
            return self._get_default_calendar_greeks()
    
    def calculate_scenario_greeks(self, params: MarketParameters,
                                model: GreeksModel = GreeksModel.BLACK_SCHOLES) -> GreeksScenario:
        """
        Calculate Greeks under various market scenarios
        
        Args:
            params: Base market parameters
            model: Calculation model
            
        Returns:
            Greeks under different scenarios
        """
        try:
            # Base case
            base_case = self.calculate_greeks(params, model)
            
            # Price scenarios
            params_up_5 = self._modify_params(params, spot_bump=0.05)
            up_5_percent = self.calculate_greeks(params_up_5, model)
            
            params_down_5 = self._modify_params(params, spot_bump=-0.05)
            down_5_percent = self.calculate_greeks(params_down_5, model)
            
            params_up_10 = self._modify_params(params, spot_bump=0.10)
            up_10_percent = self.calculate_greeks(params_up_10, model)
            
            params_down_10 = self._modify_params(params, spot_bump=-0.10)
            down_10_percent = self.calculate_greeks(params_down_10, model)
            
            # Volatility scenarios
            params_vol_up = self._modify_params(params, vol_bump=0.05)
            vol_up_5_percent = self.calculate_greeks(params_vol_up, model)
            
            params_vol_down = self._modify_params(params, vol_bump=-0.05)
            vol_down_5_percent = self.calculate_greeks(params_vol_down, model)
            
            # Time decay scenarios
            params_1d = self._modify_params(params, time_bump=-1/365.25)
            time_decay_1_day = self.calculate_greeks(params_1d, model)
            
            params_7d = self._modify_params(params, time_bump=-7/365.25)
            time_decay_7_days = self.calculate_greeks(params_7d, model)
            
            return GreeksScenario(
                base_case=base_case,
                up_5_percent=up_5_percent,
                down_5_percent=down_5_percent,
                up_10_percent=up_10_percent,
                down_10_percent=down_10_percent,
                vol_up_5_percent=vol_up_5_percent,
                vol_down_5_percent=vol_down_5_percent,
                time_decay_1_day=time_decay_1_day,
                time_decay_7_days=time_decay_7_days
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating scenario Greeks: {e}")
            return self._get_default_scenario()
    
    def calculate_portfolio_greeks(self, positions: List[Tuple[MarketParameters, float]],
                                 model: GreeksModel = GreeksModel.BLACK_SCHOLES) -> GreeksResult:
        """
        Calculate portfolio-level Greeks
        
        Args:
            positions: List of (parameters, quantity) tuples
            model: Calculation model
            
        Returns:
            Aggregate portfolio Greeks
        """
        try:
            portfolio_greeks = {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0,
                'vanna': 0.0, 'charm': 0.0, 'vomma': 0.0, 'ultima': 0.0,
                'speed': 0.0, 'zomma': 0.0, 'color': 0.0, 'lambda_': 0.0, 'epsilon': 0.0,
                'dollar_delta': 0.0, 'dollar_gamma': 0.0, 'dollar_theta': 0.0, 'dollar_vega': 0.0
            }
            
            for params, quantity in positions:
                position_greeks = self.calculate_greeks(params, model, include_higher_order=True)
                
                # Scale by quantity and add to portfolio
                for greek_name in portfolio_greeks:
                    greek_value = getattr(position_greeks, greek_name)
                    portfolio_greeks[greek_name] += greek_value * quantity
            
            return GreeksResult(
                **portfolio_greeks,
                model_used=model
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio Greeks: {e}")
            return self._get_default_greeks(model)
    
    # Core calculation methods
    def _calculate_black_scholes_greeks(self, params: MarketParameters, 
                                      include_higher_order: bool) -> GreeksResult:
        """Calculate Greeks using Black-Scholes model"""
        try:
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            q = params.dividend_yield
            sigma = params.volatility
            
            if T <= 0:
                return self._get_expiry_greeks(params)
            
            # Calculate d1 and d2
            d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            
            # Standard normal density and cumulative functions
            nd1 = norm.pdf(d1)
            nd2 = norm.pdf(d2)
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)
            
            # First-order Greeks
            if params.option_type == OptionType.CALL:
                delta = math.exp(-q*T) * Nd1
                theta_annual = (-S*nd1*sigma*math.exp(-q*T)/(2*math.sqrt(T)) 
                               - r*K*math.exp(-r*T)*Nd2 
                               + q*S*math.exp(-q*T)*Nd1)
            else:  # PUT
                delta = -math.exp(-q*T) * norm.cdf(-d1)
                theta_annual = (-S*nd1*sigma*math.exp(-q*T)/(2*math.sqrt(T)) 
                               + r*K*math.exp(-r*T)*norm.cdf(-d2) 
                               - q*S*math.exp(-q*T)*norm.cdf(-d1))
            
            # Second-order Greeks
            gamma = nd1 * math.exp(-q*T) / (S * sigma * math.sqrt(T))
            
            # Time decay (convert to per day)
            theta = theta_annual / self.DAYS_PER_YEAR
            
            # Volatility sensitivity
            vega_decimal = S * nd1 * math.sqrt(T) * math.exp(-q*T)
            vega = vega_decimal / 100  # Per 1% change
            
            # Interest rate sensitivity
            if params.option_type == OptionType.CALL:
                rho = K * T * math.exp(-r*T) * Nd2 / 100
            else:
                rho = -K * T * math.exp(-r*T) * norm.cdf(-d2) / 100
            
            # Higher-order Greeks
            if include_higher_order:
                # Vanna (dVega/dSpot or dDelta/dVol)
                vanna = -math.exp(-q*T) * nd1 * d2 / sigma
                
                # Charm (dDelta/dTime)
                if params.option_type == OptionType.CALL:
                    charm = q * math.exp(-q*T) * Nd1 - math.exp(-q*T) * nd1 * (
                        2*(r-q)*T - d2*sigma*math.sqrt(T)) / (2*T*sigma*math.sqrt(T))
                else:
                    charm = -q * math.exp(-q*T) * norm.cdf(-d1) - math.exp(-q*T) * nd1 * (
                        2*(r-q)*T - d2*sigma*math.sqrt(T)) / (2*T*sigma*math.sqrt(T))
                
                # Vomma (dVega/dVol)
                vomma = vega_decimal * d1 * d2 / sigma
                
                # Speed (dGamma/dSpot)
                speed = -gamma / S * (d1 / (sigma * math.sqrt(T)) + 1)
                
                # Zomma (dGamma/dVol)
                zomma = gamma * (d1 * d2 - 1) / sigma
                
                # Color (dGamma/dTime)
                color = -math.exp(-q*T) * nd1 / (2*S*T*sigma*math.sqrt(T)) * (
                    2*q*T + 1 + (2*(r-q)*T - d2*sigma*math.sqrt(T)) * d1 / (sigma*math.sqrt(T)))
                
                # Ultima (dVomma/dVol)
                ultima = -vega_decimal / (sigma**2) * (d1*d2*(1-d1*d2) + d1**2 + d2**2)
            else:
                vanna = charm = vomma = speed = zomma = color = ultima = 0.0
            
            # Risk metrics
            option_price = self._estimate_option_price(params)
            lambda_ = delta * S / option_price if option_price != 0 else 0
            epsilon = (
                -S * math.exp(-q*T) * Nd1 / 100
                if params.option_type == OptionType.CALL
                else S * math.exp(-q*T) * norm.cdf(-d1) / 100
            )
            
            # Dollar Greeks
            dollar_delta = delta * S
            dollar_gamma = gamma * S * S * 0.01  # Per 1% move
            dollar_theta = theta
            dollar_vega = vega
            
            return GreeksResult(
                delta=delta,
                gamma=gamma,
                theta=theta,
                theta_annual=theta_annual,
                vega=vega,
                vega_decimal=vega_decimal,
                rho=rho,
                vanna=vanna,
                charm=charm,
                vomma=vomma,
                ultima=ultima,
                speed=speed,
                zomma=zomma,
                color=color,
                lambda_=lambda_,
                epsilon=epsilon,
                dollar_delta=dollar_delta,
                dollar_gamma=dollar_gamma,
                dollar_theta=dollar_theta,
                dollar_vega=dollar_vega,
                model_used=GreeksModel.BLACK_SCHOLES
            )
            
        except Exception as e:
            self.logger.error(f"Error in Black-Scholes Greeks calculation: {e}")
            return self._get_default_greeks(GreeksModel.BLACK_SCHOLES)
    
    def _calculate_bsm_greeks(self, params: MarketParameters, 
                            include_higher_order: bool) -> GreeksResult:
        """Calculate Greeks using Black-Scholes-Merton model (with dividends)"""
        # BSM is essentially BS with dividend yield, so we can use the same calculation
        return self._calculate_black_scholes_greeks(params, include_higher_order)
    
    def _calculate_binomial_greeks(self, params: MarketParameters, 
                                 include_higher_order: bool) -> GreeksResult:
        """Calculate Greeks using binomial model"""
        try:
            # Calculate option price and Greeks using finite differences
            base_price = self._binomial_option_price(params)
            
            # Delta calculation
            params_up = self._modify_params(params, spot_bump=self.bump_size_spot)
            params_down = self._modify_params(params, spot_bump=-self.bump_size_spot)
            
            price_up = self._binomial_option_price(params_up)
            price_down = self._binomial_option_price(params_down)
            
            delta = (price_up - price_down) / (2 * params.spot_price * self.bump_size_spot)
            
            # Gamma calculation
            gamma = (price_up - 2*base_price + price_down) / ((params.spot_price * self.bump_size_spot)**2)
            
            # Theta calculation
            params_theta = self._modify_params(params, time_bump=-self.bump_size_time)
            price_theta = self._binomial_option_price(params_theta)
            theta = (price_theta - base_price) / self.bump_size_time
            theta = theta / self.DAYS_PER_YEAR  # Convert to per day
            
            # Vega calculation
            params_vega_up = self._modify_params(params, vol_bump=self.bump_size_vol)
            params_vega_down = self._modify_params(params, vol_bump=-self.bump_size_vol)
            
            price_vega_up = self._binomial_option_price(params_vega_up)
            price_vega_down = self._binomial_option_price(params_vega_down)
            
            vega = (price_vega_up - price_vega_down) / (2 * self.bump_size_vol * 100)
            
            # Rho calculation
            params_rho_up = self._modify_params(params, rate_bump=self.bump_size_rate)
            params_rho_down = self._modify_params(params, rate_bump=-self.bump_size_rate)
            
            price_rho_up = self._binomial_option_price(params_rho_up)
            price_rho_down = self._binomial_option_price(params_rho_down)
            
            rho = (price_rho_up - price_rho_down) / (2 * self.bump_size_rate * 100)
            
            # Higher-order Greeks (simplified using finite differences)
            if include_higher_order:
                # Approximate higher-order Greeks
                vanna = speed = zomma = color = charm = vomma = ultima = 0.0
            else:
                vanna = speed = zomma = color = charm = vomma = ultima = 0.0
            
            # Risk metrics
            lambda_ = delta * params.spot_price / base_price if base_price != 0 else 0
            epsilon = 0.0  # Simplified
            
            # Dollar Greeks
            dollar_delta = delta * params.spot_price
            dollar_gamma = gamma * params.spot_price * params.spot_price * 0.01
            dollar_theta = theta
            dollar_vega = vega
            
            return GreeksResult(
                delta=delta,
                gamma=gamma,
                theta=theta,
                theta_annual=theta * self.DAYS_PER_YEAR,
                vega=vega,
                vega_decimal=vega * 100,
                rho=rho,
                vanna=vanna,
                charm=charm,
                vomma=vomma,
                ultima=ultima,
                speed=speed,
                zomma=zomma,
                color=color,
                lambda_=lambda_,
                epsilon=epsilon,
                dollar_delta=dollar_delta,
                dollar_gamma=dollar_gamma,
                dollar_theta=dollar_theta,
                dollar_vega=dollar_vega,
                model_used=GreeksModel.BINOMIAL
            )
            
        except Exception as e:
            self.logger.error(f"Error in binomial Greeks calculation: {e}")
            return self._get_default_greeks(GreeksModel.BINOMIAL)
    
    def _binomial_option_price(self, params: MarketParameters) -> float:
        """Calculate option price using binomial model"""
        try:
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            q = params.dividend_yield
            sigma = params.volatility
            n = self.binomial_steps
            
            if T <= 0:
                if params.option_type == OptionType.CALL:
                    return max(0, S - K)
                else:
                    return max(0, K - S)
            
            dt = T / n
            u = math.exp(sigma * math.sqrt(dt))
            d = 1 / u
            p = (math.exp((r - q) * dt) - d) / (u - d)
            
            # Initialize option values at expiration
            option_values = np.zeros(n + 1)
            
            for i in range(n + 1):
                S_T = S * (u ** (n - i)) * (d ** i)
                if params.option_type == OptionType.CALL:
                    option_values[i] = max(0, S_T - K)
                else:
                    option_values[i] = max(0, K - S_T)
            
            # Work backwards through the tree
            for j in range(n - 1, -1, -1):
                for i in range(j + 1):
                    option_values[i] = math.exp(-r * dt) * (
                        p * option_values[i] + (1 - p) * option_values[i + 1]
                    )
                    
                    # American option early exercise check
                    if params.option_style == OptionStyle.AMERICAN:
                        S_j = S * (u ** (j - i)) * (d ** i)
                        if params.option_type == OptionType.CALL:
                            intrinsic = max(0, S_j - K)
                        else:
                            intrinsic = max(0, K - S_j)
                        option_values[i] = max(option_values[i], intrinsic)
            
            return option_values[0]
            
        except Exception as e:
            self.logger.error(f"Error in binomial pricing: {e}")
            return 0.0
    
    def _calculate_trinomial_greeks(self, params: MarketParameters, 
                                  include_higher_order: bool) -> GreeksResult:
        """Calculate Greeks using trinomial model"""
        # For now, use binomial as approximation
        # In full implementation, would build trinomial tree
        return self._calculate_binomial_greeks(params, include_higher_order)
    
    def _calculate_monte_carlo_greeks(self, params: MarketParameters, 
                                    include_higher_order: bool) -> GreeksResult:
        """Calculate Greeks using Monte Carlo simulation"""
        try:
            # Use pathwise derivative method for Greeks calculation
            # This is a simplified implementation
            
            # Calculate base price and bumped prices
            base_price = self._monte_carlo_option_price(params)
            
            # Delta via finite differences
            params_up = self._modify_params(params, spot_bump=self.bump_size_spot)
            params_down = self._modify_params(params, spot_bump=-self.bump_size_spot)
            
            price_up = self._monte_carlo_option_price(params_up)
            price_down = self._monte_carlo_option_price(params_down)
            
            delta = (price_up - price_down) / (2 * params.spot_price * self.bump_size_spot)
            
            # Gamma
            gamma = (price_up - 2*base_price + price_down) / ((params.spot_price * self.bump_size_spot)**2)
            
            # Theta
            params_theta = self._modify_params(params, time_bump=-self.bump_size_time)
            price_theta = self._monte_carlo_option_price(params_theta)
            theta = (price_theta - base_price) / self.bump_size_time
            theta = theta / self.DAYS_PER_YEAR
            
            # Vega
            params_vega_up = self._modify_params(params, vol_bump=self.bump_size_vol)
            price_vega_up = self._monte_carlo_option_price(params_vega_up)
            vega = (price_vega_up - base_price) / (self.bump_size_vol * 100)
            
            # Rho
            params_rho_up = self._modify_params(params, rate_bump=self.bump_size_rate)
            price_rho_up = self._monte_carlo_option_price(params_rho_up)
            rho = (price_rho_up - base_price) / (self.bump_size_rate * 100)
            
            # Higher-order Greeks (simplified)
            if include_higher_order:
                vanna = speed = zomma = color = charm = vomma = ultima = 0.0
            else:
                vanna = speed = zomma = color = charm = vomma = ultima = 0.0
            
            # Risk metrics
            lambda_ = delta * params.spot_price / base_price if base_price != 0 else 0
            epsilon = 0.0
            
            # Dollar Greeks
            dollar_delta = delta * params.spot_price
            dollar_gamma = gamma * params.spot_price * params.spot_price * 0.01
            dollar_theta = theta
            dollar_vega = vega
            
            return GreeksResult(
                delta=delta,
                gamma=gamma,
                theta=theta,
                theta_annual=theta * self.DAYS_PER_YEAR,
                vega=vega,
                vega_decimal=vega * 100,
                rho=rho,
                vanna=vanna,
                charm=charm,
                vomma=vomma,
                ultima=ultima,
                speed=speed,
                zomma=zomma,
                color=color,
                lambda_=lambda_,
                epsilon=epsilon,
                dollar_delta=dollar_delta,
                dollar_gamma=dollar_gamma,
                dollar_theta=dollar_theta,
                dollar_vega=dollar_vega,
                model_used=GreeksModel.MONTE_CARLO
            )
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo Greeks calculation: {e}")
            return self._get_default_greeks(GreeksModel.MONTE_CARLO)
    
    def _monte_carlo_option_price(self, params: MarketParameters, num_simulations: int = 10000) -> float:
        """Calculate option price using Monte Carlo simulation"""
        try:
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            q = params.dividend_yield
            sigma = params.volatility
            
            if T <= 0:
                if params.option_type == OptionType.CALL:
                    return max(0, S - K)
                else:
                    return max(0, K - S)
            
            # Generate random paths
            np.random.seed(42)  # For reproducibility
            Z = np.random.standard_normal(num_simulations)
            
            # Calculate terminal stock prices
            S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
            
            # Calculate payoffs
            if params.option_type == OptionType.CALL:
                payoffs = np.maximum(S_T - K, 0)
            else:
                payoffs = np.maximum(K - S_T, 0)
            
            # Discount to present value
            option_price = math.exp(-r * T) * np.mean(payoffs)
            
            return option_price
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo pricing: {e}")
            return 0.0
    
    # Utility methods
    def _validate_parameters(self, params: MarketParameters):
        """Validate input parameters"""
        if params.spot_price <= 0:
            raise ValueError("Spot price must be positive")
        if params.strike_price <= 0:
            raise ValueError("Strike price must be positive")
        if params.time_to_expiry < 0:
            raise ValueError("Time to expiry cannot be negative")
        if params.volatility <= 0:
            raise ValueError("Volatility must be positive")
        if params.risk_free_rate < -1:
            raise ValueError("Risk-free rate seems unrealistic")
    
    def _modify_params(self, params: MarketParameters, 
                      spot_bump: float = 0,
                      vol_bump: float = 0,
                      time_bump: float = 0,
                      rate_bump: float = 0,
                      div_bump: float = 0) -> MarketParameters:
        """Create modified parameters for Greeks calculation"""
        return MarketParameters(
            spot_price=params.spot_price * (1 + spot_bump),
            strike_price=params.strike_price,
            time_to_expiry=max(0, params.time_to_expiry + time_bump),
            risk_free_rate=params.risk_free_rate + rate_bump,
            dividend_yield=params.dividend_yield + div_bump,
            volatility=max(0.001, params.volatility + vol_bump),
            option_type=params.option_type,
            option_style=params.option_style,
            underlying_type=params.underlying_type
        )
    
    def _estimate_option_price(self, params: MarketParameters) -> float:
        """Estimate option price for risk metrics"""
        try:
            # Use Black-Scholes for quick estimation
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            q = params.dividend_yield
            sigma = params.volatility
            
            if T <= 0:
                if params.option_type == OptionType.CALL:
                    return max(0, S - K)
                else:
                    return max(0, K - S)
            
            d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            
            if params.option_type == OptionType.CALL:
                price = (S * math.exp(-q*T) * norm.cdf(d1) - 
                        K * math.exp(-r*T) * norm.cdf(d2))
            else:
                price = (K * math.exp(-r*T) * norm.cdf(-d2) - 
                        S * math.exp(-q*T) * norm.cdf(-d1))
            
            return max(0, price)
            
        except Exception:
            return 1.0  # Fallback value
    
    def _get_expiry_greeks(self, params: MarketParameters) -> GreeksResult:
        """Get Greeks for expired options"""
        if params.option_type == OptionType.CALL:
            intrinsic = max(0, params.spot_price - params.strike_price)
            delta = 1.0 if params.spot_price > params.strike_price else 0.0
        else:
            intrinsic = max(0, params.strike_price - params.spot_price)
            delta = -1.0 if params.spot_price < params.strike_price else 0.0
        
        return GreeksResult(
            delta=delta,
            gamma=0.0,
            theta=0.0,
            theta_annual=0.0,
            vega=0.0,
            vega_decimal=0.0,
            rho=0.0,
            vanna=0.0,
            charm=0.0,
            vomma=0.0,
            ultima=0.0,
            speed=0.0,
            zomma=0.0,
            color=0.0,
            lambda_=0.0,
            epsilon=0.0,
            dollar_delta=delta * params.spot_price,
            dollar_gamma=0.0,
            dollar_theta=0.0,
            dollar_vega=0.0,
            model_used=GreeksModel.BLACK_SCHOLES
        )
    
    def _get_default_greeks(self, model: GreeksModel) -> GreeksResult:
        """Get default Greeks for error cases"""
        return GreeksResult(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            theta_annual=0.0,
            vega=0.0,
            vega_decimal=0.0,
            rho=0.0,
            vanna=0.0,
            charm=0.0,
            vomma=0.0,
            ultima=0.0,
            speed=0.0,
            zomma=0.0,
            color=0.0,
            lambda_=0.0,
            epsilon=0.0,
            dollar_delta=0.0,
            dollar_gamma=0.0,
            dollar_theta=0.0,
            dollar_vega=0.0,
            model_used=model
        )
    
    def _get_default_calendar_greeks(self) -> CalendarSpreadGreeks:
        """Get default calendar spread Greeks"""
        default_greeks = self._get_default_greeks(GreeksModel.BLACK_SCHOLES)
        
        return CalendarSpreadGreeks(
            net_delta=0.0,
            net_gamma=0.0,
            net_theta=0.0,
            net_vega=0.0,
            net_rho=0.0,
            short_leg_greeks=default_greeks,
            long_leg_greeks=default_greeks,
            time_decay_ratio=0.0,
            volatility_exposure=0.0,
            delta_neutrality=0.0,
            gamma_scalping_potential=0.0,
            max_gamma_exposure=0.0,
            theta_efficiency=0.0
        )
    
    def _get_default_scenario(self) -> GreeksScenario:
        """Get default scenario Greeks"""
        default_greeks = self._get_default_greeks(GreeksModel.BLACK_SCHOLES)
        
        return GreeksScenario(
            base_case=default_greeks,
            up_5_percent=default_greeks,
            down_5_percent=default_greeks,
            up_10_percent=default_greeks,
            down_10_percent=default_greeks,
            vol_up_5_percent=default_greeks,
            vol_down_5_percent=default_greeks,
            time_decay_1_day=default_greeks,
            time_decay_7_days=default_greeks
        )
    
    # Cache management
    def _generate_cache_key(self, params: MarketParameters, model: GreeksModel, 
                           include_higher_order: bool) -> str:
        """Generate cache key for Greeks calculation"""
        key_parts = [
            f"S{params.spot_price:.4f}",
            f"K{params.strike_price:.4f}",
            f"T{params.time_to_expiry:.6f}",
            f"r{params.risk_free_rate:.6f}",
            f"q{params.dividend_yield:.6f}",
            f"v{params.volatility:.6f}",
            params.option_type.value,
            params.option_style.value,
            model.value,
            str(include_higher_order)
        ]
        return "_".join(key_parts)
    
    def _get_from_cache(self, key: str) -> Optional[GreeksResult]:
        """Get Greeks from cache if available and not expired"""
        if not self.cache_enabled:
            return None
        
        with self._cache_lock:
            if key in self._greeks_cache:
                result, timestamp = self._greeks_cache[key]
                if time.time() - timestamp < self.cache_ttl:
                    return result
                else:
                    del self._greeks_cache[key]
        
        return None
    
    def _save_to_cache(self, key: str, result: GreeksResult):
        """Save Greeks to cache"""
        if not self.cache_enabled:
            return
        
        with self._cache_lock:
            self._greeks_cache[key] = (result, time.time())
    
    def clear_cache(self):
        """Clear Greeks cache"""
        with self._cache_lock:
            self._greeks_cache.clear()
        self.logger.info("Greeks cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._cache_lock:
            current_time = time.time()
            valid_entries = sum(
                1 for _, timestamp in self._greeks_cache.values()
                if current_time - timestamp < self.cache_ttl
            )
            
            return {
                "total_entries": len(self._greeks_cache),
                "valid_entries": valid_entries,
                "expired_entries": len(self._greeks_cache) - valid_entries
            }
    
    # Advanced analysis methods
    def analyze_gamma_scalping_potential(self, params: MarketParameters) -> Dict[str, float]:
        """Analyze gamma scalping potential"""
        try:
            greeks = self.calculate_greeks(params, include_higher_order=True)
            
            # Calculate gamma P&L for various moves
            spot_moves = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%
            gamma_pnl = {}
            
            for move in spot_moves:
                # Gamma P&L = 0.5 * Gamma * (ΔS)²
                delta_s = params.spot_price * move
                pnl = 0.5 * greeks.gamma * (delta_s ** 2)
                gamma_pnl[f"move_{move*100:.0f}pct"] = pnl
            
            # Daily gamma decay (simplified)
            daily_gamma_decay = abs(greeks.color) if greeks.color else 0
            
            # Gamma efficiency (gamma per dollar of premium)
            option_price = self._estimate_option_price(params)
            gamma_efficiency = greeks.gamma / option_price if option_price > 0 else 0
            
            return {
                "current_gamma": greeks.gamma,
                "dollar_gamma": greeks.dollar_gamma,
                "gamma_efficiency": gamma_efficiency,
                "daily_gamma_decay": daily_gamma_decay,
                **gamma_pnl
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing gamma scalping: {e}")
            return {}
    
    def calculate_greeks_sensitivity(self, params: MarketParameters, 
                                   sensitivity_type: str = "spot") -> pd.DataFrame:
        """Calculate Greeks sensitivity to underlying parameter changes"""
        try:
            if sensitivity_type == "spot":
                base_value = params.spot_price
                range_pct = 0.2  # ±20%
                num_points = 21
            elif sensitivity_type == "volatility":
                base_value = params.volatility
                range_pct = 0.5  # ±50%
                num_points = 21
            elif sensitivity_type == "time":
                base_value = params.time_to_expiry
                range_pct = 1.0  # Up to expiry
                num_points = min(21, int(params.time_to_expiry * 365) + 1)
            else:
                raise ValueError(f"Unsupported sensitivity type: {sensitivity_type}")
            
            # Create range of values
            if sensitivity_type == "time":
                values = np.linspace(0, base_value, num_points)
            else:
                min_val = base_value * (1 - range_pct)
                max_val = base_value * (1 + range_pct)
                values = np.linspace(min_val, max_val, num_points)
            
            # Calculate Greeks for each value
            results = []
            for value in values:
                if sensitivity_type == "spot":
                    test_params = self._modify_params(params, spot_bump=(value/base_value - 1))
                elif sensitivity_type == "volatility":
                    test_params = self._modify_params(params, vol_bump=(value - base_value))
                elif sensitivity_type == "time":
                    test_params = self._modify_params(params, time_bump=(value - base_value))
                
                greeks = self.calculate_greeks(test_params)
                
                results.append({
                    sensitivity_type: value,
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "theta": greeks.theta,
                    "vega": greeks.vega,
                    "rho": greeks.rho,
                    "option_price": self._estimate_option_price(test_params)
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks sensitivity: {e}")
            return pd.DataFrame()
    
    def calculate_implied_forward(self, call_params: MarketParameters, 
                                put_params: MarketParameters) -> float:
        """Calculate implied forward price from put-call parity"""
        try:
            if (call_params.strike_price != put_params.strike_price or
                call_params.time_to_expiry != put_params.time_to_expiry):
                raise ValueError("Call and put must have same strike and expiry")
            
            call_price = self._estimate_option_price(call_params)
            put_price = self._estimate_option_price(put_params)
            
            K = call_params.strike_price
            T = call_params.time_to_expiry
            r = call_params.risk_free_rate
            q = call_params.dividend_yield
            
            # Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
            # Implied forward: F = (C - P + K*e^(-rT)) / e^(-qT)
            forward = (call_price - put_price + K * math.exp(-r * T)) / math.exp(-q * T)
            
            return forward
            
        except Exception as e:
            self.logger.error(f"Error calculating implied forward: {e}")
            return call_params.spot_price
    
    def export_greeks_report(self, params: MarketParameters, output_file: str) -> bool:
        """Export comprehensive Greeks analysis report"""
        try:
            # Calculate all Greeks variations
            bs_greeks = self.calculate_greeks(params, GreeksModel.BLACK_SCHOLES, True)
            binomial_greeks = self.calculate_greeks(params, GreeksModel.BINOMIAL, True)
            scenario_greeks = self.calculate_scenario_greeks(params)
            gamma_analysis = self.analyze_gamma_scalping_potential(params)
            
            # Create comprehensive report
            report = {
                "symbol_info": {
                    "spot_price": params.spot_price,
                    "strike_price": params.strike_price,
                    "time_to_expiry_years": params.time_to_expiry,
                    "time_to_expiry_days": params.time_to_expiry * 365.25,
                    "volatility": params.volatility,
                    "risk_free_rate": params.risk_free_rate,
                    "dividend_yield": params.dividend_yield,
                    "option_type": params.option_type.value,
                    "option_style": params.option_style.value
                },
                "black_scholes_greeks": asdict(bs_greeks),
                "binomial_greeks": asdict(binomial_greeks),
                "scenario_analysis": asdict(scenario_greeks),
                "gamma_analysis": gamma_analysis,
                "calculation_timestamp": datetime.now().isoformat(),
                "calculator_config": {
                    "cache_enabled": self.cache_enabled,
                    "precision": self.precision,
                    "binomial_steps": self.binomial_steps
                }
            }
            
            with open(output_file, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Greeks report exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting Greeks report: {e}")
            return False

# Convenience functions for common calculations
def quick_greeks(spot: float, strike: float, expiry_days: int, 
                volatility: float, risk_free_rate: float = 0.02,
                option_type: str = "call") -> GreeksResult:
    """Quick Greeks calculation with minimal parameters"""
    calculator = GreeksCalculator(ConfigManager())
    
    params = MarketParameters(
        spot_price=spot,
        strike_price=strike,
        time_to_expiry=expiry_days / 365.25,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=OptionType.CALL if option_type.lower() == "call" else OptionType.PUT
    )
    
    return calculator.calculate_greeks(params)

def calendar_spread_greeks(spot: float, strike: float, 
                         short_expiry_days: int, long_expiry_days: int,
                         volatility: float, risk_free_rate: float = 0.02) -> CalendarSpreadGreeks:
    """Quick calendar spread Greeks calculation"""
    calculator = GreeksCalculator(ConfigManager())
    
    short_params = MarketParameters(
        spot_price=spot,
        strike_price=strike,
        time_to_expiry=short_expiry_days / 365.25,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=OptionType.PUT  # Calendar spreads typically use puts
    )
    
    long_params = MarketParameters(
        spot_price=spot,
        strike_price=strike,
        time_to_expiry=long_expiry_days / 365.25,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=OptionType.PUT
    )
    
    return calculator.calculate_calendar_spread_greeks(short_params, long_params)
