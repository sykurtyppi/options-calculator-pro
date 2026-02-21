"""
Volatility Service - Professional Volatility Analysis Engine
===========================================================

Handles all volatility-related calculations including:
- Yang-Zhang volatility estimation
- IV term structure analysis and modeling
- IV rank and percentile calculations
- Volatility forecasting and regime detection
- Volatility surface construction
- Risk-neutral density estimation

Part of Professional Options Calculator v9.1
Optimized for Apple Silicon and PySide6
"""

import logging
import threading
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, griddata, RBFInterpolator
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm, gaussian_kde
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import your existing utilities
from utils.config_manager import ConfigManager
from utils.logger import get_logger
from services.market_data import MarketDataService

logger = get_logger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    VERY_LOW = "very_low"          # < 12 VIX
    LOW = "low"                    # 12-16 VIX
    NORMAL = "normal"              # 16-24 VIX
    ELEVATED = "elevated"          # 24-32 VIX
    HIGH = "high"                  # 32-45 VIX
    EXTREME = "extreme"            # > 45 VIX

class VolatilityModel(Enum):
    """Volatility estimation models"""
    CLOSE_TO_CLOSE = "close_to_close"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"
    ROGERS_SATCHELL = "rogers_satchell"
    YANG_ZHANG = "yang_zhang"

@dataclass
class VolatilityMetrics:
    """Comprehensive volatility metrics"""
    realized_vol_30d: float
    realized_vol_60d: float
    realized_vol_90d: float
    implied_vol_30d: float
    iv_rv_ratio: float
    vol_regime: VolatilityRegime
    vol_percentile: float
    vol_rank: float
    vol_trend: float
    vol_mean_reversion: float
    vol_persistence: float
    vol_clustering: float
    
@dataclass
class IVTermStructure:
    """IV term structure data and analysis"""
    symbol: str
    calculation_date: datetime
    expirations: List[str]
    days_to_expiration: List[int]
    atm_ivs: List[float]
    call_ivs: List[float]
    put_ivs: List[float]
    interpolation_function: Optional[Callable]
    slope_0_30: float
    slope_30_60: float
    slope_60_90: float
    curvature: float
    term_structure_score: float
    
@dataclass
class VolatilitySurface:
    """Volatility surface data"""
    symbol: str
    calculation_date: datetime
    strikes: np.ndarray
    expirations: np.ndarray
    volatilities: np.ndarray
    moneyness: np.ndarray
    time_to_expiry: np.ndarray
    surface_function: Optional[Callable]
    skew_30d: float
    skew_60d: float
    smile_coefficient: float
    wing_risk: float

@dataclass
class VolatilityForecast:
    """Volatility forecast results"""
    symbol: str
    forecast_horizon: int
    current_vol: float
    forecasted_vol: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    forecast_model: str
    regime_probability: Dict[VolatilityRegime, float]
    forecast_timestamp: datetime

class VolatilityService:
    """
    Professional Volatility Analysis Service
    
    Provides comprehensive volatility analysis including:
    - Multiple volatility estimators (Yang-Zhang, Garman-Klass, etc.)
    - IV term structure construction and analysis
    - Volatility surface modeling
    - Volatility forecasting and regime detection
    - Risk metrics and scenario analysis
    """
    
    def __init__(self, config_manager: ConfigManager, market_data_service: MarketDataService):
        self.config = config_manager
        self.market_data = market_data_service
        self.logger = logger
        
        # Configuration
        self.default_window = 30
        self.trading_periods = 252
        self.min_data_points = 20
        self.cache_ttl = 1800  # 30 minutes
        
        # Volatility bounds
        self.min_volatility = 0.05
        self.max_volatility = 3.00
        self.default_volatility = 0.25
        
        # Cache
        self._vol_cache = {}
        self._iv_cache = {}
        self._surface_cache = {}
        
        # Thread safety
        self._cache_lock = threading.Lock()
        
        # Volatility regime thresholds (VIX-based)
        self.regime_thresholds = {
            VolatilityRegime.VERY_LOW: (0, 12),
            VolatilityRegime.LOW: (12, 16),
            VolatilityRegime.NORMAL: (16, 24),
            VolatilityRegime.ELEVATED: (24, 32),
            VolatilityRegime.HIGH: (32, 45),
            VolatilityRegime.EXTREME: (45, 100)
        }
        
        self.logger.info("VolatilityService initialized")

    @staticmethod
    def _period_from_days(days: int) -> str:
        """Convert lookback days to a valid yfinance period string."""
        if days <= 5:
            return "5d"
        if days <= 30:
            return "1mo"
        if days <= 90:
            return "3mo"
        if days <= 180:
            return "6mo"
        if days <= 365:
            return "1y"
        if days <= 730:
            return "2y"
        if days <= 1825:
            return "5y"
        if days <= 3650:
            return "10y"
        return "max"
    
    def calculate_realized_volatility(self, symbol: str, window: int = 30, 
                                    model: VolatilityModel = VolatilityModel.YANG_ZHANG,
                                    return_last_only: bool = True) -> Union[float, pd.Series]:
        """
        Calculate realized volatility using specified model
        
        Args:
            symbol: Stock ticker symbol
            window: Rolling window size
            model: Volatility estimation model
            return_last_only: Return only the last value or full series
            
        Returns:
            Volatility estimate(s)
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_rv_{window}_{model.value}"
            cached_data = self._get_from_cache(cache_key, cache_type='vol')
            if cached_data is not None:
                return cached_data if not return_last_only else cached_data.iloc[-1]
            
            # Get historical price data with buffer to stabilize rolling windows
            period = self._period_from_days(window + 60)
            price_data = self.market_data.get_historical_data(symbol, period=period, interval="1d")
            
            if price_data.empty or len(price_data) < self.min_data_points:
                self.logger.warning(f"Insufficient data for {symbol} volatility calculation")
                if return_last_only:
                    return self.default_volatility
                else:
                    # Return Series with default values to maintain type consistency
                    return pd.Series([self.default_volatility], name='volatility')
            
            # Calculate volatility based on model
            if model == VolatilityModel.YANG_ZHANG:
                vol_series = self._yang_zhang_volatility(price_data, window)
            elif model == VolatilityModel.GARMAN_KLASS:
                vol_series = self._garman_klass_volatility(price_data, window)
            elif model == VolatilityModel.PARKINSON:
                vol_series = self._parkinson_volatility(price_data, window)
            elif model == VolatilityModel.ROGERS_SATCHELL:
                vol_series = self._rogers_satchell_volatility(price_data, window)
            else:  # CLOSE_TO_CLOSE
                vol_series = self._close_to_close_volatility(price_data, window)
            
            # Cache result
            self._save_to_cache(cache_key, vol_series, cache_type='vol')
            
            return vol_series if not return_last_only else vol_series.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating realized volatility for {symbol}: {e}")
            if return_last_only:
                return self.default_volatility
            else:
                # Return Series with default values to maintain type consistency
                return pd.Series([self.default_volatility], name='volatility')
    
    def calculate_volatility_metrics(self, symbol: str) -> VolatilityMetrics:
        """
        Calculate comprehensive volatility metrics
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            VolatilityMetrics object with all metrics
        """
        try:
            # Calculate multiple timeframe realized volatilities
            rv_30d = self.calculate_realized_volatility(symbol, 30)
            rv_60d = self.calculate_realized_volatility(symbol, 60)
            rv_90d = self.calculate_realized_volatility(symbol, 90)
            
            # Get implied volatility (simplified - in practice would come from options)
            iv_30d = self._estimate_implied_volatility(symbol)
            
            # Calculate IV/RV ratio
            iv_rv_ratio = iv_30d / rv_30d if rv_30d > 0 else 1.0
            
            # Determine volatility regime
            vix = self.market_data.get_vix()
            vol_regime = self._classify_volatility_regime(vix)
            
            # Calculate volatility rank and percentile
            vol_rank, vol_percentile = self._calculate_vol_rank_percentile(symbol, rv_30d)
            
            # Calculate volatility characteristics
            vol_trend = self._calculate_volatility_trend(symbol)
            vol_mean_reversion = self._calculate_mean_reversion_speed(symbol)
            vol_persistence = self._calculate_volatility_persistence(symbol)
            vol_clustering = self._calculate_volatility_clustering(symbol)
            
            return VolatilityMetrics(
                realized_vol_30d=rv_30d,
                realized_vol_60d=rv_60d,
                realized_vol_90d=rv_90d,
                implied_vol_30d=iv_30d,
                iv_rv_ratio=iv_rv_ratio,
                vol_regime=vol_regime,
                vol_percentile=vol_percentile,
                vol_rank=vol_rank,
                vol_trend=vol_trend,
                vol_mean_reversion=vol_mean_reversion,
                vol_persistence=vol_persistence,
                vol_clustering=vol_clustering
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics for {symbol}: {e}")
            return self._get_default_vol_metrics()
    
    def build_iv_term_structure(self, symbol: str, option_chains: Dict[str, Any]) -> IVTermStructure:
        """
        Build implied volatility term structure from option chains
        
        Args:
            symbol: Stock ticker symbol
            option_chains: Dictionary of option chain data by expiration
            
        Returns:
            IVTermStructure object
        """
        try:
            expirations = []
            days_to_exp = []
            atm_ivs = []
            call_ivs = []
            put_ivs = []
            
            today = datetime.now().date()
            
            # Process each expiration
            for exp_date, chain_data in option_chains.items():
                try:
                    # Calculate days to expiration
                    exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
                    days = (exp_date_obj - today).days
                    
                    if days <= 0:
                        continue
                    
                    # Extract ATM IVs
                    atm_call_iv = chain_data.get('atm_call_iv', 0)
                    atm_put_iv = chain_data.get('atm_put_iv', 0)
                    
                    if atm_call_iv <= 0 and atm_put_iv <= 0:
                        continue
                    
                    # Calculate average ATM IV
                    if atm_call_iv > 0 and atm_put_iv > 0:
                        atm_iv = (atm_call_iv + atm_put_iv) / 2
                    else:
                        atm_iv = atm_call_iv or atm_put_iv
                    
                    expirations.append(exp_date)
                    days_to_exp.append(days)
                    atm_ivs.append(atm_iv)
                    call_ivs.append(atm_call_iv)
                    put_ivs.append(atm_put_iv)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing expiration {exp_date}: {e}")
                    continue
            
            if len(days_to_exp) < 2:
                return self._get_default_term_structure(symbol)
            
            # Create interpolation function
            interpolation_function = None
            if len(days_to_exp) >= 2:
                try:
                    # Sort by days to expiration
                    sorted_data = sorted(zip(days_to_exp, atm_ivs))
                    sorted_days, sorted_ivs = zip(*sorted_data)
                    
                    interpolation_function = interp1d(
                        sorted_days, sorted_ivs, 
                        kind='linear', 
                        fill_value='extrapolate',
                        bounds_error=False
                    )
                except Exception as e:
                    self.logger.warning(f"Could not create interpolation function: {e}")
            
            # Calculate term structure slopes
            slope_0_30 = self._calculate_term_structure_slope(days_to_exp, atm_ivs, 0, 30)
            slope_30_60 = self._calculate_term_structure_slope(days_to_exp, atm_ivs, 30, 60)
            slope_60_90 = self._calculate_term_structure_slope(days_to_exp, atm_ivs, 60, 90)
            
            # Calculate curvature
            curvature = self._calculate_term_structure_curvature(days_to_exp, atm_ivs)
            
            # Calculate term structure score
            ts_score = self._score_term_structure(slope_0_30, slope_30_60, curvature)
            
            return IVTermStructure(
                symbol=symbol,
                calculation_date=datetime.now(),
                expirations=expirations,
                days_to_expiration=days_to_exp,
                atm_ivs=atm_ivs,
                call_ivs=call_ivs,
                put_ivs=put_ivs,
                interpolation_function=interpolation_function,
                slope_0_30=slope_0_30,
                slope_30_60=slope_30_60,
                slope_60_90=slope_60_90,
                curvature=curvature,
                term_structure_score=ts_score
            )
            
        except Exception as e:
            self.logger.error(f"Error building IV term structure for {symbol}: {e}")
            return self._get_default_term_structure(symbol)
    
    def build_volatility_surface(self, symbol: str, full_option_data: Dict[str, Any]) -> VolatilitySurface:
        """
        Build volatility surface from complete option data
        
        Args:
            symbol: Stock ticker symbol
            full_option_data: Complete option data with strikes and expirations
            
        Returns:
            VolatilitySurface object
        """
        try:
            strikes = []
            expirations = []
            volatilities = []
            underlying_price = full_option_data.get('underlying_price', 100)
            
            today = datetime.now().date()
            
            # Collect all data points
            for exp_date, exp_data in full_option_data.get('expirations', {}).items():
                try:
                    exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
                    days_to_exp = (exp_date_obj - today).days
                    
                    if days_to_exp <= 0:
                        continue
                    
                    time_to_expiry = days_to_exp / 365.25
                    
                    # Process strikes
                    for strike_data in exp_data.get('strikes', []):
                        strike = strike_data.get('strike')
                        iv = strike_data.get('iv')
                        
                        if strike and iv and iv > 0:
                            strikes.append(strike)
                            expirations.append(time_to_expiry)
                            volatilities.append(iv)
                            
                except Exception as e:
                    self.logger.warning(f"Error processing surface data for {exp_date}: {e}")
                    continue
            
            if len(strikes) < 10:  # Need minimum data points
                return self._get_default_surface(symbol, underlying_price)
            
            # Convert to arrays
            strikes = np.array(strikes)
            expirations = np.array(expirations)
            volatilities = np.array(volatilities)
            
            # Calculate moneyness
            moneyness = strikes / underlying_price
            
            # Create surface interpolation function
            surface_function = None
            try:
                # Use RBF interpolation for smooth surface
                points = np.column_stack([moneyness, expirations])
                surface_function = RBFInterpolator(
                    points, volatilities, 
                    kernel='thin_plate_spline',
                    smoothing=0.01
                )
            except Exception as e:
                self.logger.warning(f"Could not create surface function: {e}")
            
            # Calculate surface characteristics
            skew_30d = self._calculate_volatility_skew(strikes, volatilities, expirations, 30/365.25)
            skew_60d = self._calculate_volatility_skew(strikes, volatilities, expirations, 60/365.25)
            smile_coefficient = self._calculate_smile_coefficient(moneyness, volatilities)
            wing_risk = self._calculate_wing_risk(moneyness, volatilities)
            
            return VolatilitySurface(
                symbol=symbol,
                calculation_date=datetime.now(),
                strikes=strikes,
                expirations=expirations,
                volatilities=volatilities,
                moneyness=moneyness,
                time_to_expiry=expirations,
                surface_function=surface_function,
                skew_30d=skew_30d,
                skew_60d=skew_60d,
                smile_coefficient=smile_coefficient,
                wing_risk=wing_risk
            )
            
        except Exception as e:
            self.logger.error(f"Error building volatility surface for {symbol}: {e}")
            return self._get_default_surface(symbol, 100)
    
    def forecast_volatility(self, symbol: str, horizon_days: int = 30,
                           model: str = "garch") -> VolatilityForecast:
        """
        Forecast future volatility using specified model
        
        Args:
            symbol: Stock ticker symbol
            horizon_days: Forecast horizon in days
            model: Forecasting model ("garch", "exponential_smoothing", "mean_reversion")
            
        Returns:
            VolatilityForecast object
        """
        try:
            # Get current volatility
            current_vol = self.calculate_realized_volatility(symbol, 30)
            
            # Get historical volatility series for forecasting
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)
            
            if model == "garch":
                forecast_vol, lower_ci, upper_ci = self._garch_forecast(vol_series, horizon_days)
            elif model == "exponential_smoothing":
                forecast_vol, lower_ci, upper_ci = self._exponential_smoothing_forecast(vol_series, horizon_days)
            else:  # mean_reversion
                forecast_vol, lower_ci, upper_ci = self._mean_reversion_forecast(vol_series, horizon_days)
            
            # Calculate regime probabilities
            regime_probs = self._calculate_regime_probabilities(forecast_vol)
            
            return VolatilityForecast(
                symbol=symbol,
                forecast_horizon=horizon_days,
                current_vol=current_vol,
                forecasted_vol=forecast_vol,
                confidence_interval_lower=lower_ci,
                confidence_interval_upper=upper_ci,
                forecast_model=model,
                regime_probability=regime_probs,
                forecast_timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error forecasting volatility for {symbol}: {e}")
            return self._get_default_forecast(symbol, horizon_days)
    
    # Core volatility calculation methods
    def _yang_zhang_volatility(self, price_data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Yang-Zhang volatility estimator"""
        try:
            # Input validation
            required_columns = ["Open", "High", "Low", "Close"]
            if not all(col in price_data.columns for col in required_columns):
                raise ValueError("Missing required OHLC columns")
            
            if len(price_data) < window:
                raise ValueError(f"Insufficient data: need {window}, got {len(price_data)}")
            
            # Clean data
            data = price_data[required_columns].copy()
            
            # Replace zeros and NaNs
            for col in required_columns:
                data.loc[data[col] <= 0, col] = np.nan
                data[col] = data[col].ffill().bfill()
                
                # If still NaN, use median
                if data[col].isna().any():
                    median_val = data[col].median()
                    if pd.isna(median_val):
                        median_val = 1.0
                    data[col] = data[col].fillna(median_val)
            
            # Yang-Zhang calculation:
            # overnight returns: O_t / C_{t-1}
            # open-to-close returns: C_t / O_t
            # Rogers-Satchell intraday estimator
            log_ho = np.log(data["High"] / data["Open"])
            log_lo = np.log(data["Low"] / data["Open"])
            log_co = np.log(data["Close"] / data["Open"])
            log_oc = np.log(data["Open"] / data["Close"].shift(1))
            
            rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
            
            # Rolling sample variances (ddof=1) with drift removal.
            open_vol = log_oc.rolling(window=window, min_periods=window).var(ddof=1)
            close_vol = log_co.rolling(window=window, min_periods=window).var(ddof=1)
            window_rs = rs.rolling(window=window, min_periods=window).mean()
            
            # Yang-Zhang formula
            k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
            yz_variance = open_vol + k * close_vol + (1 - k) * window_rs
            yz_variance = yz_variance.clip(lower=0.0)
            result = np.sqrt(yz_variance) * np.sqrt(self.trading_periods)
            
            # Clean and bound results
            result = result.fillna(self.default_volatility)
            result = np.clip(result, self.min_volatility, self.max_volatility)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Yang-Zhang calculation: {e}")
            return pd.Series(self.default_volatility, index=price_data.index)
    
    def _garman_klass_volatility(self, price_data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        try:
            log_hl = np.log(price_data["High"] / price_data["Low"])
            log_co = np.log(price_data["Close"] / price_data["Open"])
            
            gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
            
            result = np.sqrt(gk.rolling(window=window).mean() * self.trading_periods)
            result = result.fillna(self.default_volatility)
            result = np.clip(result, self.min_volatility, self.max_volatility)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Garman-Klass calculation: {e}")
            return pd.Series(self.default_volatility, index=price_data.index)
    
    def _parkinson_volatility(self, price_data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Parkinson volatility estimator"""
        try:
            log_hl = np.log(price_data["High"] / price_data["Low"])
            
            result = np.sqrt((log_hl ** 2).rolling(window=window).mean() / (4 * np.log(2)) * self.trading_periods)
            result = result.fillna(self.default_volatility)
            result = np.clip(result, self.min_volatility, self.max_volatility)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Parkinson calculation: {e}")
            return pd.Series(self.default_volatility, index=price_data.index)
    
    def _rogers_satchell_volatility(self, price_data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Rogers-Satchell volatility estimator"""
        try:
            log_ho = np.log(price_data["High"] / price_data["Open"])
            log_hc = np.log(price_data["High"] / price_data["Close"])
            log_lo = np.log(price_data["Low"] / price_data["Open"])
            log_lc = np.log(price_data["Low"] / price_data["Close"])
            
            rs = log_ho * log_hc + log_lo * log_lc
            
            result = np.sqrt(rs.rolling(window=window).mean() * self.trading_periods)
            result = result.fillna(self.default_volatility)
            result = np.clip(result, self.min_volatility, self.max_volatility)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Rogers-Satchell calculation: {e}")
            return pd.Series(self.default_volatility, index=price_data.index)
    
    def _close_to_close_volatility(self, price_data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate simple close-to-close volatility"""
        try:
            returns = price_data["Close"].pct_change()
            
            result = returns.rolling(window=window).std() * np.sqrt(self.trading_periods)
            result = result.fillna(self.default_volatility)
            result = np.clip(result, self.min_volatility, self.max_volatility)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in close-to-close calculation: {e}")
            return pd.Series(self.default_volatility, index=price_data.index)
    
    # Helper methods for volatility analysis
    def _estimate_implied_volatility(self, symbol: str) -> float:
        """Estimate 30-day implied volatility (simplified)"""
        try:
            # In practice, this would come from options data
            # For now, use a simple relationship with realized vol and VIX
            rv = self.calculate_realized_volatility(symbol, 30)
            vix = self.market_data.get_vix()
            
            # Simple estimation: blend RV with VIX
            iv_estimate = 0.7 * rv + 0.3 * (vix / 100)
            
            return max(self.min_volatility, min(self.max_volatility, iv_estimate))
            
        except Exception as e:
            self.logger.error(f"Error estimating IV for {symbol}: {e}")
            return self.default_volatility
    
    def _classify_volatility_regime(self, vix: float) -> VolatilityRegime:
        """Classify current volatility regime based on VIX"""
        for regime, (low, high) in self.regime_thresholds.items():
            if low <= vix < high:
                return regime
        return VolatilityRegime.EXTREME
    
    def _calculate_vol_rank_percentile(self, symbol: str, current_vol: float) -> Tuple[float, float]:
        """Calculate volatility rank and percentile"""
        try:
            # Get longer history for ranking
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)

            # Handle case where insufficient data returns scalar instead of Series
            if isinstance(vol_series, (int, float)):
                return 0.5, 50.0

            if len(vol_series) < 30:
                return 0.5, 50.0
            
            # Take last 252 days (1 year)
            recent_vols = vol_series.tail(252)
            
            # Calculate rank
            vol_min = recent_vols.min()
            vol_max = recent_vols.max()
            vol_range = vol_max - vol_min
            
            if vol_range > 0:
                vol_rank = (current_vol - vol_min) / vol_range
                vol_rank = max(0, min(1, vol_rank))
            else:
                vol_rank = 0.5
            
            # Calculate percentile
            vol_percentile = (recent_vols <= current_vol).sum() / len(recent_vols) * 100
            vol_percentile = max(0, min(100, vol_percentile))
            
            return vol_rank, vol_percentile
            
        except Exception as e:
            self.logger.error(f"Error calculating vol rank/percentile: {e}")
            return 0.5, 50.0
    
    def _calculate_volatility_trend(self, symbol: str) -> float:
        """Calculate volatility trend (positive = increasing)"""
        try:
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)
            
            if len(vol_series) < 10:
                return 0.0
            
            # Simple linear trend over last 30 observations
            recent_vols = vol_series.tail(30)
            x = np.arange(len(recent_vols))
            y = recent_vols.values
            
            # Linear regression
            if len(x) >= 2:
                slope = np.polyfit(x, y, 1)[0]
                return float(slope)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating volatility trend: {e}")
            return 0.0
    
    def _calculate_mean_reversion_speed(self, symbol: str) -> float:
        """Calculate mean reversion speed coefficient"""
        try:
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)
            
            if len(vol_series) < 50:
                return 0.5
            
            # Calculate mean reversion using AR(1) model
            recent_vols = vol_series.tail(100)
            vol_mean = recent_vols.mean()
            
            # Deviations from mean
            deviations = recent_vols - vol_mean
            
            # AR(1) coefficient: vol[t] = alpha + beta * vol[t-1] + error
            # Mean reversion speed = 1 - beta
            lagged_devs = deviations.shift(1).dropna()
            current_devs = deviations[1:]
            
            if len(lagged_devs) >= 10:
                correlation = np.corrcoef(lagged_devs, current_devs)[0, 1]
                beta = correlation if not np.isnan(correlation) else 0.5
                mean_reversion_speed = 1 - abs(beta)
                return max(0, min(1, mean_reversion_speed))
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion speed: {e}")
            return 0.5
    
    def _calculate_volatility_persistence(self, symbol: str) -> float:
        """Calculate volatility persistence (autocorrelation)"""
        try:
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)
            
            if len(vol_series) < 30:
                return 0.5
            
            # Calculate first-order autocorrelation
            recent_vols = vol_series.tail(60)
            autocorr = recent_vols.autocorr(lag=1)
            
            if np.isnan(autocorr):
                return 0.5
            
            return max(0, min(1, autocorr))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility persistence: {e}")
            return 0.5
    
    def _calculate_volatility_clustering(self, symbol: str) -> float:
        """Calculate volatility clustering coefficient"""
        try:
            # Get returns data
            period = self._period_from_days(120)
            price_data = self.market_data.get_historical_data(symbol, period=period, interval="1d")
            
            if price_data.empty or len(price_data) < 50:
                return 0.5
            
            # Calculate returns
            returns = price_data['Close'].pct_change().dropna()
            
            # Calculate squared returns (proxy for volatility)
            squared_returns = returns ** 2
            
            # ARCH test: autocorrelation in squared returns
            if len(squared_returns) >= 20:
                arch_coeff = squared_returns.autocorr(lag=1)
                if not np.isnan(arch_coeff):
                    return max(0, min(1, arch_coeff))
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility clustering: {e}")
            return 0.5
    
    def _calculate_term_structure_slope(self, days: List[int], ivs: List[float], 
                                      start_day: int, end_day: int) -> float:
        """Calculate term structure slope between two points"""
        try:
            if len(days) < 2 or len(ivs) < 2:
                return 0.0
            
            if end_day <= start_day:
                return 0.0

            days_array = np.asarray(days, dtype=float)
            ivs_array = np.asarray(ivs, dtype=float)
            valid_mask = np.isfinite(days_array) & np.isfinite(ivs_array)
            if valid_mask.sum() < 2:
                return 0.0

            days_valid = days_array[valid_mask]
            ivs_valid = ivs_array[valid_mask]

            # Sort and average duplicate tenors.
            sort_idx = np.argsort(days_valid)
            days_sorted = days_valid[sort_idx]
            ivs_sorted = ivs_valid[sort_idx]
            unique_days, inverse = np.unique(days_sorted, return_inverse=True)
            if len(unique_days) < 2:
                return 0.0
            unique_ivs = np.array([
                float(np.mean(ivs_sorted[inverse == i])) for i in range(len(unique_days))
            ], dtype=float)

            def _interp_or_extrapolate(target_day: float) -> float:
                if target_day <= unique_days[0]:
                    x0, x1 = unique_days[0], unique_days[1]
                    y0, y1 = unique_ivs[0], unique_ivs[1]
                elif target_day >= unique_days[-1]:
                    x0, x1 = unique_days[-2], unique_days[-1]
                    y0, y1 = unique_ivs[-2], unique_ivs[-1]
                else:
                    return float(np.interp(target_day, unique_days, unique_ivs))
                if x1 == x0:
                    return float(y0)
                return float(y0 + (y1 - y0) * ((target_day - x0) / (x1 - x0)))

            start_iv = _interp_or_extrapolate(float(start_day))
            end_iv = _interp_or_extrapolate(float(end_day))
            slope = (end_iv - start_iv) / float(end_day - start_day)
            return float(slope)
                
        except Exception as e:
            self.logger.warning(f"Error calculating term structure slope: {e}")
            return 0.0
    
    def _calculate_term_structure_curvature(self, days: List[int], ivs: List[float]) -> float:
        """Calculate term structure curvature (second derivative)"""
        try:
            if len(days) < 3:
                return 0.0
            
            # Fit polynomial and calculate second derivative
            days_array = np.array(days)
            ivs_array = np.array(ivs)
            
            # Sort by days
            sorted_indices = np.argsort(days_array)
            sorted_days = days_array[sorted_indices]
            sorted_ivs = ivs_array[sorted_indices]
            
            # Fit second-order polynomial
            if len(sorted_days) >= 3:
                coeffs = np.polyfit(sorted_days, sorted_ivs, 2)
                # Second derivative of ax^2 + bx + c is 2a
                curvature = 2 * coeffs[0]
                return curvature
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating curvature: {e}")
            return 0.0
    
    def _score_term_structure(self, slope_0_30: float, slope_30_60: float, curvature: float) -> float:
        """Score term structure attractiveness for calendar spreads"""
        try:
            score = 0.0
            
            # Negative slope is good for calendar spreads
            if slope_0_30 < 0:
                score += 30
            
            # Steeper negative slope is better
            score += abs(slope_0_30) * 1000
            
            # Consistency in slope
            if slope_0_30 < 0 and slope_30_60 < 0:
                score += 20
            
            # Moderate curvature is preferred
            score += max(0, 10 - abs(curvature) * 10000)
            
            return max(0, min(100, score))
            
        except Exception as e:
            self.logger.warning(f"Error scoring term structure: {e}")
            return 50.0
    
    def _calculate_volatility_skew(self, strikes: np.ndarray, volatilities: np.ndarray, 
                                 expirations: np.ndarray, target_expiry: float) -> float:
        """Calculate volatility skew for specific expiration"""
        try:
            # Filter data for target expiration (within tolerance)
            tolerance = 0.05  # 0.05 years ~ 18 days
            mask = np.abs(expirations - target_expiry) < tolerance
            
            if not np.any(mask):
                return 0.0
            
            exp_strikes = strikes[mask]
            exp_vols = volatilities[mask]
            
            if len(exp_strikes) < 3:
                return 0.0
            
            # Calculate skew as slope of IV vs log(moneyness)
            # Assuming we have underlying price, use median strike as proxy
            underlying_approx = np.median(exp_strikes)
            log_moneyness = np.log(exp_strikes / underlying_approx)
            
            # Linear regression
            if len(log_moneyness) >= 2:
                slope = np.polyfit(log_moneyness, exp_vols, 1)[0]
                return slope
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating volatility skew: {e}")
            return 0.0
    
    def _calculate_smile_coefficient(self, moneyness: np.ndarray, volatilities: np.ndarray) -> float:
        """Calculate volatility smile coefficient"""
        try:
            if len(moneyness) < 5:
                return 0.0
            
            # Fit quadratic function: IV = a + b*m + c*m^2
            # where m is log(moneyness)
            log_moneyness = np.log(moneyness)
            
            # Quadratic fit
            coeffs = np.polyfit(log_moneyness, volatilities, 2)
            
            # Smile coefficient is the quadratic term
            smile_coeff = coeffs[0]
            
            return smile_coeff
            
        except Exception as e:
            self.logger.warning(f"Error calculating smile coefficient: {e}")
            return 0.0
    
    def _calculate_wing_risk(self, moneyness: np.ndarray, volatilities: np.ndarray) -> float:
        """Calculate wing risk (extreme moneyness volatility behavior)"""
        try:
            if len(moneyness) < 5:
                return 0.0
            
            # Sort by moneyness
            sorted_indices = np.argsort(moneyness)
            sorted_moneyness = moneyness[sorted_indices]
            sorted_vols = volatilities[sorted_indices]
            
            # Look at extreme wings (< 0.8 or > 1.2 moneyness)
            left_wing_mask = sorted_moneyness < 0.8
            right_wing_mask = sorted_moneyness > 1.2
            
            wing_risk = 0.0
            
            # Calculate volatility increase in wings
            if np.any(left_wing_mask):
                left_wing_vols = sorted_vols[left_wing_mask]
                if len(left_wing_vols) > 1:
                    left_wing_slope = np.polyfit(
                        sorted_moneyness[left_wing_mask], left_wing_vols, 1
                    )[0]
                    wing_risk += abs(left_wing_slope)
            
            if np.any(right_wing_mask):
                right_wing_vols = sorted_vols[right_wing_mask]
                if len(right_wing_vols) > 1:
                    right_wing_slope = np.polyfit(
                        sorted_moneyness[right_wing_mask], right_wing_vols, 1
                    )[0]
                    wing_risk += abs(right_wing_slope)
            
            return wing_risk
            
        except Exception as e:
            self.logger.warning(f"Error calculating wing risk: {e}")
            return 0.0
    
    # Volatility forecasting methods
    def _garch_forecast(self, vol_series: pd.Series, horizon: int) -> Tuple[float, float, float]:
        """Simple GARCH(1,1) volatility forecast"""
        try:
            # Simplified GARCH implementation
            # In practice, use arch package for proper GARCH modeling
            
            if len(vol_series) < 50:
                current_vol = vol_series.iloc[-1] if len(vol_series) > 0 else self.default_volatility
                return current_vol, current_vol * 0.8, current_vol * 1.2
            
            # Use exponential weighted moving average as GARCH proxy
            ewm_vol = vol_series.ewm(span=20).mean().iloc[-1]
            
            # Simple persistence assumption
            forecast_vol = ewm_vol
            
            # Confidence intervals (simplified)
            std_error = vol_series.tail(30).std()
            lower_ci = forecast_vol - 1.96 * std_error
            upper_ci = forecast_vol + 1.96 * std_error
            
            return forecast_vol, lower_ci, upper_ci
            
        except Exception as e:
            self.logger.error(f"Error in GARCH forecast: {e}")
            current_vol = vol_series.iloc[-1] if len(vol_series) > 0 else self.default_volatility
            return current_vol, current_vol * 0.8, current_vol * 1.2
    
    def _exponential_smoothing_forecast(self, vol_series: pd.Series, horizon: int) -> Tuple[float, float, float]:
        """Exponential smoothing volatility forecast"""
        try:
            if len(vol_series) < 10:
                current_vol = vol_series.iloc[-1] if len(vol_series) > 0 else self.default_volatility
                return current_vol, current_vol * 0.8, current_vol * 1.2
            
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing parameter
            forecast_vol = vol_series.ewm(alpha=alpha).mean().iloc[-1]
            
            # Confidence intervals based on historical volatility of volatility
            vol_of_vol = vol_series.tail(30).std()
            lower_ci = forecast_vol - 1.96 * vol_of_vol
            upper_ci = forecast_vol + 1.96 * vol_of_vol
            
            return forecast_vol, lower_ci, upper_ci
            
        except Exception as e:
            self.logger.error(f"Error in exponential smoothing forecast: {e}")
            current_vol = vol_series.iloc[-1] if len(vol_series) > 0 else self.default_volatility
            return current_vol, current_vol * 0.8, current_vol * 1.2
    
    def _mean_reversion_forecast(self, vol_series: pd.Series, horizon: int) -> Tuple[float, float, float]:
        """Mean reversion volatility forecast"""
        try:
            if len(vol_series) < 20:
                current_vol = vol_series.iloc[-1] if len(vol_series) > 0 else self.default_volatility
                return current_vol, current_vol * 0.8, current_vol * 1.2
            
            # Calculate long-term mean
            long_term_mean = vol_series.tail(252).mean() if len(vol_series) >= 252 else vol_series.mean()
            current_vol = vol_series.iloc[-1]
            
            # Mean reversion speed (simplified)
            reversion_speed = 0.1  # 10% reversion per period
            
            # Forecast: current + reversion_speed * (mean - current)
            forecast_vol = current_vol + reversion_speed * (long_term_mean - current_vol)
            
            # Confidence intervals
            historical_std = vol_series.tail(60).std()
            lower_ci = forecast_vol - 1.96 * historical_std
            upper_ci = forecast_vol + 1.96 * historical_std
            
            return forecast_vol, lower_ci, upper_ci
            
        except Exception as e:
            self.logger.error(f"Error in mean reversion forecast: {e}")
            current_vol = vol_series.iloc[-1] if len(vol_series) > 0 else self.default_volatility
            return current_vol, current_vol * 0.8, current_vol * 1.2
    
    def _calculate_regime_probabilities(self, forecast_vol: float) -> Dict[VolatilityRegime, float]:
        """Calculate probability of each volatility regime"""
        try:
            # Convert volatility to VIX equivalent (simplified)
            vix_equivalent = forecast_vol * 100
            
            # Create normal distributions around regime centers
            regime_centers = {
                VolatilityRegime.VERY_LOW: 10,
                VolatilityRegime.LOW: 14,
                VolatilityRegime.NORMAL: 20,
                VolatilityRegime.ELEVATED: 28,
                VolatilityRegime.HIGH: 38,
                VolatilityRegime.EXTREME: 55
            }
            
            probabilities = {}
            total_prob = 0
            
            for regime, center in regime_centers.items():
                # Use normal distribution with std=5
                prob = norm.pdf(vix_equivalent, loc=center, scale=5)
                probabilities[regime] = prob
                total_prob += prob
            
            # Normalize to sum to 1
            if total_prob > 0:
                for regime in probabilities:
                    probabilities[regime] /= total_prob
            else:
                # Equal probabilities as fallback
                equal_prob = 1.0 / len(regime_centers)
                probabilities = {regime: equal_prob for regime in regime_centers}
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error calculating regime probabilities: {e}")
            equal_prob = 1.0 / len(VolatilityRegime)
            return {regime: equal_prob for regime in VolatilityRegime}
    
    # Default/fallback methods
    def _get_default_vol_metrics(self) -> VolatilityMetrics:
        """Get default volatility metrics"""
        return VolatilityMetrics(
            realized_vol_30d=self.default_volatility,
            realized_vol_60d=self.default_volatility,
            realized_vol_90d=self.default_volatility,
            implied_vol_30d=self.default_volatility,
            iv_rv_ratio=1.0,
            vol_regime=VolatilityRegime.NORMAL,
            vol_percentile=50.0,
            vol_rank=0.5,
            vol_trend=0.0,
            vol_mean_reversion=0.5,
            vol_persistence=0.5,
            vol_clustering=0.5
        )
    
    def _get_default_term_structure(self, symbol: str) -> IVTermStructure:
        """Get default term structure"""
        return IVTermStructure(
            symbol=symbol,
            calculation_date=datetime.now(),
            expirations=[],
            days_to_expiration=[],
            atm_ivs=[],
            call_ivs=[],
            put_ivs=[],
            interpolation_function=None,
            slope_0_30=0.0,
            slope_30_60=0.0,
            slope_60_90=0.0,
            curvature=0.0,
            term_structure_score=50.0
        )
    
    def _get_default_surface(self, symbol: str, underlying_price: float) -> VolatilitySurface:
        """Get default volatility surface"""
        return VolatilitySurface(
            symbol=symbol,
            calculation_date=datetime.now(),
            strikes=np.array([]),
            expirations=np.array([]),
            volatilities=np.array([]),
            moneyness=np.array([]),
            time_to_expiry=np.array([]),
            surface_function=None,
            skew_30d=0.0,
            skew_60d=0.0,
            smile_coefficient=0.0,
            wing_risk=0.0
        )
    
    def _get_default_forecast(self, symbol: str, horizon: int) -> VolatilityForecast:
        """Get default volatility forecast"""
        equal_prob = 1.0 / len(VolatilityRegime)
        regime_probs = {regime: equal_prob for regime in VolatilityRegime}
        
        return VolatilityForecast(
            symbol=symbol,
            forecast_horizon=horizon,
            current_vol=self.default_volatility,
            forecasted_vol=self.default_volatility,
            confidence_interval_lower=self.default_volatility * 0.8,
            confidence_interval_upper=self.default_volatility * 1.2,
            forecast_model="default",
            regime_probability=regime_probs,
            forecast_timestamp=datetime.now()
        )
    
    # Cache management methods
    def _get_from_cache(self, key: str, cache_type: str = 'vol') -> Any:
        """Get data from cache if not expired"""
        with self._cache_lock:
            cache_dict = getattr(self, f'_{cache_type}_cache')
            
            if key in cache_dict:
                data, timestamp = cache_dict[key]
                if time.time() - timestamp < self.cache_ttl:
                    return data
                else:
                    del cache_dict[key]
            return None
    
    def _save_to_cache(self, key: str, data: Any, cache_type: str = 'vol') -> None:
        """Save data to cache with timestamp"""
        with self._cache_lock:
            cache_dict = getattr(self, f'_{cache_type}_cache')
            cache_dict[key] = (data, time.time())
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        with self._cache_lock:
            self._vol_cache.clear()
            self._iv_cache.clear()
            self._surface_cache.clear()
        self.logger.info("Volatility cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                "vol_cache_entries": len(self._vol_cache),
                "iv_cache_entries": len(self._iv_cache),
                "surface_cache_entries": len(self._surface_cache)
            }
    
    # Advanced analysis methods
    def analyze_volatility_regime_transitions(self, symbol: str, lookback_days: int = 252) -> Dict[str, Any]:
        """Analyze volatility regime transition probabilities"""
        try:
            # Get historical volatility
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)
            
            if len(vol_series) < lookback_days:
                return {"error": "Insufficient data"}
            
            # Convert to VIX equivalent and classify regimes
            recent_vols = vol_series.tail(lookback_days)
            vix_equivalent = recent_vols * 100
            
            regimes = []
            for vix_val in vix_equivalent:
                regime = self._classify_volatility_regime(vix_val)
                regimes.append(regime)
            
            # Calculate transition matrix
            unique_regimes = list(VolatilityRegime)
            transition_matrix = np.zeros((len(unique_regimes), len(unique_regimes)))
            
            for i in range(len(regimes) - 1):
                current_regime = regimes[i]
                next_regime = regimes[i + 1]
                
                current_idx = unique_regimes.index(current_regime)
                next_idx = unique_regimes.index(next_regime)
                
                transition_matrix[current_idx, next_idx] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1)
            for i in range(len(row_sums)):
                if row_sums[i] > 0:
                    transition_matrix[i] = transition_matrix[i] / row_sums[i]
            
            return {
                "transition_matrix": transition_matrix.tolist(),
                "regime_labels": [r.value for r in unique_regimes],
                "current_regime": regimes[-1].value if regimes else "unknown",
                "regime_persistence": np.diag(transition_matrix).mean()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime transitions: {e}")
            return {"error": str(e)}
    
    def calculate_volatility_risk_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculate advanced volatility risk metrics"""
        try:
            vol_metrics = self.calculate_volatility_metrics(symbol)
            
            # VaR of volatility (volatility of volatility)
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)
            
            if len(vol_series) < 30:
                return {"error": "Insufficient data"}
            
            recent_vols = vol_series.tail(60)
            vol_of_vol = recent_vols.std()
            vol_var_95 = recent_vols.quantile(0.05)  # 5th percentile
            vol_var_99 = recent_vols.quantile(0.01)  # 1st percentile
            
            # Maximum drawdown of volatility
            running_max = recent_vols.expanding().max()
            drawdowns = (recent_vols - running_max) / running_max
            max_vol_drawdown = drawdowns.min()
            
            # Volatility Sharpe ratio (return per unit of volatility risk)
            period = self._period_from_days(90)
            price_data = self.market_data.get_historical_data(symbol, period=period, interval="1d")
            
            vol_sharpe = 0.0
            if not price_data.empty:
                returns = price_data['Close'].pct_change().dropna()
                if len(returns) > 0 and vol_of_vol > 0:
                    vol_sharpe = returns.mean() / vol_of_vol
            
            return {
                "volatility_of_volatility": vol_of_vol,
                "vol_var_95": vol_var_95,
                "vol_var_99": vol_var_99,
                "max_volatility_drawdown": abs(max_vol_drawdown),
                "volatility_sharpe_ratio": vol_sharpe,
                "vol_skewness": float(recent_vols.skew()),
                "vol_kurtosis": float(recent_vols.kurtosis()),
                "vol_regime": vol_metrics.vol_regime.value,
                "vol_persistence": vol_metrics.vol_persistence
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility risk metrics: {e}")
            return {"error": str(e)}

    def calculate_iv_rank(self, symbol: str, current_iv: float, period_days: int = 252) -> float:
        """
        Calculate implied volatility rank - current IV vs historical IV range over period
        Returns rank from 0-1 where 1 = highest IV in period
        """
        try:
            # Get historical IV data (approximated from RV for now)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            # Get historical volatility series as proxy for IV
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)

            if len(vol_series) < 30:
                # Fallback calculation if insufficient data
                return 0.5  # Neutral rank

            # Calculate rank: (current - min) / (max - min)
            hist_vols = vol_series.dropna()
            if len(hist_vols) < 10:
                return 0.5

            vol_min = hist_vols.min()
            vol_max = hist_vols.max()

            if vol_max <= vol_min:
                return 0.5

            iv_rank = (current_iv - vol_min) / (vol_max - vol_min)
            return max(0.0, min(1.0, iv_rank))  # Clamp to 0-1 range

        except Exception as e:
            self.logger.warning(f"Error calculating IV rank for {symbol}: {e}")
            return 0.5  # Default neutral rank

    def calculate_iv_percentile(self, symbol: str, current_iv: float, period_days: int = 252) -> float:
        """
        Calculate implied volatility percentile - what % of time IV was below current level
        Returns percentile from 0-100 where 100 = highest IV in period
        """
        try:
            # Get historical IV data (approximated from RV for now)
            vol_series = self.calculate_realized_volatility(symbol, 30, return_last_only=False)

            if len(vol_series) < 30:
                return 50.0  # Neutral percentile

            hist_vols = vol_series.dropna()
            if len(hist_vols) < 10:
                return 50.0

            # Calculate what percentage of historical values are below current IV
            below_current = (hist_vols < current_iv).sum()
            total_count = len(hist_vols)

            percentile = (below_current / total_count) * 100.0
            return max(0.0, min(100.0, percentile))

        except Exception as e:
            self.logger.warning(f"Error calculating IV percentile for {symbol}: {e}")
            return 50.0  # Default neutral percentile

    def export_volatility_report(self, symbol: str, output_file: str) -> bool:
        """Export comprehensive volatility analysis report"""
        try:
            # Gather all volatility data
            vol_metrics = self.calculate_volatility_metrics(symbol)
            vol_forecast = self.forecast_volatility(symbol)
            risk_metrics = self.calculate_volatility_risk_metrics(symbol)
            regime_analysis = self.analyze_volatility_regime_transitions(symbol)
            
            report = {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "volatility_metrics": asdict(vol_metrics),
                "volatility_forecast": asdict(vol_forecast),
                "risk_metrics": risk_metrics,
                "regime_analysis": regime_analysis,
                "cache_stats": self.get_cache_stats(),
                "service_config": {
                    "default_window": self.default_window,
                    "trading_periods": self.trading_periods,
                    "min_volatility": self.min_volatility,
                    "max_volatility": self.max_volatility
                }
            }
            
            with open(output_file, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Volatility report exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting volatility report: {e}")
            return False
