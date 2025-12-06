"""
Options Service - Professional Options Data Management
======================================================

Handles all options-related data operations including:
- Option chain retrieval with timeout protection
- Expiration date management
- Option pricing and Greeks calculations
- IV term structure analysis
- Calendar spread optimization

Part of Professional Options Calculator v9.1
Optimized for Apple Silicon and PySide6
"""

import logging
import signal
import threading
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import interp1d
from scipy.stats import norm

# Import your existing utilities (adjust imports based on your structure)
from utils.config_manager import ConfigManager
from utils.logger import get_logger
from services.market_data import MarketDataService

logger = get_logger(__name__)

class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"

@dataclass
class OptionContract:
    """Individual option contract data"""
    symbol: str
    strike: float
    expiration: str
    option_type: OptionType
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price between bid and ask"""
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        elif self.last and self.last > 0:
            return self.last
        return 0.0
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

@dataclass
class OptionChain:
    """Complete option chain for an expiration"""
    symbol: str
    expiration: str
    underlying_price: float
    calls: List[OptionContract]
    puts: List[OptionContract]
    days_to_expiration: int
    
    def get_atm_strike(self) -> float:
        """Get at-the-money strike price"""
        if not self.calls and not self.puts:
            return self.underlying_price
        
        # Get all available strikes
        strikes = []
        if self.calls:
            strikes.extend([call.strike for call in self.calls])
        if self.puts:
            strikes.extend([put.strike for put in self.puts])
        
        if not strikes:
            return self.underlying_price
        
        # Find closest strike to underlying price
        strikes = sorted(set(strikes))
        closest_strike = min(strikes, key=lambda x: abs(x - self.underlying_price))
        return closest_strike
    
    def get_contract(self, strike: float, option_type: OptionType) -> Optional[OptionContract]:
        """Get specific option contract"""
        contracts = self.calls if option_type == OptionType.CALL else self.puts
        
        for contract in contracts:
            if abs(contract.strike - strike) < 0.01:  # Account for floating point precision
                return contract
        return None
    
    def get_straddle_price(self, strike: Optional[float] = None) -> float:
        """Calculate straddle price at given strike (ATM if None)"""
        if strike is None:
            strike = self.get_atm_strike()
        
        call = self.get_contract(strike, OptionType.CALL)
        put = self.get_contract(strike, OptionType.PUT)
        
        call_price = call.mid_price if call else 0.0
        put_price = put.mid_price if put else 0.0
        
        return call_price + put_price

@dataclass
class CalendarSpreadData:
    """Calendar spread analysis data"""
    symbol: str
    strike: float
    short_expiration: str
    long_expiration: str
    short_chain: OptionChain
    long_chain: OptionChain
    short_contract: OptionContract
    long_contract: OptionContract
    debit: float
    max_profit_estimate: float
    breakeven_range: Tuple[float, float]
    
    @property
    def short_premium(self) -> float:
        """Premium collected from short leg"""
        return self.short_contract.mid_price
    
    @property
    def long_premium(self) -> float:
        """Premium paid for long leg"""
        return self.long_contract.mid_price
    
    @property
    def time_decay_advantage(self) -> float:
        """Time decay advantage (short theta / long theta ratio)"""
        if (self.short_contract.theta and self.long_contract.theta and 
            self.long_contract.theta != 0):
            return abs(self.short_contract.theta) / abs(self.long_contract.theta)
        return 1.0

class OptionsService:
    """
    Professional Options Data Service
    
    Provides comprehensive options data management with:
    - Timeout-protected data retrieval
    - Intelligent caching
    - Error handling and fallbacks
    - Calendar spread optimization
    - IV term structure analysis
    """
    
    def __init__(self, config_manager: ConfigManager, market_data_service: MarketDataService):
        self.config = config_manager
        self.market_data = market_data_service
        self.logger = logger
        
        # Configuration
        self.default_timeout = 30
        self.max_retries = 3
        self.cache_ttl = 300  # 5 minutes
        
        # Cache
        self._option_chain_cache = {}
        self._expiration_cache = {}
        
        # Thread safety
        self._cache_lock = threading.Lock()
        
        self.logger.info("OptionsService initialized")
    
    def get_available_expirations(self, symbol: str, timeout: Optional[int] = None) -> List[str]:
        """
        Get available option expiration dates with timeout protection
        
        Args:
            symbol: Stock ticker symbol
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            List of expiration date strings in YYYY-MM-DD format
        """
        if timeout is None:
            timeout = self.default_timeout
        
        # Check cache first
        cache_key = f"{symbol}_expirations"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        self.logger.info(f"Fetching option expirations for {symbol} with {timeout}s timeout")
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Options expiration fetch timed out for {symbol}")
        
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                stock = yf.Ticker(symbol)
                expirations = list(stock.options)
                signal.alarm(0)  # Cancel timeout
                
                if expirations:
                    # Filter out expired options
                    today = datetime.now().date()
                    valid_expirations = []
                    
                    for exp_str in expirations:
                        try:
                            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                            if exp_date > today:
                                valid_expirations.append(exp_str)
                        except ValueError:
                            continue
                    
                    # Cache result
                    self._save_to_cache(cache_key, valid_expirations)
                    
                    self.logger.info(f"Found {len(valid_expirations)} valid expirations for {symbol}")
                    return valid_expirations
                else:
                    self.logger.warning(f"No option expirations found for {symbol}")
                    return []
                    
            except TimeoutError:
                signal.alarm(0)
                self.logger.error(f"Options expiration fetch timed out for {symbol}")
                return []
            except Exception as e:
                signal.alarm(0)
                self.logger.error(f"Options expiration error for {symbol}: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f"Options expiration setup error for {symbol}: {e}")
            return []
    
    def get_option_chain(self, symbol: str, expiration: str, 
                        timeout: Optional[int] = None) -> Optional[OptionChain]:
        """
        Get option chain for specific expiration with timeout protection
        
        Args:
            symbol: Stock ticker symbol
            expiration: Expiration date string (YYYY-MM-DD)
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            OptionChain object or None if failed
        """
        if timeout is None:
            timeout = self.default_timeout
        
        # Check cache first
        cache_key = f"{symbol}_{expiration}_chain"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        self.logger.info(f"Fetching option chain for {symbol} {expiration} with {timeout}s timeout")
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Option chain fetch timed out for {symbol} {expiration}")
        
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                stock = yf.Ticker(symbol)
                option_chain = stock.option_chain(expiration)
                signal.alarm(0)  # Cancel timeout
                
                if option_chain and hasattr(option_chain, 'calls') and hasattr(option_chain, 'puts'):
                    # Get underlying price
                    underlying_price = self.market_data.get_current_price(symbol)
                    if not underlying_price:
                        underlying_price = 100.0  # Fallback
                    
                    # Calculate days to expiration
                    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                    today = datetime.now().date()
                    days_to_exp = (exp_date - today).days
                    
                    # Convert calls and puts to our format
                    calls = self._convert_options_dataframe(
                        option_chain.calls, symbol, expiration, OptionType.CALL
                    )
                    puts = self._convert_options_dataframe(
                        option_chain.puts, symbol, expiration, OptionType.PUT
                    )
                    
                    # Create OptionChain object
                    chain = OptionChain(
                        symbol=symbol,
                        expiration=expiration,
                        underlying_price=underlying_price,
                        calls=calls,
                        puts=puts,
                        days_to_expiration=days_to_exp
                    )
                    
                    # Cache result
                    self._save_to_cache(cache_key, chain)
                    
                    self.logger.info(f"Successfully retrieved option chain for {symbol} {expiration}")
                    return chain
                else:
                    self.logger.warning(f"Empty option chain for {symbol} {expiration}")
                    return None
                    
            except TimeoutError:
                signal.alarm(0)
                self.logger.error(f"Option chain fetch timed out for {symbol} {expiration}")
                return None
            except Exception as e:
                signal.alarm(0)
                self.logger.error(f"Option chain error for {symbol} {expiration}: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Option chain setup error for {symbol}: {e}")
            return None
    
    def get_calendar_spread_data(self, symbol: str, strike: Optional[float] = None) -> Optional[CalendarSpreadData]:
        """
        Get calendar spread analysis data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            strike: Strike price (uses ATM if None)
            
        Returns:
            CalendarSpreadData object or None if failed
        """
        try:
            # Get available expirations
            expirations = self.get_available_expirations(symbol)
            if len(expirations) < 2:
                self.logger.error(f"Need at least 2 expirations for calendar spread, got {len(expirations)}")
                return None
            
            # Get first two expirations (short and long)
            short_exp = expirations[0]
            long_exp = expirations[1]
            
            # Get option chains
            short_chain = self.get_option_chain(symbol, short_exp)
            long_chain = self.get_option_chain(symbol, long_exp)
            
            if not short_chain or not long_chain:
                self.logger.error(f"Failed to get option chains for {symbol}")
                return None
            
            # Determine strike price
            if strike is None:
                strike = short_chain.get_atm_strike()
            
            # Get specific contracts (using puts for calendar spread)
            short_contract = short_chain.get_contract(strike, OptionType.PUT)
            long_contract = long_chain.get_contract(strike, OptionType.PUT)
            
            if not short_contract or not long_contract:
                self.logger.error(f"Could not find contracts at strike {strike} for {symbol}")
                return None
            
            # Calculate debit
            debit = long_contract.mid_price - short_contract.mid_price
            
            # Estimate max profit (simplified)
            max_profit_estimate = debit * 0.5  # Conservative estimate
            
            # Estimate breakeven range (simplified)
            breakeven_range = (strike * 0.95, strike * 1.05)
            
            return CalendarSpreadData(
                symbol=symbol,
                strike=strike,
                short_expiration=short_exp,
                long_expiration=long_exp,
                short_chain=short_chain,
                long_chain=long_chain,
                short_contract=short_contract,
                long_contract=long_contract,
                debit=debit,
                max_profit_estimate=max_profit_estimate,
                breakeven_range=breakeven_range
            )
            
        except Exception as e:
            self.logger.error(f"Error getting calendar spread data for {symbol}: {e}")
            return None
    
    def build_iv_term_structure(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Build implied volatility term structure for a symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with term structure data or None if failed
        """
        try:
            # Get available expirations
            expirations = self.get_available_expirations(symbol)
            if not expirations:
                return None
            
            term_structure_data = {
                'symbol': symbol,
                'expirations': [],
                'days_to_expiration': [],
                'atm_ivs': [],
                'call_ivs': [],
                'put_ivs': []
            }
            
            today = datetime.now().date()
            
            # Process each expiration
            for exp_str in expirations[:6]:  # Limit to first 6 expirations for performance
                try:
                    # Get option chain
                    chain = self.get_option_chain(symbol, exp_str)
                    if not chain:
                        continue
                    
                    # Calculate days to expiration
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    days_to_exp = (exp_date - today).days
                    
                    if days_to_exp <= 0:
                        continue
                    
                    # Get ATM contracts
                    atm_strike = chain.get_atm_strike()
                    atm_call = chain.get_contract(atm_strike, OptionType.CALL)
                    atm_put = chain.get_contract(atm_strike, OptionType.PUT)
                    
                    if not atm_call or not atm_put:
                        continue
                    
                    # Calculate average IV
                    call_iv = atm_call.implied_volatility if atm_call.implied_volatility else 0
                    put_iv = atm_put.implied_volatility if atm_put.implied_volatility else 0
                    avg_iv = (call_iv + put_iv) / 2 if (call_iv and put_iv) else (call_iv or put_iv)
                    
                    if avg_iv <= 0:
                        continue
                    
                    # Store data
                    term_structure_data['expirations'].append(exp_str)
                    term_structure_data['days_to_expiration'].append(days_to_exp)
                    term_structure_data['atm_ivs'].append(avg_iv)
                    term_structure_data['call_ivs'].append(call_iv)
                    term_structure_data['put_ivs'].append(put_iv)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing expiration {exp_str} for {symbol}: {e}")
                    continue
            
            if not term_structure_data['days_to_expiration']:
                return None
            
            # Create interpolation function
            if len(term_structure_data['days_to_expiration']) >= 2:
                days = np.array(term_structure_data['days_to_expiration'])
                ivs = np.array(term_structure_data['atm_ivs'])
                
                # Sort by days
                sort_idx = days.argsort()
                days_sorted = days[sort_idx]
                ivs_sorted = ivs[sort_idx]
                
                # Create interpolation function
                try:
                    iv_interp = interp1d(days_sorted, ivs_sorted, 
                                       kind='linear', fill_value='extrapolate')
                    term_structure_data['interpolation_function'] = iv_interp
                except Exception as e:
                    self.logger.warning(f"Could not create interpolation function: {e}")
            
            return term_structure_data
            
        except Exception as e:
            self.logger.error(f"Error building IV term structure for {symbol}: {e}")
            return None
    
    def calculate_iv_rank_percentile(self, symbol: str, current_iv: float, 
                                   lookback_days: int = 252) -> Dict[str, float]:
        """
        Calculate IV rank and percentile based on historical data
        
        Args:
            symbol: Stock ticker symbol
            current_iv: Current implied volatility
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with iv_rank and iv_percentile
        """
        try:
            # Get historical price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 60)  # Extra buffer
            
            historical_data = self.market_data.get_historical_data(
                symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            
            if historical_data.empty or len(historical_data) < 30:
                return {"iv_rank": 0.5, "iv_percentile": 50}
            
            # Calculate historical realized volatility as proxy for IV
            # This is a simplified approach - in production you'd want actual IV history
            returns = historical_data['Close'].pct_change().dropna()
            
            # Calculate rolling 30-day volatility
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) < 10:
                return {"iv_rank": 0.5, "iv_percentile": 50}
            
            # Calculate IV rank (position within min-max range)
            vol_min = rolling_vol.min()
            vol_max = rolling_vol.max()
            vol_range = vol_max - vol_min
            
            if vol_range > 0:
                iv_rank = (current_iv - vol_min) / vol_range
                iv_rank = max(0, min(1, iv_rank))  # Clamp between 0 and 1
            else:
                iv_rank = 0.5
            
            # Calculate IV percentile
            iv_percentile = (rolling_vol <= current_iv).sum() / len(rolling_vol) * 100
            iv_percentile = max(0, min(100, iv_percentile))  # Clamp between 0 and 100
            
            return {
                "iv_rank": iv_rank,
                "iv_percentile": iv_percentile
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating IV rank/percentile for {symbol}: {e}")
            return {"iv_rank": 0.5, "iv_percentile": 50}
    
    def find_optimal_calendar_strikes(self, symbol: str, 
                                    max_strikes: int = 5) -> List[CalendarSpreadData]:
        """
        Find optimal strikes for calendar spreads
        
        Args:
            symbol: Stock ticker symbol
            max_strikes: Maximum number of strikes to analyze
            
        Returns:
            List of CalendarSpreadData sorted by attractiveness
        """
        try:
            # Get current price
            current_price = self.market_data.get_current_price(symbol)
            if not current_price:
                return []
            
            # Get available expirations
            expirations = self.get_available_expirations(symbol)
            if len(expirations) < 2:
                return []
            
            # Get short-term option chain to find available strikes
            short_chain = self.get_option_chain(symbol, expirations[0])
            if not short_chain:
                return []
            
            # Get strikes around current price
            all_strikes = []
            if short_chain.puts:
                all_strikes.extend([put.strike for put in short_chain.puts])
            if short_chain.calls:
                all_strikes.extend([call.strike for call in short_chain.calls])
            
            if not all_strikes:
                return []
            
            # Filter strikes around current price (±20%)
            price_range = current_price * 0.2
            relevant_strikes = [
                strike for strike in set(all_strikes)
                if abs(strike - current_price) <= price_range
            ]
            
            # Sort by distance from current price and limit
            relevant_strikes.sort(key=lambda x: abs(x - current_price))
            relevant_strikes = relevant_strikes[:max_strikes]
            
            # Analyze each strike
            calendar_spreads = []
            for strike in relevant_strikes:
                calendar_data = self.get_calendar_spread_data(symbol, strike)
                if calendar_data:
                    calendar_spreads.append(calendar_data)
            
            # Sort by attractiveness (simplified scoring)
            calendar_spreads.sort(key=lambda x: self._score_calendar_spread(x), reverse=True)
            
            return calendar_spreads
            
        except Exception as e:
            self.logger.error(f"Error finding optimal calendar strikes for {symbol}: {e}")
            return []
    
    def _convert_options_dataframe(self, df: pd.DataFrame, symbol: str, 
                                 expiration: str, option_type: OptionType) -> List[OptionContract]:
        """Convert yfinance options dataframe to OptionContract objects"""
        contracts = []
        
        for _, row in df.iterrows():
            try:
                contract = OptionContract(
                    symbol=symbol,
                    strike=float(row.get('strike', 0)),
                    expiration=expiration,
                    option_type=option_type,
                    bid=float(row.get('bid', 0)),
                    ask=float(row.get('ask', 0)),
                    last=float(row.get('lastPrice', 0)),
                    volume=int(row.get('volume', 0)),
                    open_interest=int(row.get('openInterest', 0)),
                    implied_volatility=float(row.get('impliedVolatility', 0))
                )
                contracts.append(contract)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error converting option contract: {e}")
                continue
        
        return contracts
    
    def _score_calendar_spread(self, calendar_data: CalendarSpreadData) -> float:
        """
        Score calendar spread attractiveness (higher is better)
        
        Simple scoring based on:
        - Time decay advantage
        - Debit amount (lower is better)
        - Distance from ATM
        """
        try:
            score = 0.0
            
            # Time decay advantage (higher is better)
            score += calendar_data.time_decay_advantage * 20
            
            # Debit amount (lower debit gets higher score)
            underlying_price = calendar_data.short_chain.underlying_price
            debit_ratio = calendar_data.debit / underlying_price
            score += max(0, (0.05 - debit_ratio) * 100)  # Prefer debits < 5% of underlying
            
            # Distance from ATM (closer is better for calendar spreads)
            atm_distance = abs(calendar_data.strike - underlying_price) / underlying_price
            score += max(0, (0.1 - atm_distance) * 50)  # Prefer strikes within 10% of ATM
            
            # Liquidity (higher volume and open interest)
            liquidity_score = (
                np.log(max(1, calendar_data.short_contract.volume)) +
                np.log(max(1, calendar_data.long_contract.volume)) +
                np.log(max(1, calendar_data.short_contract.open_interest)) +
                np.log(max(1, calendar_data.long_contract.open_interest))
            )
            score += liquidity_score
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Error scoring calendar spread: {e}")
            return 0.0
    
    def _get_from_cache(self, key: str) -> Any:
        """Get data from cache if not expired"""
        with self._cache_lock:
            if key in self._option_chain_cache:
                data, timestamp = self._option_chain_cache[key]
                if time.time() - timestamp < self.cache_ttl:
                    return data
                else:
                    # Remove expired entry
                    del self._option_chain_cache[key]
            return None
    
    def _save_to_cache(self, key: str, data: Any) -> None:
        """Save data to cache with timestamp"""
        with self._cache_lock:
            self._option_chain_cache[key] = (data, time.time())
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        with self._cache_lock:
            self._option_chain_cache.clear()
            self._expiration_cache.clear()
        self.logger.info("Options cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._cache_lock:
            current_time = time.time()
            valid_entries = 0
            expired_entries = 0
            
            for data, timestamp in self._option_chain_cache.values():
                if current_time - timestamp < self.cache_ttl:
                    valid_entries += 1
                else:
                    expired_entries += 1
            
            return {
                "total_entries": len(self._option_chain_cache),
                "valid_entries": valid_entries,
                "expired_entries": expired_entries
            }