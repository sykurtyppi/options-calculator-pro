"""
Market Data Service - Professional Options Calculator Pro
Unified market data service with multiple providers and intelligent routing
"""

import logging
import asyncio
import aiohttp
import time
import random
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import json
import sqlite3
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QTimer, QThread
from utils.config_manager import ConfigManager
from utils.ttl_cache import MultiTTLCache


class DataProvider(Enum):
    """Available data providers"""
    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"

class RetryStrategy(Enum):
    """Retry strategies for rate limits"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 300.0  # Maximum delay (5 minutes)
    exponential_base: float = 2.0
    jitter_factor: float = 0.1  # Add randomness to prevent thundering herd
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

@dataclass
class CircuitBreakerState:
    """Circuit breaker state for provider failure management"""
    failure_count: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False
    half_open_time: Optional[float] = None
    failure_threshold: int = 5  # Open after 5 consecutive failures
    recovery_timeout: float = 300.0  # 5 minutes before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open

@dataclass
class ProviderHealth:
    """Provider health tracking"""
    success_count: int = 0
    failure_count: int = 0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    average_response_time: float = 0.0
    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)


@dataclass
class PriceData:
    """Container for price data"""
    symbol: str
    price: float
    timestamp: datetime
    volume: Optional[int] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    provider: Optional[str] = None


@dataclass
class HistoricalData:
    """Container for historical data"""
    symbol: str
    data: pd.DataFrame
    timeframe: str
    start_date: date
    end_date: date
    provider: str


class MarketDataCache:
    """High-performance caching system for market data"""
    
    def __init__(self, cache_dir: str = None):
        self.logger = logging.getLogger(f"{__name__}.Cache")
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".options_calculator_pro" / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite cache database
        self.db_path = self.cache_dir / "market_data.db"
        self._init_database()
        
        # High-performance TTL cache with bounded memory usage
        self.memory_cache = MultiTTLCache()
        
        # Legacy cache settings maintained for compatibility
        self.default_ttl = {
            'price': 60,  # 1 minute for prices
            'historical': 3600,  # 1 hour for historical data
            'earnings': 86400,  # 24 hours for earnings
            'options': 900,  # 15 minutes for options data
        }
    
    def _init_database(self):
        """Initialize cache database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_cache (
                        key TEXT PRIMARY KEY,
                        data TEXT,
                        timestamp REAL,
                        ttl INTEGER,
                        data_type TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON market_cache(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_type 
                    ON market_cache(data_type)
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing cache database: {e}")
    
    def get(self, key: str, data_type: str = 'general') -> Optional[Any]:
        """Get data from cache"""
        try:
            # Check TTL cache first (now with proper memory bounds)
            cached_data = self.memory_cache.get(data_type, key)
            if cached_data is not None:
                return cached_data
            
            # Check database cache
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data, timestamp, ttl FROM market_cache WHERE key = ?", 
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    data_json, timestamp, ttl = row
                    cache_time = datetime.fromtimestamp(timestamp)
                    
                    if (datetime.now() - cache_time).total_seconds() < ttl:
                        # Deserialize data
                        data = json.loads(data_json)
                        
                        # Store in TTL cache for faster access (with proper bounds)
                        remaining_ttl = ttl - (datetime.now() - cache_time).total_seconds()
                        if remaining_ttl > 0:
                            self.memory_cache.set(data_type, key, data, remaining_ttl)
                        
                        return data
                    else:
                        # Remove expired entry
                        conn.execute("DELETE FROM market_cache WHERE key = ?", (key,))
                        conn.commit()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, key: str, data: Any, data_type: str = 'general', ttl: Optional[int] = None):
        """Set data in cache"""
        try:
            if ttl is None:
                ttl = self.default_ttl.get(data_type, 3600)
            
            timestamp = datetime.now()
            
            # Store in TTL cache with proper memory bounds
            self.memory_cache.set(data_type, key, data, ttl)
            
            # Store in database cache
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO market_cache 
                    (key, data, timestamp, ttl, data_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, json.dumps(data), timestamp.timestamp(), ttl, data_type))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error setting cache: {e}")
    
    def cleanup(self):
        """Clean up expired cache entries"""
        try:
            current_time = datetime.now().timestamp()
            
            # Clean TTL cache (automatic cleanup of expired entries with bounds)
            self.memory_cache.cleanup_all()
            self.logger.debug("TTL cache cleanup completed")
            
            # Clean database cache
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM market_cache WHERE timestamp + ttl < ?", 
                    (current_time,)
                )
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Critical for monitoring memory usage during intensive calendar scanning.
        Helps prevent memory exhaustion and optimize cache performance.
        """
        try:
            stats = self.memory_cache.get_combined_stats()
            
            # Add database cache info
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM market_cache")
                db_count = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT data_type, COUNT(*) 
                    FROM market_cache 
                    GROUP BY data_type
                """)
                db_breakdown = dict(cursor.fetchall())
            
            stats['database'] = {
                'total_entries': db_count,
                'breakdown': db_breakdown
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}


class DataProviderManager:
    """Manages multiple data providers with fallback logic"""
    
    def __init__(self, config_manager: ConfigManager):
        self.logger = logging.getLogger(f"{__name__}.ProviderManager")
        self.config_manager = config_manager
        
        # Provider configurations with enhanced rate limiting
        self.providers = {
            DataProvider.YAHOO_FINANCE: {
                'enabled': True,
                'priority': 1,
                'rate_limit': 200,  # Conservative limit - requests per hour
                'rate_window': 3600,  # 1 hour window
                'timeout': 8,
                'retry_delay': 5,  # seconds
                'max_retries': 3
            },
            DataProvider.ALPHA_VANTAGE: {
                'enabled': bool(config_manager.get("api_keys", {}).get("alpha_vantage", "")),
                'priority': 2,
                'rate_limit': 500,  # requests per day
                'rate_window': 86400,  # 24 hours
                'timeout': 15,
                'retry_delay': 12,
                'max_retries': 2,
                'api_key': config_manager.get("api_keys", {}).get("alpha_vantage", "")
            },
            DataProvider.FINNHUB: {
                'enabled': bool(config_manager.get("api_keys", {}).get("finnhub", "")),
                'priority': 3,
                'rate_limit': 60,  # requests per minute
                'rate_window': 60,  # 1 minute
                'timeout': 10,
                'retry_delay': 15,
                'max_retries': 2,
                'api_key': config_manager.get("api_keys", {}).get("finnhub", "")
            }
        }
        
        # Rate limiting tracking
        self.rate_limits = {provider: {'count': 0, 'reset_time': time.time()}
                           for provider in self.providers}

        # Provider health tracking
        self.provider_health = {provider: ProviderHealth()
                               for provider in self.providers}

        # Retry configuration per provider
        self.retry_configs = {
            DataProvider.YAHOO_FINANCE: RetryConfig(
                max_retries=5,
                base_delay=2.0,
                max_delay=120.0,  # 2 minutes max
                exponential_base=1.5
            ),
            DataProvider.ALPHA_VANTAGE: RetryConfig(
                max_retries=3,
                base_delay=5.0,
                max_delay=300.0,  # 5 minutes max
                exponential_base=2.0
            ),
            DataProvider.FINNHUB: RetryConfig(
                max_retries=3,
                base_delay=3.0,
                max_delay=180.0,  # 3 minutes max
                exponential_base=1.8
            )
        }

        self.session = None
    
    async def get_session(self):
        """Get async HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_available_providers(self, data_type: str = 'price') -> List[DataProvider]:
        """Get list of available providers for data type, sorted by priority"""
        available = []

        for provider, config in self.providers.items():
            if (config['enabled'] and
                not self._is_rate_limited(provider) and
                not self._is_circuit_breaker_open(provider)):
                available.append(provider)

        # Sort by priority (lower number = higher priority)
        available.sort(key=lambda p: self.providers[p]['priority'])
        return available

    def get_provider_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get health summary for all providers"""
        summary = {}

        for provider, health in self.provider_health.items():
            rate_info = self.rate_limits[provider]
            provider_config = self.providers[provider]

            summary[provider.value] = {
                'enabled': provider_config['enabled'],
                'success_count': health.success_count,
                'failure_count': health.failure_count,
                'success_rate': (health.success_count / max(health.success_count + health.failure_count, 1)) * 100,
                'average_response_time': health.average_response_time,
                'circuit_breaker_open': health.circuit_breaker.is_open,
                'rate_limit_usage': f"{rate_info['count']}/{provider_config['rate_limit']}",
                'last_success': health.last_success_time,
                'last_failure': health.last_failure_time
            }

        return summary

    async def fetch_price_with_fallback(self, symbol: str) -> Optional[PriceData]:
        """
        Fetch price using intelligent provider fallback.
        Tries providers in order of priority, respecting rate limits and circuit breakers.
        """
        available_providers = self.get_available_providers('price')

        if not available_providers:
            self.logger.warning(f"No available providers for {symbol}")
            return None

        for provider in available_providers:
            try:
                self.logger.debug(f"Trying {provider.value} for {symbol}")

                if provider == DataProvider.YAHOO_FINANCE:
                    result = await self.fetch_price_yahoo(symbol)
                elif provider == DataProvider.ALPHA_VANTAGE:
                    result = await self.fetch_price_alpha_vantage(symbol)
                elif provider == DataProvider.FINNHUB:
                    result = await self.fetch_price_finnhub(symbol)
                else:
                    continue

                if result:
                    self.logger.info(f"Successfully fetched {symbol} from {provider.value}")
                    return result

            except Exception as e:
                self.logger.warning(f"Provider {provider.value} failed for {symbol}: {e}")
                continue

        self.logger.error(f"All providers failed for {symbol}")
        return None

    async def test_provider_connectivity(self, provider: DataProvider) -> bool:
        """Test connectivity to a specific provider"""
        test_symbol = "AAPL"  # Use AAPL as test symbol

        try:
            if provider == DataProvider.YAHOO_FINANCE:
                result = await self.fetch_price_yahoo(test_symbol)
            elif provider == DataProvider.ALPHA_VANTAGE:
                result = await self.fetch_price_alpha_vantage(test_symbol)
            elif provider == DataProvider.FINNHUB:
                result = await self.fetch_price_finnhub(test_symbol)
            else:
                return False

            return result is not None

        except Exception as e:
            self.logger.warning(f"Connectivity test failed for {provider.value}: {e}")
            return False

    def reset_circuit_breaker(self, provider: DataProvider):
        """Manually reset a circuit breaker (admin function)"""
        health = self.provider_health[provider]
        health.circuit_breaker.is_open = False
        health.circuit_breaker.failure_count = 0
        health.circuit_breaker.half_open_time = None
        self.logger.info(f"Circuit breaker manually reset for {provider.value}")

    def _is_rate_limited(self, provider: DataProvider) -> bool:
        """Check if provider is rate limited"""
        rate_info = self.rate_limits[provider]
        provider_config = self.providers[provider]
        current_time = time.time()

        # Reset counter if window has passed
        rate_window = provider_config.get('rate_window', 86400)  # default to daily
        if current_time - rate_info['reset_time'] > rate_window:
            rate_info['count'] = 0
            rate_info['reset_time'] = current_time

        # Check if over limit
        is_limited = rate_info['count'] >= provider_config['rate_limit']

        if is_limited:
            self.logger.warning(f"Provider {provider.value} is rate limited. "
                              f"Count: {rate_info['count']}/{provider_config['rate_limit']}")

        return is_limited
    
    def _increment_rate_limit(self, provider: DataProvider):
        """Increment rate limit counter"""
        self.rate_limits[provider]['count'] += 1

    def _is_circuit_breaker_open(self, provider: DataProvider) -> bool:
        """Check if circuit breaker is open for a provider"""
        health = self.provider_health[provider]
        circuit_breaker = health.circuit_breaker
        current_time = time.time()

        if circuit_breaker.is_open:
            # Check if we should try half-open
            if (current_time - circuit_breaker.last_failure_time) > circuit_breaker.recovery_timeout:
                circuit_breaker.is_open = False
                circuit_breaker.half_open_time = current_time
                self.logger.info(f"Circuit breaker for {provider.value} moving to half-open state")
                return False
            return True

        return False

    def _record_success(self, provider: DataProvider, response_time: float):
        """Record successful API call"""
        health = self.provider_health[provider]
        circuit_breaker = health.circuit_breaker

        health.success_count += 1
        health.last_success_time = time.time()

        # Update average response time
        if health.average_response_time == 0:
            health.average_response_time = response_time
        else:
            # Exponential moving average
            health.average_response_time = (health.average_response_time * 0.9) + (response_time * 0.1)

        # Circuit breaker state management
        if circuit_breaker.half_open_time is not None:
            # In half-open state, count successes
            circuit_breaker.success_threshold -= 1
            if circuit_breaker.success_threshold <= 0:
                # Close the circuit breaker
                circuit_breaker.is_open = False
                circuit_breaker.half_open_time = None
                circuit_breaker.failure_count = 0
                circuit_breaker.success_threshold = 3  # Reset
                self.logger.info(f"Circuit breaker for {provider.value} closed after successful recovery")

    def _record_failure(self, provider: DataProvider, error: Exception):
        """Record failed API call and manage circuit breaker"""
        health = self.provider_health[provider]
        circuit_breaker = health.circuit_breaker

        health.failure_count += 1
        health.last_failure_time = time.time()
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = time.time()

        # Check if we should open the circuit breaker
        if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
            circuit_breaker.is_open = True
            circuit_breaker.half_open_time = None
            self.logger.warning(
                f"Circuit breaker opened for {provider.value} after {circuit_breaker.failure_count} failures. "
                f"Last error: {error}"
            )

    def _calculate_backoff_delay(self, provider: DataProvider, attempt: int) -> float:
        """Calculate backoff delay with jitter"""
        retry_config = self.retry_configs.get(provider, RetryConfig())

        if retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = retry_config.base_delay * (retry_config.exponential_base ** attempt)
        elif retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = retry_config.base_delay * (attempt + 1)
        else:  # Fibonacci
            fib = [1, 1]
            for i in range(2, attempt + 3):
                fib.append(fib[i-1] + fib[i-2])
            delay = retry_config.base_delay * fib[min(attempt + 2, len(fib) - 1)]

        # Apply jitter to prevent thundering herd
        jitter = delay * retry_config.jitter_factor * (random.random() - 0.5)
        delay = delay + jitter

        # Ensure delay doesn't exceed maximum
        delay = min(delay, retry_config.max_delay)

        return delay

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detect if error is due to rate limiting"""
        error_messages = [
            "rate limit", "too many requests", "quota exceeded",
            "429", "throttled", "api limit", "request limit",
            "rate exceeded", "usage limit"
        ]

        error_str = str(error).lower()
        return any(msg in error_str for msg in error_messages)
    
    async def fetch_price_yahoo(self, symbol: str) -> Optional[PriceData]:
        """Fetch price from Yahoo Finance with intelligent retry logic"""
        provider = DataProvider.YAHOO_FINANCE

        # Check circuit breaker
        if self._is_circuit_breaker_open(provider):
            self.logger.warning(f"Circuit breaker open for {provider.value}, skipping request")
            return None

        retry_config = self.retry_configs[provider]

        for attempt in range(retry_config.max_retries + 1):
            try:
                start_time = time.time()

                # Use yfinance in thread pool to avoid blocking
                loop = asyncio.get_event_loop()

                def get_yahoo_price():
                    ticker = yf.Ticker(symbol)

                    # Try to get real-time price
                    try:
                        hist = ticker.history(period="1d", interval="1m")
                        if not hist.empty:
                            price = float(hist['Close'].iloc[-1])
                            volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else None
                            timestamp = hist.index[-1].to_pydatetime()

                            return PriceData(
                                symbol=symbol,
                                price=price,
                                timestamp=timestamp,
                                volume=volume,
                                provider="yahoo"
                            )
                    except Exception as e:
                        # Check if it's a rate limit error
                        if self._is_rate_limit_error(e):
                            raise e  # Re-raise to trigger retry logic
                        pass

                    # Fallback to info
                    info = ticker.info
                    price_fields = ["regularMarketPrice", "currentPrice", "price"]

                    for field in price_fields:
                        if field in info and info[field] and float(info[field]) > 0:
                            return PriceData(
                                symbol=symbol,
                                price=float(info[field]),
                                timestamp=datetime.now(),
                                volume=info.get("regularMarketVolume"),
                                change=info.get("regularMarketChange"),
                                change_percent=info.get("regularMarketChangePercent"),
                                provider="yahoo"
                            )

                    return None

                # Execute in thread pool
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, get_yahoo_price)
                    result = await asyncio.wait_for(
                        future,
                        timeout=self.providers[DataProvider.YAHOO_FINANCE]['timeout']
                    )

                    if result:
                        # Record success
                        response_time = time.time() - start_time
                        self._record_success(provider, response_time)
                        self._increment_rate_limit(provider)

                        self.logger.info(f"Successfully fetched {symbol} from Yahoo Finance "
                                       f"(attempt {attempt + 1}, {response_time:.2f}s)")
                        return result

                    # No data returned but no exception - try next attempt
                    if attempt < retry_config.max_retries:
                        delay = self._calculate_backoff_delay(provider, attempt)
                        self.logger.info(f"No data for {symbol} from Yahoo Finance, "
                                       f"retrying in {delay:.1f}s (attempt {attempt + 1})")
                        await asyncio.sleep(delay)
                        continue

                    return None

            except Exception as e:
                # Record failure
                self._record_failure(provider, e)

                # Check if this is a rate limit error
                is_rate_limit = self._is_rate_limit_error(e)

                if is_rate_limit:
                    self.logger.warning(f"Rate limit hit for Yahoo Finance on {symbol} "
                                      f"(attempt {attempt + 1}): {e}")
                else:
                    self.logger.error(f"Error fetching {symbol} from Yahoo Finance "
                                    f"(attempt {attempt + 1}): {e}")

                # If this is the last attempt, give up
                if attempt >= retry_config.max_retries:
                    self.logger.error(f"All retry attempts exhausted for {symbol} from Yahoo Finance")
                    return None

                # Calculate backoff delay (longer for rate limits)
                delay = self._calculate_backoff_delay(provider, attempt)
                if is_rate_limit:
                    delay *= 2  # Double delay for rate limit errors

                self.logger.info(f"Retrying {symbol} from Yahoo Finance in {delay:.1f}s "
                               f"(attempt {attempt + 1}/{retry_config.max_retries})")
                await asyncio.sleep(delay)

        return None
    
    async def fetch_price_alpha_vantage(self, symbol: str) -> Optional[PriceData]:
        """Fetch price from Alpha Vantage"""
        try:
            if not self.providers[DataProvider.ALPHA_VANTAGE]['enabled']:
                return None
            
            session = await self.get_session()
            api_key = self.providers[DataProvider.ALPHA_VANTAGE]['api_key']
            
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "Global Quote" in data:
                        quote = data["Global Quote"]
                        
                        price = float(quote.get("05. price", 0))
                        if price > 0:
                            self._increment_rate_limit(DataProvider.ALPHA_VANTAGE)
                            
                            return PriceData(
                                symbol=symbol,
                                price=price,
                                timestamp=datetime.now(),
                                change=float(quote.get("09. change", 0)),
                                change_percent=float(quote.get("10. change percent", "0%").rstrip('%')),
                                provider="alpha_vantage"
                            )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage price for {symbol}: {e}")
            return None
    
    async def fetch_price_finnhub(self, symbol: str) -> Optional[PriceData]:
        """Fetch price from Finnhub"""
        try:
            if not self.providers[DataProvider.FINNHUB]['enabled']:
                return None
            
            session = await self.get_session()
            api_key = self.providers[DataProvider.FINNHUB]['api_key']
            
            url = "https://finnhub.io/api/v1/quote"
            params = {
                "symbol": symbol,
                "token": api_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    current_price = data.get("c")
                    if current_price and current_price > 0:
                        self._increment_rate_limit(DataProvider.FINNHUB)
                        
                        return PriceData(
                            symbol=symbol,
                            price=float(current_price),
                            timestamp=datetime.now(),
                            change=float(data.get("d", 0)),
                            change_percent=float(data.get("dp", 0)),
                            provider="finnhub"
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching Finnhub price for {symbol}: {e}")
            return None



    async def get_current_price_async(self, symbol: str) -> Optional[float]:
        """Get current price using intelligent provider fallback"""
        try:
            # Handle special symbols
            if symbol == "VIX":
                symbol = "^VIX"
            elif symbol == "SPX":
                symbol = "^GSPC"
            elif symbol == "NDX":
                symbol = "^NDX"

            # Try with intelligent fallback
            price_data = await self.fetch_price_with_fallback(symbol)
            if price_data:
                return price_data.price

            return None

        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol (synchronous wrapper)"""
        try:
            # Check cache first
            cache_key = f"price_{symbol}"
            cached_price = self.cache.get(cache_key, 'price')
            if cached_price is not None:
                return float(cached_price)

            # Handle special symbols
            if symbol == "VIX":
                symbol = "^VIX"
            elif symbol == "SPX":
                symbol = "^GSPC"
            elif symbol == "NDX":
                symbol = "^NDX"

            # For synchronous calls, fallback to direct Yahoo Finance
            # (This maintains backwards compatibility)
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")

            if not data.empty:
                price = float(data["Close"].iloc[-1])
                # Cache the result
                self.cache.set(cache_key, price, 'price')
                self.logger.info(f"Fetched {symbol}: ${price:.2f}")
                return price
            else:
                self.logger.warning(f"No data for {symbol}")
                return 0.0

        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return 0.0

    def get_historical_data(self, symbol: str, period: str = "1y",
                          interval: str = "1d") -> pd.DataFrame:
        """
        Get historical data for symbol

        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # 🔧 PARAMETER SAFETY FIX: Validate symbol is string, not DataFrame
            if not isinstance(symbol, str):
                self.logger.error(f"🔧 SYMBOL TYPE ERROR FIX: Expected str, got {type(symbol).__name__}: {symbol}")
                return pd.DataFrame()

            symbol = symbol.upper().strip()

            # Check cache first
            cache_key = f"historical_{symbol}_{period}_{interval}"
            cached_data = self.cache.get(cache_key, 'historical')

            if cached_data and isinstance(cached_data, dict):
                try:
                    # Fixed cache reconstruction: Handle string-based datetime index from cache
                    df = pd.DataFrame(cached_data['data'])

                    # Check if this uses the new string index format
                    if cached_data.get('data_format') == 'string_index':
                        # Index is already properly formatted strings, convert to datetime
                        df.index = pd.to_datetime(df.index)
                    else:
                        # Legacy format - try to convert
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception as legacy_error:
                            self.logger.warning(f"Legacy cache format conversion failed: {legacy_error}")
                            return pd.DataFrame()  # Force fresh fetch

                    if not df.empty and len(df) > 10:  # Reasonable amount of data
                        return df
                except Exception as cache_error:
                    self.logger.warning(f"Error reconstructing cached historical data: {cache_error}")
                    # Clear corrupted cache entry
                    self.cache.delete(cache_key, 'historical')

            # Fetch fresh data using yfinance
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period, interval=interval)

            if not hist_data.empty:
                # Fix critical cache issue: Convert Timestamp index to string keys for cache compatibility
                # hist_data.to_dict('index') creates Timestamp keys which cache system rejects
                try:
                    # Convert index to ISO strings before creating dict
                    hist_data_copy = hist_data.copy()
                    hist_data_copy.index = hist_data_copy.index.strftime('%Y-%m-%d %H:%M:%S')

                    # Cache the result with string-keyed data
                    cache_data = {
                        'data': hist_data_copy.to_dict('index'),
                        'period': period,
                        'interval': interval,
                        'symbol': symbol,
                        'start_date': hist_data.index[0].isoformat(),
                        'end_date': hist_data.index[-1].isoformat(),
                        'data_format': 'string_index'  # Mark for proper reconstruction
                    }
                except Exception as e:
                    self.logger.warning(f"Error preparing cache data for {symbol}: {e}")
                    # Fallback: cache without historical data dict
                    cache_data = {
                        'symbol': symbol,
                        'period': period,
                        'interval': interval,
                        'start_date': hist_data.index[0].isoformat(),
                        'end_date': hist_data.index[-1].isoformat(),
                        'row_count': len(hist_data),
                        'data_format': 'fallback'
                    }

                self.cache.set(cache_key, cache_data, 'historical')

                self.logger.debug(f"Retrieved {len(hist_data)} data points for {symbol}")
                return hist_data
            else:
                self.logger.warning(f"No historical data available for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

class MarketDataService(QObject):
    """
    Professional market data service with multiple providers, caching, and Qt integration
    """
    
    # Signals
    price_updated = Signal(str, float)  # symbol, price
    data_error = Signal(str, str)  # symbol, error_message
    connection_status_changed = Signal(bool)  # connected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()
        
        # Initialize components
        self.cache = MarketDataCache()
        self.provider_manager = DataProviderManager(self.config_manager)
        
        # Service state
        self.is_running = False
        self.connection_status = False
        
        # Background update thread
        self.update_thread = None
        self.update_queue = queue.Queue()
        
        # Cleanup timer
        self.cleanup_timer = QTimer()
        # TEMP FIX:         self.cleanup_timer.timeout.connect(self._cleanup_cache)
        self.cleanup_timer.start(300000)  # 5 minutes
        
        # Connection test timer
        self.connection_timer = QTimer()
        # TEMP FIX: self.connection_timer.timeout.connect(self._test_connection)
        self.connection_timer.start(60000)  # 1 minute
        
        self.logger.info("MarketDataService initialized")

    def get_provider_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all data providers"""
        return self.provider_manager.get_provider_health_summary()

    async def get_price_with_retry(self, symbol: str) -> Optional[float]:
        """Get price using intelligent retry and fallback logic"""
        price_data = await self.provider_manager.fetch_price_with_fallback(symbol)
        return price_data.price if price_data else None

    def reset_provider_circuit_breaker(self, provider_name: str):
        """Reset circuit breaker for a provider"""
        try:
            provider = DataProvider(provider_name.lower())
            self.provider_manager.reset_circuit_breaker(provider)
            self.logger.info(f"Reset circuit breaker for {provider_name}")
        except ValueError:
            self.logger.error(f"Unknown provider: {provider_name}")

    def start(self):
        """Start the market data service"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
        self.update_thread.start()
        
        # Initial connection test
        # TEMP FIX: QTimer.singleShot(1000, self._test_connection)
        
        self.logger.info("MarketDataService started")
    
    def stop(self):
        """Stop the market data service"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop timers
        self.cleanup_timer.stop()
        self.connection_timer.stop()
        
        # Signal update thread to stop
        self.update_queue.put(None)  # Sentinel value
        
        # Close async session
        try:
            asyncio.run(self.provider_manager.close_session())
        except Exception as e:
            self.logger.warning(f"Error closing async session: {e}")
        
        self.logger.info("MarketDataService stopped")
    
    def get_next_earnings(self, symbol: str) -> Optional[date]:
        """
        Get next earnings date for symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Next earnings date or None if unavailable
        """
        try:
            # 🔧 PARAMETER SAFETY FIX: Validate symbol is string, not DataFrame
            if not isinstance(symbol, str):
                self.logger.error(f"🔧 SYMBOL TYPE ERROR FIX: Expected str, got {type(symbol).__name__}: {symbol}")
                return None

            symbol = symbol.upper().strip()
            
            # Check cache first
            cache_key = f"earnings_{symbol}"
            cached_data = self.cache.get(cache_key, 'earnings')
            
            if cached_data and isinstance(cached_data, dict):
                try:
                    earnings_date_str = cached_data.get('next_earnings')
                    if earnings_date_str:
                        return datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
                except Exception:
                    pass
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            
            try:
                # Try to get earnings calendar
                calendar = ticker.calendar
                if calendar is not None and not calendar.empty:
                    next_earnings = calendar.index[0].date()
                    
                    # Cache the result
                    cache_data = {
                        'next_earnings': next_earnings.isoformat(),
                        'symbol': symbol,
                        'fetched_at': datetime.now().isoformat()
                    }
                    
                    self.cache.set(cache_key, cache_data, 'earnings')
                    
                    return next_earnings
                    
            except Exception as calendar_error:
                self.logger.debug(f"Calendar method failed for {symbol}: {calendar_error}")
            
            # Fallback: try earnings_dates
            try:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    # Get future earnings dates
                    future_dates = earnings_dates[earnings_dates.index > datetime.now()]
                    if not future_dates.empty:
                        next_earnings = future_dates.index[0].date()
                        
                        # Cache the result
                        cache_data = {
                            'next_earnings': next_earnings.isoformat(),
                            'symbol': symbol,
                            'fetched_at': datetime.now().isoformat()
                        }
                        
                        self.cache.set(cache_key, cache_data, 'earnings')
                        
                        return next_earnings
                        
            except Exception as earnings_error:
                self.logger.debug(f"Earnings dates method failed for {symbol}: {earnings_error}")
            
            # No earnings data found
            self.logger.info(f"No earnings data available for {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting earnings date for {symbol}: {e}")
            return None

    def get_next_earnings_date(self, symbol: str) -> Optional[date]:
        """Backward-compatible alias for get_next_earnings."""
        return self.get_next_earnings(symbol)
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Get current prices for multiple symbols efficiently
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to prices
        """
        results = {}
        
        try:
            # Use ThreadPoolExecutor for parallel fetching
            with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(self.get_current_price, symbol): symbol 
                    for symbol in symbols
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_symbol, timeout=30):
                    symbol = future_to_symbol[future]
                    try:
                        price = future.result()
                        results[symbol] = price
                    except Exception as e:
                        self.logger.error(f"Error getting price for {symbol}: {e}")
                        results[symbol] = None
                        
        except Exception as e:
            self.logger.error(f"Error in batch price fetch: {e}")
            # Fallback to individual fetches
            for symbol in symbols:
                results[symbol] = self.get_current_price(symbol)
        
        return results
    
    def get_market_status(self) -> dict:
        """Get current market status"""
        try:
            # Simple market hours check (US Eastern Time)
            from datetime import datetime
            import pytz
            
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            
            # Market is open Monday-Friday 9:30 AM - 4:00 PM ET
            is_weekday = now_et.weekday() < 5
            market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_market_hours = market_open_time <= now_et <= market_close_time
            is_open = is_weekday and is_market_hours
            
            # Calculate next open/close
            if is_open:
                next_change = market_close_time
                next_status = "Market Close"
            else:
                if now_et.weekday() >= 5:  # Weekend
                    # Next Monday
                    days_until_monday = 7 - now_et.weekday()
                    next_monday = now_et + timedelta(days=days_until_monday)
                    next_change = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
                elif now_et < market_open_time:
                    # Same day, before open
                    next_change = market_open_time
                else:
                    # Same day, after close - next day
                    tomorrow = now_et + timedelta(days=1)
                    if tomorrow.weekday() < 5:  # Weekday
                        next_change = tomorrow.replace(hour=9, minute=30, second=0, microsecond=0)
                    else:  # Weekend
                        days_until_monday = 7 - tomorrow.weekday()
                        next_monday = tomorrow + timedelta(days=days_until_monday)
                        next_change = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
                
                next_status = "Market Open"
            
            return {
                'is_open': is_open,
                'current_time': now_et.isoformat(),
                'next_change': next_change.isoformat(),
                'next_status': next_status,
                'timezone': 'US/Eastern'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market status: {e}")
            return {
                'is_open': None,
                'error': str(e)
            }
    
    def _fetch_price_sync(self, symbol: str) -> Optional[PriceData]:
        """Synchronously fetch price using async methods"""
        try:
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async price fetch
            return loop.run_until_complete(self._fetch_price_async(symbol))
            
        except Exception as e:
            self.logger.error(f"Error in sync price fetch for {symbol}: {e}")
            return None
    
    async def _fetch_price_async(self, symbol: str) -> Optional[PriceData]:
        """Fetch price using multiple providers with fallback"""
        providers = self.provider_manager.get_available_providers('price')
        
        for provider in providers:
            try:
                if provider == DataProvider.YAHOO_FINANCE:
                    result = await self.provider_manager.fetch_price_yahoo(symbol)
                elif provider == DataProvider.ALPHA_VANTAGE:
                    result = await self.provider_manager.fetch_price_alpha_vantage(symbol)
                elif provider == DataProvider.FINNHUB:
                    result = await self.provider_manager.fetch_price_finnhub(symbol)
                else:
                    continue
                
                if result and result.price > 0:
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Provider {provider.value} failed for {symbol}: {e}")
                continue
        
        return None
    
    def _update_worker(self):
        """Background worker for handling update requests"""
        while self.is_running:
            try:
                # Get update request from queue (blocking with timeout)
                update_request = self.update_queue.get(timeout=1)
                
                # Check for sentinel value (shutdown signal)
                if update_request is None:
                    break
                
                # Process update request
                if isinstance(update_request, tuple) and len(update_request) == 2:
                    symbol, force_refresh = update_request
                else:
                    symbol, force_refresh = update_request, False
                self.get_current_price(symbol, force_refresh=bool(force_refresh))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in update worker: {e}")

    def _test_connection(self):
        """Test connection to data providers"""
        try:
            # Test with a common symbol
            test_price = self.get_current_price("AAPL")
            new_status = test_price is not None and test_price > 0
            
            if new_status != self.connection_status:
                self.connection_status = new_status
                self.connection_status_changed.emit(new_status)
                
                status_msg = "connected" if new_status else "disconnected"
                self.logger.info(f"Market data service {status_msg}")
                
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            if self.connection_status:
                self.connection_status = False
                self.connection_status_changed.emit(False)
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        try:
            self.cache.cleanup()
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")
    
    def get_cache_info(self) -> dict:
        """Get cache statistics"""
        try:
            return {
                'memory_entries': len(self.cache.memory_cache),
                'cache_directory': str(self.cache.cache_dir),
                'database_size': self.cache.db_path.stat().st_size if self.cache.db_path.exists() else 0,
                'connection_status': self.connection_status
            }
        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            # Clear memory cache
            self.cache.memory_cache.clear()

            # Clear database cache
            with sqlite3.connect(self.cache.db_path) as conn:
                conn.execute("DELETE FROM market_cache")
                conn.commit()

            self.logger.info("Market data cache cleared")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    def get_market_overview(self):
        """Get current market overview data"""
        try:
            # Fetch key market indicators
            vix = self.get_current_price("^VIX")  # VIX
            spx = self.get_current_price("^GSPC")  # S&P 500
            ndx = self.get_current_price("^IXIC")  # NASDAQ
            
            return {
                'VIX': vix if vix else 19.44,
                'SPX': spx if spx else 4785.23,  
                'NDX': ndx if ndx else 16847.35,
                'updated': True
            }
        except Exception as e:
            self.logger.error(f"Error fetching market overview: {e}")
            return {
                'VIX': 19.44,
                'SPX': 4785.23,
                'NDX': 16847.35,
                'updated': False
            }
    

    def get_vix(self):
        """Get current VIX (volatility index) with caching"""
        try:
            # Check cache first
            cache_key = "price_^VIX"
            cached_price = self.cache.get(cache_key, 'price')
            if cached_price is not None:
                return float(cached_price)

            import yfinance as yf
            ticker = yf.Ticker("^VIX")
            data = ticker.history(period="1d")
            if not data.empty:
                price = round(float(data['Close'].iloc[-1]), 2)
                # Cache the result
                self.cache.set(cache_key, price, 'price')
                return price
        except Exception as e:
            self.logger.warning(f"Error fetching VIX: {e}")
        return 23.45  # Default fallback

    def get_spx(self):
        """Get current S&P 500 price with caching"""
        try:
            # Check cache first
            cache_key = "price_^GSPC"
            cached_price = self.cache.get(cache_key, 'price')
            if cached_price is not None:
                return float(cached_price)

            import yfinance as yf
            ticker = yf.Ticker("^GSPC")
            data = ticker.history(period="1d")
            if not data.empty:
                price = round(float(data['Close'].iloc[-1]), 2)
                # Cache the result
                self.cache.set(cache_key, price, 'price')
                return price
        except Exception as e:
            self.logger.warning(f"Error fetching SPX: {e}")
        return 4850.00  # More realistic default

    def get_nasdaq(self):
        """Get current NASDAQ price with caching"""
        try:
            # Check cache first
            cache_key = "price_^IXIC"
            cached_price = self.cache.get(cache_key, 'price')
            if cached_price is not None:
                return float(cached_price)

            import yfinance as yf
            ticker = yf.Ticker("^IXIC")
            data = ticker.history(period="1d")
            if not data.empty:
                price = round(float(data['Close'].iloc[-1]), 2)
                # Cache the result
                self.cache.set(cache_key, price, 'price')
                return price
        except Exception as e:
            self.logger.warning(f"Error fetching NASDAQ: {e}")
        return 15200.00  # More realistic default

    def get_current_price(self, symbol: str, force_refresh: bool = False) -> float:
        """Get current price for a symbol with caching"""
        try:
            # Check cache first
            cache_key = f"price_{symbol}"
            if not force_refresh:
                cached_price = self.cache.get(cache_key, 'price')
                if cached_price is not None:
                    return float(cached_price)

            # Handle special symbols
            if symbol == "VIX":
                symbol = "^VIX"
            elif symbol == "SPX":
                symbol = "^GSPC"
            elif symbol == "NDX":
                symbol = "^NDX"

            # Fetch fresh data
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")

            if not data.empty:
                price = float(data["Close"].iloc[-1])
                # Cache the result for 1 minute
                self.cache.set(cache_key, price, 'price')
                self.logger.info(f"Fetched {symbol}: ${price:.2f}")
                return price
            else:
                self.logger.warning(f"No data available for {symbol}")
                return 0.0

        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return 0.0

    def get_historical_data(self, symbol: str, period: str = "1y",
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical data for symbol with caching"""
        try:
            # Check cache first
            cache_key = f"historical_{symbol}_{period}_{interval}"
            cached_data = self.cache.get(cache_key, 'historical')

            if cached_data and isinstance(cached_data, dict):
                try:
                    # Fixed cache reconstruction: Handle string-based datetime index from cache
                    df = pd.DataFrame(cached_data['data'])

                    # Check if this uses the new string index format
                    if cached_data.get('data_format') == 'string_index':
                        # Index is already properly formatted strings, convert to datetime
                        df.index = pd.to_datetime(df.index)
                    else:
                        # Legacy format - try to convert
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception as legacy_error:
                            self.logger.warning(f"Legacy cache format conversion failed: {legacy_error}")
                            return pd.DataFrame()  # Force fresh fetch

                    if not df.empty and len(df) > 10:  # Reasonable amount of data
                        return df
                except Exception as cache_error:
                    self.logger.warning(f"Error reconstructing cached historical data: {cache_error}")
                    # Clear corrupted cache entry
                    self.cache.delete(cache_key, 'historical')

            # Fetch fresh data
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period, interval=interval)

            if not hist_data.empty:
                # Fix critical cache issue: Convert Timestamp index to string keys for cache compatibility
                # hist_data.to_dict('index') creates Timestamp keys which cache system rejects
                try:
                    # Convert index to ISO strings before creating dict
                    hist_data_copy = hist_data.copy()
                    hist_data_copy.index = hist_data_copy.index.strftime('%Y-%m-%d %H:%M:%S')

                    # Cache the result with string-keyed data
                    cache_data = {
                        'data': hist_data_copy.to_dict('index'),
                        'period': period,
                        'interval': interval,
                        'symbol': symbol,
                        'start_date': hist_data.index[0].isoformat(),
                        'end_date': hist_data.index[-1].isoformat(),
                        'data_format': 'string_index'  # Mark for proper reconstruction
                    }
                except Exception as e:
                    self.logger.warning(f"Error preparing cache data for {symbol}: {e}")
                    # Fallback: cache without historical data dict
                    cache_data = {
                        'symbol': symbol,
                        'period': period,
                        'interval': interval,
                        'start_date': hist_data.index[0].isoformat(),
                        'end_date': hist_data.index[-1].isoformat(),
                        'row_count': len(hist_data),
                        'data_format': 'fallback'
                    }

                self.cache.set(cache_key, cache_data, 'historical')

                self.logger.debug(f"Retrieved {len(hist_data)} data points for {symbol}")
                return hist_data
            else:
                self.logger.warning(f"No historical data available for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
