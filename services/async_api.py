"""
Asynchronous API client for high-performance market data operations.

This module provides production-ready async HTTP clients for market data providers
with connection pooling, rate limiting, circuit breakers, and robust error handling.
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import ssl
from urllib.parse import urlencode
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import backoff
from contextlib import asynccontextmanager

from ..models.option_data import OptionContract, OptionChain, OptionType
from ..utils.logger import get_logger

class DataProvider(Enum):
    """Supported market data providers"""
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage" 
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    IEX_CLOUD = "iex_cloud"
    TRADIER = "tradier"

class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class RateLimitConfig:
    """Rate limiting configuration per provider"""
    requests_per_second: float = 5.0
    requests_per_minute: int = 300
    requests_per_hour: int = 1000
    burst_limit: int = 10
    
@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    expected_exception: tuple = (aiohttp.ClientError, asyncio.TimeoutError)

@dataclass
class RetryConfig:
    """Retry configuration for failed requests"""
    max_tries: int = 3
    backoff_factor: float = 2.0
    jitter: bool = True
    timeout: float = 30.0

@dataclass
class APIRequest:
    """API request with metadata"""
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    data: Optional[Union[str, bytes]] = None
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: float = 30.0
    provider: DataProvider = DataProvider.YAHOO
    symbol: Optional[str] = None
    request_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass  
class APIResponse:
    """API response with metadata"""
    status: int
    data: Any
    headers: Dict[str, str]
    request_time: float
    provider: DataProvider
    symbol: Optional[str] = None
    request_id: Optional[str] = None
    error: Optional[str] = None
    from_cache: bool = False

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for API resilience"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.logger = get_logger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker half-open, testing recovery")
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker closed, service recovered")
            
            return result
            
        except self.config.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.error(f"Circuit breaker opened due to {self.failure_count} failures")
            
            raise

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a token for request"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on rate
            self.tokens = min(
                self.config.burst_limit,
                self.tokens + elapsed * self.config.requests_per_second
            )
            
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            
            return False
    
    async def wait_for_token(self):
        """Wait until a token is available"""
        while not await self.acquire():
            await asyncio.sleep(0.1)

class RequestQueue:
    """Priority queue for API requests"""
    
    def __init__(self, max_size: int = 1000):
        self.queues = {
            RequestPriority.CRITICAL: asyncio.Queue(),
            RequestPriority.HIGH: asyncio.Queue(),
            RequestPriority.NORMAL: asyncio.Queue(),
            RequestPriority.LOW: asyncio.Queue()
        }
        self.max_size = max_size
        self.total_size = 0
    
    async def put(self, request: APIRequest):
        """Add request to appropriate queue"""
        if self.total_size >= self.max_size:
            raise asyncio.QueueFull("Request queue is full")
        
        await self.queues[request.priority].put(request)
        self.total_size += 1
    
    async def get(self) -> APIRequest:
        """Get next highest priority request"""
        for priority in [RequestPriority.CRITICAL, RequestPriority.HIGH, 
                        RequestPriority.NORMAL, RequestPriority.LOW]:
            queue = self.queues[priority]
            try:
                request = queue.get_nowait()
                self.total_size -= 1
                return request
            except asyncio.QueueEmpty:
                continue
        
        # If all queues empty, wait for normal priority
        request = await self.queues[RequestPriority.NORMAL].get()
        self.total_size -= 1
        return request

class ResponseCache:
    """TTL-based response cache"""
    
    def __init__(self, max_size: int = 10000, default_ttl: float = 60.0):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = asyncio.Lock()
    
    def _make_key(self, request: APIRequest) -> str:
        """Create cache key from request"""
        key_parts = [request.url, request.method]
        if request.params:
            key_parts.append(urlencode(sorted(request.params.items())))
        return "|".join(key_parts)
    
    async def get(self, request: APIRequest) -> Optional[Any]:
        """Get cached response if available and fresh"""
        key = self._make_key(request)
        
        async with self.lock:
            if key in self.cache:
                data, expires_at = self.cache[key]
                if time.time() < expires_at:
                    return data
                else:
                    del self.cache[key]
        
        return None
    
    async def set(self, request: APIRequest, data: Any, ttl: Optional[float] = None):
        """Cache response data"""
        key = self._make_key(request)
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        async with self.lock:
            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Simple FIFO eviction
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = (data, expires_at)

class AsyncMarketDataClient:
    """
    High-performance async market data client.
    
    Features:
    - Connection pooling and session reuse
    - Rate limiting per provider
    - Circuit breaker for resilience
    - Request prioritization and queuing
    - Response caching with TTL
    - Automatic retries with exponential backoff
    - Comprehensive error handling and logging
    """
    
    def __init__(
        self,
        provider_configs: Dict[DataProvider, Dict[str, Any]] = None,
        rate_limit_configs: Dict[DataProvider, RateLimitConfig] = None,
        circuit_breaker_config: CircuitBreakerConfig = None,
        cache_config: Dict[str, Any] = None,
        max_concurrent_requests: int = 50
    ):
        self.provider_configs = provider_configs or {}
        self.rate_limiters = {}
        self.circuit_breakers = {}
        self.sessions: Dict[DataProvider, aiohttp.ClientSession] = {}
        
        # Initialize rate limiters
        default_rate_limit = RateLimitConfig()
        for provider in DataProvider:
            config = rate_limit_configs.get(provider, default_rate_limit) if rate_limit_configs else default_rate_limit
            self.rate_limiters[provider] = RateLimiter(config)
        
        # Initialize circuit breakers  
        default_cb_config = circuit_breaker_config or CircuitBreakerConfig()
        for provider in DataProvider:
            self.circuit_breakers[provider] = CircuitBreaker(default_cb_config)
        
        # Request queue and cache
        self.request_queue = RequestQueue()
        cache_config = cache_config or {}
        self.cache = ResponseCache(**cache_config)
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound parsing
        
        self.logger = get_logger(__name__)
        self._running = False
        self._worker_tasks = []
    
    async def start(self):
        """Start the async client and worker tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Create HTTP sessions for each provider
        for provider in DataProvider:
            self.sessions[provider] = await self._create_session(provider)
        
        # Start request processing workers
        num_workers = min(10, asyncio.cpu_count() * 2)
        for i in range(num_workers):
            task = asyncio.create_task(self._request_worker(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        self.logger.info(f"AsyncMarketDataClient started with {num_workers} workers")
    
    async def stop(self):
        """Stop the client and clean up resources"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Close HTTP sessions
        for session in self.sessions.values():
            await session.close()
        
        self.sessions.clear()
        self.executor.shutdown(wait=True)
        
        self.logger.info("AsyncMarketDataClient stopped")
    
    async def _create_session(self, provider: DataProvider) -> aiohttp.ClientSession:
        """Create HTTP session with optimal settings"""
        # SSL context
        ssl_context = ssl.create_default_context()
        
        # Connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            ssl_context=ssl_context
        )
        
        # Request timeout
        timeout = aiohttp.ClientTimeout(
            total=30,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=20  # Socket read timeout
        )
        
        # Default headers
        headers = {
            'User-Agent': 'OptionsCalculatorPro/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        }
        
        # Provider-specific headers
        if provider in self.provider_configs:
            config = self.provider_configs[provider]
            if 'api_key' in config:
                if provider == DataProvider.ALPHA_VANTAGE:
                    headers['X-API-Key'] = config['api_key']
                elif provider == DataProvider.FINNHUB:
                    headers['X-Finnhub-Token'] = config['api_key']
                elif provider == DataProvider.POLYGON:
                    headers['Authorization'] = f"Bearer {config['api_key']}"
        
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers,
            raise_for_status=False  # Handle status codes manually
        )
    
    async def _request_worker(self, worker_name: str):
        """Background worker to process queued requests"""
        while self._running:
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                async with self.semaphore:
                    await self._execute_request(request)
                    
            except asyncio.TimeoutError:
                # No requests available, continue
                continue
            except Exception as e:
                self.logger.error(f"Request worker {worker_name} error: {e}")
    
    async def _execute_request(self, request: APIRequest) -> APIResponse:
        """Execute a single API request with all protections"""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_data = await self.cache.get(request)
            if cached_data is not None:
                return APIResponse(
                    status=200,
                    data=cached_data,
                    headers={},
                    request_time=time.time() - start_time,
                    provider=request.provider,
                    symbol=request.symbol,
                    request_id=request.request_id,
                    from_cache=True
                )
            
            # Apply rate limiting
            await self.rate_limiters[request.provider].wait_for_token()
            
            # Execute with circuit breaker
            response = await self.circuit_breakers[request.provider].call(
                self._make_http_request,
                request
            )
            
            # Cache successful responses
            if response.status == 200:
                await self.cache.set(request, response.data)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request execution failed: {e}")
            return APIResponse(
                status=500,
                data=None,
                headers={},
                request_time=time.time() - start_time,
                provider=request.provider,
                symbol=request.symbol,
                request_id=request.request_id,
                error=str(e)
            )
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        jitter=backoff.full_jitter
    )
    async def _make_http_request(self, request: APIRequest) -> APIResponse:
        """Make HTTP request with retries"""
        start_time = time.time()
        session = self.sessions[request.provider]
        
        try:
            async with session.request(
                method=request.method,
                url=request.url,
                headers=request.headers,
                params=request.params,
                data=request.data,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                
                response_data = await response.json() if response.content_type.startswith('application/json') else await response.text()
                
                return APIResponse(
                    status=response.status,
                    data=response_data,
                    headers=dict(response.headers),
                    request_time=time.time() - start_time,
                    provider=request.provider,
                    symbol=request.symbol,
                    request_id=request.request_id
                )
                
        except Exception as e:
            self.logger.error(f"HTTP request failed: {e}")
            raise
    
    async def get_stock_price(
        self,
        symbol: str,
        provider: DataProvider = DataProvider.YAHOO,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> Optional[float]:
        """Get current stock price"""
        try:
            url = self._build_price_url(symbol, provider)
            request = APIRequest(
                url=url,
                provider=provider,
                symbol=symbol,
                priority=priority,
                request_id=f"price-{symbol}-{int(time.time())}"
            )
            
            response = await self._execute_request(request)
            
            if response.status == 200:
                return self._parse_price_response(response.data, provider)
            else:
                self.logger.error(f"Price request failed: {response.status} - {response.error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Get stock price failed for {symbol}: {e}")
            return None
    
    async def get_option_chain(
        self,
        symbol: str,
        expiration_date: str = None,
        provider: DataProvider = DataProvider.YAHOO,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> Optional[OptionChain]:
        """Get options chain for a symbol"""
        try:
            url = self._build_options_url(symbol, expiration_date, provider)
            request = APIRequest(
                url=url,
                provider=provider,
                symbol=symbol,
                priority=priority,
                request_id=f"options-{symbol}-{int(time.time())}"
            )
            
            response = await self._execute_request(request)
            
            if response.status == 200:
                return await self._parse_options_response(response.data, provider, symbol)
            else:
                self.logger.error(f"Options request failed: {response.status} - {response.error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Get option chain failed for {symbol}: {e}")
            return None
    
    async def get_multiple_prices(
        self,
        symbols: List[str],
        provider: DataProvider = DataProvider.YAHOO,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> Dict[str, Optional[float]]:
        """Get prices for multiple symbols concurrently"""
        tasks = []
        for symbol in symbols:
            task = self.get_stock_price(symbol, provider, priority)
            tasks.append((symbol, task))
        
        results = {}
        completed = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (symbol, _), result in zip(tasks, completed):
            if isinstance(result, Exception):
                self.logger.error(f"Price fetch failed for {symbol}: {result}")
                results[symbol] = None
            else:
                results[symbol] = result
        
        return results
    
    def _build_price_url(self, symbol: str, provider: DataProvider) -> str:
        """Build URL for price request"""
        if provider == DataProvider.YAHOO:
            return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        elif provider == DataProvider.ALPHA_VANTAGE:
            api_key = self.provider_configs.get(provider, {}).get('api_key', '')
            return f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        elif provider == DataProvider.FINNHUB:
            return f"https://finnhub.io/api/v1/quote?symbol={symbol}"
        else:
            raise ValueError(f"Price URL not implemented for provider: {provider}")
    
    def _build_options_url(self, symbol: str, expiration: str, provider: DataProvider) -> str:
        """Build URL for options request"""
        if provider == DataProvider.YAHOO:
            url = f"https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
            if expiration:
                url += f"?date={expiration}"
            return url
        elif provider == DataProvider.TRADIER:
            api_key = self.provider_configs.get(provider, {}).get('api_key', '')
            url = f"https://sandbox.tradier.com/v1/markets/options/chains?symbol={symbol}"
            if expiration:
                url += f"&expiration={expiration}"
            return url
        else:
            raise ValueError(f"Options URL not implemented for provider: {provider}")
    
    def _parse_price_response(self, data: Any, provider: DataProvider) -> Optional[float]:
        """Parse price from provider response"""
        try:
            if provider == DataProvider.YAHOO:
                return float(data['chart']['result'][0]['meta']['regularMarketPrice'])
            elif provider == DataProvider.ALPHA_VANTAGE:
                return float(data['Global Quote']['05. price'])
            elif provider == DataProvider.FINNHUB:
                return float(data['c'])  # Current price
            else:
                self.logger.warning(f"Price parsing not implemented for provider: {provider}")
                return None
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to parse price response: {e}")
            return None
    
    async def _parse_options_response(
        self, 
        data: Any, 
        provider: DataProvider, 
        symbol: str
    ) -> Optional[OptionChain]:
        """Parse options chain from provider response (CPU-intensive, use thread pool)"""
        try:
            # Run parsing in thread pool to avoid blocking event loop
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._parse_options_sync,
                data, provider, symbol
            )
        except Exception as e:
            self.logger.error(f"Failed to parse options response: {e}")
            return None
    
    def _parse_options_sync(self, data: Any, provider: DataProvider, symbol: str) -> Optional[OptionChain]:
        """Synchronous options parsing"""
        if provider == DataProvider.YAHOO:
            return self._parse_yahoo_options(data, symbol)
        elif provider == DataProvider.TRADIER:
            return self._parse_tradier_options(data, symbol)
        else:
            self.logger.warning(f"Options parsing not implemented for provider: {provider}")
            return None
    
    def _parse_yahoo_options(self, data: Any, symbol: str) -> Optional[OptionChain]:
        """Parse Yahoo Finance options response"""
        try:
            result = data['optionChain']['result'][0]
            quote = result.get('quote', {})
            underlying_price = quote.get('regularMarketPrice', 0.0)
            
            options = result.get('options', [])
            if not options:
                return None
            
            option_data = options[0]
            expiration_timestamp = option_data.get('expirationDate')
            expiration_date = datetime.fromtimestamp(expiration_timestamp).date()
            
            calls = []
            puts = []
            
            # Parse calls
            for call_data in option_data.get('calls', []):
                calls.append(OptionContract(
                    symbol=call_data.get('contractSymbol', ''),
                    strike=float(call_data.get('strike', 0)),
                    expiration=expiration_date,
                    option_type=OptionType.CALL,
                    bid=float(call_data.get('bid', 0)),
                    ask=float(call_data.get('ask', 0)),
                    last=float(call_data.get('lastPrice', 0)),
                    volume=int(call_data.get('volume', 0)),
                    open_interest=int(call_data.get('openInterest', 0)),
                    implied_volatility=float(call_data.get('impliedVolatility', 0))
                ))
            
            # Parse puts
            for put_data in option_data.get('puts', []):
                puts.append(OptionContract(
                    symbol=put_data.get('contractSymbol', ''),
                    strike=float(put_data.get('strike', 0)),
                    expiration=expiration_date,
                    option_type=OptionType.PUT,
                    bid=float(put_data.get('bid', 0)),
                    ask=float(put_data.get('ask', 0)),
                    last=float(put_data.get('lastPrice', 0)),
                    volume=int(put_data.get('volume', 0)),
                    open_interest=int(put_data.get('openInterest', 0)),
                    implied_volatility=float(put_data.get('impliedVolatility', 0))
                ))
            
            return OptionChain(
                underlying_symbol=symbol,
                expiration=expiration_date,
                underlying_price=underlying_price,
                calls=calls,
                puts=puts
            )
            
        except Exception as e:
            self.logger.error(f"Yahoo options parsing failed: {e}")
            return None
    
    def _parse_tradier_options(self, data: Any, symbol: str) -> Optional[OptionChain]:
        """Parse Tradier options response"""
        # Implementation would go here
        self.logger.warning("Tradier options parsing not yet implemented")
        return None
    
    @asynccontextmanager
    async def session_scope(self):
        """Context manager for client lifecycle"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

# Convenience functions for common operations

async def fetch_current_price(
    symbol: str,
    provider: DataProvider = DataProvider.YAHOO,
    provider_config: Dict[str, Any] = None
) -> Optional[float]:
    """Quick function to fetch current price"""
    provider_configs = {provider: provider_config} if provider_config else {}
    
    async with AsyncMarketDataClient(provider_configs=provider_configs).session_scope() as client:
        return await client.get_stock_price(symbol, provider)

async def fetch_option_chain(
    symbol: str,
    expiration: str = None,
    provider: DataProvider = DataProvider.YAHOO,
    provider_config: Dict[str, Any] = None
) -> Optional[OptionChain]:
    """Quick function to fetch option chain"""
    provider_configs = {provider: provider_config} if provider_config else {}
    
    async with AsyncMarketDataClient(provider_configs=provider_configs).session_scope() as client:
        return await client.get_option_chain(symbol, expiration, provider)

async def fetch_multiple_prices(
    symbols: List[str],
    provider: DataProvider = DataProvider.YAHOO,
    provider_config: Dict[str, Any] = None
) -> Dict[str, Optional[float]]:
    """Quick function to fetch multiple prices concurrently"""
    provider_configs = {provider: provider_config} if provider_config else {}
    
    async with AsyncMarketDataClient(provider_configs=provider_configs).session_scope() as client:
        return await client.get_multiple_prices(symbols, provider)

# Example usage and testing functions

async def benchmark_performance():
    """Benchmark async client performance"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
    
    # Sequential timing
    start_time = time.time()
    sequential_results = {}
    for symbol in symbols:
        price = await fetch_current_price(symbol)
        sequential_results[symbol] = price
    sequential_time = time.time() - start_time
    
    # Concurrent timing
    start_time = time.time()
    concurrent_results = await fetch_multiple_prices(symbols)
    concurrent_time = time.time() - start_time
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s") 
    print(f"Speedup: {sequential_time/concurrent_time:.1f}x")
    
    return {
        'sequential_time': sequential_time,
        'concurrent_time': concurrent_time,
        'speedup': sequential_time / concurrent_time,
        'results': concurrent_results
    }

if __name__ == "__main__":
    # Example usage
    async def main():
        # Test basic functionality
        price = await fetch_current_price("AAPL")
        print(f"AAPL price: ${price}")
        
        # Test option chain
        options = await fetch_option_chain("AAPL")
        if options:
            print(f"AAPL options: {len(options.calls)} calls, {len(options.puts)} puts")
        
        # Benchmark performance
        await benchmark_performance()
    
    asyncio.run(main())