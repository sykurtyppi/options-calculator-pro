"""
Service interfaces for Options Calculator Pro.

Defines abstract base classes for all core services to enable dependency
injection, improve testability, and reduce coupling between components.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
from datetime import date
from decimal import Decimal
import pandas as pd

from models.option_data import OptionData
from models.analysis_result import AnalysisResult
from models.trade_data import TradeData
from models.greeks import GreeksResult


class IMarketDataService(ABC):
    """Interface for market data providers"""
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price for symbol"""
        pass
    
    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: date) -> Optional[Dict[str, Any]]:
        """Get options chain for symbol and expiry"""
        pass
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        pass
    
    @abstractmethod
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """Get real-time price with fallback"""
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        pass


class IVolatilityService(ABC):
    """Interface for volatility calculations"""
    
    @abstractmethod
    def calculate_historical_volatility(
        self, 
        symbol: str, 
        period: int = 30
    ) -> Optional[float]:
        """Calculate historical volatility"""
        pass
    
    @abstractmethod
    def calculate_implied_volatility(
        self,
        option: OptionData,
        market_price: float
    ) -> Optional[float]:
        """Calculate implied volatility from market price"""
        pass
    
    @abstractmethod
    def get_volatility_surface(
        self,
        symbol: str,
        expiry: date
    ) -> Optional[Dict[str, float]]:
        """Get volatility surface for strikes"""
        pass
    
    @abstractmethod
    def calculate_term_structure(
        self,
        symbol: str
    ) -> Optional[Dict[str, float]]:
        """Calculate volatility term structure"""
        pass


class IOptionsService(ABC):
    """Interface for options pricing and analysis"""
    
    @abstractmethod
    def calculate_option_price(
        self,
        option: OptionData,
        spot_price: float,
        volatility: float,
        risk_free_rate: float = 0.05
    ) -> Optional[float]:
        """Calculate theoretical option price"""
        pass
    
    @abstractmethod
    def calculate_greeks(
        self,
        option: OptionData,
        spot_price: float,
        volatility: float,
        risk_free_rate: float = 0.05
    ) -> Optional[GreeksResult]:
        """Calculate option Greeks"""
        pass
    
    @abstractmethod
    def analyze_calendar_spread(
        self,
        symbol: str,
        strike: float,
        short_expiry: date,
        long_expiry: date
    ) -> Optional[AnalysisResult]:
        """Analyze calendar spread opportunity"""
        pass
    
    @abstractmethod
    def calculate_profit_loss(
        self,
        trades: List[TradeData],
        spot_prices: List[float]
    ) -> Dict[str, Any]:
        """Calculate P&L for trade portfolio"""
        pass


class IMLService(ABC):
    """Interface for machine learning predictions"""
    
    @abstractmethod
    def predict_price_movement(
        self,
        symbol: str,
        days_ahead: int = 30
    ) -> Optional[Dict[str, float]]:
        """Predict stock price movement probability"""
        pass
    
    @abstractmethod
    def predict_volatility(
        self,
        symbol: str,
        days_ahead: int = 30
    ) -> Optional[float]:
        """Predict future volatility"""
        pass
    
    @abstractmethod
    def analyze_earnings_impact(
        self,
        symbol: str,
        earnings_date: date
    ) -> Optional[Dict[str, Any]]:
        """Analyze earnings impact on options"""
        pass
    
    @abstractmethod
    def train_model(
        self,
        symbol: str,
        retrain: bool = False
    ) -> bool:
        """Train ML model for symbol"""
        pass


class IAsyncAPIService(ABC):
    """Interface for async market data operations"""
    
    @abstractmethod
    async def fetch_multiple_prices(
        self,
        symbols: List[str]
    ) -> Dict[str, Optional[float]]:
        """Fetch multiple stock prices concurrently"""
        pass
    
    @abstractmethod
    async def fetch_option_chains(
        self,
        symbols: List[str],
        expiries: List[date]
    ) -> Dict[str, Any]:
        """Fetch multiple option chains concurrently"""
        pass
    
    @abstractmethod
    async def stream_prices(
        self,
        symbols: List[str],
        callback: Callable[[str, float], None]
    ) -> None:
        """Stream real-time price updates"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close async connections"""
        pass


class IConfigService(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        pass
    
    @abstractmethod
    def save(self) -> bool:
        """Save configuration to storage"""
        pass
    
    @abstractmethod
    def load(self) -> bool:
        """Load configuration from storage"""
        pass
    
    @abstractmethod
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key securely"""
        pass
    
    @abstractmethod
    def set_api_key(self, service: str, api_key: str) -> bool:
        """Set API key securely"""
        pass


class ICacheService(ABC):
    """Interface for caching operations"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set cached value with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached value"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


# Exception hierarchy for proper error handling
class OptionsCalculatorError(Exception):
    """Base exception for options calculator"""
    pass


class MarketDataError(OptionsCalculatorError):
    """Market data related errors"""
    pass


class InvalidSymbolError(MarketDataError):
    """Invalid stock symbol"""
    pass


class NetworkError(MarketDataError):
    """Network connectivity issues"""
    pass


class APILimitError(MarketDataError):
    """API rate limit exceeded"""
    pass


class CalculationError(OptionsCalculatorError):
    """Options calculation errors"""
    pass


class InvalidParameterError(CalculationError):
    """Invalid calculation parameters"""
    pass


class VolatilityError(CalculationError):
    """Volatility calculation errors"""
    pass


class ConfigurationError(OptionsCalculatorError):
    """Configuration related errors"""
    pass


class SecurityError(OptionsCalculatorError):
    """Security and credential errors"""
    pass