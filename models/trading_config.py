"""
Lightweight trading configuration for Options Calculator Pro.

Focuses on essential validation for calendar spread trading without
over-engineering. Designed for real-world trading conditions.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, Literal, Callable
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
import re


class Environment(str, Enum):
    """Runtime environments"""
    PRODUCTION = "production"
    BACKTESTING = "backtesting"
    DEVELOPMENT = "development"


class TradingConfig(BaseModel):
    """Essential trading configuration for calendar spreads"""
    
    # Environment settings
    environment: Environment = Environment.PRODUCTION
    as_of_date: Optional[date] = None  # For backtesting
    
    # Risk management (the critical stuff)
    max_risk_per_trade: float = Field(default=0.02, gt=0, le=0.1)  # 2% max
    account_size: Decimal = Field(..., gt=1000)
    
    # Calendar spread settings
    min_days_to_expiry: int = Field(default=1, ge=1)
    max_days_to_expiry: int = Field(default=45, le=365) 
    max_calendar_spread_cost: Decimal = Field(default=Decimal('5.00'), gt=0)
    
    # Data quality filters
    min_option_volume: int = Field(default=10, ge=0)
    max_bid_ask_spread_pct: float = Field(default=15.0, gt=0, le=50)
    
    class Config:
        """Pydantic config"""
        validate_assignment = True
        arbitrary_types_allowed = True


def validate_symbol(ticker: str, strict: bool = False) -> str:
    """
    Flexible ticker validation for real-world symbols.
    
    Args:
        ticker: Symbol to validate
        strict: If True, only allows basic A-Z format
    
    Returns:
        Cleaned ticker symbol
    """
    ticker = ticker.strip().upper()
    
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    
    if len(ticker) > 12:  # Reasonable max length
        raise ValueError(f"Ticker '{ticker}' too long (max 12 chars)")
    
    if strict:
        # Strict mode for basic validation
        if not re.match(r'^[A-Z]{1,8}$', ticker):
            raise ValueError(f"Ticker '{ticker}' contains invalid characters (strict mode)")
    else:
        # Flexible mode for real trading
        # Allow: Letters, numbers, dots, dashes (for real symbols like BRK.B, BF-B)
        if not re.match(r'^[A-Z0-9.-]+$', ticker):
            raise ValueError(f"Ticker '{ticker}' contains invalid characters")
        
        # Must start with a letter
        if not ticker[0].isalpha():
            raise ValueError(f"Ticker '{ticker}' must start with a letter")
    
    return ticker


def validate_expiration(
    exp_date: date, 
    as_of_date: Optional[date] = None,
    environment: Environment = Environment.PRODUCTION
) -> date:
    """
    Environment-aware expiration validation.
    
    Args:
        exp_date: Expiration date to validate
        as_of_date: Reference date (for backtesting)
        environment: Runtime environment
    """
    reference_date = as_of_date or date.today()
    
    # In backtesting/development, allow past dates
    if environment in [Environment.BACKTESTING, Environment.DEVELOPMENT]:
        # Just check it's not absurdly old/future
        min_date = date(2020, 1, 1)  # Reasonable lower bound
        max_date = date(2030, 12, 31)  # Reasonable upper bound
        
        if not (min_date <= exp_date <= max_date):
            raise ValueError(f"Expiration {exp_date} outside reasonable range {min_date} to {max_date}")
    else:
        # Production: must be future
        if exp_date <= reference_date:
            raise ValueError(f"Expiration {exp_date} must be after {reference_date}")
        
        # Not too far in future
        max_future = date(reference_date.year + 2, reference_date.month, reference_date.day)
        if exp_date > max_future:
            raise ValueError(f"Expiration {exp_date} too far in future (max 2 years)")
    
    return exp_date


def validate_strike(strike: Decimal, tolerance: Decimal = Decimal('0.01')) -> Decimal:
    """
    Flexible strike validation with tolerance.
    
    Args:
        strike: Strike price to validate
        tolerance: Tolerance for increment validation
    """
    if strike <= 0:
        raise ValueError(f"Strike {strike} must be positive")
    
    if strike > Decimal('10000'):
        raise ValueError(f"Strike {strike} unreasonably high (max $10,000)")
    
    # Note: Removed rigid increment validation per feedback
    # Real exchanges have varying increments and pilot programs
    # Let the market data determine valid strikes
    
    return strike


def validate_calendar_spread(
    symbol: str,
    strike: Decimal,
    short_expiration: date,
    long_expiration: date,
    config: TradingConfig
) -> Dict[str, Any]:
    """
    Validate calendar spread structure.
    
    Returns validated parameters with warnings rather than hard failures.
    """
    errors = []
    warnings = []
    
    try:
        # Validate components
        validated_symbol = validate_symbol(symbol)
        validated_strike = validate_strike(strike)
        
        # Use config's environment settings for date validation
        short_exp = validate_expiration(
            short_expiration, 
            config.as_of_date, 
            config.environment
        )
        long_exp = validate_expiration(
            long_expiration, 
            config.as_of_date, 
            config.environment
        )
        
        # Calendar spread specific validations
        if long_exp <= short_exp:
            errors.append("Long expiration must be after short expiration")
        
        time_spread = (long_exp - short_exp).days
        if time_spread < 7:
            warnings.append(f"Time spread only {time_spread} days (usually 14-45 days optimal)")
        elif time_spread > 90:
            warnings.append(f"Time spread {time_spread} days very wide (usually 14-45 days)")
        
        # Check days to expiry against config
        reference_date = config.as_of_date or date.today()
        short_dte = (short_exp - reference_date).days
        
        if short_dte < config.min_days_to_expiry:
            errors.append(f"Short expiry in {short_dte} days, below minimum {config.min_days_to_expiry}")
        elif short_dte > config.max_days_to_expiry:
            errors.append(f"Short expiry in {short_dte} days, above maximum {config.max_days_to_expiry}")
        
        result = {
            'valid': len(errors) == 0,
            'symbol': validated_symbol,
            'strike': validated_strike,
            'short_expiration': short_exp,
            'long_expiration': long_exp,
            'time_spread_days': time_spread,
            'short_dte': short_dte,
            'errors': errors,
            'warnings': warnings
        }
        
        return result
        
    except ValueError as e:
        errors.append(str(e))
        return {
            'valid': False,
            'errors': errors,
            'warnings': warnings
        }


def validate_position_size(
    calendar_cost: Decimal,
    config: TradingConfig,
    win_probability: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate safe position size for calendar spread.
    
    Uses Kelly Criterion if win probability provided, otherwise conservative sizing.
    """
    if calendar_cost <= 0:
        raise ValueError("Calendar cost must be positive")
    
    # Risk-based sizing (primary method)
    max_risk_dollars = config.account_size * Decimal(str(config.max_risk_per_trade))
    max_contracts_risk = int(max_risk_dollars / calendar_cost)
    
    # Kelly sizing (if we have win probability)
    kelly_contracts = max_contracts_risk  # Default to risk-based
    
    if win_probability is not None:
        if 0.5 <= win_probability <= 0.9:  # Reasonable range for calendar spreads
            # Conservative Kelly: assume avg win = 50% of cost, avg loss = 100% of cost
            avg_win = calendar_cost * Decimal('0.5')
            avg_loss = calendar_cost
            
            # Kelly fraction: (p * b - q) / b where b = avg_win/avg_loss
            odds = float(avg_win / avg_loss)  # 0.5
            kelly_fraction = (win_probability * odds - (1 - win_probability)) / odds
            
            # Use conservative 25% of Kelly
            conservative_kelly = max(0, kelly_fraction * 0.25)
            kelly_risk = config.account_size * Decimal(str(conservative_kelly))
            kelly_contracts = int(kelly_risk / calendar_cost)
    
    # Final position size (smaller of risk-based and Kelly)
    recommended_contracts = min(max_contracts_risk, kelly_contracts)
    recommended_contracts = max(1, min(recommended_contracts, 10))  # 1-10 contracts
    
    total_cost = recommended_contracts * calendar_cost
    risk_percentage = float((total_cost / config.account_size) * 100)
    
    return {
        'recommended_contracts': recommended_contracts,
        'total_cost': total_cost,
        'risk_percentage': risk_percentage,
        'max_loss': total_cost,  # Calendar max loss = premium paid
        'within_risk_limits': risk_percentage <= config.max_risk_per_trade * 100
    }


class CalendarSpreadValidator:
    """
    Stateful validator for calendar spread setups.
    
    Maintains configuration and provides validation methods.
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def validate_setup(
        self,
        symbol: str,
        strike: float,
        short_expiration: str,
        long_expiration: str,
        estimated_cost: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive calendar spread validation.
        
        Args:
            symbol: Underlying ticker
            strike: Strike price
            short_expiration: Short leg expiry (YYYY-MM-DD)
            long_expiration: Long leg expiry (YYYY-MM-DD)
            estimated_cost: Estimated calendar cost per contract
        
        Returns:
            Validation result with recommendations
        """
        try:
            # Parse dates
            short_date = datetime.strptime(short_expiration, "%Y-%m-%d").date()
            long_date = datetime.strptime(long_expiration, "%Y-%m-%d").date()
            
            # Validate calendar structure
            calendar_result = validate_calendar_spread(
                symbol,
                Decimal(str(strike)),
                short_date,
                long_date,
                self.config
            )
            
            # Add position sizing if cost provided
            if estimated_cost and calendar_result['valid']:
                sizing_result = validate_position_size(
                    Decimal(str(estimated_cost)),
                    self.config
                )
                calendar_result['position_sizing'] = sizing_result
            
            return calendar_result
            
        except (ValueError, TypeError) as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


# Factory functions for common scenarios

def create_production_config(account_size: float) -> TradingConfig:
    """Create production trading configuration"""
    return TradingConfig(
        environment=Environment.PRODUCTION,
        account_size=Decimal(str(account_size)),
        max_risk_per_trade=0.02,  # Conservative 2%
        min_days_to_expiry=2,     # Minimum for earnings
        max_days_to_expiry=30     # Reasonable maximum
    )


def create_backtesting_config(
    account_size: float, 
    as_of_date: date
) -> TradingConfig:
    """Create backtesting configuration"""
    return TradingConfig(
        environment=Environment.BACKTESTING,
        as_of_date=as_of_date,
        account_size=Decimal(str(account_size)),
        max_risk_per_trade=0.02
    )


# Quick validation functions for immediate use

def quick_validate_calendar(
    symbol: str,
    strike: float,
    short_exp: str,
    long_exp: str,
    account_size: float = 25000,
    cost_estimate: Optional[float] = None
) -> bool:
    """
    Quick validation for calendar spread - returns True/False
    
    Perfect for rapid screening of calendar opportunities.
    """
    try:
        config = create_production_config(account_size)
        validator = CalendarSpreadValidator(config)
        result = validator.validate_setup(symbol, strike, short_exp, long_exp, cost_estimate)
        return result.get('valid', False)
    except:
        return False


def get_validation_errors(
    symbol: str,
    strike: float,
    short_exp: str,
    long_exp: str,
    account_size: float = 25000
) -> list:
    """
    Get detailed validation errors for debugging.
    
    Use this when quick_validate_calendar returns False.
    """
    try:
        config = create_production_config(account_size)
        validator = CalendarSpreadValidator(config)
        result = validator.validate_setup(symbol, strike, short_exp, long_exp)
        return result.get('errors', []) + result.get('warnings', [])
    except Exception as e:
        return [f"Validation failed: {str(e)}"]