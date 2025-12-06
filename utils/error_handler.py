"""
Professional error handling system for Options Calculator Pro.

Provides domain-specific exceptions, error recovery mechanisms, and
comprehensive logging to ensure trading calculation failures are
never silently ignored.
"""

import logging
import traceback
import functools
from typing import Optional, Dict, Any, Callable, Union, Type
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from services.interfaces import (
    OptionsCalculatorError, MarketDataError, InvalidSymbolError,
    NetworkError, APILimitError, CalculationError, InvalidParameterError,
    VolatilityError, ConfigurationError, SecurityError
)


class ErrorSeverity(Enum):
    """Error severity levels for prioritization"""
    LOW = "low"           # Minor issues, graceful degradation possible
    MEDIUM = "medium"     # Significant issues, some functionality impacted
    HIGH = "high"         # Critical issues, major functionality broken
    CRITICAL = "critical" # System-breaking issues, immediate attention required


class ErrorCategory(Enum):
    """Error categories for classification"""
    DATA = "data"                 # Market data issues
    CALCULATION = "calculation"   # Options pricing/Greeks errors
    NETWORK = "network"          # Connectivity problems
    CONFIGURATION = "config"     # Setup/configuration errors
    SECURITY = "security"        # Authentication/authorization issues
    SYSTEM = "system"           # System-level errors
    USER_INPUT = "user_input"   # Invalid user parameters


@dataclass
class ErrorInfo:
    """Structured error information for logging and recovery"""
    error_id: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    context: Dict[str, Any]
    original_exception: Optional[Exception] = None
    recovery_suggested: Optional[str] = None
    user_message: Optional[str] = None


class ErrorHandler:
    """
    Professional error handling with recovery mechanisms.
    
    Features:
    - Domain-specific exception classification
    - Automatic error logging with context
    - Recovery suggestion system
    - User-friendly error messages
    - Error statistics and monitoring
    """
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self._error_counts: Dict[str, int] = {}
        self._recent_errors: Dict[str, datetime] = {}
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        severity: Optional[ErrorSeverity] = None
    ) -> ErrorInfo:
        """
        Process and log an error with full context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            user_message: User-friendly error message
            severity: Override severity classification
            
        Returns:
            Structured error information
        """
        error_info = self._classify_error(
            error, context or {}, user_message, severity
        )
        
        # Log error with appropriate level
        self._log_error(error_info)
        
        # Track error statistics
        self._track_error(error_info)
        
        return error_info
    
    def _classify_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        user_message: Optional[str],
        severity_override: Optional[ErrorSeverity]
    ) -> ErrorInfo:
        """Classify error and determine appropriate handling"""
        
        error_id = f"{type(error).__name__}_{hash(str(error)) % 10000:04d}"
        timestamp = datetime.now()
        
        # Classify by exception type
        category, severity, recovery, default_user_msg = self._get_error_classification(error)
        
        # Override severity if specified
        if severity_override:
            severity = severity_override
        
        # Use custom user message or default
        final_user_message = user_message or default_user_msg
        
        return ErrorInfo(
            error_id=error_id,
            message=str(error),
            category=category,
            severity=severity,
            timestamp=timestamp,
            context=context,
            original_exception=error,
            recovery_suggested=recovery,
            user_message=final_user_message
        )
    
    def _get_error_classification(
        self, 
        error: Exception
    ) -> tuple[ErrorCategory, ErrorSeverity, str, str]:
        """Get error classification details"""
        
        if isinstance(error, InvalidSymbolError):
            return (
                ErrorCategory.DATA,
                ErrorSeverity.MEDIUM,
                "Verify symbol format and check market data provider",
                "Invalid stock symbol. Please check the ticker symbol and try again."
            )
        
        if isinstance(error, NetworkError):
            return (
                ErrorCategory.NETWORK,
                ErrorSeverity.HIGH,
                "Check internet connection and retry. Consider offline mode.",
                "Network connection issue. Please check your internet connection."
            )
        
        if isinstance(error, APILimitError):
            return (
                ErrorCategory.DATA,
                ErrorSeverity.HIGH,
                "Wait for rate limit reset or use alternative data source",
                "API rate limit exceeded. Please wait a moment before trying again."
            )
        
        if isinstance(error, InvalidParameterError):
            return (
                ErrorCategory.USER_INPUT,
                ErrorSeverity.MEDIUM,
                "Review input parameters and valid ranges",
                "Invalid input parameters. Please check your values and try again."
            )
        
        if isinstance(error, VolatilityError):
            return (
                ErrorCategory.CALCULATION,
                ErrorSeverity.HIGH,
                "Use historical volatility as fallback or adjust calculation parameters",
                "Volatility calculation failed. Using historical volatility as estimate."
            )
        
        if isinstance(error, CalculationError):
            return (
                ErrorCategory.CALCULATION,
                ErrorSeverity.CRITICAL,
                "Verify market data quality and calculation inputs",
                "Options calculation failed. Please verify your inputs and try again."
            )
        
        if isinstance(error, SecurityError):
            return (
                ErrorCategory.SECURITY,
                ErrorSeverity.CRITICAL,
                "Check API credentials and security configuration",
                "Authentication error. Please check your API credentials."
            )
        
        if isinstance(error, ConfigurationError):
            return (
                ErrorCategory.CONFIGURATION,
                ErrorSeverity.HIGH,
                "Review configuration file and required settings",
                "Configuration error. Please check your settings."
            )
        
        if isinstance(error, MarketDataError):
            return (
                ErrorCategory.DATA,
                ErrorSeverity.HIGH,
                "Check data provider status and fallback sources",
                "Market data error. Attempting to use backup data source."
            )
        
        # Default for unknown errors
        return (
            ErrorCategory.SYSTEM,
            ErrorSeverity.CRITICAL,
            "Contact support with error details",
            "An unexpected error occurred. Please try again or contact support."
        )
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error with appropriate level and formatting"""
        
        log_msg = (
            f"[{error_info.error_id}] {error_info.category.value.upper()}: "
            f"{error_info.message}"
        )
        
        if error_info.context:
            log_msg += f" | Context: {error_info.context}"
        
        if error_info.recovery_suggested:
            log_msg += f" | Recovery: {error_info.recovery_suggested}"
        
        # Log with appropriate level based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_msg, exc_info=error_info.original_exception)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_msg, exc_info=error_info.original_exception)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
    
    def _track_error(self, error_info: ErrorInfo) -> None:
        """Track error statistics for monitoring"""
        error_key = f"{error_info.category.value}:{type(error_info.original_exception).__name__}"
        
        # Count occurrences
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        # Track most recent occurrence
        self._recent_errors[error_key] = error_info.timestamp
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            'total_errors': sum(self._error_counts.values()),
            'error_counts_by_type': dict(self._error_counts),
            'recent_errors': {
                error_type: timestamp.isoformat()
                for error_type, timestamp in self._recent_errors.items()
            },
            'top_errors': sorted(
                self._error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def reset_stats(self) -> None:
        """Reset error tracking statistics"""
        self._error_counts.clear()
        self._recent_errors.clear()


# Global error handler instance
_default_handler = ErrorHandler()


def handle_exceptions(
    error_handler: Optional[ErrorHandler] = None,
    context: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None,
    severity: Optional[ErrorSeverity] = None,
    reraise: bool = True
):
    """
    Decorator for automatic exception handling with context.
    
    Args:
        error_handler: Custom error handler (uses default if None)
        context: Additional context to include with errors
        user_message: Custom user-friendly message
        severity: Override error severity
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or _default_handler
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Build context from function info
                func_context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs': list(kwargs.keys())
                }
                
                if context:
                    func_context.update(context)
                
                # Handle the error
                error_info = handler.handle_error(
                    e, func_context, user_message, severity
                )
                
                if reraise:
                    raise
                
                return None
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return=None,
    context: Optional[Dict[str, Any]] = None,
    error_handler: Optional[ErrorHandler] = None,
    **kwargs
):
    """
    Execute function safely with automatic error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return on error
        context: Error context information
        error_handler: Custom error handler
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    handler = error_handler or _default_handler
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        exec_context = {
            'function': getattr(func, '__name__', str(func)),
            'args_provided': len(args),
            'kwargs_provided': list(kwargs.keys())
        }
        
        if context:
            exec_context.update(context)
        
        handler.handle_error(e, exec_context)
        return default_return


def create_trading_error_handler() -> ErrorHandler:
    """
    Create error handler optimized for trading operations.
    
    Returns:
        Configured error handler for trading context
    """
    handler = ErrorHandler("trading")
    
    # Configure for trading-specific needs
    trading_logger = logging.getLogger("trading")
    trading_logger.setLevel(logging.INFO)
    
    # Ensure critical trading errors are never silent
    def critical_error_alert(error_info: ErrorInfo):
        if error_info.severity == ErrorSeverity.CRITICAL:
            # In production, this could send alerts/notifications
            print(f"🚨 CRITICAL TRADING ERROR: {error_info.message}")
    
    return handler


# Convenience functions for common error scenarios

def raise_market_data_error(symbol: str, operation: str, details: str = None):
    """Raise standardized market data error"""
    message = f"Market data error for {symbol} during {operation}"
    if details:
        message += f": {details}"
    raise MarketDataError(message)


def raise_calculation_error(operation: str, parameters: Dict[str, Any], details: str = None):
    """Raise standardized calculation error"""
    param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
    message = f"Calculation error in {operation} with parameters [{param_str}]"
    if details:
        message += f": {details}"
    raise CalculationError(message)


def raise_validation_error(parameter: str, value: Any, expected: str):
    """Raise standardized parameter validation error"""
    message = f"Invalid {parameter}='{value}': {expected}"
    raise InvalidParameterError(message)