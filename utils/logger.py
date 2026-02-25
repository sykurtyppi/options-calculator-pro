# Logging setup
"""
Professional Logging System - Options Calculator Pro
Advanced logging with rotation, filtering, and performance monitoring
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
import functools
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.now()
    
    def filter(self, record):
        # Add runtime to record
        runtime = (datetime.now() - self.start_time).total_seconds()
        record.runtime = f"{runtime:.3f}s"
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            record.memory = f"{memory_mb:.1f}MB"
        except Exception:
            record.memory = "N/A"
        
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'process': record.process
        }
        
        # Add extra fields if available
        if hasattr(record, 'runtime'):
            log_entry['runtime'] = record.runtime
        if hasattr(record, 'memory'):
            log_entry['memory'] = record.memory
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        if hasattr(record, 'analysis_type'):
            log_entry['analysis_type'] = record.analysis_type
        
        return json.dumps(log_entry)


class TradingLogAdapter(logging.LoggerAdapter):
    """Specialized logger adapter for trading operations"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def analysis_start(self, symbol, analysis_type):
        """Log analysis start"""
        self.info(
            f"Starting {analysis_type} analysis for {symbol}",
            extra={'symbol': symbol, 'analysis_type': analysis_type}
        )
    
    def analysis_complete(self, symbol, analysis_type, confidence, duration):
        """Log analysis completion"""
        self.info(
            f"Completed {analysis_type} for {symbol}: {confidence:.1f}% confidence in {duration:.2f}s",
            extra={'symbol': symbol, 'analysis_type': analysis_type, 'confidence': confidence}
        )
    
    def analysis_error(self, symbol, analysis_type, error):
        """Log analysis error"""
        self.error(
            f"Analysis failed for {symbol} ({analysis_type}): {error}",
            extra={'symbol': symbol, 'analysis_type': analysis_type}
        )
    
    def trade_executed(self, symbol, trade_type, contracts, price):
        """Log trade execution"""
        self.info(
            f"Trade executed: {contracts} {trade_type} {symbol} @ ${price:.2f}",
            extra={'symbol': symbol, 'trade_type': trade_type, 'contracts': contracts, 'price': price}
        )
    
    def risk_alert(self, symbol, risk_type, value, threshold):
        """Log risk alerts"""
        self.warning(
            f"Risk alert for {symbol}: {risk_type} = {value} (threshold: {threshold})",
            extra={'symbol': symbol, 'risk_type': risk_type, 'value': value}
        )


def setup_logger(
    name: str = "options_calculator_pro",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_output: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup professional logging system
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ~/.options_calculator_pro/logs)
        console_output: Enable console output
        file_output: Enable file output
        json_output: Enable JSON structured logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Setup log directory
    if log_dir is None:
        log_dir = Path.home() / ".options_calculator_pro" / "logs"
    else:
        log_dir = Path(log_dir)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        try:
            fallback_root = Path(os.getenv("TMPDIR", "/tmp"))
            log_dir = fallback_root / ".options_calculator_pro" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            file_output = False
            json_output = False
    
    # Performance filter
    perf_filter = PerformanceFilter()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Colored formatter for console
        console_format = (
            "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | "
            "%(message)s | [%(runtime)s, %(memory)s]"
        )
        console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(perf_filter)
        
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        try:
            log_file = log_dir / f"{name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # File gets all levels

            # Detailed formatter for file
            file_format = (
                "%(asctime)s | %(levelname)-8s | %(process)d.%(thread)d | "
                "%(name)s.%(module)s.%(funcName)s:%(lineno)d | %(message)s | "
                "[%(runtime)s, %(memory)s]"
            )
            file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(perf_filter)
            logger.addHandler(file_handler)
        except Exception:
            file_output = False
    
    # JSON structured logging handler
    if json_output:
        try:
            json_file = log_dir / f"{name}_structured.jsonl"
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(logging.INFO)

            json_formatter = JSONFormatter()
            json_handler.setFormatter(json_formatter)
            json_handler.addFilter(perf_filter)
            logger.addHandler(json_handler)
        except Exception:
            json_output = False
    
    # Error handler for critical errors
    if file_output:
        try:
            error_file = log_dir / f"{name}_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=max_file_size // 2,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)

            error_format = (
                "%(asctime)s | %(levelname)-8s | %(process)d.%(thread)d | "
                "%(name)s.%(module)s.%(funcName)s:%(lineno)d | %(message)s\n"
                "Stack Trace:\n%(stack_info)s\n"
            )
            error_formatter = logging.Formatter(error_format, datefmt="%Y-%m-%d %H:%M:%S")
            error_handler.setFormatter(error_formatter)
            logger.addHandler(error_handler)
        except Exception:
            file_output = False
    
    # Log setup completion
    logger.info(f"Logging system initialized: level={level}, console={console_output}, "
                f"file={file_output}, json={json_output}")
    logger.info(f"Log directory: {log_dir}")
    
    return logger


def get_trading_logger(name: str = "trading") -> TradingLogAdapter:
    """Get specialized trading logger with extra functionality"""
    base_logger = logging.getLogger(f"options_calculator_pro.{name}")
    return TradingLogAdapter(base_logger)


def log_function_call(func):
    """Decorator to log function calls with timing"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"options_calculator_pro.{func.__module__}")
        
        # Log function entry
        args_str = ", ".join([str(arg) for arg in args[:3]])  # First 3 args only
        if len(args) > 3:
            args_str += "..."
        
        kwargs_str = ", ".join([f"{k}={v}" for k, v in list(kwargs.items())[:3]])
        if len(kwargs) > 3:
            kwargs_str += "..."
        
        logger.debug(f"Entering {func.__name__}({args_str}, {kwargs_str})")
        
        # Execute function with timing
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error in {func.__name__} after {duration:.3f}s: {e}")
            raise
    
    return wrapper


def log_performance(operation: str):
    """Context manager for performance logging"""
    class PerformanceLogger:
        def __init__(self, operation):
            self.operation = operation
            self.logger = logging.getLogger("options_calculator_pro.performance")
            self.start_time = None
        
        def __enter__(self):
            self.start_time = datetime.now()
            self.logger.debug(f"Starting {self.operation}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
            else:
                self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
    
    return PerformanceLogger(operation)


class LogAnalyzer:
    """Analyze log files for insights and debugging"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of errors in the last N hours"""
        errors = []
        error_file = self.log_dir / "options_calculator_pro_errors.log"
        
        if not error_file.exists():
            return {"error_count": 0, "errors": []}
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        try:
            with open(error_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if " | ERROR " in line or " | CRITICAL " in line:
                        # Parse timestamp
                        try:
                            timestamp_str = line.split(" | ")[0]
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            
                            if timestamp.timestamp() > cutoff_time:
                                errors.append({
                                    "timestamp": timestamp_str,
                                    "message": line.strip()
                                })
                        except Exception:
                            continue
        except Exception as e:
            return {"error": f"Failed to analyze logs: {e}"}
        
        return {
            "error_count": len(errors),
            "errors": errors[-10:],  # Last 10 errors
            "analysis_period_hours": hours
        }
    
    def get_performance_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics from logs"""
        stats = {
            "total_operations": 0,
            "avg_duration": 0.0,
            "max_duration": 0.0,
            "min_duration": float('inf'),
            "success_rate": 100.0
        }
        
        log_file = self.log_dir / "options_calculator_pro.log"
        if not log_file.exists():
            return stats
        
        durations = []
        operations = 0
        failures = 0
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if operation and operation not in line:
                        continue
                    
                    if "Completed" in line and "in " in line and "s" in line:
                        try:
                            # Extract duration
                            parts = line.split("in ")
                            if len(parts) > 1:
                                duration_part = parts[1].split("s")[0]
                                duration = float(duration_part)
                                durations.append(duration)
                                operations += 1
                        except:
                            continue
                    
                    elif "Failed" in line or "Error" in line:
                        if operation and operation in line:
                            failures += 1
        
        except Exception as e:
            stats["error"] = f"Failed to analyze performance: {e}"
            return stats
        
        if durations:
            stats["total_operations"] = operations
            stats["avg_duration"] = sum(durations) / len(durations)
            stats["max_duration"] = max(durations)
            stats["min_duration"] = min(durations)
            stats["success_rate"] = ((operations - failures) / operations * 100) if operations > 0 else 100.0
        
        return stats


# Convenience function for quick setup
def init_logging(debug: bool = False, json_logs: bool = False) -> logging.Logger:
    """Initialize logging with sensible defaults"""
    level = "DEBUG" if debug else "INFO"
    
    return setup_logger(
        level=level,
        console_output=True,
        file_output=True,
        json_output=json_logs
    )


# Export main functions
__all__ = [
    'setup_logger',
    'get_trading_logger',
    'log_function_call',
    'log_performance',
    'LogAnalyzer',
    'init_logging'
]
def get_logger(name=None):
    """Get a logger instance - compatibility function"""
    return setup_logger(name or __name__)
