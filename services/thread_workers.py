"""
Background thread workers for Options Calculator Pro.

Provides high-performance background processing for intensive calendar spread
calculations, options scanning, and Monte Carlo simulations without blocking
the main UI thread.
"""

import logging
import threading
import queue
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Dict, List, Optional, Callable, Any, Union, TypeVar
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from models.option_data import OptionContract
from models.analysis_result import AnalysisResult
from utils.error_handler import ErrorHandler, handle_exceptions, ErrorSeverity

T = TypeVar('T')
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for work queue management"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class TaskStatus(Enum):
    """Task execution status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerTask:
    """Background task definition"""
    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Status tracking
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None


class BackgroundWorker(ABC):
    """Abstract base class for background workers"""
    
    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.max_workers = max_workers
        self.error_handler = ErrorHandler(f"worker.{name}")
        self.logger = logging.getLogger(f"worker.{name}")
        
        # Task management
        self._task_queue = queue.PriorityQueue()
        self._active_tasks: Dict[str, WorkerTask] = {}
        self._completed_tasks: Dict[str, WorkerTask] = {}
        self._task_counter = 0
        self._lock = threading.RLock()
        
        # Thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"worker-{name}"
        )
        self._futures: Dict[str, Future] = {}
        
        # Worker state
        self._running = False
        self._shutdown_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the background worker"""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"worker-{self.name}-main",
            daemon=True
        )
        self._worker_thread.start()
        
        self.logger.info(f"Started worker '{self.name}' with {self.max_workers} threads")
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the background worker with graceful shutdown"""
        if not self._running:
            return
        
        self.logger.info(f"Stopping worker '{self.name}'...")
        self._running = False
        self._shutdown_event.set()
        
        # Cancel pending tasks
        with self._lock:
            while not self._task_queue.empty():
                try:
                    _, task = self._task_queue.get_nowait()
                    task.status = TaskStatus.CANCELLED
                except queue.Empty:
                    break
        
        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        self.logger.info(f"Worker '{self.name}' stopped")
    
    def submit_task(
        self,
        function: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Submit a task for background execution.
        
        Args:
            function: Function to execute
            *args: Function arguments
            task_id: Custom task ID (auto-generated if None)
            priority: Task priority
            callback: Success callback function
            error_callback: Error callback function
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
        """
        if not self._running:
            raise RuntimeError(f"Worker '{self.name}' is not running")
        
        # Generate task ID if not provided
        if task_id is None:
            with self._lock:
                self._task_counter += 1
                task_id = f"{self.name}-{self._task_counter:06d}"
        
        task = WorkerTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            callback=callback,
            error_callback=error_callback,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Add to queue with priority
        priority_value = -priority.value  # Negative for max priority queue
        self._task_queue.put((priority_value, task))
        
        self.logger.debug(f"Submitted task {task_id} with priority {priority.name}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current status of a task"""
        with self._lock:
            if task_id in self._active_tasks:
                return self._active_tasks[task_id].status
            elif task_id in self._completed_tasks:
                return self._completed_tasks[task_id].status
        return None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of completed task"""
        with self._lock:
            if task_id in self._completed_tasks:
                task = self._completed_tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise task.error
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                if task.status == TaskStatus.QUEUED:
                    task.status = TaskStatus.CANCELLED
                    return True
                elif task_id in self._futures:
                    future = self._futures[task_id]
                    return future.cancel()
        return False
    
    def get_queue_size(self) -> int:
        """Get number of queued tasks"""
        return self._task_queue.qsize()
    
    def get_active_count(self) -> int:
        """Get number of active tasks"""
        with self._lock:
            return len([t for t in self._active_tasks.values() 
                       if t.status == TaskStatus.RUNNING])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        with self._lock:
            completed = len([t for t in self._completed_tasks.values() 
                           if t.status == TaskStatus.COMPLETED])
            failed = len([t for t in self._completed_tasks.values() 
                         if t.status == TaskStatus.FAILED])
            
            return {
                'name': self.name,
                'running': self._running,
                'queued_tasks': self.get_queue_size(),
                'active_tasks': self.get_active_count(),
                'completed_tasks': completed,
                'failed_tasks': failed,
                'total_tasks': completed + failed,
                'max_workers': self.max_workers
            }
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks"""
        self.logger.debug(f"Worker loop started for '{self.name}'")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # Get next task with timeout
                try:
                    priority, task = self._task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                # Execute task in thread pool
                self._execute_task(task)
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def _execute_task(self, task: WorkerTask) -> None:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        with self._lock:
            self._active_tasks[task.task_id] = task
        
        # Submit to thread pool
        future = self._executor.submit(self._run_task_with_error_handling, task)
        self._futures[task.task_id] = future
        
        # Monitor completion
        def on_task_complete(fut):
            self._handle_task_completion(task, fut)
        
        future.add_done_callback(on_task_complete)
    
    def _run_task_with_error_handling(self, task: WorkerTask) -> Any:
        """Run task with comprehensive error handling"""
        try:
            self.logger.debug(f"Executing task {task.task_id}")
            
            # Execute the task function
            result = task.function(*task.args, **task.kwargs)
            
            self.logger.debug(f"Task {task.task_id} completed successfully")
            return result
            
        except Exception as e:
            # Handle error with context
            context = {
                'task_id': task.task_id,
                'function': getattr(task.function, '__name__', str(task.function)),
                'args_count': len(task.args),
                'retry_count': task.retry_count
            }
            
            self.error_handler.handle_error(
                e, context, severity=ErrorSeverity.HIGH
            )
            
            # Determine if retry is appropriate
            if task.retry_count < task.max_retries:
                self.logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count + 1}/{task.max_retries})")
                task.retry_count += 1
                task.status = TaskStatus.QUEUED
                
                # Re-queue with delay
                time.sleep(min(2 ** task.retry_count, 30))  # Exponential backoff
                priority_value = -task.priority.value
                self._task_queue.put((priority_value, task))
                return None
            
            raise e
    
    def _handle_task_completion(self, task: WorkerTask, future: Future) -> None:
        """Handle task completion or failure"""
        task.completed_at = datetime.now()
        
        try:
            if future.cancelled():
                task.status = TaskStatus.CANCELLED
            elif future.exception():
                task.status = TaskStatus.FAILED
                task.error = future.exception()
                
                # Call error callback if provided
                if task.error_callback:
                    try:
                        task.error_callback(task.task_id, task.error)
                    except Exception as e:
                        self.logger.error(f"Error in error callback for task {task.task_id}: {e}")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = future.result()
                
                # Call success callback if provided
                if task.callback:
                    try:
                        task.callback(task.task_id, task.result)
                    except Exception as e:
                        self.logger.error(f"Error in success callback for task {task.task_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error handling task completion for {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = e
        
        finally:
            # Move from active to completed
            with self._lock:
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                self._completed_tasks[task.task_id] = task
                
                if task.task_id in self._futures:
                    del self._futures[task.task_id]
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class CalendarSpreadWorker(BackgroundWorker):
    """
    Specialized worker for calendar spread analysis and scanning.
    
    Optimized for intensive options calculations required for
    institutional hedging detection around earnings.
    """
    
    def __init__(self, max_workers: int = 6):
        super().__init__("calendar_spread", max_workers)
        
    def analyze_calendar_spread(
        self,
        symbol: str,
        strike: float,
        short_expiry: date,
        long_expiry: date,
        callback: Optional[Callable] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Submit calendar spread analysis task.
        
        Args:
            symbol: Stock symbol
            strike: Strike price
            short_expiry: Short leg expiration
            long_expiry: Long leg expiration
            callback: Result callback function
            priority: Task priority
            
        Returns:
            Task ID for tracking
        """
        from services.container import get_container
        from services.interfaces import IOptionsService
        
        def analyze():
            container = get_container()
            options_service = container.get_service(IOptionsService)
            return options_service.analyze_calendar_spread(
                symbol, strike, short_expiry, long_expiry
            )
        
        task_id = f"calendar-{symbol}-{strike}-{short_expiry}-{long_expiry}"
        
        return self.submit_task(
            analyze,
            task_id=task_id,
            priority=priority,
            callback=callback,
            timeout=30.0  # 30 second timeout
        )
    
    def scan_calendar_opportunities(
        self,
        symbols: List[str],
        earnings_dates: Dict[str, date],
        max_strike_count: int = 10,
        callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Scan multiple symbols for calendar spread opportunities.
        
        Args:
            symbols: List of stock symbols to scan
            earnings_dates: Earnings dates for each symbol
            max_strike_count: Maximum strikes to analyze per symbol
            callback: Progress callback function
            
        Returns:
            List of task IDs for tracking
        """
        task_ids = []
        
        for symbol in symbols:
            if symbol not in earnings_dates:
                continue
            
            earnings_date = earnings_dates[symbol]
            
            # Submit scan task for this symbol
            task_id = self.submit_task(
                self._scan_symbol_opportunities,
                symbol,
                earnings_date,
                max_strike_count,
                task_id=f"scan-{symbol}",
                priority=TaskPriority.HIGH,
                callback=callback,
                timeout=60.0
            )
            
            task_ids.append(task_id)
        
        return task_ids
    
    def _scan_symbol_opportunities(
        self,
        symbol: str,
        earnings_date: date,
        max_strike_count: int
    ) -> List[Dict[str, Any]]:
        """Scan opportunities for a single symbol"""
        from services.container import get_container
        from services.interfaces import IMarketDataService, IOptionsService
        
        container = get_container()
        market_data = container.get_service(IMarketDataService)
        options_service = container.get_service(IOptionsService)
        
        opportunities = []
        
        try:
            # Get current price and option chain
            current_price = market_data.get_current_price(symbol)
            if not current_price:
                return opportunities
            
            # Find suitable expiration dates around earnings
            option_chains = market_data.get_option_chain(symbol, earnings_date)
            if not option_chains:
                return opportunities
            
            # Analyze top strikes around current price
            strikes = self._get_target_strikes(current_price, max_strike_count)
            
            for strike in strikes:
                # Find short and long expiration dates
                short_expiry, long_expiry = self._find_calendar_expirations(
                    earnings_date, option_chains
                )
                
                if short_expiry and long_expiry:
                    result = options_service.analyze_calendar_spread(
                        symbol, strike, short_expiry, long_expiry
                    )
                    
                    if result and result.is_profitable:
                        opportunities.append({
                            'symbol': symbol,
                            'strike': strike,
                            'short_expiry': short_expiry,
                            'long_expiry': long_expiry,
                            'analysis': result
                        })
        
        except Exception as e:
            self.logger.warning(f"Error scanning {symbol}: {e}")
        
        return opportunities
    
    def _get_target_strikes(self, current_price: float, max_count: int) -> List[float]:
        """Get target strikes around current price"""
        strikes = []
        
        # Focus on ATM and slightly OTM strikes for calendar spreads
        strike_range = current_price * 0.1  # ±10% range
        strike_step = strike_range / (max_count / 2)
        
        for i in range(max_count):
            strike = current_price - strike_range + (i * strike_step)
            strikes.append(round(strike, 2))
        
        return strikes
    
    def _find_calendar_expirations(
        self,
        earnings_date: date,
        option_chains: Dict[str, Any]
    ) -> tuple[Optional[date], Optional[date]]:
        """Find suitable short and long expiration dates for calendar spread"""
        
        # For earnings plays:
        # Short leg: Week of earnings (capture IV crush)
        # Long leg: 2-4 weeks after earnings (maintain time value)
        
        # This is a simplified implementation
        # Real implementation would analyze available chains
        
        from datetime import timedelta
        
        # Find expiration closest to earnings (short leg)
        short_expiry = earnings_date + timedelta(days=3)  # Friday after earnings
        
        # Find expiration 2-3 weeks later (long leg)
        long_expiry = short_expiry + timedelta(days=21)  # 3 weeks later
        
        return short_expiry, long_expiry


class MonteCarloWorker(BackgroundWorker):
    """Worker specialized for Monte Carlo simulations"""
    
    def __init__(self, max_workers: int = 4):
        super().__init__("monte_carlo", max_workers)
    
    def run_simulation(
        self,
        option: OptionContract,
        num_simulations: int = 10000,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit Monte Carlo simulation task"""
        from utils.monte_carlo import MonteCarloEngine
        from services.market_data import MarketDataService
        import pandas as pd
        import numpy as np
        from datetime import date
        
        def simulate():
            engine = MonteCarloEngine()
            market_data = MarketDataService()
            symbol = str(getattr(option, "symbol", "UNKNOWN"))

            spot_guess = float(
                getattr(option, "underlying_price", 0.0)
                or getattr(option, "spot_price", 0.0)
                or getattr(option, "underlying_last", 0.0)
                or getattr(option, "strike", 0.0)
                or 100.0
            )
            spot_guess = max(0.01, spot_guess)

            historical_data = market_data.get_historical_data(symbol, period="6mo", interval="1d")
            if historical_data is None or historical_data.empty:
                # Synthetic flat history fallback to keep worker deterministic/stable.
                closes = np.full(120, spot_guess, dtype=float)
                historical_data = pd.DataFrame({"Close": closes})

            implied_vol = float(getattr(option, "implied_volatility", 0.25) or 0.25)
            implied_vol = max(0.01, min(3.0, implied_vol))
            rv_proxy = max(0.01, min(3.0, implied_vol * 0.85))

            dte = 30
            expiration = getattr(option, "expiration", None)
            if expiration is not None:
                try:
                    if hasattr(expiration, "date"):
                        expiration = expiration.date()
                    dte = max(1, int((expiration - date.today()).days))
                except Exception:
                    dte = 30

            return engine.run_simulation(
                symbol=symbol,
                current_price=spot_guess,
                historical_data=historical_data,
                volatility_metrics={
                    "iv30": implied_vol,
                    "rv30": rv_proxy,
                },
                simulations=max(1000, int(num_simulations)),
                days_to_expiration=dte,
                use_jump_diffusion=True,
            )
        
        return self.submit_task(
            simulate,
            task_id=f"mc-{option.symbol}-{option.strike}",
            priority=TaskPriority.HIGH,
            callback=callback,
            timeout=120.0  # 2 minute timeout for large simulations
        )


# Global worker instances
_calendar_worker: Optional[CalendarSpreadWorker] = None
_monte_carlo_worker: Optional[MonteCarloWorker] = None
_worker_lock = threading.Lock()


def get_calendar_worker() -> CalendarSpreadWorker:
    """Get or create the global calendar spread worker"""
    global _calendar_worker
    
    with _worker_lock:
        if _calendar_worker is None:
            _calendar_worker = CalendarSpreadWorker()
            _calendar_worker.start()
        
        return _calendar_worker


def get_monte_carlo_worker() -> MonteCarloWorker:
    """Get or create the global Monte Carlo worker"""
    global _monte_carlo_worker
    
    with _worker_lock:
        if _monte_carlo_worker is None:
            _monte_carlo_worker = MonteCarloWorker()
            _monte_carlo_worker.start()
        
        return _monte_carlo_worker


def shutdown_all_workers(timeout: float = 30.0) -> None:
    """Shutdown all global workers"""
    global _calendar_worker, _monte_carlo_worker
    
    with _worker_lock:
        if _calendar_worker:
            _calendar_worker.stop(timeout)
            _calendar_worker = None
        
        if _monte_carlo_worker:
            _monte_carlo_worker.stop(timeout)
            _monte_carlo_worker = None


# Convenience functions for common tasks

@handle_exceptions(reraise=False)
def analyze_calendar_spread_async(
    symbol: str,
    strike: float,
    short_expiry: date,
    long_expiry: date,
    callback: Optional[Callable] = None
) -> str:
    """Convenience function for async calendar spread analysis"""
    worker = get_calendar_worker()
    return worker.analyze_calendar_spread(
        symbol, strike, short_expiry, long_expiry, callback
    )


@handle_exceptions(reraise=False)
def scan_earnings_opportunities_async(
    symbols: List[str],
    earnings_dates: Dict[str, date],
    progress_callback: Optional[Callable] = None
) -> List[str]:
    """Convenience function for async earnings opportunity scanning"""
    worker = get_calendar_worker()
    return worker.scan_calendar_opportunities(
        symbols, earnings_dates, callback=progress_callback
    )
