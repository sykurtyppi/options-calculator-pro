"""
Thread Manager - Professional Threading System for PySide6
=========================================================

Handles all background processing operations including:
- Safe GUI thread communication
- Worker thread management
- Progress tracking and cancellation
- Error handling and recovery
- Resource cleanup and lifecycle management
- Thread pooling and optimization

Part of Professional Options Calculator v9.1
Optimized for Apple Silicon and PySide6
"""

import logging
import threading
import time
import queue
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import traceback

from PySide6.QtCore import QObject, QThread, Signal, QTimer, QMutex, QMutexLocker
from PySide6.QtWidgets import QApplication

# Import your existing utilities
from utils.config_manager import ConfigManager
from utils.logger import get_logger

logger = get_logger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkerType(Enum):
    """Types of worker threads"""
    ANALYSIS = "analysis"
    DATA_FETCH = "data_fetch"
    ML_TRAINING = "ml_training"
    SCANNER = "scanner"
    BACKTEST = "backtest"
    GENERAL = "general"

@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TaskInfo:
    """Information about a task"""
    task_id: str
    name: str
    worker_type: WorkerType
    priority: TaskPriority
    function: Callable
    args: tuple
    kwargs: dict
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    progress_callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancellation_token: Optional['CancellationToken'] = None

class CancellationToken:
    """Thread-safe cancellation token"""
    
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
    
    def cancel(self):
        """Cancel the operation"""
        with self._lock:
            self._cancelled = True
    
    def is_cancelled(self) -> bool:
        """Check if operation is cancelled"""
        with self._lock:
            return self._cancelled
    
    def raise_if_cancelled(self):
        """Raise exception if cancelled"""
        if self.is_cancelled():
            raise TaskCancelledException("Operation was cancelled")

class TaskCancelledException(Exception):
    """Exception raised when task is cancelled"""
    pass

class ProgressReporter:
    """Thread-safe progress reporting"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self._progress = 0.0
        self._message = ""
        self._lock = threading.Lock()
    
    def report(self, progress: float, message: str = ""):
        """Report progress (0.0 to 1.0)"""
        with self._lock:
            self._progress = max(0.0, min(1.0, progress))
            self._message = message
            
            if self.callback:
                try:
                    self.callback(self._progress, self._message)
                except Exception as e:
                    logger.warning(f"Error in progress callback: {e}")
    
    def get_progress(self) -> Tuple[float, str]:
        """Get current progress"""
        with self._lock:
            return self._progress, self._message

class WorkerThread(QThread):
    """
    Base worker thread class with signals for PySide6 integration
    """
    
    # Signals for communication with GUI thread
    task_started = Signal(str)  # task_id
    task_progress = Signal(str, float, str)  # task_id, progress, message
    task_completed = Signal(str, object)  # task_id, result
    task_failed = Signal(str, str)  # task_id, error_message
    task_cancelled = Signal(str)  # task_id
    
    def __init__(self, task_info: TaskInfo, parent=None):
        super().__init__(parent)
        self.task_info = task_info
        self.result = None
        self.error = None
        self.start_time = None
        
        # Progress reporter
        self.progress_reporter = ProgressReporter(
            callback=lambda p, m: self.task_progress.emit(task_info.task_id, p, m)
        )
        
        # Connect internal signals
        self.finished.connect(self._on_finished)
    
    def run(self):
        """Execute the task"""
        try:
            self.start_time = time.time()
            self.task_info.started_at = datetime.now()
            
            logger.debug(f"Starting task {self.task_info.task_id}: {self.task_info.name}")
            self.task_started.emit(self.task_info.task_id)
            
            # Check for cancellation
            if self.task_info.cancellation_token:
                self.task_info.cancellation_token.raise_if_cancelled()
            
            # Execute the function with progress reporting
            if self._function_accepts_progress():
                self.result = self.task_info.function(
                    *self.task_info.args,
                    progress_reporter=self.progress_reporter,
                    cancellation_token=self.task_info.cancellation_token,
                    **self.task_info.kwargs
                )
            else:
                self.result = self.task_info.function(
                    *self.task_info.args,
                    **self.task_info.kwargs
                )
            
            logger.debug(f"Task {self.task_info.task_id} completed successfully")
            
        except TaskCancelledException:
            logger.info(f"Task {self.task_info.task_id} was cancelled")
            self.task_cancelled.emit(self.task_info.task_id)
            return
            
        except Exception as e:
            self.error = str(e)
            error_details = f"Task {self.task_info.task_id} failed: {e}"
            logger.error(error_details)
            logger.error(traceback.format_exc())
            self.task_failed.emit(self.task_info.task_id, error_details)
            return
    
    def _on_finished(self):
        """Handle thread completion"""
        self.task_info.completed_at = datetime.now()
        
        if self.error is None and not (
            self.task_info.cancellation_token and 
            self.task_info.cancellation_token.is_cancelled()
        ):
            self.task_completed.emit(self.task_info.task_id, self.result)
    
    def _function_accepts_progress(self) -> bool:
        """Check if function accepts progress reporting parameters"""
        import inspect
        sig = inspect.signature(self.task_info.function)
        params = sig.parameters
        return 'progress_reporter' in params or 'cancellation_token' in params

class ThreadSafeCounter:
    """Thread-safe counter for generating unique IDs"""
    
    def __init__(self, start_value: int = 0):
        self._value = start_value
        self._lock = threading.Lock()
    
    def next(self) -> int:
        """Get next value"""
        with self._lock:
            self._value += 1
            return self._value
    
    def current(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value

class ThreadManager(QObject):
    """
    Professional Thread Manager for PySide6 Applications
    
    Provides comprehensive thread management including:
    - Task queuing with priorities
    - Worker thread pooling
    - Progress tracking and cancellation
    - Error handling and recovery
    - Resource management
    - Performance monitoring
    """
    
    # Signals for GUI integration
    task_queued = Signal(str, str)  # task_id, task_name
    task_started = Signal(str, str)  # task_id, task_name
    task_progress = Signal(str, float, str)  # task_id, progress, message
    task_completed = Signal(str, object)  # task_id, result
    task_failed = Signal(str, str)  # task_id, error_message
    task_cancelled = Signal(str)  # task_id
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.logger = logger
        
        # Configuration
        self.max_workers = self.config.get("threading.max_workers", 4)
        self.enable_threading = self.config.get("threading.enabled", True)
        self.task_timeout = self.config.get("threading.task_timeout", 300)  # 5 minutes
        
        # Thread management
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="OptionsCalc"
        )
        
        # Task management
        self._task_counter = ThreadSafeCounter()
        self._active_tasks: Dict[str, TaskInfo] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._worker_threads: Dict[str, WorkerThread] = {}
        
        # Thread safety
        self._tasks_mutex = QMutex()
        
        # Cleanup timer
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_completed_tasks)
        self._cleanup_timer.start(60000)  # Cleanup every minute
        
        # Performance tracking
        self._start_time = time.time()
        self._total_tasks_completed = 0
        self._total_execution_time = 0.0
        
        self.logger.info(f"ThreadManager initialized with {self.max_workers} workers")
    
    def submit_task(self, 
                   function: Callable,
                   args: tuple = (),
                   kwargs: dict = None,
                   name: str = None,
                   worker_type: WorkerType = WorkerType.GENERAL,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   callback: Optional[Callable] = None,
                   error_callback: Optional[Callable] = None,
                   progress_callback: Optional[Callable] = None,
                   cancellable: bool = True) -> str:
        """
        Submit a task for background execution
        
        Args:
            function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            name: Human-readable task name
            worker_type: Type of worker thread
            priority: Task priority
            callback: Success callback
            error_callback: Error callback
            progress_callback: Progress callback
            cancellable: Whether task can be cancelled
            
        Returns:
            Task ID for tracking
        """
        if kwargs is None:
            kwargs = {}
        
        # Generate unique task ID
        task_id = f"{worker_type.value}_{self._task_counter.next()}_{int(time.time())}"
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            name=name or f"{function.__name__}",
            worker_type=worker_type,
            priority=priority,
            function=function,
            args=args,
            kwargs=kwargs,
            callback=callback,
            error_callback=error_callback,
            progress_callback=progress_callback,
            cancellation_token=CancellationToken() if cancellable else None
        )
        
        # Store task
        with QMutexLocker(self._tasks_mutex):
            self._active_tasks[task_id] = task_info
        
        # Submit based on threading configuration
        if self.enable_threading:
            self._submit_threaded_task(task_info)
        else:
            self._submit_sync_task(task_info)
        
        self.task_queued.emit(task_id, task_info.name)
        self.logger.debug(f"Task submitted: {task_id} ({task_info.name})")
        
        return task_id
    
    def _submit_threaded_task(self, task_info: TaskInfo):
        """Submit task to thread pool"""
        try:
            # Create worker thread
            worker = WorkerThread(task_info, parent=self)
            
            # Connect signals
            worker.task_started.connect(lambda tid: self.task_started.emit(tid, task_info.name))
            worker.task_progress.connect(self.task_progress.emit)
            worker.task_completed.connect(self._on_task_completed)
            worker.task_failed.connect(self._on_task_failed)
            worker.task_cancelled.connect(self._on_task_cancelled)
            
            # Store worker reference
            self._worker_threads[task_info.task_id] = worker
            
            # Start worker
            worker.start()
            
        except Exception as e:
            self.logger.error(f"Error starting worker thread: {e}")
            self._on_task_failed(task_info.task_id, str(e))
    
    def _submit_sync_task(self, task_info: TaskInfo):
        """Execute task synchronously (for debugging/testing)"""
        try:
            self.task_started.emit(task_info.task_id, task_info.name)
            
            result = task_info.function(*task_info.args, **task_info.kwargs)
            self._on_task_completed(task_info.task_id, result)
            
        except Exception as e:
            self._on_task_failed(task_info.task_id, str(e))
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if cancellation was successful
        """
        try:
            with QMutexLocker(self._tasks_mutex):
                if task_id in self._active_tasks:
                    task_info = self._active_tasks[task_id]
                    
                    # Signal cancellation
                    if task_info.cancellation_token:
                        task_info.cancellation_token.cancel()
                    
                    # Terminate worker thread if exists
                    if task_id in self._worker_threads:
                        worker = self._worker_threads[task_id]
                        if worker.isRunning():
                            worker.terminate()
                            worker.wait(1000)  # Wait up to 1 second
                    
                    self.logger.info(f"Task cancelled: {task_id}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    def cancel_all_tasks(self, worker_type: Optional[WorkerType] = None) -> int:
        """
        Cancel all tasks, optionally filtered by worker type
        
        Args:
            worker_type: Optional filter by worker type
            
        Returns:
            Number of tasks cancelled
        """
        cancelled_count = 0
        
        try:
            with QMutexLocker(self._tasks_mutex):
                tasks_to_cancel = []
                
                for task_id, task_info in self._active_tasks.items():
                    if worker_type is None or task_info.worker_type == worker_type:
                        tasks_to_cancel.append(task_id)
                
                for task_id in tasks_to_cancel:
                    if self.cancel_task(task_id):
                        cancelled_count += 1
            
            self.logger.info(f"Cancelled {cancelled_count} tasks")
            return cancelled_count
            
        except Exception as e:
            self.logger.error(f"Error cancelling tasks: {e}")
            return cancelled_count
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        with QMutexLocker(self._tasks_mutex):
            if task_id in self._active_tasks:
                task_info = self._active_tasks[task_id]
                if task_info.started_at is None:
                    return TaskStatus.PENDING
                elif task_info.completed_at is None:
                    return TaskStatus.RUNNING
            elif task_id in self._completed_tasks:
                return self._completed_tasks[task_id].status
        
        return None
    
    def get_active_tasks(self, worker_type: Optional[WorkerType] = None) -> List[TaskInfo]:
        """Get list of active tasks"""
        with QMutexLocker(self._tasks_mutex):
            tasks = list(self._active_tasks.values())
            
            if worker_type:
                tasks = [t for t in tasks if t.worker_type == worker_type]
            
            return tasks
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of completed task"""
        with QMutexLocker(self._tasks_mutex):
            return self._completed_tasks.get(task_id)
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Wait for task completion
        
        Args:
            task_id: Task ID to wait for
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            TaskResult when complete, None if timeout
        """
        start_time = time.time()
        
        while True:
            result = self.get_task_result(task_id)
            if result is not None:
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            # Process events to allow GUI updates
            QApplication.processEvents()
            time.sleep(0.1)
    
    def is_busy(self, worker_type: Optional[WorkerType] = None) -> bool:
        """Check if there are active tasks"""
        active_tasks = self.get_active_tasks(worker_type)
        return len(active_tasks) > 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with QMutexLocker(self._tasks_mutex):
            uptime = time.time() - self._start_time
            
            active_count = len(self._active_tasks)
            completed_count = len(self._completed_tasks)
            total_count = active_count + completed_count
            
            avg_execution_time = (
                self._total_execution_time / self._total_tasks_completed
                if self._total_tasks_completed > 0 else 0
            )
            
            # Task counts by type
            task_counts_by_type = {}
            for task_info in self._active_tasks.values():
                worker_type = task_info.worker_type.value
                task_counts_by_type[worker_type] = task_counts_by_type.get(worker_type, 0) + 1
            
            return {
                "uptime_seconds": uptime,
                "max_workers": self.max_workers,
                "active_tasks": active_count,
                "completed_tasks": completed_count,
                "total_tasks": total_count,
                "tasks_per_minute": (total_count / (uptime / 60)) if uptime > 0 else 0,
                "average_execution_time": avg_execution_time,
                "task_counts_by_type": task_counts_by_type,
                "threading_enabled": self.enable_threading
            }
    
    # Signal handlers
    def _on_task_completed(self, task_id: str, result: Any):
        """Handle task completion"""
        try:
            with QMutexLocker(self._tasks_mutex):
                if task_id in self._active_tasks:
                    task_info = self._active_tasks.pop(task_id)
                    
                    # Calculate execution time
                    execution_time = 0.0
                    if task_info.started_at and task_info.completed_at:
                        execution_time = (task_info.completed_at - task_info.started_at).total_seconds()
                    
                    # Store result
                    task_result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        result=result,
                        execution_time=execution_time
                    )
                    self._completed_tasks[task_id] = task_result
                    
                    # Update performance stats
                    self._total_tasks_completed += 1
                    self._total_execution_time += execution_time
                    
                    # Call success callback
                    if task_info.callback:
                        try:
                            task_info.callback(result)
                        except Exception as e:
                            self.logger.warning(f"Error in task callback: {e}")
            
            # Clean up worker thread
            if task_id in self._worker_threads:
                del self._worker_threads[task_id]
            
            self.task_completed.emit(task_id, result)
            self.logger.debug(f"Task completed: {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling task completion: {e}")
    
    def _on_task_failed(self, task_id: str, error_message: str):
        """Handle task failure"""
        try:
            with QMutexLocker(self._tasks_mutex):
                if task_id in self._active_tasks:
                    task_info = self._active_tasks.pop(task_id)
                    
                    # Store error result
                    task_result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=error_message
                    )
                    self._completed_tasks[task_id] = task_result
                    
                    # Call error callback
                    if task_info.error_callback:
                        try:
                            task_info.error_callback(error_message)
                        except Exception as e:
                            self.logger.warning(f"Error in error callback: {e}")
            
            # Clean up worker thread
            if task_id in self._worker_threads:
                del self._worker_threads[task_id]
            
            self.task_failed.emit(task_id, error_message)
            self.logger.debug(f"Task failed: {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling task failure: {e}")
    
    def _on_task_cancelled(self, task_id: str):
        """Handle task cancellation"""
        try:
            with QMutexLocker(self._tasks_mutex):
                if task_id in self._active_tasks:
                    task_info = self._active_tasks.pop(task_id)
                    
                    # Store cancelled result
                    task_result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.CANCELLED
                    )
                    self._completed_tasks[task_id] = task_result
            
            # Clean up worker thread
            if task_id in self._worker_threads:
                del self._worker_threads[task_id]
            
            self.task_cancelled.emit(task_id)
            self.logger.debug(f"Task cancelled: {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling task cancellation: {e}")
    
    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)  # Keep for 1 hour
            
            with QMutexLocker(self._tasks_mutex):
                tasks_to_remove = [
                    task_id for task_id, result in self._completed_tasks.items()
                    if result.timestamp < cutoff_time
                ]
                
                for task_id in tasks_to_remove:
                    del self._completed_tasks[task_id]
                
                if tasks_to_remove:
                    self.logger.debug(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
            
        except Exception as e:
            self.logger.error(f"Error during task cleanup: {e}")
    
    def shutdown(self, wait_for_completion: bool = True, timeout: float = 10.0):
        """
        Shutdown thread manager
        
        Args:
            wait_for_completion: Whether to wait for running tasks
            timeout: Maximum time to wait for completion
        """
        try:
            self.logger.info("Shutting down ThreadManager...")
            
            # Stop cleanup timer
            self._cleanup_timer.stop()
            
            # Cancel all active tasks if not waiting
            if not wait_for_completion:
                cancelled_count = self.cancel_all_tasks()
                self.logger.info(f"Cancelled {cancelled_count} active tasks")
            
            # Shutdown thread pool
            self.executor.shutdown(wait=wait_for_completion)
            
            # Clean up worker threads
            for task_id, worker in self._worker_threads.items():
                if worker.isRunning():
                    if not wait_for_completion:
                        worker.terminate()
                    worker.wait(int(timeout * 1000))
            
            self._worker_threads.clear()
            
            # Clear tasks
            with QMutexLocker(self._tasks_mutex):
                self._active_tasks.clear()
            
            self.logger.info("ThreadManager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during ThreadManager shutdown: {e}")

# Convenience decorators and utilities
def run_in_background(thread_manager: ThreadManager, 
                     worker_type: WorkerType = WorkerType.GENERAL,
                     priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to run function in background thread"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_id = thread_manager.submit_task(
                function=func,
                args=args,
                kwargs=kwargs,
                name=func.__name__,
                worker_type=worker_type,
                priority=priority
            )
            return task_id
        return wrapper
    return decorator

def cancellable_task(func):
    """Decorator to make a function cancellable"""
    @wraps(func)
    def wrapper(*args, cancellation_token=None, progress_reporter=None, **kwargs):
        # Add periodic cancellation checks
        original_func = func
        
        def check_cancellation():
            if cancellation_token and cancellation_token.is_cancelled():
                raise TaskCancelledException("Task was cancelled")
        
        # You would need to modify the original function to call check_cancellation
        # This is a simplified example
        check_cancellation()
        return original_func(*args, **kwargs)
    
    return wrapper

class TaskBatch:
    """Utility for managing multiple related tasks"""
    
    def __init__(self, thread_manager: ThreadManager, name: str = "Batch"):
        self.thread_manager = thread_manager
        self.name = name
        self.task_ids: List[str] = []
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self._completed_count = 0
        self._lock = threading.Lock()
    
    def add_task(self, function: Callable, *args, **kwargs) -> str:
        """Add task to batch"""
        task_id = self.thread_manager.submit_task(
            function=function,
            args=args,
            kwargs=kwargs,
            name=f"{self.name}_task_{len(self.task_ids)}",
            callback=self._on_task_complete,
            error_callback=self._on_task_error
        )
        
        with self._lock:
            self.task_ids.append(task_id)
        
        return task_id
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks in batch to complete"""
        start_time = time.time()
        
        while True:
            with self._lock:
                if self._completed_count >= len(self.task_ids):
                    return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            QApplication.processEvents()
            time.sleep(0.1)
    
    def cancel_all(self) -> int:
        """Cancel all tasks in batch"""
        cancelled_count = 0
        with self._lock:
            for task_id in self.task_ids:
                if self.thread_manager.cancel_task(task_id):
                    cancelled_count += 1
        return cancelled_count
    
    def get_progress(self) -> float:
        """Get overall batch progress (0.0 to 1.0)"""
        with self._lock:
            if not self.task_ids:
                return 1.0
            return self._completed_count / len(self.task_ids)
    
    def _on_task_complete(self, result: Any):
        """Handle individual task completion"""
        with self._lock:
            self._completed_count += 1
    
    def _on_task_error(self, error: str):
        """Handle individual task error"""
        with self._lock:
            self._completed_count += 1