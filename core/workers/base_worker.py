"""
Base worker class for background processing tasks
Professional Options Calculator - Core Workers Module
"""

from PySide6.QtCore import QObject, QThread, Signal, QMutex, QWaitCondition
from PySide6.QtWidgets import QApplication
import logging
import traceback
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseWorker(QObject):
    """Base class for all background workers with proper thread management"""
    
    # Signals for communication with main thread
    started = Signal()
    finished = Signal()
    error = Signal(str, str)  # title, message
    progress = Signal(int)  # progress percentage (0-100)
    status_update = Signal(str)  # status message
    result_ready = Signal(object)  # result data
    
    def __init__(self):
        super().__init__()
        self._is_cancelled = False
        self._mutex = QMutex()
        self._wait_condition = QWaitCondition()
        self._progress_value = 0
        
    def run(self):
        """Main worker execution - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run() method")
    
    def cancel(self):
        """Cancel the worker operation"""
        with QMutex(self._mutex):
            self._is_cancelled = True
            self._wait_condition.wakeAll()
    
    def is_cancelled(self) -> bool:
        """Check if worker has been cancelled"""
        with QMutex(self._mutex):
            return self._is_cancelled
    
    def emit_progress(self, value: int, status: str = ""):
        """Emit progress update with optional status"""
        self._progress_value = max(0, min(100, value))
        self.progress.emit(self._progress_value)
        if status:
            self.status_update.emit(status)
    
    def emit_status(self, message: str):
        """Emit status update"""
        self.status_update.emit(message)
        logger.info(f"Worker status: {message}")
    
    def emit_error(self, title: str, message: str):
        """Emit error signal"""
        self.error.emit(title, message)
        logger.error(f"Worker error - {title}: {message}")
    
    def emit_result(self, result: Any):
        """Emit result signal"""
        self.result_ready.emit(result)
    
    def safe_execute(self, func, *args, **kwargs):
        """Safely execute a function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.emit_error("Execution Error", error_msg)
            return None

class WorkerThread(QThread):
    """Thread wrapper for workers"""
    
    def __init__(self, worker: BaseWorker):
        super().__init__()
        self.worker = worker
        self.worker.moveToThread(self)
        
        # Connect signals
        self.started.connect(worker.run)
        self.finished.connect(self.cleanup)
        
    def cleanup(self):
        """Clean up thread resources"""
        self.worker.deleteLater()
        self.deleteLater()

class WorkerManager(QObject):
    """Manages multiple worker threads"""
    
    def __init__(self):
        super().__init__()
        self.active_workers = {}
        self.worker_threads = {}
    
    def start_worker(self, worker_id: str, worker: BaseWorker) -> bool:
        """Start a worker in a new thread"""
        try:
            if worker_id in self.active_workers:
                logger.warning(f"Worker {worker_id} already running")
                return False
            
            thread = WorkerThread(worker)
            
            # Store references
            self.active_workers[worker_id] = worker
            self.worker_threads[worker_id] = thread
            
            # Connect cleanup
            thread.finished.connect(lambda: self._cleanup_worker(worker_id))
            
            # Start thread
            thread.start()
            logger.info(f"Started worker: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start worker {worker_id}: {e}")
            return False
    
    def stop_worker(self, worker_id: str) -> bool:
        """Stop a specific worker"""
        try:
            if worker_id not in self.active_workers:
                return False
            
            worker = self.active_workers[worker_id]
            worker.cancel()
            
            thread = self.worker_threads[worker_id]
            thread.quit()
            thread.wait(5000)  # Wait up to 5 seconds
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop worker {worker_id}: {e}")
            return False
    
    def stop_all_workers(self):
        """Stop all active workers"""
        worker_ids = list(self.active_workers.keys())
        for worker_id in worker_ids:
            self.stop_worker(worker_id)
    
    def _cleanup_worker(self, worker_id: str):
        """Clean up worker references"""
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
        if worker_id in self.worker_threads:
            del self.worker_threads[worker_id]
        logger.info(f"Cleaned up worker: {worker_id}")