"""
Professional Options Calculator - Core Module
"""

from .workers.base_worker import BaseWorker, WorkerThread, WorkerManager

__all__ = [
    'BaseWorker',
    'WorkerThread', 
    'WorkerManager'
]

__version__ = "1.0.0"
