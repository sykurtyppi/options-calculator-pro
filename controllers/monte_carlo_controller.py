# controllers/monte_carlo_controller.py

from PySide6.QtCore import QObject
from core.workers.monte_carlo_worker import MonteCarloWorker


class MonteCarloController(QObject):
    def __init__(self, result_callback, progress_callback):
        super().__init__()
        self.worker = None
        self.result_callback = result_callback
        self.progress_callback = progress_callback

    def start_simulation(self, spot, strike, rate, volatility, time_to_expiry, paths=10000, steps=100):
        self.worker = MonteCarloWorker(spot, strike, rate, volatility, time_to_expiry, paths, steps)
        self.worker.calculation_complete.connect(self.result_callback)
        self.worker.progress_update.connect(self.progress_callback)
        self.worker.start()
