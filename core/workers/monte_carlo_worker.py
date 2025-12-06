# core/workers/monte_carlo_worker.py

from PySide6.QtCore import QThread, Signal
import numpy as np


class MonteCarloWorker(QThread):
    calculation_complete = Signal(dict)
    progress_update = Signal(int)

    def __init__(self, spot, strike, rate, volatility, time_to_expiry, paths=10000, steps=100):
        super().__init__()
        self.spot = spot
        self.strike = strike
        self.rate = rate
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        self.paths = paths
        self.steps = steps

    def run(self):
        try:
            dt = self.time_to_expiry / self.steps
            drift = (self.rate - 0.5 * self.volatility ** 2) * dt
            diffusion = self.volatility * np.sqrt(dt)

            prices = np.zeros((self.paths, self.steps))
            prices[:, 0] = self.spot

            for t in range(1, self.steps):
                z = np.random.standard_normal(self.paths)
                prices[:, t] = prices[:, t - 1] * np.exp(drift + diffusion * z)
                if t % 10 == 0:
                    self.progress_update.emit(int(t / self.steps * 100))

            terminal_prices = prices[:, -1]
            payoff = np.maximum(terminal_prices - self.strike, 0)

            discounted_payoff = np.exp(-self.rate * self.time_to_expiry) * payoff
            option_price = np.mean(discounted_payoff)

            result = {
                "option_price": option_price,
                "mean_terminal_price": np.mean(terminal_prices),
                "std_terminal_price": np.std(terminal_prices),
            }

            self.calculation_complete.emit(result)

        except Exception as e:
            self.calculation_complete.emit({"error": str(e)})
