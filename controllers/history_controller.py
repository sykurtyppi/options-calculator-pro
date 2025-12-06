"""
History Controller - Manages trading history and records
"""

from PySide6.QtCore import QObject, Signal
import logging

class HistoryController(QObject):
    """Controller for managing trade history"""
    
    # Signals that the view expects
    history_updated = Signal()
    statistics_updated = Signal()
    error_occurred = Signal(str)
    trade_saved = Signal(dict)    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.trade_history = []
    
    def add_trade(self, trade_data):
        """Add a trade to history"""
        self.trade_history.append(trade_data)
        self.history_updated.emit()
        self.statistics_updated.emit()
        self.logger.info(f"Added trade to history: {trade_data}")
    
    def get_history(self):
        """Get all trade history"""
        return self.trade_history
    
    def clear_history(self):
        """Clear all trade history"""
        self.trade_history.clear()
        self.history_updated.emit()
        self.statistics_updated.emit()
        self.logger.info("Trade history cleared")
    
    def refresh_data(self):
        """Refresh history data"""
        self.history_updated.emit()
        self.statistics_updated.emit()
    
    def show_error(self, error_message):
        """Handle and emit error"""
        self.logger.error(f"History error: {error_message}")
        self.error_occurred.emit(error_message)
    
    def get_all_trades(self):
        """Get all trades - alias for get_history"""
        return self.get_history()
    
    def save_trade(self, trade_data):
        """Save a trade"""
        self.add_trade(trade_data)
        self.trade_saved.emit(trade_data)
    def get_all_trades(self): return self.get_history()
    def save_trade(self, trade_data): self.add_trade(trade_data); self.trade_saved.emit(trade_data)
