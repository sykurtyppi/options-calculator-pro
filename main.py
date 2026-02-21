"""
Options Calculator launcher.

Default launch path is the institutional dashboard (core.app/MainWindow).
The legacy simple calculator remains as a fallback UI.
"""

import sys
import logging
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox, QGridLayout,
    QSpinBox, QDoubleSpinBox, QProgressBar, QFrame
)
from PySide6.QtCore import QTimer, QThread, Signal
from PySide6.QtGui import QFont, QPalette, QColor

# Import our services
from services.market_data import MarketDataService
from services.options_service import OptionsService
from controllers.analysis_controller import AnalysisController
from utils.config_manager import ConfigManager

class DataFetcher(QThread):
    """Background thread for fetching market data"""
    data_ready = Signal(str, dict)  # symbol, data
    error_occurred = Signal(str)    # error message

    def __init__(self, market_service, symbol):
        super().__init__()
        self.market_service = market_service
        self.symbol = symbol

    def run(self):
        try:
            # Get current price
            current_price = self.market_service.get_current_price(self.symbol)

            if current_price <= 0:
                self.error_occurred.emit(f"No data available for {self.symbol}")
                return

            # Get historical data for volatility calculation
            historical = self.market_service.get_historical_data(self.symbol, period="1mo")

            volatility = 0.0
            if not historical.empty:
                returns = historical['Close'].pct_change().dropna()
                volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility %

            # Get market overview
            vix = self.market_service.get_vix()
            spx = self.market_service.get_spx()

            data = {
                'current_price': current_price,
                'historical_vol': volatility,
                'vix': vix,
                'spx': spx,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }

            self.data_ready.emit(self.symbol, data)

        except Exception as e:
            self.error_occurred.emit(f"Error fetching data: {str(e)}")

class SimpleOptionsCalculator(QMainWindow):
    """Clean, simple options calculator GUI"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Options Calculator Pro - Simple Edition")
        self.setGeometry(100, 100, 800, 600)

        # Initialize services
        self.market_service = MarketDataService()
        self.config_manager = ConfigManager()

        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #404040;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #666666;
                border-radius: 3px;
                color: #ffffff;
                font-family: 'Courier New', monospace;
            }
            QLabel {
                color: #ffffff;
            }
        """)

        self.setup_ui()

        # Timer for live updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.auto_update)
        self.update_timer.start(30000)  # Update every 30 seconds

    def setup_ui(self):
        """Create the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("Options Calculator Pro")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("margin: 10px; padding: 10px;")
        main_layout.addWidget(title)

        # Create sections
        self.create_input_section(main_layout)
        self.create_market_data_section(main_layout)
        self.create_results_section(main_layout)

        # Load default symbol
        self.symbol_input.setText("AAPL")
        self.fetch_data()

    def create_input_section(self, parent_layout):
        """Create input controls section"""
        input_group = QGroupBox("Symbol & Parameters")
        input_layout = QGridLayout(input_group)

        # Symbol input
        input_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_input = QLineEdit("AAPL")
        self.symbol_input.setPlaceholderText("Enter stock symbol...")
        input_layout.addWidget(self.symbol_input, 0, 1)

        # Strike price
        input_layout.addWidget(QLabel("Strike Price:"), 0, 2)
        self.strike_input = QDoubleSpinBox()
        self.strike_input.setRange(1.0, 9999.0)
        self.strike_input.setValue(200.0)
        self.strike_input.setDecimals(2)
        input_layout.addWidget(self.strike_input, 0, 3)

        # Days to expiration
        input_layout.addWidget(QLabel("Days to Exp:"), 1, 0)
        self.days_input = QSpinBox()
        self.days_input.setRange(1, 365)
        self.days_input.setValue(30)
        input_layout.addWidget(self.days_input, 1, 1)

        # Contracts
        input_layout.addWidget(QLabel("Contracts:"), 1, 2)
        self.contracts_input = QSpinBox()
        self.contracts_input.setRange(1, 1000)
        self.contracts_input.setValue(1)
        input_layout.addWidget(self.contracts_input, 1, 3)

        # Buttons
        button_layout = QHBoxLayout()

        self.fetch_button = QPushButton("Get Live Data")
        self.fetch_button.clicked.connect(self.fetch_data)
        button_layout.addWidget(self.fetch_button)

        self.analyze_button = QPushButton("Analyze Options")
        self.analyze_button.clicked.connect(self.analyze_options)
        button_layout.addWidget(self.analyze_button)

        button_layout.addStretch()

        input_layout.addLayout(button_layout, 2, 0, 1, 4)
        parent_layout.addWidget(input_group)

    def create_market_data_section(self, parent_layout):
        """Create market data display section"""
        market_group = QGroupBox("Live Market Data")
        market_layout = QGridLayout(market_group)

        # Current price
        market_layout.addWidget(QLabel("Current Price:"), 0, 0)
        self.price_label = QLabel("$0.00")
        self.price_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        market_layout.addWidget(self.price_label, 0, 1)

        # Volatility
        market_layout.addWidget(QLabel("Historical Vol:"), 0, 2)
        self.vol_label = QLabel("0.00%")
        self.vol_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #FFA726;")
        market_layout.addWidget(self.vol_label, 0, 3)

        # VIX
        market_layout.addWidget(QLabel("VIX:"), 1, 0)
        self.vix_label = QLabel("0.00")
        self.vix_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #EF5350;")
        market_layout.addWidget(self.vix_label, 1, 1)

        # S&P 500
        market_layout.addWidget(QLabel("S&P 500:"), 1, 2)
        self.spx_label = QLabel("0.00")
        self.spx_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #42A5F5;")
        market_layout.addWidget(self.spx_label, 1, 3)

        # Last update
        market_layout.addWidget(QLabel("Last Update:"), 2, 0)
        self.update_label = QLabel("Never")
        self.update_label.setStyleSheet("font-size: 10px; color: #888888;")
        market_layout.addWidget(self.update_label, 2, 1, 1, 3)

        parent_layout.addWidget(market_group)

    def create_results_section(self, parent_layout):
        """Create results display section"""
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        results_layout.addWidget(self.progress_bar)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setPlainText("Enter a symbol and click 'Get Live Data' to start...")
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)

        parent_layout.addWidget(results_group)

    def fetch_data(self):
        """Fetch live market data"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            self.show_error("Please enter a symbol")
            return

        self.fetch_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Start background data fetching
        self.data_fetcher = DataFetcher(self.market_service, symbol)
        self.data_fetcher.data_ready.connect(self.on_data_ready)
        self.data_fetcher.error_occurred.connect(self.on_data_error)
        self.data_fetcher.finished.connect(self.on_fetch_finished)
        self.data_fetcher.start()

    def on_data_ready(self, symbol, data):
        """Handle successful data fetch"""
        # Update price displays
        self.price_label.setText(f"${data['current_price']:.2f}")
        self.vol_label.setText(f"{data['historical_vol']:.1f}%")
        self.vix_label.setText(f"{data['vix']:.2f}")
        self.spx_label.setText(f"{data['spx']:.2f}")
        self.update_label.setText(f"Updated: {data['timestamp']}")

        # Update strike price suggestion
        current_price = data['current_price']
        # Round to nearest $5 for strikes
        suggested_strike = round(current_price / 5) * 5
        self.strike_input.setValue(suggested_strike)

        # Show data in results
        self.results_text.setPlainText(f"""
LIVE MARKET DATA for {symbol}
================================

Current Price: ${current_price:.2f}
Historical Volatility: {data['historical_vol']:.1f}%
VIX (Fear Index): {data['vix']:.2f}
S&P 500: {data['spx']:.2f}

Suggested Strike: ${suggested_strike:.2f}
Last Updated: {data['timestamp']}

Ready for options analysis!
Click 'Analyze Options' to calculate theoretical prices and Greeks.
""")

    def on_data_error(self, error_message):
        """Handle data fetch error"""
        self.show_error(error_message)

    def on_fetch_finished(self):
        """Clean up after data fetch"""
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def analyze_options(self):
        """Perform options analysis"""
        try:
            symbol = self.symbol_input.text().strip().upper()
            if not symbol:
                self.show_error("Please enter a symbol")
                return

            strike = self.strike_input.value()
            days = self.days_input.value()
            contracts = self.contracts_input.value()

            # Get current price
            current_price = float(self.price_label.text().replace('$', ''))
            if current_price <= 0:
                self.show_error("Please fetch live data first")
                return

            # Get volatility
            vol_text = self.vol_label.text().replace('%', '')
            volatility = float(vol_text) / 100 if vol_text != '0.00' else 0.25

            # Simple Black-Scholes calculation
            from math import log, sqrt, exp
            from scipy.stats import norm

            # Risk-free rate (approximate)
            r = 0.05

            # Time to expiration
            T = days / 365.0

            # Black-Scholes calculations
            d1 = (log(current_price / strike) + (r + 0.5 * volatility**2) * T) / (volatility * sqrt(T))
            d2 = d1 - volatility * sqrt(T)

            # Call option price
            call_price = current_price * norm.cdf(d1) - strike * exp(-r * T) * norm.cdf(d2)

            # Put option price
            put_price = strike * exp(-r * T) * norm.cdf(-d2) - current_price * norm.cdf(-d1)

            # Greeks
            call_delta = norm.cdf(d1)
            put_delta = call_delta - 1

            gamma = norm.pdf(d1) / (current_price * volatility * sqrt(T))
            theta_call = -(current_price * norm.pdf(d1) * volatility) / (2 * sqrt(T)) - r * strike * exp(-r * T) * norm.cdf(d2)
            theta_put = -(current_price * norm.pdf(d1) * volatility) / (2 * sqrt(T)) + r * strike * exp(-r * T) * norm.cdf(-d2)

            # Convert daily theta
            theta_call /= 365
            theta_put /= 365

            vega = current_price * norm.pdf(d1) * sqrt(T) / 100

            # Calculate position values
            call_position_value = call_price * contracts * 100
            put_position_value = put_price * contracts * 100

            # Display results
            results = f"""
OPTIONS ANALYSIS for {symbol}
===============================

UNDERLYING:
Current Price: ${current_price:.2f}
Strike Price: ${strike:.2f}
Days to Expiration: {days}
Implied Volatility: {volatility*100:.1f}%
Contracts: {contracts}

CALL OPTION:
Theoretical Price: ${call_price:.2f}
Position Value: ${call_position_value:,.2f} ({contracts} contracts)
Delta: {call_delta:.4f}
Gamma: {gamma:.4f}
Theta: ${theta_call:.2f}/day
Vega: ${vega:.2f}

PUT OPTION:
Theoretical Price: ${put_price:.2f}
Position Value: ${put_position_value:,.2f} ({contracts} contracts)
Delta: {put_delta:.4f}
Gamma: {gamma:.4f}
Theta: ${theta_put:.2f}/day
Vega: ${vega:.2f}

ANALYSIS:
Moneyness: {'ITM' if current_price > strike else 'OTM'} Call, {'ITM' if current_price < strike else 'OTM'} Put
Time Value: {abs(current_price - strike):.2f} points
Break-even Call: ${strike + call_price:.2f}
Break-even Put: ${strike - put_price:.2f}

Analysis completed at {datetime.now().strftime('%H:%M:%S')}
"""

            self.results_text.setPlainText(results)

        except Exception as e:
            self.show_error(f"Analysis error: {str(e)}")

    def auto_update(self):
        """Automatically update data"""
        symbol = self.symbol_input.text().strip()
        if symbol and not self.data_fetcher.isRunning() if hasattr(self, 'data_fetcher') else True:
            self.fetch_data()

    def show_error(self, message):
        """Show error message"""
        self.results_text.setPlainText(f"ERROR: {message}")

def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')
    app.setApplicationName("Options Calculator Pro")

    window = None
    try:
        # Preferred UI: institutional multi-tab dashboard.
        from core.app import OptionsCalculatorApp
        window = OptionsCalculatorApp()
    except Exception:
        logging.exception("Institutional UI launch failed, falling back to SimpleOptionsCalculator")
        window = SimpleOptionsCalculator()
        window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
