"""
Custom widgets for the Professional Options Calculator
Advanced UI components with professional styling
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
    QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QProgressBar, QTextEdit, QFrame,
    QGroupBox, QScrollArea, QSplitter
)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPalette, QColor, QPainter, QPen
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalysisInputWidget(QWidget):
    """Professional input widget for analysis parameters"""
    
    analysis_requested = Signal(dict)  # Emits analysis parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_validation()
        
    def setup_ui(self):
        """Setup input widget UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Analysis Parameters")
        header.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #60a3d9;
                padding: 8px 0px;
                border-bottom: 2px solid #4a4a4a;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(header)
        
        # Input form
        form_layout = QGridLayout()
        
        # Stock symbols input
        form_layout.addWidget(QLabel("Stock Symbol(s):"), 0, 0)
        self.symbols_input = QLineEdit()
        self.symbols_input.setPlaceholderText("Enter symbols separated by commas (e.g., AAPL, MSFT, GOOGL)")
        self.symbols_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                font-size: 10pt;
                min-width: 300px;
            }
        """)
        form_layout.addWidget(self.symbols_input, 0, 1, 1, 3)
        
        # Contracts input
        form_layout.addWidget(QLabel("Contracts:"), 1, 0)
        self.contracts_input = QSpinBox()
        self.contracts_input.setRange(1, 100)
        self.contracts_input.setValue(1)
        self.contracts_input.setMaximumWidth(100)
        form_layout.addWidget(self.contracts_input, 1, 1)
        
        # Debit input
        form_layout.addWidget(QLabel("Debit ($):"), 1, 2)
        self.debit_input = QDoubleSpinBox()
        self.debit_input.setRange(0.01, 50.00)
        self.debit_input.setDecimals(2)
        self.debit_input.setValue(1.00)
        self.debit_input.setMaximumWidth(100)
        self.debit_input.setSpecialValueText("Auto")
        form_layout.addWidget(self.debit_input, 1, 3)
        
        # Advanced options (collapsible)
        self.advanced_group = QGroupBox("Advanced Options")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        self.advanced_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                margin-top: 15px;
            }
            QGroupBox::title {
                color: #60a3d9;
            }
        """)
        
        advanced_layout = QGridLayout(self.advanced_group)
        
        # Portfolio value
        advanced_layout.addWidget(QLabel("Portfolio Value ($):"), 0, 0)
        self.portfolio_input = QSpinBox()
        self.portfolio_input.setRange(1000, 10000000)
        self.portfolio_input.setValue(100000)
        self.portfolio_input.setMaximumWidth(120)
        advanced_layout.addWidget(self.portfolio_input, 0, 1)
        
        # Max risk percentage
        advanced_layout.addWidget(QLabel("Max Risk (%):"), 0, 2)
        self.max_risk_input = QDoubleSpinBox()
        self.max_risk_input.setRange(0.1, 10.0)
        self.max_risk_input.setValue(2.0)
        self.max_risk_input.setDecimals(1)
        self.max_risk_input.setSuffix("%")
        self.max_risk_input.setMaximumWidth(100)
        advanced_layout.addWidget(self.max_risk_input, 0, 3)
        
        # Analysis depth
        advanced_layout.addWidget(QLabel("Analysis Depth:"), 1, 0)
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["Quick", "Standard", "Comprehensive"])
        self.depth_combo.setCurrentText("Standard")
        self.depth_combo.setMaximumWidth(120)
        advanced_layout.addWidget(self.depth_combo, 1, 1)
        
        # Use ML predictions
        self.use_ml_checkbox = QCheckBox("Use ML Predictions")
        self.use_ml_checkbox.setChecked(True)
        advanced_layout.addWidget(self.use_ml_checkbox, 1, 2)
        
        # Add to main layout
        layout.addLayout(form_layout)
        layout.addWidget(self.advanced_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("🔍 Analyze")
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                font-size: 11pt;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
        self.analyze_button.clicked.connect(self.request_analysis)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_inputs)
        
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def setup_validation(self):
        """Setup input validation"""
        # Real-time validation for symbols
        self.symbols_input.textChanged.connect(self.validate_symbols)
        
        # Enable/disable analyze button based on validation
        self.symbols_input.textChanged.connect(self.update_analyze_button)
    
    def validate_symbols(self):
        """Validate stock symbols input"""
        text = self.symbols_input.text().strip()
        
        if not text:
            self.symbols_input.setStyleSheet("""
                QLineEdit {
                    padding: 8px;
                    font-size: 10pt;
                    min-width: 300px;
                    border: 2px solid #dc3545;
                }
            """)
            return False
        
        # Basic symbol validation
        symbols = [s.strip().upper() for s in text.split(',') if s.strip()]
        valid_symbols = []
        
        for symbol in symbols:
            if len(symbol) <= 5 and symbol.isalpha():
                valid_symbols.append(symbol)
        
        if valid_symbols:
            self.symbols_input.setStyleSheet("""
                QLineEdit {
                    padding: 8px;
                    font-size: 10pt;
                    min-width: 300px;
                    border: 2px solid #28a745;
                }
            """)
            return True
        else:
            self.symbols_input.setStyleSheet("""
                QLineEdit {
                    padding: 8px;
                    font-size: 10pt;
                    min-width: 300px;
                    border: 2px solid #dc3545;
                }
            """)
            return False
    
    def update_analyze_button(self):
        """Update analyze button state based on validation"""
        is_valid = self.validate_symbols()
        self.analyze_button.setEnabled(is_valid)
    
    def request_analysis(self):
        """Request analysis with current parameters"""
        if not self.validate_symbols():
            return
        
        # Collect parameters
        symbols_text = self.symbols_input.text().strip()
        symbols = [s.strip().upper() for s in symbols_text.split(',') if s.strip()]
        
        parameters = {
            'symbols': symbols,
            'contracts': self.contracts_input.value(),
            'debit': self.debit_input.value() if self.debit_input.value() > 0 else None,
            'portfolio_value': self.portfolio_input.value(),
            'max_risk_pct': self.max_risk_input.value() / 100,
            'analysis_depth': self.depth_combo.currentText().lower(),
            'use_ml': self.use_ml_checkbox.isChecked(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.analysis_requested.emit(parameters)
    
    def clear_inputs(self):
        """Clear all input fields"""
        self.symbols_input.clear()
        self.contracts_input.setValue(1)
        self.debit_input.setValue(1.00)
        self.portfolio_input.setValue(100000)
        self.max_risk_input.setValue(2.0)
        self.depth_combo.setCurrentText("Standard")
        self.use_ml_checkbox.setChecked(True)
    
    def set_symbols(self, symbols: List[str]):
        """Set symbols from external source (e.g., favorites)"""
        self.symbols_input.setText(', '.join(symbols))


class ConfidenceIndicator(QWidget):
    """Professional confidence indicator with visual feedback"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.confidence_value = 0
        self.setup_ui()
        
    def setup_ui(self):
        """Setup confidence indicator UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        self.title_label = QLabel("Confidence Score")
        self.title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #cccccc;
                font-size: 10pt;
                margin-bottom: 5px;
            }
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Confidence value display
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("""
            QLabel {
                font-size: 24pt;
                font-weight: bold;
                color: #ffffff;
                margin: 5px;
            }
        """)
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(15)
        layout.addWidget(self.progress_bar)
        
        # Recommendation label
        self.recommendation_label = QLabel("No Analysis")
        self.recommendation_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #cccccc;
                font-size: 11pt;
                margin-top: 5px;
            }
        """)
        self.recommendation_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.recommendation_label)
        
        # Set initial styling
        self.update_styling(0)
    
    def set_confidence(self, confidence: float):
        """Set confidence value and update display"""
        self.confidence_value = max(0, min(100, confidence))
        
        # Update displays
        self.value_label.setText(f"{self.confidence_value:.1f}%")
        self.progress_bar.setValue(int(self.confidence_value))
        
        # Update recommendation
        if self.confidence_value >= 75:
            recommendation = "STRONG BUY"
        elif self.confidence_value >= 60:
            recommendation = "BUY"
        elif self.confidence_value >= 45:
            recommendation = "CONSIDER"
        else:
            recommendation = "AVOID"
        
        self.recommendation_label.setText(recommendation)
        
        # Update styling based on confidence
        self.update_styling(self.confidence_value)
        
        # Animate the update
        self.animate_update()
    
    def update_styling(self, confidence: float):
        """Update styling based on confidence level"""
        if confidence >= 70:
            color = "#28a745"  # Green
            bg_color = "rgba(40, 167, 69, 0.1)"
        elif confidence >= 50:
            color = "#ffc107"  # Yellow
            bg_color = "rgba(255, 193, 7, 0.1)"
        else:
            color = "#dc3545"  # Red
            bg_color = "rgba(220, 53, 69, 0.1)"
        
        # Update value label color
        self.value_label.setStyleSheet(f"""
            QLabel {{
                font-size: 24pt;
                font-weight: bold;
                color: {color};
                margin: 5px;
            }}
        """)
        
        # Update progress bar color
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #5a5a5a;
                border-radius: 7px;
                text-align: center;
                background-color: #404040;
            }}
            
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 5px;
            }}
        """)
        
        # Update recommendation color
        self.recommendation_label.setStyleSheet(f"""
            QLabel {{
                font-weight: bold;
                color: {color};
                font-size: 11pt;
                margin-top: 5px;
            }}
        """)
        
        # Update widget background
        self.setStyleSheet(f"""
            ConfidenceIndicator {{
                background-color: {bg_color};
                border: 2px solid {color};
                border-radius: 8px;
            }}
        """)
    
    def animate_update(self):
        """Animate confidence update"""
        # Create a simple animation effect
        self.animation = QPropertyAnimation(self.progress_bar, b"value")
        self.animation.setDuration(1000)  # 1 second
        self.animation.setStartValue(0)
        self.animation.setEndValue(int(self.confidence_value))
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()


class TradingMetricsPanel(QWidget):
    """Panel displaying key trading metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup metrics panel UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Trading Metrics")
        header.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #60a3d9;
                padding: 8px 0px;
                border-bottom: 2px solid #4a4a4a;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(header)
        
        # Metrics grid
        self.metrics_layout = QGridLayout()
        layout.addLayout(self.metrics_layout)
        
        # Add default metrics
        self.add_default_metrics()
        
        layout.addStretch()
    
    def add_default_metrics(self):
        """Add default trading metrics"""
        default_metrics = [
            ("Expected Move", "--", "#cccccc"),
            ("Max Loss", "--", "#dc3545"),
            ("Max Profit", "--", "#28a745"),
            ("Breakeven", "--", "#ffc107"),
            ("Days to Expiry", "--", "#cccccc"),
            ("IV Rank", "--", "#60a3d9"),
            ("Volume", "--", "#cccccc"),
            ("Sector", "--", "#cccccc")
        ]
        
        for i, (label, value, color) in enumerate(default_metrics):
            self.add_metric(label, value, color)
    
    def add_metric(self, label: str, value: str, color: str = "#ffffff"):
        """Add a metric to the panel"""
        row = len(self.metrics) // 2
        col = (len(self.metrics) % 2) * 2
        
        # Label
        label_widget = QLabel(f"{label}:")
        label_widget.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #cccccc;
                padding: 4px;
                font-size: 10pt;
            }
        """)
        
        # Value
        value_widget = QLabel(value)
        value_widget.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                padding: 4px;
                font-size: 10pt;
            }}
        """)
        
        self.metrics_layout.addWidget(label_widget, row, col)
        self.metrics_layout.addWidget(value_widget, row, col + 1)
        
        self.metrics[label] = value_widget
    
    def update_metric(self, label: str, value: str, color: str = None):
        """Update an existing metric"""
        if label in self.metrics:
            self.metrics[label].setText(value)
            if color:
                self.metrics[label].setStyleSheet(f"""
                    QLabel {{
                        color: {color};
                        font-weight: bold;
                        padding: 4px;
                        font-size: 10pt;
                    }}
                """)
    
    def update_all_metrics(self, data: Dict[str, Any]):
        """Update all metrics from analysis data"""
        try:
            # Update metrics based on analysis data structure
            self.update_metric("Expected Move", data.get('expected_move', '--'))
            self.update_metric("Max Loss", data.get('max_loss', '--'), "#dc3545")
            self.update_metric("Max Profit", data.get('max_profit', '--'), "#28a745")
            
            if 'days_to_earnings' in data:
                self.update_metric("Days to Expiry", f"{data['days_to_earnings']} days")
            
            if 'volatility_metrics' in data:
                vol_data = data['volatility_metrics']
                iv_rank = vol_data.get('iv_rank', 0)
                self.update_metric("IV Rank", f"{iv_rank:.1%}")
            
            if 'underlying_price' in data:
                price = data['underlying_price']
                self.update_metric("Current Price", f"${price:.2f}")
            
            # Add more metric updates as needed
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")


class ProgressWidget(QWidget):
    """Advanced progress widget with status and cancellation"""
    
    cancel_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup progress widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                font-weight: bold;
                color: #60a3d9;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #5a5a5a;
                border-radius: 8px;
                text-align: center;
                background-color: #404040;
                color: #ffffff;
                font-weight: bold;
                height: 25px;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d7377, stop:1 #14a085);
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Details label
        self.details_label = QLabel("")
        self.details_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 9pt;
                margin-top: 5px;
            }
        """)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMaximumWidth(100)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.cancel_button.setVisible(False)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
    
    def set_status(self, status: str, details: str = "", show_cancel: bool = False):
        """Set progress status"""
        self.status_label.setText(status)
        self.details_label.setText(details)
        self.cancel_button.setVisible(show_cancel)
    
    def set_progress(self, value: int):
        """Set progress value (0-100)"""
        self.progress_bar.setValue(max(0, min(100, value)))
    
    def set_indeterminate(self, indeterminate: bool = True):
        """Set indeterminate progress"""
        if indeterminate:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
    
    def reset(self):
        """Reset progress widget"""
        self.status_label.setText("Ready")
        self.details_label.setText("")
        self.progress_bar.setValue(0)
        self.cancel_button.setVisible(False)
        self.set_indeterminate(False)


class OptionChainWidget(QWidget):
    """Widget for displaying option chain data"""
    
    option_selected = Signal(dict)  # Emits selected option data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.option_data = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup option chain widget UI"""
        layout = QVBoxLayout(self)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        header_label = QLabel("Option Chain")
        header_label.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #60a3d9;
            }
        """)
        
        # Expiration selector
        self.expiry_combo = QComboBox()
        self.expiry_combo.setMinimumWidth(120)
        self.expiry_combo.currentTextChanged.connect(self.load_expiry_data)
        
        # Strike filter
        self.strike_filter = QComboBox()
        self.strike_filter.addItems(["All Strikes", "ITM Only", "OTM Only", "Near ATM"])
        self.strike_filter.currentTextChanged.connect(self.apply_filters)
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Expiry:"))
        header_layout.addWidget(self.expiry_combo)
        header_layout.addWidget(QLabel("Filter:"))
        header_layout.addWidget(self.strike_filter)
        
        layout.addLayout(header_layout)
        
        # Option chain table
        self.create_option_table()
        layout.addWidget(self.option_table)
    
    def create_option_table(self):
        """Create option chain table"""
        from views.base_view import ProfessionalTable
        
        headers = [
            "Strike", "Call Bid", "Call Ask", "Call IV", "Call Volume",
            "Put Bid", "Put Ask", "Put IV", "Put Volume"
        ]
        
        self.option_table = ProfessionalTable(headers)
        self.option_table.row_selected.connect(self.on_option_selected)
        
        # Set column widths
        header = self.option_table.horizontalHeader()
        for i in range(len(headers)):
            header.resizeSection(i, 80)
    
    def load_option_data(self, symbol: str, option_data: Dict[str, Any]):
        """Load option chain data for symbol"""
        self.option_data = option_data
        
        # Update expiry combo
        self.expiry_combo.clear()
        if 'expirations' in option_data:
            self.expiry_combo.addItems(option_data['expirations'])
        
        # Load first expiry by default
        if self.expiry_combo.count() > 0:
            self.load_expiry_data(self.expiry_combo.currentText())
    
    def load_expiry_data(self, expiry: str):
        """Load data for specific expiry"""
        if not expiry or 'chains' not in self.option_data:
            return
        
        chains = self.option_data['chains']
        if expiry not in chains:
            return
        
        chain_data = chains[expiry]
        calls = chain_data.get('calls', [])
        puts = chain_data.get('puts', [])
        
        # Clear existing data
        self.option_table.clear_table()
        
        # Combine calls and puts by strike
        strikes = {}
        
        for call in calls:
            strike = call.get('strike', 0)
            strikes[strike] = strikes.get(strike, {})
            strikes[strike]['call'] = call
        
        for put in puts:
            strike = put.get('strike', 0)
            strikes[strike] = strikes.get(strike, {})
            strikes[strike]['put'] = put
        
        # Add rows to table
        for strike in sorted(strikes.keys()):
            self.add_option_row(strike, strikes[strike])
    
    def add_option_row(self, strike: float, option_data: Dict[str, Any]):
        """Add option row to table"""
        call_data = option_data.get('call', {})
        put_data = option_data.get('put', {})
        
        row_data = [
            f"${strike:.2f}",
            f"${call_data.get('bid', 0):.2f}",
            f"${call_data.get('ask', 0):.2f}",
            f"{call_data.get('impliedVolatility', 0)*100:.1f}%",
            str(call_data.get('volume', 0)),
            f"${put_data.get('bid', 0):.2f}",
            f"${put_data.get('ask', 0):.2f}",
            f"{put_data.get('impliedVolatility', 0)*100:.1f}%",
            str(put_data.get('volume', 0))
        ]
        
        self.option_table.add_row(row_data, row_data={
            'strike': strike,
            'call': call_data,
            'put': put_data
        })
    
    def apply_filters(self):
        """Apply strike filters to table"""
        # Implement filtering logic based on self.strike_filter.currentText()
        pass
    
    def on_option_selected(self, row: int):
        """Handle option selection"""
        row_data = self.option_table.get_selected_row_data()
        if row_data:
            self.option_selected.emit(row_data)


class RiskManagementWidget(QWidget):
    """Widget for risk management controls and display"""
    
    risk_parameters_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup risk management widget UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Risk Management")
        header.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #60a3d9;
                padding: 8px 0px;
                border-bottom: 2px solid #4a4a4a;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(header)
        
        # Risk parameters
        params_layout = QGridLayout()
        
        # Portfolio value
        params_layout.addWidget(QLabel("Portfolio Value:"), 0, 0)
        self.portfolio_input = QSpinBox()
        self.portfolio_input.setRange(1000, 50000000)
        self.portfolio_input.setValue(100000)
        self.portfolio_input.setSuffix(" $")
        self.portfolio_input.valueChanged.connect(self.update_risk_calculations)
        params_layout.addWidget(self.portfolio_input, 0, 1)
        
        # Max risk per trade
        params_layout.addWidget(QLabel("Max Risk per Trade:"), 0, 2)
        self.max_risk_input = QDoubleSpinBox()
        self.max_risk_input.setRange(0.1, 10.0)
        self.max_risk_input.setValue(2.0)
        self.max_risk_input.setDecimals(1)
        self.max_risk_input.setSuffix("%")
        self.max_risk_input.valueChanged.connect(self.update_risk_calculations)
        params_layout.addWidget(self.max_risk_input, 0, 3)
        
        # Position size
        params_layout.addWidget(QLabel("Position Size:"), 1, 0)
        self.position_size_label = QLabel("1 contract")
        self.position_size_label.setStyleSheet("font-weight: bold; color: #28a745;")
        params_layout.addWidget(self.position_size_label, 1, 1)
        
        # Max dollar risk
        params_layout.addWidget(QLabel("Max Dollar Risk:"), 1, 2)
        self.max_dollar_risk_label = QLabel("$2,000")
        self.max_dollar_risk_label.setStyleSheet("font-weight: bold; color: #dc3545;")
        params_layout.addWidget(self.max_dollar_risk_label, 1, 3)
        
        layout.addLayout(params_layout)
        
        # Risk visualization
        self.create_risk_visualization()
        layout.addWidget(self.risk_viz_group)
        
        # Initialize calculations
        self.update_risk_calculations()
    
    def create_risk_visualization(self):
        """Create risk visualization section"""
        self.risk_viz_group = QGroupBox("Risk Visualization")
        viz_layout = QVBoxLayout(self.risk_viz_group)
        
        # Risk meter
        self.risk_meter = QProgressBar()
        self.risk_meter.setRange(0, 100)
        self.risk_meter.setValue(20)  # 2% default
        self.risk_meter.setFormat("Risk Level: %p%")
        self.risk_meter.setStyleSheet("""
            QProgressBar {
                border: 2px solid #5a5a5a;
                border-radius: 8px;
                text-align: center;
                background-color: #404040;
                color: #ffffff;
                font-weight: bold;
                height: 25px;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #28a745, stop:0.5 #ffc107, stop:1 #dc3545);
                border-radius: 6px;
            }
        """)
        viz_layout.addWidget(self.risk_meter)
        
        # Risk breakdown
        breakdown_layout = QGridLayout()
        
        self.current_risk_label = QLabel("Current Risk: 2.0%")
        self.current_risk_label.setStyleSheet("color: #28a745; font-weight: bold;")
        breakdown_layout.addWidget(self.current_risk_label, 0, 0)
        
        self.risk_capacity_label = QLabel("Remaining Capacity: 8.0%")
        self.risk_capacity_label.setStyleSheet("color: #60a3d9; font-weight: bold;")
        breakdown_layout.addWidget(self.risk_capacity_label, 0, 1)
        
        viz_layout.addLayout(breakdown_layout)
    
    def update_risk_calculations(self):
        """Update risk calculations and display"""
        portfolio_value = self.portfolio_input.value()
        max_risk_pct = self.max_risk_input.value()
        
        # Calculate max dollar risk
        max_dollar_risk = portfolio_value * (max_risk_pct / 100)
        
        # Update displays
        self.max_dollar_risk_label.setText(f"${max_dollar_risk:,.0f}")
        self.risk_meter.setValue(int(max_risk_pct * 10))  # Scale to 0-100
        
        # Update risk labels
        self.current_risk_label.setText(f"Current Risk: {max_risk_pct:.1f}%")
        remaining_capacity = 10.0 - max_risk_pct  # Assume 10% max total risk
        self.risk_capacity_label.setText(f"Remaining Capacity: {remaining_capacity:.1f}%")
        
        # Calculate recommended position size (placeholder logic)
        # This would integrate with actual option pricing
        estimated_loss_per_contract = 200  # Placeholder
        recommended_contracts = max(1, int(max_dollar_risk / estimated_loss_per_contract))
        self.position_size_label.setText(f"{recommended_contracts} contracts")
        
        # Emit parameter changes
        risk_params = {
            'portfolio_value': portfolio_value,
            'max_risk_pct': max_risk_pct,
            'max_dollar_risk': max_dollar_risk,
            'recommended_contracts': recommended_contracts
        }
        self.risk_parameters_changed.emit(risk_params)
    
    def set_trade_risk(self, risk_per_contract: float, num_contracts: int):
        """Set trade-specific risk parameters"""
        total_risk = risk_per_contract * num_contracts
        portfolio_value = self.portfolio_input.value()
        risk_pct = (total_risk / portfolio_value) * 100
        
        # Update current risk display
        self.current_risk_label.setText(f"Current Trade Risk: {risk_pct:.1f}%")
        
        # Update risk meter to show actual vs max
        max_risk_pct = self.max_risk_input.value()
        risk_ratio = min(100, (risk_pct / max_risk_pct) * 100)
        self.risk_meter.setValue(int(risk_ratio))
        
        # Color code based on risk level
        if risk_pct <= max_risk_pct * 0.5:
            color = "#28a745"  # Green - low risk
        elif risk_pct <= max_risk_pct:
            color = "#ffc107"  # Yellow - moderate risk
        else:
            color = "#dc3545"  # Red - high risk
        
        self.current_risk_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        