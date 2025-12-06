"""
Main Window View - Professional Options Calculator Pro
Primary analysis interface with symbol input and parameter controls
"""

import logging
from typing import List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QGroupBox, QFrame, QTextEdit,
    QCompleter, QScrollArea, QSplitter
)
from PySide6.QtCore import Qt, Signal, QTimer, QStringListModel
from PySide6.QtGui import QFont, QPalette, QValidator, QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression

from utils.config_manager import ConfigManager


class SymbolValidator(QValidator):
    """Custom validator for stock symbols"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Allow letters, numbers, dots, and spaces/commas for multiple symbols
        self.regex = QRegularExpression(r'^[A-Za-z0-9.,\s\-]*$')
    
    def validate(self, input_str: str, pos: int):
        if self.regex.match(input_str).hasMatch():
            return QValidator.Acceptable, input_str, pos
        return QValidator.Invalid, input_str, pos


class ProfessionalGroupBox(QGroupBox):
    """Enhanced group box with professional styling"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)


class AdvancedParameterWidget(QWidget):
    """Widget for advanced trading parameters"""
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup the advanced parameters UI"""
        layout = QVBoxLayout(self)
        
        # Trading Parameters Group
        trading_group = ProfessionalGroupBox("Trading Parameters")
        trading_layout = QGridLayout(trading_group)
        
        # Contracts
        trading_layout.addWidget(QLabel("Contracts:"), 0, 0)
        self.contracts_spin = QSpinBox()
        self.contracts_spin.setRange(1, 100)
        self.contracts_spin.setValue(1)
        self.contracts_spin.setToolTip("Number of option contracts to trade")
        trading_layout.addWidget(self.contracts_spin, 0, 1)
        
        # Portfolio Value
        trading_layout.addWidget(QLabel("Portfolio Value:"), 0, 2)
        self.portfolio_spin = QDoubleSpinBox()
        self.portfolio_spin.setRange(1000, 10000000)
        self.portfolio_spin.setPrefix("$")
        self.portfolio_spin.setDecimals(0)
        self.portfolio_spin.setValue(100000)
        self.portfolio_spin.setToolTip("Total portfolio value for position sizing")
        trading_layout.addWidget(self.portfolio_spin, 0, 3)
        
        # Max Position Risk
        trading_layout.addWidget(QLabel("Max Risk %:"), 1, 0)
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 10.0)
        self.risk_spin.setDecimals(1)
        self.risk_spin.setSuffix("%")
        self.risk_spin.setValue(2.0)
        self.risk_spin.setToolTip("Maximum percentage of portfolio to risk per trade")
        trading_layout.addWidget(self.risk_spin, 1, 1)
        
        # Debit Override
        trading_layout.addWidget(QLabel("Debit Override:"), 1, 2)
        self.debit_spin = QDoubleSpinBox()
        self.debit_spin.setRange(0.0, 50.0)
        self.debit_spin.setPrefix("$")
        self.debit_spin.setDecimals(2)
        self.debit_spin.setValue(0.0)
        self.debit_spin.setSpecialValueText("Auto")
        self.debit_spin.setToolTip("Manual debit override (0 = automatic calculation)")
        trading_layout.addWidget(self.debit_spin, 1, 3)
        
        layout.addWidget(trading_group)
        
        # Analysis Parameters Group
        analysis_group = ProfessionalGroupBox("Analysis Parameters")
        analysis_layout = QGridLayout(analysis_group)
        
        # Monte Carlo Simulations
        analysis_layout.addWidget(QLabel("MC Simulations:"), 0, 0)
        self.mc_spin = QSpinBox()
        self.mc_spin.setRange(1000, 100000)
        self.mc_spin.setSingleStep(1000)
        self.mc_spin.setValue(10000)
        self.mc_spin.setToolTip("Number of Monte Carlo simulations")
        analysis_layout.addWidget(self.mc_spin, 0, 1)
        
        # Volatility Lookback
        analysis_layout.addWidget(QLabel("Vol Lookback:"), 0, 2)
        self.vol_spin = QSpinBox()
        self.vol_spin.setRange(7, 252)
        self.vol_spin.setSuffix(" days")
        self.vol_spin.setValue(30)
        self.vol_spin.setToolTip("Lookback period for volatility calculation")
        analysis_layout.addWidget(self.vol_spin, 0, 3)
        
        # Min Confidence Threshold
        analysis_layout.addWidget(QLabel("Min Confidence:"), 1, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 100.0)
        self.confidence_spin.setSuffix("%")
        self.confidence_spin.setValue(50.0)
        self.confidence_spin.setToolTip("Minimum confidence threshold for recommendations")
        analysis_layout.addWidget(self.confidence_spin, 1, 1)
        
        # Use ML Predictions
        self.ml_checkbox = QCheckBox("Use ML Predictions")
        self.ml_checkbox.setChecked(True)
        self.ml_checkbox.setToolTip("Include machine learning predictions in analysis")
        analysis_layout.addWidget(self.ml_checkbox, 1, 2, 1, 2)
        
        layout.addWidget(analysis_group)
    
    def _load_settings(self):
        """Load settings from configuration"""
        try:
            # Trading parameters
            self.contracts_spin.setValue(
                self.config_manager.get("trading.default_contracts", 1)
            )
            self.portfolio_spin.setValue(
                self.config_manager.get("trading.portfolio_value", 100000.0)
            )
            self.risk_spin.setValue(
                self.config_manager.get("trading.max_position_risk", 0.02) * 100
            )
            
            # Analysis parameters
            self.mc_spin.setValue(
                self.config_manager.get("analysis.monte_carlo_simulations", 10000)
            )
            self.vol_spin.setValue(
                self.config_manager.get("analysis.volatility_lookback_days", 30)
            )
            self.confidence_spin.setValue(
                self.config_manager.get("trading.min_confidence_threshold", 50.0)
            )
            self.ml_checkbox.setChecked(
                self.config_manager.get("ml.enabled", True)
            )
            
        except Exception as e:
            self.logger.warning(f"Error loading advanced parameter settings: {e}")
    
    def save_settings(self):
        """Save current settings to configuration"""
        try:
            # Trading parameters
            self.config_manager.set("trading.default_contracts", self.contracts_spin.value())
            self.config_manager.set("trading.portfolio_value", self.portfolio_spin.value())
            self.config_manager.set("trading.max_position_risk", self.risk_spin.value() / 100)
            
            # Analysis parameters
            self.config_manager.set("analysis.monte_carlo_simulations", self.mc_spin.value())
            self.config_manager.set("analysis.volatility_lookback_days", self.vol_spin.value())
            self.config_manager.set("trading.min_confidence_threshold", self.confidence_spin.value())
            self.config_manager.set("ml.enabled", self.ml_checkbox.isChecked())
            
            self.config_manager.save()
            
        except Exception as e:
            self.logger.error(f"Error saving advanced parameter settings: {e}")
    
    def get_parameters(self) -> dict:
        """Get current parameter values"""
        return {
            'contracts': self.contracts_spin.value(),
            'portfolio_value': self.portfolio_spin.value(),
            'max_position_risk': self.risk_spin.value() / 100,
            'debit_override': self.debit_spin.value() if self.debit_spin.value() > 0 else None,
            'monte_carlo_sims': self.mc_spin.value(),
            'volatility_lookback': self.vol_spin.value(),
            'min_confidence': self.confidence_spin.value(),
            'use_ml': self.ml_checkbox.isChecked()
        }


class MarketOverviewWidget(QWidget):
    """Widget showing current market conditions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        self._setup_ui()
        self._setup_timer()
    
    def _setup_ui(self):
        """Setup market overview UI"""
        layout = QVBoxLayout(self)
        
        # Market Overview Group
        market_group = ProfessionalGroupBox("Market Overview")
        market_layout = QGridLayout(market_group)
        
        # VIX
        market_layout.addWidget(QLabel("VIX:"), 0, 0)
        self.vix_label = QLabel("Loading...")
        self.vix_label.setStyleSheet("font-weight: bold;")
        market_layout.addWidget(self.vix_label, 0, 1)
        
        # Market Status
        market_layout.addWidget(QLabel("Market:"), 0, 2)
        self.market_status_label = QLabel("Checking...")
        market_layout.addWidget(self.market_status_label, 0, 3)
        
        # SPX Level
        market_layout.addWidget(QLabel("SPX:"), 1, 0)
        self.spx_label = QLabel("Loading...")
        self.spx_label.setStyleSheet("font-weight: bold;")
        market_layout.addWidget(self.spx_label, 1, 1)
        
        # NDX Level
        market_layout.addWidget(QLabel("NDX:"), 1, 2)
        self.ndx_label = QLabel("Loading...")
        self.ndx_label.setStyleSheet("font-weight: bold;")
        market_layout.addWidget(self.ndx_label, 1, 3)
        
        layout.addWidget(market_group)
        
        # Set fixed height
        self.setMaximumHeight(120)
    
    def _setup_timer(self):
        """Setup update timer for market data"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_market_data)
        self.update_timer.start(60000)  # Update every minute
        
        # Initial update
        QTimer.singleShot(1000, self.update_market_data)
    
    def update_market_data(self):
        """Update market data display"""
        try:
            # This would be connected to actual market data service
            # For now, showing placeholder
            self.vix_label.setText("Loading...")
            self.market_status_label.setText("Loading...")
            self.spx_label.setText("Loading...")
            self.ndx_label.setText("Loading...")
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")


class MainWindowView(QWidget):
    """
    Main analysis interface with symbol input and controls
    """
    
    # Signals
    analysis_requested = Signal(str, int, float)  # symbol, contracts, debit
    batch_analysis_requested = Signal(list, dict)  # symbols, parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()
        
        # UI components
        self.symbol_input = None
        self.analyze_button = None
        self.batch_analyze_button = None
        self.advanced_params = None
        self.market_overview = None
        self.status_display = None
        
        # Symbol completion
        self.symbol_completer = None
        self.favorite_symbols = []
        
        # Advanced mode toggle
        self.advanced_mode = False
        
        self._setup_ui()
        self._setup_connections()
        self._load_favorites()
        
        self.logger.debug("MainWindowView initialized")
    
    def _setup_ui(self):
        """Setup the main UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Market Overview (always visible)
        self.market_overview = MarketOverviewWidget()
        main_layout.addWidget(self.market_overview)
        
        # Main Analysis Section
        analysis_frame = QFrame()
        analysis_frame.setFrameStyle(QFrame.StyledPanel)
        analysis_layout = QVBoxLayout(analysis_frame)
        
        # Title
        title_label = QLabel("Professional Options Analysis")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        analysis_layout.addWidget(title_label)
        
        # Symbol Input Section
        symbol_group = ProfessionalGroupBox("Symbol Analysis")
        symbol_layout = QVBoxLayout(symbol_group)
        
        # Symbol input with autocomplete
        input_layout = QHBoxLayout()
        
        input_layout.addWidget(QLabel("Stock Symbol(s):"))
        
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol (e.g., AAPL) or multiple symbols (AAPL, MSFT, GOOGL)")
        self.symbol_input.setValidator(SymbolValidator())
        self.symbol_input.setMinimumHeight(35)
        
        # Setup autocomplete
        self._setup_symbol_completion()
        
        input_layout.addWidget(self.symbol_input)
        
        # Quick symbol buttons
        quick_symbols_layout = QHBoxLayout()
        quick_symbols_layout.addWidget(QLabel("Quick Select:"))
        
        default_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "SPY", "QQQ"]
        for symbol in default_symbols:
            btn = QPushButton(symbol)
            btn.setMaximumWidth(60)
            btn.clicked.connect(lambda checked, s=symbol: self.symbol_input.setText(s))
            quick_symbols_layout.addWidget(btn)
        
        quick_symbols_layout.addStretch()
        
        symbol_layout.addLayout(input_layout)
        symbol_layout.addLayout(quick_symbols_layout)
        
        analysis_layout.addWidget(symbol_group)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("Analyze Options")
        self.analyze_button.setMinimumHeight(40)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.batch_analyze_button = QPushButton("Batch Analysis")
        self.batch_analyze_button.setMinimumHeight(40)
        
        self.advanced_toggle_button = QPushButton("Advanced Parameters ▼")
        self.advanced_toggle_button.setMinimumHeight(40)
        
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.batch_analyze_button)
        button_layout.addStretch()
        button_layout.addWidget(self.advanced_toggle_button)
        
        analysis_layout.addLayout(button_layout)
        
        main_layout.addWidget(analysis_frame)
        
        # Advanced Parameters (initially hidden)
        self.advanced_params = AdvancedParameterWidget(self.config_manager)
        self.advanced_params.setVisible(False)
        main_layout.addWidget(self.advanced_params)
        
        # Status Display
        self._setup_status_display(main_layout)
        
        # Add stretch to push everything to top
        main_layout.addStretch()
    
    def _setup_symbol_completion(self):
        """Setup symbol autocomplete functionality"""
        try:
            # Load symbol list (this could be loaded from a file or API)
            common_symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD",
                "INTC", "NFLX", "CRM", "ADBE", "PYPL", "SHOP", "ZOOM", "ROKU",
                "SQ", "SPOT", "UBER", "LYFT", "SNAP", "TWTR", "PINS", "DOCU",
                "SPY", "QQQ", "IWM", "DIA", "VTI", "VXX", "UVXY", "SQQQ"
            ]
            
            # Add favorite symbols
            common_symbols.extend(self.favorite_symbols)
            
            # Remove duplicates and sort
            symbols = sorted(list(set(common_symbols)))
            
            # Create completer
            model = QStringListModel(symbols)
            self.symbol_completer = QCompleter(model)
            self.symbol_completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.symbol_completer.setCompletionMode(QCompleter.InlineCompletion)
            
            self.symbol_input.setCompleter(self.symbol_completer)
            
        except Exception as e:
            self.logger.warning(f"Error setting up symbol completion: {e}")
    
    def _setup_status_display(self, parent_layout):
        """Setup status display area"""
        status_group = ProfessionalGroupBox("Analysis Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(150)
        self.status_display.setReadOnly(True)
        self.status_display.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
        """)
        
        # Initial status message
        self.status_display.setText(
            "Professional Options Calculator Pro v10.0\n"
            "Ready for analysis. Enter a stock symbol and click 'Analyze Options'.\n\n"
            "Features:\n"
            "• Real-time options analysis with Monte Carlo simulations\n"
            "• Machine learning predictions based on historical data\n"
            "• Advanced Greeks calculations and risk management\n"
            "• Professional-grade reporting and visualization\n"
        )
        
        status_layout.addWidget(self.status_display)
        parent_layout.addWidget(status_group)
    
    def _setup_connections(self):
        """Setup signal connections"""
        # Button connections
        self.analyze_button.clicked.connect(self._on_analyze_clicked)
        self.batch_analyze_button.clicked.connect(self._on_batch_analyze_clicked)
        self.advanced_toggle_button.clicked.connect(self._toggle_advanced_mode)
        
        # Enter key in symbol input
        self.symbol_input.returnPressed.connect(self._on_analyze_clicked)
        
        # Symbol input validation
        self.symbol_input.textChanged.connect(self._on_symbol_text_changed)
    
    def _load_favorites(self):
        """Load favorite symbols from configuration"""
        try:
            self.favorite_symbols = self.config_manager.get("trading.favorite_symbols", [])
            self.logger.debug(f"Loaded {len(self.favorite_symbols)} favorite symbols")
        except Exception as e:
            self.logger.warning(f"Error loading favorite symbols: {e}")
            self.favorite_symbols = []
    
    def _on_symbol_text_changed(self, text):
        """Handle symbol input text changes"""
        # Enable/disable analyze button based on input
        has_text = bool(text.strip())
        self.analyze_button.setEnabled(has_text)
        self.batch_analyze_button.setEnabled(has_text and ',' in text)
    
    def _on_analyze_clicked(self):
        """Handle analyze button click"""
        try:
            symbols_text = self.symbol_input.text().strip().upper()
            
            if not symbols_text:
                self._update_status("Please enter a stock symbol.")
                return
            
            # Parse symbols
            symbols = [s.strip() for s in symbols_text.replace(',', ' ').split() if s.strip()]
            
            if not symbols:
                self._update_status("Please enter a valid stock symbol.")
                return
            
            # Get parameters
            params = self.advanced_params.get_parameters() if self.advanced_mode else {}
            contracts = params.get('contracts', 1)
            debit = params.get('debit_override', 0.0)
            
            if len(symbols) == 1:
                # Single symbol analysis
                symbol = symbols[0]
                self._update_status(f"Starting analysis for {symbol}...")
                self.analysis_requested.emit(symbol, contracts, debit)
            else:
                # Batch analysis
                self._update_status(f"Starting batch analysis for {len(symbols)} symbols...")
                self.batch_analysis_requested.emit(symbols, params)
                
        except Exception as e:
            self.logger.error(f"Error in analyze clicked: {e}")
            self._update_status(f"Error: {e}")
    
    def _on_batch_analyze_clicked(self):
        """Handle batch analyze button click"""
        try:
            symbols_text = self.symbol_input.text().strip().upper()
            
            if not symbols_text:
                self._update_status("Please enter stock symbols separated by commas.")
                return
            
            # Parse symbols
            symbols = [s.strip() for s in symbols_text.replace(',', ' ').split() if s.strip()]
            
            if len(symbols) < 2:
                self._update_status("Please enter multiple symbols for batch analysis.")
                return
            
            # Get parameters
            params = self.advanced_params.get_parameters()
            
            self._update_status(f"Starting batch analysis for {len(symbols)} symbols...")
            self.batch_analysis_requested.emit(symbols, params)
            
        except Exception as e:
            self.logger.error(f"Error in batch analyze: {e}")
            self._update_status(f"Error: {e}")
    
    def _toggle_advanced_mode(self):
        """Toggle advanced parameters visibility"""
        self.advanced_mode = not self.advanced_mode
        self.advanced_params.setVisible(self.advanced_mode)
        
        # Update button text
        arrow = "▲" if self.advanced_mode else "▼"
        self.advanced_toggle_button.setText(f"Advanced Parameters {arrow}")
        
        # Save settings if shown
        if self.advanced_mode:
            self.advanced_params.save_settings()
    
    def _update_status(self, message: str):
        """Update status display with new message"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
            
            self.status_display.append(formatted_message)
            
            # Auto-scroll to bottom
            cursor = self.status_display.textCursor()
            cursor.movePosition(cursor.End)
            self.status_display.setTextCursor(cursor)
            
        except Exception as e:
            self.logger.error(f"Error updating status: {e}")
    
    def clear_status(self):
        """Clear the status display"""
        self.status_display.clear()
    
    def trigger_analysis(self):
        """Trigger analysis from external source (toolbar, etc.)"""
        if self.symbol_input.text().strip():
            self._on_analyze_clicked()
        else:
            self._update_status("Please enter a stock symbol first.")
    
    def set_symbol(self, symbol: str):
        """Set symbol in input field"""
        self.symbol_input.setText(symbol.upper())
    
    def add_favorite_symbol(self, symbol: str):
        """Add symbol to favorites"""
        symbol = symbol.upper()
        if symbol not in self.favorite_symbols:
            self.favorite_symbols.append(symbol)
            self.config_manager.set("trading.favorite_symbols", self.favorite_symbols)
            self.config_manager.save()
            
            # Update autocomplete
            self._setup_symbol_completion()
            
            self._update_status(f"Added {symbol} to favorites.")
    
    def get_current_symbol(self) -> str:
        """Get currently entered symbol"""
        return self.symbol_input.text().strip().upper()
    
    def get_current_symbols(self) -> List[str]:
        """Get list of currently entered symbols"""
        symbols_text = self.symbol_input.text().strip().upper()
        return [s.strip() for s in symbols_text.replace(',', ' ').split() if s.strip()]
    
    def is_advanced_mode(self) -> bool:
        """Check if advanced mode is enabled"""
        return self.advanced_mode
    
    def get_analysis_parameters(self) -> dict:
        """Get current analysis parameters"""
        if self.advanced_mode:
            return self.advanced_params.get_parameters()
        else:
            return {
                'contracts': 1,
                'portfolio_value': 100000.0,
                'max_position_risk': 0.02,
                'debit_override': None,
                'monte_carlo_sims': 10000,
                'volatility_lookback': 30,
                'min_confidence': 50.0,
                'use_ml': True
            }

