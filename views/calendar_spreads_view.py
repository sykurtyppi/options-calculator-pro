"""
Streamlined Calendar Spreads View - Options Calculator Pro
=========================================================

Clean calendar spreads interface with exact requirements:
- QSplitter: left (Trade Setup in QScrollArea) + right (Results)
- Right panel: Results Summary strip + QTabWidget (Payoff/Candidates/Greeks)
- Single accent color (#0ea5e9), subtle borders (#2d2d2d), muted text (#94a3b8)
- No neon multi-colored outlines

Author: Professional Trading Tools
Version: 3.0.0 (Streamlined Redesign)
"""

import logging
import re
import csv
from datetime import datetime
from typing import Dict, Optional, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QFrame, QCompleter, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QTabWidget, QSplitter, QScrollArea,
    QTextEdit, QFileDialog, QApplication
)
from PySide6.QtCore import Qt, Signal, QStringListModel, Slot, QTimer
from PySide6.QtGui import QFont, QBrush, QColor

from views.design_system import (
    app_font,
    app_theme_stylesheet,
    button_style as ds_button_style,
    input_style as ds_input_style,
    panel_style as ds_panel_style,
    status_strip_style as ds_status_strip_style,
    table_style as ds_table_style,
)


class CalendarSpreadsView(QWidget):
    """
    Streamlined Calendar Spreads view with QSplitter design:
    - Left: Trade Setup (scrollable)
    - Right: Results Summary + QTabWidget (Payoff/Candidates/Greeks)
    """

    # Signals for backend integration
    symbol_analysis_requested = Signal(str)
    trade_calculation_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Service references (will be set via set_services)
        self.market_data_service = None
        self.options_service = None
        self.ml_service = None
        self.volatility_service = None
        self.greeks_calculator = None
        self._active_symbol = ""
        self._analysis_feedback_symbol = ""
        self._analysis_feedback_dots = 0

        self._analysis_feedback_timer = QTimer(self)
        self._analysis_feedback_timer.setInterval(450)
        self._analysis_feedback_timer.timeout.connect(self._update_analysis_feedback)

        self._setup_ui()
        self._apply_consistent_styling()

        self.logger.info("Streamlined Calendar Spreads view initialized")

    def _setup_ui(self):
        """Setup the streamlined QSplitter UI structure"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create QSplitter for left/right layout
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Panel: Trade Setup (scrollable)
        self._setup_left_panel()

        # Right Panel: Results Summary + QTabWidget
        self._setup_right_panel()

        # Set splitter proportions: left 30%, right 70%
        self.splitter.setStretchFactor(0, 30)
        self.splitter.setStretchFactor(1, 70)
        self.splitter.setSizes([350, 800])

        main_layout.addWidget(self.splitter)

    def _setup_left_panel(self):
        """Setup scrollable left panel with trade setup controls"""
        # Create QScrollArea for left panel
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.left_scroll.setMinimumWidth(300)
        self.left_scroll.setMaximumWidth(400)

        # Create content widget for scrollable area
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title_label = QLabel("Trade Setup")
        title_label.setFont(app_font(16, bold=True))
        title_label.setStyleSheet("color: #0ea5e9; margin-bottom: 10px;")
        left_layout.addWidget(title_label)

        # Symbol input section
        symbol_frame = self._create_input_section("Symbol", self._create_symbol_input())
        left_layout.addWidget(symbol_frame)

        # Strategy parameters section
        self.strategy_frame = self._create_input_section("Strategy Parameters", self._create_strategy_inputs())
        left_layout.addWidget(self.strategy_frame)
        self.strategy_frame.setEnabled(False)

        # Advanced controls (hidden until needed)
        self.advanced_toggle_btn = QPushButton("Show Advanced Controls")
        self.advanced_toggle_btn.setCheckable(True)
        self.advanced_toggle_btn.setChecked(False)
        self.advanced_toggle_btn.setFixedHeight(30)
        self.advanced_toggle_btn.setStyleSheet(self._get_tertiary_button_style())
        self.advanced_toggle_btn.setToolTip("Reveal position sizing and debit override controls.")
        self.advanced_toggle_btn.toggled.connect(self._toggle_advanced_controls)
        left_layout.addWidget(self.advanced_toggle_btn)

        self.advanced_controls_frame = QFrame()
        self.advanced_controls_frame.setStyleSheet(ds_panel_style())
        advanced_layout = QVBoxLayout(self.advanced_controls_frame)
        advanced_layout.setSpacing(10)
        advanced_layout.setContentsMargins(10, 10, 10, 10)

        self.position_frame = self._create_input_section("Position Sizing", self._create_position_inputs())
        self.debit_frame = self._create_input_section("Debit Override", self._create_debit_inputs())
        advanced_layout.addWidget(self.position_frame)
        advanced_layout.addWidget(self.debit_frame)

        self.advanced_controls_frame.setVisible(False)
        left_layout.addWidget(self.advanced_controls_frame)

        # Action buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)

        self.calculate_btn = QPushButton("Calculate Spread")
        self.calculate_btn.setStyleSheet(self._get_primary_button_style())
        self.calculate_btn.setFixedHeight(36)
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setToolTip("Validate structure and estimate debit/risk profile.")
        self.calculate_btn.clicked.connect(self._calculate_spread)
        buttons_layout.addWidget(self.calculate_btn)

        self.analyze_btn = QPushButton("Run Live Analysis")
        self.analyze_btn.setStyleSheet(self._get_secondary_button_style())
        self.analyze_btn.setFixedHeight(32)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setToolTip("Run live market/options analysis for the selected symbol.")
        self.analyze_btn.clicked.connect(self._analyze_symbol)
        buttons_layout.addWidget(self.analyze_btn)

        self.reset_btn = QPushButton("Reset Setup")
        self.reset_btn.setStyleSheet(self._get_tertiary_button_style())
        self.reset_btn.setFixedHeight(28)
        self.reset_btn.setToolTip("Clear inputs and return to default setup values.")
        self.reset_btn.clicked.connect(self._reset_setup_form)
        buttons_layout.addWidget(self.reset_btn)

        self.left_hint_label = QLabel("Enter a valid symbol (e.g., AAPL) to enable actions.")
        self.left_hint_label.setWordWrap(True)
        self.left_hint_label.setStyleSheet("color: #64748b; font-size: 11px;")
        buttons_layout.addWidget(self.left_hint_label)

        left_layout.addLayout(buttons_layout)

        # Add stretch to push content to top
        left_layout.addStretch()

        self.left_scroll.setWidget(left_content)
        self.splitter.addWidget(self.left_scroll)
        self._on_strategy_input_changed(announce_feedback=False)

    def _toggle_advanced_controls(self, visible: bool):
        """Show or hide advanced controls to reduce form density."""
        self.advanced_controls_frame.setVisible(visible)
        self.advanced_toggle_btn.setText(
            "Hide Advanced Controls" if visible else "Show Advanced Controls"
        )
        if visible:
            self.left_hint_label.setText(
                "Advanced controls are visible. Use them only if you need manual sizing/debit overrides."
            )
        else:
            self.left_hint_label.setText(
                "Enter a valid symbol (e.g., AAPL) to enable actions."
            )

    def _setup_right_panel(self):
        """Setup right panel with Results Summary + QTabWidget"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.feedback_strip = QLabel("Ready | Enter a symbol to start calendar spread analysis.")
        self.feedback_strip.setFixedHeight(32)
        self.feedback_strip.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.feedback_strip.setStyleSheet(
            ds_status_strip_style("info").replace("border-radius: 6px;", "border-radius: 0px;")
        )
        right_layout.addWidget(self.feedback_strip)

        self.context_strip = QLabel("No analysis yet | Configure setup and run analysis to populate live metrics.")
        self.context_strip.setFixedHeight(26)
        self.context_strip.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.context_strip.setStyleSheet(
            ds_status_strip_style("neutral").replace("border-radius: 6px;", "border-radius: 0px;")
        )
        right_layout.addWidget(self.context_strip)
        self.context_strip.setVisible(False)

        # Results Summary strip (KPI tiles)
        self.results_summary = self._create_results_summary()
        right_layout.addWidget(self.results_summary)

        # QTabWidget for detailed results
        self.results_tabs = QTabWidget()
        self._setup_results_tabs()
        right_layout.addWidget(self.results_tabs)

        self.splitter.addWidget(right_widget)

    def _create_input_section(self, title, content_widget):
        """Create a consistent input section with title and content"""
        frame = QFrame()
        frame.setStyleSheet(ds_panel_style())

        layout = QVBoxLayout(frame)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 10, 12, 10)

        # Section title
        title_label = QLabel(title)
        title_label.setFont(app_font(12, bold=True))
        title_label.setStyleSheet("color: #0ea5e9; margin-bottom: 5px;")
        layout.addWidget(title_label)

        # Content
        layout.addWidget(content_widget)

        return frame

    def _create_symbol_input(self):
        """Create symbol input with autocomplete"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)

        # Symbol input
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol (e.g., AAPL)")
        self.symbol_input.setStyleSheet(self._get_input_style())
        self.symbol_input.setToolTip("Ticker symbol: examples AAPL, BRK.B, BTC-USD")
        self.symbol_input.textChanged.connect(self._on_symbol_changed)
        self.symbol_input.returnPressed.connect(self._calculate_spread)

        # Setup autocomplete
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ", "IWM"]
        completer = QCompleter(QStringListModel(symbols))
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.symbol_input.setCompleter(completer)

        layout.addWidget(QLabel("Symbol:"))
        layout.addWidget(self.symbol_input)

        self.symbol_validation_label = QLabel("Format: 1-8 chars, letters/numbers (. and - allowed)")
        self.symbol_validation_label.setStyleSheet("color: #64748b; font-size: 10px;")
        layout.addWidget(self.symbol_validation_label)

        return widget

    def _create_strategy_inputs(self):
        """Create strategy parameter inputs"""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(8)

        # Near-term expiration
        layout.addWidget(QLabel("Near Exp:"), 0, 0)
        self.near_exp_combo = QComboBox()
        self.near_exp_combo.addItems(["Jan 31", "Feb 7", "Feb 14", "Feb 21", "Feb 28"])
        self.near_exp_combo.setStyleSheet(self._get_input_style())
        self.near_exp_combo.setToolTip("Short leg expiration (typically before earnings).")
        layout.addWidget(self.near_exp_combo, 0, 1)

        # Far-term expiration
        layout.addWidget(QLabel("Far Exp:"), 1, 0)
        self.far_exp_combo = QComboBox()
        self.far_exp_combo.addItems(["Feb 28", "Mar 7", "Mar 14", "Mar 21", "Mar 28"])
        self.far_exp_combo.setStyleSheet(self._get_input_style())
        self.far_exp_combo.setToolTip("Long leg expiration (must be later than near expiration).")
        layout.addWidget(self.far_exp_combo, 1, 1)

        # Strike selection
        layout.addWidget(QLabel("Strike:"), 2, 0)
        self.strike_combo = QComboBox()
        self.strike_combo.addItems(["ATM", "ATM+5", "ATM-5", "Custom"])
        self.strike_combo.setStyleSheet(self._get_input_style())
        self.strike_combo.setToolTip("Strike placement for both legs.")
        layout.addWidget(self.strike_combo, 2, 1)

        self.term_structure_label = QLabel("Term spacing: --")
        self.term_structure_label.setStyleSheet("color: #64748b; font-size: 10px;")
        layout.addWidget(self.term_structure_label, 3, 0, 1, 2)

        self.near_exp_combo.currentTextChanged.connect(self._on_strategy_input_changed)
        self.far_exp_combo.currentTextChanged.connect(self._on_strategy_input_changed)
        self.strike_combo.currentTextChanged.connect(self._on_strategy_input_changed)

        return widget

    def _create_position_inputs(self):
        """Create position sizing inputs"""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(8)

        # Contracts
        layout.addWidget(QLabel("Contracts:"), 0, 0)
        self.contracts_spin = QSpinBox()
        self.contracts_spin.setRange(1, 100)
        self.contracts_spin.setValue(1)
        self.contracts_spin.setStyleSheet(self._get_input_style())
        self.contracts_spin.setToolTip("Number of calendar spreads to simulate.")
        layout.addWidget(self.contracts_spin, 0, 1)

        # Max risk
        layout.addWidget(QLabel("Max Risk:"), 1, 0)
        self.max_risk_spin = QDoubleSpinBox()
        self.max_risk_spin.setRange(0, 10000)
        self.max_risk_spin.setValue(500)
        self.max_risk_spin.setPrefix("$")
        self.max_risk_spin.setStyleSheet(self._get_input_style())
        self.max_risk_spin.setToolTip("Hard risk cap for position sizing guidance.")
        layout.addWidget(self.max_risk_spin, 1, 1)

        return widget

    def _create_debit_inputs(self):
        """Create debit override inputs"""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(8)

        # Debit override
        layout.addWidget(QLabel("Debit Override:"), 0, 0)
        self.debit_spin = QDoubleSpinBox()
        self.debit_spin.setRange(0, 100)
        self.debit_spin.setValue(0)
        self.debit_spin.setPrefix("$")
        self.debit_spin.setDecimals(2)
        self.debit_spin.setStyleSheet(self._get_input_style())
        self.debit_spin.setToolTip("Set to 0 for automatic debit calculation")
        layout.addWidget(self.debit_spin, 0, 1)

        return widget

    def _create_results_summary(self):
        """Create results summary strip with KPI tiles"""
        frame = QFrame()
        frame.setFixedHeight(80)
        frame.setStyleSheet("""
            QFrame {
                background-color: #262626;
                border: 1px solid #2d2d2d;
                border-radius: 0px;
            }
        """)

        layout = QHBoxLayout(frame)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 10, 20, 10)

        # KPI tiles
        self.net_debit_tile = self._create_kpi_tile("Net Debit", "--")
        self.max_profit_tile = self._create_kpi_tile("Max Profit", "--")
        self.max_loss_tile = self._create_kpi_tile("Max Loss", "--")
        self.prob_profit_tile = self._create_kpi_tile("Prob. Profit", "--")
        self.tx_cost_tile = self._create_kpi_tile("Tx Cost", "--")

        layout.addWidget(self.net_debit_tile)
        layout.addWidget(self.max_profit_tile)
        layout.addWidget(self.max_loss_tile)
        layout.addWidget(self.prob_profit_tile)
        layout.addWidget(self.tx_cost_tile)
        layout.addStretch()

        return frame

    def _create_kpi_tile(self, label, value):
        """Create a KPI tile for results summary"""
        tile = QFrame()
        tile.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(tile)
        layout.setSpacing(2)
        layout.setContentsMargins(12, 8, 12, 8)

        # Label
        label_widget = QLabel(label)
        label_widget.setFont(QFont("Segoe UI", 9))
        label_widget.setStyleSheet("color: #94a3b8;")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_widget)

        # Value
        value_widget = QLabel(value)
        value_widget.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        value_widget.setStyleSheet("color: #ffffff;")
        value_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_widget)
        tile._value_label = value_widget

        return tile

    def _setup_results_tabs(self):
        """Setup QTabWidget with Payoff, Candidates, and Greeks tabs"""
        # Payoff Chart tab
        payoff_widget = QWidget()
        payoff_layout = QVBoxLayout(payoff_widget)
        payoff_placeholder = QLabel(
            "Payoff Chart\n\nRun Calculate Spread to render the payoff profile.\n"
            "Use this to verify max loss, break-even zones, and theta shape before entry."
        )
        payoff_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        payoff_placeholder.setStyleSheet("color: #94a3b8; padding: 40px; font-size: 14px;")
        payoff_layout.addWidget(payoff_placeholder)
        self.results_tabs.addTab(payoff_widget, "Payoff Chart")

        # Candidate Spreads tab
        candidates_widget = QWidget()
        candidates_layout = QVBoxLayout(candidates_widget)
        self.candidates_table = QTableWidget(0, 7)
        self.candidates_table.setHorizontalHeaderLabels(
            ["Symbol", "Strike", "Near Exp", "Far Exp", "Debit", "Edge", "Confidence"]
        )
        self.candidates_table.verticalHeader().setVisible(False)
        self.candidates_table.setAlternatingRowColors(True)
        self.candidates_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.candidates_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.candidates_table.horizontalHeader().setStretchLastSection(True)
        self.candidates_table.setSortingEnabled(True)
        self.candidates_table.setStyleSheet(self._get_table_style())
        candidates_layout.addWidget(self.candidates_table)

        self.edge_rationale = QTextEdit()
        self.edge_rationale.setReadOnly(True)
        self.edge_rationale.setMinimumHeight(100)
        self.edge_rationale.setPlaceholderText(
            "Edge rationale will appear here after analysis.\n"
            "Expect IV term-structure, crush confidence, and risk notes."
        )
        self.edge_rationale.setStyleSheet("""
            QTextEdit {
                background-color: #111111;
                border: 1px solid #2d2d2d;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        candidates_layout.addWidget(self.edge_rationale)

        export_actions = QHBoxLayout()
        export_actions.setSpacing(8)

        self.export_candidates_btn = QPushButton("Export Candidates CSV")
        self.export_candidates_btn.setStyleSheet(self._get_secondary_button_style())
        self.export_candidates_btn.setToolTip("Export current candidate table to CSV.")
        self.export_candidates_btn.setEnabled(False)
        self.export_candidates_btn.clicked.connect(self._export_candidates_csv)
        export_actions.addWidget(self.export_candidates_btn)

        self.copy_rationale_btn = QPushButton("Copy Rationale")
        self.copy_rationale_btn.setStyleSheet(self._get_secondary_button_style())
        self.copy_rationale_btn.setToolTip("Copy edge rationale text to clipboard.")
        self.copy_rationale_btn.setEnabled(False)
        self.copy_rationale_btn.clicked.connect(self._copy_edge_rationale)
        export_actions.addWidget(self.copy_rationale_btn)

        export_actions.addStretch()
        candidates_layout.addLayout(export_actions)
        self.results_tabs.addTab(candidates_widget, "Candidates")

        # Greeks/Exposure tab
        greeks_widget = QWidget()
        greeks_layout = QVBoxLayout(greeks_widget)
        self.greeks_table = QTableWidget(4, 2)
        self.greeks_table.setHorizontalHeaderLabels(["Greek", "Value"])
        self.greeks_table.verticalHeader().setVisible(False)
        self.greeks_table.horizontalHeader().setStretchLastSection(True)
        self.greeks_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.greeks_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.greeks_table.setStyleSheet(self._get_table_style())
        greeks_layout.addWidget(self.greeks_table)
        self.results_tabs.addTab(greeks_widget, "Greeks")
        self.results_tabs.setTabToolTip(0, "Payoff profile and risk shape preview")
        self.results_tabs.setTabToolTip(1, "Ranked candidates and edge rationale")
        self.results_tabs.setTabToolTip(2, "Position-level Greeks and exposure")

    def _apply_consistent_styling(self):
        """Apply consistent styling with single accent color"""
        self.setStyleSheet(
            app_theme_stylesheet()
            + """
            QTabWidget::pane {
                border: 1px solid #2d2d2d;
                background-color: #1a1a1a;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #94a3b8;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #2d2d2d;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #0ea5e9;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #374151;
            }
            """
        )

    def _get_primary_button_style(self):
        """Get primary button style with accent color"""
        return ds_button_style("primary")

    def _get_secondary_button_style(self):
        """Get secondary button style"""
        return ds_button_style("secondary")

    def _get_tertiary_button_style(self):
        """Get tertiary button style for non-primary actions."""
        return ds_button_style("tertiary")

    def _get_input_style(self, error: bool = False):
        """Get consistent input field style"""
        return ds_input_style(error=error)

    def _get_table_style(self):
        """Get consistent table style"""
        return ds_table_style()

    def _calculate_spread(self):
        """Handle calculate spread button click"""
        try:
            symbol = self.symbol_input.text().strip().upper()
            if not self._is_valid_symbol(symbol):
                self._set_feedback("Input error | Enter a valid symbol before calculating.", level="error")
                return
            term_valid, _, _ = self._evaluate_term_structure()
            if not term_valid:
                self._set_feedback("Setup error | Far expiration must be later with enough spacing.", level="error")
                return

            # Gather parameters
            params = {
                'symbol': symbol,
                'near_exp': self.near_exp_combo.currentText(),
                'far_exp': self.far_exp_combo.currentText(),
                'strike': self.strike_combo.currentText(),
                'contracts': self.contracts_spin.value(),
                'max_risk': self.max_risk_spin.value()
            }

            # Emit signal for backend processing
            self.trade_calculation_requested.emit(params)
            self._set_feedback(
                f"Calculated setup for {symbol}. Review payoff and candidate tabs.",
                level="success",
            )
            self._set_context(
                f"Setup staged {datetime.now().strftime('%H:%M:%S')} | {symbol} {params['near_exp']} -> {params['far_exp']} | "
                f"{params['contracts']} contract(s), max risk ${params['max_risk']:.0f}",
                level="info",
            )

            # Update UI with placeholder results
            self._update_results_placeholder()

        except Exception as e:
            self.logger.error(f"Error calculating spread: {e}")
            self._set_feedback("Calculation failed. Check logs and try again.", level="error")

    def _analyze_symbol(self):
        """Handle analyze symbol button click - use live data worker"""
        try:
            symbol = self.symbol_input.text().strip().upper()
            if not self._is_valid_symbol(symbol):
                self._set_feedback("Input error | Enter a valid symbol before analysis.", level="error")
                return
            term_valid, _, _ = self._evaluate_term_structure()
            if not term_valid:
                self._set_feedback("Setup error | Fix expiration spacing before running analysis.", level="error")
                return

            # Disable analyze button during analysis
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("Analyzing...")
            self._start_analysis_feedback(symbol)
            self._set_context("Live analysis running | Pulling options chain and volatility context...", level="info")

            # Get user inputs
            contracts = self.contracts_spin.value()
            debit_override = self.debit_spin.value() if self.debit_spin.value() > 0 else None

            # Use live data worker if available
            if hasattr(self, 'data_worker') and self.data_worker:
                self.logger.info(f"Starting live analysis for {symbol}")
                self.data_worker.analyze_calendar_spread(symbol, contracts, debit_override)
            else:
                # Fallback to signal emission for backward compatibility
                self.logger.warning(f"Live data worker not available, using fallback for {symbol}")
                self.symbol_analysis_requested.emit(symbol)
                self._set_feedback(
                    "Live worker unavailable. Routed analysis request through controller.",
                    level="warn",
                )
                self._reset_analyze_button()

        except Exception as e:
            self.logger.error(f"Error analyzing symbol: {e}")
            self._set_feedback("Analysis failed unexpectedly.", level="error")
            self._reset_analyze_button()

    def _on_symbol_changed(self, text):
        """Handle symbol input changes"""
        raw_symbol = text.strip().upper()
        self._active_symbol = raw_symbol
        if not raw_symbol:
            self._stop_analysis_feedback()
            self.strategy_frame.setEnabled(False)
            self.calculate_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.left_hint_label.setText("Step 1: Enter a symbol to unlock setup controls.")
            self.symbol_validation_label.setText("Format: 1-8 chars, letters/numbers (. and - allowed)")
            self.symbol_validation_label.setStyleSheet("color: #64748b; font-size: 10px;")
            self._set_feedback("Ready | Enter a symbol to start calendar spread analysis.", level="info")
            self._set_context("No analysis yet | Configure setup and run analysis to populate live metrics.", level="neutral")
            self.symbol_input.setStyleSheet(self._get_input_style())
            return

        is_valid = self._is_valid_symbol(raw_symbol)
        self.strategy_frame.setEnabled(is_valid)
        self._on_strategy_input_changed(announce_feedback=False)

        if is_valid:
            self.symbol_validation_label.setText(f"Symbol looks valid: {raw_symbol}")
            self.symbol_validation_label.setStyleSheet("color: #10b981; font-size: 10px;")
            if self.calculate_btn.isEnabled():
                self.left_hint_label.setText("Setup ready. Calculate first, then run live analysis if needed.")
                self._set_feedback(f"Ready | {raw_symbol} loaded. Choose an action.", level="success")
            else:
                self.left_hint_label.setText("Step 2: Adjust expirations until term spacing is valid.")
                self._set_feedback("Setup warning | Expiration spacing needs adjustment.", level="warn")
            self.symbol_input.setStyleSheet(self._get_input_style())
        else:
            self._stop_analysis_feedback()
            self.left_hint_label.setText("Symbol format is invalid. Example: AAPL")
            self.symbol_validation_label.setText("Invalid symbol format")
            self.symbol_validation_label.setStyleSheet("color: #ef4444; font-size: 10px;")
            self._set_feedback("Input error | Invalid symbol format.", level="error")
            self.symbol_input.setStyleSheet(self._get_input_style(error=True))

    def _on_strategy_input_changed(self, _text: str = "", announce_feedback: bool = True):
        """Recompute setup readiness when expirations/structure fields change."""
        term_valid, term_message, level = self._evaluate_term_structure()
        level_color = {
            "success": "#10b981",
            "warn": "#f59e0b",
            "error": "#ef4444",
        }.get(level, "#64748b")
        self.term_structure_label.setText(term_message)
        self.term_structure_label.setStyleSheet(f"color: {level_color}; font-size: 10px;")

        symbol = self.symbol_input.text().strip().upper()
        symbol_valid = self._is_valid_symbol(symbol)
        ready = symbol_valid and term_valid
        self.calculate_btn.setEnabled(ready)
        self.analyze_btn.setEnabled(ready)

        if symbol_valid and not term_valid:
            self.left_hint_label.setText("Step 2: Far expiration must be at least 7 days after near expiration.")
            if announce_feedback:
                self._set_feedback("Setup warning | Fix expiration spacing before running.", level="warn")
        elif symbol_valid and term_valid:
            self.left_hint_label.setText("Step 3: Run Calculate Spread to preview risk/reward.")
            if announce_feedback:
                self._set_feedback(f"Ready | {symbol} setup validated.", level="success")

    def _evaluate_term_structure(self) -> tuple[bool, str, str]:
        """Validate near/far expiration spacing for calendar spread structure."""
        near_label = self.near_exp_combo.currentText().strip()
        far_label = self.far_exp_combo.currentText().strip()
        near_dt = self._parse_exp_label(near_label)
        far_dt = self._parse_exp_label(far_label)

        if near_label == far_label:
            return False, "Term spacing: invalid (near and far expirations are identical)", "error"
        if not near_dt or not far_dt:
            return False, "Term spacing: invalid expiration format", "error"

        if far_dt <= near_dt:
            far_dt = far_dt.replace(year=far_dt.year + 1)
        gap_days = (far_dt - near_dt).days

        if gap_days < 7:
            return False, f"Term spacing: {gap_days} day(s) (minimum 7 required)", "error"
        if gap_days < 14:
            return True, f"Term spacing: {gap_days} day(s) (tight but usable)", "warn"
        if gap_days > 120:
            return True, f"Term spacing: {gap_days} day(s) (wide structure)", "warn"
        return True, f"Term spacing: {gap_days} day(s) (healthy)", "success"

    def _parse_exp_label(self, exp_label: str) -> Optional[datetime]:
        """Parse expiration label values like 'Feb 28' into comparable datetime values."""
        try:
            parsed = datetime.strptime(exp_label, "%b %d")
            return parsed.replace(year=2000)
        except ValueError:
            return None

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Validate symbol format before enabling workflow actions."""
        return bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", symbol))

    def _set_feedback(self, message: str, level: str = "info"):
        """Update right-panel workflow feedback strip."""
        self.feedback_strip.setText(message)
        style = ds_status_strip_style(level)
        self.feedback_strip.setStyleSheet(style.replace("border-radius: 6px;", "border-radius: 0px;"))

    def _set_context(self, message: str, level: str = "neutral"):
        """Update secondary context strip with compact workflow details."""
        show_context = level not in {"neutral"} and bool(message.strip())
        self.context_strip.setVisible(show_context)
        if not show_context:
            return
        self.context_strip.setText(message)
        style_level = "neutral" if level == "neutral" else level
        style = ds_status_strip_style(style_level)
        self.context_strip.setStyleSheet(style.replace("border-radius: 6px;", "border-radius: 0px;"))

    def _start_analysis_feedback(self, symbol: str):
        """Start animated feedback while analysis is running."""
        self._analysis_feedback_symbol = symbol
        self._analysis_feedback_dots = 0
        if not self._analysis_feedback_timer.isActive():
            self._analysis_feedback_timer.start()
        self._update_analysis_feedback()

    def _stop_analysis_feedback(self):
        """Stop animated analysis feedback."""
        if self._analysis_feedback_timer.isActive():
            self._analysis_feedback_timer.stop()

    @Slot()
    def _update_analysis_feedback(self):
        """Animate feedback strip to indicate live processing."""
        if not self._analysis_feedback_symbol:
            return
        dot_count = (self._analysis_feedback_dots % 3) + 1
        self._analysis_feedback_dots += 1
        dots = "." * dot_count
        self._set_feedback(
            f"Running live calendar analysis for {self._analysis_feedback_symbol}{dots}",
            level="info",
        )

    def _update_results_placeholder(self):
        """Update results with placeholder data"""
        # Update KPI tiles with placeholder values
        for tile, value, color in [
            (self.net_debit_tile, "$2.45", "#ffffff"),
            (self.max_profit_tile, "$1.55", "#10b981"),
            (self.max_loss_tile, "$2.45", "#ef4444"),
            (self.prob_profit_tile, "62%", "#f59e0b"),
            (self.tx_cost_tile, "$4.20", "#ffffff"),
        ]:
            self._set_kpi_tile_value(tile, value, color=color)
        self.copy_rationale_btn.setEnabled(bool(self.edge_rationale.toPlainText().strip()))
        self.export_candidates_btn.setEnabled(self.candidates_table.rowCount() > 0)

    def _export_candidates_csv(self):
        """Export candidate table contents to CSV."""
        row_count = self.candidates_table.rowCount()
        if row_count == 0:
            self._set_feedback("Export skipped | No candidate rows available.", level="warn")
            return

        default_name = f"calendar_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Calendar Candidates",
            default_name,
            "CSV Files (*.csv)",
        )
        if not file_path:
            self._set_feedback("Export cancelled.", level="info")
            return

        try:
            headers = [
                self.candidates_table.horizontalHeaderItem(col).text()
                if self.candidates_table.horizontalHeaderItem(col) is not None
                else f"col_{col}"
                for col in range(self.candidates_table.columnCount())
            ]
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headers)
                for row in range(row_count):
                    row_values = []
                    for col in range(self.candidates_table.columnCount()):
                        item = self.candidates_table.item(row, col)
                        row_values.append(item.text() if item is not None else "")
                    writer.writerow(row_values)
            self._set_feedback(f"Export complete | Saved {row_count} candidate row(s).", level="success")
            self._set_context(f"Candidates exported: {file_path}", level="success")
        except Exception as exc:
            self.logger.error(f"Failed to export candidates CSV: {exc}")
            self._set_feedback(f"Export failed | {exc}", level="error")

    def _copy_edge_rationale(self):
        """Copy edge rationale text to clipboard."""
        text = self.edge_rationale.toPlainText().strip()
        if not text:
            self._set_feedback("Copy skipped | No rationale text available.", level="warn")
            return
        QApplication.clipboard().setText(text)
        self._set_feedback("Copied edge rationale to clipboard.", level="success")

    def _set_kpi_tile_value(self, tile: QFrame, value: str, color: str = "#ffffff"):
        """Update KPI tile value text and color safely."""
        value_label = getattr(tile, "_value_label", None)
        if isinstance(value_label, QLabel):
            value_label.setText(value)
            value_label.setStyleSheet(f"color: {color};")

    def _reset_analyze_button(self):
        """Reset analyze button to initial state"""
        self._stop_analysis_feedback()
        self._analysis_feedback_symbol = ""
        self._on_strategy_input_changed(announce_feedback=False)
        self.analyze_btn.setText("Run Live Analysis")

    def _reset_setup_form(self):
        """Reset setup controls and clear stale output placeholders."""
        self._stop_analysis_feedback()
        self._analysis_feedback_symbol = ""
        if hasattr(self, "advanced_toggle_btn") and self.advanced_toggle_btn.isChecked():
            self.advanced_toggle_btn.setChecked(False)
        self.symbol_input.clear()
        self.near_exp_combo.setCurrentIndex(0)
        self.far_exp_combo.setCurrentIndex(0)
        self.strike_combo.setCurrentIndex(0)
        self.contracts_spin.setValue(1)
        self.max_risk_spin.setValue(500)
        self.debit_spin.setValue(0.0)
        self.edge_rationale.clear()
        self.candidates_table.setRowCount(0)
        self.copy_rationale_btn.setEnabled(False)
        self.export_candidates_btn.setEnabled(False)
        self._set_context("Setup reset | Enter a symbol and configure term structure.", level="neutral")
        self._set_feedback("Ready | Setup reset to defaults.", level="info")
        self._update_results_placeholder()

    # Live Data Worker Signal Handlers

    @Slot(str)
    def _on_analysis_started(self, symbol: str):
        """Handle analysis start notification"""
        self.logger.info(f"Analysis started for {symbol}")
        # Update UI to show analysis is running
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")
        self._start_analysis_feedback(symbol)
        self._set_context(
            f"Live analysis started {datetime.now().strftime('%H:%M:%S')} | fetching options chain and volatility metrics",
            level="info",
        )

    @Slot(str, int, str)
    def _on_analysis_progress(self, symbol: str, progress: int, status: str):
        """Handle analysis progress updates"""
        self.logger.debug(f"Analysis progress for {symbol}: {progress}% - {status}")
        if self.analyze_btn.text() != "Run Live Analysis":
            self.analyze_btn.setText(f"Analyzing {progress}%")
        self._set_context(
            f"Live analysis {progress}% | {status}",
            level="info",
        )

    @Slot(str, dict)
    def _on_analysis_completed(self, symbol: str, calendar_data: Dict[str, Any]):
        """Handle analysis completion and update UI with live data"""
        try:
            self.logger.info(f"Analysis completed for {symbol}")

            # Reset analyze button
            self._reset_analyze_button()

            # Update KPI tiles with live data
            self._update_kpi_tiles(calendar_data)

            # Update results tables with live data
            self._update_results_tables(calendar_data)
            self._set_feedback(
                f"Analysis complete for {symbol}. Review Candidates and Greeks tabs.",
                level="success",
            )
            setup_score = calendar_data.get("setup_score")
            crush_conf = calendar_data.get("crush_confidence")
            regime = calendar_data.get("regime_label", "n/a")
            setup_score_text = f"{float(setup_score):.2f}" if setup_score is not None else "n/a"
            crush_text = f"{float(crush_conf):.2f}" if crush_conf is not None else "n/a"
            self._set_context(
                f"Updated {datetime.now().strftime('%H:%M:%S')} | setup score {setup_score_text} | crush confidence {crush_text} | regime {regime}",
                level="success",
            )

            self.logger.info(f"✓ Live data displayed for {symbol}")

        except Exception as e:
            self.logger.error(f"Error updating UI with analysis results: {e}")
            self._set_feedback("Analysis completed but UI update failed.", level="error")

    @Slot(str, str)
    def _on_analysis_error(self, symbol: str, error_message: str):
        """Handle analysis errors"""
        self.logger.error(f"Analysis error for {symbol}: {error_message}")

        # Reset analyze button
        self._reset_analyze_button()

        # Show error in UI (could be improved with a status bar)
        self.logger.warning(f"Analysis failed for {symbol}: {error_message}")
        self._set_feedback(f"Analysis error for {symbol}: {error_message}", level="error")
        self._set_context("Live analysis failed | validate data connectivity and retry.", level="error")

    @Slot(bool, str)
    def _on_connection_status_changed(self, is_connected: bool, status_message: str):
        """Handle connection status changes"""
        status_emoji = "✓" if is_connected else "✗"
        self.logger.info(f"{status_emoji} Connection status: {status_message}")
        self._set_context(
            f"Connection {status_emoji} | {status_message}",
            level="success" if is_connected else "warn",
        )

    @Slot(str, dict)
    def _on_market_data_updated(self, symbol: str, market_data: Dict[str, Any]):
        """Handle market data updates"""
        self.logger.debug(f"Market data updated for {symbol}")
        if market_data:
            price = market_data.get("price")
            if isinstance(price, (int, float)):
                self._set_context(
                    f"Live quote {symbol} ${price:.2f} | updated {datetime.now().strftime('%H:%M:%S')}",
                    level="info",
                )

    def _update_kpi_tiles(self, calendar_data: Dict[str, Any]):
        """Update KPI tiles with live calendar spread data"""
        try:
            # Extract data with safe defaults
            net_debit = calendar_data.get('net_debit', 0.0)
            max_profit = calendar_data.get('max_profit', 0.0)
            max_loss = calendar_data.get('max_loss', 0.0)
            prob_profit = calendar_data.get('prob_profit', 0.0)
            tx_cost = calendar_data.get('transaction_cost', 0.0)
            tx_cost_dollars = float(tx_cost) * 100.0

            # Format values
            kpi_updates = [
                (self.net_debit_tile, f"${net_debit:.2f}", "#ffffff"),
                (self.max_profit_tile, f"${max_profit:.2f}", "#10b981"),
                (self.max_loss_tile, f"${max_loss:.2f}", "#ef4444"),
                (self.prob_profit_tile, f"{prob_profit:.1f}%", "#f59e0b"),
                (self.tx_cost_tile, f"${tx_cost_dollars:.2f}", "#ffffff"),
            ]

            # Update each KPI tile
            for tile, value, color in kpi_updates:
                try:
                    self._set_kpi_tile_value(tile, value, color=color)
                except Exception as e:
                    self.logger.warning(f"Error updating KPI tile: {e}")

        except Exception as e:
            self.logger.error(f"Error updating KPI tiles: {e}")

    def _update_results_tables(self, calendar_data: Dict[str, Any]):
        """Update results tables with live data"""
        try:
            # Update Greeks table if it exists
            if hasattr(self, 'greeks_table'):
                self._update_greeks_table(calendar_data)

            # Update candidates table if it exists
            if hasattr(self, 'candidates_table'):
                self._update_candidates_table(calendar_data)

            if hasattr(self, 'edge_rationale'):
                rationale_lines = calendar_data.get('edge_rationale', [])
                if rationale_lines:
                    self.edge_rationale.setText("\n".join(f"- {line}" for line in rationale_lines))
                else:
                    self.edge_rationale.setPlainText("No edge rationale available for this symbol.")
                self.copy_rationale_btn.setEnabled(bool(self.edge_rationale.toPlainText().strip()))

            self.logger.debug("Results tables updated with live data")

        except Exception as e:
            self.logger.error(f"Error updating results tables: {e}")

    def _update_greeks_table(self, calendar_data: Dict[str, Any]):
        """Update Greeks table with live data"""
        try:
            greeks_data = [
                ("Delta", f"{calendar_data.get('delta', 0.0):.4f}"),
                ("Gamma", f"{calendar_data.get('gamma', 0.0):.4f}"),
                ("Theta", f"{calendar_data.get('theta', 0.0):.4f}"),
                ("Vega", f"{calendar_data.get('vega', 0.0):.4f}"),
            ]

            for row, (greek, value) in enumerate(greeks_data):
                if row < self.greeks_table.rowCount():
                    self.greeks_table.setItem(row, 0, QTableWidgetItem(greek))
                    self.greeks_table.setItem(row, 1, QTableWidgetItem(value))

        except Exception as e:
            self.logger.warning(f"Error updating Greeks table: {e}")

    def _update_candidates_table(self, calendar_data: Dict[str, Any]):
        """Update candidates table with live data"""
        try:
            # Add basic spread information
            symbol = calendar_data.get('symbol', 'N/A')
            try:
                strike = float(calendar_data.get('recommended_strike', 0.0))
            except (TypeError, ValueError):
                strike = 0.0
            short_exp = calendar_data.get('short_expiration', 'N/A')
            long_exp = calendar_data.get('long_expiration', 'N/A')

            # Clear existing data
            self.candidates_table.setRowCount(1)

            # Set spread data
            self.candidates_table.setItem(0, 0, QTableWidgetItem(symbol))
            self.candidates_table.setItem(0, 1, QTableWidgetItem(f"{strike:.2f}"))
            self.candidates_table.setItem(0, 2, QTableWidgetItem(short_exp))
            self.candidates_table.setItem(0, 3, QTableWidgetItem(long_exp))
            self.candidates_table.setItem(0, 4, QTableWidgetItem(f"{calendar_data.get('net_debit', 0.0):.2f}"))
            edge_grade = str(calendar_data.get('edge_grade', 'C'))
            confidence = float(calendar_data.get('confidence_score', 50.0))
            edge_item = QTableWidgetItem(edge_grade)
            confidence_item = QTableWidgetItem(f"{confidence:.1f}%")
            if edge_grade.upper().startswith("A"):
                edge_item.setForeground(QBrush(QColor("#10b981")))
            elif edge_grade.upper().startswith("B"):
                edge_item.setForeground(QBrush(QColor("#f59e0b")))
            else:
                edge_item.setForeground(QBrush(QColor("#cbd5e1")))
            if confidence >= 70.0:
                confidence_item.setForeground(QBrush(QColor("#10b981")))
            elif confidence < 45.0:
                confidence_item.setForeground(QBrush(QColor("#ef4444")))
            self.candidates_table.setItem(0, 5, edge_item)
            self.candidates_table.setItem(0, 6, confidence_item)
            self.export_candidates_btn.setEnabled(self.candidates_table.rowCount() > 0)

        except Exception as e:
            self.logger.warning(f"Error updating candidates table: {e}")

    def set_services(self, market_data=None, options_service=None, ml_service=None,
                     volatility_service=None, greeks_calculator=None, thread_manager=None):
        """Connect external services and initialize live data worker"""
        self.logger.info("Connecting services to calendar spreads view")
        self.market_data_service = market_data
        self.options_service = options_service
        self.ml_service = ml_service
        self.volatility_service = volatility_service
        self.greeks_calculator = greeks_calculator
        self.thread_manager = thread_manager

        # Initialize live data worker if all services are available
        if all([market_data, options_service, ml_service, volatility_service, greeks_calculator, thread_manager]):
            self._init_live_data_worker()
        else:
            self.logger.warning("Some services missing - live data worker not initialized")

    def handle_analysis_result(self, symbol: str, analysis_result):
        """Public method to handle analysis results from MainWindow"""
        try:
            self.logger.info(f"Calendar spreads view handling analysis result for {symbol}")

            # Convert analysis result to calendar spread format
            if isinstance(analysis_result, dict):
                calendar_data = analysis_result
            elif hasattr(analysis_result, '__dict__'):
                calendar_data = analysis_result.__dict__
            else:
                # Try to convert using data fetch worker format
                if hasattr(self, 'data_worker') and hasattr(self.data_worker, '_convert_to_calendar_data'):
                    calendar_data = self.data_worker._convert_to_calendar_data(analysis_result)
                else:
                    self.logger.warning("Unable to convert analysis result format")
                    return

            # Update the results tables with the calendar spread data
            self._update_results_tables(calendar_data)
            self.logger.info(f"Calendar spread results updated for {symbol}")

        except Exception as e:
            self.logger.error(f"Error handling analysis result for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    def _init_live_data_worker(self):
        """Initialize the DataFetchWorker for live data integration"""
        try:
            from workers.data_fetch_worker import DataFetchWorker

            # Create live data worker
            self.data_worker = DataFetchWorker(
                market_data_service=self.market_data_service,
                options_service=self.options_service,
                volatility_service=self.volatility_service,
                ml_service=self.ml_service,
                greeks_calculator=self.greeks_calculator,
                thread_manager=self.thread_manager,
                parent=self
            )

            # Connect worker signals to UI updates
            self.data_worker.analysis_started.connect(self._on_analysis_started)
            self.data_worker.analysis_progress.connect(self._on_analysis_progress)
            self.data_worker.analysis_completed.connect(self._on_analysis_completed)
            self.data_worker.analysis_error.connect(self._on_analysis_error)
            self.data_worker.connection_status_changed.connect(self._on_connection_status_changed)
            self.data_worker.market_data_updated.connect(self._on_market_data_updated)

            self.logger.info("✓ Live data worker initialized and connected")

        except Exception as e:
            self.logger.error(f"Failed to initialize live data worker: {e}")
            self.data_worker = None
