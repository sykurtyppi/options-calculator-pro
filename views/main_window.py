"""
Options Calculator Pro - Modern Main Window Interface
====================================================

A modern, clean trading interface that fits standard screens (1366x768+) with
proper navigation, scrolling, and live data integration.

Key Features:
- Navigation tabs for different trading functions
- Full screen expansion with no maximum width restrictions
- Real-time market data integration
- Professional dark theme
- Scrollable content areas
- Responsive layout design

Author: Professional Trading Tools
Version: 12.0.0 (Modern Interface Redesign)
"""

import logging
import re
import csv
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QFrame, QCompleter, QTabWidget, QSplitter,
    QScrollArea, QSizePolicy, QComboBox,
    QListWidget, QListWidgetItem,
    QHeaderView,
    QTableWidget, QTableWidgetItem, QTextEdit, QFileDialog, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer, QStringListModel
from PySide6.QtGui import QFont, QKeySequence, QBrush, QColor, QShortcut

from utils.config_manager import ConfigManager
from views.design_system import (
    app_font,
    app_theme_stylesheet,
    button_style as ds_button_style,
    mono_text_style as ds_mono_text_style,
    panel_style as ds_panel_style,
    status_strip_style as ds_status_strip_style,
    summary_strip_style as ds_summary_strip_style,
    table_style as ds_table_style,
)


def _panel_style() -> str:
    return ds_panel_style()


def _status_strip_style(level: str = "info") -> str:
    return ds_status_strip_style(level)


def _summary_strip_style() -> str:
    return ds_summary_strip_style()


def _table_style() -> str:
    return ds_table_style()


def _mono_text_style() -> str:
    return ds_mono_text_style()


def _button_style(kind: str = "secondary") -> str:
    return ds_button_style(kind)


class MarketOverviewWidget(QFrame):
    """Compact market overview for the dashboard"""

    MARKET_TZ = ZoneInfo("America/New_York")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(84)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a1a, stop:1 #2d2d2d);
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 2px;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(16, 8, 16, 8)

        # Market indicators
        self._add_market_indicator(layout, "S&P 500", "4,850.23", "+0.45%", "#10b981")
        self._add_market_indicator(layout, "VIX", "18.42", "-2.1%", "#10b981")
        self._add_market_indicator(layout, "NASDAQ", "15,234.67", "+0.82%", "#10b981")

        layout.addStretch()

        # Market status
        status_block = QVBoxLayout()
        status_block.setSpacing(1)
        status_block.setContentsMargins(0, 0, 0, 0)

        self.status_label = QLabel("● MARKET CLOSED")
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #ef4444;")
        status_block.addWidget(self.status_label)

        self.session_label = QLabel("Weekend")
        self.session_label.setFont(QFont("Segoe UI", 8))
        self.session_label.setStyleSheet("color: #94a3b8;")
        status_block.addWidget(self.session_label)

        self.playbook_label = QLabel("Weekend mode: research and backtest workflows")
        self.playbook_label.setFont(QFont("Segoe UI", 8))
        self.playbook_label.setStyleSheet("color: #64748b;")
        status_block.addWidget(self.playbook_label)
        layout.addLayout(status_block)

        self.time_label = QLabel("--:-- ET")
        self.time_label.setFont(QFont("Consolas", 9))
        self.time_label.setStyleSheet("color: #94a3b8;")
        layout.addWidget(self.time_label)

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)
        self._update_time()

    def _add_market_indicator(self, layout, name, value, change, color):
        """Add market data indicator"""
        container = QVBoxLayout()
        container.setSpacing(0)

        name_label = QLabel(name)
        name_label.setFont(QFont("Segoe UI", 8))
        name_label.setStyleSheet("color: #94a3b8;")
        container.addWidget(name_label)

        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        value_label = QLabel(value)
        value_label.setFont(QFont("Consolas", 10, QFont.Bold))
        row_layout.addWidget(value_label)

        change_label = QLabel(change)
        change_label.setFont(QFont("Consolas", 9))
        change_label.setStyleSheet(f"color: {color};")
        row_layout.addWidget(change_label)

        container.addLayout(row_layout)
        layout.addLayout(container)

    def _update_time(self):
        """Update time display"""
        now_et = datetime.now(self.MARKET_TZ)
        self.time_label.setText(now_et.strftime("%H:%M:%S ET"))
        is_open, session_text, phase = self._resolve_market_session(now_et)
        if is_open:
            self.status_label.setText("● MARKET OPEN")
            self.status_label.setStyleSheet("color: #10b981;")
        else:
            self.status_label.setText("● MARKET CLOSED")
            self.status_label.setStyleSheet("color: #ef4444;")
        self.session_label.setText(session_text)
        self.playbook_label.setText(self._resolve_session_playbook(phase))

    def _resolve_market_session(self, now_et: datetime) -> tuple[bool, str, str]:
        """Resolve US regular session state and next open/close timing."""
        open_dt = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        def _next_business_open(start_dt: datetime) -> datetime:
            candidate = start_dt.replace(hour=9, minute=30, second=0, microsecond=0)
            while candidate.weekday() >= 5:
                candidate += timedelta(days=1)
                candidate = candidate.replace(hour=9, minute=30, second=0, microsecond=0)
            return candidate

        if now_et.weekday() >= 5:
            next_open = _next_business_open(now_et + timedelta(days=1))
            return False, f"Weekend | Next open {next_open.strftime('%a %H:%M')}", "weekend"
        if now_et < open_dt:
            return False, f"Pre-market | Opens {open_dt.strftime('%H:%M')}", "pre_market"
        if now_et >= close_dt:
            next_open = _next_business_open(now_et + timedelta(days=1))
            return False, f"After-hours | Next open {next_open.strftime('%a %H:%M')}", "after_hours"
        return True, f"Regular session | Closes {close_dt.strftime('%H:%M')}", "open"

    def _resolve_session_playbook(self, phase: str) -> str:
        """Provide concise workflow guidance by market session."""
        playbook = {
            "weekend": "Weekend mode: tune thresholds, run backtests, review edge notes",
            "pre_market": "Prep mode: queue symbols and validate term structure",
            "open": "Live mode: prioritize execution quality and risk controls",
            "after_hours": "Post-close mode: review outcomes and update assumptions",
        }
        return playbook.get(phase, "Workflow mode: monitor setup quality and risk")


class QuickAnalysisWidget(QFrame):
    """Quick analysis input widget for the dashboard"""

    symbol_entered = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(96)
        self.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
                margin: 2px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 12, 16, 12)

        # Symbol input
        symbol_label = QLabel("Symbol:")
        symbol_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        symbol_label.setStyleSheet("color: #0ea5e9;")
        layout.addWidget(symbol_label)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("e.g., AAPL, MSFT")
        self.symbol_input.setFixedWidth(120)
        self.symbol_input.setFixedHeight(36)
        self.symbol_input.setClearButtonEnabled(True)
        self.symbol_input.setToolTip("Ticker symbol for quick analysis (AAPL, NVDA, SPY).")
        self._apply_symbol_input_style("idle")
        layout.addWidget(self.symbol_input)

        # Quick analyze button
        self.analyze_btn = QPushButton("Quick Analysis")
        self.analyze_btn.setFixedHeight(36)
        self.analyze_btn.setFixedWidth(120)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0ea5e9, stop:1 #0284c7);
                border: none;
                border-radius: 6px;
                color: white;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0284c7, stop:1 #0369a1);
            }
            QPushButton:pressed {
                background: #0369a1;
            }
            QPushButton:disabled {
                background-color: #374151;
                color: #6b7280;
            }
        """)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setToolTip("Run quick model analysis for the current symbol.")
        layout.addWidget(self.analyze_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedHeight(36)
        self.clear_btn.setFixedWidth(72)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #cbd5e1;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #374151;
            }
        """)
        self.clear_btn.setToolTip("Clear symbol input and keep focus in the field.")
        self.clear_btn.clicked.connect(self._clear_symbol)
        layout.addWidget(self.clear_btn)

        layout.addStretch()

        # Current analysis summary + input hint
        summary_block = QVBoxLayout()
        summary_block.setSpacing(2)
        summary_block.setContentsMargins(0, 0, 0, 0)

        self.analysis_summary = QLabel("Enter symbol for analysis")
        self.analysis_summary.setFont(QFont("Segoe UI", 10))
        self.analysis_summary.setStyleSheet("color: #94a3b8;")
        summary_block.addWidget(self.analysis_summary)

        self.input_hint = QLabel("Shortcut: Ctrl+L to focus, Enter to analyze")
        self.input_hint.setFont(QFont("Segoe UI", 8))
        self.input_hint.setStyleSheet("color: #64748b;")
        summary_block.addWidget(self.input_hint)

        layout.addLayout(summary_block)

        # Setup connections
        self.symbol_input.textChanged.connect(self._on_text_changed)
        self.symbol_input.returnPressed.connect(self._emit_symbol)
        self.analyze_btn.clicked.connect(self._emit_symbol)

        # Setup autocomplete
        self._setup_autocomplete()

    def _setup_autocomplete(self):
        """Setup symbol autocomplete"""
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
            "SPY", "QQQ", "IWM", "DIA", "VTI", "GLD", "TLT",
            "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA"
        ]

        model = QStringListModel(symbols)
        completer = QCompleter(model)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.symbol_input.setCompleter(completer)

    def _on_text_changed(self, text):
        """Handle text changes"""
        if text and text != text.upper():
            cursor_pos = self.symbol_input.cursorPosition()
            self.symbol_input.blockSignals(True)
            self.symbol_input.setText(text.upper())
            self.symbol_input.setCursorPosition(cursor_pos)
            self.symbol_input.blockSignals(False)
            text = self.symbol_input.text()

        symbol = text.strip().upper()
        if not symbol:
            self.analyze_btn.setEnabled(False)
            self.input_hint.setText("Shortcut: Ctrl+L to focus, Enter to analyze")
            self.input_hint.setStyleSheet("color: #64748b;")
            self._apply_symbol_input_style("idle")
            return

        is_valid = bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", symbol))
        self.analyze_btn.setEnabled(is_valid)
        if is_valid:
            self.input_hint.setText("Press Enter to run quick analysis")
            self.input_hint.setStyleSheet("color: #10b981;")
            self._apply_symbol_input_style("valid")
        else:
            self.input_hint.setText("Invalid symbol format")
            self.input_hint.setStyleSheet("color: #ef4444;")
            self._apply_symbol_input_style("invalid")

    def _emit_symbol(self):
        """Emit symbol for analysis"""
        symbol = self.symbol_input.text().strip().upper()
        if symbol and bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", symbol)):
            self.symbol_entered.emit(symbol)
            self.input_hint.setText(f"Submitted {symbol}")
            self.input_hint.setStyleSheet("color: #0ea5e9;")

    def update_analysis_summary(self, summary_text):
        """Update analysis summary"""
        self.analysis_summary.setText(summary_text)

    def focus_symbol_input(self):
        """Focus symbol input and select existing text."""
        self.symbol_input.setFocus()
        self.symbol_input.selectAll()

    def trigger_analysis(self):
        """Trigger analysis from keyboard shortcut."""
        if self.analyze_btn.isEnabled():
            self._emit_symbol()

    def _clear_symbol(self):
        """Clear symbol field and return focus to input."""
        self.symbol_input.clear()
        self.focus_symbol_input()

    def _apply_symbol_input_style(self, state: str):
        """Apply symbol input style for idle/valid/invalid state."""
        border_color = "#404040"
        if state == "valid":
            border_color = "#10b981"
        elif state == "invalid":
            border_color = "#ef4444"
        self.symbol_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #0f0f0f;
                border: 2px solid {border_color};
                border-radius: 6px;
                color: #ffffff;
                font-size: 14px;
                font-family: 'Consolas', monospace;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QLineEdit:focus {{
                border-color: #0ea5e9;
            }}
        """)


class DashboardView(QScrollArea):
    """Dashboard centered around a single daily action plan."""

    analysis_requested = Signal(str)
    ACTION_PLAN_CANDIDATES = [
        ("NVDA", "0.73", "79.5%", "Front IV rich"),
        ("AAPL", "0.67", "74.0%", "Stable liquidity"),
        ("MSFT", "0.64", "72.1%", "Cleaner spread profile"),
        ("AMD", "0.66", "71.3%", "Near-term premium"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setup_ui()

    def setup_ui(self):
        """Build a simplified dashboard around one primary action card."""
        content_widget = QWidget()
        self.setWidget(content_widget)

        main_layout = QVBoxLayout(content_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Top row: market context + quick analysis entry.
        top_section = QHBoxLayout()
        top_section.setSpacing(12)

        self.market_overview = MarketOverviewWidget()
        self.quick_analysis = QuickAnalysisWidget()
        self.quick_analysis.symbol_entered.connect(self._handle_symbol_entry)

        top_section.addWidget(self.market_overview, 1)
        top_section.addWidget(self.quick_analysis, 1)
        main_layout.addLayout(top_section)

        # Primary workspace: single action plan card.
        action_frame = QFrame()
        action_frame.setStyleSheet(_panel_style())
        action_layout = QVBoxLayout(action_frame)
        action_layout.setSpacing(10)
        action_layout.setContentsMargins(16, 14, 16, 14)

        title_row = QHBoxLayout()
        title = QLabel("Today's Action Plan")
        title.setFont(app_font(16, bold=True))
        title.setStyleSheet("color: #e0f2fe;")
        title_row.addWidget(title)
        title_row.addStretch()
        date_label = QLabel(datetime.now().strftime("%a %Y-%m-%d"))
        date_label.setFont(app_font(10))
        date_label.setStyleSheet("color: #64748b;")
        title_row.addWidget(date_label)
        action_layout.addLayout(title_row)

        self.plan_status_label = QLabel(
            "Step 1: Validate regime and event timing. Then shortlist and execute only top-confidence setups."
        )
        self.plan_status_label.setStyleSheet(_status_strip_style("info"))
        action_layout.addWidget(self.plan_status_label)

        self.plan_context_label = QLabel(
            "Workflow: 1) Regime check  2) Candidate shortlisting  3) Symbol-level execution."
        )
        self.plan_context_label.setStyleSheet(_summary_strip_style())
        action_layout.addWidget(self.plan_context_label)

        steps_grid = QGridLayout()
        steps_grid.setHorizontalSpacing(10)
        steps_grid.setVerticalSpacing(8)
        steps_grid.addWidget(self._create_step_card("1", "Regime Check", "Confirm term structure and IV/RV context."), 0, 0)
        steps_grid.addWidget(self._create_step_card("2", "Build Shortlist", "Use scanner confidence + liquidity filters."), 0, 1)
        steps_grid.addWidget(self._create_step_card("3", "Execute Setup", "Run analysis only on strongest candidates."), 0, 2)
        action_layout.addLayout(steps_grid)

        self.plan_table = QTableWidget(0, 4)
        self.plan_table.setHorizontalHeaderLabels(["Symbol", "Setup", "Confidence", "Why"])
        self.plan_table.verticalHeader().setVisible(False)
        self.plan_table.setAlternatingRowColors(True)
        self.plan_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.plan_table.setSelectionMode(QTableWidget.SingleSelection)
        self.plan_table.setStyleSheet(_table_style())
        self.plan_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.plan_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.plan_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.plan_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.plan_table.itemSelectionChanged.connect(self._on_plan_selection_changed)
        action_layout.addWidget(self.plan_table)
        self._populate_plan_candidates()

        actions_row = QHBoxLayout()
        self.analyze_selected_btn = QPushButton("Analyze Selected Candidate")
        self.analyze_selected_btn.setStyleSheet(_button_style("primary"))
        self.analyze_selected_btn.clicked.connect(self._analyze_selected_candidate)
        actions_row.addWidget(self.analyze_selected_btn)

        self.refresh_plan_btn = QPushButton("Refresh Plan Context")
        self.refresh_plan_btn.setStyleSheet(_button_style("secondary"))
        self.refresh_plan_btn.clicked.connect(self._refresh_plan_context)
        actions_row.addWidget(self.refresh_plan_btn)
        actions_row.addStretch()
        action_layout.addLayout(actions_row)

        main_layout.addWidget(action_frame)
        main_layout.addStretch(1)

    def _create_step_card(self, step_number: str, title: str, subtitle: str) -> QFrame:
        """Create compact action-step card."""
        card = QFrame()
        card.setStyleSheet(
            """
            QFrame {
                background-color: #0b1220;
                border: 1px solid #1f2937;
                border-radius: 8px;
            }
            """
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        badge = QLabel(f"Step {step_number}")
        badge.setFont(app_font(9, bold=True))
        badge.setStyleSheet("color: #0ea5e9;")
        layout.addWidget(badge)

        title_label = QLabel(title)
        title_label.setFont(app_font(11, bold=True))
        title_label.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(title_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setFont(app_font(10))
        subtitle_label.setStyleSheet("color: #94a3b8;")
        layout.addWidget(subtitle_label)

        return card

    def _populate_plan_candidates(self):
        """Load candidate rows for the daily action plan."""
        self.plan_table.setRowCount(len(self.ACTION_PLAN_CANDIDATES))
        for row_idx, row in enumerate(self.ACTION_PLAN_CANDIDATES):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(value)
                if col_idx == 0:
                    item.setFont(app_font(10, bold=True, mono=True))
                if col_idx == 2:
                    item.setForeground(QBrush(QColor("#10b981")))
                self.plan_table.setItem(row_idx, col_idx, item)
        if self.plan_table.rowCount() > 0:
            self.plan_table.selectRow(0)
            self._on_plan_selection_changed()

    def _refresh_plan_context(self):
        """Refresh summary context text."""
        now = datetime.now().strftime("%H:%M:%S")
        self.plan_status_label.setText(
            f"Plan refreshed at {now} | prioritize top confidence setups with clean execution windows."
        )

    def _on_plan_selection_changed(self):
        """Update context strip when candidate row changes."""
        row_idx = self.plan_table.currentRow()
        if row_idx < 0:
            return
        symbol_item = self.plan_table.item(row_idx, 0)
        reason_item = self.plan_table.item(row_idx, 3)
        if symbol_item is None:
            return
        symbol = symbol_item.text()
        reason = reason_item.text() if reason_item is not None else ""
        self.plan_context_label.setText(
            f"Selected {symbol} | {reason} | Click 'Analyze Selected Candidate' to continue."
        )

    def _analyze_selected_candidate(self):
        """Trigger analysis for currently selected candidate."""
        row_idx = self.plan_table.currentRow()
        if row_idx < 0:
            self.plan_status_label.setText("No candidate selected.")
            return
        symbol_item = self.plan_table.item(row_idx, 0)
        if symbol_item is None:
            return
        symbol = symbol_item.text().strip().upper()
        self.quick_analysis.symbol_input.setText(symbol)
        self._handle_symbol_entry(symbol)

    def _handle_symbol_entry(self, symbol: str):
        """Handle quick-analysis symbol submission."""
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            return
        self.plan_status_label.setText(
            f"Running action plan for {clean_symbol} | switching to execution analysis."
        )
        self.analysis_requested.emit(clean_symbol)


# Import the new professional Calendar Spreads view
# Import views with try/except to avoid circular issues
try:
    from views.calendar_spreads_view import CalendarSpreadsView
except ImportError:
    CalendarSpreadsView = None


class ScannerView(QScrollArea):
    """Options scanner/screener view"""

    SAMPLE_SCAN_RESULTS = [
        {
            "symbol": "AAPL", "iv": 31.8, "iv_rv": 1.42, "volume": 18_500_000,
            "days_to_earnings": 6, "edge": 0.67, "confidence": 74.0,
            "recommendation": "Consider", "notes": "Steep front-term IV with stable realized vol."
        },
        {
            "symbol": "NVDA", "iv": 58.4, "iv_rv": 1.76, "volume": 29_300_000,
            "days_to_earnings": 8, "edge": 0.73, "confidence": 79.5,
            "recommendation": "Consider", "notes": "Event premium elevated; term curve supports long back-month."
        },
        {
            "symbol": "TSLA", "iv": 54.1, "iv_rv": 1.55, "volume": 26_900_000,
            "days_to_earnings": 12, "edge": 0.58, "confidence": 63.2,
            "recommendation": "Watchlist", "notes": "High convexity; maintain tighter risk caps."
        },
        {
            "symbol": "MSFT", "iv": 27.6, "iv_rv": 1.29, "volume": 9_700_000,
            "days_to_earnings": 5, "edge": 0.64, "confidence": 72.1,
            "recommendation": "Consider", "notes": "Cleaner liquidity profile; lower slippage risk."
        },
        {
            "symbol": "AMZN", "iv": 41.3, "iv_rv": 1.48, "volume": 14_800_000,
            "days_to_earnings": 7, "edge": 0.61, "confidence": 68.4,
            "recommendation": "Watchlist", "notes": "Edge present but confidence below top tier."
        },
        {
            "symbol": "META", "iv": 36.2, "iv_rv": 1.37, "volume": 7_600_000,
            "days_to_earnings": 11, "edge": 0.52, "confidence": 57.8,
            "recommendation": "Pass", "notes": "Edge quality below threshold; monitor for repricing."
        },
        {
            "symbol": "NFLX", "iv": 48.7, "iv_rv": 1.62, "volume": 4_400_000,
            "days_to_earnings": 9, "edge": 0.69, "confidence": 76.2,
            "recommendation": "Consider", "notes": "Strong term dislocation with acceptable volume."
        },
        {
            "symbol": "AMD", "iv": 45.9, "iv_rv": 1.51, "volume": 11_100_000,
            "days_to_earnings": 4, "edge": 0.66, "confidence": 71.3,
            "recommendation": "Consider", "notes": "Short-dated premium rich; calendar structure attractive."
        },
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scan_results = []
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Main content widget
        content_widget = QWidget()
        self.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Title and summary
        title = QLabel("Options Scanner")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #0ea5e9; margin-bottom: 8px;")
        layout.addWidget(title)

        subtitle = QLabel("Filter earnings candidates by IV regime, liquidity, and confidence.")
        subtitle.setStyleSheet("color: #94a3b8; margin-top: -6px; margin-bottom: 6px;")
        layout.addWidget(subtitle)

        # Scanner controls
        controls_frame = QFrame()
        controls_frame.setStyleSheet(_panel_style())

        controls_layout = QGridLayout(controls_frame)
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(16, 12, 16, 12)

        # Scan criteria
        controls_layout.addWidget(QLabel("Min IV:"), 0, 0)
        self.iv_min_spin = QDoubleSpinBox()
        self.iv_min_spin.setRange(0, 200)
        self.iv_min_spin.setValue(25)
        self.iv_min_spin.setSuffix("%")
        self.iv_min_spin.setToolTip("Minimum implied volatility threshold.")
        controls_layout.addWidget(self.iv_min_spin, 0, 1)

        controls_layout.addWidget(QLabel("Max IV:"), 0, 2)
        self.iv_max_spin = QDoubleSpinBox()
        self.iv_max_spin.setRange(0, 200)
        self.iv_max_spin.setValue(100)
        self.iv_max_spin.setSuffix("%")
        self.iv_max_spin.setToolTip("Maximum implied volatility threshold.")
        controls_layout.addWidget(self.iv_max_spin, 0, 3)

        controls_layout.addWidget(QLabel("Min Volume:"), 1, 0)
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(0, 100_000_000)
        self.volume_spin.setSingleStep(500_000)
        self.volume_spin.setValue(2_000_000)
        self.volume_spin.setToolTip("Minimum average daily volume (shares).")
        controls_layout.addWidget(self.volume_spin, 1, 1)

        controls_layout.addWidget(QLabel("Max DTE:"), 1, 2)
        self.max_dte_spin = QSpinBox()
        self.max_dte_spin.setRange(1, 60)
        self.max_dte_spin.setValue(14)
        self.max_dte_spin.setToolTip("Maximum days to the next earnings event.")
        controls_layout.addWidget(self.max_dte_spin, 1, 3)

        controls_layout.addWidget(QLabel("Min Confidence:"), 2, 0)
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0, 100)
        self.min_confidence_spin.setValue(55)
        self.min_confidence_spin.setSuffix("%")
        self.min_confidence_spin.setToolTip("Minimum model confidence required.")
        controls_layout.addWidget(self.min_confidence_spin, 2, 1)

        controls_layout.addWidget(QLabel("Scan Mode:"), 2, 2)
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems(["Balanced", "Aggressive Crush", "Conservative"])
        self.scan_mode_combo.setToolTip("Preset thresholds for edge strictness.")
        controls_layout.addWidget(self.scan_mode_combo, 2, 3)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.scan_btn = QPushButton("Run Scan")
        self.scan_btn.setStyleSheet(_button_style("primary"))
        self.scan_btn.clicked.connect(self._run_scan)
        action_row.addWidget(self.scan_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet(_button_style("secondary"))
        self.export_btn.clicked.connect(self._export_scan_results)
        action_row.addWidget(self.export_btn)

        self.clear_results_btn = QPushButton("Clear")
        self.clear_results_btn.setStyleSheet(_button_style("tertiary"))
        self.clear_results_btn.clicked.connect(self._clear_scan_results)
        action_row.addWidget(self.clear_results_btn)
        action_row.addStretch()
        controls_layout.addLayout(action_row, 3, 0, 1, 4)

        layout.addWidget(controls_frame)

        self.scan_status_label = QLabel("Ready | Configure filters and run a scan.")
        self.scan_status_label.setStyleSheet(_status_strip_style("info"))
        layout.addWidget(self.scan_status_label)

        self.results_summary_label = QLabel("No candidates loaded.")
        self.results_summary_label.setStyleSheet(_summary_strip_style())
        layout.addWidget(self.results_summary_label)

        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels([
            "Symbol", "IV", "IV/RV", "Volume", "DTE", "Edge", "Confidence", "Action"
        ])
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.setSortingEnabled(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setStyleSheet(_table_style())
        self.results_table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.results_table, 1)

        self.selection_detail = QTextEdit()
        self.selection_detail.setReadOnly(True)
        self.selection_detail.setMinimumHeight(110)
        self.selection_detail.setStyleSheet(_mono_text_style())
        self.selection_detail.setPlainText(
            "Scanner detail panel.\nSelect a row to view setup rationale, constraints, and execution notes."
        )
        layout.addWidget(self.selection_detail)

        layout.addStretch()

    def _run_scan(self):
        """Filter sample scanner universe and refresh table."""
        iv_min = float(self.iv_min_spin.value())
        iv_max = float(self.iv_max_spin.value())
        if iv_min > iv_max:
            self._set_scan_status("Input error | Min IV cannot be greater than Max IV.", level="error")
            return

        min_volume = int(self.volume_spin.value())
        max_dte = int(self.max_dte_spin.value())
        min_confidence = float(self.min_confidence_spin.value())
        mode = self.scan_mode_combo.currentText()

        mode_edge_threshold = {
            "Balanced": 0.55,
            "Aggressive Crush": 0.50,
            "Conservative": 0.62,
        }.get(mode, 0.55)

        filtered = []
        for row in self.SAMPLE_SCAN_RESULTS:
            if not (iv_min <= row["iv"] <= iv_max):
                continue
            if row["volume"] < min_volume:
                continue
            if row["days_to_earnings"] > max_dte:
                continue
            if row["confidence"] < min_confidence:
                continue
            if row["edge"] < mode_edge_threshold:
                continue
            filtered.append(row)

        self._scan_results = sorted(
            filtered,
            key=lambda item: (item["edge"], item["confidence"], item["iv_rv"]),
            reverse=True,
        )
        self._populate_scan_table(self._scan_results)
        self.export_btn.setEnabled(bool(self._scan_results))

        if self._scan_results:
            top = self._scan_results[0]
            self._set_scan_status(
                f"Scan complete | {len(self._scan_results)} candidate(s) matched filters.",
                level="success",
            )
            self.results_summary_label.setText(
                f"Top candidate: {top['symbol']} | edge {top['edge']:.2f} | confidence {top['confidence']:.1f}% | "
                f"DTE {top['days_to_earnings']}"
            )
        else:
            self._set_scan_status("No candidates matched current filters. Loosen constraints and rerun.", level="warn")
            self.results_summary_label.setText("No candidates loaded.")

    def _populate_scan_table(self, rows):
        """Populate scanner table rows with conditional formatting."""
        self.results_table.setSortingEnabled(False)
        self.results_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            volume_m = row["volume"] / 1_000_000.0
            values = [
                row["symbol"],
                f"{row['iv']:.1f}%",
                f"{row['iv_rv']:.2f}",
                f"{volume_m:.1f}M",
                str(row["days_to_earnings"]),
                f"{row['edge']:.2f}",
                f"{row['confidence']:.1f}%",
                row["recommendation"],
            ]
            for col_idx, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col_idx == 0:
                    item.setFont(QFont("Consolas", 10, QFont.Bold))
                    item.setData(Qt.UserRole, row["symbol"])
                if col_idx == 5:
                    if row["edge"] >= 0.68:
                        item.setForeground(QBrush(QColor("#10b981")))
                    elif row["edge"] < 0.58:
                        item.setForeground(QBrush(QColor("#ef4444")))
                if col_idx == 6:
                    if row["confidence"] >= 72:
                        item.setForeground(QBrush(QColor("#10b981")))
                    elif row["confidence"] < 60:
                        item.setForeground(QBrush(QColor("#ef4444")))
                if col_idx == 7:
                    rec = row["recommendation"].lower()
                    if rec == "consider":
                        item.setForeground(QBrush(QColor("#10b981")))
                    elif rec == "watchlist":
                        item.setForeground(QBrush(QColor("#f59e0b")))
                    else:
                        item.setForeground(QBrush(QColor("#ef4444")))
                self.results_table.setItem(row_idx, col_idx, item)

        self.results_table.setSortingEnabled(True)
        if rows:
            self.results_table.selectRow(0)
            self._on_selection_changed()
        else:
            self.selection_detail.setPlainText(
                "Scanner detail panel.\nNo rows available. Adjust filters and run scan again."
            )

    def _on_selection_changed(self):
        """Update details panel for selected scanner row."""
        row_idx = self.results_table.currentRow()
        if row_idx < 0:
            return
        symbol_item = self.results_table.item(row_idx, 0)
        if symbol_item is None:
            return

        selected_symbol = symbol_item.data(Qt.UserRole) or symbol_item.text()
        selected = next(
            (row for row in self._scan_results if row["symbol"] == selected_symbol),
            None,
        )
        if not selected:
            return
        details = (
            f"Symbol: {selected['symbol']}\n"
            f"Recommendation: {selected['recommendation']}\n"
            f"Edge Score: {selected['edge']:.2f}\n"
            f"Crush Confidence: {selected['confidence']:.1f}%\n"
            f"IV / RV: {selected['iv_rv']:.2f}\n"
            f"Implied Volatility: {selected['iv']:.1f}%\n"
            f"Days to Earnings: {selected['days_to_earnings']}\n"
            f"Liquidity: {selected['volume'] / 1_000_000.0:.1f}M shares\n\n"
            f"Notes: {selected['notes']}"
        )
        self.selection_detail.setPlainText(details)

    def _export_scan_results(self):
        """Export current scanner results to CSV."""
        if not self._scan_results:
            self._set_scan_status("Export skipped | No scanner results available.", level="warn")
            return

        default_name = f"scanner_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Scanner Results",
            default_name,
            "CSV Files (*.csv)",
        )
        if not file_path:
            self._set_scan_status("Export cancelled.", level="info")
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    ["symbol", "iv", "iv_rv", "volume", "days_to_earnings", "edge", "confidence", "recommendation", "notes"]
                )
                for row in self._scan_results:
                    writer.writerow([
                        row["symbol"],
                        row["iv"],
                        row["iv_rv"],
                        row["volume"],
                        row["days_to_earnings"],
                        row["edge"],
                        row["confidence"],
                        row["recommendation"],
                        row["notes"],
                    ])
            self._set_scan_status(f"Export complete | {len(self._scan_results)} row(s) saved to {file_path}", level="success")
        except Exception as exc:
            self._set_scan_status(f"Export failed | {exc}", level="error")

    def _clear_scan_results(self):
        """Clear scanner table and detail panels."""
        self._scan_results = []
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        self.results_summary_label.setText("No candidates loaded.")
        self.selection_detail.setPlainText(
            "Scanner detail panel.\nSelect a row to view setup rationale, constraints, and execution notes."
        )
        self.export_btn.setEnabled(False)
        self._set_scan_status("Cleared scanner results.", level="info")

    def _set_scan_status(self, message: str, level: str = "info"):
        """Update scanner workflow status line."""
        self.scan_status_label.setText(message)
        self.scan_status_label.setStyleSheet(_status_strip_style(level))


class AnalysisView(QScrollArea):
    """Detailed analysis results view"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_symbol = ""
        self.project_root = Path(__file__).resolve().parent.parent
        self.reports_dir = self.project_root / "exports" / "reports"
        self.db_path = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Main content widget
        content_widget = QWidget()
        self.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Title
        title = QLabel("Detailed Analysis")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #0ea5e9; margin-bottom: 8px;")
        layout.addWidget(title)

        subtitle = QLabel("Generate institutional analysis briefs, copy, and export reports.")
        subtitle.setStyleSheet("color: #94a3b8; margin-top: -6px; margin-bottom: 6px;")
        layout.addWidget(subtitle)

        controls_frame = QFrame()
        controls_frame.setStyleSheet(_panel_style())
        controls_layout = QGridLayout(controls_frame)
        controls_layout.setContentsMargins(14, 10, 14, 10)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(8)

        controls_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.symbol_input = QLineEdit("AAPL")
        self.symbol_input.setPlaceholderText("AAPL")
        self.symbol_input.setMaxLength(8)
        self.symbol_input.setToolTip("Ticker symbol for analysis brief generation.")
        controls_layout.addWidget(self.symbol_input, 0, 1)

        controls_layout.addWidget(QLabel("Focus:"), 0, 2)
        self.focus_combo = QComboBox()
        self.focus_combo.addItems([
            "Earnings Crush Setup",
            "Term Structure + IV/RV",
            "Risk and Position Sizing",
        ])
        controls_layout.addWidget(self.focus_combo, 0, 3)

        self.generate_btn = QPushButton("Generate Brief")
        self.generate_btn.setStyleSheet(_button_style("primary"))
        self.generate_btn.setToolTip("Generate a structured analysis brief for the symbol.")
        self.generate_btn.clicked.connect(self._generate_local_brief)
        controls_layout.addWidget(self.generate_btn, 1, 0)

        self.copy_btn = QPushButton("Copy Report")
        self.copy_btn.setStyleSheet(_button_style("secondary"))
        self.copy_btn.setToolTip("Copy current report text to clipboard.")
        self.copy_btn.clicked.connect(self._copy_report)
        controls_layout.addWidget(self.copy_btn, 1, 1)

        self.export_btn = QPushButton("Export TXT")
        self.export_btn.setStyleSheet(_button_style("secondary"))
        self.export_btn.setToolTip("Export current report as a text file.")
        self.export_btn.clicked.connect(self._export_report)
        controls_layout.addWidget(self.export_btn, 1, 2)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(_button_style("tertiary"))
        self.clear_btn.setToolTip("Clear report and reset metrics.")
        self.clear_btn.clicked.connect(self._clear_report)
        controls_layout.addWidget(self.clear_btn, 1, 3)

        self.load_latest_btn = QPushButton("Load Latest Outputs")
        self.load_latest_btn.setStyleSheet(_button_style("secondary"))
        self.load_latest_btn.setToolTip("Load latest threshold/regime outputs and backtest summary.")
        self.load_latest_btn.clicked.connect(self._load_latest_project_outputs)
        controls_layout.addWidget(self.load_latest_btn, 2, 0, 1, 2)

        layout.addWidget(controls_frame)

        self.report_status_label = QLabel("Ready | Enter symbol and generate a report.")
        self.report_status_label.setStyleSheet(_status_strip_style("info"))
        layout.addWidget(self.report_status_label)

        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
        """)
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setContentsMargins(12, 10, 12, 10)
        metrics_layout.setHorizontalSpacing(20)
        metrics_layout.setVerticalSpacing(8)
        self.metrics_labels = {}
        metric_keys = [
            ("Symbol", "symbol", 0, 0),
            ("Setup Score", "setup_score", 0, 1),
            ("Confidence", "confidence", 0, 2),
            ("IV/RV", "iv_rv", 1, 0),
            ("Regime", "regime", 1, 1),
            ("Updated", "updated", 1, 2),
        ]
        for label_text, key, row, col in metric_keys:
            block = QVBoxLayout()
            k_label = QLabel(label_text)
            k_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
            v_label = QLabel("--")
            v_label.setStyleSheet("color: #ffffff; font-weight: bold;")
            block.addWidget(k_label)
            block.addWidget(v_label)
            metrics_layout.addLayout(block, row, col)
            self.metrics_labels[key] = v_label
        layout.addWidget(metrics_frame)

        self.analysis_report = QTextEdit()
        self.analysis_report.setReadOnly(True)
        self.analysis_report.setMinimumHeight(300)
        self.analysis_report.setStyleSheet(_mono_text_style())
        layout.addWidget(self.analysis_report)

        layout.addStretch()
        if not self._load_latest_project_outputs():
            self._generate_local_brief()

    def _clean_symbol(self) -> str:
        symbol = self.symbol_input.text().strip().upper()
        if symbol:
            self.symbol_input.setText(symbol)
        return symbol

    def _set_report_status(self, message: str, level: str = "info"):
        self.report_status_label.setText(message)
        self.report_status_label.setStyleSheet(_status_strip_style(level))

    def _set_metric(self, key: str, value: str):
        label = self.metrics_labels.get(key)
        if label:
            label.setText(value)

    def _generate_local_brief(self):
        symbol = self._clean_symbol()
        if not symbol or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", symbol):
            self._set_report_status("Input error | Enter a valid symbol (e.g., AAPL).", level="error")
            return

        focus = self.focus_combo.currentText()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines = [
            f"INSTITUTIONAL ANALYSIS BRIEF | {symbol}",
            f"Generated: {timestamp}",
            f"Focus: {focus}",
            "",
            "1. Setup Context",
            "- Confirm earnings date and event window alignment.",
            "- Validate near/far term structure supports positive crush edge.",
            "",
            "2. Volatility Structure",
            "- Compare ATM IV to 20-30d realized vol (IV/RV).",
            "- Review term slope around earnings cycle for front-month richness.",
            "",
            "3. Risk and Execution",
            "- Verify max risk and expected drawdown constraints.",
            "- Prefer liquid strikes with stable bid/ask spreads.",
            "",
            "4. Decision Framework",
            "- Enter only if setup score and confidence pass threshold.",
            "- Log assumptions and post-event outcome for model feedback.",
        ]
        self.analysis_report.setPlainText("\n".join(report_lines))
        self._last_symbol = symbol
        self._set_metric("symbol", symbol)
        self._set_metric("setup_score", "0.60")
        self._set_metric("confidence", "65.0%")
        self._set_metric("iv_rv", "1.35")
        self._set_metric("regime", "neutral")
        self._set_metric("updated", timestamp.split(" ")[1])
        self._set_report_status(f"Brief generated for {symbol}.", level="success")

    def update_with_analysis_result(self, symbol: str, analysis_result):
        """Populate report view from live analysis output."""
        data = analysis_result if isinstance(analysis_result, dict) else getattr(analysis_result, "__dict__", {})

        def _safe_float(*keys):
            for key in keys:
                value = data.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
                try:
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        setup_score = _safe_float("setup_score", "edge_score", "crush_edge_score")
        confidence = _safe_float("confidence_score", "crush_confidence", "prob_profit")
        iv_rv = _safe_float("iv_rv", "iv_rv_ratio")
        regime = data.get("regime_label", "n/a")
        rationale = data.get("edge_rationale", [])
        rationale_lines = rationale if isinstance(rationale, list) else [str(rationale)]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_lines = [
            f"LIVE ANALYSIS REPORT | {symbol}",
            f"Updated: {timestamp}",
            "",
            "Summary",
            f"- Setup score: {setup_score:.2f}" if setup_score is not None else "- Setup score: n/a",
            f"- Confidence: {confidence:.2f}" if confidence is not None else "- Confidence: n/a",
            f"- IV/RV: {iv_rv:.2f}" if iv_rv is not None else "- IV/RV: n/a",
            f"- Regime: {regime}",
            "",
            "Edge Rationale",
        ]
        if rationale_lines and any(line.strip() for line in rationale_lines):
            report_lines.extend(f"- {line}" for line in rationale_lines)
        else:
            report_lines.append("- No edge rationale provided by current model output.")

        self.analysis_report.setPlainText("\n".join(report_lines))
        self._last_symbol = symbol
        self.symbol_input.setText(symbol)
        self._set_metric("symbol", symbol)
        self._set_metric("setup_score", f"{setup_score:.2f}" if setup_score is not None else "n/a")
        self._set_metric("confidence", f"{confidence:.2f}" if confidence is not None else "n/a")
        self._set_metric("iv_rv", f"{iv_rv:.2f}" if iv_rv is not None else "n/a")
        self._set_metric("regime", str(regime))
        self._set_metric("updated", timestamp.split(" ")[1])
        self._set_report_status(f"Live analysis report updated for {symbol}.", level="success")

    def _copy_report(self):
        text = self.analysis_report.toPlainText().strip()
        if not text:
            self._set_report_status("Copy skipped | No report content.", level="warn")
            return
        QApplication.clipboard().setText(text)
        self._set_report_status("Copied report to clipboard.", level="success")

    def _export_report(self):
        text = self.analysis_report.toPlainText().strip()
        if not text:
            self._set_report_status("Export skipped | No report content.", level="warn")
            return

        symbol = self._last_symbol or self._clean_symbol() or "report"
        default_name = f"analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Report",
            default_name,
            "Text Files (*.txt)",
        )
        if not file_path:
            self._set_report_status("Export cancelled.", level="info")
            return
        try:
            with open(file_path, "w", encoding="utf-8") as report_file:
                report_file.write(text)
            self._set_report_status(f"Exported report to {file_path}", level="success")
        except Exception as exc:
            self._set_report_status(f"Export failed | {exc}", level="error")

    def _clear_report(self):
        self.analysis_report.clear()
        for key in self.metrics_labels:
            self._set_metric(key, "--")
        self._set_report_status("Report cleared.", level="info")

    def _load_latest_project_outputs(self) -> bool:
        """Load latest persisted outputs from reports directory and backtest DB."""
        summary = self._read_latest_backtest_summary()
        threshold_data = self._read_latest_json("earnings_threshold_tuning_*.json")
        regime_data = self._read_latest_json("earnings_regime_diagnostics_*.json")

        if not summary and not threshold_data and not regime_data:
            self._set_report_status("No persisted outputs found yet. Run backfill/backtest first.", level="warn")
            return False

        symbol = "MARKET"
        if summary:
            symbol = "UNIVERSE"
        elif threshold_data and threshold_data.get("session_id"):
            symbol = "SESSION"
        self._last_symbol = symbol
        self._set_metric("symbol", symbol)
        self._set_metric("updated", datetime.now().strftime("%H:%M:%S"))

        report_lines = [
            "PROJECT OUTPUT DIGEST",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if summary:
            report_lines.extend([
                "Latest Backtest Session",
                f"- Session ID: {summary['session_id']}",
                f"- Created: {summary['created_at']}",
                f"- Trades: {summary['total_trades']}",
                f"- Win Rate: {summary['win_rate']:.2%}",
                f"- Total P&L: ${summary['total_pnl']:.2f}",
                f"- Sharpe: {summary['sharpe_ratio']:.2f}",
                f"- Max Drawdown: ${summary['max_drawdown']:.2f}",
                "",
            ])
            self._set_metric("setup_score", f"{summary['sharpe_ratio']:.2f}")
            self._set_metric("confidence", f"{summary['win_rate'] * 100.0:.1f}%")
            self._set_metric("iv_rv", "n/a")
            self._set_metric("regime", "session")

        if threshold_data:
            rec = threshold_data.get("recommendation") or {}
            report_lines.extend([
                "Threshold Tuning",
                f"- Generated: {threshold_data.get('generated_at', 'n/a')}",
                f"- Session: {threshold_data.get('session_id', 'n/a')}",
                f"- Min Confidence: {threshold_data.get('min_confidence', 'n/a')}",
                f"- Recommended Threshold: {rec.get('threshold', 'n/a')}",
                f"- Alpha Score: {rec.get('alpha_score', 'n/a')}",
                f"- Candidate Trade Count: {rec.get('trade_count', 'n/a')}",
                "",
            ])

        if regime_data:
            reg_sum = regime_data.get("summary") or {}
            report_lines.extend([
                "Regime Diagnostics",
                f"- Generated: {regime_data.get('generated_at', 'n/a')}",
                f"- Session: {regime_data.get('session_id', 'n/a')}",
                f"- Mean Net Return: {reg_sum.get('overall_mean_net_return_pct', 'n/a')}",
                f"- Win Rate: {reg_sum.get('overall_win_rate', 'n/a')}",
                f"- Regime Rows: {reg_sum.get('regime_rows', 'n/a')}",
                "",
            ])
            self._set_metric("regime", "diagnostics")

        report_lines.append("Use Export TXT to save this digest, or run live analysis for symbol-level detail.")
        self.analysis_report.setPlainText("\n".join(report_lines))
        self._set_report_status("Loaded latest persisted project outputs.", level="success")
        return True

    def _read_latest_json(self, pattern: str):
        if not self.reports_dir.exists():
            return None
        matches = sorted(self.reports_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return None
        try:
            return json.loads(matches[0].read_text(encoding="utf-8"))
        except Exception:
            return None

    def _read_latest_backtest_summary(self):
        if not self.db_path.exists():
            return None
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, created_at, total_trades, win_rate, total_pnl, sharpe_ratio, max_drawdown
                    FROM backtest_sessions
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "session_id": row[0],
                    "created_at": row[1],
                    "total_trades": int(row[2] or 0),
                    "win_rate": float(row[3] or 0.0),
                    "total_pnl": float(row[4] or 0.0),
                    "sharpe_ratio": float(row[5] or 0.0),
                    "max_drawdown": float(row[6] or 0.0),
                }
        except Exception:
            return None


class HistoryView(QScrollArea):
    """Historical analysis and backtesting view"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_root = Path(__file__).resolve().parent.parent
        self.reports_dir = self.project_root / "exports" / "reports"
        self.db_path = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
        self._fallback_history_rows = [
            {
                "date": "2026-02-20",
                "symbol": "NVDA",
                "strategy": "Calendar Spread",
                "trades": 3,
                "win_rate": 66.7,
                "pnl": 148.20,
                "sharpe": 1.44,
                "max_dd": -42.10,
                "notes": "Strong front-month richness into earnings window.",
            },
            {
                "date": "2026-02-18",
                "symbol": "AAPL",
                "strategy": "Calendar Spread",
                "trades": 2,
                "win_rate": 50.0,
                "pnl": 28.50,
                "sharpe": 0.78,
                "max_dd": -33.40,
                "notes": "Edge decayed quickly; tighter exits improved result.",
            },
            {
                "date": "2026-02-14",
                "symbol": "AMD",
                "strategy": "Vol Crush",
                "trades": 4,
                "win_rate": 75.0,
                "pnl": 212.90,
                "sharpe": 1.91,
                "max_dd": -55.00,
                "notes": "Best setup score cohort; good liquidity maintained.",
            },
            {
                "date": "2026-02-10",
                "symbol": "TSLA",
                "strategy": "Diagonal",
                "trades": 2,
                "win_rate": 50.0,
                "pnl": -34.60,
                "sharpe": -0.12,
                "max_dd": -80.00,
                "notes": "Execution slippage dominated expected edge.",
            },
        ]
        self._history_rows = []
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Main content widget
        content_widget = QWidget()
        self.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Title
        title = QLabel("Historical Analysis & Backtesting")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #0ea5e9; margin-bottom: 8px;")
        layout.addWidget(title)

        subtitle = QLabel("Review historical sessions, summarize performance, and export reports.")
        subtitle.setStyleSheet("color: #94a3b8; margin-top: -6px; margin-bottom: 6px;")
        layout.addWidget(subtitle)

        controls_frame = QFrame()
        controls_frame.setStyleSheet(_panel_style())
        controls_layout = QGridLayout(controls_frame)
        controls_layout.setContentsMargins(14, 10, 14, 10)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(8)

        controls_layout.addWidget(QLabel("Window:"), 0, 0)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["30D", "90D", "1Y", "All"])
        controls_layout.addWidget(self.window_combo, 0, 1)

        controls_layout.addWidget(QLabel("Strategy:"), 0, 2)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["All", "Calendar Spread", "Vol Crush", "Diagonal"])
        controls_layout.addWidget(self.strategy_combo, 0, 3)
        self.window_combo.currentTextChanged.connect(lambda _text: self._refresh_history())
        self.strategy_combo.currentTextChanged.connect(lambda _text: self._refresh_history())

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet(_button_style("primary"))
        self.refresh_btn.setToolTip("Reload historical rows with current filters.")
        self.refresh_btn.clicked.connect(self._refresh_history)
        controls_layout.addWidget(self.refresh_btn, 1, 0)

        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.setStyleSheet(_button_style("secondary"))
        self.export_csv_btn.setToolTip("Export filtered history table to CSV.")
        self.export_csv_btn.clicked.connect(self._export_history_csv)
        controls_layout.addWidget(self.export_csv_btn, 1, 1)

        self.export_summary_btn = QPushButton("Export Summary")
        self.export_summary_btn.setStyleSheet(_button_style("secondary"))
        self.export_summary_btn.setToolTip("Export aggregate historical summary to text file.")
        self.export_summary_btn.clicked.connect(self._export_history_summary)
        controls_layout.addWidget(self.export_summary_btn, 1, 2)

        self.load_outputs_btn = QPushButton("Load Outputs")
        self.load_outputs_btn.setStyleSheet(_button_style("secondary"))
        self.load_outputs_btn.setToolTip("Refresh sessions from institutional DB and reports folder.")
        self.load_outputs_btn.clicked.connect(self._refresh_history)
        controls_layout.addWidget(self.load_outputs_btn, 1, 3)

        layout.addWidget(controls_frame)

        self.history_status_label = QLabel("Ready | Select filters and review historical sessions.")
        self.history_status_label.setStyleSheet(_status_strip_style("info"))
        layout.addWidget(self.history_status_label)

        self.history_summary_label = QLabel("No history loaded.")
        self.history_summary_label.setStyleSheet(_summary_strip_style())
        layout.addWidget(self.history_summary_label)

        self.history_table = QTableWidget(0, 9)
        self.history_table.setHorizontalHeaderLabels([
            "Date", "Symbol", "Strategy", "Trades", "Win Rate", "P&L", "Sharpe", "Max DD", "Notes"
        ])
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SingleSelection)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSortingEnabled(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setStyleSheet(_table_style())
        self.history_table.itemSelectionChanged.connect(self._on_history_selection_changed)
        layout.addWidget(self.history_table, 1)

        self.history_detail = QTextEdit()
        self.history_detail.setReadOnly(True)
        self.history_detail.setMinimumHeight(120)
        self.history_detail.setStyleSheet(_mono_text_style())
        self.history_detail.setPlainText("Select a history row to inspect execution notes and metrics.")
        layout.addWidget(self.history_detail)

        layout.addStretch()
        self._refresh_history()

    def _set_history_status(self, message: str, level: str = "info"):
        self.history_status_label.setText(message)
        self.history_status_label.setStyleSheet(_status_strip_style(level))

    def _filtered_rows(self):
        strategy = self.strategy_combo.currentText()
        window = self.window_combo.currentText()
        today = datetime.now().date()
        days_map = {"30D": 30, "90D": 90, "1Y": 365}
        days_limit = days_map.get(window)
        rows = []
        for row in self._history_rows:
            if strategy != "All" and row["strategy"] != strategy:
                continue
            if days_limit is not None:
                row_date = datetime.strptime(str(row["date"])[:10], "%Y-%m-%d").date()
                if (today - row_date).days > days_limit:
                    continue
            rows.append(row)
        rows.sort(key=lambda item: item["date"], reverse=True)
        return rows

    def _load_history_sources(self):
        """Load history rows from institutional DB, then fallback to baked sample rows."""
        rows = []
        if self.db_path.exists():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT s.session_id,
                               s.created_at,
                               s.total_trades,
                               s.win_rate,
                               s.total_pnl,
                               s.sharpe_ratio,
                               s.max_drawdown,
                               COALESCE(COUNT(t.id), 0) AS trade_rows,
                               COALESCE(GROUP_CONCAT(DISTINCT t.symbol), '') AS symbols
                        FROM backtest_sessions s
                        LEFT JOIN backtest_trades t ON t.session_id = s.session_id
                        GROUP BY s.session_id
                        ORDER BY s.created_at DESC
                        LIMIT 200
                        """
                    )
                    for rec in cursor.fetchall():
                        symbol_list = [sym for sym in (rec["symbols"] or "").split(",") if sym]
                        symbol_display = symbol_list[0] if len(symbol_list) == 1 else (f"{symbol_list[0]}+" if symbol_list else "UNIVERSE")
                        rows.append({
                            "session_id": rec["session_id"],
                            "date": str(rec["created_at"])[:10],
                            "created_at": rec["created_at"],
                            "symbol": symbol_display,
                            "strategy": "Calendar Spread",
                            "trades": int(rec["total_trades"] or rec["trade_rows"] or 0),
                            "win_rate": float(rec["win_rate"] or 0.0) * 100.0,
                            "pnl": float(rec["total_pnl"] or 0.0),
                            "sharpe": float(rec["sharpe_ratio"] or 0.0),
                            "max_dd": float(rec["max_drawdown"] or 0.0),
                            "notes": f"Session {rec['session_id']} | symbols tracked: {len(symbol_list) or 0}",
                        })
            except Exception as exc:
                self._set_history_status(f"DB read warning | {exc}", level="warn")

        if rows:
            return rows
        return list(self._fallback_history_rows)

    def _refresh_history(self):
        self._history_rows = self._load_history_sources()
        rows = self._filtered_rows()
        self.history_table.setSortingEnabled(False)
        self.history_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            cells = [
                row["date"],
                row["symbol"],
                row["strategy"],
                str(row["trades"]),
                f"{row['win_rate']:.1f}%",
                f"${row['pnl']:.2f}",
                f"{row['sharpe']:.2f}",
                f"${row['max_dd']:.2f}",
                row["notes"],
            ]
            for col_idx, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col_idx == 1:
                    item.setFont(QFont("Consolas", 10, QFont.Bold))
                    item.setData(Qt.UserRole, row.get("session_id", ""))
                if col_idx == 5:
                    pnl = row["pnl"]
                    if pnl > 0:
                        item.setForeground(QBrush(QColor("#10b981")))
                    elif pnl < 0:
                        item.setForeground(QBrush(QColor("#ef4444")))
                if col_idx == 4:
                    if row["win_rate"] >= 65:
                        item.setForeground(QBrush(QColor("#10b981")))
                    elif row["win_rate"] < 50:
                        item.setForeground(QBrush(QColor("#ef4444")))
                self.history_table.setItem(row_idx, col_idx, item)
        self.history_table.setSortingEnabled(True)

        if rows:
            total_pnl = sum(row["pnl"] for row in rows)
            total_trades = sum(row["trades"] for row in rows)
            avg_win = sum(row["win_rate"] for row in rows) / len(rows)
            source = "institutional DB" if self.db_path.exists() else "fallback sample rows"
            self.history_summary_label.setText(
                f"Source: {source} | Sessions: {len(rows)} | Trades: {total_trades} | Avg Win Rate: {avg_win:.1f}% | Total P&L: ${total_pnl:.2f}"
            )
            self._set_history_status(f"Loaded {len(rows)} historical session(s).", level="success")
            self.history_table.selectRow(0)
            self._on_history_selection_changed()
        else:
            self.history_summary_label.setText("No history rows match current filters.")
            self.history_detail.setPlainText("No rows selected.")
            self._set_history_status("No history found for selected filters.", level="warn")

    def _on_history_selection_changed(self):
        row_idx = self.history_table.currentRow()
        if row_idx < 0:
            return
        date_item = self.history_table.item(row_idx, 0)
        symbol_item = self.history_table.item(row_idx, 1)
        if date_item is None or symbol_item is None:
            return

        session_id = symbol_item.data(Qt.UserRole) if symbol_item is not None else None
        selected = None
        if session_id:
            selected = next((row for row in self._history_rows if row.get("session_id") == session_id), None)
        if selected is None:
            date_value = date_item.text()
            symbol_value = symbol_item.text()
            selected = next(
                (row for row in self._history_rows if row["date"] == date_value and row["symbol"] == symbol_value),
                None,
            )
        if not selected:
            return

        report_files = self._latest_report_files()
        self.history_detail.setPlainText(
            f"Date: {selected['date']}\n"
            f"Symbol: {selected['symbol']}\n"
            f"Strategy: {selected['strategy']}\n"
            f"Trades: {selected['trades']}\n"
            f"Win Rate: {selected['win_rate']:.1f}%\n"
            f"P&L: ${selected['pnl']:.2f}\n"
            f"Sharpe: {selected['sharpe']:.2f}\n"
            f"Max Drawdown: ${selected['max_dd']:.2f}\n"
            f"Session ID: {selected.get('session_id', 'n/a')}\n\n"
            f"Notes: {selected['notes']}\n\n"
            f"Latest report artifacts:\n"
            f"- Threshold tuning: {report_files.get('threshold', 'n/a')}\n"
            f"- Regime diagnostics: {report_files.get('regime', 'n/a')}"
        )

    def _latest_report_files(self):
        """Resolve latest report filenames for quick context display."""
        results = {"threshold": None, "regime": None}
        if not self.reports_dir.exists():
            return results
        threshold = sorted(self.reports_dir.glob("earnings_threshold_tuning_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        regime = sorted(self.reports_dir.glob("earnings_regime_diagnostics_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if threshold:
            results["threshold"] = threshold[0].name
        if regime:
            results["regime"] = regime[0].name
        return results

    def _selected_session_id(self):
        row_idx = self.history_table.currentRow()
        if row_idx < 0:
            return None
        symbol_item = self.history_table.item(row_idx, 1)
        if symbol_item is None:
            return None
        session_id = symbol_item.data(Qt.UserRole)
        return str(session_id) if session_id else None

    def _export_history_csv(self):
        rows = self._filtered_rows()
        if not rows:
            self._set_history_status("Export skipped | No filtered history rows.", level="warn")
            return
        selected_session_id = self._selected_session_id()
        if selected_session_id:
            default_name = f"backtest_trades_{selected_session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            default_name = f"history_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export History CSV",
            default_name,
            "CSV Files (*.csv)",
        )
        if not file_path:
            self._set_history_status("Export cancelled.", level="info")
            return
        try:
            exported_trade_rows = False
            if selected_session_id and self.db_path.exists():
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT *
                        FROM backtest_trades
                        WHERE session_id = ?
                        ORDER BY trade_date, symbol
                        """,
                        (selected_session_id,),
                    )
                    trade_rows = cursor.fetchall()
                    if trade_rows:
                        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                            writer = csv.writer(csv_file)
                            headers = trade_rows[0].keys()
                            writer.writerow(headers)
                            for rec in trade_rows:
                                writer.writerow([rec[h] for h in headers])
                        exported_trade_rows = True
                        self._set_history_status(
                            f"Exported {len(trade_rows)} trade row(s) for {selected_session_id}.",
                            level="success",
                        )

            if not exported_trade_rows:
                with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["date", "symbol", "strategy", "trades", "win_rate", "pnl", "sharpe", "max_dd", "notes", "session_id"])
                    for row in rows:
                        writer.writerow([
                            row["date"], row["symbol"], row["strategy"], row["trades"], row["win_rate"],
                            row["pnl"], row["sharpe"], row["max_dd"], row["notes"], row.get("session_id", "")
                        ])
                self._set_history_status(f"Exported history session CSV to {file_path}", level="success")
        except Exception as exc:
            self._set_history_status(f"Export failed | {exc}", level="error")

    def _export_history_summary(self):
        rows = self._filtered_rows()
        if not rows:
            self._set_history_status("Summary export skipped | No filtered rows.", level="warn")
            return

        total_pnl = sum(row["pnl"] for row in rows)
        total_trades = sum(row["trades"] for row in rows)
        avg_win = sum(row["win_rate"] for row in rows) / len(rows)
        best = max(rows, key=lambda row: row["pnl"])
        worst = min(rows, key=lambda row: row["pnl"])
        report_files = self._latest_report_files()

        summary_lines = [
            "HISTORICAL PERFORMANCE SUMMARY",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Data source: {'institutional DB' if self.db_path.exists() else 'fallback sample'}",
            f"Rows: {len(rows)}",
            f"Total trades: {total_trades}",
            f"Average win rate: {avg_win:.1f}%",
            f"Total P&L: ${total_pnl:.2f}",
            f"Best session: {best['date']} {best['symbol']} (${best['pnl']:.2f})",
            f"Worst session: {worst['date']} {worst['symbol']} (${worst['pnl']:.2f})",
            f"Latest threshold report: {report_files.get('threshold', 'n/a')}",
            f"Latest regime report: {report_files.get('regime', 'n/a')}",
            "",
            "Top Notes",
        ]
        for row in rows[:5]:
            summary_lines.append(f"- {row['date']} {row['symbol']}: {row['notes']}")

        default_name = f"history_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export History Summary",
            default_name,
            "Text Files (*.txt)",
        )
        if not file_path:
            self._set_history_status("Summary export cancelled.", level="info")
            return
        try:
            with open(file_path, "w", encoding="utf-8") as summary_file:
                summary_file.write("\n".join(summary_lines))
            self._set_history_status(f"Exported summary to {file_path}", level="success")
        except Exception as exc:
            self._set_history_status(f"Summary export failed | {exc}", level="error")


class MainWindow(QWidget):
    """
    Modern Options Calculator Pro Main Window

    Features:
    - Navigation tabs for different sections
    - Maximum width: 1200px for standard screens
    - Scrollable content areas
    - Real-time market data
    - Professional dark theme
    - Clean, modern design
    """

    # Signals for external integration
    analysis_requested = Signal(str, dict)
    batch_analysis_requested = Signal(list, dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()

        # Service references (will be set via set_services)
        self.market_data_service = None
        self.options_service = None
        self.ml_service = None
        self.volatility_service = None
        self.greeks_calculator = None

        self._setup_ui()
        self._apply_theme()

        self.logger.info("Modern Options Calculator Main Window initialized")

    def _setup_ui(self):
        """Setup a cleaner workspace-oriented interface."""
        self.setMinimumWidth(1100)
        self.setMinimumHeight(700)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._workspace_descriptions = {
            "Dashboard": "Session overview, market posture, and quick-entry actions.",
            "Calendar Lab": "Build and validate calendar spreads with live setup feedback.",
            "Scanner": "Filter earnings candidates by edge, confidence, and liquidity.",
            "Analysis Brief": "Generate and export structured institutional research notes.",
            "History": "Review backtests, sessions, and execution outcomes.",
        }

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # Header: concise branding + global quick action.
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 10px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(12)

        brand_block = QVBoxLayout()
        brand_block.setSpacing(2)
        app_title = QLabel("Options Calculator Pro")
        app_title.setFont(app_font(16, bold=True))
        app_title.setStyleSheet("color: #f8fafc;")
        brand_block.addWidget(app_title)

        app_subtitle = QLabel("Earnings Volatility Workspace")
        app_subtitle.setFont(app_font(10))
        app_subtitle.setStyleSheet("color: #94a3b8;")
        brand_block.addWidget(app_subtitle)
        header_layout.addLayout(brand_block)

        header_layout.addStretch()

        quick_label = QLabel("Quick Analyze")
        quick_label.setFont(app_font(10, bold=True))
        quick_label.setStyleSheet("color: #94a3b8; font-weight: bold;")
        header_layout.addWidget(quick_label)

        self.global_symbol_input = QLineEdit()
        self.global_symbol_input.setPlaceholderText("Symbol")
        self.global_symbol_input.setMaxLength(8)
        self.global_symbol_input.setFixedWidth(120)
        self.global_symbol_input.returnPressed.connect(self._run_global_analysis)
        header_layout.addWidget(self.global_symbol_input)

        self.global_analyze_btn = QPushButton("Run")
        self.global_analyze_btn.setFixedWidth(76)
        self.global_analyze_btn.setStyleSheet(_button_style("primary"))
        self.global_analyze_btn.clicked.connect(self._run_global_analysis)
        header_layout.addWidget(self.global_analyze_btn)

        version_label = QLabel("v13.0")
        version_label.setStyleSheet("color: #64748b;")
        header_layout.addWidget(version_label)

        main_layout.addWidget(header_frame)

        # Main body: left navigation + right workspace.
        body_layout = QHBoxLayout()
        body_layout.setSpacing(10)

        nav_frame = QFrame()
        nav_frame.setFixedWidth(250)
        nav_frame.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 10px;
            }
        """)
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setContentsMargins(12, 12, 12, 12)
        nav_layout.setSpacing(10)

        nav_title = QLabel("Workspace")
        nav_title.setFont(app_font(11, bold=True))
        nav_title.setStyleSheet("color: #cbd5e1;")
        nav_layout.addWidget(nav_title)

        self.nav_list = QListWidget()
        self.nav_list.setSpacing(6)
        self.nav_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for label in ["Dashboard", "Calendar Lab", "Scanner", "Analysis Brief", "History"]:
            item = QListWidgetItem(label)
            item.setToolTip(self._workspace_descriptions.get(label, ""))
            self.nav_list.addItem(item)
        self.nav_list.currentRowChanged.connect(self._switch_to_tab)
        nav_layout.addWidget(self.nav_list, 1)

        nav_hint = QLabel("Use Ctrl+1..5 to switch sections.\nCtrl+L focuses quick symbol.")
        nav_hint.setWordWrap(True)
        nav_hint.setStyleSheet("color: #64748b; font-size: 10px;")
        nav_layout.addWidget(nav_hint)

        body_layout.addWidget(nav_frame)

        content_frame = QFrame()
        content_frame.setStyleSheet("""
            QFrame {
                background-color: #020617;
                border: 1px solid #1e293b;
                border-radius: 10px;
            }
        """)
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(12, 10, 12, 12)
        content_layout.setSpacing(8)

        self.workspace_title_label = QLabel("Dashboard")
        self.workspace_title_label.setFont(app_font(14, bold=True))
        self.workspace_title_label.setStyleSheet("color: #f8fafc;")
        content_layout.addWidget(self.workspace_title_label)

        self.workspace_subtitle_label = QLabel(self._workspace_descriptions["Dashboard"])
        self.workspace_subtitle_label.setStyleSheet("color: #94a3b8; margin-top: -4px;")
        content_layout.addWidget(self.workspace_subtitle_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.tabBar().hide()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #020617;
            }
        """)

        self.dashboard_view = DashboardView()
        self.dashboard_view.analysis_requested.connect(self._handle_analysis_request)
        if CalendarSpreadsView:
            self.calendar_view = CalendarSpreadsView()
        else:
            self.calendar_view = self._create_fallback_calendar_view()
        self.scanner_view = ScannerView()
        self.analysis_view = AnalysisView()
        self.history_view = HistoryView()

        self.tab_widget.addTab(self.dashboard_view, "Dashboard")
        self.tab_widget.addTab(self.calendar_view, "Calendar Lab")
        self.tab_widget.addTab(self.scanner_view, "Scanner")
        self.tab_widget.addTab(self.analysis_view, "Analysis Brief")
        self.tab_widget.addTab(self.history_view, "History")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(self.tab_widget, 1)
        body_layout.addWidget(content_frame, 1)

        main_layout.addLayout(body_layout, 1)

        self._setup_status_bar(main_layout)
        self._setup_shortcuts()
        self.nav_list.setCurrentRow(0)
        self._on_tab_changed(0)

    def _setup_status_bar(self, parent_layout):
        """Setup status bar."""
        status_frame = QFrame()
        status_frame.setFixedHeight(34)
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 6px;
            }
        """)

        layout = QHBoxLayout(status_frame)
        layout.setContentsMargins(12, 5, 12, 5)

        self.status_label = QLabel("Ready | Select a tab to begin")
        self.status_label.setFont(app_font(9, mono=True))
        self.status_label.setStyleSheet("color: #94a3b8;")
        layout.addWidget(self.status_label)

        self.tab_context_label = QLabel("Tab: Dashboard")
        self.tab_context_label.setFont(app_font(9, mono=True))
        self.tab_context_label.setStyleSheet("color: #64748b;")
        layout.addWidget(self.tab_context_label)

        self.shortcuts_hint_label = QLabel("Shortcuts: Ctrl+1..5 switch | Ctrl+L focus | Ctrl+Enter analyze")
        self.shortcuts_hint_label.setFont(app_font(9, mono=True))
        self.shortcuts_hint_label.setStyleSheet("color: #64748b;")
        layout.addWidget(self.shortcuts_hint_label)

        layout.addStretch()

        self.last_action_label = QLabel("Last action: --")
        self.last_action_label.setFont(app_font(9, mono=True))
        self.last_action_label.setStyleSheet("color: #64748b;")
        layout.addWidget(self.last_action_label)

        # Connection indicator
        self.connection_indicator = QLabel("● Connected")
        self.connection_indicator.setFont(app_font(9, mono=True))
        self.connection_indicator.setStyleSheet("color: #10b981;")
        layout.addWidget(self.connection_indicator)

        parent_layout.addWidget(status_frame)

    def _setup_shortcuts(self):
        """Configure keyboard shortcuts for faster workflow navigation."""
        self._shortcuts = []

        self._register_shortcut("Ctrl+L", self._focus_global_symbol)
        self._register_shortcut("Ctrl+Return", self._run_global_analysis)
        self._register_shortcut("Ctrl+Enter", self._run_global_analysis)

        for idx in range(min(5, self.tab_widget.count())):
            self._register_shortcut(
                f"Ctrl+{idx + 1}",
                lambda tab_idx=idx: self._switch_to_tab(tab_idx),
            )

    def _register_shortcut(self, key_sequence: str, callback):
        """Create and retain a QShortcut to avoid garbage collection."""
        shortcut = QShortcut(QKeySequence(key_sequence), self)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    def _switch_to_tab(self, index: int):
        """Switch to tab index if valid."""
        if 0 <= index < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(index)

    def _focus_global_symbol(self):
        """Focus global symbol input and select text."""
        self.global_symbol_input.setFocus()
        self.global_symbol_input.selectAll()

    def _run_global_analysis(self):
        """Run quick analysis from top header input."""
        symbol = self.global_symbol_input.text().strip().upper()
        if not symbol or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", symbol):
            self._update_status("Input error | Enter a valid symbol (e.g., AAPL).")
            return
        self.global_symbol_input.setText(symbol)
        if hasattr(self.dashboard_view, "quick_analysis"):
            self.dashboard_view.quick_analysis.symbol_input.setText(symbol)
        if hasattr(self.analysis_view, "symbol_input"):
            self.analysis_view.symbol_input.setText(symbol)
        self._handle_analysis_request(symbol)

    def _on_tab_changed(self, index: int):
        """Update status context when navigation tabs change."""
        if index < 0 or index >= self.tab_widget.count():
            return
        tab_name = self.tab_widget.tabText(index)
        if hasattr(self, "nav_list") and self.nav_list.currentRow() != index:
            self.nav_list.blockSignals(True)
            self.nav_list.setCurrentRow(index)
            self.nav_list.blockSignals(False)
        self.workspace_title_label.setText(tab_name)
        self.workspace_subtitle_label.setText(
            self._workspace_descriptions.get(tab_name, "Workspace ready.")
        )
        self.tab_context_label.setText(f"Tab: {tab_name}")
        self._update_status(f"{tab_name} ready", track_action=False)

    def _update_status(self, message: str, track_action: bool = True):
        """Update status bar text with optional action timestamp tracking."""
        self.status_label.setText(message)
        if track_action:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.last_action_label.setText(f"Last action: {timestamp} | {message}")

    def _apply_theme(self):
        """Apply the shared theme plus navigation-specific styles."""
        self.setStyleSheet(
            app_theme_stylesheet()
            + """
            QListWidget {
                background-color: #0b1220;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 4px;
                outline: none;
            }

            QListWidget::item {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                color: #cbd5e1;
                padding: 10px 10px;
                margin: 2px 0;
                font-weight: bold;
            }

            QListWidget::item:selected {
                background-color: #0b3b5a;
                border: 1px solid #0ea5e9;
                color: #e0f2fe;
            }

            QListWidget::item:hover:!selected {
                background-color: #111827;
                border: 1px solid #334155;
            }
            """
        )

    def _create_fallback_calendar_view(self):
        """Create a fallback calendar spreads view with proper structure"""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter

        fallback_widget = QWidget()
        main_layout = QVBoxLayout(fallback_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Create QSplitter for left/right structure as required
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - inputs
        left_panel = QFrame()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
            }
        """)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)

        left_title = QLabel("Trade Setup")
        left_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        left_title.setStyleSheet("color: #0ea5e9; margin-bottom: 12px;")
        left_layout.addWidget(left_title)

        left_content = QLabel("Calendar spreads setup will appear here.\nThis is a structured left panel for inputs.")
        left_content.setStyleSheet("color: #94a3b8; padding: 20px;")
        left_content.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_content)
        left_layout.addStretch()

        # Right panel - results
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
            }
        """)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)

        right_title = QLabel("Results & Analysis")
        right_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        right_title.setStyleSheet("color: #0ea5e9; margin-bottom: 12px;")
        right_layout.addWidget(right_title)

        right_content = QLabel("Calendar spreads analysis and results will appear here.\nThis is the dominant right panel for results.")
        right_content.setStyleSheet("color: #94a3b8; padding: 40px;")
        right_content.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_content)
        right_layout.addStretch()

        # Add panels to splitter with proper proportions
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 30)  # Left 30%
        splitter.setStretchFactor(1, 70)  # Right 70% - dominant
        splitter.setSizes([350, 800])

        main_layout.addWidget(splitter)

        return fallback_widget

    def _handle_analysis_request(self, symbol):
        """Handle analysis request from dashboard"""
        try:
            self._update_status(f"Analyzing {symbol}...")

            # Switch to analysis tab
            self.tab_widget.setCurrentWidget(self.analysis_view)

            # Emit signal for external analysis
            params = {
                'symbol': symbol,
                'analysis_type': 'comprehensive',
                'source': 'dashboard'
            }

            self.analysis_requested.emit(symbol, params)

            # Update quick analysis summary
            self.dashboard_view.quick_analysis.update_analysis_summary(f"Analyzing {symbol}...")
            if hasattr(self.dashboard_view, "plan_status_label"):
                self.dashboard_view.plan_status_label.setText(
                    f"Executing plan for {symbol} | analysis requested and routing to Analysis Brief."
                )

        except Exception as e:
            self.logger.error(f"Error handling analysis request: {e}")
            self._update_status(f"Error analyzing {symbol}")

    def _handle_calendar_analysis_request(self, symbol):
        """Handle analysis request from Calendar Spreads view"""
        try:
            self._update_status(f"Analyzing calendar spreads for {symbol}...")
            self.logger.info(f"Calendar Spreads analysis requested for {symbol}")

            # Switch to calendar spreads tab
            self.tab_widget.setCurrentWidget(self.calendar_view)

            # Emit signal for external analysis with calendar spread specific params
            params = {
                'symbol': symbol,
                'analysis_type': 'calendar_spread',
                'source': 'calendar_spreads',
                'use_ml': True,
                'monte_carlo_sims': 10000
            }

            self.analysis_requested.emit(symbol, params)

        except Exception as e:
            self.logger.error(f"Error handling calendar analysis request: {e}")
            self._update_status(f"Error analyzing {symbol}")

    def set_services(self, market_data=None, options_service=None, ml_service=None,
                     volatility_service=None, greeks_calculator=None, thread_manager=None):
        """Connect external services"""
        self.logger.info("Connecting services to main window")

        self.market_data_service = market_data
        self.options_service = options_service
        self.ml_service = ml_service
        self.volatility_service = volatility_service
        self.greeks_calculator = greeks_calculator

        # Pass services to Calendar Spreads view
        if hasattr(self, 'calendar_view') and hasattr(self.calendar_view, 'set_services'):
            self.calendar_view.set_services(
                market_data=market_data,
                options_service=options_service,
                ml_service=ml_service,
                volatility_service=volatility_service,
                greeks_calculator=greeks_calculator,
                thread_manager=thread_manager
            )

            # Connect Calendar Spreads view signals to analysis system
            if hasattr(self.calendar_view, 'symbol_analysis_requested'):
                self.calendar_view.symbol_analysis_requested.connect(self._handle_calendar_analysis_request)
                self.logger.info("✓ Calendar Spreads analysis signal connected")

        if market_data:
            self.connection_indicator.setText("● Connected")
            self.connection_indicator.setStyleSheet("color: #10b981;")
            self._update_status("Ready | Market data connected")
        else:
            self.connection_indicator.setText("● Disconnected")
            self.connection_indicator.setStyleSheet("color: #ef4444;")
            self._update_status("Ready | Market data disconnected")

        self.logger.info("Services connected to main window")

    def display_analysis_result(self, symbol: str, analysis_result):
        """Display analysis results - routes to appropriate view based on analysis type"""
        try:
            self.logger.info(f"Displaying analysis for {symbol}")

            # Check if this is a calendar spread analysis
            is_calendar_analysis = False
            if hasattr(analysis_result, '__dict__') and hasattr(analysis_result, 'analysis_type'):
                is_calendar_analysis = analysis_result.analysis_type == 'calendar_spread'
            elif isinstance(analysis_result, dict):
                is_calendar_analysis = analysis_result.get('analysis_type') == 'calendar_spread' or \
                                      analysis_result.get('source') == 'calendar_spreads'

            # Route to appropriate view
            if is_calendar_analysis and hasattr(self, 'calendar_view'):
                # Route calendar spread results to calendar view using public method
                try:
                    if hasattr(self.calendar_view, 'handle_analysis_result'):
                        self.calendar_view.handle_analysis_result(symbol, analysis_result)
                        self.logger.info(f"Calendar spread results routed to calendar view for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error routing analysis result to calendar view: {e}")

            # Always update analysis report tab when available
            try:
                if hasattr(self, 'analysis_view') and hasattr(self.analysis_view, 'update_with_analysis_result'):
                    self.analysis_view.update_with_analysis_result(symbol, analysis_result)
            except Exception as e:
                self.logger.error(f"Error updating analysis report view for {symbol}: {e}")

            # Update dashboard summary
            analysis_type = "Calendar Spread" if is_calendar_analysis else "Standard"
            summary = f"{symbol}: {analysis_type} analysis complete"
            if hasattr(self, 'dashboard_view'):
                self.dashboard_view.quick_analysis.update_analysis_summary(summary)

            # Update status
            self._update_status(f"Analysis completed for {symbol}")

        except Exception as e:
            self.logger.error(f"Error displaying analysis result: {e}")
            self._update_status(f"Error displaying results for {symbol}")

    def get_current_symbol(self) -> str:
        """Get currently entered symbol"""
        return self.dashboard_view.quick_analysis.symbol_input.text().strip().upper()

    def get_analysis_parameters(self) -> dict:
        """Get current analysis parameters"""
        current_tab = self.tab_widget.currentWidget()

        return {
            'analysis_type': 'comprehensive',
            'current_view': current_tab.__class__.__name__,
            'source': 'main_window'
        }


# Backward compatibility aliases
MainWindowView = MainWindow
ProfessionalCalendarTradingDashboard = MainWindow
InstitutionalMainWindow = MainWindow
