from PySide6.QtWidgets import (
   QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
   QPushButton, QHeaderView, QMessageBox, QDialog, QFormLayout,
   QLineEdit, QTextEdit, QComboBox, QLabel, QGroupBox, QProgressBar,
   QSplitter, QFrame, QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog,
   QTabWidget, QGridLayout, QSlider, QListWidget, QListWidgetItem,
   QTreeWidget, QTreeWidgetItem, QScrollArea, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QPainter
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ScannerView(QWidget):
   """Main stock scanner view widget"""
   
   # Signals
   scan_requested = Signal(list, dict)  # tickers, criteria
   scan_stop_requested = Signal()
   export_requested = Signal(str, list)  # file_path, results
   analyze_ticker_requested = Signal(str)  # ticker
   add_to_watchlist_requested = Signal(list)  # tickers
   
   def __init__(self, scanner_controller, parent=None):
       super().__init__(parent)
       self.scanner_controller = scanner_controller
       self.scan_results = []
       self.is_scanning = False
       
       self.setup_ui()
       self.connect_signals()
       self.setup_styles()
       self.load_predefined_lists()
   
   def setup_ui(self):
       """Setup the user interface"""
       layout = QVBoxLayout(self)
       
       # Create main splitter
       main_splitter = QSplitter(Qt.Horizontal)
       layout.addWidget(main_splitter)
       
       # Left panel - Scanner configuration
       config_panel = self.create_configuration_panel()
       main_splitter.addWidget(config_panel)
       
       # Right panel - Results
       results_panel = self.create_results_panel()
       main_splitter.addWidget(results_panel)
       
       # Set splitter proportions (30% config, 70% results)
       main_splitter.setSizes([300, 700])
       
       # Status bar
       self.create_status_bar(layout)
   
   def create_configuration_panel(self) -> QWidget:
       """Create scanner configuration panel"""
       panel = QFrame()
       panel.setFrameStyle(QFrame.StyledPanel)
       panel.setMaximumWidth(350)
       layout = QVBoxLayout(panel)
       
       # Title
       title = QLabel("Scanner Configuration")
       title.setFont(QFont("Arial", 14, QFont.Bold))
       title.setAlignment(Qt.AlignCenter)
       layout.addWidget(title)
       
       # Create tabs for different sections
       config_tabs = QTabWidget()
       layout.addWidget(config_tabs)
       
       # Stock Selection tab
       stocks_tab = self.create_stocks_tab()
       config_tabs.addTab(stocks_tab, "Stocks")
       
       # Criteria tab
       criteria_tab = self.create_criteria_tab()
       config_tabs.addTab(criteria_tab, "Criteria")
       
       # Templates tab
       templates_tab = self.create_templates_tab()
       config_tabs.addTab(templates_tab, "Templates")
       
       # Control buttons
       self.create_control_buttons(layout)
       
       return panel
   
   def create_stocks_tab(self) -> QWidget:
       """Create stock selection tab"""
       widget = QWidget()
       layout = QVBoxLayout(widget)
       
       # Predefined lists
       lists_group = QGroupBox("Predefined Lists")
       lists_layout = QVBoxLayout(lists_group)
       
       self.stock_lists_combo = QComboBox()
       lists_layout.addWidget(self.stock_lists_combo)
       
       load_list_btn = QPushButton("Load Selected List")
       load_list_btn.clicked.connect(self.load_selected_list)
       lists_layout.addWidget(load_list_btn)
       
       layout.addWidget(lists_group)
       
       # Custom stock entry
       custom_group = QGroupBox("Custom Stock List")
       custom_layout = QVBoxLayout(custom_group)
       
       custom_layout.addWidget(QLabel("Enter symbols (comma-separated):"))
       self.stocks_text = QTextEdit()
       self.stocks_text.setMaximumHeight(100)
       self.stocks_text.setPlaceholderText("AAPL, MSFT, GOOGL, AMZN...")
       custom_layout.addWidget(self.stocks_text)
       
       # Stock count label
       self.stock_count_label = QLabel("0 stocks selected")
       self.stock_count_label.setAlignment(Qt.AlignCenter)
       custom_layout.addWidget(self.stock_count_label)
       
       # Validate button
       validate_btn = QPushButton("Validate Symbols")
       validate_btn.clicked.connect(self.validate_symbols)
       custom_layout.addWidget(validate_btn)
       
       layout.addWidget(custom_group)
       
       # Connect text changes to update count
       self.stocks_text.textChanged.connect(self.update_stock_count)
       
       return widget
   
   def create_criteria_tab(self) -> QWidget:
       """Create scanning criteria tab"""
       widget = QWidget()
       layout = QVBoxLayout(widget)
       
       # Scroll area for criteria
       scroll = QScrollArea()
       scroll_widget = QWidget()
       scroll_layout = QVBoxLayout(scroll_widget)
       
       # Confidence criteria
       conf_group = QGroupBox("Confidence & Probability")
       conf_layout = QFormLayout(conf_group)
       
       self.min_confidence = QDoubleSpinBox()
       self.min_confidence.setRange(0, 100)
       self.min_confidence.setValue(50)
       self.min_confidence.setSuffix("%")
       conf_layout.addRow("Minimum Confidence:", self.min_confidence)
       
       scroll_layout.addWidget(conf_group)
       
       # Volatility criteria
       vol_group = QGroupBox("Volatility Metrics")
       vol_layout = QFormLayout(vol_group)
       
       self.max_iv_rv = QDoubleSpinBox()
       self.max_iv_rv.setRange(0.1, 10.0)
       self.max_iv_rv.setValue(2.0)
       self.max_iv_rv.setDecimals(1)
       vol_layout.addRow("Max IV/RV Ratio:", self.max_iv_rv)
       
       self.max_vix = QDoubleSpinBox()
       self.max_vix.setRange(5, 100)
       self.max_vix.setValue(35)
       self.max_vix.setDecimals(1)
       vol_layout.addRow("Max VIX Level:", self.max_vix)
       
       scroll_layout.addWidget(vol_group)
       
       # Earnings criteria
       earnings_group = QGroupBox("Earnings Timing")
       earnings_layout = QFormLayout(earnings_group)
       
       self.min_days_earnings = QSpinBox()
       self.min_days_earnings.setRange(0, 365)
       self.min_days_earnings.setValue(1)
       earnings_layout.addRow("Min Days to Earnings:", self.min_days_earnings)
       
       self.max_days_earnings = QSpinBox()
       self.max_days_earnings.setRange(0, 365)
       self.max_days_earnings.setValue(14)
       earnings_layout.addRow("Max Days to Earnings:", self.max_days_earnings)
       
       scroll_layout.addWidget(earnings_group)
       
       # Market criteria
       market_group = QGroupBox("Market Metrics")
       market_layout = QFormLayout(market_group)
       
       self.min_volume = QDoubleSpinBox()
       self.min_volume.setRange(0, 1000)
       self.min_volume.setValue(50)
       self.min_volume.setSuffix("M")
       market_layout.addRow("Min Daily Volume:", self.min_volume)
       
       self.min_price = QDoubleSpinBox()
       self.min_price.setRange(1, 1000)
       self.min_price.setValue(10)
       self.min_price.setPrefix("$")
       market_layout.addRow("Min Stock Price:", self.min_price)
       
       self.max_price = QDoubleSpinBox()
       self.max_price.setRange(1, 10000)
       self.max_price.setValue(1000)
       self.max_price.setPrefix("$")
       market_layout.addRow("Max Stock Price:", self.max_price)
       
       scroll_layout.addWidget(market_group)
       
       # Sector filters
       sector_group = QGroupBox("Sector Filters")
       sector_layout = QVBoxLayout(sector_group)
       
       # Include sectors
       sector_layout.addWidget(QLabel("Include Sectors:"))
       self.include_sectors = QListWidget()
       self.include_sectors.setMaximumHeight(100)
       self.include_sectors.setSelectionMode(QListWidget.MultiSelection)
       sectors = [
           "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
           "Communication Services", "Industrials", "Consumer Defensive",
           "Energy", "Utilities", "Real Estate", "Basic Materials"
       ]
       for sector in sectors:
           item = QListWidgetItem(sector)
           self.include_sectors.addItem(item)
       sector_layout.addWidget(self.include_sectors)
       
       # Exclude sectors checkbox
       self.exclude_mode = QCheckBox("Exclude selected sectors instead")
       sector_layout.addWidget(self.exclude_mode)
       
       scroll_layout.addWidget(sector_group)
       
       # Set scroll widget
       scroll.setWidget(scroll_widget)
       scroll.setWidgetResizable(True)
       layout.addWidget(scroll)
       
       return widget
   
   def create_templates_tab(self) -> QWidget:
       """Create scan templates tab"""
       widget = QWidget()
       layout = QVBoxLayout(widget)
       
       # Predefined templates
       templates_group = QGroupBox("Predefined Templates")
       templates_layout = QVBoxLayout(templates_group)
       
       self.templates_list = QListWidget()
       templates_layout.addWidget(self.templates_list)
       
       templates_buttons = QHBoxLayout()
       load_template_btn = QPushButton("Load Template")
       load_template_btn.clicked.connect(self.load_template)
       templates_buttons.addWidget(load_template_btn)
       
       save_template_btn = QPushButton("Save Template")
       save_template_btn.clicked.connect(self.save_template)
       templates_buttons.addWidget(save_template_btn)
       
       templates_layout.addLayout(templates_buttons)
       layout.addWidget(templates_group)
       
       # Custom templates
       custom_group = QGroupBox("Custom Templates")
       custom_layout = QVBoxLayout(custom_group)
       
       self.custom_templates_list = QListWidget()
       custom_layout.addWidget(self.custom_templates_list)
       
       custom_buttons = QHBoxLayout()
       load_custom_btn = QPushButton("Load Custom")
       load_custom_btn.clicked.connect(self.load_custom_template)
       custom_buttons.addWidget(load_custom_btn)
       
       delete_custom_btn = QPushButton("Delete")
       delete_custom_btn.clicked.connect(self.delete_custom_template)
       custom_buttons.addWidget(delete_custom_btn)
       
       custom_layout.addLayout(custom_buttons)
       layout.addWidget(custom_group)
       
       # Load templates
       self.load_templates()
       
       return widget
   
   def create_control_buttons(self, layout):
       """Create scanner control buttons"""
       buttons_group = QGroupBox("Scanner Control")
       buttons_layout = QVBoxLayout(buttons_group)
       
       # Start scan button
       self.start_scan_btn = QPushButton("Start Scan")
       self.start_scan_btn.setIcon(QIcon("icons/play.png"))
       self.start_scan_btn.clicked.connect(self.start_scan)
       buttons_layout.addWidget(self.start_scan_btn)
       
       # Stop scan button
       self.stop_scan_btn = QPushButton("Stop Scan")
       self.stop_scan_btn.setIcon(QIcon("icons/stop.png"))
       self.stop_scan_btn.clicked.connect(self.stop_scan)
       self.stop_scan_btn.setEnabled(False)
       buttons_layout.addWidget(self.stop_scan_btn)
       
       # Reset button
       reset_btn = QPushButton("Reset Criteria")
       reset_btn.setIcon(QIcon("icons/reset.png"))
       reset_btn.clicked.connect(self.reset_criteria)
       buttons_layout.addWidget(reset_btn)
       
       layout.addWidget(buttons_group)
   
   def create_results_panel(self) -> QWidget:
       """Create scan results panel"""
       panel = QFrame()
       panel.setFrameStyle(QFrame.StyledPanel)
       layout = QVBoxLayout(panel)
       
       # Results header
       header_layout = QHBoxLayout()
       
       results_label = QLabel("Scan Results")
       results_label.setFont(QFont("Arial", 14, QFont.Bold))
       header_layout.addWidget(results_label)
       
       # Results count
       self.results_count_label = QLabel("0 opportunities found")
       header_layout.addWidget(self.results_count_label)
       
       header_layout.addStretch()
       
       # Results actions
       self.export_results_btn = QPushButton("Export Results")
       self.export_results_btn.setIcon(QIcon("icons/export.png"))
       self.export_results_btn.clicked.connect(self.export_results)
       self.export_results_btn.setEnabled(False)
       header_layout.addWidget(self.export_results_btn)
       
       self.add_to_watchlist_btn = QPushButton("Add to Watchlist")
       self.add_to_watchlist_btn.setIcon(QIcon("icons/star.png"))
       self.add_to_watchlist_btn.clicked.connect(self.add_selected_to_watchlist)
       self.add_to_watchlist_btn.setEnabled(False)
       header_layout.addWidget(self.add_to_watchlist_btn)
       
       layout.addLayout(header_layout)
       
       # Results table
       self.create_results_table(layout)
       
       # Results summary
       self.create_results_summary(layout)
       
       return panel
   
   def create_results_table(self, layout):
       """Create results table"""
       self.results_table = QTableWidget()
       self.setup_results_table()
       layout.addWidget(self.results_table)
   
   def setup_results_table(self):
       """Setup results table"""
       columns = [
           "Symbol", "Confidence", "Recommendation", "IV/RV", "Days to Earnings",
           "Price", "Expected Move", "Volume", "Sector", "Action"
       ]
       
       self.results_table.setColumnCount(len(columns))
       self.results_table.setHorizontalHeaderLabels(columns)
       
       # Configure table
       self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
       self.results_table.setSelectionMode(QTableWidget.MultiSelection)
       self.results_table.setAlternatingRowColors(True)
       self.results_table.setSortingEnabled(True)
       
       # Configure column widths
       header = self.results_table.horizontalHeader()
       header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Symbol
       header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Confidence
       header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Recommendation
       header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # IV/RV
       header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Days to Earnings
       header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Price
       header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Expected Move
       header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Volume
       header.setSectionResizeMode(8, QHeaderView.Stretch)          # Sector
       header.setSectionResizeMode(9, QHeaderView.ResizeToContents)  # Action
       
       # Connect signals
       self.results_table.itemSelectionChanged.connect(self.on_results_selection_changed)
       self.results_table.itemDoubleClicked.connect(self.on_result_double_clicked)
   
   def create_results_summary(self, layout):
       """Create results summary section"""
       summary_frame = QFrame()
       summary_frame.setFrameStyle(QFrame.StyledPanel)
       summary_frame.setMaximumHeight(120)
       summary_layout = QVBoxLayout(summary_frame)
       
       summary_label = QLabel("Scan Summary")
       summary_label.setFont(QFont("Arial", 12, QFont.Bold))
       summary_layout.addWidget(summary_label)
       
       # Summary stats grid
       stats_grid = QGridLayout()
       
       # Create summary labels
       self.summary_labels = {}
       summary_items = [
           ("Total Scanned:", "total_scanned", 0, 0),
           ("Opportunities:", "opportunities", 0, 1),
           ("Success Rate:", "success_rate", 0, 2),
           ("Avg Confidence:", "avg_confidence", 1, 0),
           ("High Confidence:", "high_confidence", 1, 1),
           ("Scan Time:", "scan_time", 1, 2),
       ]
       
       for label_text, key, row, col in summary_items:
           label = QLabel(label_text)
           label.setFont(QFont("Arial", 9, QFont.Bold))
           stats_grid.addWidget(label, row, col * 2)
           
           value_label = QLabel("--")
           self.summary_labels[key] = value_label
           stats_grid.addWidget(value_label, row, col * 2 + 1)
       
       summary_layout.addLayout(stats_grid)
       layout.addWidget(summary_frame)
   
   def create_status_bar(self, layout):
       """Create status bar"""
       status_frame = QFrame()
       status_frame.setFrameStyle(QFrame.StyledPanel)
       status_frame.setMaximumHeight(50)
       status_layout = QHBoxLayout(status_frame)
       
       # Status label
       self.status_label = QLabel("Ready to scan")
       status_layout.addWidget(self.status_label)
       
       # Progress bar
       self.progress_bar = QProgressBar()
       self.progress_bar.setVisible(False)
       status_layout.addWidget(self.progress_bar)
       
       # Current stock label
       self.current_stock_label = QLabel("")
       status_layout.addWidget(self.current_stock_label)
       
       layout.addWidget(status_frame)
   
   def connect_signals(self):
       """Connect signals"""
       # Scanner controller signals
       self.scanner_controller.scan_started.connect(self.on_scan_started)
       self.scanner_controller.scan_progress.connect(self.on_scan_progress)
       self.scanner_controller.scan_completed.connect(self.on_scan_completed)
       self.scanner_controller.scan_stopped.connect(self.on_scan_stopped)
       self.scanner_controller.opportunity_found.connect(self.on_opportunity_found)
       self.scanner_controller.error_occurred.connect(self.show_error)
   
   def setup_styles(self):
       """Setup widget styles"""
       self.setStyleSheet("""
           QGroupBox {
               font-weight: bold;
               border: 2px solid gray;
               border-radius: 5px;
               margin-top: 1ex;
               padding-top: 10px;
           }
           
           QGroupBox::title {
               subcontrol-origin: margin;
               left: 10px;
               padding: 0 5px 0 5px;
           }
           
           QTableWidget {
               gridline-color: #d0d0d0;
               background-color: white;
               alternate-background-color: #f0f0f0;
           }
           
           QTableWidget::item {
               padding: 5px;
           }
           
           QTableWidget::item:selected {
               background-color: #0078d4;
               color: white;
           }
           
           QPushButton {
               padding: 8px 16px;
               border: 1px solid #ccc;
               border-radius: 4px;
               background-color: #f0f0f0;
               font-weight: bold;
           }
           
           QPushButton:hover {
               background-color: #e0e0e0;
           }
           
           QPushButton:pressed {
               background-color: #d0d0d0;
           }
           
           QPushButton:disabled {
               color: #999;
               background-color: #f5f5f5;
           }
           
           QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
               padding: 5px;
               border: 1px solid #ccc;
               border-radius: 3px;
           }
       """)
   
   def load_predefined_lists(self):
       """Load predefined stock lists"""
       lists = [
           "S&P 100", "High Volume Tech", "High IV Stocks", 
           "Earnings Favorites", "Biotech", "Energy", "Custom List"
       ]
       self.stock_lists_combo.addItems(lists)
   
   def load_templates(self):
       """Load scan templates"""
       templates = [
           "Conservative", "Aggressive", "High Probability", 
           "Earnings Play", "Tech Focus"
       ]
       
       for template in templates:
           self.templates_list.addItem(template)
   
   def load_selected_list(self):
       """Load selected predefined list"""
       selected = self.stock_lists_combo.currentText()
       
       # Get predefined lists from controller
       try:
           lists = self.scanner_controller.get_predefined_ticker_lists()
           if selected in lists:
               tickers = lists[selected]
               self.stocks_text.setPlainText(", ".join(tickers))
               self.update_stock_count()
       except Exception as e:
           self.show_error("Error", f"Failed to load list: {e}")
   
   def validate_symbols(self):
       """Validate entered stock symbols"""
       symbols = self.get_stock_list()
       if not symbols:
           QMessageBox.warning(self, "Warning", "No symbols entered")
           return
       
       try:
           # Validate symbols using controller
           validation_results = self.scanner_controller.validate_tickers(symbols)
           
           valid_count = sum(1 for valid in validation_results.values() if valid)
           invalid_symbols = [symbol for symbol, valid in validation_results.items() if not valid]
           
           if invalid_symbols:
               message = f"Found {valid_count} valid symbols.\n\nInvalid symbols:\n{', '.join(invalid_symbols)}"
               QMessageBox.information(self, "Validation Results", message)
           else:
               QMessageBox.information(self, "Validation Results", f"All {valid_count} symbols are valid!")
               
       except Exception as e:
           self.show_error("Validation Error", f"Failed to validate symbols: {e}")
   
   def update_stock_count(self):
       """Update stock count label"""
       symbols = self.get_stock_list()
       count = len(symbols)
       self.stock_count_label.setText(f"{count} stocks selected")
   
   def get_stock_list(self) -> List[str]:
       """Get list of stocks from text input"""
       text = self.stocks_text.toPlainText().strip()
       if not text:
           return []
       
       symbols = [s.strip().upper() for s in text.split(",") if s.strip()]
       return symbols
   
   def get_scan_criteria(self) -> Dict[str, Any]:
       """Get current scan criteria"""
       # Get selected sectors
       selected_sectors = []
       for i in range(self.include_sectors.count()):
           item = self.include_sectors.item(i)
           if item.isSelected():
               selected_sectors.append(item.text())
       
       criteria = {
           "min_confidence": self.min_confidence.value(),
           "max_iv_rv_ratio": self.max_iv_rv.value(),
           "min_days_to_earnings": self.min_days_earnings.value(),
           "max_days_to_earnings": self.max_days_earnings.value(),
           "min_volume": self.min_volume.value() * 1000000,  # Convert to actual volume
           "max_vix": self.max_vix.value(),
           "min_price": self.min_price.value(),
           "max_price": self.max_price.value(),
       }
       
       if selected_sectors:
           if self.exclude_mode.isChecked():
               criteria["exclude_sectors"] = selected_sectors
           else:
               criteria["sectors"] = selected_sectors
       
       return criteria
   
   def start_scan(self):
       """Start scanning process"""
       symbols = self.get_stock_list()
       if not symbols:
           QMessageBox.warning(self, "Warning", "Please enter stock symbols to scan")
           return
       
       criteria = self.get_scan_criteria()
       
       # Clear previous results
       self.clear_results()
       
       # Emit scan request
       self.scan_requested.emit(symbols, criteria)
   
   def stop_scan(self):
       """Stop scanning process"""
       self.scan_stop_requested.emit()
   
   def reset_criteria(self):
       """Reset scan criteria to defaults"""
       self.min_confidence.setValue(50)
       self.max_iv_rv.setValue(2.0)
       self.min_days_earnings.setValue(1)
       self.max_days_earnings.setValue(14)
       self.min_volume.setValue(50)
       self.max_vix.setValue(35)
       self.min_price.setValue(10)
       self.max_price.setValue(1000)
       
       # Clear sector selections
       for i in range(self.include_sectors.count()):
           item = self.include_sectors.item(i)
           item.setSelected(False)
       
       self.exclude_mode.setChecked(False)
   
   def clear_results(self):
       """Clear scan results"""
       self.results_table.setRowCount(0)
       self.scan_results = []
       self.update_results_count()
       self.update_summary_stats()
   
   def load_template(self):
       """Load selected template"""
       current_item = self.templates_list.currentItem()
       if not current_item:
           return
       
       template_name = current_item.text()
       
       try:
           templates = self.scanner_controller.get_scan_templates()
           if template_name in templates:
               criteria = templates[template_name]
               self.apply_criteria(criteria)
       except Exception as e:
           self.show_error("Template Error", f"Failed to load template: {e}")
   
   def save_template(self):
       """Save current criteria as template"""
       from PySide6.QtWidgets import QInputDialog
       
       name, ok = QInputDialog.getText(self, "Save Template", "Template name:")
       if ok and name:
           criteria = self.get_scan_criteria()
           try:
               success = self.scanner_controller.save_custom_criteria(name, criteria)
               if success:
                   QMessageBox.information(self, "Success", f"Template '{name}' saved successfully")
                   self.refresh_custom_templates()
               else:
                   QMessageBox.warning(self, "Error", "Failed to save template")
           except Exception as e:
               self.show_error("Save Error", f"Failed to save template: {e}")
   
   def load_custom_template(self):
       """Load selected custom template"""
       current_item = self.custom_templates_list.currentItem()
       if not current_item:
           return
       
       template_name = current_item.text()
       
       try:
           criteria = self.scanner_controller.load_custom_criteria(template_name)
           if criteria:
               self.apply_criteria(criteria)
           else:
               QMessageBox.warning(self, "Error", "Template not found")
       except Exception as e:
           self.show_error("Template Error", f"Failed to load template: {e}")
   
   def delete_custom_template(self):
       """Delete selected custom template"""
       current_item = self.custom_templates_list.currentItem()
       if not current_item:
           return
       
       template_name = current_item.text()
       
       reply = QMessageBox.question(
           self, "Confirm Delete", 
           f"Are you sure you want to delete template '{template_name}'?",
           QMessageBox.Yes | QMessageBox.No, QMessageBox.No
       )
       
       if reply == QMessageBox.Yes:
           try:
               success = self.scanner_controller.delete_custom_criteria(template_name)
               if success:
                   self.refresh_custom_templates()
               else:
                   QMessageBox.warning(self, "Error", "Failed to delete template")
           except Exception as e:
               self.show_error("Delete Error", f"Failed to delete template: {e}")
   
   def refresh_custom_templates(self):
       """Refresh custom templates list"""
       # This would load custom templates from config
       # For now, just clear the list
       self.custom_templates_list.clear()
   
   def apply_criteria(self, criteria):
       """Apply criteria to form fields"""
       self.min_confidence.setValue(criteria.min_confidence)
       self.max_iv_rv.setValue(criteria.max_iv_rv_ratio)
       self.min_days_earnings.setValue(criteria.min_days_to_earnings)
       self.max_days_earnings.setValue(criteria.max_days_to_earnings)
       self.min_volume.setValue(criteria.min_volume / 1000000)  # Convert to millions
       self.max_vix.setValue(criteria.max_vix)
       self.min_price.setValue(criteria.min_price)
       self.max_price.setValue(criteria.max_price)
       
       # Apply sector filters
       for i in range(self.include_sectors.count()):
           item = self.include_sectors.item(i)
           sector = item.text()
           
           if criteria.sectors and sector in criteria.sectors:
               item.setSelected(True)
           elif criteria.exclude_sectors and sector in criteria.exclude_sectors:
               item.setSelected(True)
               self.exclude_mode.setChecked(True)
           else:
               item.setSelected(False)
   
   def on_scan_started(self):
       """Handle scan started"""
       self.is_scanning = True
       self.start_scan_btn.setEnabled(False)
       self.stop_scan_btn.setEnabled(True)
       self.progress_bar.setVisible(True)
       self.progress_bar.setRange(0, 100)
       self.status_label.setText("Scanning in progress...")
       
       # Clear previous results
       self.clear_results()
   
   def on_scan_progress(self, current: int, total: int, ticker: str):
       """Handle scan progress update"""
       if total > 0:
           progress = int((current / total) * 100)
           self.progress_bar.setValue(progress)
       
       self.current_stock_label.setText(f"Analyzing: {ticker}")
       self.status_label.setText(f"Scanning {current}/{total} stocks...")
   
   def on_scan_completed(self, results: List):
       """Handle scan completion"""
       self.is_scanning = False
       self.start_scan_btn.setEnabled(True)
       self.stop_scan_btn.setEnabled(False)
       self.progress_bar.setVisible(False)
       self.current_stock_label.setText("")
       
       self.scan_results = results
       self.populate_results_table(results)
       self.update_results_count()
       self.update_summary_stats()
       
       # Enable export if we have results
       self.export_results_btn.setEnabled(len(results) > 0)
       
       self.status_label.setText(f"Scan completed. Found {len(results)} opportunities.")
   
   def on_scan_stopped(self):
       """Handle scan stopped"""
       self.is_scanning = False
       self.start_scan_btn.setEnabled(True)
       self.stop_scan_btn.setEnabled(False)
       self.progress_bar.setVisible(False)
       self.current_stock_label.setText("")
       self.status_label.setText("Scan stopped by user")
   
   def on_opportunity_found(self, result):
       """Handle new opportunity found during scan"""
       # Add result to table immediately
       self.add_result_to_table(result)
       self.update_results_count()
   
   def populate_results_table(self, results: List):
       """Populate results table with scan results"""
       self.results_table.setRowCount(len(results))
       
       for row, result in enumerate(results):
           self.populate_result_row(row, result)
       
       # Sort by confidence (highest first)
       self.results_table.sortItems(1, Qt.DescendingOrder)
   
   def add_result_to_table(self, result):
       """Add single result to table"""
       row = self.results_table.rowCount()
       self.results_table.insertRow(row)
       self.populate_result_row(row, result)
   
   def populate_result_row(self, row: int, result):
       """Populate a single result row"""
       try:
           # Symbol
           symbol_item = QTableWidgetItem(result.ticker)
           symbol_item.setFont(QFont("Arial", 10, QFont.Bold))
           self.results_table.setItem(row, 0, symbol_item)
           
           # Confidence
           confidence_item = QTableWidgetItem(f"{result.confidence:.1f}%")
           
           # Color code confidence
           if result.confidence >= 70:
               confidence_item.setBackground(QColor(0, 255, 0, 100))  # Green
           elif result.confidence >= 60:
               confidence_item.setBackground(QColor(255, 255, 0, 100))  # Yellow
           else:
               confidence_item.setBackground(QColor(255, 200, 200))  # Light red
           
           self.results_table.setItem(row, 1, confidence_item)
           
           # Recommendation
           rec_item = QTableWidgetItem(result.recommendation)
           rec_item.setFont(QFont("Arial", 9, QFont.Bold))
           
           # Color code recommendation
           if result.recommendation == "STRONG BUY":
               rec_item.setForeground(QColor(0, 128, 0))
           elif result.recommendation == "BUY":
               rec_item.setForeground(QColor(0, 100, 0))
           elif result.recommendation == "CONSIDER":
               rec_item.setForeground(QColor(200, 100, 0))
           else:
               rec_item.setForeground(QColor(128, 128, 128))
           
           self.results_table.setItem(row, 2, rec_item)
           
           # IV/RV Ratio
           iv_rv_item = QTableWidgetItem(f"{result.iv_rv_ratio:.2f}")
           self.results_table.setItem(row, 3, iv_rv_item)
           
           # Days to Earnings
           days_item = QTableWidgetItem(str(result.days_to_earnings))
           self.results_table.setItem(row, 4, days_item)
           
           # Price
           price_item = QTableWidgetItem(f"${result.current_price:.2f}")
           self.results_table.setItem(row, 5, price_item)
           
           # Expected Move
           move_item = QTableWidgetItem(result.expected_move)
           self.results_table.setItem(row, 6, move_item)
           
           # Volume
           volume_item = QTableWidgetItem(f"${result.volume/1000000:.1f}M")
           self.results_table.setItem(row, 7, volume_item)
           
           # Sector
           sector_item = QTableWidgetItem(result.sector)
           self.results_table.setItem(row, 8, sector_item)
           
           # Action button
           action_btn = QPushButton("Analyze")
           action_btn.clicked.connect(lambda checked, ticker=result.ticker: self.analyze_ticker_requested.emit(ticker))
           self.results_table.setCellWidget(row, 9, action_btn)
           
       except Exception as e:
           logger.error(f"Error populating result row {row}: {e}")
   
   def update_results_count(self):
       """Update results count label"""
       count = self.results_table.rowCount()
       self.results_count_label.setText(f"{count} opportunities found")
   
   def update_summary_stats(self):
       """Update summary statistics"""
       try:
           total_scanned = len(self.get_stock_list())
           opportunities = len(self.scan_results)
           
           if total_scanned > 0:
               success_rate = (opportunities / total_scanned) * 100
           else:
               success_rate = 0
           
           if self.scan_results:
               avg_confidence = sum(r.confidence for r in self.scan_results) / len(self.scan_results)
               high_confidence = sum(1 for r in self.scan_results if r.confidence >= 70)
           else:
               avg_confidence = 0
               high_confidence = 0
           
           # Update labels
           self.summary_labels["total_scanned"].setText(str(total_scanned))
           self.summary_labels["opportunities"].setText(str(opportunities))
           self.summary_labels["success_rate"].setText(f"{success_rate:.1f}%")
           self.summary_labels["avg_confidence"].setText(f"{avg_confidence:.1f}%")
           self.summary_labels["high_confidence"].setText(str(high_confidence))
           self.summary_labels["scan_time"].setText("--")  # Would track actual scan time
           
       except Exception as e:
           logger.error(f"Error updating summary stats: {e}")
   
   def on_results_selection_changed(self):
       """Handle results selection change"""
       selected_rows = set()
       for item in self.results_table.selectedItems():
           selected_rows.add(item.row())
       
       has_selection = len(selected_rows) > 0
       self.add_to_watchlist_btn.setEnabled(has_selection)
   
   def on_result_double_clicked(self, item):
       """Handle result double-click"""
       row = item.row()
       if row < len(self.scan_results):
           result = self.scan_results[row]
           self.analyze_ticker_requested.emit(result.ticker)
   
   def export_results(self):
       """Export scan results"""
       if not self.scan_results:
           QMessageBox.information(self, "No Results", "No scan results to export")
           return
       
       # Show export dialog
       dialog = ScanResultsExportDialog(self)
       if dialog.exec() == QDialog.Accepted:
           export_options = dialog.get_export_options()
           
           file_path, _ = QFileDialog.getSaveFileName(
               self,
               "Export Scan Results",
               f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
               "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
           )
           
           if file_path:
               self.export_requested.emit(file_path, self.scan_results)
   
   def add_selected_to_watchlist(self):
       """Add selected results to watchlist"""
       selected_rows = set()
       for item in self.results_table.selectedItems():
           selected_rows.add(item.row())
       
       if not selected_rows:
           return
       
       selected_tickers = []
       for row in selected_rows:
           if row < len(self.scan_results):
               selected_tickers.append(self.scan_results[row].ticker)
       
       if selected_tickers:
           self.add_to_watchlist_requested.emit(selected_tickers)
           QMessageBox.information(
               self, "Added to Watchlist", 
               f"Added {len(selected_tickers)} stocks to watchlist"
           )
   
   def show_error(self, title: str, message: str):
       """Show error message"""
       QMessageBox.critical(self, title, message)
   
   def show_info(self, title: str, message: str):
       """Show info message"""
       QMessageBox.information(self, title, message)

class ScanResultsExportDialog(QDialog):
   """Dialog for configuring scan results export"""
   
   def __init__(self, parent=None):
       super().__init__(parent)
       self.setup_ui()
   
   def setup_ui(self):
       """Setup dialog UI"""
       self.setWindowTitle("Export Scan Results")
       self.setModal(True)
       self.resize(350, 200)
       
       layout = QVBoxLayout(self)
       
       # Export options
       options_group = QGroupBox("Export Options")
       options_layout = QFormLayout(options_group)
       
       # Include analysis data
       self.include_analysis_cb = QCheckBox()
       self.include_analysis_cb.setChecked(True)
       options_layout.addRow("Include Analysis Data:", self.include_analysis_cb)
       
       # Include timestamps
       self.include_timestamps_cb = QCheckBox()
       self.include_timestamps_cb.setChecked(True)
       options_layout.addRow("Include Timestamps:", self.include_timestamps_cb)
       
       # Export format
       self.format_combo = QComboBox()
       self.format_combo.addItems(["CSV", "Excel", "JSON"])
       options_layout.addRow("Format:", self.format_combo)
       
       # Minimum confidence filter
       self.min_confidence_export = QDoubleSpinBox()
       self.min_confidence_export.setRange(0, 100)
       self.min_confidence_export.setValue(0)
       self.min_confidence_export.setSuffix("%")
       options_layout.addRow("Min Confidence:", self.min_confidence_export)
       
       layout.addWidget(options_group)
       
       # Buttons
       button_layout = QHBoxLayout()
       
       export_btn = QPushButton("Export")
       export_btn.clicked.connect(self.accept)
       button_layout.addWidget(export_btn)
       
       cancel_btn = QPushButton("Cancel")
       cancel_btn.clicked.connect(self.reject)
       button_layout.addWidget(cancel_btn)
       
       layout.addLayout(button_layout)
   
   def get_export_options(self) -> Dict[str, Any]:
       """Get export options"""
       return {
           "include_analysis": self.include_analysis_cb.isChecked(),
           "include_timestamps": self.include_timestamps_cb.isChecked(),
           "format": self.format_combo.currentText().lower(),
           "min_confidence": self.min_confidence_export.value()
       }

class ScanProgressDialog(QDialog):
   """Dialog showing scan progress"""
   
   def __init__(self, parent=None):
       super().__init__(parent)
       self.setup_ui()
   
   def setup_ui(self):
       """Setup dialog UI"""
       self.setWindowTitle("Scanning in Progress")
       self.setModal(True)
       self.resize(400, 150)
       
       layout = QVBoxLayout(self)
       
       # Status label
       self.status_label = QLabel("Initializing scan...")
       self.status_label.setAlignment(Qt.AlignCenter)
       layout.addWidget(self.status_label)
       
       # Progress bar
       self.progress_bar = QProgressBar()
       self.progress_bar.setRange(0, 100)
       layout.addWidget(self.progress_bar)
       
       # Current stock label
       self.current_stock_label = QLabel("")
       self.current_stock_label.setAlignment(Qt.AlignCenter)
       layout.addWidget(self.current_stock_label)
       
       # Results so far
       self.results_label = QLabel("0 opportunities found")
       self.results_label.setAlignment(Qt.AlignCenter)
       layout.addWidget(self.results_label)
       
       # Cancel button
       self.cancel_btn = QPushButton("Cancel Scan")
       self.cancel_btn.clicked.connect(self.reject)
       layout.addWidget(self.cancel_btn)
   
   def update_progress(self, current: int, total: int, ticker: str):
       """Update progress display"""
       if total > 0:
           progress = int((current / total) * 100)
           self.progress_bar.setValue(progress)
           self.status_label.setText(f"Scanning {current}/{total} stocks...")
       
       self.current_stock_label.setText(f"Analyzing: {ticker}")
   
   def update_results_count(self, count: int):
       """Update results count"""
       self.results_label.setText(f"{count} opportunities found")

class ScanCriteriaWidget(QWidget):
   """Reusable scan criteria widget"""
   
   criteria_changed = Signal()
   
   def __init__(self, parent=None):
       super().__init__(parent)
       self.setup_ui()
   
   def setup_ui(self):
       """Setup criteria UI"""
       layout = QFormLayout(self)
       
       # Basic criteria
       self.min_confidence = QDoubleSpinBox()
       self.min_confidence.setRange(0, 100)
       self.min_confidence.setValue(50)
       self.min_confidence.setSuffix("%")
       self.min_confidence.valueChanged.connect(self.criteria_changed)
       layout.addRow("Min Confidence:", self.min_confidence)
       
       self.max_iv_rv = QDoubleSpinBox()
       self.max_iv_rv.setRange(0.1, 10.0)
       self.max_iv_rv.setValue(2.0)
       self.max_iv_rv.setDecimals(1)
       self.max_iv_rv.valueChanged.connect(self.criteria_changed)
       layout.addRow("Max IV/RV:", self.max_iv_rv)
       
       # Earnings timing
       self.min_days_earnings = QSpinBox()
       self.min_days_earnings.setRange(0, 365)
       self.min_days_earnings.setValue(1)
       self.min_days_earnings.valueChanged.connect(self.criteria_changed)
       layout.addRow("Min Days to Earnings:", self.min_days_earnings)
       
       self.max_days_earnings = QSpinBox()
       self.max_days_earnings.setRange(0, 365)
       self.max_days_earnings.setValue(14)
       self.max_days_earnings.valueChanged.connect(self.criteria_changed)
       layout.addRow("Max Days to Earnings:", self.max_days_earnings)
   
   def get_criteria(self) -> Dict[str, Any]:
       """Get current criteria"""
       return {
           "min_confidence": self.min_confidence.value(),
           "max_iv_rv_ratio": self.max_iv_rv.value(),
           "min_days_to_earnings": self.min_days_earnings.value(),
           "max_days_to_earnings": self.max_days_earnings.value(),
       }
   
   def set_criteria(self, criteria: Dict[str, Any]):
       """Set criteria values"""
       self.min_confidence.setValue(criteria.get("min_confidence", 50))
       self.max_iv_rv.setValue(criteria.get("max_iv_rv_ratio", 2.0))
       self.min_days_earnings.setValue(criteria.get("min_days_to_earnings", 1))
       self.max_days_earnings.setValue(criteria.get("max_days_to_earnings", 14))

class ScannerResultsChart(QWidget):
   """Widget for displaying scanner results charts"""
   
   def __init__(self, parent=None):
       super().__init__(parent)
       self.setup_ui()
   
   def setup_ui(self):
       """Setup chart UI"""
       layout = QVBoxLayout(self)
       
       # Chart controls
       controls_layout = QHBoxLayout()
       
       self.chart_type_combo = QComboBox()
       self.chart_type_combo.addItems([
           "Confidence Distribution", "Sector Breakdown", "IV/RV Analysis",
           "Price vs Volume", "Days to Earnings"
       ])
       self.chart_type_combo.currentTextChanged.connect(self.update_chart)
       
       controls_layout.addWidget(QLabel("Chart Type:"))
       controls_layout.addWidget(self.chart_type_combo)
       controls_layout.addStretch()
       
       layout.addLayout(controls_layout)
       
       # Chart area (placeholder)
       self.chart_label = QLabel("Chart will be displayed here")
       self.chart_label.setAlignment(Qt.AlignCenter)
       self.chart_label.setMinimumHeight(300)
       self.chart_label.setStyleSheet("border: 1px solid gray; background-color: white;")
       layout.addWidget(self.chart_label)
   
   def update_chart(self):
       """Update chart display"""
       chart_type = self.chart_type_combo.currentText()
       self.chart_label.setText(f"{chart_type} chart would be displayed here")
   
   def set_results(self, results: List):
       """Set scan results for charting"""
       # This would process results and create charts
       pass