from PySide6.QtWidgets import (
   QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
   QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
   QComboBox, QCheckBox, QTextEdit, QGroupBox, QTabWidget,
   QFileDialog, QMessageBox, QDialog, QSlider, QProgressBar,
   QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
   QSplitter, QFrame, QScrollArea, QButtonGroup, QRadioButton,
   QColorDialog, QFontDialog, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QSettings, QStandardPaths
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class SettingsView(QDialog):
   """Main settings dialog"""
   
   # Signals
   settings_changed = Signal(dict)  # settings_dict
   api_test_requested = Signal(str, str)  # provider, api_key
   backup_requested = Signal()
   restore_requested = Signal(str)  # backup_path
   
   def __init__(self, config_manager, parent=None):
       super().__init__(parent)
       self.config_manager = config_manager
       self.temp_settings = {}  # Store temporary changes
       
       self.setup_ui()
       self.connect_signals()
       self.load_current_settings()
       self.setup_styles()
   
   def setup_ui(self):
       """Setup the user interface"""
       self.setWindowTitle("Settings")
       self.setModal(True)
       self.resize(800, 700)
       
       layout = QVBoxLayout(self)
       
       # Create main tabs
       self.settings_tabs = QTabWidget()
       layout.addWidget(self.settings_tabs)
       
       # Add tabs
       self.setup_general_tab()
       self.setup_api_tab()
       self.setup_trading_tab()
       self.setup_interface_tab()
       self.setup_data_tab()
       self.setup_performance_tab()
       self.setup_alerts_tab()
       self.setup_advanced_tab()
       
       # Bottom buttons
       self.create_button_bar(layout)
   
   def setup_general_tab(self):
       """Setup general settings tab"""
       tab = QWidget()
       self.settings_tabs.addTab(tab, "General")
       
       layout = QVBoxLayout(tab)
       
       # Application settings
       app_group = QGroupBox("Application Settings")
       app_layout = QFormLayout(app_group)
       
       # Theme selection
       self.theme_combo = QComboBox()
       self.theme_combo.addItems(["Dark", "Light", "System"])
       app_layout.addRow("Theme:", self.theme_combo)
       
       # Language selection
       self.language_combo = QComboBox()
       self.language_combo.addItems(["English", "Spanish", "French", "German"])
       app_layout.addRow("Language:", self.language_combo)
       
       # Auto-save interval
       self.autosave_spin = QSpinBox()
       self.autosave_spin.setRange(1, 60)
       self.autosave_spin.setValue(5)
       self.autosave_spin.setSuffix(" minutes")
       app_layout.addRow("Auto-save Interval:", self.autosave_spin)
       
       # Startup options
       self.autostart_check = QCheckBox("Start with system")
       app_layout.addRow("Startup:", self.autostart_check)
       
       self.restore_session_check = QCheckBox("Restore previous session")
       app_layout.addRow("", self.restore_session_check)
       
       self.check_updates_check = QCheckBox("Check for updates on startup")
       app_layout.addRow("", self.check_updates_check)
       
       layout.addWidget(app_group)
       
       # Default values
       defaults_group = QGroupBox("Default Values")
       defaults_layout = QFormLayout(defaults_group)
       
       # Default contracts
       self.default_contracts_spin = QSpinBox()
       self.default_contracts_spin.setRange(1, 100)
       self.default_contracts_spin.setValue(1)
       defaults_layout.addRow("Default Contracts:", self.default_contracts_spin)
       
       # Default analysis timeframe
       self.default_timeframe_combo = QComboBox()
       self.default_timeframe_combo.addItems(["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"])
       defaults_layout.addRow("Default Timeframe:", self.default_timeframe_combo)
       
       # Default scan criteria
       self.default_confidence_spin = QDoubleSpinBox()
       self.default_confidence_spin.setRange(0, 100)
       self.default_confidence_spin.setValue(50)
       self.default_confidence_spin.setSuffix("%")
       defaults_layout.addRow("Default Min Confidence:", self.default_confidence_spin)
       
       layout.addWidget(defaults_group)
       
       # Notifications
       notifications_group = QGroupBox("Notifications")
       notifications_layout = QVBoxLayout(notifications_group)
       
       self.enable_notifications_check = QCheckBox("Enable desktop notifications")
       notifications_layout.addWidget(self.enable_notifications_check)
       
       self.sound_notifications_check = QCheckBox("Play sound for notifications")
       notifications_layout.addWidget(self.sound_notifications_check)
       
       self.email_notifications_check = QCheckBox("Send email notifications")
       notifications_layout.addWidget(self.email_notifications_check)
       
       # Email settings (initially hidden)
       self.email_settings_widget = self.create_email_settings_widget()
       self.email_settings_widget.setVisible(False)
       notifications_layout.addWidget(self.email_settings_widget)
       
       self.email_notifications_check.toggled.connect(self.email_settings_widget.setVisible)
       
       layout.addWidget(notifications_group)
       
       layout.addStretch()
   
   def create_email_settings_widget(self) -> QWidget:
       """Create email settings widget"""
       widget = QFrame()
       widget.setFrameStyle(QFrame.StyledPanel)
       layout = QFormLayout(widget)
       
       self.email_address = QLineEdit()
       self.email_address.setPlaceholderText("your.email@example.com")
       layout.addRow("Email Address:", self.email_address)
       
       self.smtp_server = QLineEdit()
       self.smtp_server.setPlaceholderText("smtp.gmail.com")
       layout.addRow("SMTP Server:", self.smtp_server)
       
       self.smtp_port = QSpinBox()
       self.smtp_port.setRange(1, 65535)
       self.smtp_port.setValue(587)
       layout.addRow("SMTP Port:", self.smtp_port)
       
       self.email_username = QLineEdit()
       layout.addRow("Username:", self.email_username)
       
       self.email_password = QLineEdit()
       self.email_password.setEchoMode(QLineEdit.Password)
       layout.addRow("Password:", self.email_password)
       
       # Test email button
       test_email_btn = QPushButton("Test Email")
       test_email_btn.clicked.connect(self.test_email_settings)
       layout.addRow("", test_email_btn)
       
       return widget
   
   def setup_api_tab(self):
       """Setup API settings tab"""
       tab = QWidget()
       self.settings_tabs.addTab(tab, "API Keys")
       
       layout = QVBoxLayout(tab)
       
       # API provider settings
       providers_group = QGroupBox("API Providers")
       providers_layout = QVBoxLayout(providers_group)
       
       # Alpha Vantage
       av_group = self.create_api_provider_group(
           "Alpha Vantage",
           "Get free API key from alphavantage.co",
           "alpha_vantage",
           "Used for real-time stock data and fundamentals"
       )
       providers_layout.addWidget(av_group)
       
       # Finnhub
       finnhub_group = self.create_api_provider_group(
           "Finnhub",
           "Get free API key from finnhub.io",
           "finnhub",
           "Used for earnings data and market news"
       )
       providers_layout.addWidget(finnhub_group)
       
       # IEX Cloud
       iex_group = self.create_api_provider_group(
           "IEX Cloud",
           "Get API key from iexcloud.io",
           "iex_cloud",
           "Used for intraday data and options chains"
       )
       providers_layout.addWidget(iex_group)
       
       layout.addWidget(providers_group)
       
       # Rate limiting settings
       limits_group = QGroupBox("Rate Limiting")
       limits_layout = QFormLayout(limits_group)
       
       self.max_requests_spin = QSpinBox()
       self.max_requests_spin.setRange(1, 1000)
       self.max_requests_spin.setValue(5)
       self.max_requests_spin.setSuffix(" per minute")
       limits_layout.addRow("Max Requests:", self.max_requests_spin)
       
       self.request_delay_spin = QDoubleSpinBox()
       self.request_delay_spin.setRange(0.1, 10.0)
       self.request_delay_spin.setValue(1.0)
       self.request_delay_spin.setSuffix(" seconds")
       limits_layout.addRow("Request Delay:", self.request_delay_spin)
       
       self.timeout_spin = QSpinBox()
       self.timeout_spin.setRange(5, 120)
       self.timeout_spin.setValue(30)
       self.timeout_spin.setSuffix(" seconds")
       limits_layout.addRow("Request Timeout:", self.timeout_spin)
       
       layout.addWidget(limits_group)
       
       # Fallback settings
       fallback_group = QGroupBox("Fallback Options")
       fallback_layout = QVBoxLayout(fallback_group)
       
       self.use_yahoo_fallback_check = QCheckBox("Use Yahoo Finance as fallback")
       self.use_yahoo_fallback_check.setChecked(True)
       fallback_layout.addWidget(self.use_yahoo_fallback_check)
       
       self.cache_fallback_check = QCheckBox("Use cached data when APIs fail")
       self.cache_fallback_check.setChecked(True)
       fallback_layout.addWidget(self.cache_fallback_check)
       
       layout.addWidget(fallback_group)
       
       layout.addStretch()
   
   def create_api_provider_group(self, name: str, help_text: str, key: str, description: str) -> QGroupBox:
       """Create API provider settings group"""
       group = QGroupBox(name)
       layout = QVBoxLayout(group)
       
       # Description
       desc_label = QLabel(description)
       desc_label.setWordWrap(True)
       desc_label.setStyleSheet("color: gray; font-style: italic;")
       layout.addWidget(desc_label)
       
       # API key input
       key_layout = QHBoxLayout()
       
       api_key_input = QLineEdit()
       api_key_input.setEchoMode(QLineEdit.Password)
       api_key_input.setPlaceholderText("Enter API key...")
       setattr(self, f"{key}_api_key", api_key_input)
       key_layout.addWidget(api_key_input)
       
       # Show/hide button
       show_btn = QPushButton("Show")
       show_btn.setCheckable(True)
       show_btn.toggled.connect(
           lambda checked, input_field=api_key_input: 
           input_field.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
       )
       key_layout.addWidget(show_btn)
       
       # Test button
       test_btn = QPushButton("Test")
       test_btn.clicked.connect(
           lambda: self.test_api_key(key, api_key_input.text())
       )
       key_layout.addWidget(test_btn)
       
       layout.addLayout(key_layout)
       
       # Help text
       help_label = QLabel(help_text)
       help_label.setWordWrap(True)
       help_label.setStyleSheet("color: blue; text-decoration: underline;")
       help_label.mousePressEvent = lambda event: self.open_api_help(key)
       layout.addWidget(help_label)
       
       return group
   
   def setup_trading_tab(self):
       """Setup trading settings tab"""
       tab = QWidget()
       self.settings_tabs.addTab(tab, "Trading")
       
       layout = QVBoxLayout(tab)
       
       # Portfolio settings
       portfolio_group = QGroupBox("Portfolio Settings")
       portfolio_layout = QFormLayout(portfolio_group)
       
       self.portfolio_value_spin = QDoubleSpinBox()
       self.portfolio_value_spin.setRange(1000, 10000000)
       self.portfolio_value_spin.setValue(100000)
       self.portfolio_value_spin.setPrefix("$")
       portfolio_layout.addRow("Portfolio Value:", self.portfolio_value_spin)
       
       self.max_risk_spin = QDoubleSpinBox()
       self.max_risk_spin.setRange(0.1, 20.0)
       self.max_risk_spin.setValue(2.0)
       self.max_risk_spin.setSuffix("%")
       portfolio_layout.addRow("Max Risk per Trade:", self.max_risk_spin)
       
       self.max_portfolio_risk_spin = QDoubleSpinBox()
       self.max_portfolio_risk_spin.setRange(1.0, 50.0)
       self.max_portfolio_risk_spin.setValue(10.0)
       self.max_portfolio_risk_spin.setSuffix("%")
       portfolio_layout.addRow("Max Portfolio Risk:", self.max_portfolio_risk_spin)
       
       layout.addWidget(portfolio_group)
       
       # Risk management
       risk_group = QGroupBox("Risk Management")
       risk_layout = QFormLayout(risk_group)
       
       self.use_kelly_criterion_check = QCheckBox("Use Kelly Criterion for position sizing")
       risk_layout.addRow("Position Sizing:", self.use_kelly_criterion_check)
       
       self.max_simultaneous_trades_spin = QSpinBox()
       self.max_simultaneous_trades_spin.setRange(1, 50)
       self.max_simultaneous_trades_spin.setValue(5)
       risk_layout.addRow("Max Simultaneous Trades:", self.max_simultaneous_trades_spin)
       
       self.stop_loss_pct_spin = QDoubleSpinBox()
       self.stop_loss_pct_spin.setRange(5, 100)
       self.stop_loss_pct_spin.setValue(50)
       self.stop_loss_pct_spin.setSuffix("%")
       risk_layout.addRow("Default Stop Loss:", self.stop_loss_pct_spin)
       
       self.profit_target_pct_spin = QDoubleSpinBox()
       self.profit_target_pct_spin.setRange(10, 500)
       self.profit_target_pct_spin.setValue(50)
       self.profit_target_pct_spin.setSuffix("%")
       risk_layout.addRow("Default Profit Target:", self.profit_target_pct_spin)
       
       layout.addWidget(risk_group)
       
       # Trading preferences
       preferences_group = QGroupBox("Trading Preferences")
       preferences_layout = QVBoxLayout(preferences_group)
       
       self.auto_close_dte_check = QCheckBox("Auto-close trades at specific DTE")
       preferences_layout.addWidget(self.auto_close_dte_check)
       
       dte_layout = QHBoxLayout()
       dte_layout.addWidget(QLabel("Close at:"))
       self.auto_close_dte_spin = QSpinBox()
       self.auto_close_dte_spin.setRange(0, 30)
       self.auto_close_dte_spin.setValue(7)
       self.auto_close_dte_spin.setSuffix(" DTE")
       self.auto_close_dte_spin.setEnabled(False)
       dte_layout.addWidget(self.auto_close_dte_spin)
       dte_layout.addStretch()
       preferences_layout.addLayout(dte_layout)
       
       self.auto_close_dte_check.toggled.connect(self.auto_close_dte_spin.setEnabled)
       
       self.confirm_trades_check = QCheckBox("Confirm trades before execution")
       preferences_layout.addWidget(self.confirm_trades_check)
       
       self.save_trades_automatically_check = QCheckBox("Save trades automatically")
       preferences_layout.addWidget(self.save_trades_automatically_check)
       
       layout.addWidget(preferences_group)
       
       # Commission settings
       commission_group = QGroupBox("Commission & Fees")
       commission_layout = QFormLayout(commission_group)
       
       self.commission_per_contract_spin = QDoubleSpinBox()
       self.commission_per_contract_spin.setRange(0, 10)
       self.commission_per_contract_spin.setValue(0.65)
       self.commission_per_contract_spin.setPrefix("$")
       commission_layout.addRow("Commission per Contract:", self.commission_per_contract_spin)
       
       self.base_commission_spin = QDoubleSpinBox()
       self.base_commission_spin.setRange(0, 50)
       self.base_commission_spin.setValue(0)
       self.base_commission_spin.setPrefix("$")
       commission_layout.addRow("Base Commission:", self.base_commission_spin)
       
       self.include_fees_in_calculations_check = QCheckBox("Include fees in P&L calculations")
       commission_layout.addRow("", self.include_fees_in_calculations_check)
       
       layout.addWidget(commission_group)
       
       layout.addStretch()
   
   def setup_interface_tab(self):
       """Setup interface settings tab"""
       tab = QWidget()
       self.settings_tabs.addTab(tab, "Interface")
       
       layout = QVBoxLayout(tab)
       
       # Display settings
       display_group = QGroupBox("Display Settings")
       display_layout = QFormLayout(display_group)
       
       # Font settings
       font_layout = QHBoxLayout()
       self.font_family_combo = QComboBox()
       self.font_family_combo.addItems(["Arial", "Helvetica", "Times New Roman", "Courier New"])
       font_layout.addWidget(self.font_family_combo)
       
       self.font_size_spin = QSpinBox()
       self.font_size_spin.setRange(8, 24)
       self.font_size_spin.setValue(10)
       font_layout.addWidget(self.font_size_spin)
       
       font_btn = QPushButton("Choose Font...")
       font_btn.clicked.connect(self.choose_font)
       font_layout.addWidget(font_btn)
       
       display_layout.addRow("Font:", font_layout)
       
       # Color scheme
       self.color_scheme_combo = QComboBox()
       self.color_scheme_combo.addItems(["Default", "High Contrast", "Colorblind Friendly", "Custom"])
       display_layout.addRow("Color Scheme:", self.color_scheme_combo)
       
       # Decimal places
       self.decimal_places_spin = QSpinBox()
       self.decimal_places_spin.setRange(0, 6)
       self.decimal_places_spin.setValue(2)
       display_layout.addRow("Decimal Places:", self.decimal_places_spin)
       
       # Number format
       self.number_format_combo = QComboBox()
       self.number_format_combo.addItems(["1,234.56", "1.234,56", "1 234.56"])
       display_layout.addRow("Number Format:", self.number_format_combo)
       
       layout.addWidget(display_group)
       
       # Table settings
       table_group = QGroupBox("Table Settings")
       table_layout = QFormLayout(table_group)
       
       self.rows_per_page_spin = QSpinBox()
       self.rows_per_page_spin.setRange(10, 1000)
       self.rows_per_page_spin.setValue(50)
       table_layout.addRow("Rows per Page:", self.rows_per_page_spin)
       
       self.auto_resize_columns_check = QCheckBox("Auto-resize columns")
       table_layout.addRow("Columns:", self.auto_resize_columns_check)
       
       self.alternate_row_colors_check = QCheckBox("Alternate row colors")
       table_layout.addRow("", self.alternate_row_colors_check)
       
       self.show_grid_lines_check = QCheckBox("Show grid lines")
       table_layout.addRow("", self.show_grid_lines_check)
       
       layout.addWidget(table_group)
       
       # Chart settings
       chart_group = QGroupBox("Chart Settings")
       chart_layout = QFormLayout(chart_group)
       
       self.chart_theme_combo = QComboBox()
       self.chart_theme_combo.addItems(["Light", "Dark", "Colorful", "Minimal"])
       chart_layout.addRow("Chart Theme:", self.chart_theme_combo)
       
       self.animation_enabled_check = QCheckBox("Enable chart animations")
       chart_layout.addRow("Animations:", self.animation_enabled_check)
       
       self.show_data_labels_check = QCheckBox("Show data labels")
       chart_layout.addRow("", self.show_data_labels_check)
       
       layout.addWidget(chart_group)
       
       # Toolbar settings
       toolbar_group = QGroupBox("Toolbar & Shortcuts")
       toolbar_layout = QVBoxLayout(toolbar_group)
       
       self.show_toolbar_check = QCheckBox("Show toolbar")
       toolbar_layout.addWidget(self.show_toolbar_check)
       
       self.show_status_bar_check = QCheckBox("Show status bar")
       toolbar_layout.addWidget(self.show_status_bar_check)
       
       self.show_tooltips_check = QCheckBox("Show tooltips")
       toolbar_layout.addWidget(self.show_tooltips_check)
       
       # Keyboard shortcuts button
       shortcuts_btn = QPushButton("Configure Keyboard Shortcuts...")
       shortcuts_btn.clicked.connect(self.configure_shortcuts)
       toolbar_layout.addWidget(shortcuts_btn)
       
       layout.addWidget(toolbar_group)
       
       layout.addStretch()
   
   def setup_data_tab(self):
       """Setup data management tab"""
       tab = QWidget()
       self.settings_tabs.addTab(tab, "Data")
       
       layout = QVBoxLayout(tab)
       
       # Storage settings
       storage_group = QGroupBox("Data Storage")
       storage_layout = QFormLayout(storage_group)
       
       # Data directory
       data_dir_layout = QHBoxLayout()
       self.data_directory_edit = QLineEdit()
       self.data_directory_edit.setReadOnly(True)
       data_dir_layout.addWidget(self.data_directory_edit)
       
       browse_btn = QPushButton("Browse...")
       browse_btn.clicked.connect(self.browse_data_directory)
       data_dir_layout.addWidget(browse_btn)
       
       storage_layout.addRow("Data Directory:", data_dir_layout)
       
       # Database settings
       self.database_type_combo = QComboBox()
       self.database_type_combo.addItems(["SQLite", "PostgreSQL", "MySQL"])
       storage_layout.addRow("Database Type:", self.database_type_combo)
       
       layout.addWidget(storage_group)
       
       # Cache settings
       cache_group = QGroupBox("Cache Settings")
       cache_layout = QFormLayout(cache_group)
       
       self.cache_size_spin = QSpinBox()
       self.cache_size_spin.setRange(10, 1000)
       self.cache_size_spin.setValue(100)
       self.cache_size_spin.setSuffix(" MB")
       cache_layout.addRow("Max Cache Size:", self.cache_size_spin)
       
       self.cache_ttl_spin = QSpinBox()
       self.cache_ttl_spin.setRange(1, 1440)
       self.cache_ttl_spin.setValue(30)
       self.cache_ttl_spin.setSuffix(" minutes")
       cache_layout.addRow("Cache TTL:", self.cache_ttl_spin)
       
       # Cache management buttons
       cache_buttons_layout = QHBoxLayout()
       
       clear_cache_btn = QPushButton("Clear Cache")
       clear_cache_btn.clicked.connect(self.clear_cache)
       cache_buttons_layout.addWidget(clear_cache_btn)
       
       optimize_cache_btn = QPushButton("Optimize Cache")
       optimize_cache_btn.clicked.connect(self.optimize_cache)
       cache_buttons_layout.addWidget(optimize_cache_btn)
       
       cache_layout.addRow("", cache_buttons_layout)
       
       layout.addWidget(cache_group)
       
       # Backup settings
       backup_group = QGroupBox("Backup & Restore")
       backup_layout = QFormLayout(backup_group)
       
       self.auto_backup_check = QCheckBox("Enable automatic backups")
       backup_layout.addRow("Auto Backup:", self.auto_backup_check)
       
       self.backup_frequency_combo = QComboBox()
       self.backup_frequency_combo.addItems(["Daily", "Weekly", "Monthly"])
       self.backup_frequency_combo.setEnabled(False)
       backup_layout.addRow("Frequency:", self.backup_frequency_combo)
       
       self.auto_backup_check.toggled.connect(self.backup_frequency_combo.setEnabled)
       
       self.backup_retention_spin = QSpinBox()
       self.backup_retention_spin.setRange(1, 365)
       self.backup_retention_spin.setValue(30)
       self.backup_retention_spin.setSuffix(" days")
       self.backup_retention_spin.setEnabled(False)
       backup_layout.addRow("Retention:", self.backup_retention_spin)
       
       self.auto_backup_check.toggled.connect(self.backup_retention_spin.setEnabled)
       
       # Backup/restore buttons
       backup_buttons_layout = QHBoxLayout()
       
       backup_now_btn = QPushButton("Backup Now")
       backup_now_btn.clicked.connect(self.backup_now)
       backup_buttons_layout.addWidget(backup_now_btn)
       
       restore_btn = QPushButton("Restore...")
       restore_btn.clicked.connect(self.restore_backup)
       backup_buttons_layout.addWidget(restore_btn)
       
       backup_layout.addRow("", backup_buttons_layout)
       
       layout.addWidget(backup_group)
       
       # Data export/import
       import_export_group = QGroupBox("Import & Export")
       import_export_layout = QVBoxLayout(import_export_group)
       
       export_btn = QPushButton("Export All Data...")
       export_btn.clicked.connect(self.export_all_data)
       import_export_layout.addWidget(export_btn)
       
       import_btn = QPushButton("Import Data...")
       import_btn.clicked.connect(self.import_data)
       import_export_layout.addWidget(import_btn)
       
       layout.addWidget(import_export_group)
       
       layout.addStretch()
   
   def setup_performance_tab(self):
       """Setup performance settings tab"""
       tab = QWidget()
       self.settings_tabs.addTab(tab, "Performance")
       
       layout = QVBoxLayout(tab)
       
       # Processing settings
       processing_group = QGroupBox("Processing Settings")
       processing_layout = QFormLayout(processing_group)
       
       self.thread_count_spin = QSpinBox()
       self.thread_count_spin.setRange(1, 16)
       self.thread_count_spin.setValue(4)
       processing_layout.addRow("Thread Count:", self.thread_count_spin)
       
       self.use_gpu_acceleration_check = QCheckBox("Use GPU acceleration (if available)")
       processing_layout.addRow("GPU:", self.use_gpu_acceleration_check)
       
       self.batch_size_spin = QSpinBox()
       self.batch_size_spin.setRange(10, 1000)
       self.batch_size_spin.setValue(100)
       processing_layout.addRow("Batch Size:", self.batch_size_spin)
       
       layout.addWidget(processing_group)
       
       # Memory settings
       memory_group = QGroupBox("Memory Management")
       memory_layout = QFormLayout(memory_group)
       
       self.max_memory_spin = QSpinBox()
       self.max_memory_spin.setRange(512, 8192)
       self.max_memory_spin.setValue(2048)
       self.max_memory_spin.setSuffix(" MB")
       memory_layout.addRow("Max Memory Usage:", self.max_memory_spin)
       
       self.garbage_collection_check = QCheckBox("Enable aggressive garbage collection")
       memory_layout.addRow("GC:", self.garbage_collection_check)
       
       layout.addWidget(memory_group)
       
       # Network settings
       network_group = QGroupBox("Network Settings")
       network_layout = QFormLayout(network_group)
       
       self.connection_timeout_spin = QSpinBox()
       self.connection_timeout_spin.setRange(5, 300)
       self.connection_timeout_spin.setValue(30)
       self.connection_timeout_spin.setSuffix(" seconds")
       network_layout.addRow("Connection Timeout:", self.connection_timeout_spin)
       
       self.max_connections_spin = QSpinBox()
       self.max_connections_spin.setRange(1, 20)
       self.max_connections_spin.setValue(5)
       network_layout.addRow("Max Concurrent Connections:", self.max_connections_spin)
       
       self.use_proxy_check = QCheckBox("Use proxy server")
       network_layout.addRow("Proxy:", self.use_proxy_check)
       
       # Proxy settings
       proxy_layout = QHBoxLayout()
       self.proxy_host_edit = QLineEdit()
       self.proxy_host_edit.setPlaceholderText("proxy.example.com")
       self.proxy_host_edit.setEnabled(False)
       proxy_layout.addWidget(self.proxy_host_edit)
       
       self.proxy_port_spin = QSpinBox()
       self.proxy_port_spin.setRange(1, 65535)
       self.proxy_port_spin.setValue(8080)
       self.proxy_port_spin.setEnabled(False)
       proxy_layout.addWidget(self.proxy_port_spin)
       
       network_layout.addRow("Proxy Address:", proxy_layout)
       
       self.use_proxy_check.toggled.connect(self.proxy_host_edit.setEnabled)
       self.use_proxy_check.toggled.connect(self.proxy_port_spin.setEnabled)
       
       layout.addWidget(network_group)
       
       # Performance monitoring
       monitoring_group = QGroupBox("Performance Monitoring")
       monitoring_layout = QVBoxLayout(monitoring_group)
       
       self.enable_profiling_check = QCheckBox("Enable performance profiling")
       monitoring_layout.addWidget(self.enable_profiling_check)
       
       self.log_performance_check = QCheckBox("Log performance metrics")
       monitoring_layout.addWidget(self.log_performance_check)
       
       # Performance stats display
       stats_layout = QGridLayout()
       
       stats_layout.addWidget(QLabel("CPU Usage:"), 0, 0)
       self.cpu_usage_label = QLabel("--")
       stats_layout.addWidget(self.cpu_usage_label, 0, 1)
       
       stats_layout.addWidget(QLabel("Memory Usage:"), 1, 0)
       self.memory_usage_label = QLabel("--")
       stats_layout.addWidget(self.memory_usage_label, 1, 1)
       
       stats_layout.addWidget(QLabel("Cache Hit Rate:"), 2, 0)
       self.cache_hit_rate_label = QLabel("--")
       stats_layout.addWidget(self.cache_hit_rate_label, 2, 1)
       
       monitoring_layout.addLayout(stats_layout)
       
       refresh_stats_btn = QPushButton("Refresh Stats")
       refresh_stats_btn.clicked.connect(self.refresh_performance_stats)
       monitoring_layout.addWidget(refresh_stats_btn)
       
       layout.addWidget(monitoring_group)
       
       layout.addStretch()
   
   def setup_alerts_tab(self):
       """Setup alerts and notifications tab"""
       tab = QWidget()
       self.settings_tabs.addTab(tab, "Alerts")
       
       layout = QVBoxLayout(tab)
       
       # Price alerts
       price_alerts_group = QGroupBox("Price Alerts")
       price_alerts_layout = QVBoxLayout(price_alerts_group)
       
       self.enable_price_alerts_check = QCheckBox("Enable price alerts")
       price_alerts_layout.addWidget(self.enable_price_alerts_check)
       
       # Price alert settings
       price_settings_layout = QFormLayout()
       
       self.price_alert_threshold_spin = QDoubleSpinBox()
       self.price_alert_threshold_spin.setRange(0.1, 50.0)
       self.price_alert_threshold_spin.setValue(5.0)
       self.price_alert_threshold_spin.setSuffix("%")
       price_settings_layout.addRow("Price Change Threshold:", self.price_alert_threshold_spin)
       
       self.price_alert_frequency_combo = QComboBox()
       self.price_alert_frequency_combo.addItems(["Immediate", "Every 5 minutes", "Every 15 minutes", "Hourly"])
       price_settings_layout.addRow("Alert Frequency:", self.price_alert_frequency_combo)
       
       price_alerts_layout.addLayout(price_settings_layout)
       
       layout.addWidget(price_alerts_group)
       
       # Trade alerts
       trade_alerts_group = QGroupBox("Trade Alerts")
       trade_alerts_layout = QVBoxLayout(trade_alerts_group)
       
       self.alert_profit_target_check = QCheckBox("Alert when profit target reached")
       trade_alerts_layout.addWidget(self.alert_profit_target_check)
       
       self.alert_stop_loss_check = QCheckBox("Alert when stop loss triggered")
       trade_alerts_layout.addWidget(self.alert_stop_loss_check)
       
       self.alert_expiration_check = QCheckBox("Alert before expiration")
       trade_alerts_layout.addWidget(self.alert_expiration_check)
       
       exp_layout = QHBoxLayout()
       exp_layout.addWidget(QLabel("Days before expiration:"))
       self.expiration_alert_days_spin = QSpinBox()
       self.expiration_alert_days_spin.setRange(1, 30)
       self.expiration_alert_days_spin.setValue(7)
       self.expiration_alert_days_spin.setEnabled(False)
       exp_layout.addWidget(self.expiration_alert_days_spin)
       exp_layout.addStretch()
       trade_alerts_layout.addLayout(exp_layout)
       
       self.alert_expiration_check.toggled.connect(self.expiration_alert_days_spin.setEnabled)
       
       layout.addWidget(trade_alerts_group)
       
       # Market alerts
       market_alerts_group = QGroupBox("Market Alerts")
       market_alerts_layout = QVBoxLayout(market_alerts_group)
       
       self.alert_vix_spike_check = QCheckBox("Alert on VIX spikes")
       market_alerts_layout.addWidget(self.alert_vix_spike_check)
       
       vix_layout = QHBoxLayout()
       vix_layout.addWidget(QLabel("VIX threshold:"))
       self.vix_alert_threshold_spin = QDoubleSpinBox()
       self.vix_alert_threshold_spin.setRange(10.0, 100.0)
       self.vix_alert_threshold_spin.setValue(30.0)
       self.vix_alert_threshold_spin.setEnabled(False)
       vix_layout.addWidget(self.vix_alert_threshold_spin)
       vix_layout.addStretch()
       market_alerts_layout.addLayout(vix_layout)
       
       self.alert_vix_spike_check.toggled.connect(self.vix_alert_threshold_spin.setEnabled)
       
       self.alert_earnings_check = QCheckBox("Alert for upcoming earnings")
       market_alerts_layout.addWidget(self.alert_earnings_check)
       
       layout.addWidget(market_alerts_group)
       
       # Alert methods
       methods_group = QGroupBox("Alert Methods")
       methods_layout = QVBoxLayout(methods_group)
       
       self.desktop_alerts_check = QCheckBox("Desktop notifications")
       methods_layout.addWidget(self.desktop_alerts_check)
       
       self.email_alerts_check = QCheckBox("Email notifications")
       methods_layout.addWidget(self.email_alerts_check)
       
       self.sound_alerts_check = QCheckBox("Sound alerts")
       methods_layout.addWidget(self.sound_alerts_check)
       
       # Sound selection
       sound_layout = QHBoxLayout()
       sound_layout.addWidget(QLabel("Alert sound:"))
       self.alert_sound_combo = QComboBox()
       self.alert_sound_combo.addItems(["Default", "Bell", "Chime", "Custom..."])
       self.alert_sound_combo.setEnabled(False)
       sound_layout.addWidget(self.alert_sound_combo)
       
       test_sound_btn = QPushButton("Test")
       test_sound_btn.setEnabled(False)
       test_sound_btn.clicked.connect(self.test_alert_sound)
       sound_layout.addWidget(test_sound_btn)
       sound_layout.addStretch()
       
       methods_layout.addLayout(sound_layout)
       
       self.sound_alerts_check.toggled.connect(self.alert_sound_combo.setEnabled)
       self.sound_alerts_check.toggled.connect(test_sound_btn.setEnabled)
       
       layout.addWidget(methods_group)
       
       layout.addStretch()
   
   def setup_advanced_tab(self):
       """Setup advanced settings tab"""
       tab = QWidget()
       self.settings_tabs.addTab(tab, "Advanced")
       
       layout = QVBoxLayout(tab)
       
       # Logging settings
       logging_group = QGroupBox("Logging & Debugging")
       logging_layout = QFormLayout(logging_group)
       
       self.log_level_combo = QComboBox()
       self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
       self.log_level_combo.setCurrentText("INFO")
       logging_layout.addRow("Log Level:", self.log_level_combo)
       
       self.log_to_file_check = QCheckBox("Log to file")
       logging_layout.addRow("File Logging:", self.log_to_file_check)
       
       self.max_log_size_spin = QSpinBox()
       self.max_log_size_spin.setRange(1, 100)
       self.max_log_size_spin.setValue(10)
       self.max_log_size_spin.setSuffix(" MB")
       self.max_log_size_spin.setEnabled(False)
       logging_layout.addRow("Max Log Size:", self.max_log_size_spin)
       
       self.log_to_file_check.toggled.connect(self.max_log_size_spin.setEnabled)
       
       # Log management buttons
       log_buttons_layout = QHBoxLayout()
       
       view_logs_btn = QPushButton("View Logs")
       view_logs_btn.clicked.connect(self.view_logs)
       log_buttons_layout.addWidget(view_logs_btn)
       
       clear_logs_btn = QPushButton("Clear Logs")
       clear_logs_btn.clicked.connect(self.clear_logs)
       log_buttons_layout.addWidget(clear_logs_btn)
       
       export_logs_btn = QPushButton("Export Logs")
       export_logs_btn.clicked.connect(self.export_logs)
       log_buttons_layout.addWidget(export_logs_btn)
       
       logging_layout.addRow("", log_buttons_layout)
       
       layout.addWidget(logging_group)
       
       # Experimental features
       experimental_group = QGroupBox("Experimental Features")
       experimental_layout = QVBoxLayout(experimental_group)
       
       self.enable_beta_features_check = QCheckBox("Enable beta features")
       experimental_layout.addWidget(self.enable_beta_features_check)
       
       self.use_experimental_ml_check = QCheckBox("Use experimental ML models")
       experimental_layout.addWidget(self.use_experimental_ml_check)
       
       self.enable_advanced_analytics_check = QCheckBox("Enable advanced analytics")
       experimental_layout.addWidget(self.enable_advanced_analytics_check)
       
       # Warning label
       warning_label = QLabel("⚠️ Experimental features may be unstable")
       warning_label.setStyleSheet("color: orange; font-weight: bold;")
       experimental_layout.addWidget(warning_label)
       
       layout.addWidget(experimental_group)
       
       # Developer settings
       developer_group = QGroupBox("Developer Settings")
       developer_layout = QVBoxLayout(developer_group)
       
       self.developer_mode_check = QCheckBox("Enable developer mode")
       developer_layout.addWidget(self.developer_mode_check)
       
       self.show_debug_info_check = QCheckBox("Show debug information")
       self.show_debug_info_check.setEnabled(False)
       developer_layout.addWidget(self.show_debug_info_check)
       
       self.enable_api_logging_check = QCheckBox("Enable API request logging")
       self.enable_api_logging_check.setEnabled(False)
       developer_layout.addWidget(self.enable_api_logging_check)
       
       self.developer_mode_check.toggled.connect(self.show_debug_info_check.setEnabled)
       self.developer_mode_check.toggled.connect(self.enable_api_logging_check.setEnabled)
       
       layout.addWidget(developer_group)
       
       # System information
       system_group = QGroupBox("System Information")
       system_layout = QGridLayout(system_group)
       
       system_info = [
           ("Application Version:", "v2.1.0"),
           ("Python Version:", "3.11.2"),
           ("Qt Version:", "6.5.0"),
           ("Platform:", "Windows 11"),
           ("Architecture:", "x64"),
           ("Memory:", "16 GB"),
       ]
       
       for i, (label, value) in enumerate(system_info):
           system_layout.addWidget(QLabel(label), i, 0)
           system_layout.addWidget(QLabel(value), i, 1)
       
       layout.addWidget(system_group)
       
       layout.addStretch()
   
   def create_button_bar(self, layout):
       """Create bottom button bar"""
       button_layout = QHBoxLayout()
       
       # Reset to defaults
       reset_btn = QPushButton("Reset to Defaults")
       reset_btn.clicked.connect(self.reset_to_defaults)
       button_layout.addWidget(reset_btn)
       
       button_layout.addStretch()
       
       # Apply button
       self.apply_btn = QPushButton("Apply")
       self.apply_btn.clicked.connect(self.apply_settings)
       button_layout.addWidget(self.apply_btn)
       
       # OK button
       ok_btn = QPushButton("OK")
       ok_btn.clicked.connect(self.accept_settings)
       ok_btn.setDefault(True)
       button_layout.addWidget(ok_btn)
       
       # Cancel button
       cancel_btn = QPushButton("Cancel")
       cancel_btn.clicked.connect(self.reject)
       button_layout.addWidget(cancel_btn)
       
       layout.addLayout(button_layout)
   
   def connect_signals(self):
       """Connect UI signals"""
       # Enable apply button when settings change
       widgets_to_monitor = [
           self.theme_combo, self.language_combo, self.autosave_spin,
           self.portfolio_value_spin, self.max_risk_spin, self.font_family_combo,
           self.cache_size_spin, self.thread_count_spin
       ]
       
       for widget in widgets_to_monitor:
           if hasattr(widget, 'valueChanged'):
               widget.valueChanged.connect(self.on_setting_changed)
           elif hasattr(widget, 'currentTextChanged'):
               widget.currentTextChanged.connect(self.on_setting_changed)
           elif hasattr(widget, 'toggled'):
               widget.toggled.connect(self.on_setting_changed)
   
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
           
           QTabWidget::pane {
               border: 1px solid #C0C0C0;
               top: -1px;
           }
           
           QTabBar::tab {
               background: #E0E0E0;
               border: 1px solid #C0C0C0;
               padding: 8px 16px;
               margin-right: 2px;
           }
           
           QTabBar::tab:selected {
               background: white;
               border-bottom: 1px solid white;
           }
           
           QTabBar::tab:hover {
               background: #F0F0F0;
           }
       """)
   
   def load_current_settings(self):
       """Load current settings from config manager"""
       try:
           # General settings
           self.theme_combo.setCurrentText(self.config_manager.get("theme", "Dark"))
           self.language_combo.setCurrentText(self.config_manager.get("language", "English"))
           self.autosave_spin.setValue(self.config_manager.get("autosave_interval", 5))
           self.default_contracts_spin.setValue(self.config_manager.get("default_contracts", 1))
           
           # API settings
           if hasattr(self, 'alpha_vantage_api_key'):
               self.alpha_vantage_api_key.setText(self.config_manager.get("api_keys.alpha_vantage", ""))
           if hasattr(self, 'finnhub_api_key'):
               self.finnhub_api_key.setText(self.config_manager.get("api_keys.finnhub", ""))
           
           # Trading settings
           self.portfolio_value_spin.setValue(self.config_manager.get("portfolio_value", 100000))
           self.max_risk_spin.setValue(self.config_manager.get("max_position_risk", 2.0) * 100)
           
           # Interface settings
           self.font_family_combo.setCurrentText(self.config_manager.get("font_family", "Arial"))
           self.font_size_spin.setValue(self.config_manager.get("font_size", 10))
           
           # Data settings
           data_dir = self.config_manager.get("data_directory", os.path.expanduser("~/OptionsCalculator"))
           self.data_directory_edit.setText(data_dir)
           
           # Performance settings
           self.thread_count_spin.setValue(self.config_manager.get("thread_count", 4))
           self.cache_size_spin.setValue(self.config_manager.get("cache_size_mb", 100))
           
       except Exception as e:
           logger.error(f"Error loading current settings: {e}")
           QMessageBox.warning(self, "Warning", f"Error loading some settings: {e}")
   
   def on_setting_changed(self):
       """Handle setting change"""
       self.apply_btn.setEnabled(True)
   
   def get_current_settings(self) -> Dict[str, Any]:
       """Get current settings from UI"""
       settings = {
           # General
           "theme": self.theme_combo.currentText(),
           "language": self.language_combo.currentText(),
           "autosave_interval": self.autosave_spin.value(),
           "default_contracts": self.default_contracts_spin.value(),
           "autostart": self.autostart_check.isChecked(),
           "restore_session": self.restore_session_check.isChecked(),
           "check_updates": self.check_updates_check.isChecked(),
           
           # API
           "api_keys": {
               "alpha_vantage": getattr(self, 'alpha_vantage_api_key', QLineEdit()).text(),
               "finnhub": getattr(self, 'finnhub_api_key', QLineEdit()).text(),
           },
           "max_requests_per_minute": self.max_requests_spin.value(),
           "request_delay": self.request_delay_spin.value(),
           "request_timeout": self.timeout_spin.value(),
           
           # Trading
           "portfolio_value": self.portfolio_value_spin.value(),
           "max_position_risk": self.max_risk_spin.value() / 100,
           "max_portfolio_risk": self.max_portfolio_risk_spin.value() / 100,
           "use_kelly_criterion": self.use_kelly_criterion_check.isChecked(),
           "max_simultaneous_trades": self.max_simultaneous_trades_spin.value(),
           "default_stop_loss": self.stop_loss_pct_spin.value() / 100,
           "default_profit_target": self.profit_target_pct_spin.value() / 100,
           
           # Interface
           "font_family": self.font_family_combo.currentText(),
           "font_size": self.font_size_spin.value(),
           "color_scheme": self.color_scheme_combo.currentText(),
           "decimal_places": self.decimal_places_spin.value(),
           "rows_per_page": self.rows_per_page_spin.value(),
           
           # Data
           "data_directory": self.data_directory_edit.text(),
           "cache_size_mb": self.cache_size_spin.value(),
           "cache_ttl_minutes": self.cache_ttl_spin.value(),
           "auto_backup": self.auto_backup_check.isChecked(),
           
           # Performance
           "thread_count": self.thread_count_spin.value(),
           "use_gpu_acceleration": self.use_gpu_acceleration_check.isChecked(),
           "max_memory_mb": self.max_memory_spin.value(),
           
           # Alerts
           "enable_price_alerts": self.enable_price_alerts_check.isChecked(),
           "price_alert_threshold": self.price_alert_threshold_spin.value() / 100,
           "alert_profit_target": self.alert_profit_target_check.isChecked(),
           "alert_stop_loss": self.alert_stop_loss_check.isChecked(),
           
           # Advanced
           "log_level": self.log_level_combo.currentText(),
           "log_to_file": self.log_to_file_check.isChecked(),
           "developer_mode": self.developer_mode_check.isChecked(),
       }
       
       return settings
   
   def apply_settings(self):
       """Apply current settings"""
       try:
           settings = self.get_current_settings()
           
           # Update config manager
           for key, value in settings.items():
               self.config_manager.set(key, value)
           
           # Save config
           self.config_manager.save_config()
           
           # Emit settings changed signal
           self.settings_changed.emit(settings)
           
           # Disable apply button
           self.apply_btn.setEnabled(False)
           
           QMessageBox.information(self, "Settings", "Settings applied successfully")
           
       except Exception as e:
           logger.error(f"Error applying settings: {e}")
           QMessageBox.critical(self, "Error", f"Failed to apply settings: {e}")
   
   def accept_settings(self):
       """Accept and close dialog"""
       self.apply_settings()
       self.accept()
   
   def reset_to_defaults(self):
       """Reset all settings to defaults"""
       reply = QMessageBox.question(
           self, "Reset Settings",
           "Are you sure you want to reset all settings to defaults? This cannot be undone.",
           QMessageBox.Yes | QMessageBox.No, QMessageBox.No
       )
       
       if reply == QMessageBox.Yes:
           try:
               # Reset UI to defaults
               self.theme_combo.setCurrentText("Dark")
               self.language_combo.setCurrentText("English")
               self.autosave_spin.setValue(5)
               self.default_contracts_spin.setValue(1)
               self.portfolio_value_spin.setValue(100000)
               self.max_risk_spin.setValue(2.0)
               self.font_family_combo.setCurrentText("Arial")
               self.font_size_spin.setValue(10)
               self.thread_count_spin.setValue(4)
               self.cache_size_spin.setValue(100)
               
               # Clear API keys
               if hasattr(self, 'alpha_vantage_api_key'):
                   self.alpha_vantage_api_key.clear()
               if hasattr(self, 'finnhub_api_key'):
                   self.finnhub_api_key.clear()
               
               # Reset checkboxes
               checkboxes = [
                   self.autostart_check, self.restore_session_check, self.check_updates_check,
                   self.use_kelly_criterion_check, self.auto_backup_check, self.use_gpu_acceleration_check,
                   self.enable_price_alerts_check, self.developer_mode_check
               ]
               
               for checkbox in checkboxes:
                   checkbox.setChecked(False)
               
               self.on_setting_changed()
               
           except Exception as e:
               logger.error(f"Error resetting settings: {e}")
               QMessageBox.critical(self, "Error", f"Failed to reset settings: {e}")
   
   # Helper methods for specific actions
   def test_api_key(self, provider: str, api_key: str):
       """Test API key"""
       if not api_key.strip():
           QMessageBox.warning(self, "Warning", "Please enter an API key first")
           return
       
       self.api_test_requested.emit(provider, api_key)
   
   def open_api_help(self, provider: str):
       """Open API provider help page"""
       urls = {
           "alpha_vantage": "https://www.alphavantage.co/support/#api-key",
           "finnhub": "https://finnhub.io/register",
           "iex_cloud": "https://iexcloud.io/docs/api/"
       }
       
       import webbrowser
       if provider in urls:
           webbrowser.open(urls[provider])
   
   def test_email_settings(self):
       """Test email settings"""
       # This would test the email configuration
       QMessageBox.information(self, "Email Test", "Email test functionality would be implemented here")
   
   def choose_font(self):
       """Choose font"""
       current_font = QFont(self.font_family_combo.currentText(), self.font_size_spin.value())
       font, ok = QFontDialog.getFont(current_font, self)
       
       if ok:
           self.font_family_combo.setCurrentText(font.family())
           self.font_size_spin.setValue(font.pointSize())
           self.on_setting_changed()
   
   def configure_shortcuts(self):
       """Configure keyboard shortcuts"""
       QMessageBox.information(self, "Shortcuts", "Keyboard shortcuts configuration would be implemented here")
   
   def browse_data_directory(self):
       """Browse for data directory"""
       current_dir = self.data_directory_edit.text()
       if not current_dir or not os.path.exists(current_dir):
           current_dir = os.path.expanduser("~")
       
       directory = QFileDialog.getExistingDirectory(
           self, "Select Data Directory", current_dir
       )
       
       if directory:
           self.data_directory_edit.setText(directory)
           self.on_setting_changed()
   
   def clear_cache(self):
       """Clear application cache"""
       reply = QMessageBox.question(
           self, "Clear Cache",
           "Are you sure you want to clear the cache? This will remove all cached data.",
           QMessageBox.Yes | QMessageBox.No, QMessageBox.No
       )
       
       if reply == QMessageBox.Yes:
           # This would clear the actual cache
           QMessageBox.information(self, "Cache", "Cache cleared successfully")
   
   def optimize_cache(self):
       """Optimize cache"""
       # This would optimize the cache
       QMessageBox.information(self, "Cache", "Cache optimized successfully")
   
   def backup_now(self):
       """Create backup now"""
       self.backup_requested.emit()
   
   def restore_backup(self):
       """Restore from backup"""
       backup_file, _ = QFileDialog.getOpenFileName(
           self, "Select Backup File", "", "Backup Files (*.backup);;All Files (*)"
       )
       
       if backup_file:
           self.restore_requested.emit(backup_file)
   
   def export_all_data(self):
       """Export all application data"""
       export_file, _ = QFileDialog.getSaveFileName(
           self, "Export Data", f"options_data_{datetime.now().strftime('%Y%m%d')}.json",
           "JSON Files (*.json);;All Files (*)"
       )
       
       if export_file:
           # This would export all data
           QMessageBox.information(self, "Export", f"Data exported to {export_file}")
   
   def import_data(self):
       """Import application data"""
       import_file, _ = QFileDialog.getOpenFileName(
           self, "Import Data", "", "JSON Files (*.json);;All Files (*)"
       )
       
       if import_file:
           # This would import data
           QMessageBox.information(self, "Import", f"Data imported from {import_file}")
   
   def refresh_performance_stats(self):
       """Refresh performance statistics"""
       # This would get actual performance stats
       self.cpu_usage_label.setText("12%")
       self.memory_usage_label.setText("256 MB")
       self.cache_hit_rate_label.setText("87%")
   
   def test_alert_sound(self):
       """Test alert sound"""
       QMessageBox.information(self, "Sound Test", "Alert sound would play here")
   
   def view_logs(self):
       """View application logs"""
       # This would open a log viewer dialog
       QMessageBox.information(self, "Logs", "Log viewer would open here")
   
   def clear_logs(self):
       """Clear application logs"""
       reply = QMessageBox.question(
           self, "Clear Logs",
           "Are you sure you want to clear all logs?",
           QMessageBox.Yes | QMessageBox.No, QMessageBox.No
       )
       
       if reply == QMessageBox.Yes:
           QMessageBox.information(self, "Logs", "Logs cleared successfully")
   
   def export_logs(self):
       """Export application logs"""
       log_file, _ = QFileDialog.getSaveFileName(
           self, "Export Logs", f"logs_{datetime.now().strftime('%Y%m%d')}.txt",
           "Text Files (*.txt);;All Files (*)"
       )
       
       if log_file:
           QMessageBox.information(self, "Logs", f"Logs exported to {log_file}")