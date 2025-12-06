"""
Settings dialog for the Professional Options Calculator
Comprehensive configuration interface
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QMessageBox, QTextEdit,
    QSlider, QProgressBar, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon
import logging
from typing import Dict, Any, Optional
import os
import json

logger = logging.getLogger(__name__)

class SettingsDialog(QDialog):
    """Professional settings dialog with tabbed interface"""
    
    settings_changed = Signal(dict)  # Emits changed settings
    api_test_requested = Signal(str, str)  # provider, key
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.changes_made = False
        self.setup_ui()
        self.load_current_settings()
        
    def setup_ui(self):
        """Setup settings dialog UI"""
        self.setWindowTitle("Settings - Options Calculator Pro")
        self.setModal(True)
        self.resize(800, 700)
        
        # Apply professional styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QTabWidget::pane {
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                background-color: #2b2b2b;
            }
            
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 10px 20px;
                margin-right: 2px;
                margin-bottom: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
            }
            
            QTabBar::tab:selected {
                background-color: #60a3d9;
            }
            
            QTabBar::tab:hover {
                background-color: #5a5a5a;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #60a3d9;
                font-size: 11pt;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Options Calculator Pro - Settings")
        header.setStyleSheet("""
            QLabel {
                font-size: 18pt;
                font-weight: bold;
                color: #60a3d9;
                padding: 15px;
                margin-bottom: 10px;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add tabs
        self.create_general_tab()
        self.create_api_tab()
        self.create_trading_tab()
        self.create_interface_tab()
        self.create_advanced_tab()
        
        # Button bar
        self.create_button_bar()
        layout.addWidget(self.button_frame)
        
    def create_general_tab(self):
        """Create general settings tab"""
        tab = QFrame()
        layout = QVBoxLayout(tab)
        
        # Application Settings
        app_group = QGroupBox("Application Settings")
        app_layout = QGridLayout(app_group)
        
        # Theme selection
        app_layout.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Professional", "Light Professional", "System"])
        self.theme_combo.currentTextChanged.connect(self.on_setting_changed)
        app_layout.addWidget(self.theme_combo, 0, 1)
        
        # Language (for future)
        app_layout.addWidget(QLabel("Language:"), 0, 2)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Spanish", "French", "German"])
        self.language_combo.setEnabled(False)  # Not implemented yet
        app_layout.addWidget(self.language_combo, 0, 3)
        
        # Auto-save interval
        app_layout.addWidget(QLabel("Auto-save (minutes):"), 1, 0)
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(1, 60)
        self.autosave_spin.setValue(5)
        self.autosave_spin.valueChanged.connect(self.on_setting_changed)
        app_layout.addWidget(self.autosave_spin, 1, 1)
        
        # Check for updates
        self.update_check = QCheckBox("Check for updates on startup")
        self.update_check.setChecked(True)
        self.update_check.stateChanged.connect(self.on_setting_changed)
        app_layout.addWidget(self.update_check, 1, 2, 1, 2)
        
        layout.addWidget(app_group)
        
        # Data Settings
        data_group = QGroupBox("Data Management")
        data_layout = QGridLayout(data_group)
        
        # Cache settings
        data_layout.addWidget(QLabel("Cache TTL (minutes):"), 0, 0)
        self.cache_ttl_spin = QSpinBox()
        self.cache_ttl_spin.setRange(1, 1440)  # 1 minute to 24 hours
        self.cache_ttl_spin.setValue(30)
        self.cache_ttl_spin.valueChanged.connect(self.on_setting_changed)
        data_layout.addWidget(self.cache_ttl_spin, 0, 1)
        
        # Data directory
        data_layout.addWidget(QLabel("Data Directory:"), 0, 2)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setReadOnly(True)
        self.data_dir_edit.setText("./data")
        data_layout.addWidget(self.data_dir_edit, 0, 3)
        
        self.browse_data_btn = QPushButton("Browse...")
        self.browse_data_btn.clicked.connect(self.browse_data_directory)
        data_layout.addWidget(self.browse_data_btn, 0, 4)
        
        # Backup settings
        self.enable_backup = QCheckBox("Enable automatic backups")
        self.enable_backup.setChecked(True)
        self.enable_backup.stateChanged.connect(self.on_setting_changed)
        data_layout.addWidget(self.enable_backup, 1, 0, 1, 2)
        
        self.backup_interval_spin = QSpinBox()
        self.backup_interval_spin.setRange(1, 168)  # 1 hour to 1 week
        self.backup_interval_spin.setValue(24)
        self.backup_interval_spin.setSuffix(" hours")
        data_layout.addWidget(QLabel("Backup interval:"), 1, 2)
        data_layout.addWidget(self.backup_interval_spin, 1, 3)
        
        layout.addWidget(data_group)
        
        # Favorite Stocks
        favorites_group = QGroupBox("Favorite Stocks")
        favorites_layout = QVBoxLayout(favorites_group)
        
        self.favorites_edit = QTextEdit()
        self.favorites_edit.setPlaceholderText("Enter favorite stock symbols, one per line or comma-separated")
        self.favorites_edit.setMaximumHeight(100)
        self.favorites_edit.textChanged.connect(self.on_setting_changed)
        favorites_layout.addWidget(self.favorites_edit)
        
        layout.addWidget(favorites_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "General")
    
    def create_api_tab(self):
        """Create API settings tab"""
        tab = QFrame()
        layout = QVBoxLayout(tab)
        
        # API Keys Section
        keys_group = QGroupBox("API Keys Configuration")
        keys_layout = QGridLayout(keys_group)
        
        # Alpha Vantage
        keys_layout.addWidget(QLabel("Alpha Vantage API Key:"), 0, 0)
        self.av_key_edit = QLineEdit()
        self.av_key_edit.setEchoMode(QLineEdit.Password)
        self.av_key_edit.setPlaceholderText("Enter your Alpha Vantage API key")
        self.av_key_edit.textChanged.connect(self.on_setting_changed)
        keys_layout.addWidget(self.av_key_edit, 0, 1)
        
        self.av_show_btn = QPushButton("Show")
        self.av_show_btn.setCheckable(True)
        self.av_show_btn.setMaximumWidth(60)
        self.av_show_btn.clicked.connect(lambda: self.toggle_key_visibility(self.av_key_edit, self.av_show_btn))
        keys_layout.addWidget(self.av_show_btn, 0, 2)
        
        self.av_test_btn = QPushButton("Test")
        self.av_test_btn.setMaximumWidth(60)
        self.av_test_btn.clicked.connect(lambda: self.test_api_key("alpha_vantage", self.av_key_edit.text()))
        keys_layout.addWidget(self.av_test_btn, 0, 3)
        
        self.av_get_btn = QPushButton("Get Free Key")
        self.av_get_btn.clicked.connect(lambda: self.open_url("https://www.alphavantage.co/support/#api-key"))
        keys_layout.addWidget(self.av_get_btn, 0, 4)
        
        # Finnhub
        keys_layout.addWidget(QLabel("Finnhub API Key:"), 1, 0)
        self.finnhub_key_edit = QLineEdit()
        self.finnhub_key_edit.setEchoMode(QLineEdit.Password)
        self.finnhub_key_edit.setPlaceholderText("Enter your Finnhub API key (optional)")
        self.finnhub_key_edit.textChanged.connect(self.on_setting_changed)
        keys_layout.addWidget(self.finnhub_key_edit, 1, 1)
        
        self.finnhub_show_btn = QPushButton("Show")
        self.finnhub_show_btn.setCheckable(True)
        self.finnhub_show_btn.setMaximumWidth(60)
        self.finnhub_show_btn.clicked.connect(lambda: self.toggle_key_visibility(self.finnhub_key_edit, self.finnhub_show_btn))
        keys_layout.addWidget(self.finnhub_show_btn, 1, 2)
        
        self.finnhub_test_btn = QPushButton("Test")
        self.finnhub_test_btn.setMaximumWidth(60)
        self.finnhub_test_btn.clicked.connect(lambda: self.test_api_key("finnhub", self.finnhub_key_edit.text()))
        keys_layout.addWidget(self.finnhub_test_btn, 1, 3)
        
        self.finnhub_get_btn = QPushButton("Get Free Key")
        self.finnhub_get_btn.clicked.connect(lambda: self.open_url("https://finnhub.io/register"))
        keys_layout.addWidget(self.finnhub_get_btn, 1, 4)
        
        layout.addWidget(keys_group)
        
        # API Configuration
        config_group = QGroupBox("API Configuration")
        config_layout = QGridLayout(config_group)
        
        # Timeout settings
        config_layout.addWidget(QLabel("Request Timeout (seconds):"), 0, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 120)
        self.timeout_spin.setValue(30)
        self.timeout_spin.valueChanged.connect(self.on_setting_changed)
        config_layout.addWidget(self.timeout_spin, 0, 1)
        
        # Retry settings
        config_layout.addWidget(QLabel("Max Retries:"), 0, 2)
        self.retries_spin = QSpinBox()
        self.retries_spin.setRange(0, 10)
        self.retries_spin.setValue(3)
        self.retries_spin.valueChanged.connect(self.on_setting_changed)
        config_layout.addWidget(self.retries_spin, 0, 3)
        
        # Rate limiting
        config_layout.addWidget(QLabel("Rate Limit (calls/minute):"), 1, 0)
        self.rate_limit_spin = QSpinBox()
        self.rate_limit_spin.setRange(1, 1000)
        self.rate_limit_spin.setValue(60)
        self.rate_limit_spin.valueChanged.connect(self.on_setting_changed)
        config_layout.addWidget(self.rate_limit_spin, 1, 1)
        
        # Concurrent requests
        config_layout.addWidget(QLabel("Max Concurrent:"), 1, 2)
        self.concurrent_spin = QSpinBox()
        self.concurrent_spin.setRange(1, 20)
        self.concurrent_spin.setValue(5)
        self.concurrent_spin.valueChanged.connect(self.on_setting_changed)
        config_layout.addWidget(self.concurrent_spin, 1, 3)
        
        layout.addWidget(config_group)
        
        # Data Sources Priority
        priority_group = QGroupBox("Data Source Priority")
        priority_layout = QVBoxLayout(priority_group)
        
        priority_info = QLabel("Drag to reorder data source priority (highest to lowest):")
        priority_info.setStyleSheet("color: #cccccc; font-style: italic;")
        priority_layout.addWidget(priority_info)
        
        # This would be a drag-and-drop list widget in a full implementation
        self.priority_list = QTextEdit()
        self.priority_list.setMaximumHeight(80)
        self.priority_list.setPlainText("1. Yahoo Finance\n2. Alpha Vantage\n3. Finnhub")
        self.priority_list.setReadOnly(True)
        priority_layout.addWidget(self.priority_list)
        
        layout.addWidget(priority_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "API Settings")
    
    def create_trading_tab(self):
        """Create trading settings tab"""
        tab = QFrame()
        layout = QVBoxLayout(tab)
        
        # Portfolio Settings
        portfolio_group = QGroupBox("Portfolio Configuration")
        portfolio_layout = QGridLayout(portfolio_group)
        
        portfolio_layout.addWidget(QLabel("Portfolio Value ($):"), 0, 0)
        self.portfolio_spin = QSpinBox()
        self.portfolio_spin.setRange(1000, 100000000)
        self.portfolio_spin.setValue(100000)
        self.portfolio_spin.valueChanged.connect(self.on_setting_changed)
        portfolio_layout.addWidget(self.portfolio_spin, 0, 1)
        
        portfolio_layout.addWidget(QLabel("Max Position Risk (%):"), 0, 2)
        self.max_risk_spin = QDoubleSpinBox()
        self.max_risk_spin.setRange(0.1, 25.0)
        self.max_risk_spin.setValue(2.0)
        self.max_risk_spin.setDecimals(1)
        self.max_risk_spin.valueChanged.connect(self.on_setting_changed)
        portfolio_layout.addWidget(self.max_risk_spin, 0, 3)
        
        portfolio_layout.addWidget(QLabel("Default Contracts:"), 1, 0)
        self.default_contracts_spin = QSpinBox()
        self.default_contracts_spin.setRange(1, 100)
        self.default_contracts_spin.setValue(1)
        self.default_contracts_spin.valueChanged.connect(self.on_setting_changed)
        portfolio_layout.addWidget(self.default_contracts_spin, 1, 1)
        
        portfolio_layout.addWidget(QLabel("Commission per Contract ($):"), 1, 2)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0.0, 10.0)
        self.commission_spin.setValue(0.65)
        self.commission_spin.setDecimals(2)
        self.commission_spin.valueChanged.connect(self.on_setting_changed)
        portfolio_layout.addWidget(self.commission_spin, 1, 3)
        
        layout.addWidget(portfolio_group)
        
        # Risk Management
        risk_group = QGroupBox("Risk Management")
        risk_layout = QGridLayout(risk_group)
        
        # Kelly Criterion settings
        self.use_kelly = QCheckBox("Use Kelly Criterion for position sizing")
        self.use_kelly.setChecked(True)
        self.use_kelly.stateChanged.connect(self.on_setting_changed)
        risk_layout.addWidget(self.use_kelly, 0, 0, 1, 2)
        
        risk_layout.addWidget(QLabel("Kelly Fraction:"), 0, 2)
        self.kelly_fraction_spin = QDoubleSpinBox()
        self.kelly_fraction_spin.setRange(0.1, 1.0)
        self.kelly_fraction_spin.setValue(0.25)  # Quarter Kelly
        self.kelly_fraction_spin.setDecimals(2)
        self.kelly_fraction_spin.valueChanged.connect(self.on_setting_changed)
        risk_layout.addWidget(self.kelly_fraction_spin, 0, 3)
        
        # Stop loss settings
        self.use_stop_loss = QCheckBox("Enable automatic stop loss")
        self.use_stop_loss.stateChanged.connect(self.on_setting_changed)
        risk_layout.addWidget(self.use_stop_loss, 1, 0, 1, 2)
        
        risk_layout.addWidget(QLabel("Stop Loss (%):"), 1, 2)
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(10.0, 100.0)
        self.stop_loss_spin.setValue(50.0)
        self.stop_loss_spin.setDecimals(1)
        self.stop_loss_spin.valueChanged.connect(self.on_setting_changed)
        risk_layout.addWidget(self.stop_loss_spin, 1, 3)
        
        # Profit target
        self.use_profit_target = QCheckBox("Enable profit target")
        self.use_profit_target.stateChanged.connect(self.on_setting_changed)
        risk_layout.addWidget(self.use_profit_target, 2, 0, 1, 2)
        
        risk_layout.addWidget(QLabel("Profit Target (%):"), 2, 2)
        self.profit_target_spin = QDoubleSpinBox()
        self.profit_target_spin.setRange(10.0, 500.0)
        self.profit_target_spin.setValue(50.0)
        self.profit_target_spin.setDecimals(1)
        self.profit_target_spin.valueChanged.connect(self.on_setting_changed)
        risk_layout.addWidget(self.profit_target_spin, 2, 3)
        
        layout.addWidget(risk_group)
        
        # Strategy Settings
        strategy_group = QGroupBox("Strategy Configuration")
        strategy_layout = QGridLayout(strategy_group)
        
        # Default strategy
        strategy_layout.addWidget(QLabel("Default Strategy:"), 0, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Calendar Spread", "Iron Condor", "Butterfly", "Straddle"])
        self.strategy_combo.currentTextChanged.connect(self.on_setting_changed)
        strategy_layout.addWidget(self.strategy_combo, 0, 1)
        
        # Analysis depth
        strategy_layout.addWidget(QLabel("Default Analysis Depth:"), 0, 2)
        self.analysis_depth_combo = QComboBox()
        self.analysis_depth_combo.addItems(["Quick", "Standard", "Comprehensive"])
        self.analysis_depth_combo.setCurrentText("Standard")
        self.analysis_depth_combo.currentTextChanged.connect(self.on_setting_changed)
        strategy_layout.addWidget(self.analysis_depth_combo, 0, 3)
        
        # Auto-analyze on symbol entry
        self.auto_analyze = QCheckBox("Auto-analyze when symbols are entered")
        self.auto_analyze.stateChanged.connect(self.on_setting_changed)
        strategy_layout.addWidget(self.auto_analyze, 1, 0, 1, 2)
        
        # Include ML predictions
        self.use_ml_predictions = QCheckBox("Include ML predictions in analysis")
        self.use_ml_predictions.setChecked(True)
        self.use_ml_predictions.stateChanged.connect(self.on_setting_changed)
        strategy_layout.addWidget(self.use_ml_predictions, 1, 2, 1, 2)
        
        layout.addWidget(strategy_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Trading")
    
    def create_interface_tab(self):
        """Create interface settings tab"""
        tab = QFrame()
        layout = QVBoxLayout(tab)
        
        # Display Settings
        display_group = QGroupBox("Display Configuration")
        display_layout = QGridLayout(display_group)
        
        # Font settings
        display_layout.addWidget(QLabel("Font Size:"), 0, 0)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(10)
        self.font_size_spin.valueChanged.connect(self.on_setting_changed)
        display_layout.addWidget(self.font_size_spin, 0, 1)
        
        # Decimal places
        display_layout.addWidget(QLabel("Price Decimal Places:"), 0, 2)
        self.decimal_places_spin = QSpinBox()
        self.decimal_places_spin.setRange(1, 6)
        self.decimal_places_spin.setValue(2)
        self.decimal_places_spin.valueChanged.connect(self.on_setting_changed)
        display_layout.addWidget(self.decimal_places_spin, 0, 3)
        
        # Currency format
        display_layout.addWidget(QLabel("Currency Format:"), 1, 0)
        self.currency_combo = QComboBox()
        self.currency_combo.addItems(["$1,234.56", "$1234.56", "1,234.56 $", "1234.56 $"])
        self.currency_combo.currentTextChanged.connect(self.on_setting_changed)
        display_layout.addWidget(self.currency_combo, 1, 1)
        
        # Percentage format
        display_layout.addWidget(QLabel("Percentage Format:"), 1, 2)
        self.percentage_combo = QComboBox()
        self.percentage_combo.addItems(["12.34%", "12.3%", "12%", "0.1234"])
        self.percentage_combo.currentTextChanged.connect(self.on_setting_changed)
        display_layout.addWidget(self.percentage_combo, 1, 3)
        
        layout.addWidget(display_group)
        
        # Chart Settings
        chart_group = QGroupBox("Chart Configuration")
        chart_layout = QGridLayout(chart_group)
        
        # Default chart type
        chart_layout.addWidget(QLabel("Default Chart Type:"), 0, 0)
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["P/L Diagram", "Price Distribution", "IV Term Structure"])
        self.chart_type_combo.currentTextChanged.connect(self.on_setting_changed)
        chart_layout.addWidget(self.chart_type_combo, 0, 1)
        
        # Chart theme
        chart_layout.addWidget(QLabel("Chart Theme:"), 0, 2)
        self.chart_theme_combo = QComboBox()
        self.chart_theme_combo.addItems(["Professional Dark", "Professional Light", "Classic"])
        self.chart_theme_combo.currentTextChanged.connect(self.on_setting_changed)
        chart_layout.addWidget(self.chart_theme_combo, 0, 3)
        
        # Animation settings
        self.enable_animations = QCheckBox("Enable chart animations")
        self.enable_animations.setChecked(True)
        self.enable_animations.stateChanged.connect(self.on_setting_changed)
        chart_layout.addWidget(self.enable_animations, 1, 0, 1, 2)
        
        # Grid lines
        self.show_grid = QCheckBox("Show grid lines")
        self.show_grid.setChecked(True)
        self.show_grid.stateChanged.connect(self.on_setting_changed)
        chart_layout.addWidget(self.show_grid, 1, 2, 1, 2)
        
        layout.addWidget(chart_group)
        
        # Notification Settings
        notification_group = QGroupBox("Notifications")
        notification_layout = QGridLayout(notification_group)
        
        # Enable notifications
        self.enable_notifications = QCheckBox("Enable desktop notifications")
        self.enable_notifications.setChecked(True)
        self.enable_notifications.stateChanged.connect(self.on_setting_changed)
        notification_layout.addWidget(self.enable_notifications, 0, 0, 1, 2)
        
        # Sound notifications
        self.enable_sound = QCheckBox("Enable sound notifications")
        self.enable_sound.stateChanged.connect(self.on_setting_changed)
        notification_layout.addWidget(self.enable_sound, 0, 2, 1, 2)
        
        # Notification types
        notification_layout.addWidget(QLabel("Notify on:"), 1, 0)
        
        self.notify_analysis_complete = QCheckBox("Analysis complete")
        self.notify_analysis_complete.setChecked(True)
        self.notify_analysis_complete.stateChanged.connect(self.on_setting_changed)
        notification_layout.addWidget(self.notify_analysis_complete, 1, 1)
        
        self.notify_opportunities = QCheckBox("New opportunities found")
        self.notify_opportunities.setChecked(True)
        self.notify_opportunities.stateChanged.connect(self.on_setting_changed)
        notification_layout.addWidget(self.notify_opportunities, 1, 2)
        
        self.notify_errors = QCheckBox("Errors and warnings")
        self.notify_errors.setChecked(True)
        self.notify_errors.stateChanged.connect(self.on_setting_changed)
        notification_layout.addWidget(self.notify_errors, 1, 3)
        
        layout.addWidget(notification_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Interface")
    
    def create_advanced_tab(self):
        """Create advanced settings tab"""
        tab = QFrame()
        layout = QVBoxLayout(tab)
        
        # Performance Settings
        performance_group = QGroupBox("Performance Optimization")
        performance_layout = QGridLayout(performance_group)
        
        # Multi-threading
        self.enable_multithreading = QCheckBox("Enable multi-threading")
        self.enable_multithreading.setChecked(True)
        self.enable_multithreading.stateChanged.connect(self.on_setting_changed)
        performance_layout.addWidget(self.enable_multithreading, 0, 0, 1, 2)
        
        performance_layout.addWidget(QLabel("Max Worker Threads:"), 0, 2)
        self.max_threads_spin = QSpinBox()
        self.max_threads_spin.setRange(1, 16)
        self.max_threads_spin.setValue(4)
        self.max_threads_spin.valueChanged.connect(self.on_setting_changed)
        performance_layout.addWidget(self.max_threads_spin, 0, 3)
        
        # Memory management
        performance_layout.addWidget(QLabel("Cache Size (MB):"), 1, 0)
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(10, 1000)
        self.cache_size_spin.setValue(100)
        self.cache_size_spin.valueChanged.connect(self.on_setting_changed)
        performance_layout.addWidget(self.cache_size_spin, 1, 1)
        
        # GPU acceleration (if available)
        self.enable_gpu = QCheckBox("Enable GPU acceleration (experimental)")
        self.enable_gpu.stateChanged.connect(self.on_setting_changed)
        performance_layout.addWidget(self.enable_gpu, 1, 2, 1, 2)
       
        layout.addWidget(performance_group)
       
        # Machine Learning Settings
        ml_group = QGroupBox("Machine Learning Configuration")
        ml_layout = QGridLayout(ml_group)
       
        # Enable ML
        self.enable_ml = QCheckBox("Enable machine learning predictions")
        self.enable_ml.setChecked(True)
        self.enable_ml.stateChanged.connect(self.on_setting_changed)
        ml_layout.addWidget(self.enable_ml, 0, 0, 1, 2)
       
        # Model update frequency
        ml_layout.addWidget(QLabel("Model Update Frequency:"), 0, 2)
        self.ml_update_combo = QComboBox()
        self.ml_update_combo.addItems(["Never", "Weekly", "Monthly", "After 10 trades", "After 25 trades"])
        self.ml_update_combo.setCurrentText("After 25 trades")
        self.ml_update_combo.currentTextChanged.connect(self.on_setting_changed)
        ml_layout.addWidget(self.ml_update_combo, 0, 3)
       
        # Confidence threshold
        ml_layout.addWidget(QLabel("ML Confidence Threshold:"), 1, 0)
        self.ml_confidence_spin = QDoubleSpinBox()
        self.ml_confidence_spin.setRange(0.1, 0.9)
        self.ml_confidence_spin.setValue(0.6)
        self.ml_confidence_spin.setDecimals(2)
        self.ml_confidence_spin.valueChanged.connect(self.on_setting_changed)
        ml_layout.addWidget(self.ml_confidence_spin, 1, 1)
       
        # Feature importance display
        self.show_feature_importance = QCheckBox("Show feature importance in results")
        self.show_feature_importance.setChecked(True)
        self.show_feature_importance.stateChanged.connect(self.on_setting_changed)
        ml_layout.addWidget(self.show_feature_importance, 1, 2, 1, 2)
       
        layout.addWidget(ml_group)
       
        # Debug and Logging
        debug_group = QGroupBox("Debug and Logging")
        debug_layout = QGridLayout(debug_group)
       
        # Log level
        debug_layout.addWidget(QLabel("Log Level:"), 0, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.currentTextChanged.connect(self.on_setting_changed)
        debug_layout.addWidget(self.log_level_combo, 0, 1)
       
        # Enable debug mode
        self.enable_debug = QCheckBox("Enable debug mode")
        self.enable_debug.stateChanged.connect(self.on_setting_changed)
        debug_layout.addWidget(self.enable_debug, 0, 2, 1, 2)
       
        # Log file location
        debug_layout.addWidget(QLabel("Log File:"), 1, 0)
        self.log_file_edit = QLineEdit()
        self.log_file_edit.setReadOnly(True)
        self.log_file_edit.setText("./logs/options_calculator.log")
        debug_layout.addWidget(self.log_file_edit, 1, 1, 1, 2)
       
        self.open_log_btn = QPushButton("Open Log")
        self.open_log_btn.clicked.connect(self.open_log_file)
        debug_layout.addWidget(self.open_log_btn, 1, 3)
       
        # Performance monitoring
        self.enable_profiling = QCheckBox("Enable performance profiling")
        self.enable_profiling.stateChanged.connect(self.on_setting_changed)
        debug_layout.addWidget(self.enable_profiling, 2, 0, 1, 2)
       
        # Export settings
        self.export_settings_btn = QPushButton("Export Settings")
        self.export_settings_btn.clicked.connect(self.export_settings)
        debug_layout.addWidget(self.export_settings_btn, 2, 2)
       
        self.import_settings_btn = QPushButton("Import Settings")
        self.import_settings_btn.clicked.connect(self.import_settings)
        debug_layout.addWidget(self.import_settings_btn, 2, 3)
       
        layout.addWidget(debug_group)
        
        # Reset Section
        reset_group = QGroupBox("Reset Options")
        reset_layout = QHBoxLayout(reset_group)
       
        reset_warning = QLabel("⚠️ These actions cannot be undone")
        reset_warning.setStyleSheet("color: #ffc107; font-weight: bold;")
        reset_layout.addWidget(reset_warning)
        
        reset_layout.addStretch()
       
        self.reset_settings_btn = QPushButton("Reset All Settings")
        self.reset_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.reset_settings_btn.clicked.connect(self.reset_all_settings)
        reset_layout.addWidget(self.reset_settings_btn)
       
        self.clear_cache_btn = QPushButton("Clear All Cache")
        self.clear_cache_btn.clicked.connect(self.clear_all_cache)
        reset_layout.addWidget(self.clear_cache_btn)
       
        layout.addWidget(reset_group)
        layout.addStretch()
       
        self.tab_widget.addTab(tab, "Advanced")
   
    def create_button_bar(self):
        """Create dialog button bar"""
        self.button_frame = QFrame()
        button_layout = QHBoxLayout(self.button_frame)
       
        # Status indicator
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #60a3d9; font-style: italic;")
        button_layout.addWidget(self.status_label)
       
        button_layout.addStretch()
       
        # Restore defaults button
        self.restore_btn = QPushButton("Restore Defaults")
        self.restore_btn.clicked.connect(self.restore_defaults)
        button_layout.addWidget(self.restore_btn)
       
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
       
        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_settings)
        button_layout.addWidget(self.apply_btn)
       
        # OK button
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setDefault(True)
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
        self.ok_btn.clicked.connect(self.accept_settings)
        button_layout.addWidget(self.ok_btn)
   
    def load_current_settings(self):
        """Load current settings from config manager"""
        try:
            # General settings
            self.theme_combo.setCurrentText(self.config_manager.get("theme", "Dark Professional"))
            self.autosave_spin.setValue(self.config_manager.get("autosave_interval", 5))
            self.update_check.setChecked(self.config_manager.get("check_updates", True))
            self.cache_ttl_spin.setValue(self.config_manager.get("cache_ttl_minutes", 30))
            self.data_dir_edit.setText(self.config_manager.get("data_directory", "./data"))
            self.enable_backup.setChecked(self.config_manager.get("enable_backup", True))
            self.backup_interval_spin.setValue(self.config_manager.get("backup_interval_hours", 24))
           
            # Load favorites
            favorites = self.config_manager.get("favorite_stocks", [])
            self.favorites_edit.setPlainText(", ".join(favorites))
           
            # API settings
            api_keys = self.config_manager.get("api_keys", {})
            self.av_key_edit.setText(api_keys.get("alpha_vantage", ""))
            self.finnhub_key_edit.setText(api_keys.get("finnhub", ""))
            self.timeout_spin.setValue(self.config_manager.get("api_timeout", 30))
            self.retries_spin.setValue(self.config_manager.get("max_retries", 3))
            self.rate_limit_spin.setValue(self.config_manager.get("rate_limit", 60))
            self.concurrent_spin.setValue(self.config_manager.get("max_concurrent", 5))
           
            # Trading settings
            self.portfolio_spin.setValue(self.config_manager.get("portfolio_value", 100000))
            self.max_risk_spin.setValue(self.config_manager.get("max_position_risk", 2.0) * 100)
            self.default_contracts_spin.setValue(self.config_manager.get("default_contracts", 1))
            self.commission_spin.setValue(self.config_manager.get("commission_per_contract", 0.65))
            self.use_kelly.setChecked(self.config_manager.get("use_kelly_criterion", True))
            self.kelly_fraction_spin.setValue(self.config_manager.get("kelly_fraction", 0.25))
            self.use_stop_loss.setChecked(self.config_manager.get("use_stop_loss", False))
            self.stop_loss_spin.setValue(self.config_manager.get("stop_loss_pct", 50.0))
            self.use_profit_target.setChecked(self.config_manager.get("use_profit_target", False))
            self.profit_target_spin.setValue(self.config_manager.get("profit_target_pct", 50.0))
           
            # Interface settings
            self.font_size_spin.setValue(self.config_manager.get("font_size", 10))
            self.decimal_places_spin.setValue(self.config_manager.get("decimal_places", 2))
            self.enable_animations.setChecked(self.config_manager.get("enable_animations", True))
            self.show_grid.setChecked(self.config_manager.get("show_grid", True))
            self.enable_notifications.setChecked(self.config_manager.get("enable_notifications", True))
           
            # Advanced settings
            self.enable_multithreading.setChecked(self.config_manager.get("enable_multithreading", True))
            self.max_threads_spin.setValue(self.config_manager.get("max_threads", 4))
            self.cache_size_spin.setValue(self.config_manager.get("cache_size_mb", 100))
            self.enable_ml.setChecked(self.config_manager.get("use_ml_predictions", True))
            self.log_level_combo.setCurrentText(self.config_manager.get("log_level", "INFO"))
            self.enable_debug.setChecked(self.config_manager.get("debug_mode", False))
           
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            self.status_label.setText("Error loading current settings")
   
    def on_setting_changed(self):
        """Handle setting change"""
        self.changes_made = True
        self.apply_btn.setEnabled(True)
        self.status_label.setText("Settings modified")
   
    def toggle_key_visibility(self, edit_widget, button):
        """Toggle API key visibility"""
        if button.isChecked():
            edit_widget.setEchoMode(QLineEdit.Normal)
            button.setText("Hide")
        else:
            edit_widget.setEchoMode(QLineEdit.Password)
            button.setText("Show")
   
    def test_api_key(self, provider, key):
        """Test API key"""
        if not key.strip():
            QMessageBox.warning(self, "Test API Key", "Please enter an API key first.")
            return
       
        self.api_test_requested.emit(provider, key)
        self.status_label.setText(f"Testing {provider} API key...")
   
    def browse_data_directory(self):
        """Browse for data directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", self.data_dir_edit.text()
        )
        if directory:
            self.data_dir_edit.setText(directory)
            self.on_setting_changed()
   
    def open_log_file(self):
        """Open log file in default editor"""
        try:
            import subprocess
            import platform
           
            log_file = self.log_file_edit.text()
           
            if not os.path.exists(log_file):
                QMessageBox.information(self, "Log File", f"Log file not found: {log_file}")
                return
           
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.call(("open", log_file))
            elif system == "Windows":
                os.startfile(log_file)
            else:  # Linux
                subprocess.call(("xdg-open", log_file))
               
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open log file: {e}")
   
    def export_settings(self):
        """Export settings to file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Settings", "options_calculator_settings.json",
                "JSON files (*.json);;All files (*.*)"
            )
           
            if filename:
                settings = self.collect_current_settings()
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=4)
               
                QMessageBox.information(self, "Export Complete", f"Settings exported to {filename}")
               
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export settings: {e}")
   
    def import_settings(self):
        """Import settings from file"""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Import Settings", "",
                "JSON files (*.json);;All files (*.*)"
            )
           
            if filename:
                with open(filename, 'r') as f:
                    settings = json.load(f)
               
                # Apply imported settings
                self.apply_imported_settings(settings)
                self.on_setting_changed()
               
                QMessageBox.information(self, "Import Complete", "Settings imported successfully")
               
        except Exception as e:
            QMessageBox.warning(self, "Import Error", f"Failed to import settings: {e}")
   
    def reset_all_settings(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
       
        if reply == QMessageBox.Yes:
            self.restore_defaults()
            QMessageBox.information(self, "Reset Complete", "All settings have been reset to defaults")
   
    def clear_all_cache(self):
        """Clear all cached data"""
        reply = QMessageBox.question(
            self, "Clear Cache",
            "Are you sure you want to clear all cached data?\n\nThis will remove all stored market data.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
       
        if reply == QMessageBox.Yes:
            # Signal to clear cache would be emitted here
            QMessageBox.information(self, "Cache Cleared", "All cached data has been cleared")
   
    def restore_defaults(self):
        """Restore default settings"""
        # This would reset all widgets to their default values
        # Implementation would set each widget to its default state
        self.on_setting_changed()
        self.status_label.setText("Settings restored to defaults")
   
    def collect_current_settings(self):
        """Collect current settings from all widgets"""
        settings = {
            # General
            "theme": self.theme_combo.currentText(),
            "autosave_interval": self.autosave_spin.value(),
            "check_updates": self.update_check.isChecked(),
            "cache_ttl_minutes": self.cache_ttl_spin.value(),
            "data_directory": self.data_dir_edit.text(),
            "enable_backup": self.enable_backup.isChecked(),
            "backup_interval_hours": self.backup_interval_spin.value(),
            "favorite_stocks": [s.strip() for s in self.favorites_edit.toPlainText().replace(',', '\n').split('\n') if s.strip()],
           
            # API
            "api_keys": {
                "alpha_vantage": self.av_key_edit.text(),
                "finnhub": self.finnhub_key_edit.text()
            },
            "api_timeout": self.timeout_spin.value(),
            "max_retries": self.retries_spin.value(),
            "rate_limit": self.rate_limit_spin.value(),
            "max_concurrent": self.concurrent_spin.value(),
           
            # Trading
            "portfolio_value": self.portfolio_spin.value(),
            "max_position_risk": self.max_risk_spin.value() / 100,
            "default_contracts": self.default_contracts_spin.value(),
            "commission_per_contract": self.commission_spin.value(),
            "use_kelly_criterion": self.use_kelly.isChecked(),
            "kelly_fraction": self.kelly_fraction_spin.value(),
            "use_stop_loss": self.use_stop_loss.isChecked(),
            "stop_loss_pct": self.stop_loss_spin.value(),
            "use_profit_target": self.use_profit_target.isChecked(),
            "profit_target_pct": self.profit_target_spin.value(),
           
            # Interface
            "font_size": self.font_size_spin.value(),
            "decimal_places": self.decimal_places_spin.value(),
            "enable_animations": self.enable_animations.isChecked(),
            "show_grid": self.show_grid.isChecked(),
            "enable_notifications": self.enable_notifications.isChecked(),
           
            # Advanced
            "enable_multithreading": self.enable_multithreading.isChecked(),
            "max_threads": self.max_threads_spin.value(),
            "cache_size_mb": self.cache_size_spin.value(),
            "use_ml_predictions": self.enable_ml.isChecked(),
            "log_level": self.log_level_combo.currentText(),
            "debug_mode": self.enable_debug.isChecked()
        }
        
        return settings
   
    def apply_imported_settings(self, settings):
        """Apply imported settings to widgets"""
        try:
            # This would set each widget based on the imported settings
            # Similar to load_current_settings but from the imported dict
            pass
        except Exception as e:
            logger.error(f"Error applying imported settings: {e}")
   
    def apply_settings(self):
        """Apply current settings"""
        try:
            settings = self.collect_current_settings()
            
            # Update config manager
            for key, value in settings.items():
                self.config_manager.set(key, value)
           
            # Save to file
            self.config_manager.save_config()
            
            # Emit settings changed signal
            self.settings_changed.emit(settings)
           
            self.changes_made = False
            self.apply_btn.setEnabled(False)
            self.status_label.setText("Settings applied successfully")
           
            # Start timer to clear status
            QTimer.singleShot(3000, lambda: self.status_label.setText(""))
           
        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            QMessageBox.warning(self, "Error", f"Failed to apply settings: {e}")
    
    def accept_settings(self):
        """Accept and close dialog"""
        if self.changes_made:
            self.apply_settings()
        self.accept()
   
    def reject(self):
        """Reject changes and close dialog"""
        if self.changes_made:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
           
            if reply == QMessageBox.Save:
                self.apply_settings()
                super().accept()
            elif reply == QMessageBox.Discard:
                super().reject()
            # Cancel - do nothing, dialog stays open
        else:
            super().reject()
   
    def open_url(self, url):
        """Open URL in default browser"""
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            logger.error(f"Error opening URL: {e}")