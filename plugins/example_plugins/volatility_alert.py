"""
Volatility Alert Plugin - Example Alert Plugin
Monitors volatility changes and sends alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from PySide6.QtCore import QTimer
from ..base_plugin import UIPlugin, PluginMetadata

def get_plugin_metadata() -> PluginMetadata:
    """Return plugin metadata"""
    return PluginMetadata(
        name="Volatility Alert System",
        version="1.0.0",
        description="Real-time volatility monitoring and alert system",
        author="Options Calculator Pro",
        category="alert"
    )

class VolatilityAlertPlugin(UIPlugin):
    """Volatility monitoring and alert plugin"""
    
    def initialize(self, app_context: Dict[str, Any]) -> bool:
        """Initialize volatility alert plugin"""
        try:
            self.market_data_service = app_context.get('market_data_service')
            self.watchlist = []
            self.alert_conditions = []
            self.monitoring_active = False
            
            # Configuration
            self.config = {
                'check_interval_seconds': 60,  # Check every minute
                'iv_change_threshold': 10,  # 10% IV change
                'volume_spike_threshold': 200,  # 200% of average volume
                'price_move_threshold': 5,  # 5% price move
                'enable_notifications': True,
                'enable_sound': False
            }
            
            # Setup monitoring timer
            self.monitor_timer = QTimer()
            self.monitor_timer.timeout.connect(self.check_alerts)
            
            return True
            
        except Exception as e:
            logger.error(f"Volatility alert plugin initialization failed: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup volatility alert plugin"""
        try:
            if self.monitor_timer:
                self.monitor_timer.stop()
            return True
        except:
            return False
    
    def get_widget(self):
        """Return main widget for volatility alerts"""
        from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                      QListWidget, QPushButton, QLabel,
                                      QLineEdit, QSpinBox, QCheckBox, QGroupBox)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header = QLabel("Volatility Alert System")
        header.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #60a3d9;
                padding: 10px;
            }
        """)
        layout.addWidget(header)
        
        # Monitoring controls
        controls_group = QGroupBox("Monitoring Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.start_button = QPushButton("Start Monitoring")
        self.start_button.clicked.connect(self.start_monitoring)
        controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Monitoring")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)
        
        # Status indicator
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setStyleSheet("color: #dc3545;")
        controls_layout.addWidget(self.status_label)
        
        layout.addWidget(controls_group)
        
        # Watchlist management
        watchlist_group = QGroupBox("Watchlist")
        watchlist_layout = QVBoxLayout(watchlist_group)
        
        # Add symbol controls
        add_layout = QHBoxLayout()
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol to monitor...")
        add_layout.addWidget(self.symbol_input)
        
        add_button = QPushButton("Add to Watchlist")
        add_button.clicked.connect(self.add_symbol)
        add_layout.addWidget(add_button)
        
        watchlist_layout.addLayout(add_layout)
        
        # Watchlist display
        self.watchlist_widget = QListWidget()
        watchlist_layout.addWidget(self.watchlist_widget)
        
        # Remove button
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_symbol)
        watchlist_layout.addWidget(remove_button)
        
        layout.addWidget(watchlist_group)
        
        # Alert settings
        settings_group = QGroupBox("Alert Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # IV change threshold
        iv_layout = QHBoxLayout()
        iv_layout.addWidget(QLabel("IV Change Threshold (%):"))
        self.iv_threshold_spin = QSpinBox()
        self.iv_threshold_spin.setRange(1, 100)
        self.iv_threshold_spin.setValue(self.config['iv_change_threshold'])
        self.iv_threshold_spin.valueChanged.connect(self.update_iv_threshold)
        iv_layout.addWidget(self.iv_threshold_spin)
        settings_layout.addLayout(iv_layout)
        
        # Volume spike threshold
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Volume Spike Threshold (%):"))
        self.vol_threshold_spin = QSpinBox()
        self.vol_threshold_spin.setRange(50, 1000)
        self.vol_threshold_spin.setValue(self.config['volume_spike_threshold'])
        self.vol_threshold_spin.valueChanged.connect(self.update_vol_threshold)
        vol_layout.addWidget(self.vol_threshold_spin)
        settings_layout.addLayout(vol_layout)
        
        # Notification settings
        self.notifications_check = QCheckBox("Enable Desktop Notifications")
        self.notifications_check.setChecked(self.config['enable_notifications'])
        self.notifications_check.stateChanged.connect(self.update_notifications)
        settings_layout.addWidget(self.notifications_check)
        
        self.sound_check = QCheckBox("Enable Sound Alerts")
        self.sound_check.setChecked(self.config['enable_sound'])
        self.sound_check.stateChanged.connect(self.update_sound)
        settings_layout.addWidget(self.sound_check)
        
        layout.addWidget(settings_group)
        
        # Recent alerts
        alerts_group = QGroupBox("Recent Alerts")
        alerts_layout = QVBoxLayout(alerts_group)
        
        self.alerts_widget = QListWidget()
        self.alerts_widget.setMaximumHeight(100)
        alerts_layout.addWidget(self.alerts_widget)
        
        clear_alerts_button = QPushButton("Clear Alerts")
        clear_alerts_button.clicked.connect(self.clear_alerts)
        alerts_layout.addWidget(clear_alerts_button)
        
        layout.addWidget(alerts_group)
        
        return widget
    
    def start_monitoring(self):
        """Start volatility monitoring"""
        if not self.watchlist:
            self.add_alert("No symbols in watchlist. Add symbols to monitor.")
            return
        
        self.monitoring_active = True
        self.monitor_timer.start(self.config['check_interval_seconds'] * 1000)
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Monitoring")
        self.status_label.setStyleSheet("color: #28a745;")
        
        self.add_alert(f"Started monitoring {len(self.watchlist)} symbols")
        self.plugin_status_changed.emit("Volatility monitoring started")
    
    def stop_monitoring(self):
        """Stop volatility monitoring"""
        self.monitoring_active = False
        self.monitor_timer.stop()
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: #dc3545;")
        
        self.add_alert("Monitoring stopped")
        self.plugin_status_changed.emit("Volatility monitoring stopped")
    
    def add_symbol(self):
        """Add symbol to watchlist"""
        symbol = self.symbol_input.text().strip().upper()
        if symbol and symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.watchlist_widget.addItem(symbol)
            self.symbol_input.clear()
            self.add_alert(f"Added {symbol} to watchlist")
    
    def remove_symbol(self):
        """Remove selected symbol from watchlist"""
        current_item = self.watchlist_widget.currentItem()
        if current_item:
            symbol = current_item.text()
            self.watchlist.remove(symbol)
            self.watchlist_widget.takeItem(self.watchlist_widget.row(current_item))
            self.add_alert(f"Removed {symbol} from watchlist")
    
    def check_alerts(self):
        """Check for alert conditions"""
        if not self.monitoring_active or not self.market_data_service:
            return
        
        for symbol in self.watchlist:
            try:
                # Get current market data
                current_data = self.get_symbol_data(symbol)
                if not current_data:
                    continue
                
                # Check various alert conditions
                self.check_iv_alerts(symbol, current_data)
                self.check_volume_alerts(symbol, current_data)
                self.check_price_alerts(symbol, current_data)
                
            except Exception as e:
                logger.error(f"Alert check error for {symbol}: {e}")
    
    def get_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        try:
            # This would use the market data service to get current data
            # For now, return placeholder data
            return {
                'current_price': 100.0,
                'volume': 1000000,
                'avg_volume': 500000,
                'iv': 0.30,
                'previous_iv': 0.25
            }
        except:
            return {}
    
    def check_iv_alerts(self, symbol: str, data: Dict[str, Any]):
        """Check for IV change alerts"""
        try:
            current_iv = data.get('iv', 0)
            previous_iv = data.get('previous_iv', current_iv)
            
            if previous_iv > 0:
                iv_change_pct = ((current_iv - previous_iv) / previous_iv) * 100
                
                if abs(iv_change_pct) >= self.config['iv_change_threshold']:
                    direction = "increased" if iv_change_pct > 0 else "decreased"
                    alert_msg = f"{symbol}: IV {direction} by {abs(iv_change_pct):.1f}%"
                    self.trigger_alert(alert_msg, 'volatility')
                    
        except Exception as e:
            logger.error(f"IV alert check error: {e}")
    
    def check_volume_alerts(self, symbol: str, data: Dict[str, Any]):
        """Check for volume spike alerts"""
        try:
            current_volume = data.get('volume', 0)
            avg_volume = data.get('avg_volume', 0)
            
            if avg_volume > 0:
                volume_ratio = (current_volume / avg_volume) * 100
                
                if volume_ratio >= self.config['volume_spike_threshold']:
                    alert_msg = f"{symbol}: Volume spike - {volume_ratio:.0f}% of average"
                    self.trigger_alert(alert_msg, 'volume')
                    
        except Exception as e:
            logger.error(f"Volume alert check error: {e}")
    
    def check_price_alerts(self, symbol: str, data: Dict[str, Any]):
        """Check for significant price moves"""
        try:
            # This would compare current price to previous close or moving average
            # Placeholder implementation
            pass
        except Exception as e:
            logger.error(f"Price alert check error: {e}")
    
    def trigger_alert(self, message: str, alert_type: str):
        """Trigger an alert"""
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        alert_with_time = f"[{timestamp}] {message}"
        
        # Add to alerts list
        self.add_alert(alert_with_time)
        
        # Send desktop notification if enabled
        if self.config['enable_notifications']:
            self.send_notification(message, alert_type)
        
        # Play sound if enabled
        if self.config['enable_sound']:
            self.play_alert_sound()
        
        # Emit signal for other components
        self.data_updated.emit({
            'type': 'alert',
            'message': message,
            'alert_type': alert_type,
            'timestamp': timestamp
        })
    
    def add_alert(self, message: str):
        """Add alert to display"""
        if hasattr(self, 'alerts_widget'):
            self.alerts_widget.addItem(message)
            # Keep only last 50 alerts
            while self.alerts_widget.count() > 50:
                self.alerts_widget.takeItem(0)
            # Scroll to bottom
            self.alerts_widget.scrollToBottom()
    
    def clear_alerts(self):
        """Clear all alerts"""
        if hasattr(self, 'alerts_widget'):
            self.alerts_widget.clear()
    
    def send_notification(self, message: str, alert_type: str):
        """Send desktop notification"""
        try:
            # This would integrate with system notifications
            # For now, just log the notification
            logger.info(f"Notification: {message}")
        except Exception as e:
            logger.error(f"Notification error: {e}")
    
    def play_alert_sound(self):
        """Play alert sound"""
        try:
            # This would play a sound file
            # For now, just log
            logger.info("Alert sound played")
        except Exception as e:
            logger.error(f"Sound alert error: {e}")
    
    def update_iv_threshold(self, value: int):
        """Update IV change threshold"""
        self.config['iv_change_threshold'] = value
    
    def update_vol_threshold(self, value: int):
        """Update volume spike threshold"""
        self.config['volume_spike_threshold'] = value
    
    def update_notifications(self, state: int):
        """Update notification setting"""
        self.config['enable_notifications'] = state == 2
    
    def update_sound(self, state: int):
        """Update sound alert setting"""
        self.config['enable_sound'] = state == 2
    
    def get_dock_widget_info(self) -> Dict[str, Any]:
        """Return dock widget configuration"""
        return {
            'title': 'Volatility Alerts',
            'area': 'right',  # Dock on right side
            'allowed_areas': ['left', 'right', 'bottom'],
            'features': ['closable', 'movable', 'floatable']
        }