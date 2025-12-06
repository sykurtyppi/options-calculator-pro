# plugins/my_complete_plugin.py

from typing import Dict, Any, List
from plugins.base_plugin import AnalysisPlugin, PluginMetadata
import logging

logger = logging.getLogger(__name__)

def get_plugin_metadata() -> PluginMetadata:
    return PluginMetadata(
        name="My Complete Plugin",
        version="1.0.0",
        description="Example showing where all code pieces go",
        author="Your Name",
        category="analysis"
    )

class MyCompletePlugin(AnalysisPlugin):
    def initialize(self, app_context: Dict[str, Any]) -> bool:
        self.market_data_service = app_context.get('market_data_service')
        self.config = {
            'rsi_period': 14,
            'enable_alerts': True,
            'threshold': 70
        }
        
        # 5. PLUGIN COMMUNICATION - Status update
        self.plugin_status_changed.emit("Plugin initialized successfully")
        return True
    
    def cleanup(self) -> bool:
        # 5. PLUGIN COMMUNICATION - Status update  
        self.plugin_status_changed.emit("Plugin cleanup completed")
        return True
    
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Do analysis work
            results = {
                'symbol': symbol,
                'rsi': 65,
                'signal': 'BUY'
            }
            
            # 5. PLUGIN COMMUNICATION - Emit results
            self.data_updated.emit(results)
            self.analysis_complete.emit(symbol, results)
            
            return results
            
        except Exception as e:
            # 5. PLUGIN COMMUNICATION - Emit error
            self.plugin_error.emit(f"Analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def get_config_widget(self):
        """6. CONFIGURATION WIDGET - This entire method"""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QCheckBox
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # RSI Period setting
        layout.addWidget(QLabel("RSI Period:"))
        rsi_spin = QSpinBox()
        rsi_spin.setRange(5, 50)
        rsi_spin.setValue(self.config['rsi_period'])
        rsi_spin.valueChanged.connect(lambda v: self.config.update({'rsi_period': v}))
        layout.addWidget(rsi_spin)
        
        # Enable alerts
        alerts_check = QCheckBox("Enable Alerts")
        alerts_check.setChecked(self.config['enable_alerts'])
        alerts_check.stateChanged.connect(lambda state: self.config.update({'enable_alerts': state == 2}))
        layout.addWidget(alerts_check)
        
        return widget
    
    def get_menu_actions(self):
        """7. MENU ACTIONS - This entire method"""
        return [
            {
                'text': 'Run RSI Analysis',
                'callback': self.run_rsi_analysis,
                'shortcut': 'Ctrl+R'
            },
            {
                'text': 'Show RSI Settings',
                'callback': self.show_settings_dialog,
                'shortcut': 'Ctrl+Shift+R'
            }
        ]
    
    def get_toolbar_actions(self):
        """7. TOOLBAR ACTIONS - This entire method"""
        return [
            {
                'text': 'RSI Tool',
                'callback': self.rsi_toolbar_action,
                'icon': 'rsi_icon.png',
                'tooltip': 'Quick RSI analysis'
            }
        ]
    
    # Callback methods for menu/toolbar actions
    def run_rsi_analysis(self):
        """Menu action callback"""
        # 5. PLUGIN COMMUNICATION - Status update
        self.plugin_status_changed.emit("Running RSI analysis...")
        
        # Your analysis code here
        
        # 5. PLUGIN COMMUNICATION - Completion notification
        self.plugin_status_changed.emit("RSI analysis completed")
    
    def show_settings_dialog(self):
        """Menu action callback"""
        # Show settings dialog
        pass
    
    def rsi_toolbar_action(self):
        """Toolbar action callback"""
        # 5. PLUGIN COMMUNICATION
        self.plugin_status_changed.emit("RSI toolbar action executed")