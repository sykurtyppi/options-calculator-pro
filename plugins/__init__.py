"""
Professional Options Calculator - Plugins Module
Extensible plugin architecture for custom features and strategies
"""

from .base_plugin import (
    BasePlugin,
    AnalysisPlugin,
    StrategyPlugin,
    DataPlugin,
    UIPlugin,
    PluginManager,
    PluginMetadata
)

__all__ = [
    'BasePlugin',
    'AnalysisPlugin', 
    'StrategyPlugin',
    'DataPlugin',
    'UIPlugin',
    'PluginManager',
    'PluginMetadata'
]

__version__ = "1.0.0"

# Plugin discovery and registration helpers
def register_plugin_directory(directory: str):
   """Register a directory for plugin discovery"""
   import os
   import sys
   if os.path.exists(directory):
       sys.path.insert(0, directory)

def get_available_plugins():
   """Get list of available plugin categories and descriptions"""
   return {
       'analysis': {
           'name': 'Analysis Plugins',
           'description': 'Technical analysis and market indicators',
           'examples': ['RSI Analysis', 'MACD Signals', 'Bollinger Bands']
       },
       'strategy': {
           'name': 'Strategy Plugins', 
           'description': 'Trading strategies and position recommendations',
           'examples': ['Fibonacci Retracement', 'Mean Reversion', 'Momentum Trading']
       },
       'data': {
           'name': 'Data Provider Plugins',
           'description': 'External data sources and market feeds',
           'examples': ['Custom API Provider', 'Economic Calendar', 'News Feed']
       },
       'ui': {
           'name': 'UI Enhancement Plugins',
           'description': 'User interface enhancements and tools',
           'examples': ['Advanced Charts', 'Custom Dashboards', 'Theme Manager']
       },
       'alert': {
           'name': 'Alert System Plugins',
           'description': 'Monitoring and notification systems',
           'examples': ['Volatility Alerts', 'Price Targets', 'Economic Events']
       },
       'export': {
           'name': 'Export Plugins',
           'description': 'Data export and reporting tools',
           'examples': ['PDF Reports', 'Excel Export', 'Email Reports']
       }
   }