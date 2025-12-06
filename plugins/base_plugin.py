"""
Base plugin architecture for the Professional Options Calculator
Extensible plugin system for custom strategies and features
"""

from abc import ABC, abstractmethod
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class PluginMetadata:
    """Plugin metadata container"""
    
    def __init__(self, name: str, version: str, description: str, 
                 author: str, category: str = "general"):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.category = category
        self.enabled = True
        self.install_date = datetime.now()
        self.dependencies = []
        self.min_app_version = "1.0.0"
        self.max_app_version = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'category': self.category,
            'enabled': self.enabled,
            'install_date': self.install_date.isoformat(),
            'dependencies': self.dependencies,
            'min_app_version': self.min_app_version,
            'max_app_version': self.max_app_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Create metadata from dictionary"""
        metadata = cls(
            name=data['name'],
            version=data['version'],
            description=data['description'],
            author=data['author'],
            category=data.get('category', 'general')
        )
        metadata.enabled = data.get('enabled', True)
        metadata.dependencies = data.get('dependencies', [])
        metadata.min_app_version = data.get('min_app_version', '1.0.0')
        metadata.max_app_version = data.get('max_app_version')
        
        if 'install_date' in data:
            metadata.install_date = datetime.fromisoformat(data['install_date'])
            
        return metadata


class BasePlugin(QObject, ABC):
    """Base class for all plugins"""
    
    # Plugin lifecycle signals
    plugin_loaded = Signal()
    plugin_unloaded = Signal()
    plugin_error = Signal(str)  # Error message
    plugin_status_changed = Signal(str)  # Status message
    
    # Data signals
    data_updated = Signal(object)  # Emits updated data
    result_ready = Signal(object)  # Emits analysis results
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__()
        self.metadata = metadata
        self.is_loaded = False
        self.is_active = False
        self.config = {}
        self.services = {}
        
    @abstractmethod
    def initialize(self, app_context: Dict[str, Any]) -> bool:
        """Initialize the plugin with app context
        
        Args:
            app_context: Dictionary containing app services and config
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up plugin resources
        
        Returns:
            True if cleanup successful, False otherwise
        """
        pass
    
    def load(self, app_context: Dict[str, Any]) -> bool:
        """Load the plugin"""
        try:
            if self.is_loaded:
                return True
                
            success = self.initialize(app_context)
            if success:
                self.is_loaded = True
                self.plugin_loaded.emit()
                logger.info(f"Plugin '{self.metadata.name}' loaded successfully")
            else:
                logger.error(f"Plugin '{self.metadata.name}' initialization failed")
                
            return success
            
        except Exception as e:
            error_msg = f"Error loading plugin '{self.metadata.name}': {str(e)}"
            logger.error(error_msg)
            self.plugin_error.emit(error_msg)
            return False
    
    def unload(self) -> bool:
        """Unload the plugin"""
        try:
            if not self.is_loaded:
                return True
                
            success = self.cleanup()
            if success:
                self.is_loaded = False
                self.is_active = False
                self.plugin_unloaded.emit()
                logger.info(f"Plugin '{self.metadata.name}' unloaded successfully")
            else:
                logger.error(f"Plugin '{self.metadata.name}' cleanup failed")
                
            return success
            
        except Exception as e:
            error_msg = f"Error unloading plugin '{self.metadata.name}': {str(e)}"
            logger.error(error_msg)
            self.plugin_error.emit(error_msg)
            return False
    
    def activate(self) -> bool:
        """Activate the plugin"""
        if not self.is_loaded:
            return False
            
        self.is_active = True
        self.plugin_status_changed.emit(f"Plugin '{self.metadata.name}' activated")
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the plugin"""
        self.is_active = False
        self.plugin_status_changed.emit(f"Plugin '{self.metadata.name}' deactivated")
        return True
    
    def get_config_widget(self) -> Optional[QWidget]:
        """Return configuration widget for plugin settings
        
        Returns:
            QWidget for plugin configuration or None if no config needed
        """
        return None
    
    def get_menu_actions(self) -> List[Dict[str, Any]]:
        """Return list of menu actions this plugin provides
        
        Returns:
            List of menu action dictionaries with keys: 'text', 'callback', 'icon', 'shortcut'
        """
        return []
    
    def get_toolbar_actions(self) -> List[Dict[str, Any]]:
        """Return list of toolbar actions this plugin provides
        
        Returns:
            List of toolbar action dictionaries
        """
        return []
    
    def save_config(self, config: Dict[str, Any]):
        """Save plugin configuration"""
        self.config = config
    
    def load_config(self) -> Dict[str, Any]:
        """Load plugin configuration"""
        return self.config
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information"""
        return {
            'name': self.metadata.name,
            'version': self.metadata.version,
            'loaded': self.is_loaded,
            'active': self.is_active,
            'category': self.metadata.category
        }


class AnalysisPlugin(BasePlugin):
    """Base class for analysis plugins"""
    
    analysis_complete = Signal(str, object)  # symbol, results
    
    @abstractmethod
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on symbol data
        
        Args:
            symbol: Stock symbol
            data: Market data dictionary
            
        Returns:
            Analysis results dictionary
        """
        pass
    
    def get_analysis_priority(self) -> int:
        """Get analysis priority (lower number = higher priority)
        
        Returns:
            Priority value (0-100)
        """
        return 50
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbol patterns
        
        Returns:
            List of symbol patterns (e.g., ['*', 'AAPL', 'TECH:*'])
        """
        return ['*']  # Support all symbols by default


class StrategyPlugin(BasePlugin):
    """Base class for strategy plugins"""
    
    strategy_signal = Signal(str, str, object)  # signal_type, symbol, data
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy display name"""
        pass
    
    @abstractmethod
    def calculate_position(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate recommended position for strategy
        
        Args:
            market_data: Current market data
            
        Returns:
            Position recommendation dictionary
        """
        pass
    
    @abstractmethod
    def get_risk_metrics(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for position
        
        Args:
            position: Position data
            
        Returns:
            Risk metrics dictionary
        """
        pass


class DataPlugin(BasePlugin):
    """Base class for data provider plugins"""
    
    data_received = Signal(str, object)  # data_type, data
    
    @abstractmethod
    def get_data_types(self) -> List[str]:
        """Get list of data types this plugin provides
        
        Returns:
            List of data type strings
        """
        pass
    
    @abstractmethod
    def fetch_data(self, symbol: str, data_type: str, **kwargs) -> Any:
        """Fetch data for symbol
        
        Args:
            symbol: Stock symbol
            data_type: Type of data to fetch
            **kwargs: Additional parameters
            
        Returns:
            Requested data
        """
        pass
    
    def get_data_priority(self) -> int:
        """Get data source priority (lower number = higher priority)
        
        Returns:
            Priority value (0-100)
        """
        return 50


class UIPlugin(BasePlugin):
    """Base class for UI enhancement plugins"""
    
    @abstractmethod
    def get_widget(self) -> QWidget:
        """Get main widget for this plugin
        
        Returns:
            QWidget to be integrated into the main UI
        """
        pass
    
    def get_dock_widget_info(self) -> Optional[Dict[str, Any]]:
        """Get dock widget information
        
        Returns:
            Dictionary with dock widget configuration or None
        """
        return None
    
    def get_tab_info(self) -> Optional[Dict[str, Any]]:
        """Get tab information for main tab widget
        
        Returns:
            Dictionary with tab configuration or None
        """
        return None


class PluginManager(QObject):
    """Central plugin management system"""
    
    plugin_loaded = Signal(str)  # plugin name
    plugin_unloaded = Signal(str)  # plugin name
    plugin_error = Signal(str, str)  # plugin name, error message
    
    def __init__(self, plugins_directory: str = "plugins"):
        super().__init__()
        self.plugins_directory = plugins_directory
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.app_context = {}
        
        # Create plugins directory if it doesn't exist
        os.makedirs(plugins_directory, exist_ok=True)
        
        # Plugin categories
        self.categories = {
            'analysis': 'Analysis Tools',
            'strategy': 'Trading Strategies', 
            'data': 'Data Providers',
            'ui': 'User Interface',
            'export': 'Export Tools',
            'alert': 'Alert Systems',
            'general': 'General Plugins'
        }
    
    def set_app_context(self, context: Dict[str, Any]):
        """Set application context for plugins"""
        self.app_context = context
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugins directory
        
        Returns:
            List of discovered plugin module names
        """
        discovered = []
        
        try:
            for item in os.listdir(self.plugins_directory):
                plugin_path = os.path.join(self.plugins_directory, item)
                
                # Check for Python module
                if item.endswith('.py') and not item.startswith('__'):
                    module_name = item[:-3]
                    discovered.append(module_name)
                    
                # Check for plugin package
                elif os.path.isdir(plugin_path):
                    init_file = os.path.join(plugin_path, '__init__.py')
                    if os.path.exists(init_file):
                        discovered.append(item)
                        
        except Exception as e:
            logger.error(f"Error discovering plugins: {e}")
            
        return discovered
    
    def load_plugin(self, module_name: str) -> bool:
        """Load a plugin by module name
        
        Args:
            module_name: Name of the plugin module
            
        Returns:
            True if plugin loaded successfully
        """
        try:
            # Import the plugin module
            import importlib.util
            import sys
            
            plugin_path = os.path.join(self.plugins_directory, f"{module_name}.py")
            if not os.path.exists(plugin_path):
                # Try package format
                plugin_path = os.path.join(self.plugins_directory, module_name, '__init__.py')
                
            if not os.path.exists(plugin_path):
                logger.error(f"Plugin file not found: {plugin_path}")
                return False
            
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Look for plugin class
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    plugin_class = attr
                    break
            
            if not plugin_class:
                logger.error(f"No plugin class found in {module_name}")
                return False
            
            # Get metadata
            metadata_func = getattr(module, 'get_plugin_metadata', None)
            if metadata_func:
                metadata = metadata_func()
            else:
                # Create default metadata
                metadata = PluginMetadata(
                    name=module_name,
                    version="1.0.0",
                    description=f"Plugin: {module_name}",
                    author="Unknown"
                )
            
            # Create plugin instance
            plugin = plugin_class(metadata)
            
            # Load the plugin
            if plugin.load(self.app_context):
                self.plugins[module_name] = plugin
                self.plugin_metadata[module_name] = metadata
                
                # Connect signals
                plugin.plugin_error.connect(
                    lambda msg: self.plugin_error.emit(module_name, msg)
                )
                
                self.plugin_loaded.emit(module_name)
                logger.info(f"Plugin '{module_name}' loaded successfully")
                return True
            else:
                logger.error(f"Failed to load plugin '{module_name}'")
                return False
                
        except Exception as e:
            error_msg = f"Error loading plugin '{module_name}': {str(e)}"
            logger.error(error_msg)
            self.plugin_error.emit(module_name, error_msg)
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if plugin unloaded successfully
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin '{plugin_name}' not found")
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            if plugin.unload():
                del self.plugins[plugin_name]
                del self.plugin_metadata[plugin_name]
                self.plugin_unloaded.emit(plugin_name)
                return True
            else:
                return False
                
        except Exception as e:
            error_msg = f"Error unloading plugin '{plugin_name}': {str(e)}"
            logger.error(error_msg)
            self.plugin_error.emit(plugin_name, error_msg)
            return False
    
    def get_plugins_by_category(self, category: str) -> List[BasePlugin]:
        """Get all plugins in a specific category
        
        Args:
            category: Plugin category
            
        Returns:
            List of plugins in the category
        """
        return [
            plugin for plugin in self.plugins.values()
            if plugin.metadata.category == category and plugin.is_loaded
        ]
    
    def get_active_plugins(self) -> List[BasePlugin]:
        """Get all active plugins
        
        Returns:
            List of active plugins
        """
        return [
            plugin for plugin in self.plugins.values()
            if plugin.is_active
        ]
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(name)
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins
        
        Returns:
            Dictionary mapping plugin names to status info
        """
        return {
            name: plugin.get_status()
            for name, plugin in self.plugins.items()
        }
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            True if reload successful
        """
        if self.unload_plugin(plugin_name):
            return self.load_plugin(plugin_name)
        return False
    
    def load_all_plugins(self):
        """Load all discovered plugins"""
        discovered = self.discover_plugins()
        
        for plugin_name in discovered:
            try:
                self.load_plugin(plugin_name)
            except Exception as e:
                logger.error(f"Failed to load plugin '{plugin_name}': {e}")
    
    def unload_all_plugins(self):
        """Unload all plugins"""
        plugin_names = list(self.plugins.keys())
        for plugin_name in plugin_names:
            self.unload_plugin(plugin_name)
    
    def save_plugin_states(self, config_file: str = "plugin_states.json"):
        """Save current plugin states to file"""
        try:
            states = {}
            for name, plugin in self.plugins.items():
                states[name] = {
                    'enabled': plugin.metadata.enabled,
                    'active': plugin.is_active,
                    'config': plugin.load_config()
                }
            
            config_path = os.path.join(self.plugins_directory, config_file)
            with open(config_path, 'w') as f:
                json.dump(states, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving plugin states: {e}")
    
    def load_plugin_states(self, config_file: str = "plugin_states.json"):
        """Load plugin states from file"""
        try:
            config_path = os.path.join(self.plugins_directory, config_file)
            if not os.path.exists(config_path):
                return
                
            with open(config_path, 'r') as f:
                states = json.load(f)
            
            for name, state in states.items():
                if name in self.plugins:
                    plugin = self.plugins[name]
                    plugin.metadata.enabled = state.get('enabled', True)
                    
                    if state.get('config'):
                        plugin.save_config(state['config'])
                    
                    if state.get('active', False):
                        plugin.activate()
                        
        except Exception as e:
            logger.error(f"Error loading plugin states: {e}")