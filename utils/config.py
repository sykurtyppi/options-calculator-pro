# Config loader/saver
"""
Configuration Manager - Professional Options Calculator Pro
Handles all application settings and preferences
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from PySide6.QtCore import QObject, Signal


class ConfigManager(QObject):
    """
    Professional configuration manager with type safety and validation
    """
    
    # Signals
    config_changed = Signal(str, object)  # key, value
    
    def __init__(self, config_file: str = "config.json"):
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration file path
        self.config_dir = Path.home() / ".options_calculator_pro"
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / config_file
        
        # Default configuration
        self._defaults = self._get_default_config()
        
        # Current configuration
        self._config = {}
        
        # Load configuration
        self.load()
        
        self.logger.info(f"Configuration manager initialized: {self.config_file}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Application settings
            "app": {
                "version": "10.0.0",
                "first_run": True,
                "auto_save": True,
                "auto_backup": True,
                "debug_mode": False
            },
            
            # User Interface settings
            "ui": {
                "theme": "dark",  # dark, light, auto
                "font_family": "Segoe UI",
                "font_size": 9,
                "show_tooltips": True,
                "animation_enabled": True,
                "high_dpi": True
            },
            
            # Window settings
            "window": {
                "geometry": None,
                "state": None,
                "remember_size": True,
                "start_maximized": False
            },
            
            # Market Data API settings
            "api": {
                "alpha_vantage_key": "",
                "finnhub_key": "",
                "polygon_key": "",
                "rate_limit_per_minute": 60,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "cache_duration_minutes": 15
            },
            
            # Trading settings
            "trading": {
                "default_contracts": 1,
                "portfolio_value": 100000.0,
                "max_position_risk": 0.02,  # 2%
                "commission_per_contract": 0.65,
                "use_kelly_sizing": True,
                "min_confidence_threshold": 50.0,
                "favorite_symbols": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
            },
            
            # Analysis settings
            "analysis": {
                "monte_carlo_simulations": 10000,
                "use_heston_model": True,
                "volatility_lookback_days": 30,
                "earnings_proximity_days": 14,
                "min_option_volume": 100,
                "min_open_interest": 500,
                "iv_rank_percentile": 30.0
            },
            
            # Machine Learning settings
            "ml": {
                "enabled": True,
                "model_type": "ensemble",  # ensemble, random_forest, logistic
                "retrain_frequency_days": 30,
                "min_training_samples": 50,
                "cross_validation_folds": 5,
                "feature_selection": True
            },
            
            # Risk Management settings
            "risk": {
                "max_portfolio_heat": 0.10,  # 10%
                "position_sizing_method": "kelly",  # kelly, fixed, percent
                "stop_loss_percentage": 0.50,  # 50%
                "profit_target_multiplier": 2.0,
                "max_days_to_expiration": 45,
                "min_days_to_expiration": 7
            },
            
            # Notifications settings
            "notifications": {
                "enabled": True,
                "email_alerts": False,
                "desktop_notifications": True,
                "price_alerts": True,
                "earnings_reminders": True,
                "analysis_completion": True
            },
            
            # Data storage settings
            "storage": {
                "data_directory": str(Path.home() / ".options_calculator_pro" / "data"),
                "backup_directory": str(Path.home() / ".options_calculator_pro" / "backups"),
                "max_backup_files": 10,
                "compress_backups": True,
                "auto_cleanup_days": 90
            },
            
            # Performance settings
            "performance": {
                "use_multiprocessing": True,
                "max_worker_threads": 4,
                "cache_size_mb": 100,
                "preload_data": True,
                "optimize_for_battery": False
            }
        }
    
    def load(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                self._config = self._merge_configs(self._defaults, loaded_config)
                
                self.logger.info("Configuration loaded successfully")
                return True
            else:
                # First run - use defaults
                self._config = self._defaults.copy()
                self.save()  # Create initial config file
                self.logger.info("Created default configuration file")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Fall back to defaults
            self._config = self._defaults.copy()
            return False
    
    def save(self) -> bool:
        """Save configuration to file"""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            
            self.logger.debug("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'ui.theme', 'trading.portfolio_value')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Split key by dots
            keys = key.split('.')
            value = self._config
            
            # Navigate through nested dictionary
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.warning(f"Error getting config key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'ui.theme')
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split key by dots
            keys = key.split('.')
            config_level = self._config
            
            # Navigate to parent level
            for k in keys[:-1]:
                if k not in config_level:
                    config_level[k] = {}
                config_level = config_level[k]
            
            # Set the value
            old_value = config_level.get(keys[-1])
            config_level[keys[-1]] = value
            
            # Emit signal if value changed
            if old_value != value:
                self.config_changed.emit(key, value)
            
            self.logger.debug(f"Configuration set: {key} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config key '{key}': {e}")
            return False
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        return self.get(key, None) is not None
    
    def remove(self, key: str) -> bool:
        """Remove configuration key"""
        try:
            keys = key.split('.')
            config_level = self._config
            
            # Navigate to parent level
            for k in keys[:-1]:
                if k not in config_level:
                    return False
                config_level = config_level[k]
            
            # Remove the key
            if keys[-1] in config_level:
                del config_level[keys[-1]]
                self.config_changed.emit(key, None)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing config key '{key}': {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            self._config = self._defaults.copy()
            self.save()
            self.logger.info("Configuration reset to defaults")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {e}")
            return False
    
    def export_config(self, file_path: str) -> bool:
        """Export configuration to file"""
        try:
            export_path = Path(file_path)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Configuration exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, file_path: str) -> bool:
        """Import configuration from file"""
        try:
            import_path = Path(file_path)
            
            if not import_path.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # Validate and merge with defaults
            self._config = self._merge_configs(self._defaults, imported_config)
            self.save()
            
            self.logger.info(f"Configuration imported from: {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.get(section, {})
    
    def set_section(self, section: str, values: Dict[str, Any]) -> bool:
        """Set entire configuration section"""
        try:
            if section in self._config:
                old_section = self._config[section].copy()
            else:
                old_section = {}
            
            self._config[section] = values
            
            # Emit signals for changed values
            for key, value in values.items():
                full_key = f"{section}.{key}"
                old_value = old_section.get(key)
                if old_value != value:
                    self.config_changed.emit(full_key, value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config section '{section}': {e}")
            return False
    
    def validate_config(self) -> Dict[str, str]:
        """
        Validate configuration and return errors
        
        Returns:
            Dictionary of validation errors (key -> error message)
        """
        errors = {}
        
        try:
            # Validate portfolio value
            portfolio_value = self.get("trading.portfolio_value", 0)
            if not isinstance(portfolio_value, (int, float)) or portfolio_value <= 0:
                errors["trading.portfolio_value"] = "Portfolio value must be a positive number"
            
            # Validate max position risk
            max_risk = self.get("trading.max_position_risk", 0)
            if not isinstance(max_risk, (int, float)) or not (0 < max_risk <= 1):
                errors["trading.max_position_risk"] = "Max position risk must be between 0 and 1"
            
            # Validate Monte Carlo simulations
            mc_sims = self.get("analysis.monte_carlo_simulations", 0)
            if not isinstance(mc_sims, int) or mc_sims < 1000:
                errors["analysis.monte_carlo_simulations"] = "Monte Carlo simulations must be at least 1000"
            
            # Validate API timeout
            timeout = self.get("api.timeout_seconds", 0)
            if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 300:
                errors["api.timeout_seconds"] = "API timeout must be between 1 and 300 seconds"
            
            # Validate data directories
            data_dir = self.get("storage.data_directory", "")
            if not data_dir:
                errors["storage.data_directory"] = "Data directory path is required"
            else:
                try:
                    Path(data_dir).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors["storage.data_directory"] = f"Cannot create data directory: {e}"
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            errors["general"] = f"Configuration validation error: {e}"
        
        return errors
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_backup(self) -> Optional[str]:
        """Create backup of current configuration"""
        try:
            from datetime import datetime
            
            backup_dir = Path(self.get("storage.backup_directory"))
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"config_backup_{timestamp}.json"
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Configuration backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Error creating configuration backup: {e}")
            return None
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration metadata and statistics"""
        return {
            "config_file": str(self.config_file),
            "config_size": self.config_file.stat().st_size if self.config_file.exists() else 0,
            "last_modified": self.config_file.stat().st_mtime if self.config_file.exists() else None,
            "sections": list(self._config.keys()),
            "total_keys": sum(len(v) if isinstance(v, dict) else 1 for v in self._config.values()),
            "version": self.get("app.version", "unknown")
        }
    
    @property
    def config_path(self) -> Path:
        """Get configuration file path"""
        return self.config_file
    
    @property
    def all_config(self) -> Dict[str, Any]:
        """Get entire configuration (read-only)"""
        return self._config.copy()