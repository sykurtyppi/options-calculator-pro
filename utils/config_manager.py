"""
Configuration Manager for Options Calculator Pro
Handles loading, saving, and managing application configuration
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self):
        self.config_dir = self.get_config_directory()
        self.config_file = self.config_dir / "config.json"
        self.config_data = self.load_config()
        
        # Ensure directories exist
        self.ensure_directories()
    
    def get_config_directory(self) -> Path:
        """Get platform-specific configuration directory"""
        import platform
        system = platform.system()
        
        if system == "Windows":
            config_dir = Path(os.environ.get("APPDATA", "")) / "Options Calculator Pro"
        elif system == "Darwin":  # macOS
            config_dir = Path.home() / ".options_calculator_pro"
        else:  # Linux and others
            config_dir = Path.home() / ".options_calculator_pro"
        
        return config_dir
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "version": "10.0.0",
            "first_run": True,
            "theme": "Dark Professional",
            "api_keys": {
                "alpha_vantage": "",
                "finnhub": "",
                "yahoo_finance": "enabled"
            },
            "trading": {
                "portfolio_value": 100000,
                "max_position_risk": 0.02,
                "default_contracts": 1,
                "preferred_strategies": ["calendar_spread"]
            },
            "analysis": {
                "monte_carlo_simulations": 10000,
                "confidence_level": 0.95,
                "volatility_model": "heston",
                "use_ml_predictions": True
            },
            "ui": {
                "window_width": 1200,
                "window_height": 800,
                "theme": "dark",
                "font_size": 10,
                "show_tooltips": True
            },
            "data": {
                "cache_ttl_minutes": 30,
                "data_source_priority": ["alpha_vantage", "yahoo_finance", "finnhub"],
                "enable_real_time": True
            },
            "logging": {
                "log_level": "INFO",
                "max_log_files": 5,
                "log_rotation_days": 7
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults (in case new fields were added)
                merged_config = self._merge_configs(default_config, loaded_config)
                return merged_config
                
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with defaults"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config_data, f, indent=4)
            
            return True
            
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.config_dir,
            self.config_dir / "data",
            self.config_dir / "cache",
            self.config_dir / "logs",
            self.config_dir / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'api_keys.alpha_vantage')"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config_data
        
        try:
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            # Save to file
            return self.save_config()
            
        except Exception as e:
            print(f"Error setting config value: {e}")
            return False
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for specific provider"""
        return self.get(f"api_keys.{provider}", "")
    
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """Set API key for specific provider"""
        return self.set(f"api_keys.{provider}", api_key)
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return self.get("trading.portfolio_value", 100000)
    
    def set_portfolio_value(self, value: float) -> bool:
        """Set portfolio value"""
        return self.set("trading.portfolio_value", value)
    
    def is_first_run(self) -> bool:
        """Check if this is the first run"""
        return self.get("first_run", True)
    
    def mark_first_run_complete(self) -> bool:
        """Mark first run as complete"""
        return self.set("first_run", False)
    
    def get_data_directories(self) -> Dict[str, Path]:
        """Get all data directory paths"""
        return {
            "config": self.config_dir,
            "data": self.config_dir / "data",
            "cache": self.config_dir / "cache",
            "logs": self.config_dir / "logs",
            "backups": self.config_dir / "backups"
        }
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            if self.config_file.exists():
                self.config_file.unlink()
            
            self.config_data = self.load_config()
            return self.save_config()
            
        except Exception as e:
            print(f"Error resetting config: {e}")
            return False
    
    def export_config(self, export_path: Path) -> bool:
        """Export configuration to specified path"""
        try:
            # Remove sensitive data for export
            export_config = self.config_data.copy()
            if "api_keys" in export_config:
                for key in export_config["api_keys"]:
                    if export_config["api_keys"][key]:
                        export_config["api_keys"][key] = "***REDACTED***"
            
            with open(export_path, 'w') as f:
                json.dump(export_config, f, indent=4)
            
            return True
            
        except Exception as e:
            print(f"Error exporting config: {e}")
            return False
    
    def save(self):
        """Save current configuration to file"""
        try:
            self.save_config()
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
