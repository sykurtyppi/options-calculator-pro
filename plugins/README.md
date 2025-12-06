# Plugins System

The Professional Options Calculator features a comprehensive plugin architecture that allows for easy extension and customization of functionality.

## Plugin Types

### 1. Analysis Plugins
Extend the analysis capabilities with custom technical indicators and market analysis tools.

**Base Class:** `AnalysisPlugin`

**Required Methods:**
- `analyze(symbol, data)` - Perform analysis and return results

**Examples:**
- RSI Analysis
- MACD Signals  
- Custom Technical Indicators
- Sentiment Analysis

### 2. Strategy Plugins
Implement custom trading strategies and position recommendations.

**Base Class:** `StrategyPlugin`

**Required Methods:**
- `get_strategy_name()` - Return strategy name
- `calculate_position(market_data)` - Calculate position recommendation
- `get_risk_metrics(position)` - Calculate risk metrics

**Examples:**
- Fibonacci Retracement
- Mean Reversion Strategies
- Momentum Trading
- Custom Options Strategies

### 3. Data Provider Plugins
Add new data sources and market feeds.

**Base Class:** `DataPlugin`

**Required Methods:**
- `get_data_types()` - Return supported data types
- `fetch_data(symbol, data_type, **kwargs)` - Fetch data

**Examples:**
- Custom API Providers
- Economic Calendar Data
- News Feed Integration
- Alternative Data Sources

### 4. UI Enhancement Plugins
Extend the user interface with custom components and tools.

**Base Class:** `UIPlugin`

**Required Methods:**
- `get_widget()` - Return main widget

**Optional Methods:**
- `get_dock_widget_info()` - Dock widget configuration
- `get_tab_info()` - Tab widget configuration

**Examples:**
- Advanced Charting Tools
- Custom Dashboards
- Theme Managers
- Layout Enhancements

### 5. Alert System Plugins
Implement monitoring and notification systems.

**Base Class:** `BasePlugin` (with custom alert logic)

**Examples:**
- Volatility Alerts
- Price Target Monitors
- Economic Event Notifications
- Custom Alert Conditions

## Creating a Plugin

### 1. Basic Plugin Structure

```python
from plugins.base_plugin import BasePlugin, PluginMetadata

def get_plugin_metadata() -> PluginMetadata:
    return PluginMetadata(
        name="My Plugin",
        version="1.0.0",
        description="Plugin description",
        author="Your Name",
        category="analysis"  # or strategy, data, ui, alert, export
    )

class MyPlugin(BasePlugin):
    def initialize(self, app_context):
        # Initialize your plugin
        return True
    
    def cleanup(self):
        # Clean up resources
        return True