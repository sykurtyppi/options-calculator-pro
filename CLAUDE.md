# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Environment
- **Initial setup**: `./setup.sh` - Creates virtual environment and installs dependencies
- **Quick run**: `./run.sh` - Activates venv and runs the application
- **Manual run**: `source .venv/bin/activate && python main.py`

### Testing
- **Run all tests**: `python scripts/test_runner.py all`
- **Run unit tests**: `python scripts/test_runner.py unit`
- **Run specific suite**: `python scripts/test_runner.py [unit|integration|gui|performance|api]`
- **Run with coverage**: `python scripts/test_runner.py unit --coverage`
- **Run linting**: `python scripts/test_runner.py --lint`
- **Install test deps**: `python scripts/test_runner.py --install-deps`

### Building
- **Build executable**: `python scripts/build.py executable`
- **Build installer**: `python scripts/build.py installer`
- **Build all**: `python scripts/build.py all`
- **Clean build**: `python scripts/build.py clean`

## Architecture Overview

### Core Components

**Application Entry Point**
- `main.py`: Main application entry with startup diagnostics, dependency checking, and professional dark theme setup
- `core/app.py`: Core application class that integrates all services and UI components

**Service Layer (services/)**
- `market_data.py`: Real-time market data integration using yfinance
- `options_service.py`: Core options pricing and analysis logic
- `volatility_service.py`: Volatility calculations and modeling
- `ml_service.py`: Machine learning predictions for trade outcomes
- `async_api.py`: Asynchronous API handling for performance

**Models (models/)**
- `option_data.py`: Options data structures
- `greeks.py`: Greeks calculations and data models
- `monte_carlo.py`: Monte Carlo simulation models
- `analysis_result.py`: Analysis result data structures
- `trade_data.py`: Trade data models

**Controllers (controllers/)**
- `analysis_controller.py`: Handles options analysis workflows
- `monte_carlo_controller.py`: Monte Carlo simulation controller
- `scanner_controller.py`: Options scanner functionality
- `history_controller.py`: Historical data and backtesting

**Views (views/)**
- `main_window.py`: Primary application window
- `analysis_view.py`: Options analysis interface
- `scanner_view.py`: Options scanner UI
- `history_view.py`: Historical analysis view
- `components/`: Reusable UI components (charts, widgets)

**Utilities (utils/)**
- `config_manager.py`: Configuration management with QObject signals
- `greeks_calculator.py`: Advanced Greeks calculations
- `logger.py`: Professional logging setup
- `thread_manager.py`: Threading utilities for performance

**Plugins (plugins/)**
- Extensible plugin system with base classes and example implementations
- Plugin manager handles loading and lifecycle
- Example plugins: Fibonacci retracement, RSI analysis, volatility alerts

### Key Technologies

- **UI Framework**: PySide6 (Qt6) with professional dark theme
- **Data Analysis**: pandas, numpy, scipy for financial calculations
- **Machine Learning**: scikit-learn for trade outcome predictions
- **Market Data**: yfinance for real-time data feeds
- **Visualization**: matplotlib for charts and analysis plots
- **Monte Carlo**: Advanced Heston stochastic volatility models

### Application Flow

1. **Startup**: Dependency checking, market data connectivity test, configuration loading
2. **Service Initialization**: Core services loaded in dependency order
3. **UI Setup**: Professional themed interface with modular views
4. **Analysis Workflow**: Data retrieval → Options pricing → Greeks calculation → ML predictions → Risk assessment
5. **Plugin System**: Dynamic loading of analysis extensions

### Key Features

- **Advanced Analytics**: Monte Carlo simulations using Heston stochastic volatility
- **Risk Management**: Kelly Criterion position sizing and comprehensive risk metrics
- **Real-time Data**: Live market data integration with fallback modes
- **Professional UI**: Dark themed interface optimized for trading workflows
- **Extensibility**: Plugin architecture for custom analysis tools
- **Performance**: Async processing and thread management for responsive UI

### Configuration

- Configuration stored in `~/.options_calculator_pro/config.json`
- Managed through `ConfigManager` with type safety and validation
- Settings include market data sources, analysis parameters, UI preferences

### Testing Strategy

- **Unit tests**: Core calculations and service logic
- **Integration tests**: Full analysis workflows
- **GUI tests**: UI components and user interactions (requires display)
- **Performance tests**: Benchmarking of analysis operations
- **API tests**: Market data connectivity (requires network)

### Error Handling

- Graceful degradation when market data is unavailable
- Fallback modes for offline operation
- Professional error logging and user notifications
- Startup diagnostics report system health