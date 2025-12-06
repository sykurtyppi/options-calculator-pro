"""
Pytest configuration and shared fixtures
Global test setup and teardown
"""

import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import json

# Import application modules for testing
from services.config_manager import ConfigManager
from services.market_data_service import MarketDataService
from services.cache_manager import CacheManager
from models.market_data import MarketData, OptionChain


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        'test_mode': True,
        'use_mock_data': True,
        'test_symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'test_date_range': {
            'start': '2024-01-01',
            'end': '2024-03-01'
        }
    }


@pytest.fixture(scope="session") 
def temp_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="options_calc_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config_manager(temp_data_dir):
    """Mock configuration manager for tests"""
    config_file = temp_data_dir / "test_config.json"
    
    # Create test configuration
    test_config = {
        'api_keys': {
            'alpha_vantage': 'TEST_KEY',
            'finnhub': 'TEST_KEY'
        },
        'cache_ttl_minutes': 1,
        'max_retries': 2,
        'timeout_seconds': 10,
        'data_directory': str(temp_data_dir),
        'portfolio_value': 100000,
        'max_position_risk': 0.02
    }
    
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
    
    config_manager = ConfigManager(str(config_file))
    return config_manager


@pytest.fixture
def sample_historical_data():
    """Generate sample historical market data"""
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible tests
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0.01, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0.01, 0.005))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    data.set_index('Date', inplace=True)
    return data


@pytest.fixture
def sample_options_data():
    """Generate sample options chain data"""
    current_price = 150.0
    expiry_date = datetime.now() + timedelta(days=30)
    
    # Generate strikes around current price
    strikes = np.arange(130, 171, 5)
    
    calls = []
    puts = []
    
    for strike in strikes:
        # Simple Black-Scholes approximation for test data
        moneyness = strike / current_price
        
        # Call option
        call_iv = 0.25 + 0.1 * abs(moneyness - 1)  # Volatility smile
        call_price = max(0.1, (current_price - strike) + call_iv * 10)
        
        calls.append({
            'strike': strike,
            'bid': call_price * 0.98,
            'ask': call_price * 1.02,
            'last': call_price,
            'volume': np.random.randint(0, 1000),
            'openInterest': np.random.randint(0, 5000),
            'impliedVolatility': call_iv,
            'delta': max(0, min(1, 0.5 + (current_price - strike) / 20)),
            'gamma': 0.05,
            'theta': -0.02,
            'vega': 0.1
        })
        
        # Put option
        put_iv = 0.25 + 0.1 * abs(moneyness - 1)
        put_price = max(0.1, (strike - current_price) + put_iv * 10)
        
        puts.append({
            'strike': strike,
            'bid': put_price * 0.98,
            'ask': put_price * 1.02,
            'last': put_price,
            'volume': np.random.randint(0, 1000),
            'openInterest': np.random.randint(0, 5000),
            'impliedVolatility': put_iv,
            'delta': max(-1, min(0, -0.5 + (current_price - strike) / 20)),
            'gamma': 0.05,
            'theta': -0.02,
            'vega': 0.1
        })
    
    return {
        'symbol': 'AAPL',
        'expiry': expiry_date.strftime('%Y-%m-%d'),
        'calls': calls,
        'puts': puts,
        'underlying_price': current_price
    }


@pytest.fixture
def mock_market_data_service(sample_historical_data, sample_options_data):
    """Mock market data service for tests"""
    service = Mock(spec=MarketDataService)
    
    # Configure mock responses
    service.get_historical_data.return_value = sample_historical_data
    service.get_options_chain.return_value = sample_options_data
    service.get_current_price.return_value = 150.0
    service.get_earnings_date.return_value = datetime.now() + timedelta(days=45)
    service.is_connected.return_value = True
    
    return service


@pytest.fixture
def mock_cache_manager(temp_data_dir):
    """Mock cache manager for tests"""
    cache_manager = CacheManager(str(temp_data_dir / "cache"))
    return cache_manager


@pytest.fixture
def sample_market_data_model(sample_historical_data):
    """Create sample MarketData model instance"""
    return MarketData(
        symbol="AAPL",
        current_price=150.0,
        historical_data=sample_historical_data,
        volume=5000000,
        avg_volume=3000000,
        market_cap=2500000000000,
        pe_ratio=25.5,
        beta=1.2,
        fifty_two_week_high=180.0,
        fifty_two_week_low=120.0,
        analyst_target=160.0,
        earnings_date=datetime.now() + timedelta(days=45),
        sector="Technology",
        industry="Consumer Electronics"
    )


@pytest.fixture
def sample_option_chain_model(sample_options_data):
    """Create sample OptionChain model instance"""
    return OptionChain(
        symbol=sample_options_data['symbol'],
        expiry_date=sample_options_data['expiry'],
        underlying_price=sample_options_data['underlying_price'],
        calls=sample_options_data['calls'],
        puts=sample_options_data['puts']
    )


@pytest.fixture
def qt_app():
    """Qt application fixture for GUI tests"""
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    yield app
    
    # Clean up
    app.processEvents()


@pytest.fixture
def mock_analysis_results():
    """Sample analysis results for testing"""
    return {
        'symbol': 'AAPL',
        'analysis_type': 'calendar_spread',
        'recommendation': 'BUY',
        'confidence': 0.75,
        'expected_profit': 250.0,
        'max_loss': 500.0,
        'probability_profit': 0.65,
        'days_to_expiry': 30,
        'iv_rank': 0.45,
        'earnings_effect': 'neutral',
        'volatility_metrics': {
            'current_iv': 0.28,
            'historical_vol': 0.25,
            'iv_percentile': 60
        },
        'greeks': {
            'delta': 0.15,
            'gamma': 0.02,
            'theta': -5.50,
            'vega': 15.25
        },
        'monte_carlo_results': {
            'simulations': 10000,
            'profit_probability': 0.65,
            'expected_return': 0.125,
            'var_95': -450.0,
            'max_drawdown': -500.0
        },
        'ml_predictions': {
            'model_confidence': 0.72,
            'predicted_direction': 'UP',
            'probability_up': 0.68,
            'feature_importance': {
                'iv_rank': 0.25,
                'momentum': 0.20,
                'earnings_proximity': 0.18
            }
        }
    }


# Performance testing fixtures
@pytest.fixture
def benchmark_data():
    """Large dataset for performance testing"""
    np.random.seed(42)
    size = 10000
    
    return {
        'prices': np.random.lognormal(mean=4.6, sigma=0.2, size=size),
        'volumes': np.random.randint(100000, 10000000, size=size),
        'timestamps': pd.date_range(start='2020-01-01', periods=size, freq='H')
    }


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gui: marks tests as GUI tests requiring Qt"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require external API access"
    )


# Skip conditions
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and conditions"""
    import os
    
    # Skip slow tests if requested
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip API tests if no API keys available
    if not os.getenv('ALPHA_VANTAGE_API_KEY'):
        skip_api = pytest.mark.skip(reason="No API keys available")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow running tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true", 
        default=False,
        help="Run integration tests"
    )