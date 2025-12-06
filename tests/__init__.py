"""
Professional Options Calculator - Test Suite
Comprehensive testing infrastructure for all application components
"""

import sys
import os
import pytest
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests/test_output.log')
    ]
)

# Test configuration
TEST_CONFIG = {
    'use_mock_data': True,
    'test_data_dir': 'tests/fixtures',
    'timeout_seconds': 30,
    'skip_slow_tests': False,
    'test_api_keys': {
        'alpha_vantage': 'TEST_KEY_AV',
        'finnhub': 'TEST_KEY_FINNHUB'
    }
}

# Test fixtures and utilities
__all__ = [
    'TEST_CONFIG',
    'project_root'
]