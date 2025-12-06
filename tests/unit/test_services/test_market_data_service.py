"""
Unit tests for MarketDataService
Tests data fetching, caching, and error handling
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from services.market_data_service import MarketDataService
from models.market_data import MarketData


class TestMarketDataService:
    """Test suite for MarketDataService"""
    
    def test_initialization(self, mock_config_manager):
        """Test service initialization"""
        service = MarketDataService(mock_config_manager)
        
        assert service.config_manager == mock_config_manager
        assert service.cache_manager is not None
        assert service.rate_limiter is not None
        assert len(service.providers) > 0
    
    def test_get_historical_data_success(self, mock_config_manager, sample_historical_data):
        """Test successful historical data retrieval"""
        service = MarketDataService(mock_config_manager)
        
        # Mock the provider response
        with patch.object(service.providers[0], 'get_historical_data', 
                         return_value=sample_historical_data):
            result = service.get_historical_data('AAPL', period='3mo')
            
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'Close' in result.columns
            assert 'Volume' in result.columns
    
    def test_get_historical_data_cached(self, mock_config_manager, sample_historical_data):
        """Test cached historical data retrieval"""
        service = MarketDataService(mock_config_manager)
        
        # First call - should cache
        with patch.object(service.providers[0], 'get_historical_data', 
                         return_value=sample_historical_data) as mock_provider:
            result1 = service.get_historical_data('AAPL', period='3mo')
            
            # Second call - should use cache
            result2 = service.get_historical_data('AAPL', period='3mo')
            
            # Provider should only be called once
            assert mock_provider.call_count == 1
            pd.testing.assert_frame_equal(result1, result2)
    
    def test_get_current_price_success(self, mock_config_manager):
        """Test successful current price retrieval"""
        service = MarketDataService(mock_config_manager)
        expected_price = 150.0
        
        with patch.object(service.providers[0], 'get_current_price', 
                         return_value=expected_price):
            result = service.get_current_price('AAPL')
            
            assert result == expected_price
            assert isinstance(result, float)
    
    def test_get_options_chain_success(self, mock_config_manager, sample_options_data):
        """Test successful options chain retrieval"""
        service = MarketDataService(mock_config_manager)
        
        with patch.object(service.providers[0], 'get_options_chain', 
                         return_value=sample_options_data):
            result = service.get_options_chain('AAPL')
            
            assert isinstance(result, dict)
            assert 'calls' in result
            assert 'puts' in result
            assert 'underlying_price' in result
            assert len(result['calls']) > 0
            assert len(result['puts']) > 0
    
    def test_provider_failover(self, mock_config_manager, sample_historical_data):
        """Test failover to backup data provider"""
        service = MarketDataService(mock_config_manager)
        
        # Mock first provider to fail
        service.providers[0].get_historical_data.side_effect = Exception("API Error")
        
        # Mock second provider to succeed
        with patch.object(service.providers[1], 'get_historical_data', 
                         return_value=sample_historical_data):
            result = service.get_historical_data('AAPL', period='3mo')
            
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
    
    def test_rate_limiting(self, mock_config_manager):
        """Test rate limiting functionality"""
        service = MarketDataService(mock_config_manager)
        
        # Mock rate limiter to simulate limiting
        service.rate_limiter.acquire.return_value = False
        
        with pytest.raises(Exception, match="Rate limit"):
            service.get_current_price('AAPL')
    
    def test_invalid_symbol_handling(self, mock_config_manager):
        """Test handling of invalid symbols"""
        service = MarketDataService(mock_config_manager)
        
        with patch.object(service.providers[0], 'get_current_price', 
                         side_effect=ValueError("Invalid symbol")):
            with pytest.raises(ValueError):
                service.get_current_price('INVALID')
    
    def test_network_error_handling(self, mock_config_manager):
        """Test network error handling and retries"""
        service = MarketDataService(mock_config_manager)
        
        # Mock network errors for all providers
        for provider in service.providers:
            provider.get_current_price.side_effect = ConnectionError("Network error")
        
        with pytest.raises(ConnectionError):
            service.get_current_price('AAPL')
    
    def test_get_earnings_date(self, mock_config_manager):
        """Test earnings date retrieval"""
        service = MarketDataService(mock_config_manager)
        expected_date = datetime.now() + timedelta(days=15)
        
        with patch.object(service.providers[0], 'get_earnings_date', 
                         return_value=expected_date):
            result = service.get_earnings_date('AAPL')
            
            assert isinstance(result, datetime)
            assert result == expected_date
    
    def test_get_market_data_model(self, mock_config_manager, 
                                 sample_historical_data, sample_options_data):
        """Test complete market data model creation"""
        service = MarketDataService(mock_config_manager)
        
        # Mock all required data
        with patch.object(service, 'get_historical_data', 
                         return_value=sample_historical_data), \
             patch.object(service, 'get_current_price', return_value=150.0), \
             patch.object(service, 'get_options_chain', 
                         return_value=sample_options_data), \
             patch.object(service, 'get_earnings_date', 
                         return_value=datetime.now() + timedelta(days=30)):
            
            result = service.get_market_data('AAPL')
            
            assert isinstance(result, MarketData)
            assert result.symbol == 'AAPL'
            assert result.current_price == 150.0
            assert not result.historical_data.empty
    
    @pytest.mark.slow
    def test_bulk_data_retrieval(self, mock_config_manager, sample_historical_data):
        """Test bulk data retrieval for multiple symbols"""
        service = MarketDataService(mock_config_manager)
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        with patch.object(service, 'get_historical_data', 
                         return_value=sample_historical_data):
            results = service.get_bulk_historical_data(symbols, period='1mo')
            
            assert isinstance(results, dict)
            assert len(results) == len(symbols)
            for symbol in symbols:
                assert symbol in results
                assert isinstance(results[symbol], pd.DataFrame)
    
    def test_cache_invalidation(self, mock_config_manager, sample_historical_data):
        """Test cache invalidation and refresh"""
        service = MarketDataService(mock_config_manager)
        
        with patch.object(service.providers[0], 'get_historical_data', 
                         return_value=sample_historical_data) as mock_provider:
            # First call
            service.get_historical_data('AAPL', period='3mo')
            
            # Invalidate cache
            service.invalidate_cache('AAPL')
            
            # Second call should hit provider again
            service.get_historical_data('AAPL', period='3mo')
            
            assert mock_provider.call_count == 2
    
    def test_concurrent_requests(self, mock_config_manager, sample_historical_data):
        """Test handling of concurrent requests"""
        import threading
        
        service = MarketDataService(mock_config_manager)
        results = []
        errors = []
        
        def fetch_data(symbol):
            try:
                with patch.object(service.providers[0], 'get_historical_data', 
                                 return_value=sample_historical_data):
                    result = service.get_historical_data(symbol, period='1mo')
                    results.append((symbol, result))
            except Exception as e:
                errors.append((symbol, e))
        
        # Create multiple threads
        threads = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        for symbol in symbols:
            thread = threading.Thread(target=fetch_data, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == len(symbols)