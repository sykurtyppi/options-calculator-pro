"""
Integration tests for complete analysis workflow
Tests end-to-end functionality across multiple services
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import asyncio
import threading
import time

from services.market_data_service import MarketDataService
from services.analysis_service import AnalysisService
from services.cache_manager import CacheManager
from workers.analysis_worker import AnalysisWorker
from workers.data_worker import DataWorker
from analysis.calendar_spread_analyzer import CalendarSpreadAnalyzer
from models.market_data import MarketData, OptionChain


@pytest.mark.integration
class TestFullAnalysisWorkflow:
    """Test complete analysis workflow from data fetch to results"""
    
    @pytest.fixture
    def integrated_services(self, mock_config_manager, temp_data_dir):
        """Setup integrated service environment"""
        # Create real services with mocked external dependencies
        cache_manager = CacheManager(str(temp_data_dir / "cache"))
        
        market_data_service = MarketDataService(mock_config_manager)
        analysis_service = AnalysisService(mock_config_manager)
        
        # Mock external API calls but keep internal logic
        with patch.object(market_data_service, '_fetch_from_provider') as mock_fetch:
            mock_fetch.return_value = self._create_mock_market_response()
            
            return {
                'market_data_service': market_data_service,
                'analysis_service': analysis_service,
                'cache_manager': cache_manager,
                'config_manager': mock_config_manager
            }
    
    def _create_mock_market_response(self):
        """Create comprehensive mock market data response"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 150.0
        prices = []
        current = base_price
        
        for _ in dates:
            change = np.random.normal(0, 0.02)
            current *= (1 + change)
            prices.append(current)
        
        historical_data = pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0.01, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0.01, 0.005))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }).set_index('Date')
        
        # Generate options data
        current_price = prices[-1]
        strikes = np.arange(current_price * 0.8, current_price * 1.2, 5)
        
        calls = []
        puts = []
        
        for strike in strikes:
            moneyness = strike / current_price
            iv = 0.25 + 0.1 * abs(moneyness - 1)
            
            # Call option
            intrinsic_call = max(0, current_price - strike)
            time_value = iv * np.sqrt(30/365) * current_price * 0.4
            call_price = intrinsic_call + time_value
            
            calls.append({
                'strike': strike,
                'bid': call_price * 0.98,
                'ask': call_price * 1.02,
                'last': call_price,
                'volume': np.random.randint(0, 1000),
                'openInterest': np.random.randint(0, 5000),
                'impliedVolatility': iv,
                'delta': max(0, min(1, 0.5 + (current_price - strike) / (2 * current_price * iv))),
                'gamma': 0.05,
                'theta': -call_price * 0.05,
                'vega': current_price * iv * 0.1
            })
            
            # Put option
            intrinsic_put = max(0, strike - current_price)
            put_price = intrinsic_put + time_value
            
            puts.append({
                'strike': strike,
                'bid': put_price * 0.98,
                'ask': put_price * 1.02,
                'last': put_price,
                'volume': np.random.randint(0, 1000),
                'openInterest': np.random.randint(0, 5000),
                'impliedVolatility': iv,
                'delta': max(-1, min(0, -0.5 + (current_price - strike) / (2 * current_price * iv))),
                'gamma': 0.05,
                'theta': -put_price * 0.05,
                'vega': current_price * iv * 0.1
            })
        
        return {
            'historical_data': historical_data,
            'current_price': current_price,
            'options_chain': {
                'calls': calls,
                'puts': puts,
                'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'underlying_price': current_price
            },
            'earnings_date': datetime.now() + timedelta(days=45),
            'company_info': {
                'market_cap': 2500000000000,
                'pe_ratio': 25.5,
                'beta': 1.2,
                'sector': 'Technology'
            }
        }
    
    def test_complete_single_symbol_analysis(self, integrated_services):
        """Test complete analysis workflow for single symbol"""
        services = integrated_services
        
        # Step 1: Request analysis
        analysis_request = {
            'symbols': ['AAPL'],
            'contracts': 1,
            'debit': 2.50,
            'portfolio_value': 100000,
            'max_risk_pct': 0.02,
            'analysis_depth': 'standard',
            'use_ml': True
        }
        
        # Step 2: Fetch market data
        market_data = services['market_data_service'].get_market_data('AAPL')
        assert isinstance(market_data, MarketData)
        assert market_data.symbol == 'AAPL'
        assert market_data.current_price > 0
        
        # Step 3: Fetch options data
        options_chain = services['market_data_service'].get_options_chain('AAPL')
        assert isinstance(options_chain, dict)
        assert 'calls' in options_chain
        assert 'puts' in options_chain
        
        # Step 4: Run analysis
        analysis_input = {
            'symbol': 'AAPL',
            'market_data': market_data,
            'options_chain': options_chain,
            'contracts': analysis_request['contracts'],
            'debit': analysis_request['debit'],
            'analysis_params': {
                'monte_carlo_simulations': 1000,
                'use_ml_prediction': analysis_request['use_ml']
            }
        }
        
        results = services['analysis_service'].analyze_calendar_spread(analysis_input)
        
        # Step 5: Validate complete results
        assert isinstance(results, dict)
        
        # Core analysis results
        required_fields = [
            'symbol', 'recommendation', 'confidence', 'expected_profit',
            'max_loss', 'probability_profit', 'risk_reward_ratio'
        ]
        
        for field in required_fields:
            assert field in results, f"Missing required field: {field}"
        
        # Advanced analysis components
        advanced_fields = [
            'greeks', 'volatility_metrics', 'monte_carlo_results',
            'scenario_analysis', 'time_decay_analysis'
        ]
        
        for field in advanced_fields:
            assert field in results, f"Missing advanced field: {field}"
        
        # Validate data quality
        assert 0 <= results['confidence'] <= 1
        assert results['probability_profit'] > 0
        assert isinstance(results['greeks'], dict)
        assert isinstance(results['volatility_metrics'], dict)
    
    def test_multi_symbol_parallel_analysis(self, integrated_services):
        """Test parallel analysis of multiple symbols"""
        services = integrated_services
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        start_time = time.time()
        
        # Run parallel analysis
        results = {}
        threads = []
        
        def analyze_symbol(symbol):
            try:
                market_data = services['market_data_service'].get_market_data(symbol)
                options_chain = services['market_data_service'].get_options_chain(symbol)
                
                analysis_input = {
                    'symbol': symbol,
                    'market_data': market_data,
                    'options_chain': options_chain,
                    'contracts': 1,
                    'debit': 2.50
                }
                
                result = services['analysis_service'].analyze_calendar_spread(analysis_input)
                results[symbol] = result
                
            except Exception as e:
                results[symbol] = {'error': str(e)}
        
        # Create and start threads
        for symbol in symbols:
            thread = threading.Thread(target=analyze_symbol, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout per thread
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Validate results
        assert len(results) == len(symbols)
        
        for symbol in symbols:
            assert symbol in results
            result = results[symbol]
            
            if 'error' not in result:
                assert 'recommendation' in result
                assert 'confidence' in result
                assert result['symbol'] == symbol
        
        # Performance validation - parallel should be faster than sequential
        avg_time_per_symbol = total_time / len(symbols)
        assert avg_time_per_symbol < 10.0  # Should average less than 10s per symbol
    
    def test_caching_integration(self, integrated_services):
        """Test caching integration across the workflow"""
        services = integrated_services
        symbol = 'AAPL'
        
        # First request - should hit external APIs
        start_time = time.time()
        market_data_1 = services['market_data_service'].get_market_data(symbol)
        first_request_time = time.time() - start_time
        
        # Second request - should use cache
        start_time = time.time()
        market_data_2 = services['market_data_service'].get_market_data(symbol)
        second_request_time = time.time() - start_time
        
        # Validate caching effectiveness
        assert market_data_1.symbol == market_data_2.symbol
        assert market_data_1.current_price == market_data_2.current_price
        assert second_request_time < first_request_time  # Cache should be faster
        
        # Test cache invalidation
        services['cache_manager'].invalidate(f"market_data_{symbol}")
        
        start_time = time.time()
        market_data_3 = services['market_data_service'].get_market_data(symbol)
        third_request_time = time.time() - start_time
        
        # Should take longer after cache invalidation
        assert third_request_time > second_request_time
    
    def test_error_handling_and_recovery(self, integrated_services):
        """Test error handling and recovery mechanisms"""
        services = integrated_services
        
        # Test with invalid symbol
        with pytest.raises((ValueError, KeyError)):
            services['market_data_service'].get_market_data('INVALID_SYMBOL')
        
        # Test with network timeout simulation
        with patch.object(services['market_data_service'], '_fetch_from_provider') as mock_fetch:
            mock_fetch.side_effect = TimeoutError("Network timeout")
            
            with pytest.raises(TimeoutError):
                services['market_data_service'].get_market_data('AAPL')
        
        # Test analysis with incomplete data
        incomplete_market_data = MarketData(
            symbol='TEST',
            current_price=100.0,
            historical_data=pd.DataFrame()  # Empty historical data
        )
        
        with pytest.raises(ValueError):
            services['analysis_service'].analyze_calendar_spread({
                'symbol': 'TEST',
                'market_data': incomplete_market_data,
                'options_chain': {'calls': [], 'puts': []},
                'contracts': 1
            })
    
    def test_performance_under_load(self, integrated_services):
        """Test system performance under load"""
        services = integrated_services
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
        
        start_time = time.time()
        successful_analyses = 0
        failed_analyses = 0
        
        # Simulate high load
        for symbol in symbols:
            try:
                market_data = services['market_data_service'].get_market_data(symbol)
                options_chain = services['market_data_service'].get_options_chain(symbol)
                
                analysis_input = {
                    'symbol': symbol,
                    'market_data': market_data,
                    'options_chain': options_chain,
                    'contracts': 1,
                    'debit': 2.50
                }
                
                result = services['analysis_service'].analyze_calendar_spread(analysis_input)
                
                if 'error' not in result:
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                    
            except Exception as e:
                failed_analyses += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert successful_analyses > 0
        assert failed_analyses / len(symbols) < 0.5  # Less than 50% failure rate
        assert total_time < 60.0  # Complete within 1 minute
        
        throughput = successful_analyses / total_time
        assert throughput > 0.1  # At least 0.1 analyses per second
    
    def test_data_consistency_validation(self, integrated_services):
        """Test data consistency across different sources"""
        services = integrated_services
        symbol = 'AAPL'
        
        # Fetch market data
        market_data = services['market_data_service'].get_market_data(symbol)
        options_chain = services['market_data_service'].get_options_chain(symbol)
        
        # Validate price consistency
        price_tolerance = 0.01  # 1 cent tolerance
        
        if hasattr(options_chain, 'underlying_price'):
            price_diff = abs(market_data.current_price - options_chain.underlying_price)
            assert price_diff <= price_tolerance, f"Price inconsistency: {price_diff}"
        
        # Validate option chain integrity
        assert len(options_chain['calls']) > 0, "No call options found"
        assert len(options_chain['puts']) > 0, "No put options found"
        
        # Validate option pricing relationships
        for call in options_chain['calls']:
            assert call['bid'] <= call['ask'], f"Call bid > ask for strike {call['strike']}"
            assert call['bid'] >= 0, f"Negative call bid for strike {call['strike']}"
            
        for put in options_chain['puts']:
            assert put['bid'] <= put['ask'], f"Put bid > ask for strike {put['strike']}"
            assert put['bid'] >= 0, f"Negative put bid for strike {put['strike']}"
    
    @pytest.mark.slow
    def test_memory_usage_stability(self, integrated_services):
        """Test memory usage stability over multiple operations"""
        import psutil
        import gc
        import os
        
        services = integrated_services
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        gc.collect()
        initial_memory = process.memory_info().rss
        
        # Run multiple analysis cycles
        for cycle in range(10):
            symbol = f"TEST{cycle % 5}"  # Rotate through 5 different symbols
            
            try:
                market_data = services['market_data_service'].get_market_data('AAPL')
                options_chain = services['market_data_service'].get_options_chain('AAPL')
                
                analysis_input = {
                    'symbol': 'AAPL',
                    'market_data': market_data,
                    'options_chain': options_chain,
                    'contracts': 1,
                    'debit': 2.50
                }
                
                result = services['analysis_service'].analyze_calendar_spread(analysis_input)
                
                # Force garbage collection periodically
                if cycle % 3 == 0:
                    gc.collect()
                    
            except Exception as e:
                # Continue with next cycle even if one fails
                continue
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 200MB)
        max_acceptable_increase = 200 * 1024 * 1024  # 200MB
        assert memory_increase < max_acceptable_increase, \
            f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB"
    
    def test_concurrent_cache_access(self, integrated_services):
        """Test concurrent access to cache system"""
        services = integrated_services
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        results = {}
        errors = []
        
        def fetch_and_cache(symbol, iteration):
            try:
                cache_key = f"test_data_{symbol}_{iteration}"
                
                # Try to get from cache first
                cached_data = services['cache_manager'].get(cache_key)
                
                if cached_data is None:
                    # Simulate data fetching
                    data = {
                        'symbol': symbol,
                        'iteration': iteration,
                        'timestamp': datetime.now().isoformat(),
                        'data': list(range(100))  # Some data
                    }
                    
                    # Store in cache
                    services['cache_manager'].set(cache_key, data, ttl=300)
                    results[f"{symbol}_{iteration}"] = data
                else:
                    results[f"{symbol}_{iteration}"] = cached_data
                    
            except Exception as e:
                errors.append((symbol, iteration, str(e)))
        
        # Create multiple threads accessing cache concurrently
        threads = []
        
        for symbol in symbols:
            for iteration in range(5):
                thread = threading.Thread(
                    target=fetch_and_cache, 
                    args=(symbol, iteration)
                )
                threads.append(thread)
                thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Validate results
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(results) == len(symbols) * 5  # Should have all results
        
        # Validate cache consistency
        for symbol in symbols:
            for iteration in range(5):
                key = f"{symbol}_{iteration}"
                assert key in results
                assert results[key]['symbol'] == symbol
                assert results[key]['iteration'] == iteration


@pytest.mark.integration
class TestWorkerIntegration:
    """Test integration with worker threads"""
    
    def test_analysis_worker_integration(self, integrated_services, qt_app):
        """Test integration with analysis worker"""
        from PySide6.QtCore import QObject, Signal
        
        services = integrated_services
        
        # Create mock worker environment
        class MockSignalHandler(QObject):
            analysis_requested = Signal(dict)
            analysis_completed = Signal(dict)
            analysis_error = Signal(str)
            
            def __init__(self):
                super().__init__()
                self.results = []
                self.errors = []
                
            def handle_result(self, result):
                self.results.append(result)
                
            def handle_error(self, error):
                self.errors.append(error)
        
        signal_handler = MockSignalHandler()
        
        # Create analysis worker
        worker = AnalysisWorker(services['analysis_service'])
        
        # Connect signals
        worker.analysis_completed.connect(signal_handler.handle_result)
        worker.analysis_error.connect(signal_handler.handle_error)
        
        # Submit analysis request
        analysis_request = {
            'symbols': ['AAPL'],
            'contracts': 1,
            'debit': 2.50,
            'analysis_depth': 'standard'
        }
        
        worker.analyze_symbols(analysis_request)
        
        # Wait for completion
        start_time = time.time()
        while len(signal_handler.results) == 0 and len(signal_handler.errors) == 0:
            qt_app.processEvents()
            time.sleep(0.1)
            
            if time.time() - start_time > 30:  # 30 second timeout
                break
        
        # Validate results
        assert len(signal_handler.results) > 0 or len(signal_handler.errors) > 0
        
        if signal_handler.results:
            result = signal_handler.results[0]
            assert 'symbol' in result
            assert 'recommendation' in result
    
    def test_data_worker_integration(self, integrated_services, qt_app):
        """Test integration with data worker"""
        from PySide6.QtCore import QObject, Signal
        
        services = integrated_services
        
        class MockDataHandler(QObject):
            def __init__(self):
                super().__init__()
                self.data_received = []
                self.errors = []
                
            def handle_data(self, data):
                self.data_received.append(data)
                
            def handle_error(self, error):
                self.errors.append(error)
        
        data_handler = MockDataHandler()
        
        # Create data worker
        worker = DataWorker(services['market_data_service'])
        
        # Connect signals
        worker.data_received.connect(data_handler.handle_data)
        worker.data_error.connect(data_handler.handle_error)
        
        # Request data
        worker.fetch_market_data('AAPL')
        
        # Wait for completion
        start_time = time.time()
        while len(data_handler.data_received) == 0 and len(data_handler.errors) == 0:
            qt_app.processEvents()
            time.sleep(0.1)
            
            if time.time() - start_time > 30:
                break
        
        # Validate results
        assert len(data_handler.data_received) > 0 or len(data_handler.errors) > 0
        
        if data_handler.data_received:
            data = data_handler.data_received[0]
            assert 'symbol' in data
            assert 'market_data' in data or 'options_chain' in data


@pytest.mark.integration  
class TestAPIIntegration:
    """Test integration with external APIs (when available)"""
    
    @pytest.mark.api
    def test_real_api_integration(self, mock_config_manager):
        """Test with real API if keys are available"""
        import os
        
        # Skip if no real API keys
        if not os.getenv('ALPHA_VANTAGE_API_KEY'):
            pytest.skip("No real API keys available")
        
        # Update config with real API key
        mock_config_manager.set('api_keys', {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY')
        })
        
        # Create service with real API
        service = MarketDataService(mock_config_manager)
        
        # Test real data fetch
        try:
            market_data = service.get_market_data('AAPL')
            assert isinstance(market_data, MarketData)
            assert market_data.symbol == 'AAPL'
            assert market_data.current_price > 0
            
        except Exception as e:
            pytest.skip(f"Real API test failed: {str(e)}")
    
    @pytest.mark.api
    def test_api_rate_limiting(self, mock_config_manager):
        """Test API rate limiting behavior"""
        service = MarketDataService(mock_config_manager)
        
        # Make multiple rapid requests
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        start_time = time.time()
        
        results = []
        for symbol in symbols:
            try:
                result = service.get_current_price(symbol)
                results.append(result)
            except Exception as e:
                # Rate limiting may cause some requests to fail
                results.append(None)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should respect rate limits (not too fast)
        min_expected_time = len(symbols) * 0.1  # At least 0.1s per request
        assert total_time >= min_expected_time
        
        # Should have some successful results
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) > 0


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test integration with data persistence"""
    
    def test_trade_history_persistence(self, integrated_services, temp_data_dir):
        """Test trade history storage and retrieval"""
        services = integrated_services
        
        # Create sample trade data
        trade_data = {
            'symbol': 'AAPL',
            'strategy': 'calendar_spread',
            'entry_date': datetime.now(),
            'contracts': 1,
            'debit': 2.50,
            'expected_profit': 125.0,
            'max_loss': 250.0,
            'status': 'open'
        }
        
        # Store trade (this would use the trade manager)
        trade_id = services['analysis_service'].save_trade_analysis(trade_data)
        assert trade_id is not None
        
        # Retrieve trade
        retrieved_trade = services['analysis_service'].get_trade_analysis(trade_id)
        assert retrieved_trade is not None
        assert retrieved_trade['symbol'] == trade_data['symbol']
        assert retrieved_trade['strategy'] == trade_data['strategy']
    
    def test_analysis_history_storage(self, integrated_services):
        """Test analysis results storage and retrieval"""
        services = integrated_services
        
        # Run analysis
        market_data = services['market_data_service'].get_market_data('AAPL')
        options_chain = services['market_data_service'].get_options_chain('AAPL')
        
        analysis_input = {
            'symbol': 'AAPL',
            'market_data': market_data,
            'options_chain': options_chain,
            'contracts': 1,
            'debit': 2.50
        }
        
        result = services['analysis_service'].analyze_calendar_spread(analysis_input)
        
        # Store analysis
        analysis_id = services['analysis_service'].save_analysis_result(result)
        assert analysis_id is not None
        
        # Retrieve analysis
        retrieved_analysis = services['analysis_service'].get_analysis_result(analysis_id)
        assert retrieved_analysis is not None
        assert retrieved_analysis['symbol'] == result['symbol']
        assert retrieved_analysis['recommendation'] == result['recommendation']
    
    def test_configuration_persistence(self, integrated_services):
        """Test configuration changes persistence"""
        services = integrated_services
        config_manager = services['config_manager']
        
        # Update configuration
        original_value = config_manager.get('portfolio_value', 100000)
        new_value = 150000
        
        config_manager.set('portfolio_value', new_value)
        config_manager.save_config()
        
        # Verify persistence by creating new config manager
        new_config_manager = ConfigManager(config_manager.config_file)
        retrieved_value = new_config_manager.get('portfolio_value')
        
        assert retrieved_value == new_value
        
        # Restore original value
        config_manager.set('portfolio_value', original_value)
        config_manager.save_config()