"""
Performance tests for analysis engine
Benchmarks and stress tests for analysis components
"""

import pytest
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import psutil
import os
import gc

from analysis.calendar_spread_analyzer import CalendarSpreadAnalyzer
from analysis.monte_carlo_simulation import MonteCarloSimulator
from analysis.volatility_analysis import VolatilityAnalyzer
from services.analysis_service import AnalysisService


@pytest.mark.slow
@pytest.mark.performance
class TestAnalysisPerformance:
    """Performance benchmarks for analysis components"""
    
    @pytest.fixture
    def performance_data(self):
        """Generate large dataset for performance testing"""
        np.random.seed(42)
        
        # Large historical dataset
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        prices = []
        current = 100.0
        
        for _ in dates:
            change = np.random.normal(0.001, 0.02)
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
        
        # Large options chain
        current_price = prices[-1]
        strikes = np.arange(current_price * 0.5, current_price * 1.5, 1)
        
        calls = []
        puts = []
        
        for strike in strikes:
            iv = 0.25 + 0.1 * np.random.random()
            
            calls.append({
                'strike': strike,
                'bid': max(0.01, np.random.lognormal(0, 0.5)),
                'ask': max(0.02, np.random.lognormal(0.1, 0.5)),
                'last': max(0.01, np.random.lognormal(0.05, 0.5)),
                'volume': np.random.randint(0, 10000),
                'openInterest': np.random.randint(0, 50000),
                'impliedVolatility': iv,
                'delta': np.random.uniform(0, 1),
                'gamma': np.random.uniform(0, 0.1),
                'theta': np.random.uniform(-10, 0),
                'vega': np.random.uniform(0, 50)
            })
            
            puts.append({
                'strike': strike,
                'bid': max(0.01, np.random.lognormal(0, 0.5)),
                'ask': max(0.02, np.random.lognormal(0.1, 0.5)),
                'last': max(0.01, np.random.lognormal(0.05, 0.5)),
                'volume': np.random.randint(0, 10000),
                'openInterest': np.random.randint(0, 50000),
                'impliedVolatility': iv,
                'delta': np.random.uniform(-1, 0),
                'gamma': np.random.uniform(0, 0.1),
                'theta': np.random.uniform(-10, 0),
                'vega': np.random.uniform(0, 50)
            })
        
        return {
            'historical_data': historical_data,
            'current_price': current_price,
            'options_chain': {
                'calls': calls,
                'puts': puts,
                'underlying_price': current_price
            }
        }
    
    def test_single_analysis_performance(self, mock_config_manager, performance_data):
        """Benchmark single analysis performance"""
        analyzer = CalendarSpreadAnalyzer(mock_config_manager)
        
        # Create analysis input
        from models.market_data import MarketData
        market_data = MarketData(
            symbol="PERF_TEST",
            current_price=performance_data['current_price'],
            historical_data=performance_data['historical_data']
        )
        
        analysis_input = {
            'symbol': 'PERF_TEST',
            'market_data': market_data,
            'options_chain': performance_data['options_chain'],
            'contracts': 1,
            'debit': 2.50,
            'analysis_params': {
                'monte_carlo_simulations': 10000,
                'confidence_level': 0.95
            }
        }
        
        # Benchmark analysis
        start_time = time.time()
        result = analyzer.analyze(analysis_input)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Performance assertions
        assert analysis_time < 30.0, f"Analysis took {analysis_time:.2f}s, expected < 30s"
        assert isinstance(result, dict)
        assert 'recommendation' in result
        
        print(f"Single analysis completed in {analysis_time:.2f}s")
        return analysis_time
    
    def test_monte_carlo_performance(self, mock_config_manager, performance_data):
        """Benchmark Monte Carlo simulation performance"""
        simulator = MonteCarloSimulator(mock_config_manager)
        
        # Test different simulation sizes
        simulation_sizes = [1000, 5000, 10000, 50000]
        results = {}
        
        for size in simulation_sizes:
            start_time = time.time()
            
            result = simulator.run_simulation(
                current_price=performance_data['current_price'],
                volatility=0.25,
                time_to_expiry=30,
                simulations=size,
                price_range=(0.8, 1.2)
            )
            
            end_time = time.time()
            simulation_time = end_time - start_time
            results[size] = simulation_time
            
            # Performance requirements
            max_time_per_1k = 0.5  # 0.5 seconds per 1000 simulations
            expected_max_time = (size / 1000) * max_time_per_1k
            
            assert simulation_time < expected_max_time, \
                f"{size} simulations took {simulation_time:.2f}s, expected < {expected_max_time:.2f}s"
            
            print(f"{size} Monte Carlo simulations: {simulation_time:.2f}s")
        
        # Performance should scale reasonably
        time_1k = results[1000]
        time_10k = results[10000]
        
        # 10x simulations should take less than 15x time (allowing for some overhead)
        assert time_10k < time_1k * 15, "Monte Carlo scaling is poor"
        
        return results
    
    def test_volatility_analysis_performance(self, mock_config_manager, performance_data):
        """Benchmark volatility analysis performance"""
        analyzer = VolatilityAnalyzer(mock_config_manager)
        
        historical_data = performance_data['historical_data']
        
        start_time = time.time()
        
        result = analyzer.analyze_volatility(
            historical_data=historical_data,
            current_price=performance_data['current_price'],
            options_chain=performance_data['options_chain']
        )
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Should complete quickly even with large dataset
        assert analysis_time < 5.0, f"Volatility analysis took {analysis_time:.2f}s, expected < 5s"
        assert isinstance(result, dict)
        assert 'historical_volatility' in result
        
        print(f"Volatility analysis completed in {analysis_time:.2f}s")
        return analysis_time
    
    def test_bulk_analysis_performance(self, mock_config_manager, performance_data):
        """Benchmark bulk analysis of multiple symbols"""
        service = AnalysisService(mock_config_manager)
        
        # Create multiple analysis inputs
        num_symbols = 10
        analysis_inputs = []
        
        for i in range(num_symbols):
            from models.market_data import MarketData
            market_data = MarketData(
                symbol=f"TEST{i}",
                current_price=performance_data['current_price'] * (0.9 + 0.2 * np.random.random()),
                historical_data=performance_data['historical_data'].copy()
            )
            
            analysis_inputs.append({
                'symbol': f'TEST{i}',
                'market_data': market_data,
                'options_chain': performance_data['options_chain'].copy(),
                'contracts': 1,
                'debit': 2.50
            })
        
        # Sequential analysis
        start_time = time.time()
        sequential_results = []
        
        for analysis_input in analysis_inputs:
            result = service.analyze_calendar_spread(analysis_input)
            sequential_time = time.time() - start_time
       
        # Parallel analysis
        start_time = time.time()
        parallel_results = []
       
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for analysis_input in analysis_inputs:
                future = executor.submit(service.analyze_calendar_spread, analysis_input)
                futures.append(future)
           
            for future in as_completed(futures):
                result = future.result()
                parallel_results.append(result)
       
        parallel_time = time.time() - start_time
       
        # Performance assertions
        assert len(sequential_results) == num_symbols
        assert len(parallel_results) == num_symbols
       
        # Parallel should be faster for multiple symbols
        speedup = sequential_time / parallel_time
        assert speedup > 1.5, f"Parallel speedup is only {speedup:.2f}x"
       
        print(f"Sequential: {sequential_time:.2f}s, Parallel: {parallel_time:.2f}s, Speedup: {speedup:.2f}x")
       
        return {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup
        }
    
    def test_memory_usage_analysis(self, mock_config_manager, performance_data):
        """Test memory usage during analysis"""
        analyzer = CalendarSpreadAnalyzer(mock_config_manager)
        process = psutil.Process(os.getpid())
       
       # Baseline memory
        gc.collect()
        initial_memory = process.memory_info().rss
       
        # Run multiple analyses
        for i in range(20):
            from models.market_data import MarketData
            market_data = MarketData(
                symbol=f"MEM_TEST{i}",
                current_price=performance_data['current_price'],
                historical_data=performance_data['historical_data']
            )
           
            analysis_input = {
                'symbol': f'MEM_TEST{i}',
                'market_data': market_data,
                'options_chain': performance_data['options_chain'],
                'contracts': 1,
                'debit': 2.50
            }
           
            result = analyzer.analyze(analysis_input)
           
            # Clean up explicitly
            del result
            del market_data
            del analysis_input
           
            if i % 5 == 0:
                gc.collect()
       
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
       
        # Memory increase should be reasonable
        max_acceptable_increase = 200 * 1024 * 1024  # 200MB
        assert memory_increase < max_acceptable_increase, \
            f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB"
       
        print(f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB over 20 analyses")
        return memory_increase
   
    def test_caching_performance_impact(self, mock_config_manager, performance_data):
        """Test performance impact of caching"""
        from services.cache_manager import CacheManager
        import tempfile
       
       # Test with caching
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(temp_dir)
           
            analyzer = CalendarSpreadAnalyzer(mock_config_manager)
            analyzer.cache_manager = cache_manager
           
            from models.market_data import MarketData
            market_data = MarketData(
                symbol="CACHE_TEST",
                current_price=performance_data['current_price'],
                historical_data=performance_data['historical_data']
            )
           
            analysis_input = {
                'symbol': 'CACHE_TEST',
                'market_data': market_data,
                'options_chain': performance_data['options_chain'],
                'contracts': 1,
                'debit': 2.50
            }
           
            # First run - populate cache
            start_time = time.time()
            result1 = analyzer.analyze(analysis_input)
            first_run_time = time.time() - start_time
           
            # Second run - use cache
            start_time = time.time()
            result2 = analyzer.analyze(analysis_input)
            second_run_time = time.time() - start_time
           
            # Cached run should be significantly faster
            speedup = first_run_time / second_run_time
            assert speedup > 2.0, f"Cache speedup is only {speedup:.2f}x"
           
            print(f"Cache speedup: {speedup:.2f}x ({first_run_time:.2f}s -> {second_run_time:.2f}s)")
           
            return speedup


@pytest.mark.slow
@pytest.mark.performance
class TestDataProcessingPerformance:
    """Performance tests for data processing components"""
   
    def test_large_dataset_processing(self, mock_config_manager):
        """Test processing of large datasets"""
        # Generate large dataset
        size = 100000
        np.random.seed(42)
       
        large_dataset = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=size, freq='min'),
            'price': np.random.lognormal(4.6, 0.2, size),
            'volume': np.random.randint(1000, 100000, size)
        })
       
        # Test various processing operations
        start_time = time.time()
       
        # Calculate returns
        returns = large_dataset['price'].pct_change()
       
        # Calculate rolling statistics
        rolling_mean = large_dataset['price'].rolling(window=100).mean()
        rolling_std = large_dataset['price'].rolling(window=100).std()
       
        # Calculate volatility
        volatility = returns.rolling(window=252).std() * np.sqrt(252)
       
        end_time = time.time()
        processing_time = end_time - start_time
       
        # Should process efficiently
        max_time = 10.0  # 10 seconds for 100k data points
        assert processing_time < max_time, \
            f"Large dataset processing took {processing_time:.2f}s, expected < {max_time}s"
       
        print(f"Processed {size} data points in {processing_time:.2f}s")
        return processing_time
   
    def test_options_chain_processing(self, performance_data):
        """Test processing of large options chains"""
        options_chain = performance_data['options_chain']
       
        start_time = time.time()
       
        # Process all options
        all_options = options_chain['calls'] + options_chain['puts']
       
        # Calculate statistics
        total_volume = sum(opt['volume'] for opt in all_options)
        avg_iv = np.mean([opt['impliedVolatility'] for opt in all_options])
       
        # Find ATM options
        current_price = performance_data['current_price']
        atm_strikes = [
            opt for opt in all_options 
            if abs(opt['strike'] - current_price) < current_price * 0.05
        ]
       
        # Calculate Greeks aggregations
        total_delta = sum(opt['delta'] for opt in all_options)
        total_gamma = sum(opt['gamma'] for opt in all_options)
       
        end_time = time.time()
        processing_time = end_time - start_time
       
        # Should process quickly
        assert processing_time < 1.0, \
            f"Options chain processing took {processing_time:.2f}s, expected < 1s"
       
        print(f"Processed {len(all_options)} options in {processing_time:.2f}s")
        return processing_time


@pytest.mark.slow
@pytest.mark.performance
class TestConcurrencyPerformance:
    """Performance tests for concurrent operations"""
   
    def test_concurrent_analysis_scaling(self, mock_config_manager, performance_data):
        """Test how analysis scales with concurrent workers"""
        service = AnalysisService(mock_config_manager)
       
        # Test with different worker counts
        worker_counts = [1, 2, 4, 8]
        results = {}
       
        # Create analysis inputs
        num_analyses = 20
        analysis_inputs = []
       
        for i in range(num_analyses):
            from models.market_data import MarketData
            market_data = MarketData(
                symbol=f"CONCURRENT_TEST{i}",
                current_price=performance_data['current_price'],
                historical_data=performance_data['historical_data'].copy()
            )
           
            analysis_inputs.append({
                'symbol': f'CONCURRENT_TEST{i}',
                'market_data': market_data,
                'options_chain': performance_data['options_chain'].copy(),
                'contracts': 1,
                'debit': 2.50
            })
       
        for worker_count in worker_counts:
            start_time = time.time()
            completed_analyses = []
           
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                for analysis_input in analysis_inputs:
                    future = executor.submit(service.analyze_calendar_spread, analysis_input)
                    futures.append(future)
               
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)
                        completed_analyses.append(result)
                    except Exception as e:
                        print(f"Analysis failed: {e}")
           
            end_time = time.time()
            total_time = end_time - start_time
            results[worker_count] = {
                'time': total_time,
                'completed': len(completed_analyses),
                'throughput': len(completed_analyses) / total_time
            }
           
            print(f"{worker_count} workers: {total_time:.2f}s, {len(completed_analyses)} completed")
       
        # Verify scaling behavior
        single_worker_time = results[1]['time']
       
        for worker_count in [2, 4]:
            if worker_count in results:
                parallel_time = results[worker_count]['time']
                speedup = single_worker_time / parallel_time
               
                # Should get some speedup (not necessarily linear due to GIL and overhead)
                min_expected_speedup = min(1.5, worker_count * 0.7)
                assert speedup >= min_expected_speedup, \
                    f"Speedup with {worker_count} workers is only {speedup:.2f}x"
       
        return results
   
    def test_memory_pressure_under_load(self, mock_config_manager, performance_data):
        """Test memory behavior under concurrent load"""
        process = psutil.Process(os.getpid())
        service = AnalysisService(mock_config_manager)
       
        # Monitor memory during concurrent operations
        initial_memory = process.memory_info().rss
        peak_memory = initial_memory
       
        def run_analysis_with_monitoring(analysis_input):
            nonlocal peak_memory
           
            result = service.analyze_calendar_spread(analysis_input)
           
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)
           
            return result
       
        # Run concurrent analyses
        num_concurrent = 10
        analysis_inputs = []
       
        for i in range(num_concurrent):
            from models.market_data import MarketData
            market_data = MarketData(
                symbol=f"MEMORY_TEST{i}",
                current_price=performance_data['current_price'],
                historical_data=performance_data['historical_data'].copy()
            )
           
            analysis_inputs.append({
                'symbol': f'MEMORY_TEST{i}',
                'market_data': market_data,
                'options_chain': performance_data['options_chain'].copy(),
                'contracts': 1,
                'debit': 2.50
            })
       
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for analysis_input in analysis_inputs:
                future = executor.submit(run_analysis_with_monitoring, analysis_input)
                futures.append(future)
           
            # Wait for completion
            for future in as_completed(futures):
                future.result()
       
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss
       
        memory_increase = final_memory - initial_memory
        peak_increase = peak_memory - initial_memory
       
        # Memory usage should be reasonable
        max_acceptable_peak = 500 * 1024 * 1024  # 500MB peak increase
        max_acceptable_final = 200 * 1024 * 1024  # 200MB final increase
       
        assert peak_increase < max_acceptable_peak, \
            f"Peak memory increased by {peak_increase / 1024 / 1024:.1f}MB"
        assert memory_increase < max_acceptable_final, \
            f"Final memory increased by {memory_increase / 1024 / 1024:.1f}MB"
       
        print(f"Memory - Peak increase: {peak_increase / 1024 / 1024:.1f}MB, "
              f"Final increase: {memory_increase / 1024 / 1024:.1f}MB")
       
        return {
            'peak_increase': peak_increase,
            'final_increase': memory_increase
        }


@pytest.mark.slow
@pytest.mark.performance  
class TestRealWorldScenarios:
    """Performance tests simulating real-world usage patterns"""
   
    def test_market_hours_simulation(self, mock_config_manager, performance_data):
        """Simulate market hours trading activity"""
        service = AnalysisService(mock_config_manager)
       
        # Simulate market hours (6.5 hours)
        market_duration = 390  # minutes
        analysis_interval = 5  # minutes between analyses
       
        num_analyses = market_duration // analysis_interval
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
       
        start_time = time.time()
        completed_analyses = 0
        failed_analyses = 0
       
        for minute in range(0, market_duration, analysis_interval):
            # Select random symbol
            symbol = symbols[minute % len(symbols)]
           
            try:
                from models.market_data import MarketData
               
                # Simulate price changes throughout day
                price_change = 1 + (np.random.random() - 0.5) * 0.02  # ±1% change
                current_price = performance_data['current_price'] * price_change
               
                market_data = MarketData(
                    symbol=symbol,
                    current_price=current_price,
                    historical_data=performance_data['historical_data']
                )
               
                analysis_input = {
                    'symbol': symbol,
                    'market_data': market_data,
                    'options_chain': performance_data['options_chain'],
                    'contracts': 1,
                    'debit': 2.50
                }
               
                result = service.analyze_calendar_spread(analysis_input)
                completed_analyses += 1
               
            except Exception as e:
                failed_analyses += 1
       
        end_time = time.time()
        total_time = end_time - start_time
       
        # Performance requirements for market hours simulation
        assert total_time < 300, f"Market simulation took {total_time:.1f}s, expected < 300s"
        assert failed_analyses / num_analyses < 0.1, f"Failure rate too high: {failed_analyses}/{num_analyses}"
       
        throughput = completed_analyses / total_time
        assert throughput > 0.2, f"Throughput too low: {throughput:.2f} analyses/second"
       
        print(f"Market simulation: {completed_analyses} analyses in {total_time:.1f}s "
              f"({throughput:.2f}/sec, {failed_analyses} failures)")
       
        return {
            'completed': completed_analyses,
            'failed': failed_analyses,
            'total_time': total_time,
            'throughput': throughput
        }
   
    def test_portfolio_monitoring_simulation(self, mock_config_manager, performance_data):
        """Simulate continuous portfolio monitoring"""
        service = AnalysisService(mock_config_manager)
       
        # Portfolio of 25 positions
        portfolio_symbols = [f"STOCK{i:02d}" for i in range(25)]
        monitoring_cycles = 10  # Check portfolio 10 times
       
        start_time = time.time()
        total_analyses = 0
       
        for cycle in range(monitoring_cycles):
            cycle_start = time.time()
           
            # Analyze each position in portfolio
            for symbol in portfolio_symbols:
                try:
                    from models.market_data import MarketData
                    market_data = MarketData(
                        symbol=symbol,
                        current_price=performance_data['current_price'] * (0.9 + 0.2 * np.random.random()),
                        historical_data=performance_data['historical_data']
                    )
                   
                    analysis_input = {
                        'symbol': symbol,
                        'market_data': market_data,
                        'options_chain': performance_data['options_chain'],
                        'contracts': 1,
                        'debit': 2.50
                    }
                   
                    result = service.analyze_calendar_spread(analysis_input)
                    total_analyses += 1
                   
                except Exception as e:
                    continue
           
            cycle_time = time.time() - cycle_start
           
            # Each portfolio scan should complete within reasonable time
            assert cycle_time < 120, f"Portfolio scan {cycle} took {cycle_time:.1f}s, expected < 120s"
       
        end_time = time.time()
        total_time = end_time - start_time
       
        # Overall performance requirements
        avg_cycle_time = total_time / monitoring_cycles
        assert avg_cycle_time < 90, f"Average cycle time {avg_cycle_time:.1f}s too high"
       
        print(f"Portfolio monitoring: {total_analyses} analyses in {total_time:.1f}s "
              f"({monitoring_cycles} cycles, {avg_cycle_time:.1f}s avg cycle)")
       
        return {
            'total_analyses': total_analyses,
            'cycles': monitoring_cycles,
            'total_time': total_time,
            'avg_cycle_time': avg_cycle_time
        }


def generate_performance_report(test_results):
    """Generate performance test report"""
    report = """
    PERFORMANCE TEST REPORT
    ======================
   
    Test Results Summary:
    """
   
    for test_name, results in test_results.items():
        report += f"\n{test_name}:\n"
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, float):
                    report += f"  {key}: {value:.3f}\n"
                else:
                    report += f"  {key}: {value}\n"
        else:
            report += f"  Result: {results}\n"
   
    return report


@pytest.mark.performance
def test_performance_regression(mock_config_manager, performance_data):
    """Comprehensive performance regression test"""
    # This test can be run to detect performance regressions
    # Store baseline results and compare against them
   
    analyzer = CalendarSpreadAnalyzer(mock_config_manager)
   
    from models.market_data import MarketData
    market_data = MarketData(
        symbol="REGRESSION_TEST",
        current_price=performance_data['current_price'],
        historical_data=performance_data['historical_data']
    )
   
    analysis_input = {
        'symbol': 'REGRESSION_TEST',
        'market_data': market_data,
        'options_chain': performance_data['options_chain'],
        'contracts': 1,
        'debit': 2.50,
        'analysis_params': {
            'monte_carlo_simulations': 5000
        }
    }
    
    # Run multiple times to get stable measurement
    times = []
    for _ in range(5):
        start_time = time.time()
        result = analyzer.analyze(analysis_input)
        end_time = time.time()
        times.append(end_time - start_time)
   
    avg_time = np.mean(times)
    std_time = np.std(times)
   
    # Define performance baseline (these would be updated based on target hardware)
    baseline_time = 15.0  # seconds
    baseline_std = 2.0   # seconds
   
    assert avg_time < baseline_time, \
        f"Performance regression: avg time {avg_time:.2f}s > baseline {baseline_time}s"
    assert std_time < baseline_std, \
        f"Performance inconsistency: std {std_time:.2f}s > baseline {baseline_std}s"
   
    print(f"Performance regression test: {avg_time:.2f}±{std_time:.2f}s (baseline: {baseline_time}s)")
   
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'baseline_time': baseline_time,
        'passed': True
    }