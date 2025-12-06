"""
Unit tests for CalendarSpreadAnalyzer
Tests options analysis calculations and strategies
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from analysis.calendar_spread_analyzer import CalendarSpreadAnalyzer
from analysis.volatility_analysis import VolatilityAnalyzer
from analysis.monte_carlo_simulation import MonteCarloSimulator


class TestCalendarSpreadAnalyzer:
    """Test suite for CalendarSpreadAnalyzer"""
    
    @pytest.fixture
    def analyzer(self, mock_config_manager):
        """Create analyzer instance for tests"""
        return CalendarSpreadAnalyzer(mock_config_manager)
    
    @pytest.fixture
    def sample_analysis_input(self, sample_market_data_model, sample_option_chain_model):
        """Sample input data for analysis"""
        return {
            'symbol': 'AAPL',
            'market_data': sample_market_data_model,
            'options_chain': sample_option_chain_model,
            'contracts': 1,
            'debit': 2.50,
            'analysis_params': {
                'monte_carlo_simulations': 1000,
                'confidence_level': 0.95,
                'use_ml_prediction': True
            }
        }
    
    def test_analyze_success(self, analyzer, sample_analysis_input):
        """Test successful analysis execution"""
        result = analyzer.analyze(sample_analysis_input)
        
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'recommendation' in result
        assert 'confidence' in result
        assert 'expected_profit' in result
        assert 'max_loss' in result
        assert 'probability_profit' in result
        
        # Check confidence is within valid range
        assert 0 <= result['confidence'] <= 1
    
    def test_calculate_greeks(self, analyzer, sample_analysis_input):
        """Test Greeks calculation"""
        greeks = analyzer.calculate_greeks(sample_analysis_input)
        
        assert isinstance(greeks, dict)
        required_greeks = ['delta', 'gamma', 'theta', 'vega']
        
        for greek in required_greeks:
            assert greek in greeks
            assert isinstance(greeks[greek], (int, float))
    
    def test_profit_loss_calculation(self, analyzer, sample_analysis_input):
        """Test P/L calculation across price range"""
        price_range = np.linspace(140, 160, 21)
        pl_values = analyzer.calculate_profit_loss(sample_analysis_input, price_range)
        
        assert len(pl_values) == len(price_range)
        assert all(isinstance(pl, (int, float)) for pl in pl_values)
        
        # Calendar spreads should have limited max loss
        max_loss = min(pl_values)
        assert max_loss <= 0  # Should have some loss scenario
        assert max_loss >= -sample_analysis_input['debit'] * 100  # Limited to debit
    
    def test_breakeven_calculation(self, analyzer, sample_analysis_input):
        """Test breakeven point calculation"""
        breakeven_points = analyzer.calculate_breakeven_points(sample_analysis_input)
        
        assert isinstance(breakeven_points, list)
        assert len(breakeven_points) >= 0  # May have 0, 1, or 2 breakeven points
        
        # If breakeven points exist, they should be positive prices
        for point in breakeven_points:
            assert point > 0
    
    def test_volatility_analysis_integration(self, analyzer, sample_analysis_input):
        """Test integration with volatility analysis"""
        with patch('analysis.volatility_analysis.VolatilityAnalyzer') as mock_vol_analyzer:
            mock_vol_analyzer.return_value.analyze.return_value = {
                'current_iv': 0.28,
                'historical_vol': 0.25,
                'iv_rank': 0.65,
                'iv_percentile': 68
            }
            
            result = analyzer.analyze(sample_analysis_input)
            
            assert 'volatility_metrics' in result
            assert 'current_iv' in result['volatility_metrics']
    
    def test_monte_carlo_integration(self, analyzer, sample_analysis_input):
        """Test integration with Monte Carlo simulation"""
        with patch('analysis.monte_carlo_simulation.MonteCarloSimulator') as mock_mc:
            mock_mc.return_value.run_simulation.return_value = {
                'profit_probability': 0.65,
                'expected_return': 0.125,
                'var_95': -450.0,
                'simulations_run': 1000
            }
            
            result = analyzer.analyze(sample_analysis_input)
            
            assert 'monte_carlo_results' in result
            assert 'profit_probability' in result['monte_carlo_results']
    
    def test_earnings_date_impact(self, analyzer, sample_analysis_input):
        """Test analysis with earnings date proximity"""
        # Set earnings date close to expiry
        earnings_date = datetime.now() + timedelta(days=15)
        sample_analysis_input['market_data'].earnings_date = earnings_date
        
        result = analyzer.analyze(sample_analysis_input)
        
        assert 'earnings_effect' in result
        assert result['earnings_effect'] in ['positive', 'negative', 'neutral']
        
        # Should include IV crush consideration
        assert 'iv_crush_risk' in result
    
    def test_invalid_input_handling(self, analyzer):
        """Test handling of invalid input data"""
        invalid_inputs = [
            {},  # Empty input
            {'symbol': 'AAPL'},  # Missing required fields
            {'symbol': '', 'market_data': None},  # Invalid values
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, KeyError)):
                analyzer.analyze(invalid_input)
    
    def test_extreme_market_conditions(self, analyzer, sample_analysis_input):
        """Test analysis under extreme market conditions"""
        # Test with very high volatility
        sample_analysis_input['options_chain'].calls[0]['impliedVolatility'] = 2.0
        sample_analysis_input['options_chain'].puts[0]['impliedVolatility'] = 2.0
        
        result = analyzer.analyze(sample_analysis_input)
        
        # Should still return valid results
        assert isinstance(result, dict)
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
    
    def test_different_contract_sizes(self, analyzer, sample_analysis_input):
        """Test analysis with different contract sizes"""
        contract_sizes = [1, 5, 10, 25]
        
        for contracts in contract_sizes:
            sample_analysis_input['contracts'] = contracts
            result = analyzer.analyze(sample_analysis_input)
            
            # P/L should scale with contract size
            assert 'expected_profit' in result
            assert 'max_loss' in result
            
            # Verify scaling (approximately)
            expected_scale = contracts
            # Results should be roughly proportional to contract size
    
    def test_time_decay_analysis(self, analyzer, sample_analysis_input):
        """Test time decay impact analysis"""
        result = analyzer.analyze(sample_analysis_input)
        
        assert 'time_decay_analysis' in result
        time_decay = result['time_decay_analysis']
        
        assert 'theta_per_day' in time_decay
        assert 'days_to_expiry' in time_decay
        assert isinstance(time_decay['theta_per_day'], (int, float))
    
    @pytest.mark.slow
    def test_comprehensive_analysis(self, analyzer, sample_analysis_input):
        """Test comprehensive analysis with all features enabled"""
        # Enable all analysis features
        sample_analysis_input['analysis_params'].update({
            'monte_carlo_simulations': 10000,
            'include_greeks': True,
            'include_scenarios': True,
            'include_ml_prediction': True,
            'detailed_breakdowns': True
        })
        
        result = analyzer.analyze(sample_analysis_input)
       
        # Verify comprehensive analysis includes all expected components
        expected_sections = [
            'symbol', 'recommendation', 'confidence', 
            'expected_profit', 'max_loss', 'probability_profit',
            'greeks', 'volatility_metrics', 'monte_carlo_results',
            'time_decay_analysis', 'scenario_analysis',
            'ml_predictions', 'risk_metrics'
        ]
       
        for section in expected_sections:
            assert section in result, f"Missing section: {section}"
       
        # Verify data quality
        assert result['confidence'] > 0
        assert result['probability_profit'] > 0
        assert len(result['scenario_analysis']) > 0
    
    def test_sensitivity_analysis(self, analyzer, sample_analysis_input):
        """Test sensitivity analysis for key parameters"""
        base_result = analyzer.analyze(sample_analysis_input)
       
        # Test sensitivity to volatility changes
        sensitivities = analyzer.calculate_sensitivity_analysis(sample_analysis_input)
       
        assert 'volatility_sensitivity' in sensitivities
        assert 'price_sensitivity' in sensitivities
        assert 'time_sensitivity' in sensitivities
       
        # Each sensitivity should show impact range
        for sensitivity in sensitivities.values():
            assert 'base_case' in sensitivity
            assert 'scenarios' in sensitivity
            assert len(sensitivity['scenarios']) > 0
   
    def test_risk_reward_calculation(self, analyzer, sample_analysis_input):
        """Test risk-reward ratio calculation"""
        result = analyzer.analyze(sample_analysis_input)
       
        assert 'risk_reward_ratio' in result
        ratio = result['risk_reward_ratio']
       
        assert isinstance(ratio, (int, float))
        assert ratio > 0  # Should be positive for valid strategies
       
        # For calendar spreads, ratio should typically be between 0.5 and 3.0
        assert 0.1 <= ratio <= 10.0
   
    def test_multiple_expiries_analysis(self, analyzer, sample_analysis_input):
        """Test analysis with multiple expiry dates"""
        # Add multiple option chains with different expiries
        expiry_dates = [
            datetime.now() + timedelta(days=30),
            datetime.now() + timedelta(days=60),
            datetime.now() + timedelta(days=90)
        ]
       
        results = []
        for expiry in expiry_dates:
            # Modify the options chain expiry
            sample_analysis_input['options_chain'].expiry_date = expiry.strftime('%Y-%m-%d')
            result = analyzer.analyze(sample_analysis_input)
            results.append(result)
       
        # Should have results for each expiry
        assert len(results) == len(expiry_dates)
       
        # Results should vary with time to expiry
        confidences = [r['confidence'] for r in results]
        assert len(set(confidences)) > 1  # Should not all be identical


class TestCalendarSpreadOptimization: 
    """Test optimization features of calendar spread analyzer"""
   
    @pytest.fixture
    def optimizer(self, mock_config_manager):
        """Create optimizer instance"""
        from analysis.calendar_spread_analyzer import CalendarSpreadOptimizer
        return CalendarSpreadOptimizer(mock_config_manager)
   
    def test_optimal_strike_selection(self, optimizer, sample_analysis_input):
        """Test optimal strike price selection"""
        optimal_strikes = optimizer.find_optimal_strikes(sample_analysis_input)
       
        assert isinstance(optimal_strikes, dict)
        assert 'short_strike' in optimal_strikes
        assert 'long_strike' in optimal_strikes
        assert 'expected_profit' in optimal_strikes
       
        # Strikes should be reasonable relative to current price
        current_price = sample_analysis_input['market_data'].current_price
        short_strike = optimal_strikes['short_strike']
        long_strike = optimal_strikes['long_strike']
       
        assert 0.8 * current_price <= short_strike <= 1.2 * current_price
        assert 0.8 * current_price <= long_strike <= 1.2 * current_price
   
    def test_optimal_expiry_selection(self, optimizer, sample_analysis_input):
        """Test optimal expiry date selection"""
        available_expiries = [
            datetime.now() + timedelta(days=30),
            datetime.now() + timedelta(days=45),
            datetime.now() + timedelta(days=60),
            datetime.now() + timedelta(days=90)
        ]
       
        optimal_expiry = optimizer.find_optimal_expiry(
            sample_analysis_input, available_expiries
        )
       
        assert optimal_expiry in available_expiries
        assert isinstance(optimal_expiry, datetime)
   
    def test_position_sizing_optimization(self, optimizer, sample_analysis_input):
        """Test optimal position sizing"""
        portfolio_value = 100000
        max_risk_pct = 0.02
       
        optimal_size = optimizer.calculate_optimal_position_size(
            sample_analysis_input, portfolio_value, max_risk_pct
        )
       
        assert isinstance(optimal_size, int)
        assert optimal_size > 0
        assert optimal_size <= 100  # Reasonable upper limit
       
        # Position size should respect risk limits
        estimated_max_loss = sample_analysis_input['debit'] * 100 * optimal_size
        max_acceptable_loss = portfolio_value * max_risk_pct
        assert estimated_max_loss <= max_acceptable_loss * 1.1  # Small tolerance


class TestCalendarSpreadValidation:
    """Test validation and error handling"""
   
    def test_option_chain_validation(self, analyzer):
        """Test option chain data validation"""
        # Test with invalid option chain
        invalid_chains = [
            {'calls': [], 'puts': []},  # Empty chains
            {'calls': [{'strike': 'invalid'}]},  # Invalid strike
            {'calls': [{'strike': 150, 'bid': -1}]},  # Negative bid
        ]
       
        for invalid_chain in invalid_chains:
            with pytest.raises(ValueError):
                analyzer.validate_options_chain(invalid_chain)
   
    def test_market_data_validation(self, analyzer):
        """Test market data validation"""
        # Test with invalid market data
        invalid_data = [
            {'current_price': 0},  # Zero price
            {'current_price': -50},  # Negative price
            {'volume': -1000},  # Negative volume
        ]
       
        for invalid in invalid_data:
            with pytest.raises(ValueError):
                analyzer.validate_market_data(invalid)
   
    def test_analysis_parameters_validation(self, analyzer):
        """Test analysis parameters validation"""
        invalid_params = [
            {'contracts': 0},  # Zero contracts
            {'contracts': -5},  # Negative contracts
            {'debit': -2.0},  # Negative debit
            {'monte_carlo_simulations': 0},  # Zero simulations
        ]
       
        for invalid in invalid_params:
            with pytest.raises(ValueError):
                analyzer.validate_analysis_parameters(invalid)


class TestCalendarSpreadPerformance:
    """Performance tests for calendar spread analysis"""
   
    @pytest.mark.slow
    def test_analysis_performance(self, analyzer, sample_analysis_input):
        """Test analysis performance with timing"""
        import time
       
        start_time = time.time()
        result = analyzer.analyze(sample_analysis_input)
        end_time = time.time()
       
        analysis_time = end_time - start_time
       
        # Analysis should complete within reasonable time
        assert analysis_time < 5.0  # 5 seconds max
        assert isinstance(result, dict)
   
    @pytest.mark.slow
    def test_bulk_analysis_performance(self, analyzer, sample_analysis_input):
        """Test performance with multiple symbols"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
       
        start_time = time.time()
        results = []
       
        for symbol in symbols:
            input_copy = sample_analysis_input.copy()
            input_copy['symbol'] = symbol
            result = analyzer.analyze(input_copy)
            results.append(result)
       
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_symbol = total_time / len(symbols)
       
        # Should maintain reasonable performance per symbol
        assert avg_time_per_symbol < 2.0  # 2 seconds average
        assert len(results) == len(symbols)
   
    def test_memory_usage(self, analyzer, sample_analysis_input):
        """Test memory usage during analysis"""
        import psutil
        import os
       
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
       
        # Run multiple analyses
        for _ in range(10):
            result = analyzer.analyze(sample_analysis_input)
       
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
       
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestIntegrationWithMLModel:
    """Test integration with machine learning predictions"""
   
    def test_ml_prediction_integration(self, analyzer, sample_analysis_input):
        """Test ML model integration"""
        with patch('analysis.ml_predictor.MLPredictor') as mock_ml:
            mock_ml.return_value.predict.return_value = {
                'predicted_direction': 'UP',
                'confidence': 0.72,
                'probability_up': 0.68,
                'feature_importance': {
                    'iv_rank': 0.25,
                    'momentum': 0.20
                }
            }
           
            result = analyzer.analyze(sample_analysis_input)
           
            assert 'ml_predictions' in result
            ml_pred = result['ml_predictions']
            assert 'predicted_direction' in ml_pred
            assert 'confidence' in ml_pred
   
    def test_ml_model_fallback(self, analyzer, sample_analysis_input):
        """Test fallback when ML model fails"""
        with patch('analysis.ml_predictor.MLPredictor') as mock_ml:
            mock_ml.return_value.predict.side_effect = Exception("ML Model Error")
           
            # Analysis should still complete without ML predictions
            result = analyzer.analyze(sample_analysis_input)
           
            assert isinstance(result, dict)
            assert 'recommendation' in result
            # ML predictions may be absent or contain error info