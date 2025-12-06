"""
Unit tests for MarketData and related models
Tests data models, validation, and serialization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from models.market_data import MarketData, OptionChain, OptionContract, Greeks


class TestMarketData:
    """Test suite for MarketData model"""
    
    def test_market_data_creation(self, sample_historical_data):
        """Test MarketData model creation"""
        market_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            historical_data=sample_historical_data,
            volume=5000000,
            avg_volume=3000000
        )
        
        assert market_data.symbol == "AAPL"
        assert market_data.current_price == 150.0
        assert isinstance(market_data.historical_data, pd.DataFrame)
        assert market_data.volume == 5000000
        assert market_data.avg_volume == 3000000
    
    def test_market_data_validation(self):
        """Test MarketData validation"""
        # Test invalid symbol
        with pytest.raises(ValueError):
            MarketData(symbol="", current_price=150.0)
        
        # Test negative price
        with pytest.raises(ValueError):
            MarketData(symbol="AAPL", current_price=-10.0)
        
        # Test negative volume
        with pytest.raises(ValueError):
            MarketData(symbol="AAPL", current_price=150.0, volume=-1000)
    
    def test_calculated_properties(self, sample_historical_data):
        """Test calculated properties of MarketData"""
        market_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            historical_data=sample_historical_data,
            volume=5000000,
            avg_volume=3000000
        )
        
        # Test volume ratio
        expected_volume_ratio = 5000000 / 3000000
        assert abs(market_data.volume_ratio - expected_volume_ratio) < 0.01
        
        # Test price range
        assert market_data.day_range is not None
        assert len(market_data.day_range) == 2  # [low, high]
        
        # Test returns calculation
        returns = market_data.calculate_returns(period=20)
        assert isinstance(returns, pd.Series)
        assert len(returns) <= len(sample_historical_data)
    
    def test_volatility_calculation(self, sample_historical_data):
        """Test volatility calculation methods"""
        market_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            historical_data=sample_historical_data
        )
        
        # Test historical volatility
        vol_30d = market_data.calculate_historical_volatility(30)
        assert isinstance(vol_30d, float)
        assert 0 < vol_30d < 5.0  # Reasonable volatility range
        
        # Test different periods
        vol_60d = market_data.calculate_historical_volatility(60)
        assert isinstance(vol_60d, float)
        assert vol_60d != vol_30d  # Should be different for different periods
    
    def test_support_resistance_levels(self, sample_historical_data):
        """Test support and resistance level calculation"""
        market_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            historical_data=sample_historical_data
        )
        
        levels = market_data.calculate_support_resistance_levels()
        
        assert isinstance(levels, dict)
        assert 'support_levels' in levels
        assert 'resistance_levels' in levels
        assert isinstance(levels['support_levels'], list)
        assert isinstance(levels['resistance_levels'], list)
    
    def test_serialization(self, sample_historical_data):
        """Test MarketData serialization and deserialization"""
        market_data = MarketData(
            symbol="AAPL",
            current_price=150.0,
            historical_data=sample_historical_data,
            volume=5000000,
            market_cap=2500000000000,
            pe_ratio=25.5
        )
        
        # Test to_dict
        data_dict = market_data.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == "AAPL"
        assert data_dict['current_price'] == 150.0
        
        # Test from_dict
        reconstructed = MarketData.from_dict(data_dict)
        assert reconstructed.symbol == market_data.symbol
        assert reconstructed.current_price == market_data.current_price
    
    def test_comparison_methods(self, sample_historical_data):
        """Test MarketData comparison methods"""
        market_data1 = MarketData(symbol="AAPL", current_price=150.0)
        market_data2 = MarketData(symbol="AAPL", current_price=155.0)
        market_data3 = MarketData(symbol="MSFT", current_price=150.0)
        
        # Test equality
        assert market_data1 == market_data1
        assert market_data1 != market_data2
        assert market_data1 != market_data3
        
        # Test comparison by price
        assert market_data1 < market_data2
        assert market_data2 > market_data1


class TestOptionContract:
    """Test suite for OptionContract model"""
    
    def test_option_contract_creation(self):
        """Test OptionContract creation"""
        contract = OptionContract(
            symbol="AAPL240315C00150000",
            strike=150.0,
            expiry=datetime(2024, 3, 15),
            option_type="call",
            bid=5.20,
            ask=5.40,
            last=5.30,
            volume=1500,
            open_interest=5000,
            implied_volatility=0.28
        )
        
        assert contract.symbol == "AAPL240315C00150000"
        assert contract.strike == 150.0
        assert contract.option_type == "call"
        assert contract.bid == 5.20
        assert contract.ask == 5.40
    
    def test_option_contract_validation(self):
        """Test OptionContract validation"""
        # Test invalid option type
        with pytest.raises(ValueError):
            OptionContract(
                symbol="AAPL240315X00150000",
                strike=150.0,
                expiry=datetime(2024, 3, 15),
                option_type="invalid",
                bid=5.20,
                ask=5.40
            )
        
        # Test negative strike
        with pytest.raises(ValueError):
            OptionContract(
                symbol="AAPL240315C00150000",
                strike=-150.0,
                expiry=datetime(2024, 3, 15),
                option_type="call"
            )
        
        # Test bid > ask
        with pytest.raises(ValueError):
            OptionContract(
                symbol="AAPL240315C00150000",
                strike=150.0,
                expiry=datetime(2024, 3, 15),
                option_type="call",
                bid=5.50,
                ask=5.30
            )
    
    def test_moneyness_calculation(self):
        """Test moneyness calculation"""
        current_price = 150.0
        
        # ITM call
        itm_call = OptionContract(
            symbol="AAPL240315C00140000",
            strike=140.0,
            expiry=datetime(2024, 3, 15),
            option_type="call"
        )
        assert itm_call.calculate_moneyness(current_price) > 1.0
        assert itm_call.is_itm(current_price) == True
        
        # OTM call
        otm_call = OptionContract(
            symbol="AAPL240315C00160000",
            strike=160.0,
            expiry=datetime(2024, 3, 15),
            option_type="call"
        )
        assert otm_call.calculate_moneyness(current_price) < 1.0
        assert otm_call.is_itm(current_price) == False
        
        # ATM call
        atm_call = OptionContract(
            symbol="AAPL240315C00150000",
            strike=150.0,
            expiry=datetime(2024, 3, 15),
            option_type="call"
        )
        assert abs(atm_call.calculate_moneyness(current_price) - 1.0) < 0.01
    
    def test_time_value_calculation(self):
        """Test time value calculation"""
        current_price = 150.0
        
        contract = OptionContract(
            symbol="AAPL240315C00140000",
            strike=140.0,
            expiry=datetime(2024, 3, 15),
            option_type="call",
            last=12.50
        )
        
        intrinsic_value = contract.calculate_intrinsic_value(current_price)
        time_value = contract.calculate_time_value(current_price)
        
        assert intrinsic_value == 10.0  # 150 - 140
        assert time_value == 2.50  # 12.50 - 10.0
        assert intrinsic_value + time_value == contract.last
    
    def test_days_to_expiry(self):
        """Test days to expiry calculation"""
        future_date = datetime.now() + timedelta(days=30)
        
        contract = OptionContract(
            symbol="AAPL240315C00150000",
            strike=150.0,
            expiry=future_date,
            option_type="call"
        )
        
        days = contract.days_to_expiry()
        assert 29 <= days <= 31  # Allow for some timing variance


class TestGreeks:
    """Test suite for Greeks model"""
    
    def test_greeks_creation(self):
        """Test Greeks model creation"""
        greeks = Greeks(
            delta=0.65,
            gamma=0.02,
            theta=-5.50,
            vega=15.25,
            rho=8.75
        )
        
        assert greeks.delta == 0.65
        assert greeks.gamma == 0.02
        assert greeks.theta == -5.50
        assert greeks.vega == 15.25
        assert greeks.rho == 8.75
    
    def test_greeks_validation(self):
        """Test Greeks validation"""
        # Test delta out of range for calls
        with pytest.raises(ValueError):
            Greeks(delta=1.5, gamma=0.02, theta=-5.50, vega=15.25)
        
        # Test negative gamma
        with pytest.raises(ValueError):
            Greeks(delta=0.5, gamma=-0.02, theta=-5.50, vega=15.25)
        
        # Test positive theta (unusual)
        with pytest.warns(UserWarning):
            Greeks(delta=0.5, gamma=0.02, theta=5.50, vega=15.25)
    
    def test_greeks_calculations(self):
        """Test Greeks calculation methods"""
        greeks = Greeks(
            delta=0.65,
            gamma=0.02,
            theta=-5.50,
            vega=15.25
        )
        
        # Test position Greeks for multiple contracts
        position_greeks = greeks.calculate_position_greeks(contracts=5)
        
        assert position_greeks['delta'] == 0.65 * 5
        assert position_greeks['gamma'] == 0.02 * 5
        assert position_greeks['theta'] == -5.50 * 5
        assert position_greeks['vega'] == 15.25 * 5


class TestOptionChain:
    """Test suite for OptionChain model"""
    
    def test_option_chain_creation(self, sample_options_data):
        """Test OptionChain creation"""
        chain = OptionChain(
            symbol=sample_options_data['symbol'],
            expiry_date=sample_options_data['expiry'],
            underlying_price=sample_options_data['underlying_price'],
            calls=sample_options_data['calls'],
            puts=sample_options_data['puts']
        )
        
        assert chain.symbol == sample_options_data['symbol']
        assert chain.underlying_price == sample_options_data['underlying_price']
        assert len(chain.calls) == len(sample_options_data['calls'])
        assert len(chain.puts) == len(sample_options_data['puts'])
    
    def test_strike_filtering(self, sample_options_data):
        """Test strike price filtering"""
        chain = OptionChain(
            symbol=sample_options_data['symbol'],
            expiry_date=sample_options_data['expiry'],
            underlying_price=sample_options_data['underlying_price'],
            calls=sample_options_data['calls'],
            puts=sample_options_data['puts']
        )
        
        # Test ITM calls
        itm_calls = chain.get_itm_calls()
        for call in itm_calls:
            assert call['strike'] < chain.underlying_price
        
        # Test OTM puts
        otm_puts = chain.get_otm_puts()
        for put in otm_puts:
            assert put['strike'] < chain.underlying_price
    
    def test_closest_strikes(self, sample_options_data):
        """Test finding closest strikes to current price"""
        chain = OptionChain(
            symbol=sample_options_data['symbol'],
            expiry_date=sample_options_data['expiry'],
            underlying_price=sample_options_data['underlying_price'],
            calls=sample_options_data['calls'],
            puts=sample_options_data['puts']
        )
        
        closest_call = chain.get_closest_call_strike()
        closest_put = chain.get_closest_put_strike()
        
        assert closest_call is not None
        assert closest_put is not None
        assert isinstance(closest_call['strike'], (int, float))
        assert isinstance(closest_put['strike'], (int, float))
    
    def test_implied_volatility_analysis(self, sample_options_data):
        """Test implied volatility analysis"""
        chain = OptionChain(
            symbol=sample_options_data['symbol'],
            expiry_date=sample_options_data['expiry'],
            underlying_price=sample_options_data['underlying_price'],
            calls=sample_options_data['calls'],
            puts=sample_options_data['puts']
        )
        
        iv_analysis = chain.analyze_implied_volatility()
        
        assert 'avg_call_iv' in iv_analysis
        assert 'avg_put_iv' in iv_analysis
        assert 'iv_skew' in iv_analysis
        assert isinstance(iv_analysis['avg_call_iv'], float)
        assert isinstance(iv_analysis['avg_put_iv'], float)
    
    def test_volume_analysis(self, sample_options_data):
        """Test volume and open interest analysis"""
        chain = OptionChain(
            symbol=sample_options_data['symbol'],
            expiry_date=sample_options_data['expiry'],
            underlying_price=sample_options_data['underlying_price'],
            calls=sample_options_data['calls'],
            puts=sample_options_data['puts']
        )
        
        volume_analysis = chain.analyze_volume()
        
        assert 'total_call_volume' in volume_analysis
        assert 'total_put_volume' in volume_analysis
        assert 'put_call_ratio' in volume_analysis
        assert 'max_pain' in volume_analysis
    
    def test_option_chain_serialization(self, sample_options_data):
        """Test OptionChain serialization"""
        chain = OptionChain(
            symbol=sample_options_data['symbol'],
            expiry_date=sample_options_data['expiry'],
            underlying_price=sample_options_data['underlying_price'],
            calls=sample_options_data['calls'],
            puts=sample_options_data['puts']
        )
        
        # Test to_dict
        chain_dict = chain.to_dict()
        assert isinstance(chain_dict, dict)
        assert 'symbol' in chain_dict
        assert 'calls' in chain_dict
        assert 'puts' in chain_dict
        
        # Test from_dict
        reconstructed = OptionChain.from_dict(chain_dict)
        assert reconstructed.symbol == chain.symbol
        assert len(reconstructed.calls) == len(chain.calls)
        assert len(reconstructed.puts) == len(chain.puts)


class TestModelIntegration:
    """Test integration between different models"""
    
    def test_market_data_option_chain_integration(self, sample_market_data_model, 
                                                 sample_option_chain_model):
        """Test integration between MarketData and OptionChain"""
        market_data = sample_market_data_model
        option_chain = sample_option_chain_model
        
        # Verify price consistency
        assert abs(market_data.current_price - option_chain.underlying_price) < 0.01
        
        # Test combined analysis
        combined_analysis = market_data.analyze_with_options(option_chain)
        
        assert 'volatility_comparison' in combined_analysis
        assert 'volume_analysis' in combined_analysis
        assert 'price_targets' in combined_analysis
    
    def test_model_validation_chain(self, sample_options_data):
        """Test validation across related models"""
        # Create option chain
        chain = OptionChain(
            symbol="AAPL",
            expiry_date="2024-03-15",
            underlying_price=150.0,
            calls=sample_options_data['calls'],
            puts=sample_options_data['puts']
        )
        
        # Validate that all option contracts are valid
        for call in chain.calls:
            contract = OptionContract(
                symbol=f"AAPL240315C{int(call['strike']*1000):08d}",
                strike=call['strike'],
                expiry=datetime(2024, 3, 15),
                option_type="call",
                bid=call['bid'],
                ask=call['ask'],
                implied_volatility=call['impliedVolatility']
            )
            assert contract.strike == call['strike']
            assert contract.bid <= contract.ask