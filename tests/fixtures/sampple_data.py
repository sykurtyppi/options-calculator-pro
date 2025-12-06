"""
Sample data fixtures for testing
Provides realistic market data for test scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List


def create_sample_price_data(symbol: str = "AAPL", 
                           start_date: str = "2024-01-01",
                           end_date: str = "2024-03-01",
                           initial_price: float = 150.0) -> pd.DataFrame:
    """Create realistic sample price data"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic price movements
    np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
    
    prices = []
    current = initial_price
    
    for i, date in enumerate(dates):
        # Add weekly pattern (lower volume on weekends)
        weekday_factor = 1.0 if date.weekday() < 5 else 0.3
        
        # Add some trending behavior
        trend = 0.0005 * np.sin(i / 30)  # Monthly cycle
        
        # Random walk with trend
        daily_return = np.random.normal(trend, 0.02 * weekday_factor)
        current *= (1 + daily_return)
        prices.append(current)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close
        daily_range = close * np.random.uniform(0.01, 0.05)
        
        open_price = close * (1 + np.random.normal(0, 0.005))
        high = max(open_price, close) + np.random.uniform(0, daily_range)
        low = min(open_price, close) - np.random.uniform(0, daily_range)
        
        # Volume with realistic patterns
        base_volume = 5000000
        volume_multiplier = np.random.lognormal(0, 0.5)
        volume = int(base_volume * volume_multiplier)
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df


def create_sample_options_chain(symbol: str = "AAPL",
                               current_price: float = 150.0,
                               expiry_days: int = 30,
                               strike_range: float = 0.2) -> Dict[str, Any]:
    """Create realistic sample options chain"""
    expiry_date = datetime.now() + timedelta(days=expiry_days)
    
    # Generate strikes around current price
    strike_min = current_price * (1 - strike_range)
    strike_max = current_price * (1 + strike_range)
    strikes = np.arange(
        round(strike_min / 5) * 5,  # Round to nearest $5
        round(strike_max / 5) * 5 + 5,
        5
    )
    
    calls = []
    puts = []
    
    # Volatility smile parameters
    base_iv = 0.25
    time_factor = np.sqrt(expiry_days / 365)
    
    for strike in strikes:
        moneyness = strike / current_price
        
        # Create volatility smile
        smile_factor = 0.1 * (moneyness - 1) ** 2
        iv = base_iv + smile_factor + np.random.normal(0, 0.02)
        iv = max(0.05, iv)  # Minimum 5% IV
        
        # Black-Scholes approximation for option prices
        d1 = (np.log(current_price / strike) + (0.05 + 0.5 * iv**2) * time_factor) / (iv * np.sqrt(time_factor))
        d2 = d1 - iv * np.sqrt(time_factor)
        
        from scipy.stats import norm
        
        # Call option
        call_price = (current_price * norm.cdf(d1) - 
                     strike * np.exp(-0.05 * time_factor) * norm.cdf(d2))
        call_price = max(0.01, call_price)  # Minimum $0.01
        
        # Greeks approximation
        call_delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (current_price * iv * np.sqrt(time_factor))
        call_theta = -(current_price * norm.pdf(d1) * iv / (2 * np.sqrt(time_factor)) +
                      0.05 * strike * np.exp(-0.05 * time_factor) * norm.cdf(d2))
        vega = current_price * norm.pdf(d1) * np.sqrt(time_factor)
        
        calls.append({
            'strike': float(strike),
            'bid': round(call_price * 0.98, 2),
            'ask': round(call_price * 1.02, 2),
            'last': round(call_price, 2),
            'volume': int(np.random.exponential(500)),
            'openInterest': int(np.random.exponential(2000)),
            'impliedVolatility': round(iv, 4),
            'delta': round(call_delta, 4),
            'gamma': round(gamma, 4),
            'theta': round(call_theta / 365, 4),  # Per day
            'vega': round(vega / 100, 4),  # Per 1% vol change
            'rho': round(strike * time_factor * np.exp(-0.05 * time_factor) * norm.cdf(d2) / 100, 4)
        })
        
        # Put option (put-call parity)
        put_price = call_price - current_price + strike * np.exp(-0.05 * time_factor)
        put_price = max(0.01, put_price)
        
        put_delta = call_delta - 1
        put_theta = call_theta + 0.05 * strike * np.exp(-0.05 * time_factor)
        
        puts.append({
            'strike': float(strike),
            'bid': round(put_price * 0.98, 2),
            'ask': round(put_price * 1.02, 2),
            'last': round(put_price, 2),
            'volume': int(np.random.exponential(300)),
            'openInterest': int(np.random.exponential(1500)),
            'impliedVolatility': round(iv, 4),
            'delta': round(put_delta, 4),
            'gamma': round(gamma, 4),
            'theta': round(put_theta / 365, 4),
            'vega': round(vega / 100, 4),
            'rho': round(-strike * time_factor * np.exp(-0.05 * time_factor) * norm.cdf(-d2) / 100, 4)
        })
    
    return {
        'symbol': symbol,
        'expiry': expiry_date.strftime('%Y-%m-%d'),
        'underlying_price': current_price,
        'calls': calls,
        'puts': puts
    }


def create_sample_earnings_calendar() -> List[Dict[str, Any]]:
    """Create sample earnings calendar data"""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    earnings = []
    
    base_date = datetime.now()
    
    for i, symbol in enumerate(symbols):
        earnings_date = base_date + timedelta(days=7 + i * 7)  # Weekly spread
        
        earnings.append({
            'symbol': symbol,
            'company_name': f"{symbol} Inc.",
            'earnings_date': earnings_date.strftime('%Y-%m-%d'),
            'earnings_time': 'AMC' if i % 2 == 0 else 'BMO',  # After/Before market close
            'fiscal_quarter': f"Q{(i % 4) + 1}",
            'fiscal_year': 2024,
            'estimated_eps': round(np.random.uniform(1.0, 5.0), 2),
            'estimated_revenue': round(np.random.uniform(10, 50), 2) * 1e9,
            'analyst_count': np.random.randint(15, 40)
        })
    
    return earnings


def create_sample_market_conditions() -> Dict[str, Any]:
    """Create sample market condition data"""
    return {
        'vix': {
            'current': 18.5,
            'change': -0.8,
            'change_percent': -4.1
        },
        'spy': {
            'current': 485.2,
            'change': 2.1,
            'change_percent': 0.43
        },
        'qqq': {
            'current': 412.8,
            'change': 1.7,
            'change_percent': 0.41
        },
        'iwm': {
            'current': 195.4,
            'change': 0.8,
            'change_percent': 0.41
        },
        'market_sentiment': 'BULLISH',
        'fear_greed_index': 72,
        'put_call_ratio': 0.85,
        'yield_10y': 4.25,
        'dxy': 103.8
    }


def create_sample_company_fundamentals(symbol: str = "AAPL") -> Dict[str, Any]:
    """Create sample company fundamental data"""
    # Base data varies by symbol for consistency
    np.random.seed(hash(symbol) % 2**32)
    
    market_caps = {
        'AAPL': 3000e9,
        'MSFT': 2800e9,
        'GOOGL': 1800e9,
        'TSLA': 800e9,
        'NVDA': 1700e9,
        'META': 900e9,
        'AMZN': 1600e9,
        'NFLX': 200e9
    }
   
    sectors = {
        'AAPL': 'Technology',
        'MSFT': 'Technology', 
        'GOOGL': 'Communication Services',
        'TSLA': 'Consumer Discretionary',
        'NVDA': 'Technology',
        'META': 'Communication Services',
        'AMZN': 'Consumer Discretionary',
        'NFLX': 'Communication Services'
    }
   
    base_market_cap = market_caps.get(symbol, 500e9)
    sector = sectors.get(symbol, 'Technology')
   
    # Generate realistic fundamental metrics
    price = 150.0 * (0.8 + 0.4 * np.random.random())
    shares_outstanding = base_market_cap / price
   
    revenue = base_market_cap * (0.15 + 0.1 * np.random.random())
    net_income = revenue * (0.15 + 0.15 * np.random.random())
    eps = net_income / shares_outstanding
   
    return {
        'symbol': symbol,
        'company_name': f"{symbol} Inc.",
        'sector': sector,
        'industry': f"{sector} Equipment" if sector == 'Technology' else f"{sector} Services",
        'market_cap': base_market_cap,
        'shares_outstanding': shares_outstanding,
        'price': round(price, 2),
        'revenue_ttm': revenue,
        'net_income_ttm': net_income,
        'eps_ttm': round(eps, 2),
        'pe_ratio': round(price / eps, 2),
        'pb_ratio': round(1.5 + 3.0 * np.random.random(), 2),
        'ps_ratio': round(base_market_cap / revenue, 2),
        'debt_to_equity': round(0.1 + 0.4 * np.random.random(), 2),
        'roe': round(0.1 + 0.2 * np.random.random(), 4),
        'roa': round(0.05 + 0.15 * np.random.random(), 4),
        'beta': round(0.8 + 0.8 * np.random.random(), 2),
        'dividend_yield': round(0.01 * np.random.random(), 4) if symbol != 'TSLA' else 0.0,
        'fifty_two_week_high': round(price * (1.1 + 0.3 * np.random.random()), 2),
        'fifty_two_week_low': round(price * (0.7 + 0.2 * np.random.random()), 2),
        'analyst_target': round(price * (0.95 + 0.1 * np.random.random()), 2),
        'analyst_rating': np.random.choice(['Strong Buy', 'Buy', 'Hold', 'Sell'], 
                                        p=[0.3, 0.4, 0.25, 0.05])
    }


def create_sample_volatility_surface(symbol: str = "AAPL",
                                   current_price: float = 150.0) -> Dict[str, Any]:
    """Create sample implied volatility surface"""
    # Expiry dates (1 week to 6 months)
    expiry_days = [7, 14, 21, 30, 45, 60, 90, 120, 180]
   
    # Moneyness levels (80% to 120% of current price)
    moneyness_levels = np.arange(0.8, 1.21, 0.05)
    strikes = [round(current_price * m, 0) for m in moneyness_levels]
   
    # Base volatility parameters
    base_iv = 0.25
    term_structure_slope = 0.02  # Upward sloping term structure
    skew_factor = -0.15  # Negative skew (puts more expensive)
   
    surface_data = []
   
    for days in expiry_days:
        time_factor = days / 365
        term_structure_adj = term_structure_slope * np.sqrt(time_factor)
       
        for strike, moneyness in zip(strikes, moneyness_levels):
            # Calculate IV with smile and term structure
            log_moneyness = np.log(moneyness)
           
            # Volatility smile (quadratic in log-moneyness)
            smile_adj = 0.1 * log_moneyness**2 + skew_factor * log_moneyness
           
            # Add some random noise
            noise = np.random.normal(0, 0.01)
           
            iv = base_iv + term_structure_adj + smile_adj + noise
            iv = max(0.05, iv)  # Minimum 5% IV
           
            surface_data.append({
                'strike': strike,
                'expiry_days': days,
                'moneyness': round(moneyness, 3),
                'implied_volatility': round(iv, 4),
                'call_price': None,  # Would be calculated
                'put_price': None    # Would be calculated
            })
   
    return {
        'symbol': symbol,
        'underlying_price': current_price,
        'surface_data': surface_data,
        'base_iv': base_iv,
        'skew': skew_factor,
        'term_structure': term_structure_slope
    }


def create_sample_trade_history() -> List[Dict[str, Any]]:
    """Create sample trade history for backtesting"""
    trades = []
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    strategies = ['calendar_spread', 'iron_condor', 'butterfly', 'straddle']
   
    base_date = datetime.now() - timedelta(days=180)  # 6 months ago
   
    for i in range(50):  # 50 sample trades
        entry_date = base_date + timedelta(days=i * 3)
        exit_date = entry_date + timedelta(days=np.random.randint(7, 45))
       
        symbol = np.random.choice(symbols)
        strategy = np.random.choice(strategies)
       
        # Generate trade parameters
        contracts = np.random.randint(1, 10)
        debit = round(np.random.uniform(1.0, 5.0), 2)
       
        # Generate outcome (70% win rate for sample)
        is_winner = np.random.random() < 0.7
       
        if is_winner:
            # Winning trade
            profit_factor = np.random.uniform(0.2, 0.8)  # 20-80% of max profit
            pnl = contracts * 100 * debit * profit_factor
        else:
            # Losing trade
            loss_factor = np.random.uniform(0.3, 1.0)  # 30-100% of max loss
            pnl = -contracts * 100 * debit * loss_factor
       
        trades.append({
            'trade_id': f"TRADE_{i+1:03d}",
            'symbol': symbol,
            'strategy': strategy,
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'contracts': contracts,
            'debit': debit,
            'credit': 0.0,  # Debit strategies
            'pnl': round(pnl, 2),
            'pnl_percent': round((pnl / (contracts * 100 * debit)) * 100, 2),
            'max_profit': round(contracts * 100 * debit * 0.5, 2),
            'max_loss': round(-contracts * 100 * debit, 2),
            'days_held': (exit_date - entry_date).days,
            'exit_reason': np.random.choice(['profit_target', 'stop_loss', 'expiration', 'manual']),
            'iv_rank_entry': round(np.random.uniform(0.1, 0.9), 2),
            'iv_rank_exit': round(np.random.uniform(0.1, 0.9), 2)
        })
   
    return trades


def create_sample_economic_indicators() -> Dict[str, Any]:
    """Create sample economic indicator data"""
    return {
        'inflation': {
            'cpi_monthly': 0.2,
            'cpi_annual': 3.2,
            'core_cpi_annual': 3.8,
            'pce_annual': 2.9
        },
        'employment': {
            'unemployment_rate': 3.7,
            'payrolls_change': 187000,
            'labor_participation': 62.8
        },
        'monetary_policy': {
            'fed_funds_rate': 5.25,
            'next_meeting_date': '2024-03-20',
            'rate_hike_probability': 0.15
        },
        'growth': {
            'gdp_growth_annual': 2.1,
            'gdp_growth_quarterly': 0.5,
            'ism_manufacturing': 47.8,
            'ism_services': 52.7
        },
        'international': {
            'dxy_index': 103.8,
            'crude_oil': 78.50,
            'gold': 2045.30,
            'bitcoin': 42150.00
        }
    }


def create_sample_news_sentiment() -> List[Dict[str, Any]]:
    """Create sample news sentiment data"""
    news_items = []
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'SPY', 'QQQ']
   
    base_time = datetime.now()
   
    headlines = [
        "Strong quarterly earnings beat expectations",
        "New product launch receives positive reviews",
        "Management guidance raised for upcoming quarter",
        "Analyst upgrade based on market share gains",
        "Partnership announcement with major tech company",
        "Regulatory concerns weigh on stock performance",
        "Supply chain disruptions impact production",
        "Competitive pressure from emerging rivals",
        "Economic headwinds affect sector outlook",
        "Geopolitical tensions create market uncertainty"
    ]
   
    for i in range(30):  # 30 news items
        timestamp = base_time - timedelta(hours=i * 2)
        symbol = np.random.choice(symbols)
        headline = np.random.choice(headlines)
       
        # Sentiment score based on headline type
        if any(word in headline.lower() for word in ['strong', 'beat', 'positive', 'raised', 'upgrade', 'gains']):
            sentiment_score = np.random.uniform(0.6, 0.9)
            sentiment_label = 'POSITIVE'
        elif any(word in headline.lower() for word in ['concerns', 'disruptions', 'pressure', 'headwinds', 'tensions']):
            sentiment_score = np.random.uniform(0.1, 0.4)
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_score = np.random.uniform(0.4, 0.6)
            sentiment_label = 'NEUTRAL'
       
        news_items.append({
            'id': f"NEWS_{i+1:03d}",
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'headline': f"{symbol}: {headline}",
            'source': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance']),
            'sentiment_score': round(sentiment_score, 3),
            'sentiment_label': sentiment_label,
            'relevance_score': round(np.random.uniform(0.5, 1.0), 3),
            'impact_score': round(np.random.uniform(0.3, 0.8), 3)
        })
   
    return news_items


def create_sample_portfolio_positions() -> List[Dict[str, Any]]:
    """Create sample portfolio positions for testing"""
    positions = []
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
    strategies = ['calendar_spread', 'iron_condor', 'butterfly', 'covered_call']
   
    for i, symbol in enumerate(symbols):
        strategy = np.random.choice(strategies)
       
        # Position details
        contracts = np.random.randint(1, 5)
        entry_price = round(np.random.uniform(2.0, 8.0), 2)
        current_price = entry_price * (0.8 + 0.4 * np.random.random())
       
        # Greeks
        delta = round(np.random.uniform(-0.5, 0.5), 3)
        gamma = round(np.random.uniform(0.01, 0.05), 3)
        theta = round(np.random.uniform(-8.0, -1.0), 2)
        vega = round(np.random.uniform(5.0, 25.0), 2)
       
        # Dates
        entry_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
        expiry_date = entry_date + timedelta(days=np.random.randint(15, 60))
        days_to_expiry = (expiry_date - datetime.now()).days
       
        positions.append({
            'position_id': f"POS_{i+1:03d}",
            'symbol': symbol,
            'strategy': strategy,
            'contracts': contracts,
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
            'days_to_expiry': max(0, days_to_expiry),
            'entry_price': entry_price,
            'current_price': round(current_price, 2),
            'unrealized_pnl': round((current_price - entry_price) * contracts * 100, 2),
            'unrealized_pnl_percent': round(((current_price - entry_price) / entry_price) * 100, 2),
            'max_profit': round(entry_price * 0.5 * contracts * 100, 2),
            'max_loss': round(-entry_price * contracts * 100, 2),
            'delta': delta * contracts,
            'gamma': gamma * contracts,
            'theta': theta * contracts,
            'vega': vega * contracts,
            'iv_rank': round(np.random.uniform(0.2, 0.8), 2),
            'probability_profit': round(np.random.uniform(0.4, 0.8), 2),
            'status': np.random.choice(['open', 'monitoring', 'closing']),
            'notes': f"Entered on {symbol} strength, target 50% profit"
        })
   
    return positions


def create_sample_backtest_results() -> Dict[str, Any]:
    """Create sample backtesting results"""
    # Generate trade results
    num_trades = 100
    win_rate = 0.68
   
    trades = []
    cumulative_pnl = []
    running_total = 0
   
    for i in range(num_trades):
        # Win/loss determination
        is_winner = np.random.random() < win_rate
       
        if is_winner:
            pnl = round(np.random.uniform(50, 400), 2)
        else:
            pnl = round(np.random.uniform(-500, -100), 2)
       
        running_total += pnl
       
        trades.append({
            'trade_number': i + 1,
            'pnl': pnl,
            'win': is_winner
        })
       
        cumulative_pnl.append(running_total)
   
    # Calculate statistics
    winning_trades = [t['pnl'] for t in trades if t['win']]
    losing_trades = [t['pnl'] for t in trades if not t['win']]
   
    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    # Risk metrics
    returns = np.diff([0] + cumulative_pnl)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
   
    # Maximum drawdown
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdowns = np.array(cumulative_pnl) - peak
    max_drawdown = np.min(drawdowns)
   
    return {
        'summary': {
            'total_trades': num_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(len(winning_trades) / num_trades, 3),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(abs(sum(winning_trades) / sum(losing_trades)), 2) if losing_trades else float('inf'),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_drawdown': round(max_drawdown, 2),
            'calmar_ratio': round(total_pnl / abs(max_drawdown), 3) if max_drawdown != 0 else float('inf')
        },
        'trades': trades,
        'cumulative_pnl': cumulative_pnl,
        'monthly_returns': [round(np.random.normal(0.05, 0.15), 4) for _ in range(12)],
        'risk_metrics': {
            'var_95': round(np.percentile(returns, 5), 2),
            'var_99': round(np.percentile(returns, 1), 2),
            'expected_shortfall': round(np.mean(returns[returns <= np.percentile(returns, 5)]), 2),
            'volatility': round(np.std(returns) * np.sqrt(252), 3),
            'beta': round(np.random.uniform(0.3, 1.2), 3),
            'alpha': round(np.random.uniform(-0.02, 0.08), 4)
        }
    }


# Utility functions for test data generation
def save_sample_data_to_files(output_dir: str = "tests/fixtures/data"):
    """Save all sample data to files for test loading"""
    import os
    import json
   
    os.makedirs(output_dir, exist_ok=True)
   
    # Save various data types
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
   
    for symbol in symbols:
        # Price data
        price_data = create_sample_price_data(symbol)
        price_data.to_csv(f"{output_dir}/{symbol}_price_data.csv")
       
        # Options chain
        options_data = create_sample_options_chain(symbol)
        with open(f"{output_dir}/{symbol}_options_chain.json", 'w') as f:
            json.dump(options_data, f, indent=2)
       
        # Company fundamentals
        fundamentals = create_sample_company_fundamentals(symbol)
        with open(f"{output_dir}/{symbol}_fundamentals.json", 'w') as f:
            json.dump(fundamentals, f, indent=2)
   
    # Market-wide data
    earnings_calendar = create_sample_earnings_calendar()
    with open(f"{output_dir}/earnings_calendar.json", 'w') as f:
        json.dump(earnings_calendar, f, indent=2, default=str)
   
    market_conditions = create_sample_market_conditions()
    with open(f"{output_dir}/market_conditions.json", 'w') as f:
        json.dump(market_conditions, f, indent=2)
   
    economic_data = create_sample_economic_indicators()
    with open(f"{output_dir}/economic_indicators.json", 'w') as f:
        json.dump(economic_data, f, indent=2)
   
    news_data = create_sample_news_sentiment()
    with open(f"{output_dir}/news_sentiment.json", 'w') as f:
        json.dump(news_data, f, indent=2, default=str)
   
    trade_history = create_sample_trade_history()
    with open(f"{output_dir}/trade_history.json", 'w') as f:
        json.dump(trade_history, f, indent=2, default=str)
   
    portfolio_positions = create_sample_portfolio_positions()
    with open(f"{output_dir}/portfolio_positions.json", 'w') as f:
        json.dump(portfolio_positions, f, indent=2, default=str)
   
    backtest_results = create_sample_backtest_results()
    with open(f"{output_dir}/backtest_results.json", 'w') as f:
        json.dump(backtest_results, f, indent=2)
   
    print(f"Sample data saved to {output_dir}")


def load_sample_data_from_files(data_dir: str = "tests/fixtures/data") -> Dict[str, Any]:
    """Load sample data from files"""
    import os
    import json
   
    if not os.path.exists(data_dir):
        # Generate data if it doesn't exist
        save_sample_data_to_files(data_dir)
   
    data = {}
   
    # Load JSON files
    json_files = [
        'earnings_calendar', 'market_conditions', 'economic_indicators',
        'news_sentiment', 'trade_history', 'portfolio_positions', 'backtest_results'
    ]
   
    for filename in json_files:
        filepath = os.path.join(data_dir, f"{filename}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[filename] = json.load(f)
   
    # Load symbol-specific data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    data['symbols'] = {}
   
    for symbol in symbols:
        data['symbols'][symbol] = {}
       
        # Price data
        price_file = os.path.join(data_dir, f"{symbol}_price_data.csv")
        if os.path.exists(price_file):
            data['symbols'][symbol]['price_data'] = pd.read_csv(price_file, index_col=0, parse_dates=True)
       
        # Options chain
        options_file = os.path.join(data_dir, f"{symbol}_options_chain.json")
        if os.path.exists(options_file):
            with open(options_file, 'r') as f:
                data['symbols'][symbol]['options_chain'] = json.load(f)
       
        # Fundamentals
        fundamentals_file = os.path.join(data_dir, f"{symbol}_fundamentals.json")
        if os.path.exists(fundamentals_file):
            with open(fundamentals_file, 'r') as f:
                data['symbols'][symbol]['fundamentals'] = json.load(f)
   
    return data


if __name__ == "__main__":
    # Generate and save sample data when run directly
    save_sample_data_to_files()
    print("Sample data generated successfully!")