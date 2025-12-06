"""
Bulk Analysis Example
Demonstrates processing multiple symbols efficiently
"""

from options_calculator import OptionsCalculator
from options_calculator.services import AnalysisService
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
import time
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BulkAnalyzer:
    """Bulk analysis processor with optimization"""
    
    def __init__(self, max_workers: int = 4):
        self.calc = OptionsCalculator()
        self.max_workers = max_workers
        self.results = []
    
    def analyze_symbol_list(self, symbols: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Analyze multiple symbols in parallel"""
        
        logger.info(f"Starting bulk analysis of {len(symbols)} symbols...")
        start_time = time.time()
        
        results = []
        errors = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_symbol = {
                executor.submit(self._analyze_single, symbol, **kwargs): symbol 
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    if result:
                        results.append(result)
                        logger.info(f"✅ Completed {symbol}: {result['recommendation']}")
                    else:
                        errors.append({'symbol': symbol, 'error': 'No result'})
                        logger.warning(f"❌ Failed {symbol}: No result")
                except Exception as e:
                    errors.append({'symbol': symbol, 'error': str(e)})
                    logger.error(f"❌ Failed {symbol}: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Bulk analysis completed in {total_time:.1f}s")
        logger.info(f"Success: {len(results)}/{len(symbols)} symbols")
        
        if errors:
            logger.warning(f"Errors: {len(errors)} symbols failed")
            for error in errors:
                logger.warning(f"  {error['symbol']}: {error['error']}")
        
        self.results = results
        return results
    
    def _analyze_single(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Analyze a single symbol"""
        try:
            result = self.calc.analyze(symbol=symbol, **kwargs)
            
            return {
                'symbol': symbol,
                'recommendation': result.recommendation,
                'confidence': result.confidence,
                'expected_profit': result.expected_profit,
                'max_loss': result.max_loss,
                'probability_profit': result.probability_profit,
                'risk_reward_ratio': result.risk_reward_ratio,
                'analysis_timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return None
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        df['analysis_timestamp'] = pd.to_datetime(df['analysis_timestamp'], unit='s')
        return df
    
    def get_top_opportunities(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N opportunities by confidence score"""
        if not self.results:
            return []
        
        # Sort by confidence * risk_reward_ratio for best opportunities
        sorted_results = sorted(
            self.results, 
            key=lambda x: x['confidence'] * x['risk_reward_ratio'], 
            reverse=True
        )
        
        return sorted_results[:n]
    
    def export_results(self, filename: str):
        """Export results to CSV file"""
        df = self.get_results_dataframe()
        if not df.empty:
            df.to_csv(filename, index=False)
            logger.info(f"Results exported to {filename}")
        else:
            logger.warning("No results to export")

def sp500_screening_example():
    """Screen S&P 500 stocks for opportunities"""
    
    # Sample of S&P 500 symbols (in practice, you'd load the full list)
    sp500_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
        'JNJ', 'JPM', 'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'ABBV', 'PFE',
        'KO', 'PEP', 'AVGO', 'COST', 'TMO', 'DHR', 'ABT', 'ACN', 'NKE',
        'LIN', 'VZ', 'ADBE', 'CRM', 'TXN', 'WMT', 'CSCO', 'XOM', 'NEE'
    ]
    
    print(f"\n=== S&P 500 Screening ({len(sp500_symbols)} symbols) ===")
    
    # Initialize bulk analyzer
    analyzer = BulkAnalyzer(max_workers=6)
    
    # Run analysis with consistent parameters
    results = analyzer.analyze_symbol_list(
        symbols=sp500_symbols,
        contracts=1,
        portfolio_value=250000,
        max_risk_pct=0.02
    )
    
    if results:
        # Get results as DataFrame
        df = analyzer.get_results_dataframe()
        
        print(f"\n=== Summary Statistics ===")
        print(f"Total Analyzed: {len(df)}")
        print(f"BUY Recommendations: {len(df[df['recommendation'] == 'BUY'])}")
        print(f"HOLD Recommendations: {len(df[df['recommendation'] == 'HOLD'])}")
        print(f"AVOID Recommendations: {len(df[df['recommendation'] == 'AVOID'])}")
        print(f"Average Confidence: {df['confidence'].mean():.1%}")
        print(f"Average Risk/Reward: {df['risk_reward_ratio'].mean():.2f}")
        
        # Top opportunities
        top_opportunities = analyzer.get_top_opportunities(10)
        
        print(f"\n=== Top 10 Opportunities ===")
        for i, opp in enumerate(top_opportunities, 1):
            print(f"{i:2d}. {opp['symbol']:5s} | {opp['recommendation']:4s} | "
                  f"Conf: {opp['confidence']:5.1%} | R/R: {opp['risk_reward_ratio']:4.2f} | "
                  f"Profit: ${opp['expected_profit']:6.0f}")
        
        # Export results
        analyzer.export_results('sp500_screening_results.csv')
        
        return df
    
    return None

def sector_analysis_example():
    """Analyze stocks by sector"""
    
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'ADBE', 'CRM', 'ORCL'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'ABT'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK'],
        'Consumer': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX', 'MCD', 'DIS'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'OXY']
    }
    
    print(f"\n=== Sector Analysis ===")
    
    analyzer = BulkAnalyzer(max_workers=4)
    sector_results = {}
    
    for sector_name, symbols in sectors.items():
        print(f"\nAnalyzing {sector_name} sector ({len(symbols)} stocks)...")
        
        results = analyzer.analyze_symbol_list(
            symbols=symbols,
            contracts=1,
            portfolio_value=200000
        )
        
        if results:
            df = pd.DataFrame(results)
            
            sector_stats = {
                'sector': sector_name,
                'total_stocks': len(df),
                'buy_signals': len(df[df['recommendation'] == 'BUY']),
                'avg_confidence': df['confidence'].mean(),
                'avg_risk_reward': df['risk_reward_ratio'].mean(),
                'top_stock': df.loc[df['confidence'].idxmax(), 'symbol'],
                'top_confidence': df['confidence'].max()
            }
            
            sector_results[sector_name] = sector_stats
            
            print(f"  BUY signals: {sector_stats['buy_signals']}/{sector_stats['total_stocks']}")
            print(f"  Avg confidence: {sector_stats['avg_confidence']:.1%}")
            print(f"  Top stock: {sector_stats['top_stock']} ({sector_stats['top_confidence']:.1%})")
    
    # Sector comparison
    if sector_results:
        print(f"\n=== Sector Comparison ===")
        sector_df = pd.DataFrame(sector_results.values())
        sector_df = sector_df.sort_values('avg_confidence', ascending=False)
        
        for _, row in sector_df.iterrows():
            print(f"{row['sector']:12s} | Conf: {row['avg_confidence']:5.1%} | "
                  f"BUY: {row['buy_signals']:2d}/{row['total_stocks']:2d} | "
                  f"Top: {row['top_stock']}")
    
    return sector_results

def watchlist_monitoring_example():
    """Monitor a custom watchlist"""
    
    # Custom watchlist
    watchlist = [
        'AAPL', 'TSLA', 'NVDA', 'AMD', 'GOOGL', 
        'META', 'NFLX', 'AMZN', 'MSFT', 'CRM'
    ]
    
    print(f"\n=== Watchlist Monitoring ===")
    print(f"Monitoring {len(watchlist)} symbols...")
    
    analyzer = BulkAnalyzer(max_workers=5)
    
    # Run analysis multiple times to simulate monitoring
    monitoring_results = []
    
    for cycle in range(3):  # 3 monitoring cycles
        print(f"\n--- Monitoring Cycle {cycle + 1} ---")
        
        results = analyzer.analyze_symbol_list(
            symbols=watchlist,
            contracts=1,
            portfolio_value=150000
        )
        
        if results:
            # Find symbols with high confidence
            high_confidence = [r for r in results if r['confidence'] > 0.70]
            
            if high_confidence:
                print(f"High confidence opportunities found:")
                for opp in high_confidence:
                    print(f"  🎯 {opp['symbol']}: {opp['recommendation']} "
                          f"({opp['confidence']:.1%} confidence)")
            else:
                print("No high confidence opportunities found")
            
            # Store results with cycle info
            for result in results:
                result['cycle'] = cycle + 1
            
            monitoring_results.extend(results)
        
        # In real monitoring, you would wait between cycles
        time.sleep(1)  # Short delay for demo
    
    # Analysis of monitoring results
    if monitoring_results:
        df = pd.DataFrame(monitoring_results)
        
        print(f"\n=== Monitoring Summary ===")
        
        # Consistency analysis
        symbol_consistency = df.groupby('symbol').agg({
            'confidence': ['mean', 'std'],
            'recommendation': lambda x: x.mode().iloc[0] if not x.mode().empty else 'MIXED'
        }).round(3)
        
        print("Symbol consistency (across monitoring cycles):")
        for symbol in watchlist:
            symbol_data = symbol_consistency.loc[symbol]
            avg_conf = symbol_data[('confidence', 'mean')]
            conf_std = symbol_data[('confidence', 'std')]
            rec = symbol_data[('recommendation', '<lambda>')]
            
            print(f"  {symbol:5s}: {rec:4s} | Conf: {avg_conf:5.1%} ± {conf_std:4.1%}")
    
    return monitoring_results

def performance_comparison():
    """Compare sequential vs parallel processing performance"""
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
    print(f"\n=== Performance Comparison ===")
    print(f"Testing with {len(test_symbols)} symbols...")
    
    # Sequential processing
    calc = OptionsCalculator()
    
    start_time = time.time()
    sequential_results = []
    
    for symbol in test_symbols:
        try:
            result = calc.analyze(symbol, contracts=1)
            sequential_results.append({
                'symbol': symbol,
                'recommendation': result.recommendation,
                'confidence': result.confidence
            })
        except Exception as e:
            logger.error(f"Sequential analysis failed for {symbol}: {e}")
    
    sequential_time = time.time() - start_time
    
    # Parallel processing
    analyzer = BulkAnalyzer(max_workers=4)
    
    start_time = time.time()
    parallel_results = analyzer.analyze_symbol_list(test_symbols, contracts=1)
    parallel_time = time.time() - start_time
    
    # Results
    print(f"\nPerformance Results:")
    print(f"Sequential: {sequential_time:.1f}s ({len(sequential_results)} successful)")
    print(f"Parallel:   {parallel_time:.1f}s ({len(parallel_results)} successful)")
    
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"Speedup:    {speedup:.1f}x")
   
    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'sequential_results': len(sequential_results),
        'parallel_results': len(parallel_results)
    }

if __name__ == "__main__":
    print("Options Calculator Pro - Bulk Analysis Examples")
    print("=" * 60)
   
    try:
        # Run examples
        print("\n1. S&P 500 Screening Example")
        sp500_results = sp500_screening_example()
       
        print("\n2. Sector Analysis Example")
        sector_results = sector_analysis_example()
       
        print("\n3. Watchlist Monitoring Example")
        watchlist_results = watchlist_monitoring_example()
       
        print("\n4. Performance Comparison")
        perf_results = performance_comparison()
       
        print("\n" + "=" * 60)
        print("Bulk analysis examples completed successfully!")
       
        # Summary
        if sp500_results is not None:
            print(f"\nTotal symbols analyzed: {len(sp500_results)}")
            print(f"Results exported to: sp500_screening_results.csv")
       
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(f"Example execution failed: {e}")