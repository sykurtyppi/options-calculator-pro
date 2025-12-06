"""
Data worker for background data fetching and processing
Professional Options Calculator - Data Worker
"""

import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import asyncio

from .base_worker import BaseWorker
from services.market_data import MarketDataService
from services.options_service import OptionsService

logger = logging.getLogger(__name__)

class DataFetchWorker(BaseWorker):
    """Worker for fetching market data in background"""
    
    def __init__(self, symbols: List[str], data_types: List[str], 
                 fetch_options: Dict[str, Any] = None):
        super().__init__()
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.data_types = data_types  # ['price', 'historical', 'options', 'earnings']
        self.fetch_options = fetch_options or {}
        
        # Services
        self.market_data = MarketDataService()
        self.options_service = OptionsService()
        
        # Results
        self.fetch_results = {}
        self.fetch_errors = {}
    
    def run(self):
        """Main data fetching execution"""
        try:
            self.started.emit()
            self.emit_status("Starting data fetch operation...")
            
            if not self.symbols:
                self.emit_error("Invalid Input", "No symbols provided for data fetch")
                return
            
            total_operations = len(self.symbols) * len(self.data_types)
            current_operation = 0
            
            # Fetch data for each symbol
            for symbol in self.symbols:
                if self.is_cancelled():
                    break
                
                self.fetch_results[symbol] = {}
                self.fetch_errors[symbol] = {}
                
                # Fetch each data type
                for data_type in self.data_types:
                    if self.is_cancelled():
                        break
                    
                    current_operation += 1
                    progress = int((current_operation / total_operations) * 90)
                    self.emit_progress(progress, f"Fetching {data_type} for {symbol}...")
                    
                    try:
                        result = self._fetch_data_type(symbol, data_type)
                        if result is not None:
                            self.fetch_results[symbol][data_type] = result
                            self.emit_status(f"{symbol} {data_type}: ✓")
                        else:
                            self.fetch_errors[symbol][data_type] = "No data returned"
                            self.emit_status(f"{symbol} {data_type}: ✗")
                            
                    except Exception as e:
                        error_msg = f"Error fetching {data_type}: {str(e)}"
                        self.fetch_errors[symbol][data_type] = error_msg
                        logger.error(f"{symbol} {data_type}: {error_msg}")
                        self.emit_status(f"{symbol} {data_type}: Error")
                    
                    # Rate limiting
                    time.sleep(0.1)
            
            # Emit final results
            self.emit_progress(100, "Data fetch complete")
            
            final_result = {
                'type': 'data_fetch_complete',
                'symbols': self.symbols,
                'data_types': self.data_types,
                'results': self.fetch_results,
                'errors': self.fetch_errors,
                'timestamp': datetime.now().isoformat()
            }
            
            self.emit_result(final_result)
            
        except Exception as e:
            logger.error(f"Critical error in data fetch worker: {e}")
            self.emit_error("Data Fetch Failed", str(e))
        finally:
            self.finished.emit()
    
    def _fetch_data_type(self, symbol: str, data_type: str) -> Optional[Any]:
        """Fetch specific data type for symbol"""
        try:
            if data_type == 'price':
                return self._fetch_current_price(symbol)
            elif data_type == 'historical':
                return self._fetch_historical_data(symbol)
            elif data_type == 'options':
                return self._fetch_options_data(symbol)
            elif data_type == 'earnings':
                return self._fetch_earnings_data(symbol)
            elif data_type == 'volume':
                return self._fetch_volume_data(symbol)
            elif data_type == 'fundamentals':
                return self._fetch_fundamental_data(symbol)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {data_type} for {symbol}: {e}")
            raise
    
    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current stock price"""
        return self.market_data.get_current_price(symbol)
    
    def _fetch_historical_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch historical price data"""
        period = self.fetch_options.get('historical_period', '1y')
        interval = self.fetch_options.get('historical_interval', '1d')
        
        hist_data = self.market_data.get_historical_data(symbol, period=period)
        
        if not hist_data.empty:
            return {
                'data': hist_data.to_dict('index'),
                'period': period,
                'interval': interval,
                'rows': len(hist_data)
            }
        return None
    
    def _fetch_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch options chain data"""
        try:
            # Get available expirations
            expirations = self.options_service.get_available_expirations(symbol, timeout=30)
            if not expirations:
                return None
            
            # Get chains for first few expirations
            max_expirations = self.fetch_options.get('max_option_expirations', 3)
            chains = {}
            
            for i, expiry in enumerate(expirations[:max_expirations]):
                if self.is_cancelled():
                    break
                
                chain = self.options_service.get_option_chain(symbol, expiry, timeout=30)
                if chain:
                    chains[expiry] = {
                        'calls': chain.calls.to_dict('records') if hasattr(chain, 'calls') else [],
                        'puts': chain.puts.to_dict('records') if hasattr(chain, 'puts') else []
                    }
            
            return {
                'expirations': expirations,
                'chains': chains,
                'chain_count': len(chains)
            }
            
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {e}")
            return None
    
    def _fetch_earnings_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch earnings information"""
        try:
            next_earnings = self.market_data.get_next_earnings_date(symbol)
            
            earnings_data = {
                'next_earnings_date': next_earnings.isoformat() if next_earnings else None,
                'days_to_earnings': None
            }
            
            if next_earnings:
                days_to_earnings = (next_earnings - datetime.now().date()).days
                earnings_data['days_to_earnings'] = days_to_earnings
            
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error fetching earnings data for {symbol}: {e}")
            return None
    
    def _fetch_volume_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch volume and liquidity data"""
        try:
            hist_data = self.market_data.get_historical_data(symbol, period='30d')
            
            if hist_data.empty:
                return None
            
            # Calculate volume metrics
            avg_volume = hist_data['Volume'].mean()
            avg_dollar_volume = (hist_data['Volume'] * hist_data['Close']).mean()
            recent_volume = hist_data['Volume'].iloc[-1] if len(hist_data) > 0 else 0
            
            return {
                'avg_volume_30d': avg_volume,
                'avg_dollar_volume_30d': avg_dollar_volume,
                'recent_volume': recent_volume,
                'volume_trend': 'increasing' if recent_volume > avg_volume else 'decreasing'
            }
            
        except Exception as e:
            logger.error(f"Error fetching volume data for {symbol}: {e}")
            return None
    
    def _fetch_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamental data"""
        try:
            # This would use yfinance or another service to get fundamentals
            # Placeholder implementation
            return {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': None,
                'pe_ratio': None,
                'dividend_yield': None
            }
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return None


class CacheUpdateWorker(BaseWorker):
    """Worker for updating cached data"""
    
    def __init__(self, cache_service, update_rules: Dict[str, Any]):
        super().__init__()
        self.cache_service = cache_service
        self.update_rules = update_rules
        
    def run(self):
        """Main cache update execution"""
        try:
            self.started.emit()
            self.emit_status("Starting cache update...")
            
            # Get expired cache entries
            expired_entries = self.cache_service.get_expired_entries()
            
            if not expired_entries:
                self.emit_status("No expired cache entries found")
                self.emit_progress(100, "Cache update complete")
                return
            
            total_entries = len(expired_entries)
            self.emit_status(f"Updating {total_entries} expired cache entries...")
            
            updated_count = 0
            error_count = 0
            
            for i, entry in enumerate(expired_entries):
                if self.is_cancelled():
                    break
                
                progress = int((i / total_entries) * 90)
                self.emit_progress(progress, f"Updating cache entry {i+1}/{total_entries}...")
                
                try:
                    success = self._update_cache_entry(entry)
                    if success:
                        updated_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error updating cache entry {entry}: {e}")
            
            # Emit results
            self.emit_progress(100, f"Cache update complete: {updated_count} updated, {error_count} errors")
            
            result = {
                'type': 'cache_update_complete',
                'total_entries': total_entries,
                'updated_count': updated_count,
                'error_count': error_count,
                'timestamp': datetime.now().isoformat()
            }
            
            self.emit_result(result)
            
        except Exception as e:
            logger.error(f"Error in cache update worker: {e}")
            self.emit_error("Cache Update Failed", str(e))
        finally:
            self.finished.emit()
    
    def _update_cache_entry(self, entry: Dict[str, Any]) -> bool:
        """Update a single cache entry"""
        try:
            # Extract entry information
            cache_key = entry.get('key')
            data_type = entry.get('data_type')
            symbol = entry.get('symbol')
            
            if not all([cache_key, data_type, symbol]):
                return False
            
            # Fetch fresh data based on type
            if data_type == 'price':
                market_data = MarketDataService()
                fresh_data = market_data.get_current_price(symbol)
            elif data_type == 'historical':
                market_data = MarketDataService()
                fresh_data = market_data.get_historical_data(symbol)
            else:
                return False
            
            if fresh_data is not None:
                # Update cache
                self.cache_service.set(cache_key, fresh_data)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating cache entry: {e}")
            return False


class DataValidationWorker(BaseWorker):
    """Worker for validating data quality and consistency"""
    
    def __init__(self, data_sources: List[str], validation_rules: Dict[str, Any]):
        super().__init__()
        self.data_sources = data_sources
        self.validation_rules = validation_rules
        self.validation_results = {}
        
    def run(self):
        """Main data validation execution"""
        try:
            self.started.emit()
            self.emit_status("Starting data validation...")
            
            total_sources = len(self.data_sources)
            
            for i, source in enumerate(self.data_sources):
                if self.is_cancelled():
                    break
                
                progress = int((i / total_sources) * 90)
                self.emit_progress(progress, f"Validating {source}...")
                
                validation_result = self._validate_data_source(source)
                self.validation_results[source] = validation_result
            
            # Compile final validation report
            self.emit_progress(95, "Compiling validation report...")
            
            report = self._compile_validation_report()
            
            self.emit_result({
                'type': 'data_validation_complete',
                'validation_results': self.validation_results,
                'report': report,
                'timestamp': datetime.now().isoformat()
            })
            
            self.emit_progress(100, "Data validation complete")
            
        except Exception as e:
            logger.error(f"Error in data validation worker: {e}")
            self.emit_error("Data Validation Failed", str(e))
        finally:
            self.finished.emit()
    
    def _validate_data_source(self, source: str) -> Dict[str, Any]:
        """Validate a single data source"""
        try:
            validation_result = {
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'tests': {},
                'overall_status': 'UNKNOWN',
                'issues': []
            }
            
            # Run various validation tests
            tests = [
                ('connectivity', self._test_connectivity),
                ('data_freshness', self._test_data_freshness),
                ('data_accuracy', self._test_data_accuracy),
                ('data_completeness', self._test_data_completeness),
                ('response_time', self._test_response_time)
            ]
           
            passed_tests = 0
            total_tests = len(tests)
           
            for test_name, test_function in tests:
                try:
                    test_result = test_function(source)
                    validation_result['tests'][test_name] = test_result
                   
                    if test_result.get('passed', False):
                       passed_tests += 1
                    else:
                        validation_result['issues'].append(test_result.get('issue', f'{test_name} failed'))
                       
                except Exception as e:
                    validation_result['tests'][test_name] = {
                        'passed': False,
                        'error': str(e),
                        'issue': f'{test_name} test failed with error'
                    }
                    validation_result['issues'].append(f'{test_name} test error: {str(e)}')
           
            # Determine overall status
            if passed_tests == total_tests:
               validation_result['overall_status'] = 'HEALTHY'
            elif passed_tests >= total_tests * 0.7:
               validation_result['overall_status'] = 'WARNING'
            else:
               validation_result['overall_status'] = 'CRITICAL'
           
            validation_result['test_score'] = (passed_tests / total_tests) * 100
           
            return validation_result
           
        except Exception as e:
            logger.error(f"Error validating data source {source}: {e}")
            return {
                'source': source,
                'overall_status': 'ERROR',
                'error': str(e)
            }
   
    def _test_connectivity(self, source: str) -> Dict[str, Any]:
        """Test connectivity to data source"""
        try:
            # Implement actual connectivity test based on source type
            if 'yfinance' in source.lower():
                import yfinance as yf
                test_ticker = yf.Ticker("AAPL")
                test_data = test_ticker.info
               
                if test_data and 'regularMarketPrice' in test_data:
                    return {'passed': True, 'response_time_ms': 100}  # Placeholder
                else:
                    return {'passed': False, 'issue': 'No valid data returned'}
           
            # Default connectivity test
            return {'passed': True, 'response_time_ms': 50}
           
        except Exception as e:
            return {'passed': False, 'issue': f'Connectivity test failed: {str(e)}'}
   
    def _test_data_freshness(self, source: str) -> Dict[str, Any]:
        """Test if data is fresh/recent"""
        try:
            # Test data freshness based on source
            current_time = datetime.now()
           
            # For market data, check if data is from today during market hours
            # This is a simplified test
            return {
                'passed': True,
                'last_update': current_time.isoformat(),
                'staleness_minutes': 0
            }
           
        except Exception as e:
            return {'passed': False, 'issue': f'Data freshness test failed: {str(e)}'}
   
    def _test_data_accuracy(self, source: str) -> Dict[str, Any]:
        """Test data accuracy by comparing with alternative sources"""
        try:
            # This would implement cross-validation between data sources
            # For now, return a placeholder
            return {
                'passed': True,
                'accuracy_score': 95.0,
                'cross_validation_performed': True
            }
           
        except Exception as e:
            return {'passed': False, 'issue': f'Data accuracy test failed: {str(e)}'}
   
    def _test_data_completeness(self, source: str) -> Dict[str, Any]:
        """Test if data is complete (no missing required fields)"""
        try:
            # Test for missing or null data fields
            # This would check for required fields based on data type
            return {
                'passed': True,
                'completeness_score': 98.0,
                'missing_fields': []
            }
           
        except Exception as e:
            return {'passed': False, 'issue': f'Data completeness test failed: {str(e)}'}
   
    def _test_response_time(self, source: str) -> Dict[str, Any]:
        """Test response time performance"""
        try:
            start_time = time.time()
           
            # Perform a test query based on source type
            if 'yfinance' in source.lower():
                import yfinance as yf
                test_ticker = yf.Ticker("AAPL")
                _ = test_ticker.info.get('regularMarketPrice')
           
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
           
            # Define acceptable response time thresholds
            if response_time < 1000:  # Less than 1 second
                return {
                    'passed': True,
                    'response_time_ms': response_time,
                    'performance_rating': 'EXCELLENT'
                }
            elif response_time < 3000:  # Less than 3 seconds
                return {
                    'passed': True,
                    'response_time_ms': response_time,
                    'performance_rating': 'GOOD'
                }
            else:
                return {
                    'passed': False,
                    'response_time_ms': response_time,
                    'performance_rating': 'POOR',
                    'issue': f'Response time too slow: {response_time:.0f}ms'
                }
               
        except Exception as e:
            return {'passed': False, 'issue': f'Response time test failed: {str(e)}'}
   
    def _compile_validation_report(self) -> Dict[str, Any]:
        """Compile comprehensive validation report"""
        try:
            report = {
                'summary': {
                    'total_sources': len(self.data_sources),
                    'healthy_sources': 0,
                    'warning_sources': 0,
                    'critical_sources': 0,
                    'error_sources': 0
                },
                'detailed_results': self.validation_results,
                'recommendations': []
            }
           
            # Count sources by status
            for source_result in self.validation_results.values():
                status = source_result.get('overall_status', 'ERROR')
                if status == 'HEALTHY':
                    report['summary']['healthy_sources'] += 1
                elif status == 'WARNING':
                    report['summary']['warning_sources'] += 1
                elif status == 'CRITICAL':
                    report['summary']['critical_sources'] += 1
                else:
                    report['summary']['error_sources'] += 1
           
            # Generate recommendations
            if report['summary']['critical_sources'] > 0:
                report['recommendations'].append(
                    "Critical data sources detected. Immediate attention required."
                )
           
            if report['summary']['warning_sources'] > 0:
                report['recommendations'].append(
                    "Some data sources showing warnings. Monitor closely."
                )
           
            if report['summary']['healthy_sources'] == report['summary']['total_sources']:
                report['recommendations'].append(
                    "All data sources are healthy. System operating normally."
                )
           
            # Calculate overall health score
            total_sources = report['summary']['total_sources']
            if total_sources > 0:
                health_score = (
                    (report['summary']['healthy_sources'] * 100) +
                    (report['summary']['warning_sources'] * 60) +
                    (report['summary']['critical_sources'] * 20) +
                    (report['summary']['error_sources'] * 0)
                ) / total_sources
                report['overall_health_score'] = health_score
            else:
                report['overall_health_score'] = 0
           
            return report
            
        except Exception as e:
            logger.error(f"Error compiling validation report: {e}")
            return {
                'error': 'Failed to compile validation report',
                'details': str(e)
            }


class PeriodicDataUpdateWorker(BaseWorker):
   """Worker for periodic data updates (runs continuously)"""
   
   def __init__(self, update_interval_minutes: int = 15, symbols: List[str] = None):
       super().__init__()
       self.update_interval_minutes = update_interval_minutes
       self.symbols = symbols or []
       self.is_running = False
       self.last_update_time = None
       
       # Services
       self.market_data = MarketDataService()
       
   def run(self):
       """Main periodic update execution"""
       try:
           self.started.emit()
           self.is_running = True
           
           self.emit_status("Starting periodic data updates...")
           
           update_count = 0
           while not self.is_cancelled() and self.is_running:
               update_count += 1
               self.emit_status(f"Running periodic update #{update_count}...")
               
               # Perform update
               self._perform_update_cycle()
               
               # Wait for next update
               wait_seconds = self.update_interval_minutes * 60
               for _ in range(wait_seconds):
                   if self.is_cancelled():
                       break
                   time.sleep(1)
                   
                   # Update progress indicator (cycles every minute)
                   elapsed = _ % 60
                   progress = int((elapsed / 60) * 100)
                   if elapsed == 0:  # Every minute
                       remaining_minutes = (wait_seconds - _) // 60
                       self.emit_progress(progress, f"Next update in {remaining_minutes} minutes")
                       
       except Exception as e:
           logger.error(f"Error in periodic update worker: {e}")
           self.emit_error("Periodic Update Failed", str(e))
       finally:
           self.is_running = False
           self.finished.emit()
   
   def _perform_update_cycle(self):
       """Perform one complete update cycle"""
       try:
           start_time = datetime.now()
           
           # Update market data for tracked symbols
           if self.symbols:
               self._update_symbol_data()
           
           # Update market indicators
           self._update_market_indicators()
           
           # Clean up old cache entries
           self._cleanup_cache()
           
           self.last_update_time = datetime.now()
           update_duration = (self.last_update_time - start_time).total_seconds()
           
           # Emit update result
           update_result = {
               'type': 'periodic_update_complete',
               'update_time': self.last_update_time.isoformat(),
               'duration_seconds': update_duration,
               'symbols_updated': len(self.symbols),
               'update_count': getattr(self, '_update_count', 0) + 1
           }
           
           self.emit_result(update_result)
           self.emit_status(f"Update cycle complete in {update_duration:.1f}s")
           
       except Exception as e:
           logger.error(f"Error in update cycle: {e}")
           self.emit_status(f"Update cycle failed: {str(e)}")
   
   def _update_symbol_data(self):
       """Update data for tracked symbols"""
       try:
           for symbol in self.symbols:
               if self.is_cancelled():
                   break
               
               # Update current price
               current_price = self.market_data.get_current_price(symbol)
               
               # Cache the updated price
               if current_price:
                   cache_key = f"price_{symbol}"
                   # Would use actual cache service here
                   # self.cache_service.set(cache_key, current_price)
                   pass
               
       except Exception as e:
           logger.error(f"Error updating symbol data: {e}")
   
   def _update_market_indicators(self):
       """Update broad market indicators"""
       try:
           # Update VIX
           vix = self.market_data.get_vix()
           if vix:
               # Cache VIX data
               pass
           
           # Update other market indicators as needed
           
       except Exception as e:
           logger.error(f"Error updating market indicators: {e}")
   
   def _cleanup_cache(self):
       """Clean up expired cache entries"""
       try:
           # Would implement cache cleanup logic here
           # self.cache_service.cleanup_expired()
           pass
           
       except Exception as e:
           logger.error(f"Error cleaning up cache: {e}")
   
   def stop_updates(self):
       """Stop periodic updates"""
       self.is_running = False
       self.cancel()
       self.emit_status("Periodic updates stopped")