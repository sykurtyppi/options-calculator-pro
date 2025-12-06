from typing import Dict, Any, Optional
from datetime import datetime

class ScanCriteria:
    def __init__(self, min_price=0, max_price=1000, min_volume=0):
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume

class ScanResult:
    def __init__(self, ticker: str, score: float = 0.0, signals: list = None):
        self.ticker = ticker
        self.score = score
        self.signals = signals or []
        self.timestamp = datetime.now()

class ScanResult:
    def __init__(self, ticker: str, score: float = 0.0, signals: list = None):
        self.ticker = ticker
        self.score = score
        self.signals = signals or []
        self.timestamp = None

class ScanCriteria:
    def __init__(self, min_price=0, max_price=1000, min_volume=0):
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
from dataclasses import dataclass

@dataclass
class ScanCriteria:
    min_price: float = 0.0
    max_price: float = 1000.0
    min_volume: int = 0
class ScanCriteria:
    """Basic scan criteria class"""
    def __init__(self):
        self.min_price = 0
        self.max_price = 1000
        self.min_volume = 0
        
from PySide6.QtCore import Signal, QObject
from PySide6.QtCore import Signal
# Asyncio scanner controller here
# Signals
progress_updated = Signal(int, int, str)  # current, total, ticker
result_found = Signal(object)  # ScanResult
scan_completed = Signal(list)  # list[ScanResult]
error_occurred = Signal(str, str)  # ticker, error_message

def __init__(self, tickers: list[str], criteria: ScanCriteria, 
             recommendation_engine, parent=None):
    super().__init__(parent)
    self.tickers = tickers
    self.criteria = criteria
    self.recommendation_engine = recommendation_engine
    self.should_stop = False
    self.results = []

def stop(self):
    """Stop the scanning process"""
    self.should_stop = True

def run(self):
    """Main scanning loop"""
    try:
        total_tickers = len(self.tickers)
        self.results = []
        
        for i, ticker in enumerate(self.tickers):
            if self.should_stop:
                break
            
            self.progress_updated.emit(i + 1, total_tickers, ticker)
            
            try:
                # Get analysis result
                analysis_result = self.recommendation_engine.compute_recommendation(ticker)
                
                if isinstance(analysis_result, str):
                    # Error occurred
                    self.error_occurred.emit(ticker, analysis_result)
                    continue
                
                if not isinstance(analysis_result, dict):
                    self.error_occurred.emit(ticker, "Invalid analysis result format")
                    continue
                
                # Check if meets criteria
                if self._meets_criteria(analysis_result):
                    scan_result = self._create_scan_result(ticker, analysis_result)
                    self.results.append(scan_result)
                    self.result_found.emit(scan_result)
            
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                self.error_occurred.emit(ticker, str(e))
        
        # Sort results by confidence (highest first)
        self.results.sort(key=lambda x: x.confidence, reverse=True)
        
        if not self.should_stop:
            self.scan_completed.emit(self.results)
            
    except Exception as e:
        logger.error(f"Critical error in scanner worker: {e}")
        self.error_occurred.emit("SCANNER", f"Critical scanner error: {e}")

def _meets_criteria(self, result: Dict[str, Any]) -> bool:
    """Check if analysis result meets scan criteria"""
    try:
        confidence = result.get("confidence", 0)
        iv_rv_ratio = result.get("iv30_rv30_value", 0)
        days_to_earnings = result.get("days_to_earnings", 999)
        volume = result.get("avg_volume_value", 0)
        current_price = result.get("underlying_price", 0)
        sector = result.get("sector", "")
        
        # Basic criteria checks
        if confidence < self.criteria.min_confidence:
            return False
        
        if iv_rv_ratio > self.criteria.max_iv_rv_ratio:
            return False
        
        if not (self.criteria.min_days_to_earnings <= days_to_earnings <= self.criteria.max_days_to_earnings):
            return False
        
        if volume < self.criteria.min_volume:
            return False
        
        if not (self.criteria.min_price <= current_price <= self.criteria.max_price):
            return False
        
        # VIX check (get from result)
        vix = result.get("vix_value", 20)
        if vix > self.criteria.max_vix:
            return False
        
        # Sector filters
        if self.criteria.sectors and sector not in self.criteria.sectors:
            return False
        
        if self.criteria.exclude_sectors and sector in self.criteria.exclude_sectors:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking criteria: {e}")
        return False

def _create_scan_result(self, ticker: str, analysis_result: Dict[str, Any]) -> ScanResult:
    """Create ScanResult from analysis result"""
    confidence = analysis_result.get("confidence", 0)
    
    # Determine recommendation
    if confidence >= 70:
        recommendation = "STRONG BUY"
    elif confidence >= 60:
        recommendation = "BUY"
    elif confidence >= 50:
        recommendation = "CONSIDER"
    else:
        recommendation = "AVOID"
    
    return ScanResult(
        ticker=ticker,
        confidence=confidence,
        iv_rv_ratio=analysis_result.get("iv30_rv30_value", 0),
        days_to_earnings=analysis_result.get("days_to_earnings", 0),
        current_price=analysis_result.get("underlying_price", 0),
        expected_move=analysis_result.get("expected_move", "N/A"),
        volume=analysis_result.get("avg_volume_value", 0),
        sector=analysis_result.get("sector", "Unknown"),
        recommendation=recommendation,
        analysis_result=analysis_result
    )  
class ScannerController(QObject):
    """Controller for stock scanning operations""" 
    scan_started = Signal()  # Signals
    scan_progress = Signal(int, int, str)  # current, total, ticker
    scan_completed = Signal(list)  # list[ScanResult]
    scan_stopped = Signal()
    opportunity_found = Signal(object)  # ScanResult
    error_occurred = Signal(str, str)  # title, message

    def __init__(self, config_manager, recommendation_engine):
        super().__init__()
        self.config_manager = config_manager
        self.recommendation_engine = recommendation_engine
        self.current_worker = None
        self.scan_results = []
    
        # Default criteria
        self.default_criteria = ScanCriteria()
    
    def start_scan(self, tickers: list[str], criteria: ScanCriteria = None):
        """Start scanning process"""
        if self.is_scanning():
            self.error_occurred.emit("Scanner Error", "Scan already in progress")
            return
    
        if not tickers:
            self.error_occurred.emit("Scanner Error", "No tickers provided for scanning")
            return
    
        if criteria is None:
            criteria = self.default_criteria
    
        logger.info(f"Starting scan of {len(tickers)} tickers")
    
        # Create and start worker
        self.current_worker = ScannerWorker(tickers, criteria, self.recommendation_engine)
    
        # Connect signals
        self.current_worker.progress_updated.connect(self.scan_progress)
        self.current_worker.result_found.connect(self.opportunity_found)
        self.current_worker.scan_completed.connect(self._on_scan_completed)
        self.current_worker.error_occurred.connect(self._on_scan_error)
        self.current_worker.finished.connect(self._cleanup_worker)
    
        self.current_worker.start()
        self.scan_started.emit()

    def stop_scan(self):
        """Stop current scanning process"""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
            self.current_worker.wait(5000)  # Wait up to 5 seconds
        
            if self.current_worker.isRunning():
                self.current_worker.terminate()
                self.current_worker.wait()
        
            self.scan_stopped.emit()
            logger.info("Scan stopped by user")

    def is_scanning(self) -> bool:
        """Check if scan is currently running"""
        return self.current_worker is not None and self.current_worker.isRunning()

    def get_last_results(self) -> list[ScanResult]:
        """Get results from last scan"""
        return self.scan_results.copy()

    def _on_scan_completed(self, results: list[ScanResult]):
        """Handle scan completion"""
        self.scan_results = results
        self.scan_completed.emit(results)
        logger.info(f"Scan completed. Found {len(results)} opportunities")

    def _on_scan_error(self, ticker: str, error_message: str):
        """Handle individual ticker scan error"""
        logger.warning(f"Scan error for {ticker}: {error_message}")
        # Could emit individual error signals here if needed

    def _cleanup_worker(self):
        """Clean up worker thread"""
        if self.current_worker:
            self.current_worker.deleteLater()
            self.current_worker = None

    def export_results(self, file_path: str, results: list[ScanResult] = None) -> bool:
        """Export scan results to CSV"""
        try:
            if results is None:
                results = self.scan_results
        
            if not results:
                self.error_occurred.emit("Export Error", "No results to export")
                return False
        
            # Convert to DataFrame-like structure
            import pandas as pd
        
            data = []
            for result in results:
                data.append({
                    "Ticker": result.ticker,
                    "Confidence": f"{result.confidence:.1f}%",
                    "Recommendation": result.recommendation,
                    "IV/RV Ratio": f"{result.iv_rv_ratio:.2f}",
                    "Days to Earnings": result.days_to_earnings,
                    "Current Price": f"${result.current_price:.2f}",
                    "Expected Move": result.expected_move,
                    "Volume": f"${result.volume/1000000:.1f}M",
                    "Sector": result.sector,
                    "Scan Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
        
            logger.info(f"Scan results exported to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting scan results: {e}")
            self.error_occurred.emit("Export Error", f"Failed to export results: {e}")
            return False

    def create_watchlist_from_results(self, results: list[ScanResult] = None, 
                                    min_confidence: float = 60.0) -> list[str]:
        """Create watchlist from scan results"""
        try:
            if results is None:
                results = self.scan_results
        
            # Filter by minimum confidence
            filtered_results = [r for r in results if r.confidence >= min_confidence]
        
            # Extract tickers
            watchlist = [result.ticker for result in filtered_results]
        
            logger.info(f"Created watchlist with {len(watchlist)} tickers")
            return watchlist
        
        except Exception as e:
            logger.error(f"Error creating watchlist: {e}")
            return []

    def get_sector_summary(self, results: list[ScanResult] = None) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by sector"""
        try:
            if results is None:
                results = self.scan_results
        
            if not results:
                return {}
        
            sector_stats = {}
        
            for result in results:
                sector = result.sector
                if sector not in sector_stats:
                    sector_stats[sector] = {
                        "count": 0,
                        "avg_confidence": 0,
                        "avg_iv_rv": 0,
                        "tickers": []
                    }
            
                stats = sector_stats[sector]
                stats["count"] += 1
                stats["avg_confidence"] += result.confidence
                stats["avg_iv_rv"] += result.iv_rv_ratio
                stats["tickers"].append(result.ticker)
        
            # Calculate averages
            for sector, stats in sector_stats.items():
                count = stats["count"]
                stats["avg_confidence"] = stats["avg_confidence"] / count
                stats["avg_iv_rv"] = stats["avg_iv_rv"] / count
        
            return sector_stats
        
        except Exception as e:
            logger.error(f"Error calculating sector summary: {e}")
            return {}

    def get_predefined_ticker_lists(self) -> Dict[str, list[str]]:
        """Get predefined ticker lists for scanning"""
        return {
            "S&P 100": [
                "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", 
                "BRK.B", "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "CVX", 
                "MA", "ABBV", "PFE", "AVGO", "BAC", "COST", "DIS", "ADBE", 
                "CRM", "NFLX", "ACN", "VZ", "KO", "MRK", "ABT", "PEP", "TMO",
                    "CSCO", "LIN", "WFC", "AMD", "NKE", "ORCL", "DHR", "QCOM", "TXN",
                    "PM", "UPS", "NEE", "RTX", "LOW", "INTC", "COP", "HON", "UNP",
                    "IBM", "SBUX", "T", "AMAT", "LMT", "CAT", "DE", "SPGI", "MDT",
                    "GS", "BLK", "ELV", "AXP", "BKNG", "GILD", "CVS", "ISRG", "ADP",
                    "BA", "MMC", "SYK", "TJX", "LRCX", "MDLZ", "MO", "ZTS", "CB",
                    "CI", "SCHW", "SO", "FIS", "DUK", "BSX", "MU", "ITW", "CSX",
                    "CME", "AON", "MMM", "EMR", "HCA", "NSC", "USB", "PNC", "ICE"
                ],                                                                                                                                                                    "High Volume Tech": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD",
                "INTC", "ORCL", "CRM", "ADBE", "NFLX", "PYPL", "UBER", "LYFT",
                "SPOT", "SQ", "ZOOM", "DOCU", "OKTA", "SNOW", "PLTR", "RBLX"
            ],
        
            "High IV Stocks": [
                "TSLA", "AMD", "NVDA", "NFLX", "AMZN", "META", "GOOGL", "UBER",
                "LYFT", "ROKU", "TWLO", "ZM", "PTON", "DASH", "COIN", "RBLX",
                "GME", "AMC", "SPCE", "PLTR", "TLRY", "SNDL", "BB", "NOK"
            ],
        
            "Earnings Favorites": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
                "JPM", "BAC", "WFC", "GS", "JNJ", "PFE", "MRK", "UNH", "CVS",
                "WMT", "TGT", "HD", "LOW", "COST", "NKE", "SBUX", "MCD", "DIS"
            ],
        
            "Biotech": [
                "GILD", "BIIB", "REGN", "VRTX", "AMGN", "CELG", "ALXN", "BMRN",
                "SGEN", "MRNA", "BNTX", "NVAX", "INO", "OCGN", "SRNE", "KDMN"
            ],
        
            "Energy": [
                "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "OXY", "DVN", "FANG",
                "MRO", "APA", "CNX", "EQT", "AR", "CLR", "WLL", "SM", "NOG"
            ],
        
            "Custom List": self.config_manager.get("favorite_stocks", [])
        }

    def     get_earnings_calendar_tickers(self, days_ahead: int = 30) -> list[str]:
        """Get tickers with earnings in the next N days"""
        try:
            # This would typically connect to an earnings calendar API
            # For now, return a subset of common tickers
            # In a real implementation, you'd fetch from earnings calendar services
        
            common_earnings_tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
                "JPM", "BAC", "JNJ", "PFE", "WMT", "HD", "DIS", "SBUX", "NKE"
            ]
        
            return common_earnings_tickers
        
        except Exception as e:
            logger.error(f"Error getting earnings calendar: {e}")
            return []

    def validate_tickers(self, tickers: list[str]) -> Dict[str, bool]:
        """Validate ticker symbols"""
        try:
            validation_results = {}
        
            for ticker in tickers:
                # Basic validation
                if not ticker or not isinstance(ticker, str):
                    validation_results[ticker] = False
                    continue
            
                ticker = ticker.strip().upper()
            
                # Check format (basic)
                if len(ticker) > 10 or not ticker.isalpha():
                    validation_results[ticker] = False
                    continue
            
                # Mark as valid (in real implementation, you might check against a symbol list)
                validation_results[ticker] = True
        
            return validation_results
        
        except Exception as e:
            logger.error(f"Error validating tickers: {e}")
            return {}

    def get_scan_templates(self) -> Dict[str, ScanCriteria]:
        """Get predefined scan templates"""
        return {
            "Conservative": ScanCriteria(
                min_confidence=65.0,
                max_iv_rv_ratio=1.5,
                min_days_to_earnings=3,
                max_days_to_earnings=10,
                min_volume=100000000,
                max_vix=25.0
            ),
        
            "Aggressive": ScanCriteria(
                min_confidence=50.0,
                max_iv_rv_ratio=3.0,
                min_days_to_earnings=1,
                max_days_to_earnings=7,
                min_volume=25000000,
                max_vix=40.0
            ),
        
            "High Probability": ScanCriteria(
                min_confidence=70.0,
                max_iv_rv_ratio=2.0,
                min_days_to_earnings=2,
                max_days_to_earnings=14,
                min_volume=75000000,
                max_vix=30.0
            ),
        
            "Earnings Play": ScanCriteria(
                min_confidence=55.0,
                max_iv_rv_ratio=2.5,
                min_days_to_earnings=1,
                max_days_to_earnings=5,
                min_volume=50000000,
                max_vix=35.0
            ),
        
            "Tech Focus": ScanCriteria(
                min_confidence=60.0,
                max_iv_rv_ratio=2.0,
                min_days_to_earnings=2,
                max_days_to_earnings=12,
                min_volume=100000000,
                sectors=["Technology", "Communication Services"],
                max_vix=32.0
            )
        }

    def save_custom_criteria(self, name: str, criteria: ScanCriteria) -> bool:
        """Save custom scan criteria"""
        try:
            custom_criteria = self.config_manager.get("custom_scan_criteria", {})
        
            # Convert criteria to dict for storage
            criteria_dict = {
                "min_confidence": criteria.min_confidence,
                "max_iv_rv_ratio": criteria.max_iv_rv_ratio,
                "min_days_to_earnings": criteria.min_days_to_earnings,
                "max_days_to_earnings": criteria.max_days_to_earnings,
                "min_volume": criteria.min_volume,
                "max_vix": criteria.max_vix,
                "sectors": criteria.sectors,
                "exclude_sectors": criteria.exclude_sectors,
                "min_price": criteria.min_price,
                "max_price": criteria.max_price,
            }
        
            custom_criteria[name] = criteria_dict
            self.config_manager.set("custom_scan_criteria", custom_criteria)
            self.config_manager.save_config()
        
            logger.info(f"Saved custom scan criteria: {name}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving custom criteria: {e}")
            return False

    def load_custom_criteria(self, name: str) -> Optional[ScanCriteria]:
        """Load custom scan criteria"""
        try:
            custom_criteria = self.config_manager.get("custom_scan_criteria", {})
        
            if name not in custom_criteria:
                return None
        
            criteria_dict = custom_criteria[name]
        
            # Convert dict back to ScanCriteria
            criteria = ScanCriteria(
                min_confidence=criteria_dict.get("min_confidence", 50.0),
                max_iv_rv_ratio=criteria_dict.get("max_iv_rv_ratio", 2.0),
                min_days_to_earnings=criteria_dict.get("min_days_to_earnings", 1),
                max_days_to_earnings=criteria_dict.get("max_days_to_earnings", 14),
                min_volume=criteria_dict.get("min_volume", 50000000),
                max_vix=criteria_dict.get("max_vix", 35.0),
                sectors=criteria_dict.get("sectors"),
                exclude_sectors=criteria_dict.get("exclude_sectors"),
                min_price=criteria_dict.get("min_price", 10.0),
                max_price=criteria_dict.get("max_price", 1000.0),
            )
        
            return criteria
        
        except Exception as e:
            logger.error(f"Error loading custom criteria: {e}")
            return None

    def delete_custom_criteria(self, name: str) -> bool:
        """Delete custom scan criteria"""
        try:
            custom_criteria = self.config_manager.get("custom_scan_criteria", {})
        
            if name in custom_criteria:
                del custom_criteria[name]
                self.config_manager.set("custom_scan_criteria", custom_criteria)
                self.config_manager.save_config()
                logger.info(f"Deleted custom scan criteria: {name}")
                return True
        
            return False
        
        except Exception as e:
            logger.error(f"Error deleting custom criteria: {e}")
            return False