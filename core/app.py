"""
Core Application - Using Your Existing Advanced Main Window
"""

from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import QTimer

class OptionsCalculatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Professional Options Calculator Pro v10.0')
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize your existing services first
        self.setup_services()
        
        # Use your existing main window
        self.setup_ui()
        
        self.show()
    
    def setup_services(self):
        """Initialize services using dependency injection container"""
        try:
            from services.container import get_container
            from services.interfaces import (
                IMarketDataService, IVolatilityService, IOptionsService,
                IMLService, IConfigService, ICacheService
            )
            from utils.error_handler import create_trading_error_handler
            
            # Get the global service container
            self.container = get_container()
            self.error_handler = create_trading_error_handler()
            print("✓ Service container initialized")
            
            # Resolve services through the container (no direct instantiation)
            self.config_manager = self.container.get_service(IConfigService)
            print("✓ ConfigManager resolved")
            
            self.cache_service = self.container.get_service(ICacheService)
            print("✓ Cache service resolved")
            
            self.market_data_service = self.container.get_service(IMarketDataService)
            print("✓ MarketDataService resolved")
            
            self.volatility_service = self.container.get_service(IVolatilityService)
            print("✓ VolatilityService resolved")
            
            self.options_service = self.container.get_service(IOptionsService)
            print("✓ OptionsService resolved")
            
            self.ml_service = self.container.get_service(IMLService)
            print("✓ MLService resolved")
            
            # Additional utilities (fallback to direct instantiation if not in container)
            try:
                from utils.greeks_calculator import GreeksCalculator
                self.greeks_calculator = GreeksCalculator(self.config_manager)
                print("✓ GreeksCalculator loaded")
            except Exception as e:
                print(f"GreeksCalculator fallback error: {e}")
            
            try:
                from utils.thread_manager import ThreadManager
                self.thread_manager = ThreadManager()
                print("✓ ThreadManager loaded")
            except Exception as e:
                print(f"ThreadManager fallback error: {e}")
            
            # Start background workers for calendar spread calculations
            from services.thread_workers import get_calendar_worker, get_monte_carlo_worker
            self.calendar_worker = get_calendar_worker()
            self.monte_carlo_worker = get_monte_carlo_worker()
            print("✓ Background workers started")
            
        except Exception as e:
            error_context = {
                'method': 'setup_services',
                'component': 'core.app'
            }
            
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_error(e, error_context)
            else:
                print(f"Service initialization error: {e}")
                import traceback
                traceback.print_exc()
                
            # Fallback to legacy direct instantiation if container fails
            self._setup_services_fallback()
    
    def setup_ui(self):
        """Use your existing main window"""
        try:
            from views.main_window import MainWindow
            self.main_widget = MainWindow(self)
            self.setCentralWidget(self.main_widget)
            print("✓ Your MainWindow loaded successfully")
            
            # Connect services to the main window
            if hasattr(self.main_widget, 'set_services'):
                self.main_widget.set_services(
                    market_data=self.market_data_service,
                    ml_service=self.ml_service,
                    volatility_service=self.volatility_service,
                    options_service=self.options_service,
                    greeks_calculator=self.greeks_calculator,
                    thread_manager=self.thread_manager
                )
                print("✓ Services connected to MainWindow")
            
        except Exception as e:
            print(f"MainWindow load error: {e}")
            print("Falling back to other views...")
            
            # Try your scanner_view (45KB - looks advanced!)
            try:
                from views.scanner_view import ScannerView
                self.scanner_widget = ScannerView(self)
                self.setCentralWidget(self.scanner_widget)
                print("✓ ScannerView loaded as fallback")
            except Exception as e2:
                print(f"ScannerView error: {e2}")
                
                # Last resort - use analysis_view
                try:
                    from views.analysis_view import AnalysisView
                    self.analysis_widget = AnalysisView(self)
                    self.setCentralWidget(self.analysis_widget)
                    print("✓ AnalysisView loaded as final fallback")
                except Exception as e3:
                    print(f"All views failed: {e3}")
    
    def _setup_services_fallback(self):
        """Fallback to direct service instantiation if container fails"""
        print("⚠️  Using fallback service initialization")
        try:
            from utils.config_manager import ConfigManager
            self.config_manager = ConfigManager()
            print("✓ ConfigManager loaded (fallback)")
            
            from services.market_data import MarketDataService
            self.market_data_service = MarketDataService()
            print("✓ MarketDataService loaded (fallback)")
            
            from services.ml_service import MLService
            self.ml_service = MLService(self.config_manager)
            print("✓ MLService loaded (fallback)")
            
            try:
                from services.volatility_service import VolatilityService
                self.volatility_service = VolatilityService(self.config_manager, self.market_data_service)
                print("✓ VolatilityService loaded (fallback)")
            except TypeError:
                # Try with just config_manager
                self.volatility_service = VolatilityService(self.config_manager)
                print("✓ VolatilityService loaded (fallback alt constructor)")
            
            from services.options_service import OptionsService
            self.options_service = OptionsService()
            print("✓ OptionsService loaded (fallback)")
            
            from utils.greeks_calculator import GreeksCalculator
            self.greeks_calculator = GreeksCalculator(self.config_manager)
            print("✓ GreeksCalculator loaded (fallback)")
            
            from utils.thread_manager import ThreadManager
            self.thread_manager = ThreadManager()
            print("✓ ThreadManager loaded (fallback)")
            
        except Exception as e:
            print(f"Fallback service error: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup(self):
        """Comprehensive cleanup method"""
        try:
            # Shutdown background workers
            if hasattr(self, 'calendar_worker'):
                from services.thread_workers import shutdown_all_workers
                shutdown_all_workers(timeout=10.0)
                print("✓ Background workers shutdown")
            
            # Cleanup thread manager
            if hasattr(self, 'thread_manager'):
                self.thread_manager.shutdown()
                print("✓ ThreadManager shutdown")
            
            # Close async services
            if hasattr(self, 'container'):
                from services.interfaces import IAsyncAPIService
                try:
                    async_service = self.container.get_service(IAsyncAPIService)
                    if hasattr(async_service, 'close'):
                        async_service.close()
                        print("✓ Async services closed")
                except:
                    pass  # Service may not be registered
                
                # Clear the container
                self.container.clear()
                print("✓ Service container cleared")
            
            print("✓ Cleanup complete")
            
        except Exception as e:
            error_context = {
                'method': 'cleanup',
                'component': 'core.app'
            }
            
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_error(e, error_context)
            else:
                print(f"Cleanup error: {e}")
    
    def closeEvent(self, event):
        self.cleanup()
        event.accept()
