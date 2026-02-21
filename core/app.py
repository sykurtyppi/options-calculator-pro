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

        # Initialize background workers after all services are ready
        self._init_background_workers()

        self.show()

    def setup_services(self):
        """Initialize services using dependency injection container with fallback"""
        try:
            print("⚠️  Temporarily bypassing container for immediate functionality")
            self._setup_services_fallback()
            return

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

            # Background workers will be initialized later to avoid circular imports
            self.calendar_worker = None
            self.monte_carlo_worker = None
            print("✓ Background workers deferred")

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
            # CRITICAL FIX: Create MainWindow with proper parent reference
            self.main_widget = MainWindow(parent=self)
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

            # Initialize analysis controller for handling analysis requests
            try:
                from controllers.analysis_controller import AnalysisController
                # CRITICAL FIX: Pass self as parent to prevent garbage collection
                self.analysis_controller = AnalysisController(
                    market_data_service=self.market_data_service,
                    config_manager=self.config_manager,
                    ml_service=self.ml_service,
                    thread_manager=self.thread_manager,
                    parent=self  # CRITICAL: Set parent to prevent garbage collection
                )

                # Connect MainWindow analysis signals to analysis controller
                if hasattr(self.main_widget, 'analysis_requested'):
                    self.main_widget.analysis_requested.connect(self.analysis_controller.analyze_symbol)
                    print("✓ Single analysis signal connected")

                if hasattr(self.main_widget, 'batch_analysis_requested'):
                    self.main_widget.batch_analysis_requested.connect(self._handle_batch_analysis)
                    print("✓ Batch analysis signal connected")

                # Connect analysis controller results back to main window
                try:
                    if hasattr(self.main_widget, 'display_analysis_result'):
                        self.analysis_controller.analysis_completed.connect(self.main_widget.display_analysis_result)
                        print("✓ Analysis results signal connected")

                    # Connect analysis progress updates
                    if hasattr(self.main_widget, '_update_status'):
                        self.analysis_controller.analysis_started.connect(
                            lambda symbol: self.main_widget._update_status(f"Analyzing {symbol}...")
                        )
                        self.analysis_controller.analysis_progress.connect(
                            lambda symbol, progress, status: self.main_widget._update_status(f"{status} ({progress}%)")
                        )
                        self.analysis_controller.analysis_error.connect(
                            lambda symbol, error: self.main_widget._update_status(f"Analysis error for {symbol}: {error}")
                        )
                        print("✓ Analysis progress signals connected")
                except RuntimeError as e:
                    print(f"Signal connection warning: {e}")
                    print("Analysis will work but result display may need fallback")

                print("✓ Analysis controller initialized and connected with proper parent")

            except Exception as e:
                print(f"Analysis controller initialization error: {e}")
                import traceback
                traceback.print_exc()

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

    def _init_background_workers(self):
        """Initialize background workers after services are set up"""
        try:
            from services.thread_workers import get_calendar_worker, get_monte_carlo_worker
            self.calendar_worker = get_calendar_worker()
            self.monte_carlo_worker = get_monte_carlo_worker()
            print("✓ Background workers initialized")
        except Exception as e:
            print(f"Background worker initialization error: {e}")
            self.calendar_worker = None
            self.monte_carlo_worker = None

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
            self.options_service = OptionsService(self.config_manager, self.market_data_service)
            print("✓ OptionsService loaded (fallback)")

            from utils.greeks_calculator import GreeksCalculator
            self.greeks_calculator = GreeksCalculator(self.config_manager)
            print("✓ GreeksCalculator loaded (fallback)")

            from utils.thread_manager import ThreadManager
            self.thread_manager = ThreadManager(self.config_manager)
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

    def _handle_batch_analysis(self, symbols, parameters):
        """Handle batch analysis request"""
        try:
            if hasattr(self, 'analysis_controller'):
                # For now, analyze symbols one by one
                for symbol in symbols:
                    contracts = parameters.get('contracts', 1)
                    debit = parameters.get('debit_override', 0.0)
                    self.analysis_controller.analyze_symbol(symbol, contracts, debit)

                print(f"Started batch analysis for {len(symbols)} symbols")
            else:
                print("Analysis controller not available for batch analysis")
        except Exception as e:
            print(f"Batch analysis error: {e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        self.cleanup()
        event.accept()