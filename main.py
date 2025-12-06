#!/usr/bin/env python3
"""
Professional Options Calculator Pro
Main Entry Point

Author: Tristan Alejandro
Version: 10.0.0 (PySide6 Edition)
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QFont

from core.app import OptionsCalculatorApp
from utils.config_manager import ConfigManager
from utils.logger import setup_logger
from services.market_data import MarketDataService


def setup_application():
    """Setup QApplication with professional styling"""
    app = QApplication(sys.argv)
    
    # Application properties
    app.setApplicationName("Options Calculator Pro")
    app.setApplicationVersion("10.0.0")
    app.setOrganizationName("Professional Trading Tools")
    app.setOrganizationDomain("optionscalc.pro")
    
    # High DPI support
    # app.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Deprecated in Qt 6
    # app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Deprecated in Qt 6
    
    # Professional dark theme
    app.setStyle("Fusion")
    
    # Dark palette
    from PySide6.QtGui import QPalette, QColor
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)
    
    return app


def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'PySide6',
        'yfinance', 
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'sklearn'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print("Missing required dependencies:")
        for module in missing:
            print(f"  - {module}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    return True


def run_startup_tests():
    """Run startup diagnostic tests"""
    print("\n" + "="*60)
    print("PROFESSIONAL OPTIONS CALCULATOR PRO v10.0")
    print("PySide6 Edition - Startup Diagnostics")
    print("="*60)
    
    # Test 1: Dependencies
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    if not deps_ok:
        return False
    print("✓ All dependencies available")
    
    # Test 2: Market data connectivity
    print("Testing market data connectivity...")
    try:
        market_service = MarketDataService()
        test_price = market_service.get_current_price("AAPL")
        if test_price and test_price > 0:
            print(f"✓ Market data OK - AAPL: ${test_price:.2f}")
        else:
            print("⚠ Market data limited - using fallback mode")
    except Exception as e:
        print(f"⚠ Market data error: {e}")
        print("Application will run in offline mode")
    
    # Test 3: Configuration
    print("Loading configuration...")
    try:
        config = ConfigManager()
        print("✓ Configuration loaded")
    except Exception as e:
        print(f"⚠ Config error: {e}")
    
    print("="*60)
    print("Startup diagnostics complete. Launching application...")
    print("="*60)
    
    return True


def main():
    """Main application entry point"""
    try:
        # Setup logging first
        logger = setup_logger()
        logger.info("Professional Options Calculator Pro v10.0 starting...")
        
        # Run startup tests
        if not run_startup_tests():
            print("Startup tests failed. Please fix issues and try again.")
            return 1
        
        # Setup Qt Application
        app = setup_application()
        
        # Create main application window
        main_window = OptionsCalculatorApp()
        main_window.show()
        
        # Setup graceful shutdown
        def cleanup():
            logger.info("Application shutting down...")
            main_window.cleanup()
        
        app.aboutToQuit.connect(cleanup)
        
        # Start the application event loop
        logger.info("Application launched successfully")
        return app.exec()
        
    except Exception as e:
        print(f"Critical startup error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to log error if logger is available
        try:
            logger.error(f"Critical startup error: {e}")
            logger.error(traceback.format_exc())
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())