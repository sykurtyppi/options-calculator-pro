"""
Analysis View - Working Version with Real-time Market Data
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QLineEdit, QTextEdit, QGridLayout)
from PySide6.QtCore import Qt, QTimer

class AnalysisView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setup_ui()
        
        # Force immediate market data update
        QTimer.singleShot(1000, self.force_market_update)
        
        # Setup regular updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.force_market_update) 
        self.timer.start(5000)  # Every 5 seconds
    
    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout()
        
        # Market Overview
        market_layout = QGridLayout()
        market_layout.addWidget(QLabel('Market Overview'), 0, 0, 1, 4)
        
        # VIX
        market_layout.addWidget(QLabel('VIX:'), 1, 0)
        self.vix_label = QLabel('Loading...')
        market_layout.addWidget(self.vix_label, 1, 1)
        
        # SPX  
        market_layout.addWidget(QLabel('SPX:'), 1, 2)
        self.spx_label = QLabel('Loading...')
        market_layout.addWidget(self.spx_label, 1, 3)
        
        # NDX
        market_layout.addWidget(QLabel('NDX:'), 2, 2) 
        self.ndx_label = QLabel('Loading...')
        market_layout.addWidget(self.ndx_label, 2, 3)
        
        main_layout.addLayout(market_layout)
        
        # Symbol Input
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(QLabel('Stock Symbol:'))
        self.symbol_input = QLineEdit()
        self.symbol_input.setText('AAPL')
        symbol_layout.addWidget(self.symbol_input)
        
        # Analyze Button
        self.analyze_button = QPushButton('Analyze Options')
        self.analyze_button.clicked.connect(self.do_analysis)
        symbol_layout.addWidget(self.analyze_button)
        
        main_layout.addLayout(symbol_layout)
        
        # Results
        self.results = QTextEdit()
        main_layout.addWidget(self.results)
        
        # Status
        self.status = QLabel('Ready')
        main_layout.addWidget(self.status)
        
        self.setLayout(main_layout)
    
    def force_market_update(self):
        """Force market data update using yfinance directly"""
        try:
            import yfinance as yf
            
            # Get VIX
            vix_ticker = yf.Ticker('^VIX')
            vix_data = vix_ticker.history(period='1d')
            if not vix_data.empty:
                vix_price = vix_data['Close'].iloc[-1]
                self.vix_label.setText(f'{vix_price:.2f}')
            
            # Get SPX
            spx_ticker = yf.Ticker('^GSPC') 
            spx_data = spx_ticker.history(period='1d')
            if not spx_data.empty:
                spx_price = spx_data['Close'].iloc[-1]
                self.spx_label.setText(f'{spx_price:,.2f}')
                
            # Get NDX
            ndx_ticker = yf.Ticker('^NDX')
            ndx_data = ndx_ticker.history(period='1d')
            if not ndx_data.empty:
                ndx_price = ndx_data['Close'].iloc[-1] 
                self.ndx_label.setText(f'{ndx_price:,.2f}')
                
            print(f"Market updated - VIX: {vix_price:.2f}, SPX: {spx_price:.2f}")
            
        except Exception as e:
            print(f"Error updating market data: {e}")
    
    def do_analysis(self):
        """Run comprehensive analysis using your existing analysis_worker"""
        try:
            symbol = self.symbol_input.text().strip().upper()
            if not symbol:
                self.status.setText('Enter a symbol')
                return
                
            self.status.setText(f'Running advanced analysis for {symbol}...')
            self.results.clear()
            
            # Import and initialize your analysis worker
            from core.workers.analysis_worker import AnalysisWorker
            
            self.results.append(f'=== PROFESSIONAL ANALYSIS FOR {symbol} ===
')
            self.results.append('Initializing advanced calculation engines...')
            
            # Create analysis worker instance
            worker = AnalysisWorker()
            
            # Basic stock data first
            self.results.append('
--- BASIC STOCK DATA ---')
            try:
                basic_data = worker.get_basic_stock_data(symbol)
                if basic_data:
                    for key, value in basic_data.items():
                        self.results.append(f'{key}: {value}')
            except Exception as e:
                self.results.append(f'Basic data error: {e}')
            
            # Heston Model Analysis
            self.results.append('
--- HESTON VOLATILITY MODEL ---')
            try:
                heston_results = worker.run_heston_simulation(symbol)
                if heston_results:
                    for key, value in heston_results.items():
                        if isinstance(value, (int, float)):
                            self.results.append(f'{key}: {value:.4f}')
                        else:
                            self.results.append(f'{key}: {value}')
            except Exception as e:
                self.results.append(f'Heston model error: {e}')
            
            # Yang-Zhang Volatility
            self.results.append('
--- YANG-ZHANG VOLATILITY ---')
            try:
                # Use your volatility service
                from services.volatility_service import VolatilityService
                vol_service = VolatilityService()
                yz_vol = vol_service._yang_zhang_volatility(symbol)
                if yz_vol is not None:
                    self.results.append(f'Yang-Zhang Volatility: {yz_vol:.4f}')
            except Exception as e:
                self.results.append(f'Yang-Zhang error: {e}')
            
            # Monte Carlo Simulation
            self.results.append('
--- MONTE CARLO SIMULATION ---')
            try:
                mc_results = worker.run_monte_carlo_analysis(symbol)
                if mc_results:
                    for key, value in mc_results.items():
                        if isinstance(value, (int, float)):
                            self.results.append(f'{key}: {value:.4f}')
                        else:
                            self.results.append(f'{key}: {value}')
            except Exception as e:
                self.results.append(f'Monte Carlo error: {e}')
            
            # Black-Scholes Greeks
            self.results.append('
--- BLACK-SCHOLES GREEKS ---')
            try:
                from utils.greeks_calculator import GreeksCalculator
                from utils.config_manager import ConfigManager
                config = ConfigManager()
                greeks_calc = GreeksCalculator(config)
                
                # Calculate Greeks for ATM option
                greeks = greeks_calc._calculate_black_scholes_greeks(symbol)
                if greeks:
                    for key, value in greeks.items():
                        if isinstance(value, (int, float)):
                            self.results.append(f'{key}: {value:.4f}')
                        else:
                            self.results.append(f'{key}: {value}')
            except Exception as e:
                self.results.append(f'Greeks calculation error: {e}')
            
            # Kelly Criterion Position Sizing
            self.results.append('
--- KELLY CRITERION SIZING ---')
            try:
                kelly_results = worker.calculate_kelly_sizing(symbol)
                if kelly_results:
                    for key, value in kelly_results.items():
                        if isinstance(value, (int, float)):
                            self.results.append(f'{key}: {value:.4f}')
                        else:
                            self.results.append(f'{key}: {value}')
            except Exception as e:
                self.results.append(f'Kelly sizing error: {e}')
            
            # Machine Learning Prediction
            self.results.append('
--- ML PREDICTION ---')
            try:
                ml_results = worker.get_ml_prediction(symbol)
                if ml_results:
                    for key, value in ml_results.items():
                        self.results.append(f'{key}: {value}')
            except Exception as e:
                self.results.append(f'ML prediction error: {e}')
                
            self.status.setText(f'Advanced analysis complete for {symbol}')
            
        except Exception as e:
            error_msg = f'Analysis error: {e}'
            self.results.append(error_msg)
            self.status.setText('Analysis failed')
            print(error_msg)

    