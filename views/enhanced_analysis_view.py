"""
Enhanced Analysis View with Advanced Options Features
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QLineEdit, QTextEdit, QGridLayout,
                               QTabWidget, QProgressBar, QSpinBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, QTimer, QThread, pyqtSignal
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class AdvancedAnalysisWorker(QThread):
    analysis_complete = pyqtSignal(dict)
    progress_update = pyqtSignal(int)
    
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol
        
    def run(self):
        """Run advanced analysis in background thread"""
        try:
            results = {}
            
            # Get extended historical data
            self.progress_update.emit(10)
            ticker = yf.Ticker(self.symbol)
            hist_1y = ticker.history(period="1y")
            hist_30d = ticker.history(period="30d")
            
            if hist_1y.empty:
                return
                
            self.progress_update.emit(25)
            
            # Time Series Analysis
            results['ts_analysis'] = self.calculate_time_series_metrics(hist_1y)
            
            self.progress_update.emit(40)
            
            # Volatility Analysis
            results['volatility'] = self.calculate_volatility_metrics(hist_1y, hist_30d)
            
            self.progress_update.emit(60)
            
            # Volume Analysis
            results['volume'] = self.calculate_volume_metrics(hist_1y)
            
            self.progress_update.emit(80)
            
            # Monte Carlo Simulation
            results['monte_carlo'] = self.run_monte_carlo_simulation(hist_30d)
            
            self.progress_update.emit(90)
            
            # ML Predictions (basic trend prediction)
            results['ml_prediction'] = self.calculate_ml_prediction(hist_1y)
            
            self.progress_update.emit(100)
            
            self.analysis_complete.emit(results)
            
        except Exception as e:
            print(f"Analysis error: {e}")
            self.analysis_complete.emit({})
    
    def calculate_time_series_metrics(self, data):
        """Calculate time series metrics"""
        try:
            prices = data['Close']
            returns = prices.pct_change().dropna()
            
            # Calculate slope (linear regression)
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices, 1)
            
            # Moving averages
            ma_20 = prices.rolling(window=20).mean().iloc[-1]
            ma_50 = prices.rolling(window=50).mean().iloc[-1]
            
            return {
                'slope': slope,
                'trend': 'Bullish' if slope > 0 else 'Bearish',
                'ma_20': ma_20,
                'ma_50': ma_50,
                'ma_signal': 'Buy' if ma_20 > ma_50 else 'Sell',
                'momentum': returns.rolling(window=14).mean().iloc[-1] * 100
            }
        except:
            return {}
    
    def calculate_volatility_metrics(self, data_1y, data_30d):
        """Calculate volatility metrics"""
        try:
            returns_1y = data_1y['Close'].pct_change().dropna()
            returns_30d = data_30d['Close'].pct_change().dropna()
            
            # Realized Volatility (annualized)
            rv_30d = returns_30d.std() * np.sqrt(252) * 100
            rv_1y = returns_1y.std() * np.sqrt(252) * 100
            
            # Implied Volatility estimate (simplified)
            current_price = data_30d['Close'].iloc[-1]
            price_range = data_30d['High'].max() - data_30d['Low'].min()
            iv_estimate = (price_range / current_price) * np.sqrt(252/30) * 100
            
            return {
                'rv_30d': rv_30d,
                'rv_1y': rv_1y,
                'iv_estimate': iv_estimate,
                'vol_regime': 'High' if rv_30d > 30 else 'Normal' if rv_30d > 15 else 'Low'
            }
        except:
            return {}
    
    def calculate_volume_metrics(self, data):
        """Calculate volume metrics"""
        try:
            volume = data['Volume']
            avg_volume_20d = volume.rolling(window=20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            
            volume_ratio = current_volume / avg_volume_20d
            
            return {
                'avg_volume_20d': avg_volume_20d,
                'current_volume': current_volume,
                'volume_ratio': volume_ratio,
                'volume_signal': 'High' if volume_ratio > 1.5 else 'Normal'
            }
        except:
            return {}
    
    def run_monte_carlo_simulation(self, data, simulations=1000, days=30):
        """Run Monte Carlo price simulation"""
        try:
            returns = data['Close'].pct_change().dropna()
            current_price = data['Close'].iloc[-1]
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Run simulations
            final_prices = []
            for _ in range(simulations):
                price = current_price
                for _ in range(days):
                    daily_return = np.random.normal(mean_return, std_return)
                    price *= (1 + daily_return)
                final_prices.append(price)
            
            final_prices = np.array(final_prices)
            
            return {
                'current_price': current_price,
                'mean_target': np.mean(final_prices),
                'upside_95': np.percentile(final_prices, 95),
                'downside_5': np.percentile(final_prices, 5),
                'probability_up': np.sum(final_prices > current_price) / simulations * 100
            }
        except:
            return {}
    
    def calculate_ml_prediction(self, data):
        """Basic ML trend prediction using linear regression"""
        try:
            prices = data['Close'].values
            
            # Simple linear regression for trend
            x = np.arange(len(prices)).reshape(-1, 1)
            
            # Calculate correlation and trend strength
            correlation = np.corrcoef(x.flatten(), prices)[0, 1]
            
            # Simple moving average crossover prediction
            short_ma = pd.Series(prices).rolling(window=5).mean().iloc[-1]
            long_ma = pd.Series(prices).rolling(window=20).mean().iloc[-1]
            
            trend_strength = abs(correlation)
            direction = 'Bullish' if short_ma > long_ma else 'Bearish'
            confidence = min(trend_strength * 100, 95)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'trend_strength': trend_strength,
                'signal': 'Strong ' + direction if confidence > 70 else 'Weak ' + direction
            }
        except:
            return {}

class EnhancedAnalysisView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setup_ui()
        
        # Setup market data timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_market_data) 
        self.timer.start(5000)
        
        # Force immediate update
        QTimer.singleShot(1000, self.update_market_data)
    
    def setup_ui(self):
        """Setup enhanced UI with tabs for different analysis types"""
        main_layout = QVBoxLayout()
        
        # Market Overview (same as before)
        market_layout = QGridLayout()
        market_layout.addWidget(QLabel('Market Overview'), 0, 0, 1, 6)
        
        market_layout.addWidget(QLabel('VIX:'), 1, 0)
        self.vix_label = QLabel('Loading...')
        market_layout.addWidget(self.vix_label, 1, 1)
        
        market_layout.addWidget(QLabel('SPX:'), 1, 2)
        self.spx_label = QLabel('Loading...')
        market_layout.addWidget(self.spx_label, 1, 3)
        
        market_layout.addWidget(QLabel('NDX:'), 1, 4)
        self.ndx_label = QLabel('Loading...')
        market_layout.addWidget(self.ndx_label, 1, 5)
        
        main_layout.addLayout(market_layout)
        
        # Symbol Input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel('Symbol:'))
        self.symbol_input = QLineEdit()
        self.symbol_input.setText('AAPL')
        input_layout.addWidget(self.symbol_input)
        
        self.analyze_btn = QPushButton('Advanced Analysis')
        self.analyze_btn.clicked.connect(self.run_advanced_analysis)
        input_layout.addWidget(self.analyze_btn)
        
        main_layout.addLayout(input_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Tabbed results
        self.tabs = QTabWidget()
        
        # Basic Analysis Tab
        self.basic_tab = QTextEdit()
        self.tabs.addTab(self.basic_tab, "Basic Analysis")
        
        # Time Series Tab
        self.ts_tab = QTextEdit()
        self.tabs.addTab(self.ts_tab, "Time Series & Slope")
        
        # Volatility Tab
        self.vol_tab = QTextEdit()
        self.tabs.addTab(self.vol_tab, "RV & IV Analysis")
        
        # Volume Tab
        self.volume_tab = QTextEdit()
        self.tabs.addTab(self.volume_tab, "Volume Analysis")
        
        # Monte Carlo Tab
        self.mc_tab = QTextEdit()
        self.tabs.addTab(self.mc_tab, "Monte Carlo")
        
        # ML Tab
        self.ml_tab = QTextEdit()
        self.tabs.addTab(self.ml_tab, "ML Predictions")
        
        main_layout.addWidget(self.tabs)
        
        # Status
        self.status = QLabel('Ready for advanced analysis')
        main_layout.addWidget(self.status)
        
        self.setLayout(main_layout)
    
    def update_market_data(self):
        """Update market data display"""
        try:
            vix_ticker = yf.Ticker('^VIX')
            vix_data = vix_ticker.history(period='1d')
            if not vix_data.empty:
                vix_price = vix_data['Close'].iloc[-1]
                self.vix_label.setText(f'{vix_price:.2f}')
            
            spx_ticker = yf.Ticker('^GSPC') 
            spx_data = spx_ticker.history(period='1d')
            if not spx_data.empty:
                spx_price = spx_data['Close'].iloc[-1]
                self.spx_label.setText(f'{spx_price:,.2f}')
                
            ndx_ticker = yf.Ticker('^NDX')
            ndx_data = ndx_ticker.history(period='1d')
            if not ndx_data.empty:
                ndx_price = ndx_data['Close'].iloc[-1] 
                self.ndx_label.setText(f'{ndx_price:,.2f}')
                
        except Exception as e:
            print(f"Market data update error: {e}")
    
    def run_advanced_analysis(self):
        """Start advanced analysis in background thread"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            self.status.setText('Enter a symbol')
            return
        
        self.status.setText(f'Running advanced analysis for {symbol}...')
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(False)
        
        # Clear previous results
        for i in range(self.tabs.count()):
            if hasattr(self.tabs.widget(i), 'clear'):
                self.tabs.widget(i).clear()
        
        # Start analysis worker
        self.worker = AdvancedAnalysisWorker(symbol)
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.analysis_complete.connect(self.display_results)
        self.worker.start()
    
    def display_results(self, results):
        """Display analysis results in tabs"""
        try:
            symbol = self.symbol_input.text().strip().upper()
            
            if not results:
                self.status.setText('Analysis failed - no data')
                return
            
            # Basic Analysis
            if 'ts_analysis' in results and results['ts_analysis']:
                ts = results['ts_analysis']
                self.basic_tab.append(f"=== {symbol} BASIC ANALYSIS ===\\n")
                self.basic_tab.append(f"Current Trend: {ts.get('trend', 'N/A')}")
                self.basic_tab.append(f"MA Signal: {ts.get('ma_signal', 'N/A')}")
                self.basic_tab.append(f"20-day MA: ${ts.get('ma_20', 0):.2f}")
                self.basic_tab.append(f"50-day MA: ${ts.get('ma_50', 0):.2f}")
            
            # Time Series Analysis
            if 'ts_analysis' in results and results['ts_analysis']:
                ts = results['ts_analysis']
                self.ts_tab.append(f"=== TIME SERIES & SLOPE ANALYSIS ===\\n")
                self.ts_tab.append(f"Price Slope: {ts.get('slope', 0):.4f}")
                self.ts_tab.append(f"Trend Direction: {ts.get('trend', 'N/A')}")
                self.ts_tab.append(f"Momentum (14d): {ts.get('momentum', 0):.2f}%")
                self.ts_tab.append(f"Moving Average Signal: {ts.get('ma_signal', 'N/A')}")
            
            # Volatility Analysis  
            if 'volatility' in results and results['volatility']:
                vol = results['volatility']
                self.vol_tab.append(f"=== REALIZED & IMPLIED VOLATILITY ===\\n")
                self.vol_tab.append(f"30-day RV: {vol.get('rv_30d', 0):.2f}%")
                self.vol_tab.append(f"1-year RV: {vol.get('rv_1y', 0):.2f}%")
                self.vol_tab.append(f"IV Estimate: {vol.get('iv_estimate', 0):.2f}%")
                self.vol_tab.append(f"Vol Regime: {vol.get('vol_regime', 'N/A')}")
            
            # Volume Analysis
            if 'volume' in results and results['volume']:
                vol = results['volume']
                self.volume_tab.append(f"=== VOLUME ANALYSIS ===\\n")
                self.volume_tab.append(f"Current Volume: {vol.get('current_volume', 0):,.0f}")
                self.volume_tab.append(f"20-day Avg Volume: {vol.get('avg_volume_20d', 0):,.0f}")
                self.volume_tab.append(f"Volume Ratio: {vol.get('volume_ratio', 0):.2f}x")
                self.volume_tab.append(f"Volume Signal: {vol.get('volume_signal', 'N/A')}")
            
            # Monte Carlo
            if 'monte_carlo' in results and results['monte_carlo']:
                mc = results['monte_carlo']
                self.mc_tab.append(f"=== MONTE CARLO SIMULATION (30 days) ===\\n")
                self.mc_tab.append(f"Current Price: ${mc.get('current_price', 0):.2f}")
                self.mc_tab.append(f"Mean Target: ${mc.get('mean_target', 0):.2f}")
                self.mc_tab.append(f"95% Upside: ${mc.get('upside_95', 0):.2f}")
                self.mc_tab.append(f"5% Downside: ${mc.get('downside_5', 0):.2f}")
                self.mc_tab.append(f"Prob. of Gain: {mc.get('probability_up', 0):.1f}%")
            
            # ML Predictions
            if 'ml_prediction' in results and results['ml_prediction']:
                ml = results['ml_prediction']
                self.ml_tab.append(f"=== MACHINE LEARNING PREDICTION ===\\n")
                self.ml_tab.append(f"Direction: {ml.get('direction', 'N/A')}")
                self.ml_tab.append(f"Confidence: {ml.get('confidence', 0):.1f}%")
                self.ml_tab.append(f"Signal: {ml.get('signal', 'N/A')}")
                self.ml_tab.append(f"Trend Strength: {ml.get('trend_strength', 0):.3f}")
            
            self.status.setText(f'Advanced analysis complete for {symbol}')
            
        except Exception as e:
            self.status.setText(f'Display error: {e}')
        finally:
            self.progress_bar.setVisible(False)
            self.analyze_btn.setEnabled(True)
