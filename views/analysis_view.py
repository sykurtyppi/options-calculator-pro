"""
Analysis View - Working Version with Real-time Market Data
"""

import json
import sqlite3
from pathlib import Path

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QLineEdit, QTextEdit, QGridLayout,
                               QFileDialog, QApplication)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from datetime import datetime

class AnalysisView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.latest_symbol = ""
        self.project_root = Path(__file__).resolve().parent.parent
        self.reports_dir = self.project_root / "exports" / "reports"
        self.db_path = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
        self.setup_ui()
        self.apply_styles()
        
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
        market_title = QLabel('Market Overview')
        market_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        market_title.setStyleSheet("color: #0ea5e9;")
        market_layout.addWidget(market_title, 0, 0, 1, 4)
        
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

        self.copy_button = QPushButton('Copy Report')
        self.copy_button.clicked.connect(self.copy_results)
        symbol_layout.addWidget(self.copy_button)

        self.export_button = QPushButton('Export Report')
        self.export_button.clicked.connect(self.export_results)
        symbol_layout.addWidget(self.export_button)

        self.load_outputs_button = QPushButton('Load Latest Outputs')
        self.load_outputs_button.clicked.connect(self.load_latest_outputs)
        symbol_layout.addWidget(self.load_outputs_button)

        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.clear_results)
        symbol_layout.addWidget(self.clear_button)
        
        main_layout.addLayout(symbol_layout)
        
        # Results
        self.results = QTextEdit()
        main_layout.addWidget(self.results)
        
        # Status
        self.status = QLabel('Ready')
        main_layout.addWidget(self.status)
        
        self.setLayout(main_layout)

    def apply_styles(self):
        """Apply a clean, consistent dark theme."""
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #1a1a1a;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #ffffff;
                padding: 6px 8px;
            }
            QLineEdit:focus {
                border-color: #0ea5e9;
            }
            QTextEdit {
                background-color: #111111;
                border: 1px solid #2d2d2d;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0ea5e9, stop:1 #0284c7);
                border: none;
                border-radius: 6px;
                color: white;
                font-weight: bold;
                padding: 7px 14px;
            }
            QPushButton:hover { background: #0284c7; }
        """)
        secondary_style = """
            QPushButton {
                background-color: #1f2937;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #cbd5e1;
                font-weight: bold;
                padding: 7px 12px;
            }
            QPushButton:hover { background-color: #334155; }
        """
        self.copy_button.setStyleSheet(secondary_style)
        self.export_button.setStyleSheet(secondary_style)
        self.load_outputs_button.setStyleSheet(secondary_style)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #cbd5e1;
                font-weight: bold;
                padding: 7px 12px;
            }
            QPushButton:hover { background-color: #1f2937; }
        """)
        self.status.setStyleSheet("color: #93c5fd;")
    
    def force_market_update(self):
        """Force market data update using yfinance directly"""
        try:
            import yfinance as yf

            # Initialize default values to prevent variable scope errors
            vix_price, spx_price, ndx_price = 16.0, 4500.0, 15000.0

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
                
            self.latest_symbol = symbol
            self.status.setText(f'Running advanced analysis for {symbol}...')
            self.results.clear()
            
            # Import and initialize your analysis worker
            from core.workers.analysis_worker import AnalysisWorker
            
            self.results.append(f'=== PROFESSIONAL ANALYSIS FOR {symbol} ===')
            self.results.append('Initializing advanced calculation engines...')
            
            # Create analysis worker instance
            worker = AnalysisWorker([symbol])
            
            # Basic stock data first
            self.results.append('')
            self.results.append('--- BASIC STOCK DATA ---')
            try:
                basic_data = worker.get_basic_stock_data(symbol)
                if basic_data:
                    for key, value in basic_data.items():
                        self.results.append(f'{key}: {value}')
            except Exception as e:
                self.results.append(f'Basic data error: {e}')
            
            # Heston Model Analysis
            self.results.append('')
            self.results.append('--- HESTON VOLATILITY MODEL ---')
            try:
                import yfinance as yf
                import numpy as np

                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='60d')

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    returns = hist['Close'].pct_change().dropna()

                    # Basic Heston parameters estimation
                    historical_vol = returns.std() * np.sqrt(252)
                    mean_reversion_speed = 2.0  # kappa
                    long_term_vol = historical_vol  # theta
                    vol_of_vol = 0.3  # sigma
                    correlation = -0.7  # rho (typically negative for equities)

                    # Feller condition check
                    feller_condition = 2 * mean_reversion_speed * long_term_vol**2
                    feller_check = feller_condition > vol_of_vol**2

                    self.results.append(f'Current Price: ${current_price:.2f}')
                    self.results.append(f'Historical Vol (theta): {historical_vol:.1%}')
                    self.results.append(f'Mean Reversion (kappa): {mean_reversion_speed:.2f}')
                    self.results.append(f'Vol of Vol (sigma): {vol_of_vol:.1%}')
                    self.results.append(f'Correlation (rho): {correlation:.2f}')
                    self.results.append(f'Feller Condition: {"SATISFIED" if feller_check else "VIOLATED"}')

                    if not feller_check:
                        self.results.append('⚠️ Warning: Feller condition violated - vol may hit zero')

                    # Simple volatility smile estimation
                    atm_vol = historical_vol
                    skew_factor = 0.02  # Typical equity skew

                    self.results.append(f'ATM Volatility: {atm_vol:.1%}')
                    self.results.append(f'Vol Smile Skew: {skew_factor:.1%} per strike')

                    # Volatility clustering check
                    vol_clustering = returns.rolling(5).std().std()
                    self.results.append(f'Vol Clustering Index: {vol_clustering:.4f}')

                else:
                    self.results.append('Error: No historical data available')

            except Exception as e:
                self.results.append(f'Heston model error: {e}')
            
            # Yang-Zhang Volatility
            self.results.append('')
            self.results.append('--- YANG-ZHANG VOLATILITY ---')
            try:
                # Use your volatility service
                from services.volatility_service import VolatilityService
                from services.market_data import MarketDataService
                from utils.config_manager import ConfigManager
                config = ConfigManager()
                market_data = MarketDataService()
                vol_service = VolatilityService(config, market_data)
                # Get historical data first
                import yfinance as yf
                import numpy as np
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period='60d')

                if not hist_data.empty:
                    # Calculate Yang-Zhang volatility correctly from scratch
                    # Avoid the broken service implementation

                    # Required: Open, High, Low, Close prices
                    O = hist_data['Open']
                    H = hist_data['High']
                    L = hist_data['Low']
                    C = hist_data['Close']
                    C_prev = C.shift(1)

                    # Drop first row (no previous close)
                    valid_data = hist_data.iloc[1:]
                    O = O.iloc[1:]
                    H = H.iloc[1:]
                    L = L.iloc[1:]
                    C = C.iloc[1:]
                    C_prev = C_prev.iloc[1:]

                    if len(valid_data) >= 30:
                        # Yang-Zhang components
                        # Overnight return variance
                        overnight = np.log(O / C_prev)

                        # Open-to-close return variance
                        open_close = np.log(C / O)

                        # Rogers-Satchell variance (intraday)
                        rs = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)

                        # Calculate k (bias correction factor)
                        n = len(overnight)
                        k = 0.34 / (1.34 + (n + 1) / (n - 1))

                        # Yang-Zhang estimator
                        overnight_var = np.var(overnight, ddof=1)
                        open_close_var = np.var(open_close, ddof=1)
                        rs_var = np.mean(rs)

                        yz_variance = overnight_var + k * open_close_var + (1 - k) * rs_var

                        # Convert to annual volatility (DO NOT multiply by 252 again!)
                        yz_vol = np.sqrt(yz_variance * 252)

                        self.results.append(f'Yang-Zhang Volatility: {yz_vol:.4f} ({yz_vol:.1%})')
                        self.results.append(f'Window: {n} days, k-factor: {k:.3f}')
                    else:
                        self.results.append('Insufficient data for Yang-Zhang calculation')
                else:
                    self.results.append('No historical data available')
            except Exception as e:
                self.results.append(f'Yang-Zhang error: {e}')
            
            # Monte Carlo Simulation
            self.results.append('')
            self.results.append('--- MONTE CARLO SIMULATION ---')
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
            self.results.append('')
            self.results.append('--- BLACK-SCHOLES GREEKS ---')
            try:
                from utils.greeks_calculator import GreeksCalculator
                from utils.config_manager import ConfigManager
                config = ConfigManager()
                greeks_calc = GreeksCalculator(config)
                
                # Get current stock price and set up ATM option parameters
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                current_price = hist['Close'].iloc[-1]

                # Get nearest expiration date
                exp_dates = ticker.options
                if exp_dates:
                    nearest_exp = exp_dates[0]
                    # Convert to days
                    from datetime import datetime
                    exp_date = datetime.strptime(nearest_exp, '%Y-%m-%d')
                    days_to_exp = (exp_date - datetime.now()).days
                    time_to_exp = days_to_exp / 365.0

                    # ATM strike (round to nearest $5)
                    atm_strike = round(current_price / 5) * 5

                    # Get implied volatility from options chain
                    try:
                        chain = ticker.option_chain(nearest_exp)
                        calls = chain.calls
                        calls['distance'] = abs(calls['strike'] - current_price)
                        atm_call = calls.loc[calls['distance'].idxmin()]
                        iv = atm_call['impliedVolatility']
                    except:
                        # Fallback to historical volatility
                        hist_60d = ticker.history(period='60d')
                        returns = hist_60d['Close'].pct_change().dropna()
                        iv = returns.std() * (252**0.5)

                    # Risk-free rate (approximate)
                    risk_free_rate = 0.045  # 4.5% current Fed rate

                    # Create MarketParameters object for Greeks calculation
                    from utils.greeks_calculator import MarketParameters, OptionType as GreeksOptionType

                    market_params = MarketParameters(
                        spot_price=current_price,
                        strike_price=atm_strike,
                        time_to_expiry=time_to_exp,
                        risk_free_rate=risk_free_rate,
                        volatility=iv,
                        option_type=GreeksOptionType.CALL
                    )

                    # Calculate Greeks with proper parameters
                    greeks = greeks_calc._calculate_black_scholes_greeks(
                        market_params, include_higher_order=True
                    )
                else:
                    greeks = None
                if greeks:
                    # Handle GreeksResult object
                    if hasattr(greeks, '__dict__'):
                        for key, value in greeks.__dict__.items():
                            if isinstance(value, (int, float)):
                                self.results.append(f'{key}: {value:.4f}')
                            else:
                                self.results.append(f'{key}: {value}')
                    else:
                        self.results.append(f'Greeks: {greeks}')
            except Exception as e:
                self.results.append(f'Greeks calculation error: {e}')
            
            # Kelly Criterion Position Sizing
            self.results.append('')
            self.results.append('--- KELLY CRITERION SIZING ---')
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
            
            # Options Contract Analysis
            self.results.append('')
            self.results.append('--- OPTIONS CHAIN ANALYSIS ---')
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)

                # Get current stock price
                current_price = ticker.history(period='1d')['Close'].iloc[-1]

                # Get options chain
                exp_dates = ticker.options
                if exp_dates:
                    # Get nearest expiration
                    nearest_exp = exp_dates[0]
                    chain = ticker.option_chain(nearest_exp)

                    # Find ATM options
                    calls = chain.calls
                    puts = chain.puts

                    # Find closest to ATM
                    calls['distance'] = abs(calls['strike'] - current_price)
                    puts['distance'] = abs(puts['strike'] - current_price)

                    atm_call = calls.loc[calls['distance'].idxmin()]
                    atm_put = puts.loc[puts['distance'].idxmin()]

                    self.results.append(f'Nearest Expiration: {nearest_exp}')
                    self.results.append(f'ATM Call Strike: ${atm_call["strike"]:.0f}')
                    self.results.append(f'Call Volume: {atm_call["volume"]:,}')
                    self.results.append(f'Call Open Interest: {atm_call["openInterest"]:,}')
                    self.results.append(f'Call IV: {atm_call["impliedVolatility"]:.1%}')
                    self.results.append(f'Call Bid-Ask Spread: ${atm_call["ask"] - atm_call["bid"]:.2f}')

                    self.results.append(f'ATM Put Strike: ${atm_put["strike"]:.0f}')
                    self.results.append(f'Put Volume: {atm_put["volume"]:,}')
                    self.results.append(f'Put Open Interest: {atm_put["openInterest"]:,}')
                    self.results.append(f'Put IV: {atm_put["impliedVolatility"]:.1%}')
                    self.results.append(f'Put Bid-Ask Spread: ${atm_put["ask"] - atm_put["bid"]:.2f}')

                    # Liquidity Analysis
                    total_call_volume = calls['volume'].sum()
                    total_put_volume = puts['volume'].sum()
                    avg_spread = (atm_call["ask"] - atm_call["bid"] + atm_put["ask"] - atm_put["bid"]) / 2

                    self.results.append('')
                    self.results.append('--- LIQUIDITY ANALYSIS ---')
                    self.results.append(f'Total Call Volume: {total_call_volume:,}')
                    self.results.append(f'Total Put Volume: {total_put_volume:,}')
                    self.results.append(f'Put/Call Ratio: {total_put_volume/max(total_call_volume,1):.2f}')
                    self.results.append(f'Average Bid-Ask Spread: ${avg_spread:.2f}')

                    if avg_spread < 0.10:
                        liquidity = "Excellent - Easy to exit"
                    elif avg_spread < 0.25:
                        liquidity = "Good - Manageable exit"
                    elif avg_spread < 0.50:
                        liquidity = "Moderate - Watch spreads"
                    else:
                        liquidity = "Poor - Difficult to exit"

                    self.results.append(f'Liquidity Assessment: {liquidity}')

            except Exception as e:
                self.results.append(f'Options chain error: {e}')

            # IV/RV Analysis
            self.results.append('')
            self.results.append('--- IV vs RV ANALYSIS ---')
            try:
                import yfinance as yf
                import numpy as np
                import pandas as pd
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='30d')

                if not hist.empty:
                    # Calculate Realized Volatility (30-day)
                    returns = hist['Close'].pct_change().dropna()
                    realized_vol = returns.std() * np.sqrt(252)

                    # Get Implied Volatility from ATM options
                    exp_dates = ticker.options
                    if exp_dates:
                        chain = ticker.option_chain(exp_dates[0])
                        current_price = hist['Close'].iloc[-1]
                        calls = chain.calls
                        calls['distance'] = abs(calls['strike'] - current_price)
                        atm_call = calls.loc[calls['distance'].idxmin()]
                        implied_vol = atm_call['impliedVolatility']

                        iv_rv_ratio = implied_vol / realized_vol

                        self.results.append(f'30-Day Realized Volatility: {realized_vol:.1%}')
                        self.results.append(f'ATM Implied Volatility: {implied_vol:.1%}')
                        self.results.append(f'IV/RV Ratio: {iv_rv_ratio:.2f}')

                        if iv_rv_ratio > 1.2:
                            iv_assessment = "HIGH - Good for selling premium"
                        elif iv_rv_ratio > 0.8:
                            iv_assessment = "FAIR - Neutral"
                        else:
                            iv_assessment = "LOW - Consider buying premium"

                        self.results.append(f'IV Assessment: {iv_assessment}')

                        # Time decay slope analysis
                        days_to_exp = (pd.to_datetime(exp_dates[0]) - pd.Timestamp.now()).days
                        theta_per_day = abs(atm_call.get('theta', 0))

                        self.results.append('')
                        self.results.append('--- TIME DECAY ANALYSIS ---')
                        self.results.append(f'Days to Expiration: {days_to_exp}')
                        self.results.append(f'Theta (per day): -${theta_per_day:.2f}')

                        if days_to_exp < 7:
                            theta_risk = "EXTREME - Rapid decay"
                        elif days_to_exp < 21:
                            theta_risk = "HIGH - Accelerating decay"
                        elif days_to_exp < 45:
                            theta_risk = "MODERATE - Steady decay"
                        else:
                            theta_risk = "LOW - Slow decay"

                        self.results.append(f'Time Decay Risk: {theta_risk}')

            except Exception as e:
                self.results.append(f'IV/RV analysis error: {e}')

            # Machine Learning Prediction
            self.results.append('')
            self.results.append('--- ML PREDICTION ---')
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

    def copy_results(self):
        """Copy analysis report to clipboard."""
        text = self.results.toPlainText().strip()
        if not text:
            self.status.setText('Copy skipped: no report content')
            return
        QApplication.clipboard().setText(text)
        self.status.setText('Report copied to clipboard')

    def export_results(self):
        """Export analysis report to text file."""
        text = self.results.toPlainText().strip()
        if not text:
            self.status.setText('Export skipped: no report content')
            return

        symbol = self.latest_symbol or self.symbol_input.text().strip().upper() or "report"
        default_name = f"analysis_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Report",
            default_name,
            "Text Files (*.txt)",
        )
        if not file_path:
            self.status.setText('Export cancelled')
            return
        try:
            with open(file_path, "w", encoding="utf-8") as output_file:
                output_file.write(text)
            self.status.setText(f'Exported report to {file_path}')
        except Exception as e:
            self.status.setText(f'Export failed: {e}')

    def clear_results(self):
        """Clear the report output panel."""
        self.results.clear()
        self.status.setText('Report cleared')

    def load_latest_outputs(self):
        """Load latest threshold/regime report artifacts and session summary."""
        threshold = self._read_latest_json("earnings_threshold_tuning_*.json")
        regime = self._read_latest_json("earnings_regime_diagnostics_*.json")
        session = self._read_latest_session()

        if not threshold and not regime and not session:
            self.status.setText("No persisted outputs found")
            return

        lines = [
            "PROJECT OUTPUT DIGEST",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        if session:
            lines.extend([
                "Latest Backtest Session",
                f"- Session ID: {session['session_id']}",
                f"- Trades: {session['total_trades']}",
                f"- Win Rate: {session['win_rate']:.2%}",
                f"- Total P&L: ${session['total_pnl']:.2f}",
                "",
            ])
        if threshold:
            rec = threshold.get("recommendation") or {}
            lines.extend([
                "Threshold Tuning",
                f"- Session: {threshold.get('session_id', 'n/a')}",
                f"- Threshold: {rec.get('threshold', 'n/a')}",
                f"- Alpha Score: {rec.get('alpha_score', 'n/a')}",
                "",
            ])
        if regime:
            summary = regime.get("summary") or {}
            lines.extend([
                "Regime Diagnostics",
                f"- Session: {regime.get('session_id', 'n/a')}",
                f"- Overall Win Rate: {summary.get('overall_win_rate', 'n/a')}",
                f"- Regime Rows: {summary.get('regime_rows', 'n/a')}",
            ])

        self.results.setPlainText("\n".join(lines))
        self.status.setText("Loaded latest project outputs")

    def _read_latest_json(self, pattern: str):
        if not self.reports_dir.exists():
            return None
        matches = sorted(self.reports_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return None
        try:
            return json.loads(matches[0].read_text(encoding="utf-8"))
        except Exception:
            return None

    def _read_latest_session(self):
        if not self.db_path.exists():
            return None
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, total_trades, win_rate, total_pnl
                    FROM backtest_sessions
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "session_id": row[0],
                    "total_trades": int(row[1] or 0),
                    "win_rate": float(row[2] or 0.0),
                    "total_pnl": float(row[3] or 0.0),
                }
        except Exception:
            return None

    
