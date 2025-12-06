"""
RSI Analysis Plugin - Example Analysis Plugin
Demonstrates how to create custom analysis plugins
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from ..base_plugin import AnalysisPlugin, PluginMetadata

def get_plugin_metadata() -> PluginMetadata:
    """Return plugin metadata"""
    return PluginMetadata(
        name="RSI Analysis",
        version="1.0.0",
        description="Relative Strength Index analysis for options trading signals",
        author="Options Calculator Pro",
        category="analysis"
    )

class RSIAnalysisPlugin(AnalysisPlugin):
    """RSI Analysis Plugin for additional technical analysis"""
    
    def initialize(self, app_context: Dict[str, Any]) -> bool:
        """Initialize the RSI analysis plugin"""
        try:
            # Store references to app services
            self.market_data_service = app_context.get('market_data_service')
            self.config = {
                'rsi_period': 14,
                'overbought_threshold': 70,
                'oversold_threshold': 30,
                'signal_strength_threshold': 0.7
            }
            
            logger.info("RSI Analysis Plugin initialized")
            return True
            
        except Exception as e:
            logger.error(f"RSI Plugin initialization failed: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup RSI plugin resources"""
        return True
    
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform RSI analysis on symbol data"""
        try:
            # Get historical price data
            if 'historical_data' not in data:
                # Fetch historical data if not provided
                if self.market_data_service:
                    hist_data = self.market_data_service.get_historical_data(symbol, period='3mo')
                else:
                    return {'error': 'No historical data available'}
            else:
                hist_data = data['historical_data']
            
            if hist_data.empty:
                return {'error': 'No historical data for RSI calculation'}
            
            # Calculate RSI
            rsi_values = self.calculate_rsi(hist_data['Close'], self.config['rsi_period'])
            current_rsi = rsi_values.iloc[-1] if not rsi_values.empty else 50
            
            # Generate signals
            signals = self.generate_rsi_signals(rsi_values)
            
            # Calculate signal strength
            signal_strength = self.calculate_signal_strength(rsi_values, current_rsi)
            
            # Determine market bias
            market_bias = self.get_market_bias(current_rsi)
            
            results = {
                'plugin_name': 'RSI Analysis',
                'symbol': symbol,
                'current_rsi': float(current_rsi),
                'rsi_signal': signals[-1] if signals else 'NEUTRAL',
                'signal_strength': signal_strength,
                'market_bias': market_bias,
                'overbought': current_rsi > self.config['overbought_threshold'],
                'oversold': current_rsi < self.config['oversold_threshold'],
                'rsi_history': rsi_values.tail(20).tolist(),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'recommendations': self.generate_recommendations(current_rsi, signals)
            }
            
            # Emit analysis complete signal
            self.analysis_complete.emit(symbol, results)
            
            return results
            
        except Exception as e:
            error_msg = f"RSI analysis failed for {symbol}: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI values"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral RSI
            
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return pd.Series([50] * len(prices))
    
    def generate_rsi_signals(self, rsi_values: pd.Series) -> list:
        """Generate trading signals based on RSI"""
        signals = []
        
        for rsi in rsi_values:
            if rsi > self.config['overbought_threshold']:
                signals.append('SELL')
            elif rsi < self.config['oversold_threshold']:
                signals.append('BUY')
            else:
                signals.append('NEUTRAL')
        
        return signals
    
    def calculate_signal_strength(self, rsi_values: pd.Series, current_rsi: float) -> float:
        """Calculate signal strength based on RSI patterns"""
        try:
            # Check for RSI divergence and momentum
            recent_rsi = rsi_values.tail(5)
            
            # Calculate momentum
            rsi_momentum = (current_rsi - recent_rsi.iloc[0]) / 5
            
            # Calculate distance from extreme levels
            if current_rsi > 70:
                distance_factor = (current_rsi - 70) / 30  # 0 to 1 scale
            elif current_rsi < 30:
                distance_factor = (30 - current_rsi) / 30  # 0 to 1 scale
            else:
                distance_factor = 0
            
            # Combine factors
            signal_strength = min(1.0, abs(rsi_momentum * 0.3) + distance_factor * 0.7)
            
            return round(signal_strength, 3)
            
        except Exception as e:
            logger.error(f"Signal strength calculation error: {e}")
            return 0.5
    
    def get_market_bias(self, current_rsi: float) -> str:
        """Determine market bias based on RSI"""
        if current_rsi > 60:
            return 'BULLISH'
        elif current_rsi < 40:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def generate_recommendations(self, current_rsi: float, signals: list) -> Dict[str, str]:
        """Generate trading recommendations based on RSI analysis"""
        recommendations = {}
        
        current_signal = signals[-1] if signals else 'NEUTRAL'
        
        if current_signal == 'SELL' and current_rsi > 75:
            recommendations['options_strategy'] = 'Consider PUT calendar spreads or credit spreads'
            recommendations['risk_level'] = 'MODERATE'
            recommendations['time_horizon'] = 'SHORT_TERM'
            
        elif current_signal == 'BUY' and current_rsi < 25:
            recommendations['options_strategy'] = 'Consider CALL calendar spreads or debit spreads'
            recommendations['risk_level'] = 'MODERATE'
            recommendations['time_horizon'] = 'SHORT_TERM'
            
        else:
            recommendations['options_strategy'] = 'Neutral - consider calendar spreads around earnings'
            recommendations['risk_level'] = 'LOW'
            recommendations['time_horizon'] = 'MEDIUM_TERM'
        
        # Add RSI-specific insights
        if 45 <= current_rsi <= 55:
            recommendations['iv_strategy'] = 'Prime conditions for calendar spreads'
        elif current_rsi > 70:
            recommendations['iv_strategy'] = 'High probability of mean reversion'
        elif current_rsi < 30:
            recommendations['iv_strategy'] = 'Potential oversold bounce opportunity'
        
        return recommendations
    
    def get_analysis_priority(self) -> int:
        """RSI analysis has medium priority"""
        return 60
    
    def get_config_widget(self):
        """Return configuration widget for RSI settings"""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QSlider
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # RSI Period
        layout.addWidget(QLabel("RSI Period:"))
        period_spin = QSpinBox()
        period_spin.setRange(5, 50)
        period_spin.setValue(self.config['rsi_period'])
        period_spin.valueChanged.connect(lambda v: self.config.update({'rsi_period': v}))
        layout.addWidget(period_spin)
        
        # Overbought threshold
        layout.addWidget(QLabel("Overbought Threshold:"))
        ob_slider = QSlider(1)  # Horizontal
        ob_slider.setRange(60, 90)
        ob_slider.setValue(self.config['overbought_threshold'])
        ob_slider.valueChanged.connect(lambda v: self.config.update({'overbought_threshold': v}))
        layout.addWidget(ob_slider)
        
        # Oversold threshold
        layout.addWidget(QLabel("Oversold Threshold:"))
        os_slider = QSlider(1)  # Horizontal
        os_slider.setRange(10, 40)
        os_slider.setValue(self.config['oversold_threshold'])
        os_slider.valueChanged.connect(lambda v: self.config.update({'oversold_threshold': v}))
        layout.addWidget(os_slider)
        
        return widget
    
    def get_menu_actions(self) -> list:
        """Return menu actions for RSI plugin"""
        return [
            {
                'text': 'Run RSI Analysis',
                'callback': self.run_standalone_analysis,
                'icon': None,
                'shortcut': 'Ctrl+R'
            }
        ]
    
    def run_standalone_analysis(self):
        """Run RSI analysis as standalone operation"""
        # This would trigger a dialog to select symbol and run analysis
        self.plugin_status_changed.emit("RSI standalone analysis started")