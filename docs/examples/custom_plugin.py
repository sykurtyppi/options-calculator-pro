"""
Custom Plugin Example
Demonstrates how to create and integrate custom plugins
"""

from plugins.base_plugin import AnalysisPlugin, PluginMetadata, UIPlugin
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PySide6.QtCore import Signal
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_plugin_metadata() -> PluginMetadata:
    """Plugin metadata - required function"""
    return PluginMetadata(
        name="Bollinger Bands Analysis",
        version="1.0.0",
        description="Bollinger Bands analysis for options trading signals",
        author="Options Calculator Pro Team",
        category="analysis"
    )

class BollingerBandsPlugin(AnalysisPlugin):
    """
    Custom analysis plugin that adds Bollinger Bands analysis
    to complement the main calendar spread analysis
    """
    
    def initialize(self, app_context: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        try:
            self.market_data_service = app_context.get('market_data_service')
            self.config = {
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'signal_threshold': 0.8
            }
            
            logger.info("Bollinger Bands Plugin initialized")
            return True
            
        except Exception as e:
            logger.error(f"Bollinger Bands Plugin initialization failed: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup plugin resources"""
        return True
    
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Bollinger Bands analysis"""
        try:
            # Get historical data
            historical_data = data.get('historical_data')
            if historical_data is None or historical_data.empty:
                return {'error': 'No historical data available'}
            
            current_price = data.get('current_price', 0)
            
            # Calculate Bollinger Bands
            bb_data = self.calculate_bollinger_bands(
                historical_data['Close'], 
                self.config['bb_period'],
                self.config['bb_std_dev']
            )
            
            # Generate trading signals
            signals = self.generate_bb_signals(bb_data, current_price)
            
            # Calculate signal strength
            signal_strength = self.calculate_signal_strength(bb_data, current_price)
            
            results = {
                'plugin_name': 'Bollinger Bands Analysis',
                'symbol': symbol,
                'current_price': current_price,
                'bb_upper': bb_data['upper'][-1],
                'bb_middle': bb_data['middle'][-1],
                'bb_lower': bb_data['lower'][-1],
                'bb_position': self.calculate_bb_position(current_price, bb_data),
                'signal': signals['current_signal'],
                'signal_strength': signal_strength,
                'volatility_squeeze': signals['volatility_squeeze'],
                'bb_width': bb_data['width'][-1],
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'recommendations': self.generate_bb_recommendations(signals, bb_data, current_price)
            }
            
            # Emit analysis complete signal
            self.analysis_complete.emit(symbol, results)
            
            return results
            
        except Exception as e:
            error_msg = f"Bollinger Bands analysis failed for {symbol}: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        # Simple Moving Average (Middle Band)
        middle = prices.rolling(window=period).mean()
        
        # Standard Deviation
        std = prices.rolling(window=period).std()
        
        # Upper and Lower Bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Band Width (for volatility analysis)
        width = (upper - lower) / middle
        
        return {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'width': width,
            'std': std
        }
    
    def generate_bb_signals(self, bb_data: Dict[str, pd.Series], current_price: float) -> Dict[str, Any]:
        """Generate trading signals based on Bollinger Bands"""
        latest_upper = bb_data['upper'].iloc[-1]
        latest_middle = bb_data['middle'].iloc[-1]
        latest_lower = bb_data['lower'].iloc[-1]
        latest_width = bb_data['width'].iloc[-1]
        
        # Position relative to bands
        if current_price > latest_upper:
            current_signal = 'SELL'  # Price above upper band - overbought
            signal_type = 'OVERBOUGHT'
        elif current_price < latest_lower:
            current_signal = 'BUY'   # Price below lower band - oversold
            signal_type = 'OVERSOLD'
        else:
            current_signal = 'NEUTRAL'
            signal_type = 'NORMAL'
        
        # Volatility squeeze detection (narrow bands)
        recent_widths = bb_data['width'].tail(5)
        avg_width = bb_data['width'].tail(50).mean()
        volatility_squeeze = latest_width < avg_width * 0.7
        
        return {
            'current_signal': current_signal,
            'signal_type': signal_type,
            'volatility_squeeze': volatility_squeeze,
            'band_width_percentile': self.calculate_width_percentile(bb_data['width'])
        }
    
    def calculate_bb_position(self, current_price: float, bb_data: Dict[str, pd.Series]) -> float:
        """Calculate position within Bollinger Bands (0 = lower band, 1 = upper band)"""
        latest_upper = bb_data['upper'].iloc[-1]
        latest_lower = bb_data['lower'].iloc[-1]
        
        if latest_upper == latest_lower:
            return 0.5
        
        position = (current_price - latest_lower) / (latest_upper - latest_lower)
        return max(0, min(1, position))  # Clamp between 0 and 1
    
    def calculate_signal_strength(self, bb_data: Dict[str, pd.Series], current_price: float) -> float:
        """Calculate signal strength based on band position and volatility"""
        bb_position = self.calculate_bb_position(current_price, bb_data)
        
        # Stronger signals near the bands
        if bb_position > 0.8 or bb_position < 0.2:
            strength = min(1.0, abs(bb_position - 0.5) * 2)
        else:
            strength = 0.3  # Weak signal in middle range
        
        # Adjust for volatility squeeze
        latest_width = bb_data['width'].iloc[-1]
        avg_width = bb_data['width'].tail(50).mean()
        
        if latest_width < avg_width * 0.7:  # Volatility squeeze
            strength *= 1.3  # Increase signal strength
        
        return min(1.0, strength)
    
    def calculate_width_percentile(self, width_series: pd.Series) -> float:
        """Calculate current band width percentile over lookback period"""
        current_width = width_series.iloc[-1]
        historical_widths = width_series.tail(252)  # 1 year lookback
        
        percentile = (historical_widths < current_width).sum() / len(historical_widths)
        return percentile
    
    def generate_bb_recommendations(self, signals: Dict[str, Any], bb_data: Dict[str, pd.Series], 
                                  current_price: float) -> Dict[str, str]:
        """Generate trading recommendations based on Bollinger Bands analysis"""
        recommendations = {}
        
        signal = signals['current_signal']
        volatility_squeeze = signals['volatility_squeeze']
        
        if signal == 'BUY' and volatility_squeeze:
            recommendations['strategy'] = 'Consider CALL options - oversold with low volatility'
            recommendations['risk_level'] = 'MODERATE'
            recommendations['time_horizon'] = 'SHORT_TERM'
            
        elif signal == 'SELL' and volatility_squeeze:
            recommendations['strategy'] = 'Consider PUT options - overbought with low volatility'
            recommendations['risk_level'] = 'MODERATE'
            recommendations['time_horizon'] = 'SHORT_TERM'
            
        elif volatility_squeeze and signal == 'NEUTRAL':
            recommendations['strategy'] = 'Volatility squeeze - consider straddle/strangle strategies'
            recommendations['risk_level'] = 'HIGH'
            recommendations['time_horizon'] = 'SHORT_TERM'
            
        else:
            recommendations['strategy'] = 'No clear Bollinger Bands signal'
            recommendations['risk_level'] = 'LOW'
            recommendations['time_horizon'] = 'MEDIUM_TERM'
        
        # Add BB-specific insights
        bb_position = self.calculate_bb_position(current_price, bb_data)
        if bb_position > 0.9:
            recommendations['bb_insight'] = 'Price at extreme upper band - mean reversion likely'
        elif bb_position < 0.1:
            recommendations['bb_insight'] = 'Price at extreme lower band - bounce potential'
        elif 0.4 <= bb_position <= 0.6:
            recommendations['bb_insight'] = 'Price near middle band - trend continuation possible'
        
        return recommendations
    
    def get_config_widget(self):
        """Return configuration widget for Bollinger Bands settings"""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QDoubleSpinBox
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # BB Period
        layout.addWidget(QLabel("Bollinger Bands Period:"))
        period_spin = QSpinBox()
        period_spin.setRange(5, 100)
        period_spin.setValue(self.config['bb_period'])
        period_spin.valueChanged.connect(lambda v: self.config.update({'bb_period': v}))
        layout.addWidget(period_spin)
        
        # Standard Deviation
        layout.addWidget(QLabel("Standard Deviation:"))
        std_spin = QDoubleSpinBox()
        std_spin.setRange(1.0, 4.0)
        std_spin.setDecimals(1)
        std_spin.setValue(self.config['bb_std_dev'])
        std_spin.valueChanged.connect(lambda v: self.config.update({'bb_std_dev': v}))
        layout.addWidget(std_spin)
        
        # Signal Threshold
        layout.addWidget(QLabel("Signal Threshold:"))
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.1, 1.0)
        threshold_spin.setDecimals(2)
        threshold_spin.setValue(self.config['signal_threshold'])
        threshold_spin.valueChanged.connect(lambda v: self.config.update({'signal_threshold': v}))
        layout.addWidget(threshold_spin)
        
        return widget
    
    def get_menu_actions(self):
        """Return menu actions for Bollinger Bands plugin"""
        return [
            {
                'text': 'Run Bollinger Bands Analysis',
                'callback': self.run_standalone_analysis,
                'icon': None,
                'shortcut': 'Ctrl+B'
            }
        ]
    
    def run_standalone_analysis(self):
        """Run Bollinger Bands analysis as standalone operation"""
        self.plugin_status_changed.emit("Bollinger Bands standalone analysis started")
        # This would trigger a dialog to select symbol and run analysis


class BollingerBandsWidget(QWidget):
    """Custom UI widget for displaying Bollinger Bands analysis"""
    
    analysis_requested = Signal(str)  # symbol
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_data = None
    
    def setup_ui(self):
        """Setup the widget UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Bollinger Bands Analysis")
        title.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #60a3d9;
                padding: 8px;
            }
        """)
        layout.addWidget(title)
        
        # Current position indicator
        self.position_label = QLabel("Position: --")
        layout.addWidget(self.position_label)
        
        # Position bar
        self.position_bar = QProgressBar()
        self.position_bar.setRange(0, 100)
        self.position_bar.setValue(50)
        self.position_bar.setFormat("BB Position: %p%")
        layout.addWidget(self.position_bar)
        
        # Signal display
        self.signal_label = QLabel("Signal: NEUTRAL")
        self.signal_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.signal_label)
        
        # Band values
        self.upper_label = QLabel("Upper Band: --")
        self.middle_label = QLabel("Middle Band: --")
        self.lower_label = QLabel("Lower Band: --")
        
        for label in [self.upper_label, self.middle_label, self.lower_label]:
            label.setStyleSheet("font-family: monospace; padding: 2px;")
            layout.addWidget(label)
        
        # Volatility squeeze indicator
        self.squeeze_label = QLabel("Volatility Squeeze: --")
        layout.addWidget(self.squeeze_label)
        
        # Analysis button
        self.analyze_button = QPushButton("Analyze Current Symbol")
        self.analyze_button.clicked.connect(lambda: self.analysis_requested.emit("CURRENT"))
        layout.addWidget(self.analyze_button)
    
    def update_display(self, analysis_data: Dict[str, Any]):
        """Update widget with new analysis data"""
        self.current_data = analysis_data
        
        try:
            # Update position
            bb_position = analysis_data.get('bb_position', 0.5)
            self.position_bar.setValue(int(bb_position * 100))
            self.position_label.setText(f"Position: {bb_position:.1%}")
            
            # Update signal
            signal = analysis_data.get('signal', 'NEUTRAL')
            self.signal_label.setText(f"Signal: {signal}")
            
            # Color code signal
            if signal == 'BUY':
                color = "#28a745"
            elif signal == 'SELL':
                color = "#dc3545"
            else:
                color = "#ffc107"
            
            self.signal_label.setStyleSheet(f"font-weight: bold; padding: 4px; color: {color};")
            
            # Update band values
            current_price = analysis_data.get('current_price', 0)
            upper = analysis_data.get('bb_upper', 0)
            middle = analysis_data.get('bb_middle', 0)
            lower = analysis_data.get('bb_lower', 0)
            
            self.upper_label.setText(f"Upper Band: ${upper:.2f}")
            self.middle_label.setText(f"Middle Band: ${middle:.2f}")
            self.lower_label.setText(f"Lower Band: ${lower:.2f}")
            
            # Update volatility squeeze
            squeeze = analysis_data.get('volatility_squeeze', False)
            squeeze_text = "YES" if squeeze else "NO"
            squeeze_color = "#ffc107" if squeeze else "#28a745"
            
            self.squeeze_label.setText(f"Volatility Squeeze: {squeeze_text}")
            self.squeeze_label.setStyleSheet(f"font-weight: bold; color: {squeeze_color};")
            
        except Exception as e:
            logger.error(f"Error updating Bollinger Bands display: {e}")


class BollingerBandsUIPlugin(UIPlugin):
    """UI plugin that integrates Bollinger Bands widget"""
    
    def initialize(self, app_context: Dict[str, Any]) -> bool:
        """Initialize UI plugin"""
        self.app_context = app_context
        return True
    
    def cleanup(self) -> bool:
        """Cleanup UI plugin"""
        return True
    
    def get_widget(self) -> QWidget:
        """Return the Bollinger Bands widget"""
        return BollingerBandsWidget()
    
    def get_dock_widget_info(self) -> Dict[str, Any]:
        """Return dock widget configuration"""
        return {
            'title': 'Bollinger Bands',
            'area': 'right',
            'allowed_areas': ['left', 'right', 'bottom'],
            'features': ['closable', 'movable', 'floatable']
        }

def demo_plugin_usage():
    """Demonstrate plugin usage"""
    
    print("=== Bollinger Bands Plugin Demo ===")
    
    # Create plugin instance
    metadata = get_plugin_metadata()
    plugin = BollingerBandsPlugin(metadata)
    
    # Mock app context
    app_context = {
        'market_data_service': None,  # Would be real service
        'config_manager': None
    }
    
    # Initialize plugin
    if plugin.initialize(app_context):
        print(f"✅ Plugin '{metadata.name}' initialized successfully")
        
        # Create sample data for analysis
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        
        # Generate sample price data with trend
        np.random.seed(42)
        prices = []
        current = 150.0
        
        for i in range(len(dates)):
            # Add some trend and volatility
            change = np.random.normal(0.001, 0.02)
            current *= (1 + change)
            prices.append(current)
        
        sample_data = {
            'historical_data': pd.DataFrame({
                'Date': dates,
                'Close': prices
            }).set_index('Date'),
            'current_price': prices[-1]
        }
        
        # Run analysis
        print(f"\nRunning Bollinger Bands analysis...")
        result = plugin.analyze('DEMO', sample_data)
        
        if 'error' not in result:
            print(f"✅ Analysis completed successfully")
            print(f"Symbol: {result['symbol']}")
            print(f"Current Price: ${result['current_price']:.2f}")
            print(f"Upper Band: ${result['bb_upper']:.2f}")
            print(f"Middle Band: ${result['bb_middle']:.2f}")
            print(f"Lower Band: ${result['bb_lower']:.2f}")
            print(f"BB Position: {result['bb_position']:.1%}")
            print(f"Signal: {result['signal']}")
            print(f"Signal Strength: {result['signal_strength']:.1%}")
            print(f"Volatility Squeeze: {result['volatility_squeeze']}")
            
            # Show recommendations
            recommendations = result.get('recommendations', {})
            if recommendations:
                print(f"\nRecommendations:")
                for key, value in recommendations.items():
                    print(f"  {key}: {value}")
        else:
            print(f"❌ Analysis failed: {result['error']}")
        
        # Cleanup
        plugin.cleanup()
        print(f"✅ Plugin cleaned up")
        
    else:
        print(f"❌ Plugin initialization failed")

if __name__ == "__main__":
    print("Options Calculator Pro - Custom Plugin Example")
    print("=" * 50)
    
    # Run plugin demo
    demo_plugin_usage()
    
    print("\n" + "=" * 50)
    print("Plugin example completed!")
    print("\nTo use this plugin:")
    print("1. Save this file in the plugins/ directory")
    print("2. Restart the application")
    print("3. The plugin will be automatically discovered")
    print("4. Enable it in Settings → Plugins")