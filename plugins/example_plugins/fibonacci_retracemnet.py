"""
Fibonacci Retracement Plugin - Example Strategy Plugin
Calculates Fibonacci retracement levels for options trading
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from ..base_plugin import StrategyPlugin, PluginMetadata

def get_plugin_metadata() -> PluginMetadata:
    """Return plugin metadata"""
    return PluginMetadata(
        name="Fibonacci Retracement",
        version="1.0.0", 
        description="Fibonacci retracement levels for support/resistance analysis",
        author="Options Calculator Pro",
        category="strategy"
    )

class FibonacciRetracementPlugin(StrategyPlugin):
    """Fibonacci retracement analysis for options trading"""
    
    def initialize(self, app_context: Dict[str, Any]) -> bool:
        """Initialize Fibonacci plugin"""
        try:
            self.market_data_service = app_context.get('market_data_service')
            self.config = {
                'lookback_period': 50,  # Days to look back for high/low
                'fib_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
                'support_resistance_threshold': 0.02  # 2% threshold for S/R
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Fibonacci plugin initialization failed: {e}")
            return False
    def cleanup(self) -> bool:
       """Cleanup Fibonacci plugin resources"""
       return True
   
    def get_strategy_name(self) -> str:
        """Get strategy display name"""
        return "Fibonacci Retracement Strategy"
   
    def calculate_position(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position recommendation based on Fibonacci levels"""
        try:
            symbol = market_data.get('symbol', '')
            current_price = market_data.get('current_price', 0)
            historical_data = market_data.get('historical_data')
           
            if historical_data is None or historical_data.empty:
                return {'error': 'No historical data available'}
           
            # Calculate Fibonacci levels
            fib_analysis = self.calculate_fibonacci_levels(historical_data, current_price)
           
            # Determine position based on current price relative to Fib levels
            position_rec = self.analyze_position_opportunity(
                current_price, fib_analysis, market_data
            )
           
            return {
                'strategy': 'fibonacci_retracement',
                'symbol': symbol,
                'current_price': current_price,
                'fibonacci_levels': fib_analysis,
                'position_recommendation': position_rec,
                'confidence': self.calculate_confidence(fib_analysis, current_price),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
           
        except Exception as e:
            logger.error(f"Fibonacci position calculation failed: {e}")
            return {'error': str(e)}
   
    def calculate_fibonacci_levels(self, historical_data: pd.DataFrame, 
                                 current_price: float) -> Dict[str, Any]:
        """Calculate Fibonacci retracement levels"""
        try:
            # Get recent high and low
            lookback_data = historical_data.tail(self.config['lookback_period'])
           
            swing_high = lookback_data['High'].max()
            swing_low = lookback_data['Low'].min()
            swing_range = swing_high - swing_low
           
            # Calculate Fibonacci levels
            fib_levels = {}
           
            # For uptrend (retracement from high)
            uptrend_levels = {}
            for level in self.config['fib_levels']:
                price_level = swing_high - (swing_range * level)
                uptrend_levels[f'fib_{level:.3f}'] = {
                    'price': round(price_level, 2),
                    'level': level,
                    'type': 'retracement_from_high'
                }
           
            # For downtrend (extension from low)
            downtrend_levels = {}
            for level in self.config['fib_levels']:
                price_level = swing_low + (swing_range * level)
                downtrend_levels[f'fib_{level:.3f}'] = {
                    'price': round(price_level, 2),
                    'level': level,
                    'type': 'extension_from_low'
                }
           
            # Determine current trend
            recent_prices = lookback_data['Close'].tail(10)
            trend = 'uptrend' if recent_prices.iloc[-1] > recent_prices.iloc[0] else 'downtrend'
           
            # Select appropriate levels based on trend
            active_levels = uptrend_levels if trend == 'uptrend' else downtrend_levels
           
            return {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'swing_range': swing_range,
                'current_trend': trend,
                'uptrend_levels': uptrend_levels,
                'downtrend_levels': downtrend_levels,
                'active_levels': active_levels,
                'nearest_support': self.find_nearest_level(current_price, active_levels, 'support'),
                'nearest_resistance': self.find_nearest_level(current_price, active_levels, 'resistance')
            }
           
        except Exception as e:
            logger.error(f"Fibonacci calculation error: {e}")
            return {}
   
    def find_nearest_level(self, current_price: float, levels: Dict[str, Any], 
                          level_type: str) -> Dict[str, Any]:
        """Find nearest support or resistance level"""
        try:
            if level_type == 'support':
                # Find highest level below current price
                support_levels = [
                    (name, data) for name, data in levels.items() 
                    if data['price'] < current_price
                ]
                if support_levels:
                    nearest = max(support_levels, key=lambda x: x[1]['price'])
                    return {
                        'level_name': nearest[0],
                        'price': nearest[1]['price'],
                        'distance': current_price - nearest[1]['price'],
                        'distance_pct': ((current_price - nearest[1]['price']) / current_price) * 100
                    }
            else:  # resistance
                # Find lowest level above current price
                resistance_levels = [
                    (name, data) for name, data in levels.items() 
                    if data['price'] > current_price
                ]
                if resistance_levels:
                    nearest = min(resistance_levels, key=lambda x: x[1]['price'])
                    return {
                        'level_name': nearest[0],
                        'price': nearest[1]['price'],
                        'distance': nearest[1]['price'] - current_price,
                        'distance_pct': ((nearest[1]['price'] - current_price) / current_price) * 100
                    }
           
            return {}
           
        except Exception as e:
            logger.error(f"Nearest level calculation error: {e}")
            return {}
   
    def analyze_position_opportunity(self, current_price: float, 
                                   fib_analysis: Dict[str, Any],
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading opportunity based on Fibonacci levels"""
        try:
            nearest_support = fib_analysis.get('nearest_support', {})
            nearest_resistance = fib_analysis.get('nearest_resistance', {})
           
            # Determine position type based on price location
            position_rec = {
                'action': 'HOLD',
                'strategy_type': 'calendar_spread',
                'reasoning': '',
                'target_levels': [],
                'stop_levels': [],
                'time_horizon': 'medium_term'
            }
           
            # Near support level - potential bounce
            if nearest_support and nearest_support.get('distance_pct', 100) < 2:
                position_rec.update({
                    'action': 'BUY_CALLS',
                    'strategy_type': 'call_calendar_spread',
                    'reasoning': f"Price near Fibonacci support at ${nearest_support['price']:.2f}",
                    'target_levels': [nearest_resistance.get('price', current_price * 1.05)],
                    'stop_levels': [nearest_support['price'] * 0.98],
                    'confidence_factor': 0.7
                })
           
            # Near resistance level - potential rejection
            elif nearest_resistance and nearest_resistance.get('distance_pct', 100) < 2:
                position_rec.update({
                    'action': 'BUY_PUTS',
                    'strategy_type': 'put_calendar_spread',
                    'reasoning': f"Price near Fibonacci resistance at ${nearest_resistance['price']:.2f}",
                    'target_levels': [nearest_support.get('price', current_price * 0.95)],
                    'stop_levels': [nearest_resistance['price'] * 1.02],
                    'confidence_factor': 0.7
                })
           
            # Between levels - neutral strategy
            else:
                mid_support = nearest_support.get('price', current_price * 0.95)
                mid_resistance = nearest_resistance.get('price', current_price * 1.05)
               
                position_rec.update({
                    'action': 'NEUTRAL',
                    'strategy_type': 'iron_condor',
                    'reasoning': 'Price between Fibonacci levels - range-bound strategy',
                    'target_levels': [mid_support, mid_resistance],
                    'stop_levels': [mid_support * 0.98, mid_resistance * 1.02],
                    'confidence_factor': 0.6
                })
           
            return position_rec
            
        except Exception as e:
            logger.error(f"Position analysis error: {e}")
            return {'action': 'HOLD', 'reasoning': 'Analysis error'}
   
    def calculate_confidence(self, fib_analysis: Dict[str, Any], 
                           current_price: float) -> float:
        """Calculate confidence in Fibonacci analysis"""
        try:
            confidence = 0.5  # Base confidence
           
            # Higher confidence if price is near key Fibonacci levels
            nearest_support = fib_analysis.get('nearest_support', {})
            nearest_resistance = fib_analysis.get('nearest_resistance', {})
           
            # Check proximity to key levels (38.2%, 50%, 61.8%)
            key_levels = [0.382, 0.5, 0.618]
           
            for level_data in [nearest_support, nearest_resistance]:
                if level_data and 'distance_pct' in level_data:
                    distance_pct = level_data['distance_pct']
                   
                    # Check if this is a key Fibonacci level
                    level_name = level_data.get('level_name', '')
                    for key_level in key_levels:
                        if f'{key_level:.3f}' in level_name:
                            if distance_pct < 1:  # Within 1%
                                confidence += 0.3
                            elif distance_pct < 2:  # Within 2%
                                confidence += 0.2
                            elif distance_pct < 3:  # Within 3%
                                confidence += 0.1
           
            # Higher confidence if clear trend direction
            trend = fib_analysis.get('current_trend', 'sideways')
            if trend in ['uptrend', 'downtrend']:
                confidence += 0.1
            
            return min(1.0, confidence)
           
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5
   
    def get_risk_metrics(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for Fibonacci-based position"""
        try:
            position_rec = position.get('position_recommendation', {})
            fibonacci_levels = position.get('fibonacci_levels', {})
            current_price = position.get('current_price', 0)
           
            # Calculate risk based on distance to stop levels
            stop_levels = position_rec.get('stop_levels', [])
            target_levels = position_rec.get('target_levels', [])
           
            risk_metrics = {
                'strategy_type': position_rec.get('strategy_type', 'unknown'),
                'max_risk_pct': 0,
                'reward_risk_ratio': 0,
                'probability_success': 0.5,
                'time_decay_factor': 0.5
            }
           
            if stop_levels and target_levels:
                # Calculate risk and reward
                if position_rec.get('action') == 'BUY_CALLS':
                    max_risk = current_price - min(stop_levels)
                    max_reward = max(target_levels) - current_price
                elif position_rec.get('action') == 'BUY_PUTS':
                    max_risk = max(stop_levels) - current_price
                    max_reward = current_price - min(target_levels)
                else:  # Neutral strategies
                    support = min(target_levels) if target_levels else current_price * 0.95
                    resistance = max(target_levels) if target_levels else current_price * 1.05
                    max_risk = min(current_price - support, resistance - current_price) * 0.5
                    max_reward = max_risk * 0.3  # Typical for neutral strategies
               
                # Calculate metrics
                risk_metrics['max_risk_pct'] = (abs(max_risk) / current_price) * 100
                risk_metrics['reward_risk_ratio'] = abs(max_reward / max_risk) if max_risk != 0 else 0
               
                # Probability based on Fibonacci level strength
                confidence = position.get('confidence', 0.5)
                risk_metrics['probability_success'] = min(0.8, confidence + 0.2)
           
            return risk_metrics
           
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return {'error': str(e)}
    
    def get_config_widget(self):
        """Return configuration widget for Fibonacci settings"""
        from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, 
                                     QSpinBox, QDoubleSpinBox, QCheckBox)
       
        widget = QWidget()
        layout = QVBoxLayout(widget)
       
        # Lookback period
        layout.addWidget(QLabel("Lookback Period (days):"))
        lookback_spin = QSpinBox()
        lookback_spin.setRange(20, 200)
        lookback_spin.setValue(self.config['lookback_period'])
        lookback_spin.valueChanged.connect(
            lambda v: self.config.update({'lookback_period': v})
        )
        layout.addWidget(lookback_spin)
       
        # Support/Resistance threshold
        layout.addWidget(QLabel("S/R Threshold (%):"))
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.5, 10.0)
        threshold_spin.setValue(self.config['support_resistance_threshold'] * 100)
        threshold_spin.valueChanged.connect(
            lambda v: self.config.update({'support_resistance_threshold': v / 100})
        )
        layout.addWidget(threshold_spin)
       
        # Fibonacci levels checkboxes
        layout.addWidget(QLabel("Active Fibonacci Levels:"))
        for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
            checkbox = QCheckBox(f"{level:.1%}")
            checkbox.setChecked(level in self.config['fib_levels'])
            checkbox.stateChanged.connect(
                lambda state, l=level: self.toggle_fib_level(l, state)
            )
            layout.addWidget(checkbox)
       
        return widget
   
    def toggle_fib_level(self, level: float, state: int):
        """Toggle Fibonacci level on/off"""
        if state == 2:  # Checked
            if level not in self.config['fib_levels']:
                self.config['fib_levels'].append(level)
                self.config['fib_levels'].sort()
        else:  # Unchecked
            if level in self.config['fib_levels']:
                self.config['fib_levels'].remove(level)
    
    def get_menu_actions(self) -> List[Dict[str, Any]]:
        """Return menu actions for Fibonacci plugin"""
        return [
            {
                'text': 'Calculate Fibonacci Levels',
                'callback': self.show_fibonacci_analysis,
                'icon': None,
                'shortcut': 'Ctrl+F'
            }
        ]
   
    def show_fibonacci_analysis(self):
        """Show Fibonacci analysis dialog"""
        self.plugin_status_changed.emit("Fibonacci analysis dialog opened")
        # This would open a dialog to show detailed Fibonacci analysis