"""
Chart components for data visualization
Professional Options Calculator - Chart Components
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseChartWidget(QWidget):
    """Base chart widget with common functionality"""
    
    chart_clicked = Signal(object)  # Emits click data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.setup_ui()
        self.setup_styling()
        
    def setup_ui(self):
        """Setup chart UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar for chart controls
        self.toolbar = QHBoxLayout()
        self.create_toolbar()
        
        layout.addLayout(self.toolbar)
        layout.addWidget(self.canvas)
        
    def create_toolbar(self):
        """Create chart toolbar - to be implemented by subclasses"""
        pass
        
    def setup_styling(self):
        """Setup professional chart styling"""
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        
        # Configure figure
        self.figure.patch.set_facecolor('#2b2b2b')
        
        # Default colors
        self.colors = {
            'primary': '#60a3d9',
            'secondary': '#0d7377',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'grid': '#4a4a4a',
            'text': '#ffffff'
        }
    
    def clear_chart(self):
        """Clear the chart"""
        self.figure.clear()
        self.canvas.draw()
    
    def refresh_chart(self):
        """Refresh the chart display"""
        self.canvas.draw()


class PriceDistributionChart(BaseChartWidget):
    """Chart for displaying price distribution from Monte Carlo"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.price_data = None
        self.current_price = None
        
    def create_toolbar(self):
        """Create toolbar for price distribution chart"""
        title = QLabel("Price Distribution Analysis")
        title.setStyleSheet("font-weight: bold; color: #60a3d9; font-size: 11pt;")
        
        export_btn = QPushButton("Export")
        export_btn.setMaximumWidth(80)
        export_btn.clicked.connect(self.export_chart)
        
        self.toolbar.addWidget(title)
        self.toolbar.addStretch()
        self.toolbar.addWidget(export_btn)
    
    def plot_distribution(self, price_data: np.ndarray, current_price: float,
                         expected_move: float, confidence_intervals: Dict[str, float]):
        """Plot price distribution with analysis"""
        self.price_data = price_data
        self.current_price = current_price
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot histogram
        n, bins, patches = ax.hist(price_data, bins=50, alpha=0.7, 
                                  color=self.colors['primary'], 
                                  edgecolor='none', density=True)
        
        # Add current price line
        ax.axvline(current_price, color=self.colors['danger'], 
                  linestyle='--', linewidth=2, label=f'Current Price: ${current_price:.2f}')
        
        # Add expected move lines
        move_up = current_price * (1 + expected_move)
        move_down = current_price * (1 - expected_move)
        
        ax.axvline(move_up, color=self.colors['success'], 
                  linestyle='--', linewidth=2, alpha=0.8, 
                  label=f'+1σ Move: ${move_up:.2f}')
        ax.axvline(move_down, color=self.colors['success'], 
                  linestyle='--', linewidth=2, alpha=0.8,
                  label=f'-1σ Move: ${move_down:.2f}')
        
        # Add confidence intervals
        for level, price in confidence_intervals.items():
            ax.axvline(price, color=self.colors['warning'], 
                      linestyle=':', alpha=0.6, 
                      label=f'{level}: ${price:.2f}')
        
        # Styling
        ax.set_xlabel('Price at Expiration ($)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Monte Carlo Price Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.legend()
        
        # Set colors
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.label.set_color(self.colors['text'])
        ax.yaxis.label.set_color(self.colors['text'])
        ax.title.set_color(self.colors['text'])
        
        self.figure.tight_layout()
        self.refresh_chart()
    
    def export_chart(self):
        """Export chart to file"""
        try:
            from PySide6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Chart", "price_distribution.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
            )
            
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight',
                                   facecolor='#2b2b2b')
                logger.info(f"Chart exported to {filename}")
                
        except Exception as e:
            logger.error(f"Error exporting chart: {e}")


class IVTermStructureChart(BaseChartWidget):
    """Chart for displaying IV term structure"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def create_toolbar(self):
        """Create toolbar for IV term structure chart"""
        title = QLabel("IV Term Structure")
        title.setStyleSheet("font-weight: bold; color: #60a3d9; font-size: 11pt;")
        
        self.toolbar.addWidget(title)
        self.toolbar.addStretch()
    
    def plot_term_structure(self, days_to_expiry: List[int], 
                           implied_vols: List[float],
                           earnings_date: Optional[int] = None,
                           short_dte: Optional[int] = None,
                           long_dte: Optional[int] = None):
        """Plot IV term structure"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Convert to percentages
        iv_pct = [iv * 100 for iv in implied_vols]
        
        # Plot term structure
        ax.plot(days_to_expiry, iv_pct, 'o-', 
               color=self.colors['primary'], linewidth=2, markersize=6)
        
        # Highlight short and long expirations
        if short_dte and short_dte in days_to_expiry:
            idx = days_to_expiry.index(short_dte)
            ax.plot(short_dte, iv_pct[idx], 'o', 
                   color=self.colors['danger'], markersize=10, 
                   label='Short Leg')
        
        if long_dte and long_dte in days_to_expiry:
            idx = days_to_expiry.index(long_dte)
            ax.plot(long_dte, iv_pct[idx], 'o', 
                   color=self.colors['success'], markersize=10,
                   label='Long Leg')
        
        # Add earnings line
        if earnings_date:
            ax.axvline(earnings_date, color=self.colors['warning'], 
                      linestyle='--', linewidth=2, alpha=0.8,
                      label='Earnings')
        
        # Styling
        ax.set_xlabel('Days to Expiration')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title('Implied Volatility Term Structure', fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
        if short_dte or long_dte or earnings_date:
            ax.legend()
        
        # Set colors
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.label.set_color(self.colors['text'])
        ax.yaxis.label.set_color(self.colors['text'])
        ax.title.set_color(self.colors['text'])
       
        self.figure.tight_layout()
        self.refresh_chart()


class ProfitLossChart(BaseChartWidget):
   """Chart for displaying P/L analysis"""
   
   def __init__(self, parent=None):
       super().__init__(parent)
       
   def create_toolbar(self):
       """Create toolbar for P/L chart"""
       title = QLabel("Profit/Loss Analysis")
       title.setStyleSheet("font-weight: bold; color: #60a3d9; font-size: 11pt;")
       
       self.show_greeks_btn = QPushButton("Show Greeks")
       self.show_greeks_btn.setCheckable(True)
       self.show_greeks_btn.setMaximumWidth(100)
       self.show_greeks_btn.clicked.connect(self.toggle_greeks_display)
       
       self.toolbar.addWidget(title)
       self.toolbar.addStretch()
       self.toolbar.addWidget(self.show_greeks_btn)
   
   def plot_profit_loss(self, price_range: np.ndarray, pl_values: np.ndarray,
                       current_price: float, strike_price: float,
                       max_profit: float, max_loss: float,
                       breakeven_points: List[float] = None):
       """Plot profit/loss diagram"""
       self.figure.clear()
       ax = self.figure.add_subplot(111)
       
       # Plot P/L curve
       ax.plot(price_range, pl_values, '-', 
              color=self.colors['primary'], linewidth=3, label='P/L')
       
       # Fill profit and loss areas
       ax.fill_between(price_range, pl_values, 0, 
                      where=(pl_values > 0), 
                      color=self.colors['success'], alpha=0.3, label='Profit')
       ax.fill_between(price_range, pl_values, 0, 
                      where=(pl_values < 0), 
                      color=self.colors['danger'], alpha=0.3, label='Loss')
       
       # Add reference lines
       ax.axhline(0, color=self.colors['text'], linestyle='-', alpha=0.5)
       ax.axvline(current_price, color=self.colors['warning'], 
                 linestyle='--', linewidth=2, label=f'Current: ${current_price:.2f}')
       ax.axvline(strike_price, color=self.colors['secondary'], 
                 linestyle='--', alpha=0.8, label=f'Strike: ${strike_price:.2f}')
       
       # Add breakeven points
       if breakeven_points:
           for i, be_price in enumerate(breakeven_points):
               ax.axvline(be_price, color='orange', linestyle=':', 
                         alpha=0.8, label=f'Breakeven {i+1}: ${be_price:.2f}')
       
       # Add max profit/loss annotations
       ax.text(0.02, 0.98, f'Max Profit: ${max_profit:.2f}', 
              transform=ax.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor=self.colors['success'], alpha=0.7))
       ax.text(0.02, 0.88, f'Max Loss: ${max_loss:.2f}', 
              transform=ax.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor=self.colors['danger'], alpha=0.7))
       
       # Styling
       ax.set_xlabel('Stock Price at Expiration ($)')
       ax.set_ylabel('Profit/Loss ($)')
       ax.set_title('Calendar Spread Profit/Loss Diagram', fontweight='bold')
       ax.grid(True, alpha=0.3, color=self.colors['grid'])
       ax.legend()
       
       # Set colors
       ax.set_facecolor('#2b2b2b')
       ax.tick_params(colors=self.colors['text'])
       ax.xaxis.label.set_color(self.colors['text'])
       ax.yaxis.label.set_color(self.colors['text'])
       ax.title.set_color(self.colors['text'])
       
       self.figure.tight_layout()
       self.refresh_chart()
   
   def toggle_greeks_display(self):
       """Toggle Greeks overlay on P/L chart"""
       # This would add Greeks curves to the chart
       # Implementation depends on Greeks data availability
       pass


class PerformanceChart(BaseChartWidget):
   """Chart for displaying performance metrics and backtesting results"""
   
   def __init__(self, parent=None):
       super().__init__(parent)
       
   def create_toolbar(self):
       """Create toolbar for performance chart"""
       title = QLabel("Performance Analysis")
       title.setStyleSheet("font-weight: bold; color: #60a3d9; font-size: 11pt;")
       
       self.chart_type = QPushButton("Cumulative P/L")
       self.chart_type.setMaximumWidth(120)
       # Add menu for different chart types
       
       self.toolbar.addWidget(title)
       self.toolbar.addStretch()
       self.toolbar.addWidget(self.chart_type)
   
   def plot_cumulative_performance(self, dates: List[str], 
                                 cumulative_pl: List[float],
                                 individual_trades: List[Dict[str, Any]] = None):
       """Plot cumulative performance over time"""
       self.figure.clear()
       ax = self.figure.add_subplot(111)
       
       # Convert dates to datetime if needed
       if isinstance(dates[0], str):
           from datetime import datetime
           dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
       
       # Plot cumulative P/L
       ax.plot(dates, cumulative_pl, '-', 
              color=self.colors['primary'], linewidth=2, label='Cumulative P/L')
       
       # Add individual trade markers
       if individual_trades:
           winning_dates = []
           winning_values = []
           losing_dates = []
           losing_values = []
           
           for i, trade in enumerate(individual_trades):
               if i < len(dates) and i < len(cumulative_pl):
                   if trade.get('pl', 0) > 0:
                       winning_dates.append(dates[i])
                       winning_values.append(cumulative_pl[i])
                   else:
                       losing_dates.append(dates[i])
                       losing_values.append(cumulative_pl[i])
           
           if winning_dates:
               ax.scatter(winning_dates, winning_values, 
                         color=self.colors['success'], s=50, 
                         alpha=0.8, label='Winning Trades', zorder=5)
           
           if losing_dates:
               ax.scatter(losing_dates, losing_values, 
                         color=self.colors['danger'], s=50, 
                         alpha=0.8, label='Losing Trades', zorder=5)
       
       # Add zero line
       ax.axhline(0, color=self.colors['text'], linestyle='-', alpha=0.5)
       
       # Styling
       ax.set_xlabel('Date')
       ax.set_ylabel('Cumulative P/L ($)')
       ax.set_title('Strategy Performance Over Time', fontweight='bold')
       ax.grid(True, alpha=0.3, color=self.colors['grid'])
       ax.legend()
       
       # Format x-axis dates
       import matplotlib.dates as mdates
       ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
       ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
       plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
       
       # Set colors
       ax.set_facecolor('#2b2b2b')
       ax.tick_params(colors=self.colors['text'])
       ax.xaxis.label.set_color(self.colors['text'])
       ax.yaxis.label.set_color(self.colors['text'])
       ax.title.set_color(self.colors['text'])
       
       self.figure.tight_layout()
       self.refresh_chart()


class VolumeAnalysisChart(BaseChartWidget):
   """Chart for volume and liquidity analysis"""
   
   def __init__(self, parent=None):
       super().__init__(parent)
       
   def create_toolbar(self):
       """Create toolbar for volume analysis chart"""
       title = QLabel("Volume Analysis")
       title.setStyleSheet("font-weight: bold; color: #60a3d9; font-size: 11pt;")
       
       self.toolbar.addWidget(title)
       self.toolbar.addStretch()
   
   def plot_volume_analysis(self, dates: List[str], volumes: List[float],
                          prices: List[float], avg_volume: float):
       """Plot volume analysis with price overlay"""
       self.figure.clear()
       
       # Create subplots
       ax1 = self.figure.add_subplot(211)  # Price
       ax2 = self.figure.add_subplot(212)  # Volume
       
       # Convert dates
       if isinstance(dates[0], str):
           from datetime import datetime
           dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
       
       # Plot price
       ax1.plot(dates, prices, color=self.colors['primary'], linewidth=2)
       ax1.set_ylabel('Price ($)')
       ax1.set_title('Price and Volume Analysis', fontweight='bold')
       ax1.grid(True, alpha=0.3, color=self.colors['grid'])
       
       # Plot volume
       colors = [self.colors['success'] if v > avg_volume else self.colors['danger'] 
                for v in volumes]
       ax2.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
       ax2.axhline(avg_volume, color=self.colors['warning'], 
                  linestyle='--', linewidth=2, label=f'Avg Volume: {avg_volume:,.0f}')
       ax2.set_ylabel('Volume')
       ax2.set_xlabel('Date')
       ax2.legend()
       ax2.grid(True, alpha=0.3, color=self.colors['grid'])
       
       # Format x-axis dates for both subplots
       import matplotlib.dates as mdates
       for ax in [ax1, ax2]:
           ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
           ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
           plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
           
           # Set colors
           ax.set_facecolor('#2b2b2b')
           ax.tick_params(colors=self.colors['text'])
           ax.xaxis.label.set_color(self.colors['text'])
           ax.yaxis.label.set_color(self.colors['text'])
       
       ax1.title.set_color(self.colors['text'])
       
       self.figure.tight_layout()
       self.refresh_chart()


class MultiMetricDashboard(QWidget):
   """Dashboard widget containing multiple charts and metrics"""
   
   def __init__(self, parent=None):
       super().__init__(parent)
       self.charts = {}
       self.setup_ui()
       
   def setup_ui(self):
       """Setup dashboard UI"""
       layout = QVBoxLayout(self)
       
       # Header
       header = QLabel("Analysis Dashboard")
       header.setStyleSheet("""
           QLabel {
               font-size: 16pt;
               font-weight: bold;
               color: #60a3d9;
               padding: 10px;
               border-bottom: 2px solid #4a4a4a;
               margin-bottom: 10px;
           }
       """)
       layout.addWidget(header)
       
       # Create tab widget for different chart views
       from PySide6.QtWidgets import QTabWidget
       self.tab_widget = QTabWidget()
       self.tab_widget.setStyleSheet("""
           QTabWidget::pane {
               border: 2px solid #4a4a4a;
               background-color: #2b2b2b;
           }
           
           QTabBar::tab {
               background-color: #404040;
               color: #ffffff;
               padding: 8px 16px;
               margin-right: 2px;
               margin-bottom: 2px;
               border-top-left-radius: 4px;
               border-top-right-radius: 4px;
           }
           
           QTabBar::tab:selected {
               background-color: #60a3d9;
           }
           
           QTabBar::tab:hover {
               background-color: #5a5a5a;
           }
       """)
       
       # Add chart tabs
       self.add_chart_tabs()
       
       layout.addWidget(self.tab_widget)
       
   def add_chart_tabs(self):
       """Add different chart tabs to the dashboard"""
       # Price Distribution
       self.charts['price_dist'] = PriceDistributionChart()
       self.tab_widget.addTab(self.charts['price_dist'], "Price Distribution")
       
       # IV Term Structure
       self.charts['iv_term'] = IVTermStructureChart()
       self.tab_widget.addTab(self.charts['iv_term'], "IV Term Structure")
       
       # P/L Analysis
       self.charts['pl_analysis'] = ProfitLossChart()
       self.tab_widget.addTab(self.charts['pl_analysis'], "P/L Analysis")
       
       # Performance
       self.charts['performance'] = PerformanceChart()
       self.tab_widget.addTab(self.charts['performance'], "Performance")
       
       # Volume Analysis
       self.charts['volume'] = VolumeAnalysisChart()
       self.tab_widget.addTab(self.charts['volume'], "Volume Analysis")
   
   def update_charts(self, analysis_data: Dict[str, Any]):
       """Update all charts with new analysis data"""
       try:
           # Update price distribution chart
           if 'monte_carlo_results' in analysis_data:
               mc_data = analysis_data['monte_carlo_results']
               if 'price_distribution' in mc_data:
                   self.charts['price_dist'].plot_distribution(
                       price_data=mc_data['price_distribution'],
                       current_price=analysis_data.get('underlying_price', 0),
                       expected_move=mc_data.get('expected_move_pct', 0.08),
                       confidence_intervals=mc_data.get('confidence_intervals', {})
                   )
           
           # Update IV term structure
           if 'volatility_metrics' in analysis_data:
               vol_data = analysis_data['volatility_metrics']
               if 'term_structure' in vol_data:
                   ts_data = vol_data['term_structure']
                   self.charts['iv_term'].plot_term_structure(
                       days_to_expiry=ts_data.get('days', []),
                       implied_vols=ts_data.get('ivs', []),
                       earnings_date=analysis_data.get('days_to_earnings'),
                       short_dte=ts_data.get('short_dte'),
                       long_dte=ts_data.get('long_dte')
                   )
           
           # Update other charts as needed...
           
       except Exception as e:
           logger.error(f"Error updating charts: {e}")
   
   def get_chart(self, chart_name: str) -> Optional[BaseChartWidget]:
       """Get specific chart by name"""
       return self.charts.get(chart_name)
   
   def export_all_charts(self):
       """Export all charts to files"""
       try:
           from PySide6.QtWidgets import QFileDialog
           
           directory = QFileDialog.getExistingDirectory(
               self, "Select Export Directory"
           )
           
           if directory:
               for name, chart in self.charts.items():
                   filename = f"{directory}/chart_{name}.png"
                   chart.figure.savefig(filename, dpi=300, bbox_inches='tight',
                                      facecolor='#2b2b2b')
               
               logger.info(f"All charts exported to {directory}")
               
       except Exception as e:
           logger.error(f"Error exporting charts: {e}")