"""
Professional Options Calculator - View Components
Custom widgets and components for the PySide6 interface
"""

from .charts import (
    BaseChartWidget,
    PriceDistributionChart,
    IVTermStructureChart,
    ProfitLossChart,
    PerformanceChart,
    VolumeAnalysisChart,
    MultiMetricDashboard
)

from .widgets import (
    AnalysisInputWidget,
    ConfidenceIndicator,
    TradingMetricsPanel,
    ProgressWidget,
    OptionChainWidget,
    RiskManagementWidget
)

__all__ = [
    # Chart components
    'BaseChartWidget',
    'PriceDistributionChart',
    'IVTermStructureChart',
    'ProfitLossChart',
    'PerformanceChart',
    'VolumeAnalysisChart',
    'MultiMetricDashboard',
    
    # Widget components
    'AnalysisInputWidget',
    'ConfidenceIndicator',
    'TradingMetricsPanel',
    'ProgressWidget',
    'OptionChainWidget',
    'RiskManagementWidget'
]

__version__ = "1.0.0"