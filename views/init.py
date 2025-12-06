"""
Professional Options Calculator - Views Module
PySide6 interface components and views
"""

from .base_view import (
    BaseView,
    ProfessionalTable,
    StatusBar,
    CollapsibleGroupBox,
    MetricsDisplayWidget,
    LoadingOverlay
)

from .components import (
    # Chart components
    BaseChartWidget,
    PriceDistributionChart,
    IVTermStructureChart,
    ProfitLossChart,
    PerformanceChart,
    VolumeAnalysisChart,
    MultiMetricDashboard,
    
    # Widget components
    AnalysisInputWidget,
    ConfidenceIndicator,
    TradingMetricsPanel,
    ProgressWidget,
    OptionChainWidget,
    RiskManagementWidget
)

from .dialogs import (
    SettingsDialog,
    AboutDialog
)

__all__ = [
    # Base view components
    'BaseView',
    'ProfessionalTable',
    'StatusBar',
    'CollapsibleGroupBox',
    'MetricsDisplayWidget',
    'LoadingOverlay',
    
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
    'RiskManagementWidget',
    
    # Dialogs
    'SettingsDialog',
    'AboutDialog'
]

__version__ = "1.0.0"