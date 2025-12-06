"""
Analysis Result Models - Professional Options Calculator Pro
Data structures for analysis results and related components
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, date
from enum import Enum


class RecommendationType(Enum):
    """Trade recommendation types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY" 
    CONSIDER = "CONSIDER"
    WEAK = "WEAK"
    AVOID = "AVOID"


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass
class ConfidenceScore:
    """Confidence score with breakdown"""
    overall_score: float
    factors: List[Tuple[str, float]]  # (factor_name, score_contribution)
    recommendation: str
    risk_level: str
    
    def get_recommendation_enum(self) -> RecommendationType:
        """Get recommendation as enum"""
        try:
            return RecommendationType(self.recommendation)
        except ValueError:
            return RecommendationType.CONSIDER
    
    def get_risk_level_enum(self) -> RiskLevel:
        """Get risk level as enum"""
        try:
            return RiskLevel(self.risk_level)
        except ValueError:
            return RiskLevel.UNKNOWN


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    max_loss_per_contract: float
    max_loss_total: float
    max_profit_per_contract: float
    max_profit_total: float
    break_even_upper: float
    break_even_lower: float
    probability_of_profit: float
    risk_reward_ratio: float
    expected_value: float
    debit_paid: float
    contracts: int
    
    @property
    def is_profitable_setup(self) -> bool:
        """Check if setup has positive expected value"""
        return self.expected_value > 0
    
    @property
    def risk_per_dollar(self) -> float:
        """Risk per dollar invested"""
        if self.debit_paid > 0:
            return self.max_loss_per_contract / self.debit_paid
        return 0.0


@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics"""
    iv30: float
    rv30: float
    iv_rv_ratio: float
    iv_rank: float
    iv_percentile: float
    term_structure_slope: float
    vix_level: float
    hv_percentile: float = 0.0
    
    @property
    def is_high_iv_environment(self) -> bool:
        """Check if in high IV environment"""
        return self.iv_rank > 0.7 or self.iv_percentile > 70
    
    @property
    def is_contango(self) -> bool:
        """Check if term structure is in contango"""
        return self.term_structure_slope < -0.002


@dataclass
class GreeksData:
    """Options Greeks data"""
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    short_leg_delta: float
    short_leg_gamma: float
    short_leg_theta: float
    short_leg_vega: float
    long_leg_delta: float
    long_leg_gamma: float
    long_leg_theta: float
    long_leg_vega: float
    
    @property
    def theta_decay_per_day(self) -> float:
        """Daily theta decay"""
        return self.net_theta
    
    @property
    def is_theta_positive(self) -> bool:
        """Check if net theta is positive"""
        return self.net_theta > 0


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    prob_exceed_1x: float
    prob_exceed_1_5x: float
    prob_exceed_2x: float
    upside_probability: float
    downside_probability: float
    expected_return: float
    value_at_risk_95: float
    simulations_run: int
    price_distribution: Optional[List[float]] = None
    
    @property
    def is_favorable_distribution(self) -> bool:
        """Check if price distribution favors the trade"""
        return 40 <= self.prob_exceed_1x <= 60  # Ideal for calendar spreads


@dataclass
class OptionsData:
    """Options chain and related data"""
    symbol: str
    strike: float
    short_expiry: str
    long_expiry: str
    short_premium: float
    long_premium: float
    debit: float
    average_volume: float
    open_interest: float
    bid_ask_spread: float = 0.0
    
    @property
    def net_debit(self) -> float:
        """Net debit for the spread"""
        return self.long_premium - self.short_premium
    
    @property
    def is_liquid(self) -> bool:
        """Check if options are liquid enough"""
        return self.average_volume > 50000 and self.open_interest > 100


@dataclass
class MLPrediction:
    """Machine learning prediction results"""
    success_probability: float
    confidence_interval: Tuple[float, float]
    model_accuracy: float
    feature_importance: Dict[str, float]
    prediction_timestamp: datetime
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if ML prediction has high confidence"""
        return self.model_accuracy > 0.7 and self.success_probability > 0.6


@dataclass
class AnalysisResult:
    """Complete analysis result container"""
    # Basic information
    symbol: str
    timestamp: datetime
    request_id: str
    current_price: float
    analysis_duration: float
    
    # Core analysis components
    confidence_score: ConfidenceScore
    volatility_metrics: VolatilityMetrics
    options_data: OptionsData
    greeks: GreeksData
    monte_carlo_result: MonteCarloResult
    risk_metrics: RiskMetrics
    
    # Optional components
    ml_prediction: Optional[MLPrediction] = None
    earnings_date: Optional[date] = None
    days_to_earnings: int = 90
    
    # Additional metadata
    sector: str = "Unknown"
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id,
            'current_price': self.current_price,
            'analysis_duration': self.analysis_duration,
            'sector': self.sector,
            'days_to_earnings': self.days_to_earnings,
            
            # Confidence
            'confidence_score': self.confidence_score.overall_score,
            'recommendation': self.confidence_score.recommendation,
            'risk_level': self.confidence_score.risk_level,
            
            # Volatility
            'iv30': self.volatility_metrics.iv30,
            'rv30': self.volatility_metrics.rv30,
            'iv_rv_ratio': self.volatility_metrics.iv_rv_ratio,
            'iv_rank': self.volatility_metrics.iv_rank,
            'iv_percentile': self.volatility_metrics.iv_percentile,
            'vix_level': self.volatility_metrics.vix_level,
            
            # Options
            'strike': self.options_data.strike,
            'debit': self.options_data.debit,
            'short_premium': self.options_data.short_premium,
            'long_premium': self.options_data.long_premium,
            
            # Greeks
            'net_delta': self.greeks.net_delta,
            'net_gamma': self.greeks.net_gamma,
            'net_theta': self.greeks.net_theta,
            'net_vega': self.greeks.net_vega,
            
            # Monte Carlo
            'prob_exceed_1x': self.monte_carlo_result.prob_exceed_1x,
            'expected_return': self.monte_carlo_result.expected_return,
            
            # Risk
            'max_loss': self.risk_metrics.max_loss_total,
            'max_profit': self.risk_metrics.max_profit_total,
            'probability_of_profit': self.risk_metrics.probability_of_profit,
            'expected_value': self.risk_metrics.expected_value,
        }
        
        # Add ML prediction if available
        if self.ml_prediction:
            result['ml_success_probability'] = self.ml_prediction.success_probability
            result['ml_confidence'] = self.ml_prediction.model_accuracy
        
        # Add earnings date if available
        if self.earnings_date:
            result['earnings_date'] = self.earnings_date.isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary"""
        # This would implement the reverse conversion
        # For now, returning a basic structure
        raise NotImplementedError("from_dict not yet implemented")
    
    @property
    def is_tradeable(self) -> bool:
        """Check if the setup is tradeable"""
        return (
            self.confidence_score.overall_score >= 50 and
            self.options_data.is_liquid and
            self.risk_metrics.is_profitable_setup
        )
    
    @property
    def trade_summary(self) -> str:
        """Get a concise trade summary"""
        return (
            f"{self.symbol} Calendar Spread: "
            f"{self.confidence_score.overall_score:.1f}% confidence, "
            f"${self.risk_metrics.max_loss_total:.0f} max risk, "
            f"{self.risk_metrics.probability_of_profit:.1f}% prob profit"
        )
    
    def get_display_data(self) -> Dict[str, str]:
        """Get formatted data for UI display"""
        return {
            'Symbol': self.symbol,
            'Price': f"${self.current_price:.2f}",
            'Confidence': f"{self.confidence_score.overall_score:.1f}%",
            'Recommendation': self.confidence_score.recommendation,
            'Risk Level': self.confidence_score.risk_level,
            'IV Rank': f"{self.volatility_metrics.iv_rank:.2f}",
            'IV/RV Ratio': f"{self.volatility_metrics.iv_rv_ratio:.2f}",
            'Days to Earnings': f"{self.days_to_earnings} days",
            'Strike': f"${self.options_data.strike:.2f}",
            'Debit': f"${self.options_data.debit:.2f}",
            'Max Loss': f"${self.risk_metrics.max_loss_total:.0f}",
            'Max Profit': f"${self.risk_metrics.max_profit_total:.0f}",
            'Prob Profit': f"{self.risk_metrics.probability_of_profit:.1f}%",
            'Expected Value': f"${self.risk_metrics.expected_value:.0f}",
            'Net Theta': f"${self.greeks.net_theta:.2f}/day",
            'Monte Carlo 1x': f"{self.monte_carlo_result.prob_exceed_1x:.1f}%",
            'Analysis Time': f"{self.analysis_duration:.1f}s"
        }


@dataclass
class BatchAnalysisResult:
    """Results from batch analysis"""
    total_symbols: int
    completed_symbols: int
    failed_symbols: int
    results: List[AnalysisResult]
    errors: Dict[str, str]  # symbol -> error_message
    start_time: datetime
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_symbols == 0:
            return 0.0
        return self.completed_symbols / self.total_symbols
    
    @property
    def duration(self) -> float:
        """Get total duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_top_opportunities(self, min_confidence: float = 60.0) -> List[AnalysisResult]:
        """Get top trading opportunities"""
        opportunities = [
            result for result in self.results 
            if result.confidence_score.overall_score >= min_confidence
        ]
        
        # Sort by confidence score
        opportunities.sort(
            key=lambda x: x.confidence_score.overall_score, 
            reverse=True
        )
        
        return opportunities
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.results:
            return {'error': 'No results available'}
        
        confidence_scores = [r.confidence_score.overall_score for r in self.results]
        
        return {
            'total_analyzed': len(self.results),
            'avg_confidence': sum(confidence_scores) / len(confidence_scores),
            'max_confidence': max(confidence_scores),
            'min_confidence': min(confidence_scores),
            'opportunities_60plus': len([s for s in confidence_scores if s >= 60]),
            'opportunities_70plus': len([s for s in confidence_scores if s >= 70]),
            'success_rate': self.success_rate,
            'duration': self.duration,
            'symbols_per_second': len(self.results) / max(self.duration, 1)
        }


# Additional utility functions
def create_empty_analysis_result(symbol: str, error_message: str) -> AnalysisResult:
    """Create an empty analysis result for error cases"""
    return AnalysisResult(
        symbol=symbol,
        timestamp=datetime.now(),
        request_id=f"error_{symbol}_{int(datetime.now().timestamp())}",
        current_price=0.0,
        analysis_duration=0.0,
        confidence_score=ConfidenceScore(
            overall_score=0.0,
            factors=[("Error", 0.0)],
            recommendation="ERROR",
            risk_level="UNKNOWN"
        ),
        volatility_metrics=VolatilityMetrics(
            iv30=0.0, rv30=0.0, iv_rv_ratio=0.0, iv_rank=0.0,
            iv_percentile=0.0, term_structure_slope=0.0,
            vix_level=20.0, hv_percentile=0.0
        ),
        options_data=OptionsData(
            symbol=symbol, strike=0.0, short_expiry="", long_expiry="",
            short_premium=0.0, long_premium=0.0, debit=0.0,
            average_volume=0.0, open_interest=0.0
        ),
        greeks=GreeksData(
            net_delta=0.0, net_gamma=0.0, net_theta=0.0, net_vega=0.0,
            short_leg_delta=0.0, short_leg_gamma=0.0, short_leg_theta=0.0, short_leg_vega=0.0,
            long_leg_delta=0.0, long_leg_gamma=0.0, long_leg_theta=0.0, long_leg_vega=0.0
        ),
        monte_carlo_result=MonteCarloResult(
            prob_exceed_1x=0.0, prob_exceed_1_5x=0.0, prob_exceed_2x=0.0,
            upside_probability=50.0, downside_probability=50.0,
            expected_return=0.0, value_at_risk_95=0.0, simulations_run=0
        ),
        risk_metrics=RiskMetrics(
            max_loss_per_contract=0.0, max_loss_total=0.0,
            max_profit_per_contract=0.0, max_profit_total=0.0,
            break_even_upper=0.0, break_even_lower=0.0,
            probability_of_profit=0.0, risk_reward_ratio=0.0,
            expected_value=0.0, debit_paid=0.0, contracts=1
        )
    )