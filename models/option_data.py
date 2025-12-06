from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import date, datetime
import pandas as pd
import numpy as np
from enum import Enum

class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"

class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

@dataclass
class OptionContract:
    """Single option contract data"""
    symbol: str
    strike: float
    expiration: date
    option_type: OptionType
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last if self.last > 0 else 0.0
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask - self.bid if self.ask > self.bid else 0.0
    
    @property
    def spread_percentage(self) -> float:
        """Calculate spread as percentage of mid price"""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0.0
    
    @property
    def moneyness(self) -> str:
        """Calculate moneyness relative to underlying price"""
        return "ATM"  # Placeholder
    
    def is_liquid(self, min_volume: int = 100, max_spread_pct: float = 10.0) -> bool:
        """Check if option is liquid enough for trading"""
        return (self.volume >= min_volume and 
                self.open_interest > 0 and 
                self.spread_percentage <= max_spread_pct)

@dataclass
class OptionChain:
    """Complete option chain for a given expiration"""
    underlying_symbol: str
    expiration: date
    underlying_price: float
    calls: List[OptionContract] = field(default_factory=list)
    puts: List[OptionContract] = field(default_factory=list)
    
    def get_strikes(self) -> List[float]:
        """Get all available strikes"""
        call_strikes = [c.strike for c in self.calls]
        put_strikes = [p.strike for p in self.puts]
        return sorted(list(set(call_strikes + put_strikes)))
    
    def get_atm_strike(self) -> float:
        """Get at-the-money strike"""
        strikes = self.get_strikes()
        if not strikes:
            return self.underlying_price
        return min(strikes, key=lambda x: abs(x - self.underlying_price))
    
    def get_call_by_strike(self, strike: float) -> Optional[OptionContract]:
        """Get call option by strike"""
        for call in self.calls:
            if call.strike == strike:
                return call
        return None
    
    def get_put_by_strike(self, strike: float) -> Optional[OptionContract]:
        """Get put option by strike"""
        for put in self.puts:
            if put.strike == strike:
                return put
        return None
    
    def get_liquid_strikes(self, min_volume: int = 100) -> List[float]:
        """Get strikes with liquid options"""
        liquid_strikes = set()
        for call in self.calls:
            if call.is_liquid(min_volume):
                liquid_strikes.add(call.strike)
        for put in self.puts:
            if put.is_liquid(min_volume):
                liquid_strikes.add(put.strike)
        return sorted(list(liquid_strikes))
    
    def calculate_straddle_price(self, strike: float) -> Optional[float]:
        """Calculate straddle price at given strike"""
        call = self.get_call_by_strike(strike)
        put = self.get_put_by_strike(strike)
        if call and put:
            return call.mid_price + put.mid_price
        return None
    
    def calculate_expected_move(self) -> Optional[float]:
        """Calculate expected move based on ATM straddle"""
        atm_strike = self.get_atm_strike()
        straddle_price = self.calculate_straddle_price(atm_strike)
        if straddle_price and self.underlying_price > 0:
            return straddle_price / self.underlying_price
        return None

@dataclass
class CalendarSpread:
    """Calendar spread option strategy"""
    underlying_symbol: str
    strike: float
    short_option: OptionContract
    long_option: OptionContract
    contracts: int = 1
    
    @property
    def debit(self) -> float:
        """Calculate net debit paid"""
        return (self.long_option.mid_price - self.short_option.mid_price) * self.contracts * 100
    
    @property
    def max_loss(self) -> float:
        """Calculate maximum loss"""
        return max(0, self.debit)
    
    @property
    def days_to_short_expiration(self) -> int:
        """Days to short leg expiration"""
        return (self.short_option.expiration - date.today()).days
    
    @property
    def days_to_long_expiration(self) -> int:
        """Days to long leg expiration"""
        return (self.long_option.expiration - date.today()).days
    
    @property
    def time_spread(self) -> int:
        """Days between expirations"""
        return self.days_to_long_expiration - self.days_to_short_expiration
    
    def calculate_net_greeks(self) -> Dict[str, float]:
        """Calculate net Greeks for the spread"""
        return {
            'delta': self.long_option.delta - self.short_option.delta,
            'gamma': self.long_option.gamma - self.short_option.gamma,
            'theta': self.long_option.theta - self.short_option.theta,
            'vega': self.long_option.vega - self.short_option.vega,
            'rho': self.long_option.rho - self.short_option.rho,
        }
    
    def estimate_pnl(self, new_underlying_price: float, iv_change: float = 0.0, days_passed: int = 0) -> float:
        """Estimate P&L for given scenario"""
        net_greeks = self.calculate_net_greeks()
        price_change = new_underlying_price - self.short_option.symbol
        delta_pnl = net_greeks['delta'] * price_change * self.contracts * 100
        theta_pnl = net_greeks['theta'] * days_passed * self.contracts * 100
        vega_pnl = net_greeks['vega'] * iv_change * self.contracts * 100
        return delta_pnl + theta_pnl + vega_pnl

@dataclass
class IVTermStructure:
    """Implied volatility term structure"""
    underlying_symbol: str
    date: date
    data_points: List[Tuple[int, float]] = field(default_factory=list)
    
    def add_point(self, days_to_expiry: int, iv: float):
        """Add a data point to the term structure"""
        self.data_points.append((days_to_expiry, iv))
        self.data_points.sort(key=lambda x: x[0])
    
    def get_iv_at_dte(self, days_to_expiry: int) -> Optional[float]:
        """Get IV at specific days to expiry (interpolated)"""
        if not self.data_points:
            return None
        for dte, iv in self.data_points:
            if dte == days_to_expiry:
                return iv
        dtes = [point[0] for point in self.data_points]
        ivs = [point[1] for point in self.data_points]
        if days_to_expiry < min(dtes):
            return ivs[0]
        elif days_to_expiry > max(dtes):
            return ivs[-1]
        else:
            return np.interp(days_to_expiry, dtes, ivs)
    
    def calculate_slope(self, dte1: int, dte2: int) -> Optional[float]:
        """Calculate slope between two DTEs"""
        iv1 = self.get_iv_at_dte(dte1)
        iv2 = self.get_iv_at_dte(dte2)
        if iv1 is not None and iv2 is not None and dte2 != dte1:
            return (iv2 - iv1) / (dte2 - dte1)
        return None
    
    @property
    def is_contango(self) -> bool:
        """Check if term structure is in contango (upward sloping)"""
        if len(self.data_points) < 2:
            return False
        first_iv = self.data_points[0][1]
        last_iv = self.data_points[-1][1]
        return last_iv > first_iv
    
    @property
    def is_backwardation(self) -> bool:
        """Check if term structure is in backwardation (downward sloping)"""
        return not self.is_contango

@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics"""
    underlying_symbol: str
    current_iv: float
    historical_vol_30d: float
    historical_vol_60d: float
    iv_rank: float
    iv_percentile: float
    hv_iv_ratio: float
    
    @property
    def iv_rank_description(self) -> str:
        """Get description of IV rank"""
        if self.iv_rank >= 0.8:
            return "Very High"
        elif self.iv_rank >= 0.6:
            return "High"
        elif self.iv_rank >= 0.4:
            return "Medium"
        elif self.iv_rank >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    @property
    def is_iv_elevated(self) -> bool:
        """Check if IV is elevated relative to HV"""
        return self.hv_iv_ratio >= 1.2
    
    @property
    def is_mean_reversion_candidate(self) -> bool:
        """Check if this is a mean reversion candidate"""
        return self.iv_rank > 0.7 and self.is_iv_elevated

@dataclass
class MarketRegime:
    """Market regime classification"""
    vix_level: float
    vix_trend: str
    regime_type: str
    regime_confidence: float
    
    @classmethod
    def from_vix(cls, vix_level: float, vix_history: List[float] = None):
        """Create market regime from VIX data"""
        if vix_level < 16:
            regime_type = "low"
        elif vix_level < 24:
            regime_type = "normal"
        elif vix_level < 32:
            regime_type = "elevated"
        else:
            regime_type = "crisis"
        
        vix_trend = "stable"
        if vix_history and len(vix_history) >= 5:
            recent_avg = np.mean(vix_history[-5:])
            older_avg = np.mean(vix_history[-10:-5]) if len(vix_history) >= 10 else recent_avg
            if recent_avg > older_avg * 1.1:
                vix_trend = "rising"
            elif recent_avg < older_avg * 0.9:
                vix_trend = "falling"
        
        confidence = min(1.0, abs(vix_level - 20) / 20)
        
        return cls(
            vix_level=vix_level,
            vix_trend=vix_trend,
            regime_type=regime_type,
            regime_confidence=confidence
        )
    
    @property
    def is_favorable_for_calendar_spreads(self) -> bool:
        """Check if regime is favorable for calendar spreads"""
        return (self.regime_type in ["normal", "elevated"] and 
                self.vix_trend in ["stable", "falling"])

@dataclass
class EarningsEvent:
    """Earnings event data"""
    symbol: str
    earnings_date: date
    confirmed: bool = False
    time_of_day: str = ""
    estimate_eps: Optional[float] = None
    previous_eps: Optional[float] = None
    
    @property
    def days_until_earnings(self) -> int:
        """Days until earnings"""
        return (self.earnings_date - date.today()).days
    
    @property
    def is_earnings_week(self) -> bool:
        """Check if earnings are within a week"""
        return 0 <= self.days_until_earnings <= 7
    
    @property
    def is_post_earnings(self) -> bool:
        """Check if earnings have already occurred"""
        return self.days_until_earnings < 0

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        return (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        """Calculate profit factor"""
        gross_profit = self.winning_trades * self.avg_win
        gross_loss = abs(self.losing_trades * self.avg_loss)
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def avg_pnl_per_trade(self) -> float:
        """Calculate average P&L per trade"""
        return self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio"""
        return abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 0.0

def calculate_moneyness(strike: float, underlying_price: float, option_type: OptionType) -> Tuple[str, float]:
    """Calculate option moneyness"""
    ratio = strike / underlying_price
    if option_type == OptionType.CALL:
        if ratio < 0.95:
            return "Deep ITM", ratio
        elif ratio < 0.98:
            return "ITM", ratio
        elif ratio <= 1.02:
            return "ATM", ratio
        elif ratio <= 1.05:
            return "OTM", ratio
        else:
            return "Deep OTM", ratio
    else:
        if ratio > 1.05:
            return "Deep ITM", ratio
        elif ratio > 1.02:
            return "ITM", ratio
        elif ratio >= 0.98:
            return "ATM", ratio
        elif ratio >= 0.95:
            return "OTM", ratio
        else:
            return "Deep OTM", ratio

def filter_liquid_options(options: List[OptionContract], min_volume: int = 100, min_open_interest: int = 50, max_spread_pct: float = 10.0) -> List[OptionContract]:
    """Filter options by liquidity criteria"""
    return [opt for opt in options if (
        opt.volume >= min_volume and
        opt.open_interest >= min_open_interest and
        opt.spread_percentage <= max_spread_pct
    )]

def find_calendar_spread_candidates(chains: List[OptionChain], min_time_spread: int = 7, max_time_spread: int = 45) -> List[CalendarSpread]:
    """Find calendar spread candidates from option chains"""
    candidates = []
    if len(chains) < 2:
        return candidates
    
    sorted_chains = sorted(chains, key=lambda x: x.expiration)
    for i in range(len(sorted_chains) - 1):
        short_chain = sorted_chains[i]
        for j in range(i + 1, len(sorted_chains)):
            long_chain = sorted_chains[j]
            time_spread = (long_chain.expiration - short_chain.expiration).days
            
            if min_time_spread <= time_spread <= max_time_spread:
                short_strikes = set(p.strike for p in short_chain.puts)
                long_strikes = set(p.strike for p in long_chain.puts)
                common_strikes = short_strikes.intersection(long_strikes)
                
                for strike in common_strikes:
                    short_put = short_chain.get_put_by_strike(strike)
                    long_put = long_chain.get_put_by_strike(strike)
                    
                    if (short_put and long_put and 
                        short_put.is_liquid() and long_put.is_liquid()):
                        
                        spread = CalendarSpread(
                            underlying_symbol=short_chain.underlying_symbol,
                            strike=strike,
                            short_option=short_put,
                            long_option=long_put
                        )
                        candidates.append(spread)
    
    return candidates