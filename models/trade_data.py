from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import date, datetime, timedelta
from enum import Enum
import json

class TradeType(Enum):
   """Trade type enumeration"""
   CALENDAR_SPREAD = "calendar_spread"
   IRON_CONDOR = "iron_condor"
   STRADDLE = "straddle"
   STRANGLE = "strangle"
   BUTTERFLY = "butterfly"
   COVERED_CALL = "covered_call"
   CASH_SECURED_PUT = "cash_secured_put"
   LONG_CALL = "long_call"
   LONG_PUT = "long_put"
   SHORT_CALL = "short_call"
   SHORT_PUT = "short_put"

class TradeStatus(Enum):
   """Trade status enumeration"""
   PLANNED = "planned"
   PENDING = "pending"
   OPEN = "open"
   PARTIAL = "partial"
   CLOSED = "closed"
   EXPIRED = "expired"
   ASSIGNED = "assigned"
   CANCELLED = "cancelled"

class TradeDirection(Enum):
   """Trade direction enumeration"""
   BULLISH = "bullish"
   BEARISH = "bearish"
   NEUTRAL = "neutral"

class ExitReason(Enum):
   """Exit reason enumeration"""
   PROFIT_TARGET = "profit_target"
   STOP_LOSS = "stop_loss"
   TIME_DECAY = "time_decay"
   EARNINGS_CRUSH = "earnings_crush"
   MANUAL_CLOSE = "manual_close"
   EXPIRATION = "expiration"
   ASSIGNMENT = "assignment"
   ROLL_FORWARD = "roll_forward"

@dataclass
class TradeLeg:
   """Individual trade leg (buy/sell of specific option)"""
   symbol: str
   strike: float
   expiration: date
   option_type: str  # "call" or "put"
   action: str  # "buy" or "sell"
   quantity: int
   entry_price: float = 0.0
   exit_price: float = 0.0
   commission: float = 0.0
   
   @property
   def is_long(self) -> bool:
       """Check if this is a long position"""
       return self.action.lower() == "buy"
   
   @property
   def is_short(self) -> bool:
       """Check if this is a short position"""
       return self.action.lower() == "sell"
   
   @property
   def premium_received(self) -> float:
       """Calculate premium received (negative for long positions)"""
       multiplier = -1 if self.is_long else 1
       return self.entry_price * self.quantity * 100 * multiplier
   
   @property
   def current_value(self) -> float:
       """Current value of the position"""
       multiplier = 1 if self.is_long else -1
       return self.exit_price * self.quantity * 100 * multiplier if self.exit_price > 0 else 0
   
   @property
   def unrealized_pnl(self) -> float:
       """Unrealized P&L"""
       if self.exit_price > 0:
           return self.current_value - self.premium_received - self.commission
       return -self.premium_received - self.commission  # Only entry cost
   
   def to_dict(self) -> Dict[str, Any]:
       """Convert to dictionary for serialization"""
       return {
           "symbol": self.symbol,
           "strike": self.strike,
           "expiration": self.expiration.isoformat(),
           "option_type": self.option_type,
           "action": self.action,
           "quantity": self.quantity,
           "entry_price": self.entry_price,
           "exit_price": self.exit_price,
           "commission": self.commission
       }
   
   @classmethod
   def from_dict(cls, data: Dict[str, Any]) -> 'TradeLeg':
       """Create from dictionary"""
       return cls(
           symbol=data["symbol"],
           strike=data["strike"],
           expiration=date.fromisoformat(data["expiration"]),
           option_type=data["option_type"],
           action=data["action"],
           quantity=data["quantity"],
           entry_price=data.get("entry_price", 0.0),
           exit_price=data.get("exit_price", 0.0),
           commission=data.get("commission", 0.0)
       )

@dataclass
class Trade:
   """Complete trade record"""
   id: str
   symbol: str
   trade_type: TradeType
   direction: TradeDirection
   status: TradeStatus
   legs: List[TradeLeg] = field(default_factory=list)
   
   # Entry information
   entry_date: Optional[date] = None
   entry_underlying_price: float = 0.0
   entry_iv: float = 0.0
   entry_vix: float = 0.0
   
   # Exit information
   exit_date: Optional[date] = None
   exit_underlying_price: float = 0.0
   exit_iv: float = 0.0
   exit_reason: Optional[ExitReason] = None
   
   # Analysis metrics
   confidence_score: float = 0.0
   iv_rank: float = 0.0
   days_to_earnings: int = 0
   sector: str = ""
   
   # Risk management
   max_loss: float = 0.0
   profit_target: float = 0.0
   stop_loss_percentage: float = 0.0
   
   # Performance tracking
   realized_pnl: float = 0.0
   max_profit: float = 0.0
   max_drawdown: float = 0.0
   
   # Metadata
   notes: str = ""
   tags: List[str] = field(default_factory=list)
   created_at: datetime = field(default_factory=datetime.now)
   updated_at: datetime = field(default_factory=datetime.now)
   
   @property
   def is_open(self) -> bool:
       """Check if trade is currently open"""
       return self.status in [TradeStatus.OPEN, TradeStatus.PARTIAL]
   
   @property
   def is_closed(self) -> bool:
       """Check if trade is closed"""
       return self.status in [TradeStatus.CLOSED, TradeStatus.EXPIRED, TradeStatus.ASSIGNED]
   
   @property
   def net_premium(self) -> float:
       """Calculate net premium (credit positive, debit negative)"""
       return sum(leg.premium_received for leg in self.legs)
   
   @property
   def total_commission(self) -> float:
       """Calculate total commission paid"""
       return sum(leg.commission for leg in self.legs)
   
   @property
   def days_in_trade(self) -> int:
       """Calculate days in trade"""
       if self.entry_date is None:
           return 0
       end_date = self.exit_date if self.exit_date else date.today()
       return (end_date - self.entry_date).days
   
   @property
   def time_decay_benefit(self) -> float:
       """Calculate time decay benefit (positive if we benefit from theta)"""
       theta_exposure = 0
       for leg in self.legs:
           # Short positions benefit from time decay
           multiplier = 1 if leg.is_short else -1
           theta_exposure += multiplier * leg.quantity
       return theta_exposure
   
   @property
   def breakeven_points(self) -> List[float]:
       """Calculate breakeven points (simplified)"""
       # This would need more sophisticated calculation based on trade type
       if self.trade_type == TradeType.CALENDAR_SPREAD:
           # For calendar spreads, breakeven is typically near the strike
           if self.legs:
               return [self.legs[0].strike]
       return []
   
   @property
   def profit_potential(self) -> str:
       """Describe profit potential"""
       if self.max_loss > 0 and self.profit_target > 0:
           ratio = self.profit_target / self.max_loss
           return f"Limited (Max Profit: ${self.profit_target:.2f}, Risk/Reward: {ratio:.2f})"
       return "Limited"
   
   @property
   def risk_profile(self) -> str:
       """Get risk profile description"""
       if abs(self.net_premium) < 100:
           return "Low Risk"
       elif abs(self.net_premium) < 500:
           return "Medium Risk"
       else:
           return "High Risk"
   
   def add_leg(self, leg: TradeLeg):
       """Add a trade leg"""
       self.legs.append(leg)
       self.updated_at = datetime.now()
   
   def remove_leg(self, index: int) -> bool:
       """Remove a trade leg"""
       if 0 <= index < len(self.legs):
           self.legs.pop(index)
           self.updated_at = datetime.now()
           return True
       return False
   
   def update_status(self, new_status: TradeStatus, exit_reason: Optional[ExitReason] = None):
       """Update trade status"""
       self.status = new_status
       self.updated_at = datetime.now()
       if exit_reason:
           self.exit_reason = exit_reason
       if new_status in [TradeStatus.CLOSED, TradeStatus.EXPIRED, TradeStatus.ASSIGNED]:
           self.exit_date = date.today()
   
   def calculate_current_pnl(self) -> float:
       """Calculate current unrealized P&L"""
       if self.is_closed:
           return self.realized_pnl
       return sum(leg.unrealized_pnl for leg in self.legs)
   
   def close_trade(self, exit_prices: Dict[int, float], exit_reason: ExitReason = ExitReason.MANUAL_CLOSE):
       """Close the trade with given exit prices"""
       for i, price in exit_prices.items():
           if 0 <= i < len(self.legs):
               self.legs[i].exit_price = price
       
       self.realized_pnl = self.calculate_current_pnl()
       self.exit_date = date.today()
       self.exit_reason = exit_reason
       self.status = TradeStatus.CLOSED
       self.updated_at = datetime.now()
   
   def add_tag(self, tag: str):
       """Add a tag to the trade"""
       if tag not in self.tags:
           self.tags.append(tag)
           self.updated_at = datetime.now()
   
   def remove_tag(self, tag: str):
       """Remove a tag from the trade"""
       if tag in self.tags:
           self.tags.remove(tag)
           self.updated_at = datetime.now()
   
   def to_dict(self) -> Dict[str, Any]:
       """Convert trade to dictionary for serialization"""
       return {
           "id": self.id,
           "symbol": self.symbol,
           "trade_type": self.trade_type.value,
           "direction": self.direction.value,
           "status": self.status.value,
           "legs": [leg.to_dict() for leg in self.legs],
           "entry_date": self.entry_date.isoformat() if self.entry_date else None,
           "entry_underlying_price": self.entry_underlying_price,
           "entry_iv": self.entry_iv,
           "entry_vix": self.entry_vix,
           "exit_date": self.exit_date.isoformat() if self.exit_date else None,
           "exit_underlying_price": self.exit_underlying_price,
           "exit_iv": self.exit_iv,
           "exit_reason": self.exit_reason.value if self.exit_reason else None,
           "confidence_score": self.confidence_score,
           "iv_rank": self.iv_rank,
           "days_to_earnings": self.days_to_earnings,
           "sector": self.sector,
           "max_loss": self.max_loss,
           "profit_target": self.profit_target,
           "stop_loss_percentage": self.stop_loss_percentage,
           "realized_pnl": self.realized_pnl,
           "max_profit": self.max_profit,
           "max_drawdown": self.max_drawdown,
           "notes": self.notes,
           "tags": self.tags,
           "created_at": self.created_at.isoformat(),
           "updated_at": self.updated_at.isoformat()
       }
   
   @classmethod
   def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
       """Create trade from dictionary"""
       trade = cls(
           id=data["id"],
           symbol=data["symbol"],
           trade_type=TradeType(data["trade_type"]),
           direction=TradeDirection(data["direction"]),
           status=TradeStatus(data["status"]),
           legs=[TradeLeg.from_dict(leg_data) for leg_data in data.get("legs", [])],
           entry_date=date.fromisoformat(data["entry_date"]) if data.get("entry_date") else None,
           entry_underlying_price=data.get("entry_underlying_price", 0.0),
           entry_iv=data.get("entry_iv", 0.0),
           entry_vix=data.get("entry_vix", 0.0),
           exit_date=date.fromisoformat(data["exit_date"]) if data.get("exit_date") else None,
           exit_underlying_price=data.get("exit_underlying_price", 0.0),
           exit_iv=data.get("exit_iv", 0.0),
           exit_reason=ExitReason(data["exit_reason"]) if data.get("exit_reason") else None,
           confidence_score=data.get("confidence_score", 0.0),
           iv_rank=data.get("iv_rank", 0.0),
           days_to_earnings=data.get("days_to_earnings", 0),
           sector=data.get("sector", ""),
           max_loss=data.get("max_loss", 0.0),
           profit_target=data.get("profit_target", 0.0),
           stop_loss_percentage=data.get("stop_loss_percentage", 0.0),
           realized_pnl=data.get("realized_pnl", 0.0),
           max_profit=data.get("max_profit", 0.0),
           max_drawdown=data.get("max_drawdown", 0.0),
           notes=data.get("notes", ""),
           tags=data.get("tags", []),
           created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
           updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
       )
       return trade

@dataclass
class TradeJournal:
   """Collection of trades with analysis capabilities"""
   trades: List[Trade] = field(default_factory=list)
   
   def add_trade(self, trade: Trade):
       """Add a trade to the journal"""
       self.trades.append(trade)
   
   def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
       """Get trade by ID"""
       for trade in self.trades:
           if trade.id == trade_id:
               return trade
       return None
   
   def get_open_trades(self) -> List[Trade]:
       """Get all open trades"""
       return [trade for trade in self.trades if trade.is_open]
   
   def get_closed_trades(self) -> List[Trade]:
       """Get all closed trades"""
       return [trade for trade in self.trades if trade.is_closed]
   
   def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
       """Get trades for specific symbol"""
       return [trade for trade in self.trades if trade.symbol.upper() == symbol.upper()]
   
   def get_trades_by_type(self, trade_type: TradeType) -> List[Trade]:
       """Get trades by type"""
       return [trade for trade in self.trades if trade.trade_type == trade_type]
   
   def get_trades_by_date_range(self, start_date: date, end_date: date) -> List[Trade]:
       """Get trades within date range"""
       return [trade for trade in self.trades 
               if trade.entry_date and start_date <= trade.entry_date <= end_date]
   
   def calculate_statistics(self) -> Dict[str, Any]:
       """Calculate comprehensive trading statistics"""
       closed_trades = self.get_closed_trades()
       
       if not closed_trades:
           return {"error": "No closed trades available"}
       
       winning_trades = [t for t in closed_trades if t.realized_pnl > 0]
       losing_trades = [t for t in closed_trades if t.realized_pnl <= 0]
       
       total_pnl = sum(t.realized_pnl for t in closed_trades)
       total_commission = sum(t.total_commission for t in closed_trades)
       net_pnl = total_pnl - total_commission
       
       stats = {
           "total_trades": len(closed_trades),
           "winning_trades": len(winning_trades),
           "losing_trades": len(losing_trades),
           "win_rate": len(winning_trades) / len(closed_trades) * 100,
           "total_pnl": total_pnl,
           "total_commission": total_commission,
           "net_pnl": net_pnl,
           "avg_pnl_per_trade": total_pnl / len(closed_trades),
           "avg_win": sum(t.realized_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
           "avg_loss": sum(t.realized_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
           "largest_win": max((t.realized_pnl for t in winning_trades), default=0),
           "largest_loss": min((t.realized_pnl for t in losing_trades), default=0),
           "avg_days_in_trade": sum(t.days_in_trade for t in closed_trades) / len(closed_trades),
       }
       
       # Profit factor
       gross_profit = sum(t.realized_pnl for t in winning_trades)
       gross_loss = abs(sum(t.realized_pnl for t in losing_trades))
       stats["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
       
       # Sharpe ratio (simplified)
       pnls = [t.realized_pnl for t in closed_trades]
       if len(pnls) > 1:
           import statistics
           mean_pnl = statistics.mean(pnls)
           std_pnl = statistics.stdev(pnls)
           stats["sharpe_ratio"] = mean_pnl / std_pnl if std_pnl > 0 else 0
       else:
           stats["sharpe_ratio"] = 0
       
       # Consecutive wins/losses
       stats["max_consecutive_wins"] = self._calculate_max_consecutive_wins()
       stats["max_consecutive_losses"] = self._calculate_max_consecutive_losses()
       
       return stats
   
   def _calculate_max_consecutive_wins(self) -> int:
       """Calculate maximum consecutive wins"""
       closed_trades = sorted(self.get_closed_trades(), key=lambda t: t.exit_date or date.min)
       max_consecutive = 0
       current_consecutive = 0
       
       for trade in closed_trades:
           if trade.realized_pnl > 0:
               current_consecutive += 1
               max_consecutive = max(max_consecutive, current_consecutive)
           else:
               current_consecutive = 0
       
       return max_consecutive
   
   def _calculate_max_consecutive_losses(self) -> int:
       """Calculate maximum consecutive losses"""
       closed_trades = sorted(self.get_closed_trades(), key=lambda t: t.exit_date or date.min)
       max_consecutive = 0
       current_consecutive = 0
       
       for trade in closed_trades:
           if trade.realized_pnl <= 0:
               current_consecutive += 1
               max_consecutive = max(max_consecutive, current_consecutive)
           else:
               current_consecutive = 0
       
       return max_consecutive
   
   def get_performance_by_symbol(self) -> Dict[str, Dict[str, Any]]:
       """Get performance statistics by symbol"""
       symbols = set(trade.symbol for trade in self.trades)
       performance = {}
       
       for symbol in symbols:
           symbol_trades = self.get_trades_by_symbol(symbol)
           closed_symbol_trades = [t for t in symbol_trades if t.is_closed]
           
           if closed_symbol_trades:
               total_pnl = sum(t.realized_pnl for t in closed_symbol_trades)
               winning_trades = [t for t in closed_symbol_trades if t.realized_pnl > 0]
               
               performance[symbol] = {
                   "total_trades": len(closed_symbol_trades),
                   "winning_trades": len(winning_trades),
                   "win_rate": len(winning_trades) / len(closed_symbol_trades) * 100,
                   "total_pnl": total_pnl,
                   "avg_pnl": total_pnl / len(closed_symbol_trades)
               }
       
       return performance
   
   def export_to_csv(self, filename: str):
       """Export trades to CSV file"""
       import csv
       
       with open(filename, 'w', newline='') as csvfile:
           fieldnames = [
               'id', 'symbol', 'trade_type', 'direction', 'status',
               'entry_date', 'exit_date', 'days_in_trade',
               'entry_price', 'exit_price', 'realized_pnl',
               'confidence_score', 'sector', 'notes'
           ]
           
           writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
           writer.writeheader()
           
           for trade in self.trades:
               writer.writerow({
                   'id': trade.id,
                   'symbol': trade.symbol,
                   'trade_type': trade.trade_type.value,
                   'direction': trade.direction.value,
                   'status': trade.status.value,
                   'entry_date': trade.entry_date.isoformat() if trade.entry_date else '',
                   'exit_date': trade.exit_date.isoformat() if trade.exit_date else '',
                   'days_in_trade': trade.days_in_trade,
                   'entry_price': trade.entry_underlying_price,
                   'exit_price': trade.exit_underlying_price,
                   'realized_pnl': trade.realized_pnl,
                   'confidence_score': trade.confidence_score,
                   'sector': trade.sector,
                   'notes': trade.notes
               })

def create_calendar_spread_trade(symbol: str, strike: float, short_expiration: date, 
                              long_expiration: date, entry_price: float, quantity: int = 1) -> Trade:
   """Helper function to create a calendar spread trade"""
   import uuid
   
   trade_id = str(uuid.uuid4())
   
   # Create legs
   short_leg = TradeLeg(
       symbol=symbol,
       strike=strike,
       expiration=short_expiration,
       option_type="put",
       action="sell",
       quantity=quantity,
       entry_price=entry_price
   )
   
   long_leg = TradeLeg(
       symbol=symbol,
       strike=strike,
       expiration=long_expiration,
       option_type="put",
       action="buy",
       quantity=quantity,
       entry_price=entry_price * 1.5  # Approximate long leg price
   )
   
   trade = Trade(
       id=trade_id,
       symbol=symbol,
       trade_type=TradeType.CALENDAR_SPREAD,
       direction=TradeDirection.NEUTRAL,
       status=TradeStatus.OPEN,
       entry_date=date.today(),
       legs=[short_leg, long_leg]
   )
   
   return trade