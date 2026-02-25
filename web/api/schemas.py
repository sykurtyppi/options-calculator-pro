from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class EdgeAnalyzeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol")


class EdgeAnalyzeResponse(BaseModel):
    generated_at: datetime
    symbol: str
    recommendation: str
    confidence_pct: float
    setup_score: float
    metrics: Dict[str, Any]
    rationale: list[str]


class OOSReportRequest(BaseModel):
    lookback_days: int = 730
    max_backtest_symbols: int = 20
    backtest_start_date: Optional[str] = "2023-01-01"
    backtest_end_date: Optional[str] = "2025-12-31"
    min_signal_score: float = 0.50
    min_crush_confidence: float = 0.30
    min_crush_magnitude: float = 0.06
    min_crush_edge: float = 0.02
    target_entry_dte: int = 6
    entry_dte_band: int = 6
    min_daily_share_volume: int = 1_000_000
    max_abs_momentum_5d: float = 0.11

    oos_train_days: int = 252
    oos_test_days: int = 63
    oos_step_days: int = 63
    oos_top_n_train: int = 1

    oos_min_splits: int = 8
    oos_min_total_test_trades: int = 80
    oos_min_trades_per_split: float = 5.0


class OOSReportResponse(BaseModel):
    generated_at: datetime
    summary: Dict[str, Any]
    output_files: Dict[str, Optional[str]]
