"""
IV Crush Calendar Spread — Live Trading System

Strategy: earnings IV crush calendar spread (long back / short front)
Signal:   NBR (near_back_ratio = pre_front_iv / pre_back_iv) ≥ threshold
Entry:    T-2 (2 business days before pre_capture_date)
Exit:     T+1 (1 business day after post_capture_date)

Modules:
  database.py           — SQLite schema, connection, CRUD helpers
  signal_engine.py      — scan upcoming earnings, compute NBR, generate signals
  liquidity_filter.py   — OI and spread gate
  risk_manager.py       — position limits, concentration, loss controls
  execution_simulator.py— fill model (mid ± slippage), paper or live hook
  trade_manager.py      — open/close trades, MTM, lifecycle
  performance_monitor.py— Sharpe, drawdown, win rate, vs-backtest tracking
  reporting.py          — weekly/monthly summaries
  daily_runner.py       — orchestrator: runs the full daily workflow
"""

VERSION = "system_v1"
"""
DB_PATH          = ~/.options_calculator_pro/trading_system.db
PARQUET_BASE     = /Volumes/T9/market_data/research/options_features_eod
INSTITUTIONAL_DB = ~/.options_calculator_pro/institutional_ml.db
"""
