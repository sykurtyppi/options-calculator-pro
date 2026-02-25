"""
Institutional ML Training Database - Calendar Spread Specialization
================================================================

Enhanced database system for institutional-grade options trading:
- Calendar spread specific schemas
- 50-100 ticker universe management
- 2-year historical backtesting data
- Feature pipeline for ML training
- Professional backtesting framework

Part of Professional Options Calculator v10.0 - Institutional Edition
"""

import sqlite3
import logging
import os
import json
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
import asyncio

from utils.logger import setup_logger as get_logger
from services.execution_cost_model import ExecutionCostModel

logger = get_logger(__name__)

# Institutional ticker universe (top 50 optionable stocks)
INSTITUTIONAL_UNIVERSE = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'ORCL', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'AMD',

    # Financial services
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA',

    # Healthcare & biotech
    'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'MRK', 'CVS', 'LLY', 'GILD',

    # Consumer & retail
    'HD', 'PG', 'KO', 'PEP', 'WMT', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',

    # Industrial & energy
    'CAT', 'BA', 'GE', 'MMM', 'HON', 'XOM', 'CVX', 'SLB', 'COP', 'EOG'
]

@dataclass
class CalendarSpreadRecord:
    """Calendar spread trade record for institutional backtesting"""
    trade_id: str
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]

    # Front month leg
    front_strike: float
    front_expiry: datetime
    front_entry_price: float
    front_exit_price: Optional[float]
    front_days_to_expiry: int

    # Back month leg
    back_strike: float
    back_expiry: datetime
    back_entry_price: float
    back_exit_price: Optional[float]
    back_days_to_expiry: int

    # Spread characteristics
    spread_width: float  # Time between expirations in days
    net_debit: float     # Total cost of spread
    max_profit_potential: float
    breakeven_price: float

    # Market context at entry
    underlying_price: float
    iv_rank: float
    iv30_rv30_ratio: float
    vix_level: float

    # Greeks at entry
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float

    # Trade outcome
    exit_reason: str    # 'profit_target', 'stop_loss', 'expiration', 'manual'
    days_held: Optional[int]
    profit_loss: Optional[float]
    return_pct: Optional[float]
    max_drawdown: Optional[float]

    # Performance metrics
    sharpe_ratio: Optional[float]
    win_rate: Optional[float]

@dataclass
class TickerMetadata:
    """Ticker universe metadata for systematic backtesting"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: str  # 'large', 'mega'
    avg_daily_volume: int
    beta: float
    options_liquidity_rank: int  # 1-10 scale
    earnings_frequency: str      # 'quarterly', 'irregular'
    backtest_start_date: datetime
    backtest_end_date: datetime
    data_quality_score: float   # 0.0-1.0

@dataclass
class BacktestSession:
    """Backtesting session metadata"""
    session_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    universe: List[str]
    parameters: Dict[str, Any]
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    created_at: datetime


@dataclass
class BacktestTrade:
    """Per-trade record for deterministic walk-forward backtesting."""
    session_id: str
    symbol: str
    trade_date: datetime
    event_date: datetime
    days_to_earnings: int
    contracts: int
    hold_days: int
    setup_score: float
    debit_per_contract: float
    transaction_cost_per_contract: float
    gross_return_pct: float
    net_return_pct: float
    pnl_per_contract: float
    underlying_return: float
    expected_move: float
    move_ratio: float
    predicted_front_iv_crush_pct: float
    crush_confidence: float
    crush_edge_score: float
    crush_profile_sample_size: float
    execution_profile: str

class InstitutionalMLDatabase:
    """
    Institutional-grade ML training database for calendar spreads
    Designed for systematic backtesting and professional trading systems
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize institutional ML database"""
        self.logger = logger

        # Database path
        if db_path is None:
            config_dir = os.path.expanduser("~/.options_calculator_pro")
            os.makedirs(config_dir, exist_ok=True)
            db_path = os.path.join(config_dir, "institutional_ml.db")

        self.db_path = db_path
        self.logger.info(f"🏛️ Institutional ML Database initialized: {db_path}")

        # Initialize database with comprehensive schema
        self._init_institutional_schema()
        self._populate_ticker_universe()

    def _init_institutional_schema(self):
        """Create institutional-grade database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys = ON")

                # Ticker universe table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ticker_universe (
                        symbol TEXT PRIMARY KEY,
                        company_name TEXT NOT NULL,
                        sector TEXT NOT NULL,
                        industry TEXT NOT NULL,
                        market_cap TEXT NOT NULL,
                        avg_daily_volume INTEGER NOT NULL,
                        beta REAL NOT NULL,
                        options_liquidity_rank INTEGER NOT NULL,
                        earnings_frequency TEXT NOT NULL,
                        backtest_start_date TEXT NOT NULL,
                        backtest_end_date TEXT NOT NULL,
                        data_quality_score REAL NOT NULL DEFAULT 1.0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Calendar spread trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS calendar_spreads (
                        trade_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        entry_date TEXT NOT NULL,
                        exit_date TEXT,

                        front_strike REAL NOT NULL,
                        front_expiry TEXT NOT NULL,
                        front_entry_price REAL NOT NULL,
                        front_exit_price REAL,
                        front_days_to_expiry INTEGER NOT NULL,

                        back_strike REAL NOT NULL,
                        back_expiry TEXT NOT NULL,
                        back_entry_price REAL NOT NULL,
                        back_exit_price REAL,
                        back_days_to_expiry INTEGER NOT NULL,

                        spread_width REAL NOT NULL,
                        net_debit REAL NOT NULL,
                        max_profit_potential REAL NOT NULL,
                        breakeven_price REAL NOT NULL,

                        underlying_price REAL NOT NULL,
                        iv_rank REAL NOT NULL,
                        iv30_rv30_ratio REAL NOT NULL,
                        vix_level REAL NOT NULL,

                        net_delta REAL NOT NULL,
                        net_gamma REAL NOT NULL,
                        net_theta REAL NOT NULL,
                        net_vega REAL NOT NULL,

                        exit_reason TEXT,
                        days_held INTEGER,
                        profit_loss REAL,
                        return_pct REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        win_rate REAL,

                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (symbol) REFERENCES ticker_universe (symbol)
                    )
                """)

                # Daily price data with technical indicators
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open_price REAL NOT NULL,
                        high_price REAL NOT NULL,
                        low_price REAL NOT NULL,
                        close_price REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        adj_close REAL NOT NULL,

                        -- Technical indicators
                        rsi_14 REAL,
                        macd_signal REAL,
                        macd_histogram REAL,
                        bb_upper REAL,
                        bb_lower REAL,
                        bb_position REAL,

                        -- Volatility metrics
                        realized_vol_30d REAL,
                        realized_vol_60d REAL,
                        iv_rank REAL,
                        iv_percentile REAL,

                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date),
                        FOREIGN KEY (symbol) REFERENCES ticker_universe (symbol)
                    )
                """)

                # Options chain historical data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS options_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        expiry_date TEXT NOT NULL,
                        strike_price REAL NOT NULL,
                        option_type TEXT NOT NULL, -- 'call' or 'put'

                        bid_price REAL,
                        ask_price REAL,
                        last_price REAL,
                        volume INTEGER,
                        open_interest INTEGER,
                        implied_volatility REAL,

                        delta_value REAL,
                        gamma_value REAL,
                        theta_value REAL,
                        vega_value REAL,
                        rho_value REAL,

                        underlying_price REAL NOT NULL,
                        time_to_expiry REAL NOT NULL,

                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date, expiry_date, strike_price, option_type),
                        FOREIGN KEY (symbol) REFERENCES ticker_universe (symbol)
                    )
                """)

                # ML training features (engineered)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ml_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,

                        -- Market features
                        underlying_price REAL NOT NULL,
                        price_momentum_5d REAL,
                        price_momentum_20d REAL,
                        volume_ratio_10d REAL,

                        -- Volatility features
                        iv_rank REAL NOT NULL,
                        iv_percentile REAL NOT NULL,
                        iv30_rv30_ratio REAL NOT NULL,
                        vol_term_structure_slope REAL,

                        -- Technical features
                        rsi_14 REAL,
                        bb_position REAL,
                        macd_momentum REAL,

                        -- Options flow features
                        put_call_ratio REAL,
                        options_volume_ratio REAL,

                        -- Macro features
                        vix_level REAL NOT NULL,
                        vix_contango REAL,
                        yield_curve_10y2y REAL,

                        -- Sector rotation features
                        sector_momentum REAL,
                        beta_adjusted_return REAL,

                        -- Target variables (forward-looking)
                        forward_return_1d REAL,
                        forward_return_5d REAL,
                        forward_return_21d REAL,
                        forward_volatility_21d REAL,

                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date),
                        FOREIGN KEY (symbol) REFERENCES ticker_universe (symbol)
                    )
                """)

                # Backtest sessions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_sessions (
                        session_id TEXT PRIMARY KEY,
                        strategy_name TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        universe TEXT NOT NULL, -- JSON array
                        parameters TEXT NOT NULL, -- JSON object
                        total_trades INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        total_pnl REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        calmar_ratio REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Earnings event cache (source: yfinance or proxy fallback)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS earnings_events (
                        symbol TEXT NOT NULL,
                        event_date TEXT NOT NULL,
                        source TEXT NOT NULL,
                        release_timing TEXT NOT NULL DEFAULT 'UNKNOWN',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, event_date)
                    )
                """)
                cursor.execute("PRAGMA table_info(earnings_events)")
                earnings_columns = {row[1] for row in cursor.fetchall()}
                if "release_timing" not in earnings_columns:
                    cursor.execute(
                        "ALTER TABLE earnings_events ADD COLUMN release_timing TEXT NOT NULL DEFAULT 'UNKNOWN'"
                    )

                # Option snapshots captured around earnings events (for IV-crush calibration)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS earnings_option_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        event_date TEXT NOT NULL,
                        capture_date TEXT NOT NULL,
                        relative_day INTEGER NOT NULL,
                        release_timing TEXT NOT NULL,
                        snapshot_phase TEXT NOT NULL,
                        short_expiry TEXT,
                        long_expiry TEXT,
                        atm_strike REAL,
                        front_iv REAL NOT NULL,
                        back_iv REAL NOT NULL,
                        term_ratio REAL NOT NULL,
                        underlying_price REAL NOT NULL,
                        source TEXT NOT NULL DEFAULT 'yfinance_live',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, event_date, capture_date, short_expiry, long_expiry)
                    )
                """)

                # Calibrated post-earnings IV-decay labels from real snapshots
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS earnings_iv_decay_labels (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        event_date TEXT NOT NULL,
                        release_timing TEXT NOT NULL,
                        pre_capture_date TEXT NOT NULL,
                        post_capture_date TEXT NOT NULL,
                        pre_front_iv REAL NOT NULL,
                        post_front_iv REAL NOT NULL,
                        pre_back_iv REAL NOT NULL,
                        post_back_iv REAL NOT NULL,
                        front_iv_crush_pct REAL NOT NULL,
                        back_iv_crush_pct REAL NOT NULL,
                        term_ratio_change REAL NOT NULL,
                        underlying_move_pct REAL NOT NULL,
                        quality_score REAL NOT NULL,
                        source TEXT NOT NULL DEFAULT 'snapshot_pair',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, event_date)
                    )
                """)

                # Trade-level records for walk-forward diagnostics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        trade_date TEXT NOT NULL,
                        event_date TEXT NOT NULL,
                        days_to_earnings INTEGER NOT NULL,
                        contracts INTEGER NOT NULL,
                        hold_days INTEGER NOT NULL,
                        setup_score REAL NOT NULL,
                        debit_per_contract REAL NOT NULL,
                        transaction_cost_per_contract REAL NOT NULL,
                        gross_return_pct REAL NOT NULL,
                        net_return_pct REAL NOT NULL,
                        pnl_per_contract REAL NOT NULL,
                        underlying_return REAL NOT NULL,
                        expected_move REAL NOT NULL,
                        move_ratio REAL NOT NULL,
                        predicted_front_iv_crush_pct REAL NOT NULL DEFAULT -0.18,
                        crush_confidence REAL NOT NULL DEFAULT 0.0,
                        crush_edge_score REAL NOT NULL DEFAULT 0.0,
                        crush_profile_sample_size REAL NOT NULL DEFAULT 0.0,
                        execution_profile TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES backtest_sessions (session_id)
                    )
                """)
                cursor.execute("PRAGMA table_info(backtest_trades)")
                backtest_trade_columns = {row[1] for row in cursor.fetchall()}
                if "contracts" not in backtest_trade_columns:
                    cursor.execute(
                        "ALTER TABLE backtest_trades ADD COLUMN contracts INTEGER NOT NULL DEFAULT 1"
                    )
                if "event_date" not in backtest_trade_columns:
                    cursor.execute(
                        "ALTER TABLE backtest_trades ADD COLUMN event_date TEXT NOT NULL DEFAULT '1970-01-01'"
                    )
                if "days_to_earnings" not in backtest_trade_columns:
                    cursor.execute(
                        "ALTER TABLE backtest_trades ADD COLUMN days_to_earnings INTEGER NOT NULL DEFAULT 0"
                    )
                if "predicted_front_iv_crush_pct" not in backtest_trade_columns:
                    cursor.execute(
                        "ALTER TABLE backtest_trades ADD COLUMN predicted_front_iv_crush_pct REAL NOT NULL DEFAULT -0.18"
                    )
                if "crush_confidence" not in backtest_trade_columns:
                    cursor.execute(
                        "ALTER TABLE backtest_trades ADD COLUMN crush_confidence REAL NOT NULL DEFAULT 0.0"
                    )
                if "crush_edge_score" not in backtest_trade_columns:
                    cursor.execute(
                        "ALTER TABLE backtest_trades ADD COLUMN crush_edge_score REAL NOT NULL DEFAULT 0.0"
                    )
                if "crush_profile_sample_size" not in backtest_trade_columns:
                    cursor.execute(
                        "ALTER TABLE backtest_trades ADD COLUMN crush_profile_sample_size REAL NOT NULL DEFAULT 0.0"
                    )

                # Performance indexes for fast queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_calendar_spreads_symbol_date ON calendar_spreads (symbol, entry_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol_date ON daily_prices (symbol, date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_data_symbol_date ON options_data (symbol, date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_features_symbol_date ON ml_features (symbol, date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_trades_session_date ON backtest_trades (session_id, trade_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_earnings_events_symbol_date ON earnings_events (symbol, event_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_earnings_snapshots_symbol_event_date ON earnings_option_snapshots (symbol, event_date, capture_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_earnings_labels_symbol_date ON earnings_iv_decay_labels (symbol, event_date)")

                conn.commit()
                self.logger.info("🏗️ Institutional database schema created successfully")

        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            raise

    def _populate_ticker_universe(self):
        """Populate initial ticker universe with institutional stocks"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if already populated
                cursor.execute("SELECT COUNT(*) FROM ticker_universe")
                if cursor.fetchone()[0] > 0:
                    self.logger.info("📊 Ticker universe already populated")
                    return

                self.logger.info("🏢 Populating institutional ticker universe...")

                # Sample institutional data (in production, fetch from API)
                institutional_data = [
                    ('AAPL', 'Apple Inc', 'Technology', 'Consumer Electronics', 'mega', 50000000, 1.2, 10, 'quarterly'),
                    ('MSFT', 'Microsoft Corp', 'Technology', 'Software', 'mega', 30000000, 0.9, 10, 'quarterly'),
                    ('GOOGL', 'Alphabet Inc', 'Technology', 'Internet', 'mega', 25000000, 1.1, 9, 'quarterly'),
                    ('JPM', 'JPMorgan Chase', 'Financial', 'Banking', 'mega', 15000000, 1.5, 8, 'quarterly'),
                    ('TSLA', 'Tesla Inc', 'Consumer Discretionary', 'Auto Manufacturing', 'large', 80000000, 2.1, 9, 'quarterly'),
                ]

                backtest_start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                backtest_end = datetime.now().strftime('%Y-%m-%d')

                for symbol, company, sector, industry, cap, volume, beta, liquidity, frequency in institutional_data:
                    cursor.execute("""
                        INSERT OR IGNORE INTO ticker_universe
                        (symbol, company_name, sector, industry, market_cap, avg_daily_volume,
                         beta, options_liquidity_rank, earnings_frequency, backtest_start_date, backtest_end_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, company, sector, industry, cap, volume, beta, liquidity, frequency,
                          backtest_start, backtest_end))

                conn.commit()
                self.logger.info(f"✅ Populated universe with {len(institutional_data)} institutional tickers")

        except Exception as e:
            self.logger.error(f"❌ Failed to populate ticker universe: {e}")

    async def backfill_historical_data(self, symbols: List[str] = None, years_back: int = 2,
                                      batch_size: int = 5) -> bool:
        """
        Institutional-grade historical data backfill

        Args:
            symbols: List of symbols to backfill (default: full universe)
            years_back: Years of historical data to collect
            batch_size: Concurrent download batch size

        Returns:
            True if successful
        """
        if symbols is None:
            symbols = INSTITUTIONAL_UNIVERSE[:10]  # Start with top 10 for testing

        self.logger.info(f"🔄 Starting institutional backfill for {len(symbols)} symbols, {years_back} years")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)

        success_count = 0
        total_symbols = len(symbols)

        try:
            # Process in batches for performance
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]

                self.logger.info(f"📥 Processing batch {i//batch_size + 1}/{(total_symbols-1)//batch_size + 1}: {batch}")

                # Download batch data concurrently
                tasks = []
                for symbol in batch:
                    tasks.append(self._download_symbol_data(symbol, start_date, end_date))

                # Wait for batch completion
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for symbol, result in zip(batch, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"❌ Failed to download {symbol}: {result}")
                    elif result:
                        success_count += 1
                        self.logger.info(f"✅ {symbol} data backfilled successfully")
                    else:
                        self.logger.warning(f"⚠️ {symbol} data backfill returned no data")

                # Rate limiting - be respectful to Yahoo Finance
                await asyncio.sleep(1)

            completion_rate = success_count / total_symbols
            self.logger.info(f"🎯 Backfill completed: {success_count}/{total_symbols} symbols ({completion_rate:.1%})")

            return completion_rate >= 0.8  # 80% success threshold

        except Exception as e:
            self.logger.error(f"❌ Backfill process failed: {e}")
            return False

    async def _download_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """Download and store data for a single symbol"""
        try:
            # Download historical price data
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                back_adjust=True
            )

            if hist_data.empty:
                self.logger.warning(f"📉 No historical data found for {symbol}")
                return False

            enriched_data = self._add_technical_indicators(hist_data)

            # Store price data with technical indicators
            await self._store_price_data(symbol, enriched_data)

            # Calculate and store ML features
            await self._calculate_and_store_features(symbol, enriched_data)

            return True

        except Exception as e:
            self.logger.error(f"❌ Error downloading {symbol}: {e}")
            return False

    async def _store_price_data(self, symbol: str, hist_data: pd.DataFrame):
        """Store price data with technical indicators"""
        try:
            if "RSI_14" not in hist_data.columns or "RealizedVol_30" not in hist_data.columns:
                hist_data = self._add_technical_indicators(hist_data)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for date, row in hist_data.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO daily_prices
                        (symbol, date, open_price, high_price, low_price, close_price,
                         volume, adj_close, rsi_14, macd_signal, macd_histogram,
                         bb_upper, bb_lower, bb_position, realized_vol_30d, realized_vol_60d)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, date.strftime('%Y-%m-%d'),
                        row['Open'], row['High'], row['Low'], row['Close'],
                        int(row['Volume']), row['Close'],  # Adj close same as close after auto_adjust
                        row.get('RSI_14', 50.0), row.get('MACD_Signal', 0.0), row.get('MACD_Hist', 0.0),
                        row.get('BB_Upper', row['Close']), row.get('BB_Lower', row['Close']), row.get('BB_Position', 0.5),
                        row.get('RealizedVol_30', 0.2), row.get('RealizedVol_60', 0.2)
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Error storing price data for {symbol}: {e}")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        try:
            df = df.copy()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

            # Bollinger Bands
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['MA_20'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['MA_20'] - (df['BB_Std'] * 2)
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

            # Realized Volatility
            df['Returns'] = df['Close'].pct_change()
            df['RealizedVol_30'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
            df['RealizedVol_60'] = df['Returns'].rolling(window=60).std() * np.sqrt(252)

            return df

        except Exception as e:
            self.logger.error(f"❌ Error calculating technical indicators: {e}")
            return df

    async def _calculate_and_store_features(self, symbol: str, hist_data: pd.DataFrame):
        """Calculate and store ML training features"""
        try:
            if "RealizedVol_30" not in hist_data.columns or "Returns" not in hist_data.columns:
                hist_data = self._add_technical_indicators(hist_data)

            data = hist_data.copy()
            rv30 = data['RealizedVol_30'].replace(0, np.nan)
            rv60 = data['RealizedVol_60'].replace(0, np.nan)
            rv30 = rv30.fillna(rv60).fillna(0.20).clip(lower=0.05, upper=2.50)
            rv60 = rv60.fillna(rv30).fillna(0.22).clip(lower=0.05, upper=2.50)
            slope_ratio = ((rv30 - rv60) / rv60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            slope_ratio = slope_ratio.clip(-0.5, 1.0)

            iv_proxy = (rv30 * (1.10 + 0.35 * slope_ratio)).clip(lower=0.08, upper=2.50)
            iv_low = iv_proxy.rolling(window=252, min_periods=30).min()
            iv_high = iv_proxy.rolling(window=252, min_periods=30).max()
            iv_rank = ((iv_proxy - iv_low) / (iv_high - iv_low)).replace([np.inf, -np.inf], np.nan).fillna(0.5)
            iv_rank = iv_rank.clip(lower=0.0, upper=1.0)
            iv_percentile = iv_proxy.rolling(window=252, min_periods=30).apply(
                lambda values: float(np.mean(values <= values[-1]) * 100.0),
                raw=True
            ).fillna(50.0)
            iv30_rv30_ratio = (iv_proxy / rv30.clip(lower=0.01)).replace([np.inf, -np.inf], np.nan).fillna(1.05)
            iv30_rv30_ratio = iv30_rv30_ratio.clip(lower=0.70, upper=2.50)
            vix_proxy = (12.0 + 70.0 * rv30).clip(lower=10.0, upper=45.0)
            forward_vol_21d = (data['Returns'].rolling(window=21).std() * np.sqrt(252)).shift(-21)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for i, (date, row) in enumerate(data.iterrows()):
                    if i < 60:  # Need 60 days for some calculations
                        continue

                    # Price momentum
                    price_momentum_5d = row['Close'] / data['Close'].iloc[i-5] - 1 if i >= 5 else 0
                    price_momentum_20d = row['Close'] / data['Close'].iloc[i-20] - 1 if i >= 20 else 0

                    # Volume analysis
                    avg_volume_10d = data['Volume'].iloc[i-10:i].mean()
                    volume_ratio = row['Volume'] / avg_volume_10d if avg_volume_10d > 0 else 1

                    # Forward returns for training targets
                    forward_return_1d = data['Close'].iloc[i+1] / row['Close'] - 1 if i < len(data) - 1 else None
                    forward_return_5d = data['Close'].iloc[i+5] / row['Close'] - 1 if i < len(data) - 5 else None
                    forward_return_21d = data['Close'].iloc[i+21] / row['Close'] - 1 if i < len(data) - 21 else None

                    cursor.execute("""
                        INSERT OR REPLACE INTO ml_features
                        (symbol, date, underlying_price, price_momentum_5d, price_momentum_20d,
                         volume_ratio_10d, iv_rank, iv_percentile, iv30_rv30_ratio,
                         vol_term_structure_slope,
                         rsi_14, bb_position, vix_level, forward_return_1d, forward_return_5d,
                         forward_return_21d, forward_volatility_21d)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, date.strftime('%Y-%m-%d'), row['Close'],
                        price_momentum_5d, price_momentum_20d, volume_ratio,
                        float(iv_rank.iloc[i]), float(iv_percentile.iloc[i]), float(iv30_rv30_ratio.iloc[i]),
                        float(slope_ratio.iloc[i]),
                        row.get('RSI_14', 50.0), row.get('BB_Position', 0.5), float(vix_proxy.iloc[i]),
                        forward_return_1d, forward_return_5d, forward_return_21d,
                        None if pd.isna(forward_vol_21d.iloc[i]) else float(forward_vol_21d.iloc[i])
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Error calculating features for {symbol}: {e}")

    def run_calendar_spread_backtest(self, strategy_params: Dict[str, Any]) -> str:
        """
        Run deterministic walk-forward calendar spread backtesting.

        Args:
            strategy_params: Strategy configuration

        Returns:
            Session ID for tracking results
        """
        strategy_params = dict(strategy_params or {})
        session_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"🎯 Starting calendar spread backtest: {session_id}")

        try:
            now = datetime.now()
            lookback_days = max(120, int(strategy_params.get('lookback_days', 365)))
            start_date = now - timedelta(days=lookback_days)
            end_date = now

            raw_start = strategy_params.get('start_date')
            raw_end = strategy_params.get('end_date')
            if raw_start:
                start_date = datetime.strptime(str(raw_start), '%Y-%m-%d')
            if raw_end:
                end_date = datetime.strptime(str(raw_end), '%Y-%m-%d')
            if start_date >= end_date:
                start_date = end_date - timedelta(days=lookback_days)

            requested_universe = strategy_params.get('universe')
            max_symbols = max(1, int(strategy_params.get('max_symbols', 10)))
            if isinstance(requested_universe, str):
                universe = [s.strip().upper() for s in requested_universe.split(',') if s.strip()]
            elif isinstance(requested_universe, list):
                universe = [str(s).strip().upper() for s in requested_universe if str(s).strip()]
            else:
                universe = INSTITUTIONAL_UNIVERSE[:max_symbols]
            if not universe:
                universe = INSTITUTIONAL_UNIVERSE[:max_symbols]

            # Initialize backtest session
            session = BacktestSession(
                session_id=session_id,
                strategy_name="Calendar_Spread_WalkForward",
                start_date=start_date,
                end_date=end_date,
                universe=universe,
                parameters=strategy_params,
                total_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                created_at=now
            )

            # Store session metadata
            self._store_backtest_session(session)

            # Run deterministic walk-forward simulation
            trades = self._run_walk_forward_backtest(session, strategy_params)
            total_trades = len(trades)
            if total_trades == 0:
                self.logger.warning("⚠️ No valid trades generated for current backtest window")
                self._update_backtest_results(session)
                return session_id

            pnl_per_trade = np.array([trade.pnl_per_contract * trade.contracts for trade in trades], dtype=float)
            return_series = np.array([trade.net_return_pct for trade in trades], dtype=float)
            hold_days = max(1, int(strategy_params.get('hold_days', 7)))
            annualization_factor = np.sqrt(252 / hold_days)
            return_std = float(np.std(return_series))

            win_rate = float(np.mean(pnl_per_trade > 0))
            pnl = float(np.sum(pnl_per_trade))
            sharpe = float((np.mean(return_series) / return_std) * annualization_factor) if return_std > 0 else 0.0
            max_dd = self._calculate_max_drawdown(pnl_per_trade)

            # Update session results
            session.total_trades = total_trades
            session.win_rate = win_rate
            session.total_pnl = pnl
            session.sharpe_ratio = sharpe
            session.max_drawdown = max_dd
            session.calmar_ratio = float(pnl / abs(max_dd)) if max_dd < 0 else 0.0

            self._update_backtest_results(session)
            self._store_backtest_trades(session_id, trades)

            self.logger.info(f"✅ Backtest completed: {total_trades} trades, {win_rate:.1%} win rate, {pnl:.2f} PnL")
            return session_id

        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            return session_id

    def _run_walk_forward_backtest(self, session: BacktestSession,
                                 strategy_params: Dict[str, Any]) -> List[BacktestTrade]:
        """Generate deterministic trade outcomes from historical feature snapshots."""
        earnings_event_mode = bool(strategy_params.get('earnings_event_mode', True))
        entry_days_before_earnings = max(1, int(strategy_params.get('entry_days_before_earnings', 7)))
        exit_days_after_earnings = max(0, int(strategy_params.get('exit_days_after_earnings', 1)))
        require_true_earnings = bool(strategy_params.get('require_true_earnings', False))
        allow_proxy_earnings = bool(strategy_params.get('allow_proxy_earnings', True))
        hold_days = max(1, int(strategy_params.get('hold_days', 7)))
        min_signal_score = float(np.clip(float(strategy_params.get('min_signal_score', 0.58)), 0.0, 1.0))
        max_trades_per_day = max(1, int(strategy_params.get('max_trades_per_day', strategy_params.get('max_positions', 2))))
        position_size_contracts = max(1, int(strategy_params.get('position_size_contracts', 1)))
        iv_rv_min = float(strategy_params.get('iv_rv_min', 0.95))
        iv_rv_max = float(strategy_params.get('iv_rv_max', 2.30))
        use_crush_confidence_gate = bool(strategy_params.get('use_crush_confidence_gate', True))
        allow_global_crush_profile = bool(strategy_params.get('allow_global_crush_profile', True))
        min_crush_confidence = float(np.clip(float(strategy_params.get('min_crush_confidence', 0.45)), 0.0, 1.0))
        min_crush_magnitude = float(np.clip(float(strategy_params.get('min_crush_magnitude', 0.08)), 0.0, 1.0))
        min_crush_edge = float(np.clip(float(strategy_params.get('min_crush_edge', 0.02)), 0.0, 1.0))
        target_entry_dte = max(1, int(strategy_params.get('target_entry_dte', 6)))
        entry_dte_band = max(1, int(strategy_params.get('entry_dte_band', 6)))
        min_daily_share_volume = max(
            0.0,
            self._safe_numeric(strategy_params.get('min_daily_share_volume', 1_000_000), 1_000_000.0),
        )
        max_abs_momentum_5d = max(
            0.01,
            self._safe_numeric(strategy_params.get('max_abs_momentum_5d', 0.11), 0.11),
        )
        execution_profile = str(strategy_params.get('execution_profile', 'institutional')).strip().lower() or 'institutional'
        execution_cost_model = ExecutionCostModel(execution_profile)

        dataset = self._load_walk_forward_dataset(session.universe, session.start_date, session.end_date)
        if dataset.empty:
            self.logger.warning("⚠️ No feature data available for walk-forward simulation")
            return []
        crush_profiles = self._load_iv_crush_profiles(session.universe)

        trades: List[BacktestTrade] = []
        daily_candidates: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
        if earnings_event_mode:
            daily_candidates = self._build_earnings_event_candidates(
                dataset=dataset,
                universe=session.universe,
                start_date=session.start_date,
                end_date=session.end_date,
                hold_days_default=hold_days,
                entry_days_before_earnings=entry_days_before_earnings,
                exit_days_after_earnings=exit_days_after_earnings,
                require_true_earnings=require_true_earnings,
                allow_proxy_earnings=allow_proxy_earnings,
            )
            if not daily_candidates:
                self.logger.warning("⚠️ No earnings-window candidates found in selected timeframe")

        grouped_source = (
            daily_candidates.items()
            if daily_candidates
            else ((trade_date, list(day_slice.itertuples(index=False)))
                  for trade_date, day_slice in dataset.groupby('date', sort=True))
        )

        for trade_date, day_entries in grouped_source:
            scored_rows = []
            for entry in day_entries:
                if isinstance(entry, dict):
                    row = entry['row']
                    row_hold_days = int(entry['hold_days'])
                    event_date = entry['event_date']
                    days_to_earnings = int(entry['days_to_earnings'])
                else:
                    row = entry
                    row_hold_days = hold_days
                    event_date = pd.Timestamp(trade_date).to_pydatetime()
                    days_to_earnings = 0

                iv_rv_ratio = self._safe_numeric(getattr(row, 'iv30_rv30_ratio', np.nan), np.nan)
                if not np.isfinite(iv_rv_ratio):
                    continue
                if iv_rv_ratio < iv_rv_min or iv_rv_ratio > iv_rv_max:
                    continue
                momentum_5d = abs(self._safe_numeric(getattr(row, 'price_momentum_5d', 0.0), 0.0))
                if momentum_5d > max_abs_momentum_5d:
                    continue
                daily_share_volume = self._safe_numeric(getattr(row, 'volume', np.nan), np.nan)
                if np.isfinite(daily_share_volume) and daily_share_volume < min_daily_share_volume:
                    continue

                symbol = str(getattr(row, 'symbol', 'UNKNOWN'))
                crush_context = self._derive_crush_signal_context(
                    symbol=symbol,
                    iv_rv_ratio=iv_rv_ratio,
                    crush_profiles=crush_profiles,
                )
                if use_crush_confidence_gate:
                    if crush_context['profile_source'] == 'global' and not allow_global_crush_profile:
                        continue
                    if crush_context['confidence'] < min_crush_confidence:
                        continue
                    if crush_context['magnitude'] < min_crush_magnitude:
                        continue
                    if crush_context['edge_score'] < min_crush_edge:
                        continue

                setup_score = self._score_setup_quality(
                    row=row,
                    days_to_earnings=days_to_earnings,
                    target_entry_dte=target_entry_dte,
                    entry_dte_band=entry_dte_band,
                )
                if setup_score < min_signal_score:
                    continue
                rank_score = self._rank_candidate_for_alpha(
                    setup_score=setup_score,
                    crush_context=crush_context,
                    days_to_earnings=days_to_earnings,
                    target_entry_dte=target_entry_dte,
                    entry_dte_band=entry_dte_band,
                )
                scored_rows.append((rank_score, setup_score, row, row_hold_days, event_date, days_to_earnings, crush_context))

            if not scored_rows:
                continue

            scored_rows.sort(key=lambda item: item[0], reverse=True)
            for _, setup_score, row, row_hold_days, event_date, days_to_earnings, crush_context in scored_rows[:max_trades_per_day]:
                trade = self._simulate_walk_forward_trade(
                    session_id=session.session_id,
                    trade_date=pd.Timestamp(trade_date).to_pydatetime(),
                    row=row,
                    setup_score=setup_score,
                    hold_days=row_hold_days,
                    contracts=position_size_contracts,
                    execution_profile=execution_cost_model.profile_name,
                    execution_cost_model=execution_cost_model,
                    event_date=event_date,
                    days_to_earnings=days_to_earnings,
                    crush_context=crush_context,
                    crush_profiles=crush_profiles,
                )
                if trade is not None:
                    trades.append(trade)

        self.logger.info(
            f"📈 Walk-forward generated {len(trades)} trades over {dataset['date'].nunique()} trading days"
        )
        return trades

    def _load_walk_forward_dataset(self, universe: List[str], start_date: datetime,
                                 end_date: datetime) -> pd.DataFrame:
        """Load backtest features aligned with daily price/volatility context."""
        if not universe:
            return pd.DataFrame()

        placeholders = ",".join(["?"] * len(universe))
        query = f"""
            SELECT
                f.symbol,
                f.date,
                f.underlying_price,
                f.price_momentum_5d,
                f.price_momentum_20d,
                f.volume_ratio_10d,
                f.iv30_rv30_ratio,
                f.vol_term_structure_slope,
                f.rsi_14,
                f.bb_position,
                f.put_call_ratio,
                f.options_volume_ratio,
                f.vix_level,
                f.forward_return_1d,
                f.forward_return_5d,
                f.forward_return_21d,
                p.volume,
                p.realized_vol_30d
            FROM ml_features f
            LEFT JOIN daily_prices p ON f.symbol = p.symbol AND f.date = p.date
            WHERE f.symbol IN ({placeholders})
              AND f.date >= ?
              AND f.date <= ?
            ORDER BY f.date, f.symbol
        """
        params = list(universe) + [
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
        ]

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date'])
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna(subset=['underlying_price', 'iv30_rv30_ratio'])

    def _build_earnings_event_candidates(self, dataset: pd.DataFrame, universe: List[str],
                                       start_date: datetime, end_date: datetime,
                                       hold_days_default: int,
                                       entry_days_before_earnings: int,
                                       exit_days_after_earnings: int,
                                       require_true_earnings: bool,
                                       allow_proxy_earnings: bool) -> Dict[pd.Timestamp, List[Dict[str, Any]]]:
        """Build entry candidates anchored to earnings events."""
        candidates: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
        used_keys = set()

        for symbol in universe:
            symbol_df = dataset[dataset['symbol'] == symbol].sort_values('date').reset_index(drop=True)
            if symbol_df.empty:
                continue

            trading_dates = pd.to_datetime(symbol_df['date'])
            earnings_dates = self._get_symbol_earnings_dates(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                trading_dates=trading_dates,
                require_true_earnings=require_true_earnings,
                allow_proxy_earnings=allow_proxy_earnings,
            )
            if not earnings_dates:
                continue

            for event in earnings_dates:
                event_ts = pd.Timestamp(event.get('event_date'))
                release_timing = str(event.get('release_timing', 'UNKNOWN')).upper()
                entry_target = event_ts - pd.Timedelta(days=entry_days_before_earnings)
                entry_idx = int(trading_dates.searchsorted(entry_target, side='right')) - 1
                if entry_idx < 0 or entry_idx >= len(symbol_df):
                    continue

                entry_date = pd.Timestamp(trading_dates.iloc[entry_idx]).normalize()
                days_to_earnings = int((event_ts.normalize() - entry_date).days)
                if days_to_earnings <= 0:
                    continue
                if days_to_earnings > entry_days_before_earnings + 10:
                    # Feature coverage is too stale relative to this event.
                    continue

                # BMO releases happen before the session, so force entry before event day.
                if release_timing == 'BMO' and entry_date >= event_ts.normalize():
                    entry_idx = entry_idx - 1
                    if entry_idx < 0:
                        continue
                    entry_date = pd.Timestamp(trading_dates.iloc[entry_idx]).normalize()
                    days_to_earnings = int((event_ts.normalize() - entry_date).days)
                    if days_to_earnings <= 0:
                        continue

                unique_key = (symbol, entry_date.date(), event_ts.normalize().date())
                if unique_key in used_keys:
                    continue
                used_keys.add(unique_key)

                # AMC releases are effectively "post-event" on the next trading day.
                if release_timing == 'AMC':
                    base_exit_event = event_ts + pd.Timedelta(days=1)
                else:
                    base_exit_event = event_ts
                exit_target = base_exit_event + pd.Timedelta(days=exit_days_after_earnings)
                exit_idx = int(trading_dates.searchsorted(exit_target, side='left'))
                if exit_idx <= entry_idx:
                    row_hold_days = hold_days_default
                else:
                    row_hold_days = exit_idx - entry_idx
                row_hold_days = int(np.clip(row_hold_days, 1, 21))

                row = next(symbol_df.iloc[[entry_idx]].itertuples(index=False))
                candidates.setdefault(entry_date, []).append({
                    'row': row,
                    'hold_days': row_hold_days,
                    'event_date': event_ts.to_pydatetime(),
                    'days_to_earnings': days_to_earnings,
                    'release_timing': release_timing,
                })

        return candidates

    def _get_symbol_earnings_dates(self, symbol: str, start_date: datetime, end_date: datetime,
                                 trading_dates: pd.Series,
                                 require_true_earnings: bool,
                                 allow_proxy_earnings: bool) -> List[Dict[str, Any]]:
        """Get earnings dates from cache/API, with optional proxy fallback."""
        window_start = start_date - timedelta(days=60)
        window_end = end_date + timedelta(days=30)

        cached = self._get_cached_earnings_dates(symbol, window_start, window_end)
        true_dates = sorted(
            {(event_date, release_timing) for event_date, source, release_timing in cached if source == 'yfinance'},
            key=lambda item: item[0]
        )
        if true_dates:
            return [
                {'event_date': event_date, 'source': 'yfinance', 'release_timing': release_timing}
                for event_date, release_timing in true_dates
            ]
        proxy_dates_cached = sorted(
            {(event_date, release_timing) for event_date, source, release_timing in cached if source == 'proxy'},
            key=lambda item: item[0]
        )
        today_floor = datetime.combine(datetime.now().date(), datetime.min.time())
        window_extends_into_future = bool(end_date >= today_floor)

        fetched_true_dates = self._fetch_and_cache_earnings_dates(symbol, window_start, window_end)
        if fetched_true_dates:
            return fetched_true_dates

        if require_true_earnings:
            self.logger.warning(f"⚠️ No true earnings dates available for {symbol}")
            return []
        if not allow_proxy_earnings:
            return []

        if proxy_dates_cached and not window_extends_into_future:
            return [
                {'event_date': event_date, 'source': 'proxy', 'release_timing': release_timing}
                for event_date, release_timing in proxy_dates_cached
            ]

        # Rebuild proxy schedules in forward-looking windows so phase improvements
        # and stale-cache corrections propagate automatically.
        proxy_dates = self._build_proxy_earnings_schedule(
            trading_dates,
            window_start,
            window_end,
            symbol=symbol,
        )
        if proxy_dates:
            self._cache_earnings_dates(symbol, proxy_dates, source='proxy', release_timing='UNKNOWN')
            return [
                {
                    'event_date': pd.Timestamp(item.get('event_date')).to_pydatetime(),
                    'source': 'proxy',
                    'release_timing': str(item.get('release_timing', 'UNKNOWN')).upper(),
                }
                for item in proxy_dates
            ]
        if proxy_dates_cached:
            return [
                {'event_date': event_date, 'source': 'proxy', 'release_timing': release_timing}
                for event_date, release_timing in proxy_dates_cached
            ]
        return []

    def _get_cached_earnings_dates(self, symbol: str, window_start: datetime,
                                 window_end: datetime) -> List[Tuple[datetime, str, str]]:
        """Read cached earnings dates for the given symbol and time window."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT event_date, source, release_timing
                    FROM earnings_events
                    WHERE symbol = ?
                      AND event_date >= ?
                      AND event_date <= ?
                    ORDER BY event_date
                """
                rows = conn.execute(
                    query,
                    (
                        symbol,
                        window_start.strftime('%Y-%m-%d'),
                        window_end.strftime('%Y-%m-%d'),
                    )
                ).fetchall()
        except Exception:
            return []

        parsed: List[Tuple[datetime, str, str]] = []
        for event_date_str, source, release_timing in rows:
            try:
                parsed.append((
                    datetime.strptime(event_date_str, '%Y-%m-%d'),
                    str(source),
                    str(release_timing or 'UNKNOWN').upper()
                ))
            except ValueError:
                continue
        return parsed

    def _cache_earnings_dates(self, symbol: str, event_dates: List[Dict[str, Any]],
                            source: str, release_timing: str = 'UNKNOWN'):
        """Store earnings dates in local cache table."""
        if not event_dates:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO earnings_events (symbol, event_date, source, release_timing)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            symbol,
                            pd.Timestamp(item.get('event_date')).strftime('%Y-%m-%d'),
                            source,
                            str(item.get('release_timing', release_timing or 'UNKNOWN')).upper(),
                        )
                        for item in event_dates
                    ]
                )
                conn.commit()
        except Exception:
            return

    def _fetch_and_cache_earnings_dates(self, symbol: str, window_start: datetime,
                                      window_end: datetime) -> List[Dict[str, Any]]:
        """Fetch earnings dates from yfinance and cache them."""
        event_rows: List[Dict[str, Any]] = []
        try:
            ticker = yf.Ticker(symbol)
            earnings_df = None
            get_dates = getattr(ticker, 'get_earnings_dates', None)
            if callable(get_dates):
                try:
                    earnings_df = get_dates(limit=60)
                except Exception:
                    earnings_df = None
            if earnings_df is None:
                earnings_df = getattr(ticker, 'earnings_dates', None)
            if earnings_df is None or earnings_df.empty:
                return []

            for idx, ts in enumerate(pd.to_datetime(earnings_df.index)):
                ts_obj = pd.Timestamp(ts)
                if ts_obj.tzinfo is not None:
                    ts_obj = ts_obj.tz_localize(None)
                dt = ts_obj.to_pydatetime()
                if window_start <= dt <= window_end:
                    release_timing = self._infer_release_timing_from_earnings_row(
                        dt,
                        earnings_df.iloc[idx] if idx < len(earnings_df) else None
                    )
                    event_rows.append({
                        'event_date': dt.replace(hour=0, minute=0, second=0, microsecond=0),
                        'release_timing': release_timing,
                    })

            dedup = {}
            for row in event_rows:
                key = row['event_date'].strftime('%Y-%m-%d')
                dedup[key] = row
            event_rows = [dedup[key] for key in sorted(dedup.keys())]
            if event_rows:
                self._cache_earnings_dates(symbol, event_rows, source='yfinance')
        except Exception as e:
            self.logger.debug(f"Unable to fetch earnings dates for {symbol}: {e}")
            return []

        return event_rows

    def _infer_release_timing_from_earnings_row(self, event_dt: datetime,
                                              row: Optional[pd.Series]) -> str:
        """Infer release timing bucket (AMC/BMO/UNKNOWN) from row/time metadata."""
        # Heuristic 1: explicit columns from provider
        if row is not None:
            for key, value in row.items():
                key_text = str(key).lower()
                if "time" in key_text or "hour" in key_text:
                    timing = self._normalize_release_timing_token(value)
                    if timing != 'UNKNOWN':
                        return timing

        # Heuristic 2: timestamp hour if provided
        hour = int(event_dt.hour)
        if hour >= 15:
            return 'AMC'
        if 0 <= hour <= 10:
            return 'BMO'
        return 'UNKNOWN'

    def _normalize_release_timing_token(self, value: Any) -> str:
        """Normalize free-form timing token into AMC/BMO/UNKNOWN."""
        text = str(value or '').strip().upper()
        if not text:
            return 'UNKNOWN'
        if 'AMC' in text or 'AFTER' in text:
            return 'AMC'
        if 'BMO' in text or 'BEFORE' in text:
            return 'BMO'
        if text in {'POST', 'PM'}:
            return 'AMC'
        if text in {'PRE', 'AM'}:
            return 'BMO'
        return 'UNKNOWN'

    def _build_proxy_earnings_schedule(self, trading_dates: pd.Series, window_start: datetime,
                                     window_end: datetime,
                                     symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback quarterly-like earnings schedule when true dates are unavailable."""
        if trading_dates.empty:
            return []

        sorted_dates = (
            pd.Series(pd.to_datetime(trading_dates))
            .dropna()
            .sort_values()
            .reset_index(drop=True)
        )
        if not sorted_dates.empty:
            sorted_dates = (
                pd.Series(pd.to_datetime(sorted_dates))
                .dt.normalize()
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )

        # Project business-day dates into the lookahead horizon so proxy events can be scheduled in the future.
        if not sorted_dates.empty:
            last_known_date = pd.Timestamp(sorted_dates.iloc[-1]).normalize()
            target_end_date = pd.Timestamp(window_end).normalize()
            if last_known_date < target_end_date:
                projected_future_dates = pd.bdate_range(
                    start=last_known_date + pd.offsets.BDay(1),
                    end=target_end_date,
                )
                if len(projected_future_dates) > 0:
                    sorted_dates = (
                        pd.concat([sorted_dates, pd.Series(projected_future_dates)], ignore_index=True)
                        .drop_duplicates()
                        .sort_values()
                        .reset_index(drop=True)
                    )
        if len(sorted_dates) < 90:
            return []

        proxy_dates: List[Dict[str, Any]] = []
        step = 63  # Approximate quarterly trading sessions
        # Deterministic per-symbol phase offset prevents all proxy events from clustering
        # on the same dates when true earnings are unavailable.
        symbol_text = str(symbol or "").upper()
        symbol_offset = (sum(ord(ch) for ch in symbol_text) % step) if symbol_text else 0
        start_idx = 32 + symbol_offset
        for idx in range(start_idx, len(sorted_dates), step):
            dt = pd.Timestamp(sorted_dates.iloc[idx]).to_pydatetime()
            if window_start <= dt <= window_end:
                proxy_dates.append({
                    'event_date': dt.replace(hour=0, minute=0, second=0, microsecond=0),
                    'release_timing': 'UNKNOWN',
                })
        return proxy_dates

    def capture_earnings_option_snapshots(self, symbols: Optional[List[str]] = None,
                                        lookback_days: int = 14,
                                        lookahead_days: int = 45,
                                        max_expiries: int = 2,
                                        require_true_earnings: bool = False,
                                        allow_proxy_earnings: bool = True) -> Dict[str, Any]:
        """
        Capture live option-IV snapshots around earnings events.
        This is intended for forward data collection used by IV-crush label calibration.
        """
        if symbols is None:
            symbols = INSTITUTIONAL_UNIVERSE[:10]
        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        lookback_days = max(1, int(lookback_days))
        lookahead_days = max(1, int(lookahead_days))
        max_expiries = max(1, int(max_expiries))

        today = datetime.now().date()
        window_start = datetime.combine(today - timedelta(days=lookback_days), datetime.min.time())
        window_end = datetime.combine(today + timedelta(days=lookahead_days), datetime.min.time())
        history_padding_days = max(365, lookback_days + lookahead_days + 240)
        history_start = (window_start - timedelta(days=history_padding_days)).strftime('%Y-%m-%d')
        history_end = (today + timedelta(days=1)).strftime('%Y-%m-%d')

        captured = 0
        attempts = 0
        errors = 0
        no_price_symbols = 0
        no_event_symbols = 0
        no_expiry_symbols = 0
        no_iv_events = 0
        chain_errors = 0
        eligible_events = 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        price_hist = ticker.history(
                            start=history_start,
                            end=history_end,
                            auto_adjust=True,
                            back_adjust=True,
                        )
                        if price_hist.empty:
                            # Fallback for providers that reject explicit start/end requests.
                            price_hist = ticker.history(period='18mo', auto_adjust=True, back_adjust=True)
                        if price_hist.empty:
                            no_price_symbols += 1
                            continue
                        idx = pd.to_datetime(price_hist.index)
                        if getattr(idx, 'tz', None) is not None:
                            idx = idx.tz_localize(None)
                        trading_dates = pd.Series(idx.normalize()).sort_values().drop_duplicates().reset_index(drop=True)

                        last_close = float(price_hist['Close'].dropna().iloc[-1]) if not price_hist['Close'].dropna().empty else np.nan
                        if not np.isfinite(last_close) or last_close <= 0:
                            continue

                        events = self._get_symbol_earnings_dates(
                            symbol=symbol,
                            start_date=window_start,
                            end_date=window_end,
                            trading_dates=trading_dates,
                            require_true_earnings=require_true_earnings,
                            allow_proxy_earnings=allow_proxy_earnings,
                        )
                        if not events:
                            no_event_symbols += 1
                            continue

                        expiry_strings = list(getattr(ticker, 'options', []) or [])
                        expiry_dates = []
                        for exp in expiry_strings:
                            try:
                                expiry_dates.append(datetime.strptime(exp, "%Y-%m-%d"))
                            except ValueError:
                                continue
                        expiry_dates = sorted(expiry_dates)
                        if not expiry_dates:
                            no_expiry_symbols += 1
                            continue

                        for event in events:
                            event_date = pd.Timestamp(event.get('event_date')).to_pydatetime().date()
                            release_timing = str(event.get('release_timing', 'UNKNOWN')).upper()
                            relative_day = (today - event_date).days
                            if relative_day < -lookahead_days or relative_day > lookback_days:
                                continue
                            eligible_events += 1

                            if relative_day < 0:
                                snapshot_phase = 'pre'
                            elif relative_day > 0:
                                snapshot_phase = 'post'
                            else:
                                if release_timing == 'BMO':
                                    snapshot_phase = 'post'
                                elif release_timing == 'AMC':
                                    snapshot_phase = 'pre'
                                else:
                                    snapshot_phase = 'neutral'

                            valid_exps = [exp for exp in expiry_dates if exp.date() >= event_date]
                            if not valid_exps:
                                continue
                            selected_exps = valid_exps[:max_expiries]
                            if len(selected_exps) == 1:
                                selected_exps = selected_exps + selected_exps
                            short_exp = selected_exps[0].strftime('%Y-%m-%d')
                            long_exp = selected_exps[1].strftime('%Y-%m-%d')

                            attempts += 1
                            try:
                                front_chain = ticker.option_chain(short_exp)
                                back_chain = ticker.option_chain(long_exp)
                            except Exception:
                                chain_errors += 1
                                continue
                            front_iv, atm_strike = self._extract_atm_iv_from_chain(front_chain, last_close)
                            back_iv, _ = self._extract_atm_iv_from_chain(back_chain, last_close)
                            if not np.isfinite(front_iv) or not np.isfinite(back_iv):
                                no_iv_events += 1
                                continue
                            if front_iv <= 0 or back_iv <= 0:
                                no_iv_events += 1
                                continue

                            term_ratio = float(front_iv / max(back_iv, 1e-6))
                            cursor.execute(
                                """
                                INSERT OR REPLACE INTO earnings_option_snapshots
                                (symbol, event_date, capture_date, relative_day, release_timing, snapshot_phase,
                                 short_expiry, long_expiry, atm_strike, front_iv, back_iv, term_ratio,
                                 underlying_price, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    symbol,
                                    event_date.strftime('%Y-%m-%d'),
                                    today.strftime('%Y-%m-%d'),
                                    int(relative_day),
                                    release_timing,
                                    snapshot_phase,
                                    short_exp,
                                    long_exp,
                                    float(atm_strike) if np.isfinite(atm_strike) else None,
                                    float(front_iv),
                                    float(back_iv),
                                    term_ratio,
                                    float(last_close),
                                    'yfinance_live',
                                )
                            )
                            captured += 1

                    except Exception:
                        errors += 1
                        continue

                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Failed to capture earnings option snapshots: {e}")
            return {'captured': 0, 'attempts': 0, 'errors': len(symbols), 'symbols': symbols}

        self.logger.info(
            "📸 Captured %d earnings option snapshots "
            "(%d chain attempts, %d eligible events, %d symbol errors)",
            captured,
            attempts,
            eligible_events,
            errors,
        )
        self.logger.info(
            "📸 Snapshot diagnostics: no-price=%d, no-events=%d, no-expiries=%d, "
            "no-iv=%d, chain-errors=%d",
            no_price_symbols,
            no_event_symbols,
            no_expiry_symbols,
            no_iv_events,
            chain_errors,
        )
        return {
            'captured': int(captured),
            'attempts': int(attempts),
            'errors': int(errors),
            'symbols': symbols,
            'capture_date': today.strftime('%Y-%m-%d'),
            'eligible_events': int(eligible_events),
            'diagnostics': {
                'no_price_symbols': int(no_price_symbols),
                'no_event_symbols': int(no_event_symbols),
                'no_expiry_symbols': int(no_expiry_symbols),
                'no_iv_events': int(no_iv_events),
                'chain_errors': int(chain_errors),
            },
        }

    def _extract_atm_iv_from_chain(self, option_chain: Any, underlying_price: float) -> Tuple[float, float]:
        """Extract ATM implied volatility from call/put chain pair."""
        try:
            calls = getattr(option_chain, 'calls', None)
            puts = getattr(option_chain, 'puts', None)
            if calls is None or puts is None or calls.empty or puts.empty:
                return float('nan'), float('nan')

            calls_df = calls[['strike', 'impliedVolatility']].dropna()
            puts_df = puts[['strike', 'impliedVolatility']].dropna()
            if calls_df.empty or puts_df.empty:
                return float('nan'), float('nan')

            call_idx = (calls_df['strike'] - float(underlying_price)).abs().idxmin()
            put_idx = (puts_df['strike'] - float(underlying_price)).abs().idxmin()
            call_row = calls_df.loc[call_idx]
            put_row = puts_df.loc[put_idx]
            atm_strike = float(np.nanmean([call_row['strike'], put_row['strike']]))
            atm_iv = float(np.nanmean([call_row['impliedVolatility'], put_row['impliedVolatility']]))
            return atm_iv, atm_strike
        except Exception:
            return float('nan'), float('nan')

    def calibrate_earnings_iv_decay_labels(self, min_pre_days: int = 1,
                                         max_pre_days: int = 12,
                                         min_post_days: int = 0,
                                         max_post_days: int = 5) -> pd.DataFrame:
        """
        Build IV-decay labels from captured pre/post earnings option snapshots.
        """
        min_pre_days = max(1, int(min_pre_days))
        max_pre_days = max(min_pre_days, int(max_pre_days))
        min_post_days = max(0, int(min_post_days))
        max_post_days = max(min_post_days, int(max_post_days))

        query = """
            SELECT symbol, event_date, capture_date, relative_day, release_timing, snapshot_phase,
                   short_expiry, long_expiry, atm_strike, front_iv, back_iv, term_ratio, underlying_price
            FROM earnings_option_snapshots
            ORDER BY symbol, event_date, capture_date
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                snapshots = pd.read_sql_query(query, conn)
        except Exception as e:
            self.logger.error(f"❌ Failed to read snapshots for calibration: {e}")
            return pd.DataFrame()

        if snapshots.empty:
            self.logger.warning("⚠️ No earnings option snapshots available for calibration")
            return pd.DataFrame()

        snapshots['capture_date'] = pd.to_datetime(snapshots['capture_date'])
        snapshot_event_count = int(snapshots[['symbol', 'event_date']].drop_duplicates().shape[0])
        pre_snapshot_count = int(
            (
                (snapshots['relative_day'] <= -min_pre_days)
                & (snapshots['relative_day'] >= -max_pre_days)
                & (snapshots['front_iv'] > 0)
                & (snapshots['back_iv'] > 0)
            ).sum()
        )
        post_snapshot_count = int(
            (
                (snapshots['relative_day'] >= min_post_days)
                & (snapshots['relative_day'] <= max_post_days)
                & (snapshots['front_iv'] > 0)
                & (snapshots['back_iv'] > 0)
                & ((snapshots['snapshot_phase'] == 'post') | (snapshots['relative_day'] > 0))
            ).sum()
        )
        label_rows: List[Dict[str, Any]] = []

        for (symbol, event_date), group in snapshots.groupby(['symbol', 'event_date']):
            g = group.sort_values('capture_date').copy()
            pre_candidates = g[
                (g['relative_day'] <= -min_pre_days) &
                (g['relative_day'] >= -max_pre_days) &
                (g['front_iv'] > 0) & (g['back_iv'] > 0)
            ]
            post_candidates = g[
                (g['relative_day'] >= min_post_days) &
                (g['relative_day'] <= max_post_days) &
                (g['front_iv'] > 0) & (g['back_iv'] > 0) &
                ((g['snapshot_phase'] == 'post') | (g['relative_day'] > 0))
            ]
            if pre_candidates.empty or post_candidates.empty:
                continue

            pre_row = pre_candidates.sort_values('capture_date').iloc[-1]
            post_row = post_candidates.sort_values('capture_date').iloc[0]

            pre_front = float(pre_row['front_iv'])
            post_front = float(post_row['front_iv'])
            pre_back = float(pre_row['back_iv'])
            post_back = float(post_row['back_iv'])
            if pre_front <= 0 or pre_back <= 0:
                continue

            front_iv_crush_pct = (post_front - pre_front) / pre_front
            back_iv_crush_pct = (post_back - pre_back) / pre_back if pre_back > 0 else 0.0
            term_ratio_change = float(post_row['term_ratio']) - float(pre_row['term_ratio'])
            underlying_move_pct = (
                (float(post_row['underlying_price']) - float(pre_row['underlying_price'])) /
                max(float(pre_row['underlying_price']), 1e-6)
            )

            pre_dist = abs(abs(int(pre_row['relative_day'])) - min_pre_days)
            post_dist = abs(int(post_row['relative_day']) - min_post_days)
            quality_score = float(np.clip(1.0 - 0.04 * pre_dist - 0.06 * post_dist, 0.10, 1.00))

            label_rows.append({
                'symbol': str(symbol),
                'event_date': str(event_date),
                'release_timing': str(pre_row['release_timing'] or 'UNKNOWN').upper(),
                'pre_capture_date': pd.Timestamp(pre_row['capture_date']).strftime('%Y-%m-%d'),
                'post_capture_date': pd.Timestamp(post_row['capture_date']).strftime('%Y-%m-%d'),
                'pre_front_iv': pre_front,
                'post_front_iv': post_front,
                'pre_back_iv': pre_back,
                'post_back_iv': post_back,
                'front_iv_crush_pct': float(front_iv_crush_pct),
                'back_iv_crush_pct': float(back_iv_crush_pct),
                'term_ratio_change': float(term_ratio_change),
                'underlying_move_pct': float(underlying_move_pct),
                'quality_score': quality_score,
                'source': 'snapshot_pair',
            })

        if not label_rows:
            self.logger.warning(
                "⚠️ No valid pre/post snapshot pairs for IV-decay calibration "
                "(events=%d, pre_candidates=%d, post_candidates=%d). "
                "Capture snapshots over multiple days around the same earnings cycle.",
                snapshot_event_count,
                pre_snapshot_count,
                post_snapshot_count,
            )
            return pd.DataFrame()

        labels_df = pd.DataFrame(label_rows)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO earnings_iv_decay_labels
                    (symbol, event_date, release_timing, pre_capture_date, post_capture_date,
                     pre_front_iv, post_front_iv, pre_back_iv, post_back_iv,
                     front_iv_crush_pct, back_iv_crush_pct, term_ratio_change,
                     underlying_move_pct, quality_score, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row['symbol'],
                            row['event_date'],
                            row['release_timing'],
                            row['pre_capture_date'],
                            row['post_capture_date'],
                            row['pre_front_iv'],
                            row['post_front_iv'],
                            row['pre_back_iv'],
                            row['post_back_iv'],
                            row['front_iv_crush_pct'],
                            row['back_iv_crush_pct'],
                            row['term_ratio_change'],
                            row['underlying_move_pct'],
                            row['quality_score'],
                            row['source'],
                        )
                        for row in label_rows
                    ]
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Failed to store IV-decay labels: {e}")
            return pd.DataFrame()

        self.logger.info(f"🧩 Calibrated {len(labels_df)} earnings IV-decay labels from snapshots")
        return labels_df

    def summarize_snapshot_pairing_progress(self, min_pre_days: int = 1,
                                          max_pre_days: int = 12,
                                          min_post_days: int = 0,
                                          max_post_days: int = 5) -> Dict[str, Any]:
        """Summarize pre/post snapshot coverage needed for IV-decay label calibration."""
        min_pre_days = max(1, int(min_pre_days))
        max_pre_days = max(min_pre_days, int(max_pre_days))
        min_post_days = max(0, int(min_post_days))
        max_post_days = max(min_post_days, int(max_post_days))

        query = """
            SELECT symbol, event_date, capture_date, relative_day, snapshot_phase, front_iv, back_iv
            FROM earnings_option_snapshots
            ORDER BY symbol, event_date, capture_date
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                snapshots = pd.read_sql_query(query, conn)
        except Exception as e:
            self.logger.error(f"❌ Failed to summarize snapshot pairing progress: {e}")
            return {
                'total_snapshots': 0,
                'total_events': 0,
                'events_with_pre': 0,
                'events_with_post': 0,
                'pairable_events': 0,
                'pending_pre_only_events': 0,
                'pending_post_only_events': 0,
                'unqualified_events': 0,
                'pairable_event_pct': 0.0,
                'capture_days': 0,
                'min_relative_day': None,
                'max_relative_day': None,
            }

        if snapshots.empty:
            return {
                'total_snapshots': 0,
                'total_events': 0,
                'events_with_pre': 0,
                'events_with_post': 0,
                'pairable_events': 0,
                'pending_pre_only_events': 0,
                'pending_post_only_events': 0,
                'unqualified_events': 0,
                'pairable_event_pct': 0.0,
                'capture_days': 0,
                'min_relative_day': None,
                'max_relative_day': None,
            }

        snapshots = snapshots.copy()
        snapshots['capture_date'] = pd.to_datetime(snapshots['capture_date'], errors='coerce')
        snapshots['relative_day'] = pd.to_numeric(snapshots['relative_day'], errors='coerce')
        snapshots['front_iv'] = pd.to_numeric(snapshots['front_iv'], errors='coerce')
        snapshots['back_iv'] = pd.to_numeric(snapshots['back_iv'], errors='coerce')
        valid_iv = (snapshots['front_iv'] > 0) & (snapshots['back_iv'] > 0)

        pre_mask = (
            valid_iv
            & (snapshots['relative_day'] <= -min_pre_days)
            & (snapshots['relative_day'] >= -max_pre_days)
        )
        post_mask = (
            valid_iv
            & (snapshots['relative_day'] >= min_post_days)
            & (snapshots['relative_day'] <= max_post_days)
            & ((snapshots['snapshot_phase'] == 'post') | (snapshots['relative_day'] > 0))
        )

        grouped = (
            snapshots.assign(
                pre_hit=pre_mask.astype(int),
                post_hit=post_mask.astype(int),
            )
            .groupby(['symbol', 'event_date'], dropna=False, as_index=False)
            .agg(
                pre_count=('pre_hit', 'sum'),
                post_count=('post_hit', 'sum'),
                observations=('relative_day', 'size'),
            )
        )

        total_events = int(len(grouped))
        events_with_pre = int((grouped['pre_count'] > 0).sum())
        events_with_post = int((grouped['post_count'] > 0).sum())
        pairable_events = int(((grouped['pre_count'] > 0) & (grouped['post_count'] > 0)).sum())
        pending_pre_only = int(((grouped['pre_count'] > 0) & (grouped['post_count'] == 0)).sum())
        pending_post_only = int(((grouped['pre_count'] == 0) & (grouped['post_count'] > 0)).sum())
        unqualified_events = int(((grouped['pre_count'] == 0) & (grouped['post_count'] == 0)).sum())

        if total_events > 0:
            pairable_event_pct = float(pairable_events / total_events)
        else:
            pairable_event_pct = 0.0

        relative_day_series = snapshots['relative_day'].dropna()
        if relative_day_series.empty:
            min_relative_day = None
            max_relative_day = None
        else:
            min_relative_day = int(relative_day_series.min())
            max_relative_day = int(relative_day_series.max())

        return {
            'total_snapshots': int(len(snapshots)),
            'total_events': total_events,
            'events_with_pre': events_with_pre,
            'events_with_post': events_with_post,
            'pairable_events': pairable_events,
            'pending_pre_only_events': pending_pre_only,
            'pending_post_only_events': pending_post_only,
            'unqualified_events': unqualified_events,
            'pairable_event_pct': pairable_event_pct,
            'capture_days': int(snapshots['capture_date'].dropna().dt.date.nunique()),
            'min_relative_day': min_relative_day,
            'max_relative_day': max_relative_day,
        }

    @staticmethod
    def _score_entry_timing(days_to_earnings: Optional[int], target_entry_dte: int,
                            entry_dte_band: int) -> float:
        """Prefer setups close to the pre-event timing sweet-spot."""
        if days_to_earnings is None or days_to_earnings <= 0:
            return 0.50
        band = float(max(1, entry_dte_band))
        z = (float(days_to_earnings) - float(max(1, target_entry_dte))) / band
        return float(np.clip(np.exp(-0.5 * z * z), 0.0, 1.0))

    @staticmethod
    def _safe_numeric(value: Any, default: float = 0.0) -> float:
        """Convert optional/messy inputs to float with fallback."""
        try:
            if value is None:
                return float(default)
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(numeric):
            return float(default)
        return numeric

    @staticmethod
    def _score_iv_rv_quality(iv_rv_ratio: float) -> float:
        """
        Reward elevated IV/RV while avoiding a purely monotonic chase of extremes.
        A bell-shaped core plus a premium-presence gate is more robust out-of-sample.
        """
        target = 1.28
        width = 0.38
        z = (float(iv_rv_ratio) - target) / width
        bell_component = float(np.exp(-0.5 * z * z))
        premium_gate = float(np.clip((float(iv_rv_ratio) - 0.92) / 0.40, 0.0, 1.0))
        return float(np.clip(0.60 * bell_component + 0.40 * premium_gate, 0.0, 1.0))

    @staticmethod
    def _score_term_structure_quality(term_slope: float) -> float:
        """
        Prefer moderate front-end richness in the event window, not extreme dislocations.
        """
        if not np.isfinite(term_slope):
            return 0.50
        target = 0.10
        width = 0.22
        z = (float(term_slope) - target) / width
        return float(np.clip(np.exp(-0.5 * z * z), 0.0, 1.0))

    def _score_setup_quality(self, row: Any, days_to_earnings: Optional[int] = None,
                             target_entry_dte: int = 6, entry_dte_band: int = 6) -> float:
        """Map feature state into a normalized [0, 1] pre-earnings setup quality score."""
        iv_rv_ratio = self._safe_numeric(getattr(row, 'iv30_rv30_ratio', 1.0), 1.0)
        momentum_5d = abs(self._safe_numeric(getattr(row, 'price_momentum_5d', 0.0), 0.0))
        rsi_14 = self._safe_numeric(getattr(row, 'rsi_14', 50.0), 50.0)
        bb_position = self._safe_numeric(getattr(row, 'bb_position', 0.5), 0.5)
        volume_ratio = self._safe_numeric(getattr(row, 'volume_ratio_10d', 1.0), 1.0)
        vix_level = self._safe_numeric(getattr(row, 'vix_level', 20.0), 20.0)
        term_slope = self._safe_numeric(getattr(row, 'vol_term_structure_slope', np.nan), np.nan)
        daily_share_volume = self._safe_numeric(getattr(row, 'volume', np.nan), np.nan)

        iv_component = self._score_iv_rv_quality(iv_rv_ratio)
        timing_component = self._score_entry_timing(days_to_earnings, target_entry_dte, entry_dte_band)
        stability_component = 1.0 - np.clip(momentum_5d / 0.06, 0.0, 1.0)
        rsi_component = 1.0 - np.clip(abs(rsi_14 - 50.0) / 35.0, 0.0, 1.0)
        bb_component = 1.0 - np.clip(abs(bb_position - 0.5) / 0.5, 0.0, 1.0)
        flow_component = np.clip((volume_ratio - 0.85) / 1.15, 0.0, 1.0)
        regime_component = 1.0 - np.clip(abs(vix_level - 22.0) / 18.0, 0.0, 1.0)
        term_component = self._score_term_structure_quality(term_slope)

        if np.isfinite(daily_share_volume):
            liquidity_component = np.clip(
                (np.log1p(max(1.0, daily_share_volume)) - np.log1p(250_000.0))
                / (np.log1p(25_000_000.0) - np.log1p(250_000.0)),
                0.0,
                1.0,
            )
        else:
            liquidity_component = 0.45

        # Very crowded prints can reverse quickly and carry wider execution drag.
        crowding_penalty = np.clip(max(volume_ratio - 2.6, 0.0) / 2.4, 0.0, 1.0)

        score = (
            0.24 * iv_component
            + 0.18 * timing_component
            + 0.16 * stability_component
            + 0.10 * rsi_component
            + 0.08 * bb_component
            + 0.12 * flow_component
            + 0.06 * regime_component
            + 0.06 * term_component
            + 0.10 * liquidity_component
            - 0.16 * crowding_penalty
        )
        return float(np.clip(score, 0.0, 1.0))

    def _rank_candidate_for_alpha(self, setup_score: float, crush_context: Dict[str, Any],
                                  days_to_earnings: Optional[int], target_entry_dte: int,
                                  entry_dte_band: int) -> float:
        """Blend setup quality with crush profile quality into final candidate rank."""
        confidence = float(np.clip(float(crush_context.get('confidence', 0.0)), 0.0, 1.0))
        signal_strength = float(np.clip(float(crush_context.get('signal_strength', 0.0)), 0.0, 1.0))
        magnitude_component = float(np.clip(float(crush_context.get('magnitude', 0.0)) / 0.30, 0.0, 1.0))
        edge_component = float(np.clip(float(crush_context.get('edge_score', 0.0)) / 0.10, 0.0, 1.0))
        timing_component = self._score_entry_timing(days_to_earnings, target_entry_dte, entry_dte_band)

        crush_quality = (
            0.34 * confidence
            + 0.26 * signal_strength
            + 0.24 * magnitude_component
            + 0.16 * edge_component
        )
        rank_score = (
            0.62 * float(setup_score)
            + 0.30 * crush_quality
            + 0.08 * timing_component
        )
        return float(np.clip(rank_score, 0.0, 1.35))

    def _resolve_forward_return(self, row: Any, hold_days: int) -> Optional[float]:
        """Project a hold-period return from available forward-return labels."""
        if hold_days <= 2:
            base_return = getattr(row, 'forward_return_1d', None)
            scale = float(hold_days)
        elif hold_days <= 7:
            base_return = getattr(row, 'forward_return_5d', None)
            scale = float(hold_days) / 5.0
        else:
            base_return = getattr(row, 'forward_return_21d', None)
            scale = float(hold_days) / 21.0

        if base_return is None or pd.isna(base_return):
            return None

        projected_return = float(base_return) * scale
        return float(np.clip(projected_return, -0.45, 0.45))

    def _estimate_debit_per_contract(self, underlying_price: float, iv_rv_ratio: float) -> float:
        """Estimate calendar debit as a smooth function of underlying level and vol premium."""
        debit_per_share = underlying_price * 0.012 * np.clip(
            0.85 + 0.40 * max(0.0, iv_rv_ratio - 0.90),
            0.70,
            1.80,
        )
        return float(np.clip(debit_per_share, 0.35, 12.0) * 100.0)

    def _derive_crush_signal_context(self, symbol: str, iv_rv_ratio: float,
                                   crush_profiles: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Resolve expected IV-crush edge and confidence for candidate scoring."""
        profile_map = crush_profiles or {}
        symbol_profile = profile_map.get(symbol)
        if symbol_profile is None:
            profile = profile_map.get('__global__', {})
            profile_source = 'global'
        else:
            profile = symbol_profile
            profile_source = 'symbol'

        expected_crush_pct = float(profile.get('expected_front_iv_crush_pct', profile.get('median_front_iv_crush_pct', -0.18)))
        sample_size = float(profile.get('sample_size', 0.0))
        confidence = profile.get('confidence', None)
        if confidence is None or not np.isfinite(float(confidence)):
            confidence = float(np.clip(sample_size / 8.0, 0.0, 1.0))
        else:
            confidence = float(np.clip(float(confidence), 0.0, 1.0))
        crush_magnitude = float(max(0.0, -expected_crush_pct))
        crush_signal = float(np.clip((iv_rv_ratio - 0.95) / 0.60, 0.0, 1.0))
        edge_score = float(crush_magnitude * confidence * crush_signal)

        return {
            'expected_front_iv_crush_pct': expected_crush_pct,
            'sample_size': sample_size,
            'confidence': confidence,
            'magnitude': crush_magnitude,
            'signal_strength': crush_signal,
            'edge_score': edge_score,
            'profile_source': profile_source,
        }

    def _simulate_walk_forward_trade(self, session_id: str, trade_date: datetime, row: Any,
                                   setup_score: float, hold_days: int, contracts: int,
                                   execution_profile: str,
                                   execution_cost_model: ExecutionCostModel,
                                   event_date: datetime,
                                   days_to_earnings: int,
                                   crush_context: Optional[Dict[str, Any]] = None,
                                   crush_profiles: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[BacktestTrade]:
        """Compute deterministic gross/net calendar spread return for a single row."""
        forward_return = self._resolve_forward_return(row, hold_days)
        if forward_return is None:
            return None

        symbol = str(getattr(row, 'symbol', 'UNKNOWN'))
        underlying_price = max(1.0, float(getattr(row, 'underlying_price', 0.0) or 0.0))
        iv_rv_ratio = float(np.clip(float(getattr(row, 'iv30_rv30_ratio', 1.0) or 1.0), 0.60, 2.50))
        realized_vol_30d = float(getattr(row, 'realized_vol_30d', np.nan))
        if not np.isfinite(realized_vol_30d) or realized_vol_30d <= 0:
            realized_vol_30d = 0.22

        expected_move = max(
            0.01,
            realized_vol_30d * np.sqrt(hold_days / 252.0) * (0.95 + 0.30 * max(0.0, iv_rv_ratio - 1.0)),
        )
        move_ratio = abs(forward_return) / expected_move

        momentum_5d = abs(float(getattr(row, 'price_momentum_5d', 0.0) or 0.0))
        theta_carry = (0.09 + 0.20 * max(0.0, iv_rv_ratio - 0.95)) * np.sqrt(hold_days / 7.0)
        stability_bonus = 0.14 * max(0.0, 1.0 - move_ratio)
        overshoot_penalty = 0.34 * max(0.0, move_ratio - 1.0)
        trend_penalty = 0.10 * np.clip(momentum_5d / 0.05, 0.0, 1.5)
        regime_bonus = 0.08 * (setup_score - 0.5)

        # Learned IV-crush profile adjustment from captured event snapshots.
        if crush_context is None:
            crush_context = self._derive_crush_signal_context(
                symbol=symbol,
                iv_rv_ratio=iv_rv_ratio,
                crush_profiles=crush_profiles,
            )
        predicted_front_iv_crush_pct = float(crush_context.get('expected_front_iv_crush_pct', -0.18))
        crush_confidence = float(np.clip(float(crush_context.get('confidence', 0.0)), 0.0, 1.0))
        crush_edge_score = float(max(0.0, float(crush_context.get('edge_score', 0.0))))
        crush_profile_sample_size = float(max(0.0, float(crush_context.get('sample_size', 0.0))))
        crush_adjustment = 0.12 * np.clip(crush_edge_score / 0.10, 0.0, 1.5)

        gross_return_pct = float(np.clip(
            theta_carry + stability_bonus + regime_bonus + crush_adjustment - overshoot_penalty - trend_penalty,
            -1.0,
            1.35,
        ))

        debit_per_contract = self._estimate_debit_per_contract(underlying_price, iv_rv_ratio)
        volume_ratio = float(np.clip(float(getattr(row, 'volume_ratio_10d', 1.0) or 1.0), 0.30, 3.0))
        daily_share_volume = max(1.0, float(getattr(row, 'volume', 1.0) or 1.0))
        option_volume_proxy = max(20.0, np.sqrt(daily_share_volume) * 0.50 * volume_ratio)
        open_interest_proxy = max(80.0, option_volume_proxy * 6.0)
        short_spread = float(np.clip(0.04 + 0.16 / (volume_ratio + 0.35), 0.03, 0.30))
        long_spread = float(np.clip(short_spread + 0.02, 0.05, 0.36))

        cost_estimate = execution_cost_model.estimate_calendar_round_trip_cost(
            short_spread=short_spread,
            long_spread=long_spread,
            average_volume=option_volume_proxy,
            open_interest=open_interest_proxy,
            contracts=contracts,
        )
        transaction_cost_per_contract = float(cost_estimate['cost_per_contract'])

        gross_pnl_per_contract = debit_per_contract * gross_return_pct
        net_pnl_per_contract = gross_pnl_per_contract - transaction_cost_per_contract
        max_loss_per_contract = debit_per_contract + transaction_cost_per_contract
        net_pnl_per_contract = float(np.clip(net_pnl_per_contract, -max_loss_per_contract, debit_per_contract * 1.5))
        net_return_pct = float(net_pnl_per_contract / debit_per_contract) if debit_per_contract > 0 else 0.0

        return BacktestTrade(
            session_id=session_id,
            symbol=symbol,
            trade_date=trade_date,
            event_date=event_date,
            days_to_earnings=days_to_earnings,
            contracts=contracts,
            hold_days=hold_days,
            setup_score=setup_score,
            debit_per_contract=debit_per_contract,
            transaction_cost_per_contract=transaction_cost_per_contract,
            gross_return_pct=gross_return_pct,
            net_return_pct=net_return_pct,
            pnl_per_contract=net_pnl_per_contract,
            underlying_return=float(forward_return),
            expected_move=float(expected_move),
            move_ratio=float(move_ratio),
            predicted_front_iv_crush_pct=predicted_front_iv_crush_pct,
            crush_confidence=crush_confidence,
            crush_edge_score=crush_edge_score,
            crush_profile_sample_size=crush_profile_sample_size,
            execution_profile=execution_profile,
        )

    def _calculate_max_drawdown(self, pnl_per_trade: np.ndarray) -> float:
        """Compute peak-to-trough drawdown from cumulative PnL."""
        if pnl_per_trade.size == 0:
            return 0.0
        cumulative = np.cumsum(pnl_per_trade)
        peaks = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peaks
        return float(drawdowns.min()) if len(drawdowns) else 0.0

    def _store_backtest_session(self, session: BacktestSession):
        """Store backtest session metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO backtest_sessions
                    (session_id, strategy_name, start_date, end_date, universe,
                     parameters, total_trades, win_rate, total_pnl, sharpe_ratio,
                     max_drawdown, calmar_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id, session.strategy_name,
                    session.start_date.strftime('%Y-%m-%d'),
                    session.end_date.strftime('%Y-%m-%d'),
                    json.dumps(session.universe),
                    json.dumps(self._normalize_json_value(session.parameters)),
                    session.total_trades, session.win_rate, session.total_pnl,
                    session.sharpe_ratio, session.max_drawdown, session.calmar_ratio
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Failed to store backtest session: {e}")

    def _update_backtest_results(self, session: BacktestSession):
        """Update backtest session with final results"""
        self._store_backtest_session(session)  # Same method, will replace due to OR REPLACE

    def _store_backtest_trades(self, session_id: str, trades: List[BacktestTrade]):
        """Persist trade-level walk-forward records for diagnostics and research."""
        if not trades:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM backtest_trades WHERE session_id = ?", (session_id,))
                cursor.executemany("""
                    INSERT INTO backtest_trades
                    (session_id, symbol, trade_date, event_date, days_to_earnings, contracts, hold_days, setup_score, debit_per_contract,
                     transaction_cost_per_contract, gross_return_pct, net_return_pct, pnl_per_contract,
                     underlying_return, expected_move, move_ratio, predicted_front_iv_crush_pct,
                     crush_confidence, crush_edge_score, crush_profile_sample_size, execution_profile)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        trade.session_id,
                        trade.symbol,
                        trade.trade_date.strftime('%Y-%m-%d'),
                        trade.event_date.strftime('%Y-%m-%d'),
                        trade.days_to_earnings,
                        trade.contracts,
                        trade.hold_days,
                        trade.setup_score,
                        trade.debit_per_contract,
                        trade.transaction_cost_per_contract,
                        trade.gross_return_pct,
                        trade.net_return_pct,
                        trade.pnl_per_contract,
                        trade.underlying_return,
                        trade.expected_move,
                        trade.move_ratio,
                        trade.predicted_front_iv_crush_pct,
                        trade.crush_confidence,
                        trade.crush_edge_score,
                        trade.crush_profile_sample_size,
                        trade.execution_profile,
                    )
                    for trade in trades
                ])
                conn.commit()
        except Exception as e:
            self.logger.error(f"❌ Failed to store backtest trades: {e}")

    def get_training_dataset(self, symbols: List[str] = None,
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get ML training dataset with features and targets

        Args:
            symbols: List of symbols to include
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with features and targets for ML training
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT
                        f.*,
                        p.volume,
                        p.realized_vol_30d,
                        p.realized_vol_60d
                    FROM ml_features f
                    LEFT JOIN daily_prices p ON f.symbol = p.symbol AND f.date = p.date
                    WHERE 1=1
                """

                params = []
                if symbols:
                    query += " AND f.symbol IN ({})".format(','.join(['?'] * len(symbols)))
                    params.extend(symbols)

                if start_date:
                    query += " AND f.date >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND f.date <= ?"
                    params.append(end_date)

                query += " ORDER BY f.symbol, f.date"

                df = pd.read_sql_query(query, conn, params=params)

                self.logger.info(f"📊 Training dataset: {len(df)} samples, {len(df.columns)} features")
                return df

        except Exception as e:
            self.logger.error(f"❌ Failed to get training dataset: {e}")
            return pd.DataFrame()

    def get_backtest_results(self, session_id: str = None) -> pd.DataFrame:
        """Get backtest results summary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if session_id:
                    query = "SELECT * FROM backtest_sessions WHERE session_id = ?"
                    df = pd.read_sql_query(query, conn, params=[session_id])
                else:
                    query = "SELECT * FROM backtest_sessions ORDER BY created_at DESC LIMIT 10"
                    df = pd.read_sql_query(query, conn)

                return df

        except Exception as e:
            self.logger.error(f"❌ Failed to get backtest results: {e}")
            return pd.DataFrame()

    def get_backtest_trades(self, session_id: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Get trade-level records for a backtest session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT *
                    FROM backtest_trades
                    WHERE session_id = ?
                    ORDER BY trade_date, crush_edge_score DESC, setup_score DESC
                """
                params: List[Any] = [session_id]
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(int(limit))
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            self.logger.error(f"❌ Failed to get backtest trades: {e}")
            return pd.DataFrame()

    def _load_iv_crush_profiles(self, universe: List[str]) -> Dict[str, Dict[str, float]]:
        """Load expected IV-crush profile with shrinkage and confidence calibration."""
        default_profile = {
            'expected_front_iv_crush_pct': -0.18,
            'median_front_iv_crush_pct': -0.18,
            'sample_size': 6.0,
            'confidence': 0.50,
            'crush_std': 0.12,
            'avg_quality_score': 0.75,
            'profile_source': 'prior',
        }

        placeholders = ",".join(["?"] * len(universe)) if universe else "?"
        symbol_filter = f"WHERE symbol IN ({placeholders})" if universe else "WHERE 1=0"
        query = f"""
            SELECT symbol,
                   COUNT(*) as sample_size,
                   AVG(front_iv_crush_pct) as avg_front_iv_crush_pct,
                   AVG(front_iv_crush_pct * front_iv_crush_pct) as avg_front_iv_sq,
                   AVG(quality_score) as avg_quality_score
            FROM earnings_iv_decay_labels
            {symbol_filter}
            GROUP BY symbol
        """
        profiles: Dict[str, Dict[str, float]] = {}

        def _profile_confidence(sample_size: float, crush_std: float, avg_quality: float) -> float:
            sample_factor = float(np.clip(sample_size / 10.0, 0.0, 1.0))
            quality_factor = float(np.clip(avg_quality, 0.20, 1.00))
            stability_factor = float(np.clip(1.0 - np.clip(crush_std / 0.22, 0.0, 0.90), 0.10, 1.0))
            return float(np.clip(
                0.15 + 0.55 * sample_factor + 0.20 * quality_factor + 0.10 * stability_factor,
                0.15,
                0.98,
            ))

        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(query, universe if universe else ['__none__']).fetchall()
                global_row = conn.execute(
                    """
                    SELECT COUNT(*) as sample_size,
                           AVG(front_iv_crush_pct) as avg_front_iv_crush_pct,
                           AVG(front_iv_crush_pct * front_iv_crush_pct) as avg_front_iv_sq,
                           AVG(quality_score) as avg_quality_score
                    FROM earnings_iv_decay_labels
                    """
                ).fetchone()

            global_profile = dict(default_profile)
            if global_row and global_row[0]:
                g_sample = float(global_row[0] or 0.0)
                g_avg = float(global_row[1]) if global_row[1] is not None else default_profile['expected_front_iv_crush_pct']
                g_avg_sq = float(global_row[2]) if global_row[2] is not None else g_avg * g_avg
                g_quality = float(global_row[3]) if global_row[3] is not None else default_profile['avg_quality_score']
                g_var = float(max(0.0, g_avg_sq - g_avg * g_avg))
                g_std = float(np.sqrt(g_var))
                g_conf = _profile_confidence(g_sample, g_std, g_quality)
                global_profile = {
                    'expected_front_iv_crush_pct': g_avg,
                    'median_front_iv_crush_pct': g_avg,
                    'sample_size': g_sample,
                    'confidence': g_conf,
                    'crush_std': g_std,
                    'avg_quality_score': g_quality,
                    'profile_source': 'global_labels',
                }

            profiles['__global__'] = global_profile

            for symbol, sample_size, avg_crush, avg_sq, avg_quality in rows:
                s_sample = float(sample_size or 0.0)
                s_avg = float(avg_crush) if avg_crush is not None else global_profile['expected_front_iv_crush_pct']
                s_avg_sq = float(avg_sq) if avg_sq is not None else s_avg * s_avg
                s_quality = float(avg_quality) if avg_quality is not None else global_profile['avg_quality_score']
                s_var = float(max(0.0, s_avg_sq - s_avg * s_avg))
                s_std = float(np.sqrt(s_var))
                raw_confidence = _profile_confidence(s_sample, s_std, s_quality)

                shrink = float(np.clip(s_sample / (s_sample + 8.0), 0.0, 1.0))
                expected = float(
                    shrink * s_avg + (1.0 - shrink) * float(global_profile['expected_front_iv_crush_pct'])
                )
                confidence = float(
                    np.clip(
                        shrink * raw_confidence + (1.0 - shrink) * float(global_profile['confidence']),
                        0.10,
                        0.99,
                    )
                )

                profiles[str(symbol)] = {
                    'expected_front_iv_crush_pct': expected,
                    'median_front_iv_crush_pct': expected,
                    'sample_size': s_sample,
                    'confidence': confidence,
                    'crush_std': s_std,
                    'avg_quality_score': s_quality,
                    'profile_source': 'symbol_labels',
                }
        except Exception:
            profiles['__global__'] = dict(default_profile)
        return profiles

    def _get_session_crush_prediction_metrics(self, session_id: str) -> Dict[str, float]:
        """Compute crush prediction quality metrics for one backtest session."""
        query = """
            SELECT
                COUNT(*) AS total_trades,
                SUM(CASE WHEN l.front_iv_crush_pct IS NOT NULL THEN 1 ELSE 0 END) AS labeled_trades,
                AVG(t.crush_confidence) AS mean_crush_confidence,
                AVG(t.crush_edge_score) AS mean_crush_edge,
                AVG(CASE WHEN l.front_iv_crush_pct IS NOT NULL
                         THEN ABS(t.predicted_front_iv_crush_pct - l.front_iv_crush_pct)
                         ELSE NULL END) AS crush_mae,
                AVG(CASE WHEN l.front_iv_crush_pct IS NOT NULL AND
                               ((t.predicted_front_iv_crush_pct < 0 AND l.front_iv_crush_pct < 0) OR
                                (t.predicted_front_iv_crush_pct >= 0 AND l.front_iv_crush_pct >= 0))
                         THEN 1.0
                         WHEN l.front_iv_crush_pct IS NOT NULL THEN 0.0
                         ELSE NULL END) AS directional_accuracy
            FROM backtest_trades t
            LEFT JOIN earnings_iv_decay_labels l
              ON t.symbol = l.symbol
             AND t.event_date = l.event_date
            WHERE t.session_id = ?
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(query, [session_id]).fetchone()
            if not row:
                return {
                    'total_trades': 0.0,
                    'labeled_trades': 0.0,
                    'coverage': 0.0,
                    'mean_crush_confidence': 0.0,
                    'mean_crush_edge': 0.0,
                    'crush_mae': np.nan,
                    'directional_accuracy': np.nan,
                }
            total_trades = float(row[0] or 0.0)
            labeled_trades = float(row[1] or 0.0)
            coverage = float(labeled_trades / total_trades) if total_trades > 0 else 0.0
            return {
                'total_trades': total_trades,
                'labeled_trades': labeled_trades,
                'coverage': coverage,
                'mean_crush_confidence': float(row[2]) if row[2] is not None else 0.0,
                'mean_crush_edge': float(row[3]) if row[3] is not None else 0.0,
                'crush_mae': float(row[4]) if row[4] is not None else np.nan,
                'directional_accuracy': float(row[5]) if row[5] is not None else np.nan,
            }
        except Exception:
            return {
                'total_trades': 0.0,
                'labeled_trades': 0.0,
                'coverage': 0.0,
                'mean_crush_confidence': 0.0,
                'mean_crush_edge': 0.0,
                'crush_mae': np.nan,
                'directional_accuracy': np.nan,
            }

    def build_rolling_crush_scorecard(self, session_id: Optional[str] = None,
                                    window_size: int = 40,
                                    min_confidence: float = 0.0) -> pd.DataFrame:
        """
        Build rolling prediction-vs-realization scorecard for IV-crush forecasts.
        Returns one row per labeled trade event.
        """
        window_size = max(5, int(window_size))
        min_confidence = float(np.clip(float(min_confidence), 0.0, 1.0))

        query = """
            SELECT
                t.session_id,
                t.symbol,
                t.trade_date,
                t.event_date,
                t.days_to_earnings,
                t.setup_score,
                t.net_return_pct,
                t.predicted_front_iv_crush_pct,
                t.crush_confidence,
                t.crush_edge_score,
                t.crush_profile_sample_size,
                l.front_iv_crush_pct AS realized_front_iv_crush_pct,
                l.quality_score AS label_quality_score
            FROM backtest_trades t
            LEFT JOIN earnings_iv_decay_labels l
              ON t.symbol = l.symbol
             AND t.event_date = l.event_date
            WHERE t.crush_confidence >= ?
        """
        params: List[Any] = [min_confidence]
        if session_id:
            query += " AND t.session_id = ?"
            params.append(session_id)
        query += " ORDER BY t.trade_date, t.symbol"

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            self.logger.error(f"❌ Failed to build crush scorecard: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df = df.dropna(subset=['trade_date']).copy()

        labeled = df.dropna(subset=['realized_front_iv_crush_pct']).copy()
        if labeled.empty:
            return labeled

        labeled['predicted_front_iv_crush_pct'] = labeled['predicted_front_iv_crush_pct'].astype(float)
        labeled['realized_front_iv_crush_pct'] = labeled['realized_front_iv_crush_pct'].astype(float)
        labeled['prediction_error'] = (
            labeled['predicted_front_iv_crush_pct'] - labeled['realized_front_iv_crush_pct']
        )
        labeled['abs_error'] = labeled['prediction_error'].abs()
        labeled['directional_match'] = (
            (labeled['predicted_front_iv_crush_pct'] < 0) == (labeled['realized_front_iv_crush_pct'] < 0)
        ).astype(float)
        labeled['rolling_mae'] = labeled['abs_error'].rolling(
            window=window_size,
            min_periods=5
        ).mean()
        labeled['rolling_directional_accuracy'] = labeled['directional_match'].rolling(
            window=window_size,
            min_periods=5
        ).mean()
        labeled['rolling_prediction_bias'] = labeled['prediction_error'].rolling(
            window=window_size,
            min_periods=5
        ).mean()
        labeled['rolling_predicted_crush'] = labeled['predicted_front_iv_crush_pct'].rolling(
            window=window_size,
            min_periods=5
        ).mean()
        labeled['rolling_realized_crush'] = labeled['realized_front_iv_crush_pct'].rolling(
            window=window_size,
            min_periods=5
        ).mean()
        labeled['rolling_confidence'] = labeled['crush_confidence'].rolling(
            window=window_size,
            min_periods=5
        ).mean()
        labeled['rolling_calibration_score'] = 1.0 - np.clip(labeled['rolling_mae'] / 0.25, 0.0, 1.0)
        labeled['rolling_institutional_score'] = (
            0.55 * labeled['rolling_calibration_score'].fillna(0.0)
            + 0.30 * labeled['rolling_directional_accuracy'].fillna(0.0)
            + 0.15 * labeled['rolling_confidence'].fillna(0.0)
        )
        labeled['evaluation_count'] = np.arange(1, len(labeled) + 1)

        return labeled.reset_index(drop=True)

    def summarize_crush_scorecard(self, session_id: Optional[str] = None,
                                window_size: int = 40,
                                min_confidence: float = 0.0) -> Dict[str, Any]:
        """Summarize rolling crush prediction quality and symbol-level calibration."""
        scorecard_df = self.build_rolling_crush_scorecard(
            session_id=session_id,
            window_size=window_size,
            min_confidence=min_confidence,
        )
        if scorecard_df.empty:
            return {
                'rows': 0,
                'session_id': session_id,
                'mean_mae': np.nan,
                'directional_accuracy': np.nan,
                'mean_confidence': np.nan,
                'mean_predicted_crush': np.nan,
                'mean_realized_crush': np.nan,
                'latest_institutional_score': np.nan,
                'window_size': int(max(5, int(window_size))),
                'by_symbol': [],
            }

        latest = scorecard_df.iloc[-1]
        grouped = (
            scorecard_df.groupby('symbol')
            .agg(
                observations=('symbol', 'count'),
                mean_mae=('abs_error', 'mean'),
                directional_accuracy=('directional_match', 'mean'),
                mean_confidence=('crush_confidence', 'mean'),
                mean_predicted_crush=('predicted_front_iv_crush_pct', 'mean'),
                mean_realized_crush=('realized_front_iv_crush_pct', 'mean'),
                avg_net_return=('net_return_pct', 'mean'),
            )
            .reset_index()
            .sort_values(by=['directional_accuracy', 'mean_mae'], ascending=[False, True])
            .reset_index(drop=True)
        )

        return {
            'rows': int(len(scorecard_df)),
            'session_id': session_id,
            'mean_mae': float(scorecard_df['abs_error'].mean()),
            'directional_accuracy': float(scorecard_df['directional_match'].mean()),
            'mean_confidence': float(scorecard_df['crush_confidence'].mean()),
            'mean_predicted_crush': float(scorecard_df['predicted_front_iv_crush_pct'].mean()),
            'mean_realized_crush': float(scorecard_df['realized_front_iv_crush_pct'].mean()),
            'latest_institutional_score': float(latest.get('rolling_institutional_score', np.nan)),
            'window_size': int(max(5, int(window_size))),
            'by_symbol': grouped.to_dict(orient='records'),
        }

    def _load_trade_diagnostics_frame(self, session_id: Optional[str] = None,
                                    min_confidence: float = 0.0) -> pd.DataFrame:
        """Load a normalized diagnostics frame for backtest-trade analytics."""
        min_confidence = float(np.clip(float(min_confidence), 0.0, 1.0))

        query = """
            SELECT
                t.session_id,
                t.symbol,
                t.trade_date,
                t.event_date,
                t.days_to_earnings,
                t.hold_days,
                t.contracts,
                t.setup_score,
                t.debit_per_contract,
                t.transaction_cost_per_contract,
                t.net_return_pct,
                t.pnl_per_contract,
                t.crush_confidence,
                t.crush_edge_score,
                t.predicted_front_iv_crush_pct,
                f.vix_level,
                f.iv30_rv30_ratio,
                l.front_iv_crush_pct AS realized_front_iv_crush_pct,
                l.quality_score AS label_quality_score
            FROM backtest_trades t
            LEFT JOIN ml_features f
              ON t.symbol = f.symbol
             AND t.trade_date = f.date
            LEFT JOIN earnings_iv_decay_labels l
              ON t.symbol = l.symbol
             AND t.event_date = l.event_date
            WHERE t.crush_confidence >= ?
        """
        params: List[Any] = [min_confidence]
        if session_id:
            query += " AND t.session_id = ?"
            params.append(session_id)
        query += " ORDER BY t.trade_date, t.symbol"

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            self.logger.error(f"❌ Failed to load trade diagnostics frame: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df = df.dropna(subset=['trade_date']).copy()

        numeric_cols = [
            'days_to_earnings', 'hold_days', 'contracts', 'setup_score',
            'debit_per_contract', 'transaction_cost_per_contract',
            'net_return_pct', 'pnl_per_contract',
            'crush_confidence', 'crush_edge_score', 'predicted_front_iv_crush_pct',
            'vix_level', 'iv30_rv30_ratio', 'realized_front_iv_crush_pct',
            'label_quality_score',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Conservative fallback values to keep grouping stable when upstream joins are sparse.
        if 'vix_level' in df.columns:
            df['vix_level'] = df['vix_level'].fillna(20.0)
        if 'iv30_rv30_ratio' in df.columns:
            df['iv30_rv30_ratio'] = df['iv30_rv30_ratio'].fillna(1.10)
        if 'setup_score' in df.columns:
            df['setup_score'] = df['setup_score'].fillna(0.5).clip(lower=0.0, upper=1.0)
        if 'crush_edge_score' in df.columns:
            df['crush_edge_score'] = df['crush_edge_score'].fillna(0.0).clip(lower=0.0)
        if 'crush_confidence' in df.columns:
            df['crush_confidence'] = df['crush_confidence'].fillna(0.0).clip(lower=0.0, upper=1.0)
        if 'contracts' in df.columns:
            df['contracts'] = df['contracts'].fillna(1).clip(lower=1)
        if 'hold_days' in df.columns:
            df['hold_days'] = df['hold_days'].fillna(7).clip(lower=1)

        return df

    def build_regime_diagnostics(self, session_id: Optional[str] = None,
                               min_confidence: float = 0.0) -> pd.DataFrame:
        """
        Build regime-level diagnostics for calendar-spread alpha quality.
        Regimes bucket by VIX, days-to-earnings, and IV/RV state.
        """
        df = self._load_trade_diagnostics_frame(session_id=session_id, min_confidence=min_confidence)
        if df.empty:
            return df

        df = df.copy()
        df['vix_regime'] = pd.cut(
            df['vix_level'],
            bins=[-np.inf, 15.0, 25.0, np.inf],
            labels=['LOW_VOL', 'MID_VOL', 'HIGH_VOL'],
            include_lowest=True,
        )
        df['dte_bucket'] = pd.cut(
            df['days_to_earnings'].fillna(-1),
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=['DTE_0_2', 'DTE_3_5', 'DTE_6_10', 'DTE_11_PLUS'],
            include_lowest=True,
        )
        df['ivrv_bucket'] = pd.cut(
            df['iv30_rv30_ratio'],
            bins=[-np.inf, 1.05, 1.35, np.inf],
            labels=['IVRV_LOW', 'IVRV_MID', 'IVRV_HIGH'],
            include_lowest=True,
        )

        rows: List[Dict[str, Any]] = []
        for (vix_regime, dte_bucket, ivrv_bucket), group in df.groupby(
            ['vix_regime', 'dte_bucket', 'ivrv_bucket'],
            dropna=False
        ):
            trade_count = int(len(group))
            if trade_count == 0:
                continue

            labeled = group.dropna(subset=['realized_front_iv_crush_pct'])
            labeled_count = int(len(labeled))
            coverage = float(labeled_count / trade_count)

            mean_net_return = float(group['net_return_pct'].mean())
            median_net_return = float(group['net_return_pct'].median())
            return_std = float(group['net_return_pct'].std(ddof=0))
            sharpe_proxy = float(mean_net_return / return_std) if return_std > 0 else 0.0
            win_rate = float((group['net_return_pct'] > 0).mean())

            predicted_crush = float(group['predicted_front_iv_crush_pct'].mean())
            if labeled_count > 0:
                realized_crush = float(labeled['realized_front_iv_crush_pct'].mean())
                crush_mae = float(
                    (labeled['predicted_front_iv_crush_pct'] - labeled['realized_front_iv_crush_pct']).abs().mean()
                )
                crush_directional_accuracy = float(
                    ((labeled['predicted_front_iv_crush_pct'] < 0) == (labeled['realized_front_iv_crush_pct'] < 0)).mean()
                )
            else:
                realized_crush = np.nan
                crush_mae = np.nan
                crush_directional_accuracy = np.nan

            return_component = float(np.clip(mean_net_return / 0.06, -1.0, 1.5))
            win_component = float(np.clip((win_rate - 0.50) / 0.25, -1.0, 1.0))
            sharpe_component = float(np.clip(sharpe_proxy / 1.50, -1.0, 1.5))
            coverage_component = float(np.clip(coverage, 0.0, 1.0))
            if np.isfinite(crush_mae):
                calibration_component = float(np.clip(1.0 - (float(crush_mae) / 0.25), 0.0, 1.0))
            else:
                calibration_component = 0.0
            alpha_proxy_score = float(
                0.35 * return_component
                + 0.25 * win_component
                + 0.20 * sharpe_component
                + 0.10 * coverage_component
                + 0.10 * calibration_component
            )

            rows.append({
                'vix_regime': str(vix_regime),
                'dte_bucket': str(dte_bucket),
                'ivrv_bucket': str(ivrv_bucket),
                'trade_count': trade_count,
                'labeled_trade_count': labeled_count,
                'label_coverage': coverage,
                'mean_setup_score': float(group['setup_score'].mean()),
                'mean_net_return_pct': mean_net_return,
                'median_net_return_pct': median_net_return,
                'net_return_std': return_std,
                'win_rate': win_rate,
                'sharpe_proxy': sharpe_proxy,
                'mean_tx_cost_per_contract': float(group['transaction_cost_per_contract'].mean()),
                'mean_predicted_front_iv_crush_pct': predicted_crush,
                'mean_realized_front_iv_crush_pct': realized_crush,
                'crush_prediction_mae': crush_mae,
                'crush_directional_accuracy': crush_directional_accuracy,
                'alpha_proxy_score': alpha_proxy_score,
            })

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values(by=['alpha_proxy_score', 'trade_count'], ascending=[False, False])
            .reset_index(drop=True)
        )

    def summarize_regime_diagnostics(self, session_id: Optional[str] = None,
                                   min_confidence: float = 0.0,
                                   top_n: int = 5) -> Dict[str, Any]:
        """Summarize best/worst regimes and global diagnostics context."""
        top_n = max(1, int(top_n))
        diagnostics = self.build_regime_diagnostics(
            session_id=session_id,
            min_confidence=min_confidence,
        )
        if diagnostics.empty:
            return {
                'rows': 0,
                'session_id': session_id,
                'min_confidence': float(min_confidence),
                'overall_mean_net_return_pct': np.nan,
                'overall_win_rate': np.nan,
                'overall_crush_mae': np.nan,
                'overall_crush_directional_accuracy': np.nan,
                'best_regimes': [],
                'weak_regimes': [],
            }

        frame = self._load_trade_diagnostics_frame(
            session_id=session_id,
            min_confidence=min_confidence,
        )
        labeled = frame.dropna(subset=['realized_front_iv_crush_pct'])
        if labeled.empty:
            overall_crush_mae = np.nan
            overall_directional_accuracy = np.nan
        else:
            overall_crush_mae = float(
                (labeled['predicted_front_iv_crush_pct'] - labeled['realized_front_iv_crush_pct']).abs().mean()
            )
            overall_directional_accuracy = float(
                ((labeled['predicted_front_iv_crush_pct'] < 0) == (labeled['realized_front_iv_crush_pct'] < 0)).mean()
            )

        return {
            'rows': int(len(diagnostics)),
            'session_id': session_id,
            'min_confidence': float(np.clip(float(min_confidence), 0.0, 1.0)),
            'overall_mean_net_return_pct': float(frame['net_return_pct'].mean()) if not frame.empty else np.nan,
            'overall_win_rate': float((frame['net_return_pct'] > 0).mean()) if not frame.empty else np.nan,
            'overall_crush_mae': overall_crush_mae,
            'overall_crush_directional_accuracy': overall_directional_accuracy,
            'best_regimes': diagnostics.head(top_n).to_dict(orient='records'),
            'weak_regimes': diagnostics.tail(top_n).sort_values(
                by=['alpha_proxy_score', 'trade_count'],
                ascending=[True, False]
            ).to_dict(orient='records'),
        }

    def build_signal_decile_table(self, session_id: Optional[str] = None,
                                min_confidence: float = 0.0,
                                use_composite_signal: bool = True) -> pd.DataFrame:
        """
        Build decile-level performance table for entry-threshold tuning.
        """
        df = self._load_trade_diagnostics_frame(session_id=session_id, min_confidence=min_confidence)
        if df.empty:
            return df

        df = df.copy()
        base_score = df['setup_score'].clip(lower=0.0, upper=1.0)
        edge_score = np.clip(df['crush_edge_score'] / 0.12, 0.0, 1.0)
        if use_composite_signal:
            df['signal_score'] = np.clip(0.75 * base_score + 0.25 * edge_score, 0.0, 1.0)
        else:
            df['signal_score'] = base_score

        bucket_count = min(10, max(2, int(df['signal_score'].notna().sum())))
        ranked = df['signal_score'].rank(method='first')
        df['signal_decile'] = pd.qcut(ranked, q=bucket_count, labels=False, duplicates='drop') + 1

        rows: List[Dict[str, Any]] = []
        for decile, group in df.groupby('signal_decile', dropna=False):
            if group.empty:
                continue
            labeled = group.dropna(subset=['realized_front_iv_crush_pct'])
            labeled_count = int(len(labeled))
            trade_count = int(len(group))
            mean_net_return = float(group['net_return_pct'].mean())
            std_return = float(group['net_return_pct'].std(ddof=0))
            sharpe_proxy = float(mean_net_return / std_return) if std_return > 0 else 0.0
            if labeled_count > 0:
                crush_mae = float(
                    (labeled['predicted_front_iv_crush_pct'] - labeled['realized_front_iv_crush_pct']).abs().mean()
                )
                crush_directional_accuracy = float(
                    ((labeled['predicted_front_iv_crush_pct'] < 0) == (labeled['realized_front_iv_crush_pct'] < 0)).mean()
                )
            else:
                crush_mae = np.nan
                crush_directional_accuracy = np.nan

            rows.append({
                'signal_decile': int(decile),
                'min_signal_score': float(group['signal_score'].min()),
                'max_signal_score': float(group['signal_score'].max()),
                'trade_count': trade_count,
                'win_rate': float((group['net_return_pct'] > 0).mean()),
                'mean_net_return_pct': mean_net_return,
                'median_net_return_pct': float(group['net_return_pct'].median()),
                'net_return_std': std_return,
                'sharpe_proxy': sharpe_proxy,
                'mean_setup_score': float(group['setup_score'].mean()),
                'mean_crush_edge_score': float(group['crush_edge_score'].mean()),
                'mean_tx_cost_per_contract': float(group['transaction_cost_per_contract'].mean()),
                'labeled_trade_count': labeled_count,
                'label_coverage': float(labeled_count / trade_count),
                'crush_prediction_mae': crush_mae,
                'crush_directional_accuracy': crush_directional_accuracy,
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(by=['signal_decile']).reset_index(drop=True)

    def recommend_signal_threshold_from_deciles(self, session_id: Optional[str] = None,
                                              min_confidence: float = 0.0,
                                              min_trades: int = 30,
                                              use_composite_signal: bool = True) -> Dict[str, Any]:
        """
        Recommend a signal threshold using decile-conditioned, cost-aware trade outcomes.
        """
        min_trades = max(5, int(min_trades))
        diagnostics = self._load_trade_diagnostics_frame(session_id=session_id, min_confidence=min_confidence)
        if diagnostics.empty:
            return {
                'recommended': False,
                'reason': 'no_trade_data',
                'session_id': session_id,
                'min_confidence': float(min_confidence),
                'min_trades': min_trades,
                'available_trade_count': 0,
                'max_trade_count_for_thresholds': 0,
                'suggested_min_trades': None,
                'candidate_count': 0,
                'best_threshold': None,
                'best_metrics': None,
                'decile_table': [],
                'candidate_thresholds': [],
            }

        diagnostics = diagnostics.copy()
        available_trade_count = int(len(diagnostics))
        base_score = diagnostics['setup_score'].clip(lower=0.0, upper=1.0)
        edge_score = np.clip(diagnostics['crush_edge_score'] / 0.12, 0.0, 1.0)
        if use_composite_signal:
            diagnostics['signal_score'] = np.clip(0.75 * base_score + 0.25 * edge_score, 0.0, 1.0)
            signal_name = 'composite'
        else:
            diagnostics['signal_score'] = base_score
            signal_name = 'setup_score'

        decile_table = self.build_signal_decile_table(
            session_id=session_id,
            min_confidence=min_confidence,
            use_composite_signal=use_composite_signal,
        )
        if decile_table.empty:
            return {
                'recommended': False,
                'reason': 'no_decile_data',
                'session_id': session_id,
                'signal_name': signal_name,
                'min_confidence': float(min_confidence),
                'min_trades': min_trades,
                'available_trade_count': available_trade_count,
                'max_trade_count_for_thresholds': 0,
                'suggested_min_trades': None,
                'candidate_count': 0,
                'best_threshold': None,
                'best_metrics': None,
                'decile_table': [],
                'candidate_thresholds': [],
            }

        thresholds = sorted({
            float(v)
            for v in decile_table['min_signal_score'].dropna().tolist()
            if np.isfinite(v)
        })
        candidate_rows: List[Dict[str, Any]] = []
        max_trade_count_for_thresholds = 0
        for threshold in thresholds:
            subset = diagnostics[diagnostics['signal_score'] >= float(threshold)].copy()
            trade_count = int(len(subset))
            max_trade_count_for_thresholds = max(max_trade_count_for_thresholds, trade_count)
            if trade_count < min_trades:
                continue

            pnl_series = (
                subset['pnl_per_contract'].fillna(0.0).astype(float)
                * subset['contracts'].fillna(1.0).astype(float)
            ).to_numpy(dtype=float)
            total_pnl = float(np.sum(pnl_series))
            max_drawdown = self._calculate_max_drawdown(pnl_series)

            net_returns = subset['net_return_pct'].fillna(0.0).astype(float)
            mean_net_return = float(net_returns.mean())
            return_std = float(net_returns.std(ddof=0))
            hold_days_mean = float(subset['hold_days'].fillna(7.0).mean())
            annualization = np.sqrt(252.0 / max(hold_days_mean, 1.0))
            sharpe_ratio = float((mean_net_return / return_std) * annualization) if return_std > 0 else 0.0

            labeled = subset.dropna(subset=['realized_front_iv_crush_pct'])
            if labeled.empty:
                crush_mae = None
                crush_directional_accuracy = 0.0
                crush_coverage = 0.0
            else:
                crush_mae = float(
                    (labeled['predicted_front_iv_crush_pct'] - labeled['realized_front_iv_crush_pct']).abs().mean()
                )
                crush_directional_accuracy = float(
                    ((labeled['predicted_front_iv_crush_pct'] < 0) == (labeled['realized_front_iv_crush_pct'] < 0)).mean()
                )
                crush_coverage = float(len(labeled) / trade_count)

            win_rate = float((net_returns > 0).mean())
            alpha_score = self._compute_alpha_score(
                sharpe=sharpe_ratio,
                win_rate=win_rate,
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
                mean_net_return=mean_net_return,
                return_std=return_std,
                crush_confidence=float(subset['crush_confidence'].mean()),
                crush_edge=float(subset['crush_edge_score'].mean()),
                crush_mae=crush_mae,
                crush_coverage=crush_coverage,
                crush_directional_accuracy=crush_directional_accuracy,
            )

            candidate_rows.append({
                'threshold': float(threshold),
                'trade_count': trade_count,
                'win_rate': win_rate,
                'mean_net_return_pct': mean_net_return,
                'net_return_std': return_std,
                'sharpe_ratio': sharpe_ratio,
                'total_pnl': total_pnl,
                'max_drawdown': float(max_drawdown),
                'mean_tx_cost_per_contract': float(subset['transaction_cost_per_contract'].mean()),
                'mean_crush_confidence': float(subset['crush_confidence'].mean()),
                'mean_crush_edge_score': float(subset['crush_edge_score'].mean()),
                'crush_label_coverage': crush_coverage,
                'crush_prediction_mae': crush_mae if crush_mae is not None else np.nan,
                'crush_directional_accuracy': (
                    crush_directional_accuracy if not labeled.empty else np.nan
                ),
                'alpha_score': float(alpha_score),
            })

        if not candidate_rows:
            suggested_min_trades = None
            if max_trade_count_for_thresholds >= 5:
                suggested_min_trades = int(
                    max(5, round(0.67 * float(max_trade_count_for_thresholds)))
                )
            return {
                'recommended': False,
                'reason': 'insufficient_trade_count_for_thresholds',
                'session_id': session_id,
                'signal_name': signal_name,
                'min_confidence': float(min_confidence),
                'min_trades': min_trades,
                'available_trade_count': available_trade_count,
                'max_trade_count_for_thresholds': int(max_trade_count_for_thresholds),
                'suggested_min_trades': suggested_min_trades,
                'candidate_count': 0,
                'best_threshold': None,
                'best_metrics': None,
                'decile_table': decile_table.to_dict(orient='records'),
                'candidate_thresholds': [],
            }

        candidates_df = pd.DataFrame(candidate_rows).sort_values(
            by=['alpha_score', 'sharpe_ratio', 'mean_net_return_pct'],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best = candidates_df.iloc[0]

        return {
            'recommended': True,
            'reason': 'ok',
            'session_id': session_id,
            'signal_name': signal_name,
            'min_confidence': float(np.clip(float(min_confidence), 0.0, 1.0)),
            'min_trades': min_trades,
            'available_trade_count': available_trade_count,
            'max_trade_count_for_thresholds': int(max_trade_count_for_thresholds),
            'suggested_min_trades': None,
            'candidate_count': int(len(candidates_df)),
            'best_threshold': float(best['threshold']),
            'best_metrics': self._normalize_json_value(best.to_dict()),
            'decile_table': decile_table.to_dict(orient='records'),
            'candidate_thresholds': candidates_df.to_dict(orient='records'),
        }

    def _compute_alpha_score(self, sharpe: float, win_rate: float, total_pnl: float,
                           max_drawdown: float, mean_net_return: float,
                           return_std: float, crush_confidence: float = 0.0,
                           crush_edge: float = 0.0, crush_mae: Optional[float] = None,
                           crush_coverage: float = 0.0,
                           crush_directional_accuracy: float = 0.0) -> float:
        """Compute composite alpha score used for sweep/OOS ranking."""
        sharpe_component = float(np.clip(float(sharpe) / 2.0, -1.0, 2.0))
        pnl_drawdown_raw = float(float(total_pnl) / max(abs(float(max_drawdown)), 1.0))
        pnl_drawdown_component = float(
            np.clip(np.sign(pnl_drawdown_raw) * np.log1p(abs(pnl_drawdown_raw)), -2.5, 2.5)
        )
        consistency_raw = (
            float(float(mean_net_return) / max(float(return_std), 1e-6))
            if float(return_std) > 0 else 0.0
        )
        consistency_component = float(np.clip(consistency_raw, -2.0, 2.0))
        crush_conf_component = float(np.clip(crush_confidence, 0.0, 1.0))
        crush_edge_component = float(np.clip(crush_edge / 0.12, 0.0, 1.0))
        crush_coverage_component = float(np.clip(crush_coverage, 0.0, 1.0))
        crush_dir_component = float(np.clip(crush_directional_accuracy, 0.0, 1.0))
        if crush_mae is None or not np.isfinite(float(crush_mae)):
            crush_calibration_component = 0.0
        else:
            crush_calibration_component = float(np.clip(1.0 - (float(crush_mae) / 0.25), 0.0, 1.0))

        alpha_score = (
            0.33 * sharpe_component
            + 0.22 * float(win_rate)
            + 0.18 * pnl_drawdown_component
            + 0.15 * consistency_component
            + 0.05 * crush_conf_component
            + 0.03 * crush_edge_component
            + 0.02 * crush_coverage_component
            + 0.01 * crush_dir_component
            + 0.01 * crush_calibration_component
        )
        return float(alpha_score)

    def _normalize_json_value(self, value: Any) -> Any:
        """Convert numpy/pandas values into JSON-safe Python primitives."""
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, pd.Timestamp):
            return value.strftime('%Y-%m-%d')
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')
        if isinstance(value, list):
            return [self._normalize_json_value(v) for v in value]
        if isinstance(value, tuple):
            return [self._normalize_json_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._normalize_json_value(v) for k, v in value.items()}
        return value

    def _extract_param_mapping(self, row: pd.Series) -> Dict[str, Any]:
        """Extract strategy params from sweep result row."""
        params: Dict[str, Any] = {}
        for col in row.index:
            if col.startswith('param_'):
                params[col[6:]] = self._normalize_json_value(row[col])
        return params

    def _resolve_universe_from_params(self, params: Dict[str, Any]) -> List[str]:
        """Resolve backtest universe from parameters."""
        requested = params.get('universe')
        max_symbols = max(1, int(params.get('max_symbols', 10)))
        if isinstance(requested, str):
            symbols = [s.strip().upper() for s in requested.split(',') if s.strip()]
            return symbols or INSTITUTIONAL_UNIVERSE[:max_symbols]
        if isinstance(requested, list):
            symbols = [str(s).strip().upper() for s in requested if str(s).strip()]
            return symbols or INSTITUTIONAL_UNIVERSE[:max_symbols]
        return INSTITUTIONAL_UNIVERSE[:max_symbols]

    def _infer_feature_date_bounds(self, universe: List[str]) -> Optional[Tuple[datetime, datetime]]:
        """Infer available ml_features date range for the selected universe."""
        if not universe:
            return None
        placeholders = ",".join(["?"] * len(universe))
        query = f"""
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM ml_features
            WHERE symbol IN ({placeholders})
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(query, universe).fetchone()
            if not row or not row[0] or not row[1]:
                return None
            return (
                datetime.strptime(str(row[0]), '%Y-%m-%d'),
                datetime.strptime(str(row[1]), '%Y-%m-%d'),
            )
        except Exception:
            return None

    def run_backtest_parameter_sweep(self, base_params: Dict[str, Any],
                                   parameter_grid: Dict[str, List[Any]],
                                   top_n: Optional[int] = None) -> pd.DataFrame:
        """Run parameter sweep and rank configurations by net risk-adjusted score."""
        try:
            if not parameter_grid:
                return pd.DataFrame()

            normalized_grid: Dict[str, List[Any]] = {}
            for key, values in parameter_grid.items():
                if isinstance(values, (list, tuple)):
                    normalized_grid[key] = list(values)
                else:
                    normalized_grid[key] = [values]
            if not normalized_grid:
                return pd.DataFrame()

            grid_keys = sorted(normalized_grid.keys())
            combinations = list(itertools.product(*[normalized_grid[key] for key in grid_keys]))
            self.logger.info(f"🧪 Running parameter sweep with {len(combinations)} combinations")

            rows: List[Dict[str, Any]] = []
            for combo_index, combo_values in enumerate(combinations, start=1):
                sweep_params = dict(base_params or {})
                combo_mapping = dict(zip(grid_keys, combo_values))
                sweep_params.update(combo_mapping)

                session_id = self.run_calendar_spread_backtest(sweep_params)
                session_df = self.get_backtest_results(session_id)
                if session_df.empty:
                    continue
                session = session_df.iloc[0]
                trades_df = self.get_backtest_trades(session_id)

                mean_net_return = float(trades_df['net_return_pct'].mean()) if not trades_df.empty else 0.0
                return_std = float(trades_df['net_return_pct'].std(ddof=0)) if not trades_df.empty else 0.0
                setup_mean = float(trades_df['setup_score'].mean()) if not trades_df.empty else 0.0
                tx_cost_mean = float(trades_df['transaction_cost_per_contract'].mean()) if not trades_df.empty else 0.0
                crush_confidence_mean = (
                    float(trades_df['crush_confidence'].mean())
                    if (not trades_df.empty and 'crush_confidence' in trades_df.columns)
                    else 0.0
                )
                crush_edge_mean = (
                    float(trades_df['crush_edge_score'].mean())
                    if (not trades_df.empty and 'crush_edge_score' in trades_df.columns)
                    else 0.0
                )
                crush_metrics = self._get_session_crush_prediction_metrics(session_id)

                sharpe = float(session['sharpe_ratio'])
                win_rate = float(session['win_rate'])
                total_pnl = float(session['total_pnl'])
                max_drawdown = float(session['max_drawdown'])

                alpha_score = self._compute_alpha_score(
                    sharpe=sharpe,
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    max_drawdown=max_drawdown,
                    mean_net_return=mean_net_return,
                    return_std=return_std,
                    crush_confidence=max(crush_confidence_mean, float(crush_metrics['mean_crush_confidence'])),
                    crush_edge=max(crush_edge_mean, float(crush_metrics['mean_crush_edge'])),
                    crush_mae=(
                        float(crush_metrics['crush_mae'])
                        if np.isfinite(float(crush_metrics['crush_mae']))
                        else None
                    ),
                    crush_coverage=float(crush_metrics['coverage']),
                    crush_directional_accuracy=(
                        float(crush_metrics['directional_accuracy'])
                        if np.isfinite(float(crush_metrics['directional_accuracy']))
                        else 0.0
                    ),
                )

                result_row: Dict[str, Any] = {
                    'session_id': session_id,
                    'combo_index': combo_index,
                    'alpha_score': float(alpha_score),
                    'total_trades': int(session['total_trades']),
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'calmar_ratio': float(session['calmar_ratio']),
                    'mean_net_return_pct': mean_net_return,
                    'net_return_std': return_std,
                    'mean_setup_score': setup_mean,
                    'mean_tx_cost_per_contract': tx_cost_mean,
                    'mean_crush_confidence': crush_confidence_mean,
                    'mean_crush_edge_score': crush_edge_mean,
                    'crush_label_coverage': float(crush_metrics['coverage']),
                    'crush_prediction_mae': (
                        float(crush_metrics['crush_mae'])
                        if np.isfinite(float(crush_metrics['crush_mae']))
                        else np.nan
                    ),
                    'crush_directional_accuracy': (
                        float(crush_metrics['directional_accuracy'])
                        if np.isfinite(float(crush_metrics['directional_accuracy']))
                        else np.nan
                    ),
                }
                for key in grid_keys:
                    result_row[f'param_{key}'] = combo_mapping[key]
                rows.append(result_row)

            if not rows:
                return pd.DataFrame()

            results = pd.DataFrame(rows).sort_values(
                by=['alpha_score', 'sharpe_ratio', 'total_pnl'],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            if top_n is not None and top_n > 0:
                return results.head(int(top_n)).reset_index(drop=True)
            return results

        except Exception as e:
            self.logger.error(f"❌ Parameter sweep failed: {e}")
            return pd.DataFrame()

    def run_rolling_oos_validation(self, base_params: Dict[str, Any],
                                 parameter_grid: Dict[str, List[Any]],
                                 train_days: int = 252,
                                 test_days: int = 63,
                                 step_days: int = 63,
                                 top_n_train: int = 1) -> pd.DataFrame:
        """
        Run rolling out-of-sample validation:
        - Tune params on each train window via sweep
        - Apply best train params on subsequent test window
        """
        try:
            train_days = max(63, int(train_days))
            test_days = max(21, int(test_days))
            step_days = max(21, int(step_days))
            top_n_train = max(1, int(top_n_train))

            base_params = dict(base_params or {})
            universe = self._resolve_universe_from_params(base_params)
            date_bounds = self._infer_feature_date_bounds(universe)
            if not date_bounds:
                self.logger.warning("⚠️ Unable to determine feature date bounds for OOS validation")
                return pd.DataFrame()

            min_date, max_date = date_bounds
            if base_params.get('start_date'):
                min_date = max(min_date, datetime.strptime(str(base_params['start_date']), '%Y-%m-%d'))
            if base_params.get('end_date'):
                max_date = min(max_date, datetime.strptime(str(base_params['end_date']), '%Y-%m-%d'))
            if min_date >= max_date:
                return pd.DataFrame()

            rows: List[Dict[str, Any]] = []
            split_start = min_date
            split_index = 1

            while split_start + timedelta(days=train_days + test_days) <= max_date:
                train_start = split_start
                train_end = train_start + timedelta(days=train_days - 1)
                test_start = train_end + timedelta(days=1)
                test_end = min(test_start + timedelta(days=test_days - 1), max_date)
                if test_start > test_end:
                    break

                train_params = dict(base_params)
                train_params['start_date'] = train_start.strftime('%Y-%m-%d')
                train_params['end_date'] = train_end.strftime('%Y-%m-%d')
                train_ranked = self.run_backtest_parameter_sweep(
                    base_params=train_params,
                    parameter_grid=parameter_grid,
                    top_n=None,
                )
                if train_ranked.empty:
                    split_start += timedelta(days=step_days)
                    split_index += 1
                    continue

                train_top = train_ranked.head(top_n_train)
                best_train = train_top.iloc[0]
                best_params = self._extract_param_mapping(best_train)

                test_params = dict(base_params)
                test_params.update(best_params)
                test_params['start_date'] = test_start.strftime('%Y-%m-%d')
                test_params['end_date'] = test_end.strftime('%Y-%m-%d')

                test_session_id = self.run_calendar_spread_backtest(test_params)
                test_session_df = self.get_backtest_results(test_session_id)
                if test_session_df.empty:
                    split_start += timedelta(days=step_days)
                    split_index += 1
                    continue
                test_session = test_session_df.iloc[0]
                test_trades = self.get_backtest_trades(test_session_id)
                mean_net_return = float(test_trades['net_return_pct'].mean()) if not test_trades.empty else 0.0
                return_std = float(test_trades['net_return_pct'].std(ddof=0)) if not test_trades.empty else 0.0
                setup_mean = float(test_trades['setup_score'].mean()) if not test_trades.empty else 0.0
                tx_cost_mean = float(test_trades['transaction_cost_per_contract'].mean()) if not test_trades.empty else 0.0
                crush_confidence_mean = (
                    float(test_trades['crush_confidence'].mean())
                    if (not test_trades.empty and 'crush_confidence' in test_trades.columns)
                    else 0.0
                )
                crush_edge_mean = (
                    float(test_trades['crush_edge_score'].mean())
                    if (not test_trades.empty and 'crush_edge_score' in test_trades.columns)
                    else 0.0
                )
                crush_metrics = self._get_session_crush_prediction_metrics(test_session_id)

                test_alpha_score = self._compute_alpha_score(
                    sharpe=float(test_session['sharpe_ratio']),
                    win_rate=float(test_session['win_rate']),
                    total_pnl=float(test_session['total_pnl']),
                    max_drawdown=float(test_session['max_drawdown']),
                    mean_net_return=mean_net_return,
                    return_std=return_std,
                    crush_confidence=max(crush_confidence_mean, float(crush_metrics['mean_crush_confidence'])),
                    crush_edge=max(crush_edge_mean, float(crush_metrics['mean_crush_edge'])),
                    crush_mae=(
                        float(crush_metrics['crush_mae'])
                        if np.isfinite(float(crush_metrics['crush_mae']))
                        else None
                    ),
                    crush_coverage=float(crush_metrics['coverage']),
                    crush_directional_accuracy=(
                        float(crush_metrics['directional_accuracy'])
                        if np.isfinite(float(crush_metrics['directional_accuracy']))
                        else 0.0
                    ),
                )

                result_row: Dict[str, Any] = {
                    'split_index': split_index,
                    'train_start': train_start.strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'),
                    'test_start': test_start.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d'),
                    'train_best_session_id': str(best_train['session_id']),
                    'train_best_alpha_score': float(best_train['alpha_score']),
                    'test_session_id': test_session_id,
                    'test_alpha_score': float(test_alpha_score),
                    'test_total_trades': int(test_session['total_trades']),
                    'test_win_rate': float(test_session['win_rate']),
                    'test_total_pnl': float(test_session['total_pnl']),
                    'test_sharpe_ratio': float(test_session['sharpe_ratio']),
                    'test_max_drawdown': float(test_session['max_drawdown']),
                    'test_calmar_ratio': float(test_session['calmar_ratio']),
                    'test_mean_net_return_pct': mean_net_return,
                    'test_net_return_std': return_std,
                    'test_mean_setup_score': setup_mean,
                    'test_mean_tx_cost_per_contract': tx_cost_mean,
                    'test_mean_crush_confidence': crush_confidence_mean,
                    'test_mean_crush_edge_score': crush_edge_mean,
                    'test_crush_label_coverage': float(crush_metrics['coverage']),
                    'test_crush_prediction_mae': (
                        float(crush_metrics['crush_mae'])
                        if np.isfinite(float(crush_metrics['crush_mae']))
                        else np.nan
                    ),
                    'test_crush_directional_accuracy': (
                        float(crush_metrics['directional_accuracy'])
                        if np.isfinite(float(crush_metrics['directional_accuracy']))
                        else np.nan
                    ),
                }
                for key, value in best_params.items():
                    result_row[f'selected_{key}'] = value
                rows.append(result_row)

                split_start += timedelta(days=step_days)
                split_index += 1

            if not rows:
                return pd.DataFrame()

            return pd.DataFrame(rows).sort_values(by=['split_index']).reset_index(drop=True)

        except Exception as e:
            self.logger.error(f"❌ OOS validation failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def _normal_mean_confidence_interval(values: pd.Series, z_score: float = 1.96) -> Dict[str, Optional[float]]:
        """Compute normal-approximation CI for the sample mean."""
        series = pd.to_numeric(values, errors='coerce').dropna()
        n = int(len(series))
        if n == 0:
            return {'mean': None, 'low': None, 'high': None, 'n': 0}
        mean_val = float(series.mean())
        if n < 2:
            return {'mean': mean_val, 'low': mean_val, 'high': mean_val, 'n': 1}

        std = float(series.std(ddof=1))
        if not np.isfinite(std):
            return {'mean': mean_val, 'low': mean_val, 'high': mean_val, 'n': n}

        half_width = float(z_score * std / np.sqrt(max(n, 1)))
        return {
            'mean': mean_val,
            'low': float(mean_val - half_width),
            'high': float(mean_val + half_width),
            'n': n,
        }

    @staticmethod
    def _wilson_confidence_interval(successes: int, trials: int, z_score: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
        """Wilson score interval for binomial proportion."""
        n = int(max(0, trials))
        if n <= 0:
            return None, None
        s = float(np.clip(successes, 0, n))
        p_hat = s / n
        z2 = float(z_score ** 2)
        denom = 1.0 + z2 / n
        center = (p_hat + z2 / (2.0 * n)) / denom
        spread = (
            z_score
            * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
            / denom
        )
        return float(max(0.0, center - spread)), float(min(1.0, center + spread))

    def build_oos_report_card(self, oos_df: pd.DataFrame, min_splits: int = 8,
                              min_total_test_trades: int = 80,
                              min_trades_per_split: float = 5.0) -> Dict[str, Any]:
        """
        Build institutional report card for walk-forward OOS robustness.
        Includes sample-size gates and confidence intervals.
        """
        min_splits = max(1, int(min_splits))
        min_total_test_trades = max(1, int(min_total_test_trades))
        min_trades_per_split = max(1.0, float(min_trades_per_split))

        if oos_df is None or oos_df.empty:
            return {
                'ready': False,
                'reason': 'no_oos_rows',
                'sample': {
                    'splits': 0,
                    'total_test_trades': 0,
                    'avg_trades_per_split': 0.0,
                },
                'gates': {},
                'metrics': {},
                'verdict': {
                    'overall_pass': False,
                    'grade': 'F',
                    'message': 'No OOS rows available.',
                },
            }

        required_cols = [
            'test_alpha_score',
            'test_sharpe_ratio',
            'test_total_pnl',
            'test_total_trades',
            'test_win_rate',
        ]
        missing = [col for col in required_cols if col not in oos_df.columns]
        if missing:
            return {
                'ready': False,
                'reason': 'missing_columns',
                'missing_columns': missing,
                'sample': {
                    'splits': int(len(oos_df)),
                    'total_test_trades': 0,
                    'avg_trades_per_split': 0.0,
                },
                'gates': {},
                'metrics': {},
                'verdict': {
                    'overall_pass': False,
                    'grade': 'F',
                    'message': f"Missing required OOS columns: {', '.join(missing)}.",
                },
            }

        df = oos_df.copy()
        trades = pd.to_numeric(df['test_total_trades'], errors='coerce').fillna(0.0).clip(lower=0.0)
        split_count = int(len(df))
        total_test_trades = int(np.round(float(trades.sum())))
        avg_trades_per_split = float(trades.mean()) if split_count > 0 else 0.0

        alpha_ci = self._normal_mean_confidence_interval(df['test_alpha_score'])
        sharpe_ci = self._normal_mean_confidence_interval(df['test_sharpe_ratio'])
        pnl_ci = self._normal_mean_confidence_interval(df['test_total_pnl'])

        win_rate = pd.to_numeric(df['test_win_rate'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=1.0)
        expected_wins = int(np.round(float((win_rate * trades).sum())))
        win_rate_mean = (
            float(np.clip(expected_wins / total_test_trades, 0.0, 1.0))
            if total_test_trades > 0 else 0.0
        )
        win_rate_ci_low, win_rate_ci_high = self._wilson_confidence_interval(expected_wins, total_test_trades)

        positive_alpha_rate = float((pd.to_numeric(df['test_alpha_score'], errors='coerce').fillna(0.0) > 0.0).mean())
        positive_pnl_rate = float((pd.to_numeric(df['test_total_pnl'], errors='coerce').fillna(0.0) > 0.0).mean())

        gates = {
            'min_splits': {
                'required': min_splits,
                'actual': split_count,
                'passed': bool(split_count >= min_splits),
            },
            'min_total_test_trades': {
                'required': min_total_test_trades,
                'actual': total_test_trades,
                'passed': bool(total_test_trades >= min_total_test_trades),
            },
            'min_trades_per_split': {
                'required': float(min_trades_per_split),
                'actual': float(avg_trades_per_split),
                'passed': bool(avg_trades_per_split >= min_trades_per_split),
            },
            'alpha_ci_positive': {
                'required': '> 0',
                'actual': alpha_ci.get('low'),
                'passed': bool(alpha_ci.get('low') is not None and float(alpha_ci['low']) > 0.0),
            },
            'sharpe_ci_positive': {
                'required': '> 0',
                'actual': sharpe_ci.get('low'),
                'passed': bool(sharpe_ci.get('low') is not None and float(sharpe_ci['low']) > 0.0),
            },
            'pnl_ci_positive': {
                'required': '> 0',
                'actual': pnl_ci.get('low'),
                'passed': bool(pnl_ci.get('low') is not None and float(pnl_ci['low']) > 0.0),
            },
        }

        overall_pass = all(bool(gate.get('passed')) for gate in gates.values())
        high_confidence_pass = (
            overall_pass
            and split_count >= 12
            and total_test_trades >= 150
            and positive_alpha_rate >= 0.60
            and positive_pnl_rate >= 0.60
        )
        if high_confidence_pass:
            grade = 'A'
            message = "Institutional robustness gate passed with high sample quality."
        elif overall_pass:
            grade = 'B'
            message = "Robustness gate passed; monitor live drift and execution quality."
        elif gates['min_splits']['passed'] and gates['min_total_test_trades']['passed']:
            grade = 'C'
            message = "Coverage is acceptable, but confidence intervals still cross zero."
        else:
            grade = 'D'
            message = "Insufficient OOS evidence. Expand samples before trusting the edge."

        return {
            'ready': True,
            'reason': 'ok',
            'sample': {
                'splits': split_count,
                'total_test_trades': total_test_trades,
                'avg_trades_per_split': avg_trades_per_split,
                'estimated_wins': expected_wins,
            },
            'metrics': {
                'alpha': alpha_ci,
                'sharpe': sharpe_ci,
                'pnl': pnl_ci,
                'win_rate': {
                    'mean': win_rate_mean,
                    'low': win_rate_ci_low,
                    'high': win_rate_ci_high,
                    'n': total_test_trades,
                },
                'positive_alpha_split_rate': positive_alpha_rate,
                'positive_pnl_split_rate': positive_pnl_rate,
            },
            'gates': gates,
            'verdict': {
                'overall_pass': bool(overall_pass),
                'grade': grade,
                'message': message,
            },
        }

    async def train_ml_model_on_historical_spreads(self) -> dict:
        """
        Train ML model on collected historical calendar spread data

        Returns:
            Dictionary with model performance metrics
        """
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            import joblib
            import os

            self.logger.info("🧠 Starting ML model training on calendar spread data")

            # Get training dataset from available data (simulate calendar spread performance)
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT
                        f.symbol,
                        f.date,
                        f.underlying_price as current_price,
                        f.rsi_14,
                        p.macd_signal,
                        f.bb_position,
                        (f.iv_rank * 100.0) as volatility_rank,
                        f.price_momentum_5d as momentum_5d,
                        f.price_momentum_20d as momentum_20d,
                        p.volume,
                        p.realized_vol_30d,
                        p.realized_vol_60d,
                        f.forward_return_21d
                    FROM ml_features f
                    LEFT JOIN daily_prices p ON f.symbol = p.symbol AND f.date = p.date
                    WHERE f.forward_return_21d IS NOT NULL
                    ORDER BY f.symbol, f.date
                """

                df_raw = pd.read_sql_query(query, conn)

                # Deterministic proxy target for calendar spread performance
                df_raw['calendar_pnl'] = self._simulate_calendar_spread_pnl(df_raw)

                # Filter for complete data
                df = df_raw.dropna(subset=['forward_return_21d', 'calendar_pnl'])

            if len(df) < 50:
                self.logger.warning(f"⚠️ Only {len(df)} training samples - need more data for robust ML")
                return None

            self.logger.info(f"📊 Training on {len(df)} feature samples")

            # Feature engineering - select available features for calendar spreads
            feature_columns = [
                'current_price', 'rsi_14', 'macd_signal', 'bb_position',
                'volatility_rank', 'momentum_5d', 'momentum_20d',
                'volume', 'realized_vol_30d', 'realized_vol_60d'
            ]

            # Clean and prepare features
            df_clean = df[feature_columns + ['calendar_pnl', 'forward_return_21d']].dropna()

            if len(df_clean) < 30:
                self.logger.error(f"❌ Only {len(df_clean)} clean samples - insufficient for training")
                return None

            X = df_clean[feature_columns].values
            y_regression = df_clean['calendar_pnl'].values  # For PnL prediction
            y_classification = (df_clean['calendar_pnl'] > 0).astype(int)  # For win/loss prediction

            # Split data
            X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
                X, y_regression, y_classification, test_size=0.3, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train return prediction model (Regression)
            self.logger.info("🎯 Training return prediction model...")
            return_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            return_model.fit(X_train_scaled, y_reg_train)

            # Train direction prediction model (Classification)
            self.logger.info("🎯 Training direction prediction model...")
            direction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            direction_model.fit(X_train_scaled, y_cls_train)

            # Evaluate models
            return_pred = return_model.predict(X_test_scaled)
            direction_pred = direction_model.predict(X_test_scaled)

            # Calculate metrics
            mse = mean_squared_error(y_reg_test, return_pred)
            rmse = np.sqrt(mse)
            accuracy = accuracy_score(y_cls_test, direction_pred)

            # Cross-validation scores
            cv_scores_reg = cross_val_score(return_model, X_train_scaled, y_reg_train, cv=5, scoring='neg_mean_squared_error')
            cv_scores_cls = cross_val_score(direction_model, X_train_scaled, y_cls_train, cv=5, scoring='accuracy')

            # Feature importance
            feature_importance = dict(zip(feature_columns, return_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            # Save models
            model_dir = os.path.expanduser("~/.options_calculator_pro/models")
            os.makedirs(model_dir, exist_ok=True)

            return_model_path = os.path.join(model_dir, "calendar_spread_return_model.pkl")
            direction_model_path = os.path.join(model_dir, "calendar_spread_direction_model.pkl")
            scaler_path = os.path.join(model_dir, "feature_scaler.pkl")

            joblib.dump(return_model, return_model_path)
            joblib.dump(direction_model, direction_model_path)
            joblib.dump(scaler, scaler_path)

            results = {
                'training_samples': len(df_clean),
                'test_samples': len(X_test),
                'return_rmse': rmse,
                'return_cv_score': -cv_scores_reg.mean(),
                'direction_accuracy': accuracy,
                'direction_cv_score': cv_scores_cls.mean(),
                'top_features': top_features,
                'model_paths': {
                    'return_model': return_model_path,
                    'direction_model': direction_model_path,
                    'scaler': scaler_path
                }
            }

            self.logger.info("✅ ML model training completed successfully")
            self.logger.info(f"📊 Return RMSE: {rmse:.4f}")
            self.logger.info(f"🎯 Direction Accuracy: {accuracy:.1%}")
            self.logger.info(f"🔝 Top Features: {[f[0] for f in top_features]}")

            return results

        except Exception as e:
            self.logger.error(f"❌ ML model training failed: {e}")
            return None

    def _simulate_calendar_spread_pnl(self, df) -> pd.Series:
        """
        Deterministic calendar-spread-style P&L proxy for ML target generation.
        """
        volatility_rank = pd.to_numeric(df['volatility_rank'], errors='coerce').fillna(50.0)
        momentum_5d = pd.to_numeric(df['momentum_5d'], errors='coerce').fillna(0.0)
        rsi_14 = pd.to_numeric(df['rsi_14'], errors='coerce').fillna(50.0)
        bb_position = pd.to_numeric(df['bb_position'], errors='coerce').fillna(0.5)
        forward_return_21d = pd.to_numeric(df['forward_return_21d'], errors='coerce').fillna(0.0)
        realized_vol_30d = pd.to_numeric(df['realized_vol_30d'], errors='coerce').fillna(0.22).clip(lower=0.05, upper=2.0)

        vol_component = np.clip((volatility_rank - 45.0) / 55.0, -0.35, 0.70)
        trend_penalty = np.clip(np.abs(momentum_5d) / 0.06, 0.0, 1.5) * 0.22
        rsi_component = 1.0 - np.clip(np.abs(rsi_14 - 50.0) / 35.0, 0.0, 1.0)
        bb_component = 1.0 - np.clip(np.abs(bb_position - 0.5) / 0.5, 0.0, 1.0)
        expected_move_21d = np.clip(realized_vol_30d * np.sqrt(21 / 252), 0.015, 0.25)
        move_ratio = np.abs(forward_return_21d) / expected_move_21d
        stability_component = np.clip(1.0 - move_ratio, -1.0, 1.0)

        return_on_debit = (
            0.14
            + 0.30 * vol_component
            + 0.12 * rsi_component
            + 0.10 * bb_component
            + 0.24 * stability_component
            - trend_penalty
        )
        return_on_debit = np.clip(return_on_debit, -1.0, 1.4)

        debit_per_contract = np.clip(df['current_price'].astype(float) * 0.012, 35.0, 1200.0)
        pnl = np.clip(debit_per_contract * return_on_debit, -250.0, 350.0)
        return pd.Series(pnl, index=df.index)

# Async wrapper for synchronous usage
def run_backfill(db: InstitutionalMLDatabase, symbols: List[str] = None, years: int = 2):
    """Run backfill in sync context"""
    return asyncio.run(db.backfill_historical_data(symbols, years))

if __name__ == "__main__":
    # Example usage
    db = InstitutionalMLDatabase()

    # Backfill data for top 5 institutional stocks
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'TSLA']
    success = run_backfill(db, test_symbols, years=1)

    if success:
        print("✅ Backfill completed successfully")

        # Run sample backtest
        strategy_params = {
            'max_dte': 45,
            'min_dte': 15,
            'iv_rank_min': 30,
            'profit_target': 0.25,
            'stop_loss': 0.50
        }

        session_id = db.run_calendar_spread_backtest(strategy_params)

        # Get results
        results = db.get_backtest_results(session_id)
        print(f"📊 Backtest results:\n{results}")

        # Get training data
        training_data = db.get_training_dataset(test_symbols)
        print(f"🎯 Training dataset: {len(training_data)} samples")

    else:
        print("❌ Backfill failed")
