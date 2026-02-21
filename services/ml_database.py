"""
ML Training Database Service - Historical Data Management
========================================================

Manages historical trading data for machine learning model training:
- SQLite database for persistent storage
- Yahoo Finance data collection
- Feature engineering for options trading
- Trade outcome tracking
- Model training data preparation

Part of Professional Options Calculator v10.0
"""

import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass

from utils.logger import setup_logger as get_logger

logger = get_logger(__name__)

@dataclass
class TradeRecord:
    """Structured trade record for ML training"""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime]
    exit_price: Optional[float]
    trade_type: str  # 'long_call', 'short_put', 'iron_condor', etc.
    strike_price: float
    expiry_date: datetime
    days_to_expiry: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    outcome: str  # 'profitable', 'loss', 'break_even'
    profit_loss: float
    return_pct: float

@dataclass
class MarketFeatures:
    """Market features for ML training"""
    symbol: str
    date: datetime
    price: float
    volume: int
    volatility_30d: float
    volatility_rank: float
    iv_percentile: float
    rsi_14: float
    macd_signal: float
    bollinger_position: float
    vix_level: float
    market_trend: str  # 'bullish', 'bearish', 'sideways'

class MLDatabase:
    """ML Training Database Service"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize ML database"""
        self.logger = logger

        # Database path
        if db_path is None:
            config_dir = os.path.expanduser("~/.options_calculator_pro")
            os.makedirs(config_dir, exist_ok=True)
            db_path = os.path.join(config_dir, "ml_training.db")

        self.db_path = db_path
        self.logger.info(f"ML Database initialized: {db_path}")

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Trade records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_date TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_date TEXT,
                        exit_price REAL,
                        trade_type TEXT NOT NULL,
                        strike_price REAL NOT NULL,
                        expiry_date TEXT NOT NULL,
                        days_to_expiry INTEGER NOT NULL,
                        implied_volatility REAL NOT NULL,
                        delta_value REAL NOT NULL,
                        gamma_value REAL NOT NULL,
                        theta_value REAL NOT NULL,
                        vega_value REAL NOT NULL,
                        outcome TEXT NOT NULL,
                        profit_loss REAL NOT NULL,
                        return_pct REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Market features table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        price REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        volatility_30d REAL NOT NULL,
                        volatility_rank REAL NOT NULL,
                        iv_percentile REAL NOT NULL,
                        rsi_14 REAL NOT NULL,
                        macd_signal REAL NOT NULL,
                        bollinger_position REAL NOT NULL,
                        vix_level REAL NOT NULL,
                        market_trend TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                """)

                # Historical price data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS historical_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open_price REAL NOT NULL,
                        high_price REAL NOT NULL,
                        low_price REAL NOT NULL,
                        close_price REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        adj_close REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                """)

                # Training sessions table for tracking model performance
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS training_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_date TEXT NOT NULL,
                        symbols_trained TEXT NOT NULL,
                        records_used INTEGER NOT NULL,
                        model_accuracy REAL,
                        validation_score REAL,
                        model_path TEXT,
                        features_used TEXT,
                        hyperparameters TEXT,
                        notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                self.logger.info("✅ ML Database tables initialized")

        except Exception as e:
            self.logger.error(f"❌ Error initializing database: {e}")
            raise

    def collect_historical_data(self, symbols: List[str], days_back: int = 365) -> bool:
        """Collect historical price data from Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            self.logger.info(f"🔄 Collecting historical data for {len(symbols)} symbols...")

            for symbol in symbols:
                try:
                    # Fetch data from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d")
                    )

                    if hist_data.empty:
                        self.logger.warning(f"⚠️ No data found for {symbol}")
                        continue

                    # Store in database
                    self._store_historical_prices(symbol, hist_data)

                    # Calculate and store market features
                    features = self._calculate_market_features(symbol, hist_data)
                    self._store_market_features(symbol, features)

                    self.logger.info(f"✅ Collected {len(hist_data)} records for {symbol}")

                except Exception as e:
                    self.logger.error(f"❌ Error collecting data for {symbol}: {e}")
                    continue

            return True

        except Exception as e:
            self.logger.error(f"❌ Error in historical data collection: {e}")
            return False

    def _store_historical_prices(self, symbol: str, hist_data: pd.DataFrame):
        """Store historical price data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for date, row in hist_data.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO historical_prices
                        (symbol, date, open_price, high_price, low_price,
                         close_price, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        date.strftime("%Y-%m-%d"),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row['Adj Close'])
                    ))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Error storing prices for {symbol}: {e}")
            raise

    def _calculate_market_features(self, symbol: str, hist_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and market features"""
        try:
            df = hist_data.copy()

            # Calculate returns
            df['returns'] = df['Close'].pct_change()

            # 30-day volatility
            df['volatility_30d'] = df['returns'].rolling(30).std() * np.sqrt(252)

            # RSI (14-day)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()

            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bollinger_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # Volatility rank (simplified)
            df['vol_rank'] = df['volatility_30d'].rolling(252).rank(pct=True)

            # IV percentile (using volatility as proxy)
            df['iv_percentile'] = df['volatility_30d'].rolling(252).rank(pct=True) * 100

            # Market trend (simple momentum)
            df['price_sma_20'] = df['Close'].rolling(20).mean()
            df['trend'] = np.where(df['Close'] > df['price_sma_20'], 'bullish', 'bearish')
            df.loc[(df['Close'] > df['price_sma_20'] * 0.98) &
                   (df['Close'] < df['price_sma_20'] * 1.02), 'trend'] = 'sideways'

            return df

        except Exception as e:
            self.logger.error(f"Error calculating features for {symbol}: {e}")
            raise

    def _store_market_features(self, symbol: str, features_df: pd.DataFrame):
        """Store calculated market features"""
        try:
            # Get VIX level (simplified - using volatility as proxy)
            vix_level = features_df['volatility_30d'].iloc[-1] * 100 if len(features_df) > 0 else 20.0

            with sqlite3.connect(self.db_path) as conn:
                for date, row in features_df.iterrows():
                    if pd.isna(row['volatility_30d']):  # Skip rows with insufficient data
                        continue

                    conn.execute("""
                        INSERT OR REPLACE INTO market_features
                        (symbol, date, price, volume, volatility_30d, volatility_rank,
                         iv_percentile, rsi_14, macd_signal, bollinger_position,
                         vix_level, market_trend)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        date.strftime("%Y-%m-%d"),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row['volatility_30d']) if not pd.isna(row['volatility_30d']) else 0.2,
                        float(row['vol_rank']) if not pd.isna(row['vol_rank']) else 0.5,
                        float(row['iv_percentile']) if not pd.isna(row['iv_percentile']) else 50.0,
                        float(row['rsi_14']) if not pd.isna(row['rsi_14']) else 50.0,
                        float(row['macd_signal']) if not pd.isna(row['macd_signal']) else 0.0,
                        float(row['bollinger_position']) if not pd.isna(row['bollinger_position']) else 0.5,
                        float(vix_level),
                        str(row['trend'])
                    ))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Error storing features for {symbol}: {e}")
            raise

    def add_trade_record(self, trade: TradeRecord) -> bool:
        """Add a trade record to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trade_records
                    (symbol, entry_date, entry_price, exit_date, exit_price,
                     trade_type, strike_price, expiry_date, days_to_expiry,
                     implied_volatility, delta_value, gamma_value, theta_value, vega_value,
                     outcome, profit_loss, return_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.symbol,
                    trade.entry_date.isoformat(),
                    trade.entry_price,
                    trade.exit_date.isoformat() if trade.exit_date else None,
                    trade.exit_price,
                    trade.trade_type,
                    trade.strike_price,
                    trade.expiry_date.isoformat(),
                    trade.days_to_expiry,
                    trade.implied_volatility,
                    trade.delta,
                    trade.gamma,
                    trade.theta,
                    trade.vega,
                    trade.outcome,
                    trade.profit_loss,
                    trade.return_pct
                ))
                conn.commit()

            self.logger.info(f"✅ Added trade record: {trade.symbol} {trade.trade_type}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error adding trade record: {e}")
            return False

    def get_training_data(self, min_records: int = 100) -> Optional[pd.DataFrame]:
        """Get prepared training data for ML models"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Join trade records with market features
                query = """
                    SELECT
                        t.*,
                        m.volatility_30d, m.volatility_rank, m.iv_percentile,
                        m.rsi_14, m.macd_signal, m.bollinger_position,
                        m.vix_level, m.market_trend
                    FROM trade_records t
                    LEFT JOIN market_features m ON t.symbol = m.symbol
                        AND DATE(t.entry_date) = m.date
                    WHERE t.outcome IS NOT NULL
                    ORDER BY t.entry_date DESC
                """

                df = pd.read_sql_query(query, conn)

                if len(df) < min_records:
                    self.logger.warning(f"⚠️ Only {len(df)} records available, need {min_records}")
                    return None

                # Feature engineering for ML
                df = self._prepare_ml_features(df)

                self.logger.info(f"✅ Prepared {len(df)} training records")
                return df

        except Exception as e:
            self.logger.error(f"❌ Error getting training data: {e}")
            return None

    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML training"""
        try:
            # Create binary outcome (profitable vs not)
            df['is_profitable'] = (df['outcome'] == 'profitable').astype(int)

            # Handle missing values
            df = df.fillna({
                'volatility_30d': df['volatility_30d'].median(),
                'volatility_rank': 0.5,
                'iv_percentile': 50.0,
                'rsi_14': 50.0,
                'macd_signal': 0.0,
                'bollinger_position': 0.5,
                'vix_level': 20.0
            })

            # Encode categorical variables
            df['market_trend_encoded'] = pd.Categorical(df['market_trend']).codes
            df['trade_type_encoded'] = pd.Categorical(df['trade_type']).codes

            return df

        except Exception as e:
            self.logger.error(f"Error preparing ML features: {e}")
            raise

    def generate_sample_data(self, symbols: List[str] = None, num_trades: int = 200) -> bool:
        """Generate sample trade data for testing ML models"""
        try:
            if symbols is None:
                symbols = ["AAPL", "TSLA", "SPY", "QQQ", "NVDA", "MSFT", "GOOGL"]

            self.logger.info(f"🔄 Generating {num_trades} sample trades...")

            # First collect some historical data
            self.collect_historical_data(symbols, days_back=365)

            # Generate sample trades
            np.random.seed(42)  # For reproducible results

            for i in range(num_trades):
                symbol = np.random.choice(symbols)

                # Random trade characteristics
                entry_date = datetime.now() - timedelta(days=np.random.randint(1, 300))
                days_to_expiry = np.random.randint(7, 90)
                expiry_date = entry_date + timedelta(days=days_to_expiry)

                # Realistic options parameters
                strike_price = 100 + np.random.normal(0, 20)
                entry_price = strike_price * (0.01 + np.random.exponential(0.05))
                implied_vol = 0.15 + np.random.exponential(0.15)

                # Greeks (simplified realistic values)
                delta = np.random.uniform(0.2, 0.8)
                gamma = np.random.uniform(0.01, 0.1)
                theta = -np.random.uniform(0.01, 0.1)
                vega = np.random.uniform(0.1, 0.5)

                # Trade outcome (bias toward some profitability)
                is_profitable = np.random.choice([True, False], p=[0.4, 0.6])
                if is_profitable:
                    exit_price = entry_price * np.random.uniform(1.1, 3.0)
                    outcome = 'profitable'
                else:
                    exit_price = entry_price * np.random.uniform(0.1, 0.9)
                    outcome = 'loss'

                profit_loss = exit_price - entry_price
                return_pct = (profit_loss / entry_price) * 100

                trade_types = ['long_call', 'long_put', 'short_call', 'short_put',
                              'iron_condor', 'straddle', 'strangle']
                trade_type = np.random.choice(trade_types)

                # Create trade record
                trade = TradeRecord(
                    symbol=symbol,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=entry_date + timedelta(days=np.random.randint(1, days_to_expiry)),
                    exit_price=exit_price,
                    trade_type=trade_type,
                    strike_price=strike_price,
                    expiry_date=expiry_date,
                    days_to_expiry=days_to_expiry,
                    implied_volatility=implied_vol,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    outcome=outcome,
                    profit_loss=profit_loss,
                    return_pct=return_pct
                )

                self.add_trade_record(trade)

            self.logger.info(f"✅ Generated {num_trades} sample trades")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error generating sample data: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}

                # Trade records count
                cursor = conn.execute("SELECT COUNT(*) FROM trade_records")
                stats['total_trades'] = cursor.fetchone()[0]

                # Profitable trades percentage
                cursor = conn.execute("""
                    SELECT
                        COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) * 100.0 / COUNT(*) as win_rate
                    FROM trade_records
                """)
                result = cursor.fetchone()
                stats['win_rate'] = round(result[0], 2) if result[0] else 0.0

                # Market features count
                cursor = conn.execute("SELECT COUNT(*) FROM market_features")
                stats['market_feature_records'] = cursor.fetchone()[0]

                # Historical prices count
                cursor = conn.execute("SELECT COUNT(*) FROM historical_prices")
                stats['historical_price_records'] = cursor.fetchone()[0]

                # Unique symbols
                cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM trade_records")
                stats['unique_symbols'] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    db = MLDatabase()

    # Generate sample data for testing
    print("Generating sample data...")
    db.generate_sample_data(num_trades=50)

    # Get stats
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")

    # Get training data
    training_data = db.get_training_data()
    if training_data is not None:
        print(f"Training data shape: {training_data.shape}")
        print(f"Columns: {list(training_data.columns)}")