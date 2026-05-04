"""Read-only access layer for the local options research feature warehouse."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd


SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,15}$")
CALL_PUT_VALUES = {"C", "P"}


class OptionsFeatureStoreError(Exception):
    """Raised when the local feature store cannot satisfy a query."""


def _date_literal(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


def _quote_sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notna(frame), None)
    return [
        {str(key): _jsonable_value(value) for key, value in row.items()}
        for row in clean.to_dict(orient="records")
    ]


def _jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if value.time() == time.min:
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, datetime):
        if value.time() == time.min:
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable_value(item())
        except Exception:
            pass
    return value


class OptionsFeatureStore:
    """DuckDB-backed reader for `/research/options_features_eod` Parquet files."""

    DEFAULT_COLUMNS = [
        "trade_date",
        "expiry",
        "underlying_symbol",
        "option_symbol",
        "call_put",
        "is_call",
        "is_put",
        "dte",
        "dte_bucket",
        "volume",
        "open_interest",
        "strike",
        "bid",
        "ask",
        "mid",
        "last",
        "underlying_price",
        "moneyness_pct",
        "iv",
        "delta",
        "abs_delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "spread",
        "spread_pct",
        "quote_quality_flag",
        "liquidity_score",
    ]

    def __init__(self, data_root: Optional[str | Path] = None, feature_root: Optional[str | Path] = None):
        configured_root = data_root or os.environ.get("MARKET_DATA_ROOT") or "/Volumes/T9/market_data"
        self.data_root = Path(configured_root).expanduser().resolve()
        self.feature_root = (
            Path(feature_root).expanduser().resolve()
            if feature_root
            else self.data_root / "research" / "options_features_eod"
        )

    def is_available(self) -> bool:
        return self.feature_root.exists()

    def list_symbols(self) -> list[str]:
        if not self.feature_root.exists():
            return []
        prefix = "underlying_symbol="
        symbols = []
        for path in sorted(self.feature_root.iterdir()):
            if path.is_dir() and path.name.startswith(prefix):
                symbol = path.name[len(prefix) :].upper()
                if self._is_safe_symbol(symbol):
                    symbols.append(symbol)
        return symbols

    def coverage(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        source = self._source_glob(symbol)
        where = ""
        if symbol:
            safe_symbol = self._normalize_symbol(symbol)
            where = f"WHERE underlying_symbol = {_quote_sql_literal(safe_symbol)}"

        query = f"""
            SELECT
                underlying_symbol AS symbol,
                MIN(trade_date) AS start_date,
                MAX(trade_date) AS end_date,
                COUNT(*)::BIGINT AS rows,
                COUNT(DISTINCT expiry)::BIGINT AS expiries,
                COUNT(DISTINCT option_symbol)::BIGINT AS option_symbols
            FROM read_parquet({_quote_sql_literal(source)}, hive_partitioning=true)
            {where}
            GROUP BY underlying_symbol
            ORDER BY underlying_symbol
        """
        with duckdb.connect(database=":memory:") as con:
            frame = con.sql(query).df()
        return _records_from_frame(frame)

    def query_chain(
        self,
        symbol: str,
        *,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        expiry: Optional[str] = None,
        min_dte: Optional[int] = None,
        max_dte: Optional[int] = None,
        call_put: Optional[str] = None,
        min_abs_delta: Optional[float] = None,
        max_abs_delta: Optional[float] = None,
        limit: int = 1_000,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        safe_symbol = self._normalize_symbol(symbol)
        self._validate_symbol_exists(safe_symbol)

        selected_columns = self._select_columns(columns)
        limit = max(1, min(int(limit), 10_000))
        clauses = [f"underlying_symbol = {_quote_sql_literal(safe_symbol)}"]

        if trade_date:
            parsed = _date_literal(trade_date)
            clauses.append(f"trade_date = DATE {_quote_sql_literal(parsed)}")
        else:
            if start_date:
                clauses.append(f"trade_date >= DATE {_quote_sql_literal(_date_literal(start_date))}")
            if end_date:
                clauses.append(f"trade_date <= DATE {_quote_sql_literal(_date_literal(end_date))}")

        if expiry:
            clauses.append(f"expiry = DATE {_quote_sql_literal(_date_literal(expiry))}")
        if min_dte is not None:
            clauses.append(f"dte >= {int(min_dte)}")
        if max_dte is not None:
            clauses.append(f"dte <= {int(max_dte)}")
        if call_put:
            normalized_call_put = call_put.strip().upper()
            if normalized_call_put not in CALL_PUT_VALUES:
                raise ValueError("call_put must be 'C' or 'P'")
            clauses.append(f"call_put = {_quote_sql_literal(normalized_call_put)}")
        if min_abs_delta is not None:
            clauses.append(f"abs_delta >= {float(min_abs_delta)}")
        if max_abs_delta is not None:
            clauses.append(f"abs_delta <= {float(max_abs_delta)}")

        query = f"""
            SELECT {", ".join(selected_columns)}
            FROM read_parquet({_quote_sql_literal(self._source_glob(safe_symbol))}, hive_partitioning=true)
            WHERE {" AND ".join(clauses)}
            ORDER BY trade_date, expiry, call_put, strike
            LIMIT {limit}
        """
        with duckdb.connect(database=":memory:") as con:
            return con.sql(query).df()

    def query_chain_records(self, symbol: str, **kwargs: Any) -> list[dict[str, Any]]:
        return _records_from_frame(self.query_chain(symbol, **kwargs))

    def _source_glob(self, symbol: Optional[str] = None) -> str:
        if not self.feature_root.exists():
            raise OptionsFeatureStoreError(f"Feature store does not exist: {self.feature_root}")
        if symbol:
            safe_symbol = self._normalize_symbol(symbol)
            return str(self.feature_root / f"underlying_symbol={safe_symbol}" / "**" / "*.parquet")
        return str(self.feature_root / "**" / "*.parquet")

    def _validate_symbol_exists(self, symbol: str) -> None:
        symbol_dir = self.feature_root / f"underlying_symbol={symbol}"
        if not symbol_dir.exists():
            raise OptionsFeatureStoreError(f"No feature partitions found for symbol {symbol}")

    @staticmethod
    def _is_safe_symbol(symbol: str) -> bool:
        return bool(SYMBOL_RE.match(symbol))

    @classmethod
    def _normalize_symbol(cls, symbol: str) -> str:
        safe_symbol = symbol.strip().upper()
        if not cls._is_safe_symbol(safe_symbol):
            raise ValueError(f"Unsafe symbol value: {symbol!r}")
        return safe_symbol

    def _select_columns(self, columns: Optional[list[str]]) -> list[str]:
        if not columns:
            return list(self.DEFAULT_COLUMNS)
        allowed = set(self.DEFAULT_COLUMNS) | {"option_id", "stock_id", "vendor", "source_path", "ingested_at"}
        selected = []
        for column in columns:
            normalized = column.strip()
            if normalized not in allowed:
                raise ValueError(f"Unsupported feature column: {column!r}")
            selected.append(normalized)
        return selected or list(self.DEFAULT_COLUMNS)
