from pathlib import Path

import pytest

from scripts.build_options_feature_layer import (
    _discover_symbols,
    _feature_select_sql,
    _parse_symbols,
    date_literal,
)


def test_date_literal_validates_iso_date():
    assert date_literal("2025-01-15") == "2025-01-15"
    with pytest.raises(ValueError):
        date_literal("01/15/2025")


def test_discover_symbols_from_hive_directories(tmp_path: Path):
    root = tmp_path / "chains_eod"
    (root / "underlying_symbol=SPY").mkdir(parents=True)
    (root / "underlying_symbol=AAPL").mkdir(parents=True)
    (root / "bad").mkdir()

    assert _discover_symbols(root) == ["AAPL", "SPY"]


def test_parse_symbols_rejects_unsafe_values(tmp_path: Path):
    with pytest.raises(ValueError):
        _parse_symbols("SPY,AAPL;DROP", tmp_path)


def test_feature_sql_decodes_fixed_point_columns():
    sql = _feature_select_sql("/tmp/source/*.parquet", start_date="2025-01-01", end_date="2025-01-31")
    assert "bid_10000 / 10000.0 AS bid" in sql
    assert "iv_1000000 / 1000000.0 AS iv" in sql
    assert "spread_pct" in sql
    assert "quote_quality_flag" in sql
    assert "trade_date >= DATE '2025-01-01'" in sql
    assert "trade_date <= DATE '2025-01-31'" in sql
    assert "option_id IS NOT NULL" not in sql
