from pathlib import Path

import duckdb
import pytest

from services.options_feature_store import OptionsFeatureStore


def _write_feature_partition(root: Path) -> None:
    partition = root / "underlying_symbol=SPY" / "year=2025" / "month=01"
    partition.mkdir(parents=True)
    output_path = partition / "spy_options_features_eod_2025-01.parquet"
    duckdb.sql(
        f"""
        COPY (
            SELECT *
            FROM (
                VALUES
                    (DATE '2025-01-02', DATE '2025-01-17', 'SPY', 'SPY250117C00600000', 'C', true, false, 15, 'dte_015_019', 10, 100, 600.0, 4.0, 4.2, 4.1, 4.05, 598.0, 0.3, 0.20, 0.51, 0.51, 0.01, -0.02, 0.08, 0.01, 0.2, 0.04878, 'ok', 5.0),
                    (DATE '2025-01-02', DATE '2025-01-17', 'SPY', 'SPY250117P00600000', 'P', false, true, 15, 'dte_015_019', 20, 200, 600.0, 3.8, 4.1, 3.95, 4.00, 598.0, 0.3, 0.22, -0.49, 0.49, 0.01, -0.02, 0.08, 0.01, 0.3, 0.07595, 'ok', 6.0),
                    (DATE '2025-01-03', DATE '2025-02-21', 'SPY', 'SPY250221C00605000', 'C', true, false, 49, 'dte_046_090', 5, 50, 605.0, 5.0, 5.5, 5.25, 5.10, 601.0, 0.6, 0.25, 0.45, 0.45, 0.01, -0.02, 0.09, 0.01, 0.5, 0.09524, 'ok', 3.0)
            ) AS t(
                trade_date, expiry, underlying_symbol, option_symbol, call_put, is_call, is_put,
                dte, dte_bucket, volume, open_interest, strike, bid, ask, mid, last,
                underlying_price, moneyness_pct, iv, delta, abs_delta, gamma, theta, vega,
                rho, spread, spread_pct, quote_quality_flag, liquidity_score
            )
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def test_list_symbols_and_coverage(tmp_path: Path):
    feature_root = tmp_path / "features"
    _write_feature_partition(feature_root)
    store = OptionsFeatureStore(data_root=tmp_path, feature_root=feature_root)

    assert store.list_symbols() == ["SPY"]
    coverage = store.coverage("SPY")
    assert coverage[0]["symbol"] == "SPY"
    assert str(coverage[0]["start_date"]) == "2025-01-02"
    assert str(coverage[0]["end_date"]) == "2025-01-03"
    assert coverage[0]["rows"] == 3


def test_query_chain_filters_by_date_side_and_dte(tmp_path: Path):
    feature_root = tmp_path / "features"
    _write_feature_partition(feature_root)
    store = OptionsFeatureStore(data_root=tmp_path, feature_root=feature_root)

    rows = store.query_chain_records("SPY", trade_date="2025-01-02", call_put="C", min_dte=10, max_dte=20)

    assert len(rows) == 1
    assert rows[0]["option_symbol"] == "SPY250117C00600000"
    assert rows[0]["bid"] == 4.0
    assert rows[0]["quote_quality_flag"] == "ok"


def test_query_chain_rejects_unsafe_symbol(tmp_path: Path):
    store = OptionsFeatureStore(data_root=tmp_path, feature_root=tmp_path / "features")

    with pytest.raises(ValueError):
        store.query_chain_records("SPY;DROP", trade_date="2025-01-02")
