from pathlib import Path

from scripts.audit_ivol_parquet_store import (
    _aggregate_reports,
    _discover_symbols,
    _quality_grade,
    _resolve_symbols,
    _safe_ratio,
)


def test_safe_ratio_handles_zero_denominator():
    assert _safe_ratio(10, 0) is None


def test_safe_ratio_returns_fraction():
    assert _safe_ratio(25, 100) == 0.25


def test_quality_grade_is_excellent_for_clean_metrics():
    assert _quality_grade(0, 0, 0, 0) == "excellent"


def test_quality_grade_escalates_with_bad_metrics():
    assert _quality_grade(10, 10, 0, 0) == "good"
    assert _quality_grade(500, 10, 0, 0) == "needs_review"
    assert _quality_grade(2000, 10, 0, 0) == "high_risk"


def test_discover_symbols_reads_partition_directories(tmp_path: Path):
    root = tmp_path / "normalized" / "options" / "chains_eod"
    (root / "underlying_symbol=SPY").mkdir(parents=True)
    (root / "underlying_symbol=AAPL").mkdir(parents=True)
    (root / "not_a_symbol").mkdir(parents=True)

    assert _discover_symbols(root) == ["AAPL", "SPY"]


def test_resolve_symbols_prefers_cli_inputs(tmp_path: Path):
    class Args:
        symbol = "spy"
        symbols = "aapl,msft"

    resolved = _resolve_symbols(Args(), tmp_path)
    assert resolved == ["AAPL", "MSFT", "SPY"]


def test_aggregate_reports_combines_metrics():
    reports = [
        {
            "symbol": "SPY",
            "raw": {"option_files": 10, "underlying_files": 1, "options_bytes": 100, "underlyings_bytes": 10, "total_bytes": 110},
            "normalized": {
                "option_partitions": 2,
                "underlying_partitions": 2,
                "options_bytes": 20,
                "underlyings_bytes": 2,
                "total_bytes": 22,
                "option_rows": 1000,
                "underlying_rows": 100,
            },
            "catalog": {
                "option_trade_date_range": {"min": "2024-01-01", "max": "2024-12-31"},
                "option_expiry_range": {"min": "2024-01-19", "max": "2025-03-21"},
                "underlying_trade_date_range": {"min": "2024-01-01", "max": "2024-12-31"},
            },
            "data_quality": {
                "grade": "excellent",
                "duplicate_rows": 0,
                "bid_ask_inversions": 0,
                "zero_bid_positive_ask": 1,
                "null_critical_fields": 0,
            },
        },
        {
            "symbol": "AAPL",
            "raw": {"option_files": 20, "underlying_files": 2, "options_bytes": 200, "underlyings_bytes": 20, "total_bytes": 220},
            "normalized": {
                "option_partitions": 3,
                "underlying_partitions": 3,
                "options_bytes": 40,
                "underlyings_bytes": 4,
                "total_bytes": 44,
                "option_rows": 2000,
                "underlying_rows": 200,
            },
            "catalog": {
                "option_trade_date_range": {"min": "2023-01-01", "max": "2025-01-31"},
                "option_expiry_range": {"min": "2023-01-20", "max": "2025-06-20"},
                "underlying_trade_date_range": {"min": "2023-01-01", "max": "2025-01-31"},
            },
            "data_quality": {
                "grade": "needs_review",
                "duplicate_rows": 2,
                "bid_ask_inversions": 3,
                "zero_bid_positive_ask": 4,
                "null_critical_fields": 5,
            },
        },
    ]

    aggregate = _aggregate_reports(reports)
    assert aggregate["symbol_count"] == 2
    assert aggregate["raw"]["total_bytes"] == 330
    assert aggregate["normalized"]["total_bytes"] == 66
    assert aggregate["normalized"]["option_rows"] == 3000
    assert aggregate["compression"]["normalized_vs_raw_ratio"] == 0.2
    assert aggregate["catalog"]["option_trade_date_range"] == {"min": "2023-01-01", "max": "2025-01-31"}
    assert aggregate["data_quality"]["grade_counts"] == {"excellent": 1, "needs_review": 1}
    assert aggregate["data_quality"]["symbols_requiring_review"] == ["AAPL"]
