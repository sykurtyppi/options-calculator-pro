import json
from pathlib import Path

import pandas as pd

from scripts.compact_ivol_options_to_parquet import (
    _canonicalize_temporal_columns,
    _load_ivol_underlying_payload,
    _materialize_details_frame,
    _normalize_option_frame,
    _normalize_underlying_frame,
    _scale_nullable_series,
)


def test_scale_nullable_series_preserves_nullability_and_rounds():
    series = pd.Series([1.23456, None, 2.0])
    scaled = _scale_nullable_series(series, 10_000)
    assert list(scaled.astype("object")) == [12346, pd.NA, 20000]


def test_normalize_option_frame_produces_scaled_reversible_columns():
    frame = pd.DataFrame(
        [
            {
                "c_date": "2025-01-15",
                "expiration_date": "2025-02-21",
                "option_symbol": "SPY   250221C00600000",
                "option_id": 123,
                "stocks_id": 627,
                "call_put": "C",
                "dte": 37,
                "price_strike": 600.0,
                "Bid": 4.95,
                "Ask": 5.05,
                "price": 5.0,
                "volume": 10,
                "openinterest": 100,
                "iv": 0.202345,
                "delta": 0.512345,
                "gamma": 0.012345,
                "theta": -0.123456,
                "vega": 0.234567,
                "rho": 0.111111,
                "preiv": 0.202345,
                "underlying_price": 598.12,
                "calc_OTM": 0.31,
                "is_settlement": 0,
            }
        ]
    )
    normalized = _normalize_option_frame(frame, source_path="raw.json")
    row = normalized.iloc[0]
    assert row["underlying_symbol"] == "SPY"
    assert row["strike_10000"] == 6_000_000
    assert row["bid_10000"] == 49_500
    assert row["ask_10000"] == 50_500
    assert row["mid_10000"] == 50_000
    assert row["iv_1000000"] == 202345
    assert row["delta_1000000"] == 512345


def test_normalize_underlying_frame_produces_scaled_price_columns():
    frame = pd.DataFrame(
        [
            {
                "date": "2025-01-15",
                "symbol": "SPY",
                "open": 590.12,
                "high": 592.44,
                "low": 589.77,
                "close": 591.32,
                "adjClose": 591.32,
                "volume": 123456789,
            }
        ]
    )
    normalized = _normalize_underlying_frame(frame, source_path="underlying.json")
    row = normalized.iloc[0]
    assert row["underlying_symbol"] == "SPY"
    assert row["open_10000"] == 5_901_200
    assert row["adjusted_close_10000"] == 5_913_200
    assert row["volume"] == 123456789


def test_load_ivol_underlying_payload_materializes_details(monkeypatch, tmp_path: Path):
    payload_path = tmp_path / "underlying.json"
    payload_path.write_text(
        json.dumps(
            {
                "status": {
                    "recordsFound": 2,
                    "urlForDetails": "https://example.test/details",
                },
                "data": [],
            }
        )
    )

    def fake_materialize(details_url: str, api_key: str, session):
        assert details_url == "https://example.test/details"
        assert api_key == "secret"
        return (
            pd.DataFrame(
                [
                    {"date": "2025-01-02", "symbol": "AAPL", "close": 200.0},
                    {"date": "2025-01-03", "symbol": "AAPL", "close": 201.0},
                ]
            ),
            "csv",
        )

    monkeypatch.setattr(
        "scripts.compact_ivol_options_to_parquet._materialize_details_frame",
        fake_materialize,
    )

    result = _load_ivol_underlying_payload(payload_path, api_key="secret", session=object())
    assert result.deferred_materialized is True
    assert result.records_found == 2
    assert list(result.frame["symbol"]) == ["AAPL", "AAPL"]


def test_materialize_details_frame_follows_url_for_download(monkeypatch):
    class DummyResponse:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    calls: list[str] = []

    class DummySession:
        def get(self, url: str, timeout: int = 60):
            calls.append(url)
            if "data/info" in url:
                return DummyResponse(
                    json.dumps(
                        {
                            "data": [
                                {
                                    "urlForDownload": "https://restapi.ivolatility.com/data/download/abc123"
                                }
                            ]
                        }
                    ).encode("utf-8")
                )
            return DummyResponse(b"date,symbol,close\n2025-01-02,AAPL,200.0\n")

    frame, source_format = _materialize_details_frame(
        "https://restapi.ivolatility.com/data/info/abc123",
        api_key="secret",
        session=DummySession(),
    )
    assert source_format == "csv"
    assert list(frame.columns) == ["date", "symbol", "close"]
    assert len(calls) == 2
    assert "apiKey=secret" in calls[0]
    assert "apiKey=secret" in calls[1]


def test_canonicalize_temporal_columns_normalizes_mixed_types():
    frame = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2025-01-02"), "expiry": pd.Timestamp("2025-02-21")},
            {"trade_date": "2025-01-03", "expiry": "2025-02-22"},
        ]
    )
    normalized = _canonicalize_temporal_columns(frame)
    assert str(normalized["trade_date"].dtype).startswith("datetime64")
    assert str(normalized["expiry"].dtype).startswith("datetime64")
