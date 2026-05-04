from __future__ import annotations

import pandas as pd

from services.option_surface_quality import diagnose_option_surface_quality


def _chain(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["expiration", "side", "strike", "bid", "ask", "mid", "iv"],
    )


def test_empty_chain_is_record_only() -> None:
    result = diagnose_option_surface_quality(pd.DataFrame(), underlying_price=100.0).to_dict()

    assert result["status"] == "record_only"
    assert "missing_option_chain" in result["warning_flags"]


def test_crossed_zero_and_extreme_spreads_are_counted() -> None:
    frame = _chain(
        [
            {"expiration": "2026-05-01", "side": "call", "strike": 100, "bid": 2.0, "ask": 1.0, "mid": 1.5, "iv": 0.4},
            {"expiration": "2026-05-01", "side": "put", "strike": 100, "bid": 0.0, "ask": 1.0, "mid": 0.5, "iv": 0.4},
            {"expiration": "2026-05-01", "side": "call", "strike": 101, "bid": 0.1, "ask": 1.9, "mid": 1.0, "iv": 0.4},
        ]
    )

    result = diagnose_option_surface_quality(frame, underlying_price=100.0).to_dict()

    assert result["status"] == "record_only"
    assert result["crossed_quote_count"] == 1
    assert result["zero_bid_count"] == 1
    assert result["extreme_spread_count"] == 2


def test_sparse_atm_and_missing_iv_are_flagged() -> None:
    rows = []
    for expiration in ["2026-05-01", "2026-05-08"]:
        for side in ["call", "put"]:
            for strike in [70, 130, 140]:
                rows.append(
                    {
                        "expiration": expiration,
                        "side": side,
                        "strike": strike,
                        "bid": 1.0,
                        "ask": 1.2,
                        "mid": 1.1,
                        "iv": None if not rows else 0.4,
                    }
                )
    frame = _chain(rows)

    result = diagnose_option_surface_quality(frame, underlying_price=100.0).to_dict()

    assert result["status"] == "degraded_surface"
    assert result["missing_iv_count"] == 1
    assert result["sparse_atm_expiration_count"] == 2


def test_reasonable_chain_is_clean() -> None:
    rows = []
    for expiration, iv in [("2026-05-01", 0.35), ("2026-05-08", 0.34)]:
        for side in ["call", "put"]:
            for strike in [95, 100, 105]:
                rows.append(
                    {
                        "expiration": expiration,
                        "side": side,
                        "strike": strike,
                        "bid": 1.0,
                        "ask": 1.2,
                        "mid": 1.1,
                        "iv": iv,
                    }
                )
    result = diagnose_option_surface_quality(_chain(rows), underlying_price=100.0).to_dict()

    assert result["status"] == "clean_surface"
    assert result["warning_flags"] == []
