from __future__ import annotations

import pandas as pd

from services.earnings_vol_snapshot import _normalize_option_chain


def test_last_trade_is_not_promoted_to_mid_without_valid_bid_ask() -> None:
    frame = pd.DataFrame(
        [
            {
                "expiration_date": "2026-05-15",
                "option_type": "C",
                "strike": 100,
                "lastPrice": 4.25,
                "impliedVolatility": 0.45,
            }
        ]
    )

    normalized, source = _normalize_option_chain(frame)

    assert source == "provided"
    assert len(normalized) == 1
    assert pd.isna(normalized.loc[0, "mid"])


def test_valid_bid_ask_still_builds_mid() -> None:
    frame = pd.DataFrame(
        [
            {
                "expiration_date": "2026-05-15",
                "option_type": "P",
                "strike": 100,
                "bid": 3.0,
                "ask": 3.4,
                "impliedVolatility": 0.45,
            }
        ]
    )

    normalized, _source = _normalize_option_chain(frame)

    assert normalized.loc[0, "mid"] == 3.2
