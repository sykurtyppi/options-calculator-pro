"""F6: last prints must not be promoted into `mid` on the implied-move path.

_nearest_atm_option_stats and _nearest_common_strike_pair_stats feed
near_term_implied_move_pct (ATM straddle-mid implied move). A strike with no
valid two-sided quote must yield mid=None, never a stale lastPrice.
"""
import numpy as np
import pandas as pd

from web.api.edge_engine import (
    _nearest_atm_option_stats,
    _nearest_common_strike_pair_stats,
)


def _chain(rows):
    return pd.DataFrame(rows)


def test_atm_stats_mid_is_none_without_two_sided_quote():
    # Valid two-sided quote -> mid from bid/ask.
    valid = _nearest_atm_option_stats(
        _chain([{"strike": 100.0, "bid": 1.8, "ask": 2.2, "lastPrice": 9.99,
                 "impliedVolatility": 0.5, "openInterest": 10, "volume": 5}]),
        current_price=100.0,
    )
    assert valid["mid"] == 2.0

    # No market (bid/ask zero) but a last print -> mid must be None, NOT 9.99.
    only_last = _nearest_atm_option_stats(
        _chain([{"strike": 100.0, "bid": 0.0, "ask": 0.0, "lastPrice": 9.99,
                 "impliedVolatility": 0.5, "openInterest": 10, "volume": 5}]),
        current_price=100.0,
    )
    assert only_last["mid"] is None


def test_pair_stats_mids_are_none_without_two_sided_quote():
    calls = _chain([{"strike": 100.0, "bid": 0.0, "ask": 0.0, "lastPrice": 8.0,
                     "impliedVolatility": 0.5, "openInterest": 10, "volume": 5}])
    puts = _chain([{"strike": 100.0, "bid": 0.0, "ask": 0.0, "lastPrice": 7.0,
                    "impliedVolatility": 0.5, "openInterest": 10, "volume": 5}])
    stats = _nearest_common_strike_pair_stats(calls, puts, current_price=100.0)
    assert stats["call_mid"] is None
    assert stats["put_mid"] is None

    # With real two-sided quotes the mids compute normally.
    calls_ok = _chain([{"strike": 100.0, "bid": 1.9, "ask": 2.1, "lastPrice": 8.0,
                        "impliedVolatility": 0.5, "openInterest": 10, "volume": 5}])
    puts_ok = _chain([{"strike": 100.0, "bid": 1.4, "ask": 1.6, "lastPrice": 7.0,
                       "impliedVolatility": 0.5, "openInterest": 10, "volume": 5}])
    stats_ok = _nearest_common_strike_pair_stats(calls_ok, puts_ok, current_price=100.0)
    assert stats_ok["call_mid"] == 2.0
    assert stats_ok["put_mid"] == 1.5
