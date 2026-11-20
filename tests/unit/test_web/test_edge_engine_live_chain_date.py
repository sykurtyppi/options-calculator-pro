"""A live provider chain that carries no date must still reach the analysis.

``build_vol_snapshot`` drops every chain row without a valid ``trade_date``.
IB Gateway on a delayed feed stamps neither ``trade_date`` nor ``updated`` (a
delayed quote has no honest per-quote time), so its whole chain was discarded:
all 23 IB-sourced recommendations recorded ``missing_option_chain`` and no
structure could ever be eligible. The edge engine now dates such a chain with
the day it was fetched for, the way the raw-yfinance collector always has.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from services.earnings_vol_snapshot import build_vol_snapshot
from tests.unit.test_web import test_edge_engine_provider_gate as gate
from web.api import edge_engine

AS_OF = date(2026, 11, 19)


def _ib_delayed_chain() -> pd.DataFrame:
    """The frame IBGatewayMarketDataClient returns on a delayed feed: its
    normalized columns, quotes present, and no ``updated`` / ``trade_date``."""
    rows = []
    for expiry, dte in (("2026-11-27", 8), ("2026-12-18", 29)):
        for strike in (175.0, 180.0, 185.0, 190.0):
            for side in ("call", "put"):
                rows.append({
                    "optionSymbol": f"CRM{expiry.replace('-', '')[2:]}{side[0].upper()}{int(strike * 1000):08d}",
                    "underlying": "CRM", "side": side, "strike": strike,
                    "bid": 4.9, "ask": 5.1, "mid": 5.0, "lastPrice": 5.0,
                    "volume": 120.0, "openInterest": 900.0, "impliedVolatility": 0.42,
                    "delta": 0.5 if side == "call" else -0.5, "gamma": 0.02, "theta": -0.1, "vega": 0.2,
                    "dte": dte, "underlyingPrice": 182.34, "inTheMoney": False,
                    "intrinsicValue": 0.0, "extrinsicValue": 5.0, "expiration_date": expiry,
                })
    frame = pd.DataFrame(rows)
    frame.attrs.update({"provider": "ib_gateway", "market_data_type": "delayed"})
    return frame


def _snapshot(frame: pd.DataFrame):
    return build_vol_snapshot("CRM", AS_OF, option_chain_data=frame, earnings_metadata=None,
                              price_data=None, options_provider="ib_gateway")


def test_undated_ib_chain_is_discarded_by_the_snapshot_without_the_fix() -> None:
    # The failure this file exists for, pinned so the reason stays visible.
    snapshot = _snapshot(_ib_delayed_chain())
    assert snapshot.option_source is None
    assert snapshot.null_reasons["option_chain"] == "no_valid_as_of_trade_dates"
    assert snapshot.surface_quality["warning_flags"] == ["missing_option_chain"]


def test_dated_ib_chain_reaches_the_snapshot_with_its_surface() -> None:
    raw = _ib_delayed_chain()
    dated = edge_engine._dated_live_chain(raw, AS_OF)
    assert (dated["trade_date"] == "2026-11-19").all()
    snapshot = _snapshot(dated)
    assert snapshot.option_source is not None
    assert "option_chain" not in snapshot.null_reasons
    assert snapshot.surface_quality["row_count"] == len(raw)
    assert snapshot.surface_quality["expiration_count"] == 2
    assert "missing_option_chain" not in snapshot.surface_quality["warning_flags"]
    assert snapshot.underlying_price == 182.34


def test_dating_copies_and_never_invents_a_quote_time() -> None:
    raw = _ib_delayed_chain()
    dated = edge_engine._dated_live_chain(raw, AS_OF)
    assert dated is not raw and "trade_date" not in raw.columns   # the provider frame is untouched
    assert "updated" not in dated.columns                         # no fabricated per-quote time
    assert dated.attrs.get("market_data_type") == "delayed"       # feed type survives the copy


def test_live_feed_without_ticks_is_dated_but_its_updated_stays_empty() -> None:
    raw = _ib_delayed_chain()
    raw["updated"] = np.nan                                       # live feed, nothing ticked yet
    dated = edge_engine._dated_live_chain(raw, AS_OF)
    assert (dated["trade_date"] == "2026-11-19").all()
    assert dated["updated"].isna().all()


def test_a_chain_that_dates_itself_is_returned_untouched() -> None:
    quoted = _ib_delayed_chain()
    quoted["updated"] = pd.Timestamp("2026-11-19T16:45:00Z").timestamp()   # MarketData.app / IB live
    assert edge_engine._dated_live_chain(quoted, AS_OF) is quoted
    stamped = _ib_delayed_chain()
    stamped["trade_date"] = "2026-11-18"                                   # replay / yfinance collector
    assert edge_engine._dated_live_chain(stamped, AS_OF) is stamped
    reported = _ib_delayed_chain()
    reported["report_date"] = "2026-11-18"                                 # the snapshot's alias for it
    assert edge_engine._dated_live_chain(reported, AS_OF) is reported


def test_nothing_to_date_is_passed_through() -> None:
    assert edge_engine._dated_live_chain(None, AS_OF) is None
    empty = pd.DataFrame()
    assert edge_engine._dated_live_chain(empty, AS_OF) is empty


def test_edge_engine_hands_the_snapshot_a_dated_provider_chain() -> None:
    # End to end through analyze_single_ticker: whatever the provider returned,
    # the frame build_vol_snapshot receives must carry the as-of date.
    hist = pd.DataFrame(
        {"Open": np.linspace(100, 102, 140), "High": np.linspace(101, 103, 140),
         "Low": np.linspace(99, 101, 140), "Close": np.linspace(100.2, 102.2, 140),
         "Volume": np.full(140, 5e6)},
        index=pd.bdate_range("2024-01-02", periods=140),
    )
    ticker = MagicMock()
    ticker.history.side_effect = [hist.iloc[-126:].copy(), hist.copy()]
    ticker.info = {}
    ticker.options = []
    ticker.get_earnings_dates.return_value = pd.DataFrame()

    client = MagicMock()
    client.provider_name = "ib_gateway"
    client.is_available.return_value = True
    client.get_option_chain.return_value = _ib_delayed_chain()
    client.get_earnings.return_value = pd.DataFrame()

    seen = {}

    def fake_snapshot(symbol, as_of_date, **kwargs):
        seen["as_of"] = as_of_date
        seen["frame"] = kwargs["option_chain_data"]
        seen["provider"] = kwargs.get("options_provider")
        return gate._minimal_snapshot()

    ts_tuple = ([4.0, 25.0], [0.26, 0.30], np.nan, np.nan, 0.0, 0.0,
                None, None, None, None, None, 4.0, 25.0)
    profile = {"event_count": 0, "median_move_pct": None, "p90_move_pct": None, "avg_last4_move_pct": None}
    with patch.object(edge_engine.yf, "Ticker", return_value=ticker), \
         patch.object(edge_engine, "build_vol_snapshot", side_effect=fake_snapshot), \
         patch.object(edge_engine, "build_structure_scorecards", return_value=gate._minimal_scorecards()), \
         patch.object(edge_engine, "select_best_structure", return_value=gate._minimal_selector()), \
         patch.object(edge_engine, "_term_structure_from_mda_chain", return_value=ts_tuple), \
         patch.object(edge_engine, "_term_structure_points_yf", return_value=ts_tuple), \
         patch.object(edge_engine, "_get_feature_store", return_value=None), \
         patch.object(edge_engine, "_get_pricing_risk_free_rate", return_value=(0.04, "test")), \
         patch.object(edge_engine, "get_dividend_yield", return_value=(0.0, "fallback_zero")), \
         patch("services.calibration_service.get_calibration") as gc:
        gc.return_value.diagnostics.return_value = {"phase": "observational", "n_observations": 58,
                                                    "min_for_observational": 40, "min_for_fit": 120}
        edge_engine.analyze_single_ticker("CRM", mda_client=client, record_to_ledger=False)

    frame = seen["frame"]
    assert seen["provider"] == "ib_gateway"
    assert len(frame) == len(_ib_delayed_chain())
    assert (pd.to_datetime(frame["trade_date"]).dt.date == seen["as_of"]).all()   # dated, and never in the future
    assert "updated" not in frame.columns
