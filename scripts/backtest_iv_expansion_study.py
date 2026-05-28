#!/usr/bin/env python3
"""Historical validation study for the pre-earnings IV expansion mode.

This script audits the new "IV Expansion Candidate" path with mechanically
exact historical simulations:

1. Load historical earnings events from the local research database.
2. Build contemporaneous signal snapshots from local option-chain and OHLCV data.
3. Simulate exact-contract entry/exit for multiple pre-earnings structures.
4. Track walk-forward historical expectations without future leakage.
5. Summarize P&L under multiple execution-cost assumptions.
6. Emit trade-level CSV, JSON summaries, and a markdown research memo.

The implementation never substitutes missing contracts. When an exact exit
quote is unavailable, the trade is flagged and excluded from realized P&L.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.earnings_vol_snapshot import _event_contaminated_session_dates
from services.event_vol_decomposition import decompose_event_vol
from services.iv_term_structure import bounded_interp
from services.realized_vol import rs_trailing_mean_forecast as _rs_trailing_mean_forecast
from utils.logger import setup_logger
from web.api.edge_engine import (
    MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION,
    MIN_RV_PERCENTILE_FOR_EXPANSION,
    MIN_TS_SLOPE_FOR_EXPANSION,
    _classify_move_risk,
    _compute_move_anchor,
    _compute_move_uncertainty_pct,
    _historical_earnings_move_profile,
    _har_rv_forecast,
    _normalize_release_timing,
    _rs_daily_vol_series,
    _rv_percentile_and_regime,
    _yang_zhang_rv30,
)

# Legacy expansion floor on IV/RV. The production constant
# MIN_IV_RV_FOR_EXPANSION was removed from web.api.edge_engine because it
# encoded the wrong direction — low IV/RV is favourable (cheap implied vol)
# for long-vol entry, not unfavourable. Removal is enforced by
# tests/unit/test_fixes/test_four_fixes.py::test_min_iv_rv_for_expansion_constant_removed.
# The production gate now uses IV_RV_CROWDING_WARNING_THRESHOLD = 1.60 as a
# soft ceiling instead.
#
# This script retains the legacy floor purely for replay continuity. Default
# 0.0 makes the gate a no-op (iv_rv < 0.0 is never True for valid input) and
# anchors the linear score component at zero. Operators replaying against an
# older scoreboard with a non-zero floor must override via --signal-min-iv-rv.
MIN_IV_RV_FOR_EXPANSION: float = 0.0


DEFAULT_DB_PATH = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
DEFAULT_OPTIONS_ROOT = Path("/Volumes/T9/market_data/research/options_features_eod")
DEFAULT_PRICE_ROOT = Path("/Volumes/T9/market_data/normalized/underlyings/daily_ohlcv")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "exports" / "reports"

STATUS_REALIZED = "REALIZED"
STATUS_SIGNAL_REJECTED = "SIGNAL_REJECTED"
STATUS_UNSAFE_EXIT = "UNSAFE_EXIT"
STATUS_MISSING_ENTRY_CHAIN = "MISSING_ENTRY_CHAIN"
STATUS_MISSING_EXIT_CHAIN = "MISSING_EXIT_CHAIN"
STATUS_MISSING_ENTRY_LEG = "MISSING_ENTRY_LEG"
STATUS_MISSING_EXIT_QUOTE = "MISSING_EXIT_QUOTE"
STATUS_NEGATIVE_DEBIT = "NEGATIVE_DEBIT"
STATUS_STRUCTURAL_INVALID = "STRUCTURAL_INVALID"
STATUS_MISSING_PRICE_HISTORY = "MISSING_PRICE_HISTORY"
STATUS_ENTRY_DATE_UNAVAILABLE = "ENTRY_DATE_UNAVAILABLE"
STATUS_EXIT_DATE_UNAVAILABLE = "EXIT_DATE_UNAVAILABLE"

BASE_ENTRY_OFFSET = 7
BASE_EXIT_OFFSET = 1
BASE_STRUCTURE_ROWS = ("call_calendar", "put_calendar", "atm_straddle", "otm_strangle")
ROBUST_ENTRY_OFFSETS = (3, 5, 7, 10)
ROBUST_EXIT_OFFSETS = (2, 1, 0)
EXECUTION_SCENARIOS = (
    ("mid", 0.0, 0.0),
    ("cross_25", 0.25, 0.0),
    ("cross_50", 0.50, 0.0),
    ("cross_100", 1.00, 0.0),
    ("cross_100_commission", 1.00, 0.65),
)
MAX_EVENTS_PER_SYMBOL_HISTORY = 24


class StudyInvariantError(RuntimeError):
    """Raised when the study would violate contract-identity invariants."""


@dataclass(frozen=True)
class ExecutionScenario:
    name: str
    half_spread_fraction: float
    commission_per_contract: float


@dataclass(frozen=True)
class StructureSpec:
    name: str
    label: str
    long_legs: Tuple[str, ...]
    short_legs: Tuple[str, ...]
    call_put: Optional[str] = None
    same_expiry: bool = False
    same_strike: bool = False
    back_gap_days: int = 14
    otm_target_pct: Optional[float] = None


@dataclass(frozen=True)
class ContractIdentity:
    symbol: str
    option_type: str
    strike: float
    expiry: str
    option_symbol: Optional[str]
    option_id: Optional[int]


@dataclass(frozen=True)
class LegSnapshot:
    identity: ContractIdentity
    side: int
    quote_date: str
    bid: float
    ask: float
    mid: float
    spread: float
    spread_pct: float
    iv: Optional[float]
    volume: int
    open_interest: int
    liquidity_score: float
    moneyness_pct: float
    underlying_price: float


@dataclass
class TradeRecord:
    symbol: str
    event_date: str
    release_timing: str
    structure: str
    entry_offset_bdays: int
    exit_offset_bdays: int
    status: str
    exclusion_reason: str
    candidate_signal_pass: bool
    entry_date: Optional[str]
    exit_date: Optional[str]
    accepted_for_entry: bool
    exact_exit_quote_found: bool
    priceable_realized: bool
    wf_history_count: int
    wf_expected_iv_change: Optional[float]
    wf_expected_pnl_mid: Optional[float]
    wf_expected_pnl_adjusted: Optional[float]
    wf_ranking_score: Optional[float]
    signal_iv_rv30: Optional[float]
    signal_iv_rv_har: Optional[float]
    signal_term_structure_slope: Optional[float]
    signal_near_back_iv_ratio: Optional[float]
    signal_rv30: Optional[float]
    signal_rv_har_forecast: Optional[float]
    signal_rv_percentile_rank: Optional[float]
    signal_vol_regime: Optional[str]
    signal_near_term_dte: Optional[int]
    signal_near_term_spread_pct: Optional[float]
    signal_near_term_liquidity_proxy: Optional[float]
    signal_event_implied_move_pct: Optional[float]
    signal_non_event_move_pct: Optional[float]
    signal_implied_move_pct: Optional[float]
    signal_move_anchor_pct: Optional[float]
    signal_historical_p90_move_pct: Optional[float]
    signal_move_risk_level: Optional[str]
    signal_move_risk_ratio: Optional[float]
    signal_smile_curvature: Optional[float]
    signal_smile_points: Optional[int]
    signal_front_oi: Optional[int]
    signal_back_oi: Optional[int]
    signal_fail_reasons: str
    leg1_option_type: Optional[str]
    leg1_expiry: Optional[str]
    leg1_strike: Optional[float]
    leg1_option_symbol: Optional[str]
    leg1_option_id: Optional[int]
    leg1_side: Optional[int]
    leg1_bid_entry: Optional[float]
    leg1_ask_entry: Optional[float]
    leg1_mid_entry: Optional[float]
    leg1_bid_exit: Optional[float]
    leg1_ask_exit: Optional[float]
    leg1_mid_exit: Optional[float]
    leg1_iv_entry: Optional[float]
    leg1_iv_exit: Optional[float]
    leg1_iv_change: Optional[float]
    leg1_spread_entry: Optional[float]
    leg1_spread_exit: Optional[float]
    leg1_open_interest_entry: Optional[int]
    leg1_volume_entry: Optional[int]
    leg2_option_type: Optional[str]
    leg2_expiry: Optional[str]
    leg2_strike: Optional[float]
    leg2_option_symbol: Optional[str]
    leg2_option_id: Optional[int]
    leg2_side: Optional[int]
    leg2_bid_entry: Optional[float]
    leg2_ask_entry: Optional[float]
    leg2_mid_entry: Optional[float]
    leg2_bid_exit: Optional[float]
    leg2_ask_exit: Optional[float]
    leg2_mid_exit: Optional[float]
    leg2_iv_entry: Optional[float]
    leg2_iv_exit: Optional[float]
    leg2_iv_change: Optional[float]
    leg2_spread_entry: Optional[float]
    leg2_spread_exit: Optional[float]
    leg2_open_interest_entry: Optional[int]
    leg2_volume_entry: Optional[int]
    avg_iv_change: Optional[float]
    entry_value_mid: Optional[float]
    exit_value_mid: Optional[float]
    mid_pnl_per_spread: Optional[float]
    debit_capital_per_spread: Optional[float]
    return_mid_pct: Optional[float]
    pnl_mid: Optional[float]
    entry_half_spread_cost: Optional[float]
    exit_half_spread_cost: Optional[float]
    pnl_cross_25: Optional[float]
    pnl_cross_50: Optional[float]
    pnl_cross_100: Optional[float]
    pnl_cross_100_commission: Optional[float]
    return_cross_25_pct: Optional[float]
    return_cross_50_pct: Optional[float]
    return_cross_100_pct: Optional[float]
    return_cross_100_commission_pct: Optional[float]


@dataclass
class StructureTrade:
    entry_legs: List[LegSnapshot]
    exit_legs: List[LegSnapshot]
    entry_value_mid: float
    exit_value_mid: float
    entry_half_spread_cost: float
    exit_half_spread_cost: float
    debit_capital_per_spread: float
    avg_iv_change: Optional[float]


@dataclass(frozen=True)
class HistoryOutcome:
    avg_iv_change: Optional[float]
    pnl_mid: Optional[float]
    pnl_cross_100: Optional[float]
    return_mid_pct: Optional[float]
    return_cross_100_pct: Optional[float]


@dataclass(frozen=True)
class StudyConfig:
    db_path: Path
    options_root: Path
    price_root: Path
    output_root: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    quality_tiers: Tuple[str, ...] = ("A", "B")
    entry_offsets: Tuple[int, ...] = ROBUST_ENTRY_OFFSETS
    exit_offsets: Tuple[int, ...] = ROBUST_EXIT_OFFSETS
    min_history_count: int = 3
    signal_min_iv_rv: float = MIN_IV_RV_FOR_EXPANSION
    signal_min_slope: float = MIN_TS_SLOPE_FOR_EXPANSION
    signal_min_rv_percentile: float = MIN_RV_PERCENTILE_FOR_EXPANSION
    signal_max_near_term_spread_pct: float = 12.0
    signal_min_near_term_liquidity: float = MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION
    base_min_leg_open_interest: int = 25
    base_max_leg_spread_pct: float = 25.0
    strict_min_leg_open_interest: int = 100
    strict_max_leg_spread_pct: float = 10.0
    otm_target_pct: float = 3.0
    contract_multiplier: int = 100
    exit_same_day_amc_only: bool = True


STRUCTURES: Dict[str, StructureSpec] = {
    "call_calendar": StructureSpec(
        name="call_calendar",
        label="Same-Strike ATM Call Calendar",
        long_legs=("back_call",),
        short_legs=("front_call",),
        call_put="C",
        same_strike=True,
    ),
    "put_calendar": StructureSpec(
        name="put_calendar",
        label="Same-Strike ATM Put Calendar",
        long_legs=("back_put",),
        short_legs=("front_put",),
        call_put="P",
        same_strike=True,
    ),
    "atm_straddle": StructureSpec(
        name="atm_straddle",
        label="ATM Straddle",
        long_legs=("front_call", "front_put"),
        short_legs=(),
        same_expiry=True,
        same_strike=True,
    ),
    "otm_strangle": StructureSpec(
        name="otm_strangle",
        label="Slightly OTM Strangle",
        long_legs=("front_call", "front_put"),
        short_legs=(),
        same_expiry=True,
        same_strike=False,
        otm_target_pct=3.0,
    ),
}

SCENARIOS: Dict[str, ExecutionScenario] = {
    name: ExecutionScenario(name=name, half_spread_fraction=spread_fraction, commission_per_contract=commission)
    for name, spread_fraction, commission in EXECUTION_SCENARIOS
}


def build_logger(verbose: bool) -> logging.Logger:
    level = "DEBUG" if verbose else "INFO"
    return setup_logger("scripts.backtest_iv_expansion_study", level=level, file_output=False)


def connect_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def parse_date(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)) and math.isnan(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def pct_or_none(value: Optional[float]) -> Optional[float]:
    return float(value) if value is not None and np.isfinite(value) else None


def load_events(config: StudyConfig, logger: logging.Logger) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in config.quality_tiers)
    sql = f"""
        SELECT
            symbol,
            event_date,
            release_timing,
            pre_capture_date,
            post_capture_date,
            pre_front_iv,
            pre_back_iv,
            front_expiry,
            back_expiry,
            quality_tier
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ({placeholders})
          AND event_date >= ?
          AND event_date <= ?
        ORDER BY event_date, symbol
    """
    with connect_readonly_sqlite(config.db_path) as conn:
        frame = pd.read_sql(
            sql,
            conn,
            params=[*config.quality_tiers, config.start_date.strftime("%Y-%m-%d"), config.end_date.strftime("%Y-%m-%d")],
        )
    for col in ("event_date", "pre_capture_date", "post_capture_date", "front_expiry", "back_expiry"):
        frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.normalize()
    frame["release_timing"] = frame["release_timing"].map(_normalize_release_timing)
    frame["nbr_ratio"] = frame["pre_front_iv"] / frame["pre_back_iv"]
    frame = frame.dropna(subset=["symbol", "event_date"]).reset_index(drop=True)
    logger.info("Loaded %d earnings events across %d symbols", len(frame), frame["symbol"].nunique())
    return frame


def build_options_month_path(options_root: Path, symbol: str, dt: pd.Timestamp) -> Path:
    return (
        options_root
        / f"underlying_symbol={symbol}"
        / f"year={dt.year}"
        / f"month={dt.month:02d}"
        / f"{symbol.lower()}_options_features_eod_{dt.year}-{dt.month:02d}.parquet"
    )


def build_price_glob(price_root: Path, symbol: str) -> str:
    return str(price_root / f"underlying_symbol={symbol}" / "**" / "*.parquet")


class DataCache:
    def __init__(self, config: StudyConfig):
        self.config = config
        self.options_month_cache: Dict[Tuple[str, int, int], Optional[pd.DataFrame]] = {}
        self.day_chain_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
        self.price_cache: Dict[str, Optional[pd.DataFrame]] = {}

    def load_day_chain(self, symbol: str, dt: pd.Timestamp) -> Optional[pd.DataFrame]:
        key = (symbol, dt.strftime("%Y-%m-%d"))
        if key in self.day_chain_cache:
            return self.day_chain_cache[key]

        month_key = (symbol, dt.year, dt.month)
        if month_key not in self.options_month_cache:
            path = build_options_month_path(self.config.options_root, symbol, dt)
            if not path.exists():
                self.options_month_cache[month_key] = None
            else:
                frame = pd.read_parquet(path)
                frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
                frame["expiry"] = pd.to_datetime(frame["expiry"]).dt.normalize()
                self.options_month_cache[month_key] = frame
        month_frame = self.options_month_cache[month_key]
        if month_frame is None:
            self.day_chain_cache[key] = None
            return None

        day = month_frame[
            (month_frame["trade_date"] == dt.normalize())
            & (month_frame["quote_quality_flag"] == "ok")
            & (month_frame["bid"] >= 0)
            & (month_frame["ask"] >= month_frame["bid"])
        ].copy()
        if day.empty:
            self.day_chain_cache[key] = None
        else:
            for col in ("strike", "bid", "ask", "mid", "iv", "spread", "spread_pct", "liquidity_score", "underlying_price", "moneyness_pct"):
                if col in day.columns:
                    day[col] = pd.to_numeric(day[col], errors="coerce")
            self.day_chain_cache[key] = day
        return self.day_chain_cache[key]

    def load_price_history(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        glob_path = build_price_glob(self.config.price_root, symbol)
        source_dir = self.config.price_root / f"underlying_symbol={symbol}"
        if not source_dir.exists():
            self.price_cache[symbol] = None
            return None
        sql = f"""
            SELECT
                trade_date,
                open_10000 / 10000.0 AS open,
                high_10000 / 10000.0 AS high,
                low_10000 / 10000.0 AS low,
                close_10000 / 10000.0 AS close,
                volume
            FROM read_parquet('{glob_path}', hive_partitioning=true)
            ORDER BY trade_date
        """
        with duckdb.connect(database=":memory:") as conn:
            frame = conn.sql(sql).df()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        frame = frame.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)
        self.price_cache[symbol] = frame if not frame.empty else None
        return self.price_cache[symbol]


def build_contract_identity(row: pd.Series, symbol: str) -> ContractIdentity:
    option_id = None if pd.isna(row.get("option_id")) else int(row["option_id"])
    option_symbol = row.get("option_symbol")
    return ContractIdentity(
        symbol=symbol,
        option_type=str(row["call_put"]),
        strike=float(row["strike"]),
        expiry=pd.Timestamp(row["expiry"]).strftime("%Y-%m-%d"),
        option_symbol=str(option_symbol) if option_symbol is not None else None,
        option_id=option_id,
    )


def build_leg_snapshot(row: pd.Series, symbol: str, quote_date: pd.Timestamp, side: int) -> LegSnapshot:
    identity = build_contract_identity(row, symbol)
    return LegSnapshot(
        identity=identity,
        side=int(side),
        quote_date=quote_date.strftime("%Y-%m-%d"),
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        mid=float(row["mid"]),
        spread=float(row["spread"]),
        spread_pct=float(row["spread_pct"]),
        iv=float(row["iv"]) if pd.notna(row.get("iv")) else None,
        volume=int(row.get("volume", 0) or 0),
        open_interest=int(row.get("open_interest", 0) or 0),
        liquidity_score=float(row.get("liquidity_score", 0.0) or 0.0),
        moneyness_pct=float(row.get("moneyness_pct", np.nan)),
        underlying_price=float(row["underlying_price"]),
    )


def exact_lookup(chain: pd.DataFrame, identity: ContractIdentity) -> Optional[pd.Series]:
    matches = chain[
        (chain["call_put"] == identity.option_type)
        & (chain["strike"] == identity.strike)
        & (chain["expiry"] == pd.Timestamp(identity.expiry))
    ].copy()
    if matches.empty:
        return None
    if len(matches) == 1:
        return matches.iloc[0]
    if identity.option_id is not None and "option_id" in matches.columns:
        same_id = matches[matches["option_id"] == identity.option_id]
        if len(same_id) == 1:
            return same_id.iloc[0]
        if len(same_id) > 1:
            raise StudyInvariantError(f"Ambiguous option_id match for {identity.symbol} {identity.expiry} {identity.strike}")
    if identity.option_symbol is not None and "option_symbol" in matches.columns:
        same_symbol = matches[matches["option_symbol"] == identity.option_symbol]
        if len(same_symbol) == 1:
            return same_symbol.iloc[0]
        if len(same_symbol) > 1:
            raise StudyInvariantError(f"Ambiguous option_symbol match for {identity.symbol} {identity.expiry} {identity.strike}")
    raise StudyInvariantError(f"Ambiguous exact lookup for {identity.symbol} {identity.expiry} {identity.strike}")


def previous_business_day(price_history: pd.DataFrame, event_date: pd.Timestamp, count: int) -> Optional[pd.Timestamp]:
    sessions = pd.DatetimeIndex(price_history["trade_date"])
    pos = int(sessions.searchsorted(event_date.to_datetime64(), side="left"))
    idx = pos - count
    if idx < 0 or idx >= len(sessions):
        return None
    return pd.Timestamp(sessions[idx]).normalize()


def select_common_strike_pair(front: pd.DataFrame, back: pd.DataFrame, price: float) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    common = sorted(set(front["strike"].dropna().tolist()) & set(back["strike"].dropna().tolist()))
    if not common:
        return None, None
    strike = min(common, key=lambda s: abs(float(s) - price))
    front_row = front.loc[(front["strike"] == strike), :].copy()
    back_row = back.loc[(back["strike"] == strike), :].copy()
    return front_row.sort_values("spread_pct").iloc[0], back_row.sort_values("spread_pct").iloc[0]


def pick_front_expiry(chain: pd.DataFrame, event_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    expiries = sorted({pd.Timestamp(v).normalize() for v in chain["expiry"].dropna().tolist()})
    eligible = [exp for exp in expiries if exp > event_date]
    return eligible[0] if eligible else None


def pick_back_expiry(chain: pd.DataFrame, front_expiry: pd.Timestamp, gap_days: int) -> Optional[pd.Timestamp]:
    expiries = sorted({pd.Timestamp(v).normalize() for v in chain["expiry"].dropna().tolist()})
    eligible = [exp for exp in expiries if exp > front_expiry and (exp - front_expiry).days >= gap_days]
    if eligible:
        return eligible[0]
    later = [exp for exp in expiries if exp > front_expiry]
    return later[0] if later else None


def best_row(frame: pd.DataFrame, ascending_moneyness: bool = True) -> Optional[pd.Series]:
    if frame.empty:
        return None
    ranked = frame.copy()
    ranked["_liq"] = pd.to_numeric(ranked["liquidity_score"], errors="coerce").fillna(0.0)
    ranked["_oi"] = pd.to_numeric(ranked["open_interest"], errors="coerce").fillna(0.0)
    ranked["_vol"] = pd.to_numeric(ranked["volume"], errors="coerce").fillna(0.0)
    ranked["_spread"] = pd.to_numeric(ranked["spread_pct"], errors="coerce").fillna(np.inf)
    if "moneyness_pct" in ranked.columns:
        ranked["_mny"] = ranked["moneyness_pct"].abs()
        ranked = ranked.sort_values(
            ["_mny", "_spread", "_liq", "_oi", "_vol"],
            ascending=[ascending_moneyness, True, False, False, False],
        )
    else:
        ranked = ranked.sort_values(["_spread", "_liq", "_oi", "_vol"], ascending=[True, False, False, False])
    return ranked.iloc[0]


def select_structure_trade(
    structure: StructureSpec,
    chain: pd.DataFrame,
    symbol: str,
    entry_date: pd.Timestamp,
    event_date: pd.Timestamp,
    config: StudyConfig,
) -> Tuple[Optional[List[LegSnapshot]], str]:
    if chain is None or chain.empty:
        return None, "missing_entry_chain"

    front_expiry = pick_front_expiry(chain, event_date)
    if front_expiry is None:
        return None, "no_front_expiry_after_event"
    back_expiry = pick_back_expiry(chain, front_expiry, structure.back_gap_days)

    current_price = float(chain["underlying_price"].dropna().median())
    calls_front = chain[(chain["expiry"] == front_expiry) & (chain["call_put"] == "C")].copy()
    puts_front = chain[(chain["expiry"] == front_expiry) & (chain["call_put"] == "P")].copy()
    if structure.name in {"call_calendar", "put_calendar"}:
        if back_expiry is None:
            return None, "no_back_expiry_after_front"
        subset_front = chain[(chain["expiry"] == front_expiry) & (chain["call_put"] == structure.call_put)].copy()
        subset_back = chain[(chain["expiry"] == back_expiry) & (chain["call_put"] == structure.call_put)].copy()
        front_row, back_row = select_common_strike_pair(subset_front, subset_back, current_price)
        if front_row is None or back_row is None:
            return None, "no_same_strike_pair"
        side_front = -1 if structure.name == "call_calendar" else -1
        side_back = 1
        return [
            build_leg_snapshot(front_row, symbol, entry_date, side=-1),
            build_leg_snapshot(back_row, symbol, entry_date, side=1),
        ], ""

    if structure.name == "atm_straddle":
        call_row, put_row = select_common_strike_pair(calls_front, puts_front, current_price)
        if call_row is None or put_row is None:
            return None, "no_common_strike_straddle_pair"
        return [
            build_leg_snapshot(call_row, symbol, entry_date, side=1),
            build_leg_snapshot(put_row, symbol, entry_date, side=1),
        ], ""

    if structure.name == "otm_strangle":
        target = float(structure.otm_target_pct or config.otm_target_pct)
        call_candidates = calls_front[calls_front["moneyness_pct"] >= target - 1.0].copy()
        put_candidates = puts_front[puts_front["moneyness_pct"] <= -(target - 1.0)].copy()
        if call_candidates.empty:
            call_candidates = calls_front[calls_front["moneyness_pct"] > 0].copy()
        if put_candidates.empty:
            put_candidates = puts_front[puts_front["moneyness_pct"] < 0].copy()
        if call_candidates.empty or put_candidates.empty:
            return None, "no_otm_strangle_candidates"
        call_candidates["_target_gap"] = (call_candidates["moneyness_pct"] - target).abs()
        put_candidates["_target_gap"] = (put_candidates["moneyness_pct"].abs() - target).abs()
        call_row = call_candidates.sort_values(["_target_gap", "spread_pct"]).iloc[0]
        put_row = put_candidates.sort_values(["_target_gap", "spread_pct"]).iloc[0]
        return [
            build_leg_snapshot(call_row, symbol, entry_date, side=1),
            build_leg_snapshot(put_row, symbol, entry_date, side=1),
        ], ""

    return None, f"unsupported_structure_{structure.name}"


def compute_position_value_mid(legs: Sequence[LegSnapshot]) -> float:
    return float(sum(leg.side * leg.mid for leg in legs))


def compute_half_spread_cost(legs: Sequence[LegSnapshot]) -> float:
    return float(sum(abs(leg.side) * (leg.spread / 2.0) for leg in legs))


def compute_adjusted_pnl(
    trade: StructureTrade,
    scenario: ExecutionScenario,
    contract_multiplier: int,
) -> float:
    entry_value = trade.entry_value_mid + scenario.half_spread_fraction * trade.entry_half_spread_cost
    exit_value = trade.exit_value_mid - scenario.half_spread_fraction * trade.exit_half_spread_cost
    pnl = (exit_value - entry_value) * contract_multiplier
    leg_count = len(trade.entry_legs)
    commission = scenario.commission_per_contract * leg_count * 2.0
    return float(pnl - commission)


def build_signal_snapshot(
    symbol: str,
    event: pd.Series,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    day_chain: pd.DataFrame,
    price_history: pd.DataFrame,
) -> Dict[str, Any]:
    signals: Dict[str, Any] = {}
    event_date = pd.Timestamp(event["event_date"])
    history = price_history[price_history["trade_date"] <= entry_date].copy()
    if history.empty or entry_date not in set(history["trade_date"]):
        return {"signal_ready": False, "signal_fail_reasons": ["missing_price_history"]}

    hist_idx = history.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    ).set_index("trade_date")
    close = hist_idx["Close"]
    current_price = float(close.iloc[-1])

    # Build the session-exclusion set once: past earnings days bias the
    # non-event RV baseline upward, which biases event_implied_move_pct
    # downward and hist/tail ratios upward. Production
    # (services.earnings_vol_snapshot) excludes these sessions before
    # YZ/HAR; mirror that here so the decomposition denominator matches.
    prior_events = [
        {"event_date": pd.Timestamp(row["event_date"]), "release_timing": row["release_timing"]}
        for _, row in event.get("prior_symbol_events").iterrows()
    ] if "prior_symbol_events" in event else []
    excluded_sessions = _event_contaminated_session_dates(hist_idx, prior_events) or None

    rv30 = _yang_zhang_rv30(hist_idx, window=30, excluded_sessions=excluded_sessions)
    if not np.isfinite(rv30) or rv30 <= 0:
        rv30 = float(close.pct_change().dropna().tail(30).std(ddof=1) * math.sqrt(252))
        rv_estimator = "close_to_close"
    else:
        rv_estimator = "yang_zhang"
    rv30 = max(rv30, 1e-6)

    # Production parity: feed har_rv_forecast a Series of daily RS
    # annualised vols (excluded_sessions removed), not the raw OHLC frame
    # the prior code mistakenly passed. Fall back to the trailing-mean RS
    # estimator when n < HAR_MIN_OBS, matching the snapshot pipeline.
    rs_daily = _rs_daily_vol_series(hist_idx, excluded_sessions=excluded_sessions)
    rv_har = _har_rv_forecast(rs_daily)
    if rv_har is None or not np.isfinite(rv_har) or rv_har <= 0:
        rv_har = _rs_trailing_mean_forecast(rs_daily)
    rv_pct, vol_regime = _rv_percentile_and_regime(hist_idx.reset_index(), rv30)
    rv_pct_val = float(rv_pct) if rv_pct is not None and np.isfinite(rv_pct) else np.nan

    expiry_rows: List[Tuple[float, float, float, float]] = []
    near_term_implied_move_pct: Optional[float] = None
    near_term_spread_pct: Optional[float] = None
    near_term_dte: Optional[int] = None
    near_term_liq: Optional[float] = None

    for expiry, group in day_chain.groupby("expiry"):
        exp = pd.Timestamp(expiry).normalize()
        dte = int((exp - entry_date).days)
        if dte <= 0:
            continue
        calls = group[group["call_put"] == "C"].copy()
        puts = group[group["call_put"] == "P"].copy()
        if calls.empty or puts.empty:
            continue
        call_row, put_row = select_common_strike_pair(calls, puts, current_price)
        if call_row is None or put_row is None:
            continue
        iv_val = np.nanmean([call_row.get("iv"), put_row.get("iv")])
        if not np.isfinite(iv_val) or iv_val <= 0:
            continue
        expiry_rows.append(
            (
                float(dte),
                float(iv_val),
                float(call_row.get("open_interest", 0) + put_row.get("open_interest", 0)),
                float(call_row.get("volume", 0) + put_row.get("volume", 0)),
            )
        )
        if near_term_implied_move_pct is None:
            call_mid = float(call_row["mid"])
            put_mid = float(put_row["mid"])
            near_term_implied_move_pct = float(((call_mid + put_mid) / current_price) * 100.0)
            near_term_spread_pct = float(np.nanmean([call_row.get("spread_pct"), put_row.get("spread_pct")]))
            near_term_dte = dte
            near_term_liq = float(
                call_row.get("open_interest", 0)
                + put_row.get("open_interest", 0)
                + call_row.get("volume", 0)
                + put_row.get("volume", 0)
            )

    if len(expiry_rows) < 2:
        return {"signal_ready": False, "signal_fail_reasons": ["insufficient_term_structure_points"]}

    expiry_rows.sort(key=lambda item: item[0])
    days = np.array([row[0] for row in expiry_rows], dtype=float)
    ivs = np.array([row[1] for row in expiry_rows], dtype=float)
    # PR #72: bounded interpolation — refuse to extrapolate. When
    # the target tenor isn't bracketed by real expiries (e.g., the
    # nearest expiry is 31D and we want iv30), return a signal-fail
    # row rather than silently fabricating a value clamped to the
    # endpoint. The np.interp pre-fix produced exactly this kind of
    # fabrication, which then propagated into the prior the
    # selector consumes.
    _iv30_value, _iv30_status = bounded_interp(30.0, days, ivs)
    _iv45_value, _iv45_status = bounded_interp(45.0, days, ivs)
    if _iv30_value is None or _iv45_value is None:
        _fail_reason = (
            _iv30_status if _iv30_value is None else _iv45_status
        )
        return {
            "signal_ready": False,
            "signal_fail_reasons": [_fail_reason],
        }
    iv30 = float(_iv30_value)
    iv45 = float(_iv45_value)
    near_dte = float(days[0])
    far_idx = int(np.argmin(np.abs(days - 45.0)))
    far_dte = float(days[far_idx])
    ts_slope = float((ivs[far_idx] - ivs[0]) / max(far_dte - near_dte, 1.0))
    near_back_ratio = float(ivs[0] / ivs[1]) if len(ivs) >= 2 and ivs[1] > 0 else np.nan
    iv_rv = float(iv30 / rv30) if rv30 > 0 else np.nan
    iv_rv_har = float(iv30 / rv_har) if rv_har is not None and rv_har > 0 else np.nan

    # P-5a single source of truth for the event-vol decomposition.  Production
    # uses HAR with an RS-trailing-mean fallback for short histories; this
    # script's HAR pipeline often returns None on per-event 60–90-day windows,
    # so we fall back to Yang-Zhang annualised σ (rv30) when HAR is missing.
    # That mirrors the snapshot module's HAR→trailing-mean cascade in spirit.
    _rv_for_decomp = (
        rv_har if (rv_har is not None and rv_har > 0)
        else (rv30 if (rv30 is not None and np.isfinite(rv30) and rv30 > 0) else None)
    )
    _decomp = decompose_event_vol(
        near_term_implied_move_pct=near_term_implied_move_pct,
        near_term_dte=near_term_dte,
        rv_annual_calendar=_rv_for_decomp,
    )
    non_event_move_pct = _decomp.non_event_move_pct if _decomp.non_event_move_pct is not None else np.nan
    event_implied_move_pct = _decomp.event_implied_move_pct if _decomp.event_implied_move_pct is not None else np.nan

    move_profile = _historical_earnings_move_profile(close=close, earnings_events=prior_events)
    move_anchor = _compute_move_anchor(
        median_move_pct=float(move_profile.get("median_move_pct")) if move_profile.get("median_move_pct") is not None else np.nan,
        avg_last4_move_pct=float(move_profile.get("avg_last4_move_pct")) if move_profile.get("avg_last4_move_pct") is not None else np.nan,
    )
    move_uncertainty = _compute_move_uncertainty_pct(
        move_std_pct=float(move_profile.get("std_move_pct")) if move_profile.get("std_move_pct") is not None else np.nan,
        sample_size=int(move_profile.get("event_count", 0) or 0),
        move_source=str(move_profile.get("source", "none")),
    )
    move_risk_level, move_risk_ratio = _classify_move_risk(
        p90_earnings_move_pct=move_profile.get("p90_move_pct"),
        event_implied_move_pct=event_implied_move_pct if np.isfinite(event_implied_move_pct) and event_implied_move_pct > 0 else None,
        sample_size=int(move_profile.get("event_count", 0) or 0),
    )

    near_front_expiry = pick_front_expiry(day_chain, event_date)
    smile_curvature = np.nan
    smile_points = 0
    if near_front_expiry is not None:
        smile_df = day_chain[
            (day_chain["expiry"] == near_front_expiry)
            & (day_chain["moneyness_pct"].abs() <= 10.0)
            & day_chain["iv"].notna()
        ][["moneyness_pct", "iv"]].dropna()
        if len(smile_df) >= 5:
            try:
                a, _, _ = np.polyfit(smile_df["moneyness_pct"].astype(float), smile_df["iv"].astype(float), 2)
                smile_curvature = float(a)
                smile_points = int(len(smile_df))
            except Exception:
                pass

    signal_fail_reasons: List[str] = []
    if not np.isfinite(iv_rv) or iv_rv < MIN_IV_RV_FOR_EXPANSION:
        signal_fail_reasons.append("iv_rv_below_threshold")
    if not np.isfinite(ts_slope) or ts_slope <= MIN_TS_SLOPE_FOR_EXPANSION:
        signal_fail_reasons.append("term_structure_not_positive")
    if not np.isfinite(rv_pct_val) or rv_pct_val < MIN_RV_PERCENTILE_FOR_EXPANSION:
        signal_fail_reasons.append("vol_regime_percentile_too_low")
    if near_term_spread_pct is None or not np.isfinite(near_term_spread_pct) or near_term_spread_pct > 12.0:
        signal_fail_reasons.append("near_term_spread_too_wide")
    if near_term_liq is None or near_term_liq < MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION:
        signal_fail_reasons.append("near_term_liquidity_too_low")

    expansion_live_score = float(np.clip(
        0.32 * np.clip((iv_rv - MIN_IV_RV_FOR_EXPANSION) / 0.55, 0.0, 1.0)
        + 0.24 * np.clip((ts_slope - MIN_TS_SLOPE_FOR_EXPANSION) / 0.010, 0.0, 1.0)
        + 0.22 * np.clip((rv_pct_val - MIN_RV_PERCENTILE_FOR_EXPANSION) / 40.0, 0.0, 1.0)
        + 0.22 * np.clip(
            (math.log1p(max(float(near_term_liq or 1.0), 1.0)) - math.log1p(MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION))
            / (math.log1p(25_000.0) - math.log1p(MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION)),
            0.0,
            1.0,
        ),
        0.0,
        1.0,
    ))

    signals.update(
        {
            "signal_ready": True,
            "signal_fail_reasons": signal_fail_reasons,
            "current_price": current_price,
            "rv_estimator": rv_estimator,
            "rv30": float(rv30),
            "rv_har_forecast": float(rv_har) if rv_har is not None else None,
            "rv_percentile_rank": rv_pct_val if np.isfinite(rv_pct_val) else None,
            "vol_regime": vol_regime,
            "iv30": float(iv30),
            "iv45": float(iv45),
            "iv_rv30": float(iv_rv) if np.isfinite(iv_rv) else None,
            "iv_rv_har": float(iv_rv_har) if np.isfinite(iv_rv_har) else None,
            "term_structure_slope": float(ts_slope),
            "near_back_iv_ratio": float(near_back_ratio) if np.isfinite(near_back_ratio) else None,
            "near_term_dte": near_term_dte,
            "near_term_spread_pct": float(near_term_spread_pct) if near_term_spread_pct is not None else None,
            "near_term_liquidity_proxy": float(near_term_liq) if near_term_liq is not None else None,
            "implied_move_pct": float(near_term_implied_move_pct) if near_term_implied_move_pct is not None else None,
            "event_implied_move_pct": float(event_implied_move_pct) if np.isfinite(event_implied_move_pct) else None,
            "non_event_move_pct": float(non_event_move_pct) if np.isfinite(non_event_move_pct) else None,
            "move_anchor_pct": float(move_anchor) if move_anchor is not None else None,
            "historical_p90_move_pct": float(move_profile.get("p90_move_pct")) if move_profile.get("p90_move_pct") is not None else None,
            "move_uncertainty_pct": float(move_uncertainty) if move_uncertainty is not None else None,
            "move_risk_level": move_risk_level,
            "move_risk_ratio": float(move_risk_ratio) if move_risk_ratio is not None else None,
            "smile_curvature": float(smile_curvature) if np.isfinite(smile_curvature) else None,
            "smile_points": smile_points,
            "nbr_ratio": float(event.get("nbr_ratio")) if event.get("nbr_ratio") is not None else None,
            "pre_front_iv_label": float(event.get("pre_front_iv")) if event.get("pre_front_iv") is not None else None,
            "pre_back_iv_label": float(event.get("pre_back_iv")) if event.get("pre_back_iv") is not None else None,
            "expansion_live_score": expansion_live_score,
        }
    )
    return signals


def enforce_leg_liquidity(legs: Sequence[LegSnapshot], min_oi: int, max_spread_pct: float) -> Tuple[bool, str]:
    for leg in legs:
        if leg.open_interest < min_oi:
            return False, f"leg_oi_below_{min_oi}"
        if leg.spread_pct > max_spread_pct:
            return False, f"leg_spread_above_{max_spread_pct:.1f}"
    return True, ""


def realize_structure_trade(
    structure: StructureSpec,
    symbol: str,
    event: pd.Series,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_chain: Optional[pd.DataFrame],
    exit_chain: Optional[pd.DataFrame],
    config: StudyConfig,
) -> Tuple[Optional[StructureTrade], str]:
    if entry_chain is None or entry_chain.empty:
        return None, "missing_entry_chain"
    entry_legs, reason = select_structure_trade(structure, entry_chain, symbol, entry_date, pd.Timestamp(event["event_date"]), config)
    if entry_legs is None:
        return None, reason
    ok, liq_reason = enforce_leg_liquidity(entry_legs, config.base_min_leg_open_interest, config.base_max_leg_spread_pct)
    if not ok:
        return None, liq_reason
    if exit_chain is None or exit_chain.empty:
        return None, "missing_exit_chain"

    exit_legs: List[LegSnapshot] = []
    for leg in entry_legs:
        row = exact_lookup(exit_chain, leg.identity)
        if row is None:
            return None, "missing_exact_exit_quote"
        exit_leg = build_leg_snapshot(row, symbol, exit_date, side=leg.side)
        if exit_leg.identity.expiry != leg.identity.expiry:
            raise StudyInvariantError(f"Expiry mismatch for {symbol} {event['event_date']} {structure.name}")
        if exit_leg.identity.option_type != leg.identity.option_type:
            raise StudyInvariantError(f"Type mismatch for {symbol} {event['event_date']} {structure.name}")
        if not math.isclose(exit_leg.identity.strike, leg.identity.strike, rel_tol=0.0, abs_tol=1e-12):
            raise StudyInvariantError(f"Strike mismatch for {symbol} {event['event_date']} {structure.name}")
        exit_legs.append(exit_leg)

    entry_value_mid = compute_position_value_mid(entry_legs)
    exit_value_mid = compute_position_value_mid(exit_legs)
    debit_capital = max(entry_value_mid, 0.0)
    if debit_capital <= 0:
        return None, "negative_or_zero_debit"
    # Side-weight so positive ⇔ "in the position's favor" regardless of
    # leg direction. For a long leg (side=+1), favorable IV change is
    # exit_iv > entry_iv. For a short leg (side=-1), favorable is the
    # opposite — IV crush profits the short. Without side-weighting,
    # calendar avg_iv_change confuses front-crush (favorable) and
    # back-rise (also favorable) by averaging them as opposite signs.
    avg_iv_change = float(np.nanmean([
        entry_leg.side * (exit_leg.iv - entry_leg.iv)
        for entry_leg, exit_leg in zip(entry_legs, exit_legs)
        if entry_leg.iv is not None and exit_leg.iv is not None
    ])) if any(entry_leg.iv is not None and exit_leg.iv is not None for entry_leg, exit_leg in zip(entry_legs, exit_legs)) else None

    return StructureTrade(
        entry_legs=entry_legs,
        exit_legs=exit_legs,
        entry_value_mid=entry_value_mid,
        exit_value_mid=exit_value_mid,
        entry_half_spread_cost=compute_half_spread_cost(entry_legs),
        exit_half_spread_cost=compute_half_spread_cost(exit_legs),
        debit_capital_per_spread=debit_capital,
        avg_iv_change=avg_iv_change,
    ), ""


def walk_forward_summary(prior_records: Sequence[HistoryOutcome]) -> Dict[str, Any]:
    realized = [record for record in prior_records if record.pnl_mid is not None and np.isfinite(record.pnl_mid)]
    if len(realized) == 0:
        return {
            "wf_history_count": 0,
            "wf_expected_iv_change": None,
            "wf_expected_pnl_mid": None,
            "wf_expected_pnl_adjusted": None,
            "wf_ranking_score": None,
        }
    iv_changes = [record.avg_iv_change for record in realized if record.avg_iv_change is not None and np.isfinite(record.avg_iv_change)]
    mid_pnls = [record.pnl_mid for record in realized if record.pnl_mid is not None and np.isfinite(record.pnl_mid)]
    adj_pnls = [record.pnl_cross_100 for record in realized if record.pnl_cross_100 is not None and np.isfinite(record.pnl_cross_100)]
    adj_returns = [
        record.return_cross_100_pct for record in realized if record.return_cross_100_pct is not None and np.isfinite(record.return_cross_100_pct)
    ]
    if not mid_pnls or not adj_pnls:
        return {
            "wf_history_count": len(realized),
            "wf_expected_iv_change": float(np.mean(iv_changes)) if iv_changes else None,
            "wf_expected_pnl_mid": None,
            "wf_expected_pnl_adjusted": None,
            "wf_ranking_score": None,
        }
    avg_adj = float(np.mean(adj_pnls))
    avg_mid = float(np.mean(mid_pnls))
    avg_iv = float(np.mean(iv_changes)) if iv_changes else None
    win_rate = float(np.mean([1.0 if val > 0 else 0.0 for val in adj_pnls]))
    avg_adj_ret = float(np.mean(adj_returns)) if adj_returns else np.nan
    score = float(np.clip(
        0.45 * np.clip((avg_adj_ret + 10.0) / 25.0, 0.0, 1.0)
        + 0.25 * np.clip(((float(np.mean([record.return_mid_pct for record in realized if record.return_mid_pct is not None])) if any(record.return_mid_pct is not None for record in realized) else -10.0) + 10.0) / 25.0, 0.0, 1.0)
        + 0.15 * np.clip(win_rate, 0.0, 1.0)
        + 0.15 * np.clip(len(realized) / 8.0, 0.0, 1.0),
        0.0,
        1.0,
    ))
    return {
        "wf_history_count": len(realized),
        "wf_expected_iv_change": avg_iv,
        "wf_expected_pnl_mid": avg_mid,
        "wf_expected_pnl_adjusted": avg_adj,
        "wf_ranking_score": score,
    }


def history_outcome_from_trade(structure_trade: StructureTrade, config: StudyConfig) -> HistoryOutcome:
    pnl_mid = (structure_trade.exit_value_mid - structure_trade.entry_value_mid) * config.contract_multiplier
    pnl_cross_100 = compute_adjusted_pnl(structure_trade, SCENARIOS["cross_100"], config.contract_multiplier)
    debit_capital = structure_trade.debit_capital_per_spread * config.contract_multiplier
    return HistoryOutcome(
        avg_iv_change=structure_trade.avg_iv_change,
        pnl_mid=pnl_mid,
        pnl_cross_100=pnl_cross_100,
        return_mid_pct=(pnl_mid / debit_capital) * 100.0 if debit_capital > 0 else None,
        return_cross_100_pct=(pnl_cross_100 / debit_capital) * 100.0 if debit_capital > 0 else None,
    )


def trade_to_record(
    symbol: str,
    event: pd.Series,
    structure: StructureSpec,
    entry_offset: int,
    exit_offset: int,
    status: str,
    exclusion_reason: str,
    signal_snapshot: Dict[str, Any],
    wf_summary: Dict[str, Any],
    entry_date: Optional[pd.Timestamp],
    exit_date: Optional[pd.Timestamp],
    structure_trade: Optional[StructureTrade],
    config: StudyConfig,
) -> TradeRecord:
    signal_ready = bool(signal_snapshot.get("signal_ready"))
    signal_fail_reasons = list(signal_snapshot.get("signal_fail_reasons", [])) if signal_ready else list(signal_snapshot.get("signal_fail_reasons", []))
    signal_pass = signal_ready and not signal_fail_reasons

    leg_defaults = {
        "leg1_option_type": None,
        "leg1_expiry": None,
        "leg1_strike": None,
        "leg1_option_symbol": None,
        "leg1_option_id": None,
        "leg1_side": None,
        "leg1_bid_entry": None,
        "leg1_ask_entry": None,
        "leg1_mid_entry": None,
        "leg1_bid_exit": None,
        "leg1_ask_exit": None,
        "leg1_mid_exit": None,
        "leg1_iv_entry": None,
        "leg1_iv_exit": None,
        "leg1_iv_change": None,
        "leg1_spread_entry": None,
        "leg1_spread_exit": None,
        "leg1_open_interest_entry": None,
        "leg1_volume_entry": None,
        "leg2_option_type": None,
        "leg2_expiry": None,
        "leg2_strike": None,
        "leg2_option_symbol": None,
        "leg2_option_id": None,
        "leg2_side": None,
        "leg2_bid_entry": None,
        "leg2_ask_entry": None,
        "leg2_mid_entry": None,
        "leg2_bid_exit": None,
        "leg2_ask_exit": None,
        "leg2_mid_exit": None,
        "leg2_iv_entry": None,
        "leg2_iv_exit": None,
        "leg2_iv_change": None,
        "leg2_spread_entry": None,
        "leg2_spread_exit": None,
        "leg2_open_interest_entry": None,
        "leg2_volume_entry": None,
    }

    avg_iv_change = None
    entry_value_mid = None
    exit_value_mid = None
    mid_pnl_per_spread = None
    debit_capital = None
    return_mid_pct = None
    pnl_mid = None
    entry_cost = None
    exit_cost = None
    pnl_values = {name: None for name in SCENARIOS}
    return_values = {name: None for name in SCENARIOS}

    if structure_trade is not None:
        legs = list(zip(structure_trade.entry_legs, structure_trade.exit_legs))
        for idx, (entry_leg, exit_leg) in enumerate(legs[:2], start=1):
            leg_defaults[f"leg{idx}_option_type"] = entry_leg.identity.option_type
            leg_defaults[f"leg{idx}_expiry"] = entry_leg.identity.expiry
            leg_defaults[f"leg{idx}_strike"] = entry_leg.identity.strike
            leg_defaults[f"leg{idx}_option_symbol"] = entry_leg.identity.option_symbol
            leg_defaults[f"leg{idx}_option_id"] = entry_leg.identity.option_id
            leg_defaults[f"leg{idx}_side"] = entry_leg.side
            leg_defaults[f"leg{idx}_bid_entry"] = entry_leg.bid
            leg_defaults[f"leg{idx}_ask_entry"] = entry_leg.ask
            leg_defaults[f"leg{idx}_mid_entry"] = entry_leg.mid
            leg_defaults[f"leg{idx}_bid_exit"] = exit_leg.bid
            leg_defaults[f"leg{idx}_ask_exit"] = exit_leg.ask
            leg_defaults[f"leg{idx}_mid_exit"] = exit_leg.mid
            leg_defaults[f"leg{idx}_iv_entry"] = entry_leg.iv
            leg_defaults[f"leg{idx}_iv_exit"] = exit_leg.iv
            leg_defaults[f"leg{idx}_iv_change"] = (
                exit_leg.iv - entry_leg.iv
                if entry_leg.iv is not None and exit_leg.iv is not None
                else None
            )
            leg_defaults[f"leg{idx}_spread_entry"] = entry_leg.spread
            leg_defaults[f"leg{idx}_spread_exit"] = exit_leg.spread
            leg_defaults[f"leg{idx}_open_interest_entry"] = entry_leg.open_interest
            leg_defaults[f"leg{idx}_volume_entry"] = entry_leg.volume

        avg_iv_change = structure_trade.avg_iv_change
        entry_value_mid = structure_trade.entry_value_mid
        exit_value_mid = structure_trade.exit_value_mid
        mid_pnl_per_spread = structure_trade.exit_value_mid - structure_trade.entry_value_mid
        debit_capital = structure_trade.debit_capital_per_spread * config.contract_multiplier
        return_mid_pct = (
            (mid_pnl_per_spread / structure_trade.debit_capital_per_spread) * 100.0
            if structure_trade.debit_capital_per_spread > 0
            else None
        )
        pnl_mid = mid_pnl_per_spread * config.contract_multiplier
        entry_cost = structure_trade.entry_half_spread_cost * config.contract_multiplier
        exit_cost = structure_trade.exit_half_spread_cost * config.contract_multiplier
        for name, scenario in SCENARIOS.items():
            pnl = compute_adjusted_pnl(structure_trade, scenario, config.contract_multiplier)
            pnl_values[name] = pnl
            return_values[name] = (pnl / debit_capital) * 100.0 if debit_capital and debit_capital > 0 else None

    return TradeRecord(
        symbol=symbol,
        event_date=pd.Timestamp(event["event_date"]).strftime("%Y-%m-%d"),
        release_timing=str(event["release_timing"]),
        structure=structure.name,
        entry_offset_bdays=entry_offset,
        exit_offset_bdays=exit_offset,
        status=status,
        exclusion_reason=exclusion_reason,
        candidate_signal_pass=signal_pass,
        entry_date=entry_date.strftime("%Y-%m-%d") if entry_date is not None else None,
        exit_date=exit_date.strftime("%Y-%m-%d") if exit_date is not None else None,
        accepted_for_entry=status in {STATUS_REALIZED, STATUS_MISSING_EXIT_QUOTE, STATUS_NEGATIVE_DEBIT, STATUS_STRUCTURAL_INVALID},
        exact_exit_quote_found=structure_trade is not None and status == STATUS_REALIZED,
        priceable_realized=status == STATUS_REALIZED,
        wf_history_count=int(wf_summary.get("wf_history_count", 0)),
        wf_expected_iv_change=wf_summary.get("wf_expected_iv_change"),
        wf_expected_pnl_mid=wf_summary.get("wf_expected_pnl_mid"),
        wf_expected_pnl_adjusted=wf_summary.get("wf_expected_pnl_adjusted"),
        wf_ranking_score=wf_summary.get("wf_ranking_score"),
        signal_iv_rv30=signal_snapshot.get("iv_rv30"),
        signal_iv_rv_har=signal_snapshot.get("iv_rv_har"),
        signal_term_structure_slope=signal_snapshot.get("term_structure_slope"),
        signal_near_back_iv_ratio=signal_snapshot.get("near_back_iv_ratio"),
        signal_rv30=signal_snapshot.get("rv30"),
        signal_rv_har_forecast=signal_snapshot.get("rv_har_forecast"),
        signal_rv_percentile_rank=signal_snapshot.get("rv_percentile_rank"),
        signal_vol_regime=signal_snapshot.get("vol_regime"),
        signal_near_term_dte=signal_snapshot.get("near_term_dte"),
        signal_near_term_spread_pct=signal_snapshot.get("near_term_spread_pct"),
        signal_near_term_liquidity_proxy=signal_snapshot.get("near_term_liquidity_proxy"),
        signal_event_implied_move_pct=signal_snapshot.get("event_implied_move_pct"),
        signal_non_event_move_pct=signal_snapshot.get("non_event_move_pct"),
        signal_implied_move_pct=signal_snapshot.get("implied_move_pct"),
        signal_move_anchor_pct=signal_snapshot.get("move_anchor_pct"),
        signal_historical_p90_move_pct=signal_snapshot.get("historical_p90_move_pct"),
        signal_move_risk_level=signal_snapshot.get("move_risk_level"),
        signal_move_risk_ratio=signal_snapshot.get("move_risk_ratio"),
        signal_smile_curvature=signal_snapshot.get("smile_curvature"),
        signal_smile_points=signal_snapshot.get("smile_points"),
        signal_front_oi=signal_snapshot.get("signal_front_oi"),
        signal_back_oi=signal_snapshot.get("signal_back_oi"),
        signal_fail_reasons=";".join(signal_fail_reasons),
        avg_iv_change=avg_iv_change,
        entry_value_mid=entry_value_mid,
        exit_value_mid=exit_value_mid,
        mid_pnl_per_spread=mid_pnl_per_spread,
        debit_capital_per_spread=(structure_trade.debit_capital_per_spread if structure_trade is not None else None),
        return_mid_pct=return_mid_pct,
        pnl_mid=pnl_mid,
        entry_half_spread_cost=entry_cost,
        exit_half_spread_cost=exit_cost,
        pnl_cross_25=pnl_values["cross_25"],
        pnl_cross_50=pnl_values["cross_50"],
        pnl_cross_100=pnl_values["cross_100"],
        pnl_cross_100_commission=pnl_values["cross_100_commission"],
        return_cross_25_pct=return_values["cross_25"],
        return_cross_50_pct=return_values["cross_50"],
        return_cross_100_pct=return_values["cross_100"],
        return_cross_100_commission_pct=return_values["cross_100_commission"],
        **leg_defaults,
    )


def return_column_for_scenario(scenario_name: str) -> str:
    if scenario_name == "mid":
        return "return_mid_pct"
    if scenario_name == "cross_25":
        return "return_cross_25_pct"
    if scenario_name == "cross_50":
        return "return_cross_50_pct"
    if scenario_name == "cross_100":
        return "return_cross_100_pct"
    if scenario_name == "cross_100_commission":
        return "return_cross_100_commission_pct"
    raise KeyError(scenario_name)


def pnl_column_for_scenario(scenario_name: str) -> str:
    if scenario_name == "mid":
        return "pnl_mid"
    return f"pnl_{scenario_name}"


def compute_equity_curve(trades: pd.DataFrame, pnl_col: str, date_col: str = "exit_date") -> pd.DataFrame:
    realized = trades[(trades["status"] == STATUS_REALIZED) & trades[pnl_col].notna()].copy()
    if realized.empty:
        return pd.DataFrame(columns=["date", "equity", "daily_pnl", "daily_return"])
    realized[date_col] = pd.to_datetime(realized[date_col])
    daily = realized.groupby(date_col, as_index=False)[pnl_col].sum().rename(columns={pnl_col: "daily_pnl", date_col: "date"})
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["equity"] = daily["daily_pnl"].cumsum()
    prev_equity = daily["equity"].shift(1).replace(0.0, np.nan)
    capital = realized.groupby("entry_date", as_index=False)["debit_capital_per_spread"].mean()
    capital["entry_date"] = pd.to_datetime(capital["entry_date"])
    median_capital = float(realized["debit_capital_per_spread"].median() * 100.0) if realized["debit_capital_per_spread"].notna().any() else np.nan
    daily["daily_return"] = np.where(
        np.isfinite(median_capital) and median_capital > 0,
        daily["daily_pnl"] / median_capital,
        np.nan,
    )
    return daily


def compute_max_drawdown(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return np.nan
    eq = equity_curve["equity"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(eq)
    drawdowns = peaks - eq
    return float(np.max(drawdowns)) if len(drawdowns) else np.nan


def compute_return_sharpe(returns: pd.Series) -> float:
    series = returns.dropna()
    if len(series) < 2 or float(series.std(ddof=1)) <= 0:
        return np.nan
    return float(series.mean() / series.std(ddof=1) * math.sqrt(252))


def summarize_group(
    trades: pd.DataFrame,
    scenario_name: str,
    *,
    label: str,
) -> Dict[str, Any]:
    pnl_col = pnl_column_for_scenario(scenario_name)
    ret_col = return_column_for_scenario(scenario_name)
    realized = trades[(trades["status"] == STATUS_REALIZED) & trades[pnl_col].notna()].copy()
    equity = compute_equity_curve(trades, pnl_col=pnl_col)
    by_symbol = realized.groupby("symbol")[pnl_col].sum().sort_values(ascending=False) if not realized.empty else pd.Series(dtype=float)
    total_pnl = float(realized[pnl_col].sum()) if not realized.empty else 0.0
    winners = realized[realized[pnl_col] > 0]
    losers = realized[realized[pnl_col] <= 0]
    return {
        "label": label,
        "trade_count": int(len(trades)),
        "accepted_count": int(trades["accepted_for_entry"].sum()),
        "realized_count": int(len(realized)),
        "missing_quote_count": int((trades["status"] == STATUS_MISSING_EXIT_QUOTE).sum()),
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": float(realized[pnl_col].mean()) if not realized.empty else np.nan,
        "win_rate": float((realized[pnl_col] > 0).mean()) if not realized.empty else np.nan,
        "avg_win": float(winners[pnl_col].mean()) if not winners.empty else np.nan,
        "avg_loss": float(losers[pnl_col].mean()) if not losers.empty else np.nan,
        "return_based_sharpe": compute_return_sharpe(realized[ret_col]) if not realized.empty else np.nan,
        "max_drawdown": compute_max_drawdown(equity),
        "p10_trade_return_pct": float(realized[ret_col].quantile(0.10)) if not realized.empty else np.nan,
        "p50_trade_return_pct": float(realized[ret_col].quantile(0.50)) if not realized.empty else np.nan,
        "p90_trade_return_pct": float(realized[ret_col].quantile(0.90)) if not realized.empty else np.nan,
        "top5_concentration_pct": float(by_symbol.head(5).sum() / total_pnl * 100.0) if not by_symbol.empty and total_pnl != 0 else np.nan,
        "ex_top3_total_pnl": float(realized[~realized["symbol"].isin(by_symbol.head(3).index)][pnl_col].sum()) if not by_symbol.empty else np.nan,
        "ex_top5_total_pnl": float(realized[~realized["symbol"].isin(by_symbol.head(5).index)][pnl_col].sum()) if not by_symbol.empty else np.nan,
    }


def summarize_scoreboard(base_trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for structure in BASE_STRUCTURE_ROWS:
        subset = base_trades[base_trades["structure"] == structure].copy()
        for scenario_name in SCENARIOS:
            summary = summarize_group(subset, scenario_name, label=f"{structure}:{scenario_name}")
            summary["structure"] = structure
            summary["scenario"] = scenario_name
            rows.append(summary)
    return pd.DataFrame(rows)


def summarize_simulated_group(
    trades: pd.DataFrame,
    scenario_name: str,
    *,
    label: str,
) -> Dict[str, Any]:
    pnl_col = pnl_column_for_scenario(scenario_name)
    ret_col = return_column_for_scenario(scenario_name)
    simulated = trades[trades[pnl_col].notna()].copy()
    equity = compute_equity_curve(simulated.assign(status=STATUS_REALIZED), pnl_col=pnl_col) if not simulated.empty else pd.DataFrame()
    by_symbol = simulated.groupby("symbol")[pnl_col].sum().sort_values(ascending=False) if not simulated.empty else pd.Series(dtype=float)
    total_pnl = float(simulated[pnl_col].sum()) if not simulated.empty else 0.0
    winners = simulated[simulated[pnl_col] > 0]
    losers = simulated[simulated[pnl_col] <= 0]
    return {
        "label": label,
        "trade_count": int(len(trades)),
        "accepted_count": int(trades["accepted_for_entry"].sum()),
        "simulated_priceable_count": int(len(simulated)),
        "simulated_missing_quote_count": int(
            trades["exclusion_reason"].fillna("").str.contains("missing_exit_chain|missing_exact_exit_quote", regex=True).sum()
        ),
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": float(simulated[pnl_col].mean()) if not simulated.empty else np.nan,
        "win_rate": float((simulated[pnl_col] > 0).mean()) if not simulated.empty else np.nan,
        "avg_win": float(winners[pnl_col].mean()) if not winners.empty else np.nan,
        "avg_loss": float(losers[pnl_col].mean()) if not losers.empty else np.nan,
        "return_based_sharpe": compute_return_sharpe(simulated[ret_col]) if not simulated.empty else np.nan,
        "max_drawdown": compute_max_drawdown(equity),
        "p10_trade_return_pct": float(simulated[ret_col].quantile(0.10)) if not simulated.empty else np.nan,
        "p50_trade_return_pct": float(simulated[ret_col].quantile(0.50)) if not simulated.empty else np.nan,
        "p90_trade_return_pct": float(simulated[ret_col].quantile(0.90)) if not simulated.empty else np.nan,
        "top5_concentration_pct": float(by_symbol.head(5).sum() / total_pnl * 100.0) if not by_symbol.empty and total_pnl != 0 else np.nan,
        "ex_top3_total_pnl": float(simulated[~simulated["symbol"].isin(by_symbol.head(3).index)][pnl_col].sum()) if not by_symbol.empty else np.nan,
        "ex_top5_total_pnl": float(simulated[~simulated["symbol"].isin(by_symbol.head(5).index)][pnl_col].sum()) if not by_symbol.empty else np.nan,
    }


def summarize_simulated_scoreboard(base_trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for structure in BASE_STRUCTURE_ROWS:
        subset = base_trades[base_trades["structure"] == structure].copy()
        for scenario_name in SCENARIOS:
            summary = summarize_simulated_group(subset, scenario_name, label=f"{structure}:{scenario_name}")
            summary["structure"] = structure
            summary["scenario"] = scenario_name
            rows.append(summary)
    return pd.DataFrame(rows)


def feature_rankings(base_trades: pd.DataFrame) -> pd.DataFrame:
    realized = base_trades[base_trades["pnl_mid"].notna()].copy()
    if realized.empty:
        return pd.DataFrame()
    features = {
        "nbr_ratio": "signal_near_back_iv_ratio",
        "iv_rv30": "signal_iv_rv30",
        "iv_rv_har": "signal_iv_rv_har",
        "term_structure_slope": "signal_term_structure_slope",
        "rv_percentile_rank": "signal_rv_percentile_rank",
        "near_term_spread_pct": "signal_near_term_spread_pct",
        "near_term_liquidity_proxy": "signal_near_term_liquidity_proxy",
        "event_implied_move_pct": "signal_event_implied_move_pct",
        "non_event_move_pct": "signal_non_event_move_pct",
        "smile_curvature": "signal_smile_curvature",
        "wf_ranking_score": "wf_ranking_score",
    }
    rows: List[Dict[str, Any]] = []
    # Partition by structure: long-vol structures (atm_straddle,
    # otm_strangle) profit from IV expansion, while calendars profit from
    # front IV crush relative to back. Pooling these in one Spearman
    # correlation averages opposing direction conventions and biases all
    # reported correlations toward zero. Per-structure correlations
    # preserve the within-structure signal→PnL relationship.
    for structure_name, structure_df in realized.groupby("structure"):
        for label, col in features.items():
            subset = structure_df[[col, "avg_iv_change", "return_mid_pct", "return_cross_50_pct"]].dropna()
            if len(subset) < 10:
                continue
            iv_corr = subset[col].corr(subset["avg_iv_change"], method="spearman")
            mid_corr = subset[col].corr(subset["return_mid_pct"], method="spearman")
            adj_corr = subset[col].corr(subset["return_cross_50_pct"], method="spearman")
            top_cut = subset[col].quantile(0.75)
            bot_cut = subset[col].quantile(0.25)
            top = subset[subset[col] >= top_cut]
            bot = subset[subset[col] <= bot_cut]
            rows.append(
                {
                    "structure": structure_name,
                    "feature": label,
                    "sample_size": int(len(subset)),
                    "spearman_iv_change": float(iv_corr) if iv_corr is not None else np.nan,
                    "spearman_mid_return": float(mid_corr) if mid_corr is not None else np.nan,
                    "spearman_adj_return_50pct": float(adj_corr) if adj_corr is not None else np.nan,
                    "top_quartile_avg_iv_change": float(top["avg_iv_change"].mean()) if not top.empty else np.nan,
                    "bottom_quartile_avg_iv_change": float(bot["avg_iv_change"].mean()) if not bot.empty else np.nan,
                    "top_quartile_adj_return_50pct": float(top["return_cross_50_pct"].mean()) if not top.empty else np.nan,
                    "bottom_quartile_adj_return_50pct": float(bot["return_cross_50_pct"].mean()) if not bot.empty else np.nan,
                }
            )
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking["rank_score"] = ranking["spearman_iv_change"].abs().fillna(0.0) + ranking["spearman_adj_return_50pct"].abs().fillna(0.0)
        ranking = ranking.sort_values(["structure", "rank_score"], ascending=[True, False]).reset_index(drop=True)
    return ranking


def breakdown_table(base_trades: pd.DataFrame, scenario_name: str, group_cols: Sequence[str]) -> pd.DataFrame:
    pnl_col = pnl_column_for_scenario(scenario_name)
    realized = base_trades[(base_trades["status"] == STATUS_REALIZED) & base_trades[pnl_col].notna()].copy()
    if realized.empty:
        return pd.DataFrame()
    grouped = realized.groupby(list(group_cols), dropna=False)
    rows: List[Dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: key[idx] for idx, col in enumerate(group_cols)}
        row.update(
            {
                "trade_count": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "total_pnl": float(group[pnl_col].sum()),
                "avg_pnl": float(group[pnl_col].mean()),
                "win_rate": float((group[pnl_col] > 0).mean()),
                "avg_return_pct": float(group[return_column_for_scenario(scenario_name)].mean()),
                "missing_quote_pct": float((group["status"] == STATUS_MISSING_EXIT_QUOTE).mean() * 100.0) if "status" in group else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trade_count", ascending=False).reset_index(drop=True)


def liquidity_tier(value: Optional[float]) -> str:
    val = float(value) if value is not None and np.isfinite(value) else np.nan
    if not np.isfinite(val):
        return "unknown"
    if val >= 10_000:
        return "high"
    if val >= 3_000:
        return "mid"
    return "low"


def dte_bucket(value: Optional[int]) -> str:
    if value is None or not np.isfinite(float(value)):
        return "unknown"
    v = int(value)
    if v <= 5:
        return "3-5"
    if v <= 7:
        return "6-7"
    return "8-10"


def strict_liquidity_filter(trades: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    mask = (
        trades["leg1_open_interest_entry"].fillna(0) >= config.strict_min_leg_open_interest
    ) & (
        trades["leg2_open_interest_entry"].fillna(0) >= config.strict_min_leg_open_interest
    ) & (
        trades["leg1_spread_entry"].fillna(np.inf) <= config.strict_max_leg_spread_pct / 100.0 * trades["leg1_mid_entry"].fillna(0)
    ) & (
        trades["leg2_spread_entry"].fillna(np.inf) <= config.strict_max_leg_spread_pct / 100.0 * trades["leg2_mid_entry"].fillna(0)
    )
    return trades[mask].copy()


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: Optional[int] = None) -> str:
    if frame.empty:
        return "No rows."
    subset = frame.loc[:, list(columns)].copy()
    if limit is not None:
        subset = subset.head(limit)

    def format_cell(value: Any) -> str:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            if abs(float(value)) >= 100 or float(value).is_integer():
                return f"{float(value):.0f}"
            return f"{float(value):.3f}"
        return str(value)

    headers = [str(col) for col in subset.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def build_memo(
    config: StudyConfig,
    results: Dict[str, Any],
    out_dir: Path,
) -> str:
    scoreboard = results["scoreboard"]
    simulated_scoreboard = results["simulated_scoreboard"]
    rankings = results["signal_rankings"]
    base_call = scoreboard[scoreboard["structure"] == "call_calendar"].copy()
    base_mid = base_call[base_call["scenario"] == "mid"]
    base_adj = base_call[base_call["scenario"] == "cross_100"]
    sim_base_call = simulated_scoreboard[simulated_scoreboard["structure"] == "call_calendar"].copy()
    sim_base_mid = sim_base_call[sim_base_call["scenario"] == "mid"]
    sim_base_adj = sim_base_call[sim_base_call["scenario"] == "cross_100"]
    best_adj_row = scoreboard.sort_values("total_pnl", ascending=False).iloc[0] if not scoreboard.empty else None
    best_sim_adj_row = simulated_scoreboard.sort_values("total_pnl", ascending=False).iloc[0] if not simulated_scoreboard.empty else None

    if scoreboard.empty:
        verdict = "UNVERIFIED"
        verdict_text = "No priceable trades were realized, so the pre-earnings IV expansion mode could not be validated."
    else:
        positive_adj = scoreboard[(scoreboard["scenario"] == "cross_100") & (scoreboard["total_pnl"] > 0)]
        positive_50 = scoreboard[(scoreboard["scenario"] == "cross_50") & (scoreboard["total_pnl"] > 0)]
        if positive_adj.empty and positive_50.empty:
            verdict = "signal exists but not monetizable"
        elif positive_adj.empty:
            verdict = "promising but fragile"
        else:
            verdict = "tradeable"
        verdict_text = f"The strongest structure under the tested rules was {best_adj_row['structure']} in scenario {best_adj_row['scenario']} with total P&L {best_adj_row['total_pnl']:.2f}." if best_adj_row is not None else "No strongest structure available."

    scoreboard_md = markdown_table(
        scoreboard,
        [
            "structure",
            "scenario",
            "accepted_count",
            "realized_count",
            "missing_quote_count",
            "total_pnl",
            "win_rate",
            "return_based_sharpe",
            "max_drawdown",
            "top5_concentration_pct",
        ],
    )
    simulated_scoreboard_md = markdown_table(
        simulated_scoreboard,
        [
            "structure",
            "scenario",
            "accepted_count",
            "simulated_priceable_count",
            "simulated_missing_quote_count",
            "total_pnl",
            "win_rate",
            "return_based_sharpe",
            "max_drawdown",
            "top5_concentration_pct",
        ],
    )

    # Per-structure sub-tables so each structure's top features stay
    # visible rather than the global top-N collapsing onto whichever
    # structure has the strongest correlations.
    if rankings.empty:
        rankings_md = "(no structures with sufficient sample)"
    else:
        rankings_md_parts: List[str] = []
        for structure_name, group in rankings.groupby("structure", sort=True):
            rankings_md_parts.append(f"### {structure_name}\n")
            rankings_md_parts.append(markdown_table(group, list(group.columns), limit=5))
            rankings_md_parts.append("")
        rankings_md = "\n".join(rankings_md_parts)

    memo = f"""# IV Expansion Candidate Historical Study

## A. Executive Summary
Verdict: **{verdict}**

{verdict_text}

This study backtests the new pre-earnings mode using only pre-event entries and exits, exact contract identity persistence, and no contract substitution at exit. The baseline structure comparison uses **7 business days before earnings** for entry and **1 business day before earnings** for exit because the live engine's timing score is centered on 7 DTE. Robustness sections then test 3/5/7/10 day entries and 2/1/0 day exits.

## B. Implementation Audit
- Event universe source: local SQLite table `earnings_iv_decay_labels` at `{config.db_path}`.
- Option-chain source: local parquet feature store at `{config.options_root}`.
- Price / RV source: local OHLCV parquet store at `{config.price_root}`.
- Exact exit lookup policy: primary identity `(option_type, strike, expiry)` with `option_symbol` / `option_id` as tie-breakers; missing exact quotes are flagged and excluded from realized P&L.
- No post-earnings exits are used anywhere in the strategy study.
- Same-day exit (`T-0`) is only allowed for `AMC` events; `BMO` / `UNKNOWN` are excluded as unsafe.

## C. Scoreboard
Accepted live-mode candidate trades:
{scoreboard_md}

Ungated mechanically priceable simulations:
{simulated_scoreboard_md}

## D. Signal Quality
Top univariate rankings are based on priceable 7D/-1D baseline trades, using Spearman correlation to IV change and 50%-spread adjusted return.

{rankings_md}

## E. Key Findings
- Baseline call-calendar mid vs conservative results:
  - Mid: {base_mid['total_pnl'].iloc[0]:.2f} total P&L on {int(base_mid['realized_count'].iloc[0])} realized trades.
  - Cross 100: {base_adj['total_pnl'].iloc[0]:.2f} total P&L on {int(base_adj['realized_count'].iloc[0])} realized trades.
- Ungated baseline call-calendar simulation:
  - Mid: {sim_base_mid['total_pnl'].iloc[0]:.2f} total P&L on {int(sim_base_mid['simulated_priceable_count'].iloc[0])} priceable trades.
  - Cross 100: {sim_base_adj['total_pnl'].iloc[0]:.2f} total P&L on {int(sim_base_adj['simulated_priceable_count'].iloc[0])} priceable trades.
- Best tested adjusted scenario/structure:
  - {best_adj_row['structure']} / {best_adj_row['scenario']} / total P&L {best_adj_row['total_pnl']:.2f} / Sharpe {best_adj_row['return_based_sharpe']:.3f}
- Best ungated adjusted scenario/structure:
  - {best_sim_adj_row['structure']} / {best_sim_adj_row['scenario']} / total P&L {best_sim_adj_row['total_pnl']:.2f} / Sharpe {best_sim_adj_row['return_based_sharpe']:.3f}
- Missing exact exits are always explicit; they are never backfilled with substitute contracts.
- Sector breakdown is unavailable from local offline data, so the study reports symbol, regime, liquidity, DTE, and release-timing breakdowns instead.

## F. Caveats
- Historical price-based RV features use the local OHLCV store; this reproduces YZ/HAR style inputs without external network calls.
- Ticker-tier / market-cap confidence calibration from the live UI is not reproduced historically because no local point-in-time market-cap store is available.
- The engine's current `IV Expansion Candidate` implementation does not define a single execution rule across 5-10 DTE; this study treats 7 DTE as baseline and reports fixed-window robustness separately.

## G. Output Files
- Trade log: `{results['artifacts']['trade_csv']}`
- Summary JSON: `{results['artifacts']['summary_json']}`
- Scoreboard CSV: `{results['artifacts']['scoreboard_csv']}`
- Simulated scoreboard CSV: `{results['artifacts']['simulated_scoreboard_csv']}`
- Signal rankings CSV: `{results['artifacts']['signal_csv']}`
"""
    return memo


def run_study(config: StudyConfig, logger: logging.Logger) -> Dict[str, Any]:
    events = load_events(config, logger)
    cache = DataCache(config)
    records: List[TradeRecord] = []
    history: Dict[Tuple[str, str, int, int], List[HistoryOutcome]] = {}
    signal_cache: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}

    for event in events.itertuples(index=False):
        event_series = pd.Series(event._asdict())
        symbol = str(event_series["symbol"])
        price_history = cache.load_price_history(symbol)
        if price_history is None or price_history.empty:
            for structure in STRUCTURES.values():
                for entry_offset in config.entry_offsets:
                    for exit_offset in config.exit_offsets:
                        records.append(
                            TradeRecord(
                                symbol=symbol,
                                event_date=pd.Timestamp(event_series["event_date"]).strftime("%Y-%m-%d"),
                                release_timing=str(event_series["release_timing"]),
                                structure=structure.name,
                                entry_offset_bdays=entry_offset,
                                exit_offset_bdays=exit_offset,
                                status=STATUS_MISSING_PRICE_HISTORY,
                                exclusion_reason="missing_price_history",
                                candidate_signal_pass=False,
                                entry_date=None,
                                exit_date=None,
                                accepted_for_entry=False,
                                exact_exit_quote_found=False,
                                priceable_realized=False,
                                wf_history_count=0,
                                wf_expected_iv_change=None,
                                wf_expected_pnl_mid=None,
                                wf_expected_pnl_adjusted=None,
                                wf_ranking_score=None,
                                signal_iv_rv30=None,
                                signal_iv_rv_har=None,
                                signal_term_structure_slope=None,
                                signal_near_back_iv_ratio=None,
                                signal_rv30=None,
                                signal_rv_har_forecast=None,
                                signal_rv_percentile_rank=None,
                                signal_vol_regime=None,
                                signal_near_term_dte=None,
                                signal_near_term_spread_pct=None,
                                signal_near_term_liquidity_proxy=None,
                                signal_event_implied_move_pct=None,
                                signal_non_event_move_pct=None,
                                signal_implied_move_pct=None,
                                signal_move_anchor_pct=None,
                                signal_historical_p90_move_pct=None,
                                signal_move_risk_level=None,
                                signal_move_risk_ratio=None,
                                signal_smile_curvature=None,
                                signal_smile_points=None,
                                signal_front_oi=None,
                                signal_back_oi=None,
                                signal_fail_reasons="missing_price_history",
                                leg1_option_type=None,
                                leg1_expiry=None,
                                leg1_strike=None,
                                leg1_option_symbol=None,
                                leg1_option_id=None,
                                leg1_side=None,
                                leg1_bid_entry=None,
                                leg1_ask_entry=None,
                                leg1_mid_entry=None,
                                leg1_bid_exit=None,
                                leg1_ask_exit=None,
                                leg1_mid_exit=None,
                                leg1_iv_entry=None,
                                leg1_iv_exit=None,
                                leg1_iv_change=None,
                                leg1_spread_entry=None,
                                leg1_spread_exit=None,
                                leg1_open_interest_entry=None,
                                leg1_volume_entry=None,
                                leg2_option_type=None,
                                leg2_expiry=None,
                                leg2_strike=None,
                                leg2_option_symbol=None,
                                leg2_option_id=None,
                                leg2_side=None,
                                leg2_bid_entry=None,
                                leg2_ask_entry=None,
                                leg2_mid_entry=None,
                                leg2_bid_exit=None,
                                leg2_ask_exit=None,
                                leg2_mid_exit=None,
                                leg2_iv_entry=None,
                                leg2_iv_exit=None,
                                leg2_iv_change=None,
                                leg2_spread_entry=None,
                                leg2_spread_exit=None,
                                leg2_open_interest_entry=None,
                                leg2_volume_entry=None,
                                avg_iv_change=None,
                                entry_value_mid=None,
                                exit_value_mid=None,
                                mid_pnl_per_spread=None,
                                debit_capital_per_spread=None,
                                return_mid_pct=None,
                                pnl_mid=None,
                                entry_half_spread_cost=None,
                                exit_half_spread_cost=None,
                                pnl_cross_25=None,
                                pnl_cross_50=None,
                                pnl_cross_100=None,
                                pnl_cross_100_commission=None,
                                return_cross_25_pct=None,
                                return_cross_50_pct=None,
                                return_cross_100_pct=None,
                                return_cross_100_commission_pct=None,
                            )
                        )
            continue

        prior_symbol_events = events[(events["symbol"] == symbol) & (events["event_date"] < event_series["event_date"])].copy()
        event_series["prior_symbol_events"] = prior_symbol_events

        for entry_offset in config.entry_offsets:
            entry_date = previous_business_day(price_history, pd.Timestamp(event_series["event_date"]), entry_offset)
            if entry_date is None:
                for structure in STRUCTURES.values():
                    wf = walk_forward_summary(history.get((symbol, structure.name, entry_offset, BASE_EXIT_OFFSET), []))
                    records.append(
                        trade_to_record(
                            symbol=symbol,
                            event=event_series,
                            structure=structure,
                            entry_offset=entry_offset,
                            exit_offset=BASE_EXIT_OFFSET,
                            status=STATUS_ENTRY_DATE_UNAVAILABLE,
                            exclusion_reason="entry_date_unavailable",
                            signal_snapshot={"signal_ready": False, "signal_fail_reasons": ["entry_date_unavailable"]},
                            wf_summary=wf,
                            entry_date=None,
                            exit_date=None,
                            structure_trade=None,
                            config=config,
                        )
                    )
                continue

            for exit_offset in config.exit_offsets:
                if exit_offset == 0 and config.exit_same_day_amc_only and str(event_series["release_timing"]) != "after market close":
                    for structure in STRUCTURES.values():
                        wf = walk_forward_summary(history.get((symbol, structure.name, entry_offset, exit_offset), []))
                        records.append(
                            trade_to_record(
                                symbol=symbol,
                                event=event_series,
                                structure=structure,
                                entry_offset=entry_offset,
                                exit_offset=exit_offset,
                                status=STATUS_UNSAFE_EXIT,
                                exclusion_reason="same_day_exit_only_safe_for_amc",
                                signal_snapshot={"signal_ready": False, "signal_fail_reasons": ["unsafe_same_day_exit"]},
                                wf_summary=wf,
                                entry_date=entry_date,
                                exit_date=None,
                                structure_trade=None,
                                config=config,
                            )
                        )
                    continue

                if exit_offset == 0:
                    exit_date = pd.Timestamp(event_series["event_date"]).normalize()
                else:
                    exit_date = previous_business_day(price_history, pd.Timestamp(event_series["event_date"]), exit_offset)
                if exit_date is None or entry_date >= exit_date:
                    for structure in STRUCTURES.values():
                        wf = walk_forward_summary(history.get((symbol, structure.name, entry_offset, exit_offset), []))
                        records.append(
                            trade_to_record(
                                symbol=symbol,
                                event=event_series,
                                structure=structure,
                                entry_offset=entry_offset,
                                exit_offset=exit_offset,
                                status=STATUS_EXIT_DATE_UNAVAILABLE,
                                exclusion_reason="exit_date_unavailable",
                                signal_snapshot={"signal_ready": False, "signal_fail_reasons": ["exit_date_unavailable"]},
                                wf_summary=wf,
                                entry_date=entry_date,
                                exit_date=exit_date,
                                structure_trade=None,
                                config=config,
                            )
                        )
                    continue

                signal_key = (symbol, pd.Timestamp(event_series["event_date"]).strftime("%Y-%m-%d"), entry_offset, exit_offset)
                if signal_key not in signal_cache:
                    day_chain = cache.load_day_chain(symbol, entry_date)
                    if day_chain is None:
                        signal_cache[signal_key] = {"signal_ready": False, "signal_fail_reasons": ["missing_entry_chain"]}
                    else:
                        signal_cache[signal_key] = build_signal_snapshot(
                            symbol=symbol,
                            event=event_series,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            day_chain=day_chain,
                            price_history=price_history,
                        )
                signal_snapshot = signal_cache[signal_key]
                entry_chain = cache.load_day_chain(symbol, entry_date)
                exit_chain = cache.load_day_chain(symbol, exit_date)

                for structure in STRUCTURES.values():
                    wf_key = (symbol, structure.name, entry_offset, exit_offset)
                    wf_summary = walk_forward_summary(history.get(wf_key, []))
                    trade: Optional[StructureTrade] = None
                    reason = ""
                    if entry_chain is not None and exit_chain is not None:
                        trade, reason = realize_structure_trade(
                            structure=structure,
                            symbol=symbol,
                            event=event_series,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            entry_chain=entry_chain,
                            exit_chain=exit_chain,
                            config=config,
                        )
                    if not signal_snapshot.get("signal_ready"):
                        record = trade_to_record(
                            symbol=symbol,
                            event=event_series,
                            structure=structure,
                            entry_offset=entry_offset,
                            exit_offset=exit_offset,
                            status=STATUS_SIGNAL_REJECTED,
                            exclusion_reason="signal_inputs_unavailable",
                            signal_snapshot=signal_snapshot,
                            wf_summary=wf_summary,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            structure_trade=trade,
                            config=config,
                        )
                        records.append(record)
                        if trade is not None:
                            history.setdefault(wf_key, []).append(history_outcome_from_trade(trade, config))
                        continue

                    combined_signal_failures = list(signal_snapshot.get("signal_fail_reasons", []))
                    if wf_summary["wf_history_count"] < config.min_history_count:
                        combined_signal_failures.append("insufficient_walk_forward_history")
                    if wf_summary["wf_expected_pnl_adjusted"] is not None and wf_summary["wf_expected_pnl_adjusted"] <= 0:
                        combined_signal_failures.append("wf_expected_adjusted_pnl_non_positive")
                    signal_snapshot_for_trade = dict(signal_snapshot)
                    signal_snapshot_for_trade["signal_fail_reasons"] = combined_signal_failures

                    if combined_signal_failures:
                        exclusion_parts = list(combined_signal_failures)
                        if trade is None and reason:
                            exclusion_parts.append(f"sim_{reason}")
                        record = trade_to_record(
                            symbol=symbol,
                            event=event_series,
                            structure=structure,
                            entry_offset=entry_offset,
                            exit_offset=exit_offset,
                            status=STATUS_SIGNAL_REJECTED,
                            exclusion_reason=";".join(exclusion_parts),
                            signal_snapshot=signal_snapshot_for_trade,
                            wf_summary=wf_summary,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            structure_trade=trade,
                            config=config,
                        )
                        records.append(record)
                        if trade is not None:
                            history.setdefault(wf_key, []).append(history_outcome_from_trade(trade, config))
                        continue
                    status = STATUS_REALIZED
                    if trade is None:
                        if reason == "missing_exit_chain" or reason == "missing_exact_exit_quote":
                            status = STATUS_MISSING_EXIT_QUOTE
                        elif reason == "negative_or_zero_debit":
                            status = STATUS_NEGATIVE_DEBIT
                        elif reason.startswith("missing_entry"):
                            status = STATUS_MISSING_ENTRY_CHAIN
                        elif reason.startswith("leg_") or "no_" in reason:
                            status = STATUS_STRUCTURAL_INVALID
                        else:
                            status = STATUS_STRUCTURAL_INVALID
                    record = trade_to_record(
                        symbol=symbol,
                        event=event_series,
                        structure=structure,
                        entry_offset=entry_offset,
                        exit_offset=exit_offset,
                        status=status,
                        exclusion_reason=reason,
                        signal_snapshot=signal_snapshot_for_trade,
                        wf_summary=wf_summary,
                        entry_date=entry_date,
                        exit_date=exit_date,
                        structure_trade=trade,
                        config=config,
                    )
                    records.append(record)
                    if trade is not None:
                        history.setdefault(wf_key, []).append(history_outcome_from_trade(trade, config))

    trades = pd.DataFrame([asdict(record) for record in records])
    base_trades = trades[
        (trades["entry_offset_bdays"] == BASE_ENTRY_OFFSET)
        & (trades["exit_offset_bdays"] == BASE_EXIT_OFFSET)
    ].copy()

    scoreboard = summarize_scoreboard(base_trades)
    simulated_scoreboard = summarize_simulated_scoreboard(base_trades)
    signal_rankings = feature_rankings(base_trades)

    robustness_rows: List[Dict[str, Any]] = []
    for structure in BASE_STRUCTURE_ROWS:
        for entry_offset in config.entry_offsets:
            for exit_offset in config.exit_offsets:
                subset = trades[
                    (trades["structure"] == structure)
                    & (trades["entry_offset_bdays"] == entry_offset)
                    & (trades["exit_offset_bdays"] == exit_offset)
                ].copy()
                if subset.empty:
                    continue
                summary = summarize_group(subset, "cross_50", label=f"{structure}:{entry_offset}:{exit_offset}")
                summary["structure"] = structure
                summary["entry_offset_bdays"] = entry_offset
                summary["exit_offset_bdays"] = exit_offset
                robustness_rows.append(summary)
    robustness = pd.DataFrame(robustness_rows)

    symbol_breakdown = breakdown_table(base_trades, "cross_50", ["structure", "symbol"])
    release_breakdown = breakdown_table(base_trades, "cross_50", ["structure", "release_timing"])

    base_trades["liquidity_tier"] = base_trades["signal_near_term_liquidity_proxy"].map(liquidity_tier)
    base_trades["dte_bucket"] = base_trades["signal_near_term_dte"].map(dte_bucket)
    base_trades["vol_regime_bucket"] = base_trades["signal_vol_regime"].fillna("unknown")
    liquidity_breakdown = breakdown_table(base_trades, "cross_50", ["structure", "liquidity_tier"])
    dte_breakdown = breakdown_table(base_trades, "cross_50", ["structure", "dte_bucket"])
    vol_breakdown = breakdown_table(base_trades, "cross_50", ["structure", "vol_regime_bucket"])

    time_splits = []
    split_defs = [
        ("2023_2024", "2023-01-01", "2024-12-31"),
        ("2025_2026", "2025-01-01", config.end_date.strftime("%Y-%m-%d")),
    ]
    for name, start_text, end_text in split_defs:
        split_start = pd.Timestamp(start_text)
        split_end = pd.Timestamp(end_text)
        split_df = base_trades[
            (pd.to_datetime(base_trades["event_date"]) >= split_start)
            & (pd.to_datetime(base_trades["event_date"]) <= split_end)
        ].copy()
        for structure in BASE_STRUCTURE_ROWS:
            subset = split_df[split_df["structure"] == structure].copy()
            if subset.empty:
                continue
            summary = summarize_group(subset, "cross_50", label=f"{name}:{structure}")
            summary["split"] = name
            summary["structure"] = structure
            time_splits.append(summary)
    time_split_df = pd.DataFrame(time_splits)

    strict_rows = []
    strict_subset = strict_liquidity_filter(base_trades, config)
    for structure in BASE_STRUCTURE_ROWS:
        subset = strict_subset[strict_subset["structure"] == structure].copy()
        if subset.empty:
            continue
        summary = summarize_group(subset, "cross_50", label=f"strict:{structure}")
        summary["structure"] = structure
        strict_rows.append(summary)
    strict_liquidity = pd.DataFrame(strict_rows)

    return {
        "trades": trades,
        "base_trades": base_trades,
        "scoreboard": scoreboard,
        "simulated_scoreboard": simulated_scoreboard,
        "signal_rankings": signal_rankings,
        "robustness": robustness,
        "symbol_breakdown": symbol_breakdown,
        "release_breakdown": release_breakdown,
        "liquidity_breakdown": liquidity_breakdown,
        "dte_breakdown": dte_breakdown,
        "vol_breakdown": vol_breakdown,
        "time_splits": time_split_df,
        "strict_liquidity": strict_liquidity,
    }


def write_outputs(results: Dict[str, Any], config: StudyConfig) -> Dict[str, str]:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.output_root / f"iv_expansion_study_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trade_csv = out_dir / "iv_expansion_trade_log.csv"
    summary_json = out_dir / "iv_expansion_summary.json"
    scoreboard_csv = out_dir / "iv_expansion_scoreboard.csv"
    simulated_scoreboard_csv = out_dir / "iv_expansion_simulated_scoreboard.csv"
    signal_csv = out_dir / "iv_expansion_signal_rankings.csv"
    robustness_csv = out_dir / "iv_expansion_robustness.csv"
    memo_md = out_dir / "iv_expansion_research_memo.md"
    symbol_csv = out_dir / "iv_expansion_symbol_breakdown.csv"
    release_csv = out_dir / "iv_expansion_release_breakdown.csv"
    dte_csv = out_dir / "iv_expansion_dte_breakdown.csv"
    liquidity_csv = out_dir / "iv_expansion_liquidity_breakdown.csv"
    vol_csv = out_dir / "iv_expansion_vol_breakdown.csv"
    split_csv = out_dir / "iv_expansion_time_splits.csv"

    results["trades"].to_csv(trade_csv, index=False)
    results["scoreboard"].to_csv(scoreboard_csv, index=False)
    results["simulated_scoreboard"].to_csv(simulated_scoreboard_csv, index=False)
    results["signal_rankings"].to_csv(signal_csv, index=False)
    results["robustness"].to_csv(robustness_csv, index=False)
    results["symbol_breakdown"].to_csv(symbol_csv, index=False)
    results["release_breakdown"].to_csv(release_csv, index=False)
    results["dte_breakdown"].to_csv(dte_csv, index=False)
    results["liquidity_breakdown"].to_csv(liquidity_csv, index=False)
    results["vol_breakdown"].to_csv(vol_csv, index=False)
    results["time_splits"].to_csv(split_csv, index=False)

    artifacts = {
        "output_dir": str(out_dir),
        "trade_csv": str(trade_csv),
        "summary_json": str(summary_json),
        "scoreboard_csv": str(scoreboard_csv),
        "simulated_scoreboard_csv": str(simulated_scoreboard_csv),
        "signal_csv": str(signal_csv),
        "robustness_csv": str(robustness_csv),
        "memo_md": str(memo_md),
        "symbol_breakdown_csv": str(symbol_csv),
        "release_breakdown_csv": str(release_csv),
        "dte_breakdown_csv": str(dte_csv),
        "liquidity_breakdown_csv": str(liquidity_csv),
        "vol_breakdown_csv": str(vol_csv),
        "time_splits_csv": str(split_csv),
    }

    summary = {
        "assumptions": sanitize_for_json(asdict(config)),
        "scoreboard": sanitize_for_json(results["scoreboard"].to_dict(orient="records")),
        "simulated_scoreboard": sanitize_for_json(results["simulated_scoreboard"].to_dict(orient="records")),
        "signal_rankings": sanitize_for_json(results["signal_rankings"].to_dict(orient="records")),
        "robustness": sanitize_for_json(results["robustness"].to_dict(orient="records")),
        "artifacts": artifacts,
    }
    summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False))

    results["artifacts"] = artifacts
    memo_md.write_text(build_memo(config, results, out_dir))
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the IV Expansion Candidate mode")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--options-root", default=str(DEFAULT_OPTIONS_ROOT))
    parser.add_argument("--price-root", default=str(DEFAULT_PRICE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> StudyConfig:
    return StudyConfig(
        db_path=Path(args.db_path),
        options_root=Path(args.options_root),
        price_root=Path(args.price_root),
        output_root=Path(args.output_root),
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
    )


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    logger = build_logger(args.verbose)
    logger.info("Running IV expansion historical study")
    results = run_study(config, logger)
    artifacts = write_outputs(results, config)
    logger.info("Trade log: %s", artifacts["trade_csv"])
    logger.info("Summary JSON: %s", artifacts["summary_json"])
    logger.info("Research memo: %s", artifacts["memo_md"])


if __name__ == "__main__":
    main()
