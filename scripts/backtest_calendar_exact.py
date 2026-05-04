"""
Mechanically exact same-strike earnings calendar backtest.

This script rebuilds the calendar spread backtest from first principles:
  1. ingest candidate earnings events / signal inputs
  2. select entry contracts on the configured entry date
  3. persist exact held contract identity
  4. lookup the same contracts on the configured exit date
  5. price each trade with explicit mid and conservative assumptions
  6. aggregate portfolio capital usage and equity
  7. emit auditable trade-level outputs and run-level summaries
  8. validate invariants and fail hard on impossible "priceable" states

The implementation intentionally has no silent contract substitution.
If the exact held contracts are missing at exit, the trade is flagged
MISSING_EXIT_QUOTE and excluded from realized P&L.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.logger import setup_logger


DEFAULT_DB_PATH = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
DEFAULT_PARQUET_ROOT = Path("/Volumes/T9/market_data/research/options_features_eod")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "exports" / "reports"
DEFAULT_PRIOR_REPORT = (
    Path.home() / ".options_calculator_pro" / "reports" / "backtest_backtest_v1_20260417T080508Z.json"
)

STATUS_REALIZED = "REALIZED"
STATUS_MISSING_EXIT_QUOTE = "MISSING_EXIT_QUOTE"
STATUS_EXCLUDED_NO_PRIOR_THRESHOLD = "EXCLUDED_NO_PRIOR_THRESHOLD"
STATUS_EXCLUDED_NBR_BELOW_THRESHOLD = "EXCLUDED_NBR_BELOW_THRESHOLD"
STATUS_EXCLUDED_NO_SCREEN_CHAIN = "EXCLUDED_NO_SCREEN_CHAIN"
STATUS_EXCLUDED_SCREEN_FILTER = "EXCLUDED_SCREEN_FILTER"
STATUS_EXCLUDED_NO_ENTRY_CHAIN = "EXCLUDED_NO_ENTRY_CHAIN"
STATUS_EXCLUDED_NO_ENTRY_LEG = "EXCLUDED_NO_ENTRY_LEG"
STATUS_EXCLUDED_STRUCTURAL_INVALIDITY = "EXCLUDED_STRUCTURAL_INVALIDITY"
STATUS_EXCLUDED_NEGATIVE_DEBIT = "EXCLUDED_NEGATIVE_DEBIT"
STATUS_EXCLUDED_ENTRY_AMBIGUOUS = "EXCLUDED_ENTRY_AMBIGUOUS"


class BacktestInvariantError(RuntimeError):
    """Raised when the backtest violates a non-negotiable mechanical invariant."""


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Explicit execution settings for conservative repricing."""

    half_spread_fraction: float = 1.0
    slippage_per_leg: float = 0.0
    commission_per_contract: float = 0.0


@dataclass(frozen=True)
class BacktestConfig:
    """Run configuration."""

    db_path: Path = DEFAULT_DB_PATH
    parquet_root: Path = DEFAULT_PARQUET_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    prior_report_path: Optional[Path] = DEFAULT_PRIOR_REPORT
    quality_tiers: Tuple[str, ...] = ("A", "B")
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    option_type: str = "C"
    same_strike_only: bool = True
    contract_multiplier: int = 100
    spreads_per_trade: int = 2
    screen_offset_bdays: int = 1
    entry_offset_bdays: int = 2
    exit_offset_bdays: int = 1
    atm_moneyness_window: float = 5.0
    min_front_oi: Optional[int] = None
    min_back_oi: Optional[int] = None
    max_spread_pct: Optional[float] = None
    walk_forward_pctile: float = 0.60
    min_nbr: Optional[float] = None
    threshold_cold_start_policy: str = "exclude"
    exclude_negative_debit: bool = True
    daily_equity_enabled: bool = True
    conservative_execution: ExecutionAssumptions = ExecutionAssumptions()


@dataclass(frozen=True)
class ContractIdentity:
    """Stable contract identity persisted across entry and exit."""

    symbol: str
    option_type: str
    strike: float
    expiry: str
    option_symbol: Optional[str]
    option_id: Optional[int]


@dataclass(frozen=True)
class ContractSnapshot:
    """Snapshot for one contract on one date."""

    identity: ContractIdentity
    quote_date: str
    bid: float
    ask: float
    mid: float
    spread: float
    spread_pct: float
    volume: int
    open_interest: int
    moneyness_pct: float
    underlying_price: float


@dataclass(frozen=True)
class EntrySelection:
    """Persisted exact entry contracts for a trade."""

    symbol: str
    event_date: str
    release_timing: str
    signal_nbr: float
    threshold_value: Optional[float]
    entry_date: str
    exit_date: str
    front_expiry: str
    back_expiry: str
    front: ContractSnapshot
    back: ContractSnapshot
    entry_debit_mid: float
    entry_debit_adjusted: float
    entry_penalty: float
    debit_capital: float
    actual_entry_cashflow: float
    entry_offset_bdays: int
    exit_offset_bdays: int
    screen_date: Optional[str] = None
    screen_front_oi: Optional[int] = None
    screen_back_oi: Optional[int] = None
    screen_front_spread_pct: Optional[float] = None
    screen_back_spread_pct: Optional[float] = None


@dataclass
class TradeOutcome:
    """Full trade record emitted to CSV and summary outputs."""

    status: str
    exclusion_reason: str
    symbol: str
    event_date: str
    release_timing: str
    signal_nbr: Optional[float]
    threshold_value: Optional[float]
    option_type: str
    screen_date: Optional[str]
    screen_front_oi: Optional[int]
    screen_back_oi: Optional[int]
    screen_front_spread_pct: Optional[float]
    screen_back_spread_pct: Optional[float]
    entry_date: str
    exit_date: str
    entry_offset_bdays: Optional[int]
    exit_offset_bdays: Optional[int]
    front_expiry: Optional[str]
    back_expiry: Optional[str]
    front_option_symbol: Optional[str]
    back_option_symbol: Optional[str]
    front_option_id: Optional[int]
    back_option_id: Optional[int]
    strike: Optional[float]
    front_bid_entry: Optional[float]
    front_ask_entry: Optional[float]
    front_mid_entry: Optional[float]
    back_bid_entry: Optional[float]
    back_ask_entry: Optional[float]
    back_mid_entry: Optional[float]
    entry_front_spread: Optional[float]
    entry_back_spread: Optional[float]
    front_bid_exit: Optional[float]
    front_ask_exit: Optional[float]
    front_mid_exit: Optional[float]
    back_bid_exit: Optional[float]
    back_ask_exit: Optional[float]
    back_mid_exit: Optional[float]
    exit_front_spread: Optional[float]
    exit_back_spread: Optional[float]
    entry_debit_mid: Optional[float]
    entry_debit_adjusted: Optional[float]
    exit_value_mid: Optional[float]
    exit_value_adjusted: Optional[float]
    realized_pnl_mid: Optional[float]
    realized_pnl_adjusted: Optional[float]
    return_on_capital_mid: Optional[float]
    return_on_capital_adjusted: Optional[float]
    debit_capital: Optional[float]
    actual_entry_cashflow: Optional[float]
    exact_exit_quote_found: bool
    front_exact_match: bool
    back_exact_match: bool
    timing_validated: bool


def connect_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    """
    Open SQLite in immutable read-only mode when possible.

    This avoids sandbox/file-lock edge cases and guarantees the backtest
    cannot mutate the source database.
    """
    uri = f"file:{db_path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def build_logger(verbose: bool) -> logging.Logger:
    level = "DEBUG" if verbose else "INFO"
    return setup_logger("scripts.backtest_calendar_exact", level=level, file_output=False)


def parse_date(text: Optional[str]) -> Optional[pd.Timestamp]:
    if not text:
        return None
    return pd.Timestamp(text)


def bday_offset_from_event(event_date: pd.Timestamp, target_date: pd.Timestamp) -> int:
    return len(pd.bdate_range(target_date, event_date)) - 1 if target_date <= event_date else len(pd.bdate_range(event_date, target_date)) - 1


def compute_entry_date(event_date: pd.Timestamp, offset_bdays: int) -> pd.Timestamp:
    return event_date - pd.offsets.BDay(offset_bdays)


def compute_exit_date(event_date: pd.Timestamp, offset_bdays: int) -> pd.Timestamp:
    return event_date + pd.offsets.BDay(offset_bdays)


def load_candidate_events(config: BacktestConfig, logger: logging.Logger) -> pd.DataFrame:
    """Load candidate earnings events from the institutional labels table."""
    placeholders = ",".join("?" for _ in config.quality_tiers)
    sql = f"""
        SELECT symbol,
               event_date,
               release_timing,
               front_expiry,
               back_expiry,
               pre_front_iv,
               pre_back_iv,
               quality_tier
        FROM earnings_iv_decay_labels
        WHERE quality_tier IN ({placeholders})
        ORDER BY event_date, symbol
    """
    with connect_readonly_sqlite(config.db_path) as conn:
        events = pd.read_sql(sql, conn, params=list(config.quality_tiers))

    for col in ["event_date", "front_expiry", "back_expiry"]:
        events[col] = pd.to_datetime(events[col])
    events["signal_nbr"] = events["pre_front_iv"] / events["pre_back_iv"]
    events["event_year"] = events["event_date"].dt.year

    if config.start_date is not None:
        events = events[events["event_date"] >= config.start_date]
    if config.end_date is not None:
        events = events[events["event_date"] <= config.end_date]

    logger.info(
        "Loaded %d candidate events across %d symbols",
        len(events),
        events["symbol"].nunique(),
    )
    return events.reset_index(drop=True)


def compute_walk_forward_thresholds(
    events: pd.DataFrame,
    pctile: float,
    cold_start_policy: str,
) -> Dict[int, Optional[float]]:
    """Compute yearly walk-forward thresholds from strictly prior years only."""
    thresholds: Dict[int, Optional[float]] = {}
    years = sorted(events["event_year"].unique().tolist())
    for year in years:
        prior = events[events["event_year"] < year]
        if prior.empty:
            if cold_start_policy == "exclude":
                thresholds[year] = None
            else:
                raise ValueError(f"Unsupported cold-start policy: {cold_start_policy}")
            continue
        thresholds[year] = float(prior["signal_nbr"].quantile(pctile))
    return thresholds


def build_parquet_path(parquet_root: Path, symbol: str, dt: pd.Timestamp) -> Path:
    yr = f"{dt.year}"
    mo = f"{dt.month:02d}"
    return (
        parquet_root
        / f"underlying_symbol={symbol}"
        / f"year={yr}"
        / f"month={mo}"
        / f"{symbol.lower()}_options_features_eod_{yr}-{mo}.parquet"
    )


def load_day_chain(
    cache: Dict[Tuple[str, str], Optional[pd.DataFrame]],
    parquet_root: Path,
    symbol: str,
    quote_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """Load one symbol/day chain from parquet."""
    key = (symbol, quote_date.strftime("%Y-%m-%d"))
    if key in cache:
        return cache[key]

    path = build_parquet_path(parquet_root, symbol, quote_date)
    if not path.exists():
        cache[key] = None
        return None

    chain = pd.read_parquet(path)
    chain["trade_date"] = pd.to_datetime(chain["trade_date"])
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    day = chain[
        (chain["trade_date"] == quote_date)
        & (chain["quote_quality_flag"] == "ok")
        & (chain["bid"] > 0)
    ].copy()
    cache[key] = day if len(day) else None
    return cache[key]


def option_candidates(
    chain: pd.DataFrame,
    expiry: pd.Timestamp,
    option_type: str,
    atm_window: float,
) -> pd.DataFrame:
    """Filter a chain down to the candidate contracts for one leg."""
    return chain[
        (chain["expiry"] == expiry)
        & (chain["call_put"] == option_type)
        & (chain["moneyness_pct"].abs() <= atm_window)
    ].copy()


def choose_same_strike_pair(front_candidates: pd.DataFrame, back_candidates: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], str]:
    """
    Pick the same-strike pair that is closest to ATM across both expiries.

    Returns the selected front/back rows and a reason string on failure.
    """
    common_strikes = sorted(set(front_candidates["strike"]).intersection(set(back_candidates["strike"])))
    if not common_strikes:
        return None, None, "no_same_strike_pair_within_atm_window"

    best: Optional[Tuple[pd.Series, pd.Series]] = None
    best_score: Optional[float] = None
    for strike in common_strikes:
        front_rows = front_candidates[front_candidates["strike"] == strike]
        back_rows = back_candidates[back_candidates["strike"] == strike]
        front_row = front_rows.loc[front_rows["moneyness_pct"].abs().idxmin()]
        back_row = back_rows.loc[back_rows["moneyness_pct"].abs().idxmin()]
        score = float(abs(front_row["moneyness_pct"]) + abs(back_row["moneyness_pct"]))
        if best_score is None or score < best_score:
            best_score = score
            best = (front_row, back_row)

    assert best is not None
    return best[0], best[1], ""


def build_contract_snapshot(row: pd.Series, quote_date: pd.Timestamp, symbol: str) -> ContractSnapshot:
    option_id = None if pd.isna(row.get("option_id")) else int(row["option_id"])
    identity = ContractIdentity(
        symbol=symbol,
        option_type=str(row["call_put"]),
        strike=float(row["strike"]),
        expiry=pd.Timestamp(row["expiry"]).strftime("%Y-%m-%d"),
        option_symbol=str(row.get("option_symbol")) if row.get("option_symbol") is not None else None,
        option_id=option_id,
    )
    return ContractSnapshot(
        identity=identity,
        quote_date=quote_date.strftime("%Y-%m-%d"),
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        mid=float(row["mid"]),
        spread=float(row["spread"]),
        spread_pct=float(row["spread_pct"]),
        volume=int(row.get("volume", 0)),
        open_interest=int(row.get("open_interest", 0)),
        moneyness_pct=float(row["moneyness_pct"]),
        underlying_price=float(row["underlying_price"]),
    )


def conservative_entry_value(front: ContractSnapshot, back: ContractSnapshot, assumptions: ExecutionAssumptions) -> Tuple[float, float]:
    """Entry debit under explicit conservative execution assumptions."""
    penalty = assumptions.half_spread_fraction * ((front.spread / 2.0) + (back.spread / 2.0))
    penalty += assumptions.slippage_per_leg * 2.0
    adjusted = (back.mid - front.mid) + penalty
    return adjusted, penalty


def conservative_exit_value(front: ContractSnapshot, back: ContractSnapshot, assumptions: ExecutionAssumptions) -> Tuple[float, float]:
    """Exit value under explicit conservative execution assumptions."""
    penalty = assumptions.half_spread_fraction * ((front.spread / 2.0) + (back.spread / 2.0))
    penalty += assumptions.slippage_per_leg * 2.0
    adjusted = (back.mid - front.mid) - penalty
    return adjusted, penalty


def build_trade_outcome(
    status: str,
    exclusion_reason: str,
    candidate: pd.Series,
    config: BacktestConfig,
    threshold_value: Optional[float],
) -> TradeOutcome:
    """Construct a mostly-empty trade outcome row for excluded candidates."""
    event_date = pd.Timestamp(candidate["event_date"])
    entry_date = compute_entry_date(event_date, config.entry_offset_bdays)
    exit_date = compute_exit_date(event_date, config.exit_offset_bdays)
    return TradeOutcome(
        status=status,
        exclusion_reason=exclusion_reason,
        symbol=str(candidate["symbol"]),
        event_date=event_date.strftime("%Y-%m-%d"),
        release_timing=str(candidate.get("release_timing", "UNKNOWN")),
        signal_nbr=float(candidate["signal_nbr"]) if not pd.isna(candidate["signal_nbr"]) else None,
        threshold_value=threshold_value,
        option_type=config.option_type,
        screen_date=(event_date - pd.offsets.BDay(config.screen_offset_bdays)).strftime("%Y-%m-%d"),
        screen_front_oi=None,
        screen_back_oi=None,
        screen_front_spread_pct=None,
        screen_back_spread_pct=None,
        entry_date=entry_date.strftime("%Y-%m-%d"),
        exit_date=exit_date.strftime("%Y-%m-%d"),
        entry_offset_bdays=config.entry_offset_bdays,
        exit_offset_bdays=config.exit_offset_bdays,
        front_expiry=pd.Timestamp(candidate["front_expiry"]).strftime("%Y-%m-%d") if pd.notna(candidate["front_expiry"]) else None,
        back_expiry=pd.Timestamp(candidate["back_expiry"]).strftime("%Y-%m-%d") if pd.notna(candidate["back_expiry"]) else None,
        front_option_symbol=None,
        back_option_symbol=None,
        front_option_id=None,
        back_option_id=None,
        strike=None,
        front_bid_entry=None,
        front_ask_entry=None,
        front_mid_entry=None,
        back_bid_entry=None,
        back_ask_entry=None,
        back_mid_entry=None,
        entry_front_spread=None,
        entry_back_spread=None,
        front_bid_exit=None,
        front_ask_exit=None,
        front_mid_exit=None,
        back_bid_exit=None,
        back_ask_exit=None,
        back_mid_exit=None,
        exit_front_spread=None,
        exit_back_spread=None,
        entry_debit_mid=None,
        entry_debit_adjusted=None,
        exit_value_mid=None,
        exit_value_adjusted=None,
        realized_pnl_mid=None,
        realized_pnl_adjusted=None,
        return_on_capital_mid=None,
        return_on_capital_adjusted=None,
        debit_capital=None,
        actual_entry_cashflow=None,
        exact_exit_quote_found=False,
        front_exact_match=False,
        back_exact_match=False,
        timing_validated=False,
    )


def apply_screen_filters(
    candidate: pd.Series,
    config: BacktestConfig,
    chain_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]],
) -> Tuple[Optional[Dict[str, Any]], Optional[TradeOutcome]]:
    """Apply explicit T-1 liquidity screens relative to the actual event date."""
    if (
        config.min_front_oi is None
        and config.min_back_oi is None
        and config.max_spread_pct is None
    ):
        return None, None

    event_date = pd.Timestamp(candidate["event_date"])
    screen_date = event_date - pd.offsets.BDay(config.screen_offset_bdays)
    chain = load_day_chain(chain_cache, config.parquet_root, str(candidate["symbol"]), screen_date)
    threshold_value = candidate["threshold_value"]
    if chain is None:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_NO_SCREEN_CHAIN,
            "missing_screen_chain",
            candidate,
            config,
            threshold_value,
        )

    front_expiry = pd.Timestamp(candidate["front_expiry"])
    back_expiry = pd.Timestamp(candidate["back_expiry"])
    front_candidates = option_candidates(chain, front_expiry, config.option_type, config.atm_moneyness_window)
    back_candidates = option_candidates(chain, back_expiry, config.option_type, config.atm_moneyness_window)
    if front_candidates.empty or back_candidates.empty:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_SCREEN_FILTER,
            "missing_screen_leg_candidates",
            candidate,
            config,
            threshold_value,
        )

    if config.same_strike_only:
        front_row, back_row, reason = choose_same_strike_pair(front_candidates, back_candidates)
        if front_row is None or back_row is None:
            return None, build_trade_outcome(
                STATUS_EXCLUDED_STRUCTURAL_INVALIDITY,
                f"screen_{reason}",
                candidate,
                config,
                threshold_value,
            )
    else:
        front_row = front_candidates.loc[front_candidates["moneyness_pct"].abs().idxmin()]
        back_row = back_candidates.loc[back_candidates["moneyness_pct"].abs().idxmin()]

    front_oi = int(front_row["open_interest"])
    back_oi = int(back_row["open_interest"])
    front_spread_pct = float(front_row["spread_pct"])
    back_spread_pct = float(back_row["spread_pct"])

    if config.min_front_oi is not None and front_oi < config.min_front_oi:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_SCREEN_FILTER,
            "screen_front_oi_below_min",
            candidate,
            config,
            threshold_value,
        )
    if config.min_back_oi is not None and back_oi < config.min_back_oi:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_SCREEN_FILTER,
            "screen_back_oi_below_min",
            candidate,
            config,
            threshold_value,
        )
    if config.max_spread_pct is not None and front_spread_pct > config.max_spread_pct:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_SCREEN_FILTER,
            "screen_front_spread_above_max",
            candidate,
            config,
            threshold_value,
        )
    if config.max_spread_pct is not None and back_spread_pct > config.max_spread_pct:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_SCREEN_FILTER,
            "screen_back_spread_above_max",
            candidate,
            config,
            threshold_value,
        )

    return {
        "screen_date": screen_date.strftime("%Y-%m-%d"),
        "screen_front_oi": front_oi,
        "screen_back_oi": back_oi,
        "screen_front_spread_pct": front_spread_pct,
        "screen_back_spread_pct": back_spread_pct,
    }, None


def select_entry_contracts(
    candidate: pd.Series,
    config: BacktestConfig,
    chain_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]],
    logger: Optional[logging.Logger],
) -> Tuple[Optional[EntrySelection], Optional[TradeOutcome]]:
    """
    Select the exact entry contracts for one candidate.

    If the entry is invalid or structurally unavailable, returns an exclusion row.
    """
    threshold_value = candidate["threshold_value"]
    if threshold_value is None:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_NO_PRIOR_THRESHOLD,
            "no_prior_walk_forward_threshold",
            candidate,
            config,
            threshold_value,
        )

    if config.min_nbr is not None:
        passed_signal = float(candidate["signal_nbr"]) >= config.min_nbr
    else:
        passed_signal = float(candidate["signal_nbr"]) >= float(threshold_value)
    if not passed_signal:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_NBR_BELOW_THRESHOLD,
            "signal_nbr_below_threshold",
            candidate,
            config,
            threshold_value,
        )

    screen_snapshot, screen_exclusion = apply_screen_filters(candidate, config, chain_cache)
    if screen_exclusion is not None:
        return None, screen_exclusion

    event_date = pd.Timestamp(candidate["event_date"])
    entry_date = compute_entry_date(event_date, config.entry_offset_bdays)
    exit_date = compute_exit_date(event_date, config.exit_offset_bdays)
    chain = load_day_chain(chain_cache, config.parquet_root, str(candidate["symbol"]), entry_date)
    if chain is None:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_NO_ENTRY_CHAIN,
            "missing_entry_chain",
            candidate,
            config,
            threshold_value,
        )

    front_expiry = pd.Timestamp(candidate["front_expiry"])
    back_expiry = pd.Timestamp(candidate["back_expiry"])
    front_candidates = option_candidates(chain, front_expiry, config.option_type, config.atm_moneyness_window)
    back_candidates = option_candidates(chain, back_expiry, config.option_type, config.atm_moneyness_window)
    if front_candidates.empty or back_candidates.empty:
        return None, build_trade_outcome(
            STATUS_EXCLUDED_NO_ENTRY_LEG,
            "missing_entry_leg_candidates",
            candidate,
            config,
            threshold_value,
        )

    if config.same_strike_only:
        front_row, back_row, reason = choose_same_strike_pair(front_candidates, back_candidates)
        if front_row is None or back_row is None:
            return None, build_trade_outcome(
                STATUS_EXCLUDED_STRUCTURAL_INVALIDITY,
                reason,
                candidate,
                config,
                threshold_value,
            )
    else:
        front_row = front_candidates.loc[front_candidates["moneyness_pct"].abs().idxmin()]
        back_row = back_candidates.loc[back_candidates["moneyness_pct"].abs().idxmin()]
        reason = ""

    front = build_contract_snapshot(front_row, entry_date, str(candidate["symbol"]))
    back = build_contract_snapshot(back_row, entry_date, str(candidate["symbol"]))

    if config.same_strike_only and not math.isclose(front.identity.strike, back.identity.strike, rel_tol=0.0, abs_tol=1e-12):
        raise BacktestInvariantError(
            f"same-strike mode accepted mismatched entry strikes for {candidate['symbol']} {candidate['event_date']}"
        )

    entry_debit_mid = back.mid - front.mid
    entry_debit_adjusted, entry_penalty = conservative_entry_value(front, back, config.conservative_execution)
    debit_capital = entry_debit_mid * config.contract_multiplier * config.spreads_per_trade
    actual_entry_cashflow = entry_debit_mid * config.contract_multiplier * config.spreads_per_trade
    if config.exclude_negative_debit and entry_debit_mid <= 0:
        excluded = build_trade_outcome(
            STATUS_EXCLUDED_NEGATIVE_DEBIT,
            "entry_debit_non_positive",
            candidate,
            config,
            threshold_value,
        )
        excluded.strike = front.identity.strike if math.isclose(front.identity.strike, back.identity.strike) else None
        excluded.front_option_symbol = front.identity.option_symbol
        excluded.back_option_symbol = back.identity.option_symbol
        excluded.front_option_id = front.identity.option_id
        excluded.back_option_id = back.identity.option_id
        excluded.front_bid_entry = front.bid
        excluded.front_ask_entry = front.ask
        excluded.front_mid_entry = front.mid
        excluded.back_bid_entry = back.bid
        excluded.back_ask_entry = back.ask
        excluded.back_mid_entry = back.mid
        excluded.entry_front_spread = front.spread
        excluded.entry_back_spread = back.spread
        excluded.entry_debit_mid = entry_debit_mid
        excluded.entry_debit_adjusted = entry_debit_adjusted
        excluded.debit_capital = debit_capital
        excluded.actual_entry_cashflow = actual_entry_cashflow
        excluded.timing_validated = True
        return None, excluded

    selection = EntrySelection(
        symbol=str(candidate["symbol"]),
        event_date=event_date.strftime("%Y-%m-%d"),
        release_timing=str(candidate.get("release_timing", "UNKNOWN")),
        signal_nbr=float(candidate["signal_nbr"]),
        threshold_value=float(threshold_value),
        entry_date=entry_date.strftime("%Y-%m-%d"),
        exit_date=exit_date.strftime("%Y-%m-%d"),
        front_expiry=front.identity.expiry,
        back_expiry=back.identity.expiry,
        front=front,
        back=back,
        entry_debit_mid=entry_debit_mid,
        entry_debit_adjusted=entry_debit_adjusted,
        entry_penalty=entry_penalty,
        debit_capital=debit_capital,
        actual_entry_cashflow=actual_entry_cashflow,
        entry_offset_bdays=config.entry_offset_bdays,
        exit_offset_bdays=config.exit_offset_bdays,
        screen_date=screen_snapshot["screen_date"] if screen_snapshot else None,
        screen_front_oi=screen_snapshot["screen_front_oi"] if screen_snapshot else None,
        screen_back_oi=screen_snapshot["screen_back_oi"] if screen_snapshot else None,
        screen_front_spread_pct=screen_snapshot["screen_front_spread_pct"] if screen_snapshot else None,
        screen_back_spread_pct=screen_snapshot["screen_back_spread_pct"] if screen_snapshot else None,
    )
    if logger is not None:
        logger.debug(
            "Accepted entry %s %s strike=%.4f entry_debit=%.4f",
            selection.symbol,
            selection.event_date,
            selection.front.identity.strike,
            selection.entry_debit_mid,
        )
    return selection, None


def lookup_exact_contract(
    chain: pd.DataFrame,
    identity: ContractIdentity,
) -> Optional[pd.Series]:
    """
    Lookup the exact held contract by identity.

    Primary identity is (option_type, strike, expiry). option_id is used as an
    extra disambiguator when it matches, but never as the sole key.
    """
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
        id_matches = matches[matches["option_id"] == identity.option_id]
        if len(id_matches) == 1:
            return id_matches.iloc[0]
        if len(id_matches) > 1:
            raise BacktestInvariantError(
                f"Ambiguous option_id match for {identity.symbol} {identity.expiry} {identity.strike}"
            )

    if identity.option_symbol is not None and "option_symbol" in matches.columns:
        symbol_matches = matches[matches["option_symbol"] == identity.option_symbol]
        if len(symbol_matches) == 1:
            return symbol_matches.iloc[0]
        if len(symbol_matches) > 1:
            raise BacktestInvariantError(
                f"Ambiguous option_symbol match for {identity.symbol} {identity.expiry} {identity.strike}"
            )

    raise BacktestInvariantError(
        f"Ambiguous exact contract lookup for {identity.symbol} {identity.expiry} {identity.strike}"
    )


def realize_trade(
    selection: EntrySelection,
    config: BacktestConfig,
    chain_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]],
) -> TradeOutcome:
    """Lookup the exact exit quotes and price the held spread."""
    exit_date = pd.Timestamp(selection.exit_date)
    chain = load_day_chain(chain_cache, config.parquet_root, selection.symbol, exit_date)
    if chain is None:
        missing = build_trade_outcome(
            STATUS_MISSING_EXIT_QUOTE,
            "missing_exit_chain",
            pd.Series(
                {
                    "symbol": selection.symbol,
                    "event_date": selection.event_date,
                    "release_timing": selection.release_timing,
                    "signal_nbr": selection.signal_nbr,
                    "front_expiry": selection.front_expiry,
                    "back_expiry": selection.back_expiry,
                    "threshold_value": selection.threshold_value,
                }
            ),
            config,
            selection.threshold_value,
        )
        populate_from_selection(missing, selection)
        return missing

    front_exit_row = lookup_exact_contract(chain, selection.front.identity)
    back_exit_row = lookup_exact_contract(chain, selection.back.identity)
    if front_exit_row is None or back_exit_row is None:
        missing = build_trade_outcome(
            STATUS_MISSING_EXIT_QUOTE,
            "missing_exact_exit_quote",
            pd.Series(
                {
                    "symbol": selection.symbol,
                    "event_date": selection.event_date,
                    "release_timing": selection.release_timing,
                    "signal_nbr": selection.signal_nbr,
                    "front_expiry": selection.front_expiry,
                    "back_expiry": selection.back_expiry,
                    "threshold_value": selection.threshold_value,
                }
            ),
            config,
            selection.threshold_value,
        )
        populate_from_selection(missing, selection)
        return missing

    front_exit = build_contract_snapshot(front_exit_row, exit_date, selection.symbol)
    back_exit = build_contract_snapshot(back_exit_row, exit_date, selection.symbol)

    if config.same_strike_only and (
        not math.isclose(selection.front.identity.strike, selection.back.identity.strike, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(front_exit.identity.strike, back_exit.identity.strike, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise BacktestInvariantError(f"same-strike invariant violated for {selection.symbol} {selection.event_date}")

    if front_exit.identity.expiry != selection.front.identity.expiry:
        raise BacktestInvariantError(f"Front expiry mismatch at exit for {selection.symbol} {selection.event_date}")
    if back_exit.identity.expiry != selection.back.identity.expiry:
        raise BacktestInvariantError(f"Back expiry mismatch at exit for {selection.symbol} {selection.event_date}")
    if not math.isclose(front_exit.identity.strike, selection.front.identity.strike, rel_tol=0.0, abs_tol=1e-12):
        raise BacktestInvariantError(f"Front strike mismatch at exit for {selection.symbol} {selection.event_date}")
    if not math.isclose(back_exit.identity.strike, selection.back.identity.strike, rel_tol=0.0, abs_tol=1e-12):
        raise BacktestInvariantError(f"Back strike mismatch at exit for {selection.symbol} {selection.event_date}")

    exit_value_mid = back_exit.mid - front_exit.mid
    exit_value_adjusted, _ = conservative_exit_value(front_exit, back_exit, config.conservative_execution)
    realized_pnl_mid = (
        (exit_value_mid - selection.entry_debit_mid)
        * config.contract_multiplier
        * config.spreads_per_trade
    )
    commission_cost = (
        config.conservative_execution.commission_per_contract
        * 4
        * config.spreads_per_trade
    )
    realized_pnl_adjusted = (
        (exit_value_adjusted - selection.entry_debit_adjusted)
        * config.contract_multiplier
        * config.spreads_per_trade
        - commission_cost
    )
    return_on_capital_mid = (
        realized_pnl_mid / selection.debit_capital if selection.debit_capital and selection.debit_capital > 0 else np.nan
    )
    return_on_capital_adjusted = (
        realized_pnl_adjusted / selection.debit_capital if selection.debit_capital and selection.debit_capital > 0 else np.nan
    )

    outcome = TradeOutcome(
        status=STATUS_REALIZED,
        exclusion_reason="",
        symbol=selection.symbol,
        event_date=selection.event_date,
        release_timing=selection.release_timing,
        signal_nbr=selection.signal_nbr,
        threshold_value=selection.threshold_value,
        option_type=selection.front.identity.option_type,
        screen_date=selection.screen_date,
        screen_front_oi=selection.screen_front_oi,
        screen_back_oi=selection.screen_back_oi,
        screen_front_spread_pct=selection.screen_front_spread_pct,
        screen_back_spread_pct=selection.screen_back_spread_pct,
        entry_date=selection.entry_date,
        exit_date=selection.exit_date,
        entry_offset_bdays=selection.entry_offset_bdays,
        exit_offset_bdays=selection.exit_offset_bdays,
        front_expiry=selection.front.identity.expiry,
        back_expiry=selection.back.identity.expiry,
        front_option_symbol=selection.front.identity.option_symbol,
        back_option_symbol=selection.back.identity.option_symbol,
        front_option_id=selection.front.identity.option_id,
        back_option_id=selection.back.identity.option_id,
        strike=selection.front.identity.strike,
        front_bid_entry=selection.front.bid,
        front_ask_entry=selection.front.ask,
        front_mid_entry=selection.front.mid,
        back_bid_entry=selection.back.bid,
        back_ask_entry=selection.back.ask,
        back_mid_entry=selection.back.mid,
        entry_front_spread=selection.front.spread,
        entry_back_spread=selection.back.spread,
        front_bid_exit=front_exit.bid,
        front_ask_exit=front_exit.ask,
        front_mid_exit=front_exit.mid,
        back_bid_exit=back_exit.bid,
        back_ask_exit=back_exit.ask,
        back_mid_exit=back_exit.mid,
        exit_front_spread=front_exit.spread,
        exit_back_spread=back_exit.spread,
        entry_debit_mid=selection.entry_debit_mid,
        entry_debit_adjusted=selection.entry_debit_adjusted,
        exit_value_mid=exit_value_mid,
        exit_value_adjusted=exit_value_adjusted,
        realized_pnl_mid=realized_pnl_mid,
        realized_pnl_adjusted=realized_pnl_adjusted,
        return_on_capital_mid=return_on_capital_mid,
        return_on_capital_adjusted=return_on_capital_adjusted,
        debit_capital=selection.debit_capital,
        actual_entry_cashflow=selection.actual_entry_cashflow,
        exact_exit_quote_found=True,
        front_exact_match=True,
        back_exact_match=True,
        timing_validated=True,
    )
    validate_trade_outcome(outcome, config)
    return outcome


def populate_from_selection(outcome: TradeOutcome, selection: EntrySelection) -> None:
    """Populate a partially-filled exclusion row from a valid entry selection."""
    outcome.option_type = selection.front.identity.option_type
    outcome.screen_date = selection.screen_date
    outcome.screen_front_oi = selection.screen_front_oi
    outcome.screen_back_oi = selection.screen_back_oi
    outcome.screen_front_spread_pct = selection.screen_front_spread_pct
    outcome.screen_back_spread_pct = selection.screen_back_spread_pct
    outcome.front_option_symbol = selection.front.identity.option_symbol
    outcome.back_option_symbol = selection.back.identity.option_symbol
    outcome.front_option_id = selection.front.identity.option_id
    outcome.back_option_id = selection.back.identity.option_id
    outcome.strike = selection.front.identity.strike if math.isclose(
        selection.front.identity.strike, selection.back.identity.strike, rel_tol=0.0, abs_tol=1e-12
    ) else None
    outcome.front_bid_entry = selection.front.bid
    outcome.front_ask_entry = selection.front.ask
    outcome.front_mid_entry = selection.front.mid
    outcome.back_bid_entry = selection.back.bid
    outcome.back_ask_entry = selection.back.ask
    outcome.back_mid_entry = selection.back.mid
    outcome.entry_front_spread = selection.front.spread
    outcome.entry_back_spread = selection.back.spread
    outcome.entry_debit_mid = selection.entry_debit_mid
    outcome.entry_debit_adjusted = selection.entry_debit_adjusted
    outcome.debit_capital = selection.debit_capital
    outcome.actual_entry_cashflow = selection.actual_entry_cashflow
    outcome.timing_validated = True


def validate_trade_outcome(outcome: TradeOutcome, config: BacktestConfig) -> None:
    """Hard mechanical validations for a realized trade."""
    if outcome.status != STATUS_REALIZED:
        return

    if config.same_strike_only and outcome.strike is None:
        raise BacktestInvariantError(f"same-strike trade missing strike for {outcome.symbol} {outcome.event_date}")

    if not outcome.front_exact_match or not outcome.back_exact_match:
        raise BacktestInvariantError(f"Realized trade missing exact exit match for {outcome.symbol} {outcome.event_date}")

    event_date = pd.Timestamp(outcome.event_date)
    entry_date = pd.Timestamp(outcome.entry_date)
    exit_date = pd.Timestamp(outcome.exit_date)
    actual_entry_offset = len(pd.bdate_range(entry_date, event_date)) - 1
    actual_exit_offset = len(pd.bdate_range(event_date, exit_date)) - 1
    if actual_entry_offset != config.entry_offset_bdays:
        raise BacktestInvariantError(
            f"Entry timing mismatch for {outcome.symbol} {outcome.event_date}: expected {config.entry_offset_bdays}, got {actual_entry_offset}"
        )
    if actual_exit_offset != config.exit_offset_bdays:
        raise BacktestInvariantError(
            f"Exit timing mismatch for {outcome.symbol} {outcome.event_date}: expected {config.exit_offset_bdays}, got {actual_exit_offset}"
        )


def outcomes_to_frame(outcomes: Iterable[TradeOutcome]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(outcome) for outcome in outcomes])
    if frame.empty:
        return frame
    frame["event_date_ts"] = pd.to_datetime(frame["event_date"])
    frame["entry_date_ts"] = pd.to_datetime(frame["entry_date"])
    frame["exit_date_ts"] = pd.to_datetime(frame["exit_date"])
    return frame


def summarize_counts(trades: pd.DataFrame) -> Dict[str, Any]:
    status_counts = trades["status"].value_counts().to_dict() if not trades.empty else {}
    exclusion_counts = (
        trades.loc[trades["status"] != STATUS_REALIZED, "exclusion_reason"].value_counts().to_dict()
        if not trades.empty
        else {}
    )
    realized = trades[trades["status"] == STATUS_REALIZED]
    return {
        "total_candidate_trades": int(len(trades)),
        "accepted_entry_trades": int(trades["status"].isin([STATUS_REALIZED, STATUS_MISSING_EXIT_QUOTE]).sum()),
        "priceable_realized_trade_count": int(len(realized)),
        "missing_exit_quote_count": int((trades["status"] == STATUS_MISSING_EXIT_QUOTE).sum()),
        "excluded_trade_count": int((trades["status"] != STATUS_REALIZED).sum()),
        "status_counts": status_counts,
        "exclusion_counts": exclusion_counts,
    }


def compute_symbol_filtered_metrics(
    realized: pd.DataFrame,
    pnl_col: str,
    capital_col: str,
    return_col: str,
) -> Dict[str, Any]:
    if realized.empty:
        return {
            "trade_count": 0,
            "total_pnl": 0.0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "p10_trade": np.nan,
            "p50_trade": np.nan,
            "p90_trade": np.nan,
            "top5_concentration_pct": np.nan,
            "top3_excluded_total_pnl": np.nan,
            "top5_excluded_total_pnl": np.nan,
        }

    pnl = realized[pnl_col]
    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]
    by_symbol = realized.groupby("symbol")[pnl_col].sum().sort_values(ascending=False)
    total_pnl = float(pnl.sum())
    top5_pct = float(by_symbol.head(5).sum() / total_pnl * 100.0) if total_pnl != 0 else np.nan

    def _exclude_top(n: int) -> float:
        excluded = realized[~realized["symbol"].isin(by_symbol.head(n).index)]
        return float(excluded[pnl_col].sum()) if not excluded.empty else 0.0

    returns = realized[return_col].dropna()
    trade_return_sharpe = (
        float(returns.mean() / returns.std(ddof=1))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )

    return {
        "trade_count": int(len(realized)),
        "symbol_count": int(realized["symbol"].nunique()),
        "total_pnl": total_pnl,
        "win_rate": float((pnl > 0).mean()),
        "avg_win": float(winners.mean()) if not winners.empty else np.nan,
        "avg_loss": float(losers.mean()) if not losers.empty else np.nan,
        "p10_trade": float(pnl.quantile(0.10)),
        "p50_trade": float(pnl.quantile(0.50)),
        "p90_trade": float(pnl.quantile(0.90)),
        "top5_concentration_pct": top5_pct,
        "top3_excluded_total_pnl": _exclude_top(3),
        "top5_excluded_total_pnl": _exclude_top(5),
        "trade_return_sharpe": trade_return_sharpe,
        "median_capital": float(realized[capital_col].median()),
    }


def build_capital_usage_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Daily capital usage from accepted entries with positive debit capital."""
    accepted = trades[trades["status"].isin([STATUS_REALIZED, STATUS_MISSING_EXIT_QUOTE])].copy()
    accepted = accepted[accepted["debit_capital"].fillna(0) > 0]
    if accepted.empty:
        return pd.DataFrame(columns=["date", "capital_in_use", "open_positions"])

    start = accepted["entry_date_ts"].min()
    end = accepted["exit_date_ts"].max()
    rows: List[Dict[str, Any]] = []
    for dt in pd.bdate_range(start, end):
        open_mask = (accepted["entry_date_ts"] <= dt) & (accepted["exit_date_ts"] >= dt)
        open_trades = accepted[open_mask]
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "capital_in_use": float(open_trades["debit_capital"].sum()),
                "open_positions": int(len(open_trades)),
            }
        )
    return pd.DataFrame(rows)


def build_daily_equity_curve(
    realized: pd.DataFrame,
    config: BacktestConfig,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build a daily exact-contract MTM equity curve.

    Trades that are missing any intermediate mark are excluded from the daily
    equity curve entirely so the resulting drawdown is computed on a complete,
    explicit mark set.
    """
    if not config.daily_equity_enabled or realized.empty:
        return pd.DataFrame(columns=["date", "equity_mid", "capital_in_use", "daily_return"]), {
            "daily_equity_available": False,
            "complete_daily_mark_trade_count": 0,
            "daily_mark_missing_trade_count": 0,
        }

    chain_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
    complete_rows: List[Dict[str, Any]] = []
    dropped_trades: List[Tuple[str, str]] = []

    for trade in realized.itertuples(index=False):
        start = pd.Timestamp(trade.entry_date)
        end = pd.Timestamp(trade.exit_date)
        complete = True
        trade_marks: List[Tuple[pd.Timestamp, float]] = []
        front_identity = ContractIdentity(
            symbol=trade.symbol,
            option_type=trade.option_type,
            strike=float(trade.strike),
            expiry=trade.front_expiry,
            option_symbol=trade.front_option_symbol,
            option_id=trade.front_option_id,
        )
        back_identity = ContractIdentity(
            symbol=trade.symbol,
            option_type=trade.option_type,
            strike=float(trade.strike),
            expiry=trade.back_expiry,
            option_symbol=trade.back_option_symbol,
            option_id=trade.back_option_id,
        )
        for dt in pd.bdate_range(start, end):
            if dt == start:
                current_value = float(trade.entry_debit_mid)
            elif dt == end:
                current_value = float(trade.exit_value_mid)
            else:
                chain = load_day_chain(chain_cache, config.parquet_root, trade.symbol, dt)
                if chain is None:
                    complete = False
                    break
                front_row = lookup_exact_contract(chain, front_identity)
                back_row = lookup_exact_contract(chain, back_identity)
                if front_row is None or back_row is None:
                    complete = False
                    break
                current_value = float(back_row["mid"]) - float(front_row["mid"])
            mtm_pnl = (
                (current_value - float(trade.entry_debit_mid))
                * config.contract_multiplier
                * config.spreads_per_trade
            )
            trade_marks.append((dt, mtm_pnl))

        if not complete:
            dropped_trades.append((trade.symbol, trade.event_date))
            continue
        for dt, mtm_pnl in trade_marks:
            complete_rows.append(
                {
                    "date": dt,
                    "symbol": trade.symbol,
                    "event_date": trade.event_date,
                    "mtm_pnl": mtm_pnl,
                    "debit_capital": float(trade.debit_capital),
                    "entry_date_ts": start,
                    "exit_date_ts": end,
                }
            )

    if not complete_rows:
        return pd.DataFrame(columns=["date", "equity_mid", "capital_in_use", "daily_return"]), {
            "daily_equity_available": False,
            "complete_daily_mark_trade_count": 0,
            "daily_mark_missing_trade_count": len(dropped_trades),
            "daily_mark_missing_trades": dropped_trades,
        }

    marks = pd.DataFrame(complete_rows)
    rows: List[Dict[str, Any]] = []
    for dt, dt_marks in marks.groupby("date"):
        open_mask = (dt_marks["entry_date_ts"] <= dt) & (dt_marks["exit_date_ts"] >= dt)
        capital_in_use = float(dt_marks.loc[open_mask, "debit_capital"].sum())
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "equity_mid": float(dt_marks["mtm_pnl"].sum()),
                "capital_in_use": capital_in_use,
            }
        )

    equity = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    equity["equity_change"] = equity["equity_mid"].diff().fillna(equity["equity_mid"])
    prev_cap = equity["capital_in_use"].shift(1)
    equity["daily_return"] = np.where(
        prev_cap > 0,
        equity["equity_change"] / prev_cap,
        np.nan,
    )
    logger.info(
        "Daily equity built from %d realized trades; %d trades dropped for incomplete daily marks",
        marks[["symbol", "event_date"]].drop_duplicates().shape[0],
        len(dropped_trades),
    )
    return equity, {
        "daily_equity_available": True,
        "complete_daily_mark_trade_count": int(marks[["symbol", "event_date"]].drop_duplicates().shape[0]),
        "daily_mark_missing_trade_count": len(dropped_trades),
        "daily_mark_missing_trades": dropped_trades,
    }


def compute_max_drawdown(equity_curve: pd.DataFrame) -> Dict[str, Any]:
    if equity_curve.empty:
        return {"max_drawdown": np.nan, "drawdown_start": None, "drawdown_end": None}

    equity_vals = equity_curve["equity_mid"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(equity_vals)
    drawdowns = peaks - equity_vals
    max_idx = int(np.argmax(drawdowns))
    start_idx = int(np.argmax(equity_vals[: max_idx + 1]))
    return {
        "max_drawdown": float(drawdowns[max_idx]),
        "drawdown_start": equity_curve.loc[start_idx, "date"],
        "drawdown_end": equity_curve.loc[max_idx, "date"],
    }


def daily_return_sharpe(equity_curve: pd.DataFrame) -> float:
    returns = equity_curve["daily_return"].dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return np.nan
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))


def build_validation_summary(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {}
    realized = trades[trades["status"] == STATUS_REALIZED]
    excluded_pct = float((trades["status"] != STATUS_REALIZED).mean() * 100.0)
    missing_pct = float((trades["status"] == STATUS_MISSING_EXIT_QUOTE).mean() * 100.0)
    exact_match_pct = (
        float((realized["front_exact_match"] & realized["back_exact_match"]).mean() * 100.0)
        if not realized.empty
        else np.nan
    )
    return {
        "exact_contract_match_at_exit_pct": exact_match_pct,
        "excluded_pct": excluded_pct,
        "missing_quote_pct": missing_pct,
        "counts_by_exclusion_reason": trades.loc[trades["status"] != STATUS_REALIZED, "exclusion_reason"].value_counts().to_dict(),
    }


def load_prior_report(prior_report_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if prior_report_path is None or not prior_report_path.exists():
        return None
    return json.loads(prior_report_path.read_text())


def prior_metrics_from_report(prior_report: Dict[str, Any]) -> Dict[str, Any]:
    trades = pd.DataFrame(prior_report.get("trade_log", []))
    if trades.empty:
        return {
            "trade_count": 0,
            "priceable_trade_count": 0,
            "total_pnl": np.nan,
            "win_rate": np.nan,
            "sharpe": prior_report.get("sharpe_monthly"),
            "max_drawdown": prior_report.get("drawdown", {}).get("max_drawdown_dollars"),
            "top5_concentration_pct": np.nan,
            "missing_quote_count": None,
            "excluded_negative_debit_count": None,
        }
    by_symbol = trades.groupby("symbol")["net_total"].sum().sort_values(ascending=False)
    return {
        "trade_count": int(len(trades)),
        "priceable_trade_count": int(len(trades)),
        "total_pnl": float(trades["net_total"].sum()),
        "win_rate": float((trades["net_total"] > 0).mean()),
        "sharpe": prior_report.get("sharpe_monthly"),
        "max_drawdown": prior_report.get("drawdown", {}).get("max_drawdown_dollars"),
        "top5_concentration_pct": float(by_symbol.head(5).sum() / trades["net_total"].sum() * 100.0),
        "missing_quote_count": None,
        "excluded_negative_debit_count": None,
    }


def build_reconciliation(
    trades: pd.DataFrame,
    prior_report: Optional[Dict[str, Any]],
    metrics_mid: Dict[str, Any],
    drawdown: Dict[str, Any],
    counts: Dict[str, Any],
) -> Dict[str, Any]:
    if prior_report is None:
        return {"available": False}

    prior_metrics = prior_metrics_from_report(prior_report)
    realized = trades[trades["status"] == STATUS_REALIZED].copy()
    reconciled = pd.DataFrame(prior_report.get("trade_log", []))
    if reconciled.empty:
        return {"available": True, "prior_metrics": prior_metrics, "trade_level": []}

    trade_level = trades[["symbol", "event_date", "status", "realized_pnl_mid", "exclusion_reason"]].copy()
    merged = reconciled.merge(trade_level, on=["symbol", "event_date"], how="outer")
    merged["difference"] = merged["net_total"] - merged["realized_pnl_mid"]
    return {
        "available": True,
        "prior_metrics": prior_metrics,
        "new_metrics": {
            "trade_count": int(counts["accepted_entry_trades"]),
            "priceable_trade_count": int(counts["priceable_realized_trade_count"]),
            "total_pnl": metrics_mid["total_pnl"],
            "win_rate": metrics_mid["win_rate"],
            "sharpe": metrics_mid.get("daily_return_sharpe"),
            "max_drawdown": drawdown["max_drawdown"],
            "top5_concentration_pct": metrics_mid["top5_concentration_pct"],
            "missing_quote_count": int((trades["status"] == STATUS_MISSING_EXIT_QUOTE).sum()),
            "excluded_negative_debit_count": int((trades["status"] == STATUS_EXCLUDED_NEGATIVE_DEBIT).sum()),
        },
        "trade_level": merged[
            ["symbol", "event_date", "net_total", "realized_pnl_mid", "difference", "status", "exclusion_reason"]
        ].to_dict("records"),
    }


def run_backtest(config: BacktestConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Main engine runner."""
    events = load_candidate_events(config, logger)
    thresholds = compute_walk_forward_thresholds(
        events,
        pctile=config.walk_forward_pctile,
        cold_start_policy=config.threshold_cold_start_policy,
    )
    events = events.copy()
    events["threshold_value"] = events["event_year"].map(thresholds)

    chain_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
    outcomes: List[TradeOutcome] = []
    for candidate in events.itertuples(index=False):
        candidate_series = pd.Series(candidate._asdict())
        selection, exclusion = select_entry_contracts(candidate_series, config, chain_cache, logger)
        if exclusion is not None:
            outcomes.append(exclusion)
            continue
        assert selection is not None
        outcomes.append(realize_trade(selection, config, chain_cache))

    trades = outcomes_to_frame(outcomes)
    counts = summarize_counts(trades)
    realized = trades[trades["status"] == STATUS_REALIZED].copy()

    metrics_mid = compute_symbol_filtered_metrics(
        realized,
        "realized_pnl_mid",
        "debit_capital",
        "return_on_capital_mid",
    )
    metrics_adjusted = compute_symbol_filtered_metrics(
        realized,
        "realized_pnl_adjusted",
        "debit_capital",
        "return_on_capital_adjusted",
    )
    capital_usage = build_capital_usage_frame(trades)
    equity_curve, equity_meta = build_daily_equity_curve(realized, config, logger)
    drawdown = compute_max_drawdown(equity_curve)
    daily_sharpe = daily_return_sharpe(equity_curve)
    metrics_mid["daily_return_sharpe"] = daily_sharpe
    metrics_adjusted["daily_return_sharpe"] = daily_sharpe
    validation = build_validation_summary(trades)
    prior_report = load_prior_report(config.prior_report_path)
    reconciliation = build_reconciliation(trades, prior_report, metrics_mid, drawdown, counts)

    return {
        "trades": trades,
        "counts": counts,
        "metrics_mid": metrics_mid,
        "metrics_adjusted": metrics_adjusted,
        "capital_usage": capital_usage,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "validation": validation,
        "thresholds": thresholds,
        "equity_meta": equity_meta,
        "reconciliation": reconciliation,
    }


def write_outputs(result: Dict[str, Any], config: BacktestConfig) -> Dict[str, str]:
    """Persist the run outputs to CSV / JSON / optional daily equity CSV."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trade_csv = config.output_dir / f"calendar_exact_trades_{ts}.csv"
    summary_json = config.output_dir / f"calendar_exact_summary_{ts}.json"
    equity_csv = config.output_dir / f"calendar_exact_equity_{ts}.csv"

    result["trades"].drop(columns=["event_date_ts", "entry_date_ts", "exit_date_ts"], errors="ignore").to_csv(trade_csv, index=False)
    if not result["equity_curve"].empty:
        result["equity_curve"].to_csv(equity_csv, index=False)

    summary = {
        "assumptions": {
            "db_path": str(config.db_path),
            "parquet_root": str(config.parquet_root),
            "quality_tiers": list(config.quality_tiers),
            "option_type": config.option_type,
            "same_strike_only": config.same_strike_only,
            "screen_offset_bdays_relative_to_event": config.screen_offset_bdays,
            "entry_offset_bdays_relative_to_event": config.entry_offset_bdays,
            "exit_offset_bdays_relative_to_event": config.exit_offset_bdays,
            "atm_moneyness_window_pct": config.atm_moneyness_window,
            "min_front_oi": config.min_front_oi,
            "min_back_oi": config.min_back_oi,
            "max_spread_pct": config.max_spread_pct,
            "walk_forward_pctile": config.walk_forward_pctile,
            "min_nbr": config.min_nbr,
            "threshold_cold_start_policy": config.threshold_cold_start_policy,
            "exclude_negative_debit": config.exclude_negative_debit,
            "spreads_per_trade": config.spreads_per_trade,
            "contract_multiplier": config.contract_multiplier,
            "conservative_execution": asdict(config.conservative_execution),
            "signal_source": "earnings_iv_decay_labels.pre_front_iv / pre_back_iv",
        },
        "counts": result["counts"],
        "metrics_mid": result["metrics_mid"],
        "metrics_adjusted": result["metrics_adjusted"],
        "drawdown": result["drawdown"],
        "validation": result["validation"],
        "walk_forward_thresholds": result["thresholds"],
        "equity_meta": result["equity_meta"],
        "reconciliation_vs_prior_backtest": result["reconciliation"],
        "artifacts": {
            "trade_csv": str(trade_csv),
            "equity_csv": str(equity_csv) if not result["equity_curve"].empty else None,
        },
        "mechanical_validity_statement": (
            "This backtest DOES represent a mechanically valid trading simulation"
            if result["validation"].get("exact_contract_match_at_exit_pct") == 100.0
            else "This backtest DOES NOT represent a mechanically valid trading simulation"
        ),
    }
    summary_json.write_text(json.dumps(sanitize_for_json(summary), indent=2, allow_nan=False))
    return {
        "trade_csv": str(trade_csv),
        "summary_json": str(summary_json),
        "equity_csv": str(equity_csv) if not result["equity_curve"].empty else "",
    }


def sanitize_for_json(value: Any) -> Any:
    """Convert NaN-like values into strict JSON nulls."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)) and math.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-contract same-strike calendar backtest")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--parquet-root", default=str(DEFAULT_PARQUET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prior-report-path", default=str(DEFAULT_PRIOR_REPORT))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--option-type", default="C", choices=["C", "P"])
    parser.add_argument("--screen-offset-bdays", type=int, default=1)
    parser.add_argument("--entry-offset-bdays", type=int, default=2)
    parser.add_argument("--exit-offset-bdays", type=int, default=1)
    parser.add_argument("--atm-window", type=float, default=5.0)
    parser.add_argument("--min-front-oi", type=int)
    parser.add_argument("--min-back-oi", type=int)
    parser.add_argument("--max-spread-pct", type=float)
    parser.add_argument("--walk-forward-pctile", type=float, default=0.60)
    parser.add_argument("--min-nbr", type=float)
    parser.add_argument("--contract-multiplier", type=int, default=100)
    parser.add_argument("--spreads-per-trade", type=int, default=2)
    parser.add_argument("--half-spread-fraction", type=float, default=1.0)
    parser.add_argument("--slippage-per-leg", type=float, default=0.0)
    parser.add_argument("--commission-per-contract", type=float, default=0.0)
    parser.add_argument("--threshold-cold-start-policy", choices=["exclude"], default="exclude")
    parser.add_argument("--allow-negative-debit", action="store_true")
    parser.add_argument("--disable-daily-equity", action="store_true")
    parser.add_argument("--legacy-base-config", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> BacktestConfig:
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    screen_offset_bdays = args.screen_offset_bdays
    min_front_oi = args.min_front_oi
    min_back_oi = args.min_back_oi
    max_spread_pct = args.max_spread_pct
    if args.legacy_base_config:
        if start_date is None:
            start_date = pd.Timestamp("2023-01-01")
        if end_date is None:
            end_date = pd.Timestamp("2025-12-31")
        if min_front_oi is None:
            min_front_oi = 100
        if min_back_oi is None:
            min_back_oi = 50
        if max_spread_pct is None:
            max_spread_pct = 0.20

    return BacktestConfig(
        db_path=Path(args.db_path),
        parquet_root=Path(args.parquet_root),
        output_dir=Path(args.output_dir),
        prior_report_path=Path(args.prior_report_path) if args.prior_report_path else None,
        start_date=start_date,
        end_date=end_date,
        option_type=args.option_type,
        screen_offset_bdays=screen_offset_bdays,
        entry_offset_bdays=args.entry_offset_bdays,
        exit_offset_bdays=args.exit_offset_bdays,
        atm_moneyness_window=args.atm_window,
        min_front_oi=min_front_oi,
        min_back_oi=min_back_oi,
        max_spread_pct=max_spread_pct,
        walk_forward_pctile=args.walk_forward_pctile,
        min_nbr=args.min_nbr,
        contract_multiplier=args.contract_multiplier,
        spreads_per_trade=args.spreads_per_trade,
        threshold_cold_start_policy=args.threshold_cold_start_policy,
        exclude_negative_debit=not args.allow_negative_debit,
        daily_equity_enabled=not args.disable_daily_equity,
        conservative_execution=ExecutionAssumptions(
            half_spread_fraction=args.half_spread_fraction,
            slippage_per_leg=args.slippage_per_leg,
            commission_per_contract=args.commission_per_contract,
        ),
    )


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    logger = build_logger(args.verbose)
    logger.info("Running exact-contract calendar backtest")
    result = run_backtest(config, logger)
    artifacts = write_outputs(result, config)
    logger.info("Trade CSV: %s", artifacts["trade_csv"])
    logger.info("Summary JSON: %s", artifacts["summary_json"])
    if artifacts["equity_csv"]:
        logger.info("Equity CSV: %s", artifacts["equity_csv"])
    logger.info(
        "Realized trades: %d | Missing exit quotes: %d | Excluded: %d",
        result["counts"]["priceable_realized_trade_count"],
        result["counts"]["missing_exit_quote_count"],
        result["counts"]["excluded_trade_count"],
    )


if __name__ == "__main__":
    main()
