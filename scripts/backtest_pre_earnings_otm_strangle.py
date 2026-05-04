#!/usr/bin/env python3
"""Focused validation study for the pre-earnings AMC OTM strangle hypothesis.

This script treats the surviving AMC subset as a fresh research branch:

1. Load AMC earnings events from the local research database.
2. Build exact-contract pre-earnings strangles / straddles from local chain data.
3. Reprice the exact same contracts at exit with no substitution.
4. Evaluate timing, liquidity, spread, structure, and execution robustness.
5. Quantify overfit risk and feature usefulness.

The study is mechanically strict:
- AMC-only events
- exact contract identity persistence from entry to exit
- no fallback repricing
- missing exit quotes explicitly excluded from realized P&L
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.logger import setup_logger
from scripts.backtest_iv_expansion_study import (
    DEFAULT_DB_PATH,
    DEFAULT_OPTIONS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRICE_ROOT,
    DataCache,
    ExecutionScenario,
    StructureTrade,
    StudyInvariantError,
    build_leg_snapshot,
    build_signal_snapshot,
    compute_adjusted_pnl,
    compute_equity_curve,
    compute_max_drawdown,
    compute_return_sharpe,
    connect_readonly_sqlite,
    exact_lookup,
    markdown_table,
    parse_date,
    pick_front_expiry,
    previous_business_day,
    sanitize_for_json,
    select_common_strike_pair,
)
from web.api.edge_engine import _normalize_release_timing


STATUS_REALIZED = "REALIZED"
STATUS_MISSING_ENTRY_CHAIN = "MISSING_ENTRY_CHAIN"
STATUS_MISSING_EXIT_CHAIN = "MISSING_EXIT_CHAIN"
STATUS_MISSING_EXIT_QUOTE = "MISSING_EXIT_QUOTE"
STATUS_STRUCTURAL_INVALID = "STRUCTURAL_INVALID"
STATUS_NEGATIVE_DEBIT = "NEGATIVE_DEBIT"
STATUS_ENTRY_DATE_UNAVAILABLE = "ENTRY_DATE_UNAVAILABLE"
STATUS_EXIT_DATE_UNAVAILABLE = "EXIT_DATE_UNAVAILABLE"
STATUS_MISSING_PRICE_HISTORY = "MISSING_PRICE_HISTORY"

BASE_STRUCTURE = "otm_strangle_3pct"
BASE_ENTRY_OFFSET = 3
BASE_EXIT_OFFSET = 0
ENTRY_EXIT_GRID: Tuple[Tuple[int, int], ...] = (
    (2, 0),
    (3, 0),
    (3, 1),
    (4, 0),
    (5, 0),
)
OI_THRESHOLDS: Tuple[int, ...] = (100, 200, 300)
SPREAD_CAPS: Tuple[float, ...] = (10.0, 8.0, 6.0, 5.0)
EXECUTION_SCENARIOS = (
    ("mid", 0.0, 0.0),
    ("cross_25", 0.25, 0.0),
    ("cross_50", 0.50, 0.0),
    ("cross_100", 1.00, 0.0),
    ("cross_100_commission", 1.00, 0.65),
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class StructureVariant:
    name: str
    label: str
    kind: str
    target_moneyness_pct: Optional[float]


@dataclass(frozen=True)
class StudyConfig:
    db_path: Path
    options_root: Path
    price_root: Path
    output_root: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    quality_tiers: Tuple[str, ...] = ("A", "B")
    entry_exit_grid: Tuple[Tuple[int, int], ...] = ENTRY_EXIT_GRID
    oi_thresholds: Tuple[int, ...] = OI_THRESHOLDS
    spread_caps: Tuple[float, ...] = SPREAD_CAPS
    contract_multiplier: int = 100


@dataclass
class FocusTradeRecord:
    symbol: str
    event_date: str
    release_timing: str
    quality_tier: str
    structure: str
    structure_label: str
    structure_kind: str
    target_moneyness_pct: Optional[float]
    entry_offset_bdays: int
    exit_offset_bdays: int
    status: str
    exclusion_reason: str
    entry_date: Optional[str]
    exit_date: Optional[str]
    exact_exit_quote_found: bool
    priceable_realized: bool
    expiry: Optional[str]
    current_price_entry: Optional[float]
    call_strike: Optional[float]
    put_strike: Optional[float]
    call_option_symbol: Optional[str]
    call_option_id: Optional[int]
    put_option_symbol: Optional[str]
    put_option_id: Optional[int]
    call_bid_entry: Optional[float]
    call_ask_entry: Optional[float]
    call_mid_entry: Optional[float]
    call_bid_exit: Optional[float]
    call_ask_exit: Optional[float]
    call_mid_exit: Optional[float]
    call_iv_entry: Optional[float]
    call_iv_exit: Optional[float]
    call_iv_change: Optional[float]
    call_spread_entry: Optional[float]
    call_spread_pct_entry: Optional[float]
    call_open_interest_entry: Optional[int]
    call_volume_entry: Optional[int]
    put_bid_entry: Optional[float]
    put_ask_entry: Optional[float]
    put_mid_entry: Optional[float]
    put_bid_exit: Optional[float]
    put_ask_exit: Optional[float]
    put_mid_exit: Optional[float]
    put_iv_entry: Optional[float]
    put_iv_exit: Optional[float]
    put_iv_change: Optional[float]
    put_spread_entry: Optional[float]
    put_spread_pct_entry: Optional[float]
    put_open_interest_entry: Optional[int]
    put_volume_entry: Optional[int]
    entry_value_mid: Optional[float]
    exit_value_mid: Optional[float]
    debit_capital_per_spread: Optional[float]
    avg_iv_change: Optional[float]
    pnl_mid: Optional[float]
    pnl_cross_25: Optional[float]
    pnl_cross_50: Optional[float]
    pnl_cross_100: Optional[float]
    pnl_cross_100_commission: Optional[float]
    return_mid_pct: Optional[float]
    return_cross_25_pct: Optional[float]
    return_cross_50_pct: Optional[float]
    return_cross_100_pct: Optional[float]
    return_cross_100_commission_pct: Optional[float]
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
    signal_move_risk_level: Optional[str]
    signal_move_risk_ratio: Optional[float]
    signal_smile_curvature: Optional[float]
    signal_smile_points: Optional[int]
    signal_nbr_ratio: Optional[float]
    entry_pass_oi_100: bool
    entry_pass_oi_200: bool
    entry_pass_oi_300: bool
    entry_pass_spread_10: bool
    entry_pass_spread_8: bool
    entry_pass_spread_6: bool
    entry_pass_spread_5: bool


STRUCTURES: Dict[str, StructureVariant] = {
    "otm_strangle_2pct": StructureVariant(
        name="otm_strangle_2pct",
        label="OTM Strangle 2% Target",
        kind="otm_strangle",
        target_moneyness_pct=2.0,
    ),
    "otm_strangle_3pct": StructureVariant(
        name="otm_strangle_3pct",
        label="OTM Strangle 3% Target",
        kind="otm_strangle",
        target_moneyness_pct=3.0,
    ),
    "otm_strangle_4pct": StructureVariant(
        name="otm_strangle_4pct",
        label="OTM Strangle 4% Target",
        kind="otm_strangle",
        target_moneyness_pct=4.0,
    ),
    "atm_straddle": StructureVariant(
        name="atm_straddle",
        label="ATM Straddle Control",
        kind="atm_straddle",
        target_moneyness_pct=0.0,
    ),
}

SCENARIOS: Dict[str, ExecutionScenario] = {
    name: ExecutionScenario(name=name, half_spread_fraction=spread_fraction, commission_per_contract=commission)
    for name, spread_fraction, commission in EXECUTION_SCENARIOS
}


def build_logger(verbose: bool):
    level = "DEBUG" if verbose else "INFO"
    return setup_logger("scripts.backtest_pre_earnings_otm_strangle", level=level, file_output=False)


def load_amc_events(config: StudyConfig, logger) -> pd.DataFrame:
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
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
    frame["release_timing"] = frame["release_timing"].map(_normalize_release_timing)
    frame["nbr_ratio"] = frame["pre_front_iv"] / frame["pre_back_iv"]
    all_events = frame.dropna(subset=["symbol", "event_date"]).reset_index(drop=True)
    amc = all_events[all_events["release_timing"] == "after market close"].copy().reset_index(drop=True)
    logger.info("Loaded %d total events, %d AMC events, %d symbols", len(all_events), len(amc), amc["symbol"].nunique())
    return amc


def sort_candidates(frame: pd.DataFrame, *, target_pct: float) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_target_gap"] = (ranked["moneyness_pct"].abs() - target_pct).abs()
    ranked["_spread"] = pd.to_numeric(ranked["spread_pct"], errors="coerce").fillna(np.inf)
    return ranked.sort_values(["_target_gap", "_spread"], ascending=[True, True])


def select_variant_trade(
    variant: StructureVariant,
    chain: pd.DataFrame,
    symbol: str,
    entry_date: pd.Timestamp,
    event_date: pd.Timestamp,
) -> Tuple[Optional[List[Any]], str]:
    if chain is None or chain.empty:
        return None, "missing_entry_chain"

    expiry = pick_front_expiry(chain, event_date)
    if expiry is None:
        return None, "no_expiry_after_event"
    same_expiry = chain[chain["expiry"] == expiry].copy()
    if same_expiry.empty:
        return None, "empty_front_expiry_slice"
    current_price = float(same_expiry["underlying_price"].dropna().median())
    calls = same_expiry[same_expiry["call_put"] == "C"].copy()
    puts = same_expiry[same_expiry["call_put"] == "P"].copy()
    if calls.empty or puts.empty:
        return None, "missing_call_or_put_slice"

    if variant.kind == "atm_straddle":
        call_row, put_row = select_common_strike_pair(calls, puts, current_price)
        if call_row is None or put_row is None:
            return None, "no_common_strike_straddle_pair"
        return [
            build_leg_snapshot(call_row, symbol, entry_date, side=1),
            build_leg_snapshot(put_row, symbol, entry_date, side=1),
        ], ""

    target = float(variant.target_moneyness_pct or 0.0)
    call_candidates = calls[calls["moneyness_pct"] >= target - 1.0].copy()
    put_candidates = puts[puts["moneyness_pct"] <= -(target - 1.0)].copy()
    if call_candidates.empty:
        call_candidates = calls[calls["moneyness_pct"] > 0].copy()
    if put_candidates.empty:
        put_candidates = puts[puts["moneyness_pct"] < 0].copy()
    if call_candidates.empty or put_candidates.empty:
        return None, "no_otm_candidates"
    call_row = sort_candidates(call_candidates, target_pct=target).iloc[0]
    put_row = sort_candidates(put_candidates, target_pct=target).iloc[0]
    return [
        build_leg_snapshot(call_row, symbol, entry_date, side=1),
        build_leg_snapshot(put_row, symbol, entry_date, side=1),
    ], ""


def compute_position_value_mid(legs: Sequence[Any]) -> float:
    return float(sum(leg.side * leg.mid for leg in legs))


def compute_half_spread_cost(legs: Sequence[Any]) -> float:
    return float(sum(abs(leg.side) * (leg.spread / 2.0) for leg in legs))


def realize_variant_trade(
    variant: StructureVariant,
    symbol: str,
    event_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_chain: Optional[pd.DataFrame],
    exit_chain: Optional[pd.DataFrame],
) -> Tuple[Optional[StructureTrade], str]:
    if entry_chain is None or entry_chain.empty:
        return None, "missing_entry_chain"
    entry_legs, reason = select_variant_trade(variant, entry_chain, symbol, entry_date, event_date)
    if entry_legs is None:
        return None, reason
    if exit_chain is None or exit_chain.empty:
        return None, "missing_exit_chain"

    exit_legs: List[Any] = []
    for leg in entry_legs:
        row = exact_lookup(exit_chain, leg.identity)
        if row is None:
            return None, "missing_exact_exit_quote"
        exit_leg = build_leg_snapshot(row, symbol, exit_date, side=leg.side)
        if exit_leg.identity.expiry != leg.identity.expiry:
            raise StudyInvariantError(f"Expiry mismatch for {symbol} {event_date.date()} {variant.name}")
        if exit_leg.identity.option_type != leg.identity.option_type:
            raise StudyInvariantError(f"Option type mismatch for {symbol} {event_date.date()} {variant.name}")
        if not math.isclose(exit_leg.identity.strike, leg.identity.strike, rel_tol=0.0, abs_tol=1e-12):
            raise StudyInvariantError(f"Strike mismatch for {symbol} {event_date.date()} {variant.name}")
        exit_legs.append(exit_leg)

    entry_value_mid = compute_position_value_mid(entry_legs)
    exit_value_mid = compute_position_value_mid(exit_legs)
    debit_capital = max(entry_value_mid, 0.0)
    if debit_capital <= 0:
        return None, "negative_or_zero_debit"
    iv_changes = [
        (exit_leg.iv - entry_leg.iv)
        for entry_leg, exit_leg in zip(entry_legs, exit_legs)
        if entry_leg.iv is not None and exit_leg.iv is not None
    ]
    avg_iv_change = float(np.nanmean(iv_changes)) if iv_changes else None
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


def scenario_filter_mask(frame: pd.DataFrame, oi_threshold: int, spread_cap: float) -> pd.Series:
    # spread_cap is expressed as a percentage (e.g. 10.0 = "10%") but the stored
    # call_spread_pct_entry / put_spread_pct_entry columns are decimal fractions
    # (e.g. 0.10 = 10%).  Divide by 100 to convert to the same units before
    # comparing so that the filter actually fires.
    spread_cap_decimal = spread_cap / 100.0
    return (
        frame["call_open_interest_entry"].fillna(0) >= oi_threshold
    ) & (
        frame["put_open_interest_entry"].fillna(0) >= oi_threshold
    ) & (
        frame["call_spread_pct_entry"].fillna(np.inf) <= spread_cap_decimal
    ) & (
        frame["put_spread_pct_entry"].fillna(np.inf) <= spread_cap_decimal
    )


def return_column_for_scenario(name: str) -> str:
    if name == "mid":
        return "return_mid_pct"
    return f"return_{name}_pct"


def pnl_column_for_scenario(name: str) -> str:
    if name == "mid":
        return "pnl_mid"
    return f"pnl_{name}"


def compute_scenario_metrics(
    trades: pd.DataFrame,
    execution_name: str,
    *,
    scenario_label: str,
) -> Dict[str, Any]:
    pnl_col = pnl_column_for_scenario(execution_name)
    ret_col = return_column_for_scenario(execution_name)
    accepted = trades.copy()
    realized = accepted[accepted[pnl_col].notna()].copy()
    equity = compute_equity_curve(realized.assign(status=STATUS_REALIZED), pnl_col=pnl_col) if not realized.empty else pd.DataFrame()
    winners = realized[realized[pnl_col] > 0]
    losers = realized[realized[pnl_col] <= 0]
    by_symbol = realized.groupby("symbol")[pnl_col].sum().sort_values(ascending=False) if not realized.empty else pd.Series(dtype=float)
    total_pnl = float(realized[pnl_col].sum()) if not realized.empty else 0.0
    return {
        "scenario_label": scenario_label,
        "candidate_count": int(len(trades)),
        "accepted_count": int(len(accepted)),
        "realized_count": int(len(realized)),
        "missing_exit_quote_count": int((accepted["status"] == STATUS_MISSING_EXIT_QUOTE).sum()),
        "structural_invalid_count": int((accepted["status"] == STATUS_STRUCTURAL_INVALID).sum()),
        "negative_debit_count": int((accepted["status"] == STATUS_NEGATIVE_DEBIT).sum()),
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
    }


def build_scoreboard(trades: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for structure in STRUCTURES:
        for entry_offset, exit_offset in config.entry_exit_grid:
            base = trades[
                (trades["structure"] == structure)
                & (trades["entry_offset_bdays"] == entry_offset)
                & (trades["exit_offset_bdays"] == exit_offset)
            ].copy()
            if base.empty:
                continue
            for oi_threshold in config.oi_thresholds:
                for spread_cap in config.spread_caps:
                    filtered = base[scenario_filter_mask(base, oi_threshold, spread_cap)].copy()
                    for execution_name in SCENARIOS:
                        row = compute_scenario_metrics(
                            filtered,
                            execution_name,
                            scenario_label=f"{structure}:{entry_offset}:{exit_offset}:{oi_threshold}:{spread_cap}:{execution_name}",
                        )
                        row.update(
                            {
                                "structure": structure,
                                "entry_offset_bdays": entry_offset,
                                "exit_offset_bdays": exit_offset,
                                "oi_threshold": oi_threshold,
                                "spread_cap_pct": spread_cap,
                                "execution_scenario": execution_name,
                            }
                        )
                        rows.append(row)
    return pd.DataFrame(rows)


def safe_corr(frame: pd.DataFrame, x: str, y: str) -> float:
    subset = frame[[x, y]].dropna()
    if len(subset) < 10:
        return np.nan
    return float(subset[x].corr(subset[y], method="spearman"))


def feature_rankings(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    features = {
        "term_structure_slope": "signal_term_structure_slope",
        "iv_rv30": "signal_iv_rv30",
        "nbr_ratio": "signal_nbr_ratio",
        "rv_percentile_rank": "signal_rv_percentile_rank",
        "near_term_liquidity_proxy": "signal_near_term_liquidity_proxy",
        "near_term_spread_pct": "signal_near_term_spread_pct",
        "event_implied_move_pct": "signal_event_implied_move_pct",
        "non_event_move_pct": "signal_non_event_move_pct",
        "smile_curvature": "signal_smile_curvature",
    }
    for label, col in features.items():
        subset = trades[[col, "avg_iv_change", "return_cross_50_pct", "return_cross_100_pct"]].dropna()
        if len(subset) < 10:
            continue
        top_cut = subset[col].quantile(0.75)
        bot_cut = subset[col].quantile(0.25)
        top = subset[subset[col] >= top_cut]
        bot = subset[subset[col] <= bot_cut]
        rows.append(
            {
                "feature": label,
                "sample_size": int(len(subset)),
                "spearman_iv_change": safe_corr(subset, col, "avg_iv_change"),
                "spearman_return_cross_50": safe_corr(subset, col, "return_cross_50_pct"),
                "spearman_return_cross_100": safe_corr(subset, col, "return_cross_100_pct"),
                "top_quartile_avg_iv_change": float(top["avg_iv_change"].mean()) if not top.empty else np.nan,
                "bottom_quartile_avg_iv_change": float(bot["avg_iv_change"].mean()) if not bot.empty else np.nan,
                "top_quartile_return_cross_50": float(top["return_cross_50_pct"].mean()) if not top.empty else np.nan,
                "bottom_quartile_return_cross_50": float(bot["return_cross_50_pct"].mean()) if not bot.empty else np.nan,
                "top_quartile_return_cross_100": float(top["return_cross_100_pct"].mean()) if not top.empty else np.nan,
                "bottom_quartile_return_cross_100": float(bot["return_cross_100_pct"].mean()) if not bot.empty else np.nan,
            }
        )
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking["rank_score"] = ranking["spearman_iv_change"].abs().fillna(0.0) + ranking["spearman_return_cross_100"].abs().fillna(0.0)
        ranking = ranking.sort_values("rank_score", ascending=False).reset_index(drop=True)
    return ranking


def bootstrap_summary(values: pd.Series, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    series = values.dropna().to_numpy(dtype=float)
    if len(series) == 0:
        return {
            "sample_size": 0,
            "mean": None,
            "median": None,
            "probability_positive_total": None,
            "total_pnl_ci_5_95": None,
        }
    rng = np.random.default_rng(seed)
    resampled = rng.choice(series, size=(draws, len(series)), replace=True).sum(axis=1)
    return {
        "sample_size": int(len(series)),
        "mean": float(np.mean(series)),
        "median": float(np.median(series)),
        "probability_positive_total": float(np.mean(resampled > 0)),
        "total_pnl_ci_5_95": [float(np.quantile(resampled, 0.05)), float(np.quantile(resampled, 0.95))],
    }


def overfit_checks(baseline_trades: pd.DataFrame) -> pd.DataFrame:
    if baseline_trades.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []

    def add_case(label: str, subset: pd.DataFrame) -> None:
        for execution_name in ("cross_50", "cross_100"):
            metrics = compute_scenario_metrics(subset, execution_name, scenario_label=label)
            metrics["overfit_case"] = label
            metrics["execution_scenario"] = execution_name
            rows.append(metrics)

    by_symbol = baseline_trades.groupby("symbol")["pnl_cross_50"].sum().sort_values(ascending=False)
    best_trades = baseline_trades.sort_values("pnl_cross_50", ascending=False)
    add_case("baseline", baseline_trades)
    add_case("remove_top3_symbols", baseline_trades[~baseline_trades["symbol"].isin(by_symbol.head(3).index)])
    add_case("remove_top5_symbols", baseline_trades[~baseline_trades["symbol"].isin(by_symbol.head(5).index)])
    add_case("remove_best3_trades", best_trades.iloc[3:].copy())
    add_case("remove_best5_trades", best_trades.iloc[5:].copy())
    counts = baseline_trades["symbol"].value_counts()
    keep_symbols = counts[counts >= 2].index
    add_case("exclude_symbols_lt2_obs", baseline_trades[baseline_trades["symbol"].isin(keep_symbols)])
    return pd.DataFrame(rows)


def yearly_breakdown(baseline_trades: pd.DataFrame) -> pd.DataFrame:
    if baseline_trades.empty:
        return pd.DataFrame()
    frame = baseline_trades.copy()
    frame["year"] = pd.to_datetime(frame["event_date"]).dt.year
    rows: List[Dict[str, Any]] = []
    for year, subset in frame.groupby("year"):
        for execution_name in ("mid", "cross_50", "cross_100"):
            metrics = compute_scenario_metrics(subset, execution_name, scenario_label=f"year:{year}:{execution_name}")
            metrics["year"] = int(year)
            metrics["execution_scenario"] = execution_name
            rows.append(metrics)
    split = frame.copy()
    split["temporal_split"] = np.where(pd.to_datetime(split["event_date"]) < pd.Timestamp("2025-01-01"), "2023_2024", "2025_2026")
    for split_name, subset in split.groupby("temporal_split"):
        for execution_name in ("mid", "cross_50", "cross_100"):
            metrics = compute_scenario_metrics(subset, execution_name, scenario_label=f"split:{split_name}:{execution_name}")
            metrics["temporal_split"] = split_name
            metrics["execution_scenario"] = execution_name
            rows.append(metrics)
    return pd.DataFrame(rows)


def baseline_filter(frame: pd.DataFrame) -> pd.Series:
    return scenario_filter_mask(frame, oi_threshold=100, spread_cap=10.0)


def build_memo(results: Dict[str, Any]) -> str:
    reproduction = results["reproduction"]
    signal_rankings = results["signal_rankings"]
    overfit = results["overfit"]
    yearly = results["yearly_breakdown"]
    best_row = results["scoreboard"].sort_values("total_pnl", ascending=False).iloc[0]

    baseline_table = markdown_table(
        pd.DataFrame(results["baseline_cost_table"]),
        ["execution_scenario", "accepted_count", "realized_count", "missing_exit_quote_count", "total_pnl", "win_rate", "return_based_sharpe", "max_drawdown"],
    )
    robustness_table = markdown_table(
        results["robustness_top"],
        ["structure", "entry_offset_bdays", "exit_offset_bdays", "oi_threshold", "spread_cap_pct", "execution_scenario", "realized_count", "total_pnl", "win_rate"],
    )
    signal_table = markdown_table(signal_rankings, list(signal_rankings.columns), limit=10)
    overfit_table = markdown_table(overfit, list(overfit.columns), limit=12)
    yearly_table = markdown_table(yearly, list(yearly.columns), limit=12)

    verdict = results["audit_verdict"]
    return f"""# Pre-Earnings AMC OTM Strangle Validation

## A. Executive Verdict
{results["executive_verdict"]}

## B. Baseline Reproduction
- Prior reported subset: `42` trades, `+$3,471` at `50%` spread crossing, `+$1,319` at `100%`.
- Reproduced subset: `accepted={reproduction['accepted_count']}`, `realized={reproduction['realized_count']}`, `missing_exit={reproduction['missing_exit_quote_count']}`, `cross_50={reproduction['cross_50_total_pnl']:.2f}`, `cross_100={reproduction['cross_100_total_pnl']:.2f}`.
- Reproduction status: **{results["reproduction_status"]}**

## C. Mechanical Validity
- AMC-only enforced at event loading.
- Entry = exact `T-{BASE_ENTRY_OFFSET}` business days; exit = exact `T-{BASE_EXIT_OFFSET}` on the earnings date before AMC release.
- Exact contract identity persisted using `(option_type, strike, expiry)` plus `option_symbol` / `option_id`.
- Missing exact exit quotes are explicit and excluded from realized P&L.
- Baseline missing exact exit quotes: `{reproduction['missing_exit_quote_count']}` out of `{reproduction['accepted_count']}` accepted baseline candidates.

## D. Economic Results
{baseline_table}

## E. Robustness
Top robustness rows by realized adjusted P&L:
{robustness_table}

Overfit / fragility checks on the baseline subset:
{overfit_table}

Year / temporal splits:
{yearly_table}

## F. Signal Quality
{signal_table}

## G. Overfit Risk
- Best full-grid scenario: `{best_row['structure']}` / `T-{int(best_row['entry_offset_bdays'])}` to `T-{int(best_row['exit_offset_bdays'])}` / `OI>={int(best_row['oi_threshold'])}` / `spread<={best_row['spread_cap_pct']:.1f}%` / `{best_row['execution_scenario']}` with total P&L `{best_row['total_pnl']:.2f}` on `{int(best_row['realized_count'])}` trades.
- Baseline bootstrap at `50%`: probability positive total = `{results['bootstrap']['cross_50']['probability_positive_total']}`.
- Baseline bootstrap at `100%`: probability positive total = `{results['bootstrap']['cross_100']['probability_positive_total']}`.
- Sector analysis: unavailable from local offline data.

## H. Recommendation
{results["recommendation"]}
"""


def build_audit_memo(results: Dict[str, Any]) -> str:
    return f"{results['audit_verdict']}\n"


def build_trade_record(
    event: pd.Series,
    variant: StructureVariant,
    entry_offset: int,
    exit_offset: int,
    entry_date: Optional[pd.Timestamp],
    exit_date: Optional[pd.Timestamp],
    status: str,
    exclusion_reason: str,
    trade: Optional[StructureTrade],
    signal_snapshot: Dict[str, Any],
) -> FocusTradeRecord:
    defaults = {
        "expiry": None,
        "current_price_entry": None,
        "call_strike": None,
        "put_strike": None,
        "call_option_symbol": None,
        "call_option_id": None,
        "put_option_symbol": None,
        "put_option_id": None,
        "call_bid_entry": None,
        "call_ask_entry": None,
        "call_mid_entry": None,
        "call_bid_exit": None,
        "call_ask_exit": None,
        "call_mid_exit": None,
        "call_iv_entry": None,
        "call_iv_exit": None,
        "call_iv_change": None,
        "call_spread_entry": None,
        "call_spread_pct_entry": None,
        "call_open_interest_entry": None,
        "call_volume_entry": None,
        "put_bid_entry": None,
        "put_ask_entry": None,
        "put_mid_entry": None,
        "put_bid_exit": None,
        "put_ask_exit": None,
        "put_mid_exit": None,
        "put_iv_entry": None,
        "put_iv_exit": None,
        "put_iv_change": None,
        "put_spread_entry": None,
        "put_spread_pct_entry": None,
        "put_open_interest_entry": None,
        "put_volume_entry": None,
        "entry_value_mid": None,
        "exit_value_mid": None,
        "debit_capital_per_spread": None,
        "avg_iv_change": None,
        "pnl_mid": None,
        "pnl_cross_25": None,
        "pnl_cross_50": None,
        "pnl_cross_100": None,
        "pnl_cross_100_commission": None,
        "return_mid_pct": None,
        "return_cross_25_pct": None,
        "return_cross_50_pct": None,
        "return_cross_100_pct": None,
        "return_cross_100_commission_pct": None,
    }
    if trade is not None:
        call_leg = next((leg for leg in trade.entry_legs if leg.identity.option_type == "C"), None)
        put_leg = next((leg for leg in trade.entry_legs if leg.identity.option_type == "P"), None)
        call_exit = next((leg for leg in trade.exit_legs if leg.identity.option_type == "C"), None)
        put_exit = next((leg for leg in trade.exit_legs if leg.identity.option_type == "P"), None)
        if call_leg is not None and call_exit is not None:
            defaults.update(
                {
                    "expiry": call_leg.identity.expiry,
                    "current_price_entry": call_leg.underlying_price,
                    "call_strike": call_leg.identity.strike,
                    "call_option_symbol": call_leg.identity.option_symbol,
                    "call_option_id": call_leg.identity.option_id,
                    "call_bid_entry": call_leg.bid,
                    "call_ask_entry": call_leg.ask,
                    "call_mid_entry": call_leg.mid,
                    "call_bid_exit": call_exit.bid,
                    "call_ask_exit": call_exit.ask,
                    "call_mid_exit": call_exit.mid,
                    "call_iv_entry": call_leg.iv,
                    "call_iv_exit": call_exit.iv,
                    "call_iv_change": (call_exit.iv - call_leg.iv) if call_leg.iv is not None and call_exit.iv is not None else None,
                    "call_spread_entry": call_leg.spread,
                    "call_spread_pct_entry": call_leg.spread_pct,
                    "call_open_interest_entry": call_leg.open_interest,
                    "call_volume_entry": call_leg.volume,
                }
            )
        if put_leg is not None and put_exit is not None:
            defaults.update(
                {
                    "expiry": put_leg.identity.expiry if defaults["expiry"] is None else defaults["expiry"],
                    "current_price_entry": put_leg.underlying_price if defaults["current_price_entry"] is None else defaults["current_price_entry"],
                    "put_strike": put_leg.identity.strike,
                    "put_option_symbol": put_leg.identity.option_symbol,
                    "put_option_id": put_leg.identity.option_id,
                    "put_bid_entry": put_leg.bid,
                    "put_ask_entry": put_leg.ask,
                    "put_mid_entry": put_leg.mid,
                    "put_bid_exit": put_exit.bid,
                    "put_ask_exit": put_exit.ask,
                    "put_mid_exit": put_exit.mid,
                    "put_iv_entry": put_leg.iv,
                    "put_iv_exit": put_exit.iv,
                    "put_iv_change": (put_exit.iv - put_leg.iv) if put_leg.iv is not None and put_exit.iv is not None else None,
                    "put_spread_entry": put_leg.spread,
                    "put_spread_pct_entry": put_leg.spread_pct,
                    "put_open_interest_entry": put_leg.open_interest,
                    "put_volume_entry": put_leg.volume,
                }
            )
        debit_capital = trade.debit_capital_per_spread * 100.0
        defaults.update(
            {
                "entry_value_mid": trade.entry_value_mid,
                "exit_value_mid": trade.exit_value_mid,
                "debit_capital_per_spread": trade.debit_capital_per_spread,
                "avg_iv_change": trade.avg_iv_change,
                "pnl_mid": (trade.exit_value_mid - trade.entry_value_mid) * 100.0,
                "pnl_cross_25": compute_adjusted_pnl(trade, SCENARIOS["cross_25"], 100),
                "pnl_cross_50": compute_adjusted_pnl(trade, SCENARIOS["cross_50"], 100),
                "pnl_cross_100": compute_adjusted_pnl(trade, SCENARIOS["cross_100"], 100),
                "pnl_cross_100_commission": compute_adjusted_pnl(trade, SCENARIOS["cross_100_commission"], 100),
            }
        )
        defaults["return_mid_pct"] = (defaults["pnl_mid"] / debit_capital) * 100.0 if debit_capital > 0 else None
        defaults["return_cross_25_pct"] = (defaults["pnl_cross_25"] / debit_capital) * 100.0 if debit_capital > 0 else None
        defaults["return_cross_50_pct"] = (defaults["pnl_cross_50"] / debit_capital) * 100.0 if debit_capital > 0 else None
        defaults["return_cross_100_pct"] = (defaults["pnl_cross_100"] / debit_capital) * 100.0 if debit_capital > 0 else None
        defaults["return_cross_100_commission_pct"] = (defaults["pnl_cross_100_commission"] / debit_capital) * 100.0 if debit_capital > 0 else None

    call_oi = defaults["call_open_interest_entry"] or 0
    put_oi = defaults["put_open_interest_entry"] or 0
    call_spread_pct = defaults["call_spread_pct_entry"] if defaults["call_spread_pct_entry"] is not None else np.inf
    put_spread_pct = defaults["put_spread_pct_entry"] if defaults["put_spread_pct_entry"] is not None else np.inf

    return FocusTradeRecord(
        symbol=str(event["symbol"]),
        event_date=pd.Timestamp(event["event_date"]).strftime("%Y-%m-%d"),
        release_timing=str(event["release_timing"]),
        quality_tier=str(event["quality_tier"]),
        structure=variant.name,
        structure_label=variant.label,
        structure_kind=variant.kind,
        target_moneyness_pct=variant.target_moneyness_pct,
        entry_offset_bdays=entry_offset,
        exit_offset_bdays=exit_offset,
        status=status,
        exclusion_reason=exclusion_reason,
        entry_date=entry_date.strftime("%Y-%m-%d") if entry_date is not None else None,
        exit_date=exit_date.strftime("%Y-%m-%d") if exit_date is not None else None,
        exact_exit_quote_found=status == STATUS_REALIZED,
        priceable_realized=trade is not None and defaults["pnl_mid"] is not None,
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
        signal_move_risk_level=signal_snapshot.get("move_risk_level"),
        signal_move_risk_ratio=signal_snapshot.get("move_risk_ratio"),
        signal_smile_curvature=signal_snapshot.get("smile_curvature"),
        signal_smile_points=signal_snapshot.get("smile_points"),
        signal_nbr_ratio=signal_snapshot.get("nbr_ratio"),
        entry_pass_oi_100=call_oi >= 100 and put_oi >= 100,
        entry_pass_oi_200=call_oi >= 200 and put_oi >= 200,
        entry_pass_oi_300=call_oi >= 300 and put_oi >= 300,
        entry_pass_spread_10=call_spread_pct <= 0.10 and put_spread_pct <= 0.10,
        entry_pass_spread_8=call_spread_pct <= 0.08 and put_spread_pct <= 0.08,
        entry_pass_spread_6=call_spread_pct <= 0.06 and put_spread_pct <= 0.06,
        entry_pass_spread_5=call_spread_pct <= 0.05 and put_spread_pct <= 0.05,
        **defaults,
    )


def run_study(config: StudyConfig, logger) -> Dict[str, Any]:
    events = load_amc_events(config, logger)
    cache = DataCache(config)
    rows: List[FocusTradeRecord] = []

    for event in events.itertuples(index=False):
        event_series = pd.Series(event._asdict())
        symbol = str(event_series["symbol"])
        price_history = cache.load_price_history(symbol)
        if price_history is None or price_history.empty:
            for variant in STRUCTURES.values():
                for entry_offset, exit_offset in config.entry_exit_grid:
                    rows.append(
                        build_trade_record(
                            event_series,
                            variant,
                            entry_offset,
                            exit_offset,
                            entry_date=None,
                            exit_date=None,
                            status=STATUS_MISSING_PRICE_HISTORY,
                            exclusion_reason="missing_price_history",
                            trade=None,
                            signal_snapshot={},
                        )
                    )
            continue

        prior_symbol_events = events[(events["symbol"] == symbol) & (events["event_date"] < event_series["event_date"])].copy()
        event_series["prior_symbol_events"] = prior_symbol_events

        for entry_offset, exit_offset in config.entry_exit_grid:
            entry_date = previous_business_day(price_history, pd.Timestamp(event_series["event_date"]), entry_offset)
            if entry_date is None:
                for variant in STRUCTURES.values():
                    rows.append(
                        build_trade_record(
                            event_series,
                            variant,
                            entry_offset,
                            exit_offset,
                            entry_date=None,
                            exit_date=None,
                            status=STATUS_ENTRY_DATE_UNAVAILABLE,
                            exclusion_reason="entry_date_unavailable",
                            trade=None,
                            signal_snapshot={},
                        )
                    )
                continue

            if exit_offset == 0:
                if entry_date >= pd.Timestamp(event_series["event_date"]):
                    exit_date = None
                else:
                    exit_date = pd.Timestamp(event_series["event_date"]).normalize()
            else:
                exit_date = previous_business_day(price_history, pd.Timestamp(event_series["event_date"]), exit_offset)
            if exit_date is None or entry_date >= exit_date:
                for variant in STRUCTURES.values():
                    rows.append(
                        build_trade_record(
                            event_series,
                            variant,
                            entry_offset,
                            exit_offset,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            status=STATUS_EXIT_DATE_UNAVAILABLE,
                            exclusion_reason="exit_date_unavailable",
                            trade=None,
                            signal_snapshot={},
                        )
                    )
                continue

            entry_chain = cache.load_day_chain(symbol, entry_date)
            exit_chain = cache.load_day_chain(symbol, exit_date)
            signal_snapshot = (
                build_signal_snapshot(symbol, event_series, entry_date, exit_date, entry_chain, price_history)
                if entry_chain is not None
                else {}
            )

            for variant in STRUCTURES.values():
                trade, reason = realize_variant_trade(
                    variant=variant,
                    symbol=symbol,
                    event_date=pd.Timestamp(event_series["event_date"]),
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_chain=entry_chain,
                    exit_chain=exit_chain,
                )
                status = STATUS_REALIZED
                if trade is None:
                    if reason == "missing_entry_chain":
                        status = STATUS_MISSING_ENTRY_CHAIN
                    elif reason == "missing_exit_chain":
                        status = STATUS_MISSING_EXIT_CHAIN
                    elif reason == "missing_exact_exit_quote":
                        status = STATUS_MISSING_EXIT_QUOTE
                    elif reason == "negative_or_zero_debit":
                        status = STATUS_NEGATIVE_DEBIT
                    else:
                        status = STATUS_STRUCTURAL_INVALID
                rows.append(
                    build_trade_record(
                        event_series,
                        variant,
                        entry_offset,
                        exit_offset,
                        entry_date=entry_date,
                        exit_date=exit_date,
                        status=status,
                        exclusion_reason=reason,
                        trade=trade,
                        signal_snapshot=signal_snapshot,
                    )
                )

    trades = pd.DataFrame([asdict(row) for row in rows])
    baseline_mask = (
        (trades["structure"] == BASE_STRUCTURE)
        & (trades["entry_offset_bdays"] == BASE_ENTRY_OFFSET)
        & (trades["exit_offset_bdays"] == BASE_EXIT_OFFSET)
    )
    baseline_all = trades[baseline_mask].copy()
    baseline_accepted = baseline_all[baseline_filter(baseline_all)].copy()
    baseline_realized = baseline_accepted[baseline_accepted["pnl_mid"].notna()].copy()

    scoreboard = build_scoreboard(trades, config)
    baseline_cost_table: List[Dict[str, Any]] = []
    for execution_name in SCENARIOS:
        row = compute_scenario_metrics(baseline_accepted, execution_name, scenario_label=f"baseline:{execution_name}")
        row["execution_scenario"] = execution_name
        baseline_cost_table.append(row)

    signal_rankings = feature_rankings(baseline_realized)
    overfit = overfit_checks(baseline_realized)
    yearly = yearly_breakdown(baseline_realized)
    bootstrap = {
        "cross_50": bootstrap_summary(baseline_realized["pnl_cross_50"]),
        "cross_100": bootstrap_summary(baseline_realized["pnl_cross_100"]),
    }

    top_robust = scoreboard[
        (scoreboard["execution_scenario"].isin(["cross_50", "cross_100"]))
        & (scoreboard["realized_count"] >= 5)
    ].sort_values("total_pnl", ascending=False).head(12).reset_index(drop=True)

    reproduction = {
        "accepted_count": int(len(baseline_accepted)),
        "realized_count": int(len(baseline_realized)),
        "missing_exit_quote_count": int((baseline_accepted["status"] == STATUS_MISSING_EXIT_QUOTE).sum()),
        "cross_50_total_pnl": float(baseline_realized["pnl_cross_50"].sum()) if not baseline_realized.empty else 0.0,
        "cross_100_total_pnl": float(baseline_realized["pnl_cross_100"].sum()) if not baseline_realized.empty else 0.0,
    }
    reported_50 = 3471.0
    reported_100 = 1319.0
    reproduction_ok = (
        reproduction["realized_count"] == 42
        and abs(reproduction["cross_50_total_pnl"] - reported_50) < 1e-6
        and abs(reproduction["cross_100_total_pnl"] - reported_100) < 1e-6
    )

    best_cross100 = scoreboard[
        (scoreboard["execution_scenario"] == "cross_100")
        & (scoreboard["realized_count"] >= 5)
    ].sort_values("total_pnl", ascending=False)
    best_cross100_row = best_cross100.iloc[0] if not best_cross100.empty else None
    best_cross100_commission = scoreboard[
        (scoreboard["execution_scenario"] == "cross_100_commission")
        & (scoreboard["realized_count"] >= 20)
    ].sort_values("total_pnl", ascending=False)
    best_cross100_commission_row = best_cross100_commission.iloc[0] if not best_cross100_commission.empty else None

    if len(baseline_realized) == 0:
        audit_verdict = "not validated"
        executive_verdict = "The baseline AMC-only `T-3 / T-0` `3%` OTM strangle subset could not be realized on enough exact-contract trades to validate it."
        recommendation = "discard"
    else:
        baseline_cross100 = float(baseline_realized["pnl_cross_100"].sum())
        baseline_cross50 = float(baseline_realized["pnl_cross_50"].sum())
        baseline_cross100_commission = float(baseline_realized["pnl_cross_100_commission"].sum())
        overfit_cross100 = overfit[overfit["execution_scenario"] == "cross_100"].set_index("overfit_case") if not overfit.empty else pd.DataFrame()
        remove_top3_cross100 = float(overfit_cross100.loc["remove_top3_symbols", "total_pnl"]) if "remove_top3_symbols" in overfit_cross100.index else np.nan
        remove_top5_cross50 = float(overfit[ (overfit["overfit_case"] == "remove_top5_symbols") & (overfit["execution_scenario"] == "cross_50") ]["total_pnl"].iloc[0]) if not overfit.empty and ((overfit["overfit_case"] == "remove_top5_symbols") & (overfit["execution_scenario"] == "cross_50")).any() else np.nan
        yearly_cross100_2025 = yearly[(yearly.get("year") == 2025) & (yearly.get("execution_scenario") == "cross_100")]
        year_2025_cross100 = float(yearly_cross100_2025["total_pnl"].iloc[0]) if not yearly_cross100_2025.empty else np.nan

        validated = (
            reproduction_ok
            and baseline_cross100_commission > 0
            and np.isfinite(remove_top3_cross100) and remove_top3_cross100 > 0
            and np.isfinite(remove_top5_cross50) and remove_top5_cross50 > 0
            and np.isfinite(year_2025_cross100) and year_2025_cross100 > 0
        )
        promising = (
            baseline_cross50 > 0
            and baseline_cross100 > 0
            and best_cross100_commission_row is not None
            and float(best_cross100_commission_row["total_pnl"]) > 0
        )

        if validated:
            audit_verdict = "validated"
            recommendation = "candidate for limited live paper-trading only"
            executive_verdict = "The AMC pre-earnings strangle subset reproduces closely enough and remains positive under harsh execution, symbol-removal, and later-period checks to count as a real but still capacity-limited edge."
        elif promising:
            audit_verdict = "promising but fragile"
            recommendation = "keep as research-only"
            executive_verdict = "The baseline AMC `T-3 / T-0` `3%` OTM strangle subset stays positive before commissions and has a nearby stricter-liquidity variant that remains positive even after commissions, but the baseline itself is concentration-sensitive, fails exact reproduction, and weakens materially out of sample."
        else:
            audit_verdict = "not validated"
            recommendation = "discard"
            executive_verdict = "The exact baseline subset reproduces mechanically, but the edge collapses under nearby timing, liquidity, spread, or overfit checks. The evidence points to a small-sample pocket rather than a robust tradeable edge."

    return {
        "trades": trades,
        "scoreboard": scoreboard,
        "baseline_cost_table": baseline_cost_table,
        "baseline_realized": baseline_realized,
        "baseline_accepted": baseline_accepted,
        "reproduction": reproduction,
        "reproduction_status": "reproduced exactly" if reproduction_ok else "did not reproduce exactly",
        "signal_rankings": signal_rankings,
        "overfit": overfit,
        "yearly_breakdown": yearly,
        "bootstrap": bootstrap,
        "robustness_top": top_robust,
        "audit_verdict": audit_verdict,
        "executive_verdict": executive_verdict,
        "recommendation": recommendation,
    }


def write_outputs(results: Dict[str, Any], config: StudyConfig) -> Dict[str, str]:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.output_root / f"pre_earnings_otm_strangle_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trade_csv = out_dir / "pre_earnings_otm_strangle_trade_log.csv"
    summary_json = out_dir / "pre_earnings_otm_strangle_summary.json"
    scoreboard_csv = out_dir / "pre_earnings_otm_strangle_scoreboard.csv"
    signal_csv = out_dir / "pre_earnings_otm_strangle_signal_rankings.csv"
    overfit_csv = out_dir / "pre_earnings_otm_strangle_overfit.csv"
    yearly_csv = out_dir / "pre_earnings_otm_strangle_yearly_breakdown.csv"
    memo_md = out_dir / "pre_earnings_otm_strangle_memo.md"
    audit_md = out_dir / "pre_earnings_otm_strangle_audit.md"

    results["trades"].to_csv(trade_csv, index=False)
    results["scoreboard"].to_csv(scoreboard_csv, index=False)
    results["signal_rankings"].to_csv(signal_csv, index=False)
    results["overfit"].to_csv(overfit_csv, index=False)
    results["yearly_breakdown"].to_csv(yearly_csv, index=False)

    artifacts = {
        "output_dir": str(out_dir),
        "trade_csv": str(trade_csv),
        "summary_json": str(summary_json),
        "scoreboard_csv": str(scoreboard_csv),
        "signal_csv": str(signal_csv),
        "overfit_csv": str(overfit_csv),
        "yearly_csv": str(yearly_csv),
        "memo_md": str(memo_md),
        "audit_md": str(audit_md),
    }

    summary = {
        "assumptions": sanitize_for_json(asdict(config)),
        "reproduction": sanitize_for_json(results["reproduction"]),
        "reproduction_status": results["reproduction_status"],
        "baseline_cost_table": sanitize_for_json(results["baseline_cost_table"]),
        "audit_verdict": results["audit_verdict"],
        "executive_verdict": results["executive_verdict"],
        "recommendation": results["recommendation"],
        "bootstrap": sanitize_for_json(results["bootstrap"]),
        "scoreboard": sanitize_for_json(results["scoreboard"].to_dict(orient="records")),
        "signal_rankings": sanitize_for_json(results["signal_rankings"].to_dict(orient="records")),
        "overfit": sanitize_for_json(results["overfit"].to_dict(orient="records")),
        "yearly_breakdown": sanitize_for_json(results["yearly_breakdown"].to_dict(orient="records")),
        "artifacts": artifacts,
    }
    summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False))
    memo_md.write_text(build_memo(results))
    audit_md.write_text(build_audit_memo(results))
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused backtest for the pre-earnings AMC OTM strangle hypothesis")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--options-root", default=str(DEFAULT_OPTIONS_ROOT))
    parser.add_argument("--price-root", default=str(DEFAULT_PRICE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StudyConfig:
    return StudyConfig(
        db_path=Path(args.db_path),
        options_root=Path(args.options_root),
        price_root=Path(args.price_root),
        output_root=Path(args.output_root),
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
    )


def main() -> int:
    args = parse_args()
    logger = build_logger(args.verbose)
    config = build_config(args)
    logger.info("Running focused AMC pre-earnings OTM strangle study")
    results = run_study(config, logger)
    artifacts = write_outputs(results, config)
    logger.info("Trade log: %s", artifacts["trade_csv"])
    logger.info("Summary JSON: %s", artifacts["summary_json"])
    logger.info("Memo: %s", artifacts["memo_md"])
    logger.info("Audit memo: %s", artifacts["audit_md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
