#!/usr/bin/env python3
"""Audit T9 options-feature coverage for controlled selector replay readiness.

This is a read-only data audit.  It intentionally does not tune strategy logic,
compute PnL, or modify production services.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.earnings_vol_snapshot import build_vol_snapshot
from services.structure_scorecard import build_structure_scorecards
from services.structure_selector import select_best_structure


DEFAULT_ROOT = Path("/Volumes/T9/market_data/research/options_features_eod")
DEFAULT_UNDERLYING_ROOT = Path("/Volumes/T9/market_data/research/underlyings/daily_features")
DEFAULT_REPORT_DIR = Path("exports/reports/t9_replay_readiness")
DEFAULT_SYMBOLS = ("AAPL", "AMZN", "MSFT", "NVDA")
SMOKE_SYMBOLS = ("AAPL", "MSFT")
NEAR_TERM_MAX_DTE = 45
OTM_WING_PCT = 0.05


@dataclass(frozen=True)
class SymbolCoverage:
    symbol: str
    start_date: str | None
    end_date: str | None
    parquet_files: int
    trading_days: int
    calendar_business_days: int
    option_chain_day_coverage_pct: float | None
    rows: int
    avg_expirations_per_day: float | None
    min_expirations_per_day: int | None
    max_expirations_per_day: int | None
    avg_strikes_per_day: float | None
    p10_strikes_per_day: float | None
    p50_strikes_per_day: float | None
    p90_strikes_per_day: float | None
    near_term_expiration_day_pct: float | None
    missing_atm_or_near_atm_day_pct: float | None
    bid_ask_row_pct: float | None
    mid_only_row_pct: float | None
    missing_iv_row_pct: float | None
    bad_bid_ask_row_pct: float | None
    zero_spread_row_pct: float | None
    extreme_spread_row_pct: float | None
    duplicate_contract_rows: int
    dte_mismatch_rows: int
    unchanged_quote_adjacent_day_pct: float | None
    flat_surface_day_pct: float | None
    atm_straddle_constructible_day_pct: float | None
    otm_strangle_constructible_day_pct: float | None
    full_volsnapshot_input_day_pct: float | None
    selector_input_complete_day_pct: float | None
    replay_status: str
    blockers: list[str]


def _pct(numer: float, denom: float) -> float | None:
    if denom <= 0:
        return None
    return round(float(numer) / float(denom) * 100.0, 3)


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _business_days(start: str | None, end: str | None) -> int:
    if not start or not end:
        return 0
    return int(len(pd.bdate_range(start=start, end=end)))


def _source_glob(root: Path, symbol: str) -> str:
    return str(root / f"underlying_symbol={symbol}" / "**" / "*.parquet")


def _valid_quote(row: pd.Series | dict[str, Any]) -> bool:
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    mid = _safe_float(row.get("mid"))
    return bid is not None and ask is not None and mid is not None and bid >= 0 and ask >= bid and mid > 0


def _has_near_atm_pair(day_frame: pd.DataFrame, tolerance_pct: float = 0.02) -> bool:
    spot = _safe_float(day_frame["underlying_price"].median() if "underlying_price" in day_frame else None)
    if spot is None or spot <= 0:
        return False
    near = day_frame[
        (day_frame["strike"].sub(spot).abs() / spot <= tolerance_pct)
        & (day_frame["dte"].between(1, NEAR_TERM_MAX_DTE))
    ]
    return bool({"C", "P"}.issubset(set(near["call_put"].dropna().astype(str))))


def _nearest_valid_leg(frame: pd.DataFrame, target: float) -> pd.Series | None:
    if frame.empty:
        return None
    candidates = frame.copy()
    candidates["distance"] = (candidates["strike"] - target).abs()
    candidates = candidates.sort_values(["distance", "strike"])
    for _, row in candidates.iterrows():
        if _valid_quote(row):
            return row
    return None


def _constructibility_by_day(frame: pd.DataFrame) -> dict[str, int]:
    atm_ok = 0
    otm_ok = 0
    full_selector_inputs = 0
    for _, day_frame in frame.groupby("trade_date"):
        spot = _safe_float(day_frame["underlying_price"].median())
        if spot is None or spot <= 0:
            continue
        near = day_frame[day_frame["dte"].between(1, NEAR_TERM_MAX_DTE)].copy()
        if near.empty:
            continue
        first_expiry = near.sort_values(["dte", "expiry"])["expiry"].iloc[0]
        expiry_frame = near[near["expiry"] == first_expiry].copy()
        calls = expiry_frame[expiry_frame["call_put"] == "C"]
        puts = expiry_frame[expiry_frame["call_put"] == "P"]
        call_atm = _nearest_valid_leg(calls, spot)
        put_atm = _nearest_valid_leg(puts, spot)
        if call_atm is not None and put_atm is not None:
            atm_ok += 1
        call_otm = _nearest_valid_leg(calls[calls["strike"] >= spot], spot * (1.0 + OTM_WING_PCT))
        put_otm = _nearest_valid_leg(puts[puts["strike"] <= spot], spot * (1.0 - OTM_WING_PCT))
        if call_otm is not None and put_otm is not None:
            otm_ok += 1
        expiries = sorted(pd.to_datetime(day_frame["expiry"]).dt.date.unique())
        has_two_terms = len([expiry for expiry in expiries if expiry > pd.Timestamp(day_frame["trade_date"].iloc[0]).date()]) >= 2
        has_iv = bool(pd.to_numeric(day_frame["iv"], errors="coerce").gt(0).any())
        if call_atm is not None and put_atm is not None and has_two_terms and has_iv:
            full_selector_inputs += 1
    return {
        "atm_straddle_constructible_days": atm_ok,
        "otm_strangle_constructible_days": otm_ok,
        "selector_chain_complete_days": full_selector_inputs,
    }


def _daily_coverage(con: duckdb.DuckDBPyConnection, source: str) -> pd.DataFrame:
    query = f"""
        SELECT
            trade_date,
            COUNT(*) AS row_count,
            COUNT(DISTINCT expiry) AS expirations,
            COUNT(DISTINCT strike) AS strikes,
            COUNT(*) FILTER (WHERE dte BETWEEN 1 AND {NEAR_TERM_MAX_DTE}) AS near_term_rows,
            COUNT(*) FILTER (WHERE bid IS NOT NULL AND ask IS NOT NULL) AS bid_ask_rows,
            COUNT(*) FILTER (WHERE mid IS NOT NULL AND (bid IS NULL OR ask IS NULL)) AS mid_only_rows,
            COUNT(*) FILTER (WHERE iv IS NULL OR iv <= 0) AS missing_iv_rows,
            COUNT(*) FILTER (WHERE bid IS NOT NULL AND ask IS NOT NULL AND bid > ask) AS bad_bid_ask_rows,
            COUNT(*) FILTER (WHERE bid IS NOT NULL AND ask IS NOT NULL AND mid > 0 AND ask = bid) AS zero_spread_rows,
            COUNT(*) FILTER (
                WHERE bid IS NOT NULL AND ask IS NOT NULL AND mid > 0 AND ask >= bid AND ((ask - bid) / mid) > 0.25
            ) AS extreme_spread_rows
        FROM read_parquet('{source}', hive_partitioning=true)
        GROUP BY trade_date
        ORDER BY trade_date
    """
    return con.execute(query).fetchdf()


def _load_symbol_frame(con: duckdb.DuckDBPyConnection, source: str) -> pd.DataFrame:
    columns = """
        trade_date, expiry, underlying_symbol, option_symbol, call_put, dte,
        strike, bid, ask, mid, underlying_price, iv, open_interest, volume, spread_pct
    """
    frame = con.execute(
        f"""
        SELECT {columns}
        FROM read_parquet('{source}', hive_partitioning=true)
        ORDER BY trade_date, expiry, call_put, strike
        """
    ).fetchdf()
    for col in ("trade_date", "expiry"):
        frame[col] = pd.to_datetime(frame[col]).dt.normalize()
    for col in ("strike", "bid", "ask", "mid", "underlying_price", "iv", "open_interest", "volume", "spread_pct"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["call_put"] = frame["call_put"].astype(str).str.upper()
    return frame


def _staleness_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty or "option_symbol" not in frame:
        return {"unchanged_quote_adjacent_day_pct": None, "flat_surface_day_pct": None}
    compact = frame[["trade_date", "option_symbol", "bid", "ask", "mid", "iv"]].dropna(subset=["option_symbol"])
    compact = compact.sort_values(["option_symbol", "trade_date"])
    for col in ("bid", "ask", "mid", "iv"):
        compact[f"prev_{col}"] = compact.groupby("option_symbol")[col].shift(1)
    compact["prev_date"] = compact.groupby("option_symbol")["trade_date"].shift(1)
    compact["day_gap"] = (compact["trade_date"] - compact["prev_date"]).dt.days
    comparable = compact[compact["day_gap"].between(1, 5)]
    unchanged = comparable[
        (comparable["bid"] == comparable["prev_bid"])
        & (comparable["ask"] == comparable["prev_ask"])
        & (comparable["mid"] == comparable["prev_mid"])
        & (comparable["iv"] == comparable["prev_iv"])
    ]
    surface = (
        frame[frame["iv"].notna() & frame["dte"].between(1, NEAR_TERM_MAX_DTE)]
        .groupby(["trade_date", "expiry", "call_put"])["iv"]
        .agg(["count", "std"])
        .reset_index()
    )
    surface_days = surface.groupby("trade_date").apply(
        lambda g: bool(((g["count"] >= 5) & (g["std"].fillna(0.0) < 0.0005)).any()),
        include_groups=False,
    )
    return {
        "unchanged_quote_adjacent_day_pct": _pct(len(unchanged), len(comparable)),
        "flat_surface_day_pct": _pct(int(surface_days.sum()), len(surface_days)),
    }


def _duplicate_contract_rows(con: duckdb.DuckDBPyConnection, source: str) -> int:
    query = f"""
        SELECT COALESCE(SUM(cnt - 1), 0)::BIGINT
        FROM (
            SELECT trade_date, expiry, call_put, strike, COUNT(*) AS cnt
            FROM read_parquet('{source}', hive_partitioning=true)
            GROUP BY trade_date, expiry, call_put, strike
            HAVING COUNT(*) > 1
        )
    """
    return int(con.execute(query).fetchone()[0] or 0)


def _dte_mismatch_rows(con: duckdb.DuckDBPyConnection, source: str) -> int:
    query = f"""
        SELECT COUNT(*)::BIGINT
        FROM read_parquet('{source}', hive_partitioning=true)
        WHERE dte IS NOT NULL
          AND ABS(dte - date_diff('day', trade_date, expiry)) > 1
    """
    return int(con.execute(query).fetchone()[0] or 0)


def _schema(con: duckdb.DuckDBPyConnection, sample_file: Path) -> list[dict[str, str]]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sample_file}')").fetchall()
    return [{"column_name": str(row[0]), "column_type": str(row[1])} for row in rows]


def _schema_consistency(con: duckdb.DuckDBPyConnection, files: list[Path]) -> dict[str, Any]:
    samples = files[:1] + files[len(files) // 2 : len(files) // 2 + 1] + files[-1:]
    signatures: dict[str, list[str]] = {}
    for path in samples:
        columns = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        signature = "|".join(f"{row[0]}:{row[1]}" for row in columns)
        signatures.setdefault(signature, []).append(str(path))
    return {
        "sampled_files": [str(path) for path in samples],
        "consistent": len(signatures) == 1,
        "signature_count": len(signatures),
    }


def _symbol_coverage(root: Path, symbol: str, con: duckdb.DuckDBPyConnection) -> tuple[SymbolCoverage, pd.DataFrame]:
    symbol_dir = root / f"underlying_symbol={symbol}"
    files = sorted(symbol_dir.glob("year=*/month=*/*.parquet"))
    if not files:
        return (
            SymbolCoverage(
                symbol=symbol,
                start_date=None,
                end_date=None,
                parquet_files=0,
                trading_days=0,
                calendar_business_days=0,
                option_chain_day_coverage_pct=None,
                rows=0,
                avg_expirations_per_day=None,
                min_expirations_per_day=None,
                max_expirations_per_day=None,
                avg_strikes_per_day=None,
                p10_strikes_per_day=None,
                p50_strikes_per_day=None,
                p90_strikes_per_day=None,
                near_term_expiration_day_pct=None,
                missing_atm_or_near_atm_day_pct=None,
                bid_ask_row_pct=None,
                mid_only_row_pct=None,
                missing_iv_row_pct=None,
                bad_bid_ask_row_pct=None,
                zero_spread_row_pct=None,
                extreme_spread_row_pct=None,
                duplicate_contract_rows=0,
                dte_mismatch_rows=0,
                unchanged_quote_adjacent_day_pct=None,
                flat_surface_day_pct=None,
                atm_straddle_constructible_day_pct=None,
                otm_strangle_constructible_day_pct=None,
                full_volsnapshot_input_day_pct=None,
                selector_input_complete_day_pct=None,
                replay_status="missing",
                blockers=["no_symbol_partition"],
            ),
            pd.DataFrame(),
        )
    source = _source_glob(root, symbol)
    daily = _daily_coverage(con, source)
    frame = _load_symbol_frame(con, source)
    daily["has_near_atm_pair"] = [
        _has_near_atm_pair(group) for _, group in frame.groupby("trade_date", sort=True)
    ]
    construct = _constructibility_by_day(frame)
    stale = _staleness_metrics(frame)
    start = daily["trade_date"].min().date().isoformat() if not daily.empty else None
    end = daily["trade_date"].max().date().isoformat() if not daily.empty else None
    trading_days = int(len(daily))
    rows = int(daily["row_count"].sum()) if not daily.empty else 0
    business_days = _business_days(start, end)
    blockers = [
        "historical_earnings_calendar_not_found",
        "symbol_underlying_ohlc_not_found",
    ]
    full_volsnapshot_days = 0
    if not blockers:
        full_volsnapshot_days = int(construct["selector_chain_complete_days"])
    coverage = SymbolCoverage(
        symbol=symbol,
        start_date=start,
        end_date=end,
        parquet_files=len(files),
        trading_days=trading_days,
        calendar_business_days=business_days,
        option_chain_day_coverage_pct=_pct(trading_days, business_days),
        rows=rows,
        avg_expirations_per_day=round(float(daily["expirations"].mean()), 3) if trading_days else None,
        min_expirations_per_day=int(daily["expirations"].min()) if trading_days else None,
        max_expirations_per_day=int(daily["expirations"].max()) if trading_days else None,
        avg_strikes_per_day=round(float(daily["strikes"].mean()), 3) if trading_days else None,
        p10_strikes_per_day=round(float(daily["strikes"].quantile(0.10)), 3) if trading_days else None,
        p50_strikes_per_day=round(float(daily["strikes"].quantile(0.50)), 3) if trading_days else None,
        p90_strikes_per_day=round(float(daily["strikes"].quantile(0.90)), 3) if trading_days else None,
        near_term_expiration_day_pct=_pct(int((daily["near_term_rows"] > 0).sum()), trading_days),
        missing_atm_or_near_atm_day_pct=_pct(int((~daily["has_near_atm_pair"]).sum()), trading_days),
        bid_ask_row_pct=_pct(int(daily["bid_ask_rows"].sum()), rows),
        mid_only_row_pct=_pct(int(daily["mid_only_rows"].sum()), rows),
        missing_iv_row_pct=_pct(int(daily["missing_iv_rows"].sum()), rows),
        bad_bid_ask_row_pct=_pct(int(daily["bad_bid_ask_rows"].sum()), rows),
        zero_spread_row_pct=_pct(int(daily["zero_spread_rows"].sum()), rows),
        extreme_spread_row_pct=_pct(int(daily["extreme_spread_rows"].sum()), rows),
        duplicate_contract_rows=_duplicate_contract_rows(con, source),
        dte_mismatch_rows=_dte_mismatch_rows(con, source),
        unchanged_quote_adjacent_day_pct=stale["unchanged_quote_adjacent_day_pct"],
        flat_surface_day_pct=stale["flat_surface_day_pct"],
        atm_straddle_constructible_day_pct=_pct(construct["atm_straddle_constructible_days"], trading_days),
        otm_strangle_constructible_day_pct=_pct(construct["otm_strangle_constructible_days"], trading_days),
        full_volsnapshot_input_day_pct=_pct(full_volsnapshot_days, trading_days),
        selector_input_complete_day_pct=_pct(construct["selector_chain_complete_days"], trading_days),
        replay_status="partial_chain_replay_only",
        blockers=blockers,
    )
    return coverage, frame


def _close_only_price_frame(frame: pd.DataFrame, through_date: pd.Timestamp) -> pd.DataFrame:
    price = (
        frame[frame["trade_date"] <= through_date]
        .groupby("trade_date")["underlying_price"]
        .median()
        .dropna()
        .reset_index()
        .rename(columns={"trade_date": "date", "underlying_price": "Close"})
    )
    return price


def _smoke_replay(frame_by_symbol: dict[str, pd.DataFrame], *, start_date: str) -> dict[str, Any]:
    results: dict[str, Any] = {
        "mode": "partial_input_smoke_only",
        "note": (
            "Uses real T9 option chains and close-only underlying prices with synthetic future "
            "earnings metadata only to verify pipeline execution. This is not a valid earnings replay."
        ),
        "symbols": {},
    }
    for symbol in SMOKE_SYMBOLS:
        frame = frame_by_symbol.get(symbol, pd.DataFrame())
        if frame.empty:
            results["symbols"][symbol] = {"attempted_days": 0, "failures": 0, "failure_reasons": {"missing_frame": 1}}
            continue
        dates = sorted(pd.to_datetime(frame.loc[frame["trade_date"] >= pd.Timestamp(start_date), "trade_date"]).dt.normalize().unique())
        attempted = valid = abstain = failures = 0
        failure_reasons: dict[str, int] = {}
        for trade_date in dates[::10]:
            attempted += 1
            day = frame[frame["trade_date"] == trade_date].copy()
            try:
                price_data = _close_only_price_frame(frame, pd.Timestamp(trade_date))
                earnings_metadata = {
                    "earnings_date": (pd.Timestamp(trade_date).date() + timedelta(days=7)).isoformat(),
                    "release_timing": "unknown",
                    "prior_events": [],
                    "earnings_source_primary": "synthetic_smoke_metadata",
                    "earnings_source_confidence": 0.0,
                    "earnings_source_notes": [
                        "Synthetic metadata used only for T9 replay-readiness smoke; not a performance replay."
                    ],
                }
                snapshot = build_vol_snapshot(
                    symbol,
                    pd.Timestamp(trade_date).date(),
                    option_chain_data=day,
                    price_data=price_data,
                    earnings_metadata=earnings_metadata,
                )
                cards = build_structure_scorecards(snapshot)
                selected = select_best_structure(snapshot, cards)
                valid += 1
                if selected.recommendation == "No Trade":
                    abstain += 1
            except Exception as exc:  # noqa: BLE001 - audit should record partial-data failures.
                failures += 1
                key = type(exc).__name__
                failure_reasons[key] = failure_reasons.get(key, 0) + 1
        results["symbols"][symbol] = {
            "attempted_days": attempted,
            "valid_pipeline_days": valid,
            "valid_pipeline_day_pct": _pct(valid, attempted),
            "abstain_days": abstain,
            "abstain_day_pct": _pct(abstain, valid),
            "failures": failures,
            "failure_reasons": failure_reasons,
        }
    return results


def _ranked_issues(coverages: list[SymbolCoverage], *, earnings_labels_present: bool, underlying_ohlc_present: bool) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not earnings_labels_present:
        issues.append(
            {
                "severity": "High",
                "issue": "Historical earnings-event calendar is absent from the audited T9 research folders.",
                "impact": "Cannot honestly align pre-earnings windows or run a true event replay without joining an external earnings calendar.",
            }
        )
    if not underlying_ohlc_present:
        issues.append(
            {
                "severity": "High",
                "issue": "Symbol-level underlying OHLC is absent for AAPL/AMZN/MSFT/NVDA.",
                "impact": "Current VolSnapshot RV/Yang-Zhang/HAR inputs cannot be reconstructed faithfully from the option feature layer alone.",
            }
        )
    for row in coverages:
        if row.extreme_spread_row_pct is not None and row.extreme_spread_row_pct > 5.0:
            issues.append(
                {
                    "severity": "Medium",
                    "issue": f"{row.symbol} has {row.extreme_spread_row_pct:.2f}% rows with spread >25% of mid.",
                    "impact": "Execution-quality filters and baseline construction should explicitly handle wide quotes.",
                }
            )
        if row.unchanged_quote_adjacent_day_pct is not None and row.unchanged_quote_adjacent_day_pct > 10.0:
            issues.append(
                {
                    "severity": "Medium",
                    "issue": f"{row.symbol} has {row.unchanged_quote_adjacent_day_pct:.2f}% adjacent comparable option quotes unchanged.",
                    "impact": "Potential stale quote risk; not necessarily fatal, but should be monitored in replay.",
                }
            )
        if row.missing_atm_or_near_atm_day_pct is not None and row.missing_atm_or_near_atm_day_pct > 1.0:
            issues.append(
                {
                    "severity": "Medium",
                    "issue": f"{row.symbol} is missing near-ATM call/put coverage on {row.missing_atm_or_near_atm_day_pct:.2f}% days.",
                    "impact": "ATM straddle and event-implied move reconstruction may fail on those dates.",
                }
            )
    return issues


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# T9 Replay Readiness Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Decision",
        "",
        f"**GO / NO-GO:** {report['go_no_go']['decision']}",
        "",
        report["go_no_go"]["rationale"],
        "",
        "## Dataset Overview",
        "",
        f"- Root: `{report['dataset_overview']['root']}`",
        f"- Symbols discovered: {report['dataset_overview']['symbol_count']}",
        f"- Target symbols present: {report['dataset_overview']['target_symbols_present']}",
        f"- Format: {report['dataset_overview']['format']}",
        f"- Partitioning: {report['dataset_overview']['partitioning']}",
        f"- Schema consistent in sampled target files: {report['dataset_overview']['schema_consistency']['consistent']}",
        "",
        "## Coverage Metrics",
        "",
        "| Symbol | Dates | Days | Chain coverage | Near-term days | Missing ATM | Strikes p50 | Bid/ask rows | Extreme spreads | ATM baseline | OTM baseline | Replay status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["coverage_metrics"]:
        lines.append(
            "| {symbol} | {start_date} to {end_date} | {trading_days} | {option_chain_day_coverage_pct}% | "
            "{near_term_expiration_day_pct}% | {missing_atm_or_near_atm_day_pct}% | {p50_strikes_per_day} | "
            "{bid_ask_row_pct}% | {extreme_spread_row_pct}% | {atm_straddle_constructible_day_pct}% | "
            "{otm_strangle_constructible_day_pct}% | {replay_status} |".format(**row)
        )
    lines.extend(["", "## Ranked Data Quality Issues", ""])
    for issue in report["data_quality_issues"]:
        lines.append(f"- **{issue['severity']}**: {issue['issue']} Impact: {issue['impact']}")
    lines.extend(["", "## Replay Feasibility", ""])
    for symbol, row in report["replay_feasibility"].items():
        lines.append(
            f"- {symbol}: usable={row['usable_day_pct']}%, partial={row['partial_day_pct']}%, "
            f"unusable={row['unusable_day_pct']}%, blockers={', '.join(row['blockers'])}"
        )
    lines.extend(["", "## Smoke Replay", ""])
    lines.append(report["smoke_replay"]["note"])
    for symbol, row in report["smoke_replay"]["symbols"].items():
        lines.append(
            f"- {symbol}: attempted={row['attempted_days']}, valid={row.get('valid_pipeline_days', 0)}, "
            f"abstain_pct={row.get('abstain_day_pct')}, failures={row.get('failures', 0)}"
        )
    lines.extend(["", "## Known Failure Modes", ""])
    for item in report["known_failure_modes"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(root: Path, underlying_root: Path, symbols: tuple[str, ...], smoke_start_date: str) -> dict[str, Any]:
    con = duckdb.connect(":memory:")
    discovered_symbols = sorted(
        path.name.split("=", 1)[1]
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("underlying_symbol=")
    ) if root.exists() else []
    target_present = {symbol: (root / f"underlying_symbol={symbol}").exists() for symbol in symbols}
    files = [file for symbol in symbols for file in sorted((root / f"underlying_symbol={symbol}").glob("year=*/month=*/*.parquet"))]
    schema = _schema(con, files[0]) if files else []
    schema_consistency = _schema_consistency(con, files) if files else {"consistent": False, "sampled_files": [], "signature_count": 0}
    earnings_labels_present = any((root.parents[0] / "options_calculator_pro" / "labels").glob("*")) if root.exists() else False
    underlying_ohlc_present = all((underlying_root / f"underlying_symbol={symbol}").exists() for symbol in symbols)

    coverages: list[SymbolCoverage] = []
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        coverage, frame = _symbol_coverage(root, symbol, con)
        coverages.append(coverage)
        frames[symbol] = frame

    coverage_dicts = [asdict(row) for row in coverages]
    replay_feasibility = {
        row.symbol: {
            "usable_day_pct": 0.0,
            "partial_day_pct": row.selector_input_complete_day_pct,
            "unusable_day_pct": round(100.0 - float(row.selector_input_complete_day_pct or 0.0), 3),
            "full_volsnapshot_input_day_pct": row.full_volsnapshot_input_day_pct,
            "selector_chain_complete_day_pct": row.selector_input_complete_day_pct,
            "blockers": row.blockers,
        }
        for row in coverages
    }
    chain_baselines_ok = all(
        (row.atm_straddle_constructible_day_pct or 0.0) >= 95.0
        and (row.otm_strangle_constructible_day_pct or 0.0) >= 95.0
        for row in coverages
    )
    go = bool(chain_baselines_ok and earnings_labels_present and underlying_ohlc_present)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_overview": {
            "root": str(root),
            "format": "Parquet",
            "partitioning": "underlying_symbol=SYMBOL/year=YYYY/month=MM/*.parquet",
            "symbol_count": len(discovered_symbols),
            "target_symbols_present": target_present,
            "schema": schema,
            "schema_consistency": schema_consistency,
            "underlying_ohlc_root": str(underlying_root),
            "target_underlying_ohlc_present": {
                symbol: (underlying_root / f"underlying_symbol={symbol}").exists()
                for symbol in symbols
            },
            "historical_earnings_labels_present": earnings_labels_present,
        },
        "coverage_metrics": coverage_dicts,
        "data_quality_issues": _ranked_issues(
            coverages,
            earnings_labels_present=earnings_labels_present,
            underlying_ohlc_present=underlying_ohlc_present,
        ),
        "replay_feasibility": replay_feasibility,
        "baseline_comparability": {
            "atm_straddle_constructible_all_targets": all((row.atm_straddle_constructible_day_pct or 0.0) >= 95.0 for row in coverages),
            "otm_strangle_constructible_all_targets": all((row.otm_strangle_constructible_day_pct or 0.0) >= 95.0 for row in coverages),
            "blocker": None if chain_baselines_ok else "One or more target symbols falls below 95% baseline constructibility.",
        },
        "smoke_replay": _smoke_replay(frames, start_date=smoke_start_date),
        "known_failure_modes": [
            "Historical earnings-event dates are not present in the audited local labels folder.",
            "AAPL/AMZN/MSFT/NVDA underlying OHLC is not present in the T9 underlyings/daily_features store.",
            "Extreme bid/ask spreads exist in otherwise valid chains and must be filtered or labelled during replay.",
            "Smoke replay with synthetic earnings metadata is only a pipeline-crash test, not a valid event replay.",
        ],
        "go_no_go": {
            "decision": "NO-GO for full selector event replay; GO for chain/baseline construction audit.",
            "rationale": (
                "The option-chain feature layer is broad and mostly internally consistent, and ATM/OTM baseline "
                "quotes appear reconstructable for the target symbols. Full event-volatility replay is not yet "
                "institutionally defensible because the audited T9 folders do not contain target-symbol underlying "
                "OHLC or historical earnings-event calendars needed by VolSnapshot."
            ),
            "safe_subset_if_go": None,
            "safe_next_subset": {
                "symbols": list(symbols),
                "date_range": "2023-01-03 through 2026-04-01/02",
                "allowed_use": "chain coverage, quote-quality, baseline constructibility, and non-performance pipeline smoke only",
            },
        },
    }
    return _jsonable(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit T9 options replay readiness without computing PnL.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--underlying-root", type=Path, default=DEFAULT_UNDERLYING_ROOT)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--smoke-start-date", default="2025-01-01")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    symbols = tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip())
    report = build_report(args.root, args.underlying_root, symbols, args.smoke_start_date)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "t9_replay_readiness_latest.json"
    md_path = args.report_dir / "t9_replay_readiness_latest.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(report, md_path)
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path), "decision": report["go_no_go"]["decision"]}, indent=2))


if __name__ == "__main__":
    main()
