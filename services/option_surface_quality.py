from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptionSurfaceQualityDiagnostics:
    status: str
    warning_flags: list[str] = field(default_factory=list)
    row_count: int = 0
    expiration_count: int = 0
    crossed_quote_count: int = 0
    zero_bid_count: int = 0
    extreme_spread_count: int = 0
    missing_iv_count: int = 0
    iv_outlier_count: int = 0
    sparse_atm_expiration_count: int = 0
    term_structure_anomaly_count: int = 0
    put_call_parity_warning_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_option_surface_quality(
    chain_frame: pd.DataFrame | None,
    *,
    underlying_price: float | None = None,
    as_of_date: date | None = None,
    extreme_spread_pct: float = 25.0,
    atm_window_pct: float = 0.05,
) -> OptionSurfaceQualityDiagnostics:
    if chain_frame is None or chain_frame.empty:
        return OptionSurfaceQualityDiagnostics(
            status="record_only",
            warning_flags=["missing_option_chain"],
        )

    df = chain_frame.copy()
    if "iv" not in df.columns and "impliedVolatility" in df.columns:
        df["iv"] = df["impliedVolatility"]
    if "expiration" not in df.columns and "expiration_date" in df.columns:
        df["expiration"] = df["expiration_date"]
    if "expiration" not in df.columns and "expiry" in df.columns:
        df["expiration"] = df["expiry"]
    if "side" not in df.columns and "call_put" in df.columns:
        side = df["call_put"].astype(str).str.upper()
        df["side"] = np.where(side.str.startswith("C"), "call", np.where(side.str.startswith("P"), "put", side))
    if "mid" not in df.columns and {"bid", "ask"}.issubset(df.columns):
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")
        valid = bid.notna() & ask.notna() & (ask >= bid)
        df["mid"] = np.nan
        df.loc[valid, "mid"] = (bid.loc[valid] + ask.loc[valid]) / 2.0
    row_count = int(len(df))
    for col in ("strike", "bid", "ask", "mid", "iv", "expiration", "side"):
        if col not in df.columns:
            df[col] = np.nan
    for col in ("strike", "bid", "ask", "mid", "iv"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "mid" not in df.columns or df["mid"].isna().all():
        valid = df["bid"].notna() & df["ask"].notna() & (df["ask"] >= df["bid"])
        df.loc[valid, "mid"] = (df.loc[valid, "bid"] + df.loc[valid, "ask"]) / 2.0

    crossed = int(((df["bid"].notna()) & (df["ask"].notna()) & (df["ask"] < df["bid"])).sum())
    zero_bid = int(((df["bid"].fillna(np.nan) == 0) & (df["ask"].fillna(0) > 0)).sum())
    valid_spread = df["bid"].notna() & df["ask"].notna() & df["mid"].notna() & (df["mid"] > 0) & (df["ask"] >= df["bid"])
    spread_pct = pd.Series(np.nan, index=df.index)
    spread_pct.loc[valid_spread] = ((df.loc[valid_spread, "ask"] - df.loc[valid_spread, "bid"]) / df.loc[valid_spread, "mid"]) * 100.0
    extreme_spread = int((spread_pct >= extreme_spread_pct).sum())
    missing_iv = int((df["iv"].isna() | (df["iv"] <= 0)).sum())
    iv_outliers = int(((df["iv"] > 3.0) | (df["iv"] < 0.01)).sum())

    expiration_count = int(df["expiration"].dropna().astype(str).nunique())
    sparse_atm = 0
    if underlying_price is not None and underlying_price > 0:
        low = float(underlying_price) * (1.0 - atm_window_pct)
        high = float(underlying_price) * (1.0 + atm_window_pct)
        for _expiry, group in df.groupby(df["expiration"].astype(str)):
            atm_rows = group[(group["strike"] >= low) & (group["strike"] <= high)]
            if len(atm_rows) < 4:
                sparse_atm += 1

    term_anomalies = _term_structure_anomaly_count(df)
    parity_warnings = _put_call_parity_warning_count(df, underlying_price)

    flags: list[str] = []
    if crossed:
        flags.append("crossed_quotes")
    if zero_bid:
        flags.append("zero_bid_quotes")
    if extreme_spread:
        flags.append("extreme_bid_ask_spreads")
    if missing_iv:
        flags.append("missing_or_non_positive_iv")
    if iv_outliers:
        flags.append("iv_outliers")
    if sparse_atm:
        flags.append("sparse_strikes_around_atm")
    if expiration_count < 2:
        flags.append("missing_expiration_depth")
    if term_anomalies:
        flags.append("term_structure_anomalies")
    if parity_warnings:
        flags.append("put_call_parity_warnings")

    blocking = bool(crossed or expiration_count == 0 or row_count < 10)
    degraded = bool(flags)
    status = "record_only" if blocking else "degraded_surface" if degraded else "clean_surface"
    return OptionSurfaceQualityDiagnostics(
        status=status,
        warning_flags=flags,
        row_count=row_count,
        expiration_count=expiration_count,
        crossed_quote_count=crossed,
        zero_bid_count=zero_bid,
        extreme_spread_count=extreme_spread,
        missing_iv_count=missing_iv,
        iv_outlier_count=iv_outliers,
        sparse_atm_expiration_count=sparse_atm,
        term_structure_anomaly_count=term_anomalies,
        put_call_parity_warning_count=parity_warnings,
    )


def _term_structure_anomaly_count(df: pd.DataFrame) -> int:
    if "expiration" not in df.columns or df.empty:
        return 0
    atm = df.dropna(subset=["expiration", "iv", "strike"])
    if atm.empty:
        return 0
    expiry_iv = atm.groupby(atm["expiration"].astype(str))["iv"].median().dropna()
    if len(expiry_iv) < 2:
        return 0
    values = expiry_iv.to_numpy(dtype=float)
    diffs = np.diff(values)
    return int((np.abs(diffs) > 0.75).sum())


def _put_call_parity_warning_count(df: pd.DataFrame, underlying_price: float | None) -> int:
    if underlying_price is None or underlying_price <= 0 or "side" not in df.columns:
        return 0
    work = df.dropna(subset=["expiration", "strike", "mid"]).copy()
    if work.empty:
        return 0
    side = work["side"].astype(str).str.lower()
    work["_side"] = np.where(side.str.startswith("c"), "call", np.where(side.str.startswith("p"), "put", "other"))
    warnings = 0
    for (_expiry, strike), group in work.groupby([work["expiration"].astype(str), "strike"]):
        calls = group[group["_side"] == "call"]
        puts = group[group["_side"] == "put"]
        if calls.empty or puts.empty:
            continue
        call_mid = float(calls["mid"].iloc[0])
        put_mid = float(puts["mid"].iloc[0])
        synthetic_forward = float(strike) + call_mid - put_mid
        if abs(synthetic_forward - float(underlying_price)) / float(underlying_price) > 0.15:
            warnings += 1
    return warnings
