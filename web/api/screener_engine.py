from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import threading
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from services.earnings_event_service import resolve_upcoming_earnings_event
from services.market_data_client import MarketDataClient

ExpiryMode = Literal["front_after_earnings", "next_monthly_opex"]

DEFAULT_UNIVERSE: list[str] = [
    "AMZN", "AAPL", "GOOGL", "MSFT", "ADBE", "PANW", "NFLX", "FTNT", "NKE",
    "META", "NVDA", "TSLA", "AVGO", "AMD", "ORCL", "CRM", "INTU", "NOW",
    "QCOM", "INTC", "LRCX", "KLAC", "CDNS", "SNPS", "AMAT",
    "TTD", "ZS", "OKTA", "TEAM", "WDAY", "SNOW", "DDOG", "NET",
    "PYPL", "EBAY", "DIS", "UBER", "COST", "SBUX", "LULU",
]

QUALIFIED_SPREAD_PCT = 3.0
MARGINAL_SPREAD_PCT = 5.0
MIN_OI = 100
TARGET_OTM_PCT = 0.03
_SPREAD_DRIFT_EPSILON = 0.05

_spread_snapshot_cache: dict[str, dict[str, Any]] = {}
_spread_cache_lock = threading.Lock()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        num = float(value)
        if not np.isfinite(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _mid_from_bid_ask(bid: Any, ask: Any) -> Optional[float]:
    bid_val = _safe_float(bid)
    ask_val = _safe_float(ask)
    if bid_val is None or ask_val is None or bid_val < 0 or ask_val < 0:
        return None
    mid_val = (bid_val + ask_val) / 2.0
    return mid_val if mid_val > 0 else None


def _spread_pct_from_bid_ask(bid: Any, ask: Any) -> Optional[float]:
    mid_val = _mid_from_bid_ask(bid, ask)
    bid_val = _safe_float(bid)
    ask_val = _safe_float(ask)
    if mid_val is None or bid_val is None or ask_val is None:
        return None
    return ((ask_val - bid_val) / mid_val) * 100.0


def _business_days_before(event_date: date, offset: int) -> date:
    d = event_date
    steps = 0
    while steps < offset:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            steps += 1
    return d


def _extract_calendar_dates(raw_calendar: Any) -> list[date]:
    values: list[Any] = []
    if raw_calendar is None:
        return []
    if isinstance(raw_calendar, dict):
        values = raw_calendar.get("Earnings Date", []) or []
    elif isinstance(raw_calendar, pd.DataFrame):
        if "Earnings Date" in raw_calendar.index:
            raw = raw_calendar.loc["Earnings Date"]
            if isinstance(raw, pd.Series):
                values = raw.tolist()
            else:
                values = [raw]
        elif "Earnings Date" in raw_calendar.columns:
            values = raw_calendar["Earnings Date"].tolist()
    elif hasattr(raw_calendar, "tolist"):
        values = list(raw_calendar.tolist())

    parsed: list[date] = []
    for value in values if isinstance(values, list) else [values]:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        parsed.append(ts.date())
    return sorted({d for d in parsed})


def _infer_release_timing(info: dict[str, Any]) -> str:
    ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
    if ts is None:
        return "UNKNOWN"
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return "UNKNOWN"
    if dt.hour >= 20:
        return "AMC"
    if dt.hour < 16:
        return "BMO"
    return "UNKNOWN"


def _load_upcoming_event(
    symbol: str,
    today: date,
    cutoff: date,
    *,
    mda_client: Optional[MarketDataClient] = None,
) -> tuple[Optional[date], str, dict[str, Any], yf.Ticker, dict[str, Any]]:
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    resolved_event = resolve_upcoming_earnings_event(
        symbol,
        today,
        cutoff,
        ticker=ticker,
        mda_client=mda_client,
    )
    event_date = resolved_event.earnings_date
    release_timing = resolved_event.release_timing if event_date is not None else _infer_release_timing(info)
    return event_date, release_timing, info, ticker, {
        "earnings_source_primary": resolved_event.primary_source,
        "earnings_source_confirmed": resolved_event.confirmed_source,
        "earnings_source_confidence": resolved_event.source_confidence,
        "earnings_source_stale": resolved_event.source_stale,
        "release_timing_source": resolved_event.release_timing_source,
        "earnings_source_notes": resolved_event.source_notes,
    }


def _pick_front_expiry_after(expirations: Iterable[str], event_date: date) -> Optional[str]:
    for exp_str in sorted(expirations):
        try:
            if date.fromisoformat(str(exp_str)[:10]) > event_date:
                return str(exp_str)[:10]
        except ValueError:
            continue
    return None


def _pick_next_monthly_opex(expirations: Iterable[str], event_date: date) -> Optional[str]:
    fallback = None
    for exp_str in sorted(expirations):
        exp_norm = str(exp_str)[:10]
        try:
            exp_date = date.fromisoformat(exp_norm)
        except ValueError:
            continue
        if exp_date <= event_date:
            continue
        if fallback is None:
            fallback = exp_norm
        if exp_date.day >= 15:
            return exp_norm
    return fallback


def _normalize_yfinance_chain(frame: pd.DataFrame, side: str, expiry: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    normalized["side"] = "call" if side == "call" else "put"
    normalized["optionSymbol"] = normalized.get("contractSymbol")
    # `expiry` is the queried expiration (YYYY-MM-DD); broadcast across all rows so
    # downstream consumers reading `expiration_date` get the actual expiry rather
    # than the per-row last-trade timestamp (which is what was previously written
    # here and is unrelated to the contract's expiration).
    normalized["expiration_date"] = pd.to_datetime(expiry).strftime("%Y-%m-%d")
    if "openInterest" not in normalized.columns:
        normalized["openInterest"] = np.nan
    return normalized


def _load_chain_and_spot(
    symbol: str,
    expiry: str,
    *,
    ticker: yf.Ticker,
    mda_client: Optional[MarketDataClient] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[float], datetime]:
    now_utc = datetime.now(timezone.utc)
    if mda_client and mda_client.is_available():
        chain = mda_client.get_option_chain(symbol, expiration=expiry, strike_limit=40)
        if not chain.empty:
            calls = chain[chain["side"].astype(str).str.lower() == "call"].copy()
            puts = chain[chain["side"].astype(str).str.lower() == "put"].copy()
            spot = _safe_float(chain.get("underlyingPrice").dropna().iloc[0] if "underlyingPrice" in chain and not chain["underlyingPrice"].dropna().empty else None)
            updated_col = pd.to_datetime(chain.get("updated"), unit="s", utc=True, errors="coerce") if "updated" in chain.columns else None
            updated_at = now_utc
            if updated_col is not None and not updated_col.dropna().empty:
                updated_at = updated_col.dropna().max().to_pydatetime()
            if spot is None:
                spot = _safe_float(mda_client.get_quote(symbol))
            return calls, puts, spot, updated_at

    chain = ticker.option_chain(expiry)
    calls = _normalize_yfinance_chain(chain.calls, "call", expiry)
    puts = _normalize_yfinance_chain(chain.puts, "put", expiry)
    spot = _safe_float(getattr(ticker.fast_info, "last_price", None))
    if spot is None:
        fast_info = getattr(ticker, "fast_info", {}) or {}
        spot = _safe_float(fast_info.get("lastPrice") if isinstance(fast_info, dict) else None)
    return calls, puts, spot, now_utc


def _select_otm_leg(frame: pd.DataFrame, *, spot: float, side: str, target_otm_pct: float) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None

    strikes = pd.to_numeric(frame.get("strike"), errors="coerce")
    working = frame.assign(strike_numeric=strikes).dropna(subset=["strike_numeric"]).copy()
    if working.empty:
        return None

    if side == "call":
        target_strike = spot * (1.0 + target_otm_pct)
        working = working[working["strike_numeric"] >= spot * 1.01]
        if working.empty:
            return None
        working["_distance"] = (working["strike_numeric"] - target_strike).abs()
    else:
        target_strike = spot * (1.0 - target_otm_pct)
        working = working[working["strike_numeric"] <= spot * 0.99]
        if working.empty:
            return None
        working["_distance"] = (working["strike_numeric"] - target_strike).abs()

    return working.sort_values(["_distance", "strike_numeric"]).iloc[0]


def _implied_move_pct(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> Optional[float]:
    if spot <= 0 or calls.empty or puts.empty:
        return None
    calls_work = calls.assign(strike_numeric=pd.to_numeric(calls.get("strike"), errors="coerce")).dropna(subset=["strike_numeric"])
    puts_work = puts.assign(strike_numeric=pd.to_numeric(puts.get("strike"), errors="coerce")).dropna(subset=["strike_numeric"])
    if calls_work.empty or puts_work.empty:
        return None
    call_row = calls_work[calls_work["strike_numeric"] >= spot].sort_values("strike_numeric").head(1)
    put_row = puts_work[puts_work["strike_numeric"] <= spot].sort_values("strike_numeric", ascending=False).head(1)
    if call_row.empty or put_row.empty:
        return None
    call_mid = _mid_from_bid_ask(call_row.iloc[0].get("bid"), call_row.iloc[0].get("ask"))
    put_mid = _mid_from_bid_ask(put_row.iloc[0].get("bid"), put_row.iloc[0].get("ask"))
    if call_mid is None or put_mid is None:
        return None
    return ((call_mid + put_mid) / spot) * 100.0


def _status_from_checks(release_timing: str, oi_ok: bool, avg_spread_pct: Optional[float]) -> tuple[str, str, list[str]]:
    caveats: list[str] = []
    if release_timing != "AMC":
        return "EXCLUDED", "AMC-only timing is required for this pre-earnings setup.", caveats
    if not oi_ok:
        return "EXCLUDED", "Liquidity fails the minimum open-interest threshold.", caveats
    if avg_spread_pct is None:
        return "EXCLUDED", "Spread snapshot is unavailable for one or both legs.", ["Missing live bid/ask on at least one leg."]
    if avg_spread_pct <= QUALIFIED_SPREAD_PCT:
        return "QUALIFIED", "Spread and liquidity are inside the current paper-trade threshold.", caveats
    if avg_spread_pct <= MARGINAL_SPREAD_PCT:
        caveats.append("Spread is inside watch-list range only; re-check close to the decision time.")
        return "MARGINAL", "Spread is acceptable for monitoring but not inside the strict qualify band.", caveats
    return "EXCLUDED", "Average spread is too wide for the current research threshold.", caveats


def _spread_drift(key: str, current_spread_pct: Optional[float], observed_at: datetime) -> tuple[Optional[float], Optional[float], str]:
    with _spread_cache_lock:
        cached = _spread_snapshot_cache.get(key)
        previous_spread_pct = cached.get("avg_spread_pct") if cached else None
        change = None
        state = "new"
        if current_spread_pct is not None and previous_spread_pct is not None:
            change = current_spread_pct - previous_spread_pct
            if abs(change) <= _SPREAD_DRIFT_EPSILON:
                state = "unchanged"
            elif change < 0:
                state = "improved"
            else:
                state = "worsened"
        elif current_spread_pct is None:
            state = "unchanged"

        _spread_snapshot_cache[key] = {
            "avg_spread_pct": current_spread_pct,
            "observed_at": observed_at,
        }
    return previous_spread_pct, change, state


def _screen_one_symbol(
    symbol: str,
    expiry_mode: ExpiryMode,
    as_of_date: date,
    cutoff: date,
    mda_client: Optional[MarketDataClient],
) -> Optional[dict[str, Any]]:
    """Process a single symbol and return its screener row, or None if no earnings found."""
    try:
        event_date, release_timing, _info, ticker, event_provenance = _load_upcoming_event(
            symbol,
            as_of_date,
            cutoff,
            mda_client=mda_client,
        )
    except Exception:
        return None
    if event_date is None:
        return None

    entry_date = _business_days_before(event_date, 3)
    expirations: list[str] = []
    if mda_client and mda_client.is_available():
        try:
            expirations = mda_client.get_expirations(symbol)
        except Exception:
            pass
    if not expirations:
        expirations = [str(exp)[:10] for exp in getattr(ticker, "options", [])]

    front_expiry = _pick_front_expiry_after(expirations, event_date) if expirations else None
    monthly_expiry = _pick_next_monthly_opex(expirations, event_date) if expirations else None
    selected_expiry = front_expiry if expiry_mode == "front_after_earnings" else monthly_expiry
    alternative_expiry = monthly_expiry if expiry_mode == "front_after_earnings" else front_expiry

    if not selected_expiry:
        observed_at = datetime.now(timezone.utc)
        return {
            "symbol": symbol,
            "earnings_date": event_date,
            "release_timing": release_timing,
            "entry_date": entry_date,
            "entry_label": "T-3",
            "selected_expiry": None,
            "alternative_expiry": alternative_expiry,
            "expiry_mode": expiry_mode,
            "avg_spread_pct": None,
            "previous_avg_spread_pct": None,
            "spread_change_pct": None,
            "spread_change_state": "unchanged",
            "call_oi": None,
            "put_oi": None,
            "implied_move_pct": None,
            "call_strike": None,
            "put_strike": None,
            "call_iv": None,
            "put_iv": None,
            "entry_debit_mid": None,
            "status": "EXCLUDED",
            "status_reason": "No expiry is available under the selected methodology.",
            "compact_signal_summary": ["No valid expiry after earnings."],
            "caveats": [],
            "checks": [
                {"label": "Expiry selection", "threshold": expiry_mode.replace("_", " "), "actual": "missing", "passed": False, "severity": "hard", "note": None},
            ],
            "notes": [],
            "last_updated": observed_at,
            "detail_metrics": {
                "front_expiry": front_expiry,
                "monthly_expiry": monthly_expiry,
                "days_to_earnings": (event_date - as_of_date).days,
                "days_to_entry": (entry_date - as_of_date).days,
                **event_provenance,
            },
        }

    try:
        calls, puts, spot, observed_at = _load_chain_and_spot(
            symbol,
            selected_expiry,
            ticker=ticker,
            mda_client=mda_client,
        )
    except Exception:
        return None

    if spot is None:
        spot = _safe_float(calls.get("underlyingPrice").dropna().iloc[0] if "underlyingPrice" in calls and not calls["underlyingPrice"].dropna().empty else None)

    if spot is None:
        return {
            "symbol": symbol,
            "earnings_date": event_date,
            "release_timing": release_timing,
            "entry_date": entry_date,
            "entry_label": "T-3",
            "selected_expiry": selected_expiry,
            "alternative_expiry": alternative_expiry,
            "expiry_mode": expiry_mode,
            "avg_spread_pct": None,
            "previous_avg_spread_pct": None,
            "spread_change_pct": None,
            "spread_change_state": "unchanged",
            "call_oi": None,
            "put_oi": None,
            "implied_move_pct": None,
            "call_strike": None,
            "put_strike": None,
            "call_iv": None,
            "put_iv": None,
            "entry_debit_mid": None,
            "status": "EXCLUDED",
            "status_reason": "Underlying price is unavailable, so strike selection cannot be trusted.",
            "compact_signal_summary": ["Spot unavailable."],
            "caveats": [],
            "checks": [
                {"label": "Spot price", "threshold": "required", "actual": "missing", "passed": False, "severity": "hard", "note": None},
            ],
            "notes": [],
            "last_updated": observed_at,
            "detail_metrics": {
                "front_expiry": front_expiry,
                "monthly_expiry": monthly_expiry,
                "days_to_earnings": (event_date - as_of_date).days,
                "days_to_entry": (entry_date - as_of_date).days,
                **event_provenance,
            },
        }

    call_row = _select_otm_leg(calls, spot=spot, side="call", target_otm_pct=TARGET_OTM_PCT)
    put_row = _select_otm_leg(puts, spot=spot, side="put", target_otm_pct=TARGET_OTM_PCT)

    if call_row is None or put_row is None:
        return {
            "symbol": symbol,
            "earnings_date": event_date,
            "release_timing": release_timing,
            "entry_date": entry_date,
            "entry_label": "T-3",
            "selected_expiry": selected_expiry,
            "alternative_expiry": alternative_expiry,
            "expiry_mode": expiry_mode,
            "avg_spread_pct": None,
            "previous_avg_spread_pct": None,
            "spread_change_pct": None,
            "spread_change_state": "unchanged",
            "call_oi": None,
            "put_oi": None,
            "implied_move_pct": None,
            "call_strike": None,
            "put_strike": None,
            "call_iv": None,
            "put_iv": None,
            "entry_debit_mid": None,
            "status": "EXCLUDED",
            "status_reason": "The live chain does not contain a clean 3% OTM strangle pair.",
            "compact_signal_summary": ["No valid 3% OTM call/put pair."],
            "caveats": [],
            "checks": [
                {"label": "OTM structure", "threshold": "3% OTM call + 3% OTM put", "actual": "missing leg", "passed": False, "severity": "hard", "note": None},
            ],
            "notes": [],
            "last_updated": observed_at,
            "detail_metrics": {
                "spot": spot,
                "front_expiry": front_expiry,
                "monthly_expiry": monthly_expiry,
                "days_to_earnings": (event_date - as_of_date).days,
                "days_to_entry": (entry_date - as_of_date).days,
                **event_provenance,
            },
        }

    call_spread_pct = _spread_pct_from_bid_ask(call_row.get("bid"), call_row.get("ask"))
    put_spread_pct = _spread_pct_from_bid_ask(put_row.get("bid"), put_row.get("ask"))
    spread_values = [value for value in (call_spread_pct, put_spread_pct) if value is not None]
    avg_spread_pct = float(np.mean(spread_values)) if spread_values else None
    call_oi = _safe_int(call_row.get("openInterest"))
    put_oi = _safe_int(put_row.get("openInterest"))
    oi_ok = (call_oi or 0) >= MIN_OI and (put_oi or 0) >= MIN_OI
    status, status_reason, caveats = _status_from_checks(release_timing, oi_ok, avg_spread_pct)
    implied_move_pct = _implied_move_pct(calls, puts, spot)
    call_mid = _mid_from_bid_ask(call_row.get("bid"), call_row.get("ask"))
    put_mid = _mid_from_bid_ask(put_row.get("bid"), put_row.get("ask"))
    entry_debit_mid = (call_mid or 0.0) + (put_mid or 0.0) if (call_mid is not None and put_mid is not None) else None

    cache_key = "|".join([symbol, str(event_date), expiry_mode, selected_expiry])
    previous_spread_pct, spread_change_pct, spread_change_state = _spread_drift(cache_key, avg_spread_pct, observed_at)

    checks = [
        {
            "label": "Release timing",
            "threshold": "AMC only",
            "actual": release_timing,
            "passed": release_timing == "AMC",
            "severity": "hard",
            "note": "BMO setups are excluded because the decision window differs.",
        },
        {
            "label": "Average spread",
            "threshold": f"<= {QUALIFIED_SPREAD_PCT:.1f}% qualify / <= {MARGINAL_SPREAD_PCT:.1f}% watch",
            "actual": f"{avg_spread_pct:.2f}%" if avg_spread_pct is not None else "missing",
            "passed": avg_spread_pct is not None and avg_spread_pct <= MARGINAL_SPREAD_PCT,
            "severity": "hard",
            "note": "Spread quality is the primary gate for this screener.",
        },
        {
            "label": "Call open interest",
            "threshold": f">= {MIN_OI}",
            "actual": str(call_oi) if call_oi is not None else "missing",
            "passed": (call_oi or 0) >= MIN_OI,
            "severity": "hard",
            "note": None,
        },
        {
            "label": "Put open interest",
            "threshold": f">= {MIN_OI}",
            "actual": str(put_oi) if put_oi is not None else "missing",
            "passed": (put_oi or 0) >= MIN_OI,
            "severity": "hard",
            "note": None,
        },
    ]

    signal_summary = [
        f"Avg spread {avg_spread_pct:.2f}%" if avg_spread_pct is not None else "Spread unavailable",
        f"OI {call_oi or 0}/{put_oi or 0}",
        f"Impl move {implied_move_pct:.1f}%" if implied_move_pct is not None else "Impl move n/a",
    ]
    if spread_change_state != "new" and spread_change_pct is not None:
        direction = "tightened" if spread_change_pct < 0 else "widened"
        signal_summary.append(f"Spread {direction} {abs(spread_change_pct):.2f} pts")
    if release_timing == "AMC":
        signal_summary.append("AMC timing")

    return {
        "symbol": symbol,
        "earnings_date": event_date,
        "release_timing": release_timing,
        "earnings_source_primary": event_provenance["earnings_source_primary"],
        "earnings_source_confirmed": event_provenance["earnings_source_confirmed"],
        "earnings_source_confidence": event_provenance["earnings_source_confidence"],
        "earnings_source_stale": event_provenance.get("earnings_source_stale", False),
        "entry_date": entry_date,
        "entry_label": "T-3",
        "selected_expiry": selected_expiry,
        "alternative_expiry": alternative_expiry,
        "expiry_mode": expiry_mode,
        "avg_spread_pct": round(avg_spread_pct, 2) if avg_spread_pct is not None else None,
        "previous_avg_spread_pct": round(previous_spread_pct, 2) if previous_spread_pct is not None else None,
        "spread_change_pct": round(spread_change_pct, 2) if spread_change_pct is not None else None,
        "spread_change_state": spread_change_state,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "implied_move_pct": round(implied_move_pct, 2) if implied_move_pct is not None else None,
        "call_strike": round(float(call_row.get("strike_numeric")), 2),
        "put_strike": round(float(put_row.get("strike_numeric")), 2),
        "call_iv": round(float(call_row.get("impliedVolatility")), 4) if _safe_float(call_row.get("impliedVolatility")) is not None else None,
        "put_iv": round(float(put_row.get("impliedVolatility")), 4) if _safe_float(put_row.get("impliedVolatility")) is not None else None,
        "entry_debit_mid": round(entry_debit_mid, 2) if entry_debit_mid is not None else None,
        "status": status,
        "status_reason": status_reason,
        "compact_signal_summary": signal_summary,
        "caveats": caveats,
        "checks": checks,
        "notes": [
            f"Selected expiry uses {expiry_mode.replace('_', ' ')} methodology.",
            f"Alternative expiry: {alternative_expiry}" if alternative_expiry else "No alternate expiry available.",
        ],
        "last_updated": observed_at,
        "detail_metrics": {
            "spot": round(spot, 2),
            "days_to_earnings": (event_date - as_of_date).days,
            "days_to_entry": (entry_date - as_of_date).days,
            "call_spread_pct": round(call_spread_pct, 2) if call_spread_pct is not None else None,
            "put_spread_pct": round(put_spread_pct, 2) if put_spread_pct is not None else None,
            "front_expiry": front_expiry,
            "monthly_expiry": monthly_expiry,
            "call_contract_id": str(call_row.get("optionSymbol")) if call_row.get("optionSymbol") is not None else None,
            "put_contract_id": str(put_row.get("optionSymbol")) if put_row.get("optionSymbol") is not None else None,
            "call_bid": _safe_float(call_row.get("bid")),
            "call_ask": _safe_float(call_row.get("ask")),
            "call_mid": call_mid,
            "put_bid": _safe_float(put_row.get("bid")),
            "put_ask": _safe_float(put_row.get("ask")),
            "put_mid": put_mid,
            **event_provenance,
        },
    }


def build_edge_screener(
    *,
    expiry_mode: ExpiryMode = "front_after_earnings",
    weeks: int = 6,
    today: Optional[date] = None,
    symbols: Optional[Iterable[str]] = None,
    mda_client: Optional[MarketDataClient] = None,
) -> dict[str, Any]:
    as_of_date = today or date.today()
    cutoff = as_of_date + timedelta(weeks=weeks)
    universe = [s.upper() for s in (symbols or DEFAULT_UNIVERSE)]
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_screen_one_symbol, symbol, expiry_mode, as_of_date, cutoff, mda_client): symbol
            for symbol in universe
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)

    rows.sort(
        key=lambda row: (
            {"QUALIFIED": 0, "MARGINAL": 1, "EXCLUDED": 2}.get(str(row["status"]), 9),
            row["avg_spread_pct"] if row["avg_spread_pct"] is not None else 999.0,
            str(row["earnings_date"]),
            row["symbol"],
        )
    )

    return {
        "generated_at": datetime.now(timezone.utc),
        "expiry_mode": expiry_mode,
        "as_of_date": as_of_date,
        "universe_size": len(universe),
        "qualified_count": sum(1 for row in rows if row["status"] == "QUALIFIED"),
        "marginal_count": sum(1 for row in rows if row["status"] == "MARGINAL"),
        "excluded_count": sum(1 for row in rows if row["status"] == "EXCLUDED"),
        "rows": rows,
    }
