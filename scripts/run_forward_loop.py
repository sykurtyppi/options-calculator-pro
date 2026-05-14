#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yfinance as yf

try:
    from dotenv import load_dotenv

    # Keep the forward loop consistent with the rest of the research/runtime
    # scripts so API tokens and feature flags resolve deterministically.
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from services.learning_diagnostics import build_learning_diagnostics
from services.baseline_evidence_store import (
    BASELINE_STRUCTURES,
    BaselineEvidenceStore,
    get_baseline_evidence_store,
)
from services.evidence_quality import evaluate_evidence_quality
from services.execution_scenarios import (
    build_execution_scenarios,
    compare_execution_scenarios,
)
from services import external_io_gate
from services.market_data_client import MarketDataClient
from services.option_surface_quality import diagnose_option_surface_quality
from services.outcome_recorder import (
    OutcomeStore,
    finalize_trade_and_update_learning,
    make_snapshot_hash,
    make_trade_id,
)
from services.provider_telemetry import classify_error, record_provider_telemetry
from services.recommendation_ledger import (
    RecommendationLedger,
    make_recommendation_id,
    record_recommendation,
)
from services.screener_service import (
    DEFAULT_UNIVERSE,
    DTE_MAX_DEFAULT,
    DTE_MIN_DEFAULT,
    build_ranked_screener,
)
from web.api.edge_engine import analyze_single_ticker
from web.api.screener_engine import build_edge_screener

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = (
    Path.home() / ".options_calculator_pro" / "logs" / "learning_log.jsonl"
)
ALLOWED_RECOMMENDATIONS = {"Best Candidate", "Candidate"}
FORWARD_DISCOVERY_WEEKS = 6
OTM_STRANGLE_FALLBACK_DISTANCE_PCT = 0.025
OTM_STRANGLE_MIN_FALLBACK_DISTANCE = 2.5


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value)[:19]).date()
    except Exception:
        return None


def _append_learning_log(log_path: Path, payload: Dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _record_skip(summary: Dict[str, Any], reason: str) -> None:
    summary["skipped"] += 1
    skip_reasons = summary.setdefault("skip_reasons", {})
    skip_reasons[reason] = int(skip_reasons.get(reason, 0)) + 1


def _analyze_for_forward_loop(
    analyzer: Callable[..., Any],
    symbol: str,
    *,
    mda_client: Any,
    dry_run: bool,
) -> Any:
    """Call analyzers with ledger suppression when they support it.

    ``analyze_single_ticker`` records to the recommendation ledger by default
    for API calls. The forward loop owns its own ledger/write policy, so dry-runs
    must suppress analyzer-side writes and normal runs should avoid duplicate IDs.
    """
    try:
        params = inspect.signature(analyzer).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: Dict[str, Any] = {"mda_client": mda_client}
    if "record_to_ledger" in params:
        kwargs["record_to_ledger"] = not dry_run
    return analyzer(symbol, **kwargs)


def _fetch_quote_for_forward_loop(
    price_fetcher: Callable[..., Dict[str, Any]],
    *,
    symbol: str,
    structure: str,
    earnings_date: date,
    as_of_date: date,
    context: Optional[Dict[str, Any]],
    mda_client: Any,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "symbol": symbol,
        "structure": structure,
        "earnings_date": earnings_date,
        "as_of_date": as_of_date,
        "context": context,
    }
    try:
        params = inspect.signature(price_fetcher).parameters
    except (TypeError, ValueError):
        params = {}
    if "mda_client" in params:
        kwargs["mda_client"] = mda_client
    return price_fetcher(**kwargs)


def _get_marketdata_client(candidate: Any = None) -> Optional[MarketDataClient]:
    if candidate is not None and hasattr(candidate, "is_available"):
        return candidate if bool(candidate.is_available()) else None
    if not external_io_gate.is_allowed(external_io_gate.Category.MARKETDATA):
        return None
    try:
        resolved = MarketDataClient()
    except Exception:
        logger.debug("forward_loop: failed to construct MarketDataClient", exc_info=True)
        return None
    return resolved if resolved.is_available() else None


def _normalize_discovery_row(row: Dict[str, Any], *, today: date, source: str) -> Optional[Dict[str, Any]]:
    symbol = str(row.get("symbol") or "").upper()
    if not symbol:
        return None

    earnings_date = _parse_date(row.get("earnings_date"))
    detail_metrics = row.get("detail_metrics") or {}
    dte = row.get("days_to_earnings", row.get("dte"))
    if dte is None:
        dte = detail_metrics.get("days_to_earnings")
    if dte is None and earnings_date is not None:
        dte = (earnings_date - today).days
    try:
        dte_val = int(dte) if dte is not None else None
    except (TypeError, ValueError):
        dte_val = None

    return {
        "symbol": symbol,
        "status": str(row.get("status") or "discovered"),
        "days_to_earnings": dte_val,
        "dte": dte_val,
        "earnings_date": earnings_date.isoformat() if earnings_date is not None else row.get("earnings_date"),
        "release_timing": row.get("release_timing"),
        "discovery_source": source,
        "discovery_status_reason": row.get("status_reason"),
    }


def _build_forward_discovery_payload(
    *,
    today: date,
    symbols: Optional[list[str]],
    screener_builder: Callable[..., Dict[str, Any]],
    mda_client: Any,
) -> Dict[str, Any]:
    universe = symbols or DEFAULT_UNIVERSE

    # Default forward-loop discovery prefers the edge screener because it uses
    # the more reliable ticker.calendar/info earnings path plus MarketData-aware
    # chain/expiration inspection. Recommendation truth still comes exclusively
    # from analyze_single_ticker() and the selector stack below.
    if screener_builder is build_ranked_screener:
        payload = build_edge_screener(
            today=today,
            weeks=FORWARD_DISCOVERY_WEEKS,
            symbols=universe,
            mda_client=_get_marketdata_client(mda_client),
        )
        normalized_rows = [
            normalized
            for normalized in (
                _normalize_discovery_row(row, today=today, source="edge_screener")
                for row in payload.get("rows", [])
            )
            if normalized is not None
        ]
        if normalized_rows:
            return {
                "rows": normalized_rows,
                "source": "edge_screener",
                "summary": {
                    "qualified_count": payload.get("qualified_count", 0),
                    "marginal_count": payload.get("marginal_count", 0),
                    "excluded_count": payload.get("excluded_count", 0),
                },
            }

    payload = screener_builder(symbols=universe, today=today)
    normalized_rows = [
        normalized
        for normalized in (
            _normalize_discovery_row(row, today=today, source="ranked_screener")
            for row in payload.get("rows", [])
        )
        if normalized is not None
    ]
    return {
        "rows": normalized_rows,
        "source": "ranked_screener",
        "summary": {},
    }


def _structure_comparison_summary(snapshot: Any) -> list[Dict[str, Any]]:
    cards = _get(snapshot, "structure_scorecards", None) or []
    ranked_cards = sorted(
        list(cards),
        key=lambda card: (
            float(_get(card, "composite_structure_score", 0.0) or 0.0),
            float(_get(card, "expected_edge_pct", 0.0) or 0.0),
        ),
        reverse=True,
    )
    summary: list[Dict[str, Any]] = []
    for card in ranked_cards:
        summary.append(
            {
                "structure": _get(card, "structure"),
                "eligible": bool(_get(card, "eligible", False)),
                "composite_structure_score": round(float(_get(card, "composite_structure_score", 0.0) or 0.0), 4),
                "expected_edge_pct": round(float(_get(card, "expected_edge_pct", 0.0) or 0.0), 4),
                "execution_penalty": round(float(_get(card, "execution_penalty", 0.0) or 0.0), 4),
                "sample_confidence": round(float(_get(card, "sample_confidence", 0.0) or 0.0), 4),
                "walk_forward_history_count": int(_get(card, "walk_forward_history_count", 0) or 0),
            }
        )
    return summary


def _latest_spot_price(symbol: str) -> Optional[float]:
    start = time.perf_counter()
    try:
        hist = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
        record_provider_telemetry(
            provider_name="yfinance",
            endpoint_type="forward_loop_price_quote",
            symbol=symbol,
            success=hist is not None and not hist.empty,
            error_category=None if hist is not None and not hist.empty else "empty_response",
            latency_ms=(time.perf_counter() - start) * 1000.0,
            response_quality_note="paper/research quote, not execution-grade",
        )
    except Exception as exc:
        record_provider_telemetry(
            provider_name="yfinance",
            endpoint_type="forward_loop_price_quote",
            symbol=symbol,
            success=False,
            error_category=classify_error(str(exc)),
            latency_ms=(time.perf_counter() - start) * 1000.0,
            response_quality_note="paper/research quote, not execution-grade",
        )
        raise
    if hist is None or hist.empty:
        return None
    close = pd.to_numeric(hist.get("Close"), errors="coerce").dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def _mid_from_row(row: Optional[pd.Series]) -> Optional[float]:
    if row is None:
        return None
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    if bid is None or ask is None or bid < 0 or ask < 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return float(mid) if mid > 0 else None


def _quote_fields_from_row(
    row: Optional[pd.Series],
    *,
    quote_quality_label: Optional[str] = None,
    requested_strike: Optional[float] = None,
    fallback_distance: Optional[float] = None,
) -> Dict[str, Any]:
    if row is None:
        return {}
    mid = _mid_from_row(row)
    payload = {
        "contract": str(row.get("contractSymbol")) if row.get("contractSymbol") is not None else None,
        "strike": _safe_float(row.get("strike")),
        "bid": _safe_float(row.get("bid")),
        "ask": _safe_float(row.get("ask")),
        "mid": mid,
        "last": _safe_float(row.get("lastPrice")),
        "volume": _safe_float(row.get("volume")),
        "open_interest": _safe_float(row.get("openInterest", row.get("open_interest"))),
        "implied_volatility": _safe_float(row.get("impliedVolatility", row.get("iv"))),
    }
    if quote_quality_label is not None:
        payload["quote_quality_label"] = quote_quality_label
    if requested_strike is not None:
        payload["requested_strike"] = round(float(requested_strike), 4)
    if fallback_distance is not None:
        payload["fallback_distance"] = round(float(fallback_distance), 4)
    return payload


def _quote_payload(
    *,
    source: str = "yfinance",
    quality: str = "paper_research_mid_not_execution_grade",
    source_quality_note: Optional[str] = None,
    legs: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    surface_quality: Optional[Dict[str, Any]] = None,
    final_reason: Optional[str] = None,
) -> Dict[str, Any]:
    note = source_quality_note or (
        f"Paper/research quote from {source} option-chain fields. "
        "Not execution-grade and not a live broker fill."
    )
    return {
        "quote_source": source,
        "quote_timestamp": datetime.now(timezone.utc).isoformat(),
        "quote_quality": quality,
        "surface_quality": surface_quality or {},
        "bid_ask_mid": {
            "source_quality_note": note,
            "legs": legs or {},
            "provenance": provenance or {},
            "surface_quality": surface_quality or {},
            "final_reason": final_reason,
        },
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if pd.notna(parsed) else None
    except Exception:
        return None


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _quote_invalid_reason(row: Optional[pd.Series], *, leg: str) -> str:
    if row is None:
        return f"missing_{leg}_mid"
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    if bid is None or ask is None or bid < 0 or ask < 0 or ask < bid:
        return "bad_bid_ask"
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return f"missing_{leg}_mid"
    return "partial_or_missing_mid"


def _max_strangle_fallback_distance(spot: float) -> float:
    return max(OTM_STRANGLE_MIN_FALLBACK_DISTANCE, float(spot) * OTM_STRANGLE_FALLBACK_DISTANCE_PCT)


def _select_otm_wing(
    frame: pd.DataFrame,
    *,
    target_strike: float,
    spot: float,
    leg: str,
) -> Dict[str, Any]:
    """Select the requested OTM wing, with a bounded nearest-valid fallback."""
    empty_reason = "empty_call_chain" if leg == "call" else "empty_put_chain"
    missing_mid_reason = "missing_call_mid" if leg == "call" else "missing_put_mid"

    if frame is None or frame.empty:
        return {"row": None, "reason": empty_reason, "quote_quality_label": "partial_or_missing_mid"}

    work = frame.copy()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work = work.dropna(subset=["strike"])
    work = work[work["strike"] >= spot] if leg == "call" else work[work["strike"] <= spot]
    if work.empty:
        return {"row": None, "reason": empty_reason, "quote_quality_label": "partial_or_missing_mid"}

    work["fallback_distance"] = (work["strike"] - float(target_strike)).abs()
    work = work.sort_values(["fallback_distance", "strike"], ascending=[True, leg != "put"])
    max_distance = _max_strangle_fallback_distance(spot)
    primary = work.iloc[0]
    if float(primary["fallback_distance"]) > max_distance:
        return {
            "row": None,
            "reason": "wing_strike_unavailable",
            "quote_quality_label": "partial_or_missing_mid",
            "max_fallback_distance": round(max_distance, 4),
        }

    primary_mid = _mid_from_row(primary)
    if primary_mid is not None:
        return {
            "row": primary,
            "reason": None,
            "quote_quality_label": "exact_wing_mid",
            "fallback_distance": float(primary["fallback_distance"]),
            "requested_strike": float(target_strike),
            "max_fallback_distance": round(max_distance, 4),
        }

    bounded = work[work["fallback_distance"] <= max_distance]
    invalid_reasons = [_quote_invalid_reason(row, leg=leg) for _, row in bounded.iterrows()]
    for _, candidate in bounded.iterrows():
        candidate_mid = _mid_from_row(candidate)
        if candidate_mid is None:
            continue
        return {
            "row": candidate,
            "reason": None,
            "quote_quality_label": "nearest_valid_wing_mid",
            "fallback_distance": float(candidate["fallback_distance"]),
            "requested_strike": float(target_strike),
            "max_fallback_distance": round(max_distance, 4),
        }

    if not len(bounded):
        reason = "wing_strike_unavailable"
    elif "bad_bid_ask" in invalid_reasons:
        reason = "bad_bid_ask"
    else:
        reason = missing_mid_reason
    return {
        "row": None,
        "reason": reason,
        "quote_quality_label": "partial_or_missing_mid",
        "requested_strike": float(target_strike),
        "max_fallback_distance": round(max_distance, 4),
    }


def _normalize_forward_chain_frame(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    if "contractSymbol" not in normalized.columns and "optionSymbol" in normalized.columns:
        normalized["contractSymbol"] = normalized["optionSymbol"]
    if "lastPrice" not in normalized.columns and "last" in normalized.columns:
        normalized["lastPrice"] = normalized["last"]
    normalized["strike"] = pd.to_numeric(normalized.get("strike"), errors="coerce")
    return normalized


def _surface_quality_for_chain(
    *,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: Optional[float],
    expiration: Optional[str],
) -> Dict[str, Any]:
    frames: list[pd.DataFrame] = []
    for frame, side in ((calls, "call"), (puts, "put")):
        if frame is None or frame.empty:
            continue
        work = _normalize_forward_chain_frame(frame)
        if "side" not in work.columns:
            work["side"] = side
        if expiration and "expiration" not in work.columns and "expiration_date" not in work.columns:
            work["expiration"] = expiration
        frames.append(work)
    chain = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return diagnose_option_surface_quality(
        chain,
        underlying_price=spot,
    ).to_dict()


def _merge_surface_quality(*items: Dict[str, Any]) -> Dict[str, Any]:
    clean = [item for item in items if item]
    if not clean:
        return {}
    status_rank = {"clean_surface": 0, "degraded_surface": 1, "record_only": 2}
    status = max((str(item.get("status") or "clean_surface") for item in clean), key=lambda s: status_rank.get(s, 0))
    flags: list[str] = []
    for item in clean:
        flags.extend(str(flag) for flag in (item.get("warning_flags") or []))
    return {
        "status": status,
        "warning_flags": sorted(set(flags)),
        "row_count": sum(int(item.get("row_count") or 0) for item in clean),
        "expiration_count": sum(int(item.get("expiration_count") or 0) for item in clean),
        "crossed_quote_count": sum(int(item.get("crossed_quote_count") or 0) for item in clean),
        "zero_bid_count": sum(int(item.get("zero_bid_count") or 0) for item in clean),
        "extreme_spread_count": sum(int(item.get("extreme_spread_count") or 0) for item in clean),
        "missing_iv_count": sum(int(item.get("missing_iv_count") or 0) for item in clean),
        "iv_outlier_count": sum(int(item.get("iv_outlier_count") or 0) for item in clean),
        "sparse_atm_expiration_count": sum(int(item.get("sparse_atm_expiration_count") or 0) for item in clean),
        "term_structure_anomaly_count": sum(int(item.get("term_structure_anomaly_count") or 0) for item in clean),
        "put_call_parity_warning_count": sum(int(item.get("put_call_parity_warning_count") or 0) for item in clean),
    }


def _split_marketdata_chain(chain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = _normalize_forward_chain_frame(chain)
    if normalized.empty or "side" not in normalized.columns:
        return pd.DataFrame(), pd.DataFrame()
    side = normalized["side"].astype(str).str.lower()
    calls = normalized[side.isin({"call", "calls", "c"})].copy()
    puts = normalized[side.isin({"put", "puts", "p"})].copy()
    return calls, puts


def _spot_from_marketdata_chain(chain: pd.DataFrame) -> Optional[float]:
    if chain is None or chain.empty or "underlyingPrice" not in chain.columns:
        return None
    values = pd.to_numeric(chain["underlyingPrice"], errors="coerce").dropna()
    if values.empty:
        return None
    value = float(values.iloc[0])
    return value if value > 0 else None


def _nearest_row(frame: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    work = frame.copy()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work = work.dropna(subset=["strike"])
    if work.empty:
        return None
    work["dist"] = (work["strike"] - float(strike)).abs()
    return work.sort_values("dist").iloc[0]


def _expiry_after(options: list[str], target: date) -> Optional[str]:
    parsed = sorted(
        [
            datetime.strptime(expiry, "%Y-%m-%d").date()
            for expiry in options
            if expiry
        ]
    )
    for expiry in parsed:
        if expiry >= target:
            return expiry.isoformat()
    return None


def _expiry_after_gap(options: list[str], start: date, gap_days: int) -> Optional[str]:
    return _expiry_after(options, start + timedelta(days=gap_days))


def fetch_structure_quote(
    *,
    symbol: str,
    structure: str,
    earnings_date: date,
    as_of_date: date,
    context: Optional[Dict[str, Any]] = None,
    mda_client: Any = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    resolved_mda_client = _get_marketdata_client(mda_client) if mda_client is not None else None
    quote_source = "yfinance"
    quote_quality = "paper_research_mid_not_execution_grade"

    def _record_quote(success: bool, reason: Optional[str] = None) -> None:
        record_provider_telemetry(
            provider_name=quote_source,
            endpoint_type="forward_loop_option_quote",
            symbol=symbol,
            success=success,
            error_category=None if success else classify_error(reason or "empty_response"),
            latency_ms=(time.perf_counter() - start) * 1000.0,
            fallback_used=quote_source == "yfinance",
            response_quality_note=reason or "paper/research option-chain mid, not execution-grade",
        )

    def _record_marketdata_fallback(reason: str) -> None:
        record_provider_telemetry(
            provider_name="marketdata_app",
            endpoint_type="forward_loop_option_quote",
            symbol=symbol,
            success=False,
            error_category=classify_error(reason),
            latency_ms=(time.perf_counter() - start) * 1000.0,
            fallback_used=True,
            response_quality_note=reason,
        )

    def _payload(**kwargs: Any) -> Dict[str, Any]:
        return _quote_payload(
            source=quote_source,
            quality=quote_quality,
            source_quality_note=(
                f"Paper/research quote from {quote_source} option-chain fields. "
                "Not execution-grade and not a live broker fill."
            ),
            **kwargs,
        )

    ticker: Optional[yf.Ticker] = None
    options: list[str] = []
    if resolved_mda_client is not None:
        options = list(resolved_mda_client.get_expirations(symbol) or [])
    if not options:
        ticker = yf.Ticker(symbol)
        options = list(getattr(ticker, "options", []) or [])
    if not options:
        _record_quote(False, "no_option_expiries")
        return {"mid": None, "reason": "no_option_expiries", **_payload()}

    spot = resolved_mda_client.get_quote(symbol) if resolved_mda_client is not None else None
    if spot is None or spot <= 0:
        spot = _latest_spot_price(symbol)
    if spot is None or spot <= 0:
        _record_quote(False, "no_spot_price")
        return {"mid": None, "reason": "no_spot_price", **_payload()}

    pricing_context = dict(context or {})
    if not pricing_context:
        front_expiry = _expiry_after(options, earnings_date)
        if front_expiry is None:
            _record_quote(False, "no_front_expiry_after_earnings")
            return {"mid": None, "reason": "no_front_expiry_after_earnings", **_payload()}
        pricing_context["front_expiry"] = front_expiry
        if structure in {"call_calendar", "put_calendar"}:
            back_expiry = _expiry_after_gap(options, datetime.strptime(front_expiry, "%Y-%m-%d").date(), 14)
            if back_expiry is None:
                _record_quote(False, "no_back_expiry_for_calendar")
                return {"mid": None, "reason": "no_back_expiry_for_calendar", **_payload()}
            pricing_context["back_expiry"] = back_expiry

    front_calls = pd.DataFrame()
    front_puts = pd.DataFrame()
    if resolved_mda_client is not None:
        mda_chain = resolved_mda_client.get_option_chain(
            symbol,
            expiration=str(pricing_context["front_expiry"]),
            strike_limit=80,
        )
        mda_calls, mda_puts = _split_marketdata_chain(mda_chain)
        if not mda_calls.empty and not mda_puts.empty:
            quote_source = "marketdata_app"
            quote_quality = "marketdata_app_paper_research_mid_not_execution_grade"
            front_calls, front_puts = mda_calls, mda_puts
            chain_spot = _spot_from_marketdata_chain(mda_chain)
            if chain_spot is not None:
                spot = chain_spot
        else:
            _record_marketdata_fallback("marketdata_forward_chain_unusable")

    if front_calls.empty or front_puts.empty:
        quote_source = "yfinance"
        quote_quality = "paper_research_mid_not_execution_grade"
        ticker = ticker or yf.Ticker(symbol)
        try:
            front_chain = ticker.option_chain(pricing_context["front_expiry"])
        except Exception as exc:
            _record_quote(False, str(exc))
            raise
        front_calls = _normalize_forward_chain_frame(front_chain.calls)
        front_puts = _normalize_forward_chain_frame(front_chain.puts)

    front_surface_quality = _surface_quality_for_chain(
        calls=front_calls,
        puts=front_puts,
        spot=spot,
        expiration=str(pricing_context.get("front_expiry") or ""),
    )

    if structure == "atm_straddle":
        strike = float(pricing_context.get("strike") or spot)
        call_row = _nearest_row(front_calls, strike)
        put_row = _nearest_row(front_puts, strike)
        if call_row is None or put_row is None:
            _record_quote(False, "missing_atm_pair")
            return {"mid": None, "reason": "missing_atm_pair", **_payload(surface_quality=front_surface_quality)}
        pricing_context.update(
            {
                "strike": float(call_row["strike"]),
                "call_contract": str(call_row.get("contractSymbol")),
                "put_contract": str(put_row.get("contractSymbol")),
            }
        )
        mid = _mid_from_row(call_row)
        put_mid = _mid_from_row(put_row)
        if mid is None or put_mid is None:
            _record_quote(False, "missing_atm_mid")
            return {"mid": None, "reason": "missing_atm_mid", **_payload(surface_quality=front_surface_quality)}
        legs = {"call": _quote_fields_from_row(call_row), "put": _quote_fields_from_row(put_row)}
        _record_quote(True)
        return {"mid": float(mid + put_mid), "spot": spot, "context": pricing_context, **_payload(legs=legs, surface_quality=front_surface_quality)}

    if structure == "otm_strangle":
        call_target = float(pricing_context.get("call_strike") or (spot * 1.03))
        put_target = float(pricing_context.get("put_strike") or (spot * 0.97))
        call_selection = _select_otm_wing(
            front_calls,
            target_strike=call_target,
            spot=spot,
            leg="call",
        )
        put_selection = _select_otm_wing(
            front_puts,
            target_strike=put_target,
            spot=spot,
            leg="put",
        )
        provenance = {
            "requested_call_wing_strike": round(call_target, 4),
            "selected_call_wing_strike": _safe_float(call_selection.get("row", {}).get("strike")) if call_selection.get("row") is not None else None,
            "requested_put_wing_strike": round(put_target, 4),
            "selected_put_wing_strike": _safe_float(put_selection.get("row", {}).get("strike")) if put_selection.get("row") is not None else None,
            "call_quote_quality_label": call_selection.get("quote_quality_label"),
            "put_quote_quality_label": put_selection.get("quote_quality_label"),
            "call_fallback_distance": round(float(call_selection.get("fallback_distance", 0.0) or 0.0), 4)
            if call_selection.get("row") is not None else None,
            "put_fallback_distance": round(float(put_selection.get("fallback_distance", 0.0) or 0.0), 4)
            if put_selection.get("row") is not None else None,
            "max_fallback_distance": call_selection.get("max_fallback_distance") or put_selection.get("max_fallback_distance"),
        }
        if call_selection.get("row") is None:
            reason = str(call_selection.get("reason") or "missing_call_mid")
            provenance["final_reason"] = reason
            _record_quote(False, reason)
            return {"mid": None, "reason": reason, **_payload(provenance=provenance, surface_quality=front_surface_quality, final_reason=reason)}
        if put_selection.get("row") is None:
            reason = str(put_selection.get("reason") or "missing_put_mid")
            provenance["final_reason"] = reason
            _record_quote(False, reason)
            return {"mid": None, "reason": reason, **_payload(provenance=provenance, surface_quality=front_surface_quality, final_reason=reason)}
        call_row = call_selection["row"]
        put_row = put_selection["row"]
        pricing_context.update(
            {
                "requested_call_wing_strike": call_target,
                "selected_call_wing_strike": float(call_row["strike"]),
                "requested_put_wing_strike": put_target,
                "selected_put_wing_strike": float(put_row["strike"]),
                "call_strike": float(call_row["strike"]),
                "put_strike": float(put_row["strike"]),
                "call_contract": str(call_row.get("contractSymbol")),
                "put_contract": str(put_row.get("contractSymbol")),
                "call_quote_quality_label": call_selection.get("quote_quality_label"),
                "put_quote_quality_label": put_selection.get("quote_quality_label"),
                "call_fallback_distance": provenance["call_fallback_distance"],
                "put_fallback_distance": provenance["put_fallback_distance"],
                "max_fallback_distance": provenance["max_fallback_distance"],
            }
        )
        call_mid = _mid_from_row(call_row)
        put_mid = _mid_from_row(put_row)
        if call_mid is None:
            _record_quote(False, "missing_call_mid")
            return {"mid": None, "reason": "missing_call_mid", **_payload(provenance=provenance, surface_quality=front_surface_quality, final_reason="missing_call_mid")}
        if put_mid is None:
            _record_quote(False, "missing_put_mid")
            return {"mid": None, "reason": "missing_put_mid", **_payload(provenance=provenance, surface_quality=front_surface_quality, final_reason="missing_put_mid")}
        legs = {
            "call": _quote_fields_from_row(
                call_row,
                quote_quality_label=str(call_selection.get("quote_quality_label")),
                requested_strike=call_target,
                fallback_distance=float(call_selection.get("fallback_distance", 0.0) or 0.0),
            ),
            "put": _quote_fields_from_row(
                put_row,
                quote_quality_label=str(put_selection.get("quote_quality_label")),
                requested_strike=put_target,
                fallback_distance=float(put_selection.get("fallback_distance", 0.0) or 0.0),
            ),
        }
        provenance.update(
            {
                "call_bid": legs["call"].get("bid"),
                "call_ask": legs["call"].get("ask"),
                "call_mid": legs["call"].get("mid"),
                "put_bid": legs["put"].get("bid"),
                "put_ask": legs["put"].get("ask"),
                "put_mid": legs["put"].get("mid"),
                "final_reason": None,
            }
        )
        _record_quote(True)
        return {
            "mid": float(call_mid + put_mid),
            "spot": spot,
            "context": pricing_context,
            **_payload(legs=legs, provenance=provenance, surface_quality=front_surface_quality),
        }

    if structure in {"call_calendar", "put_calendar"}:
        front_frame = front_calls if structure == "call_calendar" else front_puts
        if quote_source == "marketdata_app" and resolved_mda_client is not None:
            mda_back_chain = resolved_mda_client.get_option_chain(
                symbol,
                expiration=str(pricing_context["back_expiry"]),
                strike_limit=80,
            )
            back_calls, back_puts = _split_marketdata_chain(mda_back_chain)
            back_frame = back_calls if structure == "call_calendar" else back_puts
            back_surface_quality = _surface_quality_for_chain(
                calls=back_calls,
                puts=back_puts,
                spot=spot,
                expiration=str(pricing_context.get("back_expiry") or ""),
            )
        else:
            ticker = ticker or yf.Ticker(symbol)
            try:
                back_chain = ticker.option_chain(pricing_context["back_expiry"])
            except Exception as exc:
                _record_quote(False, str(exc))
                raise
            back_frame = _normalize_forward_chain_frame(
                back_chain.calls if structure == "call_calendar" else back_chain.puts
            )
            if structure == "call_calendar":
                back_surface_quality = _surface_quality_for_chain(
                    calls=back_frame,
                    puts=pd.DataFrame(),
                    spot=spot,
                    expiration=str(pricing_context.get("back_expiry") or ""),
                )
            else:
                back_surface_quality = _surface_quality_for_chain(
                    calls=pd.DataFrame(),
                    puts=back_frame,
                    spot=spot,
                    expiration=str(pricing_context.get("back_expiry") or ""),
                )
        calendar_surface_quality = _merge_surface_quality(front_surface_quality, back_surface_quality)
        strike = float(pricing_context.get("strike") or spot)
        front_row = _nearest_row(front_frame, strike)
        if front_row is None:
            _record_quote(False, "missing_front_leg")
            return {"mid": None, "reason": "missing_front_leg", **_payload(surface_quality=calendar_surface_quality)}
        chosen_strike = float(front_row["strike"])
        back_row = _nearest_row(back_frame, chosen_strike)
        if back_row is None:
            _record_quote(False, "missing_back_leg")
            return {"mid": None, "reason": "missing_back_leg", **_payload(surface_quality=calendar_surface_quality)}
        front_mid = _mid_from_row(front_row)
        back_mid = _mid_from_row(back_row)
        if front_mid is None or back_mid is None:
            _record_quote(False, "missing_calendar_mid")
            return {"mid": None, "reason": "missing_calendar_mid", **_payload(surface_quality=calendar_surface_quality)}
        pricing_context.update(
            {
                "strike": chosen_strike,
                "front_contract": str(front_row.get("contractSymbol")),
                "back_contract": str(back_row.get("contractSymbol")),
            }
        )
        legs = {"front": _quote_fields_from_row(front_row), "back": _quote_fields_from_row(back_row)}
        _record_quote(True)
        return {"mid": float(back_mid - front_mid), "spot": spot, "context": pricing_context, **_payload(legs=legs, surface_quality=calendar_surface_quality)}

    _record_quote(False, f"unsupported_structure:{structure}")
    return {"mid": None, "reason": f"unsupported_structure:{structure}", **_payload()}


def _find_structure_scorecard(snapshot: Any, structure: str) -> Optional[Dict[str, Any]]:
    cards = _get(snapshot, "structure_scorecards", None) or []
    for card in cards:
        if _get(card, "structure") == structure:
            return card if isinstance(card, dict) else None
    return None


def _record_baseline_entries(
    *,
    baseline_store: BaselineEvidenceStore,
    price_fetcher: Callable[..., Dict[str, Any]],
    symbol: str,
    earnings_date: date,
    as_of: date,
    recommendation_id: str,
    selector_structure: str,
    snapshot: Any,
    vol_snapshot: Dict[str, Any],
    mda_client: Any,
) -> Dict[str, int]:
    summary = {"baseline_entries": 0, "baseline_skipped": 0}
    for baseline_name, structure in BASELINE_STRUCTURES.items():
        card = _find_structure_scorecard(snapshot, structure) or {}
        quote = _fetch_quote_for_forward_loop(
            price_fetcher,
            symbol=symbol,
            structure=structure,
            earnings_date=earnings_date,
            as_of_date=as_of,
            context=None,
            mda_client=mda_client,
        )
        entry_mid = _safe_float(quote.get("mid"))
        execution_penalty = _safe_float(_get(card, "execution_penalty"))
        evidence_quality = evaluate_evidence_quality(
            quote_payload=quote,
            vol_snapshot=vol_snapshot,
        ).to_dict()
        entry_execution_scenarios = build_execution_scenarios(
            structure=structure,
            quote_payload=quote,
            phase="entry",
        ).to_dict()
        inserted = baseline_store.insert_entry(
            recommendation_id=recommendation_id,
            symbol=symbol,
            baseline_name=baseline_name,
            structure=structure,
            entry_date=as_of,
            earnings_date=earnings_date,
            selector_structure=selector_structure,
            entry_mid=entry_mid,
            modeled_cost_pct=(26.0 * execution_penalty) if execution_penalty is not None else None,
            execution_penalty_at_entry=execution_penalty,
            data_quality_score_at_entry=_safe_float(_get(vol_snapshot, "data_quality_score")),
            iv_rv_har_at_entry=_safe_float(_get(vol_snapshot, "iv_rv_har")),
            iv_rv_yz_at_entry=_safe_float(_get(vol_snapshot, "iv_rv_yz")),
            quote_source_at_entry=quote.get("quote_source"),
            quote_quality_at_entry=quote.get("quote_quality"),
            entry_bid_ask_mid=quote.get("bid_ask_mid", {}),
            evidence_quality_status=evidence_quality.get("evidence_quality_status"),
            evidence_quality_reasons=evidence_quality.get("evidence_quality_reasons", []),
            claim_allowed=bool(evidence_quality.get("claim_allowed")),
            execution_grade=bool(evidence_quality.get("execution_grade")),
            entry_execution_scenarios=entry_execution_scenarios,
            surface_quality=quote.get("surface_quality") or {},
            status="open" if entry_mid is not None and entry_mid > 0 else "entry_skipped",
            skip_reason=None if entry_mid is not None and entry_mid > 0 else str(quote.get("reason", "missing_entry_mid")),
            metadata={
                "source": "forward_loop_shadow_baseline",
                "evidence_quality": evidence_quality,
            },
        )
        if inserted and entry_mid is not None and entry_mid > 0:
            summary["baseline_entries"] += 1
        elif inserted:
            summary["baseline_skipped"] += 1
    return summary


def _finalize_baseline_exits(
    *,
    baseline_store: BaselineEvidenceStore,
    price_fetcher: Callable[..., Dict[str, Any]],
    as_of: date,
    log_path: Path,
    dry_run: bool,
    mda_client: Any,
) -> Dict[str, int]:
    summary = {"baseline_exits": 0, "baseline_skipped": 0}
    for row in baseline_store.baselines_due_for_exit(as_of):
        entry_mid = _safe_float(row.get("entry_mid"))
        earnings_date = _parse_date(row.get("earnings_date"))
        if entry_mid is None or entry_mid <= 0 or earnings_date is None:
            summary["baseline_skipped"] += 1
            continue
        quote = _fetch_quote_for_forward_loop(
            price_fetcher,
            symbol=str(row.get("symbol")),
            structure=str(row.get("structure")),
            earnings_date=earnings_date,
            as_of_date=as_of,
            context=None,
            mda_client=mda_client,
        )
        exit_execution_scenarios = build_execution_scenarios(
            structure=str(row.get("structure")),
            quote_payload=quote,
            phase="exit",
        ).to_dict()
        exit_mid = _safe_float(quote.get("mid"))
        if exit_mid is None or exit_mid <= 0:
            summary["baseline_skipped"] += 1
            if not dry_run:
                baseline_store.update_exit(
                    baseline_id=str(row.get("baseline_id")),
                    exit_date=as_of,
                    exit_mid=None,
                    realized_return_pct=None,
                    realized_expansion_pct=None,
                    quote_source_at_exit=quote.get("quote_source"),
                    quote_quality_at_exit=quote.get("quote_quality"),
                    exit_bid_ask_mid=quote.get("bid_ask_mid", {}),
                    exit_execution_scenarios=exit_execution_scenarios,
                    status="exit_skipped",
                    skip_reason=str(quote.get("reason", "missing_exit_mid")),
                )
            continue
        expansion_pct = ((exit_mid - entry_mid) / entry_mid) * 100.0
        cost_pct = _safe_float(row.get("modeled_cost_pct")) or 0.0
        realized_return_pct = expansion_pct - cost_pct
        if not dry_run:
            baseline_store.update_exit(
                baseline_id=str(row.get("baseline_id")),
                exit_date=as_of,
                exit_mid=exit_mid,
                realized_return_pct=realized_return_pct,
                realized_expansion_pct=expansion_pct,
                quote_source_at_exit=quote.get("quote_source"),
                quote_quality_at_exit=quote.get("quote_quality"),
                exit_bid_ask_mid=quote.get("bid_ask_mid", {}),
                exit_execution_scenarios={
                    **exit_execution_scenarios,
                    "scenario_outcomes": compare_execution_scenarios(
                        entry=row.get("entry_execution_scenarios_json") or {},
                        exit=exit_execution_scenarios,
                    ),
                },
                status="resolved",
            )
        summary["baseline_exits"] += 1
        _append_learning_log(
            log_path,
            {
                "event_type": "baseline_exit",
                "symbol": row.get("symbol"),
                "structure": row.get("structure"),
                "baseline_name": row.get("baseline_name"),
                "realized_return_pct": round(float(realized_return_pct), 4),
                "source": "paper_baseline",
                "recommendation_id": row.get("recommendation_id"),
                "quote_source": quote.get("quote_source"),
                "quote_quality": quote.get("quote_quality"),
            },
            dry_run=dry_run,
        )
    return summary


def run_forward_screener(
    *,
    today: Optional[date] = None,
    dry_run: bool = False,
    store: Optional[OutcomeStore] = None,
    log_path: Path = DEFAULT_LOG_PATH,
    screener_builder: Callable[..., Dict[str, Any]] = build_ranked_screener,
    analyzer: Callable[..., Any] = analyze_single_ticker,
    price_fetcher: Callable[..., Dict[str, Any]] = fetch_structure_quote,
    ledger: Optional[RecommendationLedger] = None,
    baseline_store: Optional[BaselineEvidenceStore] = None,
    symbols: Optional[list[str]] = None,
    mda_client: Any = None,
) -> Dict[str, Any]:
    as_of = today or date.today()
    trade_store = store or OutcomeStore()
    screener_payload = _build_forward_discovery_payload(
        today=as_of,
        symbols=symbols,
        screener_builder=screener_builder,
        mda_client=mda_client,
    )
    summary = {
        "entries": 0,
        "duplicates": 0,
        "skipped": 0,
        "skip_reasons": {},
        "discovered": len(screener_payload.get("rows", [])),
        "analyzed": 0,
        "ledger_records": 0,
        "ledger_failures": 0,
        "baseline_entries": 0,
        "baseline_skipped": 0,
        "discovery_source": str(screener_payload.get("source") or "unknown"),
    }

    for row in screener_payload.get("rows", []):
        dte = row.get("days_to_earnings", row.get("dte"))
        if dte is None or not (DTE_MIN_DEFAULT <= int(dte) <= DTE_MAX_DEFAULT):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue

        snapshot = _analyze_for_forward_loop(
            analyzer,
            symbol,
            mda_client=mda_client,
            dry_run=dry_run,
        )
        summary["analyzed"] += 1
        selector_output = _get(snapshot, "selector_output", {}) or {}
        recommendation = _get(selector_output, "recommendation", _get(snapshot, "recommendation"))
        structure = _get(selector_output, "best_structure")
        earnings_date = _parse_date(_get(selector_output, "earnings_date")) or _parse_date(_get(_get(snapshot, "vol_snapshot", {}), "earnings_date"))
        structure_comparison = _structure_comparison_summary(snapshot)
        vol_snapshot = _get(snapshot, "vol_snapshot", {}) or {}
        recommendation_id = _get(_get(snapshot, "metrics", {}) or {}, "recommendation_id") or make_recommendation_id(
            symbol=symbol,
            as_of_date=_get(vol_snapshot, "as_of_date") or as_of,
            earnings_date=earnings_date or _get(vol_snapshot, "earnings_date"),
            selected_structure=structure,
        )
        if not dry_run:
            try:
                record_recommendation(
                    snapshot,
                    ledger=ledger,
                    recommendation_id=recommendation_id,
                    metadata={
                        "source": "forward_loop",
                        "discovery_source": row.get("discovery_source"),
                        "discovery_status": row.get("status"),
                    },
                )
                summary["ledger_records"] += 1
            except Exception as exc:
                summary["ledger_failures"] += 1
                logger.warning("forward_loop: recommendation ledger write failed for %s: %s", symbol, exc)
                _append_learning_log(
                    log_path,
                    {
                        "event_type": "skip",
                        "symbol": symbol,
                        "structure": structure,
                        "setup_score": _get(snapshot, "setup_score"),
                        "source": "paper",
                        "reason": "recommendation_ledger_write_failed",
                        "recommendation_id": recommendation_id,
                        "error": str(exc),
                    },
                    dry_run=dry_run,
                )

        if recommendation not in ALLOWED_RECOMMENDATIONS or not structure or earnings_date is None:
            _record_skip(summary, "recommendation_not_actionable")
            _append_learning_log(
                log_path,
                {
                    "event_type": "skip",
                    "symbol": symbol,
                    "structure": structure,
                    "setup_score": _get(snapshot, "setup_score"),
                    "source": "paper",
                    "reason": "recommendation_not_actionable",
                    "recommendation_id": recommendation_id,
                    "discovery_source": row.get("discovery_source"),
                    "discovery_status": row.get("status"),
                    "structure_comparison": structure_comparison,
                },
                dry_run=dry_run,
            )
            continue

        existing = trade_store.find_active_trade_for_event(
            symbol=symbol,
            structure=structure,
            earnings_date=earnings_date,
        )
        if existing is not None:
            summary["duplicates"] += 1
            _record_skip(summary, "duplicate_active_trade")
            _append_learning_log(
                log_path,
                {
                    "event_type": "skip",
                    "symbol": symbol,
                    "structure": structure,
                    "setup_score": _get(snapshot, "setup_score"),
                    "source": "paper",
                    "reason": "duplicate_active_trade",
                    "trade_id": existing.get("trade_id"),
                    "recommendation_id": recommendation_id,
                    "discovery_source": row.get("discovery_source"),
                    "structure_comparison": structure_comparison,
                },
                dry_run=dry_run,
            )
            continue

        quote = _fetch_quote_for_forward_loop(
            price_fetcher,
            symbol=symbol,
            structure=structure,
            earnings_date=earnings_date,
            as_of_date=as_of,
            context=None,
            mda_client=mda_client,
        )
        try:
            if not dry_run:
                record_recommendation(
                    snapshot,
                    ledger=ledger,
                    recommendation_id=recommendation_id,
                    quote_payload=quote,
                    metadata={
                        "source": "forward_loop_entry_quote",
                        "discovery_source": row.get("discovery_source"),
                        "pricing_context": quote.get("context", {}),
                    },
                )
        except Exception as exc:
            summary["ledger_failures"] += 1
            logger.warning("forward_loop: recommendation quote ledger update failed for %s: %s", symbol, exc)
        entry_mid = quote.get("mid")
        evidence_quality = evaluate_evidence_quality(
            quote_payload=quote,
            vol_snapshot=vol_snapshot,
        ).to_dict()
        entry_execution_scenarios = build_execution_scenarios(
            structure=structure,
            quote_payload=quote,
            phase="entry",
        ).to_dict()
        if entry_mid is None or entry_mid <= 0:
            _record_skip(summary, str(quote.get("reason", "missing_entry_mid")))
            _append_learning_log(
                log_path,
                {
                    "event_type": "skip",
                    "symbol": symbol,
                    "structure": structure,
                    "setup_score": _get(snapshot, "setup_score"),
                    "source": "paper",
                    "reason": quote.get("reason", "missing_entry_mid"),
                    "recommendation_id": recommendation_id,
                    "quote_source": quote.get("quote_source"),
                    "quote_quality": quote.get("quote_quality"),
                    "evidence_quality_status": evidence_quality.get("evidence_quality_status"),
                    "evidence_quality_reasons": evidence_quality.get("evidence_quality_reasons", []),
                    "discovery_source": row.get("discovery_source"),
                    "structure_comparison": structure_comparison,
                },
                dry_run=dry_run,
            )
            continue

        structure_card = next(
            (
                card
                for card in (_get(snapshot, "structure_scorecards", []) or [])
                if _get(card, "structure") == structure
            ),
            {},
        )
        notes_payload = {
            "vol_snapshot": vol_snapshot,
            "selector_output": selector_output,
            "structure_comparison": structure_comparison,
            "discovery_context": {
                "source": row.get("discovery_source"),
                "status": row.get("status"),
                "status_reason": row.get("discovery_status_reason"),
                "release_timing": row.get("release_timing"),
            },
            "pricing_context": quote.get("context", {}),
            "entry_price_source": f"{quote.get('quote_source', 'unknown')}_option_chain_mid",
            "entry_quote_source": quote.get("quote_source", "yfinance"),
            "entry_quote_quality": quote.get("quote_quality", "paper_research_mid_not_execution_grade"),
            "entry_bid_ask_mid": quote.get("bid_ask_mid", {}),
            "evidence_quality": evidence_quality,
            "entry_execution_scenarios": entry_execution_scenarios,
            "recommendation_id": recommendation_id,
        }
        trade_id = make_trade_id(symbol, as_of, structure)

        inserted = True
        if not dry_run:
            inserted = trade_store.insert_entry(
                trade_id=trade_id,
                recommendation_id=recommendation_id,
                symbol=symbol,
                structure=structure,
                entry_date=as_of,
                setup_score=float(_get(snapshot, "setup_score", 0.0) or 0.0),
                source_type="paper",
                release_timing=str(_get(vol_snapshot, "release_timing", "")),
                earnings_date=earnings_date,
                as_of_date_at_entry=_parse_date(_get(vol_snapshot, "as_of_date")) or as_of,
                selector_recommendation=str(recommendation),
                selector_confidence_pct=_get(selector_output, "confidence_pct"),
                expected_edge_pct=_get(selector_output, "expected_edge_pct"),
                expected_return_pct=_get(selector_output, "expected_return_pct"),
                best_structure_at_entry=structure,
                runner_up_structure_at_entry=(
                    (_get(selector_output, "runner_up_structures", []) or [None])[0]
                ),
                data_quality_score_at_entry=_get(vol_snapshot, "data_quality_score"),
                days_to_earnings=_get(vol_snapshot, "days_to_earnings"),
                iv_rv_yz=_get(vol_snapshot, "iv_rv_yz"),
                iv_rv_har=_get(vol_snapshot, "iv_rv_har"),
                historical_vs_implied_move_ratio=_get(vol_snapshot, "historical_vs_implied_move_ratio"),
                term_structure_slope=_get(vol_snapshot, "term_structure_slope"),
                near_term_spread_pct=_get(vol_snapshot, "near_term_spread_pct"),
                liquidity_tier=_get(vol_snapshot, "liquidity_tier"),
                calibration_phase_at_entry=_get(_get(snapshot, "metrics", {}), "calibration_phase"),
                entry_mid=float(entry_mid),
                execution_penalty_at_entry=_get(structure_card, "execution_penalty"),
                assumed_cost_model="paper_mid_model",
                evidence_quality_status=evidence_quality.get("evidence_quality_status"),
                evidence_quality_reasons=evidence_quality.get("evidence_quality_reasons", []),
                claim_allowed=bool(evidence_quality.get("claim_allowed")),
                execution_grade=bool(evidence_quality.get("execution_grade")),
                entry_quote_source=quote.get("quote_source"),
                entry_quote_quality=quote.get("quote_quality"),
                entry_quote_timestamp=quote.get("quote_timestamp"),
                entry_bid_ask_mid=quote.get("bid_ask_mid", {}),
                entry_execution_scenarios=entry_execution_scenarios,
                surface_quality=quote.get("surface_quality") or {},
                snapshot_hash=make_snapshot_hash(vol_snapshot),
                notes=json.dumps(notes_payload, default=str),
            )
            if inserted:
                baseline_summary = _record_baseline_entries(
                    baseline_store=baseline_store or get_baseline_evidence_store(),
                    price_fetcher=price_fetcher,
                    symbol=symbol,
                    earnings_date=earnings_date,
                    as_of=as_of,
                    recommendation_id=recommendation_id,
                    selector_structure=structure,
                    snapshot=snapshot,
                    vol_snapshot=vol_snapshot,
                    mda_client=mda_client,
                )
                summary["baseline_entries"] += baseline_summary["baseline_entries"]
                summary["baseline_skipped"] += baseline_summary["baseline_skipped"]
        if inserted:
            summary["entries"] += 1
            _append_learning_log(
                log_path,
                {
                    "event_type": "entry",
                    "symbol": symbol,
                    "structure": structure,
                    "setup_score": _get(snapshot, "setup_score"),
                    "realized_return_pct": None,
                    "source": "paper",
                    "trade_id": trade_id,
                    "recommendation_id": recommendation_id,
                    "quote_source": quote.get("quote_source"),
                    "quote_quality": quote.get("quote_quality"),
                    "evidence_quality_status": evidence_quality.get("evidence_quality_status"),
                    "claim_allowed": evidence_quality.get("claim_allowed"),
                    "discovery_source": row.get("discovery_source"),
                    "structure_comparison": structure_comparison,
                },
                dry_run=dry_run,
            )
        else:
            summary["duplicates"] += 1
            _record_skip(summary, "duplicate_trade_id")
            _append_learning_log(
                log_path,
                {
                    "event_type": "skip",
                    "symbol": symbol,
                    "structure": structure,
                    "setup_score": _get(snapshot, "setup_score"),
                    "source": "paper",
                    "reason": "duplicate_trade_id",
                    "trade_id": trade_id,
                    "recommendation_id": recommendation_id,
                    "discovery_source": row.get("discovery_source"),
                    "structure_comparison": structure_comparison,
                },
                dry_run=dry_run,
            )
    return summary


def run_exit_detection(
    *,
    today: Optional[date] = None,
    dry_run: bool = False,
    store: Optional[OutcomeStore] = None,
    log_path: Path = DEFAULT_LOG_PATH,
    price_fetcher: Callable[..., Dict[str, Any]] = fetch_structure_quote,
    finalizer: Callable[..., Dict[str, Any]] = finalize_trade_and_update_learning,
    baseline_store: Optional[BaselineEvidenceStore] = None,
    mda_client: Any = None,
) -> Dict[str, int]:
    as_of = today or date.today()
    trade_store = store or OutcomeStore()
    summary = {"exits": 0, "skipped": 0, "baseline_exits": 0, "baseline_skipped": 0}

    for row in trade_store.trades_due_for_exit(as_of):
        trade_id = str(row["trade_id"])
        structure = str(row["structure"])
        symbol = str(row["symbol"]).upper()
        entry_mid = float(row["entry_mid"]) if row.get("entry_mid") is not None else None
        earnings_date = _parse_date(row.get("earnings_date"))
        if entry_mid is None or entry_mid <= 0 or earnings_date is None:
            summary["skipped"] += 1
            _append_learning_log(
                log_path,
                {
                    "event_type": "skip",
                    "symbol": symbol,
                    "structure": structure,
                    "setup_score": row.get("setup_score"),
                    "source": "paper",
                    "reason": "missing_entry_context",
                    "trade_id": trade_id,
                    "recommendation_id": row.get("recommendation_id"),
                },
                dry_run=dry_run,
            )
            continue

        note_payload: Dict[str, Any] = {}
        raw_notes = row.get("notes")
        if raw_notes:
            try:
                note_payload = json.loads(str(raw_notes))
            except json.JSONDecodeError:
                note_payload = {}

        quote = _fetch_quote_for_forward_loop(
            price_fetcher,
            symbol=symbol,
            structure=structure,
            earnings_date=earnings_date,
            as_of_date=as_of,
            context=note_payload.get("pricing_context"),
            mda_client=mda_client,
        )
        exit_execution_scenarios = build_execution_scenarios(
            structure=structure,
            quote_payload=quote,
            phase="exit",
        ).to_dict()
        exit_mid = quote.get("mid")
        if exit_mid is None:
            summary["skipped"] += 1
            _append_learning_log(
                log_path,
                {
                    "event_type": "skip",
                    "symbol": symbol,
                    "structure": structure,
                    "setup_score": row.get("setup_score"),
                    "source": "paper",
                    "reason": quote.get("reason", "missing_exit_mid"),
                    "trade_id": trade_id,
                    "recommendation_id": row.get("recommendation_id"),
                    "quote_source": quote.get("quote_source"),
                    "quote_quality": quote.get("quote_quality"),
                },
                dry_run=dry_run,
            )
            continue

        realized_expansion_pct = ((float(exit_mid) - entry_mid) / entry_mid) * 100.0
        execution_penalty = float(row.get("execution_penalty_at_entry") or 0.0)
        modeled_cost_pct = 26.0 * execution_penalty
        realized_return_pct = realized_expansion_pct - modeled_cost_pct
        realized_pnl = (float(exit_mid) - entry_mid) * 100.0
        entry_execution_scenarios = _loads_dict(row.get("entry_execution_scenarios_json"))
        if not entry_execution_scenarios:
            entry_execution_scenarios = note_payload.get("entry_execution_scenarios", {})
        scenario_outcomes = compare_execution_scenarios(
            entry=entry_execution_scenarios,
            exit=exit_execution_scenarios,
        )
        exit_execution_scenarios = {
            **exit_execution_scenarios,
            "scenario_outcomes": scenario_outcomes,
        }

        if not dry_run:
            finalizer(
                trade_id=trade_id,
                exit_date=as_of,
                exit_mid=float(exit_mid),
                realized_return_pct=float(realized_return_pct),
                realized_pnl=float(realized_pnl),
                realized_expansion_pct=float(realized_expansion_pct),
                exit_quote_source=quote.get("quote_source"),
                exit_quote_quality=quote.get("quote_quality"),
                exit_quote_timestamp=quote.get("quote_timestamp"),
                exit_bid_ask_mid=quote.get("bid_ask_mid", {}),
                exit_execution_scenarios=exit_execution_scenarios,
                store=trade_store,
                source_type="paper",
            )
        summary["exits"] += 1
        _append_learning_log(
            log_path,
            {
                "event_type": "exit",
                "symbol": symbol,
                "structure": structure,
                "setup_score": row.get("setup_score"),
                "realized_return_pct": round(float(realized_return_pct), 4),
                "source": "paper",
                "trade_id": trade_id,
                "recommendation_id": row.get("recommendation_id"),
                "quote_source": quote.get("quote_source"),
                "quote_quality": quote.get("quote_quality"),
                "execution_scenario_returns": scenario_outcomes.get("realized_return_pct", {}),
            },
            dry_run=dry_run,
        )
    baseline_summary = _finalize_baseline_exits(
        baseline_store=baseline_store or get_baseline_evidence_store(),
        price_fetcher=price_fetcher,
        as_of=as_of,
        log_path=log_path,
        dry_run=dry_run,
        mda_client=mda_client,
    )
    summary.update(baseline_summary)
    return summary


def run_daily_cycle(
    *,
    today: Optional[date] = None,
    dry_run: bool = False,
    store: Optional[OutcomeStore] = None,
    log_path: Path = DEFAULT_LOG_PATH,
    screener_builder: Callable[..., Dict[str, Any]] = build_ranked_screener,
    analyzer: Callable[..., Any] = analyze_single_ticker,
    price_fetcher: Callable[..., Dict[str, Any]] = fetch_structure_quote,
    finalizer: Callable[..., Dict[str, Any]] = finalize_trade_and_update_learning,
    ledger: Optional[RecommendationLedger] = None,
    baseline_store: Optional[BaselineEvidenceStore] = None,
    symbols: Optional[list[str]] = None,
    mda_client: Any = None,
) -> Dict[str, Any]:
    as_of = today or date.today()
    trade_store = store or OutcomeStore()
    resolved_mda_client = _get_marketdata_client(mda_client)
    entry_summary = run_forward_screener(
        today=as_of,
        dry_run=dry_run,
        store=trade_store,
        log_path=log_path,
        screener_builder=screener_builder,
        analyzer=analyzer,
        price_fetcher=price_fetcher,
        ledger=ledger,
        baseline_store=baseline_store,
        symbols=symbols,
        mda_client=resolved_mda_client,
    )
    exit_summary = run_exit_detection(
        today=as_of,
        dry_run=dry_run,
        store=trade_store,
        log_path=log_path,
        price_fetcher=price_fetcher,
        finalizer=finalizer,
        baseline_store=baseline_store,
        mda_client=resolved_mda_client,
    )
    diagnostics = build_learning_diagnostics()
    _append_learning_log(
        log_path,
        {
            "event_type": "diagnostics",
            "symbol": "SYSTEM",
            "structure": None,
            "setup_score": None,
            "realized_return_pct": None,
            "source": "paper",
            "summary": {
                "entries": entry_summary,
                "exits": exit_summary,
                "learning_health": diagnostics["learning_health"],
            },
        },
        dry_run=dry_run,
    )
    return {
        "as_of_date": as_of.isoformat(),
        "entries": entry_summary,
        "exits": exit_summary,
        "diagnostics": diagnostics,
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the daily forward learning loop.")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing trades, outcomes, or logs.")
    parser.add_argument("--date", type=str, default=None, help="Override as-of date (YYYY-MM-DD).")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated override universe.")
    args = parser.parse_args()

    as_of = _parse_date(args.date) if args.date else date.today()
    symbols = [token.strip().upper() for token in args.symbols.split(",") if token.strip()] if args.symbols else None

    result = run_daily_cycle(
        today=as_of,
        dry_run=args.dry_run,
        symbols=symbols,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
