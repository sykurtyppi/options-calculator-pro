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

from services.move_statistics import MOVE_RATIO_UNITS_VERSION
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
from services.market_data_provider import build_market_data_client, get_options_provider_name
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
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    # PR #73 P3: route through the shared fcntl.LOCK_EX-protected
    # appender. Pre-fix this opened the file in append mode without a
    # writer lock, so two daily forward-loop invocations (or an
    # operator running this script manually while launchd fires)
    # could interleave a partial line. The resolver had this fix
    # under Hardening P1-3; PR #73 lifted the pattern into
    # services.jsonl_helpers for shared use.
    from services.jsonl_helpers import append_jsonl_locked
    append_jsonl_locked(log_path, [record])


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


def _get_marketdata_client(candidate: Any = None) -> Optional[Any]:
    if candidate is not None and hasattr(candidate, "is_available"):
        return candidate if bool(candidate.is_available()) else None
    # Provider-selected (default: yfinance). Gate on the active provider's
    # category so the IO guard stays accurate for whichever feed is live.
    provider = get_options_provider_name()
    category = (
        external_io_gate.Category.YFINANCE
        if provider in {"yfinance", "yahoo", "yf"}
        else external_io_gate.Category.MARKETDATA
    )
    if not external_io_gate.is_allowed(category):
        return None
    try:
        resolved = build_market_data_client(provider=provider)
    except Exception:
        logger.debug("forward_loop: failed to construct market-data client", exc_info=True)
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
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:  # H3: zero bid not executable
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
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:  # H3: zero bid not executable
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
    """Provider boundary for the forward loop's option chains.

    LOAD-BEARING PROVIDER ASSUMPTION — a worthless leg must arrive as a numeric
    ``0.0`` bid, NOT as NaN.

    ``_closing_leg_mid`` distinguishes the two: a valid ``bid == 0.0`` means
    "genuinely worthless", and an exit is still priced from the ask; a NaN bid
    means "unknown" and the leg is refused, because treating unknown as zero
    once converted a -95% loss into a +267% win. That split is only safe while
    providers really do emit 0.0 for worthless legs.

    Both current providers satisfy it: yfinance
    (yfinance_market_data_client.py:207) and MarketData
    (market_data_client.py:563) each run bid through
    ``pd.to_numeric(..., errors="coerce")``, so a real zero stays 0.0 and only an
    absent/unparseable quote becomes NaN. This function deliberately does NOT
    fill or drop NaN bid/ask — it coerces ``strike`` only — so that distinction
    survives intact down to the pricing layer.

    If a provider is ever added that reports absent quotes as NaN rather than
    0.0, the assumption inverts: condors at MAX WIN (every leg decayed to
    nothing) would start being refused as unknown, silently deleting the modal
    successful outcome from the learning ledger and biasing the structure's
    measured return downward. Symptom to watch for: a run of exits skipped with
    reason ``missing_condor_mid``. Guarded by
    test_condor_max_win_exit_is_recorded_not_deleted and
    test_closing_leg_mid_rejects_an_unknown_bid_but_accepts_a_real_zero.
    """
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


# ── Iron condor geometry ─────────────────────────────────────────────────────
# The short legs sit at the SAME OTM distance as the otm_strangle's wings (±3%
# of spot) so the sold body of the condor is directly comparable to the strangle
# the engine would otherwise buy. The protective long wings sit a further
# CONDOR_WING_OFFSET_PCT beyond them — that offset is what converts an undefined-
# risk strangle sale into a defined-risk condor, and it sets the max loss
# (wing width - net credit).
CONDOR_SHORT_OTM_PCT = 0.03
CONDOR_WING_OFFSET_PCT = 0.02
# How far outside the no-arbitrage bounds a re-priced condor value may sit before
# it is treated as unusable data rather than quote noise. Two ticks on a
# penny-increment chain. Repairing more than this fabricates an outcome.
CONDOR_QUOTE_REPAIR_TOLERANCE = 0.02
# ...and never let a repaired error exceed this fraction of the position's
# capital at risk, so the tolerance stays proportionate on thin-risk condors
# where a flat two ticks would be a large slice of the return.
CONDOR_QUOTE_REPAIR_MAX_RETURN_IMPACT = 0.05


def _row_at_strike(frame: pd.DataFrame, strike: Optional[float], *, tolerance: float = 1e-6) -> Optional[pd.Series]:
    """Exact-strike lookup, used to RE-PRICE an already-booked leg.

    Deliberately NOT _select_otm_wing: that helper filters to strikes still OTM
    relative to the CURRENT spot, which is right when discovering a new structure
    but wrong when re-pricing an open position. If spot has moved through a short
    strike — precisely the case where a condor loses money — the OTM filter drops
    the booked strike and the fallback silently prices a DIFFERENT option, so the
    realized P&L would describe a trade that was never held.
    """
    if strike is None or frame is None or frame.empty:
        return None
    row = _nearest_row(frame, float(strike))
    if row is None:
        return None
    return row if abs(float(row["strike"]) - float(strike)) <= tolerance else None


def _reprice_selection(frame: pd.DataFrame, strike: Optional[float], *, leg: str) -> Dict[str, Any]:
    """_select_otm_wing-shaped result that resolves an ALREADY-BOOKED strike exactly.

    Re-pricing an open position must quote the strike actually held. Running OTM
    discovery instead drops that strike the moment spot trades through it, and the
    nearest-valid fallback then substitutes a strike that is always FURTHER out of
    the money, and therefore cheaper, on both sides:

        call: spot rises above K → filter keeps strikes >= spot > K → K' > K
        put:  spot falls below K → filter keeps strikes <= spot < K → K' < K

    so the exit is systematically under-valued and the realized return is
    understated. Observed live on AMZN (booked 260 call, re-priced on the 265) and
    TTD (booked 24.5, re-priced on the 25).
    """
    if strike is None:
        return {"row": None, "reason": "missing_booked_strike", "quote_quality_label": "partial_or_missing_mid"}
    row = _row_at_strike(frame, strike)
    if row is None:
        return {
            "row": None,
            "reason": "booked_strike_no_longer_listed",
            "quote_quality_label": "partial_or_missing_mid",
            "requested_strike": float(strike),
        }
    return {
        "row": row,
        "reason": None,
        "quote_quality_label": "booked_strike_reprice",
        "fallback_distance": 0.0,
        "requested_strike": float(strike),
    }


def _closing_leg_mid(row: Optional[pd.Series], *, action: str) -> Optional[float]:
    """Mark one leg of an OPEN position, aware of which side of the market closes it.

    `_mid_from_row` voids any leg quoted with `bid <= 0` ("H3: zero bid not
    executable"). That is correct for a leg you must SELL to close: no bid means
    no exit. It is wrong for a leg you must BUY to close, where the ASK governs
    and a `0.00 x 0.05` market is the BEST possible news, not an error.

    Applying the sell-side rule to all four condor legs deleted condors at max
    WIN — the modal successful outcome for a short-vol structure is every leg
    decaying to nothing, and one bid-less leg out of four was enough to void the
    whole exit. Losers, by contrast, have expensive two-sided ITM legs that always
    quote, so the ledger kept losers and dropped winners.

    Only used when re-pricing an open position. Entry keeps the strict rule for
    every leg: refusing to OPEN on a phantom quote biases nothing, because no
    trade is booked.
    """
    if row is None:
        return None
    _INF = float("inf")
    # _safe_float already returns None for NaN/missing; infinity needs screening.
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    if ask is None or ask < 0 or ask == _INF:
        return None
    # A MISSING/NaN/negative/infinite bid means UNKNOWN, not zero. Coercing it to
    # zero silently halves the cost of buying a leg back: a moderately ITM short
    # quoted `NaN x 3.00` marked at 1.50 instead of ~2.95, converting a -95% loss
    # into a +267% win that passed every downstream check because the number
    # looked entirely plausible. Reject the leg exactly as _mid_from_row does.
    if bid is None or bid < 0 or bid == _INF:
        return None
    if ask < bid:  # crossed quote — unusable either way
        return None
    # The ONLY relaxation over _mid_from_row: a PRESENT, VALID zero bid.
    if bid == 0.0 and action == "sell_to_close":
        # Genuinely unsellable: the position gets nothing for it. Mark it
        # worthless rather than voiding the entire structure. Conservative —
        # net_credit = shorts - longs, so a zero here RAISES the cost to close.
        return 0.0
    return float((bid + ask) / 2.0)


def _select_condor_wing(
    frame: pd.DataFrame,
    *,
    short_strike: float,
    target_strike: float,
    leg: str,
) -> Optional[pd.Series]:
    """Nearest protective long wing STRICTLY beyond the short strike.

    Direction is filtered before snapping: a wing at or inside the short strike
    would invert the vertical and silently turn the position into something that
    is not a condor (and not defined-risk), so it must never be selected. Returns
    None when the chain has no strike beyond the short leg, which the caller
    turns into an explicit no-wing skip rather than a naked short.
    """
    if frame is None or frame.empty or "strike" not in frame.columns:
        return None
    candidates = frame[frame["strike"] > short_strike] if leg == "call" else frame[frame["strike"] < short_strike]
    if candidates.empty:
        return None
    return _nearest_row(candidates, target_strike)


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
        _booked_call = _safe_float(pricing_context.get("call_strike"))
        _booked_put = _safe_float(pricing_context.get("put_strike"))
        if _booked_call is not None and _booked_put is not None:
            # RE-PRICE an open position: quote the booked strikes exactly. Running
            # OTM discovery here silently prices a different, cheaper strike once
            # spot trades through the booked one. See _reprice_selection.
            call_target = float(_booked_call)
            put_target = float(_booked_put)
            call_selection = _reprice_selection(front_calls, _booked_call, leg="call")
            put_selection = _reprice_selection(front_puts, _booked_put, leg="put")
        else:
            # DISCOVERY of a new structure: pick the wings against current spot.
            call_target = float(spot * 1.03)
            put_target = float(spot * 0.97)
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

    if structure == "iron_condor":
        booked = {
            "short_call": _safe_float(pricing_context.get("short_call_strike")),
            "long_call": _safe_float(pricing_context.get("long_call_strike")),
            "short_put": _safe_float(pricing_context.get("short_put_strike")),
            "long_put": _safe_float(pricing_context.get("long_put_strike")),
        }
        is_reprice = all(value is not None for value in booked.values())
        condor_provenance: Dict[str, Any] = {"reprice_of_booked_strikes": is_reprice}
        short_call_label = short_put_label = "booked_strike_reprice"
        short_call_fallback = short_put_fallback = 0.0

        if is_reprice:
            # RE-PRICE an open position: resolve the four booked strikes exactly.
            # Never re-discover them — spot may have moved through a short strike
            # (the condor's loss case), and re-discovery would price a different
            # structure than the one actually held. See _row_at_strike.
            short_call_row = _row_at_strike(front_calls, booked["short_call"])
            long_call_row = _row_at_strike(front_calls, booked["long_call"])
            short_put_row = _row_at_strike(front_puts, booked["short_put"])
            long_put_row = _row_at_strike(front_puts, booked["long_put"])
            missing = [
                name for name, row in (
                    ("short_call", short_call_row), ("long_call", long_call_row),
                    ("short_put", short_put_row), ("long_put", long_put_row),
                ) if row is None
            ]
            if missing:
                reason = "condor_strike_no_longer_listed"
                condor_provenance.update({"missing_booked_legs": missing, "final_reason": reason})
                _record_quote(False, reason)
                return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}
            short_call_target = float(booked["short_call"])
            short_put_target = float(booked["short_put"])
            wing_offset = float(pricing_context.get("wing_offset") or (spot * CONDOR_WING_OFFSET_PCT))
        else:
            # DISCOVERY: reuse the strangle's wing selection for the sold body so
            # the short strikes (and their quote-quality/fallback provenance) are
            # picked exactly the way the comparable strangle would pick them.
            short_call_target = float(spot * (1.0 + CONDOR_SHORT_OTM_PCT))
            short_put_target = float(spot * (1.0 - CONDOR_SHORT_OTM_PCT))
            short_call_sel = _select_otm_wing(front_calls, target_strike=short_call_target, spot=spot, leg="call")
            short_put_sel = _select_otm_wing(front_puts, target_strike=short_put_target, spot=spot, leg="put")
            condor_provenance.update(
                {
                    "requested_short_call_strike": round(short_call_target, 4),
                    "requested_short_put_strike": round(short_put_target, 4),
                    "short_call_quote_quality_label": short_call_sel.get("quote_quality_label"),
                    "short_put_quote_quality_label": short_put_sel.get("quote_quality_label"),
                }
            )
            if short_call_sel.get("row") is None:
                reason = str(short_call_sel.get("reason") or "missing_condor_short_call")
                condor_provenance["final_reason"] = reason
                _record_quote(False, reason)
                return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}
            if short_put_sel.get("row") is None:
                reason = str(short_put_sel.get("reason") or "missing_condor_short_put")
                condor_provenance["final_reason"] = reason
                _record_quote(False, reason)
                return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}
            short_call_row = short_call_sel["row"]
            short_put_row = short_put_sel["row"]
            short_call_label = str(short_call_sel.get("quote_quality_label"))
            short_put_label = str(short_put_sel.get("quote_quality_label"))
            short_call_fallback = float(short_call_sel.get("fallback_distance", 0.0) or 0.0)
            short_put_fallback = float(short_put_sel.get("fallback_distance", 0.0) or 0.0)

            # Protective wings a configured offset beyond each short strike.
            wing_offset = float(spot * CONDOR_WING_OFFSET_PCT)
            long_call_row = _select_condor_wing(
                front_calls, short_strike=float(short_call_row["strike"]),
                target_strike=float(short_call_row["strike"]) + wing_offset, leg="call",
            )
            long_put_row = _select_condor_wing(
                front_puts, short_strike=float(short_put_row["strike"]),
                target_strike=float(short_put_row["strike"]) - wing_offset, leg="put",
            )
            if long_call_row is None:
                reason = "no_condor_call_wing"
                condor_provenance["final_reason"] = reason
                _record_quote(False, reason)
                return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}
            if long_put_row is None:
                reason = "no_condor_put_wing"
                condor_provenance["final_reason"] = reason
                _record_quote(False, reason)
                return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}

        short_call_strike = float(short_call_row["strike"])
        short_put_strike = float(short_put_row["strike"])

        if is_reprice:
            # Closing the position: the short body is BOUGHT back (ask governs),
            # the long wings are SOLD (bid governs). See _closing_leg_mid.
            short_call_mid = _closing_leg_mid(short_call_row, action="buy_to_close")
            short_put_mid = _closing_leg_mid(short_put_row, action="buy_to_close")
            long_call_mid = _closing_leg_mid(long_call_row, action="sell_to_close")
            long_put_mid = _closing_leg_mid(long_put_row, action="sell_to_close")
        else:
            short_call_mid = _mid_from_row(short_call_row)
            short_put_mid = _mid_from_row(short_put_row)
            long_call_mid = _mid_from_row(long_call_row)
            long_put_mid = _mid_from_row(long_put_row)
        if None in (short_call_mid, short_put_mid, long_call_mid, long_put_mid):
            reason = "missing_condor_mid"
            condor_provenance["final_reason"] = reason
            _record_quote(False, reason)
            return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}

        long_call_strike = float(long_call_row["strike"])
        long_put_strike = float(long_put_row["strike"])
        # NET CREDIT: what the four-leg structure is worth. Positive = we are paid
        # to put it on. This is the value convention `mid` carries for a condor,
        # so entry books a credit and exit books the cost to close.
        # Value the two short verticals SEPARATELY. Each is independently bounded
        # by [0, its own width]; checking only the aggregate lets a broken single
        # vertical hide behind a cheap one (a call vertical marked 2.50 against a
        # 2.00 width passed because the total 2.515 sat under a 4.00 ceiling, and
        # booked a loss roughly double what the position can structurally lose).
        call_vertical = float(short_call_mid - long_call_mid)
        put_vertical = float(short_put_mid - long_put_mid)
        call_width = long_call_strike - short_call_strike
        put_width = short_put_strike - long_put_strike
        net_credit = call_vertical + put_vertical
        # Only one side can be breached, so the binding risk is the WIDER wing.
        wing_width = float(max(call_width, put_width))
        # No-arbitrage bound on the STRUCTURE's value, used to repair broken
        # quotes when re-pricing an open position.
        #
        # The ceiling is call_width + put_width, NOT max(...): a condor is two
        # short verticals, each worth [0, its own width], and only at EXPIRATION
        # can just one be in the money. This system never holds to expiration —
        # `front_expiry` is the first expiry after earnings while
        # `trades_due_for_exit` fires at earnings_date - 1 day — so the untested
        # side still carries time value at every exit. Clamping at max(width)
        # therefore truncated REAL early-close losses to a suspiciously clean
        # -100%, hiding a genuine cost of the early-close policy.
        #
        # Exceeding the expiration max loss on an early close is informative, not
        # a data error, so no floor is placed on the resulting return.
        value_ceiling = float(call_width + put_width)
        if is_reprice:
            # Repair only NOISE. A large violation is unusable data, and silently
            # clamping it fabricates an outcome — a raw -0.80 became a recorded
            # maximum win. Size the tolerance against the position's capital at
            # risk as well: a flat 0.02 is a couple of ticks on a 0.40 risk base
            # but 20 percentage points of return on a thin 0.10 base.
            _entry_risk = _safe_float(pricing_context.get("max_loss_per_unit"))
            repair_tolerance = CONDOR_QUOTE_REPAIR_TOLERANCE
            if _entry_risk is not None and _entry_risk > 0:
                repair_tolerance = min(
                    repair_tolerance,
                    CONDOR_QUOTE_REPAIR_MAX_RETURN_IMPACT * float(_entry_risk),
                )
            _legs = (
                ("call_vertical", call_vertical, call_width),
                ("put_vertical", put_vertical, put_width),
            )
            # The budget is the TOTAL repair applied to the structure, not a
            # per-leg allowance. Both verticals can violate in the same direction,
            # so a per-leg budget silently delivered up to 2x the advertised error
            # (two legs each repaired just under tolerance manufactured ~10pp of
            # return on a 0.40 base, and booked a maximum win on a position whose
            # true value was negative).
            _violations = {
                name: max(0.0 - value, value - width, 0.0) for name, value, width in _legs
            }
            total_violation = float(sum(_violations.values()))
            if total_violation > repair_tolerance:
                reason = "condor_quote_outside_no_arbitrage_bounds"
                _worst = max(_violations, key=lambda k: _violations[k])
                condor_provenance.update(
                    {"violating_leg": _worst,
                     "leg_violations": {k: round(v, 4) for k, v in _violations.items()},
                     "total_violation": round(total_violation, 4),
                     "repair_tolerance": round(repair_tolerance, 4),
                     "final_reason": reason}
                )
                _record_quote(False, reason)
                return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}
            _clamped: Dict[str, float] = {}
            _bounded: Dict[str, float] = {}
            for name, value, width in _legs:
                bounded_leg = float(min(max(value, 0.0), width))
                if bounded_leg != value:
                    _clamped[name] = round(value, 4)
                _bounded[name] = bounded_leg
            call_vertical = _bounded["call_vertical"]
            put_vertical = _bounded["put_vertical"]
            if _clamped:
                condor_provenance["reprice_value_clamped_from"] = _clamped
            # Each vertical is now inside its own bound, so the total is inside
            # the aggregate bound by construction.
            net_credit = float(call_vertical + put_vertical)
        max_loss = float(wing_width - net_credit)
        condor_provenance.update(
            {
                "selected_short_call_strike": short_call_strike,
                "selected_long_call_strike": long_call_strike,
                "selected_short_put_strike": short_put_strike,
                "selected_long_put_strike": long_put_strike,
                "call_width": round(call_width, 4),
                "put_width": round(put_width, 4),
                "wing_width": round(wing_width, 4),
                "net_credit": round(net_credit, 4),
                "max_loss": round(max_loss, 4),
            }
        )
        # These two sanity gates decide whether a structure is worth OPENING, so
        # they must fire on discovery ONLY. Applying them when re-pricing an open
        # position is destructive: as a condor approaches max loss the breached
        # vertical's cost-to-close approaches the wing width, so ordinary deep-ITM
        # bid/ask noise pushes it past and trips `max_loss <= 0` — precisely on the
        # worst losers. The exit is one-shot (`trades_due_for_exit` matches only
        # earnings_date == as_of + 1 day, and the skip path leaves status='open'),
        # so a single dropped quote orphans that trade permanently and it is never
        # recorded. That removes outcomes from the learning ledger in an
        # outcome-CORRELATED way, inflating measured performance. An already-open
        # position must be marked to market and booked at whatever it is worth,
        # including at or through max loss.
        if not is_reprice:
            if net_credit <= 0:
                # A condor that pays nothing (or costs money) has no premium to
                # capture — never open it.
                reason = "non_positive_condor_credit"
                condor_provenance["final_reason"] = reason
                _record_quote(False, reason)
                return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}
            if max_loss <= 0:
                # Credit >= wing width implies a risk-free structure, which in
                # practice means the quotes are broken rather than a free lunch.
                reason = "condor_credit_exceeds_wing_width"
                condor_provenance["final_reason"] = reason
                _record_quote(False, reason)
                return {"mid": None, "reason": reason, **_payload(provenance=condor_provenance, surface_quality=front_surface_quality, final_reason=reason)}

        pricing_context.update(
            {
                "short_call_strike": short_call_strike,
                "long_call_strike": long_call_strike,
                "short_put_strike": short_put_strike,
                "long_put_strike": long_put_strike,
                "wing_offset": wing_offset,
                "wing_width": wing_width,
                # Capital at risk for this structure. Persisted at ENTRY so the
                # exit can compute return-on-risk against the entry's own credit.
                "max_loss_per_unit": max_loss,
                "net_credit_at_quote": net_credit,
                "short_call_contract": str(short_call_row.get("contractSymbol")),
                "long_call_contract": str(long_call_row.get("contractSymbol")),
                "short_put_contract": str(short_put_row.get("contractSymbol")),
                "long_put_contract": str(long_put_row.get("contractSymbol")),
            }
        )
        # Leg names carry the short_/long_ prefix because execution_scenarios
        # derives fill direction from them (short legs fill toward the bid).
        legs = {
            "short_call": _quote_fields_from_row(
                short_call_row,
                quote_quality_label=short_call_label,
                requested_strike=short_call_target,
                fallback_distance=short_call_fallback,
            ),
            "long_call": _quote_fields_from_row(long_call_row),
            "short_put": _quote_fields_from_row(
                short_put_row,
                quote_quality_label=short_put_label,
                requested_strike=short_put_target,
                fallback_distance=short_put_fallback,
            ),
            "long_put": _quote_fields_from_row(long_put_row),
        }
        condor_provenance["final_reason"] = None
        _record_quote(True)
        return {
            "mid": net_credit,
            "spot": spot,
            "context": pricing_context,
            **_payload(legs=legs, provenance=condor_provenance, surface_quality=front_surface_quality),
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
        trade_id = make_trade_id(symbol, as_of, structure, earnings_date=earnings_date)

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
                move_ratio_units_version=MOVE_RATIO_UNITS_VERSION,  # F1: 1.0-fair basis
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


# Structures whose `mid` is a NET CREDIT received rather than a debit paid.
CREDIT_STRUCTURES = frozenset({"iron_condor"})


def _realized_trade_math(
    *,
    structure: str,
    entry_mid: float,
    exit_mid: float,
    capital_at_risk: Optional[float] = None,
) -> tuple[float, float, float]:
    """Return ``(gross_return_pct, realized_pnl, realized_expansion_pct)``.

    One rule, applied with the correct sign and base for the structure::

        gross_return_pct = (position P&L per share) / (capital at risk) * 100

    For a LONG DEBIT structure the P&L is ``exit - entry`` and the capital at risk
    IS the premium paid (``entry_mid``), so this reduces EXACTLY to the original
    ``((exit - entry) / entry) * 100``. Long-structure numbers are unchanged.

    For a CREDIT structure (iron condor) ``mid`` is the net credit: we are PAID
    ``entry_mid`` to open and PAY ``exit_mid`` to close, so the P&L sign flips to
    ``entry - exit``, and the capital at risk is the defined max loss
    (wing width - credit), NOT the credit. Basing the condor on max loss keeps it
    on the same "return on capital at risk" footing as the long structures, so
    ``avg_return_pct`` stays comparable across structures in the prior store —
    using the credit as the base instead would let a condor print -300% and
    corrupt cross-structure ranking.

    ``realized_expansion_pct`` keeps ONE meaning for every structure: the percent
    change in the structure's own market value (positive = it got more expensive
    = vol expanded). It is a volatility diagnostic, not the position return, and
    for a credit structure it is deliberately the opposite sign to the P&L.
    """
    entry_mid = float(entry_mid)
    exit_mid = float(exit_mid)
    expansion_pct = ((exit_mid - entry_mid) / entry_mid) * 100.0
    if structure in CREDIT_STRUCTURES:
        pnl_per_share = entry_mid - exit_mid
        # Falling back to the credit as the base produces a wrong-but-plausible
        # number (the "-300%" case the docstring warns about) that is worse for a
        # learning ledger than an obviously-missing one. Keep the fallback so the
        # trade still closes with a correctly SIGNED P&L, but the caller is
        # expected to treat a missing capital_at_risk as a provenance defect.
        if capital_at_risk is not None and float(capital_at_risk) > 0:
            base = float(capital_at_risk)
        else:
            base = entry_mid
            logger.warning(
                "credit-structure return re-based onto the entry credit because "
                "max_loss_per_unit was missing from the persisted pricing context; "
                "this return is NOT return-on-risk and is not comparable to the "
                "other structures.",
            )
    else:
        pnl_per_share = exit_mid - entry_mid
        base = entry_mid
    gross_return_pct = (pnl_per_share / base) * 100.0
    return gross_return_pct, pnl_per_share * 100.0, expansion_pct


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

        # Credit structures (iron condor) invert the P&L sign and use the defined
        # max loss recorded at ENTRY as the return base; long debit structures are
        # unchanged. See _realized_trade_math.
        entry_pricing_context = note_payload.get("pricing_context") or {}
        gross_return_pct, realized_pnl, realized_expansion_pct = _realized_trade_math(
            structure=structure,
            entry_mid=entry_mid,
            exit_mid=float(exit_mid),
            capital_at_risk=_safe_float(entry_pricing_context.get("max_loss_per_unit")),
        )
        execution_penalty = float(row.get("execution_penalty_at_entry") or 0.0)
        modeled_cost_pct = 26.0 * execution_penalty
        realized_return_pct = gross_return_pct - modeled_cost_pct
        entry_execution_scenarios = _loads_dict(row.get("entry_execution_scenarios_json"))
        if not entry_execution_scenarios:
            entry_execution_scenarios = note_payload.get("entry_execution_scenarios", {})
        scenario_outcomes = compare_execution_scenarios(
            entry=entry_execution_scenarios,
            exit=exit_execution_scenarios,
            structure=structure,
            capital_at_risk=_safe_float(entry_pricing_context.get("max_loss_per_unit")),
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
