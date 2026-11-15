"""Which contracts each structure books: one set of rules for scoring and for entry.

The selector used to score strangles and condors on the AT-THE-MONEY spreads of
the nearest expiry, while the forward loop picked the real OTM legs only at
entry, on another expiry and with its own rules. The structure that was scored
was not the structure that was booked.

These are the discovery rules both now use (the helpers moved unchanged from
scripts/run_forward_loop.py, which re-exports them under their old names):

* the trade expiry is the first listed expiry on or after the session on which
  the earnings reaction can print (services.release_timing.reaction_session),
  so a position always spans the event it is priced for; unknown timing is
  treated as after the close. (The loop used to take the first expiry on or
  after the earnings DATE, which for an after-close report is an option that
  expires before the market can react.) Calendars add a back expiry at least
  CALENDAR_BACK_GAP_DAYS later;
* straddle: the strike nearest spot, call and put at the same strike;
* strangle: OTM wings near spot x (1 +/- OTM_WING_PCT), bounded fallback;
* iron condor: the sold body like the strangle's wings, protective wings
  CONDOR_WING_OFFSET_PCT of spot further out, strictly beyond the shorts;
* calendars: the front strike nearest spot, the back leg at the same strike.

``discover_legs`` applies them to a chain and returns the legs with their
quotes, or the reason the structure cannot be formed.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from services.release_timing import reaction_session

OTM_WING_PCT = 0.03
OTM_STRANGLE_FALLBACK_DISTANCE_PCT = 0.025
OTM_STRANGLE_MIN_FALLBACK_DISTANCE = 2.5
# The condor's sold body sits at the strangle's wing distance, so the two are
# comparable; the protective wings sit CONDOR_WING_OFFSET_PCT further out.
CONDOR_SHORT_OTM_PCT = OTM_WING_PCT
CONDOR_WING_OFFSET_PCT = 0.02
CALENDAR_BACK_GAP_DAYS = 14

STRUCTURES = ("atm_straddle", "otm_strangle", "call_calendar", "put_calendar", "iron_condor")


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        return parsed if pd.notna(parsed) else None
    except Exception:
        return None


def mid_from_row(row: Optional[pd.Series]) -> Optional[float]:
    if row is None:
        return None
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:  # H3: zero bid not executable
        return None
    mid = (bid + ask) / 2.0
    return float(mid) if mid > 0 else None


def quote_invalid_reason(row: Optional[pd.Series], *, leg: str) -> str:
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


def max_strangle_fallback_distance(spot: float) -> float:
    return max(OTM_STRANGLE_MIN_FALLBACK_DISTANCE, float(spot) * OTM_STRANGLE_FALLBACK_DISTANCE_PCT)


def select_otm_wing(
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
    max_distance = max_strangle_fallback_distance(spot)
    primary = work.iloc[0]
    if float(primary["fallback_distance"]) > max_distance:
        return {
            "row": None,
            "reason": "wing_strike_unavailable",
            "quote_quality_label": "partial_or_missing_mid",
            "max_fallback_distance": round(max_distance, 4),
        }

    primary_mid = mid_from_row(primary)
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
    invalid_reasons = [quote_invalid_reason(row, leg=leg) for _, row in bounded.iterrows()]
    for _, candidate in bounded.iterrows():
        candidate_mid = mid_from_row(candidate)
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


def nearest_row(frame: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    work = frame.copy()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work = work.dropna(subset=["strike"])
    if work.empty:
        return None
    work["dist"] = (work["strike"] - float(strike)).abs()
    return work.sort_values("dist").iloc[0]


def row_at_strike(frame: pd.DataFrame, strike: Optional[float], *, tolerance: float = 1e-6) -> Optional[pd.Series]:
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
    row = nearest_row(frame, float(strike))
    if row is None:
        return None
    return row if abs(float(row["strike"]) - float(strike)) <= tolerance else None


def select_condor_wing(
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
    return nearest_row(candidates, target_strike)


def trade_expiry(expiries: Iterable[Any], earnings_date: date, release_timing: Any = None) -> Optional[date]:
    """First listed expiry on or after the session the earnings reaction prints."""
    first_session = reaction_session(earnings_date, release_timing)
    parsed = sorted({pd.Timestamp(value).date() for value in expiries if value is not None and str(value)})
    return next((expiry for expiry in parsed if expiry >= first_session), None)


def calendar_back_expiry(expiries: Iterable[Any], front: date, gap_days: int = CALENDAR_BACK_GAP_DAYS) -> Optional[date]:
    target = front + timedelta(days=gap_days)
    parsed = sorted({pd.Timestamp(value).date() for value in expiries if value is not None and str(value)})
    return next((expiry for expiry in parsed if expiry >= target), None)


def _leg_quote(row: pd.Series) -> Dict[str, Any]:
    bid, ask = _safe_float(row.get("bid")), _safe_float(row.get("ask"))
    mid = mid_from_row(row)
    contract = row.get("contractSymbol")
    return {
        "strike": float(row["strike"]),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_pct": None if mid is None else round((ask - bid) / mid * 100.0, 6),
        "contract": None if contract is None or (isinstance(contract, float) and np.isnan(contract)) else str(contract),
    }


def _result(legs: Dict[str, pd.Series], reason: Optional[str] = None) -> Dict[str, Any]:
    if reason is not None:
        return {"legs": {}, "reason": reason, "leg_spread_pct": None}
    quotes = {name: _leg_quote(row) for name, row in legs.items()}
    spreads = [q["spread_pct"] for q in quotes.values()]
    if any(value is None for value in spreads):
        return {"legs": quotes, "reason": "missing_leg_mid", "leg_spread_pct": None}
    return {"legs": quotes, "reason": None, "leg_spread_pct": round(float(np.mean(spreads)), 6)}


def discover_legs(
    structure: str,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    *,
    back_calls: Optional[pd.DataFrame] = None,
    back_puts: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """The legs ``structure`` would book on this chain, or why it cannot be formed.

    Returns ``{"legs": {name: {strike, bid, ask, mid, spread_pct, contract}},
    "reason": None | str, "leg_spread_pct": mean leg spread as % of its mid}``.
    """
    if spot is None or not np.isfinite(spot) or spot <= 0:
        return _result({}, "no_spot_price")
    if structure == "atm_straddle":
        call = nearest_row(calls, float(spot))
        put = None if call is None else row_at_strike(puts, float(call["strike"]))
        if call is None or put is None:
            return _result({}, "missing_atm_pair")
        return _result({"call": call, "put": put})
    if structure == "otm_strangle":
        call = select_otm_wing(calls, target_strike=float(spot * (1.0 + OTM_WING_PCT)), spot=spot, leg="call")
        put = select_otm_wing(puts, target_strike=float(spot * (1.0 - OTM_WING_PCT)), spot=spot, leg="put")
        for selection, default in ((call, "missing_call_mid"), (put, "missing_put_mid")):
            if selection.get("row") is None:
                return _result({}, str(selection.get("reason") or default))
        return _result({"call": call["row"], "put": put["row"]})
    if structure == "iron_condor":
        short_call = select_otm_wing(calls, target_strike=float(spot * (1.0 + CONDOR_SHORT_OTM_PCT)), spot=spot, leg="call")
        short_put = select_otm_wing(puts, target_strike=float(spot * (1.0 - CONDOR_SHORT_OTM_PCT)), spot=spot, leg="put")
        if short_call.get("row") is None:
            return _result({}, str(short_call.get("reason") or "missing_condor_short_call"))
        if short_put.get("row") is None:
            return _result({}, str(short_put.get("reason") or "missing_condor_short_put"))
        offset = float(spot * CONDOR_WING_OFFSET_PCT)
        sc, sp = short_call["row"], short_put["row"]
        long_call = select_condor_wing(calls, short_strike=float(sc["strike"]), target_strike=float(sc["strike"]) + offset, leg="call")
        long_put = select_condor_wing(puts, short_strike=float(sp["strike"]), target_strike=float(sp["strike"]) - offset, leg="put")
        if long_call is None:
            return _result({}, "no_condor_call_wing")
        if long_put is None:
            return _result({}, "no_condor_put_wing")
        return _result({"short_call": sc, "long_call": long_call, "short_put": sp, "long_put": long_put})
    if structure in ("call_calendar", "put_calendar"):
        front_frame = calls if structure == "call_calendar" else puts
        back_frame = back_calls if structure == "call_calendar" else back_puts
        if back_frame is None or getattr(back_frame, "empty", True):
            return _result({}, "no_back_expiry_for_calendar")
        front = nearest_row(front_frame, float(spot))
        if front is None:
            return _result({}, "missing_front_leg")
        back = row_at_strike(back_frame, float(front["strike"]))
        if back is None:
            return _result({}, "missing_back_leg")
        return _result({"front": front, "back": back})
    raise ValueError(f"unknown structure {structure!r}")


# Booked-context keys (scripts/run_forward_loop.fetch_structure_quote) per leg.
_BOOKED_STRIKE_KEYS = {
    "atm_straddle": {"call": "strike", "put": "straddle_put_strike"},
    "otm_strangle": {"call": "call_strike", "put": "put_strike"},
    "iron_condor": {"short_call": "short_call_strike", "long_call": "long_call_strike",
                    "short_put": "short_put_strike", "long_put": "long_put_strike"},
    "call_calendar": {"front": "strike", "back": "calendar_back_strike"},
    "put_calendar": {"front": "strike", "back": "calendar_back_strike"},
}


def compare_booked_to_scored(structure: str, booked_context: Any, structure_legs: Any) -> Dict[str, Any]:
    """Did entry book the contracts the selector scored?

    ``match`` is True/False when both sides are known, None when the analysis
    carried no scored legs (e.g. a snapshot built without a chain). Entry
    quotes are fetched minutes after the analysis, so a move in spot can shift
    a strike; the record makes that visible instead of silent.
    """
    context = booked_context if isinstance(booked_context, dict) else {}
    legs = structure_legs if isinstance(structure_legs, dict) else {}
    scored = ((legs.get("structures") or {}).get(structure) or {}).get("legs") or {}
    keys = _BOOKED_STRIKE_KEYS.get(structure, {})
    booked_strikes = {leg: _safe_float(context.get(key)) for leg, key in keys.items()}
    scored_strikes = {leg: _safe_float((scored.get(leg) or {}).get("strike")) for leg in keys}
    scored_expiry = legs.get("trade_expiry")
    booked_expiry = context.get("front_expiry")
    if not scored or scored_expiry is None:
        match = None
    else:
        match = (
            str(booked_expiry) == str(scored_expiry)
            and all(booked_strikes[leg] is not None and scored_strikes[leg] is not None
                    and abs(booked_strikes[leg] - scored_strikes[leg]) < 1e-6 for leg in keys)
        )
        if structure in ("call_calendar", "put_calendar"):
            match = match and str(context.get("back_expiry")) == str(legs.get("back_expiry"))
    return {
        "match": match,
        "scored_expiry": scored_expiry,
        "booked_expiry": booked_expiry,
        "scored_strikes": scored_strikes,
        "booked_strikes": booked_strikes,
    }


__all__ = [
    "CALENDAR_BACK_GAP_DAYS", "CONDOR_SHORT_OTM_PCT", "CONDOR_WING_OFFSET_PCT", "OTM_STRANGLE_FALLBACK_DISTANCE_PCT",
    "OTM_STRANGLE_MIN_FALLBACK_DISTANCE", "OTM_WING_PCT", "STRUCTURES",
    "calendar_back_expiry", "compare_booked_to_scored", "discover_legs", "max_strangle_fallback_distance", "mid_from_row", "nearest_row",
    "quote_invalid_reason", "row_at_strike", "select_condor_wing", "select_otm_wing", "trade_expiry",
]
