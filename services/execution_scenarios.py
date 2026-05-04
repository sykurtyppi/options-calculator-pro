from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import math


SCENARIO_LEVELS: tuple[tuple[str, float], ...] = (
    ("mid", 0.0),
    ("cross_25", 0.25),
    ("cross_50", 0.50),
    ("conservative", 0.50),
)


@dataclass(frozen=True)
class ExecutionScenarioSet:
    structure: str
    phase: str
    scenario_values: dict[str, float | None]
    spread_cost_vs_mid: dict[str, float | None]
    spread_cost_vs_mid_pct: dict[str, float | None]
    total_leg_mid: float | None
    total_leg_spread: float | None
    spread_as_pct_of_premium: float | None
    legs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_execution_scenarios(
    *,
    structure: str,
    quote_payload: Mapping[str, Any] | None,
    phase: str,
) -> ExecutionScenarioSet:
    """Compute paper fill scenarios from per-leg bid/ask/mid quotes.

    ``phase`` is ``entry`` or ``exit``. For long-vol debit structures, entry
    moves fills toward asks and exit moves fills toward bids. For calendars, the
    front leg is treated as short and the back leg as long.
    """
    quote = quote_payload or {}
    bid_ask_mid = quote.get("bid_ask_mid") if isinstance(quote.get("bid_ask_mid"), Mapping) else {}
    legs_raw = bid_ask_mid.get("legs") if isinstance(bid_ask_mid, Mapping) else {}
    legs = _normalize_legs(legs_raw if isinstance(legs_raw, Mapping) else {})
    signs = _leg_signs(structure, legs)
    normalized_phase = "exit" if str(phase).lower() == "exit" else "entry"

    values: dict[str, float | None] = {}
    for name, distance in SCENARIO_LEVELS:
        values[name] = _scenario_value(
            legs=legs,
            signs=signs,
            distance=distance,
            phase=normalized_phase,
        )

    mid_value = values.get("mid")
    spread_cost: dict[str, float | None] = {}
    spread_cost_pct: dict[str, float | None] = {}
    for name, value in values.items():
        if value is None or mid_value is None:
            spread_cost[name] = None
            spread_cost_pct[name] = None
            continue
        cost = (value - mid_value) if normalized_phase == "entry" else (mid_value - value)
        spread_cost[name] = round(float(cost), 6)
        spread_cost_pct[name] = round((float(cost) / abs(mid_value)) * 100.0, 6) if mid_value else None

    total_mid = _sum_or_none([leg.get("mid") for leg in legs.values()])
    total_spread = _sum_or_none([
        (leg.get("ask") - leg.get("bid"))
        for leg in legs.values()
        if leg.get("bid") is not None and leg.get("ask") is not None
    ])
    spread_as_pct = (
        round((float(total_spread) / abs(float(mid_value))) * 100.0, 6)
        if total_spread is not None and mid_value not in (None, 0)
        else None
    )
    return ExecutionScenarioSet(
        structure=structure,
        phase=normalized_phase,
        scenario_values={key: _round_or_none(val) for key, val in values.items()},
        spread_cost_vs_mid=spread_cost,
        spread_cost_vs_mid_pct=spread_cost_pct,
        total_leg_mid=_round_or_none(total_mid),
        total_leg_spread=_round_or_none(total_spread),
        spread_as_pct_of_premium=spread_as_pct,
        legs=legs,
    )


def compare_execution_scenarios(
    *,
    entry: Mapping[str, Any] | None,
    exit: Mapping[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    entry_values = ((entry or {}).get("scenario_values") or {}) if isinstance(entry, Mapping) else {}
    exit_values = ((exit or {}).get("scenario_values") or {}) if isinstance(exit, Mapping) else {}
    returns: dict[str, float | None] = {}
    pnl: dict[str, float | None] = {}
    for name, _distance in SCENARIO_LEVELS:
        entry_val = _finite_float(entry_values.get(name))
        exit_val = _finite_float(exit_values.get(name))
        if entry_val is None or entry_val <= 0 or exit_val is None:
            returns[name] = None
            pnl[name] = None
            continue
        pnl_value = (exit_val - entry_val) * 100.0
        pnl[name] = round(pnl_value, 6)
        returns[name] = round(((exit_val - entry_val) / entry_val) * 100.0, 6)
    return {"realized_return_pct": returns, "realized_pnl": pnl}


def _normalize_legs(raw_legs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, raw in raw_legs.items():
        if not isinstance(raw, Mapping):
            continue
        bid = _finite_float(raw.get("bid"))
        ask = _finite_float(raw.get("ask"))
        mid = _finite_float(raw.get("mid"))
        if mid is None and bid is not None and ask is not None and ask >= bid:
            mid = (bid + ask) / 2.0
        result[str(name)] = {
            **dict(raw),
            "bid": bid,
            "ask": ask,
            "mid": mid,
        }
    return result


def _leg_signs(structure: str, legs: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    if structure in {"call_calendar", "put_calendar"}:
        return {name: (-1 if name == "front" else 1) for name in legs}
    return {name: 1 for name in legs}


def _scenario_value(
    *,
    legs: Mapping[str, Mapping[str, Any]],
    signs: Mapping[str, int],
    distance: float,
    phase: str,
) -> float | None:
    if not legs:
        return None
    value = 0.0
    for name, leg in legs.items():
        sign = int(signs.get(name, 1))
        bid = _finite_float(leg.get("bid"))
        ask = _finite_float(leg.get("ask"))
        mid = _finite_float(leg.get("mid"))
        if bid is None or ask is None or mid is None or ask < bid:
            return None
        spread = ask - bid
        if phase == "entry":
            fill = mid + sign * distance * spread
        else:
            fill = mid - sign * distance * spread
        value += sign * fill
    return value


def _sum_or_none(values: list[Any]) -> float | None:
    parsed = [_finite_float(value) for value in values]
    clean = [value for value in parsed if value is not None]
    return sum(clean) if clean else None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _round_or_none(value: Any) -> float | None:
    parsed = _finite_float(value)
    return round(parsed, 6) if parsed is not None else None
