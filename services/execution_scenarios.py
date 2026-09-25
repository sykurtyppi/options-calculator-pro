from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import math

from utils.quotes import safe_mid


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
    # Spread costs above are computed on the RAW signed values, where "more
    # adverse" always moves the number the same way regardless of credit/debit.
    # Only the REPORTED scenario values are flipped for credit structures, so a
    # condor's scenario value reads as a positive net credit like its `mid`.
    orientation = _structure_value_orientation(structure)
    reported_values = {
        key: (None if val is None else orientation * float(val)) for key, val in values.items()
    }
    return ExecutionScenarioSet(
        structure=structure,
        phase=normalized_phase,
        scenario_values={key: _round_or_none(val) for key, val in reported_values.items()},
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
    structure: str | None = None,
    capital_at_risk: float | None = None,
) -> dict[str, dict[str, float | None]]:
    """Per-scenario realized return/P&L between an entry and an exit fill set.

    ``structure`` and ``capital_at_risk`` are optional so existing callers keep
    their exact behaviour. When ``structure`` is a CREDIT structure (iron condor)
    the P&L sign flips — we are paid the entry credit and pay the exit cost to
    close — and the return is taken against ``capital_at_risk`` (the defined max
    loss) so it stays on the same basis as the debit structures' return on
    premium paid. This mirrors run_forward_loop._realized_trade_math.
    """
    entry_values = ((entry or {}).get("scenario_values") or {}) if isinstance(entry, Mapping) else {}
    exit_values = ((exit or {}).get("scenario_values") or {}) if isinstance(exit, Mapping) else {}
    is_credit = structure in _CREDIT_STRUCTURES if structure else False
    risk_base = _finite_float(capital_at_risk)
    returns: dict[str, float | None] = {}
    pnl: dict[str, float | None] = {}
    for name, _distance in SCENARIO_LEVELS:
        entry_val = _finite_float(entry_values.get(name))
        exit_val = _finite_float(exit_values.get(name))
        if entry_val is None or exit_val is None:
            returns[name] = None
            pnl[name] = None
            continue
        # The `entry_val <= 0` guard exists to avoid dividing by a non-positive
        # premium base. A credit structure with an explicit capital_at_risk base
        # does not divide by entry_val at all, so the guard must not apply: a
        # thin-credit condor legitimately shows a non-positive net credit in its
        # ADVERSE fill scenarios, and nulling those dropped exactly the worst
        # execution cases, biasing the scenario diagnostics optimistic.
        _has_risk_base = is_credit and risk_base is not None and risk_base > 0
        if entry_val <= 0 and not _has_risk_base:
            returns[name] = None
            pnl[name] = None
            continue
        if is_credit:
            pnl_per_share = entry_val - exit_val
            if entry_val <= 0:
                # This scenario says the "credit" structure was actually opened
                # for a DEBIT, so capital at risk is not the position's nominal
                # max loss and no coherent return base exists. The P&L is still
                # arithmetically meaningful; the ratio would not be (dividing an
                # adverse-fill numerator by a mid-fill base inflates it by an
                # order of magnitude). Report the P&L, withhold the return.
                pnl[name] = round(pnl_per_share * 100.0, 6)
                returns[name] = None
                continue
            base = risk_base if _has_risk_base else entry_val
        else:
            pnl_per_share = exit_val - entry_val
            base = entry_val
        pnl[name] = round(pnl_per_share * 100.0, 6)
        returns[name] = round((pnl_per_share / base) * 100.0, 6)
    return {"realized_return_pct": returns, "realized_pnl": pnl}


def _normalize_legs(raw_legs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, raw in raw_legs.items():
        if not isinstance(raw, Mapping):
            continue
        bid = _finite_float(raw.get("bid"))
        ask = _finite_float(raw.get("ask"))
        mid = _finite_float(raw.get("mid"))
        # Canonical rule: only derive a mid from an executable two-sided quote.
        if mid is None:
            mid = safe_mid(bid, ask)
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
    if structure == "iron_condor":
        # Four legs named short_call / long_call / short_put / long_put. The sold
        # legs must be -1 so an adverse entry fill moves them toward the BID (a
        # seller receives less), not the ask.
        return {name: (-1 if str(name).startswith("short") else 1) for name in legs}
    return {name: 1 for name in legs}


# Structures whose natural signed value (long legs minus short legs) is negative
# because they are opened for a NET CREDIT. Their scenario values are reported
# flipped so the number carries the same meaning as the structure's quoted `mid`
# (a positive credit), while the per-leg fill directions above stay physically
# correct.
_CREDIT_STRUCTURES = frozenset({"iron_condor"})


def _structure_value_orientation(structure: str) -> int:
    return -1 if structure in _CREDIT_STRUCTURES else 1


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
