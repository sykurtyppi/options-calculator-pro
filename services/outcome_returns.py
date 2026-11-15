"""Which return an outcome row means, named.

Every resolved paper outcome (selector trade or baseline) carries more than one
return, and they answer different questions:

* ``mid`` (RETURN_MID): exit mid against entry mid, before any cost.
* ``modeled_net`` (RETURN_MODELED_NET): the mid return minus the cost the
  selector *modeled* at entry (26 x execution_penalty for selector trades,
  ``modeled_cost_pct`` for baselines). This is what the ``realized_return_pct``
  column holds. It is a diagnostic of the selector's own cost guess, so it
  must never be the measure of whether the selector works.
* ``conservative_fill`` (RETURN_CLAIM): the return at the claim contract's
  conservative fill (natural price plus slippage and fees on every contract,
  services/claim_contract.py), computed from the booked legs' quotes and stored
  in the exit scenarios' ``scenario_outcomes``. This is the claimed return.

Evidence claims (the evidence report's ``claimed`` analysis, the
pre-registration's H1/H2) and learning (structure priors) use
``conservative_fill``. A row without one has no claimed return: it is withheld
and counted, never filled in from another basis.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional

RETURN_MID = "mid"
RETURN_MODELED_NET = "modeled_net"
RETURN_CLAIM = "conservative_fill"
RETURN_BASES = (RETURN_MID, RETURN_MODELED_NET, RETURN_CLAIM)

# The basis structure priors learn from (services/outcome_recorder.py).
PRIOR_LEARNING_RETURN_BASIS = RETURN_CLAIM


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_dict(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, bytes)) and value:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _exit_scenarios(row: Mapping[str, Any]) -> dict:
    for key in ("exit_execution_scenarios_json", "exit_execution_scenarios"):
        parsed = _as_dict(row.get(key))
        if parsed:
            return parsed
    return {}


def scenario_return_pct(row: Mapping[str, Any], level: str) -> Optional[float]:
    """The stored return at one execution-scenario level, or None."""
    outcomes = _as_dict(_exit_scenarios(row).get("scenario_outcomes"))
    returns = _as_dict(outcomes.get("realized_return_pct"))
    return _finite(returns.get(level))


def claimed_return_pct(row: Mapping[str, Any]) -> Optional[float]:
    """The conservative-fill return the claim contract prices a claim at."""
    return scenario_return_pct(row, RETURN_CLAIM)


def return_pct(row: Mapping[str, Any], basis: str) -> Optional[float]:
    """The row's return on ``basis`` (one of RETURN_BASES), or None."""
    if basis == RETURN_MODELED_NET:
        return _finite(row.get("realized_return_pct"))
    if basis in (RETURN_MID, RETURN_CLAIM):
        return scenario_return_pct(row, basis)
    raise ValueError(f"unknown return basis {basis!r}; expected one of {RETURN_BASES}")


def outcome_returns(row: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """All three named returns of one row."""
    return {basis: return_pct(row, basis) for basis in RETURN_BASES}


__all__ = [
    "PRIOR_LEARNING_RETURN_BASIS",
    "RETURN_BASES",
    "RETURN_CLAIM",
    "RETURN_MID",
    "RETURN_MODELED_NET",
    "claimed_return_pct",
    "outcome_returns",
    "return_pct",
    "scenario_return_pct",
]
