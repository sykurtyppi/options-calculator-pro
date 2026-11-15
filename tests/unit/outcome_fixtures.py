"""Exit scenarios carrying a claimed (conservative-fill) return, for tests.

Structure priors learn the conservative-fill return stored in an exit's
``scenario_outcomes`` (services.outcome_returns), not the modeled-net
``realized_return_pct``. Tests that exercise finalization mechanics pass these
so the prior update runs; ``conservative`` defaults to the modeled value so the
numbers those tests assert are unchanged.
"""
from __future__ import annotations

from typing import Any, Optional


def claimed_exit_scenarios(modeled: float, conservative: Optional[float] = None) -> dict[str, Any]:
    value = modeled if conservative is None else conservative
    return {
        "scenario_outcomes": {
            "realized_return_pct": {"mid": modeled, "natural": value, "conservative_fill": value},
            "realized_pnl": {},
        }
    }
