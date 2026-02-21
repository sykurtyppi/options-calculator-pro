"""
Execution Cost Model
====================

Reusable transaction-cost and slippage estimates for option spreads.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np


@dataclass(frozen=True)
class ExecutionProfile:
    """Execution profile parameters."""
    name: str
    commission_per_contract: float
    spread_capture_fraction: float
    impact_coefficient: float
    min_slippage_per_share: float
    regulatory_fee_per_contract: float


DEFAULT_PROFILES: Dict[str, ExecutionProfile] = {
    "paper": ExecutionProfile(
        name="paper",
        commission_per_contract=0.0,
        spread_capture_fraction=0.12,
        impact_coefficient=0.18,
        min_slippage_per_share=0.003,
        regulatory_fee_per_contract=0.0,
    ),
    "retail": ExecutionProfile(
        name="retail",
        commission_per_contract=0.65,
        spread_capture_fraction=0.30,
        impact_coefficient=0.32,
        min_slippage_per_share=0.010,
        regulatory_fee_per_contract=0.015,
    ),
    "institutional": ExecutionProfile(
        name="institutional",
        commission_per_contract=0.25,
        spread_capture_fraction=0.20,
        impact_coefficient=0.24,
        min_slippage_per_share=0.006,
        regulatory_fee_per_contract=0.010,
    ),
    "institutional_tight": ExecutionProfile(
        name="institutional_tight",
        commission_per_contract=0.18,
        spread_capture_fraction=0.16,
        impact_coefficient=0.20,
        min_slippage_per_share=0.004,
        regulatory_fee_per_contract=0.008,
    ),
}


class ExecutionCostModel:
    """Estimate round-trip execution cost for calendar spreads."""

    def __init__(self, profile_name: str = "institutional", overrides: Optional[Dict[str, Any]] = None):
        profile = DEFAULT_PROFILES.get(profile_name, DEFAULT_PROFILES["institutional"])
        if overrides:
            profile = ExecutionProfile(
                name=profile.name,
                commission_per_contract=float(overrides.get("commission_per_contract", profile.commission_per_contract)),
                spread_capture_fraction=float(overrides.get("spread_capture_fraction", profile.spread_capture_fraction)),
                impact_coefficient=float(overrides.get("impact_coefficient", profile.impact_coefficient)),
                min_slippage_per_share=float(overrides.get("min_slippage_per_share", profile.min_slippage_per_share)),
                regulatory_fee_per_contract=float(overrides.get("regulatory_fee_per_contract", profile.regulatory_fee_per_contract)),
            )
        self.profile = profile

    @property
    def profile_name(self) -> str:
        return self.profile.name

    @staticmethod
    def available_profiles() -> Dict[str, Dict[str, float]]:
        """Return available execution profiles."""
        return {
            name: {
                "commission_per_contract": profile.commission_per_contract,
                "spread_capture_fraction": profile.spread_capture_fraction,
                "impact_coefficient": profile.impact_coefficient,
                "min_slippage_per_share": profile.min_slippage_per_share,
                "regulatory_fee_per_contract": profile.regulatory_fee_per_contract,
            }
            for name, profile in DEFAULT_PROFILES.items()
        }

    def estimate_calendar_round_trip_cost(
        self,
        short_spread: float,
        long_spread: float,
        average_volume: float,
        open_interest: float,
        contracts: int = 1,
    ) -> Dict[str, float]:
        """
        Estimate calendar spread cost.

        Returns:
            dict with per-share, per-contract and total cost estimates.
        """
        short_spread = max(0.0, float(short_spread))
        long_spread = max(0.0, float(long_spread))
        average_volume = max(1.0, float(average_volume))
        open_interest = max(1.0, float(open_interest))
        contracts = max(1, int(contracts))

        gross_spread = short_spread + long_spread
        liq_penalty = np.clip(
            1.30 - (np.log1p(average_volume) / 8.2) - (np.log1p(open_interest) / 9.0),
            0.20,
            1.45,
        )

        slippage_per_share = max(
            self.profile.min_slippage_per_share,
            gross_spread * self.profile.spread_capture_fraction * (1.0 + self.profile.impact_coefficient * liq_penalty),
        )
        commission_per_share = (self.profile.commission_per_contract * 2.0) / 100.0
        fee_per_share = (self.profile.regulatory_fee_per_contract * 2.0) / 100.0
        cost_per_share = slippage_per_share + commission_per_share + fee_per_share
        cost_per_contract = cost_per_share * 100.0
        total_cost = cost_per_contract * contracts

        return {
            "profile": self.profile.name,
            "slippage_per_share": float(slippage_per_share),
            "commission_per_share": float(commission_per_share),
            "fee_per_share": float(fee_per_share),
            "cost_per_share": float(cost_per_share),
            "cost_per_contract": float(cost_per_contract),
            "total_cost": float(total_cost),
        }

