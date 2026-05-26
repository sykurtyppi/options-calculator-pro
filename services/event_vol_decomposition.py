"""
Event-vol decomposition — single source of truth (P-5a math).

Centralises the variance-additive split of the near-term ATM straddle premium
into event vs non-event components.  The same decomposition used to live in
two places (services.earnings_vol_snapshot and scripts.backtest_iv_expansion_study);
this module is the canonical implementation that both call.

Math (P-5a, see commit ``fix(event_vol): variance-additive decomposition…``):

Step 1 — Convert the legacy MAD-form straddle premium to a 1σ-form pct move.
    Brenner-Subrahmanyam (1988): for an ATM straddle priced under risk-neutral
    lognormal at calendar-time T_years,
        (call_mid + put_mid)/S = σ_implied · √(2 · T_years / π)
    Therefore the 1σ percent move over [0, T_years] is
        σ_implied · √T_years · 100 = (call+put)/S · √(π/2) · 100
                                   = near_term_implied_move_pct · √(π/2).
    The conversion folds out T cleanly.

Step 2 — Variance-additive event split, in calendar time.
    Assume earnings consumes one calendar day of total variance; both
    σ_implied and σ_HAR are quoted as annualised vols on calendar time per
    industry convention:
        σ_imp²·T = σ_HAR²·(T-1d) + σ_event²·1d
    ⇒  event_implied_move_pct² = near_term_implied_sigma_pct²
                               − non_event_move_pct²
    where both terms are 1σ-form on calendar time.

Known bias — calendar vs trading days
    σ_HAR is annualised via realized-vol kernels using √252 (trading-day
    basis). The calendar-day formula here scales it by √(T_calendar/365),
    which implicitly assumes the trading-to-calendar-day ratio in the
    window equals 252/365 ≈ 0.690. In practice that ratio drifts:
        7 calendar days  → 5 trading days   → ratio 0.71 (slightly above)
        21 calendar days → ~15 trading days → ratio 0.71 (slightly above)
        across long weekends / holidays     → ratio can fall to 0.50-0.60
    The downstream impact on event_implied_move_pct is typically <1% for
    clean Mon-Fri windows and 2-5% across long weekends or holiday weeks.
    Direction: calendar-day over-attributes variance to non-event diffusion
    (when the ratio is below 252/365), which slightly understates event_move.

    Worked example: 7-day setup, σ_annual = 0.20, near_term_implied_move
    = 5.0 (MAD-form → σ_implied = 6.27%). Calendar-day decomposition
    yields event_move = 5.72%; a 4-trading-day-diffusion variant yields
    5.74%. Difference here ≈ 0.35% relative. Wider for windows that span
    Memorial Day, Thanksgiving, etc.

    The cleanest fix is to count actual trading days in the sub-window
    (Mon-Fri, holiday-aware) and use T_trading/252. That fix is deferred
    because (a) the bias magnitude is small relative to other selector
    noise, (b) introducing a market-calendar dependency requires its own
    design pass. Documented here so the next person investigating
    unexpectedly-low event_implied_move values across holiday weeks has
    the context.

Inputs and outputs are scaled to PERCENT of underlying (e.g. 5.0 → 5%).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EventVolDecomposition:
    """Outputs of :func:`decompose_event_vol`.

    All four fields are ``Optional[float]``; any field is ``None`` exactly
    when its required input was unavailable. ``event_implied_move_pct`` is
    clipped to 0 (never negative) when the implied total-variance is below
    the non-event variance under the 1σ-additive convention.
    """

    near_term_implied_sigma_pct: Optional[float]
    non_event_move_pct: Optional[float]
    event_implied_move_pct: Optional[float]
    event_move_share_of_total: Optional[float]


def decompose_event_vol(
    *,
    near_term_implied_move_pct: Optional[float],
    near_term_dte: Optional[int],
    rv_annual_calendar: Optional[float],
) -> EventVolDecomposition:
    """Compute the four event-vol fields from primary inputs.

    Parameters
    ----------
    near_term_implied_move_pct
        Legacy MAD-form straddle premium proxy: ``(call_mid + put_mid)/S × 100``.
        Treated as a percent of underlying (e.g. ``5.5`` for 5.5%).
    near_term_dte
        Calendar days to the near-term expiry (event horizon).  ``None`` or
        non-positive disables the decomposition (returns sigma_pct only).
    rv_annual_calendar
        Annualised σ (decimal, e.g. ``0.30`` for 30%) for the non-event
        baseline.  Production uses HAR-RV (services.realized_vol.har_rv_forecast)
        with an RS-trailing-mean fallback for short histories
        (services.realized_vol.rs_trailing_mean_forecast).  Backtest scripts
        may substitute Yang-Zhang σ directly when their HAR pipeline returns
        None.  ``None`` disables the variance subtraction.

    Returns
    -------
    EventVolDecomposition
        Frozen struct of:
        - ``near_term_implied_sigma_pct``: 1σ percent move over [0, T_years]
          (always populated when ``near_term_implied_move_pct`` is provided).
        - ``non_event_move_pct``: 1σ percent move attributable to non-event
          drift over (T - 1 day) calendar days; ``None`` if ``rv_annual_calendar``
          or ``near_term_dte`` is missing.
        - ``event_implied_move_pct``: 1σ percent move attributable to the
          event over 1 calendar day, clipped to 0 when implied total variance
          < non-event variance.
        - ``event_move_share_of_total``: ``event / sigma`` ratio in [0, 1];
          dimensionally consistent because both are 1σ-form pct moves.
    """
    near_term_implied_sigma_pct: Optional[float] = None
    non_event_move_pct: Optional[float] = None
    event_implied_move_pct: Optional[float] = None
    event_move_share_of_total: Optional[float] = None

    if near_term_implied_move_pct is not None:
        near_term_implied_sigma_pct = float(
            near_term_implied_move_pct * math.sqrt(math.pi / 2.0)
        )

    if (
        near_term_implied_sigma_pct is not None
        and near_term_dte is not None
        and rv_annual_calendar is not None
        and rv_annual_calendar > 0
    ):
        non_event_T_years = max(int(near_term_dte) - 1, 0) / 365.0
        non_event_move_pct = float(
            rv_annual_calendar * math.sqrt(non_event_T_years) * 100.0
        )
        event_variance_pct_sq = max(
            (near_term_implied_sigma_pct ** 2) - (non_event_move_pct ** 2),
            0.0,
        )
        event_implied_move_pct = float(math.sqrt(event_variance_pct_sq))
        if near_term_implied_sigma_pct > 0:
            event_move_share_of_total = float(
                event_implied_move_pct / near_term_implied_sigma_pct
            )

    return EventVolDecomposition(
        near_term_implied_sigma_pct=near_term_implied_sigma_pct,
        non_event_move_pct=non_event_move_pct,
        event_implied_move_pct=event_implied_move_pct,
        event_move_share_of_total=event_move_share_of_total,
    )


__all__ = [
    "EventVolDecomposition",
    "decompose_event_vol",
]
