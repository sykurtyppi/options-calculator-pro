"""
Edge-engine tunable constants and the heuristic-threshold registry.

Extracted verbatim from web/api/edge_engine.py (Phase 3.1) — values and
comments are unchanged. edge_engine re-exports every name defined here, so
existing `from web.api.edge_engine import <CONST>` paths keep working.
"""
from __future__ import annotations

from typing import Any, Dict


# ─── Constants (unchanged) ────────────────────────────────────────────────────

MIN_EARNINGS_EVENTS_FOR_FULL_SIGNAL = 8
HARD_NO_TRADE_CONFIDENCE_CAP_PCT = 55.0
TS_SLOPE_TARGET = -0.004
TS_SLOPE_BAND = 0.025
# Mirrors services/structure_scorecard.ABSOLUTE_SPREAD_THRESHOLD_PCT (12.0).
# De Silva, Smith & So (2025, Review of Finance): at 12% spread, round-trip cost
# ≈ 24% of option value, at or above the ceiling of typical pre-earnings IV
# expansion. Keeping this aligned with the scorecard threshold prevents the
# contradictory state where the selector returns NO_TRADE on ineligibility
# while the edge-engine hard gate does not fire.
MAX_NEAR_TERM_SPREAD_PCT_FOR_TRADE = 12.0
MIN_SHORT_LEG_DTE = 2
MIN_NEAR_BACK_IV_RATIO_FOR_EVENT = 1.02
MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_TRADE = 400.0
MOVE_UNCERTAINTY_Z_SCORE = 1.28
IV_RV_CROWDING_WARNING_THRESHOLD = 1.60  # above this, IV may already price the event (soft warning only)
MIN_TS_SLOPE_FOR_EXPANSION = 0.0005
MIN_RV_PERCENTILE_FOR_EXPANSION = 60.0
MAX_NEAR_TERM_SPREAD_PCT_FOR_EXPANSION = 12.0
MIN_NEAR_TERM_LIQUIDITY_PROXY_FOR_EXPANSION = 600.0
MIN_PRICEABLE_EXPANSION_TRADES = 3
EXPANSION_ENTRY_OFFSETS = tuple(range(5, 11))
EXPANSION_EXIT_OFFSET = 1
EXPANSION_BACK_EXPIRY_GAP_DAYS = 14

# ─── Stale-data enforcement ───────────────────────────────────────────────────
# IV is live (from option chain); RV is computed from yfinance price bars.
# If the price bars are stale, IV/RV is not contemporaneous — this violates a
# core assumption of the edge model.  Cap confidence rather than hard-reject
# so signal information is preserved but the output is clearly degraded.
# Cap = 40%: well below the 62% "Consider" floor, signals clear data quality issue.
STALE_DATA_THRESHOLD_DAYS = 2           # bars older than N calendar days → stale
STALE_DATA_CONFIDENCE_CAP_PCT = 40.0   # hard ceiling when IV/RV non-contemporaneous

# ─── Soft gates replacing hard-gates 9 & 10 ──────────────────────────────────
# Principle: degraded evidence → degraded confidence, NOT hard rejection.
# A ticker with 4 earnings events and a consistent crush pattern still has
# information.  Capping at 55% keeps it off "Consider" (requires ≥62%) while
# exposing the signal for watchlist/monitoring.  Low-event-count cap is looser
# (60%) because the history is thin but at least real earnings history.
# If both flags are active the stricter cap wins (fallback takes elif precedence).
FALLBACK_MODEL_CONFIDENCE_CAP_PCT = 55.0   # move_source != "earnings_history"
LOW_EVENT_COUNT_CONFIDENCE_CAP_PCT = 60.0  # 1–7 earnings events (real but thin)

# ─── Fix 7: Heuristic threshold registry ─────────────────────────────────────
# Every scoring threshold that is NOT empirically derived is tagged here so
# the system can be audited.  Values are intentionally NOT changed — they are
# working priors that require a calibration pass on the backtest corpus to
# justify or revise.  Label: assumption=True means "prior belief, unvalidated."
_HEURISTIC_THRESHOLDS: Dict[str, Any] = {
    # ── Move anchor blend ─────────────────────────────────────────────────────
    "move_anchor_avg_last4_weight": {
        "value": 0.65,
        "assumption": True,
        "rationale": "Weight on avg(last 4 earnings moves) vs median. "
                     "Recency bias intentional but magnitude (65/35) is subjective.",
    },
    # ── Ticker tier breakpoints ───────────────────────────────────────────────
    "ticker_tier_mega_cap_usd": {
        "value": 200e9,
        "assumption": True,
        "rationale": "Market cap cutoff for mega-cap tier (1.00x multiplier). "
                     "No empirical derivation; standard institutional convention.",
    },
    # ── Kurtosis penalty breakpoints ──────────────────────────────────────────
    "kurtosis_penalty_ek_threshold_mild": {
        "value": 1.0,
        "assumption": True,
        "rationale": "Excess kurtosis above which mild penalty begins. "
                     "Based on normal distribution EK=0 plus 1σ buffer. Not calibrated.",
    },
    "kurtosis_penalty_ek_threshold_severe": {
        "value": 3.0,
        "assumption": True,
        "rationale": "Excess kurtosis above which severe penalty applies. "
                     "Matches leptokurtic threshold; not empirically derived.",
    },
    # ── Crush calibration rate thresholds ─────────────────────────────────────
    "crush_rate_boost_threshold": {
        "value": 0.72,
        "assumption": True,
        "rationale": "Historical crush rate above which confidence is boosted. "
                     "Represents 'stock stays inside IV ~3 out of 4 times'. Not calibrated.",
    },
    "crush_rate_penalise_threshold": {
        "value": 0.35,
        "assumption": True,
        "rationale": "Historical crush rate below which strong penalty applied. "
                     "Represents 'stock blows through IV majority of the time'. Not calibrated.",
    },
    # ── Transaction cost heuristic ────────────────────────────────────────────
    "tx_cost_base_pct": {
        "value": 0.18,
        "assumption": True,
        "rationale": "Base friction estimate (18bp). "
                     "Represents ~retail broker commission + minimal slippage on liquid names. "
                     "Does not account for notional size or multi-leg fills.",
    },
    # ── Calendar scenario fallbacks ────────────────────────────────────────────
    "calendar_scenario_fallback_expand": {
        "value": 1.20,
        "assumption": True,
        "rationale": "IV back-leg expansion multiplier when no historical data. "
                     "+20% represents a moderate IV spike scenario. Arbitrary.",
    },
    "calendar_scenario_fallback_crush_mild": {
        "value": 0.75,
        "assumption": True,
        "rationale": "Back-leg IV after mild crush (−25%). "
                     "Based on typical observed post-earnings IV crush range. Not calibrated.",
    },
    "calendar_scenario_fallback_crush_severe": {
        "value": 0.55,
        "assumption": True,
        "rationale": "Back-leg IV after severe crush (−45%). "
                     "Represents extreme crush seen in mega-cap earnings. Not calibrated.",
    },
    # ── Ticker tier market-cap cutoffs ────────────────────────────────────────
    "ticker_tier_large_cap_usd": {
        "value": 20e9,
        "assumption": True,
        "rationale": "Market cap cutoff for large-cap tier (20B). Standard convention; not empirically calibrated.",
    },
    "ticker_tier_mid_cap_usd": {
        "value": 2e9,
        "assumption": True,
        "rationale": "Market cap cutoff for mid-cap tier (2B). Standard convention; not empirically calibrated.",
    },
    "ticker_tier_small_cap_usd": {
        "value": 300e6,
        "assumption": True,
        "rationale": "Market cap cutoff for small-cap tier (300M). Below this = micro-cap.",
    },
    # ── Ticker tier confidence multipliers ────────────────────────────────────
    "ticker_tier_large_cap_mult": {
        "value": 0.95,
        "assumption": True,
        "rationale": "Large-cap confidence multiplier (5% haircut vs mega-cap). Not empirically derived.",
    },
    "ticker_tier_mid_cap_mult": {
        "value": 0.85,
        "assumption": True,
        "rationale": "Mid-cap confidence multiplier (15% haircut). Larger discount for less-covered names.",
    },
    "ticker_tier_small_cap_mult": {
        "value": 0.72,
        "assumption": True,
        "rationale": "Small-cap confidence multiplier (28% haircut). IV dynamics more erratic.",
    },
    "ticker_tier_micro_cap_mult": {
        "value": 0.55,
        "assumption": True,
        "rationale": "Micro-cap confidence multiplier (45% haircut). Binary event risk; thin option markets.",
    },
    # ── Kurtosis penalty shape parameters ────────────────────────────────────
    "kurtosis_penalty_slope_mild": {
        "value": 0.11,
        "assumption": True,
        "rationale": "Confidence multiplier decrease per unit excess kurtosis above mild threshold. "
                     "~11pp reduction per EK unit. Not calibrated.",
    },
    "kurtosis_penalty_slope_severe": {
        "value": 0.046,
        "assumption": True,
        "rationale": "Confidence multiplier decrease per unit EK above severe threshold. "
                     "Flatter slope because confidence floor is already low. Not calibrated.",
    },
    "kurtosis_penalty_mild_floor": {
        "value": 0.78,
        "assumption": True,
        "rationale": "Minimum multiplier in the mild regime (EK 1-3). "
                     "Floor prevents over-penalising moderately fat-tailed names.",
    },
    "kurtosis_penalty_severe_floor": {
        "value": 0.55,
        "assumption": True,
        "rationale": "Absolute minimum kurtosis confidence multiplier (EK ≥3). "
                     "45% haircut for extreme tail-risk names.",
    },
    # ── Crush calibration rate shape parameters ───────────────────────────────
    "crush_rate_boost_coeff": {
        "value": 0.55,
        "assumption": True,
        "rationale": "Sensitivity of confidence boost per unit crush rate above boost threshold. Not calibrated.",
    },
    "crush_rate_penalty_coeff_moderate": {
        "value": 1.13,
        "assumption": True,
        "rationale": "Sensitivity of confidence penalty for crush rates 35-50%. Not calibrated.",
    },
    "crush_rate_penalty_coeff_severe": {
        "value": 1.0,
        "assumption": True,
        "rationale": "Sensitivity of confidence penalty for crush rates below 35%. Not calibrated.",
    },
}
