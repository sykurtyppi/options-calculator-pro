"""Move-statistic unit conversions — the single source of truth (Codex F1).

Why this module exists
----------------------
The engine compares an *implied* event move against *realized* historical
moves in several places, and the two are different statistics of the same
distribution:

* ``event_implied_move_pct`` (services.event_vol_decomposition) is a **1σ**
  move — it has to be, because variance subtraction only works in σ-space.
* Historical earnings moves are realized **absolute** moves ``|Δ|/S``. Their
  mean, median and P90 are all *different multiples of σ*.

For X ~ N(0, σ²):

    E|X|        = σ·√(2/π)      ≈ 0.7979·σ   (mean absolute move)
    median|X|   = σ·Φ⁻¹(0.75)   ≈ 0.6745·σ
    P90(|X|)    = σ·Φ⁻¹(0.95)   ≈ 1.6449·σ

So a fairly priced event — implied σ equal to realized σ — did NOT produce a
ratio of 1.0 anywhere in the engine. ``anchor/σ`` sat at ≈0.755 and
``p90/σ`` at ≈1.645, and every rationale that said "above 1 favors …" was
reading an unlabelled scale. The scorecard bounds had been *calibrated on the
corpus in those biased units*, which hid the problem inside the scores while
leaving the displayed ratios, the rationale text, and every hand-set literal
threshold (``_event_risk_score``, the UI's ``<= 1.0`` / ``> 1.3`` gates)
structurally wrong.

These helpers restate both sides on a common basis — expected absolute move
(E|move|) for anchor comparisons, P90 absolute move for tail comparisons — so
that **1.0 means fairly priced** everywhere. The conversion is a *constant*
(shape factors × the anchor blend weight; no data dependence), which has two
consequences worth stating plainly:

1. A corpus-calibrated percentile bound rescales EXACTLY: p_k(new) =
   p_k(old) / scale. Regenerating the 743-event corpus in corrected units
   reproduces the analytic rescale to ≤2.2e-16 (see the PR that introduced
   this module). Scores built from those bounds are therefore byte-identical.
2. Hand-set literal thresholds do NOT rescale — that is the behaviour change,
   and it is the intended one.

This lives under ``services/`` (not ``web/api/``) because the snapshot layer
must use it and must not import upward from the web layer.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = [
    "SIGMA_TO_EXPECTED_ABS_MOVE",
    "SIGMA_TO_P90_ABS_MOVE",
    "MEDIAN_TO_MEAN_ABS_MOVE",
    "MOVE_ANCHOR_AVG_LAST4_WEIGHT",
    "ANCHOR_RATIO_UNIT_SCALE",
    "TAIL_RATIO_UNIT_SCALE",
    "MOVE_RATIO_UNITS_VERSION",
    "anchor_blend_to_expected_abs_scale",
    "sigma_to_expected_abs_move",
    "sigma_to_p90_abs_move",
    "anchor_to_expected_abs_move",
    "historical_vs_implied_ratio",
    "tail_vs_implied_ratio",
]

# ── Normal shape factors ──────────────────────────────────────────────────
SIGMA_TO_EXPECTED_ABS_MOVE: float = math.sqrt(2.0 / math.pi)              # E|X|/σ   ≈ 0.797885
SIGMA_TO_P90_ABS_MOVE: float = 1.6448536269514722                         # Φ⁻¹(0.95) ≈ 1.644854
MEDIAN_TO_MEAN_ABS_MOVE: float = 0.6744897501960817 / SIGMA_TO_EXPECTED_ABS_MOVE  # ≈ 0.845348

# ── Anchor blend weight (single definition; edge_constants mirrors it) ────
# ``_compute_move_anchor`` = w·mean|last 4| + (1-w)·median|all|. Recency bias
# is intentional; the 65/35 magnitude is a documented assumption
# (web/api/edge_constants._HEURISTIC_THRESHOLDS["move_anchor_avg_last4_weight"]).
MOVE_ANCHOR_AVG_LAST4_WEIGHT: float = 0.65


def anchor_blend_to_expected_abs_scale(weight: float = MOVE_ANCHOR_AVG_LAST4_WEIGHT) -> float:
    """Scale of the blended anchor relative to a true E|move|.

    The mean term estimates E|X|; the median term estimates only ≈0.845·E|X|,
    so the blend sits at ``w + (1-w)·0.845`` of E|move| — ≈0.9459 at w=0.65.
    """
    w = float(weight)
    return float(w + (1.0 - w) * MEDIAN_TO_MEAN_ABS_MOVE)


# ── Legacy-ratio → corrected-ratio scales (constants) ─────────────────────
# legacy anchor ratio  = anchor / σ                 = corrected × ANCHOR_RATIO_UNIT_SCALE
# legacy tail ratio    = p90    / σ                 = corrected × TAIL_RATIO_UNIT_SCALE
# where corrected ratios are 1.0 for a fairly priced event.
ANCHOR_RATIO_UNIT_SCALE: float = SIGMA_TO_EXPECTED_ABS_MOVE * anchor_blend_to_expected_abs_scale()  # ≈ 0.754696
TAIL_RATIO_UNIT_SCALE: float = SIGMA_TO_P90_ABS_MOVE                                                  # ≈ 1.644854

# Persisted evidence rows (services.outcome_recorder) record which scale their
# stored ratio uses. NULL/absent = legacy (fair ≈ 0.755); 2 = corrected (fair = 1.0).
MOVE_RATIO_UNITS_VERSION: int = 2


def _finite(value: Optional[float]) -> Optional[float]:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def sigma_to_expected_abs_move(sigma_pct: Optional[float]) -> Optional[float]:
    """1σ implied move → expected absolute move E|move| (same units in/out)."""
    s = _finite(sigma_pct)
    return None if s is None else float(s * SIGMA_TO_EXPECTED_ABS_MOVE)


def sigma_to_p90_abs_move(sigma_pct: Optional[float]) -> Optional[float]:
    """1σ implied move → 90th-percentile absolute move."""
    s = _finite(sigma_pct)
    return None if s is None else float(s * SIGMA_TO_P90_ABS_MOVE)


def anchor_to_expected_abs_move(
    anchor_pct: Optional[float], weight: float = MOVE_ANCHOR_AVG_LAST4_WEIGHT
) -> Optional[float]:
    """Blended historical anchor → E|move| basis (divide out the median shrink)."""
    a = _finite(anchor_pct)
    return None if a is None else float(a / anchor_blend_to_expected_abs_scale(weight))


def historical_vs_implied_ratio(
    anchor_pct: Optional[float],
    event_implied_sigma_pct: Optional[float],
    *,
    weight: float = MOVE_ANCHOR_AVG_LAST4_WEIGHT,
) -> Optional[float]:
    """Historical anchor ÷ implied, both on the E|move| basis. 1.0 = fairly priced."""
    a = anchor_to_expected_abs_move(anchor_pct, weight)
    i = sigma_to_expected_abs_move(event_implied_sigma_pct)
    if a is None or i is None or i <= 0:
        return None
    return float(a / i)


def tail_vs_implied_ratio(
    p90_move_pct: Optional[float], event_implied_sigma_pct: Optional[float]
) -> Optional[float]:
    """Historical P90 ÷ implied P90. 1.0 = fairly priced tail."""
    p = _finite(p90_move_pct)
    i = sigma_to_p90_abs_move(event_implied_sigma_pct)
    if p is None or i is None or i <= 0:
        return None
    return float(p / i)
