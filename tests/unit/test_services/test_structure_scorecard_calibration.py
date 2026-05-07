"""
Calibration regression-contract test for P-5c scorecard thresholds.

Pins the four (lower, upper) tuples chosen for the ratio-based scoring
sites in services.structure_scorecard against the post-P-5c corpus
percentiles. If anyone retunes a bound without re-running the corpus
(or re-runs the corpus and forgets to re-anchor), this test trips and
forces an explicit decision.

Corpus reference
----------------
calibration_basis = "post_p5c_v2_real_corpus_743_events_2024_2025_h1"
- 743 deduped (symbol, event_date) events
- window: 2024-01-01 → 2025-06-30
- denominator: event_implied_move_pct (post-P-5a 1σ form, earnings-excluded σ_HAR)

Empirical percentiles (run dir: reports/p5c_real_corpus_v2/iv_expansion_study_20260505T010758Z/):
    historical_vs_implied_move_ratio:
        p10=0.123, p25=0.227, p50=0.521, p75=0.874, p90=1.143, p95=1.291
    tail_vs_implied_move_ratio:
        p10=0.239, p25=0.418, p50=0.829, p75=1.218, p90=1.534, p95=1.797

Thresholds chosen per scoring site (rounded to 2 decimals):
    score_atm_straddle / hist           (p10, p90)  = (0.12, 1.14)
    score_atm_straddle / tail           (p10, p90)  = (0.24, 1.53)
    score_otm_strangle / tail           (p25, p90)  = (0.42, 1.53)
    score_put_calendar / tail penalty   (p75, p95)  = (1.22, 1.80)
"""
from __future__ import annotations

import math

from services.structure_scorecard import _score_high_good


CALIBRATION_BASIS = "post_p5c_v2_real_corpus_743_events_2024_2025_h1"


# Each row pins one threshold tuple. The *_basis fields name the corpus
# percentile each bound was derived from, plus the empirical value, so a
# future reader can trace any number back to the source corpus.
P5C_THRESHOLDS = [
    {
        "name": "atm_straddle.historical_vs_implied_move_ratio",
        "lower": 0.12,
        "upper": 1.14,
        "lower_basis": ("p10", 0.123),
        "upper_basis": ("p90", 1.143),
    },
    {
        "name": "atm_straddle.tail_vs_implied_move_ratio",
        "lower": 0.24,
        "upper": 1.53,
        "lower_basis": ("p10", 0.239),
        "upper_basis": ("p90", 1.534),
    },
    {
        "name": "otm_strangle.tail_vs_implied_move_ratio",
        "lower": 0.42,
        "upper": 1.53,
        "lower_basis": ("p25", 0.418),
        "upper_basis": ("p90", 1.534),
    },
    {
        "name": "put_calendar.tail_risk_penalty",
        "lower": 1.22,
        "upper": 1.80,
        "lower_basis": ("p75", 1.218),
        "upper_basis": ("p95", 1.797),
    },
]


def test_score_high_good_returns_zero_at_lower_bound():
    """Each P-5c lower bound must map to a 0.0 score (no contribution below floor)."""
    for t in P5C_THRESHOLDS:
        score = _score_high_good(t["lower"], t["lower"], t["upper"])
        assert score == 0.0, f"{t['name']} at lower={t['lower']}: score={score}"


def test_score_high_good_returns_one_at_upper_bound():
    """Each P-5c upper bound must map to a 1.0 score (full contribution at ceiling)."""
    for t in P5C_THRESHOLDS:
        score = _score_high_good(t["upper"], t["lower"], t["upper"])
        assert score == 1.0, f"{t['name']} at upper={t['upper']}: score={score}"


def test_score_high_good_midpoint_is_approximately_half():
    """The linear midpoint between lower and upper must score ~0.5."""
    for t in P5C_THRESHOLDS:
        mid = 0.5 * (t["lower"] + t["upper"])
        score = _score_high_good(mid, t["lower"], t["upper"])
        assert math.isclose(score, 0.5, abs_tol=1e-9), (
            f"{t['name']} at mid={mid}: score={score}"
        )


def test_thresholds_match_corpus_percentiles_within_rounding():
    """
    Each chosen bound must equal its corpus percentile to within 2-decimal
    rounding (matches the precision we emit in scorecard calibration-basis
    comments). Trips if a bound drifts away from its documented percentile
    source without an accompanying corpus re-run.
    """
    for t in P5C_THRESHOLDS:
        lower_label, lower_corpus = t["lower_basis"]
        upper_label, upper_corpus = t["upper_basis"]
        assert math.isclose(t["lower"], round(lower_corpus, 2), abs_tol=0.005), (
            f"{t['name']}: lower {t['lower']} vs corpus {lower_label}={lower_corpus:.3f}"
        )
        assert math.isclose(t["upper"], round(upper_corpus, 2), abs_tol=0.005), (
            f"{t['name']}: upper {t['upper']} vs corpus {upper_label}={upper_corpus:.3f}"
        )
