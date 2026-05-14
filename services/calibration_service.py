"""
IV expansion calibration service.

Maps a pre-earnings long-vega setup_score ∈ [0, 1] to an expected IV expansion
summary based on walk-forward observations collected during live use.

Design
------
This service is intentionally phase-aware so sparse data is not presented as a
smooth empirical calibration:

* Phase 1 (N < 40): ``bootstrap_prior``.
  Research prior used for ordering only. Not an empirical measurement.
* Phase 2 (40 ≤ N < 120): ``observational``.
  Uses raw bucket-level observations where available, otherwise falls back to
  the prior. Still not a fitted curve.
* Phase 3 (120 ≤ N < 250): ``fitted_moderate``.
  Isotonic fit allowed, but still labelled as moderate empirical coverage.
* Phase 4 (N ≥ 250): ``fitted_high``.
  Same isotonic family with materially stronger sample depth.

Persistence
-----------
Observations are stored in a JSON file at
``~/.options_calculator_pro/calibration/iv_expansion.json``.
The file is loaded on first use and flushed on every ``update()`` call.

Thread safety
-------------
``update()`` and ``save()`` are protected by a module-level ``threading.Lock``.
Concurrent ``apply()`` calls are read-only and do not acquire the lock.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import threading
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Persistence path ──────────────────────────────────────────────────────────

_DEFAULT_STORE = pathlib.Path.home() / ".options_calculator_pro" / "calibration" / "iv_expansion.json"

# ── Bootstrap prior ───────────────────────────────────────────────────────────
# Research-grounded estimates for the pre-earnings long-vega strategy.
# Source: mean IV expansion (T-entry to T-1) by setup-score decile,
# derived from back-testing 2018-2024 liquid names (internal research).
# Each tuple: (score_low_inclusive, score_high_exclusive, expected_expansion_pct)
_BOOTSTRAP_BINS: List[Tuple[float, float, float]] = [
    (0.0, 0.2, 1.5),   # weak setup — little excess IV expansion expected
    (0.2, 0.4, 4.0),   # below-average
    (0.4, 0.6, 7.5),   # average
    (0.6, 0.8, 12.0),  # above-average
    (0.8, 1.0, 18.0),  # strong setup — large historical moves + cheap IV
]

# Phase thresholds
_MIN_OBS_FOR_OBSERVATIONAL = 40
_MIN_OBS_FOR_FIT = 120
_MIN_OBS_FOR_HIGH_FIT = 250
_KNOWN_SOURCE_TYPES = {"replay", "synthetic", "paper", "live"}

# Score bins used for the summary table (10 equal-width bins in [0, 1])
_SUMMARY_BIN_EDGES = [i / 10.0 for i in range(11)]  # 0.0, 0.1, …, 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bootstrap_estimate(score: float) -> float:
    """Return the bootstrap-prior expected IV expansion for *score*."""
    for lo, hi, est in _BOOTSTRAP_BINS:
        if lo <= score < hi:
            return est
    # score == 1.0 falls through the half-open top bin
    return _BOOTSTRAP_BINS[-1][2]


def _isotonic_fit(
    scores: List[float],
    expansions: List[float],
) -> Tuple[List[float], List[float]]:
    """
    Fit an isotonic regression on (scores, expansions) using the pool-adjacent
    violators (PAV) algorithm.

    Returns two parallel lists (fitted_scores, fitted_values) in ascending
    score order, suitable for linear interpolation.
    """
    if len(scores) != len(expansions) or len(scores) == 0:
        return [], []

    # Sort by score
    pairs = sorted(zip(scores, expansions), key=lambda p: p[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    # PAV algorithm — isotonic non-decreasing regression
    # Blocks store (mean, count, lo_idx, hi_idx)
    blocks: List[List] = [[ys[i], 1, i, i] for i in range(len(ys))]

    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            # Merge blocks i and i+1
            merged_count = blocks[i][1] + blocks[i + 1][1]
            merged_mean = (
                blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            ) / merged_count
            blocks[i] = [merged_mean, merged_count, blocks[i][2], blocks[i + 1][3]]
            blocks.pop(i + 1)
            # Back-check
            if i > 0:
                i -= 1
        else:
            i += 1

    # Expand blocks back to per-point fitted values
    fitted_ys = [0.0] * len(ys)
    for block in blocks:
        mean_val, _, lo, hi = block
        for j in range(lo, hi + 1):
            fitted_ys[j] = mean_val

    # Deduplicate by keeping one point per distinct fitted value transition
    result_xs: List[float] = []
    result_ys: List[float] = []
    prev = None
    for x, y in zip(xs, fitted_ys):
        if y != prev:
            result_xs.append(x)
            result_ys.append(y)
            prev = y
    # Always include the last point for complete range coverage
    if result_xs and result_xs[-1] != xs[-1]:
        result_xs.append(xs[-1])
        result_ys.append(fitted_ys[-1])

    return result_xs, result_ys


def _interp(score: float, xs: List[float], ys: List[float]) -> float:
    """Linear interpolation / extrapolation on a sorted (xs, ys) curve."""
    if not xs:
        return 0.0
    if score <= xs[0]:
        return ys[0]
    if score >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= score <= xs[i + 1]:
            t = (score - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


# ── Main class ────────────────────────────────────────────────────────────────

class IVExpansionCalibration:
    """
    Calibrates the setup_score → expected IV expansion mapping.

    Parameters
    ----------
    store_path : pathlib.Path, optional
        Where to persist observations.  Defaults to
        ``~/.options_calculator_pro/calibration/iv_expansion.json``.

    Usage
    -----
    >>> cal = IVExpansionCalibration()
    >>> result = cal.apply(0.72)
    >>> cal.update(0.72, observed_expansion_pct=9.3)
    >>> cal.save()
    """

    def __init__(self, store_path: pathlib.Path = _DEFAULT_STORE) -> None:
        self._path = store_path
        self._lock = threading.Lock()
        # Parallel lists of (score, observed_expansion_pct) observations
        self._scores: List[float] = []
        self._expansions: List[float] = []
        self._sources: List[str] = []
        self._timestamps: List[str] = []  # ISO date strings, parallel to _scores (#18)
        self._observation_ids: Set[str] = set()
        # Cached isotonic curve (rebuilt lazily after each update, no-filter path only)
        self._fitted_xs: List[float] = []
        self._fitted_ys: List[float] = []
        self._curve_dirty = True
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load persisted observations from disk (silent no-op if missing)."""
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text())
                scores = raw.get("scores", [])
                expansions = raw.get("expansions", [])
                sources = raw.get("sources", [])
                timestamps = raw.get("timestamps", [])
                observation_ids = raw.get("observation_ids", [])
                if len(scores) == len(expansions):
                    self._scores = [float(s) for s in scores]
                    self._expansions = [float(e) for e in expansions]
                    if len(sources) == len(scores):
                        self._sources = [self._normalize_source_type(src) for src in sources]
                    else:
                        self._sources = ["paper"] * len(self._scores)
                    # Migration: if no timestamps stored, synthesize from file mtime
                    if len(timestamps) == len(scores):
                        self._timestamps = [str(t) for t in timestamps]
                    else:
                        try:
                            fallback_date = pathlib.Path(self._path).stat().st_mtime
                            import datetime as _dt
                            fallback_str = _dt.datetime.fromtimestamp(fallback_date).strftime("%Y-%m-%d")
                        except Exception:
                            fallback_str = "1970-01-01"
                        self._timestamps = [fallback_str] * len(self._scores)
                    self._observation_ids = {
                        str(obs_id)
                        for obs_id in observation_ids
                        if obs_id not in (None, "")
                    }
                    self._curve_dirty = True
                    logger.info(
                        "calibration: loaded %d observations from %s",
                        len(self._scores),
                        self._path,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("calibration: could not load store (%s) — starting fresh", exc)

    def save(self) -> None:
        """Flush current observations to disk."""
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """Save (must be called with self._lock held)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 2,
                "scores": self._scores,
                "expansions": self._expansions,
                "sources": self._sources,
                "timestamps": self._timestamps,
                "observation_ids": sorted(self._observation_ids),
                "n": len(self._scores),
            }
            self._path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.error("calibration: save failed (%s)", exc)

    # ── Curve management ─────────────────────────────────────────────────────

    def _rebuild_curve(self) -> None:
        """Refit isotonic regression from current observations."""
        if len(self._scores) >= _MIN_OBS_FOR_FIT:
            self._fitted_xs, self._fitted_ys = _isotonic_fit(
                self._scores, self._expansions
            )
        else:
            self._fitted_xs, self._fitted_ys = [], []
        self._curve_dirty = False

    def _n(self) -> int:
        return len(self._scores)

    def _is_fitted(self) -> bool:
        return self._n() >= _MIN_OBS_FOR_FIT

    def _phase(self) -> str:
        n = self._n()
        if n >= _MIN_OBS_FOR_HIGH_FIT:
            return "fitted_high"
        if n >= _MIN_OBS_FOR_FIT:
            return "fitted_moderate"
        if n >= _MIN_OBS_FOR_OBSERVATIONAL:
            return "observational"
        return "bootstrap_prior"

    def _bucket_bounds(self, score: float) -> Tuple[float, float]:
        clipped = float(np.clip(score, 0.0, 1.0))
        lo = math.floor(clipped * 10.0) / 10.0
        hi = min(lo + 0.10, 1.0)
        if clipped >= 1.0:
            lo = 0.9
            hi = 1.0
        return lo, hi

    # ── as_of_date filtered path ─────────────────────────────────────────────

    def _apply_as_of(self, setup_score: float, as_of_date: date) -> Dict[str, Any]:
        """apply() with as_of_date filtering.  Does not touch the production cache."""
        as_of_str = as_of_date.isoformat()
        mask = [t <= as_of_str for t in self._timestamps]
        filtered_scores = [s for s, m in zip(self._scores, mask) if m]
        filtered_expansions = [e for e, m in zip(self._expansions, mask) if m]

        n = len(filtered_scores)
        score = float(np.clip(setup_score, 0.0, 1.0))

        # Determine phase from filtered count
        if n >= _MIN_OBS_FOR_HIGH_FIT:
            phase = "fitted_high"
        elif n >= _MIN_OBS_FOR_FIT:
            phase = "fitted_moderate"
        elif n >= _MIN_OBS_FOR_OBSERVATIONAL:
            phase = "observational"
        else:
            phase = "bootstrap_prior"

        if phase == "bootstrap_prior":
            est = _bootstrap_estimate(score)
            width = max(2.0, est * 0.35)
            return {
                "expected_expansion_pct": round(est, 2),
                "low_pct": round(max(0.0, est - width), 2),
                "high_pct": round(est + width, 2),
                "n_observations": n,
                "prior_only": True,
                "phase": phase,
                "score_input": round(score, 4),
                "note": (
                    f"Research prior only ({n}/{_MIN_OBS_FOR_OBSERVATIONAL} observations "
                    f"as of {as_of_str}). Use for ordering only."
                ),
            }

        if phase == "observational":
            lo, hi = self._bucket_bounds(score)
            local_obs = [
                e for s, e in zip(filtered_scores, filtered_expansions)
                if lo <= s < hi or (hi >= 1.0 and lo <= s <= hi)
            ]
            est = float(np.mean(local_obs)) if len(local_obs) >= 3 else _bootstrap_estimate(score)
            if len(local_obs) >= 3:
                std = float(np.std(local_obs, ddof=1)) if len(local_obs) > 1 else max(1.0, abs(est) * 0.20)
                prior_only = False
            else:
                std = max(2.0, abs(est) * 0.35)
                prior_only = True
            margin = max(std, 1.0)
            return {
                "expected_expansion_pct": round(est, 2),
                "low_pct": round(max(0.0, est - margin), 2),
                "high_pct": round(est + margin, 2),
                "n_observations": n,
                "n_local": len(local_obs),
                "prior_only": prior_only,
                "phase": phase,
                "score_input": round(score, 4),
                "note": f"Observational (as_of={as_of_str}, total N={n}).",
            }

        # fitted phase — compute isotonic curve on filtered subset in-place (no cache impact)
        fitted_xs, fitted_ys = _isotonic_fit(filtered_scores, filtered_expansions)
        est = _interp(score, fitted_xs, fitted_ys)

        lo_band = max(0.0, score - 0.10)
        hi_band = min(1.0, score + 0.10)
        local_obs = [
            e for s, e in zip(filtered_scores, filtered_expansions)
            if lo_band <= s <= hi_band
        ]
        margin = float(np.std(local_obs, ddof=1)) if len(local_obs) >= 3 else est * 0.30

        return {
            "expected_expansion_pct": round(est, 2),
            "low_pct": round(max(0.0, est - margin), 2),
            "high_pct": round(est + margin, 2),
            "n_observations": n,
            "n_local": len(local_obs),
            "prior_only": False,
            "phase": phase,
            "score_input": round(score, 4),
            "note": f"Fitted (as_of={as_of_str}, total N={n}, local N={len(local_obs)}).",
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def _normalize_source_type(self, source_type: Optional[str]) -> str:
        source = str(source_type or "paper").strip().lower()
        return source if source in _KNOWN_SOURCE_TYPES else "paper"

    def update(
        self,
        setup_score: float,
        observed_expansion_pct: float,
        *,
        observation_id: Optional[str] = None,
        source_type: str = "paper",
        observation_date: Optional[date] = None,
    ) -> bool:
        """
        Record a new (score, expansion) observation and invalidate the curve cache.

        Parameters
        ----------
        setup_score : float
            The ranking_score from screener_service.compute_ranking_score() at entry.
        observed_expansion_pct : float
            Realised IV expansion from entry to exit (T-entry → T-1 before event).
            Positive = IV rose, negative = IV fell.
        observation_id : str, optional
            Stable deduplication key. When provided, duplicate calls are ignored.
        source_type : str, default "paper"
            Observation provenance. Accepted values are replay, synthetic, paper, live.
        observation_date : date, optional
            The date to associate with this observation for as_of_date filtering
            in walk-forward backtesting (#18).  Defaults to today.
        """
        normalized_source = self._normalize_source_type(source_type)
        normalized_id = str(observation_id) if observation_id not in (None, "") else None
        obs_date_str = (observation_date or date.today()).isoformat()
        with self._lock:
            if normalized_id is not None and normalized_id in self._observation_ids:
                logger.debug(
                    "calibration: skipped duplicate observation_id=%s", normalized_id
                )
                return False
            self._scores.append(float(setup_score))
            self._expansions.append(float(observed_expansion_pct))
            self._sources.append(normalized_source)
            self._timestamps.append(obs_date_str)
            if normalized_id is not None:
                self._observation_ids.add(normalized_id)
            self._curve_dirty = True
            self._save_locked()
        logger.debug(
            "calibration: update score=%.3f expansion=%.2f%% source=%s obs_id=%s (N=%d total)",
            setup_score,
            observed_expansion_pct,
            normalized_source,
            normalized_id,
            self._n(),
        )
        return True

    def apply(
        self,
        setup_score: float,
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Return calibration estimate for *setup_score*.

        Parameters
        ----------
        setup_score : float
        as_of_date : date, optional
            If provided, only observations with observation_date <= as_of_date
            are used to build the calibration curve.  The production cache
            (no as_of_date) is not touched.  Pass snapshot.as_of_date from
            the backtest evaluation loop.  (#18)

        Response dict keys
        ------------------
        expected_expansion_pct : float
            Point estimate of IV expansion (%).
        low_pct / high_pct : float
            ~68% interval (±1σ empirical). Derived from the empirical
            ±1-sigma band when fitted; from bootstrap bin width when on prior.
            This is NOT an 80% interval — the margin equals one empirical
            standard deviation, which corresponds to ~68% coverage under a
            normal assumption.
        n_observations : int
            Number of real observations backing the estimate.
        prior_only : bool
            True when using the bootstrap prior (N < 30).
        phase : str
            ``"bootstrap_prior"``, ``"observational"``, ``"fitted_moderate"``,
            or ``"fitted_high"``.
        score_input : float
            Echo of the input score for traceability.
        """
        if as_of_date is not None:
            return self._apply_as_of(setup_score, as_of_date)

        if self._curve_dirty:
            self._rebuild_curve()

        n = self._n()
        score = float(np.clip(setup_score, 0.0, 1.0))

        phase = self._phase()

        if phase == "bootstrap_prior":
            est = _bootstrap_estimate(score)
            width = max(2.0, est * 0.35)
            return {
                "expected_expansion_pct": round(est, 2),
                "low_pct": round(max(0.0, est - width), 2),
                "high_pct": round(est + width, 2),
                "n_observations": n,
                "prior_only": True,
                "phase": phase,
                "score_input": round(score, 4),
                "note": (
                    f"Research prior only ({n}/{_MIN_OBS_FOR_OBSERVATIONAL} observations). "
                    "Use this for ordering and context, not as an empirical estimate."
                ),
            }

        if phase == "observational":
            lo, hi = self._bucket_bounds(score)
            local_obs = [
                e
                for s, e in zip(self._scores, self._expansions)
                if lo <= s < hi or (hi >= 1.0 and lo <= s <= hi)
            ]
            est = float(np.mean(local_obs)) if len(local_obs) >= 3 else _bootstrap_estimate(score)
            if len(local_obs) >= 3:
                std = float(np.std(local_obs, ddof=1)) if len(local_obs) > 1 else max(1.0, abs(est) * 0.20)
                note = (
                    f"Observational phase: raw bucket evidence only (bucket N={len(local_obs)}, total N={n}). "
                    "No fitted curve is used yet."
                )
                prior_only = False
            else:
                std = max(2.0, abs(est) * 0.35)
                note = (
                    f"Observational phase with sparse local evidence (bucket N={len(local_obs)}, total N={n}). "
                    "Bucket estimate falls back to the research prior."
                )
                prior_only = True
            margin = max(std, 1.0)
            return {
                "expected_expansion_pct": round(est, 2),
                "low_pct": round(max(0.0, est - margin), 2),
                "high_pct": round(est + margin, 2),
                "n_observations": n,
                "n_local": len(local_obs),
                "prior_only": prior_only,
                "phase": phase,
                "score_input": round(score, 4),
                "note": note,
            }

        est = _interp(score, self._fitted_xs, self._fitted_ys)

        # Empirical standard deviation in a ±0.10 band around the score
        lo_band = max(0.0, score - 0.10)
        hi_band = min(1.0, score + 0.10)
        local_obs = [
            e
            for s, e in zip(self._scores, self._expansions)
            if lo_band <= s <= hi_band
        ]
        if len(local_obs) >= 3:
            std = float(np.std(local_obs, ddof=1))
            # ±1σ empirical band (~68% coverage under normality)
            margin = std
        else:
            margin = est * 0.30  # fallback: ±30% of estimate

        return {
            "expected_expansion_pct": round(est, 2),
            "low_pct": round(max(0.0, est - margin), 2),
            "high_pct": round(est + margin, 2),
            "n_observations": n,
            "n_local": len(local_obs),
            "prior_only": False,
            "phase": phase,
            "score_input": round(score, 4),
            "note": (
                f"{'High' if phase == 'fitted_high' else 'Moderate'}-coverage fitted phase "
                f"(total N={n}, local N={len(local_obs)})."
            ),
        }

    def get_curve_summary(self) -> List[Dict[str, Any]]:
        """
        Return a score-bucketed summary table suitable for charting.

        Each row covers a 0.10-wide score bin and contains the mean,
        standard deviation, and observation count for that bin.  If the
        model is in bootstrap_prior phase, rows are populated from the
        prior (flagged accordingly).

        Returns
        -------
        list of dict with keys:
            score_mid, score_lo, score_hi,
            expected_expansion_pct, std_pct, n, prior_only
        """
        if self._curve_dirty:
            self._rebuild_curve()

        rows = []
        bin_edges = _SUMMARY_BIN_EDGES
        for i in range(len(bin_edges) - 1):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            mid = (lo + hi) / 2.0

            phase = self._phase()

            if phase in {"fitted_moderate", "fitted_high"}:
                obs = [
                    e
                    for s, e in zip(self._scores, self._expansions)
                    if lo <= s < hi
                ]
                n_bin = len(obs)
                if n_bin >= 2:
                    mean_exp = float(np.mean(obs))
                    std_exp = float(np.std(obs, ddof=1))
                else:
                    # Interpolate from curve for bins with sparse coverage
                    mean_exp = _interp(mid, self._fitted_xs, self._fitted_ys)
                    std_exp = mean_exp * 0.30
                    n_bin = n_bin  # could be 0 or 1
                rows.append(
                    {
                        "score_lo": lo,
                        "score_hi": hi,
                        "score_mid": mid,
                        "expected_expansion_pct": round(mean_exp, 2),
                        "std_pct": round(std_exp, 2),
                        "n": n_bin,
                        "prior_only": False,
                    }
                )
            elif phase == "observational":
                obs = [
                    e
                    for s, e in zip(self._scores, self._expansions)
                    if lo <= s < hi or (hi >= 1.0 and lo <= s <= hi)
                ]
                if len(obs) >= 2:
                    mean_exp = float(np.mean(obs))
                    std_exp = float(np.std(obs, ddof=1)) if len(obs) > 1 else max(1.0, abs(mean_exp) * 0.20)
                    prior_only = False
                    n_bin = len(obs)
                else:
                    mean_exp = _bootstrap_estimate(mid)
                    std_exp = max(2.0, mean_exp * 0.35)
                    prior_only = True
                    n_bin = len(obs)
                rows.append(
                    {
                        "score_lo": lo,
                        "score_hi": hi,
                        "score_mid": mid,
                        "expected_expansion_pct": round(mean_exp, 2),
                        "std_pct": round(std_exp, 2),
                        "n": n_bin,
                        "prior_only": prior_only,
                    }
                )
            else:
                est = _bootstrap_estimate(mid)
                width = max(2.0, est * 0.35)
                rows.append(
                    {
                        "score_lo": lo,
                        "score_hi": hi,
                        "score_mid": mid,
                        "expected_expansion_pct": round(est, 2),
                        "std_pct": round(width, 2),
                        "n": 0,
                        "prior_only": True,
                    }
                )

        return rows

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def diagnostics(self) -> Dict[str, Any]:
        """Return health summary for logging / monitoring."""
        n = self._n()
        source_counts = Counter(self._sources)
        return {
            "n_observations": n,
            "phase": self._phase(),
            "is_prior_only": self._phase() == "bootstrap_prior",
            "min_for_observational": _MIN_OBS_FOR_OBSERVATIONAL,
            "min_for_fit": _MIN_OBS_FOR_FIT,
            "min_for_high_fit": _MIN_OBS_FOR_HIGH_FIT,
            "store_path": str(self._path),
            "store_exists": self._path.exists(),
            "mean_expansion": round(float(np.mean(self._expansions)), 2) if n else None,
            "std_expansion": round(float(np.std(self._expansions, ddof=1)), 2) if n > 1 else None,
            "score_distribution": {
                "min": round(float(np.min(self._scores)), 4) if n else None,
                "max": round(float(np.max(self._scores)), 4) if n else None,
                "mean": round(float(np.mean(self._scores)), 4) if n else None,
            },
            "expansion_distribution": {
                "min": round(float(np.min(self._expansions)), 2) if n else None,
                "max": round(float(np.max(self._expansions)), 2) if n else None,
                "mean": round(float(np.mean(self._expansions)), 2) if n else None,
            },
            "n_replay": int(source_counts.get("replay", 0)),
            "n_synthetic": int(source_counts.get("synthetic", 0)),
            "n_paper": int(source_counts.get("paper", 0)),
            "n_live": int(source_counts.get("live", 0)),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_calibration: Optional[IVExpansionCalibration] = None
_singleton_lock = threading.Lock()


def get_calibration(store_path: Optional[pathlib.Path] = None) -> IVExpansionCalibration:
    """
    Return the process-level calibration singleton.

    First call initialises the instance (and loads persisted data).
    Subsequent calls return the same object.
    """
    global _calibration
    if _calibration is None:
        with _singleton_lock:
            if _calibration is None:
                _calibration = IVExpansionCalibration(
                    store_path=store_path or _DEFAULT_STORE
                )
    return _calibration
