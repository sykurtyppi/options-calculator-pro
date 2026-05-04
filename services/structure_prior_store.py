"""
Structure prior store — durable per-structure win/loss accumulator.

Persists walk-forward prior statistics across process restarts and deployments.
Accumulated observations override the report-based priors in structure_scorecard
once a structure has MIN_OBS_FOR_OVERRIDE (5) real observations.

Storage
-------
~/.options_calculator_pro/priors/structure_priors.json

Design choices
--------------
Win rate uses Laplace-1 smoothing for N < LAPLACE_SMOOTHING_THRESHOLD (10).
  Formula: (wins + 1) / (N + 2)
  Reason: avoids 0%/100% with tiny samples.  For N >= 10, raw win rate is used.
  This is the only smoothing applied — it is explicit, documented, and intentional.

avg_return_pct and avg_realized_expansion_pct are straight running means.
  No shrinkage or EWMA — the history_count field drives natural shrinkage via
  _blend_expected_return() in structure_scorecard, which weights the prior
  proportionally to min(history_count / 40, 1) × 0.55.

rank_score replicates the formula in structure_scorecard._compute_rank_score():
  0.45 × return_score + 0.30 × win_score + 0.25 × history_score
  (score helpers are inlined here to avoid a circular import).

Circular import note
--------------------
This module does NOT import from services.structure_scorecard.
It returns plain dicts.  structure_scorecard imports from here and
builds WalkForwardPrior objects from those dicts.

Provenance
----------
Each structure entry tracks source_types: {"replay": N, "paper": N, "live": N}
so callers can audit how observations were accumulated.

MIN_OBS_FOR_OVERRIDE = 5
Below this count, get_prior() / get_all_priors() return nothing — the caller
(structure_scorecard._load_walk_forward_priors) falls back to report-based priors.
The threshold is intentionally small so the store starts contributing quickly.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE = (
    Path.home() / ".options_calculator_pro" / "priors" / "structure_priors.json"
)
_WRITE_LOCK = threading.Lock()

# Supported structure names.  Kept as a literal here (not imported from
# structure_scorecard) to break the circular dependency.
SUPPORTED_STRUCTURES: tuple = (
    "atm_straddle",
    "otm_strangle",
    "call_calendar",
    "put_calendar",
)

# The persistent store only overrides report-based priors once a structure
# has this many real observations.  Below this, the report prior governs.
MIN_OBS_FOR_OVERRIDE = 5

# Win rate: apply Laplace-1 smoothing for N < this threshold.
# For N >= this, use raw win rate.
LAPLACE_SMOOTHING_THRESHOLD = 10


# ── Rank score (replicated from structure_scorecard to avoid circular import) ──


def _score_high_good(value: Optional[float], low: float, high: float) -> float:
    if value is None or high <= low:
        return 0.50
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.50
    if high == low:
        return 0.50
    return max(0.0, min(1.0, (v - low) / (high - low)))


def _compute_rank_score(
    win_rate: Optional[float],
    avg_return_pct: Optional[float],
    history_count: int,
) -> float:
    """
    Replicate structure_scorecard._compute_rank_score() without importing it.
    Formula: 0.45 × return_score + 0.30 × win_score + 0.25 × history_score
    """
    return_score = _score_high_good(avg_return_pct, -10.0, 15.0)
    win_score = _score_high_good(win_rate, 0.35, 0.70)
    history_score = _score_high_good(float(history_count), 5.0, 50.0)
    return max(0.0, min(1.0, 0.45 * return_score + 0.30 * win_score + 0.25 * history_score))


# ── StructurePriorStore ────────────────────────────────────────────────────────


class StructurePriorStore:
    """
    Durable per-structure win/loss accumulator.

    Parameters
    ----------
    store_path : Path, optional
        JSON file path.
    """

    def __init__(self, store_path: Path = _DEFAULT_STORE) -> None:
        self._path = store_path
        self._data: Dict[str, Dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text())
                self._data = raw.get("structures", {})
                n = sum(
                    e.get("observation_count", 0) for e in self._data.values()
                )
                logger.info(
                    "structure_prior_store: loaded %d structure entries, %d total observations from %s",
                    len(self._data),
                    n,
                    self._path,
                )
        except Exception as exc:
            logger.warning(
                "structure_prior_store: load failed (%s) — starting fresh", exc
            )
            self._data = {}

    def save(self) -> None:
        """Flush to disk (thread-safe, acquires write lock)."""
        with _WRITE_LOCK:
            self._save_locked()

    def _save_locked(self) -> None:
        """Save — must be called with _WRITE_LOCK held."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "structures": self._data,
            }
            self._path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.error("structure_prior_store: save failed (%s)", exc)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        structure: str,
        realized_return_pct: float,
        realized_expansion_pct: float,
        source_type: str = "paper",
    ) -> None:
        """
        Record one finalized outcome observation for *structure*.

        Parameters
        ----------
        structure : str
            Must be one of SUPPORTED_STRUCTURES.
        realized_return_pct : float
            Net return after costs, in % (9.0 = 9%).
            Positive = win for the win-rate counter.
        realized_expansion_pct : float
            Gross option value change, in %.
        source_type : str
            "replay", "paper", or "live".  Tracked for provenance auditing.
        """
        if structure not in SUPPORTED_STRUCTURES:
            logger.warning(
                "structure_prior_store: unrecognised structure %r — skipping update",
                structure,
            )
            return

        with _WRITE_LOCK:
            entry = self._data.get(
                structure,
                {
                    "structure": structure,
                    "observation_count": 0,
                    "positive_count": 0,
                    "sum_return_pct": 0.0,
                    "sum_expansion_pct": 0.0,
                    "win_rate": 0.50,
                    "avg_return_pct": 0.0,
                    "avg_realized_expansion_pct": 0.0,
                    "rank_score": 0.50,
                    "source_types": {},
                    "last_updated": None,
                },
            )

            n = entry["observation_count"] + 1
            wins = entry["positive_count"] + (1 if float(realized_return_pct) > 0.0 else 0)
            sum_ret = entry["sum_return_pct"] + float(realized_return_pct)
            sum_exp = entry["sum_expansion_pct"] + float(realized_expansion_pct)

            # Win rate: Laplace-1 smoothing for small samples.
            # For N < LAPLACE_SMOOTHING_THRESHOLD: (wins + 1) / (N + 2)
            # For N >= LAPLACE_SMOOTHING_THRESHOLD: raw win rate
            if n < LAPLACE_SMOOTHING_THRESHOLD:
                win_rate = (wins + 1) / (n + 2)
            else:
                win_rate = wins / n

            avg_return = sum_ret / n
            avg_expansion = sum_exp / n
            rank = _compute_rank_score(
                win_rate=win_rate,
                avg_return_pct=avg_return,
                history_count=n,
            )

            src_counts = dict(entry.get("source_types", {}))
            src_counts[source_type] = src_counts.get(source_type, 0) + 1

            entry.update(
                {
                    "observation_count": n,
                    "positive_count": wins,
                    "sum_return_pct": float(sum_ret),
                    "sum_expansion_pct": float(sum_exp),
                    "win_rate": float(win_rate),
                    "avg_return_pct": float(avg_return),
                    "avg_realized_expansion_pct": float(avg_expansion),
                    "rank_score": float(rank),
                    "source_types": src_counts,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._data[structure] = entry
            self._save_locked()

        logger.debug(
            "structure_prior_store: updated %s n=%d win_rate=%.2f avg_ret=%.2f%%",
            structure,
            n,
            win_rate,
            avg_return,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_prior_dict(self, structure: str) -> Optional[Dict[str, Any]]:
        """
        Return a dict with WalkForwardPrior-compatible keys, or None.

        Returns None when:
          - no observations have been recorded for this structure, OR
          - observation_count < MIN_OBS_FOR_OVERRIDE (5)

        In both cases, the caller should fall back to the report-based prior.

        Dict keys: structure, history_count, win_rate, avg_return_pct,
                   rank_score, source
        """
        entry = self._data.get(structure)
        if entry is None:
            return None
        n = entry.get("observation_count", 0)
        if n < MIN_OBS_FOR_OVERRIDE:
            return None
        last = (entry.get("last_updated") or "unknown")[:10]
        return {
            "structure": structure,
            "history_count": n,
            "win_rate": float(entry["win_rate"]),
            "avg_return_pct": float(entry["avg_return_pct"]),
            "rank_score": float(entry["rank_score"]),
            "source": f"persistent_store:{last}",
        }

    def get_all_prior_dicts(self) -> Dict[str, Dict[str, Any]]:
        """
        Return dicts for all structures meeting MIN_OBS_FOR_OVERRIDE.

        Returns an empty dict if none qualify.
        """
        result = {}
        for s in SUPPORTED_STRUCTURES:
            d = self.get_prior_dict(s)
            if d is not None:
                result[s] = d
        return result

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "structures": {
                s: {
                    "observation_count": self._data.get(s, {}).get(
                        "observation_count", 0
                    ),
                    "win_rate": self._data.get(s, {}).get("win_rate"),
                    "avg_return_pct": self._data.get(s, {}).get("avg_return_pct"),
                    "avg_realized_expansion_pct": self._data.get(s, {}).get(
                        "avg_realized_expansion_pct"
                    ),
                    "rank_score": self._data.get(s, {}).get("rank_score"),
                    "source_types": self._data.get(s, {}).get("source_types", {}),
                    "last_updated": self._data.get(s, {}).get("last_updated"),
                    "overrides_report_prior": self._data.get(s, {}).get(
                        "observation_count", 0
                    )
                    >= MIN_OBS_FOR_OVERRIDE,
                }
                for s in SUPPORTED_STRUCTURES
            },
            "min_obs_for_override": MIN_OBS_FOR_OVERRIDE,
            "laplace_smoothing_threshold": LAPLACE_SMOOTHING_THRESHOLD,
            "store_path": str(self._path),
            "store_exists": self._path.exists(),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[StructurePriorStore] = None
_singleton_lock = threading.Lock()


def get_structure_prior_store(
    store_path: Optional[Path] = None,
) -> StructurePriorStore:
    """Return the process-level singleton StructurePriorStore."""
    global _store
    if _store is None:
        with _singleton_lock:
            if _store is None:
                _store = StructurePriorStore(
                    store_path=store_path or _DEFAULT_STORE
                )
    return _store


def load_all_structure_priors(
    store_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Module-level convenience used by structure_scorecard._load_walk_forward_priors().

    Returns a dict of structure → prior_dict for structures that have
    >= MIN_OBS_FOR_OVERRIDE observations.  Empty dict if none qualify.
    """
    return get_structure_prior_store(store_path).get_all_prior_dicts()
