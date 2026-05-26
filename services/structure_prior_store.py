"""
Structure prior store — durable per-structure win/loss accumulator.

Persists walk-forward prior statistics across process restarts and deployments.
Accumulated observations override the report-based priors in structure_scorecard
once a structure has MIN_OBS_FOR_OVERRIDE (5) real observations.

Storage
-------
~/.options_calculator_pro/priors/structure_priors.json

Schema versions
---------------
v1 (legacy): aggregate-only.  Each structure entry stores only running sums
  and counts; individual observations are not retained.

v2 (current): per-observation list.  Each observation carries an
  observation_date field enabling as_of_date filtering for walk-forward
  backtesting (issue #18).  Aggregate cache fields are still maintained
  alongside the observations list so the no-filter read path stays O(1).

Migration on load: a v1 entry is upgraded by synthesizing one aggregate
  placeholder observation dated at the entry's last_updated timestamp.
  This preserves the no-filter read path exactly and makes the timestamp
  visible for future as_of filtering (Phase 1.2+).

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
import os
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE = Path(
    os.environ.get("OPTIONS_CALCULATOR_PRIORS_PATH", "")
    or Path.home() / ".options_calculator_pro" / "priors" / "structure_priors.json"
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

_SCHEMA_VERSION = 2


class BacktestLeakageError(RuntimeError):
    """Raised when the persistent prior store contains future observations during a backtest.

    If as_of_date is provided to build_structure_scorecards and the store holds
    paper/live observations dated after as_of_date, those observations must not
    influence the backtest result.  This error fires before any such influence
    can occur.  To suppress it, remove the offending observations or use
    source_type='replay' (replay observations are the sanctioned backtest signal).
    """


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


# ── Aggregate helpers ──────────────────────────────────────────────────────────


def _recompute_aggregates(observations: List[Dict]) -> Dict[str, Any]:
    """Recompute all aggregate fields from a filtered observations list."""
    n = len(observations)
    if n == 0:
        return {
            "observation_count": 0,
            "positive_count": 0,
            "sum_return_pct": 0.0,
            "sum_expansion_pct": 0.0,
            "win_rate": 0.50,
            "avg_return_pct": 0.0,
            "avg_realized_expansion_pct": 0.0,
            "rank_score": 0.50,
            "source_types": {},
        }

    wins = sum(1 for o in observations if float(o.get("realized_return_pct", 0.0)) > 0.0)
    sum_ret = sum(float(o.get("realized_return_pct", 0.0)) for o in observations)
    sum_exp = sum(float(o.get("realized_expansion_pct", 0.0)) for o in observations)

    if n < LAPLACE_SMOOTHING_THRESHOLD:
        win_rate = (wins + 1) / (n + 2)
    else:
        win_rate = wins / n

    avg_return = sum_ret / n
    avg_expansion = sum_exp / n
    rank = _compute_rank_score(win_rate=win_rate, avg_return_pct=avg_return, history_count=n)

    src_counts: Dict[str, int] = {}
    for o in observations:
        src = o.get("source_type", "paper")
        src_counts[src] = src_counts.get(src, 0) + 1

    return {
        "observation_count": n,
        "positive_count": wins,
        "sum_return_pct": float(sum_ret),
        "sum_expansion_pct": float(sum_exp),
        "win_rate": float(win_rate),
        "avg_return_pct": float(avg_return),
        "avg_realized_expansion_pct": float(avg_expansion),
        "rank_score": float(rank),
        "source_types": src_counts,
    }


def _migrate_v1_entry(structure: str, entry: Dict) -> Dict:
    """Upgrade a schema_version=1 entry to v2 by synthesizing a placeholder observation.

    The aggregate cache fields are preserved exactly.  A single placeholder
    observation is added dated at the entry's last_updated timestamp so that
    as_of_date queries on or after that date see the historical data.

    On-disk promotion to v2 is intentionally lazy: the file stays at v1 until
    something explicitly calls save().  _load() migrates in memory only — do not
    change this to write back eagerly on load.
    """
    last_updated = entry.get("last_updated")
    if last_updated:
        obs_date_str = str(last_updated)[:10]
    else:
        obs_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    placeholder: Dict[str, Any] = {
        "observation_id": "v1_aggregate_migration",
        "observation_date": obs_date_str,
        "source_type": next(iter(entry.get("source_types", {})), "paper"),
        "realized_return_pct": float(entry.get("avg_return_pct", 0.0)),
        "realized_expansion_pct": float(entry.get("avg_realized_expansion_pct", 0.0)),
        "_migrated": True,
    }

    upgraded = dict(entry)
    upgraded["observations"] = [placeholder]
    upgraded["schema_version"] = _SCHEMA_VERSION
    return upgraded


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
                structures = raw.get("structures", {})
                # Migrate any v1 entries to v2
                migrated = {}
                for s, entry in structures.items():
                    if entry.get("schema_version", 1) < _SCHEMA_VERSION:
                        entry = _migrate_v1_entry(s, entry)
                    if "observations" not in entry:
                        entry["observations"] = []
                    migrated[s] = entry
                self._data = migrated
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
                "schema_version": _SCHEMA_VERSION,
                "structures": self._data,
            }
            tmp_path = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, self._path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        logger.warning("structure_prior_store: failed to remove temp file %s", tmp_path)
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
        observation_date: Optional[date] = None,
        observation_id: Optional[str] = None,
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
        observation_date : date, optional
            The date to associate with this observation.  Used for as_of_date
            filtering in walk-forward backtesting (#18).  Defaults to today.
        observation_id : str, optional
            Stable deduplication key.  When provided, duplicate calls with the
            same id are silently ignored.
        """
        if structure not in SUPPORTED_STRUCTURES:
            logger.warning(
                "structure_prior_store: unrecognised structure %r — skipping update",
                structure,
            )
            return

        obs_date_str = (observation_date or date.today()).isoformat()
        obs_id = str(observation_id) if observation_id is not None else str(uuid.uuid4())

        with _WRITE_LOCK:
            entry = self._data.get(
                structure,
                {
                    "structure": structure,
                    "schema_version": _SCHEMA_VERSION,
                    "observations": [],
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

            # Deduplication: skip if observation_id already recorded
            existing_ids = {o.get("observation_id") for o in entry.get("observations", [])}
            if obs_id in existing_ids:
                logger.debug(
                    "structure_prior_store: skipped duplicate observation_id=%s for %s",
                    obs_id, structure,
                )
                return

            # Append to per-observation list
            observation: Dict[str, Any] = {
                "observation_id": obs_id,
                "observation_date": obs_date_str,
                "source_type": source_type,
                "realized_return_pct": float(realized_return_pct),
                "realized_expansion_pct": float(realized_expansion_pct),
            }
            observations = list(entry.get("observations", []))
            observations.append(observation)

            # Update aggregate cache (O(1) — do not recompute from full list)
            n = entry["observation_count"] + 1
            wins = entry["positive_count"] + (1 if float(realized_return_pct) > 0.0 else 0)
            sum_ret = entry["sum_return_pct"] + float(realized_return_pct)
            sum_exp = entry["sum_expansion_pct"] + float(realized_expansion_pct)

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

            # PR-W: build a fresh dict and atomically swap it in, rather
            # than mutating `entry` in place. Readers of get_prior_dict
            # acquire no lock — they call `self._data.get(structure)` and
            # then read multiple keys sequentially. Under CPython the GIL
            # makes individual key reads atomic, but the GIL can be
            # released between bytecodes. If `entry` is the SAME dict that
            # readers reference, a writer's `entry.update({...})` between
            # two reader `.get()` calls causes the reader's sequential
            # reads to see two different snapshots of the same dict
            # (torn read).
            #
            # The fix is the standard persistent-data-structure pattern:
            # build a NEW dict, assign it to self._data[structure]. The
            # dict assignment is atomic (single STORE_SUBSCR bytecode),
            # and a reader holding the OLD dict reference continues to
            # see the OLD consistent snapshot.
            new_entry = {
                **entry,
                "schema_version": _SCHEMA_VERSION,
                "observations": observations,
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
            self._data[structure] = new_entry
            self._save_locked()

        logger.debug(
            "structure_prior_store: updated %s n=%d win_rate=%.2f avg_ret=%.2f%%",
            structure,
            n,
            win_rate,
            avg_return,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_prior_dict(
        self,
        structure: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return a dict with WalkForwardPrior-compatible keys, or None.

        Parameters
        ----------
        structure : str
        as_of_date : date, optional
            If provided, only observations with observation_date <= as_of_date
            are counted.  If None, uses the aggregate cache (O(1), no filtering).

        Returns None when:
          - no observations have been recorded for this structure, OR
          - observation_count (possibly filtered) < MIN_OBS_FOR_OVERRIDE (5)

        Dict keys: structure, history_count, win_rate, avg_return_pct,
                   rank_score, source
        """
        entry = self._data.get(structure)
        if entry is None:
            return None

        if as_of_date is None:
            # Fast path: use the denormalized aggregate cache
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

        # Filtered path: iterate observations, filter by date, recompute.
        # PR-W: comparison is now date-based (was ISO string lex compare).
        # The lex compare was only safe when every observation_date was in
        # the bare YYYY-MM-DD form; a stray time component would lex-compare
        # as larger than the same-day as_of_str, mis-excluding the row.
        # Parse the first 10 chars to a date object; rows with unparseable
        # dates are conservatively excluded (skip rather than crash).
        def _on_or_before(observation_date_raw: Any) -> bool:
            if not observation_date_raw:
                return False
            try:
                obs_date = date.fromisoformat(str(observation_date_raw)[:10])
            except (TypeError, ValueError):
                return False
            return obs_date <= as_of_date

        as_of_str = as_of_date.isoformat()
        filtered = [
            o for o in entry.get("observations", [])
            if _on_or_before(o.get("observation_date"))
            and not o.get("_migrated", False)  # exclude v1 migration placeholder from filtered path
        ]
        # Also include non-migrated if all observations are from migration, fallback to all
        if not filtered:
            # Try including migrated entries (legacy data has no real dates)
            migrated_filtered = [
                o for o in entry.get("observations", [])
                if _on_or_before(o.get("observation_date"))
            ]
            filtered = migrated_filtered

        agg = _recompute_aggregates(filtered)
        n = agg["observation_count"]
        if n < MIN_OBS_FOR_OVERRIDE:
            return None

        return {
            "structure": structure,
            "history_count": n,
            "win_rate": agg["win_rate"],
            "avg_return_pct": agg["avg_return_pct"],
            "rank_score": agg["rank_score"],
            "source": f"persistent_store:as_of:{as_of_str}",
        }

    def get_all_prior_dicts(
        self,
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return dicts for all structures meeting MIN_OBS_FOR_OVERRIDE.

        Returns an empty dict if none qualify.
        """
        result = {}
        for s in SUPPORTED_STRUCTURES:
            d = self.get_prior_dict(s, as_of_date=as_of_date)
            if d is not None:
                result[s] = d
        return result

    # ── Leakage sentinel ─────────────────────────────────────────────────────

    def check_for_leakage(self, as_of_date: date) -> None:
        """Raise BacktestLeakageError if the store contains future non-replay observations.

        Call before using this store in a backtest context.  Replays are exempt
        because they are the sanctioned backtest signal; paper and live
        observations that are dated after as_of_date represent real trades that
        have not yet happened and must not influence the backtest result.

        Parameters
        ----------
        as_of_date : date
            The backtest evaluation date.  Any observation with
            observation_date > as_of_date and source_type != "replay" triggers
            the error.

        Implementation notes (PR-V)
        ---------------------------
        - Iteration is guarded by ``_WRITE_LOCK`` so a concurrent ``update()``
          can't mutate ``self._data`` mid-scan (which would either skip
          observations or raise ``RuntimeError: dictionary changed size``).
        - ``observation_date`` is parsed into a real ``date`` before
          comparison.  The previous version compared ISO strings
          lexicographically, which is only correct when every observation's
          date is in the bare ``YYYY-MM-DD`` form.  If a single observation
          ever picks up a time component (e.g. ``2026-05-26T14:00:00``),
          lex-compare against ``as_of_str = "2026-05-26"`` would mis-classify
          it as in the future and trigger a spurious leakage error.
          Malformed dates that can't be parsed are skipped (conservative —
          we'd rather miss a malformed-row violation than crash the scan).
        """
        violations: List[str] = []
        with _WRITE_LOCK:
            # Snapshot the structures dict under the lock so we can release
            # it before parsing dates and assembling the error message.
            structures_snapshot = [
                (structure, list(entry.get("observations", []) or []))
                for structure, entry in self._data.items()
            ]
        for structure, observations in structures_snapshot:
            for obs in observations:
                src = obs.get("source_type", "paper")
                if src == "replay":
                    continue
                raw = obs.get("observation_date", "")
                if not raw:
                    continue
                # Accept both YYYY-MM-DD and any datetime-prefixed form
                # ("YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DD HH:MM:SS", etc.) by
                # slicing the first 10 chars before parsing.
                try:
                    obs_date = date.fromisoformat(str(raw)[:10])
                except (TypeError, ValueError):
                    continue
                if obs_date > as_of_date:
                    violations.append(f"{structure}/{src}@{obs_date.isoformat()}")
        if violations:
            raise BacktestLeakageError(
                f"Persistent prior store contains {len(violations)} future non-replay "
                f"observation(s) for as_of_date={as_of_date.isoformat()}.  "
                f"First offenders: {', '.join(violations[:5])}.  "
                "Remove the offending observations or use source_type='replay' to proceed."
            )

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
    as_of_date: Optional[date] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Module-level convenience used by structure_scorecard._load_walk_forward_priors().

    Returns a dict of structure → prior_dict for structures that have
    >= MIN_OBS_FOR_OVERRIDE observations.  Empty dict if none qualify.

    Parameters
    ----------
    as_of_date : date, optional
        If provided, only observations on or before this date are counted.
        Pass snapshot.as_of_date from the backtest evaluation loop.
    """
    return get_structure_prior_store(store_path).get_all_prior_dicts(as_of_date=as_of_date)
