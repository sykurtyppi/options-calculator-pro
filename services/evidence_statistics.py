"""Uncertainty for forward evidence: bootstrap intervals and outlier dependence.

Averages alone cannot say whether a result is signal or noise. Every figure
here is a percentile bootstrap with a FIXED seed, so a report regenerated on
the same rows gives the same intervals. Below MIN_CI_SAMPLE observations no
interval is produced at all: a bootstrap over a handful of trades looks
precise and is not.

Paper/research evidence only; none of this makes a result execution-grade.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

MIN_CI_SAMPLE = 10
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260927
CONFIDENCE = 0.95
OUTLIER_K = 5
_CHUNK = 250  # resamples per batch; bounds memory at large n

METHOD_NOTE = (
    f"{int(CONFIDENCE * 100)}% percentile bootstrap, {BOOTSTRAP_RESAMPLES} resamples, fixed seed; "
    f"no interval below n={MIN_CI_SAMPLE}. An interval that includes 0 means no detectable difference."
)


def _clean(values: Iterable[Any]) -> np.ndarray:
    """Finite numbers only, in canonical (sorted) order.

    Booleans are rejected: True would otherwise count as a +1 return. Sorting
    makes the seeded bootstrap depend only on the multiset of values, not on
    incidental row order, so the same evidence always gets the same verdict.
    """
    out: List[float] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return np.sort(np.asarray(out, dtype=float))


def _bootstrap_means(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_RESAMPLES, _CHUNK):
        stop = min(start + _CHUNK, BOOTSTRAP_RESAMPLES)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[idx].mean(axis=1)
    return means


def _interval(samples: np.ndarray) -> tuple[float, float]:
    tail = (1.0 - CONFIDENCE) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def _verdict(low: Optional[float], high: Optional[float]) -> str:
    if low is None or high is None:
        return "insufficient_sample"
    if low > 0:
        return "above_zero"
    if high < 0:
        return "below_zero"
    return "includes_zero"


def mean_with_ci(values: Iterable[Any]) -> Dict[str, Any]:
    """Mean with a bootstrap interval, or no interval below MIN_CI_SAMPLE."""
    data = _clean(values)
    n = len(data)
    result: Dict[str, Any] = {
        "n": n,
        "mean": float(data.mean()) if n else None,
        "median": float(np.median(data)) if n else None,
        "ci_low": None,
        "ci_high": None,
    }
    if n >= MIN_CI_SAMPLE:
        low, high = _interval(_bootstrap_means(data, np.random.default_rng(BOOTSTRAP_SEED)))
        result.update({"ci_low": low, "ci_high": high})
    result["verdict"] = _verdict(result["ci_low"], result["ci_high"])
    return result


def outlier_dependence(values: Iterable[Any], k: int = OUTLIER_K) -> Dict[str, Any]:
    """How much the mean leans on its k best (and k worst) observations.

    ``fragile`` is True when dropping the k best results turns a positive mean
    non-positive: the headline then describes a few events, not a strategy.
    ``top_k_share_of_profit`` is the share of total gains contributed by the k
    largest gains.
    """
    data = np.sort(_clean(values))
    n = len(data)
    if n <= k:
        return {"k": k, "n": n, "mean_without_top_k": None, "mean_without_bottom_k": None,
                "top_k_share_of_profit": None, "fragile": None}
    mean = float(data.mean())
    without_top = float(data[:-k].mean())
    without_bottom = float(data[k:].mean())
    gains = data[data > 0]
    total_gain = float(gains.sum())
    top_share = float(np.sort(gains)[-k:].sum() / total_gain) if total_gain > 0 else None
    return {
        "k": k,
        "n": n,
        "mean_without_top_k": without_top,
        "mean_without_bottom_k": without_bottom,
        "top_k_share_of_profit": top_share,
        "fragile": bool(mean > 0 and without_top <= 0),
    }


def summarize_returns(values: Iterable[Any]) -> Dict[str, Any]:
    data = list(_clean(values))
    return {**mean_with_ci(data), "outlier_dependence": outlier_dependence(data)}


def paired_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    """Mean of (left - right) over keys present in both, with a bootstrap CI.

    Pairing on the same event removes the event's own noise (the market move
    both positions saw), which is what makes selector-vs-baseline comparable
    at small n.
    """
    keys = sorted(set(left) & set(right))
    diffs = [float(left[key]) - float(right[key]) for key in keys
             if _clean([left[key], right[key]]).size == 2]
    return {**mean_with_ci(diffs), "pairs": len(diffs)}


def two_sample_difference(first: Sequence[Any], second: Sequence[Any]) -> Dict[str, Any]:
    """mean(first) - mean(second) for independent groups, resampled separately."""
    a, b = _clean(first), _clean(second)
    result: Dict[str, Any] = {
        "n_first": len(a),
        "n_second": len(b),
        "mean": float(a.mean() - b.mean()) if len(a) and len(b) else None,
        "ci_low": None,
        "ci_high": None,
    }
    if len(a) >= MIN_CI_SAMPLE and len(b) >= MIN_CI_SAMPLE:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        diffs = _bootstrap_means(a, rng) - _bootstrap_means(b, rng)
        low, high = _interval(diffs)
        result.update({"ci_low": low, "ci_high": high})
    result["verdict"] = _verdict(result["ci_low"], result["ci_high"])
    return result
