#!/usr/bin/env python3
"""
analyze_replay_quartiles.py
===========================
Quartile-bucket the multi-structure replay observations by setup_score and
report mean return / expansion / win rate per bucket.

The point: the scorecard's composite_structure_score is the candidate
ordering signal. If sorting events by score produces monotonic differences
in realized outcomes (top quartile > bottom quartile), the scorecard has
predictive juice. If the buckets all look the same, score is noise.

Usage::

    # Auto-find the most recent multi-structure summary:
    python scripts/analyze_replay_quartiles.py

    # Specific summary:
    python scripts/analyze_replay_quartiles.py --summary path/to/replay_multi_structure_*.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_latest_summary() -> Optional[Path]:
    candidates = sorted(
        REPO_ROOT.glob("tmp/replay_multi_*/replay_multi_structure_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _quartile_bounds(values: List[float]) -> List[float]:
    """Return the 25/50/75 percentile bounds (no min/max)."""
    if len(values) < 4:
        return []
    sorted_v = sorted(values)
    return [
        sorted_v[len(sorted_v) // 4],
        sorted_v[len(sorted_v) // 2],
        sorted_v[(3 * len(sorted_v)) // 4],
    ]


def _bucket_for(value: float, bounds: List[float]) -> int:
    for i, b in enumerate(bounds):
        if value <= b:
            return i
    return len(bounds)


def _analyze_selector(events: list, structures: list) -> None:
    """Per-structure: what did selector pick vs pass, and how did each fare?"""
    from collections import Counter
    picks = Counter()
    pick_returns_by_pick: Dict[str, list] = {}
    for ev in events:
        pick = ev.get("selector_pick")
        picks[pick] += 1
        if pick and pick != "None" and pick in structures:
            a = ev.get("attempts", {}).get(pick, {})
            if a.get("status") == "ok":
                pick_returns_by_pick.setdefault(pick, []).append(a["realized_return_pct"])

    print("── SELECTOR DECISIONS ──")
    print(f"  {'pick':<22} {'n':>4}   mean realized return on its own picks")
    for pick, n in picks.most_common():
        rets = pick_returns_by_pick.get(pick, [])
        if rets:
            mean = sum(rets) / len(rets)
            wins = sum(1 for r in rets if r > 0)
            print(f"  {str(pick):<22} {n:>4}   {mean:+7.2f}%   wins {wins}/{len(rets)} = {100*wins/len(rets):.0f}%")
        else:
            print(f"  {str(pick):<22} {n:>4}   (no-trade)")
    print()

    # For each structure, contrast its outcome when selector PICKED any
    # trade vs PASSED. Tests whether selector's no-trade discipline has
    # predictive value.
    print("  Picked-vs-passed by structure (does the selector's gate sort signal?):")
    for structure in structures:
        picked, passed = [], []
        for ev in events:
            a = ev.get("attempts", {}).get(structure, {})
            if a.get("status") != "ok":
                continue
            pick = ev.get("selector_pick")
            if pick is None or pick == "None":
                passed.append(a["realized_return_pct"])
            else:
                picked.append(a["realized_return_pct"])
        if not (picked or passed):
            continue
        print(f"  {structure}:")
        if picked:
            wins = sum(1 for r in picked if r > 0)
            print(f"    selector PICKED  n={len(picked):2d}   mean {statistics.mean(picked):+7.2f}%   "
                  f"median {statistics.median(picked):+7.2f}%   wins {wins}/{len(picked)} = {100*wins/len(picked):.0f}%")
        if passed:
            wins = sum(1 for r in passed if r > 0)
            print(f"    selector PASSED  n={len(passed):2d}   mean {statistics.mean(passed):+7.2f}%   "
                  f"median {statistics.median(passed):+7.2f}%   wins {wins}/{len(passed)} = {100*wins/len(passed):.0f}%")
    print()


def analyze(summary: dict) -> None:
    structures = summary.get("structures", [])
    events = summary.get("events", [])

    print()
    print("=" * 78)
    print(f"  QUARTILE ANALYSIS — {summary.get('timestamp')}")
    print(f"  symbols: {','.join(summary.get('symbols', []))}")
    print(f"  events:  {summary.get('events_total')}")
    print("=" * 78)
    print()

    _analyze_selector(events, structures)

    for structure in structures:
        attempts = []
        for ev in events:
            a = ev.get("attempts", {}).get(structure, {})
            if a.get("status") != "ok":
                continue
            attempts.append({
                "symbol": ev["symbol"],
                "event_date": ev["event_date"],
                "score": a["setup_score"],
                "return_pct": a["realized_return_pct"],
                "expansion_pct": a["realized_expansion_pct"],
            })

        if len(attempts) < 8:
            print(f"── {structure}: only {len(attempts)} observations — skipping quartile split")
            print()
            continue

        scores = [a["score"] for a in attempts]
        bounds = _quartile_bounds(scores)
        buckets: Dict[int, List[dict]] = {0: [], 1: [], 2: [], 3: []}
        for a in attempts:
            buckets[_bucket_for(a["score"], bounds)].append(a)

        print(f"── {structure} — n={len(attempts)} — score range [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"   quartile bounds (25/50/75): {[f'{b:.3f}' for b in bounds]}")
        print(f"   {'bucket':<8} {'n':>4} {'score range':<18} "
              f"{'mean return':>13} {'median return':>15} {'mean exp':>11} {'win rate':>10}")
        for q in range(4):
            obs = buckets[q]
            if not obs:
                print(f"   Q{q+1:<7} {0:>4} (empty)")
                continue
            scs = [a["score"] for a in obs]
            rets = [a["return_pct"] for a in obs]
            exps = [a["expansion_pct"] for a in obs]
            wins = sum(1 for r in rets if r > 0)
            print(f"   Q{q+1:<7} {len(obs):>4} "
                  f"[{min(scs):.3f}, {max(scs):.3f}]   "
                  f"{statistics.mean(rets):>+10.2f}%  "
                  f"{statistics.median(rets):>+12.2f}%  "
                  f"{statistics.mean(exps):>+8.2f}%  "
                  f"{wins}/{len(obs)} = {100*wins/len(obs):>3.0f}%")
        # Monotonicity check: do bucket means trend?
        bucket_means = [
            statistics.mean([a["return_pct"] for a in buckets[q]]) if buckets[q] else None
            for q in range(4)
        ]
        non_null = [m for m in bucket_means if m is not None]
        if len(non_null) >= 3:
            strict_inc = all(a < b for a, b in zip(non_null, non_null[1:]))
            strict_dec = all(a > b for a, b in zip(non_null, non_null[1:]))
            print(f"   trend: ", end="")
            if strict_inc:
                print("monotonically INCREASING (high score → better return) ✓")
            elif strict_dec:
                print("monotonically DECREASING (high score → worse return) ✗")
            else:
                print("non-monotonic")
            print(f"   Q4-Q1 spread: {non_null[-1] - non_null[0]:+.2f}%")
        print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=None,
                        help="Path to a replay_multi_structure_*.json (default: latest in tmp/)")
    args = parser.parse_args(argv)

    summary_path = args.summary or _find_latest_summary()
    if summary_path is None or not summary_path.exists():
        print("ERROR: no replay summary found.", file=sys.stderr)
        return 2

    print(f"Reading: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    analyze(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
