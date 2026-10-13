"""Screener-row fixtures for the premarket alert tests.

The alert recomputes each row's score from its own inputs and rejects a row
whose ranking_score disagrees, so fixtures must be internally consistent:
``alert_row`` picks the IV/RV and median move that produce the requested
score (as nearly as the other inputs allow) and records the score those
inputs actually give.
"""
from __future__ import annotations

import math
from typing import Any, Dict

from services.screener_service import ranking_score_components

BASE: Dict[str, Any] = {
    "release_timing": "AMC", "iv30": 0.35, "avg_spread_pct": 3.0, "term_structure_ratio": 1.0,
    "sample_size": 8, "earnings_source_confirmed": "sec_edgar", "earnings_source_stale": False,
    "status": "ranked",
}


def _score(row: Dict[str, Any]) -> float:
    return ranking_score_components(
        iv_rv_ratio=row.get("iv_rv_ratio"), ts_ratio=row.get("term_structure_ratio"),
        median_earnings_move_pct=row.get("median_earnings_move_pct"), sample_size=int(row.get("sample_size") or 0),
        dte=row.get("days_to_earnings"), spread_pct=row.get("avg_spread_pct"),
        iv_rv_har=row.get("iv_rv_har"), rv_percentile_rank=row.get("rv_percentile_rank"),
    )["score"]


def alert_row(symbol: str, score: float, dte: int = 7, **extra: Any) -> Dict[str, Any]:
    row = {"symbol": symbol, "days_to_earnings": dte, **BASE, **extra}
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        row.setdefault("iv_rv_ratio", 1.10)
        row.setdefault("median_earnings_move_pct", 6.0)
        row["ranking_score"] = score
        return row
    free_iv_rv = "iv_rv_ratio" not in extra
    free_move = "median_earnings_move_pct" not in extra

    def at(t: float) -> Dict[str, Any]:
        candidate = dict(row)
        if free_iv_rv:
            candidate["iv_rv_ratio"] = round(1.6 - 0.8 * t, 6)
        if free_move:
            candidate["median_earnings_move_pct"] = round(2.0 + 10.0 * t, 6)
        return candidate

    try:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if _score(at(mid)) < score:
                lo = mid
            else:
                hi = mid
        row = at(hi)
        row["ranking_score"] = round(_score(row), 4)
    except (TypeError, ValueError):
        # Deliberately malformed inputs (a test of the alert's own gates)
        # cannot be scored; keep the requested score as given.
        row = at(0.5)
        row["ranking_score"] = score
    return row
