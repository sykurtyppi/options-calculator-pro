# PR #71 Design Note — Straddle/Strangle Prior Loaders

**Status**: Draft for review (no code yet).
**Audit ref**: Calculations audit P1 #2 (multi-counting in
`structure_scorecard._load_straddle_prior_from_reports` and the
scoreboard-fallback path of `_load_strangle_prior_from_reports`).
**Author**: Claude with verification by Codex.

---

## Problem (one paragraph)

`services/structure_scorecard.py:825` (`_load_straddle_prior_from_reports`)
reads `pre_earnings_otm_strangle_scoreboard.csv`, filters by
`(structure, execution_scenario)`, then weights `win_rate` by
`realized_count`. The scoreboard is the **cartesian product** of
`(structure × entry_offset × exit_offset × oi_threshold ×
spread_cap × execution_scenario)`. The same physical trade appears
in multiple rows because looser OI/spread filters inherit trades
from tighter ones, so the weighted average over-weights the loose
filters and produces a contaminated prior. The same code shape
exists as the SCOREBOARD FALLBACK on `_load_strangle_prior_from_reports`
(line 885+); its PRIMARY path is already clean (reads
`baseline_cost_table` from the summary JSON, one canonical row).

## Verified contamination magnitude

Measured against the latest report set at
`exports/reports/pre_earnings_otm_strangle_20260417T180430Z/`:

| | scoreboard path (today) | raw trade log, baseline filter |
|---|---|---|
| **atm_straddle, cross_25** | 60 rows, weighted win_rate **0.5392** | 65 physical trades, win_rate **0.5077** |
| **otm_strangle_3pct, cross_25** | 60 rows, realized_count sum 1291 | 42 physical trades, win_rate **0.6905** |

The straddle prior is inflated by ~3.1 percentage points purely
from the multi-counting bias. The strangle scoreboard fallback
shows the same shape but is only reached when the summary JSON is
missing — in practice rare.

`realized_count` sum on the contaminated path is 2409 for
straddle vs the clean baseline physical-trade count of 65 —
**aggregate count inflation of ~37×**. The inflation factor is
NOT uniform per trade (OI/spread filter passage varies by trade,
so a trade that only passes the loose-filter rows contributes
fewer scoreboard rows than one that passes every filter
combination); the ~37× is the average across the whole subset.
Net effect is the same regardless: loose-filter rows dominate the
weighted-by-`realized_count` average.

## Two approaches considered

### Approach A — canonical filter row from scoreboard

Pin a single `(entry_offset, exit_offset, oi_threshold,
spread_cap)` combo and filter the scoreboard down to it. Mirrors
the existing `_load_calendar_prior_from_reports` contract (which
reads a single-row baseline).

| Pros | Cons |
|---|---|
| Smallest diff. | Requires choosing the "right" combo. Implicitly enshrines whatever values look good today. |
| Matches the calendar-loader pattern. | Scoreboard schema may quietly change one day and break the row picker silently. |
| No new file reads. | The scoreboard's `win_rate` is post-aggregation — if the underlying aggregation logic in `backtest_pre_earnings_otm_strangle.py` ever changes, the loader inherits the change without a test signal. |

### Approach B — aggregate from raw trade log *(recommended)*

Read `pre_earnings_otm_strangle_trade_log.csv` directly. The
trade log has one row per physical trade (12,900 rows across the
full backtest) with per-trade boolean filter columns
(`entry_pass_oi_100`, `entry_pass_spread_10`, etc.). Filter
explicitly to the baseline configuration and aggregate at the
trade level.

| Pros | Cons |
|---|---|
| Each physical trade counted exactly once **by construction**. | Larger diff — new code path replaces the scoreboard read. |
| The win_rate calculation lives in the loader, not buried in the backtest's aggregation step. Visible & testable. | Backtest's trade-log schema must stay stable. The columns we need (`structure`, `entry_offset_bdays`, `exit_offset_bdays`, `entry_pass_oi_100`, `entry_pass_spread_10`, `priceable_realized`, `pnl_cross_25`) all exist today and are documented inside the backtest script. |
| Strangle's primary path (summary JSON `baseline_cost_table`) stays unchanged — Approach B only touches the scoreboard-fallback path. |  |
| Surfaces directly via Codex's check: I reproduced 65 straddle / 42 strangle baseline trades with five lines of pandas — easy to audit. |  |

**Recommended: Approach B.** The raw trade log is the cleaner
source of truth, exists in production artifacts (verified above),
and removes the implicit "trust the scoreboard's aggregation"
dependency.

## Implementation sketch

### Changes to `services/structure_scorecard.py`

Introduce a shared helper:

```python
# Baseline backtest filter — must match the constants in
# scripts/backtest_pre_earnings_otm_strangle.py (BASE_ENTRY_OFFSET=3,
# BASE_EXIT_OFFSET=0) plus a fixed (OI>=100, spread<=10%) liquidity
# filter. Pinned in code rather than read from the backtest config
# because a backtest-side change to these baselines is a
# selector-affecting decision that should be reviewed deliberately.
_BACKTEST_BASELINE_ENTRY_OFFSET = 3
_BACKTEST_BASELINE_EXIT_OFFSET = 0
_BACKTEST_BASELINE_OI_FILTER_COL = "entry_pass_oi_100"
_BACKTEST_BASELINE_SPREAD_FILTER_COL = "entry_pass_spread_10"


def _load_prior_from_trade_log(
    *,
    structure_kind: str,           # "atm_straddle" or "otm_strangle_3pct"
    prior_structure_label: str,    # "atm_straddle" or "otm_strangle"
    scenario_pnl_col: str,         # "pnl_cross_25" today
) -> Optional[WalkForwardPrior]:
    """Read pre_earnings_otm_strangle_trade_log.csv, filter to the
    canonical baseline configuration, and compute (n, win_rate,
    avg_return) at the physical-trade level.

    Returns None when the trade log file is absent — caller routes
    to a documented fallback (scoreboard or neutral prior).
    """
    path = _latest_report_file(
        "pre_earnings_otm_strangle_*/pre_earnings_otm_strangle_trade_log.csv"
    )
    if path is None:
        return None
    frame = pd.read_csv(path)
    subset = frame[
        (frame["structure"] == structure_kind)
        & (frame["entry_offset_bdays"] == _BACKTEST_BASELINE_ENTRY_OFFSET)
        & (frame["exit_offset_bdays"] == _BACKTEST_BASELINE_EXIT_OFFSET)
        & (frame[_BACKTEST_BASELINE_OI_FILTER_COL] == True)  # noqa: E712
        & (frame[_BACKTEST_BASELINE_SPREAD_FILTER_COL] == True)  # noqa: E712
        & (frame["priceable_realized"] == True)  # noqa: E712
        & (frame[scenario_pnl_col].notna())
    ]
    if subset.empty:
        return None
    n = int(len(subset))
    win_rate = float((subset[scenario_pnl_col] > 0).mean())
    # Avg return at the physical-trade level — no realized_count
    # weighting; each row is one trade.
    return_col = scenario_pnl_col.replace("pnl_", "return_") + "_pct"
    if return_col in subset.columns:
        avg_return = float(subset[return_col].dropna().mean())
    else:
        avg_return = 0.0
    rank = _compute_rank_score(
        win_rate=win_rate, avg_return_pct=avg_return, history_count=n,
    )
    return WalkForwardPrior(
        structure=prior_structure_label,
        history_count=n,
        win_rate=win_rate,
        avg_return_pct=avg_return,
        rank_score=rank,
        source=f"{path.parent.name}:trade_log:baseline:{scenario_pnl_col}",
    )
```

Then:

```python
def _load_straddle_prior_from_reports() -> WalkForwardPrior:
    # Primary: raw trade log at baseline filter.
    prior = _load_prior_from_trade_log(
        structure_kind="atm_straddle",
        prior_structure_label="atm_straddle",
        scenario_pnl_col=f"pnl_{PROMOTION_BASELINE_SCENARIO}",
    )
    if prior is not None:
        return prior
    # Fallback: legacy scoreboard aggregation. KEPT for back-compat
    # if a future deployment has a scoreboard but no trade log
    # (unlikely; backtest writes both, but the fallback prevents a
    # hard failure). Source label flags the contamination so
    # downstream readers can see it.
    return _load_straddle_prior_from_scoreboard_legacy()

def _load_strangle_prior_from_reports() -> WalkForwardPrior:
    # Primary: summary JSON's baseline_cost_table (UNCHANGED — this
    # path was already clean).
    [...existing code...]
    # Secondary: raw trade log at baseline filter.
    prior = _load_prior_from_trade_log(
        structure_kind="otm_strangle_3pct",
        prior_structure_label="otm_strangle",
        scenario_pnl_col=f"pnl_{PROMOTION_BASELINE_SCENARIO}",
    )
    if prior is not None:
        return prior
    # Tertiary: legacy scoreboard aggregation. KEPT with the
    # contamination-flagged source label, same as straddle.
    return _load_strangle_prior_from_scoreboard_legacy()
```

The current `_load_straddle_prior_from_reports` body gets renamed
to `_load_straddle_prior_from_scoreboard_legacy` (similarly for
strangle); its source label changes to include
`"legacy_scoreboard_fallback"` so downstream consumers can
recognize the contaminated path when it fires.

### Why we keep the scoreboard fallback

Two scenarios where it fires:

1. The trade log file is missing or unreadable. Today's deployment
   always writes both, but a partial copy or a future refactor
   could break this.
2. A future scenario column we don't yet have (`pnl_cross_75`?)
   exists in the scoreboard but not the trade log.

In both cases the fallback degrades to the old behavior with a
visibly-different `source` label so the downstream
`composite_structure_score` can be traced.

## Tests

Codex review correction: unit tests use **synthetic fixtures
only**. Live-report numerical pinning is moved to a manual smoke
captured in the PR body, NOT a recurring unit-suite check —
those report artifacts are not guaranteed stable across future
backtest re-runs, and pinning them in pytest would create a
brittle test that fails every time the data is refreshed.

New test class `TestPriorLoadersFromTradeLog` in
`tests/unit/test_services/test_structure_scorecard.py`:

1. **Baseline filter selects exactly the right subset.** Synthetic
   trade-log fixture with ~20 trades distributed across structures,
   entry/exit offsets, and OI/spread filter flags. Loader's filter
   must yield only the baseline-eligible rows (entry=3, exit=0,
   OI>=100, spread<=10%, priceable, finite cross_25), not the
   cartesian product of robustness filters.

2. **No realized_count weighting / scoreboard-equivalence diverges
   from trade-level.** Construct a fixture where the
   hypothetical scoreboard-weighted win_rate would differ from the
   trade-level win_rate by a known amount. Assert the loader
   produces the trade-level value, not a weighted average.

3. **Returns None when trade log missing.** Loader on an empty
   reports dir returns None so the caller can fall through to the
   legacy scoreboard path.

4. **Fallback chain source labels.** When the trade log is
   missing AND the scoreboard exists, the straddle/strangle
   loaders fall back to scoreboard aggregation and emit a `source`
   string containing `"legacy_scoreboard_fallback"`. Downstream
   readers can detect when the contaminated path fires.

5. **Schema column-existence guard.** Assert the live trade log
   (when present) has the columns the loader depends on
   (`structure`, `entry_offset_bdays`, `exit_offset_bdays`,
   `entry_pass_oi_100`, `entry_pass_spread_10`, `priceable_realized`,
   `pnl_cross_25`, `return_cross_25_pct`). Per Codex: only the
   columns, NOT specific win_rate values. This catches a backtest
   schema regression without coupling to specific data values.

6. **Strangle primary path (`baseline_cost_table`) unchanged.**
   Construct a synthetic summary JSON fixture and assert the
   strangle loader uses it without falling through to the trade
   log. Guards against PR #71 accidentally regressing the
   already-clean primary path.

**Live-artifact verification (manual, PR body only).** Run the
loader against the current report set
(`pre_earnings_otm_strangle_20260417T180430Z/`) and document the
before/after numbers in the PR body so the reviewer can confirm
the magnitude of the behavior change without baking those numbers
into the test suite.

Total estimated test additions: ~6 tests, ~150 LOC.

## Risks + open questions

- **Backtest schema stability.** The trade-log columns we depend
  on (`structure`, `entry_offset_bdays`, `exit_offset_bdays`,
  `entry_pass_oi_100`, `entry_pass_spread_10`, `priceable_realized`,
  `pnl_cross_25`, `return_cross_25_pct`) are documented inside
  `scripts/backtest_pre_earnings_otm_strangle.py` but not formally
  versioned. Test #6 above adds a schema-stability guard.

- **Behavior change.** Selector scores will shift for both
  straddle and strangle (when the scoreboard-fallback path was
  firing on strangle). This is the **corrected** behavior — the
  selector should not prefer multi-counted evidence to a clean
  trade-level prior — but it IS a real behavior change and the PR
  body should call it out explicitly, same shape as PR #70.

- **Avg-return convention.** The scoreboard exposes
  `avg_pnl_per_trade` (dollars). The trade log has `pnl_cross_25`
  (dollars) and `return_cross_25_pct` (percent). The existing
  `_compute_rank_score` takes `avg_return_pct` — so we use
  `return_cross_25_pct.mean()`. Documented in the test fixture so
  the conversion is explicit.

- **What if `baseline_cost_table` in the summary JSON also has
  bugs?** Out of scope for PR #71. Strangle's primary path uses
  it; if a separate audit finds issues there, that's a follow-up.

## Recommendation

Proceed with **Approach B as scoped above**. The contamination is
real (~3.1 pp inflation on straddle today), the cleaner data
source exists in production artifacts, the diff is bounded
(~150 LOC + ~150 LOC of tests), and the fallback keeps
back-compat in case of report-path edge cases.

Awaiting Codex review of this design note before any code.
