# Walk-forward pricing harness

`services/pricing_harness.py`, run by `scripts/run_pricing_harness.py`.

The selector's `expected_edge_pct` had never been compared with what the
structures went on to return, so no one could say how accurate it is. The
harness replays the historical end-of-day options archive point in time and
makes that comparison. Every later pricing change (event-variance forecast,
daily-vol forecast, empirical costs, per-strike IV) is judged against its
numbers: a new model replaces the score only if it does better here.

It changes nothing in live selection and writes only its own report directory.

## What one replay does

For every earnings event of every symbol, dated in `[start, end]`:

| Step | Rule |
|---|---|
| Entry | Every archive session 3 to 14 calendar days before the event (the forward loop's entry window, `DTE_MIN_DEFAULT`..`DTE_MAX_DEFAULT`), strictly before the exit session. The first is the event's *first look*. |
| Snapshot | `build_vol_snapshot` on that session's chain and the price history through it. Only events reported before the session count as history. The same full chain is passed as the entry-equivalent `structure_chain`, so the legs scored are the legs entry books (coverage `complete`). |
| Predictions | `build_structure_scorecards` (expected edge, expected return, composite score) and `select_best_structure`. |
| Priors | `walk_forward` (default): each structure's prior holds only the harness's own conservative-fill outcomes that **exited before** the entry session, one per event (its first look), aggregated and shrunk into the neutral prior exactly as production shrinks live observations. `neutral`: the no-data prior. The report files and the persistent prior store are never read; they were computed over the whole history. |
| Exit | The last NYSE session before the event date. The same contracts (expiry, side, strike) are re-priced from that session's chain. |
| Returns | Mid, natural and the claim contract's conservative fill (`services/execution_scenarios.py`). The condor's return is on its capital at risk (wing width less the mid credit). Predictions are judged against `conservative_fill`. |
| Not priced | A contract missing from the exit chain, quoted twice, with a zero bid (kept apart, as a likely loss), or without a valid two-sided quote is unpriced with its reason. It is counted, never filled in, and never reaches the priors. |

## The report

`summary.json`, `rows.csv` (one row per event, entry session and structure) and
`report.md`, under `exports/reports/pricing_harness/<UTC timestamp>/`.

- **Prediction quality** per structure and per prediction field, on the first
  look (one row per event) and on every entry. It covers:
  - Spearman and Pearson correlation;
  - sign hit rate;
  - bias and mean absolute error;
  - mean realized return when the prediction is positive or not;
  - calibration by prediction quintile.

  These are also reported for eligible structures only, and by year.
- **Policies**, one trade per event:
  - `selector`: the first session whose decision is traded, in the chosen structure, which is what the loop books.
  - `straddle_same_entry`: the ATM straddle at that same session.
  - `selector_minus_straddle_same_entry`: the paired difference between the two.
  - `straddle_first_look`: the straddle at every event's first look.
  - `rank_pick_first_look`: the highest expected edge with no gates.

  Each policy has a 95% bootstrap interval of its mean.
- **Counts:** every event and entry skipped or failed, and every unpriced row, by reason.

## Limits, stated in every report

- **No claims.** The archive is end of day: there is no live source and no per-leg quote time, so no fill here is a claim under the claim contract. The conservative fill is used as a price, not as evidence.
- **Post-hoc event dates.** Every source in `historical_earnings_events` currently writes `lookahead_safe_for_pre_event = 0`, and the only `known_as_of` is an SEC filing time, which comes after the event. So the replay uses each event's final date, and a rescheduled event appears where it finally landed. `--require-lookahead-safe` keeps only entries on which the date was verifiably known; today that leaves nothing. The report shows how many first-look rows had the date known beforehand.
- **Correlated entries.** Entries within one event are correlated. The first-look metrics and the policies count each event once.
- **Not fully out of sample.** The scorecards' hand-set constants are not fitted here, but they may have been tuned on this period.

## A finding: the loop's exit day for Monday events

The forward loop exits on calendar day `earnings_date - 1`
(`OutcomeStore.trades_due_for_exit`, `mark_missing_exits`). For a Monday event
that is a Sunday, and for an event after a holiday it is the holiday. The
harness exits on the last session before the event instead, and counts the
events where the two differ (`events_loop_exit_day_not_a_session`). Whether
the loop's daily cycle runs on those days, and what quote it gets there, needs
checking on the Mac.

## Running it (on the Mac, where the archive is)

```
.venv_arm64/bin/python3 scripts/run_pricing_harness.py --start 2023-01-01
.venv_arm64/bin/python3 scripts/run_pricing_harness.py --symbols AAPL,MSFT,NVDA --prior-mode neutral
```

The default data root is `$MARKET_DATA_ROOT` or `/Volumes/T9/market_data`. The
inputs are:

- `research/options_features_eod`
- `research/underlyings/daily_features`
- `research/options_calculator_pro/earnings_calendar/historical_earnings.sqlite`

Each can be overridden: `--options-root`, `--ohlc-root`, `--earnings-db`. Run it
after the earnings labels are rebuilt under the release-timing rule (#205),
because the trade expiry depends on each event's timing.
