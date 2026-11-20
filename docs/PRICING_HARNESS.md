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

## Event variance: measured, and its change forecast

Step 2 of the pricing plan (`services/event_variance.py`). What a T-1 straddle
earns is mostly the change in the event variance the market prices (E)
between entry and exit, plus ordinary daily moves, less costs. So E is
measured at both ends, and its change is forecast.

**Measurement.** Two expiries that both span the event (on or after the
reaction session; unknown timing counts as after the close) each give a total
implied variance W = IV² × calendar days / 365, the archive's IV convention.
With T the NYSE sessions from the as-of date through the expiry, W = b·T + E,
so

    E = (W1·T2 − W2·T1) / (T2 − T1),    b = (W2 − W1) / (T2 − T1).

ATM IV is interpolated linearly in strike at the spot price, never
extrapolated, with calls and puts averaged. The exit is measured on the
entry's expiry pair, so the change is not partly a change of expiries. Unlike
the snapshot's split, this uses no realized-vol forecast. It also counts trading
sessions, so a long weekend is not read as base variance: this is the
calendar-vs-trading-day bias documented in `event_vol_decomposition.py`. An
entry that cannot be measured is counted by reason:

- `fewer_than_two_spanning_expiries`
- `atm_iv_unavailable`
- `expiries_not_separable`
- `non_positive_event_variance`

**Forecast.** The target is log(E_exit / E_entry), which is scale-free and
so can be pooled across names. It is predicted from observations whose exit
came before the entry, built in three layers:

1. the pooled mean for the holding-period bucket (1–2, 3–4, 5–6, 7–8 or 9+ sessions);
2. a group offset (by sector, `--group-by sector`, from `scripts/stress_test_expansion.SECTOR_MAP`);
3. a symbol offset.

Each offset is an average of per-event mean residuals, shrunk by n / (n + 8)
with n the number of events behind it. Every entry of an event is an
observation, but an event counts once in an offset.

**Judged against two baselines:**

- no change (log ratio 0);
- the pooled bucket mean alone.

The report gives RMSE, out-of-sample R² against each baseline (above 0 beats
it), Spearman correlation, and the realized and forecast means by holding
period. These are reported for first looks and for all entries, with one row
per entry in `variance_rows.csv`.

The sector map is incomplete:

- ADBE, SNPS, ZS, DDOG, NET, TEAM, WDAY, SNOW, OKTA, TTD, PYPL and EBAY are missing; unmapped names get no group offset.
- GIS appears twice, so the second entry ("Indust") is the one kept.
- DGX is labelled REIT.

Fix it before relying on group offsets.

## Ordinary daily variance during the hold

Step 3 of the pricing plan (`services/daily_vol.py`). Between entry and exit a
straddle also gains when the stock moves more on ordinary days than the base
variance the options price, and loses when it moves less.

**Target:** the realized close-to-close variance of the hold, which is the
mean squared log return of the sessions after the entry through the exit.
Every session must have a close; a gap is reported as `hold_closes_missing`
and never averaged over. The hold ends before the event, so the event is
never in it.

**Forecasts, all in variance per session and point in time:**

| Forecast | What it is |
|---|---|
| `implied` | The market's base variance b from the two-expiry fit. |
| `har_rs` | What production uses today: the snapshot's HAR on Rogers-Satchell variance, which is open-to-close only. It leaves out the overnight gap, but an option is paid on close-to-close moves. |
| `har_cc` | The same HAR regression on close-to-close returns, with the sessions earlier earnings moved excluded (the snapshot's rule). |
| `calibrated` | `har_cc` times a walk-forward scale. The pooled scale is the ratio of summed realized to summed forecast variance over every earlier hold. Group and symbol factors are each a per-event mean ratio, shrunk toward 1 by n / (n + 8). |

**Losses.** A few sessions' squared returns are a noisy, skewed estimate of
the true variance. By Jensen's inequality, a log loss (and a log-ratio
calibration) rewards forecasts that are too low, and on synthetic data it ranks
a forecast 30% too low ahead of the truth. So the report uses only the
losses that rank correctly against a noisy target (Patton 2011):

- **QLIKE:** realized/forecast − log(realized/forecast) − 1.
- **Squared error in levels.**

For each loss it reports skill against `implied` (1 − loss / loss(implied),
above 0 beats the market). Bias is reported as a level ratio, mean realized ÷
mean forecast (above 1 means the forecast is too low). Spread Spearman asks
whether a forecast knows when the market's base variance is rich or cheap.
All forecasts are compared on the same rows.

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
