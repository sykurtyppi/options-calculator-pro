# Institutional Backtest Upgrade (2026-02-21)

## What Changed

- Replaced random backtest simulation in `services/institutional_ml_db.py` with a deterministic walk-forward engine.
- Added earnings-event anchored entry/exit windows (true earnings dates from yfinance cache, optional proxy fallback).
- Added execution-friction integration using `services/execution_cost_model.py` profiles (`paper`, `retail`, `institutional`, `institutional_tight`).
- Added trade-level persistence table: `backtest_trades` (including `event_date`, `days_to_earnings`).
- Added earnings event cache table: `earnings_events`.
- Added option snapshot table: `earnings_option_snapshots`.
- Added calibrated label table: `earnings_iv_decay_labels`.
- Added trade-level retrieval API: `get_backtest_trades(session_id, limit=None)`.
- Added parameter sweep API: `run_backtest_parameter_sweep(base_params, parameter_grid, top_n=None)`.
- Added rolling OOS API: `run_rolling_oos_validation(...)`.
- Added strict IV-crush confidence gate controls in walk-forward candidate selection.
- Added rolling IV-crush prediction scorecard APIs:
  - `build_rolling_crush_scorecard(...)`
  - `summarize_crush_scorecard(...)`
- Added regime diagnostics APIs:
  - `build_regime_diagnostics(...)`
  - `summarize_regime_diagnostics(...)`
- Added decile threshold tuning APIs:
  - `build_signal_decile_table(...)`
  - `recommend_signal_threshold_from_deciles(...)`
- Improved feature generation for `ml_features` to remove hardcoded IV/VIX placeholders and derive volatility proxies from actual historical price behavior.
- Updated `scripts/institutional_backfill.py` with walk-forward backtest controls.

## Deterministic Walk-Forward Model

For each symbol and earnings event:

1. Resolve true earnings dates from cached/API data (`source=yfinance`), with optional proxy fallback (`source=proxy`).
2. Create entry date around `entry_days_before_earnings` and exit target around `exit_days_after_earnings`.
3. Build setup quality score from IV/RV, momentum stability, RSI, Bollinger position, volume pressure, and regime proxy.
4. Filter by score threshold and IV/RV bounds.
5. Simulate calendar-style gross return from:
   - theta/vol carry
   - move stability vs expected move
   - trend penalty
6. Apply profile-based transaction costs using execution-cost model.
7. Store net trade record and aggregate session metrics (win rate, total PnL, Sharpe, max drawdown, Calmar).

No random draws are used in the backtest engine.

## New Backtest CLI Controls

Available in `scripts/institutional_backfill.py`:

- `--execution-profile`
- `--hold-days`
- `--min-signal-score`
- `--max-trades-per-day`
- `--position-contracts`
- `--lookback-days`
- `--max-backtest-symbols`
- `--entry-days-before-earnings`
- `--exit-days-after-earnings`
- `--require-true-earnings`
- `--no-proxy-earnings`
- `--disable-crush-confidence-gate`
- `--no-global-crush-profile`
- `--min-crush-confidence`
- `--min-crush-magnitude`
- `--min-crush-edge`
- `--backtest-start-date`
- `--backtest-end-date`

Example:

```bash
python scripts/institutional_backfill.py --backtest-only \
  --execution-profile institutional_tight \
  --hold-days 7 --min-signal-score 0.60 --max-trades-per-day 2 \
  --entry-days-before-earnings 7 --exit-days-after-earnings 1 \
  --require-true-earnings --backtest-start-date 2024-01-01 --backtest-end-date 2025-12-31
```

## Parameter Sweep CLI

Use `--parameter-sweep` to generate ranked CSV + markdown reports:

```bash
python scripts/institutional_backfill.py --backtest-only --parameter-sweep \
  --sweep-profiles institutional,institutional_tight \
  --sweep-hold-days 5,7 --sweep-min-signal-scores 0.55,0.62 \
  --sweep-max-trades-per-day 1,2 \
  --sweep-entry-days-before 5,7 --sweep-exit-days-after 0,1 \
  --require-true-earnings \
  --backtest-start-date 2024-01-01 --backtest-end-date 2025-12-31 \
  --sweep-output-dir exports/reports --sweep-top-n 20
```

Sweep now also exports a machine-readable best-config file:

- `earnings_walkforward_sweep_<timestamp>_best_params.json`

## IV-Crush Calibration Pipeline

Use live snapshot capture + calibration so the engine learns symbol-specific crush behavior:

```bash
python scripts/institutional_backfill.py --backtest-only \
  --capture-snapshots --calibrate-iv-decay \
  --full-universe \
  --snapshot-lookback-days 120 --snapshot-lookahead-days 120 --snapshot-max-expiries 2 \
  --label-min-pre-days 1 --label-max-pre-days 12 \
  --label-min-post-days 0 --label-max-post-days 5
```

Notes:
- Run snapshot capture daily to accumulate both pre- and post-event observations for the same earnings cycle.
- Keep proxy earnings enabled unless you have a verified true-earnings feed; `--require-true-earnings` can reduce coverage sharply.
- When using only `--capture-snapshots` and/or `--calibrate-iv-decay`, the script now skips the default sample backtest.
  Add `--run-sample-backtest` if you want to force a backtest in the same command.
- Use `--snapshot-status` to print current pre/post pairing progress without running a backtest.

This updates:

- `earnings_option_snapshots` (raw pre/post event IV observations)
- `earnings_iv_decay_labels` (paired crush labels used by backtest scoring)

## Rolling OOS Validation CLI

Use `--oos-validation` to run train/test rolling windows:

```bash
python scripts/institutional_backfill.py --backtest-only --oos-validation \
  --sweep-profiles institutional,institutional_tight \
  --sweep-hold-days 5,7 --sweep-min-signal-scores 0.55,0.62 \
  --sweep-max-trades-per-day 1,2 \
  --sweep-entry-days-before 5,7 --sweep-exit-days-after 0,1 \
  --oos-train-days 252 --oos-test-days 63 --oos-step-days 63 \
  --oos-top-n-train 1 --require-true-earnings \
  --backtest-start-date 2023-01-01 --backtest-end-date 2025-12-31 \
  --oos-output-dir exports/reports
```

OOS exports:

- `earnings_oos_validation_<timestamp>.csv`
- `earnings_oos_validation_<timestamp>.md`
- `earnings_oos_validation_<timestamp>_best_params.json`

## Crush Confidence Gate

The walk-forward engine now supports a strict pre-trade filter requiring:

- sufficient crush confidence (`min_crush_confidence`)
- minimum expected crush magnitude (`min_crush_magnitude`)
- minimum combined edge score (`min_crush_edge`)

Gate defaults are strict and can be relaxed or disabled.

Example:

```bash
python scripts/institutional_backfill.py --backtest-only \
  --hold-days 7 --min-signal-score 0.60 --max-trades-per-day 2 \
  --min-crush-confidence 0.45 --min-crush-magnitude 0.08 --min-crush-edge 0.05 \
  --require-true-earnings --backtest-start-date 2024-01-01 --backtest-end-date 2025-12-31
```

## Rolling Crush Scorecard CLI

Generate a rolling predicted-vs-realized IV-crush calibration report:

```bash
python scripts/institutional_backfill.py --backtest-only --crush-scorecard \
  --scorecard-window 40 --scorecard-min-confidence 0.35 \
  --scorecard-output-dir exports/reports
```

Optional session filter:

```bash
python scripts/institutional_backfill.py --backtest-only --crush-scorecard \
  --scorecard-session-id backtest_YYYYMMDD_HHMMSS_xxxxxx
```

Scorecard exports:

- `earnings_crush_scorecard_<timestamp>.csv`
- `earnings_crush_scorecard_<timestamp>.md`
- `earnings_crush_scorecard_<timestamp>.json`

## Regime Diagnostics CLI

Generate regime diagnostics by VIX, days-to-earnings, and IV/RV buckets:

```bash
python scripts/institutional_backfill.py --backtest-only --regime-diagnostics \
  --regime-min-confidence 0.35 \
  --regime-output-dir exports/reports
```

Optional session filter:

```bash
python scripts/institutional_backfill.py --backtest-only --regime-diagnostics \
  --regime-session-id backtest_YYYYMMDD_HHMMSS_xxxxxx
```

Regime diagnostics exports:

- `earnings_regime_diagnostics_<timestamp>.csv`
- `earnings_regime_diagnostics_<timestamp>.md`
- `earnings_regime_diagnostics_<timestamp>.json`

## Decile Threshold Tuning CLI

Generate decile-conditioned threshold recommendations:

```bash
python scripts/institutional_backfill.py --backtest-only --threshold-tuning \
  --tuning-min-confidence 0.35 --tuning-min-trades 30 \
  --tuning-output-dir exports/reports
```

Use raw setup score instead of composite (setup + crush edge):

```bash
python scripts/institutional_backfill.py --backtest-only --threshold-tuning \
  --tuning-raw-setup-score
```

Optional session filter:

```bash
python scripts/institutional_backfill.py --backtest-only --threshold-tuning \
  --tuning-session-id backtest_YYYYMMDD_HHMMSS_xxxxxx
```

Threshold tuning exports:

- `earnings_signal_deciles_<timestamp>.csv`
- `earnings_threshold_candidates_<timestamp>.csv`
- `earnings_threshold_tuning_<timestamp>.md`
- `earnings_threshold_tuning_<timestamp>.json`

## Notes

- This is now suitable for reproducible strategy iteration and parameter sweeps.
- It remains a proxy engine (no full historical option chain replay), so treat outputs as ranking/selection guidance, not final production execution expectancy.
