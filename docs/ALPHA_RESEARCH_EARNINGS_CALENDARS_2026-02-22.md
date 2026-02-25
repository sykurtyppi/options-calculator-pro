# Earnings Calendar Spread Alpha Research (2026-02-22)

## Objective
Improve pre-earnings calendar spread signal quality for institutional-style IV-crush capture with stricter execution realism.

## What Research Says (Primary Sources)
- Earnings-day options contain a distinct event risk premium and should not be treated like normal-day vol surfaces.  
  Source: [Event-Day Options](https://www.nber.org/papers/w24708)
- Around earnings, option prices can be rich relative to realized outcomes after accounting for fees/spreads, especially for unsophisticated flow.  
  Source: [Losing is Optional? Retail Option Trading and Earnings Announcements (PDF)](https://www.timdesilva.com/files/Losing_is_Optional_2024.pdf)
- Anticipated earnings uncertainty embedded in options is informative, but edge depends on conditioning and timing state.  
  Source: [The Term Structure of Equity Option Implied Volatility around Earnings Announcements](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2986841)
- Implied-vs-realized volatility spread is one of the strongest cross-sectional return signals in options; however, implementation frictions matter.  
  Source: [The Volatility Spread (Journal of Banking & Finance abstract)](https://www.sciencedirect.com/science/article/abs/pii/S037842661830140X)
- Calendar spreads have assignment/exercise and structure-specific risks that must be treated as first-class constraints.  
  Source: [Calendar Spreads (Options Industry Council)](https://www.optionseducation.org/strategies/all-strategies/calendar-spreads)

## Strategy Implications
- Prefer a **timing sweet-spot** before earnings instead of static thresholds.
- Avoid monotonic “higher IV/RV is always better” logic; extreme readings can be crowded and expensive.
- Penalize crowded tape and low-liquidity contexts before ranking candidates.
- Blend setup quality with crush-profile quality (magnitude, confidence, signal strength), not setup score alone.

## Implemented in Code
- Added timing-aware and non-monotonic IV/RV setup scoring:
  - `/Users/tristanalejandro/Downloads/options_calculator_pro/services/institutional_ml_db.py`
  - methods: `_score_entry_timing`, `_score_iv_rv_quality`, `_score_term_structure_quality`, `_score_setup_quality`
- Added alpha-rank combiner for candidate selection:
  - `/Users/tristanalejandro/Downloads/options_calculator_pro/services/institutional_ml_db.py`
  - method: `_rank_candidate_for_alpha`
- Added execution-reality prefilters in walk-forward candidate loop:
  - min underlying share volume (`min_daily_share_volume`)
  - max abs 5d momentum (`max_abs_momentum_5d`)
- Added term-structure proxy persistence in feature pipeline:
  - `ml_features.vol_term_structure_slope` now populated during feature generation
- Added CLI/config knobs for these controls:
  - `/Users/tristanalejandro/Downloads/options_calculator_pro/scripts/institutional_backfill.py`
  - flags: `--target-entry-dte`, `--entry-dte-band`, `--min-daily-share-volume`, `--max-abs-momentum-5d`

## New Defaults Added
- `target_entry_dte=6`
- `entry_dte_band=6`
- `min_daily_share_volume=1,000,000`
- `max_abs_momentum_5d=0.11`

## Recommended Validation Run
```bash
python3 /Users/tristanalejandro/Downloads/options_calculator_pro/scripts/institutional_backfill.py \
  --backtest-only --threshold-tuning --regime-diagnostics \
  --hold-days 7 \
  --entry-days-before-earnings 7 \
  --exit-days-after-earnings 1 \
  --target-entry-dte 6 \
  --entry-dte-band 6 \
  --min-daily-share-volume 1000000 \
  --max-abs-momentum-5d 0.11 \
  --min-signal-score 0.50 \
  --min-crush-confidence 0.30 \
  --min-crush-magnitude 0.06 \
  --min-crush-edge 0.02 \
  --max-trades-per-day 4 \
  --lookback-days 730 \
  --max-backtest-symbols 20 \
  --backtest-start-date 2023-01-01 \
  --backtest-end-date 2025-12-31
```

## Caution
- This improves process quality and robustness, but alpha is regime-dependent.
- Require rolling out-of-sample checks and transaction-cost-aware evaluation before production capital.
