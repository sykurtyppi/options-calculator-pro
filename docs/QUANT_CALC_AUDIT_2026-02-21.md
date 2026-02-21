# Quant Calculation Audit (2026-02-21)

## Scope

This audit reviewed and strengthened:

- Yang-Zhang realized volatility estimation
- IV term-structure slope handling (including 0-45 day anchor)
- Volume/liquidity feature construction for ML
- Monte Carlo Heston calibration and simulation stability
- Merton jump-diffusion integration in the active Monte Carlo engine

## Critical Fixes Applied

1. Yang-Zhang estimator corrected in `services/volatility_service.py`
- Fixed close-variance component to use open-to-close returns.
- Switched to rolling sample variance (`ddof=1`) for overnight and open-close terms.
- Applied nonnegative variance clamp before square root.

2. Term-structure slope robustness in `services/volatility_service.py`
- Replaced nearest-point slope logic with interpolation/extrapolation-based slope.
- Prevents degenerate same-index slope selection and reduces tenor-grid instability.

3. Added explicit 0-45 slope outputs in `services/options_service.py`
- `build_iv_term_structure` now emits:
  - `slope_0_45`
  - `iv_30d`
  - `iv_45d`
- Includes fallback estimation when interpolation fails.

4. Fixed volume-feature leakage/scale mismatch in `controllers/analysis_controller.py`
- Added `underlying_avg_volume` and `underlying_last_volume` from underlying share data.
- ML `avg_volume` now uses underlying volume (compatible with existing ML normalization).
- Replaced the old option volume ratio that collapsed toward a near-constant value.

5. Upgraded active Monte Carlo engine in `utils/monte_carlo.py`
- Added Merton jump parameter calibration from IV-RV dislocation and earnings proximity.
- Added optional jump-diffusion overlay in Heston path simulation with risk-neutral drift compensation.
- Added dividend yield support in simulation drift.
- Added Feller-consistency safety enforcement during Heston calibration.
- Extended expected-move estimate to include jump variance contribution.

6. Compatibility hardening in `models/monte_carlo.py`
- Added missing `os` import used by CPU worker-count logic.
- Added `simulate_option_price` compatibility wrapper expected by worker code path.

7. Monte Carlo schema normalization in `core/workers/analysis_worker.py`
- Updated worker-side readers to consume active engine fields:
  - `heston_parameters`
  - `jump_parameters`
  - `simulations_run`
  - `value_at_risk_95` and `price_range_95`
- Removed stale assumptions on legacy keys (`heston_params`, `num_simulations`, `price_percentiles`).

## Validation Performed

- `python3 -m py_compile` passed for all edited files.
- Targeted synthetic sanity checks passed:
  - Yang-Zhang output finite and stable.
  - 0-45 slope computed as expected from synthetic term structure.
  - Jump intensity rises near earnings vs far-from-earnings in Monte Carlo outputs.
- Deterministic regression tests added and passing (`tests/unit/test_quant_regressions.py`):
  - Yang-Zhang estimator parity against independent reference calculation.
  - IV term-structure 0-45 interpolation/extrapolation behavior.
  - Earnings-proximity jump sensitivity.
  - Feller-condition stability check for calibrated Heston parameters.
  - Jump-disable runtime flag behavior.

## External Research Anchors

- Yang, D. and Zhang, Q. (2000), drift-independent OHLC estimator:
  - https://EconPapers.repec.org/RePEc:ucp:jnlbus:v:73:y:2000:i:3:p:477-91
- Heston, S. (1993), stochastic volatility model foundation:
  - https://doi.org/10.1093/rfs/6.2.327
- Merton, R. (1976), jump-diffusion option pricing foundation:
  - https://EconPapers.repec.org/RePEc:eee:jfinec:v:3:y:1976:i:1-2:p:125-144
- Euler-fix bias and full-truncation discussion for Heston simulation:
  - https://EconPapers.repec.org/RePEc:taf:quantf:v:10:y:2010:i:2:p:177-194

## Remaining High-Value Improvements

- Add a calibration job that fits Heston/jump parameters to observed option surfaces per symbol and regime.
- Add regime-segmented scorecards (earnings week vs non-earnings week) for jump calibration quality.
- Add calibration diagnostics (fit residuals and confidence bounds) to guard against overfitting on sparse option chains.
