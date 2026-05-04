# Model Card: StructureScorecard

## Purpose
`StructureScorecard` evaluates each supported structure independently using the same `VolSnapshot`.

## Inputs
VolSnapshot fields, walk-forward prior data when available, execution/liquidity metrics, historical move ratios, term structure, and surface shape.

## Outputs
Eligibility, expected edge diagnostics, expected return signal, penalties, walk-forward evidence, composite score, and rationale bullets.

## Limitations
Scores are deterministic and interpretable, not machine-learned predictions. Expected values are score-derived diagnostics unless backed by separate empirical calibration.

## Known Failure Modes
Thin walk-forward history, stale priors, wide spreads, unpriced structures, or sparse surface points can make a score less reliable.

## Data Dependencies
VolSnapshot, structure prior store, historical walk-forward outputs, and current option-chain liquidity.

## Validation Approach
Tests cover structure differentiation, monotonic penalties, no-trade eligibility, determinism, and low-sample behavior.

## Do Not Infer
Do not infer that the highest score guarantees profit or that score differences are precise expected-return differences.
