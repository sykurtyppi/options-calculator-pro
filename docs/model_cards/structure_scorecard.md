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

## Iron Condor Horizon
Every structure exits the session before earnings, so the iron condor is scored for that window: it never collects the post-earnings crush or faces the earnings move, and the crush classifier and earnings-move ratios do not enter its score. Under the event split's assumption (non-event implied vol = HAR forecast) the premium that decays before the exit is priced at fair value at best, so the snapshot signal tops out at zero; a positive condor edge can only come from real condor outcomes through the walk-forward prior. A hold-through-earnings condor would be a different strategy with its own exit rule.
