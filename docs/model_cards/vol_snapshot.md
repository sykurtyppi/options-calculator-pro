# Model Card: VolSnapshot

## Purpose
`VolSnapshot` is the canonical event-volatility truth layer for one ticker, as-of date, and earnings event.

## Inputs
Underlying price data, option chain data, earnings metadata, historical price bars, historical earnings moves, and source/staleness metadata.

## Outputs
Timing, market state, realized volatility, implied volatility, event move pricing, historical event behavior, surface shape, liquidity, and derived relationship fields.

## Limitations
It does not choose a trade, rank structures, or decide whether to enter. Missing fields remain null rather than fabricated.

## Known Failure Modes
Stale earnings calendars, sparse option chains, missing historical event data, stale prices, and low-liquidity contracts can reduce reliability.

## Data Dependencies
Market data providers, earnings-date providers, historical OHLCV data, and option quote quality.

## Validation Approach
Unit tests cover deterministic output, missing data, stale source behavior, signal direction, and snapshot neutrality.

## Do Not Infer
Do not infer profitability, probability of success, or the best structure from `VolSnapshot` alone.
