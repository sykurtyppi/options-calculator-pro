# Disclaimer

## Not financial advice

This project is a **research and decision-support tool**. It is not investment
advice, not a recommendation to buy or sell any security, and not a validated
trading system. Nothing it outputs should be treated as a solicitation or as a
personalized recommendation. The author is not a licensed investment adviser,
broker-dealer, or financial professional.

Options trading involves substantial risk of loss and is not suitable for every
investor. You can lose more than your initial investment in some strategies. Any
decision to trade is solely your own and taken entirely at your own risk.

## What the numbers actually mean

The interface uses words like *recommendation*, *confidence*, *expected edge*,
*expected return*, and *setup score*. Read them narrowly:

- They are **score-derived diagnostics** computed from heuristic rules, hand-set
  constants, and corpus-calibrated bounds.
- They are **not calibrated probabilities**. A "confidence" of 70% does not mean
  the trade wins 70% of the time. No reliability diagram has established
  calibration.
- They are **not return forecasts**. An "expected return" figure is a relative
  ordering signal, not an estimate of conditional expected profit and loss.
- They are **not evidence of a demonstrated edge**. The engine's own output
  carries this caveat: *"Expected-edge and expected-return fields are
  score-derived diagnostics. They are not empirical return forecasts and are not
  calibrated to live retail execution."*

## Evidence status

The forward paper-trade ledger is small, recent, and clustered in time. It is
labelled *paper, not execution-grade* in the interface for that reason. Sample
sizes at this scale cannot establish statistical significance: roughly 30 trades
is where significance can begin to be estimated at all, and a few hundred is
where an edge becomes believable.

Backtest and walk-forward results in this repository are **research artifacts,
not a track record**. They have not been corrected for selection bias across the
number of structures, parameter sets, and scenario variants that were evaluated
to arrive at the current configuration. Uncorrected, the best result from a set
of trials is inflated by construction.

Paper-trade results also do not include real fills, partial fills, queue
position, assignment, early exercise, borrow, margin changes, or taxes.

## Data

Market data comes from third-party providers and may be delayed, cached, stale,
incomplete, or wrong. The software attempts to label quote quality and staleness,
but those labels are best-effort. Provider data is governed by each provider's
own terms; see LICENSE.

## No warranty

The software is provided "as is", without warranty of any kind. The author
accepts no liability for any loss or damage arising from its use. See LICENSE.
