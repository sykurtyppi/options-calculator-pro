# Claim contract: `conservative_fill_v1`

Forward paper outcomes are priced from option-chain snapshots, not broker
fills. This contract decides when such an outcome may be **claimed** as
evidence, and at what price. It is frozen: the code constants live in
`services/claim_contract.py`, and changing any of them means a new
`CONTRACT_VERSION`, so claims made under different rules are never pooled.

## Fill source

Claims use a **conservative modeled fill** on live quotes. There is no broker
fill and no independent fill timestamp, so a claim is evidence that the trade
*could have been done at least this badly*, not that it was done.

## When a quote passes

A structure quote passes, at entry or at exit, only if every rule holds.

| Rule | Requirement |
|---|---|
| Source | Live quote-level feed: `marketdata_app`. yfinance chains are delayed and indicative and never pass. |
| Contract identity | Every leg names a standard OCC option symbol (`ROOT` `YYMMDD` `C/P` `strike×1000`). |
| Corporate actions | An adjusted contract (its root gains a digit after a split or special dividend, e.g. `AAPL1`) fails identity, so adjusted deliverables never enter a claim. |
| Bid/ask completeness | Every leg has a finite bid ≥ $0.05, an ask ≥ the bid, and a spread ≤ 25% of its mid. |
| Leg quote times | Every leg carries its own quote time (MarketData `updated`), inside the regular session (09:30–16:15 ET) of an NYSE trading day. |
| Synchronized legs | All leg quote times are within 60 seconds of each other. |
| Quote age | The oldest leg is at most 300 seconds older than the reference time. The reference is the capture time, or the 16:00 ET close when the quote was captured after it, so the after-close daily cycle claims closing quotes. All legs belong to the capture's quote session. |

## When an outcome is claimable

A resolved outcome is claimable when all three hold:

1. **Entry:** the entry quote passed, and the entry record has no reason that puts the event itself in doubt:
   - missing or yfinance quote source
   - stale earnings source
   - low earnings-date confidence
   - any blocking evidence-quality reason
2. **Exit:** the exit quote passed.
3. **Same contracts:** the exit re-priced exactly the contracts booked at entry, with the same leg names and the same OCC symbols.

Chain-wide surface-quality warnings, such as IV anomalies on other strikes, degrade the record but not the fill, so they do not block a claim.

At entry, `claim_allowed` means "claimable if the exit passes too". When the exit is recorded, the store keeps `claim_allowed` only if the exit passed, and adds the exit's reasons to `evidence_quality_reasons`.

## The claimed price

Every claimable outcome is reported three ways in the forward-performance
diagnostics, under `claimable_evidence.claimable_returns`:

| Level | Fill per leg |
|---|---|
| `mid` | The mid. Reported, never claimed. |
| `natural` | Pay the ask when buying, receive the bid when selling. |
| `conservative_fill` | **The claimed price.** The natural side, moved a further slippage of max($0.01, 10% of the spread) per share, plus $0.65 commission and $0.02 regulatory fee per contract on every leg at entry and at exit. |

## What this does not cover

- **Size and market impact beyond one contract per leg.**
- **Early-close sessions.** Quotes at the 13:00 close fail the age rule, which is conservative.
- **Broker-specific fees and assignment risk.**
- **Same-day re-pricing:** the exit quote is a new snapshot, not a held order.
