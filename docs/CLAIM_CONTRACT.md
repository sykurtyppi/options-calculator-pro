# Claim contract: `conservative_fill_v3`

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
| Contract identity | Every leg names a standard OCC option symbol (`ROOT` `YYMMDD` `C/P` `strike×1000`) with a real expiration date and a positive strike. |
| Underlying | Every leg's root is the recommendation's own underlying (share-class punctuation dropped: `BRK.B` trades as `BRKB`). The symbol comes from the caller, else the quote's `symbol`; with neither, the quote fails (`claim_missing_underlying`). A leg on another symbol fails (`claim_<leg>_contract_not_on_underlying`). |
| Unexpired | No leg expired before the quote's session day; expiration day itself is still live (`claim_<leg>_contract_expired`). |
| Corporate actions | An adjusted contract (its root gains a digit after a split or special dividend, e.g. `AAPL1`) fails identity, so adjusted deliverables never enter a claim. |
| Bid/ask completeness | Every leg has a finite bid ≥ $0.05, an ask ≥ the bid, and a spread ≤ 25% of its mid. |
| Leg quote times | Every leg carries its own quote time (MarketData `updated`), inside the regular session of an NYSE trading day: 09:30 to the close plus 15 minutes. The close is 16:00 ET, or 13:00 ET on early-close days (July 3, the day after Thanksgiving, December 24, when each is a trading day). |
| Point in time | No leg is newer than the capture time beyond a 5-second clock-skew allowance. A newer quote was not known at capture and would be look-ahead. |
| Synchronized legs | All leg quote times are within 60 seconds of each other. |
| Quote age | The oldest leg is at most 300 seconds older than the reference time. The reference is the capture time, or the session close when the quote was captured after it, so the after-close daily cycle claims closing quotes. All legs belong to the capture's quote session. |

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

A label is not a pass: a quote marked "execution grade" is claimable only if it passes the contract like any other.

### Verdicts

Each side's verdict is stored with the trade: the entry verdict in `entry_execution_scenarios.claim`, the exit verdict in `exit_execution_scenarios.claim`. A verdict is well formed when:

- `passed` is a bool and `reasons` is a list of strings
- `passed` is true exactly when `reasons` is empty
- `contract_version` is the current version

At entry, `claim_allowed` means "claimable if the exit passes too", and the row stores the `claim_contract_version` it was checked under. The outcome and baseline stores do not take a caller's `claim_allowed=True` on trust: they record it only with a well-formed, passing entry verdict under the current version, and otherwise store `claim_allowed = 0` with the reason.

When the exit is recorded, the store keeps `claim_allowed` only if all of these hold, and adds the verdicts' reasons to `evidence_quality_reasons`:

- the stored entry verdict is still present, well formed and passed
- the exit verdict is present, well formed and passed
- both the entry and the exit were checked under the current version

Anything else fails closed:

- **No verdict:** `claim_entry_verdict_missing` / `claim_exit_verdict_missing`
- **Wrong types (e.g. `reasons` a string):** `claim_<side>_verdict_malformed`
- **`passed` disagreeing with `reasons` (a pass that lists a reason, or a failure with none):** `claim_<side>_verdict_inconsistent`
- **Entry or exit checked under another version:** `claim_contract_version_mismatch`

Reports count a row as claimable only under the current version, so claims made under different rules are never pooled.

## The claimed price

Every claimable outcome is reported three ways in the forward-performance
diagnostics, under `claimable_evidence.claimable_returns`:

| Level | Fill per leg |
|---|---|
| `mid` | The mid. Reported, never claimed. |
| `natural` | Pay the ask when buying, receive the bid when selling. |
| `conservative_fill` | **The claimed price.** The natural side, moved a further slippage of max($0.01, 10% of the spread) per share, plus $0.65 commission and $0.02 regulatory fee per contract on every leg at entry and at exit. |

## Versions

- `conservative_fill_v1` (#170): let through future-dated legs, quotes after an early close, impossible contract dates, and entries with no exit verdict.
- `conservative_fill_v2` (#172): all of these fail. Still let through expired contracts, legs on another underlying, a `passed` verdict listing blocking reasons, string `reasons`, entries with no entry verdict, and execution-grade labels without the contract.
- `conservative_fill_v3` (current): all of these fail. Rows claimed under v2 are not pooled with v3 claims.

## What this does not cover

- **Size and market impact beyond one contract per leg.**
- **Broker-specific fees and assignment risk.**
- **Same-day re-pricing:** the exit quote is a new snapshot, not a held order.
- **Modeled, not executed:** "realized" returns are deterministic arithmetic on recorded quotes, not broker executions.
- **A selected subset:** claimable outcomes are the liquid, well-quoted subset of trades, not overall strategy performance.
