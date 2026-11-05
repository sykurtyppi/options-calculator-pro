# Pre-registration: forward evidence, cohort 1

**Status: DRAFT. Not in force until signed off below.** Once signed, this
document is changed only by appending a dated amendment (section 9); the
original text stays as written.

The purpose is to fix, before any cohort-1 outcome is seen, what question the
forward evidence answers, which events count, how they are analysed, and what
result means stop. Choices marked **DECIDE** are the owner's to make before
sign-off; the proposed value is filled in.

## 1. Preconditions: the cohort cannot start until all hold

1. **Claim-grade quotes.** The claim contract
   ([CLAIM_CONTRACT.md](CLAIM_CONTRACT.md), `conservative_fill_v4`) accepts only
   live quote-level quotes: MarketData.app, or IB Gateway on a live market-data
   subscription (`IB_GATEWAY_MARKET_DATA_TYPE=1`); yfinance quotes never pass,
   and an IB delayed feed fails on missing leg quote times. Every forward
   analysis so far has been priced from yfinance, so no outcome can be
   claimable until one of those feeds supplies the chains the loop prices.
   **DECIDE:** subscribe to MarketData.app, or add a live OPRA subscription to
   the IB account used by the gateway (or stop here: without one of them this
   pre-registration can never be tested).
2. **In-session pricing.** The evidence cycle runs during the NYSE regular
   session on trading days (#193), and every analysis carries quote provenance
   with the untrusted-quote gate active (#194).
3. **Universe fixed.** The universe is chosen and frozen before the start date
   (section 3). Changing it later starts a new cohort.
4. **Clean runs.** At least 5 consecutive scheduled trading-day runs with no
   `untrusted_quotes` stops caused by the pipeline itself and evidence health
   `OK`.

The **start date** is the first trading day on which all four hold. It is
recorded here at sign-off together with the commit SHA the Mac is running
(`/api/version`, and `code_identity` in that day's report).

## 2. Question and hypotheses

**Question.** Does the structure selector, taking one paper position per
earnings event priced at conservative fills and exited on the session before
earnings (T-1), make money, and does it beat simply always buying the same
structure on the same events?

All returns are `realized_return_pct` of **claimable** outcomes under
`conservative_fill_v4`, one per event (the claim contract and the
one-trade-per-event rule define both). The unit of analysis is the event.

- **H1 (primary): the selector beats no trade.** The mean selector return per
  claimable event is greater than zero: the 95% interval's lower bound is above
  0.
- **H2 (primary comparison): the selector beats always-straddle.** On the
  events where both have a claimable outcome, the mean paired difference
  `selector − always_atm_straddle` is greater than zero: the lower bound of its
  95% interval is above 0. **DECIDE:** always-straddle is proposed as the one
  primary baseline because it is the plain long pre-earnings volatility trade
  the selector claims to improve on.
- **Secondary (descriptive, not tests):** `selector − always_otm_strangle`,
  `selector − always_iron_condor` (a short-vol control exited at T-1), and the
  universe cohort's picked-minus-skipped difference. These are reported; none
  of them can rescue a failed H1 or H2.

Two primaries are tested; both must hold for "edge". No other metric, subset,
structure or time window is promoted to a headline after the fact.

## 3. Sample

- **Universe.** **DECIDE:** currently 39 symbols. The plan is to widen it before
  the start date (the gate needs far more events than 39 names produce in a
  season); the rule that picks the members (e.g. "S&P 500 members with listed
  weekly options and median ATM spread below X% during the session") is written
  here, not the list alone.
- **Events.** Earnings events of universe symbols whose earnings date is on or
  after the start date, discovered and analysed by the scheduled forward loop.
- **Included.** Claimable outcomes, as the claim contract defines them.
- **Excluded, and counted.** Outcomes the claim contract does not admit, events
  invalidated by the documented reason codes (`event_date_superseded`,
  `exit_repriced_unheld_strike`, and any added later by amendment), and
  candidates stopped by `untrusted_quotes`. Every exclusion keeps its row and
  reason; the report states how many were excluded and why. Nothing is dropped
  by hand.
- **Membership is frozen.** At each decision point the events in the window are
  frozen with `scripts/freeze_evidence_cohort.py freeze --name cohort-1-<n>`
  before the analysis is read, and the analysis is run on the frozen members
  only.

## 4. Decision points

| Point | When | What may be decided |
|---|---|---|
| Interim | 30 claimable selector outcomes (the paid-beta gate's sample) | Futility only (section 6). No claim of edge. |
| Final | **DECIDE:** 100 claimable selector outcomes, or the end of the 4th earnings season after the start date, whichever comes first | H1 and H2. |

Between decision points the daily and weekly reports keep running, but no
decision is taken on them and nothing in section 7 is changed because of them.

## 5. Analysis (fixed)

The analysis is the existing evidence report code, unchanged
(`services/evidence_statistics.py`, `services/evidence_report.py`):

- 95% percentile bootstrap, 2000 resamples, fixed seed 20260927; no interval
  below n = 10.
- H1: `mean_with_ci` over claimable selector returns.
- H2: `paired_difference` of selector and baseline returns keyed by event.
- Robustness: `outlier_dependence` with k = 5. A result that is positive only
  because of its 5 best events (`fragile`) does not count as edge.
- Exit attrition and quote integrity as the health check reports them.

## 6. Outcomes and kill criteria

**At the interim:**
- *Stop for futility* if the upper bound of the H1 interval is below 0 (the
  selector is losing money with confidence). Record the result; do not tune and
  continue within this cohort.

**At the final:**
- *Edge supported* only if H1 and H2 both hold **and** the H1 result is not
  `fragile`.
- *No edge* otherwise. The result is reported as no edge. The commercialization
  gate stays closed. Any new idea (other gates, structures, universe) becomes a
  new pre-registered cohort; cohort-1 data may motivate it but never confirm it.

**At any time, evidence invalid (fix, then restart as a new cohort):**
- exit attrition above 20% over the window (the health check's threshold:
  results are selected by quote availability);
- `untrusted_quotes` stops above **DECIDE:** 10% of analysed candidates over any
  20 trading days (the data, not the market, is deciding);
- a change in section 7 made without an amendment.

**Throughput:** if, after two full earnings seasons, the claimable rate puts
the final decision more than **DECIDE:** 18 months away, stop and redesign
(universe, data or structures) as a new pre-registered cohort, rather than
loosening gates inside this one.

## 7. Frozen for the life of the cohort

Any change to the following starts a new cohort (with an amendment), because it
can change which events are traded or how they are priced:

- the claim contract (`CONTRACT_VERSION`);
- the selector: its gates and thresholds, structure scorecards, eligibility
  rules, and the models feeding them;
- the universe and its selection rule;
- the data provider for chains and quotes;
- the schedule, entry window (DTE) and exit rule (T-1);
- the baselines and the analysis in section 5.

Changes that cannot alter which events are traded or how they are priced (bug
fixes in reporting, monitoring, deployment) are allowed; the commit SHA in each
report's `code_identity` shows exactly what ran.

## 8. Reporting

Every decision point is reported whatever the result, including no edge and
stopping, with: the frozen cohort's name and member hash, the commit range that
produced it, the counts excluded by reason, both primary results with intervals,
the secondary comparisons, and the fragility check.

## 9. Amendments

Append only. Each entry: date, what changes, why, and whether it starts a new
cohort.

_None yet._

## Sign-off

| | |
|---|---|
| Owner | |
| Date signed | |
| Start date | |
| Commit running at start (`/api/version`) | |
| Universe rule and size | |
| Decisions above (each **DECIDE**) | |
