# Calendar Picker Promotion Criteria — PR-AC

**Status**: Provisional. **Last reviewed**: 2026-05-27.

This document defines the governance rules for moving the
`candidate_min_dte` calendar leg-picker rule from experimental shadow
mode to production default. It is intentionally **conservative** about
what is measurable today versus what would require additional
infrastructure work before being evaluable.

---

## TL;DR

PR-AC records *which contracts each picker would have chosen* at trade
entry. It does **not** record *what those contracts would have realized
at exit* for the candidate picker. Therefore:

- **Outcome-based criteria** (candidate mean return, candidate win rate)
  are **not yet measurable**. Treating them as actionable today would
  recreate the research-leakage failure mode PR-AC was designed to
  prevent.
- **Selection-quality criteria** (divergence rate, coverage, stability)
  **are measurable today**. They can establish that the rule is
  well-defined and produces sensible contracts — but they do **not**
  prove it makes money.

A full promotion decision requires the outcome-based criteria. Until
candidate shadow outcome resolution is built (see _Prerequisites_
below), promotion-by-fiat is not allowed under this governance.

---

## Background

The `candidate_min_dte` rule (`CANDIDATE_FRONT_MIN_DTE_DAYS = 14`) was
discovered in PR-AB by sweeping holding periods across 10 symbols × 139
earnings events × 3 horizons × 4 structures and selecting the
configuration with the best in-sample mean return. On that dataset,
requiring a ≥14-day front DTE improved `put_calendar` mean return from
+12% to +23% with win rate 61% → 73%.

PR-AC introduces a dual-path shadow logger so the same rule can be
evaluated on **future** events without ever changing trading behavior.
The empirical claim "the rule lifts returns by ~11pp" is in-sample and
not validated; the system must show evidence on out-of-sample events
before the candidate becomes default.

---

## What is measurable today

After PR-AC commit 4, every recorded recommendation carries a
`picker_provenance_json` block with:

- `picker_variant` for both legacy and candidate selections
- `front_expiry`, `back_expiry`, `strike` for each
- `pickers_diverged` boolean
- `candidate_min_front_dte_days` threshold used
- Quote details (mid/bid/ask/iv/spread/OI/volume) for each leg

Aggregated across observations, this supports the following measures:

### Selection-quality measures (measurable today)

| Measure | Definition | Stable threshold for promotion-eligibility |
|---|---|---|
| `divergence_rate` | `pickers_diverged_count / picker_evaluated_count` (event-level) | **≥ 30%** — if pickers agree on 70%+ of events, the rule is essentially a no-op and divergent comparisons would be sample-thin. |
| `candidate_coverage` | `candidate_succeeded_count / legacy_succeeded_count` (event-level) | **≥ 80%** — the candidate must successfully pick contracts in at least 80% of events where legacy also picked. Below 80% means the +14d DTE floor disqualifies too many real-world setups. |
| `selection_stability` | Same event, same chain → same selection across re-runs | **100%** — the picker must be deterministic. Any randomness invalidates outcome comparisons. |

These are **pre-promotion gates**. Passing them does not prove the rule
makes money. Failing them rules the rule out as too unstable / too
narrow / too redundant to be worth promoting.

---

## What is NOT measurable today

The following criteria **cannot** be evaluated with current data
collection. Writing them as actionable thresholds would create a
governance hole exactly the size of the research-leakage problem
PR-AC was designed to prevent.

### Outcome-quality criteria (NOT measurable until prerequisites land)

- **Candidate mean return** — requires simulating the candidate picker's
  selections through entry + exit pricing on each historical event. The
  PR-AB sweep did this in-sample only; the live ledger does not.
- **Candidate win rate** — same prerequisite as mean return.
- **Per-quartile (or any conditional) candidate-vs-legacy comparison**
  — same prerequisite.

### Why not just use the PR-AB sweep numbers?

Because the PR-AB sweep was the data **on which the rule was
discovered**. Reusing it to "validate" the rule is the textbook
multiple-testing failure mode (Harvey & Liu, "Evaluating Trading
Strategies"). Out-of-sample evidence means evidence collected on events
the rule did not see during development — i.e., events after the PR-AC
merge date.

---

## Prerequisites for evaluating outcome-based criteria

These are concrete code changes that would need to land in a separate
PR before any return-based or win-rate-based promotion threshold can be
applied:

1. **Candidate shadow simulation through historical events.** Extend
   `_simulate_pre_earnings_calendar_trade` (or write a parallel
   simulator) to also use the candidate picker's selection and look up
   that selection's exit pricing in the exit chain. Produce a
   `candidate_pnl` per event.

2. **Schema extension.** Add `candidate_realized_return_pct` and
   `candidate_pickers_outcome_resolved: bool` to the persisted
   evidence (either as JSON keys inside `picker_provenance_json` or as
   dedicated columns).

3. **Forward-data accumulation period.** After (1) and (2) merge,
   accumulate observations on events **after that merge date**. The
   PR-AB sweep's 139 events do not count because they are in-sample.

4. **Put-side parity.** PR-AC commit 2 only queries the calls chain in
   the historical backtest. `put_calendar` evidence is currently
   absent from the dual-path logger by design (Codex review). Promoting
   a `candidate_min_dte` rule that applies to put_calendar requires
   put-side simulation parity first.

Only after all four prerequisites land can the outcome-based criteria
below be evaluated honestly.

---

## Proposed outcome-based criteria (NOT YET EVALUABLE)

These are documented for future use. They are **not actionable today**
because the data they reference does not yet exist in the ledger.

| Criterion | Threshold | Rationale |
|---|---|---|
| Forward sample size (call_calendar) | `unique_events_picker_evaluated ≥ 40` | Minimum sample for non-trivial outcome stats. Per Harvey/Liu and SR 11-7 model-validation guidance, ≥40 distinct out-of-sample events is the smallest defensible threshold for a one-rule comparison. |
| Forward sample size (put_calendar) | Same threshold, evaluated separately after put-side parity lands | Calls and puts are different distributions — combining them is statistical sleight of hand. |
| Mean-return improvement | `candidate_mean_return_pct - legacy_mean_return_pct ≥ 3pp` | Smaller than the PR-AB in-sample lift (+11pp). Sets a deliberately lower out-of-sample bar so we don't require exact replication of the in-sample magnitude. |
| Win rate | `candidate_win_rate ≥ legacy_win_rate` | Direction matters more than magnitude. A non-decrease is the lowest defensible bar. |
| Rolling-window check | Both conditions above must hold on the most recent 20-event rolling window in addition to the cumulative sample | Guards against the rule working only in one specific market regime. |

All four conditions must be satisfied for promotion. Any single
condition failing → no promotion. A passing event count alone (e.g.,
"n ≥ 40, no other condition checked") is **not** sufficient.

---

## Promotion mechanics

When (and only when) all four outcome-based conditions above are met:

1. The reviewer changes the default `picker_variant` used by the
   public-facing recommendation surface from `legacy_first_expiry` to
   `candidate_min_dte`. **`legacy_first_expiry` is not removed** — it
   remains available so the shadow comparison can continue running
   in the *other* direction.
2. A new shadow comparison begins: the (newly-promoted) candidate
   becomes the canonical baseline, and any future picker variant must
   beat it on the same out-of-sample protocol.
3. The promotion event itself is recorded in a new `picker_promotion`
   row in the ledger, including the cumulative + rolling-window stats
   at the moment of promotion. This is the audit record.

---

## Revert mechanics

After promotion, if the post-promotion rolling-window check ever fails
(candidate underperforms legacy on the most recent 20 events), the
default is **reverted** to `legacy_first_expiry`. The revert is
automatic, not discretionary. The revert event is also recorded in the
ledger.

This is the SR 11-7 ongoing-monitoring principle: a model is not
"validated and done"; validation continues for the life of the model.

---

## Anti-patterns

The following actions are **explicitly forbidden** under this
governance:

- **In-sample re-validation.** Using the PR-AB 139-event sweep, or
  any subset of it, to argue the candidate is ready for promotion.
  That data was the discovery set; it is not a validation set.
- **Sample stitching.** Combining the in-sample PR-AB events with
  post-merge forward events into a single n-count. The thresholds in
  this document refer exclusively to events whose `created_at`
  timestamp is **after the PR-AC merge commit**.
- **Threshold drift.** Lowering the mean-return improvement threshold
  below 3pp, or lowering the sample-size threshold below 40, without
  publishing a new version of this document explaining why and
  acknowledging the multiple-testing implications.
- **Outcome reading from selection alone.** Treating a high
  `divergence_rate` or `candidate_coverage` as outcome evidence.
  These are pre-promotion gates, not performance proxies.

---

## Open governance questions

These are deliberately left unresolved in this revision of the
document. They should be settled before the prerequisites work
begins:

- Should `put_calendar` promotion require its own parity-threshold
  satisfaction independent of `call_calendar`, or should they be
  bundled? (Current draft assumes independent.)
- Should the rolling-window check be 20 events, 30 events, or 50?
  Smaller windows are noisier; larger windows are slower to react.
- Should there be a minimum "elapsed calendar time" requirement
  (e.g., ≥6 months of forward data) in addition to the event count?

---

## Document history

- 2026-05-27: Initial version. Conservative scope per Codex review of
  PR-AC commit 4 — outcome-based criteria documented but explicitly
  marked NOT YET EVALUABLE.
