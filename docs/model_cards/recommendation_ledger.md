# Model Card: Recommendation Ledger

## Purpose
The recommendation ledger records every selector recommendation and its evidence surface for auditability.

## Inputs
VolSnapshot, StructureScorecards, SelectorOutput, provider names, quote provenance, stale-source flags, bid/ask/mid context, and schema metadata.

## Outputs
Durable SQLite recommendation records with IDs that can link to forward paper trades and outcomes.

## Limitations
The ledger records what the system knew at recommendation time. It does not validate whether a user acted or received executable fills.

## Known Failure Modes
If upstream data is wrong, the ledger preserves that wrong state. Missing quote provenance can limit later audit quality.

## Data Dependencies
Selector output, snapshot data, provider metadata, and local SQLite persistence.

## Validation Approach
Tests cover record writing, stale/provenance preservation, no-trade records, schema migration, exports, and linkage.

## Do Not Infer
Do not infer that a ledger record is a trade, live fill, or performance claim.
