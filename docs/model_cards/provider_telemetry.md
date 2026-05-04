# Model Card: Provider Telemetry

## Purpose
Provider telemetry observes request health for market-data, earnings, quote, and fallback paths.

## Inputs
Provider name, endpoint type, symbol, timestamp, success/failure, error category, latency, fallback use, stale use, and response-quality notes.

## Outputs
Recent events, provider summaries, failure rates, average latency, fallback counts, stale counts, retention metadata, and operational warnings.

## Limitations
Telemetry is best-effort. Failures to write telemetry must not block analysis.

## Known Failure Modes
Local SQLite files can grow without retention, provider errors can be underclassified, and missing instrumentation can leave blind spots.

## Data Dependencies
Instrumented provider call sites and local SQLite persistence.

## Validation Approach
Tests cover recording, filtering, pagination, cleanup, provider summaries, and warning logic.

## Do Not Infer
Do not infer that provider telemetry measures market correctness. It measures request health and fallback behavior.
