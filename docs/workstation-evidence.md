# Workstation evidence export

`POST /api/workstation/evidence/options-edge` accepts the same body as
`POST /api/edge/analyze` and returns `workstation.evidence.v1`. It uses the
same authenticated API surface, analyzer/provider path, and `analyze` rate-limit
bucket. The native analysis is performed once per request.

## Contract boundaries

- `native_payload` is the complete `EdgeAnalyzeResponse`; consumers should use
  `native_payload_schema` when interpreting it.
- `evidence_id` is `sha256:<hex>` over canonical JSON for every envelope field
  except `evidence_id`. Any payload or metadata change therefore changes the ID.
- A ticker is only a producer-local identifier. The subject is always exported
  with `scheme=ticker`, `canonical_identifier=null`, and
  `mapping_status=unresolved`; entity resolution belongs to a separate system.
- The envelope is an `analytical_signal`, not a human decision. It carries
  confidence, uncertainty, provenance, invalidation conditions, and blocking
  quality reasons only when those values exist in the native result.
  `human_decision` is constrained to `null`.

## Point-in-time use

`generated_at` records when the native API response was built. `as_of` uses a
native snapshot date when one is present and records the exact source field in
`as_of_basis`; otherwise it uses the generated-at date and labels that fallback
explicitly. `event_date` is optional and is emitted only when the native result
contains an earnings date. The export does not make current data historically
point-in-time safe: consumers must preserve the envelope and source snapshots
used for the original analysis rather than re-fetching current data later.

This separation is intentional: source observations are facts, the native edge
analysis and envelope are signals, and any portfolio action remains a human or
separately governed decision-system responsibility.
