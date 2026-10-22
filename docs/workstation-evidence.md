# Workstation evidence export

`POST /api/workstation/evidence/options-edge` accepts the same body as
`POST /api/edge/analyze` and returns `workstation.evidence.v1`. It uses the
same authenticated API surface, analyzer/provider path, and `analyze` rate-limit
bucket. The native analysis is performed once per request.

## Contract boundaries

- The common envelope and its typed nested models are strict, frozen Pydantic
  models with unknown fields forbidden. `human_decision` is constrained to
  `null`.
- `input_digest` binds the evidence to the complete non-secret analysis input.
  `EdgeAnalyzeRequest` currently contains only `symbol`; authentication is
  transport middleware state and is not included in the request model or hash.
- `native_payload_schema` is the versioned wire ID
  `options-calculator-pro.edge-analyze.v1`. `native_payload` is an opaque JSON
  object containing the complete serialized `EdgeAnalyzeResponse`; consumers
  should interpret it only according to that wire ID.
- A ticker is only a producer-local identifier. The subject is exported as a
  `security` with `{scheme: ticker, value: <symbol>}`,
  `canonical_identifier=null`, and `mapping_status=unresolved`; entity
  resolution belongs to a separate system.
- The envelope is an `analytical_signal`, not a human decision. Native
  `confidence_pct` remains confidence. The model's `setup_score` is explicitly
  labeled under `uncertainty.model_scores` rather than being presented as
  confidence.

## Integrity and serialization profile

OCP computes both digests with this producer-local deterministic profile:

1. Pydantic `model_dump(mode="json")` values.
2. UTF-8 JSON with object keys sorted, separators `,` and `:`, and non-ASCII
   characters emitted directly.
3. JSON `NaN`, positive infinity, and negative infinity are rejected.
4. `input_digest` is `sha256:<lowercase hex>` over the analysis-input object.
5. `evidence_id` is `sha256:<lowercase hex>` over every final envelope field
   except `evidence_id` itself.

This is not an RFC 8785 claim. Consumers must treat `evidence_id` as an opaque
producer-issued integrity identifier rather than independently reproducing it
across languages. On model validation, OCP recomputes the ID and rejects stale
or tampered envelopes.

The typed models are frozen, but Python dictionaries and lists inside the
opaque native JSON payload are not recursively immutable. Persisted/serialized
envelopes must therefore be treated as immutable evidence records. Any parsed
record should be revalidated before trust; changing nested payload content while
retaining the old ID fails validation.

## Point-in-time use

`generated_at` must be timezone-aware and is normalized to UTC in the common
temporal context. `as_of` uses a native snapshot date when one is present and
records the exact source field in `as_of_basis`; otherwise it uses the
UTC-normalized generated-at date and labels that fallback explicitly.
`event_date` is optional and is emitted only when the native result contains an
earnings date. Fiscal labels and first/last period ends are unused by this OCP
signal and are emitted as an empty list and `null` values.

The export does not make current data historically point-in-time safe:
consumers must preserve the envelope and source snapshots used for the original
analysis rather than re-fetching current data later. Source observations are
facts, the native edge analysis and envelope are signals, and any portfolio
action remains a human or separately governed decision-system responsibility.
