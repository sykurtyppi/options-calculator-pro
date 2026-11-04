"""Quote provenance for fake analyses in forward-loop tests.

The forward loop refuses to price evidence from an analysis whose quotes were
fetched outside the NYSE session, are stale, or have no provenance
(services.quote_provenance). A test standing in for the analyser supplies
this in-session record unless it is testing that gate.
"""
IN_SESSION_QUOTES = {
    "version": 1,
    "quote_source": "marketdata",
    "fetched_at": "2026-11-04T16:45:00+00:00",
    "quote_timestamp": "2026-11-04T16:44:00+00:00",
    "timestamp_basis": "provider_quote_time",
    "quote_age_minutes": 1.0,
    "oldest_quote_timestamp": "2026-11-04T16:43:00+00:00",
    "contracts_with_quote_time": 40,
    "in_regular_session": True,
    "quote_session": "2026-11-04",
}
