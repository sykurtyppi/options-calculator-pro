from __future__ import annotations

import json
import os
import time
from datetime import date
from types import SimpleNamespace

import pandas as pd

from services.earnings_event_service import (
    EarningsEventCandidate,
    EarningsEventService,
    _redact_secret_text,
)


def test_provider_error_redaction_removes_query_credentials() -> None:
    message = (
        "500 Server Error for url: "
        "https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey=super-secret-token"
    )

    redacted = _redact_secret_text(message)

    assert "super-secret-token" not in redacted
    assert "apikey=<redacted>" in redacted


def test_alpha_vantage_bulk_calendar_is_cached_and_reused(tmp_path) -> None:
    call_count = {"alpha": 0}

    service = EarningsEventService(
        cache_dir=tmp_path,
        alpha_vantage_api_key="alpha-key",
    )

    def fake_alpha_loader(*, horizon: str):
        call_count["alpha"] += 1
        assert horizon == "3month"
        return [
            {"symbol": "AAPL", "reportDate": "2026-05-01"},
            {"symbol": "MSFT", "reportDate": "2026-05-02"},
        ]

    service._fetch_alpha_vantage_calendar = fake_alpha_loader  # type: ignore[method-assign]
    service._marketdata_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_confirmed_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_calendar_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._yfinance_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]

    first = service.resolve_upcoming_event("AAPL", date(2026, 4, 23), date(2026, 5, 31))
    second = service.resolve_upcoming_event("MSFT", date(2026, 4, 23), date(2026, 5, 31))

    assert first.primary_source == "alpha_vantage_calendar"
    assert first.earnings_date == date(2026, 5, 1)
    assert second.primary_source == "alpha_vantage_calendar"
    assert second.earnings_date == date(2026, 5, 2)
    assert call_count["alpha"] == 1


def test_fmp_confirmed_beats_weaker_calendar_sources(tmp_path) -> None:
    service = EarningsEventService(
        cache_dir=tmp_path,
        alpha_vantage_api_key="alpha-key",
        fmp_api_key="fmp-key",
    )

    service._marketdata_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_confirmed_candidates = lambda *args, **kwargs: [  # type: ignore[method-assign]
        EarningsEventCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 5, 1),
            release_timing="BMO",
            source="fmp_confirmed",
            source_rank=90,
            source_confidence=0.92,
            confirmed=True,
            detail="confirmed",
        )
    ]
    service._fmp_calendar_candidates = lambda *args, **kwargs: [  # type: ignore[method-assign]
        EarningsEventCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 5, 1),
            release_timing="UNKNOWN",
            source="fmp_calendar",
            source_rank=80,
            source_confidence=0.82,
        )
    ]
    service._alpha_vantage_candidates = lambda *args, **kwargs: [  # type: ignore[method-assign]
        EarningsEventCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 5, 2),
            release_timing="UNKNOWN",
            source="alpha_vantage_calendar",
            source_rank=70,
            source_confidence=0.72,
        )
    ]
    service._yfinance_candidates = lambda *args, **kwargs: [  # type: ignore[method-assign]
        EarningsEventCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 5, 1),
            release_timing="AMC",
            source="yfinance_calendar",
            source_rank=55,
            source_confidence=0.58,
        )
    ]

    resolved = service.resolve_upcoming_event("AAPL", date(2026, 4, 23), date(2026, 5, 31))

    assert resolved.primary_source == "fmp_confirmed"
    assert resolved.earnings_date == date(2026, 5, 1)
    assert resolved.release_timing == "BMO"
    assert resolved.confirmed_source == "fmp_calendar"
    assert resolved.source_confidence > 0.92


def test_yfinance_calendar_remains_last_resort_fallback(tmp_path) -> None:
    service = EarningsEventService(cache_dir=tmp_path)
    service._marketdata_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_confirmed_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_calendar_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._alpha_vantage_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]

    ticker = SimpleNamespace(
        info={"earningsTimestamp": 1777669200},
        calendar={"Earnings Date": [pd.Timestamp("2026-05-01T20:00:00Z")]},
    )

    resolved = service.resolve_upcoming_event(
        "AAPL",
        date(2026, 4, 23),
        date(2026, 5, 31),
        ticker=ticker,
    )

    assert resolved.primary_source == "yfinance_calendar"
    assert resolved.earnings_date == date(2026, 5, 1)
    assert resolved.release_timing == "AMC"


def test_sec_confirmation_boosts_confidence_without_pretending_to_be_calendar_truth(tmp_path) -> None:
    service = EarningsEventService(cache_dir=tmp_path, sec_user_agent="tester@example.com app/1.0")
    service._marketdata_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_confirmed_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_calendar_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._alpha_vantage_candidates = lambda *args, **kwargs: [  # type: ignore[method-assign]
        EarningsEventCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 5, 1),
            release_timing="UNKNOWN",
            source="alpha_vantage_calendar",
            source_rank=70,
            source_confidence=0.72,
        )
    ]
    service._yfinance_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._lookup_sec_cik = lambda symbol: "0000320193"  # type: ignore[method-assign]
    service._sec_submissions = lambda cik: {  # type: ignore[method-assign]
        "filings": {
            "recent": {
                "form": ["8-K"],
                "filingDate": ["2026-05-01"],
                "acceptanceDateTime": ["2026-05-01T20:05:00.000Z"],
            }
        }
    }

    resolved = service.resolve_upcoming_event("AAPL", date(2026, 4, 23), date(2026, 5, 31))

    assert resolved.primary_source == "alpha_vantage_calendar"
    assert resolved.confirmed_source == "sec_submissions"
    assert resolved.release_timing == "AMC"
    assert resolved.release_timing_source == "sec_submissions"
    assert resolved.source_confidence > 0.72
    assert any("SEC EDGAR corroborated" in note for note in resolved.source_notes)


def test_sec_confirmation_is_disabled_without_configured_user_agent(tmp_path) -> None:
    service = EarningsEventService(cache_dir=tmp_path, sec_user_agent="")
    service._marketdata_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_confirmed_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_calendar_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._alpha_vantage_candidates = lambda *args, **kwargs: [  # type: ignore[method-assign]
        EarningsEventCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 5, 1),
            release_timing="UNKNOWN",
            source="alpha_vantage_calendar",
            source_rank=70,
            source_confidence=0.72,
        )
    ]
    service._yfinance_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    called = {"lookup": False}

    def _lookup(_symbol: str) -> str:
        called["lookup"] = True
        return "0000320193"

    service._lookup_sec_cik = _lookup  # type: ignore[method-assign]

    resolved = service.resolve_upcoming_event("AAPL", date(2026, 4, 23), date(2026, 5, 31))

    assert resolved.primary_source == "alpha_vantage_calendar"
    assert resolved.confirmed_source is None
    assert resolved.source_confidence == 0.72
    assert called["lookup"] is False


def test_stale_alpha_vantage_cache_is_marked_and_confidence_downgraded(tmp_path) -> None:
    service = EarningsEventService(
        cache_dir=tmp_path,
        alpha_vantage_api_key="alpha-key",
    )
    service._marketdata_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_confirmed_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_calendar_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._yfinance_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fetch_alpha_vantage_calendar = lambda *, horizon: (_ for _ in ()).throw(RuntimeError("provider down"))  # type: ignore[method-assign]

    cache_path = tmp_path / "alpha_vantage_earnings_3month.json"
    cache_path.write_text(json.dumps([{"symbol": "AAPL", "reportDate": "2026-05-01"}]))
    old = time.time() - (3 * 86400)
    os.utime(cache_path, (old, old))

    resolved = service.resolve_upcoming_event("AAPL", date(2026, 4, 23), date(2026, 5, 31))

    assert resolved.primary_source == "alpha_vantage_calendar"
    assert resolved.earnings_date == date(2026, 5, 1)
    assert resolved.source_stale is True
    assert resolved.source_confidence == 0.54
    assert any("stale local cache" in note for note in resolved.source_notes)


def test_sec_confirmation_is_absent_when_no_nearby_filings_exist(tmp_path) -> None:
    service = EarningsEventService(cache_dir=tmp_path, sec_user_agent="tester@example.com app/1.0")
    service._marketdata_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_confirmed_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_calendar_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._alpha_vantage_candidates = lambda *args, **kwargs: [  # type: ignore[method-assign]
        EarningsEventCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 5, 1),
            release_timing="UNKNOWN",
            source="alpha_vantage_calendar",
            source_rank=70,
            source_confidence=0.72,
        )
    ]
    service._yfinance_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._lookup_sec_cik = lambda symbol: "0000320193"  # type: ignore[method-assign]
    service._sec_submissions = lambda cik: {  # type: ignore[method-assign]
        "filings": {"recent": {"form": ["8-K"], "filingDate": ["2026-05-10"], "acceptanceDateTime": ["2026-05-10T20:05:00.000Z"]}}
    }

    resolved = service.resolve_upcoming_event("AAPL", date(2026, 4, 23), date(2026, 5, 31))

    assert resolved.primary_source == "alpha_vantage_calendar"
    assert resolved.confirmed_source is None
    assert resolved.source_confidence == 0.72


def test_unresolved_event_is_explicit(tmp_path) -> None:
    service = EarningsEventService(cache_dir=tmp_path)
    service._marketdata_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_confirmed_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._fmp_calendar_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._alpha_vantage_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]
    service._yfinance_candidates = lambda *args, **kwargs: []  # type: ignore[method-assign]

    resolved = service.resolve_upcoming_event("AAPL", date(2026, 4, 23), date(2026, 5, 31))

    assert resolved.earnings_date is None
    assert resolved.primary_source == "unresolved"
    assert resolved.source_confidence == 0.0
