from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd
import requests
import yfinance as yf

from services import external_io_gate
from services.market_data_client import MarketDataClient
from services.provider_telemetry import classify_error, record_provider_telemetry

logger = logging.getLogger(__name__)

_SECRET_QUERY_RE = re.compile(r"([?&](?:apikey|api_key|token|key)=)([^&\s]+)", re.IGNORECASE)


def _redact_secret_text(value: Any) -> str:
    """Redact query-string credentials before classifying or logging provider errors."""
    return _SECRET_QUERY_RE.sub(r"\1<redacted>", str(value))

_ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
_FMP_BASE_URL = "https://financialmodelingprep.com/stable"
_FMP_LEGACY_BASE_URL = "https://financialmodelingprep.com/api/v4"
_SEC_SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"
_SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_DEFAULT_CACHE_DIR = Path.home() / ".options_calculator_pro" / "cache" / "earnings_events"

_ALPHA_VANTAGE_KEY_ENV = "ALPHA_VANTAGE_API_KEY"
_FMP_KEY_ENV = "FMP_API_KEY"
_SEC_USER_AGENT_ENV = "SEC_API_USER_AGENT"
_SEC_CONFIRMATION_FORMS = {"8-K", "10-Q", "10-K", "6-K", "20-F", "40-F"}
_SEC_CONFIRMATION_WINDOW_DAYS = 1


@dataclass(frozen=True)
class EarningsEventCandidate:
    symbol: str
    earnings_date: date
    release_timing: str
    source: str
    source_rank: int
    source_confidence: float
    confirmed: bool = False
    detail: Optional[str] = None
    stale_cache: bool = False


@dataclass(frozen=True)
class ResolvedEarningsEvent:
    symbol: str
    earnings_date: Optional[date]
    release_timing: str
    primary_source: str
    confirmed_source: Optional[str]
    source_confidence: float
    source_notes: list[str]
    release_timing_source: Optional[str] = None
    source_stale: bool = False


def _normalize_release_timing(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip().lower()
    if not text:
        return "UNKNOWN"
    if text in {"amc", "after market close", "after-hours", "after hours"}:
        return "AMC"
    if text in {"bmo", "before market open", "pre-market", "pre market"}:
        return "BMO"
    if text in {"dmh", "during market hours", "during market"}:
        return "DMH"
    if text in {"unknown", "tbd", "time not supplied"}:
        return "UNKNOWN"
    if ":" in text:
        try:
            parts = text.replace("am", " am").replace("pm", " pm")
            parsed = datetime.strptime(parts.strip(), "%I:%M %p")
            if parsed.hour < 9 or (parsed.hour == 9 and parsed.minute < 30):
                return "BMO"
            if parsed.hour >= 16:
                return "AMC"
            return "DMH"
        except ValueError:
            pass
    return "UNKNOWN"


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date()


def _extract_calendar_dates(raw_calendar: Any) -> list[date]:
    values: list[Any] = []
    if raw_calendar is None:
        return []
    if isinstance(raw_calendar, dict):
        values = raw_calendar.get("Earnings Date", []) or []
    elif isinstance(raw_calendar, pd.DataFrame):
        if "Earnings Date" in raw_calendar.index:
            raw = raw_calendar.loc["Earnings Date"]
            values = raw.tolist() if isinstance(raw, pd.Series) else [raw]
        elif "Earnings Date" in raw_calendar.columns:
            values = raw_calendar["Earnings Date"].tolist()
    elif hasattr(raw_calendar, "tolist"):
        values = list(raw_calendar.tolist())
    elif isinstance(raw_calendar, Iterable):
        values = list(raw_calendar)

    parsed = [_parse_date(value) for value in values]
    return sorted({item for item in parsed if item is not None})


def _infer_release_timing_from_info(info: dict[str, Any]) -> str:
    ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
    if ts is None:
        return "UNKNOWN"
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return "UNKNOWN"
    if dt.hour >= 20:
        return "AMC"
    if dt.hour < 16:
        return "BMO"
    return "UNKNOWN"


def _infer_release_timing_from_acceptance(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    if not text:
        return "UNKNOWN"
    try:
        accepted = pd.to_datetime(text, utc=True, errors="coerce")
    except Exception:
        accepted = pd.NaT
    if pd.isna(accepted):
        return "UNKNOWN"
    accepted_dt = accepted.to_pydatetime()
    if accepted_dt.hour >= 20:
        return "AMC"
    if accepted_dt.hour < 16:
        return "BMO"
    return "UNKNOWN"


def _timing_label_to_full(label: str) -> str:
    if label == "AMC":
        return "after market close"
    if label == "BMO":
        return "before market open"
    if label == "DMH":
        return "during market hours"
    return "unknown"


class EarningsEventService:
    """Resolve upcoming earnings events with explicit source ranking and cheap caching.

    Rationale:
    - Alpha Vantage free keys are useful for one cached bulk calendar pull, not for
      high-frequency per-symbol polling.
    - FMP can provide calendar and confirmed-event context when available.
    - MarketData.app remains the highest-quality source if the entitlement exists.
    - yfinance is kept as the research-grade fallback, not the preferred truth source.
    """

    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        alpha_vantage_api_key: Optional[str] = None,
        fmp_api_key: Optional[str] = None,
        sec_user_agent: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
        self.alpha_vantage_api_key = (
            os.environ.get(_ALPHA_VANTAGE_KEY_ENV, "").strip()
            if alpha_vantage_api_key is None
            else str(alpha_vantage_api_key).strip()
        )
        self.fmp_api_key = (
            os.environ.get(_FMP_KEY_ENV, "").strip()
            if fmp_api_key is None
            else str(fmp_api_key).strip()
        )
        configured_sec_user_agent = (
            os.environ.get(_SEC_USER_AGENT_ENV, "").strip()
            if sec_user_agent is None
            else str(sec_user_agent).strip()
        )
        self._sec_enabled = bool(configured_sec_user_agent)
        self.sec_user_agent = configured_sec_user_agent or "options-calculator-pro/1.0 contact-required"
        self._session = session or requests.Session()
        self._stale_cache_keys: set[str] = set()

    def resolve_upcoming_event(
        self,
        symbol: str,
        as_of: date,
        cutoff: date,
        *,
        ticker: Optional[yf.Ticker] = None,
        mda_client: Optional[MarketDataClient] = None,
    ) -> ResolvedEarningsEvent:
        candidates: list[EarningsEventCandidate] = []
        notes: list[str] = []

        candidates.extend(self._marketdata_candidates(symbol, as_of, cutoff, mda_client))
        candidates.extend(self._fmp_confirmed_candidates(symbol, as_of, cutoff))
        candidates.extend(self._fmp_calendar_candidates(symbol, as_of, cutoff))
        candidates.extend(self._alpha_vantage_candidates(symbol, as_of, cutoff))
        candidates.extend(self._yfinance_candidates(symbol, as_of, cutoff, ticker=ticker))
        if not candidates:
            return ResolvedEarningsEvent(
                symbol=symbol.upper(),
                earnings_date=None,
                release_timing="UNKNOWN",
                primary_source="unresolved",
                confirmed_source=None,
                source_confidence=0.0,
                source_notes=["No upcoming earnings event was found across configured sources."],
                release_timing_source=None,
            )

        sec_confirmations = self._sec_confirmation_candidates(symbol, candidates)
        candidates.sort(
            key=lambda item: (
                -item.source_rank,
                -item.source_confidence,
                item.earnings_date.toordinal(),
                item.source,
            )
        )
        primary = candidates[0]
        corroborating = next(
            (
                candidate
                for candidate in candidates[1:]
                if abs((candidate.earnings_date - primary.earnings_date).days) <= 1
            ),
            None,
        )
        sec_confirmation = next(
            (
                candidate
                for candidate in sec_confirmations
                if abs((candidate.earnings_date - primary.earnings_date).days) <= 1
            ),
            None,
        )
        confirmed_source = (
            sec_confirmation.source
            if sec_confirmation is not None
            else (corroborating.source if corroborating else (primary.source if primary.confirmed else None))
        )
        confidence = primary.source_confidence
        if corroborating is not None:
            confidence = min(0.99, confidence + 0.08)
            notes.append(
                f"Primary event date corroborated by {corroborating.source} within one calendar day."
            )
        if sec_confirmation is not None:
            confidence = min(0.99, confidence + 0.06)
            notes.append(
                "SEC EDGAR corroborated the selected event date; SEC is used as confirmation, not the primary calendar source."
            )
            if sec_confirmation.detail:
                notes.append(sec_confirmation.detail)
        release_timing = primary.release_timing
        release_timing_source = primary.source if primary.release_timing != "UNKNOWN" else confirmed_source
        if release_timing == "UNKNOWN" and sec_confirmation is not None and sec_confirmation.release_timing != "UNKNOWN":
            release_timing = sec_confirmation.release_timing
            release_timing_source = sec_confirmation.source
        source_stale = bool(primary.stale_cache)
        if primary.detail:
            notes.append(primary.detail)
        if source_stale:
            notes.append(
                "Primary earnings source came from a stale local cache after the provider refresh failed; confidence was downgraded."
            )
        notes.append(
            "Source ranking favors direct earnings feeds first, then bulk calendars, then research-grade fallbacks."
        )
        return ResolvedEarningsEvent(
            symbol=symbol.upper(),
            earnings_date=primary.earnings_date,
            release_timing=release_timing,
            primary_source=primary.source,
            confirmed_source=confirmed_source,
            source_confidence=round(confidence, 4),
            source_notes=notes,
            release_timing_source=release_timing_source,
            source_stale=source_stale,
        )

    def _marketdata_candidates(
        self,
        symbol: str,
        as_of: date,
        cutoff: date,
        mda_client: Optional[MarketDataClient],
    ) -> list[EarningsEventCandidate]:
        if not mda_client or not mda_client.is_available():
            return []
        try:
            earnings_df = mda_client.get_earnings(symbol, countback=24)
        except Exception as exc:
            logger.debug("MarketData earnings lookup failed for %s: %s", symbol, exc)
            return []
        if earnings_df is None or earnings_df.empty:
            return []

        candidates: list[EarningsEventCandidate] = []
        for _, row in earnings_df.iterrows():
            event_date = row.get("report_date") or row.get("event_date")
            parsed = _parse_date(event_date)
            if parsed is None or parsed < as_of or parsed > cutoff:
                continue
            candidates.append(
                EarningsEventCandidate(
                    symbol=symbol.upper(),
                    earnings_date=parsed,
                    release_timing=_normalize_release_timing(row.get("reportTime")),
                    source="marketdata",
                    source_rank=100,
                    source_confidence=0.95,
                    confirmed=True,
                    detail="MarketData.app earnings feed returned a directly timed event.",
                )
            )
        return candidates

    def _fmp_confirmed_candidates(self, symbol: str, as_of: date, cutoff: date) -> list[EarningsEventCandidate]:
        if not self.fmp_api_key:
            return []
        cache_key = f"fmp_confirmed_{as_of.isoformat()}_{cutoff.isoformat()}.json"
        rows = self._load_cached_json(
            cache_key=cache_key,
            ttl_seconds=3600.0,
            loader=lambda: self._fetch_fmp_confirmed_json(
                "/earning-calendar-confirmed",
                params={"from": as_of.isoformat(), "to": cutoff.isoformat()},
            ),
        )
        stale_cache = cache_key in self._stale_cache_keys
        if not isinstance(rows, list):
            return []
        candidates: list[EarningsEventCandidate] = []
        for row in rows:
            if str(row.get("symbol", "")).upper() != symbol.upper():
                continue
            parsed = _parse_date(row.get("date"))
            if parsed is None or parsed < as_of or parsed > cutoff:
                continue
            candidates.append(
                EarningsEventCandidate(
                    symbol=symbol.upper(),
                    earnings_date=parsed,
                    release_timing=_normalize_release_timing(row.get("time")),
                    source="fmp_confirmed",
                    source_rank=90,
                    source_confidence=0.92 if not stale_cache else 0.69,
                    confirmed=True,
                    detail=(
                        "FMP confirmed-earnings feed matched the requested date window."
                        if not stale_cache
                        else "Stale FMP confirmed-earnings cache used after refresh failure."
                    ),
                    stale_cache=stale_cache,
                )
            )
        return candidates

    def _fmp_calendar_candidates(self, symbol: str, as_of: date, cutoff: date) -> list[EarningsEventCandidate]:
        if not self.fmp_api_key:
            return []
        cache_key = f"fmp_calendar_{as_of.isoformat()}_{cutoff.isoformat()}.json"
        rows = self._load_cached_json(
            cache_key=cache_key,
            ttl_seconds=3600.0,
            loader=lambda: self._fetch_fmp_json(
                "/earnings-calendar",
                params={"from": as_of.isoformat(), "to": cutoff.isoformat()},
            ),
        )
        stale_cache = cache_key in self._stale_cache_keys
        if not isinstance(rows, list):
            return []
        candidates: list[EarningsEventCandidate] = []
        for row in rows:
            if str(row.get("symbol", "")).upper() != symbol.upper():
                continue
            parsed = _parse_date(row.get("date"))
            if parsed is None or parsed < as_of or parsed > cutoff:
                continue
            candidates.append(
                EarningsEventCandidate(
                    symbol=symbol.upper(),
                    earnings_date=parsed,
                    release_timing=_normalize_release_timing(row.get("time")),
                    source="fmp_calendar",
                    source_rank=80,
                    source_confidence=0.82 if not stale_cache else 0.61,
                    confirmed=False,
                    detail=(
                        "FMP earnings calendar supplied the upcoming event window."
                        if not stale_cache
                        else "Stale FMP earnings-calendar cache used after refresh failure."
                    ),
                    stale_cache=stale_cache,
                )
            )
        return candidates

    def _alpha_vantage_candidates(self, symbol: str, as_of: date, cutoff: date) -> list[EarningsEventCandidate]:
        if not self.alpha_vantage_api_key:
            return []
        horizon = "3month"
        days_forward = (cutoff - as_of).days
        if days_forward > 92:
            horizon = "6month"
        if days_forward > 183:
            horizon = "12month"
        cache_key = f"alpha_vantage_earnings_{horizon}.json"
        rows = self._load_cached_json(
            cache_key=cache_key,
            ttl_seconds=86400.0,
            loader=lambda: self._fetch_alpha_vantage_calendar(horizon=horizon),
        )
        stale_cache = cache_key in self._stale_cache_keys
        if not isinstance(rows, list):
            return []
        candidates: list[EarningsEventCandidate] = []
        for row in rows:
            if str(row.get("symbol", "")).upper() != symbol.upper():
                continue
            parsed = _parse_date(row.get("reportDate"))
            if parsed is None or parsed < as_of or parsed > cutoff:
                continue
            candidates.append(
                EarningsEventCandidate(
                    symbol=symbol.upper(),
                    earnings_date=parsed,
                    release_timing="UNKNOWN",
                    source="alpha_vantage_calendar",
                    source_rank=70,
                    source_confidence=0.72 if not stale_cache else 0.54,
                    confirmed=False,
                    detail=(
                        "Alpha Vantage bulk earnings calendar supplied the event date. "
                        "Free-tier rate limits make this best used as a cached discovery feed."
                        if not stale_cache
                        else "Stale Alpha Vantage earnings-calendar cache used after refresh failure."
                    ),
                    stale_cache=stale_cache,
                )
            )
        return candidates

    def _yfinance_candidates(
        self,
        symbol: str,
        as_of: date,
        cutoff: date,
        *,
        ticker: Optional[yf.Ticker],
    ) -> list[EarningsEventCandidate]:
        ticker = ticker or yf.Ticker(symbol)
        candidates: list[EarningsEventCandidate] = []
        start = time.perf_counter()
        try:
            info = ticker.info or {}
            record_provider_telemetry(
                provider_name="yfinance",
                endpoint_type="earnings_info",
                symbol=symbol,
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:
            record_provider_telemetry(
                provider_name="yfinance",
                endpoint_type="earnings_info",
                symbol=symbol,
                success=False,
                error_category=classify_error(_redact_secret_text(exc)),
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
            info = {}
        release_timing = _infer_release_timing_from_info(info)
        start = time.perf_counter()
        for event_date in _extract_calendar_dates(getattr(ticker, "calendar", None)):
            record_provider_telemetry(
                provider_name="yfinance",
                endpoint_type="earnings_calendar",
                symbol=symbol,
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                fallback_used=True,
                response_quality_note="research-grade earnings fallback",
            )
            if as_of <= event_date <= cutoff:
                candidates.append(
                    EarningsEventCandidate(
                        symbol=symbol.upper(),
                        earnings_date=event_date,
                        release_timing=release_timing,
                        source="yfinance_calendar",
                        source_rank=55,
                        source_confidence=0.58,
                        confirmed=False,
                        detail="yfinance calendar fallback provided the event date.",
                    )
                )
                return candidates

        get_dates = getattr(ticker, "get_earnings_dates", None)
        if callable(get_dates):
            start = time.perf_counter()
            try:
                earnings_df = get_dates(limit=12)
                record_provider_telemetry(
                    provider_name="yfinance",
                    endpoint_type="earnings_dates",
                    symbol=symbol,
                    success=earnings_df is not None and not earnings_df.empty,
                    error_category=None if earnings_df is not None and not earnings_df.empty else "empty_response",
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    fallback_used=True,
                    response_quality_note="research-grade earnings fallback",
                )
            except Exception as exc:
                record_provider_telemetry(
                    provider_name="yfinance",
                    endpoint_type="earnings_dates",
                    symbol=symbol,
                    success=False,
                    error_category=classify_error(_redact_secret_text(exc)),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    fallback_used=True,
                )
                earnings_df = None
            if earnings_df is not None and not earnings_df.empty:
                for idx in earnings_df.index:
                    parsed = _parse_date(idx)
                    if parsed is None or parsed < as_of or parsed > cutoff:
                        continue
                    idx_ts = pd.Timestamp(idx)
                    timing = "UNKNOWN"
                    if idx_ts.hour or idx_ts.minute:
                        timing = _normalize_release_timing(idx_ts.strftime("%I:%M %p"))
                    candidates.append(
                        EarningsEventCandidate(
                            symbol=symbol.upper(),
                            earnings_date=parsed,
                            release_timing=timing,
                            source="yfinance_earnings_dates",
                            source_rank=50,
                            source_confidence=0.54,
                            confirmed=False,
                            detail="yfinance earnings_dates fallback provided the event date.",
                        )
                    )
                    return candidates
        return candidates

    def _sec_confirmation_candidates(
        self,
        symbol: str,
        seed_candidates: list[EarningsEventCandidate],
    ) -> list[EarningsEventCandidate]:
        """Use SEC filings to corroborate a candidate date when EDGAR has nearby filings.

        SEC EDGAR is not a clean future earnings-calendar source. We only use it as
        a confirmation layer around already-discovered candidate dates, and only when
        a recent filing falls within a narrow tolerance window.
        """
        if not self._sec_enabled:
            return []
        if not seed_candidates:
            return []
        seed_dates = sorted({candidate.earnings_date for candidate in seed_candidates})
        timing_by_date: dict[date, str] = {}
        for candidate in sorted(seed_candidates, key=lambda item: (-item.source_rank, -item.source_confidence)):
            if candidate.release_timing == "UNKNOWN":
                continue
            timing_by_date.setdefault(candidate.earnings_date, candidate.release_timing)
        cik = self._lookup_sec_cik(symbol)
        if cik is None:
            return []
        submissions = self._sec_submissions(cik)
        if not isinstance(submissions, dict):
            return []
        recent = (submissions.get("filings") or {}).get("recent") or {}
        forms = list(recent.get("form", []) or [])
        filing_dates = list(recent.get("filingDate", []) or [])
        acceptance_times = list(recent.get("acceptanceDateTime", []) or [])

        candidates: list[EarningsEventCandidate] = []
        for seed_date in seed_dates:
            matched_timing = timing_by_date.get(seed_date, "UNKNOWN")
            matched_form: Optional[str] = None
            for idx, form in enumerate(forms):
                form_text = str(form or "").upper()
                if form_text not in _SEC_CONFIRMATION_FORMS:
                    continue
                filing_date = _parse_date(filing_dates[idx] if idx < len(filing_dates) else None)
                if filing_date is None:
                    continue
                if abs((filing_date - seed_date).days) > _SEC_CONFIRMATION_WINDOW_DAYS:
                    continue
                accepted = acceptance_times[idx] if idx < len(acceptance_times) else None
                inferred_timing = _infer_release_timing_from_acceptance(accepted)
                if inferred_timing != "UNKNOWN":
                    matched_timing = inferred_timing
                matched_form = form_text
                break
            if matched_form is None:
                continue
            candidates.append(
                EarningsEventCandidate(
                    symbol=symbol.upper(),
                    earnings_date=seed_date,
                    release_timing=matched_timing,
                    source="sec_submissions",
                    source_rank=85,
                    source_confidence=0.89,
                    confirmed=True,
                    detail=(
                        f"SEC EDGAR submissions showed a nearby {matched_form} filing within "
                        f"+/-{_SEC_CONFIRMATION_WINDOW_DAYS} day of the candidate earnings date."
                    ),
                )
            )
        return candidates

    def _fetch_alpha_vantage_calendar(self, *, horizon: str) -> list[dict[str, Any]]:
        external_io_gate.assert_allowed(external_io_gate.Category.EARNINGS_ALPHA_VANTAGE)
        start = time.perf_counter()
        try:
            response = self._session.get(
                _ALPHA_VANTAGE_BASE_URL,
                params={
                    "function": "EARNINGS_CALENDAR",
                    "horizon": horizon,
                    "apikey": self.alpha_vantage_api_key,
                },
                timeout=20.0,
            )
            response.raise_for_status()
            content = response.text.strip()
            rows = [dict(row) for row in csv.DictReader(content.splitlines())] if content else []
            record_provider_telemetry(
                provider_name="alpha_vantage",
                endpoint_type="earnings_calendar",
                success=bool(rows),
                error_category=None if rows else "empty_response",
                latency_ms=(time.perf_counter() - start) * 1000.0,
                response_quality_note=f"horizon={horizon}",
            )
            return rows
        except Exception as exc:
            record_provider_telemetry(
                provider_name="alpha_vantage",
                endpoint_type="earnings_calendar",
                success=False,
                error_category=classify_error(_redact_secret_text(exc)),
                latency_ms=(time.perf_counter() - start) * 1000.0,
                response_quality_note=f"horizon={horizon}",
            )
            raise

    def _fetch_fmp_json(self, endpoint: str, *, params: dict[str, Any]) -> Any:
        return self._fetch_json_with_telemetry(
            provider_name="fmp",
            endpoint_type="earnings_calendar",
            url=f"{_FMP_BASE_URL}{endpoint}",
            params={**params, "apikey": self.fmp_api_key},
        )

    def _fetch_fmp_confirmed_json(self, endpoint: str, *, params: dict[str, Any]) -> Any:
        return self._fetch_json_with_telemetry(
            provider_name="fmp",
            endpoint_type="earnings_confirmed",
            url=f"{_FMP_LEGACY_BASE_URL}{endpoint}",
            params={**params, "apikey": self.fmp_api_key},
        )

    def _lookup_sec_cik(self, symbol: str) -> Optional[str]:
        payload = self._load_cached_json(
            cache_key="sec_company_tickers.json",
            ttl_seconds=86400.0,
            loader=self._fetch_sec_company_tickers,
        )
        if not isinstance(payload, dict):
            return None
        target = symbol.upper()
        for item in payload.values():
            if str(item.get("ticker", "")).upper() != target:
                continue
            cik = item.get("cik_str")
            if cik is None:
                return None
            return f"{int(cik):010d}"
        return None

    def _sec_submissions(self, cik: str) -> Any:
        return self._load_cached_json(
            cache_key=f"sec_submissions_{cik}.json",
            ttl_seconds=1800.0,
            loader=lambda: self._fetch_sec_json(f"{_SEC_SUBMISSIONS_BASE_URL}/CIK{cik}.json"),
        )

    def _fetch_sec_company_tickers(self) -> Any:
        return self._fetch_sec_json(_SEC_COMPANY_TICKERS_URL)

    def _fetch_sec_json(self, url: str) -> Any:
        external_io_gate.assert_allowed(external_io_gate.Category.EARNINGS_SEC_EDGAR)
        endpoint_type = "sec_company_tickers" if "company_tickers" in url else "sec_submissions"
        start = time.perf_counter()
        try:
            response = self._session.get(
                url,
                headers={
                    "User-Agent": self.sec_user_agent,
                    "Accept-Encoding": "gzip, deflate",
                    "Host": "www.sec.gov" if "www.sec.gov" in url else "data.sec.gov",
                },
                timeout=20.0,
            )
            response.raise_for_status()
            payload = response.json()
            record_provider_telemetry(
                provider_name="sec_edgar",
                endpoint_type=endpoint_type,
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                response_quality_note="confirmation-only source",
            )
            return payload
        except Exception as exc:
            record_provider_telemetry(
                provider_name="sec_edgar",
                endpoint_type=endpoint_type,
                success=False,
                error_category=classify_error(_redact_secret_text(exc)),
                latency_ms=(time.perf_counter() - start) * 1000.0,
                response_quality_note="confirmation-only source",
            )
            raise

    def _fetch_json_with_telemetry(
        self,
        *,
        provider_name: str,
        endpoint_type: str,
        url: str,
        params: dict[str, Any],
    ) -> Any:
        external_io_gate.assert_allowed(external_io_gate.Category.EARNINGS_FMP)
        start = time.perf_counter()
        try:
            response = self._session.get(url, params=params, timeout=20.0)
            response.raise_for_status()
            payload = response.json()
            has_data = bool(payload)
            record_provider_telemetry(
                provider_name=provider_name,
                endpoint_type=endpoint_type,
                success=has_data,
                error_category=None if has_data else "empty_response",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
            return payload
        except Exception as exc:
            record_provider_telemetry(
                provider_name=provider_name,
                endpoint_type=endpoint_type,
                success=False,
                error_category=classify_error(_redact_secret_text(exc)),
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
            raise

    def _load_cached_json(
        self,
        *,
        cache_key: str,
        ttl_seconds: float,
        loader: Callable[[], Any],
    ) -> Any:
        cache_path = self.cache_dir / cache_key
        try:
            if cache_path.exists():
                age_seconds = datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime
                if age_seconds <= ttl_seconds:
                    self._stale_cache_keys.discard(cache_key)
                    return json.loads(cache_path.read_text())
        except Exception as exc:
            logger.debug("Failed to read earnings cache %s: %s", cache_path, _redact_secret_text(exc))

        try:
            payload = loader()
        except Exception as exc:
            logger.debug("Failed to refresh earnings cache %s: %s", cache_key, _redact_secret_text(exc))
            try:
                if cache_path.exists():
                    self._stale_cache_keys.add(cache_key)
                    provider_name, endpoint_type = _telemetry_from_cache_key(cache_key)
                    record_provider_telemetry(
                        provider_name=provider_name,
                        endpoint_type=endpoint_type,
                        success=True,
                        stale_used=True,
                        fallback_used=True,
                        response_quality_note="stale local cache used after provider refresh failure",
                    )
                    return json.loads(cache_path.read_text())
            except Exception:
                return []
            return []

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload))
            self._stale_cache_keys.discard(cache_key)
        except Exception as exc:
            logger.debug("Failed to persist earnings cache %s: %s", cache_path, _redact_secret_text(exc))
        return payload


def _telemetry_from_cache_key(cache_key: str) -> tuple[str, str]:
    if cache_key.startswith("alpha_vantage"):
        return "alpha_vantage", "earnings_calendar"
    if cache_key.startswith("fmp_confirmed"):
        return "fmp", "earnings_confirmed"
    if cache_key.startswith("fmp_calendar"):
        return "fmp", "earnings_calendar"
    if cache_key.startswith("sec_"):
        return "sec_edgar", "sec_cache"
    return "unknown", "earnings_cache"


def resolve_upcoming_earnings_event(
    symbol: str,
    as_of: date,
    cutoff: date,
    *,
    ticker: Optional[yf.Ticker] = None,
    mda_client: Optional[MarketDataClient] = None,
    cache_dir: Optional[Path] = None,
    alpha_vantage_api_key: Optional[str] = None,
    fmp_api_key: Optional[str] = None,
) -> ResolvedEarningsEvent:
    service = EarningsEventService(
        cache_dir=cache_dir,
        alpha_vantage_api_key=alpha_vantage_api_key,
        fmp_api_key=fmp_api_key,
    )
    return service.resolve_upcoming_event(
        symbol,
        as_of,
        cutoff,
        ticker=ticker,
        mda_client=mda_client,
    )


__all__ = [
    "EarningsEventCandidate",
    "EarningsEventService",
    "ResolvedEarningsEvent",
    "resolve_upcoming_earnings_event",
    "_timing_label_to_full",
]
