"""Provider reliability: credit budget, circuit breaker, retry backoff, disk cache.

The paid MarketData.app plan meters credits per day. Without a budget, a
discovery sweep could spend the day's credits before the exit pass re-priced
the open trades, and a provider outage was hammered by every later candidate.

Request priorities (highest first), bound by the caller with
``request_priority``:

- ``exit``: re-pricing open trades. May spend the whole budget.
- ``entry``: analysing and pricing candidates in the entry window.
  Unbound requests (the web API, scripts) default to this tier.
- ``discovery``: the screener sweep. Stops first.

Each tier may spend up to its ceiling share of the daily limit, so the last
credits are always left for exits. When the provider reports its own quota in
response headers, the reported remaining credits win over the local estimate.

A circuit breaker opens after consecutive transient failures (timeouts,
5xx, connection errors, rate limits after retries) and skips calls for a
cool-down instead of sending them. A skipped call is not a provider request:
it is recorded as a ``pipeline`` telemetry row and in the per-candidate skip
log the forward loop turns into an evidence-funnel disposition.

The disk cache persists responses for endpoints whose payload does not move
within its TTL (historical chains, earnings, expirations), so separate
processes in the same window share them. Live chains and quotes are never
persisted: a later process must not price the present with an older quote.
Cache keys are built from the URL and parameters only; credentials travel in
headers and are never stored.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PRIORITY_EXIT = "exit"
PRIORITY_ENTRY = "entry"
PRIORITY_DISCOVERY = "discovery"
PRIORITIES = (PRIORITY_EXIT, PRIORITY_ENTRY, PRIORITY_DISCOVERY)
DEFAULT_PRIORITY = PRIORITY_ENTRY

# Share of the daily limit each tier may spend (cumulative across tiers).
PRIORITY_CEILINGS = {PRIORITY_EXIT: 1.0, PRIORITY_ENTRY: 0.85, PRIORITY_DISCOVERY: 0.60}

SKIP_BUDGET = "provider_budget_exhausted"
SKIP_CIRCUIT_OPEN = "provider_circuit_open"

MARKETDATA_PROVIDER = "marketdata_app"
DAILY_CREDITS_ENV = "OPTIONS_CALCULATOR_MARKETDATA_DAILY_CREDITS"
DEFAULT_DAILY_CREDITS = 100_000
DISK_CACHE_ENV = "OPTIONS_CALCULATOR_PROVIDER_DISK_CACHE"

QUOTA_WARNING_UTILIZATION = 0.80
QUOTA_CRITICAL_UTILIZATION = 0.95

BREAKER_FAILURE_THRESHOLD = 5
BREAKER_COOLDOWN_SECONDS = 300.0

RETRY_BASE_SECONDS = 1.5
RETRY_CAP_SECONDS = 30.0

_BUDGET_TZ = ZoneInfo("America/New_York")
_ROOT = Path.home() / ".options_calculator_pro"
_DEFAULT_BUDGET_STORE = _ROOT / "telemetry" / "provider_budget.sqlite"
_DEFAULT_CACHE_STORE = _ROOT / "cache" / "provider_response_cache.sqlite"


# ── request priority and the per-candidate skip log ─────────────────────────

_priority: contextvars.ContextVar[str] = contextvars.ContextVar("provider_request_priority", default=DEFAULT_PRIORITY)
_skip_log: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "provider_skip_log", default=None
)


@contextmanager
def request_priority(priority: str) -> Iterator[None]:
    if priority not in PRIORITIES:
        raise ValueError(f"unknown request priority: {priority!r}")
    token = _priority.set(priority)
    try:
        yield
    finally:
        _priority.reset(token)


def current_priority() -> str:
    return _priority.get()


def start_skip_log() -> List[Dict[str, Any]]:
    """Start a fresh skip log for the current candidate and return it."""
    log: List[Dict[str, Any]] = []
    _skip_log.set(log)
    return log


@contextmanager
def skip_log_scope() -> Iterator[None]:
    """Bound the skip log to a run so it does not outlive it."""
    token = _skip_log.set(None)
    try:
        yield
    finally:
        _skip_log.reset(token)


def current_skips() -> List[Dict[str, Any]]:
    return list(_skip_log.get() or [])


def note_skip(*, provider: str, endpoint_type: str, reason: str, symbol: Optional[str] = None) -> None:
    log = _skip_log.get()
    if log is not None:
        log.append({"provider": provider, "endpoint_type": endpoint_type, "reason": reason,
                    "symbol": symbol, "priority": current_priority()})


# ── retry backoff ───────────────────────────────────────────────────────────


def retry_delay(
    attempt: int,
    *,
    retry_after: Any = None,
    base: float = RETRY_BASE_SECONDS,
    cap: float = RETRY_CAP_SECONDS,
    rng: Optional[random.Random] = None,
) -> float:
    """Seconds to wait before retry ``attempt`` (0-based).

    A server's Retry-After (seconds) is honoured up to the cap. Otherwise the
    wait is exponential with equal jitter: half fixed, half random, so
    concurrent clients spread out without retrying immediately.
    """
    try:
        hinted = float(retry_after) if retry_after not in (None, "") else None
    except (TypeError, ValueError):
        hinted = None
    if hinted is not None and hinted >= 0:
        return min(cap, hinted)
    ceiling = min(cap, base * (2 ** max(0, int(attempt))))
    return ceiling / 2.0 + (rng or random).uniform(0.0, ceiling / 2.0)


# ── circuit breaker ─────────────────────────────────────────────────────────


class CircuitBreaker:
    """Opens after ``threshold`` consecutive transient failures.

    While open, ``allow()`` is False until the cool-down passes; then calls
    are let through again (half-open). The next success closes the breaker;
    the next failure re-opens it for another cool-down.
    """

    def __init__(self, *, threshold: int = BREAKER_FAILURE_THRESHOLD,
                 cooldown_seconds: float = BREAKER_COOLDOWN_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold = max(1, int(threshold))
        self.cooldown_seconds = float(cooldown_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self.consecutive_failures = 0
        self.opened_at: Optional[float] = None
        self.open_count = 0

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            return self._clock() - self.opened_at >= self.cooldown_seconds

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0
            self.opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1
            # The count only resets on success, so a failed half-open trial
            # re-opens the breaker too.
            if self.consecutive_failures >= self.threshold:
                if self.opened_at is None:
                    self.open_count += 1
                    logger.warning("provider circuit opened after %d consecutive failures",
                                   self.consecutive_failures)
                self.opened_at = self._clock()

    def state(self) -> str:
        with self._lock:
            if self.opened_at is None:
                return "closed"
            return "half_open" if self._clock() - self.opened_at >= self.cooldown_seconds else "open"

    def snapshot(self) -> Dict[str, Any]:
        return {"state": self.state(), "consecutive_failures": self.consecutive_failures,
                "open_count": self.open_count}


_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(provider: str) -> CircuitBreaker:
    with _breakers_lock:
        if provider not in _breakers:
            _breakers[provider] = CircuitBreaker()
        return _breakers[provider]


def reset_breakers() -> None:
    with _breakers_lock:
        _breakers.clear()


# ── daily credit budget ─────────────────────────────────────────────────────

_BUDGET_DDL = """
CREATE TABLE IF NOT EXISTS provider_credit_spend (
    provider       TEXT NOT NULL,
    budget_day     TEXT NOT NULL,
    priority       TEXT NOT NULL,
    credits        REAL NOT NULL DEFAULT 0,
    request_count  INTEGER NOT NULL DEFAULT 0,
    skipped_count  INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (provider, budget_day, priority)
);
CREATE TABLE IF NOT EXISTS provider_reported_quota (
    provider            TEXT NOT NULL,
    budget_day          TEXT NOT NULL,
    reported_limit      REAL,
    reported_remaining  REAL,
    reported_at         TEXT NOT NULL,
    PRIMARY KEY (provider, budget_day)
);
"""


def budget_day(now: Optional[datetime] = None) -> str:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(_BUDGET_TZ).date().isoformat()


def configured_daily_limit(provider: str = MARKETDATA_PROVIDER) -> float:
    if provider != MARKETDATA_PROVIDER:
        return float("inf")
    raw = str(os.environ.get(DAILY_CREDITS_ENV, "") or "").strip()
    try:
        value = float(raw) if raw else float(DEFAULT_DAILY_CREDITS)
    except ValueError:
        value = float(DEFAULT_DAILY_CREDITS)
    return value if value > 0 else float(DEFAULT_DAILY_CREDITS)


class CreditBudget:
    """Per-provider daily credit ledger, shared by every process via SQLite."""

    def __init__(self, store_path: Path = _DEFAULT_BUDGET_STORE,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        from services.sqlite_helpers import open_db_conn

        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._lock = threading.Lock()
        self._conn = open_db_conn(self.store_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_BUDGET_DDL)
        self._conn.commit()
        self._alerted: Dict[str, str] = {}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # reads

    def _spent(self, provider: str, day: str) -> Dict[str, Dict[str, float]]:
        rows = self._conn.execute(
            "SELECT priority, credits, request_count, skipped_count FROM provider_credit_spend "
            "WHERE provider = ? AND budget_day = ?", (provider, day),
        ).fetchall()
        return {str(r["priority"]): {"credits": float(r["credits"]), "requests": int(r["request_count"]),
                                     "skipped": int(r["skipped_count"])} for r in rows}

    def _reported(self, provider: str, day: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT reported_limit, reported_remaining, reported_at FROM provider_reported_quota "
            "WHERE provider = ? AND budget_day = ?", (provider, day),
        ).fetchone()

    def status(self, provider: str = MARKETDATA_PROVIDER) -> Dict[str, Any]:
        day = budget_day(self._now())
        with self._lock:
            by_priority = self._spent(provider, day)
            reported = self._reported(provider, day)
        limit = configured_daily_limit(provider)
        limit_source = "env" if os.environ.get(DAILY_CREDITS_ENV) else "default"
        local_spent = sum(item["credits"] for item in by_priority.values())
        spent = local_spent
        if reported is not None and reported["reported_limit"] is not None:
            limit = float(reported["reported_limit"])
            limit_source = "provider_reported"
        if reported is not None and reported["reported_remaining"] is not None:
            # The provider's count also includes spend from other clients.
            spent = max(local_spent, limit - float(reported["reported_remaining"]))
        utilization = spent / limit if limit and limit != float("inf") else 0.0
        if utilization >= 1.0:
            level = "exhausted"
        elif utilization >= QUOTA_CRITICAL_UTILIZATION:
            level = "critical"
        elif utilization >= QUOTA_WARNING_UTILIZATION:
            level = "warning"
        else:
            level = "ok"
        return {
            "provider": provider,
            "budget_day": day,
            "daily_limit": None if limit == float("inf") else limit,
            "limit_source": limit_source,
            "spent": round(spent, 3),
            "remaining": None if limit == float("inf") else round(max(0.0, limit - spent), 3),
            "utilization": round(utilization, 6),
            "alert_level": level,
            "by_priority": by_priority,
            "ceilings": dict(PRIORITY_CEILINGS),
        }

    def allows(self, provider: str, priority: str, estimated_credits: float = 1.0) -> bool:
        if provider != MARKETDATA_PROVIDER:
            return True
        status = self.status(provider)
        if status["daily_limit"] is None:
            return True
        ceiling = PRIORITY_CEILINGS.get(priority, PRIORITY_CEILINGS[DEFAULT_PRIORITY]) * status["daily_limit"]
        return status["spent"] + max(0.0, float(estimated_credits)) <= ceiling

    # writes

    def _bump(self, provider: str, priority: str, *, credits: float = 0.0, requests: int = 0,
              skipped: int = 0) -> None:
        day = budget_day(self._now())
        stamp = self._now().isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO provider_credit_spend
                       (provider, budget_day, priority, credits, request_count, skipped_count, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(provider, budget_day, priority) DO UPDATE SET
                       credits = credits + excluded.credits,
                       request_count = request_count + excluded.request_count,
                       skipped_count = skipped_count + excluded.skipped_count,
                       updated_at = excluded.updated_at""",
                (provider, day, priority, float(credits), int(requests), int(skipped), stamp),
            )
            self._conn.commit()

    def charge(self, provider: str, priority: str, credits: float, *,
               reported_limit: Optional[float] = None, reported_remaining: Optional[float] = None) -> None:
        self._bump(provider, priority, credits=max(0.0, float(credits)), requests=1)
        if reported_limit is not None or reported_remaining is not None:
            day = budget_day(self._now())
            with self._lock:
                self._conn.execute(
                    """INSERT INTO provider_reported_quota
                           (provider, budget_day, reported_limit, reported_remaining, reported_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(provider, budget_day) DO UPDATE SET
                           reported_limit = COALESCE(excluded.reported_limit, reported_limit),
                           reported_remaining = excluded.reported_remaining,
                           reported_at = excluded.reported_at""",
                    (provider, day, reported_limit, reported_remaining, self._now().isoformat()),
                )
                self._conn.commit()
        self._maybe_alert(provider)

    def record_skip(self, provider: str, priority: str) -> None:
        self._bump(provider, priority, skipped=1)

    def _maybe_alert(self, provider: str) -> None:
        status = self.status(provider)
        level = status["alert_level"]
        key = f"{provider}|{status['budget_day']}"
        if level != "ok" and self._alerted.get(key) != level:
            self._alerted[key] = level
            logger.warning("%s credit budget %s: %.0f of %.0f credits used (%.0f%%)", provider, level,
                           status["spent"], status["daily_limit"] or 0, status["utilization"] * 100)


_budget: Optional[CreditBudget] = None
_budget_lock = threading.Lock()


def get_credit_budget(store_path: Optional[Path] = None) -> CreditBudget:
    global _budget
    if _budget is None or store_path is not None:
        with _budget_lock:
            if _budget is None or store_path is not None:
                _budget = CreditBudget(store_path or _DEFAULT_BUDGET_STORE)
    return _budget


def blocked_reason(provider: str, priority: Optional[str] = None, estimated_credits: float = 1.0) -> Optional[str]:
    """Why a call to ``provider`` at ``priority`` would be skipped now, or None."""
    if not get_breaker(provider).allow():
        return SKIP_CIRCUIT_OPEN
    try:
        allowed = get_credit_budget().allows(provider, priority or current_priority(), estimated_credits)
    except Exception:  # noqa: BLE001 - an unreadable ledger must not stop data collection
        logger.debug("provider budget unavailable", exc_info=True)
        return None
    return None if allowed else SKIP_BUDGET


def reliability_status() -> Dict[str, Any]:
    try:
        budget = get_credit_budget().status(MARKETDATA_PROVIDER)
    except Exception as exc:  # noqa: BLE001
        budget = {"error": str(exc)[:200]}
    with _breakers_lock:
        breakers = {name: breaker.snapshot() for name, breaker in _breakers.items()}
    return {"marketdata_budget": budget, "circuit_breakers": breakers}


def build_provider_budget_health(*, budget: Optional[CreditBudget] = None) -> Dict[str, Any]:
    """Watchdog issues for the MarketData credit budget, in the evidence-health issue shape."""
    try:
        status = (budget or get_credit_budget()).status(MARKETDATA_PROVIDER)
    except Exception as exc:  # noqa: BLE001
        return {
            "summary": {"error": f"{type(exc).__name__}: {exc}"},
            "issues": [{"severity": "WARN", "alertable": False,
                        "message": f"Provider credit budget unreadable: {type(exc).__name__}"}],
        }
    level = status["alert_level"]
    issues = []
    if level != "ok":
        issues.append({
            "severity": "WARN",
            # Exhausted or nearly so means exits may go unpriced today; 80% is
            # a heads-up only.
            "alertable": level in {"critical", "exhausted"},
            "message": (
                f"MarketData credit budget {level}: {status['spent']:.0f} of "
                f"{status['daily_limit']:.0f} credits used on {status['budget_day']}."
            ),
        })
    return {"summary": {key: status[key] for key in ("budget_day", "daily_limit", "limit_source", "spent",
                                                     "utilization", "alert_level")},
            "issues": issues}


# ── persistent response cache ───────────────────────────────────────────────

_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS provider_response_cache (
    cache_key   TEXT PRIMARY KEY,
    stored_at   REAL NOT NULL,
    expires_at  REAL NOT NULL,
    payload     TEXT NOT NULL
);
"""


class ResponseCache:
    """Wall-clock TTL cache of JSON responses, shared across processes."""

    def __init__(self, store_path: Path = _DEFAULT_CACHE_STORE,
                 clock: Callable[[], float] = time.time) -> None:
        from services.sqlite_helpers import open_db_conn

        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = open_db_conn(self.store_path)
        self._conn.executescript(_CACHE_DDL)
        self._conn.execute("DELETE FROM provider_response_cache WHERE expires_at <= ?", (self._clock(),))
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, expires_at FROM provider_response_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None or float(row[1]) <= self._clock():
            return None
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            return None

    def put(self, key: str, payload: Any, ttl_seconds: float) -> None:
        try:
            text = json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError):
            return
        now = self._clock()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO provider_response_cache (cache_key, stored_at, expires_at, payload) "
                "VALUES (?, ?, ?, ?)", (key, now, now + float(ttl_seconds), text),
            )
            self._conn.commit()


_cache: Optional[ResponseCache] = None
_cache_lock = threading.Lock()


def disk_cache_enabled() -> bool:
    return str(os.environ.get(DISK_CACHE_ENV, "1")).strip().lower() not in {"0", "false", "no", "off"}


def get_response_cache(store_path: Optional[Path] = None) -> Optional[ResponseCache]:
    global _cache
    if store_path is None and not disk_cache_enabled():
        return None
    if _cache is None or store_path is not None:
        with _cache_lock:
            if _cache is None or store_path is not None:
                try:
                    _cache = ResponseCache(store_path or _DEFAULT_CACHE_STORE)
                except Exception:  # noqa: BLE001 - the cache is an optimisation
                    logger.debug("provider response cache unavailable", exc_info=True)
                    return None
    return _cache


__all__ = [
    "PRIORITIES", "PRIORITY_CEILINGS", "PRIORITY_DISCOVERY", "PRIORITY_ENTRY", "PRIORITY_EXIT",
    "SKIP_BUDGET", "SKIP_CIRCUIT_OPEN", "CircuitBreaker", "CreditBudget", "ResponseCache",
    "blocked_reason", "build_provider_budget_health", "current_priority", "current_skips", "get_breaker", "get_credit_budget",
    "get_response_cache", "note_skip", "reliability_status", "request_priority", "reset_breakers",
    "retry_delay", "skip_log_scope", "start_skip_log",
]
