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
credits are always left for exits. A call is admitted by ``reserve``, which
checks the ceiling and holds the call's estimated cost in one write
transaction, so concurrent processes cannot both pass a check that only one
fits; the hold is settled at the actual cost (the provider's consumed-credits
header when present) or released if the call was not served. When the
provider reports its quota in response headers, on errors as well as
successes, the reported remaining credits win over the local estimate, until
the reset time it reported. A 429 that says the quota is used up is not
retried and marks the budget exhausted; concurrency throttling and plain
rate limits are retried with backoff.

The budget window starts at RESET_TIME_ENV (New York time; midnight unless
set). That default is not verified against the provider; set it to the
plan's documented reset.

A circuit breaker opens after consecutive transient failures (timeouts,
5xx, connection errors, rate limits after retries) and skips calls for a
cool-down instead of sending them. Half-open, it lets one probe through at
a time. It is per process: it protects a run, not every process on the
machine. A skipped call is not a provider request:
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
from datetime import datetime, timedelta, timezone
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
# When the provider's daily credits reset, as HH:MM New York time. The
# default (midnight) is not verified against MarketData.app; a reset time the
# provider reports in its X-Api-Ratelimit-Reset header always wins over it.
RESET_TIME_ENV = "OPTIONS_CALCULATOR_MARKETDATA_RESET_ET"
# A reservation whose request never settled (the process died) stops
# counting after this long.
RESERVATION_TTL_SECONDS = 600
UNMETERED = "unmetered"
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

    While open, ``allow()`` is False until the cool-down passes. Then exactly
    one call is let through as a probe (half-open); every other caller is
    still refused until the probe reports. Its success closes the breaker, its
    failure re-opens it for another cool-down. A probe that reports neither
    (e.g. a client error that says nothing about provider health) calls
    ``release_probe``; one that is lost entirely expires after a cool-down so
    the breaker can never wedge open.

    The breaker is per process: it protects one run from hammering a failing
    provider, not every process on the machine.
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
        self.probe_started_at: Optional[float] = None

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            now = self._clock()
            if now - self.opened_at < self.cooldown_seconds:
                return False
            if self.probe_started_at is not None and now - self.probe_started_at < self.cooldown_seconds:
                return False  # a probe is already out
            self.probe_started_at = now
            return True

    def release_probe(self) -> None:
        """The probe's outcome says nothing about provider health; let another probe go."""
        with self._lock:
            self.probe_started_at = None

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0
            self.opened_at = None
            self.probe_started_at = None

    def record_failure(self) -> None:
        with self._lock:
            self.probe_started_at = None
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
CREATE TABLE IF NOT EXISTS provider_credit_reservations (
    reservation_id  TEXT PRIMARY KEY,
    provider        TEXT NOT NULL,
    budget_day      TEXT NOT NULL,
    priority        TEXT NOT NULL,
    credits         REAL NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def _reset_offset() -> timedelta:
    raw = str(os.environ.get(RESET_TIME_ENV, "") or "").strip()
    try:
        hours, minutes = (int(part) for part in raw.split(":")) if raw else (0, 0)
        if 0 <= hours < 24 and 0 <= minutes < 60:
            return timedelta(hours=hours, minutes=minutes)
    except ValueError:
        pass
    logger.warning("ignoring %s=%r (expected HH:MM)", RESET_TIME_ENV, raw)
    return timedelta(0)


def budget_day(now: Optional[datetime] = None) -> str:
    """The budget window ``now`` falls in, named by the New York date it began."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (moment.astimezone(_BUDGET_TZ) - _reset_offset()).date().isoformat()


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
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(provider_reported_quota)")}
        if "reported_reset_at" not in columns:
            try:
                self._conn.execute("ALTER TABLE provider_reported_quota ADD COLUMN reported_reset_at TEXT")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
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

    def _reported(self, provider: str, day: str, now: datetime) -> Optional[sqlite3.Row]:
        row = self._conn.execute(
            "SELECT reported_limit, reported_remaining, reported_at, reported_reset_at FROM provider_reported_quota "
            "WHERE provider = ? AND budget_day = ?", (provider, day),
        ).fetchone()
        if row is not None and row["reported_reset_at"]:
            # The provider said when this quota resets; past that it is stale.
            if now >= datetime.fromisoformat(row["reported_reset_at"]):
                return None
        return row

    def _reserved(self, provider: str, day: str, now: datetime) -> float:
        cutoff = (now - timedelta(seconds=RESERVATION_TTL_SECONDS)).isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(credits), 0) FROM provider_credit_reservations"
            " WHERE provider = ? AND budget_day = ? AND created_at > ?", (provider, day, cutoff),
        ).fetchone()
        return float(row[0] or 0.0)

    def status(self, provider: str = MARKETDATA_PROVIDER) -> Dict[str, Any]:
        with self._lock:
            return self._status_locked(provider, self._now())

    def _status_locked(self, provider: str, now: datetime) -> Dict[str, Any]:
        day = budget_day(now)
        by_priority = self._spent(provider, day)
        reported = self._reported(provider, day, now)
        reserved = self._reserved(provider, day, now)
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
            # In flight: admitted but not yet settled against actual cost.
            "reserved": round(reserved, 3),
            "remaining": None if limit == float("inf") else round(max(0.0, limit - spent - reserved), 3),
            "utilization": round(utilization, 6),
            "alert_level": level,
            "by_priority": by_priority,
            "ceilings": dict(PRIORITY_CEILINGS),
        }

    @staticmethod
    def _fits(status: Dict[str, Any], priority: str, estimated_credits: float) -> bool:
        if status["daily_limit"] is None:
            return True
        ceiling = PRIORITY_CEILINGS.get(priority, PRIORITY_CEILINGS[DEFAULT_PRIORITY]) * status["daily_limit"]
        return status["spent"] + status["reserved"] + max(0.0, float(estimated_credits)) <= ceiling

    def allows(self, provider: str, priority: str, estimated_credits: float = 1.0) -> bool:
        """Would a call fit now? Advisory: only ``reserve`` admits atomically."""
        if provider != MARKETDATA_PROVIDER:
            return True
        return self._fits(self.status(provider), priority, estimated_credits)

    def reserve(self, provider: str, priority: str, estimated_credits: float) -> Optional[str]:
        """Admit a call and hold its estimated cost, atomically across processes.

        Returns a reservation id to ``settle`` (actual cost) or ``release``
        (not charged), or None when the call does not fit under the tier's
        ceiling. Check and hold happen under the database write lock, so two
        processes can no longer both pass a check the second would fail.
        """
        if provider != MARKETDATA_PROVIDER:
            return UNMETERED
        import uuid

        now = self._now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cutoff = (now - timedelta(seconds=RESERVATION_TTL_SECONDS)).isoformat()
                self._conn.execute("DELETE FROM provider_credit_reservations WHERE created_at <= ?", (cutoff,))
                if not self._fits(self._status_locked(provider, now), priority, estimated_credits):
                    self._conn.commit()
                    return None
                reservation_id = uuid.uuid4().hex
                self._conn.execute(
                    "INSERT INTO provider_credit_reservations"
                    " (reservation_id, provider, budget_day, priority, credits, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (reservation_id, provider, budget_day(now), priority, max(0.0, float(estimated_credits)),
                     now.isoformat()),
                )
                self._conn.commit()
                return reservation_id
            except BaseException:
                self._conn.rollback()
                raise

    def release(self, reservation_id: Optional[str]) -> None:
        """The call was not charged (refused, failed before the provider counted it)."""
        if not reservation_id or reservation_id == UNMETERED:
            return
        with self._lock:
            self._conn.execute("DELETE FROM provider_credit_reservations WHERE reservation_id = ?",
                               (reservation_id,))
            self._conn.commit()

    def settle(self, reservation_id: Optional[str], provider: str, priority: str, credits: float, *,
               reported_limit: Optional[float] = None, reported_remaining: Optional[float] = None,
               reported_reset_at: Optional[datetime] = None) -> None:
        """Replace a reservation with the call's actual cost (reconciliation)."""
        self.release(reservation_id)
        self.charge(provider, priority, credits, reported_limit=reported_limit,
                    reported_remaining=reported_remaining, reported_reset_at=reported_reset_at)

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
               reported_limit: Optional[float] = None, reported_remaining: Optional[float] = None,
               reported_reset_at: Optional[datetime] = None) -> None:
        self._bump(provider, priority, credits=max(0.0, float(credits)), requests=1)
        self.record_reported_quota(provider, reported_limit=reported_limit, reported_remaining=reported_remaining,
                                   reported_reset_at=reported_reset_at)

    def record_reported_quota(self, provider: str, *, reported_limit: Optional[float] = None,
                              reported_remaining: Optional[float] = None,
                              reported_reset_at: Optional[datetime] = None) -> None:
        """Keep what the provider says about the quota, from any response (errors included)."""
        if reported_limit is not None or reported_remaining is not None:
            day = budget_day(self._now())
            with self._lock:
                self._conn.execute(
                    """INSERT INTO provider_reported_quota
                           (provider, budget_day, reported_limit, reported_remaining, reported_at,
                            reported_reset_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(provider, budget_day) DO UPDATE SET
                           reported_limit = COALESCE(excluded.reported_limit, reported_limit),
                           reported_remaining = COALESCE(excluded.reported_remaining, reported_remaining),
                           reported_at = excluded.reported_at,
                           reported_reset_at = COALESCE(excluded.reported_reset_at, reported_reset_at)""",
                    (provider, day, reported_limit, reported_remaining, self._now().isoformat(),
                     reported_reset_at.isoformat() if reported_reset_at else None),
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
    """Why a call to ``provider`` at ``priority`` would be skipped now, or None.

    Advisory only (it takes a half-open probe when it allows one); a caller
    that will spend credits reserves them with ``CreditBudget.reserve``.
    """
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
