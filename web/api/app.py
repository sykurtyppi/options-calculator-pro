from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import hmac
import io
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple
import sys
import sqlite3

_OOS_VALIDATION_TIMEOUT_SECONDS: float = 300.0  # 5 min hard wall-clock limit per call
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env before importing any service that reads env vars
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

# ── Access-control config (loaded from .env) ─────────────────────────────────
_SHARE_AUTH_ENABLED = os.getenv("ENABLE_SHARE_AUTH", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SHARE_PASSWORD = os.getenv("SHARE_PASSWORD", "")
_SESSION_SECRET = os.getenv("SESSION_SECRET", "change-me-in-env")
_SESSION_COOKIE = "ops_session"
_SESSION_MAX_AGE = 7 * 24 * 3600  # 1 week
_MAX_SCREENER_SYMBOLS = max(1, int(os.getenv("OPTIONS_CALCULATOR_MAX_SCREENER_SYMBOLS", "100")))
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,15}$")
_HOSTED_MODE = os.getenv("OPTIONS_CALCULATOR_HOSTED_MODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_PROTECT_API_DOCS = (
    os.getenv("OPTIONS_CALCULATOR_PROTECT_API_DOCS", "").strip().lower() in {"1", "true", "yes", "on"}
    or _HOSTED_MODE
)
_SECURE_SESSION_COOKIE = (
    os.getenv("OPTIONS_CALCULATOR_SECURE_COOKIES", "").strip().lower() in {"1", "true", "yes", "on"}
    or _HOSTED_MODE
)
# Web-audit P1-3: explicit escape hatch for the plaintext-HTTP local dev case.
# When share-auth is on, _validate_auth_config requires either Secure cookies
# (recommended) or this opt-in flag — acknowledging the cookie-disclosure risk
# on any HTTP sibling page.
_ALLOW_INSECURE_SESSION_COOKIE = (
    os.getenv("OPTIONS_CALCULATOR_ALLOW_INSECURE_SESSION_COOKIE", "")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
# Web-audit P1-1: opt-in flag for honouring X-Forwarded-For in _client_ip().
# OFF by default (so untrusted direct requests can't spoof their IP); ON when
# behind a trusted proxy/tunnel (Cloudflare, Tailscale Funnel, nginx) — without
# this, /login rate-limiting collapses to a single global bucket because every
# request comes from the proxy's loopback IP.
_TRUST_PROXY_HEADERS = (
    os.getenv("OPTIONS_CALCULATOR_TRUST_PROXY_HEADERS", "").strip().lower()
    in {"1", "true", "yes", "on"}
    or _HOSTED_MODE  # hosted mode implies a trusted tunnel/proxy is in front
)
_MAX_OOS_RUNNING_JOBS = max(1, int(os.getenv("OPTIONS_CALCULATOR_MAX_OOS_RUNNING_JOBS", "1")))
_MAX_ML_RUNNING_JOBS = max(1, int(os.getenv("OPTIONS_CALCULATOR_MAX_ML_RUNNING_JOBS", "1")))
_MAX_RETAINED_JOBS = max(10, int(os.getenv("OPTIONS_CALCULATOR_MAX_RETAINED_JOBS", "100")))

# ── Per-IP rate limits ────────────────────────────────────────────────────────
# Both windows on every limited endpoint are checked; either trips a 429.
# Defaults are picked to bound brute-force / DoS-by-loop throughput while
# leaving normal use unaffected.
_LOGIN_RATE_LIMIT_PER_MIN = max(1, int(os.getenv("OPTIONS_LOGIN_RATE_LIMIT_PER_MIN", "5")))
_LOGIN_RATE_LIMIT_PER_HOUR = max(1, int(os.getenv("OPTIONS_LOGIN_RATE_LIMIT_PER_HOUR", "30")))
# Web-audit P2-3: /api/ml/train and /api/oos/submit had a 1-job concurrency
# lock but no rate limit — an auth'd user (or stolen cookie) could loop POSTs
# to drive sustained CPU + model-file churn.
_ML_TRAIN_RATE_LIMIT_PER_MIN = max(1, int(os.getenv("OPTIONS_ML_TRAIN_RATE_LIMIT_PER_MIN", "2")))
_ML_TRAIN_RATE_LIMIT_PER_HOUR = max(1, int(os.getenv("OPTIONS_ML_TRAIN_RATE_LIMIT_PER_HOUR", "10")))
_OOS_SUBMIT_RATE_LIMIT_PER_MIN = max(1, int(os.getenv("OPTIONS_OOS_SUBMIT_RATE_LIMIT_PER_MIN", "3")))
_OOS_SUBMIT_RATE_LIMIT_PER_HOUR = max(1, int(os.getenv("OPTIONS_OOS_SUBMIT_RATE_LIMIT_PER_HOUR", "20")))
# F2: screener endpoints fan out to the external MDA provider (one API call per
# symbol). A tight loop by an auth'd client burns provider quota fast.
# Defaults: 6/min, 40/hour per IP. Both screener variants share one bucket.
_SCREENER_RATE_LIMIT_PER_MIN = max(1, int(os.getenv("OPTIONS_SCREENER_RATE_LIMIT_PER_MIN", "6")))
_SCREENER_RATE_LIMIT_PER_HOUR = max(1, int(os.getenv("OPTIONS_SCREENER_RATE_LIMIT_PER_HOUR", "40")))

# (bucket_name, ip) -> list of recent attempt timestamps. Same shape used for
# all rate-limited endpoints; keyed by bucket so /login attempts can't be
# counted against /api/ml/train, etc.
_rate_limit_buckets: Dict[Tuple[str, str], List[float]] = {}
_rate_limit_lock = threading.Lock()


# Web-audit P1-2: session-nonce revocation set. POST /logout records the
# nonce here so _valid_session rejects the cookie even before its 7-day TTL
# expires. In-memory only — server restart "forgets" revocations, but new
# logins get fresh nonces anyway, so all that means is that a cookie which
# was previously logged-out may become valid again until either (a) the
# original 7-day expiry fires or (b) the user logs out again on the new
# instance. The bigger threat — undetected cookie theft — is what this
# defends against.
_revoked_nonces: Dict[str, float] = {}  # nonce -> original cookie's expiry timestamp
_revoked_nonces_lock = threading.Lock()

# Paths that don't require authentication. /logout is here so users with an
# expired or already-revoked cookie can still hit it (the handler is a no-op
# on missing/invalid cookies; it still clears whatever's there).
_PUBLIC_PATHS = {"/login", "/logout", "/favicon.ico", "/api/health"}

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Options Calculator Pro</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117;
      color: #c9d1d9;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }
    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 40px;
      width: 100%;
      max-width: 380px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    h1 { font-size: 1.25rem; font-weight: 600; margin-bottom: 6px; color: #e6edf3; }
    p.sub { font-size: 0.82rem; color: #8b949e; margin-bottom: 28px; }
    label { display: block; font-size: 0.8rem; color: #8b949e; margin-bottom: 6px; letter-spacing: 0.02em; }
    input[type="password"] {
      width: 100%;
      padding: 10px 14px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #e6edf3;
      font-size: 1.05rem;
      outline: none;
      transition: border-color 0.2s;
      letter-spacing: 0.18em;
    }
    input[type="password"]:focus { border-color: #388bfd; }
    .error { color: #f85149; font-size: 0.82rem; margin-top: 8px; min-height: 1.2em; }
    button {
      margin-top: 20px;
      width: 100%;
      padding: 10px;
      background: #1f6feb;
      border: none;
      border-radius: 6px;
      color: #fff;
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover { background: #388bfd; }
    .footer { text-align: center; margin-top: 28px; font-size: 0.72rem; color: #484f58; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Options Calculator Pro</h1>
    <p class="sub">Enter your access code to continue.</p>
    <form method="POST" action="/login">
      <label for="pw">Access Code</label>
      <input type="password" id="pw" name="password" autofocus autocomplete="current-password" />
      <div class="error">{{ERROR}}</div>
      <button type="submit">Continue &rarr;</button>
    </form>
    <div class="footer">Created by Tristan Alejandro &middot; Not financial advice.</div>
  </div>
</body>
</html>"""


def _is_strong_session_secret(secret: str) -> bool:
    stripped = str(secret or "").strip()
    return bool(stripped and stripped != "change-me-in-env" and len(stripped) >= 24)


def _validate_auth_config() -> None:
    if _HOSTED_MODE and not _SHARE_AUTH_ENABLED:
        raise RuntimeError("OPTIONS_CALCULATOR_HOSTED_MODE=true requires ENABLE_SHARE_AUTH=true.")
    if not _SHARE_AUTH_ENABLED:
        return
    if not _SHARE_PASSWORD:
        raise RuntimeError("ENABLE_SHARE_AUTH=true requires SHARE_PASSWORD.")
    if not _is_strong_session_secret(_SESSION_SECRET):
        raise RuntimeError(
            "ENABLE_SHARE_AUTH=true requires SESSION_SECRET to be non-default and at least 24 characters."
        )
    # Web-audit P1-3: a session cookie without the Secure flag can be read by
    # any plaintext-HTTP page on the same eTLD+1 (sibling subdomain misconfig,
    # MITM on an HTTP fetch the browser issues to the same host). Force the
    # operator to either deploy with TLS (recommended — OPTIONS_CALCULATOR_
    # SECURE_COOKIES=true, or HOSTED_MODE=true which implies it) or opt in
    # explicitly to the no-TLS local-dev case via OPTIONS_CALCULATOR_ALLOW_
    # INSECURE_SESSION_COOKIE=true.
    if not _SECURE_SESSION_COOKIE and not _ALLOW_INSECURE_SESSION_COOKIE:
        raise RuntimeError(
            "ENABLE_SHARE_AUTH=true requires either OPTIONS_CALCULATOR_SECURE_COOKIES=true "
            "(or OPTIONS_CALCULATOR_HOSTED_MODE=true, which implies it) — recommended for any "
            "TLS deployment — OR an explicit OPTIONS_CALCULATOR_ALLOW_INSECURE_SESSION_COOKIE="
            "true escape hatch for plaintext-HTTP local dev. Without one of those, the session "
            "cookie can be disclosed by any HTTP page on the same eTLD+1."
        )


def _docs_path(path: str) -> bool:
    return path.startswith(("/docs", "/openapi", "/redoc"))


def _sanitize_exception(exc: Exception, public_message: str) -> str:
    logger.exception("%s: %s", public_message, exc)
    return public_message


def _raise_public_error(status_code: int, public_message: str, exc: Exception) -> None:
    raise HTTPException(status_code=status_code, detail=_sanitize_exception(exc, public_message))


_validate_auth_config()


def _session_signature(payload: str) -> str:
    return hmac.new(_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _session_token(now: Optional[datetime] = None) -> str:
    """Issue a signed, expiring session token with a per-login nonce."""
    issued_at = int((now or datetime.now(timezone.utc)).timestamp())
    payload = f"v1.{issued_at}.{uuid.uuid4().hex}"
    return f"{payload}.{_session_signature(payload)}"


def _extract_session_token(cookie_val: str) -> Optional[Tuple[str, int]]:
    """Parse and HMAC-verify *cookie_val*.

    Returns ``(nonce, issued_at)`` if the cookie's signature is valid
    AND it's within the normal session age window; otherwise ``None``.
    Does NOT consult the revocation set — that's the caller's job in
    :func:`_valid_session`. By keeping the verification gate separate
    from the revocation check, ``POST /logout`` can use this helper
    to decide whether a cookie is worth revoking *at all* (Codex
    web-audit follow-up P1).

    Without this gate, ``/logout`` accepts any syntactically shaped
    cookie and inserts its nonce into ``_revoked_nonces`` — turning a
    public endpoint into a memory-pressure surface. Forged cookies
    with absurd timestamps could also blow up ``int(issued_at_raw)``
    with ``OverflowError``; that's caught here too.
    """
    if not _SHARE_AUTH_ENABLED or not cookie_val or not _SHARE_PASSWORD:
        return None
    try:
        version, issued_at_raw, nonce, signature = cookie_val.split(".", 3)
        if version != "v1" or not nonce:
            return None
        issued_at = int(issued_at_raw)
        age = int(datetime.now(timezone.utc).timestamp()) - issued_at
        if age < 0 or age > _SESSION_MAX_AGE:
            return None
        payload = f"{version}.{issued_at_raw}.{nonce}"
        if not hmac.compare_digest(signature, _session_signature(payload)):
            return None
        return (nonce, issued_at)
    except (ValueError, TypeError, OverflowError):
        return None


def _valid_session(cookie_val: str) -> bool:
    if not _SHARE_AUTH_ENABLED:
        return True
    parsed = _extract_session_token(cookie_val)
    if parsed is None:
        return False
    nonce, _issued_at = parsed
    # Web-audit P1-2: even a signature-valid cookie is rejected if the
    # nonce has been revoked via POST /logout. Without this check, the
    # only way to invalidate a leaked cookie before its 7-day TTL was
    # to rotate SESSION_SECRET — which kicks every other session.
    return not _is_nonce_revoked(nonce)


def _revoke_nonce(nonce: str, expiry_ts: float) -> None:
    """Add *nonce* to the in-memory revocation set with its original cookie
    expiry. Subsequent calls to :func:`_valid_session` reject the cookie even
    though the HMAC still verifies.

    Opportunistically GCs any revoked nonces whose original cookies have
    already expired — the auth path would reject them on age anyway, so
    there's no point keeping them in memory.
    """
    if not nonce:
        return
    now = time.time()
    with _revoked_nonces_lock:
        # GC first so the dict can't grow unbounded under a /logout flood.
        for n, exp in list(_revoked_nonces.items()):
            if exp < now:
                _revoked_nonces.pop(n, None)
        _revoked_nonces[nonce] = expiry_ts


def _is_nonce_revoked(nonce: str) -> bool:
    if not nonce:
        return False
    with _revoked_nonces_lock:
        return nonce in _revoked_nonces


def _reset_revoked_nonces() -> None:
    """Drop the revoked-nonce set. Primarily for tests."""
    with _revoked_nonces_lock:
        _revoked_nonces.clear()


def _client_ip(request: Request) -> str:
    """Return the client IP for rate-limiting purposes.

    Web-audit P1-1: behind a trusted reverse proxy (Cloudflare, Tailscale
    Funnel, nginx), ``request.client.host`` collapses to the proxy's
    loopback IP — so every login attempt counts against the same bucket,
    making `/login` rate-limiting effectively global. When
    ``OPTIONS_CALCULATOR_TRUST_PROXY_HEADERS=true`` (or HOSTED_MODE=true,
    which implies it), honour the leftmost ``X-Forwarded-For`` instead.

    Trust is opt-in because an untrusted direct caller can set
    ``X-Forwarded-For`` to whatever they want and spoof their IP — only
    safe when a real proxy is enforced in front of this process.
    """
    if _TRUST_PROXY_HEADERS:
        xff = request.headers.get("x-forwarded-for", "").strip()
        if xff:
            # XFF format is "client, proxy1, proxy2"; leftmost is the
            # original client. Strip whitespace and reject empty entries
            # (some buggy proxies emit ",realip").
            first = xff.split(",", 1)[0].strip()
            if first:
                return first
    return request.client.host if request.client else "unknown"


def _check_rate_limit(bucket: str, ip: str, limit_per_min: int, limit_per_hour: int) -> bool:
    """Generic per-IP rate limiter. Used by /login, /api/ml/train, and
    /api/oos/submit. Both windows trip independently; either limit blocks
    further attempts until the window slides.

    Counts every attempt regardless of success — a successful request
    doesn't reset the counter, so brute-force / loop-DoS can't hide behind
    an eventual win. Buckets are keyed by ``(bucket_name, ip)`` so
    endpoints can't poison each other's quotas.

    Memory bound: O(unique IPs × buckets); each IP enumeration costs the
    attacker at least one TCP connection. Acceptable for the solo-deploy
    threat model.
    """
    now = time.time()
    cutoff_min = now - 60.0
    cutoff_hour = now - 3600.0
    key = (bucket, ip)
    with _rate_limit_lock:
        attempts = _rate_limit_buckets.setdefault(key, [])
        attempts[:] = [t for t in attempts if t >= cutoff_hour]
        recent_min = sum(1 for t in attempts if t >= cutoff_min)
        if recent_min >= limit_per_min:
            return False
        if len(attempts) >= limit_per_hour:
            return False
        attempts.append(now)
        return True


def _check_login_rate_limit(ip: str) -> bool:
    """Thin wrapper retained for clarity at the call site. Delegates to
    the generic limiter using the ``"login"`` bucket and the configured
    /login per-minute/per-hour caps."""
    return _check_rate_limit(
        "login", ip, _LOGIN_RATE_LIMIT_PER_MIN, _LOGIN_RATE_LIMIT_PER_HOUR
    )


def _reset_login_rate_limit() -> None:
    """Drop ALL rate-limit history (every bucket, every IP). Primarily for
    tests. Despite the name, this clears every bucket — the original
    helper predates the multi-endpoint refactor and is kept under the
    same name so existing tests don't break. Use ``_reset_rate_limits()``
    in new code.
    """
    _reset_rate_limits()


def _reset_rate_limits() -> None:
    """Drop ALL rate-limit history. Primarily for tests."""
    with _rate_limit_lock:
        _rate_limit_buckets.clear()


def _parse_symbol_universe(symbols: Optional[str], default_universe: List[str]) -> List[str]:
    universe = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else list(default_universe)
    )
    if len(universe) > _MAX_SCREENER_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol universe is limited to {_MAX_SCREENER_SYMBOLS} symbols per request.",
        )
    invalid = [s for s in universe if not _SYMBOL_RE.fullmatch(s)]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid symbol(s): {', '.join(invalid[:5])}",
        )
    return universe


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Let CORS preflight requests through — they carry no session cookie
        if request.method == "OPTIONS":
            return await call_next(request)
        # Let public paths through unauthenticated; API docs are public only in local mode.
        path = request.url.path
        if path in _PUBLIC_PATHS or (_docs_path(path) and not _PROTECT_API_DOCS):
            return await call_next(request)

        session = request.cookies.get(_SESSION_COOKIE, "")
        if not _valid_session(session):
            # API requests get a 401 (so the frontend can handle it gracefully)
            if path.startswith("/api/") or _docs_path(path):
                return JSONResponse(
                    content={"detail": "Unauthorized — please log in at /login"},
                    status_code=401,
                )
            # Everything else → redirect to login
            return RedirectResponse(url="/login", status_code=302)

        return await call_next(request)


from scripts.institutional_backfill import InstitutionalDataCollector
from services import external_io_gate
from services.market_data_client import MarketDataClient
from services.options_feature_store import OptionsFeatureStore, OptionsFeatureStoreError
from web.api.edge_engine import analyze_single_ticker
from web.api.screener_engine import build_edge_screener
from web.api.schemas import (
    CalibrationBucket,
    CalibrationCurveResponse,
    EdgeScreenerResponse,
    EdgeAnalyzeRequest,
    EdgeAnalyzeResponse,
    HistoricalOptionsCoverageResponse,
    HistoricalOptionsResponse,
    OOSReportRequest,
    OOSReportResponse,
    RankedScreenerResponse,
    RankedSetupRow,
    LearningDiagnosticsResponse,
)

# Module-level MarketDataClient singleton — shared across requests so the
# in-process cache persists between calls (saves credits on repeated symbols).
_mda_client: Optional[MarketDataClient] = None
_feature_store: Optional[OptionsFeatureStore] = None


def _get_mda_client() -> MarketDataClient:
    global _mda_client
    if _mda_client is None:
        external_io_gate.assert_allowed(external_io_gate.Category.MARKETDATA)
        _mda_client = MarketDataClient()
    return _mda_client


def _get_feature_store() -> OptionsFeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = OptionsFeatureStore()
    return _feature_store


app = FastAPI(
    title="Options Calculator Pro API",
    version="0.1.0",
    description="Event-volatility structure selection, evidence diagnostics, and OOS robustness reports.",
)

_LOCALHOST_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]


def _compute_allowed_origins(share_auth_enabled: bool, origins_env: str) -> List[str]:
    """Resolve the CORS ``allow_origins`` list with the safety rails from
    Web-audit P2-2 plus the Codex audit follow-up.

    The localhost-dev defaults grant credentialed CORS access to any
    cross-port local web service. That's fine when share-auth is off
    (the auth middleware lets everything through anyway, so there's
    nothing privileged to exfiltrate). But when share-auth is on, any
    malicious localhost service the operator runs in the same browser
    can ``fetch(credentials: "include")`` against this app and read
    session-cookie-protected data. Force operators to set
    ``OPTIONS_CALCULATOR_ALLOWED_ORIGINS`` explicitly in that case AND
    refuse the two values that would undo the entire gate:

      * ``"*"`` — combined with ``allow_credentials=True`` this is the
        confused-deputy footgun the gate was meant to close. Starlette
        does fail-closed at the middleware layer when both are passed,
        but failing here (with a clear error) catches the misconfig at
        startup instead of producing silent 4xx CORS errors later.
      * ``"null"`` — the literal origin string emitted by sandboxed
        iframes, ``file://`` documents, and some redirect chains.
        Allowing it grants implicit trust to anything that opens this
        app inside a sandbox.

    Extracted into a function so tests can exercise the contract without
    importing the FastAPI app at startup (which would lock in whatever
    env vars happen to be set at import time).
    """
    if origins_env.strip():
        origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
        if share_auth_enabled:
            forbidden = {"*", "null"}
            bad = sorted({o for o in origins if o in forbidden})
            if bad:
                raise RuntimeError(
                    "OPTIONS_CALCULATOR_ALLOWED_ORIGINS cannot include "
                    f"{bad!r} when ENABLE_SHARE_AUTH=true: allow_credentials=True "
                    "is on and these values would undo the credentialed-CORS gate. "
                    "Use an explicit list of origins (e.g., 'https://app.example.com')."
                )
        return origins
    if share_auth_enabled:
        raise RuntimeError(
            "ENABLE_SHARE_AUTH=true requires OPTIONS_CALCULATOR_ALLOWED_ORIGINS to be "
            "set explicitly (comma-separated list of origins). The localhost-dev "
            "defaults would grant credentialed CORS access to any cross-port local "
            "web service — a real risk when running multiple local services."
        )
    return list(_LOCALHOST_DEV_ORIGINS)


origins_env = os.getenv("OPTIONS_CALCULATOR_ALLOWED_ORIGINS", "")
allowed_origins = _compute_allowed_origins(_SHARE_AUTH_ENABLED, origins_env)

# Starlette applies middleware in LIFO order: the last add_middleware() call
# becomes the outermost layer (first to see requests, last to see responses).
# _AuthMiddleware must be INNER so its 401 responses flow back through
# CORSMiddleware before reaching the client. If auth were outer, 401s would
# bypass CORS entirely and the browser would surface a CORS error instead of
# the 401, breaking the frontend's /login redirect.
app.add_middleware(_AuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Login routes ──────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    return HTMLResponse(_LOGIN_HTML.replace("{{ERROR}}", ""))


@app.post("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_submit(request: Request):
    ip = _client_ip(request)
    if not _check_login_rate_limit(ip):
        # Don't disclose the limit values in the response — a tighter
        # message gives an attacker less feedback.
        return HTMLResponse(
            _LOGIN_HTML.replace(
                "{{ERROR}}",
                "Too many login attempts. Wait a minute and try again.",
            ),
            status_code=429,
        )

    form = await request.form()
    password = form.get("password", "")
    if _SHARE_AUTH_ENABLED and _SHARE_PASSWORD and hmac.compare_digest(str(password), _SHARE_PASSWORD):
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            _SESSION_COOKIE,
            _session_token(),
            httponly=True,
            samesite="lax",
            secure=_SECURE_SESSION_COOKIE,
            max_age=_SESSION_MAX_AGE,
        )
        return response
    # Wrong password — re-render with error
    return HTMLResponse(
        _LOGIN_HTML.replace("{{ERROR}}", "Incorrect access code."),
        status_code=401,
    )


@app.post("/logout", include_in_schema=False)
async def logout(request: Request):
    """Clear the session cookie and add its nonce to the in-memory
    revocation set so the (still-signature-valid) cookie can't be replayed
    even if it was captured before expiry.

    Web-audit P1-2. POST-only on purpose — a GET would let an attacker
    log the operator out via an ``<img src="…/logout">`` tag from any
    site. Annoyance, not security breach, but easy to close.

    Codex web-audit follow-up P1: only revoke a nonce after the cookie's
    HMAC + age have been verified by :func:`_extract_session_token`.
    Otherwise an unauthenticated attacker could POST forged cookies with
    arbitrary nonces and timestamps to grow ``_revoked_nonces``
    unboundedly. The ``delete_cookie`` below still runs on every call,
    so users with malformed or expired cookies are still cleanly
    logged out — they just don't get anything written to the revocation
    set (there's nothing valid to revoke).
    """
    session = request.cookies.get(_SESSION_COOKIE, "")
    if session:
        parsed = _extract_session_token(session)
        if parsed is not None:
            nonce, issued_at = parsed
            _revoke_nonce(nonce, float(issued_at + _SESSION_MAX_AGE))
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(
        _SESSION_COOKIE,
        path="/",
        secure=_SECURE_SESSION_COOKIE,
        httponly=True,
        samesite="lax",
    )
    return response


_OOS_STABILITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "evidence_balanced": {
        "execution_profiles": ["institutional"],
        "hold_days_grid": [1, 3],
        "trades_per_day_grid": [2, 3],
        "entry_days_grid": [3, 5, 7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.48,
            "min_crush_confidence": 0.28,
            "min_crush_magnitude": 0.06,
            "min_crush_edge": 0.025,
            "target_entry_dte": 6,
            "entry_dte_band": 5,
            "min_daily_share_volume": 1_500_000,
            "max_abs_momentum_5d": 0.11,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
    "sample_expansion": {
        "execution_profiles": ["institutional"],
        "hold_days_grid": [1, 3],
        "trades_per_day_grid": [2, 3],
        "entry_days_grid": [3, 5, 7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.45,
            "min_crush_confidence": 0.25,
            "min_crush_magnitude": 0.05,
            "min_crush_edge": 0.015,
            "target_entry_dte": 6,
            "entry_dte_band": 6,
            "min_daily_share_volume": 1_000_000,
            "max_abs_momentum_5d": 0.11,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
    "variance_control": {
        "execution_profiles": ["institutional_tight", "institutional"],
        "hold_days_grid": [1],
        "trades_per_day_grid": [2],
        "entry_days_grid": [5, 7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.65,
            "min_crush_confidence": 0.50,
            "min_crush_magnitude": 0.09,
            "min_crush_edge": 0.025,
            "target_entry_dte": 6,
            "entry_dte_band": 4,
            "min_daily_share_volume": 10_000_000,
            "max_abs_momentum_5d": 0.09,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
    "alpha_focus": {
        "execution_profiles": ["institutional_tight"],
        "hold_days_grid": [1],
        "trades_per_day_grid": [1, 2],
        "entry_days_grid": [5, 7],
        "exit_days_grid": [1],
        "defaults": {
            "min_signal_score": 0.65,
            "min_crush_confidence": 0.50,
            "min_crush_magnitude": 0.09,
            "min_crush_edge": 0.03,
            "target_entry_dte": 6,
            "entry_dte_band": 3,
            "min_daily_share_volume": 5_000_000,
            "max_abs_momentum_5d": 0.08,
            "lookback_days": 1095,
            "max_backtest_symbols": 50,
        },
    },
}

_OOS_AUTO_PROFILE_ORDER: Tuple[str, ...] = (
    "evidence_balanced",
    "sample_expansion",
    "variance_control",
    "alpha_focus",
)


def _resolve_stability_profile(profile: Optional[str]) -> str:
    normalized = (profile or "stability_auto").strip().lower()
    if normalized == "stability_auto":
        return normalized
    if normalized in _OOS_STABILITY_PROFILES:
        return normalized
    return "stability_auto"


def _build_profiled_run_kwargs(
    request: OOSReportRequest,
    profile_name: str,
    train_days: int,
    test_days: int,
    step_days: int,
    use_profile_defaults: bool,
) -> Dict[str, Any]:
    profile = _OOS_STABILITY_PROFILES[profile_name]
    defaults = profile["defaults"]

    if use_profile_defaults:
        min_signal_score = float(defaults["min_signal_score"])
        min_crush_confidence = float(defaults["min_crush_confidence"])
        min_crush_magnitude = float(defaults["min_crush_magnitude"])
        min_crush_edge = float(defaults["min_crush_edge"])
        target_entry_dte = int(defaults["target_entry_dte"])
        entry_dte_band = int(defaults["entry_dte_band"])
        min_daily_share_volume = int(defaults["min_daily_share_volume"])
        max_abs_momentum_5d = float(defaults["max_abs_momentum_5d"])
    else:
        min_signal_score = max(float(request.min_signal_score), float(defaults["min_signal_score"]))
        min_crush_confidence = max(float(request.min_crush_confidence), float(defaults["min_crush_confidence"]))
        min_crush_magnitude = max(float(request.min_crush_magnitude), float(defaults["min_crush_magnitude"]))
        min_crush_edge = max(float(request.min_crush_edge), float(defaults["min_crush_edge"]))
        target_entry_dte = int(request.target_entry_dte)
        entry_dte_band = min(int(request.entry_dte_band), int(defaults["entry_dte_band"]))
        min_daily_share_volume = max(int(request.min_daily_share_volume), int(defaults["min_daily_share_volume"]))
        max_abs_momentum_5d = min(float(request.max_abs_momentum_5d), float(defaults["max_abs_momentum_5d"]))

    return {
        "execution_profiles": list(profile["execution_profiles"]),
        "hold_days_grid": list(profile["hold_days_grid"]),
        "signal_threshold_grid": [min_signal_score],
        "trades_per_day_grid": list(profile["trades_per_day_grid"]),
        "entry_days_grid": list(profile["entry_days_grid"]),
        "exit_days_grid": list(profile["exit_days_grid"]),
        "target_entry_dte": target_entry_dte,
        "entry_dte_band": max(1, entry_dte_band),
        "min_daily_share_volume": min_daily_share_volume,
        "max_abs_momentum_5d": max_abs_momentum_5d,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "top_n_train": request.oos_top_n_train,
        "lookback_days": max(int(request.lookback_days), int(defaults["lookback_days"])),
        "max_backtest_symbols": max(int(request.max_backtest_symbols), int(defaults["max_backtest_symbols"])),
        "use_crush_confidence_gate": True,
        "allow_global_crush_profile": True,
        "min_crush_confidence": min_crush_confidence,
        "min_crush_magnitude": min_crush_magnitude,
        "min_crush_edge": min_crush_edge,
        "allow_proxy_earnings": True,
        "min_splits": request.oos_min_splits,
        "min_total_test_trades": request.oos_min_total_test_trades,
        "min_trades_per_split": request.oos_min_trades_per_split,
        "output_dir": "exports/reports",
        "start_date": request.backtest_start_date,
        "end_date": request.backtest_end_date,
    }


def _compute_cross_split_sharpe(
    splits_detail: List[Dict[str, Any]],
    test_days: int,
) -> Optional[float]:
    """
    Fix 3: Portfolio-level Sharpe computed across OOS splits.

    Per-trade Sharpe (mean_trade_pnl / std_trade_pnl * sqrt(252/hold_days)) is
    inflated by low within-trade variance. The cross-split Sharpe treats each split
    as one 'period' return and annualises using the test window length — a more
    realistic portfolio-level measure.

    Formula: (mean_split_pnl / std_split_pnl) * sqrt(252 / test_days)
    """
    import math as _math
    if not splits_detail or test_days <= 0:
        return None
    pnls = [float(s["pnl"]) for s in splits_detail if s.get("pnl") is not None]
    if len(pnls) < 3:
        return None
    mean_p = sum(pnls) / len(pnls)
    var_p  = sum((x - mean_p) ** 2 for x in pnls) / (len(pnls) - 1)
    std_p  = _math.sqrt(var_p) if var_p > 0 else None
    if std_p is None or std_p < 1e-9:
        return None
    annualisation = _math.sqrt(max(252 / test_days, 1.0))
    sharpe = (mean_p / std_p) * annualisation
    return round(float(sharpe), 3) if _math.isfinite(sharpe) else None


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas/path-like objects to JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    # numpy/pandas scalar support (np.int64, np.float64, pd.Int64Dtype scalars, etc.)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    # numpy arrays / pandas containers
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass

    return str(value)


def _oos_report_card_from_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    report_card = result.get("report_card", {})
    return report_card if isinstance(report_card, dict) else {}


def _oos_result_score(result: Optional[Dict[str, Any]]) -> Tuple[int, int, int, int, float, float, float]:
    report_card = _oos_report_card_from_result(result)
    verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
    sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
    metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
    alpha = metrics.get("alpha", {}) if isinstance(metrics, dict) else {}
    sharpe = metrics.get("sharpe", {}) if isinstance(metrics, dict) else {}
    pnl = metrics.get("pnl", {}) if isinstance(metrics, dict) else {}

    overall_pass = 1 if bool(verdict.get("overall_pass")) else 0
    alpha_low = float(alpha.get("low", -999.0) or -999.0)
    sharpe_low = float(sharpe.get("low", -999.0) or -999.0)
    pnl_low = float(pnl.get("low", -999.0) or -999.0)
    ci_positive_count = int(alpha_low > 0.0) + int(sharpe_low > 0.0) + int(pnl_low > 0.0)
    total_trades = int(sample.get("total_test_trades", 0) or 0)
    splits = int(sample.get("splits", 0) or 0)
    return overall_pass, ci_positive_count, total_trades, splits, alpha_low, sharpe_low, pnl_low


def _oos_profile_summary(profile_name: str, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    report_card = _oos_report_card_from_result(result)
    verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
    sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
    metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
    return {
        "profile": profile_name,
        "grade": verdict.get("grade"),
        "overall_pass": verdict.get("overall_pass"),
        "total_test_trades": sample.get("total_test_trades"),
        "avg_trades_per_split": sample.get("avg_trades_per_split"),
        "alpha_low": (metrics.get("alpha", {}) if isinstance(metrics, dict) else {}).get("low"),
        "sharpe_low": (metrics.get("sharpe", {}) if isinstance(metrics, dict) else {}).get("low"),
        "pnl_low": (metrics.get("pnl", {}) if isinstance(metrics, dict) else {}).get("low"),
    }


def _oos_sample_insufficient(result: Optional[Dict[str, Any]]) -> bool:
    report_card = _oos_report_card_from_result(result)
    if not report_card:
        return True
    if not bool(report_card.get("ready", False)):
        return True
    verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
    if bool(verdict.get("overall_pass", False)):
        return False
    gates = report_card.get("gates", {}) if isinstance(report_card, dict) else {}
    for gate_name in ("min_splits", "min_total_test_trades", "min_trades_per_split"):
        gate = gates.get(gate_name, {}) if isinstance(gates, dict) else {}
        if isinstance(gate, dict) and gate.get("passed") is False:
            return True
    return False


def _run_oos_with_timeout(
    collector: Any,
    kwargs: Dict[str, Any],
    timeout_seconds: float = _OOS_VALIDATION_TIMEOUT_SECONDS,
) -> Optional[Dict[str, Any]]:
    """Run run_oos_validation in a daemon thread with a hard wall-clock timeout.

    PR-N: a cooperative cancel_event is passed into the worker. On
    timeout we set the event so the rolling-OOS loop breaks at the next
    split boundary, releasing the worker's pandas/duckdb memory instead
    of orphaning the thread to keep running until the process exits.

    Returns None on timeout. The thread may still be alive briefly after
    we return — it will exit cleanly once it reaches its next
    cancellation checkpoint (typically <60s for a typical split width).
    Exceptions from the worker propagate normally.
    """
    result_box: List[Any] = []
    exc_box: List[BaseException] = []
    cancel_event = threading.Event()
    worker_kwargs = dict(kwargs)
    worker_kwargs["cancel_event"] = cancel_event

    def _target() -> None:
        try:
            result_box.append(collector.run_oos_validation(**worker_kwargs))
        except Exception as exc:  # noqa: BLE001
            exc_box.append(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        # Signal cancel; the worker will exit at the next split boundary.
        # We deliberately do NOT t.join() again — we don't want the
        # request to block past the timeout. Daemon=True ensures the
        # thread can't keep the process alive at shutdown.
        cancel_event.set()
        logger.info(
            "OOS validation timed out after %.0fs; cancellation signalled",
            timeout_seconds,
        )
        return None
    if exc_box:
        raise exc_box[0]
    return result_box[0] if result_box else None


def _oos_sample_bottleneck_note(result: Optional[Dict[str, Any]]) -> Optional[str]:
    report_card = _oos_report_card_from_result(result)
    if not report_card:
        return None
    gates = report_card.get("gates", {}) if isinstance(report_card, dict) else {}
    if not isinstance(gates, dict):
        return None

    alpha_gate = gates.get("alpha_ci_positive", {}) if isinstance(gates.get("alpha_ci_positive"), dict) else {}
    sharpe_gate = gates.get("sharpe_ci_positive", {}) if isinstance(gates.get("sharpe_ci_positive"), dict) else {}
    pnl_gate = gates.get("pnl_ci_positive", {}) if isinstance(gates.get("pnl_ci_positive"), dict) else {}
    split_gate = gates.get("min_splits", {}) if isinstance(gates.get("min_splits"), dict) else {}
    trades_gate = gates.get("min_total_test_trades", {}) if isinstance(gates.get("min_total_test_trades"), dict) else {}
    per_split_gate = gates.get("min_trades_per_split", {}) if isinstance(gates.get("min_trades_per_split"), dict) else {}

    ci_positive_count = sum([
        bool(alpha_gate.get("passed")),
        bool(sharpe_gate.get("passed")),
        bool(pnl_gate.get("passed")),
    ])
    sample_failed = (
        split_gate.get("passed") is False
        or trades_gate.get("passed") is False
        or per_split_gate.get("passed") is False
    )
    if ci_positive_count >= 2 and sample_failed:
        return (
            "Signal quality passed CI gates but OOS sample size is insufficient. "
            "Increase history (e.g. run backfill with >=4 years) or use `sample_expansion` "
            "to build evidence before enforcing strict sample floors."
        )
    return None


def _oos_sparse_alpha_note(result: Optional[Dict[str, Any]]) -> Optional[str]:
    """Warn when profitable splits are fewer than half — returns are event-concentrated."""
    report_card = _oos_report_card_from_result(result)
    if not report_card:
        return None
    metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
    sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
    if not isinstance(metrics, dict) or not isinstance(sample, dict):
        return None
    rate = metrics.get("positive_alpha_split_rate")
    splits = sample.get("splits", 0)
    if isinstance(rate, (int, float)) and int(splits or 0) >= 8 and rate < 0.5:
        return (
            f"Alpha is positive in only {rate:.0%} of OOS splits. "
            "Returns are concentrated in a small number of earnings events; "
            "CI bounds may be unreliable."
        )
    return None


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/edge/analyze", response_model=EdgeAnalyzeResponse)
def analyze_edge(request: EdgeAnalyzeRequest) -> EdgeAnalyzeResponse:
    try:
        snapshot = analyze_single_ticker(request.symbol, mda_client=_get_mda_client())
        return EdgeAnalyzeResponse(
            generated_at=datetime.now(timezone.utc),
            symbol=snapshot.symbol,
            recommendation=snapshot.recommendation,
            confidence_pct=snapshot.confidence_pct,
            setup_score=snapshot.setup_score,
            metrics=snapshot.metrics,
            rationale=snapshot.rationale,
            selector_output=snapshot.selector_output,
            structure_scorecards=snapshot.structure_scorecards,
            vol_snapshot=snapshot.vol_snapshot,
        )
    except Exception as exc:
        _raise_public_error(400, "Analysis failed. Check the ticker, data availability, and provider configuration.", exc)


@app.get("/api/edge/screener", response_model=EdgeScreenerResponse)
def edge_screener(
    expiry_mode: Literal["front_after_earnings", "next_monthly_opex"] = "front_after_earnings",
    weeks: int = Query(6, ge=1, le=26),
    http_request: Request = None,
) -> EdgeScreenerResponse:
    if not _check_rate_limit(
        "screener", _client_ip(http_request),
        _SCREENER_RATE_LIMIT_PER_MIN, _SCREENER_RATE_LIMIT_PER_HOUR,
    ):
        raise HTTPException(
            status_code=429,
            detail="Screener rate limit exceeded. Try again shortly.",
        )
    try:
        payload = build_edge_screener(
            expiry_mode=expiry_mode,
            weeks=weeks,
            mda_client=_get_mda_client(),
        )
        return EdgeScreenerResponse(**payload)
    except Exception as exc:
        _raise_public_error(400, "Screener failed. Check data availability and provider configuration.", exc)


@app.get("/api/historical/options/symbols")
def historical_option_symbols() -> Dict[str, Any]:
    store = _get_feature_store()
    return {
        "generated_at": datetime.now(timezone.utc),
        "data_root": str(store.data_root),
        "feature_root": str(store.feature_root),
        "available": store.is_available(),
        "symbols": store.list_symbols(),
    }


@app.get("/api/historical/options/coverage", response_model=HistoricalOptionsCoverageResponse)
def historical_options_coverage() -> HistoricalOptionsCoverageResponse:
    try:
        coverage = _get_feature_store().coverage()
        return HistoricalOptionsCoverageResponse(
            generated_at=datetime.now(timezone.utc),
            coverage=_json_safe(coverage),
        )
    except OptionsFeatureStoreError as exc:
        raise HTTPException(status_code=404, detail="Historical options feature store is unavailable.")
    except Exception as exc:
        _raise_public_error(400, "Historical coverage query failed.", exc)


@app.get("/api/historical/options/{symbol}/coverage", response_model=HistoricalOptionsCoverageResponse)
def historical_options_symbol_coverage(symbol: str) -> HistoricalOptionsCoverageResponse:
    try:
        coverage = _get_feature_store().coverage(symbol=symbol)
        return HistoricalOptionsCoverageResponse(
            generated_at=datetime.now(timezone.utc),
            coverage=_json_safe(coverage),
        )
    except OptionsFeatureStoreError as exc:
        raise HTTPException(status_code=404, detail="Historical options feature store is unavailable for this symbol.")
    except ValueError as exc:
        _raise_public_error(400, "Invalid historical options coverage request.", exc)
    except Exception as exc:
        _raise_public_error(400, "Historical symbol coverage query failed.", exc)


@app.get("/api/historical/options/{symbol}/chain", response_model=HistoricalOptionsResponse)
def historical_options_chain(
    symbol: str,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    expiry: Optional[str] = None,
    min_dte: Optional[int] = None,
    max_dte: Optional[int] = None,
    call_put: Optional[str] = None,
    min_abs_delta: Optional[float] = None,
    max_abs_delta: Optional[float] = None,
    limit: int = 1_000,
) -> HistoricalOptionsResponse:
    filters = {
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
        "expiry": expiry,
        "min_dte": min_dte,
        "max_dte": max_dte,
        "call_put": call_put,
        "min_abs_delta": min_abs_delta,
        "max_abs_delta": max_abs_delta,
        "limit": limit,
    }
    try:
        rows = _get_feature_store().query_chain_records(
            symbol,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            expiry=expiry,
            min_dte=min_dte,
            max_dte=max_dte,
            call_put=call_put,
            min_abs_delta=min_abs_delta,
            max_abs_delta=max_abs_delta,
            limit=limit,
        )
        return HistoricalOptionsResponse(
            generated_at=datetime.now(timezone.utc),
            symbol=symbol.upper(),
            filters=_json_safe(filters),
            row_count=len(rows),
            rows=_json_safe(rows),
        )
    except OptionsFeatureStoreError as exc:
        raise HTTPException(status_code=404, detail="Historical options feature store is unavailable for this symbol.")
    except ValueError as exc:
        _raise_public_error(400, "Invalid historical options chain request.", exc)
    except Exception as exc:
        _raise_public_error(400, "Historical option query failed.", exc)


def _execute_oos_logic(request: OOSReportRequest) -> Dict[str, Any]:
    """Core OOS computation — returns a plain dict with 'summary', 'output_files',
    and 'generated_at' keys.  Raises raw exceptions; callers wrap as needed.
    Safe to call from a background thread."""
    collector = InstitutionalDataCollector()
    train_days = int(request.oos_train_days)
    test_days = int(request.oos_test_days)
    step_days = int(request.oos_step_days)
    adjustment_note = None
    notes: List[str] = []
    warnings: List[str] = []

    # Auto-fit OOS windows to available feature history so first-run users
    # get a diagnostic result instead of hard 422 failures.
    try:
        with sqlite3.connect(collector.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(date), MAX(date) FROM ml_features")
            row = cursor.fetchone() or (None, None)
        min_date, max_date = row
        if min_date and max_date:
            start_dt = datetime.fromisoformat(str(min_date))
            end_dt = datetime.fromisoformat(str(max_date))
            available_days = max(0, (end_dt - start_dt).days)
            if available_days > 0 and (train_days + test_days) > available_days:
                fitted_train = max(63, int(available_days * 0.60))
                fitted_test = max(21, int(available_days * 0.25))
                fitted_step = max(21, min(fitted_test, int(available_days * 0.20)))
                if (fitted_train + fitted_test) <= available_days:
                    adjustment_note = (
                        f"Adjusted OOS windows to dataset: "
                        f"train={fitted_train}, test={fitted_test}, step={fitted_step} "
                        f"(available_days={available_days})."
                    )
                    train_days, test_days, step_days = fitted_train, fitted_test, fitted_step
    except Exception:
        pass
    if adjustment_note:
        notes.append(adjustment_note)

    stability_profile_requested = _resolve_stability_profile(request.oos_stability_profile)
    stability_profile_used: str = stability_profile_requested
    profile_summaries: List[Dict[str, Any]] = []

    run_kwargs: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    if stability_profile_requested == "stability_auto":
        auto_candidates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for profile_name in _OOS_AUTO_PROFILE_ORDER:
            candidate_kwargs = _build_profiled_run_kwargs(
                request=request,
                profile_name=profile_name,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                use_profile_defaults=True,
            )
            candidate_result = _run_oos_with_timeout(collector, candidate_kwargs)
            if candidate_result:
                auto_candidates.append((profile_name, candidate_result, candidate_kwargs))
                profile_summaries.append(_oos_profile_summary(profile_name, candidate_result))
            elif candidate_result is None:
                notes.append(f"Auto profile `{profile_name}` produced no OOS rows.")

        if auto_candidates:
            stability_profile_used, result, run_kwargs = max(
                auto_candidates,
                key=lambda row: _oos_result_score(row[1]),
            )
            notes.append(f"Stability auto selected `{stability_profile_used}` profile.")
        else:
            stability_profile_used = "evidence_balanced"
            run_kwargs = _build_profiled_run_kwargs(
                request=request,
                profile_name=stability_profile_used,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                use_profile_defaults=True,
            )
            result = None
    else:
        run_kwargs = _build_profiled_run_kwargs(
            request=request,
            profile_name=stability_profile_requested,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            use_profile_defaults=False,
        )
        result = _run_oos_with_timeout(collector, run_kwargs)
        profile_summaries.append(_oos_profile_summary(stability_profile_requested, result))

    # Evidence-first fallback: if sample coverage is weak, rerun once with
    # larger universe and denser rolling windows to increase OOS evidence.
    adaptive_kwargs = dict(run_kwargs)
    adaptive_kwargs["lookback_days"] = max(int(run_kwargs["lookback_days"]), 1095)
    adaptive_kwargs["max_backtest_symbols"] = max(int(run_kwargs["max_backtest_symbols"]), 50)
    adaptive_kwargs["train_days"] = max(63, min(int(run_kwargs["train_days"]), 189))
    adaptive_kwargs["test_days"] = max(21, min(int(run_kwargs["test_days"]), 42))
    adaptive_kwargs["step_days"] = max(21, min(int(run_kwargs["step_days"]), 42))
    adaptive_changed = any(
        adaptive_kwargs[key] != run_kwargs[key]
        for key in ("lookback_days", "max_backtest_symbols", "train_days", "test_days", "step_days")
    )
    adaptive_used = False
    if adaptive_changed and (result is None or _oos_sample_insufficient(result)):
        retry_result = _run_oos_with_timeout(collector, adaptive_kwargs)
        if retry_result:
            baseline_score = _oos_result_score(result)
            retry_score = _oos_result_score(retry_result)
            if result is None or retry_score > baseline_score:
                result = retry_result
                train_days = int(adaptive_kwargs["train_days"])
                test_days = int(adaptive_kwargs["test_days"])
                step_days = int(adaptive_kwargs["step_days"])
                adaptive_used = True
                stability_profile_used = (
                    f"{stability_profile_used}+adaptive"
                    if not stability_profile_used.endswith("+adaptive")
                    else stability_profile_used
                )
                notes.append(
                    "Adaptive OOS retry applied (evidence-first): "
                    f"symbols={adaptive_kwargs['max_backtest_symbols']}, "
                    f"lookback={adaptive_kwargs['lookback_days']}, "
                    f"train/test/step={train_days}/{test_days}/{step_days}."
                )
            else:
                notes.append("Adaptive OOS retry completed but baseline run retained.")
        else:
            notes.append("Adaptive OOS retry produced no rows.")

    if not result:
        timeout_secs = int(_OOS_VALIDATION_TIMEOUT_SECONDS)
        warnings.append(
            f"OOS validation did not complete within {timeout_secs}s. "
            "Reduce lookback_days or max_backtest_symbols and retry."
        )
        summary = _json_safe({
            "status": "no_oos_rows",
            "grade": "N/A",
            "overall_pass": False,
            "message": "No OOS rows produced for current filters/history.",
            "windows_used": {
                "train_days": train_days,
                "test_days": test_days,
                "step_days": step_days,
                "lookback_days": int(run_kwargs["lookback_days"]),
                "max_backtest_symbols": int(run_kwargs["max_backtest_symbols"]),
            },
            "stability_profile_requested": stability_profile_requested,
            "stability_profile_used": stability_profile_used,
            "stability_profiles_evaluated": profile_summaries,
            "adaptive_retry_used": adaptive_used,
            "notes": notes,
            "warnings": warnings,
        })
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "output_files": {},
        }

    report_card = result.get("report_card", {}) if isinstance(result, dict) else {}
    verdict = report_card.get("verdict", {}) if isinstance(report_card, dict) else {}
    sample = report_card.get("sample", {}) if isinstance(report_card, dict) else {}
    metrics = report_card.get("metrics", {}) if isinstance(report_card, dict) else {}
    gates = report_card.get("gates", {}) if isinstance(report_card, dict) else {}
    sample_bottleneck_note = _oos_sample_bottleneck_note(result)
    if sample_bottleneck_note:
        warnings.append(sample_bottleneck_note)
    sparse_alpha_note = _oos_sparse_alpha_note(result)
    if sparse_alpha_note:
        warnings.append(sparse_alpha_note)
    summary = {
        "splits": int(result.get("splits", 0)),
        "best_params": result.get("best_params", {}),
        "grade": verdict.get("grade"),
        "overall_pass": verdict.get("overall_pass"),
        "verdict": verdict,
        "sample": sample,
        "metrics": metrics,
        "gates": gates,
        "splits_detail": result.get("splits_detail", []),
        "cross_split_sharpe": _compute_cross_split_sharpe(
            result.get("splits_detail", []), test_days
        ),
        "status": "ok",
        "windows_used": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "lookback_days": int(adaptive_kwargs["lookback_days"] if adaptive_used else run_kwargs["lookback_days"]),
            "max_backtest_symbols": int(adaptive_kwargs["max_backtest_symbols"] if adaptive_used else run_kwargs["max_backtest_symbols"]),
        },
        "stability_profile_requested": stability_profile_requested,
        "stability_profile_used": stability_profile_used,
        "stability_profiles_evaluated": profile_summaries,
        "adaptive_retry_used": adaptive_used,
        "notes": notes,
        "warnings": warnings,
    }
    output_files = {
        "csv": result.get("csv_path"),
        "summary_markdown": result.get("markdown_path"),
        "summary_json": result.get("json_path"),
        "report_card_markdown": result.get("report_card_markdown_path"),
        "report_card_json": result.get("report_card_json_path"),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": _json_safe(summary),
        "output_files": _json_safe(output_files),
    }


@app.post("/api/oos/report-card", response_model=OOSReportResponse)
def run_oos_report_card(
    request: OOSReportRequest, http_request: Request
) -> OOSReportResponse:
    """Synchronous OOS endpoint (legacy). Prefer /api/oos/submit for long runs.

    Codex web-audit follow-up P2: shares the ``oos_submit`` rate-limit
    bucket with the async submit endpoint — otherwise an auth'd client
    (or stolen cookie) could bypass the new /api/oos/submit limit by
    hammering this synchronous variant instead, burning a worker thread
    per call for the full _OOS_VALIDATION_TIMEOUT_SECONDS wall-clock.
    Same bucket key (`"oos_submit"`) on purpose: the user's *intent* is
    the same (run an OOS analysis), regardless of which endpoint they
    chose, so attempts should count against a single quota.
    """
    if not _check_rate_limit(
        "oos_submit", _client_ip(http_request),
        _OOS_SUBMIT_RATE_LIMIT_PER_MIN, _OOS_SUBMIT_RATE_LIMIT_PER_HOUR,
    ):
        raise HTTPException(
            status_code=429,
            detail="OOS submission rate limit exceeded. Wait a minute and try again.",
        )
    # F3: shared concurrency guard. Register this sync job in _oos_jobs so
    # _active_job_count_locked counts it alongside async submit jobs. Both
    # paths then see the same cap, preventing a running async job from being
    # bypassed by a simultaneous sync call (or vice-versa).
    sync_job_id = f"sync-{uuid.uuid4()}"
    with _oos_jobs_lock:
        _purge_jobs_locked(_oos_jobs, ttl_seconds=_OOS_JOB_TTL_SECONDS)
        if _active_job_count_locked(_oos_jobs) >= _MAX_OOS_RUNNING_JOBS:
            raise HTTPException(
                status_code=429,
                detail="OOS job capacity reached. Wait for the current job to finish before submitting another.",
            )
        _oos_jobs[sync_job_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "data": None,
            "error": None,
        }
    try:
        data = _execute_oos_logic(request)
        return OOSReportResponse(
            generated_at=datetime.fromisoformat(data["generated_at"]),
            summary=data["summary"],
            output_files=data["output_files"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_public_error(500, "OOS report generation failed.", exc)
    finally:
        with _oos_jobs_lock:
            _oos_jobs.pop(sync_job_id, None)


# ── Async OOS job store ───────────────────────────────────────────────────────
# Simple in-process dict; keys are UUID strings, values track job state.
# Sufficient for single-instance deployment; replaced by a task queue (Celery,
# ARQ) when moving to multi-process / multi-worker.

_oos_jobs: Dict[str, Dict[str, Any]] = {}
_oos_jobs_lock = threading.Lock()

# Fix 2: TTL for completed/errored jobs.  Pending/running jobs are never evicted.
# Two hours is long enough for a user to poll the result; short enough that a
# server handling many sessions doesn't accumulate unbounded state.
_OOS_JOB_TTL_SECONDS: float = 7200.0


def _purge_stale_oos_jobs() -> None:
    """Evict completed or errored OOS jobs older than TTL.

    Called on every new submit so cleanup is lazy (no background thread needed).
    Access is lock-protected because submit, poll, and worker threads can all
    touch the in-process job store.
    """
    with _oos_jobs_lock:
        _purge_jobs_locked(_oos_jobs, ttl_seconds=_OOS_JOB_TTL_SECONDS)


def _purge_jobs_locked(jobs: Dict[str, Dict[str, Any]], *, ttl_seconds: float) -> None:
    now = datetime.now(timezone.utc)
    to_delete: List[str] = []
    for jid, job in jobs.items():
        if job.get("status") not in {"complete", "error"}:
            continue
        try:
            age = (now - datetime.fromisoformat(job["started_at"])).total_seconds()
            if age > ttl_seconds:
                to_delete.append(jid)
        except Exception:
            to_delete.append(jid)
    if len(jobs) - len(to_delete) > _MAX_RETAINED_JOBS:
        completed = [
            (jid, job.get("started_at", ""))
            for jid, job in jobs.items()
            if job.get("status") in {"complete", "error"} and jid not in to_delete
        ]
        completed.sort(key=lambda item: item[1])
        overflow = (len(jobs) - len(to_delete)) - _MAX_RETAINED_JOBS
        to_delete.extend(jid for jid, _ in completed[:max(0, overflow)])
    for jid in to_delete:
        jobs.pop(jid, None)


def _active_job_count_locked(jobs: Dict[str, Dict[str, Any]]) -> int:
    return sum(1 for job in jobs.values() if job.get("status") in {"pending", "running"})


def _oos_job_worker(job_id: str, request: OOSReportRequest) -> None:
    """Background thread target — runs OOS logic and writes result to _oos_jobs."""
    with _oos_jobs_lock:
        if job_id in _oos_jobs:
            _oos_jobs[job_id]["status"] = "running"
    try:
        data = _execute_oos_logic(request)
        with _oos_jobs_lock:
            if job_id in _oos_jobs:
                _oos_jobs[job_id].update({"status": "complete", "data": data, "error": None})
    except Exception as exc:
        logger.exception("OOS background job failed: %s", exc)
        with _oos_jobs_lock:
            if job_id in _oos_jobs:
                _oos_jobs[job_id].update({"status": "error", "data": None, "error": "OOS job failed."})


@app.post("/api/oos/submit")
def submit_oos_job(request: OOSReportRequest, http_request: Request) -> Dict[str, str]:
    """Submit an OOS job and return immediately with a job_id.
    Poll GET /api/oos/status/{job_id} every 2–3 s for completion.
    """
    # Web-audit P2-3: cap how fast a single client can fire OOS jobs.
    # The concurrency lock below already caps in-flight jobs at 1, but
    # without a rate limit an authenticated client (or stolen cookie)
    # could submit then poll-wait-then-resubmit in a tight loop to burn
    # CPU continuously. Defaults: 3/min, 20/hour per IP.
    if not _check_rate_limit(
        "oos_submit", _client_ip(http_request),
        _OOS_SUBMIT_RATE_LIMIT_PER_MIN, _OOS_SUBMIT_RATE_LIMIT_PER_HOUR,
    ):
        raise HTTPException(
            status_code=429,
            detail="OOS submission rate limit exceeded. Wait a minute and try again.",
        )
    job_id = str(uuid.uuid4())
    with _oos_jobs_lock:
        _purge_jobs_locked(_oos_jobs, ttl_seconds=_OOS_JOB_TTL_SECONDS)
        if _active_job_count_locked(_oos_jobs) >= _MAX_OOS_RUNNING_JOBS:
            raise HTTPException(
                status_code=429,
                detail="OOS job capacity reached. Wait for the current job to finish before submitting another.",
            )
        _oos_jobs[job_id] = {
            "status": "pending",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "data": None,
            "error": None,
        }
    t = threading.Thread(target=_oos_job_worker, args=(job_id, request), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/oos/status/{job_id}")
def get_oos_job_status(job_id: str) -> Dict[str, Any]:
    """Poll job status. Returns status, elapsed_sec, and (when complete) data."""
    with _oos_jobs_lock:
        job = dict(_oos_jobs.get(job_id) or {})
    if not job:
        raise HTTPException(status_code=404, detail=f"OOS job {job_id!r} not found")
    elapsed_sec: Optional[int] = None
    try:
        start = datetime.fromisoformat(job["started_at"])
        elapsed_sec = int((datetime.now(timezone.utc) - start).total_seconds())
    except Exception:
        pass
    return {
        "job_id": job_id,
        "status": job["status"],
        "elapsed_sec": elapsed_sec,
        "data": job.get("data"),
        "error": job.get("error"),
    }


# ── ML training job store ─────────────────────────────────────────────────────

_ml_train_jobs: Dict[str, Dict[str, Any]] = {}
_ml_train_jobs_lock = threading.Lock()


def _ml_train_worker(job_id: str) -> None:
    """Background thread: calibrates labels, trains the legacy post-event-vol classifier, reloads it."""
    with _ml_train_jobs_lock:
        if job_id in _ml_train_jobs:
            _ml_train_jobs[job_id]["status"] = "running"
    try:
        collector = InstitutionalDataCollector()
        # Always calibrate labels first — this pairs any new pre/post snapshots from
        # a recent backfill into earnings_iv_decay_labels before training reads from it.
        collector.db.calibrate_earnings_iv_decay_labels()
        result = collector.db.train_ml_model_on_historical_spreads()
        # Reload the model in the edge engine so subsequent analyses pick it up
        from web.api.edge_engine import reload_crush_model
        reload_crush_model()
        with _ml_train_jobs_lock:
            if job_id in _ml_train_jobs:
                _ml_train_jobs[job_id].update({"status": "complete", "data": result, "error": None})
    except Exception as exc:
        logger.exception("ML training background job failed: %s", exc)
        with _ml_train_jobs_lock:
            if job_id in _ml_train_jobs:
                _ml_train_jobs[job_id].update({"status": "error", "data": None, "error": "ML training job failed."})


@app.post("/api/ml/train")
def submit_ml_train_job(http_request: Request) -> Dict[str, str]:
    """Train the legacy post-event volatility classifier in the background.
    Returns job_id — poll GET /api/ml/train-status/{job_id} for completion.
    Training requires earnings_iv_decay_labels rows (run backfill first).
    """
    # Web-audit P2-3: same shape as the OOS rate limit — bound how fast a
    # single client can fire trainings. A 1-job concurrency lock isn't
    # enough on its own; without a rate limit an auth'd client could
    # loop submit-then-wait-then-resubmit and drive sustained CPU plus
    # ~/.options_calculator_pro/models/crush_model_meta.json churn.
    # Defaults: 2/min, 10/hour per IP.
    if not _check_rate_limit(
        "ml_train", _client_ip(http_request),
        _ML_TRAIN_RATE_LIMIT_PER_MIN, _ML_TRAIN_RATE_LIMIT_PER_HOUR,
    ):
        raise HTTPException(
            status_code=429,
            detail="ML train rate limit exceeded. Wait a minute and try again.",
        )
    job_id = str(uuid.uuid4())
    with _ml_train_jobs_lock:
        _purge_jobs_locked(_ml_train_jobs, ttl_seconds=_OOS_JOB_TTL_SECONDS)
        if _active_job_count_locked(_ml_train_jobs) >= _MAX_ML_RUNNING_JOBS:
            raise HTTPException(
                status_code=429,
                detail="ML training capacity reached. Wait for the current job to finish before submitting another.",
            )
        _ml_train_jobs[job_id] = {
            "status": "pending",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "data": None,
            "error": None,
        }
    t = threading.Thread(target=_ml_train_worker, args=(job_id,), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/ml/status")
def get_ml_model_status() -> Dict[str, Any]:
    """Return the last-trained ML model metadata (from persisted crush_model_meta.json).

    Fields: trained_at, n_events, crush_rate, cv_auc, cv_auc_std, cv_accuracy,
            precision_at_threshold, recall_at_threshold, insample_auc, insample_brier,
            features, label, algorithm.
    Returns 404 if no model has been trained yet.
    """
    import json as _json
    meta_path = os.path.expanduser("~/.options_calculator_pro/models/crush_model_meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(
            status_code=404,
            detail="No trained model found. POST /api/ml/train to train first.",
        )
    try:
        with open(meta_path) as f:
            meta = _json.load(f)
        return {"status": "ok", "model_metadata": meta}
    except Exception as exc:
        _raise_public_error(500, "Failed to read model metadata.", exc)


@app.get("/api/ml/train-status/{job_id}")
def get_ml_train_status(job_id: str) -> Dict[str, Any]:
    """Poll ML training job. Returns status, elapsed_sec, metrics when complete."""
    with _ml_train_jobs_lock:
        job = dict(_ml_train_jobs.get(job_id) or {})
    if not job:
        raise HTTPException(status_code=404, detail=f"ML training job {job_id!r} not found")
    elapsed_sec: Optional[int] = None
    try:
        start = datetime.fromisoformat(job["started_at"])
        elapsed_sec = int((datetime.now(timezone.utc) - start).total_seconds())
    except Exception:
        pass
    return {
        "job_id": job_id,
        "status": job["status"],
        "elapsed_sec": elapsed_sec,
        "data": job.get("data"),
        "error": job.get("error"),
    }


# ── Ranked screener endpoint ─────────────────────────────────────────────────

@app.get("/api/screener/ranked", response_model=RankedScreenerResponse)
def ranked_screener(
    dte_min: int = Query(3, ge=0, le=60),
    dte_max: int = Query(10, ge=1, le=120),
    min_sample_size: int = Query(4, ge=0, le=40),
    release_filter: str = Query("all", pattern="^(all|bmo|amc|dmh|unknown)$"),
    weeks: int = Query(4, ge=1, le=26),
    symbols: Optional[str] = Query(None, max_length=2000),
    http_request: Request = None,
) -> RankedScreenerResponse:
    """
    Rank upcoming earnings candidates by pre-earnings long-vega setup quality.

    Query params
    ------------
    dte_min / dte_max  : DTE window for entry (default 3–10)
    min_sample_size    : Minimum earnings history required (default 4)
    release_filter     : ``all``, ``bmo``, or ``amc``
    weeks              : Look-ahead window in weeks (default 4)
    symbols            : Comma-separated override universe (default: DEFAULT_UNIVERSE)
    """
    if not _check_rate_limit(
        "screener", _client_ip(http_request),
        _SCREENER_RATE_LIMIT_PER_MIN, _SCREENER_RATE_LIMIT_PER_HOUR,
    ):
        raise HTTPException(
            status_code=429,
            detail="Screener rate limit exceeded. Try again shortly.",
        )
    from services.screener_service import build_ranked_screener, DEFAULT_UNIVERSE
    from datetime import date as _date

    today = _date.today()
    if dte_min > dte_max:
        raise HTTPException(status_code=400, detail="dte_min must be less than or equal to dte_max.")
    universe = _parse_symbol_universe(symbols, DEFAULT_UNIVERSE)

    try:
        payload = build_ranked_screener(
            symbols=universe,
            dte_min=dte_min,
            dte_max=dte_max,
            min_sample_size=min_sample_size,
            release_filter=release_filter,
            weeks=weeks,
            today=today,
        )
    except Exception as exc:
        _raise_public_error(400, "Ranked screener failed. Check data availability and request parameters.", exc)

    rows = [RankedSetupRow(**r) for r in payload["rows"]]
    return RankedScreenerResponse(
        generated_at=datetime.now(timezone.utc),
        as_of_date=payload["as_of_date"],
        universe_size=payload["universe_size"],
        rows_returned=payload["rows_returned"],
        in_entry_window=payload["in_entry_window"],
        ranking_weights=payload["ranking_weights"],
        strategy_note=payload["strategy_note"],
        rows=rows,
    )


# ── Calibration endpoint ──────────────────────────────────────────────────────

@app.get("/api/calibration/curve", response_model=CalibrationCurveResponse)
def calibration_curve() -> CalibrationCurveResponse:
    """
    Return the setup_score → expected IV expansion calibration curve.

    Phases are intentionally conservative:
    - ``bootstrap_prior``: research prior only, not empirical.
    - ``observational``: raw bucket observations where available, no fitted curve.
    - ``fitted_moderate`` / ``fitted_high``: isotonic fit with moderate/high sample depth.
    """
    from services.calibration_service import get_calibration

    cal = get_calibration()
    diag = cal.diagnostics()
    buckets_raw = cal.get_curve_summary()
    buckets = [CalibrationBucket(**b) for b in buckets_raw]
    return CalibrationCurveResponse(
        generated_at=datetime.now(timezone.utc),
        phase=diag["phase"],
        n_observations=diag["n_observations"],
        min_for_observational=diag["min_for_observational"],
        min_for_fit=diag["min_for_fit"],
        min_for_high_fit=diag["min_for_high_fit"],
        buckets=buckets,
    )


@app.get("/api/diagnostics/learning", response_model=LearningDiagnosticsResponse)
def learning_diagnostics() -> LearningDiagnosticsResponse:
    from services.learning_diagnostics import build_learning_diagnostics

    return LearningDiagnosticsResponse(**build_learning_diagnostics())


@app.get("/api/diagnostics/data-quality")
def data_quality_diagnostics() -> Dict[str, Any]:
    from services.data_quality_diagnostics import build_data_quality_diagnostics

    try:
        return build_data_quality_diagnostics()
    except Exception as exc:
        _raise_public_error(500, "Data-quality diagnostics failed.", exc)


@app.get("/api/diagnostics/provider-telemetry")
def provider_telemetry_diagnostics(
    limit: int = 100,
    offset: int = 0,
    provider: Optional[str] = None,
    endpoint_type: Optional[str] = None,
    symbol: Optional[str] = None,
    success: Optional[bool] = None,
    failures_only: bool = False,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> Dict[str, Any]:
    from services.provider_telemetry import build_provider_telemetry_diagnostics

    try:
        return build_provider_telemetry_diagnostics(
            limit=limit,
            offset=offset,
            provider=provider,
            endpoint_type=endpoint_type,
            symbol=symbol,
            success=success,
            failures_only=failures_only,
            since=since,
            until=until,
        )
    except Exception as exc:
        _raise_public_error(500, "Provider telemetry diagnostics failed.", exc)


@app.get("/api/diagnostics/forward-performance")
def forward_performance_diagnostics(limit: int = 10_000, recent_limit: int = 25) -> Dict[str, Any]:
    from services.forward_performance_diagnostics import build_forward_performance_diagnostics

    try:
        return build_forward_performance_diagnostics(max_rows=limit, recent_limit=recent_limit)
    except Exception as exc:
        _raise_public_error(500, "Forward-performance diagnostics failed.", exc)


@app.get("/api/diagnostics/evidence-report")
def evidence_report_diagnostics(limit: int = 10_000, recent_limit: int = 25) -> Dict[str, Any]:
    from services.evidence_report import build_evidence_report

    try:
        return build_evidence_report(max_rows=limit, recent_limit=recent_limit)
    except Exception as exc:
        _raise_public_error(500, "Evidence report diagnostics failed.", exc)


# ── Recommendation ledger diagnostics ─────────────────────────────────────────

def _ledger_row_summary(row: Dict[str, Any], outcome: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "recommendation_id": row.get("recommendation_id"),
        "created_at": row.get("created_at"),
        "symbol": row.get("symbol"),
        "as_of_date": row.get("as_of_date"),
        "earnings_date": row.get("earnings_date"),
        "recommendation": row.get("recommendation"),
        "selected_structure": row.get("selected_structure"),
        "no_trade_reason": row.get("no_trade_reason"),
        "data_quality_score": row.get("data_quality_score"),
        "earnings_source": row.get("earnings_source"),
        "earnings_source_confidence": row.get("earnings_source_confidence"),
        "earnings_source_stale": bool(row.get("earnings_source_stale")),
        "quote_source": row.get("quote_source"),
        "quote_quality": row.get("quote_quality"),
        "outcome_status": outcome.get("status") if outcome else None,
        "trade_id": outcome.get("trade_id") if outcome else None,
        "realized_return_pct": outcome.get("realized_return_pct") if outcome else None,
        "realized_expansion_pct": outcome.get("realized_expansion_pct") if outcome else None,
    }


def _ledger_with_outcome(row: Dict[str, Any]) -> Dict[str, Any]:
    from services.outcome_recorder import get_outcome_store

    try:
        outcome = get_outcome_store().find_by_recommendation_id(str(row.get("recommendation_id")))
    except Exception as exc:
        logger.warning("Ledger outcome linkage lookup failed: %s", exc)
        outcome = None
    return _ledger_row_summary(row, outcome)


def _ledger_page_payload(*, rows: List[Dict[str, Any]], total: int, limit: int, offset: int) -> Dict[str, Any]:
    safe_limit = max(1, min(int(limit or 50), 500))
    safe_offset = max(0, int(offset or 0))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "total": int(total),
        "limit": safe_limit,
        "offset": safe_offset,
        "has_more": safe_offset + len(rows) < int(total),
        "rows": [_ledger_with_outcome(row) for row in rows],
    }


def _csv_download(filename: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> Response:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fieldnames})
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


_LEDGER_EXPORT_FIELDS = [
    "recommendation_id",
    "created_at",
    "symbol",
    "as_of_date",
    "earnings_date",
    "recommendation",
    "selected_structure",
    "no_trade_reason",
    "data_quality_score",
    "earnings_source",
    "earnings_source_confidence",
    "earnings_source_stale",
    "quote_source",
    "quote_quality",
    "outcome_status",
    "trade_id",
    "realized_return_pct",
    "realized_expansion_pct",
]


@app.get("/api/diagnostics/recommendations/summary")
def recommendation_ledger_summary() -> Dict[str, Any]:
    from services.recommendation_ledger import get_recommendation_ledger

    try:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": get_recommendation_ledger().summarize(),
        }
    except Exception as exc:
        _raise_public_error(500, "Recommendation ledger summary failed.", exc)


@app.get("/api/diagnostics/recommendations/symbol/{symbol}")
def recommendation_ledger_search_by_symbol(symbol: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    from services.recommendation_ledger import get_recommendation_ledger

    try:
        ledger = get_recommendation_ledger()
        rows = ledger.list_recent(limit=limit, offset=offset, symbol=symbol)
        payload = _ledger_page_payload(rows=rows, total=ledger.count(symbol=symbol), limit=limit, offset=offset)
        payload["symbol"] = symbol.upper()
        return payload
    except Exception as exc:
        _raise_public_error(500, "Recommendation ledger symbol search failed.", exc)


@app.get("/api/diagnostics/recommendations/export")
def recommendation_ledger_export(
    format: Literal["json", "csv"] = "json",
    limit: int = Query(500, ge=1, le=500),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = None,
) -> Any:
    from services.recommendation_ledger import get_recommendation_ledger

    try:
        ledger = get_recommendation_ledger()
        rows = [_ledger_with_outcome(row) for row in ledger.list_recent(limit=limit, offset=offset, symbol=symbol)]
        if format == "csv":
            return _csv_download("recommendation_ledger.csv", rows, _LEDGER_EXPORT_FIELDS)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "export_type": "recommendations",
            "count": len(rows),
            "total": ledger.count(symbol=symbol),
            "limit": limit,
            "offset": offset,
            "symbol": symbol.upper() if symbol else None,
            "rows": rows,
        }
    except Exception as exc:
        _raise_public_error(500, "Recommendation ledger export failed.", exc)


@app.get("/api/diagnostics/recommendations/linkage/export")
def recommendation_ledger_linkage_export(
    format: Literal["json", "csv"] = "json",
    limit: int = Query(500, ge=1, le=500),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = None,
) -> Any:
    from services.outcome_recorder import get_outcome_store
    from services.recommendation_ledger import get_recommendation_ledger

    try:
        ledger = get_recommendation_ledger()
        outcome_store = get_outcome_store()
        linkages = []
        for row in ledger.list_recent(limit=limit, offset=offset, symbol=symbol):
            outcome = outcome_store.find_by_recommendation_id(str(row.get("recommendation_id")))
            summary = _ledger_row_summary(row, outcome)
            linkages.append(
                {
                    **summary,
                    "has_linked_outcome": outcome is not None,
                    "paper_trade_status": outcome.get("status") if outcome else None,
                    "paper_trade_source_type": outcome.get("source_type") if outcome else None,
                }
            )
        if format == "csv":
            return _csv_download(
                "recommendation_linkage_ledger.csv",
                linkages,
                [*_LEDGER_EXPORT_FIELDS, "has_linked_outcome", "paper_trade_status", "paper_trade_source_type"],
            )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "export_type": "recommendation_linkages",
            "count": len(linkages),
            "total": ledger.count(symbol=symbol),
            "limit": limit,
            "offset": offset,
            "symbol": symbol.upper() if symbol else None,
            "rows": linkages,
        }
    except Exception as exc:
        _raise_public_error(500, "Recommendation linkage export failed.", exc)


@app.get("/api/diagnostics/recommendations")
def recommendation_ledger_recent(limit: int = 50, offset: int = 0, symbol: Optional[str] = None) -> Dict[str, Any]:
    from services.recommendation_ledger import get_recommendation_ledger

    try:
        ledger = get_recommendation_ledger()
        rows = ledger.list_recent(limit=limit, offset=offset, symbol=symbol)
        return _ledger_page_payload(rows=rows, total=ledger.count(symbol=symbol), limit=limit, offset=offset)
    except Exception as exc:
        _raise_public_error(500, "Recommendation ledger query failed.", exc)


@app.get("/api/diagnostics/recommendations/{recommendation_id}/linkage")
def recommendation_ledger_linkage(recommendation_id: str) -> Dict[str, Any]:
    from services.outcome_recorder import get_outcome_store
    from services.recommendation_ledger import get_recommendation_ledger

    try:
        row = get_recommendation_ledger().get(recommendation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Recommendation not found.")
        outcome = get_outcome_store().find_by_recommendation_id(recommendation_id)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "recommendation_id": recommendation_id,
            "recommendation": _ledger_row_summary(row, outcome),
            "paper_trade": outcome,
            "has_linked_outcome": outcome is not None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        _raise_public_error(500, "Recommendation linkage query failed.", exc)


@app.get("/api/diagnostics/recommendations/{recommendation_id}")
def recommendation_ledger_detail(recommendation_id: str) -> Dict[str, Any]:
    from services.recommendation_ledger import get_recommendation_ledger

    try:
        row = get_recommendation_ledger().get(recommendation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Recommendation not found.")
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "recommendation": _ledger_row_summary(row),
            "vol_snapshot": row.get("vol_snapshot_json") or {},
            "structure_scorecards": row.get("structure_scorecards_json") or [],
            "selector_output": row.get("selector_output_json") or {},
            "selector_explanation": row.get("explanation_json") or {},
            "quote_provenance": {
                "quote_timestamp": row.get("quote_timestamp"),
                "quote_source": row.get("quote_source"),
                "quote_quality": row.get("quote_quality"),
                "bid_ask_mid": row.get("bid_ask_mid_json") or {},
                "providers": row.get("provider_names_json") or {},
            },
            "metadata": row.get("metadata_json") or {},
            "schema_version": row.get("schema_version"),
            "engine_version": row.get("engine_version"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        _raise_public_error(500, "Recommendation ledger detail failed.", exc)


# ── Serve the built React frontend (must be LAST — catches all remaining paths) ─
_PRODUCT_DOCS = project_root / "docs"
if _PRODUCT_DOCS.exists():
    app.mount("/product-docs", StaticFiles(directory=str(_PRODUCT_DOCS), html=False), name="product-docs")

_FRONTEND_DIST = project_root / "web" / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
