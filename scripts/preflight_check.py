#!/usr/bin/env python3
"""Pre-deployment / pre-start smoke check.

Reads the operator's current state and verifies the things that
would otherwise fail at runtime or, worse, fail silently. Designed to
be run **before** flipping ``ENABLE_SHARE_AUTH=true``, before
installing the launchd jobs, and as a sanity sweep before any
non-trivial restart.

Codex scope (PR #62 follow-up): keep it small. This is "am I safe to
deploy/run?", not a giant validator. The runbook
(``docs/DEPLOYMENT.md``) tells a human what to verify; this script
turns the highest-risk parts into executable checks.

Checks (all read-only, none mutate state):

  1. .env file exists and is readable.
  2. ``${PROJECT_ROOT}/.venv311/bin/python`` is present and executable.
  3. Frontend ``web/frontend/dist/`` exists (WARN if missing — dev mode
     is fine, production-same-origin needs the built bundle).
  4. Every launchd wrapper the installer references (``scripts/automation/run_*.sh``)
     exist and are executable.
  5. Every plist template the installer lists exists and contains the placeholders the
     install script substitutes.
  6. LaunchAgents installed under ``~/Library/LaunchAgents`` — PASS if
     all 5; WARN if partial; SKIP if none (likely a fresh checkout or
     a dev box).
  7. ``web.api.app`` imports cleanly — exercises the PR #59 startup
     validators (SESSION_SECRET ≥24 chars, ALLOWED_ORIGINS not '*' /
     'null', Secure-cookie posture, etc.). Run in a subprocess so the
     parent process doesn't get polluted by the import.
  8. Known SQLite stores under ``~/.options_calculator_pro/`` pass
     ``PRAGMA quick_check`` if they exist (SKIP if not yet).
  9. ``scripts/rotate_launchd_logs.py --dry-run`` exits 0.
 10. Optional ``GET /api/health`` if a backend is already running —
     SKIP if connection refused (we never start uvicorn here; it's
     the operator's job).
 11. IB Gateway, when ``OPTIONS_CALCULATOR_OPTIONS_PROVIDER=ib_gateway``:
     its API port accepts connections (WARN if not — start it and log
     in) and is not bound on every interface (WARN — restrict it to
     localhost in the gateway's API settings). SKIP for other providers.
 12. Power management on this Mac: WARN when the system sleeps on AC
     power (scheduled jobs and IB Gateway stop while asleep) or when
     automatic restart after a power failure is off. SKIP off macOS.
 13. Backup destination: WARN when ``OPTIONS_CALCULATOR_BACKUP_DIR`` is
     unset (nightly archives land on the state's own disk) or on the
     same disk as the state; FAIL when it is set but missing or
     unwritable (an unmounted external drive).

Exit codes (match the watchdog convention so this can drop into
existing CI):

  0 — all checks PASS or SKIP
  1 — at least one WARN, no FAIL
  2 — at least one FAIL

Output:

  Human-readable summary by default. ``--json`` emits a structured
  payload suitable for CI ingestion.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HOME = Path.home()
_LAUNCH_AGENTS = _HOME / "Library" / "LaunchAgents"
_OPTIONS_HOME = _HOME / ".options_calculator_pro"

# Canonical SQLite stores. Each is a service-owned data file; if it
# exists, it should pass quick_check. Missing → SKIP (fresh install).
_KNOWN_SQLITE_STORES: tuple[Path, ...] = (
    _OPTIONS_HOME / "recommendations" / "recommendation_ledger.sqlite",
    _OPTIONS_HOME / "outcomes" / "outcome_store.sqlite",
    _OPTIONS_HOME / "evidence" / "baseline_evidence.sqlite",
    _OPTIONS_HOME / "telemetry" / "provider_telemetry.sqlite",
)

# The launchd job list is DERIVED from the installer rather than duplicated
# here. A hand-maintained copy drifted to 5 while the installer grew to 8, and
# preflight kept reporting "All 5 LaunchAgents installed" with three jobs
# absent (Codex audit F4). The installer is the single source of truth; the
# plist templates then say which wrapper each job runs.
_INSTALLER_REL = Path("scripts") / "automation" / "install_launchd_jobs.sh"
_PLIST_RE = re.compile(r"(com\.optionscalculator\.[\w.-]+\.plist)")


def installer_plist_names(project_root: Path = _PROJECT_ROOT) -> tuple[str, ...]:
    """Every plist the installer will render+load, in installer order.

    Raises rather than returning a partial list: a preflight that cannot see the
    job list must not certify the install.
    """
    installer = project_root / _INSTALLER_REL
    text = installer.read_text(encoding="utf-8")
    match = re.search(r"for\s+plist\s+in\s*\\?\s*\n(.*?)\n\s*do\b", text, re.S)
    names = _PLIST_RE.findall(match.group(1)) if match else []
    if not names:
        raise RuntimeError(f"preflight could not parse the launchd job list from {installer}")
    return tuple(dict.fromkeys(names))  # de-dupe, keep order


def wrapper_names_for(
    plist_names: tuple[str, ...], project_root: Path = _PROJECT_ROOT
) -> tuple[str, ...]:
    """Basename of ``ProgramArguments[0]`` from each plist template."""
    auto = project_root / "scripts" / "automation"
    wrappers: list[str] = []
    for name in plist_names:
        path = auto / name
        if not path.exists():
            continue  # check_plist_templates reports the missing template itself
        with path.open("rb") as fh:
            data = plistlib.load(fh)
        args = data.get("ProgramArguments") or []
        if args:
            wrappers.append(Path(str(args[0])).name)
    return tuple(dict.fromkeys(wrappers))


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ── Individual checks ────────────────────────────────────────────────────


def check_env_file(project_root: Path = _PROJECT_ROOT) -> CheckResult:
    """The repo's ``.env`` is loaded by every entrypoint via
    ``python-dotenv``. Missing or unreadable means every command that
    reads `os.getenv(...)` will silently fall back to defaults — which
    includes the startup validators that gate share-auth."""
    env_path = project_root / ".env"
    if not env_path.exists():
        return CheckResult(
            name="env_file",
            status=Status.WARN,
            message=(
                f".env not found at {env_path}. The backend will boot with "
                "default env values; share-auth will be off, no provider "
                "tokens, etc. OK for a smoke test, not for a real deploy."
            ),
            details={"path": str(env_path)},
        )
    if not os.access(env_path, os.R_OK):
        return CheckResult(
            name="env_file",
            status=Status.FAIL,
            message=f".env exists at {env_path} but is not readable.",
            details={"path": str(env_path)},
        )
    return CheckResult(
        name="env_file",
        status=Status.PASS,
        message=f".env present and readable ({env_path.stat().st_size} bytes).",
        details={"path": str(env_path)},
    )


def check_python_venv(project_root: Path = _PROJECT_ROOT) -> CheckResult:
    """Every launchd wrapper hardcodes ``${PROJECT_ROOT}/.venv311/bin/python``.
    If that's not present, all five jobs will fail with
    ``No such file or directory`` on first fire."""
    py = project_root / ".venv311" / "bin" / "python"
    if not py.exists():
        return CheckResult(
            name="python_venv",
            status=Status.FAIL,
            message=(
                f".venv311 python not found at {py}. Every launchd wrapper "
                f"hardcodes this path; jobs will fail on first fire. Run "
                f"`uv venv .venv311 --python 3.11` from {project_root}."
            ),
            details={"path": str(py)},
        )
    if not os.access(py, os.X_OK):
        return CheckResult(
            name="python_venv",
            status=Status.FAIL,
            message=f".venv311 python exists at {py} but is not executable.",
            details={"path": str(py)},
        )
    return CheckResult(
        name="python_venv",
        status=Status.PASS,
        message=f".venv311 python present and executable.",
        details={"path": str(py)},
    )


def check_frontend_dist(project_root: Path = _PROJECT_ROOT) -> CheckResult:
    """Production deployments serve the React app same-origin from
    ``web/frontend/dist/``. Missing is a WARN, not a FAIL — dev mode
    (Vite on :5173) works fine without it. The runbook is explicit
    that you need ``npm run build`` before flipping share-auth on for
    a real deploy."""
    dist = project_root / "web" / "frontend" / "dist"
    index = dist / "index.html"
    if not index.exists():
        return CheckResult(
            name="frontend_dist",
            status=Status.WARN,
            message=(
                f"Frontend not built at {dist}. Dev mode works without "
                "this, but production same-origin deployment needs "
                "`cd web/frontend && npm run build` first."
            ),
            details={"path": str(dist)},
        )
    return CheckResult(
        name="frontend_dist",
        status=Status.PASS,
        message=f"Frontend build present ({dist}).",
        details={"path": str(dist)},
    )


def check_wrapper_scripts(
    project_root: Path = _PROJECT_ROOT,
    wrapper_names: tuple[str, ...] | None = None,
) -> CheckResult:
    """Every launchd wrapper the installer references must exist and be executable. The
    install script copies their absolute paths into the rendered
    plists, so a missing or non-executable wrapper means launchd will
    silently log permission errors."""
    auto = project_root / "scripts" / "automation"
    # `wrapper_names` is a test seam only. Production derives the list from the
    # installer so it cannot drift (Codex F4).
    if wrapper_names is None:
        wrapper_names = wrapper_names_for(installer_plist_names(project_root), project_root)
    missing: list[str] = []
    not_exec: list[str] = []
    for name in wrapper_names:
        p = auto / name
        if not p.exists():
            missing.append(name)
        elif not os.access(p, os.X_OK):
            not_exec.append(name)
    if missing:
        return CheckResult(
            name="wrapper_scripts",
            status=Status.FAIL,
            message=f"Missing wrapper scripts: {missing}",
            details={"missing": missing, "not_executable": not_exec},
        )
    if not_exec:
        return CheckResult(
            name="wrapper_scripts",
            status=Status.FAIL,
            message=f"Wrapper scripts present but not executable: {not_exec}",
            details={"missing": missing, "not_executable": not_exec},
        )
    return CheckResult(
        name="wrapper_scripts",
        status=Status.PASS,
        message=f"All {len(wrapper_names)} launchd wrappers present and executable.",
        details={"count": len(wrapper_names), "wrappers": list(wrapper_names)},
    )


def check_plist_templates(
    project_root: Path = _PROJECT_ROOT,
    plist_names: tuple[str, ...] | None = None,
) -> CheckResult:
    """Every plist the installer lists exists and contains the placeholders the install
    script substitutes (``__PROJECT_ROOT__`` and ``__HOME__``).
    Detects the case where someone hand-edited a plist and removed
    the placeholders — the install script would then leave the wrong
    paths in the rendered output, but everything LOOKS fine.
    """
    auto = project_root / "scripts" / "automation"
    if plist_names is None:  # test seam; production derives from the installer
        plist_names = installer_plist_names(project_root)
    issues: dict[str, str] = {}
    for name in plist_names:
        p = auto / name
        if not p.exists():
            issues[name] = "missing"
            continue
        text = p.read_text(encoding="utf-8")
        if "__PROJECT_ROOT__" not in text:
            issues[name] = "missing __PROJECT_ROOT__ placeholder"
        elif "__HOME__" not in text:
            issues[name] = "missing __HOME__ placeholder"
    if issues:
        return CheckResult(
            name="plist_templates",
            status=Status.FAIL,
            message=f"Plist template issues: {issues}",
            details={"issues": issues},
        )
    return CheckResult(
        name="plist_templates",
        status=Status.PASS,
        message=f"All {len(plist_names)} plists present with placeholders intact.",
        details={"count": len(plist_names), "plists": list(plist_names)},
    )


def check_launchagents_installed(
    launch_agents_dir: Path = _LAUNCH_AGENTS,
    project_root: Path = _PROJECT_ROOT,
    plist_names: tuple[str, ...] | None = None,
) -> CheckResult:
    """How many of the installer's plists are installed to ``~/Library/LaunchAgents``.

    Zero installed → SKIP (likely a fresh checkout or dev box). Partial
    → WARN (operator probably forgot to re-run installer after a
    PR-added job — that was the case for the log-rotation job after
    PR #61). Every job listed by the installer present → PASS.
    """
    if plist_names is None:  # test seam; production derives from the installer
        plist_names = installer_plist_names(project_root)
    if not launch_agents_dir.exists():
        return CheckResult(
            name="launchagents_installed",
            status=Status.SKIP,
            message=(
                f"{launch_agents_dir} does not exist (no LaunchAgents on "
                "this machine). Run scripts/automation/install_launchd_jobs.sh "
                "to install."
            ),
            details={"installed_count": 0, "total": len(plist_names)},
        )
    installed = [
        name for name in plist_names if (launch_agents_dir / name).exists()
    ]
    if not installed:
        return CheckResult(
            name="launchagents_installed",
            status=Status.SKIP,
            message=(
                f"No LaunchAgents installed. Run "
                "scripts/automation/install_launchd_jobs.sh when ready."
            ),
            details={"installed_count": 0, "total": len(plist_names)},
        )
    if len(installed) < len(plist_names):
        missing = sorted(set(plist_names) - set(installed))
        return CheckResult(
            name="launchagents_installed",
            status=Status.WARN,
            message=(
                f"Partial install: {len(installed)} of {len(plist_names)} "
                f"LaunchAgents present. Re-run install_launchd_jobs.sh to "
                f"pick up: {missing}"
            ),
            details={
                "installed_count": len(installed),
                "total": len(plist_names),
                "missing": missing,
            },
        )
    return CheckResult(
        name="launchagents_installed",
        status=Status.PASS,
        message=f"All {len(plist_names)} LaunchAgents installed.",
        details={"installed_count": len(installed), "total": len(plist_names)},
    )


def check_startup_validators(
    project_root: Path = _PROJECT_ROOT,
    python_bin: Path | None = None,
    timeout_seconds: float = 30.0,
) -> CheckResult:
    """Import ``web.api.app`` in a subprocess to exercise the PR #59
    startup validators. If the import succeeds, the current .env is
    valid for the current SHARE_AUTH posture.

    Subprocess isolation is deliberate: the import has side effects
    (loads .env, registers FastAPI routes, opens log handles). Running
    it in the parent process would pollute state and make the preflight
    non-reentrant.
    """
    py = python_bin or (project_root / ".venv311" / "bin" / "python")
    if not py.exists():
        return CheckResult(
            name="startup_validators",
            status=Status.SKIP,
            message=(
                f"Cannot run startup validators: {py} not found. "
                "(Earlier python_venv check should have already flagged this.)"
            ),
            details={"python_bin": str(py)},
        )
    # ``-c "import web.api.app"`` triggers _validate_auth_config() at
    # module load. Print a known marker so we can distinguish "import
    # succeeded silently" from "stdout was empty due to early exit".
    try:
        result = subprocess.run(
            [str(py), "-c", "import web.api.app; print('preflight_ok')"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="startup_validators",
            status=Status.FAIL,
            message=(
                f"web.api.app import timed out after {timeout_seconds}s "
                "(infinite loop at module load or pathologically slow .env)."
            ),
            details={"timeout_seconds": timeout_seconds},
        )
    if result.returncode == 0 and "preflight_ok" in result.stdout:
        return CheckResult(
            name="startup_validators",
            status=Status.PASS,
            message="web.api.app loads cleanly; PR #59 startup validators all pass.",
            details={},
        )
    # The validator's RuntimeError messages are clear; surface the
    # last error line verbatim so the operator can copy-paste the env
    # var to fix.
    last_err_line = ""
    if result.stderr:
        for line in reversed(result.stderr.strip().splitlines()):
            if line.strip():
                last_err_line = line.strip()
                break
    return CheckResult(
        name="startup_validators",
        status=Status.FAIL,
        message=(
            f"web.api.app failed to import (rc={result.returncode}). "
            f"Last stderr line: {last_err_line or '(empty)'}"
        ),
        details={
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        },
    )


def check_sqlite_stores(stores: tuple[Path, ...] = _KNOWN_SQLITE_STORES) -> CheckResult:
    """``PRAGMA quick_check`` on each known SQLite path. Fresh installs
    have no stores yet → SKIP. Existing stores must report "ok"; any
    other result → FAIL (corruption requires manual recovery).
    """
    checked: dict[str, str] = {}
    bad: dict[str, str] = {}
    for path in stores:
        if not path.exists():
            checked[str(path)] = "absent"
            continue
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                row = conn.execute("PRAGMA quick_check;").fetchone()
            verdict = row[0] if row else "(no row)"
            checked[str(path)] = verdict
            if verdict != "ok":
                bad[str(path)] = verdict
        except sqlite3.Error as exc:
            bad[str(path)] = f"sqlite3.Error: {exc}"
            checked[str(path)] = "error"
    if bad:
        return CheckResult(
            name="sqlite_stores",
            status=Status.FAIL,
            message=f"SQLite quick_check failed for: {sorted(bad)}",
            details={"checked": checked, "failures": bad},
        )
    if all(v == "absent" for v in checked.values()):
        return CheckResult(
            name="sqlite_stores",
            status=Status.SKIP,
            message="No SQLite stores yet — fresh install.",
            details={"checked": checked},
        )
    return CheckResult(
        name="sqlite_stores",
        status=Status.PASS,
        message=(
            f"{sum(1 for v in checked.values() if v == 'ok')} SQLite store(s) "
            f"pass quick_check; {sum(1 for v in checked.values() if v == 'absent')} "
            "absent."
        ),
        details={"checked": checked},
    )


def check_log_rotation_dryrun(
    project_root: Path = _PROJECT_ROOT,
    python_bin: Path | None = None,
    timeout_seconds: float = 30.0,
) -> CheckResult:
    """Dry-run the log rotator. It walks ``~/.options_calculator_pro/logs/``
    and reports what it would rotate — no mutations. Exit 0 means
    the rotator's discovery + size logic is intact.
    """
    py = python_bin or (project_root / ".venv311" / "bin" / "python")
    if not py.exists():
        return CheckResult(
            name="log_rotation_dryrun",
            status=Status.SKIP,
            message=f"Cannot run rotator dry-run: {py} not found.",
            details={},
        )
    try:
        result = subprocess.run(
            [str(py), str(project_root / "scripts" / "rotate_launchd_logs.py"), "--dry-run"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="log_rotation_dryrun",
            status=Status.FAIL,
            message=f"rotate_launchd_logs.py --dry-run timed out after {timeout_seconds}s.",
            details={"timeout_seconds": timeout_seconds},
        )
    if result.returncode != 0:
        return CheckResult(
            name="log_rotation_dryrun",
            status=Status.FAIL,
            message=f"rotate_launchd_logs.py --dry-run exited with {result.returncode}.",
            details={
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            },
        )
    # Count what it found so the operator sees the rotator is wired.
    line_count = sum(1 for line in result.stdout.splitlines() if line.strip().startswith("["))
    return CheckResult(
        name="log_rotation_dryrun",
        status=Status.PASS,
        message=f"Rotator dry-run clean; {line_count} log file(s) scanned.",
        details={"files_scanned": line_count},
    )


def check_backend_health(
    base_url: str = "http://127.0.0.1:8000",
    timeout_seconds: float = 2.0,
) -> CheckResult:
    """Optional probe of ``/api/health``. SKIP if no backend is running
    (connection refused / DNS failure) — preflight is a deploy-time
    check, not a runtime monitor; we never start uvicorn here. PASS
    if 200 OK with a recognizable body; FAIL on any other response."""
    url = f"{base_url.rstrip('/')}/api/health"
    try:
        with urlopen(Request(url), timeout=timeout_seconds) as resp:  # noqa: S310 — local URL
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status == 200:
                # Light schema check: backend returns {"status": "ok", ...}.
                try:
                    payload = json.loads(body)
                    ok = payload.get("status") == "ok"
                except json.JSONDecodeError:
                    ok = False
                if ok:
                    return CheckResult(
                        name="backend_health",
                        status=Status.PASS,
                        message=f"Backend responding 200 OK at {url}.",
                        details={"url": url},
                    )
                return CheckResult(
                    name="backend_health",
                    status=Status.FAIL,
                    message=(
                        f"Backend at {url} returned 200 but body wasn't the "
                        "expected health shape."
                    ),
                    details={"url": url, "body_head": body[:200]},
                )
            return CheckResult(
                name="backend_health",
                status=Status.FAIL,
                message=f"Backend at {url} returned HTTP {resp.status}.",
                details={"url": url, "status_code": resp.status},
            )
    except HTTPError as exc:
        # Codex P1 (PR #63 follow-up): HTTPError is a subclass of
        # URLError. Without this explicit branch, a backend that's
        # UP but returning 4xx/5xx (auth misconfig → 401, app crash
        # → 500) would have been swallowed by the URLError block
        # below as "no backend reachable / SKIP" — green-lighting a
        # broken deployment. A reachable backend that refuses must
        # FAIL, not SKIP.
        return CheckResult(
            name="backend_health",
            status=Status.FAIL,
            message=(
                f"Backend at {url} is up but returned HTTP {exc.code} "
                f"({exc.reason})."
            ),
            details={"url": url, "status_code": exc.code, "reason": str(exc.reason)},
        )
    except URLError as exc:
        # Connection refused / DNS fail / TLS handshake error → SKIP.
        # Preflight isn't responsible for *starting* the backend;
        # it's checking that IF you have one running, it's healthy.
        return CheckResult(
            name="backend_health",
            status=Status.SKIP,
            message=(
                f"No backend reachable at {url} ({type(exc).__name__}: "
                f"{exc.reason if hasattr(exc, 'reason') else exc}). Skipped — "
                "preflight does not start uvicorn."
            ),
            details={"url": url},
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="backend_health",
            status=Status.FAIL,
            message=f"Unexpected error probing {url}: {type(exc).__name__}: {exc}",
            details={"url": url},
        )


# ── Operations checks (IB Gateway, power, backups) ───────────────────────


_IB_PROVIDER_ALIASES = frozenset({"ib_gateway", "ib", "ibkr", "ibgateway", "interactive_brokers", "tws"})
_ALL_INTERFACES_PREFIXES = ("*:", "0.0.0.0:", "[::]:")


def env_file_values(project_root: Path = _PROJECT_ROOT) -> dict[str, str]:
    """``KEY=VALUE`` pairs from the repo ``.env``, parsed and never sourced.

    Comments and blank lines are skipped; surrounding quotes are stripped; the
    last assignment of a key wins, as python-dotenv would read it.
    """
    env_path = project_root / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _tcp_reachable(host: str, port: int, timeout_seconds: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _listening_addresses(port: int) -> list[str] | None:
    """``addr:port`` of every LISTEN socket on ``port`` per ``lsof``; None when
    that cannot be determined (no ``lsof``, or it failed)."""
    if shutil.which("lsof") is None:
        return None
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode not in (0, 1):  # 1 = nothing listening
        return None
    addresses: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 9:
            addresses.append(parts[8])
    return addresses


def check_ib_gateway(
    project_root: Path = _PROJECT_ROOT,
    env: dict[str, str] | None = None,
    reachable: Callable[[str, int], bool] = _tcp_reachable,
    listeners: Callable[[int], list[str] | None] = _listening_addresses,
) -> CheckResult:
    """The ``ib_gateway`` provider's data source is a GUI app someone must
    have started and logged into. A gateway that is down makes every
    analysis fall back to yfinance (no greeks) without failing anything, so
    preflight says so. Its API port should also be reachable from this Mac
    only: ``TrustedIPs`` in the gateway is an application-layer filter, and
    the socket itself can be bound to localhost in its API settings."""
    env = env_file_values(project_root) if env is None else env
    provider = (env.get("OPTIONS_CALCULATOR_OPTIONS_PROVIDER") or "yfinance").strip().lower()
    if provider not in _IB_PROVIDER_ALIASES:
        return CheckResult(
            name="ib_gateway",
            status=Status.SKIP,
            message=f"Options provider is {provider!r}; IB Gateway is not in use.",
            details={"provider": provider},
        )
    host = (env.get("IB_GATEWAY_HOST") or "127.0.0.1").strip()
    try:
        port = int((env.get("IB_GATEWAY_PORT") or "4002").strip())
    except ValueError:
        return CheckResult(
            name="ib_gateway",
            status=Status.FAIL,
            message=f"IB_GATEWAY_PORT={env.get('IB_GATEWAY_PORT')!r} is not a port number.",
            details={"host": host, "port": env.get("IB_GATEWAY_PORT")},
        )
    up = bool(reachable(host, port))
    bound = listeners(port)
    exposed = [a for a in (bound or []) if a.startswith(_ALL_INTERFACES_PREFIXES)]
    details = {"host": host, "port": port, "reachable": up,
               "listening": bound if bound is not None else "unknown", "all_interfaces": exposed}
    if not up:
        return CheckResult(
            name="ib_gateway",
            status=Status.WARN,
            message=(
                f"IB Gateway is not accepting API connections at {host}:{port}. Start it and "
                "log in (the com.optionscalculator.ib-gateway job starts it at login but cannot "
                "sign in); until then analyses fall back to yfinance, which has no greeks."
            ),
            details=details,
        )
    if exposed:
        return CheckResult(
            name="ib_gateway",
            status=Status.WARN,
            message=(
                f"IB Gateway API port {port} is listening on every interface ({', '.join(exposed)}), "
                "so other machines on the network can reach it. In IB Gateway: Configure → Settings "
                "→ API → Settings, tick 'Allow connections from localhost only'."
            ),
            details=details,
        )
    return CheckResult(
        name="ib_gateway",
        status=Status.PASS,
        message=(
            f"IB Gateway accepting API connections at {host}:{port}"
            + (" (socket bound to localhost only)." if bound else " (socket binding not determinable).")
        ),
        details=details,
    )


def _pmset_custom() -> str | None:
    if shutil.which("pmset") is None:
        return None
    try:
        result = subprocess.run(["pmset", "-g", "custom"], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def parse_pmset_ac_settings(output: str) -> dict[str, str]:
    """``setting -> value`` from ``pmset -g custom``, from the ``AC Power:``
    section when the output has sections (a laptop), else from the whole
    output (a desktop reports one unlabelled set)."""
    settings: dict[str, str] = {}
    section = None
    has_sections = any(line.strip().endswith("Power:") for line in output.splitlines())
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.endswith("Power:"):
            section = stripped[:-len("Power:")].strip().lower()
            continue
        if has_sections and section != "ac":
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            # "Sleep On Power Button 1" has a multi-word key; keep the last token as value.
            settings[" ".join(parts[:-1]).lower()] = parts[-1]
    return settings


def check_power_management(pmset_output: str | None = None, *, on_macos: bool | None = None) -> CheckResult:
    """Scheduled jobs and IB Gateway stop while the Mac sleeps, and a Mac
    that stays down after a power failure stops them until someone notices.
    Both are system settings only an administrator can change, so this
    check reports and names the command rather than changing anything."""
    if on_macos is None:
        on_macos = sys.platform == "darwin"
    if not on_macos:
        return CheckResult(name="power_management", status=Status.SKIP,
                           message="Not macOS; pmset not applicable.", details={})
    output = _pmset_custom() if pmset_output is None else pmset_output
    if output is None:
        return CheckResult(name="power_management", status=Status.SKIP,
                           message="pmset unavailable; power settings not checked.", details={})
    settings = parse_pmset_ac_settings(output)
    issues: list[str] = []
    sleep = settings.get("sleep")
    if sleep is not None and sleep != "0":
        issues.append(
            f"system sleeps after {sleep} min on AC power, stopping scheduled jobs and IB Gateway "
            "(fix: `sudo pmset -c sleep 0`)"
        )
    autorestart = settings.get("autorestart")
    if autorestart is not None and autorestart != "1":
        issues.append("automatic restart after a power failure is off (fix: `sudo pmset -a autorestart 1`)")
    details = {"sleep": sleep, "autorestart": autorestart, "displaysleep": settings.get("displaysleep")}
    if issues:
        return CheckResult(name="power_management", status=Status.WARN,
                           message="; ".join(issues) + ".", details=details)
    return CheckResult(
        name="power_management",
        status=Status.PASS,
        message="Mac stays awake on AC power and restarts itself after a power failure.",
        details=details,
    )


def check_backup_destination(
    project_root: Path = _PROJECT_ROOT,
    env: dict[str, str] | None = None,
    state_dir: Path = _OPTIONS_HOME,
    device_of: Callable[[Path], int] = lambda path: path.stat().st_dev,
) -> CheckResult:
    """The nightly backup job writes where ``OPTIONS_CALCULATOR_BACKUP_DIR``
    says. Unset, it writes next to the state it is protecting, which
    survives a typo but not the disk. An external destination that is
    missing is most likely an unmounted drive: the job would then create a
    plain folder on the system disk at that path and look healthy."""
    env = env_file_values(project_root) if env is None else env
    raw = (env.get("OPTIONS_CALCULATOR_BACKUP_DIR") or "").strip()
    if not raw:
        return CheckResult(
            name="backup_destination",
            status=Status.WARN,
            message=(
                "OPTIONS_CALCULATOR_BACKUP_DIR is unset: nightly backups land in "
                f"{state_dir / 'backups'}, on the same disk as the state. Set it to an "
                "external drive or sync folder for disaster recovery."
            ),
            details={"configured": None},
        )
    dest = Path(raw).expanduser()
    if not dest.is_dir():
        return CheckResult(
            name="backup_destination",
            status=Status.FAIL,
            message=(
                f"Backup destination {dest} does not exist (an external drive not mounted?). "
                "The nightly job would create a folder at that path on the system disk instead."
            ),
            details={"configured": str(dest), "exists": False},
        )
    if not os.access(dest, os.W_OK):
        return CheckResult(
            name="backup_destination",
            status=Status.FAIL,
            message=f"Backup destination {dest} is not writable.",
            details={"configured": str(dest), "exists": True, "writable": False},
        )
    same_disk = None
    if state_dir.exists():
        try:
            same_disk = device_of(dest) == device_of(state_dir)
        except OSError:
            same_disk = None
    details = {"configured": str(dest), "exists": True, "writable": True, "same_disk_as_state": same_disk}
    if same_disk:
        return CheckResult(
            name="backup_destination",
            status=Status.WARN,
            message=(
                f"Backup destination {dest} is on the same disk as {state_dir}; a disk failure "
                "takes the backups with the state."
            ),
            details=details,
        )
    return CheckResult(
        name="backup_destination",
        status=Status.PASS,
        message=f"Backup destination {dest} is writable"
                + (" and on a different disk from the state." if same_disk is False else "."),
        details=details,
    )


# ── Aggregation + output ─────────────────────────────────────────────────


_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_env_file,
    check_python_venv,
    check_frontend_dist,
    check_wrapper_scripts,
    check_plist_templates,
    check_launchagents_installed,
    check_startup_validators,
    check_sqlite_stores,
    check_log_rotation_dryrun,
    check_backend_health,
    check_ib_gateway,
    check_power_management,
    check_backup_destination,
)


def run_all_checks() -> list[CheckResult]:
    """Run every check in declaration order. Each check is independent;
    a FAIL in one does NOT skip subsequent checks (we want the
    operator to see all the issues at once, not play whack-a-mole)."""
    return [fn() for fn in _CHECKS]


def aggregate_exit_code(results: list[CheckResult]) -> int:
    """Watchdog-convention exit codes:
      0 — all PASS or SKIP
      1 — at least one WARN, no FAIL
      2 — at least one FAIL
    """
    if any(r.status == Status.FAIL for r in results):
        return 2
    if any(r.status == Status.WARN for r in results):
        return 1
    return 0


_STATUS_GLYPHS = {
    Status.PASS: "✓",
    Status.WARN: "~",
    Status.FAIL: "✗",
    Status.SKIP: "·",
}


def format_human(results: list[CheckResult]) -> str:
    """Human-readable summary. Cap message length so terminals don't
    wrap awkwardly, but keep the full message in --json output."""
    out: list[str] = ["Pre-deployment smoke check", ""]
    width = max(len(r.name) for r in results)
    for r in results:
        glyph = _STATUS_GLYPHS[r.status]
        msg = r.message
        if len(msg) > 100:
            msg = msg[:97] + "..."
        out.append(f"  {glyph} {r.status.value:<4} {r.name:<{width}}  {msg}")
    counts = {s: sum(1 for r in results if r.status == s) for s in Status}
    out.append("")
    out.append(
        f"  Summary: {counts[Status.PASS]} PASS, {counts[Status.WARN]} WARN, "
        f"{counts[Status.FAIL]} FAIL, {counts[Status.SKIP]} SKIP"
    )
    return "\n".join(out)


def format_json(results: list[CheckResult]) -> str:
    return json.dumps(
        {
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ],
            "summary": {
                s.value: sum(1 for r in results if r.status == s) for s in Status
            },
        },
        indent=2,
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-deployment / pre-start smoke check for options-calculator-pro.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON (for CI ingestion). Default is human-readable.",
    )
    args = parser.parse_args(argv)
    results = run_all_checks()
    if args.json:
        print(format_json(results))
    else:
        print(format_human(results))
    return aggregate_exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
