"""Tell the operator when IB Gateway is not logged in before a job needs it.

IB Gateway is a GUI app that a person signs into. It logs itself out on IB's
weekly reset, when a nightly restart's session token is refused, or when the
session is closed mid-day. Nothing fails when that happens: every analysis
quietly falls back to another data provider, and the only trace is a different
``option_source`` in that day's rows. This check makes it loud instead.

It runs at the start of the jobs that precede and perform the daily evidence
cycle (the 15:00 screener alert and the 16:45 cycle itself). On a trading day,
with ``ib_gateway`` as the options provider, it sends the gateway the first
frame of an API handshake on loopback. A gateway that is signed in answers
with its server version; one sitting at its login screen has no API socket
at all. Anything but an answer sends one iMessage per job per day.

The probe opens no API session: it never sends the ``startApi`` message, so
it takes no client id and cannot disturb a connected client. The check
never enters credentials and cannot sign in; it only reports.
"""
from __future__ import annotations

import socket
import struct
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from services import external_io_gate
from services.automation_watchdog import (
    DEFAULT_HOME,
    AlertStateUnreadable,
    HandoffNotStarted,
    IMessageConfig,
    _load_alert_state,
    _redacted_address,
    _utc_now,
    _write_alert_state,
    send_imessage,
)
from services.market_calendar import is_trading_day
from services.market_data_provider import _IB_GATEWAY_ALIASES, get_options_provider_name

DEFAULT_STATE_PATH = DEFAULT_HOME / "state" / "ib_gateway_login_alerts.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002
DEFAULT_PROBE_TIMEOUT_SECONDS = 5.0

#: The jobs that run the check. ``premarket`` leaves time to sign in before the
#: cycle; ``cycle`` says the day's cycle is starting without IB.
CONTEXT_PREMARKET = "premarket"
CONTEXT_CYCLE = "cycle"
CONTEXTS = (CONTEXT_PREMARKET, CONTEXT_CYCLE)

STATE_READY = "ready"                  # answered the handshake: signed in
STATE_NOT_LISTENING = "not_listening"  # nothing on the port: not running, or at the login screen
STATE_SILENT = "silent"                # connected, no answer: a dialog is probably waiting
STATE_CLOSED = "closed"                # connected, then hung up without answering
STATE_UNREACHABLE = "unreachable"      # the address itself could not be reached

_API_HANDSHAKE = b"API\0" + struct.pack(">I", len(b"v100..178")) + b"v100..178"
_MARKET_TZ = ZoneInfo("America/New_York")

_STATE_PHRASES = {
    STATE_NOT_LISTENING: "nothing is accepting API connections at {where}; it is not running or is "
                         "waiting at its login screen",
    STATE_SILENT: "it took a connection at {where} but did not answer; a dialog is probably waiting "
                  "in its window",
    STATE_CLOSED: "it closed an API connection at {where} without answering",
    STATE_UNREACHABLE: "{where} could not be reached",
}


def probe_gateway(host: str, port: int, *, timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS) -> str:
    """Ask the gateway for its API server version; return one of the STATE_* values."""
    external_io_gate.assert_allowed(external_io_gate.Category.IB_GATEWAY)
    try:
        conn = socket.create_connection((host, port), timeout=timeout_seconds)
    except ConnectionRefusedError:
        return STATE_NOT_LISTENING
    except OSError:
        return STATE_UNREACHABLE
    with conn:
        conn.settimeout(timeout_seconds)
        try:
            conn.sendall(_API_HANDSHAKE)
            answer = conn.recv(64)
        except socket.timeout:
            return STATE_SILENT
        except OSError:
            return STATE_CLOSED
    return STATE_READY if answer else STATE_CLOSED


def _market_today() -> date:
    return datetime.now(_MARKET_TZ).date()


def build_login_status(
    *,
    context: str,
    environ: Mapping[str, str],
    today: date | None = None,
    probe: Callable[[str, int], str] = probe_gateway,
) -> dict[str, Any]:
    """Decide whether the gateway matters right now and, if so, whether it answers.

    ``ok`` is True when there is nothing to tell anyone: another provider is
    configured, the market is closed today (the gateway logs itself out every
    weekend, and a text about that would only teach the reader to ignore
    texts), or the gateway answered.
    """
    if context not in CONTEXTS:
        raise ValueError(f"context must be one of {CONTEXTS}, got {context!r}")
    day = today or _market_today()
    status: dict[str, Any] = {"context": context, "date": day.isoformat(), "checked": False, "ok": True}

    provider = get_options_provider_name(environ)
    status["provider"] = provider
    if provider.lower() not in _IB_GATEWAY_ALIASES:
        return {**status, "reason": "provider_is_not_ib_gateway"}
    if not is_trading_day(day):
        return {**status, "reason": "not_a_trading_day"}

    host = str(environ.get("IB_GATEWAY_HOST") or "").strip() or DEFAULT_HOST
    raw_port = str(environ.get("IB_GATEWAY_PORT") or "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_PORT
    except ValueError:
        return {**status, "checked": True, "ok": False, "host": host, "port": raw_port,
                "state": "bad_port", "reason": f"IB_GATEWAY_PORT={raw_port!r} is not a port number"}

    state = probe(host, port)
    return {**status, "checked": True, "ok": state == STATE_READY, "host": host, "port": port,
            "state": state, "reason": "gateway_ready" if state == STATE_READY else "gateway_not_ready"}


def alert_message(status: Mapping[str, Any]) -> str:
    where = f"{status.get('host')}:{status.get('port')}"
    if status.get("state") == "bad_port":
        what = str(status.get("reason"))
    else:
        what = _STATE_PHRASES.get(str(status.get("state")), "it is not answering at {where}").format(where=where)
    if status.get("context") == CONTEXT_CYCLE:
        consequence = ("Today's evidence cycle is starting now and will use the fallback data "
                       "provider instead of IB.")
    else:
        consequence = ("Sign in on the Mac before today's evidence cycle, or it will use the "
                       "fallback data provider instead of IB.")
    return f"[Options Calculator] IB Gateway is not ready: {what}. {consequence}"


def _signature(status: Mapping[str, Any]) -> str:
    return f"{status.get('date')}::{status.get('context')}"


def maybe_send_login_alert(
    status: Mapping[str, Any],
    *,
    imessage_config: IMessageConfig | None = None,
    state_path: Path = DEFAULT_STATE_PATH,
    force: bool = False,
    dry_run: bool = False,
    send: Callable[..., dict[str, Any]] = send_imessage,
) -> dict[str, Any]:
    """Send at most one alert per job per day. Mirrors the watchdog's rules:
    an unreadable dedup file is moved aside and nothing is sent this time; a
    hand-off that started but did not finish cleanly is recorded as sent, so
    it is never sent twice."""
    if bool(status.get("ok")):
        return {"attempted": False, "reason": str(status.get("reason") or "gateway_ready")}

    signature = _signature(status)
    try:
        state = _load_alert_state(state_path)
    except AlertStateUnreadable as exc:
        if not force:
            moved = None
            try:
                moved = state_path.with_name(
                    f"{state_path.name}.unreadable-{_utc_now().strftime('%Y%m%dT%H%M%S')}")
                state_path.replace(moved)
            except OSError:
                moved = None
            return {"attempted": False, "reason": "alert_state_unreadable", "error": str(exc),
                    "moved_to": str(moved) if moved else None}
        state = {}
    sent_signatures = [str(item) for item in state.get("sent_signatures", []) if item]
    if not force and signature in sent_signatures:
        return {"attempted": False, "reason": "duplicate_alert_suppressed"}

    cfg = imessage_config or IMessageConfig.from_env()
    if cfg is None:
        return {"attempted": False, "reason": "imessage_not_configured"}

    message = alert_message(status)
    if dry_run:
        return {"attempted": True, "sent": False, "dry_run": True, "provider": "imessage",
                "to": _redacted_address(cfg.to_address), "message": message}
    try:
        result = send(message, config=cfg)
    except HandoffNotStarted as exc:
        # Nothing went out, so nothing is recorded: the next job tries again.
        return {"attempted": True, "sent": False, "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        result = {"sent": None, "delivery_unknown": True, "provider": "imessage",
                  "error": f"{type(exc).__name__}: {exc}"}

    try:
        _write_alert_state(state_path, {
            # Two contexts a day; keep a fortnight so the file never grows.
            "sent_signatures": (sent_signatures + [signature])[-28:],
            "last_alerted_at": _utc_now().isoformat(),
            "last_status": dict(status),
            "last_alert_result": result,
        })
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, **result, "state_persisted": False,
                "state_error": f"{type(exc).__name__}: {exc}"}
    return {"attempted": True, **result, "state_persisted": True}
