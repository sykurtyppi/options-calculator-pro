"""The IB Gateway login check: when it speaks up, when it stays quiet, and that
it can never stop the job that called it. No gateway and no Messages.app: the
probe talks to throwaway loopback sockets and the send is a recorder."""
from __future__ import annotations

import json
import socket
import subprocess
import threading
from datetime import date
from pathlib import Path

import pytest

from services import external_io_gate
from services import ib_gateway_login_alert as alert
from services.automation_watchdog import HandoffNotStarted, IMessageConfig
from services.external_io_gate import Category

REPO_ROOT = Path(__file__).resolve().parents[3]
TRADING_DAY = date(2026, 11, 20)      # a Friday
SATURDAY = date(2026, 11, 21)
THANKSGIVING = date(2026, 11, 26)
IB_ENV = {"OPTIONS_CALCULATOR_OPTIONS_PROVIDER": "ib_gateway"}
CONFIG = IMessageConfig(to_address="+15555550123")


@pytest.fixture
def gateway_io():
    external_io_gate.enable(Category.IB_GATEWAY)
    try:
        yield
    finally:
        external_io_gate.disable(Category.IB_GATEWAY)


def _fake_gateway(behaviour):
    """One-connection loopback server: 'answer' replies with a server version,
    'hang' reads and never replies, 'close' hangs up after the handshake."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5)
    seen = {}

    def serve():
        try:
            conn, _ = server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            try:
                seen["handshake"] = conn.recv(64)
            except OSError:
                return
            if behaviour == "answer":
                conn.sendall(b"\x00\x00\x00\x0c178\x0020261120 16\x00")
            elif behaviour == "hang":
                try:
                    conn.recv(1)
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server, server.getsockname()[1], seen, thread


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Recorder:
    def __init__(self, error: Exception | None = None) -> None:
        self.messages = []
        self.error = error

    def __call__(self, message, *, config):
        self.messages.append(message)
        if self.error is not None:
            raise self.error
        return {"sent": True, "provider": "imessage", "to": "***0123"}


# ── the probe ─────────────────────────────────────────────────────────────────

def test_signed_in_gateway_answers_the_handshake(gateway_io):
    server, port, seen, thread = _fake_gateway("answer")
    try:
        assert alert.probe_gateway("127.0.0.1", port) == alert.STATE_READY
    finally:
        thread.join(2)
        server.close()
    # The first frame of the API handshake and nothing after it: no startApi,
    # so no client id is taken and no connected client is disturbed.
    assert seen["handshake"] == b"API\x00\x00\x00\x00\tv100..178"


def test_login_screen_has_no_api_socket(gateway_io):
    assert alert.probe_gateway("127.0.0.1", _closed_port()) == alert.STATE_NOT_LISTENING


def test_gateway_that_accepts_and_says_nothing_is_silent(gateway_io):
    server, port, _, thread = _fake_gateway("hang")
    try:
        assert alert.probe_gateway("127.0.0.1", port, timeout_seconds=0.3) == alert.STATE_SILENT
    finally:
        server.close()
        thread.join(2)


def test_gateway_that_hangs_up_is_closed(gateway_io):
    server, port, _, thread = _fake_gateway("close")
    try:
        assert alert.probe_gateway("127.0.0.1", port) == alert.STATE_CLOSED
    finally:
        thread.join(2)
        server.close()


def test_probe_respects_the_external_io_gate():
    with pytest.raises(external_io_gate.ExternalIOBlocked):
        alert.probe_gateway("127.0.0.1", _closed_port())


# ── when the check applies ────────────────────────────────────────────────────

def _never(host, port):
    raise AssertionError("the gateway must not be probed")


def test_other_providers_are_not_checked():
    status = alert.build_login_status(context="cycle", environ={"OPTIONS_CALCULATOR_OPTIONS_PROVIDER": "yfinance"},
                                      today=TRADING_DAY, probe=_never)
    assert status["ok"] and not status["checked"] and status["reason"] == "provider_is_not_ib_gateway"
    assert alert.build_login_status(context="cycle", environ={}, today=TRADING_DAY, probe=_never)["ok"]


@pytest.mark.parametrize("day", [SATURDAY, THANKSGIVING])
def test_closed_market_days_are_not_checked(day):
    """The gateway signs itself out every weekend; a text about that is noise."""
    status = alert.build_login_status(context="premarket", environ=IB_ENV, today=day, probe=_never)
    assert status["ok"] and not status["checked"] and status["reason"] == "not_a_trading_day"


def test_ready_gateway_is_ok_and_uses_configured_address():
    seen = []
    env = {**IB_ENV, "IB_GATEWAY_HOST": "10.0.0.5", "IB_GATEWAY_PORT": "7497"}
    status = alert.build_login_status(context="cycle", environ=env, today=TRADING_DAY,
                                      probe=lambda host, port: seen.append((host, port)) or alert.STATE_READY)
    assert seen == [("10.0.0.5", 7497)]
    assert status["ok"] and status["checked"] and status["state"] == alert.STATE_READY


def test_default_address_is_the_paper_gateway_on_loopback():
    seen = []
    alert.build_login_status(context="cycle", environ=IB_ENV, today=TRADING_DAY,
                             probe=lambda host, port: seen.append((host, port)) or alert.STATE_READY)
    assert seen == [("127.0.0.1", 4002)]


@pytest.mark.parametrize("state", [alert.STATE_NOT_LISTENING, alert.STATE_SILENT, alert.STATE_CLOSED,
                                   alert.STATE_UNREACHABLE])
def test_anything_but_an_answer_is_not_ok(state):
    status = alert.build_login_status(context="premarket", environ=IB_ENV, today=TRADING_DAY,
                                      probe=lambda host, port: state)
    assert status["checked"] and not status["ok"] and status["state"] == state


def test_unusable_port_setting_is_reported_not_raised():
    status = alert.build_login_status(context="cycle", environ={**IB_ENV, "IB_GATEWAY_PORT": "paper"},
                                      today=TRADING_DAY, probe=_never)
    assert not status["ok"] and status["state"] == "bad_port"
    assert "IB_GATEWAY_PORT='paper'" in alert.alert_message(status)


def test_unknown_context_is_a_programming_error():
    with pytest.raises(ValueError):
        alert.build_login_status(context="lunch", environ=IB_ENV, today=TRADING_DAY, probe=_never)


# ── the message ───────────────────────────────────────────────────────────────

def _down(context, state=alert.STATE_NOT_LISTENING, day=TRADING_DAY):
    return alert.build_login_status(context=context, environ=IB_ENV, today=day, probe=lambda host, port: state)


def test_premarket_message_says_there_is_still_time():
    text = alert.alert_message(_down("premarket"))
    assert text.startswith("[Options Calculator] IB Gateway is not ready")
    assert "127.0.0.1:4002" in text and "login screen" in text
    assert "Sign in on the Mac before today's evidence cycle" in text


def test_cycle_message_says_the_cycle_is_running_without_ib():
    text = alert.alert_message(_down("cycle", alert.STATE_SILENT))
    assert "did not answer" in text and "dialog" in text
    assert "starting now and will use the fallback data provider" in text


# ── sending, once ─────────────────────────────────────────────────────────────

def test_ready_gateway_sends_nothing(tmp_path):
    send = _Recorder()
    ready = alert.build_login_status(context="cycle", environ=IB_ENV, today=TRADING_DAY,
                                     probe=lambda host, port: alert.STATE_READY)
    result = alert.maybe_send_login_alert(ready, imessage_config=CONFIG, state_path=tmp_path / "s.json", send=send)
    assert result == {"attempted": False, "reason": "gateway_ready"}
    assert send.messages == [] and not (tmp_path / "s.json").exists()


def test_each_job_alerts_once_a_day(tmp_path):
    send, state = _Recorder(), tmp_path / "state" / "alerts.json"
    first = alert.maybe_send_login_alert(_down("premarket"), imessage_config=CONFIG, state_path=state, send=send)
    again = alert.maybe_send_login_alert(_down("premarket"), imessage_config=CONFIG, state_path=state, send=send)
    cycle = alert.maybe_send_login_alert(_down("cycle"), imessage_config=CONFIG, state_path=state, send=send)
    assert first["sent"] is True and first["state_persisted"] is True
    assert again == {"attempted": False, "reason": "duplicate_alert_suppressed"}
    assert cycle["sent"] is True                       # a different job, a different message
    assert len(send.messages) == 2 and send.messages[0] != send.messages[1]

    monday = alert.maybe_send_login_alert(_down("premarket", day=date(2026, 11, 23)), imessage_config=CONFIG,
                                          state_path=state, send=send)
    assert monday["sent"] is True and len(send.messages) == 3
    # Friday's premarket alert is still remembered after Monday's was recorded.
    assert alert.maybe_send_login_alert(_down("premarket"), imessage_config=CONFIG, state_path=state,
                                        send=send)["reason"] == "duplicate_alert_suppressed"


def test_force_sends_again(tmp_path):
    send, state = _Recorder(), tmp_path / "alerts.json"
    alert.maybe_send_login_alert(_down("cycle"), imessage_config=CONFIG, state_path=state, send=send)
    alert.maybe_send_login_alert(_down("cycle"), imessage_config=CONFIG, state_path=state, send=send, force=True)
    assert len(send.messages) == 2


def test_dry_run_composes_but_neither_sends_nor_records(tmp_path):
    send, state = _Recorder(), tmp_path / "alerts.json"
    result = alert.maybe_send_login_alert(_down("premarket"), imessage_config=CONFIG, state_path=state,
                                          send=send, dry_run=True)
    assert result["dry_run"] is True and result["sent"] is False
    assert result["to"] == "***0123" and "+1555" not in json.dumps(result)
    assert "IB Gateway is not ready" in result["message"]
    assert send.messages == [] and not state.exists()


def test_no_recipient_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.delenv("WATCHDOG_IMESSAGE_TO", raising=False)
    monkeypatch.delenv("WATCHDOG_SMS_TO", raising=False)
    result = alert.maybe_send_login_alert(_down("cycle"), state_path=tmp_path / "alerts.json", send=_Recorder())
    assert result == {"attempted": False, "reason": "imessage_not_configured"}


def test_send_that_never_started_is_retried_by_the_next_job(tmp_path):
    state = tmp_path / "alerts.json"
    failed = alert.maybe_send_login_alert(_down("premarket"), imessage_config=CONFIG, state_path=state,
                                          send=_Recorder(HandoffNotStarted(2, "osascript not started")))
    assert failed["sent"] is False and not state.exists()
    send = _Recorder()
    assert alert.maybe_send_login_alert(_down("premarket"), imessage_config=CONFIG, state_path=state,
                                        send=send)["sent"] is True


def test_send_of_unknown_outcome_is_recorded_so_it_is_not_sent_twice(tmp_path):
    state = tmp_path / "alerts.json"
    unknown = alert.maybe_send_login_alert(_down("premarket"), imessage_config=CONFIG, state_path=state,
                                           send=_Recorder(subprocess.TimeoutExpired("osascript", 30)))
    assert unknown["sent"] is None and unknown["delivery_unknown"] is True
    send = _Recorder()
    assert alert.maybe_send_login_alert(_down("premarket"), imessage_config=CONFIG, state_path=state,
                                        send=send)["reason"] == "duplicate_alert_suppressed"
    assert send.messages == []


def test_unreadable_state_is_moved_aside_and_nothing_is_sent(tmp_path):
    state = tmp_path / "alerts.json"
    state.write_text("{not json")
    send = _Recorder()
    result = alert.maybe_send_login_alert(_down("cycle"), imessage_config=CONFIG, state_path=state, send=send)
    assert result["reason"] == "alert_state_unreadable" and send.messages == []
    assert not state.exists() and Path(result["moved_to"]).exists()


def test_recorded_state_never_holds_the_recipient(tmp_path):
    state = tmp_path / "alerts.json"
    alert.maybe_send_login_alert(_down("cycle"), imessage_config=CONFIG, state_path=state, send=_Recorder())
    assert "5555550123" not in state.read_text()


# ── the command and its two callers ───────────────────────────────────────────

def _load_cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ib_gateway_login_alert_cli",
                                                  REPO_ROOT / "scripts" / "ib_gateway_login_alert.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exits_zero_whether_or_not_the_gateway_answers(monkeypatch, capsys, tmp_path):
    cli = _load_cli()
    monkeypatch.setattr(cli, "build_login_status", lambda **kw: _down(kw["context"]))
    monkeypatch.setenv("WATCHDOG_IMESSAGE_TO", "+15555550123")
    assert cli.main(["--context", "premarket", "--dry-run", "--state-path", str(tmp_path / "s.json")]) == 0
    payload = json.loads(capsys.readouterr().out)["ib_gateway_login_check"]
    assert payload["status"]["state"] == alert.STATE_NOT_LISTENING and payload["alert"]["dry_run"] is True


def test_cli_exits_zero_even_when_the_check_itself_breaks(monkeypatch, capsys):
    cli = _load_cli()

    def _boom(**kwargs):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(cli, "build_login_status", _boom)
    assert cli.main(["--context", "cycle"]) == 0
    assert "RuntimeError: calendar unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("wrapper, context", [
    ("run_daily_evidence_cycle.sh", "cycle"),
    ("run_premarket_screener_alert.sh", "premarket"),
])
def test_wrappers_run_the_check_first_and_cannot_be_failed_by_it(wrapper, context):
    text = (REPO_ROOT / "scripts" / "automation" / wrapper).read_text()
    call = f'"${{PYTHON_BIN}}" scripts/ib_gateway_login_alert.py --context {context} \\\n    || echo'
    assert text.count(call) == 1
    # Before the job's own work (the timeout-bounded run), so the text arrives
    # while it still matters...
    job = '"${PYTHON_BIN}" - "${PYTHON_BIN}"'
    assert text.count(job) == 1 and text.index(call) < text.index(job)
    # ...and inside the logged block, so its one line of output lands in the job log.
    assert text.index(" start =====") < text.index(call)
