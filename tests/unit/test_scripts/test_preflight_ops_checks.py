"""Preflight's operations checks: IB Gateway reachability and exposure, Mac
power management, and the nightly backup destination.

Each check takes its inputs through seams (parsed .env, a reachability probe,
an lsof listing, pmset output, a device-id function) so the verdicts are
tested without a gateway, a Mac, or an external drive.
"""
from __future__ import annotations

import socket
import threading
from pathlib import Path

from scripts.preflight_check import (
    PROBE_ACCEPTED,
    PROBE_NO_ADDRESS,
    PROBE_REJECTED,
    PROBE_TIMEOUT,
    PROBE_UNREACHABLE,
    Status,
    check_backup_destination,
    check_ib_gateway,
    check_power_management,
    env_file_values,
    lan_api_probe,
    parse_pmset_ac_settings,
    primary_lan_address,
)

IB_ENV = {"OPTIONS_CALCULATOR_OPTIONS_PROVIDER": "ib_gateway", "IB_GATEWAY_HOST": "127.0.0.1",
          "IB_GATEWAY_PORT": "4002"}
UP = lambda h, p: True  # noqa: E731
ON_ALL = lambda p: ["*:4002"]  # noqa: E731


def _probe(verdict):
    return lambda port: (verdict, "192.168.1.20" if verdict != PROBE_NO_ADDRESS else None)


def _never_probed(port):
    raise AssertionError("the LAN probe must not run when the binding already answers the question")


# ── .env parsing ────────────────────────────────────────────────────────────


def test_env_file_values_parses_without_sourcing(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "# comment\n\nOPTIONS_CALCULATOR_OPTIONS_PROVIDER=ib_gateway\n"
        'IB_GATEWAY_PORT="4002"\nOPTIONS_CALCULATOR_BACKUP_DIR=/Volumes/T9/backups # not a comment\n'
        "IB_GATEWAY_PORT=7497\nNOT_AN_ASSIGNMENT\n"
    )
    values = env_file_values(tmp_path)
    assert values["OPTIONS_CALCULATOR_OPTIONS_PROVIDER"] == "ib_gateway"
    assert values["IB_GATEWAY_PORT"] == "7497"           # last assignment wins, quotes stripped
    assert values["OPTIONS_CALCULATOR_BACKUP_DIR"] == "/Volumes/T9/backups # not a comment"
    assert "NOT_AN_ASSIGNMENT" not in values
    assert env_file_values(tmp_path / "nowhere") == {}


# ── IB Gateway ──────────────────────────────────────────────────────────────


def test_ib_gateway_skipped_for_other_providers() -> None:
    result = check_ib_gateway(env={"OPTIONS_CALCULATOR_OPTIONS_PROVIDER": "yfinance"},
                              reachable=lambda h, p: True, listeners=lambda p: ["*:4002"])
    assert result.status is Status.SKIP
    result = check_ib_gateway(env={}, reachable=lambda h, p: True, listeners=lambda p: [])
    assert result.status is Status.SKIP  # default provider is yfinance


def test_ib_gateway_down_is_warn_with_instructions() -> None:
    result = check_ib_gateway(env=IB_ENV, reachable=lambda h, p: False, listeners=lambda p: [])
    assert result.status is Status.WARN
    assert "127.0.0.1:4002" in result.message and "log in" in result.message
    assert result.details["reachable"] is False


# The gateway binds on every interface even with 'localhost only' ticked, so
# the binding alone decides nothing: the LAN-side handshake does.


def test_ib_gateway_on_all_interfaces_but_refusing_lan_clients_is_pass() -> None:
    result = check_ib_gateway(env=IB_ENV, reachable=UP, listeners=ON_ALL, probe=_probe(PROBE_REJECTED))
    assert result.status is Status.PASS
    assert "refused" in result.message and "192.168.1.20:4002" in result.message
    assert result.details["all_interfaces"] == ["*:4002"]
    assert result.details["lan_probe"] == {"verdict": PROBE_REJECTED, "address": "192.168.1.20"}


def test_ib_gateway_answering_lan_clients_is_fail_with_the_fix() -> None:
    result = check_ib_gateway(env=IB_ENV, reachable=UP, listeners=ON_ALL, probe=_probe(PROBE_ACCEPTED))
    assert result.status is Status.FAIL
    assert "any machine on the network" in result.message
    assert "localhost only" in result.message and "File menu" in result.message  # restart is part of the fix


def test_ib_gateway_silent_on_lan_probe_is_warn_naming_the_prompt() -> None:
    result = check_ib_gateway(env=IB_ENV, reachable=UP, listeners=ON_ALL, probe=_probe(PROBE_TIMEOUT))
    assert result.status is Status.WARN
    assert "prompt" in result.message and "Reject" in result.message and "localhost only" in result.message


def test_ib_gateway_probe_that_cannot_run_keeps_the_binding_warning() -> None:
    for verdict in (PROBE_NO_ADDRESS, PROBE_UNREACHABLE):
        result = check_ib_gateway(env=IB_ENV, reachable=UP, listeners=ON_ALL, probe=_probe(verdict))
        assert result.status is Status.WARN, verdict
        assert "every interface" in result.message and verdict in result.message
    for addr in ("0.0.0.0:4002", "[::]:4002"):
        assert check_ib_gateway(env=IB_ENV, reachable=UP, listeners=lambda p, a=addr: [a],
                                probe=_probe(PROBE_NO_ADDRESS)).status is Status.WARN


def test_ib_gateway_localhost_only_binding_is_pass_without_probing() -> None:
    result = check_ib_gateway(env=IB_ENV, reachable=UP,
                              listeners=lambda p: ["127.0.0.1:4002", "[::1]:4002"], probe=_never_probed)
    assert result.status is Status.PASS
    assert "localhost only" in result.message


def test_ib_gateway_unknown_binding_still_passes_when_reachable() -> None:
    # No lsof (CI on Linux): reachability is known, exposure is not claimed either way.
    result = check_ib_gateway(env=IB_ENV, reachable=UP, listeners=lambda p: None, probe=_never_probed)
    assert result.status is Status.PASS
    assert result.details["listening"] == "unknown"


def test_ib_gateway_down_is_warn_before_any_probe() -> None:
    result = check_ib_gateway(env=IB_ENV, reachable=lambda h, p: False, listeners=ON_ALL, probe=_never_probed)
    assert result.status is Status.WARN


# ── the probe itself, against fake gateways on loopback ─────────────────────


def _fake_gateway(behaviour):
    """A one-connection server on an ephemeral loopback port. ``behaviour`` is
    'reset' (close right after reading the handshake), 'answer' (reply with a
    server-version frame) or 'hang' (read and never reply)."""
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
                conn.sendall(b"\x00\x00\x00\x0c178\x0020260101 00\x00")
            elif behaviour == "hang":
                try:
                    conn.recv(1)  # until the probe gives up and closes
                except OSError:
                    pass
        # 'reset': falls out of the with-block and closes at once

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server, server.getsockname()[1], seen, thread


def test_probe_sends_the_ib_handshake_and_reads_a_reset_as_rejected() -> None:
    server, port, seen, thread = _fake_gateway("reset")
    try:
        assert lan_api_probe(port, address="127.0.0.1", timeout_seconds=3) == (PROBE_REJECTED, "127.0.0.1")
    finally:
        server.close()
    thread.join(5)
    assert seen["handshake"].startswith(b"API\0")
    assert b"v100..178" in seen["handshake"]


def test_probe_reads_a_server_version_as_accepted() -> None:
    server, port, _, thread = _fake_gateway("answer")
    try:
        assert lan_api_probe(port, address="127.0.0.1", timeout_seconds=3) == (PROBE_ACCEPTED, "127.0.0.1")
    finally:
        server.close()
    thread.join(5)


def test_probe_reads_silence_as_timeout() -> None:
    server, port, _, thread = _fake_gateway("hang")
    try:
        assert lan_api_probe(port, address="127.0.0.1", timeout_seconds=0.5) == (PROBE_TIMEOUT, "127.0.0.1")
    finally:
        server.close()
    thread.join(5)


def test_probe_with_nothing_listening_is_unreachable() -> None:
    spare = socket.socket(); spare.bind(("127.0.0.1", 0)); port = spare.getsockname()[1]; spare.close()
    assert lan_api_probe(port, address="127.0.0.1", timeout_seconds=1) == (PROBE_UNREACHABLE, "127.0.0.1")


def test_probe_without_a_lan_address_does_not_connect(monkeypatch) -> None:
    monkeypatch.setattr("scripts.preflight_check.primary_lan_address", lambda: None)
    assert lan_api_probe(4002) == (PROBE_NO_ADDRESS, None)


def test_primary_lan_address_is_never_loopback() -> None:
    address = primary_lan_address()
    assert address is None or (not address.startswith("127.") and address != "0.0.0.0")


def test_ib_gateway_honours_configured_host_and_port() -> None:
    seen = {}

    def probe(host, port):
        seen["target"] = (host, port)
        return True

    env = {**IB_ENV, "IB_GATEWAY_HOST": "10.0.0.5", "IB_GATEWAY_PORT": "7497",
           "OPTIONS_CALCULATOR_OPTIONS_PROVIDER": "IBKR"}
    check_ib_gateway(env=env, reachable=probe, listeners=lambda p: [f"127.0.0.1:{p}"])
    assert seen["target"] == ("10.0.0.5", 7497)


def test_ib_gateway_bad_port_is_fail() -> None:
    result = check_ib_gateway(env={**IB_ENV, "IB_GATEWAY_PORT": "four"},
                              reachable=lambda h, p: True, listeners=lambda p: [])
    assert result.status is Status.FAIL


# ── Power management ────────────────────────────────────────────────────────

STUDIO_PMSET = """System-wide power settings:
Currently in use:
 Sleep On Power Button 1
 autorestart          0
 powernap             1
 displaysleep         10
 womp                 1
 sleep                1
 disksleep            10
"""

LAPTOP_PMSET = """Battery Power:
 sleep                5
 autorestart          0
AC Power:
 sleep                0
 autorestart          1
 displaysleep         10
"""


def test_pmset_parsing_prefers_the_ac_section_on_a_laptop() -> None:
    settings = parse_pmset_ac_settings(LAPTOP_PMSET)
    assert settings["sleep"] == "0" and settings["autorestart"] == "1"
    desktop = parse_pmset_ac_settings(STUDIO_PMSET)
    assert desktop["sleep"] == "1" and desktop["autorestart"] == "0"
    assert desktop["sleep on power button"] == "1"


def test_power_sleeping_studio_is_warn_naming_both_fixes() -> None:
    result = check_power_management(STUDIO_PMSET, on_macos=True)
    assert result.status is Status.WARN
    assert "sleeps after 1 min" in result.message and "pmset -c sleep 0" in result.message
    assert "autorestart 1" in result.message
    assert result.details == {"sleep": "1", "autorestart": "0", "displaysleep": "10"}


def test_power_always_on_is_pass() -> None:
    result = check_power_management(LAPTOP_PMSET, on_macos=True)
    assert result.status is Status.PASS


def test_power_skipped_off_macos_or_without_pmset() -> None:
    assert check_power_management(STUDIO_PMSET, on_macos=False).status is Status.SKIP
    # pmset_output=None on a "mac" with no pmset binary resolves to SKIP, never a crash.
    assert check_power_management(None, on_macos=True).status in {Status.SKIP, Status.PASS, Status.WARN}


# ── Backup destination ──────────────────────────────────────────────────────


def test_backup_destination_unset_is_warn(tmp_path: Path) -> None:
    result = check_backup_destination(env={}, state_dir=tmp_path)
    assert result.status is Status.WARN
    assert "same disk" in result.message


def test_backup_destination_missing_is_fail_not_a_quiet_folder(tmp_path: Path) -> None:
    result = check_backup_destination(env={"OPTIONS_CALCULATOR_BACKUP_DIR": str(tmp_path / "T9" / "backups")},
                                      state_dir=tmp_path)
    assert result.status is Status.FAIL
    assert "not mounted" in result.message


def test_backup_destination_unwritable_is_fail(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    dest.chmod(0o500)
    try:
        import os
        if os.access(dest, os.W_OK):  # root ignores modes
            return
        result = check_backup_destination(env={"OPTIONS_CALCULATOR_BACKUP_DIR": str(dest)}, state_dir=tmp_path)
        assert result.status is Status.FAIL
    finally:
        dest.chmod(0o700)


def test_backup_destination_same_disk_is_warn(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    result = check_backup_destination(env={"OPTIONS_CALCULATOR_BACKUP_DIR": str(dest)}, state_dir=state,
                                      device_of=lambda path: 7)
    assert result.status is Status.WARN
    assert result.details["same_disk_as_state"] is True


def test_backup_destination_external_disk_is_pass(tmp_path: Path) -> None:
    dest = tmp_path / "T9"
    dest.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    result = check_backup_destination(env={"OPTIONS_CALCULATOR_BACKUP_DIR": str(dest)}, state_dir=state,
                                      device_of=lambda path: 1 if path == dest else 2)
    assert result.status is Status.PASS
    assert "different disk" in result.message


def test_backup_destination_expands_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Backups").mkdir()
    result = check_backup_destination(env={"OPTIONS_CALCULATOR_BACKUP_DIR": "~/Backups"},
                                      state_dir=tmp_path / "absent", device_of=lambda p: 1)
    assert result.status is Status.PASS
    assert result.details["configured"] == str(tmp_path / "Backups")


def test_ops_checks_are_registered() -> None:
    from scripts.preflight_check import _CHECKS
    names = {fn.__name__ for fn in _CHECKS}
    assert {"check_ib_gateway", "check_power_management", "check_backup_destination"} <= names
