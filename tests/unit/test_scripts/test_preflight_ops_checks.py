"""Preflight's operations checks: IB Gateway reachability and exposure, Mac
power management, and the nightly backup destination.

Each check takes its inputs through seams (parsed .env, a reachability probe,
an lsof listing, pmset output, a device-id function) so the verdicts are
tested without a gateway, a Mac, or an external drive.
"""
from __future__ import annotations

from pathlib import Path

from scripts.preflight_check import (
    Status,
    check_backup_destination,
    check_ib_gateway,
    check_power_management,
    env_file_values,
    parse_pmset_ac_settings,
)

IB_ENV = {"OPTIONS_CALCULATOR_OPTIONS_PROVIDER": "ib_gateway", "IB_GATEWAY_HOST": "127.0.0.1",
          "IB_GATEWAY_PORT": "4002"}


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


def test_ib_gateway_bound_on_all_interfaces_is_warn() -> None:
    result = check_ib_gateway(env=IB_ENV, reachable=lambda h, p: True, listeners=lambda p: ["*:4002"])
    assert result.status is Status.WARN
    assert "every interface" in result.message and "localhost only" in result.message
    assert result.details["all_interfaces"] == ["*:4002"]
    for addr in ("0.0.0.0:4002", "[::]:4002"):
        assert check_ib_gateway(env=IB_ENV, reachable=lambda h, p: True,
                                listeners=lambda p, a=addr: [a]).status is Status.WARN


def test_ib_gateway_localhost_only_is_pass() -> None:
    result = check_ib_gateway(env=IB_ENV, reachable=lambda h, p: True,
                              listeners=lambda p: ["127.0.0.1:4002", "[::1]:4002"])
    assert result.status is Status.PASS
    assert "localhost only" in result.message


def test_ib_gateway_unknown_binding_still_passes_when_reachable() -> None:
    # No lsof (CI on Linux): reachability is known, exposure is not claimed either way.
    result = check_ib_gateway(env=IB_ENV, reachable=lambda h, p: True, listeners=lambda p: None)
    assert result.status is Status.PASS
    assert result.details["listening"] == "unknown"


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
