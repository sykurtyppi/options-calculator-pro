"""
Verify the pre-market screener alert: selection logic, message honesty,
idempotency, and the launchd wiring.

We do NOT load the launchd job here (CI runs on Linux; launchctl is
macOS-only), and we never let the screener touch the network — the ranked
screener is stubbed. Everything else is exercised on any platform:
  - plist parses + correct Label / wrapper path / weekday schedule / RunAtLoad
  - install_launchd_jobs.sh and uninstall_launchd_jobs.sh include the plist
  - the install script's sed substitution produces a valid rendered plist
  - the wrapper is executable, set -euo pipefail, with lock/timeout/exit-code
    plumbing matching the established state-backup pattern
  - qualifying-row selection, message contents, and per-day idempotency
"""
from __future__ import annotations

import json
import plistlib
import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[3]
AUTOMATION = REPO / "scripts" / "automation"
PLIST = AUTOMATION / "com.optionscalculator.premarket-screener-alert.plist"
WRAPPER = AUTOMATION / "run_premarket_screener_alert.sh"
INSTALL = AUTOMATION / "install_launchd_jobs.sh"
UNINSTALL = AUTOMATION / "uninstall_launchd_jobs.sh"
README = AUTOMATION / "README.md"

LABEL = "com.optionscalculator.premarket-screener-alert"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.premarket_screener_alert import (  # noqa: E402
    format_alert_message,
    main,
    select_qualifying_rows,
)


def _row(symbol, score, dte=7, **extra):
    row = {
        "symbol": symbol, "ranking_score": score, "days_to_earnings": dte,
        "release_timing": "AMC", "iv_rv_ratio": 1.10, "iv30": 0.35,
        "avg_spread_pct": 3.0, "status": "ranked",
    }
    row.update(extra)
    return row


# ── Selection ─────────────────────────────────────────────────────────────


def test_selection_filters_by_score_window_and_errors():
    rows = [
        _row("AAA", 0.80),                      # qualifies
        _row("BBB", 0.40),                      # below threshold
        _row("CCC", 0.90, dte=30),              # outside entry window
        _row("DDD", 0.85, error_note="boom"),   # errored row
        _row("EEE", None),                      # never scored
        {"symbol": "FFF", "status": "upcoming"},  # no score at all
    ]
    picked = select_qualifying_rows(rows, min_score=0.65, top_n=5, dte_min=3, dte_max=10)
    assert [r["symbol"] for r in picked] == ["AAA"]


def test_selection_fails_closed_on_nonfinite_scores_and_missing_evidence():
    """Codex F2: NaN/inf scores and evidence-less rows used to qualify."""
    rows = [
        _row("NAN", float("nan")),
        _row("INF", float("inf")),
        _row("NOIVRV", 0.90, iv_rv_ratio=None),          # no IV/RV -> neutral 0.25 fallback
        _row("ZEROIVRV", 0.90, iv_rv_ratio=0.0),
        _row("NOATM", 0.90, iv30=None, atm_iv=None),     # no chain evidence at all
        _row("PLACEHOLDERIV", 0.90, iv30=0.0039),       # pre-open yfinance placeholder
        _row("NANSPREAD", 0.90, avg_spread_pct=float("nan")),
        _row("GOOD", 0.90),
        _row("NOSPREAD", 0.88, avg_spread_pct=None),    # absent spread is tolerated
    ]
    picked = select_qualifying_rows(rows, min_score=0.65, top_n=9, dte_min=3, dte_max=10)
    assert [r["symbol"] for r in picked] == ["GOOD", "NOSPREAD"]


def test_selection_rejects_nonfinite_threshold():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        select_qualifying_rows([_row("A", 0.9)], min_score=float("nan"), top_n=5, dte_min=3, dte_max=10)


def test_cli_rejects_nan_min_score():
    """``--min-score nan`` must be a usage error, not "alert on everything"."""
    import pytest as _pytest

    from scripts.premarket_screener_alert import parse_args

    with _pytest.raises(SystemExit) as exc:
        parse_args(["--min-score", "nan"])
    assert exc.value.code == 2
    with _pytest.raises(SystemExit):
        parse_args(["--min-score", "inf"])
    assert parse_args(["--min-score", "0.7"]).min_score == 0.7


def test_selection_sorts_by_score_and_caps_at_top_n():
    rows = [_row("AAA", 0.70), _row("BBB", 0.90), _row("CCC", 0.80), _row("DDD", 0.75)]
    picked = select_qualifying_rows(rows, min_score=0.65, top_n=2, dte_min=3, dte_max=10)
    assert [r["symbol"] for r in picked] == ["BBB", "CCC"]


def test_selection_accepts_api_shaped_dte_alias():
    """The service emits days_to_earnings; the API row calls it dte."""
    row = {"symbol": "AAA", "ranking_score": 0.80, "dte": 7, "release_timing": "AMC",
           "iv_rv_ratio": 1.1, "atm_iv": 0.35}  # API-shaped evidence keys too
    assert select_qualifying_rows([row], min_score=0.65, top_n=5, dte_min=3, dte_max=10)


# ── Message ───────────────────────────────────────────────────────────────


def test_message_lists_setups_and_refuses_to_claim_a_win_rate():
    msg = format_alert_message(
        [_row("AAA", 0.80), _row("BBB", 0.70, iv_regime_conditioned=True)],
        as_of=date(2026, 8, 19), min_score=0.65,
    )
    assert "2026-08-19" in msg
    assert "AAA" in msg and "BBB" in msg
    assert "[regime]" in msg, "regime-conditioned cheapness must be flagged"
    # The load-bearing honesty claims.
    assert "not a win probability" in msg
    assert "not financial advice" in msg.lower()
    # send_imessage truncates at 1500 chars; stay well inside it.
    assert len(msg) < 1000


# ── End-to-end behaviour (screener stubbed, nothing sent) ─────────────────


def _table(rows):
    return {"rows": rows, "universe_size": 39, "in_entry_window": len(rows)}


def test_no_qualifying_setups_is_silent_and_successful(tmp_path):
    with patch("scripts.premarket_screener_alert.build_ranked_screener",
               return_value=_table([_row("AAA", 0.10)])), \
         patch("scripts.premarket_screener_alert.send_imessage") as send:
        code = main(["--state-path", str(tmp_path / "s.json")])
    assert code == 0
    send.assert_not_called()


def test_dry_run_never_sends_and_never_writes_state(tmp_path, capsys):
    state = tmp_path / "s.json"
    with patch("scripts.premarket_screener_alert.build_ranked_screener",
               return_value=_table([_row("AAA", 0.90)])), \
         patch("scripts.premarket_screener_alert.send_imessage") as send:
        code = main(["--dry-run", "--state-path", str(state)])
    assert code == 0
    send.assert_not_called()
    assert not state.exists()
    assert "AAA" in capsys.readouterr().out


def test_identical_digest_is_not_resent_same_day(tmp_path):
    state = tmp_path / "s.json"
    table = _table([_row("AAA", 0.90)])
    cfg = object()
    with patch("scripts.premarket_screener_alert.build_ranked_screener", return_value=table), \
         patch("scripts.premarket_screener_alert.IMessageConfig.from_env", return_value=cfg), \
         patch("scripts.premarket_screener_alert.send_imessage",
               return_value={"to": "***1234"}) as send:
        assert main(["--state-path", str(state)]) == 0
        assert send.call_count == 1
        # Same day, same qualifying set -> suppressed.
        assert main(["--state-path", str(state)]) == 0
        assert send.call_count == 1
        # --force overrides.
        assert main(["--state-path", str(state), "--force"]) == 0
        assert send.call_count == 2

    saved = json.loads(state.read_text())
    assert saved["symbols"] == ["AAA"]
    assert saved["last_digest"].startswith(date.today().isoformat())


def test_missing_recipient_fails_loudly(tmp_path):
    with patch("scripts.premarket_screener_alert.build_ranked_screener",
               return_value=_table([_row("AAA", 0.90)])), \
         patch("scripts.premarket_screener_alert.IMessageConfig.from_env", return_value=None), \
         patch("scripts.premarket_screener_alert.send_imessage") as send:
        code = main(["--state-path", str(tmp_path / "s.json")])
    assert code == 1, "a misconfigured recipient must not look like success"
    send.assert_not_called()


def test_screener_failure_exits_nonzero(tmp_path):
    with patch("scripts.premarket_screener_alert.build_ranked_screener",
               side_effect=RuntimeError("provider down")), \
         patch("scripts.premarket_screener_alert.send_imessage") as send:
        code = main(["--state-path", str(tmp_path / "s.json")])
    assert code == 1
    send.assert_not_called()


# ── launchd wiring ────────────────────────────────────────────────────────


def test_plist_parses_and_has_expected_structure():
    data = plistlib.loads(PLIST.read_bytes())
    assert data["Label"] == LABEL
    assert data["ProgramArguments"] == [
        "__PROJECT_ROOT__/scripts/automation/run_premarket_screener_alert.sh"
    ]
    assert data["RunAtLoad"] is False, "must fire on schedule only, never on load"
    schedule = data["StartCalendarInterval"]
    assert [e["Weekday"] for e in schedule] == [1, 2, 3, 4, 5], "weekdays only"
    # Must run >=30 min AFTER the 09:30 ET open in BOTH DST regimes (Codex F6):
    # 15:00 UTC = 11:00 EDT / 10:00 EST. 14:30 was the opening bell in EST, and
    # 13:00 was pre-open — yfinance returns placeholder IV before the bell.
    assert all(e["Hour"] == 15 and e["Minute"] == 0 for e in schedule)


def test_install_and_uninstall_lists_include_the_job():
    assert f"{LABEL}.plist" in INSTALL.read_text()
    assert f"{LABEL}.plist" in UNINSTALL.read_text()


def test_rendered_plist_is_valid_after_substitution(tmp_path):
    rendered = (
        PLIST.read_text()
        .replace("__PROJECT_ROOT__", "/opt/ocp")
        .replace("__HOME__", "/Users/tester")
    )
    out = tmp_path / "rendered.plist"
    out.write_text(rendered)
    data = plistlib.loads(out.read_bytes())
    assert data["ProgramArguments"][0].startswith("/opt/ocp/")
    assert data["StandardOutPath"].startswith("/Users/tester/")
    assert "__PROJECT_ROOT__" not in rendered and "__HOME__" not in rendered


def test_wrapper_is_executable_and_hardened():
    import os

    assert os.access(WRAPPER, os.X_OK), "wrapper must be executable"
    body = WRAPPER.read_text()
    assert body.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in body
    # Same lock / timeout / exit-code plumbing as the other jobs.
    assert "LOCK_DIR=" in body and "LOCK_MAX_AGE_SECONDS" in body
    assert "trap cleanup EXIT" in body
    assert "TIMEOUT_SECONDS" in body
    assert 'exit "${EXIT_CODE}"' in body


def test_wrapper_shell_syntax_is_valid():
    assert subprocess.run(["bash", "-n", str(WRAPPER)]).returncode == 0


def test_readme_documents_the_job():
    text = README.read_text()
    assert LABEL in text
    assert "premarket_screener_alert.py" in text
    assert "WATCHDOG_IMESSAGE_TO" in text
