"""Tests for scripts/train_from_t9.py orchestrator.

These tests exercise the orchestrator's argument resolution + preflight
checks WITHOUT invoking any of the underlying stage scripts (which would
fan out to API calls and a multi-hour backtest). The downstream
integration is verified manually by the smoke run documented in the
PR-X description.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_from_t9.py"


def _load_module():
    """Import scripts/train_from_t9.py without it being on the package path."""
    spec = importlib.util.spec_from_file_location("train_from_t9", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


train_from_t9 = _load_module()


class TestScalePresets:
    def test_smoke_is_smallest(self):
        smoke = train_from_t9.SCALE_PRESETS["smoke"]
        small = train_from_t9.SCALE_PRESETS["small"]
        full = train_from_t9.SCALE_PRESETS["full"]
        assert smoke["limit_symbols"] < small["limit_symbols"]
        assert smoke["years"] < small["years"]
        assert full["limit_symbols"] is None  # unbounded
        assert full["years"] >= small["years"]

    def test_every_preset_has_required_keys(self):
        for name, preset in train_from_t9.SCALE_PRESETS.items():
            for key in ("limit_symbols", "years", "runtime", "label"):
                assert key in preset, f"preset {name} missing {key}"


class TestSymbolResolution:
    def test_resolve_symbols_returns_list(self):
        # We can't mock T9 here easily, so just confirm the function returns
        # SOMETHING list-like (it falls back to an empty list if T9 isn't
        # mounted, which is fine — preflight catches that).
        symbols = train_from_t9._resolve_symbols(limit=5)
        assert isinstance(symbols, list)

    def test_resolve_symbols_respects_limit(self, monkeypatch):
        # Mock the institutional universe import and T9 listing.
        fake_universe = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        fake_t9 = {"AAA", "BBB", "CCC", "DDD", "EEE"}  # FFF missing on T9
        import types
        fake_module = types.ModuleType("services.institutional_ml_db")
        fake_module.INSTITUTIONAL_UNIVERSE = fake_universe
        monkeypatch.setitem(sys.modules, "services.institutional_ml_db", fake_module)
        monkeypatch.setattr(train_from_t9, "_t9_chain_symbols", lambda: fake_t9)
        result = train_from_t9._resolve_symbols(limit=3)
        assert result == ["AAA", "BBB", "CCC"]

    def test_resolve_symbols_intersects_universe_with_t9(self, monkeypatch):
        # Symbols on T9 that aren't in the universe must NOT be returned —
        # avoids ingesting data for tickers the selector doesn't analyse.
        fake_universe = ["AAA", "BBB"]
        fake_t9 = {"AAA", "BBB", "ZZZ"}  # ZZZ extra on T9
        import types
        fake_module = types.ModuleType("services.institutional_ml_db")
        fake_module.INSTITUTIONAL_UNIVERSE = fake_universe
        monkeypatch.setitem(sys.modules, "services.institutional_ml_db", fake_module)
        monkeypatch.setattr(train_from_t9, "_t9_chain_symbols", lambda: fake_t9)
        result = train_from_t9._resolve_symbols(limit=None)
        assert "ZZZ" not in result
        assert result == ["AAA", "BBB"]


class TestPreflightChecks:
    def _args(self, **overrides):
        import argparse
        defaults = dict(
            scale="smoke",
            target="tmp",
            dry_run=True,
            skip_earnings_fetch=False,
            skip_replay=False,
            max_debit=600,
            db_path=str(REPO_ROOT / "tmp" / "fake_ml.db"),
            earnings_db=str(REPO_ROOT / "tmp" / "fake_earnings.sqlite"),
            start_date="2024-01-01",
            end_date="2025-01-01",
            report_dir=str(REPO_ROOT / "tmp" / "fake_reports"),
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smoke_plus_production_is_rejected(self):
        """The preflight gate must refuse to write smoke-scale outcomes
        into the live calibration/prior stores. PR-X scope: smoke is
        always a tmp/inspect run; production requires small or full."""
        symbols = ["AAA", "BBB"]  # non-empty so the empty-set check passes
        args = self._args(scale="smoke", target="production")
        failures = train_from_t9._preflight_checks(args, symbols)
        joined = " | ".join(failures)
        assert "smoke" in joined.lower()
        assert "production" in joined.lower()

    def test_missing_t9_is_rejected(self, monkeypatch, tmp_path):
        # Point T9 root to a non-existent path.
        monkeypatch.setattr(train_from_t9, "T9_CHAINS_ROOT", tmp_path / "no_such_t9")
        args = self._args(scale="full", target="tmp")
        failures = train_from_t9._preflight_checks(args, ["AAA"])
        assert any("T9" in f for f in failures)

    def test_empty_symbol_set_is_rejected(self):
        args = self._args(scale="full", target="tmp")
        failures = train_from_t9._preflight_checks(args, [])
        assert any("empty" in f.lower() for f in failures)

    def test_clean_inputs_pass(self, monkeypatch, tmp_path):
        # Synthesise a "T9 root" with one fake symbol directory.
        fake_t9 = tmp_path / "chains_eod"
        (fake_t9 / "underlying_symbol=AAA").mkdir(parents=True)
        monkeypatch.setattr(train_from_t9, "T9_CHAINS_ROOT", fake_t9)
        args = self._args(
            scale="small",
            target="tmp",
            db_path=str(tmp_path / "ml.db"),
            earnings_db=str(tmp_path / "earnings.sqlite"),
        )
        failures = train_from_t9._preflight_checks(args, ["AAA"])
        assert failures == []


class TestDryRunExitsBeforeAnyStage(object):
    def test_dry_run_returns_zero_with_no_subprocess_calls(self, monkeypatch, tmp_path):
        # Synthesise a usable T9 so preflight passes.
        fake_t9 = tmp_path / "chains_eod"
        (fake_t9 / "underlying_symbol=AAPL").mkdir(parents=True)
        monkeypatch.setattr(train_from_t9, "T9_CHAINS_ROOT", fake_t9)
        # If the orchestrator ever shells out under --dry-run that's a bug.
        called = []
        monkeypatch.setattr(
            train_from_t9, "_run_subprocess",
            lambda cmd, log_path: called.append(cmd) or (0, 0.0),
        )
        rc = train_from_t9.main([
            "--scale", "smoke",
            "--target", "tmp",
            "--dry-run",
        ])
        assert rc == 0
        assert called == [], (
            "--dry-run must not invoke any stage subprocesses, "
            f"but got {len(called)} calls"
        )
