"""
Shared pytest fixtures for Options Calculator Pro tests.
"""
import datetime
import os
import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ── Singleton reset: web.api.app._mda_client ──────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_web_app_mda_singleton():
    """Reset the _mda_client module-level singleton in web.api.app after each test.

    _get_mda_client() caches a MarketDataClient in a module-level global.
    The gate check (assert_allowed(MARKETDATA)) only fires when the singleton
    is None. If a test with enable_external_io(MARKETDATA) were to initialize
    the singleton, every subsequent test that does not patch _get_mda_client
    directly would receive the live client without a gate check.

    This fixture is a no-op for tests that never import web.api.app (the
    sys.modules.get guard avoids triggering load_dotenv on import).
    """
    yield
    web_app = sys.modules.get("web.api.app")
    if web_app is not None:
        web_app._mda_client = None


# ── .env leak defence: clear dangerous env vars before each test ──────────────

_ENV_VARS_LEAKED_BY_DOTENV = [
    "WATCHDOG_IMESSAGE_TO",
]


@pytest.fixture(autouse=True)
def _clear_dotenv_leaked_vars():
    """Remove env vars that load_dotenv() leaks from .env into os.environ.

    web/api/app.py calls load_dotenv(project_root / ".env") at module-import
    time. Once any test imports the web module the .env values persist in
    os.environ for the rest of the session, regardless of monkeypatch scopes.

    Defense-in-depth until #13 (move load_dotenv out of import time) lands.
    This fixture shadows that leak for every test by removing the specific
    vars that have been shown to cause harm:

    - WATCHDOG_IMESSAGE_TO  (#11): IMessageConfig.from_env() reads this as a
      fallback when imessage_config=None, causing maybe_send_watchdog_alert to
      reach the iMessage send path instead of returning "imessage_not_configured"

    Tests that legitimately need one of these vars should set it explicitly via
    monkeypatch.setenv() — that will take effect after this fixture's setup.
    """
    saved = {k: os.environ.pop(k, None) for k in _ENV_VARS_LEAKED_BY_DOTENV}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ── Default singleton-store isolation ─────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_default_singleton_stores(tmp_path_factory, monkeypatch):
    """Redirect the default-path singleton stores to a per-test temp dir.

    BaselineEvidenceStore and RecommendationLedger expose module-global
    singletons (``_store`` / ``_ledger``) that lazily initialize to
    ``Path.home()/.options_calculator_pro/...`` via their getters. Production
    code (``run_daily_cycle``, ``evidence_report``, ...) reaches for those
    singletons whenever a store is not threaded through explicitly.

    conftest.py redirects HOME to the repo-local ``.pytest_home``, so those
    fall-through reaches wrote WAL SQLite files into
    ``.pytest_home/.options_calculator_pro/{evidence,recommendations}``. On any
    host where that directory is read-only (stale WAL/SHM perms from a prior
    run, a read-only checkout, a sandbox) the writes fail with
    ``sqlite3.OperationalError: attempt to write a readonly database`` — three
    such full-suite failures, even though the targeted tests pass.

    Pointing the defaults at a fresh per-test temp dir (and resetting the
    singletons before and after each test) keeps every test off the repo-local
    path and isolated from one another. Tests that already inject their own
    temp store are unaffected; this only governs the default fall-through.
    """
    import services.baseline_evidence_store as bes
    import services.recommendation_ledger as rl

    root = tmp_path_factory.mktemp("default_stores")
    bes_path = root / "evidence" / "baseline_evidence.sqlite"
    rl_path = root / "recommendations" / "recommendation_ledger.sqlite"
    bes_path.parent.mkdir(parents=True, exist_ok=True)
    rl_path.parent.mkdir(parents=True, exist_ok=True)

    # Getters read these module globals at call time, so monkeypatch reaches the
    # fall-through path. monkeypatch auto-restores them after the test.
    monkeypatch.setattr(bes, "_DEFAULT_STORE", bes_path)
    monkeypatch.setattr(rl, "_DEFAULT_LEDGER", rl_path)

    def _reset_singletons():
        for module, attr in ((bes, "_store"), (rl, "_ledger")):
            existing = getattr(module, attr, None)
            if existing is not None:
                try:
                    existing.close()
                except Exception:
                    pass
            setattr(module, attr, None)

    _reset_singletons()
    yield
    _reset_singletons()


# ── Sample DataFrames ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_chain_df():
    """MDApp-normalized option chain DataFrame with calls and puts."""
    today = datetime.date.today()
    near_exp = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    far_exp  = (today + datetime.timedelta(days=42)).strftime("%Y-%m-%d")

    rows = []
    for exp, dte in [(near_exp, 7), (far_exp, 42)]:
        for side in ("call", "put"):
            for strike, delta_sign in [(490.0, 1), (500.0, 1), (510.0, 1)]:
                delta = 0.5 if strike == 500.0 else (0.65 if strike == 490.0 else 0.35)
                if side == "put":
                    delta = -delta
                rows.append({
                    "expiration_date": exp,
                    "strike": strike,
                    "side": side,
                    "impliedVolatility": 0.28 + 0.02 * (abs(strike - 500.0) / 10.0),
                    "delta": delta,
                    "lastPrice": 5.0,
                    "bid": 4.8,
                    "ask": 5.2,
                    "openInterest": 1200,
                    "volume": 400,
                    "dte": dte,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_earnings_df():
    """MDApp-normalized earnings DataFrame."""
    today = datetime.date.today()
    rows = [
        {
            "report_date": today + datetime.timedelta(days=8),
            "reportTime": "after market close",
            "fiscal_year": 2025,
            "fiscal_quarter": 1,
            "eps_estimate": 1.50,
            "eps_actual": None,
        },
        {
            "report_date": today - datetime.timedelta(days=84),
            "reportTime": "after market close",
            "fiscal_year": 2024,
            "fiscal_quarter": 4,
            "eps_estimate": 1.42,
            "eps_actual": 1.48,
        },
        {
            "report_date": today - datetime.timedelta(days=175),
            "reportTime": "before market open",
            "fiscal_year": 2024,
            "fiscal_quarter": 3,
            "eps_estimate": 1.35,
            "eps_actual": 1.41,
        },
    ]
    return pd.DataFrame(rows)


# ── Mock MDApp client ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_mda_client(sample_chain_df, sample_earnings_df):
    """Returns a MagicMock that mimics MarketDataClient's public interface."""
    client = MagicMock()
    client.is_available.return_value = True

    today = datetime.date.today()
    near_exp = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    far_exp  = (today + datetime.timedelta(days=42)).strftime("%Y-%m-%d")

    client.get_expirations.return_value = [near_exp, far_exp]
    client.get_option_chain.return_value = sample_chain_df
    client.get_earnings.return_value = sample_earnings_df
    client.get_quote.return_value = 500.0

    return client


@pytest.fixture
def unavailable_mda_client():
    """MDApp client stub with no token — simulates unconfigured state."""
    client = MagicMock()
    client.is_available.return_value = False
    return client
