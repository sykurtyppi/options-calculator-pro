import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep test logs/cache in workspace-local paths to avoid host-specific HOME permissions.
PYTEST_HOME = PROJECT_ROOT / ".pytest_home"
PYTEST_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(PYTEST_HOME)


from services import external_io_gate  # noqa: E402  (after sys.path mutation above)


@pytest.fixture(scope="session", autouse=True)
def _install_external_io_guards():
    """Install transport-level guards once per session.

    `yfinance.Ticker` is patched globally so all 16 inline `yf.Ticker(...)`
    call sites consult the gate. The patch is idempotent.
    """
    external_io_gate.install_yfinance_guard()
    yield


@pytest.fixture(autouse=True)
def _block_external_io():
    """Block every external-IO category for each test by default.

    Tests that legitimately need a category opt in via `enable_external_io`.
    State resets before and after each test so nothing leaks between tests.
    """
    external_io_gate.disable_all()
    yield
    external_io_gate.disable_all()


@pytest.fixture
def enable_external_io():
    """Opt a single test back in to one (or all) external-IO categories.

    Usage:
        def test_sec_lookup(enable_external_io):
            enable_external_io(Category.EARNINGS_SEC_EDGAR)  # one category
            ...

        def test_kitchen_sink(enable_external_io):
            enable_external_io()  # everything — avoid unless truly needed
    """
    def grant(category=None):
        external_io_gate.enable(category)
    return grant
