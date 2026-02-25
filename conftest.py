import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep test logs/cache in workspace-local paths to avoid host-specific HOME permissions.
PYTEST_HOME = PROJECT_ROOT / ".pytest_home"
PYTEST_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(PYTEST_HOME)
