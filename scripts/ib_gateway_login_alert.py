#!/usr/bin/env python3
"""Check that IB Gateway is signed in; text the operator if it is not.

Called at the start of the 15:00 screener-alert job (``--context premarket``)
and of the 16:45 evidence cycle (``--context cycle``). It reports and never
blocks: the exit code is 0 whether or not the gateway answered, so the job
that called it always runs. See services/ib_gateway_login_alert.py.

    .venv311/bin/python scripts/ib_gateway_login_alert.py --context premarket --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv is optional; the launchd env may already be complete
    pass

from services.ib_gateway_login_alert import (  # noqa: E402
    CONTEXTS,
    DEFAULT_STATE_PATH,
    build_login_status,
    maybe_send_login_alert,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context", choices=CONTEXTS, required=True,
                        help="which job is asking: 'premarket' (time left to sign in) or 'cycle' (starting now)")
    parser.add_argument("--dry-run", action="store_true", help="compose the alert but do not send or record it")
    parser.add_argument("--force", action="store_true", help="send even if this job already alerted today")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        status = build_login_status(context=args.context, environ=os.environ)
        alert = maybe_send_login_alert(status, state_path=args.state_path, force=args.force, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - a broken check must never stop the job that called it
        print(json.dumps({"ib_gateway_login_check": {"error": f"{type(exc).__name__}: {exc}"}}))
        return 0
    print(json.dumps({"ib_gateway_login_check": {"status": status, "alert": alert}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
