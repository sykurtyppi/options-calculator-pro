#!/usr/bin/env python3
"""Where forward evidence stops, per earnings event, and how fast it arrives.

Read-only: opens the evidence funnel store with mode=ro and writes nothing.

  .venv311/bin/python scripts/diagnose_evidence_funnel.py            # text
  .venv311/bin/python scripts/diagnose_evidence_funnel.py --json     # full detail
  .venv311/bin/python scripts/diagnose_evidence_funnel.py --since 2026-09-01
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.evidence_funnel import _DEFAULT_STORE  # noqa: E402
from services.funnel_diagnosis import (  # noqa: E402
    DEFAULT_TRAILING_WEEKS,
    diagnose_funnel,
    read_funnel_rows,
    render_text,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=_DEFAULT_STORE, help=f"Funnel store (default {_DEFAULT_STORE}).")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None, help="Evaluate as of this date (default today).")
    parser.add_argument("--since", type=date.fromisoformat, default=None,
                        help="Only events first seen on or after this date (default: all history).")
    parser.add_argument("--trailing-weeks", type=int, default=DEFAULT_TRAILING_WEEKS,
                        help="Weeks used for the claimable rate.")
    parser.add_argument("--json", action="store_true", help="Print the full diagnosis as JSON.")
    args = parser.parse_args(argv)
    if args.trailing_weeks < 1:
        parser.error("--trailing-weeks must be at least 1")
    if not args.store.exists():
        print(f"[diagnose_evidence_funnel] no funnel store at {args.store}", file=sys.stderr)
        return 1
    rows = read_funnel_rows(args.store)
    result = diagnose_funnel(rows["events"], rows["cycles"], as_of=args.as_of or date.today(),
                             since=args.since, trailing_weeks=args.trailing_weeks)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else render_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
