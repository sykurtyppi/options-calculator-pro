#!/usr/bin/env python3
"""Mark a paper outcome as invalid evidence, with a recorded reason.

The row stays in the outcome store for audit; every evidence report and
diagnostic then excludes it, and exit detection will not finalize it into the
calibration/prior stores. Invalidate BEFORE an outcome is finalized when you
can: those stores cannot drop one observation afterwards, and the command says
so when that has already happened.

    python scripts/invalidate_outcome.py --list
    python scripts/invalidate_outcome.py --trade-id 'AMZN|2026-04-20|otm_strangle' \\
        --reason 'exit repriced a different strike than booked'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.outcome_recorder import (
    OutcomeStore,
    _DEFAULT_STORE,
    is_outcome_evidence_valid,
    outcome_invalidation_reason,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mark a paper outcome as invalid evidence.")
    parser.add_argument("--store", type=Path, default=_DEFAULT_STORE, help="Outcome store SQLite path.")
    parser.add_argument("--trade-id", action="append", default=[], help="Trade id to invalidate (repeatable).")
    parser.add_argument("--reason", type=str, default="", help="Why the outcome is not valid evidence (required).")
    parser.add_argument("--list", action="store_true", help="List resolved outcomes with their validity and exit.")
    args = parser.parse_args(argv)

    if not args.store.exists():
        print(f"outcome store not found: {args.store}", file=sys.stderr)
        return 2
    store = OutcomeStore(store_path=args.store)
    try:
        if args.list:
            for row in store.list_for_diagnostics():
                if row.get("realized_return_pct") is None:
                    continue
                print(json.dumps({
                    "trade_id": row.get("trade_id"),
                    "status": row.get("status"),
                    "realized_return_pct": row.get("realized_return_pct"),
                    "evidence_valid": is_outcome_evidence_valid(row),
                    "invalidation_reason": outcome_invalidation_reason(row),
                }))
            return 0
        if not args.trade_id:
            parser.error("--trade-id is required unless --list is given")
        if not args.reason.strip():
            parser.error("--reason is required")
        exit_code = 0
        for trade_id in args.trade_id:
            try:
                result = store.invalidate(trade_id, reason=args.reason)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                exit_code = 1
                continue
            print(json.dumps(result))
            if result["learning_already_applied"]:
                print(
                    f"WARNING: {trade_id} was already finalized into calibration and structure "
                    "priors; those stores still contain it.",
                    file=sys.stderr,
                )
        return exit_code
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
