#!/usr/bin/env python3
"""Freeze an evidence cohort, or report on a frozen one (F6).

    # Preview, then freeze, the events with earnings in a closed window:
    python scripts/freeze_evidence_cohort.py freeze --name cohort-1 --since 2026-10-01 --until 2026-12-31 --dry-run
    python scripts/freeze_evidence_cohort.py freeze --name cohort-1 --since 2026-10-01 --until 2026-12-31

    # Exit states of the frozen members only:
    python scripts/freeze_evidence_cohort.py report --name cohort-1

Read-only against the evidence stores. A freeze writes one new file under
~/.options_calculator_pro/evidence/cohorts/ and never overwrites one.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.evidence_cohort import (  # noqa: E402
    CohortError,
    build_cohort_manifest,
    cohort_report,
    load_cohort,
    write_cohort,
)
from services.event_resolution import load_event_resolutions  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="freeze the events in a closed earnings-date window")
    freeze.add_argument("--name", required=True)
    freeze.add_argument("--since", required=True, type=date.fromisoformat)
    freeze.add_argument("--until", required=True, type=date.fromisoformat)
    freeze.add_argument("--dry-run", action="store_true", help="print the manifest without writing it")
    report = sub.add_parser("report", help="exit states of a frozen cohort's members")
    report.add_argument("--name", required=True)
    for command in (freeze, report):
        command.add_argument("--as-of", type=date.fromisoformat, default=date.today())
        command.add_argument("--directory", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        if args.command == "freeze":
            manifest = build_cohort_manifest(name=args.name, since=args.since, until=args.until, as_of=args.as_of,
                                             events=load_event_resolutions(as_of=args.as_of))
            if args.dry_run:
                print(json.dumps(manifest, indent=2, sort_keys=True))
                return 0
            path = write_cohort(manifest, directory=args.directory)
            print(json.dumps({"frozen": str(path), "member_count": manifest["member_count"],
                              "members_sha256": manifest["members_sha256"]}, indent=2))
            return 0
        manifest = load_cohort(args.name, directory=args.directory)
        events = load_event_resolutions(as_of=args.as_of, event_keys={m["event_key"] for m in manifest["members"]})
        print(json.dumps(cohort_report(manifest, events), indent=2, sort_keys=True))
        return 0
    except CohortError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
