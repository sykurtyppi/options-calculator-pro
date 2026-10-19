"""Frozen evidence cohorts (F6): fix who is in a sample before reading results.

A cohort is the set of earnings events in a date window, with the exact rows
each track holds for them (paper trade ids, baseline ids, the resolver's
recommendation id). Freezing writes that membership once, with a hash, to a
file that is never overwritten, so a later analysis cannot quietly add or drop
events after seeing outcomes.

A window can only be frozen once it has closed, i.e. every member event is
past the resolver's post-event wait, so membership no longer changes.
Results are then read only for the frozen members: rows added later for the
same events are reported but left out, and rows that disappeared are named.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from services.claim_contract import CONTRACT_VERSION
from services.event_resolution import (
    RESOLVER_MIN_DAYS_AFTER_EVENT,
    combine_states,
    summarize_event_resolutions,
)

MANIFEST_VERSION = 2
_DEFAULT_DIR = Path.home() / ".options_calculator_pro" / "evidence" / "cohorts"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class CohortError(ValueError):
    """A cohort cannot be frozen or read as asked."""


def _members_hash(members: list[Dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_INTEGRITY_FIELDS = (
    "name", "manifest_version", "claim_contract_version", "window",
    "frozen_as_of", "frozen_at", "member_count", "members",
)


def _integrity_hash(manifest: Dict[str, Any]) -> str:
    """Bind every authoritative field used to interpret or report a cohort."""
    bound = {field: manifest.get(field) for field in _INTEGRITY_FIELDS}
    return hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_cohort_manifest(
    *, name: str, since: date, until: date, as_of: date, events: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not _NAME.match(name):
        raise CohortError("cohort name must be lowercase letters, digits, '-' or '_'")
    if since > until:
        raise CohortError("since must be on or before until")
    closes_on = until + timedelta(days=RESOLVER_MIN_DAYS_AFTER_EVENT)
    if as_of < closes_on:
        raise CohortError(f"the window is still open: it can be frozen from {closes_on.isoformat()}")
    members = []
    for key, event in sorted(events.items()):
        earnings = date.fromisoformat(event["earnings_date"])
        if since <= earnings <= until:
            members.append({
                "event_key": key,
                "tracks": {track: sorted(item["id"] for item in items)
                           for track, items in sorted(event["tracks"].items())},
            })
    window = {"since": since.isoformat(), "until": until.isoformat()}
    manifest = {
        "name": name,
        "manifest_version": MANIFEST_VERSION,
        "claim_contract_version": CONTRACT_VERSION,
        "window": window,
        "frozen_as_of": as_of.isoformat(),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "member_count": len(members),
        "members": members,
        "members_sha256": _members_hash(members),
        "integrity_verification": "metadata_and_members",
    }
    manifest["integrity_sha256"] = _integrity_hash(manifest)
    return manifest


def write_cohort(manifest: Dict[str, Any], *, directory: Optional[Path] = None) -> Path:
    """Write a manifest once; an existing cohort of the same name is never replaced."""
    folder = Path(directory or _DEFAULT_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{manifest['name']}.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise CohortError(f"cohort {manifest['name']!r} is already frozen at {path}") from exc
    return path


def load_cohort(name: str, *, directory: Optional[Path] = None) -> Dict[str, Any]:
    if not _NAME.match(name):
        raise CohortError("cohort name must be lowercase letters, digits, '-' or '_'")
    path = Path(directory or _DEFAULT_DIR) / f"{name}.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CohortError(f"no frozen cohort {name!r} at {path}") from exc
    members = manifest.get("members") or []
    if _members_hash(members) != manifest.get("members_sha256"):
        raise CohortError(f"cohort {name!r} does not match its frozen hash; its membership was edited")
    version = manifest.get("manifest_version", 1)
    if version == MANIFEST_VERSION:
        actual = _integrity_hash(manifest)
        if actual != manifest.get("integrity_sha256"):
            raise CohortError(f"cohort {name!r} failed integrity verification; interpretation metadata was edited")
        if manifest.get("name") != name:
            raise CohortError(f"cohort {name!r} failed integrity verification; embedded name does not match")
        if manifest.get("member_count") != len(members):
            raise CohortError(f"cohort {name!r} failed integrity verification; member count does not match")
        # This label describes verification performed by this reader; never
        # trust an editable value stored in the manifest itself.
        manifest["integrity_verification"] = "metadata_and_members"
    elif version == 1 and "integrity_sha256" not in manifest and "integrity_verification" not in manifest:
        # Genuine v1 files predate both v2 integrity fields.  A v2 file cannot
        # opt itself out of metadata verification merely by editing its
        # manifest_version value while retaining the v2 envelope.
        manifest["integrity_verification"] = "legacy_members_only"
    else:
        raise CohortError(f"cohort {name!r} has an unsupported version or attempted integrity downgrade")
    return manifest


def cohort_report(manifest: Dict[str, Any], events: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Current exit states of the frozen members only."""
    member_events: Dict[str, Dict[str, Any]] = {}
    missing_rows: list[Dict[str, str]] = []
    added_rows: list[Dict[str, str]] = []
    for member in manifest["members"]:
        key = member["event_key"]
        live = events.get(key, {"tracks": {}})
        live_by_track = {track: {item["id"]: item["state"] for item in items}
                         for track, items in live["tracks"].items()}
        frozen_tracks: Dict[str, list[Dict[str, str]]] = {}
        for track, ids in member["tracks"].items():
            for row_id in ids:
                state = live_by_track.get(track, {}).get(row_id)
                if state is None:
                    missing_rows.append({"event_key": key, "track": track, "id": row_id})
                    continue
                frozen_tracks.setdefault(track, []).append({"id": row_id, "state": state})
        for track, states in live_by_track.items():
            for row_id in states:
                if row_id not in member["tracks"].get(track, []):
                    added_rows.append({"event_key": key, "track": track, "id": row_id})
        track_states = {track: combine_states(item["state"] for item in items)
                        for track, items in frozen_tracks.items()}
        member_events[key] = {"event_key": key, "tracks": frozen_tracks, "track_states": track_states,
                               "state": combine_states(track_states.values())}
    return {
        "name": manifest["name"],
        "window": manifest["window"],
        "claim_contract_version": manifest["claim_contract_version"],
        "members_sha256": manifest["members_sha256"],
        "summary": summarize_event_resolutions(member_events),
        # Rows that left the stores since the freeze (e.g. invalidated), and
        # rows created later for member events: reported, never counted.
        "frozen_rows_missing": missing_rows,
        "rows_added_after_freeze": added_rows,
    }


__all__ = ["CohortError", "build_cohort_manifest", "cohort_report", "load_cohort", "write_cohort"]
