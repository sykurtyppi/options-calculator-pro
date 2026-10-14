#!/usr/bin/env python3
"""Pre-market ranked-screener alert.

Runs the canonical ranked pre-earnings long-vega screener
(``services.screener_service.build_ranked_screener`` — the same code path
behind ``/api/screener/ranked``) and pushes the qualifying setups to the
operator over iMessage, so a setup surfaces without anyone remembering to
open the UI.

Design notes
------------
* **No backend dependency.** The screener is called in-process, so the job
  does not require uvicorn to be running.
* **Sent at most once.** Each alert digest (date + qualifying symbols) is
  reserved in a durable outbox (``services.alert_outbox``) before anything is
  handed to Messages, so concurrent runs cannot both send. A timed-out
  hand-off is recorded as ``delivery_unknown`` and never retried
  automatically; an unreadable outbox or state file is quarantined and
  nothing is sent. A re-run with the same qualifying set is a no-op unless
  ``--force`` is passed.
* **Silence is a valid outcome.** Qualifying setups are rare by design
  (~10-40/year). "Nothing qualified" exits 0 without sending, rather than
  manufacturing a daily message the operator learns to ignore.
* **Honesty in the payload.** ``ranking_score`` is a heuristic setup-quality
  ordering (weights and the 0.65 cutoff are not empirically calibrated), NOT
  a win probability, and the message says so. A setup is a candidate: the
  forward loop's selector still decides whether it is a trade.
* **Hard evidence gates.** The score fills missing inputs with neutral
  defaults, so a finite score can hide missing data. The alert therefore
  requires the inputs themselves: term structure, move history and an ATM
  spread no wider than ``MAX_ATM_SPREAD_PCT``, an ATM IV inside
  [``MIN_PLAUSIBLE_ATM_IV``, ``MAX_PLAUSIBLE_ATM_IV``], and a non-stale
  earnings date. An unconfirmed earnings date is labelled, not dropped.
* **Audit trail.** Every sent alert appends its full rows, score components
  (including the IV/RV actually scored) and drop counts to
  ``DEFAULT_AUDIT_PATH``.
* **Timing.** launchd runs it at 15:00 machine-local time (Reykjavik = UTC):
  11:00 ET in daylight time, 10:00 ET in standard time, always after the
  09:30 ET open. The message states the New York time it was built.

Exit codes: ``0`` success (accepted, already sent, or nothing to send);
``1`` failure, or a delivery that is unknown and needs a look.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# launchd starts jobs with a bare environment, so the recipient (and provider
# tokens) must come from the repo .env rather than the inherited shell.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from services.alert_outbox import (  # noqa: E402
    ACCEPTED,
    DELIVERY_UNKNOWN,
    FAILED,
    PREPARED,
    PROVIDER_SUBMITTED,
    AlertOutbox,
    OutboxCorrupt,
    OutboxUnavailable,
    quarantine,
)
from services.automation_watchdog import IMessageConfig, send_imessage  # noqa: E402
from services.screener_service import build_ranked_screener, ranking_score_components  # noqa: E402

logger = logging.getLogger("premarket_screener_alert")

DEFAULT_MIN_SCORE = 0.65
DEFAULT_TOP_N = 5
DEFAULT_STATE_PATH = (
    Path.home() / ".options_calculator_pro" / "state" / "premarket_alert_state.json"
)
DEFAULT_AUDIT_PATH = (
    Path.home() / ".options_calculator_pro" / "logs" / "premarket_alert_audit.jsonl"
)
# Execution gate: where the screener's liquidity sub-score reaches zero.
MAX_ATM_SPREAD_PCT = 15.0
# The screener rounds ranking_score to 4 decimals; a recomputed score further
# from it than this is not the score of the row's own inputs.
SCORE_RECONCILE_TOLERANCE = 1e-4


def _number(value: Any) -> Optional[float]:
    """A real, finite number; else None. Strict: booleans and strings are not
    numbers here (this boundary texts a human, so nothing is coerced)."""
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _whole_number(value: Any) -> Optional[int]:
    """An integral count (7 or 7.0), else None; 10.9 is not truncated to 10."""
    f = _number(value)
    if f is None or not f.is_integer():
        return None
    return int(f)


def _row_dte(row: Mapping[str, Any]) -> Optional[int]:
    """DTE from the service key, else the API-shaped alias; must be whole."""
    for key in ("days_to_earnings", "dte"):
        if row.get(key) is not None:
            return _whole_number(row.get(key))
    return None


def _row_atm_iv(row: Mapping[str, Any]) -> Optional[float]:
    """ATM IV from ``iv30``, else the ``atm_iv`` alias. A present but invalid
    ``iv30`` is not replaced by the alias."""
    if row.get("iv30") is not None:
        return _number(row.get("iv30"))
    return _number(row.get("atm_iv"))


def _is_stale(row: Mapping[str, Any]) -> bool:
    """Only an explicit "not stale" (None, False, 0, "false") passes; any
    other value counts as stale (fail closed)."""
    value = row.get("earnings_source_stale")
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "no"}
    if not isinstance(value, bool) and _number(value) == 0:
        return False
    return True


def _finite(value: Any) -> Optional[float]:
    """float(value) if it is a real, finite number; else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# Below this, an "ATM IV" is a provider placeholder, not a quote. Mirrors
# services.earnings_vol_snapshot.MIN_PLAUSIBLE_ATM_IV so the two layers agree.
MIN_PLAUSIBLE_ATM_IV = 0.01
# Above this an "ATM IV" (500%) is a solver or feed artifact, not a quote.
MAX_PLAUSIBLE_ATM_IV = 5.0


def select_qualifying_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_score: float,
    top_n: int,
    dte_min: int,
    dte_max: int,
    dropped: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Filter the screener table down to setups worth interrupting someone for.

    This job runs unattended and texts a human, so a row must prove its
    evidence rather than merely avoid tripping an error. ``dropped`` (if
    given) is filled with the count per drop reason:

      * a finite ``ranking_score`` — ``NaN`` compares False against every
        threshold and ``inf`` beats every threshold, so both used to qualify;
      * a finite, positive ``iv_rv_ratio`` — the 32%-weight IV-entry component
        falls back to a "neutral" 0.25 when IV/RV is missing, and the other
        68% of weight can carry a row over the threshold with no volatility
        evidence at all;
      * a near-term ATM IV (``iv30`` / ``atm_iv``) inside
        [MIN_PLAUSIBLE_ATM_IV, MAX_PLAUSIBLE_ATM_IV] — sub-1% values are
        pre-open placeholders, values above 500% are solver/feed artifacts;
      * the inputs the score would otherwise fill with a neutral default:
        term structure, median historical move, and a finite ATM spread no
        wider than MAX_ATM_SPREAD_PCT;
      * an earnings date that is not flagged stale.

    ``min_score`` itself must be finite; ``--min-score nan`` previously admitted
    every scored row.
    """
    if _finite(min_score) is None:
        raise ValueError(f"min_score must be a finite number, got {min_score!r}")

    qualifying: List[Dict[str, Any]] = []
    if dropped is None:
        dropped = {}

    def _drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for row in rows:
        if row.get("error") or row.get("error_note"):
            _drop("error"); continue
        score = _number(row.get("ranking_score"))
        if score is None:
            _drop("non_finite_score"); continue
        if not 0.0 <= score <= 1.0:
            _drop("score_out_of_range"); continue
        dte = _row_dte(row)
        if dte is None or not (dte_min <= dte <= dte_max):
            _drop("outside_entry_window"); continue
        iv_rv = _number(row.get("iv_rv_ratio"))
        if iv_rv is None or iv_rv <= 0:
            _drop("missing_iv_rv"); continue
        atm_iv = _row_atm_iv(row)
        if atm_iv is None or not (MIN_PLAUSIBLE_ATM_IV <= atm_iv <= MAX_PLAUSIBLE_ATM_IV):
            _drop("implausible_or_missing_atm_iv"); continue
        spread = _row_spread(row)
        if spread is None:
            _drop("missing_spread"); continue
        if spread > MAX_ATM_SPREAD_PCT:
            _drop("spread_too_wide"); continue
        # The scorer treats a ratio <= 0 as missing (neutral 0.30).
        term_structure = _number(row.get("term_structure_ratio"))
        if term_structure is None or term_structure <= 0:
            _drop("missing_term_structure"); continue
        # An absolute move cannot be negative.
        move = _number(row.get("median_earnings_move_pct"))
        if move is None or move < 0:
            _drop("missing_move_history"); continue
        sample_size = _whole_number(row.get("sample_size"))
        if sample_size is None or sample_size < 0:
            _drop("invalid_sample_size"); continue
        if _is_stale(row):
            _drop("stale_earnings_date"); continue
        # The score must be what its own inputs produce, with no input scored
        # as a neutral default; the audit then records exactly what was scored.
        breakdown = score_breakdown(row)
        if breakdown["neutral_fallbacks"]:
            _drop("neutral_fallback_inputs"); continue
        if abs(breakdown["score"] - score) > SCORE_RECONCILE_TOLERANCE:
            _drop("score_mismatch"); continue
        if score < min_score:
            _drop("below_threshold"); continue
        qualifying.append(dict(row))

    if dropped:
        logger.info("rows dropped by fail-closed filter: %s", dropped)
    qualifying.sort(key=lambda r: float(r.get("ranking_score") or 0.0), reverse=True)
    return qualifying[: max(int(top_n), 0)]


def _row_spread(row: Mapping[str, Any]) -> Optional[float]:
    """The ATM spread %, tolerating both the service key and the API alias."""
    for key in ("avg_spread_pct", "spread_pct"):
        if row.get(key) is not None:
            value = _number(row[key])
            return value if value is not None and value >= 0 else None
    return None


def _date_unconfirmed(row: Mapping[str, Any]) -> bool:
    """The earnings layer names the confirming source (e.g. "sec_edgar"), or
    leaves it empty; a bare True is also accepted."""
    value = row.get("earnings_source_confirmed")
    if value is True:
        return False
    return not (isinstance(value, str) and value.strip())


def score_breakdown(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute the row's score components from its own inputs (audit).

    Never raises on malformed inputs: an invalid value is passed as missing,
    so it shows up in ``neutral_fallbacks``."""
    return ranking_score_components(
        iv_rv_ratio=_number(row.get("iv_rv_ratio")),
        ts_ratio=_number(row.get("term_structure_ratio")),
        median_earnings_move_pct=_number(row.get("median_earnings_move_pct")),
        sample_size=max(_whole_number(row.get("sample_size")) or 0, 0),
        dte=_row_dte(row),
        spread_pct=_row_spread(row),
        iv_rv_har=_number(row.get("iv_rv_har")),
        rv_percentile_rank=_number(row.get("rv_percentile_rank")),
    )


def _new_york_time(moment: datetime) -> str:
    from zoneinfo import ZoneInfo

    return moment.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M %Z")


def _market_phase(moment: datetime) -> str:
    """Where ``moment`` falls relative to the NYSE regular session."""
    from datetime import time
    from zoneinfo import ZoneInfo

    from services.market_calendar import is_trading_day

    local = moment.astimezone(ZoneInfo("America/New_York"))
    if not is_trading_day(local.date()):
        return "US market closed today"
    if local.time() < time(9, 30):
        return "before the US open"
    if local.time() >= time(16, 0):
        return "after the US close"
    return "after the US open"


def _fmt(value: Any, spec: str = ".2f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "—"


def format_alert_message(
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of: date,
    min_score: float,
    built_at: Optional[datetime] = None,
) -> str:
    """Render the iMessage body.

    Kept well under the 1500-char truncation in ``send_imessage`` — at the
    default top-5 this lands around 300 characters.
    """
    lines = [f"Earnings vol setups — {as_of.isoformat()}", ""]
    for idx, row in enumerate(rows, start=1):
        symbol = str(row.get("symbol") or "?")
        dte = _row_dte(row)
        timing = str(row.get("release_timing") or "?")
        score = _fmt(row.get("ranking_score"))
        iv_rv = row.get("iv_rv_ratio")
        iv_rv_txt = f" IV/RV {_fmt(iv_rv)}" if iv_rv is not None else " IV/RV —"
        flag = ""
        if row.get("iv_regime_conditioned"):
            # The score used the less favourable forward IV/RV; say which.
            scored = score_breakdown(row).get("scored_iv_rv")
            flag = f" [regime: scored IV/RV {_fmt(scored)}]"
        date_flag = " [date unconfirmed]" if _date_unconfirmed(row) else ""
        lines.append(
            f"{idx}. {symbol} {dte if dte is not None else '?'}d {timing} "
            f"score {score}{iv_rv_txt}{flag}{date_flag}"
        )
    built = built_at or datetime.now(timezone.utc)
    lines += [
        "",
        f"Pre-earnings long-vol candidates (heuristic score >= {min_score:g}), "
        "not a win probability. Selector approval still required.",
        f"Built {_new_york_time(built)} ({_market_phase(built)}).",
        "Research only - not financial advice.",
    ]
    return "\n".join(lines)


def _digest(as_of: date, rows: Sequence[Mapping[str, Any]]) -> str:
    symbols = ",".join(sorted(str(r.get("symbol") or "") for r in rows))
    return f"{as_of.isoformat()}|{symbols}"


class StateUnreadable(RuntimeError):
    """The idempotency state exists but cannot be read; never treated as empty."""


def _load_state(path: Path) -> Dict[str, Any]:
    """The last-sent state; {} only when there is none yet."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise StateUnreadable(f"{path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise StateUnreadable(f"{path}: not valid JSON") from exc
    if not isinstance(payload, dict):
        raise StateUnreadable(f"{path}: not a JSON object")
    return payload


def _save_state(path: Path, payload: Mapping[str, Any]) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    # A temp file of its own per writer: concurrent runs never share one.
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)  # atomic swap; a torn write can't poison the state file
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _json_safe(value: Any) -> Any:
    """Strict-JSON form: non-finite floats become null (a NaN in a field the
    gates do not use must not block the record)."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _append_audit(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one strict-JSON line, flushed to disk. Raises on failure."""
    line = json.dumps(_json_safe(dict(payload)), default=str, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _try_append_audit(path: Path, payload: Mapping[str, Any]) -> bool:
    try:
        _append_audit(path, payload)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Could not append to the alert audit %s.", path)
        return False


def _finite_float_arg(text: str) -> float:
    """argparse type: a finite float. Rejects nan/inf, which would otherwise
    disable the threshold entirely (``score < nan`` is always False)."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"threshold must be finite, got {text!r}")
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ranked earnings screener and alert on qualifying setups.",
    )
    parser.add_argument("--min-score", type=_finite_float_arg, default=DEFAULT_MIN_SCORE,
                        help=f"Minimum ranking_score to alert (default {DEFAULT_MIN_SCORE}).")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help=f"Max setups per alert (default {DEFAULT_TOP_N}).")
    parser.add_argument("--dte-min", type=int, default=3, help="Entry-window minimum DTE.")
    parser.add_argument("--dte-max", type=int, default=10, help="Entry-window maximum DTE.")
    parser.add_argument("--weeks", type=int, default=4, help="Forward earnings search window.")
    parser.add_argument("--min-sample-size", type=int, default=4,
                        help="Minimum historical earnings events per symbol.")
    parser.add_argument("--release", choices=("all", "amc", "bmo"), default="all",
                        help="Restrict to AMC/BMO reporters.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the alert without sending it or writing state.")
    parser.add_argument("--force", action="store_true",
                        help="Send even if an identical digest already went out today.")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH,
                        help="Idempotency state file.")
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH,
                        help="JSONL audit trail of every sent alert.")
    parser.add_argument("--outbox-path", type=Path, default=None,
                        help="Durable send outbox (default: next to the state file).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args(argv)
    as_of = date.today()

    release_filter = None if args.release == "all" else args.release.upper()
    try:
        table = build_ranked_screener(
            dte_min=args.dte_min,
            dte_max=args.dte_max,
            min_sample_size=args.min_sample_size,
            release_filter=release_filter,
            weeks=args.weeks,
            today=as_of,
        )
    except Exception:
        logger.exception("Ranked screener failed; no alert sent.")
        return 1

    rows = table.get("rows") or []
    dropped: Dict[str, int] = {}
    qualifying = select_qualifying_rows(
        rows,
        min_score=args.min_score,
        top_n=args.top,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        dropped=dropped,
    )
    logger.info(
        "screened universe=%s rows=%s in_window=%s qualifying=%s (min_score=%s)",
        table.get("universe_size"), len(rows), table.get("in_entry_window"),
        len(qualifying), args.min_score,
    )

    if not qualifying:
        # Deliberately silent: qualifying setups are rare, and a daily
        # "nothing today" message trains the operator to ignore the channel.
        logger.info("No qualifying setups; nothing to send.")
        return 0

    built_at = datetime.now(timezone.utc)
    message = format_alert_message(qualifying, as_of=as_of, min_score=args.min_score, built_at=built_at)
    digest = _digest(as_of, qualifying)

    # Dry run is a preview and must stay observable, so it is checked BEFORE
    # the idempotency checks — otherwise once a real alert has gone out,
    # --dry-run prints "already sent" instead of the message you asked to see.
    if args.dry_run:
        print(message)
        logger.info("Dry run — not sent, state not written.")
        return 0

    # Two independent guards against a second send: the last-sent state file
    # and the outbox. Neither is ever read as empty when it is unreadable.
    try:
        state = _load_state(args.state_path)
    except StateUnreadable as exc:
        moved = quarantine(args.state_path)
        logger.error("Alert state is unreadable (%s); moved to %s. Nothing sent.", exc, moved)
        return 1
    if not args.force and state.get("last_digest") == digest:
        logger.info("Identical alert already sent today (digest match); skipping.")
        return 0

    config = IMessageConfig.from_env(os.environ)
    if config is None:
        logger.error(
            "No iMessage recipient configured. Set WATCHDOG_IMESSAGE_TO in .env "
            "(or run with --dry-run)."
        )
        return 1

    import uuid

    event_id = uuid.uuid4().hex
    outbox_path = args.outbox_path or args.state_path.with_name("premarket_alert_outbox.sqlite")
    try:
        outbox = AlertOutbox(outbox_path)
    except OutboxCorrupt as exc:
        moved = quarantine(outbox_path)
        logger.error("Alert outbox is damaged (%s); moved to %s. Nothing sent.", exc, moved)
        return 1
    except OutboxUnavailable as exc:
        # Contention or I/O, not damage: leave the outbox where it is.
        logger.error("Alert outbox is unavailable right now (%s). Nothing sent.", exc)
        return 1
    # --force sends again under a fresh key; otherwise one key per digest.
    alert_key = f"{digest}|force|{event_id}" if args.force else digest
    reservation = outbox.reserve(alert_key, event_id=event_id)
    if not reservation.acquired:
        # A hand-off that timed out, or one whose run died mid-send (still
        # "provider_submitted" long after any live run would have finished),
        # may have been delivered: flag it, never resend it.
        if reservation.status == DELIVERY_UNKNOWN or (
                reservation.status == PROVIDER_SUBMITTED and outbox.is_stale(alert_key)):
            logger.error("Alert %s: an earlier send may or may not have been delivered; "
                         "not resending automatically.", digest)
            return 1
        logger.info("Alert %s already %s; not sending again.", digest, reservation.status)
        return 0

    def _audit_status(status: str, **extra: Any) -> bool:
        return _try_append_audit(args.audit_path, {
            "event_type": "premarket_alert", "event_id": event_id, "alert_key": alert_key,
            "status": status, "at_utc": datetime.now(timezone.utc).isoformat(), **extra,
        })

    # The full audit record is written as "prepared" BEFORE the send, so every
    # sent alert has a durable record. If it cannot be written, nothing is sent.
    try:
        record = {
            "event_type": "premarket_alert",
            "event_id": event_id,
            "alert_key": alert_key,
            "status": PREPARED,
            "attempt": reservation.attempts,
            "as_of_date": as_of.isoformat(),
            "built_at_utc": built_at.isoformat(),
            "built_at_new_york": _new_york_time(built_at),
            "min_score": args.min_score,
            "gates": {
                "dte_min": args.dte_min, "dte_max": args.dte_max,
                "max_atm_spread_pct": MAX_ATM_SPREAD_PCT,
                "atm_iv_range": [MIN_PLAUSIBLE_ATM_IV, MAX_PLAUSIBLE_ATM_IV],
            },
            "screened": {"universe_size": table.get("universe_size"), "rows": len(rows),
                         "in_entry_window": table.get("in_entry_window")},
            "dropped": dropped,
            "message": message,
            "setups": [{"row": row, "score_breakdown": score_breakdown(row)} for row in qualifying],
        }
        _append_audit(args.audit_path, record)
    except Exception as exc:
        logger.exception("Could not record the alert audit before sending; nothing sent.")
        outbox.transition(alert_key, event_id=event_id, to=FAILED, error=f"audit: {type(exc).__name__}")
        return 1

    # Committed before the hand-off: if this run dies mid-send, the alert is
    # treated as possibly delivered and never resent automatically.
    outbox.transition(alert_key, event_id=event_id, to=PROVIDER_SUBMITTED)
    try:
        # No automatic retry: a timed-out hand-off may already have gone out.
        result = send_imessage(message, config=config, retries=0)
    except subprocess.TimeoutExpired as exc:
        error = f"{type(exc).__name__}: {exc}"
        outbox.transition(alert_key, event_id=event_id, to=DELIVERY_UNKNOWN, error=error)
        logger.error("iMessage hand-off timed out; delivery unknown, not retried.")
        _audit_status(DELIVERY_UNKNOWN, error=error)
        return 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        outbox.transition(alert_key, event_id=event_id, to=FAILED, error=error)
        logger.exception("iMessage send failed; nothing was sent.")
        _audit_status(FAILED, error=error)
        return 1

    outbox.transition(alert_key, event_id=event_id, to=ACCEPTED, detail={"to": result.get("to")})
    # Accepted: the outbox already prevents a resend, so a failure to update
    # the state file or the closing audit line is reported but never resends.
    code = 0
    try:
        _save_state(args.state_path, {
            "last_digest": digest,
            "last_sent_at": datetime.now(timezone.utc).isoformat(),
            "symbols": [str(r.get("symbol")) for r in qualifying],
            "audit_event_id": event_id,
        })
    except Exception:
        logger.exception("Alert %s was accepted but the state file could not be updated.", event_id)
        code = 1
    if not _audit_status(ACCEPTED):
        logger.error("Alert %s was accepted but its closing audit line could not be written.", event_id)
        code = 1
    logger.info("Alert accepted for %s (%s setups).", result.get("to"), len(qualifying))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
