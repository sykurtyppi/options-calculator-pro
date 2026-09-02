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
* **Idempotent per day.** The alert digest (date + qualifying symbols) is
  recorded under ``~/.options_calculator_pro/state/``. A re-run on the same
  day with the same qualifying set is a no-op unless ``--force`` is passed,
  matching the forward-paper-collector's idempotency discipline.
* **Silence is a valid outcome.** Qualifying setups are rare by design
  (~10-40/year). "Nothing qualified" exits 0 without sending, rather than
  manufacturing a daily message the operator learns to ignore.
* **Honesty in the payload.** ``ranking_score`` is setup-quality ordering,
  NOT a calibrated win probability, and the message says so — the same
  claim discipline the UI and model cards hold to.

Exit codes: ``0`` success (sent, or nothing to send), ``1`` failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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

from services.automation_watchdog import IMessageConfig, send_imessage  # noqa: E402
from services.screener_service import build_ranked_screener  # noqa: E402

logger = logging.getLogger("premarket_screener_alert")

DEFAULT_MIN_SCORE = 0.65
DEFAULT_TOP_N = 5
DEFAULT_STATE_PATH = (
    Path.home() / ".options_calculator_pro" / "state" / "premarket_alert_state.json"
)


def _row_dte(row: Mapping[str, Any]) -> Optional[int]:
    """DTE, tolerating both the service key and the API-shaped alias."""
    for key in ("days_to_earnings", "dte"):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def select_qualifying_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_score: float,
    top_n: int,
    dte_min: int,
    dte_max: int,
) -> List[Dict[str, Any]]:
    """Filter the screener table down to setups worth interrupting someone for.

    A row qualifies when it actually scored (no error / not merely "upcoming"),
    sits inside the entry window, and clears ``min_score``.
    """
    qualifying: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("error") or row.get("error_note"):
            continue
        score = row.get("ranking_score")
        if score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        dte = _row_dte(row)
        if dte is None or not (dte_min <= dte <= dte_max):
            continue
        if score < min_score:
            continue
        qualifying.append(dict(row))

    qualifying.sort(key=lambda r: float(r.get("ranking_score") or 0.0), reverse=True)
    return qualifying[: max(int(top_n), 0)]


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
        flag = " [regime]" if row.get("iv_regime_conditioned") else ""
        lines.append(
            f"{idx}. {symbol} {dte if dte is not None else '?'}d {timing} "
            f"score {score}{iv_rv_txt}{flag}"
        )
    lines += [
        "",
        f"Setup-quality ranking (>= {min_score:g}), not a win probability.",
        "Research only - not financial advice.",
    ]
    return "\n".join(lines)


def _digest(as_of: date, rows: Sequence[Mapping[str, Any]]) -> str:
    symbols = ",".join(sorted(str(r.get("symbol") or "") for r in rows))
    return f"{as_of.isoformat()}|{symbols}"


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic swap; a torn write can't poison the state file


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ranked earnings screener and alert on qualifying setups.",
    )
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
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
    qualifying = select_qualifying_rows(
        rows,
        min_score=args.min_score,
        top_n=args.top,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
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

    message = format_alert_message(qualifying, as_of=as_of, min_score=args.min_score)
    digest = _digest(as_of, qualifying)
    state = _load_state(args.state_path)

    # Dry run is a preview and must stay observable, so it is checked BEFORE
    # the idempotency digest — otherwise once a real alert has gone out,
    # --dry-run prints "already sent" instead of the message you asked to see.
    if args.dry_run:
        print(message)
        logger.info("Dry run — not sent, state not written.")
        return 0

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

    try:
        result = send_imessage(message, config=config)
    except Exception:
        logger.exception("iMessage send failed.")
        return 1

    _save_state(args.state_path, {
        "last_digest": digest,
        "last_sent_at": datetime.now(timezone.utc).isoformat(),
        "symbols": [str(r.get("symbol")) for r in qualifying],
    })
    logger.info("Alert sent to %s (%s setups).", result.get("to"), len(qualifying))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
