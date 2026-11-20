"""Walk-forward pricing harness: how accurate is the model, priced the way we trade.

Nobody could say how accurate the selector's ``expected_edge_pct`` is: it had
never been compared with what the structures went on to return. This harness
replays the historical end-of-day options archive point in time and does that
comparison, so every later pricing change can be judged against a number.

For each earnings event in the archive:

* **Entry** on every archive session whose calendar distance to the event is
  inside the forward loop's entry window (DTE_MIN..DTE_MAX days). The snapshot
  is built from that session's chain and the price history through it, with
  only the events before it as history, and that day's full chain as the
  entry-equivalent structure chain, so the legs scored are the legs entry
  books (coverage "complete").
* **Predictions** are the scorecards' (expected edge, expected return,
  composite) and the selector's decision, with priors the harness supplies:
  ``walk_forward`` learns each structure's prior only from the harness's own
  outcomes that EXITED BEFORE the entry session (one per event, from its first
  look), shrunk exactly as production shrinks live observations; ``neutral``
  uses the no-data prior. The report files and the persistent prior store are
  never read: they were computed over the whole history.
* **Exit** on the last NYSE session before the event date, re-pricing the same
  contracts (expiry, side, strike) from that session's chain. A contract with
  no two-sided quote there is unpriced and counted, never filled in.
* **Returns** at mid, natural and the claim contract's conservative fill
  (services/execution_scenarios.py); the conservative fill is what the
  predictions are judged against.

What it is not: the archive is end-of-day, so no fill here is a claim under
the claim contract (no live source, no per-leg quote time); and the event
dates come from post-hoc sources unless ``require_lookahead_safe`` is set, so
a rescheduled event appears at its final date. Both are stated in every report.
"""
from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from services import release_timing as _release_timing_rule
from services.earnings_vol_snapshot import COVERAGE_COMPLETE, build_vol_snapshot
from services import daily_vol as _daily_vol
from services.event_variance import (
    EventVarianceForecaster,
    VarianceObservation,
    forecast_quality,
    implied_event_variance,
    sessions_between,
)
from services.execution_scenarios import build_execution_scenarios, compare_execution_scenarios
from services.market_calendar import is_trading_day
from services.structure_legs import STRUCTURES
from services.structure_scorecard import build_structure_scorecards, prior_from_returns
from services.structure_selector import select_best_structure

HARNESS_VERSION = "pricing_harness_v2"

# The forward loop's entry window (services/screener_service.DTE_MIN_DEFAULT /
# DTE_MAX_DEFAULT) and the recommendations it books (scripts/run_forward_loop.
# ALLOWED_RECOMMENDATIONS). Copied so this module does not import the live
# clients those modules pull in; a test keeps them equal.
DTE_MIN = 3
DTE_MAX = 14
TRADED_RECOMMENDATIONS = frozenset({"Best Candidate", "Candidate"})

# Fewer prior closes than this and the realized-vol inputs cannot be formed.
MIN_PRICE_HISTORY = 80

PRIOR_WALK_FORWARD = "walk_forward"
PRIOR_NEUTRAL = "neutral"
PRIOR_MODES = (PRIOR_WALK_FORWARD, PRIOR_NEUTRAL)

RETURN_LEVELS = ("mid", "natural", "conservative_fill")
JUDGED_LEVEL = "conservative_fill"
PREDICTION_FIELDS = ("expected_edge_pct", "expected_return_pct", "composite_structure_score")

CREDIT_STRUCTURES = frozenset({"iron_condor"})


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EarningsEvent:
    symbol: str
    event_date: date
    release_timing: str
    source: str = "unknown"
    source_confidence: float = 0.0
    lookahead_safe: bool = False
    known_as_of: Optional[date] = None


class Archive(Protocol):
    """Point-in-time reads the harness makes; nothing else is consulted."""

    def sessions(self, symbol: str) -> List[date]:
        """Archive trade dates holding option rows for ``symbol``, ascending."""

    def chain(self, symbol: str, day: date) -> pd.DataFrame:
        """Every option row of ``symbol`` quoted on ``day``."""

    def prices(self, symbol: str, through: date) -> pd.DataFrame:
        """Daily OHLC bars of ``symbol`` on or before ``through``."""

    def events(self, symbol: str) -> List[EarningsEvent]:
        """Historical earnings events of ``symbol``."""


_CHAIN_COLUMNS = (
    "trade_date", "expiry", "call_put", "strike", "bid", "ask", "mid", "iv",
    "open_interest", "volume", "underlying_price", "spread_pct", "option_symbol",
)


def _as_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed.date()


class ParquetArchive:
    """The T9 research archive: options and OHLC parquet, earnings SQLite.

    Layout as written by scripts/build_options_feature_layer.py and
    scripts/integrate_historical_replay_inputs.py:
    ``<options_root>/underlying_symbol=SYM/**.parquet``,
    ``<ohlc_root>/underlying_symbol=SYM/**.parquet`` and the
    ``historical_earnings_events`` table.
    """

    def __init__(self, options_root: Path, ohlc_root: Path, earnings_db: Path):
        import duckdb

        self.options_root = Path(options_root)
        self.ohlc_root = Path(ohlc_root)
        self.earnings_db = Path(earnings_db)
        self._con = duckdb.connect(":memory:")
        self._sessions: Dict[str, List[date]] = {}
        self._ohlc: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _safe(symbol: str) -> str:
        sym = str(symbol).strip().upper()
        if not sym or not all(ch.isalnum() or ch in ".-" for ch in sym):
            raise ValueError(f"unsafe symbol {symbol!r}")
        return sym

    def _glob(self, root: Path, symbol: str) -> Optional[str]:
        folder = root / f"underlying_symbol={self._safe(symbol)}"
        return str(folder / "**" / "*.parquet") if folder.exists() else None

    def sessions(self, symbol: str) -> List[date]:
        sym = self._safe(symbol)
        if sym not in self._sessions:
            source = self._glob(self.options_root, sym)
            rows = [] if source is None else self._con.execute(
                "SELECT DISTINCT trade_date FROM read_parquet(?, hive_partitioning=true) ORDER BY trade_date",
                [source],
            ).fetchall()
            self._sessions[sym] = [d for d in (_as_date(r[0]) for r in rows) if d is not None]
        return list(self._sessions[sym])

    def chain(self, symbol: str, day: date) -> pd.DataFrame:
        source = self._glob(self.options_root, symbol)
        if source is None:
            return pd.DataFrame(columns=list(_CHAIN_COLUMNS))
        frame = self._con.execute(
            f"SELECT {', '.join(_CHAIN_COLUMNS)} FROM read_parquet(?, hive_partitioning=true) "
            "WHERE CAST(trade_date AS DATE) = ? ORDER BY expiry, call_put, strike",
            [source, day],
        ).fetchdf()
        return frame

    def prices(self, symbol: str, through: date) -> pd.DataFrame:
        sym = self._safe(symbol)
        if sym not in self._ohlc:
            source = self._glob(self.ohlc_root, sym)
            if source is None:
                self._ohlc[sym] = pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "volume"])
            else:
                frame = self._con.execute(
                    "SELECT trade_date, open, high, low, close, volume "
                    "FROM read_parquet(?, hive_partitioning=true) ORDER BY trade_date",
                    [source],
                ).fetchdf()
                frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
                self._ohlc[sym] = frame.drop_duplicates(["trade_date"], keep="last").reset_index(drop=True)
        frame = self._ohlc[sym]
        return frame[frame["trade_date"] <= pd.Timestamp(through)].copy()

    def events(self, symbol: str) -> List[EarningsEvent]:
        if not self.earnings_db.exists():
            return []
        with sqlite3.connect(f"file:{self.earnings_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT symbol, event_date, release_timing, primary_source, source_confidence, "
                "lookahead_safe_for_pre_event, known_as_of FROM historical_earnings_events "
                "WHERE symbol = ? ORDER BY event_date",
                (self._safe(symbol),),
            ).fetchall()
        out = []
        for row in rows:
            event_date = _as_date(row["event_date"])
            if event_date is None:
                continue
            out.append(EarningsEvent(
                symbol=str(row["symbol"]).upper(),
                event_date=event_date,
                release_timing=_release_timing_rule.to_code(row["release_timing"]),
                source=str(row["primary_source"] or "unknown"),
                source_confidence=float(row["source_confidence"] or 0.0),
                lookahead_safe=bool(row["lookahead_safe_for_pre_event"]),
                known_as_of=_as_date(row["known_as_of"]),
            ))
        return out


# ── calendar rules ───────────────────────────────────────────────────────────

def exit_session(event_date: date) -> date:
    """The last NYSE session strictly before the event date (the T-1 exit)."""
    day = event_date - timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def loop_exit_day(event_date: date) -> date:
    """The forward loop's exit day: calendar earnings_date - 1
    (services/outcome_recorder.trades_due_for_exit). It is not a session
    for a Monday (or post-holiday) event; the report counts how often."""
    return event_date - timedelta(days=1)


def entry_sessions(event_date: date, sessions: Iterable[date]) -> List[date]:
    """Archive sessions inside the loop's entry window, before the exit session."""
    last = exit_session(event_date)
    return sorted(
        s for s in sessions
        if DTE_MIN <= (event_date - s).days <= DTE_MAX and s < last
    )


def event_known_on(event: EarningsEvent, day: date) -> bool:
    """Was the event date verifiably known on ``day``?"""
    if event.known_as_of is not None:
        return event.known_as_of < day
    return bool(event.lookahead_safe)


# ── priors ───────────────────────────────────────────────────────────────────

@dataclass
class OutcomeHistory:
    """Realized conservative-fill returns, each usable only after its exit."""

    rows: List[tuple] = field(default_factory=list)  # (exit_date, structure, return_pct)

    def add(self, exit_date: date, structure: str, return_pct: Optional[float]) -> None:
        if return_pct is not None and math.isfinite(return_pct):
            self.rows.append((exit_date, structure, float(return_pct)))

    def returns_before(self, structure: str, day: date) -> List[float]:
        return [r for (x, s, r) in self.rows if s == structure and x < day]

    def priors(self, day: date, mode: str) -> Dict[str, Any]:
        source = f"harness_{mode}"
        if mode == PRIOR_NEUTRAL:
            return {s: prior_from_returns(s, [], source=source) for s in STRUCTURES}
        if mode == PRIOR_WALK_FORWARD:
            return {s: prior_from_returns(s, self.returns_before(s, day), source=source) for s in STRUCTURES}
        raise ValueError(f"unknown prior mode {mode!r}; expected one of {PRIOR_MODES}")


# ── pricing ──────────────────────────────────────────────────────────────────

def _leg_identity(structure: str, leg: str, trade_expiry: str, back_expiry: Optional[str]) -> tuple:
    expiry = back_expiry if leg == "back" else trade_expiry
    if structure == "call_calendar":
        side = "C"
    elif structure == "put_calendar":
        side = "P"
    else:
        side = "C" if "call" in leg else "P"
    return expiry, side


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def exit_legs(structure: str, entry_legs: Mapping[str, Mapping[str, Any]], exit_chain: pd.DataFrame, *,
              trade_expiry: str, back_expiry: Optional[str]) -> tuple[Optional[Dict[str, Dict[str, Any]]], Optional[str]]:
    """The booked contracts' quotes in the exit chain, or why they cannot be read."""
    if exit_chain is None or exit_chain.empty:
        return None, "exit_chain_missing"
    expiries = pd.to_datetime(exit_chain["expiry"], errors="coerce").dt.date
    sides = exit_chain["call_put"].astype(str).str.upper()
    strikes = pd.to_numeric(exit_chain["strike"], errors="coerce")
    out: Dict[str, Dict[str, Any]] = {}
    for leg, quote in entry_legs.items():
        expiry, side = _leg_identity(structure, leg, trade_expiry, back_expiry)
        strike = _finite(quote.get("strike"))
        if expiry is None or strike is None:
            return None, f"exit_leg_unidentified:{leg}"
        match = exit_chain[(expiries == date.fromisoformat(expiry)) & (sides == side)
                           & ((strikes - strike).abs() < 1e-6)]
        if match.empty:
            return None, f"exit_contract_missing:{leg}"
        if len(match) > 1:
            return None, f"exit_contract_ambiguous:{leg}"
        row = match.iloc[0]
        bid, ask = _finite(row.get("bid")), _finite(row.get("ask"))
        if bid is not None and ask is not None and bid == 0 and ask > 0:
            # Worth little or nothing, but not priceable at a two-sided mid:
            # kept apart so a likely loss is not read as missing data.
            return None, f"exit_leg_zero_bid:{leg}"
        if bid is None or ask is None or bid <= 0 or ask < bid:
            return None, f"exit_leg_unquoted:{leg}"
        out[leg] = {"strike": strike, "bid": bid, "ask": ask}
    return out, None


def condor_capital_at_risk(entry_legs: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    """Wing width less the mid credit: the defined max loss per share."""
    try:
        credit = ((entry_legs["short_call"]["mid"] - entry_legs["long_call"]["mid"])
                  + (entry_legs["short_put"]["mid"] - entry_legs["long_put"]["mid"]))
        width = max(entry_legs["long_call"]["strike"] - entry_legs["short_call"]["strike"],
                    entry_legs["short_put"]["strike"] - entry_legs["long_put"]["strike"])
    except (KeyError, TypeError):
        return None
    risk = _finite(width - credit)
    return risk if risk is not None and risk > 0 else None


def price_round_trip(structure: str, entry_legs: Mapping[str, Mapping[str, Any]],
                     exit_quotes: Mapping[str, Mapping[str, Any]]) -> tuple[Dict[str, Optional[float]], Optional[str]]:
    """Returns (%) at each level for entering on ``entry_legs`` and exiting on ``exit_quotes``."""
    entry_payload = {"bid_ask_mid": {"legs": {k: {"bid": v.get("bid"), "ask": v.get("ask")}
                                              for k, v in entry_legs.items()}}}
    exit_payload = {"bid_ask_mid": {"legs": {k: {"bid": v.get("bid"), "ask": v.get("ask")}
                                             for k, v in exit_quotes.items()}}}
    entry = build_execution_scenarios(structure=structure, quote_payload=entry_payload, phase="entry")
    exit_ = build_execution_scenarios(structure=structure, quote_payload=exit_payload, phase="exit")
    risk = condor_capital_at_risk(entry_legs) if structure in CREDIT_STRUCTURES else None
    if structure in CREDIT_STRUCTURES and risk is None:
        return {level: None for level in RETURN_LEVELS}, "condor_risk_undefined"
    outcome = compare_execution_scenarios(entry=entry.to_dict(), exit=exit_.to_dict(),
                                          structure=structure, capital_at_risk=risk)
    returns = {level: _finite(outcome["realized_return_pct"].get(level)) for level in RETURN_LEVELS}
    if returns[JUDGED_LEVEL] is None:
        return returns, "non_positive_entry_value"
    return returns, None


# ── the replay ───────────────────────────────────────────────────────────────

def _earnings_metadata(event: EarningsEvent, all_events: Sequence[EarningsEvent], as_of: date) -> Dict[str, Any]:
    return {
        "earnings_date": event.event_date.isoformat(),
        "release_timing": event.release_timing,
        # Only events already reported by the entry session are history.
        "prior_events": [
            {"event_date": e.event_date.isoformat(), "release_timing": e.release_timing}
            for e in all_events if e.event_date < as_of
        ],
        "earnings_source_primary": event.source,
        "earnings_source_confidence": float(event.source_confidence),
        "earnings_source_notes": ["pricing harness replay"],
    }


def _structure_chain(chain: pd.DataFrame, as_of: date) -> Dict[str, Any]:
    listing = sorted({d for d in pd.to_datetime(chain["expiry"], errors="coerce").dt.date.dropna() if d > as_of})
    return {"listing": [d.isoformat() for d in listing], "chain": [chain], "fetched_on": as_of.isoformat()}


def _prepare_chain(chain: pd.DataFrame) -> pd.DataFrame:
    frame = chain.copy()
    if "option_symbol" in frame.columns and "contractSymbol" not in frame.columns:
        frame["contractSymbol"] = frame["option_symbol"]
    return frame


@dataclass
class HarnessResult:
    rows: List[Dict[str, Any]]
    counts: Dict[str, int]
    config: Dict[str, Any]
    # One row per scored entry: the event variance at entry and at exit, and
    # the walk-forward forecast of its change (services/event_variance.py).
    variance_rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows, "counts": self.counts, "config": self.config,
                "variance_rows": self.variance_rows}


def _daily_fields(event: EarningsEvent, day: date, exit_day: date, held: int, implied_base: Optional[float],
                  har_rs_annual: Any, prices_entry: pd.DataFrame, prices_exit: pd.DataFrame,
                  past_events: Sequence[Mapping[str, Any]], forecaster: "_daily_vol.DailyVarianceCalibrator",
                  counts: Dict[str, int]) -> Dict[str, Any]:
    """Each forecast of the hold's per-session variance, and what it realized."""
    har_cc = _daily_vol.har_close_to_close(prices_entry, past_events)
    calibrated, correction = None, None
    if har_cc is not None:
        correction = forecaster.calibration(event.symbol, day)
        calibrated = har_cc * correction.factor
    realized, reason = _daily_vol.realized_hold_variance(prices_exit, day, exit_day, held)
    if realized is not None and har_cc is not None:
        forecaster.add(event.symbol, event.event_date, exit_day, realized, har_cc)
    _count(counts, f"daily:{'measured' if reason is None else reason}")
    return {
        "daily_realized_variance": realized,
        "daily_forecast_implied": implied_base if implied_base is not None and implied_base > 0 else None,
        "daily_forecast_har_rs": _daily_vol.per_session(har_rs_annual),
        "daily_forecast_har_cc": har_cc,
        "daily_forecast_calibrated": calibrated,
        "daily_calibration_factor": correction.factor if correction else None,
        "daily_calibration_n_pooled": correction.n_pooled if correction else 0,
        "daily_unmeasured_reason": reason,
    }


def _variance_row(event: EarningsEvent, day: date, exit_day: date, chain: pd.DataFrame,
                  exit_chain: pd.DataFrame, forecaster: EventVarianceForecaster, is_first: bool,
                  counts: Dict[str, int], *, daily: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """E at entry and at exit (same expiry pair), the forecast of its change, and
    (``daily``: the inputs of _daily_fields) the ordinary daily variance of the hold."""
    entry = implied_event_variance(chain, day, event.event_date, event.release_timing)
    exit_ = None
    if entry.usable:
        exit_ = implied_event_variance(exit_chain, exit_day, event.event_date, event.release_timing,
                                       expiries=entry.expiries)
    held = sessions_between(day, exit_day)
    forecast = forecaster.predict(event.symbol, held, day)
    realized = None
    reason = entry.reason if not entry.usable else (exit_.reason if not exit_.usable else None)
    if reason is None:
        realized = math.log(exit_.event_variance / entry.event_variance)
        forecaster.add(VarianceObservation(event.symbol, event.event_date, day, exit_day, held, realized))
    _count(counts, f"variance:{'measured' if reason is None else reason.split(':')[0]}")
    return {
        "symbol": event.symbol,
        "event_date": event.event_date.isoformat(),
        "entry_date": day.isoformat(),
        "exit_date": exit_day.isoformat(),
        "first_look": is_first,
        "sessions_held": held,
        "expiries": [e.isoformat() for e in entry.expiries],
        "entry_event_variance": _finite(entry.event_variance),
        "entry_event_move_pct": _finite(entry.event_move_pct),
        "entry_base_variance_per_session": _finite(entry.base_variance_per_session),
        "exit_expiries": [e.isoformat() for e in exit_.expiries] if exit_ else [],
        "exit_event_variance": _finite(exit_.event_variance) if exit_ else None,
        "exit_event_move_pct": _finite(exit_.event_move_pct) if exit_ else None,
        "realized_log_ratio": realized,
        "forecast_log_ratio": forecast.log_ratio,
        "pooled_log_ratio": forecast.pooled,
        "group_offset": forecast.group_offset,
        "symbol_offset": forecast.symbol_offset,
        "forecast_n_pooled": forecast.n_pooled,
        "forecast_n_symbol_events": forecast.n_symbol_events,
        "unmeasured_reason": reason,
        **(_daily_fields(event, day, exit_day, held, entry.base_variance_per_session, counts=counts, **daily)
           if daily is not None else {}),
    }


def _count(counts: Dict[str, int], key: str, n: int = 1) -> None:
    counts[key] = counts.get(key, 0) + n


def run_harness(
    archive: Archive,
    symbols: Sequence[str],
    *,
    start: date,
    end: date,
    prior_mode: str = PRIOR_WALK_FORWARD,
    require_lookahead_safe: bool = False,
    snapshot_builder: Callable[..., Any] = build_vol_snapshot,
    variance_forecaster: Optional[EventVarianceForecaster] = None,
    daily_calibrator: Optional["_daily_vol.DailyVarianceCalibrator"] = None,
) -> HarnessResult:
    """Replay every event of ``symbols`` dated in [start, end]; one row per
    (event, entry session, structure)."""
    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"unknown prior mode {prior_mode!r}; expected one of {PRIOR_MODES}")
    counts: Dict[str, int] = {}
    plan: List[tuple] = []  # (entry_day, event, all_events, sessions, is_first)
    for symbol in symbols:
        events = archive.events(symbol)
        sessions = archive.sessions(symbol)
        session_set = set(sessions)
        for event in events:
            if not (start <= event.event_date <= end):
                continue
            _count(counts, "events_in_range")
            if loop_exit_day(event.event_date) != exit_session(event.event_date):
                _count(counts, "events_loop_exit_day_not_a_session")
            if exit_session(event.event_date) not in session_set:
                _count(counts, "events_skipped:exit_session_not_in_archive")
                continue
            days = entry_sessions(event.event_date, sessions)
            if require_lookahead_safe:
                days = [d for d in days if event_known_on(event, d)]
            if not days:
                _count(counts, "events_skipped:no_entry_session")
                continue
            _count(counts, "events_replayed")
            for i, day in enumerate(days):
                plan.append((day, event, events, i == 0))
    # Chronological across symbols, so a prior only ever holds earlier exits.
    plan.sort(key=lambda item: (item[0], item[1].symbol, item[1].event_date))

    history = OutcomeHistory()
    forecaster = variance_forecaster if variance_forecaster is not None else EventVarianceForecaster()
    daily_calibrator = daily_calibrator if daily_calibrator is not None else _daily_vol.DailyVarianceCalibrator(
        group_of=forecaster.group_of)
    variance_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    exit_cache: Dict[tuple, pd.DataFrame] = {}
    for day, event, events, is_first in plan:
        _count(counts, "entries_attempted")
        chain = _prepare_chain(archive.chain(event.symbol, day))
        prices = archive.prices(event.symbol, day)
        if chain.empty:
            _count(counts, "entries_skipped:empty_chain")
            continue
        if len(prices) < MIN_PRICE_HISTORY:
            _count(counts, "entries_skipped:short_price_history")
            continue
        try:
            snapshot = snapshot_builder(
                event.symbol, day,
                option_chain_data=chain,
                price_data=prices,
                earnings_metadata=_earnings_metadata(event, events, day),
                structure_chain=_structure_chain(chain, day),
            )
            priors = history.priors(day, prior_mode)
            cards = build_structure_scorecards(snapshot, as_of_date=snapshot.as_of_date, priors=priors)
            selection = select_best_structure(snapshot, cards)
        except Exception as exc:  # noqa: BLE001 - every failure is counted, none hidden
            _count(counts, f"entries_failed:{type(exc).__name__}")
            continue
        legs = snapshot.structure_legs or {}
        coverage = legs.get("coverage")
        if coverage != COVERAGE_COMPLETE:
            _count(counts, f"entries_legs_not_complete:{coverage or 'none'}")
        exit_day = exit_session(event.event_date)
        key = (event.symbol, exit_day)
        if key not in exit_cache:
            exit_cache[key] = archive.chain(event.symbol, exit_day)
        exit_chain = exit_cache[key]
        _count(counts, "entries_scored")
        daily = {
            "har_rs_annual": getattr(snapshot, "rv_har_forecast", None),
            "prices_entry": prices,
            "prices_exit": archive.prices(event.symbol, exit_day),
            "past_events": _earnings_metadata(event, events, day)["prior_events"],
            "forecaster": daily_calibrator,
        }
        variance_rows.append(_variance_row(event, day, exit_day, chain, exit_chain, forecaster, is_first, counts,
                                           daily=daily))
        by_name = {card.structure: card for card in cards}
        for structure in STRUCTURES:
            card = by_name.get(structure)
            found = (legs.get("structures") or {}).get(structure) or {}
            returns: Dict[str, Optional[float]] = {level: None for level in RETURN_LEVELS}
            unpriced = None
            if coverage != COVERAGE_COMPLETE:
                unpriced = "legs_not_complete"
            elif found.get("reason"):
                unpriced = f"unformable:{found['reason']}"
            else:
                quotes, unpriced = exit_legs(structure, found["legs"], exit_chain,
                                             trade_expiry=legs["trade_expiry"], back_expiry=legs.get("back_expiry"))
                if quotes is not None:
                    returns, unpriced = price_round_trip(structure, found["legs"], quotes)
            _count(counts, f"rows:{structure}:{'priced' if unpriced is None else unpriced}")
            if is_first and unpriced is None:
                history.add(exit_day, structure, returns[JUDGED_LEVEL])
            rows.append({
                "symbol": event.symbol,
                "event_date": event.event_date.isoformat(),
                "release_timing": event.release_timing,
                "event_date_known_beforehand": event_known_on(event, day),
                "entry_date": day.isoformat(),
                "exit_date": exit_day.isoformat(),
                "days_to_earnings": (event.event_date - day).days,
                "first_look": is_first,
                "structure": structure,
                "eligible": bool(card.eligible) if card else False,
                "eligibility_flags": list(card.eligibility_flags) if card else [],
                **{f: (_finite(getattr(card, f)) if card else None) for f in PREDICTION_FIELDS},
                "prior_history_count": int(card.walk_forward_history_count) if card else 0,
                "recommendation": selection.recommendation,
                "selected_structure": selection.best_structure,
                "selected": selection.best_structure == structure,
                "traded": selection.best_structure == structure and selection.recommendation in TRADED_RECOMMENDATIONS,
                "data_quality_score": _finite(getattr(snapshot, "data_quality_score", None)),
                "unpriced_reason": unpriced,
                **{f"return_{level}": returns[level] for level in RETURN_LEVELS},
            })
    config = {
        "harness_version": HARNESS_VERSION,
        "symbols": list(symbols),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "prior_mode": prior_mode,
        "require_lookahead_safe": bool(require_lookahead_safe),
        "entry_window_days": [DTE_MIN, DTE_MAX],
        "exit_rule": "last NYSE session before the event date",
        "judged_return": JUDGED_LEVEL,
        "variance_shrinkage_events": forecaster.shrinkage_events,
        "variance_grouping": "by group" if forecaster.group_of else "none",
    }
    return HarnessResult(rows=rows, counts=dict(sorted(counts.items())), config=config,
                         variance_rows=variance_rows)


# ── metrics ──────────────────────────────────────────────────────────────────

def _ranks(values: Sequence[float]) -> np.ndarray:
    return pd.Series(values, dtype=float).rank(method="average").to_numpy()


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Rank correlation with average ranks for ties; None when undefined."""
    if len(x) != len(y):
        raise ValueError("spearman needs equal-length inputs")
    return _pearson(_ranks(x), _ranks(y))


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def calibration_bins(pred: Sequence[float], realized: Sequence[float], bins: int = 5) -> List[Dict[str, Any]]:
    """Mean prediction vs mean realized return by prediction quantile."""
    if len(pred) < bins:
        return []
    order = np.argsort(np.asarray(pred, dtype=float), kind="stable")
    out = []
    for i, chunk in enumerate(np.array_split(order, bins)):
        if len(chunk) == 0:
            continue
        p = np.asarray(pred, dtype=float)[chunk]
        r = np.asarray(realized, dtype=float)[chunk]
        out.append({"bin": i + 1, "n": int(len(chunk)), "mean_predicted": _round(float(p.mean())),
                    "mean_realized": _round(float(r.mean())), "realized_win_rate": _round(float((r > 0).mean()))})
    return out


def prediction_quality(rows: Sequence[Mapping[str, Any]], field_name: str = "expected_edge_pct") -> Dict[str, Any]:
    """How one prediction field matches the realized conservative-fill return."""
    pairs = [(float(r[field_name]), float(r[f"return_{JUDGED_LEVEL}"])) for r in rows
             if r.get(field_name) is not None and r.get(f"return_{JUDGED_LEVEL}") is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    pred = [p for p, _ in pairs]
    real = [r for _, r in pairs]
    signed = [(p, r) for p, r in pairs if p != 0 and r != 0]
    positive = [r for p, r in pairs if p > 0]
    non_positive = [r for p, r in pairs if p <= 0]
    out = {
        "n": n,
        "mean_predicted": _round(float(np.mean(pred))),
        "mean_realized": _round(float(np.mean(real))),
        "realized_win_rate": _round(float(np.mean([r > 0 for r in real]))),
        "spearman": _round(spearman(pred, real)),
        "pearson": _round(_pearson(pred, real)),
        "sign_hit_rate": _round(float(np.mean([(p > 0) == (r > 0) for p, r in signed]))) if signed else None,
        "calibration": calibration_bins(pred, real),
    }
    if field_name != "composite_structure_score":
        out["bias"] = _round(float(np.mean(np.asarray(pred) - np.asarray(real))))
        out["mean_abs_error"] = _round(float(np.mean(np.abs(np.asarray(pred) - np.asarray(real)))))
        out["mean_realized_when_predicted_positive"] = _round(float(np.mean(positive))) if positive else None
        out["mean_realized_when_predicted_non_positive"] = _round(float(np.mean(non_positive))) if non_positive else None
        out["n_predicted_positive"] = len(positive)
    return out


def bootstrap_mean_ci(values: Sequence[float], *, resamples: int = 2000, seed: int = 7,
                      level: float = 0.95) -> Optional[List[float]]:
    """Percentile bootstrap interval of the mean (one value per event)."""
    data = [float(v) for v in values]
    if len(data) < 5:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(data) for _ in data) / len(data) for _ in range(resamples))
    lo = means[int(((1 - level) / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - (1 - level) / 2) * resamples))]
    return [round(lo, 4), round(hi, 4)]


def _event_key(row: Mapping[str, Any]) -> tuple:
    return row["symbol"], row["event_date"]


def policy_comparison(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The selector's trades against an always-straddle baseline, one per event.

    ``selector``: the first entry session whose decision is traded, in the
    structure chosen (what the forward loop books). ``straddle_same_entry``:
    the ATM straddle at that same session, so the pair differs only in the
    choice. ``straddle_first_look``: the ATM straddle at every event's first
    session, traded unconditionally. ``rank_pick``: the structure with the
    highest expected edge at the first look, with no gates.
    """
    judged = f"return_{JUDGED_LEVEL}"
    by_event: Dict[tuple, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(_event_key(row), []).append(row)
    selector, paired, first_straddle, rank_pick = [], [], [], []
    selector_unpriced = 0
    for key, event_rows in sorted(by_event.items()):
        first_day = min(r["entry_date"] for r in event_rows)
        first = [r for r in event_rows if r["entry_date"] == first_day]
        straddle = next((r for r in first if r["structure"] == "atm_straddle"), None)
        if straddle is not None and straddle[judged] is not None:
            first_straddle.append(straddle[judged])
        candidates = [r for r in first if r[judged] is not None and r.get("expected_edge_pct") is not None]
        if candidates:
            best = max(candidates, key=lambda r: (r["expected_edge_pct"], r["structure"]))
            rank_pick.append(best[judged])
        traded = sorted((r for r in event_rows if r["traded"]), key=lambda r: r["entry_date"])
        if not traded:
            continue
        trade = traded[0]
        if trade[judged] is None:
            selector_unpriced += 1
            continue
        selector.append(trade[judged])
        same_day_straddle = next((r for r in event_rows if r["entry_date"] == trade["entry_date"]
                                  and r["structure"] == "atm_straddle"), None)
        if same_day_straddle is not None and same_day_straddle[judged] is not None:
            paired.append((trade[judged], same_day_straddle[judged]))

    def _summary(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"n_events": 0}
        return {"n_events": len(values), "mean_return": _round(float(np.mean(values))),
                "median_return": _round(float(np.median(values))),
                "win_rate": _round(float(np.mean([v > 0 for v in values]))),
                "mean_ci95": bootstrap_mean_ci(values)}

    diffs = [a - b for a, b in paired]
    return {
        "events": len(by_event),
        "selector": {**_summary(selector), "traded_but_unpriced": selector_unpriced},
        "straddle_same_entry": _summary([b for _, b in paired]),
        "selector_minus_straddle_same_entry": _summary(diffs),
        "straddle_first_look": _summary(first_straddle),
        "rank_pick_first_look": _summary(rank_pick),
    }


def summarize(result: HarnessResult) -> Dict[str, Any]:
    """The report: prediction quality per structure, policies, stability by year."""
    rows = result.rows
    first = [r for r in rows if r["first_look"]]
    per_structure: Dict[str, Any] = {}
    for structure in STRUCTURES:
        s_all = [r for r in rows if r["structure"] == structure]
        s_first = [r for r in first if r["structure"] == structure]
        priced = [r for r in s_all if r["unpriced_reason"] is None]
        per_structure[structure] = {
            "rows": len(s_all),
            "priced_rows": len(priced),
            "priced_share": _round(len(priced) / len(s_all)) if s_all else None,
            "first_look": {f: prediction_quality(s_first, f) for f in PREDICTION_FIELDS},
            "all_entries": {f: prediction_quality(s_all, f) for f in PREDICTION_FIELDS},
            "eligible_first_look": prediction_quality([r for r in s_first if r["eligible"]]),
            "by_year_first_look": {
                year: prediction_quality([r for r in s_first if r["event_date"][:4] == year])
                for year in sorted({r["event_date"][:4] for r in s_first})
            },
        }
    known = sum(1 for r in first if r["event_date_known_beforehand"])
    return {
        "config": result.config,
        "counts": result.counts,
        "event_dates": {
            "first_look_rows": len(first),
            "known_beforehand_rows": known,
            "posthoc_rows": len(first) - known,
        },
        "per_structure": per_structure,
        "policies": policy_comparison(rows),
        "event_variance": {
            "measured_entries": sum(1 for r in result.variance_rows if r["realized_log_ratio"] is not None),
            "entries": len(result.variance_rows),
            "first_look": forecast_quality([r for r in result.variance_rows if r["first_look"]]),
            "all_entries": forecast_quality(result.variance_rows),
        },
        "daily_variance": {
            "first_look": _daily_vol.forecast_quality([r for r in result.variance_rows if r["first_look"]]),
            "all_entries": _daily_vol.forecast_quality(result.variance_rows),
        },
        "caveats": [
            "End-of-day archive quotes: no fill here is a claim under the claim contract "
            "(no live source and no per-leg quote time).",
            "Event dates come from post-hoc sources unless known beforehand is shown; a rescheduled "
            "event appears at its final date.",
            "Entries within one event are correlated; first-look metrics and the per-event policies "
            "count each event once.",
            "The scorecards' hand-set constants were not fitted here, but may have been tuned on "
            "this archive's period, so even walk-forward numbers are not fully out of sample.",
        ],
    }


__all__ = [
    "Archive", "DTE_MAX", "DTE_MIN", "EarningsEvent", "HARNESS_VERSION", "HarnessResult", "OutcomeHistory",
    "PRIOR_MODES", "PRIOR_NEUTRAL", "PRIOR_WALK_FORWARD", "ParquetArchive", "TRADED_RECOMMENDATIONS",
    "bootstrap_mean_ci", "calibration_bins", "condor_capital_at_risk", "entry_sessions", "event_known_on",
    "exit_legs", "exit_session", "loop_exit_day", "policy_comparison", "prediction_quality",
    "price_round_trip", "run_harness", "spearman", "summarize",
]
