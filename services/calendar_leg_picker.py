"""
calendar_leg_picker
===================
Pure leg-selection for call/put calendar spreads.

Provides two picker variants for dual-path shadow comparison:

  PICKER_LEGACY ("legacy_first_expiry"):
      front_expiry = first expiry strictly after event_date.
      Matches the pre-PR-AC edge_engine behavior.

  PICKER_CANDIDATE ("candidate_min_dte"):
      front_expiry = first expiry where
      (front_expiry - event_date).days >= CANDIDATE_FRONT_MIN_DTE_DAYS.
      Derived from the empirical replay sweep (PR-AB): on a 10-symbol x
      139-event in-sample run, requiring a front >=14 days past the event
      improved put_calendar mean return from +12% to +23%. NOT yet
      out-of-sample validated. Surfaced under `experimental_contract_selection`
      until promotion criteria are met — the explicit thresholds are
      forthcoming in PR-AC commit 5; until then, see the PR-AC pull
      request description.

Both pickers produce a CalendarSelection that includes the picker variant
and parameters used, so every recommendation is auditable.

This module is intentionally engine-free: no imports from web/, no
singleton stores, no global state. Operates on a pandas DataFrame with
the conventional chain schema:

    required columns:  expiry (date or pd.Timestamp), strike (float),
                       call_put (str "C" or "P"), mid (float)
    optional but used: underlying_price, spread_pct, open_interest,
                       volume, iv, bid, ask

Caller's responsibility — quote sanity
--------------------------------------
This module filters `mid > 0` but does NOT enforce bid/ask sanity (no
crossed-market check, no spread-pct ceiling, no surface-quality gate).
Callers MUST pass chains that have already been through their normal
chain-normalization + surface-quality pipeline. If you feed a raw chain
that contains crossed quotes, garbage mids, or wide-spread tail strikes,
the picker can return a CalendarSelection for legs that are not actually
executable. The integration adapter (commit 2 in PR-AC) is the right
place to enforce those gates.

Column-name compatibility
-------------------------
The required columns use the canonical names (expiry, strike, call_put,
mid). Some providers use aliases (e.g. "expiration_date", "type"). The
integration adapter must normalize column names before calling here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# Public constants
# ──────────────────────────────────────────────────────────────────────────

PICKER_LEGACY = "legacy_first_expiry"
PICKER_CANDIDATE = "candidate_min_dte"
SUPPORTED_PICKERS = (PICKER_LEGACY, PICKER_CANDIDATE)

# Research provenance: discovered in PR-AB by sweeping holding periods
# across 10 symbols × 3 horizons × 4 structures. SAME dataset that
# produced the finding — out-of-sample validation pending.
#
# Do NOT silently lower this constant without:
#   1. Re-running the replay sweep (scripts/replay_holding_period_sweep.py)
#   2. Adding a holdout split to validate
#   3. Updating the promotion criterion in PR-AC's commit message
CANDIDATE_FRONT_MIN_DTE_DAYS = 14
DEFAULT_BACK_GAP_DAYS = 14

SIDE_CALL = "call"
SIDE_PUT = "put"
SUPPORTED_SIDES = (SIDE_CALL, SIDE_PUT)
_SIDE_TO_CHAIN_CODE = {SIDE_CALL: "C", SIDE_PUT: "P"}

_REQUIRED_COLUMNS = ("expiry", "strike", "call_put", "mid")

# Canonical leg field set returned by CalendarSelection.to_metadata_dict().
# Every key is always present in the serialized leg dict; values are None
# when the source chain row didn't carry the column. Exposed at module
# level so downstream consumers (ledger schema, frontend types) can
# import the authoritative key list rather than duplicating it.
LEG_FIELDS = (
    "mid", "bid", "ask", "iv", "spread_pct",
    "open_interest", "volume", "delta", "dte",
)


# ──────────────────────────────────────────────────────────────────────────
# Public data class
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalendarSelection:
    """One front/back calendar leg pair, with picker metadata.

    The metadata fields exist so that observations recorded into the
    outcome ledger remain attributable to the specific picker rule that
    generated them — required for valid dual-path comparison."""
    side: str               # "call" or "put"
    strike: float
    front_expiry: date
    back_expiry: date
    front_row: pd.Series
    back_row: pd.Series

    # Picker provenance — reproduce-the-choice metadata.
    picker_variant: str
    picker_min_front_dte_days: int
    picker_back_gap_days: int
    front_dte_days: int          # actual (front_expiry - event_date).days
    back_minus_front_days: int   # actual (back_expiry - front_expiry).days

    def to_metadata_dict(self) -> dict:
        """Serialize to a JSON-safe dict for ledger / API surfaces.

        Drops the raw pd.Series leg rows and extracts only stable scalar
        fields: expiries, strike, side, plus all picker provenance, plus
        a leg dict with a stable canonical shape (every LEG_FIELDS key is
        present, value is None when the underlying row didn't carry it).

        Use this rather than dataclasses.asdict() — the latter would try
        to serialize pd.Series.
        """
        def _row(r):
            if r is None:
                return None
            out = {}
            for k in LEG_FIELDS:
                if k in r.index:
                    v = r.get(k)
                    out[k] = (
                        float(v) if v is not None and not pd.isna(v) else None
                    )
                else:
                    # Always present in the dict, even when the source row
                    # didn't carry the column. Stable shape simplifies
                    # downstream consumers (ledger, frontend diagnostics).
                    out[k] = None
            return out

        return {
            "side": self.side,
            "strike": float(self.strike),
            "front_expiry": self.front_expiry.isoformat(),
            "back_expiry": self.back_expiry.isoformat(),
            "front_dte_days": int(self.front_dte_days),
            "back_minus_front_days": int(self.back_minus_front_days),
            "picker_variant": self.picker_variant,
            "picker_min_front_dte_days": int(self.picker_min_front_dte_days),
            "picker_back_gap_days": int(self.picker_back_gap_days),
            "front_leg": _row(self.front_row),
            "back_leg": _row(self.back_row),
        }


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def select_calendar_contracts(
    chain_df: pd.DataFrame,
    *,
    event_date,
    side: str,
    picker_variant: str,
    min_front_dte_days: int = CANDIDATE_FRONT_MIN_DTE_DAYS,
    back_gap_days: int = DEFAULT_BACK_GAP_DAYS,
) -> Optional[CalendarSelection]:
    """
    Pick a (front, back) calendar leg pair from a chain.

    Parameters
    ----------
    chain_df
        Option chain DataFrame. Must contain expiry, strike, call_put, mid.
        Other columns (underlying_price, spread_pct, open_interest, volume,
        iv, bid, ask) are optional but used for tie-breaking and downstream
        pricing.
    event_date
        The reference date — typically the earnings event date. The legacy
        picker requires front_expiry > event_date. The candidate picker
        requires (front_expiry - event_date).days >= min_front_dte_days.
    side
        ``SIDE_CALL`` or ``SIDE_PUT``.
    picker_variant
        ``PICKER_LEGACY`` or ``PICKER_CANDIDATE``.
    min_front_dte_days
        Only used by the candidate picker. Floor on
        (front_expiry - event_date).days.
    back_gap_days
        Minimum (back_expiry - front_expiry).days. If no back expiry meets
        this, falls back to the first expiry strictly past front_expiry.

    Returns
    -------
    CalendarSelection or None if the chain has no valid leg pair under the
    chosen picker.
    """
    if side not in SUPPORTED_SIDES:
        raise ValueError(f"side must be one of {SUPPORTED_SIDES}, got {side!r}")
    if picker_variant not in SUPPORTED_PICKERS:
        raise ValueError(
            f"picker_variant must be one of {SUPPORTED_PICKERS}, got {picker_variant!r}"
        )
    if min_front_dte_days < 0:
        raise ValueError(f"min_front_dte_days must be >= 0, got {min_front_dte_days}")
    if back_gap_days < 0:
        raise ValueError(f"back_gap_days must be >= 0, got {back_gap_days}")

    ref_date = _to_date(event_date)
    if ref_date is None:
        return None

    chain_code = _SIDE_TO_CHAIN_CODE[side]
    df = _prepare_chain(chain_df, side_code=chain_code)
    if df is None or df.empty:
        return None

    expiries = sorted({_to_date(v) for v in df["expiry"].dropna().tolist() if _to_date(v) is not None})
    if not expiries:
        return None

    # Front expiry — variant-dependent
    if picker_variant == PICKER_LEGACY:
        eligible_front = [e for e in expiries if e > ref_date]
    else:  # PICKER_CANDIDATE
        eligible_front = [e for e in expiries if (e - ref_date).days >= min_front_dte_days]
    if not eligible_front:
        return None
    front_expiry = eligible_front[0]

    # Back expiry — prefer ≥back_gap_days past front, else any later expiry
    eligible_back = [e for e in expiries if (e - front_expiry).days >= back_gap_days]
    if not eligible_back:
        eligible_back = [e for e in expiries if e > front_expiry]
    if not eligible_back:
        return None
    back_expiry = eligible_back[0]

    front_df = df[df["expiry"].apply(_to_date) == front_expiry]
    back_df = df[df["expiry"].apply(_to_date) == back_expiry]
    common_strikes = sorted(
        set(front_df["strike"].dropna().tolist()) & set(back_df["strike"].dropna().tolist())
    )
    if not common_strikes:
        return None

    underlying = _underlying_estimate(df, common_strikes)
    chosen_strike = min(common_strikes, key=lambda s: abs(float(s) - underlying))

    front_row = _best_row(front_df[front_df["strike"] == chosen_strike])
    back_row = _best_row(back_df[back_df["strike"] == chosen_strike])
    if front_row is None or back_row is None:
        return None

    return CalendarSelection(
        side=side,
        strike=float(chosen_strike),
        front_expiry=front_expiry,
        back_expiry=back_expiry,
        front_row=front_row,
        back_row=back_row,
        picker_variant=picker_variant,
        picker_min_front_dte_days=int(min_front_dte_days),
        picker_back_gap_days=int(back_gap_days),
        front_dte_days=int((front_expiry - ref_date).days),
        back_minus_front_days=int((back_expiry - front_expiry).days),
    )


# ──────────────────────────────────────────────────────────────────────────
# Helpers (private)
# ──────────────────────────────────────────────────────────────────────────

def _prepare_chain(chain_df: pd.DataFrame, *, side_code: str) -> Optional[pd.DataFrame]:
    """Validate + filter chain to one side. Returns None if unusable."""
    if chain_df is None or len(chain_df) == 0:
        return None
    missing = [c for c in _REQUIRED_COLUMNS if c not in chain_df.columns]
    if missing:
        return None
    df = chain_df.copy()
    df["call_put"] = df["call_put"].astype(str).str.upper().str.strip()
    df = df[df["call_put"] == side_code]
    if df.empty:
        return None
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df = df.dropna(subset=["expiry", "strike", "mid"])
    df = df[df["mid"] > 0]
    return df if not df.empty else None


def _to_date(value) -> Optional[date]:
    """Coerce many time-like things to date. Returns None if not coercible."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.date()
    except (ValueError, TypeError):
        return None


def _underlying_estimate(df: pd.DataFrame, common_strikes: list) -> float:
    if "underlying_price" in df.columns:
        s = pd.to_numeric(df["underlying_price"], errors="coerce").dropna()
        if not s.empty:
            return float(s.median())
    return float(np.median(common_strikes))


def _best_row(subset: pd.DataFrame) -> Optional[pd.Series]:
    """When multiple rows share (expiry, strike, side), pick the most liquid."""
    if subset is None or subset.empty:
        return None
    if len(subset) == 1:
        return subset.iloc[0]
    s = subset.copy()

    # Build liquidity proxy. We can't use `s.get("col", 0)` here: when the
    # column is absent, `.get(default)` returns the scalar default, and
    # `pd.to_numeric(scalar).fillna(...)` raises AttributeError because
    # the scalar has no .fillna. Use an index-aligned zero Series instead.
    zero = pd.Series(0.0, index=s.index)
    inf_series = pd.Series(np.inf, index=s.index)
    oi = (
        pd.to_numeric(s["open_interest"], errors="coerce").fillna(0.0)
        if "open_interest" in s.columns else zero
    )
    vol = (
        pd.to_numeric(s["volume"], errors="coerce").fillna(0.0)
        if "volume" in s.columns else zero
    )
    spread = (
        pd.to_numeric(s["spread_pct"], errors="coerce").fillna(np.inf)
        if "spread_pct" in s.columns else inf_series
    )
    s["_liq"] = oi + vol
    s["_spread"] = spread
    s = s.sort_values(["_liq", "_spread"], ascending=[False, True])
    return s.iloc[0]
