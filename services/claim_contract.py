"""Claimability contract (F5): when a forward outcome may be claimed as evidence.

Forward quotes are option-chain snapshots, not broker fills, so a claim is
priced at a conservative modeled fill rather than at the mid. This module is
the frozen contract that decides when that modeled fill is credible enough to
claim; docs/CLAIM_CONTRACT.md is its written form. Changing a constant here
changes what "claimable" means, so it must come with a new CONTRACT_VERSION.

A quote passes (entry or exit) when every one of these holds:

1. Source: a live, quote-level feed (MarketData.app). yfinance chains are
   delayed and indicative and never pass.
2. Contract identity: each leg names a standard OCC option symbol. An adjusted
   contract (after a split or special dividend its root gains a digit, e.g.
   AAPL1) does not pass, so corporate actions cannot slip into a claim.
3. Bid/ask completeness: each leg has a finite bid >= MIN_LEG_BID, an ask at
   or above the bid, and a spread within MAX_LEG_SPREAD_PCT of its mid.
4. Synchronized, point-in-time quotes: each leg carries its own quote time,
   inside the regular session of the quote's session day (13:00 on NYSE
   early-close days); no leg is newer than the capture time (beyond
   MAX_CLOCK_SKEW_SECONDS), so nothing unknown at capture can enter; the legs
   are within MAX_LEG_SKEW_SECONDS of each other; and the oldest leg is within
   MAX_QUOTE_AGE_SECONDS of the reference time (the capture time, or the
   session close when captured after it).

A resolved trade is claimable when its entry and exit quotes both pass under
this CONTRACT_VERSION and the exit re-priced exactly the contracts booked at
entry. A missing, malformed or other-version verdict fails closed. Its claimed return
uses the conservative fill: the natural price (pay the ask, receive the bid)
moved a further SLIPPAGE per leg, with commission and regulatory fees for
every contract traded on both sides.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from services.market_calendar import EXCHANGE_TZ, SESSION_OPEN, is_trading_day, quote_session, session_close

# v2: future-dated legs, early closes, impossible contract dates and missing or
# other-version exit verdicts all fail (v1 let each of them through).
CONTRACT_VERSION = "conservative_fill_v2"

CLAIM_SOURCES = frozenset({"marketdata_app"})
MIN_LEG_BID = 0.05
MAX_LEG_SPREAD_PCT = 25.0
MAX_LEG_SKEW_SECONDS = 60.0
MAX_QUOTE_AGE_SECONDS = 300.0
# A leg may carry a quote time at most this far past the capture time
# (provider and local clocks differ slightly); anything newer is look-ahead.
MAX_CLOCK_SKEW_SECONDS = 5.0
# Quotes stamped a few minutes after the bell still belong to the session.
SESSION_CLOSE_GRACE = timedelta(minutes=15)

# Conservative fill, per leg per share (one contract = CONTRACT_MULTIPLIER shares).
CONTRACT_MULTIPLIER = 100
SLIPPAGE_FRACTION_OF_SPREAD = 0.10
MIN_SLIPPAGE_PER_SHARE = 0.01
COMMISSION_PER_CONTRACT = 0.65
REGULATORY_FEE_PER_CONTRACT = 0.02

# Evidence-quality reasons that disqualify a claim even when they only
# degrade the record: they put the event itself in doubt, not the model.
CLAIM_DISQUALIFYING_REASONS = frozenset({
    "missing_quote_source",
    "provider_research_grade_yfinance",
    "stale_earnings_source",
    "low_earnings_source_confidence",
})

_OCC_SYMBOL = re.compile(r"^[A-Z]{1,6}(\d{2})(\d{2})(\d{2})[CP](\d{8})$")


@dataclass(frozen=True)
class ContractCheck:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    leg_contracts: dict[str, str] = field(default_factory=dict)
    max_leg_skew_seconds: Optional[float] = None
    quote_age_seconds: Optional[float] = None
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fee_per_share() -> float:
    """Commission plus regulatory fee for one contract, per share."""
    return (COMMISSION_PER_CONTRACT + REGULATORY_FEE_PER_CONTRACT) / CONTRACT_MULTIPLIER


def slippage_per_share(bid: float, ask: float) -> float:
    return max(MIN_SLIPPAGE_PER_SHARE, SLIPPAGE_FRACTION_OF_SPREAD * max(0.0, ask - bid))


def evaluate_quote_contract(quote_payload: Mapping[str, Any] | None) -> ContractCheck:
    """Does this structure quote meet the contract (either phase)?"""
    quote = quote_payload or {}
    reasons: list[str] = []
    source = str(quote.get("quote_source") or "").lower()
    if source not in CLAIM_SOURCES:
        reasons.append(f"claim_source_not_live:{source or 'missing'}")

    bid_ask_mid = quote.get("bid_ask_mid") if isinstance(quote.get("bid_ask_mid"), Mapping) else {}
    legs = bid_ask_mid.get("legs") if isinstance(bid_ask_mid, Mapping) else None
    if not isinstance(legs, Mapping) or not legs:
        return ContractCheck(passed=False, reasons=sorted(set(reasons + ["claim_missing_legs"])))

    contracts: dict[str, str] = {}
    leg_times: list[datetime] = []
    for name, raw in legs.items():
        leg = raw if isinstance(raw, Mapping) else {}
        prefix = f"claim_{name}_"
        contract = str(leg.get("contract") or "").strip().upper()
        if not contract:
            reasons.append(prefix + "missing_contract")
        elif not _valid_occ_symbol(contract):
            reasons.append(prefix + "non_standard_contract")
        else:
            contracts[str(name)] = contract
        bid, ask = _finite(leg.get("bid")), _finite(leg.get("ask"))
        if bid is None or ask is None:
            reasons.append(prefix + "missing_bid_ask")
        elif bid < MIN_LEG_BID:
            reasons.append(prefix + "bid_below_minimum")
        elif ask < bid:
            reasons.append(prefix + "crossed_quote")
        else:
            mid = (bid + ask) / 2.0
            if (ask - bid) / mid * 100.0 > MAX_LEG_SPREAD_PCT:
                reasons.append(prefix + "spread_too_wide")
        stamp = _parse_time(leg.get("quote_time"))
        if stamp is None:
            reasons.append(prefix + "missing_quote_time")
        else:
            if not _in_session(stamp):
                reasons.append(prefix + "quote_outside_session")
            leg_times.append(stamp)

    skew = age = None
    if leg_times:
        skew = (max(leg_times) - min(leg_times)).total_seconds()
        if skew > MAX_LEG_SKEW_SECONDS:
            reasons.append("claim_legs_not_synchronized")
        captured = _parse_time(quote.get("quote_timestamp"))
        if captured is None:
            reasons.append("claim_missing_capture_time")
        else:
            if (max(leg_times) - captured).total_seconds() > MAX_CLOCK_SKEW_SECONDS:
                reasons.append("claim_quote_after_capture")
            session = quote_session(captured)
            close = datetime.combine(session, session_close(session), tzinfo=EXCHANGE_TZ)
            reference = min(captured, close)
            age = (reference - min(leg_times)).total_seconds()
            if age > MAX_QUOTE_AGE_SECONDS:
                reasons.append("claim_quote_stale")
            if any(stamp.astimezone(EXCHANGE_TZ).date() != session for stamp in leg_times):
                reasons.append("claim_quote_not_from_session")
    return ContractCheck(
        passed=not reasons,
        reasons=sorted(set(reasons)),
        leg_contracts=contracts,
        max_leg_skew_seconds=None if skew is None else round(skew, 3),
        quote_age_seconds=None if age is None else round(age, 3),
    )


def _in_session(stamp: datetime) -> bool:
    local = stamp.astimezone(EXCHANGE_TZ)
    day = local.date()
    if not is_trading_day(day):
        return False
    opens = datetime.combine(day, SESSION_OPEN, tzinfo=EXCHANGE_TZ)
    closes = datetime.combine(day, session_close(day), tzinfo=EXCHANGE_TZ) + SESSION_CLOSE_GRACE
    return opens <= local <= closes


def _valid_occ_symbol(contract: str) -> bool:
    """A standard OCC symbol with a real expiration date and a positive strike."""
    match = _OCC_SYMBOL.match(contract)
    if match is None:
        return False
    year, month, day, strike = match.groups()
    try:
        date(2000 + int(year), int(month), int(day))
    except ValueError:
        return False
    return int(strike) > 0


def evaluate_exit_claim(
    *, entry_quote_legs: Mapping[str, Any] | None, exit_quote_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The exit side of a claim: the exit quote passes and re-priced the booked contracts."""
    exit_check = evaluate_quote_contract(exit_quote_payload)
    reasons = [f"exit_{reason}" for reason in exit_check.reasons]
    entry_contracts = {
        str(name): str((leg or {}).get("contract") or "").strip().upper()
        for name, leg in (entry_quote_legs or {}).items()
        if isinstance(leg, Mapping)
    }
    if not entry_contracts or entry_contracts != exit_check.leg_contracts:
        reasons.append("claim_exit_contracts_differ_from_entry")
    return {
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "contract_version": CONTRACT_VERSION,
        "exit_quote_age_seconds": exit_check.quote_age_seconds,
        "exit_max_leg_skew_seconds": exit_check.max_leg_skew_seconds,
    }


def apply_exit_claim(
    *, claim_allowed: Any, evidence_reasons: list[str], exit_execution_scenarios: Mapping[str, Any] | None,
    entry_contract_version: Optional[str] = None,
) -> tuple[Optional[bool], list[str]]:
    """Fold the exit claim verdict into a row's entry-time claim fields.

    Fails closed: an entry that was claim-eligible stays claimable only if the
    exit verdict is present, well-formed, passed, and was made under this
    CONTRACT_VERSION, as was the entry. Anything else (a legacy caller with no
    verdict, a malformed one, an entry or exit checked under another version)
    makes the row non-claimable and says why. A row that was never
    claim-eligible is left as it was.
    """
    reasons = list(evidence_reasons)
    if claim_allowed is None or not bool(claim_allowed):
        return (None if claim_allowed is None else False), reasons
    verdict = (exit_execution_scenarios or {}).get("claim") if isinstance(exit_execution_scenarios, Mapping) else None
    if not isinstance(verdict, Mapping) or not isinstance(verdict.get("passed"), bool):
        return False, sorted(set(reasons + ["claim_exit_verdict_missing"]))
    reasons = sorted(set(reasons + [str(r) for r in verdict.get("reasons") or []]))
    if entry_contract_version != CONTRACT_VERSION or verdict.get("contract_version") != CONTRACT_VERSION:
        return False, sorted(set(reasons + ["claim_contract_version_mismatch"]))
    return verdict["passed"], reasons


def claimable_under_current_contract(row: Mapping[str, Any]) -> Optional[bool]:
    """A stored row's claim status for reporting: claimable only under this
    CONTRACT_VERSION, so claims made under different rules are never pooled.
    None when the row never had a claim status."""
    raw = row.get("claim_allowed")
    if raw is None:
        return None
    allowed = str(raw).strip().lower() in {"1", "true", "yes"} if isinstance(raw, str) else bool(raw)
    return allowed and row.get("claim_contract_version") == CONTRACT_VERSION


def _finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_time(value: Any) -> Optional[datetime]:
    """A timezone-aware time from an ISO string or unix seconds, else None."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc) if math.isfinite(float(value)) else None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


__all__ = [
    "CLAIM_DISQUALIFYING_REASONS", "CONTRACT_VERSION", "ContractCheck", "apply_exit_claim",
    "claimable_under_current_contract",
    "evaluate_exit_claim", "evaluate_quote_contract", "fee_per_share", "slippage_per_share",
]
