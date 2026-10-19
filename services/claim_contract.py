"""Claimability contract (F5): when a forward outcome may be claimed as evidence.

Forward quotes are option-chain snapshots, not broker fills, so a claim is
priced at a conservative modeled fill rather than at the mid. This module is
the frozen contract that decides when that modeled fill is credible enough to
claim; docs/CLAIM_CONTRACT.md is its written form. Changing a constant here
changes what "claimable" means, so it must come with a new CONTRACT_VERSION.

A quote passes (entry or exit) when every one of these holds:

1. Source: a live, quote-level feed (MarketData.app). yfinance chains are
   delayed and indicative and never pass.
2. Contract identity: each leg names a standard OCC option symbol on the
   recommendation's own underlying, with a real expiration that had not passed
   on the quote's session day. An adjusted contract (after a split or special
   dividend its root gains a digit, e.g. AAPL1) does not pass, so corporate
   actions cannot slip into a claim; neither can another symbol's leg or an
   expired contract still listed in a stale chain.
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
entry. Each side's verdict is stored with the trade and must be well formed:
``passed`` a bool, ``reasons`` a list of strings, and ``passed`` true exactly
when there are no reasons. A missing, malformed, inconsistent or other-version
verdict (entry or exit) fails closed, and the stores derive claimability from
the verdicts rather than trusting a caller's flag. Its claimed return
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
# v3: legs must be on the recommendation's underlying and unexpired at the
# quote's session; entry and exit verdicts are typed, consistent and mandatory.
CONTRACT_VERSION = "conservative_fill_v3"

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

_OCC_SYMBOL = re.compile(r"^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})[CP](\d{8})$")


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


def occ_root(symbol: Any) -> str:
    """The OCC root a listed underlying trades under (BRK.B -> BRKB)."""
    return re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())


def evaluate_quote_contract(quote_payload: Mapping[str, Any] | None, *,
                            expected_underlying: Optional[str] = None) -> ContractCheck:
    """Does this structure quote meet the contract (either phase)?

    ``expected_underlying`` is the recommendation's symbol (else the quote's
    own ``symbol``); without one the legs cannot be bound to it and the quote
    fails.
    """
    quote = quote_payload or {}
    reasons: list[str] = []
    underlying = occ_root(expected_underlying or quote.get("symbol"))
    if not underlying:
        reasons.append("claim_missing_underlying")
    source = str(quote.get("quote_source") or "").lower()
    if source not in CLAIM_SOURCES:
        reasons.append(f"claim_source_not_live:{source or 'missing'}")

    bid_ask_mid = quote.get("bid_ask_mid") if isinstance(quote.get("bid_ask_mid"), Mapping) else {}
    legs = bid_ask_mid.get("legs") if isinstance(bid_ask_mid, Mapping) else None
    if not isinstance(legs, Mapping) or not legs:
        return ContractCheck(passed=False, reasons=sorted(set(reasons + ["claim_missing_legs"])))

    contracts: dict[str, str] = {}
    expirations: dict[str, date] = {}
    leg_times: list[datetime] = []
    for name, raw in legs.items():
        leg = raw if isinstance(raw, Mapping) else {}
        prefix = f"claim_{name}_"
        contract = str(leg.get("contract") or "").strip().upper()
        parsed = _parse_occ_symbol(contract) if contract else None
        if not contract:
            reasons.append(prefix + "missing_contract")
        elif parsed is None:
            reasons.append(prefix + "non_standard_contract")
        else:
            root, expiration = parsed
            if underlying and root != underlying:
                reasons.append(prefix + "contract_not_on_underlying")
            expirations[str(name)] = expiration
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
            for name, expiration in expirations.items():
                if expiration < session:
                    reasons.append(f"claim_{name}_contract_expired")
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


def _parse_occ_symbol(contract: str) -> Optional[tuple[str, date]]:
    """(root, expiration) of a standard OCC symbol with a real expiration date
    and a positive strike, else None."""
    match = _OCC_SYMBOL.match(contract)
    if match is None:
        return None
    root, year, month, day, strike = match.groups()
    try:
        expiration = date(2000 + int(year), int(month), int(day))
    except ValueError:
        return None
    return (root, expiration) if int(strike) > 0 else None


def contract_verdict(check: ContractCheck) -> dict[str, Any]:
    """The stored form of one side's verdict (see ``verdict_problems``)."""
    return {
        "passed": bool(check.passed) and not check.reasons,
        "reasons": list(check.reasons),
        "contract_version": check.contract_version,
        "leg_contracts": dict(check.leg_contracts),
    }


def verdict_problems(verdict: Any, *, side: str) -> list[str]:
    """Why a stored verdict cannot be relied on; empty when it can.

    Well formed means ``passed`` is a bool, ``reasons`` a list of strings,
    ``passed`` is true exactly when ``reasons`` is empty, and the verdict was
    made under this CONTRACT_VERSION.
    """
    if not isinstance(verdict, Mapping):
        return [f"claim_{side}_verdict_missing"]
    passed, reasons = verdict.get("passed"), verdict.get("reasons")
    if not isinstance(passed, bool) or not isinstance(reasons, list) or not all(
            isinstance(reason, str) for reason in reasons):
        return [f"claim_{side}_verdict_malformed"]
    problems = []
    if passed == bool(reasons):
        problems.append(f"claim_{side}_verdict_inconsistent")
    if verdict.get("contract_version") != CONTRACT_VERSION:
        problems.append("claim_contract_version_mismatch")
    return problems


def _verdict_reasons(verdict: Any) -> list[str]:
    reasons = verdict.get("reasons") if isinstance(verdict, Mapping) else None
    return [reason for reason in reasons if isinstance(reason, str)] if isinstance(reasons, list) else []


def _claim_flag(value: Any) -> Optional[bool]:
    """Normalize only explicit, historically accepted boolean representations."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return False


def entry_claim_status(
    *, claim_allowed: Any, claim_contract_version: Any, entry_execution_scenarios: Any,
) -> tuple[Optional[bool], list[str]]:
    """The claim flag a store may record at entry, and why it was withdrawn.

    A caller's ``claim_allowed=True`` stands only with a well-formed, passing
    entry verdict (``entry_execution_scenarios["claim"]``) made under this
    CONTRACT_VERSION, recorded under that same version.
    """
    normalized = _claim_flag(claim_allowed)
    if normalized is not True:
        return normalized, []
    verdict = entry_execution_scenarios.get("claim") if isinstance(entry_execution_scenarios, Mapping) else None
    problems = verdict_problems(verdict, side="entry")
    if not problems and not verdict["passed"]:
        problems = sorted(set(_verdict_reasons(verdict))) or ["claim_entry_verdict_failed"]
    if claim_contract_version != CONTRACT_VERSION and "claim_contract_version_mismatch" not in problems:
        problems.append("claim_contract_version_mismatch")
    return (False, sorted(set(problems))) if problems else (True, [])


def evaluate_exit_claim(
    *, entry_quote_legs: Mapping[str, Any] | None, exit_quote_payload: Mapping[str, Any] | None,
    expected_underlying: Optional[str] = None,
) -> dict[str, Any]:
    """The exit side of a claim: the exit quote passes and re-priced the booked contracts."""
    exit_check = evaluate_quote_contract(exit_quote_payload, expected_underlying=expected_underlying)
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
    entry_contract_version: Optional[str] = None, entry_execution_scenarios: Any = None,
) -> tuple[Optional[bool], list[str]]:
    """Fold the exit claim verdict into a row's entry-time claim fields.

    Fails closed: an entry that was claim-eligible stays claimable only if
    both its stored entry verdict and the exit verdict are well formed (see
    ``verdict_problems``), passed, and were made under this CONTRACT_VERSION,
    as was the entry. Anything else (a legacy caller with no verdict, a
    malformed or self-contradictory one, a passed verdict that still lists
    reasons, another version) makes the row non-claimable and says why. A row
    that was never claim-eligible is left as it was.
    """
    reasons = [str(reason) for reason in evidence_reasons if isinstance(reason, str)]
    normalized = _claim_flag(claim_allowed)
    if normalized is not True:
        return normalized, reasons
    entry_allowed, entry_problems = entry_claim_status(
        claim_allowed=claim_allowed, claim_contract_version=entry_contract_version,
        entry_execution_scenarios=entry_execution_scenarios)
    verdict = (exit_execution_scenarios or {}).get("claim") if isinstance(exit_execution_scenarios, Mapping) else None
    exit_problems = verdict_problems(verdict, side="exit")
    reasons = sorted(set(reasons + entry_problems + exit_problems + _verdict_reasons(verdict)))
    passed = bool(entry_allowed) and not exit_problems and verdict["passed"] is True
    return passed, reasons


def claimable_under_current_contract(row: Mapping[str, Any]) -> Optional[bool]:
    """A stored row's claim status for reporting: claimable only under this
    CONTRACT_VERSION, so claims made under different rules are never pooled.
    None when the row never had a claim status."""
    raw = row.get("claim_allowed")
    allowed = _claim_flag(raw)
    if allowed is None:
        return None
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
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc) if math.isfinite(float(value)) else None
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


__all__ = [
    "CLAIM_DISQUALIFYING_REASONS", "CONTRACT_VERSION", "ContractCheck", "apply_exit_claim",
    "claimable_under_current_contract", "contract_verdict", "entry_claim_status", "occ_root",
    "verdict_problems",
    "evaluate_exit_claim", "evaluate_quote_contract", "fee_per_share", "slippage_per_share",
]
