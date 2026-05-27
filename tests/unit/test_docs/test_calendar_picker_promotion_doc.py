"""Doc-consistency tests for docs/CALENDAR_PICKER_PROMOTION_2026-05-27.md.

Anchors the governance document to the actual code state. The
document makes specific claims about prerequisites and the
implementation surface; if a future refactor renames or removes
those surfaces, the doc's claims would silently go stale. These
tests fail loudly when that happens.

NB: PR-AE C6 explicitly disclaims any claim of "validation." The
promotion-eligible block can populate only after the scheduled
resolver has actually run on post-PR-AE-merge forward events. The
doc must reflect that. Tests below enforce both:

  - The named code surfaces still exist (no silent rename).
  - The doc never uses the word "validated" to describe the +14d
    rule, only "infrastructure landed" / "accumulating evidence".

The doc has a few legitimate uses of "validation" (e.g., SR 11-7
ongoing validation, validation-set terminology), so the test is
targeted at the specific "validated" claim about the candidate
rule, not all variants of the word.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


DOC_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "CALENDAR_PICKER_PROMOTION_2026-05-27.md"
)


@pytest.fixture(scope="module")
def doc_text() -> str:
    """Read the promotion governance doc once per test module."""
    return DOC_PATH.read_text(encoding="utf-8")


def test_doc_exists_and_is_readable() -> None:
    """Trivial precondition: the doc file we're testing actually
    exists at the expected path. Catches refactors that move it."""
    assert DOC_PATH.exists()
    assert DOC_PATH.stat().st_size > 0


def test_doc_names_the_resolver_service_module(doc_text: str) -> None:
    """The doc claims PR-AE infrastructure landed; it must name the
    actual module that carries the implementation so a future
    refactor that renames the module fails this test rather than
    silently rendering the doc inaccurate."""
    assert "services/candidate_exit_resolver" in doc_text


def test_doc_names_the_resolver_cli_script(doc_text: str) -> None:
    """Same contract for the C5 CLI script — operators reading the
    doc to schedule the resolver must find the right path."""
    assert "scripts/resolve_candidate_exits" in doc_text


def test_doc_names_the_shared_simulator_module(doc_text: str) -> None:
    """C2 extracted the simulator to services/candidate_shadow_outcome
    so the resolver could call it without a services -> web edge. The
    doc must point at that module so the architectural separation
    stays visible in the governance trail."""
    assert "services/candidate_shadow_outcome" in doc_text


def test_doc_names_the_atomic_write_helper(doc_text: str) -> None:
    """The atomic combined helper (C1b) is what closes the
    revision-write + counter-increment race. Doc must name it."""
    assert "record_resolution_and_attempt" in doc_text


def test_doc_names_the_eligibility_filter_method(doc_text: str) -> None:
    """The ledger's eligibility filter is the entry point the
    resolver uses. Doc names it so operators tracing the flow can
    follow it."""
    assert "list_pending_candidate_exit_resolutions" in doc_text


def test_doc_marks_resolver_prerequisite_as_landed_not_validated(
    doc_text: str,
) -> None:
    """Codex C6 hard requirement: the doc must say "INFRASTRUCTURE
    LANDED" for prerequisite (3), not "validated".

    Validation requires post-merge forward observations that have
    been resolved by an actually-scheduled resolver — landing the
    code is necessary but nowhere near sufficient. The doc must
    not blur that line."""
    assert "INFRASTRUCTURE LANDED in PR-AE" in doc_text


def test_doc_does_not_claim_the_plus_14d_rule_is_validated(
    doc_text: str,
) -> None:
    """REGRESSION (Codex C5 + C6 audit): the doc must NOT contain
    any claim that the +14d rule has been validated by forward
    evidence. Specific phrases that would be wrong:

      - "+14d rule validated"
      - "candidate_min_dte validated"
      - "rule has been validated"

    Codex specifically called out: do not say 'validated', say
    'infrastructure landed' or 'accumulating evidence'.
    """
    forbidden_phrases = [
        # The +14d rule itself
        r"\+14d rule (?:has been |is )?validated",
        r"candidate_min_dte (?:has been |is )?validated",
        # The rule getting "validation" attached to it as a verb
        r"rule (?:has been |is )?validated by",
    ]
    for pattern in forbidden_phrases:
        matches = re.findall(pattern, doc_text, flags=re.IGNORECASE)
        assert matches == [], (
            f"Doc must not claim the candidate rule is validated. "
            f"Found forbidden pattern {pattern!r}: {matches}"
        )


def test_doc_acknowledges_promotion_criteria_still_not_actionable(
    doc_text: str,
) -> None:
    """Even though prerequisite (3) is now LANDED, the
    outcome-based promotion criteria themselves are NOT YET
    ACTIONABLE until forward-data accumulation (prerequisite 4)
    catches up. The doc must keep this distinction explicit."""
    # Phrase variants that satisfy the contract
    not_actionable_phrases = [
        "NOT YET ACTIONABLE",
        "not actionable today",
        "not yet evaluable",
    ]
    found = any(phrase in doc_text for phrase in not_actionable_phrases)
    assert found, (
        "Doc must keep stating that promotion criteria are NOT YET "
        "ACTIONABLE — landing the infrastructure is NOT the same "
        "as accumulating enough forward observations to evaluate "
        "the criteria. If you removed that phrasing, restore it."
    )


def test_doc_preserves_anti_patterns_section(doc_text: str) -> None:
    """Anti-patterns (in-sample re-validation, sample stitching,
    threshold drift, outcome reading from selection alone) must
    survive every revision. PR-AE adds infrastructure; it does
    NOT relax governance."""
    for anti_pattern in (
        "In-sample re-validation",
        "Sample stitching",
        "Threshold drift",
        "Outcome reading from selection alone",
    ):
        assert anti_pattern in doc_text, (
            f"Anti-pattern '{anti_pattern}' missing from doc. The "
            f"anti-patterns section is the governance backbone — "
            f"do not remove or weaken it without an explicit policy "
            f"change."
        )


def test_doc_history_records_pr_ae_update(doc_text: str) -> None:
    """The document history section must record the PR-AE C6
    update so future readers can trace when the prerequisite (3)
    status flip happened."""
    assert "PR-AE" in doc_text
    # The history line itself
    assert "PR-AE update" in doc_text or "PR-AE merge" in doc_text


def test_doc_preserves_in_sample_replay_warning(doc_text: str) -> None:
    """The historical-replay candidate outcome stats MUST stay
    labeled as "diagnostic only" / "not promotion evidence" so
    nobody accidentally reads them as forward evidence. PR-AE C6
    must not weaken that framing."""
    assert "diagnostic only" in doc_text or "diagnostic_candidate_stats" in doc_text
    # And the in-sample tag must still appear as not-promotion-eligible
    assert "historical_replay_in_sample_or_research" in doc_text


def test_doc_clock_starts_at_pr_ae_merge(doc_text: str) -> None:
    """Codex C6 requirement: the forward-data accumulation clock
    starts at PR-AE merge, not earlier. The doc must say so
    explicitly so future readers can't argue PR-AC or PR-AD events
    count."""
    # Look for the explicit clock-start language
    clock_phrases = [
        "clock starts at PR-AE",
        "after the PR-AE merge",
        "post-PR-AE",
    ]
    found = any(phrase in doc_text for phrase in clock_phrases)
    assert found, (
        "Doc must explicitly state that the forward-data clock "
        "starts at the PR-AE merge commit. Without this, future "
        "readers could argue PR-AC/PR-AD events count as forward "
        "evidence — they do not."
    )
