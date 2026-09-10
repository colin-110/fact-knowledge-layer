"""Fact-pair relationship classification: cheap deterministic rules first,
LLM classification (with both facts' source evidence) for anything ambiguous.

Relationship types: CORROBORATES, CONTRADICTS, LIKELY_CONTRADICTION,
CONTEXT_RECONCILES, UNRELATED.
"""

import difflib
from dataclasses import dataclass

from app.llm import complete_json
from app.models import RelationshipClassificationLLM
from app.pipeline.candidate_matcher import FactRecord

SYSTEM_PROMPT = """You compare two facts extracted from (possibly different) source documents and classify \
their relationship, using the source evidence text provided for each.

Relationship types:
- CORROBORATES: the two facts state the same underlying claim and are consistent once you account for units, \
  rounding, or phrasing.
- CONTRADICTS: the two facts make materially incompatible claims about the same subject/metric/period/scope with \
  no reasonable explanation visible in the evidence.
- LIKELY_CONTRADICTION: the two facts appear to conflict but you are not fully certain it is a genuine error - \
  e.g. small numeric discrepancy with no clear cause found in the evidence. Do not invent an explanation you \
  cannot see in the text - if there's no visible reason for a small discrepancy, call it LIKELY_CONTRADICTION \
  and say so plainly (e.g. "no explicit scope/time distinction found; may be rounding or a reporting error").
- CONTEXT_RECONCILES: the facts differ, but the evidence shows why - different time period, data vintage \
  (e.g. an early estimate vs. a later actual/revised figure), reporting scope (consolidated vs standalone), \
  currency, or units.
- UNRELATED: the two facts are not actually about the same underlying claim, despite superficial similarity.

Be conservative: only claim CORROBORATES or CONTEXT_RECONCILES if the evidence actually supports the specific \
reason you give. If you are guessing, prefer LIKELY_CONTRADICTION or UNRELATED and say your reasoning is uncertain.

Common false positives to watch for - these look like the same metric by wording alone but are UNRELATED unless \
the evidence explicitly equates them:
- An adjusted figure vs. its unadjusted counterpart (e.g. "Adjusted EBITDA" vs. "EBITDA", "EBITDA margin" vs. \
  "Adj. EBITDA margin") - these are different metrics by definition, not a restatement of each other.
- A year-over-year change vs. a quarter-over-quarter change (or any two growth rates measured over different \
  bases/windows) - both can be true at once and are not comparable as if they were the same number.
- An incremental/delta figure (e.g. "FY24 increase") vs. an absolute/level figure (e.g. "FY24 amount") for the \
  same metric - a change and a level are not the same claim.
- Two different named people, entities, or line-item categories that happen to share a predicate (e.g. two \
  different individuals' remuneration, or two different categories of "contingent liability") - check the \
  subject is actually the same real-world entity, not just a similar-looking label.
- Two different metrics that both mention the same headline word (e.g. "total borrowings" vs. "total income") -
  matching on "total ..." is not matching on the metric."""


def _sim(a: str | None, b: str | None) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower().strip(), (b or "").lower().strip()).ratio()


def _same_period(a: FactRecord, b: FactRecord) -> bool:
    if a.period_start and a.period_end and b.period_start and b.period_end:
        return a.period_start == b.period_start and a.period_end == b.period_end
    if a.period_label and b.period_label:
        return a.period_label.strip().lower() == b.period_label.strip().lower()
    return False


def is_plausible_pair(a: FactRecord, b: FactRecord) -> bool:
    if a.id == b.id:
        return False
    return _sim(a.subject, b.subject) >= 0.35 and _sim(a.predicate, b.predicate) >= 0.35


# Character-level similarity on short financial-jargon phrases is not a reliable signal of
# "same real-world metric" - verified against a live run of the actual dataset, where genuinely
# unrelated pairs like "total borrowings" vs "total income" (0.57), "Adj. EBITDA" vs "EBITDA"
# (0.71), and even two different people's names ("Kapil Bharati" vs "Sahil Barua", 0.67) all
# score well above is_plausible_pair's 0.35 candidate-generation threshold, while genuine
# paraphrases across documents ("revenue from services" vs "revenue", 0.50) score little higher.
# There is no threshold in between that separates the two classes - so the deterministic,
# LLM-skipping shortcut below is restricted to near-identical wording only (the one case where
# "same metric" needs no semantic judgment); everything else defers to the LLM, which reads the
# actual evidence text and can tell "total borrowings" from "total income".
_NEAR_IDENTICAL_THRESHOLD = 0.92

# Overall string similarity alone still isn't enough: "express parcel shipments YoY growth" vs
# "...QoQ growth" scores 0.94 (above the threshold above) despite being different, non-comparable
# metrics - only two characters differ. If a marker below appears in exactly one of the two
# predicates, that asymmetry alone means "not the same metric," regardless of how similar the
# rest of the wording is.
_DISTINGUISHING_MARKERS = (
    "yoy", "y-o-y", "year-on-year", "year over year",
    "qoq", "q-o-q", "quarter-on-quarter", "quarter over quarter",
    "mom", "m-o-m", "month-on-month",
    "adj", "adjusted",
)


def _has_asymmetric_marker(a_text: str, b_text: str) -> bool:
    a_l, b_l = (a_text or "").lower(), (b_text or "").lower()
    return any((marker in a_l) != (marker in b_l) for marker in _DISTINGUISHING_MARKERS)


def _same_underlying_metric_by_wording_alone(a: FactRecord, b: FactRecord) -> bool:
    if _has_asymmetric_marker(a.predicate, b.predicate):
        return False
    return _sim(a.subject, b.subject) >= _NEAR_IDENTICAL_THRESHOLD and _sim(a.predicate, b.predicate) >= _NEAR_IDENTICAL_THRESHOLD


def rule_based_classify(a: FactRecord, b: FactRecord) -> RelationshipClassificationLLM | None:
    """Return a classification only when the deterministic rules are confident; None means 'ask the LLM'."""
    if not _same_underlying_metric_by_wording_alone(a, b):
        return None

    same_scope = (a.scope or "").strip().lower() == (b.scope or "").strip().lower()
    same_status = (a.status or "").strip().lower() == (b.status or "").strip().lower()
    same_period = _same_period(a, b)

    has_comparable_numbers = (
        a.normalized_value is not None
        and b.normalized_value is not None
        and a.normalized_unit
        and a.normalized_unit == b.normalized_unit
    )

    if not (same_period and has_comparable_numbers):
        return None

    va, vb = a.normalized_value, b.normalized_value
    denom = max(abs(va), abs(vb), 1e-9)
    rel_diff = abs(va - vb) / denom

    if same_scope and same_status:
        if rel_diff <= 0.01:
            return RelationshipClassificationLLM(
                relationship_type="CORROBORATES",
                confidence=0.95,
                context_dimension="none",
                reason=(
                    f"Same subject, metric, period ({a.period_label}), scope, and status; normalized values "
                    f"{va} vs {vb} {a.normalized_unit} agree within rounding."
                ),
            )
        if rel_diff <= 0.15:
            return RelationshipClassificationLLM(
                relationship_type="LIKELY_CONTRADICTION",
                confidence=0.6,
                context_dimension="none",
                reason=(
                    f"Same subject, metric, period ({a.period_label}), scope, and status, but normalized values "
                    f"differ ({va} vs {vb} {a.normalized_unit}) with no distinguishing qualifier found - possible "
                    f"rounding or reporting inconsistency."
                ),
            )
        return RelationshipClassificationLLM(
            relationship_type="CONTRADICTS",
            confidence=0.75,
            context_dimension="none",
            reason=(
                f"Same subject, metric, period ({a.period_label}), scope, and status, but values are materially "
                f"different ({va} vs {vb} {a.normalized_unit})."
            ),
        )

    if not same_scope:
        return RelationshipClassificationLLM(
            relationship_type="CONTEXT_RECONCILES",
            confidence=0.7,
            context_dimension="scope",
            reason=f"Same metric and period but different reporting scope ({a.scope!r} vs {b.scope!r}).",
        )

    if not same_status:
        return RelationshipClassificationLLM(
            relationship_type="CONTEXT_RECONCILES",
            confidence=0.65,
            context_dimension="estimate_vintage",
            reason=f"Same metric and period but different disclosed status ({a.status!r} vs {b.status!r}).",
        )

    return None


def llm_classify(a: FactRecord, b: FactRecord, evidence_a: str, evidence_b: str) -> RelationshipClassificationLLM:
    # Since the stricter same-metric gate above now sends most candidate pairs here instead of
    # the free deterministic path, call volume is much higher - kept the evidence excerpt short
    # (the sentence with the actual number is rarely more than a couple hundred characters from
    # its start) so more classifications fit in a free-tier account's tight tokens-per-minute
    # budget (see GROQ_TPM_LIMIT).
    user_prompt = f"""Fact A:
Subject: {a.subject}
Predicate: {a.predicate}
Period: {a.period_label}
Scope: {a.scope}
Status: {a.status}
Normalized value: {a.normalized_value} {a.normalized_unit}
Source evidence: {evidence_a[:500]}

Fact B:
Subject: {b.subject}
Predicate: {b.predicate}
Period: {b.period_label}
Scope: {b.scope}
Status: {b.status}
Normalized value: {b.normalized_value} {b.normalized_unit}
Source evidence: {evidence_b[:500]}

Classify the relationship between Fact A and Fact B."""
    return complete_json(SYSTEM_PROMPT, user_prompt, RelationshipClassificationLLM)


def classify_pair(a: FactRecord, b: FactRecord, evidence_a: str, evidence_b: str) -> RelationshipClassificationLLM:
    rule_result = rule_based_classify(a, b)
    if rule_result is not None:
        return rule_result
    return llm_classify(a, b, evidence_a, evidence_b)
