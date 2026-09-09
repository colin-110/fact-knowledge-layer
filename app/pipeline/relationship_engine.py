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
reason you give. If you are guessing, prefer LIKELY_CONTRADICTION or UNRELATED and say your reasoning is uncertain."""


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


def rule_based_classify(a: FactRecord, b: FactRecord) -> RelationshipClassificationLLM | None:
    """Return a classification only when the deterministic rules are confident; None means 'ask the LLM'."""
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
        if rel_diff <= 0.02:
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
    user_prompt = f"""Fact A:
Subject: {a.subject}
Predicate: {a.predicate}
Period: {a.period_label}
Scope: {a.scope}
Status: {a.status}
Normalized value: {a.normalized_value} {a.normalized_unit}
Source evidence: {evidence_a[:1200]}

Fact B:
Subject: {b.subject}
Predicate: {b.predicate}
Period: {b.period_label}
Scope: {b.scope}
Status: {b.status}
Normalized value: {b.normalized_value} {b.normalized_unit}
Source evidence: {evidence_b[:1200]}

Classify the relationship between Fact A and Fact B."""
    return complete_json(SYSTEM_PROMPT, user_prompt, RelationshipClassificationLLM)


def classify_pair(a: FactRecord, b: FactRecord, evidence_a: str, evidence_b: str) -> RelationshipClassificationLLM:
    rule_result = rule_based_classify(a, b)
    if rule_result is not None:
        return rule_result
    return llm_classify(a, b, evidence_a, evidence_b)
