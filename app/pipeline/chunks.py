"""Canonical retrieval-text builders.

We deliberately do NOT embed raw PDF fragments. Embedding a well-formed
sentence describing the fact/evidence (with document, period, scope context
folded in) makes both dense retrieval and human inspection far more useful.
"""


def build_fact_retrieval_text(
    document_title: str | None,
    subject: str,
    predicate: str,
    raw_value: str | None,
    raw_unit: str | None,
    normalized_value: float | None,
    normalized_unit: str | None,
    period_label: str | None,
    scope: str | None,
    status: str | None,
) -> str:
    lines = [f"Subject: {subject}", f"Metric: {predicate}"]
    value_str = f"{raw_value} {raw_unit}".strip() if raw_unit else (raw_value or "")
    if value_str:
        lines.append(f"Value: {value_str}")
    if normalized_value is not None and normalized_unit:
        lines.append(f"Normalized value: {normalized_value} {normalized_unit}")
    if period_label:
        lines.append(f"Period: {period_label}")
    if scope:
        lines.append(f"Scope: {scope}")
    if status:
        lines.append(f"Status: {status}")
    if document_title:
        lines.append(f"Source: {document_title}")
    return "\n".join(lines)


def build_evidence_retrieval_text(document_title: str | None, page_number: int, text: str) -> str:
    header = f"Document: {document_title or 'unknown'}\nPage: {page_number}"
    return f"{header}\n\n{text}".strip()
