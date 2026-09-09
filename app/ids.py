import hashlib


def _hash(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def document_id_from_hash(sha256_hex: str) -> str:
    return f"doc_{sha256_hex[:20]}"


def page_id(document_id: str, page_number: int) -> str:
    return f"pg_{_hash(document_id, str(page_number))}"


def evidence_id(document_id: str, page_number: int, evidence_type: str, index: int, text_seed: str) -> str:
    return f"ev_{_hash(document_id, str(page_number), evidence_type, str(index), text_seed[:80])}"


def fact_id(document_id: str, evidence_ids: list[str], subject: str, predicate: str, raw_value: str, period_label: str) -> str:
    return f"fa_{_hash(document_id, ','.join(sorted(evidence_ids)), subject, predicate, raw_value, period_label or '')}"


def relationship_id(fact_a_id: str, fact_b_id: str) -> str:
    a, b = sorted([fact_a_id, fact_b_id])
    return f"rel_{_hash(a, b)}"
