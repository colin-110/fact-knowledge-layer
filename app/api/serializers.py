import json

from app import storage
from app.models import EvidenceOut, FactOut


def evidence_row_to_out(row) -> EvidenceOut:
    return EvidenceOut(
        id=row["id"], document_id=row["document_id"], document_filename=row["document_filename"],
        page_number=row["pdf_page_number"], evidence_type=row["evidence_type"], text=row["text"],
        artifact_path=row["artifact_path"], extraction_method=row["extraction_method"], confidence=row["confidence"],
    )


def fact_row_to_out(row, include_evidence: bool = True) -> FactOut:
    evidence_out = []
    if include_evidence:
        evidence_ids = json.loads(row["evidence_ids_json"] or "[]")
        evidence_out = [evidence_row_to_out(r) for r in storage.get_evidence_batch(evidence_ids)]
    return FactOut(
        id=row["id"], document_id=row["document_id"], document_filename=row["document_filename"],
        subject=row["subject"], predicate=row["predicate"], raw_value=row["raw_value"], raw_unit=row["raw_unit"],
        numeric_value=row["numeric_value"], normalized_value=row["normalized_value"],
        normalized_unit=row["normalized_unit"], period_label=row["period_label"], scope=row["scope"],
        status=row["status"], qualifiers=json.loads(row["qualifiers_json"] or "{}"),
        extraction_confidence=row["extraction_confidence"], evidence=evidence_out,
    )
