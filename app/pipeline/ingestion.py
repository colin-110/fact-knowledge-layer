"""Top-level per-document pipeline orchestration.

Runs page by page, persisting after every stage so a crash mid-document only
loses the current page's work (pages already marked processed are skipped on
retry - see `run_document_pipeline`).
"""

import json
import traceback

from app import embeddings, ids, storage
from app.config import CHARTS_DIR
from app.pipeline import candidate_matcher, chunks, extractor, facts as fact_extraction, normalize
from app.pipeline import relationship_engine, vision
from app.pipeline.candidate_matcher import FactRecord

CHART_CONFIDENCE_THRESHOLD = 0.35


def _build_fact_record(row) -> FactRecord:
    return FactRecord(
        id=row["id"],
        document_id=row["document_id"],
        subject=row["subject"],
        predicate=row["predicate"],
        numeric_value=row["numeric_value"],
        normalized_value=row["normalized_value"],
        normalized_unit=row["normalized_unit"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        period_label=row["period_label"],
        scope=row["scope"],
        status=row["status"],
        retrieval_text=row["retrieval_text"],
    )


def _persist_facts_from_text(document_id: str, document_title: str, evidence_id: str, evidence_text: str) -> int:
    try:
        extracted = fact_extraction.extract_facts_from_text(evidence_text, document_context=document_title)
    except Exception as exc:  # noqa: BLE001
        storage.insert_extraction_issue(
            id=ids.evidence_id(document_id, 0, "fact_extraction_error", 0, str(exc)),
            document_id=document_id, page_id=None, evidence_id=evidence_id,
            issue_type="fact_extraction_error", description=str(exc)[:500], confidence=None,
        )
        return 0

    count = 0
    fact_ids, fact_texts, fact_metas = [], [], []
    for ef in extracted:
        norm_unit = normalize.normalize_value_unit(ef.numeric_value, ef.raw_unit, ef.raw_value)
        norm_period = normalize.normalize_period(ef.period_label)
        status = ef.status or normalize.infer_status(ef.raw_value, ef.period_label, str(ef.qualifiers))

        fid = ids.fact_id(document_id, [evidence_id], ef.subject, ef.predicate, ef.raw_value, ef.period_label or "")
        retrieval_text = chunks.build_fact_retrieval_text(
            document_title, ef.subject, ef.predicate, ef.raw_value, ef.raw_unit,
            norm_unit.normalized_value, norm_unit.normalized_unit, norm_period.period_label, ef.scope, status,
        )
        storage.insert_fact(
            id=fid, document_id=document_id, subject=ef.subject, predicate=ef.predicate,
            raw_value=ef.raw_value, raw_unit=ef.raw_unit, numeric_value=ef.numeric_value,
            normalized_value=norm_unit.normalized_value, normalized_unit=norm_unit.normalized_unit,
            period_start=norm_period.period_start, period_end=norm_period.period_end,
            period_label=norm_period.period_label, scope=ef.scope, status=status, qualifiers=ef.qualifiers,
            evidence_ids=[evidence_id], source_method="llm", extraction_confidence=0.8,
            retrieval_text=retrieval_text,
        )
        fact_ids.append(fid)
        fact_texts.append(retrieval_text)
        fact_metas.append({
            "document_id": document_id, "fact_id": fid, "subject": ef.subject, "predicate": ef.predicate,
            "period_label": norm_period.period_label, "scope": ef.scope, "content_type": "fact",
        })
        count += 1

    if fact_ids:
        embeddings.upsert("facts_index", fact_ids, fact_texts, fact_metas)
    return count


def _process_page(document_id: str, document_title: str, pdf_path: str, page) -> int:
    page_id = ids.page_id(document_id, page.pdf_page_number)
    storage.upsert_page(
        id=page_id, document_id=document_id, pdf_page_number=page.pdf_page_number,
        printed_page_label=page.printed_page_label, raw_text=page.raw_text,
        width=page.width, height=page.height, is_visually_complex=page.is_visually_complex,
    )

    facts_extracted = 0
    evidence_ids_batch, evidence_texts_batch, evidence_metas_batch = [], [], []

    if page.raw_text and page.raw_text.strip():
        text_ev_id = ids.evidence_id(document_id, page.pdf_page_number, "text", 0, page.raw_text)
        storage.insert_evidence(
            id=text_ev_id, document_id=document_id, page_id=page_id, evidence_type="text",
            text=page.raw_text, bbox=None, artifact_path=None, extraction_method="pymupdf_text", confidence=0.9,
        )
        evidence_ids_batch.append(text_ev_id)
        evidence_texts_batch.append(chunks.build_evidence_retrieval_text(document_title, page.pdf_page_number, page.raw_text))
        evidence_metas_batch.append({
            "document_id": document_id, "evidence_id": text_ev_id, "page": page.pdf_page_number, "content_type": "evidence",
        })
        facts_extracted += _persist_facts_from_text(document_id, document_title, text_ev_id, page.raw_text)

    for i, table in enumerate(page.tables):
        table_text = extractor.serialize_table(table.rows)
        table_ev_id = ids.evidence_id(document_id, page.pdf_page_number, "table", i, table_text)
        storage.insert_evidence(
            id=table_ev_id, document_id=document_id, page_id=page_id, evidence_type="table",
            text=table_text, bbox=table.bbox, artifact_path=None, extraction_method="pdfplumber_table", confidence=0.85,
        )
        evidence_ids_batch.append(table_ev_id)
        evidence_texts_batch.append(chunks.build_evidence_retrieval_text(document_title, page.pdf_page_number, table_text))
        evidence_metas_batch.append({
            "document_id": document_id, "evidence_id": table_ev_id, "page": page.pdf_page_number, "content_type": "evidence",
        })
        facts_extracted += _persist_facts_from_text(document_id, document_title, table_ev_id, table_text)

    if page.is_visually_complex:
        try:
            png_bytes = extractor.render_page_png(pdf_path, page.pdf_page_number)
            chart_dir = CHARTS_DIR / document_id
            chart_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = chart_dir / f"page_{page.pdf_page_number}.png"
            artifact_path.write_bytes(png_bytes)

            chart_result = vision.extract_chart_data(png_bytes, page.raw_text)
            chart_text = chart_result.values_text or ""
            summary_bits = [b for b in [chart_result.title, chart_text, chart_result.source_note] if b]
            chart_summary = "\n".join(summary_bits) or "(vision model found no extractable chart data on this page)"

            chart_ev_id = ids.evidence_id(document_id, page.pdf_page_number, "chart", 0, chart_summary)
            storage.insert_evidence(
                id=chart_ev_id, document_id=document_id, page_id=page_id, evidence_type="chart",
                text=chart_summary, bbox=None, artifact_path=str(artifact_path),
                extraction_method="vision_llm", confidence=chart_result.confidence,
            )
            evidence_ids_batch.append(chart_ev_id)
            evidence_texts_batch.append(chunks.build_evidence_retrieval_text(document_title, page.pdf_page_number, chart_summary))
            evidence_metas_batch.append({
                "document_id": document_id, "evidence_id": chart_ev_id, "page": page.pdf_page_number, "content_type": "evidence",
            })

            if not chart_result.has_extractable_data or chart_result.confidence < CHART_CONFIDENCE_THRESHOLD:
                storage.insert_extraction_issue(
                    id=ids.evidence_id(document_id, page.pdf_page_number, "low_confidence_chart", 0, chart_summary),
                    document_id=document_id, page_id=page_id, evidence_id=chart_ev_id,
                    issue_type="low_confidence_visual_extraction",
                    description=chart_result.uncertainty_note or "Vision model could not confidently extract chart data.",
                    confidence=chart_result.confidence,
                )
            else:
                facts_extracted += _persist_facts_from_text(document_id, document_title, chart_ev_id, chart_summary)
        except Exception as exc:  # noqa: BLE001
            storage.insert_extraction_issue(
                id=ids.evidence_id(document_id, page.pdf_page_number, "vision_error", 0, str(exc)),
                document_id=document_id, page_id=page_id, evidence_id=None,
                issue_type="vision_extraction_error", description=str(exc)[:500], confidence=None,
            )

    if evidence_ids_batch:
        embeddings.upsert("evidence_index", evidence_ids_batch, evidence_texts_batch, evidence_metas_batch)

    storage.mark_page_processed(page_id)
    return facts_extracted


def run_relationship_pass(document_id: str, on_progress=None) -> int:
    """Candidate-match every fact from this document against the whole corpus and classify relationships."""
    new_fact_rows = storage.list_facts_for_document(document_id)
    found = 0
    for row in new_fact_rows:
        fact_a = _build_fact_record(row)
        candidate_ids = candidate_matcher.find_candidates(fact_a, top_k=5)
        candidate_rows = storage.get_facts_batch(candidate_ids)
        for crow in candidate_rows:
            fact_b = _build_fact_record(crow)
            if not relationship_engine.is_plausible_pair(fact_a, fact_b):
                continue
            if storage.relationship_exists(fact_a.id, fact_b.id):
                continue
            evidence_a_rows = storage.get_evidence_batch(json.loads(row["evidence_ids_json"]))
            evidence_b_rows = storage.get_evidence_batch(json.loads(crow["evidence_ids_json"]))
            evidence_a_text = "\n".join(e["text"] or "" for e in evidence_a_rows)
            evidence_b_text = "\n".join(e["text"] or "" for e in evidence_b_rows)
            try:
                classification = relationship_engine.classify_pair(fact_a, fact_b, evidence_a_text, evidence_b_text)
            except Exception:  # noqa: BLE001
                continue
            if classification.relationship_type == "UNRELATED":
                continue
            storage.insert_relationship(
                id=ids.relationship_id(fact_a.id, fact_b.id), fact_a_id=fact_a.id, fact_b_id=fact_b.id,
                relationship_type=classification.relationship_type, confidence=classification.confidence,
                context_dimension=classification.context_dimension, reason=classification.reason,
                method="rule" if classification.confidence >= 0.9 else "llm",
            )
            found += 1
        if on_progress:
            on_progress()
    return found


def run_document_pipeline(document_id: str, job_id: str, on_stage_update=None):
    doc = storage.get_document(document_id)
    pdf_path = doc["file_path"]
    document_title = doc["title"] or doc["filename"]

    storage.update_job(job_id, status="processing", stage="extracting_pages", started_at=storage.now())

    try:
        pages = extractor.extract_document(pdf_path)
    except Exception as exc:  # noqa: BLE001
        storage.update_job(job_id, status="failed", error=f"PDF extraction failed: {exc}")
        return

    total_pages = len(pages)
    storage.update_job(job_id, total_pages=total_pages, stage="processing_pages")

    pages_processed = 0
    facts_extracted = 0

    for page in pages:
        existing = storage.get_page(document_id, page.pdf_page_number)
        if existing is not None and existing["processed_at"]:
            pages_processed += 1
            continue
        try:
            facts_extracted += _process_page(document_id, document_title, pdf_path, page)
        except Exception:  # noqa: BLE001
            storage.insert_extraction_issue(
                id=ids.evidence_id(document_id, page.pdf_page_number, "page_error", 0, traceback.format_exc()[:200]),
                document_id=document_id, page_id=ids.page_id(document_id, page.pdf_page_number), evidence_id=None,
                issue_type="page_processing_error", description=traceback.format_exc()[:1000], confidence=None,
            )
        pages_processed += 1
        progress = int(70 * pages_processed / max(total_pages, 1))
        storage.update_job(
            job_id, pages_processed=pages_processed, facts_extracted=facts_extracted, progress=progress,
        )
        if on_stage_update:
            on_stage_update()

    storage.update_job(job_id, stage="finding_relationships", progress=75)
    relationships_found = run_relationship_pass(document_id)

    storage.update_job(
        job_id, status="completed", stage="completed", progress=100,
        relationships_found=relationships_found, completed_at=storage.now(),
    )
