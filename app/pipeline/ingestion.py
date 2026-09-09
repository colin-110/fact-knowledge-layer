"""Top-level per-document pipeline orchestration.

Pages are processed concurrently (a thread pool, not one at a time) because
the actual cost per page is almost entirely waiting on Groq API calls -
network I/O releases the GIL, so threads are a real win here without needing
multiprocessing. Each page still persists its own evidence/facts as soon as
it finishes, so a crash mid-document only loses whatever pages were still
in flight (pages already marked processed are skipped on retry).

Two LLM calls per page in the common case (one combined text+table fact
extraction call, one vision call only on visually-complex pages) rather than
one call per table - merging table text into the same extraction call cut
per-page LLM calls roughly 2-3x on table-heavy pages.
"""

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from app import embeddings, ids, llm, storage
from app.config import CHARTS_DIR, GROQ_VISION_MODEL, INGESTION_WORKERS
from app.pipeline import candidate_matcher, chunks, extractor, facts as fact_extraction, normalize
from app.pipeline import relationship_engine, vision
from app.pipeline import spatial_text as spatial_text_mod
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


def _persist_facts_from_text(
    document_id: str, document_title: str, evidence_ids: list[str], source_text: str
) -> int:
    try:
        extracted = fact_extraction.extract_facts_from_text(source_text, document_context=document_title)
    except Exception as exc:  # noqa: BLE001
        storage.insert_extraction_issue(
            id=ids.evidence_id(document_id, 0, "fact_extraction_error", 0, str(exc)),
            document_id=document_id, page_id=None, evidence_id=evidence_ids[0] if evidence_ids else None,
            issue_type="fact_extraction_error", description=str(exc)[:500], confidence=None,
        )
        return 0

    count = 0
    fact_ids, fact_texts, fact_metas = [], [], []
    for ef in extracted:
        norm_unit = normalize.normalize_value_unit(ef.numeric_value, ef.raw_unit, ef.raw_value)
        norm_period = normalize.normalize_period(ef.period_label)
        status = ef.status or normalize.infer_status(ef.raw_value, ef.period_label, str(ef.qualifiers))

        fid = ids.fact_id(document_id, evidence_ids, ef.subject, ef.predicate, ef.raw_value, ef.period_label or "")
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
            evidence_ids=evidence_ids, source_method="llm", extraction_confidence=0.8,
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


class _RunState:
    """Shared, lock-guarded counters/flags for one document's concurrent page workers."""

    def __init__(self, job_id: str, total_pages: int):
        self.lock = threading.Lock()
        self.job_id = job_id
        self.total_pages = total_pages
        self.pages_processed = 0
        self.facts_extracted = 0
        self.vision_issue_logged = False

    def record_page_done(self, facts_from_page: int):
        with self.lock:
            self.pages_processed += 1
            self.facts_extracted += facts_from_page
            progress = int(70 * self.pages_processed / max(self.total_pages, 1))
            storage.update_job(
                self.job_id, pages_processed=self.pages_processed,
                facts_extracted=self.facts_extracted, progress=progress,
            )

    def maybe_log_vision_unavailable_once(self, document_id: str, page_id: str):
        with self.lock:
            if self.vision_issue_logged:
                return False
            self.vision_issue_logged = True
            return True


def _process_page(document_id: str, document_title: str, pdf_path: str, page, run_state: _RunState) -> int:
    page_id = ids.page_id(document_id, page.pdf_page_number)
    storage.upsert_page(
        id=page_id, document_id=document_id, pdf_page_number=page.pdf_page_number,
        printed_page_label=page.printed_page_label, raw_text=page.raw_text,
        width=page.width, height=page.height, is_visually_complex=page.is_visually_complex,
    )

    facts_extracted = 0
    evidence_ids_batch, evidence_texts_batch, evidence_metas_batch = [], [], []
    combined_text_parts: list[str] = []
    combined_evidence_ids: list[str] = []

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
        combined_text_parts.append(page.raw_text)
        combined_evidence_ids.append(text_ev_id)

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
        combined_text_parts.append(f"--- TABLE {i + 1} ---\n{table_text}")
        combined_evidence_ids.append(table_ev_id)

    # One fact-extraction call for the whole page (text + all tables together) instead of
    # one call per evidence unit - this is the single biggest reduction in LLM call count.
    if combined_text_parts:
        combined_text = "\n\n".join(combined_text_parts)
        facts_extracted += _persist_facts_from_text(document_id, document_title, combined_evidence_ids, combined_text)

    if page.is_visually_complex:
        try:
            png_bytes = extractor.render_page_png(pdf_path, page.pdf_page_number)
            chart_dir = CHARTS_DIR / document_id
            chart_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = chart_dir / f"page_{page.pdf_page_number}.png"
            artifact_path.write_bytes(png_bytes)

            # Reflow this page's text by spatial position rather than raw reading order - a
            # bar chart's value labels scattered around the page often read back scrambled in
            # block order but land near their axis label once grouped by proximity. Free (no
            # API call), degrades to "" for scanned/rasterized pages with no text layer.
            spatial_text = spatial_text_mod.extract_clustered_text(pdf_path, page.pdf_page_number)

            if not vision.vision_available():
                # Already confirmed unusable earlier in this run (Groq dead and no Gemini key) -
                # keep the page image (still useful for manual inspection) but don't waste a call
                # we know will fail. Still worth a fact-extraction pass over the spatially-reflowed
                # text, since that's strictly better-ordered than the raw text already tried above.
                chart_ev_id = ids.evidence_id(document_id, page.pdf_page_number, "chart", 0, "vision_unavailable")
                storage.insert_evidence(
                    id=chart_ev_id, document_id=document_id, page_id=page_id, evidence_type="chart",
                    text=spatial_text or None, bbox=None, artifact_path=str(artifact_path),
                    extraction_method="spatial_text_fallback" if spatial_text else "vision_llm_unavailable",
                    confidence=0.4 if spatial_text else None,
                )
                if spatial_text and len(spatial_text) > 20:
                    evidence_ids_batch.append(chart_ev_id)
                    evidence_texts_batch.append(chunks.build_evidence_retrieval_text(document_title, page.pdf_page_number, spatial_text))
                    evidence_metas_batch.append({
                        "document_id": document_id, "evidence_id": chart_ev_id, "page": page.pdf_page_number, "content_type": "evidence",
                    })
                    facts_extracted += _persist_facts_from_text(document_id, document_title, [chart_ev_id], spatial_text)
                if run_state.maybe_log_vision_unavailable_once(document_id, page_id):
                    storage.insert_extraction_issue(
                        id=ids.evidence_id(document_id, 0, "vision_model_unavailable", 0, GROQ_VISION_MODEL),
                        document_id=document_id, page_id=page_id, evidence_id=chart_ev_id,
                        issue_type="vision_model_unavailable",
                        description=(
                            f"Neither '{GROQ_VISION_MODEL}' nor Gemini (no GEMINI_API_KEY configured) is "
                            f"available - confirmed on an earlier page this run. Visually-complex pages (this "
                            f"and any others) fall back to spatially-reflowed text extraction instead of a "
                            f"real chart read; the rendered page image is still saved for manual review."
                        ),
                        confidence=None,
                    )
            else:
                chart_result = vision.extract_chart_data(png_bytes, spatial_text or page.raw_text)
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
                    facts_extracted += _persist_facts_from_text(document_id, document_title, [chart_ev_id], chart_summary)
        except llm.ModelUnavailableError as exc:
            if run_state.maybe_log_vision_unavailable_once(document_id, page_id):
                storage.insert_extraction_issue(
                    id=ids.evidence_id(document_id, page.pdf_page_number, "vision_error", 0, str(exc)),
                    document_id=document_id, page_id=page_id, evidence_id=None,
                    issue_type="vision_model_unavailable", description=str(exc)[:500], confidence=None,
                )
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

    pages_to_process = []
    already_done = 0
    for page in pages:
        existing = storage.get_page(document_id, page.pdf_page_number)
        if existing is not None and existing["processed_at"]:
            already_done += 1
        else:
            pages_to_process.append(page)

    run_state = _RunState(job_id, total_pages)
    run_state.pages_processed = already_done

    def _worker(page):
        try:
            facts = _process_page(document_id, document_title, pdf_path, page, run_state)
        except Exception:  # noqa: BLE001
            storage.insert_extraction_issue(
                id=ids.evidence_id(document_id, page.pdf_page_number, "page_error", 0, traceback.format_exc()[:200]),
                document_id=document_id, page_id=ids.page_id(document_id, page.pdf_page_number), evidence_id=None,
                issue_type="page_processing_error", description=traceback.format_exc()[:1000], confidence=None,
            )
            facts = 0
        run_state.record_page_done(facts)
        if on_stage_update:
            on_stage_update()

    with ThreadPoolExecutor(max_workers=INGESTION_WORKERS) as pool:
        futures = [pool.submit(_worker, page) for page in pages_to_process]
        for future in as_completed(futures):
            future.result()  # re-raise any programming error that escaped _worker's own try/except

    storage.update_job(job_id, stage="finding_relationships", progress=75)
    relationships_found = run_relationship_pass(document_id)

    storage.update_job(
        job_id, status="completed", stage="completed", progress=100,
        relationships_found=relationships_found, completed_at=storage.now(),
    )
