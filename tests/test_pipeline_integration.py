"""End-to-end pipeline test with the LLM and embedding calls mocked out.

Everything else is real: PDF extraction (a real synthetic PDF via PyMuPDF),
SQLite storage, deterministic IDs, and the page-processing/job-orchestration
logic in ingestion.py. This is the one test that actually exercises
run_document_pipeline as a whole rather than a single function in isolation -
none of the unit tests elsewhere touch this orchestration path at all.
"""

from unittest.mock import patch

import pymupdf
import pytest

from app import ids, storage
from app.db import init_db
from app.models import ExtractedFactLLM
from app.pipeline import ingestion


@pytest.fixture(autouse=True, scope="module")
def _ensure_db():
    init_db()


def _build_single_page_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((20, 20), text)
    doc.save(path)
    doc.close()


def _make_document(tmp_path, sha, text):
    pdf_path = tmp_path / f"{sha}.pdf"
    _build_single_page_pdf(pdf_path, text)
    doc_id = ids.document_id_from_hash(sha)
    storage.create_document(
        id=doc_id, filename="sample.pdf", sha256=sha, title="Sample Report",
        page_count=1, file_path=str(pdf_path),
    )
    return doc_id, str(pdf_path)


class TestEndToEndPipeline:
    def test_pipeline_creates_page_evidence_and_facts(self, tmp_path):
        doc_id, _ = _make_document(tmp_path, "sha_e2e_1", "Revenue was 100 million in FY24.")
        job_id = "job_e2e_1"
        storage.create_job(id=job_id, document_id=doc_id)

        fake_fact = ExtractedFactLLM(
            subject="TestCo", predicate="revenue", raw_value="100", raw_unit="million",
            numeric_value=100.0, period_label="FY24", supporting_quote="Revenue was 100 million in FY24.",
        )

        with patch("app.pipeline.ingestion.fact_extraction.extract_facts_from_text", return_value=[fake_fact]), \
             patch("app.pipeline.ingestion.embeddings.upsert"):
            ingestion.run_document_pipeline(doc_id, job_id)

        job = storage.get_job(job_id)
        assert job["status"] == "completed"
        assert job["pages_processed"] == 1
        assert job["facts_extracted"] == 1

        facts = storage.list_facts_for_document(doc_id)
        assert len(facts) == 1
        assert facts[0]["subject"] == "TestCo"
        assert facts[0]["normalized_value"] == 100_000_000  # "100 million" with no currency -> plain count

        # every fact must carry at least one evidence id - the hard invariant from the README
        import json
        evidence_ids = json.loads(facts[0]["evidence_ids_json"])
        assert len(evidence_ids) >= 1
        evidence_rows = storage.get_evidence_batch(evidence_ids)
        assert len(evidence_rows) >= 1
        assert "Revenue was 100 million" in (evidence_rows[0]["text"] or "")

    def test_reprocessing_same_document_skips_already_processed_pages(self, tmp_path):
        """Idempotency: a page marked processed must not trigger a second LLM call on retry."""
        doc_id, _ = _make_document(tmp_path, "sha_e2e_2", "EBITDA was 50 crore in FY23.")
        job_id_1 = "job_e2e_2a"
        storage.create_job(id=job_id_1, document_id=doc_id)

        fake_fact = ExtractedFactLLM(
            subject="TestCo", predicate="EBITDA", raw_value="50", raw_unit="crore",
            numeric_value=50.0, period_label="FY23", supporting_quote="EBITDA was 50 crore in FY23.",
        )

        with patch("app.pipeline.ingestion.fact_extraction.extract_facts_from_text", return_value=[fake_fact]) as mock_extract, \
             patch("app.pipeline.ingestion.embeddings.upsert"):
            ingestion.run_document_pipeline(doc_id, job_id_1)
            assert mock_extract.call_count == 1

        job_id_2 = "job_e2e_2b"
        storage.create_job(id=job_id_2, document_id=doc_id)
        with patch("app.pipeline.ingestion.fact_extraction.extract_facts_from_text") as mock_extract_2, \
             patch("app.pipeline.ingestion.embeddings.upsert"):
            ingestion.run_document_pipeline(doc_id, job_id_2)
            mock_extract_2.assert_not_called()

        job_2 = storage.get_job(job_id_2)
        assert job_2["status"] == "completed"
        assert job_2["pages_processed"] == 1

        # still exactly one fact - reprocessing didn't duplicate it
        assert len(storage.list_facts_for_document(doc_id)) == 1

    def test_pdf_extraction_failure_marks_job_failed_not_stuck(self, tmp_path):
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_bytes(b"not actually a pdf")
        doc_id = ids.document_id_from_hash("sha_e2e_corrupt")
        storage.create_document(
            id=doc_id, filename="corrupt.pdf", sha256="sha_e2e_corrupt", title="Corrupt",
            page_count=0, file_path=str(bad_pdf),
        )
        job_id = "job_e2e_corrupt"
        storage.create_job(id=job_id, document_id=doc_id)

        ingestion.run_document_pipeline(doc_id, job_id)

        job = storage.get_job(job_id)
        assert job["status"] == "failed"
        assert job["error"]

    def test_fact_extraction_exception_does_not_abort_the_whole_page(self, tmp_path):
        """One evidence unit failing extraction (e.g. a transient LLM error) must not prevent
        the page from being marked processed, and must not crash the pipeline."""
        doc_id, _ = _make_document(tmp_path, "sha_e2e_3", "Some text that will fail extraction.")
        job_id = "job_e2e_3"
        storage.create_job(id=job_id, document_id=doc_id)

        with patch("app.pipeline.ingestion.fact_extraction.extract_facts_from_text", side_effect=RuntimeError("boom")), \
             patch("app.pipeline.ingestion.embeddings.upsert"):
            ingestion.run_document_pipeline(doc_id, job_id)

        job = storage.get_job(job_id)
        assert job["status"] == "completed"  # the document still finishes even though this page's facts failed
        assert job["facts_extracted"] == 0

        issues = storage.list_extraction_issues(document_id=doc_id)
        assert any(i["issue_type"] == "fact_extraction_error" for i in issues)
