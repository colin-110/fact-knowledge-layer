"""Tests for storage helpers that don't fit under the pipeline/retrieval test files.

list_incomplete_jobs backs the startup-resume fix: real finding was that killing the server
mid-ingestion (or the dev reloader restarting it) left a job stuck at status="processing"
forever, with nothing to notice and continue it - even though run_document_pipeline itself
already skips pages it finished before the interruption.
"""

import pytest

from app import ids, storage
from app.db import init_db


@pytest.fixture(autouse=True, scope="module")
def _ensure_db():
    init_db()


def _make_document(sha):
    doc_id = ids.document_id_from_hash(sha)
    storage.create_document(
        id=doc_id, filename="sample.pdf", sha256=sha, title="Sample",
        page_count=1, file_path=f"/tmp/{sha}.pdf",
    )
    return doc_id


class TestListIncompleteJobs:
    def test_finds_queued_and_processing_jobs(self):
        doc_id = _make_document("sha_incomplete_1")
        storage.create_job(id="job_queued_1", document_id=doc_id)
        storage.create_job(id="job_processing_1", document_id=doc_id)
        storage.update_job("job_processing_1", status="processing")

        incomplete_ids = {j["id"] for j in storage.list_incomplete_jobs()}
        assert "job_queued_1" in incomplete_ids
        assert "job_processing_1" in incomplete_ids

    def test_ignores_completed_and_failed_jobs(self):
        doc_id = _make_document("sha_incomplete_2")
        storage.create_job(id="job_completed_1", document_id=doc_id)
        storage.update_job("job_completed_1", status="completed")
        storage.create_job(id="job_failed_1", document_id=doc_id)
        storage.update_job("job_failed_1", status="failed")

        incomplete_ids = {j["id"] for j in storage.list_incomplete_jobs()}
        assert "job_completed_1" not in incomplete_ids
        assert "job_failed_1" not in incomplete_ids
