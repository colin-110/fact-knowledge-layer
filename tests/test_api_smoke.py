from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_list_documents_empty_on_fresh_db():
    with TestClient(app) as client:
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert resp.json() == []


def test_job_not_found():
    with TestClient(app) as client:
        resp = client.get("/jobs/does-not-exist")
        assert resp.status_code == 404


def test_list_facts_empty_on_fresh_db():
    with TestClient(app) as client:
        resp = client.get("/facts")
        assert resp.status_code == 200
        assert resp.json() == []


def test_list_relationships_empty_on_fresh_db():
    with TestClient(app) as client:
        resp = client.get("/relationships")
        assert resp.status_code == 200
        assert resp.json() == []


def test_upload_rejects_non_pdf():
    with TestClient(app) as client:
        resp = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert resp.status_code == 400


def test_upload_rejects_empty_filename():
    # FastAPI/Starlette reject a part with no filename during multipart parsing itself
    # (422) before our own filename check (400) ever runs - either way, it must not 202.
    with TestClient(app) as client:
        resp = client.post("/documents", files={"file": ("", b"hello", "application/pdf")})
        assert resp.status_code in (400, 422)


def test_reuploading_identical_content_returns_same_document_id():
    """sha256-based idempotency: the same bytes (even under a different filename) must map to
    the same document_id rather than creating a duplicate document row."""
    content = b"%PDF-1.4 not a real pdf but consistent bytes for hashing"
    with TestClient(app) as client:
        first = client.post("/documents", files={"file": ("a.pdf", content, "application/pdf")})
        second = client.post("/documents", files={"file": ("b.pdf", content, "application/pdf")})
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["document_id"] == second.json()["document_id"]


def test_malformed_pdf_content_fails_the_job_instead_of_hanging():
    """A file with a .pdf extension but garbage content should be accepted (202 - we don't
    parse synchronously) and then fail cleanly in the background job, not crash the server
    or leave the job stuck at 'queued' forever."""
    import time

    with TestClient(app) as client:
        resp = client.post(
            "/documents", files={"file": ("garbage.pdf", b"this is not a pdf file at all", "application/pdf")}
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        job = None
        for _ in range(50):
            job = client.get(f"/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert job is not None
        assert job["status"] == "failed"
        assert job["error"]


def test_query_on_empty_knowledge_base_does_not_crash_or_call_the_llm():
    """With no facts/evidence in the DB, hybrid retrieval returns nothing, and the answer
    service must short-circuit to a canned response rather than attempting an LLM call with
    empty context (which would otherwise be the only network-dependent path in the test suite)."""
    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "What was the revenue in FY24?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["facts"] == []
    assert body["evidence"] == []
    assert "couldn't find" in body["answer"].lower() or "could not find" in body["answer"].lower()


def test_query_rejects_missing_question_field():
    with TestClient(app) as client:
        resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_facts_search_with_no_matches_returns_empty_list_not_error():
    with TestClient(app) as client:
        resp = client.get("/facts", params={"search": "something that will never match anything"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_nonexistent_fact_returns_404():
    with TestClient(app) as client:
        resp = client.get("/facts/fa_does_not_exist")
        assert resp.status_code == 404


def test_get_nonexistent_evidence_artifact_returns_404():
    with TestClient(app) as client:
        resp = client.get("/evidence/ev_does_not_exist/artifact")
        assert resp.status_code == 404


def test_get_nonexistent_document_returns_404():
    with TestClient(app) as client:
        resp = client.get("/documents/doc_does_not_exist")
        assert resp.status_code == 404
