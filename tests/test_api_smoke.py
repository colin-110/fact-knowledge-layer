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
