import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app import ids, jobs, storage
from app.config import UPLOADS_DIR
from app.models import DocumentOut, UploadResponse

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=UploadResponse, status_code=202)
async def upload_document(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    content = await file.read()
    sha256 = hashlib.sha256(content).hexdigest()

    existing = storage.get_document_by_hash(sha256)
    if existing is not None:
        document_id = existing["id"]
        latest_job = storage.get_latest_job_for_document(document_id)
        if latest_job is not None and latest_job["status"] in ("queued", "processing", "completed"):
            return UploadResponse(document_id=document_id, job_id=latest_job["id"], status=latest_job["status"])
    else:
        document_id = ids.document_id_from_hash(sha256)
        file_path = UPLOADS_DIR / f"{document_id}.pdf"
        file_path.write_bytes(content)

        page_count = _safe_page_count(file_path)
        title = Path(file.filename).stem.replace("_", " ").replace("-", " ").strip()
        storage.create_document(
            id=document_id, filename=file.filename, sha256=sha256, title=title,
            page_count=page_count, file_path=str(file_path),
        )

    job_id = f"job_{uuid.uuid4().hex[:16]}"
    storage.create_job(id=job_id, document_id=document_id)
    jobs.enqueue(document_id, job_id)

    return UploadResponse(document_id=document_id, job_id=job_id, status="queued")


def _safe_page_count(path: Path) -> int:
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:
        return 0


@router.get("/documents", response_model=list[DocumentOut])
def list_documents():
    rows = storage.list_documents()
    return [
        DocumentOut(
            id=r["id"], filename=r["filename"], title=r["title"], page_count=r["page_count"],
            created_at=r["created_at"], latest_job_status=r["latest_job_status"],
            fact_count=r["fact_count"], relationship_count=r["relationship_count"],
        )
        for r in rows
    ]


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str):
    rows = storage.list_documents()
    for r in rows:
        if r["id"] == document_id:
            return DocumentOut(
                id=r["id"], filename=r["filename"], title=r["title"], page_count=r["page_count"],
                created_at=r["created_at"], latest_job_status=r["latest_job_status"],
                fact_count=r["fact_count"], relationship_count=r["relationship_count"],
            )
    raise HTTPException(404, "Document not found")
