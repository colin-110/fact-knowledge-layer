import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

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


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str):
    """Serves the original uploaded PDF - used so evidence can link straight to
    '#page=N' in the browser's native PDF viewer, not just a static page render."""
    doc = storage.get_document(document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    path = Path(doc["file_path"])
    if not path.exists():
        raise HTTPException(404, "Source PDF file missing on disk")
    # content_disposition_type="inline" is the whole point here - FileResponse defaults to
    # "attachment" when a filename is given, which makes the browser download the PDF instead
    # of rendering it, so the side panel's <iframe> (and "#page=N" deep links) showed blank.
    return FileResponse(
        path, media_type="application/pdf", filename=doc["filename"], content_disposition_type="inline"
    )


@router.get("/documents/{document_id}/pages/{page_number}/image")
def get_document_page_image(document_id: str, page_number: int, dpi: int = 150):
    """Renders any page of the source PDF to PNG on demand - unlike chart evidence (which has
    a pre-rendered crop saved at ingestion time), plain text/table evidence has no stored
    image, so this lets any evidence card show its actual source page without needing one
    stored per page up front."""
    doc = storage.get_document(document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    path = Path(doc["file_path"])
    if not path.exists():
        raise HTTPException(404, "Source PDF file missing on disk")

    import pymupdf

    try:
        with pymupdf.open(path) as pdf:
            if page_number < 1 or page_number > pdf.page_count:
                raise HTTPException(404, f"Page {page_number} out of range (document has {pdf.page_count} pages)")
            pix = pdf[page_number - 1].get_pixmap(dpi=min(max(dpi, 72), 300))
            png_bytes = pix.tobytes("png")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Could not render page: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png")
