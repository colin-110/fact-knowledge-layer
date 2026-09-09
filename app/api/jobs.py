from fastapi import APIRouter, HTTPException

from app import storage
from app.models import JobStatusOut

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatusOut)
def get_job(job_id: str):
    row = storage.get_job(job_id)
    if row is None:
        raise HTTPException(404, "Job not found")
    return JobStatusOut(
        job_id=row["id"], document_id=row["document_id"], status=row["status"], stage=row["stage"],
        progress=row["progress"] or 0, total_pages=row["total_pages"] or 0,
        pages_processed=row["pages_processed"] or 0, facts_extracted=row["facts_extracted"] or 0,
        relationships_found=row["relationships_found"] or 0, error=row["error"],
    )
