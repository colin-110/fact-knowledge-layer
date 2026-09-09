from fastapi import APIRouter, HTTPException, Response

from app import storage
from app.api.serializers import fact_row_to_out
from app.models import FactOut

router = APIRouter(tags=["facts"])


@router.get("/facts", response_model=list[FactOut])
def list_facts(document_id: str | None = None, search: str | None = None, limit: int = 200):
    rows = storage.list_facts(document_id=document_id, search=search, limit=limit)
    return [fact_row_to_out(r) for r in rows]


@router.get("/facts/{fact_id}", response_model=FactOut)
def get_fact(fact_id: str):
    row = storage.get_fact(fact_id)
    if row is None:
        raise HTTPException(404, "Fact not found")
    return fact_row_to_out(row)


@router.get("/evidence/{evidence_id}/artifact")
def get_evidence_artifact(evidence_id: str):
    row = storage.get_evidence(evidence_id)
    if row is None or not row["artifact_path"]:
        raise HTTPException(404, "No artifact for this evidence")
    from pathlib import Path

    path = Path(row["artifact_path"])
    if not path.exists():
        raise HTTPException(404, "Artifact file missing on disk")
    return Response(content=path.read_bytes(), media_type="image/png")


@router.get("/extraction-issues")
def list_extraction_issues(document_id: str | None = None, limit: int = 200):
    rows = storage.list_extraction_issues(document_id=document_id, limit=limit)
    return [dict(r) for r in rows]
