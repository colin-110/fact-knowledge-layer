from fastapi import APIRouter, HTTPException

from app import storage
from app.api.serializers import fact_row_to_out
from app.models import RelationshipOut

router = APIRouter(tags=["relationships"])


@router.get("/relationships", response_model=list[RelationshipOut])
def list_relationships(relationship_type: str | None = None, limit: int = 200):
    rows = storage.list_relationships(relationship_type=relationship_type, limit=limit)
    out = []
    for r in rows:
        fact_a = storage.get_fact(r["fact_a_id"])
        fact_b = storage.get_fact(r["fact_b_id"])
        if fact_a is None or fact_b is None:
            continue
        out.append(
            RelationshipOut(
                id=r["id"], relationship_type=r["relationship_type"], confidence=r["confidence"],
                context_dimension=r["context_dimension"], reason=r["reason"],
                fact_a=fact_row_to_out(fact_a), fact_b=fact_row_to_out(fact_b),
            )
        )
    return out
