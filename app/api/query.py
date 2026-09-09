from fastapi import APIRouter

from app.models import QueryRequest, QueryResponse
from app.pipeline.answer_service import answer_question

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return answer_question(req.question)
