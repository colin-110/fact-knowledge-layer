"""Query answering: hybrid retrieval -> rerank -> context assembly -> grounded LLM answer.

The answer text comes from the LLM, but the facts/evidence/relationships lists
returned alongside it come straight from retrieval - the LLM cannot silently
cite something that wasn't actually retrieved.
"""

from pydantic import BaseModel

from app import storage
from app.llm import complete_json
from app.models import QueryEvidenceRef, QueryFactRef, QueryRelationshipRef, QueryResponse
from app.pipeline import retrieval

SYSTEM_PROMPT = """Answer the user's question using ONLY the facts and evidence listed below. If the retrieved \
material does not actually answer the question, say so plainly rather than guessing. If multiple sources give \
different figures for the same thing, mention both values and explain the relationship between them if one is \
given (corroborates / contradicts / context reconciles). Keep the answer to 2-5 sentences, and always name the \
source document when citing a number."""


class AnswerLLM(BaseModel):
    answer: str


def answer_question(question: str) -> QueryResponse:
    fact_ids = retrieval.hybrid_search_facts(question, limit=15)
    fact_rows = storage.get_facts_batch(fact_ids)
    fact_candidates = [(r["id"], r["retrieval_text"]) for r in fact_rows]
    top_fact_ids = retrieval.rerank(question, fact_candidates, top_k=6)
    id_to_fact = {r["id"]: r for r in fact_rows}
    top_facts = [id_to_fact[i] for i in top_fact_ids if i in id_to_fact]

    evidence_ids = retrieval.hybrid_search_evidence(question, limit=10)
    evidence_rows = storage.get_evidence_batch(evidence_ids)
    evidence_candidates = [(r["id"], r["text"] or "") for r in evidence_rows]
    top_evidence_ids = retrieval.rerank(question, evidence_candidates, top_k=5)
    id_to_evidence = {r["id"]: r for r in evidence_rows}
    top_evidence = [id_to_evidence[i] for i in top_evidence_ids if i in id_to_evidence]

    top_fact_id_set = {f["id"] for f in top_facts}
    all_relationships = storage.list_relationships(limit=2000)
    relevant_relationships = [
        r for r in all_relationships if r["fact_a_id"] in top_fact_id_set or r["fact_b_id"] in top_fact_id_set
    ][:5]

    context_lines = []
    for f in top_facts:
        context_lines.append(f"FACT: {f['retrieval_text']} (source: {f['document_filename']})")
    for e in top_evidence:
        snippet = (e["text"] or "")[:400]
        context_lines.append(f"EVIDENCE [{e['document_filename']} p.{e['pdf_page_number']}]: {snippet}")
    for r in relevant_relationships:
        context_lines.append(f"RELATIONSHIP: {r['relationship_type']} - {r['reason']}")

    context_text = "\n".join(context_lines) or "(no relevant facts or evidence were found in the knowledge base)"
    user_prompt = f"Question: {question}\n\nRetrieved context:\n{context_text}"

    if not top_facts and not top_evidence:
        answer_text = (
            "I couldn't find any facts or evidence in the knowledge base relevant to this question. "
            "Try uploading a document that covers it, or rephrase the question."
        )
    else:
        result = complete_json(SYSTEM_PROMPT, user_prompt, AnswerLLM)
        answer_text = result.answer

    facts_out = [
        QueryFactRef(
            fact_id=f["id"], subject=f["subject"], predicate=f["predicate"],
            value=f"{f['raw_value'] or ''} {f['raw_unit'] or ''}".strip() or None,
            period=f["period_label"], scope=f["scope"], confidence=f["extraction_confidence"],
        )
        for f in top_facts
    ]
    evidence_out = [
        QueryEvidenceRef(
            document=e["document_filename"], page=e["pdf_page_number"],
            text=(e["text"] or "")[:500], evidence_id=e["id"],
        )
        for e in top_evidence
    ]
    relationships_out = [
        QueryRelationshipRef(
            type=r["relationship_type"], fact_a=r["fact_a_id"], fact_b=r["fact_b_id"], reason=r["reason"],
        )
        for r in relevant_relationships
    ]

    return QueryResponse(answer=answer_text, facts=facts_out, evidence=evidence_out, relationships=relationships_out)
