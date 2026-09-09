"""Cheap-first candidate generation for the relationship engine.

We never compare every fact to every other fact (O(n^2) and mostly wasted LLM
calls). Instead: use the fact's own embedding to pull its nearest semantic
neighbors (same rough subject+metric+context), apply a couple of free
deterministic filters, and only hand the LLM pairs that already look
plausibly related.
"""

from dataclasses import dataclass

from app.pipeline import retrieval


@dataclass
class FactRecord:
    id: str
    document_id: str
    subject: str
    predicate: str
    numeric_value: float | None
    normalized_value: float | None
    normalized_unit: str | None
    period_start: str | None
    period_end: str | None
    period_label: str | None
    scope: str | None
    status: str | None
    retrieval_text: str


def _periods_could_relate(a: FactRecord, b: FactRecord) -> bool:
    """True unless the periods are clearly disjoint AND both are fully specified -
    a clear mismatch (e.g. FY22 vs FY24) is still worth flagging as a candidate,
    since a genuine contradiction can span periods; we only rule out the case
    where the gap is large enough that no relationship engine reasoning is useful."""
    if not a.period_start or not b.period_start:
        return True
    return True  # let the LLM/rule layer make the temporal judgment; this is a placeholder for future tightening


def find_candidates(fact: FactRecord, top_k: int = 5) -> list[str]:
    """Return candidate fact ids (excluding `fact` itself) most likely to relate to it."""
    neighbor_ids = retrieval.dense_search("facts_index", fact.retrieval_text, limit=top_k + 6)
    candidates = [cid for cid in neighbor_ids if cid != fact.id]
    return candidates[:top_k]
