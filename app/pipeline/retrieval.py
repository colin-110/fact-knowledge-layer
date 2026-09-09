"""Hybrid retrieval: dense (Chroma) + lexical (SQLite FTS5), fused with
Reciprocal Rank Fusion, with an optional cross-encoder rerank as a final pass.

RRF fuses by *rank*, not raw score, which sidesteps the fact that cosine
similarity and BM25 scores live on incomparable scales.
"""

import re

from app import embeddings
from app.db import db_session

_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


def _fts_query_string(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9%]+", text)
    tokens = [t for t in tokens if len(t) > 1]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens[:12])


def fts_search(table: str, id_column: str, text_column: str, query_text: str, limit: int = 20) -> list[str]:
    fts_table = f"{table}_fts" if not table.endswith("_fts") else table
    q = _fts_query_string(query_text)
    with db_session() as conn:
        rows = conn.execute(
            f"SELECT {id_column} FROM {fts_table} WHERE {fts_table} MATCH ? ORDER BY bm25({fts_table}) ASC LIMIT ?",
            (q, limit),
        ).fetchall()
    return [r[0] for r in rows]


def dense_search(collection_name: str, query_text: str, limit: int = 20, where: dict | None = None) -> list[str]:
    res = embeddings.query(collection_name, query_text, n_results=limit, where=where)
    return list(res.get("ids", [[]])[0])


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def hybrid_search_facts(query_text: str, limit: int = 10) -> list[str]:
    dense_ids = dense_search("facts_index", query_text, limit=20)
    lexical_ids = fts_search("facts_fts", "fact_id", "retrieval_text", query_text, limit=20)
    fused = rrf_fuse([dense_ids, lexical_ids])
    return [fact_id for fact_id, _ in fused[:limit]]


def hybrid_search_evidence(query_text: str, limit: int = 10) -> list[str]:
    dense_ids = dense_search("evidence_index", query_text, limit=20)
    lexical_ids = fts_search("evidence_fts", "evidence_id", "text", query_text, limit=20)
    fused = rrf_fuse([dense_ids, lexical_ids])
    return [ev_id for ev_id, _ in fused[:limit]]


def rerank(query_text: str, candidates: list[tuple[str, str]], top_k: int = 8) -> list[str]:
    """candidates: list of (id, text). Returns ids sorted by cross-encoder relevance.
    Falls back to input order (already RRF-ranked) if the reranker can't be loaded."""
    if not candidates:
        return []
    try:
        model = _get_cross_encoder()
        pairs = [(query_text, text) for _, text in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip([c[0] for c in candidates], scores), key=lambda x: -x[1])
        return [cid for cid, _ in ranked[:top_k]]
    except Exception:
        return [c[0] for c in candidates[:top_k]]
